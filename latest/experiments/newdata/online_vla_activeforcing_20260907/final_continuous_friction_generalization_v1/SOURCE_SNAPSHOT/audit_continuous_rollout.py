"""Independent provenance and intervention audit for each valid branch."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import read, sha, array_sha, payload_sha, INSTRUCTIONS, TABERO


def experiment_root(job: Path) -> Path:
    for parent in (job, *job.parents):
        if (parent / "CONTINUOUS_FRICTION_RUNTIME_MANIFEST.json").exists():
            return parent
    raise RuntimeError("Cannot locate frozen experiment root")


def audit(job):
    job = Path(job)
    out = experiment_root(job)
    result = read(job / "BRANCH_RESULT.json")
    trace = read(job / "BRANCH_TRACE.json")
    metadata = read(job / "VLA_SERVER_METADATA.json")
    decision = read(job / "PLANNER_DECISION.json")
    intervention = read(job / "FRICTION_INTERVENTION_READBACK.json")
    method = decision["method"]
    errors, chunks, used = [], {}, set()
    records = {}
    server_log = Path(metadata["server_log_dir"]) / "INFERENCE.jsonl"
    for line in server_log.read_text().splitlines():
        row = json.loads(line)
        if row["request_id"] in records:
            errors.append("duplicate server request id")
        records[row["request_id"]] = row
    port = int(read(job / "PROCESS.json")["command"][-1])
    expected_metadata = read(out / f"POLICY_SERVER_{port}" / "SERVER_READY.json")
    if metadata != expected_metadata:
        errors.append("wrong policy server instance")
    adapter = read(job / "VLA_OBSERVATION_ADAPTER.json")
    if adapter["source_sha256"] != sha(TABERO / "analysis/p6g1_primitive_ik_vla_grasp_realization.py"):
        errors.append("observation adapter changed")
    proc = read(job / "PROCESS_EXIT.json")
    if proc["exit_code"] != 0 or not (proc.get("completion") or {}).get("logical_success"):
        errors.append("worker incomplete")
    expected_force = float(decision["executed_force_N"])
    if float(result.get("force")) != expected_force:
        errors.append("result/decision force mismatch")
    mu = float(result["plan"]["mu"])
    mats = intervention["object_material_properties_static_dynamic_restitution"]
    if not mats or any(abs(float(x[0]) - mu) > 1e-6 or abs(float(x[1]) - mu) > 1e-6 for x in mats):
        errors.append("object friction readback mismatch")
    if intervention.get("finger_gelpad_case_material_modified") is not False:
        errors.append("non-object material modification reported")
    if intervention.get("material_combine_mode_modified") is not False:
        errors.append("combine mode modification reported")
    for step, row in enumerate(trace, 1):
        request_step = 1 + ((step - 1) // 10) * 10
        chunk_index = (step - 1) % 10
        if request_step not in chunks:
            receipt_path = job / "RPC" / f"{request_step:04d}.json"
            receipt = read(receipt_path)
            data = np.load(receipt_path.with_suffix(".npz"))
            server = records.get(receipt["request_id"])
            if not server or any(receipt.get(key) != value for key, value in server.items()):
                errors.append("missing/mismatched server receipt")
            if receipt.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
                errors.append("checkpoint mismatch")
            if not receipt.get("model_inference_called") or receipt.get("downstream_action_source") != "ONLINE_VLA":
                errors.append("inference not online")
            if receipt.get("instruction") != INSTRUCTIONS[result["plan"]["task"]]:
                errors.append("instruction mismatch")
            if sha(receipt_path.with_suffix(".npz")) != receipt.get("artifact_sha256"):
                errors.append("RPC artifact mismatch")
            raw_model = data["raw_vla_action"]
            post = data["postprocessed_vla_action"]
            if array_sha(raw_model) != receipt.get("raw_model_action_sha256") or array_sha(post) != receipt.get("action_sha256"):
                errors.append("action hash mismatch")
            payload = {key.removeprefix("payload__"): data[key] for key in data.files if key.startswith("payload__")}
            payload["prompt"] = receipt["instruction"]
            if payload_sha(payload) != receipt.get("observation_sha256"):
                errors.append("observation hash mismatch")
            chunks[request_step] = (receipt, post)
        receipt, post = chunks[request_step]
        used.add(receipt["request_id"])
        policy = post[chunk_index].astype(np.float32)
        action = np.asarray(row["action"], np.float32)
        if not np.array_equal(action[:6], policy[:6]):
            errors.append("executed arm is not current VLA output")
        if not np.array_equal(np.asarray(row["raw_vla_arm_command"], np.float32), policy[:6]):
            errors.append("raw arm telemetry mismatch")
        if np.float32(row["raw_vla_gripper_command"]) != policy[6]:
            errors.append("raw gripper telemetry mismatch")
        if not np.array_equal(np.asarray(row["raw_vla_force_command"], np.float32), policy[7:13]):
            errors.append("raw force telemetry mismatch")
        opened = bool(row["vla_release_intent"])
        expected_slots = np.array([0, 0, 0 if opened else expected_force / 2,
                                   0, 0, 0 if opened else expected_force / 2], np.float32)
        if not np.array_equal(action[7:13], expected_slots):
            errors.append("force override mismatch")
        if float(row["selected_force_setpoint"]) != expected_force:
            errors.append("selected force trace mismatch")
        if sha(row["raw_observation_path"]) != row["raw_observation_sha256"]:
            errors.append("per-step observation mismatch")
    posterior = read(job / "PREACTION_POSTERIOR.json")
    if posterior.get("hidden_friction_used") is not False or posterior.get("posterior_or_quadrature_modified") is not False:
        errors.append("physical belief receipt violates frozen contract")
    if method == "GT_PHYSICS":
        if decision.get("used_integration_nodes") != [mu] or decision.get("used_integration_weights") != [1.0]:
            errors.append("GT delta belief mismatch")
        if not decision.get("gt_physics_used"):
            errors.append("GT receipt missing")
    elif decision.get("gt_physics_used"):
        errors.append("non-GT method used privileged friction")
    if method == "FIXED_4" and expected_force != 4.0:
        errors.append("Fixed-4 did not execute 4 N")
    if method == "ACTIVEFORCING" and expected_force != float(decision["model_selected_force_N"]):
        errors.append("AF did not execute frozen planner selection")
    if not read(job / "POSTPROBE_EQUALITY.json")["passed"]:
        errors.append("postprobe equality failed")
    if read(job / "SOURCE_HASHES_BEFORE.json") != read(job / "SOURCE_HASHES_AFTER.json"):
        errors.append("source changed during rollout")
    expected_calls = (len(trace) + 9) // 10
    if result.get("rpc_count") != expected_calls or len(used) != expected_calls:
        errors.append("wrong online inference count")
    outcome = result.get("outcome", {})
    if outcome.get("label_valid") is not True:
        errors.append("invalid full-task outcome")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "method": method,
        "steps": len(trace),
        "real_online_requests": len(used),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "DOWNSTREAM_ACTION_SOURCE": "ONLINE_VLA" if not errors else "QUARANTINED",
        "VLA_CHECKPOINT_LOADED": metadata.get("checkpoint_loaded") is True,
        "ONLINE_POLICY_INFERENCE": bool(used) and not errors,
        "VLA_ACTION_PROVENANCE_VERIFIED": not errors,
        "object_material_readback_verified": "object friction readback mismatch" not in errors,
        "outcome": outcome,
    }
