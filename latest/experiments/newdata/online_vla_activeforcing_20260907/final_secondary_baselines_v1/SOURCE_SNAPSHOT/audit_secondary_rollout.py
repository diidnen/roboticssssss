"""Independent provenance/action audit for frozen secondary variants."""
import json
from pathlib import Path
import numpy as np

from common import read, sha, array_sha, payload_sha, INSTRUCTIONS, TABERO


def audit(job):
    job = Path(job)
    result = read(job / "BRANCH_RESULT.json")
    trace = read(job / "BRANCH_TRACE.json")
    meta = read(job / "VLA_SERVER_METADATA.json")
    decision = read(job / "PLANNER_DECISION.json")
    method = decision["method"]
    errors, chunks, used = [], {}, set()
    records = {}
    for line in (Path(meta["server_log_dir"]) / "INFERENCE.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["request_id"] in records:
            errors.append("duplicate server request id")
        records[row["request_id"]] = row
    expected_meta = read(job.parent.parent / "POLICY_SERVER_18885" / "SERVER_READY.json")
    if meta != expected_meta:
        errors.append("wrong policy server instance")
    adapter = read(job / "VLA_OBSERVATION_ADAPTER.json")
    if adapter["source_sha256"] != sha(TABERO / "analysis/p6g1_primitive_ik_vla_grasp_realization.py"):
        errors.append("observation adapter changed")
    proc = read(job / "PROCESS_EXIT.json")
    if proc["exit_code"] != 0 or not (proc.get("completion") or {}).get("logical_success"):
        errors.append("worker incomplete")
    expected_force = decision.get("executed_force_N")
    if result.get("force") != expected_force:
        errors.append("result/decision force mismatch")
    for n, row in enumerate(trace, 1):
        request_step = 1 + ((n - 1) // 10) * 10
        index = (n - 1) % 10
        if request_step not in chunks:
            p = job / "RPC" / f"{request_step:04d}.json"
            receipt = read(p)
            data = np.load(p.with_suffix(".npz"))
            server = records.get(receipt["request_id"])
            if not server or any(receipt.get(k) != v for k, v in server.items()):
                errors.append("missing/mismatched server receipt")
            if receipt.get("checkpoint_sha256") != meta.get("checkpoint_sha256"):
                errors.append("checkpoint mismatch")
            if not receipt.get("model_inference_called") or receipt.get("downstream_action_source") != "ONLINE_VLA":
                errors.append("inference not online")
            if receipt.get("instruction") != INSTRUCTIONS[result["plan"]["task"]]:
                errors.append("instruction mismatch")
            if sha(p.with_suffix(".npz")) != receipt.get("artifact_sha256"):
                errors.append("RPC artifact mismatch")
            raw_model, post = data["raw_vla_action"], data["postprocessed_vla_action"]
            if array_sha(raw_model) != receipt.get("raw_model_action_sha256") or array_sha(post) != receipt.get("action_sha256"):
                errors.append("action hash mismatch")
            payload = {k.removeprefix("payload__"): data[k] for k in data.files if k.startswith("payload__")}
            payload["prompt"] = receipt["instruction"]
            if payload_sha(payload) != receipt.get("observation_sha256"):
                errors.append("observation hash mismatch")
            chunks[request_step] = (receipt, post)
        receipt, post = chunks[request_step]
        used.add(receipt["request_id"])
        policy = post[index].astype(np.float32)
        action = np.asarray(row["action"], np.float32)
        if not np.array_equal(action[:6], policy[:6]):
            errors.append("arm is not current VLA output")
        if not np.array_equal(np.asarray(row["raw_vla_arm_command"], np.float32), policy[:6]):
            errors.append("raw arm telemetry mismatch")
        if np.float32(row["raw_vla_gripper_command"]) != policy[6]:
            errors.append("raw gripper telemetry mismatch")
        if not np.array_equal(np.asarray(row["raw_vla_force_command"], np.float32), policy[7:13]):
            errors.append("raw force telemetry mismatch")
        opened = bool(row["vla_release_intent"])
        if method == "TABERO_NEUTRAL":
            expected = policy.copy()
            if opened:
                expected[6] = np.float32(0.04)
                expected[7:13] = 0.0
            if not np.array_equal(action, expected):
                errors.append("Tabero native-grasp/shared-release mapping mismatch")
            if row["selected_force_setpoint"] is not None:
                errors.append("Tabero falsely reports scalar setpoint")
        else:
            force = float(expected_force)
            expected_slots = np.array([0, 0, 0 if opened else force/2, 0, 0, 0 if opened else force/2], np.float32)
            if not np.array_equal(action[7:13], expected_slots):
                errors.append("AF force override mismatch")
            if row["selected_force_setpoint"] != force:
                errors.append("AF setpoint mismatch")
        if sha(row["raw_observation_path"]) != row["raw_observation_sha256"]:
            errors.append("per-step observation mismatch")
    posterior = read(job / "PREACTION_POSTERIOR.json")
    if posterior.get("hidden_friction_used") is not False:
        errors.append("physical belief receipt exposed hidden friction")
    if method == "GT_PHYSICS_DIRECT":
        mu = float(result["plan"]["mu"])
        if decision.get("used_integration_nodes") != [mu] or decision.get("used_integration_weights") != [1.0]:
            errors.append("GT delta belief mismatch")
        if not decision.get("gt_physics_used"):
            errors.append("GT receipt missing")
    elif decision.get("gt_physics_used"):
        errors.append("non-GT method used privileged friction")
    if not read(job / "POSTPROBE_EQUALITY.json")["passed"]:
        errors.append("postprobe equality failed")
    if read(job / "SOURCE_HASHES_BEFORE.json") != read(job / "SOURCE_HASHES_AFTER.json"):
        errors.append("source changed during rollout")
    expected_calls = (len(trace) + 9) // 10
    if result.get("rpc_count") != expected_calls or len(used) != expected_calls:
        errors.append("wrong online inference count")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "method": method,
        "steps": len(trace),
        "real_online_requests": len(used),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "downstream_action_source": "ONLINE_VLA" if not errors else "QUARANTINED",
        "outcome": result.get("outcome"),
    }
