"""Current-architecture online-VLA worker for MASS development/final plans."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/exouser/FORTE")
BASE = ROOT / "analysis/results"
FRICTION_SOURCE = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def clean(value):
    if hasattr(value, "detach"): return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(clean(value), stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def make_sequence(raw, saved, task, chunk):
    last = raw[-1]; state = np.zeros(13, np.float32); mask = np.zeros(13, np.float32); mask[:3] = 1
    left = np.asarray([float(last["left_f" + axis]) for axis in "xyz"])
    right = np.asarray([float(last["right_f" + axis]) for axis in "xyz"])
    state[6:10] = [abs(left[2]), abs(right[2]), np.linalg.norm(left[:2]), np.linalg.norm(right[:2])]; mask[6:10] = 1
    joints = np.asarray(clean(saved["state"]["articulation"]["robot"]["joint_position"])).reshape(-1)
    if len(joints) != 9 or not np.isclose(joints[-2], float(last["gripper_opening"]), rtol=0, atol=1e-6):
        raise RuntimeError("probe/snapshot joint mismatch")
    state[11:13] = [joints[-2], -joints[-1]]; mask[11:13] = 1
    velocity = np.asarray(clean(saved["state"]["rigid_object"][last["object_id"]]["root_velocity"])).reshape(-1)[:3]
    state[3:6] = velocity; state[10] = np.linalg.norm(velocity[:2]); mask[3:6] = 1; mask[10] = 1
    x = np.zeros((8, 64), np.float32); x[:, 6 + (0, 1, 5, 6).index(task)] = 1
    x[:, 12:25] = state; x[:, 25:38] = mask; x[:, 38:51] = state; x[:, 51:64] = mask
    commands = np.asarray(chunk[:8, :3], np.float64)
    x[:, :3] = commands - commands[0]; x[:, 3:6] = np.vstack([np.zeros((1, 3)), np.diff(commands, axis=0)])
    return x


def posterior_payload(pred):
    distribution = pred["continuous_posterior"]
    quantiles = {str(q): distribution.ppf(q) for q in (0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975)}
    logs = [float(x) for x in pred["member_log_sigmas"]]
    return {"interface": pred["interface"], "feature_schema_id": pred["feature_schema_id"],
            "member_means": [float(x) for x in pred["member_means"]], "member_log_sigmas": logs,
            "member_sigmas": [math.exp(x) for x in logs], "posterior_moments": pred["posterior_moments"],
            "posterior_quantiles": quantiles, "interval_68": [quantiles["0.16"], quantiles["0.84"]],
            "interval_90": [quantiles["0.05"], quantiles["0.95"]],
            "interval_95": [quantiles["0.025"], quantiles["0.975"]],
            "integration_qa": pred["integration_qa"], "integration_nodes": pred["integration_nodes"].tolist(),
            "integration_weights": pred["integration_weights"].tolist(), "candidate_actions_executed": 0,
            "hidden_mass_used": False, "derived_summary_only": True,
            "posterior_or_quadrature_modified": False}


def setup_paths():
    for path in (FRICTION_SOURCE, HERE, ROOT, BASE / "current_runtime_recovery_v2_20260905",
                 BASE / "current_runtime_sensor_repair_v3_candidate_20260905",
                 BASE / "current_runtime_branch_execution_v6_20260905"):
        sys.path.insert(0, str(path))


def run(args):
    out, job = Path(args.out), Path(args.job)
    plan = read(Path(args.plan))["contexts"][args.context]
    manifest = read(Path(args.runtime_manifest))
    for path, digest in manifest["source_hashes"].items():
        if sha(path) != digest: raise RuntimeError("frozen source changed: " + path)
    for path, digest in manifest["artifact_hashes"].items():
        if sha(path) != digest: raise RuntimeError("frozen artifact changed: " + path)
    write(job / "SOURCE_HASHES_BEFORE.json", manifest["source_hashes"])
    setup_paths()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    try:
        import torch
        import mass_runtime_core as core
        import measurement_hooks
        import mass_belief
        from mass_online_variants import MassMethod, VALID_METHODS
        from activeforcing_execution_snapshot import capture
        from activeforcing_command_handoff import command_from_probe
        from branch_execution import identical, SNAPSHOT_GROUPS
        import runtime

        if args.method not in VALID_METHODS and args.method != "REFERENCE": raise RuntimeError("method outside frozen plan")
        belief = mass_belief.ContinuousMassBelief(HERE / "MASS_BELIEF_MANIFEST.json",
            manifest_sha256=sha(HERE / "MASS_BELIEF_MANIFEST.json"))
        feasibility = (MassMethod(args.method, plan["mass_kg"] if args.method == "GT_MASS" else None)
                       if args.method != "REFERENCE" else None)
        reference_dir = out / "references" / plan["id"]

        def decision(env, p4, p5, live_plan, live_job):
            with (job / "RAW_PROBE.csv").open() as stream: raw = list(csv.DictReader(stream))
            readback = read(job / "CONTACT_PATCH_READBACK.json")
            saved = torch.load(job / "DECISION_STATE.pt", map_location="cpu", weights_only=False)
            live = capture(env)
            equality = {key: identical(saved[key], live[key]) for key in SNAPSHOT_GROUPS}
            if args.method != "REFERENCE":
                reference = torch.load(reference_dir / "DECISION_STATE.pt", map_location="cpu", weights_only=False)
                equality.update({"reference_" + key: identical(saved[key], reference[key]) for key in SNAPSHOT_GROUPS})
                with (reference_dir / "RAW_PROBE.csv").open() as stream: equality["probe_rows"] = raw == list(csv.DictReader(stream))
                equality["probe_records"] = identical(readback, read(reference_dir / "CONTACT_PATCH_READBACK.json"))
            write(job / "POSTPROBE_EQUALITY.json", {"passed": all(equality.values()), "checks": equality,
                                                       "candidate_actions_executed": 0})
            if not all(equality.values()): raise RuntimeError("common MASS post-query state failed")
            py_state, np_state = random.getstate(), np.random.get_state()
            torch_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            pred = belief.rows(raw, readback, query_runtime_manifest_sha256=sha(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"))
            posterior = posterior_payload(pred); write(job / "PREACTION_POSTERIOR.json", posterior)
            random.setstate(py_state); np.random.set_state(np_state); torch.set_rng_state(torch_state); torch.cuda.set_rng_state_all(cuda_state)
            if args.method == "REFERENCE":
                runtime.save_reference_observation(env, plan, job)
                return {"reference_saved": True, "candidate_actions_executed": 0}
            if posterior != read(reference_dir / "PREACTION_POSTERIOR.json"): raise RuntimeError("MASS posterior parity failed")
            prepared = runtime.first_chunk(env, plan, job, args.port, reference_dir); chunk = prepared[3]
            x = make_sequence(raw, saved, plan["task"], chunk)
            with (job / "PREACTION_SEQUENCE.npy").open("xb") as stream: np.save(stream, x)
            shared = out / "initial_chunk_identity"; shared.mkdir(exist_ok=True)
            canonical = shared / (plan["id"] + ".json")
            identity = {"payload_sha256": prepared[4]["observation_sha256"], "noise_sha256": prepared[4]["noise_sha256"],
                        "online_chunk_sha256": prepared[4]["action_sha256"],
                        "preaction_sequence_sha256": sha(job / "PREACTION_SEQUENCE.npy")}
            if canonical.exists() and read(canonical) != identity: raise RuntimeError("methods do not share initial online chunk/state")
            if not canonical.exists(): write(canonical, identity)
            write(job / "INITIAL_ONLINE_CHUNK_IDENTITY.json", identity)
            selected = feasibility.select(x, posterior); force = float(selected["executed_force_N"])
            selected.update(feasibility_sequence_source="ONLINE_VLA_ACTION_CHUNK", request_id=prepared[4]["request_id"],
                            phase_channels="NONE", scripted_prefix_used=False,
                            transfer_scope=plan["split"])
            write(job / "PLANNER_DECISION.json", selected)
            handoff = command_from_probe(readback[-1]["action"], float(saved["objects"]["arm_action"]["_gripper_abs_cmd"][0, 0]),
                                         p4.D_CLOSED, p4.D_OPEN)
            write(job / "HANDOFF.json", {"command": handoff, "source": "last physical query action", "same_query": True})
            return runtime.rollout(env, p4, p5, plan, job, force, handoff, prepared)

        rc = core.run_probe(plan, job, measurement_hooks=measurement_hooks, on_decision=decision,
                            runtime_manifest_sha256=sha(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"))
        result = read(job / "RESULT.json") if (job / "RESULT.json").exists() else {}
        valid = rc == 0 and result.get("probe_qualified", False)
        if args.method != "REFERENCE":
            branch = read(job / "BRANCH_RESULT.json") if (job / "BRANCH_RESULT.json").exists() else {}
            valid = valid and branch.get("online_vla_verified", False) and branch.get("outcome", {}).get("label_valid", False)
        write(job / "WORKER_COMPLETION.json", {"logical_success": bool(valid), "method": args.method})
        write(job / "SOURCE_HASHES_AFTER.json", {path: sha(path) for path in manifest["source_hashes"]})
        return 0 if valid else 2
    finally:
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True); parser.add_argument("--job", required=True)
    parser.add_argument("--plan", required=True); parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--context", type=int, required=True); parser.add_argument("--method", required=True)
    parser.add_argument("--port", type=int, default=18885)
    raise SystemExit(run(parser.parse_args()))
