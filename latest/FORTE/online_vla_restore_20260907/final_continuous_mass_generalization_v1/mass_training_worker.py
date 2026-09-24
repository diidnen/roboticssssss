"""Current-contract MASS reference/controlled-branch worker.

The worker collects one P4-B reference or one matched full-task force branch.
It performs no model fitting and no outcome-dependent selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/exouser/FORTE")
BASE = ROOT / "analysis/results"
CORE = HERE
SENSORS = BASE / "current_runtime_sensor_repair_v3_candidate_20260905"
BRANCH = BASE / "current_detached_no_chase_candidate_v1_20260906"
SHARED = BASE / "current_runtime_recovery_v2_20260905"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def check(directory):
    directory = Path(directory)
    protocol = read(directory / "MASS_TRAINING_PROTOCOL.json")
    manifest_path = directory / "MASS_TRAINING_RUNTIME_MANIFEST.json"
    if sha(manifest_path) != protocol["runtime_manifest_sha256"]:
        raise ValueError("Runtime protocol mismatch")
    manifest = read(manifest_path)
    if manifest["version"] != "CURRENT_MASS_MATCHED_TRAINING_RUNTIME_V1":
        raise ValueError("Wrong MASS training runtime")
    if manifest["historical_labels_used"] or manifest["postaction_belief_features_allowed"]:
        raise ValueError("Forbidden supervision contract")
    for path, digest in manifest["source_hashes"].items():
        if sha(path) != digest:
            raise ValueError("Frozen source changed: " + path)
    if sha(directory / "MASS_TRAINING_CONTEXT_PLAN.json") != manifest["context_plan_sha256"]:
        raise ValueError("MASS context plan changed")
    return manifest, protocol


def run(directory, context_id, force):
    directory = Path(directory)
    manifest, protocol = check(directory)
    plan = next(item for item in read(directory / "MASS_TRAINING_CONTEXT_PLAN.json")["contexts"] if item["id"] == context_id)
    if force is not None and force not in manifest["candidate_forces_N"]:
        raise ValueError("Unplanned force")
    job = directory / "references" / context_id if force is None else directory / "branches" / context_id / f"F{force:g}"
    claim = read(job / "LAUNCH_CLAIM.json")
    if claim != {"context_id": context_id, "force": force, "runtime_manifest_sha256": protocol["runtime_manifest_sha256"]}:
        raise ValueError("Missing or incorrect exclusive launch claim")
    receipt = {"runtime_manifest_sha256": protocol["runtime_manifest_sha256"], "source_hashes": manifest["source_hashes"]}
    write(job / "SOURCE_HASHES_BEFORE.json", receipt)
    for path in (ROOT, SHARED, CORE, SENSORS, BRANCH):
        sys.path.insert(0, str(path))
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    try:
        import torch
        import mass_runtime_core as core
        import measurement_hooks

        spec = importlib.util.spec_from_file_location("mass_training_explicit_branch", BRANCH / "branch_execution.py")
        branch = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = branch
        spec.loader.exec_module(branch)
        if Path(branch.__file__).resolve() != (BRANCH / "branch_execution.py").resolve():
            raise ValueError("Wrong branch adapter")

        def decision(env, p4, p5, live_plan, live_job):
            live_job = Path(live_job)
            if force is None:
                saved = torch.load(live_job / "DECISION_STATE.pt", map_location="cpu", weights_only=False)
                with (live_job / "RAW_PROBE.csv").open() as stream:
                    raw = list(csv.DictReader(stream))
                records = read(live_job / "CONTACT_PATCH_READBACK.json")
                x = branch.preaction_features(
                    raw, saved, np.asarray(records[-1]["eef_pose"], np.float32).reshape(-1), p4, plan["task"]
                )
                branch.npwrite(live_job / "PREACTION_SEQUENCE.npy", x)
                return {"reference_saved": True, "candidate_actions_executed": 0}
            reference = directory / "references" / context_id
            result = branch.run_branch(
                env, p4, p5, live_plan, live_job, reference, force, 0,
                drop_contract_version=manifest["label_version"],
            )
            np.testing.assert_array_equal(np.load(reference / "PREACTION_SEQUENCE.npy"), np.load(live_job / "PREACTION_SEQUENCE.npy"))
            return result

        code = core.run_probe(
            plan, job, measurement_hooks=measurement_hooks, on_decision=decision,
            runtime_manifest_sha256=protocol["runtime_manifest_sha256"],
        )
        check(directory)
        result = read(job / "RESULT.json") if (job / "RESULT.json").exists() else None
        valid = bool(code == 0 and result and result["probe_qualified"] and (job / "MASS_INTERVENTION_READBACK.json").exists())
        if force is not None:
            branch_result = read(job / "BRANCH_RESULT.json") if (job / "BRANCH_RESULT.json").exists() else None
            valid = bool(valid and branch_result and branch_result["passed"] and branch_result["outcome"]["label_valid"])
        write(job / "SOURCE_HASHES_AFTER.json", receipt)
        status = 0 if valid else 2
        write(job / "WORKER_COMPLETION.json", {"requested_exit_code": status, "logical_success": valid, "written_before_app_shutdown": True})
        return status
    finally:
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--force", type=float)
    args = parser.parse_args()
    raise SystemExit(run(args.directory, args.context, args.force))
