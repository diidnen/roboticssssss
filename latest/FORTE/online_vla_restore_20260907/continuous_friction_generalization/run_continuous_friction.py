"""Execute exactly the frozen 48-context/144-valid-branch campaign."""
from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from common import read, write, sha
from launch import launch
from audit_continuous_rollout import audit
from audit_geometric_label import audit as geometry_audit


def append_jsonl(path: Path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(out: Path, port: int):
    out = Path(out)
    manifest = read(out / "CONTINUOUS_FRICTION_RUNTIME_MANIFEST.json")
    for path, digest in manifest["source_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError("Frozen runtime source changed: " + path)
    for path, digest in manifest["frozen_artifact_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError("Frozen plan/audit changed: " + path)
    with (out / "CONTINUOUS_FRICTION_EXECUTION_PLAN.csv").open() as handle:
        queue = list(csv.DictReader(handle))
    contexts = read(out / "DEV_PLAN.json")["contexts"]
    by_id = {item["id"]: (index, item) for index, item in enumerate(contexts)}
    if len(queue) != 144 or len(contexts) != 48:
        raise RuntimeError("Frozen context/branch count mismatch")
    if set(item["method"] for item in queue) != {"ACTIVEFORCING", "GT_PHYSICS", "FIXED_4"}:
        raise RuntimeError("Frozen method set mismatch")
    for context_id in by_id:
        methods = [x for x in queue if x["context_id"] == context_id]
        if len(methods) != 3 or set(x["method"] for x in methods) != {"ACTIVEFORCING", "GT_PHYSICS", "FIXED_4"}:
            raise RuntimeError("Matched context method mismatch")
        if sorted(int(x["method_order"]) for x in methods) != [1, 2, 3]:
            raise RuntimeError("Within-context randomized method order missing")
    if not (out / "CONTINUOUS_FRICTION_QUEUE_STARTED.json").exists():
        write(out / "CONTINUOUS_FRICTION_QUEUE_STARTED.json", {
            "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
            "planned_valid_full_task_branches": len(queue), "physics_started_before_plan_freeze": False,
            "policy_server_port": port,
        })
    admitted_rows = []
    initialized = set()
    for position, item in enumerate(queue, 1):
        context_id = item["context_id"]
        index, plan = by_id[context_id]
        free = shutil.disk_usage(out).free
        if free < 2 * (1 << 30):
            raise RuntimeError("ENOSPC safety stop before next physics branch")
        if context_id not in initialized:
            reference = launch(out, index, "REFERENCE", port)
            initialized.add(context_id)
            if not read(reference / "WORKER_COMPLETION.json").get("logical_success"):
                raise RuntimeError("Invalid reference attempt")
        method = item["method"]
        job = launch(out, index, method, port)
        admission_path = job / "FINAL_EVIDENCE_ADMISSION.json"
        if admission_path.exists():
            admission = read(admission_path)
            if admission.get("admitted") is not True:
                raise RuntimeError("Existing branch admission is not valid")
        else:
            provenance = audit(job)
            geometry = geometry_audit(job)
            admission = {"online_provenance": provenance, "geometry": geometry,
                         "admitted": bool(provenance["passed"] and geometry["passed"])}
            write(admission_path, admission)
            if not admission["admitted"]:
                raise RuntimeError(f"Branch quarantined: {job}: {provenance['errors']}")
        admitted_rows.append({
            "execution_index": int(item["execution_index"]), "context_id": context_id,
            "method": method, "job": str(job), "root": int(item["root"]),
            "task": int(item["task"]), "mu_test": float(item["mu_test"]),
        })
        append_jsonl(out / "CONTINUOUS_FRICTION_PROGRESS.jsonl", {
            "completed_valid_branches": len(admitted_rows), "planned_valid_branches": 144,
            "execution_index": int(item["execution_index"]), "context_id": context_id,
            "method": method, "job": str(job), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "free_bytes": shutil.disk_usage(out).free,
        })
        print("CONTINUOUS_FRICTION_BRANCH_DONE", len(admitted_rows), 144, context_id, method, flush=True)
    if len(admitted_rows) != 144 or len({(x["context_id"], x["method"]) for x in admitted_rows}) != 144:
        raise RuntimeError("Final admitted branch set is not exactly 144")
    if not (out / "CONTINUOUS_FRICTION_EXECUTION_COMPLETE.json").exists():
        write(out / "CONTINUOUS_FRICTION_EXECUTION_COMPLETE.json", {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "contexts": len(initialized), "valid_full_task_branches": len(admitted_rows),
            "rows": admitted_rows, "methods": ["ACTIVEFORCING", "GT_PHYSICS", "FIXED_4"],
            "scripted_or_replay_fallbacks": 0, "new_training_steps": 0,
            "physics_stopped": True, "stop_condition_reached": True,
        })
    print("CONTINUOUS_FRICTION_EXECUTION_COMPLETE 48 144 STOP_PHYSICS", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18885)
    args = parser.parse_args()
    run(args.out, args.port)
