"""Execute the prospectively frozen secondary-only plan with one worker."""
import csv
import os
from pathlib import Path

from common import read, write, sha
from launch import launch
from audit_secondary_rollout import audit
from audit_geometric_label import audit as geometry_audit


def run(out):
    out = Path(out)
    manifest = read(out / "CANDIDATE_RUNTIME_MANIFEST.json")
    for path, digest in manifest["source_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError("Frozen secondary source changed: " + path)
    with (out / "FINAL_SECONDARY_EXECUTION_PLAN.csv").open() as handle:
        queue = list(csv.DictReader(handle))
    contexts = read(out / "DEV_PLAN.json")["contexts"]
    by_id = {x["id"]: (i, x) for i, x in enumerate(contexts)}
    if len(queue) != 72 or set(x["variant"] for x in queue) != {"ACTIVEFORCING", "GT_PHYSICS_DIRECT", "TABERO_NEUTRAL"}:
        raise RuntimeError("Frozen 72-branch plan mismatch")
    write(out / "SECONDARY_QUEUE_STARTED.json", {"pid": os.getpid(), "planned_branches": len(queue), "physics_started_before_plan_freeze": False})
    initialized = set()
    rows = []
    for item in queue:
        context_id = item["context_id"]
        index, plan = by_id[context_id]
        if context_id not in initialized:
            launch(out, index, "REFERENCE")
            initialized.add(context_id)
        method = item["variant"]
        launch(out, index, method)
        job = out / "branches" / f"{context_id}__{method}"
        proof = audit(job)
        geometry = geometry_audit(job)
        if not proof["passed"] or not geometry["passed"]:
            raise RuntimeError(f"Secondary branch quarantine: {job}: {proof['errors']}")
        admission = {"online_provenance": proof, "geometry": geometry, "admitted": True}
        write(job / "SECONDARY_ADMISSION.json", admission)
        rows.append({"execution_index": int(item["execution_index"]), "context": context_id, "method": method, "job": str(job)})
        print("SECONDARY_BRANCH_DONE", len(rows), 72, context_id, method, flush=True)
    write(out / "SECONDARY_EXECUTION_COMPLETE.json", {"contexts": len(initialized), "branches": len(rows), "rows": rows, "scripted_fallbacks": 0, "physics_stopped": True})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.out)
