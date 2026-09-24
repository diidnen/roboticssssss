#!/usr/bin/env python3
"""Seal HELDOUT branch bytes without parsing or exposing scientific labels."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from time_bounded_mass_common import HERE, load_frozen_subset, sha256


def main() -> None:
    output = HERE / "TIME_BOUNDED_MASS_HELDOUT_SEAL.json"
    if output.exists(): raise FileExistsError(output)
    subset, subset_sha = load_frozen_subset()
    rows = []
    required = ["LAUNCH_CLAIM.json", "WORKER_COMPLETION.json", "BRANCH_RESULT.json",
                "SOURCE_HASHES_BEFORE.json", "SOURCE_HASHES_AFTER.json",
                "PREACTION_STATE_COMPARISON.json", "PREACTION_SEQUENCE.npy",
                "MASS_INTERVENTION_READBACK.json"]
    for spec in subset["jobs"]:
        if spec["split"] != "HELDOUT": continue
        job = HERE / spec["relative_branch_path"]
        missing = [name for name in required if not (job / name).is_file()]
        if missing: raise RuntimeError(f"cannot seal incomplete HELDOUT branch {job}: {missing}")
        rows.append({"queue_index": spec["queue_index"], "context_id": spec["context_id"],
                     "force_N": spec["force_N"], "branch_path": str(job),
                     "artifact_sha256": {name: sha256(job / name) for name in required}})
    if len(rows) != 36: raise RuntimeError("HELDOUT seal must contain exactly 36 branches")
    seal = {
        "schema": "TIME_BOUNDED_MASS_HELDOUT_SEAL_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset_sha256": subset_sha,
        "branch_count": len(rows),
        "scientific_labels_parsed_or_exported": False,
        "training_or_selection_access_authorized": False,
        "opening_condition": "Selected checkpoints, evaluator, analyzer, validator, and analysis protocol are hash-locked.",
        "branches": rows,
        "source_sha256": sha256(Path(__file__)),
    }
    with output.open("x") as stream: json.dump(seal, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps({"status": "SEALED", "branches": len(rows), "seal_sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
