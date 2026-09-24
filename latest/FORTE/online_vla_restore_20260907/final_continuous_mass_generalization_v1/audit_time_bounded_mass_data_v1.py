#!/usr/bin/env python3
"""Independent data-quality gates for the reduced TRAIN/VAL and final 216 sets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from time_bounded_mass_common import HERE, load_frozen_subset, read, sha256, validate_branch


def write_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train-val", "final"))
    args = parser.parse_args()
    output = HERE / ("TIME_BOUNDED_MASS_TRAIN_VAL_DATA_AUDIT.json" if args.stage == "train-val" else "TIME_BOUNDED_MASS_DATA_AUDIT.json")
    subset, subset_sha = load_frozen_subset()
    runtime = read(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json")
    splits = ["TRAIN", "VAL"] if args.stage == "train-val" else ["TRAIN", "VAL", "HELDOUT"]
    expected = sum(subset["split_branch_targets"][split] for split in splits)
    rows, violations = [], []
    sibling_forces = defaultdict(set)
    label_counts = Counter()
    artifact_hashes = {}
    for spec in subset["jobs"]:
        if spec["split"] not in splits:
            continue
        job = HERE / spec["relative_branch_path"]
        check = validate_branch(job, spec, runtime, HERE / "references" / spec["context_id"])
        if not check["valid"]:
            violations.append({"path": str(job), "failed_checks": check["failed_checks"]})
            continue
        result = read(job / "BRANCH_RESULT.json")
        label = int(result["full_task_success_y"])
        rows.append({"queue_index": spec["queue_index"], "split": spec["split"],
                     "context_id": spec["context_id"], "task": spec["task"],
                     "root": spec["root"], "mass_kg": spec["mass_kg"],
                     "force_N": spec["force_N"], "label": label, "path": str(job)})
        sibling_forces[spec["context_id"]].add(float(spec["force_N"]))
        label_counts[(spec["split"], spec["task"], spec["mass_kg"], spec["force_N"], label)] += 1
        artifact_hashes[f"{spec['context_id']}|F{spec['force_N']:g}"] = check["artifact_sha256"]

    split_roots = {split: sorted({row["root"] for row in rows if row["split"] == split}) for split in splits}
    expected_roots = {split: subset["roots_by_split"][split] for split in splits}
    expected_contexts = sum(subset["split_context_targets"][split] for split in splits)
    checks = {
        "exact_primary_cardinality": len(rows) == expected,
        "unique_context_force_key": len({(row["context_id"], row["force_N"]) for row in rows}) == expected,
        "only_frozen_primary_forces": {row["force_N"] for row in rows} == {3.0, 4.0, 5.0},
        "all_sibling_force_sets_exact": len(sibling_forces) == expected_contexts and all(values == {3.0, 4.0, 5.0} for values in sibling_forces.values()),
        "split_roots_exact": split_roots == expected_roots,
        "split_roots_disjoint": len(set().union(*(set(values) for values in split_roots.values()))) == sum(len(values) for values in split_roots.values()),
        "all_labels_valid_binary": len(rows) == expected and all(row["label"] in (0, 1) for row in rows),
        "all_runtime_state_hash_checks_pass": not violations,
        "auxiliary_forces_excluded": not any(row["force_N"] not in {3.0, 4.0, 5.0} for row in rows),
        "no_result_driven_reruns": read(HERE / "TIME_BOUNDED_MASS_INFRASTRUCTURE_INCIDENTS.json")["result_driven_reruns"] == 0,
    }
    passed = all(checks.values())
    result = {
        "schema": "TIME_BOUNDED_MASS_DATA_AUDIT_V1",
        "stage": args.stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "subset_sha256": subset_sha,
        "splits_audited": splits,
        "expected_branches": expected,
        "valid_branches": len(rows),
        "checks": checks,
        "violations": violations,
        "roots_by_split": split_roots,
        "label_profile": [
            {"split": key[0], "task": key[1], "mass_kg": key[2], "force_N": key[3], "label": key[4], "count": value}
            for key, value in sorted(label_counts.items())
        ],
        "branch_artifact_sha256": artifact_hashes,
        "source_sha256": {str(Path(__file__)): sha256(Path(__file__)),
                          str(HERE / "time_bounded_mass_common.py"): sha256(HERE / "time_bounded_mass_common.py")},
    }
    write_new(output, result)
    print(json.dumps({"status": result["status"], "stage": args.stage,
                      "valid": len(rows), "expected": expected, "violations": len(violations)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
