#!/usr/bin/env python3
"""Hard-gate audit for the 216-branch task-form collection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


EXPECTED_TASKS = {"handover_mic", "dump_bin_bigbin"}
EXPECTED_MUS = {0.25, 0.55, 0.85}
EXPECTED_FORCES = {3, 4, 5}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    errors = []
    if len(rows) != 216:
        errors.append(f"expected 216 rows, found {len(rows)}")
    keys = [row["branch_key"] for row in rows]
    if len(keys) != len(set(keys)):
        errors.append("duplicate branch keys")
    if {row["task"] for row in rows} != EXPECTED_TASKS:
        errors.append("task set mismatch")
    if {row["friction"] for row in rows} != EXPECTED_MUS:
        errors.append("friction support mismatch")
    if {row["force_n"] for row in rows} != EXPECTED_FORCES:
        errors.append("force support mismatch")
    invalid = [row["branch_key"] for row in rows if not row.get("valid")]
    if invalid:
        errors.append(f"invalid query branches: {len(invalid)}")

    sibling_groups = defaultdict(list)
    for row in rows:
        evidence = row["evidence"]
        sibling_groups[(row["task"], row["root_slot"], row["friction"])].append(row)
        query = evidence.get("query_info") or {}
        if query.get("schema_id") != "ROBOTWIN_SHEAR_QUERY_OBSERVABLES_V1":
            errors.append(f"missing query schema: {row['branch_key']}")
        if any("friction" in key.lower() for key in query):
            errors.append(f"hidden friction leaked into query fields: {row['branch_key']}")
        action = np.asarray(evidence.get("preaction_sequence"), dtype=np.float32)
        state = np.asarray(evidence.get("preaction_state"), dtype=np.float32)
        if action.shape != (50, 14) or not np.isfinite(action).all():
            errors.append(f"bad preaction shape: {row['branch_key']} {action.shape}")
        if state.shape != (14,) or not np.isfinite(state).all():
            errors.append(f"bad state shape: {row['branch_key']} {state.shape}")
        metrics = evidence.get("dynamic_metrics") or {}
        requested = float(row["force_n"])
        for side in ("left", "right"):
            value = metrics.get(f"{side}_force_limit_n")
            if value is None or abs(float(value) - requested) > 1e-6:
                errors.append(f"force-limit readback mismatch: {row['branch_key']} {side}")
        if evidence.get("af_object_mass_kg") is not None:
            errors.append(f"mass varied: {row['branch_key']}")

    nonmonotone = []
    task_labels = defaultdict(list)
    for key, siblings in sibling_groups.items():
        forces = {row["force_n"] for row in siblings}
        seeds = {row["actual_seed"] for row in siblings}
        if forces != EXPECTED_FORCES:
            errors.append(f"incomplete siblings: {key}")
            continue
        if len(seeds) != 1:
            errors.append(f"sibling seed mismatch: {key} -> {sorted(seeds)}")
        outcomes = {
            row["force_n"]: int(row["evidence"]["success"]) for row in siblings
        }
        if not outcomes[3] <= outcomes[4] <= outcomes[5]:
            nonmonotone.append({"context": key, "outcomes": outcomes})
        task_labels[key[0]].extend(outcomes.values())

    split_seeds = defaultdict(set)
    for row in rows:
        split_seeds[row["split"]].add((row["task"], row["root_slot"]))
    names = list(split_seeds)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = split_seeds[left] & split_seeds[right]
            if overlap:
                errors.append(f"split root overlap: {left}/{right} {sorted(overlap)}")
    for task, labels in task_labels.items():
        if len(set(labels)) < 2:
            errors.append(f"task has no success/failure contrast: {task}")

    context_count = len(sibling_groups)
    nonmonotone_rate = len(nonmonotone) / max(context_count, 1)
    if nonmonotone_rate > 0.15:
        errors.append(f"nonmonotone context rate too high: {nonmonotone_rate:.3f}")
    report = {
        "schema_id": "AF_ROBOTWIN_TASKFORMS_COLLECTION_AUDIT_V1",
        "rows": len(rows),
        "contexts": context_count,
        "valid_rows": len(rows) - len(invalid),
        "success_counts": Counter(
            f"{row['task']}:{row['force_n']}:{int(row['evidence']['success'])}" for row in rows
        ),
        "nonmonotone_contexts": nonmonotone,
        "nonmonotone_rate": nonmonotone_rate,
        "errors": errors,
        "passed": not errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
