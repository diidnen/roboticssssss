#!/usr/bin/env python3
"""Integrity, force-authority, and failure-slice audit for the 648-row grid."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_TASKS = {"handover_mic", "dump_bin_bigbin"}
EXPECTED_MUS = {0.25, 0.55, 0.85}
EXPECTED_FORCES = (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0)
EXPECTED_PARENT_SHA256 = "234082d3843a65ea32a82768ff0b9c80080d175beaffb93747991f2cb1681dc1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def spearman(values: list[float]) -> float:
    x = np.arange(len(values), dtype=np.float64)
    y = ranks(np.asarray(values, dtype=np.float64))
    if np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--parent-snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(EXPECTED_TASKS))
    parser.add_argument("--expected-rows", type=int, default=648)
    parser.add_argument("--expected-contexts", type=int, default=72)
    parser.add_argument("--expected-parent-sha256", default=EXPECTED_PARENT_SHA256)
    args = parser.parse_args()

    expected_tasks = {args.task} if args.task else EXPECTED_TASKS

    rows = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    warnings: list[str] = []
    if sha256(args.parent_snapshot) != args.expected_parent_sha256:
        errors.append("parent snapshot hash mismatch")
    parent_lines = args.parent_snapshot.read_text(encoding="utf-8").splitlines(keepends=True)
    record_lines = args.records.read_text(encoding="utf-8").splitlines(keepends=True)
    if record_lines[: len(parent_lines)] != parent_lines:
        errors.append("parent rows are not a byte-identical prefix")
    if len(rows) != args.expected_rows:
        errors.append(f"expected {args.expected_rows} rows, found {len(rows)}")
    keys = [row["branch_key"] for row in rows]
    if len(keys) != len(set(keys)):
        errors.append("duplicate branch keys")
    if {row["task"] for row in rows} != expected_tasks:
        errors.append("task set mismatch")
    if {float(row["friction"]) for row in rows} != EXPECTED_MUS:
        errors.append("friction support mismatch")
    if {float(row["force_n"]) for row in rows} != set(EXPECTED_FORCES):
        errors.append("force support mismatch")
    invalid = [row["branch_key"] for row in rows if not row.get("valid")]
    if invalid:
        errors.append(f"invalid query branches: {len(invalid)}")

    sibling_groups: dict[tuple[str, int, float], list[dict]] = defaultdict(list)
    measured: dict[str, dict[float, list[float]]] = {
        task: {force: [] for force in EXPECTED_FORCES} for task in expected_tasks
    }
    for row in rows:
        sibling_groups[(row["task"], int(row["root_slot"]), float(row["friction"]))].append(row)
        evidence = row["evidence"]
        query = evidence.get("query_info") or {}
        if query.get("schema_id") != "ROBOTWIN_SHEAR_QUERY_OBSERVABLES_V1":
            errors.append(f"missing query schema: {row['branch_key']}")
        if any("friction" in key.lower() for key in query):
            errors.append(f"hidden friction leaked into query: {row['branch_key']}")
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
        sample_count = int(metrics.get("samples", 0) or 0)
        contact_ratio = float(metrics.get("contact_ratio", 0.0) or 0.0)
        mean_force = metrics.get("measured_force_mean_n")
        if (
            mean_force is not None
            and np.isfinite(float(mean_force))
            and sample_count >= 20
            and contact_ratio >= 0.50
        ):
            measured[row["task"]][requested].append(float(mean_force))

    if len(sibling_groups) != args.expected_contexts:
        errors.append(f"expected {args.expected_contexts} contexts, found {len(sibling_groups)}")
    split_roots: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        split_roots[row["split"]].add((row["task"], int(row["root_slot"])))
    split_names = sorted(split_roots)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = split_roots[left] & split_roots[right]
            if overlap:
                errors.append(f"split root overlap: {left}/{right} {sorted(overlap)}")

    masks = Counter()
    by_task_masks: dict[str, Counter] = defaultdict(Counter)
    context_rows = []
    nonmonotone = []
    for key, siblings in sorted(sibling_groups.items()):
        force_map = {float(row["force_n"]): row for row in siblings}
        if set(force_map) != set(EXPECTED_FORCES):
            errors.append(f"incomplete force siblings: {key}")
            continue
        seeds = {int(row["actual_seed"]) for row in siblings}
        if len(seeds) != 1:
            errors.append(f"sibling seed mismatch: {key} -> {sorted(seeds)}")
        outcomes = [int(bool(force_map[force]["evidence"]["success"])) for force in EXPECTED_FORCES]
        mask = "".join(map(str, outcomes))
        masks[mask] += 1
        by_task_masks[key[0]][mask] += 1
        inversions = sum(outcomes[index] > outcomes[index + 1] for index in range(len(outcomes) - 1))
        if inversions:
            nonmonotone.append({"context": list(key), "mask": mask, "adjacent_inversions": inversions})
        successful_forces = [force for force, outcome in zip(EXPECTED_FORCES, outcomes) if outcome]
        context_rows.append(
            {
                "task": key[0],
                "root_slot": key[1],
                "friction": key[2],
                "mask": mask,
                "all_fail": not any(outcomes),
                "all_success": all(outcomes),
                "force_sensitive": any(outcomes) and not all(outcomes),
                "minimum_observed_success_n": min(successful_forces) if successful_forces else None,
            }
        )

    authority = {}
    authority_passed = True
    for task in sorted(expected_tasks):
        medians = []
        counts = []
        for force in EXPECTED_FORCES:
            values = measured[task][force]
            counts.append(len(values))
            medians.append(float(np.median(values)) if values else None)
        if any(value is None for value in medians):
            rho = None
            adjacent_reversals = None
            task_passed = False
        else:
            numeric = [float(value) for value in medians]
            rho = spearman(numeric)
            adjacent_reversals = sum(
                numeric[index + 1] + 0.05 < numeric[index]
                for index in range(len(numeric) - 1)
            )
            task_passed = rho >= 0.80 and adjacent_reversals <= 1
        authority[task] = {
            "eligible_rows_per_force": dict(zip(map(str, EXPECTED_FORCES), counts)),
            "median_measured_force_n": dict(zip(map(str, EXPECTED_FORCES), medians)),
            "spearman_requested_vs_median_measured": rho,
            "adjacent_gross_reversals": adjacent_reversals,
            "passed": task_passed,
        }
        authority_passed = authority_passed and task_passed
        if not task_passed:
            warnings.append(f"continuous force authority not supported for {task}")

    integrity_passed = not errors
    report = {
        "schema_id": "AF_ROBOTWIN_TASKFORMS_FORCEGRID648_AUDIT_V1",
        "records": str(args.records),
        "records_sha256": sha256(args.records),
        "parent_snapshot_sha256": sha256(args.parent_snapshot),
        "rows": len(rows),
        "contexts": len(sibling_groups),
        "tasks": sorted(expected_tasks),
        "valid_rows": len(rows) - len(invalid),
        "integrity_passed": integrity_passed,
        "continuous_force_authority_passed": authority_passed,
        "passed": integrity_passed and authority_passed,
        "errors": errors,
        "warnings": warnings,
        "requested_realized_authority": authority,
        "failure_slices": {
            "all_fail_contexts": sum(row["all_fail"] for row in context_rows),
            "all_success_contexts": sum(row["all_success"] for row in context_rows),
            "force_sensitive_contexts": sum(row["force_sensitive"] for row in context_rows),
            "nonmonotone_contexts": len(nonmonotone),
            "mask_counts": dict(masks),
            "mask_counts_by_task": {task: dict(counter) for task, counter in by_task_masks.items()},
        },
        "context_rows": context_rows,
        "nonmonotone_details": nonmonotone,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
