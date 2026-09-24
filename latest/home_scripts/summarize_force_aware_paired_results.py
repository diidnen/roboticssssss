#!/usr/bin/env python3
"""Recompute success/force trade-offs for paired AF and strict No-Query."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def summarize(values):
    successful = [row for row in values if row["success"]]
    return {
        "n": len(values),
        "successes": sum(row["success"] for row in values),
        "success_rate": statistics.mean(row["success"] for row in values),
        "mean_selected_force_N_all": statistics.mean(row["selected_force_N"] for row in values),
        "mean_selected_force_N_success_only": statistics.mean(
            row["selected_force_N"] for row in successful
        ),
        "mean_measured_squeeze_N_all": statistics.mean(
            row["measured_squeeze_N"] for row in values
        ),
        "mean_measured_squeeze_N_success_only": statistics.mean(
            row["measured_squeeze_N"] for row in successful
        ),
        "mean_realized_utility": statistics.mean(row["utility"] for row in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = read(args.root / "PAIRED_RESULTS.json")
    freeze = read(args.freeze)
    methods = {}
    branch_names = {"strict": "STRICT_NOQUERY", "af": "ACTIVEFORCING"}
    for method, branch_name in branch_names.items():
        rows = []
        for paired in result["rows"]:
            branch = read(args.root / "contexts" / paired["context_id"] / branch_name / "BRANCH_RESULT.json")
            success = int(paired[f"{method}_success"])
            force = float(paired[f"{method}_force_N"])
            rows.append(
                {
                    "context_id": paired["context_id"],
                    "task": int(paired["task"]),
                    "success": success,
                    "selected_force_N": force,
                    "measured_squeeze_N": float(
                        branch["outcome"]["mean_measured_bilateral_squeeze"]
                    ),
                    "utility": (5.0 - force) / 5.0 if success else -1.0,
                }
            )
        methods[method] = {"overall": summarize(rows)}
        methods[method]["per_task"] = {
            str(task): summarize([row for row in rows if row["task"] == task])
            for task in sorted({row["task"] for row in rows})
        }
        methods[method]["rows"] = rows

    # Verify that the original scalar-utility freeze also equals the requested
    # lexicographic rule: maximize TRAIN success, then choose the lowest force.
    lexicographic = {}
    for task, stats in freeze["per_task_primary"].items():
        force_stats = stats["force_statistics"]
        max_success = max(item["success_rate"] for item in force_stats.values())
        selected = min(
            float(force)
            for force, item in force_stats.items()
            if item["success_rate"] == max_success
        )
        lexicographic[task] = {
            "max_train_success_rate": max_success,
            "lowest_force_at_max_success_N": selected,
            "matches_frozen_force": selected
            == float(freeze["frozen_strict_noquery_policy"]["force_by_task"][task]),
        }

    strict_rows = methods["strict"]["rows"]
    af_rows = methods["af"]["rows"]
    by_id_strict = {row["context_id"]: row for row in strict_rows}
    by_id_af = {row["context_id"]: row for row in af_rows}
    force_differences = [
        by_id_af[context_id]["selected_force_N"] - by_id_strict[context_id]["selected_force_N"]
        for context_id in by_id_strict
    ]
    both_success = [
        context_id
        for context_id in by_id_strict
        if by_id_strict[context_id]["success"] and by_id_af[context_id]["success"]
    ]
    output = {
        "schema": "FORCE_AWARE_PAIRED_AF_STRICT_NOQUERY_V1",
        "objective_interpretation": "lexicographic: maximize success first, then minimize force",
        "train_lexicographic_freeze_check": lexicographic,
        "all_tasks_match_original_frozen_force": all(
            item["matches_frozen_force"] for item in lexicographic.values()
        ),
        "methods": methods,
        "paired_force_comparison": {
            "mean_af_minus_strict_selected_force_N": statistics.mean(force_differences),
            "af_higher_force_pairs": sum(delta > 0 for delta in force_differences),
            "af_equal_force_pairs": sum(delta == 0 for delta in force_differences),
            "af_lower_force_pairs": sum(delta < 0 for delta in force_differences),
            "both_success_pairs": len(both_success),
            "mean_af_minus_strict_force_N_among_both_success": statistics.mean(
                by_id_af[context_id]["selected_force_N"]
                - by_id_strict[context_id]["selected_force_N"]
                for context_id in both_success
            ),
        },
        "strict_lexicographically_dominates_af": (
            methods["strict"]["overall"]["success_rate"]
            > methods["af"]["overall"]["success_rate"]
            and methods["strict"]["overall"]["mean_selected_force_N_success_only"]
            < methods["af"]["overall"]["mean_selected_force_N_success_only"]
        ),
        "no_new_physics_executed": True,
    }
    # Keep the visible summary compact; row-level evidence remains in the source tree.
    for method in methods:
        methods[method].pop("rows")
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
