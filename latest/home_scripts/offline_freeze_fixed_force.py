#!/usr/bin/env python3
"""Freeze fixed-force baselines using TRAIN rollouts only.

The selector deliberately ignores TEST rows.  It first restricts the primary
calculation to contexts with every empirically executed force, so all forces
are compared on paired contexts.  Available-row estimates are emitted only as
a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utility(success: int, force: float, fmax: float) -> float:
    return (fmax - force) / fmax if success else -1.0


def summarize(rows, forces, fmax):
    by_force = {}
    for force in forces:
        selected = [row for row in rows if row["force"] == force]
        n = len(selected)
        successes = sum(row["success"] for row in selected)
        by_force[str(force)] = {
            "n": n,
            "successes": successes,
            "success_rate": successes / n if n else None,
            "mean_utility": (
                sum(utility(row["success"], force, fmax) for row in selected) / n
                if n
                else None
            ),
        }
    admissible = [
        (stats["mean_utility"], float(force))
        for force, stats in by_force.items()
        if stats["mean_utility"] is not None
    ]
    # Frozen tie break: among equal utilities, select the lower force.
    best_utility, best_force = max(admissible, key=lambda item: (item[0], -item[1]))
    return {
        "selected_force": best_force,
        "selected_mean_utility": best_utility,
        "force_statistics": by_force,
    }


def score_policy(rows, force_by_task, fmax):
    chosen = [row for row in rows if row["force"] == force_by_task[row["task"]]]
    successes = sum(row["success"] for row in chosen)
    return {
        "n": len(chosen),
        "successes": successes,
        "success_rate": successes / len(chosen),
        "mean_utility": sum(
            utility(row["success"], row["force"], fmax) for row in chosen
        )
        / len(chosen),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fmax", type=float, default=5.0)
    args = parser.parse_args()

    train_rows = []
    val_rows = []
    with args.csv.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            split = raw["split"].strip().upper()
            if split not in {"TRAIN", "VAL"}:
                # In particular, TEST outcomes are never parsed or evaluated.
                continue
            row = {
                "context": (
                    raw["task"],
                    raw["root_family"],
                    raw["friction_context"],
                    raw["context_id"],
                ),
                "task": raw["task"],
                "force": float(raw["executed_candidate_F"]),
                "success": int(raw["full_task_success_y_recomputed"]),
            }
            (train_rows if split == "TRAIN" else val_rows).append(row)

    tasks = sorted({row["task"] for row in train_rows})
    forces = sorted({row["force"] for row in train_rows})
    observed = defaultdict(set)
    for row in train_rows:
        observed[row["context"]].add(row["force"])
    complete_contexts = {
        context for context, context_forces in observed.items() if context_forces == set(forces)
    }
    incomplete = {
        "|".join(context): sorted(set(forces) - context_forces)
        for context, context_forces in observed.items()
        if context_forces != set(forces)
    }
    paired_rows = [row for row in train_rows if row["context"] in complete_contexts]

    global_primary = summarize(paired_rows, forces, args.fmax)
    per_task_primary = {
        task: summarize(
            [row for row in paired_rows if row["task"] == task], forces, args.fmax
        )
        for task in tasks
    }
    global_force_map = {task: global_primary["selected_force"] for task in tasks}
    per_task_force_map = {
        task: result["selected_force"] for task, result in per_task_primary.items()
    }

    output = {
        "schema": "STRICT_NOQUERY_FIXED_FORCE_FREEZE_V1",
        "selection_split": "TRAIN_ONLY",
        "test_outcomes_accessed": False,
        "source_csv": str(args.csv.resolve()),
        "source_csv_sha256": sha256(args.csv),
        "objective": "mean[p*(Fmax-F)/Fmax + (1-p)*(-1)]",
        "fmax": args.fmax,
        "candidate_forces": forces,
        "tie_break": "lower_force",
        "primary_pairing": "complete TRAIN contexts only",
        "train_rows_total": len(train_rows),
        "train_contexts_total": len(observed),
        "train_contexts_complete": len(complete_contexts),
        "incomplete_train_contexts": incomplete,
        "global_primary": global_primary,
        "per_task_primary": per_task_primary,
        "policy_comparison_train_primary": {
            "global_fixed": score_policy(paired_rows, global_force_map, args.fmax),
            "per_task_fixed": score_policy(paired_rows, per_task_force_map, args.fmax),
        },
        "available_row_sensitivity": {
            "global": summarize(train_rows, forces, args.fmax),
            "per_task": {
                task: summarize(
                    [row for row in train_rows if row["task"] == task],
                    forces,
                    args.fmax,
                )
                for task in tasks
            },
        },
        "frozen_strict_noquery_policy": {
            "kind": "per_task_fixed_force",
            "force_by_task": per_task_force_map,
            "uses_probe": False,
            "loads_belief": False,
            "loads_feasibility": False,
        },
    }

    # VAL is evaluated only at the already selected forces; it never selects a force.
    if val_rows:
        output["post_freeze_val_diagnostic"] = {
            "global_fixed": score_policy(val_rows, global_force_map, args.fmax),
            "per_task_fixed": score_policy(val_rows, per_task_force_map, args.fmax),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
