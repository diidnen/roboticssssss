#!/usr/bin/env python3
"""Outcome-blind freeze for the task0/task5 independent-root learning curve.

This file changes only the population size.  It intentionally preserves the
authoritative collector, frozen visual feature, PCA17, model/loss constants,
and full-task outcome semantics used by the completed visual-context study.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831"
SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
RUNNER = Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py")
COLLECTOR = ROOT / "prospective_visual_context_collect.py"
OLD_TASK0 = ROOT / "task0_context_sample_complexity_20260831"

TASKS = [0, 5]
OBJECTS = {0: "alphabet soup", 5: "tomato sauce"}
ROOT_LEVELS = [6, 15, 30, 50]
SEEDS = [0, 1, 2]
ROOT_BASE_SEED = 5100
FRICTION_SEED = 2026082306
TRAIN_FORCE_SEEDS = {0: 2026083104, 5: 2026083105}
BANDS = {"LOW": (0.20, 0.30), "MID": (0.45, 0.60), "HIGH": (0.90, 1.00)}
BAND_ORDER = ["LOW", "MID", "HIGH"]
FORCE_RANGE = {0: (3.0, 5.0), 5: (3.0, 5.0)}
TEST_FORCES = {t: [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00] for t in TASKS}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def task_friction_plan(task: int, max_index: int) -> dict[int, dict[str, float]]:
    """Continue the authoritative RNG stream while reproducing task prefix."""
    rng = np.random.default_rng(FRICTION_SEED)
    official_task_order = [0, 1, 5, 6]
    for prior in official_task_order:
        if prior == task:
            break
        for _ in range(12):
            for band in BAND_ORDER:
                rng.uniform(*BANDS[band])
    ans: dict[int, dict[str, float]] = {}
    for idx in range(max_index + 1):
        ans[idx] = {band: float(rng.uniform(*BANDS[band])) for band in BAND_ORDER}
    return ans


def context(split: str, task: int, idx: int, band: str, mu: float) -> dict:
    seed = ROOT_BASE_SEED + idx
    return {
        "context_id": f"pv_{split.lower()}_t{task}_r{idx:02d}_s{seed}_{band.lower()}_mu{mu:.6f}",
        "split": split,
        "task": task,
        "object_identity": OBJECTS[task],
        "root_id": f"pv_{split.lower()}_t{task}_root{idx:02d}_s{seed}",
        "source_root_id": f"p5s0c_{split.lower()}_t{task}_root{idx:02d}_s{seed}",
        "root_index": idx,
        "root_seed": seed,
        "friction_band": band,
        "mu_GT": mu,
        "outcome_selection": "fixed before collection; no outcome-dependent inclusion",
        "state_selection": "fresh legal same-task reset under unique authoritative-style root seed",
        "scientific_retry": 0,
        "eligible": 1,
    }


def make_target(contexts: list[dict], split: str, task: int) -> tuple[dict, list[dict], list[dict]]:
    target: dict[str, list[dict]] = {}
    force_cells: list[dict] = []
    run_rows: list[dict] = []
    if split == "TEST":
        for c in contexts:
            specs = []
            for force in TEST_FORCES[task]:
                for repeat in range(1, 6):
                    label = f"TEST_F{force:.2f}_R{repeat}"
                    specs.append({"force_N": force, "repeat_index": repeat, "branch_label": label})
                    run_rows.append({**c, "force_N": force, "repeat": repeat,
                                     "branch_label": label, "scientific_retry": 0})
            target[c["context_id"]] = specs
        return target, force_cells, run_rows

    rng = np.random.default_rng(TRAIN_FORCE_SEEDS[task])
    lo, hi = FORCE_RANGE[task]
    for c in contexts:
        specs = []
        for stratum in range(5):
            slo = lo + (hi - lo) * stratum / 5
            shi = lo + (hi - lo) * (stratum + 1) / 5
            force = float(rng.uniform(slo, shi))
            force_cells.append({**c, "stratum_index": stratum, "stratum_low_N": slo,
                                "stratum_high_N": shi, "force_N": force,
                                "force_sampling": "uniform_continuous_one_draw_per_equal_width_stratum",
                                "global_rng_seed": TRAIN_FORCE_SEEDS[task], "frozen_before_outcomes": 1})
            for repeat in (1, 2):
                label = f"TRAIN_S{stratum}_F{force:.8f}_R{repeat}"
                specs.append({"force_N": force, "repeat_index": repeat, "branch_label": label})
                run_rows.append({**c, "stratum_index": stratum, "stratum_low_N": slo,
                                 "stratum_high_N": shi, "force_N": force,
                                 "repeat": repeat, "branch_label": label, "scientific_retry": 0})
        target[c["context_id"]] = specs
    return target, force_cells, run_rows


def main() -> None:
    forbidden = [OUT / "collection_test", OUT / "collection_train", OUT / "checkpoints",
                 OUT / "ROOT_SCALING_CHECKPOINT_MANIFEST.json"]
    if any(p.exists() for p in forbidden):
        raise RuntimeError("cannot create/supersede freeze after collection or scaling checkpoints")
    OUT.mkdir(parents=True, exist_ok=True)

    source_rows = list(csv.DictReader((SOURCE / "PROSPECTIVE_CONTEXT_MANIFEST.csv").open(newline="")))
    source_train: dict[int, list[dict]] = {}
    for task in TASKS:
        rows = sorted([r for r in source_rows if r["split"] == "TRAIN" and int(r["task"]) == task],
                      key=lambda r: r["context_id"])
        if len(rows) != 18 or len({r["root_id"] for r in rows}) != 6:
            raise RuntimeError(f"task{task}: authoritative prefix is not 18 contexts / 6 root families")
        plan = task_friction_plan(task, 83)
        for r in rows:
            if abs(float(r["mu_GT"]) - plan[int(r["root_index"])][r["friction_band"]]) > 1e-12:
                raise RuntimeError(f"task{task}: friction RNG continuation does not reproduce prefix")
        source_train[task] = rows

    # Root indices 6--7 were inspected and 8--11 were historical TEST.  Keep
    # all of them outside the untouched scaling population.
    test: dict[int, list[dict]] = {}
    additions: dict[int, list[dict]] = {}
    for task in TASKS:
        plan = task_friction_plan(task, 83)
        additions[task] = [context("TRAIN", task, idx, BAND_ORDER[(idx - 12) % 3],
                                   plan[idx][BAND_ORDER[(idx - 12) % 3]])
                           for idx in range(12, 56)]
        test_bands = ["LOW", "MID", "HIGH", "LOW", "MID", "HIGH", "LOW", "MID", "HIGH", "LOW"]
        test[task] = [context("TEST", task, idx, band, plan[idx][band])
                      for idx, band in zip(range(74, 84), test_bands)]

    # Preserve the already outcome-blind task0 TEST freeze verbatim in
    # identity and numeric friction.  No old collection/checkpoint exists.
    old_test = json.loads((OLD_TASK0 / "TASK0_FROZEN_TEST_CONTEXTS.json").read_text())
    if (OLD_TASK0 / "collection_test_frozen").exists() or (OLD_TASK0 / "checkpoints").exists():
        raise RuntimeError("old task0 TEST is no longer untouched")
    for old, new in zip(sorted(old_test, key=lambda x: x["context_id"]),
                        sorted(test[0], key=lambda x: x["context_id"])):
        if old["context_id"] != new["context_id"] or abs(float(old["mu_GT"]) - float(new["mu_GT"])) > 1e-12:
            raise RuntimeError("task0 frozen TEST identity differs from preserved pre-outcome freeze")

    levels: dict[str, dict[str, dict]] = {}
    for n in ROOT_LEVELS:
        added = n - 6
        levels[f"S{n}"] = {}
        for task in TASKS:
            ids = [r["context_id"] for r in source_train[task]] + [r["context_id"] for r in additions[task][:added]]
            roots = sorted({r["root_id"] for r in source_train[task]} | {r["root_id"] for r in additions[task][:added]})
            if len(roots) != n:
                raise RuntimeError(f"task{task}/S{n}: independent root count mismatch")
            levels[f"S{n}"][str(task)] = {
                "independent_root_count": n,
                "friction_conditioned_context_count": len(ids),
                "root_ids": roots,
                "context_ids": ids,
            }
    for task in TASKS:
        sets = [set(levels[f"S{n}"][str(task)]["context_ids"]) for n in ROOT_LEVELS]
        if not all(a < b for a, b in zip(sets[:-1], sets[1:])):
            raise RuntimeError(f"task{task}: nested context invariant failed")

    plan_dir = OUT / "collection_plans"
    all_test_contexts = [x for task in TASKS for x in test[task]]
    all_train_contexts = [x for task in TASKS for x in additions[task]]
    write_json(plan_dir / "TEST_CONTEXTS.json", all_test_contexts)
    write_json(plan_dir / "TRAIN_CONTEXTS.json", all_train_contexts)
    test_targets = {}
    train_targets = {}
    planned_counts = []
    for task in TASKS:
        tt, _, truns = make_target(test[task], "TEST", task)
        nt, cells, nruns = make_target(additions[task], "TRAIN", task)
        test_targets.update(tt); train_targets.update(nt)
        write_csv(plan_dir / f"TASK{task}_TEST_RUN_MANIFEST.csv", truns)
        write_csv(plan_dir / f"TASK{task}_TRAIN_FORCE_MANIFEST.csv", cells)
        write_csv(plan_dir / f"TASK{task}_TRAIN_RUN_MANIFEST.csv", nruns)
        planned_counts.extend([
            {"task": task, "split": "TEST", "status": "PLANNED", "independent_roots": 10,
             "friction_conditioned_contexts": 10, "force_cells": 90, "branches": 450,
             "scientific_failures": "", "infrastructure_failures": "", "retries": 0},
            {"task": task, "split": "TRAIN_ADDITIONS_TO_S50", "status": "PLANNED",
             "independent_roots": 44, "friction_conditioned_contexts": 44,
             "force_cells": 220, "branches": 440, "scientific_failures": "",
             "infrastructure_failures": "", "retries": 0},
        ])
    write_json(plan_dir / "TEST_TARGET_MANIFEST.json", {
        "manifest_name": "ROOT_SCALING_UNTOUCHED_TEST_TARGET", "split": "TEST",
        "contexts": test_targets, "expected_contexts": 20, "expected_force_cells": 180,
        "expected_branches": 900,
        "branch_state": "strict pre-probe last-hold snapshot restored before every branch",
        "frontier_definition": "minimum tested F with >=4/5 full-task successes", "scientific_retry": 0,
    })
    write_json(plan_dir / "TRAIN_TARGET_MANIFEST.json", {
        "manifest_name": "ROOT_SCALING_NEW_TRAIN_TARGET_TO_S50", "split": "TRAIN",
        "contexts": train_targets, "expected_contexts": 88, "expected_force_cells": 440,
        "expected_branches": 880,
        "branch_state": "strict pre-probe last-hold snapshot restored before every branch", "scientific_retry": 0,
    })

    created = datetime.now(timezone.utc).isoformat()
    test_manifest = {
        "status": "FROZEN_UNTOUCHED_BEFORE_NEW_TRAIN_COLLECTION_OR_TRAINING",
        "created_utc": created, "tasks": TASKS, "test_roots_per_task": 10,
        "contexts": {str(t): test[t] for t in TASKS},
        "excluded_seen_root_indices": [6, 7], "excluded_historical_test_root_indices": [8, 9, 10, 11],
        "task0_test_preserved_from_prior_outcome_blind_freeze": True,
        "task0_prior_test_freeze_sha256": sha256(OLD_TASK0 / "TASK0_FROZEN_TEST_CONTEXTS.json"),
        "dense_force_grid_N": TEST_FORCES, "repeats_per_force": 5,
        "frontier_definition": "minimum tested F with >=4/5 full downstream task successes",
        "forbidden_uses": ["PCA fit", "normalization fit", "model/checkpoint/seed selection",
                           "early stopping", "feature/loss/architecture changes", "calibration"],
        "target_manifest": str(plan_dir / "TEST_TARGET_MANIFEST.json"),
        "outcomes_read_during_freeze": False,
    }
    train_manifest = {
        "status": "FROZEN_NESTED_INDEPENDENT_ROOT_SETS_BEFORE_COLLECTION_OR_TRAINING",
        "created_utc": created, "tasks": TASKS, "root_levels": ROOT_LEVELS,
        "independence_unit": "unique root_seed/root family; friction/repeats/timesteps are not independent roots",
        "S6_semantics": "all existing 18 friction-conditioned contexts grouped into six authoritative root-seed families",
        "new_root_semantics": "one unique root seed with one preregistered cycling friction band and 5 strata x2 repeats",
        "nested_sets": levels,
        "new_contexts": {str(t): additions[t] for t in TASKS},
        "collection_protocol": "5 equal-width continuous force strata over 3--5 N x2 repeats/root",
        "models": ["Base", "Full Visual", "Visual Joint"], "seeds": SEEDS,
        "preprocessing": {"raw_visual_dimension": 4096, "PCA_dimension": 17,
                          "fit_current_TRAIN_only": True, "TEST_transform_only": True},
        "frozen_training": {"epochs": 80, "optimizer": "AdamW",
                            "architecture_loss_lambda_force_outcome_normalization_changes_forbidden": True},
        "target_manifest": str(plan_dir / "TRAIN_TARGET_MANIFEST.json"),
        "outcomes_read_during_freeze": False,
    }
    write_json(OUT / "ROOT_SCALING_TEST_MANIFEST.json", test_manifest)
    write_json(OUT / "ROOT_SCALING_TRAIN_MANIFEST.json", train_manifest)
    write_csv(OUT / "ROOT_SCALING_DATA_COUNTS.csv", planned_counts)
    write_json(OUT / "ROOT_SCALING_PREREGISTRATION.json", {
        "status": "FROZEN_BEFORE_UNTOUCHED_TEST_OUTCOMES_AND_NEW_TRAIN_OUTCOMES",
        "hypothesis": "H_sample_complexity", "single_varied_factor": "independent_root_count",
        "minimum_completed_scale_before_conclusion": "S30 and one larger point (S50) if S30 is flat",
        "S15_is_final_evidence": False,
        "gate_A": ["Full Visual TEST frontier and/or NLL improves stably with N",
                   "not driven by one TEST root", "under-force nonworse", "three seeds directionally consistent",
                   "TRAIN-to-TEST gap clearly shrinks"],
        "stable_trend_rule": "S50 better than S6 and at least two of S6->S15, S15->S30, S30->S50 have claimed sign",
        "multiple_root_rule": "endpoint benefit on >=6/10 TEST roots/task and paired median has claimed sign",
        "joint_gate": ["frontier MAE < Full Visual", "under-force <= Full Visual",
                       "probability metrics non-inferior", "multiple roots benefit", "seed stable"],
        "representation_inefficiency_gate": "at S50 train remains strong, TEST lacks systematic improvement/advantage, and gap does not shrink",
        "allowed_classifications": [
            "VISUAL_CONTEXT_IS_INDEPENDENT_ROOT_SAMPLE_LIMITED",
            "JOINT_VALUE_EMERGES_WITH_CONTEXT_SCALE",
            "VISUAL_IMPROVES_BUT_JOINT_NOT_NEEDED",
            "JOINT_FRONTIER_SIGNAL_WITHOUT_SAFE_CONTROL",
            "GENERIC_VISUAL_CONTEXT_REMAINS_INEFFICIENT_WITH_MORE_ROOTS",
            "INSUFFICIENT_ROOT_SCALE_TO_DISTINGUISH_SAMPLE_COMPLEXITY_FROM_REPRESENTATION",
        ],
    })

    frozen_files = [OUT / "ROOT_SCALING_TEST_MANIFEST.json", OUT / "ROOT_SCALING_TRAIN_MANIFEST.json",
                    OUT / "ROOT_SCALING_PREREGISTRATION.json", plan_dir / "TEST_CONTEXTS.json",
                    plan_dir / "TRAIN_CONTEXTS.json", plan_dir / "TEST_TARGET_MANIFEST.json",
                    plan_dir / "TRAIN_TARGET_MANIFEST.json", RUNNER, COLLECTOR,
                    SOURCE / "PROSPECTIVE_VISUAL_FEATURE_SPEC.json"]
    write_json(OUT / "ROOT_SCALING_FREEZE_SHA256.json", {
        "status": "FROZEN", "created_utc": created,
        "new_collection_outcomes_present": False, "new_scaling_models_trained": False,
        "hashes": {str(p): sha256(p) for p in frozen_files},
    })
    print(json.dumps({"status": "FROZEN", "tasks": TASKS, "root_levels": ROOT_LEVELS,
                      "new_TEST_roots": 20, "new_TEST_branches": 900,
                      "new_TRAIN_roots_to_S50": 88, "new_TRAIN_branches": 880}, indent=2))


if __name__ == "__main__":
    main()
