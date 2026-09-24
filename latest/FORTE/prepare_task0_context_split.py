#!/usr/bin/env python3
"""Freeze the same-task0 context learning-curve TRAIN/TEST split.

This script reads no outcome file.  It extends the authoritative task0 root
seed/friction construction, preserves the 5-strata x2 TRAIN protocol, and
freezes a new 9-force x5 TEST before any new model training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/exouser/FORTE")
SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
OUT = ROOT / "task0_context_sample_complexity_20260831"
RUNNER = Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py")
COLLECTOR = ROOT / "prospective_visual_context_collect.py"
VISUAL_SPEC = SOURCE / "PROSPECTIVE_VISUAL_FEATURE_SPEC.json"
FRICTION_SEED = 2026082306
NEW_FORCE_SEED = 2026083104
ROOT_BASE_SEED = 5100
BANDS = {"LOW": (0.20, 0.30), "MID": (0.45, 0.60), "HIGH": (0.90, 1.00)}
BAND_ORDER = ["LOW", "MID", "HIGH"]
TEST_FORCES = [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def friction_plan(max_index: int) -> dict[int, dict[str, float]]:
    rng = np.random.default_rng(FRICTION_SEED)
    ans = {}
    for idx in range(max_index + 1):
        ans[idx] = {band: float(rng.uniform(*BANDS[band])) for band in BAND_ORDER}
    return ans


def cid(split: str, idx: int, band: str, mu: float) -> str:
    return f"pv_{split.lower()}_t0_r{idx:02d}_s{ROOT_BASE_SEED + idx}_{band.lower()}_mu{mu:.6f}"


def context(split: str, idx: int, band: str, mu: float) -> dict:
    return {
        "context_id": cid(split, idx, band, mu), "split": split, "task": 0,
        "root_id": f"pv_{split.lower()}_t0_root{idx:02d}_s{ROOT_BASE_SEED + idx}",
        "source_root_id": f"p5s0c_{split.lower()}_t0_root{idx:02d}_s{ROOT_BASE_SEED + idx}",
        "root_index": idx, "root_seed": ROOT_BASE_SEED + idx,
        "friction_band": band, "mu_GT": mu,
        "outcome_selection": "fixed before outcomes; no outcome-dependent inclusion",
        "state_selection": "fresh legal task0 reset under authoritative root seed",
        "eligible": 1,
    }


def main(supersede_preoutcome_v1: bool = False):
    if (ROOT / "TASK0_CONTEXT_SPLIT_MANIFEST.json").exists() and not supersede_preoutcome_v1:
        print(json.dumps({"status": "ALREADY_FROZEN", "sha256": sha256(ROOT / "TASK0_CONTEXT_SPLIT_MANIFEST.json")}, indent=2))
        return
    if supersede_preoutcome_v1:
        forbidden = [
            OUT / "collection_train_new", OUT / "collection_test_frozen",
            OUT / "checkpoints", OUT / "TASK0_CONTEXT_CHECKPOINT_MANIFEST.json",
        ]
        if any(p.exists() for p in forbidden):
            raise RuntimeError("cannot supersede split after collection or new checkpoint creation")
    OUT.mkdir(parents=True, exist_ok=True)
    existing = list(csv.DictReader((SOURCE / "PROSPECTIVE_CONTEXT_MANIFEST.csv").open()))
    existing = [r for r in existing if r["split"] == "TRAIN" and int(r["task"]) == 0]
    if len(existing) != 18 or len({r["context_id"] for r in existing}) != 18:
        raise RuntimeError("authoritative existing task0 TRAIN is not 18 contexts")
    if len({r["root_id"] for r in existing}) != 6:
        raise RuntimeError("authoritative existing task0 TRAIN is not six root clusters")

    frictions = friction_plan(83)
    # Validate that the deterministic extension reproduces the authoritative
    # task0 values for roots 0--5 before using roots 12+.
    for r in existing:
        idx = int(r["root_index"]); band = r["friction_band"]
        if abs(float(r["mu_GT"]) - frictions[idx][band]) > 1e-12:
            raise RuntimeError(f"friction extension fails authoritative prefix at {idx}/{band}")

    # Each new context gets a distinct simulator root seed.  Cycling the
    # friction band keeps the task/object family fixed while avoiding three
    # near-duplicate visual snapshots from one root realization.
    additions = []
    for offset, idx in enumerate(range(12, 74)):
        band = BAND_ORDER[offset % len(BAND_ORDER)]
        additions.append(context("TRAIN", idx, band, frictions[idx][band]))
    if len(additions) != 62:
        raise RuntimeError("new TRAIN additions must be 62")

    test_bands = ["LOW", "MID", "HIGH", "LOW", "MID", "HIGH", "LOW", "MID", "HIGH", "LOW"]
    test = [context("TEST", idx, band, frictions[idx][band]) for idx, band in zip(range(74, 84), test_bands)]
    if len(test) != 10:
        raise RuntimeError("TEST must contain ten independent roots/contexts")

    existing_ids = sorted(r["context_id"] for r in existing)
    add_ids = [r["context_id"] for r in additions]
    s18 = existing_ids
    s30 = s18 + add_ids[:12]
    s50 = s30 + add_ids[12:32]
    s80 = s50 + add_ids[32:62]
    if [len(s18), len(s30), len(s50), len(s80)] != [18, 30, 50, 80]:
        raise RuntimeError("learning-curve sizes wrong")
    if not (set(s18) < set(s30) < set(s50) < set(s80)):
        raise RuntimeError("nested learning-curve invariant failed")

    rng = np.random.default_rng(NEW_FORCE_SEED)
    train_cells, train_runs, train_target = [], [], {}
    for c in additions:
        specs = []
        for stratum in range(5):
            lo = 3.0 + 2.0 * stratum / 5
            hi = 3.0 + 2.0 * (stratum + 1) / 5
            force = float(rng.uniform(lo, hi))
            train_cells.append({
                **c, "stratum_index": stratum, "stratum_low_N": lo,
                "stratum_high_N": hi, "force_N": force,
                "force_sampling": "uniform_continuous_one_draw_per_equal_width_stratum",
                "global_rng_seed": NEW_FORCE_SEED, "frozen_before_outcomes": 1,
            })
            for rep in [1, 2]:
                label = f"TRAIN_S{stratum}_F{force:.8f}_R{rep}"
                spec = {"force_N": force, "repeat_index": rep, "branch_label": label}
                specs.append(spec)
                train_runs.append({**c, "stratum_index": stratum, "stratum_low_N": lo,
                                   "stratum_high_N": hi, "force_N": force,
                                   "repeat": rep, "branch_label": label,
                                   "scientific_retry": 0})
        train_target[c["context_id"]] = specs

    test_runs, test_target = [], {}
    for c in test:
        specs = []
        for force in TEST_FORCES:
            for rep in range(1, 6):
                label = f"TEST_F{force:.2f}_R{rep}"
                spec = {"force_N": force, "repeat_index": rep, "branch_label": label}
                specs.append(spec)
                test_runs.append({**c, "force_N": force, "repeat": rep,
                                  "branch_label": label, "scientific_retry": 0})
        test_target[c["context_id"]] = specs

    write_json(OUT / "TASK0_NEW_TRAIN_CONTEXTS.json", additions)
    write_csv(OUT / "TASK0_NEW_TRAIN_FORCE_MANIFEST.csv", train_cells)
    write_csv(OUT / "TASK0_NEW_TRAIN_RUN_MANIFEST.csv", train_runs)
    write_json(OUT / "TASK0_NEW_TRAIN_TARGET_MANIFEST.json", {
        "manifest_name": "TASK0_NEW_TRAIN_TARGET_MANIFEST", "split": "TRAIN",
        "contexts": train_target, "expected_contexts": 62,
        "expected_force_cells": 310, "expected_branches": 620,
        "branch_state": "strict pre-probe last-hold snapshot restored before every branch",
        "scientific_retry": 0,
    })
    write_json(OUT / "TASK0_FROZEN_TEST_CONTEXTS.json", test)
    write_csv(OUT / "TASK0_FROZEN_TEST_RUN_MANIFEST.csv", test_runs)
    write_json(OUT / "TASK0_FROZEN_TEST_TARGET_MANIFEST.json", {
        "manifest_name": "TASK0_FROZEN_TEST_TARGET_MANIFEST", "split": "TEST",
        "contexts": test_target, "expected_contexts": 10,
        "expected_force_cells": 90, "expected_branches": 450,
        "branch_state": "strict pre-probe last-hold snapshot restored before every branch",
        "scientific_retry": 0,
    })

    manifest = {
        "status": "FROZEN_BEFORE_NEW_COLLECTION_OUTCOMES_OR_NEW_MODEL_TRAINING",
        "split_version": 2,
        "supersedes_preoutcome_v1": str(OUT / "split_v1_preoutcome_invalidated"),
        "v1_invalidation_reason": "v1 used 62 contexts on only 21 new simulator roots; invalidated before any new outcome or checkpoint so each v2 addition has a unique root seed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "same task0; same alphabet_soup object/task distribution; held-out execution roots/contexts",
        "not_cross_task": True, "not_cross_object": True, "unseen_task_claim": False,
        "existing_authoritative_TRAIN": {
            "contexts": 18, "simulator_root_clusters": 6,
            "context_ids": existing_ids,
            "source_context_manifest": str(SOURCE / "PROSPECTIVE_CONTEXT_MANIFEST.csv"),
            "source_context_manifest_sha256": sha256(SOURCE / "PROSPECTIVE_CONTEXT_MANIFEST.csv"),
        },
        "excluded_previously_inspected": {
            "root_indices": [6, 7], "root_seeds": [5106, 5107],
            "reason": "already inspected retrospective roots; forbidden from new TEST",
        },
        "excluded_historical_original_TEST_indices": {
            "root_indices": [8, 9, 10, 11],
            "reason": "avoid reusing historically designated TEST roots as newly unseen contexts",
        },
        "new_TRAIN": {
            "contexts": 62, "new_simulator_root_clusters": 62,
            "one_unique_simulator_root_seed_per_new_context": True,
            "root_index_range": [12, 73], "root_seed_range": [5112, 5173],
            "target_manifest": str(OUT / "TASK0_NEW_TRAIN_TARGET_MANIFEST.json"),
            "protocol": "5 equal-width force strata over 3.00--5.00 N x 2 repeats",
            "force_rng_seed": NEW_FORCE_SEED,
        },
        "nested_TRAIN_context_ids": {"N18": s18, "N30": s30, "N50": s50, "N80": s80},
        "learning_curve_N_frozen_before_TEST": [18, 30, 50, 80],
        "new_frozen_TEST": {
            "contexts": 10, "simulator_root_clusters": 10,
            "one_unique_simulator_root_seed_per_context": True,
            "root_index_range": [74, 83], "root_seed_range": [5174, 5183],
            "context_ids": [x["context_id"] for x in test],
            "friction_band_counts": {"LOW": 4, "MID": 3, "HIGH": 3},
            "forces_N": TEST_FORCES, "repeats_per_force": 5,
            "force_cells": 90, "branches": 450,
            "target_manifest": str(OUT / "TASK0_FROZEN_TEST_TARGET_MANIFEST.json"),
            "atomic_commit_after_collection_required": True,
            "forbidden_uses": ["PCA fitting", "model selection", "calibration", "architecture/loss/lambda tuning", "choosing learning-curve N"],
        },
        "visual_preprocessing": {
            "raw_feature_dimension": 4096, "PCA_dimension_fixed": 17,
            "PCA_fit_separately_on_corresponding_TRAIN_subset": True,
            "TEST_transform_only": True, "pre_PCA_feature_retained": True,
            "feature_spec": str(VISUAL_SPEC), "feature_spec_sha256": sha256(VISUAL_SPEC),
        },
        "models_each_N": ["Base", "Residual", "Full Visual", "Joint-NoVisual", "Visual Joint"],
        "seeds": [0, 1, 2], "epochs": 80, "optimizer": "AdamW",
        "architecture_loss_lambda_optimizer_selection_changes_forbidden": True,
        "probe_reopened": False, "VLA_frozen": True, "visual_encoder_frozen": True,
        "force_success_restore_semantics_unchanged": True,
        "source_hashes": {
            str(RUNNER): sha256(RUNNER), str(COLLECTOR): sha256(COLLECTOR),
            str(VISUAL_SPEC): sha256(VISUAL_SPEC),
        },
        "friction_sampler": {
            "seed": FRICTION_SEED, "bands": BANDS,
            "construction": "deterministic continuation of authoritative task0 prefix; prefix roots0--5 verified exactly",
        },
        "outcomes_read_by_prepare": False,
    }
    write_json(ROOT / "TASK0_CONTEXT_SPLIT_MANIFEST.json", manifest)
    hashes = {}
    for path in [
        ROOT / "TASK0_CONTEXT_SPLIT_MANIFEST.json",
        OUT / "TASK0_NEW_TRAIN_CONTEXTS.json", OUT / "TASK0_NEW_TRAIN_FORCE_MANIFEST.csv",
        OUT / "TASK0_NEW_TRAIN_RUN_MANIFEST.csv", OUT / "TASK0_NEW_TRAIN_TARGET_MANIFEST.json",
        OUT / "TASK0_FROZEN_TEST_CONTEXTS.json", OUT / "TASK0_FROZEN_TEST_RUN_MANIFEST.csv",
        OUT / "TASK0_FROZEN_TEST_TARGET_MANIFEST.json",
    ]:
        hashes[str(path)] = sha256(path)
    write_json(OUT / "TASK0_CONTEXT_SPLIT_FREEZE_SHA256.json", {
        "status": "FROZEN", "hashes": hashes,
        "outcomes_read": False, "new_models_trained": False,
    })
    print(json.dumps({"status": manifest["status"], "TRAIN_sizes": [18,30,50,80],
                      "new_TRAIN_branches": 620, "new_TEST_branches": 450,
                      "manifest_sha256": sha256(ROOT / "TASK0_CONTEXT_SPLIT_MANIFEST.json")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--supersede-preoutcome-v1", action="store_true")
    args = ap.parse_args()
    main(args.supersede_preoutcome_v1)
