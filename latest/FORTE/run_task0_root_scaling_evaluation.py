#!/usr/bin/env python3
"""Evaluate the frozen Task 0 root-scaling checkpoints on committed Task 0 TEST.

This is an explicitly scoped Task 0-only result.  It reuses the frozen
training/evaluation implementation, but does not require or imply Task 5
collection, checkpoints, evaluation, or an across-task conclusion.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch


ROOT = Path("/home/exouser/FORTE")
AUTHORITATIVE = ROOT / "root_scaling_20260831"
OUT = AUTHORITATIVE / "task0_diagnostic"
SOURCE_FILE = ROOT / "root_scaling_learning_curve.py"


def load_frozen_module():
    spec = importlib.util.spec_from_file_location("root_scaling_learning_curve_task0_eval", SOURCE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    mod = load_frozen_module()
    mod.OUT = OUT
    mod.TASKS = [0]

    frozen_path = AUTHORITATIVE / "ROOT_SCALING_FREEZE_SHA256.json"
    train_manifest_path = AUTHORITATIVE / "ROOT_SCALING_TRAIN_MANIFEST.json"

    def verify_task0_freeze():
        freeze = json.loads(frozen_path.read_text())
        for raw, expected in freeze["hashes"].items():
            if mod.sha256(Path(raw)) != expected:
                raise RuntimeError(f"frozen input changed: {raw}")
        return freeze, json.loads(train_manifest_path.read_text())

    mod.verify_freeze = verify_task0_freeze

    checkpoint_manifest_path = OUT / "ROOT_SCALING_CHECKPOINT_MANIFEST.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
    if (checkpoint_manifest.get("status") != "TASK0_DIAGNOSTIC_ONLY" or
            checkpoint_manifest.get("checkpoint_count") != 36 or
            checkpoint_manifest.get("task_scope") != [0] or
            checkpoint_manifest.get("TEST_used") is not False):
        raise RuntimeError("Task 0 checkpoint manifest is not the frozen 36-checkpoint diagnostic")

    model_hashes_path = OUT / "ROOT_SCALING_MODEL_HASHES.csv"
    model_hashes = pd.read_csv(model_hashes_path)
    if len(model_hashes) != 36 or set(model_hashes.task.astype(int)) != {0}:
        raise RuntimeError("Task 0 model hash table identity/count mismatch")
    for row in model_hashes.itertuples(index=False):
        checkpoint = Path(str(row.checkpoint))
        if mod.sha256(checkpoint) != str(row.checkpoint_sha256):
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")

    commit_path = AUTHORITATIVE / "TASK0_TEST_COLLECTION_COMMIT.json"
    commit = json.loads(commit_path.read_text())
    if commit.get("status") != "ATOMICALLY_COMMITTED_UNTOUCHED_TEST" or commit.get("task") != 0:
        raise RuntimeError("Task 0 TEST is not atomically committed")

    original_load_test = mod.load_test

    def load_authoritative_task0_test(task, tpi, pca):
        if task != 0:
            raise RuntimeError("Task 0-only evaluator received another task")
        previous = mod.OUT
        mod.OUT = AUTHORITATIVE
        try:
            return original_load_test(task, tpi, pca)
        finally:
            mod.OUT = previous

    mod.load_test = load_authoritative_task0_test

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tpi, cf, full = mod.modules()
    device = torch.device("cpu")
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    summary, safety, gaps, perroots = [], [], [], []

    for n in mod.LEVELS:
        _, train_branches, train_cmap, train_traces, _, _, _, norm, pca, _ = mod.load_train(0, n, tpi, cf)
        _, test_branches, test_cmap, templates = mod.load_test(0, tpi, pca)
        models = mod.load_models(0, n, tpi, full, device)
        test_forces = {cid: mod.FORCES_DENSE for cid in templates}
        pred = mod.predict(models, test_cmap, templates, norm, tpi, cf, test_forces)
        real = mod.real_cells(test_branches)
        pr = mod.per_root(0, n, pred, real, test_cmap)
        perroots.append(pr)

        train_templates = {
            cid: next(x for x in train_traces if x.context_id == cid)
            for cid in train_cmap
        }
        train_forces = {
            cid: sorted(train_branches[
                train_branches.context_id.astype(str) == cid
            ].requested_force_N.astype(float).unique())
            for cid in train_templates
        }
        train_pred = mod.predict(models, train_cmap, train_templates, norm, tpi, cf, train_forces)
        train_real = mod.real_cells(train_branches)
        for model in mod.MODELS:
            for aggregation in ["ENSEMBLE"] + [f"SEED_{seed}" for seed in mod.SEEDS]:
                main_row, safety_row, gap_row = mod.aggregate_one(
                    0, n, model, aggregation, pr, pred, real, train_pred, train_real
                )
                summary.append(main_row)
                safety.append(safety_row)
                gaps.append(gap_row)

        pred.to_csv(OUT / f"predictions/TASK0_S{n}_TEST_DENSE_PREDICTIONS.csv", index=False)
        print(f"[task0-evaluate] S{n} complete", flush=True)

    learning_curve = pd.DataFrame(summary)
    safety_frame = pd.DataFrame(safety)
    gap_frame = pd.DataFrame(gaps)
    main_numeric = [
        c for c in learning_curve.columns
        if c not in {"task", "N_root", "model", "aggregation"}
        and pd.api.types.is_numeric_dtype(learning_curve[c])
    ]
    safety_numeric = [
        c for c in safety_frame.columns
        if c not in {"task", "N_root", "model", "aggregation"}
        and pd.api.types.is_numeric_dtype(safety_frame[c])
    ]
    gap_numeric = [
        c for c in gap_frame.columns
        if c not in {"task", "N_root", "model", "aggregation"}
        and pd.api.types.is_numeric_dtype(gap_frame[c])
    ]
    learning_curve = mod.add_seed_mean_std(
        learning_curve, ["task", "N_root", "model"], main_numeric
    )
    safety_frame = mod.add_seed_mean_std(
        safety_frame, ["task", "N_root", "model"], safety_numeric
    )
    gap_frame = mod.add_seed_mean_std(
        gap_frame, ["task", "N_root", "model"], gap_numeric
    )

    mod.write_csv(OUT / "ROOT_SCALING_LEARNING_CURVE.csv", learning_curve)
    mod.write_csv(OUT / "ROOT_SCALING_PER_ROOT_METRICS.csv", pd.concat(perroots, ignore_index=True))
    mod.write_csv(OUT / "ROOT_SCALING_SAFETY_METRICS.csv", safety_frame)
    mod.write_csv(OUT / "ROOT_SCALING_TRAIN_TEST_GAP.csv", gap_frame)
    mod.write_json(OUT / "TASK0_ROOT_SCALING_EVALUATION_COMPLETE.json", {
        "status": "TASK0_COMPLETE_NO_TEST_SELECTION_CALIBRATION_OR_TUNING",
        "scope": "Task 0 only; no Task 5 or across-task conclusion",
        "task": 0,
        "root_levels": mod.LEVELS,
        "test_roots": 10,
        "checkpoint_count": 36,
        "learning_curve_rows": len(learning_curve),
        "per_root_rows": sum(len(frame) for frame in perroots),
        "safety_rows": len(safety_frame),
        "gap_rows": len(gap_frame),
        "checkpoint_manifest_sha256": mod.sha256(checkpoint_manifest_path),
        "model_hash_csv_sha256": mod.sha256(model_hashes_path),
        "task0_test_commit_sha256": mod.sha256(commit_path),
        "frozen_evaluator_sha256": mod.sha256(SOURCE_FILE),
        "task5_skipped": True,
    })
    print(learning_curve[learning_curve.aggregation == "SEED_MEAN"].to_string(index=False))


if __name__ == "__main__":
    main()
