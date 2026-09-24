#!/usr/bin/env python3
"""Independent arithmetic and data-grain validation for mechanism artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "joint_mechanism_20260831"


def close(a, b, tol=1e-8):
    if pd.isna(a) and pd.isna(b):
        return True
    return abs(float(a)-float(b)) <= tol


def validate_fold_means(path: Path, keys: list[str]) -> list[dict]:
    d = pd.read_csv(path)
    folds = d[d.fold.astype(str) != "__MEAN__"]
    means = d[d.fold.astype(str) == "__MEAN__"]
    group = ["model"] + (["task"] if "task" in d else [])
    rows = []
    for r in means.to_dict("records"):
        q = folds[folds.model == r["model"]]
        if "task" in d:
            q = q[q.task == r["task"]]
        checks = {k: close(q[k].mean(), r[k]) for k in keys}
        rows.append({"path": str(path), **{k: r[k] for k in group},
                     "all_metrics_match": all(checks.values()), "checks": checks})
    return rows


def validate_task0_dev() -> list[dict]:
    real = pd.read_csv(ROOT / "task0_visual_generalization_20260831_040609/TASK0_REAL_DEV_CURVES.csv")
    pred = pd.read_csv(ROOT / "task0_joint_novisual_diagnostic_20260831/TASK0_JOINT_NOVISUAL_RETROSPECTIVE_PREDICTIONS.csv")
    reported = pd.read_csv(ROOT / "TASK0_JOINT_NOVISUAL_DIAGNOSTIC.csv").set_index("model")
    rows = []
    for model in reported.index:
        q = real.merge(pred[pred.model == model], on=["context_id", "force_N"])
        p, y = q.ensemble_prob.to_numpy(), q.p_real.to_numpy()
        recomputed = {
            "probability_MAE": float(np.mean(np.abs(p-y))),
            "Brier": float(np.mean((p-y)**2)),
            "NLL": float(-np.mean(y*np.log(np.clip(p, 1e-8, 1)) + (1-y)*np.log(np.clip(1-p, 1e-8, 1)))),
        }
        checks = {k: close(v, reported.loc[model, k]) for k, v in recomputed.items()}
        rows.append({"model": model, "cells": len(q), "recomputed": recomputed,
                     "all_metrics_match": all(checks.values()), "checks": checks})
    return rows


def probability_triplet(frame: pd.DataFrame, pred_col: str, real_col: str) -> dict:
    p = np.clip(frame[pred_col].to_numpy(dtype=float), 1e-8, 1-1e-8)
    y = frame[real_col].to_numpy(dtype=float)
    return {
        "probability_MAE": float(np.mean(np.abs(p-y))),
        "Brier": float(np.mean((p-y)**2)),
        "NLL": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
    }


def validate_taskwise_dev() -> list[dict]:
    rows = []
    for task in [1, 5, 6]:
        summary_path = OUT / f"task{task}_joint_novisual/TASK{task}_JOINT_NOVISUAL_DEV_SUMMARY.csv"
        pred_path = OUT / f"task{task}_joint_novisual/TASK{task}_JOINT_NOVISUAL_DEV_PREDICTIONS.csv"
        real_path = ROOT / f"taskwise_visual_context_20260831_074000/task{task}_dev_validation/TASK{task}_REAL_DEV_CURVES.csv"
        if not all(p.exists() for p in [summary_path, pred_path, real_path]):
            continue
        reported = pd.read_csv(summary_path).set_index("model")
        pred = pd.read_csv(pred_path)
        real = pd.read_csv(real_path)
        for model in reported.index:
            q = real.merge(pred[pred.model == model], on=["context_id", "force_N"])
            recomputed = probability_triplet(q, "ensemble_prob", "p_real")
            checks = {k: close(v, reported.loc[model, k]) for k, v in recomputed.items()}
            rows.append({"task": task, "model": model, "cells": len(q), "recomputed": recomputed,
                         "all_metrics_match": all(checks.values()), "checks": checks})
    return rows


def validate_matched_pooled_dev() -> list[dict]:
    summary_path = OUT / "pooled_matched/POOLED_MATCHED_DEV_TASKWISE_SUMMARY.csv"
    pred_path = OUT / "pooled_matched/POOLED_MATCHED_DEV_PREDICTIONS.csv"
    if not summary_path.exists() or not pred_path.exists():
        return []
    reported = pd.read_csv(summary_path).set_index(["task", "model"])
    pred = pd.read_csv(pred_path)
    rows = []
    for task, model in reported.index:
        if int(task) == 0:
            real_path = ROOT / "task0_visual_generalization_20260831_040609/TASK0_REAL_DEV_CURVES.csv"
        else:
            real_path = ROOT / f"taskwise_visual_context_20260831_074000/task{int(task)}_dev_validation/TASK{int(task)}_REAL_DEV_CURVES.csv"
        real = pd.read_csv(real_path)
        q = real.merge(pred[(pred.task == int(task)) & (pred.model == model)], on=["context_id", "force_N"])
        recomputed = probability_triplet(q, "ensemble_prob", "p_real")
        checks = {k: close(v, reported.loc[(task, model), k]) for k, v in recomputed.items()}
        rows.append({"task": int(task), "model": model, "cells": len(q), "recomputed": recomputed,
                     "all_metrics_match": all(checks.values()), "checks": checks})
    return rows


def validate_original_pooled_dev() -> list[dict]:
    base = OUT / "original_pooled_visual_taskwise"
    summary_path = base / "ORIGINAL_POOLED_VISUAL_TASKWISE.csv"
    cell_path = base / "ORIGINAL_POOLED_VISUAL_TASKWISE_CELL_PREDICTIONS.csv"
    if not summary_path.exists() or not cell_path.exists():
        return []
    reported = pd.read_csv(summary_path).set_index(["task", "model"])
    cells = pd.read_csv(cell_path)
    rows = []
    for task, model in reported.index:
        q = cells[(cells.task == int(task)) & (cells.model == model)]
        recomputed = probability_triplet(q, "p_pred", "p_real")
        checks = {k: close(v, reported.loc[(task, model), k]) for k, v in recomputed.items()}
        rows.append({"task": int(task), "model": model, "cells": len(q), "recomputed": recomputed,
                     "all_metrics_match": all(checks.values()), "checks": checks})
    return rows


def data_quality() -> dict:
    audits = {
        0: json.loads((ROOT / "task0_visual_context_early_20260831_025000/TASK0_DATA_AUDIT.json").read_text()),
        1: json.loads((ROOT / "taskwise_visual_context_20260831_074000/task1_train_frozen/TASK1_DATA_AUDIT.json").read_text()),
        5: json.loads((ROOT / "taskwise_visual_context_20260831_074000/task5_train_frozen/TASK5_DATA_AUDIT.json").read_text()),
        6: json.loads((ROOT / "taskwise_visual_context_20260831_074000/task6_train_frozen/TASK6_DATA_AUDIT.json").read_text()),
    }
    return {
        "taskwise": {str(t): {
            "status": a["status"], "contexts": a["contexts"], "roots": a.get("roots", 6),
            "branches": a["branches"], "direct_labels": a.get("direct_labels", 180 if t == 0 else None),
            "reconstructed_labels": a.get("reconstructed_labels", 0),
            "corrected_physical_telemetry": a["corrected_physical_telemetry"],
            "visual_alignment": a["visual_alignment"],
        } for t, a in audits.items()},
        "all_counts_match_frozen_design": all(a["contexts"] == 18 and a["branches"] == 180 for a in audits.values()),
        "task1_material_label_caveat_retained": audits[1]["reconstructed_labels"] == 140,
        "task5_task6_direct_label_replications": audits[5]["direct_labels"] == 180 and audits[6]["direct_labels"] == 180,
    }


def main() -> None:
    metric_keys = ["BCE", "Brier"]
    root_checks = []
    for t in [0, 1, 5, 6]:
        root_checks += validate_fold_means(OUT / f"task{t}_joint_novisual/TASK{t}_JOINT_NOVISUAL_ROOT_CV.csv",
                                           metric_keys + ["probability_MAE", "frontier_MAE_N", "under_force_rate"])
    pooled_checks = validate_fold_means(OUT / "pooled_matched/POOLED_MATCHED_ROOT_CV.csv",
                                        metric_keys + ["probability_MAE", "frontier_MAE_N", "under_force_rate"])
    task0_dev = validate_task0_dev()
    taskwise_dev = validate_taskwise_dev()
    matched_pooled_dev = validate_matched_pooled_dev()
    original_pooled_dev = validate_original_pooled_dev()
    all_checks = root_checks + pooled_checks + task0_dev + taskwise_dev + matched_pooled_dev + original_pooled_dev
    result = {
        "status": "PASS" if all(r["all_metrics_match"] for r in all_checks) else "FAIL",
        "data_quality": data_quality(),
        "taskwise_root_cv_mean_recomputation": root_checks,
        "pooled_root_cv_mean_recomputation": pooled_checks,
        "task0_dev_probability_recomputation": task0_dev,
        "taskwise_dev_probability_recomputation": taskwise_dev,
        "matched_pooled_dev_probability_recomputation": matched_pooled_dev,
        "original_pooled_dev_probability_recomputation": original_pooled_dev,
        "known_limitations": [
            "TRAIN root-CV uses seed0 by frozen diagnostic design; full-data checkpoints use three seeds.",
            "TRAIN root-CV empirical frontier uses five random continuous forces with two repeats, not the denser prospective DEV grid.",
            "task1 has 140/180 reconstructed TRAIN labels; task5/task6 have direct labels.",
            "Original pooled visual implementation is excluded from matched arithmetic because its architecture/target/IE loss differ."
        ]
    }
    (OUT / "VALIDATION_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "root_checks": len(root_checks),
                      "pooled_checks": len(pooled_checks), "task0_dev_checks": len(task0_dev),
                      "taskwise_dev_checks": len(taskwise_dev),
                      "matched_pooled_dev_checks": len(matched_pooled_dev),
                      "original_pooled_dev_checks": len(original_pooled_dev)}, indent=2))


if __name__ == "__main__":
    main()
