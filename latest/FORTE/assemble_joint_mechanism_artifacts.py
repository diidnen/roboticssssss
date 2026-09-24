#!/usr/bin/env python3
"""Assemble normalized taskwise/pooled mechanism comparison CSVs.

Missing prospective DEV artifacts are skipped until the collector and frozen
evaluation complete; rerunning this assembler appends them without changing
any model or metric.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "joint_mechanism_20260831"
FROZEN = {
    0: ROOT / "task0_visual_context_early_20260831_025000",
    1: ROOT / "taskwise_visual_context_20260831_074000/task1_train_frozen",
    5: ROOT / "taskwise_visual_context_20260831_074000/task5_train_frozen",
    6: ROOT / "taskwise_visual_context_20260831_074000/task6_train_frozen",
}
CV_SOURCE = {
    **FROZEN,
    0: ROOT / "task0_visual_generalization_20260831_040609",
}
JNV = {t: OUT / f"task{t}_joint_novisual" for t in [0, 1, 5, 6]}
MODEL_LABEL = {
    "PROSPECTIVE_BASE_FEAS": "Base",
    "VISUAL_INTERCEPT_RESIDUAL": "Residual",
    "VISUAL_CONTEXT_FULL_FEAS": "Full",
    "VISUAL_CONTEXT_JOINT": "Visual Joint",
    "JOINT_NOVISUAL": "Joint-NoVisual",
    "POOLED_BASE": "Pooled Base",
    "POOLED_JOINT_NOVISUAL": "Pooled Joint-NoVisual",
}


def dev_root_count(task: int) -> int:
    manifest = pd.read_csv(
        "/home/exouser/Tabero/analysis/results/"
        "gnp_style_visual_context_prospective_20260831_011000/"
        "PROSPECTIVE_CONTEXT_MANIFEST.csv"
    )
    q = manifest[(manifest.task == task) & (manifest.split == "DEV")]
    return int(q.root_id.astype(str).nunique())


def records_train(task: int) -> list[dict]:
    visual = pd.read_csv(FROZEN[task] / f"TASK{task}_TRAIN_IN_SAMPLE_METRICS.csv")
    jnv = pd.read_csv(JNV[task] / f"TASK{task}_JOINT_NOVISUAL_TRAIN_METRICS.csv")
    d = pd.concat([visual, jnv], ignore_index=True)
    rows = []
    for model, q in d.groupby("variant"):
        rows.append({
            "task": task, "evaluation_scope": "TRAIN_IN_SAMPLE", "model": MODEL_LABEL[model],
            "raw_model": model, "contexts": 18, "roots": 6, "branches": 180,
            "seeds": 3, "BCE": float(q.bce_nll.mean()), "NLL": float(q.bce_nll.mean()),
            "probability_MAE": np.nan, "Brier": float(q.brier.mean()),
            "frontier_MAE_N": np.nan, "under_force_rate": np.nan,
            "mean_local_monotonicity": np.nan, "context_monotonic_rate": np.nan,
            "selection_eligible": False,
        })
    return rows


def records_cv(task: int) -> list[dict]:
    visual = pd.read_csv(CV_SOURCE[task] / f"TASK{task}_GROUP_HELDOUT_CV.csv")
    visual = visual[visual.fold.astype(str) == "__MEAN__"].copy()
    enriched = pd.read_csv(JNV[task] / f"TASK{task}_JOINT_NOVISUAL_ROOT_CV.csv")
    enriched = enriched[enriched.fold.astype(str) == "__MEAN__"].set_index("model")
    rows = []
    for r in visual.itertuples(index=False):
        extra = enriched.loc[r.model] if r.model in enriched.index else None
        rows.append({
            "task": task, "evaluation_scope": "RETROSPECTIVE_TRAIN_ROOT_HELDOUT_CV",
            "model": MODEL_LABEL[r.model], "raw_model": r.model,
            "contexts": int(r.heldout_contexts), "roots": int(r.heldout_roots),
            "branches": int(r.heldout_branches), "seeds": 1,
            "BCE": float(r.BCE), "NLL": float(r.BCE),
            "probability_MAE": float(extra.probability_MAE) if extra is not None else np.nan,
            "Brier": float(r.Brier),
            "frontier_MAE_N": float(extra.frontier_MAE_N) if extra is not None else np.nan,
            "under_force_rate": float(extra.under_force_rate) if extra is not None else np.nan,
            "mean_local_monotonicity": float(extra.mean_local_monotonicity) if extra is not None else np.nan,
            "context_monotonic_rate": float(extra.context_monotonic_rate) if extra is not None else np.nan,
            "selection_eligible": False,
        })
    r = enriched.loc["JOINT_NOVISUAL"]
    rows.append({
        "task": task, "evaluation_scope": "RETROSPECTIVE_TRAIN_ROOT_HELDOUT_CV",
        "model": "Joint-NoVisual", "raw_model": "JOINT_NOVISUAL",
        "contexts": int(r.heldout_contexts), "roots": int(r.heldout_roots),
        "branches": int(r.heldout_branches), "seeds": 1,
        **{k: float(r[k]) for k in ["BCE", "NLL", "probability_MAE", "Brier", "frontier_MAE_N",
                                         "under_force_rate", "mean_local_monotonicity", "context_monotonic_rate"]},
        "selection_eligible": False,
    })
    return rows


def records_dev(task: int) -> list[dict]:
    if task == 0:
        p = ROOT / "TASK0_JOINT_NOVISUAL_DIAGNOSTIC.csv"
    else:
        p = JNV[task] / f"TASK{task}_JOINT_NOVISUAL_DEV_SUMMARY.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p)
    if "aggregation" in d:
        d = d[d.aggregation == "ENSEMBLE"]
    rows = []
    for r in d.to_dict("records"):
        model = r["model"]
        rows.append({
            "task": task, "evaluation_scope": "RETROSPECTIVE_PROSPECTIVE_DEV_ROOTS",
            "model": MODEL_LABEL.get(model, model), "raw_model": model,
            "contexts": int(r.get("DEV_contexts", np.nan)),
            "roots": dev_root_count(task),
            "branches": int(r.get("physical_repeats", r.get("DEV_branches", 0))),
            "seeds": 3, "BCE": float(r["NLL"]), "NLL": float(r["NLL"]),
            "probability_MAE": float(r["probability_MAE"]), "Brier": float(r["Brier"]),
            "frontier_MAE_N": float(r["frontier_MAE_N"]),
            "under_force_rate": float(r["under_force_rate"]),
            "mean_local_monotonicity": float(r.get("context_monotonic", np.nan)),
            "context_monotonic_rate": float(r.get("context_monotonic", np.nan)),
            "selection_eligible": False,
        })
    return rows


def add_gaps(rows: list[dict]) -> list[dict]:
    train = {(r["task"], r["model"]): r["BCE"] for r in rows if r["evaluation_scope"] == "TRAIN_IN_SAMPLE"}
    for r in rows:
        tr = train.get((r["task"], r["model"]))
        r["TRAIN_BCE_reference"] = tr
        r["heldout_minus_TRAIN_NLL_gap"] = (r["NLL"] - tr) if tr is not None and r["evaluation_scope"] != "TRAIN_IN_SAMPLE" else np.nan
        r["heldout_to_TRAIN_NLL_ratio"] = (r["NLL"] / tr) if tr and r["evaluation_scope"] != "TRAIN_IN_SAMPLE" else np.nan
    return rows


def pooled_records() -> list[dict]:
    p = OUT / "pooled_matched/POOLED_MATCHED_ROOT_CV.csv"
    d = pd.read_csv(p)
    d = d[d.fold.astype(str) == "__MEAN__"]
    rows = []
    for r in d.to_dict("records"):
        rows.append({
            "task": int(r["task"]), "evaluation_scope": "MATCHED_POOLED_TRAIN_ROOT_HELDOUT_CV",
            "model": MODEL_LABEL[r["model"]], "raw_model": r["model"],
            "train_contexts": int(r["pooled_train_contexts"]), "train_roots": int(r["pooled_train_roots"]),
            "heldout_contexts": int(r["heldout_contexts"]), "heldout_roots": int(r["heldout_roots"]),
            "BCE": float(r["BCE"]), "NLL": float(r["NLL"]),
            "probability_MAE": float(r["probability_MAE"]), "Brier": float(r["Brier"]),
            "frontier_MAE_N": float(r["frontier_MAE_N"]), "under_force_rate": float(r["under_force_rate"]),
            "mean_local_monotonicity": float(r["mean_local_monotonicity"]),
            "context_monotonic_rate": float(r["context_monotonic_rate"]),
            "implementation_confounded": False, "selection_eligible": False,
        })
    dev = OUT / "pooled_matched/POOLED_MATCHED_DEV_TASKWISE_SUMMARY.csv"
    if dev.exists():
        for r in pd.read_csv(dev).to_dict("records"):
            rows.append({
                "task": int(r["task"]), "evaluation_scope": "MATCHED_POOLED_PROSPECTIVE_DEV_ROOTS",
                "model": MODEL_LABEL[r["model"]], "raw_model": r["model"],
                "train_contexts": 72, "train_roots": 24,
                "heldout_contexts": int(r["DEV_contexts"]), "heldout_roots": dev_root_count(int(r["task"])),
                "BCE": float(r["NLL"]), "NLL": float(r["NLL"]),
                "probability_MAE": float(r["probability_MAE"]), "Brier": float(r["Brier"]),
                "frontier_MAE_N": float(r["frontier_MAE_N"]), "under_force_rate": float(r["under_force_rate"]),
                "mean_local_monotonicity": float(r.get("context_monotonic", np.nan)),
                "context_monotonic_rate": float(r.get("context_monotonic", np.nan)),
                "implementation_confounded": False, "selection_eligible": False,
            })
    original = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000/EVALUATION_SUMMARIES.json")
    if original.exists():
        obj = json.loads(original.read_text())
        for r in obj.get("summaries", []):
            rows.append({
                "task": "ALL_ONLY_ORIGINAL_PIPELINE", "evaluation_scope": "ORIGINAL_POOLED_VISUAL_DEV_AGGREGATE",
                "model": MODEL_LABEL.get(r.get("model"), r.get("model")), "raw_model": r.get("model"),
                "train_contexts": 72, "train_roots": 24,
                "heldout_contexts": np.nan, "heldout_roots": np.nan,
                "BCE": r.get("NLL"), "NLL": r.get("NLL"),
                "probability_MAE": r.get("probability_MAE"), "Brier": r.get("Brier"),
                "frontier_MAE_N": r.get("frontier_MAE_N"), "under_force_rate": r.get("under_force_rate"),
                "mean_local_monotonicity": r.get("context_monotonic", np.nan),
                "context_monotonic_rate": r.get("context_monotonic", np.nan),
                "implementation_confounded": True, "selection_eligible": False,
            })
    original_taskwise = OUT / "original_pooled_visual_taskwise/ORIGINAL_POOLED_VISUAL_TASKWISE.csv"
    if original_taskwise.exists():
        for r in pd.read_csv(original_taskwise).to_dict("records"):
            task = int(r["task"])
            rows.append({
                "task": task, "evaluation_scope": "ORIGINAL_POOLED_VISUAL_DEV_TASKWISE",
                "model": MODEL_LABEL.get(r["model"], r["model"]), "raw_model": r["model"],
                "train_contexts": 72, "train_roots": 24,
                "heldout_contexts": int(r["DEV_contexts"]), "heldout_roots": dev_root_count(task),
                "BCE": float(r["NLL"]), "NLL": float(r["NLL"]),
                "probability_MAE": float(r["probability_MAE"]), "Brier": float(r["Brier"]),
                "frontier_MAE_N": float(r["frontier_MAE_N"]),
                "under_force_rate": float(r["under_force_or_missing_rate"]),
                "mean_local_monotonicity": float(r["monotonic_context_rate"]),
                "context_monotonic_rate": float(r["monotonic_context_rate"]),
                "implementation_confounded": True, "selection_eligible": False,
            })
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    all_rows = []
    for task in [0, 1, 5, 6]:
        rows = add_gaps(records_train(task) + records_cv(task) + records_dev(task))
        pd.DataFrame(rows).to_csv(OUT / f"TASK{task}_JOINT_NOVISUAL.csv", index=False)
        all_rows.extend([r for r in rows if r["evaluation_scope"] == "RETROSPECTIVE_TRAIN_ROOT_HELDOUT_CV"])
    pd.DataFrame(all_rows).to_csv(OUT / "TASKWISE_ROOT_HELDOUT_COMPARISON.csv", index=False)
    pd.DataFrame(pooled_records()).to_csv(OUT / "POOLED_JOINT_COMPARISON.csv", index=False)
    print(json.dumps({"status": "ASSEMBLED", "taskwise_root_rows": len(all_rows),
                      "pooled_rows": len(pooled_records())}, indent=2))


if __name__ == "__main__":
    main()
