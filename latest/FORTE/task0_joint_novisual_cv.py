#!/usr/bin/env python3
"""Retrospective task0 Joint-NoVisual TRAIN metrics and root-heldout CV.

Full-data Joint-NoVisual checkpoints predate the current mechanism protocol and
are reused without retraining.  CV retrains deterministic seed0 Base and
Joint-NoVisual solely to obtain aligned held-root probability/frontier metrics;
the recomputed Base BCE/Brier must exactly reproduce the existing frozen CV.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import task0_visual_context_early as early
import task0_visual_generalization as gen
import per_task_visual_context_early as taskwise
import taskwise_joint_novisual as jnv


ROOT = Path("/home/exouser/FORTE")
FROZEN = ROOT / "task0_visual_context_early_20260831_025000"
EXISTING_JNV = ROOT / "task0_joint_novisual_diagnostic_20260831"
EXISTING_CV = ROOT / "task0_visual_generalization_20260831_040609/TASK0_GROUP_HELDOUT_CV.csv"
OUT = ROOT / "joint_mechanism_20260831/task0_joint_novisual"


def population(scratch: Path):
    tpi = early.load_module("task0_jnv_cv_tpi", early.TPI_CODE)
    cf = early.load_module("task0_jnv_cv_cf", early.CF_CODE)
    full = early.load_module("task0_jnv_cv_full", early.FULL_CODE)
    _, branches, cmap, traces, meta, _ = early.audit_and_load(scratch, tpi)
    pairs = early.build_pairs(cf, traces, meta)
    segs, computed_norm = early.build_segments_and_norm(cf, tpi, traces)
    q = np.load(FROZEN / "TASK0_TRAIN_NORMALIZATION.npz")
    norm = tuple(q[k].astype(np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
    if not all(np.array_equal(a, b) for a, b in zip(norm, computed_norm)):
        raise RuntimeError("task0 normalization mismatch")
    return tpi, cf, full, branches, cmap, traces, meta, pairs, segs, norm


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = OUT / "_audit"
    scratch.mkdir(exist_ok=True)
    tpi, cf, full, branches, cmap, traces, meta, _, all_segs, norm = population(scratch)
    device = torch.device("cpu")

    train_rows, checkpoint_rows = [], []
    for seed in jnv.SEEDS:
        path = EXISTING_JNV / f"TASK0_JOINT_NOVISUAL_seed{seed}.pt"
        ck = torch.load(path, map_location=device, weights_only=False)
        base_ck, _, _ = full.load_base(tpi, seed, device)
        model = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
        model.load_state_dict(ck["state_dict"])
        logits = jnv.logits_for(model, traces, all_segs, norm, cmap, device)
        train_rows.append({**early.train_metrics(jnv.MODEL, seed, traces, logits), "task": 0})
        checkpoint_rows.append({"seed": seed, "path": str(path), "sha256": jnv.sha256(path),
                                "optimizer_steps": int(ck["optimizer_steps"]),
                                "physical_units": int(ck["physical_units"])})
    early.write_csv(OUT / "TASK0_JOINT_NOVISUAL_TRAIN_METRICS.csv", train_rows)
    early.write_json(OUT / "TASK0_PREEXISTING_JOINT_NOVISUAL_CHECKPOINT_AUDIT.json", {
        "status": "PASS_REUSED_WITHOUT_RETRAINING", "checkpoint_count": 3,
        "preexisting_before_current_protocol": True, "selection_use_forbidden": True,
        "checkpoints": checkpoint_rows,
    })

    roots = sorted({tr.root_id for tr in traces})
    folds = [set(roots[i::3]) for i in range(3)]
    existing = pd.read_csv(EXISTING_CV)
    reference = existing[(existing.model == "PROSPECTIVE_BASE_FEAS") & (existing.fold.astype(str) != "__MEAN__")]
    rows, prediction_rows = [], []
    for fold, held_roots in enumerate(folds):
        train = [tr for tr in traces if tr.root_id not in held_roots]
        held = [tr for tr in traces if tr.root_id in held_roots]
        train_meta = {tr.branch_id: meta[tr.branch_id] for tr in train}
        train_segs = {tr.branch_id: all_segs[tr.branch_id] for tr in train}
        fold_norm = early.build_segments_and_norm(cf, tpi, train)[1]
        pairs = taskwise.build_pairs(0, cf, train, train_meta)
        print(f"[task0 JNV CV] fold={fold} train={len(train)} held={len(held)}", flush=True)
        base, _, _ = early.train_base(full, train, train_segs, fold_norm, cmap, train_meta, device, 0)
        joint, _, _, _, _ = jnv.train_seed(full, cf, tpi, train, pairs, train_segs,
                                           fold_norm, cmap, train_meta, 0, device)
        base_logits = early.logits_for(base, "BASE", held, all_segs, fold_norm, cmap, device)
        joint_logits = jnv.logits_for(joint, held, all_segs, fold_norm, cmap, device)
        base_row = jnv.branch_metrics("PROSPECTIVE_BASE_FEAS", fold, held, base_logits)
        joint_row = jnv.branch_metrics(jnv.MODEL, fold, held, joint_logits)
        ref = reference[reference.fold.astype(int) == fold].iloc[0]
        if abs(base_row["BCE"] - float(ref.BCE)) > 1e-9 or abs(base_row["Brier"] - float(ref.Brier)) > 1e-9:
            raise RuntimeError(f"task0 fold {fold} Base reproducibility mismatch")
        templates = {cid: next(tr for tr in held if tr.context_id == cid)
                     for cid in sorted({tr.context_id for tr in held})}
        for name, model, row in [
            ("PROSPECTIVE_BASE_FEAS", ("BASE", base), base_row),
            (jnv.MODEL, (jnv.MODEL, joint), joint_row),
        ]:
            curves = {cid: jnv.dense_curve(model, tpi, cf, tr, fold_norm, gen.FORCES_DENSE)
                      for cid, tr in templates.items()}
            row.update(jnv.frontier_metrics(0, name, held, curves, gen.FORCES_DENSE))
            row.update({"task": 0, "seed": 0, "DEV_used": 0,
                        "heldout_root_ids": json.dumps(sorted(held_roots))})
            rows.append(row)
            for cid, curve in curves.items():
                for force, probability in zip(gen.FORCES_DENSE, curve):
                    prediction_rows.append({"task": 0, "fold": fold, "model": name,
                                            "context_id": cid, "force_N": float(force),
                                            "probability": float(probability)})
    for name in ["PROSPECTIVE_BASE_FEAS", jnv.MODEL]:
        q = [r for r in rows if r["model"] == name]
        mean = {"task": 0, "fold": "__MEAN__", "model": name, "seed": 0,
                "DEV_used": 0, "heldout_branches": 60, "heldout_contexts": 6,
                "heldout_roots": 2, "heldout_root_ids": ""}
        for key in ["BCE", "NLL", "probability_MAE", "Brier", "accuracy_0.5", "signed_bias",
                    "valid_frontier_contexts", "finite_decision_contexts", "frontier_MAE_N",
                    "under_force_rate", "mean_local_monotonicity", "context_monotonic_rate"]:
            mean[key] = float(np.mean([r[key] for r in q]))
        rows.append(mean)
    early.write_csv(OUT / "TASK0_JOINT_NOVISUAL_ROOT_CV.csv", rows)
    early.write_csv(OUT / "TASK0_JOINT_NOVISUAL_ROOT_CV_DENSE_PREDICTIONS.csv", prediction_rows)
    means = [r for r in rows if r["fold"] == "__MEAN__"]
    early.write_json(OUT / "TASK0_JOINT_NOVISUAL_ROOT_CV_SUMMARY.json", {
        "status": "COMPLETE_RETROSPECTIVE_TRAIN_ROOT_HELDOUT_DIAGNOSTIC",
        "Base_recomputed_exactly_matches_existing_CV": True,
        "selection_use_forbidden": True, "models": means,
    })
    print(pd.DataFrame(means).to_string(index=False), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run()
