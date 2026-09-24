#!/usr/bin/env python3
"""Read-only taskwise evaluator for the original pooled visual checkpoints.

The original pipeline reports only pooled probability summaries.  This script
does not retrain or select models; it reconstructs the same frozen arrays and
reports each held-out task separately.  Results retain a critical
implementation-confound flag because this Joint is not the authoritative
H8x13/target-matched-IE formulation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import prospective_visual_context_pipeline as pool


MODELS = [
    ("BASE", "PROSPECTIVE_BASE_FEAS"),
    ("RESIDUAL", "VISUAL_INTERCEPT_RESIDUAL"),
    ("FULL", "VISUAL_CONTEXT_FULL_FEAS"),
    ("JOINT", "VISUAL_CONTEXT_JOINT"),
]


def visual_map():
    caps = pool.captures()
    p = np.load(pool.ROOT / "VISUAL_PCA_TRAIN_ONLY.npz")
    ans = {}
    for cid, row in caps.items():
        raw = np.load(row["visual_feature_path"], allow_pickle=False).astype(np.float32)
        ans[cid] = (((raw-p["mean"])/p["scale"]) @ p["components"].T).astype(np.float32)
    return ans


def load_models(device):
    result = {}
    for kind, name in MODELS:
        result[name] = []
        for seed in pool.SEEDS:
            ck = torch.load(pool.ROOT / f"{name}_seed{seed}.pt", map_location=device, weights_only=False)
            if kind == "BASE":
                model = pool.Base()
            elif kind == "RESIDUAL":
                model = torch.nn.Linear(64, 1)
            elif kind == "FULL":
                model = pool.Full()
            else:
                model = pool.Joint()
            model.load_state_dict(ck["state_dict"])
            result[name].append(model.to(device).eval())
    return result


def predict(models, name, seq, cond, visual, device):
    s = torch.tensor(seq, device=device)
    c = torch.tensor(cond, device=device)
    x = torch.tensor(visual, device=device)
    values = []
    with torch.no_grad():
        for i, model in enumerate(models[name]):
            if name == "PROSPECTIVE_BASE_FEAS":
                z = model(s, c)
            elif name == "VISUAL_INTERCEPT_RESIDUAL":
                z = models["PROSPECTIVE_BASE_FEAS"][i](s, c) + model(x).squeeze(1)
            elif name == "VISUAL_CONTEXT_JOINT":
                z = model(s, c, x)[0]
            else:
                z = model(s, c, x)
            values.append(pool.sigmoid(z.cpu().numpy()))
    return np.mean(values, axis=0)


def run(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    if not (pool.ROOT / "ALL_PROSPECTIVE_VISUAL_MODELS_FROZEN.json").exists():
        raise RuntimeError("original pooled visual checkpoints are not frozen")
    xmap = visual_map()
    train = pool.build_arrays(pool.branch_records("TRAIN"), xmap)
    dev = pool.build_arrays(pool.branch_records("DEV"), xmap, train[-1])
    seq, cond, y, _, meta, _ = dev
    if len(meta) != 385:
        raise RuntimeError(f"expected 385 DEV branches, got {len(meta)}")
    device = torch.device("cpu")
    models = load_models(device)
    cells = defaultdict(list)
    for i, row in enumerate(meta):
        cells[(int(row["task"]), row["context_id"], round(float(row["force_N"]), 8))].append(i)
    cell_rows, curve_rows, frontier_rows = [], [], []
    for _, name in MODELS:
        for (task, cid, force), indices in sorted(cells.items()):
            probability = float(predict(models, name, seq[indices[:1]], cond[indices[:1]],
                                        np.stack([xmap[cid]]), device)[0])
            cell_rows.append({"task": task, "model": name, "context_id": cid,
                              "force_N": force, "p_pred": probability,
                              "p_real": float(np.mean(y[indices])), "repeats": len(indices)})
        for task in [0, 1, 5, 6]:
            task_contexts = sorted({r["context_id"] for r in meta if int(r["task"]) == task})
            lo, hi = pool.TASK_SUPPORT[task]
            grid = np.arange(lo, hi+1e-9, 0.05)
            for cid in task_contexts:
                base_i = next(i for i, r in enumerate(meta) if r["context_id"] == cid)
                probs = []
                for force in grid:
                    qs = seq[base_i:base_i+1].copy(); qs[:, :, 3] = float(force)
                    qc = cond[base_i:base_i+1].copy(); qc[:, 0] = float(force)/8.0
                    probability = float(predict(models, name, qs, qc, np.stack([xmap[cid]]), device)[0])
                    probs.append(probability)
                    curve_rows.append({"task": task, "model": name, "context_id": cid,
                                       "force_N": float(force), "p_pred": probability})
                real = sorted([r for r in cell_rows if r["model"] == name and r["context_id"] == cid],
                              key=lambda r: r["force_N"])
                real_front = next((r["force_N"] for r in real if r["p_real"] >= 0.8), np.nan)
                pred_front = next((float(f) for f, p in zip(grid, probs) if p >= 0.8), np.nan)
                frontier_rows.append({
                    "task": task, "model": name, "context_id": cid,
                    "real_frontier_N": real_front, "pred_frontier_N": pred_front,
                    "valid_real": int(np.isfinite(real_front)), "finite_model": int(np.isfinite(pred_front)),
                    "frontier_error_N": abs(pred_front-real_front) if np.isfinite(real_front) and np.isfinite(pred_front) else np.nan,
                    "under_force": int(not np.isfinite(pred_front) or (np.isfinite(real_front) and pred_front < real_front)),
                    "monotonic": int(np.all(np.diff(probs) >= -0.02)),
                })
    summaries = []
    for task in [0, 1, 5, 6]:
        for _, name in MODELS:
            q = pd.DataFrame([r for r in cell_rows if r["task"] == task and r["model"] == name])
            p = np.clip(q.p_pred.to_numpy(), 1e-6, 1-1e-6); yy = q.p_real.to_numpy()
            f = pd.DataFrame([r for r in frontier_rows if r["task"] == task and r["model"] == name])
            valid = f[f.valid_real == 1]; finite = valid[valid.finite_model == 1]
            summaries.append({
                "task": task, "model": name, "DEV_contexts": int(q.context_id.nunique()),
                "DEV_probability_cells": len(q), "probability_MAE": float(np.mean(np.abs(p-yy))),
                "Brier": float(np.mean((p-yy)**2)),
                "NLL": float(-np.mean(yy*np.log(p)+(1-yy)*np.log(1-p))),
                "valid_frontier_contexts": len(valid), "finite_decision_contexts": len(finite),
                "frontier_MAE_N": float(finite.frontier_error_N.mean()) if len(finite) else np.nan,
                "under_force_or_missing_rate": float(valid.under_force.mean()) if len(valid) else np.nan,
                "monotonic_context_rate": float(f.monotonic.mean()),
                "implementation_confounded": True,
                "confound": "H8x4 repeated physical target; no authoritative Physics-GRU; IE minimizes adjacent predicted difference instead of matching true trajectory difference",
                "selection_eligible": False,
            })
    pd.DataFrame(summaries).to_csv(out / "ORIGINAL_POOLED_VISUAL_TASKWISE.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(out / "ORIGINAL_POOLED_VISUAL_TASKWISE_CELL_PREDICTIONS.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "ORIGINAL_POOLED_VISUAL_TASKWISE_DENSE_CURVES.csv", index=False)
    pd.DataFrame(frontier_rows).to_csv(out / "ORIGINAL_POOLED_VISUAL_TASKWISE_FRONTIERS.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    run(args.out.resolve())
