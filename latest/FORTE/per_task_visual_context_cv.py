#!/usr/bin/env python3
"""Task-specific, TRAIN-root-heldout CV for visual-context mechanisms.

No DEV file is read.  Each fold holds out two complete roots (six contexts),
fits PCA and normalization on the remaining four roots, retrains the same four
model families with seed 0, and evaluates the held-out TRAIN roots.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

import task0_visual_context_early as early
import per_task_visual_context_early as taskwise


MODELS = [
    "PROSPECTIVE_BASE_FEAS",
    "VISUAL_INTERCEPT_RESIDUAL",
    "VISUAL_CONTEXT_FULL_FEAS",
    "VISUAL_CONTEXT_JOINT",
]


def metric(model: str, fold: int, held, logits: dict[str, float]) -> dict:
    y = np.asarray([t.outcome for t in held], float)
    z = np.asarray([logits[t.branch_id] for t in held], float)
    p = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    return {
        "fold": fold,
        "model": model,
        "heldout_contexts": len({t.context_id for t in held}),
        "heldout_roots": len({t.root_id for t in held}),
        "heldout_branches": len(held),
        "BCE": float(-np.mean(y * np.log(np.clip(p, 1e-8, 1)) + (1-y) * np.log(np.clip(1-p, 1e-8, 1)))),
        "Brier": float(np.mean((p-y) ** 2)),
        "accuracy_0.5": float(np.mean((p >= .5) == y)),
        "signed_bias": float(np.mean(p-y)),
    }


def run(task: int, out: Path) -> None:
    frozen = out / f"ALL_PROSPECTIVE_VISUAL_MODELS_FROZEN_TASK{task}_ONLY.json"
    if not frozen.exists():
        raise RuntimeError(f"task{task} frozen checkpoints missing: {frozen}")
    freeze = json.loads(frozen.read_text())
    if freeze.get("checkpoint_count") != 12 or freeze.get("DEV_used") is not False:
        raise RuntimeError(f"task{task} checkpoint freeze is not valid")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    tpi = early.load_module(f"tpi_task{task}_cv", early.TPI_CODE)
    cf = early.load_module(f"cf_task{task}_cv", early.CF_CODE)
    full = early.load_module(f"full_task{task}_cv", early.FULL_CODE)
    tmp = out / "_cv_train_recheck"
    tmp.mkdir(exist_ok=True)
    cmap0, traces, meta, _, _, _, audit = taskwise.audit_and_load(task, tmp, tpi)
    roots = sorted({t.root_id for t in traces})
    if len(roots) != 6:
        raise RuntimeError(f"task{task}: expected 6 roots, got {len(roots)}")
    folds = [set(roots[i::3]) for i in range(3)]
    rows: list[dict] = []

    for fi, held_roots in enumerate(folds):
        train = [t for t in traces if t.root_id not in held_roots]
        held = [t for t in traces if t.root_id in held_roots]
        train_cids = sorted({t.context_id for t in train})
        all_cids = sorted(cmap0)
        raw = np.stack([cmap0[c]["visual_raw"] for c in train_cids])
        mean = raw.mean(0)
        _, singular, vt = np.linalg.svd(raw - mean, full_matrices=False)
        rank = min(17, len(train_cids)-1, int(np.sum(singular > singular[0] * 1e-7)))
        components = vt[:rank]
        ztrain = (raw - mean) @ components.T
        zmean, zstd = ztrain.mean(0), ztrain.std(0)
        zstd[zstd < 1e-6] = 1.0
        cmap = {}
        for cid in all_cids:
            z = ((cmap0[cid]["visual_raw"] - mean) @ components.T - zmean) / zstd
            z17 = np.zeros(17, np.float32)
            z17[:rank] = z.astype(np.float32)
            cmap[cid] = {**cmap0[cid], "visual": z17}

        segments_all = {t.branch_id: cf.build_seg(tpi, t, t.force, early.H) for t in traces}
        segments_train = {t.branch_id: segments_all[t.branch_id] for t in train}
        norm = early.build_segments_and_norm(cf, tpi, train)[1]
        train_meta = {t.branch_id: meta[t.branch_id] for t in train}
        pairs = taskwise.build_pairs(task, cf, train, train_meta)
        print(f"[task{task} CV] fold={fi} train_contexts={len(train_cids)} held_contexts={len({t.context_id for t in held})} PCA_rank={rank}", flush=True)

        base, _, _ = early.train_base(full, train, segments_train, norm, cmap, train_meta, device, 0)
        residual, _, _ = early.train_residual(base, train, segments_train, norm, cmap, train_meta, device, 0, 17)
        visual, _, _ = early.train_full(train, segments_train, norm, cmap, train_meta, device, 0, 17)
        joint, _, _, _, _ = early.train_joint(
            full, cf, tpi, train, pairs, segments_train, norm, cmap, train_meta, device, 0, 17,
        )
        for name, model, kind, base_model in [
            (MODELS[0], base, "BASE", None),
            (MODELS[1], residual, "RESIDUAL", base),
            (MODELS[2], visual, "FULL", None),
            (MODELS[3], joint, "JOINT", None),
        ]:
            logits = early.logits_for(model, kind, held, segments_all, norm, cmap, device, base=base_model)
            row = metric(name, fi, held, logits)
            row.update({
                "task": task,
                "heldout_root_ids": json.dumps(sorted(held_roots)),
                "fold_PCA_effective_rank": rank,
                "seed": 0,
                "DEV_used": 0,
            })
            rows.append(row)

    for model in MODELS:
        q = [r for r in rows if r["model"] == model]
        rows.append({
            "fold": "__MEAN__", "model": model, "task": task,
            "heldout_contexts": 6, "heldout_roots": 2, "heldout_branches": 60,
            **{k: float(np.mean([r[k] for r in q])) for k in ["BCE", "Brier", "accuracy_0.5", "signed_bias"]},
            "heldout_root_ids": "", "fold_PCA_effective_rank": 11, "seed": 0, "DEV_used": 0,
        })
    early.write_csv(out / f"TASK{task}_GROUP_HELDOUT_CV.csv", rows)
    means = {r["model"]: r for r in rows if r["fold"] == "__MEAN__"}
    base_bce = means[MODELS[0]]["BCE"]
    full_bce = means[MODELS[2]]["BCE"]
    joint_bce = means[MODELS[3]]["BCE"]
    summary = {
        "status": "COMPLETE_TRAIN_ROOT_HELDOUT_CV",
        "task": task,
        "design": "3 folds; hold out 2 complete roots / 6 contexts / 60 branches per fold; fold-local PCA and normalization; seed 0",
        "primary_label_audit_status": audit["status"],
        "mean_metrics": means,
        "Full_vs_Base_BCE_relative_improvement": (base_bce - full_bce) / base_bce,
        "Joint_vs_Full_BCE_relative_improvement": (full_bce - joint_bce) / full_bce,
        "CV_visual_support": bool(full_bce < base_bce),
        "CV_joint_support": bool(joint_bce < full_bce),
        "DEV_used": False,
        "classification": "TRAIN_ROOT_HELDOUT_DIAGNOSTIC_ONLY",
    }
    early.write_json(out / f"TASK{task}_GROUP_HELDOUT_CV_SUMMARY.json", summary)
    shutil.rmtree(tmp)
    hashes = [f"{taskwise.sha256(p)}  {p.name}" for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"]
    (out / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=taskwise.SUPPORTED_TASKS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(args.task, args.out.resolve())


if __name__ == "__main__":
    main()
