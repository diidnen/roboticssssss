#!/usr/bin/env python3
"""Retrospective diagnostic-only task0 Joint without visual conditioning.

This deliberately reuses the frozen task0 TRAIN population, architecture,
physical/IE targets, weights, epochs, optimizer and three-seed protocol.  It
does not launch the simulator, fit on roots 06/07, calibrate, select a model,
or modify any authoritative artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import task0_visual_context_early as early
import task0_visual_generalization as gen


ROOT = Path("/home/exouser/FORTE")
EARLY = ROOT / "task0_visual_context_early_20260831_025000"
OUT = ROOT / "task0_joint_novisual_diagnostic_20260831"
SEEDS = [0, 1, 2]
MODEL = "JOINT_NOVISUAL"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def train_seed(full, cf, tpi, traces, pairs, segs, norm, cmap, meta, seed, device):
    # Match current Visual Joint initialization protocol, not old Joint's
    # historical global RNG call.
    early.seed_everything(seed + 3000)
    base_ck, _, base_path = full.load_base(tpi, seed, device)
    model = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=early.LR, weight_decay=early.WEIGHT_DECAY)
    units = cf.make_units(tpi, traces, pairs)
    hist = []
    steps = 0
    for epoch in range(1, early.EPOCHS + 1):
        batches = cf.batches_for_units(units, seed, epoch)
        fids = early.sampled_ids(traces, meta, seed + 100, epoch, n=len(batches) * early.BATCH)
        bases, ies, feass, totals = [], [], [], []
        model.train()
        for bi, batch in enumerate(batches):
            _, step, cond, ytraj, mask, weight = cf.batch_tensors(batch, norm, device)
            fq = [traces[int(i)] for i in fids[bi * early.BATCH:(bi + 1) * early.BATCH]]
            fstep, fcond, _, fy = early.branch_tensors(fq, segs, norm, cmap, device)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(step, cond)
            _, flogit = model(fstep, fcond)
            physics = (
                nn.functional.smooth_l1_loss(pred, ytraj, reduction="none")
                * mask * weight[:, None, None]
            ).sum() / (mask.sum() + 1e-6)
            ie = early.physical_ie_loss(pred, batch, norm, device)
            feas = nn.functional.binary_cross_entropy_with_logits(flogit, fy)
            total = physics + ie + early.LAMBDA_FEAS * feas
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            bases.append(float(physics.item()))
            ies.append(float(ie.item()))
            feass.append(float(feas.item()))
            totals.append(float(total.item()))
        hist.append({
            "model": MODEL, "seed": seed, "epoch": epoch,
            "trajectory_loss": float(np.mean(bases)),
            "ie_loss": float(np.mean(ies)),
            "feasibility_loss": float(np.mean(feass)),
            "native_total_loss": float(np.mean(totals)),
            "optimizer_steps": steps, "physical_units": len(units),
        })
        if epoch % 10 == 0:
            print(
                f"[JOINT-NOVISUAL] seed={seed} epoch={epoch}/{early.EPOCHS} "
                f"traj={np.mean(bases):.6f} ie={np.mean(ies):.6f} feas={np.mean(feass):.6f}",
                flush=True,
            )
    model.eval()
    return model, hist, steps, len(units), base_path


def load_or_train(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    tpi = early.load_module("jnv_tpi", early.TPI_CODE)
    cf = early.load_module("jnv_cf", early.CF_CODE)
    full = early.load_module("jnv_full", early.FULL_CODE)
    _, branches, cmap, traces, meta, visual_dim = early.audit_and_load(out, tpi)
    pairs = early.build_pairs(cf, traces, meta)
    segs, recomputed_norm = early.build_segments_and_norm(cf, tpi, traces)
    q = np.load(EARLY / "TASK0_TRAIN_NORMALIZATION.npz")
    norm = tuple(q[k].astype(np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
    if not all(np.array_equal(a, b) for a, b in zip(norm, recomputed_norm)):
        raise RuntimeError("recomputed TRAIN normalization differs from frozen Base normalization")
    device = torch.device("cpu")
    models = {}
    checkpoint_rows = []
    histories = []
    for seed in SEEDS:
        path = out / f"TASK0_JOINT_NOVISUAL_seed{seed}.pt"
        if path.exists():
            ck = torch.load(path, map_location=device, weights_only=False)
            base_ck, _, base_path = full.load_base(tpi, seed, device)
            model = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            steps = int(ck["optimizer_steps"])
            units = int(ck["physical_units"])
        else:
            model, hist, steps, units, base_path = train_seed(
                full, cf, tpi, traces, pairs, segs, norm, cmap, meta, seed, device
            )
            histories.extend(hist)
            torch.save({
                "variant": MODEL, "seed": seed, "state_dict": model.state_dict(),
                "epochs": early.EPOCHS, "optimizer": "AdamW", "lr": early.LR,
                "weight_decay": early.WEIGHT_DECAY, "batch_size": early.BATCH,
                "lambda_physics": 1.0, "lambda_IE": 1.0,
                "lambda_feasibility": early.LAMBDA_FEAS,
                "optimizer_steps": steps, "physical_units": units,
                "feasibility_branches": len(traces), "physical_branches": len(traces),
                "adjacent_IE_pairs": len(pairs), "visual_input": False,
                "initial_physics_checkpoint": str(base_path),
                "initial_physics_checkpoint_sha256": sha256(Path(base_path)),
                "TRAIN_only": True, "roots06_07_used_for_training": False,
                "diagnostic_only": True,
            }, path)
        models[seed] = model
        checkpoint_rows.append({
            "model": MODEL, "seed": seed, "checkpoint": str(path),
            "sha256": sha256(path), "epochs": early.EPOCHS,
            "optimizer_steps": steps, "physical_units": units,
        })
    if histories:
        early.write_csv(out / "TASK0_JOINT_NOVISUAL_TRAINING_MANIFEST.csv", histories)
    early.write_json(out / "TASK0_JOINT_NOVISUAL_CHECKPOINTS.json", {
        "status": "DIAGNOSTIC_ONLY_3_SEEDS_FROZEN",
        "TRAIN_contexts": 18, "TRAIN_roots": 6, "TRAIN_branches": len(branches),
        "visual_input": False, "same_nonvisual_inputs_as_Base": True,
        "same_auxiliary_as_Visual_Joint": True,
        "not_eligible_for_DEV_selection": True, "checkpoints": checkpoint_rows,
    })
    return tpi, cf, full, norm, models, visual_dim


def jnv_predictions(tpi, cf, models, cmap, templates, norm):
    rows = []
    with torch.no_grad():
        for cid in sorted(templates):
            for force in gen.FORCES_DENSE:
                step_np, cond_np = gen.normalized_segment(cf, tpi, templates[cid], float(force), norm)
                step = torch.tensor(step_np[None], dtype=torch.float32)
                cond = torch.tensor(cond_np[None], dtype=torch.float32)
                probs = []
                for seed in SEEDS:
                    _, logit = models[seed](step, cond)
                    probs.append(float(torch.sigmoid(logit)[0]))
                rows.append({
                    "context_id": cid, "force_N": float(force), "model": MODEL,
                    **{f"seed{s}_prob": probs[s] for s in SEEDS},
                    "ensemble_prob": float(np.mean(probs)),
                    "seed_std": float(np.std(probs, ddof=1)),
                })
    return pd.DataFrame(rows)


def main(out: Path):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tpi, cf, full, norm, jmodels, _ = load_or_train(out)
    contexts, branches, _, cmap, templates, failures = gen.load_dev(tpi, cf)
    if failures:
        raise RuntimeError("retrospective roots06/07 audit failed: " + "; ".join(failures[:3]))
    existing_models = gen.load_models(tpi, full, torch.device("cpu"))
    existing = gen.predict_all(cf, tpi, existing_models, cmap, templates, norm, torch.device("cpu"))
    pred = pd.concat([existing, jnv_predictions(tpi, cf, jmodels, cmap, templates, norm)], ignore_index=True)
    real, real_fronts = gen.real_curves(branches)
    old_models = list(gen.MODELS)
    try:
        gen.MODELS = [
            "PROSPECTIVE_BASE_FEAS", "VISUAL_CONTEXT_FULL_FEAS",
            "VISUAL_CONTEXT_JOINT", MODEL,
        ]
        prob = gen.probability_metrics(pred, real)
        shape = gen.shape_metrics(pred)
        frontier_detail, frontier = gen.frontier_metrics(pred, real_fronts)
    finally:
        gen.MODELS = old_models
    ensemble = prob[prob.aggregation == "ENSEMBLE"].copy()
    aggregate_shape = shape[shape.context_id == "__AGGREGATE__"].copy()
    summary = ensemble.merge(frontier, on="model").merge(
        aggregate_shape[["model", "context_monotonic", "safe_to_unsafe_reversals", "nonmonotonic_steps"]],
        on="model",
    )
    summary["scope"] = "RETROSPECTIVE_ROOT06_ROOT07_DIAGNOSTIC_ONLY"
    summary["eligible_for_selection"] = False
    summary.to_csv(ROOT / "TASK0_JOINT_NOVISUAL_DIAGNOSTIC.csv", index=False)
    pred.to_csv(out / "TASK0_JOINT_NOVISUAL_RETROSPECTIVE_PREDICTIONS.csv", index=False)
    frontier_detail.to_csv(out / "TASK0_JOINT_NOVISUAL_PER_ROOT_FRONTIER.csv", index=False)
    early.write_json(out / "TASK0_JOINT_NOVISUAL_RESULT.json", {
        "status": "COMPLETE_RETROSPECTIVE_DIAGNOSTIC",
        "scope": "same task0/object distribution; inspected roots06/07 only",
        "selection_use_forbidden": True,
        "models": summary.to_dict("records"),
        "interpretation_rule": {
            "if_joint_novisual_stable_visual_joint_bad": "supports visual-by-joint context overfit",
            "if_joint_novisual_also_bad": "supports physics auxiliary sample limitation within task0",
        },
    })
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    main(args.out)
