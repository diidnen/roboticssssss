#!/usr/bin/env python3
"""Frozen taskwise Joint-NoVisual mechanism diagnostic for tasks 1/5/6.

The model, auxiliary targets, loss weights, optimizer, epochs, seeds and
nonvisual input are inherited from the frozen taskwise Visual Joint.  No DEV
row is read by ``train`` or ``cv``.  ``dev`` is retrospective evaluation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import per_task_visual_context_early as taskwise
import per_task_visual_generalization as taskgen
import task0_visual_context_early as early
import task0_visual_generalization as gen


MODEL = "JOINT_NOVISUAL"
SEEDS = [0, 1, 2]
RHO = 0.80


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def logits_for(model, traces, segs, norm, cmap, device):
    ans = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(traces), 256):
            batch = traces[start:start + 256]
            step, cond, _, _ = early.branch_tensors(batch, segs, norm, cmap, device)
            _, logits = model(step, cond)
            ans.update({tr.branch_id: float(v) for tr, v in zip(batch, logits.cpu().numpy())})
    return ans


def branch_metrics(model: str, fold, traces, logits) -> dict:
    y = np.asarray([tr.outcome for tr in traces], float)
    z = np.asarray([logits[tr.branch_id] for tr in traces], float)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    nll = float(-np.mean(y * np.log(np.clip(p, 1e-8, 1.0)) + (1-y) * np.log(np.clip(1-p, 1e-8, 1.0))))
    return {
        "fold": fold, "model": model, "heldout_branches": len(traces),
        "heldout_contexts": len({tr.context_id for tr in traces}),
        "heldout_roots": len({tr.root_id for tr in traces}),
        "BCE": nll, "NLL": nll,
        "probability_MAE": float(np.mean(np.abs(p-y))),
        "Brier": float(np.mean((p-y)**2)),
        "accuracy_0.5": float(np.mean((p >= .5) == y)),
        "signed_bias": float(np.mean(p-y)),
    }


def train_seed(full, cf, tpi, traces, pairs, segs, norm, cmap, meta, seed, device):
    # Frozen to match current taskwise Visual Joint rather than the historical
    # old-Joint RNG call.
    early.seed_everything(seed + 3000)
    base_ck, _, base_path = full.load_base(tpi, seed, device)
    model = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=early.LR, weight_decay=early.WEIGHT_DECAY)
    units = cf.make_units(tpi, traces, pairs)
    history = []
    steps = 0
    for epoch in range(1, early.EPOCHS + 1):
        batches = cf.batches_for_units(units, seed, epoch)
        fids = early.sampled_ids(traces, meta, seed + 100, epoch, n=len(batches) * early.BATCH)
        physics_values, ie_values, feas_values, total_values = [], [], [], []
        model.train()
        for bi, batch in enumerate(batches):
            _, step, cond, ytraj, mask, weight = cf.batch_tensors(batch, norm, device)
            fq = [traces[int(i)] for i in fids[bi*early.BATCH:(bi+1)*early.BATCH]]
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
            physics_values.append(float(physics.item()))
            ie_values.append(float(ie.item()))
            feas_values.append(float(feas.item()))
            total_values.append(float(total.item()))
        history.append({
            "model": MODEL, "seed": seed, "epoch": epoch,
            "trajectory_loss": float(np.mean(physics_values)),
            "ie_loss": float(np.mean(ie_values)),
            "feasibility_loss": float(np.mean(feas_values)),
            "native_total_loss": float(np.mean(total_values)),
            "optimizer_steps": steps, "physical_units": len(units),
        })
        if epoch % 10 == 0:
            print(
                f"[{MODEL}] seed={seed} epoch={epoch}/{early.EPOCHS} "
                f"traj={np.mean(physics_values):.6f} ie={np.mean(ie_values):.6f} "
                f"feas={np.mean(feas_values):.6f}", flush=True,
            )
    model.eval()
    return model, history, steps, len(units), Path(base_path)


def load_population(task: int, scratch: Path):
    suffix = f"{task}_{scratch.name}_{id(scratch)}"
    tpi = early.load_module("jnv_tpi_" + suffix, early.TPI_CODE)
    cf = early.load_module("jnv_cf_" + suffix, early.CF_CODE)
    full = early.load_module("jnv_full_" + suffix, early.FULL_CODE)
    cmap, traces, meta, visual_dim, pca_path, canonical, audit = taskwise.audit_and_load(task, scratch, tpi)
    pairs = taskwise.build_pairs(task, cf, traces, meta)
    segs, norm = early.build_segments_and_norm(cf, tpi, traces)
    return tpi, cf, full, cmap, traces, meta, visual_dim, pca_path, canonical, audit, pairs, segs, norm


def frozen_norm(frozen: Path, task: int):
    q = np.load(frozen / f"TASK{task}_TRAIN_NORMALIZATION.npz")
    return tuple(q[k].astype(np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])


def load_models(task: int, frozen: Path, out: Path, tpi, full, device):
    models = {}
    for seed in SEEDS:
        path = out / f"TASK{task}_JOINT_NOVISUAL_seed{seed}.pt"
        if not path.exists():
            raise RuntimeError(f"missing Joint-NoVisual checkpoint: {path}")
        ck = torch.load(path, map_location=device, weights_only=False)
        base_ck, _, _ = full.load_base(tpi, seed, device)
        model = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        models[seed] = model
    return models


def run_train(task: int, frozen: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "_train_audit"
    scratch.mkdir(exist_ok=True)
    tpi, cf, full, cmap, traces, meta, _, _, canonical, audit, pairs, segs, computed_norm = load_population(task, scratch)
    norm = frozen_norm(frozen, task)
    if not all(np.array_equal(a, b) for a, b in zip(norm, computed_norm)):
        raise RuntimeError("recomputed TRAIN normalization differs from frozen corresponding Base normalization")
    device = torch.device("cpu")
    histories, checkpoint_rows, train_rows = [], [], []
    for seed in SEEDS:
        path = out / f"TASK{task}_JOINT_NOVISUAL_seed{seed}.pt"
        if path.exists():
            ck = torch.load(path, map_location=device, weights_only=False)
            base_ck, _, base_path = full.load_base(tpi, seed, device)
            model = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            steps, units = int(ck["optimizer_steps"]), int(ck["physical_units"])
            base_path = Path(base_path)
        else:
            model, hist, steps, units, base_path = train_seed(
                full, cf, tpi, traces, pairs, segs, norm, cmap, meta, seed, device,
            )
            histories.extend(hist)
            torch.save({
                "variant": MODEL, "task": task, "seed": seed,
                "state_dict": model.state_dict(), "epochs": early.EPOCHS,
                "optimizer": "AdamW", "lr": early.LR,
                "weight_decay": early.WEIGHT_DECAY, "batch_size": early.BATCH,
                "lambda_physics": 1.0, "lambda_IE": 1.0,
                "lambda_feasibility": early.LAMBDA_FEAS,
                "optimizer_steps": steps, "physical_units": units,
                "feasibility_branches": len(traces), "physical_branches": len(traces),
                "adjacent_IE_pairs": len(pairs), "visual_input": False,
                "same_nonvisual_input_as_Base": True,
                "initial_physics_checkpoint": str(base_path),
                "initial_physics_checkpoint_sha256": sha256(base_path),
                "TRAIN_only": True, "DEV_used": False, "TEST_used": False,
                "diagnostic_only": True,
            }, path)
        z = logits_for(model, traces, segs, norm, cmap, device)
        train_rows.append({**early.train_metrics(MODEL, seed, traces, z), "task": task})
        checkpoint_rows.append({
            "model": MODEL, "task": task, "seed": seed, "checkpoint": str(path),
            "sha256": sha256(path), "epochs": early.EPOCHS,
            "optimizer_steps": steps, "physical_units": units,
        })
    if histories:
        early.write_csv(out / f"TASK{task}_JOINT_NOVISUAL_TRAINING_MANIFEST.csv", histories)
    early.write_csv(out / f"TASK{task}_JOINT_NOVISUAL_TRAIN_METRICS.csv", train_rows)
    early.write_json(out / f"TASK{task}_JOINT_NOVISUAL_CHECKPOINTS.json", {
        "status": "DIAGNOSTIC_ONLY_3_SEEDS_FROZEN", "task": task,
        "TRAIN_contexts": len({tr.context_id for tr in traces}),
        "TRAIN_roots": len({tr.root_id for tr in traces}),
        "TRAIN_branches": len(traces), "successes": int(sum(tr.outcome for tr in traces)),
        "direct_labels": audit["direct_labels"],
        "reconstructed_labels": audit["reconstructed_labels"],
        "visual_input": False, "same_nonvisual_inputs_as_Base": True,
        "same_auxiliary_as_Visual_Joint": True,
        "not_eligible_for_DEV_selection": True, "checkpoints": checkpoint_rows,
    })
    print(pd.DataFrame(train_rows).to_string(index=False), flush=True)


def dense_curve(model, tpi, cf, template, norm, forces):
    probs = []
    with torch.no_grad():
        for force in forces:
            step_np, cond_np = gen.normalized_segment(cf, tpi, template, float(force), norm)
            step = torch.tensor(step_np[None], dtype=torch.float32)
            cond = torch.tensor(cond_np[None], dtype=torch.float32)
            if isinstance(model, tuple):
                kind, actual = model
                logit = actual(step, cond) if kind == "BASE" else actual(step, cond)[1]
            else:
                _, logit = model(step, cond)
            probs.append(float(torch.sigmoid(logit)[0]))
    return np.asarray(probs)


def frontier_metrics(task: int, model_name: str, traces, curves: dict[str, np.ndarray], forces) -> dict:
    real_frontiers, selected, monotonic = [], [], []
    for cid, curve in curves.items():
        q = [tr for tr in traces if tr.context_id == cid]
        cells = {}
        for tr in q:
            cells.setdefault(float(tr.force), []).append(float(tr.outcome))
        real = math.nan
        for force in sorted(cells):
            if float(np.mean(cells[force])) >= RHO:
                real = force
                break
        idx = np.flatnonzero(curve >= RHO)
        sel = float(forces[int(idx[0])]) if len(idx) else math.nan
        real_frontiers.append(real)
        selected.append(sel)
        # Match the frozen taskwise evaluator's numerical monotonicity rule.
        monotonic.append(float(np.mean(np.diff(curve) >= -1e-8)))
    valid = [(r, s) for r, s in zip(real_frontiers, selected) if math.isfinite(r)]
    finite = [(r, s) for r, s in valid if math.isfinite(s)]
    return {
        "model": model_name,
        "valid_frontier_contexts": len(valid),
        "finite_decision_contexts": len(finite),
        "frontier_MAE_N": float(np.mean([abs(r-s) for r, s in finite])) if finite else math.nan,
        "under_force_rate": float(np.mean([not math.isfinite(s) or s < r-1e-9 for r, s in valid])) if valid else math.nan,
        "mean_local_monotonicity": float(np.mean(monotonic)) if monotonic else math.nan,
        "context_monotonic_rate": float(np.mean([x >= 1.0-1e-12 for x in monotonic])) if monotonic else math.nan,
    }


def run_cv(task: int, frozen: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "_cv_audit"
    scratch.mkdir(exist_ok=True)
    tpi, cf, full, cmap, traces, meta, _, _, _, _, _, _, _ = load_population(task, scratch)
    roots = sorted({tr.root_id for tr in traces})
    if len(roots) != 6:
        raise RuntimeError(f"expected six roots, got {len(roots)}")
    folds = [set(roots[i::3]) for i in range(3)]
    forces = taskgen.TASK_DENSE_FORCES[task]
    rows, prediction_rows = [], []
    existing_cv = pd.read_csv(frozen / f"TASK{task}_GROUP_HELDOUT_CV.csv")
    existing_base = existing_cv[(existing_cv.model == "PROSPECTIVE_BASE_FEAS") & (existing_cv.fold.astype(str) != "__MEAN__")]
    device = torch.device("cpu")
    for fold, held_roots in enumerate(folds):
        train = [tr for tr in traces if tr.root_id not in held_roots]
        held = [tr for tr in traces if tr.root_id in held_roots]
        train_meta = {tr.branch_id: meta[tr.branch_id] for tr in train}
        all_segs = {tr.branch_id: cf.build_seg(tpi, tr, tr.force, early.H) for tr in traces}
        train_segs = {tr.branch_id: all_segs[tr.branch_id] for tr in train}
        norm = early.build_segments_and_norm(cf, tpi, train)[1]
        pairs = taskwise.build_pairs(task, cf, train, train_meta)
        print(f"[task{task} JNV CV] fold={fold} train={len(train)} held={len(held)}", flush=True)
        base, _, _ = early.train_base(full, train, train_segs, norm, cmap, train_meta, device, 0)
        jnv, _, _, _, _ = train_seed(full, cf, tpi, train, pairs, train_segs, norm, cmap, train_meta, 0, device)
        base_logits = early.logits_for(base, "BASE", held, all_segs, norm, cmap, device)
        jnv_logits = logits_for(jnv, held, all_segs, norm, cmap, device)
        base_row = branch_metrics("PROSPECTIVE_BASE_FEAS", fold, held, base_logits)
        jnv_row = branch_metrics(MODEL, fold, held, jnv_logits)
        reference = existing_base[existing_base.fold.astype(int) == fold].iloc[0]
        if abs(base_row["BCE"] - float(reference.BCE)) > 1e-9 or abs(base_row["Brier"] - float(reference.Brier)) > 1e-9:
            raise RuntimeError(f"fold {fold} Base reproducibility mismatch")
        templates = {cid: next(tr for tr in held if tr.context_id == cid) for cid in sorted({tr.context_id for tr in held})}
        for name, model in [("PROSPECTIVE_BASE_FEAS", ("BASE", base)), (MODEL, (MODEL, jnv))]:
            curves = {cid: dense_curve(model, tpi, cf, template, norm, forces) for cid, template in templates.items()}
            fm = frontier_metrics(task, name, held, curves, forces)
            row = base_row if name == "PROSPECTIVE_BASE_FEAS" else jnv_row
            row.update(fm)
            row.update({"task": task, "seed": 0, "DEV_used": 0,
                        "heldout_root_ids": json.dumps(sorted(held_roots))})
            rows.append(row)
            for cid, curve in curves.items():
                for force, prob in zip(forces, curve):
                    prediction_rows.append({"task": task, "fold": fold, "model": name,
                                            "context_id": cid, "force_N": float(force), "probability": float(prob)})
    for model_name in ["PROSPECTIVE_BASE_FEAS", MODEL]:
        q = [r for r in rows if r["model"] == model_name]
        mean = {"task": task, "fold": "__MEAN__", "model": model_name,
                "seed": 0, "DEV_used": 0, "heldout_branches": 60,
                "heldout_contexts": 6, "heldout_roots": 2, "heldout_root_ids": ""}
        for key in ["BCE", "NLL", "probability_MAE", "Brier", "accuracy_0.5", "signed_bias",
                    "valid_frontier_contexts", "finite_decision_contexts", "frontier_MAE_N",
                    "under_force_rate", "mean_local_monotonicity", "context_monotonic_rate"]:
            mean[key] = float(np.mean([r[key] for r in q]))
        rows.append(mean)
    early.write_csv(out / f"TASK{task}_JOINT_NOVISUAL_ROOT_CV.csv", rows)
    early.write_csv(out / f"TASK{task}_JOINT_NOVISUAL_ROOT_CV_DENSE_PREDICTIONS.csv", prediction_rows)
    means = [r for r in rows if r["fold"] == "__MEAN__"]
    early.write_json(out / f"TASK{task}_JOINT_NOVISUAL_ROOT_CV_SUMMARY.json", {
        "status": "COMPLETE_RETROSPECTIVE_TRAIN_ROOT_HELDOUT_DIAGNOSTIC",
        "task": task, "design": "3 folds; two complete held-out roots; seed0; fold-local normalization",
        "Base_recomputed_exactly_matches_existing_CV": True,
        "selection_use_forbidden": True, "models": means,
    })
    print(pd.DataFrame(means).to_string(index=False), flush=True)


def jnv_dense_predictions(task, tpi, cf, models, templates, norm):
    rows = []
    forces = taskgen.TASK_DENSE_FORCES[task]
    with torch.no_grad():
        for cid in sorted(templates):
            for force in forces:
                step_np, cond_np = gen.normalized_segment(cf, tpi, templates[cid], float(force), norm)
                step = torch.tensor(step_np[None], dtype=torch.float32)
                cond = torch.tensor(cond_np[None], dtype=torch.float32)
                probs = []
                for seed in SEEDS:
                    _, logit = models[seed](step, cond)
                    probs.append(float(torch.sigmoid(logit)[0]))
                rows.append({"context_id": cid, "force_N": float(force), "model": MODEL,
                             **{f"seed{s}_prob": probs[s] for s in SEEDS},
                             "ensemble_prob": float(np.mean(probs)),
                             "seed_std": float(np.std(probs, ddof=1))})
    return pd.DataFrame(rows)


def run_dev(task: int, frozen: Path, out: Path) -> None:
    readiness = taskgen.readiness(task, frozen)
    if not readiness["ready"]:
        raise RuntimeError("authoritative DEV incomplete: " + json.dumps(readiness))
    scratch = out / "_dev_audit"
    scratch.mkdir(exist_ok=True)
    suffix = f"{task}_{out.name}"
    tpi = early.load_module("jnv_dev_tpi_" + suffix, early.TPI_CODE)
    cf = early.load_module("jnv_dev_cf_" + suffix, early.CF_CODE)
    full = early.load_module("jnv_dev_full_" + suffix, early.FULL_CODE)
    contexts, branches, _, cmap, templates, failures = taskgen.load_dev(task, frozen, tpi, cf)
    if failures:
        raise RuntimeError("DEV audit failed: " + "; ".join(failures))
    norm = frozen_norm(frozen, task)
    device = torch.device("cpu")
    jmodels = load_models(task, frozen, out, tpi, full, device)
    gen.EARLY = frozen
    gen.FORCES_REAL = taskgen.TASK_REAL_FORCES[task]
    gen.FORCES_DENSE = taskgen.TASK_DENSE_FORCES[task]
    gen.checkpoint_path = lambda model, seed: taskgen.checkpoint(frozen, task, model, seed)
    existing_models = gen.load_models(tpi, full, device)
    existing = gen.predict_all(cf, tpi, existing_models, cmap, templates, norm, device)
    pred = pd.concat([existing, jnv_dense_predictions(task, tpi, cf, jmodels, templates, norm)], ignore_index=True)
    real, real_frontiers = gen.real_curves(branches)
    previous = list(gen.MODELS)
    try:
        gen.MODELS = ["PROSPECTIVE_BASE_FEAS", "VISUAL_CONTEXT_FULL_FEAS", "VISUAL_CONTEXT_JOINT", MODEL]
        prob = gen.probability_metrics(pred, real)
        shape = gen.shape_metrics(pred)
        frontier_detail, frontier = gen.frontier_metrics(pred, real_frontiers)
    finally:
        gen.MODELS = previous
    summary = prob[prob.aggregation == "ENSEMBLE"].merge(frontier, on="model").merge(
        shape[shape.context_id == "__AGGREGATE__"][["model", "context_monotonic", "safe_to_unsafe_reversals", "nonmonotonic_steps"]],
        on="model",
    )
    summary["task"] = task
    summary["scope"] = "RETROSPECTIVE_HELDOUT_ROOT_DIAGNOSTIC_ONLY"
    summary["eligible_for_selection"] = False
    summary.to_csv(out / f"TASK{task}_JOINT_NOVISUAL_DEV_SUMMARY.csv", index=False)
    pred.to_csv(out / f"TASK{task}_JOINT_NOVISUAL_DEV_PREDICTIONS.csv", index=False)
    frontier_detail.to_csv(out / f"TASK{task}_JOINT_NOVISUAL_DEV_FRONTIERS.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "cv", "dev"])
    ap.add_argument("--task", type=int, required=True, choices=[1, 5, 6])
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.phase == "train":
        run_train(args.task, args.frozen.resolve(), args.out.resolve())
    elif args.phase == "cv":
        run_cv(args.task, args.frozen.resolve(), args.out.resolve())
    else:
        run_dev(args.task, args.frozen.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
