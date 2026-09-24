#!/usr/bin/env python3
"""Matched current-data pooled Base versus Joint-NoVisual diagnostic.

This is deliberately separate from the original pooled visual implementation,
whose physical target and IE objective do not match the authoritative taskwise
Joint.  Here Base and Joint-NoVisual use the same current 72 contexts, pooled
TRAIN normalization, fixed 80 epochs and seeds 0/1/2.  CV holds out roots from
every task and always reports taskwise metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import per_task_visual_context_early as taskwise
import per_task_visual_generalization as taskgen
import task0_visual_context_early as early
import task0_visual_generalization as gen
import taskwise_joint_novisual as jnv


TASKS = [0, 1, 5, 6]
MODELS = ["POOLED_BASE", "POOLED_JOINT_NOVISUAL"]
TASK0_FROZEN = Path("/home/exouser/FORTE/task0_visual_context_early_20260831_025000")
TASK_FROZEN = {
    1: Path("/home/exouser/FORTE/taskwise_visual_context_20260831_074000/task1_train_frozen"),
    5: Path("/home/exouser/FORTE/taskwise_visual_context_20260831_074000/task5_train_frozen"),
    6: Path("/home/exouser/FORTE/taskwise_visual_context_20260831_074000/task6_train_frozen"),
}


def load_population(scratch: Path):
    suffix = scratch.name
    tpi = early.load_module("pool_tpi_" + suffix, early.TPI_CODE)
    cf = early.load_module("pool_cf_" + suffix, early.CF_CODE)
    full = early.load_module("pool_full_" + suffix, early.FULL_CODE)
    all_traces, all_meta, all_cmap, audits, pairs = [], {}, {}, {}, []
    task_dirs = {}
    for task in TASKS:
        d = scratch / f"task{task}"
        d.mkdir(parents=True, exist_ok=True)
        task_dirs[task] = d
        if task == 0:
            _, _, cmap, traces, meta, _ = early.audit_and_load(d, tpi)
            audit = json.loads((d / "TASK0_DATA_AUDIT.json").read_text())
        else:
            cmap, traces, meta, _, _, _, audit = taskwise.audit_and_load(task, d, tpi)
        all_traces.extend(traces)
        all_meta.update(meta)
        all_cmap.update(cmap)
        audits[task] = audit
        pairs.extend(taskwise.build_pairs(task, cf, traces, meta))
    if len(all_traces) != 720 or len({tr.context_id for tr in all_traces}) != 72 or len({tr.root_id for tr in all_traces}) != 24:
        raise RuntimeError("pooled current population is not 720 branches / 72 contexts / 24 roots")
    if len(pairs) != 576:
        raise RuntimeError(f"expected 576 adjacent IE pairs, got {len(pairs)}")
    segs, norm = early.build_segments_and_norm(cf, tpi, all_traces)
    return tpi, cf, full, all_cmap, all_traces, all_meta, audits, pairs, segs, norm


def save_checkpoint(path, model, kind, seed, norm, steps, extra=None):
    obj = {
        "variant": kind, "seed": seed, "state_dict": model.state_dict(),
        "epochs": early.EPOCHS, "optimizer": "AdamW", "lr": early.LR,
        "weight_decay": early.WEIGHT_DECAY, "batch_size": early.BATCH,
        "normalization": {k: v.tolist() for k, v in zip(["x_mean", "x_std", "y_mean", "y_std"], norm)},
        "optimizer_steps": steps, "TRAIN_contexts": 72, "TRAIN_roots": 24,
        "TRAIN_branches": 720, "DEV_used": False, "TEST_used": False,
        "visual_input": False, "selection_use_forbidden": True,
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def run_train(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tpi, cf, full, cmap, traces, meta, audits, pairs, segs, norm = load_population(out / "_train_audit")
    np.savez(out / "POOLED_MATCHED_TRAIN_NORMALIZATION.npz",
             x_mean=norm[0], x_std=norm[1], y_mean=norm[2], y_std=norm[3])
    device = torch.device("cpu")
    base_hist, joint_hist, train_rows, checkpoints = [], [], [], []
    for seed in jnv.SEEDS:
        bp = out / f"POOLED_BASE_seed{seed}.pt"
        if bp.exists():
            ck = torch.load(bp, map_location=device, weights_only=False)
            base = full.FeasibilityOnly().to(device)
            base.load_state_dict(ck["state_dict"])
            bsteps = int(ck["optimizer_steps"])
        else:
            base, bh, bsteps = early.train_base(full, traces, segs, norm, cmap, meta, device, seed)
            base_hist.extend(bh)
            save_checkpoint(bp, base, "POOLED_BASE", seed, norm, bsteps)
        bz = early.logits_for(base, "BASE", traces, segs, norm, cmap, device)
        train_rows.append({**early.train_metrics("POOLED_BASE", seed, traces, bz), "task": "POOLED"})
        checkpoints.append({"model": "POOLED_BASE", "seed": seed, "path": str(bp), "sha256": jnv.sha256(bp)})

        jp = out / f"POOLED_JOINT_NOVISUAL_seed{seed}.pt"
        if jp.exists():
            ck = torch.load(jp, map_location=device, weights_only=False)
            base_ck, _, initial = full.load_base(tpi, seed, device)
            joint = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
            joint.load_state_dict(ck["state_dict"])
            jsteps, units = int(ck["optimizer_steps"]), int(ck["physical_units"])
            initial = Path(initial)
        else:
            joint, jh, jsteps, units, initial = jnv.train_seed(
                full, cf, tpi, traces, pairs, segs, norm, cmap, meta, seed, device,
            )
            joint_hist.extend(jh)
            save_checkpoint(jp, joint, "POOLED_JOINT_NOVISUAL", seed, norm, jsteps, {
                "physical_units": units, "physical_branches": 720,
                "adjacent_IE_pairs": 576, "lambda_physics": 1.0,
                "lambda_IE": 1.0, "lambda_feasibility": early.LAMBDA_FEAS,
                "initial_physics_checkpoint": str(initial),
                "initial_physics_checkpoint_sha256": jnv.sha256(Path(initial)),
                "same_authoritative_auxiliary_as_taskwise_Visual_Joint": True,
            })
        jz = jnv.logits_for(joint, traces, segs, norm, cmap, device)
        train_rows.append({**early.train_metrics("POOLED_JOINT_NOVISUAL", seed, traces, jz), "task": "POOLED"})
        checkpoints.append({"model": "POOLED_JOINT_NOVISUAL", "seed": seed, "path": str(jp),
                            "sha256": jnv.sha256(jp), "physical_units": units})
    if base_hist:
        early.write_csv(out / "POOLED_BASE_TRAINING_MANIFEST.csv", base_hist)
    if joint_hist:
        early.write_csv(out / "POOLED_JOINT_NOVISUAL_TRAINING_MANIFEST.csv", joint_hist)
    early.write_csv(out / "POOLED_MATCHED_TRAIN_METRICS.csv", train_rows)
    early.write_json(out / "POOLED_MATCHED_MODELS_FROZEN.json", {
        "status": "FROZEN_6_MATCHED_CHECKPOINTS", "contexts": 72, "roots": 24,
        "branches": 720, "tasks": TASKS, "DEV_used": False, "TEST_used": False,
        "audits": audits, "checkpoints": checkpoints,
    })
    print(pd.DataFrame(train_rows).to_string(index=False), flush=True)


def model_logits(kind, model, traces, segs, norm, cmap, device):
    return (early.logits_for(model, "BASE", traces, segs, norm, cmap, device)
            if kind == "POOLED_BASE" else jnv.logits_for(model, traces, segs, norm, cmap, device))


def run_cv(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tpi, cf, full, cmap, traces, meta, _, _, all_segs, _ = load_population(out / "_cv_audit")
    roots_by_task = {task: sorted({tr.root_id for tr in traces if tr.task == task}) for task in TASKS}
    folds = [set().union(*(set(roots_by_task[t][i::3]) for t in TASKS)) for i in range(3)]
    device = torch.device("cpu")
    rows, curves_out = [], []
    for fold, held_roots in enumerate(folds):
        train = [tr for tr in traces if tr.root_id not in held_roots]
        held = [tr for tr in traces if tr.root_id in held_roots]
        train_meta = {tr.branch_id: meta[tr.branch_id] for tr in train}
        train_segs = {tr.branch_id: all_segs[tr.branch_id] for tr in train}
        norm = early.build_segments_and_norm(cf, tpi, train)[1]
        pairs = []
        for task in TASKS:
            qt = [tr for tr in train if tr.task == task]
            qm = {tr.branch_id: train_meta[tr.branch_id] for tr in qt}
            pairs.extend(taskwise.build_pairs(task, cf, qt, qm))
        print(f"[pooled CV] fold={fold} train_contexts={len({tr.context_id for tr in train})} held_contexts={len({tr.context_id for tr in held})}", flush=True)
        base, _, _ = early.train_base(full, train, train_segs, norm, cmap, train_meta, device, 0)
        joint, _, _, _, _ = jnv.train_seed(full, cf, tpi, train, pairs, train_segs, norm, cmap, train_meta, 0, device)
        for task in TASKS:
            q = [tr for tr in held if tr.task == task]
            for name, model in [("POOLED_BASE", ("BASE", base)), ("POOLED_JOINT_NOVISUAL", (jnv.MODEL, joint))]:
                logits = (early.logits_for(base, "BASE", q, all_segs, norm, cmap, device)
                          if name == "POOLED_BASE" else jnv.logits_for(joint, q, all_segs, norm, cmap, device))
                row = jnv.branch_metrics(name, fold, q, logits)
                force_grid = gen.FORCES_DENSE if task == 0 else taskgen.TASK_DENSE_FORCES[task]
                templates = {cid: next(tr for tr in q if tr.context_id == cid) for cid in sorted({tr.context_id for tr in q})}
                curves = {cid: jnv.dense_curve(model, tpi, cf, tr, norm, force_grid) for cid, tr in templates.items()}
                row.update(jnv.frontier_metrics(task, name, q, curves, force_grid))
                row.update({"task": task, "seed": 0, "DEV_used": 0,
                            "pooled_train_contexts": 48, "pooled_train_roots": 16,
                            "heldout_root_ids": json.dumps(sorted(r for r in held_roots if f"_t{task}_" in r))})
                rows.append(row)
                for cid, curve in curves.items():
                    for force, probability in zip(force_grid, curve):
                        curves_out.append({"fold": fold, "task": task, "model": name,
                                           "context_id": cid, "force_N": float(force),
                                           "probability": float(probability)})
    metric_keys = ["BCE", "NLL", "probability_MAE", "Brier", "accuracy_0.5", "signed_bias",
                   "valid_frontier_contexts", "finite_decision_contexts", "frontier_MAE_N",
                   "under_force_rate", "mean_local_monotonicity", "context_monotonic_rate"]
    for task in TASKS:
        for name in MODELS:
            q = [r for r in rows if r["task"] == task and r["model"] == name]
            mean = {"fold": "__MEAN__", "task": task, "model": name, "seed": 0,
                    "DEV_used": 0, "pooled_train_contexts": 48, "pooled_train_roots": 16,
                    "heldout_branches": 60, "heldout_contexts": 6, "heldout_roots": 2,
                    "heldout_root_ids": ""}
            mean.update({k: float(np.mean([r[k] for r in q])) for k in metric_keys})
            rows.append(mean)
    early.write_csv(out / "POOLED_MATCHED_ROOT_CV.csv", rows)
    early.write_csv(out / "POOLED_MATCHED_ROOT_CV_DENSE_PREDICTIONS.csv", curves_out)
    means = [r for r in rows if r["fold"] == "__MEAN__"]
    early.write_json(out / "POOLED_MATCHED_ROOT_CV_SUMMARY.json", {
        "status": "COMPLETE_WITHIN_TASK_ROOT_HELDOUT_DIAGNOSTIC", "tasks": TASKS,
        "not_unseen_task_generalization": True, "selection_use_forbidden": True,
        "models": means,
    })
    print(pd.DataFrame(means).to_string(index=False), flush=True)


def load_frozen_models(out: Path, tpi, full, device):
    models = {name: {} for name in MODELS}
    for seed in jnv.SEEDS:
        bck = torch.load(out / f"POOLED_BASE_seed{seed}.pt", map_location=device, weights_only=False)
        base = full.FeasibilityOnly().to(device)
        base.load_state_dict(bck["state_dict"])
        base.eval()
        jck = torch.load(out / f"POOLED_JOINT_NOVISUAL_seed{seed}.pt", map_location=device, weights_only=False)
        initial, _, _ = full.load_base(tpi, seed, device)
        joint = full.JointIEFeasibility(tpi, initial["state_dict"]).to(device)
        joint.load_state_dict(jck["state_dict"])
        joint.eval()
        models["POOLED_BASE"][seed] = base
        models["POOLED_JOINT_NOVISUAL"][seed] = joint
    return models


def predict_dev_task(task, tpi, cf, models, templates, norm, force_grid):
    rows = []
    with torch.no_grad():
        for cid in sorted(templates):
            for force in force_grid:
                step_np, cond_np = gen.normalized_segment(cf, tpi, templates[cid], float(force), norm)
                step = torch.tensor(step_np[None], dtype=torch.float32)
                cond = torch.tensor(cond_np[None], dtype=torch.float32)
                for name in MODELS:
                    probs = []
                    for seed in jnv.SEEDS:
                        model = models[name][seed]
                        logit = model(step, cond) if name == "POOLED_BASE" else model(step, cond)[1]
                        probs.append(float(torch.sigmoid(logit)[0]))
                    rows.append({"task": task, "context_id": cid, "force_N": float(force),
                                 "model": name, **{f"seed{s}_prob": probs[s] for s in jnv.SEEDS},
                                 "ensemble_prob": float(np.mean(probs)),
                                 "seed_std": float(np.std(probs, ddof=1))})
    return pd.DataFrame(rows)


def run_dev(out: Path) -> None:
    frozen = json.loads((out / "POOLED_MATCHED_MODELS_FROZEN.json").read_text())
    if frozen.get("DEV_used") is not False or len(frozen.get("checkpoints", [])) != 6:
        raise RuntimeError("matched pooled checkpoint freeze invalid")
    q = np.load(out / "POOLED_MATCHED_TRAIN_NORMALIZATION.npz")
    norm = tuple(q[k].astype(np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
    suffix = out.name + "_dev"
    tpi = early.load_module("pool_dev_tpi_" + suffix, early.TPI_CODE)
    cf = early.load_module("pool_dev_cf_" + suffix, early.CF_CODE)
    full = early.load_module("pool_dev_full_" + suffix, early.FULL_CODE)
    device = torch.device("cpu")
    models = load_frozen_models(out, tpi, full, device)
    all_summaries, all_predictions, all_frontiers = [], [], []
    previous = list(gen.MODELS)
    try:
        gen.MODELS = MODELS
        for task in TASKS:
            if task == 0:
                contexts, branches, _, cmap, templates, failures = gen.load_dev(tpi, cf)
                force_real, force_dense = gen.FORCES_REAL.copy(), gen.FORCES_DENSE.copy()
            else:
                readiness = taskgen.readiness(task, TASK_FROZEN[task])
                if not readiness["ready"]:
                    raise RuntimeError(f"task{task} DEV incomplete: {json.dumps(readiness)}")
                contexts, branches, _, cmap, templates, failures = taskgen.load_dev(task, TASK_FROZEN[task], tpi, cf)
                force_real, force_dense = taskgen.TASK_REAL_FORCES[task], taskgen.TASK_DENSE_FORCES[task]
            if failures:
                raise RuntimeError(f"task{task} DEV audit failed: {'; '.join(failures)}")
            gen.FORCES_REAL, gen.FORCES_DENSE = force_real, force_dense
            pred = predict_dev_task(task, tpi, cf, models, templates, norm, force_dense)
            real, real_front = gen.real_curves(branches)
            prob = gen.probability_metrics(pred, real)
            shape = gen.shape_metrics(pred)
            fdetail, front = gen.frontier_metrics(pred, real_front)
            summary = prob[prob.aggregation == "ENSEMBLE"].merge(front, on="model").merge(
                shape[shape.context_id == "__AGGREGATE__"][["model", "context_monotonic", "safe_to_unsafe_reversals", "nonmonotonic_steps"]],
                on="model",
            )
            summary["task"] = task
            summary["DEV_contexts"] = len(contexts)
            summary["DEV_branches"] = len(branches)
            summary["scope"] = "WITHIN_TASK_HELDOUT_ROOT_NOT_UNSEEN_TASK"
            summary["eligible_for_selection"] = False
            all_summaries.append(summary)
            all_predictions.append(pred)
            fdetail["task"] = task
            all_frontiers.append(fdetail)
    finally:
        gen.MODELS = previous
        gen.FORCES_REAL = np.round(np.arange(3.0, 5.0001, 0.25), 2)
        gen.FORCES_DENSE = np.round(np.arange(3.0, 5.0001, 0.05), 2)
    summary = pd.concat(all_summaries, ignore_index=True)
    summary.to_csv(out / "POOLED_MATCHED_DEV_TASKWISE_SUMMARY.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(out / "POOLED_MATCHED_DEV_PREDICTIONS.csv", index=False)
    pd.concat(all_frontiers, ignore_index=True).to_csv(out / "POOLED_MATCHED_DEV_FRONTIERS.csv", index=False)
    early.write_json(out / "POOLED_MATCHED_DEV_STATUS.json", {
        "status": "COMPLETE_RETROSPECTIVE_WITHIN_TASK_HELDOUT_ROOT_EVALUATION",
        "not_unseen_task_generalization": True, "selection_use_forbidden": True,
        "taskwise_rows": summary.to_dict("records"),
    })
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "cv", "dev"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.phase == "train":
        run_train(args.out.resolve())
    elif args.phase == "cv":
        run_cv(args.out.resolve())
    else:
        run_dev(args.out.resolve())


if __name__ == "__main__":
    main()
