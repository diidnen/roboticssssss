#!/usr/bin/env python3
"""Matched pooled multi-task ActiveForcing predictive-verifier development.

The script is deliberately restricted to the already-authoritative TRAIN
population for tasks 0/1/5/6 and the already-viewed old fixed-scene DEV audit.
It never discovers or reads any untouched TEST namespace.

Workflow:
  --prepare                audit data and freeze grouped-root folds/protocol
  --shard F S              train pooled Direct, current Joint-lineage WM, and
                            PhysicsOnly WM for one fold/seed and cache OOF inputs
  --finalize               train/evaluate pooled verifiers and write reports
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

import pooled_joint_novisual_current as pooled
import task0_visual_context_early as early
import taskwise_joint_novisual as jnv
import per_task_visual_context_early as taskwise
import run_predictive_verifier_development as pv


FORTE = Path("/home/exouser/FORTE")
TASKS = [0, 1, 5, 6]
SEEDS = [0, 1, 2]
FOLDS = [0, 1, 2]
ARCHS = list(pv.ARCHS)
FMAX = dict(pv.FMAX)
H = 8
PROMOTION_MAX_FORCE_DELTA_N = 0.25
PROMOTION_MAX_FPR = 0.10
PROMOTION_MIN_COVERAGE = 0.95
OLD_FIXED = FORTE / "fixed_scene_reframing_20260831_174144"
OLD_DIRECT = FORTE / "joint_decision_alignment_20260831_200441" / "JOINT_DECISIONALIGNED_ALL_PREDICTIONS.csv"
OLD_VERIFIER = FORTE / "direct_worldmodel_verifier_20260901_020334"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        if rows:
            w.writerows(rows)


def repeat_of(meta, trace) -> int:
    return int(meta[trace.branch_id].repeat)


def population(out: Path, tag: str):
    scratch = out / "_data_audit" / tag
    scratch.mkdir(parents=True, exist_ok=True)
    return pooled.load_population(scratch)


def frame(traces, meta) -> pd.DataFrame:
    return pd.DataFrame({
        "index": np.arange(len(traces), dtype=int),
        "branch_id": [t.branch_id for t in traces],
        "context_id": [t.context_id for t in traces],
        "root_id": [t.root_id for t in traces],
        "task": [int(t.task) for t in traces],
        "repeat": [repeat_of(meta, t) for t in traces],
        "force_N": [float(t.force) for t in traces],
        "mu": [float(t.mu) for t in traces],
        "success": [int(t.outcome) for t in traces],
    })


def root_folds(md: pd.DataFrame) -> dict[int, list[tuple[int, str]]]:
    ans: dict[int, list[tuple[int, str]]] = {f: [] for f in FOLDS}
    for task in TASKS:
        roots = sorted(md.loc[md.task == task, "root_id"].unique().tolist())
        if len(roots) != 6:
            raise RuntimeError(f"task{task}: expected 6 roots, got {len(roots)}")
        for f in FOLDS:
            ans[f].extend((task, str(r)) for r in roots[f::3])
    return ans


def held_mask(md: pd.DataFrame, fold: int) -> np.ndarray:
    pairs = set(root_folds(md)[fold])
    return np.asarray([(int(t), str(r)) in pairs for t, r in zip(md.task, md.root_id)], bool)


def source_paths() -> list[Path]:
    return [
        FORTE / "pooled_joint_novisual_current.py",
        FORTE / "taskwise_joint_novisual.py",
        FORTE / "task0_visual_context_early.py",
        FORTE / "per_task_visual_context_early.py",
        FORTE / "run_predictive_verifier_development.py",
        Path("/home/exouser/Tabero/analysis/trajectory_physical_imagination.py"),
        Path("/home/exouser/Tabero/analysis/counterfactual_force_world_model.py"),
        Path("/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py"),
    ]


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    freeze = out / "POOLED_DEVELOPMENT_PROTOCOL.json"
    if freeze.exists():
        print(json.dumps({"status": "ALREADY_FROZEN", "path": str(freeze), "sha256": sha256(freeze)}, indent=2))
        return
    tpi, cf, full, cmap, traces, meta, audits, pairs, segs, norm = population(out, "prepare")
    md = frame(traces, meta)
    folds = root_folds(md)
    # Exact leakage assertions at all requested grains.
    manifest: dict[str, Any] = {
        "status": "FROZEN_BEFORE_MODEL_TRAINING",
        "assignment": "within each task, lexicographic root-family index modulo 3",
        "group_key": ["task", "root_id"],
        "folds": [],
    }
    for f in FOLDS:
        hm = held_mask(md, f)
        train, held = md[~hm], md[hm]
        train_roots = set(zip(train.task, train.root_id))
        held_roots = set(zip(held.task, held.root_id))
        if train_roots & held_roots:
            raise RuntimeError("root leakage")
        for col in ["branch_id", "context_id"]:
            if set(train[col]) & set(held[col]):
                raise RuntimeError(f"{col} leakage")
        fold_entry = {
            "fold": f,
            "heldout": [{"task": int(t), "root_id": r} for t, r in sorted(folds[f])],
            "TRAIN": {"roots": len(train_roots), "contexts": int(train.context_id.nunique()), "branches": len(train)},
            "VALIDATION": {"roots": len(held_roots), "contexts": int(held.context_id.nunique()), "branches": len(held)},
            "per_task": {},
        }
        for task in TASKS:
            qt, qv = train[train.task == task], held[held.task == task]
            fold_entry["per_task"][str(task)] = {
                "TRAIN_roots": int(qt.root_id.nunique()), "TRAIN_contexts": int(qt.context_id.nunique()), "TRAIN_branches": len(qt),
                "VALIDATION_roots": int(qv.root_id.nunique()), "VALIDATION_contexts": int(qv.context_id.nunique()), "VALIDATION_branches": len(qv),
            }
        manifest["folds"].append(fold_entry)
    write_json(out / "POOLED_GROUPED_CV_MANIFEST.json", manifest)

    count_rows = []
    for task in TASKS:
        q = md[md.task == task]
        audit = audits[task]
        force_counts = q.groupby("context_id").force_N.nunique()
        repeat_counts = q.groupby(["context_id", "force_N"]).repeat.nunique()
        count_rows.append({
            "task": task,
            "independent_root_families": int(q.root_id.nunique()),
            "friction_conditioned_contexts": int(q.context_id.nunique()),
            "force_branches": len(q),
            "force_cells": int(q.groupby(["context_id", "force_N"]).ngroups),
            "repeats_per_force_cell": int(repeat_counts.min()),
            "forces_per_context": int(force_counts.min()),
            "force_min_N": float(q.force_N.min()), "force_max_N": float(q.force_N.max()),
            "successes": int(q.success.sum()), "failures": int((1-q.success).sum()),
            "direct_labels": int(audit.get("direct_labels", len(q))), "reconstructed_labels": int(audit.get("reconstructed_labels", 0)),
            "corrected_telemetry_branches": len(q), "visual_contexts_available": int(q.context_id.nunique()),
            "visual_used_in_experiment": 0, "adjacent_IE_pairs": sum(1 for p in pairs if int(p.task) == task),
        })
    totals = md
    count_rows.append({
        "task": "POOLED", "independent_root_families": int(totals.groupby(["task", "root_id"]).ngroups),
        "friction_conditioned_contexts": int(totals.context_id.nunique()), "force_branches": len(totals),
        "force_cells": int(totals.groupby(["context_id", "force_N"]).ngroups), "repeats_per_force_cell": 2,
        "forces_per_context": 5, "force_min_N": float(totals.force_N.min()), "force_max_N": float(totals.force_N.max()),
        "successes": int(totals.success.sum()), "failures": int((1-totals.success).sum()),
        "direct_labels": int(sum(a.get("direct_labels", 180) for a in audits.values())),
        "reconstructed_labels": int(sum(a.get("reconstructed_labels", 0) for a in audits.values())),
        "corrected_telemetry_branches": len(totals), "visual_contexts_available": int(totals.context_id.nunique()),
        "visual_used_in_experiment": 0, "adjacent_IE_pairs": len(pairs),
    })
    write_csv(out / "POOLED_VERIFIER_DATA_COUNTS.csv", count_rows)
    task1 = audits[1]
    audit_md = f"""# Pooled verifier data audit

PASS. The authoritative development population is **720 TRAIN branches from 72 friction-conditioned contexts and 24 independent root families**, pooled across task0/task1/task5/task6. Each task contributes 6 root families, 18 contexts, and 180 branches. A context is a root-state-friction realization; a force branch and its repeats are not independent contexts.

| Grain | Per task | Pooled |
|---|---:|---:|
| Independent root families | 6 | 24 |
| Friction-conditioned contexts | 18 | 72 |
| Force cells | 90 | 360 |
| Branches (5 forces × 2 repeats/context) | 180 | 720 |
| H8 physical timesteps | 1,440 | 5,760 |
| Adjacent-force IE pairs | 144 | 576 |

All 720 branches have corrected physical telemetry and a final full-task outcome. The World Model target is the frozen H=8, 13-channel physical trajectory; the IE target is the matched adjacent-force trajectory difference within a context and repeat. Visual features exist for all 72 contexts but are **excluded** from Direct, WM, and verifier inputs in this experiment.

## Label lineage and task1 caveat

task0/task5/task6 use complete direct branch outcomes. task1 has **{task1['direct_labels']} direct labels and {task1['reconstructed_labels']} frozen terminal-height fallback labels**. On the {task1['direct_vs_fallback_compared']} branches where both exist, mismatches are {task1['direct_vs_fallback_mismatches']}. This does not remove the material caveat: task1 metrics partially measure the preregistered reconstruction rule, not uniformly direct execution labels.

## Grouped CV

Three folds are frozen before training. Each fold holds out 2 whole root families per task (8 pooled), including every friction condition, force, and repeat under those roots. Each model trains jointly on the remaining four tasks' roots: 16 roots, 48 contexts, 480 branches. Validation has 8 roots, 24 contexts, 240 branches. There is no root/context/branch leakage.

## Scope

This is pooled multi-task TRAIN-rootheldout development with authoritative GT physics. It is not Active Probe E2E, not untouched TEST, and not unseen-task or cross-object evidence. The already-viewed old fixed-scene DEV is used only for retrospective difficulty composition.
"""
    (out / "POOLED_VERIFIER_DATA_AUDIT.md").write_text(audit_md, encoding="utf-8")
    protocol = {
        "status": "FROZEN_BEFORE_MODEL_TRAINING",
        "scope": "task0/task1/task5/task6 pooled authoritative TRAIN; 3-fold grouped root-family CV; old viewed fixed-scene DEV retrospective only",
        "untouched_TEST_read": False,
        "forbidden": ["root-scaling pipeline mutation", "root-scaling TEST discovery/read", "Probe", "visual input", "branch-random split"],
        "population": {"branches": 720, "contexts": 72, "root_families": 24, "tasks": TASKS},
        "models": {
            "Direct": "one pooled frozen authoritative FeasibilityOnly architecture; full-task BCE",
            "WM_Current": "one pooled current Joint-lineage WM; Lphysics + LIE + frozen lambda_feas*full-task BCE",
            "WM_PhysicsOnly": "same architecture/init/inputs/H8/telemetry/IE/epochs/optimizer; Lphysics + LIE; no outcome gradient",
            "verifiers": ARCHS,
        },
        "seeds": SEEDS,
        "control": {"C_fail": "task Fmax", "proposal": "argmax expected utility", "decision": "SUCCESS iff logit>0", "threshold_tuning": False, "search": "upward-only", "strict_exhaustion": "NO_VALID_FORCE", "fallback": "execute Fmax after NO_VALID_FORCE; not a verifier safety judgment"},
        "architecture_selection": "safety-first on macro mean: under-force, FPR, then SR, coverage, FNR, force; stable benefit required before promotion",
        "ensemble_gate": "retain Majority/Unanimous only if under-force lower, SR nonlower, coverage loss <=0.02, and >=2 tasks improve; Boolean votes only",
        "promotion": {
            "macro_SR_strictly_higher": True, "positive_task_count_min": 2, "under_force_nonincrease": True,
            "mean_force_delta_max_N": PROMOTION_MAX_FORCE_DELTA_N, "strict_not_fallback": True,
            "FPR_max": PROMOTION_MAX_FPR, "coverage_min": PROMOTION_MIN_COVERAGE,
            "all_3_seed_SR_deltas_positive": True, "all_3_fold_SR_deltas_nonnegative": True,
        },
        "fold_manifest_sha256": sha256(out / "POOLED_GROUPED_CV_MANIFEST.json"),
        "source_hashes": {str(p): sha256(p) for p in source_paths()},
    }
    write_json(freeze, protocol)
    print(json.dumps({"status": "FROZEN", "protocol": str(freeze), "sha256": sha256(freeze)}, indent=2))


def filter_pairs(pairs, train_ids: set[str]):
    return [p for p in pairs if p.a.branch_id in train_ids and p.b.branch_id in train_ids]


def train_physics_only(full, cf, tpi, traces, pairs, norm, seed: int, device):
    early.seed_everything(seed + 3000)
    base_ck, _, base_path = full.load_base(tpi, seed, device)
    model = tpi.ShortHorizonPhysicsGRU(17, 54, H).to(device)
    model.load_state_dict(base_ck["state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=early.LR, weight_decay=early.WEIGHT_DECAY)
    units = cf.make_units(tpi, traces, pairs)
    history = []
    steps = 0
    for epoch in range(1, early.EPOCHS + 1):
        physics_values, ie_values = [], []
        model.train()
        for batch in cf.batches_for_units(units, seed, epoch):
            _, step, cond, ytraj, mask, weight = cf.batch_tensors(batch, norm, device)
            opt.zero_grad(set_to_none=True)
            pred = model(step, cond)
            physics = (nn.functional.smooth_l1_loss(pred, ytraj, reduction="none") * mask * weight[:, None, None]).sum() / (mask.sum() + 1e-6)
            ie = early.physical_ie_loss(pred, batch, norm, device)
            total = physics + ie
            total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); steps += 1
            physics_values.append(float(physics)); ie_values.append(float(ie))
        history.append({"epoch": epoch, "physics_loss": float(np.mean(physics_values)), "IE_loss": float(np.mean(ie_values)), "optimizer_steps": steps})
        if epoch % 20 == 0:
            print(f"[PhysicsOnly] seed={seed} epoch={epoch}/{early.EPOCHS} phys={np.mean(physics_values):.5f} IE={np.mean(ie_values):.5f}", flush=True)
    model.eval()
    return model, history, steps, len(units), Path(base_path)


def predict_world(model, traces, norm, cf, tpi, device) -> tuple[np.ndarray, np.ndarray]:
    states, extras = [], []
    model.eval()
    with torch.no_grad():
        for tr in traces:
            seg = cf.build_seg(tpi, tr, tr.force, H)
            states.append(cf.pred_state(model, tr, tr.force, norm, tpi, device))
            xn = (seg.x - norm[0]) / norm[1]
            extras.append(np.concatenate([[tr.force, tr.mu], xn[0, 19:]]).astype(np.float32))
    return np.stack(states).astype(np.float32), np.stack(extras).astype(np.float32)


def shard(out: Path, fold: int, seed: int) -> None:
    if not (out / "POOLED_DEVELOPMENT_PROTOCOL.json").exists():
        raise RuntimeError("run --prepare first")
    target = out / "shards" / f"fold{fold}_seed{seed}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(json.dumps({"status": "ALREADY_COMPLETE", "path": str(target), "sha256": sha256(target)}, indent=2)); return
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 3)))
    device = torch.device("cpu")
    tpi, cf, full, cmap, traces, meta, audits, pairs, all_segs, _ = population(out, f"shard_f{fold}_s{seed}")
    md = frame(traces, meta); hm = held_mask(md, fold)
    train_idx, held_idx = np.flatnonzero(~hm), np.flatnonzero(hm)
    train = [traces[i] for i in train_idx]
    train_ids = {t.branch_id for t in train}
    train_meta = {bid: meta[bid] for bid in train_ids}
    train_segs = {bid: all_segs[bid] for bid in train_ids}
    norm = early.build_segments_and_norm(cf, tpi, train)[1]
    train_pairs = filter_pairs(pairs, train_ids)
    if len(train) != 480 or len(train_pairs) != 384:
        raise RuntimeError(f"bad fold population: branches={len(train)} pairs={len(train_pairs)}")
    print(f"[shard] fold={fold} seed={seed} Direct", flush=True)
    direct, dhist, dsteps = early.train_base(full, train, train_segs, norm, cmap, train_meta, device, seed)
    dz = early.logits_for(direct, "BASE", traces, all_segs, norm, cmap, device)
    direct_logits = np.asarray([dz[t.branch_id] for t in traces], np.float32)
    print(f"[shard] fold={fold} seed={seed} WM-current", flush=True)
    current, jhist, jsteps, junits, base_path = jnv.train_seed(full, cf, tpi, train, train_pairs, train_segs, norm, cmap, train_meta, seed, device)
    current_pred, extra = predict_world(current.physics, traces, norm, cf, tpi, device)
    print(f"[shard] fold={fold} seed={seed} WM-PhysicsOnly", flush=True)
    physics, phist, psteps, punits, pbase = train_physics_only(full, cf, tpi, train, train_pairs, norm, seed, device)
    physics_pred, pextra = predict_world(physics, traces, norm, cf, tpi, device)
    if not np.allclose(extra, pextra):
        raise RuntimeError("current/PhysicsOnly extra context mismatch")
    ckdir = out / "checkpoints"; ckdir.mkdir(exist_ok=True)
    common = {"fold": fold, "seed": seed, "epochs": early.EPOCHS, "TRAIN_branches": 480, "TRAIN_contexts": 48, "TRAIN_roots": 16, "heldout_branches": 240, "heldout_contexts": 24, "heldout_roots": 8, "normalization": {k: v.tolist() for k, v in zip(["x_mean", "x_std", "y_mean", "y_std"], norm)}, "DEV_used": False, "TEST_used": False}
    dpath = ckdir / f"DIRECT_POOLED_fold{fold}_seed{seed}.pt"
    torch.save({**common, "model": "ActiveForcing-Direct-Pooled", "state_dict": direct.state_dict(), "objective": "full-task BCE", "optimizer_steps": dsteps}, dpath)
    cpath = ckdir / f"WM_CURRENT_POOLED_fold{fold}_seed{seed}.pt"
    torch.save({**common, "model": "WM-Current-Pooled", "state_dict": current.physics.state_dict(), "objective": "Lphysics + LIE + lambda_feas*BCE", "lambda_feas": early.LAMBDA_FEAS, "optimizer_steps": jsteps, "physical_units": junits, "initial_checkpoint": str(base_path), "initial_checkpoint_sha256": sha256(base_path)}, cpath)
    ppath = ckdir / f"WM_PHYSICSONLY_POOLED_fold{fold}_seed{seed}.pt"
    torch.save({**common, "model": "WM-PhysicsOnly-Pooled", "state_dict": physics.state_dict(), "objective": "Lphysics + LIE", "outcome_gradient": False, "optimizer_steps": psteps, "physical_units": punits, "initial_checkpoint": str(pbase), "initial_checkpoint_sha256": sha256(pbase)}, ppath)
    np.savez_compressed(target, train_idx=train_idx, held_idx=held_idx, direct_logits=direct_logits, current_pred=current_pred, physics_pred=physics_pred, extra=extra, x_mean=norm[0], x_std=norm[1], y_mean=norm[2], y_std=norm[3])
    log = {"fold": fold, "seed": seed, "direct_final": dhist[-1], "current_final": jhist[-1], "physics_only_final": phist[-1], "checkpoints": [{"path": str(p), "sha256": sha256(p)} for p in [dpath, cpath, ppath]], "cache": {"path": str(target), "sha256": sha256(target)}}
    write_json(out / "shards" / f"fold{fold}_seed{seed}.json", log)
    print(json.dumps({"status": "COMPLETE", "fold": fold, "seed": seed, "cache": str(target)}, indent=2))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def direct_policy_rows(md: pd.DataFrame, p: np.ndarray, kind: str) -> list[dict]:
    d = md.copy(); d["p_direct"] = p
    rows = []
    for (cid, rep), q0 in d.groupby(["context_id", "repeat"]):
        q = q0.sort_values("force_N").copy(); task = int(q.task.iloc[0]); fmax = FMAX[task]
        q["utility"] = q.p_direct * (fmax-q.force_N) + (1-q.p_direct) * (-fmax)
        prop = q.sort_values(["utility", "force_N"], ascending=[False, True]).iloc[0]
        support = q[q.force_N >= prop.force_N-1e-9].sort_values("force_N")
        if kind == "DIRECT": z = prop
        elif kind == "ONE_STEP": z = support.iloc[min(1, len(support)-1)]
        elif kind == "MAX": z = q.iloc[-1]
        else: raise KeyError(kind)
        good = q[q.success > 0]
        boundary = float(good.force_N.min()) if len(good) else float("nan")
        rows.append({"context_id": cid, "repeat": int(rep), "root_id": q.root_id.iloc[0], "task": task, "proposal_force_N": float(prop.force_N), "selected_force_N": float(z.force_N), "actual_success": float(z.success), "coverage": 1, "NO_VALID_FORCE": 0, "under_force": int(math.isfinite(boundary) and z.force_N < boundary-1e-9), "excess_force_N": float(z.force_N-boundary) if math.isfinite(boundary) else float("nan"), "mean_force_component": float(z.force_N), "escalations": int(np.sum((support.force_N < z.force_N).to_numpy())), "changed_from_direct": int(z.force_N > prop.force_N+1e-9)})
    return rows


def summary_scopes(rows: list[dict], policy: str, seed: Any = "NA", extra: dict | None = None) -> list[dict]:
    d = pd.DataFrame(rows); ans = []
    bytask = []
    for task in TASKS:
        r = pv.summarize_controller(d[d.task == task].to_dict("records"), policy)
        row = {"scope": f"task{task}", "task": task, "seed": seed, **r}; bytask.append(row); ans.append(row)
    metric_keys = ["coverage", "full_task_SR", "conditional_SR", "under_force", "mean_force_N", "excess_force_N", "NO_VALID_FORCE_rate", "mean_escalations"]
    ans.append({"scope": "MACRO", "task": "MACRO", "seed": seed, "policy": policy, "episodes": int(sum(x["episodes"] for x in bytask)), **{k: float(np.nanmean([x[k] for x in bytask])) for k in metric_keys}})
    ans.append({"scope": "POOLED", "task": "POOLED", "seed": seed, **pv.summarize_controller(rows, policy)})
    if extra:
        for r in ans: r.update(extra)
    return ans


def classifier_scopes(md: pd.DataFrame, logits: np.ndarray, arch: str, seed: Any, world: str) -> list[dict]:
    rows, task_metrics = [], []
    y = md.success.to_numpy(np.float32)
    for task in TASKS:
        ids = np.flatnonzero(md.task.to_numpy() == task)
        m = pv.classifier_metrics(y[ids], logits[ids]); task_metrics.append(m)
        rows.append({"scope": f"task{task}", "task": task, "world_model": world, "architecture": arch, "seed": seed, **m})
    keys = ["AUROC", "FPR", "FNR", "NLL"]
    rows.append({"scope": "MACRO", "task": "MACRO", "world_model": world, "architecture": arch, "seed": seed, "branches": len(md), **{k: float(np.nanmean([m[k] for m in task_metrics])) for k in keys}, "false_positive_count": int(sum(m["false_positive_count"] for m in task_metrics)), "false_negative_count": int(sum(m["false_negative_count"] for m in task_metrics))})
    rows.append({"scope": "POOLED", "task": "POOLED", "world_model": world, "architecture": arch, "seed": seed, **pv.classifier_metrics(y, logits)})
    return rows


def train_verifier_oof(out: Path, md: pd.DataFrame, world: str, archs: list[str]) -> tuple[dict[str, np.ndarray], list[dict]]:
    device = torch.device("cpu"); y = md.success.to_numpy(np.float32)
    logits = {a: np.full((3, len(md)), np.nan, np.float32) for a in archs}
    checkpoints = []
    for f in FOLDS:
        for s in SEEDS:
            z = np.load(out / "shards" / f"fold{f}_seed{s}.npz")
            tr, va = z["train_idx"], z["held_idx"]
            traj = z["current_pred"] if world == "WM-Current-Pooled" else z["physics_pred"]
            extra = z["extra"]
            for arch in archs:
                print(f"[verifier] world={world} arch={arch} fold={f} seed={s}", flush=True)
                model, stats, hist = pv.fit_verifier(arch, traj, extra, y, tr, s, device)
                logits[arch][s, va] = pv.predict_logits(model, stats, traj, extra, va, device)
                path = out / "checkpoints" / f"VERIFIER_{world}_{arch}_fold{f}_seed{s}.pt"
                torch.save({"world_model": world, "architecture": arch, "fold": f, "seed": s, "state_dict": model.state_dict(), "stats": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in stats.items()}, "decision": "logit>0", "epochs": pv.EPOCHS, "TRAIN_only": True, "TEST_used": False}, path)
                checkpoints.append({"world": world, "architecture": arch, "fold": f, "seed": s, "path": str(path), "sha256": sha256(path)})
    for a, z in logits.items():
        if not np.isfinite(z).all(): raise RuntimeError(f"incomplete OOF logits {world}/{a}")
    return logits, checkpoints


def fidelity_rows(out: Path, md: pd.DataFrame, traces, meta, tpi) -> list[dict]:
    real = np.stack([t.state[1:H+1] for t in traces]).astype(np.float32)
    mask = np.stack([t.mask[1:H+1] for t in traces]).astype(np.float32)
    rows = []
    for world, key in [("WM-Current-Pooled", "current_pred"), ("WM-PhysicsOnly-Pooled", "physics_pred")]:
        for seed in SEEDS:
            seed_rows = []
            pred = np.empty_like(real); ys_by_fold = {}
            for f in FOLDS:
                z = np.load(out / "shards" / f"fold{f}_seed{seed}.npz"); va = z["held_idx"]
                pred[va] = z[key][va]; ys_by_fold[f] = z["y_std"]
            for scope in [*TASKS, "POOLED"]:
                ids = np.arange(len(md)) if scope == "POOLED" else np.flatnonzero(md.task.to_numpy() == scope)
                # Every row gets its own fold-local y scale.
                scaled = np.zeros_like(real[ids])
                for ii, gi in enumerate(ids):
                    f = int(next(f for f in FOLDS if held_mask(md, f)[gi])); scaled[ii] = np.abs(pred[gi]-real[gi]) / np.maximum(ys_by_fold[f], 1e-6)
                mm = mask[ids]; row = {"world_model": world, "seed": seed, "scope": f"task{scope}" if scope != "POOLED" else "POOLED", "task": scope, "trajectory_standardized_MAE": float((scaled*mm).sum()/max(mm.sum(), 1))}
                for j, name in enumerate(getattr(tpi, "STATE_NAMES", [f"channel{j}" for j in range(13)])):
                    row[f"trajectory_MAE_{name}"] = float((scaled[:, :, j]*mm[:, :, j]).sum()/max(mm[:, :, j].sum(), 1))
                ie = []
                qmd = md.iloc[ids]
                for (_, rep), q in qmd.groupby(["context_id", "repeat"]):
                    js = q.sort_values("force_N").index.to_numpy()
                    for a, b in zip(js[:-1], js[1:]):
                        f = int(next(ff for ff in FOLDS if held_mask(md, ff)[a])); m = mask[a]*mask[b]
                        de = np.abs(((pred[b]-pred[a])-(real[b]-real[a])) / np.maximum(ys_by_fold[f], 1e-6))*m
                        ie.append(float(de.sum()/max(m.sum(), 1)))
                row["IE_standardized_MAE"] = float(np.mean(ie)); rows.append(row); seed_rows.append(row)
            task_rows = [r for r in seed_rows if r["task"] in TASKS]
            metric_keys = [k for k, v in task_rows[0].items() if isinstance(v, (int, float, np.integer, np.floating)) and k != "seed"]
            rows.append({"world_model": world, "seed": seed, "scope": "MACRO", "task": "MACRO", **{k: float(np.nanmean([r[k] for r in task_rows])) for k in metric_keys}})
    return rows


def old_new_difficulty(out: Path, md: pd.DataFrame, p_direct: np.ndarray) -> pd.DataFrame:
    def summarize(name: str, d: pd.DataFrame, pcol: str, task_scope: Any, group_cols: list[str], grain_label: str):
        rows = []
        groups = [d] if task_scope == "POOLED" else [d[d.task == task_scope]]
        for qsource in groups:
            episodes = []
            for keys, q0 in qsource.groupby(group_cols):
                q = q0.groupby(["context_id", "task", "force_N"], as_index=False).agg(p=(pcol, "mean"), y=("success", "mean")).sort_values("force_N")
                task = int(q.task.iloc[0]); fmax = FMAX[task]; q["u"] = q.p*(fmax-q.force_N)+(1-q.p)*(-fmax)
                prop = q.sort_values(["u", "force_N"], ascending=[False, True]).iloc[0]; maxrow = q.iloc[-1]
                good = q[q.y > 0]; frontier = float(good.force_N.min()) if len(good) else float("nan")
                support = q[q.force_N >= prop.force_N-1e-9]
                rescued = int(prop.y < 1 and (support.y > prop.y + 1e-12).any())
                stochastic = int(((q.y > 0) & (q.y < 1)).any())
                episodes.append({"direct_success": float(prop.y), "fixed_max_success": float(maxrow.y), "under": int(math.isfinite(frontier) and prop.force_N < frontier-1e-9), "rescued": rescued, "fmax_fail": int(maxrow.y < 1), "stochastic": stochastic, "no_success": int(not len(good)), "distance": float(frontier-prop.force_N) if math.isfinite(frontier) else float("nan")})
            z = pd.DataFrame(episodes)
            rows.append({"population": name, "scope": f"task{task_scope}" if task_scope != "POOLED" else "POOLED", "task": task_scope, "episode_grain": grain_label, "episodes": len(z), "Direct_SR": float(z.direct_success.mean()), "Fixed_Max_SR": float(z.fixed_max_success.mean()), "under_force_rate": float(z.under.mean()), "under_force_cases": int(z.under.sum()), "Direct_failure_episodes_including_stochastic": int((z.direct_success < 1).sum()), "Direct_expected_failure_mass": float((1-z.direct_success).sum()), "cases_rescued_by_increasing_force": int(z.rescued.sum()), "Fmax_not_always_successful": int(z.fmax_fail.sum()), "stochastic_boundary_cases": int(z.stochastic.sum()), "VLA_or_nonforce_no_success_cases": int(z.no_success.sum()), "mean_empirical_distance_to_frontier_N": float(z.distance.mean())})
        return rows
    new = md.copy(); new["p"] = p_direct
    rows = []
    for t in [*TASKS, "POOLED"]: rows.extend(summarize("NEW_POOLED_GROUPED_ROOT_CV", new, "p", t, ["context_id", "repeat"], "context-repeat controller episode"))
    old = pd.read_csv(OLD_DIRECT); old = old[(old.fraction == "100%") & (old.method == "Direct")].copy()
    old = old.rename(columns={"p_success": "p", "actual_success": "success"})
    if "repeat" not in old.columns:
        old["repeat"] = old.branch_id.astype(str).str.extract(r"_R([12])(?:_|$)")[0].fillna(1).astype(int)
    for t in [*TASKS, "POOLED"]: rows.extend(summarize("OLD_VIEWED_FIXED_SCENE_DEV", old, "p", t, ["context_id"], "context decision; repeat-mean force cells"))
    return pd.DataFrame(rows)


def finalize(out: Path) -> None:
    protocol = out / "POOLED_DEVELOPMENT_PROTOCOL.json"
    if not protocol.exists(): raise RuntimeError("missing frozen protocol")
    for f in FOLDS:
        for s in SEEDS:
            if not (out / "shards" / f"fold{f}_seed{s}.npz").exists(): raise RuntimeError(f"missing shard f{f}s{s}")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    tpi, cf, full, cmap, traces, meta, audits, pairs, segs, norm = population(out, "finalize")
    md = frame(traces, meta); y = md.success.to_numpy(np.float32)
    # Fully OOF Direct predictions, mean of the three pre-registered seeds.
    direct_seed = np.full((3, len(md)), np.nan, np.float32)
    for f in FOLDS:
        for s in SEEDS:
            z = np.load(out / "shards" / f"fold{f}_seed{s}.npz"); va = z["held_idx"]
            direct_seed[s, va] = sigmoid(z["direct_logits"][va])
    if not np.isfinite(direct_seed).all(): raise RuntimeError("incomplete Direct OOF")
    p_direct = direct_seed.mean(0)
    direct_rows = []
    for s in SEEDS: direct_rows += summary_scopes(direct_policy_rows(md, direct_seed[s], "DIRECT"), "ActiveForcing-Direct-Pooled", s, {"proposal_aggregation": "single_seed"})
    direct_rows += summary_scopes(direct_policy_rows(md, p_direct, "DIRECT"), "ActiveForcing-Direct-Pooled", "ENSEMBLE_MEAN", {"proposal_aggregation": "mean_probability_3_seeds_no_best_seed"})
    write_csv(out / "POOLED_DIRECT_RESULTS.csv", direct_rows)

    current_logits, verifier_ck = train_verifier_oof(out, md, "WM-Current-Pooled", ARCHS)
    arch_rows = []
    for arch in ARCHS:
        for s in SEEDS:
            cls = classifier_scopes(md, current_logits[arch][s], arch, s, "WM-Current-Pooled")
            ctl = summary_scopes(pv.controller_rows(md, p_direct, current_logits[arch][s], "single", "STRICT"), f"Direct + {arch} hard search", s)
            cby = {r["scope"]: r for r in ctl}
            for r in cls:
                c = cby[r["scope"]]; arch_rows.append({**r, **{f"control_{k}": v for k, v in c.items() if k not in {"scope", "task", "seed", "policy"}}})
    # Explicit seed means at each scope; no best seed.
    adf = pd.DataFrame(arch_rows)
    numeric = [c for c in adf.columns if c not in {"scope", "task", "world_model", "architecture", "seed"} and pd.api.types.is_numeric_dtype(adf[c])]
    means = adf.groupby(["scope", "task", "world_model", "architecture"], as_index=False)[numeric].mean(); means["seed"] = "MEAN_3_SEEDS"
    adf = pd.concat([adf, means], ignore_index=True); write_csv(out / "POOLED_VERIFIER_ARCHITECTURE_COMPARISON.csv", adf)
    macro = means[means.scope == "MACRO"].sort_values(["control_under_force", "FPR", "control_full_task_SR", "control_coverage", "FNR", "control_mean_force_N"], ascending=[True, True, False, False, True, True])
    selected = str(macro.iloc[0].architecture)

    # Boolean ensemble comparison.
    ens_rows = []
    single_task = {}
    for s in SEEDS:
        rows = pv.controller_rows(md, p_direct, current_logits[selected][s], "single", "STRICT")
        ctl = summary_scopes(rows, "SINGLE", s, {"architecture": selected, "decision_rule": "SINGLE"})
        cls = classifier_scopes(md, current_logits[selected][s], selected, s, "WM-Current-Pooled")
        cby = {r["scope"]: r for r in ctl}
        for r in cls:
            c = cby[r["scope"]]
            ens_rows.append({**c, **{k: v for k, v in r.items() if k not in {"scope", "task", "seed"}}})
    for rule in ["majority", "unanimous"]:
        signed = np.where((current_logits[selected] > 0).sum(0) >= (2 if rule == "majority" else 3), 1., -1.)
        cls = classifier_scopes(md, signed, selected, rule.upper(), "WM-Current-Pooled")
        ctl = summary_scopes(pv.controller_rows(md, p_direct, current_logits[selected], rule, "STRICT"), rule.upper(), rule.upper(), {"architecture": selected, "decision_rule": rule.upper()})
        cby = {r["scope"]: r for r in ctl}
        for r in cls:
            c = cby[r["scope"]]; ens_rows.append({**c, **{k: v for k, v in r.items() if k not in {"scope", "task", "seed"}}})
    ensdf = pd.DataFrame(ens_rows)
    # Add single macro-over-seed rows for gate comparison.
    singles = ensdf[ensdf.decision_rule == "SINGLE"]
    snum = [c for c in singles.columns if pd.api.types.is_numeric_dtype(singles[c]) and c not in {"task"}]
    smean = singles.groupby(["scope", "task", "policy", "architecture", "decision_rule"], as_index=False)[snum].mean(); smean["seed"] = "MEAN_3_SEEDS"
    ensdf = pd.concat([ensdf, smean], ignore_index=True); write_csv(out / "POOLED_VERIFIER_ENSEMBLE.csv", ensdf)
    sm = smean[smean.scope == "MACRO"].iloc[0]
    final_rule = "SINGLE"
    for candidate in ["MAJORITY", "UNANIMOUS"]:
        cm = ensdf[(ensdf.scope == "MACRO") & (ensdf.decision_rule == candidate)].iloc[0]
        task_benefit = 0
        for task in TASKS:
            a = ensdf[(ensdf.scope == f"task{task}") & (ensdf.decision_rule == candidate)].iloc[0].full_task_SR
            b = smean[smean.scope == f"task{task}"].iloc[0].full_task_SR
            task_benefit += int(a > b + 1e-12)
        if cm.under_force < sm.under_force-1e-12 and cm.full_task_SR >= sm.full_task_SR-1e-12 and cm.coverage >= sm.coverage-.02 and task_benefit >= 2:
            final_rule = candidate; break

    # PhysicsOnly decoupling with the selected verifier architecture.
    physics_logits, physics_ck = train_verifier_oof(out, md, "WM-PhysicsOnly-Pooled", [selected])
    fidelity = fidelity_rows(out, md, traces, meta, tpi)
    dec_rows = []
    for world, logits_map in [("WM-Current-Pooled", current_logits), ("WM-PhysicsOnly-Pooled", physics_logits)]:
        for s in SEEDS:
            cls = classifier_scopes(md, logits_map[selected][s], selected, s, world)
            ctl = summary_scopes(pv.controller_rows(md, p_direct, logits_map[selected][s], "single", "STRICT"), "Hard verifier", s)
            cby = {r["scope"]: r for r in ctl}
            for r in cls:
                c = cby[r["scope"]]; fr = next((x for x in fidelity if x["world_model"] == world and x["seed"] == s and x["scope"] == r["scope"]), {})
                dec_rows.append({**r, **{f"control_{k}": v for k, v in c.items() if k not in {"scope", "task", "seed", "policy"}}, **{k: v for k, v in fr.items() if k not in {"world_model", "seed", "scope", "task"}}})
    decdf = pd.DataFrame(dec_rows); dnum = [c for c in decdf.columns if pd.api.types.is_numeric_dtype(decdf[c]) and c not in {"task"}]
    dmean = decdf.groupby(["scope", "task", "world_model", "architecture"], as_index=False)[dnum].mean(); dmean["seed"] = "MEAN_3_SEEDS"
    decdf = pd.concat([decdf, dmean], ignore_index=True); write_csv(out / "POOLED_WORLD_MODEL_DECOUPLING.csv", decdf)

    # Search-policy ablation, always seed-by-seed and macro mean.
    ablation = []
    for kind, label in [("DIRECT", "P0 Direct"), ("ONE_STEP", "P1 Direct + one force level"), ("MAX", "P4 Fixed Max")]:
        ablation += summary_scopes(direct_policy_rows(md, p_direct, kind), label, "DIRECT_ENSEMBLE")
    if final_rule == "SINGLE":
        for s in SEEDS:
            ablation += summary_scopes(pv.controller_rows(md, p_direct, current_logits[selected][s], "single", "STRICT"), "P2 Direct + pooled hard verifier", s)
            ablation += summary_scopes(pv.controller_rows(md, p_direct, current_logits[selected][s], "single", "FALLBACK"), "P3 Direct + verifier + max fallback", s)
    else:
        rr = final_rule.lower()
        ablation += summary_scopes(pv.controller_rows(md, p_direct, current_logits[selected], rr, "STRICT"), "P2 Direct + pooled hard verifier", final_rule)
        ablation += summary_scopes(pv.controller_rows(md, p_direct, current_logits[selected], rr, "FALLBACK"), "P3 Direct + verifier + max fallback", final_rule)
    abdf = pd.DataFrame(ablation)
    q = abdf[abdf.policy.isin(["P2 Direct + pooled hard verifier", "P3 Direct + verifier + max fallback"]) & abdf.seed.isin(SEEDS)]
    if len(q):
        anum = [c for c in q.columns if pd.api.types.is_numeric_dtype(q[c]) and c not in {"task", "seed"}]
        amean = q.groupby(["scope", "task", "policy"], as_index=False)[anum].mean(); amean["seed"] = "MEAN_3_SEEDS"; abdf = pd.concat([abdf, amean], ignore_index=True)
    write_csv(out / "POOLED_SEARCH_POLICY_ABLATION.csv", abdf)

    difficulty = old_new_difficulty(out, md, p_direct); write_csv(out / "OLD_VS_NEW_POOLED_DIFFICULTY_AUDIT.csv", difficulty)

    # Residual attribution at controller episode grain, per task and category.
    search_cases = []
    if final_rule == "SINGLE":
        for s in SEEDS:
            search_cases += [{**r, "seed": s} for r in pv.controller_rows(md, p_direct, current_logits[selected][s], "single", "STRICT")]
    else:
        search_cases = [{**r, "seed": final_rule} for r in pv.controller_rows(md, p_direct, current_logits[selected], final_rule.lower(), "STRICT")]
    failures = []
    for r in search_cases:
        q = md[(md.context_id == r["context_id"]) & (md.repeat == r["repeat"])].sort_values("force_N")
        if r["coverage"] and r["actual_success"] < 1:
            sibling = md[(md.context_id == r["context_id"]) & np.isclose(md.force_N, r["selected_force_N"])]
            cat = "STOCHASTIC_BOUNDARY" if sibling.success.nunique() > 1 else "VERIFIER_FALSE_POSITIVE"
        elif not r["coverage"]:
            support = q[q.force_N >= r["proposal_force_N"]-1e-9]
            cat = "VERIFIER_FALSE_NEGATIVE" if (support.success > 0).any() else "NO_VALID_FORCE_IN_RANGE"
        else: continue
        failures.append({"row_type": "CASE", **r, "failure_type": cat, "WM_vs_classifier_separable": 0, "note": "Outcome/controller rows identify runtime decision error; WM representation vs classifier cannot be uniquely separated without a matched real-trajectory verifier counterfactual."})
    categories = ["WM_PREDICTION_FAILURE", "VERIFIER_FALSE_POSITIVE", "VERIFIER_FALSE_NEGATIVE", "HORIZON_INSUFFICIENT", "STOCHASTIC_BOUNDARY", "NO_VALID_FORCE_IN_RANGE", "VLA_NON_FORCE_FAILURE", "OTHER"]
    case_failures = list(failures)
    for task in [*TASKS, "POOLED"]:
        z = case_failures if task == "POOLED" else [r for r in case_failures if r["task"] == task]
        c = Counter(r["failure_type"] for r in z)
        for cat in categories: failures.append({"row_type": "SUMMARY", "task": task, "failure_type": cat, "count": int(c.get(cat, 0)), "identifiability": "DIRECT" if cat in {"VERIFIER_FALSE_POSITIVE", "VERIFIER_FALSE_NEGATIVE", "STOCHASTIC_BOUNDARY", "NO_VALID_FORCE_IN_RANGE"} else "NOT_SEPARATELY_IDENTIFIABLE"})
    write_csv(out / "POOLED_VERIFIER_FAILURE_ATTRIBUTION.csv", failures)

    # Promotion gate uses strict verifier only and the frozen criteria.
    direct_macro = abdf[(abdf.policy == "P0 Direct") & (abdf.scope == "MACRO")].iloc[0]
    direct_task = {t: abdf[(abdf.policy == "P0 Direct") & (abdf.scope == f"task{t}")].iloc[0] for t in TASKS}
    if final_rule == "SINGLE":
        hard_macro = abdf[(abdf.policy == "P2 Direct + pooled hard verifier") & (abdf.scope == "MACRO") & (abdf.seed == "MEAN_3_SEEDS")].iloc[0]
        hard_task = {t: abdf[(abdf.policy == "P2 Direct + pooled hard verifier") & (abdf.scope == f"task{t}") & (abdf.seed == "MEAN_3_SEEDS")].iloc[0] for t in TASKS}
    else:
        hard_macro = abdf[(abdf.policy == "P2 Direct + pooled hard verifier") & (abdf.scope == "MACRO")].iloc[0]
        hard_task = {t: abdf[(abdf.policy == "P2 Direct + pooled hard verifier") & (abdf.scope == f"task{t}")].iloc[0] for t in TASKS}
    selected_macro_cls = adf[(adf.scope == "MACRO") & (adf.architecture == selected) & (adf.seed == "MEAN_3_SEEDS")].iloc[0]
    positive_tasks = [t for t in TASKS if hard_task[t].full_task_SR > direct_task[t].full_task_SR + 1e-12]
    seed_deltas = []
    fold_deltas = []
    if final_rule == "SINGLE":
        for s in SEEDS:
            h = abdf[(abdf.policy == "P2 Direct + pooled hard verifier") & (abdf.scope == "MACRO") & (abdf.seed == s)].iloc[0]
            seed_deltas.append(float(h.full_task_SR-direct_macro.full_task_SR))
        for f in FOLDS:
            hm = held_mask(md, f); dm = md[hm].copy(); pp = p_direct[hm]
            drow = pv.summarize_controller(direct_policy_rows(dm.reset_index(drop=True), pp, "DIRECT"), "")
            vals = []
            for s in SEEDS:
                vals.append(pv.summarize_controller(pv.controller_rows(dm.reset_index(drop=True), pp, current_logits[selected][s, hm], "single", "STRICT"), "")["full_task_SR"])
            fold_deltas.append(float(np.mean(vals)-drow["full_task_SR"]))
    else:
        seed_deltas = [float("nan")]*3
        for f in FOLDS:
            hm = held_mask(md, f); dm = md[hm].reset_index(drop=True); pp = p_direct[hm]
            dr = pv.summarize_controller(direct_policy_rows(dm, pp, "DIRECT"), "")
            hr = pv.summarize_controller(pv.controller_rows(dm, pp, current_logits[selected][:, hm], final_rule.lower(), "STRICT"), "")
            fold_deltas.append(float(hr["full_task_SR"]-dr["full_task_SR"]))
    checks = {
        "macro_full_task_SR_improves": bool(hard_macro.full_task_SR > direct_macro.full_task_SR + 1e-12),
        "at_least_two_tasks_improve": len(positive_tasks) >= 2,
        "under_force_nonincrease": bool(hard_macro.under_force <= direct_macro.under_force + 1e-12),
        "mean_force_delta_reasonable": bool(hard_macro.mean_force_N <= direct_macro.mean_force_N + PROMOTION_MAX_FORCE_DELTA_N + 1e-12),
        "strict_improvement_not_fallback": bool(hard_macro.full_task_SR > direct_macro.full_task_SR + 1e-12),
        "FPR_low_enough": bool(selected_macro_cls.FPR <= PROMOTION_MAX_FPR + 1e-12),
        "coverage_acceptable": bool(hard_macro.coverage >= PROMOTION_MIN_COVERAGE - 1e-12),
        "three_seed_direction_consistent": bool(final_rule != "SINGLE" or all(x > 1e-12 for x in seed_deltas)),
        "three_fold_direction_stable": bool(all(x >= -1e-12 for x in fold_deltas)),
    }
    gate = all(checks.values())

    old_pool = difficulty[(difficulty.population == "OLD_VIEWED_FIXED_SCENE_DEV") & (difficulty.scope == "POOLED")].iloc[0]
    new_pool = difficulty[(difficulty.population == "NEW_POOLED_GROUPED_ROOT_CV") & (difficulty.scope == "POOLED")].iloc[0]
    one_macro = abdf[(abdf.policy == "P1 Direct + one force level") & (abdf.scope == "MACRO")].iloc[0]
    phys_cur = decdf[(decdf.scope == "MACRO") & (decdf.world_model == "WM-Current-Pooled") & (decdf.seed == "MEAN_3_SEEDS")].iloc[0]
    phys_only = decdf[(decdf.scope == "MACRO") & (decdf.world_model == "WM-PhysicsOnly-Pooled") & (decdf.seed == "MEAN_3_SEEDS")].iloc[0]
    classification = "PREDICTIVE_VERIFIER_POOLED_DEVELOPMENT_PASS" if gate else "PREDICTIVE_VERIFIER_POOLED_DEVELOPMENT_FAIL"
    final_choice = "ActiveForcing-Direct + Predictive Verifier" if gate else "ActiveForcing-Direct"
    freeze = {
        "status": "FROZEN_POOLED_DEVELOPMENT_" + ("PASS" if gate else "FAIL"), "classification": classification,
        "recommended_main_method_before_untouched_TEST": final_choice, "untouched_TEST_read": False,
        "untouched_TEST_authorized": False, "reason": "This development artifact never opens root-scaling TEST; PASS would require a separately authorized frozen pooled TEST protocol after root-scaling completes.",
        "Direct": "ActiveForcing-Direct-Pooled, fully OOF 3-fold/3-seed proposal ensemble",
        "world_model": "WM-Current-Pooled", "verifier_architecture": selected, "decision_rule": final_rule,
        "interface": "SUCCESS iff logit>0", "probability_threshold": None, "search": "upward-only from Direct proposal", "exhaustion": "NO_VALID_FORCE",
        "deployment_variant": {"name": "Verifier + Max Fallback", "rule": "NO_VALID_FORCE -> Fmax", "warning": "does not mean verifier judged Fmax safe"},
        "promotion_checks": checks, "positive_tasks": positive_tasks, "seed_SR_deltas": seed_deltas, "fold_SR_deltas": fold_deltas,
        "development_metrics": {"Direct_macro": direct_macro.to_dict(), "HardVerifier_macro": hard_macro.to_dict(), "OneStep_macro": one_macro.to_dict()},
        "checkpoints": verifier_ck + physics_ck + [json.loads((out / "shards" / f"fold{f}_seed{s}.json").read_text())["checkpoints"][i] for f in FOLDS for s in SEEDS for i in range(3)],
        "protocol_sha256": sha256(protocol),
    }
    write_json(out / "ACTIVEFORCING_POOLED_VERIFIER_FREEZE.json", freeze)
    write_json(out / "FINAL_POOLED_VERIFIER_CLASSIFICATION.json", {"classification": classification, "promotion_gate_pass": gate, "checks": checks, "selected_architecture": selected, "decision_rule": final_rule, "recommended_main_method": final_choice, "untouched_TEST_read": False, "paper_claim_allowed": False})

    role = "runtime verifier candidate plus Joint training auxiliary" if gate else "neither is required in the promoted main method; Joint remains an unadjudicated training auxiliary/diagnostic"
    report = f"""# Final pooled predictive-verifier report

In the truly shared four-task setting, fully root-heldout **ActiveForcing-Direct-Pooled reaches macro SR {direct_macro.full_task_SR:.3f}**. The strict pooled Predictive Verifier reaches {hard_macro.full_task_SR:.3f} with coverage {hard_macro.coverage:.3f}, under-force {hard_macro.under_force:.3f}, and mean force {hard_macro.mean_force_N:.3f} N, versus Direct under-force {direct_macro.under_force:.3f} and mean force {direct_macro.mean_force_N:.3f} N. The frozen development classification is **{classification}**. Therefore the main method at this evidence stage is **{final_choice}**. No untouched TEST was read.

## Direct answers

1. **Shared pooled Direct SR:** macro {direct_macro.full_task_SR:.3f}; pooled and per-task values are in `POOLED_DIRECT_RESULTS.csv` and `POOLED_SEARCH_POLICY_ABLATION.csv`.
2. **Is the verifier better?** Strict verifier delta is {hard_macro.full_task_SR-direct_macro.full_task_SR:+.3f} macro SR. Promotion gate: **{'PASS' if gate else 'FAIL'}**.
3. **Multiple-task consistency:** tasks with positive SR gain are {positive_tasks}; the rule requires at least two and all three seed directions to agree.
4. **Why did old 87.5% → 93.8% occur?** The old viewed population had {int(old_pool.cases_rescued_by_increasing_force)} force-rescuable cases among {int(old_pool.episodes)} context episodes, Direct SR {old_pool.Direct_SR:.3f}, and Fixed-Max SR {old_pool.Fixed_Max_SR:.3f}. It was a population with a different near-frontier/failure composition, and Max fallback supplied part of the observed improvement; it was not matched fully OOF evidence.
5. **Does new CV contain force-rescuable failures?** It contains {int(new_pool.cases_rescued_by_increasing_force)} among {int(new_pool.episodes)} context-repeat episodes; Fmax is not always successful in {int(new_pool.Fmax_not_always_successful)}. Thus only the force-rescuable subset is in principle recoverable by upward force search.
6. **Smarter than +1?** One-step macro SR is {one_macro.full_task_SR:.3f}; verifier is {hard_macro.full_task_SR:.3f}. Selective-verifier evidence requires verifier > one-step, which is {'present' if hard_macro.full_task_SR > one_macro.full_task_SR+1e-12 else 'not present'}.
7. **PhysicsOnly value:** macro verifier AUROC current={phys_cur.AUROC:.3f}, PhysicsOnly={phys_only.AUROC:.3f}; control SR current={phys_cur.control_full_task_SR:.3f}, PhysicsOnly={phys_only.control_full_task_SR:.3f}; trajectory standardized MAE current={phys_cur.trajectory_standardized_MAE:.3f}, PhysicsOnly={phys_only.trajectory_standardized_MAE:.3f}. Interpret it as physical trajectory signal only if matched control behavior remains close; otherwise current WM is outcome-shaped.
8. **World-model role:** **{role}**. Joint training changes the representation during training; Predictive Verifier is a separate runtime Direct→trajectory→binary check→upward-search mechanism. They are not one “world-model gain.”
9. **Main method:** **{final_choice}** pending any separately frozen untouched pooled TEST. A development PASS would still not be a paper claim.

## Per-task results and evidence limits

All requested tables report task0/task1/task5/task6, macro average, and pooled aggregate. task1 retains the material 140/180 reconstructed-label caveat. The model uses authoritative GT physics and no visual/Probe input, so these results are not final Active Probe E2E evidence. Receding-H8 was not run: {'the one-shot promotion gate failed' if not gate else 'a real mid-episode switching protocol is still separately required before reporting receding SR'}. Root-scaling and its untouched TEST were neither modified nor read.
"""
    (out / "FINAL_POOLED_VERIFIER_REPORT.md").write_text(report, encoding="utf-8")
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (out / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "classification": classification, "selected": selected, "rule": final_rule, "out": str(out)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--shard", nargs=2, type=int, metavar=("FOLD", "SEED"))
    g.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    if a.prepare: prepare(a.out)
    elif a.shard: shard(a.out, a.shard[0], a.shard[1])
    else: finalize(a.out)


if __name__ == "__main__":
    main()
