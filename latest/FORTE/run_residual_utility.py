#!/usr/bin/env python3
"""Cross-fitted ActiveForcing residual-utility development.

This module is deliberately restricted to the existing pooled TRAIN-root OOF
artifacts and the already-frozen retrospective force-critical membership.  It
does not discover or read any root-scaling or untouched TEST namespace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

import pooled_joint_novisual_current as pooled
import run_pooled_predictive_verifier as ppv
import run_predictive_verifier_development as pv
import task0_visual_context_early as early


ROOT = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
POOLED = ROOT / "pooled_predictive_verifier_20260901_033804"
CHALLENGE = ROOT / "existing_force_critical_benchmark_20260901_052000"
SOURCE = TABERO / "analysis/results/gnp_style_visual_context_prospective_20260831_011000/collection_train"
FRICTION_ROOT = TABERO / "analysis/results/active_friction_imagination_20260828_211106"
PROBE_PROTOCOL = TABERO / "analysis/results/probe_informed_imagination_20260829_121500/PROBE_INFORMED_IMAGINATION_PROTOCOL.json"
NO_PROBE_PRIOR = TABERO / "analysis/results/active_probe_necessity_20260830_063435/NO_PROBE_PHYSICS_PRIOR.json"
P5_MODEL_SOURCE = TABERO / "analysis/p5s0c_model_adjudication.py"
ACTIVE_FRICTION_SOURCE = TABERO / "analysis/active_friction_imagination.py"

TASKS = [0, 1, 5, 6]
FOLDS = [0, 1, 2]
SEEDS = [0, 1, 2]
FMAX = dict(pv.FMAX)
H = 8
RESIDUAL_HIDDEN = 32
RESIDUAL_EPOCHS = 80
RESIDUAL_BATCH = 64
RESIDUAL_LR = 8e-4
RESIDUAL_WEIGHT_DECAY = 1e-4
RESIDUAL_KINDS = ["DIRECT_ONLY", "WM_CURRENT", "WM_PHYSICS_ONLY"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["status"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def source_paths() -> list[Path]:
    paths = [
        POOLED / "POOLED_DEVELOPMENT_PROTOCOL.json",
        POOLED / "POOLED_GROUPED_CV_MANIFEST.json",
        POOLED / "FINAL_POOLED_VERIFIER_CLASSIFICATION.json",
        CHALLENGE / "FORCE_CRITICAL_EXISTING_DEFINITION.json",
        CHALLENGE / "FORCE_CRITICAL_EXISTING_MANIFEST.json",
        CHALLENGE / "FORCE_CRITICAL_EXISTING_QA.json",
        FRICTION_ROOT / "FRICTION_GRU.pt",
        FRICTION_ROOT / "PROTOCOL.json",
        TABERO / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_NORMALIZATION.json",
        PROBE_PROTOCOL,
        NO_PROBE_PRIOR,
        ROOT / "run_pooled_predictive_verifier.py",
        ROOT / "run_predictive_verifier_development.py",
        ROOT / "pooled_joint_novisual_current.py",
        P5_MODEL_SOURCE,
        ACTIVE_FRICTION_SOURCE,
    ]
    paths += [POOLED / "shards" / f"fold{f}_seed{s}.npz" for f in FOLDS for s in SEEDS]
    paths += [POOLED / "checkpoints" / f"DIRECT_POOLED_fold{f}_seed{s}.pt" for f in FOLDS for s in SEEDS]
    paths += [POOLED / "checkpoints" / f"WM_CURRENT_POOLED_fold{f}_seed{s}.pt" for f in FOLDS for s in SEEDS]
    paths += [POOLED / "checkpoints" / f"WM_PHYSICSONLY_POOLED_fold{f}_seed{s}.pt" for f in FOLDS for s in SEEDS]
    return paths


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "RESIDUAL_UTILITY_DEVELOPMENT_PROTOCOL.json"
    if protocol_path.exists():
        print(json.dumps({"status": "ALREADY_FROZEN", "path": str(protocol_path), "sha256": sha256(protocol_path)}, indent=2))
        return
    missing = [str(p) for p in source_paths() if not p.exists()]
    if missing:
        raise RuntimeError(f"missing authoritative inputs: {missing}")
    challenge_qa = json.loads((CHALLENGE / "FORCE_CRITICAL_EXISTING_QA.json").read_text())
    if challenge_qa.get("status") != "PASS_MEMBERSHIP_FROZEN":
        raise RuntimeError("frozen existing-data challenge QA is not PASS")
    probe_protocol = json.loads(PROBE_PROTOCOL.read_text())
    probe_rule = probe_protocol["conditions"]["PROBE_INFORMED_IMAGINATION"]
    if probe_rule.get("posterior_decision_rule") != "not available/inherited; point estimate only":
        raise RuntimeError("unexpected frozen Probe posterior semantics")
    prior = json.loads(NO_PROBE_PRIOR.read_text())["values"]
    protocol = {
        "status": "FROZEN_BEFORE_RESIDUAL_TRAINING_OR_CASE_LEVEL_EVALUATION",
        "scope": "task0/task1/task5/task6 pooled authoritative TRAIN; grouped-root OOF development; retrospective frozen existing-data force-critical subset",
        "untouched_TEST_read": False,
        "root_scaling_pipeline_modified": False,
        "reward": {
            "SUCCESS": "(Fmax-F)/Fmax",
            "FAILURE": -1,
            "failure_penalty_tuned": False,
            "task_Fmax_N": FMAX,
        },
        "direct": {
            "model": "existing frozen pooled ActiveForcing-Direct OOF ensemble",
            "probability": "mean sigmoid over three fixed Direct seeds",
            "DirectUtility": "argmax p_D*(Fmax-F)/Fmax + (1-p_D)*(-1); ties choose lower force",
            "historical_equivalence": "the existing planner used the same numerator without division by Fmax; decisions must be exactly identical",
            "probability_threshold": None,
        },
        "base_oof": {
            "folds": 3,
            "group": ["task", "root_id"],
            "same_root_context_force_repeat_one_fold": True,
            "Direct_checkpoint_has_not_seen_scored_root": True,
            "WM_checkpoint_has_not_seen_scored_root": True,
            "base_seed_aggregation": "mean probability / mean predicted trajectory across fixed seeds 0,1,2; no best seed",
        },
        "residual_cross_fit": {
            "evaluation_fold_k": "train residual only on OOF rows from the other two folds",
            "group": ["task", "root_id"],
            "seeds": SEEDS,
            "target": "realized normalized reward minus DirectUtility",
            "architecture": "input -> Linear(hidden=32) -> GELU -> Linear(1)",
            "wm_input": "summary of mean OOF H8x13 trajectory: final, mean, population std, max (52D), plus normalized candidate force (1D)",
            "direct_only_input": "DirectUtility, normalized force, fixed one-hot task encoding (6D); same hidden/head capacity",
            "loss": "unweighted mean squared error",
            "optimizer": "AdamW",
            "epochs": RESIDUAL_EPOCHS,
            "batch": RESIDUAL_BATCH,
            "learning_rate": RESIDUAL_LR,
            "weight_decay": RESIDUAL_WEIGHT_DECAY,
            "input_normalization": "fit only on residual TRAIN folds",
            "early_stopping": False,
            "architecture_sweep": False,
            "threshold_tuning": False,
        },
        "planners": [
            "P0 Existing Direct Rule",
            "P1 Direct Utility",
            "P2 Direct + Direct-Only Residual + Utility",
            "P3 Direct + WM-Current Residual + Utility",
            "P3b Direct + WM-PhysicsOnly Residual + Utility",
            "P4 Direct + One-Step Increase",
            "P5 Fixed Max",
            "P6 Historical Hard Verifier (ablation only)",
        ],
        "challenge": {
            "definition_sha256": sha256(CHALLENGE / "FORCE_CRITICAL_EXISTING_DEFINITION.json"),
            "manifest_sha256": sha256(CHALLENGE / "FORCE_CRITICAL_EXISTING_MANIFEST.json"),
            "qa_sha256": sha256(CHALLENGE / "FORCE_CRITICAL_EXISTING_QA.json"),
            "membership_change_forbidden": True,
            "claim_limit": "retrospective model-independent stress subset, not untouched prospective TEST",
        },
        "world_model_gate": {
            "normal_macro_SR_non_decrease": True,
            "normal_macro_realized_utility_gt_DirectUtility": True,
            "normal_macro_realized_utility_gt_DirectOnlyResidual": True,
            "normal_macro_under_force_nonincrease": True,
            "challenge_rescue_rate_improves": True,
            "challenge_excess_or_regret_lt_FixedMax": True,
            "multiple_roots_benefit": True,
            "positive_task_count_min": 2,
            "all_three_seed_SR_nonnegative_and_utility_positive": True,
        },
        "probe": {
            "estimator": str(FRICTION_ROOT / "FRICTION_GRU.pt"),
            "estimator_frozen": True,
            "Probe": "point mu_hat only",
            "sigma_mu": "diagnostic only; no posterior integration",
            "NoProbe": {"discrete_prior_mu": prior, "aggregation": "mean utility across prior support"},
            "GT": "hidden friction, offline oracle only",
            "backend_selection": "use WM-Current Residual only if preregistered WM gate passes; otherwise DirectUtility",
            "current_population_caveat": "same root seeds as estimator TRAIN; Probe comparison is not root-heldout and cannot support final Probe claim",
        },
        "forbidden": [
            "root-scaling untouched TEST discovery/read",
            "prospective challenge membership mutation",
            "probability or verifier threshold tuning",
            "failure penalty tuning",
            "Hard Verifier promotion",
            "architecture sweep",
            "best-seed selection",
        ],
        "source_hashes": {str(p): sha256(p) for p in source_paths()},
    }
    write_json(protocol_path, protocol)
    print(json.dumps({"status": protocol["status"], "path": str(protocol_path), "sha256": sha256(protocol_path)}, indent=2))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def summary52(traj: np.ndarray) -> np.ndarray:
    return np.concatenate([traj[:, -1], traj.mean(1), traj.std(1), traj.max(1)], axis=1).astype(np.float32)


def task_onehot(tasks: np.ndarray) -> np.ndarray:
    ans = np.zeros((len(tasks), len(TASKS)), np.float32)
    for j, task in enumerate(TASKS):
        ans[:, j] = (tasks == task).astype(np.float32)
    return ans


class UtilityResidual(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, RESIDUAL_HIDDEN), nn.GELU(), nn.Linear(RESIDUAL_HIDDEN, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_residual(x: np.ndarray, y: np.ndarray, train_ids: np.ndarray, seed: int) -> tuple[UtilityResidual, dict, list[dict]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mean = x[train_ids].mean(0).astype(np.float32)
    std = x[train_ids].std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    xn = ((x - mean) / std).astype(np.float32)
    model = UtilityResidual(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=RESIDUAL_LR, weight_decay=RESIDUAL_WEIGHT_DECAY)
    rng = np.random.default_rng(seed + 99173)
    hist: list[dict] = []
    for epoch in range(1, RESIDUAL_EPOCHS + 1):
        order = rng.permutation(train_ids)
        losses = []
        model.train()
        for st in range(0, len(order), RESIDUAL_BATCH):
            ids = order[st:st + RESIDUAL_BATCH]
            xb = torch.tensor(xn[ids], dtype=torch.float32)
            yb = torch.tensor(y[ids], dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        hist.append({"epoch": epoch, "train_MSE": float(np.mean(losses))})
    model.eval()
    return model, {"input_mean": mean, "input_std": std}, hist


def predict_residual(model: UtilityResidual, stats: dict, x: np.ndarray) -> np.ndarray:
    xn = ((x - stats["input_mean"]) / stats["input_std"]).astype(np.float32)
    with torch.no_grad():
        return model(torch.tensor(xn, dtype=torch.float32)).cpu().numpy().astype(np.float32)


def load_raw_metadata(md: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        p = SOURCE / f"task{task}/task{task}/branches.csv"
        q = pd.read_csv(p)
        q["task"] = task
        rows.append(q)
    d = pd.concat(rows, ignore_index=True)
    d["force_key"] = pd.to_numeric(d.requested_force_N).round(5)
    d["repeat_key"] = pd.to_numeric(d.repeat_index).astype(int)
    lookup = {(str(r.context_id), int(r.repeat_key), float(r.force_key)): r for r in d.itertuples(index=False)}
    aligned = []
    for r in md.itertuples(index=False):
        key = (str(r.context_id), int(r.repeat), round(float(r.force_N), 5))
        src = lookup.get(key)
        if src is None:
            aligned.append({
                "branch_id": str(r.branch_id), "context_id": str(r.context_id), "task": int(r.task),
                "requested_force_N": float(r.force_N), "repeat_index": int(r.repeat),
                "full_task_success_y": int(r.success), "measured_force_peak_N": math.nan,
                "failure_reason": "", "failure_stage": "", "dropped": math.nan,
                "lost_in_transit": math.nan, "raw_metadata_available": 0,
            })
        else:
            rec = src._asdict()
            rec["source_branch_id"] = rec.get("branch_id", "")
            rec["branch_id"] = str(r.branch_id)
            rec["raw_metadata_available"] = 1
            aligned.append(rec)
    out = pd.DataFrame(aligned)
    if len(out) != 720 or out.branch_id.nunique() != 720:
        raise RuntimeError("failed to align raw metadata to all 720 authoritative rows")
    return out


def build_oof(out: Path):
    scratch = out / "_data_audit" / "population"
    scratch.mkdir(parents=True, exist_ok=True)
    tpi, cf, full, cmap, traces, meta, audits, pairs, segs, _ = pooled.load_population(scratch)
    md = ppv.frame(traces, meta)
    if len(md) != 720 or md.context_id.nunique() != 72 or md.groupby(["task", "root_id"]).ngroups != 24:
        raise RuntimeError("unexpected pooled population grain")
    fold_id = np.full(len(md), -1, int)
    direct_seed = np.full((3, len(md)), np.nan, np.float32)
    current_seed = np.full((3, len(md), H, 13), np.nan, np.float32)
    physics_seed = np.full_like(current_seed, np.nan)
    for fold in FOLDS:
        for seed in SEEDS:
            z = np.load(POOLED / "shards" / f"fold{fold}_seed{seed}.npz")
            held = z["held_idx"].astype(int)
            if seed == 0:
                if np.any(fold_id[held] >= 0):
                    raise RuntimeError("OOF row assigned to more than one fold")
                fold_id[held] = fold
            direct_seed[seed, held] = sigmoid(z["direct_logits"][held])
            current_seed[seed, held] = z["current_pred"][held]
            physics_seed[seed, held] = z["physics_pred"][held]
    if np.any(fold_id < 0) or not np.isfinite(direct_seed).all() or not np.isfinite(current_seed).all() or not np.isfinite(physics_seed).all():
        raise RuntimeError("incomplete OOF base predictions")
    p_direct = direct_seed.mean(0)
    current = current_seed.mean(0)
    physics = physics_seed.mean(0)
    fmax = md.task.map(FMAX).to_numpy(float)
    force = md.force_N.to_numpy(float)
    success = md.success.to_numpy(float)
    u_direct = p_direct * ((fmax - force) / fmax) + (1.0 - p_direct) * (-1.0)
    old_u = p_direct * (fmax - force) + (1.0 - p_direct) * (-fmax)
    real_reward = np.where(success > 0, (fmax - force) / fmax, -1.0)
    target = real_reward - u_direct
    current52 = summary52(current)
    physics52 = summary52(physics)
    force_norm = (force / fmax).astype(np.float32)
    direct_x = np.column_stack([u_direct, force_norm, task_onehot(md.task.to_numpy(int))]).astype(np.float32)
    current_x = np.column_stack([current52, force_norm]).astype(np.float32)
    physics_x = np.column_stack([physics52, force_norm]).astype(np.float32)
    raw = load_raw_metadata(md)
    raw_idx = raw.set_index("branch_id")
    if set(md.branch_id) != set(raw_idx.index):
        raise RuntimeError("trace/raw branch IDs do not match")
    dataset = md.copy()
    dataset["fold"] = fold_id
    dataset["task_Fmax_N"] = fmax
    dataset["p_D_OOF_ensemble"] = p_direct
    for seed in SEEDS:
        dataset[f"p_D_OOF_seed{seed}"] = direct_seed[seed]
    dataset["U_D_normalized"] = u_direct
    dataset["U_D_historical_unnormalized"] = old_u
    dataset["R_real"] = real_reward
    dataset["delta_U_target"] = target
    for j in range(52):
        dataset[f"wm_current_summary_{j:02d}"] = current52[:, j]
        dataset[f"wm_physics_only_summary_{j:02d}"] = physics52[:, j]
    dataset.to_csv(out / "POOLED_OOF_UTILITY_DATASET.csv", index=False)
    return tpi, cf, full, cmap, traces, meta, audits, segs, md, fold_id, p_direct, u_direct, old_u, real_reward, target, {
        "DIRECT_ONLY": direct_x,
        "WM_CURRENT": current_x,
        "WM_PHYSICS_ONLY": physics_x,
    }, raw_idx


def train_all_residuals(out: Path, fold_id: np.ndarray, x_by_kind: dict[str, np.ndarray], target: np.ndarray):
    pred = {kind: np.full((3, len(target)), np.nan, np.float32) for kind in RESIDUAL_KINDS}
    logs = []
    ckdir = out / "checkpoints"
    ckdir.mkdir(exist_ok=True)
    for kind in RESIDUAL_KINDS:
        x = x_by_kind[kind]
        for fold in FOLDS:
            train_ids = np.flatnonzero(fold_id != fold)
            held_ids = np.flatnonzero(fold_id == fold)
            for seed in SEEDS:
                model, stats, hist = fit_residual(x, target, train_ids, seed)
                pred[kind][seed, held_ids] = predict_residual(model, stats, x[held_ids])
                path = ckdir / f"UTILITY_RESIDUAL_{kind}_fold{fold}_seed{seed}.pt"
                torch.save({
                    "kind": kind,
                    "fold": fold,
                    "seed": seed,
                    "state_dict": model.state_dict(),
                    "input_mean": stats["input_mean"],
                    "input_std": stats["input_std"],
                    "architecture": f"Linear({x.shape[1]},{RESIDUAL_HIDDEN})->GELU->Linear({RESIDUAL_HIDDEN},1)",
                    "target": "R_real-U_D",
                    "loss": "MSE",
                    "epochs": RESIDUAL_EPOCHS,
                    "batch": RESIDUAL_BATCH,
                    "optimizer": "AdamW",
                    "lr": RESIDUAL_LR,
                    "weight_decay": RESIDUAL_WEIGHT_DECAY,
                    "TRAIN_fold_ids": [f for f in FOLDS if f != fold],
                    "HELDOUT_fold_id": fold,
                    "TEST_used": False,
                }, path)
                logs.append({
                    "kind": kind,
                    "fold": fold,
                    "seed": seed,
                    "train_rows": len(train_ids),
                    "heldout_rows": len(held_ids),
                    "final_train_MSE": hist[-1]["train_MSE"],
                    "checkpoint": str(path),
                    "checkpoint_sha256": sha256(path),
                })
    for kind in RESIDUAL_KINDS:
        if not np.isfinite(pred[kind]).all():
            raise RuntimeError(f"incomplete residual OOF predictions for {kind}")
    write_csv(out / "RESIDUAL_TRAINING_MANIFEST.csv", logs)
    return pred


def grip_failure(row: pd.Series) -> float:
    if int(row.get("raw_metadata_available", 0)) != 1:
        return math.nan
    if int(row.get("full_task_success_y", 0)) == 1:
        return 0
    text = " ".join(str(row.get(k, "")) for k in ["failure_reason", "failure_stage"] if k in row).lower()
    flags = ["slip", "grip", "drop", "lost", "transport", "rotation", "place"]
    dropped = pd.to_numeric(pd.Series([row.get("dropped", 0)]), errors="coerce").fillna(0).iloc[0]
    lost = pd.to_numeric(pd.Series([row.get("lost_in_transit", 0)]), errors="coerce").fillna(0).iloc[0]
    return float(any(x in text for x in flags) or int(dropped) == 1 or int(lost) == 1)


def make_policy_rows(md: pd.DataFrame, raw_idx: pd.DataFrame, method: str, score: np.ndarray | None,
                     direct_score: np.ndarray, seed: Any, hard_logits: np.ndarray | None = None) -> list[dict]:
    d = md.copy()
    d["global_index"] = np.arange(len(d))
    d["score"] = np.nan if score is None else score
    d["direct_score"] = direct_score
    rows = []
    for (cid, rep), q0 in d.groupby(["context_id", "repeat"], sort=True):
        q = q0.sort_values("force_N").copy()
        task = int(q.task.iloc[0])
        fmax = FMAX[task]
        direct = q.sort_values(["direct_score", "force_N"], ascending=[False, True]).iloc[0]
        selected = None
        if method == "P0 Existing Direct Rule":
            selected = q.sort_values(["score", "force_N"], ascending=[False, True]).iloc[0]
        elif method == "P1 Direct Utility":
            selected = direct
        elif method in {"P2 Direct-Only Residual", "P3 WM-Current Residual", "P3b WM-PhysicsOnly Residual"}:
            selected = q.sort_values(["score", "force_N"], ascending=[False, True]).iloc[0]
        elif method == "P4 One-Step":
            support = q[q.force_N >= float(direct.force_N) - 1e-9]
            selected = support.iloc[min(1, len(support) - 1)]
        elif method == "P5 Fixed-Max":
            selected = q.iloc[-1]
        elif method == "P6 Historical Hard Verifier":
            if hard_logits is None:
                raise RuntimeError("hard verifier logits missing")
            support = q[q.force_N >= float(direct.force_N) - 1e-9]
            accepted = support[hard_logits[support.global_index.to_numpy(int)] > 0]
            selected = accepted.iloc[0] if len(accepted) else None
        else:
            raise KeyError(method)
        good = q[q.success > 0]
        boundary = float(good.force_N.min()) if len(good) else math.nan
        direct_success = int(direct.success)
        higher_success = int(((q.force_N > float(direct.force_N) + 1e-9) & (q.success > 0)).any())
        recoverable = int(direct_success == 0 and higher_success == 1)
        if selected is None:
            rows.append({
                "method": method, "seed": seed, "context_id": cid, "repeat": int(rep),
                "root_id": str(q.root_id.iloc[0]), "task": task, "coverage": 0,
                "selected_force_N": math.nan, "actual_success": 0, "under_force": 0,
                "grip_related_failure": 0, "peak_force_N": math.nan, "excess_force_N": math.nan,
                "normalized_force_regret_success_only": math.nan, "realized_utility": -1.0,
                "direct_force_N": float(direct.force_N), "direct_actual_success": direct_success,
                "recoverable_direct_failure": recoverable, "rescued_recoverable": 0,
                "collateral_escalation": 0, "collateral_rejection": int(direct_success == 1),
                "escalated": 0, "selective_escalation_true_positive": 0,
            })
            continue
        branch = raw_idx.loc[str(selected.branch_id)]
        actual = int(selected.success)
        force = float(selected.force_N)
        escalation = int(force > float(direct.force_N) + 1e-9)
        reward = (fmax - force) / fmax if actual else -1.0
        rows.append({
            "method": method, "seed": seed, "context_id": cid, "repeat": int(rep),
            "root_id": str(selected.root_id), "task": task, "coverage": 1,
            "selected_force_N": force, "actual_success": actual,
            "under_force": int(math.isfinite(boundary) and force < boundary - 1e-9),
            "grip_related_failure": grip_failure(branch) if not actual else 0,
            "peak_force_N": float(branch.get("measured_force_peak_N", math.nan)),
            "excess_force_N": force - boundary if math.isfinite(boundary) else math.nan,
            "normalized_force_regret_success_only": (force - boundary) / fmax if actual and math.isfinite(boundary) else math.nan,
            "realized_utility": reward,
            "direct_force_N": float(direct.force_N), "direct_actual_success": direct_success,
            "recoverable_direct_failure": recoverable,
            "rescued_recoverable": int(recoverable and actual == 1),
            "collateral_escalation": int(direct_success == 1 and escalation),
            "collateral_rejection": 0,
            "escalated": escalation,
            "selective_escalation_true_positive": int(escalation and direct_success == 0),
        })
    return rows


def summarize(rows: list[dict], scope_name: str, method: str, seed: Any) -> dict:
    q = pd.DataFrame(rows)
    rec = int(q.recoverable_direct_failure.sum())
    escalated = int(q.escalated.sum())
    direct_safe = int((q.direct_actual_success > 0).sum())
    finite = q[q.coverage > 0]
    success_regret = finite[(finite.actual_success > 0) & finite.normalized_force_regret_success_only.notna()]
    return {
        "scope": scope_name,
        "method": method,
        "seed": seed,
        "episodes": len(q),
        "coverage": float(q.coverage.mean()),
        "full_task_SR": float((q.actual_success * q.coverage).mean()),
        "under_force_rate": float(q.under_force.mean()),
        "grip_related_failure_rate": float(q.grip_related_failure.mean()),
        "mean_force_N": float(finite.selected_force_N.mean()) if len(finite) else math.nan,
        "mean_peak_force_N": float(finite.peak_force_N.mean()) if len(finite) else math.nan,
        "mean_excess_force_N": float(finite.excess_force_N.mean()) if finite.excess_force_N.notna().any() else math.nan,
        "normalized_force_regret_success_only": float(success_regret.normalized_force_regret_success_only.mean()) if len(success_regret) else math.nan,
        "successful_regret_episodes": len(success_regret),
        "mean_realized_utility": float(q.realized_utility.mean()),
        "recoverable_direct_failures": rec,
        "rescue_rate": float(q.rescued_recoverable.sum() / rec) if rec else math.nan,
        "collateral_escalation_rate": float(q.collateral_escalation.sum() / direct_safe) if direct_safe else math.nan,
        "collateral_rejection_rate": float(q.collateral_rejection.sum() / direct_safe) if direct_safe else math.nan,
        "selective_escalation_precision": float(q.selective_escalation_true_positive.sum() / escalated) if escalated else math.nan,
        "NO_VALID_FORCE_rate": float(1.0 - q.coverage.mean()),
        "benefited_roots": int(q[(q.direct_actual_success == 0) & (q.actual_success == 1)].root_id.nunique()),
        "harmed_roots": int(q[(q.direct_actual_success == 1) & ((q.actual_success == 0) | (q.coverage == 0))].root_id.nunique()),
    }


def all_scopes(rows: list[dict], method: str, seed: Any) -> list[dict]:
    d = pd.DataFrame(rows)
    ans, task_rows = [], []
    for task in sorted(d.task.unique()):
        r = summarize(d[d.task == task].to_dict("records"), f"task{task}", method, seed)
        r["task"] = int(task)
        task_rows.append(r)
        ans.append(r)
    numeric = [
        "coverage", "full_task_SR", "under_force_rate", "grip_related_failure_rate", "mean_force_N",
        "mean_peak_force_N", "mean_excess_force_N", "normalized_force_regret_success_only",
        "mean_realized_utility", "rescue_rate", "collateral_escalation_rate", "collateral_rejection_rate",
        "selective_escalation_precision", "NO_VALID_FORCE_rate",
    ]
    macro = {"scope": "MACRO", "task": "MACRO", "method": method, "seed": seed, "episodes": int(sum(r["episodes"] for r in task_rows))}
    for k in numeric:
        vals = [r[k] for r in task_rows if math.isfinite(float(r[k]))]
        macro[k] = float(np.mean(vals)) if vals else math.nan
    for k in ["successful_regret_episodes", "recoverable_direct_failures", "benefited_roots", "harmed_roots"]:
        macro[k] = int(sum(r[k] for r in task_rows))
    ans.append(macro)
    pooled_row = summarize(rows, "POOLED", method, seed)
    pooled_row["task"] = "POOLED"
    ans.append(pooled_row)
    return ans


def hard_verifier_logits(md: pd.DataFrame) -> np.ndarray:
    logits = np.full((3, len(md)), np.nan, np.float32)
    for fold in FOLDS:
        for seed in SEEDS:
            z = np.load(POOLED / "shards" / f"fold{fold}_seed{seed}.npz")
            held = z["held_idx"].astype(int)
            path = POOLED / "checkpoints" / f"VERIFIER_WM-Current-Pooled_V0_LINEAR_fold{fold}_seed{seed}.pt"
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = pv.make_verifier("V0_LINEAR")
            model.load_state_dict(ck["state_dict"])
            stats = {k: np.asarray(v, np.float32) if isinstance(v, list) else v for k, v in ck["stats"].items()}
            logits[seed, held] = pv.predict_logits(model, stats, z["current_pred"], z["extra"], held, torch.device("cpu"))
    if not np.isfinite(logits).all():
        raise RuntimeError("incomplete historical hard-verifier OOF logits")
    return logits


def evaluate_policies(md: pd.DataFrame, raw_idx: pd.DataFrame, u_direct: np.ndarray, old_u: np.ndarray,
                      residual_pred: dict[str, np.ndarray], challenge_ids: set[str] | None = None):
    keep = np.ones(len(md), bool) if challenge_ids is None else md.context_id.astype(str).isin(challenge_ids).to_numpy()
    sub = md[keep].reset_index(drop=True)
    idx = np.flatnonzero(keep)
    direct = u_direct[idx]
    old = old_u[idx]
    hard = hard_verifier_logits(md)[:, idx]
    rows, summaries = [], []
    fixed = [
        ("P0 Existing Direct Rule", old, "ENSEMBLE"),
        ("P1 Direct Utility", direct, "ENSEMBLE"),
        ("P4 One-Step", None, "ENSEMBLE"),
        ("P5 Fixed-Max", None, "ENSEMBLE"),
    ]
    for method, score, seed in fixed:
        r = make_policy_rows(sub, raw_idx, method, score, direct, seed)
        rows += r
        summaries += all_scopes(r, method, seed)
    mapping = {
        "DIRECT_ONLY": "P2 Direct-Only Residual",
        "WM_CURRENT": "P3 WM-Current Residual",
        "WM_PHYSICS_ONLY": "P3b WM-PhysicsOnly Residual",
    }
    for kind, method in mapping.items():
        for seed in SEEDS:
            score = direct + residual_pred[kind][seed, idx]
            r = make_policy_rows(sub, raw_idx, method, score, direct, seed)
            rows += r
            summaries += all_scopes(r, method, seed)
        score = direct + residual_pred[kind][:, idx].mean(0)
        r = make_policy_rows(sub, raw_idx, method, score, direct, "ENSEMBLE_MEAN")
        rows += r
        summaries += all_scopes(r, method, "ENSEMBLE_MEAN")
    for seed in SEEDS:
        r = make_policy_rows(sub, raw_idx, "P6 Historical Hard Verifier", None, direct, seed, hard_logits=hard[seed])
        rows += r
        summaries += all_scopes(r, "P6 Historical Hard Verifier", seed)
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def residual_prediction_metrics(target: np.ndarray, pred: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for kind, values in pred.items():
        for seed in [*SEEDS, "ENSEMBLE_MEAN"]:
            z = values.mean(0) if seed == "ENSEMBLE_MEAN" else values[int(seed)]
            err = z - target
            rows.append({
                "residual": kind,
                "seed": seed,
                "branch_MAE_deltaU": float(np.mean(np.abs(err))),
                "branch_RMSE_deltaU": float(np.sqrt(np.mean(err ** 2))),
                "branch_bias_deltaU": float(np.mean(err)),
                "target_std": float(np.std(target)),
            })
    return pd.DataFrame(rows)


def load_probe_estimates(traces) -> pd.DataFrame:
    af = import_file("residual_utility_active_friction", ACTIVE_FRICTION_SOURCE)
    p5 = af.P5C_MODEL
    norm_path = TABERO / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_NORMALIZATION.json"
    norm = json.loads(norm_path.read_text())
    ck = torch.load(FRICTION_ROOT / "FRICTION_GRU.pt", map_location="cpu", weights_only=False)
    model = af.FrictionGRU(int(ck["input_dim"]), int(ck["projection_dim"]), int(ck["hidden_dim"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    contexts = {}
    for tr in traces:
        contexts.setdefault(tr.context_id, tr)
    examples = []
    feature_names = norm["dynamic_feature_names"]
    phases = norm["phase_categories_from_train"]
    states = norm["contact_state_categories_from_train"]
    mean = np.asarray(norm["dynamic_mean"], np.float32)
    std = np.asarray(norm["dynamic_std"], np.float32)
    for cid, tr in sorted(contexts.items()):
        path = SOURCE / f"task{int(tr.task)}/P5S0C_PROBE_TELEMETRY/{cid}_probe_timesteps.csv"
        if not path.exists():
            raise RuntimeError(f"missing current Probe telemetry {path}")
        df = p5.sequence_dataframe(str(path), phases, states).reindex(columns=feature_names).fillna(0.0)
        arr = ((df.to_numpy(np.float32) - mean) / std).astype(np.float32)
        examples.append(af.Example(str(cid), str(tr.root_id), int(tr.task), "TRAIN", "UNKNOWN", float(tr.mu), arr))
    mu_hat, sigma = af.predict(model, examples, torch.device("cpu"))
    rows = []
    for e, mh, sg in zip(examples, mu_hat, sigma):
        rows.append({
            "context_id": e.context_id,
            "root_id": e.root_id,
            "task": e.task,
            "mu_GT": e.friction,
            "mu_hat": float(mh),
            "sigma_mu_diagnostic_only": float(sg),
            "abs_error": float(abs(mh - e.friction)),
            "estimator_root_heldout": False,
            "reason": "current roots reuse estimator TRAIN root seeds 5100-5105",
        })
    if len(rows) != 72:
        raise RuntimeError(f"expected 72 Probe estimates, got {len(rows)}")
    return pd.DataFrame(rows)


def load_models_for_mu(full, tpi, fold: int, seed: int):
    dck = torch.load(POOLED / "checkpoints" / f"DIRECT_POOLED_fold{fold}_seed{seed}.pt", map_location="cpu", weights_only=False)
    direct = full.FeasibilityOnly()
    direct.load_state_dict(dck["state_dict"])
    direct.eval()
    wck = torch.load(POOLED / "checkpoints" / f"WM_CURRENT_POOLED_fold{fold}_seed{seed}.pt", map_location="cpu", weights_only=False)
    wm = tpi.ShortHorizonPhysicsGRU(17, 54, H)
    wm.load_state_dict(wck["state_dict"])
    wm.eval()
    return direct, wm


def predict_mu_scenario(tpi, full, traces, segs, fold_id: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(traces)
    p = np.full((3, n), np.nan, np.float32)
    traj = np.full((3, n, H, 13), np.nan, np.float32)
    base_x = np.stack([segs[t.branch_id].x for t in traces]).astype(np.float32)
    initial = np.stack([t.state[0] for t in traces]).astype(np.float32)
    for fold in FOLDS:
        ids = np.flatnonzero(fold_id == fold)
        for seed in SEEDS:
            z = np.load(POOLED / "shards" / f"fold{fold}_seed{seed}.npz")
            xm, xs, ym, ys = [z[k].astype(np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"]]
            x = base_x[ids].copy()
            x[:, :, 18] = mu[ids, None]
            xn = (x - xm) / xs
            step = torch.tensor(xn[:, :, :17], dtype=torch.float32)
            cond = torch.tensor(xn[:, 0, 17:], dtype=torch.float32)
            direct, wm = load_models_for_mu(full, tpi, fold, seed)
            with torch.no_grad():
                p[seed, ids] = sigmoid(direct(step, cond).cpu().numpy())
                pred = wm(step, cond).cpu().numpy()
            traj[seed, ids] = pred * ys[None, None, :] + ym[None, None, :] + initial[ids, None, :]
    if not np.isfinite(p).all() or not np.isfinite(traj).all():
        raise RuntimeError("incomplete alternate-physics OOF inference")
    return p.mean(0), traj.mean(0)


def load_residual_checkpoint(out: Path, kind: str, fold: int, seed: int) -> tuple[UtilityResidual, dict]:
    path = out / "checkpoints" / f"UTILITY_RESIDUAL_{kind}_fold{fold}_seed{seed}.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    dim = len(ck["input_mean"])
    model = UtilityResidual(dim)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, {"input_mean": np.asarray(ck["input_mean"], np.float32), "input_std": np.asarray(ck["input_std"], np.float32)}


def residual_for_alternate(out: Path, kind: str, x: np.ndarray, fold_id: np.ndarray) -> np.ndarray:
    ans = np.full((3, len(x)), np.nan, np.float32)
    for fold in FOLDS:
        ids = np.flatnonzero(fold_id == fold)
        for seed in SEEDS:
            model, stats = load_residual_checkpoint(out, kind, fold, seed)
            ans[seed, ids] = predict_residual(model, stats, x[ids])
    if not np.isfinite(ans).all():
        raise RuntimeError("incomplete alternate-physics residual predictions")
    return ans


def probe_source_scores(out: Path, tpi, full, traces, segs, md: pd.DataFrame, fold_id: np.ndarray,
                        probe: pd.DataFrame, selected_backend: str) -> dict[str, np.ndarray]:
    probe_map = probe.set_index("context_id")
    mu_gt = md.mu.to_numpy(np.float32)
    mu_probe = md.context_id.map(probe_map.mu_hat).to_numpy(np.float32)
    prior = [float(x) for x in json.loads(NO_PROBE_PRIOR.read_text())["values"]]
    source_scenarios = {
        "GT": [mu_gt],
        "Probe": [mu_probe],
        "NoProbe": [np.full(len(md), x, np.float32) for x in prior],
    }
    fmax = md.task.map(FMAX).to_numpy(float)
    force = md.force_N.to_numpy(float)
    force_norm = (force / fmax).astype(np.float32)
    scores = {}
    for source, scenarios in source_scenarios.items():
        us = []
        for mu in scenarios:
            p, traj = predict_mu_scenario(tpi, full, traces, segs, fold_id, mu)
            ud = p * ((fmax - force) / fmax) + (1.0 - p) * (-1.0)
            if selected_backend == "P3 WM-Current Residual":
                x = np.column_stack([summary52(traj), force_norm]).astype(np.float32)
                delta = residual_for_alternate(out, "WM_CURRENT", x, fold_id).mean(0)
                us.append(ud + delta)
            else:
                us.append(ud)
        scores[source] = np.mean(np.stack(us), axis=0)
    return scores


def evaluate_probe(out: Path, tpi, full, traces, segs, md: pd.DataFrame, raw_idx: pd.DataFrame,
                   fold_id: np.ndarray, probe: pd.DataFrame, selected_backend: str, challenge_ids: set[str]):
    scores = probe_source_scores(out, tpi, full, traces, segs, md, fold_id, probe, selected_backend)
    rows = []
    for source, score in scores.items():
        planner_name = "P1 Direct Utility" if selected_backend == "P1 Direct Utility" else "P3 WM-Current Residual"
        # Physics-source comparisons must select using that source's own score.
        # For DirectUtility this means passing the source score as direct_score;
        # for a residual planner score and direct_score coincide here because
        # collateral metrics are not a Probe headline.
        r = make_policy_rows(md, raw_idx, planner_name, score, score, source)
        for z in r:
            z["physics_source"] = source
            z["evaluation_scope"] = "NORMAL_POOLED"
            z["selected_backend"] = selected_backend
        rows += r
        cr = [z.copy() for z in r if str(z["context_id"]) in challenge_ids]
        for z in cr:
            z["evaluation_scope"] = "FORCE_CRITICAL_RETROSPECTIVE"
        rows += cr
    per = pd.DataFrame(rows)
    gt = per[per.physics_source == "GT"][["evaluation_scope", "context_id", "repeat", "selected_force_N"]].rename(columns={"selected_force_N": "GT_selected_force_N"})
    per = per.merge(gt, on=["evaluation_scope", "context_id", "repeat"], how="left", validate="many_to_one")
    per["selected_force_agreement_with_GT"] = np.isclose(per.selected_force_N, per.GT_selected_force_N, equal_nan=True).astype(int)
    write_csv(out / "PROBE_UTILITY_PER_CONTEXT.csv", per)
    summaries = []
    for (scope, source), q in per.groupby(["evaluation_scope", "physics_source"], sort=False):
        for row in all_scopes(q.to_dict("records"), source, "FROZEN"):
            row["evaluation_scope"] = scope
            row["physics_source"] = source
            row["selected_backend"] = selected_backend
            row["decision_agreement_with_GT"] = float(q[q.task.astype(str) == str(row["task"])].selected_force_agreement_with_GT.mean()) if str(row["task"]).isdigit() else float(q.selected_force_agreement_with_GT.mean())
            summaries.append(row)
    sdf = pd.DataFrame(summaries)
    sdf["probe_friction_MAE"] = float(probe.abs_error.mean())
    sdf["probe_pair_ranking"] = probe_pair_ranking(probe)
    sdf["probe_estimator_root_heldout"] = False
    write_csv(out / "PROBE_UTILITY_SOURCE_TABLE.csv", sdf)
    return per, sdf


def probe_pair_ranking(probe: pd.DataFrame) -> float:
    correct, total = 0, 0
    for (_, task), q in probe.groupby(["root_id", "task"]):
        a = q.sort_values("mu_GT")
        vals = a.to_dict("records")
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if vals[i]["mu_GT"] == vals[j]["mu_GT"]:
                    continue
                total += 1
                correct += int(vals[i]["mu_hat"] < vals[j]["mu_hat"])
    return float(correct / total) if total else math.nan


def get_row(table: pd.DataFrame, method: str, scope: str = "MACRO", seed: Any = "ENSEMBLE_MEAN") -> pd.Series:
    q = table[(table.method == method) & (table.scope == scope) & (table.seed.astype(str) == str(seed))]
    if len(q) != 1:
        raise RuntimeError(f"expected one row for {method}/{scope}/{seed}, got {len(q)}")
    return q.iloc[0]


def classify(normal: pd.DataFrame, challenge: pd.DataFrame) -> tuple[dict, str]:
    direct = get_row(normal, "P1 Direct Utility", seed="ENSEMBLE")
    dore = get_row(normal, "P2 Direct-Only Residual")
    wm = get_row(normal, "P3 WM-Current Residual")
    fixed = get_row(challenge, "P5 Fixed-Max", seed="ENSEMBLE")
    cwm = get_row(challenge, "P3 WM-Current Residual")
    cdu = get_row(challenge, "P1 Direct Utility", seed="ENSEMBLE")
    per_task = normal[(normal.scope.str.startswith("task")) & (normal.seed.astype(str) == "ENSEMBLE_MEAN")]
    wm_task = per_task[per_task.method == "P3 WM-Current Residual"].set_index("task")
    du_task = normal[(normal.scope.str.startswith("task")) & (normal.seed.astype(str) == "ENSEMBLE") & (normal.method == "P1 Direct Utility")].set_index("task")
    positive_tasks = sum(
        (wm_task.loc[t, "full_task_SR"] >= du_task.loc[t, "full_task_SR"] - 1e-12)
        and (wm_task.loc[t, "mean_realized_utility"] > du_task.loc[t, "mean_realized_utility"] + 1e-12)
        for t in TASKS
    )
    seed_checks = []
    for seed in SEEDS:
        sr = get_row(normal, "P3 WM-Current Residual", seed=seed)
        seed_checks.append(bool(sr.full_task_SR >= direct.full_task_SR - 1e-12 and sr.mean_realized_utility > direct.mean_realized_utility + 1e-12))
    rescue_improves = bool(
        math.isfinite(float(cwm.rescue_rate)) and math.isfinite(float(cdu.rescue_rate))
        and cwm.rescue_rate > cdu.rescue_rate + 1e-12
    )
    checks = {
        "normal_macro_SR_non_decrease": bool(wm.full_task_SR >= direct.full_task_SR - 1e-12),
        "normal_macro_realized_utility_gt_DirectUtility": bool(wm.mean_realized_utility > direct.mean_realized_utility + 1e-12),
        "normal_macro_realized_utility_gt_DirectOnlyResidual": bool(wm.mean_realized_utility > dore.mean_realized_utility + 1e-12),
        "normal_macro_under_force_nonincrease": bool(wm.under_force_rate <= direct.under_force_rate + 1e-12),
        "challenge_rescue_rate_improves": rescue_improves,
        "challenge_excess_or_regret_lt_FixedMax": bool(
            (math.isfinite(float(cwm.mean_excess_force_N)) and cwm.mean_excess_force_N < fixed.mean_excess_force_N - 1e-12)
            or (math.isfinite(float(cwm.normalized_force_regret_success_only)) and cwm.normalized_force_regret_success_only < fixed.normalized_force_regret_success_only - 1e-12)
        ),
        "multiple_roots_benefit": bool(int(cwm.benefited_roots) >= 2),
        "positive_task_count_min_2": bool(positive_tasks >= 2),
        "all_three_seed_SR_nonnegative_and_utility_positive": bool(all(seed_checks)),
    }
    passed = all(checks.values())
    selected = "P3 WM-Current Residual" if passed else "P1 Direct Utility"
    result = {
        "classification": "ACTIVEFORCING_WM_RESIDUAL_UTILITY_PASS" if passed else "ACTIVEFORCING_WM_RESIDUAL_UTILITY_FAIL",
        "world_model_independent_value_gate_pass": passed,
        "checks": checks,
        "positive_task_count": int(positive_tasks),
        "per_seed_gate_direction": seed_checks,
        "selected_backend_before_Probe": selected,
        "untouched_TEST_read": False,
        "challenge_claim_limit": "retrospective frozen existing-data stress subset",
    }
    return result, selected


def write_reports(out: Path, normal: pd.DataFrame, challenge: pd.DataFrame, residual_metrics: pd.DataFrame,
                  decoupling: pd.DataFrame, direct_compare: pd.DataFrame, probe_summary: pd.DataFrame,
                  classification: dict, freeze: dict) -> None:
    def row(method: str, table=normal, seed="ENSEMBLE_MEAN"):
        actual_seed = seed
        if method in {"P0 Existing Direct Rule", "P1 Direct Utility", "P4 One-Step", "P5 Fixed-Max"}:
            actual_seed = "ENSEMBLE"
        return get_row(table, method, seed=actual_seed)

    du = row("P1 Direct Utility")
    dr = row("P2 Direct-Only Residual")
    wm = row("P3 WM-Current Residual")
    po = row("P3b WM-PhysicsOnly Residual")
    one = row("P4 One-Step")
    fixed = row("P5 Fixed-Max")
    cdu = row("P1 Direct Utility", challenge)
    cwm = row("P3 WM-Current Residual", challenge)
    cone = row("P4 One-Step", challenge)
    normal_probe = probe_summary[(probe_summary.evaluation_scope == "NORMAL_POOLED") & (probe_summary.scope == "MACRO")]
    pvals = {r.physics_source: r for _, r in normal_probe.iterrows()}
    direct_equiv = bool(direct_compare.decision_identical.all())
    report = f"""# ActiveForcing residual-utility development report

## Direct answer

The current historical Direct controller was **already using GNP-style expected utility**, not a probability crossing. The new normalized reward divides every candidate utility within a task by the same positive `Fmax`, so P0 and P1 are decision-identical: **{direct_equiv}**. Therefore removing 0.5/0.8 thresholds produces no new gain in this matched experiment.

The World-Model residual gate is **{'PASS' if classification['world_model_independent_value_gate_pass'] else 'FAIL'}**. On normal pooled grouped-root OOF, DirectUtility has macro SR {du.full_task_SR:.4f}, mean force {du.mean_force_N:.4f} N, under-force {du.under_force_rate:.4f}, and realized utility {du.mean_realized_utility:.4f}. Current-WM Residual has SR {wm.full_task_SR:.4f}, mean force {wm.mean_force_N:.4f} N, under-force {wm.under_force_rate:.4f}, and utility {wm.mean_realized_utility:.4f}. Direct-Only Residual has SR {dr.full_task_SR:.4f} and utility {dr.mean_realized_utility:.4f}. Thus predicted physics {'adds evidence beyond ordinary correction' if classification['world_model_independent_value_gate_pass'] else 'does not satisfy the preregistered independent-value conditions'}.

The final development choice is **{classification['selected_backend_before_Probe']}**. No untouched TEST was opened; this is not a final paper claim.

## Reliability–force tradeoff on normal pooled OOF

| Planner | Macro SR | Under-force | Mean force (N) | Success-only normalized regret | Realized utility |
|---|---:|---:|---:|---:|---:|
| DirectUtility | {du.full_task_SR:.4f} | {du.under_force_rate:.4f} | {du.mean_force_N:.4f} | {du.normalized_force_regret_success_only:.4f} | {du.mean_realized_utility:.4f} |
| Direct-Only Residual | {dr.full_task_SR:.4f} | {dr.under_force_rate:.4f} | {dr.mean_force_N:.4f} | {dr.normalized_force_regret_success_only:.4f} | {dr.mean_realized_utility:.4f} |
| Current-WM Residual | {wm.full_task_SR:.4f} | {wm.under_force_rate:.4f} | {wm.mean_force_N:.4f} | {wm.normalized_force_regret_success_only:.4f} | {wm.mean_realized_utility:.4f} |
| PhysicsOnly-WM Residual | {po.full_task_SR:.4f} | {po.under_force_rate:.4f} | {po.mean_force_N:.4f} | {po.normalized_force_regret_success_only:.4f} | {po.mean_realized_utility:.4f} |
| One-Step | {one.full_task_SR:.4f} | {one.under_force_rate:.4f} | {one.mean_force_N:.4f} | {one.normalized_force_regret_success_only:.4f} | {one.mean_realized_utility:.4f} |
| Fixed-Max | {fixed.full_task_SR:.4f} | {fixed.under_force_rate:.4f} | {fixed.mean_force_N:.4f} | {fixed.normalized_force_regret_success_only:.4f} | {fixed.mean_realized_utility:.4f} |

Per-task, pooled, and all three residual-seed rows are in `NORMAL_POOLED_UTILITY_TABLE.csv`; the table above is macro aggregation, not a hidden pooled-only average.

## Frozen force-critical stress subset

Membership was not changed. It remains the already-frozen model-independent 14-context, 11-root retrospective subset, selected using real adjacent-force outcomes before case-level model queries. DirectUtility already has SR {cdu.full_task_SR:.4f} here and {int(cdu.recoverable_direct_failures)} recoverable Direct failures, so rescue rate is {'defined' if math.isfinite(float(cdu.rescue_rate)) else '**not identifiable (zero denominator)**'}. Current-WM Residual reaches SR {cwm.full_task_SR:.4f}; One-Step reaches {cone.full_task_SR:.4f}. This subset cannot demonstrate rescue when Direct has no recoverable failure under the frozen utility proposal.

The prospective Force-Critical Challenge remains incomplete and untouched. These retrospective rows cannot replace a prospective untouched challenge confirmation.

## Current WM versus PhysicsOnly WM

Both use the same residual architecture and cross-fitting. Full branch-level residual errors and trajectory fidelity are in `WORLD_MODEL_RESIDUAL_DECOUPLING.csv`. Current-WM control utility is {wm.mean_realized_utility:.4f}; PhysicsOnly-WM is {po.mean_realized_utility:.4f}. {'PhysicsOnly is close enough to motivate physical-signal language, but the overall WM promotion gate still controls method inclusion.' if abs(po.mean_realized_utility-wm.mean_realized_utility) <= 0.01 else 'The two representations are not equivalent in this development result; any Current-WM advantage must be described as outcome-shaped rather than pure physics.'}

## Probe / NoProbe / GT

The frozen estimator has no legal posterior decision rule; `sigma_mu` is diagnostic-only, so Probe uses only the point estimate. NoProbe averages utility over the frozen support `[0.30, 0.56, 0.92]`. On the normal pooled diagnostic, NoProbe/Probe/GT macro SR values are {pvals.get('NoProbe', {}).get('full_task_SR', float('nan')):.4f}, {pvals.get('Probe', {}).get('full_task_SR', float('nan')):.4f}, and {pvals.get('GT', {}).get('full_task_SR', float('nan')):.4f}.

This is **not root-heldout Probe evidence**: the current roots reuse estimator TRAIN seeds 5100–5105. It verifies that the frozen Probe can be wired into the utility pipeline, but cannot establish the final NoProbe→Probe→GT paper gradient.

## Mechanism interpretation

- Expected-utility planning follows the GNP principle: combine success benefit, force cost, and a fixed failure penalty, then optimize the expectation.
- Residual utility is nominal-plus-correction: Direct remains the strong nominal decision and predicted physics can only shift candidate utilities; it never casts a binary veto.
- The historical Hard Verifier remains an ablation. Its risk–coverage failure came from false-negative rejection and `NO_VALID_FORCE`; it is not eligible for the selected controller.

## Answers to the preregistered questions

1. **Threshold removal:** no measured change, because the existing Direct rule was already the same utility argmax.
2. **WM residual improves DirectUtility:** {'yes under every gate' if classification['world_model_independent_value_gate_pass'] else 'no reliable independent-value result'}.
3. **Beyond ordinary calibration:** {'supported' if classification['checks']['normal_macro_realized_utility_gt_DirectOnlyResidual'] else 'not supported'}.
4. **PhysicsOnly:** see matched rows; it {'retains comparable control utility' if abs(po.mean_realized_utility-wm.mean_realized_utility) <= 0.01 else 'does not match Current-WM utility'}.
5. **Normal-safe and challenge-beneficial:** {'supported' if classification['world_model_independent_value_gate_pass'] else 'not jointly supported'}.
6. **More selective than One-Step:** {'supported' if cwm.full_task_SR >= cone.full_task_SR and cwm.mean_force_N < cone.mean_force_N else 'not supported'} on the frozen retrospective subset.
7. **Versus Fixed-Max:** Fixed-Max uses {fixed.mean_force_N:.4f} N on normal pooled; the selected method's force/utility tradeoff is reported separately rather than hidden in one scalar.
8. **NoProbe→Probe→GT:** diagnostic rows exist, but the root-overlap caveat prevents a final gradient claim.
9. **Probe changes full-task decisions:** exact per-context changes and GT agreement are in `PROBE_UTILITY_PER_CONTEXT.csv`; they are diagnostic only.
10. **Final method needs WM:** **{'yes' if classification['world_model_independent_value_gate_pass'] else 'no'}** under the frozen gate.
11. **Selected planner:** **{classification['selected_backend_before_Probe']}**, not Direct threshold and not Hard Verifier.
12. **Ready for final paper benchmark:** development components are hash-frozen, but final method evidence is **not complete** until a valid untouched evaluation and root-heldout Probe comparison are available.

## Data-quality limitations

- task1 contains reconstructed outcomes in the authoritative pooled population; see the prior pooled audit.
- The force-critical result is retrospective and has zero recoverable Direct failures under the current utility proposal.
- Probe outputs on these 72 contexts are not root-heldout from the frozen friction estimator.
- OOF results are development evidence. `ACTIVEFORCING_RESIDUAL_UTILITY_FREEZE.json` explicitly forbids treating them as untouched TEST.
"""
    (out / "FINAL_RESIDUAL_UTILITY_REPORT.md").write_text(report, encoding="utf-8")


def finalize(out: Path) -> None:
    protocol_path = out / "RESIDUAL_UTILITY_DEVELOPMENT_PROTOCOL.json"
    if not protocol_path.exists():
        raise RuntimeError("run prepare first")
    torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 4))))
    (tpi, cf, full, cmap, traces, meta, audits, segs, md, fold_id, p_direct, u_direct, old_u,
     real_reward, target, x_by_kind, raw_idx) = build_oof(out)
    residual_pred = train_all_residuals(out, fold_id, x_by_kind, target)
    normal_per, normal = evaluate_policies(md, raw_idx, u_direct, old_u, residual_pred)
    write_csv(out / "NORMAL_POOLED_UTILITY_PER_EPISODE.csv", normal_per)
    write_csv(out / "NORMAL_POOLED_UTILITY_TABLE.csv", normal)

    manifest = json.loads((CHALLENGE / "FORCE_CRITICAL_EXISTING_MANIFEST.json").read_text())
    challenge_ids = {str(x["context_id"]) for x in manifest["cases"]}
    challenge_per, challenge = evaluate_policies(md, raw_idx, u_direct, old_u, residual_pred, challenge_ids)
    write_csv(out / "FORCE_CRITICAL_UTILITY_PER_CASE.csv", challenge_per)
    write_csv(out / "FORCE_CRITICAL_UTILITY_TABLE.csv", challenge)
    rescue = challenge[[
        "scope", "task", "method", "seed", "episodes", "full_task_SR", "recoverable_direct_failures",
        "rescue_rate", "under_force_rate", "mean_excess_force_N", "normalized_force_regret_success_only",
        "collateral_escalation_rate", "collateral_rejection_rate", "selective_escalation_precision",
        "benefited_roots", "harmed_roots",
    ]].copy()
    write_csv(out / "FORCE_CRITICAL_RESCUE_REGRET.csv", rescue)

    direct_rows = normal[(normal.method.isin(["P0 Existing Direct Rule", "P1 Direct Utility"])) & (normal.seed.astype(str) == "ENSEMBLE")].copy()
    p0 = normal_per[(normal_per.method == "P0 Existing Direct Rule") & (normal_per.seed.astype(str) == "ENSEMBLE")]
    p1 = normal_per[(normal_per.method == "P1 Direct Utility") & (normal_per.seed.astype(str) == "ENSEMBLE")]
    key = ["context_id", "repeat"]
    eq = p0[key + ["selected_force_N"]].merge(p1[key + ["selected_force_N"]], on=key, suffixes=("_P0", "_P1"), validate="one_to_one")
    decision_identical = bool(np.allclose(eq.selected_force_N_P0, eq.selected_force_N_P1))
    direct_rows["decision_identical"] = decision_identical
    direct_rows["normalization_relation"] = "U_new=U_historical/Fmax; exact same argmax"
    write_csv(out / "DIRECT_UTILITY_COMPARISON.csv", direct_rows)

    residual_metrics = residual_prediction_metrics(target, residual_pred)
    residual_metrics = residual_metrics.rename(columns={"residual": "residual_kind"})
    utility_compare = normal[normal.method.isin([
        "P1 Direct Utility", "P2 Direct-Only Residual", "P3 WM-Current Residual", "P3b WM-PhysicsOnly Residual"
    ])].copy()
    utility_compare["residual_kind"] = utility_compare.method.map({
        "P2 Direct-Only Residual": "DIRECT_ONLY",
        "P3 WM-Current Residual": "WM_CURRENT",
        "P3b WM-PhysicsOnly Residual": "WM_PHYSICS_ONLY",
    })
    utility_compare = utility_compare.merge(residual_metrics, how="left", on=["seed", "residual_kind"], validate="many_to_one")
    write_csv(out / "UTILITY_RESIDUAL_COMPARISON.csv", utility_compare)

    fidelity = pd.read_csv(POOLED / "POOLED_WORLD_MODEL_DECOUPLING.csv")
    fidelity = fidelity[(fidelity.architecture == "V0_LINEAR") & (fidelity.scope.isin(["task0", "task1", "task5", "task6", "MACRO", "POOLED"]))]
    control = normal[normal.method.isin(["P3 WM-Current Residual", "P3b WM-PhysicsOnly Residual"])].copy()
    control["world_model"] = control.method.map({"P3 WM-Current Residual": "WM-Current-Pooled", "P3b WM-PhysicsOnly Residual": "WM-PhysicsOnly-Pooled"})
    control["residual_kind"] = control.method.map({"P3 WM-Current Residual": "WM_CURRENT", "P3b WM-PhysicsOnly Residual": "WM_PHYSICS_ONLY"})
    decoupling = control.merge(
        fidelity[["scope", "seed", "world_model", "trajectory_standardized_MAE", "IE_standardized_MAE"]],
        on=["scope", "seed", "world_model"], how="left", validate="many_to_one",
    )
    decoupling = decoupling.merge(residual_metrics, how="left", on=["seed", "residual_kind"], validate="many_to_one")
    write_csv(out / "WORLD_MODEL_RESIDUAL_DECOUPLING.csv", decoupling)

    classification, selected_backend = classify(normal, challenge)
    probe = load_probe_estimates(traces)
    write_csv(out / "FROZEN_PROBE_ESTIMATES_CURRENT_POPULATION.csv", probe)
    probe_per, probe_summary = evaluate_probe(out, tpi, full, traces, segs, md, raw_idx, fold_id, probe, selected_backend, challenge_ids)
    classification["probe"] = {
        "point_estimate_only": True,
        "posterior_used": False,
        "root_heldout": False,
        "claim_allowed": False,
        "friction_MAE_diagnostic": float(probe.abs_error.mean()),
        "pair_ranking_diagnostic": probe_pair_ranking(probe),
    }
    classification["final_development_method"] = "ActiveForcing-DirectUtility" if selected_backend == "P1 Direct Utility" else "ActiveForcing-ResidualUtility"
    write_json(out / "FINAL_RESIDUAL_UTILITY_CLASSIFICATION.json", classification)

    checkpoints = sorted((out / "checkpoints").glob("*.pt"))
    freeze = {
        "status": "DEVELOPMENT_CANDIDATE_FROZEN_BEFORE_ANY_UNTOUCHED_FINAL_EVALUATION",
        "classification": classification["classification"],
        "selected_backend": selected_backend,
        "method": classification["final_development_method"],
        "reward": {"SUCCESS": "(Fmax-F)/Fmax", "FAILURE": -1},
        "planner": "argmax candidate utility; lower-force tie break; no probability threshold",
        "Direct": "frozen pooled OOF architecture/checkpoints; no retraining for utility comparison",
        "WorldModel": "WM-Current residual only if gate PASS; otherwise excluded from main method",
        "residual_architecture": f"summary52+force -> Linear({RESIDUAL_HIDDEN}) -> GELU -> Linear(1)",
        "residual_seeds": SEEDS,
        "residual_decision": "mean delta utility across all three seeds; no best seed",
        "Probe": {"checkpoint": str(FRICTION_ROOT / "FRICTION_GRU.pt"), "sha256": sha256(FRICTION_ROOT / "FRICTION_GRU.pt"), "output": "point mu_hat", "sigma": "diagnostic only"},
        "NoProbe_prior": json.loads(NO_PROBE_PRIOR.read_text())["values"],
        "GT_semantics": "hidden simulator friction; offline oracle only",
        "challenge_manifest": str(CHALLENGE / "FORCE_CRITICAL_EXISTING_MANIFEST.json"),
        "challenge_manifest_sha256": sha256(CHALLENGE / "FORCE_CRITICAL_EXISTING_MANIFEST.json"),
        "untouched_TEST_read": False,
        "root_scaling_pipeline_modified": False,
        "paper_claim_allowed_now": False,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "post_protocol_QA_patches": [
                "align incomplete cumulative task manifests to the authoritative 720-row population without changing outcome labels",
                "register dynamically loaded frozen Probe module for Python dataclasses",
                "ensure each Probe/NoProbe/GT condition selects from its own utility scores",
                "make residual-metric CSV joins one-to-one by residual kind and seed",
            ],
            "scientific_protocol_changed_after_freeze": False,
        },
        "frozen_base_source_hashes": json.loads(protocol_path.read_text())["source_hashes"],
        "checkpoint_hashes": {str(p): sha256(p) for p in checkpoints},
        "source_protocol_sha256": sha256(protocol_path),
    }
    write_json(out / "ACTIVEFORCING_RESIDUAL_UTILITY_FREEZE.json", freeze)
    write_reports(out, normal, challenge, residual_metrics, decoupling, direct_rows, probe_summary, classification, freeze)

    required = [
        "POOLED_OOF_UTILITY_DATASET.csv", "DIRECT_UTILITY_COMPARISON.csv", "UTILITY_RESIDUAL_COMPARISON.csv",
        "WORLD_MODEL_RESIDUAL_DECOUPLING.csv", "NORMAL_POOLED_UTILITY_TABLE.csv", "FORCE_CRITICAL_UTILITY_TABLE.csv",
        "FORCE_CRITICAL_RESCUE_REGRET.csv", "PROBE_UTILITY_SOURCE_TABLE.csv", "PROBE_UTILITY_PER_CONTEXT.csv",
        "ACTIVEFORCING_RESIDUAL_UTILITY_FREEZE.json", "FINAL_RESIDUAL_UTILITY_REPORT.md",
        "FINAL_RESIDUAL_UTILITY_CLASSIFICATION.json",
    ]
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise RuntimeError(f"missing required outputs: {missing}")
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    lines = [f"{sha256(p)}  {p.name}" for p in files]
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETE",
        "output": str(out),
        "classification": classification["classification"],
        "selected_backend": selected_backend,
        "untouched_TEST_read": False,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "run"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args.out)
    else:
        finalize(args.out)


if __name__ == "__main__":
    main()
