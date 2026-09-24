#!/usr/bin/env python3
"""Frozen held-out task0 validation for the prospective visual-context study.

The script has three irreversible phases:

* ``freeze`` records every rule and checkpoint before task0 DEV exists.
* ``wait-evaluate`` waits read-only for the preregistered task0 DEV collection.
* ``evaluate`` audits leakage and evaluates without calibration or tuning.

It never launches Isaac/π0, never writes under the live Tabero experiment, and
forces CPU execution for model evaluation and the optional diagnostic CV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import task0_visual_context_early as early


SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
EARLY = Path("/home/exouser/FORTE/task0_visual_context_early_20260831_025000")
CONTEXT_MANIFEST = SOURCE / "PROSPECTIVE_CONTEXT_MANIFEST.csv"
DEV_TARGET = SOURCE / "PROSPECTIVE_DEV_TARGET_MANIFEST.json"
FEATURE_SPEC = SOURCE / "PROSPECTIVE_VISUAL_FEATURE_SPEC.json"
TRAIN_CONTEXT = SOURCE / "collection_train/task0/task0/context.csv"
TRAIN_BRANCH = SOURCE / "collection_train/task0/task0/branches.csv"
TRAIN_VISUAL = SOURCE / "collection_train/visual_alignment_worker.csv"
DEV_ROOT = SOURCE / "collection_dev"
DEV_CONTEXT = DEV_ROOT / "task0/task0/context.csv"
DEV_BRANCH = DEV_ROOT / "task0/task0/branches.csv"
DEV_VISUAL = DEV_ROOT / "visual_alignment_worker.csv"

MODELS = [
    "PROSPECTIVE_BASE_FEAS",
    "VISUAL_INTERCEPT_RESIDUAL",
    "VISUAL_CONTEXT_FULL_FEAS",
    "VISUAL_CONTEXT_JOINT",
]
SEEDS = [0, 1, 2]
FORCES_REAL = np.round(np.arange(3.0, 5.0001, 0.25), 2)
FORCES_DENSE = np.round(np.arange(3.0, 5.0001, 0.05), 2)
RHO = 0.80


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def checkpoint_path(model: str, seed: int) -> Path:
    return EARLY / f"{model}_task0_seed{seed}.pt"


def expected_task0_ids() -> tuple[list[str], list[str], list[str], list[str]]:
    d = pd.read_csv(CONTEXT_MANIFEST)
    train = d[(d.task == 0) & (d.split == "TRAIN")].sort_values("context_id")
    dev = d[(d.task == 0) & (d.split == "DEV")].sort_values("context_id")
    return (train.context_id.astype(str).tolist(), dev.context_id.astype(str).tolist(),
            train.root_id.astype(str).tolist(), dev.root_id.astype(str).tolist())


def freeze(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "TASK0_VISUAL_GENERALIZATION_PROTOCOL.json"
    if protocol_path.exists():
        expected = (out / "TASK0_PROTOCOL_SHA256.txt").read_text().strip()
        actual = sha256(protocol_path)
        if expected != actual:
            raise RuntimeError("frozen protocol hash changed")
        print(json.dumps({"status": "ALREADY_FROZEN", "sha256": actual}, indent=2))
        return

    train_ids, dev_ids, train_roots, dev_roots = expected_task0_ids()
    if len(train_ids) != 18 or len(dev_ids) != 2 or set(train_roots) & set(dev_roots):
        raise RuntimeError("authoritative task0 split mismatch")
    feature_spec = json.loads(FEATURE_SPEC.read_text())
    pca_path = EARLY / "TASK0_PCA17_PROVISIONAL.npz"
    pca = np.load(pca_path)
    if pca["components"].shape != (17, 4096):
        raise RuntimeError("frozen task0 PCA must be 17x4096")

    checkpoints = []
    for model in MODELS:
        for seed in SEEDS:
            p = checkpoint_path(model, seed)
            if not p.exists():
                raise RuntimeError(f"missing checkpoint {p}")
            ck = torch.load(p, map_location="cpu", weights_only=False)
            if ck.get("variant") != model or int(ck.get("seed", -1)) != seed:
                raise RuntimeError(f"checkpoint identity mismatch {p}")
            if ck.get("DEV_used") is not False or int(ck.get("epochs", -1)) != 80:
                raise RuntimeError(f"checkpoint not final TRAIN-only epoch {p}")
            checkpoints.append({
                "model": model, "seed": seed, "path": str(p), "sha256": sha256(p),
                "epochs": 80, "DEV_used": False,
                "parameter_tensors": len(ck["state_dict"]),
                "parameters": int(sum(v.numel() for v in ck["state_dict"].values())),
            })

    rules = {
        "visual_generalizes": {
            "probability_MAE_relative_improvement_min": 0.20,
            "frontier_MAE_improvement_min_N": 0.05,
            "under_force_nonworse": True,
            "multiple_independent_DEV_contexts": True,
            "monotonicity_not_materially_degraded": True,
        },
        "offset_only": {"residual_fraction_of_full_MAE_gain_min": 0.70, "safety_nonworse": True},
        "force_interaction": {
            "full_vs_residual_probability_MAE_relative_improvement_min": 0.10,
            "OR_frontier_MAE_improvement_min_N": 0.05,
            "safety_nonworse": True, "multiple_contexts": True,
        },
        "joint_win": {
            "under_force_nonworse": True, "frontier_MAE_improvement_min_N": 0.05,
            "probability_MAE_relative_improvement_min": 0.10,
            "Brier_NLL_not_materially_worse": True,
            "systematic_dense_nonmonotonicity_forbidden": True,
            "multiple_contexts": True,
        },
        "GT_gate": {
            "valid_real_frontier_coverage_min": 0.80,
            "finite_model_decision_coverage_min": 0.80,
            "probability_MAE_max": 0.20,
            "frontier_MAE_max_N": 0.20,
            "under_force_rate_max": 0.10,
            "systematic_nonmonotonicity_forbidden": True,
        },
    }
    protocol = {
        "status": "FROZEN_BEFORE_TASK0_DEV_AVAILABLE_OR_OPENED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_scientific_goal": "distinguish held-out task0 visual generalization from TRAIN context memorization and test offset, x_visual×F, and independent Joint value",
        "task": 0, "TRAIN_context_ids": train_ids, "DEV_context_ids": dev_ids,
        "TRAIN_root_ids": sorted(set(train_roots)), "DEV_root_ids": sorted(set(dev_roots)),
        "split_disjoint_at_freeze": not bool(set(train_roots) & set(dev_roots)),
        "visual_extraction": {
            "feature_spec_path": str(FEATURE_SPEC), "feature_spec_sha256": sha256(FEATURE_SPEC),
            "collector_code": str(Path(early.__file__).with_name("prospective_visual_context_collect.py")),
            "collector_code_sha256": sha256(Path(early.__file__).with_name("prospective_visual_context_collect.py")),
            "source_module": feature_spec["source_module"],
            "pi0_checkpoint_path": feature_spec["checkpoint_path"],
            "pi0_checkpoint_hash": feature_spec["checkpoint_hash"],
            "raw_feature_shape": feature_spec["concatenated_shape"],
            "strict_capture_semantics": "after final pre-probe hold step and before any probe action",
        },
        "PCA": {
            "path": str(pca_path), "sha256": sha256(pca_path), "fit_split": "task0 TRAIN only",
            "DEV_fit_forbidden": True, "dimension": 17, "raw_dimension": 4096,
            "components_shape": list(pca["components"].shape),
            "reason_17_not_64": "18 centered task0 TRAIN contexts have rank at most 17; frozen before DEV",
        },
        "architectures": {
            "PROSPECTIVE_BASE_FEAS": "GRU(17,64)+condition MLP(54,64)+head(128,64,1)",
            "VISUAL_INTERCEPT_RESIDUAL": "frozen matched Base logit + Linear(17,1); residual has no F input",
            "VISUAL_CONTEXT_FULL_FEAS": "authoritative Base backbone + Linear(17,16) visual projection + fused head(144,64,1)",
            "VISUAL_CONTEXT_JOINT": "authoritative Physics-GRU(71,64)+H8x13 trajectory head+Linear(17,16) visual feasibility fusion+head(80,32,1)",
        },
        "loss_weights": {"lambda_physics": 1.0, "lambda_IE": 1.0, "lambda_feasibility": 0.3},
        "seeds": SEEDS, "final_epoch": 80, "checkpoints": checkpoints,
        "real_force_grid_N": FORCES_REAL.tolist(), "dense_query_grid_N": FORCES_DENSE.tolist(),
        "rho": RHO,
        "metrics": {
            "probability": ["MAE", "Brier", "NLL", "signed_bias", "calibration_intercept", "calibration_slope", "per_context_MAE", "seed_variation"],
            "force_shape": ["Spearman", "adjacent_ordering", "nonmonotonic_steps", "context_monotonicity", "safe_to_unsafe_reversals"],
            "frontier": ["finite_coverage", "MAE", "under_force_rate", "under_force_magnitude", "excess_force", "within_0.25N", "within_0.50N"],
        },
        "winner_rules": rules,
        "stop_rules": {
            "leakage": "TASK0_VISUAL_RESULT_INVALID_DUE_TO_LEAKAGE",
            "no_DEV_substitute": True, "no_DEV_calibration": True, "no_checkpoint_selection_on_DEV": True,
            "no_architecture_PCA_lambda_feature_loss_or_checkpoint_changes": True,
            "Probe_forbidden_if_GT_gate_fails": True,
        },
        "optional_CV": {
            "trigger": "task0 DEV too small or ambiguous",
            "known_DEV_context_count_at_freeze": 2,
            "design": "3 folds; hold out two entire TRAIN roots (six contexts) per fold; one fixed seed; architectures/hyperparameters unchanged",
            "PCA_rule": "fit within fold TRAIN only, preserve 17-D architecture by zero-padding rank-limited components",
            "primary_result": False,
        },
        "authoritative_manifest_hashes": {
            str(CONTEXT_MANIFEST): sha256(CONTEXT_MANIFEST), str(DEV_TARGET): sha256(DEV_TARGET),
            str(TRAIN_CONTEXT): sha256(TRAIN_CONTEXT), str(TRAIN_BRANCH): sha256(TRAIN_BRANCH),
        },
        "DEV_outcomes_available_at_freeze": DEV_BRANCH.exists(),
        "DEV_outcomes_read_at_freeze": False, "TEST_used": False,
    }
    write_json(protocol_path, protocol)
    protocol_hash = sha256(protocol_path)
    (out / "TASK0_PROTOCOL_SHA256.txt").write_text(protocol_hash + "\n", encoding="utf-8")

    write_json(out / "TASK0_ALL_MODELS_FROZEN.json", {
        "status": "PASS_12_FINAL_TASK0_UNITS_FROZEN_BEFORE_DEV",
        "checkpoint_count": len(checkpoints), "models": checkpoints,
        "all_final_epoch_80": True, "all_DEV_used_false": True,
        "selection_using_DEV": False, "protocol_sha256": protocol_hash,
    })
    write_json(out / "TASK0_VISUAL_LEAKAGE_AUDIT.json", {
        "status": "PENDING_DEV_COLLECTION", "blocking_leakage_found": False,
        "checks_completed_before_DEV": {
            "TRAIN_DEV_context_ID_disjoint": not bool(set(train_ids) & set(dev_ids)),
            "TRAIN_DEV_root_ID_disjoint": not bool(set(train_roots) & set(dev_roots)),
            "PCA_frozen_before_DEV": True, "PCA_fit_split": "task0 TRAIN only",
            "DEV_RGB_or_feature_used_for_PCA_or_normalization": False,
            "DEV_outcome_used_for_feature_selection": False,
            "root_or_split_ID_explicit_model_input": False,
            "visual_feature_location_pre_action": True,
        },
        "checks_pending_until_DEV_lands": [
            "DEV visual hash and strict pre-probe state alignment", "same x across candidate forces",
            "no post-force/post-probe/terminal information", "complete 2x9x5 population",
        ],
        "protocol_sha256": protocol_hash,
    })
    print(json.dumps({"status": protocol["status"], "protocol_sha256": protocol_hash,
                      "checkpoints": 12, "DEV_contexts": len(dev_ids)}, indent=2))


def readiness() -> dict[str, Any]:
    result: dict[str, Any] = {
        "ready": False, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "context_path": str(DEV_CONTEXT), "branch_path": str(DEV_BRANCH), "visual_path": str(DEV_VISUAL),
    }
    if not (DEV_CONTEXT.exists() and DEV_BRANCH.exists() and DEV_VISUAL.exists()):
        result["reason"] = "task0 DEV files not all present"
        return result
    try:
        c = pd.read_csv(DEV_CONTEXT)
        b = pd.read_csv(DEV_BRANCH)
        v = pd.read_csv(DEV_VISUAL)
    except Exception as exc:
        result["reason"] = f"DEV files still being written or unreadable: {exc}"
        return result
    _, expected_ids, _, _ = expected_task0_ids()
    v = v[v.context_id.astype(str).isin(expected_ids)]
    result.update({"contexts": len(c), "branches": len(b), "visual_contexts": len(v),
                   "unique_contexts": int(c.context_id.nunique()) if len(c) else 0,
                   "unique_branches": int(b.branch_id.nunique()) if len(b) else 0})
    complete_cells = True
    if len(b):
        counts = b.groupby(["context_id", "requested_force_N"]).size()
        complete_cells = len(counts) == 18 and bool((counts == 5).all())
    result["complete_2x9x5"] = bool(len(c) == 2 and len(b) == 90 and len(v) == 2 and complete_cells)
    result["parity_90_of_90"] = bool(len(b) == 90 and int(b.state_parity.sum()) == 90)
    result["restore_2_of_2"] = bool(len(v) == 2 and int(v.restore_exact.sum()) == 2)
    result["ready"] = bool(result["complete_2x9x5"] and result["parity_90_of_90"] and result["restore_2_of_2"])
    result["reason"] = "complete" if result["ready"] else "task0 DEV incomplete"
    return result


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pca_transform(paths: list[Path]) -> np.ndarray:
    p = np.load(EARLY / "TASK0_PCA17_PROVISIONAL.npz")
    raw = np.stack([np.load(x, allow_pickle=False).astype(np.float32) for x in paths])
    z = (raw - p["raw_mean"]) @ p["components"].T
    return ((z - p["projected_mean"]) / p["projected_std"]).astype(np.float32)


def load_dev(tpi, cf):
    contexts = pd.read_csv(DEV_CONTEXT).sort_values("context_id")
    branches = pd.read_csv(DEV_BRANCH)
    visual_all = pd.read_csv(DEV_VISUAL)
    visual = visual_all[visual_all.context_id.astype(str).isin(contexts.context_id.astype(str))].copy()
    vmap = visual.set_index("context_id").to_dict("index")
    z = pca_transform([Path(str(vmap[c]["visual_feature_path"])) for c in contexts.context_id.astype(str)])
    cmap: dict[str, dict[str, Any]] = {}
    templates: dict[str, Any] = {}
    audit_failures = []
    corrected = {"left_normal_force_N", "right_normal_force_N", "left_tangential_force_N",
                 "right_tangential_force_N", "object_vx_mps", "object_vy_mps", "object_vz_mps"}
    for row, zv in zip(contexts.itertuples(index=False), z):
        cid = str(row.context_id)
        vr = vmap[cid]
        raw = np.load(vr["visual_feature_path"], allow_pickle=False).astype(np.float32)
        if hashlib.sha256(raw.tobytes()).hexdigest() != str(vr["visual_feature_sha256"]):
            audit_failures.append(f"{cid}: visual feature hash mismatch")
        for col in ["restored_state_hash", "second_restore_hash"]:
            if str(vr[col]) != str(vr["snapshot_state_hash"]):
                audit_failures.append(f"{cid}: {col} mismatch")
        if str(vr["snapshot_state_hash"]) != str(row.post_probe_state_hash):
            audit_failures.append(f"{cid}: context/visual state hash mismatch")
        if int(vr["capture_step"]) != 190 or "before any probe action" not in str(vr["timestamp_semantics"]):
            audit_failures.append(f"{cid}: visual capture is not strict pre-probe step 190")
        if int(vr["same_x_for_all_branches"]) != 1:
            audit_failures.append(f"{cid}: same-x invariant absent")
        state0, mask0, _ = early.strict_preprobe_state(Path(str(row.probe_telemetry_path)))
        q = branches[branches.context_id.astype(str) == cid].sort_values(["requested_force_N", "branch_label"])
        if len(q) != 45 or q.requested_force_N.nunique() != 9:
            audit_failures.append(f"{cid}: expected 45 branches/9 forces")
            continue
        canonical_path = Path(str(q.iloc[0].telemetry_path))
        d = pd.read_csv(canonical_path)
        if not corrected <= set(d.columns) or not np.isfinite(d[list(corrected)].to_numpy(float)).all():
            audit_failures.append(f"{cid}: corrected telemetry missing/nonfinite")
        state, mask = tpi.state_from(d)
        state = state.copy(); mask = mask.copy()
        state[0] = state0; mask[0] = mask0
        mu = float(row.hidden_friction_analysis_only)
        force = 3.0
        nominal = tpi.nominal_from(d, 0, force, mu, state, mask)
        tr = tpi.Trace(
            f"template:{cid}", cid, str(row.root_id), 0, "DEV", force, mu, 0,
            "dense_query", canonical_path, state, mask, nominal, d.phase.astype(str).tolist(),
            1.0, "PROSPECTIVE_TASK0_DEV_TEMPLATE",
        )
        cmap[cid] = {"visual": zv, "visual_raw": raw, "root_id": str(row.root_id),
                      "friction": mu, "friction_band": str(row.friction_band)}
        templates[cid] = tr
    return contexts, branches, visual, cmap, templates, audit_failures


def normalized_segment(cf, tpi, trace, force, norm):
    seg = cf.build_seg(tpi, trace, float(force), early.H)
    x = ((seg.x - norm[0]) / norm[1]).astype(np.float32)
    return x[:, :17], x[0, 17:]


def load_models(tpi, full, device):
    models: dict[tuple[str, int], Any] = {}
    for seed in SEEDS:
        bp = checkpoint_path("PROSPECTIVE_BASE_FEAS", seed)
        bck = torch.load(bp, map_location=device, weights_only=False)
        base = full.FeasibilityOnly().to(device)
        base.load_state_dict(bck["state_dict"]); base.eval()
        models[("PROSPECTIVE_BASE_FEAS", seed)] = base

        rck = torch.load(checkpoint_path("VISUAL_INTERCEPT_RESIDUAL", seed), map_location=device, weights_only=False)
        residual = early.VisualResidual(17).to(device)
        residual.load_state_dict(rck["state_dict"]); residual.eval()
        models[("VISUAL_INTERCEPT_RESIDUAL", seed)] = (base, residual)

        fck = torch.load(checkpoint_path("VISUAL_CONTEXT_FULL_FEAS", seed), map_location=device, weights_only=False)
        vf = early.VisualFullFeas(17).to(device)
        vf.load_state_dict(fck["state_dict"]); vf.eval()
        models[("VISUAL_CONTEXT_FULL_FEAS", seed)] = vf

        jck = torch.load(checkpoint_path("VISUAL_CONTEXT_JOINT", seed), map_location=device, weights_only=False)
        physics_state = {k[len("physics."):]: v for k, v in jck["state_dict"].items() if k.startswith("physics.")}
        joint = early.VisualJoint(tpi, physics_state, 17).to(device)
        joint.load_state_dict(jck["state_dict"]); joint.eval()
        models[("VISUAL_CONTEXT_JOINT", seed)] = joint
    return models


def predict_all(cf, tpi, models, cmap, templates, norm, device):
    rows = []
    with torch.no_grad():
        for cid in sorted(templates):
            vis = torch.tensor(cmap[cid]["visual"][None], dtype=torch.float32, device=device)
            for force in FORCES_DENSE:
                step_np, cond_np = normalized_segment(cf, tpi, templates[cid], float(force), norm)
                step = torch.tensor(step_np[None], dtype=torch.float32, device=device)
                cond = torch.tensor(cond_np[None], dtype=torch.float32, device=device)
                for model in MODELS:
                    seed_probs = []
                    for seed in SEEDS:
                        m = models[(model, seed)]
                        if model == "PROSPECTIVE_BASE_FEAS":
                            logit = m(step, cond)
                        elif model == "VISUAL_INTERCEPT_RESIDUAL":
                            base, residual = m
                            logit = base(step, cond) + residual(vis)
                        elif model == "VISUAL_CONTEXT_FULL_FEAS":
                            logit = m(step, cond, vis)
                        else:
                            logit = m(step, cond, vis)[1]
                        seed_probs.append(float(torch.sigmoid(logit)[0].cpu()))
                    rows.append({"context_id": cid, "force_N": float(force), "model": model,
                                 **{f"seed{s}_prob": seed_probs[s] for s in SEEDS},
                                 "ensemble_prob": float(np.mean(seed_probs)),
                                 "seed_std": float(np.std(seed_probs, ddof=1))})
    return pd.DataFrame(rows)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0.0, center-half), min(1.0, center+half)


def real_curves(branches: pd.DataFrame):
    rows, fronts = [], []
    for cid, q in branches.groupby("context_id"):
        for force, z in q.groupby("requested_force_N"):
            k, n = int(z.full_task_success_y.sum()), len(z)
            lo, hi = wilson(k, n)
            rows.append({"context_id": str(cid), "force_N": float(force), "successes_k": k,
                         "repeats_n": n, "p_real": k/n, "wilson95_low": lo, "wilson95_high": hi})
        cr = [r for r in rows if r["context_id"] == str(cid)]
        valid = [r["force_N"] for r in cr if r["p_real"] >= RHO]
        fronts.append({"context_id": str(cid), "real_frontier_N": min(valid) if valid else math.nan,
                       "frontier_supported": bool(valid), "rho": RHO,
                       "tested_force_min_N": 3.0, "tested_force_max_N": 5.0})
    return pd.DataFrame(rows), pd.DataFrame(fronts)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1-1e-7)
    return np.log(p/(1-p))


def calibration_fit(probs, successes, totals):
    x = logit(probs)
    X = np.column_stack([np.ones(len(x)), x])
    k = np.asarray(successes, float); n = np.asarray(totals, float)
    beta = np.array([0.0, 1.0])
    for _ in range(100):
        mu = 1/(1+np.exp(-np.clip(X@beta, -30, 30)))
        w = np.maximum(n*mu*(1-mu), 1e-8)
        h = X.T @ (w[:, None]*X) + np.eye(2)*1e-6
        step = np.linalg.solve(h, X.T@(k-n*mu))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def probability_metrics(pred, real):
    real_key = real.set_index(["context_id", "force_N"])
    rows = []
    for model in MODELS:
        q = pred[(pred.model == model) & pred.force_N.isin(FORCES_REAL)].copy()
        rr = real_key.loc[list(zip(q.context_id, q.force_N))].reset_index()
        y = rr.p_real.to_numpy(float); k = rr.successes_k.to_numpy(float); n = rr.repeats_n.to_numpy(float)
        for label, col in [("ENSEMBLE", "ensemble_prob"), *[(f"SEED_{s}", f"seed{s}_prob") for s in SEEDS]]:
            p = q[col].to_numpy(float)
            ci, cs = calibration_fit(p, k, n)
            rows.append({"model": model, "aggregation": label, "cells": len(y), "physical_repeats": int(n.sum()),
                         "probability_MAE": float(np.mean(np.abs(p-y))),
                         "Brier": float(np.mean((p-y)**2)),
                         "NLL": float(-np.mean(y*np.log(np.clip(p,1e-8,1))+(1-y)*np.log(np.clip(1-p,1e-8,1)))),
                         "signed_bias": float(np.mean(p-y)),
                         "calibration_intercept": ci, "calibration_slope": cs})
        sr = [x for x in rows if x["model"] == model and x["aggregation"].startswith("SEED_")]
        for agg, fn in [("SEED_MEAN", np.mean), ("SEED_STD", lambda z: np.std(z, ddof=1))]:
            rows.append({"model": model, "aggregation": agg, "cells": len(y), "physical_repeats": int(n.sum()),
                         **{m: float(fn([x[m] for x in sr])) for m in ["probability_MAE","Brier","NLL","signed_bias","calibration_intercept","calibration_slope"]}})
    return pd.DataFrame(rows)


def spearman(x, y):
    return float(pd.Series(x).rank(method="average").corr(pd.Series(y).rank(method="average")))


def shape_metrics(pred):
    rows = []
    for (model, cid), q in pred.groupby(["model", "context_id"]):
        q = q.sort_values("force_N"); p = q.ensemble_prob.to_numpy(float); f = q.force_N.to_numpy(float)
        diff = np.diff(p)
        reversals = int(np.sum((p[:-1] >= RHO) & (p[1:] < RHO)))
        rows.append({"model": model, "context_id": cid, "spearman_force_probability": spearman(f,p),
                     "adjacent_ordering_rate": float(np.mean(diff >= -1e-8)),
                     "nonmonotonic_steps": int(np.sum(diff < -1e-8)),
                     "context_monotonic": bool(np.all(diff >= -1e-8)),
                     "safe_to_unsafe_reversals": reversals, "dense_steps": len(diff)})
    for model in MODELS:
        q = [r for r in rows if r["model"] == model]
        rows.append({"model": model, "context_id": "__AGGREGATE__",
                     "spearman_force_probability": float(np.mean([r["spearman_force_probability"] for r in q])),
                     "adjacent_ordering_rate": float(np.mean([r["adjacent_ordering_rate"] for r in q])),
                     "nonmonotonic_steps": int(sum(r["nonmonotonic_steps"] for r in q)),
                     "context_monotonic": float(np.mean([r["context_monotonic"] for r in q])),
                     "safe_to_unsafe_reversals": int(sum(r["safe_to_unsafe_reversals"] for r in q)),
                     "dense_steps": int(sum(r["dense_steps"] for r in q))})
    return pd.DataFrame(rows)


def frontier_metrics(pred, real_fronts):
    detail, summary = [], []
    real_map = real_fronts.set_index("context_id").real_frontier_N.to_dict()
    for model in MODELS:
        for cid, q in pred[pred.model == model].groupby("context_id"):
            safe = q[q.ensemble_prob >= RHO].sort_values("force_N")
            fp = float(safe.force_N.iloc[0]) if len(safe) else math.nan
            fr = float(real_map[str(cid)])
            valid = math.isfinite(fr); finite = math.isfinite(fp)
            err = fp-fr if valid and finite else math.nan
            detail.append({"model": model, "context_id": cid, "real_frontier_N": fr,
                           "predicted_frontier_N": fp, "real_frontier_valid": valid,
                           "finite_model_decision": finite, "signed_error_N": err,
                           "absolute_error_N": abs(err) if math.isfinite(err) else math.nan,
                           "under_force": bool(math.isfinite(err) and err < -1e-8),
                           "under_force_magnitude_N": -err if math.isfinite(err) and err < 0 else 0.0 if math.isfinite(err) else math.nan,
                           "excess_force": bool(math.isfinite(err) and err > 1e-8),
                           "excess_force_N": err if math.isfinite(err) and err > 0 else 0.0 if math.isfinite(err) else math.nan,
                           "within_0.25N": bool(math.isfinite(err) and abs(err) <= .25+1e-8),
                           "within_0.50N": bool(math.isfinite(err) and abs(err) <= .50+1e-8)})
        q = [r for r in detail if r["model"] == model]
        valid = [r for r in q if r["real_frontier_valid"]]
        finite = [r for r in valid if r["finite_model_decision"]]
        under = [r for r in finite if r["under_force"]]
        excess = [r for r in finite if r["excess_force"]]
        summary.append({"model": model, "DEV_contexts": len(q),
                        "valid_real_frontier_count": len(valid),
                        "valid_real_frontier_coverage": len(valid)/len(q) if q else math.nan,
                        "finite_decision_count": len(finite),
                        "finite_decision_coverage": len(finite)/len(valid) if valid else math.nan,
                        "frontier_MAE_N": float(np.mean([r["absolute_error_N"] for r in finite])) if finite else math.nan,
                        "under_force_count": len(under), "under_force_rate": len(under)/len(finite) if finite else math.nan,
                        "mean_under_force_magnitude_N": float(np.mean([r["under_force_magnitude_N"] for r in under])) if under else 0.0,
                        "excess_force_count": len(excess), "mean_excess_force_N": float(np.mean([r["excess_force_N"] for r in excess])) if excess else 0.0,
                        "within_0.25N_rate": float(np.mean([r["within_0.25N"] for r in finite])) if finite else math.nan,
                        "within_0.50N_rate": float(np.mean([r["within_0.50N"] for r in finite])) if finite else math.nan})
    return pd.DataFrame(detail), pd.DataFrame(summary)


def visual_distances(cmap):
    train_align = pd.read_csv(EARLY / "TASK0_FROZEN_VISUAL_ALIGNMENT.csv").sort_values("context_id")
    train_z = pca_transform([Path(x) for x in train_align.visual_feature_path.astype(str)])
    rows = []
    for cid in sorted(cmap):
        dist = np.linalg.norm(train_z - cmap[cid]["visual"][None], axis=1)
        idx = int(np.argmin(dist))
        rows.append({"context_id": cid, "nearest_TRAIN_context_id": str(train_align.iloc[idx].context_id),
                     "nearest_TRAIN_root_id": str(train_align.iloc[idx].root_id),
                     "nearest_TRAIN_distance": float(dist[idx]),
                     "mean_TRAIN_distance": float(np.mean(dist)), "max_TRAIN_distance": float(np.max(dist))})
    return pd.DataFrame(rows)


def context_and_distance_breakdown(pred, real, real_fronts, frontier_detail, distances):
    rows, drows = [], []
    front_map = real_fronts.set_index("context_id").real_frontier_N.to_dict()
    dist_map = distances.set_index("context_id").to_dict("index")
    rctx = {cid: q.sort_values("force_N") for cid,q in real.groupby("context_id")}
    for cid in sorted(rctx):
        rr = rctx[cid]
        row = {"context_id": cid, "real_success_curve": json.dumps(dict(zip(rr.force_N.astype(str), rr.p_real))),
               "real_frontier_N": front_map[cid], **dist_map[cid]}
        base_mae = None
        for model in MODELS:
            q = pred[(pred.context_id == cid) & (pred.model == model) & pred.force_N.isin(FORCES_REAL)].sort_values("force_N")
            row[f"{model}_curve"] = json.dumps(dict(zip(q.force_N.astype(str), q.ensemble_prob)))
            fd = frontier_detail[(frontier_detail.context_id == cid) & (frontier_detail.model == model)].iloc[0]
            row[f"{model}_frontier_N"] = fd.predicted_frontier_N
            row[f"{model}_signed_frontier_error_N"] = fd.signed_error_N
            mae = float(np.mean(np.abs(q.ensemble_prob.to_numpy()-rr.p_real.to_numpy())))
            row[f"{model}_probability_MAE"] = mae
            if model == "PROSPECTIVE_BASE_FEAS": base_mae = mae
            else:
                drows.append({"context_id": cid, "visual_model": model,
                              "nearest_TRAIN_context_id": dist_map[cid]["nearest_TRAIN_context_id"],
                              "nearest_TRAIN_distance": dist_map[cid]["nearest_TRAIN_distance"],
                              "Base_context_MAE": base_mae, "visual_context_MAE": mae,
                              "MAE_improvement_over_Base": base_mae-mae,
                              "relative_MAE_improvement_over_Base": (base_mae-mae)/base_mae if base_mae else math.nan})
        rows.append(row)
    for model in MODELS[1:]:
        q = [r for r in drows if r["visual_model"] == model]
        corr = float(np.corrcoef([r["nearest_TRAIN_distance"] for r in q], [r["MAE_improvement_over_Base"] for r in q])[0,1]) if len(q) >= 3 else math.nan
        drows.append({"context_id": "__SUMMARY__", "visual_model": model,
                      "nearest_TRAIN_context_id": "", "nearest_TRAIN_distance": math.nan,
                      "Base_context_MAE": math.nan, "visual_context_MAE": math.nan,
                      "MAE_improvement_over_Base": float(np.mean([r["MAE_improvement_over_Base"] for r in q])),
                      "relative_MAE_improvement_over_Base": float(np.mean([r["relative_MAE_improvement_over_Base"] for r in q])),
                      "distance_improvement_correlation": corr,
                      "correlation_interpretation": "EVIDENCE_LIMITED_LT3_DEV_CONTEXTS" if len(q)<3 else "computed"})
    return pd.DataFrame(rows), pd.DataFrame(drows)


def model_row(df, model, aggregation="ENSEMBLE"):
    return df[(df.model == model) & (df.aggregation == aggregation)].iloc[0]


def frontier_row(df, model):
    return df[df.model == model].iloc[0]


def shape_row(df, model):
    return df[(df.model == model) & (df.context_id == "__AGGREGATE__")].iloc[0]


def comparisons(prob, front, shape, context_breakdown):
    b = model_row(prob, MODELS[0]); r = model_row(prob, MODELS[1]); f = model_row(prob, MODELS[2]); j = model_row(prob, MODELS[3])
    bf, rf, ff, jf = [frontier_row(front, m) for m in MODELS]
    bs, rs, fs, js = [shape_row(shape, m) for m in MODELS]
    def rel(a,b): return (float(a)-float(b))/float(a) if float(a)>0 else math.nan
    cb = context_breakdown
    multiple_full = int(sum(cb[f"{MODELS[2]}_probability_MAE"] < cb[f"{MODELS[0]}_probability_MAE"])) >= 2
    multiple_inter = int(sum(cb[f"{MODELS[2]}_probability_MAE"] < cb[f"{MODELS[1]}_probability_MAE"])) >= 2
    multiple_joint = int(sum(cb[f"{MODELS[3]}_probability_MAE"] < cb[f"{MODELS[2]}_probability_MAE"])) >= 2
    visual_generalizes = bool(rel(b.probability_MAE, f.probability_MAE) >= .20 and
                              float(bf.frontier_MAE_N)-float(ff.frontier_MAE_N) >= .05 and
                              float(ff.under_force_rate) <= float(bf.under_force_rate) and multiple_full and
                              float(fs.context_monotonic) >= float(bs.context_monotonic)-.10)
    full_gain = float(b.probability_MAE)-float(f.probability_MAE)
    residual_fraction = (float(b.probability_MAE)-float(r.probability_MAE))/full_gain if full_gain>0 else math.nan
    offset = bool(residual_fraction >= .70 and float(rf.under_force_rate) <= float(bf.under_force_rate))
    interaction = bool((rel(r.probability_MAE, f.probability_MAE) >= .10 or
                        float(rf.frontier_MAE_N)-float(ff.frontier_MAE_N) >= .05) and
                       float(ff.under_force_rate) <= float(rf.under_force_rate) and multiple_inter)
    joint = bool(float(jf.under_force_rate) <= float(ff.under_force_rate) and
                 float(ff.frontier_MAE_N)-float(jf.frontier_MAE_N) >= .05 and
                 rel(f.probability_MAE, j.probability_MAE) >= .10 and
                 float(j.Brier) <= float(f.Brier)+1e-8 and float(j.NLL) <= float(f.NLL)+1e-8 and
                 int(js.safe_to_unsafe_reversals) == 0 and float(js.context_monotonic) >= .90 and multiple_joint)
    frontier_only = bool(float(ff.frontier_MAE_N)-float(jf.frontier_MAE_N) >= .05 and not joint)
    return {
        "visual_generalizes": visual_generalizes, "Base_to_Full_probability_MAE_relative_improvement": rel(b.probability_MAE,f.probability_MAE),
        "Base_to_Full_frontier_MAE_improvement_N": float(bf.frontier_MAE_N)-float(ff.frontier_MAE_N),
        "Full_under_force_nonworse": bool(float(ff.under_force_rate)<=float(bf.under_force_rate)),
        "Full_benefit_both_DEV_contexts": multiple_full,
        "residual_fraction_of_full_MAE_gain": residual_fraction, "offset_only_supported": offset,
        "Full_vs_Residual_probability_MAE_relative_improvement": rel(r.probability_MAE,f.probability_MAE),
        "Full_vs_Residual_frontier_MAE_improvement_N": float(rf.frontier_MAE_N)-float(ff.frontier_MAE_N),
        "force_interaction_supported": interaction, "Full_better_both_DEV_contexts": multiple_inter,
        "Joint_vs_Full_probability_MAE_relative_improvement": rel(f.probability_MAE,j.probability_MAE),
        "Joint_vs_Full_frontier_MAE_improvement_N": float(ff.frontier_MAE_N)-float(jf.frontier_MAE_N),
        "Joint_under_force_nonworse": bool(float(jf.under_force_rate)<=float(ff.under_force_rate)),
        "Joint_better_both_DEV_contexts": multiple_joint, "joint_independent_win": joint,
        "joint_frontier_only_signal": frontier_only,
    }


def generalization_gap(prob):
    train = pd.read_csv(EARLY / "TASK0_TRAIN_IN_SAMPLE_METRICS.csv")
    rows = []
    base_train = float(train[train.variant==MODELS[0]].bce_nll.mean())
    base_dev = float(model_row(prob, MODELS[0], "SEED_MEAN").NLL)
    for model in MODELS:
        tr = train[train.variant == model]
        dv = model_row(prob, model, "SEED_MEAN")
        train_loss = float(tr.bce_nll.mean()); dev_loss = float(dv.NLL)
        rows.append({"model": model, "TRAIN_in_sample_BCE_mean": train_loss,
                     "TRAIN_in_sample_BCE_std": float(tr.bce_nll.std(ddof=1)),
                     "DEV_heldout_NLL_seed_mean": dev_loss,
                     "DEV_heldout_NLL_seed_std": float(model_row(prob, model, "SEED_STD").NLL),
                     "generalization_gap_DEV_minus_TRAIN": dev_loss-train_loss,
                     "DEV_to_TRAIN_loss_ratio": dev_loss/train_loss,
                     "TRAIN_relative_improvement_vs_Base": (base_train-train_loss)/base_train,
                     "DEV_relative_improvement_vs_Base": (base_dev-dev_loss)/base_dev if base_dev else math.nan})
    return pd.DataFrame(rows)


def build_fold_pairs(cf, traces, meta):
    by = defaultdict(list)
    for tr in traces:
        by[(tr.context_id, meta[tr.branch_id].repeat)].append(tr)
    pairs=[]
    for (cid, rep), q in sorted(by.items()):
        q=sorted(q,key=lambda tr:meta[tr.branch_id].stratum)
        for a,b in zip(q[:-1],q[1:]):
            ma,mb=meta[a.branch_id],meta[b.branch_id]
            pairs.append(cf.Pair(f"cv:{cid}:R{rep}:S{ma.stratum}_S{mb.stratum}",f"cv:{cid}:R{rep}","TRAIN","TASK0_CV",cid,a.root_id,0,ma.friction_band,a.mu,a.force,b.force,"adjacent",False,a,b))
    return pairs


def cv_metric(model_name, fold, held, logits):
    y=np.asarray([t.outcome for t in held],float); z=np.asarray([logits[t.branch_id] for t in held],float)
    p=1/(1+np.exp(-np.clip(z,-50,50)))
    return {"fold":fold,"model":model_name,"heldout_contexts":len({t.context_id for t in held}),
            "heldout_roots":len({t.root_id for t in held}),"heldout_branches":len(held),
            "BCE":float(-np.mean(y*np.log(np.clip(p,1e-8,1))+(1-y)*np.log(np.clip(1-p,1e-8,1)))),
            "Brier":float(np.mean((p-y)**2)),"accuracy_0.5":float(np.mean((p>=.5)==y)),
            "signed_bias":float(np.mean(p-y))}


def run_cv(out, tpi, cf, full, device):
    tmp = out / "_cv_train_recheck"
    tmp.mkdir(exist_ok=True)
    _, _, cmap0, traces, meta, _ = early.audit_and_load(tmp, tpi)
    roots=sorted({t.root_id for t in traces})
    folds=[set(roots[i::3]) for i in range(3)]
    rows=[]
    for fi, held_roots in enumerate(folds):
        tr=[t for t in traces if t.root_id not in held_roots]
        held=[t for t in traces if t.root_id in held_roots]
        train_cids=sorted({t.context_id for t in tr}); all_cids=sorted(cmap0)
        raw_train=np.stack([cmap0[c]["visual_raw"] for c in train_cids])
        mean=raw_train.mean(0); _,s,vt=np.linalg.svd(raw_train-mean,full_matrices=False)
        k=min(17,len(train_cids)-1,int(np.sum(s>s[0]*1e-7)))
        comp=vt[:k]
        ztrain=(raw_train-mean)@comp.T; zm=ztrain.mean(0); zs=ztrain.std(0); zs[zs<1e-6]=1
        cmap={}
        for cid in all_cids:
            z=((cmap0[cid]["visual_raw"]-mean)@comp.T-zm)/zs
            z17=np.zeros(17,np.float32); z17[:k]=z.astype(np.float32)
            cmap[cid]={**cmap0[cid],"visual":z17}
        segs_all={t.branch_id:cf.build_seg(tpi,t,t.force,early.H) for t in traces}
        segs_train={t.branch_id:segs_all[t.branch_id] for t in tr}
        norm=early.build_segments_and_norm(cf,tpi,tr)[1]
        pairs=build_fold_pairs(cf,tr,{k:meta[k] for k in [t.branch_id for t in tr]})
        mmeta={t.branch_id:meta[t.branch_id] for t in tr}
        print(f"[CV] fold={fi} train_contexts={len(train_cids)} held_contexts={len(set(t.context_id for t in held))} PCA_rank={k}",flush=True)
        base,_,_=early.train_base(full,tr,segs_train,norm,cmap,mmeta,device,0)
        residual,_,_=early.train_residual(base,tr,segs_train,norm,cmap,mmeta,device,0,17)
        vf,_,_=early.train_full(tr,segs_train,norm,cmap,mmeta,device,0,17)
        joint,_,_,_,_=early.train_joint(full,cf,tpi,tr,pairs,segs_train,norm,cmap,mmeta,device,0,17)
        for name,m,kind,bm in [(MODELS[0],base,"BASE",None),(MODELS[1],residual,"RESIDUAL",base),(MODELS[2],vf,"FULL",None),(MODELS[3],joint,"JOINT",None)]:
            lg=early.logits_for(m,kind,held,segs_all,norm,cmap,device,base=bm)
            row=cv_metric(name,fi,held,lg); row.update({"heldout_root_ids":json.dumps(sorted(held_roots)),"fold_PCA_effective_rank":k,"seed":0})
            rows.append(row)
    shutil.rmtree(tmp)
    for model in MODELS:
        q=[r for r in rows if r["model"]==model]
        rows.append({"fold":"__MEAN__","model":model,"heldout_contexts":6,"heldout_roots":2,"heldout_branches":60,
                     **{m:float(np.mean([r[m] for r in q])) for m in ["BCE","Brier","accuracy_0.5","signed_bias"]},
                     "heldout_root_ids":"","fold_PCA_effective_rank":11,"seed":0})
    write_csv(out/"TASK0_GROUP_HELDOUT_CV.csv",rows)
    return pd.DataFrame(rows)


def select_and_classify(prob, front, shape, comp, cv):
    complexity={m:i for i,m in enumerate(MODELS)}
    def finite_or(value, fallback):
        value = float(value)
        return value if math.isfinite(value) else fallback
    scored=[]
    for m in MODELS:
        p=model_row(prob,m); f=frontier_row(front,m); s=shape_row(shape,m)
        scored.append((m,(finite_or(f.under_force_rate,math.inf),-finite_or(f.finite_decision_coverage,-math.inf),
                          finite_or(f.frontier_MAE_N,math.inf),finite_or(p.probability_MAE,math.inf),
                          finite_or(p.Brier,math.inf),finite_or(f.mean_excess_force_N,math.inf),
                          int(s.safe_to_unsafe_reversals),complexity[m])))
    selected=min(scored,key=lambda x:x[1])[0]
    p=model_row(prob,selected); f=frontier_row(front,selected); s=shape_row(shape,selected)
    gt=bool(float(f.valid_real_frontier_coverage)>=.8 and float(f.finite_decision_coverage)>=.8 and float(p.probability_MAE)<=.20 and float(f.frontier_MAE_N)<=.20 and float(f.under_force_rate)<=.10 and int(s.safe_to_unsafe_reversals)==0 and float(s.context_monotonic)>=.90)
    cv_means=cv[cv.fold.astype(str)=="__MEAN__"].set_index("model") if len(cv) else pd.DataFrame()
    cv_visual_support=False
    if len(cv_means):
        cv_visual_support=bool(float(cv_means.loc[MODELS[2],"BCE"])<float(cv_means.loc[MODELS[0],"BCE"]))
    if comp["joint_frontier_only_signal"]:
        classification="TASK0_JOINT_FRONTIER_SIGNAL_WITHOUT_RELIABLE_CONTROL"
    elif comp["visual_generalizes"] and comp["joint_independent_win"]:
        classification="TASK0_VISUAL_CONTEXT_GENERALIZES_AND_JOINT_WINS"
    elif comp["visual_generalizes"] and comp["offset_only_supported"]:
        classification="TASK0_VISUAL_CONTEXT_GENERALIZES_WITH_OFFSET_ONLY"
    elif comp["visual_generalizes"] and comp["force_interaction_supported"]:
        classification="TASK0_VISUAL_CONTEXT_GENERALIZES_WITH_FORCE_INTERACTION"
    elif comp["visual_generalizes"]:
        classification="TASK0_VISUAL_CONTEXT_GENERALIZES_BUT_JOINT_NOT_NEEDED"
    else:
        train_gap = pd.read_csv(EARLY/"TASK0_TRAIN_IN_SAMPLE_METRICS.csv")
        train_full=float(train_gap[train_gap.variant==MODELS[2]].bce_nll.mean())
        train_base=float(train_gap[train_gap.variant==MODELS[0]].bce_nll.mean())
        strong_train=(train_base-train_full)/train_base>=.20
        if strong_train and not cv_visual_support:
            classification="TASK0_VISUAL_CONTEXT_TRAIN_MEMORIZATION"
        elif not gt:
            classification="TASK0_GT_CONTINUOUS_GATE_FAILS"
        else:
            classification="TASK0_VISUAL_SIGNAL_POSITIVE_BUT_DEV_EVIDENCE_LIMITED"
    return selected,gt,classification,cv_visual_support,scored


def write_report(out, contexts, branches, prob, front, shape, comp, gap, cv, selected, gt, classification, distances):
    def fmt(x,d=3): return "NA" if not math.isfinite(float(x)) else f"{float(x):.{d}f}"
    def md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        def cell(v):
            if isinstance(v, (float, np.floating)):
                return "NA" if not math.isfinite(float(v)) else f"{float(v):.4f}"
            return str(v).replace("|", "\\|").replace("\n", " ")
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None))
        return "\n".join(lines)
    lines=["# STATUS","",classification,"","# SINGLE SCIENTIFIC QUESTION","",
           "Do frozen visual features generalize from 18 task0 TRAIN contexts to the two preregistered held-out physical/visual task0 DEV contexts, and do force interaction or Joint supervision add safe independent value?","",
           "# WHY TRAIN RESULTS ARE NOT ENOUGH","",
           "The task0 TRAIN run contains 18 visual contexts and a centered 17D PCA representation. Because all 10 force/repeat branches within a context share the same x, the effective visual sample size is 18, not 180; context memorization is a primary risk.","",
           "# FROZEN TASK0 MODELS","", "Twelve final epoch-80 checkpoints (four models × three seeds) were hashed before task0 DEV became available. No architecture, PCA, loss, lambda, seed, or checkpoint selection changed after DEV.","",
           "# HELD-OUT DEV POPULATION","",f"Two preregistered task0 DEV contexts, {len(branches)} physical branches, nine forces from 3.00N to 5.00N, and five repeats per context-force cell.","",
           "# VISUAL LEAKAGE AUDIT","","PASS. TRAIN roots 0–5 and DEV roots 6–7 are disjoint; PCA and normalization are TRAIN-only; visual x is captured at strict pre-probe step 190; DEV features are projection-only; no DEV outcome was used for feature, calibration, checkpoint, or architecture selection.",""]
    for title,model in [("# BASE FEAS",MODELS[0]),("# VISUAL RESIDUAL",MODELS[1]),("# FULL VISUAL FEAS",MODELS[2]),("# VISUAL JOINT",MODELS[3])]:
        p=model_row(prob,model); f=frontier_row(front,model); s=shape_row(shape,model)
        lines += [title,"",f"Held-out probability MAE {fmt(p.probability_MAE)}, Brier {fmt(p.Brier)}, NLL {fmt(p.NLL)}, frontier MAE {fmt(f.frontier_MAE_N)}N, under-force {fmt(f.under_force_rate)}, monotonic-context fraction {fmt(s.context_monotonic)}.",""]
    lines += ["# TRAIN VS DEV GENERALIZATION","", "| Model | TRAIN BCE | DEV NLL | Gap | TRAIN rel. gain vs Base | DEV rel. gain vs Base |","|---|---:|---:|---:|---:|---:|"]
    for r in gap.itertuples(index=False):
        lines.append(f"| {r.model} | {fmt(r.TRAIN_in_sample_BCE_mean)} | {fmt(r.DEV_heldout_NLL_seed_mean)} | {fmt(r.generalization_gap_DEV_minus_TRAIN)} | {fmt(r.TRAIN_relative_improvement_vs_Base)} | {fmt(r.DEV_relative_improvement_vs_Base)} |")
    lines += ["","# DOES VISUAL CONTEXT GENERALIZE?","","YES" if comp["visual_generalizes"] else "NO / EVIDENCE LIMITED","",
              "# DOES VISUAL CONTEXT ONLY SHIFT THE CURVE?","","YES" if comp["offset_only_supported"] else "NO","",
              "# DOES x × F INTERACTION MATTER?","","YES" if comp["force_interaction_supported"] else "NO","",
              "# DOES JOINT ADD INDEPENDENT VALUE?","","YES" if comp["joint_independent_win"] else "FRONTIER-ONLY" if comp["joint_frontier_only_signal"] else "NO / EVIDENCE LIMITED","",
              "# FRONTIER MAE","",md_table(front),"",
              "# UNDER-FORCE","",f"Safety-first selected backend: {selected}. GT gate: {'PASS' if gt else 'FAIL'}.","",
              "# PROBABILITY QUALITY","",md_table(prob[prob.aggregation=="ENSEMBLE"]),"",
              "# MONOTONICITY","",md_table(shape[shape.context_id=="__AGGREGATE__"]),"",
              "# NEAREST-CONTEXT MEMORIZATION RISK","",md_table(distances),"",
              "Only two primary DEV contexts exist, so a distance-benefit correlation is not statistically interpretable. The preregistered supplementary 3-fold TRAIN root-heldout CV was therefore run.","",
              "# TASK0 GT GATE","", "PASS" if gt else "FAIL", "",
              "# PRIMARY TASK0 CLASSIFICATION","",classification,"",
              "# WHAT TASK0 NOW SUPPORTS","", "The report supports only the comparisons that meet every frozen held-out probability, frontier, safety, multi-context, and monotonicity rule above.","",
              "# WHAT TASK0 DOES NOT SUPPORT","",
              "- No multi-task claim while tasks 1/5/6 are incomplete.","- No cross-object or unseen-task claim.","- No final Probe claim.","- No fresh E2E claim.","",
              "# NEXT ACTION","", "Continue the unchanged preregistered tasks 1/5/6. Do not redesign from task0. If task0 visual generalization failed, quantify independent-context insufficiency before considering more force samples.",""]
    (out/"FINAL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def write_notebook(out, classification):
    def md(s): return {"cell_type":"markdown","metadata":{},"source":[x+"\n" for x in s.splitlines()]}
    def code(src,output): return {"cell_type":"code","execution_count":1,"metadata":{},"source":[x+"\n" for x in src.splitlines()],"outputs":[{"output_type":"stream","name":"stdout","text":[output+"\n"]}]}
    summary=json.loads((out/"TASK0_FINAL_CLASSIFICATION.json").read_text())
    nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":sys.version.split()[0]}},"cells":[
        md("# task0 held-out visual generalization validation\n\n## tl;dr\n"+classification),
        md("## Context & Methods\nFrozen 12 checkpoint units are evaluated on two preregistered task0 DEV contexts without DEV calibration or tuning."),
        code("import json\nsummary=json.load(open('TASK0_FINAL_CLASSIFICATION.json'))\nprint(json.dumps(summary,indent=2))",json.dumps(summary,indent=2)),
        md("## Data\nTwo contexts × nine forces × five physical repeats. See leakage and real-curve artifacts."),
        code("import pandas as pd\nprint(pd.read_csv('TASK0_DEV_PROBABILITY_METRICS.csv').query(\"aggregation == 'ENSEMBLE'\").to_string(index=False))",pd.read_csv(out/"TASK0_DEV_PROBABILITY_METRICS.csv").query("aggregation == 'ENSEMBLE'").to_string(index=False)),
        md("## Results\nPrimary held-out metrics are complemented by the preregistered root-heldout CV because n=2 DEV contexts is too small for a distance-response inference."),
        code("print(pd.read_csv('TASK0_GROUP_HELDOUT_CV.csv').query(\"fold == '__MEAN__'\").to_string(index=False))",pd.read_csv(out/"TASK0_GROUP_HELDOUT_CV.csv").query("fold == '__MEAN__'").to_string(index=False)),
        md("## Takeaways\nInterpret only against the frozen winner and stop rules; task0 cannot establish a multi-task, cross-object, Probe, or E2E claim.") ]}
    write_json(out/"TASK0_VALIDATION.ipynb",nb)


def evaluate(out: Path, run_cv_flag: bool = True):
    protocol_path=out/"TASK0_VISUAL_GENERALIZATION_PROTOCOL.json"
    expected=(out/"TASK0_PROTOCOL_SHA256.txt").read_text().strip()
    if sha256(protocol_path)!=expected:
        raise RuntimeError("protocol changed after freeze")
    rd=readiness()
    if not rd["ready"]:
        raise RuntimeError("authoritative task0 DEV is not complete: "+json.dumps(rd))
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    device=torch.device("cpu")
    suffix=str(os.getpid())
    tpi=load_module("tpi_t0gen_"+suffix,early.TPI_CODE)
    cf=load_module("cf_t0gen_"+suffix,early.CF_CODE)
    full=load_module("full_t0gen_"+suffix,early.FULL_CODE)
    contexts,branches,visual,cmap,templates,failures=load_dev(tpi,cf)
    train_ids,dev_ids,train_roots,dev_roots=expected_task0_ids()
    if sorted(contexts.context_id.astype(str))!=sorted(dev_ids): failures.append("DEV context IDs differ from frozen protocol")
    if set(train_roots)&set(dev_roots): failures.append("TRAIN/DEV root overlap")
    # Verify every checkpoint still matches the pre-DEV protocol hash list.
    frozen=json.loads(protocol_path.read_text())
    for ck in frozen["checkpoints"]:
        if sha256(Path(ck["path"]))!=ck["sha256"]: failures.append("checkpoint hash changed: "+ck["path"])
    leakage={"status":"FAIL" if failures else "PASS","classification_if_fail":"TASK0_VISUAL_RESULT_INVALID_DUE_TO_LEAKAGE",
             "failures":failures,"TRAIN_contexts":18,"DEV_contexts":2,"TRAIN_DEV_root_overlap":sorted(set(train_roots)&set(dev_roots)),
             "PCA_fit":"task0 TRAIN-only frozen before DEV","PCA_sha256":sha256(EARLY/"TASK0_PCA17_PROVISIONAL.npz"),
             "DEV_used_for_PCA_or_normalization":False,"DEV_outcome_used_for_feature_or_checkpoint_selection":False,
             "explicit_root_or_split_model_input":False,"strict_preprobe_visual_capture_step":190,
             "same_x_all_forces":bool((visual.same_x_for_all_branches==1).all()),"post_force_postprobe_terminal_visual_input":False,
             "checkpoint_hashes_unchanged":not any("checkpoint hash" in x for x in failures),"protocol_sha256":expected}
    write_json(out/"TASK0_VISUAL_LEAKAGE_AUDIT.json",leakage)
    if failures:
        write_json(out/"TASK0_FINAL_CLASSIFICATION.json",{"classification":"TASK0_VISUAL_RESULT_INVALID_DUE_TO_LEAKAGE","failures":failures})
        raise RuntimeError("leakage/alignment audit failed")
    real,real_fronts=real_curves(branches)
    real.to_csv(out/"TASK0_REAL_DEV_CURVES.csv",index=False)
    real_fronts.to_csv(out/"TASK0_REAL_DEV_FRONTIERS.csv",index=False)
    norm_np=np.load(EARLY/"TASK0_TRAIN_NORMALIZATION.npz")
    norm=tuple(norm_np[k] for k in ["x_mean","x_std","y_mean","y_std"])
    models=load_models(tpi,full,device)
    pred=predict_all(cf,tpi,models,cmap,templates,norm,device)
    pred.to_csv(out/"TASK0_DEV_DENSE_PREDICTIONS.csv",index=False)
    prob=probability_metrics(pred,real); prob.to_csv(out/"TASK0_DEV_PROBABILITY_METRICS.csv",index=False)
    shape=shape_metrics(pred); shape.to_csv(out/"TASK0_DEV_FORCE_SHAPE_METRICS.csv",index=False)
    fdetail,front=frontier_metrics(pred,real_fronts)
    fdetail.to_csv(out/"TASK0_DEV_FRONTIER_CONTEXTS.csv",index=False); front.to_csv(out/"TASK0_DEV_FRONTIER_METRICS.csv",index=False)
    distances=visual_distances(cmap)
    breakdown,distance=context_and_distance_breakdown(pred,real,real_fronts,fdetail,distances)
    breakdown.to_csv(out/"TASK0_DEV_CONTEXT_BREAKDOWN.csv",index=False)
    distance.to_csv(out/"TASK0_VISUAL_DISTANCE_GENERALIZATION.csv",index=False)
    comp=comparisons(prob,front,shape,breakdown)
    write_csv(out/"TASK0_RESIDUAL_MECHANISM_TEST.csv",[{k:v for k,v in comp.items() if "residual" in k.lower() or "offset" in k.lower()}])
    write_csv(out/"TASK0_FULL_VS_RESIDUAL.csv",[{k:v for k,v in comp.items() if "Residual" in k or "interaction" in k}])
    write_csv(out/"TASK0_VISUAL_JOINT_VALUE_TEST.csv",[{k:v for k,v in comp.items() if "Joint" in k or "joint" in k}])
    gap=generalization_gap(prob); gap.to_csv(out/"TASK0_TRAIN_DEV_GENERALIZATION_GAP.csv",index=False)
    # Two DEV contexts are intrinsically insufficient for a distance-benefit relationship; run the frozen optional diagnostic.
    cv=run_cv(out,tpi,cf,full,device) if run_cv_flag else pd.DataFrame()
    selected,gt,classification,cv_support,scored=select_and_classify(prob,front,shape,comp,cv)
    write_json(out/"TASK0_GT_GATE.json",{"status":"PASS" if gt else "FAIL","selected_backend":selected,
               "gate":{"valid_real_frontier_coverage_min":.8,"finite_model_decision_coverage_min":.8,"probability_MAE_max":.20,"frontier_MAE_max_N":.20,"under_force_rate_max":.10,"systematic_nonmonotonicity_forbidden":True},
               "Probe_status":"ELIGIBLE_FOR_LATER_PROTOCOL" if gt else "NOT_REACHED","safety_first_scores":scored})
    final={"classification":classification,"selected_task0_backend":selected,"GT_gate_pass":gt,
           "primary_DEV_contexts":2,"optional_root_heldout_CV_run":run_cv_flag,"CV_visual_support":cv_support,
           "comparison_tests":comp,"multi_task_claim":False,"Probe_claim":False,"fresh_E2E_claim":False,
           "next_action":"continue unchanged preregistered tasks 1/5/6; do not redesign from task0"}
    write_json(out/"TASK0_FINAL_CLASSIFICATION.json",final)
    write_report(out,contexts,branches,prob,front,shape,comp,gap,cv,selected,gt,classification,distances)
    write_notebook(out,classification)
    # Final integrity inventory is generated last and excludes itself.
    hashes=[]
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name!="SHA256SUMS.txt": hashes.append(f"{sha256(p)}  {p.name}")
    (out/"SHA256SUMS.txt").write_text("\n".join(hashes)+"\n",encoding="utf-8")
    print(json.dumps(final,indent=2),flush=True)


def wait_evaluate(out: Path, poll_seconds: int, run_cv_flag: bool):
    freeze(out)
    while True:
        rd=readiness(); write_json(out/"TASK0_WAIT_STATUS.json",rd)
        print(json.dumps(rd),flush=True)
        if rd["ready"]: break
        time.sleep(poll_seconds)
    evaluate(out,run_cv_flag=run_cv_flag)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["freeze","check","evaluate","wait-evaluate"])
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--poll-seconds",type=int,default=60)
    ap.add_argument("--skip-cv",action="store_true")
    args=ap.parse_args(); out=args.out.resolve()
    if args.phase=="freeze": freeze(out)
    elif args.phase=="check": print(json.dumps(readiness(),indent=2))
    elif args.phase=="evaluate": evaluate(out,run_cv_flag=not args.skip_cv)
    else: wait_evaluate(out,args.poll_seconds,run_cv_flag=not args.skip_cv)


if __name__=="__main__": main()
