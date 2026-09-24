#!/usr/bin/env python3
"""Frozen task-specific DEV evaluation for tasks 1, 5, and 6.

The freeze phase records the unchanged task0-derived rules and all TRAIN-only
checkpoint hashes without reading DEV.  The evaluate phase accepts only the
authoritative preregistered DEV population and never calibrates or selects a
checkpoint on DEV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import per_task_visual_context_early as taskwise
import task0_visual_context_early as early
import task0_visual_generalization as gen


SOURCE = taskwise.SOURCE
CONTEXT_MANIFEST = taskwise.CONTEXT_MANIFEST
MODELS = gen.MODELS
SEEDS = gen.SEEDS
EXPECTED_DEV = {1: 180, 5: 90, 6: 25}
TASK_REAL_FORCES = {
    1: np.round(np.arange(4.0, 6.0001, 0.25), 2),
    5: np.round(np.arange(3.0, 5.0001, 0.25), 2),
    6: np.round(np.arange(3.0, 4.0001, 0.25), 2),
}
TASK_DENSE_FORCES = {
    task: np.round(np.arange(float(forces.min()), float(forces.max()) + .0001, 0.05), 2)
    for task, forces in TASK_REAL_FORCES.items()
}


def sha256(path: Path) -> str:
    return taskwise.sha256(path)


def write_json(path: Path, value: Any) -> None:
    early.write_json(path, value)


def markdown_table(frame: pd.DataFrame) -> str:
    """Serialize a DataFrame without pandas' optional tabulate dependency."""
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [("" if pd.isna(value) else str(value)).replace("|", "\\|").replace("\n", " ")
                  for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def expected_split(task: int):
    d = pd.read_csv(CONTEXT_MANIFEST)
    train = d[(d.task == task) & (d.split == "TRAIN")].sort_values("context_id")
    dev = d[(d.task == task) & (d.split == "DEV")].sort_values("context_id")
    return train, dev


def paths(task: int, frozen: Path):
    root = SOURCE / "collection_dev"
    return {
        "context": root / f"task{task}/task{task}/context.csv",
        "branch": root / f"task{task}/task{task}/branches.csv",
        "visual": root / "visual_alignment_worker.csv",
        "pca": frozen / f"TASK{task}_PCA17_TRAIN_ONLY.npz",
        "norm": frozen / f"TASK{task}_TRAIN_NORMALIZATION.npz",
        "train_visual": frozen / f"TASK{task}_FROZEN_VISUAL_ALIGNMENT.csv",
        "train_metric": frozen / f"TASK{task}_TRAIN_IN_SAMPLE_METRICS.csv",
        "cv": frozen / f"TASK{task}_GROUP_HELDOUT_CV.csv",
    }


def checkpoint(frozen: Path, task: int, model: str, seed: int) -> Path:
    return frozen / f"{model}_task{task}_seed{seed}.pt"


def freeze(task: int, frozen: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / f"TASK{task}_VISUAL_GENERALIZATION_PROTOCOL.json"
    hash_path = out / f"TASK{task}_PROTOCOL_SHA256.txt"
    if protocol_path.exists():
        if not hash_path.exists() or hash_path.read_text().strip() != sha256(protocol_path):
            raise RuntimeError("existing frozen protocol hash mismatch")
        print(json.dumps({"status": "ALREADY_FROZEN", "task": task, "sha256": sha256(protocol_path)}))
        return
    train, dev = expected_split(task)
    if len(train) != 18 or len(dev) not in (1, 2, 4) or set(train.root_id) & set(dev.root_id):
        raise RuntimeError(f"task{task}: authoritative split mismatch")
    ps = paths(task, frozen)
    for key in ("pca", "norm", "train_visual", "train_metric"):
        if not ps[key].exists():
            raise RuntimeError(f"task{task}: missing frozen TRAIN artifact {ps[key]}")
    checkpoints = []
    for model in MODELS:
        for seed in SEEDS:
            p = checkpoint(frozen, task, model, seed)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            if ck.get("variant") != model or int(ck.get("seed", -1)) != seed or int(ck.get("task", -1)) != task:
                raise RuntimeError(f"checkpoint identity mismatch: {p}")
            if ck.get("DEV_used") is not False or int(ck.get("epochs", -1)) != 80:
                raise RuntimeError(f"checkpoint is not final TRAIN-only epoch: {p}")
            checkpoints.append({"model": model, "seed": seed, "path": str(p), "sha256": sha256(p)})
    rules = {
        "visual_generalizes": {"probability_MAE_relative_improvement_min": .20,
            "frontier_MAE_improvement_min_N": .05, "under_force_nonworse": True,
            "multiple_independent_DEV_contexts": True, "monotonicity_degradation_max": .10},
        "offset_only": {"residual_fraction_of_full_MAE_gain_min": .70, "safety_nonworse": True},
        "force_interaction": {"full_vs_residual_probability_MAE_relative_improvement_min": .10,
            "OR_frontier_MAE_improvement_min_N": .05, "safety_nonworse": True, "multiple_contexts": True},
        "joint_win": {"probability_MAE_relative_improvement_min": .10,
            "frontier_MAE_improvement_min_N": .05, "under_force_nonworse": True,
            "Brier_NLL_not_worse": True, "systematic_nonmonotonicity_forbidden": True,
            "multiple_contexts": True},
        "GT_gate": {"valid_real_frontier_coverage_min": .80, "finite_model_decision_coverage_min": .80,
            "probability_MAE_max": .20, "frontier_MAE_max_N": .20, "under_force_rate_max": .10,
            "systematic_nonmonotonicity_forbidden": True},
    }
    protocol = {
        "status": f"FROZEN_BEFORE_TASK{task}_DEV_OPENED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "task": task,
        "object": taskwise.TASK_NAMES[task],
        "goal": "test whether task0 visual memorization/generalization and Joint behavior recur on this task",
        "TRAIN_context_ids": train.context_id.astype(str).tolist(),
        "DEV_context_ids": dev.context_id.astype(str).tolist(),
        "TRAIN_root_ids": sorted(train.root_id.astype(str).unique()),
        "DEV_root_ids": sorted(dev.root_id.astype(str).unique()),
        "expected_DEV_branches": EXPECTED_DEV[task], "real_force_grid_N": TASK_REAL_FORCES[task].tolist(),
        "dense_query_grid_N": TASK_DENSE_FORCES[task].tolist(), "rho": gen.RHO,
        "checkpoints": checkpoints, "winner_rules": rules,
        "PCA": {"path": str(ps["pca"]), "sha256": sha256(ps["pca"]), "fit_split": f"task{task} TRAIN only", "DEV_fit_forbidden": True},
        "normalization": {"path": str(ps["norm"]), "sha256": sha256(ps["norm"]), "fit_split": f"task{task} TRAIN only"},
        "task1_label_sensitivity": {
            "required": task == 1,
            "reason": "140/180 TRAIN labels use the preregistered terminal-height fallback; available direct task1 labels agree 40/40, while task5 indicates a 2/180 rare transient-contact miss rate",
            "rule": "report task1 classification with this material caveat and mark any TRAIN/CV conclusion label-sensitive when its model gap is not larger than the empirical 2/180 reference rate",
        },
        "single_context_rule": "task6 cannot establish multi-context generalization from its one DEV context; report metrics and TRAIN root-heldout CV only" if task == 6 else None,
        "no_DEV_calibration": True, "no_DEV_checkpoint_selection": True,
        "no_post_freeze_model_PCA_feature_loss_or_threshold_changes": True,
        "DEV_outcomes_read_at_freeze": False, "TEST_used": False,
    }
    write_json(protocol_path, protocol)
    hash_path.write_text(sha256(protocol_path) + "\n", encoding="utf-8")
    write_json(out / f"TASK{task}_ALL_MODELS_FROZEN.json", {
        "status": "PASS_12_FINAL_TASK_SPECIFIC_UNITS_FROZEN_BEFORE_DEV",
        "task": task, "checkpoint_count": 12, "all_final_epoch_80": True,
        "all_DEV_used_false": True, "protocol_sha256": sha256(protocol_path),
    })
    print(json.dumps({"status": protocol["status"], "task": task, "DEV_contexts": len(dev), "checkpoints": 12}, indent=2))


def readiness(task: int, frozen: Path) -> dict[str, Any]:
    ps = paths(task, frozen)
    expected_ids = set(expected_split(task)[1].context_id.astype(str))
    result = {"ready": False, "task": task, "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    if not all(ps[k].exists() for k in ("context", "branch", "visual")):
        result["reason"] = "DEV files not all present"
        return result
    try:
        c, b, v = pd.read_csv(ps["context"]), pd.read_csv(ps["branch"]), pd.read_csv(ps["visual"])
    except Exception as exc:
        result["reason"] = f"DEV files unreadable/still being written: {exc}"
        return result
    c = c[c.context_id.astype(str).isin(expected_ids)]
    b = b[b.context_id.astype(str).isin(expected_ids)]
    v = v[v.context_id.astype(str).isin(expected_ids)]
    counts = b.groupby(["context_id", "requested_force_N"]).size() if len(b) else pd.Series(dtype=int)
    expected_force_cells = len(expected_ids) * len(TASK_REAL_FORCES[task])
    cells_ok = len(counts) == expected_force_cells and bool((counts == 5).all())
    result.update({"contexts": len(c), "branches": len(b), "visual_contexts": len(v),
                   "unique_contexts": int(c.context_id.nunique()) if len(c) else 0,
                   "unique_branches": int(b.branch_id.nunique()) if len(b) else 0,
                   "complete_force_repeat_cells": cells_ok})
    parity = "state_parity" in b and len(b) == EXPECTED_DEV[task] and int(b.state_parity.sum()) == EXPECTED_DEV[task]
    restore = "restore_exact" in v and len(v) == len(expected_ids) and int(v.restore_exact.sum()) == len(expected_ids)
    result["ready"] = bool(len(c) == len(expected_ids) and len(b) == EXPECTED_DEV[task] and
                           len(v) == len(expected_ids) and cells_ok and parity and restore)
    result["reason"] = "complete" if result["ready"] else "task-specific DEV incomplete"
    return result


def pca_transform(task: int, frozen: Path, feature_paths: list[Path]) -> np.ndarray:
    p = np.load(paths(task, frozen)["pca"])
    raw = np.stack([np.load(x, allow_pickle=False).astype(np.float32) for x in feature_paths])
    z = (raw - p["raw_mean"]) @ p["components"].T
    return ((z - p["projected_mean"]) / p["projected_std"]).astype(np.float32)


def load_dev(task: int, frozen: Path, tpi, cf):
    ps = paths(task, frozen)
    expected_ids = set(expected_split(task)[1].context_id.astype(str))
    contexts = pd.read_csv(ps["context"]); contexts = contexts[contexts.context_id.astype(str).isin(expected_ids)].sort_values("context_id")
    branches = pd.read_csv(ps["branch"]); branches = branches[branches.context_id.astype(str).isin(expected_ids)]
    visual_all = pd.read_csv(ps["visual"]); visual = visual_all[visual_all.context_id.astype(str).isin(expected_ids)].copy()
    vmap = visual.set_index("context_id").to_dict("index")
    z = pca_transform(task, frozen, [Path(str(vmap[c]["visual_feature_path"])) for c in contexts.context_id.astype(str)])
    cmap, templates, failures = {}, {}, []
    corrected = {"left_normal_force_N", "right_normal_force_N", "left_tangential_force_N",
                 "right_tangential_force_N", "object_vx_mps", "object_vy_mps", "object_vz_mps"}
    for row, zv in zip(contexts.itertuples(index=False), z):
        cid = str(row.context_id); vr = vmap[cid]
        raw = np.load(vr["visual_feature_path"], allow_pickle=False).astype(np.float32)
        if hashlib.sha256(raw.tobytes()).hexdigest() != str(vr["visual_feature_sha256"]): failures.append(f"{cid}: visual hash mismatch")
        if any(str(vr[k]) != str(vr["snapshot_state_hash"]) for k in ("restored_state_hash", "second_restore_hash")): failures.append(f"{cid}: restore mismatch")
        if str(vr["snapshot_state_hash"]) != str(row.post_probe_state_hash): failures.append(f"{cid}: context/visual state mismatch")
        if int(vr["capture_step"]) != 190 or "before any probe action" not in str(vr["timestamp_semantics"]): failures.append(f"{cid}: capture timing invalid")
        if int(vr["same_x_for_all_branches"]) != 1: failures.append(f"{cid}: same-x invariant absent")
        state0, mask0, _ = early.strict_preprobe_state(Path(str(row.probe_telemetry_path)))
        q = branches[branches.context_id.astype(str) == cid].sort_values(["requested_force_N", "branch_label"])
        expected_per_context = len(TASK_REAL_FORCES[task]) * 5
        if len(q) != expected_per_context or q.requested_force_N.nunique() != len(TASK_REAL_FORCES[task]): failures.append(f"{cid}: incomplete branches"); continue
        d = pd.read_csv(Path(str(q.iloc[0].telemetry_path)))
        if not corrected <= set(d) or not np.isfinite(d[list(corrected)].to_numpy(float)).all(): failures.append(f"{cid}: corrected telemetry invalid")
        state, mask = tpi.state_from(d); state, mask = state.copy(), mask.copy(); state[0], mask[0] = state0, mask0
        mu = float(row.hidden_friction_analysis_only); force = float(TASK_REAL_FORCES[task].min())
        nominal = tpi.nominal_from(d, 0, force, mu, state, mask)
        tr = tpi.Trace(f"template:{cid}", cid, str(row.root_id), task, "DEV", force, mu, 0,
            "dense_query", Path(str(q.iloc[0].telemetry_path)), state, mask, nominal,
            d.phase.astype(str).tolist(), 1.0, f"PROSPECTIVE_TASK{task}_DEV_TEMPLATE")
        cmap[cid] = {"visual": zv, "visual_raw": raw, "root_id": str(row.root_id),
                     "friction": mu, "friction_band": str(row.friction_band)}
        templates[cid] = tr
    return contexts, branches, visual, cmap, templates, failures


def visual_distances(task: int, frozen: Path, cmap) -> pd.DataFrame:
    train = pd.read_csv(paths(task, frozen)["train_visual"]).sort_values("context_id")
    train_z = pca_transform(task, frozen, [Path(x) for x in train.visual_feature_path.astype(str)])
    rows = []
    for cid in sorted(cmap):
        dist = np.linalg.norm(train_z - cmap[cid]["visual"][None], axis=1); i = int(np.argmin(dist))
        rows.append({"context_id": cid, "nearest_TRAIN_context_id": str(train.iloc[i].context_id),
            "nearest_TRAIN_root_id": str(train.iloc[i].root_id), "nearest_TRAIN_distance": float(dist[i]),
            "mean_TRAIN_distance": float(np.mean(dist)), "max_TRAIN_distance": float(np.max(dist))})
    return pd.DataFrame(rows)


def generalization_gap(task: int, frozen: Path, prob: pd.DataFrame) -> pd.DataFrame:
    train = pd.read_csv(paths(task, frozen)["train_metric"]); rows = []
    base_train = float(train[train.variant == MODELS[0]].bce_nll.mean())
    base_dev = float(gen.model_row(prob, MODELS[0], "SEED_MEAN").NLL)
    for model in MODELS:
        tr = train[train.variant == model]; dv = gen.model_row(prob, model, "SEED_MEAN")
        tl, dl = float(tr.bce_nll.mean()), float(dv.NLL)
        rows.append({"model": model, "TRAIN_in_sample_BCE_mean": tl, "TRAIN_in_sample_BCE_std": float(tr.bce_nll.std(ddof=1)),
            "DEV_heldout_NLL_seed_mean": dl, "DEV_heldout_NLL_seed_std": float(gen.model_row(prob, model, "SEED_STD").NLL),
            "generalization_gap_DEV_minus_TRAIN": dl-tl, "DEV_to_TRAIN_loss_ratio": dl/tl,
            "TRAIN_relative_improvement_vs_Base": (base_train-tl)/base_train,
            "DEV_relative_improvement_vs_Base": (base_dev-dl)/base_dev if base_dev else math.nan})
    return pd.DataFrame(rows)


def select(task: int, frozen: Path, prob, front, shape, comp):
    complexity = {m:i for i,m in enumerate(MODELS)}
    def safe(v, fallback):
        v = float(v); return v if math.isfinite(v) else fallback
    scores = []
    for m in MODELS:
        p, f, s = gen.model_row(prob,m), gen.frontier_row(front,m), gen.shape_row(shape,m)
        scores.append((m, (safe(f.under_force_rate, math.inf), -safe(f.finite_decision_coverage,-math.inf),
            safe(f.frontier_MAE_N,math.inf), safe(p.probability_MAE,math.inf), safe(p.Brier,math.inf),
            safe(f.mean_excess_force_N,math.inf), int(s.safe_to_unsafe_reversals), complexity[m])))
    selected = min(scores, key=lambda x:x[1])[0]
    p, f, s = gen.model_row(prob,selected), gen.frontier_row(front,selected), gen.shape_row(shape,selected)
    gt = bool(float(f.valid_real_frontier_coverage)>=.8 and float(f.finite_decision_coverage)>=.8 and
              float(p.probability_MAE)<=.20 and float(f.frontier_MAE_N)<=.20 and float(f.under_force_rate)<=.10 and
              int(s.safe_to_unsafe_reversals)==0 and float(s.context_monotonic)>=.90)
    cv = pd.read_csv(paths(task, frozen)["cv"]); cvm = cv[cv.fold.astype(str)=="__MEAN__"].set_index("model")
    cv_support = bool(float(cvm.loc[MODELS[2],"BCE"]) < float(cvm.loc[MODELS[0],"BCE"]))
    if task == 6:
        classification = "TASK6_SINGLE_DEV_CONTEXT_EVIDENCE_LIMITED"
    elif comp["joint_frontier_only_signal"]:
        classification = f"TASK{task}_JOINT_FRONTIER_SIGNAL_WITHOUT_RELIABLE_CONTROL"
    elif comp["visual_generalizes"] and comp["joint_independent_win"]:
        classification = f"TASK{task}_VISUAL_CONTEXT_GENERALIZES_AND_JOINT_WINS"
    elif comp["visual_generalizes"] and comp["offset_only_supported"]:
        classification = f"TASK{task}_VISUAL_CONTEXT_GENERALIZES_WITH_OFFSET_ONLY"
    elif comp["visual_generalizes"] and comp["force_interaction_supported"]:
        classification = f"TASK{task}_VISUAL_CONTEXT_GENERALIZES_WITH_FORCE_INTERACTION"
    elif comp["visual_generalizes"]:
        classification = f"TASK{task}_VISUAL_CONTEXT_GENERALIZES_BUT_JOINT_NOT_NEEDED"
    else:
        train = pd.read_csv(paths(task, frozen)["train_metric"])
        tb = float(train[train.variant==MODELS[0]].bce_nll.mean()); tf = float(train[train.variant==MODELS[2]].bce_nll.mean())
        strong_train = (tb-tf)/tb >= .20
        if strong_train and not cv_support: classification = f"TASK{task}_VISUAL_CONTEXT_TRAIN_MEMORIZATION"
        elif not gt: classification = f"TASK{task}_GT_CONTINUOUS_GATE_FAILS"
        else: classification = f"TASK{task}_VISUAL_SIGNAL_POSITIVE_BUT_DEV_EVIDENCE_LIMITED"
    empirical_label_rate = 2/180
    cv_gap = abs(float(cvm.loc[MODELS[2],"BCE"])-float(cvm.loc[MODELS[0],"BCE"]))
    sensitivity = {"required": task==1, "empirical_reference_rate": empirical_label_rate,
        "CV_Base_Full_absolute_BCE_gap": cv_gap,
        "status": "LABEL_SENSITIVE" if task==1 and cv_gap <= empirical_label_rate else "EMPIRICAL_RATE_SCREEN_PASSES_NOT_EXACTLY_RESOLVED" if task==1 else "NOT_APPLICABLE",
        "caveat": "This bound cannot reconstruct unlogged transient basket contacts; it is a declared uncertainty check, not an exact relabeling." if task==1 else None}
    return selected, gt, classification, cv_support, scores, sensitivity


def evaluate(task: int, frozen: Path, out: Path) -> None:
    protocol = out / f"TASK{task}_VISUAL_GENERALIZATION_PROTOCOL.json"; ph = out / f"TASK{task}_PROTOCOL_SHA256.txt"
    if not protocol.exists() or not ph.exists() or sha256(protocol) != ph.read_text().strip(): raise RuntimeError("protocol not frozen or hash mismatch")
    rd = readiness(task, frozen)
    if not rd["ready"]: raise RuntimeError("authoritative DEV incomplete: " + json.dumps(rd))
    torch.set_num_threads(1); torch.set_num_interop_threads(1); device = torch.device("cpu")
    suffix = f"{task}_{os.getpid()}"; tpi = gen.load_module("tpi_gen_"+suffix, early.TPI_CODE); cf = gen.load_module("cf_gen_"+suffix, early.CF_CODE); full = gen.load_module("full_gen_"+suffix, early.FULL_CODE)
    contexts, branches, visual, cmap, templates, failures = load_dev(task, frozen, tpi, cf)
    train, dev = expected_split(task)
    if sorted(contexts.context_id.astype(str)) != sorted(dev.context_id.astype(str)): failures.append("DEV IDs differ from frozen manifest")
    if set(train.root_id) & set(dev.root_id): failures.append("TRAIN/DEV root overlap")
    frozen_protocol = json.loads(protocol.read_text())
    for ck in frozen_protocol["checkpoints"]:
        if sha256(Path(ck["path"])) != ck["sha256"]: failures.append("checkpoint hash changed: "+ck["path"])
    leakage = {"status":"FAIL" if failures else "PASS", "task":task, "failures":failures,
        "TRAIN_DEV_root_overlap": sorted(set(train.root_id)&set(dev.root_id)), "PCA_fit":f"task{task} TRAIN only",
        "DEV_used_for_PCA_normalization_calibration_or_selection":False, "strict_preprobe_capture_step":190,
        "same_x_all_forces":bool((visual.same_x_for_all_branches==1).all()), "protocol_sha256":ph.read_text().strip()}
    write_json(out/f"TASK{task}_VISUAL_LEAKAGE_AUDIT.json", leakage)
    if failures:
        write_json(out/f"TASK{task}_FINAL_CLASSIFICATION.json", {"classification":f"TASK{task}_RESULT_INVALID_DUE_TO_LEAKAGE","failures":failures})
        raise RuntimeError("DEV leakage/alignment audit failed")
    gen.EARLY = frozen
    gen.FORCES_REAL = TASK_REAL_FORCES[task]
    gen.FORCES_DENSE = TASK_DENSE_FORCES[task]
    gen.checkpoint_path = lambda model, seed: checkpoint(frozen, task, model, seed)
    real, real_fronts = gen.real_curves(branches); real.to_csv(out/f"TASK{task}_REAL_DEV_CURVES.csv",index=False); real_fronts.to_csv(out/f"TASK{task}_REAL_DEV_FRONTIERS.csv",index=False)
    n = np.load(paths(task,frozen)["norm"]); norm = tuple(n[k] for k in ("x_mean","x_std","y_mean","y_std"))
    models = gen.load_models(tpi,full,device); pred = gen.predict_all(cf,tpi,models,cmap,templates,norm,device); pred.to_csv(out/f"TASK{task}_DEV_DENSE_PREDICTIONS.csv",index=False)
    prob = gen.probability_metrics(pred,real); prob.to_csv(out/f"TASK{task}_DEV_PROBABILITY_METRICS.csv",index=False)
    shape = gen.shape_metrics(pred); shape.to_csv(out/f"TASK{task}_DEV_FORCE_SHAPE_METRICS.csv",index=False)
    fdetail, front = gen.frontier_metrics(pred,real_fronts); fdetail.to_csv(out/f"TASK{task}_DEV_FRONTIER_CONTEXTS.csv",index=False); front.to_csv(out/f"TASK{task}_DEV_FRONTIER_METRICS.csv",index=False)
    distances = visual_distances(task,frozen,cmap); breakdown,distance = gen.context_and_distance_breakdown(pred,real,real_fronts,fdetail,distances)
    breakdown.to_csv(out/f"TASK{task}_DEV_CONTEXT_BREAKDOWN.csv",index=False); distance.to_csv(out/f"TASK{task}_VISUAL_DISTANCE_GENERALIZATION.csv",index=False)
    comp = gen.comparisons(prob,front,shape,breakdown); gap = generalization_gap(task,frozen,prob); gap.to_csv(out/f"TASK{task}_TRAIN_DEV_GENERALIZATION_GAP.csv",index=False)
    selected, gt, classification, cv_support, scores, sensitivity = select(task,frozen,prob,front,shape,comp)
    final = {"classification":classification, "task":task, "object":taskwise.TASK_NAMES[task], "selected_backend":selected,
        "GT_gate_pass":gt, "DEV_contexts":len(contexts), "DEV_branches":len(branches), "CV_visual_support":cv_support,
        "comparison_tests":comp, "task1_label_sensitivity":sensitivity, "multi_context_claim_permitted":task!=6,
        "multi_task_claim":False, "Probe_claim":False, "fresh_E2E_claim":False}
    write_json(out/f"TASK{task}_GT_GATE.json", {"status":"PASS" if gt else "FAIL","selected_backend":selected,"scores":scores})
    write_json(out/f"TASK{task}_LABEL_SENSITIVITY.json", sensitivity); write_json(out/f"TASK{task}_FINAL_CLASSIFICATION.json", final)
    summary = prob[prob.aggregation=="ENSEMBLE"][["model","probability_MAE","Brier","NLL"]].merge(front[["model","frontier_MAE_N","under_force_rate"]],on="model")
    lines = [f"# Task {task} held-out visual-context result", "", classification, "", f"Object: {taskwise.TASK_NAMES[task]}. DEV: {len(contexts)} contexts / {len(branches)} branches.", "", "## Frozen-model metrics", "", markdown_table(summary), "", "## Interpretation", "", json.dumps(comp,indent=2), "", "## Label/data caveat", "", json.dumps(sensitivity,indent=2)]
    (out/f"TASK{task}_FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":[
        {"cell_type":"markdown","metadata":{},"source":[f"# task{task} held-out validation\n",classification+"\n"]},
        {"cell_type":"code","execution_count":1,"metadata":{},"source":[f"import json\nprint(json.load(open('TASK{task}_FINAL_CLASSIFICATION.json')))\n"],"outputs":[{"output_type":"stream","name":"stdout","text":[json.dumps(final,indent=2)+"\n"]}]}
    ]}; write_json(out/f"TASK{task}_VALIDATION.ipynb",nb)
    hashes=[f"{sha256(p)}  {p.name}" for p in sorted(out.iterdir()) if p.is_file() and p.name!="SHA256SUMS.txt"]
    (out/"SHA256SUMS.txt").write_text("\n".join(hashes)+"\n",encoding="utf-8")
    print(json.dumps(final,indent=2),flush=True)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=("freeze","check","evaluate")); ap.add_argument("--task",type=int,required=True,choices=taskwise.SUPPORTED_TASKS); ap.add_argument("--frozen",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); frozen=a.frozen.resolve(); out=a.out.resolve()
    if a.phase=="freeze": freeze(a.task,frozen,out)
    elif a.phase=="check": print(json.dumps(readiness(a.task,frozen),indent=2))
    else: evaluate(a.task,frozen,out)


if __name__ == "__main__": main()
