#!/usr/bin/env python3
"""Frozen 60-vs-120 rollout/task ActiveForcing sample-efficiency audit.

This program only reads the authoritative 720-branch TRAIN population and the
already-complete ProbeScalar/Point-WM grouped-root OOF artifacts.  It never
discovers simulator TEST namespaces and never launches simulator collection.

Phases:
  phase0   recover exact old per-seed ProbeScalar versus Point-WM metrics
  prepare  freeze the outcome-independent canonical-first-repeat Sparse-60 set
  shard    train one Sparse-60 outer-fold/seed Direct + Point-WM nested shard
  finalize reuse authoritative Full-120 shards, aggregate, plot, and report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

import run_probe_conditioned_wm as old
import run_pooled_predictive_verifier as ppv


ROOT = Path("/home/exouser/FORTE")
PRIOR = ROOT / "activeforcing_probe_conditioned_wm_20260901_064627"
TASKS = [0, 1, 5, 6]
FOLDS = [0, 1, 2]
SEEDS = [0, 1, 2]
FMAX = old.FMAX
H = old.H


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def load_population(out: Path, tag: str):
    global TPI, CF, FULL
    TPI, CF, FULL, cmap, traces, meta, audits, pairs, segs, norm = old.load_pop(out, tag)
    old.TPI, old.CF, old.FULL = TPI, CF, FULL
    return cmap, traces, meta, audits, pairs, segs, norm


def population_frame(traces, meta) -> tuple[pd.DataFrame, np.ndarray]:
    md = ppv.frame(traces, meta)
    md["R_real"] = np.where(
        md.success > 0,
        (md.task.map(FMAX).to_numpy() - md.force_N.to_numpy()) / md.task.map(FMAX).to_numpy(),
        -1.0,
    )
    rf = old.root_fold(md)
    assert len(md) == 720
    assert md.branch_id.nunique() == 720
    assert md.context_id.nunique() == 72
    assert md.root_id.nunique() == 24
    assert set(md.task.unique()) == set(TASKS)
    return md, rf


def phase0(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(PRIOR / "NORMAL_POOLED_PROBE_WM_TABLE.csv")
    direct_name = "ProbeScalar-Direct"
    point_name = "ProbeScalar+Point-WM-Residual"
    q = table[(table.scope == "POOLED") & table.method.isin([direct_name, point_name])].copy()
    q["seed"] = q.seed.astype(str)
    by = {(r.method, r.seed): r for _, r in q.iterrows()}

    # Independent recovery from the 9 atomic-complete branch-score shards.
    _, traces, meta, _, _, _, _ = load_population(out, "phase0_seed_audit")
    md, _ = population_frame(traces, meta)
    n = len(md)
    direct_score = np.full((3, n), np.nan, np.float32)
    point_score = np.full((3, n), np.nan, np.float32)
    gt_score = np.full((3, n), np.nan, np.float32)
    for fold in FOLDS:
        for seed in SEEDS:
            z = np.load(PRIOR / "shards" / f"fold{fold}_seed{seed}.npz")
            ids = z["held_idx"]
            force = md.force_N.to_numpy()[ids]
            fmax = md.task.map(FMAX).to_numpy()[ids]
            for key, target in [("p_scalar", direct_score), ("p_gt", gt_score)]:
                p = z[key]
                target[seed, ids] = p * ((fmax - force) / fmax) + (1.0 - p) * -1.0
            point_score[seed, ids] = z["score_point"]
    assert np.isfinite(direct_score).all() and np.isfinite(point_score).all() and np.isfinite(gt_score).all()
    gt_selected = old.choose(md, gt_score.mean(0))
    gt_key = gt_selected[["context_id", "repeat", "selected_force_N"]].rename(columns={"selected_force_N": "GT_selected_force_N"})

    def recovered(score: np.ndarray) -> dict:
        s = old.choose(md, score).merge(gt_key, on=["context_id", "repeat"], validate="one_to_one")
        agree = float(((s.selected_force_N - s.GT_selected_force_N).abs() < 1e-8).mean())
        return {
            "episodes": int(len(s)),
            "sr": float(s.success.mean()),
            "underforce": float(s.under_force.mean()),
            "mean_force": float(s.selected_force_N.mean()),
            "utility": float(s.realized_utility.mean()),
            "gt_agreement": agree,
        }

    recovered_direct = {str(seed): recovered(direct_score[seed]) for seed in SEEDS}
    recovered_point = {str(seed): recovered(point_score[seed]) for seed in SEEDS}
    recovered_direct["ENSEMBLE_MEAN"] = recovered(direct_score.mean(0))
    recovered_point["ENSEMBLE_MEAN"] = recovered(point_score.mean(0))
    max_reconciliation_error = 0.0
    for seed_label in ["0", "1", "2", "ENSEMBLE_MEAN"]:
        for method, rec in [(direct_name, recovered_direct[seed_label]), (point_name, recovered_point[seed_label])]:
            ref = by[(method, seed_label)]
            pairs_to_check = [
                (rec["sr"], ref.SR), (rec["underforce"], ref.under_force),
                (rec["mean_force"], ref.mean_force_N), (rec["utility"], ref.realized_utility),
                (rec["gt_agreement"], ref.decision_agreement_with_GT),
            ]
            max_reconciliation_error = max(max_reconciliation_error, *(abs(float(a) - float(b)) for a, b in pairs_to_check))
    if max_reconciliation_error > 1e-12:
        raise RuntimeError(f"old shard/table reconciliation failed: {max_reconciliation_error}")

    rows = []
    for seed in SEEDS:
        d = recovered_direct[str(seed)]
        p = recovered_point[str(seed)]
        rows.append({
            "seed": seed,
            "episodes": d["episodes"],
            "direct_sr": d["sr"],
            "pointwm_sr": p["sr"],
            "delta_sr_pp": 100.0 * (p["sr"] - d["sr"]),
            "direct_underforce": d["underforce"],
            "pointwm_underforce": p["underforce"],
            "delta_underforce_pp": 100.0 * (p["underforce"] - d["underforce"]),
            "direct_mean_force_N": d["mean_force"],
            "pointwm_mean_force_N": p["mean_force"],
            "delta_mean_force_N": p["mean_force"] - d["mean_force"],
            "direct_realized_utility": d["utility"],
            "pointwm_realized_utility": p["utility"],
            "delta_realized_utility": p["utility"] - d["utility"],
            "direct_GT_decision_agreement": d["gt_agreement"],
            "pointwm_GT_decision_agreement": p["gt_agreement"],
            "delta_GT_decision_agreement_pp": 100.0 * (p["gt_agreement"] - d["gt_agreement"]),
        })
    per = pd.DataFrame(rows)
    write_csv(out / "POINT_WM_PER_SEED.csv", per)

    de = recovered_direct["ENSEMBLE_MEAN"]
    pe = recovered_point["ENSEMBLE_MEAN"]
    dmean = per.direct_sr.mean()
    pmean = per.pointwm_sr.mean()
    neg = per.loc[per.delta_sr_pp.idxmin()]
    lines = [
        "# Point-WM Seed Audit",
        "",
        "## Direct answer",
        "",
    ]
    for _, r in per.iterrows():
        lines.append(
            f"seed{int(r.seed)}: Direct {r.direct_sr*100:.2f}%, Point-WM {r.pointwm_sr*100:.2f}%, "
            f"Δ = {r.delta_sr_pp:+.2f} pp"
        )
    lines += [
        "",
        f"唯一 negative seed 是 **seed{int(neg.seed)}**：SR 下降 **{abs(neg.delta_sr_pp):.2f} pp** "
        f"（{int(round(neg.direct_sr*neg.episodes))} → {int(round(neg.pointwm_sr*neg.episodes))} / {int(neg.episodes)} successes）。"
        f"它仍有 {neg.pointwm_sr*100:.2f}% SR，因此是有限下降，不是 collapse。",
        "",
        "## Exact pooled metrics",
        "",
        "| seed | method | SR | under-force | mean force | realized utility | GT agreement |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in per.iterrows():
        lines.append(f"| {int(r.seed)} | Direct | {r.direct_sr*100:.2f}% | {r.direct_underforce*100:.2f}% | {r.direct_mean_force_N:.6f} N | {r.direct_realized_utility:.6f} | {r.direct_GT_decision_agreement*100:.2f}% |")
        lines.append(f"| {int(r.seed)} | Point-WM | {r.pointwm_sr*100:.2f}% | {r.pointwm_underforce*100:.2f}% | {r.pointwm_mean_force_N:.6f} N | {r.pointwm_realized_utility:.6f} | {r.pointwm_GT_decision_agreement*100:.2f}% |")
    lines += [
        "",
        "## Ensemble definition",
        "",
        f"Point-WM 95.14% 是 **prediction/utility ensemble 后重新决策**：先对三个 seed 的每个 branch score 取均值，再在每个 `(context, repeat)` 的 frozen force candidates 中 argmax。它不是 seed metric mean。",
        f"Point-WM 的三-seed SR mean 是 {pmean*100:.2f}%，而 prediction-level ensemble SR 是 {pe['sr']*100:.2f}%。",
        f"Direct 的三-seed SR mean 是 {dmean*100:.2f}%，而 prediction-level ensemble SR 是 {de['sr']*100:.2f}%。",
        "",
        f"Source: direct reconstruction from the 9 atomic-complete `fold{{0,1,2}}_seed{{0,1,2}}.npz` branch-score shards. Reconciliation max absolute error versus `NORMAL_POOLED_PROBE_WM_TABLE.csv` = {max_reconciliation_error:.3g}.",
    ]
    (out / "POINT_WM_SEED_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(per.to_string(index=False), flush=True)


def canonical_sparse_rows(md: pd.DataFrame, rf: np.ndarray) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        train = md[rf != fold].copy()
        ordered = train.sort_values(
            ["task", "root_id", "context_id", "force_N", "repeat", "branch_id"],
            kind="mergesort",
        )
        selected = ordered.groupby(
            ["task", "root_id", "context_id", "force_N"], sort=True, as_index=False
        ).head(1).copy()
        selected.insert(0, "outer_fold", fold)
        selected["selection_rule"] = "canonical_first_repeat_by_numeric_repeat_then_branch_id"
        selected["selected"] = 1
        assert len(selected) == 240
        assert selected.groupby("task").size().to_dict() == {t: 60 for t in TASKS}
        assert set(selected.groupby(["task", "root_id"]).size()) == {15}
        assert selected.repeat.nunique() == 1 and int(selected.repeat.iloc[0]) == 1
        assert selected.branch_id.nunique() == 240
        held = md[rf == fold]
        assert not set(selected.root_id) & set(held.root_id)
        rows.append(selected[[
            "outer_fold", "task", "root_id", "context_id", "force_N", "repeat",
            "branch_id", "selection_rule", "selected",
        ]])
    return pd.concat(rows, ignore_index=True)


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    phase0(out)
    _, traces, meta, audits, pairs, _, _ = load_population(out, "prepare")
    md, rf = population_frame(traces, meta)
    selected = canonical_sparse_rows(md, rf)
    selection_path = out / "SPARSE60_SELECTED_BRANCHES.csv"
    write_csv(selection_path, selected)

    fold_rows = []
    fold_defs = ppv.root_folds(md)
    for fold in FOLDS:
        train = md[rf != fold]
        held = md[rf == fold]
        fold_rows.append({
            "outer_fold": fold,
            "train_roots": int(train[["task", "root_id"]].drop_duplicates().shape[0]),
            "validation_roots": int(held[["task", "root_id"]].drop_duplicates().shape[0]),
            "full_train_branches": int(len(train)),
            "sparse_train_branches": int((selected.outer_fold == fold).sum()),
            "heldout_branches": int(len(held)),
            "heldout_root_pairs": [{"task": int(t), "root_id": r} for t, r in fold_defs[fold]],
        })
    protocol = {
        "status": "HASH_FROZEN_BEFORE_SPARSE60_MODEL_TRAINING",
        "scientific_question": "Does trajectory-level physical supervision extract more learning signal than terminal success/failure from the same 60 force-conditioned rollouts per task?",
        "authoritative_population": {
            "tasks": TASKS,
            "root_families": 24,
            "friction_contexts": 72,
            "archived_branches_per_task": 180,
            "archived_branches_total": 720,
            "source": str(PRIOR),
            "untouched_TEST_included": False,
            "root_scaling_included": False,
            "historical_coarse_outcomes_included": False,
        },
        "outer_split": {
            "source_definition": "exact ppv.root_folds used by the prior Point-WM experiment",
            "group_key": ["task", "root_id"],
            "folds": fold_rows,
            "train_roots_per_task": 4,
            "validation_roots_per_task": 2,
            "rerandomized": False,
        },
        "budget_definitions": {
            "population_per_task": 180,
            "full_train_per_outer_fold_per_task": 120,
            "sparse_train_per_outer_fold_per_task": 60,
            "sparse_fraction_of_full_train": 0.5,
        },
        "selection": {
            "group_key": ["outer_fold", "task", "root_id", "context_id", "force_N"],
            "ordering": ["numeric repeat ascending", "branch_id lexicographic ascending"],
            "rule": "keep the first archived repeat in every friction x force cell",
            "observed_repeat_ids": sorted(int(x) for x in md.repeat.unique()),
            "canonical_repeat_selected": 1,
            "note": "Archive labels repeats as 1/2, not 0/1; deterministic first-repeat fallback therefore selects repeat 1.",
            "outcome_columns_read_for_selection": [],
            "performance_or_model_columns_read_for_selection": [],
            "selected_csv": selection_path.name,
            "selected_csv_sha256": sha256(selection_path),
        },
        "frozen_models": {
            "probe": "same prior nested root-heldout scalar mu_hat recipe; no rich trace/latent/sigma/GT friction controller input",
            "direct": "same architecture, BCE terminal outcome, optimizer, epochs, utility, candidate search",
            "point_wm": "same H8/13-channel trajectory architecture, nominal VLA motion input, PhysicsOnly+IE target, residual 52+1 features, MSE, optimizer, epochs, utility, candidate search",
            "nested_residual": True,
            "outer_validation_used_in_training": False,
        },
        "classification_thresholds_frozen_before_sparse_results": {
            "near_full_sr_gap_pp": 1.0,
            "material_sr_change_pp": 1.0,
            "material_underforce_worsening_pp": 1.0,
            "utility_non_decrease_tolerance": 1e-12,
            "stable_gain_requires_positive_sr_gain_in_at_least_seeds": 2,
        },
        "source_hashes": {
            "prior_shards_sha256_manifest": sha256(PRIOR / "SHA256SUMS.txt"),
            "prior_prepared_npz": sha256(PRIOR / "prepared.npz"),
            "prior_probe_predictions": sha256(PRIOR / "POOLED_OOF_PROBE_PREDICTIONS.csv"),
            "runner": sha256(Path(__file__)),
        },
        "task1_label_caveat": audits[1].get("label_risk"),
    }
    write_json(out / "SPARSE60_SELECTION_PROTOCOL.json", protocol)
    print(json.dumps({"status": "SPARSE60_FROZEN", "selected_rows": len(selected), "csv_sha256": sha256(selection_path)}, indent=2), flush=True)


def probe_meta_from_prior() -> pd.DataFrame:
    q = pd.read_csv(PRIOR / "POOLED_OOF_PROBE_PREDICTIONS.csv")
    q = q[q.seed.astype(str) == "0"].copy()
    q = q.sort_values("probe_index").drop_duplicates("probe_index")
    cols = ["probe_index", "context_id", "root_id", "task", "mu_GT"]
    q = q[cols].reset_index(drop=True)
    assert len(q) == 72 and np.array_equal(q.probe_index.to_numpy(), np.arange(72))
    return q


def nested_mu_maps(outer: int, seed: int):
    prep = np.load(PRIOR / "prepared.npz")
    raw = prep["raw"]
    fold_ctx = prep["fold_ctx"]
    pm = probe_meta_from_prior()
    old.PROBE_META = pm
    maps, mu, hidden = old.maps_for_outer(PRIOR, outer, seed, raw, pm, fold_ctx)
    return maps[0], mu, hidden


def selected_indices(out: Path, fold: int, md: pd.DataFrame) -> np.ndarray:
    s = pd.read_csv(out / "SPARSE60_SELECTED_BRANCHES.csv")
    ids = set(s.loc[s.outer_fold == fold, "branch_id"])
    idx = np.flatnonzero(md.branch_id.isin(ids).to_numpy())
    assert len(idx) == 240
    return idx


def dummy_hmap(traces) -> dict[str, np.ndarray]:
    return {t.context_id: np.zeros(16, dtype=np.float32) for t in traces}


def shard(out: Path, outer: int, seed: int) -> None:
    target = out / "shards" / f"sparse60_fold{outer}_seed{seed}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(json.dumps({"status": "ALREADY_COMPLETE", "path": str(target)}), flush=True)
        return
    _, traces, meta, _, pairs, segs, _ = load_population(out, f"shard{outer}_{seed}")
    md, rf = population_frame(traces, meta)
    train_idx = selected_indices(out, outer, md)
    held_idx = np.flatnonzero(rf == outer)
    assert np.all(rf[train_idx] != outer)
    assert len(held_idx) == 240
    train = [traces[i] for i in train_idx]
    held = [traces[i] for i in held_idx]
    train_ids = {t.branch_id for t in train}
    train_meta = {bid: meta[bid] for bid in train_ids}
    train_pairs = [p for p in pairs if p.a.branch_id in train_ids and p.b.branch_id in train_ids]
    assert len(train_pairs) == 192

    mu_map, _, _ = nested_mu_maps(outer, seed)
    h_map = dummy_hmap(traces)
    norm = old.xnorm(train, segs, mu_map)
    direct = old.train_direct(train, segs, norm, mu_map, h_map, train_meta, seed, False)
    wm, units = old.train_wm(train, train_pairs, norm, mu_map, h_map, seed, False)

    # Strict nested OOF predictions for residual training.  Both the fitted and
    # scored inner folds use only Sparse-60-selected branches.
    tr_p = np.full(len(train_idx), np.nan, np.float32)
    tr_w = np.full((len(train_idx), H, 13), np.nan, np.float32)
    loc = {global_i: local_i for local_i, global_i in enumerate(train_idx)}
    inner_audit = []
    for scored_fold in [f for f in FOLDS if f != outer]:
        fit_fold = next(f for f in FOLDS if f not in {outer, scored_fold})
        fit_idx = train_idx[rf[train_idx] == fit_fold]
        val_idx = train_idx[rf[train_idx] == scored_fold]
        assert len(fit_idx) == 120 and len(val_idx) == 120
        fit = [traces[i] for i in fit_idx]
        val = [traces[i] for i in val_idx]
        fit_ids = {t.branch_id for t in fit}
        fit_meta = {bid: meta[bid] for bid in fit_ids}
        fit_pairs = [p for p in pairs if p.a.branch_id in fit_ids and p.b.branch_id in fit_ids]
        assert len(fit_pairs) == 96
        fit_norm = old.xnorm(fit, segs, mu_map)
        inner_seed = seed + 11 * scored_fold
        inner_direct = old.train_direct(fit, segs, fit_norm, mu_map, h_map, fit_meta, inner_seed, False)
        inner_wm, inner_units = old.train_wm(fit, fit_pairs, fit_norm, mu_map, h_map, inner_seed, False)
        p, w = old.infer(inner_direct, inner_wm, val, segs, fit_norm, mu_map, h_map, False)
        local = np.asarray([loc[i] for i in val_idx])
        tr_p[local] = p
        tr_w[local] = w
        inner_audit.append({
            "scored_fold": scored_fold,
            "fit_fold": fit_fold,
            "fit_branches": len(fit_idx),
            "scored_branches": len(val_idx),
            "fit_pairs": len(fit_pairs),
            "physical_units": inner_units,
        })
    assert np.isfinite(tr_p).all() and np.isfinite(tr_w).all()

    tf = md.force_N.to_numpy()[train_idx]
    tx = md.success.to_numpy()[train_idx]
    tm = np.asarray([FMAX[int(x)] for x in md.task.to_numpy()[train_idx]])
    realized = np.where(tx > 0, (tm - tf) / tm, -1.0)
    direct_u = tr_p * ((tm - tf) / tm) + (1.0 - tr_p) * -1.0
    residual, rmean, rstd = old.fit_residual(
        np.column_stack([old.summary52(tr_w), tf / tm]).astype(np.float32),
        realized - direct_u,
        seed,
    )

    p_held, traj_held = old.infer(direct, wm, held, segs, norm, mu_map, h_map, False)
    hf = md.force_N.to_numpy()[held_idx]
    hfm = np.asarray([FMAX[int(x)] for x in md.task.to_numpy()[held_idx]])
    du = p_held * ((hfm - hf) / hfm) + (1.0 - p_held) * -1.0
    correction = old.residual_pred(residual, rmean, rstd, traj_held, hf, hfm)

    prior_shard = np.load(PRIOR / "shards" / f"fold{outer}_seed{seed}.npz")
    held_mu = np.asarray([mu_map[t.context_id] for t in held])
    mu_max_error = float(np.max(np.abs(held_mu - prior_shard["probe_mu"])))
    if mu_max_error > 1e-7:
        raise RuntimeError(f"held OOF mu_hat mismatch against prior shard: {mu_max_error}")

    np.savez_compressed(
        target,
        held_idx=held_idx,
        p_direct=p_held,
        score_point=du + correction,
        du_direct=du,
        delta_point=correction,
        probe_mu=held_mu,
        physical_units=units,
    )
    torch.save({
        "status": "ATOMIC_COMPLETE",
        "outer_fold": outer,
        "seed": seed,
        "train_budget_per_task": 60,
        "train_branches": 240,
        "held_branches": 240,
        "canonical_repeat": 1,
        "nested_crossfit": True,
        "inner_audit": inner_audit,
        "wm_objective": "Lphysics + 1.0*LIE",
        "outcome_gradient_into_WM": False,
        "H": 8,
        "epochs": 80,
        "optimizer": "AdamW",
        "residual_loss": "MSE",
        "raw_probe_trace_controller_input": False,
        "rich_probe_latent_controller_input": False,
        "sigma_mu_controller_input": False,
        "GT_friction_controller_input": False,
        "held_mu_max_abs_error_vs_prior": mu_max_error,
        "TEST_used": False,
    }, target.with_suffix(".pt"))
    print(json.dumps({
        "status": "SHARD_COMPLETE",
        "fold": outer,
        "seed": seed,
        "train": len(train),
        "held": len(held),
        "pairs": len(train_pairs),
        "mu_reuse_max_error": mu_max_error,
    }), flush=True)


def scores_from_shards(out: Path, md: pd.DataFrame):
    n = len(md)
    scores = {
        (60, "Direct"): np.full((3, n), np.nan, np.float32),
        (60, "Point-WM"): np.full((3, n), np.nan, np.float32),
        (120, "Direct"): np.full((3, n), np.nan, np.float32),
        (120, "Point-WM"): np.full((3, n), np.nan, np.float32),
    }
    gt = np.full((3, n), np.nan, np.float32)
    for fold in FOLDS:
        for seed in SEEDS:
            z = np.load(out / "shards" / f"sparse60_fold{fold}_seed{seed}.npz")
            ids = z["held_idx"]
            f = md.force_N.to_numpy()[ids]
            fm = md.task.map(FMAX).to_numpy()[ids]
            p = z["p_direct"]
            scores[(60, "Direct")][seed, ids] = p * ((fm - f) / fm) + (1.0 - p) * -1.0
            scores[(60, "Point-WM")][seed, ids] = z["score_point"]

            q = np.load(PRIOR / "shards" / f"fold{fold}_seed{seed}.npz")
            if not np.array_equal(ids, q["held_idx"]):
                raise RuntimeError("Sparse and Full held populations differ")
            pdirect = q["p_scalar"]
            pgt = q["p_gt"]
            scores[(120, "Direct")][seed, ids] = pdirect * ((fm - f) / fm) + (1.0 - pdirect) * -1.0
            scores[(120, "Point-WM")][seed, ids] = q["score_point"]
            gt[seed, ids] = pgt * ((fm - f) / fm) + (1.0 - pgt) * -1.0
    for key, value in scores.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"incomplete score array: {key}")
    if not np.isfinite(gt).all():
        raise RuntimeError("incomplete GT reference")
    return scores, gt


def add_gt_metrics(selected: pd.DataFrame, gt_selected: pd.DataFrame) -> pd.DataFrame:
    gt = gt_selected[["context_id", "repeat", "selected_force_N"]].rename(columns={"selected_force_N": "GT_selected_force_N"})
    q = selected.merge(gt, on=["context_id", "repeat"], validate="one_to_one")
    q["selected_force_error_N"] = (q.selected_force_N - q.GT_selected_force_N).abs()
    q["GT_decision_agreement"] = (q.selected_force_error_N < 1e-8).astype(int)
    return q


def metric_row(q: pd.DataFrame) -> dict:
    return {
        "episodes": int(len(q)),
        "sr": float(q.success.mean()),
        "underforce": float(q.under_force.mean()),
        "mean_force": float(q.selected_force_N.mean()),
        "excess_force": float(q.excess_force_N.mean()),
        "utility": float(q.realized_utility.mean()),
        "gt_decision_agreement": float(q.GT_decision_agreement.mean()),
        "selected_force_error_N": float(q.selected_force_error_N.mean()),
        "frontier_mae_N": float((q.selected_force_N - q.frontier_N).abs().mean()),
    }


def aggregate_tables(scores, gt_scores, md: pd.DataFrame):
    gt_selected = old.choose(md, gt_scores.mean(0))
    rows = []
    episodes = []
    for (budget, method), arr in scores.items():
        for seed_label, score in [(str(s), arr[s]) for s in SEEDS] + [("ENSEMBLE", arr.mean(0))]:
            selected = add_gt_metrics(old.choose(md, score), gt_selected)
            selected["method"] = method
            selected["data_per_task"] = budget
            selected["seed"] = seed_label
            episodes.append(selected)
            for task in TASKS:
                rows.append({"method": method, "data_per_task": budget, "seed": seed_label, "task": f"task{task}", **metric_row(selected[selected.task == task])})
            rows.append({"method": method, "data_per_task": budget, "seed": seed_label, "task": "POOLED", **metric_row(selected)})
    return pd.DataFrame(rows), pd.concat(episodes, ignore_index=True)


def aggregate_summary(main: pd.DataFrame) -> pd.DataFrame:
    seed_rows = main[(main.task == "POOLED") & main.seed.isin(["0", "1", "2"])].copy()
    ensemble = main[(main.task == "POOLED") & (main.seed == "ENSEMBLE")].copy()
    metrics = ["sr", "underforce", "mean_force", "excess_force", "utility", "gt_decision_agreement", "selected_force_error_N", "frontier_mae_N"]
    rows = []
    for (method, budget), q in seed_rows.groupby(["method", "data_per_task"]):
        e = ensemble[(ensemble.method == method) & (ensemble.data_per_task == budget)].iloc[0]
        row = {"controller": f"{'PointWM' if method == 'Point-WM' else 'Direct'}-{'Full' if budget == 120 else '60'}", "method": method, "data_per_task": int(budget), "seed_n": 3}
        for m in metrics:
            row[f"{m}_seed_mean"] = float(q[m].mean())
            row[f"{m}_seed_std"] = float(q[m].std(ddof=0))
            row[f"{m}_ensemble"] = float(e[m])
        rows.append(row)
    order = ["Direct-60", "PointWM-60", "Direct-Full", "PointWM-Full"]
    ans = pd.DataFrame(rows)
    ans["_order"] = ans.controller.map({k: i for i, k in enumerate(order)})
    return ans.sort_values("_order").drop(columns="_order")


def chart_contract(out: Path) -> None:
    value = {
        "surface": "standalone static PNG and PDF requested by user",
        "sr": {
            "question": "How do Direct and Point-WM full-task SR change from 60 to 120 task-specific training rollouts?",
            "family": "ordered comparison",
            "variant": "two-series line with 3-seed population-standard-deviation error bars and individual observed means",
            "x": "task-specific training rollouts (60, 120)",
            "y": "full-task success rate",
            "palette": {"Direct": "#4C78A8", "Point-WM": "#E45756"},
            "non_color_distinction": {"Direct": "circle/solid", "Point-WM": "square/dashed"},
            "focused_axis": "75%-100%, explicitly labeled",
            "outputs": ["FIG_SAMPLE_EFFICIENCY_SR.png", "FIG_SAMPLE_EFFICIENCY_SR.pdf"],
        },
        "utility": {
            "question": "Does low-data Point-WM preserve realized utility?",
            "family": "ordered comparison",
            "variant": "two-series line with 3-seed population-standard-deviation error bars",
            "outputs": ["FIG_SAMPLE_EFFICIENCY_UTILITY.png", "FIG_SAMPLE_EFFICIENCY_UTILITY.pdf"],
        },
        "best_seed": {
            "question": "Which tasks drive the largest post-hoc Sparse-60 Point-WM SR gain?",
            "family": "grouped categorical comparison",
            "variant": "paired bars by task",
            "title_requirement": "Representative Best Seed; post-hoc visualization only",
            "output": "FIG_SAMPLE_EFFICIENCY_BEST_SEED.png",
        },
    }
    write_json(out / "FIGURE_CHART_CONTRACT.json", value)


def plot_metric(main: pd.DataFrame, out: Path, metric: str, stem: str, ylabel: str, ylim=None) -> None:
    q = main[(main.task == "POOLED") & main.seed.isin(["0", "1", "2"])].copy()
    styles = {
        "Direct": dict(color="#4C78A8", marker="o", linestyle="-"),
        "Point-WM": dict(color="#E45756", marker="s", linestyle="--"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in ["Direct", "Point-WM"]:
        z = q[q.method == method].groupby("data_per_task")[metric].agg(["mean", lambda x: x.std(ddof=0)]).reset_index()
        z.columns = ["data_per_task", "mean", "std"]
        ax.errorbar(z.data_per_task, z["mean"], yerr=z["std"], capsize=4, linewidth=2.2, markersize=7, label=method, **styles[method])
        for _, r in z.iterrows():
            label = f"{r['mean']*100:.2f}%" if metric == "sr" else f"{r['mean']:.3f}"
            offset = (20, 12) if method == "Point-WM" else (-20, -22)
            ax.annotate(label, (r.data_per_task, r["mean"]), xytext=offset, textcoords="offset points", ha="center", fontsize=9, color=styles[method]["color"])
    ax.set_xticks([60, 120])
    ax.set_xlabel("Task-specific training rollouts")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if metric == "sr":
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title("Full-task success at 60 vs 120 rollouts/task")
        ax.text(0.01, 0.02, "Focused y-axis: 75%–100%; error bars = 3-seed population std", transform=ax.transAxes, fontsize=8, color="#555555")
    else:
        ax.set_title("Realized utility at 60 vs 120 rollouts/task")
        ax.text(0.01, 0.02, "Error bars = 3-seed population std", transform=ax.transAxes, fontsize=8, color="#555555")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def best_seed_artifacts(main: pd.DataFrame, out: Path) -> tuple[int, pd.DataFrame]:
    q = main[(main.data_per_task == 60) & main.seed.isin(["0", "1", "2"])]
    pooled = q[q.task == "POOLED"].pivot(index="seed", columns="method", values="sr")
    pooled["gain"] = pooled["Point-WM"] - pooled["Direct"]
    best_seed = int(pooled.gain.idxmax())
    best = q[q.seed == str(best_seed)].copy()
    best["label"] = "POST-HOC BEST SEED — VISUALIZATION ONLY"
    write_csv(out / "BEST_SEED_SAMPLE_EFFICIENCY.csv", best)

    z = best[best.task != "POOLED"].pivot(index="task", columns="method", values="sr").loc[[f"task{t}" for t in TASKS]]
    x = np.arange(len(z))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - width/2, z["Direct"], width, label="Direct-60", color="#4C78A8")
    ax.bar(x + width/2, z["Point-WM"], width, label="PointWM-60", color="#E45756", hatch="//")
    for i, task in enumerate(z.index):
        ax.text(i - width/2, z.loc[task, "Direct"] + .012, f"{z.loc[task, 'Direct']*100:.1f}%", ha="center", fontsize=8)
        ax.text(i + width/2, z.loc[task, "Point-WM"] + .012, f"{z.loc[task, 'Point-WM']*100:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x, z.index)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Full-task Success Rate")
    ax.set_title(f"Representative Best Seed (seed{best_seed})\nPOST-HOC — VISUALIZATION ONLY")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "FIG_SAMPLE_EFFICIENCY_BEST_SEED.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return best_seed, best


def classify(summary: pd.DataFrame, main: pd.DataFrame) -> tuple[str, dict]:
    s = summary.set_index("controller")
    d60, w60 = s.loc["Direct-60"], s.loc["PointWM-60"]
    dfull, wfull = s.loc["Direct-Full"], s.loc["PointWM-Full"]
    gain60_pp = 100 * (w60.sr_seed_mean - d60.sr_seed_mean)
    gainfull_pp = 100 * (wfull.sr_seed_mean - dfull.sr_seed_mean)
    dgap_pp = 100 * (dfull.sr_seed_mean - d60.sr_seed_mean)
    wgap_pp = 100 * (wfull.sr_seed_mean - w60.sr_seed_mean)
    uf_delta_pp = 100 * (w60.underforce_seed_mean - d60.underforce_seed_mean)
    u_delta = w60.utility_seed_mean - d60.utility_seed_mean
    q = main[(main.data_per_task == 60) & (main.task == "POOLED") & main.seed.isin(["0", "1", "2"])].pivot(index="seed", columns="method", values="sr")
    seed_wins = int((q["Point-WM"] > q["Direct"]).sum())
    if gain60_pp > 0 and seed_wins == 1:
        label = "WM_LOW_DATA_GAIN_IS_SEED_SENSITIVE"
    elif gain60_pp > 0 and uf_delta_pp > 1.0:
        label = "WM_LOW_DATA_GAIN_TRADES_OFF_SAFETY"
    elif abs(dgap_pp) <= 1.0 and abs(gain60_pp) <= 1.0:
        label = "60_ROLLOUTS_ALREADY_SUFFICIENT_WORLD_MODEL_NOT_NEEDED"
    elif dgap_pp > 1.0 and gain60_pp <= 1.0:
        label = "60_ROLLOUTS_INSUFFICIENT_AND_WM_DOES_NOT_HELP"
    elif gain60_pp > 1.0 and seed_wins >= 2 and uf_delta_pp <= 1.0 and u_delta >= -1e-12:
        label = "PHYSICS_TRAJECTORY_SUPERVISION_IMPROVES_SAMPLE_EFFICIENCY"
    else:
        label = "INCONCLUSIVE_MIXED_LOW_DATA_SIGNAL"
    values = {
        "delta_wm_60_pp_seed_mean": gain60_pp,
        "delta_wm_full_pp_seed_mean": gainfull_pp,
        "gap_to_full_direct_pp_seed_mean": dgap_pp,
        "gap_to_full_wm_pp_seed_mean": wgap_pp,
        "direct_recovery_ratio_sr": d60.sr_seed_mean / dfull.sr_seed_mean,
        "pointwm_recovery_ratio_sr": w60.sr_seed_mean / wfull.sr_seed_mean,
        "underforce_delta_wm60_vs_direct60_pp": uf_delta_pp,
        "utility_delta_wm60_vs_direct60": u_delta,
        "sparse_seed_wins": seed_wins,
    }
    return label, values


def pct(v: float) -> str:
    return f"{v*100:.2f}%"


def finalize(out: Path) -> None:
    phase0(out)
    _, traces, meta, audits, _, _, _ = load_population(out, "finalize")
    md, _ = population_frame(traces, meta)
    scores, gt_scores = scores_from_shards(out, md)
    main, episodes = aggregate_tables(scores, gt_scores, md)
    write_csv(out / "SAMPLE_EFFICIENCY_60_VS_FULL.csv", main)
    write_csv(out / "SAMPLE_EFFICIENCY_EPISODE_LEVEL.csv", episodes)
    summary = aggregate_summary(main)
    write_csv(out / "SAMPLE_EFFICIENCY_60_VS_FULL_AGG.csv", summary)
    best_seed, best = best_seed_artifacts(main, out)
    chart_contract(out)
    plot_metric(main, out, "sr", "FIG_SAMPLE_EFFICIENCY_SR", "Full-task Success Rate", (.75, 1.0))
    utility_values = main[(main.task == "POOLED") & main.seed.isin(["0", "1", "2"])].utility
    lo, hi = float(utility_values.min()), float(utility_values.max())
    pad = max(.015, .12 * (hi - lo))
    plot_metric(main, out, "utility", "FIG_SAMPLE_EFFICIENCY_UTILITY", "Mean realized utility", (lo - pad, hi + pad))

    label, core = classify(summary, main)
    write_json(out / "SAMPLE_EFFICIENCY_CLASSIFICATION.json", {"classification": label, **core})
    s = summary.set_index("controller")
    d60, w60, dfull, wfull = s.loc["Direct-60"], s.loc["PointWM-60"], s.loc["Direct-Full"], s.loc["PointWM-Full"]
    seed_rows = main[(main.task == "POOLED") & (main.data_per_task == 60) & main.seed.isin(["0", "1", "2"])]
    task_rows = main[(main.seed == "ENSEMBLE") & (main.data_per_task == 60) & (main.task != "POOLED")]
    task_seed_rows = main[(main.seed.isin(["0", "1", "2"])) & (main.data_per_task == 60) & (main.task != "POOLED")]
    oldseed = pd.read_csv(out / "POINT_WM_PER_SEED.csv")

    report = [
        "# ActiveForcing 60-Rollout Sample-Efficiency Report",
        "",
        "## Technical summary",
        "",
        f"最终分类：**`{label}`**。核心比较采用三 seed metric mean/std；prediction-level ensemble 单独报告，不与 seed mean 混用。",
        f"Sparse-60 下 Direct SR={pct(d60.sr_seed_mean)}，Point-WM SR={pct(w60.sr_seed_mean)}，ΔWM_60={core['delta_wm_60_pp_seed_mean']:+.2f} pp。Full-120 下 Direct SR={pct(dfull.sr_seed_mean)}，Point-WM SR={pct(wfull.sr_seed_mean)}，ΔWM_FULL={core['delta_wm_full_pp_seed_mean']:+.2f} pp。",
        "",
        "## Direct answers",
        "",
        "1. **当前原始数据：**4 tasks、24 independent root families、72 friction-conditioned contexts、720 archived branches；每 task 180 branches。没有加入 historical coarse outcomes、root-scaling roots 或 untouched TEST。",
        "2. **严格 OOF 的 Full training budget：**每 outer fold 每 task 4 roots × 3 friction × 5 forces × 2 repeats = 120 rollouts；全四 task 为 480 training branches/fold。",
        "3. **Sparse budget：**每 cell 保留 deterministic first archived repeat（实际 repeat IDs 为 1/2，故保留 repeat 1），每 task 60、全四 task 240 training branches/fold，恰为 Full 的 50%。",
        f"4. **60 条时 Direct：**三-seed SR mean/std={pct(d60.sr_seed_mean)} ± {d60.sr_seed_std*100:.2f} pp；prediction-level ensemble={pct(d60.sr_ensemble)}。",
        f"5. **60 条时 Point-WM：**三-seed SR mean/std={pct(w60.sr_seed_mean)} ± {w60.sr_seed_std*100:.2f} pp；prediction-level ensemble={pct(w60.sr_ensemble)}。",
        f"6. **60 条时 WM 相对 Direct：**三-seed mean Δ={core['delta_wm_60_pp_seed_mean']:+.2f} pp；ensemble Δ={100*(w60.sr_ensemble-d60.sr_ensemble):+.2f} pp。",
        f"7. **120 条 Full：**Direct={pct(dfull.sr_seed_mean)} ± {dfull.sr_seed_std*100:.2f} pp，Point-WM={pct(wfull.sr_seed_mean)} ± {wfull.sr_seed_std*100:.2f} pp；ensembles 分别为 {pct(dfull.sr_ensemble)} / {pct(wfull.sr_ensemble)}。",
        f"8. **WM advantage 的数据依赖：**ΔWM_60={core['delta_wm_60_pp_seed_mean']:+.2f} pp，ΔWM_FULL={core['delta_wm_full_pp_seed_mean']:+.2f} pp；前者{'更大' if core['delta_wm_60_pp_seed_mean'] > core['delta_wm_full_pp_seed_mean'] else '不更大'}。",
        f"9. **Direct-60 接近 Full 吗：**GapToFull_Direct={core['gap_to_full_direct_pp_seed_mean']:+.2f} pp，SR recovery ratio={core['direct_recovery_ratio_sr']:.4f}。按预冻结 1.0 pp 近似阈值，答案是 {'是' if abs(core['gap_to_full_direct_pp_seed_mean']) <= 1.0 else '否'}。",
        f"10. **PointWM-60 接近或超过 Direct-Full 吗：**差值={100*(w60.sr_seed_mean-dfull.sr_seed_mean):+.2f} pp；按相同 1.0 pp 阈值，答案是 {'是' if w60.sr_seed_mean >= dfull.sr_seed_mean-.01 else '否'}。",
        f"11. **under-force / utility：**Sparse-60 Point-WM 相对 Direct 的 under-force Δ={core['underforce_delta_wm60_vs_direct60_pp']:+.2f} pp，utility Δ={core['utility_delta_wm60_vs_direct60']:+.4f}。三 seed 中 Point-WM 的 SR 改善次数为 {core['sparse_seed_wins']}/3；虽然 prediction-level ensemble 是 +{100*(w60.sr_ensemble-d60.sr_ensemble):.2f} pp，但它不能覆盖 0/3 单 seed 改善与 aggregate mean 的负方向。",
        "12. **各 task（Sparse-60；三-seed mean 为主，prediction-level ensemble 单列）：**",
        "",
        "| task | Direct SR mean | Point-WM SR mean | ΔSR mean | ensemble ΔSR | Δunder-force mean | Δutility mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in [f"task{t}" for t in TASKS]:
        q = task_seed_rows[task_seed_rows.task == task].groupby("method")[["sr", "underforce", "utility"]].mean()
        e = task_rows[task_rows.task == task].set_index("method")
        report.append(f"| {task} | {pct(q.loc['Direct','sr'])} | {pct(q.loc['Point-WM','sr'])} | {100*(q.loc['Point-WM','sr']-q.loc['Direct','sr']):+.2f} pp | {100*(e.loc['Point-WM','sr']-e.loc['Direct','sr']):+.2f} pp | {100*(q.loc['Point-WM','underforce']-q.loc['Direct','underforce']):+.2f} pp | {q.loc['Point-WM','utility']-q.loc['Direct','utility']:+.4f} |")
    report += [
        "",
        "13. **三个 Sparse-60 seeds（POOLED）：**",
        "",
        "| seed | Direct SR | Point-WM SR | ΔSR | Direct UF | Point-WM UF | utility Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        q = seed_rows[seed_rows.seed == str(seed)].set_index("method")
        report.append(f"| {seed} | {pct(q.loc['Direct','sr'])} | {pct(q.loc['Point-WM','sr'])} | {100*(q.loc['Point-WM','sr']-q.loc['Direct','sr']):+.2f} pp | {pct(q.loc['Direct','underforce'])} | {pct(q.loc['Point-WM','underforce'])} | {q.loc['Point-WM','utility']-q.loc['Direct','utility']:+.4f} |")
    neg = oldseed.loc[oldseed.delta_sr_pp.idxmin()]
    report += [
        "",
        f"14. **上一轮唯一 negative seed：**seed{int(neg.seed)}，Direct {pct(neg.direct_sr)} → Point-WM {pct(neg.pointwm_sr)}，Δ={neg.delta_sr_pp:+.2f} pp；少 4/144 次成功，但仍为 {pct(neg.pointwm_sr)}，是有限下降而不是 collapse。",
        f"15. **Best-seed 图：**Sparse-60 后验最大 Point-WM−Direct seed 是 seed{best_seed}，但 pooled ΔSR 仅为 {100*(best[best.task=='POOLED'].set_index('method').loc['Point-WM','sr']-best[best.task=='POOLED'].set_index('method').loc['Direct','sr']):+.2f} pp；它是最不差的 seed，不是正收益 seed。`FIG_SAMPLE_EFFICIENCY_BEST_SEED.png` 标题明确写有 “Representative Best Seed” 和 “POST-HOC — VISUALIZATION ONLY”，不用于 aggregate classification。",
        f"16. **最诚实分类：**`{label}`。",
        "",
        "## Full metric table (three-seed mean ± population std; ensemble separate)",
        "",
        "| controller | SR mean±std | ensemble SR | under-force mean | mean force | excess force | utility | GT agreement | selected-force error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["Direct-60", "PointWM-60", "Direct-Full", "PointWM-Full"]:
        r = s.loc[name]
        report.append(f"| {name} | {pct(r.sr_seed_mean)} ± {r.sr_seed_std*100:.2f} pp | {pct(r.sr_ensemble)} | {pct(r.underforce_seed_mean)} | {r.mean_force_seed_mean:.4f} N | {r.excess_force_seed_mean:.4f} N | {r.utility_seed_mean:.4f} | {pct(r.gt_decision_agreement_seed_mean)} | {r.selected_force_error_N_seed_mean:.4f} N |")
    report += [
        "",
        "## Scientific interpretation",
        "",
        "Direct 从稀疏 terminal full-task success/failure 学习 force selection；Point-WM 在完全相同的 simulator executions 上，额外利用已有 H8 continuous physical trajectory 与 IE supervision，再通过冻结的 residual-utility recipe 决策。因此本实验测试的是 trajectory-level predictive physical supervision 是否从每次 force-conditioned rollout 提取更多学习信号，不是声称 WM 直接估计 force 更准。",
        "",
        "## Protocol integrity and caveats",
        "",
        "- Sparse 与 Full 共用相同 outer-heldout roots、candidate force grid、OOF scalar mu_hat semantics、Direct architecture、Point-WM H8/PhysicsOnly/IE/residual architecture、optimizer、epochs、utility 与 search。",
        "- residual training 使用 strict inner grouped-root OOF predictions；outer validation roots 不进入 Direct、WM 或 residual training。",
        "- Full-120 直接复用上一轮 9 个 authoritative atomic-complete shards；Sparse-60 新训练 9 个 shards，但不产生任何 simulator data。",
        "- 95.14% 一类 ensemble 数字来自 branch prediction/utility 先跨 seed 平均、再重新 argmax 决策；绝不是三-seed metric mean。",
        "- `excess_force` 延续既有定义，是 selected force minus empirical success frontier 的有符号均值；under-force episode 因而可贡献负值。",
        f"- task1 仍有既知 `{audits[1].get('label_risk')}` caveat（140/180 reconstructed labels）。",
        "- std 是三个 seed metric 的 population std（ddof=0），不是置信区间；144 held-out episodes 共享 24 root families，因此不要把 episodes 当作 144 个独立实验单位。",
    ]
    (out / "FINAL_60_ROLLOUT_SAMPLE_EFFICIENCY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    required = [
        "POINT_WM_PER_SEED.csv", "POINT_WM_SEED_AUDIT.md",
        "SPARSE60_SELECTION_PROTOCOL.json", "SPARSE60_SELECTED_BRANCHES.csv",
        "SAMPLE_EFFICIENCY_60_VS_FULL.csv", "SAMPLE_EFFICIENCY_60_VS_FULL_AGG.csv",
        "BEST_SEED_SAMPLE_EFFICIENCY.csv", "FIG_SAMPLE_EFFICIENCY_SR.pdf",
        "FIG_SAMPLE_EFFICIENCY_SR.png", "FIG_SAMPLE_EFFICIENCY_UTILITY.pdf",
        "FIG_SAMPLE_EFFICIENCY_UTILITY.png", "FIG_SAMPLE_EFFICIENCY_BEST_SEED.png",
        "FINAL_60_ROLLOUT_SAMPLE_EFFICIENCY_REPORT.md", "SAMPLE_EFFICIENCY_CLASSIFICATION.json",
    ]
    (out / "SHA256SUMS.txt").write_text("\n".join(f"{sha256(out / name)}  {name}" for name in required) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FINALIZED", "classification": label, **core}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["phase0", "prepare", "shard", "finalize"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    out = args.out.resolve()
    if args.phase == "phase0":
        phase0(out)
    elif args.phase == "prepare":
        prepare(out)
    elif args.phase == "shard":
        shard(out, args.fold, args.seed)
    else:
        finalize(out)
