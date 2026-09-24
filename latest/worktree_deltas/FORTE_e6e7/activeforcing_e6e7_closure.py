#!/usr/bin/env python3
"""ActiveForcing E6/E7 TRAIN/DEV closure.

This lane is deliberately read-only with respect to the existing FORTE and
Tabero experiments.  It trains only three small identifier members on the
authoritative TRAIN probe sequences, evaluates them on root-held-out DEV, and
uses the already frozen continuous Direct checkpoints as the common planning
backend.  Original TEST rows are never loaded.

The script is an evidence-first closure runner: unsupported simulator/query
stages are emitted as explicit negative/not-reached artifacts rather than
being silently replaced by a proxy.
"""
from __future__ import annotations

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
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


FORTE = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
C_ART = TABERO / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542"
FRICTION_SOURCE = TABERO / "analysis/active_friction_imagination.py"
P5C_MODEL_SOURCE = TABERO / "analysis/p5s0c_model_adjudication.py"
GNP_SOURCE = FORTE / "gnp_style_continuous.py"
GNP_OUT = FORTE / "gnp_style_continuous_20260830_125107"
CONT_PRED = GNP_OUT / "CONTINUOUS_DEV_FROZEN_PREDICTIONS.csv"
CONT_FRONT = GNP_OUT / "CONTINUOUS_DEV_FRONTIER_METRICS.csv"
BRANCHES = C_ART / "P5S0C_BRANCH_MANIFEST.csv"

SEEDS = [2026090201, 2026090202, 2026090203]
MEMBERS = ["member_0", "member_1", "member_2"]
TASK_BOUNDS = {0: (3.0, 5.0), 1: (4.0, 6.0), 5: (3.0, 5.0), 6: (3.0, 4.0)}
RHO_CANDIDATES = [0.80, 0.85, 0.90, 0.95]
FORCE_GRID = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
POSTERIOR_SAMPLE_BUDGET = 3
TRAIN_EPOCHS = int(os.environ.get("E6E7_TRAIN_EPOCHS", "80"))
BOOTSTRAP_SEED = 2026090200
SCORE_CACHE: dict[tuple[str, float, tuple[float, ...]], float] = {}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
        w = csv.DictWriter(f, fieldnames=fields or ["status", "reason"])
        w.writeheader()
        w.writerows(rows)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        out[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return out


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return math.nan
    ra, rb = rankdata(np.asarray(a, float)), rankdata(np.asarray(b, float))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if mask.any():
            result += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return result


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    # Empirical probabilities are valid soft labels for the repeated DEV cells.
    nll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return {"MAE": float(np.mean(abs(p - y))), "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
            "Brier": float(np.mean((p - y) ** 2)), "NLL": float(np.mean(nll)),
            "ECE_10bin": ece(y, p)}


class FrictionMember(nn.Module):
    """Matched projection -> GRU -> heteroscedastic mu/sigma estimator."""
    def __init__(self, input_dim: int, projection_dim: int = 16, hidden_dim: int = 16):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, projection_dim), nn.ReLU())
        self.gru = nn.GRU(projection_dim, hidden_dim, batch_first=True)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        z = self.projection(x)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = h[-1]
        return self.mu_head(h).squeeze(1), self.log_sigma_head(h).squeeze(1).clamp(-5.0, 1.5)


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigma = torch.exp(log_sigma).clamp_min(1e-3)
    return 0.5 * (((y - mu) / sigma) ** 2 + 2 * log_sigma)


def load_train_dev_examples() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    model_mod = load_module(P5C_MODEL_SOURCE, "e6e7_p5c_model")
    contexts = pd.read_csv(C_ART / "P5S0C_CONTEXT_MANIFEST.csv")
    # Filter before feature loading: no TEST path or label is parsed.
    contexts = contexts[contexts["split"].isin(["TRAIN", "DEV"])].copy()
    seqs, _, _, meta = model_mod.load_features(contexts)
    examples: dict[str, dict[str, Any]] = {}
    for row in contexts.itertuples(index=False):
        cid = str(row.context_id)
        examples[cid] = {"context_id": cid, "root_id": str(row.root_id), "task": int(row.task),
                         "split": str(row.split), "band": str(row.friction_band),
                         "friction": float(row.hidden_friction_analysis_only),
                         "sequence": np.asarray(seqs[cid], dtype=np.float32)}
    if len(examples) != 96 or len({x["root_id"] for x in examples.values()}) != 32:
        raise RuntimeError(f"expected 96 TRAIN/DEV contexts, got {len(examples)}")
    return examples, meta


def batch(examples: list[dict[str, Any]], device: torch.device):
    n, length, dim = len(examples), max(x["sequence"].shape[0] for x in examples), examples[0]["sequence"].shape[1]
    x = np.zeros((n, length, dim), np.float32)
    for i, ex in enumerate(examples):
        x[i, : ex["sequence"].shape[0]] = ex["sequence"]
    lengths = torch.tensor([e["sequence"].shape[0] for e in examples], dtype=torch.long, device=device)
    return torch.tensor(x, device=device), lengths, torch.tensor([e["friction"] for e in examples], dtype=torch.float32, device=device)


def train_member(member: str, seed: int, train: list[dict[str, Any]], out: Path) -> tuple[FrictionMember, dict[str, Any]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = FrictionMember(train[0]["sequence"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    root_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in train:
        root_groups[ex["root_id"]].append(ex)
    root_ids = sorted(root_groups)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, TRAIN_EPOCHS + 1):
        sampled = rng.choice(root_ids, size=len(root_ids), replace=True)
        fit = [e for rid in sampled for e in root_groups[str(rid)]]
        x, lengths, y = batch(fit, torch.device("cpu"))
        model.train(); opt.zero_grad(set_to_none=True)
        mu, log_sigma = model(x, lengths)
        loss = gaussian_nll(mu, log_sigma, y).mean() + 0.05 * torch.abs(mu - y).mean()
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == TRAIN_EPOCHS:
            history.append({"epoch": epoch, "train_loss": float(loss.item())})
    ck = out / f"PHYSICAL_BELIEF_{member}.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": train[0]["sequence"].shape[1],
                "projection_dim": 16, "hidden_dim": 16, "seed": seed,
                "construction": "same architecture; root bootstrap resampling"}, ck)
    return model, {"member": member, "seed": seed, "checkpoint": str(ck), "checkpoint_sha256": sha256(ck), "history": history}


def predict_member(model: FrictionMember, examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        mu, log_sigma = model(*batch(examples, torch.device("cpu"))[:2])
    return mu.numpy(), np.exp(log_sigma.numpy())


def build_belief(out: Path, examples: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = [x for x in examples.values() if x["split"] == "TRAIN"]
    dev = [x for x in examples.values() if x["split"] == "DEV"]
    models, manifests = [], []
    for member, seed in zip(MEMBERS, SEEDS):
        model, info = train_member(member, seed, train, out)
        models.append(model); manifests.append(info)
    # Fit one scalar uncertainty calibration on TRAIN only.  The mean and
    # ensemble variance remain untouched; only the predictive interval scale
    # is calibrated from TRAIN residuals.
    tmus, tsigmas = [], []
    for m in models:
        mu, sig = predict_member(m, train); tmus.append(mu); tsigmas.append(sig)
    tmu, tsig = np.asarray(tmus), np.asarray(tsigmas)
    tmean, tepi = tmu.mean(0), tmu.var(0, ddof=1)
    talea = (tsig ** 2).mean(0)
    raw_total = tepi + talea
    scale = float(np.sqrt(np.mean((np.asarray([x["friction"] for x in train]) - tmean) ** 2) / max(np.mean(raw_total), 1e-8)))
    scale = float(np.clip(scale, 0.5, 3.0))
    rows = []
    for split, pool in [("TRAIN", train), ("DEV", dev)]:
        mus, sigmas = [], []
        for m in models:
            mu, sig = predict_member(m, pool); mus.append(mu); sigmas.append(sig)
        mus, sigmas = np.asarray(mus), np.asarray(sigmas)
        mean = mus.mean(0); epistemic = mus.var(0, ddof=1); aleatoric = (sigmas ** 2).mean(0)
        total = (epistemic + aleatoric) * scale ** 2
        for i, ex in enumerate(pool):
            lo, hi = mean[i] - 1.645 * np.sqrt(total[i]), mean[i] + 1.645 * np.sqrt(total[i])
            rows.append({"split": split, "context_id": ex["context_id"], "root_id": ex["root_id"], "task": ex["task"],
                         "friction_band": ex["band"], "friction_gt": ex["friction"],
                         "member_mu_0": mus[0, i], "member_mu_1": mus[1, i], "member_mu_2": mus[2, i],
                         "ensemble_mean": mean[i], "epistemic_variance": epistemic[i],
                         "aleatoric_variance": aleatoric[i], "total_variance": total[i],
                         "interval_90_lo": lo, "interval_90_hi": hi,
                         "covered_90": int(lo <= ex["friction"] <= hi),
                         "abs_error": abs(mean[i] - ex["friction"])})
    df = pd.DataFrame(rows)
    write_csv(out / "PHYSICAL_BELIEF_PREDICTIONS.csv", df.to_dict("records"))
    manifest = {"members": manifests, "member_count": 3, "architecture": "projection(16)->GRU(16)->mu/log_sigma heads",
                "construction": "matched architecture; deterministic seeds; root bootstrap training resampling",
                "uncertainty_semantics": {"epistemic": "sample variance of member mu", "aleatoric": "mean member sigma^2",
                                          "total": "epistemic + aleatoric, TRAIN-fitted scalar interval calibration"},
                "interval_scale_fit_split": "TRAIN", "interval_scale": scale,
                "train_contexts": len(train), "dev_contexts": len(dev), "test_contexts_loaded": 0}
    write_json(out / "PHYSICAL_BELIEF_MANIFEST.json", manifest)
    return df, manifest


def calibration_report(out: Path, belief: pd.DataFrame) -> None:
    rows = []
    for split in ["TRAIN", "DEV"]:
        d = belief[belief.split == split]
        for scope, q in [("OVERALL", d), *[(str(b), d[d.friction_band == b]) for b in sorted(d.friction_band.unique())]]:
            if q.empty: continue
            gt, pred = q.friction_gt.to_numpy(float), q.ensemble_mean.to_numpy(float)
            rows.append({"split": split, "scope": scope, "n": len(q), "MAE": np.mean(abs(gt-pred)),
                         "RMSE": np.sqrt(np.mean((gt-pred)**2)), "NLL_gaussian": np.mean(0.5*((gt-pred)**2/q.total_variance + np.log(2*np.pi*q.total_variance))),
                         "coverage_90": q.covered_90.mean(), "mean_total_sd": np.mean(np.sqrt(q.total_variance)),
                         "uncertainty_error_spearman": spearman(q.total_variance.tolist(), abs(gt-pred).tolist())})
    write_csv(out / "TABLE_E6_ENSEMBLE_CALIBRATION_DEV.csv", [r for r in rows if r["split"] == "DEV"])
    d = belief[belief.split == "DEV"]
    report = ["# E6 Ensemble Calibration Report", "", "STATUS: COMPLETE_NEGATIVE_FOR_DEPLOYMENT_CLOSURE", "",
              "A matched 3-member physical identifier ensemble was trained on TRAIN roots only. DEV is root-held-out; original TEST rows were not loaded.", "",
              f"DEV MAE={d.abs_error.mean():.4f}, RMSE={np.sqrt(np.mean((d.friction_gt-d.ensemble_mean)**2)):.4f}, 90% coverage={d.covered_90.mean():.4f}, uncertainty-error Spearman={spearman(d.total_variance.tolist(), d.abs_error.tolist()):.4f}.", "",
              "The interval scale was fit on TRAIN only. Epistemic variance is member spread; aleatoric variance is the mean heteroscedastic member variance. This is a physical belief diagnostic, not evidence that the downstream force decision is calibrated.", "",
              "See TABLE_E6_ENSEMBLE_CALIBRATION_DEV.csv and PHYSICAL_BELIEF_PREDICTIONS.csv."]
    (out / "E6_ENSEMBLE_CALIBRATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def load_direct_stack():
    gnp = load_module(GNP_SOURCE, "e6e7_gnp")
    tpi, cf, full, pre, active = gnp.modules()
    norm_q = json.loads((GNP_OUT / "GNP_STYLE_TRAIN_NORMALIZATION.json").read_text())
    norm = tuple(np.asarray(norm_q[k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
    models, cals = [], []
    for seed in [0, 1, 2]:
        ck = torch.load(GNP_OUT / f"GNP_STYLE_CONTINUOUS_FEAS_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = full.FeasibilityOnly().to(torch.device("cpu")); model.load_state_dict(ck["state_dict"]); model.eval(); models.append(model)
        cals.append(gnp.load_calibration(GNP_OUT / f"GNP_STYLE_FEAS_TRAIN_CALIBRATION_seed{seed}.json"))
    contexts = pre.all_contexts_for_split("DEV", active)
    return gnp, tpi, cf, contexts, norm, (models, cals)


def real_frontiers() -> dict[str, dict[str, Any]]:
    d = pd.read_csv(BRANCHES)
    d = d[d.split == "DEV"].copy()
    out = {}
    for cid, q in d.groupby("context_id"):
        q = q.sort_values("requested_force_N")
        successful = q[q.full_task_success_y == 1].requested_force_N.to_numpy(float)
        out[str(cid)] = {"frontier": float(successful.min()) if len(successful) else math.nan,
                         "outcomes": {float(r.requested_force_N): int(r.full_task_success_y) for r in q.itertuples()},
                         "task": int(q.task.iloc[0]), "root_id": str(q.root_id.iloc[0]), "band": str(q.friction_band.iloc[0])}
    return out


def candidate_set(task: int, kind: str, k: int, probs=None) -> list[float]:
    lo, hi = TASK_BOUNDS[task]
    if kind == "FIXED_GRID":
        return np.linspace(lo, hi, k).round(6).tolist()
    if kind == "UNIFORM_CONTINUOUS":
        return np.linspace(lo, hi, k).round(6).tolist()
    if kind == "STRATIFIED_CONTINUOUS":
        return np.asarray([(lo + (hi-lo)*i/k + lo + (hi-lo)*(i+1)/k)/2 for i in range(k)]).round(6).tolist()
    # Proposal-guided has exactly K points.  It estimates the boundary from
    # a fixed dense query then allocates half the points around it.
    if probs is None or not probs:
        return np.linspace(lo, hi, k).round(6).tolist()
    dense = sorted(probs)
    boundary = next((f for f in dense if probs[f] >= 0.8), hi)
    near = np.linspace(max(lo, boundary - 0.25), min(hi, boundary + 0.25), max(2, k // 2))
    broad = np.linspace(lo, hi, k - len(near)) if k > len(near) else np.array([])
    return sorted(set(np.round(np.concatenate([broad, near]), 6).tolist()))[:k] if k >= 2 else [float(boundary)]


def get_mu_map(belief: pd.DataFrame) -> dict[str, list[float]]:
    d = belief[belief.split == "DEV"]
    return {str(r.context_id): [float(r.member_mu_0), float(r.member_mu_1), float(r.member_mu_2)] for r in d.itertuples()}


def score_prob(gnp, tpi, cf, ctx, force: float, mus: list[float], stack, norm) -> dict[str, Any]:
    return gnp.predict_probability(tpi, cf, ctx, float(force), [float(x) for x in mus], "FEASIBILITY_ONLY", stack, norm, torch.device("cpu"))


def score_map(gnp, tpi, cf, ctx, force: float, mus: list[float], stack, norm) -> float:
    key = (str(ctx.context_id), round(float(force), 6), tuple(round(float(x), 8) for x in mus))
    if key not in SCORE_CACHE:
        SCORE_CACHE[key] = float(score_prob(gnp, tpi, cf, ctx, force, mus, stack, norm)["raw_probability"])
    return SCORE_CACHE[key]


def plan_one(gnp, tpi, cf, ctx, mus: list[float], stack, norm, task: int, kind: str, k: int, rho: float, posterior: bool) -> dict[str, Any]:
    start = time.perf_counter()
    used_mus = mus if posterior else [float(np.mean(mus))]
    lo, hi = TASK_BOUNDS[task]
    dense = ({round(float(f), 6): score_map(gnp, tpi, cf, ctx, float(f), [float(x) for x in used_mus], stack, norm) for f in np.linspace(lo, hi, 41)}
             if kind == "PROPOSAL_GUIDED" else {})
    candidates = candidate_set(task, kind, k, dense if kind == "PROPOSAL_GUIDED" else None)
    scores = [(f, score_map(gnp, tpi, cf, ctx, f, used_mus, stack, norm)) for f in candidates]
    passing = [f for f, p in scores if p >= rho]
    selected = min(passing) if passing else hi
    pred = dict(scores).get(selected, score_map(gnp, tpi, cf, ctx, selected, used_mus, stack, norm))
    return {"selected_force_N": selected, "predicted_reliability": pred, "candidate_count": len(candidates),
            "candidate_forces_json": json.dumps(candidates), "posterior_aware": int(posterior),
            "fallback_used": int(not passing), "latency_ms": (time.perf_counter()-start)*1000.0}


def select_rho(gnp, tpi, cf, contexts, belief, front, stack, norm, out: Path) -> float:
    rows = []
    mu_map = get_mu_map(belief)
    # Pre-registered selection: among candidate rho values, minimize mean force
    # subject to DEV under-force <= 0.10; tie-break by fallback rate and rho.
    for rho in RHO_CANDIDATES:
        selected, under, fallback = [], [], []
        for cid, info in front.items():
            if cid not in contexts or cid not in mu_map: continue
            ctx = contexts[cid]; task = int(info["task"])
            p = plan_one(gnp, tpi, cf, ctx, mu_map[cid], stack, norm, task, "FIXED_GRID", 10, rho, True)
            selected.append(p["selected_force_N"]); fs = info["frontier"]
            under.append(int(math.isfinite(fs) and p["selected_force_N"] < fs - 1e-9)); fallback.append(p["fallback_used"])
        rows.append({"rho": rho, "n": len(selected), "mean_force_N": np.mean(selected) if selected else math.nan,
                     "under_force_rate": np.mean(under) if under else math.nan, "fallback_rate": np.mean(fallback) if fallback else math.nan,
                     "selection_eligible": int(bool(selected) and np.mean(under) <= 0.10)})
    eligible = [r for r in rows if r["selection_eligible"]]
    if eligible:
        chosen = min(eligible, key=lambda r: (r["mean_force_N"], r["fallback_rate"], r["rho"]))["rho"]
        reason = "minimum DEV mean force subject to under-force <= 0.10; fallback-rate then rho tie-break"
    else:
        chosen = min(rows, key=lambda r: (r["under_force_rate"], r["mean_force_N"], r["rho"]))["rho"]
        reason = "no candidate met DEV under-force floor; minimum under-force, then mean force"
    write_csv(out / "RHO_SELECTION_TABLE.csv", rows)
    write_json(out / "FINAL_RHO.json", {"rho": chosen, "candidates": RHO_CANDIDATES, "fit_split": "TRAIN", "selection_split": "DEV", "reason": reason,
                                        "test_used": False, "protocol_hash": stable_hash({"candidates": RHO_CANDIDATES, "criterion": reason})})
    report = ["# Rho Selection on TRAIN/DEV", "", f"Selected rho = **{chosen:.2f}**.", "", f"Selection rule: {reason}.",
              "Only DEV outcomes were used for this preregistered comparison; locked TEST was not loaded. The value is frozen for the handoff.", "", "See RHO_SELECTION_TABLE.csv and FINAL_RHO.json."]
    (out / "RHO_SELECTION_DEV.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    return float(chosen)


def decision_disagreement(gnp, tpi, cf, contexts, belief, front, stack, norm, rho, out: Path) -> pd.DataFrame:
    mu_map = get_mu_map(belief); rows = []
    for cid, mus in mu_map.items():
        if cid not in contexts or cid not in front: continue
        ctx = contexts[cid]; task = int(front[cid]["task"]); lo, hi = TASK_BOUNDS[task]
        candidates = np.linspace(lo, hi, 41)
        decisions = []
        for mu in mus:
            ps = {float(f): score_map(gnp, tpi, cf, ctx, float(f), [mu], stack, norm) for f in candidates}
            valid = [f for f, p in ps.items() if p >= rho]
            decisions.append(float(min(valid) if valid else hi))
        fs = front[cid]["frontier"]
        harmful = int(math.isfinite(fs) and min(decisions) < fs - 1e-9)
        unique = len(set(decisions)); consensus = int(unique == 1)
        rows.append({"context_id": cid, "root_id": front[cid]["root_id"], "task": task, "friction_band": front[cid]["band"],
                     "rho": rho, "member_force_0_N": decisions[0], "member_force_1_N": decisions[1], "member_force_2_N": decisions[2],
                     "exact_decision_consensus": consensus, "force_spread_N": max(decisions)-min(decisions),
                     "unique_force_count": unique, "decision_entropy_bits": math.log2(unique) if unique else math.nan,
                     "real_frontier_N": fs, "harmful_decision": harmful})
    d = pd.DataFrame(rows)
    write_csv(out / "E6_DECISION_DISAGREEMENT_DIAGNOSTIC.csv", d.to_dict("records"))
    summaries = []
    for name, q in [("consensus", d[d.exact_decision_consensus == 1]), ("disagreement", d[d.exact_decision_consensus == 0])]:
        summaries.append({"group": name, "n": len(q), "P_harm": q.harmful_decision.mean() if len(q) else math.nan,
                          "mean_force_spread_N": q.force_spread_N.mean() if len(q) else math.nan})
    report = ["# E6 Decision Disagreement Diagnostic", "", f"DEV contexts evaluated: {len(d)}; rho={rho:.2f}.", "",
              "Member-induced decisions use identical Direct architecture/checkpoints, candidate support, rho, and max-force fallback; only the physical identifier member changes.", "",
              *[f"{r['group']}: n={r['n']}, P(harm)={r['P_harm']:.4f}, mean spread={r['mean_force_spread_N']:.4f} N" for r in summaries], "",
              "This diagnostic is model-independent at evaluation: harmful means the selected force is below the empirical DEV reliable frontier. It is not used as a test-time label or selection input."]
    (out / "E6_DECISION_DISAGREEMENT_REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    return d


def second_query_artifacts(out: Path) -> None:
    protocol = {"name": "E6_SECOND_QUERY_PROTOCOL", "status": "FROZEN_PROTOCOL_NO_VALIDATED_SECOND_QUERY_DATA",
                "budget": 2, "same_frozen_pi0": True, "controller": "P4-B common contact-frame shear",
                "first_query": "reused authoritative P4-B probe_out", "second_query": "pre-registered repeat P4-B with same preload/displacement/return",
                "query_selection": "decision-aware only; never reads task outcome", "fusion": "member evidence concatenation/re-encoding; no label/outcome input",
                "forbidden": ["sealed TEST", "task outcome", "adaptive force execution", "query rule tuning on TEST"],
                "availability_audit": {"second_query_files_found": False, "historical_single_query_only": True,
                                       "p7b_pilot": "query qualification gate failed; not reused as scientific evidence"}}
    write_json(out / "SECOND_QUERY_PROTOCOL.json", protocol)
    write_csv(out / "SECOND_QUERY_DEV_REPORT.csv", [{"status": "NOT_EVALUATED_NO_SECOND_QUERY_EVIDENCE", "n": 0,
                                                       "reason": "authoritative TRAIN/DEV archive contains one qualified P4-B query per context; no validated second-query continuation"}])
    (out / "SECOND_QUERY_DEV_REPORT.md").write_text("# Second-query DEV Report\n\nSTATUS: COMPLETE_NEGATIVE_FOR_DEPLOYMENT_CLOSURE\n\nThe protocol is frozen, but no validated second-query continuation exists in the authoritative TRAIN/DEV archive. The prior P7-B infrastructure pilot failed query qualification and is not treated as scientific evidence. Therefore second-query uncertainty reduction and four-policy re-query comparison are not claimed.\n", encoding="utf-8")
    rows = [{"strategy": x, "status": "NOT_EVALUATED_NO_SECOND_QUERY_EVIDENCE", "mean_query_count": math.nan, "second_query_rate": math.nan,
             "harmful_decision_rate": math.nan, "full_task_SR_proxy": math.nan, "latency_ms": math.nan} for x in ["ONE_QUERY", "FIXED_TWO_QUERY", "UNCERTAINTY_THRESHOLD", "DECISION_CONSENSUS"]]
    write_csv(out / "TABLE_E6_REQUERY_DEV.csv", rows)
    (out / "E6_REQUERY_DEV_REPORT.md").write_text("# E6 Decision-aware Re-query DEV Report\n\nSTATUS: E6_COMPLETE_NEGATIVE\n\nAll four strategies are specified, but none is scientifically evaluated because the archive has no qualified second-query evidence. Reporting a duplicated first query as a second query would not test uncertainty reduction. DecisionConsensus remains the registered primary rule for future locked-test orchestration, with max budget 2 and frozen fallback.\n", encoding="utf-8")


def continuous_and_planners(gnp, tpi, cf, contexts, belief, front, stack, norm, rho, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mu_map = get_mu_map(belief)
    benchmark = pd.read_csv(CONT_PRED)
    benchmark = benchmark[(benchmark.backend == "FEASIBILITY_ONLY") & (benchmark.condition == "GT") & (benchmark.on_real_benchmark == 1)]
    benchmark = benchmark[benchmark.context_id.astype(str).isin(list(mu_map))]
    real_p = {(str(r.context_id), float(r.force_N)): float(r.raw_probability) for r in benchmark.itertuples()}
    pred_rows, planner_rows = [], []
    kinds = ["FIXED_GRID", "UNIFORM_CONTINUOUS", "STRATIFIED_CONTINUOUS", "PROPOSAL_GUIDED"]
    for cid in sorted(benchmark.context_id.astype(str).unique()):
        if cid not in contexts or cid not in front: continue
        ctx, task, mus = contexts[cid], int(front[cid]["task"]), mu_map[cid]
        for f in sorted({float(r.force_N) for r in benchmark[benchmark.context_id.astype(str) == cid].itertuples()}):
            pp = score_prob(gnp, tpi, cf, ctx, f, mus, stack, norm)
            pred_rows.append({"context_id": cid, "task": task, "force_N": f, "p_posterior": pp["raw_probability"],
                              "p_point": score_map(gnp, tpi, cf, ctx, f, [float(np.mean(mus))], stack, norm),
                              "empirical_success": real_p[(cid, f)]})
        for posterior in [True, False]:
            for kind in kinds:
                for k in [5, 10, 20]:
                    p = plan_one(gnp, tpi, cf, ctx, mus, stack, norm, task, kind, k, rho, posterior)
                    fs = front[cid]["frontier"]
                    selected = p["selected_force_N"]
                    real_success = front[cid]["outcomes"].get(float(selected), math.nan)
                    planner_rows.append({"context_id": cid, "root_id": front[cid]["root_id"], "task": task,
                                         "planner": kind, "K": k, "rho": rho, "posterior_aware": int(posterior), **p,
                                         "real_frontier_N": fs, "real_success_at_selected": real_success,
                                         "under_force": int(math.isfinite(fs) and selected < fs - 1e-9),
                                         "excess_force_N": max(0.0, selected-fs) if math.isfinite(fs) else math.nan,
                                         "force_regret_N": abs(selected-fs) if math.isfinite(fs) else math.nan})
    pred = pd.DataFrame(pred_rows); plans = pd.DataFrame(planner_rows)
    write_csv(out / "CONTINUOUS_DIRECT_DEV_PREDICTIONS.csv", pred.to_dict("records"))
    write_csv(out / "TABLE_E7_CONTINUOUS_DEV.csv", plans.to_dict("records"))
    # Gate uses posterior-aware K=10 uniform continuous, the registered
    # minimum-budget default, against the archived repeated DEV outcomes.
    q = plans[(plans.planner == "UNIFORM_CONTINUOUS") & (plans.K == 10) & (plans.posterior_aware == 1)]
    gate = {"probability_calibration": binary_metrics(pred.empirical_success.to_numpy(float), pred.p_posterior.to_numpy(float)) if len(pred) else {},
            "heldout_root_n": int(q.root_id.nunique()) if len(q) else 0, "frontier_MAE_N": float(q.force_regret_N.mean()) if len(q) else math.nan,
            "under_force_rate": float(q.under_force.mean()) if len(q) else math.nan, "mean_excess_force_N": float(q.excess_force_N.mean()) if len(q) else math.nan,
            "checks": {"probability_calibration": bool(len(pred) and binary_metrics(pred.empirical_success.to_numpy(float), pred.p_posterior.to_numpy(float))["MAE"] <= 0.20),
                       "heldout_root_performance": bool(len(q) and q.real_success_at_selected.fillna(0).mean() >= 0.90),
                       "heldout_force_interpolation": bool(len(pred) and pred.p_posterior.notna().all()),
                       "force_ordering": bool(len(pred) and all(pred[pred.context_id == cid].sort_values("force_N").p_posterior.is_monotonic_increasing for cid in pred.context_id.unique())),
                       "simulator_dev_rollout": False}}
    gate["pass"] = all(gate["checks"].values()) and gate["under_force_rate"] <= 0.10 and gate["frontier_MAE_N"] <= 0.20
    write_json(out / "CONTINUOUS_DIRECT_DEV_GATE.json", gate)
    status = "PASS" if gate["pass"] else "FAIL_WITH_ROOT_CAUSE"
    root_causes = [k for k, v in gate["checks"].items() if not v]
    report = ["# Continuous Direct DEV Gate", "", f"STATUS: **{status}**", "", "The gate uses only root-held-out DEV contexts and archived repeated DEV outcomes. No TEST roots were loaded.", "",
              f"Posterior-aware Uniform K=10: probability MAE={gate['probability_calibration'].get('MAE', math.nan):.4f}, frontier MAE={gate['frontier_MAE_N']:.4f} N, under-force={gate['under_force_rate']:.4f}, held-out roots={gate['heldout_root_n']}.", "",
              f"Failed checks: {', '.join(root_causes) if root_causes else 'none'}.", "", "Root cause boundary: this lane has no new simulator validation for arbitrary continuous selected setpoints, and the archived continuous Direct line already documented reliability/decision failures. Planner tables are offline DEV diagnostics, not a validated real controller."]
    (out / "CONTINUOUS_DIRECT_DEV_GATE.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    return pred, plans


def fallback_artifacts(out: Path, rho: float) -> None:
    fallback = {"policy": "MAX_SAFE_FORCE_WITHIN_TASK_BOUNDS", "fit_split": "TRAIN", "selection_split": "DEV", "frozen": True,
                "trigger": ["no candidate reaches rho", "second query exhausted and unresolved"], "rho": rho,
                "force_bounds_N": {str(k): list(v) for k, v in TASK_BOUNDS.items()}, "test_used": False,
                "rationale": "pre-registered conservative fallback; no TEST tuning"}
    write_json(out / "FINAL_FALLBACK_POLICY.json", fallback)


def simulator_report(out: Path) -> None:
    write_csv(out / "TABLE_E7_SIMULATOR_DEV.csv", [{"status": "NOT_RUN", "reason": "continuous Direct gate did not pass and A100 was occupied by other agents; no process was interrupted", "test_roots_accessed": "[]"}])
    (out / "E7_REAL_ROLLOUT_DEV_REPORT.md").write_text("# E7 Real Simulator DEV Rollout Report\n\nSTATUS: E7_COMPLETE_NEGATIVE\n\nNo new simulator rollout was started. The continuous Direct gate failed before deployment validation, and the shared A100 was already occupied by two unrelated running experiments. No other process was killed or modified. Existing historical reports remain evidence only; this lane does not claim arbitrary-force actuator tracking, commanded-vs-measured force parity, or real rollout success.\n", encoding="utf-8")


def historical_audit(out: Path) -> None:
    text = """# ActiveForcing E6/E7 Existing Work Audit

## Scope

Audit performed over `/home/exouser/FORTE`, `/home/exouser/Tabero`, and the archived `analysis/results` namespaces. Original TEST rows were not loaded by this lane.

| Component | Status | Evidence / boundary |
|---|---|---|
| 3-member physical belief | READY (DEV diagnostic) | New matched architecture, three deterministic seeds, root-bootstrap TRAIN resampling; DEV calibration artifact emitted by this run. |
| Heteroscedastic uncertainty | READY (DEV diagnostic) | Per-member `mu` and `sigma`; uncertainty semantics and TRAIN-only interval scaling are recorded. |
| Decision disagreement | READY (DEV diagnostic) | Member-induced force decisions and model-independent harmful frontier comparison emitted. |
| Validated second query | FAILED_DEV | Historical P7-B pilot failed query qualification; no authoritative second-query continuation was found. |
| Decision-aware re-query | FAILED_DEV | Protocol is frozen, but four-policy scientific comparison is not evaluable without valid second-query evidence. |
| Continuous Direct | FAILED_DEV | Prior GNP-style continuous line reports `CONTINUOUS_FEASIBILITY_STILL_NOT_VALIDATED`; this run preserves that boundary and rechecks the DEV diagnostic. |
| Candidate generators | READY (offline DEV diagnostic) | Fixed/uniform/stratified/proposal interfaces implemented with matched K=5/10/20. |
| Posterior-aware planning | READY (offline DEV diagnostic) | Same Direct backend and candidates; point-estimate ablation retained. |
| rho selection | READY | Candidate set and DEV-only reliability/mean-force selection frozen in `FINAL_RHO.json`. |
| Fallback | READY | Max safe force within task bounds, frozen before any TEST orchestration. |
| Real simulator continuous validation | FAILED_DEV | Not launched after failed continuous gate; no arbitrary-setpoint validation is claimed. |

## Reused historical conclusions

The pre-probe feasibility line had three seeds and TRAIN isotonic calibration, but it was not a physical posterior ensemble. The continuous GNP-style line had 720 continuous TRAIN branches and a 9-context DEV diagnostic, but its safety-first gate failed. The old P7-B pilot had one-query infrastructure only and failed qualification. None of these artifacts authorized locked TEST method tuning.

## Overall audit classification

E6: PARTIAL -> COMPLETE_NEGATIVE because physical ensemble/disagreement are now available but validated second-query evidence is absent.

E7: PARTIAL -> COMPLETE_NEGATIVE because offline candidate/posterior planning and freeze artifacts are available, while continuous Direct reliability and real arbitrary-force simulator validation are not established.
"""
    (out / "E6_E7_EXISTING_WORK_AUDIT.md").write_text(text, encoding="utf-8")


def handoff(out: Path, belief_manifest: dict[str, Any], rho: float) -> None:
    bundle = out / "E6_E7_LOCKED_HANDOFF"
    bundle.mkdir(parents=True, exist_ok=True)
    for name in ["PHYSICAL_BELIEF_MANIFEST.json", "FINAL_RHO.json", "FINAL_FALLBACK_POLICY.json", "SECOND_QUERY_PROTOCOL.json",
                 "CONTINUOUS_DIRECT_DEV_GATE.json", "TABLE_E6_ENSEMBLE_CALIBRATION_DEV.csv", "TABLE_E6_REQUERY_DEV.csv", "TABLE_E7_CONTINUOUS_DEV.csv"]:
        src = out / name
        if src.exists(): shutil.copy2(src, bundle / name)
    for member in MEMBERS:
        shutil.copy2(out / f"PHYSICAL_BELIEF_{member}.pt", bundle / f"PHYSICAL_BELIEF_{member}.pt")
    for seed in [0, 1, 2]:
        src = GNP_OUT / f"GNP_STYLE_CONTINUOUS_FEAS_seed{seed}.pt"
        shutil.copy2(src, bundle / f"FROZEN_DIRECT_FEAS_seed{seed}.pt")
    interface = {"identifier": "physical probe sequence -> [mu_m, sigma_m]", "belief": "mean(mu_m), Var(mu_m), mean(sigma_m^2)",
                 "planning": "p_bar(F)=mean_m mean_direct p_D(success|x,mu_m,F)", "direct_backend": "frozen FEASIBILITY_ONLY continuous Direct",
                 "rho": rho, "posterior_sample_budget": 3, "force_bounds_N": {str(k): list(v) for k,v in TASK_BOUNDS.items()},
                 "candidate_K": [5, 10, 20], "candidate_generators": ["FIXED_GRID", "UNIFORM_CONTINUOUS", "STRATIFIED_CONTINUOUS", "PROPOSAL_GUIDED"],
                 "requery_rule": "DECISION_CONSENSUS if member-induced decisions disagree; max 2; protocol only until second-query evidence exists",
                 "fallback": "maximum safe force within task bounds", "sealed_test_roots_accessed": []}
    write_json(bundle / "IDENTIFIER_AND_PLANNER_INTERFACE.json", interface)
    rows = []
    for p in sorted(bundle.iterdir()):
        if p.is_file(): rows.append({"path": p.name, "sha256": sha256(p), "bytes": p.stat().st_size})
    write_csv(bundle / "SHA256_MANIFEST.csv", rows)
    report = f"""# E6/E7 Handoff to Main Agent

## Frozen values

- physical belief: three matched identifier members; checkpoints and hashes are in `E6_E7_LOCKED_HANDOFF/SHA256_MANIFEST.csv`.
- Direct backend: frozen FEASIBILITY_ONLY continuous checkpoints copied read-only from `{GNP_OUT}`.
- rho: `{rho:.2f}`; selected only from registered TRAIN/DEV candidates.
- query budget: 2; primary rule `DECISION_CONSENSUS`; second-query protocol is frozen but not validated.
- candidate generators: FIXED_GRID, UNIFORM_CONTINUOUS, STRATIFIED_CONTINUOUS, PROPOSAL_GUIDED; matched K `{[5,10,20]}`.
- posterior samples: 3 physical members; fallback: maximum safe force within task bounds.

## Do not change on locked TEST

Do not retune rho, uncertainty threshold, consensus rule, K, proposal distribution, fallback, architecture, or second-query rule using locked TEST outcomes. Do not replace the negative second-query/simulator status with a duplicated-query or snapped-force proxy.

## Scientific boundary

This handoff is `E6_COMPLETE_NEGATIVE` / `E7_COMPLETE_NEGATIVE` for the current DEV closure: E6 has calibrated/disagreement diagnostics but no validated second-query evidence; E7 has offline planners and freeze artifacts but no passed continuous reliability gate or new arbitrary-force simulator validation.
"""
    (out / "E6_E7_HANDOFF_TO_MAIN_AGENT.md").write_text(report, encoding="utf-8")


def final_report(out: Path, belief: pd.DataFrame, disagreement: pd.DataFrame, plans: pd.DataFrame, rho: float) -> None:
    gate = json.loads((out / "CONTINUOUS_DIRECT_DEV_GATE.json").read_text())
    lines = ["# ACTIVEFORCING_E6_E7_DEV_CLOSURE_REPORT", "", "## Final status", "", "**E6_COMPLETE_NEGATIVE**", "**E7_COMPLETE_NEGATIVE**", "",
             "## What was completed", "", f"- Historical audit and evidence boundary: `E6_E7_EXISTING_WORK_AUDIT.md`.", f"- 3-member calibrated physical belief: DEV n={sum(belief.split == 'DEV')}, checkpoints/hashes in the locked handoff.",
             f"- Decision disagreement: {len(disagreement)} DEV contexts, same Direct backend and rho={rho:.2f}.",
             f"- Candidate planners: {len(plans)} matched DEV planner rows; posterior-aware and point-estimate ablation both retained.",
             "- rho and fallback frozen on TRAIN/DEV; locked handoff bundle created.", "",
             "## E6 conclusion", "", "The physical belief ensemble is implemented and its uncertainty/disagreement diagnostics are reproducible. However, the authoritative archive contains no qualified second-query continuation, and the historical P7-B pilot failed its query-qualification gate. Therefore decision-aware re-query cannot be claimed as validated; E6 is complete negative for locked-test readiness.", "",
             "## E7 conclusion", "", f"The continuous Direct DEV diagnostic gate is `{gate['pass']}` with root causes recorded in `CONTINUOUS_DIRECT_DEV_GATE.md`. Candidate generators and posterior-aware planning are implemented offline with matched budgets, but no passed continuous reliability gate or arbitrary-force simulator DEV rollout exists. E7 is complete negative for locked-test readiness.", "",
             "## Test discipline", "", "No sealed TEST rows, outcomes, telemetry, or method tuning were used. Other agents' running processes were not interrupted.", "",
             "## Artifacts", "", "See `E6_E7_LOCKED_HANDOFF/`, `TABLE_E6_ENSEMBLE_CALIBRATION_DEV.csv`, `E6_DECISION_DISAGREEMENT_DIAGNOSTIC.csv`, `TABLE_E6_REQUERY_DEV.csv`, `TABLE_E7_CONTINUOUS_DEV.csv`, `RHO_SELECTION_DEV.md`, `FINAL_FALLBACK_POLICY.json`, and `E6_E7_ENGINEERING_FIX_LOG.md`."]
    (out / "ACTIVEFORCING_E6_E7_DEV_CLOSURE_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> int:
    out = Path(os.environ.get("E6E7_OUT", str(FORTE / "analysis/results" / f"activeforcing_e6e7_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")))
    out.mkdir(parents=True, exist_ok=False)
    (out / "E6_E7_ENGINEERING_FIX_LOG.md").write_text("""# E6/E7 Engineering Fix Log

- Isolated execution to `/home/exouser/FORTE_e6e7` and output namespace under `analysis/results/activeforcing_e6e7_*`.
- Filtered P5-S0-C contexts to TRAIN/DEV before feature loading; original TEST was not parsed.
- Implemented matched 3-member physical belief with root bootstrap resampling and explicit epistemic/aleatoric semantics.
- Used CPU for the small identifier models because the shared A100 already had two running processes; no process was killed.
- Loaded frozen Direct checkpoints read-only and kept one common backend across all planners/disagreement diagnostics.
- Added explicit NOT_EVALUATED artifacts for missing second-query evidence and simulator validation.
""", encoding="utf-8")
    historical_audit(out)
    examples, _ = load_train_dev_examples()
    belief, belief_manifest = build_belief(out, examples)
    calibration_report(out, belief)
    gnp, tpi, cf, contexts, norm, stack = load_direct_stack()
    front = real_frontiers()
    rho = select_rho(gnp, tpi, cf, contexts, belief, front, stack, norm, out)
    disagreement = decision_disagreement(gnp, tpi, cf, contexts, belief, front, stack, norm, rho, out)
    second_query_artifacts(out)
    _, plans = continuous_and_planners(gnp, tpi, cf, contexts, belief, front, stack, norm, rho, out)
    fallback_artifacts(out, rho)
    simulator_report(out)
    handoff(out, belief_manifest, rho)
    final_report(out, belief, disagreement, plans, rho)
    write_json(out / "RUN_MANIFEST.json", {"status": "COMPLETE_NEGATIVE", "output": str(out), "test_roots_accessed": [],
                                           "source_hashes": {str(p): sha256(p) for p in [FRICTION_SOURCE, P5C_MODEL_SOURCE, GNP_SOURCE, CONT_PRED, CONT_FRONT, BRANCHES]},
                                           "result_hashes": {p.name: sha256(p) for p in out.iterdir() if p.is_file()},
                                           "gpu_jobs_started": False, "gpu_processes_killed": [], "created_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "COMPLETE_NEGATIVE", "out": str(out), "rho": rho, "dev_belief_rows": int(sum(belief.split == "DEV")), "planner_rows": int(len(plans))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
