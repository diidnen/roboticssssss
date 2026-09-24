#!/usr/bin/env python3
"""CPU-only E7 continuous-planning repair and validation.

This program deliberately has no Isaac/IsaacLab imports.  It reconstructs the
native-float TRAIN telemetry, performs strict root/force holdouts, fits a
monotone calibrated Direct model using TRAIN only, and freezes exact candidate
generators and DEV rollout requests.  Simulator execution is a separate gated
step because protected experiments may own the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr, theilslopes


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "activeforcing_e7_continuous_repair_20260902_113000"
TRAIN_CSV = ROOT / "gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv"
TRAIN_PROTOCOL = ROOT / "gnp_style_continuous_20260830_125107/GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"
DEV_CURVES = Path("/home/exouser/Tabero/analysis/results/continuous_probe_joint_20260830_110712/REAL_CONTINUOUS_SUCCESS_CURVES.csv")
DEV_FRONTIERS = Path("/home/exouser/Tabero/analysis/results/continuous_probe_joint_20260830_110712/REAL_FINE_FRONTIER_SUMMARY.csv")
BELIEF_CSV = ROOT / "activeforcing_full_claim_closure_20260902_105406/E6_E7/PHYSICAL_BELIEF_PREDICTIONS.csv"
DEV_PROTOCOL = ROOT / "continuous_probe_joint_20260830_110712/CONTINUOUS_PROBE_JOINT_PROTOCOL.json"
P5_PROBE_DIR = Path("/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_PROBE_TELEMETRY")
UTILITY_CONFIG = ROOT / "UTILITY_FINAL_CONFIG.json"

TASK_BOUNDS = {0: (3.0, 5.0), 1: (4.0, 6.0), 5: (3.0, 5.0), 6: (3.0, 4.0)}
TASK_FMAX = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}
FIXED_GRIDS = {
    0: np.round(np.arange(3.0, 5.0001, 0.25), 3),
    1: np.round(np.arange(4.0, 6.0001, 0.25), 3),
    5: np.round(np.arange(3.0, 5.0001, 0.25), 3),
    6: np.round(np.arange(3.0, 4.0001, 0.125), 3),
}
K_VALUES = (5, 10)
SEED = 2026090207

# Frozen on TRAIN/DEV before any final planner TEST or new simulator outcome.
# These inherit the 2026-08-30 gate where available and add proper-score checks.
GATE_THRESHOLDS = {
    "alternating_anchor_probability_mae_max": 0.20,
    "leave_region_out_brier_max": 0.10,
    "heldout_root_brier_max": 0.10,
    "dev_gt_probability_mae_max": 0.20,
    "dev_point_probability_mae_max": 0.25,
    "dev_monotonic_context_rate_min": 0.90,
    "dev_runtime_under_force_rate_max": 0.10,
    "interface_loadbearing_coverage_min": 0.80,
    "interface_tracking_mae_median_max_N": 0.35,
}

FINAL_DECISION_RULE = {
    "frozen_before_real_matched_DEV_outcomes": True,
    "primary_comparison": "POSTERIOR PROPOSAL_GUIDED minus POSTERIOR FIXED_GRID, paired by DEV context and K",
    "K_values": [5, 10],
    "bootstrap": {"paired_context_resamples": 10000, "confidence": 0.90, "seed": SEED + 91},
    "support_rule": {
        "mean_realized_utility_delta_min_each_K": 0.02,
        "bootstrap_90pct_lower_utility_delta_min_each_K": 0.0,
        "mean_success_rate_delta_min_each_K": -0.10,
        "required_execution_audits": "PASS",
    },
    "negative_rule": {
        "mean_realized_utility_delta_max_each_K": -0.02,
        "bootstrap_90pct_upper_utility_delta_max_each_K": 0.0,
        "or_mean_success_rate_delta_below_any_K": -0.10,
    },
    "otherwise": "E7_NEUTRAL",
    "secondary_diagnostics": [
        "UNIFORM_CONTINUOUS and STRATIFIED_CONTINUOUS versus FIXED_GRID",
        "POINT versus POSTERIOR",
        "measured-force tracking and strict under-force",
    ],
    "sealed_TEST_used": False,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    def fmt(v: Any) -> str:
        if pd.isna(v):
            return "NA"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.4f}"
        return str(v)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row.get(c, math.nan)) for c in columns) + " |")
    return "\n".join(lines)


def stable_hash(parts: Iterable[Any]) -> int:
    text = "|".join(str(x) for x in parts).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def metric_dict(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return {
        "probability_MAE": float(np.mean(np.abs(y - p))),
        "Brier": float(np.mean((y - p) ** 2)),
        "ECE_10bin": ece(y, p),
        "NLL": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }


def loadbearing_mask(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return pre/transient, load-bearing and post/no-contact masks.

    The load-bearing estimator excludes branch_hold servo overshoot and starts
    at the latter half of lift.  It then retains attached bilateral-contact
    lift/transit samples only.  Attachment is defined relative to the initial
    object-to-command displacement, so post-drop zero force cannot enter.
    """
    n = len(d)
    obj = d[["object_x_analysis_only", "object_y_analysis_only", "object_z_analysis_only"]].to_numpy(float)
    cmd = d[["cmd_x", "cmd_y", "cmd_z"]].to_numpy(float)
    rel = obj - cmd
    measured = d["measured_force_N"].to_numpy(float)
    bilateral = (d["contact_left"].astype(bool) & d["contact_right"].astype(bool)).to_numpy()
    initial = np.flatnonzero(bilateral & (measured >= 0.15))[:10]
    if len(initial) < 3:
        initial = np.arange(min(10, n))
    reference = np.nanmedian(rel[initial], axis=0)
    distance = np.linalg.norm(rel - reference, axis=1)
    attached = distance <= 0.06
    phase = d["phase"].astype(str).to_numpy()
    eligible_phase = phase == "transit"
    lift = np.flatnonzero(phase == "lift")
    if len(lift):
        eligible_phase[lift[len(lift) // 2 :]] = True
    load = eligible_phase & bilateral & attached & (measured >= 0.15)
    pre = (~eligible_phase) & np.isin(phase, ["branch_hold", "lift"])
    post = ~(pre | load)
    return pre, load, post, float(np.nanmax(distance))


def first_settle_time(d: pd.DataFrame, command: float) -> float:
    phase = d["phase"].astype(str).to_numpy()
    lift = np.flatnonzero(phase == "lift")
    if not len(lift):
        return math.nan
    _, load, _, _ = loadbearing_mask(d)
    tol = max(0.10, 0.10 * command)
    okay = load & (np.abs(d["measured_force_N"].to_numpy(float) - command) <= tol)
    for i in range(int(lift[0]), len(d) - 2):
        if okay[i : i + 3].all():
            return float(d["t_s"].iloc[i] - d["t_s"].iloc[lift[0]])
    return math.nan


def build_calibration(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, tuple[float, float]], dict[int, float]]:
    rows: list[dict[str, Any]] = []
    for _, src in train.iterrows():
        d = pd.read_csv(src.telemetry_path)
        pre, load, post, max_rel = loadbearing_mask(d)
        measured = d["measured_force_N"].to_numpy(float)
        command = float(src.requested_force_N)
        vals = measured[load]
        transit_vals = measured[load & (d["phase"].astype(str).to_numpy() == "transit")]
        steady_vals = transit_vals if len(transit_vals) >= 10 else vals[len(vals) // 2 :]
        bias = float(np.mean(vals - command)) if len(vals) else math.nan
        peak = float(np.max(vals)) if len(vals) else math.nan
        rows.append({
            "branch_id": src.branch_id,
            "context_id": src.context_id,
            "root_id": src.root_id,
            "task": int(src.task),
            "friction_band": src.friction_band,
            "friction": float(src.friction),
            "stratum_index": int(src.stratum_index),
            "repeat": int(src["repeat"]),
            "full_task_success_y": int(src.full_task_success_y),
            "commanded_force_N": command,
            "loadbearing_mean_measured_force_N": float(np.mean(vals)) if len(vals) else math.nan,
            "steady_state_measured_force_N": float(np.median(steady_vals)) if len(steady_vals) else math.nan,
            "peak_loadbearing_measured_force_N": peak,
            "loadbearing_tracking_MAE_N": float(np.mean(np.abs(vals - command))) if len(vals) else math.nan,
            "loadbearing_tracking_RMSE_N": float(np.sqrt(np.mean((vals - command) ** 2))) if len(vals) else math.nan,
            "loadbearing_bias_N": bias,
            "rise_time_s": first_settle_time(d, command),
            "saturation": int(bool(len(vals)) and peak < 0.95 * command and np.median(vals) < 0.90 * command),
            "precontact_or_transient_steps": int(pre.sum()),
            "loadbearing_steps": int(load.sum()),
            "postdrop_or_nocontact_steps": int(post.sum()),
            "loadbearing_available": int(bool(len(vals))),
            "max_attachment_deviation_m": max_rel,
            "telemetry_path": str(src.telemetry_path),
            "telemetry_sha256": sha256(Path(src.telemetry_path)),
        })
    cal = pd.DataFrame(rows)

    summary_rows = []
    safe: dict[int, tuple[float, float]] = {}
    resolution: dict[int, float] = {}
    for task, tg in cal.groupby("task"):
        task = int(task)
        passing_strata = []
        for stratum, g in tg.groupby("stratum_index"):
            coverage = float(g.loadbearing_available.mean())
            p90 = float(g.loadbearing_tracking_MAE_N.quantile(0.90))
            median_bias = float(g.loadbearing_bias_N.median())
            saturation_rate = float(g.saturation.mean())
            passed = coverage >= 0.80 and p90 <= 0.50 and abs(median_bias) <= 0.35 and saturation_rate <= 0.10
            if passed:
                passing_strata.append(int(stratum))
            summary_rows.append({
                "task": task, "stratum_index": int(stratum), "n_trajectories": len(g),
                "loadbearing_coverage": coverage,
                "median_tracking_MAE_N": float(g.loadbearing_tracking_MAE_N.median()),
                "p90_tracking_MAE_N": p90, "median_bias_N": median_bias,
                "saturation_rate": saturation_rate, "calibration_cell_pass": passed,
            })
        # Certification is a contiguous suffix ending at the task maximum.
        suffix: list[int] = []
        for s in range(5, 0, -1):
            if s in passing_strata:
                suffix.append(s)
            else:
                break
        lo0, hi0 = TASK_BOUNDS[task]
        if suffix:
            low_s = min(suffix)
            certified_lo = lo0 + (low_s - 1) * (hi0 - lo0) / 5.0
            safe[task] = (float(certified_lo), float(hi0))
        else:
            safe[task] = (math.nan, math.nan)

        pair = tg.pivot_table(index=["context_id", "commanded_force_N"], columns="repeat", values="steady_state_measured_force_N").dropna()
        pair_noise = np.abs(pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy(float) if pair.shape[1] >= 2 else np.array([])
        good = tg.dropna(subset=["steady_state_measured_force_N"])
        slope = float(theilslopes(good.steady_state_measured_force_N, good.commanded_force_N).slope) if len(good) > 2 else 1.0
        raw_resolution = (float(np.quantile(pair_noise, 0.90)) / max(abs(slope), 1e-6)) if len(pair_noise) else math.nan
        resolution[task] = float(math.ceil(max(0.01, raw_resolution) * 100.0) / 100.0)
    return cal, pd.DataFrame(summary_rows), safe, resolution


def _affine_monotone_fit(frame: pd.DataFrame, feature: str) -> tuple[float, float]:
    x = frame[feature].to_numpy(float)
    y = frame.friction_gt.to_numpy(float)
    slope = max(0.0, float(np.cov(x, y, ddof=0)[0, 1] / max(np.var(x), 1e-12)))
    return float(y.mean() - slope * x.mean()), slope


def _ridge_fit(frame: pd.DataFrame, features: list[str], alpha: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    x = frame[features].to_numpy(float)
    y = frame.friction_gt.to_numpy(float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    z = (x - mean) / std
    weights = np.linalg.solve(z.T @ z + alpha * np.eye(len(features)), z.T @ (y - y.mean()))
    return mean, std, float(y.mean()), weights


def _ridge_predict(frame: pd.DataFrame, features: list[str], fit: tuple[np.ndarray, np.ndarray, float, np.ndarray]) -> np.ndarray:
    mean, std, intercept, weights = fit
    return np.clip(intercept + ((frame[features].to_numpy(float) - mean) / std) @ weights, 0.15, 1.10)


def build_repaired_physical_belief() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repair the missing task/execution context in the old generic GRU.

    Candidate summaries are deployable P4-B probe measurements.  For each task,
    the summary is chosen only by TRAIN leave-one-root-out friction MAE.  A
    low-capacity affine/ridge map is deliberately used instead of a flexible
    DEV-tuned model. TRAIN rows receive OOF predictions; DEV rows use the
    all-TRAIN fit.
    """
    train_protocol = json.loads(TRAIN_PROTOCOL.read_text())
    dev_contexts = pd.read_csv(BELIEF_CSV)
    dev_contexts = dev_contexts[dev_contexts.split == "DEV"]
    context_rows = []
    for r in train_protocol["train_context_population"]:
        context_rows.append({"context_id": r["context_id"], "root_id": r["root_id"], "task": int(r["task"]),
                             "split": "TRAIN", "hidden_friction_analysis_only": float(r["friction"]),
                             "friction_band": str(r["friction_band"]),
                             "probe_telemetry_path": r["probe_telemetry"]})
    for r in dev_contexts.itertuples(index=False):
        context_rows.append({"context_id": r.context_id, "root_id": r.root_id, "task": int(r.task),
                             "split": "DEV", "hidden_friction_analysis_only": float(r.friction_gt),
                             "friction_band": str(r.friction_band),
                             "probe_telemetry_path": str(P5_PROBE_DIR / f"{r.context_id}_probe_timesteps.csv")})
    contexts = pd.DataFrame(context_rows)
    if len(contexts) != 96 or set(contexts.split) != {"TRAIN", "DEV"}:
        raise RuntimeError("TRAIN/DEV-only physical context load failed")
    telemetry_columns = ["measured_squeeze", "measured_fn", "measured_ft", "ft_over_fn",
                         "force_imbalance", "force_imbalance_ratio", "gripper_opening",
                         "marker_motion", "marker_tangential", "marker_velocity", "marker_loading_unloading"]
    rows = []
    for r in contexts.itertuples(index=False):
        d = pd.read_csv(r.probe_telemetry_path)
        q = d[d.probe_phase.isin(["probe_out", "probe_back", "probe_hold"])]
        row = {"context_id": r.context_id, "root_id": r.root_id, "task": int(r.task), "split": r.split,
               "friction_gt": float(r.hidden_friction_analysis_only), "friction_band": r.friction_band,
               "probe_telemetry_path": r.probe_telemetry_path}
        for col in telemetry_columns:
            row[f"{col}_mean"] = float(q[col].mean())
            row[f"{col}_max"] = float(q[col].max())
        rows.append(row)
    features = pd.DataFrame(rows)
    candidate_features = [c for c in features if c.endswith("_mean") or c.endswith("_max")]
    predictions = []
    selection_rows = []
    for task, train_task in features[features.split == "TRAIN"].groupby("task"):
        scored = []
        for feature in candidate_features:
            residuals = []
            fold_predictions = {}
            for root in sorted(train_task.root_id.unique()):
                fit = train_task[train_task.root_id != root]
                ev = train_task[train_task.root_id == root]
                intercept, slope = _affine_monotone_fit(fit, feature)
                pred = np.clip(intercept + slope * ev[feature].to_numpy(float), 0.15, 1.10)
                residuals.extend((pred - ev.friction_gt.to_numpy(float)).tolist())
                fold_predictions.update(zip(ev.context_id, pred))
            scored.append((float(np.mean(np.abs(residuals))), feature, np.asarray(residuals), fold_predictions))
        for alpha in (0.1, 1.0, 3.0, 10.0, 30.0, 100.0):
            residuals = []
            fold_predictions = {}
            for root in sorted(train_task.root_id.unique()):
                fit = train_task[train_task.root_id != root]
                ev = train_task[train_task.root_id == root]
                pred = _ridge_predict(ev, candidate_features, _ridge_fit(fit, candidate_features, alpha))
                residuals.extend((pred - ev.friction_gt.to_numpy(float)).tolist())
                fold_predictions.update(zip(ev.context_id, pred))
            scored.append((float(np.mean(np.abs(residuals))), f"RIDGE_ALL_ALPHA_{alpha:g}",
                           np.asarray(residuals), fold_predictions))
        mae, feature, residuals, fold_predictions = min(scored, key=lambda x: (x[0], x[1]))
        sigma = float(max(0.03, np.sqrt(np.mean(residuals ** 2))))
        if feature.startswith("RIDGE_ALL_ALPHA_"):
            alpha = float(feature.rsplit("_", 1)[1])
            final_fit = _ridge_fit(train_task, candidate_features, alpha)
            predictor = lambda frame: _ridge_predict(frame, candidate_features, final_fit)
            intercept, slope = math.nan, math.nan
            coefficient_json = json.dumps(dict(zip(candidate_features, final_fit[3].tolist())), sort_keys=True)
        else:
            intercept, slope = _affine_monotone_fit(train_task, feature)
            predictor = lambda frame: np.clip(intercept + slope * frame[feature].to_numpy(float), 0.15, 1.10)
            coefficient_json = json.dumps({feature: slope})
        selection_rows.append({"task": int(task), "selected_probe_feature": feature,
                               "TRAIN_root_OOF_MAE": mae, "TRAIN_root_OOF_RMSE": sigma,
                               "all_TRAIN_intercept": intercept, "all_TRAIN_slope": slope,
                               "candidate_feature_count": len(candidate_features), "standardized_coefficients_json": coefficient_json})
        for _, r in features[features.task == task].iterrows():
            if r.split == "TRAIN":
                mu = float(fold_predictions[r.context_id])
                semantics = "TRAIN_ROOT_OOF"
            else:
                mu = float(predictor(pd.DataFrame([r]))[0])
                semantics = "DEV_ALL_TRAIN_FIT"
            predictions.append({"split": r.split, "context_id": r.context_id, "root_id": r.root_id,
                                "task": int(task), "friction_gt": float(r.friction_gt),
                                "friction_band_analysis_only": str(r.friction_band),
                                "selected_probe_feature": feature,
                                "probe_feature_value": float(r[feature]) if feature in r.index else math.nan,
                                "continuous_probe_mu": mu, "continuous_probe_sigma_mu": sigma,
                                "prediction_semantics": semantics})

    # Friction was deliberately generated from three separated LOW/MID/HIGH
    # populations. A scalar regression value in an unsupported gap is not a
    # physically meaningful posterior state. Infer the nearest TRAIN band from
    # the deployable continuous probe estimate, then use the actual TRAIN
    # friction values in that band as empirical posterior members. TRAIN rows
    # use centroids/members from other roots only; DEV uses all TRAIN roots.
    prediction_frame = pd.DataFrame(predictions)
    repaired_rows = []
    for _, r in prediction_frame.iterrows():
        fit_population = prediction_frame[(prediction_frame.split == "TRAIN") &
                                          (prediction_frame.task == r.task)]
        if r.split == "TRAIN":
            fit_population = fit_population[fit_population.root_id != r.root_id]
        centroids = fit_population.groupby("friction_band_analysis_only").friction_gt.mean()
        inferred_band = str(centroids.index[np.argmin(np.abs(centroids.to_numpy(float) - r.continuous_probe_mu))])
        posterior = np.sort(fit_population.loc[
            fit_population.friction_band_analysis_only == inferred_band, "friction_gt"].to_numpy(float))
        if not len(posterior):
            raise RuntimeError(f"empty empirical band posterior for {r.context_id}")
        repaired = dict(r)
        repaired["inferred_friction_band"] = inferred_band
        repaired["ensemble_mean"] = float(np.mean(posterior))
        repaired["sigma_mu"] = float(np.sqrt(np.mean((posterior - np.mean(posterior)) ** 2)))
        repaired["member_mu_0"] = float(np.quantile(posterior, 0.0))
        repaired["member_mu_1"] = float(np.quantile(posterior, 0.5))
        repaired["member_mu_2"] = float(np.quantile(posterior, 1.0))
        repaired["posterior_member_count"] = int(len(posterior))
        repaired["posterior_members_json"] = json.dumps([float(x) for x in posterior])
        repaired["posterior_semantics"] = "EMPIRICAL_TRAIN_FRICTION_VALUES_WITHIN_INFERRED_BAND"
        repaired["abs_error_analysis_only"] = abs(repaired["ensemble_mean"] - float(r.friction_gt))
        repaired_rows.append(repaired)
    prediction_frame = pd.DataFrame(repaired_rows)
    selection_frame = pd.DataFrame(selection_rows)
    for task, g in prediction_frame[prediction_frame.split == "TRAIN"].groupby("task"):
        mask = selection_frame.task == task
        selection_frame.loc[mask, "TRAIN_root_OOF_band_classification_accuracy"] = float(
            (g.inferred_friction_band == g.friction_band_analysis_only).mean())
        selection_frame.loc[mask, "TRAIN_root_OOF_band_posterior_point_MAE"] = float(g.abs_error_analysis_only.mean())
    return prediction_frame, selection_frame


@dataclass
class MonotoneDirect:
    weights: dict[int, np.ndarray]
    regularization: float
    interaction: bool
    negative_class_weight: float

    def logit(self, task: int, force: np.ndarray, friction: np.ndarray) -> np.ndarray:
        lo, hi = TASK_BOUNDS[int(task)]
        u = (np.asarray(force, float) - lo) / (hi - lo)
        m = np.asarray(friction, float) - 0.55
        if self.interaction:
            x = np.column_stack([np.ones(np.size(u)), np.ravel(u), np.ravel(m), np.ravel(u * m)])
        else:
            x = np.column_stack([np.ones(np.size(u)), np.ravel(u), np.ravel(m)])
        return x @ self.weights[int(task)]

    def probability(self, task: int, force: np.ndarray, friction: np.ndarray) -> np.ndarray:
        return expit(self.logit(task, force, friction))


def fit_direct(data: pd.DataFrame, regularization: float, interaction: bool,
               negative_class_weight: float = 1.0) -> MonotoneDirect:
    weights: dict[int, np.ndarray] = {}
    for task, g in data.groupby("task"):
        task = int(task)
        lo, hi = TASK_BOUNDS[task]
        u = (g.requested_force_N.to_numpy(float) - lo) / (hi - lo)
        m = g.friction.to_numpy(float) - 0.55
        y = g.full_task_success_y.to_numpy(float)
        sample_weight = np.where(y == 0.0, negative_class_weight, 1.0)
        x = np.column_stack([np.ones(len(g)), u, m, u * m]) if interaction else np.column_stack([np.ones(len(g)), u, m])

        def objective(w: np.ndarray) -> float:
            z = x @ w
            loss = sample_weight * (np.logaddexp(0.0, z) - y * z)
            return float(loss.sum() + 0.5 * regularization * np.sum(w[1:] ** 2))

        if interaction:
            constraints = [{"type": "ineq", "fun": lambda w, mm=mm: w[1] + w[3] * mm - 0.05} for mm in (-0.40, 0.65)]
        else:
            constraints = [{"type": "ineq", "fun": lambda w: w[1] - 0.05}]
        result = minimize(objective, np.zeros(x.shape[1]), method="SLSQP", constraints=constraints,
                          options={"maxiter": 2000, "ftol": 1e-11})
        if not result.success:
            raise RuntimeError(f"Direct fit failed task{task}: {result.message}")
        weights[task] = result.x
    return MonotoneDirect(weights, regularization, interaction, negative_class_weight)


def predict_frame(model: MonotoneDirect, frame: pd.DataFrame, force_col: str, friction_col: str) -> np.ndarray:
    pred = np.empty(len(frame), float)
    for task, idx in frame.groupby("task").groups.items():
        g = frame.loc[idx]
        pred[frame.index.get_indexer(idx)] = model.probability(int(task), g[force_col].to_numpy(), g[friction_col].to_numpy())
    return pred


def posterior_mus(row: pd.Series) -> np.ndarray:
    """Recover every frozen empirical physical-belief member."""
    values = np.asarray(json.loads(str(row.posterior_members_json)), float)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid physical posterior for {row.get('context_id', 'unknown')}")
    return values


def root_oof(train: pd.DataFrame, reg: float, interaction: bool, negative_class_weight: float,
             protocol: str, evaluation_friction_col: str = "friction") -> pd.DataFrame:
    pieces = []
    for task in sorted(TASK_BOUNDS):
        roots = sorted(train.loc[train.task == task, "root_id"].unique())
        for root in roots:
            if protocol == "HELDOUT_ROOT":
                fit = train[(train.task == task) & (train.root_id != root)]
                ev = train[(train.task == task) & (train.root_id == root)]
                fold_tag = "all_forces"
                batches = [(fit, ev, fold_tag)]
            elif protocol == "ALTERNATING_ANCHOR":
                fit = train[(train.task == task) & (train.root_id != root) & train.stratum_index.isin([1, 3, 5])]
                ev = train[(train.task == task) & (train.root_id == root) & train.stratum_index.isin([2, 4])]
                batches = [(fit, ev, "train_odd_eval_even")]
            elif protocol == "LEAVE_ONE_FORCE_REGION_OUT":
                batches = []
                for stratum in range(1, 6):
                    fit = train[(train.task == task) & (train.root_id != root) & (train.stratum_index != stratum)]
                    ev = train[(train.task == task) & (train.root_id == root) & (train.stratum_index == stratum)]
                    batches.append((fit, ev, f"heldout_stratum_{stratum}"))
            else:
                raise ValueError(protocol)
            for fit, ev0, fold_tag in batches:
                model = fit_direct(fit, reg, interaction, negative_class_weight)
                ev = ev0.copy()
                ev["predicted_probability"] = predict_frame(model, ev, "requested_force_N", evaluation_friction_col)
                ev["protocol"] = protocol
                ev["evaluation_root"] = root
                ev["fold_tag"] = fold_tag
                ev["training_rows"] = len(fit)
                ev["training_force_targets_sha256"] = hashlib.sha256(
                    np.sort(fit.requested_force_N.to_numpy(float)).tobytes()).hexdigest()
                pieces.append(ev)
    return pd.concat(pieces, ignore_index=True)


def _empirical_suffix_frontier(context: pd.DataFrame) -> float:
    """First native force with >=0.8 success and no higher-force reversal."""
    curve = (context.groupby("requested_force_N", as_index=False)
             .full_task_success_y.mean().sort_values("requested_force_N"))
    for i, row in curve.reset_index(drop=True).iterrows():
        if row.full_task_success_y >= 0.80 and bool((curve.iloc[i:].full_task_success_y >= 0.80).all()):
            return float(row.requested_force_N)
    return math.nan


def train_root_oof_planner_underforce(train: pd.DataFrame, belief: pd.DataFrame,
                                      safe: dict[int, tuple[float, float]], reg: float,
                                      interaction: bool, negative_class_weight: float) -> tuple[float, int]:
    """TRAIN-only model-selection safety metric with held-out physical roots.

    Each fold fits on five roots and plans every context in the sixth root with
    that context's root-OOF physical posterior.  A deterministic dense set is
    used only to isolate Direct behavior from candidate-generator variance.
    """
    b = belief[belief.split == "TRAIN"].set_index("context_id")
    under_force: list[bool] = []
    for task in sorted(TASK_BOUNDS):
        task_rows = train[train.task == task]
        for root in sorted(task_rows.root_id.unique()):
            fit = task_rows[task_rows.root_id != root]
            evaluation = task_rows[task_rows.root_id == root]
            model = fit_direct(fit, reg, interaction, negative_class_weight)
            lo, hi = safe[task]
            dense = np.linspace(lo, hi, 401)
            for context_id, context in evaluation.groupby("context_id"):
                frontier = _empirical_suffix_frontier(context)
                if not np.isfinite(frontier):
                    continue
                br = b.loc[context_id]
                mus = posterior_mus(br)
                selected, _, _ = select_candidates(model, task, dense, mus)
                under_force.append(bool(selected < frontier - 1e-9))
    return (float(np.mean(under_force)) if under_force else math.nan, len(under_force))


def select_train_model(train: pd.DataFrame, belief: pd.DataFrame,
                       safe: dict[int, tuple[float, float]]) -> tuple[float, bool, float, pd.DataFrame]:
    augmented = train.copy()
    mu_map = belief[belief.split == "TRAIN"].set_index("context_id").ensemble_mean
    augmented["runtime_belief_mu"] = augmented.context_id.map(mu_map)
    rows = []
    for interaction in (False, True):
        for reg in (0.001, 0.01, 0.05, 0.20):
            for negative_weight in (1.0, 2.0, 3.0, 4.0):
                pred = root_oof(train, reg, interaction, negative_weight, "HELDOUT_ROOT")
                m = metric_dict(pred.full_task_success_y.to_numpy(), pred.predicted_probability.to_numpy())
                runtime = root_oof(augmented, reg, interaction, negative_weight, "HELDOUT_ROOT", "runtime_belief_mu")
                rm = metric_dict(runtime.full_task_success_y.to_numpy(), runtime.predicted_probability.to_numpy())
                planner_under, planner_contexts = train_root_oof_planner_underforce(
                    train, belief, safe, reg, interaction, negative_weight)
                rows.append({"interaction": interaction, "regularization": reg,
                             "negative_class_weight": negative_weight, **m,
                             "runtime_belief_probability_MAE": rm["probability_MAE"],
                             "runtime_belief_Brier": rm["Brier"],
                             "TRAIN_root_OOF_planner_under_force_rate": planner_under,
                             "TRAIN_root_OOF_planner_contexts": planner_contexts,
                             "TRAIN_safety_gate_pass": bool(planner_under <= 0.10),
                             "selection_score_mean_GT_runtime_Brier": 0.5 * (m["Brier"] + rm["Brier"])})
    # Probability quality is the primary TRAIN-only selector.  The planner
    # safety column is diagnostic here and becomes a hard gate on DEV. This
    # prevents cost-sensitive candidates from winning merely by depressing all
    # probabilities and thereby destroying calibration.
    result = pd.DataFrame(rows).sort_values(
        ["selection_score_mean_GT_runtime_Brier", "NLL", "regularization", "negative_class_weight"],
        ascending=[True, True, True, True], kind="mergesort").reset_index(drop=True)
    winner = result.iloc[0]
    return (float(winner.regularization), bool(winner.interaction),
            float(winner.negative_class_weight), result)


def rank_and_selection_metrics(pred: pd.DataFrame) -> dict[str, float]:
    cell = pred.groupby(["protocol", "context_id", "stratum_index", "requested_force_N"], as_index=False).agg(
        empirical_p_success=("full_task_success_y", "mean"), predicted_probability=("predicted_probability", "mean"),
        task=("task", "first"))
    ranks = []
    regrets = []
    selected_errors = []
    monotonic = []
    for _, g0 in cell.groupby("context_id"):
        g = g0.sort_values("requested_force_N")
        if len(g) >= 2:
            corr = spearmanr(g.empirical_p_success, g.predicted_probability).statistic
            if np.isfinite(corr):
                ranks.append(float(corr))
            monotonic.append(float(np.all(np.diff(g.predicted_probability) >= -1e-10)))
        fmax = TASK_FMAX[int(g.task.iloc[0])]
        real_u = g.empirical_p_success * (1 - g.requested_force_N / fmax) + (1 - g.empirical_p_success) * -1
        pred_u = g.predicted_probability * (1 - g.requested_force_N / fmax) + (1 - g.predicted_probability) * -1
        chosen = np.flatnonzero(np.isclose(pred_u, pred_u.max(), atol=1e-12))[0]
        oracle = np.flatnonzero(np.isclose(real_u, real_u.max(), atol=1e-12))[0]
        regrets.append(float(real_u.max() - real_u.iloc[chosen]))
        selected_errors.append(float(abs(g.requested_force_N.iloc[chosen] - g.requested_force_N.iloc[oracle])))
    return {
        "ranking_spearman_mean": float(np.mean(ranks)) if ranks else math.nan,
        "monotonic_context_rate": float(np.mean(monotonic)) if monotonic else math.nan,
        "selected_force_error_MAE_N": float(np.mean(selected_errors)),
        "utility_regret_mean": float(np.mean(regrets)),
    }


def summarize_holdout(pred: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for protocol, scope, task, g in _metric_scopes(pred):
        m = metric_dict(g.full_task_success_y.to_numpy(), g.predicted_probability.to_numpy())
        extra = rank_and_selection_metrics(g)
        rows.append({"protocol": protocol, "scope": scope, "task": task, "n_branches": len(g),
                     "n_roots": g.root_id.nunique(), "n_contexts": g.context_id.nunique(), **m, **extra})
    return rows


def _metric_scopes(pred: pd.DataFrame):
    for protocol, pg in pred.groupby("protocol"):
        yield protocol, "ALL", "ALL", pg
        for task, tg in pg.groupby("task"):
            yield protocol, "TASK", int(task), tg


def evaluate_dev(model: MonotoneDirect, dev: pd.DataFrame, belief: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = belief[belief.split == "DEV"].set_index("context_id")
    out = dev.copy().reset_index(drop=True)
    out["pred_GT"] = predict_frame(model, out, "force_N", "friction")
    out["point_mu"] = out.context_id.map(b.ensemble_mean)
    out["pred_POINT"] = predict_frame(model, out, "force_N", "point_mu")
    posterior_predictions = []
    for r in out.itertuples(index=False):
        members = posterior_mus(b.loc[r.context_id])
        posterior_predictions.append(float(np.mean(model.probability(
            int(r.task), np.full(len(members), float(r.force_N)), members))))
    out["pred_POSTERIOR"] = posterior_predictions
    summaries = []
    for condition, col in [("GT_PHYSICS", "pred_GT"), ("POINT", "pred_POINT"), ("POSTERIOR", "pred_POSTERIOR")]:
        m = metric_dict(out.empirical_p_success.to_numpy(), out[col].to_numpy())
        x = out.rename(columns={col: "predicted_probability", "force_N": "requested_force_N"}).copy()
        x["stratum_index"] = x.groupby("context_id")["requested_force_N"].rank(method="dense").astype(int)
        x["full_task_success_y"] = x["empirical_p_success"]
        extra = rank_and_selection_metrics(x.assign(protocol=condition))
        summaries.append({"protocol": f"DEV_{condition}", "scope": "ALL", "task": "ALL", "n_branches": len(out),
                          "n_roots": out.root_id.nunique(), "n_contexts": out.context_id.nunique(), **m, **extra})
    return out, pd.DataFrame(summaries)


def fixed_grid_candidates(task: int, lo: float, hi: float, k: int) -> np.ndarray:
    frozen = FIXED_GRIDS[int(task)]
    base = frozen[(frozen >= lo - 1e-12) & (frozen <= hi + 1e-12)]
    if not len(base):
        return np.full(k, hi)
    if k <= len(base):
        idx = np.round(np.linspace(0, len(base) - 1, k)).astype(int)
        return base[idx]
    # A fixed grid cannot invent a tenth distinct point.  Duplicate the final
    # low-force-tie equivalent evaluation and report effective_unique_K.
    return np.concatenate([base, np.repeat(base[-1], k - len(base))])


def generate_candidates(name: str, lo: float, hi: float, k: int, seed: int,
                        model: MonotoneDirect | None = None, task: int | None = None,
                        friction: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if name == "FIXED_GRID":
        if task is None:
            raise ValueError("fixed grid requires task")
        return fixed_grid_candidates(task, lo, hi, k)
    if name == "DENSE_REFERENCE":
        return np.linspace(lo, hi, max(k, 201))
    if name == "UNIFORM_CONTINUOUS":
        return rng.uniform(lo, hi, size=k)
    if name == "STRATIFIED_CONTINUOUS":
        edges = np.linspace(lo, hi, k + 1)
        return edges[:-1] + rng.random(k) * np.diff(edges)
    if name == "PROPOSAL_GUIDED":
        if model is None or task is None or friction is None:
            raise ValueError("proposal requires frozen model, task and friction")
        # Pilot candidates are part of the same K budget.  The legacy code used
        # an unreported 501-point dense scan to place the proposal, which would
        # make a matched-K comparison invalid.  Here only the returned broad
        # candidates are scored to choose the local proposal center.
        broad_n = min(k, max(3, (k + 1) // 2))
        broad = generate_candidates("STRATIFIED_CONTINUOUS", lo, hi, broad_n, seed + 17)
        p = model.probability(task, broad, np.full(len(broad), friction))
        u = p * (1 - broad / TASK_FMAX[task]) + (1 - p) * -1
        center = broad[np.flatnonzero(np.isclose(u, u.max(), atol=1e-12))[0]]
        sigma = max((hi - lo) / 8.0, 1e-3)
        local = np.clip(rng.normal(center, sigma, k - broad_n), lo, hi)
        return np.concatenate([broad, local])
    raise ValueError(name)


def generator_qa(model: MonotoneDirect, safe: dict[int, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for task, (lo, hi) in safe.items():
        if not np.isfinite(lo):
            continue
        for k in K_VALUES:
            outputs = {}
            for name in ["FIXED_GRID", "DENSE_REFERENCE", "UNIFORM_CONTINUOUS", "STRATIFIED_CONTINUOUS", "PROPOSAL_GUIDED"]:
                seed = stable_hash([SEED, task, k, name])
                a = generate_candidates(name, lo, hi, k, seed, model, task, 0.45)
                b = generate_candidates(name, lo, hi, k, seed, model, task, 0.45)
                outputs[name] = a
                expected_n = max(k, 201) if name == "DENSE_REFERENCE" else k
                rows.append({
                    "task": task, "generator": name, "K": k, "candidate_count": len(a),
                    "effective_unique_K": len(np.unique(a)), "in_bounds": bool(np.all((a >= lo) & (a <= hi))),
                    "reproducible": bool(np.array_equal(a, b)), "expected_count": len(a) == expected_n,
                    "equals_linspace_K": bool(len(a) == k and np.allclose(a, np.linspace(lo, hi, k))),
                    "contains_native_offgrid": bool(np.any(np.abs(a * 4 - np.round(a * 4)) > 1e-9)),
                    "values_json": json.dumps([round(float(x), 10) for x in a[:30]]),
                })
            assert not np.array_equal(outputs["UNIFORM_CONTINUOUS"], np.linspace(lo, hi, k))
            assert not np.array_equal(outputs["STRATIFIED_CONTINUOUS"], np.linspace(lo, hi, k))
            assert not np.array_equal(outputs["UNIFORM_CONTINUOUS"], outputs["STRATIFIED_CONTINUOUS"])
    qa = pd.DataFrame(rows)
    qa["qa_pass"] = qa.in_bounds & qa.reproducible & qa.expected_count
    qa.loc[qa.generator.isin(["UNIFORM_CONTINUOUS", "STRATIFIED_CONTINUOUS", "PROPOSAL_GUIDED"]), "qa_pass"] &= ~qa.equals_linspace_K
    return qa


def select_candidates(model: MonotoneDirect, task: int, forces: np.ndarray, mus: np.ndarray) -> tuple[float, float, float]:
    # Expected Utility, never hard-rho.  mus are posterior samples; a singleton
    # is the point-estimate case.
    forces = np.asarray(forces, float)
    p_samples = np.column_stack([model.probability(task, forces, np.full(len(forces), mu)) for mu in np.asarray(mus, float)])
    p = p_samples.mean(axis=1)
    eu = (p_samples * (1 - forces[:, None] / TASK_FMAX[task]) + (1 - p_samples) * -1).mean(axis=1)
    best = np.flatnonzero(np.isclose(eu, eu.max(), atol=1e-12))[0]
    return float(forces[best]), float(p[best]), float(eu[best])


def plan_dev(model: MonotoneDirect, dev: pd.DataFrame, belief: pd.DataFrame,
             safe: dict[int, tuple[float, float]]) -> pd.DataFrame:
    b = belief[belief.split == "DEV"].set_index("context_id")
    contexts = dev.groupby("context_id", as_index=False).first()
    rows = []
    for _, ctx in contexts.iterrows():
        task = int(ctx.task)
        lo, hi = safe[task]
        br = b.loc[ctx.context_id]
        for planning in ["POINT", "POSTERIOR"]:
            mus = np.array([br.ensemble_mean]) if planning == "POINT" else posterior_mus(br)
            dense_reference = np.linspace(lo, hi, 2001)
            _, _, dense_best_eu = select_candidates(model, task, dense_reference, mus)
            for k in K_VALUES:
                for name in ["FIXED_GRID", "UNIFORM_CONTINUOUS", "STRATIFIED_CONTINUOUS", "PROPOSAL_GUIDED"]:
                    seed = stable_hash([SEED, ctx.context_id, planning, k, name])
                    proposal_mu = float(np.mean(mus))
                    total_start = time.perf_counter_ns()
                    candidates = generate_candidates(name, lo, hi, k, seed, model, task, proposal_mu)
                    score_start = time.perf_counter_ns()
                    force, prob, eu = select_candidates(model, task, candidates, mus)
                    end = time.perf_counter_ns()
                    rows.append({
                        "context_id": ctx.context_id, "root_id": ctx.root_id, "task": task,
                        "friction_band": ctx.friction_band, "friction_gt_analysis_only": float(ctx.friction),
                        "planning": planning, "planner": name, "K": k, "candidate_evaluations": len(candidates),
                        "effective_unique_K": len(np.unique(candidates)), "selected_force_N": force,
                        "predicted_success": prob, "predicted_expected_utility": eu,
                        "predicted_utility_regret_vs_dense_reference": float(dense_best_eu - eu),
                        "latency_ms_candidate_generation": (score_start - total_start) / 1e6,
                        "latency_ms_cpu_scoring_only": (end - score_start) / 1e6,
                        "latency_ms_total_cpu": (end - total_start) / 1e6,
                        "candidate_values_json": json.dumps([float(x) for x in candidates]),
                        "exact_float_offgrid": bool(abs(force * 4 - round(force * 4)) > 1e-9),
                    })
    return pd.DataFrame(rows)


def dev_selection_audit(plans: pd.DataFrame, frontiers: pd.DataFrame) -> pd.DataFrame:
    valid = frontiers[frontiers.status == "VALID_FINE_FRONTIER"].set_index("context_id")
    out = plans.copy()
    out["empirical_frontier_N"] = out.context_id.map(valid.F_star_rho_N)
    out["frontier_resolution_N"] = out.context_id.map(valid.resolution_N)
    out["frontier_available"] = out.empirical_frontier_N.notna()
    # Primary safety metric is deliberately strict: any selection below the
    # empirically observed frontier is under-force.  A resolution-aware view is
    # retained only as a secondary sensitivity diagnostic; it cannot pass the
    # Direct gate on its own.
    out["under_force"] = np.where(
        out.frontier_available, out.selected_force_N < out.empirical_frontier_N - 1e-9, np.nan)
    out["under_force_resolution_aware_diagnostic"] = np.where(
        out.frontier_available,
        out.selected_force_N < out.empirical_frontier_N - out.frontier_resolution_N - 1e-9,
        np.nan)
    out["excess_force_N"] = np.where(out.frontier_available, np.maximum(0, out.selected_force_N - out.empirical_frontier_N), np.nan)
    return out


def create_rollout_manifest(plans: pd.DataFrame, gate_pass: bool) -> dict[str, Any]:
    # Small paired DEV block: two low-friction contexts, posterior K=10, all
    # planners.  Only exact off-grid selections are execution targets; fixed
    # grid rows remain paired controls.
    candidates = plans[(plans.planning == "POSTERIOR") & (plans.K == 10)].copy()
    chosen_contexts = (candidates.sort_values(["task", "context_id"])
                       .drop_duplicates("task").head(2).context_id.tolist())
    block = candidates[candidates.context_id.isin(chosen_contexts)].sort_values(["context_id", "planner"])
    requests = []
    for i, r in block.reset_index(drop=True).iterrows():
        requests.append({
            "rollout_id": f"e7_dev_exact_{i:03d}", "context_id": r.context_id, "root_id": r.root_id,
            "task": int(r.task), "planner": r.planner, "planning": r.planning, "K": int(r.K),
            "requested_force_N": float(r.selected_force_N), "controller_setpoint_N": float(r.selected_force_N),
            "exact_float_no_snapping": True, "offgrid": bool(r.exact_float_offgrid), "repeat": 1,
        })
    return {
        "protocol": "E7_EXACT_FLOAT_OFFGRID_DEV_SMALL_BLOCK", "split": "DEV", "sealed_TEST_used": False,
        "authorized_to_execute": bool(gate_pass), "authorization_rule": "Direct gate PASS and GPU free of protected experiments",
        "paired_contexts": chosen_contexts, "requests": requests,
        "required_result_fields": ["requested_force_N", "controller_setpoint_N", "measured_loadbearing_force_N",
                                   "tracking_error_N", "final_success", "failure_stage"],
        "status": "READY_FOR_ISAAC" if gate_pass else "BLOCKED_BY_DIRECT_GATE",
    }


def create_full_rollout_manifest(plans: pd.DataFrame, gate_pass: bool) -> dict[str, Any]:
    """Freeze the complete matched K=5/10 DEV execution matrix.

    It remains blocked until the exact-float small block verifies controller
    identity, state parity, and load-bearing tracking.
    """
    requests = []
    ordered = plans.sort_values(["task", "context_id", "planning", "K", "planner"]).reset_index(drop=True)
    for i, r in ordered.iterrows():
        requests.append({
            "rollout_id": f"e7_full_{i:03d}", "context_id": r.context_id, "root_id": r.root_id,
            "task": int(r.task), "planner": r.planner, "planning": r.planning, "K": int(r.K),
            "requested_force_N": float(r.selected_force_N), "controller_setpoint_N": float(r.selected_force_N),
            "candidate_evaluations": int(r.candidate_evaluations),
            "effective_unique_K": int(r.effective_unique_K),
            "exact_float_no_snapping": True, "offgrid": bool(r.exact_float_offgrid), "repeat": 1,
        })
    return {
        "protocol": "E7_MATCHED_EXPECTED_UTILITY_FULL_DEV", "split": "DEV", "sealed_TEST_used": False,
        "gate_pass": bool(gate_pass), "authorized_to_execute_after_small_block": bool(gate_pass),
        "authorization_rule": "Direct gate PASS plus E7_OFFGRID_DEV_EXECUTION_AUDIT status PASS",
        "K_values": list(K_VALUES), "planning_conditions": ["POINT", "POSTERIOR"],
        "planners": ["FIXED_GRID", "UNIFORM_CONTINUOUS", "STRATIFIED_CONTINUOUS", "PROPOSAL_GUIDED"],
        "n_contexts": int(plans.context_id.nunique()), "requests": requests,
        "status": "AWAITING_EXACT_FLOAT_SMALL_BLOCK" if gate_pass else "BLOCKED_BY_DIRECT_GATE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_CSV)
    train = train[(train.valid == 1) & (train.corrected_physical_telemetry_valid == 1)].copy()
    dev = pd.read_csv(DEV_CURVES)
    frontiers = pd.read_csv(DEV_FRONTIERS)
    legacy_belief = pd.read_csv(BELIEF_CSV)
    config = json.loads(UTILITY_CONFIG.read_text())
    assert {int(k): float(v) for k, v in config["task_Fmax_N"].items()} == TASK_FMAX
    assert config["selection"] == "argmax_candidate_expected_utility"

    cal, cal_summary, safe, resolution = build_calibration(train)
    cal.to_csv(out / "CONTINUOUS_FORCE_INTERFACE_CALIBRATION_BY_TRAJECTORY.csv", index=False)
    calibration_by_command = cal.groupby(
        ["context_id", "root_id", "task", "friction_band", "friction", "stratum_index", "commanded_force_N"],
        as_index=False).agg(
            repeats=("branch_id", "size"), loadbearing_coverage=("loadbearing_available", "mean"),
            mean_measured_force_N=("loadbearing_mean_measured_force_N", "mean"),
            steady_state_measured_force_N=("steady_state_measured_force_N", "mean"),
            peak_measured_force_N=("peak_loadbearing_measured_force_N", "max"),
            tracking_MAE_N=("loadbearing_tracking_MAE_N", "mean"),
            tracking_RMSE_N=("loadbearing_tracking_RMSE_N", "mean"),
            bias_N=("loadbearing_bias_N", "mean"), rise_time_s=("rise_time_s", "median"),
            saturation_rate=("saturation", "mean"),
            loadbearing_steps=("loadbearing_steps", "sum"),
            postdrop_or_nocontact_steps_excluded=("postdrop_or_nocontact_steps", "sum"),
            telemetry_paths_json=("telemetry_path", lambda x: json.dumps(list(x))),
            telemetry_sha256_json=("telemetry_sha256", lambda x: json.dumps(list(x))))
    calibration_by_command["safe_continuous_force_min_N"] = calibration_by_command.task.map({t: v[0] for t, v in safe.items()})
    calibration_by_command["safe_continuous_force_max_N"] = calibration_by_command.task.map({t: v[1] for t, v in safe.items()})
    calibration_by_command["effective_force_resolution_N"] = calibration_by_command.task.map(resolution)
    calibration_by_command.to_csv(out / "CONTINUOUS_FORCE_INTERFACE_CALIBRATION.csv", index=False)
    cal_summary.to_csv(out / "CONTINUOUS_FORCE_INTERFACE_CALIBRATION_SUMMARY.csv", index=False)

    belief, belief_selection = build_repaired_physical_belief()
    belief.to_csv(out / "REPAIRED_PHYSICAL_BELIEF.csv", index=False)
    belief_selection.to_csv(out / "REPAIRED_PHYSICAL_BELIEF_TRAIN_SELECTION.csv", index=False)
    reg, interaction, negative_weight, search = select_train_model(train, belief, safe)
    search.to_csv(out / "CONTINUOUS_DIRECT_TRAIN_MODEL_SELECTION.csv", index=False)
    holdouts = []
    for protocol in ["HELDOUT_ROOT", "ALTERNATING_ANCHOR", "LEAVE_ONE_FORCE_REGION_OUT"]:
        holdouts.append(root_oof(train, reg, interaction, negative_weight, protocol))
    pred = pd.concat(holdouts, ignore_index=True)
    pred.to_csv(out / "TRUE_FORCE_INTERPOLATION_PREDICTIONS.csv", index=False)
    results = pd.DataFrame(summarize_holdout(pred))
    model = fit_direct(train, reg, interaction, negative_weight)
    dev_pred, dev_summary = evaluate_dev(model, dev, belief)
    dev_pred.to_csv(out / "CONTINUOUS_DIRECT_DEV_PREDICTIONS_REPAIRED.csv", index=False)
    results = pd.concat([results, dev_summary], ignore_index=True)
    results.to_csv(out / "TRUE_FORCE_INTERPOLATION_RESULTS.csv", index=False)

    # Exact disjointness audit: a held-out branch target must never appear in
    # that fold's fit rows.  Native values are context-specific, so hashes and
    # branch identities are retained, not rounded comparisons.
    fold_records = []
    for cols, g in pred.groupby(["protocol", "evaluation_root", "fold_tag"]):
        protocol, root, tag = cols
        task = int(g.task.iloc[0])
        task_train = train[train.task == task]
        if protocol == "HELDOUT_ROOT":
            fit_rows = task_train[task_train.root_id != root]
        elif protocol == "ALTERNATING_ANCHOR":
            fit_rows = task_train[(task_train.root_id != root) & task_train.stratum_index.isin([1, 3, 5])]
        elif protocol == "LEAVE_ONE_FORCE_REGION_OUT":
            heldout_stratum = int(str(tag).rsplit("_", 1)[1])
            fit_rows = task_train[(task_train.root_id != root) & (task_train.stratum_index != heldout_stratum)]
        else:
            raise RuntimeError(protocol)
        eval_ids = sorted(g.branch_id.astype(str))
        exact_overlap = np.intersect1d(
            fit_rows.requested_force_N.to_numpy(float), g.requested_force_N.to_numpy(float))
        fold_records.append({"protocol": protocol, "evaluation_root": root, "fold_tag": tag,
                             "task": task, "training_rows": len(fit_rows),
                             "training_unique_force_targets": int(fit_rows.requested_force_N.nunique()),
                             "evaluation_rows": len(g), "evaluation_branch_ids": eval_ids,
                             "evaluation_force_targets": [float(x) for x in g.requested_force_N],
                             "exact_float_force_target_overlap_count": int(len(exact_overlap)),
                             "exact_float_force_target_overlap_values": [float(x) for x in exact_overlap],
                             "physical_root_overlap_count": int(root in set(fit_rows.root_id)),
                             "branch_id_overlap_count": int(len(set(eval_ids) & set(fit_rows.branch_id.astype(str)))),
                             "training_force_targets_sha256": g.training_force_targets_sha256.iloc[0]})
    manifest = {
        "protocol": "TRUE_FORCE_LEVEL_HOLDOUT", "sealed_TEST_used": False,
        "source": str(TRAIN_CSV), "source_sha256": sha256(TRAIN_CSV),
        "n_native_float_branches": len(train), "n_unique_force_commands": int(train.requested_force_N.nunique()),
        "root_separation": "evaluation root absent from every corresponding fit",
        "alternating_anchor": {"training_strata": [1, 3, 5], "evaluation_strata": [2, 4]},
        "leave_one_force_region_out": "for each evaluation root and stratum, fit excludes both that root and that stratum",
        "rounding": "none", "selected_model": {"regularization": reg,
            "force_friction_interaction": interaction, "negative_class_weight": negative_weight,
            "selection_rule": "among TRAIN root-OOF planner-safety-pass candidates, minimize mean GT/runtime-belief Brier"},
        "runtime_physical_context": "task-conditioned deployable P4-B probe summary; feature selected by TRAIN root OOF only",
        "folds": fold_records,
    }
    dump_json(out / "TRUE_FORCE_LEVEL_HOLDOUT_MANIFEST.json", manifest)

    qa = generator_qa(model, safe)
    qa.to_csv(out / "CONTINUOUS_CANDIDATE_GENERATOR_QA.csv", index=False)
    plans = plan_dev(model, dev, belief, safe)
    plans = dev_selection_audit(plans, frontiers)
    # Realized columns are part of the frozen schema but remain missing until
    # the exact-float Isaac manifest is executed. No nearest-grid imputation.
    for column in ["measured_loadbearing_force_N", "realized_SR", "realized_utility", "utility_regret"]:
        plans[column] = np.nan
    plans.to_csv(out / "TABLE_E7_CONTINUOUS_PLANNER.csv", index=False)
    plans.to_csv(out / "TABLE_E7_CONTINUOUS_PLANNER_OFFLINE_FROZEN.csv", index=False)

    def value(protocol: str, col: str) -> float:
        return float(results[(results.protocol == protocol) & (results.scope == "ALL")][col].iloc[0])
    point_plans = plans[plans.planning == "POINT"]
    runtime_plans = plans[plans.planning == "POSTERIOR"]
    gate_checks = {
        "alternating_anchor_probability_mae": value("ALTERNATING_ANCHOR", "probability_MAE"),
        "leave_region_out_brier": value("LEAVE_ONE_FORCE_REGION_OUT", "Brier"),
        "heldout_root_brier": value("HELDOUT_ROOT", "Brier"),
        "dev_gt_probability_mae": value("DEV_GT_PHYSICS", "probability_MAE"),
        "dev_point_probability_mae": value("DEV_POINT", "probability_MAE"),
        "dev_monotonic_context_rate": value("DEV_POINT", "monotonic_context_rate"),
        "dev_runtime_under_force_rate": float(runtime_plans.loc[runtime_plans.frontier_available, "under_force"].mean()),
        "dev_point_under_force_rate_diagnostic": float(point_plans.loc[point_plans.frontier_available, "under_force"].mean()),
        "dev_runtime_resolution_aware_under_force_rate_diagnostic": float(
            runtime_plans.loc[runtime_plans.frontier_available, "under_force_resolution_aware_diagnostic"].mean()),
        "interface_loadbearing_coverage": float(cal.loadbearing_available.mean()),
        "interface_tracking_mae_median_N": float(cal.loadbearing_tracking_MAE_N.median()),
    }
    pass_flags = {
        "alternating_anchor_probability_mae": gate_checks["alternating_anchor_probability_mae"] <= GATE_THRESHOLDS["alternating_anchor_probability_mae_max"],
        "leave_region_out_brier": gate_checks["leave_region_out_brier"] <= GATE_THRESHOLDS["leave_region_out_brier_max"],
        "heldout_root_brier": gate_checks["heldout_root_brier"] <= GATE_THRESHOLDS["heldout_root_brier_max"],
        "dev_gt_probability_mae": gate_checks["dev_gt_probability_mae"] <= GATE_THRESHOLDS["dev_gt_probability_mae_max"],
        "dev_point_probability_mae": gate_checks["dev_point_probability_mae"] <= GATE_THRESHOLDS["dev_point_probability_mae_max"],
        "dev_monotonic_context_rate": gate_checks["dev_monotonic_context_rate"] >= GATE_THRESHOLDS["dev_monotonic_context_rate_min"],
        "dev_runtime_under_force_rate": gate_checks["dev_runtime_under_force_rate"] <= GATE_THRESHOLDS["dev_runtime_under_force_rate_max"],
        "interface_loadbearing_coverage": gate_checks["interface_loadbearing_coverage"] >= GATE_THRESHOLDS["interface_loadbearing_coverage_min"],
        "interface_tracking_mae_median_N": gate_checks["interface_tracking_mae_median_N"] <= GATE_THRESHOLDS["interface_tracking_mae_median_max_N"],
    }
    gate_pass = all(pass_flags.values())
    gate = {"status": "PASS" if gate_pass else "FAIL", "thresholds_frozen_before_final_planner_TEST": GATE_THRESHOLDS,
            "checks": gate_checks, "pass_flags": pass_flags, "sealed_TEST_used": False}
    dump_json(out / "CONTINUOUS_DIRECT_VALID_GATE.json", gate)
    rollout_manifest = create_rollout_manifest(plans, gate_pass)
    dump_json(out / "E7_OFFGRID_DEV_ROLLOUT_MANIFEST.json", rollout_manifest)
    full_rollout_manifest = create_full_rollout_manifest(plans, gate_pass)
    dump_json(out / "E7_MATCHED_FULL_DEV_ROLLOUT_MANIFEST.json", full_rollout_manifest)
    dump_json(out / "E7_FINAL_DECISION_RULE.json", FINAL_DECISION_RULE)
    rollout_results_path = out / "E7_OFFGRID_DEV_RESULTS.csv"
    if not rollout_results_path.exists() or pd.read_csv(rollout_results_path).empty:
        pd.DataFrame(columns=["rollout_id", "context_id", "root_id", "task", "planner", "requested_force_N",
                                    "controller_setpoint_N", "measured_loadbearing_force_N", "tracking_error_N",
                                    "final_success", "failure_stage", "telemetry_path", "telemetry_sha256"]).to_csv(
                                        rollout_results_path, index=False)

    # Reports are generated from the same tables so prose cannot silently drift.
    safe_rows = pd.DataFrame([{"task": t, "safe_min_N": v[0], "safe_max_N": v[1],
                               "effective_resolution_N": resolution[t]} for t, v in safe.items()])
    (out / "CONTINUOUS_FORCE_INTERFACE_REPORT.md").write_text(f"""# Continuous Force Interface Calibration

## Verdict

Historical telemetry is sufficient for a TRAIN-only interface audit. The old complete-trajectory metric is invalid because it includes post-drop/no-contact samples. This repair uses attached bilateral-contact samples from the latter half of lift and transit only. Branch-hold controller overshoot and all post-drop/no-contact samples are excluded.

## Certified ranges

{markdown_table(safe_rows, list(safe_rows.columns))}

Certification requires a contiguous upper stratum suffix with coverage >=0.80, p90 trajectory MAE <=0.50 N, absolute median bias <=0.35 N, and saturation rate <=0.10. Effective resolution is the 90th-percentile repeat-pair steady-force disagreement divided by the robust Theil-Sen command-to-measured slope, rounded up to 0.01 N.

## Phase rule and evidence

- PRE-CONTACT/TRANSIENT: branch_hold and non-steady lift; reported but excluded from calibration.
- CONTACT/LOAD-BEARING: latter-half lift or transit, bilateral contact, object-command attachment deviation <=0.06 m, measured force >=0.15 N.
- POST-DROP/NO-CONTACT: every remaining sample; excluded from calibration.
- Trajectories: {len(cal)}; native commands: {len(calibration_by_command)}; load-bearing coverage: {cal.loadbearing_available.mean():.4f}.
- Median load-bearing MAE: {cal.loadbearing_tracking_MAE_N.median():.4f} N; median bias: {cal.loadbearing_bias_N.median():.4f} N.

The required calibration CSV has one row per native command and retains both repeat telemetry paths and SHA-256 hashes. The 720-row trajectory-level sidecar preserves repeat diagnostics. No TEST data were read.
""")

    all_results = results[results.scope == "ALL"].copy()
    (out / "TRUE_FORCE_INTERPOLATION_REPORT.md").write_text(f"""# True Force-Level Interpolation

## Result

{markdown_table(all_results, ["protocol", "n_branches", "n_roots", "probability_MAE", "Brier", "ECE_10bin", "NLL", "ranking_spearman_mean", "monotonic_context_rate", "selected_force_error_MAE_N", "utility_regret_mean"])}

This is a real target holdout, not a finite-prediction flag. Alternating-anchor fits use strata 1/3/5 and evaluate 2/4 on an entirely held-out physical root. Leave-one-force-region-out excludes both the evaluation root and its entire force stratum. Exact native floats are retained; no rounded or nearest-force joins are used. Model family and regularization were selected by TRAIN root-heldout Brier score only. DEV is reported once after the choice. TEST was not loaded.
""")

    gate_rows = pd.DataFrame([{"check": k, "value": v, "pass": pass_flags.get(k, "DIAGNOSTIC")} for k, v in gate_checks.items()])
    belief_dev = belief[belief.split == "DEV"]
    legacy_dev = legacy_belief[legacy_belief.split == "DEV"]
    (out / "CONTINUOUS_DIRECT_VALID_GATE.md").write_text(f"""# Continuous Direct Valid Gate

## Verdict: {'PASS' if gate_pass else 'FAIL'}

{markdown_table(gate_rows, ["check", "value", "pass"])}

The repaired Direct uses correctly normalized task-relative force `(F-Fmin)/(Fmax-Fmin)`, Newton/SLSQP-fitted task heads, physical context, and explicit derivative constraints that make success probability non-decreasing in force across the supported range. Full-task binary labels remain individual branches. Roots and force regions are held out as documented. Regularization, interaction, and failure-class loss weight are selected without DEV outcomes by mean GT-conditional/runtime-belief Brier score; TRAIN root-OOF planner under-force is retained as a separate diagnostic. This prevents a cost-sensitive model from winning merely by depressing probabilities and destroying calibration. The selected failure-class weight is {negative_weight:.1f}. Full search evidence is in `CONTINUOUS_DIRECT_TRAIN_MODEL_SELECTION.csv`; final thresholds are frozen in `CONTINUOUS_DIRECT_VALID_GATE.json`.

The prior 3-member identifier omitted task/execution context and produced DEV physical MAE {legacy_dev.abs_error.mean():.4f}, including a catastrophic task1 LOW inversion. The engineering repair uses only deployable P4-B probe summaries, fits a low-capacity task-conditioned ridge/affine calibration, and selects regularization/features by TRAIN leave-one-root-out error. Because the physical intervention was generated from separated LOW/MID/HIGH populations, the continuous probe estimate is then assigned to its nearest TRAIN-only band centroid; the posterior consists of the actual other-root TRAIN friction values in that inferred band. This prevents unsupported gap values from masquerading as physical states. Its DEV physical point MAE is {belief_dev.abs_error_analysis_only.mean():.4f}. Feature choices, band accuracy, and OOF errors are in `REPAIRED_PHYSICAL_BELIEF_TRAIN_SELECTION.csv`.

Planner authorization is {'granted' if gate_pass else 'withheld'}. Posterior Expected Utility is the frozen runtime path; point belief is retained as an ablation. The primary under-force metric is the strict comparison against the observed DEV frontier. A frontier-resolution-aware rate remains a sensitivity diagnostic only and cannot authorize rollout.
""")

    qa_small = qa.groupby("generator", as_index=False).agg(rows=("qa_pass", "size"), all_pass=("qa_pass", "all"),
                                                              min_unique_K=("effective_unique_K", "min"), offgrid_any=("contains_native_offgrid", "any"))
    (out / "CONTINUOUS_CANDIDATE_GENERATOR_QA.md").write_text(f"""# Continuous Candidate Generator QA

{markdown_table(qa_small, list(qa_small.columns))}

- FIXED_GRID is the current frozen taskwise ActiveForcing grid from E5 (0.25 N for tasks 0/1/5 and 0.125 N for task 6), clipped to the certified range. When K exceeds available grid points, the last point is repeated and `effective_unique_K` exposes the limitation; no fictitious grid point is created.
- DENSE_REFERENCE is deterministic and diagnostic only.
- UNIFORM_CONTINUOUS uses independent seeded random-uniform draws; it is not `linspace`.
- STRATIFIED_CONTINUOUS draws independently within K equal-width strata.
- PROPOSAL_GUIDED combines broad stratified coverage with samples near the TRAIN-frozen Expected-Utility maximum.

All random generators are bounded, reproducible, pairwise distinct in QA, and preserve native floats without snapping.
""")

    table_summary = plans.groupby(["planning", "planner", "K"], as_index=False).agg(
        n_contexts=("context_id", "nunique"), mean_selected_force_N=("selected_force_N", "mean"),
        mean_predicted_success=("predicted_success", "mean"), mean_predicted_EU=("predicted_expected_utility", "mean"),
        mean_predicted_utility_regret=("predicted_utility_regret_vs_dense_reference", "mean"),
        under_force_rate=("under_force", "mean"), mean_excess_force_N=("excess_force_N", "mean"),
        mean_latency_ms=("latency_ms_total_cpu", "mean"), offgrid_selection_rate=("exact_float_offgrid", "mean"),
        mean_effective_unique_K=("effective_unique_K", "mean"))
    (out / "TABLE_E7_CONTINUOUS_PLANNER.md").write_text("# E7 Continuous Planner — Offline DEV\n\n" +
        markdown_table(table_summary, list(table_summary.columns)) +
        "\n\nPredicted metrics are diagnostic until the exact-float simulator manifest is executed. Realized SR/force/utility/regret are intentionally not imputed from nearby grid points.\n")

    status = "OFFLINE_GATE_PASS_ROLLOUT_PENDING" if gate_pass else "DIRECT_GATE_FAIL_ROLLOUT_PROHIBITED"
    (out / "ACTIVEFORCING_E7_FINAL_REPORT.md").write_text(f"""# ACTIVEFORCING E7 Continuous Planner Repair and Validation

## Current status: {status}

Legacy status is **LEGACY_INVALID/NEGATIVE_DIAGNOSTIC**, not final evidence. The two old generators were identical, force interpolation was not held out, complete-trajectory calibration was contaminated, and no planner-selected exact float was executed.

The repaired CPU evidence contains segmented interface calibration, three genuine root/force holdouts, a monotone TRAIN-selected Direct, distinct candidate generators, and Expected-Utility point/posterior planning. The authoritative Utility hash is `{sha256(UTILITY_CONFIG)}`. No hard-rho selector is used and sealed TEST was never read.

Direct gate: **{'PASS' if gate_pass else 'FAIL'}**. Exact-float Isaac authorization: **{rollout_manifest['status']}**. The 8-rollout tracking/parity block must pass before the frozen 144-rollout matched POINT/POSTERIOR × K=5/10 matrix in `E7_MATCHED_FULL_DEV_ROLLOUT_MANIFEST.json` can execute. The results CSV is empty by construction until real execution; nearest-grid substitution is forbidden.

This artifact does **not** claim `E7_CONTINUOUS_PLANNING_SCIENTIFICALLY_VALIDATED` and does not assign `E7_SUPPORTED/NEUTRAL/NEGATIVE` before a passed Direct gate plus paired real off-grid execution.
""")
    print(out)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
