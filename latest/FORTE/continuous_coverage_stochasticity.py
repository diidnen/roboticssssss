#!/usr/bin/env python3
"""Frozen forensic for continuous Feasibility-only supervision quality."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREV = Path("/home/exouser/Tabero/analysis/results/gnp_style_continuous_20260830_125107")
DEV = Path("/home/exouser/Tabero/analysis/results/continuous_probe_joint_20260830_110712")
TRAIN_BRANCHES = PREV / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv"
TRAIN_FORCES = PREV / "CONTINUOUS_FORCE_SAMPLE_MANIFEST.csv"
DEV_REPEATS = DEV / "REAL_CONTINUOUS_REPEAT_MANIFEST.csv"
DEV_CURVES = DEV / "REAL_CONTINUOUS_SUCCESS_CURVES.csv"
DEV_FRONTIERS = DEV / "REAL_FINE_FRONTIER_SUMMARY.csv"
OLD_PREDICTIONS = PREV / "CONTINUOUS_DEV_FROZEN_PREDICTIONS.csv"
OLD_DECISIONS = PREV / "CONTINUOUS_DEV_FRONTIER_METRICS.csv"
OLD_SELECTED = PREV / "SELECTED_CONTINUOUS_BACKEND.json"
OLD_PROTOCOL = PREV / "GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"
OLD_CKPT_MANIFEST = PREV / "GNP_STYLE_CHECKPOINT_MANIFEST.json"

RHO = 0.80
GRID_STEP_N = 0.05
FORCE_SUPPORT = {0: (3.0, 5.0), 5: (3.0, 5.0), 1: (4.0, 6.0), 6: (3.0, 4.0)}
OLD_COARSE_BOUNDARIES = {0: [3.0, 3.5, 4.0, 4.5, 5.0], 5: [3.0, 3.5, 4.0, 4.5, 5.0],
                         1: [4.0, 4.5, 5.0, 5.5, 6.0], 6: [3.0, 3.5, 4.0]}
CURRENT_MODEL_MAE = 0.24491496104839813
BOOTSTRAP_SEED = 2026083033
BOOTSTRAPS = 10000

# Frozen before outcome analysis.
FORENSIC_THRESHOLDS = {
    "repeat_noise_comparable_fraction_of_model_MAE_min": 0.75,
    "train_ambiguous_1of2_fraction_min": 0.10,
    "coverage_abs_spearman_error_vs_nearest_distance_min": 0.35,
    "coverage_sparse_minus_dense_error_min": 0.05,
    "dev_boundary_stochastic_fraction_min": 1.0 / 3.0,
    "dev_mean_bernoulli_variance_min": 0.10,
    "model_error_to_m2_noise_ratio_min": 1.50,
}
GT_GATE = {
    "valid_real_frontier_coverage_min": 0.80,
    "under_force_rate_max": 0.10,
    "frontier_MAE_max_N": 0.20,
    "probability_MAE_max": 0.20,
    "systematic_nonmonotonic_context_rate_max": 0.10,
}
PROBE_GATE = {
    "run_only_if_GT_gate_passes": True,
    "primary_condition": "point estimate mu_hat",
    "comparison": "strict TRAIN-only no-probe friction prior",
    "rho": RHO,
    "force_grid_step_N": GRID_STEP_N,
}


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    fields = fields or ["status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(TRAIN_BRANCHES)
    force = pd.read_csv(TRAIN_FORCES)
    devr = pd.read_csv(DEV_REPEATS)
    devc = pd.read_csv(DEV_CURVES)
    ck = json.loads(OLD_CKPT_MANIFEST.read_text())
    feas_ckpts = [r for r in ck["checkpoints"] if r["backend"] == "FEASIBILITY_ONLY"]
    if len(train) != 720 or train.groupby(["context_id", "requested_force_N"]).size().nunique() != 1:
        raise RuntimeError("unexpected TRAIN continuous grain")
    if len(force) != 360 or len(devr) != 135 or len(devc) != 27:
        raise RuntimeError("authoritative input shape mismatch")
    if len(feas_ckpts) != 3:
        raise RuntimeError("expected three frozen Feas-only checkpoints")
    sources = [TRAIN_BRANCHES, TRAIN_FORCES, DEV_REPEATS, DEV_CURVES, DEV_FRONTIERS,
               OLD_PREDICTIONS, OLD_DECISIONS, OLD_SELECTED, OLD_PROTOCOL, OLD_CKPT_MANIFEST]
    payload = {
        "status": "FROZEN_BEFORE_OUTCOME_ANALYSIS",
        "scientific_goal": "attribute remaining continuous Feasibility-only error to repeat noise, coverage, real stochasticity, or model error",
        "claim_boundary": {"current_object_task_distribution_only": True, "cross_object": False,
                           "unseen_task": False, "original_TEST_loaded": False, "fresh_E2E": False},
        "backend": "GNP_STYLE_CONTINUOUS_FEAS_ONLY",
        "joint_reopened": False,
        "authoritative_input_sha256": {str(p): sha256(p) for p in sources},
        "train_population": {"contexts": int(train.context_id.nunique()), "cells": int(force.shape[0]),
                             "existing_repeats_per_cell": 2, "branches": len(train)},
        "dev_population": {"contexts": int(devc.context_id.nunique()), "cells": len(devc),
                           "repeats_per_cell": 5, "branches": len(devr)},
        "frozen_feasibility_checkpoints": [{"seed": int(r["seed"]), "path": r["checkpoint"], "sha256": r["sha256"]} for r in feas_ckpts],
        "analysis_metrics": {
            "dev_cell": ["k/5", "empirical_p", "Bernoulli_variance", "entropy_bits", "Wilson_95_interval"],
            "exact_subsampling": ["expected_MAE", "RMSE", "P(reliability_class_mismatch)"],
            "coverage_attribution": ["nearest_task_train_force_distance", "local_task_force_density", "DEV_stochasticity", "near_real_frontier", "friction_band", "task"],
            "current_model_MAE": CURRENT_MODEL_MAE,
        },
        "subsampling_procedure": {"reference": "observed p_5", "sizes": [1, 2, 3, 4],
                                  "enumeration": "all index combinations without replacement within each 5-repeat cell",
                                  "reliability_rule": "p >= 0.80"},
        "forensic_thresholds": FORENSIC_THRESHOLDS,
        "classification_precedence": [
            "MIXED_COVERAGE_AND_STOCHASTICITY when repeat/coverage material AND stochasticity material",
            "TRAIN_REPEAT_NOISE_DOMINATES when repeat material only",
            "CONTINUOUS_TRAIN_COVERAGE_DOMINATES when coverage material only",
            "REAL_FRONTIER_STOCHASTICITY_DOMINATES when stochasticity material only",
            "MODEL_ERROR_DOMINATES when model/noise ratio threshold passes and neither data nor stochasticity criterion passes",
            "INSUFFICIENT_VALID_EVIDENCE otherwise",
        ],
        "targeted_collection_rule": {
            "eligible_classifications": ["TRAIN_REPEAT_NOISE_DOMINATES", "CONTINUOUS_TRAIN_COVERAGE_DOMINATES", "MIXED_COVERAGE_AND_STOCHASTICITY"],
            "selection_uses_TRAIN_only": True,
            "cell_rules": ["existing outcome is 1/2", "immediately adjacent to lower 0/2 -> higher 2/2 transition", "nearest sampled force on either side of that transition"],
            "new_repeats_per_selected_cell": 3,
            "no_new_forces": True,
            "scientific_failures_retried": False,
        },
        "retraining_rule": {"architecture": "exact GNP_STYLE_CONTINUOUS_FEAS", "optimizer": "AdamW lr=8e-4 wd=1e-4",
                            "epochs": 80, "seeds": [0, 1, 2], "batch_size": 64, "loss": "individual-branch BCEWithLogits",
                            "strict_preprobe": True, "train_only_normalization": True, "raw_ensemble_primary": True,
                            "no_DEV_tuning": True},
        "GT_gate": GT_GATE,
        "Probe_gate": PROBE_GATE,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAPS, "unit": "context"},
    }
    payload["protocol_payload_sha256"] = stable_hash(payload)
    path = out / "CONTINUOUS_COVERAGE_STOCHASTICITY_PROTOCOL.json"
    write_json(path, payload)
    (out / "CONTINUOUS_COVERAGE_STOCHASTICITY_PROTOCOL.sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "out": str(out), "protocol_file_sha256": sha256(path),
                      "train_cells": len(force), "dev_cells": len(devc), "feas_checkpoints": len(feas_ckpts)}, indent=2))


def exact_repeat_noise(outcomes: list[int], m: int) -> dict[str, float]:
    p5 = sum(outcomes) / 5
    vals = []
    mismatches = []
    for idx in itertools.combinations(range(5), m):
        pm = sum(outcomes[i] for i in idx) / m
        vals.append(pm - p5)
        mismatches.append(int((pm >= RHO) != (p5 >= RHO)))
    arr = np.asarray(vals, float)
    return {"combinations": len(vals), "expected_MAE": float(np.mean(np.abs(arr))),
            "RMSE": float(np.sqrt(np.mean(arr * arr))), "reliability_misclassification_probability": float(np.mean(mismatches))}


def analyze(out: Path) -> None:
    protocol = json.loads((out / "CONTINUOUS_COVERAGE_STOCHASTICITY_PROTOCOL.json").read_text())
    if protocol["status"] != "FROZEN_BEFORE_OUTCOME_ANALYSIS":
        raise RuntimeError("protocol not frozen")
    train = pd.read_csv(TRAIN_BRANCHES)
    forces = pd.read_csv(TRAIN_FORCES)
    devr = pd.read_csv(DEV_REPEATS)
    curves = pd.read_csv(DEV_CURVES)
    fronts = pd.read_csv(DEV_FRONTIERS)
    preds = pd.read_csv(OLD_PREDICTIONS)
    decisions = pd.read_csv(OLD_DECISIONS)

    # Data quality / grain checks.
    if devr.branch_id.duplicated().any() or len(devr) != 135 or int(devr.valid.sum()) != 135:
        raise RuntimeError("DEV repeat validity/grain failure")
    if train.branch_id.duplicated().any() or len(train) != 720 or int(train.valid.sum()) != 720:
        raise RuntimeError("TRAIN repeat validity/grain failure")
    if not (train.groupby(["context_id", "requested_force_N"]).size() == 2).all():
        raise RuntimeError("TRAIN does not have exactly two repeats/cell")

    dev_rows = []
    for r in curves.itertuples(index=False):
        k = int(r.success_count); p = k / 5; lo, hi = wilson(k, 5)
        dev_rows.append({"context_id": str(r.context_id), "root_id": str(r.root_id), "task": int(r.task),
                         "friction_band": str(r.friction_band), "friction": float(r.friction), "force_N": float(r.force_N),
                         "success_count": k, "valid_repeats": 5, "outcome_category": f"{k}/5", "empirical_p": p,
                         "bernoulli_variance": p * (1 - p), "entropy_bits": entropy(p),
                         "wilson_95_low": lo, "wilson_95_high": hi,
                         "deterministic_like": int(k in {0, 5}), "boundary_stochastic": int(0 < k < 5),
                         "raw_outcomes_json": str(r.raw_outcomes_json)})
    write_csv(out / "DEV_REPEAT_STOCHASTICITY.csv", dev_rows)

    noise_rows = []
    for r in dev_rows:
        outcomes = json.loads(r["raw_outcomes_json"])
        for m in [1, 2, 3, 4]:
            q = exact_repeat_noise([int(x) for x in outcomes], m)
            noise_rows.append({"scope": "CELL", "context_id": r["context_id"], "task": r["task"],
                               "friction_band": r["friction_band"], "force_N": r["force_N"],
                               "success_count_5": r["success_count"], "p_5": r["empirical_p"], "subsample_m": m, **q})
    for m in [1, 2, 3, 4]:
        q = [r for r in noise_rows if r["subsample_m"] == m]
        noise_rows.append({"scope": "ALL_27_CELLS", "context_id": "__ALL__", "task": "ALL", "friction_band": "ALL",
                           "force_N": math.nan, "success_count_5": math.nan, "p_5": math.nan, "subsample_m": m,
                           "combinations": sum(int(r["combinations"]) for r in q),
                           "expected_MAE": float(np.mean([r["expected_MAE"] for r in q])),
                           "RMSE": float(np.sqrt(np.mean([r["RMSE"] ** 2 for r in q]))),
                           "reliability_misclassification_probability": float(np.mean([r["reliability_misclassification_probability"] for r in q]))})
    write_csv(out / "REPEAT_COUNT_NOISE_FLOOR.csv", noise_rows)

    train_cells = []
    for (cid, force), g in train.groupby(["context_id", "requested_force_N"], sort=True):
        k = int(g.full_task_success_y.sum())
        train_cells.append({"context_id": str(cid), "root_id": str(g.root_id.iloc[0]), "task": int(g.task.iloc[0]),
                            "friction_band": str(g.friction_band.iloc[0]), "friction": float(g.friction.iloc[0]),
                            "stratum_index": int(g.stratum_index.iloc[0]), "requested_force_N": float(force),
                            "success_count": k, "valid_repeats": 2, "outcome_category": f"{k}/2",
                            "empirical_p_2": k / 2, "ambiguous_1of2": int(k == 1),
                            "branch_ids_json": json.dumps(sorted(g.branch_id.astype(str).tolist())),
                            "nearest_old_coarse_force_distance_N": min(abs(float(force) - x) for x in OLD_COARSE_BOUNDARIES[int(g.task.iloc[0])])})
    # Per-task force quantile and boundary selection uses TRAIN outcomes only.
    tdf = pd.DataFrame(train_cells)
    for task, idx in tdf.groupby("task").groups.items():
        ranks = tdf.loc[idx, "requested_force_N"].rank(method="first", pct=True)
        tdf.loc[idx, "task_force_quantile"] = ranks
        tdf.loc[idx, "task_force_quintile"] = np.minimum(5, np.ceil(ranks * 5)).astype(int)
    selected_keys: set[tuple[str, float]] = set()
    selection_reasons: dict[tuple[str, float], set[str]] = {}
    for cid, g in tdf.groupby("context_id"):
        q = g.sort_values("requested_force_N").reset_index(drop=True)
        for i, r in q.iterrows():
            key = (str(cid), float(r.requested_force_N))
            if int(r.success_count) == 1:
                selected_keys.add(key); selection_reasons.setdefault(key, set()).add("EXISTING_1_OF_2")
        for i in range(len(q) - 1):
            if int(q.iloc[i].success_count) == 0 and int(q.iloc[i + 1].success_count) == 2:
                for j in {max(0, i - 1), i, i + 1, min(len(q) - 1, i + 2)}:
                    rr = q.iloc[j]; key = (str(cid), float(rr.requested_force_N))
                    selected_keys.add(key); selection_reasons.setdefault(key, set()).add("ADJACENT_OR_NEAREST_TO_0_TO_2_TRANSITION")
    tdf["topup_selected_if_eligible"] = [int((str(r.context_id), float(r.requested_force_N)) in selected_keys) for r in tdf.itertuples()]
    tdf["topup_selection_reasons"] = ["|".join(sorted(selection_reasons.get((str(r.context_id), float(r.requested_force_N)), set()))) for r in tdf.itertuples()]
    write_csv(out / "TRAIN_REPEAT_CELL_AUDIT.csv", tdf.to_dict("records"))

    coverage_rows = []
    for r in tdf.itertuples(index=False):
        task_forces = tdf[tdf.task == r.task].requested_force_N.to_numpy(float)
        coverage_rows.append({"context_id": r.context_id, "task": int(r.task), "friction_band": r.friction_band,
                              "stratum_index": int(r.stratum_index), "requested_force_N": float(r.requested_force_N),
                              "task_force_quantile": float(r.task_force_quantile), "task_force_quintile": int(r.task_force_quintile),
                              "success_count_2": int(r.success_count), "empirical_p_2": float(r.empirical_p_2),
                              "nearest_old_coarse_force_distance_N": float(r.nearest_old_coarse_force_distance_N),
                              "task_local_density_within_0p10N": int(np.sum(np.abs(task_forces - float(r.requested_force_N)) <= 0.10 + 1e-12)),
                              "task_local_density_within_0p25N": int(np.sum(np.abs(task_forces - float(r.requested_force_N)) <= 0.25 + 1e-12)),
                              "task_force_min_N": float(task_forces.min()), "task_force_max_N": float(task_forces.max())})
    write_csv(out / "TRAIN_CONTINUOUS_FORCE_COVERAGE.csv", coverage_rows)

    # Attribution at the 27 frozen real cells, raw ensemble only.
    p = preds[(preds.backend == "FEASIBILITY_ONLY") & (preds.condition == "GT") & (preds.on_real_benchmark == 1)].copy()
    p = p[["context_id", "force_N", "raw_probability"]]
    attr = curves.merge(p, on=["context_id", "force_N"], how="left", validate="one_to_one")
    fvalid = fronts[fronts.status == "VALID_FINE_FRONTIER"][["context_id", "F_star_rho_N"]]
    attr = attr.merge(fvalid, on="context_id", how="left")
    attr_rows = []
    for r in attr.itertuples(index=False):
        tf = tdf[tdf.task == int(r.task)].requested_force_N.to_numpy(float)
        err = abs(float(r.raw_probability) - float(r.empirical_p_success))
        attr_rows.append({"context_id": str(r.context_id), "task": int(r.task), "friction_band": str(r.friction_band),
                          "friction": float(r.friction), "force_N": float(r.force_N), "empirical_p": float(r.empirical_p_success),
                          "raw_model_p": float(r.raw_probability), "absolute_probability_error": err,
                          "squared_probability_error": (float(r.raw_probability) - float(r.empirical_p_success)) ** 2,
                          "dev_bernoulli_variance": float(r.empirical_p_success) * (1 - float(r.empirical_p_success)),
                          "dev_boundary_stochastic": int(0 < int(r.success_count) < 5),
                          "distance_to_nearest_task_train_force_N": float(np.min(np.abs(tf - float(r.force_N)))),
                          "train_density_within_0p10N": int(np.sum(np.abs(tf - float(r.force_N)) <= 0.10 + 1e-12)),
                          "train_density_within_0p25N": int(np.sum(np.abs(tf - float(r.force_N)) <= 0.25 + 1e-12)),
                          "real_frontier_valid": int(math.isfinite(float(r.F_star_rho_N))) if pd.notna(r.F_star_rho_N) else 0,
                          "distance_to_real_frontier_N": abs(float(r.force_N) - float(r.F_star_rho_N)) if pd.notna(r.F_star_rho_N) else math.nan,
                          "near_real_frontier_0p25N": int(abs(float(r.force_N) - float(r.F_star_rho_N)) <= 0.25 + 1e-12) if pd.notna(r.F_star_rho_N) else 0})
    write_csv(out / "DEV_ERROR_ATTRIBUTION.csv", attr_rows)

    # Failure taxonomy on valid real frontiers.
    d = decisions[(decisions.backend == "FEASIBILITY_ONLY") & (decisions.condition == "GT")].copy()
    taxonomy = []
    for r in d[d.real_frontier_status == "VALID_FINE_FRONTIER"].itertuples(index=False):
        real = json.loads(r.real_probabilities_json); pred = json.loads(r.raw_probabilities_json)
        finite = math.isfinite(float(r.selected_force_N))
        monotonic = all(pred[i + 1] >= pred[i] - 1e-12 for i in range(len(pred) - 1))
        if not monotonic:
            typ = "TYPE_E_PROBABILITY_OR_ORDER_FAILURE"
        elif not finite:
            typ = "TYPE_B_NO_FINITE_DECISION"
        elif float(r.selected_force_N) < float(r.real_F_star_rho_N) - 1e-9:
            typ = "TYPE_C_UNDER_FORCE"
        elif abs(float(r.selected_force_N) - float(r.real_F_star_rho_N)) <= 0.05 + 1e-9:
            typ = "TYPE_A_CORRECT_FRONTIER"
        else:
            typ = "TYPE_D_OVER_FORCE"
        cell_attr = [x for x in attr_rows if x["context_id"] == str(r.context_id)]
        taxonomy.append({"context_id": str(r.context_id), "root_id": str(r.root_id), "task": int(r.task),
                         "friction_band": str(r.friction_band), "real_F_star_rho_N": float(r.real_F_star_rho_N),
                         "selected_force_N": float(r.selected_force_N), "failure_type": typ, "ordered_real_anchor_curve": int(monotonic),
                         "real_curve_json": r.real_probabilities_json, "predicted_curve_json": r.raw_probabilities_json,
                         "mean_nearest_train_force_distance_N": float(np.mean([x["distance_to_nearest_task_train_force_N"] for x in cell_attr])),
                         "mean_train_density_within_0p10N": float(np.mean([x["train_density_within_0p10N"] for x in cell_attr])),
                         "mean_dev_bernoulli_variance": float(np.mean([x["dev_bernoulli_variance"] for x in cell_attr])),
                         "mean_probability_error": float(np.mean([x["absolute_probability_error"] for x in cell_attr]))})
    write_csv(out / "CURRENT_CONTINUOUS_FAILURE_TAXONOMY.csv", taxonomy)

    # Frozen classification.
    outcome_counts = Counter(r["outcome_category"] for r in dev_rows)
    mixed_frac = float(np.mean([r["boundary_stochastic"] for r in dev_rows]))
    mean_var = float(np.mean([r["bernoulli_variance"] for r in dev_rows]))
    ambiguous_frac = float(np.mean(tdf.ambiguous_1of2))
    m2 = next(r for r in noise_rows if r["scope"] == "ALL_27_CELLS" and r["subsample_m"] == 2)
    m2_ratio = float(m2["expected_MAE"] / CURRENT_MODEL_MAE)
    ax = [r["distance_to_nearest_task_train_force_N"] for r in attr_rows]
    ay = [r["absolute_probability_error"] for r in attr_rows]
    coverage_corr = spearman(ax, ay)
    adf = pd.DataFrame(attr_rows)
    q25 = adf.distance_to_nearest_task_train_force_N.quantile(0.25)
    q75 = adf.distance_to_nearest_task_train_force_N.quantile(0.75)
    dense_err = float(adf[adf.distance_to_nearest_task_train_force_N <= q25].absolute_probability_error.mean())
    sparse_err = float(adf[adf.distance_to_nearest_task_train_force_N >= q75].absolute_probability_error.mean())
    repeat_material = m2_ratio >= FORENSIC_THRESHOLDS["repeat_noise_comparable_fraction_of_model_MAE_min"] and ambiguous_frac >= FORENSIC_THRESHOLDS["train_ambiguous_1of2_fraction_min"]
    coverage_material = ((math.isfinite(coverage_corr) and abs(coverage_corr) >= FORENSIC_THRESHOLDS["coverage_abs_spearman_error_vs_nearest_distance_min"] and coverage_corr > 0)
                         or sparse_err - dense_err >= FORENSIC_THRESHOLDS["coverage_sparse_minus_dense_error_min"])
    stochasticity_material = mixed_frac >= FORENSIC_THRESHOLDS["dev_boundary_stochastic_fraction_min"] and mean_var >= FORENSIC_THRESHOLDS["dev_mean_bernoulli_variance_min"]
    model_ratio = CURRENT_MODEL_MAE / max(float(m2["expected_MAE"]), 1e-12)
    model_material = model_ratio >= FORENSIC_THRESHOLDS["model_error_to_m2_noise_ratio_min"]
    train_material = repeat_material or coverage_material
    if train_material and stochasticity_material:
        classification = "MIXED_COVERAGE_AND_STOCHASTICITY"
    elif repeat_material:
        classification = "TRAIN_REPEAT_NOISE_DOMINATES"
    elif coverage_material:
        classification = "CONTINUOUS_TRAIN_COVERAGE_DOMINATES"
    elif stochasticity_material:
        classification = "REAL_FRONTIER_STOCHASTICITY_DOMINATES"
    elif model_material:
        classification = "MODEL_ERROR_DOMINATES"
    else:
        classification = "INSUFFICIENT_VALID_EVIDENCE"
    collect = classification in {"TRAIN_REPEAT_NOISE_DOMINATES", "CONTINUOUS_TRAIN_COVERAGE_DOMINATES", "MIXED_COVERAGE_AND_STOCHASTICITY"}
    result = {
        "FORENSIC_CLASSIFICATION": classification,
        "topup_scientifically_justified": collect,
        "evidence_flags": {"repeat_supervision_material": repeat_material, "coverage_material": coverage_material,
                           "real_stochasticity_material": stochasticity_material, "model_error_material": model_material},
        "dev_outcome_counts": dict(sorted(outcome_counts.items())),
        "dev_boundary_stochastic_fraction": mixed_frac,
        "dev_deterministic_like_fraction": 1 - mixed_frac,
        "dev_mean_bernoulli_variance": mean_var,
        "dev_mean_entropy_bits": float(np.mean([r["entropy_bits"] for r in dev_rows])),
        "m2_exact_expected_MAE_vs_p5": float(m2["expected_MAE"]),
        "m2_RMSE_vs_p5": float(m2["RMSE"]),
        "m2_reliability_misclassification_probability": float(m2["reliability_misclassification_probability"]),
        "m2_noise_to_current_model_MAE_ratio": m2_ratio,
        "current_model_MAE_to_m2_noise_ratio": model_ratio,
        "train_cell_counts": {f"{k}/2": int((tdf.success_count == k).sum()) for k in [0, 1, 2]},
        "train_ambiguous_1of2_fraction": ambiguous_frac,
        "coverage_error_spearman": coverage_corr,
        "coverage_sparse_minus_dense_error": sparse_err - dense_err,
        "coverage_dense_quartile_error": dense_err,
        "coverage_sparse_quartile_error": sparse_err,
        "frozen_thresholds": FORENSIC_THRESHOLDS,
        "target_cells_if_collection_eligible": len(selected_keys),
        "expected_new_branches_if_collection_eligible": 3 * len(selected_keys),
        "failure_taxonomy_counts": dict(Counter(r["failure_type"] for r in taxonomy)),
        "data_quality": {"DEV_valid_branches": 135, "DEV_duplicate_branch_ids": 0, "TRAIN_valid_branches": 720,
                         "TRAIN_duplicate_branch_ids": 0, "TRAIN_cells_exactly_2_repeats": 360},
        "important_limit": "five-repeat empirical probabilities are finite-sample references, not exact physical probabilities",
    }
    write_json(out / "FORENSIC_CLASSIFICATION.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="phase", required=True)
    for name in ["prepare", "analyze"]:
        p = sp.add_parser(name); p.add_argument("--out", required=True)
    args = ap.parse_args(); out = Path(args.out)
    if args.phase == "prepare": prepare(out)
    elif args.phase == "analyze": analyze(out)


if __name__ == "__main__":
    main()
