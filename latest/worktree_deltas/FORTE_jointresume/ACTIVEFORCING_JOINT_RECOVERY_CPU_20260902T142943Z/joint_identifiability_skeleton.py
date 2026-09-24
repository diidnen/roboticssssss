#!/usr/bin/env python3
"""CPU-only, fail-closed skeleton for 3x3 joint-physics identifiability.

This program never imports a simulator or GPU framework. It validates a future
episode-level query dataset and, only when every hard gate passes, fits
root-grouped query-only diagnostic estimators. It intentionally contains no
force-choice proxy and cannot emit controller, success-rate, or Utility claims.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Fail closed before importing numerical packages that might have optional GPU
# backends. The analysis itself uses NumPy and scikit-learn CPU estimators only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

PROTOCOL_VERSION = "ACTIVEFORCING_FULL_CLAIM_V2_UTILITY"
FRICTION_LEVELS = np.asarray([0.25, 0.50, 0.75], dtype=float)
MASS_LEVELS = np.asarray([0.05, 0.10, 0.20], dtype=float)
EXPECTED_SPLITS = ("TRAIN", "DEV", "LOCKED_TEST")

# Only aggregate values derived during the frozen physical query are legal.
# True physics labels are targets and are never features. Privileged object pose
# and any downstream outcome/frontier/force-choice fields are excluded.
LEGAL_QUERY_FEATURES = (
    "target_preload_N",
    "actual_probe_displacement_mm",
    "probe_path_mm",
    "probe_duration_s",
    "preprobe_normal_force",
    "preprobe_marker_motion",
    "preprobe_tangential_force",
    "f_meas_mean",
    "normal_force_mean",
    "normal_force_peak",
    "normal_force_hyst",
    "normal_loading_slope",
    "normal_unloading_slope",
    "ftan_mean",
    "ftan_peak",
    "ftan_hyst",
    "rho_mean",
    "rho_peak",
    "rho_hyst",
    "rho_impulse",
    "imb_mean",
    "imb_peak",
    "imb_ratio_mean",
    "imb_ratio_peak",
    "marker_mean",
    "marker_peak",
    "marker_tangential_mean",
    "marker_tangential_peak",
    "marker_hyst",
    "marker_unloading",
    "marker_vel_abs_peak",
    "residual_marker_displacement",
    "gripper_opening_mean",
)

REQUIRED_COLUMNS = {
    "protocol_version",
    "evidence_role",
    "trial_id",
    "task_id",
    "split",
    "root_seed",
    "friction",
    "mass_kg",
    "query_repeat",
    "query_valid",
    "query_rows",
    "pre_query_state_hash",
    "requested_friction",
    "applied_friction",
    "requested_mass_kg",
    "applied_mass_kg",
    "mass_ratio",
    "inertia_ratio",
}

FORBIDDEN_FEATURE_TOKENS = (
    "success",
    "outcome",
    "selected_force",
    "oracle_force",
    "frontier",
    "priv",
    "root_seed",
    "split",
    "friction",
    "mass",
)


class GateFailure(RuntimeError):
    """Raised when analysis must not produce scientific metrics."""


@dataclass(frozen=True)
class Audit:
    rows: int
    roots_by_split: dict[str, int]
    cells_by_split: dict[str, int]
    valid_by_split: dict[str, int]
    features: tuple[str, ...]


@dataclass(frozen=True)
class StandardizedRidge:
    """Small deterministic NumPy ridge model with an unpenalized intercept."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        design = np.column_stack((np.ones(len(values)), (values - self.mean) / self.scale))
        return design @ self.weights


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key)
    if raw is None or raw.strip() == "":
        raise GateFailure(f"missing required numeric value: {key}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise GateFailure(f"invalid numeric value for {key}: {raw!r}") from exc
    if not math.isfinite(value):
        raise GateFailure(f"non-finite numeric value for {key}: {raw!r}")
    return value


def nearest(values: np.ndarray, observations: np.ndarray) -> np.ndarray:
    return values[np.argmin(np.abs(observations[:, None] - values[None, :]), axis=1)]


def cell_key(row: dict[str, str]) -> tuple[str, str, float, float, int]:
    return (
        row["split"],
        row["root_seed"],
        finite_float(row, "friction"),
        finite_float(row, "mass_kg"),
        int(finite_float(row, "query_repeat")),
    )


def choose_features(header: Iterable[str], requested: Sequence[str] | None) -> tuple[str, ...]:
    features = tuple(requested) if requested else tuple(x for x in LEGAL_QUERY_FEATURES if x in header)
    if not features:
        raise GateFailure("no legal query-only features are present")
    illegal = [
        feature
        for feature in features
        if feature not in LEGAL_QUERY_FEATURES
        or any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if illegal:
        raise GateFailure(f"illegal or unregistered feature columns: {illegal}")
    return features


def validate(rows: list[dict[str, str]], requested_features: Sequence[str] | None = None) -> Audit:
    if not rows:
        raise GateFailure("episode dataset is empty")
    missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
    if missing:
        raise GateFailure(f"required columns missing: {missing}")
    if any(row.get("protocol_version") != PROTOCOL_VERSION for row in rows):
        raise GateFailure("protocol_version mismatch")
    if any(row.get("evidence_role") not in {"TRAIN", "DEV", "LOCKED_TEST"} for row in rows):
        raise GateFailure("invalid evidence_role")
    if any(row.get("split") not in EXPECTED_SPLITS for row in rows):
        raise GateFailure("invalid split")
    if any(row.get("evidence_role") != row.get("split") for row in rows):
        raise GateFailure("evidence_role and split disagree")
    if any(int(finite_float(row, "query_repeat")) != 0 for row in rows):
        raise GateFailure("this frozen plan requires one query with query_repeat=0 per cell")

    features = choose_features(rows[0], requested_features)
    seen: Counter[tuple[str, str, float, float, int]] = Counter(cell_key(row) for row in rows)
    duplicates = [key for key, count in seen.items() if count != 1]
    if duplicates:
        raise GateFailure(f"duplicate atomic cell keys: {duplicates[:5]}")

    roots_by_split: dict[str, set[str]] = defaultdict(set)
    keys_by_root: dict[tuple[str, str], set[tuple[float, float]]] = defaultdict(set)
    valid_by_split: Counter[str] = Counter()
    state_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        split, root, mu, mass, _ = cell_key(row)
        if not np.any(np.isclose(mu, FRICTION_LEVELS, atol=1e-9)):
            raise GateFailure(f"off-grid friction label in {row['trial_id']}: {mu}")
        if not np.any(np.isclose(mass, MASS_LEVELS, atol=1e-9)):
            raise GateFailure(f"off-grid mass label in {row['trial_id']}: {mass}")
        if abs(finite_float(row, "requested_friction") - finite_float(row, "applied_friction")) > 1e-6:
            raise GateFailure(f"friction readback mismatch in {row['trial_id']}")
        if abs(finite_float(row, "requested_mass_kg") - finite_float(row, "applied_mass_kg")) > 1e-6:
            raise GateFailure(f"mass readback mismatch in {row['trial_id']}")
        if abs(finite_float(row, "mass_ratio") - finite_float(row, "inertia_ratio")) > 1e-6:
            raise GateFailure(f"mass/inertia ratio mismatch in {row['trial_id']}")
        if finite_float(row, "query_rows") <= 0:
            raise GateFailure(f"no query rows in {row['trial_id']}")
        for feature in features:
            finite_float(row, feature)
        roots_by_split[split].add(root)
        keys_by_root[(split, root)].add((mu, mass))
        state_hashes[(split, root)].add(row["pre_query_state_hash"])
        valid_by_split[split] += int(finite_float(row, "query_valid") == 1.0)

    split_root_sets = [roots_by_split[name] for name in EXPECTED_SPLITS]
    for left_index, left in enumerate(split_root_sets):
        for right in split_root_sets[left_index + 1 :]:
            if left & right:
                raise GateFailure(f"root split overlap: {sorted(left & right)}")
    expected_cells = {(float(mu), float(mass)) for mu in FRICTION_LEVELS for mass in MASS_LEVELS}
    for group, cells in keys_by_root.items():
        if cells != expected_cells:
            raise GateFailure(f"incomplete 3x3 grid for {group}: {len(cells)}/9")
        if len(state_hashes[group]) != 1:
            raise GateFailure(f"pre-query state parity failure for {group}")

    # Query failures remain explicit and in the coverage denominator. Modeling
    # is blocked unless the separately frozen validity threshold allows them;
    # this initial skeleton requires complete validity.
    if any(valid_by_split[name] != 9 * len(roots_by_split[name]) for name in roots_by_split):
        raise GateFailure("one or more query cells are invalid; no conditional-only fit is allowed")

    return Audit(
        rows=len(rows),
        roots_by_split={name: len(roots_by_split[name]) for name in EXPECTED_SPLITS},
        cells_by_split={name: sum(row["split"] == name for row in rows) for name in EXPECTED_SPLITS},
        valid_by_split={name: valid_by_split[name] for name in EXPECTED_SPLITS},
        features=features,
    )


def arrays(rows: list[dict[str, str]], features: Sequence[str], split: str):
    selected = [row for row in rows if row["split"] == split]
    x = np.asarray([[finite_float(row, key) for key in features] for row in selected], dtype=float)
    y = np.asarray([[finite_float(row, "friction"), finite_float(row, "mass_kg")] for row in selected], dtype=float)
    groups = np.asarray([row["root_seed"] for row in selected], dtype=object)
    return selected, x, y, groups


def fit_ridge(values: np.ndarray, targets: np.ndarray, alpha: float) -> StandardizedRidge:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack((np.ones(len(values)), (values - mean) / scale))
    penalty = np.eye(design.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return StandardizedRidge(mean=mean, scale=scale, weights=weights)


def group_folds(groups: np.ndarray, folds: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Deterministic root-group folds without an external ML dependency."""
    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    for fold_index in range(folds):
        validation_groups = set(unique[fold_index::folds].tolist())
        validation = np.asarray([group in validation_groups for group in groups], dtype=bool)
        yield np.flatnonzero(~validation), np.flatnonzero(validation)


def select_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise GateFailure("at least three TRAIN roots are required for grouped alpha selection")
    candidates = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    folds = min(3, len(unique_groups))
    scores: list[tuple[float, float]] = []
    scale = np.asarray([0.25, 0.075], dtype=float)
    for alpha in candidates:
        fold_scores = []
        for train_index, valid_index in group_folds(groups, folds):
            model = fit_ridge(x[train_index], y[train_index], alpha)
            prediction = model.predict(x[valid_index])
            fold_scores.append(float(np.mean(np.abs(prediction - y[valid_index]) / scale)))
        scores.append((float(np.mean(fold_scores)), alpha))
    return min(scores)[1]


def confusion(true: np.ndarray, predicted: np.ndarray, levels: np.ndarray) -> list[list[int]]:
    predicted_band = nearest(levels, predicted)
    result = np.zeros((len(levels), len(levels)), dtype=int)
    for actual, estimate in zip(true, predicted_band):
        actual_index = int(np.flatnonzero(np.isclose(levels, actual, atol=1e-9))[0])
        estimate_index = int(np.flatnonzero(np.isclose(levels, estimate, atol=1e-9))[0])
        result[actual_index, estimate_index] += 1
    return result.tolist()


def spearman_or_none(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    left_rank = np.argsort(np.argsort(left)).astype(float)
    right_rank = np.argsort(np.argsort(right)).astype(float)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def root_bootstrap_intervals(
    rows: list[dict[str, str]], truth: np.ndarray, prediction: np.ndarray, draws: int = 2000
) -> dict[str, list[float]]:
    roots = sorted({row["root_seed"] for row in rows})
    if len(roots) < 2:
        return {}
    indices = {root: np.asarray([i for i, row in enumerate(rows) if row["root_seed"] == root]) for root in roots}
    random = np.random.default_rng(20260902)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        chosen = random.choice(roots, size=len(roots), replace=True)
        selected = np.concatenate([indices[str(root)] for root in chosen])
        selected_truth = truth[selected]
        selected_prediction = prediction[selected]
        mu_band = nearest(FRICTION_LEVELS, selected_prediction[:, 0])
        mass_band = nearest(MASS_LEVELS, selected_prediction[:, 1])
        samples["friction_mae"].append(float(np.mean(np.abs(selected_prediction[:, 0] - selected_truth[:, 0]))))
        samples["mass_mae_kg"].append(float(np.mean(np.abs(selected_prediction[:, 1] - selected_truth[:, 1]))))
        samples["friction_band_accuracy"].append(float(np.mean(mu_band == selected_truth[:, 0])))
        samples["mass_band_accuracy"].append(float(np.mean(mass_band == selected_truth[:, 1])))
        samples["exact_joint_cell_accuracy"].append(
            float(np.mean((mu_band == selected_truth[:, 0]) & (mass_band == selected_truth[:, 1])))
        )
    return {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in samples.items()
    }


def metrics(rows: list[dict[str, str]], truth: np.ndarray, prediction: np.ndarray) -> dict:
    pred_mu = prediction[:, 0]
    pred_mass = prediction[:, 1]
    mu_band = nearest(FRICTION_LEVELS, pred_mu)
    mass_band = nearest(MASS_LEVELS, pred_mass)
    root_values: dict[str, list[int]] = defaultdict(list)
    for row, mu_ok, mass_ok in zip(rows, mu_band == truth[:, 0], mass_band == truth[:, 1]):
        root_values[row["root_seed"]].append(int(mu_ok and mass_ok))
    friction_by_mass = {
        f"{level:.2f}": float(np.mean((mu_band == truth[:, 0])[np.isclose(truth[:, 1], level)]))
        for level in MASS_LEVELS
    }
    mass_by_friction = {
        f"{level:.2f}": float(np.mean((mass_band == truth[:, 1])[np.isclose(truth[:, 0], level)]))
        for level in FRICTION_LEVELS
    }
    friction_spread_groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    mass_spread_groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row, target, estimate in zip(rows, truth, prediction):
        friction_spread_groups[(row["root_seed"], float(target[0]))].append(float(estimate[0]))
        mass_spread_groups[(row["root_seed"], float(target[1]))].append(float(estimate[1]))
    friction_nuisance_spread = float(np.mean([np.ptp(values) for values in friction_spread_groups.values()]))
    mass_nuisance_spread = float(np.mean([np.ptp(values) for values in mass_spread_groups.values()]))
    return {
        "n_cells": len(rows),
        "n_root_clusters": len(root_values),
        "friction_mae": float(np.mean(np.abs(pred_mu - truth[:, 0]))),
        "mass_mae_kg": float(np.mean(np.abs(pred_mass - truth[:, 1]))),
        "friction_band_accuracy": float(np.mean(mu_band == truth[:, 0])),
        "mass_band_accuracy": float(np.mean(mass_band == truth[:, 1])),
        "exact_joint_cell_accuracy": float(np.mean((mu_band == truth[:, 0]) & (mass_band == truth[:, 1]))),
        "exact_joint_accuracy_by_root": {root: float(np.mean(values)) for root, values in sorted(root_values.items())},
        "friction_confusion": confusion(truth[:, 0], pred_mu, FRICTION_LEVELS),
        "mass_confusion": confusion(truth[:, 1], pred_mass, MASS_LEVELS),
        "friction_band_accuracy_by_true_mass": friction_by_mass,
        "mass_band_accuracy_by_true_friction": mass_by_friction,
        "friction_error_vs_true_mass_spearman": spearman_or_none(pred_mu - truth[:, 0], truth[:, 1]),
        "mass_error_vs_true_friction_spearman": spearman_or_none(pred_mass - truth[:, 1], truth[:, 0]),
        "friction_prediction_range_across_mass_mean": friction_nuisance_spread,
        "mass_prediction_range_across_friction_mean_kg": mass_nuisance_spread,
        "root_cluster_bootstrap_95pct": root_bootstrap_intervals(rows, truth, prediction),
    }


def axis_control(
    x_train: np.ndarray,
    y_train: np.ndarray,
    axis: int,
    nuisance_value: float,
    alpha: float,
) -> tuple[StandardizedRidge, np.ndarray]:
    nuisance_axis = 1 - axis
    mask = np.isclose(y_train[:, nuisance_axis], nuisance_value, atol=1e-9)
    if np.sum(mask) < 3:
        raise GateFailure("insufficient nominal-nuisance TRAIN slice for axis control")
    model = fit_ridge(x_train[mask], y_train[mask, axis], alpha)
    return model, mask


def analyze(rows: list[dict[str, str]], audit: Audit, evaluation_split: str) -> dict:
    if evaluation_split not in {"DEV", "LOCKED_TEST"}:
        raise GateFailure("evaluation split must be DEV or LOCKED_TEST")
    train_rows, x_train, y_train, train_groups = arrays(rows, audit.features, "TRAIN")
    eval_rows, x_eval, y_eval, _ = arrays(rows, audit.features, evaluation_split)
    if not train_rows or not eval_rows:
        raise GateFailure(f"TRAIN and {evaluation_split} must both be populated")

    alpha = select_alpha(x_train, y_train, train_groups)
    joint = fit_ridge(x_train, y_train, alpha)
    joint_prediction = joint.predict(x_eval)

    friction_model, friction_mask = axis_control(x_train, y_train, axis=0, nuisance_value=0.10, alpha=alpha)
    mass_model, mass_mask = axis_control(x_train, y_train, axis=1, nuisance_value=0.50, alpha=alpha)
    friction_prediction = np.column_stack((friction_model.predict(x_eval), np.full(len(x_eval), 0.10)))
    mass_prediction = np.column_stack((np.full(len(x_eval), 0.50), mass_model.predict(x_eval)))

    return {
        "status": "QUERY_IDENTIFIABILITY_DIAGNOSTIC_ONLY",
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_split": evaluation_split,
        "dataset_grain": "one query episode per root x friction x mass cell",
        "independence_unit": "root_seed_family",
        "selected_alpha_train_grouped_cv": alpha,
        "feature_columns": list(audit.features),
        "models": {
            "joint_multioutput_ridge": metrics(eval_rows, y_eval, joint_prediction),
            "friction_axis_control_nominal_mass_only": metrics(eval_rows, y_eval, friction_prediction),
            "mass_axis_control_nominal_friction_only": metrics(eval_rows, y_eval, mass_prediction),
        },
        "train_slice_counts": {
            "joint": len(x_train),
            "friction_axis_nominal_mass": int(np.sum(friction_mask)),
            "mass_axis_nominal_friction": int(np.sum(mass_mask)),
        },
        "unavailable_without_downstream_contract": [
            "joint force-decision accuracy",
            "interaction effect on task outcome",
            "success rate",
            "under-force and excess-force",
            "realized Utility",
        ],
        "claim_guardrail": "This output alone cannot establish a joint controller or E8b result.",
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluation-split", choices=("DEV", "LOCKED_TEST"), default="DEV")
    parser.add_argument("--feature", action="append", dest="features")
    parser.add_argument("--allow-locked-test", action="store_true")
    args = parser.parse_args()

    status = {
        "status": "BLOCKED_NO_METRICS",
        "protocol_version": PROTOCOL_VERSION,
        "episodes": str(args.episodes.resolve()),
    }
    try:
        if args.evaluation_split == "LOCKED_TEST" and not args.allow_locked_test:
            raise GateFailure("LOCKED_TEST access requires the explicit post-freeze flag")
        rows = read_rows(args.episodes)
        audit = validate(rows, args.features)
        result = analyze(rows, audit, args.evaluation_split)
        result["input_sha256"] = sha256(args.episodes)
        result["audit"] = {
            "rows": audit.rows,
            "roots_by_split": audit.roots_by_split,
            "cells_by_split": audit.cells_by_split,
            "valid_by_split": audit.valid_by_split,
        }
        write_json(args.out / f"JOINT_IDENTIFIABILITY_{args.evaluation_split}.diagnostic.json", result)
        status.update({"status": "COMPLETED_DIAGNOSTIC_ONLY", "input_sha256": result["input_sha256"]})
        return_code = 0
    except (GateFailure, FileNotFoundError, KeyError, ValueError) as exc:
        status["blocker"] = str(exc)
        return_code = 2
    finally:
        write_json(args.out / "JOINT_IDENTIFIABILITY_RUN_STATUS.json", status)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
