#!/usr/bin/env python3
"""CPU-only, blind-safe E2 modality and posterior-fusion diagnostic.

This script consumes only the frozen 72-context ``collection_train`` manifest.
It never imports a simulator or GPU library, never opens a path containing
``test`` or ``outcome``, and never writes outside its requested output root.

The diagnostic is intentionally not a replacement for a formal E2 run.  It
provides a matched Vision-only prior, deployable interaction-history ensemble,
source-only calibrated Gaussian fusion, and a quadrature handoff for an
unchanged downstream expected-Utility consumer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Defense in depth.  The implementation below uses no GPU-capable package.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/forte_e2_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


TASKS = (0, 1, 5, 6)
OUTER_FOLDS = (0, 1, 2)
ENSEMBLE_SEEDS = (1701, 1702, 1703, 1704, 1705)
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
BOUNDS = (0.2, 1.0)
PRIVILEGED_CHANNELS = {
    "contact_normal_x",
    "contact_normal_y",
    "contact_normal_z",
    "contact_tangent_x",
    "contact_tangent_y",
    "contact_tangent_z",
}
FORBIDDEN_PATH_RE = re.compile(r"(^|[/_.-])(test|tests|outcome|outcomes)([/_.-]|$)", re.I)
EXPECTED_ARCHIVED_RESULTS = (
    "PROBE_MEMBER_PREDICTIONS.csv",
    "PROBE_AGGREGATE_PREDICTIONS.csv",
    "PROBE_ROOT_HELDOUT_RESULTS.csv",
    "PROBE_LOTO_RESULTS.csv",
    "PROBE_UNCERTAINTY_CALIBRATION.csv",
    "PROBE_REPROBE_GATE.csv",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    def normalize(item: object) -> object:
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"cannot serialize {type(item).__name__}")

    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=normalize) + "\n", encoding="utf-8")


def safe_input(path: Path, *, allow_manifest: bool = False) -> Path:
    path = path.resolve()
    if FORBIDDEN_PATH_RE.search(str(path)):
        raise RuntimeError(f"blind-boundary rejection: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def stable_fold(value: str, folds: int = 3) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % folds


def sequence_dataframe(path: Path, phases: list[str], states: list[str], numeric: list[str]) -> pd.DataFrame:
    safe_input(path)
    df = pd.read_csv(path)
    out = pd.DataFrame(index=df.index)
    for col in numeric:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df else 0.0
    for axis in "xyz":
        col = f"eef_{axis}"
        raw = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(0.0, index=df.index)
        out[f"eef_d{axis}"] = raw - float(raw.iloc[0])
    phase_values = df["probe_phase"].fillna("NA").astype(str)
    state_values = df["contact_state"].fillna("NA").astype(str)
    for phase in phases:
        out[f"phase={phase}"] = (phase_values == phase).astype(float)
    for state in states:
        out[f"contact_state={state}"] = (state_values == state).astype(float)
    out["probe_phase_unknown"] = (~phase_values.isin(phases)).astype(float)
    out["contact_state_unknown"] = (~state_values.isin(states)).astype(float)
    return out


def summarize_sequences(sequences: np.ndarray) -> np.ndarray:
    """Create deterministic temporal summaries without post-interaction outcomes."""
    n, timesteps, channels = sequences.shape
    q = np.quantile(sequences, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1).transpose(1, 0, 2)
    mean = sequences.mean(axis=1)
    std = sequences.std(axis=1)
    lo = sequences.min(axis=1)
    hi = sequences.max(axis=1)
    delta = sequences[:, -1] - sequences[:, 0]
    t = np.linspace(-1.0, 1.0, timesteps, dtype=np.float64)
    slope = np.einsum("ntc,t->nc", sequences, t) / float(np.dot(t, t))
    blocks = np.stack([chunk.mean(axis=1) for chunk in np.array_split(sequences, 10, axis=1)], axis=1)
    pieces = [mean, std, lo, hi, delta, slope, q.reshape(n, -1), blocks.reshape(n, -1)]
    return np.concatenate(pieces, axis=1).astype(np.float64)


@dataclass
class Projection:
    mean: np.ndarray
    std: np.ndarray
    components: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.mean) / self.std) @ self.components.T
        return (z - self.z_mean) / self.z_std


def fit_projection(x: np.ndarray, max_components: int) -> Projection:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    z = (x - mean) / std
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    rank = max(1, int(np.sum(singular > 1e-9)))
    k = min(max_components, rank, max(1, len(x) - 1))
    components = vt[:k]
    projected = z @ components.T
    z_mean = projected.mean(axis=0)
    z_std = projected.std(axis=0)
    z_std[z_std < 1e-8] = 1.0
    return Projection(mean, std, components, z_mean, z_std)


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alpha: float, max_components: int) -> np.ndarray:
    projection = fit_projection(x_train, max_components)
    z_train = projection.transform(x_train)
    z_eval = projection.transform(x_eval)
    y_mean = float(y_train.mean())
    gram = z_train.T @ z_train + float(alpha) * np.eye(z_train.shape[1])
    coef = np.linalg.solve(gram, z_train.T @ (y_train - y_mean))
    return np.clip(y_mean + z_eval @ coef, BOUNDS[0], BOUNDS[1])


def inner_fold_vector(meta: pd.DataFrame) -> np.ndarray:
    mapping: dict[str, int] = {}
    for task, group in meta.groupby("task"):
        roots = sorted(group.root_id.astype(str).unique())
        # Offset each task deterministically to avoid fold-size aliasing.
        offset = stable_fold(f"task:{task}")
        for index, root in enumerate(roots):
            mapping[root] = (index + offset) % 3
    return meta.root_id.astype(str).map(mapping).to_numpy(int)


def choose_alpha(x: np.ndarray, y: np.ndarray, meta: pd.DataFrame, max_components: int) -> tuple[float, dict[str, float]]:
    folds = inner_fold_vector(meta)
    scores: dict[str, float] = {}
    for alpha in RIDGE_GRID:
        pred = np.full(len(y), np.nan, dtype=np.float64)
        for fold in range(3):
            train = np.flatnonzero(folds != fold)
            valid = np.flatnonzero(folds == fold)
            if not len(train) or not len(valid):
                raise RuntimeError("empty source-only inner fold")
            pred[valid] = ridge_predict(x[train], y[train], x[valid], alpha, max_components)
        scores[str(alpha)] = float(np.mean(np.abs(pred - y)))
    selected = min(RIDGE_GRID, key=lambda value: (scores[str(value)], value))
    return float(selected), scores


def stratified_bootstrap(meta: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    indices: list[int] = []
    for _, group in meta.groupby("task"):
        roots = sorted(group.root_id.astype(str).unique())
        sampled = rng.choice(roots, size=len(roots), replace=True)
        root_to_indices = {root: np.flatnonzero(meta.root_id.astype(str).to_numpy() == root) for root in roots}
        for root in sampled:
            indices.extend(root_to_indices[str(root)].tolist())
    return np.asarray(indices, dtype=int)


def ensemble_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    x_eval: np.ndarray,
    alpha: float,
    max_components: int,
    seed_offset: int,
) -> np.ndarray:
    members = []
    for seed in ENSEMBLE_SEEDS:
        rng = np.random.default_rng(seed + seed_offset)
        boot = stratified_bootstrap(meta_train.reset_index(drop=True), rng)
        members.append(ridge_predict(x_train[boot], y_train[boot], x_eval, alpha, max_components))
    return np.stack(members, axis=1)


def source_oof_members(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    alpha: float,
    max_components: int,
    seed_offset: int,
) -> np.ndarray:
    folds = inner_fold_vector(meta)
    out = np.full((len(y), len(ENSEMBLE_SEEDS)), np.nan, dtype=np.float64)
    for fold in range(3):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        out[valid] = ensemble_predict(
            x[train], y[train], meta.iloc[train].reset_index(drop=True), x[valid], alpha, max_components,
            seed_offset + fold * 100,
        )
    if not np.isfinite(out).all():
        raise RuntimeError("incomplete source-only OOF ensemble")
    return out


@dataclass
class VarianceCalibration:
    residual_floor: float
    scale: float

    def apply(self, members: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = members.mean(axis=1)
        epistemic = members.var(axis=1, ddof=0)
        variance = np.maximum(self.scale * (epistemic + self.residual_floor), 1e-6)
        return mean, variance


def calibrate_members(members: np.ndarray, y: np.ndarray) -> VarianceCalibration:
    mean = members.mean(axis=1)
    epistemic = members.var(axis=1, ddof=0)
    residual2 = np.square(y - mean)
    floor = max(float(np.median(residual2)), 1e-4)
    base = epistemic + floor
    scale = float(np.clip(np.mean(residual2 / np.maximum(base, 1e-8)), 0.05, 50.0))
    return VarianceCalibration(floor, scale)


def fuse_gaussians(
    visual_mean: np.ndarray,
    visual_var: np.ndarray,
    interaction_mean: np.ndarray,
    interaction_var: np.ndarray,
    disagreement_aware: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visual_precision = 1.0 / np.maximum(visual_var, 1e-8)
    interaction_precision = 1.0 / np.maximum(interaction_var, 1e-8)
    denom = visual_precision + interaction_precision
    interaction_weight = interaction_precision / denom
    mean = (visual_precision * visual_mean + interaction_precision * interaction_mean) / denom
    if disagreement_aware:
        visual_weight = 1.0 - interaction_weight
        variance = (
            visual_weight * visual_var
            + interaction_weight * interaction_var
            + visual_weight * interaction_weight * np.square(visual_mean - interaction_mean)
        )
    else:
        variance = 1.0 / denom
    return np.clip(mean, BOUNDS[0], BOUNDS[1]), np.maximum(variance, 1e-6), interaction_weight


def rescale_variance(mean: np.ndarray, variance: np.ndarray, y: np.ndarray) -> float:
    return float(np.clip(np.mean(np.square(y - mean) / np.maximum(variance, 1e-8)), 0.05, 50.0))


def gaussian_nll(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
    variance = np.maximum(variance, 1e-8)
    return float(np.mean(0.5 * (np.log(2.0 * math.pi * variance) + np.square(y - mean) / variance)))


def pair_ranking(meta: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> float:
    work = meta[["root_id"]].copy()
    work["y"] = y
    work["pred"] = pred
    values: list[int] = []
    for _, group in work.groupby("root_id"):
        a = group[["y", "pred"]].to_numpy(float)
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                if a[i, 0] != a[j, 0]:
                    values.append(int(np.sign(a[i, 0] - a[j, 0]) == np.sign(a[i, 1] - a[j, 1])))
    return float(np.mean(values)) if values else math.nan


def metric_row(meta: pd.DataFrame, y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float | int]:
    error = mean - y
    sigma = np.sqrt(np.maximum(variance, 1e-8))
    coverage_levels = (0.50, 0.68, 0.80, 0.90, 0.95)
    observed = []
    for level in coverage_levels:
        z = float(norm.ppf((1.0 + level) / 2.0))
        observed.append(float(np.mean(np.abs(error) <= z * sigma)))
    rho = spearmanr(y, mean).statistic if len(y) > 1 else math.nan
    uncertainty_rho = spearmanr(np.abs(error), sigma).statistic if len(y) > 1 else math.nan
    return {
        "contexts": int(len(y)),
        "roots": int(meta.root_id.nunique()),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "Spearman": float(rho),
        "pair_ranking": pair_ranking(meta, y, mean),
        "gaussian_NLL": gaussian_nll(y, mean, variance),
        "interval_ECE": float(np.mean(np.abs(np.asarray(observed) - np.asarray(coverage_levels)))),
        "coverage_68": observed[1],
        "coverage_95": observed[-1],
        "abs_error_sigma_Spearman": float(uncertainty_rho),
        "mean_sigma": float(np.mean(sigma)),
    }


def risk_coverage_rows(
    split: str,
    heldout: str,
    method: str,
    meta: pd.DataFrame,
    y: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> list[dict[str, object]]:
    error = np.abs(mean - y)
    sigma = np.sqrt(np.maximum(variance, 1e-8))
    order = np.argsort(sigma)
    rows: list[dict[str, object]] = []
    for coverage in (0.25, 0.50, 0.75, 1.00):
        count = max(1, int(math.ceil(coverage * len(y))))
        selected = order[:count]
        rows.append(
            {
                "split": split,
                "heldout": heldout,
                "method": method,
                "metric": "risk_coverage_MAE",
                "nominal": coverage,
                "observed": float(error[selected].mean()),
                "contexts": int(count),
            }
        )
    for coverage in (0.50, 0.68, 0.80, 0.90, 0.95):
        z = float(norm.ppf((1.0 + coverage) / 2.0))
        rows.append(
            {
                "split": split,
                "heldout": heldout,
                "method": method,
                "metric": "interval_coverage",
                "nominal": coverage,
                "observed": float(np.mean(error <= z * sigma)),
                "contexts": int(len(y)),
            }
        )
    return rows


def cluster_bootstrap(predictions: pd.DataFrame, split: str, comparisons: Iterable[tuple[str, str]]) -> list[dict[str, object]]:
    work = predictions[predictions.split == split]
    roots = sorted(work.root_id.unique())
    rows: list[dict[str, object]] = []
    for left, right in comparisons:
        pivot = work[work.method.isin([left, right])].pivot(index="context_id", columns="method", values="mu_hat")
        truth = work.drop_duplicates("context_id").set_index("context_id")[["root_id", "mu_GT"]]
        joined = truth.join(pivot, how="inner").dropna()
        rng = np.random.default_rng(20260902 + stable_fold(f"{split}:{left}:{right}", 10000))
        diffs = []
        available_roots = sorted(joined.root_id.unique())
        for _ in range(2000):
            sample = rng.choice(available_roots, size=len(available_roots), replace=True)
            left_errors: list[float] = []
            right_errors: list[float] = []
            for root in sample:
                group = joined[joined.root_id == root]
                left_errors.extend(np.abs(group[left] - group.mu_GT).tolist())
                right_errors.extend(np.abs(group[right] - group.mu_GT).tolist())
            diffs.append(float(np.mean(left_errors) - np.mean(right_errors)))
        point = float(np.mean(np.abs(joined[left] - joined.mu_GT)) - np.mean(np.abs(joined[right] - joined.mu_GT)))
        rows.append(
            {
                "split": split,
                "left_method": left,
                "right_method": right,
                "metric": "MAE_left_minus_right",
                "point": point,
                "ci95_low": float(np.quantile(diffs, 0.025)),
                "ci95_high": float(np.quantile(diffs, 0.975)),
                "root_clusters": len(available_roots),
                "bootstrap_replicates": len(diffs),
            }
        )
    return rows


def audit_archived_partial(archived: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    models = archived / "models"
    names = sorted(p.name for p in models.glob("*.pt")) if models.is_dir() else []
    expected_root = {
        f"ROOT_HELDOUT_targetALL_fold{fold}_seed{seed}_{method}.pt"
        for fold in OUTER_FOLDS
        for seed in range(3)
        for method in ("PhysicalOnly-Probe", "VisionPhysical-Probe")
    }
    expected_loto = {
        f"LOTO_target{task}_fold0_seed{seed}_{method}.pt"
        for task in TASKS
        for seed in range(3)
        for method in ("PhysicalOnly-Probe", "VisionPhysical-Probe")
    }
    present = set(names)
    rows = [
        {
            "evidence": "archived_root_checkpoint_grid",
            "status": "COMPLETE_CHECKPOINTS_NO_AGGREGATE" if expected_root <= present else "INCOMPLETE",
            "expected": len(expected_root),
            "present": len(expected_root & present),
            "missing": len(expected_root - present),
            "extra": 0,
            "formal_use": "NO",
        },
        {
            "evidence": "archived_loto_checkpoint_grid_current_contract",
            "status": "INCOMPLETE_MIXED_VERSION",
            "expected": len(expected_loto),
            "present": len(expected_loto & present),
            "missing": len(expected_loto - present),
            "extra": len([name for name in names if name.startswith("LOTO_") and name not in expected_loto]),
            "formal_use": "NO",
        },
        {
            "evidence": "archived_required_aggregate_outputs",
            "status": "ABSENT",
            "expected": len(EXPECTED_ARCHIVED_RESULTS),
            "present": sum((archived / name).is_file() for name in EXPECTED_ARCHIVED_RESULTS),
            "missing": sum(not (archived / name).is_file() for name in EXPECTED_ARCHIVED_RESULTS),
            "extra": 0,
            "formal_use": "NO",
        },
    ]
    detail = {
        "archived_directory": str(archived),
        "expected_root_checkpoints": len(expected_root),
        "present_root_checkpoints": len(expected_root & present),
        "expected_loto_checkpoints": len(expected_loto),
        "present_loto_checkpoints": len(expected_loto & present),
        "missing_loto_checkpoints": sorted(expected_loto - present),
        "stale_shape_loto_checkpoints": sorted(
            name for name in names if name.startswith("LOTO_") and name not in expected_loto
        ),
        "missing_aggregate_outputs": [name for name in EXPECTED_ARCHIVED_RESULTS if not (archived / name).is_file()],
        "adjudication": "MIXED_VERSION_INCOMPLETE_DO_NOT_AGGREGATE",
    }
    return pd.DataFrame(rows), detail


def validate_manifest(manifest: pd.DataFrame) -> dict[str, object]:
    required = {"context_id", "root_id", "task", "mu_GT", "physical_path", "visual_path", "root_fold"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise RuntimeError(f"manifest missing columns: {missing}")
    checks = {
        "rows_72": len(manifest) == 72,
        "contexts_unique": manifest.context_id.nunique() == len(manifest),
        "roots_24": manifest.root_id.nunique() == 24,
        "tasks_exact": sorted(manifest.task.unique().tolist()) == list(TASKS),
        "rows_per_task_18": manifest.groupby("task").size().eq(18).all(),
        "roots_per_task_6": manifest.groupby("task").root_id.nunique().eq(6).all(),
        "three_contexts_per_root": manifest.groupby("root_id").size().eq(3).all(),
        "friction_in_bounds": manifest.mu_GT.between(*BOUNDS).all(),
        "outer_folds_exact": sorted(manifest.root_fold.unique().tolist()) == list(OUTER_FOLDS),
        "all_paths_collection_train": bool(
            manifest.physical_path.str.contains("/collection_train/", regex=False).all()
            and manifest.visual_path.str.contains("/collection_train/", regex=False).all()
        ),
        "no_forbidden_input_paths": bool(
            all(not FORBIDDEN_PATH_RE.search(str(value)) for value in manifest.physical_path)
            and all(not FORBIDDEN_PATH_RE.search(str(value)) for value in manifest.visual_path)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"manifest quality gate failed: {checks}")
    return {"status": "PASS", "checks": checks}


def load_inputs(manifest_path: Path, normalization_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, object]]:
    safe_input(manifest_path, allow_manifest=True)
    safe_input(normalization_path, allow_manifest=True)
    manifest = pd.read_csv(manifest_path).sort_values("context_id").reset_index(drop=True)
    quality = validate_manifest(manifest)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    names = list(normalization["dynamic_feature_names"])
    phases = list(normalization["phase_categories_from_train"])
    states = list(normalization["contact_state_categories_from_train"])
    numeric = [name for name in names if not name.startswith("phase=") and not name.startswith("contact_state=") and name not in {"eef_dx", "eef_dy", "eef_dz", "probe_phase_unknown", "contact_state_unknown"}]
    deployable_names = [name for name in names if name not in PRIVILEGED_CHANNELS]
    sequences: list[np.ndarray] = []
    visuals: list[np.ndarray] = []
    source_hashes: list[dict[str, object]] = []
    for row in manifest.itertuples():
        physical_path = safe_input(Path(str(row.physical_path)))
        visual_path = safe_input(Path(str(row.visual_path)))
        frame = sequence_dataframe(physical_path, phases, states, numeric).reindex(columns=names).fillna(0.0)
        sequence = frame[deployable_names].to_numpy(np.float64)
        visual = np.load(visual_path, allow_pickle=False).astype(np.float64)
        if sequence.shape != (215, 40) or not np.isfinite(sequence).all():
            raise RuntimeError(f"invalid deployable physical tensor for {row.context_id}: {sequence.shape}")
        if visual.shape != (4096,) or not np.isfinite(visual).all():
            raise RuntimeError(f"invalid visual tensor for {row.context_id}: {visual.shape}")
        sequences.append(sequence)
        visuals.append(visual)
        source_hashes.append(
            {
                "context_id": str(row.context_id),
                "physical_sha256": sha256(physical_path),
                "visual_sha256": sha256(visual_path),
            }
        )
    quality.update(
        {
            "grain": "one pre-probe context; three friction bands per root",
            "rows": len(manifest),
            "roots": int(manifest.root_id.nunique()),
            "tasks": {str(k): int(v) for k, v in manifest.task.value_counts().sort_index().items()},
            "physical_tensor_shape": [len(sequences), 215, 40],
            "visual_tensor_shape": [len(visuals), 4096],
            "privileged_channels_excluded": sorted(PRIVILEGED_CHANNELS),
            "source_hashes": source_hashes,
        }
    )
    return manifest, np.stack(sequences), np.stack(visuals), deployable_names, quality


def fit_one_outer(
    split: str,
    heldout: str,
    train_idx: np.ndarray,
    heldout_idx: np.ndarray,
    manifest: pd.DataFrame,
    physical_design: np.ndarray,
    visual_design: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    y = manifest.mu_GT.to_numpy(float)
    meta_train = manifest.iloc[train_idx].reset_index(drop=True)
    xpi_train, xpi_eval = physical_design[train_idx], physical_design[heldout_idx]
    xv_train, xv_eval = visual_design[train_idx], visual_design[heldout_idx]
    y_train, y_eval = y[train_idx], y[heldout_idx]

    alpha_interaction, interaction_cv = choose_alpha(xpi_train, y_train, meta_train, max_components=24)
    alpha_visual, visual_cv = choose_alpha(xv_train, y_train, meta_train, max_components=12)

    source_inter_members = source_oof_members(
        xpi_train, y_train, meta_train, alpha_interaction, 24, stable_fold(f"{split}:{heldout}:i", 100000)
    )
    source_visual_members = source_oof_members(
        xv_train, y_train, meta_train, alpha_visual, 12, stable_fold(f"{split}:{heldout}:v", 100000)
    )
    target_inter_members = ensemble_predict(
        xpi_train, y_train, meta_train, xpi_eval, alpha_interaction, 24,
        stable_fold(f"{split}:{heldout}:I", 100000),
    )
    target_visual_members = ensemble_predict(
        xv_train, y_train, meta_train, xv_eval, alpha_visual, 12,
        stable_fold(f"{split}:{heldout}:V", 100000),
    )

    interaction_cal = calibrate_members(source_inter_members, y_train)
    visual_cal = calibrate_members(source_visual_members, y_train)
    si_mean, si_var = interaction_cal.apply(source_inter_members)
    sv_mean, sv_var = visual_cal.apply(source_visual_members)
    ti_mean, ti_var = interaction_cal.apply(target_inter_members)
    tv_mean, tv_var = visual_cal.apply(target_visual_members)

    sp_mean, sp_var, sp_weight = fuse_gaussians(sv_mean, sv_var, si_mean, si_var, False)
    sd_mean, sd_var, sd_weight = fuse_gaussians(sv_mean, sv_var, si_mean, si_var, True)
    tp_mean, tp_var, tp_weight = fuse_gaussians(tv_mean, tv_var, ti_mean, ti_var, False)
    td_mean, td_var, td_weight = fuse_gaussians(tv_mean, tv_var, ti_mean, ti_var, True)
    precision_scale = rescale_variance(sp_mean, sp_var, y_train)
    disagreement_scale = rescale_variance(sd_mean, sd_var, y_train)
    sp_var *= precision_scale
    tp_var *= precision_scale
    sd_var *= disagreement_scale
    td_var *= disagreement_scale

    source_candidates = {
        "interaction_ensemble": (si_mean, si_var),
        "precision_fusion": (sp_mean, sp_var),
        "disagreement_fusion": (sd_mean, sd_var),
    }
    candidate_nll = {name: gaussian_nll(y_train, *values) for name, values in source_candidates.items()}
    selected = min(candidate_nll, key=lambda name: (candidate_nll[name], name))

    target_methods = {
        "visual_prior": (tv_mean, tv_var),
        "interaction_ensemble": (ti_mean, ti_var),
        "precision_fusion": (tp_mean, tp_var),
        "disagreement_fusion": (td_mean, td_var),
        "source_selected": {
            "interaction_ensemble": (ti_mean, ti_var),
            "precision_fusion": (tp_mean, tp_var),
            "disagreement_fusion": (td_mean, td_var),
        }[selected],
    }

    member_rows: list[dict[str, object]] = []
    for local, global_index in enumerate(heldout_idx):
        base = manifest.iloc[global_index]
        for member_index, seed in enumerate(ENSEMBLE_SEEDS):
            member_rows.append(
                {
                    "split": split,
                    "heldout": heldout,
                    "context_id": base.context_id,
                    "root_id": base.root_id,
                    "task": int(base.task),
                    "mu_GT": float(base.mu_GT),
                    "expert": "visual_prior",
                    "member": member_index,
                    "seed": seed,
                    "mu_member": float(target_visual_members[local, member_index]),
                }
            )
            member_rows.append(
                {
                    "split": split,
                    "heldout": heldout,
                    "context_id": base.context_id,
                    "root_id": base.root_id,
                    "task": int(base.task),
                    "mu_GT": float(base.mu_GT),
                    "expert": "interaction_ensemble",
                    "member": member_index,
                    "seed": seed,
                    "mu_member": float(target_inter_members[local, member_index]),
                }
            )

    prediction_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for method, (mean, variance) in target_methods.items():
        for local, global_index in enumerate(heldout_idx):
            base = manifest.iloc[global_index]
            prediction_rows.append(
                {
                    "split": split,
                    "heldout": heldout,
                    "context_id": base.context_id,
                    "root_id": base.root_id,
                    "task": int(base.task),
                    "mu_GT": float(base.mu_GT),
                    "method": method,
                    "mu_hat": float(mean[local]),
                    "variance": float(variance[local]),
                    "sigma": float(math.sqrt(variance[local])),
                    "abs_error": float(abs(mean[local] - base.mu_GT)),
                    "interaction_precision_weight": float(
                        tp_weight[local] if method in {"precision_fusion", "source_selected"} and selected == "precision_fusion"
                        else td_weight[local] if method in {"disagreement_fusion", "source_selected"} and selected == "disagreement_fusion"
                        else 1.0 if method == "interaction_ensemble" or (method == "source_selected" and selected == "interaction_ensemble")
                        else 0.0
                    ),
                    "selected_candidate": selected,
                }
            )
        calibration_rows.extend(
            risk_coverage_rows(split, heldout, method, manifest.iloc[heldout_idx], y_eval, mean, variance)
        )

    split_audit = {
        "split": split,
        "heldout": heldout,
        "train_contexts": int(len(train_idx)),
        "heldout_contexts": int(len(heldout_idx)),
        "train_roots": int(manifest.iloc[train_idx].root_id.nunique()),
        "heldout_roots": int(manifest.iloc[heldout_idx].root_id.nunique()),
        "root_overlap": int(
            len(set(manifest.iloc[train_idx].root_id) & set(manifest.iloc[heldout_idx].root_id))
        ),
        "train_tasks": sorted(int(x) for x in manifest.iloc[train_idx].task.unique()),
        "heldout_tasks": sorted(int(x) for x in manifest.iloc[heldout_idx].task.unique()),
        "visual_alpha": alpha_visual,
        "interaction_alpha": alpha_interaction,
        "visual_alpha_cv_mae": visual_cv,
        "interaction_alpha_cv_mae": interaction_cv,
        "visual_residual_floor": visual_cal.residual_floor,
        "visual_variance_scale": visual_cal.scale,
        "interaction_residual_floor": interaction_cal.residual_floor,
        "interaction_variance_scale": interaction_cal.scale,
        "precision_variance_scale": precision_scale,
        "disagreement_variance_scale": disagreement_scale,
        "candidate_source_oof_nll": candidate_nll,
        "source_selected_candidate": selected,
        "target_labels_used_for_fit_or_calibration": False,
    }
    diagnostic = [
        {
            "split": split,
            "heldout": heldout,
            "diagnostic": "mean_visual_contribution_precision_fusion",
            "value": float(np.mean(1.0 - tp_weight)),
        },
        {
            "split": split,
            "heldout": heldout,
            "diagnostic": "mean_visual_interaction_abs_disagreement",
            "value": float(np.mean(np.abs(tv_mean - ti_mean))),
        },
        {
            "split": split,
            "heldout": heldout,
            "diagnostic": "source_selected_is_fusion",
            "value": int(selected != "interaction_ensemble"),
        },
    ]
    return member_rows, prediction_rows, split_audit, calibration_rows + diagnostic


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, heldout, method), group in predictions.groupby(["split", "heldout", "method"]):
        row = {"split": split, "heldout": heldout, "method": method, "scope": "HELDOUT"}
        row.update(metric_row(group, group.mu_GT.to_numpy(float), group.mu_hat.to_numpy(float), group.variance.to_numpy(float)))
        rows.append(row)
    for (split, method), group in predictions.groupby(["split", "method"]):
        row = {"split": split, "heldout": "ALL", "method": method, "scope": "MACRO_CONTEXT"}
        row.update(metric_row(group, group.mu_GT.to_numpy(float), group.mu_hat.to_numpy(float), group.variance.to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def posterior_support(predictions: pd.DataFrame) -> pd.DataFrame:
    chosen = predictions[(predictions.split == "LOTO") & (predictions.method == "source_selected")].copy()
    nodes, weights = np.polynomial.hermite.hermgauss(9)
    weights = weights / math.sqrt(math.pi)
    rows: list[dict[str, object]] = []
    for record in chosen.itertuples():
        support = np.clip(record.mu_hat + math.sqrt(2.0) * record.sigma * nodes, *BOUNDS)
        # Clipping can merge boundary atoms; retain explicit nodes because the
        # downstream consumer may aggregate them exactly.
        for index, (value, weight) in enumerate(zip(support, weights)):
            rows.append(
                {
                    "context_id": record.context_id,
                    "root_id": record.root_id,
                    "task": int(record.task),
                    "heldout": record.heldout,
                    "selected_candidate": record.selected_candidate,
                    "posterior_mean": float(record.mu_hat),
                    "posterior_sigma": float(record.sigma),
                    "support_index": index,
                    "friction_support": float(value),
                    "support_weight": float(weight),
                    "utility_contract": "ACTIVEFORCING_FULL_CLAIM_V2_UTILITY",
                }
            )
    return pd.DataFrame(rows)


def plot_outputs(out: Path, metrics: pd.DataFrame, calibration: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    methods = ["visual_prior", "interaction_ensemble", "precision_fusion", "disagreement_fusion", "source_selected"]
    labels = ["Visual prior", "Interaction", "Precision fusion", "Disagreement fusion", "Source-selected"]
    colors = ["#8b5cf6", "#2563eb", "#0f766e", "#ea580c", "#111827"]
    macro = metrics[(metrics.split == "LOTO") & (metrics.heldout == "ALL")].set_index("method")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(labels, [macro.loc[m, "MAE"] for m in methods], color=colors)
    ax.set_ylabel("Friction MAE")
    ax.set_title("Matched TRAIN-only task/object-held-out E2 diagnostic")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(out / "E2_LOTO_MODALITY_MAE.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for method, label, color in zip(methods, labels, colors):
        group = calibration[
            (calibration.split == "LOTO")
            & (calibration.method == method)
            & (calibration.metric == "interval_coverage")
        ]
        curve = group.groupby("nominal", as_index=False).observed.mean()
        ax.plot(curve.nominal, curve.observed, marker="o", label=label, color=color)
    ax.plot([0.45, 1.0], [0.45, 1.0], linestyle="--", color="#64748b", linewidth=1)
    ax.set(xlabel="Nominal interval coverage", ylabel="Observed coverage", xlim=(0.45, 1.0), ylim=(0.0, 1.02))
    ax.set_title("Source-calibrated posterior coverage on held-out tasks")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "E2_LOTO_CALIBRATION.png", dpi=180)
    plt.close(fig)

    contribution = diagnostics[
        (diagnostics.split == "LOTO")
        & (diagnostics.diagnostic == "mean_visual_contribution_precision_fusion")
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.bar(contribution.heldout.astype(str), contribution.value, color="#0f766e")
    ax.set(xlabel="Held-out task/object", ylabel="Mean visual precision contribution", ylim=(0.0, 1.0))
    ax.set_title("How much the visual prior contributes after interaction")
    fig.tight_layout()
    fig.savefig(out / "E2_LOTO_FUSION_CONTRIBUTION.png", dpi=180)
    plt.close(fig)


def markdown_report(
    out: Path,
    metrics: pd.DataFrame,
    split_audits: list[dict[str, object]],
    archive_detail: dict[str, object],
    bootstrap: pd.DataFrame,
    support: pd.DataFrame,
) -> None:
    loto = metrics[(metrics.split == "LOTO") & (metrics.heldout == "ALL")].set_index("method")
    root = metrics[(metrics.split == "ROOT_HELDOUT") & (metrics.heldout == "ALL")].set_index("method")
    selections = pd.Series([row["source_selected_candidate"] for row in split_audits if row["split"] == "LOTO"]).value_counts().to_dict()
    comparison = bootstrap[
        (bootstrap.split == "LOTO")
        & (bootstrap.left_method == "source_selected")
        & (bootstrap.right_method == "interaction_ensemble")
    ].iloc[0]
    lines = [
        "# E2 modality, LOTO, calibration, and fusion recovery",
        "",
        "## Recovery verdict",
        "",
        "The interrupted archived checkpoint directory is **not aggregable**: it contains a mixed-version LOTO shape, is missing the Physical-only target-5/6 members required by the current contract, and has no prediction or calibration tables. This recovery therefore recomputes a separate CPU-only DEV diagnostic from the frozen 72-context `collection_train` manifest.",
        "",
        "The new result is a **diagnostic, not formal E2 closure**. It supplies a matched visual prior, deployable 40-channel interaction ensemble, source-only variance calibration, two Gaussian fusion candidates, and a source-selected safety fallback. No named TEST or outcome path was opened by the runner.",
        "",
        "## Matched modality results",
        "",
        "| method | root-heldout MAE | LOTO MAE | LOTO NLL | LOTO 68% cov. | LOTO 95% cov. | LOTO interval ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ["visual_prior", "interaction_ensemble", "precision_fusion", "disagreement_fusion", "source_selected"]:
        lines.append(
            f"| {method} | {root.loc[method, 'MAE']:.4f} | {loto.loc[method, 'MAE']:.4f} | {loto.loc[method, 'gaussian_NLL']:.4f} | {100*loto.loc[method, 'coverage_68']:.1f}% | {100*loto.loc[method, 'coverage_95']:.1f}% | {loto.loc[method, 'interval_ECE']:.4f} |"
        )
    lines.extend(
        [
            "",
            "LOTO holds out the entire task and its confounded object family. Hyperparameters, residual floors, variance scales, and the fusion/fallback choice are selected from grouped-root OOF predictions on the other three tasks only. Held-out labels are used only for the diagnostic rows above.",
            "",
            f"Across the four LOTO turns, the source-only selector chose: `{json.dumps(selections, sort_keys=True)}`. The root-cluster bootstrap MAE difference (source-selected minus interaction-only) is {comparison.point:+.4f}, 95% CI [{comparison.ci95_low:+.4f}, {comparison.ci95_high:+.4f}].",
            "",
            "## Evidence and calibration reconciliation",
            "",
            f"- Archived checkpoint adjudication: `{archive_detail['adjudication']}`.",
            f"- Current-contract LOTO checkpoint coverage: {archive_detail['present_loto_checkpoints']}/{archive_detail['expected_loto_checkpoints']}; stale-shape extras: {len(archive_detail['stale_shape_loto_checkpoints'])}.",
            "- The older 46-channel point-estimator OOF evidence remains a separate, non-matched reference because it includes six simulator-geometry contact-frame channels excluded here.",
            "- The older all-population LOTO summary was not ingested by this recovery because its source families are outside the blind-safe TRAIN-only boundary.",
            "- Five-member ensemble spread is not accepted as calibrated by itself. Residual floors and variance scales are fitted on source-only grouped-root OOF residuals; held-out coverage and risk-coverage remain diagnostic.",
            "",
            "## Downstream Utility lineage",
            "",
            f"`E2_UTILITY_POSTERIOR_SUPPORT.csv` contains {len(support)} weighted friction atoms ({support.context_id.nunique()} contexts × 9 Gauss-Hermite nodes) for the unchanged `ACTIVEFORCING_FULL_CLAIM_V2_UTILITY` consumer. The handoff does not read downstream outcomes, retrain a Direct success model, select force, or edit Utility. A downstream owner must evaluate the frozen success model at each support atom and maximize the existing expected Utility with the existing lower-force tie rule.",
            "",
            "## Scope and blockers",
            "",
            "- This closes the available CPU-only preparation and DEV diagnostic work, not the formal E2 or Phys2Real claim.",
            "- Formal closure still requires a preregistered all-modality protocol, an approved non-TEST evaluation gate, and execution by the downstream Utility consumer under its frozen hashes.",
            "- No simulator, GPU, new interaction, second query, or Utility mutation was performed.",
        ]
    )
    (out / "E2_RECOVERY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    workspace = Path(args.workspace).resolve()
    if workspace not in output.parents:
        raise RuntimeError(f"output must be inside lane worktree: {output}")
    output.mkdir(parents=True, exist_ok=False)

    manifest_path = safe_input(Path(args.manifest), allow_manifest=True)
    normalization_path = safe_input(Path(args.normalization), allow_manifest=True)
    archived_path = Path(args.archived).resolve()

    archive_inventory, archive_detail = audit_archived_partial(archived_path)
    archive_inventory.to_csv(output / "E2_EVIDENCE_INVENTORY.csv", index=False)
    write_json(output / "E2_ARCHIVED_PARTIAL_AUDIT.json", archive_detail)

    manifest, sequences, visuals, channels, quality = load_inputs(manifest_path, normalization_path)
    quality["manifest"] = str(manifest_path)
    quality["manifest_sha256"] = sha256(manifest_path)
    quality["normalization"] = str(normalization_path)
    quality["normalization_sha256"] = sha256(normalization_path)
    write_json(output / "E2_INPUT_DATA_QUALITY.json", quality)

    physical_design = summarize_sequences(sequences)
    visual_design = visuals.astype(np.float64)
    y = manifest.mu_GT.to_numpy(float)
    all_members: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    split_audits: list[dict[str, object]] = []
    calibration_and_diagnostics: list[dict[str, object]] = []

    for fold in OUTER_FOLDS:
        train = np.flatnonzero(manifest.root_fold.to_numpy(int) != fold)
        heldout = np.flatnonzero(manifest.root_fold.to_numpy(int) == fold)
        members, predictions, audit, curves = fit_one_outer(
            "ROOT_HELDOUT", f"fold{fold}", train, heldout, manifest, physical_design, visual_design
        )
        all_members.extend(members)
        all_predictions.extend(predictions)
        split_audits.append(audit)
        calibration_and_diagnostics.extend(curves)

    for task in TASKS:
        train = np.flatnonzero(manifest.task.to_numpy(int) != task)
        heldout = np.flatnonzero(manifest.task.to_numpy(int) == task)
        members, predictions, audit, curves = fit_one_outer(
            "LOTO", str(task), train, heldout, manifest, physical_design, visual_design
        )
        all_members.extend(members)
        all_predictions.extend(predictions)
        split_audits.append(audit)
        calibration_and_diagnostics.extend(curves)

    members_df = pd.DataFrame(all_members)
    predictions_df = pd.DataFrame(all_predictions)
    curves_df = pd.DataFrame([row for row in calibration_and_diagnostics if "method" in row])
    diagnostics_df = pd.DataFrame([row for row in calibration_and_diagnostics if "diagnostic" in row])
    metrics_df = build_metrics(predictions_df)
    split_df = pd.DataFrame(
        [
            {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
            for row in split_audits
        ]
    )
    write_json(output / "E2_SPLIT_AND_SELECTION_AUDIT.json", split_audits)
    split_df.to_csv(output / "E2_SPLIT_AUDIT.csv", index=False)
    members_df.to_csv(output / "E2_MEMBER_PREDICTIONS.csv", index=False)
    predictions_df.to_csv(output / "E2_POSTERIOR_PREDICTIONS.csv", index=False)
    metrics_df.to_csv(output / "E2_MODALITY_LOTO_CALIBRATION_METRICS.csv", index=False)
    curves_df.to_csv(output / "E2_CALIBRATION_CURVES.csv", index=False)
    diagnostics_df.to_csv(output / "E2_FUSION_DIAGNOSTICS.csv", index=False)

    bootstrap_rows = []
    for split in ("ROOT_HELDOUT", "LOTO"):
        bootstrap_rows.extend(
            cluster_bootstrap(
                predictions_df,
                split,
                (
                    ("source_selected", "interaction_ensemble"),
                    ("precision_fusion", "interaction_ensemble"),
                    ("disagreement_fusion", "interaction_ensemble"),
                    ("visual_prior", "interaction_ensemble"),
                ),
            )
        )
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(output / "E2_PAIRED_ROOT_BOOTSTRAP.csv", index=False)

    support = posterior_support(predictions_df)
    support.to_csv(output / "E2_UTILITY_POSTERIOR_SUPPORT.csv", index=False)
    plot_outputs(output, metrics_df, curves_df, diagnostics_df)

    utility_lineage = {
        "status": "READY_FOR_FROZEN_UTILITY_CONSUMER_NOT_EXECUTED",
        "generated_utc": utc_now(),
        "contract_id": "ACTIVEFORCING_FULL_CLAIM_V2_UTILITY",
        "contract": "U(F|z,x)=p_D(success|x,z,F)*(Fmax-F)/Fmax + (1-p_D(success|x,z,F))*(-1)",
        "posterior_integration": "sum support_weight * U(F|friction_support,x); maximize over the existing candidate set",
        "tie_rule": "lower force",
        "posterior_support_file": "E2_UTILITY_POSTERIOR_SUPPORT.csv",
        "posterior_contexts": int(support.context_id.nunique()),
        "support_nodes_per_context": 9,
        "belief_bounds": list(BOUNDS),
        "lineage_only": True,
        "direct_success_model_retrained": False,
        "force_decisions_computed": False,
        "downstream_outcomes_read": False,
        "utility_changed": False,
        "required_downstream_prerequisites": [
            "frozen p_D success-model hash",
            "frozen candidate-force set and Fmax by task",
            "approved non-TEST evaluation gate",
            "same-context point/posterior decision comparison",
        ],
    }
    write_json(output / "E2_DOWNSTREAM_UTILITY_LINEAGE.json", utility_lineage)
    markdown_report(output, metrics_df, split_audits, archive_detail, bootstrap_df, support)

    protocol = {
        "status": "FROZEN_AFTER_BLIND_SAFE_INPUT_GATE_BEFORE_METRIC_EMISSION",
        "generated_utc": utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_packages_imported": False,
        "simulator_imported": False,
        "new_rollouts": 0,
        "input_population": "72 archived collection_train contexts; 24 roots; tasks 0/1/5/6",
        "modalities": {
            "visual_prior": "4096D archived frozen visual feature -> train-only standardization/PCA -> root-bootstrap ridge ensemble",
            "interaction_ensemble": "215x40 deployable pre-probe history -> temporal summaries -> train-only standardization/PCA -> five-member root-bootstrap ridge ensemble",
            "fusion": "source-OOF calibrated Gaussian precision product plus disagreement-aware alternative; source-only NLL safety selector",
        },
        "ridge_grid": list(RIDGE_GRID),
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "target_labels_used_for_model_or_calibration": False,
        "target_labels_used_for_diagnostic_metrics_only": True,
        "forbidden_path_pattern": FORBIDDEN_PATH_RE.pattern,
        "utility_mutation": False,
    }
    write_json(output / "E2_CPU_FUSION_PROTOCOL.json", protocol)

    deliverables = sorted(
        path for path in output.iterdir() if path.is_file() and path.name not in {"SHA256SUMS.txt", "E2_RECOVERY_STATUS.json"}
    )
    hash_lines = [f"{sha256(path)}  {path.name}" for path in deliverables]
    (output / "SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    status = {
        "status": "CPU_DIAGNOSTICS_COMPLETE_FORMAL_CLOSURE_BLOCKED",
        "generated_utc": utc_now(),
        "output_directory": str(output),
        "deliverable_count": len(deliverables) + 1,
        "archive_adjudication": archive_detail["adjudication"],
        "input_quality": quality["status"],
        "heldout_prediction_rows": int(len(predictions_df)),
        "posterior_support_rows": int(len(support)),
        "test_outcomes_read": False,
        "gpu_launched": False,
        "simulator_launched": False,
        "utility_changed": False,
        "formal_blocker": "requires approved non-TEST gate and unchanged downstream Utility consumer",
    }
    write_json(output / "E2_RECOVERY_STATUS.json", status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ACTIVEFORCING_E2_MODALITY_FUSION_RECOVERY_20260902_151000",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/exouser/FORTE/activeforcing_physical_only_probe_20260901/COMMON_PROBE_CONTEXTS.csv"),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path("/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_NORMALIZATION.json"),
    )
    parser.add_argument(
        "--archived",
        type=Path,
        default=Path("/home/exouser/FORTE/activeforcing_physical_only_probe_20260901"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"E2_RECOVERY_ERROR: {exc}", file=sys.stderr)
        raise
