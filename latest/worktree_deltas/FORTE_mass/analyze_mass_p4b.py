#!/usr/bin/env python3
"""CPU-only root-heldout identifiability analysis for the frozen P4-B query."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


OUT = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500")
DATA = OUT / "M1_TASK0_P4B/M1_TASK0_P4B_EPISODES.csv"
FEATURE_GROUPS = {
    "force_torque": ["f_meas_mean", "normal_force_mean", "normal_force_peak", "normal_force_hyst", "ftan_mean", "ftan_peak", "ftan_hyst", "rho_mean", "rho_peak", "rho_hyst", "rho_impulse", "imb_mean", "imb_peak", "imb_ratio_mean", "imb_ratio_peak"],
    "position_motion": ["actual_probe_displacement_mm", "probe_path_mm", "probe_duration_s", "obj_disp_probe_m", "obj_rot_probe_rad", "major_disturbance"],
    "gripper_response": ["preprobe_normal_force", "preprobe_marker_motion", "preprobe_tangential_force", "gripper_opening_mean"],
    "marker_tactile": ["marker_mean", "marker_peak", "marker_tangential_mean", "marker_tangential_peak", "marker_hyst", "marker_unloading", "marker_vel_abs_peak", "residual_marker_displacement"],
}
FEATURE_GROUPS["all_history"] = sorted(set(sum(FEATURE_GROUPS.values(), [])))


def read_rows() -> list[dict[str, str]]:
    with DATA.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
        return value if np.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def pairwise_accuracy(y: np.ndarray, pred: np.ndarray, roots: np.ndarray) -> float:
    correct = total = 0
    for root in np.unique(roots):
        idx = np.flatnonzero(roots == root)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                true_diff = y[idx[i]] - y[idx[j]]
                pred_diff = pred[idx[i]] - pred[idx[j]]
                if true_diff == 0:
                    continue
                total += 1
                correct += int(np.sign(true_diff) == np.sign(pred_diff))
    return float(correct / total) if total else float("nan")


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    a, b = rankdata(y), rankdata(pred)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fit_predict_ridge(x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[scale < 1e-12] = 1.0
    xt = (x_train - mean) / scale
    xa = (x_all - mean) / scale
    design = np.column_stack([np.ones(len(xt)), xt])
    reg = np.eye(design.shape[1]) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + reg, design.T @ y_train)
    return np.column_stack([np.ones(len(xa)), xa]) @ beta


def main() -> int:
    rows = read_rows()
    train = np.array([r["split"] == "DEV" for r in rows])
    test = ~train
    y = np.asarray([f(r, "mass_kg") for r in rows])
    roots = np.asarray([r["seed_idx"] for r in rows])
    output_rows: list[dict[str, object]] = []
    predictions: dict[str, list[dict[str, object]]] = {}
    for group, cols in FEATURE_GROUPS.items():
        x = np.asarray([[f(r, col) for col in cols] for r in rows], dtype=float)
        pred = fit_predict_ridge(x[train], y[train], x, alpha=1e-3)
        yt, pt = y[test], pred[test]
        ae = np.abs(pt - yt)
        rho = spearman(yt, pt)
        band_true = np.argmin(np.abs(yt[:, None] - np.array([0.05, 0.10, 0.20])[None, :]), axis=1)
        band_pred = np.argmin(np.abs(pt[:, None] - np.array([0.05, 0.10, 0.20])[None, :]), axis=1)
        output_rows.append({
            "scope": "task0_root_heldout",
            "feature_group": group,
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "mass_mae_kg": float(np.mean(ae)),
            "mass_median_ae_kg": float(np.median(ae)),
            "spearman": rho,
            "pairwise_ranking_accuracy": pairwise_accuracy(yt, pt, roots[test]),
            "mass_band_accuracy": float(np.mean(band_true == band_pred)),
        })
        predictions[group] = [
            {"trial_id": rows[i]["trial_id"], "root_seed": int(roots[i]), "mass_band": rows[i]["mass_band"], "mass_kg": float(y[i]), "pred_mass_kg": float(pred[i]), "split": rows[i]["split"]}
            for i in np.flatnonzero(test)
        ]

    table = OUT / "MASS_QUERY_P4B_IDENTIFIABILITY.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as fh:
        fields = list(output_rows[0])
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    (OUT / "MASS_QUERY_P4B_PREDICTIONS.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")

    best = max(output_rows, key=lambda r: (float(r["pairwise_ranking_accuracy"]), -float(r["mass_mae_kg"])))
    usable = bool(float(best["mass_mae_kg"]) <= 0.04 and float(best["pairwise_ranking_accuracy"]) >= 0.80 and float(best["mass_band_accuracy"]) >= 0.80)
    report = f"""# Mass query identifiability report

Status: **{'P4B_USABLE_FOR_MASS' if usable else 'P4B_INSUFFICIENT_REQUIRES_FROZEN_MASS_QUERY'}**

## Protocol

- Query: frozen P4-B contact-conditioned shear.
- Fixed nuisance variables: task0 alphabet soup, friction 0.5, appearance, geometry, initial-root generation and controller unchanged.
- Mass bands: LOW 0.05 kg, MID 0.10 kg, HIGH 0.20 kg.
- Development roots: 6100–6105; heldout roots: 6200–6201.
- No downstream force-branch or full-task outcome was read.
- Estimator: standardized Ridge on development roots only; no outcome labels.

## Heldout result

| Feature group | MAE kg | Median AE kg | Spearman | Pairwise ranking | Band accuracy |
|---|---:|---:|---:|---:|---:|
"""
    for r in output_rows:
        report += f"| {r['feature_group']} | {float(r['mass_mae_kg']):.5f} | {float(r['mass_median_ae_kg']):.5f} | {float(r['spearman']):.3f} | {float(r['pairwise_ranking_accuracy']):.3f} | {float(r['mass_band_accuracy']):.3f} |\n"
    report += f"""
The pre-registered usability gate is MAE ≤ 0.04 kg, pairwise ranking ≥ 0.80,
and mass-band accuracy ≥ 0.80 on heldout roots. Best group: **{best['feature_group']}**.
The machine-readable values and heldout predictions are in
`MASS_QUERY_P4B_IDENTIFIABILITY.csv` and `MASS_QUERY_P4B_PREDICTIONS.json`.

## Interpretation

This test answers observability, not task usefulness. A usable result means
P4-B can remain the mass query; it does not establish that mass changes the
downstream force-optimal policy. If the gate is not met, a separate short
vertical-load query must be designed on development roots and then validated
on fresh heldout roots before any formal benchmark collection.
"""
    (OUT / "MASS_QUERY_IDENTIFIABILITY_REPORT.md").write_text(report, encoding="utf-8")
    protocol = {
        "query": "P4-B frozen contact-conditioned shear",
        "mass_bands_kg": {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20},
        "development_roots": [6100, 6101, 6102, 6103, 6104, 6105],
        "heldout_roots": [6200, 6201],
        "fixed_friction": 0.5,
        "estimator": "StandardScaler + Ridge(alpha=1e-3), fit only on DEV roots",
        "usability_gate": {"mass_mae_kg_max": 0.04, "pairwise_ranking_accuracy_min": 0.80, "mass_band_accuracy_min": 0.80},
        "best_feature_group": best["feature_group"],
        "status": "P4B_USABLE_FOR_MASS" if usable else "P4B_INSUFFICIENT_REQUIRES_FROZEN_MASS_QUERY",
        "downstream_outcomes_read": False,
        "source_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
    }
    (OUT / "MASS_QUERY_PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
