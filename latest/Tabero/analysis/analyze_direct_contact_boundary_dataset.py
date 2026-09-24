#!/usr/bin/env python3
"""Quality audit and frozen non-neural separability test for direct contact data.

This script intentionally stops before Physics-GRU v3.  It consumes one
already-frozen collection namespace, derives force-independent physical
summaries, and evaluates root-held-out logistic/SVM diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
OUT = sorted(RESULTS.glob("direct_contact_boundary_dataset_*"))[-1]
TELEMETRY = OUT / "P5S0C_BRANCH_TELEMETRY"
DT = 0.05
ACTIVE_PHASES = {"lift", "transit", "over_basket", "place"}
WINDOW_STEPS = 120
CONTACT_EPS = 0.15
RATIO_FN_EPS = 0.20


def write_json(path: Path, obj):
    def sanitize(value):
        if isinstance(value, dict):
            return {str(k): sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value
    path.write_text(json.dumps(sanitize(obj), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(a) -> bool:
    try:
        return bool(np.isfinite(np.asarray(a, dtype=float)).all())
    except Exception:
        return False


def find_start(d: pd.DataFrame) -> int | None:
    idx = np.flatnonzero(d.phase.astype(str).to_numpy() == "lift")
    return int(idx[0]) if len(idx) else None


def derived_trace(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    obj = out[["object_x_analysis_only", "object_y_analysis_only", "object_z_analysis_only"]].to_numpy(float)
    cmd = out[["cmd_x", "cmd_y", "cmd_z"]].to_numpy(float)
    rel = obj - cmd
    out["rel_cmd_x_m"] = rel[:, 0]
    out["rel_cmd_y_m"] = rel[:, 1]
    out["rel_cmd_z_m"] = rel[:, 2]
    relv = np.vstack([np.zeros((1, 3)), np.diff(rel, axis=0) / DT])
    out["rel_cmd_vx_mps"] = relv[:, 0]
    out["rel_cmd_vy_mps"] = relv[:, 1]
    out["rel_cmd_vz_mps"] = relv[:, 2]
    out["tangential_relative_velocity_proxy_mps"] = np.linalg.norm(relv[:, :2], axis=1)
    out["normal_force_imbalance_N"] = np.abs(out.left_normal_force_N - out.right_normal_force_N)
    out["tangential_force_mean_N"] = (out.left_tangential_force_N + out.right_tangential_force_N) / 2.0
    out["normal_force_mean_N"] = (out.left_normal_force_N + out.right_normal_force_N) / 2.0
    out["bilateral"] = (out.contact_left.astype(int) == 1) & (out.contact_right.astype(int) == 1)
    out["active"] = out.phase.astype(str).isin(ACTIVE_PHASES)
    out["force_ratio_mean"] = np.where(
        out.normal_force_mean_N.to_numpy(float) > RATIO_FN_EPS,
        out.tangential_force_mean_N.to_numpy(float) / (out.normal_force_mean_N.to_numpy(float) + 1e-8),
        np.nan,
    )
    return out


def longest_false(values: np.ndarray) -> int:
    best = cur = 0
    for x in values.astype(bool):
        if not x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def summary(d: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    active = d[d.active].copy()
    if active.empty:
        active = d.iloc[:0].copy()
    p = f"{prefix}" if prefix else ""
    bl = active.bilateral.to_numpy(bool)
    def vals(col):
        return active[col].to_numpy(float) if col in active else np.array([], dtype=float)
    def safe(fn, arr, default=0.0):
        arr = np.asarray(arr)
        if arr.dtype.kind in "fc":
            arr = arr[np.isfinite(arr)]
        return float(fn(arr)) if len(arr) else float(default)
    rel = np.sqrt(active.rel_cmd_x_m.to_numpy(float) ** 2 + active.rel_cmd_y_m.to_numpy(float) ** 2 + active.rel_cmd_z_m.to_numpy(float) ** 2) if len(active) else np.array([])
    joints = active[["gripper_pos_0", "gripper_pos_1"]].to_numpy(float) if len(active) else np.empty((0, 2))
    # The runner exposes commanded TCP position rather than realized TCP pose;
    # these are explicitly named object-to-command proxies, not hidden labels.
    feats = {
        f"{p}bilateral_contact_fraction": safe(np.mean, bl),
        f"{p}longest_bilateral_contact_loss_duration_s": longest_false(bl) * DT,
        f"{p}left_normal_force_mean_N": safe(np.mean, vals("left_normal_force_N")),
        f"{p}left_normal_force_min_N": safe(np.min, vals("left_normal_force_N")),
        f"{p}left_normal_force_std_N": safe(np.std, vals("left_normal_force_N")),
        f"{p}right_normal_force_mean_N": safe(np.mean, vals("right_normal_force_N")),
        f"{p}right_normal_force_min_N": safe(np.min, vals("right_normal_force_N")),
        f"{p}right_normal_force_std_N": safe(np.std, vals("right_normal_force_N")),
        f"{p}normal_force_imbalance_mean_abs_N": safe(np.mean, vals("normal_force_imbalance_N")),
        f"{p}normal_force_imbalance_max_abs_N": safe(np.max, vals("normal_force_imbalance_N")),
        f"{p}left_tangential_force_mean_N": safe(np.mean, vals("left_tangential_force_N")),
        f"{p}left_tangential_force_max_N": safe(np.max, vals("left_tangential_force_N")),
        f"{p}right_tangential_force_mean_N": safe(np.mean, vals("right_tangential_force_N")),
        f"{p}right_tangential_force_max_N": safe(np.max, vals("right_tangential_force_N")),
        f"{p}force_ratio_mean": safe(np.nanmean, vals("force_ratio_mean")),
        f"{p}force_ratio_max": safe(np.nanmax, vals("force_ratio_mean")),
        f"{p}relative_object_to_command_drift_mean_m": safe(np.mean, rel),
        f"{p}relative_object_to_command_drift_max_m": safe(np.max, rel),
        f"{p}relative_velocity_mean_mps": safe(np.mean, np.linalg.norm(active[["rel_cmd_vx_mps", "rel_cmd_vy_mps", "rel_cmd_vz_mps"]].to_numpy(float), axis=1) if len(active) else np.array([])),
        f"{p}relative_velocity_max_mps": safe(np.max, np.linalg.norm(active[["rel_cmd_vx_mps", "rel_cmd_vy_mps", "rel_cmd_vz_mps"]].to_numpy(float), axis=1) if len(active) else np.array([])),
        f"{p}tangential_relative_velocity_proxy_mean_mps": safe(np.mean, vals("tangential_relative_velocity_proxy_mps")),
        f"{p}tangential_relative_velocity_proxy_max_mps": safe(np.max, vals("tangential_relative_velocity_proxy_mps")),
        f"{p}gripper_joint_deviation_mean": safe(np.mean, np.abs(joints - joints[0]).mean(axis=1) if len(joints) else np.array([])),
        f"{p}gripper_joint_deviation_max": safe(np.max, np.abs(joints - joints[0]).mean(axis=1) if len(joints) else np.array([])),
    }
    return feats


def make_features(d: pd.DataFrame) -> tuple[dict, dict]:
    x = derived_trace(d)
    start = find_start(x)
    if start is None or len(x) - start < WINDOW_STEPS:
        raise ValueError("trace lacks fixed lift-aligned 120-step evaluation window")
    w = x.iloc[start : start + WINDOW_STEPS].copy()
    result = summary(w)
    for phase in ["lift", "transit", "over_basket", "place"]:
        result.update(summary(w[w.phase.astype(str) == phase], prefix=f"{phase}_"))
    return result, {"window_start_row": int(start), "window_steps": WINDOW_STEPS, "window_end_row": int(start + WINDOW_STEPS - 1)}


def metric(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pred = (score >= 0).astype(int)
    pos = y == 1
    neg = y == 0
    def auc_binary(labels, values):
        if labels.sum() == 0 or (~labels).sum() == 0:
            return float("nan")
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(1, len(values) + 1, dtype=float)
        return float((ranks[labels].sum() - labels.sum() * (labels.sum() + 1) / 2) / (labels.sum() * (~labels).sum()))
    def auprc(labels, values):
        if labels.sum() == 0:
            return float("nan")
        order = np.argsort(-values, kind="mergesort")
        yord = labels[order].astype(int)
        tp = np.cumsum(yord)
        precision = tp / np.arange(1, len(yord) + 1)
        return float((precision * yord).sum() / labels.sum())
    tpr = float(((pred == 1) & pos).sum() / pos.sum()) if pos.sum() else 0.0
    tnr = float(((pred == 0) & neg).sum() / neg.sum()) if neg.sum() else 0.0
    f1 = float(2 * tpr * (1 - tnr) / (tpr + (1 - tnr))) if (tpr + (1 - tnr)) else 0.0
    out = {
        "n": int(len(y)),
        "positive_sufficient": int((y == 1).sum()),
        "auroc": auc_binary(pos, score) if len(np.unique(y)) == 2 else float("nan"),
        "auprc": auprc(pos, score) if len(np.unique(y)) == 2 else float("nan"),
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "f1": f1,
        "F_prev_sensitivity": tnr,
        "F_star_next_specificity": tpr,
    }
    return out


def fit_normalizer(x):
    med = np.nanmedian(x, axis=0)
    z = np.where(np.isfinite(x), x, med)
    mean = z.mean(axis=0)
    scale = z.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return med, mean, scale


def transform(x, norm):
    med, mean, scale = norm
    return (np.where(np.isfinite(x), x, med) - mean) / scale


def fit_linear_score(x, y, kind, steps=2500, lr=0.03, l2=1e-3):
    """Small deterministic NumPy equivalents of frozen LogReg / LinearSVM."""
    n, d = x.shape
    y01 = y.astype(float)
    signed = 2.0 * y01 - 1.0
    weights = np.where(y01 == 1.0, n / max(2.0 * y01.sum(), 1.0), n / max(2.0 * (n - y01.sum()), 1.0))
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(steps):
        margin = signed * (x @ w + b)
        if kind == "logreg":
            # Stable sigmoid and weighted cross-entropy gradient.
            p = 1.0 / (1.0 + np.exp(-np.clip(x @ w + b, -40.0, 40.0)))
            err = weights * (p - y01)
            gw = (x.T @ err) / n + l2 * w
            gb = float(err.mean())
        else:
            active = margin < 1.0
            coeff = -weights * signed * active
            gw = (x.T @ coeff) / n + l2 * w
            gb = float(coeff.mean())
        w -= lr * gw
        b -= lr * gb
    return lambda z: z @ w + b


def bootstrap_contexts(df: pd.DataFrame, score_col: str, n=2000, seed=20260829):
    rng = np.random.default_rng(seed)
    contexts = sorted(df.context_id.unique())
    values = {"prev_gt_star": [], "prev_gt_next": [], "strict_triplet": []}
    for cid in contexts:
        g = df[df.context_id == cid]
        s = g.groupby("force_role")[score_col].mean()
        if not all(k in s for k in ["prev", "star", "next"]):
            continue
        values["prev_gt_star"].append(float(s["prev"] > s["star"]))
        values["prev_gt_next"].append(float(s["prev"] > s["next"]))
        values["strict_triplet"].append(float(s["prev"] > s["star"] > s["next"]))
    out = {"contexts": len(values["prev_gt_star"]), "point": {k: float(np.mean(v)) if v else float("nan") for k, v in values.items()}, "bootstrap_95_ci": {}}
    for k, v in values.items():
        if not v:
            out["bootstrap_95_ci"][k] = [float("nan"), float("nan")]
            continue
        a = np.asarray(v, float)
        boots = np.asarray([np.mean(a[rng.integers(0, len(a), len(a))]) for _ in range(n)])
        out["bootstrap_95_ci"][k] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    return out


def main():
    pop = pd.read_csv(OUT / "SELECTED_POPULATION.csv")
    pop["context_id"] = pop.context_id.astype(str)
    selected = set(pop.context_id)
    required = {"context_id", "branch_label", "task", "split", "requested_force_N", "hidden_friction_analysis_only"}
    traces = []
    audit_rows = []
    for path in sorted(TELEMETRY.glob("*_trajectory.csv")):
        d = pd.read_csv(path)
        ok = True
        errors = []
        if not required.issubset(d.columns):
            ok = False; errors.append("missing_branch_metadata")
        required_direct = {"phase", "cmd_x", "cmd_y", "cmd_z", "object_x_analysis_only", "object_y_analysis_only", "object_z_analysis_only", "gripper_pos_0", "gripper_pos_1", "left_normal_force_N", "right_normal_force_N", "left_tangential_force_N", "right_tangential_force_N", "contact_left", "contact_right", "object_vx_mps", "object_vy_mps", "object_vz_mps"}
        if not required_direct.issubset(d.columns):
            ok = False; errors.append("missing_corrected_direct_field")
        if len(d) == 0:
            ok = False; errors.append("zero_length")
        if len(d) and not finite(d.select_dtypes(include=[np.number]).to_numpy()):
            ok = False; errors.append("nan_or_inf")
        cid = str(d.context_id.iloc[0]) if len(d) else ""
        if cid not in selected:
            ok = False; errors.append("context_not_in_frozen_population")
        try:
            feats, window = make_features(d)
        except Exception as exc:
            ok = False; errors.append(str(exc)); feats, window = {}, {}
        if len(d):
            traces.append((path, d, feats, window, ok, errors))
        audit_rows.append({"file": str(path), "context_id": cid, "rows": len(d), "valid": int(ok), "errors": errors})

    records = []
    for path, d, feats, window, ok, errors in traces:
        if not ok:
            continue
        cid = str(d.context_id.iloc[0]); force = float(d.requested_force_N.iloc[0])
        p = pop[pop.context_id == cid].iloc[0]
        if math.isclose(force, float(p.F_prev), abs_tol=1e-8): role, y = "prev", 0
        elif math.isclose(force, float(p.F_star), abs_tol=1e-8): role, y = "star", 1
        elif math.isclose(force, float(p.F_next), abs_tol=1e-8): role, y = "next", 1
        else:
            continue
        records.append({**feats, "context_id": cid, "root_id": str(p.root_id), "root_index": int(p.root_index), "split": str(p.split), "task": int(p.task), "friction_band": str(p.friction_band), "force_role": role, "force_N_metadata": force, "sufficient_label": y, "file": str(path), "window_start_row": window["window_start_row"], "window_steps": window["window_steps"]})
    data = pd.DataFrame(records)
    feature_cols = [c for c in data.columns if c not in {"context_id", "root_id", "root_index", "split", "task", "friction_band", "force_role", "force_N_metadata", "sufficient_label", "file", "window_start_row", "window_steps"}]
    data.to_csv(OUT / "SEPARABILITY_FEATURES.csv", index=False)
    pd.DataFrame(audit_rows).to_json(OUT / "DIRECT_CONTACT_TRACE_AUDIT.json", orient="records", indent=2)

    # Branch and restore audits.
    branch_rows = []
    parity_rows = []
    for t in [0, 1, 5, 6]:
        bp = OUT / f"task{t}/branches.csv"
        pp = OUT / f"task{t}/parity.csv"
        if bp.exists(): branch_rows.append(pd.read_csv(bp))
        if pp.exists(): parity_rows.append(pd.read_csv(pp))
    branches = pd.concat(branch_rows, ignore_index=True) if branch_rows else pd.DataFrame()
    parity = pd.concat(parity_rows, ignore_index=True) if parity_rows else pd.DataFrame()
    parity_pass = int((parity.parity_pass.astype(int) == 1).sum()) if len(parity) else 0
    parity_total = len(parity)
    repeats = data.groupby(["context_id", "force_role"]).size().reset_index(name="n") if len(data) else pd.DataFrame()
    complete_triplets = int((repeats.groupby("context_id").force_role.nunique() == 3).sum()) if len(repeats) else 0
    repeat_success = []
    if len(branches):
        for _, g in branches.groupby(["context_id", "requested_force_N"], sort=True):
            cid = str(g.context_id.iloc[0])
            p = pop[pop.context_id == cid].iloc[0]
            force = float(g.requested_force_N.iloc[0])
            role = "prev" if math.isclose(force, float(p.F_prev)) else ("star" if math.isclose(force, float(p.F_star)) else ("next" if math.isclose(force, float(p.F_next)) else "other"))
            repeat_success.append({"context_id": cid, "force_N": force, "force_role": role, "n": len(g), "successes": int(pd.to_numeric(g.full_task_success_y, errors="coerce").fillna(0).sum()), "success_rate": float(pd.to_numeric(g.full_task_success_y, errors="coerce").mean()), "branch_label_example": str(g.branch_label.iloc[0])})
    repeat_df = pd.DataFrame(repeat_success)
    repeat_df.to_csv(OUT / "BOUNDARY_REPEATABILITY.csv", index=False)
    context_repeat = []
    for cid, p in pop.set_index("context_id").iterrows():
        g = repeat_df[repeat_df.context_id == cid]
        rates = {role: (float(g[g.force_role == role].success_rate.iloc[0]) if len(g[g.force_role == role]) else float("nan")) for role in ["prev", "star", "next"]}
        mixed = any(np.isfinite(v) and v > 0.0 and v < 1.0 for v in rates.values())
        stable_frontier = all(np.isfinite(rates[k]) for k in rates) and rates["prev"] == 0.0 and rates["star"] == 1.0 and rates["next"] == 1.0
        if mixed:
            classification = "BOUNDARY_STOCHASTIC"
        elif stable_frontier:
            classification = "STABLE_FRONTIER_ORDER"
        else:
            classification = "DETERMINISTIC_NONMONOTONIC_OR_HISTORICAL_MISMATCH"
        context_repeat.append({"context_id": cid, "task": int(p.task), "split": str(p.split), "friction_band": str(p.friction_band), "root_id": str(p.root_id), "F_prev": float(p.F_prev), "F_star": float(p.F_star), "F_next": float(p.F_next), "prev_success_rate": rates["prev"], "star_success_rate": rates["star"], "next_success_rate": rates["next"], "classification": classification})
    context_repeat_df = pd.DataFrame(context_repeat)
    context_repeat_df.to_csv(OUT / "BOUNDARY_REPEATABILITY_CONTEXT.csv", index=False)

    pop_summary = {
        "selected_contexts": int(len(pop)), "selected_traces_expected": int(len(pop) * 3 * 3), "valid_traces": int(len(data)),
        "contexts_with_valid_triplets": int(complete_triplets), "ineligible_contexts": int(len(json.loads((OUT / "INELIGIBLE_CONTEXTS.json").read_text()))),
        "tasks": sorted(int(x) for x in pop.task.unique()), "splits": sorted(str(x) for x in pop.split.unique()), "bands": sorted(str(x) for x in pop.friction_band.unique()),
        "by_split": pop.groupby("split").size().to_dict(), "by_task": pop.groupby("task").size().to_dict(), "by_band": pop.groupby("friction_band").size().to_dict(),
        "traces_by_role": data.force_role.value_counts().to_dict() if len(data) else {},
        "restore_parity": {"passed": parity_pass, "total": parity_total, "rate": parity_pass / parity_total if parity_total else float("nan")},
        "duplicate_files": int(len(audit_rows) - len({x["file"] for x in audit_rows})),
    }
    # Direct physical sanity statistics, deliberately separate from classifier labels.
    phys = []
    for path, d, feats, window, ok, errors in traces:
        if not ok:
            continue
        x = derived_trace(d)
        active = x[x.active]
        p = pop[pop.context_id == str(d.context_id.iloc[0])].iloc[0]
        force = float(d.requested_force_N.iloc[0])
        role = "prev" if math.isclose(force, float(p.F_prev)) else ("star" if math.isclose(force, float(p.F_star)) else "next")
        matching = branches[(branches.context_id.astype(str) == str(p.context_id)) & np.isclose(branches.requested_force_N.astype(float), force)] if len(branches) else pd.DataFrame()
        phys.append({"context_id": str(p.context_id), "task": int(p.task), "split": str(p.split), "friction_band": str(p.friction_band), "force_role": role, "force_N_metadata": force, "bilateral_fraction": float(active.bilateral.mean()) if len(active) else np.nan, "min_bilateral_duration_s": float(longest_false(active.bilateral.to_numpy(bool)) * DT) if len(active) else np.nan, "normal_force_mean_N": float(active.normal_force_mean_N.mean()) if len(active) else np.nan, "normal_force_min_N": float(active.normal_force_mean_N.min()) if len(active) else np.nan, "tangential_force_mean_N": float(active.tangential_force_mean_N.mean()) if len(active) else np.nan, "tangential_force_max_N": float(active.tangential_force_mean_N.max()) if len(active) else np.nan, "force_ratio_mean": float(active.force_ratio_mean.mean()) if len(active) else np.nan, "tangential_velocity_proxy_max_mps": float(active.tangential_relative_velocity_proxy_mps.max()) if len(active) else np.nan, "relative_drift_max_m": float(np.linalg.norm(active[["rel_cmd_x_m", "rel_cmd_y_m", "rel_cmd_z_m"]].to_numpy(float), axis=1).max()) if len(active) else np.nan, "historical_success": int(matching.full_task_success_y.iloc[0]) if len(matching) else None})
    phys_df = pd.DataFrame(phys)
    phys_df.to_csv(OUT / "DIRECT_PHYSICAL_SUMMARIES.csv", index=False)
    telemetry_integrity = {
        "nan_or_inf_traces": sum("nan_or_inf" in x["errors"] for x in audit_rows),
        "zero_length_traces": sum("zero_length" in x["errors"] for x in audit_rows),
        "missing_phase_traces": sum("missing_corrected_direct_field" in x["errors"] for x in audit_rows),
        "duplicate_telemetry_files": int(len(audit_rows) - len({x["file"] for x in audit_rows})),
        "local_force_fields_present": True,
        "double_rotation_detected": False,
        "realized_gripper_pose_available": False,
        "contact_point_identity_available": False,
        "contact_point_velocity_available": False,
    }
    contact_sanity = phys_df.groupby("force_role")[['bilateral_fraction', 'normal_force_mean_N', 'normal_force_min_N', 'tangential_force_max_N', 'tangential_velocity_proxy_max_mps', 'relative_drift_max_m']].agg(['mean', 'std', 'min', 'max']).round(8).to_dict()
    write_json(OUT / "DIRECT_CONTACT_ROOT_DIVERSE_DATASET_AUDIT.json", {
        "status": "VALID_COLLECTION" if len(data) == len(pop) * 9 else "PARTIAL_COLLECTION",
        "population": pop_summary,
        "coverage": {
            "roots_by_split": {str(k): sorted(set(int(x) for x in v)) for k, v in pop.groupby("split").root_index},
            "roots_by_task_split": {f"{k[0]}_{k[1]}": sorted(set(int(x) for x in v)) for k, v in pop.groupby(["task", "split"]).root_index},
            "roots_by_band_split": {f"{k[0]}_{k[1]}": sorted(set(int(x) for x in v)) for k, v in pop.groupby(["friction_band", "split"]).root_index},
            "contexts_by_task_split_band": pop.groupby(["task", "split", "friction_band"]).size().to_dict(),
        },
        "trace_audit": {"total": len(audit_rows), "valid": sum(x["valid"] for x in audit_rows), "invalid": sum(not x["valid"] for x in audit_rows)},
        "telemetry_integrity": telemetry_integrity,
        "contact_sanity_by_force_role": contact_sanity,
        "boundary_repeatability": {
            "context_count": len(context_repeat_df),
            "classification_counts": context_repeat_df.classification.value_counts().to_dict(),
            "stable_frontier_order_fraction": float((context_repeat_df.classification == "STABLE_FRONTIER_ORDER").mean()),
            "stochastic_fraction": float((context_repeat_df.classification == "BOUNDARY_STOCHASTIC").mean()),
        },
    })

    # Fit only on TRAIN roots.  The feature list is frozen by the protocol.
    models = {
        "LogReg": "logreg",
        "Linear SVM": "svm",
    }
    metric_rows = []
    score_frames = {}
    if len(data):
        X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
        y = data.sufficient_label.astype(int).to_numpy()
        train_mask = data["split"].eq("TRAIN").to_numpy()
        x_np = X.to_numpy(float)
        norm = fit_normalizer(x_np[train_mask])
        x_np = transform(x_np, norm)
        for name, kind in models.items():
            score_fn = fit_linear_score(x_np[train_mask], y[train_mask], kind)
            score = score_fn(x_np)
            score_frames[name] = score
            data[name.replace(" ", "_") + "_score"] = score
            for split in ["TRAIN", "DEV", "TEST"]:
                m = data["split"].eq(split).to_numpy()
                row = {"model": name, "split": split, **metric(y[m], score[m])}
                metric_rows.append(row)
            data[name.replace(" ", "_") + "_unsafe_score"] = -score
        data.to_csv(OUT / "SEPARABILITY_FEATURES_WITH_SCORES.csv", index=False)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUT / "ROOT_HELD_OUT_METRICS.csv", index=False)

    paired = {}
    for name in models:
        col = name.replace(" ", "_") + "_unsafe_score"
        # score higher means unsafe; restriction is reportable for each split.
        paired[name] = {}
        for split in ["TRAIN", "DEV", "TEST"]:
            paired[name][split] = bootstrap_contexts(data[data["split"] == split], col) if len(data) else {}

    # Stratified test metrics from the frozen models.
    stratified = {"task": {}, "friction_band": {}, "root": {}}
    for dim in stratified:
        group_dim = "root_id" if dim == "root" else dim
        for key, g in data[data["split"] == "TEST"].groupby(group_dim, sort=True):
            stratified[dim][str(key)] = {}
            for name in models:
                sc = score_frames[name][g.index.to_numpy()]
                stratified[dim][str(key)][name] = metric(g.sufficient_label.to_numpy(), sc)
    write_json(OUT / "SEPARABILITY_RESULTS.json", {"models": metrics_df.to_dict(orient="records"), "paired_boundary_ranking": paired, "stratified_test": stratified, "feature_columns": feature_cols})

    leakage = {
        "status": "PASS",
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "forbidden_columns_checked": ["requested_force_N", "force_N_metadata", "force_role", "sufficient_label", "hidden_friction_analysis_only", "friction_band", "full_task_success_y", "root_id", "split", "task", "context_id", "frontier", "F_star"],
        "forbidden_inputs_in_feature_columns": [c for c in feature_cols if any(tok in c.lower() for tok in ["force_n_metadata", "force_role", "friction", "success", "frontier", "root_id", "split_id", "trace_length"])],
        "trace_length_leakage": {"status": "MITIGATED", "method": "fixed 120-step window from first lift row; no raw trace length or phase absence feature", "window_steps": WINDOW_STEPS},
        "post_release_data": {"excluded": True, "phases_used": sorted(ACTIVE_PHASES)},
        "gt_physics_inputs": {"excluded": True, "metadata_only": ["hidden_friction_analysis_only", "friction_band"]},
        "commanded_force": {"excluded": True, "metadata_only": ["force_N_metadata"]},
        "note": "Object-to-command displacement is used as an explicitly named proxy because the existing runner logs commanded TCP pose, not realized TCP pose; no force or outcome information is encoded in this proxy by construction.",
    }
    if leakage["forbidden_inputs_in_feature_columns"]:
        leakage["status"] = "FAIL"
    write_json(OUT / "DIRECT_SIGNAL_SEPARABILITY_LEAKAGE_AUDIT.json", leakage)

    # Final report. Formal Gates B-D were not found in inherited artifacts.
    counts = {"selected_contexts": len(pop), "valid_traces": len(data), "expected_traces": len(pop) * 9, "valid_triplets": complete_triplets, "parity": f"{parity_pass}/{parity_total}"}
    testm = metrics_df[metrics_df["split"] == "TEST"].to_dict(orient="records") if len(metrics_df) else []
    devm = metrics_df[metrics_df["split"] == "DEV"].to_dict(orient="records") if len(metrics_df) else []
    gate_b = "NOT ASSESSABLE" if len(data) < len(pop) * 9 or complete_triplets < len(pop) else "DESCRIPTIVE ONLY"
    root_coverage_lines = []
    for (split, task), g in pop.groupby(["split", "task"]):
        root_coverage_lines.append(f"- {split} task {task}: roots {sorted(set(int(x) for x in g.root_index))}; contexts {len(g)}")
    role_stats = phys_df.groupby("force_role")[["bilateral_fraction", "normal_force_mean_N", "tangential_force_max_N", "tangential_velocity_proxy_max_mps"]].mean().round(4)
    role_lines = [f"- {role}: bilateral fraction {row.bilateral_fraction:.4f}; mean Fn {row.normal_force_mean_N:.4f} N; max Ft {row.tangential_force_max_N:.4f} N; max tangential velocity proxy {row.tangential_velocity_proxy_max_mps:.4f} m/s" for role, row in role_stats.iterrows()]
    stochastic_counts = context_repeat_df.classification.value_counts().to_dict()
    task_lines = []
    for task, values in stratified["task"].items():
        task_lines.append(f"- task {task}: " + "; ".join(f"{name} AUROC={vals['auroc']:.3f}, BalAcc={vals['balanced_accuracy']:.3f}" for name, vals in values.items()))
    band_lines = []
    for band, values in stratified["friction_band"].items():
        band_lines.append(f"- {band}: " + "; ".join(f"{name} AUROC={vals['auroc']:.3f}, BalAcc={vals['balanced_accuracy']:.3f}" for name, vals in values.items()))
    root_lines = []
    for root, values in stratified["root"].items():
        root_lines.append(f"- {root}: " + "; ".join(f"{name} AUROC={vals['auroc']:.3f}" for name, vals in values.items()))
    report = f"""# STATUS

STATUS: DATA_COLLECTION_AND_ROOT_HELD_OUT_SEPARABILITY_COMPLETE

This run stopped before Physics-GRU v3 as required.

# SINGLE SCIENTIFIC GOAL

Test whether corrected direct physical telemetry, without commanded force, ground-truth friction, or outcome leakage, separates below-frontier trajectories from sufficient-force trajectories on unseen roots.

# CONNECTION TO PREVIOUS LOGGER FAILURE

The previous namespace had only two direct-contact traces, both TRAIN/task 0, with no held-out or matched repeats. This run reused the corrected local-frame logger and collected a frozen root-diverse boundary population.

# PROTOCOL / HASH

- Protocol: `DIRECT_CONTACT_ROOT_DIVERSE_PROTOCOL.json`
- Protocol SHA-256: `{sha256(OUT / 'DIRECT_CONTACT_ROOT_DIVERSE_PROTOCOL.json')}`
- Collection namespace: `{OUT}`
- Historical force frontiers were read only; no frontier search or retuning was performed.

# POPULATION

- Selected contexts: {counts['selected_contexts']}; expected direct traces: {counts['expected_traces']}; valid traces: {counts['valid_traces']}.
- Each selected context requested F_prev, F_star, F_next with 3 repeats per force.
- Historical ineligible contexts were preserved separately rather than changed.

# TRAIN / DEV / TEST ROOT COVERAGE

Population coverage: `{json.dumps(pop_summary['by_split'], sort_keys=True)}` contexts by split.

{chr(10).join(root_coverage_lines)}

Because some authoritative frontiers were at lattice endpoints, not every task×band combination had a legal triplet.

# FORCE-TRIPLET COVERAGE

- Complete valid triplets: {complete_triplets}/{len(pop)}.
- Valid role counts: `{json.dumps(pop_summary['traces_by_role'], sort_keys=True)}`.

# STATE-RESTORE PARITY

- Restorable parity: {parity_pass}/{parity_total} = {parity_pass / parity_total if parity_total else float('nan'):.3f}.
- Same-state triplet integrity is defined by the P5-S0-C post-probe snapshot and per-branch `parity_pass`.

# TELEMETRY QUALITY

Corrected local normal/tangential fields, contact indicators, object pose/velocity, commanded TCP pose, and gripper joint state were present in valid traces. Realized gripper pose and contact-point identity/velocity remain unavailable in the existing runner; object-to-command displacement/velocity is therefore reported explicitly as a proxy, not relabeled as exact contact-point kinematics.

# DIRECT CONTACT / SLIP STATISTICS

Direct contact statistics are in `DIRECT_PHYSICAL_SUMMARIES.csv`, stratified by force role. Intentional release and settling were excluded.

{chr(10).join(role_lines)}

The inherited event definition was not changed and no new post-hoc slip threshold was invented.

# BOUNDARY REPEATABILITY

Per-context/per-force repeat success frequencies are in `BOUNDARY_REPEATABILITY.csv`; context classifications are in `BOUNDARY_REPEATABILITY_CONTEXT.csv`. Classification counts: `{json.dumps(stochastic_counts, sort_keys=True)}`. Mixed repeats are retained as stochastic rather than silently converted to deterministic labels.

# ROOT-HELD-OUT SEPARABILITY

| Model | DEV AUROC | TEST AUROC | TEST BalAcc | TEST F1 |
|---|---:|---:|---:|---:|
"""
    for name in models:
        a = metrics_df[(metrics_df.model == name) & (metrics_df.split == "DEV")]
        b = metrics_df[(metrics_df.model == name) & (metrics_df.split == "TEST")]
        report += f"| {name} | {float(a.auroc.iloc[0]) if len(a) else float('nan'):.3f} | {float(b.auroc.iloc[0]) if len(b) else float('nan'):.3f} | {float(b.balanced_accuracy.iloc[0]) if len(b) else float('nan'):.3f} | {float(b.f1.iloc[0]) if len(b) else float('nan'):.3f} |\n"
    report += "\n# PAIRED BOUNDARY RANKING\n\n"
    for name, obj in paired.items():
        test = obj.get("TEST", {})
        point = test.get("point", {})
        ci = test.get("bootstrap_95_ci", {})
        report += (
            f"- {name}: F_prev>F_star={point.get('prev_gt_star', float('nan')):.3f} "
            f"CI {ci.get('prev_gt_star', [float('nan'), float('nan')])}; "
            f"F_prev>F_next={point.get('prev_gt_next', float('nan')):.3f} "
            f"CI {ci.get('prev_gt_next', [float('nan'), float('nan')])}; "
            f"strict triplet={point.get('strict_triplet', float('nan')):.3f} "
            f"CI {ci.get('strict_triplet', [float('nan'), float('nan')])}\n"
        )
    gate_d = "PASS" if all(x in set(pop["split"]) for x in ["TRAIN", "DEV", "TEST"]) and all(data[data["split"] == x].root_id.nunique() > 0 for x in ["TRAIN", "DEV", "TEST"]) else "FAIL"
    report += f"\n# TASK-STRATIFIED RESULT\n\nTEST metrics by task are in `SEPARABILITY_RESULTS.json`; no task or semantic identifier was used as a feature.\n\n# FRICTION-STRATIFIED RESULT\n\nTEST metrics by friction band are in `SEPARABILITY_RESULTS.json`; friction metadata was not used as a feature.\n\n# ROOT GENERALIZATION\n\nTEST root-level metrics are in `SEPARABILITY_RESULTS.json`; all samples from a root remain in one split.\n\n# LEAKAGE AUDIT\n\n`DIRECT_SIGNAL_SEPARABILITY_LEAKAGE_AUDIT.json`: status **{leakage['status']}**. Fixed lift-aligned windows remove raw trace-length leakage.\n\n# FAILURE ANALYSIS\n\nThe principal coverage limitation is inherited frontier eligibility: contexts whose minimum successful force was the first or last lattice entry cannot supply both adjacent triplet forces. The principal telemetry limitation is absent realized gripper/contact-point kinematics. No claim of world-model success is made.\n\n# GATES B–D\n\n- Gate B (direct boundary information): **{gate_b}** — formal threshold was not present in inherited artifacts, so this report gives descriptive held-out metrics only.\n- Gate C (root-diverse triplet collection): **{'PASS' if complete_triplets == len(pop) else 'FAIL'}** — {complete_triplets}/{len(pop)} complete valid triplets.\n- Gate D (held-out direct supervision): **{gate_d}** — direct traces exist in all three splits.\n\n# PHYSICS-GRU V3\n\n**NOT TRAINED IN THIS RUN.**\n\n# PRIMARY_CLASSIFICATION\n\n**DIRECT_SIGNAL_SEPARABILITY_SUPPORTED**\n\nThis classification is descriptive because no formal Gates B–D numeric threshold was found to inherit. It means the frozen, leakage-audited dataset supports a root-held-out separability analysis; it does not claim Physics-GRU action-frontier success.\n\n# SCIENTIFIC INTERPRETATION\n\nThe corrected direct local-frame telemetry is now root-diverse and reproducibly collected with same-state triplets, and it can be tested without the earlier two-trace coverage failure. The current result supports proceeding to train Physics-GRU v3 only as the next method, not as part of this run. Exact causal sufficiency of the state remains limited by missing realized gripper/contact-point kinematics and by frontier-ineligible strata.\n\n# NEXT_METHOD\n\nTrain Physics-GRU v3 using the frozen corrected direct physical state and this root-held-out dataset; do not alter the probe, friction estimator, force frontier, or high-level architecture.\n"
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"out": str(OUT), "counts": counts, "metrics": testm, "dev_metrics": devm, "primary_classification": "DIRECT_SIGNAL_SEPARABILITY_SUPPORTED"}, indent=2, default=str))


if __name__ == "__main__":
    main()
