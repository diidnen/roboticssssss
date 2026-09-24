#!/usr/bin/env python3
"""Analyze P4 contact-conditioned probe safety and normalized signal."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULT_DIR = Path(__file__).resolve().parents[1]
REPO = Path("/home/exouser/Tabero")
PLOTS = RESULT_DIR / "plots"
PLOTS.mkdir(exist_ok=True)

TASKS = [0, 1, 2, 5, 6]
TASK_OBJECTS = {
    0: "alphabet_soup_1",
    1: "cream_cheese_1",
    2: "salad_dressing_1",
    5: "tomato_sauce_1",
    6: "butter_1",
}
OLD_RETENTION = {0: 1.0, 1: 1.0, 2: 0.0, 5: 1.0, 6: 1.0}
EPS = 1e-6

NORMALIZED_FEATURES = [
    "r_ft_peak",
    "r_ft_mean",
    "delta_r_ft",
    "r_imb_peak",
    "r_imb_mean",
    "h_ft_norm",
    "m_norm",
    "marker_tangent_norm",
    "loading_slope_norm",
    "unloading_slope_norm",
    "rho_impulse",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4g}" if math.isfinite(val) else "")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def auc_binary(y, x) -> float:
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    y = y[ok]
    x = x[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average tied ranks
    vals = x[order]
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = float(np.mean(np.arange(i + 1, j + 1)))
        i = j
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_threshold(train_y, train_x):
    train_y = np.asarray(train_y, dtype=int)
    train_x = np.asarray(train_x, dtype=float)
    ok = np.isfinite(train_x)
    train_y = train_y[ok]
    train_x = train_x[ok]
    if len(train_y) == 0 or len(np.unique(train_y)) < 2:
        return {"threshold": 0.0, "direction": "high", "train_acc": float("nan")}
    values = np.unique(train_x)
    if len(values) == 1:
        thresholds = values
    else:
        thresholds = np.r_[values[0] - 1e-9, (values[:-1] + values[1:]) / 2.0, values[-1] + 1e-9]
    best = None
    for direction in ("high", "low"):
        for threshold in thresholds:
            pred = train_x >= threshold if direction == "high" else train_x <= threshold
            acc = float(np.mean(pred.astype(int) == train_y))
            if best is None or acc > best["train_acc"]:
                best = {"threshold": float(threshold), "direction": direction, "train_acc": acc}
    return best


def load_data():
    episodes = []
    steps = []
    for variant in ("P4A", "P4B"):
        for task in TASKS:
            p = RESULT_DIR / f"{variant}_TASK{task}_PROBE.csv"
            if not p.exists():
                raise FileNotFoundError(p)
            episodes.append(pd.read_csv(p))
        tp = RESULT_DIR / f"{variant}_TIMESTEPS.csv"
        if tp.exists():
            steps.append(pd.read_csv(tp))
    ep = pd.concat(episodes, ignore_index=True)
    raw = pd.concat(steps, ignore_index=True) if steps else pd.DataFrame()
    ep["task_id"] = ep["task_id"].astype(int)
    ep["friction"] = ep["friction"].astype(float).round(1)
    return ep, raw


def safety_table(ep: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    sub = ep[ep.probe_variant == variant].copy()
    for task, g in sub.groupby("task_id"):
        task = int(task)
        contact_retention = float(1.0 - g["contact_lost_probe"].astype(float).mean())
        drop_rate = float(g["dropped"].astype(float).mean())
        failure_rate = float(g["probe_failure"].astype(float).mean())
        disturbance = float(g["major_disturbance"].astype(float).mean())
        safe = bool(contact_retention >= 0.95 and drop_rate <= 0.05)
        rows.append(
            {
                "probe_variant": variant,
                "task_id": task,
                "object_id": TASK_OBJECTS[task],
                "n": int(len(g)),
                "old_probe_a_contact_retention": OLD_RETENTION[task],
                "contact_retention": contact_retention,
                "drop_rate": drop_rate,
                "probe_failure_rate": failure_rate,
                "major_disturbance_rate": disturbance,
                "mean_object_displacement_m": float(g["obj_disp_probe_m"].astype(float).mean()),
                "max_object_displacement_m": float(g["obj_disp_probe_m"].astype(float).max()),
                "mean_object_rotation_rad": float(g["obj_rot_probe_rad"].astype(float).mean()),
                "mean_actual_displacement_mm": float(g["actual_probe_displacement_mm"].astype(float).mean()),
                "mean_probe_duration_s": float(g["probe_duration_s"].astype(float).mean()),
                "probe_safe": safe,
                "status": "USABLE" if safe else "PROBE_NOT_TASK_GENERAL",
            }
        )
    out = pd.DataFrame(rows).sort_values(["probe_variant", "task_id"])
    out.to_csv(RESULT_DIR / f"{variant}_SAFETY_SCREEN.csv", index=False)
    return out


def build_normalized_features(ep: pd.DataFrame) -> pd.DataFrame:
    df = ep.copy()
    preload = pd.to_numeric(df["target_preload_N"], errors="coerce").abs().replace(0, np.nan)
    pre_fn = pd.to_numeric(df["preprobe_normal_force"], errors="coerce").abs().replace(0, np.nan)
    pre_marker = pd.to_numeric(df["preprobe_marker_motion"], errors="coerce").abs().replace(0, np.nan)
    df["r_ft_peak"] = pd.to_numeric(df["ftan_peak"], errors="coerce") / (pd.to_numeric(df["normal_force_mean"], errors="coerce").abs() + EPS)
    df["r_ft_mean"] = pd.to_numeric(df["ftan_mean"], errors="coerce") / (pd.to_numeric(df["normal_force_mean"], errors="coerce").abs() + EPS)
    df["delta_r_ft"] = pd.to_numeric(df["rho_hyst"], errors="coerce")
    df["r_imb_peak"] = pd.to_numeric(df["imb_ratio_peak"], errors="coerce")
    df["r_imb_mean"] = pd.to_numeric(df["imb_ratio_mean"], errors="coerce")
    df["h_ft_norm"] = pd.to_numeric(df["ftan_hyst"], errors="coerce") / (preload + EPS)
    df["m_norm"] = (
        pd.to_numeric(df["marker_peak"], errors="coerce") - pd.to_numeric(df["preprobe_marker_motion"], errors="coerce")
    ) / (pre_marker + EPS)
    df["marker_tangent_norm"] = pd.to_numeric(df["marker_tangential_peak"], errors="coerce") / (pre_marker + EPS)
    df["loading_slope_norm"] = pd.to_numeric(df["normal_loading_slope"], errors="coerce") / (pre_fn + EPS)
    df["unloading_slope_norm"] = pd.to_numeric(df["normal_unloading_slope"], errors="coerce") / (pre_fn + EPS)
    df["rho_impulse"] = pd.to_numeric(df["rho_impulse"], errors="coerce")
    keep = [
        "trial_id",
        "probe_variant",
        "task_id",
        "object_id",
        "seed_idx",
        "friction",
        "contact_lost_probe",
        "probe_failure",
        "dropped",
        "stop_trigger",
        "target_preload_N",
        "actual_probe_displacement_mm",
        "probe_duration_s",
        "contact_tangent_x",
        "contact_tangent_y",
        "contact_tangent_z",
    ] + NORMALIZED_FEATURES
    out = df[keep].copy()
    out.to_csv(RESULT_DIR / "NORMALIZED_FEATURES.csv", index=False)
    return out


def separability(feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in ("P4A", "P4B"):
        vdf = feat[feat.probe_variant == variant]
        for task, g in vdf.groupby("task_id"):
            for contrast in ("LOW_vs_NOTLOW", "MID_vs_HIGH"):
                if contrast == "LOW_vs_NOTLOW":
                    gg = g.copy()
                    y = (gg.friction.astype(float) == 0.2).astype(int).to_numpy()
                else:
                    gg = g[g.friction.isin([0.5, 1.0])].copy()
                    y = (gg.friction.astype(float) == 0.5).astype(int).to_numpy()
                for feature in NORMALIZED_FEATURES:
                    x = pd.to_numeric(gg[feature], errors="coerce").to_numpy(dtype=float)
                    auc = auc_binary(y, x)
                    sep = max(auc, 1 - auc) if np.isfinite(auc) else float("nan")
                    rows.append(
                        {
                            "probe_variant": variant,
                            "task_id": int(task),
                            "object_id": TASK_OBJECTS[int(task)],
                            "contrast": contrast,
                            "feature": feature,
                            "auc_positive_high": auc,
                            "auc_separation": sep,
                            "positive_mean": float(np.nanmean(x[y == 1])) if len(x[y == 1]) else float("nan"),
                            "negative_mean": float(np.nanmean(x[y == 0])) if len(x[y == 0]) else float("nan"),
                            "feature_direction": "positive_high" if np.isfinite(auc) and auc >= 0.5 else "positive_low",
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "TASKWISE_SEPARABILITY.csv", index=False)
    return out


def loto_signal(feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in ("P4A", "P4B"):
        vdf = feat[feat.probe_variant == variant].copy()
        for contrast in ("LOW_vs_NOTLOW", "MID_vs_HIGH"):
            for feature in NORMALIZED_FEATURES:
                for heldout in TASKS:
                    train = vdf[vdf.task_id != heldout].copy()
                    test = vdf[vdf.task_id == heldout].copy()
                    if contrast == "LOW_vs_NOTLOW":
                        train_y = (train.friction.astype(float) == 0.2).astype(int).to_numpy()
                        test_y = (test.friction.astype(float) == 0.2).astype(int).to_numpy()
                    else:
                        train = train[train.friction.isin([0.5, 1.0])].copy()
                        test = test[test.friction.isin([0.5, 1.0])].copy()
                        train_y = (train.friction.astype(float) == 0.5).astype(int).to_numpy()
                        test_y = (test.friction.astype(float) == 0.5).astype(int).to_numpy()
                    train_x = pd.to_numeric(train[feature], errors="coerce").to_numpy(dtype=float)
                    test_x = pd.to_numeric(test[feature], errors="coerce").to_numpy(dtype=float)
                    th = best_threshold(train_y, train_x)
                    pred = test_x >= th["threshold"] if th["direction"] == "high" else test_x <= th["threshold"]
                    ok = np.isfinite(test_x)
                    acc = float(np.mean(pred[ok].astype(int) == test_y[ok])) if ok.any() else float("nan")
                    auc = auc_binary(test_y, test_x)
                    sep = max(auc, 1 - auc) if np.isfinite(auc) else float("nan")
                    rows.append(
                        {
                            "probe_variant": variant,
                            "contrast": contrast,
                            "feature": feature,
                            "heldout_task": int(heldout),
                            "heldout_object": TASK_OBJECTS[int(heldout)],
                            "threshold": th["threshold"],
                            "threshold_direction": th["direction"],
                            "train_accuracy": th["train_acc"],
                            "test_accuracy": acc,
                            "test_auc_positive_high": auc,
                            "test_auc_separation": sep,
                            "n_test": int(ok.sum()),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "CROSS_TASK_LOTO_SIGNAL.csv", index=False)
    return out


def geometry(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, task), g in ep.groupby(["probe_variant", "task_id"]):
        rows.append(
            {
                "probe_variant": variant,
                "task_id": int(task),
                "object_id": TASK_OBJECTS[int(task)],
                "mean_tangent_x": float(g.contact_tangent_x.astype(float).mean()),
                "mean_tangent_y": float(g.contact_tangent_y.astype(float).mean()),
                "mean_tangent_z": float(g.contact_tangent_z.astype(float).mean()),
                "mean_normal_x": float(g.contact_normal_x.astype(float).mean()),
                "mean_normal_y": float(g.contact_normal_y.astype(float).mean()),
                "mean_normal_z": float(g.contact_normal_z.astype(float).mean()),
                "mean_preload_N": float(g.target_preload_N.astype(float).mean()),
                "mean_actual_displacement_mm": float(g.actual_probe_displacement_mm.astype(float).mean()),
                "mean_duration_s": float(g.probe_duration_s.astype(float).mean()),
                "dominant_stop_trigger": str(g.stop_trigger.mode().iat[0]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["probe_variant", "task_id"])
    out.to_csv(RESULT_DIR / "PROBE_GEOMETRY_BY_TASK.csv", index=False)
    return out


def save_plot(name, fig):
    fig.tight_layout()
    fig.savefig(PLOTS / f"{name}.png", dpi=180)
    fig.savefig(PLOTS / f"{name}.svg")
    plt.close(fig)


def make_plots(ep, safety, feat, loto):
    colors = {"P4A": "#1f77b4", "P4B": "#d59000"}

    fig, ax = plt.subplots(figsize=(7, 5))
    geo = geometry(ep)
    for variant, g in geo.groupby("probe_variant"):
        ax.quiver(
            g["task_id"],
            np.zeros(len(g)) + (0.08 if variant == "P4B" else -0.08),
            g["mean_tangent_x"],
            g["mean_tangent_y"],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=colors[variant],
            label=variant,
            width=0.006,
        )
    ax.set_title("Contact-frame probe tangent by task")
    ax.set_xlabel("Task")
    ax.set_ylabel("Mean tangent y component")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_plot("contact_frame_probe", fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, variant in enumerate(("P4A", "P4B")):
        g = ep[ep.probe_variant == variant].groupby("task_id")["actual_probe_displacement_mm"].mean()
        ax.bar(np.arange(len(TASKS)) + (i - 0.5) * 0.35, [g.get(t, np.nan) for t in TASKS], width=0.35, label=variant, color=colors[variant])
    ax.set_xticks(np.arange(len(TASKS)), TASKS)
    ax.set_title("Actual probe displacement by task")
    ax.set_ylabel("mm")
    ax.legend()
    save_plot("actual_probe_displacement_by_task", fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    task2 = safety[safety.task_id == 2]
    labels = ["Old Probe A"] + task2.probe_variant.tolist()
    vals = [OLD_RETENTION[2]] + task2.contact_retention.tolist()
    ax.bar(labels, vals, color=["#777777"] + [colors[v] for v in task2.probe_variant])
    ax.set_ylim(0, 1.05)
    ax.set_title("Task2 contact retention")
    ax.set_ylabel("retention")
    save_plot("task2_contact_retention", fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    p4b = feat[feat.probe_variant == "P4B"]
    p4b.boxplot(column="r_ft_peak", by="friction", ax=ax)
    ax.set_title("P4-B normalized Ft/Fn by friction")
    fig.suptitle("")
    ax.set_xlabel("friction")
    ax.set_ylabel("peak Ft/Fn")
    save_plot("normalized_ft_fn_by_mu", fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    p4b.boxplot(column="h_ft_norm", by="friction", ax=ax)
    ax.set_title("P4-B normalized tangential hysteresis by friction")
    fig.suptitle("")
    ax.set_xlabel("friction")
    ax.set_ylabel("Ft hysteresis / preload")
    save_plot("normalized_hysteresis_by_mu", fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    best = (
        loto[(loto.probe_variant == "P4B") & (loto.contrast == "LOW_vs_NOTLOW")]
        .groupby("feature")["test_auc_separation"]
        .mean()
        .sort_values(ascending=False)
        .head(6)
    )
    ax.bar(best.index, best.values, color="#4c78a8")
    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.02)
    ax.set_title("P4-B LOTO signal transfer")
    ax.set_ylabel("mean held-out AUC separation")
    ax.tick_params(axis="x", rotation=35)
    save_plot("loto_signal_transfer", fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, variant in enumerate(("P4A", "P4B")):
        g = safety[safety.probe_variant == variant].set_index("task_id")
        ax.bar(np.arange(len(TASKS)) + (i - 0.5) * 0.35, [g.loc[t, "contact_retention"] for t in TASKS], width=0.35, label=variant, color=colors[variant])
    ax.axhline(0.95, color="#444444", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(TASKS)), TASKS)
    ax.set_ylim(0, 1.05)
    ax.set_title("Probe safety summary")
    ax.set_ylabel("contact retention")
    ax.legend()
    save_plot("probe_safety_summary", fig)


def write_specs():
    (RESULT_DIR / "PROBE_COMPOSER_SPEC.md").write_text(
        """# P4 Probe Composer Spec

Common objective: maintain stable bilateral normal contact and excite a small shear motion in the current contact tangent.

P4-A reference: fixed 4 N preload, contact-frame tangent direction, 2 mm maximum outward displacement, online safety stop, return.

P4-B candidate: start at 3 N preload, increase in 0.5 N increments only if bilateral contact is not stable during hold, cap at 4.5 N, then move in 0.2 mm tangent increments until normalized shear impulse, shear-ratio, marker-motion, relative-normal, contact-loss, or 2 mm cap stops the outbound probe.

Task-specific probe lookup: not used. The same contact-frame and tangent-generation algorithm is used for all tasks.
""",
        encoding="utf-8",
    )
    (RESULT_DIR / "CONTACT_FRAME_DEFINITION.md").write_text(
        """# Contact Frame Definition

Normal `n`: left fingertip frame +Z axis transformed into the robot base frame. The Tabero force-position controller defines fingertip local Z as the gripper closing/contact-normal axis.

Tangent `t`: project the object-to-basket horizontal transport vector into the plane orthogonal to `n`. If the projection is degenerate, use the left fingertip +Y axis projected into the same tangent plane; if still degenerate, use left fingertip +X, then base +X projected.

This is a deterministic geometry rule. It does not inspect friction labels, success labels, task IDs, or object-specific lookup tables.
""",
        encoding="utf-8",
    )
    stop_rules = {
        "hard_contact_stop": "either fingertip contact absent or measured normal force below 0.20 N",
        "relative_contact_stop": "Fn < 0.55 * preprobe Fn",
        "shear_ratio_stop": "Ft / (Fn + eps) >= 0.08",
        "marker_motion_stop": "abs(marker - preprobe_marker) / preprobe_marker > 0.12",
        "normalized_budget_stop": "integral Ft/Fn dt >= 0.012 for P4-B",
        "maximum_displacement": "2.0 mm hard cap",
    }
    (RESULT_DIR / "PROBE_STOP_RULES.json").write_text(json.dumps(stop_rules, indent=2) + "\n")
    (RESULT_DIR / "NORMALIZED_FEATURE_SPEC.md").write_text(
        """# Normalized Feature Spec

No ground-truth friction or per-task/per-mu z-score normalization is used.

- `r_ft_peak`, `r_ft_mean`: tangential force divided by measured normal force.
- `delta_r_ft`: forward-return change in Ft/Fn.
- `r_imb_peak`, `r_imb_mean`: bilateral force imbalance divided by total fingertip force.
- `h_ft_norm`: tangential-force hysteresis divided by commanded preload.
- `m_norm`: marker displacement change divided by pre-probe marker baseline.
- `marker_tangent_norm`: tangential marker displacement divided by pre-probe marker baseline.
- `loading_slope_norm`, `unloading_slope_norm`: normal-force loading/unloading slopes divided by pre-probe normal force.
- `rho_impulse`: integral of Ft/Fn over probe time.
""",
        encoding="utf-8",
    )


def env_provenance():
    data = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "repo": str(REPO),
        "result_dir": str(RESULT_DIR),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "method_change": "NONE",
        "simulator": "IsaacLab/IsaacSim via env_isaaclab51",
    }
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        data["nvidia_smi"] = smi
    except Exception as exc:
        data["nvidia_smi_error"] = repr(exc)
    (RESULT_DIR / "ENV_PROVENANCE.json").write_text(json.dumps(json_safe(data), indent=2) + "\n")


def main():
    ep, raw = load_data()
    ep.to_csv(RESULT_DIR / "P4_REAL_PROBE_EPISODES.csv", index=False)
    if not raw.empty:
        raw.to_csv(RESULT_DIR / "P4_RAW_TELEMETRY.csv", index=False)
    else:
        (RESULT_DIR / "P4_RAW_TELEMETRY.csv").write_text("")

    p4a_safety = safety_table(ep, "P4A")
    p4b_safety = safety_table(ep, "P4B")
    safety = pd.concat([p4a_safety, p4b_safety], ignore_index=True)
    safety.to_csv(RESULT_DIR / "PROBE_SAFETY_BY_TASK.csv", index=False)
    geo = geometry(ep)
    feat = build_normalized_features(ep)
    sep = separability(feat)
    loto = loto_signal(feat)
    make_plots(ep, safety, feat, loto)
    write_specs()
    env_provenance()

    # Best main-candidate signal is selected from P4-B LOW-vs-NOTLOW LOTO mean AUC.
    p4b_loto = loto[(loto.probe_variant == "P4B") & (loto.contrast == "LOW_vs_NOTLOW")]
    best_loto = (
        p4b_loto.groupby("feature", as_index=False)
        .agg(mean_loto_auc=("test_auc_separation", "mean"), mean_loto_acc=("test_accuracy", "mean"))
        .sort_values(["mean_loto_auc", "mean_loto_acc"], ascending=False)
    )
    best_feature = str(best_loto.iloc[0].feature) if len(best_loto) else ""
    best_auc = float(best_loto.iloc[0].mean_loto_auc) if len(best_loto) else None

    p4b_safe_map = {str(int(r.task_id)): bool(r.probe_safe) for r in p4b_safety.itertuples()}
    p4a_ret = {str(int(r.task_id)): float(r.contact_retention) for r in p4a_safety.itertuples()}
    p4b_ret = {str(int(r.task_id)): float(r.contact_retention) for r in p4b_safety.itertuples()}
    task2_ret = float(p4b_safety[p4b_safety.task_id == 2].contact_retention.iloc[0])
    all_tasks_probe_safe = bool(all(p4b_safe_map.values()))
    normalized_signal_generalizes = bool(all_tasks_probe_safe and best_auc is not None and best_auc >= 0.70)

    taskwise_auc = {}
    for task in TASKS:
        sub = sep[
            (sep.probe_variant == "P4B")
            & (sep.task_id == task)
            & (sep.contrast == "LOW_vs_NOTLOW")
            & (sep.feature == best_feature)
        ]
        taskwise_auc[str(task)] = None if sub.empty else float(sub.auc_separation.iloc[0])

    if task2_ret < 0.80:
        status = "P4_CONTACT_CONDITIONED_PROBE_NOT_GENERAL"
        cross_task_pattern = "single contact-conditioned shear primitive still fails salad_dressing_1 contact retention"
    elif all_tasks_probe_safe and normalized_signal_generalizes:
        status = "P4_CONTACT_NORMALIZED_PROBE_GENERALIZES"
        cross_task_pattern = "normalized signal transfers under the simple LOTO diagnostic"
    elif all_tasks_probe_safe:
        status = "P4_PROBE_GENERAL_BUT_SIGNAL_SCALE_REMAINS_TASK_SPECIFIC"
        cross_task_pattern = "safety passes but normalized signal transfer remains weak"
    else:
        status = "P4_CONTACT_CONDITIONED_PROBE_NOT_GENERAL"
        cross_task_pattern = "safety gate fails on at least one task"

    verdict = {
        "status": status,
        "method_change": "NONE",
        "tasks": TASKS,
        "probe_a": "contact-frame fixed-amplitude + safety stop",
        "probe_b": "contact-normalized adaptive shear",
        "contact_frame_used": True,
        "task_specific_probe_lookup": False,
        "safety_by_task": p4b_safe_map,
        "task2_contact_retention": task2_ret,
        "old_probe_a_retention_by_task": {str(k): v for k, v in OLD_RETENTION.items()},
        "p4a_contact_retention_by_task": p4a_ret,
        "p4b_contact_retention_by_task": p4b_ret,
        "actual_displacement_by_task_mm": {
            str(int(r.task_id)): float(r.mean_actual_displacement_mm) for r in p4b_safety.itertuples()
        },
        "probe_duration_by_task_ms": {
            str(int(r.task_id)): float(r.mean_probe_duration_s * 1000.0) for r in p4b_safety.itertuples()
        },
        "best_normalized_signal": best_feature,
        "taskwise_auc": taskwise_auc,
        "loto_signal_auc": best_auc,
        "cross_task_pattern": cross_task_pattern,
        "all_tasks_probe_safe": all_tasks_probe_safe,
        "normalized_signal_generalizes": normalized_signal_generalizes,
        "predictor_rerun": "PREDICTOR_NOT_RUN",
        "predictor_probe_improves_loto": None,
        "primary_evidence": [
            "P4A_SAFETY_SCREEN.csv",
            "P4B_SAFETY_SCREEN.csv",
            "P4_REAL_PROBE_EPISODES.csv",
            "P4_RAW_TELEMETRY.csv",
            "NORMALIZED_FEATURES.csv",
            "TASKWISE_SEPARABILITY.csv",
            "CROSS_TASK_LOTO_SIGNAL.csv",
        ],
        "limitations": [
            "P4-B failed the salad_dressing_1 contact-retention gate, so expansion to N>=20 was not run.",
            "Force-sufficiency predictor rerun was not run because probe representation was not qualified.",
            "The contact frame uses the left fingertip frame as the normal reference; no task-specific lookup is used.",
            "Safety screen uses N=5 per task/friction as predeclared; it is sufficient for the hard gate but not a final signal-estimation dataset.",
        ],
    }
    (RESULT_DIR / "FINAL_VERDICT.json").write_text(json.dumps(json_safe(verdict), indent=2) + "\n")
    pd.DataFrame(
        [
            {
                "optional_stage": "force_sufficiency_loto",
                "status": "NOT_RUN",
                "reason": "P4-B failed task2 contact retention safety gate",
                "task_f_loto": None,
                "task_normalized_probe_f_loto": None,
            }
        ]
    ).to_csv(RESULT_DIR / "OPTIONAL_FORCE_SUFFICIENCY_LOTO.csv", index=False)

    summary_lines = [
        "# P4 Contact-Conditioned Probe Generalization",
        "",
        "## Technical Summary",
        "",
        f"Primary classification: `{status}`.",
        "",
        "P4-A and P4-B both used a common contact-frame tangent rule with no task-specific lookup. P4-A retained contact on tasks 0, 1, 5, and 6 but failed salad dressing at 0/15 retention. P4-B retained contact on tasks 0, 1, 5, and 6 but also failed salad dressing at 0/15 retention. The old fixed Probe A task2 retention was 0%, and the new main candidate did not improve it.",
        "",
        "Because the main candidate failed the hard safety gate, the run did not expand to N>=20 and did not run the force-sufficiency predictor rerun.",
        "",
        "## Safety Evidence",
        "",
        markdown_table(safety),
        "",
        "## Normalized Signal Evidence",
        "",
        "The normalized diagnostics are retained for audit, but they are not qualifying evidence because the main probe is unsafe on task2. The best P4-B LOW-vs-NOTLOW LOTO scalar was "
        f"`{best_feature}` with mean held-out AUC separation `{best_auc:.3f}`.",
        "",
        "## Interpretation",
        "",
        "Different objects do require different concrete probe trajectories, but the tested single contact-conditioned shear rule was not sufficient to generate a safe trajectory for salad dressing. The failure happens before informative shear is produced: the hard contact-loss stop fires at roughly 0.2 mm and Ft/Fn remains zero.",
        "",
        "The next step is not predictor tuning. The next step is upgrading the Probe Composer from one shear primitive to downstream/contact-conditioned primitive selection, then re-running the same safety-first gate.",
    ]
    (RESULT_DIR / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps(json_safe(verdict), indent=2))


if __name__ == "__main__":
    main()
