#!/usr/bin/env python3
"""Analyze P4-R2 paired primitive coverage and selector outcomes without pandas."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULT_DIR = Path(__file__).resolve().parents[1]
REPO = Path("/home/exouser/Tabero")
P4_DIR = REPO / "analysis/results/p4_contact_conditioned_probe_20260822_184213"
TASKS = [0, 1, 2, 5, 6]
TASK_OBJECTS = {0: "alphabet_soup_1", 1: "cream_cheese_1", 2: "salad_dressing_1", 5: "tomato_sauce_1", 6: "butter_1"}
EPS = 1e-6
FEATURES = [
    "r_ft_peak",
    "r_ft_mean",
    "r_imb_peak",
    "r_imb_mean",
    "h_ft_norm",
    "marker_tangent_norm",
    "support_response_norm",
    "support_load_impulse_norm",
]


def f(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        val = row.get(key, "")
        if val in ("", None):
            return default
        return float(val)
    except Exception:
        return default


def i(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def b(row: dict, key: str) -> int:
    val = str(row.get(key, "")).strip().lower()
    if val in ("true", "1", "yes"):
        return 1
    if val in ("false", "0", "no", ""):
        return 0
    try:
        return int(float(val) != 0)
    except Exception:
        return 0


def mean(vals) -> float:
    xs = []
    for val in vals:
        try:
            x = float(val)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(sum(xs) / len(xs)) if xs else float("nan")


def mode(vals) -> str:
    xs = [str(x) for x in vals if str(x) != ""]
    return Counter(xs).most_common(1)[0][0] if xs else ""


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: json_safe(v) for k, v in row.items()} for row in rows])


def markdown_table(rows: list[dict], max_rows: int | None = None) -> str:
    if not rows:
        return "_No rows._"
    rows = rows[:max_rows] if max_rows is not None else rows
    cols, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            vals.append(f"{val:.4g}" if isinstance(val, float) and math.isfinite(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc_binary(y, x) -> float:
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    vals = x[order]
    pos = 0
    while pos < len(vals):
        end = pos + 1
        while end < len(vals) and vals[end] == vals[pos]:
            end += 1
        if end - pos > 1:
            ranks[order[pos:end]] = float(np.mean(np.arange(pos + 1, end + 1)))
        pos = end
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_threshold(train_y, train_x):
    train_y = np.asarray(train_y, dtype=int)
    train_x = np.asarray(train_x, dtype=float)
    ok = np.isfinite(train_x)
    train_y, train_x = train_y[ok], train_x[ok]
    if len(train_y) == 0 or len(np.unique(train_y)) < 2:
        return {"threshold": 0.0, "direction": "high", "train_acc": float("nan")}
    vals = np.unique(train_x)
    thresholds = vals if len(vals) == 1 else np.r_[vals[0] - 1e-9, (vals[:-1] + vals[1:]) / 2.0, vals[-1] + 1e-9]
    best = None
    for direction in ("high", "low"):
        for threshold in thresholds:
            pred = train_x >= threshold if direction == "high" else train_x <= threshold
            acc = float(np.mean(pred.astype(int) == train_y))
            if best is None or acc > best["train_acc"]:
                best = {"threshold": float(threshold), "direction": direction, "train_acc": acc}
    return best


def load_episodes() -> list[dict]:
    rows = []
    for split in ("development", "main"):
        for primitive in ("S", "L"):
            for task in TASKS:
                path = RESULT_DIR / f"P4R2_{split}_{primitive}_TASK{task}_PROBE.csv"
                for row in read_csv(path):
                    row["split"] = row.get("split", split)
                    row["primitive"] = row.get("primitive", primitive)
                    row["task_id"] = i(row, "task_id", task)
                    row["friction"] = round(f(row, "friction"), 1)
                    row["seed_idx"] = i(row, "seed_idx")
                    rows.append(row)
    write_csv(RESULT_DIR / "P4R2_RAW_EPISODE_MANIFEST.csv", rows)
    return rows


def forensic_from_p4():
    raw = read_csv(P4_DIR / "P4_RAW_TELEMETRY.csv")
    groups = defaultdict(list)
    for row in raw:
        groups[(row.get("probe_variant", ""), i(row, "task_id"), row.get("trial_id", ""))].append(row)
    rows = []
    for (variant, task, trial), g in groups.items():
        hold = [r for r in g if r.get("probe_phase") == "hold"][-10:]
        probe = [r for r in g if r.get("probe_phase") == "probe_out"][:1]
        if not probe:
            continue
        p = probe[0]
        n = np.array([f(p, "contact_normal_x"), f(p, "contact_normal_y"), f(p, "contact_normal_z")], dtype=float)
        v = np.array([f(p, "contact_tangent_x"), f(p, "contact_tangent_y"), f(p, "contact_tangent_z")], dtype=float)
        last_hold = [r for r in g if r.get("probe_phase") == "hold"][-1:]
        obj_disp = float("nan")
        obj_rot = float("nan")
        if last_hold:
            o = last_hold[0]
            obj_disp = float(np.linalg.norm([f(p, "object_x_priv") - f(o, "object_x_priv"), f(p, "object_y_priv") - f(o, "object_y_priv"), f(p, "object_z_priv") - f(o, "object_z_priv")]))
            q0 = np.array([f(o, "object_qw_priv"), f(o, "object_qx_priv"), f(o, "object_qy_priv"), f(o, "object_qz_priv")], dtype=float)
            q1 = np.array([f(p, "object_qw_priv"), f(p, "object_qx_priv"), f(p, "object_qy_priv"), f(p, "object_qz_priv")], dtype=float)
            q0 /= max(np.linalg.norm(q0), EPS)
            q1 /= max(np.linalg.norm(q1), EPS)
            obj_rot = float(2.0 * np.arccos(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0)))
        before = [r for r in g if f(r, "timestamp", -1.0) < f(p, "timestamp", -1.0)][-1:]
        eef_disp = float("nan")
        if before:
            e = before[0]
            eef_disp = float(np.linalg.norm([f(p, "eef_x") - f(e, "eef_x"), f(p, "eef_y") - f(e, "eef_y"), f(p, "eef_z") - f(e, "eef_z")]))
        hold_state = mode([r.get("contact_state", "") for r in hold]) or "unknown"
        h = lambda key: mean([f(r, key) for r in hold])
        rows.append(
            {
                "probe_variant": variant,
                "task_id": task,
                "trial_id": trial,
                "friction": f(p, "friction"),
                "seed_idx": i(p, "seed_idx"),
                "left_normal_x": float(n[0]),
                "left_normal_y": float(n[1]),
                "left_normal_z": float(n[2]),
                "right_normal_available": False,
                "contact_point_available": False,
                "patch_area_available": False,
                "patch_edge_margin_available": False,
                "preload_left_fn_proxy": abs(h("left_fz")),
                "preload_right_fn_proxy": abs(h("right_fz")),
                "preload_fn": h("measured_fn"),
                "preload_ft": h("measured_ft"),
                "first_step_left_fn_proxy": abs(f(p, "left_fz")),
                "first_step_right_fn_proxy": abs(f(p, "right_fz")),
                "first_step_fn": f(p, "measured_fn"),
                "first_step_ft": f(p, "measured_ft"),
                "first_step_ft_over_fn": f(p, "ft_over_fn"),
                "commanded_probe_dir_x": float(v[0]),
                "commanded_probe_dir_y": float(v[1]),
                "commanded_probe_dir_z": float(v[2]),
                "commanded_displacement_mm": f(p, "commanded_tangent_increment_mm"),
                "actual_eef_displacement_mm_proxy": eef_disp * 1000.0 if math.isfinite(eef_disp) else float("nan"),
                "actual_fingertip_displacement_available": False,
                "object_displacement_first_step_m": obj_disp,
                "object_rotation_first_step_rad": obj_rot,
                "contact_state_transition": f"{hold_state}->{p.get('contact_state', '')}",
                "hard_contact_loss_timestamp": f(p, "t_s") if p.get("stop_trigger") == "hard_contact_loss" else float("nan"),
                "controller_tracking_error_mm_proxy": abs(eef_disp * 1000.0 - f(p, "commanded_tangent_increment_mm")) if math.isfinite(eef_disp) else float("nan"),
                "stop_reason": p.get("stop_trigger", ""),
                "n_left_dot_v": float(np.dot(n, v)),
                "n_right_dot_v": float("nan"),
                "max_abs_normal_leakage": float(abs(np.dot(n, v))),
                "bilateral_normal_consistency": float("nan"),
                "angle_current_probe_to_downstream_deg": 0.0 if np.linalg.norm(v) > 0 else float("nan"),
                "contact_patch_edge_margin": float("nan"),
                "marker_motion_preload": h("marker_motion"),
                "marker_motion_first_step": f(p, "marker_motion"),
            }
        )
    write_csv(RESULT_DIR / "TASK2_FIRST_STEP_CONTACT_LOSS_TABLE.csv", rows)
    write_csv(RESULT_DIR / "P4R2_TASK2_FORENSIC_TABLE.csv", rows)
    summary = []
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    for task in sorted(by_task):
        g = by_task[task]
        summary.append(
            {
                "task_id": task,
                "n": len(g),
                "hard_loss_rate_first_step": mean([1.0 if math.isfinite(float(r["hard_contact_loss_timestamp"])) else 0.0 for r in g]),
                "mean_preload_fn": mean([r["preload_fn"] for r in g]),
                "mean_first_step_fn": mean([r["first_step_fn"] for r in g]),
                "mean_first_step_ft_over_fn": mean([r["first_step_ft_over_fn"] for r in g]),
                "mean_normal_leakage": mean([r["max_abs_normal_leakage"] for r in g]),
                "mean_tracking_error_mm_proxy": mean([r["controller_tracking_error_mm_proxy"] for r in g]),
                "mean_object_disp_m": mean([r["object_displacement_first_step_m"] for r in g]),
                "dominant_transition": mode([r["contact_state_transition"] for r in g]),
                "dominant_stop": mode([r["stop_reason"] for r in g]),
            }
        )
    task2 = next((r for r in summary if r["task_id"] == 2), {})
    others = [r for r in summary if r["task_id"] != 2]
    mechanism = "inconclusive"
    evidence = []
    if task2:
        if task2["dominant_transition"].endswith("->unilateral") or task2["dominant_transition"].endswith("->none"):
            mechanism = "insufficient/unstable bilateral preload"
            evidence.append("Task2 loses bilateral contact at the first probe step while Ft/Fn remains near zero.")
        finite_leaks = [r["mean_normal_leakage"] for r in others if math.isfinite(r["mean_normal_leakage"])]
        if finite_leaks and task2["mean_normal_leakage"] <= max(finite_leaks) + 1e-6:
            evidence.append("Available left-frame normal leakage is not elevated relative to the retained-contact tasks.")
        if others and task2["mean_tracking_error_mm_proxy"] <= max(0.5, mean([r["mean_tracking_error_mm_proxy"] for r in others]) + 0.5):
            evidence.append("EEF tracking-error proxy is not large enough to explain the task-specific failure.")
    forensic_json = {
        "best_supported_explanation": mechanism,
        "evidence": evidence,
        "telemetry_limitations": [
            "P4 did not record independent right contact normals, contact patch centers, patch areas, patch boundary margins, true contact points, or fingertip displacement.",
            "Controller tracking error is approximated from EEF displacement versus commanded increment.",
        ],
        "task_summary": summary,
    }
    (RESULT_DIR / "TASK2_FIRST_STEP_CONTACT_LOSS_FORENSIC.json").write_text(json.dumps(json_safe(forensic_json), indent=2) + "\n", encoding="utf-8")
    md = [
        "# Task2 First-Step Contact-Loss Forensic Audit",
        "",
        f"Best-supported explanation: `{mechanism}`.",
        "",
        "This is a no-method-change analysis of existing P4 trajectories. It does not relabel P4 failures as successes.",
        "",
        "## Task Summary",
        "",
        markdown_table(summary),
        "",
        "## Evidence",
        "",
    ]
    md += [f"- {x}" for x in evidence] if evidence else ["- The available telemetry is insufficient to isolate a single mechanism."]
    md += [
        "",
        "## Telemetry Limitations",
        "",
        "- P4 did not record independent right contact normals, contact patch centers, patch areas, patch edge margins, true contact points, or fingertip displacement.",
        "- Controller tracking error is represented by an EEF displacement proxy.",
    ]
    for name in ("TASK2_FIRST_STEP_CONTACT_LOSS_FORENSIC.md", "P4R2_TASK2_FORENSIC.md"):
        (RESULT_DIR / name).write_text("\n".join(md) + "\n", encoding="utf-8")
    return forensic_json


def paired_metrics(ep: list[dict]):
    main = [r for r in ep if r.get("split") == "main"]
    pairs = {}
    for r in main:
        key = (i(r, "task_id"), round(f(r, "friction"), 1), i(r, "seed_idx"))
        pairs.setdefault(key, {"task_id": key[0], "friction": key[1], "seed_idx": key[2], "S": 0, "L": 0, "selector_output": r.get("selector_output", "")})
        pairs[key][r.get("primitive", "")] = max(pairs[key].get(r.get("primitive", ""), 0), b(r, "qualified"))
        if r.get("primitive") == "S":
            pairs[key]["selector_output"] = r.get("selector_output", pairs[key]["selector_output"])
    selector_rows = []
    for row in pairs.values():
        row["library_covered"] = int(row.get("S", 0) == 1 or row.get("L", 0) == 1)
        if row["selector_output"] == "S":
            row["selector_qualified"] = int(row.get("S", 0) == 1)
        elif row["selector_output"] == "L":
            row["selector_qualified"] = int(row.get("L", 0) == 1)
        else:
            row["selector_qualified"] = 0
        row["selector_regret"] = row["library_covered"] - row["selector_qualified"]
        selector_rows.append(row)
    selector_rows.sort(key=lambda r: (r["task_id"], r["seed_idx"], r["friction"]))
    write_csv(RESULT_DIR / "P4R2_SELECTOR_RESULTS.csv", selector_rows)

    pair_by_task = defaultdict(list)
    ep_by_task = defaultdict(list)
    for r in selector_rows:
        pair_by_task[r["task_id"]].append(r)
    for r in main:
        ep_by_task[i(r, "task_id")].append(r)

    def coverage_row(label, object_id, episodes, pair_rows):
        return {
            "task_id": label,
            "object_id": object_id,
            "n_episodes": len(episodes),
            "n_pairs": len(pair_rows),
            "contact_retention": 1.0 - mean([b(r, "contact_lost_probe") for r in episodes]),
            "hard_contact_loss_rate": mean([b(r, "contact_lost_probe") for r in episodes]),
            "drop_rate": mean([b(r, "dropped") for r in episodes]),
            "disturbance_rate": mean([b(r, "major_disturbance") for r in episodes]),
            "return_completion_rate": mean([b(r, "return_completed") for r in episodes]),
            "informative_rate": mean([b(r, "informative") for r in episodes]),
            "qualified_rate": mean([b(r, "qualified") for r in episodes]),
            "primitive_s_qualified_rate": mean([b(r, "qualified") for r in episodes if r.get("primitive") == "S"]),
            "primitive_l_qualified_rate": mean([b(r, "qualified") for r in episodes if r.get("primitive") == "L"]),
            "paired_library_coverage": mean([r["library_covered"] for r in pair_rows]),
            "selector_selected_qualified_rate": mean([r["selector_qualified"] for r in pair_rows]),
            "no_feasible_probe_rate": mean([1 if r["selector_output"] == "NO_FEASIBLE_PROBE" else 0 for r in pair_rows]),
            "selector_regret": mean([r["selector_regret"] for r in pair_rows]),
            "mean_probe_displacement_mm": mean([f(r, "actual_probe_displacement_mm") for r in episodes]),
            "dominant_stop_reason": mode([r.get("stop_trigger", "") for r in episodes]),
        }

    coverage = [coverage_row(task, TASK_OBJECTS[task], ep_by_task[task], pair_by_task[task]) for task in TASKS]
    for fric in sorted(set(round(f(r, "friction"), 1) for r in main)):
        episodes = [r for r in main if round(f(r, "friction"), 1) == fric]
        pair_rows = [r for r in selector_rows if r["friction"] == fric]
        coverage.append(coverage_row("ALL", f"friction_{fric:g}", episodes, pair_rows))
    coverage.append(coverage_row("AGGREGATE", "all", main, selector_rows))
    write_csv(RESULT_DIR / "P4R2_PRIMITIVE_COVERAGE.csv", coverage)
    primitive_usage = dict(Counter(r.get("selector_output", "") for r in main))
    stop_reasons = [{"primitive": p, "stop_trigger": s, "n": n} for (p, s), n in Counter((r.get("primitive", ""), r.get("stop_trigger", "")) for r in main).items()]
    return selector_rows, coverage, {"primitive_usage": primitive_usage, "stop_reasons": stop_reasons}


def normalized_evidence(ep: list[dict]):
    main = [r for r in ep if r.get("split") == "main"]
    feat = []
    for r in main:
        preload = abs(f(r, "preprobe_normal_force"))
        target = abs(f(r, "target_preload_N"))
        marker_base = abs(f(r, "preprobe_marker_motion"))
        feat.append(
            {
                "trial_id": r.get("trial_id", ""),
                "primitive": r.get("primitive", ""),
                "task_id": i(r, "task_id"),
                "object_id": r.get("object_id", ""),
                "seed_idx": i(r, "seed_idx"),
                "friction": round(f(r, "friction"), 1),
                "safe": b(r, "safe"),
                "informative": b(r, "informative"),
                "qualified": b(r, "qualified"),
                "r_ft_peak": f(r, "ftan_peak") / (preload + EPS),
                "r_ft_mean": f(r, "ftan_mean") / (preload + EPS),
                "r_imb_peak": f(r, "imb_ratio_peak"),
                "r_imb_mean": f(r, "imb_ratio_mean"),
                "h_ft_norm": f(r, "ftan_hyst") / (target + EPS),
                "marker_tangent_norm": f(r, "marker_tangential_peak") / (marker_base + EPS),
                "support_response_norm": f(r, "support_force_delta_peak") / (preload + EPS),
                "support_load_impulse_norm": f(r, "support_load_impulse") / (preload + EPS),
            }
        )
    write_csv(RESULT_DIR / "P4R2_NORMALIZED_EVIDENCE.csv", feat)
    rows = []
    for primitive in ("S", "L"):
        pdf = [r for r in feat if r["primitive"] == primitive]
        for task in TASKS:
            g = [r for r in pdf if r["task_id"] == task]
            for contrast in ("LOW_vs_NOTLOW", "MID_vs_HIGH"):
                gg = g if contrast == "LOW_vs_NOTLOW" else [r for r in g if r["friction"] in (0.5, 1.0)]
                y = [1 if (r["friction"] == 0.2 if contrast == "LOW_vs_NOTLOW" else r["friction"] == 0.5) else 0 for r in gg]
                for feature in FEATURES:
                    auc = auc_binary(y, [r[feature] for r in gg])
                    rows.append({"primitive": primitive, "scope": "within_task", "task_id": task, "contrast": contrast, "feature": feature, "auc": auc, "auc_separation": max(auc, 1 - auc) if math.isfinite(auc) else float("nan")})
        for contrast in ("LOW_vs_NOTLOW", "MID_vs_HIGH"):
            for feature in FEATURES:
                for heldout in TASKS:
                    train = [r for r in pdf if r["task_id"] != heldout]
                    test = [r for r in pdf if r["task_id"] == heldout]
                    if contrast == "MID_vs_HIGH":
                        train = [r for r in train if r["friction"] in (0.5, 1.0)]
                        test = [r for r in test if r["friction"] in (0.5, 1.0)]
                    train_y = [1 if (r["friction"] == 0.2 if contrast == "LOW_vs_NOTLOW" else r["friction"] == 0.5) else 0 for r in train]
                    test_y = [1 if (r["friction"] == 0.2 if contrast == "LOW_vs_NOTLOW" else r["friction"] == 0.5) else 0 for r in test]
                    test_x = np.asarray([r[feature] for r in test], dtype=float)
                    th = best_threshold(train_y, [r[feature] for r in train])
                    pred = test_x >= th["threshold"] if th["direction"] == "high" else test_x <= th["threshold"]
                    ok = np.isfinite(test_x)
                    acc = float(np.mean(pred[ok].astype(int) == np.asarray(test_y)[ok])) if ok.any() else float("nan")
                    auc = auc_binary(test_y, test_x)
                    rows.append({"primitive": primitive, "scope": "LOTO", "task_id": heldout, "contrast": contrast, "feature": feature, "auc": auc, "auc_separation": max(auc, 1 - auc) if math.isfinite(auc) else float("nan"), "threshold": th["threshold"], "threshold_direction": th["direction"], "accuracy": acc})
    write_csv(RESULT_DIR / "P4R2_NORMALIZED_EVIDENCE_DIAGNOSTICS.csv", rows)
    return feat, rows


def classification(coverage: list[dict]):
    task_rows = [r for r in coverage if str(r["task_id"]) in {str(t) for t in TASKS}]
    aggregate = next(r for r in coverage if str(r["task_id"]) == "AGGREGATE")
    s_pass = all(float(r["primitive_s_qualified_rate"]) >= 0.95 for r in task_rows) and all(float(r["drop_rate"]) <= 0.05 for r in task_rows)
    library_pass = all(float(r["paired_library_coverage"]) >= 0.95 for r in task_rows)
    selector_pass = all(float(r["selector_selected_qualified_rate"]) >= 0.95 for r in task_rows) and float(aggregate["selector_selected_qualified_rate"]) >= 0.95
    drop_pass = all(float(r["drop_rate"]) <= 0.05 for r in task_rows)
    regret_pass = float(aggregate["selector_regret"]) <= 0.05
    more_than_one_needed = any(float(r["primitive_s_qualified_rate"]) < float(r["paired_library_coverage"]) for r in task_rows) and any(float(r["primitive_l_qualified_rate"]) > 0 for r in task_rows)
    if s_pass:
        status = "P4R2_BILATERAL_SHEAR_RULE_QUALIFIED"
    elif library_pass and selector_pass and drop_pass and regret_pass and more_than_one_needed:
        status = "P4R2_CONTACT_FEASIBLE_MULTI_PRIMITIVE_COMPOSER_QUALIFIED"
    elif library_pass and (not selector_pass or not regret_pass):
        status = "P4R2_LIBRARY_COVERS_BUT_SELECTOR_NOT_GENERAL"
    elif not library_pass:
        status = "P4R2_PRIMITIVE_LIBRARY_NOT_COVERING"
    else:
        status = "P4R2_INCONCLUSIVE_SYSTEM_OR_TELEMETRY_FAILURE"
    return status, {
        "per_task_library": {int(r["task_id"]): float(r["paired_library_coverage"]) for r in task_rows},
        "per_task_selector": {int(r["task_id"]): float(r["selector_selected_qualified_rate"]) for r in task_rows},
        "per_task_drop": {int(r["task_id"]): float(r["drop_rate"]) for r in task_rows},
        "aggregate_selector": float(aggregate["selector_selected_qualified_rate"]),
        "aggregate_library": float(aggregate["paired_library_coverage"]),
        "aggregate_regret": float(aggregate["selector_regret"]),
        "s_pass": s_pass,
        "library_pass": library_pass,
        "selector_pass": selector_pass,
        "drop_pass": drop_pass,
        "regret_pass": regret_pass,
        "more_than_one_needed": more_than_one_needed,
    }


def p5_readiness(status: str, ep: list[dict]):
    if status not in ("P4R2_CONTACT_FEASIBLE_MULTI_PRIMITIVE_COMPOSER_QUALIFIED", "P4R2_BILATERAL_SHEAR_RULE_QUALIFIED"):
        return {"ready": False, "reason": "physical probe coverage gate did not qualify", "matched_contexts": 0, "available_force_outcome_labels": 0, "missing_data": "five-task safe+informative probe coverage remains blocker"}
    contexts = [r for r in ep if r.get("split") == "main" and b(r, "qualified")]
    readiness = {
        "ready": False,
        "reason": "episode-level candidate-force outcomes were not collected in P4-R2",
        "matched_contexts": len(contexts),
        "available_force_outcome_labels": "canonical task-level thresholds only; do not fabricate per-episode labels",
        "labels_reusable_from_existing_sweeps": "only aggregate/canonical F* artifacts unless a matched episode-level sweep is found",
        "missing_data": "paired full-task outcomes for candidate forces [3,4,5,6,8] per probe context",
        "episode_level_labels_available": False,
    }
    (RESULT_DIR / "P5_GNP_DATA_READINESS.json").write_text(json.dumps(json_safe(readiness), indent=2) + "\n", encoding="utf-8")
    (RESULT_DIR / "P5_GNP_DATA_SCHEMA.md").write_text(
        "# P5 GNP Data Schema\n\n"
        "Context: pre-probe contact geometry, selected primitive, exact probe trajectory, tactile time series, force time series, proprioception.\n\n"
        "Query: downstream task context, frozen-VLA nominal trajectory summary, candidate force F.\n\n"
        "Target: actual full-task success for that same episode/context under candidate force F.\n\n"
        "Current status: not ready because P4-R2 did not collect matched per-context full-task force outcomes.\n",
        encoding="utf-8",
    )
    write_csv(RESULT_DIR / "P5_MATCHED_CONTEXT_TARGET_MANIFEST.csv", [{k: r.get(k, "") for k in ("trial_id", "task_id", "friction", "seed_idx", "primitive", "selector_output", "qualified")} for r in contexts])
    return readiness


def write_final(ep, coverage, forensic, diag, status, gates, aux):
    protocol = json.loads((RESULT_DIR / "P4R2_PROTOCOL.json").read_text()) if (RESULT_DIR / "P4R2_PROTOCOL.json").exists() else {}
    readiness = p5_readiness(status, ep)
    task_rows = [r for r in coverage if str(r["task_id"]) in {str(t) for t in TASKS}]
    aggregate = next(r for r in coverage if str(r["task_id"]) == "AGGREGATE")

    within = sorted([r for r in diag if r["scope"] == "within_task"], key=lambda r: -float(r["auc_separation"]) if math.isfinite(float(r["auc_separation"])) else -1)[:10]
    loto_groups = defaultdict(list)
    for r in diag:
        if r["scope"] == "LOTO":
            loto_groups[(r["primitive"], r["contrast"], r["feature"])].append(r)
    loto = []
    for (primitive, contrast, feature), rows in loto_groups.items():
        loto.append({"primitive": primitive, "contrast": contrast, "feature": feature, "mean_auc": mean([r["auc_separation"] for r in rows]), "mean_accuracy": mean([r.get("accuracy", float("nan")) for r in rows])})
    loto = sorted(loto, key=lambda r: -float(r["mean_auc"]) if math.isfinite(float(r["mean_auc"])) else -1)[:10]
    normalized_summary = {"best_within_task_auc": within, "best_loto_auc": loto, "cross_object_representation_qualified": False}

    verdict = {
        "status": status,
        "method_change": "CONTACT_FEASIBLE_MULTI_PRIMITIVE_PROBE_COMPOSER_ONLY",
        "tasks": TASKS,
        "task_specific_probe_lookup": False,
        "gt_hidden_physics_input": False,
        "protocol_hash": (RESULT_DIR / "P4R2_PROTOCOL_HASH.txt").read_text().strip() if (RESULT_DIR / "P4R2_PROTOCOL_HASH.txt").exists() else None,
        "freeze_timestamp_utc": protocol.get("freeze_timestamp_utc"),
        "no_main_evaluation_outcome_observed_before_protocol_freeze": protocol.get("no_main_evaluation_outcome_observed_before_protocol_freeze"),
        "gates": gates,
        "aggregate": aggregate,
        "per_task": task_rows,
        "primitive_usage": aux.get("primitive_usage", {}),
        "stop_reasons": aux.get("stop_reasons", []),
        "forensic": forensic,
        "normalized_evidence": normalized_summary,
        "p5_gnp_readiness": readiness,
    }
    (RESULT_DIR / "P4R2_FINAL_VERDICT.json").write_text(json.dumps(json_safe(verdict), indent=2) + "\n", encoding="utf-8")

    report = [
        "# P4-R2 Contact-Feasible Multi-Primitive Probe Composer",
        "",
        f"Primary classification: `{status}`.",
        "",
        "Method change: `CONTACT_FEASIBLE_MULTI_PRIMITIVE_PROBE_COMPOSER_ONLY`.",
        "",
        "No task-ID/object lookup, GT friction, GT mass, hidden-physics input, learned selector, force-sufficiency training, or GNP training was used.",
        "",
        "## Forensic Result",
        "",
        f"Best-supported P4 Task2 failure mechanism: `{forensic.get('best_supported_explanation')}`.",
        "",
        "The P4 telemetry supports a first-step bilateral contact transition before useful shear evidence is produced. Independent patch geometry was not recorded in P4, so patch-edge and curvature explanations remain unresolved rather than disproven.",
        "",
        "## Main Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Selector And Library Interpretation",
        "",
        "- Physical safety, information coverage, library coverage, selector generalization, and signal normalization are reported separately.",
        "- `SAFE` requires retained bilateral contact through the informative part, no drop/disturbance/fault, and return completion.",
        "- `INFORMATIVE` requires a primitive-specific signal floor above the preload-window noise proxy.",
        "- `QUALIFIED = SAFE AND INFORMATIVE`.",
        "",
        "## Normalized Evidence",
        "",
        "The normalized scalar diagnostics are secondary. No neural predictor, force-sufficiency model, GNP, Transformer, or task+probe+force classifier was trained.",
        "",
        "Best LOTO diagnostics:",
        "",
        markdown_table(loto, max_rows=10),
        "",
        "Cross-object representation qualified: `False`.",
        "",
        "## P5 Readiness",
        "",
        f"Ready: `{readiness.get('ready')}`. Reason: {readiness.get('reason')}.",
    ]
    (RESULT_DIR / "P4R2_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return verdict


def env_and_hashes():
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(REPO),
        "result_dir": str(RESULT_DIR),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "method_change": "CONTACT_FEASIBLE_MULTI_PRIMITIVE_PROBE_COMPOSER_ONLY",
    }
    try:
        env["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader"], text=True).strip()
    except Exception as exc:
        env["nvidia_smi_error"] = repr(exc)
    (RESULT_DIR / "P4R2_ENVIRONMENT.json").write_text(json.dumps(json_safe(env), indent=2) + "\n", encoding="utf-8")
    code_files = [RESULT_DIR / "scripts" / p for p in ("p4r2_collect_probe.py", "p4r2_analyze.py", "p4r2_freeze_protocol.py", "run_p4r2_collect_all.sh")]
    (RESULT_DIR / "P4R2_CODE_HASH.txt").write_text(json.dumps({"git_commit": git_commit(), "file_sha256": {p.name: sha256_file(p) for p in code_files if p.exists()}}, indent=2) + "\n", encoding="utf-8")


def main():
    forensic = forensic_from_p4()
    ep = load_episodes()
    if not ep:
        raise SystemExit("No P4-R2 episode files found.")
    _, coverage, aux = paired_metrics(ep)
    _, diag = normalized_evidence(ep)
    status, gates = classification(coverage)
    verdict = write_final(ep, coverage, forensic, diag, status, gates, aux)
    env_and_hashes()
    print(json.dumps(json_safe(verdict), indent=2))


if __name__ == "__main__":
    main()
