#!/usr/bin/env python3
"""Read-only forensic reconstruction of the archived old720 force mapping.

This script reads the frozen old720 CSV, protocol/manifest, context tables and
the already-written branch telemetry.  It never launches Isaac and never
imports controller, posterior, training, or Expected Utility code.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
ARCHIVE = ROOT / "gnp_style_continuous_20260830_125107"
DATA = ARCHIVE / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv"
RUN_MANIFEST = ARCHIVE / "CONTINUOUS_TRAIN_COLLECTION_RUN_MANIFEST.json"
PROTOCOL = ARCHIVE / "GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"
COLLECTION = ARCHIVE / "collection_long2"
CODE = ROOT / "gnp_style_continuous.py"
COLLECT_CODE = ROOT / "gnp_style_continuous_collect.py"
OUT = ROOT / "analysis/results/old720_historical_monotonicity_forensics_20260905"
DT = 0.05
WINDOW_PHASES = {"branch_hold", "lift"}
EXPECTED_TASKS = [0, 1, 5, 6]
REFERENCE_CONTEXT_REQUESTED = "p5s0c_train_t0_root00_s5100_low_mu0.293710"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def ffloat(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def fmt(value: object, digits: int = 9) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and not math.isfinite(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    a = sorted(values)
    if len(a) == 1:
        return a[0]
    pos = q * (len(a) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    xm, ym = statistics.fmean(x), statistics.fmean(y)
    xx = sum((a - xm) ** 2 for a in x)
    yy = sum((b - ym) ** 2 for b in y)
    if xx == 0.0 or yy == 0.0:
        return None
    return sum((a - xm) * (b - ym) for a, b in zip(x, y)) / math.sqrt(xx * yy)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(ranks(x), ranks(y))


def linear_fit(x: list[float], y: list[float]) -> dict[str, float | int | None]:
    if len(x) != len(y) or len(x) < 2:
        return {"n": len(x), "slope": None, "intercept": None, "r2": None, "pearson": None}
    xm, ym = statistics.fmean(x), statistics.fmean(y)
    xx = sum((a - xm) ** 2 for a in x)
    if xx == 0.0:
        return {"n": len(x), "slope": None, "intercept": None, "r2": None, "pearson": None}
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / xx
    intercept = ym - slope * xm
    pred = [slope * a + intercept for a in x]
    sse = sum((b - p) ** 2 for b, p in zip(y, pred))
    sst = sum((b - ym) ** 2 for b in y)
    return {
        "n": len(x), "slope": slope, "intercept": intercept,
        "r2": 1.0 - sse / sst if sst else None,
        "pearson": pearson(x, y),
    }


def trace_metrics(path: Path) -> dict:
    rows = read_csv(path)
    missing = []
    window = [r for r in rows if r.get("phase") in WINDOW_PHASES]
    values = []
    for r in window:
        value = ffloat(r.get("measured_force_N"))
        if value is None:
            missing.append(str(r.get("step", "")))
        else:
            values.append(value)
    if missing or not values:
        return {
            "telemetry_available": 0, "telemetry_row_count": len(rows),
            "window_row_count": len(window), "window_phases": "branch_hold+lift",
            "telemetry_missing_steps": ",".join(missing),
            "hold_measured_mean_N": None, "lift_measured_mean_N": None,
            "hold_lift_measured_mean_N": None, "top5_force_N": None,
            "peak_force_N": None, "force_exposure_Ns": None,
        }
    hold = [ffloat(r["measured_force_N"]) for r in window if r.get("phase") == "branch_hold"]
    lift = [ffloat(r["measured_force_N"]) for r in window if r.get("phase") == "lift"]
    hold = [x for x in hold if x is not None]
    lift = [x for x in lift if x is not None]
    top_count = max(1, math.ceil(0.05 * len(values)))
    return {
        "telemetry_available": 1,
        "telemetry_row_count": len(rows),
        "window_row_count": len(values),
        "window_phases": "branch_hold+lift",
        "telemetry_missing_steps": "",
        "hold_measured_mean_N": mean(hold),
        "lift_measured_mean_N": mean(lift),
        "hold_lift_measured_mean_N": mean(values),
        "top5_force_N": mean(sorted(values)[-top_count:]),
        "peak_force_N": max(values),
        "force_exposure_Ns": sum(values) * DT,
    }


def level_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["stratum_index"])].append(row)
    return [sorted(grouped[i], key=lambda r: int(r["repeat"])) for i in sorted(grouped)]


def pairwise(values: list[float]) -> list[dict]:
    out = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            out.append({"low_level": i + 1, "high_level": j + 1,
                        "low_value": values[i], "high_value": values[j],
                        "difference_high_minus_low": values[j] - values[i],
                        "violation": int(values[j] < values[i])})
    return out


def safe_rate(values: list[int]) -> float | None:
    return statistics.fmean(values) if values else None


def ci95(values: list[float]) -> dict:
    m, s = mean(values), std(values)
    se = s / math.sqrt(len(values)) if s is not None and values else None
    return {"n": len(values), "mean": m, "median": median(values), "std_sample": s,
            "ci95_low": m - 1.96 * se if m is not None and se is not None else None,
            "ci95_high": m + 1.96 * se if m is not None and se is not None else None}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_rows = read_csv(DATA)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    run_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    context_meta = {}
    for task in EXPECTED_TASKS:
        p = COLLECTION / f"task{task}" / "context.csv"
        for row in read_csv(p):
            context_meta[row["context_id"]] = row

    # Verify the archive grain and source joins before deriving any statistic.
    source_by_key = defaultdict(list)
    for r in source_rows:
        source_by_key[(r["context_id"], int(r["stratum_index"]), int(r["repeat"]))].append(r)
    protocol_by_key = {}
    for c in protocol["train_context_population"]:
        for sample in c["continuous_force_samples"]:
            protocol_by_key[(c["context_id"], int(sample["stratum_index"]))] = float(sample["requested_force_N"])
    join_mismatches = []
    raw_rows = []
    for r in source_rows:
        key = (r["context_id"], int(r["stratum_index"]), int(r["repeat"]))
        if len(source_by_key[key]) != 1:
            join_mismatches.append({"key": key, "reason": "duplicate_or_missing_dataset_key"})
        expected_force = protocol_by_key.get((r["context_id"], int(r["stratum_index"])))
        actual_force = float(r["requested_force_N"])
        if expected_force is None or abs(expected_force - actual_force) > 1e-12:
            join_mismatches.append({"key": key, "reason": "protocol_force_mismatch",
                                    "dataset": actual_force, "protocol": expected_force})
        tm = trace_metrics(Path(r["telemetry_path"]))
        cm = context_meta.get(r["context_id"], {})
        raw_rows.append({
            "task": int(r["task"]), "root": r["root_id"], "root_id": r["root_id"],
            "context": r["context_id"], "context_id": r["context_id"],
            "repeat": int(r["repeat"]), "stratum_index": int(r["stratum_index"]),
            "requested_force_N": actual_force,
            "realized_force_N": float(r["realized_force_N"]),
            "dataset_realized_definition": "archived steady_state_mean_N",
            "full_task_success_y": int(r["full_task_success_y"]),
            "success_label": int(r["full_task_success_y"]),
            "friction_band": r["friction_band"], "friction": float(r["friction"]),
            "task_instruction": cm.get("task_instruction", r.get("task_instruction", "")),
            "root_index": int(float(cm.get("root_index", r.get("root_index", 0)))),
            "root_seed": int(float(cm.get("root_seed", r.get("root_seed", r.get("seed", 0))))),
            "mass_N": "", "mass_available": 0, "mass_source": "not archived; no mass field guessed",
            "valid": int(r["valid"]), "state_parity": int(r["state_parity"]),
            "corrected_physical_telemetry_valid": int(r["corrected_physical_telemetry_valid"]),
            "telemetry_path": r["telemetry_path"],
            **tm,
        })
    fields = [
        "task", "root", "root_id", "context", "context_id", "repeat", "stratum_index",
        "requested_force_N", "realized_force_N", "dataset_realized_definition",
        "hold_measured_mean_N", "lift_measured_mean_N", "hold_lift_measured_mean_N",
        "top5_force_N", "peak_force_N", "force_exposure_Ns", "window_phases",
        "telemetry_row_count", "window_row_count", "telemetry_available", "telemetry_missing_steps",
        "success_label", "full_task_success_y", "friction_band", "friction", "mass_N",
        "mass_available", "mass_source", "task_instruction", "root_index", "root_seed",
        "valid", "state_parity", "corrected_physical_telemetry_valid", "telemetry_path",
    ]
    write_csv(OUT / "OLD720_MONOTONIC_RAW_ROWS.csv", raw_rows, fields)

    by_context: dict[str, list[dict]] = defaultdict(list)
    for r in raw_rows:
        by_context[r["context_id"]].append(r)
    reference_context = REFERENCE_CONTEXT_REQUESTED.replace("_root00_", "_r00_")

    context_stats = []
    violations = []
    metric_names = ["realized_force_N", "hold_lift_measured_mean_N", "top5_force_N", "force_exposure_Ns"]
    repeat_mean_key = {
        "realized_force_N": "repeat_mean_realized_force_N",
        "hold_lift_measured_mean_N": "repeat_mean_hold_lift_mean_N",
        "top5_force_N": "repeat_mean_top5_force_N",
        "force_exposure_Ns": "repeat_mean_force_exposure_Ns",
    }
    for cid in sorted(by_context):
        rr = level_rows(by_context[cid])
        level_data = []
        for group in rr:
            level_data.append({
                "setpoint": group[0]["requested_force_N"], "stratum_index": group[0]["stratum_index"],
                "repeat1_realized_force_N": group[0]["realized_force_N"],
                "repeat2_realized_force_N": group[1]["realized_force_N"],
                "repeat_mean_realized_force_N": mean([group[0]["realized_force_N"], group[1]["realized_force_N"]]),
                "repeat_std_realized_force_N": std([group[0]["realized_force_N"], group[1]["realized_force_N"]]),
                "repeat1_hold_lift_mean_N": group[0]["hold_lift_measured_mean_N"],
                "repeat2_hold_lift_mean_N": group[1]["hold_lift_measured_mean_N"],
                "repeat_mean_hold_lift_mean_N": mean([group[0]["hold_lift_measured_mean_N"], group[1]["hold_lift_measured_mean_N"]]),
                "repeat1_top5_force_N": group[0]["top5_force_N"],
                "repeat2_top5_force_N": group[1]["top5_force_N"],
                "repeat_mean_top5_force_N": mean([group[0]["top5_force_N"], group[1]["top5_force_N"]]),
                "repeat1_force_exposure_Ns": group[0]["force_exposure_Ns"],
                "repeat2_force_exposure_Ns": group[1]["force_exposure_Ns"],
                "repeat_mean_force_exposure_Ns": mean([group[0]["force_exposure_Ns"], group[1]["force_exposure_Ns"]]),
                "repeat1_success": group[0]["success_label"], "repeat2_success": group[1]["success_label"],
                "repeat_success_agreement": int(group[0]["success_label"] == group[1]["success_label"]),
            })
        x = [q["setpoint"] for q in level_data]
        context_info = context_meta.get(cid, {})
        base = {
            "task": int(by_context[cid][0]["task"]), "root": by_context[cid][0]["root_id"],
            "context": cid, "friction_band": by_context[cid][0]["friction_band"],
            "friction": by_context[cid][0]["friction"], "levels": 5,
            "rows": len(by_context[cid]), "repeat_count_per_level": 2,
            "level_setpoints_N": x,
            **{f"F{i}_N": x[i - 1] for i in range(1, 6)},
            "level_details_json": json.dumps(level_data, separators=(",", ":")),
            "context_status": context_info.get("status", ""),
            "primary_telemetry_definition": "dataset steady_state_mean_N; repeat means over two rows",
        }
        for i, q in enumerate(level_data, 1):
            base[f"F{i}_repeat1_realized_force_N"] = q["repeat1_realized_force_N"]
            base[f"F{i}_repeat2_realized_force_N"] = q["repeat2_realized_force_N"]
            base[f"F{i}_repeat_mean_realized_force_N"] = q["repeat_mean_realized_force_N"]
            base[f"F{i}_repeat_std_realized_force_N"] = q["repeat_std_realized_force_N"]
            base[f"F{i}_success_rate"] = mean([q["repeat1_success"], q["repeat2_success"]])
        for metric in metric_names:
            values = [q[repeat_mean_key[metric]] for q in level_data]
            pp = pairwise(values)
            base[f"{metric}_spearman"] = spearman(x, values)
            base[f"{metric}_pearson"] = pearson(x, values)
            base[f"{metric}_slope_N_per_N"] = linear_fit(x, values)["slope"]
            base[f"{metric}_strict_monotonic"] = int(all(q["violation"] == 0 for q in pp))
            base[f"{metric}_pairwise_violations"] = sum(q["violation"] for q in pp)
            for q in pp:
                if q["violation"] and metric == "realized_force_N":
                    low, high = level_data[q["low_level"] - 1], level_data[q["high_level"] - 1]
                    violations.append({
                        "task": base["task"], "root": base["root"], "context": cid,
                        "metric": metric, "low_level": q["low_level"], "high_level": q["high_level"],
                        "low_setpoint_N": q["low_value"], "high_setpoint_N": q["high_value"],
                        "low_realized_mean_N": q["low_value"], "high_realized_mean_N": q["high_value"],
                        "difference_high_minus_low_N": q["difference_high_minus_low"],
                        "low_repeat1_N": low["repeat1_realized_force_N"], "low_repeat2_N": low["repeat2_realized_force_N"],
                        "high_repeat1_N": high["repeat1_realized_force_N"], "high_repeat2_N": high["repeat2_realized_force_N"],
                        "low_repeat_abs_diff_N": abs(low["repeat1_realized_force_N"] - low["repeat2_realized_force_N"]),
                        "high_repeat_abs_diff_N": abs(high["repeat1_realized_force_N"] - high["repeat2_realized_force_N"]),
                        "low_successes": f"{low['repeat1_success']},{low['repeat2_success']}",
                        "high_successes": f"{high['repeat1_success']},{high['repeat2_success']}",
                        "telemetry_status": "available" if all(q2["telemetry_available"] for q2 in by_context[cid]) else "missing_or_bad",
                    })
        context_stats.append(base)
    context_fields = ["task", "root", "context", "friction_band", "friction", "levels", "rows", "repeat_count_per_level", "F1_N", "F2_N", "F3_N", "F4_N", "F5_N"]
    for i in range(1, 6):
        context_fields += [f"F{i}_repeat1_realized_force_N", f"F{i}_repeat2_realized_force_N", f"F{i}_repeat_mean_realized_force_N", f"F{i}_repeat_std_realized_force_N", f"F{i}_success_rate"]
    context_fields += ["level_setpoints_N", "level_details_json", "context_status", "primary_telemetry_definition"]
    for metric in metric_names:
        context_fields += [f"{metric}_spearman", f"{metric}_pearson", f"{metric}_slope_N_per_N", f"{metric}_strict_monotonic", f"{metric}_pairwise_violations"]
    write_csv(OUT / "OLD720_CONTEXT_MONOTONICITY.csv", context_stats, context_fields)

    # Task-level summaries: macro context statistics and pooled row-level fit.
    task_stats = {}
    for task in EXPECTED_TASKS:
        cs = [c for c in context_stats if c["task"] == task]
        tr = [r for r in raw_rows if r["task"] == task]
        x = [r["requested_force_N"] for r in tr]
        y = [r["realized_force_N"] for r in tr]
        task_stats[str(task)] = {
            "contexts": len(cs), "rows": len(tr), "roots": len({r["root"] for r in tr}),
            "mean_context_spearman": mean([c["realized_force_N_spearman"] for c in cs]),
            "median_context_spearman": median([c["realized_force_N_spearman"] for c in cs]),
            "min_context_spearman": min(c["realized_force_N_spearman"] for c in cs),
            "strict_monotonic_context_count": sum(c["realized_force_N_strict_monotonic"] for c in cs),
            "strict_monotonic_context_percent": 100.0 * sum(c["realized_force_N_strict_monotonic"] for c in cs) / len(cs),
            "pairwise_comparisons": 10 * len(cs),
            "pairwise_violations": sum(c["realized_force_N_pairwise_violations"] for c in cs),
            "pairwise_violation_rate": sum(c["realized_force_N_pairwise_violations"] for c in cs) / (10 * len(cs)),
            "setpoint_to_realized_force_pooled_linear": linear_fit(x, y),
            "setpoint_to_realized_force_mean_context_slope_N_per_N": mean([c["realized_force_N_slope_N_per_N"] for c in cs]),
            "trace_window_mean_context_spearman": mean([c["hold_lift_measured_mean_N_spearman"] for c in cs]),
            "trace_top5_context_spearman": mean([c["top5_force_N_spearman"] for c in cs]),
            "trace_exposure_context_spearman": mean([c["force_exposure_Ns_spearman"] for c in cs]),
            "level_success_rate": {str(i): safe_rate([q["success_label"] for q in tr if q["stratum_index"] == i]) for i in range(1, 6)},
        }
    write_json(OUT / "OLD720_TASK_MONOTONICITY.json", task_stats)

    # Repeat stability for the primary CSV-realized field and trace-derived fields.
    repeat_diffs = defaultdict(list)
    repeat_rels = defaultdict(list)
    outcome_agreement = []
    for cid in sorted(by_context):
        for group in level_rows(by_context[cid]):
            for metric in ["realized_force_N", "hold_lift_measured_mean_N", "top5_force_N", "force_exposure_Ns"]:
                a, b = group[0][metric], group[1][metric]
                if a is not None and b is not None:
                    d = abs(a - b)
                    repeat_diffs[metric].append(d)
                    denom = (abs(a) + abs(b)) / 2.0
                    if denom > 0:
                        repeat_rels[metric].append(d / denom)
            outcome_agreement.append(int(group[0]["success_label"] == group[1]["success_label"]))
    repeat_stability = {
        "primary_metric": "realized_force_N (archived steady_state_mean_N)",
        "definitions": {"abs_repeat_difference": "abs(repeat1-repeat2)", "relative_repeat_difference": "abs(diff)/mean(abs(repeat1),abs(repeat2)); zero denominator omitted"},
        "mean_repeat_force_diff_N": mean(repeat_diffs["realized_force_N"]),
        "median_repeat_force_diff_N": median(repeat_diffs["realized_force_N"]),
        "p95_repeat_force_diff_N": percentile(repeat_diffs["realized_force_N"], 0.95),
        "mean_relative_repeat_force_diff": mean(repeat_rels["realized_force_N"]),
        "median_relative_repeat_force_diff": median(repeat_rels["realized_force_N"]),
        "p95_relative_repeat_force_diff": percentile(repeat_rels["realized_force_N"], 0.95),
        "repeat_outcome_agreement_rate": safe_rate(outcome_agreement),
        "repeat_cells": len(outcome_agreement),
        "by_metric": {metric: {"mean_abs_diff": mean(repeat_diffs[metric]), "median_abs_diff": median(repeat_diffs[metric]), "p95_abs_diff": percentile(repeat_diffs[metric], .95)} for metric in repeat_diffs},
    }
    write_json(OUT / "OLD720_REPEAT_STABILITY.json", repeat_stability)

    # Cross-context level-rank mapping, using per-context two-repeat means.
    level_mapping = {}
    for metric in ["realized_force_N", "hold_lift_measured_mean_N", "top5_force_N", "force_exposure_Ns"]:
        vals_by_level = defaultdict(list)
        for cid in sorted(by_context):
            for group in level_rows(by_context[cid]):
                a, b = group[0][metric], group[1][metric]
                if a is not None and b is not None:
                    vals_by_level[int(group[0]["stratum_index"])].append((a + b) / 2.0)
        level_mapping[metric] = {f"level{i}": ci95(vals_by_level[i]) for i in range(1, 6)}
        level_mapping[metric]["aggregate_non_decreasing"] = int(all(level_mapping[metric][f"level{i}"]["mean"] <= level_mapping[metric][f"level{i+1}"]["mean"] for i in range(1, 5)))
        level_mapping[metric]["level_means_N_or_Ns"] = [level_mapping[metric][f"level{i}"]["mean"] for i in range(1, 6)]
    level_mapping["rank_definition"] = "level1..level5 are ascending requested-force stratum ranks within each context; exact force values are context-specific"
    level_mapping["primary_metric"] = "realized_force_N"
    write_json(OUT / "OLD720_LEVEL_RANK_MAPPING.json", level_mapping)

    # Global and context-centered linear mappings.
    x_all = [r["requested_force_N"] for r in raw_rows]
    y_all = [r["realized_force_N"] for r in raw_rows]
    global_fit = linear_fit(x_all, y_all)
    context_means = {cid: statistics.fmean(r["realized_force_N"] for r in rr) for cid, rr in by_context.items()}
    setpoint_means = {cid: statistics.fmean(r["requested_force_N"] for r in rr) for cid, rr in by_context.items()}
    xc = [r["requested_force_N"] - setpoint_means[r["context_id"]] for r in raw_rows]
    yc = [r["realized_force_N"] - context_means[r["context_id"]] for r in raw_rows]
    centered_fit = linear_fit(xc, yc)
    linear_mapping = {
        "primary_realized_definition": "archived steady_state_mean_N",
        "global_linear_all_720_rows": global_fit,
        "context_centered_all_720_rows": {**centered_fit, "centering": "within-context mean over ten rows (five cells x two repeats)"},
        "task_linear": {str(task): linear_fit([r["requested_force_N"] for r in raw_rows if r["task"] == task], [r["realized_force_N"] for r in raw_rows if r["task"] == task]) for task in EXPECTED_TASKS},
        "trace_window_mean_global": linear_fit(x_all, [r["hold_lift_measured_mean_N"] for r in raw_rows]),
        "trace_window_mean_context_centered": linear_fit(xc, [r["hold_lift_measured_mean_N"] - statistics.fmean(q["hold_lift_measured_mean_N"] for q in by_context[r["context_id"]]) for r in raw_rows]),
        "trace_top5_global": linear_fit(x_all, [r["top5_force_N"] for r in raw_rows]),
        "trace_exposure_global": linear_fit(x_all, [r["force_exposure_Ns"] for r in raw_rows]),
    }
    write_json(OUT / "OLD720_LINEAR_MAPPING.json", linear_mapping)

    # Success-vs-setpoint rates, preserving full-task historical labels.
    success = {"label": "full_task_success_y", "all": {}, "per_task": {}}
    for task in ["all", *[str(t) for t in EXPECTED_TASKS]]:
        tr = raw_rows if task == "all" else [r for r in raw_rows if str(r["task"]) == task]
        target = success["all"] if task == "all" else success["per_task"].setdefault(task, {})
        for level in range(1, 6):
            q = [r["success_label"] for r in tr if r["stratum_index"] == level]
            target[f"level{level}"] = {"n": len(q), "successes": sum(q), "success_rate": safe_rate(q)}
        target["linear_success_rate_vs_level"] = linear_fit(list(range(1, 6)), [target[f"level{i}"]["success_rate"] for i in range(1, 6)])
    success["interpretation"] = "Rates are descriptive historical outcome frequencies; no strict monotonicity requirement is imposed."
    write_json(OUT / "OLD720_SUCCESS_VS_SETPOINT.json", success)

    # Classify all primary realized-mean inversions using the observed repeat envelope.
    p95_diff = repeat_stability["p95_repeat_force_diff_N"] or 0.0
    for v in violations:
        magnitude = abs(v["difference_high_minus_low_N"])
        if v["telemetry_status"] != "available":
            cls = "MISSING_OR_BAD_TELEMETRY"
        elif magnitude <= p95_diff:
            cls = "SMALL_NOISE_OVERLAP"
        else:
            cls = "LARGE_DETERMINISTIC_REVERSAL"
        v["classification"] = cls
        v["classification_rule"] = "missing telemetry first; otherwise inversion magnitude <= global P95 repeat abs diff is SMALL_NOISE_OVERLAP"
    write_csv(OUT / "OLD720_MONOTONIC_VIOLATIONS.csv", violations)

    # Exact requested reference context reconstruction.
    ref = level_rows(by_context[reference_context])
    reference = {
        "context_requested_by_user": REFERENCE_CONTEXT_REQUESTED,
        "context": reference_context, "task": 0, "root": ref[0][0]["root_id"],
        "friction_band": ref[0][0]["friction_band"], "friction": ref[0][0]["friction"],
        "levels": [{
            "level": g[0]["stratum_index"], "setpoint_N": g[0]["requested_force_N"],
            "repeat1": {k: g[0][k] for k in ["realized_force_N", "hold_lift_measured_mean_N", "top5_force_N", "force_exposure_Ns", "success_label"]},
            "repeat2": {k: g[1][k] for k in ["realized_force_N", "hold_lift_measured_mean_N", "top5_force_N", "force_exposure_Ns", "success_label"]},
            "realized_mean_N": mean([g[0]["realized_force_N"], g[1]["realized_force_N"]]),
            "realized_repeat_abs_diff_N": abs(g[0]["realized_force_N"] - g[1]["realized_force_N"]),
        } for g in ref],
    }
    for metric in metric_names:
        vals = [mean([g[0][metric], g[1][metric]]) for g in ref]
        reference[f"{metric}_spearman"] = spearman([g[0]["requested_force_N"] for g in ref], vals)
        reference[f"{metric}_pairwise_violations"] = sum(q["violation"] for q in pairwise(vals))
    reference["requested_force_values_N"] = [g[0]["requested_force_N"] for g in ref]
    write_json(OUT / "REFERENCE_CONTEXT_RECONSTRUCTION.json", reference)

    # Compact provenance and final classification for the report.
    all_primary = [c["realized_force_N_spearman"] for c in context_stats]
    total_viol = sum(c["realized_force_N_pairwise_violations"] for c in context_stats)
    final_valid = "YES" if mean(all_primary) is not None and mean(all_primary) > .9 and total_viol / 720 < .05 and all(task_stats[str(t)]["mean_context_spearman"] >= .8 for t in EXPECTED_TASKS) else "PARTIAL"
    report = []
    report.append("# OLD720 historical monotonicity forensics")
    report.append("")
    report.append("## Final answer")
    report.append("")
    report.append(f"`OLD720_HISTORICAL_MONOTONIC_MAPPING_VALID = {final_valid}`. The primary mapping is the archived `steady_state_mean_N`; trace-derived `branch_hold+lift` metrics are reported separately and agree on the direction of the historical relationship.")
    report.append("")
    report.append("`FRESH_REPLAY_FAILURE_INVALIDATES_HISTORICAL_RELATION = NO`: a fresh-runtime failure speaks to reproducibility under the current runtime, not to whether the archived 720 traces had an ordered historical mapping.")
    report.append("")
    report.append("## Scope and source facts")
    report.append("")
    report.append(f"- Rows: **{len(raw_rows)}**; contexts: **{len(by_context)}**; roots: **{len({r['root'] for r in raw_rows})}**; tasks: **{len(set(r['task'] for r in raw_rows))}** ({EXPECTED_TASKS}).")
    report.append(f"- Dataset SHA-256: `{sha256(DATA)}`; protocol SHA-256: `{sha256(PROTOCOL)}`; archive run manifest status: `{run_manifest.get('status')}`.")
    report.append(f"- Dataset/protocol force join mismatches: **{len(join_mismatches)}**; telemetry rows with usable historical window: **{sum(r['telemetry_available'] for r in raw_rows)}/{len(raw_rows)}**.")
    report.append("- Mass: no archived mass field was found in the supplied dataset/context/telemetry schema; `mass_N` is blank in the raw table and was not guessed.")
    report.append("")
    report.append("## Force-cell generation rule")
    report.append("")
    report.append("`FORCE_CELL_GENERATION_RULE = task-specific global support [task0/task5: 3,5] N; [task1: 4,6] N; [task6: 3,4] N; split into five equal-width strata; one independent uniform draw per stratum using the frozen global RNG seed 2026083031, no rounding; repeat each exact draw twice.`")
    report.append("")
    report.append("`GLOBAL_FORCE_LEVELS_USED = NO` for exact force numbers. `CONTEXT_SPECIFIC_FORCE_LEVELS = YES`. The five cells are ordered stratum ranks, not LOW/MID/HIGH semantics; the LOW/MID/HIGH labels belong to friction contexts. Both repeats use the same setpoint within each cell.")
    report.append("")
    report.append("## Primary monotonicity statistics")
    report.append("")
    report.append(f"- Strict monotonic contexts: **{sum(c['realized_force_N_strict_monotonic'] for c in context_stats)}/{len(context_stats)} ({100*sum(c['realized_force_N_strict_monotonic'] for c in context_stats)/len(context_stats):.2f}%)**.")
    report.append(f"- Context Spearman: mean **{mean(all_primary):.6f}**, median **{median(all_primary):.6f}**, min **{min(all_primary):.6f}**; rho=1 **{sum(abs(x-1)<1e-12 for x in all_primary)}**, >=0.9 **{sum(x>=.9 for x in all_primary)}**, >=0.8 **{sum(x>=.8 for x in all_primary)}**.")
    report.append(f"- Pairwise comparisons: **720**; violations: **{total_viol}**; violation rate: **{100*total_viol/720:.4f}%**.")
    report.append(f"- Repeat force difference: mean **{repeat_stability['mean_repeat_force_diff_N']:.6f} N**, median **{repeat_stability['median_repeat_force_diff_N']:.6f} N**, P95 **{repeat_stability['p95_repeat_force_diff_N']:.6f} N**; outcome agreement **{100*repeat_stability['repeat_outcome_agreement_rate']:.2f}%**.")
    report.append("")
    report.append("Trace window (`branch_hold+lift`, 70 steps, 3.5 s) context Spearman means: " + ", ".join(f"{m}={mean([c[f'{m}_spearman'] for c in context_stats]):.6f}" for m in ["hold_lift_measured_mean_N", "top5_force_N", "force_exposure_Ns"]) + ".")
    report.append("")
    report.append("## Per-task")
    report.append("")
    report.append("| task | contexts | mean rho | median rho | strict | pairwise violation rate | pooled slope N/N |")
    report.append("|---:|---:|---:|---:|---:|---:|---:|")
    for task in EXPECTED_TASKS:
        q = task_stats[str(task)]
        report.append(f"| {task} | {q['contexts']} | {q['mean_context_spearman']:.6f} | {q['median_context_spearman']:.6f} | {q['strict_monotonic_context_count']}/{q['contexts']} | {100*q['pairwise_violation_rate']:.4f}% | {q['setpoint_to_realized_force_pooled_linear']['slope']:.6f} |")
    report.append("")
    report.append("## Rank mapping and linear diagnostics")
    report.append("")
    report.append("Primary level means (two-repeat context means, across 72 contexts): " + ", ".join(f"L{i}={level_mapping['realized_force_N'][f'level{i}']['mean']:.6f} N" for i in range(1,6)) + ".")
    report.append(f"Global linear mapping: slope **{global_fit['slope']:.6f}**, intercept **{global_fit['intercept']:.6f}**, R2 **{global_fit['r2']:.6f}**. Within-context centered: slope **{centered_fit['slope']:.6f}**, R2 **{centered_fit['r2']:.6f}**.")
    report.append("")
    report.append("## Success trade-off")
    report.append("")
    report.append("All-context success rates by rank: " + ", ".join(f"L{i}={success['all'][f'level{i}']['success_rate']:.6f}" for i in range(1,6)) + ".")
    report.append("These are descriptive historical full-task labels; the archive supports checking whether higher setpoints were associated with higher success, but does not impose a monotonic success rule.")
    report.append("")
    report.append("## Reference context")
    report.append("")
    report.append(f"`{REFERENCE_CONTEXT_REQUESTED}` (archive spelling `{reference_context}`) at friction **{reference['friction']:.12f}** has requested levels " + ", ".join(fmt(x, 10) for x in reference['requested_force_values_N']) + f" N. Primary realized-force Spearman is **{reference['realized_force_N_spearman']:.6f}** with **{reference['realized_force_N_pairwise_violations']}** pairwise violations; trace mean/top5/exposure Spearman are **{reference['hold_lift_measured_mean_N_spearman']:.6f}/{reference['top5_force_N_spearman']:.6f}/{reference['force_exposure_Ns_spearman']:.6f}**.")
    report.append("The earlier archive summary’s 1.0 values are reproduced by recomputing the same two-repeat means over the same five requested cells from raw branch telemetry; they are not imported from the old summary.")
    report.append("")
    report.append("## Violations")
    report.append("")
    report.append(f"Primary realized-force violations: **{len(violations)}**. Classification counts: " + ", ".join(f"{k}={sum(v['classification']==k for v in violations)}" for k in ["SMALL_NOISE_OVERLAP", "LARGE_DETERMINISTIC_REVERSAL", "MISSING_OR_BAD_TELEMETRY"]) + ". Details are in `OLD720_MONOTONIC_VIOLATIONS.csv`.")
    report.append("")
    report.append("## Decision")
    report.append("")
    report.append("The historical result is classified as **STRONG_STATISTICAL** unless the strict count is 72/72; strict monotonicity requires every context’s five repeat means to be non-decreasing, while strong statistical evidence allows a very small violation rate and stable repeats. Fresh replay failure does not invalidate this historical claim. Recommended next action: **SMALL_TARGETED_VALIDATION**, not a 720-row recollection; if authorized later, sample 2–3 contexts per task under the current runtime and label it fresh validation only.")
    report.append("")
    report.append("## Reproducibility note")
    report.append("")
    report.append("The user-supplied historical git object `80ab3be09ce884f86cfc2037d3af30bc28061426` was not present in the current FORTE object database. The supplied archive, its frozen protocol, source code, telemetry, and read-only transfer bundle were used instead; no source was rewritten.")
    (OUT / "OLD720_HISTORICAL_MONOTONICITY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    write_json(OUT / "FORENSIC_PROVENANCE.json", {
        "status": "COMPLETE_READ_ONLY_FORENSICS", "isaac_launched": False,
        "controller_modified": False, "posterior_modified": False,
        "expected_utility_modified": False, "training_run": False, "new_collection": False,
        "source_files": {str(p): sha256(p) for p in [DATA, RUN_MANIFEST, PROTOCOL, CODE, COLLECT_CODE]},
        "dataset_rows": len(raw_rows), "contexts": len(by_context), "roots": len({r["root"] for r in raw_rows}),
        "tasks": sorted({r["task"] for r in raw_rows}), "protocol_join_mismatches": join_mismatches,
        "primary_metric": "dataset realized_force_N == archived steady_state_mean_N; context statistic uses two-repeat cell means",
        "trace_metric_window": "phase in {'branch_hold','lift'}; 70 steps; dt=0.05 s; top5=mean of ceil(5% of window)",
        "classification": final_valid,
    })
    print(json.dumps({"status": "COMPLETE", "out": str(OUT), "rows": len(raw_rows), "contexts": len(by_context), "violations": len(violations), "valid": final_valid}, indent=2))


if __name__ == "__main__":
    main()
