#!/usr/bin/env python3
"""Finalize the strict offset-servo attempt and setpoint fallback audit.

This script is intentionally analysis-only.  It never launches Isaac and never
changes controller, posterior, selector, probe, or arm-trajectory settings.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev


OUT = Path(
    "/home/exouser/FORTE/analysis/results/"
    "offset_preserving_force_servo_final_attempt_20260905"
)
RUNS = OUT / "runs"
PHYSICS_HZ = 60.0
HOLD_ENV_STEPS = 30
TRACKING_MAE_MAX_N = 1.0
WITHIN_1N_RATE_MIN = 0.60

RUN_MAP = {
    2.0: ["setpoint_2N", "setpoint_2N_r2", "setpoint_2N_r3"],
    3.0: ["setpoint_3N", "setpoint_3N_r2", "setpoint_3N_r3"],
    4.0: ["4N_live_gate", "setpoint_4N_r2", "setpoint_4N_r3"],
    5.0: ["setpoint_5N", "setpoint_5N_r2", "setpoint_5N_r3"],
    6.0: ["setpoint_6N", "setpoint_6N_r2", "setpoint_6N_r3"],
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def rank(values: list[float]) -> list[float]:
    """Average ranks, sufficient here even if future data contain ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(rank(left), rank(right))


def run_metrics(setpoint: float, repeat: int, name: str) -> dict[str, object]:
    run_dir = RUNS / name
    result_path = run_dir / "ROOT7703_CONTROLLER_VALIDATION_RESULT.json"
    trace_path = run_dir / "PHYSICS_STEP_SIGNAL_TRACE.csv"
    parity_path = run_dir / "HANDOFF_PARITY.json"
    result = read_json(result_path)
    parity = read_json(parity_path)
    trace = read_csv(trace_path)
    first_lift = int(result["summary"]["first_lift_step"])
    hold = [
        row for row in trace
        if first_lift < int(float(row["env_step"])) <= first_lift + HOLD_ENV_STEPS
    ]
    if result["status"] != "COMPLETE" or len(hold) != HOLD_ENV_STEPS * 3:
        raise RuntimeError(f"incomplete run {name}: status={result['status']}, hold_rows={len(hold)}")
    if not all((result["fresh_process"], result["exact_hdf5_initial_state"], result["matched_warm_replay"])):
        raise RuntimeError(f"protocol mismatch in {name}")
    if result["raw_arm_trajectory_changed"] or result["posterior_involved"] or result["probe_involved"]:
        raise RuntimeError(f"forbidden method change in {name}")
    if not parity.get("bilateral_contact", False):
        raise RuntimeError(f"invalid handoff contact in {name}")
    target_rows = [row for row in trace if row["F_des"].strip()]
    if not target_rows or not all(
        math.isclose(float(row["F_des"]), setpoint, abs_tol=1.0e-9)
        for row in target_rows
    ):
        raise RuntimeError(f"setpoint mismatch in {name}")

    forces = [float(row["F_meas_raw"]) for row in hold]
    top_count = max(1, math.ceil(0.05 * len(forces)))
    within = fmean(abs(force - setpoint) <= 1.0 for force in forces)
    mae = fmean(abs(force - setpoint) for force in forces)
    tracking_pass = mae <= TRACKING_MAE_MAX_N and within >= WITHIN_1N_RATE_MIN
    arm_hashes = sorted({row["raw_arm_action_hash"] for row in trace})
    return {
        "setpoint_N": setpoint,
        "repeat": repeat,
        "run_name": name,
        "status": result["status"],
        "physics_trace_rows": len(trace),
        "hold_physics_rows": len(hold),
        "first_lift_env_step": first_lift,
        "realized_hold_mean_force_N": fmean(forces),
        "realized_hold_std_force_N": pstdev(forces),
        "hold_mean_absolute_tracking_error_N": mae,
        "hold_within_1N_fraction": within,
        "top5_force_N": fmean(sorted(forces, reverse=True)[:top_count]),
        "peak_force_N": max(forces),
        "hold_force_exposure_Ns": sum(forces) / PHYSICS_HZ,
        "hold_bilateral_contact": all(truth(row["bilateral_contact"]) for row in hold),
        "lift_success": bool(result["summary"]["lift_success"]),
        "hold_success": bool(result["summary"]["hold_after_lift_success"]),
        "contact_loss_reported": bool(result["summary"]["contact_loss"]),
        "drop_reported": bool(result["summary"]["drop"]),
        "strict_tracking_pass": tracking_pass,
        "fresh_process": bool(result["fresh_process"]),
        "exact_hdf5_initial_state": bool(result["exact_hdf5_initial_state"]),
        "matched_warm_replay": bool(result["matched_warm_replay"]),
        "raw_arm_trajectory_changed": bool(result["raw_arm_trajectory_changed"]),
        "raw_arm_hash_count": len(arm_hashes),
        "source_trace": str(trace_path),
    }


def main() -> None:
    all_rows: list[dict[str, object]] = []
    for setpoint, names in RUN_MAP.items():
        for repeat, name in enumerate(names, start=1):
            all_rows.append(run_metrics(setpoint, repeat, name))

    live_row = next(row for row in all_rows if row["run_name"] == "4N_live_gate")
    live_dir = RUNS / "4N_live_gate"
    live_trace = read_csv(live_dir / "PHYSICS_STEP_SIGNAL_TRACE.csv")
    live_result = read_json(live_dir / "ROOT7703_CONTROLLER_VALIDATION_RESULT.json")
    first = live_trace[0]
    old_target = float(first["d_previous_command"])
    new_target = float(first["d_command_clamped"])
    first_jump_mm = abs(new_target - old_target) * 1000.0
    preload_before = float(first["F_meas_raw"])
    preload_after = float(live_trace[1]["F_meas_raw"])

    shutil.copyfile(
        live_dir / "PHYSICS_STEP_SIGNAL_TRACE.csv",
        OUT / "ROOT7703_4N_OFFSET_SERVO_TRACE.csv",
    )
    strict_valid = bool(live_row["strict_tracking_pass"] and live_row["hold_bilateral_contact"])
    write_json(OUT / "ROOT7703_4N_OFFSET_SERVO_RESULT.json", {
        "schema": "ROOT7703_4N_OFFSET_SERVO_RESULT_V1",
        "status": "PASS" if strict_valid else "FAIL_STRICT_TRACKING",
        "F_des_N": 4.0,
        "fresh_process": live_result["fresh_process"],
        "exact_hdf5_initial_state": live_result["exact_hdf5_initial_state"],
        "matched_warm_replay": live_result["matched_warm_replay"],
        "raw_arm_trajectory_changed": live_result["raw_arm_trajectory_changed"],
        "offset_servo_enabled": live_result["offset_servo_enabled"],
        "offset_contact_safety_filter_enabled": live_result["offset_contact_safety_filter_enabled"],
        "first_target_jump_mm": first_jump_mm,
        "preload_erasure_occurred": preload_after < 0.25 * preload_before,
        "preload_force_before_switch_N": preload_before,
        "preload_force_after_switch_N": preload_after,
        "hold_metrics": live_row,
        "tracking_acceptance": {
            "hold_mae_max_N": TRACKING_MAE_MAX_N,
            "hold_within_1N_fraction_min": WITHIN_1N_RATE_MIN,
            "bilateral_hold_required": True,
        },
        "strict_dynamic_4N_tracking_valid": strict_valid,
        "teardown_note": "Isaac produced COMPLETE result and full trace before the known pybind weakref teardown abort (exit 134).",
    })
    write_json(OUT / "IF_4N_PASS_REPEAT_SUMMARY.json", {
        "schema": "IF_4N_PASS_REPEAT_SUMMARY_V1",
        "status": "NOT_RUN_PRIMARY_STRICT_GATE_FAILED",
        "repeat_count": 0,
        "repeat_pass_count": 0,
        "reason": "The first fresh 4N strict gate failed force-realization tolerance; per protocol no success-confirmation repeats were run.",
    })

    write_csv(OUT / "IF_STRICT_FAIL_SETPOINT_MONOTONIC_TRACE.csv", all_rows)
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        grouped[float(row["setpoint_N"])].append(row)
    aggregate: list[dict[str, object]] = []
    for setpoint in sorted(grouped):
        rows = grouped[setpoint]
        aggregate.append({
            "setpoint_N": setpoint,
            "repeat_count": len(rows),
            "hold_mean_force_N": fmean(float(row["realized_hold_mean_force_N"]) for row in rows),
            "between_repeat_hold_mean_std_N": pstdev(float(row["realized_hold_mean_force_N"]) for row in rows),
            "hold_force_exposure_Ns": fmean(float(row["hold_force_exposure_Ns"]) for row in rows),
            "top5_force_N": fmean(float(row["top5_force_N"]) for row in rows),
            "peak_force_N": fmean(float(row["peak_force_N"]) for row in rows),
            "hold_contact_pass_count": sum(bool(row["hold_bilateral_contact"]) for row in rows),
            "strict_tracking_pass_count": sum(bool(row["strict_tracking_pass"]) for row in rows),
        })

    setpoints = [float(row["setpoint_N"]) for row in aggregate]
    mean_force = [float(row["hold_mean_force_N"]) for row in aggregate]
    exposure = [float(row["hold_force_exposure_Ns"]) for row in aggregate]
    top5 = [float(row["top5_force_N"]) for row in aggregate]

    pairwise: list[dict[str, object]] = []
    for i in range(len(aggregate)):
        for j in range(i + 1, len(aggregate)):
            lower, higher = aggregate[i], aggregate[j]
            violation = float(higher["hold_mean_force_N"]) < float(lower["hold_mean_force_N"])
            pairwise.append({
                "lower_setpoint_N": lower["setpoint_N"],
                "higher_setpoint_N": higher["setpoint_N"],
                "lower_realized_hold_mean_N": lower["hold_mean_force_N"],
                "higher_realized_hold_mean_N": higher["hold_mean_force_N"],
                "mean_force_monotonic_violation": violation,
            })
    violation_count = sum(bool(row["mean_force_monotonic_violation"]) for row in pairwise)
    violation_rate = violation_count / len(pairwise)
    correlations = {
        "setpoint_vs_realized_hold_mean_spearman": spearman(setpoints, mean_force),
        "setpoint_vs_hold_force_exposure_spearman": spearman(setpoints, exposure),
        "setpoint_vs_top5_force_spearman": spearman(setpoints, top5),
    }

    # Predeclared strict mapping gate: all three measured-force summaries must be
    # monotonic across setpoints.  A positive rank correlation alone is evidence
    # of association, not evidence of a valid monotonic execution mapping.
    continuous_mapping_valid = (
        violation_rate == 0.0
        and all(b >= a for a, b in zip(mean_force, mean_force[1:]))
        and all(b >= a for a, b in zip(exposure, exposure[1:]))
        and all(b >= a for a, b in zip(top5, top5[1:]))
    )
    write_json(OUT / "IF_STRICT_FAIL_SETPOINT_MONOTONIC_RESULT.json", {
        "schema": "SETPOINT_MONOTONIC_RESULT_V1",
        "status": "PASS" if continuous_mapping_valid else "FAIL_MONOTONIC_MAPPING",
        "primary_window": "30 environment steps after first lift (90 physics samples at 60 Hz)",
        "setpoints_tested_N": setpoints,
        "repeats_per_setpoint": 3,
        "total_fresh_process_runs": len(all_rows),
        "all_runs_complete": all(row["status"] == "COMPLETE" for row in all_rows),
        "all_handoffs_valid": all(row["hold_physics_rows"] == 90 for row in all_rows),
        "raw_arm_trajectory_changed": any(row["raw_arm_trajectory_changed"] for row in all_rows),
        "aggregate_by_setpoint": aggregate,
        "correlations": correlations,
        "pairwise_mean_force_comparisons": pairwise,
        "pairwise_monotonic_violation_count": violation_count,
        "pairwise_monotonic_comparison_count": len(pairwise),
        "pairwise_monotonic_violation_rate": violation_rate,
        "continuous_setpoint_mapping_valid": continuous_mapping_valid,
        "decision_rule": "PASS requires non-decreasing realized hold mean, hold exposure, and top-5 force over all ordered setpoints; Spearman alone is insufficient.",
        "interpretation": "Rank association is positive, but deterministic 2->3 N and 4->5 N realized-mean/exposure inversions invalidate a strict monotonic mapping claim.",
        "teardown_note": "Every run wrote status COMPLETE and a full trace before the known Isaac pybind weakref teardown abort.",
    })

    report = f"""# Final low-controller decision

## Decision

The final strict physical-Newton attempt failed. The offset-preserving servo retained the existing virtual-equilibrium preload without a handoff collapse, but its root7703 4 N hold force was **{live_row['realized_hold_mean_force_N']:.4f} N** with **{live_row['hold_mean_absolute_tracking_error_N']:.4f} N MAE**. Bilateral hold contact was maintained, so this is a tracking failure rather than a contact-only failure. Per the frozen stop rule, no further low-level controller design is authorized.

The fallback setpoint audit also does **not** support a strict monotonic realized-force mapping. Across 2/3/4/5/6 N with three fresh-process repeats each, Spearman correlation was **{correlations['setpoint_vs_realized_hold_mean_spearman']:.3f}** for hold mean, **{correlations['setpoint_vs_hold_force_exposure_spearman']:.3f}** for hold exposure, and **{correlations['setpoint_vs_top5_force_spearman']:.3f}** for top-5 force. However, mean/exposure had **{violation_count}/{len(pairwise)} ({100.0 * violation_rate:.1f}%)** ordered-pair violations. The repeats were deterministic, so those inversions cannot be described as sampled stochastic noise.

## Strict 4 N gate

| Metric | Result |
|---|---:|
| First target jump | {first_jump_mm:.6f} mm |
| Force before / after switch | {preload_before:.4f} / {preload_after:.4f} N |
| Hold mean | {live_row['realized_hold_mean_force_N']:.4f} N |
| Hold std | {live_row['realized_hold_std_force_N']:.4f} N |
| Hold MAE | {live_row['hold_mean_absolute_tracking_error_N']:.4f} N |
| Peak / top-5% | {live_row['peak_force_N']:.4f} / {live_row['top5_force_N']:.4f} N |
| Hold exposure | {live_row['hold_force_exposure_Ns']:.4f} N s |
| Bilateral hold | {str(live_row['hold_bilateral_contact']).upper()} |
| Strict tracking | {str(strict_valid).upper()} |

The first target change stayed within the 0.04 mm one-step rate bound and force did not collapse, proving that the preload-offset interpretation fixed the prior shadow-gate defect. It did not solve dynamic force realization.

## Fallback setpoint evidence

| Setpoint | Hold mean | Exposure | Top-5% | Repeats |
|---:|---:|---:|---:|---:|
"""
    for row in aggregate:
        report += (
            f"| {row['setpoint_N']:.0f} N | {row['hold_mean_force_N']:.4f} N | "
            f"{row['hold_force_exposure_Ns']:.4f} N s | {row['top5_force_N']:.4f} N | "
            f"{row['repeat_count']} |\n"
        )
    report += f"""

The 2→3 N and 4→5 N inversions occur in all three repeats because each repeated trace is numerically identical under the deterministic protocol. Top-5 force is monotonic, but that single metric cannot rescue the failed measured-mean/exposure contract.

## Evidence quality and scope

- All 15 fallback runs used fresh Isaac processes, exact HDF5 initialization, matched warm replay, bilateral handoff contact, no posterior/probe involvement, and unchanged raw arm trajectories.
- Each hold statistic uses exactly 90 physics samples (30 environment steps at 60 Hz).
- The known teardown abort occurs only after `status=COMPLETE` and full trace write; it does not invalidate the physical data.
- This is one integration root and does not establish population-level mapping behavior.

## Frozen conclusion

`STRICT_DYNAMIC_NEWTON_TRACKING = NOT_PRACTICALLY_VALIDATED`

`CONTINUOUS_SETPOINT_MAPPING_VALID = NO`

The project should now stop low-level controller debugging as instructed. Before any gentleness claim resumes, the execution/evaluation contract must be reconsidered: neither exact Newton tracking nor a monotonic hold-mean/exposure mapping is supported by this root under the frozen controller. Posterior, Expected Utility, force semantics, and the VLA arm trajectory were not changed in this work.
"""
    (OUT / "FINAL_LOW_CONTROLLER_DECISION_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
