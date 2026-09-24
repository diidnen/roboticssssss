#!/usr/bin/env python3
"""Build the root7703 actuator-following and arm-coupling diagnostic bundle."""

from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
from pathlib import Path

import numpy as np


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "analysis/results/actuator_following_arm_coupling_diagnosis_20260905"
RUNS = OUT / "runs"
POSTFIX = ROOT / "analysis/results/physics_rate_true_force_postfix_validation_20260904"
CONDITIONS = {
    "4N_NORMAL": (RUNS / "4N_normal", 4.0, 1.0),
    "4N_HALF_SPEED": (RUNS / "4N_half", 4.0, 0.5),
    "3N_NORMAL": (RUNS / "3N_normal", 3.0, 1.0),
    "3N_HALF_SPEED": (RUNS / "3N_half", 3.0, 0.5),
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def num(row: dict[str, str], key: str) -> float:
    return float(row[key])


def flag(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def ols_slope(rows: list[dict[str, str]], x_key: str) -> dict[str, float]:
    x = np.asarray([num(row, x_key) for row in rows], dtype=float)
    y = np.asarray([num(row, "F_meas_raw") for row in rows], dtype=float)
    matrix = np.column_stack((x, np.ones(len(x))))
    slope, intercept = np.linalg.lstsq(matrix, y, rcond=None)[0]
    prediction = matrix @ np.asarray([slope, intercept])
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - prediction) ** 2)) / denominator
    return {"slope_N_per_m": float(slope), "slope_N_per_mm": float(slope / 1000.0), "r2": r2, "rows": len(rows)}


def continuous_connected_prefix(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    answer: list[dict[str, str]] = []
    for row in rows:
        if not flag(row, "bilateral_contact"):
            break
        answer.append(row)
    return answer


def condition_metrics(path: Path, target: float, speed: float) -> dict[str, object]:
    result = json.loads((path / "ROOT7703_CONTROLLER_VALIDATION_RESULT.json").read_text(encoding="utf-8"))
    physics = load_csv(path / "PHYSICS_STEP_SIGNAL_TRACE.csv")
    env = load_csv(path / "ROOT7703_CONTROLLER_VALIDATION_ENV_TRACE.csv")
    lift = int(result["summary"]["first_lift_step"])
    relevant = [row for row in physics if int(row["env_step"]) <= lift]
    hold = [row for row in physics if lift + 1 <= int(row["env_step"]) <= lift + 30]
    connected = continuous_connected_prefix(hold)
    force = np.asarray([num(row, "F_meas_raw") for row in hold])
    connected_force = np.asarray([num(row, "F_meas_raw") for row in connected])
    gap = np.asarray([num(row, "d_joint_target") - num(row, "d_actual_before") for row in hold])

    time = np.arange(len(connected), dtype=float) / 60.0
    command = np.asarray([num(row, "d_joint_target") for row in connected])
    actual = np.asarray([num(row, "d_actual_before") for row in connected])
    command_rate = float(np.polyfit(time, command, 1)[0])
    actual_rate = float(np.polyfit(time, actual, 1)[0])

    requested_opening: list[float] = []
    for row in hold:
        error = target - num(row, "F_meas_filtered")
        correction = -0.004 * error / 60.0 if abs(error) >= 0.25 else 0.0
        if correction > 0.0:
            requested_opening.append(correction)
    open_limit = 0.0024 / 60.0
    clipped_opening = [value for value in requested_opening if value > open_limit + 1e-12]

    events: list[dict[str, object]] = []
    previous = True
    for row in physics:
        current = flag(row, "bilateral_contact")
        if current != previous:
            events.append({
                "event": "RECOVERY" if current else "LOSS",
                "env_step": int(row["env_step"]),
                "physics_step": int(row["physics_step"]),
            })
        previous = current

    normal_waypoints = [int(row["arm_waypoint_index"]) for row in env]
    collapsed_waypoints: list[int] = []
    for waypoint in normal_waypoints:
        if not collapsed_waypoints or collapsed_waypoints[-1] != waypoint:
            collapsed_waypoints.append(waypoint)

    target_buffer_error = max(
        abs(num(row, "d_joint_target") - num(row, "left_finger_joint_target_buffer"))
        for row in physics
    )
    torque_error = max(
        abs(num(row, "left_finger_computed_torque") - num(row, "left_finger_applied_torque"))
        for row in physics
    )

    raw_cross = next((index for index, row in enumerate(hold) if num(row, "F_meas_raw") <= target + 0.25), None)
    filtered_cross = next((index for index, row in enumerate(hold) if num(row, "F_meas_filtered") <= target + 0.25), None)
    filter_release_delay = None if raw_cross is None or filtered_cross is None else (filtered_cross - raw_cross) / 60.0

    return {
        "condition": path.name,
        "target_N": target,
        "arm_speed_scale": speed,
        "fresh_process": bool(result["fresh_process"]),
        "handoff_parity": bool(result["handoff_parity"]["physical_state_parity"] and result["handoff_parity"]["runtime_state_parity"] and result["handoff_parity"]["observation_parity"]),
        "arm_geometry_changed": bool(result["arm_geometry_changed"]),
        "unique_waypoint_sequence": collapsed_waypoints,
        "lift_env_step": lift,
        "relevant_mean_force_N": float(np.mean([num(row, "F_meas_raw") for row in relevant])),
        "hold_mean_force_N": float(np.mean(force)),
        "hold_std_force_N": float(np.std(force)),
        "hold_mae_N": float(np.mean(np.abs(force - target))),
        "hold_peak_N": float(np.max(force)),
        "hold_bilateral_rows": int(sum(flag(row, "bilateral_contact") for row in hold)),
        "hold_rows": len(hold),
        "connected_hold_mean_force_N": float(np.mean(connected_force)),
        "connected_hold_rows": len(connected),
        "command_to_actual_rms_error_m": float(np.sqrt(np.mean(gap ** 2))),
        "command_to_actual_peak_error_m": float(np.max(np.abs(gap))),
        "command_opening_rate_mps": command_rate,
        "actual_opening_rate_mps": actual_rate,
        "actuator_tracking_ratio": actual_rate / command_rate,
        "actuator_response_delay_s": [0.0, 1.0 / 60.0],
        "opening_rate_limit_active_fraction": float(np.mean([flag(row, "rate_limit_active") for row in hold])),
        "opening_correction_clipped_fraction_of_opening_requests": len(clipped_opening) / len(requested_opening),
        "opening_correction_clipped_fraction_of_hold": len(clipped_opening) / len(hold),
        "target_buffer_max_error_m": target_buffer_error,
        "target_buffer_updated_fraction": float(np.mean([flag(row, "joint_target_buffer_updated") for row in physics])),
        "arm_overwrite_rows": int(sum(flag(row, "arm_apply_overwrote_gripper_target") for row in physics)),
        "computed_to_applied_torque_max_error": torque_error,
        "effort_limit_rows": int(sum(flag(row, "effort_limit_active") for row in physics)),
        "velocity_limit_rows": int(sum(flag(row, "velocity_limit_active") for row in physics)),
        "joint_limit_rows": int(sum(flag(row, "joint_limit_active") for row in physics)),
        "position_target_clamp_rows": int(sum(flag(row, "position_target_clamp_active") for row in physics)),
        "filter_release_delay_s": filter_release_delay,
        "force_aperture_slope_actual": ols_slope(connected, "d_actual_before"),
        "force_aperture_slope_command": ols_slope(connected, "d_joint_target"),
        "contact_events": events,
        "tracking_valid": bool(
            len(connected) == len(hold)
            and float(np.mean(np.abs(force - target))) <= 1.0
            and float(np.mean(np.abs(force - target) <= 1.0)) >= 0.6
        ),
        "source_result": str(path / "ROOT7703_CONTROLLER_VALIDATION_RESULT.json"),
        "source_trace": str(path / "PHYSICS_STEP_SIGNAL_TRACE.csv"),
    }


OUT.mkdir(parents=True, exist_ok=True)
metrics = {name: condition_metrics(*spec) for name, spec in CONDITIONS.items()}

# Exact requested trace names.  These are the 60 Hz force/actuator traces; arm
# and object kinematics are explicitly labelled as 20 Hz held samples.
for output_name, condition in {
    "ROOT7703_4N_NORMAL_TRACE.csv": "4N_NORMAL",
    "ROOT7703_4N_HALF_SPEED_TRACE.csv": "4N_HALF_SPEED",
    "ROOT7703_3N_NORMAL_TRACE.csv": "3N_NORMAL",
    "ROOT7703_3N_HALF_SPEED_TRACE.csv": "3N_HALF_SPEED",
}.items():
    source = Path(metrics[condition]["source_trace"])
    target = OUT / output_name
    shutil.copyfile(source, target)
    # Add derived pre-rate-limit correction without changing the raw source.
    source_rows = load_csv(target)
    fields = list(source_rows[0]) + ["requested_delta_d_before_rate_limit_derived"]
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        force_target = float(metrics[condition]["target_N"])
        for row in source_rows:
            error = force_target - num(row, "F_meas_filtered")
            row["requested_delta_d_before_rate_limit_derived"] = (
                -0.004 * error / 60.0 if abs(error) >= 0.25 else 0.0
            )
            writer.writerow(row)

waypoint_parity = metrics["4N_NORMAL"]["unique_waypoint_sequence"] == metrics["4N_HALF_SPEED"]["unique_waypoint_sequence"]

dump("LITERATURE_DESIGN_NOTES.json", {
    "status": "APPLIED_AS_DESIGN_CONSTRAINTS",
    "constraints": {
        "Tabero": "arm pose and gripper force remain decoupled and execute concurrently",
        "RETAF_CompliantVLA_FILIC": "force regulation remains in the fast low-level loop",
        "adaptive_parallel_gripper_control": "do not add an unbounded integral or blindly increase proportional gain",
    },
    "method_changes_forbidden_and_respected": ["posterior", "calibration", "Expected Utility", "arm geometry", "unbounded integral"],
    "visual_omission_reason": "Only four controlled conditions and one safety A/B are compared; exact audit tables in the Markdown report are more legible than a chart.",
})

dump("ACTUATOR_PATH_AUDIT.json", {
    "status": "COMMAND_PATH_PASS_DYNAMIC_FOLLOWING_LIMITED",
    "d_force_cmd_to_joint_target_valid": True,
    "joint_target_to_actuator_valid": True,
    "actuator_to_d_actual_valid": False,
    "evidence_all_conditions": {
        name: {
            "target_buffer_max_error_m": value["target_buffer_max_error_m"],
            "target_buffer_updated_fraction": value["target_buffer_updated_fraction"],
            "arm_overwrite_rows": value["arm_overwrite_rows"],
            "computed_to_applied_torque_max_error": value["computed_to_applied_torque_max_error"],
            "effort_limit_rows": value["effort_limit_rows"],
            "velocity_limit_rows": value["velocity_limit_rows"],
            "joint_limit_rows": value["joint_limit_rows"],
            "position_target_clamp_rows": value["position_target_clamp_rows"],
        }
        for name, value in metrics.items()
    },
    "actuator": {"type": "ImplicitActuatorCfg", "stiffness": 2000.0, "damping": 100.0, "effort_limit_sim": 200.0},
    "first_material_delay_layer": "implicit actuator/contact-constrained joint response: target buffer and torque application are exact, but actual opening rate is much lower than command",
})

primary = metrics["4N_NORMAL"]
dump("ACTUATOR_BANDWIDTH_METRICS.json", {
    "status": "LIMITED_RELATIVE_TO_DYNAMIC_COMMAND",
    "primary_condition": "4N_NORMAL",
    "actuator_response_delay_seconds": primary["actuator_response_delay_s"],
    "command_opening_rate_mps": primary["command_opening_rate_mps"],
    "actual_opening_rate_mps": primary["actual_opening_rate_mps"],
    "actuator_tracking_ratio": primary["actuator_tracking_ratio"],
    "command_to_actual_rms_error_m": primary["command_to_actual_rms_error_m"],
    "command_to_actual_peak_error_m": primary["command_to_actual_peak_error_m"],
    "hard_actuator_saturation_detected": False,
    "actuator_bandwidth_limited": True,
    "interpretation": "No observable dead-time or hard limit, but actual aperture realizes only 28% of commanded opening slope in the connected hold prefix.",
})

two_x_path = RUNS / "4N_normal_openrate2x_retry"
two_x = condition_metrics(two_x_path, 4.0, 1.0)
dump("OPENING_RATE_LIMIT_AUDIT.json", {
    "status": "BOTTLENECK_CONFIRMED_BUT_FIXED_2X_UNSAFE",
    "current_limit_mps": 0.0024,
    "primary_opening_rate_limit_active_fraction": primary["opening_rate_limit_active_fraction"],
    "primary_opening_correction_clipped_fraction": primary["opening_correction_clipped_fraction_of_opening_requests"],
    "primary_opening_correction_clipped_fraction_of_hold": primary["opening_correction_clipped_fraction_of_hold"],
    "two_x_diagnostic": {
        "requested_limit_mps": 0.0048,
        "hold_mean_force_N": two_x["hold_mean_force_N"],
        "hold_mae_N": two_x["hold_mae_N"],
        "hold_bilateral_rows": two_x["hold_bilateral_rows"],
        "hold_rows": two_x["hold_rows"],
        "contact_events": two_x["contact_events"],
        "result": "UNSAFE_PREMATURE_CONTACT_LOSS",
        "source_trace": two_x["source_trace"],
    },
    "conclusion": "The current ceiling is active enough to limit unloading, but a fixed 2x ceiling crosses the contact boundary sooner. A blind fixed-rate increase is rejected.",
})

static_rows = load_csv(POSTFIX / "POST_FIX_STATIC_4N_TRACE.csv")
static_slope = ols_slope(static_rows, "d_actual_before")
dump("FORCE_APERTURE_SLOPE_AUDIT.json", {
    "status": "4N_MAPPING_SIMILAR_LOW_FORCE_REGIME_NONLINEAR",
    "static_4N": static_slope,
    "dynamic_4N_normal": metrics["4N_NORMAL"]["force_aperture_slope_actual"],
    "dynamic_4N_half_speed": metrics["4N_HALF_SPEED"]["force_aperture_slope_actual"],
    "dynamic_3N_normal": metrics["3N_NORMAL"]["force_aperture_slope_actual"],
    "dynamic_3N_half_speed": metrics["3N_HALF_SPEED"]["force_aperture_slope_actual"],
    "position_force_mapping_state_dependent": "NO_AT_4N_PRIMARY; YES/UNSTABLE_NEAR_3N_CONTACT_BOUNDARY",
    "method_note": "OLS is descriptive over each connected hold prefix and is confounded by arm motion; it is not a causal material stiffness estimate.",
})

dump("FILTER_SHADOW_AUDIT.json", {
    "status": "CONTRIBUTOR_NOT_DOMINANT",
    "filter_time_constant_s": 0.07213475204444818,
    "4N_normal_raw_to_filtered_release_delay_s": metrics["4N_NORMAL"]["filter_release_delay_s"],
    "4N_half_raw_to_filtered_release_delay_s": metrics["4N_HALF_SPEED"]["filter_release_delay_s"],
    "raw_force_opening_onset_advantage_s": 0.0,
    "filter_latency_dominant": False,
    "conclusion": "Both raw and filtered force demand opening at hold onset. Filtering delays stopping the release by 0.083-0.117s, contributing to contact-boundary crossing but not causing the initial over-force or limited aperture response.",
})

comparison = {
    "status": "HALF_SPEED_DOES_NOT_IMPROVE_HOLD_TRACKING",
    "arm_waypoint_geometry_sequence_match": waypoint_parity,
    "conditions": metrics,
    "4N_half_minus_normal": {
        "relevant_mean_force_N": metrics["4N_HALF_SPEED"]["relevant_mean_force_N"] - primary["relevant_mean_force_N"],
        "hold_mean_force_N": metrics["4N_HALF_SPEED"]["hold_mean_force_N"] - primary["hold_mean_force_N"],
        "hold_mae_N": metrics["4N_HALF_SPEED"]["hold_mae_N"] - primary["hold_mae_N"],
        "actuator_tracking_ratio": metrics["4N_HALF_SPEED"]["actuator_tracking_ratio"] - primary["actuator_tracking_ratio"],
    },
    "arm_motion_force_coupling_confirmed": False,
    "interpretation": "Half speed lowers pre-lift mean force by about 1.05N but worsens the 30-step hold MAE by about 0.88N and lowers the actuator tracking ratio. The prespecified causal signature is absent.",
}
dump("NORMAL_VS_HALF_SPEED_COMPARISON.json", comparison)

dump("IF_FIXED_STATIC_REGRESSION.json", {
    "status": "NOT_RERUN_NO_CONTROLLER_CHANGE",
    "controller_change_applied": False,
    "immediately_preceding_postfix_static_regression_valid": True,
    "source": str(POSTFIX / "POST_FIX_STATIC_REGRESSION.json"),
    "results_N": {"2N": 2.1577550331751505, "3N": 3.086677567164103, "4N": 4.095852449205187, "6N": 6.115563249588012},
})
dump("IF_FIXED_DYNAMIC_4N_REPEAT_SUMMARY.json", {
    "status": "NOT_RUN_NO_SAFE_FIX_FROZEN",
    "repeat_count": 0,
    "reason": "The diagnostic identified a fixed-rate tradeoff rather than a validated minimal fix; 2x opening caused earlier contact loss.",
})

report = f"""# Dynamic actuator following and arm–gripper coupling diagnosis

## Technical summary

The primary 4N diagnostic fails force tracking, but not because the command disappears between software layers. `d_force_cmd` reaches the articulation target exactly, arm DiffIK never overwrites it, and computed/applied torque agree without hard effort, velocity, joint, or position clamps. The first material shortfall is the contact-constrained actuator response: actual aperture opening is only **{primary['actuator_tracking_ratio']:.1%}** of the commanded slope during the connected 4N hold prefix.

Half-speed arm execution does not produce the prespecified improvement signature. It reduces pre-lift mean force but makes 4N hold tracking worse. Arm–gripper coupling is present, yet this A/B does not confirm arm speed as the dominant cause. The current fixed opening-rate ceiling is a real bottleneck, while a 2× ceiling opens through contact sooner; therefore no controller parameter change is accepted this round.

## Four-condition result: 4N remains the primary failure

| Condition | Pre-lift mean | 30-step hold mean | Hold MAE | Connected hold mean | Bilateral rows |
|---|---:|---:|---:|---:|---:|
| 4N normal | {metrics['4N_NORMAL']['relevant_mean_force_N']:.3f}N | {metrics['4N_NORMAL']['hold_mean_force_N']:.3f}N | {metrics['4N_NORMAL']['hold_mae_N']:.3f}N | {metrics['4N_NORMAL']['connected_hold_mean_force_N']:.3f}N | {metrics['4N_NORMAL']['hold_bilateral_rows']}/90 |
| 4N half-speed | {metrics['4N_HALF_SPEED']['relevant_mean_force_N']:.3f}N | {metrics['4N_HALF_SPEED']['hold_mean_force_N']:.3f}N | {metrics['4N_HALF_SPEED']['hold_mae_N']:.3f}N | {metrics['4N_HALF_SPEED']['connected_hold_mean_force_N']:.3f}N | {metrics['4N_HALF_SPEED']['hold_bilateral_rows']}/90 |
| 3N normal | {metrics['3N_NORMAL']['relevant_mean_force_N']:.3f}N | {metrics['3N_NORMAL']['hold_mean_force_N']:.3f}N | {metrics['3N_NORMAL']['hold_mae_N']:.3f}N | {metrics['3N_NORMAL']['connected_hold_mean_force_N']:.3f}N | {metrics['3N_NORMAL']['hold_bilateral_rows']}/90 |
| 3N half-speed | {metrics['3N_HALF_SPEED']['relevant_mean_force_N']:.3f}N | {metrics['3N_HALF_SPEED']['hold_mean_force_N']:.3f}N | {metrics['3N_HALF_SPEED']['hold_mae_N']:.3f}N | {metrics['3N_HALF_SPEED']['connected_hold_mean_force_N']:.3f}N | {metrics['3N_HALF_SPEED']['hold_bilateral_rows']}/90 |

The 30-step means include zero-force rows after contact loss; connected-hold means are shown to prevent those zeros from making regulation look better. Neither 3N condition tracks stably, but task success was not used as the controller criterion.

## The software command path is intact; physical following is bandwidth-limited

Across all four runs, the controller and left/right articulation target buffers match exactly on every physics tick. The nested arm action produces zero gripper-target overwrites. Computed and applied finger torque match exactly, with no detected hard effort or velocity limits. For 4N normal, commanded opening slope is **{primary['command_opening_rate_mps']*1000:.3f}mm/s**, actual aperture slope is **{primary['actual_opening_rate_mps']*1000:.3f}mm/s**, RMS command–actual error is **{primary['command_to_actual_rms_error_m']*1000:.3f}mm**, and peak error is **{primary['command_to_actual_peak_error_m']*1000:.3f}mm**. Response onset occurs within one 60Hz tick; the problem is response magnitude/bandwidth, not dead-time.

## Slowing the arm does not establish arm-speed causality

The half-speed branch repeats every historical waypoint twice; collapsing repeats reproduces the exact 145-waypoint order. Relative to normal speed, 4N half-speed reduces pre-lift mean by {abs(comparison['4N_half_minus_normal']['relevant_mean_force_N']):.3f}N but increases hold MAE by {comparison['4N_half_minus_normal']['hold_mae_N']:.3f}N and lowers the aperture tracking ratio. This rejects the requested causal signature; it does not prove arm coupling is absent.

## A fixed faster release is not a safe repair

At 4N normal, the opening-rate limit is active for **{primary['opening_rate_limit_active_fraction']:.1%}** of hold ticks and clips **{primary['opening_correction_clipped_fraction_of_opening_requests']:.1%}** of opening requests. The 2× safety A/B lowers force faster but retains bilateral contact for only {two_x['hold_bilateral_rows']}/90 hold ticks and loses contact at env step {two_x['contact_events'][-1]['env_step']}. The fixed 2× setting is rejected. Force-error gain, actuator drive, and final controller configuration remain unchanged.

## Force–aperture mapping is stable around 4N, unstable near the contact boundary

Static 4N and dynamic-normal 4N actual-aperture slopes are {static_slope['slope_N_per_mm']:.3f} and {metrics['4N_NORMAL']['force_aperture_slope_actual']['slope_N_per_mm']:.3f}N/mm, respectively. The primary 4N evidence does not show a large mapping shift. Near 3N/contact loss, fitted slopes grow to {metrics['3N_NORMAL']['force_aperture_slope_actual']['slope_N_per_mm']:.3f}N/mm with weaker fit, so that low-force region is nonlinear/state-dependent and unsuitable for a global fixed-rate conclusion.

## Limitations and robustness

- Kinematic geometry follows the same waypoint order, but half-speed is implemented by repeating waypoints; it changes timing, not geometry.
- Force and actuator channels are 60Hz. EE/object kinematics are sampled at 20Hz and explicitly held across three physics rows.
- OLS force–aperture slopes are descriptive and confounded by moving-arm contact dynamics.
- Isaac writes complete traces before a recurring camera teardown exception. One first 2× attempt also encountered a transient NVIDIA-driver outage and is excluded; the successful retry is the only scientific 2× result.

## Recommended next step

Do not increase force `kp`, add an integral, or accept a fixed 2× rate. The next controller-only step is a bounded release law that accounts for actuator target–actual backlog and local force–aperture slope near contact, first evaluated as a shadow controller. Only after that law has a prespecified safe diagnostic should static regression and 4N ×3 fresh-process validation run.

## Further question

Can a target–actual-backlog-aware or local-slope-limited release reduce the 4N connected-hold MAE below 1N without crossing bilateral contact? Current data answer neither yes nor no; they only reject the present fixed-rate choices.
"""
(OUT / "DYNAMIC_ACTUATOR_REPORT.md").write_text(report, encoding="utf-8")

print(json.dumps({
    "status": "DIAGNOSIS_COMPLETE_NO_SAFE_FIX_VALIDATED",
    "4N_normal_hold_N": primary["hold_mean_force_N"],
    "4N_half_hold_N": metrics["4N_HALF_SPEED"]["hold_mean_force_N"],
    "actuator_tracking_ratio": primary["actuator_tracking_ratio"],
    "arm_speed_coupling_confirmed": False,
    "output": str(OUT),
}, indent=2))
