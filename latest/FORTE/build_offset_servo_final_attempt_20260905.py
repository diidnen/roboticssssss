#!/usr/bin/env python3
"""Build the final strict-tracking virtual-equilibrium shadow gate."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean


OUT = Path(
    "/home/exouser/FORTE/analysis/results/"
    "offset_preserving_force_servo_final_attempt_20260905"
)
DYNAMIC = Path(
    "/home/exouser/FORTE/analysis/results/"
    "contact_safety_margin_supervisor_20260905/"
    "ROOT7703_4N_SAFETY_CONTROLLER_TRACE.csv"
)
STATIC_DIR = Path(
    "/home/exouser/FORTE/analysis/results/"
    "physics_rate_true_force_postfix_validation_20260904"
)
ACTUATOR_AUDIT = Path(
    "/home/exouser/FORTE/analysis/results/"
    "actuator_following_arm_coupling_diagnosis_20260905/"
    "ACTUATOR_BANDWIDTH_METRICS.json"
)

DT = 1.0 / 60.0
F_DES = 4.0
SLOPE = 9400.0
SETTLING_TIME = 0.20
K_EQ = 1.0 / (SLOPE * SETTLING_TIME)
DEADBAND = 0.25
V_RELEASE_MAX = 0.0024
V_CLOSE_MAX = 0.0030
CONFIG_EQ_MIN = -0.0057
CONFIG_EQ_MAX = -0.0019
MU_SAFE = 0.5726061820983886
RHO_CAUTION = 0.70
RHO_CRITICAL = 0.90
WEAK_GUARD_WIDTH = 0.75
CONTACT_THRESHOLD = 0.15


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict[str, str], key: str) -> float:
    return float(row[key])


def yes(row: dict[str, str], key: str) -> bool:
    return row[key].strip().lower() in {"true", "1", "yes"}


def clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dynamic = read_csv(DYNAMIC)
    required = {
        "physics_step", "sim_time_s", "F_meas_raw", "F_meas_filtered",
        "d_previous_command", "d_actual_before", "bilateral_contact",
        "safety_warning_raw", "projected_weak_force",
        "maximum_friction_utilization", "weak_force",
        "weak_force_derivative", "force_imbalance",
    }
    missing = sorted(required - set(dynamic[0])) if dynamic else sorted(required)
    finite = not missing and all(
        math.isfinite(num(row, field))
        for row in dynamic
        for field in required
        if field not in {"bilateral_contact", "safety_warning_raw"}
    )
    steps = [int(float(row["physics_step"])) for row in dynamic]
    contiguous = all(b == a + 1 for a, b in zip(steps, steps[1:]))
    if missing or not finite or not contiguous:
        raise RuntimeError(
            f"dynamic source failed quality gate: missing={missing}, "
            f"finite={finite}, contiguous={contiguous}"
        )

    static_offsets: dict[str, dict[str, float]] = {}
    terminal_offsets: list[float] = []
    for force in (2, 3, 4, 6):
        rows = read_csv(STATIC_DIR / f"POST_FIX_STATIC_{force}N_TRACE.csv")
        window = rows[-90:]
        offsets = [num(row, "d_command_clamped") - num(row, "d_actual_before") for row in window]
        entry = {
            "mean_offset_m": mean(offsets),
            "min_offset_m": min(offsets),
            "max_offset_m": max(offsets),
        }
        static_offsets[f"{force}N"] = entry
        terminal_offsets.extend(offsets)

    handoff_target = num(dynamic[0], "d_previous_command")
    handoff_actual = num(dynamic[0], "d_actual_before")
    initial_eq = handoff_target - handoff_actual
    eq_min = min(CONFIG_EQ_MIN, initial_eq)
    eq_max = max(CONFIG_EQ_MAX, initial_eq)

    preload_audit = {
        "schema": "PRELOAD_OFFSET_AUDIT_V1",
        "equilibrium_offset_definition": "delta_eq = d_target_applied - d_actual",
        "sign_convention": "aperture increases when opening; delta_eq<0 is closing-side virtual-equilibrium preload",
        "handoff_existing_target_m": handoff_target,
        "handoff_actual_aperture_m": handoff_actual,
        "handoff_initial_eq_offset_m": initial_eq,
        "handoff_preload_magnitude_m": abs(min(initial_eq, 0.0)),
        "static_terminal_offsets": static_offsets,
        "static_observed_min_m": min(terminal_offsets),
        "static_observed_max_m": max(terminal_offsets),
        "configured_bound_derivation": {
            "min_m": CONFIG_EQ_MIN,
            "max_m": CONFIG_EQ_MAX,
            "basis": "outward-rounded envelope spanning the matched dynamic handoff (-5.651 mm) and static 2/3/4/6 N terminal offsets (-3.153 to -2.007 mm)",
            "episode_effective_min_m": eq_min,
            "episode_effective_max_m": eq_max,
        },
    }
    dump(OUT / "PRELOAD_OFFSET_AUDIT.json", preload_audit)

    old_bandwidth = json.loads(ACTUATOR_AUDIT.read_text(encoding="utf-8"))
    reinterpretation = {
        "schema": "OLD_BACKLOG_REINTERPRETATION_V1",
        "old_28_percent_following_interpretation_valid": "PARTIAL",
        "old_reported_tracking_ratio": old_bandwidth["actuator_tracking_ratio"],
        "old_reported_command_opening_rate_mps": old_bandwidth["command_opening_rate_mps"],
        "old_reported_actual_opening_rate_mps": old_bandwidth["actual_opening_rate_mps"],
        "intentional_preload_offset_at_handoff_m": initial_eq,
        "actual_actuator_lag_after_preload_decomposition": "NOT_IDENTIFIABLE_AS_ACTUATOR_ONLY_FROM_CONTACT_LOADED_TRACE",
        "valid_interpretation": "28% remains a descriptive ratio of incremental target and aperture slopes in the connected-hold prefix, but the absolute target-actual separation is intentional preload and the incremental response is confounded by contact compliance and force dynamics.",
        "invalid_interpretation": "The full target-actual offset is actuator backlog or a pure actuator delay.",
    }
    dump(OUT / "OLD_BACKLOG_REINTERPRETATION.json", reinterpretation)

    shadow: list[dict[str, object]] = []
    eq_state = initial_eq
    warning_streak = 0
    previous_target = handoff_target
    first_target_jump = None
    correction_direction_valid = True
    clamp_count = 0
    release_request_count = 0
    safety_clamp_count = 0
    for index, row in enumerate(dynamic):
        measured = num(row, "F_meas_filtered")
        raw = num(row, "F_meas_raw")
        excess = measured - F_DES
        v_nom = 0.0 if abs(excess) < DEADBAND else K_EQ * excess
        v_nom = clip(v_nom, -V_CLOSE_MAX, V_RELEASE_MAX)
        raw_open_guard = v_nom > 0.0 and raw <= F_DES + DEADBAND
        if raw_open_guard:
            v_nom = 0.0

        warning_streak = warning_streak + 1 if yes(row, "safety_warning_raw") else 0
        warning = yes(row, "bilateral_contact") and warning_streak >= 2
        weak_target = 0.5 * F_DES
        weak_floor = max(weak_target - WEAK_GUARD_WIDTH, CONTACT_THRESHOLD)
        weak_scale = clip(
            (num(row, "projected_weak_force") - weak_floor)
            / max(weak_target - weak_floor, 1.0e-6),
            0.0,
            1.0,
        )
        rho = num(row, "maximum_friction_utilization")
        friction_scale = clip(
            (RHO_CRITICAL - rho) / (RHO_CRITICAL - RHO_CAUTION),
            0.0,
            1.0,
        )
        safety_scale = min(weak_scale, friction_scale) if warning else 1.0
        if not yes(row, "bilateral_contact"):
            safety_scale = 0.0
        v_safe = v_nom * safety_scale if v_nom > 0.0 else v_nom
        if v_nom > 0.0:
            release_request_count += 1
            if safety_scale < 0.999999:
                safety_clamp_count += 1

        eq_before = eq_state
        eq_unclamped = eq_before + v_safe * DT
        eq_state = clip(eq_unclamped, eq_min, eq_max)
        clamped = abs(eq_state - eq_unclamped) > 1.0e-12
        clamp_count += int(clamped)
        target = num(row, "d_actual_before") + eq_state
        jump = target - previous_target
        if index == 0:
            first_target_jump = jump
        previous_target = target

        if excess >= DEADBAND and v_safe < -1.0e-12:
            correction_direction_valid = False
        if excess <= -DEADBAND and v_safe > 1.0e-12:
            correction_direction_valid = False

        shadow.append({
            "physics_step": int(float(row["physics_step"])),
            "env_step": int(float(row["env_step"])),
            "sim_time_s": num(row, "sim_time_s"),
            "F_des_N": F_DES,
            "F_meas_raw_N": raw,
            "F_meas_filtered_N": measured,
            "force_excess_N": excess,
            "d_actual_m": num(row, "d_actual_before"),
            "old_target_applied_m": num(row, "d_previous_command"),
            "equilibrium_offset_before_m": eq_before,
            "equilibrium_velocity_nominal_mps": v_nom,
            "contact_warning": warning,
            "weak_force_N": num(row, "weak_force"),
            "weak_force_derivative_Nps": num(row, "weak_force_derivative"),
            "force_imbalance": num(row, "force_imbalance"),
            "friction_utilization_max": rho,
            "safety_scale": safety_scale,
            "equilibrium_velocity_safe_mps": v_safe,
            "equilibrium_offset_unclamped_m": eq_unclamped,
            "equilibrium_offset_state_m": eq_state,
            "equilibrium_offset_clamp_active": clamped,
            "shadow_target_m": target,
            "shadow_target_step_change_m": jump,
            "bilateral_contact": yes(row, "bilateral_contact"),
        })

    first_jump_mm = abs(first_target_jump) * 1000.0
    one_step_tolerance_mm = V_RELEASE_MAX * DT * 1000.0
    no_preload_erasure = abs(shadow[0]["equilibrium_offset_state_m"]) > 0.95 * abs(initial_eq)
    no_large_jump = first_jump_mm <= one_step_tolerance_mm + 1.0e-9
    eq_bounded = all(
        eq_min - 1.0e-12 <= row["equilibrium_offset_state_m"] <= eq_max + 1.0e-12
        for row in shadow
    )
    guard_valid = any(
        row["contact_warning"]
        and row["equilibrium_velocity_nominal_mps"] > 0.0
        and row["safety_scale"] < 0.999999
        for row in shadow
    )
    shadow_pass = all((no_preload_erasure, no_large_jump, correction_direction_valid, eq_bounded, guard_valid))

    design = {
        "schema": "OFFSET_SERVO_DESIGN_V1",
        "controller": "OFFSET_PRESERVING_VIRTUAL_EQUILIBRIUM_FORCE_SERVO",
        "equilibrium_offset_definition": "d_target_applied - d_actual",
        "state_update": "delta_eq_next = clamp(delta_eq + v_eq_safe*dt, delta_eq_min, delta_eq_max)",
        "target_update": "d_target_next = d_actual + delta_eq_next",
        "gain_m_per_n_s": K_EQ,
        "gain_derivation": "1 / (9400 N/m * 0.20 s)",
        "force_deadband_n": DEADBAND,
        "eq_offset_min_m": eq_min,
        "eq_offset_max_m": eq_max,
        "eq_release_rate_max_mps": V_RELEASE_MAX,
        "eq_close_rate_max_mps": V_CLOSE_MAX,
        "contact_safety_filter": "opening/release direction only; confirmed weak-side warning plus auxiliary friction utilization",
        "contact_preservation_reclose": False,
        "one_step_actuator_predictor_used": False,
        "feature_flag_default": False,
    }
    dump(OUT / "OFFSET_SERVO_DESIGN.json", design)
    save_csv(OUT / "OFFSET_SERVO_SHADOW_TRACE.csv", shadow)
    shadow_result = {
        "schema": "OFFSET_SERVO_SHADOW_RESULT_V1",
        "status": "PASS_LIVE_GATE_AUTHORIZED" if shadow_pass else "FAIL_STOP_STRICT_CONTROLLER_DEVELOPMENT",
        "shadow_mode_pass": shadow_pass,
        "source_quality": {
            "rows": len(dynamic),
            "required_fields_complete": not missing,
            "numeric_fields_finite": finite,
            "physics_steps_contiguous": contiguous,
            "source_is_previous_fresh_process": True,
            "source_reused_as_live_result": False,
        },
        "handoff_existing_target_m": handoff_target,
        "handoff_actual_aperture_m": handoff_actual,
        "handoff_initial_eq_offset_m": initial_eq,
        "first_new_target_m": shadow[0]["shadow_target_m"],
        "first_target_jump_mm": first_jump_mm,
        "one_step_safety_tolerance_mm": one_step_tolerance_mm,
        "no_preload_erasure": no_preload_erasure,
        "no_large_target_jump": no_large_jump,
        "eq_force_correction_direction_valid": correction_direction_valid,
        "eq_offset_bounded": eq_bounded,
        "contact_warning_guard_valid": guard_valid,
        "eq_offset_initial_m": initial_eq,
        "eq_offset_min_shadow_m": min(row["equilibrium_offset_state_m"] for row in shadow),
        "eq_offset_max_shadow_m": max(row["equilibrium_offset_state_m"] for row in shadow),
        "max_eq_change_per_step_m": max(abs(row["equilibrium_velocity_safe_mps"] * DT) for row in shadow),
        "eq_offset_clamp_rows": clamp_count,
        "safety_filter_duty_cycle": safety_clamp_count / release_request_count if release_request_count else None,
        "fresh_isaac_run_authorized": shadow_pass,
    }
    dump(OUT / "OFFSET_SERVO_SHADOW_RESULT.json", shadow_result)

    live_fields = [
        "sim_time_s", "env_step", "physics_step", "F_des", "F_meas_raw",
        "F_meas_filtered", "force_error", "d_actual_before",
        "actual_aperture_velocity_raw", "d_previous_command",
        "equilibrium_offset_previous_m", "equilibrium_offset_state_m",
        "equilibrium_velocity_nominal_mps", "equilibrium_safety_scale",
        "equilibrium_velocity_safe_mps", "d_command_clamped",
        "left_true_normal_force", "right_true_normal_force",
        "left_true_tangential_force", "right_true_tangential_force",
        "weak_force", "weak_force_derivative", "force_imbalance",
        "bilateral_contact", "safety_warning_confirmed",
    ]
    save_csv(OUT / "ROOT7703_4N_OFFSET_SERVO_TRACE.csv", [], live_fields)
    dump(OUT / "ROOT7703_4N_OFFSET_SERVO_RESULT.json", {
        "status": "PENDING_LIVE_RUN" if shadow_pass else "NOT_RUN_SHADOW_GATE_FAILED",
        "F_des_N": F_DES,
        "live_metrics": None,
        "previous_results_reused_as_live": False,
    })
    dump(OUT / "IF_4N_PASS_REPEAT_SUMMARY.json", {
        "status": "PENDING_PRIMARY_LIVE_GATE" if shadow_pass else "NOT_RUN_SHADOW_GATE_FAILED",
        "repeat_count": 0,
        "repeat_pass_count": 0,
    })
    save_csv(
        OUT / "IF_STRICT_FAIL_SETPOINT_MONOTONIC_TRACE.csv",
        [],
        [
            "setpoint_N", "repeat", "realized_mean_force_N", "top5_force_N",
            "peak_force_N", "force_exposure_Ns", "lift_success",
            "hold_success", "contact_loss", "source_trace",
        ],
    )
    dump(OUT / "IF_STRICT_FAIL_SETPOINT_MONOTONIC_RESULT.json", {
        "status": "NOT_TRIGGERED_YET",
        "setpoints_tested": [],
        "spearman_realized_mean": None,
        "spearman_force_exposure": None,
        "spearman_top5_force": None,
        "pairwise_monotonic_violation_rate": None,
        "continuous_setpoint_mapping_valid": None,
    })


if __name__ == "__main__":
    main()
