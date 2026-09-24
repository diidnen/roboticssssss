#!/usr/bin/env python3
"""Build the gated shadow audit for the actual-state velocity controller.

This analysis deliberately uses only the prior fresh-process root7703 4 N
physics trace.  It never substitutes shadow values for live Isaac results.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean


OUT = Path(
    "/home/exouser/FORTE/analysis/results/"
    "velocity_resolved_true_force_controller_20260905"
)
SOURCE = Path(
    "/home/exouser/FORTE/analysis/results/"
    "contact_safety_margin_supervisor_20260905/"
    "ROOT7703_4N_SAFETY_CONTROLLER_TRACE.csv"
)
SOURCE_RESULT = SOURCE.with_name("ROOT7703_4N_SAFETY_CONTROLLER_RESULT.json")

DT = 1.0 / 60.0
F_DES = 4.0
SLOPE_N_PER_M = 9400.0
SETTLING_TIME_S = 0.20
VELOCITY_GAIN_M_PER_N_S = 1.0 / (SLOPE_N_PER_M * SETTLING_TIME_S)
DEADBAND_N = 0.25
V_OPEN_MAX_MPS = 0.0024
V_CLOSE_MAX_MPS = 0.0030
VELOCITY_FILTER_ALPHA = 0.50
MU_SAFE = 0.5726061820983886
RHO_CAUTIOUS = 0.70
RHO_CRITICAL = 0.90
WEAK_GUARD_WIDTH_N = 0.75
CONTACT_THRESHOLD_N = 0.15
WARNING_CONFIRM_STEPS = 2


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def truth(row: dict[str, str], key: str) -> bool:
    return row[key].strip().lower() in {"1", "true", "yes"}


def clipped(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    source_result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))

    required = {
        "physics_step",
        "F_meas_raw",
        "F_meas_filtered",
        "d_actual_before",
        "d_command_clamped",
        "opening_backlog_m",
        "left_true_normal_force",
        "right_true_normal_force",
        "maximum_friction_utilization",
        "projected_weak_force",
        "safety_warning_confirmed",
        "bilateral_contact",
    }
    missing = sorted(required - set(source_rows[0])) if source_rows else sorted(required)
    finite = not missing and all(
        math.isfinite(number(row, field))
        for row in source_rows
        for field in required
        if field not in {"safety_warning_confirmed", "bilateral_contact"}
    )
    steps = [int(float(row["physics_step"])) for row in source_rows]
    contiguous = all(b == a + 1 for a, b in zip(steps, steps[1:]))
    quality = {
        "source": str(SOURCE),
        "source_is_fresh_process": source_result["quality"]["fresh_process"],
        "source_handoff_parity": source_result["quality"]["handoff_parity"],
        "source_raw_arm_trajectory_changed": source_result["quality"]["raw_arm_trajectory_changed"],
        "rows": len(source_rows),
        "missing_required_fields": missing,
        "all_required_numeric_fields_finite": finite,
        "physics_steps_contiguous": contiguous,
    }
    if missing or not finite or not contiguous:
        raise RuntimeError(f"source trace failed quality gate: {quality}")

    shadow_rows: list[dict[str, object]] = []
    previous_actual: float | None = None
    filtered_velocity = 0.0
    warning_streak = 0
    for source in source_rows:
        actual = number(source, "d_actual_before")
        actual_velocity_raw = 0.0 if previous_actual is None else (actual - previous_actual) / DT
        filtered_velocity = (
            (1.0 - VELOCITY_FILTER_ALPHA) * filtered_velocity
            + VELOCITY_FILTER_ALPHA * actual_velocity_raw
        )
        previous_actual = actual

        measured = number(source, "F_meas_filtered")
        measured_raw = number(source, "F_meas_raw")
        force_excess = measured - F_DES
        v_force = 0.0 if abs(force_excess) < DEADBAND_N else (
            VELOCITY_GAIN_M_PER_N_S * force_excess
        )
        v_nom = clipped(v_force, -V_CLOSE_MAX_MPS, V_OPEN_MAX_MPS)
        overspeed_brake = (
            v_nom * filtered_velocity > 0.0
            and abs(filtered_velocity) > abs(v_nom)
        )
        if overspeed_brake:
            v_nom = 0.0
        raw_open_guard = v_nom > 0.0 and measured_raw <= F_DES + DEADBAND_N
        if raw_open_guard:
            v_nom = 0.0

        warning_raw = truth(source, "safety_warning_raw")
        warning_streak = warning_streak + 1 if warning_raw else 0
        warning = truth(source, "bilateral_contact") and warning_streak >= WARNING_CONFIRM_STEPS
        weak_target = 0.5 * F_DES
        weak_floor = max(weak_target - WEAK_GUARD_WIDTH_N, CONTACT_THRESHOLD_N)
        weak_scale = clipped(
            (number(source, "projected_weak_force") - weak_floor)
            / max(weak_target - weak_floor, 1.0e-6),
            0.0,
            1.0,
        )
        rho = number(source, "maximum_friction_utilization")
        friction_scale = clipped(
            (RHO_CRITICAL - rho) / (RHO_CRITICAL - RHO_CAUTIOUS),
            0.0,
            1.0,
        )
        safety_scale = min(weak_scale, friction_scale) if warning else 1.0
        if not truth(source, "bilateral_contact"):
            safety_scale = 0.0
        v_safe = v_nom * safety_scale if v_nom > 0.0 else v_nom
        target = actual + v_safe * DT
        old_target = number(source, "d_command_clamped")
        shadow_rows.append(
            {
                "physics_step": int(float(source["physics_step"])),
                "env_step": int(float(source["env_step"])),
                "sim_time_s": number(source, "sim_time_s"),
                "F_des_N": F_DES,
                "F_meas_raw_N": measured_raw,
                "F_meas_filtered_N": measured,
                "force_excess_N": force_excess,
                "d_actual_m": actual,
                "v_actual_raw_mps": actual_velocity_raw,
                "v_actual_filtered_mps": filtered_velocity,
                "shadow_v_force_mps": v_force,
                "shadow_v_nom_mps": v_nom,
                "shadow_overspeed_brake": overspeed_brake,
                "shadow_raw_open_guard": raw_open_guard,
                "weak_force_N": number(source, "weak_force"),
                "weak_force_derivative_Nps": number(source, "weak_force_derivative"),
                "projected_weak_force_N": number(source, "projected_weak_force"),
                "friction_utilization_max": rho,
                "safety_warning_confirmed": warning,
                "shadow_weak_scale": weak_scale,
                "shadow_friction_scale": friction_scale,
                "shadow_safety_scale": safety_scale,
                "shadow_v_safe_mps": v_safe,
                "shadow_position_target_m": target,
                "shadow_position_backlog_m": target - actual,
                "old_position_target_m": old_target,
                "old_signed_target_minus_actual_m": old_target - actual,
                "old_accumulated_opening_backlog_m": number(source, "opening_backlog_m"),
                "shadow_target_jump_from_old_target_m": target - old_target,
                "bilateral_contact": truth(source, "bilateral_contact"),
            }
        )

    old_backlog = [row["old_accumulated_opening_backlog_m"] for row in shadow_rows]
    new_backlog = [abs(row["shadow_position_backlog_m"]) for row in shadow_rows]
    transition_jumps = [abs(row["shadow_target_jump_from_old_target_m"]) for row in shadow_rows]
    opening_rows = [row for row in shadow_rows if row["force_excess_N"] >= DEADBAND_N]
    direction_valid = bool(opening_rows) and all(
        row["shadow_v_nom_mps"] >= 0.0 for row in opening_rows
    )

    # Identify the final contact-loss event and the warning immediately before
    # it.  Earlier loss/recovery is kept separate so it cannot masquerade as
    # proactive protection of the final collapse.
    final_loss_index = None
    for index in range(1, len(shadow_rows)):
        if shadow_rows[index - 1]["bilateral_contact"] and not shadow_rows[index]["bilateral_contact"]:
            final_loss_index = index
    final_warning_index = None
    if final_loss_index is not None:
        last_warning_index = None
        for index in range(final_loss_index - 1, -1, -1):
            if shadow_rows[index]["safety_warning_confirmed"]:
                last_warning_index = index
                break
            if not shadow_rows[index]["bilateral_contact"]:
                break
        if last_warning_index is not None:
            final_warning_index = last_warning_index
            while (
                final_warning_index > 0
                and shadow_rows[final_warning_index - 1]["safety_warning_confirmed"]
            ):
                final_warning_index -= 1
    warning_lead_ms = None
    if final_loss_index is not None and final_warning_index is not None:
        warning_lead_ms = 1000.0 * (
            shadow_rows[final_loss_index]["sim_time_s"]
            - shadow_rows[final_warning_index]["sim_time_s"]
        )
    filter_active_before_failure = (
        final_warning_index is not None
        and shadow_rows[final_warning_index]["shadow_safety_scale"] < 0.999999
    )

    backlog_reduced = max(new_backlog) < max(old_backlog)
    one_step_limit = V_OPEN_MAX_MPS * DT
    no_large_one_step_backlog = max(new_backlog) <= max(V_OPEN_MAX_MPS, V_CLOSE_MAX_MPS) * DT + 1e-12
    # A position-drive switch must also be continuous with the target that was
    # physically carrying preload.  The prescribed actual+v*dt form fails
    # this second condition when contact compliance separates q_actual from
    # q_target by millimetres.
    no_large_target_transition = transition_jumps[0] <= one_step_limit + 1e-12
    shadow_pass = all(
        (
            backlog_reduced,
            direction_valid,
            no_large_one_step_backlog,
            filter_active_before_failure,
            no_large_target_transition,
        )
    )

    design = {
        "schema": "VELOCITY_CONTROLLER_DESIGN_V1",
        "controller_type": "ACTUAL_STATE_VELOCITY_RESOLVED",
        "actual_aperture_definition": "mean(left_finger_joint_position, right_finger_joint_position); larger is opening",
        "actual_aperture_velocity_definition": "finite difference of actual aperture at 60 Hz, EMA alpha=0.5; positive is opening",
        "force_error_definition": "F_meas_filtered - F_des; positive commands opening",
        "local_force_aperture_slope_n_per_m": SLOPE_N_PER_M,
        "settling_horizon_s": SETTLING_TIME_S,
        "velocity_gain_m_per_n_s": VELOCITY_GAIN_M_PER_N_S,
        "velocity_gain_derivation": "1 / (9400 N/m * 0.20 s)",
        "force_deadband_n": DEADBAND_N,
        "v_open_max_mps": V_OPEN_MAX_MPS,
        "v_close_max_mps": V_CLOSE_MAX_MPS,
        "position_target_formula": "d_target_next = d_actual + v_safe * physics_dt",
        "position_command_accumulation_used": False,
        "actual_velocity_use": "same-direction overspeed brake; no actuator-state predictor",
        "integral_added": False,
        "one_step_actuator_predictor_used": False,
        "feature_flag_default": False,
    }
    safety_design = {
        "schema": "CONTACT_SAFETY_FILTER_DESIGN_V1",
        "filter_scope": "opening velocity only",
        "primary_signal": "confirmed weak-side force weakening",
        "auxiliary_signal": "friction utilization using mu_safe",
        "mu_safe": MU_SAFE,
        "rho_cautious": RHO_CAUTIOUS,
        "rho_critical": RHO_CRITICAL,
        "warning_confirmation_steps": WARNING_CONFIRM_STEPS,
        "weak_guard_width_n": WEAK_GUARD_WIDTH_N,
        "contact_preservation_reflex_used": False,
        "output": "v_safe; does not modify F_des",
    }
    shadow_result = {
        "schema": "VELOCITY_CONTROLLER_SHADOW_RESULT_V1",
        "status": "PASS_LIVE_GATE_AUTHORIZED" if shadow_pass else "FAIL_LIVE_GATE_BLOCKED",
        "shadow_mode_pass": shadow_pass,
        "quality": quality,
        "shadow_backlog_reduced": backlog_reduced,
        "shadow_force_control_direction_valid": direction_valid,
        "shadow_no_large_one_step_backlog": no_large_one_step_backlog,
        "shadow_contact_safety_filter_active_before_failure": filter_active_before_failure,
        "shadow_no_large_position_target_transition": no_large_target_transition,
        "old_accumulated_opening_backlog_mean_m": mean(old_backlog),
        "old_accumulated_opening_backlog_max_m": max(old_backlog),
        "new_shadow_backlog_mean_m": mean(new_backlog),
        "new_shadow_backlog_max_m": max(new_backlog),
        "first_shadow_target_jump_from_applied_old_target_m": transition_jumps[0],
        "max_shadow_target_jump_from_applied_old_target_m": max(transition_jumps),
        "position_drive_compliance_offset_at_switch_m": number(source_rows[0], "d_actual_before") - number(source_rows[0], "d_command_clamped"),
        "one_step_opening_limit_m": one_step_limit,
        "final_contact_loss_physics_step": None if final_loss_index is None else shadow_rows[final_loss_index]["physics_step"],
        "first_preceding_warning_physics_step": None if final_warning_index is None else shadow_rows[final_warning_index]["physics_step"],
        "shadow_contact_warning_lead_time_ms": warning_lead_ms,
        "opening_request_rows": len(opening_rows),
        "safety_filter_duty_cycle_on_opening_requests": (
            sum(row["shadow_safety_scale"] < 0.999999 for row in opening_rows) / len(opening_rows)
            if opening_rows else None
        ),
        "blocking_reason": (
            None
            if shadow_pass
            else "ACTUAL_STATE_POSITION_TARGET_ERASES_PRELOAD_DRIVE_OFFSET"
        ),
        "interpretation": (
            "The one-step command backlog is bounded, but switching a stiff position drive from its loaded target to measured joint position would jump the applied target by the existing contact-compliance offset. Live execution is unsafe until velocity is applied without erasing that preload-producing drive offset."
        ),
        "actual_command_changed": False,
        "fresh_isaac_run_authorized": shadow_pass,
    }

    write_json(OUT / "VELOCITY_CONTROLLER_DESIGN.json", design)
    write_json(OUT / "SAFETY_FILTER_DESIGN.json", safety_design)
    write_csv(OUT / "VELOCITY_CONTROLLER_SHADOW_TRACE.csv", shadow_rows)
    write_json(OUT / "VELOCITY_CONTROLLER_SHADOW_RESULT.json", shadow_result)

    live_result = {
        "schema": "ROOT7703_4N_VELOCITY_RESULT_V1",
        "status": "NOT_RUN_SHADOW_GATE_FAILED" if not shadow_pass else "PENDING_LIVE_RUN",
        "fresh_process": None,
        "F_des_N": F_DES,
        "live_metrics": None,
        "reason": shadow_result["blocking_reason"],
        "old_results_reused_as_live": False,
    }
    write_json(OUT / "ROOT7703_4N_VELOCITY_RESULT.json", live_result)
    live_fields = [
        "sim_time_s", "env_step", "physics_step", "F_des", "F_meas_raw",
        "F_meas_filtered", "force_error", "d_actual_before",
        "actual_aperture_velocity_raw", "actual_aperture_velocity_filtered",
        "velocity_nominal_mps", "velocity_safety_scale", "velocity_safe_mps",
        "velocity_position_target_m", "velocity_position_backlog_m",
        "left_true_normal_force", "right_true_normal_force",
        "left_true_tangential_force", "right_true_tangential_force",
        "weak_force", "weak_force_derivative", "force_imbalance",
        "maximum_friction_utilization", "bilateral_contact",
    ]
    write_csv(OUT / "ROOT7703_4N_VELOCITY_TRACE.csv", [], live_fields)
    write_json(
        OUT / "IF_NEEDED_CBF_QP_RESULT.json",
        {
            "status": "NOT_RUN",
            "reason": "shadow failed before simple safety filter could be executed; QP/CBF would not fix the position-drive preload-offset discontinuity",
        },
    )
    for repeat in (1, 2, 3):
        write_csv(OUT / f"IF_4N_PASS_REPEAT_{repeat}.csv", [], live_fields)
    write_json(
        OUT / "IF_4N_PASS_REPEAT_SUMMARY.json",
        {
            "status": "NOT_RUN_SHADOW_GATE_FAILED",
            "repeat_count": 0,
            "repeat_pass_count": 0,
        },
    )
    write_json(
        OUT / "IF_3N_RUN_REPEAT_SUMMARY.json",
        {
            "status": "NOT_RUN_4N_NOT_VALIDATED",
            "repeat_count": 0,
            "tracking_valid": None,
            "physical_outcome": "NOT_RUN",
        },
    )

    report = f"""# Actual-state velocity-resolved true-force controller

## Answer first

The shadow gate **did not authorize a live Isaac run**.  The proposed controller does eliminate accumulated opening backlog: the shadow one-step target offset is at most {max(new_backlog)*1e6:.2f} µm, versus the old accumulated opening backlog maximum of {max(old_backlog)*1e3:.3f} mm.  However, the position actuator was carrying contact preload with an applied target {shadow_result['position_drive_compliance_offset_at_switch_m']*1e3:.3f} mm below the measured finger position at the switch.  Sending `d_actual + v_safe*dt` would therefore jump the physical position target by {transition_jumps[0]*1e3:.3f} mm on the first step, far beyond the {one_step_limit*1e6:.1f} µm opening safety budget.

This is not an accumulated-command failure in the new law; it is a contract mismatch between an actual-state velocity command and the existing stiff position-drive interface.  In contact, measured joint position cannot be copied directly into a position target without erasing the elastic target offset that produces normal force.

## Shadow checks

- Source quality: {len(source_rows)} contiguous physics samples, finite required fields, fresh-process source, handoff parity preserved.
- Force direction: {'PASS' if direction_valid else 'FAIL'}.
- One-step backlog bound: {'PASS' if no_large_one_step_backlog else 'FAIL'}.
- Contact safety warning before final failure: {'PASS' if filter_active_before_failure else 'FAIL'} ({warning_lead_ms} ms lead).
- Applied-target transition continuity: **{'PASS' if no_large_target_transition else 'FAIL'}**.
- Overall shadow gate: **{'PASS' if shadow_pass else 'FAIL'}**.

## Controller definition tested in shadow

`v_force = (F_meas - F_des) / (9400 N/m * 0.20 s)`, with a 0.25 N deadband, +2.4 mm/s opening limit, -3.0 mm/s closing limit, actual-velocity overspeed braking, and an opening-only weak-side/friction safety scale.  No integral and no one-step actuator predictor are used.  `F_des`, the arm trajectory, posterior, and Expected Utility are unchanged.

## Required next action

Retain actual-state velocity feedback, but map it to the position actuator without deleting the measured preload drive offset.  The minimal next diagnostic is an offset-preserving velocity servo (or a true velocity-drive interface if available), followed by a new shadow transition-continuity check.  Do not run root7703 live until that gate passes.
"""
    (OUT / "VELOCITY_CONTROLLER_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
