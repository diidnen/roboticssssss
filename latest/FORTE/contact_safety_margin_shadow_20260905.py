#!/usr/bin/env python3
"""Audit root7703 contact weakening and shadow a bounded safety supervisor.

The historical 4 N trace does not contain per-finger tangential vectors, so
this script never fabricates a friction cone.  It uses the measured weaker-side
normal force, a three-physics-step derivative, and imbalance as the auditable
fallback.  A later instrumented shadow run may supply tangential fields and
replace the fallback decision with conservative friction utilization.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


OUT = Path(
    "/home/exouser/FORTE/analysis/results/"
    "contact_safety_margin_supervisor_20260905"
)
SOURCE = Path(
    "/home/exouser/FORTE/analysis/results/"
    "backlog_slope_dynamic_release_fix_20260905/"
    "ROOT7703_DYNAMIC_4N_ADAPTIVE_TRACE.csv"
)
MU_SOURCE = Path(
    "/home/exouser/FORTE/analysis/results/"
    "probability_calibration_root7703_probe_smoke_20260904/"
    "ROOT7703_MU_POSTERIOR.json"
)
CONTROLLER_SOURCE = Path(
    "/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/"
    "manipulation/libero/mdp/force_position_action.py"
)
FORCE_HELPER_SOURCE = Path(
    "/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/"
    "manipulation/libero/mdp/terminations.py"
)

DT = 1.0 / 60.0
TARGET_N = 4.0
SLOPE_N_PER_M = 9400.0
PREDICTION_HORIZON_S = 3 * DT
RECLOSE_RATE_MPS = 0.0006
RECLOSE_LIMIT_M = 0.00005
RECLOSE_FORCE_BOUND_N = TARGET_N + 1.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def flag(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def fit_one_step_beta(rows: list[dict[str, str]]) -> dict[str, Any]:
    x: list[float] = []
    y: list[float] = []
    for current, following in zip(rows, rows[1:]):
        contiguous = int(number(following, "physics_step")) == int(
            number(current, "physics_step")
        ) + 1
        connected = flag(current, "bilateral_contact") and flag(
            following, "bilateral_contact"
        )
        overforce = number(current, "F_meas_raw") > TARGET_N + 0.25
        if contiguous and connected and overforce:
            x.append(
                number(current, "d_joint_target")
                - number(current, "d_actual_before")
            )
            y.append(
                number(following, "d_actual_before")
                - number(current, "d_actual_before")
            )
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    beta_unclipped = float(np.dot(xx, yy) / np.dot(xx, xx))
    beta = float(np.clip(beta_unclipped, 0.0, 1.0))
    prediction = beta * xx
    sse = float(np.sum((yy - prediction) ** 2))
    sst = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = None if sst <= 0 else float(1.0 - sse / sst)
    return {
        "schema": "ACTUATOR_ONE_STEP_MODEL_V1",
        "model": "d_actual_next=d_actual+beta_follow*(d_joint_target-d_actual)",
        "fit_population": "consecutive bilateral over-force physics rows from failed root7703 4N trace",
        "fit_rows": len(xx),
        "beta_follow": beta,
        "beta_unclipped": beta_unclipped,
        "r2_no_intercept": r2,
        "rmse_aperture_m_per_step": float(np.sqrt(np.mean((yy - prediction) ** 2))),
        "predictive_reliability": "LOW" if r2 is None or r2 < 0.1 else "ACCEPTABLE",
        "important_limitation": (
            "The factor is descriptive and weakly predictive; safety gating must "
            "not rely on it without an independent contact warning."
        ),
        "source_trace": str(SOURCE),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_csv(SOURCE)
    posterior = json.loads(MU_SOURCE.read_text(encoding="utf-8"))
    members = np.asarray(posterior["member_means"], dtype=float)
    mu_safe = float(np.quantile(members, 0.10))

    write_json(
        "CONTACT_FORCE_DECOMPOSITION_AUDIT.json",
        {
            "schema": "CONTACT_FORCE_DECOMPOSITION_AUDIT_V1",
            "sensor": "contact_grasp_black_book_1.data.force_matrix_w",
            "sensor_shape_contract": "(num_envs, 2 finger bodies, filtered target prims, 3 world-force components)",
            "object_filtered": True,
            "world_to_finger_transform": "object_filtered_gripper_force_local",
            "normal_definition": "abs(local z)",
            "tangential_definition": "sqrt(local x^2+local y^2)",
            "left_normal_force_available": True,
            "right_normal_force_available": True,
            "left_tangential_force_available_at_runtime": True,
            "right_tangential_force_available_at_runtime": True,
            "force_decomposition_valid": True,
            "failed_trace_contains_local_xy": False,
            "failed_trace_analysis_signal": "CONTACT_WEAKENING_PROXY",
            "risk": (
                "Historical tangential force cannot be reconstructed from scalar "
                "normal telemetry; friction utilization must be measured in a new "
                "shadow-only run."
            ),
            "controller_source": str(CONTROLLER_SOURCE),
            "transform_source": str(FORCE_HELPER_SOURCE),
        },
    )
    write_json(
        "MU_SAFE_DEFINITION.json",
        {
            "schema": "MU_SAFE_DEFINITION_V1",
            "definition": "10th percentile of the root7703 actual P4-B posterior discrete member means",
            "value": mu_safe,
            "posterior_member_count": len(members),
            "posterior_members": members.tolist(),
            "probe_valid": bool(posterior.get("probe_valid")),
            "fixed_prior_used": bool(posterior.get("fixed_prior_used")),
            "gt_mu_used": bool(posterior.get("gt_mu_used")),
            "source": str(MU_SOURCE),
            "scope": (
                "root-specific actual probe posterior; used only by the low-level "
                "safety margin, never for posterior or selector fitting"
            ),
        },
    )

    actuator = fit_one_step_beta(rows)
    write_json("ACTUATOR_ONE_STEP_MODEL.json", actuator)
    beta = float(actuator["beta_follow"])

    left = np.asarray([number(row, "left_true_normal_force") for row in rows])
    right = np.asarray([number(row, "right_true_normal_force") for row in rows])
    weak = np.minimum(left, right)
    imbalance = np.abs(left - right) / np.maximum(left + right, 1e-9)
    derivative = np.zeros(len(rows), dtype=float)
    derivative[3:] = (weak[3:] - weak[:-3]) / PREDICTION_HORIZON_S
    projected_weak = weak + derivative * PREDICTION_HORIZON_S
    bilateral = np.asarray([flag(row, "bilateral_contact") for row in rows])
    warning_raw = (
        bilateral
        & (derivative < 0.0)
        & (projected_weak < TARGET_N / 2.0)
    )
    warning = np.zeros(len(rows), dtype=bool)
    streak = 0
    for index, raw in enumerate(warning_raw):
        streak = streak + 1 if raw else 0
        warning[index] = bilateral[index] and streak >= 2

    losses = np.where(bilateral[:-1] & ~bilateral[1:])[0] + 1
    if not len(losses):
        raise RuntimeError("failed trace contains no bilateral contact loss")
    final_loss = int(losses[-1])
    recoveries = np.where(~bilateral[:-1] & bilateral[1:])[0] + 1
    episode_start = int(recoveries[recoveries < final_loss][-1])
    warnings = np.where(warning[episode_start:final_loss])[0] + episode_start
    first_warning = int(warnings[0]) if len(warnings) else None

    shadow_rows: list[dict[str, Any]] = []
    reclose_accum = 0.0
    supervisor_actions: list[str] = []
    for index, row in enumerate(rows):
        f_raw = number(row, "F_meas_raw")
        d_actual = number(row, "d_actual_before")
        d_cmd = number(row, "d_previous_command")
        release = max(number(row, "adaptive_release_after_rate_limit_m"), 0.0)
        candidate_cmd = d_cmd + release
        baseline_next_actual = d_actual + beta * (d_cmd - d_actual)
        candidate_next_actual = d_actual + beta * (candidate_cmd - d_actual)
        predicted_force_baseline = f_raw - SLOPE_N_PER_M * (
            baseline_next_actual - d_actual
        )
        predicted_force_candidate = f_raw - SLOPE_N_PER_M * (
            candidate_next_actual - d_actual
        )
        predicted_improves = abs(predicted_force_candidate - TARGET_N) < abs(
            predicted_force_baseline - TARGET_N
        )
        decision = "TRACK"
        shadow_delta = 0.0
        if f_raw > TARGET_N + 0.25 and release > 0.0:
            if warning[index]:
                decision = "HOLD_RELEASE"
            elif predicted_improves:
                decision = "CAUTIOUS_RELEASE"
                shadow_delta = release
        critical = bool(
            warning[index]
            and projected_weak[index] < 0.75 * TARGET_N / 2.0
            and derivative[index] < 0.0
        )
        can_reclose = (
            critical
            and shadow_delta == 0.0
            and f_raw <= RECLOSE_FORCE_BOUND_N
            and reclose_accum < RECLOSE_LIMIT_M
        )
        if can_reclose:
            amount = min(
                RECLOSE_RATE_MPS * DT,
                RECLOSE_LIMIT_M - reclose_accum,
            )
            shadow_delta = -amount
            reclose_accum += amount
            decision = "RECLOSE"
        elif not warning[index]:
            reclose_accum = 0.0
        supervisor_actions.append(decision)
        shadow_rows.append(
            {
                "sim_time_s": number(row, "sim_time_s"),
                "env_step": int(number(row, "env_step")),
                "physics_step": int(number(row, "physics_step")),
                "F_des": TARGET_N,
                "F_left_normal_N": left[index],
                "F_right_normal_N": right[index],
                "F_left_tangential_N": "",
                "F_right_tangential_N": "",
                "mu_safe": mu_safe,
                "rho_left": "",
                "rho_right": "",
                "rho_max": "",
                "weak_force_N": weak[index],
                "weak_force_derivative_Nps": derivative[index],
                "projected_weak_force_50ms_N": projected_weak[index],
                "imbalance": imbalance[index],
                "bilateral_contact": bool(bilateral[index]),
                "warning_raw": bool(warning_raw[index]),
                "warning_confirmed": bool(warning[index]),
                "candidate_release_m": release,
                "actuator_beta_follow": beta,
                "predicted_actual_aperture_m": candidate_next_actual,
                "predicted_next_force_N": predicted_force_candidate,
                "predicted_no_release_force_N": predicted_force_baseline,
                "predicted_force_error_improves": bool(predicted_improves),
                "shadow_action": decision,
                "shadow_command_delta_m": shadow_delta,
                "actual_command_unchanged": True,
            }
        )

    output_rows = []
    for index, row in enumerate(shadow_rows):
        output_rows.append(
            {
                **row,
                "contact_loss_event": bool(index in set(losses.tolist())),
            }
        )
    write_csv("FAILED_4N_SAFETY_MARGIN_TRACE.csv", output_rows)
    write_csv("PREDICTIVE_RELEASE_SHADOW_TRACE.csv", output_rows)

    final_loss_time = number(rows[final_loss], "sim_time_s")
    first_warning_time = (
        None if first_warning is None else number(rows[first_warning], "sim_time_s")
    )
    lead_ms = (
        None
        if first_warning_time is None
        else (final_loss_time - first_warning_time) * 1000.0
    )
    safety_action_indices = [
        index
        for index in range(episode_start, final_loss)
        if supervisor_actions[index] in {"HOLD_RELEASE", "RECLOSE"}
    ]
    reclose_indices = [
        index
        for index in range(episode_start, final_loss)
        if supervisor_actions[index] == "RECLOSE"
    ]
    warning_analysis = {
        "schema": "SAFETY_WARNING_ANALYSIS_V1",
        "signal": "CONTACT_WEAKENING_PROXY",
        "weak_force": "min(left_normal_force,right_normal_force)",
        "derivative": "three-physics-step backward difference",
        "prediction_horizon_ms": PREDICTION_HORIZON_S * 1000.0,
        "warning_rule": (
            "two consecutive bilateral samples with negative derivative and "
            "50ms projected weak force below F_des/2"
        ),
        "final_contact_loss_physics_step": int(number(rows[final_loss], "physics_step")),
        "final_contact_loss_time_s": final_loss_time,
        "first_warning_physics_step": (
            None if first_warning is None else int(number(rows[first_warning], "physics_step"))
        ),
        "first_warning_time_s": first_warning_time,
        "warning_lead_time_ms": lead_ms,
        "proactive_warning_valid": bool(lead_ms is not None and lead_ms > 0.0),
        "bilateral_still_true_at_first_warning": bool(
            first_warning is not None and bilateral[first_warning]
        ),
        "source_trace": str(SOURCE),
        "important_limitation": (
            "This historical trace supports normal-force weakening only; a new "
            "shadow-only run is required to validate friction utilization."
        ),
    }
    write_json("SAFETY_WARNING_ANALYSIS.json", warning_analysis)

    max_jump = max(abs(row["shadow_command_delta_m"]) for row in shadow_rows)
    high_force_reclose = any(
        supervisor_actions[index] == "RECLOSE"
        and number(rows[index], "F_meas_raw") > RECLOSE_FORCE_BOUND_N
        for index in range(len(rows))
    )
    shadow_result = {
        "schema": "SAFETY_SUPERVISOR_SHADOW_RESULT_V1",
        "status": "PRELIMINARY_PROXY_SHADOW",
        "safety_signal": "CONTACT_WEAKENING_PROXY",
        "proactive_warning_valid": warning_analysis["proactive_warning_valid"],
        "safety_action_occurs_before_contact_loss": bool(safety_action_indices),
        "first_safety_action_physics_step": (
            None
            if not safety_action_indices
            else int(number(rows[safety_action_indices[0]], "physics_step"))
        ),
        "release_suppressed_before_contact_loss": any(
            supervisor_actions[index] == "HOLD_RELEASE"
            for index in range(episode_start, final_loss)
        ),
        "contact_preservation_triggered_before_contact_loss": bool(reclose_indices),
        "first_reclose_physics_step": (
            None if not reclose_indices else int(number(rows[reclose_indices[0]], "physics_step"))
        ),
        "max_shadow_command_jump_m": max_jump,
        "no_large_release_command_jump": max_jump <= 0.00005 + 1e-12,
        "no_persistent_reclose_to_high_force": not high_force_reclose,
        "shadow_mode_pass": False,
        "live_gate_authorized": False,
        "blocking_reason": (
            "Historical trace lacks per-finger tangential force and the fitted "
            "one-step actuator model has low predictive reliability. Run an "
            "instrumented shadow-only fresh process before any live actuation."
        ),
    }
    write_json("SAFETY_SUPERVISOR_SHADOW_RESULT.json", shadow_result)
    print(
        json.dumps(
            {
                "mu_safe": mu_safe,
                "warning_lead_time_ms": lead_ms,
                "beta_follow": beta,
                "beta_r2": actuator["r2_no_intercept"],
                "preliminary_shadow_pass": shadow_result["shadow_mode_pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
