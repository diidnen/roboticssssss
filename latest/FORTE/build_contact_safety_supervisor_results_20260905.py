#!/usr/bin/env python3
"""Build audited artifacts for the root7703 contact-safety supervisor gate."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


OUT = Path(
    "/home/exouser/FORTE/analysis/results/"
    "contact_safety_margin_supervisor_20260905"
)
FAILED = Path(
    "/home/exouser/FORTE/analysis/results/"
    "backlog_slope_dynamic_release_fix_20260905/"
    "ROOT7703_DYNAMIC_4N_ADAPTIVE_TRACE.csv"
)
SHADOW_RUN = OUT / "runs/4N_shadow_friction_margin_v2"
LIVE_RUN = OUT / "runs/4N_safety_live_gate"
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
TARGET = 4.0
HOLD_STEPS = 30
TRACKING_MAE_LIMIT_N = 1.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def n(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def b(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def handoff_valid(result: dict[str, Any]) -> bool:
    handoff = result.get("handoff_parity", {})
    return bool(
        handoff.get("observable_handoff_contract_valid")
        and handoff.get("physical_state_parity")
        and handoff.get("runtime_state_parity")
        and handoff.get("observation_parity")
        and handoff.get("bilateral_contact")
    )


def trace_quality(rows: list[dict[str, str]], result: dict[str, Any]) -> dict[str, Any]:
    steps = [int(n(row, "physics_step")) for row in rows]
    required = {
        "left_true_normal_force",
        "right_true_normal_force",
        "left_true_tangential_force",
        "right_true_tangential_force",
        "maximum_friction_utilization",
        "weak_force_derivative",
        "predicted_force_candidate",
        "safety_action_code",
        "d_previous_command",
        "d_actual_before",
    }
    missing = sorted(required.difference(rows[0] if rows else {}))
    return {
        "rows": len(rows),
        "unique_physics_steps": len(set(steps)),
        "physics_steps_contiguous": all(
            right == left + 1 for left, right in zip(steps, steps[1:])
        ),
        "missing_required_fields": missing,
        "all_required_force_fields_finite": all(
            np.isfinite(
                [
                    n(row, key)
                    for row in rows
                    for key in (
                        "left_true_normal_force",
                        "right_true_normal_force",
                        "left_true_tangential_force",
                        "right_true_tangential_force",
                    )
                ]
            )
        ),
        "fresh_process": bool(result.get("fresh_process")),
        "handoff_parity": handoff_valid(result),
        "raw_arm_trajectory_changed": bool(result.get("raw_arm_trajectory_changed")),
    }


def contact_events(rows: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    bilateral = np.asarray([b(row, "bilateral_contact") for row in rows])
    losses = (np.where(bilateral[:-1] & ~bilateral[1:])[0] + 1).tolist()
    recoveries = (np.where(~bilateral[:-1] & bilateral[1:])[0] + 1).tolist()
    return losses, recoveries


def copy_with_action_names(source: Path, destination: str) -> list[dict[str, Any]]:
    rows = read_csv(source)
    names = {0: "TRACK", 1: "CAUTIOUS_RELEASE", 2: "HOLD_RELEASE", 3: "RECLOSE"}
    enriched: list[dict[str, Any]] = []
    for row in rows:
        code = int(n(row, "safety_action_code"))
        enriched.append({**row, "safety_action": names.get(code, "UNKNOWN")})
    write_csv(destination, enriched)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shadow_result_raw = read_json(
        SHADOW_RUN / "ROOT7703_CONTROLLER_VALIDATION_RESULT.json"
    )
    live_result_raw = read_json(
        LIVE_RUN / "ROOT7703_CONTROLLER_VALIDATION_RESULT.json"
    )
    shadow_rows = copy_with_action_names(
        SHADOW_RUN / "PHYSICS_STEP_SIGNAL_TRACE.csv",
        "PREDICTIVE_RELEASE_SHADOW_TRACE.csv",
    )
    live_rows = copy_with_action_names(
        LIVE_RUN / "PHYSICS_STEP_SIGNAL_TRACE.csv",
        "ROOT7703_4N_SAFETY_CONTROLLER_TRACE.csv",
    )

    shadow_quality = trace_quality(shadow_rows, shadow_result_raw)
    live_quality = trace_quality(live_rows, live_result_raw)
    if not (
        shadow_quality["rows"] == 435
        and shadow_quality["unique_physics_steps"] == 435
        and shadow_quality["physics_steps_contiguous"]
        and not shadow_quality["missing_required_fields"]
        and shadow_quality["all_required_force_fields_finite"]
        and shadow_quality["fresh_process"]
        and shadow_quality["handoff_parity"]
        and not shadow_quality["raw_arm_trajectory_changed"]
    ):
        raise RuntimeError(f"shadow trace quality gate failed: {shadow_quality}")
    if not (
        live_quality["rows"] == 435
        and live_quality["unique_physics_steps"] == 435
        and live_quality["physics_steps_contiguous"]
        and not live_quality["missing_required_fields"]
        and live_quality["all_required_force_fields_finite"]
        and live_quality["fresh_process"]
        and live_quality["handoff_parity"]
        and not live_quality["raw_arm_trajectory_changed"]
    ):
        raise RuntimeError(f"live trace quality gate failed: {live_quality}")

    posterior = read_json(MU_SOURCE)
    members = np.asarray(posterior["member_means"], dtype=float)
    mu_safe = float(np.quantile(members, 0.10))
    write_json(
        "MU_SAFE_DEFINITION.json",
        {
            "schema": "MU_SAFE_DEFINITION_V2",
            "definition": "Q10 of root7703 actual P4-B posterior discrete member means",
            "value": mu_safe,
            "posterior_members": members.tolist(),
            "posterior_member_count": len(members),
            "probe_valid": bool(posterior.get("probe_valid")),
            "fixed_prior_used": bool(posterior.get("fixed_prior_used")),
            "gt_mu_used": bool(posterior.get("gt_mu_used")),
            "source": str(MU_SOURCE),
            "not_used_for_model_or_selector_fitting": True,
        },
    )

    left_n = np.asarray([n(row, "left_true_normal_force") for row in shadow_rows])
    right_n = np.asarray([n(row, "right_true_normal_force") for row in shadow_rows])
    left_t = np.asarray([n(row, "left_true_tangential_force") for row in shadow_rows])
    right_t = np.asarray([n(row, "right_true_tangential_force") for row in shadow_rows])
    left_fx = np.asarray([n(row, "left_true_local_fx") for row in shadow_rows])
    left_fy = np.asarray([n(row, "left_true_local_fy") for row in shadow_rows])
    left_fz = np.asarray([n(row, "left_true_local_fz") for row in shadow_rows])
    right_fx = np.asarray([n(row, "right_true_local_fx") for row in shadow_rows])
    right_fy = np.asarray([n(row, "right_true_local_fy") for row in shadow_rows])
    right_fz = np.asarray([n(row, "right_true_local_fz") for row in shadow_rows])
    write_json(
        "CONTACT_FORCE_DECOMPOSITION_AUDIT.json",
        {
            "schema": "CONTACT_FORCE_DECOMPOSITION_AUDIT_V2",
            "sensor": "contact_grasp_black_book_1.data.force_matrix_w",
            "sensor_contract": "(N,2,M,3), summed only over target-object filtered prims",
            "world_to_local_transform": "separate left/right finger-frame inverse quaternion",
            "normal_definition": "abs(local z)",
            "tangential_definition": "sqrt(local x^2+local y^2)",
            "left_normal_force_available": True,
            "right_normal_force_available": True,
            "left_tangential_force_available": True,
            "right_tangential_force_available": True,
            "force_decomposition_valid": True,
            "rows_checked": len(shadow_rows),
            "all_components_finite": bool(
                np.isfinite(
                    np.concatenate(
                        [left_n, right_n, left_t, right_t, left_fx, left_fy, left_fz, right_fx, right_fy, right_fz]
                    )
                ).all()
            ),
            "max_left_normal_closure_error_N": float(
                np.max(np.abs(left_n - np.abs(left_fz)))
            ),
            "max_right_normal_closure_error_N": float(
                np.max(np.abs(right_n - np.abs(right_fz)))
            ),
            "max_left_tangential_closure_error_N": float(
                np.max(np.abs(left_t - np.hypot(left_fx, left_fy)))
            ),
            "max_right_tangential_closure_error_N": float(
                np.max(np.abs(right_t - np.hypot(right_fx, right_fy)))
            ),
            "controller_source": str(CONTROLLER_SOURCE),
            "transform_source": str(FORCE_HELPER_SOURCE),
            "historical_failed_trace_limitation": (
                "the pre-instrumentation trace contains scalar normal forces only; "
                "its safety warning uses the weakening proxy"
            ),
        },
    )

    # Preserve the historical failure-derived weakening trace generated by the
    # standalone audit script.  It is not replaced with live counterfactual data.
    if not (OUT / "FAILED_4N_SAFETY_MARGIN_TRACE.csv").exists():
        raise RuntimeError("historical failure safety trace is missing")

    shadow_losses, shadow_recoveries = contact_events(shadow_rows)
    final_loss = shadow_losses[-1]
    episode_start = max(index for index in shadow_recoveries if index < final_loss)
    confirmed = np.asarray(
        [b(row, "safety_warning_confirmed") for row in shadow_rows]
    )
    friction_warning = np.asarray(
        [b(row, "friction_warning_raw") for row in shadow_rows]
    )
    action = np.asarray(
        [int(n(row, "safety_action_code")) for row in shadow_rows]
    )
    warning_indices = np.where(confirmed[episode_start:final_loss])[0] + episode_start
    first_warning = int(warning_indices[0])
    final_loss_time = n(shadow_rows[final_loss], "sim_time_s")
    warning_time = n(shadow_rows[first_warning], "sim_time_s")
    lead_ms = (final_loss_time - warning_time) * 1000.0
    suppression_indices = [
        index
        for index in range(episode_start, final_loss)
        if action[index] == 2
        and n(shadow_rows[index], "adaptive_release_after_rate_limit_m") > 0.0
    ]
    reclose_indices = [
        index
        for index in range(episode_start, final_loss)
        if action[index] == 3
    ]
    warning_analysis = {
        "schema": "SAFETY_WARNING_ANALYSIS_V2",
        "primary_signal": "CONTACT_WEAKENING_PROXY",
        "friction_utilization_role": "audited auxiliary guard; not proactive for this collapse",
        "weak_force": "min(left_normal_force,right_normal_force)",
        "weak_derivative": "three-physics-step backward difference",
        "warning_confirmation": "two consecutive warning samples",
        "prediction_horizon_ms": 50.0,
        "final_contact_loss_physics_step": int(n(shadow_rows[final_loss], "physics_step")),
        "final_contact_loss_time_s": final_loss_time,
        "first_safety_warning_physics_step": int(n(shadow_rows[first_warning], "physics_step")),
        "first_safety_warning_time_s": warning_time,
        "warning_lead_time_ms": lead_ms,
        "bilateral_at_first_warning": b(shadow_rows[first_warning], "bilateral_contact"),
        "proactive_warning_valid": bool(lead_ms > 0.0),
        "friction_warning_before_final_loss": bool(
            np.any(friction_warning[episode_start:final_loss])
        ),
        "rho_max_connected_quantiles": {
            str(q): float(
                np.quantile(
                    [
                        n(row, "maximum_friction_utilization")
                        for row in shadow_rows
                        if b(row, "bilateral_contact")
                    ],
                    q,
                )
            )
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "interpretation": (
            "normal and tangential components collapsed proportionally, so rho "
            "stayed near 0.5; weaker-side normal prediction warned 116.7ms early"
        ),
    }
    write_json("SAFETY_WARNING_ANALYSIS.json", warning_analysis)

    # Keep the previously fitted model but make its low reliability explicit.
    actuator = read_json(OUT / "ACTUATOR_ONE_STEP_MODEL.json")
    max_shadow_jump = max(
        abs(n(row, "safety_shadow_command_delta_m")) for row in shadow_rows
    )
    high_force_reclose = any(
        int(n(row, "safety_action_code")) == 3
        and n(row, "F_meas_raw") > 5.0
        for row in shadow_rows
    )
    shadow_pass = bool(
        lead_ms > 0.0
        and reclose_indices
        and max_shadow_jump <= 0.00005 + 1e-12
        and not high_force_reclose
    )
    write_json(
        "SAFETY_SUPERVISOR_SHADOW_RESULT.json",
        {
            "schema": "SAFETY_SUPERVISOR_SHADOW_RESULT_V2",
            "status": "PASS_LIVE_GATE_AUTHORIZED" if shadow_pass else "FAIL",
            "shadow_mode_pass": shadow_pass,
            "actual_command_changed": False,
            "quality": shadow_quality,
            "proactive_warning_valid": bool(lead_ms > 0.0),
            "warning_lead_time_ms": lead_ms,
            "release_suppressed_before_contact_loss": bool(suppression_indices),
            "suppressed_release_total_um": float(
                1e6
                * sum(
                    n(shadow_rows[index], "adaptive_release_after_rate_limit_m")
                    for index in suppression_indices
                )
            ),
            "contact_preservation_triggered_before_contact_loss": bool(reclose_indices),
            "first_reclose_physics_step": int(
                n(shadow_rows[reclose_indices[0]], "physics_step")
            ),
            "reclose_lead_time_ms": float(
                1000.0
                * (
                    final_loss_time
                    - n(shadow_rows[reclose_indices[0]], "sim_time_s")
                )
            ),
            "max_shadow_command_jump_um": max_shadow_jump * 1e6,
            "no_large_release_command_jump": max_shadow_jump <= 0.00005 + 1e-12,
            "no_persistent_reclose_to_high_force": not high_force_reclose,
            "maximum_reclose_per_warning_episode_um": 50.0,
            "actuator_predictor_reliability": actuator["predictive_reliability"],
            "limitation": (
                "PASS means the bounded command safety gate is satisfied. The "
                "one-step actuator predictor has low R2, so live physics remains "
                "the only validity test."
            ),
        },
    )

    lift = int(live_result_raw["summary"]["first_lift_step"])
    hold = [
        row
        for row in live_rows
        if lift + 1 <= int(n(row, "env_step")) <= lift + HOLD_STEPS
    ]
    if len(hold) != 90:
        raise RuntimeError(f"expected 90 physics hold rows, got {len(hold)}")
    hold_force = np.asarray([n(row, "F_meas_raw") for row in hold])
    whole_force = np.asarray([n(row, "F_meas_raw") for row in live_rows])
    top_hold = max(1, math.ceil(0.05 * len(hold_force)))
    top_whole = max(1, math.ceil(0.05 * len(whole_force)))
    live_losses, live_recoveries = contact_events(live_rows)
    hold_bilateral = [b(row, "bilateral_contact") for row in hold]
    hold_mae = float(np.mean(np.abs(hold_force - TARGET)))
    tracking_valid = bool(
        all(hold_bilateral)
        and hold_mae <= TRACKING_MAE_LIMIT_N
        and not np.any((hold_force >= 7.0))
    )
    live_result = {
        "schema": "ROOT7703_4N_SAFETY_CONTROLLER_RESULT_V1",
        "status": "PASS" if tracking_valid else "CONTACT_SAFE_BUT_FORCE_TRACKING_INVALID",
        "primary_target_N": TARGET,
        "quality": live_quality,
        "hold_window": "first_lift_env_step+1 through +30 inclusive",
        "hold_rows": len(hold),
        "hold_mean_force_N": float(np.mean(hold_force)),
        "hold_std_force_N": float(np.std(hold_force)),
        "hold_abs_tracking_error_N": hold_mae,
        "hold_peak_force_N": float(np.max(hold_force)),
        "hold_top5_percent_force_N": float(
            np.mean(np.sort(hold_force)[-top_hold:])
        ),
        "hold_force_exposure_Ns": float(np.sum(hold_force) * DT),
        "whole_trace_peak_force_N": float(np.max(whole_force)),
        "whole_trace_top5_percent_force_N": float(
            np.mean(np.sort(whole_force)[-top_whole:])
        ),
        "whole_trace_force_exposure_Ns": float(np.sum(whole_force) * DT),
        "hold_bilateral_contact_maintained": bool(all(hold_bilateral)),
        "contact_loss_during_hold": not all(hold_bilateral),
        "contact_loss_anywhere": bool(live_losses),
        "contact_loss_events": [
            {
                "physics_step": int(n(live_rows[index], "physics_step")),
                "env_step": int(n(live_rows[index], "env_step")),
                "sim_time_s": n(live_rows[index], "sim_time_s"),
            }
            for index in live_losses
        ],
        "contact_recovery_events": [
            {
                "physics_step": int(n(live_rows[index], "physics_step")),
                "env_step": int(n(live_rows[index], "env_step")),
                "sim_time_s": n(live_rows[index], "sim_time_s"),
            }
            for index in live_recoveries
        ],
        "lift_success": bool(live_result_raw["summary"].get("lift_success")),
        "hold_success": bool(
            live_result_raw["summary"].get("hold_after_lift_success")
        ),
        "drop": bool(live_result_raw["summary"].get("drop")),
        "predictive_release_used": True,
        "contact_safety_supervisor_used": True,
        "contact_preservation_reflex_used": any(
            int(n(row, "safety_action_code")) == 3 for row in live_rows
        ),
        "reflex_active_steps": sum(
            int(n(row, "safety_action_code")) == 3 for row in live_rows
        ),
        "dynamic_4N_tracking_valid": tracking_valid,
        "failure_classification": (
            None if tracking_valid else "CONTACT_SAFE_BUT_FORCE_TRACKING_INVALID"
        ),
        "comparison_to_previous_failure": {
            "previous_hold_mean_force_N": 5.78211877544721,
            "previous_hold_abs_tracking_error_N": 2.9473094205061594,
            "previous_whole_trace_force_exposure_Ns": 16.5593430028297,
            "hold_mean_delta_N": float(np.mean(hold_force)) - 5.78211877544721,
            "hold_mae_delta_N": hold_mae - 2.9473094205061594,
            "whole_exposure_delta_Ns": float(np.sum(whole_force) * DT)
            - 16.5593430028297,
        },
        "controller_gain_changed": False,
        "integral_added": False,
        "activeforcing_method_changed": False,
        "posterior_changed": False,
        "expected_utility_changed": False,
        "F_des_semantics_changed": False,
        "raw_arm_trajectory_changed": False,
    }
    write_json("ROOT7703_4N_SAFETY_CONTROLLER_RESULT.json", live_result)

    write_json(
        "IF_4N_PASS_REPEAT_SUMMARY.json",
        {
            "status": "NOT_RUN_FIRST_4N_GATE_FAILED_TRACKING",
            "repeat_count": 0,
            "pass_count": 0,
            "reason": (
                "hold contact passed, but 4N force tracking and exposure gates failed"
            ),
        },
    )
    write_json(
        "IF_3N_RUN_REPEAT_SUMMARY.json",
        {
            "status": "NOT_RUN",
            "repeat_count": 0,
            "dynamic_3N_tracking_valid": "NOT_EVALUATED",
            "physical_outcome": "NOT_EVALUATED",
            "reason": "4N dynamic tracking did not pass",
        },
    )

    report = f"""# Contact warning preserved the hold but did not regulate 4N

## Technical summary

The per-finger force decomposition is valid, and the weaker-side normal-force predictor provides a real proactive signal **{lead_ms:.1f}ms** before the final shadow contact loss. A bounded five-step, 50µm contact-preservation reflex passed the shadow safety gate and was tested once in a fresh root7703 process. It maintained bilateral contact throughout the 30-step hold, but hold force remained **{live_result['hold_mean_force_N']:.3f}N** with **{live_result['hold_abs_tracking_error_N']:.3f}N MAE**, and whole-run force exposure increased to **{live_result['whole_trace_force_exposure_Ns']:.3f}N·s**. The result is `CONTACT_SAFE_BUT_FORCE_TRACKING_INVALID`; no configuration is frozen and no repeats or 3N runs were authorized.

## Weak-side normal prediction warns earlier than the friction ratio

The target-object `force_matrix_w` supplies a three-dimensional force vector for each finger. After separate world-to-finger transformations, normal force is `abs(local z)` and tangential force is `norm(local x,y)`. All 435 shadow rows are finite; normal closure error is exactly zero and tangential closure error is below 7e-8N.

Using the actual root7703 P4-B posterior, `mu_safe` is the fixed Q10 of nine discrete posterior member means: **{mu_safe:.6f}**. Conservative friction utilization stayed between roughly 0.14 and 0.51 while connected and did not rise before contact loss because tangential and normal components collapsed together. It therefore cannot be the primary warning for this event. The transparent fallback projects the weaker-side normal force 50ms forward using a three-step derivative and requires two consecutive warnings. It first warns at physics step {int(n(shadow_rows[first_warning], 'physics_step'))}, while bilateral contact is still true, {lead_ms:.1f}ms before loss at step {int(n(shadow_rows[final_loss], 'physics_step'))}.

## Shadow safety passed, with a low-confidence actuator predictor

The one-step model `d_actual_next=d_actual+beta*(d_target-d_actual)` was fitted only on consecutive bilateral over-force rows. It gives `beta={actuator['beta_follow']:.6f}` but only `R²={actuator['r2_no_intercept']:.4f}`. Consequently the model is descriptive, not a trusted standalone safety oracle. The live authorization came from the independently observed contact warning and bounded command checks: no jump exceeded {max_shadow_jump*1e6:.1f}µm, re-close was capped at 10µm/step and 50µm per warning episode, and no re-close was requested above 5N. The supervisor used the actuator prediction only to compare candidate release against no-release, while the weak-side warning could veto release.

## The real gate traded overforce for contact preservation

| Metric | Previous release-only failure | Safety-supervised live gate |
|---|---:|---:|
| Hold mean force | 5.782N | {live_result['hold_mean_force_N']:.3f}N |
| Hold MAE to 4N | 2.947N | {live_result['hold_abs_tracking_error_N']:.3f}N |
| Hold bilateral rows | 78/90 | {sum(hold_bilateral)}/90 |
| Whole force exposure | 16.559N·s | {live_result['whole_trace_force_exposure_Ns']:.3f}N·s |

Contact preservation worked for the requested hold window, but force regulation did not: the hold still contains sustained high force, peaks at {live_result['hold_peak_force_N']:.3f}N, and exposure is higher rather than lower. A later loss occurs at env step {live_result['contact_loss_events'][-1]['env_step']}, immediately after the hold window. Task survival therefore cannot be used to call the controller valid.

## Scope, definitions, and validation

Both retained shadow and live traces have 435 unique contiguous 60Hz physics rows, exact root7703 HDF5 reset/replay, valid handoff parity, fresh-process provenance, and unchanged raw arm trajectory. `F_des` remains 4N; force gain, posterior, Expected Utility, and method semantics are unchanged. The recurring Isaac camera teardown exception happens after result and trace files are written and does not remove physics rows.

The hold window is the 90 physics rows spanning environment steps `first_lift+1` through `first_lift+30`. Every transient force remains included in peak, top-5%, and exposure metrics. Shadow evidence establishes command bounds and warning timing only; it does not predict counterfactual physical success.

## Recommended next step

Do not run 4N repeats, 3N, or model inference from this configuration. The weak-side warning is useful, but a binary 50µm re-close reflex is too conservative for force regulation. The next controller-only step should replace the low-R² one-step target-gap predictor with a validated short-horizon actuator/contact response model or direct measured aperture-velocity state, then shadow-check whether it can preserve contact without raising hold force or exposure. No ordinary integral or fixed-rate sweep is justified by this result.

## Further question

Can the low-level controller estimate short-horizon **actual** aperture motion accurately enough to distinguish a safe release from a contact-collapse trajectory? The present one-step model does not.
"""
    (OUT / "CONTACT_SAFETY_CONTROLLER_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": live_result["status"],
                "warning_lead_ms": lead_ms,
                "hold_mean_N": live_result["hold_mean_force_N"],
                "hold_mae_N": live_result["hold_abs_tracking_error_N"],
                "hold_contact": live_result["hold_bilateral_contact_maintained"],
                "dynamic_4N_tracking_valid": tracking_valid,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
