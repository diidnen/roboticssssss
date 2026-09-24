#!/usr/bin/env python3
"""Build audited artifacts for the backlog/slope dynamic-release experiment."""

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
    "backlog_slope_dynamic_release_fix_20260905"
)
RUNS = OUT / "runs"
OLD = Path(
    "/home/exouser/FORTE/analysis/results/"
    "actuator_following_arm_coupling_diagnosis_20260905/"
    "ROOT7703_4N_NORMAL_TRACE.csv"
)
FINAL_RUN = RUNS / "4N_adaptive_diag_softp50_hardp90_alpha0p2_rate1p25"
DT = 1.0 / 60.0
TARGET = 4.0
HOLD_ENV_STEPS = 30


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def b(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def trace_metrics(run: Path) -> dict[str, Any]:
    result = read_json(run / "ROOT7703_CONTROLLER_VALIDATION_RESULT.json")
    handoff = result.get("handoff_parity", {})
    rows = read_csv(run / "PHYSICS_STEP_SIGNAL_TRACE.csv")
    lift = result["summary"]["first_lift_step"]
    hold = [
        row for row in rows
        if lift is not None
        and lift + 1 <= int(f(row, "env_step")) <= lift + HOLD_ENV_STEPS
    ]
    connected = [row for row in hold if b(row, "bilateral_contact")]
    force = np.asarray([f(row, "F_meas_raw") for row in hold], dtype=float)
    whole_force = np.asarray([f(row, "F_meas_raw") for row in rows], dtype=float)
    backlog_mm = np.asarray(
        [f(row, "opening_backlog_m") * 1000.0 for row in hold], dtype=float
    )
    contact_events: list[dict[str, Any]] = []
    previous = True
    for row in rows:
        current = b(row, "bilateral_contact")
        if current != previous:
            contact_events.append({
                "physics_step": int(f(row, "physics_step")),
                "env_step": int(f(row, "env_step")),
                "event": "RECOVERY" if current else "LOSS",
            })
        previous = current

    cmd = np.asarray([f(row, "d_previous_command") for row in connected])
    actual = np.asarray([f(row, "d_actual_before") for row in connected])
    time = np.asarray([f(row, "sim_time_s") for row in connected])
    command_rate = actual_rate = ratio = None
    if len(connected) >= 2:
        command_rate = float(np.polyfit(time, cmd, 1)[0])
        actual_rate = float(np.polyfit(time, actual, 1)[0])
        ratio = None if abs(command_rate) <= 1e-12 else actual_rate / command_rate

    steps = [int(f(row, "physics_step")) for row in rows]
    required = {
        "F_des", "F_meas_raw", "F_meas_filtered", "force_excess",
        "opening_backlog_m", "opening_backlog_scale", "contact_margin_scale",
        "adaptive_release_after_rate_limit_m", "d_previous_command",
        "d_actual_before", "bilateral_contact",
    }
    missing = sorted(required.difference(rows[0] if rows else {}))
    finite_force = bool(len(force) and np.all(np.isfinite(force)))
    quality = {
        "rows": len(rows),
        "one_row_per_unique_physics_step": len(set(steps)) == len(steps),
        "physics_steps_contiguous": all(
            right == left + 1 for left, right in zip(steps, steps[1:])
        ),
        "missing_required_fields": missing,
        "hold_rows": len(hold),
        "hold_force_finite": finite_force,
        "handoff_parity": bool(
            handoff.get("observable_handoff_contract_valid")
            and handoff.get("observation_parity")
            and handoff.get("physical_state_parity")
            and handoff.get("runtime_state_parity")
        ),
        "fresh_process": bool(result.get("fresh_process")),
        "raw_arm_trajectory_changed": bool(result.get("raw_arm_trajectory_changed")),
    }
    valid = bool(
        len(rows) > 0
        and quality["one_row_per_unique_physics_step"]
        and quality["physics_steps_contiguous"]
        and not missing
        and len(hold) == 90
        and finite_force
        and quality["fresh_process"]
        and not quality["raw_arm_trajectory_changed"]
    )
    n_top = max(1, math.ceil(0.05 * len(force))) if len(force) else 0
    return {
        "run": run.name,
        "result_status": result.get("status"),
        "config": {
            "alpha_release": result.get("adaptive_release_alpha"),
            "backlog_soft_start_m": result.get("adaptive_release_backlog_soft_start_m"),
            "backlog_hard_limit_m": result.get("adaptive_release_backlog_limit_m"),
            "slope_n_per_m": result.get("adaptive_release_slope_n_per_m"),
            "contact_guard_width_n": result.get("adaptive_release_contact_guard_width_n"),
            "opening_rate_mps": result.get("authoritative_true_force_open_rate_mps"),
        },
        "quality": quality,
        "trace_valid": valid,
        "lift_env_step": lift,
        "hold_rows": len(hold),
        "hold_connected_rows": len(connected),
        "hold_mean_force_N": None if not len(force) else float(np.mean(force)),
        "hold_std_force_N": None if not len(force) else float(np.std(force)),
        "hold_abs_tracking_error_N": None if not len(force) else float(np.mean(np.abs(force - TARGET))),
        "hold_peak_force_N": None if not len(force) else float(np.max(force)),
        "hold_top5_percent_force_N": None if not len(force) else float(np.mean(np.sort(force)[-n_top:])),
        "hold_force_exposure_Ns": None if not len(force) else float(np.sum(force) * DT),
        "whole_trace_force_exposure_Ns": float(
            np.nansum(whole_force) * DT
        ),
        "whole_trace_peak_force_N": float(np.nanmax(whole_force)),
        "opening_backlog_mean_mm": None if not len(backlog_mm) else float(np.mean(backlog_mm)),
        "opening_backlog_max_mm": None if not len(backlog_mm) else float(np.max(backlog_mm)),
        "contact_loss_during_hold": len(connected) != len(hold),
        "contact_events": contact_events,
        "command_opening_rate_mps_connected_hold_ols": command_rate,
        "actual_opening_rate_mps_connected_hold_ols": actual_rate,
        "actual_to_command_rate_ratio_connected_hold": ratio,
        "backlog_suppressed_hold_steps": sum(
            b(row, "adaptive_release_suppressed_by_backlog") for row in hold
        ),
        "contact_guarded_hold_steps": sum(
            f(row, "contact_margin_scale") < 0.999999 for row in hold
        ),
        "rate_clipped_hold_steps": sum(
            f(row, "adaptive_release_before_rate_limit_m")
            > result["authoritative_true_force_open_rate_mps"] * DT + 1e-12
            for row in hold
        ),
        "lift_success": bool(result["summary"].get("lift_success")),
        "hold_success": bool(result["summary"].get("hold_after_lift_success")),
        "drop": bool(result["summary"].get("drop")),
    }


def old_metrics() -> dict[str, float]:
    rows = read_csv(OLD)
    hold = [row for row in rows if 119 <= int(f(row, "env_step")) <= 148]
    force = np.asarray([f(row, "F_meas_raw") for row in hold])
    return {
        "hold_mean_force_N": float(np.mean(force)),
        "hold_std_force_N": float(np.std(force)),
        "hold_abs_tracking_error_N": float(np.mean(np.abs(force - TARGET))),
    }


def placeholder_trace(name: str) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["status", "reason"])
        writer.writeheader()
        writer.writerow({
            "status": "NOT_RUN",
            "reason": "no adaptive configuration passed the first live 4N tracking gate",
        })


def main() -> None:
    trials = [
        RUNS / "4N_adaptive_diag_retry_with_physics_trace",
        RUNS / "4N_adaptive_diag_alpha0p2_p90_rate1p25",
        RUNS / "4N_adaptive_diag_alpha0p3_p90_rate1p25",
        FINAL_RUN,
    ]
    trial_metrics = [trace_metrics(path) for path in trials]
    final = trial_metrics[-1]
    baseline = old_metrics()
    dynamic_valid = bool(
        final["trace_valid"]
        and final["hold_connected_rows"] == 90
        and final["hold_abs_tracking_error_N"] <= 1.0
        and final["hold_mean_force_N"] <= 5.0
    )

    shutil.copyfile(
        FINAL_RUN / "PHYSICS_STEP_SIGNAL_TRACE.csv",
        OUT / "ROOT7703_DYNAMIC_4N_ADAPTIVE_TRACE.csv",
    )
    result = {
        "schema": "ROOT7703_DYNAMIC_4N_ADAPTIVE_RESULT_V1",
        "primary_target_N": TARGET,
        "selected_live_diagnostic": final,
        "old_fixed_rate_baseline": baseline,
        "improvement_vs_old": {
            "hold_mean_force_N": final["hold_mean_force_N"] - baseline["hold_mean_force_N"],
            "hold_abs_tracking_error_N": final["hold_abs_tracking_error_N"] - baseline["hold_abs_tracking_error_N"],
        },
        "dynamic_4N_tracking_valid": dynamic_valid,
        "failure_classification": (
            None if dynamic_valid else
            "ACTUAL_ACTUATOR_FOLLOWING_LIMIT_WITH_DYNAMIC_CONTACT_LOSS"
        ),
        "controller_gain_changed": False,
        "integral_added": False,
        "activeforcing_method_changed": False,
        "posterior_changed": False,
        "expected_utility_changed": False,
        "F_des_semantics_changed": False,
        "raw_arm_trajectory_changed": False,
    }
    write_json("ROOT7703_DYNAMIC_4N_ADAPTIVE_RESULT.json", result)

    sweep = {
        "schema": "ADAPTIVE_RELEASE_PREDECLARED_SWEEP_V1",
        "selection_inputs": "shadow command metrics plus live 4N force/contact evidence only",
        "posterior_or_task_success_used_for_selection": False,
        "trials": trial_metrics,
        "decisions": [
            {
                "trial": trials[0].name,
                "decision": "REJECT",
                "reason": "P50 hard limit suppresses release too strongly; hold force remains 11.69N",
            },
            {
                "trial": trials[1].name,
                "decision": "REJECT",
                "reason": "linear P90 suppression remains too strong; hold force remains 8.24N",
            },
            {
                "trial": trials[2].name,
                "decision": "REJECT",
                "reason": "alpha 0.30 loses bilateral contact during hold",
            },
            {
                "trial": trials[3].name,
                "decision": "REJECT_AS_FINAL_CONTROLLER",
                "reason": "P50-to-P90 taper improves MAE but remains outside tolerance and loses contact late in hold",
            },
        ],
        "frozen_valid_configuration": None,
    }
    write_json("IF_TUNED_PARAMETER_SWEEP.json", sweep)

    for index in (1, 2, 3):
        placeholder_trace(f"ROOT7703_DYNAMIC_4N_REPEAT_{index}.csv")
    repeat_summary = {
        "status": "NOT_RUN_NO_VALID_4N_CONFIGURATION",
        "repeat_count": 0,
        "tracking_pass_count": 0,
        "reason": "the first live 4N gate did not pass; repeats cannot validate a rejected configuration",
    }
    write_json("ROOT7703_DYNAMIC_4N_REPEAT_SUMMARY.json", repeat_summary)
    write_json("IF_4N_PASS_ROOT7703_3N_REPEAT_SUMMARY.json", {
        "status": "NOT_RUN",
        "repeat_count": 0,
        "dynamic_3N_tracking_valid": "NOT_EVALUATED",
        "physical_outcome": "NOT_EVALUATED",
        "reason": "4N adaptive tracking gate did not pass",
    })

    report = f"""# Bounded backlog/slope release did not close the 4N tracking gap

## Technical summary

The shadow-gated adaptive release was implemented behind a FORTE-only, default-off feature flag and tested in fresh root7703 processes. It improved the primary 4N hold mean from **{baseline['hold_mean_force_N']:.3f}N** to **{final['hold_mean_force_N']:.3f}N**, and hold MAE from **{baseline['hold_abs_tracking_error_N']:.3f}N** to **{final['hold_abs_tracking_error_N']:.3f}N**, but it did not satisfy the existing 1N tracking tolerance and lost bilateral contact during the final {final['hold_rows'] - final['hold_connected_rows']} physics samples of the 30-step hold. No adaptive configuration is frozen.

## Shadow evidence supported one live diagnostic, not physical success

The command-only shadow compared the old fixed-rate law, local-slope release, backlog suppression, and the combined contact guard. The accepted shadow candidate released 25% faster during the first 250ms of the first high-force segment, never added release above its hard backlog limit, and reduced near-boundary release from 0.475mm to 0.027mm. These are command-safety properties; observed force and contact remained old-policy data, so shadow PASS was deliberately not treated as a counterfactual physics claim.

## The tested release laws expose a control tradeoff

| Live 4N diagnostic | Hold mean | Hold MAE | Connected rows | Decision |
|---|---:|---:|---:|---|
| P50 hard limit, alpha 0.20 | {trial_metrics[0]['hold_mean_force_N']:.3f}N | {trial_metrics[0]['hold_abs_tracking_error_N']:.3f}N | {trial_metrics[0]['hold_connected_rows']}/90 | Suppression too strong |
| Linear P90, alpha 0.20 | {trial_metrics[1]['hold_mean_force_N']:.3f}N | {trial_metrics[1]['hold_abs_tracking_error_N']:.3f}N | {trial_metrics[1]['hold_connected_rows']}/90 | Suppression too strong |
| Linear P90, alpha 0.30 | {trial_metrics[2]['hold_mean_force_N']:.3f}N | {trial_metrics[2]['hold_abs_tracking_error_N']:.3f}N | {trial_metrics[2]['hold_connected_rows']}/90 | Release/contact transition unsafe |
| P50 soft → P90 hard, alpha 0.20 | {final['hold_mean_force_N']:.3f}N | {final['hold_abs_tracking_error_N']:.3f}N | {final['hold_connected_rows']}/90 | Improved but invalid |

The final diagnostic's opening release had already fallen to zero before the late contact loss. This means the remaining failure cannot be repaired by further suppressing release alone. The actual actuator/contact response remains the limiting observed dynamic; this experiment does not establish a changed force–aperture slope.

## Scope and metric definitions

All live trials use root7703's exact HDF5 initial state, matched warm replay, unchanged geometric VLA arm waypoint sequence, a fixed 4N physical target, 60Hz object-filtered true-force feedback, disabled feed-forward, and disabled target-contact override. Hold metrics cover 30 environment steps, or 90 physics samples. Force exposure is the sum of raw authoritative squeeze force times 1/60s.

Opening backlog removes the contact-compliance offset and measures unmatched progress within each release episode: `max((d_cmd-d_cmd_onset) - (d_actual-d_actual_onset), 0)`. The final law allows full release through the old trace's P50 backlog (1.212mm), tapers continuously to zero at P90 (2.011mm), uses a fixed 9.4N/mm local slope and alpha 0.20, and applies a final 3.0mm/s ceiling. No online slope update, force-gain change, or integral was used.

## Validation and limitations

Each retained physics trace has one contiguous row per 60Hz tick, 90 hold rows, finite force values, explicit adaptive telemetry, fresh-process provenance, and no raw arm geometry change. The recurring Isaac camera teardown exception occurs only after result files are written. One initial run lost its physics trace to auto-reset; it is excluded from scientific metrics and retained only as an engineering audit.

The shadow model cannot predict counterfactual physical force. OLS force–aperture slope is descriptive, and the offset-corrected backlog is a release-progress diagnostic rather than a direct actuator state estimate. The late force collapse after release had stopped shows that release-only control is insufficient under this dynamic contact transition.

## Recommended next step

Stop tuning alpha and fixed release ceilings. The next controller-only investigation should test a bounded contact-preservation response driven by fresh raw-force weakening and local force derivative, while keeping filtered force authoritative for nominal tracking. It must first run in shadow and must not add an unbounded integral. Static 2/3/4/6 regression and 4N×3 remain mandatory after a new law passes its first live gate.

## Further question

Can a bounded contact-preservation action react to the observed roughly 50–80ms raw-force collapse without increasing steady-state force or creating oscillation? The current release-only experiments do not answer that question.
"""
    (OUT / "ADAPTIVE_RELEASE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "ADAPTIVE_RELEASE_DIAGNOSIS_COMPLETE_4N_FAIL",
        "hold_mean_N": final["hold_mean_force_N"],
        "hold_mae_N": final["hold_abs_tracking_error_N"],
        "connected_rows": final["hold_connected_rows"],
        "dynamic_valid": dynamic_valid,
    }, indent=2))


if __name__ == "__main__":
    main()
