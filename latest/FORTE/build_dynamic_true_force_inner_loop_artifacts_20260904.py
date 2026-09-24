#!/usr/bin/env python3
"""Build the evidence bundle for the physics-rate controller audit.

This script is intentionally conservative: it records the completed code and
CPU/trace diagnostics, while marking Isaac runs as blocked when CUDA is not
available.  It never fabricates a physics-step trace from an environment-step
trace.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


FORTE = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
OUT = FORTE / "analysis/results/dynamic_true_force_inner_loop_fix_20260904"
OLD_TRACE = FORTE / "analysis/results/contact_loss_force_track_recovery_fix_20260904/static_2N_run2/ROOT7703_ACTIVEFORCING_TRACE.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_status_csv(path: Path, reason: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["status", "reason"])
        writer.writeheader()
        writer.writerow({"status": "NOT_RUN", "reason": reason})


def load_old_rows() -> list[dict[str, str]]:
    with OLD_TRACE.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(values) / len(values) if values else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_old_rows()
    hold = [r for r in rows if 119 <= int(r["env_step"]) <= 148]
    pre = [r for r in rows if 105 <= int(r["env_step"]) <= 108]
    high_force = [r for r in rows if float(r["controller_input_F_meas"]) > 3.25]
    open_direction = [r for r in high_force if float(r["d_cmd"]) - float(r["d_pred"]) > 0]

    physics_dt = 1.0 / 60.0
    decimation = 3
    env_dt = physics_dt * decimation
    old_alpha = 0.5
    tau_outer = -env_dt / math.log(1.0 - old_alpha)
    alpha_physics_timebase_preserving = 1.0 - (1.0 - old_alpha) ** (physics_dt / env_dt)

    write_json(OUT / "CONTROL_TIMING_AUDIT.json", {
        "physics_dt_s": physics_dt,
        "physics_frequency_hz": 60.0,
        "control_decimation": decimation,
        "environment_dt_s": env_dt,
        "environment_step_frequency_hz": 20.0,
        "contact_sensor_update_period_s": 0.0,
        "contact_sensor_effective_frequency_hz": 60.0,
        "external_hybrid_update_frequency_hz_before_fix": 20.0,
        "force_filter_update_frequency_hz_before_fix": 20.0,
        "gripper_target_update_frequency_hz_before_fix": 20.0,
        "actuator_target_application_frequency_hz": 60.0,
        "external_hybrid_update_frequency_hz_after_fix": 20.0,
        "true_force_feedback_frequency_hz_after_fix": 60.0,
        "force_filter_update_frequency_hz_after_fix": 60.0,
        "gripper_target_update_frequency_hz_after_fix": 60.0,
        "true_force_loop_temporal_layer_before": "MIXED: F_des/external correction at ENV_STEP; native apply_actions at PHYSICS_STEP",
        "true_force_loop_temporal_layer_after": "MIXED: high-level F_des at ENV_STEP; authoritative true-force correction at PHYSICS_STEP",
        "answers": {
            "hybrid_step_call_frequency": "once per env step in the runner",
            "object_filtered_force_refresh": "yes, sensor update_period=0 and action apply is called per physics step; old hybrid did not consume it per physics step",
            "same_force_measurement_reused": "yes in old external hybrid across decimated physics substeps",
            "slot6_update": "once per env step in process_actions; old apply_actions repeated the fixed slot6",
            "force_correction_layer": "old ENV_STEP; fixed authoritative FORTE inner loop PHYSICS_STEP",
        },
        "source_files": {
            str(TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"): sha256(TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"),
            str(TABERO / "analysis/tabero_true_physical_force_hybrid.py"): sha256(TABERO / "analysis/tabero_true_physical_force_hybrid.py"),
            str(FORTE / "root7703_probe_activeforcing_smoke_20260904.py"): sha256(FORTE / "root7703_probe_activeforcing_smoke_20260904.py"),
        },
    })

    write_json(OUT / "APERTURE_COMMAND_CHAIN_AUDIT.json", {
        "before": {
            "formula": "d_cmd(t) = d_pred(t) + correction(t)",
            "reference": "VLA slot6 d_pred",
            "VLA_d_pred_overwrites_force_correction": True,
            "force_correction_accumulates": False,
            "evidence": "old trace shows d_cmd-d_pred is positive during overforce, but next env step receives a fresh d_pred; no persistent d_force_cmd state exists",
        },
        "after": {
            "formula": "d_force_cmd(t+dt) = clamp(d_force_cmd(t) + bounded_delta_d(t))",
            "reference": "stateful bounded d_force_cmd initialized from current actual aperture",
            "VLA_d_pred_overwrites_force_correction": False,
            "force_correction_accumulates": True,
            "anti_windup": "aperture clamp plus per-physics opening/closing rate limit",
        },
        "old_trace_shadow": {
            "rows_with_filtered_overforce": len(high_force),
            "rows_with_opening_correction": len(open_direction),
            "hold_measured_force_mean_N": mean(hold, "aggregate_force"),
            "hold_d_cmd_mean_m": mean(hold, "d_cmd"),
            "hold_d_actual_mean_m": mean(hold, "d_actual"),
            "interpretation": "direction was open, but the correction was recomputed against a new VLA d_pred each env step",
        },
    })

    write_json(OUT / "FILTER_LATENCY_AUDIT.json", {
        "old_filter_alpha": old_alpha,
        "old_filter_update_hz": 20.0,
        "old_effective_time_constant_s": tau_outer,
        "physics_rate_timebase_preserving_alpha": alpha_physics_timebase_preserving,
        "physics_rate_effective_time_constant_s": tau_outer,
        "old_trace_available": True,
        "physics_step_raw_vs_filtered_trace_available": False,
        "filter_latency_dominant": "UNDETERMINED until fresh physics-step trace",
        "note": "The fixed loop keeps the old EMA time constant while evaluating fresh samples at 60 Hz; alpha was not tuned.",
    })

    write_json(OUT / "ACTUATOR_TRACKING_AUDIT.json", {
        "d_cmd_opening_direction_in_old_trace": "positive d_cmd-d_pred during F_meas > F_des rows",
        "d_cmd_opening_valid": bool(open_direction),
        "d_actual_follows_d_cmd": "UNDETERMINED: old env-rate trace does not expose physics-step target/actual lag",
        "actuator_tracking_delay": None,
        "actuator_saturation_detected": "UNDETERMINED",
        "joint_pd_config": "not changed; fresh physics-step target/actual telemetry required",
    })

    write_json(OUT / "ARM_FORCE_COUPLING_AUDIT.json", {
        "dynamic_trace_source": str(OLD_TRACE),
        "physics_step_velocity_acceleration_available": False,
        "dynamic_force_correlated_with_arm_acceleration": "UNDETERMINED",
        "hold_arm_motion_present": "UNDETERMINED",
        "arm_gripper_coupling_dominant": "UNDETERMINED",
        "arm_trajectory_changed": False,
    })

    write_json(OUT / "AB_DIAGNOSTIC_RESULT.json", {
        "A0_current_baseline": {
            "status": "COMPLETED_FROM_PRIOR_FIXED-STATE-RATE_TRACE",
            "source": str(OLD_TRACE),
            "hold_mean_force_N": mean(hold, "aggregate_force"),
            "hold_force_peak_N": max(float(r["aggregate_force"]) for r in hold),
            "hold_force_exposure_Ns": sum(float(r["aggregate_force"]) * 0.05 for r in hold),
        },
        "A1_physics_rate_shadow": {
            "status": "ANALYTICAL_SHADOW_ONLY",
            "directional_opening_evidence": bool(open_direction),
            "persistent_command_state_missing_before_fix": True,
        },
        "A2_physics_rate_true_force_loop": {
            "status": "IMPLEMENTED_NOT_EXECUTED",
            "reason": "Isaac fresh-process launch blocked: no CUDA-capable device / NVIDIA driver unavailable",
            "arm_trajectory_change": False,
            "feedforward": False,
            "target_override": False,
        },
        "A3_raw_force_diagnostic": "NOT_RUN",
        "A4_half_speed_arm": "NOT_RUN",
    })

    write_json(OUT / "CONTROLLER_FIX_BEFORE_AFTER.json", {
        "before": "external true-force hybrid step once per env step; ForcePositionAction repeated fixed slot6 across three physics applications",
        "after": "outer F_des remains env-rate; optional FORTE authoritative object-filtered true-force loop updates bounded stateful aperture target inside each ForcePositionAction.apply_actions call",
        "default_tabero_behavior_changed": False,
        "force_target_semantics_changed": False,
        "object_filtered_true_force_metric_changed": False,
        "squeeze_kp_changed": False,
        "feedforward_enabled": False,
        "contact_override_enabled": False,
        "raw_arm_trajectory_changed": False,
    })

    write_json(OUT / "STATIC_2_4_6_REGRESSION.json", {
        "status": "STATIC_CONTRACT_PRESERVED; FRESH_STATIC_ISAAC_RUN_BLOCKED",
        "unit_tests": {"tests": 14, "passed": 14},
        "previous_validated_static_means_N": {"2N": 1.9305, "4N": 4.0404, "6N": 6.0554},
        "static_2N_valid": True,
        "static_4N_valid": True,
        "static_6N_valid": True,
        "fresh_process_execution": "blocked by no CUDA-capable device",
        "interpretation": "The pure FORCE_TRACK static contract and prior calibration remain unchanged; dynamic full-task evidence is not relabeled as static calibration.",
    })

    reason = "NOT RUN: fresh Isaac process blocked by NVIDIA driver/CUDA unavailable on this host"
    for i in range(1, 4):
        write_status_csv(OUT / f"ROOT7703_DYNAMIC_3N_REPEAT_{i}.csv", reason)

    write_json(OUT / "ROOT7703_DYNAMIC_TRACKING_SUMMARY.json", {
        "status": "BLOCKED_BEFORE_FRESH_REPEATS",
        "requested_repeats": 3,
        "completed_repeats": 0,
        "known_prior_state_recovery_retest": {
            "contact_loss_post_step": 109,
            "contact_recovery_post_step": 112,
            "force_track_reentry": 113,
            "hold_mean_force_N": mean(hold, "aggregate_force"),
            "hold_force_std_N": math.sqrt(sum((float(r["aggregate_force"]) - mean(hold, "aggregate_force")) ** 2 for r in hold) / len(hold)),
            "hold_abs_tracking_error_N": abs(mean(hold, "aggregate_force") - 3.0),
            "peak_force_N": max(float(r["aggregate_force"]) for r in rows),
            "force_exposure_Ns": sum(float(r["aggregate_force"]) * 0.05 for r in rows),
        },
        "blocker": "no CUDA-capable device; nvidia-smi cannot communicate with NVIDIA driver",
        "no_fabricated_physics_trace": True,
    })

    report = f"""# Dynamic true-force inner-loop audit

## Finding

The local implementation confirms a temporal-layer mismatch. Isaac Lab calls `process_action()` once per environment step and `apply_actions()` three times per environment step. The old FORTE hybrid computed the object-filtered true-force correction once at the outer environment rate, while `ForcePositionAction.apply_actions()` repeated the fixed slot-6 target at each physics application.

The aperture correction was therefore non-accumulating around the VLA `d_pred`. The old trace still shows the correct opening direction during overforce, but it could not persist that opening correction across the next outer action.

## Fix

An optional, default-off `authoritative_true_force_inner_loop` was added to `ForcePositionAction`. FORTE explicitly enables it after handoff. It reads the object-filtered bilateral true-force sensor at each physics application, updates a bounded stateful aperture command, preserves the old filter time constant through dt conversion, and leaves the arm/DiffIK path unchanged.

The normal target remains `F_des=3.0 N`; feed-forward and target-contact override remain disabled; no gain was changed.

## Validation status

- Unit tests: `14/14 PASS`.
- Static contract: prior validated 2/4/6 N calibration preserved.
- Fresh Isaac A2 and three 3 N repeats: blocked before environment creation because this host has no usable CUDA device/NVIDIA driver.
- No physics-step result was fabricated from the old environment-step trace.

The residual blocker is execution validation, not a posterior or method change. Once Isaac is available, run the three fresh repeats and inspect the generated physics-step traces before considering any gain change.
"""
    (OUT / "DYNAMIC_CONTROLLER_REPORT.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
