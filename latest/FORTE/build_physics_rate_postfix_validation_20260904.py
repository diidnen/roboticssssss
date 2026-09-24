#!/usr/bin/env python3
"""Build the evidence-backed post-fix true-force controller validation bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
OUT = ROOT / "analysis/results/physics_rate_true_force_postfix_validation_20260904"
RUNS = OUT / "runs"
ACTION = TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
RUNNER = ROOT / "root7703_probe_activeforcing_smoke_20260904.py"
OLD_COLLECTOR = ROOT / "gnp_style_continuous_collect.py"
OLD_RUNNER = TABERO / "analysis/p5s0c_paired_boundary_probe_value.py"
OLD_MANIFEST = ROOT / "gnp_style_continuous_20260830_125107/collection_long2/CONTINUOUS_STRICT_PREPROBE_TARGET_MANIFEST.json"
OLD_DATA = ROOT / "gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv"
DYNAMIC_RUN = RUNS / "dynamic_3N_guard_diagnostic"


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def boolean(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def stats(selected: list[dict[str, str]], target: float = 3.0) -> dict[str, object]:
    force = [number(row, "F_meas_raw") for row in selected]
    filtered = [number(row, "F_meas_filtered") for row in selected]
    n_top = max(1, math.ceil(0.05 * len(force)))
    return {
        "physics_rows": len(selected),
        "bilateral_rows": sum(boolean(row, "bilateral_contact") for row in selected),
        "mean_raw_force_N": statistics.fmean(force),
        "mean_filtered_force_N": statistics.fmean(filtered),
        "raw_force_std_N": statistics.pstdev(force),
        "raw_force_peak_N": max(force),
        "raw_force_top5pct_mean_N": statistics.fmean(sorted(force)[-n_top:]),
        "mean_abs_tracking_error_N": statistics.fmean(abs(value - target) for value in force),
        "force_exposure_Ns": sum(force) / 60.0,
        "mean_abs_d_cmd_actual_gap_m": statistics.fmean(
            abs(number(row, "d_joint_target") - number(row, "d_actual_before"))
            for row in selected
        ),
        "rate_limit_rows": sum(boolean(row, "rate_limit_active") for row in selected),
        "joint_limit_rows": sum(boolean(row, "joint_limit_active") for row in selected),
        "raw_open_guard_rows": sum(boolean(row, "raw_open_guard_active") for row in selected),
    }


OUT.mkdir(parents=True, exist_ok=True)

# Preserve exact physical evidence under the filenames required by the protocol.
for force in (2, 3, 4, 6):
    source = RUNS / f"final_static_{force}N/PHYSICS_STEP_SIGNAL_TRACE.csv"
    shutil.copyfile(source, OUT / f"POST_FIX_STATIC_{force}N_TRACE.csv")
shutil.copyfile(DYNAMIC_RUN / "PHYSICS_STEP_SIGNAL_TRACE.csv", OUT / "ROOT7703_DYNAMIC_3N_DIAGNOSTIC_TRACE.csv")

# Formal repeats were gated on a passing diagnostic.  Keep schema-only files rather
# than fabricating or relabeling exploratory diagnostics as frozen repeats.
dynamic_header = (DYNAMIC_RUN / "PHYSICS_STEP_SIGNAL_TRACE.csv").read_text(encoding="utf-8").splitlines()[0]
for repeat in (1, 2, 3):
    (OUT / f"ROOT7703_DYNAMIC_3N_REPEAT_{repeat}.csv").write_text(dynamic_header + "\n", encoding="utf-8")

gpu = {
    "status": "PASS",
    "gpu_available": True,
    "nvidia_driver_available": True,
    "driver_version": "580.173.02",
    "device": "NVIDIA A100-SXM4-40GB",
    "cuda_driver_version": "13.0",
    "pytorch": "2.7.0+cu128 (IsaacLab environment)",
    "torch_cuda_available": True,
    "isaac_fresh_process_available": True,
    "fresh_rollouts_completed": ["static_2N", "static_3N", "static_4N", "static_6N", "dynamic_3N_diagnostic"],
    "unrelated_policy_servers_left_untouched": True,
    "known_teardown_issue": "All result/trace files are flushed before a camera weak-reference teardown exception; Isaac exits 134 after completed rollout artifacts.",
}
dump("GPU_ENVIRONMENT_AUDIT.json", gpu)

temporal = {
    "status": "PASS",
    "old_force_correction_formula": "d_cmd = d_pred - 0.0002 * (F_des - F_meas), once per 20 Hz env step",
    "new_force_correction_formula": "d_force_cmd[t+1] = d_force_cmd[t] - 0.004 m/(N*s) * (F_des - F_meas_filtered) * physics_dt",
    "correction_multiplied_by_physics_dt": True,
    "old_update_hz": 20.0,
    "new_update_hz": 60.0,
    "old_effective_correction_per_second_per_newton_m": 0.004,
    "new_effective_correction_per_second_per_newton_m": 0.004,
    "temporal_gain_preserved": True,
    "parameter_dimension": "metres aperture per Newton-second",
    "physics_dt_s": 1.0 / 60.0,
    "env_dt_s": 0.05,
    "open_rate_limit_mps": 0.0024,
    "close_rate_limit_mps": 0.0030,
    "contact_recovery_uses_same_dt_semantics": True,
    "evidence": {"old_external_hybrid": str(TABERO / "analysis/tabero_true_physical_force_hybrid.py"), "new_action": str(ACTION)},
}
dump("TEMPORAL_GAIN_AUDIT.json", temporal)

feature = {
    "status": "PASS",
    "feature_flag": "ForcePositionActionCfg.authoritative_true_force_inner_loop_enabled",
    "default": False,
    "forte_runtime_value": True,
    "tabero_native_runtime_value": False,
    "activation": "explicit term.enable_authoritative_true_force_inner_loop(F_des, sensor_name=contact_grasp_<object>)",
    "native_branch_unchanged": True,
    "tabero_baseline_behavior_changed": False,
    "action_source_sha256": sha(ACTION),
    "runner_sha256": sha(RUNNER),
}
dump("FEATURE_FLAG_ISOLATION_AUDIT.json", feature)

lifecycle = {
    "status": "PASS",
    "initialization_source": "last applied gripper actuator target (_last_d_cmd) at handoff",
    "reset_on_episode_reset": True,
    "reset_on_new_env_action": False,
    "reinitialized_on_contact_recovery": "min(previous bounded command, current actual aperture), then fresh raw force/filter",
    "persists_across_physics_steps": True,
    "persists_across_env_steps": True,
    "vla_slot6_can_overwrite": False,
    "aperture_clamp": [0.0, 0.04],
    "anti_windup": "state is assigned the clamped/rate-limited command; contact loss has a 0.15 mm cumulative reacquire budget",
    "episode_cross_contamination": False,
}
dump("STATEFUL_COMMAND_LIFECYCLE_AUDIT.json", lifecycle)

dump("CONTROLLER_UNIT_TESTS.json", {
    "status": "PASS",
    "total": 24,
    "passed": 24,
    "physics_rate_tests": "10/10 PASS",
    "contact_recovery_tests": "14/14 PASS",
    "temporal_gain_unit_test": "PASS",
    "coverage": ["direction", "20-to-60Hz scaling", "stateful accumulation", "feature default-off", "reset", "handoff initialization", "contact recovery ordering", "raw opening guard"],
    "command": "python3 -m unittest analysis.test_physics_rate_true_force_inner_loop analysis.test_tabero_true_physical_force_hybrid",
})

static_results: dict[str, object] = {}
for force in (2, 3, 4, 6):
    trace = rows(RUNS / f"final_static_{force}N/PHYSICS_STEP_SIGNAL_TRACE.csv")
    window = trace[-90:]
    metric = stats(window, float(force))
    metric.update({
        "requested_force_N": float(force),
        "trace_rows_total": len(trace),
        "all_rows_bilateral": all(boolean(row, "bilateral_contact") for row in trace),
        "whole_trace_peak_N": max(number(row, "F_meas_raw") for row in trace),
        "valid": bool(abs(metric["mean_raw_force_N"] - force) <= 1.0 and metric["bilateral_rows"] == len(window)),
        "evaluation_window": "last 90 physics ticks (1.5 s)",
        "fresh_process": True,
        "teardown_exit_134_after_artifacts": True,
    })
    static_results[f"{force}N"] = metric
dump("POST_FIX_STATIC_REGRESSION.json", {
    "status": "PASS" if all(item["valid"] for item in static_results.values()) else "FAIL",
    "force_realization_tolerance_N": 1.0,
    "results": static_results,
})

dynamic = rows(DYNAMIC_RUN / "PHYSICS_STEP_SIGNAL_TRACE.csv")
events: list[dict[str, object]] = []
previous = True
for row in dynamic:
    current = boolean(row, "bilateral_contact")
    if current != previous:
        events.append({
            "event": "CONTACT_RECOVERY" if current else "CONTACT_LOSS",
            "env_step": int(number(row, "env_step")),
            "physics_step": int(number(row, "physics_step")),
            "raw_force_N": number(row, "F_meas_raw"),
            "filtered_force_N": number(row, "F_meas_filtered"),
        })
    previous = current
first_loss = next(item for item in events if item["event"] == "CONTACT_LOSS")
first_recovery = next(item for item in events if item["event"] == "CONTACT_RECOVERY")
second_loss = next((item for item in events[events.index(first_recovery) + 1:] if item["event"] == "CONTACT_LOSS"), None)
first_lift_env = 118
phases = {
    "handoff_target_switch": (96, 99),
    "pre_contact_loss": (100, int(first_loss["env_step"]) - 1),
    "contact_loss_reacquire": (int(first_loss["env_step"]), int(first_recovery["env_step"]) - 1),
    "recovery_transient": (int(first_recovery["env_step"]), first_lift_env - 1),
    "lift": (first_lift_env, first_lift_env),
    "hold_30_env_steps": (first_lift_env + 1, first_lift_env + 30),
    "post_hold": (first_lift_env + 31, max(int(number(row, "env_step")) for row in dynamic)),
}
phase_metrics = {}
for name, (start, stop) in phases.items():
    selected = [row for row in dynamic if start <= int(number(row, "env_step")) <= stop]
    phase_metrics[name] = {"env_step_range": [start, stop], **stats(selected)}

post_recovery_before_second_loss = [
    row for row in dynamic
    if int(first_recovery["env_step"]) <= int(number(row, "env_step")) < int(second_loss["env_step"])
]
overforce = [row for row in post_recovery_before_second_loss if number(row, "F_meas_filtered") > 3.25]
command_start = number(post_recovery_before_second_loss[0], "d_previous_command")
command_end = number(post_recovery_before_second_loss[-1], "d_command_clamped")
actual_start = number(post_recovery_before_second_loss[0], "d_actual_before")
actual_end = number(post_recovery_before_second_loss[-1], "d_actual_before")

# 20 Hz EE motion is the finest arm signal available from this diagnostic.
env_rows = rows(DYNAMIC_RUN / "ROOT7703_CONTROLLER_VALIDATION_ENV_TRACE.csv")
ee_samples: list[tuple[int, list[float]]] = []
for row in env_rows:
    if row.get("actual_ee_pose"):
        ee_samples.append((int(number(row, "env_step")), json.loads(row["actual_ee_pose"])[:3]))
hold_speed = []
for (step0, pos0), (step1, pos1) in zip(ee_samples, ee_samples[1:]):
    if first_lift_env + 1 <= step1 <= first_lift_env + 30:
        hold_speed.append(math.sqrt(sum((b - a) ** 2 for a, b in zip(pos0, pos1))) / 0.05)

hold = phase_metrics["hold_30_env_steps"]
whole = stats(dynamic)
dynamic_summary = {
    "status": "FAIL",
    "diagnostic_run": str(DYNAMIC_RUN),
    "fresh_process": True,
    "exact_hdf5_initial_state": True,
    "matched_warm_replay": True,
    "step95_snapshot_loaded": False,
    "requested_force_N": 3.0,
    "posterior_involved": False,
    "raw_arm_trajectory_changed": False,
    "events": events,
    "first_contact_loss": first_loss,
    "first_contact_recovery": first_recovery,
    "force_track_reentry": first_recovery,
    "second_contact_loss": second_loss,
    "phase_metrics": phase_metrics,
    "whole_trace": whole,
    "post_recovery_correction_direction_valid": bool(overforce and all(number(row, "delta_d_after_rate_limit") > 0 for row in overforce)),
    "post_recovery_correction_accumulates": command_end > command_start,
    "post_recovery_d_cmd_change_m": command_end - command_start,
    "post_recovery_d_actual_change_m": actual_end - actual_start,
    "post_recovery_d_actual_follows_d_cmd": False,
    "post_recovery_d_actual_follows_explanation": (
        f"d_force_cmd opened by {(command_end-command_start)*1000:.3f} mm while measured "
        f"joint mean moved {(actual_end-actual_start)*1000:.3f} mm before second loss; "
        "command/actual gap narrowed but physical aperture did not follow the commanded "
        "opening direction within the available window."
    ),
    "actuator_saturation_detected": False,
    "controller_rate_limit_active_fraction_post_recovery": sum(boolean(row, "rate_limit_active") for row in post_recovery_before_second_loss) / len(post_recovery_before_second_loss),
    "joint_limit_rows_post_recovery": sum(boolean(row, "joint_limit_active") for row in post_recovery_before_second_loss),
    "hold_ee_speed_mean_mps": statistics.fmean(hold_speed),
    "hold_ee_speed_max_mps": max(hold_speed),
    "hold_arm_motion_present": True,
    "filter_latency_dominant": False,
    "filter_latency_evidence": "EMA lag contributes near force crossing, but the separate alpha=1 raw-force A3 diagnostic also lost contact (env 136), so filter latency is not sufficient to explain failure.",
    "root_cause_if_still_fails": "COMBINATION: D_CMD_RATE_LIMIT + D_ACTUAL_TRACKING_FAILURE + ARM_MOTION_FORCE_COUPLING + REPEATED_CONTACT_LOSS; 3N is also below the established root7703 lift+hold frontier (3.5N,4.0N].",
    "tracking_valid": False,
    "formal_repeat_gate": "NOT_MET",
    "formal_repeat_count": 0,
    "formal_repeat_reason": "The frozen-config three-repeat stage is conditional on a passing diagnostic. This diagnostic re-lost bilateral contact at env 135 and failed 30-step force tracking.",
    "task_outcome": "lift then drop; 30-step hold failed",
    "teardown_exit_134_after_artifacts": True,
}
dump("ROOT7703_DYNAMIC_3N_SUMMARY.json", dynamic_summary)

dump("FILTER_LATENCY_AUDIT.json", {
    "status": "CONTRIBUTOR_NOT_DOMINANT",
    "reference_alpha_at_20hz": 0.5,
    "physics_rate_alpha": 1.0 - 0.5 ** (1.0 / 3.0),
    "effective_time_constant_s": -0.05 / math.log(0.5),
    "measurement_age_s": 1.0 / 60.0,
    "raw_force_A3_diagnostic": str(RUNS / "dynamic_3N_rawforce_A3"),
    "raw_force_A3_second_contact_loss_env_step": 136,
    "conclusion": "Filter lag exists around rapid unloading but removing it did not prevent repeated contact loss.",
})
dump("ACTUATOR_TRACKING_AUDIT.json", {
    "status": "FAIL_DYNAMIC_FOLLOWING",
    "d_cmd_opening_valid": True,
    "d_actual_follows_d_cmd": False,
    "d_cmd_change_before_second_loss_m": command_end - command_start,
    "d_actual_change_before_second_loss_m": actual_end - actual_start,
    "actuator_saturation_detected": False,
    "joint_limit_detected": False,
    "configured_gripper_stiffness": 2000.0,
    "configured_gripper_damping": 100.0,
    "actuator_config_source": str(TABERO / "source/tac_manip/tac_manip/assets/robots/franka.py"),
    "interpretation": "Not a hard joint-limit saturation. The position-controlled gripper did not realize the commanded opening direction before contact was crossed under moving-arm dynamics.",
})
dump("ARM_FORCE_COUPLING_AUDIT.json", {
    "status": "PRESENT_NOT_ISOLATED_AS_SOLE_CAUSE",
    "hold_arm_motion_present": True,
    "hold_ee_speed_mean_mps": statistics.fmean(hold_speed),
    "hold_ee_speed_max_mps": max(hold_speed),
    "half_speed_diagnostic_run": False,
    "dominant": "UNKNOWN",
    "conclusion": "The so-called hold outcome window is not a stationary-arm interval; the frozen VLA EE continues moving. Coupling is material, but no half-speed intervention was run because this validation stopped at the failed diagnostic gate.",
})

dump("APERTURE_COMMAND_CHAIN_AUDIT.json", {
    "status": "STRUCTURE_PASS_PHYSICAL_RESPONSE_FAIL",
    "correction_reference": "stateful d_force_cmd initialized from last applied gripper target",
    "force_correction_accumulates": True,
    "vla_d_pred_overwrites_force_correction": False,
    "post_recovery_overforce_rows": len(overforce),
    "post_recovery_correct_opening_rows": sum(number(row, "delta_d_after_rate_limit") > 0 for row in overforce),
    "d_cmd_opening_valid": True,
    "d_actual_follows_d_cmd": False,
    "rate_limit_fraction": dynamic_summary["controller_rate_limit_active_fraction_post_recovery"],
    "actuator_saturation_detected": False,
})

dump("CONTROL_TIMING_AUDIT.json", {
    "status": "PASS",
    "true_force_loop_temporal_layer": "MIXED",
    "physics_frequency_hz": 60.0,
    "env_step_frequency_hz": 20.0,
    "true_force_feedback_frequency_hz": 60.0,
    "force_sensor_effective_frequency_hz": 60.0,
    "force_measurement_age_s": [0.0, 1.0 / 60.0],
    "f_des_update_layer": "ENV_STEP",
    "force_sensor_read_layer": "PHYSICS_STEP",
    "force_filter_update_layer": "PHYSICS_STEP",
    "d_force_cmd_update_layer": "PHYSICS_STEP",
    "gripper_target_apply_layer": "PHYSICS_STEP",
    "arm_diffik_apply_layer": "PHYSICS_STEP",
    "f_des_held_across_physics_substeps": True,
})

old_manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
dump("OLD720_CONTROLLER_PROVENANCE_AUDIT.json", {
    "status": "COMPLETE_READ_ONLY",
    "implementation": "OTHER: layered P5S0C env-step stateful aperture servo around native Tabero ForcePositionAction",
    "controller_commit": "80ab3be09ce884f86cfc2037d3af30bc28061426",
    "commit_qualification": "The runner blob at this commit exactly matches the SHA-256 recorded in the old720 collection target manifest.",
    "recorded_runner_sha256": old_manifest["runner_sha256"],
    "commit_runner_sha256": "0c029e02544e7d22fbba6f253e29584db6cf796dfabf471e66f157ac66b283b2",
    "collector": str(OLD_COLLECTOR),
    "runner": str(OLD_RUNNER),
    "dataset": str(OLD_DATA),
    "force_loop_update_hz": {"native_ForcePositionAction_apply": 60.0, "P5S0C_stateful_d_pred_servo": 20.0},
    "correction_stateful": True,
    "outer_servo": "d_pred[t+1] = d_pred[t] +/- 0.0006 m when |F_des-F_meas| > 0.4N",
    "contact_recovery_semantics": "No FORTE external CONTACT_LOSS latch; native ForcePositionAction and stateful P5S0C d_pred continue each step.",
    "p5s0c_final_v2_enabled": False,
    "legacy_native_feedforward_default": 0.9,
    "affected_by_current_dynamic_controller_bug": "NO",
    "caveat": "old720 learns success under this historical commanded-force execution mapping. It is not direct evidence for the new object-filtered true-force inner-loop mapping; targeted realized-force revalidation remains appropriate before mixing execution domains.",
})

report = f"""# Physics-rate true-force controller post-fix validation

## Decision

**FINAL_STATUS: POST_FIX_PHYSICAL_VALIDATION_FAIL_DYNAMIC_3N.** The structural change is real and the post-fix static 2/3/4/6N regression passes, but root7703 at 3N re-loses bilateral contact during the 30-step outcome window. The controller therefore cannot yet be frozen as dynamically valid, and the gated three-repeat stage was not run.

## What is validated

- GPU/Isaac execution is available on an NVIDIA A100. Each formal static target ran in a fresh process with exact HDF5 reset and matched warm replay.
- `F_des` remains a 20Hz outer target. Object-filtered bilateral true force, its EMA, `d_force_cmd`, gripper target application, and arm DiffIK all run at 60Hz.
- Temporal gain is rate-normalized: both old 20Hz and new 60Hz contracts equal **0.004 m/(N·s)**. This is not a hidden 3× gain increase.
- The FORTE path is explicitly enabled and defaults off; native Tabero behavior is unchanged.
- Unit tests: **24/24 PASS**, including the existing contact-recovery **14/14 PASS**.
- Last-1.5s static means: **2N={static_results['2N']['mean_raw_force_N']:.3f}N, 3N={static_results['3N']['mean_raw_force_N']:.3f}N, 4N={static_results['4N']['mean_raw_force_N']:.3f}N, 6N={static_results['6N']['mean_raw_force_N']:.3f}N**.

## Dynamic evidence

The fresh root7703 diagnostic used the exact demo_3 state, matched replay, no step95 snapshot, no posterior, fixed `F_des=3N`, and unchanged historical arm actions. Contact was first lost at env/physics **{first_loss['env_step']}/{first_loss['physics_step']}**, recovered and re-entered FORCE_TRACK at **{first_recovery['env_step']}/{first_recovery['physics_step']}**, then was lost again at **{second_loss['env_step']}/{second_loss['physics_step']}**.

After recovery, every filtered-overforce sample generated the correct opening correction and `d_force_cmd` accumulated. However, the command was rate-limited for {dynamic_summary['controller_rate_limit_active_fraction_post_recovery']:.1%} of the pre-second-loss window; commanded aperture opened by {(command_end-command_start)*1000:.3f}mm while measured joint mean moved {(actual_end-actual_start)*1000:.3f}mm. The 30-step window raw-force mean was **{hold['mean_raw_force_N']:.3f}N**, standard deviation **{hold['raw_force_std_N']:.3f}N**, MAE **{hold['mean_abs_tracking_error_N']:.3f}N**, top-5% **{hold['raw_force_top5pct_mean_N']:.3f}N**, peak **{hold['raw_force_peak_N']:.3f}N**, and exposure **{hold['force_exposure_Ns']:.3f}N·s**. It includes zero-force rows after repeated contact loss; the connected portion before the second loss remained over-force.

The arm is not stationary in this outcome window (mean EE speed {statistics.fmean(hold_speed):.3f}m/s). Filter latency contributes, but an alpha=1 raw-force diagnostic also lost contact, so it is not the sole cause. Evidence supports a combination of rate-limited unloading, inadequate physical aperture response, moving-arm coupling, and repeated contact loss. Also, 3N is below root7703's independently established lift+hold frontier `(3.5N, 4.0N]`; controller tuning must not be used to turn an insufficient physical target into a successful grasp.

## Why no three formal repeats

The protocol requires the dynamic diagnostic to pass before freezing structure/parameters and running three repeats. It did not pass. `ROOT7703_DYNAMIC_3N_REPEAT_1/2/3.csv` are intentionally schema-only and contain no fabricated observations. Exploratory rate/filter A/B runs remain under `runs/` and are not relabeled as formal repeats.

## Old720 provenance

old720 did **not** use the buggy FORTE external CONTACT_LOSS state machine. The exact archived runner is commit `80ab3be09ce884f86cfc2037d3af30bc28061426` (its blob hash matches the collection manifest). It used a stateful 20Hz P5S0C `d_pred` servo layered around native 60Hz `ForcePositionAction`, with no external latched CONTACT_LOSS. Therefore `OLD720_AFFECTED_BY_DYNAMIC_CONTROLLER_BUG=NO`, while command-to-realized-force domain mismatch with the new controller remains a separate reuse concern.

## Caveat

Isaac completed and flushed every cited result and trace, then raised a known camera weak-reference exception during teardown (exit 134). This does not alter rollout rows, but it is recorded rather than hidden.
"""
(OUT / "POST_FIX_CONTROLLER_REPORT.md").write_text(report, encoding="utf-8")

print(json.dumps({
    "status": "POST_FIX_PHYSICAL_VALIDATION_FAIL_DYNAMIC_3N",
    "static": {key: value["mean_raw_force_N"] for key, value in static_results.items()},
    "dynamic_hold_mean_N": hold["mean_raw_force_N"],
    "dynamic_formal_repeats": 0,
    "output": str(OUT),
}, indent=2))
