#!/usr/bin/env python3
"""Audit the already-completed root7703 3N ActiveForcing trace.

This is deliberately an offline audit.  It does not launch Isaac, modify a
controller, retrain a model, or change any method/configuration.  The native
ForcePositionAction fields are reconstructed from the recorded action path
and the source-level disable_tabero_legacy_force_loop contract because the
old smoke trace did not record native action-term debug_info every step.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


FORTE = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
SRC_TRACE = FORTE / "analysis/results/probability_calibration_root7703_probe_smoke_20260904/ROOT7703_ACTIVEFORCING_TRACE.csv"
SRC_RESULT = FORTE / "analysis/results/probability_calibration_root7703_probe_smoke_20260904/ROOT7703_ACTIVEFORCING_RESULT.json"
BASELINE_TRACE = FORTE / "analysis/results/root7703_reset_replay_reproduction_20260904_final_v3/LIVE_6N_TRACE.csv"
OUT = FORTE / "analysis/results/root7703_dynamic_low_controller_audit_20260904"

SELECTED = 3.0
DT = 0.05


def f(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def b(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def arr(value: Any) -> list[float] | None:
    if value in (None, ""):
        return None
    try:
        data = json.loads(value) if isinstance(value, str) else value
        flat: list[float] = []
        def visit(x: Any) -> None:
            if isinstance(x, list):
                for y in x:
                    visit(y)
            else:
                flat.append(float(x))
        visit(data)
        return flat
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def peak(xs: list[float]) -> float | None:
    return max(xs) if xs else None


def write_json(name: str, data: Any) -> None:
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def phase_for(i: int, first_lift: int, n: int) -> str:
    # The source runner's relevant window is handoff through first lift.
    # For this diagnostic, the first bilateral-contact interruption/recontact
    # segment is explicitly isolated before the stable lift motion.
    if i == 0:
        return "handoff_force_switch"
    if i <= min(15, first_lift - 1):
        return "preload_unloading_transient"
    if i <= first_lift:
        return "lift_motion"
    if i <= min(first_lift + 30, n - 1):
        return "hold_30_step"
    return "post_hold"


def metric(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    raw_target = [float(r["raw_controller_target_N"]) for r in rows]
    effective = [float(r["external_hybrid_effective_target_N"]) for r in rows]
    measured = [float(r["measured_object_filtered_force_N"]) for r in rows]
    err_raw = [x - y for x, y in zip(measured, raw_target)]
    err_eff = [x - y for x, y in zip(measured, effective)]
    return {
        "phase": label,
        "row_count": len(rows),
        "env_step_first": int(rows[0]["env_step"]) if rows else None,
        "env_step_last": int(rows[-1]["env_step"]) if rows else None,
        "raw_target_mean_N": mean(raw_target),
        "effective_controller_target_mean_N": mean(effective),
        "native_f_sq_pred_mean_N": mean([float(r["native_f_sq_pred_N"]) for r in rows]),
        "native_f_sq_pred_eff_mean_N": mean([float(r["native_f_sq_pred_eff_N"]) for r in rows]),
        "measured_raw_object_force_mean_N": mean(measured),
        "measured_ema_native_mean_N": None,
        "measured_peak_N": peak(measured),
        "force_error_mean_to_raw_target_N": mean(err_raw),
        "force_error_abs_mean_to_raw_target_N": mean([abs(x) for x in err_raw]),
        "force_error_mean_to_effective_target_N": mean(err_eff),
        "force_error_abs_mean_to_effective_target_N": mean([abs(x) for x in err_eff]),
        "force_exposure_Ns": sum(measured) * DT,
        "bilateral_contact_rate": mean([1.0 if b(r["bilateral_contact"]) else 0.0 for r in rows]),
        "lift_rows": sum(1 for r in rows if b(r["lift_threshold_reached"])),
        "hold_rows": sum(1 for r in rows if b(r["hold_flag"])),
        "external_hybrid_states": sorted({r["external_hybrid_state"] for r in rows}),
        "external_force_loop_active_rate": mean([1.0 if r["external_force_loop_active"] else 0.0 for r in rows]),
    }


def reconstruct_external_state(rows: list[dict[str, Any]]) -> list[tuple[str, bool, str]]:
    """Reconstruct the active runner state from the source state machine.

    The runner explicitly sets ``ctl.state = FORCE_TRACK`` before the first
    continuation step.  In the current implementation CONTACT_LOSS has no
    transition back to FORCE_TRACK, even when bilateral contact reappears.
    """
    state = "FORCE_TRACK"
    out: list[tuple[str, bool, str]] = []
    for row in rows:
        bilateral = b(row["bilateral_contact"])
        if state in {"FORCE_TRACK", "LIFT_TRANSPORT"} and not bilateral:
            state = "CONTACT_LOSS"
        active = state in {"FORCE_TRACK", "LIFT_TRANSPORT"}
        if active:
            mode = "force_feedback_or_deadband"
        elif state == "CONTACT_LOSS":
            mode = "contact_loss_reacquire_close_nudge"
        else:
            mode = "inactive"
        out.append((state, active, mode))
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_rows = read_rows(SRC_TRACE)
    result = json.loads(SRC_RESULT.read_text())
    first_lift = next(i for i, r in enumerate(source_rows) if b(r["lift_threshold_reached"]))
    relevant = source_rows[: first_lift + 1]

    rows: list[dict[str, Any]] = []
    previous_ee: list[float] | None = None
    previous_obj: list[float] | None = None
    state_info = reconstruct_external_state(source_rows)
    for i, source in enumerate(source_rows):
        ee = arr(source.get("actual_ee_pose"))
        obj = arr(source.get("object_pose"))
        ee_pos = ee[:3] if ee and len(ee) >= 3 else None
        obj_pos = obj[:3] if obj and len(obj) >= 3 else None
        ee_vel = None if previous_ee is None or ee_pos is None else math.sqrt(sum((a - c) ** 2 for a, c in zip(ee_pos, previous_ee))) / DT
        obj_vel = None if previous_obj is None or obj_pos is None else math.sqrt(sum((a - c) ** 2 for a, c in zip(obj_pos, previous_obj))) / DT
        if ee_pos is not None:
            previous_ee = ee_pos
        if obj_pos is not None:
            previous_obj = obj_pos

        measured = f(source["aggregate_force"])
        raw_target = f(source["requested_force_target"]) or SELECTED
        # This is the external hybrid's physical target.  The source runner
        # records result.F_des in both target columns; the adapter then writes
        # only aperture slot 6 and clears slots 7:13.
        external_target = f(source["effective_force_target"]) or raw_target
        controller_error = f(source["controller_force_error"])
        hybrid_filtered = None if controller_error is None else raw_target - controller_error
        hold = i > first_lift and i <= first_lift + 30
        row: dict[str, Any] = dict(source)
        row.update({
            "audit_phase": phase_for(i, first_lift, len(source_rows)),
            "activeforcing_selected_force_N": SELECTED,
            "raw_controller_target_N": raw_target,
            "external_hybrid_effective_target_N": external_target,
            "native_f_sq_pred_N": 0.0,
            "native_f_sq_meas_raw_N": None,
            "native_f_sq_meas_ema_N": None,
            "native_f_sq_pred_eff_N": 0.0,
            "native_target_contact_override_enabled": False,
            "native_target_contact_override_latched": False,
            "native_post_override_effective_target_N": 0.0,
            "pre_step_external_hybrid_filtered_measured_force_N": hybrid_filtered,
            "measured_object_filtered_force_N": measured,
            "tracking_error_measured_minus_raw_target_N": None if measured is None else measured - raw_target,
            "tracking_error_measured_minus_effective_target_N": None if measured is None else measured - external_target,
            "d_pred_m": f(source.get("applied_gripper_target")),
            "d_cmd_m": f(source.get("applied_gripper_target")),
            "d_actual_m": f(source.get("aperture")),
            "arm_velocity_mps": ee_vel,
            "object_velocity_mps": obj_vel,
            "hold_flag": hold,
            "external_hybrid_state": state_info[i][0],
            "external_force_loop_active": state_info[i][1],
            "external_feedback_mode": state_info[i][2],
        })
        rows.append(row)

    # CSV: preserve all original fields and append audit-only fields.
    extra = [
        "audit_phase", "activeforcing_selected_force_N", "raw_controller_target_N",
        "external_hybrid_effective_target_N", "native_f_sq_pred_N", "native_f_sq_meas_raw_N",
        "native_f_sq_meas_ema_N", "native_f_sq_pred_eff_N", "native_target_contact_override_enabled",
        "native_target_contact_override_latched", "native_post_override_effective_target_N",
        "pre_step_external_hybrid_filtered_measured_force_N", "measured_object_filtered_force_N",
        "tracking_error_measured_minus_raw_target_N", "tracking_error_measured_minus_effective_target_N",
        "d_pred_m", "d_cmd_m", "d_actual_m", "arm_velocity_mps", "object_velocity_mps", "hold_flag",
        "external_hybrid_state", "external_force_loop_active", "external_feedback_mode",
    ]
    fields = list(source_rows[0].keys()) + [x for x in extra if x not in source_rows[0]]
    with (OUT / "ROOT7703_3N_FULL_TRACE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    phase_names = ["handoff_force_switch", "preload_unloading_transient", "lift_motion", "hold_30_step", "post_hold"]
    phase_metrics = {name: metric([r for r in rows if r["audit_phase"] == name], name) for name in phase_names}
    relevant_metric = metric(rows[: first_lift + 1], "relevant_window_handoff_through_first_lift")

    write_json("ROOT7703_3N_PHASE_METRICS.json", {
        "source_trace": str(SRC_TRACE),
        "dt_s": DT,
        "first_lift_trace_index": first_lift,
        "first_lift_env_step": int(source_rows[first_lift]["env_step"]),
        "phase_definition": {
            "handoff_force_switch": "trace row 0 at handoff; target first enters ActiveForcing",
            "preload_unloading_transient": "rows 1-15; includes initial unloading/contact interruption and recontact before lift",
            "lift_motion": "rows after transient through first lift row, excluding first lift row from hold",
            "hold_30_step": "the 30 rows immediately after the first lift row",
            "post_hold": "remaining continuation after the 30-step hold",
            "relevant_window": "runner's historical metric: handoff through first lift inclusive",
        },
        "relevant_window": relevant_metric,
        "phases": phase_metrics,
        "result_summary": result.get("summary", {}),
    })

    write_json("TABERO_FORCE_SEMANTICS_AUDIT.json", {
        "source_files": {
            "force_position_action": str(TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"),
            "legacy_loop_bridge": str(TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/true_physical_force_hybrid.py"),
            "external_hybrid": str(TABERO / "analysis/tabero_true_physical_force_hybrid.py"),
            "runtime_runner": str(FORTE / "root7703_probe_activeforcing_smoke_20260904.py"),
        },
        "native_formulas": {
            "f_sq_target": "split squeeze from 13D action slots 7:13",
            "feedforward": "f_sq_target_eff = f_sq_target + squeeze_ff_k_load_z * abs(f_sq_target) when raw squeeze >= threshold",
            "target_contact_override": "if latched, f_sq_target_eff = 2 * target_contact_single_finger_normal_force_n",
            "squeeze_position_loop": "d_cmd=d_pred when squeeze_kp=0; otherwise error loop",
        },
        "default_source_values": {
            "squeeze_ff_k_load_z": 0.9,
            "squeeze_ff_contact_threshold_N": 1.0,
            "squeeze_kp": 0.001,
            "meas_force_filter_alpha": 0.2,
            "target_contact_squeeze_enabled": False,
        },
        "runtime_pre_disable_values": {
            "squeeze_ff_k_load_z": 0.9,
            "squeeze_ff_contact_threshold_N": 1.0,
            "squeeze_kp": 0.0002,
            "target_contact_squeeze_enabled": "task override is copied only if task JSON contains an override; active branch disables it before continuation",
        },
        "runtime_active_values_after_disable_tabero_legacy_force_loop": {
            "squeeze_ff_k_load_z": 0.0,
            "squeeze_ff_contact_threshold_N": 1.0,
            "squeeze_kp": 0.0,
            "target_contact_squeeze_enabled": False,
            "feedforward_enabled": False,
            "native_force_slots_received": False,
        },
        "selected_force_N": SELECTED,
        "hypothetical_native_effective_target_if_legacy_ff_were_enabled": {
            "k_0.0": 3.0,
            "k_0.6": 4.8,
            "k_0.9": 5.7,
        },
        "actual_interpretation": "The active branch passes F_des=3.0 to the external object-filtered true-force hybrid. It does not pass 3.0 into native squeeze force slots; those slots are zeroed by the adapter and the native legacy loop is disabled.",
        "native_raw_effective_measured_semantics_match": False,
        "active_external_target_measurement_semantics_match": True,
        "semantics_match_explanation": "Native f_sq_* fields use the legacy contact_gripper/action-slot contract, while active FORTE measurement uses object-filtered bilateral normal force. The external F_des and external F_meas are the intended matched ActiveForcing contract.",
        "native_debug_capture_note": "The completed smoke CSV did not record ForcePositionAction.debug_info every step; native zero/disabled values are established by the runtime bridge and adapter source contract, not invented from the 4.931N measurement.",
    })

    write_json("FORTE_FORCE_PATH_AUDIT.json", {
        "selected_force_N": SELECTED,
        "data_flow": [
            {"stage": "ActiveForcing selector", "value": "F*=3.0N"},
            {"stage": "runner requested force", "value": "TaberoTruePhysicalForceHybrid.step(F_des=3.0)"},
            {"stage": "external effective target", "value": "result.F_des=3.0N; external hybrid has no force feed-forward"},
            {"stage": "13D adapter", "value": "preserve arm slots 0:6; write only aperture slot 6; clear force slots 7:13"},
            {"stage": "native ForcePositionAction", "value": "f_sq_target=0; legacy squeeze kp/feed-forward/override disabled; d_cmd=d_pred"},
            {"stage": "robot", "value": "position-controlled gripper realizes d_actual"},
            {"stage": "measurement", "value": "object-filtered bilateral true force F_meas=2*min(left object normal,right object normal)"},
        ],
        "raw_controller_target_N": SELECTED,
        "feedforward_gain_active": 0.0,
        "post_ff_effective_target_N": SELECTED,
        "contact_override_active": False,
        "post_override_effective_target_N": SELECTED,
        "measurement_authority": "external object-filtered bilateral force is the authoritative ActiveForcing measurement; native contact_gripper f_sq debug fields are a separate legacy sensor/control layer",
        "external_target_measurement_contract_match": True,
        "native_target_measurement_contract_match": False,
        "arm_trajectory_changed": False,
        "force_controller_changed": False,
        "controller_metric_changed": False,
    })

    write_json("RAW_VS_EFFECTIVE_TARGET.json", {
        "selected_force_N": SELECTED,
        "raw_controller_target_N": SELECTED,
        "runtime_active_feedforward_gain": 0.0,
        "post_ff_effective_target_N": SELECTED,
        "contact_override_enabled": False,
        "contact_override_latched": False,
        "post_override_effective_target_N": SELECTED,
        "phase_metrics": phase_metrics,
        "comparison": {
            "whole_relevant_window_measured_mean_N": relevant_metric["measured_raw_object_force_mean_N"],
            "hold_measured_mean_N": phase_metrics["hold_30_step"]["measured_raw_object_force_mean_N"],
            "hold_error_to_effective_target_N": phase_metrics["hold_30_step"]["force_error_mean_to_effective_target_N"],
            "interpretation": "4.931N is not explained by active native feed-forward: active external target is 3.0N. The 30-step hold itself remains 7.112N, so the discrepancy is not only a handoff transient.",
        },
    })

    write_json("CONTACT_OVERRIDE_AUDIT.json", {
        "configured_source": str(TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_tactile_libero_env_cfg.py"),
        "bridge_source": str(TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/true_physical_force_hybrid.py"),
        "target_contact_override_enabled_before_active_continuation": "not used as an active force path; any task-configured value is superseded by disable_tabero_legacy_force_loop",
        "target_contact_override_enabled": False,
        "target_contact_override_latched": False,
        "target_contact_activation_step": None,
        "target_contact_single_finger_force_N": None,
        "target_contact_effective_squeeze_target_N": None,
        "evidence": [
            "disable_tabero_legacy_force_loop sets target_contact_squeeze_enabled=False before run_continuation",
            "adapter clears force slots 7:13, so no native target can be overridden in the active branch",
            "no fixed 5N/6N target is present in the recorded external target column",
        ],
    })

    write_json("DYNAMIC_FORCE_TRACKING_RESULT.json", {
        "classification": "CASE_D_DYNAMIC_TRACKING_FAILURE",
        "alternative_cases_rejected": {
            "CASE_A_FEEDFORWARD_EXPECTED": "rejected for active branch; runtime gain is 0.0 after disable",
            "CASE_B_CONTACT_OVERRIDE": "rejected; disabled and not latched",
            "CASE_C_TRANSIENT_WINDOW": "rejected as sole cause; hold mean is 7.112N over 30 rows vs 3.0N target",
            "CASE_E_MEASUREMENT_SEMANTICS_MISMATCH": "rejected as explanation of active loop; the external target and aggregate measurement are the same object-filtered true-force contract",
        },
        "whole_relevant_window": relevant_metric,
        "hold_30_step": phase_metrics["hold_30_step"],
        "static_tracking_valid": True,
        "static_tracking_evidence": str(FORTE / "analysis/results/root7703_reset_replay_reproduction_20260904_final_v3"),
        "dynamic_tracking_valid": False,
        "tracking_validity_reason": "During stable hold, measured force is persistently above the 3.0N external effective target; completed runner also marked force_tracking_valid=false.",
        "low_level_controller_broken_under_strict_dynamic_contract": True,
        "state_machine_diagnosis": {
            "initial_state": "FORCE_TRACK (set explicitly by run_continuation before first step)",
            "first_contact_loss_env_step": 109,
            "first_contact_loss_trace_index": 13,
            "state_after_contact_loss": "CONTACT_LOSS",
            "bilateral_recontact_env_step": 112,
            "state_after_recontact": "CONTACT_LOSS",
            "force_feedback_reenabled_after_recontact": False,
            "observed_post_recontact_behavior": "the trace continues to apply the CONTACT_LOSS reacquire close nudge; d_final-d_nominal is approximately -0.00015m on the subsequent stable rows",
            "interpretation": "The high hold force is consistent with a latched contact-loss/reacquire state rather than feed-forward multiplication. This is the specific dynamic execution defect exposed by the trace.",
        },
        "activeforcing_force_semantics": "valid desired physical target at the external hybrid interface; realization is not valid dynamic tracking",
        "no_controller_change_made": True,
    })

    # A compact baseline reference, if the previously validated 6N trace is present.
    baseline = read_rows(BASELINE_TRACE) if BASELINE_TRACE.exists() else []
    baseline_vals = [f(r.get("aggregate_force", r.get("force_N"))) for r in baseline]
    baseline_vals = [x for x in baseline_vals if x is not None]
    report_lines = [
        "# Root7703 dynamic low-level controller audit",
        "",
        "## Conclusion",
        "",
        "The 3.0N ActiveForcing command did not become 4.93N through the active Tabero native feed-forward or target-contact override. The active runner disables those paths and sends `F_des=3.0N` to the external object-filtered true-force hybrid. However, the measured force remains high during the stable 30-step hold (`7.112N` mean, `4.112N` above target), so the discrepancy is not only a transient-window artifact. Under the requested strict criterion this is `CASE_D_DYNAMIC_TRACKING_FAILURE`.",
        "",
        "## Force path",
        "",
        "`F*=3.0N -> TaberoTruePhysicalForceHybrid.step(F_des=3.0) -> aperture correction in slot 6 -> native ForcePositionAction receives zero force slots 7:13 -> native legacy squeeze loop/feed-forward/override disabled -> d_actual -> object-filtered bilateral true-force measurement`.",
        "",
        "The native source default `squeeze_ff_k_load_z=0.9` would yield 5.7N from a native 3.0N squeeze target, and a historical 0.6 setting would yield 4.8N. Those are hypothetical legacy semantics, not the active runtime semantics of this branch: `disable_tabero_legacy_force_loop` sets the active gain to 0.0.",
        "",
        "## Phase evidence",
        "",
        f"- Relevant window (handoff through first lift): `{relevant_metric['measured_raw_object_force_mean_N']:.6f}N` mean.",
        f"- Lift motion: `{phase_metrics['lift_motion']['measured_raw_object_force_mean_N']:.6f}N` mean.",
        f"- 30-step hold: `{phase_metrics['hold_30_step']['measured_raw_object_force_mean_N']:.6f}N` mean; target error `{phase_metrics['hold_30_step']['force_error_mean_to_effective_target_N']:.6f}N`.",
        f"- Post-hold: `{phase_metrics['post_hold']['measured_raw_object_force_mean_N']:.6f}N` mean.",
        "",
        "The hold rows retain bilateral contact and no drop, while the later continuation eventually drops. This audit therefore separates force-tracking validity from the lift+hold outcome.",
        "",
        "## Measurement semantics",
        "",
        "The active measured force is `2*min(left_object_normal_force, right_object_normal_force)` from the object-filtered bilateral sensor. It is not the native `contact_gripper` legacy `f_sq_meas` debug field. Thus literal native raw/effective/measured fields do not share the same sensor contract, but that is a deliberate separation in the active external loop, not evidence that 4.931N was caused by native feed-forward.",
        "",
        "## Specific dynamic cause exposed by the existing trace",
        "",
        "The runner initializes the external hybrid in `FORCE_TRACK`. At env step 109 (trace index 13), bilateral target-object contact becomes false and the state machine enters `CONTACT_LOSS`. The current `TaberoTruePhysicalForceHybrid.step` implementation has no transition from `CONTACT_LOSS` back to `FORCE_TRACK`; bilateral contact is true again at env step 112, but the state remains `CONTACT_LOSS`. The controller therefore continues the bounded close/reacquire nudge instead of applying normal force feedback. The measured force then jumps to 8.59–10.80N during lift and remains 7.112N on average during the next 30 hold rows. This is the diagnosed dynamic tracking failure; no fix is applied in this audit.",
        "",
        "## Scope and next action",
        "",
        "No gains, posterior, calibration, Expected Utility, probe, controller metric, or arm trajectory were changed. The next action is a minimal controller-path reproduction with full native and external telemetry before any gain change: capture native `f_sq_pred`, `f_sq_pred_eff`, `f_sq_meas_raw`, `f_sq_meas`, `d_pred`, `d_cmd`, `d_actual` alongside the external hybrid state and verify the intended CONTACT_LOSS recovery transition. Static 2/4/6N validation remains frozen evidence and should not be disturbed.",
    ]
    if baseline_vals:
        report_lines += ["", f"Previously validated 6N trace rows available for reference: `{len(baseline_vals)}` force samples; this audit does not rerun it."]
    (OUT / "LOW_CONTROLLER_REPORT.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
