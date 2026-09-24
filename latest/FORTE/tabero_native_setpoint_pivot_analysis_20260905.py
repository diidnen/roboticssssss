#!/usr/bin/env python3
"""Finalize the frozen Tabero-native continuous-setpoint pivot audit.

This script is deliberately analysis-only.  It does not launch Isaac, alter the
controller, or select/tune any execution parameter from root7703 outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
OUT = ROOT / "analysis/results/tabero_native_continuous_setpoint_pivot_20260905"
RUNS = OUT / "runs"
LEVELS = (2, 3, 4, 5, 6)
REPEATS = (1, 2, 3)
DT = 0.05
NATIVE_DEADZONE_N = 0.25


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value not in (None, ""):
            values.append(float(value))
    return values


def top_fraction(values: list[float], fraction: float = 0.05) -> float:
    count = max(1, math.ceil(len(values) * fraction))
    return mean(sorted(values, reverse=True)[:count])


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def correlation(xs: list[float], ys: list[float]) -> float:
    xbar, ybar = mean(xs), mean(ys)
    numerator = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    xss = sum((x - xbar) ** 2 for x in xs)
    yss = sum((y - ybar) ** 2 for y in ys)
    if xss == 0.0 or yss == 0.0:
        return float("nan")
    return numerator / math.sqrt(xss * yss)


def spearman(xs: list[float], ys: list[float]) -> float:
    return correlation(average_ranks(xs), average_ranks(ys))


def linear_slope(xs: list[float], ys: list[float]) -> float:
    xbar, ybar = mean(xs), mean(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator


def isotonic_non_decreasing(values: list[float]) -> list[float]:
    blocks = [{"sum": value, "weight": 1, "start": i, "end": i} for i, value in enumerate(values)]
    index = 0
    while index < len(blocks) - 1:
        left = blocks[index]["sum"] / blocks[index]["weight"]
        right = blocks[index + 1]["sum"] / blocks[index + 1]["weight"]
        if left <= right:
            index += 1
            continue
        blocks[index]["sum"] += blocks[index + 1]["sum"]
        blocks[index]["weight"] += blocks[index + 1]["weight"]
        blocks[index]["end"] = blocks[index + 1]["end"]
        blocks.pop(index + 1)
        index = max(0, index - 1)
    fitted = [0.0] * len(values)
    for block in blocks:
        value = block["sum"] / block["weight"]
        for i in range(block["start"], block["end"] + 1):
            fitted[i] = value
    return fitted


def bool_field(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def run_metrics(level: int, repeat: int) -> dict[str, object]:
    run_dir = RUNS / f"setpoint_{level}_repeat_{repeat}"
    result_path = run_dir / "TABERO_NATIVE_DIAGNOSTIC_RESULT.json"
    trace_path = run_dir / "TABERO_NATIVE_DIAGNOSTIC_TRACE.csv"
    result = load_json(result_path)
    rows = load_csv(trace_path)
    summary = result["summary"]
    first_lift = summary.get("first_lift_step")
    lift_index = next((i for i, row in enumerate(rows) if int(float(row["env_step"])) == first_lift), None)
    hold = rows[lift_index + 1 : lift_index + 31] if lift_index is not None else []
    relevant = rows[: lift_index + 31] if lift_index is not None else rows
    true_values = numeric(relevant, "object_filtered_true_squeeze_N")
    hold_true = numeric(hold, "object_filtered_true_squeeze_N")
    native_values = numeric(relevant, "native_measured_squeeze_N")
    effective_values = numeric(relevant, "native_effective_squeeze_target_N")
    arm_unchanged = all(bool_field(row["raw_arm_action_unchanged"]) for row in rows)
    handoff = result["handoff_parity"]
    metric = {
        "setpoint": level,
        "repeat": repeat,
        "status": result.get("status"),
        "fresh_process": result.get("fresh_process"),
        "handoff_parity": bool(
            handoff.get("physical_state_parity")
            and handoff.get("runtime_state_parity")
            and handoff.get("observation_parity")
            and handoff.get("bilateral_contact")
        ),
        "raw_arm_trajectory_changed": not arm_unchanged,
        "raw_arm_hash_at_handoff": handoff.get("raw_arm_action_hash_at_handoff"),
        "raw_arm_hash_next": handoff.get("raw_arm_action_hash_next"),
        "feedforward_k": summary["feedforward_k"],
        "contact_override_enabled": summary["target_contact_override_enabled"],
        "effective_target_mean_N": mean(effective_values),
        "native_measured_mean_N": mean(native_values),
        "true_measured_mean_N": mean(true_values),
        "true_hold_mean_N": mean(hold_true) if hold_true else None,
        "true_hold_std_N": pstdev(hold_true) if hold_true else None,
        "true_peak_N": max(true_values),
        "true_top5_N": top_fraction(true_values),
        "true_force_exposure_Ns": sum(true_values) * DT,
        "first_lift_step": first_lift,
        "hold_rows": len(hold),
        "lift_success": first_lift is not None,
        "hold_success": bool(summary["hold_bilateral_contact"]),
        "lift_hold_success": bool(first_lift is not None and summary["hold_bilateral_contact"]),
        "contact_loss": summary.get("first_target_contact_loss_step") is not None,
        "drop_anywhere_in_continuation": bool(summary["drop"]),
        "evaluation_window": "handoff continuation through end of 30-step post-lift hold",
        "trace_path": str(trace_path),
        "result_path": str(result_path),
    }
    shutil.copyfile(trace_path, OUT / f"TABERO_SETPOINT_{level}_REPEAT_{repeat}.csv")
    return metric


def aggregate(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {"mean": mean(values), "std": pstdev(values), "median": sorted(values)[len(values) // 2]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    current_force_action = TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
    old_runner = Path("/media/volume/newdata/exouser/Tabero_e3lh/analysis/p5s0c_paired_boundary_probe_value.py")
    old_action = Path("/media/volume/newdata/exouser/Tabero_e3lh/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py")

    contract = {
        "status": "FROZEN",
        "custom_precise_force_controller_abandoned": True,
        "final_low_level_controller": "TABERO_NATIVE_FIXED_HYBRID",
        "final_force_semantics": "CONTINUOUS_GRASP_FORCE_SETPOINT",
        "exact_newton_tracking_claim": False,
        "method": "single P4-B probe -> friction posterior -> continuous feasibility posterior -> Expected Utility -> continuous grasp-force setpoint",
        "execution": "native ForcePositionAction; fixed config; measured physical force is authoritative for gentleness",
        "custom_paths": {
            "authoritative_true_force_inner_loop_enabled": False,
            "external_stateful_servo": False,
            "adaptive_release": False,
            "contact_preservation_controller": False,
            "velocity_controller": False,
            "offset_servo": False,
        },
        "unchanged": {
            "activeforcing_method": True,
            "posterior": True,
            "expected_utility": True,
            "probe": True,
            "raw_vla_arm_trajectory": True,
        },
        "mapping_gate": {
            "root": 7703,
            "levels": list(LEVELS),
            "repeats_per_level": len(REPEATS),
            "primary_measured_metric": "object_filtered_true_squeeze_N",
            "evaluation_window": "handoff continuation through end of 30-step post-lift hold",
            "valid_if": "all formal traces valid; rho>=0.8 and positive slope for mean/top5/exposure; no systematic group inversion exceeding native 0.25N deadzone; setpoint 6 exceeds setpoint 2",
        },
    }
    dump_json(OUT / "FINAL_EXECUTION_CONTRACT.json", contract)

    old_audit = {
        "status": "COMPLETE_READ_ONLY_AUDIT",
        "old720_rows": 720,
        "old720_contexts": 72,
        "old720_roots": 24,
        "old720_tasks": [0, 1, 5, 6],
        "controller_implementation": "P5-S0-C outer d_pred force servo feeding native ForcePositionAction",
        "forcepositionaction_version": {
            "exact_collection_action_source_hash": "NOT_ARCHIVED_IN_COLLECTION_MANIFEST",
            "nearest_collection_lineage_commit": "56a3b905 (2026-08-24)",
            "archived_runner_sha256": "0c029e02544e7d22fbba6f253e29584db6cf796dfabf471e66f157ac66b283b2",
            "recovered_runner_sha256": sha256(old_runner),
            "later_archived_action_sha256": sha256(old_action),
            "current_action_sha256": sha256(current_force_action),
        },
        "force_command_path": [
            "requested continuous numeric setpoint",
            "equal per-finger local-z target = setpoint/2 in 13D action",
            "P4-B outer 20Hz _force_servo also updates d_pred by fixed SERVO_STEP=0.0006m with 0.4N deadband",
            "native ForcePositionAction applies filtered squeeze feedback around d_pred at physics rate",
            "Franka position-drive actuator",
        ],
        "feedforward_config": {
            "squeeze_ff_k_load_z": 0.9,
            "squeeze_ff_contact_threshold_N": 1.0,
            "evidence": "old collection did not set P5S0C_FINAL_V2; ForcePositionAction default/config lineage",
        },
        "contact_override_config": {
            "enabled": False,
            "evidence": "task libero_object config did not define target_contact_squeeze_override",
        },
        "execution_compatible_with_final_tabero_native": "PARTIAL",
        "compatibility_basis": [
            "same 13D setpoint encoding and native ForcePositionAction/position-drive path",
            "same fixed feed-forward family (k=0.9) and override-off semantics",
            "old720 additionally updated policy aperture d_pred with an outer fixed-step servo",
            "old720 used historical long-horizon branch trajectories rather than root7703 frozen-VLA continuation",
        ],
        "reusable": "PARTIAL",
        "reuse_scope": "candidate setpoint/outcome supervision under the historical fixed execution contract; do not reinterpret as exact measured Newton labels or as fully matched current-native execution",
    }
    dump_json(OUT / "OLD720_NATIVE_COMPATIBILITY_AUDIT.json", old_audit)

    native_config = {
        "status": "FROZEN_BEFORE_MAPPING_ANALYSIS",
        "selection_basis": "old720-matched native configuration, not root7703 outcome",
        "action": "ForcePositionAction",
        "squeeze_kp_m_per_N": 0.0008,
        "squeeze_deadzone_N": 0.25,
        "meas_force_filter_alpha": 0.2,
        "squeeze_ff_k_load_z": 0.9,
        "squeeze_ff_contact_threshold_N": 1.0,
        "target_contact_squeeze_enabled": False,
        "gripper_stiffness": 2000.0,
        "gripper_damping": 100.0,
        "physics_rate_Hz": 60.0,
        "environment_rate_Hz": 20.0,
        "stateful_command_accumulation": False,
        "d_cmd_reference": "current policy d_pred on each simulation-step apply_actions",
        "authoritative_true_force_inner_loop_enabled": False,
        "custom_controller_paths_enabled": False,
        "force_layers_reported_separately": [
            "raw_setpoint",
            "effective_controller_target",
            "native_measured_squeeze",
            "object_filtered_true_measured_force",
        ],
    }
    dump_json(OUT / "TABERO_NATIVE_CONFIG.json", native_config)

    runs = [run_metrics(level, repeat) for level in LEVELS for repeat in REPEATS]
    write_csv(OUT / "TABERO_NATIVE_RUN_SUMMARY.csv", runs)
    formal_valid = all(
        row["status"] == "COMPLETE"
        and row["fresh_process"] is True
        and row["handoff_parity"] is True
        and row["raw_arm_trajectory_changed"] is False
        and row["lift_success"] is True
        and int(row["hold_rows"]) == 30
        and float(row["feedforward_k"]) == 0.9
        and row["contact_override_enabled"] is False
        for row in runs
    )
    arm_hashes = sorted({(row["raw_arm_hash_at_handoff"], row["raw_arm_hash_next"]) for row in runs})

    groups: dict[int, dict[str, object]] = {}
    for level in LEVELS:
        subset = [row for row in runs if row["setpoint"] == level]
        groups[level] = {
            "setpoint": level,
            "n": len(subset),
            "effective_target_N": aggregate(subset, "effective_target_mean_N"),
            "true_measured_mean_N": aggregate(subset, "true_measured_mean_N"),
            "true_hold_mean_N": aggregate(subset, "true_hold_mean_N"),
            "true_top5_N": aggregate(subset, "true_top5_N"),
            "true_peak_N": aggregate(subset, "true_peak_N"),
            "true_force_exposure_Ns": aggregate(subset, "true_force_exposure_Ns"),
            "native_measured_mean_N": aggregate(subset, "native_measured_mean_N"),
            "lift_hold_success_rate": mean([float(bool(row["lift_hold_success"])) for row in subset]),
            "hold_contact_rate": mean([float(bool(row["hold_success"])) for row in subset]),
            "repeat_values": subset,
        }

    setpoints = [float(row["setpoint"]) for row in runs]
    metric_fields = {
        "measured_mean": "true_measured_mean_N",
        "top5": "true_top5_N",
        "exposure": "true_force_exposure_Ns",
    }
    associations: dict[str, dict[str, object]] = {}
    for name, field in metric_fields.items():
        values = [float(row[field]) for row in runs]
        group_values = [float(groups[level][field]["mean"]) for level in LEVELS]
        associations[name] = {
            "individual_run_spearman": spearman(setpoints, values),
            "group_mean_spearman": spearman([float(level) for level in LEVELS], group_values),
            "linear_slope_per_setpoint": linear_slope(setpoints, values),
            "group_means": group_values,
            "isotonic_group_fit": isotonic_non_decreasing(group_values),
        }

    pairwise_violations: list[dict[str, object]] = []
    run_pair_violations = 0
    total_run_pairs = 0
    for low_index, low in enumerate(LEVELS):
        for high in LEVELS[low_index + 1 :]:
            low_mean = float(groups[low]["true_measured_mean_N"]["mean"])
            high_mean = float(groups[high]["true_measured_mean_N"]["mean"])
            low_values = [float(row["true_measured_mean_N"]) for row in groups[low]["repeat_values"]]
            high_values = [float(row["true_measured_mean_N"]) for row in groups[high]["repeat_values"]]
            pair_run_violations = sum(1 for lv in low_values for hv in high_values if lv > hv)
            run_pair_violations += pair_run_violations
            total_run_pairs += len(low_values) * len(high_values)
            inversion = low_mean - high_mean
            if inversion > 0.0:
                pairwise_violations.append(
                    {
                        "lower_setpoint": low,
                        "higher_setpoint": high,
                        "lower_group_mean_N": low_mean,
                        "higher_group_mean_N": high_mean,
                        "inversion_N": inversion,
                        "repeat_pair_violations": pair_run_violations,
                        "systematic": inversion > NATIVE_DEADZONE_N,
                    }
                )

    required_rho = 0.8
    monotonic_associations = all(
        float(associations[name]["individual_run_spearman"]) >= required_rho
        and float(associations[name]["linear_slope_per_setpoint"]) > 0.0
        for name in associations
    )
    systematic_reversal = any(bool(row["systematic"]) for row in pairwise_violations)
    endpoints_ordered = all(
        float(associations[name]["group_means"][-1]) > float(associations[name]["group_means"][0])
        for name in associations
    )
    mapping_valid = bool(formal_valid and monotonic_associations and not systematic_reversal and endpoints_ordered)

    mapping = {
        "status": "PASS" if mapping_valid else "FAIL",
        "formal_trace_valid": formal_valid,
        "setpoint_levels": list(LEVELS),
        "repeats_per_level": len(REPEATS),
        "total_runs": len(runs),
        "configuration_frozen_before_outcome_analysis": True,
        "primary_measured_force": "object_filtered_true_squeeze_N",
        "metric_window": "handoff continuation through end of 30-step post-lift hold",
        "group_summary": {str(level): groups[level] for level in LEVELS},
        "associations": associations,
        "pairwise_group_monotonic_violations": pairwise_violations,
        "pairwise_group_violation_count": len(pairwise_violations),
        "pairwise_run_violation_count": run_pair_violations,
        "pairwise_run_comparison_count": total_run_pairs,
        "pairwise_run_violation_rate": run_pair_violations / total_run_pairs,
        "systematic_reversal_present": systematic_reversal,
        "systematic_reversal_tolerance_N": NATIVE_DEADZONE_N,
        "validity_rule": contract["mapping_gate"]["valid_if"],
        "tabero_native_continuous_setpoint_mapping_valid": mapping_valid,
        "diagnosis": (
            "Setpoint 4 is a deterministic systematic inversion: all three runs lose hold contact and have lower measured force than setpoints 2 and 3."
            if not mapping_valid
            else "Native fixed execution provides ordered control of measured interaction intensity."
        ),
        "arm_hash_pair_count": len(arm_hashes),
        "arm_hash_pairs": arm_hashes,
        "teardown_note": "Isaac camera weak-reference abort occurred after COMPLETE artifacts were atomically written; it did not alter rollout data.",
        "repeatability_note": "The fixed seed, exact initial state, deterministic replay, and fixed controller produced bitwise-identical summary metrics within each setpoint group. These are independent fresh processes but do not estimate randomized-seed variance.",
    }
    dump_json(OUT / "TABERO_NATIVE_SETPOINT_MAPPING.json", mapping)

    copied_trace_hash_match = True
    for level in LEVELS:
        for repeat in REPEATS:
            source = RUNS / f"setpoint_{level}_repeat_{repeat}" / "TABERO_NATIVE_DIAGNOSTIC_TRACE.csv"
            copied = OUT / f"TABERO_SETPOINT_{level}_REPEAT_{repeat}.csv"
            copied_trace_hash_match &= sha256(source) == sha256(copied)
    dump_json(
        OUT / "DATA_QUALITY_AUDIT.json",
        {
            "status": "PASS" if formal_valid and copied_trace_hash_match else "FAIL",
            "formal_run_count": len(runs),
            "complete_result_count": sum(row["status"] == "COMPLETE" for row in runs),
            "fresh_process_count": sum(row["fresh_process"] is True for row in runs),
            "handoff_parity_pass_count": sum(row["handoff_parity"] is True for row in runs),
            "raw_arm_unchanged_count": sum(row["raw_arm_trajectory_changed"] is False for row in runs),
            "full_hold_window_available_count": sum(int(row["hold_rows"]) == 30 for row in runs),
            "configuration_match_count": sum(
                float(row["feedforward_k"]) == 0.9 and row["contact_override_enabled"] is False
                for row in runs
            ),
            "unique_arm_hash_pairs": len(arm_hashes),
            "copied_trace_sha256_match": copied_trace_hash_match,
            "scientific_retry_count": 0,
            "prelaunch_engineering_retry_count": 14,
            "prelaunch_retry_reason": "The first batch wrapper pre-created output directories; the runner refused before Isaac launch or rollout. Empty failed directories were removed and the same frozen commands were rerun.",
            "teardown_abort_after_complete_artifact_count": 15,
        },
    )

    report = f"""# Tabero native continuous-setpoint mapping report

## Decision

**TABERO_NATIVE_CONTINUOUS_SETPOINT_MAPPING_VALID = {'YES' if mapping_valid else 'NO'}**

The fixed native execution gate used 15 fresh Isaac processes: setpoints 2, 3, 4, 5, and 6 with three repeats each. All runs passed exact handoff parity, retained the same raw VLA arm action hashes, disabled every FORTE custom low-level controller, used native feed-forward `k=0.9`, and kept target-contact override off.

Because the reset, seed, replay, and controller are deterministic, all three fresh-process repeats within each setpoint group produced identical summary metrics. The repeats establish cross-process reproducibility, not randomized-seed uncertainty.

The mapping fails the predeclared ordered-control criterion. Setpoint 4 is not a small stochastic overlap: all three repeats produced the same inversion, lost 30-step hold contact, and yielded a measured mean below both setpoints 2 and 3. No controller parameter was changed after observing this result.

## Measured-force mapping

| Setpoint | True measured mean (N) | Hold mean (N) | Top-5% (N) | Exposure (N·s) | Lift+hold success |
|---:|---:|---:|---:|---:|---:|
"""
    for level in LEVELS:
        group = groups[level]
        report += (
            f"| {level} | {group['true_measured_mean_N']['mean']:.4f} ± {group['true_measured_mean_N']['std']:.4f} "
            f"| {group['true_hold_mean_N']['mean']:.4f} ± {group['true_hold_mean_N']['std']:.4f} "
            f"| {group['true_top5_N']['mean']:.4f} ± {group['true_top5_N']['std']:.4f} "
            f"| {group['true_force_exposure_Ns']['mean']:.4f} ± {group['true_force_exposure_Ns']['std']:.4f} "
            f"| {group['lift_hold_success_rate']:.0%} |\n"
        )
    report += f"""

Run-level Spearman correlations were `{associations['measured_mean']['individual_run_spearman']:.4f}` for measured mean, `{associations['top5']['individual_run_spearman']:.4f}` for top-5% force, and `{associations['exposure']['individual_run_spearman']:.4f}` for force exposure. There were `{len(pairwise_violations)}` inverted setpoint-group pairs ({run_pair_violations}/{total_run_pairs} repeat-level pair comparisons).

## Interpretation

The endpoint trend is positive, but the 4-setpoint branch is a systematic discontinuity rather than a stable continuous ordering. Under the user's frozen stop rule, this is insufficient to claim that ActiveForcing can control gentleness through this continuous native setpoint on root7703. The low-level controller is therefore not frozen as a validated final execution mapping, and the model-driven smoke is gated off.

## old720 compatibility

old720 is **partially reusable**, not execution-identical. Its branches used the same native `ForcePositionAction` family, 13D force slots, feed-forward `k=0.9`, and override-off semantics, but also ran an outer fixed-step aperture servo that updated `d_pred`. Those candidate values remain setpoints under their historical fixed execution contract; they must not be relabeled as exact measured Newton forces or treated as fully matched to the current pure-native continuation.
"""
    (OUT / "TABERO_NATIVE_SETPOINT_MAPPING_REPORT.md").write_text(report, encoding="utf-8")

    gate_reason = "NOT_RUN_MAPPING_GATE_FAILED" if not mapping_valid else "READY_FOR_MAPPING_PASS_SMOKE"
    dump_json(
        OUT / "IF_MAPPING_PASS_ROOT7703_PROBE.json",
        {"status": gate_reason, "probe_run": False, "reason": "ActiveForcing smoke is allowed only after native mapping PASS."},
    )
    write_csv(
        OUT / "IF_MAPPING_PASS_ROOT7703_ACTIVEFORCING_TRACE.csv",
        [] if mapping_valid else [{"status": gate_reason, "reason": "native continuous-setpoint mapping failed"}],
    )
    dump_json(
        OUT / "IF_MAPPING_PASS_ROOT7703_ACTIVEFORCING_RESULT.json",
        {
            "status": gate_reason,
            "activeforcing_live_run": False,
            "selected_setpoint": None,
            "activeforcing_method_changed": False,
            "posterior_changed": False,
            "expected_utility_changed": False,
        },
    )

    low, high = groups[2], groups[6]
    dump_json(
        OUT / "BASELINE_COMPARISON.json",
        {
            "status": "BASELINES_COMPLETE_ACTIVEFORCING_NOT_RUN" if not mapping_valid else "BASELINES_COMPLETE",
            "fixed_low": low,
            "fixed_high": high,
            "activeforcing": {"status": gate_reason},
            "measured_force_reduction_claim_available": False,
            "reason": "No valid ActiveForcing-vs-high comparison without a mapping-gated ActiveForcing live run.",
        },
    )

    final_report = f"""# Final execution pivot report

## Outcome

`FINAL_STATUS = TABERO_NATIVE_CONTINUOUS_SETPOINT_MAPPING_FAILED`

The custom precise-Newton controller route remains abandoned. The proposed replacement—continuous setpoint execution through the fixed native Tabero hybrid controller—was tested with 15 fresh root7703 processes and did **not** establish a stable ordered mapping to measured physical interaction force.

The decisive failure is the 4-setpoint group: measured true-force mean `{groups[4]['true_measured_mean_N']['mean']:.4f} N`, hold mean `{groups[4]['true_hold_mean_N']['mean']:.4f} N`, and lift+hold success `{groups[4]['lift_hold_success_rate']:.0%}`. This is systematically below the 2- and 3-setpoint groups and occurs identically in all three repeats.

## Frozen-method implications

No posterior, Expected Utility, probe, arm trajectory, or controller parameter was modified. Because the execution gate failed, no ActiveForcing live smoke was run. It would be scientifically invalid to report measured-force savings or to freeze this native mapping as the final gentleness interface.

## Next action

Stop controller experimentation as instructed. Revisit the execution/training contract before making a continuous-gentleness claim: either identify a historically matched execution contract whose setpoint-to-measured response is demonstrably ordered, or reformulate/recollect supervision directly against measured force/exposure. This decision belongs to the next method-contract stage, not to another low-level tuning loop.
"""
    (OUT / "FINAL_PIVOT_REPORT.md").write_text(final_report, encoding="utf-8")


if __name__ == "__main__":
    main()
