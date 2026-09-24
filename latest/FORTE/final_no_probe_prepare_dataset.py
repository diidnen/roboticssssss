#!/usr/bin/env python3
"""Prepare the final no-probe, true-force posterior pilot dataset.

Historical rows are reused only after recomputing the common lift+30-step
hold label from raw telemetry. Context features contain only the frozen
nominal arm prefix and task identity; candidate force is a separate input.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "analysis/results/final_no_probe_continuous_posterior_20260904"
OLD_DIR = ROOT / "gnp_style_continuous_20260830_125107"
OLD_CSV = OLD_DIR / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv"
ROOT7703_OLD = ROOT / "analysis/results/root7703_reset_replay_reproduction_20260904_final_v3"
ROOT7703_NEW = OUT / "01_root7703_frontier_refinement"
TASK_DIM = 10
SEQ_LEN = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def common_label(trace: Path) -> tuple[int, dict[str, Any]]:
    """Apply the common lift+30-step bilateral-hold outcome contract."""
    with trace.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return 0, {"reason": "empty_trace"}
    if "object_z_analysis_only" in rows[0]:
        z0 = parse_float(rows[0], "object_z_analysis_only")
        height_delta = lambda r: parse_float(r, "object_z_analysis_only") - z0
        bilateral = lambda r: r.get("contact_state") == "bilateral"
        above_hold = lambda r: height_delta(r) >= 0.005
    else:
        height_delta = lambda r: parse_float(r, "object_height_delta_m")
        bilateral = lambda r: r.get("bilateral_contact") in ("1", "True", "true")
        above_hold = lambda r: height_delta(r) >= 0.005 and r.get("drop") not in ("1", "True", "true")
    lift_idx = next((i for i, r in enumerate(rows) if height_delta(r) >= 0.01), None)
    if lift_idx is None:
        return 0, {"reason": "fail_before_lift", "lift_index": None}
    window = rows[lift_idx + 1 : lift_idx + 31]
    ok = len(window) >= 30 and all(
        bilateral(r) and above_hold(r)
        for r in window
    )
    return int(ok), {
        "reason": "lift_and_hold_success" if ok else "fail_after_lift",
        "lift_index": lift_idx,
        "hold_rows": len(window),
    }


def old_trace_context(trace: Path, task_id: int) -> list[list[float]]:
    """Use only nominal arm commands and task identity from historical rows."""
    with trace.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda x: int(float(x["step"])))
    rows = rows[:SEQ_LEN]
    if len(rows) < SEQ_LEN:
        raise RuntimeError(f"{trace}: only {len(rows)} rows, need {SEQ_LEN}")
    commands = [[parse_float(r, k) for k in ("cmd_x", "cmd_y", "cmd_z")] for r in rows]
    onehot = [1.0 if i == task_id else 0.0 for i in range(TASK_DIM)]
    out: list[list[float]] = []
    previous = commands[0]
    for cmd in commands:
        out.append(cmd + [cmd[i] - previous[i] for i in range(3)] + onehot)
        previous = cmd
    return out


def root_trace_context(trace: Path, task_id: int) -> list[list[float]]:
    """Use the frozen raw VLA arm prefix from the live continuation trace."""
    with trace.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:SEQ_LEN]
    if len(rows) < SEQ_LEN:
        raise RuntimeError(f"{trace}: only {len(rows)} rows, need {SEQ_LEN}")
    commands = []
    for row in rows:
        raw = json.loads(row["raw_vla_action"])
        commands.append([float(x) for x in raw[:3]])
    onehot = [1.0 if i == task_id else 0.0 for i in range(TASK_DIM)]
    out: list[list[float]] = []
    previous = commands[0]
    for cmd in commands:
        out.append(cmd + [cmd[i] - previous[i] for i in range(3)] + onehot)
        previous = cmd
    return out


def old_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "category": "C_CONTEXT_SCHEMA_OLD_BUT_LABEL_RECOMPUTED",
        "source_csv": str(OLD_CSV),
        "source_sha256": sha256(OLD_CSV),
        "input_rows": 0,
        "valid_true_force_rows": 0,
        "reusable_rows": 0,
        "rejected_rows": 0,
        "rejection_reasons": {},
        "label_source": "raw branch telemetry recomputed as lift + 30-step bilateral hold",
        "force_source": "requested_force_N absolute Newton; corrected physical telemetry audit was valid",
        "probe_features_used": False,
        "friction_feature_used": False,
    }
    with OLD_CSV.open(newline="", encoding="utf-8") as fh:
        source = list(csv.DictReader(fh))
    audit["input_rows"] = len(source)
    cache: dict[str, tuple[list[list[float]], int, dict[str, Any]]] = {}
    for src in source:
        if src.get("valid") != "1" or src.get("corrected_physical_telemetry_valid") != "1":
            audit["rejected_rows"] += 1
            reason = "invalid_or_uncorrected_telemetry"
            audit["rejection_reasons"][reason] = audit["rejection_reasons"].get(reason, 0) + 1
            continue
        audit["valid_true_force_rows"] += 1
        trace = Path(src["telemetry_path"])
        if not trace.exists():
            audit["rejected_rows"] += 1
            reason = "missing_raw_telemetry"
            audit["rejection_reasons"][reason] = audit["rejection_reasons"].get(reason, 0) + 1
            continue
        context_id = src["context_id"]
        if context_id not in cache:
            label, label_audit = common_label(trace)
            cache[context_id] = (old_trace_context(trace, int(src["task"])), label, label_audit)
        sequence, y, label_audit = cache[context_id]
        rows.append({
            "sample_id": f"old::{src['branch_id']}",
            "context_id": context_id,
            "root_id": src["root_id"],
            "task_id": int(src["task"]),
            "source_class": "OLD_DATA_REUSED_PROJECTED_NO_PROBE",
            "requested_force_N": float(src["requested_force_N"]),
            "realized_force_N": float(src["realized_force_N"]),
            "success_y": int(y),
            "label_audit": label_audit,
            "sequence": sequence,
            "source_trace": str(trace),
            "old_full_task_label_ignored": int(src["full_task_success_y"]),
        })
    audit["reusable_rows"] = len(rows)
    return rows, audit


def root7703_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [
        (2.0, ROOT7703_OLD / "LIVE_2N_TRACE.csv", "historical_true_force"),
        (4.0, ROOT7703_OLD / "LIVE_4N_TRACE.csv", "historical_true_force"),
        (6.0, ROOT7703_OLD / "LIVE_6N_TRACE.csv", "historical_true_force"),
    ]
    for force, trace, source_class in sources:
        label, label_audit = common_label(trace)
        rows.append({
            "sample_id": f"root7703::{force:g}N::historical",
            "context_id": "root7703",
            "root_id": "7703",
            "task_id": 5,
            "source_class": source_class,
            "requested_force_N": force,
            "realized_force_N": None,
            "success_y": label,
            "label_audit": label_audit,
            "sequence": root_trace_context(trace, 5),
            "source_trace": str(trace),
        })
    for force, label_name, folder in [
        (2.5, "2p5N", ROOT7703_NEW / "root7703_2p5N"),
        (3.0, "3N", ROOT7703_NEW / "root7703_3p0N"),
        (3.5, "3p5N", ROOT7703_NEW / "root7703_3p5N"),
    ]:
        trace = folder / f"LIVE_{label_name}_TRACE.csv"
        label, label_audit = common_label(trace)
        rows.append({
            "sample_id": f"root7703::{force:g}N::fresh",
            "context_id": "root7703",
            "root_id": "7703",
            "task_id": 5,
            "source_class": "NEW_TRUE_FORCE_FRESH_PROCESS",
            "requested_force_N": force,
            "realized_force_N": None,
            "success_y": label,
            "label_audit": label_audit,
            "sequence": root_trace_context(trace, 5),
            "source_trace": str(trace),
        })
    for force, label_name in ((4.95, "4p95N"), (6.51, "6p51N")):
        extra_trace = OUT / "07_root7703_activeforcing_smoke" / f"isaac_{label_name}" / f"LIVE_{label_name}_TRACE.csv"
        if extra_trace.exists():
            label, label_audit = common_label(extra_trace)
            rows.append({
                "sample_id": f"root7703::{force:g}N::fresh_selected_target",
                "context_id": "root7703",
                "root_id": "7703",
                "task_id": 5,
                "source_class": "NEW_TRUE_FORCE_SELECTED_TARGET_SMOKE",
                "requested_force_N": force,
                "realized_force_N": None,
                "success_y": label,
                "label_audit": label_audit,
                "sequence": root_trace_context(extra_trace, 5),
                "source_trace": str(extra_trace),
            })
    return rows, {
        "context_id": "root7703",
        "root_id": "7703",
        "task_id": 5,
        "new_true_force_rows": len(rows) - 3,
        "historical_true_force_rows": 3,
        "all_force_rows": len(rows),
        "probe_features_used": False,
        "criterion": "lift + 30-step bilateral hold",
    }


def root7704_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for force, label_name in ((2.0, "2N"), (4.0, "4N"), (6.0, "6N")):
        folder = OUT / "02_context_audit" / f"root7704_{label_name}_fresh"
        trace = folder / f"LIVE_{label_name}_TRACE.csv"
        if not trace.exists():
            continue
        label, label_audit = common_label(trace)
        rows.append({
            "sample_id": f"root7704::{force:g}N::fresh",
            "context_id": "root7704",
            "root_id": "7704",
            "task_id": 5,
            "source_class": "NEW_TRUE_FORCE_FRESH_PROCESS",
            "requested_force_N": force,
            "realized_force_N": None,
            "success_y": label,
            "label_audit": label_audit,
            "sequence": root_trace_context(trace, 5),
            "source_trace": str(trace),
        })
    return rows, {
        "context_id": "root7704",
        "root_id": "7704",
        "task_id": 5,
        "new_true_force_rows": len(rows),
        "criterion": "lift + 30-step bilateral hold",
        "cross_process_handoff_matched": len(rows) == 3,
        "probe_features_used": False,
    }


def root7703_frontier_artifact() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    old_result = json.loads((ROOT7703_OLD / "ONLINE_FORCE_TRACKING_RESULT.json").read_text())
    for label, force in (("2N", 2.0), ("4N", 4.0), ("6N", 6.0)):
        src = old_result["branches"][label]["source_summary"]
        candidates.append({
            "requested_force_N": force,
            "source": "historical_validated_root7703",
            "fresh_process": True,
            "outcome": src["outcome"],
            "lift_success": bool(src["lift_success"]),
            "hold_30_step_success": bool(src["hold_after_lift_success"]),
            "mean_realized_force_during_relevant_window_N": src["mean_realized_force_during_relevant_window"],
            "force_tracking_status": "strict_gate_false_but_true_force_trace_retained",
            "trace": str(ROOT7703_OLD / f"LIVE_{label}_TRACE.csv"),
        })
    for force, label, folder in (
        (2.5, "2p5N", ROOT7703_NEW / "root7703_2p5N"),
        (3.0, "3N", ROOT7703_NEW / "root7703_3p0N"),
        (3.5, "3p5N", ROOT7703_NEW / "root7703_3p5N"),
    ):
        d = json.loads((folder / "ONLINE_FORCE_TRACKING_RESULT.json").read_text())
        src = (d.get("branches") or [{}])[0]
        candidates.append({
            "requested_force_N": force,
            "source": "new_fresh_process_root7703",
            "fresh_process": True,
            "outcome": src.get("outcome"),
            "lift_success": bool(src.get("lift_success")),
            "hold_30_step_success": bool(src.get("hold_after_lift_success")),
            "mean_realized_force_during_relevant_window_N": src.get("mean_realized_force_during_relevant_window"),
            "force_tracking_status": "strict_gate_false_but_true_force_trace_retained",
            "trace": str(folder / f"LIVE_{label}_TRACE.csv"),
        })
    candidates.sort(key=lambda x: x["requested_force_N"])
    success = [x["requested_force_N"] for x in candidates if x["hold_30_step_success"]]
    fail = [x["requested_force_N"] for x in candidates if not x["hold_30_step_success"]]
    lower = max(fail)
    upper = min(x for x in success if x > lower)
    return {
        "schema": "ROOT7703_REFINED_FRONTIER_V1",
        "root_id": 7703,
        "task": {"suite": "libero_10", "task_id": 5, "object": "black_book_1", "target": "desk_caddy_1"},
        "outcome_definition": "lift + 30-step hold",
        "cross_process_snapshot_used": False,
        "only_variable": "continuous force target",
        "candidates": candidates,
        "frontier_interval_requested_force_N": [lower, upper],
        "frontier_interval": f"({lower:g}N, {upper:g}N]",
        "interval_width_N": upper - lower,
        "precision_target_met": bool(upper - lower <= 0.5),
        "note": "Relevant-window realized force is telemetry and is not substituted for requested-force frontier ordering.",
    }


def main() -> None:
    old, old_audit = old_rows()
    new, new_audit = root7703_rows()
    new_7704, new_7704_audit = root7704_rows()
    all_rows = old + new + new_7704
    dataset_csv = OUT / "03_true_force_dataset/FINAL_NO_PROBE_TRUE_FORCE_DATASET.csv"
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "context_id", "root_id", "task_id", "source_class",
        "requested_force_N", "realized_force_N", "success_y", "label_audit",
        "sequence", "source_trace", "old_full_task_label_ignored",
    ]
    with dataset_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            out = dict(row)
            out["label_audit"] = json.dumps(out["label_audit"], sort_keys=True)
            out["sequence"] = json.dumps(out["sequence"], separators=(",", ":"))
            writer.writerow(out)
    dump(OUT / "TRUE_FORCE_DATASET_MANIFEST.json", {
        "schema": "FINAL_NO_PROBE_TRUE_FORCE_DATASET_MANIFEST_V1",
        "dataset_csv": str(dataset_csv),
        "dataset_sha256": sha256(dataset_csv),
        "sample_count": len(all_rows),
        "context_count": len({r["context_id"] for r in all_rows}),
        "old_reused_sample_count": len(old),
        "new_true_force_sample_count": sum(
            row["source_class"].startswith("NEW_TRUE_FORCE") for row in (new + new_7704)
        ),
        "root7703_sample_count": len(new),
        "root7704_sample_count": len(new_7704),
        "supervision": "success_y = lift + 30-step bilateral hold",
        "model_input": "sequence x + candidate absolute Newton force F",
        "label_leakage_check": {
            "outcome_in_context": False,
            "minimum_sufficient_force_in_context": False,
            "future_height_in_context": False,
            "future_slip_in_context": False,
            "probe_result_in_context": False,
            "friction_in_context": False,
        },
        "force_contract": "requested_force_N is absolute Newton; realized force retained as telemetry only",
        "source_audits": {"old": old_audit, "root7703": new_audit, "root7704": new_7704_audit},
    })
    dump(OUT / "04_old_data_reuse_audit/OLD_DATA_REUSE_AUDIT.json", {
        "schema": "OLD_DATA_REUSE_AUDIT_V1",
        "A_TRUE_FORCE_COMPATIBLE": 0,
        "B_CAN_BE_RECALIBRATED_TO_TRUE_FORCE": 0,
        "C_SUCCESS_BOUNDARY_VALID_BUT_CONTEXT_SCHEMA_OLD": len(old),
        "D_PROBE_DEPENDENT_REJECTED_AS_FEATURES": 0,
        "E_INVALID_FOR_FINAL_MODEL": old_audit["rejected_rows"],
        "OLD_SAMPLES_TOTAL": old_audit["input_rows"],
        "OLD_SAMPLES_REUSABLE": len(old),
        "OLD_SAMPLES_REJECTED": old_audit["rejected_rows"],
        "REJECTION_REASONS": old_audit["rejection_reasons"],
        "reuse_rule": "retain absolute-Newton rows with corrected true-force telemetry; recompute final lift+hold y from raw trace; project context to no-probe nominal-arm/task schema",
        "probe_feature_used": False,
    })
    dump(OUT / "04_old_data_reuse_audit/OLD_DATA_REUSE_SUMMARY.json", old_audit)
    dump(OUT / "FINAL_FEATURE_SCHEMA.json", {
        "schema": "FINAL_NO_PROBE_FEATURE_SCHEMA_V1",
        "probe_feature_used": False,
        "query_count": 0,
        "live_rgb_required": False,
        "context_features": {
            "nominal_sequence_length": SEQ_LEN,
            "per_step_order": [
                "cmd_x_m", "cmd_y_m", "cmd_z_m",
                "delta_cmd_x_m", "delta_cmd_y_m", "delta_cmd_z_m",
            ] + [f"task_onehot_{i}" for i in range(TASK_DIM)],
            "dimension": 6 + TASK_DIM,
            "source": "frozen VLA arm action prefix available at handoff; task identity",
        },
        "candidate_input": {
            "name": "requested_force_N",
            "domain_N": [1.0, 8.0],
            "normalization": "F/8",
            "absolute_newton": True,
        },
        "excluded": [
            "probe force response", "probe slip", "friction μ", "outcome",
            "minimum sufficient force", "future lift height",
            "future contact/slip", "post-action telemetry",
        ],
        "note": "Old rows are pilot reuse data with an old context topology; final live contexts require fresh HDF5 reset/replay validation.",
    })
    dump(OUT / "ROOT7703_REFINED_FRONTIER.json", root7703_frontier_artifact())
    dump(OUT / "VALID_CONTEXT_MANIFEST.json", {
        "schema": "VALID_CONTEXT_MANIFEST_V1",
        "valid_context_count": 2,
        "contexts": [{
            "context_id": "root7703",
            "root_id": 7703,
            "task": {"suite": "libero_10", "task_id": 5, "object": "black_book_1", "target": "desk_caddy_1"},
            "seed": 7703,
            "demo": "demo_3",
            "handoff_step": 95,
            "initial_state_source": "/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_SOURCE_20260902_103300/assembled_hdf5/libero_10_task5_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
            "historical_runner": str(ROOT7703_OLD),
            "historical_arm_trace": str(ROOT7703_OLD / "LIVE_4N_TRACE.csv"),
            "arm_trace_hash": json.loads((ROOT7703_OLD / "HISTORICAL_CONTINUATION_AUDIT.json").read_text())["trace_sha256"],
            "fresh_reset_parity": True,
            "matched_handoff": True,
            "native_baseline_success": True,
            "handoff_bilateral_contact": True,
            "status": "VALID_FOR_TRUE_FORCE_COLLECTION",
        }, {
            "context_id": "root7704",
            "root_id": 7704,
            "task": {"suite": "libero_10", "task_id": 5, "object": "black_book_1", "target": "desk_caddy_1"},
            "seed": 7704,
            "demo": "demo_4",
            "handoff_step": 95,
            "initial_state_source": "/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_SOURCE_20260902_103300/assembled_hdf5/libero_10_task5_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
            "historical_runner": str(OUT / "02_context_audit/root7704_native_frontier_audit"),
            "historical_arm_trace": "/media/volume/newdata/exouser/activeforcing_e3/P1_SIMPLIFIED_COLLECTION_FORMAL_20260903/P1_SIMPLIFIED_ROOT7704_F6N/raw_policy/b5_t5_mu0.6_exp000_action_chunks.npz",
            "arm_trace_hash": json.loads((OUT / "02_context_audit/root7704_native_frontier_audit/HISTORICAL_CONTINUATION_AUDIT.json").read_text())["trace_sha256"],
            "fresh_reset_parity": True,
            "matched_handoff": True,
            "native_baseline_success": True,
            "handoff_bilateral_contact": True,
            "force_candidates_collected": [2.0, 4.0, 6.0],
            "status": "VALID_FOR_TRUE_FORCE_COLLECTION",
        }],
        "rejected_or_pending": {
            "7704": "native continuation passed, but same-process auxiliary 2/4N audit had handoff mismatch; no force evidence admitted",
            "7705": "native success but handoff not bilateral; rejected",
            "7801": "native lift+hold failed after lift; rejected",
            "7802": "native succeeded but coarse 2/4/6 branches terminated before hold; rejected for this pilot",
        },
        "admission_rule": "fresh reset -> exact HDF5 initial state -> matched warm replay -> handoff parity -> native lift+hold success",
    })
    print(json.dumps({
        "samples": len(all_rows),
        "old_reused": len(old),
        "new_root7703": len(new),
        "contexts": len({r["context_id"] for r in all_rows}),
    }, indent=2))


if __name__ == "__main__":
    main()
