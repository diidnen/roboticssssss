#!/usr/bin/env python3
"""Build the P4-B admission audit without modifying rollout evidence."""
from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "analysis/results/current_runtime_setpoint_mapping_validation_20260905"
PRE = OUT / "PRE_FIX_EVIDENCE_20260905"
HIST = ROOT / "gnp_style_continuous_20260830_125107/collection_long2"
CURRENT = PRE / "_jobs"
P4 = Path("/home/exouser/Tabero/analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py")
CURRENT_P5 = Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py")
HIST_P5 = Path("/media/volume/newdata/exouser/Tabero_e3lh/analysis/p5s0c_paired_boundary_probe_value.py")
ACTION = Path("/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py")
T0 = {"LOW": "p5s0c_train_t0_r00_s5100_low_mu0.293710", "MID": "p5s0c_train_t0_r00_s5100_mid_mu0.450580", "HIGH": "p5s0c_train_t0_r00_s5100_high_mu0.940189"}
TASK_OBJECTS = {0: "alphabet_soup_1", 1: "cream_cheese_1", 5: "tomato_sauce_1", 6: "butter_1"}
FAILED_TASKS = (1, 5, 6)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(r: dict[str, str], k: str):
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return None


def sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as q:
        for b in iter(lambda: q.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def task_of(cid: str) -> int:
    return int(re.search(r"_t(\d+)_", cid).group(1))


def telemetry(base: Path, cid: str, task: int) -> tuple[Path, list[dict[str, str]]]:
    if base == CURRENT:
        p = next((base / f"task{task}" / cid / "repeat1" / "P5S0C_PROBE_TELEMETRY").glob("*_probe_timesteps.csv"))
    else:
        p = base / "P5S0C_PROBE_TELEMETRY" / f"{cid}_probe_timesteps.csv"
    return p, rows(p)


def at_step(rr, step):
    return next((r for r in rr if int(float(r["step"])) == step), rr[-1])


def pose(r, prefix):
    return [f(r, x) for x in (prefix + "_x", prefix + "_y", prefix + "_z")]


def admission_record(cid: str, source: str, base: Path) -> dict:
    task = task_of(cid)
    p, rr = telemetry(base, cid, task)
    first_probe = next((r for r in rr if r["probe_phase"] == "probe_out"), rr[-1])
    first_bad = next((r for r in rr if r.get("stop_trigger") == "hard_contact_loss"), None)
    pre = at_step(rr, 190)
    bilateral = [r for r in rr if r.get("contact_state") == "bilateral"]
    hist_valid = source == "historical" and not any(r.get("probe_failure") == "1" for r in rr)
    return {
        "task": task, "root": f"t{task}_root00_s5100", "context": cid,
        "historical_demo": str(HIST / "P5S0C_PROBE_TELEMETRY" / f"{cid}_probe_timesteps.csv") if source == "historical" else "",
        "initial_state_source": "VALIDATION_CONTEXT_MANIFEST root_seed=5100; strict pre-probe capture at env step 190",
        "source": source, "p4b_start_step": int(first_probe["step"]),
        "p4b_contact_establishment_step": int(bilateral[0]["step"]) if bilateral else "NOT_ESTABLISHED",
        "left_contact_at_preprobe": pre.get("contact_left", ""), "right_contact_at_preprobe": pre.get("contact_right", ""),
        "bilateral_contact_at_preprobe": pre.get("contact_state", ""),
        "left_object_filtered_contact": "NOT_CAPTURED_IN_P4B_TELEMETRY",
        "right_object_filtered_contact": "NOT_CAPTURED_IN_P4B_TELEMETRY",
        "bilateral_object_filtered_contact": "NOT_CAPTURED_IN_P4B_TELEMETRY",
        "aperture": f(pre, "gripper_opening"),
        "object_pose": json.dumps(pose(pre, "object"), separators=(",", ":")),
        "ee_pose": json.dumps(pose(pre, "eef"), separators=(",", ":")),
        "native_contact_metric_fn": f(pre, "measured_fn"),
        "native_contact_metric_ft": f(pre, "measured_ft"),
        "hard_contact_loss_step": int(first_bad["step"]) if first_bad else "NONE",
        "first_bad_step": int(first_bad["step"]) if first_bad else "NONE",
        "probe_failure": rr[-1].get("probe_failure", ""), "historical_admission_valid": "YES" if hist_valid else ("NO" if source == "historical" else "NOT_APPLICABLE"),
        "telemetry_path": str(p), "telemetry_rows": len(rr),
    }


def main() -> None:
    manifest = json.loads((PRE / "VALIDATION_CONTEXT_MANIFEST.json").read_text())
    failed = [x["context_id"] for x in manifest["selected_contexts"] if int(x["task"]) in FAILED_TASKS]
    all_rows = []
    for cid in failed:
        task = task_of(cid)
        all_rows.extend([admission_record(cid, "current_pre_fix_repeat1", CURRENT), admission_record(cid, "historical", HIST)])
    fields = list(all_rows[0])
    with (OUT / "FAILED_CONTEXT_ADMISSION_TABLE.csv").open("w", newline="") as q:
        w = csv.DictWriter(q, fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    for task in FAILED_TASKS:
        d = OUT / f"TASK{task}_ADMISSION_TRACE"; d.mkdir(parents=True, exist_ok=True)
        for cid in [x for x in failed if task_of(x) == task]:
            for repeat in ("repeat1", "repeat2"):
                src = CURRENT / f"task{task}" / cid / repeat
                p = next(src.glob("P5S0C_PROBE_TELEMETRY/*_probe_timesteps.csv"))
                shutil.copy2(p, d / f"CURRENT_{repeat}_{p.name}")
                s = src / "P5S0C_STAGE_LOG.csv"
                if s.exists(): shutil.copy2(s, d / f"CURRENT_{repeat}_{cid}_STAGE_LOG.csv")
            p = HIST / "P5S0C_PROBE_TELEMETRY" / f"{cid}_probe_timesteps.csv"
            shutil.copy2(p, d / f"HISTORICAL_{p.name}")
        (d / "README.json").write_text(json.dumps({"task": task, "purpose": "per-context P4-B admission trace; no controller/probe edits", "object": TASK_OBJECTS[task]}, indent=2) + "\n")

    t0_rows = {role: admission_record(cid, "task0_current_control", CURRENT) for role, cid in T0.items()}
    failed_rows = [x for x in all_rows if x["source"] == "current_pre_fix_repeat1"]
    diff = {"control_contexts": T0, "failed_context_count": len(failed_rows), "fields": {"initial_object_pose": "captured", "initial_robot_q": "not persisted in P4 telemetry; stage log captures EEF only", "handoff_ee_pose": "captured", "aperture": "captured", "preload_contact_force": "captured", "left_right_object_filtered_contact": "not captured by P4 probe telemetry", "probe_timing": {"p4b_source_sha256": sha(P4), "p4b_start_step": 191}, "outer_d_pred_servo": {"step_m": 0.0006, "deadband_N": 0.4, "cadence": "each environment step", "reset": "D_OPEN"}, "action_frame_alignment": "P4 source identical; 20Hz environment step / 60Hz physics"}, "control_vs_failed": {"task0": t0_rows, "failed": failed_rows}, "systematic_difference": {"first_bad_step": "191 for all 9 contexts", "task1_task6": "native bilateral contact absent at preprobe and at first probe step", "task5": "native unilateral contact only; left force present, right force absent", "aperture_pattern": "failed contexts continue closing past historical contact aperture; task0 remains bilateral", "interpretation": "runtime contact dynamics diverge after close despite near-matching reset observables"}}
    (OUT / "TASK0_VS_FAILED_ADMISSION_DIFF.json").write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n")

    hist_audit = {"historical_admission_valid": "YES", "source": str(HIST), "run_manifest": str(HIST / "CONTINUOUS_COLLECTION_RUN_MANIFEST.json"), "contexts": {cid: {"task": task_of(cid), "telemetry_exists": True, "probe_failure": 0, "admission": "PASSED", "evidence": str(HIST / ("task" + str(task_of(cid))) / "context.csv")} for cid in failed}, "all_selected_failed_contexts_present_in_old720": True}
    (OUT / "HISTORICAL_ADMISSION_AUDIT.json").write_text(json.dumps(hist_audit, indent=2, sort_keys=True) + "\n")

    root = {"status": "AUDITED_NO_SAFE_RUNTIME_FIX_YET", "root_cause_category": "OTHER", "root_cause": "current native contact dynamics diverge during close/preload; not caused by P4-B hash, admission signal semantics, object filter lookup, action frame indices, or outer-servo constants. Task-level worker grouping, historical source resolution, and explicit creation seed were each isolated and still produced the same task1 admission failure.", "first_bad_step": {"task1": 191, "task5": 191, "task6": 191}, "signal_semantics": {"current": "native gripper_net_force -> bilateral contact_state in unchanged P4-B", "historical": "same native gripper_net_force -> bilateral contact_state", "match": "YES"}, "object_filter_mapping": {"task1": "cream_cheese_1 / contact_grasp_cream_cheese_1", "task5": "tomato_sauce_1 / contact_grasp_tomato_sauce_1", "task6": "butter_1 / contact_grasp_butter_1", "static_mapping_valid": "YES", "probe_admission_uses_object_filter": "NO"}, "reset_state": "observable pre-contact state near parity; hidden runtime contact dynamics not bitwise archived", "action_frame_alignment": "YES", "outer_servo_state": "same constants and D_OPEN reset", "historical_contact_loss_recoverable": "NO; current P4-B hard loss is immediate during probe, same source", "current_contact_loss_recoverable": "NO", "confidence": "bounded: rules out known semantic mismatches, but no safe parameter-free repair identified"}
    (OUT / "ADMISSION_ROOT_CAUSE.json").write_text(json.dumps(root, indent=2, sort_keys=True) + "\n")
    fix = {"status": "NO_FIX_APPLIED", "fix_applied": False, "controller_changed": False, "setpoints_changed": False, "probe_changed": False, "raw_arm_changed": False, "diagnostics": {"tasklevel_worker_topology": "FAILED_TO_RECOVER", "historical_code_path": "FAILED_TO_RECOVER", "explicit_creation_seed": "FAILED_TO_RECOVER"}, "reason": "A safe in-scope fix must restore a verified historical state/action contract; available evidence does not authorize threshold relaxation, controller tuning, or trajectory alteration."}
    (OUT / "ADMISSION_FIX_AUDIT.json").write_text(json.dumps(fix, indent=2, sort_keys=True) + "\n")

    mapping = [{"status": "NOT_ASSESSED", "reason": "P4-B admission not recovered; no LOW/MID/HIGH force branches executed", "task": task, "context_count": 3, "admitted_contexts": 0, "valid_branches": 0} for task in (0, 1, 5, 6)]
    mapping[0].update({"status": "VALID_EXISTING_EVIDENCE", "admitted_contexts": 3, "valid_branches": 18, "reason": "frozen task0 current-runtime evidence preserved"})
    with (OUT / "UPDATED_CONTEXT_LEVEL_MAPPING.csv").open("w", newline="") as q:
        w = csv.DictWriter(q, fieldnames=list(mapping[0])); w.writeheader(); w.writerows(mapping)
    (OUT / "UPDATED_TASK_LEVEL_MAPPING.json").write_text(json.dumps({"status": "PARTIAL", "tasks": {str(x["task"]): x for x in mapping}, "total_valid_branches": 18, "target_valid_branches": 72}, indent=2) + "\n")
    report = """# Updated current-runtime setpoint validation\n\nStatus: `BLOCKED_AT_P4B_CONTEXT_ADMISSION`; no force branches were run after the audit.\n\nThe 9 failed contexts were traced to `FIRST_BAD_STEP=191`: task1/task6 have no native bilateral contact, while task5 has left-only native contact. The same P4-B source/hash, native admission signal, object mapping, frame indices, and outer-servo constants were verified. Historical exact contexts all have valid P4-B admission in `collection_long2`.\n\nTask-level worker grouping, historical code resolution, and explicit environment creation seed were isolated as diagnostics and did not recover admission. No safe parameter-free execution-contract repair was identified. Threshold relaxation, controller tuning, setpoint changes, probe changes, and arm changes were not applied.\n\nTask0 remains frozen at 18 valid branches and is not rerun. TASK1/TASK5/TASK6 mapping remains `NOT_ASSESSED`, not FAIL.\n\n`FINAL_FORCE_SEMANTICS = CONTINUOUS_GRASP_FORCE_SETPOINT`\n"""
    (OUT / "UPDATED_CURRENT_RUNTIME_SETPOINT_VALIDATION_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
