#!/usr/bin/env python3
"""Blind current-runtime replacement-context mapping validation.

This is an orchestration/analysis script only.  It calls the existing
current4task wrapper and the existing P5-S0-C runner; it does not edit either
implementation or any force/probe/arm configuration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "analysis/results/current_runtime_setpoint_mapping_validation_20260905"
PROTOCOL = ROOT / "gnp_style_continuous_20260830_125107/GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"
OLD_STATS = ROOT / "analysis/results/old720_historical_monotonicity_forensics_20260905/OLD720_CONTEXT_MONOTONICITY.csv"
TASKS = (1, 5, 6)
ALL_TASKS = (0, 1, 5, 6)
ROLES = ("LOW", "MID", "HIGH")
ROLE_LEVEL = {"LOW": 1, "MID": 3, "HIGH": 5}
BAND_ORDER = {"LOW": 0, "MID": 1, "HIGH": 2}
REPEATS = (1, 2)
MAX_CANDIDATES = 12
ADMISSION_TIMEOUT_S = 1200
BRANCH_TIMEOUT_S = 1500
MAX_BRANCH_ATTEMPTS = 3
WRAPPER = ROOT / "current4task_low_force_e3.py"
RUNNER = Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py")
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
TABERO = Path("/home/exouser/Tabero")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
OPENPI = Path("/media/volume/newdata/exouser/openpi/src")

# The user explicitly stopped recovery of these old choices.  They are
# excluded before candidate ordering is frozen, never examined in this run.
OLD_FAILED_CONTEXTS = {
    1: {
        "p5s0c_train_t1_r00_s5100_low_mu0.240019",
        "p5s0c_train_t1_r00_s5100_mid_mu0.467048",
        "p5s0c_train_t1_r00_s5100_high_mu0.921500",
    },
    5: {
        "p5s0c_train_t5_r00_s5100_low_mu0.270805",
        "p5s0c_train_t5_r03_s5103_mid_mu0.579422",
        "p5s0c_train_t5_r00_s5100_high_mu0.945004",
    },
    6: {
        "p5s0c_train_t6_r00_s5100_low_mu0.257577",
        "p5s0c_train_t6_r00_s5100_mid_mu0.558595",
        "p5s0c_train_t6_r00_s5100_high_mu0.989348",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")


def freeze_candidates() -> dict[str, list[dict]]:
    existing = {}
    if all((OUT / f"TASK{t}_CANDIDATE_ORDER.json").exists() for t in TASKS):
        for task in TASKS:
            existing[str(task)] = json.loads((OUT / f"TASK{task}_CANDIDATE_ORDER.json").read_text(encoding="utf-8"))["candidates"]
        return existing
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    population = {c["context_id"]: c for c in protocol["train_context_population"]}
    all_candidates: dict[str, list[dict]] = {}
    for task in TASKS:
        rows = [c for c in protocol["train_context_population"] if int(c["task"]) == task and c["context_id"] not in OLD_FAILED_CONTEXTS[task]]
        rows.sort(key=lambda c: (int(c["root_index"]), int(c["root_seed"]), BAND_ORDER[c["friction_band"]], c["context_id"]))
        rows = rows[:MAX_CANDIDATES]
        if len(rows) != MAX_CANDIDATES:
            raise RuntimeError(f"task{task}: expected {MAX_CANDIDATES} frozen candidates, found {len(rows)}")
        entries = []
        for index, c in enumerate(rows, 1):
            samples = {int(s["stratum_index"]): float(s["requested_force_N"]) for s in c["continuous_force_samples"]}
            entries.append({
                "order_index": index, "task": task, "root_id": c["root_id"],
                "root_index": int(c["root_index"]), "root_seed": int(c["root_seed"]),
                "context_id": c["context_id"], "friction_band": c["friction_band"],
                "friction": float(c["friction"]),
                "historical_level1_N": samples[1], "historical_level3_N": samples[3], "historical_level5_N": samples[5],
                "setpoint_policy_source": "HISTORICAL_CONTEXT_LEVELS",
                "candidate_source": "FROZEN_GNP_STYLE_TRAIN_CONTEXT_POPULATION",
            })
        order_obj = {
            "artifact": f"TASK{task}_CANDIDATE_ORDER",
            "created_utc": now(), "candidate_order_frozen": "YES",
            "selection_rule": "FIRST_3_PASSES_IN_FROZEN_ORDER",
            "max_admission_candidates_per_task": MAX_CANDIDATES,
            "sort_key": ["root_index", "root_seed", "friction_band_order_LOW_MID_HIGH", "context_id"],
            "excluded_old_contexts": sorted(OLD_FAILED_CONTEXTS[task]),
            "excluded_reason": "user-directed stop: do not restore the 9 previously failed historical contexts",
            "candidate_count": len(entries), "candidates": entries,
            "source_protocol": str(PROTOCOL), "source_protocol_sha256": sha256(PROTOCOL),
        }
        write_json(OUT / f"TASK{task}_CANDIDATE_ORDER.json", order_obj)
        all_candidates[str(task)] = entries
    write_json(OUT / "REPLACEMENT_SELECTION_RULE.json", {
        "selection_rule": "FIRST_3_PASSES_IN_FROZEN_ORDER",
        "candidate_order_frozen": "YES", "max_admission_candidates_per_task": MAX_CANDIDATES,
        "admission_only_before_branching": "YES", "outcome_blind": "YES",
        "force_outcomes_read_during_selection": "NO", "posterior_or_eu_read_during_selection": "NO",
        "tasks": list(TASKS), "candidate_order_hashes": {str(t): sha256(OUT / f"TASK{t}_CANDIDATE_ORDER.json") for t in TASKS},
    })
    return all_candidates


def worker_env(job_dir: Path, task: int, manifest: Path, context_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(TABERO), str(OPENPI)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(TABERO),
        "P5S0C_OUT": str(job_dir), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": str(task),
        "P5S0C_TARGET_MANIFEST": str(manifest), "P5S0C_CONTEXT_IDS": context_id,
        "P5S0C_SKIP_REPLAY": "1", "CONTINUOUS_STRICT_PREPROBE_WRAPPER": "1",
        "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
    })
    return env


def run_worker(job_dir: Path, task: int, context_id: str, specs: list[dict], timeout_s: int) -> dict:
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest = job_dir / "WORKER_TARGET_MANIFEST.json"
    write_json(manifest, {"manifest_name": "REPLACEMENT_WORKER_TARGET", "contexts": {context_id: specs}, "expected_contexts": 1, "expected_branches": len(specs)})
    log = job_dir / "isaac_worker.log"
    cmd = [str(ISAAC_PY), "-u", str(WRAPPER), "--worker"]
    started = time.time()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=TABERO, env=worker_env(job_dir, task, manifest, context_id), stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill(); rc = proc.wait()
    return {"returncode": rc, "elapsed_s": time.time() - started, "job_dir": str(job_dir), "log": str(log)}


def first_context_row(job_dir: Path, task: int) -> dict[str, str]:
    path = job_dir / f"task{task}" / "context.csv"
    rows = read_csv(path) if path.exists() else []
    return rows[0] if rows else {}


def admission_detail(job_dir: Path, task: int, context_id: str, run_record: dict) -> dict:
    row = first_context_row(job_dir, task)
    telemetry = Path(row.get("probe_telemetry_path", ""))
    if not telemetry.exists():
        found = list((job_dir / "task" + str(task)).glob("**/*probe_timesteps.csv")) if False else []
        candidates = list((job_dir / f"task{task}").glob("**/*probe_timesteps.csv"))
        telemetry = candidates[0] if candidates else Path("")
    probe_rows = read_csv(telemetry) if telemetry and telemetry.exists() else []
    bad = next((r for r in probe_rows if r.get("stop_trigger") == "hard_contact_loss"), {})
    pre = next((r for r in probe_rows if int(float(r.get("step", 0) or 0)) == 190), {})
    if not pre:
        pre = probe_rows[-1] if probe_rows else {}
    passed = int(float(row.get("probe_qualified", 0) or 0)) == 1 and row.get("status") == "CONTEXT_READY_FOR_BRANCHING" and int(float(row.get("completed_primary_branches", 0) or 0)) == 0
    reason = "PASS" if passed else (row.get("stop_reason") or row.get("status") or ("worker_returncode" if run_record.get("returncode", 1) else "missing_context_output"))
    return {
        "task": task, "context": context_id, "PASS/FAIL": "PASS" if passed else "FAIL",
        "probe_qualified": row.get("probe_qualified", ""), "status": row.get("status", ""),
        "failure_reason": reason, "first_bad_step": bad.get("step", ""),
        "first_bad_probe_phase": bad.get("probe_phase", ""), "first_bad_contact_left": bad.get("contact_left", ""),
        "first_bad_contact_right": bad.get("contact_right", ""), "first_bad_contact_state": bad.get("contact_state", ""),
        "preprobe_step": pre.get("step", ""), "preprobe_contact_left": pre.get("contact_left", ""),
        "preprobe_contact_right": pre.get("contact_right", ""), "preprobe_contact_state": pre.get("contact_state", ""),
        "aperture": pre.get("gripper_opening", ""), "object_pose": json.dumps({k: pre.get(k, "") for k in ("object_x_priv", "object_y_priv", "object_z_priv", "object_qw_priv", "object_qx_priv", "object_qy_priv", "object_qz_priv")}, sort_keys=True),
        "ee_pose": json.dumps({k: pre.get(k, "") for k in ("eef_x", "eef_y", "eef_z")}, sort_keys=True),
        "bilateral_contact": bad.get("contact_state", pre.get("contact_state", "")),
        "contact_retained": row.get("contact_retained", ""), "completed_primary_branches": row.get("completed_primary_branches", ""),
        "probe_telemetry": str(telemetry) if telemetry else "", "worker_returncode": run_record.get("returncode", ""),
        "job_dir": str(job_dir),
    }


def scan_admission() -> dict[str, list[dict]]:
    candidates = freeze_candidates()
    selected: dict[str, list[dict]] = {str(t): [] for t in TASKS}
    blocked = []
    for task in TASKS:
        scan_path = OUT / f"TASK{task}_ADMISSION_SCAN.csv"
        # Resume safely after an already completed frozen-budget scan.  This
        # avoids rerunning simulator admission for task1 after its budget was
        # exhausted, while preserving the exact deterministic order.
        scan_rows = read_csv(scan_path) if scan_path.exists() else []
        scanned_ids = {r.get("context_id") for r in scan_rows}
        for old in scan_rows:
            if old.get("PASS/FAIL") == "PASS" and old.get("selection_accepted", "False").lower() == "true":
                selected[str(task)].append(next(c for c in candidates[str(task)] if c["context_id"] == old["context_id"]))
        for c in candidates[str(task)]:
            if len(selected[str(task)]) >= 3:
                break
            if c["context_id"] in scanned_ids:
                continue
            print(json.dumps({"event": "ADMISSION_SCAN_START", "task": task, "order_index": c["order_index"], "context": c["context_id"]}), flush=True)
            job_dir = OUT / "ADMISSION_SELECTION_RUNS" / f"task{task}" / f"{c['order_index']:02d}_{c['context_id']}"
            rec = run_worker(job_dir, task, c["context_id"], [], ADMISSION_TIMEOUT_S)
            detail = admission_detail(job_dir, task, c["context_id"], rec)
            row = {**c, **detail, "selection_accepted": detail["PASS/FAIL"] == "PASS" and len(selected[str(task)]) < 3}
            scan_rows.append(row)
            if row["selection_accepted"]:
                selected[str(task)].append(c)
            write_csv(OUT / f"TASK{task}_ADMISSION_SCAN.csv", scan_rows)
            print(json.dumps({"event": "ADMISSION_SCAN_DONE", "task": task, "order_index": c["order_index"], "status": row["PASS/FAIL"], "accepted_count": len(selected[str(task)])}), flush=True)
        write_csv(OUT / f"TASK{task}_ADMISSION_SCAN.csv", scan_rows)
        if len(selected[str(task)]) < 3:
            blocked.append({"task": task, "status": "TASK_ADMISSION_RATE_TOO_LOW", "selected": selected[str(task)], "scanned": len(scan_rows), "max_admission_candidates_per_task": MAX_CANDIDATES})
            print(json.dumps({"event": "TASK_ADMISSION_RATE_TOO_LOW", "task": task, "selected": len(selected[str(task)]), "scanned": len(scan_rows)}), flush=True)
    write_json(OUT / "REPLACEMENT_SELECTED_CONTEXTS.json", {"selection_rule": "FIRST_3_PASSES_IN_FROZEN_ORDER", "selected_contexts": selected, "candidate_order_frozen": "YES", "selected_utc": now(), "blocked_tasks": blocked})
    write_json(OUT / "REPLACEMENT_SELECTION_BLOCKED.json", {"status": "PARTIAL_SELECTION_SCAN", "blocked_tasks": blocked, "max_admission_candidates_per_task": MAX_CANDIDATES})
    return selected


def freeze_manifest(selected: dict[str, list[dict]]) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contexts = {c["context_id"]: c for c in protocol["train_context_population"]}
    selected_flat = []
    branch_jobs = []
    for task in TASKS:
        for c in selected[str(task)]:
            source = contexts[c["context_id"]]
            samples = {int(s["stratum_index"]): float(s["requested_force_N"]) for s in source["continuous_force_samples"]}
            entry = {**c, "setpoint_policy_source": "HISTORICAL_CONTEXT_LEVELS", "historical_levels": {"LOW": samples[1], "MID": samples[3], "HIGH": samples[5]}}
            selected_flat.append(entry)
            for repeat in REPEATS:
                specs = [{"force_N": samples[ROLE_LEVEL[role]], "repeat_index": repeat - 1, "branch_label": f"REPLACEMENT_{role}_F{samples[ROLE_LEVEL[role]]:.8f}".replace(".", "p") + f"_R{repeat}"} for role in ROLES]
                branch_jobs.append({"task": task, "context_id": c["context_id"], "repeat": repeat, "specs": specs})
    manifest = {
        "artifact": "REPLACEMENT_VALIDATION_CONTEXT_MANIFEST", "created_utc": now(), "status": "FROZEN_BEFORE_FORCE_BRANCHES",
        "candidate_order_frozen": "YES", "selection_rule": "FIRST_3_PASSES_IN_FROZEN_ORDER", "max_admission_candidates_per_task": MAX_CANDIDATES,
        "admission_only_completed_before_branches": "YES", "selected_contexts_per_task": 3, "selected_contexts": selected_flat,
        "tasks": list(ALL_TASKS), "replacement_tasks": list(TASKS), "roles": list(ROLES), "role_to_historical_level": ROLE_LEVEL,
        "repeats_per_level": 2, "expected_replacement_branches": 54, "expected_total_valid_branches_with_task0": 72,
        "setpoint_policy": "historical context level1/level3/level5, mapped LOW/MID/HIGH; no outcome-dependent modification",
        "branch_jobs": branch_jobs, "task0_reuse": {"valid_branches": 18, "source": str(OUT / "CURRENT_VALIDATION_BRANCHES.csv"), "rerun": "NO"},
        "execution_contract": {"wrapper": str(WRAPPER), "wrapper_sha256": sha256(WRAPPER), "runner": str(RUNNER), "runner_sha256": sha256(RUNNER), "same_as_task0": "YES", "fresh_process_per_context_repeat": "YES", "probe_unchanged": "YES", "controller_changed": "NO", "arm_changed": "NO"},
        "candidate_order_hashes": {str(t): sha256(OUT / f"TASK{t}_CANDIDATE_ORDER.json") for t in TASKS},
        "admission_scan_hashes": {str(t): sha256(OUT / f"TASK{t}_ADMISSION_SCAN.csv") for t in TASKS},
    }
    path = OUT / "REPLACEMENT_VALIDATION_CONTEXT_MANIFEST.json"
    write_json(path, manifest)
    digest = sha256(path)
    (OUT / "REPLACEMENT_VALIDATION_CONTEXT_MANIFEST.sha256").write_text(digest + "  REPLACEMENT_VALIDATION_CONTEXT_MANIFEST.json\n", encoding="utf-8")
    manifest["manifest_sha256"] = digest
    # Rewrite with the embedded digest, then write the final digest sidecar.
    write_json(path, manifest)
    digest = sha256(path)
    (OUT / "REPLACEMENT_VALIDATION_CONTEXT_MANIFEST.sha256").write_text(digest + "  REPLACEMENT_VALIDATION_CONTEXT_MANIFEST.json\n", encoding="utf-8")
    manifest["manifest_sha256"] = digest
    return manifest


def trace_metrics(path: Path) -> dict:
    rows = read_csv(path) if path.exists() else []
    window = [r for r in rows if r.get("phase") in {"branch_hold", "lift"}]
    native = [float(r["measured_force_N"]) for r in window if r.get("measured_force_N", "") not in ("", None) and math.isfinite(float(r["measured_force_N"]))]
    obj = [float(r["F_obj_bilateral_n"]) for r in window if r.get("F_obj_bilateral_n", "") not in ("", None) and math.isfinite(float(r["F_obj_bilateral_n"]))]
    eff = [float(r["F_target_eff_n"]) for r in window if r.get("F_target_eff_n", "") not in ("", None) and math.isfinite(float(r["F_target_eff_n"]))]
    n = max(1, math.ceil(0.05 * len(obj)))
    arm = [{k: r.get(k, "") for k in ("step", "phase", "cmd_x", "cmd_y", "cmd_z")} for r in rows]
    return {"telemetry_path": str(path), "telemetry_rows": len(rows), "window_rows": len(obj), "native_measured_mean_N": statistics.fmean(native) if native else None, "object_filtered_true_force_mean_N": statistics.fmean(obj) if obj else None, "object_filtered_true_force_top5_N": statistics.fmean(sorted(obj)[-n:]) if obj else None, "object_filtered_true_force_peak_N": max(obj) if obj else None, "object_filtered_force_exposure_Ns": sum(obj) * 0.05 if obj else None, "effective_target_mean_N": statistics.fmean(eff) if eff else None, "bilateral_contact_fraction": statistics.fmean([int(r.get("contact_state") == "bilateral") for r in window]) if window else None, "arm_action_hash": stable_hash(arm), "missing_primary_telemetry": int(not obj)}


def branch_rows_for_job(job_dir: Path, task: int, job: dict) -> tuple[list[dict], bool]:
    td = job_dir / f"task{task}"
    bp, cp, pp = td / "branches.csv", td / "strict_preprobe_capture.csv", td / "parity.csv"
    branches = read_csv(bp) if bp.exists() else []
    captures = {r["context_id"]: r for r in read_csv(cp)} if cp.exists() else {}
    parities = read_csv(pp) if pp.exists() else []
    expected = {s["branch_label"]: s for s in job["specs"]}
    output = []
    for b in branches:
        label = b.get("branch_label", "")
        if label not in expected:
            continue
        tp = Path(b.get("telemetry_path", ""))
        tm = trace_metrics(tp)
        cap = captures.get(job["context_id"], {})
        parity = next((p for p in parities if p.get("branch_label") == label and p.get("is_replay_diagnostic", "0") == "0"), {})
        role = next((r for r in ROLES if label.startswith(f"REPLACEMENT_{r}_")), "UNKNOWN")
        output.append({"task": task, "root_id": b.get("root_id", ""), "context_id": job["context_id"], "role": role, "repeat": int(float(b.get("repeat_index", job["repeat"] - 1))) + 1, "setpoint_N": float(b.get("requested_force_N", expected[label]["force_N"])), "requested_force_N": float(b.get("requested_force_N", expected[label]["force_N"])), **tm, "bilateral_contact_fraction": tm.get("bilateral_contact_fraction"), "lift_success": int(float(b.get("lift_success", 0) or 0)), "hold_success_available": 0, "hold_bilateral_contact_fraction": tm.get("bilateral_contact_fraction"), "drop": int(float(b.get("dropped", 0) or 0) or float(b.get("lost_in_transit", 0) or 0)), "full_task_success": int(float(b.get("full_task_success_y", 0) or 0)), "branch_restore_parity": int(float(parity.get("parity_pass", 0) or 0)), "initial_state_hash": cap.get("initial_state_hash", ""), "query_state_hash": cap.get("query_state_hash", ""), "second_query_state_hash": cap.get("second_query_state_hash", ""), "arm_action_hash": tm.get("arm_action_hash", ""), "telemetry_path": str(tp), "effective_target_available": int(tm.get("effective_target_mean_N") is not None)})
    valid = len(output) == 3 and all(r["missing_primary_telemetry"] == 0 and r["branch_restore_parity"] == 1 for r in output)
    return output, valid


def run_branches(manifest: dict) -> list[dict]:
    all_rows = []
    records = []
    for index, job in enumerate(manifest["branch_jobs"], 1):
        final_rows, final_valid = [], False
        attempts = []
        for attempt in range(1, MAX_BRANCH_ATTEMPTS + 1):
            print(json.dumps({"event": "BRANCH_JOB_START", "index": index, "total": 18, "task": job["task"], "context": job["context_id"], "repeat": job["repeat"], "attempt": attempt}), flush=True)
            jd = OUT / "REPLACEMENT_BRANCH_RUNS" / f"task{job['task']}" / job["context_id"] / f"repeat{job['repeat']}" / f"attempt{attempt}"
            rec = run_worker(jd, job["task"], job["context_id"], job["specs"], BRANCH_TIMEOUT_S)
            rows, valid = branch_rows_for_job(jd, job["task"], job)
            attempts.append({**rec, "attempt": attempt, "branch_rows": len(rows), "branch_parity_valid": valid})
            if valid:
                final_rows, final_valid = rows, True
                break
        if not final_valid:
            write_json(OUT / "BRANCH_EXECUTION_BLOCKED.json", {"status": "BRANCH_PARITY_OR_TELEMETRY_FAILURE", "job": job, "attempts": attempts})
            raise RuntimeError(f"branch job failed parity/telemetry after {MAX_BRANCH_ATTEMPTS}: {job}")
        records.append({"job": job, "attempts": attempts, "selected_attempt": attempts[-1]["attempt"]})
        all_rows.extend(final_rows)
        dest = OUT / f"TASK{job['task']}_REPLACEMENT_TRACES" / job["context_id"] / f"repeat{job['repeat']}"
        dest.mkdir(parents=True, exist_ok=True)
        src = Path(final_rows[0]["telemetry_path"]).parent if final_rows else None
        if src and src.exists():
            for item in src.glob("*.csv"):
                shutil.copy2(item, dest / item.name)
        for filename in ("context.csv", "parity.csv", "strict_preprobe_capture.csv", "result.json"):
            source = (Path(attempts[-1]["job_dir"]) / f"task{job['task']}" / filename)
            if source.exists(): shutil.copy2(source, dest / filename)
        print(json.dumps({"event": "BRANCH_JOB_DONE", "index": index, "task": job["task"], "context": job["context_id"], "repeat": job["repeat"], "rows": len(final_rows)}), flush=True)
    write_json(OUT / "REPLACEMENT_BRANCH_EXECUTION_RECORDS.json", {"status": "COMPLETE", "jobs": records, "valid_branch_rows": len(all_rows), "expected": 54})
    return all_rows


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__); out = [0.0] * len(values); i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]: j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j): out[order[k]] = rank
        i = j
    return out


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 2: return None
    x, y = ranks(a), ranks(b); xm, ym = statistics.fmean(x), statistics.fmean(y)
    xx, yy = sum((v - xm) ** 2 for v in x), sum((v - ym) ** 2 for v in y)
    return sum((u - xm) * (v - ym) for u, v in zip(x, y)) / math.sqrt(xx * yy) if xx and yy else None


def aggregate(replacement_rows: list[dict]) -> None:
    task0 = read_csv(OUT / "CURRENT_VALIDATION_BRANCHES.csv")
    rows = []
    for r in task0:
        rows.append({**r, "task": int(r["task"]), "repeat": int(r["repeat"]), "setpoint_N": float(r["setpoint_N"]), "object_filtered_true_force_mean_N": float(r["object_filtered_true_force_mean_N"]), "object_filtered_true_force_top5_N": float(r["object_filtered_true_force_top5_N"]), "object_filtered_force_exposure_Ns": float(r["object_filtered_force_exposure_Ns"]), "full_task_success": int(float(r["full_task_success"])), "branch_restore_parity": int(float(r["branch_restore_parity"]))})
    rows.extend(replacement_rows)
    fields = ["task", "root_id", "context_id", "role", "repeat", "setpoint_N", "requested_force_N", "effective_target_mean_N", "effective_target_available", "native_measured_mean_N", "object_filtered_true_force_mean_N", "object_filtered_true_force_top5_N", "object_filtered_true_force_peak_N", "object_filtered_force_exposure_Ns", "bilateral_contact_fraction", "lift_success", "hold_success_available", "hold_bilateral_contact_fraction", "drop", "full_task_success", "branch_restore_parity", "initial_state_hash", "query_state_hash", "second_query_state_hash", "arm_action_hash", "telemetry_path", "telemetry_rows", "window_rows", "missing_primary_telemetry"]
    write_csv(OUT / "FINAL_CONTEXT_LEVEL_MAPPING.csv", [], [])
    write_csv(OUT / "FINAL_VALIDATION_BRANCHES.csv", rows, fields)
    by_context = defaultdict(list)
    for r in rows: by_context[r["context_id"]].append(r)
    context_map, pairwise = [], []
    for cid, rr in sorted(by_context.items()):
        agg = {}
        for role in ROLES:
            q = [r for r in rr if r["role"] == role]
            agg[role] = {"setpoint_N": statistics.fmean([float(r["setpoint_N"]) for r in q]) if q else None, "mean": statistics.fmean([float(r["object_filtered_true_force_mean_N"]) for r in q]) if q else None, "top5": statistics.fmean([float(r["object_filtered_true_force_top5_N"]) for r in q]) if q else None, "exposure": statistics.fmean([float(r["object_filtered_force_exposure_Ns"]) for r in q]) if q else None, "success": statistics.fmean([int(r["full_task_success"]) for r in q]) if q else None, "n": len(q)}
        cr = {"task": int(rr[0]["task"]), "root_id": rr[0].get("root_id", ""), "context_id": cid}
        for role in ROLES:
            for metric in ("setpoint_N", "mean", "top5", "exposure", "success", "n"): cr[f"{role}_{metric}"] = agg[role][metric]
        for metric in ("mean", "top5", "exposure"):
            values = [agg[r][metric] for r in ROLES]
            cr[f"{metric}_rank_spearman"] = spearman([1, 2, 3], values) if all(v is not None for v in values) else None
            for i, j, name in ((0, 1, "low_mid"), (1, 2, "mid_high"), (0, 2, "low_high")):
                ordered = bool(values[i] <= values[j]) if values[i] is not None and values[j] is not None else False
                cr[f"{metric}_{name}_ordered"] = int(ordered)
                pairwise.append({"task": int(rr[0]["task"]), "context_id": cid, "metric": metric, "comparison": f"{ROLES[i]}_vs_{ROLES[j]}", "low_value": values[i], "high_value": values[j], "ordered": int(ordered), "difference_high_minus_low": values[j] - values[i] if values[i] is not None and values[j] is not None else None, "classification": "MISSING" if values[i] is None or values[j] is None else ("SMALL_OVERLAP" if abs(values[j] - values[i]) <= 0.25 else ("ORDERED" if values[j] >= values[i] else "LARGE_DETERMINISTIC_REVERSAL"))})
        context_map.append(cr)
    write_csv(OUT / "FINAL_CONTEXT_LEVEL_MAPPING.csv", context_map)
    main_pairs = [p for p in pairwise if p["metric"] == "mean"]
    violations = sum(not p["ordered"] for p in main_pairs)
    by_task = {}
    for task in ALL_TASKS:
        cs = [c for c in context_map if c["task"] == task]
        by_task[str(task)] = {"contexts": len(cs), "levels": {role: {"measured_mean_N": statistics.fmean([c[f"{role}_mean"] for c in cs]) if cs else None, "measured_top5_N": statistics.fmean([c[f"{role}_top5"] for c in cs]) if cs else None, "exposure_Ns": statistics.fmean([c[f"{role}_exposure"] for c in cs]) if cs else None, "success_rate": statistics.fmean([c[f"{role}_success"] for c in cs]) if cs else None} for role in ROLES}, "within_context_rank_spearman": {m: statistics.fmean([c[f"{m}_rank_spearman"] for c in cs]) if cs and all(c[f"{m}_rank_spearman"] is not None for c in cs) else None for m in ("mean", "top5", "exposure")}}
        by_task[str(task)]["mapping_valid"] = bool(cs and all(by_task[str(task)]["levels"][r]["measured_mean_N"] is not None for r in ROLES) and by_task[str(task)]["levels"]["LOW"]["measured_mean_N"] <= by_task[str(task)]["levels"]["MID"]["measured_mean_N"] <= by_task[str(task)]["levels"]["HIGH"]["measured_mean_N"])
        by_task[str(task)]["pairwise_violation_rate_mean"] = sum(not p["ordered"] for p in main_pairs if p["task"] == task) / (3 * len(cs)) if cs else None
    overall = {m: statistics.fmean([c[f"{m}_rank_spearman"] for c in context_map]) if context_map and all(c[f"{m}_rank_spearman"] is not None for c in context_map) else None for m in ("mean", "top5", "exposure")}
    write_json(OUT / "FINAL_TASK_LEVEL_MAPPING.json", {"tasks": by_task, "overall": overall, "task0_reused": True})
    write_json(OUT / "FINAL_PAIRWISE_MONOTONICITY.json", {"primary_metric": "object_filtered_true_force_mean_N", "total_pairwise_comparisons": len(main_pairs), "pairwise_ordering_correct": len(main_pairs) - violations, "pairwise_monotonic_violations": violations, "pairwise_violation_rate": violations / len(main_pairs) if main_pairs else None, "small_overlap": sum(p["classification"] == "SMALL_OVERLAP" for p in main_pairs), "large_deterministic_reversals": sum(p["classification"] == "LARGE_DETERMINISTIC_REVERSAL" for p in main_pairs), "all_metrics": {m: {"total": sum(p["metric"] == m for p in pairwise), "violations": sum(p["metric"] == m and not p["ordered"] for p in pairwise), "violation_rate": sum(p["metric"] == m and not p["ordered"] for p in pairwise) / sum(p["metric"] == m for p in pairwise) if sum(p["metric"] == m for p in pairwise) else None} for m in ("mean", "top5", "exposure")}, "pairs": pairwise})
    success = {role: statistics.fmean([float(r["full_task_success"]) for r in rows if r["role"] == role]) if [r for r in rows if r["role"] == role] else None for role in ROLES}
    all_valid = len(rows) == 72 and all(c["mapping_valid"] for c in by_task.values()) and all(int(r.get("branch_restore_parity", 0)) == 1 for r in rows)
    validity = "YES" if all_valid and overall["mean"] is not None and overall["top5"] is not None and overall["exposure"] is not None and violations / len(main_pairs) <= 0.10 else ("PARTIAL" if any(c["mapping_valid"] for c in by_task.values()) else "NO")
    report = ["# Final current-runtime continuous setpoint mapping validation", "", f"FINAL_STATUS = {validity}", "", "Selection was blind: the first three admission PASS contexts in each pre-frozen candidate order were selected. The nine previously failed historical contexts were not restored.", "", f"Valid branches: {len(rows)}/72 (task0 reused: 18; replacement: {len(rows)-18}).", f"Within-context rank Spearman (mean/top5/exposure): {overall['mean']} / {overall['top5']} / {overall['exposure']}.", f"Primary mean-force pairwise violations: {violations}/{len(main_pairs)} = {violations/len(main_pairs) if main_pairs else None}.", "", "Primary measured signal is object-filtered bilateral force in the frozen branch_hold + lift window; native force and effective target are retained as telemetry. Exact Newton tracking is not claimed.", "", "No controller, setpoint policy, probe, arm trajectory, posterior, or Expected Utility was changed. LOW_LEVEL_EXECUTION_FROZEN is YES only when the four-task validation is complete and valid."]
    (OUT / "FINAL_CURRENT_RUNTIME_VALIDATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "FINAL_VALIDATION_SUMMARY.json", {"final_status": validity, "task0_valid_branches": 18, "replacement_valid_branches": len(rows) - 18, "total_valid_branches": len(rows), "target_valid_branches": 72, "within_context_mean_force_spearman": overall["mean"], "within_context_top5_spearman": overall["top5"], "within_context_exposure_spearman": overall["exposure"], "task_mapping_valid": {t: by_task[t]["mapping_valid"] for t in by_task}, "success_rates": success, "controller_changed": "NO", "setpoints_changed": "NO", "probe_changed": "NO", "raw_arm_changed": "NO", "final_force_semantics": "CONTINUOUS_GRASP_FORCE_SETPOINT", "exact_newton_tracking_claim": "NO"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--branches", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.scan:
        scan_admission()
    elif args.branches:
        selected = json.loads((OUT / "REPLACEMENT_SELECTED_CONTEXTS.json").read_text(encoding="utf-8"))["selected_contexts"]
        manifest = freeze_manifest(selected)
        run_branches(manifest)
    elif args.analyze:
        replacement = []
        records = json.loads((OUT / "REPLACEMENT_BRANCH_EXECUTION_RECORDS.json").read_text(encoding="utf-8"))
        manifest = json.loads((OUT / "REPLACEMENT_VALIDATION_CONTEXT_MANIFEST.json").read_text(encoding="utf-8"))
        for jobrec in records["jobs"]:
            job = jobrec["job"]; attempt = jobrec["selected_attempt"]; jd = OUT / "REPLACEMENT_BRANCH_RUNS" / f"task{job['task']}" / job["context_id"] / f"repeat{job['repeat']}" / f"attempt{attempt}"
            rs, valid = branch_rows_for_job(jd, job["task"], job)
            if not valid: raise RuntimeError(f"selected branch attempt no longer valid: {job}")
            replacement.extend(rs)
        if len(replacement) != 54: raise RuntimeError(f"expected 54 replacement branches, found {len(replacement)}")
        aggregate(replacement)
    else:
        ap.error("choose --scan, --branches, or --analyze")


if __name__ == "__main__":
    main()
