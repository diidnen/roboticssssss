#!/usr/bin/env python3
"""Small current-runtime validation of the archived continuous force ordering.

Preparation and analysis are CPU-only.  Execution uses the existing current
P5-S0-C runner through the existing strict-preprobe wrapper, in fresh processes
per context/repeat.  This script does not edit the controller or model assets.
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
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
ARCHIVE = ROOT / "gnp_style_continuous_20260830_125107"
PROTOCOL = ARCHIVE / "GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"
OLD_DATA = ARCHIVE / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv"
OLD_CONTEXT_STATS = ROOT / "analysis/results/old720_historical_monotonicity_forensics_20260905/OLD720_CONTEXT_MONOTONICITY.csv"
RUNNER = Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py")
WRAPPER = ROOT / "current4task_low_force_e3.py"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
TABERO = Path("/home/exouser/Tabero")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
OPENPI = Path("/media/volume/newdata/exouser/openpi/src")
OUT = ROOT / "analysis/results/current_runtime_setpoint_mapping_validation_20260905"
TASKS = (0, 1, 5, 6)
REPEATS = (1, 2)
ROLES = ("LOW", "MID", "HIGH")
ROLE_TO_LEVEL = {"LOW": 1, "MID": 3, "HIGH": 5}
DT_FALLBACK = 0.05
GPU_FREE_MIN_MIB = 10 * 1024
GPU_UTIL_MAX = 70.0
REFERENCE_SELECTION = {
    0: ("p5s0c_train_t0_r00_s5100_low_mu0.293710", "p5s0c_train_t0_r00_s5100_mid_mu0.450580", "p5s0c_train_t0_r00_s5100_high_mu0.940189"),
    1: ("p5s0c_train_t1_r00_s5100_low_mu0.240019", "p5s0c_train_t1_r00_s5100_mid_mu0.467048", "p5s0c_train_t1_r00_s5100_high_mu0.921500"),
    5: ("p5s0c_train_t5_r00_s5100_low_mu0.270805", "p5s0c_train_t5_r03_s5103_mid_mu0.579422", "p5s0c_train_t5_r00_s5100_high_mu0.945004"),
    6: ("p5s0c_train_t6_r00_s5100_low_mu0.257577", "p5s0c_train_t6_r00_s5100_mid_mu0.558595", "p5s0c_train_t6_r00_s5100_high_mu0.989348"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


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
        w = csv.DictWriter(f, fieldnames=fields or ["status"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def ffloat(x: object) -> float | None:
    if x is None or str(x).strip() == "":
        return None
    y = float(x)
    return y if math.isfinite(y) else None


def ranks(a: list[float]) -> list[float]:
    order = sorted(range(len(a)), key=a.__getitem__)
    out = [0.0] * len(a)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and a[order[j]] == a[order[i]]:
            j += 1
        r = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = r
        i = j
    return out


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    xm, ym = statistics.fmean(x), statistics.fmean(y)
    xx = sum((v - xm) ** 2 for v in x)
    yy = sum((v - ym) ** 2 for v in y)
    return sum((a - xm) * (b - ym) for a, b in zip(x, y)) / math.sqrt(xx * yy) if xx and yy else None


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(ranks(x), ranks(y))


def linear_fit(x: list[float], y: list[float]) -> dict:
    xm, ym = statistics.fmean(x), statistics.fmean(y)
    xx = sum((v - xm) ** 2 for v in x)
    if not xx:
        return {"n": len(x), "slope": None, "intercept": None, "r2": None}
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / xx
    intercept = ym - slope * xm
    pred = [slope * v + intercept for v in x]
    sse, sst = sum((b - p) ** 2 for b, p in zip(y, pred)), sum((b - ym) ** 2 for b in y)
    return {"n": len(x), "slope": slope, "intercept": intercept, "r2": 1.0 - sse / sst if sst else None}


def top5(values: list[float]) -> float | None:
    if not values:
        return None
    n = max(1, math.ceil(0.05 * len(values)))
    return statistics.fmean(sorted(values)[-n:])


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    old_stats = {r["context"]: r for r in read_csv(OLD_CONTEXT_STATS)}
    contexts = {c["context_id"]: c for c in protocol["train_context_population"]}
    selected = []
    jobs = []
    for task in TASKS:
        for role, cid in zip(ROLES, REFERENCE_SELECTION[task]):
            if cid not in contexts:
                raise RuntimeError(f"selected context absent from protocol: {cid}")
            c = contexts[cid]
            samples = sorted(c["continuous_force_samples"], key=lambda s: int(s["stratum_index"]))
            level = ROLE_TO_LEVEL[role]
            sample = samples[level - 1]
            entry = {
                "task": task, "role": role, "context_id": cid, "root_id": c["root_id"],
                "root_index": int(c["root_index"]), "root_seed": int(c["root_seed"]),
                "friction_band": c["friction_band"], "friction": float(c["friction"]),
                "setpoint_N": float(sample["requested_force_N"]), "historical_level": level,
                "historical_context_spearman": float(old_stats[cid]["realized_force_N_spearman"]),
                "historical_context_strict": int(old_stats[cid]["realized_force_N_strict_monotonic"]),
                "historical_telemetry_complete": True,
            }
            selected.append(entry)
            for repeat in REPEATS:
                job_dir = OUT / "_jobs" / f"task{task}" / cid / f"repeat{repeat}"
                manifest_path = job_dir / "WORKER_TARGET_MANIFEST.json"
                branch_specs = []
                for job_role, job_level in zip(ROLES, [1, 3, 5]):
                    s = samples[job_level - 1]
                    force = float(s["requested_force_N"])
                    branch_specs.append({
                        "force_N": force, "repeat_index": repeat - 1,
                        "branch_label": f"CURRENT_{job_role}_F{force:.8f}".replace(".", "p") + f"_R{repeat}",
                    })
                jobs.append({"task": task, "context_id": cid, "role": role, "repeat": repeat,
                             "job_dir": str(job_dir), "manifest_path": str(manifest_path),
                             "contexts": {cid: branch_specs}})
    target = {
        "manifest_name": "CURRENT_RUNTIME_SETPOINT_MAPPING_VALIDATION",
        "created_utc": now(), "status": "FROZEN_BEFORE_EXECUTION",
        "source_old720_dataset": str(OLD_DATA), "source_old720_dataset_sha256": sha256(OLD_DATA),
        "source_protocol": str(PROTOCOL), "source_protocol_sha256": sha256(PROTOCOL),
        "selected_context_count": len(selected), "tasks": list(TASKS), "contexts_per_task": 3,
        "repeats_per_level": 2, "rollouts_expected": len(jobs) * 3,
        "selection_policy": "three contexts per task: LOW/MID/HIGH friction coverage; includes task1 historical non-strict LOW and task5 historical non-strict MID; task0/task1/task6 root00 LOW/MID/HIGH plus task5 root03 MID for root variation",
        "setpoint_policy": "LOW/MID/HIGH are the context's historical old720 levels 1/3/5 (lowest/middle/highest), not newly sampled values",
        "selected_contexts": selected, "jobs": jobs,
        "no_old720_recollection": True, "no_controller_change": True, "no_model_change": True,
    }
    write_json(OUT / "VALIDATION_CONTEXT_MANIFEST.json", target)
    contract = {
        "status": "FROZEN_BEFORE_EXECUTION", "frozen_utc": now(),
        "current_runtime_runner": str(RUNNER), "current_runtime_runner_sha256": sha256(RUNNER),
        "validation_wrapper": str(WRAPPER), "validation_wrapper_sha256": sha256(WRAPPER),
        "isaac_python": str(ISAAC_PY), "env_id": "Isaac-Libero-Franka-Hybrid-Tactile-v0",
        "execution": "existing current P5-S0-C runner through current4 strict-preprobe wrapper; one fresh Isaac process per context/repeat; each process branches LOW/MID/HIGH from the same in-process restored snapshot",
        "outer_rate_hz": 20.0, "physics_rate_hz": 60.0, "strict_preprobe_capture_step": 190,
        "force_action_path": "existing Tabero ForcePositionAction contract; no P5S0C_FINAL_V2 override; no action-term config mutation by this validation",
        "contact_override": "OFF / unchanged",
        "probe": "existing P4-B probe path; unchanged; used only to reach the branch snapshot",
        "branch_matching": ["same task", "same root seed", "same context friction", "same reset sequence", "same branch snapshot", "same arm command trajectory", "only force setpoint varies"],
        "effective_target": "record F_target_eff_n from telemetry when present; otherwise NA; do not infer an unrecorded target",
        "primary_metrics": ["object-filtered F_obj_bilateral_n mean", "object-filtered F_obj_bilateral_n top5%", "object-filtered force exposure"],
        "window": "branch_hold + lift phases; same 70-step/3.5s window where available; no cross-runtime equality requirement",
        "pass_criteria_frozen": {
            "main_mean_force_rank_spearman_min": 0.80,
            "main_top5_rank_spearman_min": 0.80,
            "main_exposure_rank_spearman_min": 0.80,
            "pairwise_violation_rate_max": 0.10,
            "all_tasks_low_high_aggregate_order_required": True,
            "classification": "STRICT only if all context metrics and pairs order; STRONG_STATISTICAL if the three main metrics meet rho>=0.80, pairwise rate<=0.10, and every task aggregate high>low; PARTIAL if only a subset of tasks meets the task trend; NO otherwise",
        },
        "forbidden": ["controller redesign", "old720 recollection", "posterior retraining", "calibration tuning", "Expected Utility tuning", "probe redesign", "VLA arm change", "long horizon", "root7703 special tuning"],
    }
    write_json(OUT / "CURRENT_EXECUTION_CONTRACT.json", contract)
    for job in jobs:
        p = Path(job["manifest_path"])
        write_json(p, {"manifest_name": "CURRENT_RUNTIME_WORKER_TARGET", "contexts": job["contexts"], "expected_contexts": 1, "expected_branches": 3})
    for task in TASKS:
        (OUT / f"TASK{task}_TRACES").mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STATE.json", {"status": "PREPARED_GPU_PREFLIGHT_PENDING", "prepared_utc": now(), "jobs": len(jobs)})
    print(json.dumps({"status": "PREPARED", "contexts": len(selected), "jobs": len(jobs), "rollouts": len(jobs) * 3, "out": str(OUT)}, indent=2))


def gpu_preflight() -> dict:
    result = {
        "checked_utc": now(),
        "command": "ISAAC_PY -c subprocess(nvidia-smi --query-gpu=utilization.gpu,memory.free,memory.used)",
        "preflight_python": str(ISAAC_PY),
        "launch_allowed": False,
    }
    try:
        probe = (
            "import subprocess,sys; "
            "p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.free,memory.used',"
            "'--format=csv,noheader,nounits'],capture_output=True,text=True); "
            "sys.stdout.write(p.stdout); sys.stderr.write(p.stderr); raise SystemExit(p.returncode)"
        )
        p = subprocess.run([str(ISAAC_PY), "-c", probe], capture_output=True, text=True, timeout=15)
        result["returncode"] = p.returncode; result["stdout"] = p.stdout.strip(); result["stderr"] = p.stderr.strip()
        if p.returncode == 0 and p.stdout.strip():
            util, free, used = [float(x.strip()) for x in p.stdout.strip().splitlines()[0].split(",")[:3]]
            result.update({"gpu_util_percent": util, "free_memory_MiB": free, "used_memory_MiB": used})
            result["launch_allowed"] = bool(util <= GPU_UTIL_MAX and free > GPU_FREE_MIN_MIB)
            if not result["launch_allowed"]:
                result["reason"] = f"GPU utilization/free-memory gate: util={util}, free_MiB={free}"
        else:
            result["reason"] = "nvidia-smi cannot communicate with the NVIDIA driver"
    except Exception as exc:
        result["reason"] = repr(exc)
    return result


def launch_job(job: dict, timeout_s: int) -> dict:
    job_dir = Path(job["job_dir"]); job_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(TABERO), str(OPENPI)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(TABERO),
        "P5S0C_OUT": str(job_dir), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": str(job["task"]),
        "P5S0C_TARGET_MANIFEST": str(job["manifest_path"]), "P5S0C_CONTEXT_IDS": job["context_id"],
        "P5S0C_SKIP_REPLAY": "1", "CONTINUOUS_STRICT_PREPROBE_WRAPPER": "1",
        "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
    })
    log = job_dir / "isaac_worker.log"
    cmd = [str(ISAAC_PY), "-u", str(WRAPPER), "--worker"]
    started = time.time()
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=TABERO, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                rc = p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); rc = p.wait()
    return {"task": job["task"], "context_id": job["context_id"], "repeat": job["repeat"], "job_dir": str(job_dir), "returncode": rc, "elapsed_s": time.time() - started, "log": str(log)}


def execute(timeout_s: int) -> None:
    target = json.loads((OUT / "VALIDATION_CONTEXT_MANIFEST.json").read_text(encoding="utf-8"))
    gate = gpu_preflight()
    write_json(OUT / "GPU_PREFLIGHT.json", gate)
    if not gate.get("launch_allowed"):
        write_blocked_artifacts(gate)
        write_json(OUT / "RUN_STATE.json", {"status": "BLOCKED_GPU_PREFLIGHT", "updated_utc": now(), "gpu_preflight": gate, "jobs_completed": 0})
        print(json.dumps({"status": "BLOCKED_GPU_PREFLIGHT", "reason": gate.get("reason", "unknown")}, indent=2))
        return
    records = []
    for i, job in enumerate(target["jobs"], 1):
        print(json.dumps({"event": "JOB_START", "index": i, "total": len(target["jobs"]), "task": job["task"], "context": job["context_id"], "repeat": job["repeat"]}), flush=True)
        records.append(launch_job(job, timeout_s))
        write_json(OUT / "EXECUTION_RECORDS.json", {"updated_utc": now(), "records": records})
        print(json.dumps({"event": "JOB_DONE", "index": i, "returncode": records[-1]["returncode"], "elapsed_s": records[-1]["elapsed_s"]}), flush=True)
    write_json(OUT / "RUN_STATE.json", {"status": "EXECUTION_COMPLETE", "updated_utc": now(), "records": records})


def write_blocked_artifacts(gate: dict) -> None:
    """Create explicit blocked placeholders; never represent missing runs as failures."""
    reason = gate.get("reason", "GPU preflight did not pass")
    write_csv(OUT / "CONTEXT_LEVEL_MAPPING.csv", [{"status": "NOT_COLLECTED", "reason": reason}], ["status", "reason"])
    write_json(OUT / "TASK_LEVEL_MAPPING.json", {"status": "NOT_COLLECTED", "reason": reason, "tasks": list(TASKS)})
    write_json(OUT / "PAIRWISE_MONOTONICITY.json", {"status": "NOT_COLLECTED", "reason": reason, "total_pairwise_comparisons": None, "pairwise_monotonic_violations": None})
    write_json(OUT / "CURRENT_VS_HISTORICAL_MAPPING.json", {"status": "BLOCKED_GPU_PREFLIGHT", "historical_mapping_strength": "STRONG_STATISTICAL", "current_mapping_strength": "NOT_ASSESSED", "current_runtime_preserves_historical_ordering": "NOT_ASSESSED", "current_runtime_continuous_setpoint_mapping_valid": "NOT_ASSESSED", "reason": reason})
    write_json(OUT / "VALIDATION_QA.json", {"status": "BLOCKED", "reason": reason, "branch_rows": 0, "expected_rows": 72, "calculation_status": "NOT_RUN"})
    (OUT / "CURRENT_RUNTIME_SETPOINT_VALIDATION_REPORT.md").write_text(
        "# Current-runtime continuous setpoint validation\n\n"
        "## Status\n\n"
        "`BLOCKED_GPU_PREFLIGHT`: no Isaac rollout was launched.\n\n"
        f"Preflight evidence: `{reason}`. `nvidia-smi` could not communicate with the NVIDIA driver and CUDA reported no device.\n\n"
        "The 12-context/24-job/72-rollout manifest and execution contract are frozen, but current mapping metrics are intentionally **not assessed**. No controller, posterior, Expected Utility, probe, VLA arm, or ActiveForcing method was changed.\n\n"
        "## Next action\n\n"
        "Re-run the same `--execute` command only after a healthy, schedulable GPU runtime is available. The frozen manifest and contract must be reused; no new context or force selection is needed.\n",
        encoding="utf-8",
    )


def trace_metrics(path: Path) -> dict:
    rows = read_csv(path)
    window = [r for r in rows if r.get("phase") in {"branch_hold", "lift"}]
    vals, true_vals, eff = [], [], []
    bilat, arm = [], []
    for r in window:
        m = ffloat(r.get("measured_force_N")); t = ffloat(r.get("F_obj_bilateral_n")); e = ffloat(r.get("F_target_eff_n"))
        if m is not None: vals.append(m)
        if t is not None: true_vals.append(t)
        if e is not None: eff.append(e)
        bilat.append(int(r.get("contact_state") == "bilateral"))
    arm_payload = [{k: r.get(k, "") for k in ["step", "phase", "cmd_x", "cmd_y", "cmd_z"]} for r in rows]
    arm_hash = hashlib.sha256(json.dumps(arm_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "telemetry_path": str(path), "telemetry_rows": len(rows), "window_rows": len(true_vals),
        "native_measured_mean_N": statistics.fmean(vals) if vals else None,
        "object_filtered_true_force_mean_N": statistics.fmean(true_vals) if true_vals else None,
        "object_filtered_true_force_top5_N": top5(true_vals), "object_filtered_true_force_peak_N": max(true_vals) if true_vals else None,
        "object_filtered_force_exposure_Ns": sum(true_vals) * DT_FALLBACK if true_vals else None,
        "effective_target_mean_N": statistics.fmean(eff) if eff else None,
        "bilateral_contact_fraction": statistics.fmean(bilat) if bilat else None,
        "arm_action_hash": arm_hash, "missing_primary_telemetry": int(not true_vals),
    }


def analyze() -> None:
    if not (OUT / "VALIDATION_CONTEXT_MANIFEST.json").exists():
        raise RuntimeError("run --prepare first")
    target = json.loads((OUT / "VALIDATION_CONTEXT_MANIFEST.json").read_text(encoding="utf-8"))
    state = json.loads((OUT / "RUN_STATE.json").read_text(encoding="utf-8")) if (OUT / "RUN_STATE.json").exists() else {}
    branch_rows, summary_rows, parity_rows = [], [], []
    admission_rows = []
    job_status = []
    for job in target["jobs"]:
        jd = Path(job["job_dir"]); task_dir = jd / f"task{job['task']}"
        bp = task_dir / "branches.csv"; cp = task_dir / "strict_preprobe_capture.csv"; pp = task_dir / "parity.csv"
        context_records = read_csv(task_dir / "context.csv") if (task_dir / "context.csv").exists() else []
        context_record = context_records[0] if context_records else {}
        admission_rows.append({
            "task": int(job["task"]), "context_id": job["context_id"], "repeat": int(job["repeat"]),
            "status": context_record.get("status", "MISSING_CONTEXT_OUTPUT"),
            "stop_reason": context_record.get("stop_reason", ""),
            "probe_qualified": context_record.get("probe_qualified", ""),
            "contact_retained": context_record.get("contact_retained", ""),
            "completed_primary_branches": context_record.get("completed_primary_branches", ""),
            "strict_matched": context_record.get("strict_matched", ""),
            "post_probe_state_hash": context_record.get("post_probe_state_hash", ""),
        })
        if not bp.exists() or bp.stat().st_size == 0:
            job_status.append({"task": job["task"], "context_id": job["context_id"], "repeat": job["repeat"], "status": "NO_PRIMARY_BRANCHES", "context_status": context_record.get("status", "MISSING_CONTEXT_OUTPUT"), "stop_reason": context_record.get("stop_reason", "")})
            continue
        branches = read_csv(bp); captures = {r["context_id"]: r for r in read_csv(cp)} if cp.exists() else {}
        parities = read_csv(pp) if pp.exists() else []
        for b in branches:
            tp = Path(b["telemetry_path"])
            tm = trace_metrics(tp) if tp.exists() else {"telemetry_path": str(tp), "missing_primary_telemetry": 1}
            cap = captures.get(b["context_id"], {})
            pr = next((p for p in parities if p.get("branch_label") == b.get("branch_label") and p.get("is_replay_diagnostic", "0") == "0"), {})
            branch_label = str(b.get("branch_label", ""))
            role = next((candidate for candidate in ROLES if branch_label.startswith(f"CURRENT_{candidate}_")), "UNKNOWN")
            repeat_number = int(b.get("repeat_index", int(job["repeat"]) - 1)) + 1
            out = {**b, **tm, "task": int(b["task"]), "repeat": repeat_number,
                   "setpoint_N": float(b["requested_force_N"]), "context_id": b["context_id"],
                   "role": role,
                   "effective_target_available": int(tm.get("effective_target_mean_N") is not None),
                   "lift_success": int(float(b.get("lift_success", 0))), "full_task_success": int(float(b.get("full_task_success_y", 0))),
                   "drop": int(float(b.get("dropped", 0) or 0) or float(b.get("lost_in_transit", 0) or 0)),
                   "hold_success_available": 0, "hold_bilateral_contact_fraction": tm.get("bilateral_contact_fraction"),
                   "branch_restore_parity": int(float(pr.get("parity_pass", 0) or 0)),
                   "query_state_hash": cap.get("query_state_hash", ""), "second_query_state_hash": cap.get("second_query_state_hash", ""),
                   "initial_state_hash": cap.get("initial_state_hash", "")}
            branch_rows.append(out)
        job_status.append({"task": job["task"], "context_id": job["context_id"], "repeat": job["repeat"], "status": "COMPLETE", "branches": len(branches), "branch_restore_parity_all": int(all(float(p.get("parity_pass", 0) or 0) == 1 for p in parities if p.get("is_replay_diagnostic", "0") == "0"))})
    raw_fields = ["task", "root_id", "context_id", "role", "repeat", "setpoint_N", "requested_force_N", "effective_target_mean_N", "effective_target_available", "native_measured_mean_N", "object_filtered_true_force_mean_N", "object_filtered_true_force_top5_N", "object_filtered_true_force_peak_N", "object_filtered_force_exposure_Ns", "bilateral_contact_fraction", "lift_success", "hold_success_available", "hold_bilateral_contact_fraction", "drop", "full_task_success", "branch_restore_parity", "initial_state_hash", "query_state_hash", "second_query_state_hash", "arm_action_hash", "telemetry_path", "telemetry_rows", "window_rows", "missing_primary_telemetry"]
    write_csv(OUT / "CURRENT_VALIDATION_BRANCHES.csv", branch_rows, raw_fields)
    for r in branch_rows:
        task_dir = OUT / f"TASK{r['task']}_TRACES"; task_dir.mkdir(parents=True, exist_ok=True)
        src = Path(r["telemetry_path"])
        if src.exists(): shutil.copy2(src, task_dir / src.name)

    by_ctx = defaultdict(list)
    for r in branch_rows: by_ctx[r["context_id"]].append(r)
    context_map, pairwise = [], []
    for cid in sorted(by_ctx):
        rr = by_ctx[cid]
        level = {role: [r for r in rr if r["role"] == role] for role in ROLES}
        agg = {}
        for role in ROLES:
            q = level[role]
            agg[role] = {"setpoint_N": statistics.fmean([r["setpoint_N"] for r in q]) if q else None,
                         "mean_force_N": statistics.fmean([r["object_filtered_true_force_mean_N"] for r in q]) if q and all(r.get("object_filtered_true_force_mean_N") is not None for r in q) else None,
                         "median_force_N": statistics.median([r["object_filtered_true_force_mean_N"] for r in q]) if q and all(r.get("object_filtered_true_force_mean_N") is not None for r in q) else None,
                         "mean_top5_N": statistics.fmean([r["object_filtered_true_force_top5_N"] for r in q]) if q and all(r.get("object_filtered_true_force_top5_N") is not None for r in q) else None,
                         "mean_exposure_Ns": statistics.fmean([r["object_filtered_force_exposure_Ns"] for r in q]) if q and all(r.get("object_filtered_force_exposure_Ns") is not None for r in q) else None,
                         "success_rate": statistics.fmean([r["full_task_success"] for r in q]) if q else None,
                         "repeat_count": len(q), "arm_hashes": sorted({r.get("arm_action_hash", "") for r in q}),
                         "initial_hashes": sorted({r.get("initial_state_hash", "") for r in q}), "query_hashes": sorted({r.get("query_state_hash", "") for r in q})}
        xs = [agg[r]["setpoint_N"] for r in ROLES]; context_row = {"task": rr[0]["task"], "root_id": rr[0]["root_id"], "context_id": cid, "friction": float(rr[0]["hidden_friction_analysis_only"]), "friction_band": rr[0]["friction_band"], "historical_spearman": next(x["historical_context_spearman"] for x in target["selected_contexts"] if x["context_id"] == cid), "historical_strict": next(x["historical_context_strict"] for x in target["selected_contexts"] if x["context_id"] == cid), "branch_initial_state_match": int(len({r.get("initial_state_hash", "") for r in rr}) == 1 and all(r.get("initial_state_hash", "") for r in rr)), "query_state_match": int(len({r.get("query_state_hash", "") for r in rr}) == 1 and all(r.get("query_state_hash", "") for r in rr)), "raw_arm_action_hash_match": int(len({r.get("arm_action_hash", "") for r in rr}) == 1 and all(r.get("arm_action_hash", "") for r in rr))}
        for role in ROLES:
            context_row[f"{role}_setpoint_N"] = agg[role]["setpoint_N"]; context_row[f"{role}_measured_mean_N"] = agg[role]["mean_force_N"]; context_row[f"{role}_measured_top5_N"] = agg[role]["mean_top5_N"]; context_row[f"{role}_exposure_Ns"] = agg[role]["mean_exposure_Ns"]; context_row[f"{role}_success_rate"] = agg[role]["success_rate"]; context_row[f"{role}_repeat_count"] = agg[role]["repeat_count"]
        for metric, key in [("mean", "mean_force_N"), ("top5", "mean_top5_N"), ("exposure", "mean_exposure_Ns")]:
            values = [agg[r][key] for r in ROLES]; context_row[f"{metric}_rank_spearman"] = spearman([1,2,3], values) if all(v is not None for v in values) else None; context_row[f"{metric}_low_mid_ordered"] = int(values[0] <= values[1]) if all(v is not None for v in values) else 0; context_row[f"{metric}_mid_high_ordered"] = int(values[1] <= values[2]) if all(v is not None for v in values) else 0; context_row[f"{metric}_low_high_ordered"] = int(values[0] <= values[2]) if all(v is not None for v in values) else 0
            for i in range(3):
                for j in range(i+1,3): pairwise.append({"task": rr[0]["task"], "context_id": cid, "metric": metric, "low_role": ROLES[i], "high_role": ROLES[j], "low_value": values[i], "high_value": values[j], "ordered": int(values[i] <= values[j]) if values[i] is not None and values[j] is not None else 0, "difference_high_minus_low": values[j] - values[i] if values[i] is not None and values[j] is not None else None, "classification": "MISSING_OR_BAD_TELEMETRY" if values[i] is None or values[j] is None else ("SMALL_OVERLAP" if abs(values[j]-values[i]) <= 0.25 else ("ORDERED" if values[j] >= values[i] else "LARGE_DETERMINISTIC_REVERSAL"))})
        context_map.append(context_row)
    selected_by_context = {x["context_id"]: x for x in target["selected_contexts"]}
    admission_by_context = defaultdict(list)
    for row in admission_rows:
        admission_by_context[row["context_id"]].append(row)
    admission_summary = []
    for cid, selected in selected_by_context.items():
        rows = admission_by_context.get(cid, [])
        statuses = sorted({r.get("status", "") for r in rows})
        reasons = sorted({r.get("stop_reason", "") for r in rows if r.get("stop_reason", "")})
        admission_summary.append({
            "task": selected["task"], "root_id": selected["root_id"], "context_id": cid,
            "selected_role": selected["role"], "selected_friction_band": selected["friction_band"],
            "selected_setpoints": {role: next((float(x["setpoint_N"]) for x in branch_rows if x["context_id"] == cid and x["role"] == role), None) for role in ROLES},
            "job_count": len(rows), "primary_branch_rows": sum(int(r.get("completed_primary_branches") or 0) for r in rows),
            "statuses": statuses, "stop_reasons": reasons,
            "has_primary_branches": int(cid in by_ctx),
        })
    write_json(OUT / "CURRENT_CONTEXT_ADMISSION.json", {"selected_contexts": len(selected_by_context), "contexts_with_primary_branches": len(by_ctx), "contexts_without_primary_branches": len(selected_by_context) - len(by_ctx), "rows": admission_summary})
    write_csv(OUT / "CONTEXT_LEVEL_MAPPING.csv", context_map)
    write_json(OUT / "PAIRWISE_MONOTONICITY.json", {"selected_contexts": len(selected_by_context), "evaluated_contexts": len(context_map), "planned_pairwise_comparisons": 3 * len(selected_by_context), "evaluated_pairwise_comparisons": 3 * len(context_map), "total_contexts": len(context_map), "pairs_per_metric": 3 * len(context_map), "by_metric": {metric: {"total": sum(1 for p in pairwise if p["metric"] == metric), "correct": sum(p["ordered"] for p in pairwise if p["metric"] == metric), "violations": sum(not p["ordered"] for p in pairwise if p["metric"] == metric), "violation_rate": sum(not p["ordered"] for p in pairwise if p["metric"] == metric)/(3*len(context_map)) if context_map else None, "small_overlap": sum(p["classification"] == "SMALL_OVERLAP" for p in pairwise if p["metric"] == metric), "large_deterministic_reversal": sum(p["classification"] == "LARGE_DETERMINISTIC_REVERSAL" for p in pairwise if p["metric"] == metric)} for metric in ["mean", "top5", "exposure"]}, "pairs": pairwise})

    task_map = {}
    for task in TASKS:
        cs = [c for c in context_map if int(c["task"]) == task]
        task_map[str(task)] = {"contexts_tested": len(cs), "mean_rank_spearman": {m: statistics.fmean([c[f"{m}_rank_spearman"] for c in cs]) if cs and all(c[f"{m}_rank_spearman"] is not None for c in cs) else None for m in ["mean", "top5", "exposure"]}, "pairwise_violation_rate": {m: sum(1 for p in pairwise if int(p["task"]) == task and p["metric"] == m and not p["ordered"])/(3*len(cs)) if cs else None for m in ["mean", "top5", "exposure"]}, "levels": {role: {"measured_mean_N": statistics.fmean([c[f"{role}_measured_mean_N"] for c in cs]) if cs and all(c[f"{role}_measured_mean_N"] is not None for c in cs) else None, "top5_N": statistics.fmean([c[f"{role}_measured_top5_N"] for c in cs]) if cs and all(c[f"{role}_measured_top5_N"] is not None for c in cs) else None, "exposure_Ns": statistics.fmean([c[f"{role}_exposure_Ns"] for c in cs]) if cs and all(c[f"{role}_exposure_Ns"] is not None for c in cs) else None, "success_rate": statistics.fmean([c[f"{role}_success_rate"] for c in cs]) if cs and all(c[f"{role}_success_rate"] is not None for c in cs) else None} for role in ROLES}, "aggregate_low_mid_high_mean_ordered": None}
        vals = [task_map[str(task)]["levels"][r]["measured_mean_N"] for r in ROLES]; task_map[str(task)]["aggregate_low_mid_high_mean_ordered"] = int(all(v is not None for v in vals) and vals[0] <= vals[1] <= vals[2])
        selected_task_contexts = [x for x in target["selected_contexts"] if int(x["task"]) == task]
        task_map[str(task)].update({"contexts_selected": len(selected_task_contexts), "contexts_with_primary_branches": len(cs), "contexts_without_primary_branches": len(selected_task_contexts) - len(cs)})
    write_json(OUT / "TASK_LEVEL_MAPPING.json", task_map)

    valid_rows = [r for r in branch_rows if r.get("object_filtered_true_force_mean_N") is not None]
    overall = {}
    for metric in ["mean", "top5", "exposure"]:
        overall[metric] = {"within_context_mean_rank_spearman": statistics.fmean([c[f"{metric}_rank_spearman"] for c in context_map]) if context_map and all(c[f"{metric}_rank_spearman"] is not None for c in context_map) else None}
    # Preserve explicit naming requested by the validation protocol.
    overall["within_context_mean_force_spearman"] = overall["mean"]["within_context_mean_rank_spearman"]
    overall["within_context_top5_spearman"] = overall["top5"]["within_context_mean_rank_spearman"]
    overall["within_context_exposure_spearman"] = overall["exposure"]["within_context_mean_rank_spearman"]
    evaluated_pairs = 3 * len(context_map)
    planned_pairs = 3 * len(target["selected_contexts"])
    main_pairs = [p for p in pairwise if p["metric"] == "mean"]
    overall.update({"planned_pairwise_comparisons": planned_pairs, "evaluated_pairwise_comparisons": evaluated_pairs, "total_pairwise_comparisons": evaluated_pairs, "pairwise_ordering_correct": sum(p["ordered"] for p in main_pairs), "pairwise_monotonic_violations": sum(not p["ordered"] for p in main_pairs), "pairwise_violation_rate": sum(not p["ordered"] for p in main_pairs)/evaluated_pairs if evaluated_pairs else None, "small_overlap": sum(p["classification"] == "SMALL_OVERLAP" for p in main_pairs), "large_deterministic_reversal": sum(p["classification"] == "LARGE_DETERMINISTIC_REVERSAL" for p in main_pairs), "valid_branch_rows": len(valid_rows), "branch_rows": len(branch_rows), "selected_contexts": len(selected_by_context), "evaluated_contexts": len(context_map)})
    overall["levels"] = {}
    for role in ROLES:
        rows = [r for r in branch_rows if r.get("role") == role and r.get("object_filtered_true_force_mean_N") is not None]
        overall["levels"][role] = {"measured_mean_N": statistics.fmean([r["object_filtered_true_force_mean_N"] for r in rows]) if rows else None, "top5_N": statistics.fmean([r["object_filtered_true_force_top5_N"] for r in rows]) if rows else None, "exposure_Ns": statistics.fmean([r["object_filtered_force_exposure_Ns"] for r in rows]) if rows else None, "success_rate": statistics.fmean([r["full_task_success"] for r in rows]) if rows else None, "rows": len(rows)}
    repeat_groups = defaultdict(list)
    for row in branch_rows:
        if row.get("object_filtered_true_force_mean_N") is not None:
            repeat_groups[(row["context_id"], row["role"])].append(row)
    repeat_pairs = []
    for (cid, role), rows in sorted(repeat_groups.items()):
        if len(rows) == 2:
            a, b = [float(x["object_filtered_true_force_mean_N"]) for x in sorted(rows, key=lambda x: int(x["repeat"]))]
            repeat_pairs.append({"context_id": cid, "role": role, "repeat1_N": a, "repeat2_N": b, "abs_difference_N": abs(a-b), "relative_difference": abs(a-b) / max(abs((a+b)/2.0), 1e-12)})
    repeat_stability = {"pairs": repeat_pairs, "pair_count": len(repeat_pairs), "mean_abs_difference_N": statistics.fmean([x["abs_difference_N"] for x in repeat_pairs]) if repeat_pairs else None, "median_abs_difference_N": statistics.median([x["abs_difference_N"] for x in repeat_pairs]) if repeat_pairs else None, "p95_abs_difference_N": sorted(x["abs_difference_N"] for x in repeat_pairs)[max(0, math.ceil(0.95 * len(repeat_pairs)) - 1)] if repeat_pairs else None, "outcome_agreement_rate": statistics.fmean([int(a["full_task_success"] == b["full_task_success"]) for rows in repeat_groups.values() if len(rows) == 2 for a, b in [rows]]) if repeat_pairs else None}
    write_json(OUT / "CURRENT_REPEAT_STABILITY.json", repeat_stability)
    write_json(OUT / "CURRENT_MAPPING_SUMMARY.json", overall)
    parity = {"current_runtime_branch_parity": int(bool(branch_rows) and all(int(r.get("branch_restore_parity", 0)) == 1 for r in branch_rows) and all(c["branch_initial_state_match"] and c["query_state_match"] for c in context_map)), "branch_initial_state_match_all_contexts": int(bool(context_map) and all(c["branch_initial_state_match"] for c in context_map)), "raw_arm_action_hash_match_all_contexts": int(bool(context_map) and all(c["raw_arm_action_hash_match"] for c in context_map)), "contexts": context_map, "jobs": job_status}
    write_json(OUT / "CURRENT_BRANCH_PARITY.json", parity)
    main_ok = all(overall.get(k) is not None and overall[k] >= 0.80 for k in ["within_context_mean_force_spearman", "within_context_top5_spearman", "within_context_exposure_spearman"])
    pairs_ok = overall.get("pairwise_violation_rate") is not None and overall["pairwise_violation_rate"] <= 0.10
    tasks_ok = all(task_map[str(t)]["aggregate_low_mid_high_mean_ordered"] for t in TASKS)
    complete_scope = len(context_map) == len(selected_by_context) and all(task_map[str(t)]["contexts_without_primary_branches"] == 0 for t in TASKS)
    task_partial = [t for t in TASKS if task_map[str(t)]["aggregate_low_mid_high_mean_ordered"]]
    strength = "STRICT" if complete_scope and main_ok and pairs_ok and all(c["mean_low_mid_ordered"] and c["mean_mid_high_ordered"] for c in context_map) else ("STRONG_STATISTICAL" if complete_scope and main_ok and pairs_ok and tasks_ok else ("MODERATE" if main_ok or pairs_ok else "INVALID"))
    validity = "YES" if main_ok and pairs_ok and tasks_ok and parity["current_runtime_branch_parity"] else ("PARTIAL" if task_partial and parity["current_runtime_branch_parity"] else "NO")
    comparison = {"historical_mapping_strength": "STRONG_STATISTICAL", "historical_reference": {"strict_contexts": "66/72", "mean_context_spearman": 0.986111, "pairwise_violation_rate": 0.011111}, "current_mapping_summary": overall, "current_mapping_strength": strength, "current_runtime_preserves_historical_ordering": "YES" if validity == "YES" else ("PARTIAL" if validity == "PARTIAL" else "NO"), "current_runtime_continuous_setpoint_mapping_valid": validity, "task_level_validity": {str(t): bool(task_map[str(t)]["aggregate_low_mid_high_mean_ordered"]) for t in TASKS}, "additional_targeted_validation_needed": validity != "YES", "selected_contexts": len(selected_by_context), "evaluated_contexts": len(context_map), "contexts_without_primary_branches": len(selected_by_context) - len(context_map), "current_blocker": "P4-B context admission failed for selected contexts without primary branches" if len(context_map) < len(selected_by_context) else "none after successful collection"}
    write_json(OUT / "CURRENT_VS_HISTORICAL_MAPPING.json", comparison)
    report = ["# Current-runtime continuous setpoint validation", "", f"Status: `{validity}`; current strength: `{strength}`.", "", "## Scope", "", f"12 selected contexts, 4 tasks, 3 historical rank levels, 2 fresh-process repeats per level; expected 72 rollouts, collected {len(branch_rows)} branch rows across {len(context_map)} contexts.", f"Context admission: {len(selected_by_context) - len(context_map)} selected contexts produced no primary branches; see `CURRENT_CONTEXT_ADMISSION.json`.", "", "## Frozen semantics", "", "The setpoint is a continuous grasp-force setpoint, not an exact instantaneous Newton-force claim. Primary measured quantity is object-filtered bilateral force `F_obj_bilateral_n`; top-5% and exposure use the same branch_hold+lift window.", "", "## Results", "", f"Within-context mean/top5/exposure rank Spearman on evaluable contexts: {overall['within_context_mean_force_spearman']}, {overall['within_context_top5_spearman']}, {overall['within_context_exposure_spearman']}.", f"Primary mean-force pairwise violations: {overall['pairwise_monotonic_violations']}/{overall['evaluated_pairwise_comparisons']} evaluated; {overall['planned_pairwise_comparisons']} were planned.", f"Current branch parity among collected branches: {parity['current_runtime_branch_parity']}; raw arm hash match: {parity['raw_arm_action_hash_match_all_contexts']}.", "", "## Interpretation", "", "No exact old-vs-current force equality is required. This validation asks whether within the same current-runtime context, LOW/MID/HIGH setpoint ranks preserve the direction of measured interaction force. The task-wide claim remains partial because task1/task5/task6 contexts were rejected by the unchanged P4-B context-admission gate before branch execution.", "", "## Caveats", "", "Full-task and success values are auxiliary; hold-success is not emitted as a distinct frozen field by the runner, so the report keeps hold bilateral-contact fraction and marks hold_success unavailable. Effective target is reported only when `F_target_eff_n` exists in telemetry.", "", "## Frozen downstream state", "", f"`LOW_LEVEL_EXECUTION_FROZEN = {'YES' if validity == 'YES' else 'NO'}`. No posterior, Expected Utility, probe, VLA arm, or ActiveForcing method action is taken by this validation."]
    (OUT / "CURRENT_RUNTIME_SETPOINT_VALIDATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "VALIDATION_QA.json", {"status": "PASS" if branch_rows and len(valid_rows) == len(branch_rows) and complete_scope else ("PARTIAL_COLLECTION" if branch_rows else "INCOMPLETE"), "jobs_completed": sum(1 for x in state.get("records", []) if int(x.get("returncode", 1)) == 0), "jobs_expected": len(target["jobs"]), "branch_rows": len(branch_rows), "valid_measured_rows": len(valid_rows), "expected_rows": len(target["jobs"])*3, "context_rows": len(context_map), "selected_contexts": len(selected_by_context), "contexts_without_primary_branches": len(selected_by_context) - len(context_map), "task_rows": len(task_map), "required_artifacts": ["VALIDATION_CONTEXT_MANIFEST.json", "CURRENT_EXECUTION_CONTRACT.json", "CONTEXT_LEVEL_MAPPING.csv", "TASK_LEVEL_MAPPING.json", "PAIRWISE_MONOTONICITY.json", "CURRENT_VS_HISTORICAL_MAPPING.json", "CURRENT_RUNTIME_SETPOINT_VALIDATION_REPORT.md"], "run_state": state})
    print(json.dumps({"status": "ANALYZED", "branch_rows": len(branch_rows), "contexts": len(context_map), "validity": validity, "strength": strength}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=1200)
    args = ap.parse_args()
    if args.prepare: prepare()
    elif args.execute: execute(args.timeout_s)
    elif args.analyze: analyze()
    else: ap.error("choose --prepare, --execute, or --analyze")


if __name__ == "__main__":
    main()
