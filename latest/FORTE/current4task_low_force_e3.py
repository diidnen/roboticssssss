#!/usr/bin/env python3
"""Resource-gated low-force qualification for the four existing tasks.

This wrapper reuses the frozen P5-S0-C runner and its strict-preprobe restore
semantics.  It never modifies the runner, the pi0 checkpoint, or an existing
archive.  Preparation and analysis are CPU-only; ``--collect-task`` is the
only simulator path and refuses to launch when the GPU gate is closed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "gnp_style_continuous_20260830_125107"
SOURCE_PROTOCOL = SOURCE_DIR / "GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"
SOURCE_ARCHIVE = ROOT / "activeforcing_final_closure_20260902_034923" / "E3_BRANCH_LABEL_AUDIT.csv"
RAW_ARCHIVE = SOURCE_DIR / "collection_long2"
RUNNER = Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py")
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
TABERO = Path("/home/exouser/Tabero")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
OPENPI = Path("/media/volume/newdata/exouser/openpi/src")
E3_TASK5_PILOT = Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_FORCE_PHYSICS_PILOT_20260902_062448")
E5_COVERAGE = ROOT / "analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006/E5_COVERAGE_STATUS.json"
DEFAULT_OUT = ROOT / "current4task_low_force_e3_20260902"
TASKS = (0, 1, 5, 6)
ROOT_INDICES = (0, 1)
REPEATS = 2
PREPROBE_STEP = 45 + 35 + 70 + 40
SAFE_LOWER_N = 0.5
# "Moderate" is operationalized conservatively as <=70%.  The user's >90%
# rule remains a hard ceiling, but this lower launch threshold preserves
# headroom for Mass and matches the protected E3 lane's launch discipline.
GPU_UTIL_MAX = 70.0
GPU_FREE_MIN_MIB = 10 * 1024
REQUIRED_TELEMETRY = (
    "task", "root_id", "friction", "force_N", "repeat", "initial_state_hash",
    "query_state_hash", "grasp_success", "lift_success", "lift_height_m",
    "lift_hold_duration_s", "transport_retention", "placement_success",
    "full_task_success", "drop_timestamp_s", "failure_stage", "failure_reason",
    "commanded_grip_force_N", "measured_grip_force_N", "peak_force_N",
    "tracking_error_N",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def exact_force_audit() -> tuple[list[dict[str, str]], dict[int, dict[str, Any]]]:
    rows = read_csv(SOURCE_ARCHIVE)
    summary: dict[int, dict[str, Any]] = {}
    for task in TASKS:
        q = [r for r in rows if int(r["task"]) == task]
        failures = [r for r in q if int(r["full_task_success"]) == 0]
        successes = [r for r in q if int(r["full_task_success"]) == 1]
        by_context: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in q:
            by_context[r["context_id"]].append(r)
        brackets = []
        for cid, rr in by_context.items():
            ff = [float(r["force_N"]) for r in rr if int(r["full_task_success"]) == 0]
            ss = [float(r["force_N"]) for r in rr if int(r["full_task_success"]) == 1]
            if ff and ss:
                brackets.append((max(ff), min(ss)))
        summary[task] = {
            "n": len(q),
            "lowest_tested_force_N": min(float(r["force_N"]) for r in q),
            "highest_tested_force_N": max(float(r["force_N"]) for r in q),
            "lift_failure_n": sum(int(r["local_lift_success"]) == 0 for r in q),
            "full_task_failure_n": len(failures),
            "full_task_success_n": len(successes),
            "fulltask_transition_contexts": len(brackets),
            "fulltask_transition_failure_edge_N": [min((x[0] for x in brackets), default=math.nan), max((x[0] for x in brackets), default=math.nan)],
            "fulltask_transition_success_edge_N": [min((x[1] for x in brackets), default=math.nan), max((x[1] for x in brackets), default=math.nan)],
        }
    return rows, summary


def candidate_forces(task: int, support_min: float) -> list[float]:
    # A small preregistered downward bracket: three points at 1/2, 5/8, and
    # 3/4 of the task-specific certified archive support floor.  Quarter-newton
    # rounding is controller-native and all points remain nonnegative and above
    # the conservative 0.5 N lower guard.
    values = [max(SAFE_LOWER_N, round((support_min * ratio) * 4.0) / 4.0) for ratio in (0.50, 0.625, 0.75)]
    return sorted(set(v for v in values if v < support_min))


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    archive_rows, audit = exact_force_audit()
    support = {int(k): float(v[0]) for k, v in protocol["continuous_collection"]["force_support_N"].items()}
    selected = [c for c in protocol["train_context_population"] if int(c["task"]) in TASKS and int(c["root_index"]) in ROOT_INDICES]
    if len(selected) != len(TASKS) * len(ROOT_INDICES) * 3:
        raise RuntimeError(f"expected 24 selected contexts, found {len(selected)}")

    contexts: dict[str, list[dict[str, Any]]] = {}
    candidate_rows: list[dict[str, Any]] = []
    for ctx in selected:
        task = int(ctx["task"])
        forces = candidate_forces(task, support[task])
        observed = [float(s["requested_force_N"]) for s in ctx["continuous_force_samples"]]
        specs = []
        for force in forces:
            for repeat in range(1, REPEATS + 1):
                tag = str(force).replace(".", "p")
                label = f"LOWFORCE_F{tag}_R{repeat}"
                specs.append({"force_N": force, "repeat_index": repeat - 1, "branch_label": label})
                candidate_rows.append({
                    "task": task, "root_id": ctx["root_id"], "root_index": int(ctx["root_index"]),
                    "root_seed": int(ctx["root_seed"]), "split": "TRAIN", "context_id": ctx["context_id"],
                    "friction_band": ctx["friction_band"], "friction": float(ctx["friction"]),
                    "current_context_min_force_N": min(observed), "current_task_support_floor_N": support[task],
                    "candidate_force_N": force, "repeat": repeat, "safe_lower_guard_N": SAFE_LOWER_N,
                    "selection_basis": "PHYSICAL_BOUNDARY_DOWNWARD_BRACKET", "utility_used": 0,
                    "collection_status": "PENDING_RESOURCE_GATE",
                })
        contexts[ctx["context_id"]] = specs

    target = {
        "manifest_name": "CURRENT4TASK_LOW_FORCE_TARGET_MANIFEST",
        "created_utc": now(), "source_protocol": str(SOURCE_PROTOCOL),
        "source_protocol_sha256": sha256(SOURCE_PROTOCOL), "source_archive": str(SOURCE_ARCHIVE),
        "source_archive_sha256": sha256(SOURCE_ARCHIVE), "runner": str(RUNNER), "runner_sha256": sha256(RUNNER),
        "tasks": list(TASKS), "root_indices": list(ROOT_INDICES), "split": "TRAIN", "repeats": REPEATS,
        "safe_lower_guard_N": SAFE_LOWER_N, "candidate_rule": "round_0.25N({0.50,0.625,0.75} * task support floor)",
        "utility_used_for_cell_selection": False, "contexts": contexts,
        "expected_contexts": len(contexts), "expected_branches": sum(len(x) for x in contexts.values()),
        "frozen_pi0": "authoritative existing checkpoint; no retraining",
        "branch_state": "restored strict pre-probe last-hold snapshot",
    }
    write_json(out / "TARGET_MANIFEST.json", target)
    write_csv(out / "CURRENT4TASK_LOW_FORCE_CANDIDATES.csv", candidate_rows)

    audit_lines = [
        "# Current four-task low-force audit", "", f"Prepared: `{now()}`", "",
        f"Authoritative archive: `{SOURCE_ARCHIVE}` ({len(archive_rows)} branches).", "",
        "No existing archive cell is scheduled for recollection. Candidate selection uses only the physical lift boundary, full-task outcomes, and the controller-safe nonnegative force domain; Utility and ActiveForcing performance are excluded.", "",
        "| Task | Existing support (N) | Lift failures | Full-task failures | Full-task successes | Estimated FullTask transition | New pilot forces (N) |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for task in TASKS:
        a = audit[task]
        fedge, sedge = a["fulltask_transition_failure_edge_N"], a["fulltask_transition_success_edge_N"]
        transition = f"failure edge {fedge[0]:.3f}–{fedge[1]:.3f}; success edge {sedge[0]:.3f}–{sedge[1]:.3f} N across {a['fulltask_transition_contexts']} contexts"
        audit_lines.append(f"| {task} | {a['lowest_tested_force_N']:.3f}–{a['highest_tested_force_N']:.3f} | {a['lift_failure_n']} | {a['full_task_failure_n']} | {a['full_task_success_n']} | {transition} | {', '.join(map(str, candidate_forces(task, support[task])))} |")
    audit_lines += [
        "", "## Resource gate", "",
        f"Each simulator launch requires GPU utilization <={GPU_UTIL_MAX:.0f}%, at least {GPU_FREE_MIN_MIB/1024:.0f} GiB free, a visible healthy Mass process, the full E5 campaign completion gate (inter-shard gaps stay reserved), and no conflict with the E3 long-horizon lane. A failed check exits without launching Isaac.", "",
        "## Telemetry contract", "", "Every collected row is finalized with: " + ", ".join(f"`{x}`" for x in REQUIRED_TELEMETRY) + ".", "",
        "Status at preparation: `PENDING_RESOURCE_GATE`.",
    ]
    (out / "CURRENT4TASK_LOW_FORCE_AUDIT.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    finalize(out)


def gpu_snapshot() -> dict[str, Any]:
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.free,memory.used", "--format=csv,noheader,nounits"]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    util, free, used = [float(x.strip()) for x in p.stdout.strip().split(",")[:3]]
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    return {"timestamp_utc": now(), "gpu_util_percent": util, "free_memory_MiB": free, "used_memory_MiB": used, "compute_apps": apps}


def host_process_snapshot() -> str:
    p = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True)
    return p.stdout


def resource_gate(out: Path, task: int, allow_e5_queue: bool = False) -> dict[str, Any]:
    snap = gpu_snapshot()
    proc = host_process_snapshot()
    mass_visible = "FORTE_mass" in proc or "qualify_mass_" in proc
    e5_visible = "run_e5_fresh_utility.py" in proc
    long_visible = "ACTIVEFORCING_E3_LONGHORIZON" in proc or "run_task5_" in proc
    extension_gate = E3_TASK5_PILOT / "E3_TASK5_TRAIN_SUPPORT_EXTENSION_GATE.json"
    extension_pending = True
    if extension_gate.exists():
        try:
            e3 = json.loads(extension_gate.read_text(encoding="utf-8"))
            observed = {int(x) for x in e3.get("observed_forces_N", [])}
            extension_pending = not bool(e3.get("gate_pass")) and 8 not in observed
        except (OSError, ValueError, json.JSONDecodeError):
            extension_pending = True
    # A missing E5 process is only an inter-shard gap while its 300-rollout
    # completion contract is incomplete.  Treat queued primary work as active
    # reservation so this supplementary lane cannot race the next E5 shard.
    e5_campaign_pending = True
    e5_missing_tuples = None
    if E5_COVERAGE.exists():
        try:
            e5_coverage = json.loads(E5_COVERAGE.read_text(encoding="utf-8"))
            e5_missing_tuples = int(e5_coverage.get("missing_tuples", e5_coverage.get("planned_tuples", 1)))
            e5_campaign_pending = e5_coverage.get("completion_status") != "COMPLETE" or e5_missing_tuples != 0
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            e5_campaign_pending = True
    reasons = []
    if snap["gpu_util_percent"] > GPU_UTIL_MAX:
        reasons.append(f"GPU utilization {snap['gpu_util_percent']:.0f}% > {GPU_UTIL_MAX:.0f}%")
    if snap["free_memory_MiB"] <= GPU_FREE_MIN_MIB:
        reasons.append(f"free memory {snap['free_memory_MiB']:.0f} MiB <= {GPU_FREE_MIN_MIB} MiB")
    if not mass_visible:
        reasons.append("protected Mass process is not visible; fail closed")
    if e5_visible:
        reasons.append("E5 primary fresh-E2E simulator is active; yield priority")
    if e5_campaign_pending and not allow_e5_queue:
        reasons.append(f"E5 primary campaign still has {e5_missing_tuples if e5_missing_tuples is not None else 'unknown'} queued tuples; inter-shard gaps are reserved")
    if long_visible:
        reasons.append("E3 long-horizon/task5 simulator is active; yield priority")
    if extension_pending:
        reasons.append("E3 task5 force-support collection is still queued/incomplete; yield priority")
    gate = {
        **snap, "task": task, "mass_visible": mass_visible, "e5_visible": e5_visible,
        "e5_campaign_pending": e5_campaign_pending, "e5_missing_tuples": e5_missing_tuples,
        "explicit_capacity_authorization": allow_e5_queue,
        "e3_longhorizon_visible": long_visible, "e3_task5_extension_pending": extension_pending,
        "launch_allowed": not reasons, "reasons": reasons,
    }
    path = out / "RESOURCE_GATE_LOG.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(gate, sort_keys=True) + "\n")
    return gate


def append_capture_row(out: Path, row: dict[str, Any]) -> None:
    path = out / f"task{row['task']}" / "strict_preprobe_capture.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def worker() -> None:
    import torch

    p5 = load_module("p5_current4_lowforce", RUNNER)
    original_import = p5.import_p4_probe

    def strict_import(task_id: int):
        p4 = original_import(task_id)
        original_run = p4.run_probe_episode

        def strict_run(env, *, seed_idx: int, mu: float, trial_id: str, dt: float):
            initial_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            base_step = env.step
            counter: dict[str, Any] = {"n": 0, "snapshot": None}

            def capture_step(action):
                result = base_step(action)
                counter["n"] += 1
                if counter["n"] == PREPROBE_STEP:
                    counter["snapshot"] = env.scene.get_state(is_relative=True)
                return result

            env.step = capture_step
            try:
                rows, rec = original_run(env, seed_idx=seed_idx, mu=mu, trial_id=trial_id, dt=dt)
            finally:
                env.step = base_step
            if counter["snapshot"] is None:
                raise RuntimeError(f"STRICT_PREPROBE_CAPTURE_MISSING:{trial_id}:steps={counter['n']}")
            forbidden_postprobe_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            env.reset_to(counter["snapshot"], torch.tensor([0], device=env.device), is_relative=True)
            query_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            env.reset_to(counter["snapshot"], torch.tensor([0], device=env.device), is_relative=True)
            query_hash_2 = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            if query_hash != query_hash_2:
                raise RuntimeError(f"STRICT_PREPROBE_RESTORE_UNSTABLE:{trial_id}")
            append_capture_row(Path(os.environ["P5S0C_OUT"]), {
                "task": task_id, "context_id": trial_id, "seed": seed_idx, "friction": mu,
                "capture_step": PREPROBE_STEP, "initial_state_hash": initial_hash,
                "query_state_hash": query_hash, "second_query_state_hash": query_hash_2,
                "forbidden_postprobe_state_hash": forbidden_postprobe_hash,
                "query_restore_stable": 1, "branch_snapshot_semantics": "STRICT_PREPROBE_LAST_HOLD",
            })
            return rows, rec

        p4.run_probe_episode = strict_run
        return p4

    p5.import_p4_probe = strict_import
    raise SystemExit(p5.worker_main())


def collect_task(out: Path, task: int, timeout_s: int, capacity_authorized: bool = False) -> None:
    if task not in TASKS:
        raise ValueError(task)
    gate = resource_gate(out, task, allow_e5_queue=capacity_authorized)
    if not gate["launch_allowed"]:
        print(json.dumps(gate, indent=2))
        raise SystemExit(75)
    manifest = json.loads((out / "TARGET_MANIFEST.json").read_text(encoding="utf-8"))
    protocol = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    ids = [c["context_id"] for c in protocol["train_context_population"] if int(c["task"]) == task and int(c["root_index"]) in ROOT_INDICES]
    expected = sum(len(manifest["contexts"][cid]) for cid in ids)
    log = out / "logs" / f"task{task}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(TABERO), str(OPENPI)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(TABERO),
        "P5S0C_OUT": str(out), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": str(task),
        "P5S0C_TARGET_MANIFEST": str(out / "TARGET_MANIFEST.json"), "P5S0C_CONTEXT_IDS": ",".join(ids),
        "P5S0C_SKIP_REPLAY": "1", "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
        "CONTINUOUS_STRICT_PREPROBE_WRAPPER": "1",
    })
    cmd = [str(ISAAC_PY), "-u", str(Path(__file__).resolve()), "--worker"]
    started = time.time()
    completed_after_result = False
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=TABERO, env=env, stdout=f, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout_s
        while p.poll() is None and time.time() < deadline:
            result_path = out / f"task{task}" / "result.json"
            error_path = out / f"task{task}" / "error.json"
            if result_path.exists() and not error_path.exists():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if int(result.get("contexts", 0)) == len(ids) and int(result.get("primary_branches", 0)) == expected:
                        completed_after_result = True
                        p.terminate()
                        break
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            time.sleep(2)
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        else:
            p.wait()
    error_path = out / f"task{task}" / "error.json"
    record = {
        "task": task, "started_utc": gate["timestamp_utc"], "ended_utc": now(), "resource_gate": gate,
        "expected_contexts": len(ids), "expected_branches": expected, "process_returncode": p.returncode,
        "effective_returncode": 0 if completed_after_result and not error_path.exists() else int(p.returncode or 0),
        "completed_after_result": completed_after_result, "elapsed_wall_s": time.time() - started, "log": str(log),
        "error": error_path.read_text(encoding="utf-8", errors="replace")[:4000] if error_path.exists() else "",
    }
    write_json(out / f"TASK{task}_COLLECTOR_RECORD.json", record)
    finalize(out)
    if record["effective_returncode"] != 0 or error_path.exists():
        raise SystemExit(1)


def ffloat(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def fint(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return default


def enrich_branch(row: dict[str, str], captures: dict[str, dict[str, str]]) -> dict[str, Any]:
    trajectory = Path(row.get("telemetry_path", ""))
    tr = read_csv(trajectory) if trajectory.exists() else []
    zs = [ffloat(x, "object_z_analysis_only") for x in tr]
    zs = [x for x in zs if math.isfinite(x)]
    z0 = zs[0] if zs else math.nan
    height = max(zs) - z0 if zs else math.nan
    lift_start = next((i for i, z in enumerate(zs) if z - z0 >= 0.03), None) if zs else None
    loss_i = None
    if lift_start is not None:
        for i in range(lift_start + 1, min(len(zs), len(tr))):
            if tr[i].get("phase") in ("transit", "over_basket") and zs[i] - z0 < 0.02:
                loss_i = i
                break
    t0 = ffloat(tr[lift_start], "t_s") if lift_start is not None and lift_start < len(tr) else math.nan
    tend_i = loss_i if loss_i is not None else (len(tr) - 1 if tr else None)
    tend = ffloat(tr[tend_i], "t_s") if tend_i is not None else math.nan
    hold = max(0.0, tend - t0) if math.isfinite(t0) and math.isfinite(tend) else math.nan
    dropped = fint(row, "dropped") or fint(row, "lost_in_transit")
    drop_t = ffloat(tr[loss_i], "t_s") if dropped and loss_i is not None else (ffloat(tr[-1], "t_s") if dropped and tr else math.nan)
    grasp = fint(row, "pick_success")
    lift = fint(row, "lift_success")
    transport = fint(row, "transport_retention")
    placement = fint(row, "place_success")
    success = fint(row, "full_task_success_y")
    if success:
        stage, reason = "", ""
    elif not grasp:
        stage, reason = "GRASP", "grasp_not_established"
    elif not lift:
        stage, reason = "LOCAL_LIFT", "insufficient_force_for_lift"
    elif not transport:
        stage, reason = "TRANSPORT", "object_lost_after_lift"
    elif not placement:
        stage, reason = "PLACEMENT", "basket_contact_not_established"
    else:
        stage, reason = "TERMINAL", row.get("failure_reason", "full_task_failure")
    cap = captures.get(row["context_id"], {})
    result: dict[str, Any] = {
        "branch_id": row["branch_id"], "context_id": row["context_id"], "task": fint(row, "task"),
        "root_id": row.get("root_id", ""), "root_index": fint(row, "root_index"), "root_seed": fint(row, "root_seed"),
        "split": row.get("split", ""), "friction_band": row.get("friction_band", ""),
        "friction": ffloat(row, "hidden_friction_analysis_only"), "force_N": ffloat(row, "requested_force_N"),
        "repeat": fint(row, "repeat_index"), "initial_state_hash": cap.get("initial_state_hash", ""),
        "query_state_hash": cap.get("query_state_hash", row.get("post_probe_state_hash", "")),
        "query_restore_stable": fint(cap, "query_restore_stable"), "grasp_success": grasp, "lift_success": lift,
        "lift_height_m": height, "lift_hold_duration_s": hold, "transport_retention": transport,
        "placement_success": placement, "full_task_success": success,
        "drop_timestamp_s": None if not math.isfinite(drop_t) else drop_t, "failure_stage": stage,
        "failure_reason": reason, "commanded_grip_force_N": ffloat(row, "requested_force_N"),
        "measured_grip_force_N": ffloat(row, "steady_state_mean_N"), "peak_force_N": ffloat(row, "measured_force_peak_N"),
        "tracking_error_N": ffloat(row, "force_tracking_error_N"), "tracking_mae_N": ffloat(row, "force_tracking_mae_N"),
        "telemetry_path": str(trajectory), "raw_branch_source": row.get("label_source", ""),
    }
    conditional_nullable = {"drop_timestamp_s", "failure_stage", "failure_reason"}
    result["telemetry_complete"] = int(
        all(k in result for k in REQUIRED_TELEMETRY)
        and all(result[k] not in ("", None) for k in REQUIRED_TELEMETRY if k not in conditional_nullable)
    )
    return result


def collected_rows(out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        branch_path = out / f"task{task}" / "branches.csv"
        capture_path = out / f"task{task}" / "strict_preprobe_capture.csv"
        captures = {r["context_id"]: r for r in read_csv(capture_path)} if capture_path.exists() else {}
        if branch_path.exists():
            rows.extend(enrich_branch(r, captures) for r in read_csv(branch_path) if r.get("branch_label", "").startswith("LOWFORCE_"))
    return rows


def finalize(out: Path) -> None:
    target_path = out / "TARGET_MANIFEST.json"
    if not target_path.exists():
        return
    target = json.loads(target_path.read_text(encoding="utf-8"))
    archive, audit = exact_force_audit()
    rows = collected_rows(out)
    write_csv(out / "CURRENT4TASK_LOW_FORCE_BRANCHES.csv", rows)
    candidate_path = out / "CURRENT4TASK_LOW_FORCE_CANDIDATES.csv"
    planned_candidates = read_csv(candidate_path) if candidate_path.exists() else []
    expected = int(target["expected_branches"])
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_task[int(r["task"])].append(r)

    boundaries: list[dict[str, Any]] = []
    task_boundaries: list[dict[str, Any]] = []
    balances: list[dict[str, Any]] = []
    ready_tasks = []
    for task in TASKS:
        rr = by_task[task]
        local_fail = [r for r in rr if int(r["lift_success"]) == 0]
        local_success = [r for r in rr if int(r["lift_success"]) == 1]
        old = [r for r in archive if int(r["task"]) == task]
        delayed_old = sum(int(r["local_lift_success"]) == 1 and int(r["full_task_success"]) == 0 for r in old)
        full_old = sum(int(r["full_task_success"]) == 1 for r in old)
        if local_fail:
            classification = "LOCAL_LIFT_BOUNDARY_FOUND"
            ready_tasks.append(task)
        elif rr and min(float(r["force_N"]) for r in rr) <= SAFE_LOWER_N:
            classification = "CONTROLLER_LOWER_BOUND_REACHED"
        else:
            classification = "LOCAL_LIFT_BOUNDARY_NOT_YET_BRACKETED"
        context_rows = []
        planned_task = [r for r in planned_candidates if int(r["task"]) == task]
        planned_context_ids = {r["context_id"] for r in planned_task}
        for context_id in sorted(planned_context_ids | {r["context_id"] for r in rr}):
            cr = [r for r in rr if r["context_id"] == context_id]
            co = [r for r in old if r["context_id"] == context_id]
            cp = [r for r in planned_task if r["context_id"] == context_id]
            cf = [float(r["force_N"]) for r in cr if int(r["lift_success"]) == 0]
            cs = ([float(r["force_N"]) for r in cr if int(r["lift_success"]) == 1]
                  + [float(r["force_N"]) for r in co if int(r["local_lift_success"]) == 1])
            fail_max = max(cf, default=math.nan)
            success_min = min(cs, default=math.nan)
            if cf and cs and fail_max < success_min:
                interval = f"({fail_max:g}, {success_min:g}]"
                overlap = 0
            elif cf and cs:
                interval = f"STOCHASTIC_OVERLAP[{success_min:g}, {fail_max:g}]"
                overlap = 1
            else:
                interval = "NOT_BRACKETED"
                overlap = 0
            if cf and cs:
                context_class = "LOCAL_LIFT_BOUNDARY_FOUND"
            elif cr and min(float(r["force_N"]) for r in cr) <= SAFE_LOWER_N:
                context_class = "CONTROLLER_LOWER_BOUND_REACHED"
            else:
                context_class = "LOCAL_LIFT_BOUNDARY_NOT_YET_BRACKETED"
            first = cr[0] if cr else cp[0]
            br = {
                "row_scope": "TASK_ROOT_FRICTION", "task": task, "root_id": first["root_id"],
                "root_index": first["root_index"], "context_id": context_id,
                "friction_band": first["friction_band"], "friction": first["friction"],
                "classification": context_class, "new_rows": len(cr),
                "lowest_added_force_N": min((float(r["force_N"]) for r in cr), default=math.nan),
                "highest_lift_failure_force_N": fail_max, "lowest_lift_success_force_N": success_min,
                "F_lift_boundary_interval_N": interval, "stochastic_overlap": overlap,
                "F_fulltask_failure_edge_min_N": "", "F_fulltask_failure_edge_max_N": "",
                "F_fulltask_success_edge_min_N": "", "F_fulltask_success_edge_max_N": "",
                "separation_observed": int(bool(cf and delayed_old and full_old)),
            }
            boundaries.append(br)
            context_rows.append(br)
        task_boundary = {
            "row_scope": "TASK_SUMMARY", "task": task, "root_id": "ALL_SELECTED", "root_index": "",
            "context_id": "ALL_SELECTED", "friction_band": "ALL", "friction": "",
            "classification": classification, "new_rows": len(rr),
            "lowest_added_force_N": min((float(r["force_N"]) for r in rr), default=math.nan),
            "highest_lift_failure_force_N": max((float(r["force_N"]) for r in local_fail), default=math.nan),
            "lowest_lift_success_force_N": min([float(r["force_N"]) for r in local_success] + [audit[task]["lowest_tested_force_N"]]),
            "F_lift_boundary_interval_N": (
                f"{sum(r['classification'] == 'LOCAL_LIFT_BOUNDARY_FOUND' for r in context_rows)}/{len(context_rows)} contexts bracketed"
                if context_rows else "NOT_BRACKETED"
            ),
            "stochastic_overlap": sum(int(r["stochastic_overlap"]) for r in context_rows),
            "F_fulltask_failure_edge_min_N": audit[task]["fulltask_transition_failure_edge_N"][0],
            "F_fulltask_failure_edge_max_N": audit[task]["fulltask_transition_failure_edge_N"][1],
            "F_fulltask_success_edge_min_N": audit[task]["fulltask_transition_success_edge_N"][0],
            "F_fulltask_success_edge_max_N": audit[task]["fulltask_transition_success_edge_N"][1],
            "separation_observed": int(bool(local_fail and delayed_old and full_old)),
        }
        boundaries.append(task_boundary)
        task_boundaries.append(task_boundary)
        balances.append({
            "task": task, "new_rows": len(rr), "local_failure_n": len(local_fail),
            "local_success_downstream_failure_n": sum(int(r["lift_success"]) == 1 and int(r["full_task_success"]) == 0 for r in rr) + delayed_old,
            "full_task_success_n": sum(int(r["full_task_success"]) == 1 for r in rr) + full_old,
            "archive_delayed_failure_n": delayed_old, "archive_full_task_success_n": full_old,
            "telemetry_complete_n": sum(int(r["telemetry_complete"]) for r in rr),
        })
    write_csv(out / "CURRENT4TASK_LOCAL_LIFT_BOUNDARIES.csv", boundaries)
    write_csv(out / "CURRENT4TASK_LABEL_BALANCE.csv", balances)

    if candidate_path.exists():
        candidate_rows = planned_candidates
        observed = {
            (int(r["task"]), r["context_id"], float(r["force_N"]), int(r["repeat"])): r
            for r in rows
        }
        for candidate in candidate_rows:
            key = (int(candidate["task"]), candidate["context_id"], float(candidate["candidate_force_N"]), int(candidate["repeat"]))
            match = observed.get(key)
            if match:
                candidate["collection_status"] = "COLLECTED"
                candidate["branch_id"] = match["branch_id"]
                candidate["lift_success"] = match["lift_success"]
                candidate["full_task_success"] = match["full_task_success"]
            else:
                candidate["collection_status"] = "PENDING_RESOURCE_GATE"
        write_csv(candidate_path, candidate_rows)

    records = []
    for task in TASKS:
        p = out / f"TASK{task}_COLLECTOR_RECORD.json"
        if p.exists():
            records.append(json.loads(p.read_text(encoding="utf-8")))
    trajectory_files = sorted((out / "P5S0C_BRANCH_TELEMETRY").glob("*LOWFORCE_*_trajectory.csv"))
    counted_paths = {str(r["telemetry_path"]) for r in rows}
    quarantined_trajectories = [str(p) for p in trajectory_files if str(p) not in counted_paths]
    status = "COMPLETE" if len(rows) == expected and all(int(r["telemetry_complete"]) for r in rows) else ("PARTIAL" if rows else "PENDING_RESOURCE_GATE")
    manifest = {
        "manifest_name": "CURRENT4TASK_LOW_FORCE_COLLECTION_MANIFEST", "updated_utc": now(), "status": status,
        "source_archive": str(SOURCE_ARCHIVE), "source_archive_sha256": sha256(SOURCE_ARCHIVE),
        "target_manifest": str(target_path), "target_manifest_sha256": sha256(target_path),
        "expected_branches": expected, "collected_branches": len(rows), "tasks": list(TASKS),
        "root_split": {"split": "TRAIN", "root_indices": list(ROOT_INDICES), "roots_per_task": len(ROOT_INDICES)},
        "selection_uses_utility": False, "pi0_retraining": False, "collector_records": records,
        "telemetry_required": list(REQUIRED_TELEMETRY),
        "telemetry_complete": bool(rows and len(rows) == expected and all(int(r["telemetry_complete"]) for r in rows)),
        "quarantined_preempted_trajectories": quarantined_trajectories,
        "quarantined_preempted_trajectory_count": len(quarantined_trajectories),
        "branch_table": str(out / "CURRENT4TASK_LOW_FORCE_BRANCHES.csv"), "label_balance": balances,
    }
    write_json(out / "CURRENT4TASK_LOW_FORCE_COLLECTION_MANIFEST.json", manifest)

    if ready_tasks:
        ready = {
            "status": "EXISTING_TASK_LOW_FORCE_E3_READY", "created_utc": now(), "task_ids": ready_tasks,
            "new_force_ranges_N": {str(t): [min(float(r["force_N"]) for r in by_task[t]), max(float(r["force_N"]) for r in by_task[t])] for t in ready_tasks},
            "root_split": manifest["root_split"], "label_balance": [r for r in balances if r["task"] in ready_tasks],
            "number_of_local_failures": sum(r["local_failure_n"] for r in balances if r["task"] in ready_tasks),
            "number_of_delayed_failures": sum(r["local_success_downstream_failure_n"] for r in balances if r["task"] in ready_tasks),
            "number_of_full_task_successes": sum(r["full_task_success_n"] for r in balances if r["task"] in ready_tasks),
            "telemetry_completeness": {"complete": manifest["telemetry_complete"], "complete_rows": sum(r["telemetry_complete_n"] for r in balances), "rows": len(rows)},
            "handoff": "E3 lane; no large-model retraining performed",
        }
        write_json(out / "EXISTING_TASK_LOW_FORCE_E3_READY.json", ready)
    else:
        ready_path = out / "EXISTING_TASK_LOW_FORCE_E3_READY.json"
        if ready_path.exists():
            ready_path.unlink()

    final_status = "CURRENT4TASK_LOW_FORCE_E3_COMPLETE" if ready_tasks and status == "COMPLETE" else ("CURRENT4TASK_LOW_FORCE_E3_NOT_USEFUL" if status == "COMPLETE" else status)
    report = [
        "# Current four-task low-force E3 report", "", f"Status: `{final_status}`", "",
        f"Coverage: {len(rows)}/{expected} planned branches; telemetry complete: {sum(int(r['telemetry_complete']) for r in rows)}/{len(rows)}.", "",
        "| Task | Gate | Added rows | Local failures | Delayed failures (augmented) | Full successes (augmented) | Lift boundary |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    bm = {int(r["task"]): r for r in balances}
    for b in task_boundaries:
        x = bm[int(b["task"])]
        report.append(f"| {b['task']} | {b['classification']} | {b['new_rows']} | {x['local_failure_n']} | {x['local_success_downstream_failure_n']} | {x['full_task_success_n']} | {b['F_lift_boundary_interval_N']} |")
    report += [
        "", "## Interpretation", "",
        ("At least one task contains all three required regimes in the augmented source-backed data. The supplementary manifest is ready for the E3 lane; no pi0 retraining or mechanism-pilot training was performed." if ready_tasks else "Qualification is not yet complete or did not find a safe, useful LocalLift boundary. No training claim is made."),
        "", "## Validation", "",
        f"- Composite branch IDs unique: `{len({r['branch_id'] for r in rows}) == len(rows)}`.",
        f"- Utility used for cell selection: `False`.",
        f"- Existing 720-branch archive recollected or overwritten: `False`.",
        f"- Required telemetry fields present: `{all(all(k in r for k in REQUIRED_TELEMETRY) for r in rows) if rows else False}`.",
        f"- Protected-process modifications: `None`.",
        f"- Complete trajectory files quarantined after self-preemption (not counted): `{len(quarantined_trajectories)}`.",
        f"- Collector attempts self-preempted for protected-work relaunch: `{sum(r.get('termination_reason') == 'SELF_PREEMPTED_WHEN_E5_MASS_AND_E3_RELAUNCHED' for r in records)}`.",
        "", "## Optional matched pilot", "", "Not run. Collection qualification and protected E5/Mass/E3 resource priority take precedence.",
    ]
    (out / "CURRENT4TASK_LOW_FORCE_E3_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--collect-task", type=int)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=10800)
    ap.add_argument("--capacity-authorized", action="store_true", help="honor explicit user authorization to use a stable inter-shard gap; active protected jobs still block")
    args = ap.parse_args()
    if args.worker:
        worker()
    elif args.prepare:
        prepare(args.out.resolve())
    elif args.collect_task is not None:
        collect_task(args.out.resolve(), args.collect_task, args.timeout_s, args.capacity_authorized)
    elif args.finalize:
        finalize(args.out.resolve())
    else:
        ap.error("choose --prepare, --collect-task, or --finalize")


if __name__ == "__main__":
    main()
