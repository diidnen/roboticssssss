from __future__ import annotations

import fcntl
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

# Concurrent context jobs. MUST stay 1 while a single pi0 server advances a
# global RNG on get_action order; parallel record breaks seed isolation.
PARALLEL_CONTEXTS = max(1, int(os.environ.get("AF_OPENLOOP_PARALLEL", "1")))
if PARALLEL_CONTEXTS != 1:
    raise RuntimeError(
        "AF_OPENLOOP_PARALLEL must be 1: shared pi0 server RNG is not request-isolated"
    )


HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
REPO = BASE / "RoboTwin"
PYTHON = BASE / "venv_robotwin/bin/python3"
INFER = HERE / "infer_openloop_context.py"
VAL_QUEUE_PATTERN = str(HERE.parent / "af_dump_val_policy_sanity_20260914/run_val_queue.py")
sys.path.insert(0, str(HERE))
from audit_openloop import audit_context


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def status(name: str, **fields) -> None:
    write(HERE / "STATUS.json", {"status": name, "updated_utc": now(), **fields})


def verify_plan() -> tuple[dict, dict, list[dict]]:
    plan = HERE / "plan"
    lock = read(plan / "FREEZE_LOCK.json")
    checks = {
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
        "main_contexts_sha256": sha(plan / "MAIN_CONTEXTS.json"),
    }
    if checks != lock:
        raise ValueError("Frozen fixed-force plan changed")
    protocol = read(plan / "PROTOCOL.json")
    for path, digest in protocol["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen source changed: " + path)
    smoke = read(plan / "SMOKE_CONTEXT.json")
    contexts = read(plan / "MAIN_CONTEXTS.json")
    if len(contexts) != 10 or sum(len(row["forces_N"]) for row in contexts) != 60:
        raise ValueError("Main denominator changed")
    if sorted(smoke["forces_N"]) != [12.0, 16.0, 20.0]:
        raise ValueError("Smoke force set changed")
    if protocol.get("version") != "AF_DUMP_OPENLOOP_PRIORSUCC_V2":
        raise ValueError("Unexpected open-loop protocol version")
    return protocol, smoke, contexts


def server_ready() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 6001), timeout=2):
            return True
    except OSError:
        return False


def other_val_queue_live() -> bool:
    result = subprocess.run(["pgrep", "-f", VAL_QUEUE_PATTERN], capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def gpu_has_room() -> tuple[bool, dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    used, total, utilization = [int(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
    receipt = {"memory_used_MiB": used, "memory_total_MiB": total, "utilization_percent": utilization}
    return used <= int(total * 0.88), receipt


def compact(audit: dict) -> dict:
    outcomes = []
    for branch in audit["branches"]:
        raw = branch["result"]
        metrics = branch["trace_metrics"]
        outcomes.append(
            {
                "method": branch["method"],
                "force_N": branch["force_N"],
                "success": branch["success"],
                "native_actions": raw["native_actions"],
                "physics_steps": raw["physics_steps"],
                "measured_allstep_mean_squeeze_N": raw["measured_mean_squeeze_n"],
                "measured_max_squeeze_N": raw["measured_max_squeeze_n"],
                **metrics,
                "trace_sha256": raw["trace_sha256"],
                "raw_audit_passed": branch["raw_audit"]["passed"],
            }
        )
    return {
        "context": audit["context"],
        "completed_utc": now(),
        "strict_matched": True,
        "first_chunk_pairing_exact": audit["first_chunk_pairing_exact"],
        "common_handoff_state_sha256": audit["common_handoff_state_sha256"],
        "context_result_sha256": audit["context_result_sha256"],
        "outcomes": outcomes,
        "all_failures_retained": True,
        "outcome_used_for_queue_control": False,
    }


def run_one(context: dict, category: str) -> dict:
    raw_root = HERE / f"{category}_raw"
    records = HERE / f"{category}_records"
    kept = HERE / f"{category}_kept"
    case = raw_root / context["id"]
    record_path = records / f"{context['id']}.json"
    kept_marker = kept / f"{context['id']}.json"
    if record_path.exists() and kept_marker.exists():
        return read(record_path)
    if case.exists():
        raise RuntimeError("Partial retained context requires inspection: " + str(case))
    if shutil.disk_usage(raw_root.resolve()).free < 20 * 1024**3:
        raise RuntimeError("Storage gate: less than 20 GiB free on dasdas")
    case.mkdir(parents=True)
    spec = {"context": context, "friction": context["friction"]}
    write(case / "SPEC.json", spec)
    command = [
        str(PYTHON),
        str(INFER),
        "--repo",
        str(REPO),
        "--out",
        str(case / "job"),
        "--native-ft",
        "--friction",
        str(context["friction"]),
        "--policy-seed",
        str(context["policy_seed"]),
    ]
    environment = os.environ.copy()
    for key in ["AF_COLLECTION_CONTEXT", "AF_INFERENCE_CONTEXT", "AF_FORMAL_CONTEXT"]:
        environment.pop(key, None)
    environment.update(
        AF_P4_PHYSICAL_SURFACE_CAMERA="1",
        AF_ORIGINAL_SQUEEZE_INNER="1",
        # The reused frozen rim20 initialization contract names its generic
        # evaluation-side context channel AF_INFERENCE_CONTEXT. It reads only
        # spec["context"]; no feasibility/model inference is invoked here.
        AF_INFERENCE_CONTEXT=str(case / "SPEC.json"),
        AF_OPENLOOP_CONTEXT=str(case / "SPEC.json"),
        PYTHONUTF8="1",
        PATH=str(BASE / "runtime_bin") + os.pathsep + environment.get("PATH", "/usr/bin:/bin"),
    )
    write(case / "PROCESS.json", {"command": command, "cwd": str(REPO), "started_utc": now()})
    with (case / "worker.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write(case / "PID.json", {"pid": process.pid})
        disk_stop = False
        while process.poll() is None:
            if shutil.disk_usage(raw_root.resolve()).free < 10 * 1024**3:
                disk_stop = True
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                break
            time.sleep(10)
    write(case / "EXIT.json", {"exit_code": process.returncode, "disk_guard_stop": disk_stop, "finished_utc": now()})
    if process.returncode or not (case / "job/FIXED_SWEEP_CONTEXT_RESULT.json").exists():
        raise RuntimeError("Open-loop context failed/unknown; retained for inspection: " + context["id"])
    audited = audit_context(case)
    write(case / "INDEPENDENT_CONTEXT_AUDIT.json", audited)
    record = compact(audited)
    write(case / "CASE_COMPACT.json", record)
    write(record_path, record)
    write(
        kept_marker,
        {
            "context_id": context["id"],
            "local_raw_path": str(case),
            "record_path": str(record_path),
            "retained_utc": now(),
        },
    )
    return record


def wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - half, center + half]


def mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(left_only, right_only) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def aggregate(records: list[dict]) -> dict:
    forces = [10.0, 12.0, 15.0, 16.0, 18.0, 20.0]
    summary = []
    for force in forces:
        rows = [next(item for item in record["outcomes"] if item["force_N"] == force) for record in records]
        successes = sum(row["success"] for row in rows)
        contact_means = [row["target_contact_mean_squeeze_N"] for row in rows if row["target_contact_mean_squeeze_N"] is not None]
        summary.append(
            {
                "force_N": force,
                "successes": successes,
                "contexts": len(rows),
                "success_rate": successes / len(rows),
                "success_rate_wilson95": wilson(successes, len(rows)),
                "mean_target_contact_fraction": float(np_mean([row["target_contact_fraction"] for row in rows])),
                "mean_target_contact_squeeze_N": float(np_mean(contact_means)) if contact_means else None,
            }
        )
    curves = []
    nondec = 0
    for record in records:
        outcomes = {row["force_N"]: int(row["success"]) for row in record["outcomes"]}
        seq = [outcomes[force] for force in forces]
        ok = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
        nondec += int(ok)
        curves.append({"context_id": record["context"]["id"], "success_by_force_N": {str(f): outcomes[f] for f in forces}, "nondecreasing": ok})
    return {
        "completed": True,
        "physical_root": 200002,
        "contexts": len(records),
        "paired_rollouts": sum(len(record["outcomes"]) for record in records),
        "summary": summary,
        "per_context_curves": curves,
        "full_curve_nondecreasing": {"contexts": nondec, "total": len(records), "fraction": nondec / len(records)},
        "motion_mode": "openloop_replay_fixed_chunks",
        "claim_boundary": "controlled-motion auxiliary labels; identical arm actions across forces",
        "finished_utc": now(),
    }


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def smoke_gate(protocol: dict, record: dict) -> dict:
    rows = {row["force_N"]: row for row in record["outcomes"]}
    gate = protocol["smoke_gate"]
    checks = {
        "all_raw_audits_pass": all(row["raw_audit_passed"] for row in rows.values()),
        "commands_present": sorted(rows) == [12.0, 16.0, 20.0],
        "12N_must_succeed": bool(rows[12.0]["success"]) if gate.get("12N_must_succeed") else True,
        "20N_target_contact_steps_min": rows[20.0]["target_contact_steps"] >= gate["20N_target_contact_steps_min"],
        "20N_max_below_engineering_limit": rows[20.0]["max_squeeze_N"] <= gate["20N_max_squeeze_N"],
        "apertures_within_physical_bounds": all(
            0 <= row["active_min_aperture_m"] <= gate["aperture_max_m"] for row in rows.values()
        ),
        "finite_actions": all(row["native_actions"] > 0 for row in rows.values()),
    }
    return {"passed": all(checks.values()), "checks": checks, "outcomes": record["outcomes"], "finished_utc": now()}


def main() -> None:
    lock_file = (HERE / "OPENLOOP_QUEUE.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A live open-loop queue already holds the lock") from error
    protocol, smoke, contexts = verify_plan()
    while other_val_queue_live():
        status(
            "waiting_for_existing_val_queue",
            completed_contexts=0,
            completed_rollouts=0,
            current_context=None,
            blocker=VAL_QUEUE_PATTERN,
        )
        time.sleep(30)
    while True:
        room, gpu = gpu_has_room()
        if room:
            break
        status("waiting_for_gpu_room", gpu=gpu, completed_contexts=0, completed_rollouts=0, current_context=None)
        time.sleep(30)
    if not server_ready():
        raise RuntimeError("Frozen pi0 policy server is not listening on port 6001")
    prior = protocol.get("prior_engineering_smoke")
    if prior:
        if sha(Path(prior["record_path"])) != prior["record_sha256"] or not all(prior["checks"].values()):
            raise ValueError("Reused V2 engineering admission changed")
        write(HERE / "REUSED_ENGINEERING_ADMISSION.json", prior)
    else:
        # Never reuse superseded pi0-record smoke for prior-success protocol.
        status("smoke_running", completed_contexts=0, completed_rollouts=0, current_context=smoke["id"], gpu=gpu)
        smoke_record = run_one(smoke, "smoke")
        admission = smoke_gate(protocol, smoke_record)
        write(HERE / "SMOKE_ADMISSION.json", admission)
        if not admission["passed"]:
            status("smoke_failed", completed_contexts=0, completed_rollouts=0, current_context=None)
            raise RuntimeError("open-loop smoke failed; main sweep not started")
    records_by_id = {}
    pending = list(contexts)
    done = 0
    status(
        "main_running",
        completed_contexts=0,
        completed_rollouts=0,
        current_context=pending[0]["id"] if pending else None,
        parallel_contexts=PARALLEL_CONTEXTS,
        gpu=gpu_has_room()[1],
    )
    with ThreadPoolExecutor(max_workers=PARALLEL_CONTEXTS) as pool:
        futures = {pool.submit(run_one, context, "main"): context for context in pending}
        for future in as_completed(futures):
            context = futures[future]
            record = future.result()
            records_by_id[context["id"]] = record
            done += 1
            verify_plan()
            status(
                "main_running",
                completed_contexts=done,
                completed_rollouts=done * 6,
                current_context=context["id"],
                parallel_contexts=PARALLEL_CONTEXTS,
                gpu=gpu_has_room()[1],
            )
    records = [records_by_id[context["id"]] for context in contexts]
    final = aggregate(records)
    write(HERE / "FINAL_RESULTS.json", final)
    status("complete", completed_contexts=10, completed_rollouts=60, current_context=None)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
