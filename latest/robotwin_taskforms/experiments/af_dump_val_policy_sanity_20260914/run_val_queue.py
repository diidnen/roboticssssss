from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
REPO = BASE / "RoboTwin"
OLD = HERE.parent / "af_dump_original_restore_20260912"
V4 = HERE.parent / "af_dump_maxf8_20260913"
PYTHON = BASE / "venv_robotwin/bin/python3"
MODELS = V4 / "models_v4"
INFER = HERE / "infer_val_context.py"
sys.path[:0] = [str(HERE)]
from audit_formal_results import audit_context


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_plan() -> tuple[dict, dict, list[dict]]:
    plan = HERE / "plan"
    lock = read(plan / "FREEZE_LOCK.json")
    checks = {
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
        "val_contexts_sha256": sha(plan / "VAL_CONTEXTS.json"),
    }
    if checks != lock:
        raise ValueError("Frozen VAL plan changed")
    protocol = read(plan / "PROTOCOL.json")
    for path, digest in protocol["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen source changed: " + path)
    smoke = read(plan / "SMOKE_CONTEXT.json")
    contexts = read(plan / "VAL_CONTEXTS.json")
    if len(contexts) != 6 or sum(len(row["methods"]) for row in contexts) != 18:
        raise ValueError("VAL denominator changed")
    return protocol, smoke, contexts


def server_ready() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 6001), timeout=2):
            return True
    except OSError:
        return False


def compact(audit: dict) -> dict:
    rows = []
    for branch in audit["branches"]:
        raw = branch["result"]
        rows.append(
            {
                "method": branch["method"],
                "success": int(raw["success"]),
                "commanded_force_N": raw.get("commanded_force_N", raw.get("force_setpoint_bilateral_n")),
                "measured_mean_squeeze_N": raw["measured_mean_squeeze_n"],
                "measured_max_squeeze_N": raw["measured_max_squeeze_n"],
                "native_actions": raw["native_actions"],
                "physics_steps": raw["physics_steps"],
                "trace_sha256": raw["trace_sha256"],
                "raw_audit_passed": branch["audit"]["passed"],
            }
        )
    return {
        "context": audit["context"],
        "completed_utc": now(),
        "strict_matched": True,
        "first_chunk_pairing_exact": audit["first_chunk_pairing_exact"],
        "common_handoff_state_sha256": audit["common_handoff_state_sha256"],
        "formal_context_result_sha256": audit["formal_context_result_sha256"],
        "outcomes": rows,
        "all_failures_retained": True,
        "local_raw_retained": True,
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
        raise RuntimeError("Partial retained VAL context requires inspection: " + str(case))
    # Storage gate against dasdas root (symlinked raw lives there).
    if shutil.disk_usage(raw_root.resolve()).free < int(1.4 * 1024**3):
        raise RuntimeError("Storage gate: less than 1.4 GiB before VAL context")
    case.mkdir(parents=True)
    training = read(MODELS / "TRAINING_COMPLETE.json")
    spec = {
        "context": context,
        "models": str(MODELS),
        "belief_manifest_sha256": training["belief_manifest_sha256"],
        "feasibility_manifest_sha256": training["feasibility_manifest_sha256"],
    }
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
    environment.pop("AF_COLLECTION_CONTEXT", None)
    environment.pop("AF_INFERENCE_CONTEXT", None)
    environment.update(
        AF_P4_PHYSICAL_SURFACE_CAMERA="1",
        AF_ORIGINAL_SQUEEZE_INNER="1",
        AF_INFERENCE_CONTEXT=str(case / "SPEC.json"),
        AF_FORMAL_CONTEXT=str(case / "SPEC.json"),
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
            if shutil.disk_usage(raw_root.resolve()).free < int(0.45 * 1024**3):
                disk_stop = True
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                break
            time.sleep(10)
    write(case / "EXIT.json", {"exit_code": process.returncode, "disk_guard_stop": disk_stop, "finished_utc": now()})
    if process.returncode or not (case / "job/FORMAL_CONTEXT_RESULT.json").exists():
        raise RuntimeError("VAL context failed/unknown; retained for inspection: " + context["id"])
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
            "remote_archive_required": False,
        },
    )
    return record


def aggregate(records: list[dict]) -> dict:
    methods = ["Nominal Frozen VLA", "Fixed-Strong 8N", "ActiveForcing"]
    summary = []
    for method in methods:
        rows = [next(item for item in record["outcomes"] if item["method"] == method) for record in records]
        commanded = [row["commanded_force_N"] for row in rows if row["commanded_force_N"] is not None]
        summary.append(
            {
                "method": method,
                "successes": sum(row["success"] for row in rows),
                "contexts": len(rows),
                "success_rate": sum(row["success"] for row in rows) / len(rows),
                "mean_commanded_force_N": sum(commanded) / len(commanded) if commanded else None,
                "mean_measured_squeeze_N": sum(row["measured_mean_squeeze_N"] for row in rows) / len(rows),
            }
        )
    paired = {"both_success": 0, "AF_only": 0, "Fixed_only": 0, "both_fail": 0}
    for record in records:
        outcomes = {row["method"]: row["success"] for row in record["outcomes"]}
        af, fixed = outcomes["ActiveForcing"], outcomes["Fixed-Strong 8N"]
        key = "both_success" if af and fixed else "AF_only" if af else "Fixed_only" if fixed else "both_fail"
        paired[key] += 1
    return {
        "completed": True,
        "physical_root": 200002,
        "contexts": len(records),
        "paired_rollouts": sum(len(record["outcomes"]) for record in records),
        "summary": summary,
        "AF_vs_Fixed8": paired,
        "records": [str(HERE / "main_records" / f"{record['context']['id']}.json") for record in records],
        "local_raw_retained": True,
        "claim_boundary": "feasibility-VAL motion/friction policy sanity only",
        "finished_utc": now(),
    }


def main() -> None:
    lock_file = (HERE / "VAL_QUEUE.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A live VAL queue already holds the lock") from error
    protocol, smoke, contexts = verify_plan()
    if not server_ready():
        raise RuntimeError("Frozen pi0 policy server is not listening on port 6001")
    write(
        HERE / "STATUS.json",
        {
            "status": "smoke_running",
            "completed_contexts": 0,
            "completed_rollouts": 0,
            "current_context": smoke["id"],
            "updated_utc": now(),
        },
    )
    smoke_record = run_one(smoke, "smoke")
    nominal = smoke_record["outcomes"][0]
    if nominal["method"] != "Nominal Frozen VLA" or not smoke_record["first_chunk_pairing_exact"]:
        raise ValueError("Nominal smoke admission failed")
    write(
        HERE / "SMOKE_PASS.json",
        {
            "passed": True,
            "task_success_not_required": True,
            "observed_success": bool(nominal["success"]),
            "native_actions": nominal["native_actions"],
            "measured_squeeze_telemetry_nonempty": nominal["measured_mean_squeeze_N"] is not None,
            "commanded_force_N": nominal["commanded_force_N"],
            "finished_utc": now(),
        },
    )
    records = []
    for index, context in enumerate(contexts):
        verify_plan()
        write(
            HERE / "STATUS.json",
            {
                "status": "main_running",
                "completed_contexts": index,
                "completed_rollouts": index * 3,
                "current_context": context["id"],
                "updated_utc": now(),
            },
        )
        records.append(run_one(context, "main"))
    final = aggregate(records)
    write(HERE / "FINAL_RESULTS.json", final)
    write(
        HERE / "STATUS.json",
        {
            "status": "complete",
            "completed_contexts": 6,
            "completed_rollouts": 18,
            "current_context": None,
            "updated_utc": now(),
        },
    )
    print(json.dumps(final["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
