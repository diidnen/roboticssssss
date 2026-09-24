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
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
REPO = BASE / "RoboTwin"
PYTHON = BASE / "venv_robotwin/bin/python3"
MODELS = HERE / "models_deploy"
INFER = HERE / "infer_af_context.py"
sys.path.insert(0, str(HERE))
from audit_formal_results import audit_context


PREFIX_MARKERS = (
    (
        "Contact-lost probe is not deployment qualified",
        "Original P4 contact-lost; decision prefix not deployment qualified",
    ),
    (
        "Dump shear query lost contact after established grasp",
        "Dump shear query lost contact after established grasp",
    ),
)


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


def load_prefix_failures() -> list[dict]:
    path = HERE / "PREFIX_FAILURES.json"
    return read(path) if path.exists() else []


def prefix_skip_ids() -> set[str]:
    return {row["id"] for row in load_prefix_failures()}


def append_prefix_failure(entry: dict) -> None:
    rows = load_prefix_failures()
    if any(row["id"] == entry["id"] for row in rows):
        return
    rows.append(entry)
    write(HERE / "PREFIX_FAILURES.json", rows)


def classify_prefix_failure(case: Path) -> str | None:
    if (case / "job/FORMAL_CONTEXT_RESULT.json").exists():
        return None
    log_path = case / "worker.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    if "ORIGINAL_ONLINE_CHUNK" in log or "ORIGINAL_ONLINE_BRANCH" in log:
        return None
    for marker, reason in PREFIX_MARKERS:
        if marker in log:
            return reason
    return None


def archive_prefix_failure(case: Path, context: dict, reason: str) -> dict:
    dest_root = HERE / "prefix_failures"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / context["id"]
    if dest.exists():
        dest = dest_root / f"{context['id']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    case.rename(dest)
    entry = {
        "id": context["id"],
        "friction": context["friction"],
        "policy_seed": context["policy_seed"],
        "reason": reason,
        "archived_utc": now(),
        "archive_path": str(dest),
        "worker_log": str(dest / "worker.log"),
        "pi0_remainder_started": False,
    }
    append_prefix_failure(entry)
    return entry


def verify_plan() -> tuple[dict, dict, list[dict]]:
    plan = HERE / "plan"
    lock = read(plan / "FREEZE_LOCK.json")
    checks = {
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
        "formal_contexts_sha256": sha(plan / "FORMAL_CONTEXTS.json"),
    }
    if checks != lock:
        raise ValueError("Frozen lift-clone EU Fmax5 formal plan changed")
    protocol = read(plan / "PROTOCOL.json")
    for path, digest in protocol["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen source changed: " + path)
    if protocol.get("executed_selector") != "expected_utility":
        raise ValueError("Frozen protocol is not executed EU")
    smoke = read(plan / "SMOKE_CONTEXT.json")
    contexts = read(plan / "FORMAL_CONTEXTS.json")
    if len(contexts) != 24 or any(row["methods"] != ["ActiveForcing"] for row in contexts):
        raise ValueError("Lift-clone EU AF denominator changed")
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
        "outcome_used_for_queue_control": False,
        "official_nominal_present": False,
    }


def run_one(context: dict, category: str) -> dict:
    raw_root = HERE / f"{category}_raw"
    records = HERE / f"{category}_records"
    kept = HERE / f"{category}_kept"
    case = raw_root / context["id"]
    record_path = records / f"{context['id']}.json"
    if record_path.exists():
        return read(record_path)
    if context["id"] in prefix_skip_ids():
        return {"prefix_skip": True, "context": context, "already_recorded": True}
    if case.exists():
        reason = classify_prefix_failure(case)
        if reason:
            entry = archive_prefix_failure(case, context, reason)
            return {"prefix_skip": True, "context": context, "entry": entry}
        raise RuntimeError("Partial retained lift-clone EU Fmax5 context requires inspection: " + str(case))
    if shutil.disk_usage(HERE).free < int(1.4 * 1024**3):
        raise RuntimeError("Storage gate: less than 1.4 GiB before lift-clone EU Fmax5 context")
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
        AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP="1",
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
            if shutil.disk_usage(HERE).free < int(0.45 * 1024**3):
                disk_stop = True
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                break
            time.sleep(10)
    write(case / "EXIT.json", {"exit_code": process.returncode, "disk_guard_stop": disk_stop, "finished_utc": now()})
    if process.returncode or not (case / "job/FORMAL_CONTEXT_RESULT.json").exists():
        reason = classify_prefix_failure(case)
        if reason:
            entry = archive_prefix_failure(case, context, reason)
            return {"prefix_skip": True, "context": context, "entry": entry}
        raise RuntimeError("Lift-clone EU Fmax5 context failed/unknown; retained for inspection: " + context["id"])
    audited = audit_context(case)
    write(case / "INDEPENDENT_CONTEXT_AUDIT.json", audited)
    record = compact(audited)
    decision = read(case / "job/FORMAL_CONTEXT_RESULT.json").get("af_decision")
    record["af_decision"] = decision
    write(case / "CASE_COMPACT.json", record)
    write(record_path, record)
    kept.mkdir(parents=True, exist_ok=True)
    shutil.copy2(case / "job/FORMAL_CONTEXT_RESULT.json", kept / f"{context['id']}_FORMAL_CONTEXT_RESULT.json")
    shutil.copy2(case / "job/PREACTION_AF_DECISION.json", kept / f"{context['id']}_PREACTION_AF_DECISION.json")
    shutil.copy2(case / "INDEPENDENT_CONTEXT_AUDIT.json", kept / f"{context['id']}_AUDIT.json")
    return record


def aggregate(records: list[dict], prefix_skips: list[dict]) -> dict:
    af_rows = []
    for record in records:
        af = next(item for item in record["outcomes"] if item["method"] == "ActiveForcing")
        af_rows.append(af)
    commanded = [row["commanded_force_N"] for row in af_rows]
    summary = [
        {
            "method": "ActiveForcing-liftclone-eu-fmax5",
            "successes": sum(int(row["success"]) for row in af_rows),
            "contexts": len(af_rows),
            "planned_contexts": 24,
            "prefix_skips": len(prefix_skips),
            "success_rate": sum(int(row["success"]) for row in af_rows) / len(af_rows) if af_rows else None,
            "mean_commanded_force_N": sum(commanded) / len(commanded) if commanded else None,
            "mean_measured_squeeze_N": (
                sum(row["measured_mean_squeeze_N"] for row in af_rows) / len(af_rows) if af_rows else None
            ),
            "selected_force_counts": {
                str(force): sum(abs(float(row["commanded_force_N"]) - force) < 1e-9 for row in af_rows)
                for force in sorted({round(float(row["commanded_force_N"]), 8) for row in af_rows})
            },
        }
    ]
    return {
        "completed": True,
        "physical_root": 200002,
        "planned_contexts": 24,
        "contexts": len(records),
        "af_rollouts": len(records),
        "prefix_skips": prefix_skips,
        "fixed_strong_skipped": True,
        "official_nominal_present": False,
        "official_comparator": "matched Nominal on this prefix; not yet run",
        "unofficial_reused_pregrasp_nominal_not_used": True,
        "established_grasp_N": 12.0,
        "pi0_remainder_only": True,
        "executed_selector": "expected_utility",
        "utility_normalization_N": 5.0,
        "force_support": [0.25, 5.0],
        "summary": summary,
        "records": [str(HERE / "main_records" / f"{record['context']['id']}.json") for record in records],
        "claim_boundary": (
            "same-root formal seeds; original P4 v4 belief, 12N grasp, dump shear, "
            "pi0 remainder; executed EU maxF=5; prefix skips are not AF outcomes; "
            "official comparator is matched Nominal on this prefix when that queue exists"
        ),
        "finished_utc": now(),
    }


def main() -> None:
    lock_file = (HERE / "LIFTCLONE_EU_FMAX5_FORMAL_QUEUE.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A live lift-clone EU Fmax5 formal queue already holds the lock") from error
    protocol, smoke, contexts = verify_plan()
    if not server_ready():
        raise RuntimeError("Frozen pi0 policy server is not listening on port 6001")
    write(
        HERE / "STATUS.json",
        {
            "status": "smoke_running",
            "completed_contexts": 0,
            "completed_rollouts": 0,
            "skipped_prefix_failures": len(load_prefix_failures()),
            "current_context": smoke["id"],
            "updated_utc": now(),
        },
    )
    try:
        smoke_record = run_one(smoke, "smoke")
        if smoke_record.get("prefix_skip"):
            raise ValueError("Smoke prefix skip is not an AF admission")
        af = smoke_record["outcomes"][0]
        decision = smoke_record.get("af_decision") or {}
        if af["method"] != "ActiveForcing" or af["commanded_force_N"] is None:
            raise ValueError("AF smoke admission failed")
        if decision.get("executed_selector") != "expected_utility":
            raise ValueError("Smoke did not execute expected_utility")
        if not (0.25 - 1e-9 <= float(af["commanded_force_N"]) <= 5.0 + 1e-9):
            raise ValueError("Smoke force outside [0.25, 5]")
        write(
            HERE / "SMOKE_PASS.json",
            {
                "passed": True,
                "task_success_not_required": True,
                "observed_success": bool(af["success"]),
                "native_actions": af["native_actions"],
                "measured_squeeze_telemetry_nonempty": af["measured_mean_squeeze_N"] is not None,
                "commanded_force_N": af["commanded_force_N"],
                "executed_selector": decision.get("executed_selector"),
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
                    "completed_contexts": len(records),
                    "completed_rollouts": len(records),
                    "skipped_prefix_failures": len(load_prefix_failures()),
                    "current_context": context["id"],
                    "updated_utc": now(),
                },
            )
            result = run_one(context, "main")
            if result.get("prefix_skip"):
                continue
            records.append(result)
        prefix_skips = load_prefix_failures()
        final = aggregate(records, prefix_skips)
        write(HERE / "FINAL_RESULTS.json", final)
        write(
            HERE / "STATUS.json",
            {
                "status": "complete",
                "completed_contexts": len(records),
                "completed_rollouts": len(records),
                "skipped_prefix_failures": len(prefix_skips),
                "planned_contexts": 24,
                "current_context": None,
                "updated_utc": now(),
            },
        )
        print(json.dumps(final["summary"], indent=2), flush=True)
    except Exception as error:
        write(
            HERE / "STATUS.json",
            {
                "status": "failed",
                "error": repr(error),
                "updated_utc": now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
