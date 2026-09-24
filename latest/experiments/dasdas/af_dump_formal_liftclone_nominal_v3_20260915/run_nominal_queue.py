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
AF = Path("/media/volume/dasdas/exouser/af_dump_formal_liftclone_v3_20260915")
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
REPO = BASE / "RoboTwin"
PYTHON = BASE / "venv_robotwin/bin/python3"
MODELS = HERE / "models_deploy"
INFER = HERE / "infer_nominal_context.py"
sys.path.insert(0, str(HERE))
from audit_nominal_results import audit_context


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
        "formal_contexts_sha256": sha(plan / "FORMAL_CONTEXTS.json"),
    }
    if checks != lock:
        raise ValueError("Frozen lift-clone Nominal plan changed")
    protocol = read(plan / "PROTOCOL.json")
    for path, digest in protocol["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen source changed: " + path)
    smoke = read(plan / "SMOKE_CONTEXT.json")
    contexts = read(plan / "FORMAL_CONTEXTS.json")
    if len(contexts) != 24 or any(row["methods"] != ["Nominal Frozen VLA"] for row in contexts):
        raise ValueError("Lift-clone Nominal denominator changed")
    return protocol, smoke, contexts


def require_af_complete() -> dict:
    status = read(AF / "STATUS.json")
    if status.get("status") != "complete" or status.get("completed_contexts") != 24:
        raise RuntimeError("AF lift-clone queue is not complete; Nominal must wait")
    if not (AF / "FINAL_RESULTS.json").exists():
        raise RuntimeError("AF lift-clone FINAL_RESULTS.json is missing")
    if (AF / "LIFTCLONE_V3_FORMAL_QUEUE.lock").exists():
        lock_file = (AF / "LIFTCLONE_V3_FORMAL_QUEUE.lock").open("a")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.close()
            raise RuntimeError("AF lift-clone queue still holds its lock") from error
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
    return status


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
                "commanded_force_N": raw.get("commanded_force_N"),
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
    }


def run_one(context: dict, category: str) -> dict:
    raw_root = HERE / f"{category}_raw"
    records = HERE / f"{category}_records"
    kept = HERE / f"{category}_kept"
    case = raw_root / context["id"]
    record_path = records / f"{context['id']}.json"
    if record_path.exists():
        return read(record_path)
    if case.exists():
        raise RuntimeError("Partial retained lift-clone Nominal context requires inspection: " + str(case))
    if shutil.disk_usage(HERE).free < int(1.4 * 1024**3):
        raise RuntimeError("Storage gate: less than 1.4 GiB before lift-clone Nominal context")
    case.mkdir(parents=True)
    training = read(MODELS / "TRAINING_COMPLETE.json")
    spec = {
        "context": context,
        "models": str(MODELS.resolve()),
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
        raise RuntimeError("Lift-clone Nominal context failed/unknown; retained for inspection: " + context["id"])
    audited = audit_context(case)
    write(case / "INDEPENDENT_CONTEXT_AUDIT.json", audited)
    record = compact(audited)
    counterfactual = read(case / "job/FORMAL_CONTEXT_RESULT.json").get("af_counterfactual")
    record["af_counterfactual"] = counterfactual
    write(case / "CASE_COMPACT.json", record)
    write(record_path, record)
    kept.mkdir(parents=True, exist_ok=True)
    shutil.copy2(case / "job/FORMAL_CONTEXT_RESULT.json", kept / f"{context['id']}_FORMAL_CONTEXT_RESULT.json")
    shutil.copy2(case / "job/PREACTION_AF_DECISION.json", kept / f"{context['id']}_PREACTION_AF_DECISION.json")
    shutil.copy2(case / "INDEPENDENT_CONTEXT_AUDIT.json", kept / f"{context['id']}_AUDIT.json")
    return record


def pair_with_af(records: list[dict]) -> dict:
    af_rows = []
    nominal_rows = []
    paired = {"both_success": 0, "AF_only": 0, "Nominal_only": 0, "both_fail": 0}
    first_chunk_matches = 0
    handoff_matches = 0
    for record in records:
        nominal = next(item for item in record["outcomes"] if item["method"] == "Nominal Frozen VLA")
        nominal_rows.append(nominal)
        af_id = record["context"]["matched_af_context_id"]
        af_record = read(AF / "main_records" / f"{af_id}.json")
        af = next(item for item in af_record["outcomes"] if item["method"] == "ActiveForcing")
        af_rows.append(af)
        af_s, nom_s = int(af["success"]), int(nominal["success"])
        key = (
            "both_success"
            if af_s and nom_s
            else "AF_only"
            if af_s
            else "Nominal_only"
            if nom_s
            else "both_fail"
        )
        paired[key] += 1
        af_kept = read(AF / "main_kept" / f"{af_id}_FORMAL_CONTEXT_RESULT.json")
        nom_kept = read(HERE / "main_kept" / f"{record['context']['id']}_FORMAL_CONTEXT_RESULT.json")
        if af_kept.get("first_chunk_sha256") == nom_kept.get("first_chunk_sha256"):
            first_chunk_matches += 1
        if af_kept.get("common_handoff_state_sha256") == nom_kept.get("common_handoff_state_sha256"):
            handoff_matches += 1
    commanded = [row["commanded_force_N"] for row in af_rows]
    summary = [
        {
            "method": "Nominal Frozen VLA (lift-clone prefix)",
            "successes": sum(int(row["success"]) for row in nominal_rows),
            "contexts": len(nominal_rows),
            "success_rate": sum(int(row["success"]) for row in nominal_rows) / len(nominal_rows),
            "mean_commanded_force_N": None,
            "mean_measured_squeeze_N": sum(row["measured_mean_squeeze_N"] for row in nominal_rows) / len(nominal_rows),
            "source": str(HERE),
        },
        {
            "method": "ActiveForcing-liftclone-v3",
            "successes": sum(int(row["success"]) for row in af_rows),
            "contexts": len(af_rows),
            "success_rate": sum(int(row["success"]) for row in af_rows) / len(af_rows),
            "mean_commanded_force_N": sum(commanded) / len(commanded),
            "mean_measured_squeeze_N": sum(row["measured_mean_squeeze_N"] for row in af_rows) / len(af_rows),
            "source": str(AF),
        },
    ]
    return {
        "completed": True,
        "physical_root": 200002,
        "contexts": len(records),
        "nominal_rollouts": len(records),
        "nominal_reused": False,
        "established_grasp_N": 12.0,
        "pi0_remainder_only": True,
        "matched_prefix": "12N grasp then dump shear probe then last-command handoff",
        "first_chunk_sha256_matches": first_chunk_matches,
        "common_handoff_state_sha256_matches": handoff_matches,
        "summary": summary,
        "AF_vs_Nominal": paired,
        "records": [str(HERE / "main_records" / f"{record['context']['id']}.json") for record in records],
        "claim_boundary": (
            "same-root formal seeds; 12N grasp then dump probe then native Nominal remainder "
            "vs AF-liftclone v3 remainder; reused pre-grasp Nominal is not this comparator"
        ),
        "finished_utc": now(),
    }


def main() -> None:
    lock_file = (HERE / "LIFTCLONE_NOMINAL_V3_FORMAL_QUEUE.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A live lift-clone Nominal queue already holds the lock") from error
    require_af_complete()
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
    if nominal["method"] != "Nominal Frozen VLA" or nominal["commanded_force_N"] is not None:
        raise ValueError("Nominal smoke admission failed")
    write(
        HERE / "SMOKE_PASS.json",
        {
            "passed": True,
            "task_success_not_required": True,
            "observed_success": bool(nominal["success"]),
            "native_actions": nominal["native_actions"],
            "measured_squeeze_telemetry_nonempty": nominal["measured_mean_squeeze_N"] is not None,
            "commanded_force_N": None,
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
                "completed_rollouts": index,
                "current_context": context["id"],
                "updated_utc": now(),
            },
        )
        records.append(run_one(context, "main"))
    final = pair_with_af(records)
    write(HERE / "FINAL_RESULTS.json", final)
    write(
        HERE / "STATUS.json",
        {
            "status": "complete",
            "completed_contexts": 24,
            "completed_rollouts": 24,
            "current_context": None,
            "updated_utc": now(),
        },
    )
    print(json.dumps(final["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
