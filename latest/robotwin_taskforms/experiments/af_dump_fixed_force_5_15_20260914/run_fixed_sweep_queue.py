from __future__ import annotations

import fcntl
import json
import math
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
PYTHON = BASE / "venv_robotwin/bin/python3"
INFER = HERE / "infer_fixed_sweep_context.py"
VAL_QUEUE_PATTERN = str(HERE.parent / "af_dump_val_policy_sanity_20260914/run_val_queue.py")
sys.path.insert(0, str(HERE))
from audit_fixed_sweep import audit_context


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
    if len(contexts) != 24 or sum(len(row["forces_N"]) for row in contexts) != 120:
        raise ValueError("Main denominator changed")
    if sorted(smoke["forces_N"]) != [8.0, 15.0]:
        raise ValueError("Smoke force set changed")
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
        AF_FIXED_SWEEP_CONTEXT=str(case / "SPEC.json"),
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
        raise RuntimeError("Fixed-force context failed/unknown; retained for inspection: " + context["id"])
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
    forces = [5.0, 8.0, 10.0, 12.0, 15.0]
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
                "mean_target_contact_tracking_ratio": float(
                    np_mean([row["target_contact_tracking_ratio"] for row in rows if row["target_contact_tracking_ratio"] is not None])
                ),
                "mean_longest_postcontact_loss_steps": float(np_mean([row["longest_postcontact_loss_steps"] for row in rows])),
            }
        )
    pairwise = {}
    for force in [5.0, 10.0, 12.0, 15.0]:
        counts = {"both_success": 0, "candidate_only": 0, "Fixed8_only": 0, "both_fail": 0}
        for record in records:
            outcomes = {row["force_N"]: row["success"] for row in record["outcomes"]}
            candidate, fixed8 = outcomes[force], outcomes[8.0]
            key = (
                "both_success"
                if candidate and fixed8
                else "candidate_only"
                if candidate
                else "Fixed8_only"
                if fixed8
                else "both_fail"
            )
            counts[key] += 1
        counts["net_paired_wins"] = counts["candidate_only"] - counts["Fixed8_only"]
        counts["exact_mcnemar_two_sided_p"] = mcnemar_exact(counts["candidate_only"], counts["Fixed8_only"])
        pairwise[f"Fixed{force:g}_vs_Fixed8"] = counts
    return {
        "completed": True,
        "physical_root": 200002,
        "contexts": len(records),
        "paired_rollouts": sum(len(record["outcomes"]) for record in records),
        "summary": summary,
        "paired_contrasts": pairwise,
        "records": [str(HERE / "main_records" / f"{record['context']['id']}.json") for record in records],
        "posthoc_range_diagnostic": True,
        "all_failures_retained": True,
        "claim_boundary": "same-root posthoc physical force-range diagnostic; no >8 N feasibility validity claim",
        "finished_utc": now(),
    }


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def smoke_gate(protocol: dict, record: dict) -> dict:
    rows = {row["force_N"]: row for row in record["outcomes"]}
    eight, fifteen = rows[8.0], rows[15.0]
    gate = protocol["smoke_gate"]
    checks = {
        "all_raw_audits_pass": all(row["raw_audit_passed"] for row in rows.values()),
        "15N_target_contact_steps_min": fifteen["target_contact_steps"] >= gate["15N_target_contact_steps_min"],
        "15N_tracking_ratio_in_range": gate["15N_target_contact_tracking_ratio_range"][0]
        <= fifteen["target_contact_tracking_ratio"]
        <= gate["15N_target_contact_tracking_ratio_range"][1],
        "15N_mean_exceeds_8N": fifteen["target_contact_mean_squeeze_N"]
        >= eight["target_contact_mean_squeeze_N"] + gate["15N_target_contact_mean_exceeds_8N_by_N"],
        "15N_max_below_engineering_limit": fifteen["max_squeeze_N"] <= gate["15N_max_squeeze_N"],
    }
    return {"passed": all(checks.values()), "checks": checks, "outcomes": record["outcomes"], "finished_utc": now()}


def main() -> None:
    lock_file = (HERE / "FIXED_SWEEP_QUEUE.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A live fixed-sweep queue already holds the lock") from error
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
    status("smoke_running", completed_contexts=0, completed_rollouts=0, current_context=smoke["id"], gpu=gpu)
    smoke_record = run_one(smoke, "smoke")
    admission = smoke_gate(protocol, smoke_record)
    write(HERE / "SMOKE_ADMISSION.json", admission)
    if not admission["passed"]:
        status("smoke_failed", completed_contexts=0, completed_rollouts=2, current_context=None)
        raise RuntimeError("15 N engineering smoke failed; main sweep not started")
    records = []
    for index, context in enumerate(contexts):
        verify_plan()
        status(
            "main_running",
            completed_contexts=index,
            completed_rollouts=index * 5,
            current_context=context["id"],
        )
        records.append(run_one(context, "main"))
    final = aggregate(records)
    write(HERE / "FINAL_RESULTS.json", final)
    status("complete", completed_contexts=24, completed_rollouts=120, current_context=None)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
