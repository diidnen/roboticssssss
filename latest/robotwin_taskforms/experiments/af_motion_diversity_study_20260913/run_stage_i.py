"""Run all precommitted Stage-I contexts with the frozen AF_original model.

Each context is independently audited, streamed to Anvil, verified by SHA256
and gzip integrity, then replaced locally by a compact immutable record. No
outcome affects queue membership, order, force methods, or seed choice.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
V4 = HERE.parent / "af_dump_maxf8_20260913"
OLD = HERE.parent / "af_dump_original_restore_20260912"
MODELS = V4 / "models_v4"
REPOSITORY = BASE / "RoboTwin"
PYTHON = BASE / "venv_robotwin/bin/python3"
INFER = V4 / "infer_v4.py"
KEY = Path("/home/exouser/.ssh/codex_anvil_migration_20260912")
HOST = "x-csong7@anvil.rcac.purdue.edu"
REMOTE = (
    "/anvil/projects/x-cis250966/tabero-transfer/"
    "jetstream-activeforcing-20260912/motion_diversity_20260913/stage_i"
)

sys.path.insert(0, str(OLD))
from audit_original_collected_group import audit_branch, audit_query  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def ssh_args(command: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(KEY),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=20",
        HOST,
        command,
    ]


def verify_frozen(protocol: dict, lock: dict) -> None:
    if sha(HERE / "PRECOMMITTED_MOTION_SEEDS.json") != lock["precommit_sha256"]:
        raise ValueError("Precommit drift")
    if sha(HERE / "STAGE_I_PROTOCOL.json") != lock["protocol_sha256"]:
        raise ValueError("Stage-I protocol drift")
    if sha(HERE / "CONTEXTS.json") != lock["contexts_sha256"]:
        raise ValueError("Stage-I context schedule drift")
    if sha(HERE / "AF_ORIGINAL_FREEZE_MANIFEST.json") != lock["original_freeze_sha256"]:
        raise ValueError("AF_original freeze drift")
    for path, digest in protocol["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Stage-I source drift: {path}")
    original = read(HERE / "AF_ORIGINAL_FREEZE_MANIFEST.json")
    for path, digest in original["frozen_files"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"AF_original artifact drift: {path}")


def trace_summary(path: Path, sustained_window: int = 50) -> dict:
    run = 0
    first_loss = None
    measured_sum = 0.0
    measured_n = 0
    target_contact_steps = 0
    total = 0
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            total += 1
            step = int(row.get("physics_step", total))
            contact = row.get("contact") or {}
            has_target = False
            for finger in contact.get("fingers") or []:
                for point in finger.get("points") or []:
                    if not point.get("is_target"):
                        continue
                    separation = point.get("separation_m")
                    impulse = point.get("impulse_ns") or [0.0, 0.0, 0.0]
                    if (
                        separation is not None
                        and float(separation) < 0.002
                    ) or any(abs(float(x)) > 1e-8 for x in impulse):
                        has_target = True
                        break
                if has_target:
                    break
            if has_target:
                target_contact_steps += 1
                run = 0
            else:
                run += 1
                if run >= sustained_window and first_loss is None:
                    first_loss = step - sustained_window + 1
            value = contact.get("measured_squeeze_n")
            if value is None:
                value = (row.get("original_squeeze_inner") or {}).get(
                    "measured_filtered_N"
                )
            if value is not None:
                measured_sum += float(value)
                measured_n += 1
    return {
        "logged_steps": total,
        "sustained_contact_loss": first_loss is not None,
        "first_sustained_contact_loss_step": first_loss,
        "target_contact_fraction": (
            target_contact_steps / total if total else None
        ),
        "measured_force_mean_from_trace_N": (
            measured_sum / measured_n if measured_n else None
        ),
    }


def compact_case(case: Path, context: dict) -> dict:
    job = case / "job"
    result = read(job / "AF_INFERENCE_RESULT.json")
    lock = read(job / "PREACTION_SELECTION_LOCK.json")
    decision = read(job / "PREACTION_AF_DECISION.json")
    posterior = read(job / "PREACTION_POSTERIOR.json")
    query_result = audit_query(job / "query")
    branches = []
    for index, (method, force) in enumerate(
        zip(lock["branch_methods"], lock["forces_N"])
    ):
        matches = list(job.glob(f"branch_{index}_*"))
        if len(matches) != 1:
            raise ValueError(f"Expected one branch for index {index}")
        branch = matches[0]
        branch_result = read(branch / "result.json")
        branch_audit = audit_branch(branch)
        if not branch_audit["passed"]:
            raise ValueError(f"Branch audit failed: {branch}")
        outcome = next(row for row in result["outcomes"] if row["method"] == method)
        if (
            float(outcome["force_N"]) != float(force)
            or bool(outcome["success"]) != bool(branch_result["success"])
        ):
            raise ValueError("Outcome/result mismatch")
        branches.append(
            {
                "method": method,
                "force_N": float(force),
                "success": int(outcome["success"]),
                "original_utility": float(outcome["original_utility"]),
                "measured_mean_squeeze_N": float(
                    branch_result["measured_mean_squeeze_n"]
                ),
                "measured_max_squeeze_N": float(
                    branch_result["measured_max_squeeze_n"]
                ),
                "native_actions": int(branch_result["native_actions"]),
                "physics_steps": int(branch_result["physics_steps"]),
                "result_sha256": sha(branch / "result.json"),
                "trace_sha256": branch_result["trace_sha256"],
                "raw_audit_passed": True,
                "trace_summary": trace_summary(branch / "physics_trace.jsonl.gz"),
            }
        )
    if not query_result["passed"]:
        raise ValueError("Query audit failed")
    if lock["branch_methods"] != [
        "ActiveForcing",
        "Fixed-1N",
        "Fixed-3N",
        "Fixed-5N",
        "Fixed-6N",
        "Fixed-8N",
    ]:
        raise ValueError("Method schedule drift")
    return {
        "context": context,
        "completed_utc": now(),
        "model": "AF_original",
        "strict_matched": True,
        "state_sha256": lock["state_sha256"],
        "first_chunk_sha256": lock["first_chunk_sha256"],
        "selection_lock_sha256": sha(job / "PREACTION_SELECTION_LOCK.json"),
        "posterior": {
            "mean": posterior["posterior_moments"]["mean"],
            "std": posterior["posterior_moments"]["std"],
            "member_means": posterior.get("member_means"),
            "posterior_sha256": sha(job / "PREACTION_POSTERIOR.json"),
        },
        "AF_decision": {
            "selected_force_N": decision["selected_force_N"],
            "predicted_success": decision["predicted_success"],
            "utility": decision["utility"],
            "decision_sha256": sha(job / "PREACTION_AF_DECISION.json"),
        },
        "query_audit_passed": True,
        "branches": branches,
        "all_failures_retained": True,
        "outcome_used_for_queue_control": False,
    }


def build_file_manifest(case: Path) -> dict:
    files = {}
    for path in sorted(case.rglob("*")):
        if path.is_file() and path.name != "ARCHIVE_FILE_MANIFEST.json":
            files[str(path.relative_to(case))] = {
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            }
    return {
        "case": case.name,
        "created_utc": now(),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files.values()),
    }


def stream_archive(case: Path) -> dict:
    manifest = build_file_manifest(case)
    write(case / "ARCHIVE_FILE_MANIFEST.json", manifest)
    remote_final = f"{REMOTE}/{case.name}.tar.gz"
    remote_partial = remote_final + ".partial"
    setup = subprocess.run(
        ssh_args(f"mkdir -p '{REMOTE}' && rm -f '{remote_partial}'"),
        check=True,
        capture_output=True,
        text=True,
    )
    del setup
    tar = subprocess.Popen(
        ["tar", "-C", str(case.parent), "-czf", "-", case.name],
        stdout=subprocess.PIPE,
    )
    remote = subprocess.Popen(
        ssh_args(f"umask 002; cat > '{remote_partial}' && mv '{remote_partial}' '{remote_final}'"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert tar.stdout is not None and remote.stdin is not None
    digest = hashlib.sha256()
    total = 0
    while True:
        block = tar.stdout.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
        total += len(block)
        remote.stdin.write(block)
    remote.stdin.close()
    remote.stdin = None
    tar_code = tar.wait()
    remote_out, remote_err = remote.communicate()
    if tar_code or remote.returncode:
        raise RuntimeError(
            f"Archive transfer failed tar={tar_code} ssh={remote.returncode}: "
            f"{remote_err.decode(errors='replace')}"
        )
    local_digest = digest.hexdigest()
    verify = subprocess.run(
        ssh_args(
            f"set -e; gzip -t '{remote_final}'; "
            f"test \"$(sha256sum '{remote_final}' | awk '{{print $1}}')\" = '{local_digest}'; "
            f"tar -tzf '{remote_final}' >/dev/null; "
            f"stat -c '%s' '{remote_final}'"
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "archived_utc": now(),
        "remote_path": remote_final,
        "archive_bytes": total,
        "archive_sha256": local_digest,
        "remote_archive_bytes": int(verify.stdout.strip().splitlines()[-1]),
        "remote_sha256_verified": True,
        "remote_gzip_test_passed": True,
        "remote_tar_listing_passed": True,
        "source_file_manifest_sha256": sha(
            case / "ARCHIVE_FILE_MANIFEST.json"
        ),
        "source_file_count": manifest["file_count"] + 1,
        "source_total_bytes_before_archive_manifest": manifest["total_bytes"],
    }


def run_context(context: dict, protocol: dict) -> dict:
    raw_root = HERE / "stage_i_raw"
    records = HERE / "stage_i_records"
    pointers = HERE / "stage_i_archived"
    case = raw_root / context["id"]
    record_path = records / f"{context['id']}.json"
    receipt_path = pointers / f"{context['id']}.json"
    if record_path.exists() and receipt_path.exists():
        receipt = read(receipt_path)
        if not (
            receipt["remote_sha256_verified"]
            and receipt["remote_gzip_test_passed"]
            and receipt["remote_tar_listing_passed"]
        ):
            raise RuntimeError("Existing archive receipt is not verified")
        return read(record_path)
    if case.exists():
        raise RuntimeError(
            "Retained partial/unarchived Stage-I case requires inspection: "
            + str(case)
        )
    if shutil.disk_usage(HERE).free < int(1.5 * 1024**3):
        raise RuntimeError("Storage gate: less than 1.5 GiB before context")
    case.mkdir(parents=True)
    spec = {
        "context": context,
        "models": str(MODELS),
        "belief_manifest_sha256": read(MODELS / "TRAINING_COMPLETE.json")[
            "belief_manifest_sha256"
        ],
        "feasibility_manifest_sha256": read(MODELS / "TRAINING_COMPLETE.json")[
            "feasibility_manifest_sha256"
        ],
        "stage_i_protocol_sha256": sha(HERE / "STAGE_I_PROTOCOL.json"),
    }
    write(case / "SPEC.json", spec)
    command = [
        str(PYTHON),
        str(INFER),
        "--repo",
        str(REPOSITORY),
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
    environment.update(
        AF_P4_PHYSICAL_SURFACE_CAMERA="1",
        AF_ORIGINAL_SQUEEZE_INNER="1",
        AF_INFERENCE_CONTEXT=str(case / "SPEC.json"),
        PYTHONUTF8="1",
        PATH=str(BASE / "runtime_bin")
        + os.pathsep
        + environment.get("PATH", "/usr/bin:/bin"),
    )
    write(
        case / "PROCESS.json",
        {
            "command": command,
            "cwd": str(REPOSITORY),
            "started_utc": now(),
            "stage_i_protocol_sha256": sha(HERE / "STAGE_I_PROTOCOL.json"),
        },
    )
    with (case / "worker.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write(case / "PID.json", {"pid": process.pid})
        disk_stop = False
        while process.poll() is None:
            if shutil.disk_usage(HERE).free < int(0.5 * 1024**3):
                disk_stop = True
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                break
            time.sleep(10)
    write(
        case / "EXIT.json",
        {
            "exit_code": process.returncode,
            "disk_guard_stop": disk_stop,
            "finished_utc": now(),
        },
    )
    if process.returncode or not (case / "job/AF_INFERENCE_RESULT.json").exists():
        raise RuntimeError(
            "Retained Stage-I worker failure; seed is not replaced: " + context["id"]
        )
    compact = compact_case(case, context)
    write(case / "CASE_COMPACT.json", compact)
    write(record_path, compact)
    receipt = stream_archive(case)
    write(receipt_path, receipt)
    # The raw rollout remains preserved in the verified remote archive. The
    # capacity-limited local working copy is removed only after all checks pass.
    shutil.rmtree(case)
    return compact


def execute() -> None:
    queue_lock = (HERE / "STAGE_I_QUEUE.lock").open("a")
    try:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A live Stage-I queue holds the lock") from error
    protocol = read(HERE / "STAGE_I_PROTOCOL.json")
    lock = read(HERE / "STAGE_I_FREEZE_LOCK.json")
    verify_frozen(protocol, lock)
    contexts = read(HERE / "CONTEXTS.json")
    if len(contexts) != 32:
        raise ValueError("Expected all 32 precommitted contexts")
    subprocess.run(
        ssh_args(f"mkdir -p '{REMOTE}'"),
        check=True,
        capture_output=True,
        text=True,
    )
    results = []
    for index, context in enumerate(contexts):
        verify_frozen(protocol, lock)
        write(
            HERE / "STAGE_I_STATUS.json",
            {
                "status": "running",
                "completed_contexts": index,
                "completed_rollouts": index * 6,
                "current_context": context["id"],
                "updated_utc": now(),
            },
        )
        results.append(run_context(context, protocol))
    complete = {
        "completed": True,
        "model": "AF_original",
        "contexts": 32,
        "robot_rollouts": 192,
        "records": [str(HERE / "stage_i_records" / f"{c['id']}.json") for c in contexts],
        "archive_receipts": [
            str(HERE / "stage_i_archived" / f"{c['id']}.json") for c in contexts
        ],
        "all_precommitted_contexts_retained": True,
        "all_remote_archives_verified": True,
        "finished_utc": now(),
    }
    write(HERE / "STAGE_I_COMPLETE.json", complete)
    write(
        HERE / "STAGE_I_STATUS.json",
        {
            "status": "complete",
            "completed_contexts": 32,
            "completed_rollouts": 192,
            "current_context": None,
            "updated_utc": now(),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run"])
    parser.parse_args()
    execute()
