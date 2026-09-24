"""Resume Stage I after the queue received SIGTERM mid-context.

The interrupted raw case is preserved verbatim as an engineering interruption.
It is not entered as an official Stage-I context because the matched 6-method
set is incomplete. The same precommitted context is retried. No model, seed,
method, metric, or outcome-handling rule changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import run_stage_i

HERE = Path(__file__).resolve().parent
CONTEXT_ID = "final_mu0.375_root200002_ps180200002"
FAILED = HERE / "stage_i_raw" / CONTEXT_ID
PRESERVED = HERE / "engineering_failures" / f"{CONTEXT_ID}_attempt_001"
REMOTE_FAILURE = (
    "/anvil/projects/x-cis250966/tabero-transfer/"
    "jetstream-activeforcing-20260912/motion_diversity_20260913/"
    "engineering_failures"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def live_queue() -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "af_motion_diversity_study_20260913/run_stage_i"],
        capture_output=True,
        text=True,
    )
    lines = [
        line
        for line in result.stdout.splitlines()
        if "run_stage_i" in line and str(os.getpid()) not in line
    ]
    return bool(lines)


def classify_interrupted(case: Path) -> dict:
    branches = sorted(path.name for path in (case / "job").glob("branch_*") if path.is_dir())
    completed = []
    interrupted = []
    for name in branches:
        branch = case / "job" / name
        if (branch / "result.json").exists() and not (branch / "error.json").exists():
            completed.append(name)
        else:
            interrupted.append(name)
    if (case / "job/AF_INFERENCE_RESULT.json").exists():
        raise ValueError("Interrupted case already has a scientific context result")
    if len(completed) + len(interrupted) != 6:
        raise ValueError(f"Expected 6 branches, found {branches}")
    if len(completed) == 6:
        raise ValueError("All six branches completed; inspect before retrying")
    return {
        "classified_utc": now(),
        "classification": "QUEUE_SIGTERM_INCOMPLETE_CONTEXT",
        "error": "parent Stage-I shell exit_code=143 SIGTERM",
        "cause_context": (
            "The Stage-I queue process was terminated by SIGTERM while "
            "context final_mu0.375_root200002_ps180200002 was still running. "
            "Remaining force branches then failed with BrokenPipe/EOF. "
            "This is an infrastructure interruption, not a scientific "
            "seed failure."
        ),
        "completed_branches": completed,
        "interrupted_branches": interrupted,
        "query_completed": (case / "job/query/qualification.json").exists(),
        "af_inference_result_present": False,
        "official_context_record_written": False,
        "partial_outcomes_used_for_population": False,
        "seed_replaced": False,
        "same_context_retry_required": True,
        "model_or_protocol_changed": False,
    }


def archive_preserved(case: Path) -> dict:
    remote_final = f"{REMOTE_FAILURE}/{case.name}.tar.gz"
    remote_partial = remote_final + ".partial"
    subprocess.run(
        run_stage_i.ssh_args(f"mkdir -p '{REMOTE_FAILURE}' && rm -f '{remote_partial}'"),
        check=True,
        capture_output=True,
        text=True,
    )
    original_remote = run_stage_i.REMOTE
    try:
        run_stage_i.REMOTE = REMOTE_FAILURE
        receipt = run_stage_i.stream_archive(case)
    finally:
        run_stage_i.REMOTE = original_remote
    if receipt["remote_path"] != remote_final:
        raise ValueError(f"Unexpected remote path {receipt['remote_path']}")
    return receipt


def preserve_interrupted_attempt() -> dict:
    if PRESERVED.exists():
        if FAILED.exists():
            raise RuntimeError("Both interrupted and preserved attempt paths exist")
        return run_stage_i.read(PRESERVED / "ENGINEERING_FAILURE_CLASSIFICATION.json")
    if not FAILED.exists():
        raise FileNotFoundError("Expected interrupted Stage-I raw case")
    classification = classify_interrupted(FAILED)
    write(
        FAILED / "EXIT.json",
        {
            "exit_code": 143,
            "signal": "SIGTERM",
            "disk_guard_stop": False,
            "finished_utc": now(),
            "note": "Written after parent queue termination; worker produced no EXIT.json",
        },
    )
    PRESERVED.parent.mkdir(parents=True, exist_ok=True)
    os.rename(FAILED, PRESERVED)
    write(PRESERVED / "ENGINEERING_FAILURE_CLASSIFICATION.json", classification)
    receipt = archive_preserved(PRESERVED)
    sidecar = HERE / "engineering_failures" / f"{CONTEXT_ID}_attempt_001.archive.json"
    write(sidecar, receipt)
    # Local raw is retained remotely after SHA256/gzip/tar verification.
    shutil.rmtree(PRESERVED)
    write(
        HERE / "engineering_failures" / f"{CONTEXT_ID}_attempt_001.local_removed.json",
        {
            "removed_utc": now(),
            "reason": "local capacity; verified remote engineering-failure archive retained",
            "remote_path": receipt["remote_path"],
            "archive_sha256": receipt["archive_sha256"],
        },
    )
    return classification


def main() -> None:
    if live_queue():
        raise RuntimeError("A live Stage-I runner still exists")
    if shutil.disk_usage(HERE).free < int(1.5 * 1024**3):
        raise RuntimeError("Storage gate: less than 1.5 GiB before resume")
    classification = preserve_interrupted_attempt()
    if shutil.disk_usage(HERE).free < int(1.5 * 1024**3):
        raise RuntimeError("Storage gate: less than 1.5 GiB after preserving interruption")
    extension = {
        "created_utc": now(),
        "extension": "STAGE_I_RUNTIME_RESUME_V3",
        "reason": "queue SIGTERM (exit 143) left an incomplete matched context",
        "original_runner": str(HERE / "run_stage_i.py"),
        "original_runner_sha256": sha(HERE / "run_stage_i.py"),
        "resume_runner": str(Path(__file__).resolve()),
        "resume_runner_sha256": sha(Path(__file__).resolve()),
        "precommit_sha256": sha(HERE / "PRECOMMITTED_MOTION_SEEDS.json"),
        "protocol_sha256": sha(HERE / "STAGE_I_PROTOCOL.json"),
        "interrupted_context": CONTEXT_ID,
        "same_seed_retry": 180200002,
        "partial_outcomes_excluded_from_population": True,
        "classification": classification,
        "changes": [
            "preserve incomplete SIGTERM attempt",
            "retry the same precommitted context from a clean raw directory",
            "detach later resumes from the Cursor tool shell when possible",
        ],
        "unchanged": [
            "precommitted seeds",
            "contexts",
            "methods",
            "models",
            "utility",
            "controller",
            "success criterion",
            "metrics",
        ],
    }
    write(HERE / "STAGE_I_RUNTIME_EXTENSION_02.json", extension)
    run_stage_i.execute()


if __name__ == "__main__":
    main()
