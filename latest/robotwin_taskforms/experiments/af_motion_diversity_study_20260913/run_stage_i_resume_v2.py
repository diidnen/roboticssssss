"""Versioned Stage-I resume after a pre-outcome Vulkan device-loss attempt.

The failed attempt is preserved verbatim. The same precommitted context is
retried only after the pre-existing cross-force GPU job has exited. No model,
seed, method, metric, or outcome-handling rule changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import run_stage_i

HERE = Path(__file__).resolve().parent
FAILED = (
    HERE
    / "stage_i_raw/final_mu0.375_root200002_ps160200002"
)
PRESERVED = (
    HERE
    / "engineering_failures/final_mu0.375_root200002_ps160200002/attempt_001"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def crossforce_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "af_next_stage_diagnosis_20260913/.*crossforce_worker.py"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def preserve_failed_attempt() -> None:
    if PRESERVED.exists():
        if FAILED.exists():
            raise RuntimeError("Both failed and preserved attempt paths exist")
        return
    if not FAILED.exists():
        raise FileNotFoundError("Expected retained device-loss attempt")
    exit_row = run_stage_i.read(FAILED / "EXIT.json")
    qualification = run_stage_i.read(FAILED / "job/query/qualification.json")
    if exit_row["exit_code"] != -6:
        raise ValueError("Unexpected failed-attempt exit code")
    if qualification.get("query_steps_completed") != 0:
        raise ValueError("Failed attempt advanced into the physical query")
    if (FAILED / "job/AF_INFERENCE_RESULT.json").exists():
        raise ValueError("Failed attempt contains a scientific outcome")
    PRESERVED.parent.mkdir(parents=True, exist_ok=True)
    os.rename(FAILED, PRESERVED)
    write(
        PRESERVED / "ENGINEERING_FAILURE_CLASSIFICATION.json",
        {
            "classified_utc": now(),
            "classification": "PRE_OUTCOME_INFRASTRUCTURE_FAILURE",
            "error": "vk::Queue::submit: ErrorDeviceLost",
            "cause_context": (
                "A pre-existing cross-force simulator process was concurrently "
                "using the GPU; Stage I launched before that second case was visible."
            ),
            "query_steps_completed": 0,
            "policy_actions_executed": 0,
            "rollout_outcome": None,
            "seed_replaced": False,
            "same_context_retry_required": True,
            "model_or_protocol_changed": False,
        },
    )


def main() -> None:
    if crossforce_running():
        raise RuntimeError("Wait for the pre-existing cross-force GPU job to exit")
    preserve_failed_attempt()
    extension = {
        "created_utc": now(),
        "extension": "STAGE_I_RUNTIME_RESUME_V2",
        "reason": "pre-outcome Vulkan device loss from concurrent simulator use",
        "original_runner": str(HERE / "run_stage_i.py"),
        "original_runner_sha256": sha(HERE / "run_stage_i.py"),
        "resume_runner": str(Path(__file__).resolve()),
        "resume_runner_sha256": sha(Path(__file__).resolve()),
        "precommit_sha256": sha(HERE / "PRECOMMITTED_MOTION_SEEDS.json"),
        "protocol_sha256": sha(HERE / "STAGE_I_PROTOCOL.json"),
        "failed_attempt_path": str(PRESERVED),
        "failed_attempt_exit_sha256": sha(PRESERVED / "EXIT.json"),
        "failed_attempt_worker_log_sha256": sha(PRESERVED / "worker.log"),
        "same_seed_retry": 160200002,
        "outcome_existed_before_retry": False,
        "changes": [
            "preserve pre-outcome failed attempt",
            "serialize Stage I after prior cross-force GPU job",
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
    write(HERE / "STAGE_I_RUNTIME_EXTENSION_01.json", extension)
    run_stage_i.execute()


if __name__ == "__main__":
    main()
