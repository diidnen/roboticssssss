"""Stop the live Stage-I queue after the first 16 precommitted contexts.

Does not edit frozen Stage-I sources. The cutoff is the first four policy
seeds in CONTEXTS.json order. Later seeds are left unused, not replaced.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISION = json.loads((HERE / "STAGE_I_BUDGET_REVISION_01.json").read_text())
KEEP = list(REVISION["retained_context_ids"])
RECORDS = HERE / "stage_i_records"
POINTERS = HERE / "stage_i_archived"
RAW = HERE / "stage_i_raw"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def kept_complete() -> bool:
    return all(
        (RECORDS / f"{context_id}.json").exists()
        and (POINTERS / f"{context_id}.json").exists()
        for context_id in KEEP
    )


def runner_pids() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", "af_motion_diversity_study_20260913/run_stage_i"],
        capture_output=True,
        text=True,
    )
    pids = []
    for line in result.stdout.split():
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def infer_pids() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", "af_motion_diversity_study_20260913/stage_i_raw/.*/job"],
        capture_output=True,
        text=True,
    )
    pids = []
    for line in result.stdout.split():
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def preserve_overrun() -> list[str]:
    preserved = []
    for case in sorted(RAW.glob("final_*")):
        if case.name in KEEP:
            continue
        dest_dir = HERE / "engineering_failures"
        dest = dest_dir / f"{case.name}_budget_stop"
        dest_dir.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(case), str(dest))
        write(
            dest / "BUDGET_STOP_CLASSIFICATION.json",
            {
                "classified_utc": now(),
                "classification": "BUDGET_CAP_OVERSHOOT_NOT_POPULATION",
                "context_id": case.name,
                "used_in_stage_i_population": False,
                "seed_replaced": False,
            },
        )
        preserved.append(case.name)
    return preserved


def main() -> None:
    write(
        HERE / "STAGE_I_RUNTIME_EXTENSION_03.json",
        {
            "created_utc": now(),
            "extension": "STAGE_I_BUDGET_CAP_WATCHDOG",
            "revision_path": str(HERE / "STAGE_I_BUDGET_REVISION_01.json"),
            "stop_after_contexts": 16,
            "stop_rule": "first four frozen policy seeds, all four frictions",
            "model_or_protocol_source_changed": False,
        },
    )
    while not kept_complete():
        time.sleep(10)
    for pid in runner_pids() + infer_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(5)
    for pid in runner_pids() + infer_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    overrun = preserve_overrun()
    write(
        HERE / "STAGE_I_STATUS.json",
        {
            "status": "budget_stopped_after_16",
            "completed_contexts": 16,
            "completed_rollouts": 96,
            "current_context": None,
            "updated_utc": now(),
            "revision": "STAGE_I_BUDGET_CAP_FIRST_16_CONTEXTS",
            "overrun_raw_preserved": overrun,
        },
    )
    write(
        HERE / "STAGE_I_BUDGET_STOP_RECEIPT.json",
        {
            "stopped_utc": now(),
            "retained_records": [
                context_id
                for context_id in KEEP
                if (RECORDS / f"{context_id}.json").exists()
            ],
            "overrun_raw_preserved": overrun,
        },
    )


if __name__ == "__main__":
    main()
