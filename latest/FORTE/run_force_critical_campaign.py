#!/usr/bin/env python3
"""Persistent, model-blind supervisor for the frozen challenge sweeps."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "activeforcing_final_experiment_20260901_045000"
STATUS = OUT / "FORCE_CRITICAL_COLLECTION_SUPERVISOR.json"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(state: str, **extra) -> None:
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "state": state, "updated_utc": utc(), "supervisor_pid": os.getpid(),
        "model_queries": 0, "untouched_TEST_read": False, **extra,
    }, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATUS)


def process_rows() -> list[dict]:
    ans = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except Exception:
            continue
        if cmd:
            ans.append({"pid": int(p.name), "cmdline": cmd})
    return ans


def current_challenge_collectors() -> list[dict]:
    return [r for r in process_rows() if (
        "force_critical_collect.py --task" in r["cmdline"]
        or "prospective_visual_context_collect.py --worker" in r["cmdline"]
    )]


def root_scaling_collectors() -> list[dict]:
    return [r for r in process_rows() if (
        "root_scaling_collect.py" in r["cmdline"]
        or ("prospective_visual_context_collect.py --worker" in r["cmdline"] and "force_critical" not in r["cmdline"])
    )]


def wait_for_existing() -> None:
    while True:
        rows = current_challenge_collectors()
        # The worker command does not contain the output path, but its parent
        # is the known challenge launcher. Do not infer completion until both
        # launcher and worker are gone.
        if not rows:
            return
        write_status("WAITING_FOR_EXISTING_CHALLENGE_WORKER", processes=rows)
        time.sleep(30)


def run_task(task: int) -> None:
    rivals = root_scaling_collectors()
    if rivals:
        write_status("BLOCKED_BY_ROOT_SCALING_COLLECTOR", task=task, processes=rivals)
        raise SystemExit(75)
    log = OUT / "collection_challenge" / "logs" / f"supervisor_task{task}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    write_status("RUNNING_TASK", task=task, log=str(log))
    with log.open("a") as fh:
        rc = subprocess.call(
            [sys.executable, str(ROOT / "force_critical_collect.py"), "--task", str(task), "--timeout-s", "43200"],
            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
        )
    if rc:
        write_status("TASK_FAILED_RESUMABLE", task=task, returncode=rc, log=str(log))
        raise SystemExit(rc)


def main() -> None:
    write_status("STARTED")
    wait_for_existing()
    # task0 may already be complete; the collector is resume-safe and will
    # return ALREADY_COMPLETE after checking all committed contexts.
    run_task(0)
    run_task(5)
    write_status("RUNNING_MODEL_BLIND_MEMBERSHIP_QA")
    log = OUT / "collection_challenge" / "logs" / "membership_qa.log"
    with log.open("a") as fh:
        rc = subprocess.call([sys.executable, str(ROOT / "finalize_force_critical_membership.py")], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if rc:
        write_status("MEMBERSHIP_QA_FAILED", returncode=rc, log=str(log))
        raise SystemExit(rc)
    write_status("MODEL_BLIND_COLLECTION_AND_MEMBERSHIP_COMPLETE", qa=str(OUT / "FORCE_CRITICAL_CHALLENGE_QA.json"))


if __name__ == "__main__":
    main()
