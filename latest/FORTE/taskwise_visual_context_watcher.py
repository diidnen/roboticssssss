#!/usr/bin/env python3
"""Read-only collection watcher plus task-specific CPU analysis orchestration.

This process never starts Isaac, the visual server, or the pooled pipeline.  It
waits for authoritative artifacts, freezes task6 immediately after TRAIN is
complete, and evaluates each already-frozen task only after its exact DEV
population passes the preregistered readiness audit.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import per_task_visual_generalization as validation


HERE = Path(__file__).resolve().parent
ROOT = HERE / "taskwise_visual_context_20260831_074000"
STATUS = ROOT / "TASKWISE_WATCH_STATUS.json"
SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")


def write_status(stage: str, **extra) -> None:
    value = {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    STATUS.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, sort_keys=True), flush=True)


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def task6_train_ready() -> tuple[bool, dict]:
    branch_dir = SOURCE / "collection_train/task6/P5S0C_BRANCH_TELEMETRY"
    probe_dir = SOURCE / "collection_train/task6/P5S0C_PROBE_TELEMETRY"
    visual_path = SOURCE / "collection_train/visual_alignment_worker.csv"
    branches = len(list(branch_dir.glob("*.csv"))) if branch_dir.exists() else 0
    probes = len(list(probe_dir.glob("*.csv"))) if probe_dir.exists() else 0
    visual = 0
    if visual_path.exists():
        import pandas as pd
        try:
            d = pd.read_csv(visual_path)
            visual = int((d.task.astype(int) == 6).sum())
        except Exception:
            pass
    return branches >= 180 and probes >= 18 and visual >= 18, {"task6_branch_telemetry": branches, "task6_probe_telemetry": probes, "task6_visual_contexts": visual}


def ensure_task6_frozen() -> None:
    frozen = ROOT / "task6_train_frozen"
    marker = frozen / "ALL_PROSPECTIVE_VISUAL_MODELS_FROZEN_TASK6_ONLY.json"
    while not marker.exists():
        ready, counts = task6_train_ready()
        write_status("WAIT_TASK6_TRAIN", **counts)
        if ready:
            try:
                run("per_task_visual_context_early.py", "--task", "6", "--out", str(frozen))
            except subprocess.CalledProcessError as exc:
                write_status("TASK6_FREEZE_RETRY_AFTER_INCOMPLETE_WRITER", returncode=exc.returncode)
                time.sleep(30)
                continue
        else:
            time.sleep(30)
    cv = frozen / "TASK6_GROUP_HELDOUT_CV_SUMMARY.json"
    if not cv.exists():
        write_status("RUN_TASK6_ROOT_HELDOUT_CV")
        run("per_task_visual_context_cv.py", "--task", "6", "--out", str(frozen))
    out = ROOT / "task6_dev_validation"
    out.mkdir(exist_ok=True)
    protocol = out / "TASK6_VISUAL_GENERALIZATION_PROTOCOL.json"
    if not protocol.exists():
        write_status("FREEZE_TASK6_DEV_PROTOCOL")
        run("per_task_visual_generalization.py", "freeze", "--task", "6", "--frozen", str(frozen), "--out", str(out))


def ensure_dev_results() -> None:
    pending = {1, 5, 6}
    while pending:
        snapshots = {}
        for task in sorted(list(pending)):
            frozen = ROOT / f"task{task}_train_frozen"
            out = ROOT / f"task{task}_dev_validation"
            final = out / f"TASK{task}_FINAL_CLASSIFICATION.json"
            if final.exists():
                pending.remove(task)
                continue
            state = validation.readiness(task, frozen)
            snapshots[str(task)] = state
            if state["ready"]:
                write_status(f"EVALUATE_TASK{task}_DEV", readiness=state)
                run("per_task_visual_generalization.py", "evaluate", "--task", str(task), "--frozen", str(frozen), "--out", str(out))
                pending.remove(task)
        if pending:
            write_status("WAIT_TASK_SPECIFIC_DEV", pending=sorted(pending), readiness=snapshots)
            time.sleep(30)


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    ensure_task6_frozen()
    ensure_dev_results()
    run("taskwise_visual_context_compare.py")
    write_status("TASK_SPECIFIC_ANALYSIS_COMPLETE", tasks=[1, 5, 6])


if __name__ == "__main__":
    main()
