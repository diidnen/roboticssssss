#!/usr/bin/env python3
"""Persistent, idempotent pipeline supervisor for the force-grid experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
REPO = BASE / "RoboTwin"
EXPERIMENT = BASE / "experiments/af_taskforms_forcegrid648_v1"
PYTHON = BASE / "venv_robotwin/bin/python"
RUNTIME_BIN = BASE / "runtime_bin"
STATUS = EXPERIMENT / "pipeline_status.json"
PIPELINE_LOG = EXPERIMENT / "pipeline.log"
COLLECTION_LOG = EXPERIMENT / "collection.log"


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    payload = f"{timestamp} {message}"
    print(payload, flush=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(RUNTIME_BIN) + os.pathsep + env.get("PATH", "")
    env["ROBOTWIN_SUPPRESS_EVAL_CONFIG"] = "1"
    return env


def row_count() -> int:
    path = EXPERIMENT / "branches.jsonl"
    if not path.exists():
        return 0
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def collector_pids() -> list[int]:
    result = []
    for child in Path("/proc").iterdir():
        if not child.name.isdigit():
            continue
        try:
            command = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        if "collect_af_taskforms_forcegrid648.py" in command:
            result.append(int(child.name))
    return result


def launch_collector() -> int:
    with COLLECTION_LOG.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [str(PYTHON), "scripts/collect_af_taskforms_forcegrid648.py"],
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=environment(),
            start_new_session=True,
        )
    (EXPERIMENT / "collector.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
    return process.pid


def run_stage(name: str, command: list[str], log_name: str) -> None:
    log(f"STAGE_START {name}")
    atomic_write(
        STATUS,
        {
            "status": "running",
            "stage": name,
            "collection_rows": row_count(),
            "updated_at": time.time(),
        },
    )
    with (EXPERIMENT / log_name).open("a", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=environment(),
        )
    if process.returncode != 0:
        raise RuntimeError(f"stage {name} failed with return code {process.returncode}")
    log(f"STAGE_COMPLETE {name}")


def main() -> int:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    try:
        last_reported = None
        restart_count_at_row = 0
        restart_row = None
        while row_count() < 648:
            count = row_count()
            pids = collector_pids()
            if not pids:
                if restart_row == count:
                    restart_count_at_row += 1
                else:
                    restart_row = count
                    restart_count_at_row = 1
                if restart_count_at_row > 3:
                    raise RuntimeError(
                        f"collector failed repeatedly without progress at row {count}"
                    )
                pid = launch_collector()
                log(f"COLLECTOR_RESTART rows={count}/648 pid={pid}")
                pids = [pid]
            elif count != restart_row:
                restart_row = count
                restart_count_at_row = 0
            if count != last_reported:
                log(f"COLLECTION_PROGRESS rows={count}/648 pids={pids}")
                last_reported = count
            atomic_write(
                STATUS,
                {
                    "status": "running",
                    "stage": "collection",
                    "collection_rows": count,
                    "collector_pids": pids,
                    "updated_at": time.time(),
                },
            )
            time.sleep(60)
        if row_count() != 648:
            raise RuntimeError(f"collection row count exceeded expectation: {row_count()}")
        log("COLLECTION_COMPLETE rows=648/648")

        audit_path = EXPERIMENT / "collection_audit.json"
        if not audit_path.exists() or not json.loads(audit_path.read_text(encoding="utf-8")).get("passed"):
            run_stage(
                "audit",
                [
                    str(PYTHON),
                    "scripts/audit_af_taskforms_forcegrid648.py",
                    "--records", str(EXPERIMENT / "branches.jsonl"),
                    "--parent-snapshot", str(EXPERIMENT / "parent_216.snapshot.jsonl"),
                    "--out", str(audit_path),
                ],
                "audit.log",
            )

        training_path = EXPERIMENT / "models/training_report.json"
        if not training_path.exists():
            run_stage(
                "training",
                [
                    str(PYTHON),
                    "scripts/train_af_taskforms_forcegrid648.py",
                    "--records", str(EXPERIMENT / "branches.jsonl"),
                    "--out", str(EXPERIMENT / "models"),
                    "--device", "cpu",
                ],
                "training.log",
            )

        inference_path = EXPERIMENT / "dense_inference_report.json"
        if not inference_path.exists():
            run_stage(
                "dense_inference",
                [
                    str(PYTHON),
                    "scripts/evaluate_af_taskforms_forcegrid648.py",
                    "--records", str(EXPERIMENT / "branches.jsonl"),
                    "--training-report", str(training_path),
                    "--out", str(inference_path),
                ],
                "dense_inference.log",
            )

        fresh_root_map = EXPERIMENT / "fresh_root_map.json"
        if not fresh_root_map.exists() or len(json.loads(fresh_root_map.read_text(encoding="utf-8")).get("roots", {})) != 4:
            run_stage(
                "fresh_root_qualification",
                [
                    str(PYTHON),
                    "scripts/qualify_af_taskform_fresh_roots.py",
                    "--out", str(fresh_root_map),
                ],
                "fresh_root_qualification.log",
            )

        fresh_report = EXPERIMENT / "fresh_online_report.json"
        if not fresh_report.exists():
            run_stage(
                "fresh_online_comparison",
                [
                    str(PYTHON),
                    "scripts/evaluate_af_taskforms_fresh_online.py",
                    "--experiment", str(EXPERIMENT),
                ],
                "fresh_online.log",
            )

        validation_path = EXPERIMENT / "independent_validation.json"
        if not validation_path.exists() or not json.loads(
            validation_path.read_text(encoding="utf-8")
        ).get("passed"):
            run_stage(
                "independent_validation",
                [
                    str(PYTHON),
                    "scripts/validate_af_forcegrid648_pipeline.py",
                    "--experiment", str(EXPERIMENT),
                    "--out", str(validation_path),
                ],
                "independent_validation.log",
            )

        atomic_write(
            STATUS,
            {
                "status": "complete",
                "stage": "complete",
                "collection_rows": row_count(),
                "fresh_online_report": str(fresh_report),
                "independent_validation": str(validation_path),
                "updated_at": time.time(),
            },
        )
        log("PIPELINE_COMPLETE")
        return 0
    except Exception as error:
        detail = traceback.format_exc()
        log(f"PIPELINE_BLOCKED error={type(error).__name__}: {error}")
        atomic_write(
            STATUS,
            {
                "status": "blocked",
                "stage": "unknown",
                "collection_rows": row_count(),
                "error": str(error),
                "traceback": detail,
                "updated_at": time.time(),
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
