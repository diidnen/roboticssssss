#!/usr/bin/env python3
"""Prioritize a complete, independently validated single-task result."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback


BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
REPO = BASE / "RoboTwin"
SOURCE = BASE / "experiments/af_taskforms_forcegrid648_v1"
PYTHON = BASE / "venv_robotwin/bin/python"
RUNTIME_BIN = BASE / "runtime_bin"
TASK = os.environ.get("AF_PRIORITY_TASK", "handover_mic")
if TASK not in {"handover_mic", "dump_bin_bigbin"}:
    raise RuntimeError(f"unsupported AF_PRIORITY_TASK: {TASK}")
DEFAULT_EXPERIMENT = BASE / f"experiments/af_{TASK}_forcegrid324_v1"
EXPERIMENT = Path(os.environ.get("AF_PRIORITY_EXPERIMENT", str(DEFAULT_EXPERIMENT)))
FULL_PARENT_SHA256 = "234082d3843a65ea32a82768ff0b9c80080d175beaffb93747991f2cb1681dc1"
STATUS = EXPERIMENT / "pipeline_status.json"
LOG = EXPERIMENT / "pipeline.log"
SOURCE_COLLECTION_LOG = SOURCE / f"collection_{TASK}_priority.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temp = Path(stream.name)
    temp.replace(path)


def emit(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    payload = f"{timestamp} {message}"
    print(payload, flush=True)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(RUNTIME_BIN) + os.pathsep + env.get("PATH", "")
    env["ROBOTWIN_SUPPRESS_EVAL_CONFIG"] = "1"
    return env


def valid_lines(path: Path) -> list[tuple[bytes, dict]]:
    result = []
    if not path.exists():
        return result
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            result.append((raw if raw.endswith(b"\n") else raw + b"\n", row))
    return result


def source_task_rows() -> list[tuple[bytes, dict]]:
    return [item for item in valid_lines(SOURCE / "branches.jsonl") if item[1].get("task") == TASK]


def pids_with(marker: str) -> list[int]:
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        if marker in command:
            result.append(int(entry.name))
    return sorted(result)


def launch_task_collector() -> int:
    with SOURCE_COLLECTION_LOG.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [
                str(PYTHON),
                "scripts/collect_af_taskforms_forcegrid648.py",
                "--tasks", TASK,
            ],
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=environment(),
            start_new_session=True,
        )
    emit(f"TASK_COLLECTOR_LAUNCHED task={TASK} pid={process.pid}")
    return process.pid


def descendants(pid: int) -> list[int]:
    found: list[int] = []
    queue = [pid]
    while queue:
        parent = queue.pop()
        path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = [int(value) for value in path.read_text().split()]
        except (OSError, ValueError):
            children = []
        found.extend(children)
        queue.extend(children)
    return found


def terminate_exact(pids: list[int], *, include_descendants: bool) -> None:
    targets: list[int] = []
    for pid in pids:
        if include_descendants:
            targets.extend(descendants(pid))
        targets.append(pid)
    targets = list(dict.fromkeys(targets))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in targets:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.time() + 8
        while time.time() < deadline:
            if not any(Path(f"/proc/{pid}").exists() for pid in targets):
                return
            time.sleep(0.2)


def pause_full_pipeline() -> None:
    supervisors = pids_with("run_af_forcegrid648_pipeline.py")
    collectors = pids_with("collect_af_taskforms_forcegrid648.py")
    emit(f"PAUSE_FULL supervisors={supervisors} collectors={collectors}")
    terminate_exact(supervisors, include_descendants=False)
    terminate_exact(collectors, include_descendants=True)
    source_rows = len(valid_lines(SOURCE / "branches.jsonl"))
    atomic_json(
        SOURCE / "pipeline_status.json",
        {
            "status": "paused",
            "stage": "paused_after_single_task_priority",
            "collection_rows": source_rows,
            "reason": f"user-approved {TASK} single-task priority on 2026-09-12",
            "resume_safe": True,
            "updated_at": time.time(),
        },
    )
    emit(f"FULL_PIPELINE_PAUSED source_rows={source_rows}")


def materialize() -> None:
    source_parent = SOURCE / "parent_216.snapshot.jsonl"
    if sha256(source_parent) != FULL_PARENT_SHA256:
        raise RuntimeError("sealed 216-row parent hash mismatch")
    parent = [item for item in valid_lines(source_parent) if item[1].get("task") == TASK]
    records = source_task_rows()
    if len(parent) != 108 or len(records) != 324:
        raise RuntimeError(f"{TASK} materialization counts parent={len(parent)} records={len(records)}")
    if [row for _, row in records[:108]] != [row for _, row in parent]:
        raise RuntimeError("filtered parent is not a logical prefix")
    parent_path = EXPERIMENT / "parent_108.snapshot.jsonl"
    records_path = EXPERIMENT / "branches.jsonl"
    atomic_bytes(parent_path, b"".join(raw for raw, _ in parent))
    atomic_bytes(records_path, b"".join(raw for raw, _ in records))
    source_root_map = json.loads((SOURCE / "root_map.json").read_text(encoding="utf-8"))
    filtered_roots = {
        key: value
        for key, value in source_root_map["roots"].items()
        if key.startswith(TASK + "|")
    }
    if len(filtered_roots) != 12:
        raise RuntimeError(f"expected 12 {TASK} roots, found {len(filtered_roots)}")
    root_map = {**source_root_map, "roots": filtered_roots, "task_scope": [TASK]}
    atomic_json(EXPERIMENT / "root_map.json", root_map)
    atomic_json(
        EXPERIMENT / "manifest.json",
        {
            "schema_id": "AF_SINGLE_TASK_FORCEGRID324_MANIFEST_V1",
            "scope": "single_task_priority",
            "task": TASK,
            "scope_change_approved_at": "2026-09-12",
            "forbidden_claim": "cross-task-form generalization",
            "source_experiment": str(SOURCE),
            "source_rows_at_pause": len(valid_lines(SOURCE / "branches.jsonl")),
            "source_records_sha256_at_pause": sha256(SOURCE / "branches.jsonl"),
            "source_parent_216_sha256": FULL_PARENT_SHA256,
            "parent_108_sha256": sha256(parent_path),
            "records_sha256": sha256(records_path),
            "rows": 324,
            "contexts": 36,
            "force_support_n": [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0],
            "split_counts": {"train": 216, "validation": 54, "test": 54},
        },
    )
    emit(f"MATERIALIZED records_sha256={sha256(records_path)}")


def run_stage(name: str, command: list[str], log_name: str) -> None:
    emit(f"STAGE_START {name}")
    atomic_json(
        STATUS,
        {
            "status": "running",
            "stage": name,
            "development_rows": len(valid_lines(EXPERIMENT / "branches.jsonl")),
            "updated_at": time.time(),
        },
    )
    with (EXPERIMENT / log_name).open("a", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=environment(),
        )
    if result.returncode != 0:
        raise RuntimeError(f"stage {name} failed with return code {result.returncode}")
    emit(f"STAGE_COMPLETE {name}")


def main() -> int:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    try:
        last = None
        restart_count = 0
        restart_at = None
        while len(source_task_rows()) < 324:
            count = len(source_task_rows())
            if count != last:
                emit(f"WAIT_TASK task={TASK} rows={count}/324")
                last = count
            collector_pids = pids_with("collect_af_taskforms_forcegrid648.py")
            if not collector_pids:
                if restart_at == count:
                    restart_count += 1
                else:
                    restart_at = count
                    restart_count = 1
                if restart_count > 3:
                    raise RuntimeError(
                        f"source collector failed repeatedly at {TASK} row {count}"
                    )
                collector_pids = [launch_task_collector()]
            elif count != restart_at:
                restart_at = count
                restart_count = 0
            atomic_json(
                STATUS,
                {
                    "status": "running",
                    "stage": "task_collection",
                    "task": TASK,
                    "development_rows": count,
                    "target_rows": 324,
                    "collector_pids": collector_pids,
                    "updated_at": time.time(),
                },
            )
            time.sleep(30)

        if not (EXPERIMENT / "manifest.json").exists():
            pause_full_pipeline()
            materialize()

        manifest = json.loads((EXPERIMENT / "manifest.json").read_text(encoding="utf-8"))
        parent_hash = manifest["parent_108_sha256"]
        audit = EXPERIMENT / "collection_audit.json"
        if not audit.exists() or not json.loads(audit.read_text(encoding="utf-8")).get("passed"):
            run_stage(
                "audit",
                [
                    str(PYTHON), "scripts/audit_af_taskforms_forcegrid648.py",
                    "--records", str(EXPERIMENT / "branches.jsonl"),
                    "--parent-snapshot", str(EXPERIMENT / "parent_108.snapshot.jsonl"),
                    "--out", str(audit),
                    "--task", TASK,
                    "--expected-rows", "324",
                    "--expected-contexts", "36",
                    "--expected-parent-sha256", parent_hash,
                ],
                "audit.log",
            )

        training = EXPERIMENT / "models/training_report.json"
        if not training.exists():
            run_stage(
                "training",
                [
                    str(PYTHON), "scripts/train_af_taskforms_forcegrid648.py",
                    "--records", str(EXPERIMENT / "branches.jsonl"),
                    "--out", str(EXPERIMENT / "models"),
                    "--device", "cpu",
                    "--expected-records", "324",
                    "--expected-split-counts", "216,54,54",
                ],
                "training.log",
            )

        inference = EXPERIMENT / "dense_inference_report.json"
        if not inference.exists():
            run_stage(
                "dense_inference",
                [
                    str(PYTHON), "scripts/evaluate_af_taskforms_forcegrid648.py",
                    "--records", str(EXPERIMENT / "branches.jsonl"),
                    "--training-report", str(training),
                    "--out", str(inference),
                ],
                "dense_inference.log",
            )

        fresh_roots = EXPERIMENT / "fresh_root_map.json"
        if not fresh_roots.exists() or len(
            json.loads(fresh_roots.read_text(encoding="utf-8")).get("roots", {})
        ) != 2:
            run_stage(
                "fresh_root_qualification",
                [
                    str(PYTHON), "scripts/qualify_af_taskform_fresh_roots.py",
                    "--out", str(fresh_roots),
                    "--tasks", TASK,
                ],
                "fresh_root_qualification.log",
            )

        online = EXPERIMENT / "fresh_online_report.json"
        if not online.exists():
            run_stage(
                "fresh_online_comparison",
                [
                    str(PYTHON), "scripts/evaluate_af_taskforms_fresh_online.py",
                    "--experiment", str(EXPERIMENT),
                ],
                "fresh_online.log",
            )

        validation = EXPERIMENT / "independent_validation.json"
        if not validation.exists() or not json.loads(
            validation.read_text(encoding="utf-8")
        ).get("passed"):
            run_stage(
                "independent_validation",
                [
                    str(PYTHON), "scripts/validate_af_handover_pipeline.py",
                    "--experiment", str(EXPERIMENT),
                    "--out", str(validation),
                    "--task", TASK,
                ],
                "independent_validation.log",
            )

        atomic_json(
            STATUS,
            {
                "status": "complete",
                "stage": "complete",
                "development_rows": 324,
                "fresh_online_report": str(online),
                "independent_validation": str(validation),
                "updated_at": time.time(),
            },
        )
        emit("PIPELINE_COMPLETE")
        return 0
    except Exception as error:
        emit(f"PIPELINE_BLOCKED error={type(error).__name__}: {error}")
        atomic_json(
            STATUS,
            {
                "status": "blocked",
                "stage": "unknown",
                "error": str(error),
                "traceback": traceback.format_exc(),
                "updated_at": time.time(),
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
