#!/usr/bin/env python3
"""Sequential, resume-safe collector for frozen root-scaling TEST/TRAIN plans."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import prospective_visual_context_collect as pvc


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831"
PLAN = OUT / "collection_plans"
COLLECTOR = ROOT / "prospective_visual_context_collect.py"
TASKS = [0, 5]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader(); w.writerows(rows)


def verify_freeze() -> dict:
    obj = json.loads((OUT / "ROOT_SCALING_FREEZE_SHA256.json").read_text())
    if obj.get("status") != "FROZEN":
        raise RuntimeError("root-scaling freeze status invalid")
    for raw, expected in obj["hashes"].items():
        path = Path(raw)
        got = sha256(path)
        if got != expected:
            raise RuntimeError(f"frozen input changed: {path}: {got} != {expected}")
    return obj


def competing_collectors() -> list[dict]:
    me = os.getpid()
    rows = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except Exception:
            continue
        if ("prospective_visual_context_collect.py --worker" in cmd or
                "root_scaling_collect.py" in cmd):
            rows.append({"pid": int(p.name), "cmdline": cmd.strip()})
    return rows


def completed_ids(out: Path, task: int) -> set[str]:
    path = out / f"task{task}/task{task}/context.csv"
    if not path.exists():
        return set()
    return {str(r["context_id"]) for r in csv.DictReader(path.open(newline=""))
            if str(r.get("strict_matched", "0")) == "1"}


def collect(split: str, task: int, timeout_s: int) -> None:
    verify_freeze()
    if task not in TASKS:
        raise ValueError(task)
    if split == "TRAIN":
        for t in TASKS:
            commit = OUT / f"TASK{t}_TEST_COLLECTION_COMMIT.json"
            if not commit.exists() or json.loads(commit.read_text()).get("status") != "ATOMICALLY_COMMITTED_UNTOUCHED_TEST":
                raise RuntimeError("all untouched TEST collections must pass QA and commit before TRAIN collection")
    rivals = competing_collectors()
    if rivals:
        raise RuntimeError("another Isaac collector is already active: " + json.dumps(rivals))

    split_l = split.lower()
    out = OUT / f"collection_{split_l}"
    target = PLAN / f"{split}_TARGET_MANIFEST.json"
    contexts = PLAN / f"{split}_CONTEXTS.json"
    target_obj = json.loads(target.read_text())
    task_ids = sorted(cid for cid in target_obj["contexts"] if f"_t{task}_" in cid)
    done = completed_ids(out, task)
    remaining = [cid for cid in task_ids if cid not in done]
    out.mkdir(parents=True, exist_ok=True)

    preflight = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "split": split, "task": task,
        "target_manifest": str(target), "target_manifest_sha256": sha256(target),
        "context_manifest": str(contexts), "context_manifest_sha256": sha256(contexts),
        "collector": str(COLLECTOR), "collector_sha256": sha256(COLLECTOR),
        "expected_contexts_task": len(task_ids), "already_strict_committed": len(done),
        "remaining_contexts": len(remaining), "scientific_retry": 0,
        "resume_semantics": "only infrastructure-interrupted, not-yet-strict-committed context IDs are resumed",
        "competing_collectors": rivals, "no_unrelated_process_modified": True,
    }
    write_json(out / f"TASK{task}_{split}_PREFLIGHT.json", preflight)
    if not remaining:
        record = {**preflight, "status": "ALREADY_COMMITTED_REQUIRES_QA", "returncode": 0,
                  "elapsed_wall_s": 0.0}
        write_json(out / f"TASK{task}_{split}_RUN.json", record)
        print(json.dumps(record, indent=2)); return

    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join([str(pvc.WARP_CORE), str(pvc.TABERO), str(pvc.CLIENT_SRC), str(ROOT)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(pvc.TABERO),
        "HDF5_TRAJ_SOURCE_DIR": str(pvc.TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(pvc.TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(pvc.TABERO / "benchmarks/datasets/libero/USD"),
        "P5S0C_OUT": str(out / f"task{task}"), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": str(task),
        "P5S0C_TARGET_MANIFEST": str(target), "P5S0C_CONTEXT_IDS": ",".join(remaining),
        "P5S0C_SKIP_REPLAY": "1", "PVP_COLLECTION_OUT": str(out),
        "PVP_CONTEXT_MANIFEST": str(contexts), "VISUAL_PI0_PORT": os.environ.get("VISUAL_PI0_PORT", "18881"),
    })
    log = out / f"logs/task{task}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log.open("a") as fh:
        proc = subprocess.Popen([str(pvc.ISAAC_PY), "-u", str(COLLECTOR), "--worker"],
                                cwd=pvc.TABERO, env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            rc = 124
    record = {
        **preflight,
        "status": "WORKER_COMPLETE_REQUIRES_QA" if rc == 0 else "INFRASTRUCTURE_FAILURE_RESUMABLE",
        "returncode": rc, "elapsed_wall_s": time.time() - start, "worker_pid": proc.pid,
        "contexts_requested_this_run": len(remaining), "log": str(log), "collection_out": str(out),
    }
    write_json(out / f"TASK{task}_{split}_RUN.json", record)
    prior = []
    history = out / "COLLECTION_RUN_HISTORY.csv"
    if history.exists():
        prior = list(csv.DictReader(history.open(newline="")))
    write_csv(history, prior + [record])
    print(json.dumps(record, indent=2))
    if rc != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["TEST", "TRAIN"], required=True)
    ap.add_argument("--task", type=int, choices=TASKS, required=True)
    ap.add_argument("--timeout-s", type=int, default=43200)
    args = ap.parse_args()
    collect(args.split, args.task, args.timeout_s)
