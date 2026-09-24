#!/usr/bin/env python3
"""Wait for the current task0 context boundary, then run/QA two TEST shards."""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831"
COLLECTION = OUT / "collection_test"
STATUS = OUT / "TASK0_TEST_SHARD_HANDOFF.json"
PY = Path("/media/volume/newdata/exouser/softvtbench/openpi-venv/bin/python")
SHARDER = ROOT / "run_task0_test_shards.py"
MERGER = ROOT / "merge_task0_test_shards.py"
AUDIT = ROOT / "audit_root_scaling_collection.py"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(status: str, **extra) -> None:
    STATUS.write_text(json.dumps({"status": status, "updated_utc": utc(), **extra}, indent=2, sort_keys=True) + "\n")


def process_rows() -> list[tuple[int, int, str]]:
    rows = []
    me = os.getpid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            stat = (p / "stat").read_text().split()
            pgid = int(stat[4])
        except Exception:
            continue
        if "root_scaling_collect.py --split TEST --task 0" in cmd or "prospective_visual_context_collect.py --worker" in cmd:
            rows.append((int(p.name), pgid, cmd))
    return rows


def root74_atomic() -> tuple[bool, dict]:
    target = json.loads((OUT / "collection_plans/TEST_TARGET_MANIFEST.json").read_text())
    root74 = next(cid for cid in target["contexts"] if "_t0_" in cid and "_r74_" in cid)
    base = COLLECTION / "task0/task0"
    context_rows = []
    branch_rows = []
    parity_rows = []
    for name, out in (("context", base / "context.csv"), ("branches", base / "branches.csv"), ("parity", base / "parity.csv")):
        if not out.exists():
            return False, {"root74_context_id": root74, "missing": name}
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if name == "context": context_rows = rows
        elif name == "branches": branch_rows = rows
        else: parity_rows = rows
    c = [r for r in context_rows if str(r.get("context_id")) == root74 and str(r.get("strict_matched", "0")) == "1"]
    b = [r for r in branch_rows if str(r.get("context_id")) == root74]
    v = [r for r in parity_rows if str(r.get("context_id")) == root74]
    ok = len(c) == 1 and len(b) == 45 and len(v) == 45
    return ok, {"root74_context_id": root74, "context_rows": len(c), "branch_rows": len(b), "parity_rows": len(v)}


def stop_current_group() -> list[dict]:
    rows = process_rows()
    groups = sorted({pgid for _, pgid, _ in rows if pgid != os.getpgrp()})
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 30
    while time.time() < deadline:
        left = [r for r in process_rows() if r[1] in groups]
        if not left:
            return []
        time.sleep(1)
    left = [r for r in process_rows() if r[1] in groups]
    for pgid in sorted({r[1] for r in left}):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return left


def main() -> None:
    write_status("WAITING_FOR_ROOT74_ATOMIC_BOUNDARY")
    while True:
        ok, detail = root74_atomic()
        write_status("WAITING_FOR_ROOT74_ATOMIC_BOUNDARY" if not ok else "ROOT74_ATOMIC_BOUNDARY_DETECTED", boundary=detail,
                     active_processes=[{"pid": p, "pgid": g, "cmd": c} for p, g, c in process_rows()])
        if ok:
            break
        time.sleep(5)

    left = stop_current_group()
    if left:
        raise RuntimeError(f"could not stop old collector group: {left}")
    write_status("OLD_COLLECTOR_STOPPED_AT_ATOMIC_BOUNDARY", boundary=detail)

    run = subprocess.run([str(PY), str(SHARDER), "run"], cwd=ROOT, check=False)
    if run.returncode != 0:
        write_status("SHARDS_FAILED", shard_returncode=run.returncode)
        raise SystemExit(run.returncode)
    write_status("SHARDS_COMPLETE_REQUIRES_MERGE_QA")

    merge = subprocess.run([str(PY), str(MERGER)], cwd=ROOT, check=False)
    if merge.returncode != 0:
        write_status("MERGE_FAILED", merge_returncode=merge.returncode)
        raise SystemExit(merge.returncode)
    qa = subprocess.run([str(PY), str(AUDIT), "--split", "TEST", "--task", "0"], cwd=ROOT, check=False)
    if qa.returncode != 0:
        write_status("QA_FAILED", qa_returncode=qa.returncode)
        raise SystemExit(qa.returncode)
    write_status("COMPLETE_QA_PASS", qa_returncode=0, worker_count=2, num_envs=1)


if __name__ == "__main__":
    main()
