#!/usr/bin/env python3
"""Safe process-level sharding for the frozen task0 TEST collection.

This wrapper does not modify the authoritative collector.  It uses the same
frozen prospective worker with disjoint context IDs and isolated output trees.
It must only be run after the current single worker has reached an atomic
context boundary and exited.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831"
PLAN_DIR = OUT / "collection_plans"
COLLECTION = OUT / "collection_test"
SHARD_ROOT = OUT / "collection_test_shards" / "task0"
TARGET = PLAN_DIR / "TEST_TARGET_MANIFEST.json"
CONTEXTS = PLAN_DIR / "TEST_CONTEXTS.json"
COLLECTOR = ROOT / "prospective_visual_context_collect.py"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
TABERO = Path("/home/exouser/Tabero")
WARP_CORE = Path("/home/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64")
CLIENT_SRC = TABERO / "benchmarks/openpi/openpi-client/src"
PI0_PORT = os.environ.get("VISUAL_PI0_PORT", "18881")
VRAM_LIMIT_MIB = 34000


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def active_collectors() -> list[dict]:
    me = os.getpid()
    rows = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except Exception:
            continue
        if "prospective_visual_context_collect.py --worker" in cmd or "root_scaling_collect.py" in cmd:
            rows.append({"pid": int(p.name), "cmdline": cmd.strip()})
    return rows


def verify_freeze() -> None:
    obj = json.loads((OUT / "ROOT_SCALING_FREEZE_SHA256.json").read_text())
    if obj.get("status") != "FROZEN":
        raise RuntimeError("root-scaling freeze is not FROZEN")
    for raw, expected in obj["hashes"].items():
        got = sha256(Path(raw))
        if got != expected:
            raise RuntimeError(f"frozen input changed: {raw}: {got} != {expected}")
    if sha256(COLLECTOR) != obj["hashes"].get(str(COLLECTOR), sha256(COLLECTOR)):
        raise RuntimeError("prospective collector hash is not frozen")


def target_ids() -> list[str]:
    obj = json.loads(TARGET.read_text())
    return sorted(cid for cid in obj["contexts"] if "_t0_" in cid)


def committed_main_ids() -> set[str]:
    p = COLLECTION / "task0/task0/context.csv"
    if not p.exists():
        return set()
    with p.open(newline="") as f:
        return {str(r["context_id"]) for r in csv.DictReader(f) if str(r.get("strict_matched", "0")) == "1"}


def make_plan() -> dict:
    verify_freeze()
    ids = target_ids()
    done = committed_main_ids()
    unknown = done - set(ids)
    if unknown:
        raise RuntimeError(f"main TEST output contains unknown task0 contexts: {sorted(unknown)}")
    remaining = [cid for cid in ids if cid not in done]
    if not remaining:
        raise RuntimeError("no remaining task0 TEST contexts to shard")
    # Fixed deterministic order; no outcome-dependent assignment.
    mid = (len(remaining) + 1) // 2
    shards = {"shard0": remaining[:mid], "shard1": remaining[mid:]}
    plan = {
        "status": "FROZEN_DISJOINT_SHARDS",
        "created_utc": utc(),
        "task": 0,
        "split": "TEST",
        "target_manifest": str(TARGET),
        "target_manifest_sha256": sha256(TARGET),
        "context_manifest": str(CONTEXTS),
        "context_manifest_sha256": sha256(CONTEXTS),
        "collector": str(COLLECTOR),
        "collector_sha256": sha256(COLLECTOR),
        "all_target_context_ids": ids,
        "already_atomic_main_context_ids": sorted(done),
        "shards": shards,
        "shard_assignment_rule": "sorted remaining context IDs, first ceil(n/2) to shard0, rest to shard1",
        "outcome_dependent_selection": False,
        "scientific_retry": 0,
        "worker_count": 2,
        "num_envs": 1,
        "physics_settings_changed": False,
        "pi0_changed": False,
    }
    write_json(SHARD_ROOT / "TASK0_TEST_SHARD_PLAN.json", plan)
    return plan


def worker_env(shard_dir: Path, ids: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(TABERO), str(CLIENT_SRC), str(ROOT)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(TABERO),
        "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
        "P5S0C_OUT": str(shard_dir / "task0"), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": "0",
        "P5S0C_TARGET_MANIFEST": str(TARGET), "P5S0C_CONTEXT_IDS": ",".join(ids),
        "P5S0C_SKIP_REPLAY": "1", "PVP_COLLECTION_OUT": str(shard_dir),
        "PVP_CONTEXT_MANIFEST": str(CONTEXTS), "VISUAL_PI0_PORT": PI0_PORT,
    })
    return env


def read_vram_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        )
        return int(out.strip().splitlines()[0].strip())
    except Exception:
        return None


def terminate_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run_plan(plan: dict) -> None:
    rivals = active_collectors()
    if rivals:
        raise RuntimeError("cannot start shards while collector is active: " + json.dumps(rivals))
    # Require the main worker to have committed every context it already owns.
    main_done = committed_main_ids()
    expected_main = set(plan["already_atomic_main_context_ids"])
    if not expected_main.issubset(main_done):
        raise RuntimeError("main atomic boundary from shard plan is not present")
    procs: dict[str, subprocess.Popen] = {}
    records: dict[str, dict] = {}
    try:
        for name, planned_ids in plan["shards"].items():
            shard_dir = SHARD_ROOT / name
            existing = committed_ids(shard_dir)
            ids = [cid for cid in planned_ids if cid not in existing]
            if not ids:
                records[name] = {"status": "ALREADY_COMMITTED", "context_ids": planned_ids, "returncode": 0}
                continue
            shard_dir.mkdir(parents=True, exist_ok=True)
            log = (shard_dir / "worker.log").open("a")
            proc = subprocess.Popen(
                [str(ISAAC_PY), "-u", str(COLLECTOR), "--worker"],
                cwd=TABERO, env=worker_env(shard_dir, ids), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            procs[name] = proc
            records[name] = {"status": "RUNNING", "context_ids": planned_ids,
                             "requested_this_run": ids, "pid": proc.pid,
                             "started_utc": utc(), "output": str(shard_dir)}
        write_json(SHARD_ROOT / "TASK0_TEST_SHARD_RUN.json", {"status": "RUNNING", "plan": plan, "shards": records})
        while procs:
            vram = read_vram_mib()
            if vram is not None and vram >= VRAM_LIMIT_MIB:
                for proc in procs.values():
                    terminate_group(proc)
                raise RuntimeError(f"VRAM safety limit exceeded: {vram} MiB >= {VRAM_LIMIT_MIB} MiB")
            for name, proc in list(procs.items()):
                rc = proc.poll()
                if rc is not None:
                    records[name].update({"status": "COMPLETE" if rc == 0 else "FAILED",
                                          "returncode": rc, "ended_utc": utc()})
                    del procs[name]
            write_json(SHARD_ROOT / "TASK0_TEST_SHARD_RUN.json", {
                "status": "RUNNING", "plan": plan, "shards": records,
                "vram_used_mib": vram, "vram_limit_mib": VRAM_LIMIT_MIB,
            })
            if procs:
                time.sleep(5)
        failed = [name for name, row in records.items() if row.get("returncode", 0) != 0]
        write_json(SHARD_ROOT / "TASK0_TEST_SHARD_RUN.json", {
            "status": "COMPLETE" if not failed else "FAILED",
            "plan": plan, "shards": records, "failed_shards": failed,
            "scientific_retry_count": 0,
        })
        if failed:
            raise RuntimeError(f"shards failed: {failed}")
    except Exception:
        for proc in procs.values():
            terminate_group(proc)
        raise


def committed_ids(shard_dir: Path) -> set[str]:
    p = shard_dir / "task0/task0/context.csv"
    if not p.exists():
        return set()
    with p.open(newline="") as f:
        return {str(r["context_id"]) for r in csv.DictReader(f) if str(r.get("strict_matched", "0")) == "1"}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["plan", "run"])
    args = ap.parse_args()
    plan = make_plan()
    if args.action == "run":
        run_plan(plan)
    else:
        print(json.dumps(plan, indent=2))
