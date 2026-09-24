#!/usr/bin/env python3
"""Isolated model-blind Force-Critical real-sweep collector."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import prospective_visual_context_collect as pvc


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "activeforcing_final_experiment_20260901_045000"
COLLECTION = OUT / "collection_challenge"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def competing() -> list[dict]:
    rows = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid(): continue
        try: cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except Exception: continue
        if "prospective_visual_context_collect.py --worker" in cmd or "force_critical_collect.py --task" in cmd or "root_scaling_collect.py" in cmd:
            rows.append({"pid": int(p.name), "cmdline": cmd.strip()})
    return rows


def main(task: int, timeout: int) -> None:
    protocol = json.loads((OUT / "ACTIVEFORCING_FINAL_EXPERIMENT_PROTOCOL.json").read_text())
    if protocol["status"] != "FROZEN_BEFORE_NEW_CHALLENGE_OUTCOMES": raise RuntimeError("protocol not frozen")
    definition = OUT / "FORCE_CRITICAL_CHALLENGE_DEFINITION.json"
    if sha(definition) != protocol["definition_sha256"]: raise RuntimeError("definition hash changed")
    contexts_path, target = OUT / "CHALLENGE_CONTEXTS.json", OUT / "CHALLENGE_TARGET_MANIFEST.json"
    contexts = json.loads(contexts_path.read_text()); target_obj = json.loads(target.read_text())
    ids = [r["context_id"] for r in contexts if int(r["task"]) == task]
    if task not in [0, 5] or len(ids) != 24: raise RuntimeError("unexpected frozen task pool")
    done_path = COLLECTION / f"task{task}/task{task}/context.csv"
    done = set()
    if done_path.exists():
        done = {str(r["context_id"]) for r in csv.DictReader(done_path.open()) if str(r.get("strict_matched", "0")) == "1"}
    remaining = [x for x in ids if x not in done]
    rivals = competing()
    if rivals: raise RuntimeError("competing collector: " + json.dumps(rivals))
    pre = {"status": "FROZEN_PREFLIGHT", "task": task, "expected_contexts": 24, "expected_branches": 432, "already_committed": len(done), "remaining": len(remaining), "definition_sha256": sha(definition), "contexts_sha256": sha(contexts_path), "target_sha256": sha(target), "model_queries": 0, "untouched_TEST_read": False, "competing_collectors": rivals}
    write_json(COLLECTION / f"TASK{task}_PREFLIGHT.json", pre)
    if not remaining:
        write_json(COLLECTION / f"TASK{task}_RUN.json", {**pre, "status": "ALREADY_COMPLETE_REQUIRES_QA", "returncode": 0}); return
    task_out = COLLECTION / f"task{task}"
    env = os.environ.copy(); env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join([str(pvc.WARP_CORE), str(pvc.TABERO), str(pvc.CLIENT_SRC), str(ROOT)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(pvc.TABERO),
        "HDF5_TRAJ_SOURCE_DIR": str(pvc.TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(pvc.TABERO / "benchmarks/datasets/libero/config"), "LIBERO_ASSETS_DATA_DIR": str(pvc.TABERO / "benchmarks/datasets/libero/USD"),
        "P5S0C_OUT": str(task_out), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": str(task), "P5S0C_TARGET_MANIFEST": str(target),
        "P5S0C_CONTEXT_IDS": ",".join(remaining), "P5S0C_SKIP_REPLAY": "1", "PVP_COLLECTION_OUT": str(COLLECTION),
        "PVP_CONTEXT_MANIFEST": str(contexts_path), "VISUAL_PI0_PORT": "18881",
    })
    log = COLLECTION / f"logs/task{task}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log.open("a") as fh:
        proc = subprocess.Popen([str(pvc.ISAAC_PY), "-u", str(ROOT / "prospective_visual_context_collect.py"), "--worker"], cwd=pvc.TABERO, env=env, stdout=fh, stderr=subprocess.STDOUT)
        try: rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired: proc.terminate(); proc.wait(timeout=60); rc = 124
    write_json(COLLECTION / f"TASK{task}_RUN.json", {**pre, "status": "WORKER_COMPLETE_REQUIRES_QA" if rc == 0 else "INFRASTRUCTURE_FAILURE_RESUMABLE", "returncode": rc, "worker_pid": proc.pid, "elapsed_s": time.time()-start, "log": str(log)})
    if rc: raise SystemExit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", type=int, choices=[0, 5], required=True); ap.add_argument("--timeout-s", type=int, default=43200); a = ap.parse_args(); main(a.task, a.timeout_s)
