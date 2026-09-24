#!/usr/bin/env python3
"""Collect the frozen task0 context-count TRAIN or TEST population.

The scientific runner and prospective visual-capture worker are unchanged.
This orchestrator only permits the already frozen exact context IDs and target
branches, writes to the isolated context-sample directory, and supports
infrastructure-only resume by fully committed context ID.
"""
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
from task0_context_pipeline_freeze import verify as verify_pipeline_freeze


ROOT = Path("/home/exouser/FORTE")
PROJECT = ROOT / "task0_context_sample_complexity_20260831"
COLLECTOR = ROOT / "prospective_visual_context_collect.py"
SPLIT_MANIFEST = ROOT / "TASK0_CONTEXT_SPLIT_MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader(); w.writerows(rows)


def paths(split: str):
    if split == "TRAIN":
        return (PROJECT / "TASK0_NEW_TRAIN_TARGET_MANIFEST.json",
                PROJECT / "TASK0_NEW_TRAIN_CONTEXTS.json",
                PROJECT / "collection_train_new")
    return (PROJECT / "TASK0_FROZEN_TEST_TARGET_MANIFEST.json",
            PROJECT / "TASK0_FROZEN_TEST_CONTEXTS.json",
            PROJECT / "collection_test_frozen")


def completed_contexts(out: Path) -> set[str]:
    p = out / "task0/task0/context.csv"
    if not p.exists():
        return set()
    return {
        str(r["context_id"]) for r in csv.DictReader(p.open(newline=""))
        if str(r.get("strict_matched", "0")) == "1"
    }


def collect(split: str, timeout_s: int):
    verify_pipeline_freeze()
    target_path, contexts_path, out = paths(split)
    frozen = json.loads(SPLIT_MANIFEST.read_text())
    freeze_hash = json.loads((PROJECT / "TASK0_CONTEXT_SPLIT_FREEZE_SHA256.json").read_text())["hashes"]
    for p in [SPLIT_MANIFEST, target_path, contexts_path]:
        if freeze_hash.get(str(p)) != sha256(p):
            raise RuntimeError(f"frozen manifest hash changed: {p}")
    target = json.loads(target_path.read_text())
    contexts = json.loads(contexts_path.read_text())
    ids = list(target["contexts"])
    plan_ids = [str(x["context_id"]) for x in contexts]
    if set(ids) != set(plan_ids) or len(ids) != len(set(ids)):
        raise RuntimeError("target/context plan identity mismatch")
    expected = 62 if split == "TRAIN" else 10
    if len(ids) != expected:
        raise RuntimeError(f"{split} expected {expected} contexts, got {len(ids)}")
    done = completed_contexts(out)
    remaining = [cid for cid in ids if cid not in done]
    out.mkdir(parents=True, exist_ok=True)
    all_plan = out / "FROZEN_CONTEXT_PLAN_FOR_WORKER.json"
    if all_plan.exists() and sha256(all_plan) != sha256(contexts_path):
        # JSON formatting/path hashes can differ despite identity; compare data.
        if json.loads(all_plan.read_text()) != contexts:
            raise RuntimeError("existing worker context plan changed")
    else:
        write_json(all_plan, contexts)
    preflight = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "split": split,
        "split_manifest": str(SPLIT_MANIFEST), "split_manifest_sha256": sha256(SPLIT_MANIFEST),
        "target_manifest": str(target_path), "target_manifest_sha256": sha256(target_path),
        "collector": str(COLLECTOR), "collector_sha256": sha256(COLLECTOR),
        "expected_contexts": len(ids), "already_committed_contexts": len(done),
        "remaining_contexts": len(remaining), "scientific_retry": 0,
        "resume_condition": "strict_matched context row only; final branch QA still mandatory",
        "outcome_dependent_inclusion": False,
    }
    write_json(out / "COLLECTION_PREFLIGHT.json", preflight)
    if not remaining:
        write_json(out / "COLLECTION_RUN_MANIFEST.json", {
            **preflight, "status": "ALREADY_COLLECTED_REQUIRES_QA", "returncode": 0,
        })
        print(json.dumps({"status": "ALREADY_COLLECTED_REQUIRES_QA", "split": split}, indent=2))
        return

    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join([str(pvc.WARP_CORE), str(pvc.TABERO), str(pvc.CLIENT_SRC), str(ROOT)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(pvc.TABERO),
        "HDF5_TRAJ_SOURCE_DIR": str(pvc.TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(pvc.TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(pvc.TABERO / "benchmarks/datasets/libero/USD"),
        "P5S0C_OUT": str(out / "task0"), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": "0",
        "P5S0C_TARGET_MANIFEST": str(target_path), "P5S0C_CONTEXT_IDS": ",".join(remaining),
        "P5S0C_SKIP_REPLAY": "1", "PVP_COLLECTION_OUT": str(out),
        "PVP_CONTEXT_MANIFEST": str(all_plan),
        "VISUAL_PI0_PORT": os.environ.get("VISUAL_PI0_PORT", "18881"),
    })
    log = out / "logs/task0.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log.open("w") as fh:
        proc = subprocess.Popen([str(pvc.ISAAC_PY), "-u", str(COLLECTOR), "--worker"],
                                cwd=pvc.TABERO, env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.terminate(); proc.wait(timeout=60); rc = 124
    record = {
        **preflight, "status": "WORKER_COMPLETE_REQUIRES_QA" if rc == 0 else "ENGINEERING_FAILURE",
        "returncode": rc, "elapsed_wall_s": time.time() - start, "log": str(log),
        "collection_out": str(out), "contexts_requested_this_run": len(remaining),
    }
    write_json(out / "COLLECTION_RUN_MANIFEST.json", record)
    write_csv(out / "COLLECTION_RUN_MANIFEST.csv", [record])
    print(json.dumps(record, indent=2))
    if rc != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["TRAIN", "TEST"], required=True)
    ap.add_argument("--timeout-s", type=int, default=43200)
    args = ap.parse_args()
    collect(args.split, args.timeout_s)
