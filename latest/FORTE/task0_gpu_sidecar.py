#!/usr/bin/env python3
"""GPU side-car orchestration for frozen authoritative task0 DEV.

The prepare phase is read-only with respect to the main experiment.  The
collect phase invokes the existing authoritative worker against an isolated
staging directory and changes only when task0 DEV is collected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FORTE = Path("/home/exouser/FORTE")
SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
REPORT = FORTE / "task0_gpu_sidecar_20260831_050050"
STAGING = FORTE / "task0_gpu_sidecar_staging_20260831_050050"
COLLECTOR = FORTE / "prospective_visual_context_collect.py"
TARGET_SOURCE = SOURCE / "PROSPECTIVE_DEV_TARGET_MANIFEST.json"
CONTEXT_SOURCE = SOURCE / "PROSPECTIVE_CONTEXT_MANIFEST.csv"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def command(args: list[str]) -> dict:
    p = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"argv": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def gpu_snapshot() -> dict:
    q = command([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ])
    a = command([
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ])
    procs = []
    for line in a["stdout"].splitlines() if a["returncode"] == 0 else []:
        c = [x.strip() for x in line.split(",", 3)]
        if len(c) != 4:
            continue
        ps = command(["ps", "-fp", c[0]])
        procs.append({"pid": int(c[0]), "gpu_uuid": c[1], "used_memory_MiB": int(c[2]), "process_name": c[3], "ps": ps["stdout"]})
    return {"captured_at_utc": utcnow(), "gpu_query": q, "process_query": a, "processes": procs}


def load_contexts() -> list[dict]:
    with CONTEXT_SOURCE.open(newline="") as f:
        return list(csv.DictReader(f))


def frozen_task0() -> tuple[dict, list[dict], list[dict]]:
    full = json.loads(TARGET_SOURCE.read_text())
    targets = {k: v for k, v in full["contexts"].items() if k.startswith("pv_dev_t0_")}
    contexts = [r for r in load_contexts() if r["split"] == "DEV" and r["task"] == "0"]
    by_id = {r["context_id"]: r for r in contexts}
    rows = []
    for cid, branches in sorted(targets.items()):
        c = by_id[cid]
        for b in branches:
            rows.append({
                "context_id": cid,
                "root_id": c["root_id"],
                "source_root_id": c["source_root_id"],
                "root_index": int(c["root_index"]),
                "root_seed": int(c["root_seed"]),
                "task": 0,
                "split": "DEV",
                "friction_band": c["friction_band"],
                "mu_GT": float(c["mu_GT"]),
                "branch_label": b["branch_label"],
                "force_N": float(b["force_N"]),
                "repeat_index": int(b["repeat_index"]),
            })
    target = {
        "manifest_name": "TASK0_GPU_DEV_FROZEN_MANIFEST",
        "split": "DEV",
        "contexts": targets,
        "expected_contexts": len(targets),
        "expected_force_cells": len({(r["context_id"], r["force_N"]) for r in rows}),
        "expected_branches": len(rows),
        "branch_state": full.get("branch_state"),
        "source_manifest": str(TARGET_SOURCE),
        "source_manifest_sha256": sha256(TARGET_SOURCE),
    }
    return target, contexts, rows


def prepare() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    target, contexts, rows = frozen_task0()
    if len(contexts) != 2 or len(rows) != 90:
        raise RuntimeError(f"frozen task0 population mismatch contexts={len(contexts)} branches={len(rows)}")
    force_counts = {}
    for row in rows:
        key = (row["context_id"], row["force_N"])
        force_counts[key] = force_counts.get(key, 0) + 1
    if len(force_counts) != 18 or set(force_counts.values()) != {5}:
        raise RuntimeError("task0 force/repeat structure is not 2x9x5")

    write_csv(REPORT / "TASK0_GPU_DEV_FROZEN_MANIFEST.csv", rows)
    write_json(REPORT / "TASK0_GPU_DEV_FROZEN_MANIFEST.json", target)
    write_json(STAGING / "TASK0_GPU_DEV_FROZEN_MANIFEST.json", target)
    write_json(STAGING / "ALL_CONTEXTS_FOR_WORKER.json", load_contexts())
    manifest_hashes = {
        "csv_sha256": sha256(REPORT / "TASK0_GPU_DEV_FROZEN_MANIFEST.csv"),
        "json_sha256": sha256(REPORT / "TASK0_GPU_DEV_FROZEN_MANIFEST.json"),
        "frozen_before_outcomes": True,
        "frozen_at_utc": utcnow(),
    }
    write_json(REPORT / "TASK0_GPU_DEV_FROZEN_MANIFEST_SHA256.json", manifest_hashes)

    skip_logic = {
        "audited_at_utc": utcnow(),
        "collector": str(COLLECTOR),
        "collector_sha256": sha256(COLLECTOR),
        "completion_logic": {
            "task_scope": "run_task reads taskN/taskN/context.csv when present",
            "completed_context_condition": "context_id row with strict_matched == 1",
            "skipped_work": "all frozen target context IDs found in completed_context_condition",
            "directory_existence_alone_causes_skip": False,
            "branch_count_checked_by_run_task_before_skip": False,
            "complete_marker_checked_by_run_task_before_skip": False,
            "hash_checked_by_run_task_before_skip": False,
            "worker_exit_status": "outer COLLECTION_RUN_MANIFEST status is COMPLETE iff all invoked task workers return 0",
        },
        "source_lines": {
            "completed_csv": [302, 308],
            "ids_remaining": 309,
            "empty_ids_early_success": [324, 329],
            "outer_complete_status": 369,
        },
        "risk": "A partial context.csv containing strict_matched rows could cause those contexts to be skipped even without 90-branch verification.",
        "mitigation": "All side-car output remains in isolated staging until independent 90/90 QA passes; no collection_dev path is created during collection.",
        "post_commit_verification_required": "both task0 context IDs strict_matched plus exactly 90 branch rows and 18 cells x5",
    }
    write_json(REPORT / "TASK0_MAIN_PIPELINE_SKIP_LOGIC.json", skip_logic)

    snap = gpu_snapshot()
    gpu_line = snap["gpu_query"]["stdout"].strip().split(",") if snap["gpu_query"]["returncode"] == 0 else []
    free_mib = int(gpu_line[5].strip()) if len(gpu_line) >= 7 else None
    worker_mib = max([p["used_memory_MiB"] for p in snap["processes"] if "env_isaaclab51" in p["process_name"]] or [6800])
    projected_free = free_mib - worker_mib if free_mib is not None else None
    margin = 8192
    safe = projected_free is not None and projected_free >= margin
    preflight = {
        **snap,
        "available_gpu_count": len(snap["gpu_query"]["stdout"].splitlines()) if snap["gpu_query"]["returncode"] == 0 else 0,
        "selected_physical_gpu": 0,
        "sidecar_CUDA_VISIBLE_DEVICES": "0",
        "selection_reason": "only visible GPU; same-GPU use explicitly authorized",
        "existing_authoritative_isaac_worker_memory_estimate_MiB": worker_mib,
        "visual_pi0_server_already_resident": any("openpi-venv" in p["process_name"] for p in snap["processes"]),
        "free_before_launch_MiB": free_mib,
        "projected_free_after_one_more_isaac_worker_MiB": projected_free,
        "required_conservative_residual_margin_MiB": margin,
        "safe_to_launch": safe,
        "no_existing_process_will_be_modified": True,
        "main_output": str(SOURCE),
        "staging_output": str(STAGING),
    }
    write_json(REPORT / "TASK0_GPU_SIDECAR_PREFLIGHT.json", preflight)
    if not safe:
        raise RuntimeError(f"GPU preflight unsafe: projected_free={projected_free} margin={margin}")
    print(json.dumps({"status": "PREPARED", "report": str(REPORT), "staging": str(STAGING), "manifest": manifest_hashes, "gpu_safe": safe}, indent=2))


def collect() -> None:
    spec = importlib.util.spec_from_file_location("authoritative_collector", COLLECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load authoritative collector")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    target = STAGING / "TASK0_GPU_DEV_FROZEN_MANIFEST.json"
    contexts = STAGING / "ALL_CONTEXTS_FOR_WORKER.json"
    result = mod.run_task(STAGING, target, contexts, 0, 21600)
    write_csv(REPORT / "TASK0_GPU_DEV_RUN_MANIFEST.csv", [result])
    write_json(STAGING / "COLLECTION_RUN_MANIFEST.json", {
        "split": "DEV",
        "scope": "task0_only_gpu_sidecar",
        "workers": [result],
        "status": "COMPLETE" if int(result["returncode"]) == 0 else "ENGINEERING_FAILURE",
    })
    print(json.dumps(result, indent=2))
    if int(result["returncode"]) != 0:
        raise SystemExit(int(result["returncode"]) or 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "collect"])
    args = ap.parse_args()
    {"prepare": prepare, "collect": collect}[args.phase]()


if __name__ == "__main__":
    main()
