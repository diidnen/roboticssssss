#!/usr/bin/env python3
"""Strict identity, parity, telemetry, and hash QA for root scaling data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831"
PLAN = OUT / "collection_plans"
TASKS = [0, 5]
REQUIRED_TELEMETRY = {
    "cmd_x", "cmd_y", "cmd_z", "phase", "measured_force_N",
    "left_normal_force_N", "right_normal_force_N", "left_tangential_force_N",
    "right_tangential_force_N", "object_vx_mps", "object_vy_mps", "object_vz_mps",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(path)).encode())
        h.update(bytes.fromhex(sha256(p)))
    return h.hexdigest()


def write_json(path: Path, value, atomic: bool = False) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text); os.replace(tmp, path)
    else:
        path.write_text(text)


def verify_freeze() -> None:
    obj = json.loads((OUT / "ROOT_SCALING_FREEZE_SHA256.json").read_text())
    for raw, expected in obj["hashes"].items():
        if sha256(Path(raw)) != expected:
            raise RuntimeError(f"frozen input changed: {raw}")


def audit(split: str, task: int) -> dict:
    verify_freeze()
    root = OUT / f"collection_{split.lower()}"
    target_path = PLAN / f"{split}_TARGET_MANIFEST.json"
    target_all = json.loads(target_path.read_text())
    target = {cid: specs for cid, specs in target_all["contexts"].items() if f"_t{task}_" in cid}
    expected = {(cid, str(spec["branch_label"])): float(spec["force_N"])
                for cid, specs in target.items() for spec in specs}
    context_path = root / f"task{task}/task{task}/context.csv"
    branch_path = root / f"task{task}/task{task}/branches.csv"
    visual_path = root / "visual_alignment_worker.csv"
    missing = [str(p) for p in (context_path, branch_path, visual_path) if not p.exists()]
    if missing:
        raise RuntimeError("collection incomplete: " + json.dumps(missing))
    contexts = pd.read_csv(context_path)
    branches = pd.read_csv(branch_path)
    visual_all = pd.read_csv(visual_path)
    expected_ids = set(target)
    contexts = contexts[contexts.context_id.astype(str).isin(expected_ids)].copy()
    branches = branches[branches.context_id.astype(str).isin(expected_ids)].copy()
    visual = visual_all[visual_all.context_id.astype(str).isin(expected_ids)].copy()
    failures: list[str] = []
    actual_ids = set(contexts.context_id.astype(str))
    if actual_ids != expected_ids or contexts.context_id.nunique() != len(expected_ids):
        failures.append("context identity/count mismatch")
    if "strict_matched" not in contexts or int(contexts.strict_matched.sum()) != len(expected_ids):
        failures.append("not all contexts strict_matched")

    keys = list(zip(branches.context_id.astype(str), branches.branch_label.astype(str)))
    if len(keys) != len(expected) or len(set(keys)) != len(expected) or set(keys) != set(expected):
        failures.append("branch target identity/count/uniqueness mismatch")
    if "state_parity" not in branches or int(branches.state_parity.sum()) != len(branches):
        failures.append("branch state parity incomplete")
    mismatch = [(str(r.context_id), str(r.branch_label)) for r in branches.itertuples(index=False)
                if (str(r.context_id), str(r.branch_label)) in expected and
                abs(float(r.requested_force_N) - expected[(str(r.context_id), str(r.branch_label))]) > 1e-9]
    if mismatch:
        failures.append(f"requested force mismatch count={len(mismatch)}")
    cells = branches.groupby([branches.context_id.astype(str), branches.requested_force_N.astype(float)]).size()
    repeats = 5 if split == "TEST" else 2
    expected_cells = len(expected_ids) * (9 if split == "TEST" else 5)
    if len(cells) != expected_cells or not bool((cells == repeats).all()):
        failures.append("force-cell/repeat structure mismatch")
    if "full_task_success_y" not in branches or branches.full_task_success_y.isna().any():
        failures.append("direct full-task outcome missing")

    visual_failures: list[str] = []
    if len(visual) != len(expected_ids) or visual.context_id.nunique() != len(expected_ids):
        visual_failures.append("visual identity/count")
    for r in visual.to_dict("records"):
        cid = str(r["context_id"])
        if int(r.get("restore_exact", 0)) != 1:
            visual_failures.append(f"{cid}:restore_exact")
        for key in ("restored_state_hash", "second_restore_hash"):
            if str(r.get(key, "")) != str(r.get("snapshot_state_hash", "")):
                visual_failures.append(f"{cid}:{key}")
        for pcol, hcol in (("camera0_rgb_path", "camera0_rgb_sha256"),
                           ("camera1_rgb_path", "camera1_rgb_sha256"),
                           ("visual_feature_path", "visual_feature_sha256")):
            p = Path(str(r.get(pcol, "")))
            if not p.exists():
                visual_failures.append(f"{cid}:{pcol}:missing"); continue
            arr = np.load(p, allow_pickle=False)
            got = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
            if got != str(r.get(hcol, "")):
                visual_failures.append(f"{cid}:{pcol}:hash")
        fp = Path(str(r.get("visual_feature_path", "")))
        if fp.exists():
            arr = np.load(fp, allow_pickle=False)
            if arr.shape != (4096,) or arr.dtype != np.float32:
                visual_failures.append(f"{cid}:feature-shape-dtype")
    if visual_failures:
        failures.append(f"visual failures={len(visual_failures)}")

    telemetry_failures: list[str] = []
    telemetry_steps = 0
    for r in branches.to_dict("records"):
        p = Path(str(r.get("telemetry_path", "")))
        if not p.exists():
            telemetry_failures.append(f"{r.get('branch_id')}:missing"); continue
        d = pd.read_csv(p); telemetry_steps += len(d)
        if len(d) < 9 or not REQUIRED_TELEMETRY <= set(d):
            telemetry_failures.append(f"{r.get('branch_id')}:columns-or-length"); continue
        finite = sorted(REQUIRED_TELEMETRY - {"phase"})
        if not np.isfinite(d[finite].to_numpy(float)).all():
            telemetry_failures.append(f"{r.get('branch_id')}:nonfinite")
    if telemetry_failures:
        failures.append(f"telemetry failures={len(telemetry_failures)}")

    run_path = root / f"TASK{task}_{split}_RUN.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    checks = {
        "exact_context_ids": actual_ids == expected_ids,
        "exact_unique_branch_targets": len(keys) == len(expected) and len(set(keys)) == len(expected) and set(keys) == set(expected),
        "state_parity_all": "state_parity" in branches and int(branches.state_parity.sum()) == len(branches),
        "force_cells_and_repeats_exact": len(cells) == expected_cells and bool((cells == repeats).all()),
        "direct_full_task_outcomes": "full_task_success_y" in branches and not branches.full_task_success_y.isna().any(),
        "visual_alignment_and_hashes": not visual_failures and len(visual) == len(expected_ids),
        "corrected_telemetry": not telemetry_failures,
        "no_scientific_retry_or_duplicate": len(keys) == len(set(keys)) == len(expected),
        "task_exact": set(branches.task.astype(int)) == {task},
        "split_exact": set(branches.split.astype(str)) == {split},
    }
    result = {
        "status": "PASS" if not failures and all(checks.values()) else "FAIL",
        "task": task, "split": split, "checks": checks, "failures": failures,
        "counts": {"contexts": len(contexts), "independent_roots": int(contexts.root_id.nunique()),
                   "force_cells": len(cells), "branches": len(branches),
                   "successes": int(branches.full_task_success_y.sum()) if "full_task_success_y" in branches else None,
                   "scientific_failures": int(len(branches) - branches.full_task_success_y.sum()) if "full_task_success_y" in branches else None,
                   "visual_contexts": len(visual), "raw_physical_timesteps": telemetry_steps},
        "infrastructure_status": run.get("status", "UNKNOWN"), "infrastructure_returncode": run.get("returncode"),
        "scientific_retry_count": 0 if checks["no_scientific_retry_or_duplicate"] else None,
        "target_manifest": str(target_path), "target_manifest_sha256": sha256(target_path),
        "context_csv": str(context_path), "context_csv_sha256": sha256(context_path),
        "branch_csv": str(branch_path), "branch_csv_sha256": sha256(branch_path),
        "visual_csv": str(visual_path), "visual_csv_sha256": sha256(visual_path),
        "telemetry_tree_sha256": tree_sha256(root / f"task{task}/task{task}/P5S0C_BRANCH_TELEMETRY"),
        "visual_failures": visual_failures, "telemetry_failures": telemetry_failures,
    }
    audit_path = OUT / f"TASK{task}_{split}_COLLECTION_AUDIT.json"
    write_json(audit_path, result)
    if result["status"] != "PASS":
        raise RuntimeError(f"task{task}/{split} audit failed: {failures[:4]}")
    if split == "TEST":
        commit = {
            "status": "ATOMICALLY_COMMITTED_UNTOUCHED_TEST", "task": task,
            "test_manifest": str(OUT / "ROOT_SCALING_TEST_MANIFEST.json"),
            "test_manifest_sha256": sha256(OUT / "ROOT_SCALING_TEST_MANIFEST.json"),
            "audit": str(audit_path), "audit_sha256": sha256(audit_path),
            "context_csv_sha256": sha256(context_path), "branch_csv_sha256": sha256(branch_path),
            "visual_csv_sha256": sha256(visual_path), "telemetry_tree_sha256": result["telemetry_tree_sha256"],
            "PCA_fit_forbidden": True, "normalization_fit_forbidden": True,
            "model_checkpoint_seed_selection_forbidden": True, "calibration_tuning_forbidden": True,
        }
        write_json(OUT / f"TASK{task}_TEST_COLLECTION_COMMIT.json", commit, atomic=True)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["TEST", "TRAIN"], required=True)
    ap.add_argument("--task", type=int, choices=TASKS, required=True)
    args = ap.parse_args()
    audit(args.split, args.task)
