#!/usr/bin/env python3
"""Post-collection integrity audit and atomic TEST commit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from task0_context_pipeline_freeze import verify as verify_pipeline_freeze


ROOT = Path("/home/exouser/FORTE")
PROJECT = ROOT / "task0_context_sample_complexity_20260831"
SPLIT_MANIFEST = ROOT / "TASK0_CONTEXT_SPLIT_MANIFEST.json"
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


def write_json(path: Path, obj, atomic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"
    if atomic:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text); os.replace(tmp, path)
    else:
        path.write_text(text)


def paths(split: str):
    if split == "TRAIN":
        return (PROJECT / "TASK0_NEW_TRAIN_TARGET_MANIFEST.json", PROJECT / "collection_train_new")
    return (PROJECT / "TASK0_FROZEN_TEST_TARGET_MANIFEST.json", PROJECT / "collection_test_frozen")


def audit(split: str):
    verify_pipeline_freeze()
    target_path, out = paths(split)
    target = json.loads(target_path.read_text())
    expected = {
        (cid, str(spec["branch_label"])): float(spec["force_N"])
        for cid, specs in target["contexts"].items() for spec in specs
    }
    context_path = out / "task0/task0/context.csv"
    branch_path = out / "task0/task0/branches.csv"
    visual_path = out / "visual_alignment_worker.csv"
    if not all(p.exists() for p in [context_path, branch_path, visual_path]):
        raise RuntimeError("collection files are incomplete")
    contexts = pd.read_csv(context_path)
    branches = pd.read_csv(branch_path)
    visual = pd.read_csv(visual_path)
    actual_ids = set(contexts.context_id.astype(str))
    expected_ids = set(target["contexts"])
    failures = []
    if actual_ids != expected_ids or contexts.context_id.nunique() != len(expected_ids):
        failures.append("context identity/count mismatch")
    if "strict_matched" not in contexts or int(contexts.strict_matched.sum()) != len(expected_ids):
        failures.append("not all contexts strict_matched")
    actual_keys = list(zip(branches.context_id.astype(str), branches.branch_label.astype(str)))
    if len(actual_keys) != len(expected) or len(set(actual_keys)) != len(expected) or set(actual_keys) != set(expected):
        failures.append("branch target identity/count mismatch")
    if "state_parity" not in branches or int(branches.state_parity.sum()) != len(branches):
        failures.append("branch state parity incomplete")
    force_mismatch = []
    for r in branches.itertuples(index=False):
        key = (str(r.context_id), str(r.branch_label))
        if key in expected and abs(float(r.requested_force_N) - expected[key]) > 1e-9:
            force_mismatch.append(key)
    if force_mismatch:
        failures.append(f"force mismatch count={len(force_mismatch)}")
    repeats_expected = 2 if split == "TRAIN" else 5
    cells = branches.groupby([branches.context_id.astype(str), branches.requested_force_N.astype(float)]).size()
    if len(cells) != int(target["expected_force_cells"]) or not bool((cells == repeats_expected).all()):
        failures.append("force-cell/repeat structure mismatch")

    visual = visual[visual.context_id.astype(str).isin(expected_ids)].copy()
    if len(visual) != len(expected_ids) or visual.context_id.nunique() != len(expected_ids):
        failures.append("visual context identity/count mismatch")
    visual_failures = []
    for r in visual.to_dict("records"):
        cid = str(r["context_id"])
        if int(r.get("restore_exact", 0)) != 1:
            visual_failures.append(f"{cid}:restore_exact")
        for key in ["restored_state_hash", "second_restore_hash"]:
            if str(r.get(key, "")) != str(r.get("snapshot_state_hash", "")):
                visual_failures.append(f"{cid}:{key}")
        for pcol, hcol in [
            ("camera0_rgb_path", "camera0_rgb_sha256"),
            ("camera1_rgb_path", "camera1_rgb_sha256"),
            ("visual_feature_path", "visual_feature_sha256"),
        ]:
            p = Path(str(r[pcol]))
            if not p.exists():
                visual_failures.append(f"{cid}:{pcol}:missing"); continue
            arr = np.load(p, allow_pickle=False)
            ah = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
            if ah != str(r[hcol]):
                visual_failures.append(f"{cid}:{pcol}:hash")
        fp = Path(str(r["visual_feature_path"]))
        if fp.exists():
            z = np.load(fp, allow_pickle=False)
            if z.shape != (4096,) or z.dtype != np.float32:
                visual_failures.append(f"{cid}:feature-shape-dtype")
    if visual_failures:
        failures.append(f"visual failures={len(visual_failures)}")

    telemetry_failures = []
    telemetry_steps = 0
    for r in branches.to_dict("records"):
        p = Path(str(r.get("telemetry_path", "")))
        if not p.exists():
            telemetry_failures.append(f"{r.get('branch_id')}:missing"); continue
        d = pd.read_csv(p)
        telemetry_steps += len(d)
        if len(d) < 9 or not REQUIRED_TELEMETRY <= set(d.columns):
            telemetry_failures.append(f"{r.get('branch_id')}:columns-or-length"); continue
        finite = [x for x in REQUIRED_TELEMETRY if x not in {"phase"}]
        if not np.isfinite(d[finite].to_numpy(float)).all():
            telemetry_failures.append(f"{r.get('branch_id')}:nonfinite")
    if telemetry_failures:
        failures.append(f"telemetry failures={len(telemetry_failures)}")

    checks = {
        "exact_context_ids": actual_ids == expected_ids,
        "exact_branch_targets": set(actual_keys) == set(expected),
        "unique_branches": len(set(actual_keys)) == len(actual_keys),
        "state_parity_all": "state_parity" in branches and int(branches.state_parity.sum()) == len(branches),
        "force_cells_and_repeats_exact": len(cells) == int(target["expected_force_cells"]) and bool((cells == repeats_expected).all()),
        "visual_alignment_all": not visual_failures and len(visual) == len(expected_ids),
        "corrected_telemetry_all": not telemetry_failures,
        "no_outcome_driven_retry": len(actual_keys) == len(expected) and set(actual_keys) == set(expected),
        "task0_only": set(branches.task.astype(int)) == {0},
        "split_exact": set(branches.split.astype(str)) == {split},
    }
    result = {
        "status": "PASS" if not failures and all(checks.values()) else "FAIL",
        "split": split, "checks": checks, "failures": failures,
        "counts": {
            "contexts": len(contexts), "simulator_root_clusters": contexts.root_id.nunique(),
            "force_cells": len(cells), "branches": len(branches),
            "successes": int(branches.full_task_success_y.sum()),
            "failures": int(len(branches) - branches.full_task_success_y.sum()),
            "visual_contexts": len(visual), "raw_physical_timesteps": telemetry_steps,
        },
        "target_manifest": str(target_path), "target_manifest_sha256": sha256(target_path),
        "context_csv_sha256": sha256(context_path), "branch_csv_sha256": sha256(branch_path),
        "visual_csv_sha256": sha256(visual_path),
        "telemetry_tree_sha256": tree_sha256(out / "task0/task0/P5S0C_BRANCH_TELEMETRY"),
        "scientific_retry_count": 0 if checks["no_outcome_driven_retry"] else None,
        "visual_failures": visual_failures, "telemetry_failures": telemetry_failures,
    }
    audit_path = PROJECT / f"TASK0_{split}_COLLECTION_AUDIT.json"
    write_json(audit_path, result)
    if result["status"] != "PASS":
        raise RuntimeError(f"{split} collection audit failed: {failures[:3]}")
    if split == "TEST":
        commit = {
            "status": "ATOMICALLY_COMMITTED_FROZEN_TEST",
            "split_manifest": str(SPLIT_MANIFEST), "split_manifest_sha256": sha256(SPLIT_MANIFEST),
            "audit": str(audit_path), "audit_sha256": sha256(audit_path),
            "context_csv": str(context_path), "context_csv_sha256": sha256(context_path),
            "branch_csv": str(branch_path), "branch_csv_sha256": sha256(branch_path),
            "visual_csv": str(visual_path), "visual_csv_sha256": sha256(visual_path),
            "telemetry_tree_sha256": result["telemetry_tree_sha256"],
            "PCA_fit_forbidden": True, "model_selection_forbidden": True,
            "calibration_forbidden": True, "tuning_forbidden": True,
        }
        write_json(PROJECT / "TASK0_FROZEN_TEST_COMMIT.json", commit, atomic=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=["TRAIN", "TEST"], required=True)
    args = ap.parse_args(); audit(args.split)
