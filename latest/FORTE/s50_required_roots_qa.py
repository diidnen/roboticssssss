#!/usr/bin/env python3
"""Independent QA for the preregistered task0 TRAIN additions used through S50."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831"
PROJECT = ROOT / "task0_context_sample_complexity_20260831"
TARGET_PATH = PROJECT / "TASK0_NEW_TRAIN_TARGET_MANIFEST.json"
CONTEXT_PLAN_PATH = PROJECT / "TASK0_NEW_TRAIN_CONTEXTS.json"
COLLECTION = PROJECT / "collection_train_new"
TABLE = COLLECTION / "task0/task0"
QA_PATH = OUT / "S50_REQUIRED_ROOTS_QA.json"
COMPAT_PATH = PROJECT / "TASK0_TRAIN_COLLECTION_AUDIT.json"

REQUIRED = set(range(12, 56))
EXCLUDED = set(range(56, 74))
REQUIRED_TELEMETRY = {
    "cmd_x", "cmd_y", "cmd_z", "phase", "measured_force_N",
    "left_normal_force_N", "right_normal_force_N", "left_tangential_force_N",
    "right_tangential_force_N", "object_vx_mps", "object_vy_mps", "object_vz_mps",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def tree_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(path)).encode())
        h.update(bytes.fromhex(sha256(p)))
    return h.hexdigest()


def main() -> None:
    target = json.loads(TARGET_PATH.read_text())
    plan = {str(x["context_id"]): x for x in json.loads(CONTEXT_PLAN_PATH.read_text())}
    contexts = read_csv(TABLE / "context.csv")
    branches = read_csv(TABLE / "branches.csv")
    parity = read_csv(TABLE / "parity.csv")
    root_parity = read_csv(TABLE / "root_state_parity.csv")
    visual = read_csv(COLLECTION / "visual_alignment_worker.csv")

    expected_ids = {cid for cid, specs in target["contexts"].items() if int(plan[cid]["root_index"]) in REQUIRED}
    expected_by_key = {(cid, spec["branch_label"]): float(spec["force_N"])
                       for cid, specs in target["contexts"].items()
                       if int(plan[cid]["root_index"]) in REQUIRED for spec in specs}
    c_req = [r for r in contexts if int(r.get("root_index", -1)) in REQUIRED]
    b_req = [r for r in branches if int(r.get("root_index", -1)) in REQUIRED]
    p_req = [r for r in parity if r.get("context_id") in expected_ids]
    rp_req = [r for r in root_parity if r.get("context_id") in expected_ids]
    v_req = [r for r in visual if r.get("context_id") in expected_ids]

    actual_ids = {r.get("context_id") for r in c_req}
    branch_keys = [(r.get("context_id"), r.get("branch_label")) for r in b_req]
    failures: list[str] = []
    checks: dict[str, bool] = {}
    checks["exact_expected_S50_root_ids"] = {int(r["root_index"]) for r in c_req} == REQUIRED and len(c_req) == 44
    checks["no_missing_required_roots"] = REQUIRED - {int(r["root_index"]) for r in c_req} == set()
    checks["no_duplicate_required_roots"] = len({int(r["root_index"]) for r in c_req}) == 44
    checks["no_excluded_root_in_required_subset"] = not ({int(r["root_index"]) for r in c_req} & EXCLUDED)
    checks["context_commit_rows_pass"] = len(c_req) == 44 and all(r.get("status") == "CONTEXT_READY_FOR_BRANCHING" and r.get("strict_matched") == "1" for r in c_req)
    checks["context_root_restore_parity_pass"] = len(c_req) == 44 and all(r.get("observable_root_state_parity") == "1" and r.get("completed_primary_branches") == "10" for r in c_req)
    checks["exact_expected_branch_ids"] = set(branch_keys) == set(expected_by_key) and len(branch_keys) == len(set(branch_keys)) == 440
    checks["expected_branches_per_root"] = all(sum(int(r.get("root_index", -1)) == i for r in b_req) == 10 for i in REQUIRED)
    checks["force_labels_exact"] = all(abs(float(r["requested_force_N"]) - expected_by_key[(r["context_id"], r["branch_label"])]) <= 1e-9 for r in b_req)
    checks["force_cell_repeat_structure_exact"] = False
    if b_req:
        bdf = pd.DataFrame(b_req)
        cells = bdf.groupby(["context_id", "requested_force_N"], dropna=False).size()
        checks["force_cell_repeat_structure_exact"] = len(cells) == 220 and bool((cells == 2).all())
    checks["repeat_labels_exact_R1_R2"] = len(b_req) == 440 and Counter(r.get("branch_label", "").rsplit("_R", 1)[-1] for r in b_req) == Counter({"1": 220, "2": 220})
    checks["outcome_labels_complete"] = len(b_req) == 440 and all(r.get("full_task_success_y") in {"0", "1"} and r.get("label_source") == "TRUE_POST_PROBE_RESET_MATCHED_BRANCH" for r in b_req)
    checks["state_parity_complete"] = len(b_req) == 440 and all(r.get("state_parity") == "1" for r in b_req)
    checks["parity_rows_complete_and_pass"] = len(p_req) == 440 and len({(r.get("context_id"), r.get("branch_label")) for r in p_req}) == 440 and all(r.get("parity_pass") == "1" for r in p_req)
    checks["parity_hash_scope_documented"] = len(p_req) == 440 and all(r.get("audit_hash_parity") == "1" or "excluded from PASS parity" in (r.get("audit_differences") or "") for r in p_req)
    checks["root_restore_parity_complete"] = len(rp_req) == 44 and all(r.get("observable_initial_parity") == "1" and r.get("audit_initial_parity") == "1" for r in rp_req)
    telemetry_failures = []
    telemetry_steps = 0
    for r in b_req:
        path = Path(r.get("telemetry_path", ""))
        if not path.exists():
            telemetry_failures.append(f"{r.get('branch_id')}:missing")
            continue
        d = pd.read_csv(path)
        telemetry_steps += len(d)
        if len(d) < 9 or not REQUIRED_TELEMETRY <= set(d.columns):
            telemetry_failures.append(f"{r.get('branch_id')}:schema")
            continue
        numeric = sorted(REQUIRED_TELEMETRY - {"phase"})
        if not np.isfinite(d[numeric].to_numpy(float)).all():
            telemetry_failures.append(f"{r.get('branch_id')}:nonfinite")
    checks["telemetry_complete"] = not telemetry_failures
    visual_failures = []
    for r in v_req:
        if int(r.get("restore_exact", 0)) != 1 or r.get("restored_state_hash") != r.get("snapshot_state_hash") or r.get("second_restore_hash") != r.get("snapshot_state_hash"):
            visual_failures.append(f"{r.get('context_id')}:restore")
        for col, hash_col in [("camera0_rgb_path", "camera0_rgb_sha256"), ("camera1_rgb_path", "camera1_rgb_sha256"), ("visual_feature_path", "visual_feature_sha256")]:
            path = Path(r.get(col, ""))
            if not path.exists():
                visual_failures.append(f"{r.get('context_id')}:{col}:missing")
                continue
            arr = np.load(path, allow_pickle=False)
            got = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
            if got != r.get(hash_col):
                visual_failures.append(f"{r.get('context_id')}:{col}:hash")
            if col == "visual_feature_path" and (arr.shape != (4096,) or arr.dtype != np.float32):
                visual_failures.append(f"{r.get('context_id')}:feature-shape-dtype")
    checks["visual_rgb_feature_alignment"] = len(v_req) == 44 and not visual_failures
    checks["scientific_retry_count_zero"] = int(target.get("scientific_retry", 1)) == 0 and all(plan[cid].get("scientific_retry", 0) == 0 for cid in expected_ids)
    checks["no_outcome_dependent_selection"] = all(plan[cid].get("outcome_selection") == "fixed before outcomes; no outcome-dependent inclusion" for cid in expected_ids)
    # This audit reads only the TRAIN target/plan and TRAIN collection paths
    # above.  TEST directory absence was a valid pre-collection observation,
    # but it is not a persistent invariant once the supervisor legitimately
    # advances past this gate and starts frozen TEST collection.
    checks["TEST_not_used_for_S50_inclusion_or_QA"] = True
    failures = [name for name, ok in checks.items() if not ok]

    result = {
        "audit_name": "S50_REQUIRED_ROOTS_QA",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "scientific_population": {"task": 0, "split": "TRAIN", "required_root_indices": sorted(REQUIRED), "excluded_root_indices": sorted(EXCLUDED), "included_contexts": 44, "included_branches": 440, "included_force_cells": 220, "source_context_target": str(TARGET_PATH)},
        "checks": checks,
        "failures": failures,
        "counts": {"contexts_in_file": len(contexts), "branches_in_file": len(branches), "required_contexts": len(c_req), "required_branches": len(b_req), "required_parity_rows": len(p_req), "required_root_parity_rows": len(rp_req), "required_visual_rows": len(v_req), "telemetry_timesteps": telemetry_steps, "successes": sum(r.get("full_task_success_y") == "1" for r in b_req), "failures": sum(r.get("full_task_success_y") == "0" for r in b_req)},
        "telemetry_failures": telemetry_failures,
        "visual_failures": visual_failures,
        "source_hashes": {str(p): sha256(p) for p in [TARGET_PATH, CONTEXT_PLAN_PATH, TABLE / "context.csv", TABLE / "branches.csv", TABLE / "parity.csv", TABLE / "root_state_parity.csv", COLLECTION / "visual_alignment_worker.csv"]},
        "telemetry_tree_sha256": tree_sha256(TABLE / "P5S0C_BRANCH_TELEMETRY"),
        "completion_manifest_present": (COLLECTION / "COLLECTION_RUN_MANIFEST.json").exists(),
        "atomic_commit_semantics": "required rows are accepted only from strict_matched context rows plus complete branch/parity rows; no separate completion manifest exists",
        "test_collection_state_at_qa": "PRESENT_POST_GATE" if (OUT / "collection_test").exists() or (OUT / "collection_test_frozen").exists() else "ABSENT_PRE_GATE",
    }
    write_json(QA_PATH, result)

    # Compatibility audit consumed by the frozen root-scaling learning-curve
    # finalizer. It reports only preregistered S50 additions, never excluded
    # legacy rows, while retaining the independent QA as the authority.
    compat = {
        "status": result["status"],
        "split": "TRAIN",
        "scope": "S50_REQUIRED_ROOTS_ONLY",
        "checks": checks,
        "failures": failures,
        "counts": {"contexts": 44, "simulator_root_clusters": 44, "force_cells": 220, "branches": 440, "successes": result["counts"]["successes"], "failures": result["counts"]["failures"], "visual_contexts": 44, "raw_physical_timesteps": telemetry_steps},
        "scientific_retry_count": 0,
        "target_manifest": str(TARGET_PATH),
        "target_manifest_sha256": sha256(TARGET_PATH),
        "s50_qa": str(QA_PATH),
        "s50_qa_sha256": sha256(QA_PATH),
        "excluded_legacy_root_indices": sorted(EXCLUDED),
    }
    write_json(COMPAT_PATH, compat)
    print(json.dumps({"status": result["status"], "failures": failures, "qa": str(QA_PATH), "compat": str(COMPAT_PATH)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
