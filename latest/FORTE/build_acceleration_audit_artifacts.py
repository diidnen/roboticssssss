#!/usr/bin/env python3
"""Build auditable, non-scientific acceleration artifacts for root scaling."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/home/exouser/FORTE/root_scaling_20260831")
TRAIN_PROJECT = Path("/home/exouser/FORTE/task0_context_sample_complexity_20260831")
TARGET = TRAIN_PROJECT / "TASK0_NEW_TRAIN_TARGET_MANIFEST.json"
CONTEXTS = TRAIN_PROJECT / "TASK0_NEW_TRAIN_CONTEXTS.json"
OUT = TRAIN_PROJECT / "collection_train_new"
TABLE = OUT / "task0/task0"
STAGE_LOG = OUT / "task0/P5S0C_STAGE_LOG.csv"
VISUAL = OUT / "visual_alignment_worker.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def root_index(value: str) -> int | None:
    m = re.search(r"(?:^|_)r(\d+)(?:_|$)", value or "")
    return int(m.group(1)) if m else None


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def load_target() -> tuple[dict, dict[str, list[dict]], list[dict]]:
    target = json.loads(TARGET.read_text())
    contexts = json.loads(CONTEXTS.read_text())
    context_plan = {str(x["context_id"]): x for x in contexts}
    return target, target["contexts"], list(context_plan.values())


def build_progress() -> dict:
    target, target_branches, context_plan = load_target()
    context_rows = read_csv(TABLE / "context.csv")
    branch_rows = read_csv(TABLE / "branches.csv")
    parity_rows = read_csv(TABLE / "parity.csv")
    root_parity_rows = read_csv(TABLE / "root_state_parity.csv")
    stage_rows = read_csv(STAGE_LOG)
    visual_rows = read_csv(VISUAL)

    original = list(range(12, 74))
    required = list(range(12, 56))
    completed_contexts = {
        int(r["root_index"]): r for r in context_rows
        if r.get("strict_matched") == "1" and r.get("status") == "CONTEXT_READY_FOR_BRANCHING"
    }
    completed_branch_roots = Counter()
    for row in branch_rows:
        idx = row.get("root_index")
        if idx is not None:
            completed_branch_roots[int(idx)] += 1
    parity_by_context = defaultdict(list)
    for row in parity_rows:
        parity_by_context[row.get("context_id", "")].append(row)
    root_parity_by_context = {r.get("context_id", ""): r for r in root_parity_rows}
    plan_by_index = {int(r["root_index"]): r for r in context_plan}

    required_checks = {}
    for idx in required:
        plan = plan_by_index.get(idx, {})
        cid = plan.get("context_id", "")
        c = completed_contexts.get(idx)
        b = [r for r in branch_rows if r.get("root_index") == str(idx)]
        p = parity_by_context.get(cid, [])
        rp = root_parity_by_context.get(cid, {})
        telemetry_ok = bool(b) and all(
            r.get("telemetry_path") and Path(r["telemetry_path"]).exists()
            and Path(r["telemetry_path"]).stat().st_size > 0 for r in b
        )
        required_checks[str(idx)] = {
            "context_id": cid,
            "context_commit_pass": bool(c and c.get("strict_matched") == "1" and c.get("status") == "CONTEXT_READY_FOR_BRANCHING"),
            "context_observable_root_state_parity": bool(c and c.get("observable_root_state_parity") == "1"),
            "context_expected_branch_count": int(c.get("planned_primary_branches", -1)) == 10 if c else False,
            "context_completed_branch_count": int(c.get("completed_primary_branches", -1)) == 10 if c else False,
            "branch_count": len(b),
            "branch_complete": len(b) == 10,
            "branch_state_parity_all_pass": bool(b) and all(r.get("state_parity") == "1" for r in b),
            "branch_label_set_exact": {r.get("branch_label") for r in b} == {x["branch_label"] for x in target_branches.get(cid, [])},
            "force_values_exact": len(b) == 10 and all(
                any(abs(float(r.get("requested_force_N", "nan")) - float(x["force_N"])) <= 1e-9
                    for x in target_branches.get(cid, [])) for r in b
            ),
            # The authoritative branch identity is the frozen R1/R2 suffix.
            # The runner's legacy repeat_index column is seed_idx+1 (2/3),
            # so it is retained as raw telemetry rather than reinterpreted.
            "repeat_set_exact": sorted(re.search(r"_R(\d+)$", r.get("branch_label", "")).group(1) for r in b if re.search(r"_R(\d+)$", r.get("branch_label", ""))) == ["1", "1", "1", "1", "1", "2", "2", "2", "2", "2"],
            "repeat_index_raw_values": sorted({r.get("repeat_index") for r in b}),
            "outcome_labels_present": bool(b) and all(r.get("full_task_success_y") in {"0", "1"} and r.get("label_source") == "TRUE_POST_PROBE_RESET_MATCHED_BRANCH" for r in b),
            "telemetry_complete": telemetry_ok,
            "parity_rows": len(p),
            "branch_parity_pass_all": len(p) == 10 and all(r.get("parity_pass") == "1" for r in p),
            "restore_hash_scope_documented": len(p) == 10 and all(
                r.get("parity_pass") == "1" and (r.get("audit_hash_parity") == "1" or "excluded from PASS parity" in (r.get("audit_differences") or "")) for r in p
            ),
            "root_restore_observable_parity": rp.get("observable_initial_parity") == "1",
            "root_restore_audit_parity": rp.get("audit_initial_parity") == "1",
            "scientific_retry_count": 0,
        }
        q = required_checks[str(idx)]
        q["required_root_qa_pass"] = all(v for k, v in q.items() if k not in {"context_id", "branch_count", "parity_rows", "scientific_retry_count"}) and q["scientific_retry_count"] == 0

    incomplete_stage = [r for r in stage_rows if r.get("context_id") and not r.get("stage", "").endswith(("COMPLETE", "VERIFIED"))]
    last_context_row = stage_rows[-1] if stage_rows else {}
    last_stage = str(last_context_row.get("stage", ""))
    current_idx = root_index(last_context_row.get("context_id", ""))
    # A committed CONTEXT_COMPLETE is an atomic boundary, not a live root.
    # Do not report the last completed excluded root as currently running.
    if last_stage == "CONTEXT_COMPLETE":
        current_idx = None
    completed = sorted(set(completed_contexts) & set(original))
    completed_required = sorted(set(completed) & set(required))
    completed_excluded = sorted(set(completed) - set(required))
    missing_required = sorted(set(required) - set(completed_required))
    remaining_excluded = sorted(set(original) - set(completed_excluded) - set(required))
    process_pids = {"303716": False, "339876": False}
    for pid in process_pids:
        process_pids[pid] = Path("/proc") .joinpath(pid).exists()
    checks = list(required_checks.values())
    all_required_pass = bool(checks) and all(x["required_root_qa_pass"] for x in checks)

    return {
        "audit_name": "CURRENT_COLLECTION_PROGRESS_AUDIT",
        "audit_time_utc": utc_now(),
        "authoritative_sources": {
            "target_manifest": str(TARGET),
            "target_manifest_sha256": sha256(TARGET),
            "context_plan": str(CONTEXTS),
            "context_plan_sha256": sha256(CONTEXTS),
            "context_csv": str(TABLE / "context.csv"),
            "branch_csv": str(TABLE / "branches.csv"),
            "parity_csv": str(TABLE / "parity.csv"),
            "root_state_parity_csv": str(TABLE / "root_state_parity.csv"),
            "stage_log": str(STAGE_LOG),
            "visual_alignment_manifest": str(VISUAL),
            "completion_manifest": str(OUT / "COLLECTION_RUN_MANIFEST.json"),
            "completion_manifest_present": (OUT / "COLLECTION_RUN_MANIFEST.json").exists(),
            "atomic_commit_records": "context.csv/branches.csv/parity.csv/root_state_parity.csv rows; no separate commit-record file present",
        },
        "original_target_root_ids": [f"root{i}" for i in original],
        "preregistered_S50_root_ids": [f"root{i}" for i in required],
        "completed_root_ids": [f"root{i}" for i in completed],
        "currently_running_root_id": f"root{current_idx}" if current_idx is not None else None,
        "currently_running_root_state": "last_stage_log_context_has_unfinished_stage_but_no_live_collector_pid" if current_idx is not None else "atomic_boundary_or_no_live_collector",
        "collector_processes": process_pids,
        "completed_required_S50_roots": [f"root{i}" for i in completed_required],
        "missing_required_S50_roots": [f"root{i}" for i in missing_required],
        "completed_excluded_roots": [f"root{i}" for i in completed_excluded],
        "remaining_excluded_roots": [f"root{i}" for i in remaining_excluded],
        "collected_total": len(completed),
        "included_in_S50": len(completed_required),
        "excluded_legacy": len(completed_excluded),
        "partial_or_uncommitted_excluded_contexts": [f"root{current_idx}"] if current_idx in set(remaining_excluded) else [],
        "required_root_checks": required_checks,
        "all_preregistered_S50_roots_context_complete": len(missing_required) == 0,
        "all_required_branches_complete": all(x["branch_complete"] for x in checks),
        "all_required_parity_checks_pass": all_required_pass,
        "all_required_context_commits_pass": all(x["context_commit_pass"] for x in checks),
        "S50_authoritative_ready_to_stop_legacy_collector": all_required_pass and not any(process_pids.values()),
        "raw_counts": {
            "context_rows": len(context_rows),
            "branch_rows": len(branch_rows),
            "parity_rows": len(parity_rows),
            "root_state_parity_rows": len(root_parity_rows),
            "visual_alignment_rows": len(visual_rows),
            "stage_rows": len(stage_rows),
            "unfinished_stage_rows": len(incomplete_stage),
        },
    }


def main() -> None:
    progress = build_progress()
    write_json(BASE / "CURRENT_COLLECTION_PROGRESS_AUDIT.json", progress)
    print(json.dumps({k: progress[k] for k in [
        "completed_required_S50_roots", "missing_required_S50_roots",
        "completed_excluded_roots", "remaining_excluded_roots",
        "all_required_parity_checks_pass", "S50_authoritative_ready_to_stop_legacy_collector",
    ]}, indent=2))


if __name__ == "__main__":
    main()
