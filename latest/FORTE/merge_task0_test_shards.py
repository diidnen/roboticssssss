#!/usr/bin/env python3
"""Merge isolated task0 TEST shard outputs into the standard collection tree."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831"
COLLECTION = OUT / "collection_test"
SHARD_ROOT = OUT / "collection_test_shards" / "task0"
PLAN = SHARD_ROOT / "TASK0_TEST_SHARD_PLAN.json"
TARGET = OUT / "collection_plans/TEST_TARGET_MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def relocate(value: str, source_root: Path) -> str:
    p = Path(value)
    try:
        rel = p.relative_to(source_root)
    except ValueError:
        return value
    dst = COLLECTION / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not dst.exists():
        if p.is_file():
            shutil.copy2(p, dst)
    return str(dst)


def relocate_row(row: dict, source_root: Path) -> dict:
    out = dict(row)
    for key, value in list(out.items()):
        if key.endswith("_path") and value:
            out[key] = relocate(str(value), source_root)
    return out


def merge_table(name: str, sources: list[tuple[Path, Path]], key: str | None) -> tuple[list[dict], int]:
    rows: list[dict] = []
    seen: set[str] = set()
    for source_root, path in sources:
        for row in read_csv(path):
            row = relocate_row(row, source_root)
            k = str(row.get(key, "")) if key else json.dumps(row, sort_keys=True)
            if k in seen:
                raise RuntimeError(f"duplicate {name} key: {k}")
            seen.add(k); rows.append(row)
    return rows, len(rows)


def main() -> None:
    plan = json.loads(PLAN.read_text())
    run = json.loads((SHARD_ROOT / "TASK0_TEST_SHARD_RUN.json").read_text())
    if run.get("status") != "COMPLETE":
        raise RuntimeError(f"shard run is not complete: {run.get('status')}")
    if run.get("failed_shards"):
        raise RuntimeError(f"failed shards: {run['failed_shards']}")

    roots = [(COLLECTION, COLLECTION)]
    for name in sorted(plan["shards"]):
        roots.append((SHARD_ROOT / name, SHARD_ROOT / name))

    base = COLLECTION / "task0/task0"
    tables = {
        "context.csv": (base / "context.csv", "context_id"),
        "branches.csv": (base / "branches.csv", "branch_id"),
        "parity.csv": (base / "parity.csv", None),
        "root_state_parity.csv": (base / "root_state_parity.csv", "context_id"),
        "same_force_replay.csv": (base / "same_force_replay.csv", None),
    }
    merged_counts = {}
    for name, (dst, key) in tables.items():
        sources = [(root, root / "task0/task0" / name) for root, _ in roots]
        rows, count = merge_table(name, sources, key)
        if rows:
            write_csv(dst, rows)
            merged_counts[name] = count

    # A worker can leave a pre-probe visual capture for a context that was
    # interrupted before its atomic context commit.  Accept main-output visual
    # rows only for the main contexts recorded in the frozen shard plan; shard
    # rows are accepted only for that shard's fixed assignment.
    visual_rows: list[dict] = []
    seen_visual: set[str] = set()
    visual_sources = [(COLLECTION, set(plan["already_atomic_main_context_ids"]))]
    for name, ids in plan["shards"].items():
        visual_sources.append((SHARD_ROOT / name, set(ids)))
    for source_root, allowed in visual_sources:
        for row in read_csv(source_root / "visual_alignment_worker.csv"):
            if str(row.get("context_id", "")) not in allowed:
                continue
            row = relocate_row(row, source_root)
            cid = str(row.get("context_id", ""))
            if cid in seen_visual:
                raise RuntimeError(f"duplicate visual context: {cid}")
            seen_visual.add(cid); visual_rows.append(row)
    visual_count = len(visual_rows)
    write_csv(COLLECTION / "visual_alignment_worker.csv", visual_rows)
    merged_counts["visual_alignment_worker.csv"] = visual_count

    target_ids = set(json.loads(TARGET.read_text())["contexts"])
    task0_ids = {cid for cid in target_ids if "_t0_" in cid}
    context_rows = read_csv(base / "context.csv")
    branch_rows = read_csv(base / "branches.csv")
    parity_rows = read_csv(base / "parity.csv")
    visual_rows = read_csv(COLLECTION / "visual_alignment_worker.csv")
    actual_contexts = {str(r.get("context_id")) for r in context_rows}
    actual_branches = {str(r.get("branch_id")) for r in branch_rows}
    actual_parity = {(str(r.get("context_id")), str(r.get("branch_label"))) for r in parity_rows}
    if actual_contexts != task0_ids:
        raise RuntimeError(f"context identity mismatch: expected {len(task0_ids)}, got {len(actual_contexts)}")
    if len(branch_rows) != len(task0_ids) * 45 or len(actual_branches) != len(branch_rows):
        raise RuntimeError("branch count/uniqueness mismatch after shard merge")
    if len(parity_rows) != len(task0_ids) * 45 or len(actual_parity) != len(parity_rows):
        raise RuntimeError("parity count/uniqueness mismatch after shard merge")
    if {str(r.get("context_id")) for r in visual_rows} != task0_ids:
        raise RuntimeError("visual context identity mismatch after shard merge")

    run_record = {
        "status": "SHARDED_WORKERS_COMPLETE_REQUIRES_QA",
        "task": 0, "split": "TEST", "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_manifest": str(TARGET), "target_manifest_sha256": sha256(TARGET),
        "shard_plan": str(PLAN), "shard_plan_sha256": sha256(PLAN),
        "shard_run": str(SHARD_ROOT / "TASK0_TEST_SHARD_RUN.json"),
        "worker_count": 2, "num_envs": 1, "collector_source_unchanged": True,
        "physics_settings_changed": False, "pi0_changed": False,
        "scientific_retry_count": 0, "context_count": len(context_rows),
        "branch_count": len(branch_rows), "parity_count": len(parity_rows),
        "visual_context_count": len(visual_rows), "merged_counts": merged_counts,
    }
    (COLLECTION / "TASK0_TEST_RUN.json").write_text(json.dumps(run_record, indent=2, sort_keys=True) + "\n")
    merge_record = {
        "status": "MERGED_REQUIRES_EXISTING_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_output": str(COLLECTION / "task0/task0"),
        "source_shards": [str(SHARD_ROOT / x) for x in sorted(plan["shards"])],
        "main_preexisting_output": str(COLLECTION / "task0/task0"),
        "task0_contexts": len(context_rows), "task0_branches": len(branch_rows),
        "task0_parity_rows": len(parity_rows), "task0_visual_rows": len(visual_rows),
        "scientific_retry_count": 0, "outcome_dependent_shard_assignment": False,
    }
    (SHARD_ROOT / "TASK0_TEST_SHARD_MERGE.json").write_text(json.dumps(merge_record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(merge_record, indent=2))


if __name__ == "__main__":
    main()
