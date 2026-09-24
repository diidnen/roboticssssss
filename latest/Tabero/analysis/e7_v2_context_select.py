#!/usr/bin/env python3
"""Freeze a deterministic E7_V2 context set from the E1/E2 QA PASS pool."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


POOL_ROOT = Path(
    "/media/volume/newdata/exouser/ACTIVEFORCING_DISK_ARCHIVE_20260902/Tabero/analysis/results/"
    "gnp_style_visual_context_prospective_20260831_011000/collection_train"
)
OUT_CSV = Path("/home/exouser/Tabero/E7_V2_CONTEXT_MANIFEST.csv")
OUT_RULE = Path("/home/exouser/Tabero/E7_V2_CONTEXT_SELECTION_RULE.json")
P4 = Path("/home/exouser/Tabero/analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py")

# Frozen before any V2 snapshot or planner result is read.  The keys cover all
# four E7 tasks, both roots and all three physics bands without using outcomes.
SELECTION_KEYS = [
    (0, 0, "LOW"), (0, 1, "MID"), (0, 2, "HIGH"),
    # The E1/E2 QA PASS pool contains task1 roots 4/5 only; use those
    # eligible roots rather than importing a context from another pool.
    (1, 4, "MID"), (1, 5, "HIGH"),
    (5, 0, "MID"), (5, 1, "HIGH"),
    (6, 0, "LOW"), (6, 1, "MID"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_archived(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    old = "/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000/collection_train"
    if path_text.startswith(old):
        return POOL_ROOT / path_text[len(old):].lstrip("/")
    raise FileNotFoundError(path)


def canonical_archive_path(path_text: str) -> Path:
    """Return an inspectable path for archived provenance, not a stale alias."""
    return resolve_archived(path_text)


def main() -> None:
    rows = []
    for task in (0, 1, 5, 6):
        path = POOL_ROOT / f"task{task}" / f"task{task}" / "context.csv"
        rows.extend(dict(r) for r in csv.DictReader(path.open(newline="")))

    eligible = [
        r for r in rows
        if int(float(r.get("probe_qualified", 0))) == 1
        and int(float(r.get("strict_matched", 0))) == 1
        and str(r.get("split", "")) == "TRAIN"
        and r.get("probe_telemetry_path", "")
        and r.get("post_probe_state_hash", "")
    ]
    by_key = {(int(r["task"]), int(r["root_index"]), str(r["friction_band"])): r for r in eligible}
    selected = []
    for task, root_index, band in SELECTION_KEYS:
        if (task, root_index, band) not in by_key:
            raise RuntimeError(f"missing eligible E1/E2 context {(task, root_index, band)}")
        r = by_key[(task, root_index, band)]
        # The archived prospective table uses the visual-context ``pv_``
        # namespace; the authoritative P5 runtime uses the equivalent frozen
        # ``p5s0c_`` namespace.  Normalize identifiers only, not state/data.
        context_id = r["context_id"].replace("pv_train_", "p5s0c_train_")
        root_id = r["root_id"].replace("pv_train_", "p5s0c_train_")
        selected.append({
            "context_id": context_id,
            "task": int(r["task"]),
            "task_instruction": r["task_instruction"],
            "root_id": root_id,
            "root_index": int(r["root_index"]),
            "root_seed": int(r["root_seed"]),
            "friction_band": r["friction_band"],
            "mu_GT": float(r["hidden_friction_analysis_only"]),
            "split": r["split"],
            "source_context_csv": str(POOL_ROOT / f"task{task}" / f"task{task}" / "context.csv"),
            "source_probe_telemetry": str(canonical_archive_path(r["probe_telemetry_path"])),
            "source_probe_telemetry_sha256": sha256(resolve_archived(r["probe_telemetry_path"])),
            "source_query_valid": 1,
            "source_strict_matched": 1,
            "selection_basis": "task_root_physics_query_valid_provenance_only",
        })

    rule = {
        "manifest_name": "E7_V2_FROZEN_CONTEXT_SELECTION",
        "selection_frozen_before_v2_snapshot_and_planner_outcomes": True,
        "eligible_pool": str(POOL_ROOT),
        "eligible_pool_context_count": len(eligible),
        "eligible_filter": ["split=TRAIN", "probe_qualified=1", "strict_matched=1", "physical_history_path_nonempty", "post_probe_state_hash_nonempty"],
        "selection_keys_task_root_index_band": [list(x) for x in SELECTION_KEYS],
        "excluded_fields": ["selected_force", "planner", "success", "failure_stage", "utility", "grid_continuous_posterior_performance"],
        "p4_probe_source": str(P4),
        "p4_probe_sha256": sha256(P4),
        "selected_context_ids": [r["context_id"] for r in selected],
        "selected_context_count": len(selected),
    }
    rule_bytes = (json.dumps(rule, indent=2, sort_keys=True) + "\n").encode()
    rule["selection_rule_sha256"] = hashlib.sha256(rule_bytes).hexdigest()
    OUT_RULE.write_text(json.dumps(rule, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = list(selected[0])
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps({"selected": len(selected), "rule_sha256": rule["selection_rule_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
