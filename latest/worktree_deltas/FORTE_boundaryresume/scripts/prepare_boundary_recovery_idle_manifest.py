#!/usr/bin/env python3
"""Emit a CPU-only Boundary recovery manifest that remains fail-closed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SOURCE_ROOT = Path("/home/exouser/FORTE/BOUNDARY_AWARE_FULL_TASK_FEASIBILITY_DATA_CLOSURE_20260902")
DEFAULT_MANIFEST_NAME = "BOUNDARY_FORCE_INTERFACE_PREFLIGHT_ACQUISITION_MANIFEST.json"
REQUIRED_FILES = (
    "AUTHORITATIVE_720_BOUNDARY_LABELS.csv",
    "BOUNDARY_AUGMENTED_DATASET_MANIFEST.json",
    "BOUNDARY_ACQUISITION_PROTOCOL.json",
    "MINIMAL_FORCE_INTERFACE_PREFLIGHT_PROTOCOL.json",
    "BOUNDARY_NEXT_QUERY_STATE.csv",
    "RESOURCE_GATE_STATUS.json",
    "PHASE1_INDEPENDENT_VALIDATION.json",
    "PHASE1_MANIFEST_QA.json",
    "EXISTING_BOUNDARY_AUDIT_QA.json",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_next_query(path: Path) -> dict:
    rows = load_csv_rows(path)
    contexts = {row["context_id"] for row in rows}
    tasks = sorted({int(row["task"]) for row in rows})
    launchable = sum(int(row["candidate_is_launchable"]) for row in rows)
    utility_used = sum(int(row["utility_used"]) for row in rows)
    return {
        "rows": len(rows),
        "contexts": len(contexts),
        "tasks": tasks,
        "candidate_launchable_rows": launchable,
        "utility_used_rows": utility_used,
    }


def source_inventory() -> tuple[dict[str, dict[str, object]], list[str]]:
    inventory: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for name in REQUIRED_FILES:
        path = SOURCE_ROOT / name
        exists = path.exists()
        record: dict[str, object] = {"path": str(path), "exists": exists}
        if exists:
            record["sha256"] = sha256(path)
        else:
            missing.append(name)
        inventory[name] = record
    return inventory, missing


def build_manifest() -> dict:
    files, missing = source_inventory()
    manifest: dict[str, object] = {
        "manifest_type": "BOUNDARY_FORCE_INTERFACE_PREFLIGHT_ACQUISITION_MANIFEST",
        "mode": "CPU_ONLY_REPLACEMENT",
        "created_utc": now_utc(),
        "source_root": str(SOURCE_ROOT),
        "read_only_audit": True,
        "remain_idle": True,
        "launch_authorization": "DENIED_UNTIL_EXTERNAL_PREREQUISITES_PASS",
        "mutation_guard": {
            "old_data_overwritten": False,
            "new_data_created": False,
            "gpu_or_isaac_launched": False,
            "training_launched": False,
            "mass_touched": False,
            "test_touched": False,
            "utility_touched": False,
        },
        "source_files": files,
    }

    if missing:
        manifest["status"] = "IDLE_PREREQUISITE_UNAVAILABLE"
        manifest["blocked_by"] = ["SOURCE_ARTIFACTS_MISSING"]
        manifest["missing_source_files"] = missing
        return manifest

    labels_path = SOURCE_ROOT / "AUTHORITATIVE_720_BOUNDARY_LABELS.csv"
    augmented_path = SOURCE_ROOT / "BOUNDARY_AUGMENTED_DATASET_MANIFEST.json"
    protocol_path = SOURCE_ROOT / "BOUNDARY_ACQUISITION_PROTOCOL.json"
    preflight_path = SOURCE_ROOT / "MINIMAL_FORCE_INTERFACE_PREFLIGHT_PROTOCOL.json"
    next_query_path = SOURCE_ROOT / "BOUNDARY_NEXT_QUERY_STATE.csv"
    resource_path = SOURCE_ROOT / "RESOURCE_GATE_STATUS.json"
    validation_path = SOURCE_ROOT / "PHASE1_INDEPENDENT_VALIDATION.json"
    qa_path = SOURCE_ROOT / "PHASE1_MANIFEST_QA.json"
    audit_path = SOURCE_ROOT / "EXISTING_BOUNDARY_AUDIT_QA.json"
    force_gate_pass_path = SOURCE_ROOT / "MINIMAL_FORCE_INTERFACE_CALIBRATION_PASS.json"

    labels = load_csv_rows(labels_path)
    augmented = load_json(augmented_path)
    protocol = load_json(protocol_path)
    preflight = load_json(preflight_path)
    next_query = summarize_next_query(next_query_path)
    resource = load_json(resource_path)
    validation = load_json(validation_path)
    manifest_qa = load_json(qa_path)
    audit_qa = load_json(audit_path)

    old_rows = len(labels)
    old_branches = int(augmented.get("old_branches", -1))
    new_branches = int(augmented.get("new_boundary_branches", -1))
    preflight_pass_present = force_gate_pass_path.exists()

    blocked_by: list[str] = []
    if old_rows != 720 or old_branches != 720:
        blocked_by.append("OLD_BRANCH_COUNT_MISMATCH")
    if new_branches != 0:
        blocked_by.append("UNEXPECTED_NEW_BRANCHES_PRESENT")
    if not preflight_pass_present:
        blocked_by.append("MINIMAL_FORCE_INTERFACE_CALIBRATION_PASS_MISSING")
    if resource.get("decision") == "NO_SIMULATOR_LAUNCH":
        blocked_by.append("RESOURCE_GATE_NO_SIMULATOR_LAUNCH")
    if next_query["candidate_launchable_rows"] != 0:
        blocked_by.append("CANDIDATE_ROWS_IMPROPERLY_LAUNCHABLE")
    if validation.get("status") != "PASS":
        blocked_by.append("PHASE1_INDEPENDENT_VALIDATION_NOT_PASS")
    if manifest_qa.get("status") != "PASS_CPU_PHASE1":
        blocked_by.append("PHASE1_MANIFEST_QA_NOT_PASS")
    if audit_qa.get("status") not in {"PASS", "PASS_WITH_TASK1_LABEL_CAVEAT"}:
        blocked_by.append("EXISTING_BOUNDARY_AUDIT_QA_NOT_PASSING")

    manifest.update(
        {
            "status": "IDLE_PREREQUISITE_UNAVAILABLE" if blocked_by else "READY_FOR_EXTERNAL_PREFLIGHT_ONLY",
            "blocked_by": blocked_by,
            "branch_accounting": {
                "confirmed": old_rows == 720 and old_branches == 720 and new_branches == 0,
                "old_branches_csv_rows": old_rows,
                "old_branches_manifest": old_branches,
                "new_branches_manifest": new_branches,
                "expected_statement": "720 old + 0 new",
                "authoritative_labels_csv": str(labels_path),
                "authoritative_labels_sha256": sha256(labels_path),
            },
            "phase1_scope": {
                "tasks": protocol.get("scope", {}).get("tasks", []),
                "qualification_contexts": next_query["contexts"],
                "qualification_rows": next_query["rows"],
                "root_indices_each_task": protocol.get("scope", {}).get("root_indices_each_task", []),
                "friction_bands": protocol.get("scope", {}).get("friction_bands", []),
            },
            "force_interface_preflight": {
                "required_artifact": str(force_gate_pass_path),
                "required_artifact_present": preflight_pass_present,
                "protocol_status": preflight.get("status"),
                "scientific_data": preflight.get("scientific_data"),
                "task_or_outcome_selection": preflight.get("task_or_outcome_selection"),
                "resource_requirement": preflight.get("resource_requirement"),
                "pass_criteria": preflight.get("pass_criteria", {}),
            },
            "acquisition_state": {
                "status": augmented.get("status"),
                "old_archive_overwritten": augmented.get("old_archive_overwritten"),
                "telemetry_complete": augmented.get("telemetry_complete"),
                "state_hash_parity_complete": augmented.get("state_hash_parity_complete"),
                "training_allowed": augmented.get("training_allowed"),
                "test_claim_allowed": augmented.get("test_claim_allowed"),
                "candidate_launchable_rows": next_query["candidate_launchable_rows"],
                "utility_used_rows": next_query["utility_used_rows"],
            },
            "resource_gate": {
                "checked_utc": resource.get("checked_utc"),
                "status": resource.get("status"),
                "decision": resource.get("decision"),
                "reason": resource.get("reason"),
                "processes_killed_or_modified": resource.get("processes_killed_or_modified"),
            },
            "validation": {
                "existing_boundary_audit_status": audit_qa.get("status"),
                "phase1_independent_validation_status": validation.get("status"),
                "phase1_manifest_qa_status": manifest_qa.get("status"),
            },
        }
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Unique output directory inside this worktree")
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / args.manifest_name
    manifest = build_manifest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
