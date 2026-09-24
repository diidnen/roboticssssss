#!/usr/bin/env python3
"""Independent completion validator for the continuous-force pipeline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_PARENT_SHA256 = "234082d3843a65ea32a82768ff0b9c80080d175beaffb93747991f2cb1681dc1"
FORCES = (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0)
FIXED = (3.0, 4.25, 5.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def close(left, right, tolerance=1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    exp = args.experiment
    errors: list[str] = []
    checks: dict[str, object] = {}

    required = {
        "records": exp / "branches.jsonl",
        "parent": exp / "parent_216.snapshot.jsonl",
        "manifest": exp / "manifest.json",
        "audit": exp / "collection_audit.json",
        "training": exp / "models/training_report.json",
        "inference": exp / "dense_inference_report.json",
        "fresh_roots": exp / "fresh_root_map.json",
        "selection": exp / "fresh_selection_lock.jsonl",
        "selection_manifest": exp / "fresh_selection_lock_manifest.json",
        "online_branches": exp / "fresh_online_branches.jsonl",
        "online_report": exp / "fresh_online_report.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        errors.append(f"missing required artifacts: {missing}")
        report = {"schema_id": "AF_FORCEGRID648_INDEPENDENT_VALIDATION_V1", "passed": False, "errors": errors}
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2

    rows = jsonl(required["records"])
    parent_rows = jsonl(required["parent"])
    parent_hash = sha256(required["parent"])
    checks["parent_snapshot_sha256"] = parent_hash
    if parent_hash != EXPECTED_PARENT_SHA256:
        errors.append("parent snapshot hash mismatch")
    if len(parent_rows) != 216 or rows[:216] != parent_rows:
        errors.append("sealed parent rows were not preserved exactly")
    if len(rows) != 648:
        errors.append(f"expected 648 rows, found {len(rows)}")
    row_keys = [row["branch_key"] for row in rows]
    if len(row_keys) != len(set(row_keys)):
        errors.append("duplicate development branch keys")
    development_groups: dict[tuple[str, int, float], dict[float, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["task"], int(row["root_slot"]), float(row["friction"]))
        development_groups[key][float(row["force_n"])] = row
    if len(development_groups) != 72:
        errors.append(f"expected 72 development contexts, found {len(development_groups)}")
    for key, force_map in development_groups.items():
        if set(force_map) != set(FORCES):
            errors.append(f"development force support mismatch: {key}")
    split_roots: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        split_roots[row["split"]].add((row["task"], int(row["root_slot"])))
    split_names = sorted(split_roots)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            if split_roots[left] & split_roots[right]:
                errors.append(f"development root leakage: {left}/{right}")
    checks["development"] = {
        "rows": len(rows),
        "contexts": len(development_groups),
        "records_sha256": sha256(required["records"]),
        "split_root_counts": {key: len(value) for key, value in split_roots.items()},
    }

    audit = json.loads(required["audit"].read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("records_sha256") != sha256(required["records"]):
        errors.append("collection audit does not validate the final records")
    checks["collection_audit"] = {
        "passed": audit.get("passed"),
        "integrity_passed": audit.get("integrity_passed"),
        "continuous_force_authority_passed": audit.get("continuous_force_authority_passed"),
        "failure_slices": audit.get("failure_slices"),
    }

    training = json.loads(required["training"].read_text(encoding="utf-8"))
    if training.get("records_sha256") != sha256(required["records"]):
        errors.append("training report points to different records")
    if not training.get("root_disjoint") or training.get("test_used_for_model_selection"):
        errors.append("training split/model-selection contract failed")
    expected_counts = {"train": 432, "validation": 108, "test": 108}
    if training.get("feasibility", {}).get("counts") != expected_counts:
        errors.append("feasibility split counts mismatch")
    model_hashes = []
    for family in ("belief", "feasibility"):
        members = training.get(family, {}).get("members", [])
        if len(members) != 3:
            errors.append(f"expected three {family} members")
        for member in members:
            checkpoint = Path(member["checkpoint"])
            actual = sha256(checkpoint) if checkpoint.exists() else None
            if actual != member.get("sha256"):
                errors.append(f"{family} checkpoint hash mismatch")
            model_hashes.append(actual)
    checks["training"] = {
        "root_disjoint": training.get("root_disjoint"),
        "test_used_for_model_selection": training.get("test_used_for_model_selection"),
        "model_hashes": model_hashes,
    }

    inference = json.loads(required["inference"].read_text(encoding="utf-8"))
    threshold = inference.get("selected_threshold")
    if threshold is None or not (0.30 <= float(threshold) <= 0.80):
        errors.append("invalid validation-selected threshold")
    if inference.get("threshold_selection_split") != "validation":
        errors.append("threshold was not selected on validation")
    if inference.get("force_model_input") != "continuous scalar":
        errors.append("inference is not continuous-force conditioned")
    checks["dense_inference"] = {
        "selected_threshold": threshold,
        "test": inference.get("test"),
    }

    parent_root_map = json.loads((exp / "root_map.json").read_text(encoding="utf-8"))
    parent_seeds = {int(value["actual_seed"]) for value in parent_root_map["roots"].values()}
    fresh_roots = json.loads(required["fresh_roots"].read_text(encoding="utf-8"))
    fresh_seeds = {int(value["actual_seed"]) for value in fresh_roots.get("roots", {}).values()}
    if len(fresh_roots.get("roots", {})) != 4 or len(fresh_seeds) != 4:
        errors.append("fresh root map does not contain four distinct roots")
    if parent_seeds & fresh_seeds:
        errors.append("fresh-root overlap with development roots")
    if fresh_roots.get("downstream_outcomes_opened_during_selection") is not False:
        errors.append("fresh root qualification was not outcome blind")

    selections = jsonl(required["selection"])
    selection_ids = [row["context_id"] for row in selections]
    if len(selections) != 12 or len(selection_ids) != len(set(selection_ids)):
        errors.append("selection lock must contain 12 unique contexts")
    for row in selections:
        force = float(row["selected_force_n"])
        if force < 3.0 or force > 5.0 or not close((force - 3.0) / 0.05, round((force - 3.0) / 0.05), 1e-7):
            errors.append(f"selected force is off dense support: {force}")
        if "success" in row:
            errors.append("downstream outcome leaked into selection lock")
    selection_manifest = json.loads(required["selection_manifest"].read_text(encoding="utf-8"))
    if selection_manifest.get("selection_uses_downstream_outcomes") is not False:
        errors.append("selection manifest does not certify outcome blindness")
    if selection_manifest.get("selection_lock_sha256") != sha256(required["selection"]):
        errors.append("selection-lock hash mismatch")
    if selection_manifest.get("training_report_sha256") != sha256(required["training"]):
        errors.append("selection lock used different training report")
    if selection_manifest.get("inference_report_sha256") != sha256(required["inference"]):
        errors.append("selection lock used different inference report")

    branches = jsonl(required["online_branches"])
    branch_keys = [(row["context_id"], float(row["force_n"])) for row in branches]
    if len(branch_keys) != len(set(branch_keys)):
        errors.append("duplicate fresh online branches")
    branch_groups: dict[str, dict[float, dict]] = defaultdict(dict)
    for row in branches:
        branch_groups[row["context_id"]][float(row["force_n"])] = row
    selection_map = {row["context_id"]: row for row in selections}
    if set(branch_groups) != set(selection_map):
        errors.append("fresh branch/selection context mismatch")
    recomputed = []
    for context_id, selection in selection_map.items():
        force_map = branch_groups.get(context_id, {})
        selected = float(selection["selected_force_n"])
        required_forces = set(FORCES) | {selected}
        if set(force_map) != required_forces:
            errors.append(f"fresh force support mismatch: {context_id}")
            continue
        grid_success = {force: bool(force_map[force]["success"]) for force in FORCES}
        successful = [force for force, success in grid_success.items() if success]
        recomputed.append(
            {
                "context_id": context_id,
                "task": selection["task"],
                "selected_force_n": selected,
                "selected_success": bool(force_map[selected]["success"]),
                "fixed": {force: grid_success[force] for force in FIXED},
                "all_grid_fail": not successful,
                "force_selection_failure": not force_map[selected]["success"] and bool(successful),
                "force_sensitive": bool(successful) and len(successful) < len(FORCES),
                "oracle_min": min(successful) if successful else None,
            }
        )
    online = json.loads(required["online_report"].read_text(encoding="utf-8"))
    if online.get("selection_uses_downstream_outcomes") is not False:
        errors.append("online report violates outcome-blind selection")
    if not online.get("all_selections_locked_before_any_outcome"):
        errors.append("online outcomes began before the full selection lock")
    if online.get("online_branches_sha256") != sha256(required["online_branches"]):
        errors.append("online report branch hash mismatch")
    summary = online.get("summary", {})
    if len(recomputed) == 12:
        af_successes = sum(row["selected_success"] for row in recomputed)
        if summary.get("activeforcing", {}).get("successes") != af_successes:
            errors.append("online AF success summary mismatch")
        for force in FIXED:
            value = sum(row["fixed"][force] for row in recomputed)
            if summary.get("fixed", {}).get(str(force), {}).get("successes") != value:
                errors.append(f"online fixed-{force} success summary mismatch")
        all_fail = sum(row["all_grid_fail"] for row in recomputed)
        selection_fail = sum(row["force_selection_failure"] for row in recomputed)
        force_sensitive = sum(row["force_sensitive"] for row in recomputed)
        decomposition = summary.get("failure_decomposition", {})
        if decomposition.get("all_grid_fail_contexts") != all_fail:
            errors.append("all-grid-fail decomposition mismatch")
        if decomposition.get("force_selection_failures") != selection_fail:
            errors.append("force-selection-failure decomposition mismatch")
        if decomposition.get("force_sensitive_grid_contexts") != force_sensitive:
            errors.append("force-sensitive decomposition mismatch")
    checks["fresh_online"] = {
        "fresh_roots": len(fresh_seeds),
        "selection_contexts": len(selections),
        "online_branches": len(branches),
        "recomputed_contexts": len(recomputed),
        "summary": summary,
    }

    report = {
        "schema_id": "AF_FORCEGRID648_INDEPENDENT_VALIDATION_V1",
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "artifact_hashes": {name: sha256(path) for name, path in required.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
