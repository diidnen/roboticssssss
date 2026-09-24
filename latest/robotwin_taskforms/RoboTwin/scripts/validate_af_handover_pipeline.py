#!/usr/bin/env python3
"""Independent completion validator for a single-task priority pipeline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


DEFAULT_TASK = "handover_mic"
FORCES = (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0)
FIXED = (3.0, 4.25, 5.0)
EXPECTED_SPLITS = {"train": 216, "validation": 54, "test": 54}
FULL_PARENT_SHA256 = "234082d3843a65ea32a82768ff0b9c80080d175beaffb93747991f2cb1681dc1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", choices=("handover_mic", "dump_bin_bigbin"), default=DEFAULT_TASK)
    args = parser.parse_args()
    exp = args.experiment
    task = args.task
    errors: list[str] = []
    checks: dict[str, object] = {}
    required = {
        "records": exp / "branches.jsonl",
        "parent": exp / "parent_108.snapshot.jsonl",
        "manifest": exp / "manifest.json",
        "root_map": exp / "root_map.json",
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
        report = {
            "schema_id": "AF_SINGLE_TASK_FORCEGRID324_INDEPENDENT_VALIDATION_V1",
            "task": task,
            "passed": False,
            "errors": errors,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 2

    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if manifest.get("scope") != "single_task_priority" or manifest.get("task") != task:
        errors.append("manifest scope/task mismatch")
    if manifest.get("source_parent_216_sha256") != FULL_PARENT_SHA256:
        errors.append("sealed full-parent hash mismatch")

    rows = jsonl(required["records"])
    parent = jsonl(required["parent"])
    if len(rows) != 324 or len(parent) != 108:
        errors.append(f"row count mismatch records={len(rows)} parent={len(parent)}")
    if rows[: len(parent)] != parent:
        errors.append("filtered sealed parent is not an exact logical prefix")
    if sha256(required["parent"]) != manifest.get("parent_108_sha256"):
        errors.append("filtered parent hash mismatch")
    if sha256(required["records"]) != manifest.get("records_sha256"):
        errors.append("development record hash mismatch")
    keys = [row["branch_key"] for row in rows]
    if len(keys) != len(set(keys)):
        errors.append("duplicate development branch keys")
    if {row.get("task") for row in rows} != {task}:
        errors.append("development task scope mismatch")
    if any(not row.get("valid") for row in rows):
        errors.append("invalid development query branch")
    groups: dict[tuple[int, float], dict[float, dict]] = defaultdict(dict)
    split_counts: dict[str, int] = defaultdict(int)
    split_roots: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        groups[(int(row["root_slot"]), float(row["friction"]))][
            float(row["force_n"])
        ] = row
        split_counts[row["split"]] += 1
        split_roots[row["split"]].add(int(row["root_slot"]))
    if len(groups) != 36:
        errors.append(f"expected 36 development contexts, found {len(groups)}")
    for key, force_map in groups.items():
        if set(force_map) != set(FORCES):
            errors.append(f"development force support mismatch: {key}")
    if dict(split_counts) != EXPECTED_SPLITS:
        errors.append(f"split row counts mismatch: {dict(split_counts)}")
    split_names = sorted(split_roots)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if split_roots[left] & split_roots[right]:
                errors.append(f"development root leakage: {left}/{right}")
    checks["development"] = {
        "rows": len(rows),
        "contexts": len(groups),
        "split_counts": dict(split_counts),
        "records_sha256": sha256(required["records"]),
    }

    audit = json.loads(required["audit"].read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("records_sha256") != sha256(required["records"]):
        errors.append("collection audit does not validate the development records")
    if audit.get("tasks") != [task]:
        errors.append("collection audit task scope mismatch")
    checks["audit"] = {
        "passed": audit.get("passed"),
        "force_authority": audit.get("requested_realized_authority", {}).get(task),
        "failure_slices": audit.get("failure_slices"),
    }

    training = json.loads(required["training"].read_text(encoding="utf-8"))
    if training.get("records_sha256") != sha256(required["records"]):
        errors.append("training report points to different records")
    if training.get("tasks") != [task]:
        errors.append("training task scope mismatch")
    if not training.get("root_disjoint") or training.get("test_used_for_model_selection"):
        errors.append("training split/model-selection contract failed")
    if training.get("feasibility", {}).get("counts") != EXPECTED_SPLITS:
        errors.append("training split counts mismatch")
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
    checks["training"] = {"model_hashes": model_hashes}

    inference = json.loads(required["inference"].read_text(encoding="utf-8"))
    threshold = inference.get("selected_threshold")
    if threshold is None or not (0.30 <= float(threshold) <= 0.80):
        errors.append("invalid validation-selected threshold")
    if inference.get("threshold_selection_split") != "validation":
        errors.append("threshold was not selected on validation")
    if inference.get("force_model_input") != "continuous scalar":
        errors.append("inference is not continuous-force conditioned")
    checks["inference"] = {"selected_threshold": threshold, "test": inference.get("test")}

    development_root_map = json.loads(required["root_map"].read_text(encoding="utf-8"))
    development_seeds = {
        int(value["actual_seed"])
        for key, value in development_root_map.get("roots", {}).items()
        if key.startswith(task + "|")
    }
    if len(development_root_map.get("roots", {})) != 12 or len(development_seeds) != 12:
        errors.append("development root map mismatch")
    fresh_roots = json.loads(required["fresh_roots"].read_text(encoding="utf-8"))
    fresh_entries = fresh_roots.get("roots", {})
    fresh_seeds = {int(value["actual_seed"]) for value in fresh_entries.values()}
    if len(fresh_entries) != 2 or len(fresh_seeds) != 2:
        errors.append("fresh root map must contain two distinct roots")
    if set(fresh_roots.get("tasks", [])) != {task}:
        errors.append("fresh root task scope mismatch")
    if development_seeds & fresh_seeds:
        errors.append("fresh-root overlap with development roots")
    if fresh_roots.get("downstream_outcomes_opened_during_selection") is not False:
        errors.append("fresh root qualification was not outcome blind")

    selections = jsonl(required["selection"])
    selection_ids = [row["context_id"] for row in selections]
    if len(selections) != 6 or len(selection_ids) != len(set(selection_ids)):
        errors.append("selection lock must contain six unique contexts")
    for row in selections:
        if row.get("task") != task:
            errors.append("selection lock task scope mismatch")
        force = float(row["selected_force_n"])
        if force < 3.0 or force > 5.0 or not close((force - 3.0) / 0.05, round((force - 3.0) / 0.05)):
            errors.append(f"selected force is off dense support: {force}")
        if "success" in row:
            errors.append("downstream outcome leaked into selection lock")
    lock_manifest = json.loads(required["selection_manifest"].read_text(encoding="utf-8"))
    if lock_manifest.get("contexts") != 6:
        errors.append("selection manifest context count mismatch")
    if lock_manifest.get("selection_uses_downstream_outcomes") is not False:
        errors.append("selection manifest does not certify outcome blindness")
    if lock_manifest.get("selection_lock_sha256") != sha256(required["selection"]):
        errors.append("selection-lock hash mismatch")
    if lock_manifest.get("training_report_sha256") != sha256(required["training"]):
        errors.append("selection lock used different training report")
    if lock_manifest.get("inference_report_sha256") != sha256(required["inference"]):
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
        if set(force_map) != set(FORCES) | {selected}:
            errors.append(f"fresh force support mismatch: {context_id}")
            continue
        grid = {force: bool(force_map[force]["success"]) for force in FORCES}
        successful = [force for force, success in grid.items() if success]
        recomputed.append(
            {
                "selected_success": bool(force_map[selected]["success"]),
                "fixed": {force: grid[force] for force in FIXED},
                "all_grid_fail": not successful,
                "force_selection_failure": not force_map[selected]["success"] and bool(successful),
                "force_sensitive": bool(successful) and len(successful) < len(FORCES),
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
    if len(recomputed) == 6:
        if summary.get("activeforcing", {}).get("successes") != sum(
            row["selected_success"] for row in recomputed
        ):
            errors.append("online ActiveForcing success summary mismatch")
        for force in FIXED:
            expected = sum(row["fixed"][force] for row in recomputed)
            if summary.get("fixed", {}).get(str(force), {}).get("successes") != expected:
                errors.append(f"online fixed-{force} success summary mismatch")
        decomposition = summary.get("failure_decomposition", {})
        if decomposition.get("all_grid_fail_contexts") != sum(row["all_grid_fail"] for row in recomputed):
            errors.append("all-grid-fail decomposition mismatch")
        if decomposition.get("force_selection_failures") != sum(
            row["force_selection_failure"] for row in recomputed
        ):
            errors.append("force-selection-failure decomposition mismatch")
        if decomposition.get("force_sensitive_grid_contexts") != sum(
            row["force_sensitive"] for row in recomputed
        ):
            errors.append("force-sensitive decomposition mismatch")
    checks["fresh_online"] = {
        "fresh_roots": len(fresh_seeds),
        "selection_contexts": len(selections),
        "online_branches": len(branches),
        "summary": summary,
    }

    report = {
        "schema_id": "AF_SINGLE_TASK_FORCEGRID324_INDEPENDENT_VALIDATION_V1",
        "task": task,
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
