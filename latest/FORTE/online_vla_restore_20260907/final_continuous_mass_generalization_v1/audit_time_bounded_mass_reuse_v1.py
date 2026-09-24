#!/usr/bin/env python3
"""Freeze reuse classifications after pausing the original 648 queue."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from time_bounded_mass_common import HERE, load_frozen_subset, read, sha256, spec_lookup, validate_branch


AUX_FORCES = {3.25, 3.5, 3.75, 4.25, 4.5, 4.75}


def write_new_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    csv_path = HERE / "TIME_BOUNDED_MASS_REUSE_AUDIT.csv"
    summary_path = HERE / "TIME_BOUNDED_MASS_REUSE_SUMMARY.json"
    if csv_path.exists() or summary_path.exists():
        raise FileExistsError("reuse audit is immutable and already exists")
    subset, subset_hash = load_frozen_subset()
    runtime = read(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json")
    primary = spec_lookup(subset)
    plan = read(HERE / "MASS_TRAINING_CONTEXT_PLAN.json")
    contexts = {row["id"]: row for row in plan["contexts"]}
    rows = []
    valid_total = reused = auxiliary = invalid = incomplete = 0

    for context_dir in sorted((HERE / "branches").glob("*")):
        if not context_dir.is_dir():
            continue
        context = contexts.get(context_dir.name)
        for job_dir in sorted(context_dir.glob("F*"), key=lambda p: float(p.name[1:])):
            try:
                force = float(job_dir.name[1:])
            except ValueError:
                continue
            primary_spec = primary.get((context_dir.name, force))
            spec = primary_spec or ({
                "context_id": context_dir.name, "force_N": force,
                "mass_kg": context["mass_kg"], "split": context["split"],
                "root": context["root"], "task": context["task"],
            } if context and force in AUX_FORCES else None)
            if spec is None:
                rows.append({"path": str(job_dir), "context_id": context_dir.name, "force_N": force,
                    "classification": "OUTSIDE_FROZEN_SUPERSET", "valid": False,
                    "failed_checks": "not_in_original_factorial", "branch_bundle_sha256": ""})
                invalid += 1
                continue
            check = validate_branch(job_dir, spec, runtime, HERE / "references" / context_dir.name)
            if check["valid"]:
                valid_total += 1
                if primary_spec:
                    classification = "VALID_PRIMARY_REUSE"
                    reused += 1
                else:
                    classification = "VALID_AUXILIARY_PRESERVED"
                    auxiliary += 1
            elif (job_dir / "LAUNCH_CLAIM.json").exists() and not (job_dir / "WORKER_COMPLETION.json").exists():
                classification = "INCOMPLETE_ACTIVE_OR_INFRASTRUCTURE"
                incomplete += 1
            else:
                classification = "INVALID_PARITY"
                invalid += 1
            bundle = ""
            if check.get("artifact_sha256"):
                bundle = sha256(job_dir / "WORKER_COMPLETION.json") + ":" + sha256(job_dir / "BRANCH_RESULT.json")
            rows.append({"path": str(job_dir), "context_id": context_dir.name, "force_N": force,
                "classification": classification, "valid": check["valid"],
                "failed_checks": ";".join(check.get("failed_checks", [])), "branch_bundle_sha256": bundle})

    missing = 216 - reused
    fields = ["path", "context_id", "force_N", "classification", "valid", "failed_checks", "branch_bundle_sha256"]
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "schema": "TIME_BOUNDED_MASS_REUSE_SUMMARY_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset_sha256": subset_hash,
        "total_previously_completed_valid_branches": valid_total,
        "reused_primary_branches": reused,
        "auxiliary_preserved_branches": auxiliary,
        "missing_primary_branches": missing,
        "incomplete_claims": incomplete,
        "hash_or_parity_failures": invalid,
        "primary_target": 216,
        "primary_plus_missing_equals_target": reused + missing == 216,
        "result_driven_reruns": 0,
        "audit_csv_sha256": sha256(csv_path),
        "source_sha256": {
            "audit_time_bounded_mass_reuse_v1.py": sha256(Path(__file__)),
            "time_bounded_mass_common.py": sha256(HERE / "time_bounded_mass_common.py"),
        },
    }
    write_new_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    if incomplete or invalid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
