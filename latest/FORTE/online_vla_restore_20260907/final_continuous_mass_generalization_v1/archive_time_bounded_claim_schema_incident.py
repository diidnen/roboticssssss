#!/usr/bin/env python3
"""Preserve four pre-physics claim-schema failures and reopen canonical paths."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from time_bounded_mass_common import HERE, sha256


RELATIVE_JOBS = [
    "t1_r181000_m100g/F3",
    "t1_r181001_m100g/F4",
    "t1_r181002_m100g/F5",
    "t1_r181003_m100g/F3",
]
INCIDENT_ID = "TIME_BOUNDED_CLAIM_SCHEMA_20260910T0644Z"


def main() -> None:
    record_path = HERE / "TIME_BOUNDED_MASS_INFRASTRUCTURE_INCIDENTS.json"
    if record_path.exists():
        raise FileExistsError(record_path)
    archive_root = HERE / "branches" / "_infrastructure_incidents" / INCIDENT_ID
    records = []
    for relative in RELATIVE_JOBS:
        source = HERE / "branches" / relative
        expected_names = {"LAUNCH_CLAIM.json", "WORKER_STDOUT.log", "WORKER_STDERR.log"}
        if not source.is_dir() or {path.name for path in source.iterdir()} != expected_names:
            raise RuntimeError(f"incident directory is not the exact pre-physics failure shape: {source}")
        if (source / "WORKER_STDOUT.log").stat().st_size != 0:
            raise RuntimeError(f"unexpected worker stdout indicates possible physics startup: {source}")
        stderr = (source / "WORKER_STDERR.log").read_text()
        if "Missing or incorrect exclusive launch claim" not in stderr:
            raise RuntimeError(f"unexpected failure reason: {source}")
        before = {path.name: sha256(path) for path in source.iterdir()}
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        source.rename(destination)
        after = {path.name: sha256(path) for path in destination.iterdir()}
        if before != after or source.exists():
            raise RuntimeError(f"lossless archival check failed: {source}")
        records.append({"canonical_path": str(source), "archive_path": str(destination),
                        "artifact_sha256": after, "physics_steps_executed": 0,
                        "scientific_outcome_observed": False})
    result = {
        "schema": "TIME_BOUNDED_MASS_INFRASTRUCTURE_INCIDENTS_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "incidents": [{
            "incident_id": INCIDENT_ID,
            "classification": "PRE_PHYSICS_LAUNCH_CLAIM_SCHEMA_MISMATCH",
            "cause": "Reduced launcher added provenance fields to a claim that the frozen worker requires to equal the original three-field schema.",
            "repair": "Reduced launcher restored the exact original claim schema. Failed directories were losslessly renamed to an immutable archive; none was deleted or overwritten.",
            "worker_or_runtime_modified": False,
            "physical_steps_executed": 0,
            "scientific_result_rerun": False,
            "records": records,
        }],
        "result_driven_reruns": 0,
        "source_sha256": sha256(Path(__file__)),
    }
    with record_path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps({"status": "ARCHIVED_PRE_PHYSICS", "count": len(records),
                      "archive_root": str(archive_root)}, indent=2))


if __name__ == "__main__":
    main()
