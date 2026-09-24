#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE / "launch_mass_branch_partition.py"
value = {"version": "MASS_MATCHED_BRANCH_ORCHESTRATION_ADDENDUM_V1",
    "reason": "Schedule root-disjoint matched force branches concurrently; projected single-worker runtime was 28 hours.",
    "root_partition_preserved": True, "sibling_force_branches_same_split": True,
    "query_or_physics_runtime_changed": False, "worker_changed": False, "force_order_changed": False,
    "outcomes_used": False, "partition_launcher": str(source),
    "partition_launcher_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "frozen_training_runtime_manifest_sha256": hashlib.sha256((HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json").read_bytes()).hexdigest()}
with (HERE / "MASS_MATCHED_BRANCH_ORCHESTRATION_ADDENDUM.json").open("x") as stream:
    json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
