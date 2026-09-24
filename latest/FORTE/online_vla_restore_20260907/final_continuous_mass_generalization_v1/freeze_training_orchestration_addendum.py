#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE / "launch_mass_training_partition.py"
value = {
    "version": "MASS_TRAINING_ORCHESTRATION_ADDENDUM_V1",
    "reason": "Schedule the two untouched TRAIN root partitions concurrently after VAL and HELDOUT completed.",
    "affected_roots": [181002, 181003],
    "query_or_physics_runtime_changed": False,
    "worker_changed": False,
    "context_plan_changed": False,
    "admission_changed": False,
    "outcomes_used": False,
    "partition_launcher": str(source),
    "partition_launcher_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "frozen_training_runtime_manifest_sha256": hashlib.sha256((HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json").read_bytes()).hexdigest()
}
with (HERE / "MASS_TRAINING_ORCHESTRATION_ADDENDUM.json").open("x") as stream:
    json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
