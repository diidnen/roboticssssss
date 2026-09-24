#!/usr/bin/env python3
"""Freeze collection, training, evaluation, and classification code before TEST."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831"
SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
FILES = [
    OUT / "ROOT_SCALING_TEST_MANIFEST.json", OUT / "ROOT_SCALING_TRAIN_MANIFEST.json",
    OUT / "ROOT_SCALING_PREREGISTRATION.json", OUT / "ROOT_SCALING_FREEZE_SHA256.json",
    OUT / "collection_plans/TEST_CONTEXTS.json", OUT / "collection_plans/TRAIN_CONTEXTS.json",
    OUT / "collection_plans/TEST_TARGET_MANIFEST.json", OUT / "collection_plans/TRAIN_TARGET_MANIFEST.json",
    ROOT / "prepare_root_scaling.py", ROOT / "root_scaling_collect.py",
    ROOT / "audit_root_scaling_collection.py", ROOT / "root_scaling_learning_curve.py",
    ROOT / "root_scaling_finalize.py", ROOT / "freeze_root_scaling_pipeline.py",
    ROOT / "prospective_visual_context_collect.py", ROOT / "task0_visual_context_early.py",
    ROOT / "per_task_visual_context_early.py", ROOT / "task0_visual_generalization.py",
    Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py"),
    Path("/home/exouser/Tabero/analysis/trajectory_physical_imagination.py"),
    Path("/home/exouser/Tabero/analysis/counterfactual_force_world_model.py"),
    Path("/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py"),
    SOURCE / "PROSPECTIVE_VISUAL_FEATURE_SPEC.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    path = OUT / "ROOT_SCALING_PIPELINE_FREEZE_SHA256.json"
    if path.exists():
        obj = json.loads(path.read_text())
        for raw, expected in obj["hashes"].items():
            if sha256(Path(raw)) != expected:
                raise RuntimeError(f"pipeline changed after freeze: {raw}")
        print(json.dumps({"status": "ALREADY_FROZEN_AND_VERIFIED", "files": len(obj["hashes"])}, indent=2)); return
    forbidden = [OUT / "collection_test", OUT / "collection_train", OUT / "checkpoints",
                 OUT / "ROOT_SCALING_CHECKPOINT_MANIFEST.json", OUT / "ROOT_SCALING_LEARNING_CURVE.csv"]
    if any(p.exists() for p in forbidden):
        raise RuntimeError("pipeline freeze must precede root-scaling TEST/TRAIN collection and model training")
    missing = [str(p) for p in FILES if not p.exists()]
    if missing: raise RuntimeError("freeze inputs missing: " + json.dumps(missing))
    obj = {"status": "FROZEN_BEFORE_UNTOUCHED_TEST_OUTCOMES_AND_SCALING_MODEL_TRAINING",
           "created_utc": datetime.now(timezone.utc).isoformat(),
           "task0_new_TRAIN_collection_note": "A separately pre-outcome-frozen task0 addition collector is active; its exact matching context/force prefix will be reused read-only.",
           "untouched_TEST_outcomes_present": False, "scaling_models_trained": False,
           "classification_implementation_frozen": True,
           "hashes": {str(p): sha256(p) for p in FILES}}
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": obj["status"], "files": len(FILES)}, indent=2))


if __name__ == "__main__": main()
