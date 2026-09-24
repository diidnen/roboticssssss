#!/usr/bin/env python3
"""Create or verify the pre-outcome task0 context pipeline hash freeze."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
PROJECT = ROOT / "task0_context_sample_complexity_20260831"
FREEZE = PROJECT / "TASK0_CONTEXT_PIPELINE_FREEZE_SHA256.json"
FILES = [
    ROOT / "TASK0_CONTEXT_SPLIT_MANIFEST.json",
    ROOT / "TASK0_CONTEXT_CONCLUSION_RULES.json",
    PROJECT / "TASK0_CONTEXT_SPLIT_FREEZE_SHA256.json",
    PROJECT / "TASK0_NEW_TRAIN_CONTEXTS.json",
    PROJECT / "TASK0_NEW_TRAIN_TARGET_MANIFEST.json",
    PROJECT / "TASK0_FROZEN_TEST_CONTEXTS.json",
    PROJECT / "TASK0_FROZEN_TEST_TARGET_MANIFEST.json",
    ROOT / "prepare_task0_context_split.py",
    ROOT / "task0_context_pipeline_freeze.py",
    ROOT / "task0_context_collect.py",
    ROOT / "audit_task0_context_collection.py",
    ROOT / "task0_context_learning_curve.py",
    ROOT / "prospective_visual_context_collect.py",
    ROOT / "task0_visual_context_early.py",
    ROOT / "task0_visual_generalization.py",
    ROOT / "task0_joint_novisual_diagnostic.py",
    Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def create() -> None:
    forbidden = [
        PROJECT / "collection_train_new",
        PROJECT / "collection_test_frozen",
        PROJECT / "checkpoints",
        PROJECT / "TASK0_CONTEXT_CHECKPOINT_MANIFEST.json",
    ]
    if any(p.exists() for p in forbidden):
        raise RuntimeError("pipeline freeze must precede all new collection/checkpoints")
    missing = [str(p) for p in FILES if not p.exists()]
    if missing:
        raise RuntimeError(f"pipeline freeze inputs missing: {missing}")
    obj = {
        "status": "FROZEN_BEFORE_NEW_COLLECTION_OUTCOMES_OR_NEW_MODEL_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "same task0; same object/task distribution; held-out execution roots/contexts",
        "new_collection_outcomes_present": False,
        "new_learning_curve_checkpoints_present": False,
        "hashes": {str(p): sha256(p) for p in FILES},
    }
    FREEZE.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": obj["status"], "files": len(FILES), "freeze": str(FREEZE)}, indent=2))


def verify() -> dict:
    obj = json.loads(FREEZE.read_text())
    if obj.get("status") != "FROZEN_BEFORE_NEW_COLLECTION_OUTCOMES_OR_NEW_MODEL_TRAINING":
        raise RuntimeError("pipeline freeze status invalid")
    expected = obj.get("hashes", {})
    if set(expected) != {str(p) for p in FILES}:
        raise RuntimeError("pipeline freeze file set mismatch")
    for p in FILES:
        got = sha256(p)
        if got != expected[str(p)]:
            raise RuntimeError(f"frozen pipeline hash changed: {p}: {got} != {expected[str(p)]}")
    return obj


if __name__ == "__main__":
    create()
