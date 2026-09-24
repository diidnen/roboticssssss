#!/usr/bin/env python3
"""Run the frozen root-scaling trainer for task0 only in an isolated output tree.

This is an engineering diagnostic.  It never opens TEST and never writes the
authoritative root_scaling_20260831 checkpoint manifest or model-hash files.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
AUTHORITATIVE = ROOT / "root_scaling_20260831"
DIAG = AUTHORITATIVE / "task0_diagnostic"
SOURCE_FILE = ROOT / "root_scaling_learning_curve.py"


def load_frozen_module():
    spec = importlib.util.spec_from_file_location("root_scaling_learning_curve_task0_diag", SOURCE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    DIAG.mkdir(parents=True, exist_ok=True)
    mod = load_frozen_module()
    mod.OUT = DIAG
    mod.TASKS = [0]

    # Read frozen inputs from the authoritative experiment, while keeping all
    # generated PCA/norm/checkpoint/manifest files under the diagnostic tree.
    frozen_path = AUTHORITATIVE / "ROOT_SCALING_FREEZE_SHA256.json"
    train_manifest_path = AUTHORITATIVE / "ROOT_SCALING_TRAIN_MANIFEST.json"

    def verify_diag_freeze():
        freeze = json.loads(frozen_path.read_text())
        for raw, expected in freeze["hashes"].items():
            if mod.sha256(Path(raw)) != expected:
                raise RuntimeError(f"frozen input changed: {raw}")
        return freeze, json.loads(train_manifest_path.read_text())

    mod.verify_freeze = verify_diag_freeze
    mod.train_all()

    # Make the diagnostic status explicit; the generated files are isolated,
    # but must not be mistaken for the authoritative 72-checkpoint freeze.
    manifest = DIAG / "ROOT_SCALING_CHECKPOINT_MANIFEST.json"
    payload = json.loads(manifest.read_text())
    payload.update({
        "status": "TASK0_DIAGNOSTIC_ONLY",
        "authoritative": False,
        "task_scope": [0],
        "TEST_used": False,
        "authoritative_manifest": str(AUTHORITATIVE / "ROOT_SCALING_CHECKPOINT_MANIFEST.json"),
    })
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (DIAG / "TASK0_DIAGNOSTIC_NOTICE.json").write_text(json.dumps({
        "status": "COMPLETE",
        "scope": "task0 TRAIN only",
        "scientific_protocol_changed": False,
        "TEST_used": False,
        "task5_skipped": True,
        "authoritative": False,
        "diagnostic_checkpoint_manifest": str(manifest),
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
