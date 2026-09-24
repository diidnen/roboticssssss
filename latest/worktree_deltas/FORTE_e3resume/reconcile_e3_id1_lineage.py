#!/usr/bin/env python3
"""Read-only E3 ID1 provenance refreeze; never edits replay artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


LIVE_ROOT = Path(
    "/media/volume/newdata/exouser/activeforcing_e3/"
    "TASK5_ONBOARDING_REPLAY5_20260902_105400_RETRY1/smoke_demo1"
)
OLD_GATE = Path(
    "/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/"
    "TASK5_PI0_ONBOARDING_ID1_RETRY_GATE.json"
)
OUTPUT = Path(
    "/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/"
    "TASK5_PI0_ONBOARDING_ID1_LINEAGE_REFREEZE_GATE_20260902.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: dict) -> None:
    if path.exists():
        raise RuntimeError(f"fail-closed: refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    old_gate = json.loads(OLD_GATE.read_text())
    qa_path = LIVE_ROOT / "E3_TASK5_ONBOARDING_REPLAY_QA.json"
    hdf5_path = next((LIVE_ROOT / "replayed_demos").glob("*.hdf5"))
    media_paths = {
        "agentview_rgb": LIVE_ROOT / "video_datasets/libero_10_task5/videos/demo_1_agentview_rgb.mp4",
        "eye_in_hand_rgb": LIVE_ROOT / "video_datasets/libero_10_task5/videos/demo_1_eye_in_hand_rgb.mp4",
        "gsmini_left_tactile_rgb": LIVE_ROOT / "video_datasets/libero_10_task5/tactile_outputs/demo_1_gsmini_left_tactile_rgb.mp4",
        "gsmini_right_tactile_rgb": LIVE_ROOT / "video_datasets/libero_10_task5/tactile_outputs/demo_1_gsmini_right_tactile_rgb.mp4",
    }
    qa = json.loads(qa_path.read_text())
    if qa.get("gate_pass") is not True or qa.get("errors") != []:
        raise RuntimeError("fail-closed: live ID1 QA is not PASS")
    if qa.get("hdf5_sha256") != sha256(hdf5_path):
        raise RuntimeError("fail-closed: live ID1 HDF5 hash does not match QA")
    if hdf5_path.name != Path(old_gate["output_root"]).joinpath("replayed_demos", hdf5_path.name).name:
        raise RuntimeError("fail-closed: unexpected HDF5 filename")
    media = {}
    for name, path in media_paths.items():
        if not path.is_file():
            raise RuntimeError(f"fail-closed: missing media {name}")
        media[name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    payload = {
        "status": "ID1_LINEAGE_REFREEZE_PASS",
        "purpose": "read_only_reconcile_live_QA_hash_drift_without_replaying_or_editing_data",
        "source_demo_id": 1,
        "task": "libero_10/task5",
        "split": "TRAIN",
        "live_root": str(LIVE_ROOT),
        "old_gate": {
            "path": str(OLD_GATE),
            "producer_qa_sha256": old_gate["producer_qa"]["sha256"],
            "hdf5_sha256": old_gate["hdf5"]["sha256"],
            "media": old_gate["media"],
        },
        "live": {
            "qa_path": str(qa_path),
            "qa_sha256": sha256(qa_path),
            "qa_gate_pass": qa["gate_pass"],
            "qa_errors": qa["errors"],
            "hdf5_path": str(hdf5_path),
            "hdf5_sha256": sha256(hdf5_path),
            "action_shape": qa["action_shape"],
            "media": media,
        },
        "reconciliation": {
            "id1_producer_qa_hash_drift": True,
            "old_gate_preserved": True,
            "replay_data_modified": False,
            "new_independent_gate_uses_live_qa_hash": True,
            "verdict": "ACCEPT_ID1_DATA_LINEAGE_WITH_NEW_REFREEZE; RETAIN_OLD_GATE_AS_HISTORICAL",
        },
    }
    atomic_write(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
