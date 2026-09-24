#!/usr/bin/env python3
"""Assemble immutable ID1 smoke plus a validated remaining-four batch.

This is a post-gate, copy-only operation.  It never modifies either replay root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py


ID1_ROOT = Path(
    "/media/volume/newdata/exouser/activeforcing_e3/"
    "TASK5_ONBOARDING_REPLAY5_20260902_105400_RETRY1/smoke_demo1"
)
ID1_HDF5_SHA256 = "1d3feed9de6ea2fa22cee68e32e0bb1d3676868b269b51dfb407b553430a181b"
IDS = (1, 2, 11, 12, 19)
STEPS = {1: 187, 2: 180, 11: 200, 12: 169, 19: 165}
OUT_NAME = "libero_10_task5_book_caddy_onboarding_5demo_7dpf_demo.hdf5"
STREAMS = {
    "agentview_rgb": "videos",
    "eye_in_hand_rgb": "videos",
    "gsmini_left_tactile_rgb": "tactile_outputs",
    "gsmini_right_tactile_rgb": "tactile_outputs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("remaining4_raw_root", type=Path)
    parser.add_argument("fresh_output_root", type=Path)
    args = parser.parse_args()
    raw_root = args.remaining4_raw_root.resolve()
    batch_root = raw_root / "normalized_source_ids"
    output_root = args.fresh_output_root.resolve()
    if output_root.exists():
        raise RuntimeError(f"refuse to overwrite assembled output: {output_root}")

    id1_qa = json.loads((ID1_ROOT / "E3_TASK5_ONBOARDING_REPLAY_QA.json").read_text())
    batch_qa = json.loads((raw_root / "E3_TASK5_ONBOARDING_REMAINING4_GATE.json").read_text())
    if id1_qa.get("gate_pass") is not True or id1_qa.get("errors"):
        raise RuntimeError("ID1 accepted output no longer passes producer QA")
    if batch_qa.get("gate_pass") is not True or batch_qa.get("errors"):
        raise RuntimeError("remaining-four batch has not passed strict QA")
    if batch_qa.get("selected_demo_ids") != [2, 11, 12, 19]:
        raise RuntimeError("remaining-four QA source IDs mismatch")
    if Path(batch_qa["normalized"]["root"]).resolve() != batch_root:
        raise RuntimeError("remaining-four normalized-root identity mismatch")

    id1_hdf5s = sorted((ID1_ROOT / "replayed_demos").glob("*.hdf5"))
    batch_hdf5s = sorted((batch_root / "replayed_demos").glob("*.hdf5"))
    if len(id1_hdf5s) != 1 or sha256(id1_hdf5s[0]) != ID1_HDF5_SHA256:
        raise RuntimeError("ID1 immutable HDF5 is missing or hash-mismatched")
    if len(batch_hdf5s) != 1 or sha256(batch_hdf5s[0]) != batch_qa["normalized"]["hdf5_sha256"]:
        raise RuntimeError("remaining-four normalized HDF5 is missing or hash-mismatched")

    replayed = output_root / "replayed_demos"
    media_base = output_root / "video_datasets" / "libero_10_task5"
    replayed.mkdir(parents=True)
    (media_base / "videos").mkdir(parents=True)
    (media_base / "tactile_outputs").mkdir(parents=True)
    output_hdf5 = replayed / OUT_NAME
    records: list[dict[str, object]] = []

    with h5py.File(id1_hdf5s[0], "r") as id1, h5py.File(batch_hdf5s[0], "r") as batch, h5py.File(
        output_hdf5, "x"
    ) as target:
        target_data = target.create_group("data")
        for key, value in batch.attrs.items():
            target.attrs[key] = value
        for key, value in batch["data"].attrs.items():
            target_data.attrs[key] = value
        target_data.attrs["total"] = sum(STEPS.values())
        id1_groups = sorted(id1["data"].keys())
        if len(id1_groups) != 1:
            raise RuntimeError(f"ID1 expected one group, got {id1_groups}")
        id1.copy(id1["data"][id1_groups[0]], target_data, name="demo_1")
        for demo_id in IDS[1:]:
            name = f"demo_{demo_id}"
            if name not in batch["data"]:
                raise RuntimeError(f"remaining-four normalized HDF5 missing {name}")
            batch.copy(batch["data"][name], target_data, name=name)
        for demo_id in IDS:
            group = target_data[f"demo_{demo_id}"]
            if group["actions"].shape != (STEPS[demo_id], 13) or bool(group.attrs.get("success", False)) is not True:
                raise RuntimeError(f"assembled demo {demo_id} violates length/schema/success gate")

    for demo_id in IDS:
        source_root = ID1_ROOT if demo_id == 1 else batch_root
        source_media_base = source_root / "video_datasets" / "libero_10_task5"
        media_records: list[dict[str, str]] = []
        for stream, directory in STREAMS.items():
            name = f"demo_{demo_id}_{stream}.mp4"
            source = source_media_base / directory / name
            target = media_base / directory / name
            if not source.is_file():
                raise RuntimeError(f"missing accepted media: {source}")
            shutil.copy2(source, target)
            source_hash = sha256(source)
            target_hash = sha256(target)
            if source_hash != target_hash:
                raise RuntimeError(f"media copy digest mismatch: {name}")
            media_records.append({"path": str(target), "sha256": target_hash})
        records.append({"source_demo_id": demo_id, "steps": STEPS[demo_id], "media": media_records})

    manifest = {
        "status": "TASK5_5DEMO_REAL_TACTILE_7DPF_ASSEMBLED",
        "task": "libero_10/task5",
        "split": "TRAIN",
        "selected_demo_ids": list(IDS),
        "total_steps": sum(STEPS.values()),
        "hdf5": str(output_hdf5),
        "hdf5_sha256": sha256(output_hdf5),
        "id1_root": str(ID1_ROOT),
        "remaining4_raw_root": str(raw_root),
        "records": records,
        "raw_inputs_modified": False,
        "activeforcing_or_test_outcome_used": False,
        "utility_or_fmax_changed": False,
    }
    (output_root / "ASSEMBLED_5DEMO_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
