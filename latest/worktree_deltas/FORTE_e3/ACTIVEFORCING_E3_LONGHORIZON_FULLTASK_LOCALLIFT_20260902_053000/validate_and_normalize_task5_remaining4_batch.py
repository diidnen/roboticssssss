#!/usr/bin/env python3
"""Fail-closed QA and identity normalization for the task5 remaining-four replay.

The native recorder numbers exported HDF5 groups by successful-export order,
while media files retain source demo IDs.  This tool maps episodes back to the
frozen source IDs using exact initial-state tree digests, validates the complete
13D+tactile contract, and writes a separate normalized copy.  Raw replay output
is read-only and is never renamed or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np


EXPECTED_IDS = (2, 11, 12, 19)
EXPECTED_STEPS = {2: 180, 11: 200, 12: 169, 19: 165}
EXPECTED_REPLAY_UTILS_SHA256 = "bd9a831c34219fb5ae5f318863784b0daba6809d228fd1448f23fc941c9a6097"
EXPECTED_FFMPEG_SHA256 = "8ca0469917dae545e734473175974de1aa1e500a1f5fb7f0208513cb023bf495"
REQUIRED_OBS_SHAPES = {
    "eef_pose": lambda steps: (steps, 7),
    "gripper_pos": lambda steps: (steps, 2),
    "gripper_net_force": lambda steps: (steps, 1, 2, 3),
    "gripper_marker_motion": lambda steps: (steps, 2, 2, 99, 2),
}
MEDIA_SPECS = {
    "agentview_rgb": ("videos", (512, 512)),
    "eye_in_hand_rgb": ("videos", (512, 512)),
    "gsmini_left_tactile_rgb": ("tactile_outputs", (320, 240)),
    "gsmini_right_tactile_rgb": ("tactile_outputs", (320, 240)),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attribute_bytes(value: object) -> bytes:
    array = np.asarray(value)
    return b"|".join(
        (
            str(array.dtype).encode(),
            repr(array.shape).encode(),
            np.ascontiguousarray(array).tobytes(),
        )
    )


def tree_digest(node: h5py.Group | h5py.Dataset) -> str:
    digest = hashlib.sha256()

    def visit(current: h5py.Group | h5py.Dataset, relative: str) -> None:
        digest.update(type(current).__name__.encode())
        digest.update(relative.encode())
        for key in sorted(current.attrs):
            digest.update(b"A")
            digest.update(key.encode())
            digest.update(_attribute_bytes(current.attrs[key]))
        if isinstance(current, h5py.Dataset):
            value = np.asarray(current[()])
            digest.update(str(value.dtype).encode())
            digest.update(repr(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
            return
        for key in sorted(current.keys()):
            child_relative = f"{relative}/{key}" if relative else key
            visit(current[key], child_relative)

    visit(node, "")
    return digest.hexdigest()


def inspect_video(path: Path) -> dict[str, object]:
    record: dict[str, object] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return record
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        record["opened"] = False
        return record
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded = 0
    shapes: set[tuple[int, int]] = set()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded += 1
        shapes.add((int(frame.shape[1]), int(frame.shape[0])))
    capture.release()
    record.update(
        {
            "opened": True,
            "declared_frames": declared,
            "decoded_frames": decoded,
            "fps": fps,
            "resolutions": [list(shape) for shape in sorted(shapes)],
            "sha256": file_sha256(path),
        }
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_subset_root", type=Path)
    parser.add_argument("raw_replay_root", type=Path)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    subset_root = args.input_subset_root.resolve()
    raw_root = args.raw_replay_root.resolve()
    manifest_path = subset_root / "REMAINING4_INPUT_MANIFEST.json"
    qa_path = raw_root / "E3_TASK5_ONBOARDING_REMAINING4_GATE.json"
    errors: list[str] = []

    if not manifest_path.is_file():
        raise RuntimeError(f"missing input manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("selected_demo_ids") != list(EXPECTED_IDS):
        errors.append("input manifest does not contain exactly frozen IDs [2,11,12,19]")
    if manifest.get("excluded_smoke_demo_id") != 1:
        errors.append("input manifest does not prove ID1 exclusion")
    source_hdf5 = Path(manifest["source_hdf5"])
    subset_hdf5 = Path(manifest["subset_hdf5"])
    if not subset_hdf5.is_file() or file_sha256(subset_hdf5) != manifest.get("subset_hdf5_sha256"):
        errors.append("immutable subset HDF5 missing or hash-mismatched")

    source_initial_digest_to_id: dict[str, int] = {}
    source_group_digests: dict[int, str] = {}
    with h5py.File(source_hdf5, "r") as source:
        for demo_id in EXPECTED_IDS:
            group = source["data"][f"demo_{demo_id}"]
            initial_digest = tree_digest(group["initial_state"])
            if initial_digest in source_initial_digest_to_id:
                errors.append(f"ambiguous source initial-state digest for demo {demo_id}")
            source_initial_digest_to_id[initial_digest] = demo_id
            source_group_digests[demo_id] = tree_digest(group)

    raw_hdf5s = sorted((raw_root / "replayed_demos").glob("*.hdf5"))
    if len(raw_hdf5s) != 1:
        errors.append(f"expected exactly one raw HDF5, found {len(raw_hdf5s)}")

    mapping: dict[str, int] = {}
    episode_records: list[dict[str, object]] = []
    raw_group_digests: dict[str, str] = {}
    if len(raw_hdf5s) == 1:
        with h5py.File(raw_hdf5s[0], "r") as raw:
            groups = sorted(raw.get("data", {}).keys())
            if len(groups) != len(EXPECTED_IDS):
                errors.append(f"expected four semantically successful exported groups, found {len(groups)}")
            for group_name in groups:
                group = raw["data"][group_name]
                initial_state = group.get("initial_state")
                if initial_state is None:
                    errors.append(f"{group_name}: missing initial_state")
                    continue
                initial_digest = tree_digest(initial_state)
                demo_id = source_initial_digest_to_id.get(initial_digest)
                if demo_id is None:
                    errors.append(f"{group_name}: initial state does not match a frozen source ID")
                    continue
                if demo_id in mapping.values():
                    errors.append(f"{group_name}: duplicate mapping to source demo {demo_id}")
                    continue
                mapping[group_name] = demo_id
                steps = EXPECTED_STEPS[demo_id]
                record: dict[str, object] = {
                    "raw_group": group_name,
                    "source_demo_id": demo_id,
                    "source_initial_state_digest": initial_digest,
                    "source_group_digest": source_group_digests[demo_id],
                    "success": bool(group.attrs.get("success", False)),
                }
                if record["success"] is not True:
                    errors.append(f"{group_name}: semantic success attribute is not true")
                actions = group.get("actions")
                shape = tuple(actions.shape) if actions is not None else None
                record["actions_shape"] = list(shape) if shape else None
                if shape != (steps, 13):
                    errors.append(f"{group_name}: expected actions {(steps, 13)}, got {shape}")
                elif not np.isfinite(np.asarray(actions)).all():
                    errors.append(f"{group_name}: actions contain non-finite values")

                obs = group.get("obs")
                obs_record: dict[str, object] = {}
                if obs is None:
                    errors.append(f"{group_name}: missing obs")
                else:
                    for key, shape_fn in REQUIRED_OBS_SHAPES.items():
                        dataset = obs.get(key)
                        actual_shape = tuple(dataset.shape) if dataset is not None else None
                        nonzero = int(np.count_nonzero(np.asarray(dataset))) if dataset is not None else 0
                        obs_record[key] = {
                            "shape": list(actual_shape) if actual_shape else None,
                            "finite": bool(np.isfinite(np.asarray(dataset)).all()) if dataset is not None else False,
                            "nonzero_scalars": nonzero,
                        }
                        if actual_shape != shape_fn(steps):
                            errors.append(f"{group_name}: {key} shape {actual_shape} is invalid")
                        elif not np.isfinite(np.asarray(dataset)).all():
                            errors.append(f"{group_name}: {key} contains non-finite values")
                        if key in {"gripper_net_force", "gripper_marker_motion"} and nonzero == 0:
                            errors.append(f"{group_name}: {key} is entirely zero")
                record["obs"] = obs_record
                raw_group_digests[group_name] = tree_digest(group)
                episode_records.append(record)

    if set(mapping.values()) != set(EXPECTED_IDS):
        errors.append(f"source-ID mapping is not a bijection: {mapping}")

    media_records: dict[str, object] = {}
    media_base = raw_root / "video_datasets" / "libero_10_task5"
    for demo_id in EXPECTED_IDS:
        steps = EXPECTED_STEPS[demo_id]
        demo_media: dict[str, object] = {}
        for stream, (directory, expected_resolution) in MEDIA_SPECS.items():
            path = media_base / directory / f"demo_{demo_id}_{stream}.mp4"
            record = inspect_video(path)
            demo_media[stream] = record
            if record.get("opened") is not True:
                errors.append(f"demo {demo_id}: {stream} is missing or undecodable")
                continue
            if record.get("decoded_frames") != steps or record.get("declared_frames") != steps:
                errors.append(f"demo {demo_id}: {stream} frame count is not {steps}")
            if record.get("resolutions") != [list(expected_resolution)]:
                errors.append(f"demo {demo_id}: {stream} resolution mismatch")
            if abs(float(record.get("fps", 0.0)) - 20.0) > 1e-6:
                errors.append(f"demo {demo_id}: {stream} FPS is not 20")
        media_records[str(demo_id)] = demo_media

    failure_records = sorted(raw_root.glob("failure*.jsonl"))
    nonempty_failures = [str(path) for path in failure_records if path.stat().st_size > 0]
    if nonempty_failures:
        errors.append(f"failure record is nonempty: {nonempty_failures}")

    report: dict[str, object] = {
        "status": "REMAINING4_GATE_PASS" if not errors else "REMAINING4_GATE_FAIL",
        "gate_pass": not errors,
        "errors": errors,
        "task": "libero_10/task5",
        "split": "TRAIN",
        "selected_demo_ids": list(EXPECTED_IDS),
        "excluded_smoke_demo_id": 1,
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": file_sha256(manifest_path),
        "raw_replay_root": str(raw_root),
        "raw_hdf5": str(raw_hdf5s[0]) if len(raw_hdf5s) == 1 else None,
        "raw_hdf5_sha256": file_sha256(raw_hdf5s[0]) if len(raw_hdf5s) == 1 else None,
        "raw_group_to_source_demo_id": mapping,
        "episodes": episode_records,
        "media": media_records,
        "nonempty_failure_records": nonempty_failures,
        "expected_replay_utils_sha256": EXPECTED_REPLAY_UTILS_SHA256,
        "expected_ffmpeg_sha256": EXPECTED_FFMPEG_SHA256,
        "activeforcing_or_test_outcome_used": False,
        "utility_or_fmax_changed": False,
    }

    if args.normalize and not errors:
        normalized = raw_root / "normalized_source_ids"
        if normalized.exists():
            errors.append(f"refuse to overwrite normalized output: {normalized}")
        else:
            normalized_hdf5_dir = normalized / "replayed_demos"
            normalized_media_base = normalized / "video_datasets" / "libero_10_task5"
            normalized_hdf5_dir.mkdir(parents=True)
            (normalized_media_base / "videos").mkdir(parents=True)
            (normalized_media_base / "tactile_outputs").mkdir(parents=True)
            normalized_hdf5 = normalized_hdf5_dir / raw_hdf5s[0].name
            normalized_group_digests: dict[str, str] = {}
            with h5py.File(raw_hdf5s[0], "r") as source, h5py.File(normalized_hdf5, "x") as target:
                for key, value in source.attrs.items():
                    target.attrs[key] = value
                target_data = target.create_group("data")
                for key, value in source["data"].attrs.items():
                    target_data.attrs[key] = value
                for raw_group, demo_id in sorted(mapping.items(), key=lambda item: item[1]):
                    source.copy(source["data"][raw_group], target_data, name=f"demo_{demo_id}")
                    copied_digest = tree_digest(target_data[f"demo_{demo_id}"])
                    if copied_digest != raw_group_digests[raw_group]:
                        errors.append(f"demo {demo_id}: normalized HDF5 content digest mismatch")
                    normalized_group_digests[str(demo_id)] = copied_digest

            copied_media: list[dict[str, str]] = []
            for demo_id in EXPECTED_IDS:
                for stream, (directory, _) in MEDIA_SPECS.items():
                    name = f"demo_{demo_id}_{stream}.mp4"
                    source_path = media_base / directory / name
                    target_path = normalized_media_base / directory / name
                    shutil.copy2(source_path, target_path)
                    source_hash = file_sha256(source_path)
                    target_hash = file_sha256(target_path)
                    if source_hash != target_hash:
                        errors.append(f"demo {demo_id}: normalized media hash mismatch for {stream}")
                    copied_media.append({"path": str(target_path), "sha256": target_hash})
            report["normalized"] = {
                "root": str(normalized),
                "hdf5": str(normalized_hdf5),
                "hdf5_sha256": file_sha256(normalized_hdf5),
                "group_content_digests": normalized_group_digests,
                "media": copied_media,
                "raw_output_modified": False,
            }

    report["gate_pass"] = not errors
    report["errors"] = errors
    report["status"] = "REMAINING4_GATE_PASS" if not errors else "REMAINING4_GATE_FAIL"
    qa_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
