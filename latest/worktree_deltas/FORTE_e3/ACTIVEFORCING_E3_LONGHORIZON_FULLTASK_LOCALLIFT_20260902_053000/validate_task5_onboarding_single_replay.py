#!/usr/bin/env python3
"""Independent fail-closed gate for one task5 real-tactile 7dpf replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


SOURCE = Path(
    "/media/volume/newdata/exouser/activeforcing_e3/PROJECT_LIBERO_TASK5_DEMOS_20260902/"
    "assembled_hdf5/libero_10_task5_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_"
    "back_compartment_of_the_caddy_demo.hdf5"
)
SOURCE_SHA256 = "98fb9f3fab6b46a0fbb6d63fc686dccfbb31a372c825c0c87da8fe21f99a6219"
EXPECTED_STEPS = {1: 187, 2: 180, 11: 200, 12: 169, 19: 165}
REQUIRED_OBS_SHAPES = {
    "eef_pose": lambda steps: (steps, 7),
    "gripper_pos": lambda steps: (steps, 2),
    "gripper_net_force": lambda steps: (steps, 1, 2, 3),
    "gripper_marker_motion": lambda steps: (steps, 2, 2, 99, 2),
}
MEDIA = {
    "agentview_rgb": ("videos", (512, 512)),
    "eye_in_hand_rgb": ("videos", (512, 512)),
    "gsmini_left_tactile_rgb": ("tactile_outputs", (320, 240)),
    "gsmini_right_tactile_rgb": ("tactile_outputs", (320, 240)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(node: h5py.Group | h5py.Dataset) -> str:
    digest = hashlib.sha256()

    def visit(current: h5py.Group | h5py.Dataset, relative: str) -> None:
        digest.update(type(current).__name__.encode())
        digest.update(relative.encode())
        for key in sorted(current.attrs):
            value = np.asarray(current.attrs[key])
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(repr(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        if isinstance(current, h5py.Dataset):
            value = np.asarray(current[()])
            digest.update(str(value.dtype).encode())
            digest.update(repr(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
            return
        for key in sorted(current.keys()):
            visit(current[key], f"{relative}/{key}" if relative else key)

    visit(node, "")
    return digest.hexdigest()


def inspect_video(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        result["opened"] = False
        return result
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
    result.update(
        {
            "opened": True,
            "declared_frames": declared,
            "decoded_frames": decoded,
            "fps": fps,
            "resolutions": [list(shape) for shape in sorted(shapes)],
            "sha256": sha256(path),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("demo_id", type=int, choices=sorted(EXPECTED_STEPS))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    demo_id = args.demo_id
    steps = EXPECTED_STEPS[demo_id]
    errors: list[str] = []

    if sha256(SOURCE) != SOURCE_SHA256:
        errors.append("authoritative source HDF5 hash mismatch")
    with h5py.File(SOURCE, "r") as source:
        source_group = source["data"][f"demo_{demo_id}"]
        source_initial_digest = tree_digest(source_group["initial_state"])
        source_group_digest = tree_digest(source_group)

    hdf5s = sorted((root / "replayed_demos").glob("*.hdf5"))
    episode: dict[str, object] = {}
    if len(hdf5s) != 1:
        errors.append(f"expected one replay HDF5, found {len(hdf5s)}")
    else:
        with h5py.File(hdf5s[0], "r") as replay:
            groups = sorted(replay.get("data", {}).keys())
            if len(groups) != 1:
                errors.append(f"expected one successful group, found {groups}")
            elif groups:
                group = replay["data"][groups[0]]
                replay_initial_digest = tree_digest(group["initial_state"]) if "initial_state" in group else None
                if replay_initial_digest != source_initial_digest:
                    errors.append("replay initial state does not exactly match frozen source demo")
                if bool(group.attrs.get("success", False)) is not True:
                    errors.append("semantic success attribute is not true")
                actions = group.get("actions")
                action_shape = tuple(actions.shape) if actions is not None else None
                if action_shape != (steps, 13):
                    errors.append(f"actions shape {action_shape}, expected {(steps, 13)}")
                elif not np.isfinite(np.asarray(actions)).all():
                    errors.append("actions contain non-finite values")
                obs = group.get("obs")
                obs_result: dict[str, object] = {}
                if obs is None:
                    errors.append("missing obs group")
                else:
                    for key, expected_shape in REQUIRED_OBS_SHAPES.items():
                        dataset = obs.get(key)
                        shape = tuple(dataset.shape) if dataset is not None else None
                        finite = bool(np.isfinite(np.asarray(dataset)).all()) if dataset is not None else False
                        nonzero = int(np.count_nonzero(np.asarray(dataset))) if dataset is not None else 0
                        obs_result[key] = {
                            "shape": list(shape) if shape else None,
                            "finite": finite,
                            "nonzero_scalars": nonzero,
                        }
                        if shape != expected_shape(steps):
                            errors.append(f"{key} shape {shape}, expected {expected_shape(steps)}")
                        if not finite:
                            errors.append(f"{key} contains non-finite values")
                        if key in {"gripper_net_force", "gripper_marker_motion"} and nonzero == 0:
                            errors.append(f"{key} is entirely zero")
                episode = {
                    "raw_group": groups[0],
                    "semantic_success": bool(group.attrs.get("success", False)),
                    "steps": steps,
                    "actions_shape": list(action_shape) if action_shape else None,
                    "source_initial_state_digest": source_initial_digest,
                    "replay_initial_state_digest": replay_initial_digest,
                    "source_group_digest": source_group_digest,
                    "replay_group_digest": tree_digest(group),
                    "obs": obs_result,
                }

    media_base = root / "video_datasets" / "libero_10_task5"
    media_result: dict[str, object] = {}
    for stream, (directory, resolution) in MEDIA.items():
        path = media_base / directory / f"demo_{demo_id}_{stream}.mp4"
        record = inspect_video(path)
        media_result[stream] = record
        if record.get("opened") is not True:
            errors.append(f"{stream} missing or undecodable")
        else:
            if record.get("declared_frames") != steps or record.get("decoded_frames") != steps:
                errors.append(f"{stream} frame count mismatch")
            if record.get("resolutions") != [list(resolution)]:
                errors.append(f"{stream} resolution mismatch")
            if abs(float(record.get("fps", 0.0)) - 20.0) > 1e-6:
                errors.append(f"{stream} FPS mismatch")

    failure_records = sorted(root.glob("failure*.jsonl"))
    nonempty_failures = [str(path) for path in failure_records if path.stat().st_size > 0]
    if nonempty_failures:
        errors.append(f"nonempty failure record: {nonempty_failures}")
    producer_qa = root / "E3_TASK5_ONBOARDING_REPLAY_QA.json"

    result = {
        "status": f"DEMO{demo_id}_REAL_TACTILE_7DPF_GATE_PASS" if not errors else f"DEMO{demo_id}_REAL_TACTILE_7DPF_GATE_FAIL",
        "gate_pass": not errors,
        "errors": errors,
        "task": "libero_10/task5",
        "split": "TRAIN",
        "source_demo_id": demo_id,
        "root": str(root),
        "hdf5": str(hdf5s[0]) if len(hdf5s) == 1 else None,
        "hdf5_sha256": sha256(hdf5s[0]) if len(hdf5s) == 1 else None,
        "episode": episode,
        "media": media_result,
        "nonempty_failure_records": nonempty_failures,
        "producer_qa_sha256": sha256(producer_qa) if producer_qa.is_file() else None,
        "test_used": False,
        "activeforcing_or_force_outcome_supervision_used": False,
        "utility_or_fmax_changed": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
