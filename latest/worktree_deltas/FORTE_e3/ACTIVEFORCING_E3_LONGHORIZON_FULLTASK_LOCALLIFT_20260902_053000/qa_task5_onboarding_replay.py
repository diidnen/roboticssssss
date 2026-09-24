#!/usr/bin/env python3
"""Strict, read-only QA for an E3 task5 7dpf replay output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


REQUIRED_OBS = {
    "eef_pose",
    "gripper_pos",
    "gripper_net_force",
    "gripper_marker_motion",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--demo-id", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    hdf5_files = sorted((root / "replayed_demos").glob("*.hdf5"))
    errors: list[str] = []
    if len(hdf5_files) != 1:
        errors.append(f"expected exactly one HDF5, found {len(hdf5_files)}")

    report: dict[str, object] = {
        "root": str(root),
        "hdf5_files": [str(path) for path in hdf5_files],
        "errors": errors,
    }
    if len(hdf5_files) == 1:
        path = hdf5_files[0]
        report["hdf5_sha256"] = sha256(path)
        with h5py.File(path, "r") as handle:
            demos = sorted(handle.get("data", {}).keys())
            report["demo_groups"] = demos
            if len(demos) != 1:
                errors.append(f"expected one successful exported demo, found {len(demos)}")
            if demos:
                demo = handle["data"][demos[0]]
                actions = demo.get("actions")
                action_shape = list(actions.shape) if actions is not None else None
                report["action_shape"] = action_shape
                if actions is None or actions.ndim != 2 or actions.shape[1] != 13:
                    errors.append(f"actions must be [T,13], got {action_shape}")
                obs = demo.get("obs")
                obs_keys = sorted(obs.keys()) if obs is not None else []
                report["obs_keys"] = obs_keys
                missing = sorted(REQUIRED_OBS.difference(obs_keys))
                report["missing_required_obs"] = missing
                if missing:
                    errors.append(f"missing required obs: {missing}")
                if not missing:
                    force = np.asarray(obs["gripper_net_force"])
                    marker = np.asarray(obs["gripper_marker_motion"])
                    action_force = np.asarray(actions)[:, 7:13]
                    marker_delta = marker[:, :, 1] - marker[:, :, 0]
                    signal = {
                        "action_force_abs_max": float(np.abs(action_force).max()),
                        "action_force_nonzero": int(np.count_nonzero(action_force)),
                        "gripper_net_force_shape": list(force.shape),
                        "gripper_net_force_abs_max": float(np.abs(force).max()),
                        "gripper_net_force_nonzero": int(np.count_nonzero(force)),
                        "gripper_marker_motion_shape": list(marker.shape),
                        "gripper_marker_delta_abs_max": float(np.abs(marker_delta).max()),
                        "gripper_marker_delta_nonzero": int(np.count_nonzero(marker_delta)),
                    }
                    report["real_signal_contract"] = signal
                    if force.ndim != 4 or force.shape[0] != actions.shape[0] or force.shape[-2:] != (2, 3):
                        errors.append(f"invalid gripper_net_force shape: {force.shape}")
                    if marker.ndim != 5 or marker.shape[0] != actions.shape[0] or marker.shape[1:3] != (2, 2):
                        errors.append(f"invalid gripper_marker_motion shape: {marker.shape}")
                    if not np.any(action_force) or not np.any(force) or not np.any(marker_delta):
                        errors.append("force/marker signals must contain measured nonzero values")
                success = bool(demo.attrs.get("success", False))
                report["success_attr"] = success
                if not success:
                    errors.append("exported demo success attribute is not true")

    task_media = root / "video_datasets" / "libero_10_task5"
    demo_id = args.demo_id
    required_patterns = {
        "agentview_rgb": f"videos/demo_{demo_id}_agentview_rgb.mp4",
        "eye_in_hand_rgb": f"videos/demo_{demo_id}_eye_in_hand_rgb.mp4",
        "gsmini_left_tactile_rgb": f"tactile_outputs/demo_{demo_id}_gsmini_left_tactile_rgb.mp4",
        "gsmini_right_tactile_rgb": f"tactile_outputs/demo_{demo_id}_gsmini_right_tactile_rgb.mp4",
    }
    media: dict[str, object] = {}
    for name, relative in required_patterns.items():
        path = task_media / relative
        frames = count_frames(path) if path.exists() else 0
        media[name] = {"path": str(path), "exists": path.exists(), "frames": frames}
        if not path.exists() or frames <= 0:
            errors.append(f"missing or empty media stream: {name}")
    report["media"] = media
    report["source_demo_id"] = demo_id

    report["gate_pass"] = not errors
    output = root / "E3_TASK5_ONBOARDING_REPLAY_QA.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
