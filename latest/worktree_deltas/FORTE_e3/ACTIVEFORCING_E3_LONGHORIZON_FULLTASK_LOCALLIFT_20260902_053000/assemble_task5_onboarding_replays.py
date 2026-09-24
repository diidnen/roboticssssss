#!/usr/bin/env python3
"""Assemble five independently replayed E3 demos without changing samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py


EXPECTED_IDS = (1, 2, 11, 12, 19)
TASK_STEM = "libero_10_task5_book_caddy_onboarding_5demo_7dpf_demo.hdf5"
REQUIRED_OBS = {"eef_pose", "gripper_pos", "gripper_net_force", "gripper_marker_motion"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_source(value: str) -> tuple[int, Path]:
    try:
        demo, root = value.split("=", 1)
        return int(demo), Path(root).resolve()
    except Exception as error:
        raise argparse.ArgumentTypeError("source must be DEMO_ID=/absolute/replay/root") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing existing output root: {output_root}")
    sources = dict(args.source)
    if sorted(sources) != sorted(EXPECTED_IDS) or len(args.source) != len(EXPECTED_IDS):
        raise ValueError(f"sources must contain each frozen ID exactly once: {EXPECTED_IDS}")

    out_hdf5_dir = output_root / "replayed_demos"
    out_video_dir = output_root / "video_datasets" / "libero_10_task5" / "videos"
    out_tactile_dir = output_root / "video_datasets" / "libero_10_task5" / "tactile_outputs"
    out_hdf5_dir.mkdir(parents=True)
    out_video_dir.mkdir(parents=True)
    out_tactile_dir.mkdir(parents=True)
    out_hdf5 = out_hdf5_dir / TASK_STEM

    records: list[dict[str, object]] = []
    total = 0
    env_args: object | None = None
    with h5py.File(out_hdf5, "w") as target:
        target_data = target.create_group("data")
        for demo_id in EXPECTED_IDS:
            root = sources[demo_id]
            qa_path = root / "E3_TASK5_ONBOARDING_REPLAY_QA.json"
            qa = json.loads(qa_path.read_text())
            if not qa.get("gate_pass") or int(qa.get("source_demo_id", -1)) != demo_id:
                raise ValueError(f"strict QA did not pass for source demo {demo_id}: {qa_path}")
            files = sorted((root / "replayed_demos").glob("*.hdf5"))
            if len(files) != 1:
                raise ValueError(f"expected one HDF5 for demo {demo_id}, found {len(files)}")
            source_hdf5 = files[0]
            if digest(source_hdf5) != qa.get("hdf5_sha256"):
                raise ValueError(f"post-QA HDF5 hash mismatch for demo {demo_id}")
            with h5py.File(source_hdf5, "r") as source:
                source_data = source["data"]
                names = list(source_data.keys())
                if len(names) != 1:
                    raise ValueError(f"expected one exported group for demo {demo_id}, found {names}")
                group = source_data[names[0]]
                actions = group["actions"]
                obs_keys = set(group["obs"].keys())
                if actions.ndim != 2 or actions.shape[1] != 13 or not REQUIRED_OBS.issubset(obs_keys):
                    raise ValueError(f"schema changed after QA for demo {demo_id}")
                if not bool(group.attrs.get("success", False)):
                    raise ValueError(f"demo {demo_id} is not marked successful")
                source.copy(group, target_data, name=f"demo_{demo_id}")
                steps = int(actions.shape[0])
                total += steps
                current_env_args = source_data.attrs.get("env_args")
                if env_args is None:
                    env_args = current_env_args
                elif current_env_args != env_args:
                    raise ValueError(f"env_args mismatch for demo {demo_id}")

            copied: dict[str, dict[str, object]] = {}
            media_paths = {
                "agentview_rgb": root / "video_datasets" / "libero_10_task5" / "videos" / f"demo_{demo_id}_agentview_rgb.mp4",
                "eye_in_hand_rgb": root / "video_datasets" / "libero_10_task5" / "videos" / f"demo_{demo_id}_eye_in_hand_rgb.mp4",
                "gsmini_left_tactile_rgb": root / "video_datasets" / "libero_10_task5" / "tactile_outputs" / f"demo_{demo_id}_gsmini_left_tactile_rgb.mp4",
                "gsmini_right_tactile_rgb": root / "video_datasets" / "libero_10_task5" / "tactile_outputs" / f"demo_{demo_id}_gsmini_right_tactile_rgb.mp4",
            }
            for label, source_media in media_paths.items():
                if not source_media.is_file():
                    raise FileNotFoundError(source_media)
                destination_dir = out_tactile_dir if "tactile" in label else out_video_dir
                destination = destination_dir / source_media.name
                shutil.copy2(source_media, destination)
                copied[label] = {"path": str(destination), "sha256": digest(destination)}
            records.append(
                {
                    "source_demo_id": demo_id,
                    "source_root": str(root),
                    "source_hdf5": str(source_hdf5),
                    "source_hdf5_sha256": digest(source_hdf5),
                    "steps": steps,
                    "copied_media": copied,
                }
            )

        if env_args is not None:
            target_data.attrs["env_args"] = env_args
        target_data.attrs["total"] = total

    manifest = {
        "protocol": "E3_TASK5_BENCHMARK_ONBOARDING_5DEMO_REAL_TACTILE_V1",
        "selected_demo_ids": list(EXPECTED_IDS),
        "demos": len(records),
        "total_steps": total,
        "output_hdf5": str(out_hdf5),
        "output_hdf5_sha256": digest(out_hdf5),
        "records": records,
        "sample_values_modified": False,
        "group_names_restored_to_frozen_source_ids": True,
        "gate_pass": True,
    }
    manifest_path = output_root / "E3_TASK5_ONBOARDING_5DEMO_ASSEMBLY_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
