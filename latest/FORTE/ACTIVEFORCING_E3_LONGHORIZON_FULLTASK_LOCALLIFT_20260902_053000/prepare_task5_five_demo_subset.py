#!/usr/bin/env python3
"""Build the frozen five-demo E3 onboarding subset from official replay data."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py


SELECTED = (1, 2, 11, 12, 19)
TASK_STEM = "libero_10_task5_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing existing output root: {args.output_root}")

    src_h5 = args.source_root / "replayed_demos" / f"{TASK_STEM}.hdf5"
    src_videos = args.source_root / "video_datasets" / "libero_10_task5" / "videos"
    out_h5_dir = args.output_root / "replayed_demos"
    out_video_dir = args.output_root / "video_datasets" / "libero_10_task5" / "videos"
    out_h5_dir.mkdir(parents=True)
    out_video_dir.mkdir(parents=True)
    out_h5 = out_h5_dir / f"{TASK_STEM}.hdf5"

    records = []
    with h5py.File(src_h5, "r") as source, h5py.File(out_h5, "w") as target:
        source_data = source["data"]
        target_data = target.create_group("data")
        for key, value in source_data.attrs.items():
            target_data.attrs[key] = value
        total = 0
        available = sorted(int(name.split("_", 1)[1]) for name in source_data if name.startswith("demo_"))
        if any(demo_id not in available for demo_id in SELECTED):
            raise RuntimeError(f"selected IDs missing; available={available}")
        for demo_id in SELECTED:
            name = f"demo_{demo_id}"
            source.copy(source_data[name], target_data, name=name)
            samples = int(source_data[name].attrs.get("num_samples", len(source_data[name]["actions"])))
            total += samples
            copied_videos = []
            for view in ("agentview_rgb", "eye_in_hand_rgb"):
                src_video = src_videos / f"demo_{demo_id}_{view}.mp4"
                if not src_video.is_file():
                    raise FileNotFoundError(src_video)
                dst_video = out_video_dir / src_video.name
                shutil.copy2(src_video, dst_video)
                copied_videos.append({"path": str(dst_video), "sha256": sha256(dst_video)})
            records.append({"demo_id": demo_id, "samples": samples, "videos": copied_videos})
        target_data.attrs["total"] = total

    manifest = {
        "protocol": "E3_TASK5_BENCHMARK_ONBOARDING_5DEMO_V1",
        "task_identity": "libero_10/task5:black_book_1_to_caddy",
        "selection_rule": "first five successful demo IDs in ascending numeric order",
        "selected_demo_ids": list(SELECTED),
        "demos": len(SELECTED),
        "total_samples": sum(record["samples"] for record in records),
        "source_hdf5": str(src_h5.resolve()),
        "source_hdf5_sha256": sha256(src_h5),
        "subset_hdf5": str(out_h5.resolve()),
        "subset_hdf5_sha256": sha256(out_h5),
        "records": records,
        "activeforcing_or_test_outcome_used": False,
    }
    manifest_path = args.output_root / "FIVE_DEMO_SUBSET_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
