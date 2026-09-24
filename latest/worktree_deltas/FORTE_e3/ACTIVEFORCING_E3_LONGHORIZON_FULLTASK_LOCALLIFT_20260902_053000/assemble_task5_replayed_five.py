#!/usr/bin/env python3
"""Fail-closed assembly of the five preregistered, independently replayed demos."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import h5py


DEMO_ROOTS = {
    1: Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_105400_RETRY1/smoke_demo1"),
    2: Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_110000/demo2"),
    11: Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_110000/demo11"),
    12: Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_110000/demo12"),
    19: Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_110000/demo19"),
}
OUT = Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_ASSEMBLED_20260902_113000")
H5_NAME = "libero_10_task5_book_caddy_onboarding_5demo_7dpf_demo.hdf5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refuse to overwrite assembled output: {OUT}")

    sources: dict[int, tuple[Path, dict]] = {}
    for demo_id, root in DEMO_ROOTS.items():
        qa_path = root / "E3_TASK5_ONBOARDING_REPLAY_QA.json"
        if not qa_path.is_file():
            raise RuntimeError(f"demo {demo_id}: missing QA JSON")
        qa = json.loads(qa_path.read_text())
        if qa.get("gate_pass") is not True or qa.get("errors"):
            raise RuntimeError(f"demo {demo_id}: QA did not pass")
        if int(qa.get("source_demo_id", 1)) != demo_id:
            raise RuntimeError(f"demo {demo_id}: QA source identity mismatch")
        h5s = sorted((root / "replayed_demos").glob("*.hdf5"))
        if len(h5s) != 1:
            raise RuntimeError(f"demo {demo_id}: expected one HDF5, got {len(h5s)}")
        sources[demo_id] = (h5s[0], qa)

    replayed = OUT / "replayed_demos"
    video_base = OUT / "video_datasets" / "libero_10_task5"
    cameras = video_base / "videos"
    tactile = video_base / "tactile_outputs"
    replayed.mkdir(parents=True)
    cameras.mkdir(parents=True)
    tactile.mkdir(parents=True)
    out_h5 = replayed / H5_NAME
    records = []
    total_samples = 0

    with h5py.File(out_h5, "w") as dst:
        dst_data = dst.create_group("data")
        for demo_id, (src_path, qa) in sources.items():
            with h5py.File(src_path, "r") as src:
                if demo_id == 1:
                    for key, value in src.attrs.items():
                        dst.attrs[key] = value
                    for key, value in src["data"].attrs.items():
                        dst_data.attrs[key] = value
                groups = sorted(src["data"].keys())
                if len(groups) != 1:
                    raise RuntimeError(f"demo {demo_id}: expected one exported group")
                src.copy(src["data"][groups[0]], dst_data, name=f"demo_{demo_id}")
                samples = int(src["data"][groups[0]].attrs.get("num_samples", 0))
                if samples <= 0:
                    raise RuntimeError(f"demo {demo_id}: invalid num_samples={samples}")
                total_samples += samples

            media_records = []
            for kind, target_dir, names in (
                ("camera", cameras, [f"demo_{demo_id}_agentview_rgb.mp4", f"demo_{demo_id}_eye_in_hand_rgb.mp4"]),
                ("tactile", tactile, [f"demo_{demo_id}_gsmini_left_tactile_rgb.mp4", f"demo_{demo_id}_gsmini_right_tactile_rgb.mp4"]),
            ):
                source_dir = DEMO_ROOTS[demo_id] / "video_datasets" / "libero_10_task5" / ("videos" if kind == "camera" else "tactile_outputs")
                for name in names:
                    src_media = source_dir / name
                    if not src_media.is_file():
                        raise RuntimeError(f"demo {demo_id}: missing {src_media}")
                    dst_media = target_dir / name
                    shutil.copy2(src_media, dst_media)
                    media_records.append({"path": str(dst_media), "sha256": sha256(dst_media)})
            records.append({
                "demo_id": demo_id,
                "source_hdf5": str(src_path),
                "source_hdf5_sha256": sha256(src_path),
                "qa": str(DEMO_ROOTS[demo_id] / "E3_TASK5_ONBOARDING_REPLAY_QA.json"),
                "media": media_records,
            })
        dst_data.attrs["total"] = total_samples

    manifest = {
        "protocol": "E3_TASK5_BENCHMARK_ONBOARDING_5DEMO_7DPF_ASSEMBLY_V1",
        "selected_demo_ids": list(DEMO_ROOTS),
        "activeforcing_or_test_outcome_used": False,
        "assembled_hdf5": str(out_h5),
        "assembled_hdf5_sha256": sha256(out_h5),
        "total_samples": total_samples,
        "records": records,
    }
    (OUT / "ASSEMBLED_5DEMO_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
