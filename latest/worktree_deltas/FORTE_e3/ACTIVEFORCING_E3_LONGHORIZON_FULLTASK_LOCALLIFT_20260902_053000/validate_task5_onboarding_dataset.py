#!/usr/bin/env python3
"""Fail-closed data-quality gate for the final five-demo 7dpf LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


EXPECTED_PROMPT = "pick up the book and place it in the back compartment of the caddy"
EXPECTED_DEMOS = [1, 2, 11, 12, 19]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonlines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembled", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    assembled = args.assembled.resolve()
    dataset = args.dataset.resolve()
    errors: list[str] = []

    manifest_path = assembled / "ASSEMBLED_5DEMO_MANIFEST.json"
    if not manifest_path.is_file():
        errors.append("missing assembly manifest")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text())
    if manifest.get("selected_demo_ids") != EXPECTED_DEMOS:
        errors.append("assembly demo identity/order mismatch")
    if manifest.get("activeforcing_or_test_outcome_used") is not False:
        errors.append("assembly outcome-independence assertion missing")

    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    if int(info.get("total_episodes", -1)) != 5:
        errors.append(f"expected 5 episodes, got {info.get('total_episodes')}")
    features = info.get("features", {})
    expected_shapes = {
        "state": [7],
        "actions": [13],
        "tactile_gripper_force": [8, 6],
        "tactile_marker_motion": [9, 198, 2],
    }
    for key, shape in expected_shapes.items():
        got = features.get(key, {}).get("shape")
        if got != shape:
            errors.append(f"feature {key} shape {got}, expected {shape}")
    for key in ("image", "wrist_image", "tactile_image"):
        if features.get(key, {}).get("shape") != [224, 224, 3]:
            errors.append(f"feature {key} is not 224x224x3")

    tasks_path = dataset / "meta" / "tasks.jsonl"
    tasks = jsonlines(tasks_path) if tasks_path.is_file() else []
    if len(tasks) != 1 or tasks[0].get("task") != EXPECTED_PROMPT:
        errors.append(f"task prompt mismatch: {tasks}")
    episodes_path = dataset / "meta" / "episodes.jsonl"
    episodes = jsonlines(episodes_path) if episodes_path.is_file() else []
    if len(episodes) != 5:
        errors.append(f"episode metadata count {len(episodes)}, expected 5")

    parquet_files = sorted((dataset / "data").rglob("*.parquet"))
    row_count = 0
    episode_ids: set[int] = set()
    parquet_hashes = []
    for path in parquet_files:
        table = pq.read_table(path, columns=["episode_index", "actions", "tactile_marker_motion"])
        row_count += table.num_rows
        episode_ids.update(int(value.as_py()) for value in table["episode_index"])
        for value in table["actions"]:
            if np.asarray(value.as_py()).shape != (13,):
                errors.append(f"non-13D action in {path.name}")
                break
        for value in table["tactile_marker_motion"]:
            if np.asarray(value.as_py()).shape not in ((9, 198, 2), (9 * 198 * 2,)):
                errors.append(f"bad tactile_marker_motion shape in {path.name}")
                break
        parquet_hashes.append({"path": str(path), "sha256": sha256(path), "rows": table.num_rows})
    if len(parquet_files) != 5:
        errors.append(f"parquet episode files {len(parquet_files)}, expected 5")
    if episode_ids != set(range(5)):
        errors.append(f"episode indices {sorted(episode_ids)}, expected 0..4")
    if row_count != int(info.get("total_frames", -1)):
        errors.append(f"parquet rows {row_count} != info total_frames {info.get('total_frames')}")
    if row_count != int(manifest.get("total_samples", -1)):
        errors.append(f"parquet rows {row_count} != assembly total_samples {manifest.get('total_samples')}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "gate": "E3_TASK5_5DEMO_7DPF_DATASET_QUALITY",
        "assembled": str(assembled),
        "dataset": str(dataset),
        "episodes": len(episodes),
        "frames": row_count,
        "episode_indices": sorted(episode_ids),
        "selected_demo_ids": EXPECTED_DEMOS,
        "parquet": parquet_hashes,
        "errors": errors,
    }
    (dataset / "E3_TASK5_5DEMO_DATASET_QA.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
