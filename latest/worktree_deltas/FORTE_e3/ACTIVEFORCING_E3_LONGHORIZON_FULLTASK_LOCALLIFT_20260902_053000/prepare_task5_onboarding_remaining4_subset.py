#!/usr/bin/env python3
"""Build the immutable TRAIN-only task5 input subset for the remaining replay batch.

The script copies source groups losslessly.  It does not alter actions, initial
states, labels, or source IDs, and it refuses to overwrite an existing output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


SOURCE = Path(
    "/media/volume/newdata/exouser/activeforcing_e3/PROJECT_LIBERO_TASK5_DEMOS_20260902/"
    "assembled_hdf5/libero_10_task5_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_"
    "back_compartment_of_the_caddy_demo.hdf5"
)
SOURCE_SHA256 = "98fb9f3fab6b46a0fbb6d63fc686dccfbb31a372c825c0c87da8fe21f99a6219"
SELECTED_IDS = (2, 11, 12, 19)
EXPECTED_STEPS = {2: 180, 11: 200, 12: 169, 19: 165}
OUTPUT_NAME = SOURCE.name


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
    """Hash names, attributes, dtype/shape, and uncompressed dataset bytes."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_hdf5_dir = output_root / "assembled_hdf5"
    output_hdf5 = output_hdf5_dir / OUTPUT_NAME
    manifest_path = output_root / "REMAINING4_INPUT_MANIFEST.json"

    if output_root.exists():
        raise RuntimeError(f"refuse to reuse output root: {output_root}")
    if file_sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("authoritative source HDF5 hash mismatch")
    if 1 in SELECTED_IDS or tuple(sorted(SELECTED_IDS)) != SELECTED_IDS:
        raise RuntimeError("frozen remaining-ID set is invalid")

    output_hdf5_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    with h5py.File(SOURCE, "r") as source, h5py.File(output_hdf5, "x") as target:
        if set(source.keys()) != {"data"}:
            raise RuntimeError(f"unexpected source root groups: {sorted(source.keys())}")
        source_data = source["data"]
        target_data = target.create_group("data")
        for key, value in source.attrs.items():
            target.attrs[key] = value
        for key, value in source_data.attrs.items():
            target_data.attrs[key] = value

        for demo_id in SELECTED_IDS:
            group_name = f"demo_{demo_id}"
            if group_name not in source_data:
                raise RuntimeError(f"missing frozen source group {group_name}")
            source_group = source_data[group_name]
            actions = source_group.get("actions")
            if actions is None or actions.shape != (EXPECTED_STEPS[demo_id], 8):
                raise RuntimeError(
                    f"{group_name}: expected actions {(EXPECTED_STEPS[demo_id], 8)}, "
                    f"got {None if actions is None else actions.shape}"
                )
            before = tree_digest(source_group)
            source.copy(source_group, target_data, name=group_name)
            after = tree_digest(target_data[group_name])
            if before != after:
                raise RuntimeError(f"{group_name}: lossless-copy digest mismatch")
            records.append(
                {
                    "source_demo_id": demo_id,
                    "steps": EXPECTED_STEPS[demo_id],
                    "source_group_digest": before,
                    "subset_group_digest": after,
                }
            )

        if sorted(target_data.keys()) != ["demo_11", "demo_12", "demo_19", "demo_2"]:
            raise RuntimeError(f"unexpected subset groups: {sorted(target_data.keys())}")

    manifest = {
        "status": "REMAINING4_TRAIN_INPUT_SUBSET_FROZEN",
        "split": "TRAIN",
        "task": "libero_10/task5",
        "instruction": "pick up the book and place it in the back compartment of the caddy",
        "selected_demo_ids": list(SELECTED_IDS),
        "excluded_smoke_demo_id": 1,
        "total_steps": sum(EXPECTED_STEPS.values()),
        "source_hdf5": str(SOURCE),
        "source_hdf5_sha256": SOURCE_SHA256,
        "subset_hdf5": str(output_hdf5),
        "subset_hdf5_sha256": file_sha256(output_hdf5),
        "records": records,
        "activeforcing_or_test_outcome_used": False,
        "utility_or_fmax_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
