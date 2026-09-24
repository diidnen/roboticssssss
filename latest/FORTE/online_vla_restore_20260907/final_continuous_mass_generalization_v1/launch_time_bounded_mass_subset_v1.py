#!/usr/bin/env python3
"""Deterministic, no-overwrite launcher for the frozen 216-branch subset."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from time_bounded_mass_common import HERE, load_frozen_subset, read, sha256, validate_branch


ISAAC = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
TABERO = Path("/home/exouser/Tabero")
STATUS_PATH = HERE / "TIME_BOUNDED_MASS_QUEUE_STATUS.json"


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def job_path(spec: dict) -> Path:
    return HERE / spec["relative_branch_path"]


def classify(spec: dict, runtime: dict) -> tuple[str, dict | None]:
    job = job_path(spec)
    if not job.exists():
        return "MISSING", None
    if not job.is_dir():
        return "CONFLICT", {"failed_checks": ["branch_path_not_directory"]}
    check = validate_branch(job, spec, runtime, HERE / "references" / spec["context_id"])
    if check["valid"]:
        return "VALID", check
    if (job / "LAUNCH_CLAIM.json").is_file() and not (job / "WORKER_COMPLETION.json").is_file():
        return "INCOMPLETE_CLAIM", check
    return "INVALID", check


def status(subset: dict, runtime: dict, subset_hash: str) -> dict:
    rows = []
    for spec in subset["jobs"]:
        classification, check = classify(spec, runtime)
        rows.append({
            "queue_index": spec["queue_index"], "split": spec["split"],
            "context_id": spec["context_id"], "force_N": spec["force_N"],
            "classification": classification,
            "failed_checks": [] if check is None else check.get("failed_checks", []),
        })
    counts = {name: sum(row["classification"] == name for row in rows)
              for name in ("VALID", "MISSING", "INCOMPLETE_CLAIM", "INVALID", "CONFLICT")}
    split_counts = {
        split: {name: sum(row["split"] == split and row["classification"] == name for row in rows)
                for name in counts}
        for split in ("TRAIN", "VAL", "HELDOUT")
    }
    result = {
        "schema": "TIME_BOUNDED_MASS_QUEUE_STATUS_V1",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "subset_sha256": subset_hash,
        "primary_target": 216,
        "counts": counts,
        "split_counts": split_counts,
        "complete": counts["VALID"] == 216 and all(counts[name] == 0 for name in counts if name != "VALID"),
        "automatic_expansion_to_648": False,
        "rows": rows,
        "launcher_source_sha256": sha256(Path(__file__)),
        "worker_source_sha256": sha256(HERE / "mass_training_worker.py"),
    }
    atomic_json(STATUS_PATH, result)
    return result


def environment(device: int) -> dict:
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(device), PYTHONNOUSERSITE="1",
        OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", TABERO_ROOT=str(TABERO),
        HDF5_TRAJ_SOURCE_DIR=str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        LIBERO_CONFIG_DIR=str(TABERO / "benchmarks/datasets/libero/config"),
        LIBERO_ASSETS_DATA_DIR=str(TABERO / "benchmarks/datasets/libero/USD"),
    )
    return env


def execute_one(spec: dict, protocol: dict, subset_hash: str, device: int) -> dict:
    job = job_path(spec)
    job.mkdir(parents=True, exist_ok=False)
    claim = {
        "context_id": spec["context_id"], "force": spec["force_N"],
        "runtime_manifest_sha256": protocol["runtime_manifest_sha256"],
    }
    with (job / "LAUNCH_CLAIM.json").open("x") as stream:
        json.dump(claim, stream, indent=2, sort_keys=True); stream.write("\n")
    command = [
        str(ISAAC), str(HERE / "mass_training_worker.py"), "--directory", str(HERE),
        "--context", spec["context_id"], "--force", str(spec["force_N"]),
    ]
    with (job / "WORKER_STDOUT.log").open("x") as out, (job / "WORKER_STDERR.log").open("x") as err:
        completed = subprocess.run(command, cwd=TABERO, env=environment(device), stdout=out, stderr=err, check=False)
    return {"spec": spec, "exit_code": completed.returncode, "path": str(job)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "launch"))
    parser.add_argument("--split", action="append", choices=("TRAIN", "VAL", "HELDOUT"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1,4]")

    subset, subset_hash = load_frozen_subset()
    runtime = read(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json")
    protocol = read(HERE / "MASS_TRAINING_PROTOCOL.json")
    current = status(subset, runtime, subset_hash)
    if args.command == "status":
        print(json.dumps({k: current[k] for k in ("as_of_utc", "primary_target", "counts", "split_counts", "complete")}, indent=2))
        return
    if current["counts"]["INCOMPLETE_CLAIM"] or current["counts"]["INVALID"] or current["counts"]["CONFLICT"]:
        raise RuntimeError("preflight found incomplete, invalid, or conflicting primary branch; see queue status")

    requested_splits = args.split or ["TRAIN", "VAL", "HELDOUT"]
    selected = []
    for spec in subset["jobs"]:
        if spec["split"] not in requested_splits:
            continue
        classification, _ = classify(spec, runtime)
        if classification == "MISSING":
            selected.append(spec)
        elif classification != "VALID":
            raise RuntimeError(f"non-reusable branch blocks launch: {job_path(spec)} ({classification})")

    for start in range(0, len(selected), args.workers):
        wave = selected[start:start + args.workers]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            results = list(pool.map(lambda spec: execute_one(spec, protocol, subset_hash, args.device), wave))
        failures = []
        for result in results:
            spec = result["spec"]
            classification, check = classify(spec, runtime)
            if result["exit_code"] != 0 or classification != "VALID":
                failures.append({"path": result["path"], "exit_code": result["exit_code"],
                                 "classification": classification,
                                 "failed_checks": [] if check is None else check.get("failed_checks", [])})
            else:
                print(json.dumps({"completed": result["path"], "queue_index": spec["queue_index"],
                                  "split": spec["split"]}), flush=True)
        current = status(subset, runtime, subset_hash)
        if failures:
            print(json.dumps({"failures": failures}, indent=2), flush=True)
            raise SystemExit(2)
    print(json.dumps({k: current[k] for k in ("as_of_utc", "primary_target", "counts", "split_counts", "complete")}, indent=2))


if __name__ == "__main__":
    main()
