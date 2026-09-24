#!/usr/bin/env python3
"""Root-partitioned orchestration for the already-frozen MASS worker.

This changes no physics/runtime source.  It exists only to schedule disjoint
planned roots concurrently and writes the identical exclusive launch claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
TABERO = Path("/home/exouser/Tabero")
ISAAC = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream: json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")


def valid(job):
    return ((job / "WORKER_COMPLETION.json").is_file() and read(job / "WORKER_COMPLETION.json").get("logical_success") and
            all((job / name).is_file() for name in ("RAW_PROBE.csv", "RAW_FEATURES.npy", "DECISION_STATE.pt", "PREACTION_SEQUENCE.npy")))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=int, required=True)
    args = parser.parse_args(); protocol = read(HERE / "MASS_TRAINING_PROTOCOL.json")
    plan = read(HERE / "MASS_TRAINING_CONTEXT_PLAN.json")
    contexts = [row for row in plan["contexts"] if row["split"] == "TRAIN" and row["root"] == args.root]
    if len(contexts) != 12: raise RuntimeError("root is not one complete frozen TRAIN partition")
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES="0", PYTHONNOUSERSITE="1", OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y",
        TABERO_ROOT=str(TABERO), HDF5_TRAJ_SOURCE_DIR=str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        LIBERO_CONFIG_DIR=str(TABERO / "benchmarks/datasets/libero/config"), LIBERO_ASSETS_DATA_DIR=str(TABERO / "benchmarks/datasets/libero/USD"))
    for context in contexts:
        job = HERE / "references" / context["id"]
        if valid(job): continue
        if (job / "LAUNCH_CLAIM.json").exists(): raise RuntimeError("partition collision/incomplete claim: " + str(job))
        job.mkdir(parents=True, exist_ok=False)
        write(job / "LAUNCH_CLAIM.json", {"context_id": context["id"], "force": None,
              "runtime_manifest_sha256": protocol["runtime_manifest_sha256"]})
        command = [str(ISAAC), str(HERE / "mass_training_worker.py"), "--directory", str(HERE), "--context", context["id"]]
        with (job / "WORKER_STDOUT.log").open("x") as out, (job / "WORKER_STDERR.log").open("x") as err:
            result = subprocess.run(command, cwd=TABERO, env=env, stdout=out, stderr=err, check=False)
        if result.returncode or not valid(job): raise RuntimeError("frozen worker failed: " + str(job))
        print(json.dumps({"completed": str(job), "root_partition": args.root}), flush=True)


if __name__ == "__main__": main()
