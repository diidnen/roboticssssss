#!/usr/bin/env python3
"""Root-partitioned scheduler for frozen matched MASS force branches."""
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
    if not (job / "WORKER_COMPLETION.json").is_file() or not read(job / "WORKER_COMPLETION.json").get("logical_success"): return False
    return (job / "BRANCH_RESULT.json").is_file() and read(job / "BRANCH_RESULT.json").get("passed") is True


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=int, required=True)
    args = parser.parse_args(); protocol = read(HERE / "MASS_TRAINING_PROTOCOL.json")
    manifest = read(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"); plan = read(HERE / "MASS_TRAINING_CONTEXT_PLAN.json")
    if not read(HERE / "MASS_BELIEF_QUALIFICATION.json").get("qualified"): raise RuntimeError("belief gate failed")
    contexts = [row for row in plan["contexts"] if row["root"] == args.root]
    if len(contexts) != 12: raise RuntimeError("root is not one complete frozen partition")
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES="0", PYTHONNOUSERSITE="1", OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y",
        TABERO_ROOT=str(TABERO), HDF5_TRAJ_SOURCE_DIR=str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        LIBERO_CONFIG_DIR=str(TABERO / "benchmarks/datasets/libero/config"), LIBERO_ASSETS_DATA_DIR=str(TABERO / "benchmarks/datasets/libero/USD"))
    for context in contexts:
        for force in manifest["candidate_forces_N"]:
            job = HERE / "branches" / context["id"] / f"F{force:g}"
            if valid(job): continue
            if (job / "LAUNCH_CLAIM.json").exists(): raise RuntimeError("partition collision/incomplete branch: " + str(job))
            job.mkdir(parents=True, exist_ok=False)
            write(job / "LAUNCH_CLAIM.json", {"context_id": context["id"], "force": force,
                  "runtime_manifest_sha256": protocol["runtime_manifest_sha256"]})
            command = [str(ISAAC), str(HERE / "mass_training_worker.py"), "--directory", str(HERE),
                       "--context", context["id"], "--force", str(force)]
            with (job / "WORKER_STDOUT.log").open("x") as out, (job / "WORKER_STDERR.log").open("x") as err:
                result = subprocess.run(command, cwd=TABERO, env=env, stdout=out, stderr=err, check=False)
            if result.returncode or not valid(job): raise RuntimeError("frozen matched branch failed: " + str(job))
            print(json.dumps({"completed": str(job), "root_partition": args.root, "force_N": force}), flush=True)


if __name__ == "__main__": main()
