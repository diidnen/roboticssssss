#!/usr/bin/env python3
"""No-overwrite launcher/status tool for frozen MASS acquisition."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ISAAC_PYTHON = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
TABERO = Path("/home/exouser/Tabero")


def load(path):
    return json.loads(Path(path).read_text())


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inputs():
    manifest = load(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json")
    protocol = load(HERE / "MASS_TRAINING_PROTOCOL.json")
    plan = load(HERE / "MASS_TRAINING_CONTEXT_PLAN.json")
    return manifest, protocol, plan


def jobs(stage, split=None):
    manifest, protocol, plan = inputs()
    contexts = [c for c in plan["contexts"] if split is None or c["split"] == split]
    if stage == "probes":
        return [(c, None, HERE / "references" / c["id"]) for c in contexts]
    if stage == "branches":
        qualification = HERE / "MASS_BELIEF_QUALIFICATION.json"
        if not qualification.is_file() or not load(qualification).get("qualified"):
            raise RuntimeError("feasibility physics prohibited until MASS_BELIEF_QUALIFICATION.json says PASS")
        return [(c, force, HERE / "branches" / c["id"] / f"F{force:g}")
                for c in contexts for force in manifest["candidate_forces_N"]]
    raise ValueError(stage)


def valid(stage, job):
    completion = job / "WORKER_COMPLETION.json"
    if not completion.is_file() or not load(completion).get("logical_success"):
        return False
    if stage == "probes":
        required = ("RAW_PROBE.csv", "RAW_FEATURES.npy", "DECISION_STATE.pt", "PREACTION_SEQUENCE.npy", "MASS_INTERVENTION_READBACK.json")
    else:
        required = ("BRANCH_RESULT.json", "PREACTION_SEQUENCE.npy", "MASS_INTERVENTION_READBACK.json")
    return all((job / name).is_file() for name in required)


def status(stage, split):
    planned = jobs(stage, split)
    completed = [str(job) for _, _, job in planned if valid(stage, job)]
    failed = [str(job) for _, _, job in planned if (job / "WORKER_COMPLETION.json").is_file() and not valid(stage, job)]
    unstarted = [str(job) for _, _, job in planned if not (job / "LAUNCH_CLAIM.json").exists()]
    claimed_incomplete = [str(job) for _, _, job in planned if (job / "LAUNCH_CLAIM.json").exists() and not (job / "WORKER_COMPLETION.json").exists()]
    return {"stage": stage, "split": split, "planned": len(planned), "valid": len(completed),
            "failed": failed, "unstarted": len(unstarted), "claimed_incomplete": claimed_incomplete}


def launch(stage, split, device, limit):
    manifest, protocol, _ = inputs()
    selected = []
    for context, force, job in jobs(stage, split):
        if valid(stage, job):
            continue
        if (job / "LAUNCH_CLAIM.json").exists():
            raise RuntimeError(f"claimed job is incomplete; forensic review required: {job}")
        selected.append((context, force, job))
        if limit is not None and len(selected) >= limit:
            break
    for context, force, job in selected:
        job.mkdir(parents=True, exist_ok=False)
        dump_new(job / "LAUNCH_CLAIM.json", {
            "context_id": context["id"], "force": force,
            "runtime_manifest_sha256": protocol["runtime_manifest_sha256"],
        })
        cmd = [str(ISAAC_PYTHON), str(HERE / "mass_training_worker.py"),
               "--directory", str(HERE), "--context", context["id"]]
        if force is not None:
            cmd += ["--force", str(force)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        env.update(
            PYTHONNOUSERSITE="1", OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y",
            TABERO_ROOT=str(TABERO),
            HDF5_TRAJ_SOURCE_DIR=str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
            LIBERO_CONFIG_DIR=str(TABERO / "benchmarks/datasets/libero/config"),
            LIBERO_ASSETS_DATA_DIR=str(TABERO / "benchmarks/datasets/libero/USD"),
        )
        with (job / "WORKER_STDOUT.log").open("x") as out, (job / "WORKER_STDERR.log").open("x") as err:
            completed = subprocess.run(cmd, cwd=TABERO, stdout=out, stderr=err, env=env, check=False)
        if completed.returncode != 0:
            print(json.dumps({"job": str(job), "exit_code": completed.returncode}), flush=True)
            raise SystemExit(completed.returncode)
        if not valid(stage, job):
            raise RuntimeError("worker returned zero without valid evidence: " + str(job))
        print(json.dumps({"completed": str(job), "stage": stage}), flush=True)
    print(json.dumps(status(stage, split), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "launch"))
    parser.add_argument("stage", choices=("probes", "branches"))
    parser.add_argument("--split", choices=("TRAIN", "VAL", "HELDOUT"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(args.stage, args.split), indent=2))
    else:
        launch(args.stage, args.split, args.device, args.limit)


if __name__ == "__main__":
    main()
