#!/usr/bin/env python3
"""Resumable one-worker queue for the established-grasp no-query V2 run."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PYTHON = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WORKER = Path("/home/exouser/noquery_established_v2_worker.py")
PLAN = Path(
    "/media/volume/newdata/exouser/online_vla_activeforcing_20260907/"
    "final_ablation_confirmatory_v1/FINAL_ABLATION_CONTEXT_PLAN.json"
)
FINAL_SNAPSHOT = Path(
    "/media/volume/newdata/exouser/online_vla_activeforcing_20260907/"
    "final_ablation_confirmatory_v1/SOURCE_SNAPSHOT"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def server_ready(port: int) -> None:
    with socket.create_connection(("127.0.0.1", port), timeout=5):
        pass


def main(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    contexts = json.loads(PLAN.read_text())["contexts"]
    if len(contexts) != 24:
        raise RuntimeError("expected frozen 24-context plan")
    server_ready(args.port)

    env = os.environ.copy()
    env.update(
        PYTHONNOUSERSITE="1",
        OMNI_KIT_ACCEPT_EULA="YES",
        ACCEPT_EULA="Y",
        TABERO_ROOT="/home/exouser/Tabero",
        HDF5_TRAJ_SOURCE_DIR="/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5",
        LIBERO_CONFIG_DIR="/home/exouser/Tabero/benchmarks/datasets/libero/config",
        LIBERO_ASSETS_DATA_DIR="/home/exouser/Tabero/benchmarks/datasets/libero/USD",
        PYTHONPATH=os.pathsep.join(
            [
                "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64",
                str(FINAL_SNAPSHOT),
                "/home/exouser/FORTE",
                "/home/exouser/Tabero",
                "/home/exouser/Tabero/benchmarks/openpi/openpi-client/src",
            ]
        ),
    )
    status_path = out / "QUEUE_STATUS.json"
    for index, context in enumerate(contexts):
        job_name = context["id"]
        completion = out / "branches" / job_name / "WORKER_COMPLETION.json"
        if completion.exists() and json.loads(completion.read_text()).get("logical_success"):
            continue
        state = {
            "status": "RUNNING",
            "started_or_resumed_utc": now(),
            "current_index": index,
            "current_context": context["id"],
            "completed": sum(
                1
                for item in contexts
                if (out / "branches" / item["id"] / "WORKER_COMPLETION.json").exists()
            ),
            "planned": len(contexts),
        }
        write_status(status_path, state)
        command = [
            str(PYTHON),
            "-u",
            str(WORKER),
            "--out",
            str(out),
            "--plan",
            str(PLAN),
            "--context",
            str(index),
            "--job-name",
            job_name,
            "--port",
            str(args.port),
        ]
        with (logs / f"{index:02d}_{job_name}.log").open("a") as handle:
            result = subprocess.run(command, cwd="/home/exouser/Tabero", env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            state.update(status="FAILED", failed_index=index, exit_code=result.returncode, failed_utc=now())
            write_status(status_path, state)
            return result.returncode

    rows = []
    for context in contexts:
        job = out / "branches" / context["id"]
        branch = json.loads((job / "BRANCH_RESULT.json").read_text())
        decision = json.loads((job / "PLANNER_DECISION.json").read_text())
        rows.append(
            {
                "context": context["id"],
                "task": context["task"],
                "root": context["root"],
                "band": context["band"],
                "selected_force_N": decision["selected_force_N"],
                "full_task_success_y": branch["outcome"]["full_task_success_y"],
                "lift_success": branch["outcome"]["lift_success"],
                "dropped": branch["outcome"]["dropped"],
                "measured_squeeze_N": branch["outcome"]["mean_measured_bilateral_squeeze"],
                "rpc_count": branch["rpc_count"],
                "online_vla_verified": branch["online_vla_verified"],
            }
        )
    (out / "RESULTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_status(
        status_path,
        {
            "status": "COMPLETE",
            "completed_utc": now(),
            "completed": len(rows),
            "planned": len(rows),
            "full_task_successes": sum(row["full_task_success_y"] for row in rows),
            "all_online_vla_verified": all(row["online_vla_verified"] for row in rows),
        },
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/exouser/noquery_established_v2_full_20260910")
    parser.add_argument("--port", type=int, default=18885)
    raise SystemExit(main(parser.parse_args()))
