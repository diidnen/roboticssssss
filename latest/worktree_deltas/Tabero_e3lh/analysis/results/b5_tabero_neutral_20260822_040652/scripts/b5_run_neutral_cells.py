#!/usr/bin/env python3
"""Run B5 Tabero-neutral cells without modifying Tabero core code."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


RESULT_DIR = Path("/home/exouser/Tabero/analysis/results/b5_tabero_neutral_20260822_040652")
TABERO_ROOT = Path("/home/exouser/Tabero")
CLIENT = RESULT_DIR / "scripts" / "b5_tabero_neutral_client.py"
PYTHON = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path(
    "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.warp.core-1.8.2+lx64"
)
OPENPI_CLIENT_SRC = TABERO_ROOT / "benchmarks" / "openpi" / "openpi-client" / "src"
HDF5_DIR = TABERO_ROOT / "benchmarks" / "datasets" / "libero" / "assembled_hdf5"
CONFIG_DIR = TABERO_ROOT / "benchmarks" / "datasets" / "libero" / "config"
ASSETS_DIR = TABERO_ROOT / "benchmarks" / "datasets" / "libero" / "USD"


def episode_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        return max(0, sum(1 for _ in csv.DictReader(f)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", type=int, default=[0, 1, 2, 5, 6])
    parser.add_argument("--frictions", nargs="+", type=float, default=[0.2, 0.5, 1.0])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-inference-steps", type=int, default=50)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=18019)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    log_dir = RESULT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{WARP_CORE}:{TABERO_ROOT}:{OPENPI_CLIENT_SRC}",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "TABERO_ROOT": str(TABERO_ROOT),
            "HDF5_TRAJ_SOURCE_DIR": str(HDF5_DIR),
            "LIBERO_CONFIG_DIR": str(CONFIG_DIR),
            "LIBERO_ASSETS_DATA_DIR": str(ASSETS_DIR),
        }
    )

    failures: list[str] = []
    for task_id in args.tasks:
        for mu in args.frictions:
            tag = f"task{task_id}_mu{mu:g}"
            episode_csv = log_dir / f"{tag}_episodes.csv"
            if args.resume and episode_count(episode_csv) >= args.episodes:
                print(f"[B5-RUNNER] skip existing {tag}", flush=True)
                continue

            out_log = log_dir / f"{tag}_stdout.log"
            err_log = log_dir / f"{tag}_stderr.log"
            cmd = [
                str(PYTHON),
                "-u",
                str(CLIENT),
                "--control_mode",
                "tactile",
                "--task_suite",
                "libero_object",
                "--task_id",
                str(task_id),
                "--server_host",
                args.server_host,
                "--server_port",
                str(args.server_port),
                "--num_total_experiments",
                str(args.episodes),
                "--max_inference_steps",
                str(args.max_inference_steps),
                "--replan_steps",
                "10",
                "--num_success_steps",
                "8",
                "--seed",
                "0",
                "--hdf5_folder",
                str(HDF5_DIR),
                "--headless",
                "--b5_output_dir",
                str(RESULT_DIR),
                "--b5_friction",
                str(mu),
            ]
            print(f"[B5-RUNNER] start {tag} episodes={args.episodes}", flush=True)
            with out_log.open("w") as stdout_f, err_log.open("w") as stderr_f:
                proc = subprocess.run(cmd, cwd=TABERO_ROOT, env=env, stdout=stdout_f, stderr=stderr_f)
            count = episode_count(episode_csv)
            print(f"[B5-RUNNER] done {tag} returncode={proc.returncode} episodes_written={count}", flush=True)
            if proc.returncode != 0 or count < args.episodes:
                failures.append(f"{tag}: returncode={proc.returncode}, episodes_written={count}")

    if failures:
        print("[B5-RUNNER] failures:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
