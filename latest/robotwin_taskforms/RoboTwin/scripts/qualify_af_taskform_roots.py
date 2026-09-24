#!/usr/bin/env python3
"""Freeze exact roots that support the 2 mm query at every friction level."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


TASKS = ("handover_mic", "dump_bin_bigbin")
MUS = (0.55, 0.25, 0.85)
CHECKPOINT = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "checkpoints/pi0_robotwin_30000/30000"
)
DEFAULT_OUT = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "experiments/af_taskforms_216_v1/root_map.json"
)


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


def run_query(repo: Path, task: str, slot: int, seed: int, mu: float, expert: bool):
    evaluator = repo / "scripts/eval_policy_xpolicylab.py"
    additional = ",".join(
        [
            f"ckpt_name={CHECKPOINT}",
            "action_type=joint",
            "activeforcing_enabled=true",
            "af_dynamic_evaluator=false",
            "af_supplied_grasp=true",
            "af_query_enabled=true",
            "af_query_only=true",
            "af_query_force_n=4",
            "af_query_displacement_m=0.002",
            f"af_contact_friction={mu}",
            "af_force_limit_n=4",
            f"start_seed={seed}",
            "strict_seed=true",
        ]
    )
    command = [
        sys.executable,
        str(evaluator),
        "--task_name",
        task,
        "--env_cfg_type",
        "arx_x5",
        "--policy_name",
        "Pi_0",
        "--host",
        "localhost",
        "--port",
        "6001",
        "--protocol",
        "ws",
        "--seed",
        str(slot),
        "--test_num",
        "1",
        "--expert_check",
        str(expert).lower(),
        "--frequency",
        "30",
        "--additional_info",
        additional,
    ]
    process = subprocess.Popen(
        command,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "ROBOTWIN_SUPPRESS_EVAL_CONFIG": "1"},
    )
    evidence = None
    tail = []
    assert process.stdout is not None
    for line in process.stdout:
        tail.append(line.rstrip())
        tail = tail[-12:]
        marker = "ACTIVEFORCING_EVIDENCE "
        if line.startswith(marker):
            evidence = json.loads(line[len(marker) :])
        elif "strict" in line.lower() or "error" in line.lower():
            print(line, end="", flush=True)
    returncode = process.wait()
    valid = bool(
        returncode == 0
        and evidence
        and evidence["seed"] == seed
        and evidence["success"]
        and evidence["query_info"]["final_bilateral_contact"]
        and evidence["query_info"]["contact_ratio"] >= 0.70
        and evidence["query_info"]["ee_return_error_m"] <= 0.001
    )
    return valid, evidence, tail, returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    data = (
        json.loads(args.out.read_text())
        if args.out.exists()
        else {
            "schema_id": "AF_ROBOTWIN_EXACT_ROOT_MAP_V1",
            "frictions_qualified": sorted(MUS),
            "roots": {},
        }
    )
    new_count = 0
    for task in TASKS:
        for slot in range(12):
            key = f"{task}|slot={slot}"
            if key in data["roots"]:
                continue
            if args.max_new and new_count >= args.max_new:
                return 0
            base = 100000 * (slot + 1)
            accepted = None
            for offset in range(50):
                seed = base + offset
                print(f"ROOT_CANDIDATE task={task} slot={slot} seed={seed}", flush=True)
                checks = []
                failed = False
                for index, mu in enumerate(MUS):
                    valid, evidence, tail, returncode = run_query(
                        repo, task, slot, seed, mu, expert=(index == 0)
                    )
                    checks.append(
                        {
                            "friction": mu,
                            "valid": valid,
                            "returncode": returncode,
                            "query_summary": None
                            if evidence is None
                            else {
                                key: evidence["query_info"][key]
                                for key in (
                                    "samples",
                                    "contact_ratio",
                                    "final_bilateral_contact",
                                    "ee_return_error_m",
                                    "actor_return_error_m",
                                    "relative_slip_path_m",
                                )
                            },
                        }
                    )
                    print(
                        f"ROOT_QUERY task={task} slot={slot} seed={seed} "
                        f"mu={mu} valid={valid}",
                        flush=True,
                    )
                    if not valid:
                        failed = True
                        break
                if not failed:
                    accepted = {"actual_seed": seed, "checks": checks}
                    break
            if accepted is None:
                raise RuntimeError(f"no qualified exact root for {key}")
            data["roots"][key] = accepted
            atomic_write(args.out, data)
            new_count += 1
            print(
                f"ROOT_ACCEPTED {len(data['roots'])}/24 task={task} slot={slot} "
                f"seed={accepted['actual_seed']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
