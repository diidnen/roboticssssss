#!/usr/bin/env python3
"""Freeze fresh outcome-blind roots for continuous-force online comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


TASKS = ("handover_mic", "dump_bin_bigbin")
MUS = (0.55, 0.25, 0.85)
ROOTS_PER_TASK = 2
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
CHECKPOINT = BASE / "checkpoints/pi0_robotwin_30000/30000"
DEFAULT_OUT = BASE / "experiments/af_taskforms_forcegrid648_v1/fresh_root_map.json"
PARENT_ROOT_MAP = BASE / "experiments/af_taskforms_216_v1/root_map.json"
FFMPEG_DIR = BASE / "runtime_bin"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


def evaluator_env() -> dict[str, str]:
    env = dict(os.environ)
    env["ROBOTWIN_SUPPRESS_EVAL_CONFIG"] = "1"
    env["PATH"] = str(FFMPEG_DIR) + os.pathsep + env.get("PATH", "")
    return env


def run_query(repo: Path, task: str, slot: int, seed: int, mu: float, expert: bool):
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
        str(repo / "scripts/eval_policy_xpolicylab.py"),
        "--task_name", task,
        "--env_cfg_type", "arx_x5",
        "--policy_name", "Pi_0",
        "--host", "localhost",
        "--port", "6001",
        "--protocol", "ws",
        "--seed", str(slot),
        "--test_num", "1",
        "--expert_check", str(expert).lower(),
        "--frequency", "30",
        "--additional_info", additional,
    ]
    process = subprocess.run(
        command,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=evaluator_env(),
    )
    marker = "ACTIVEFORCING_EVIDENCE "
    evidence = None
    for line in process.stdout.splitlines():
        if line.startswith(marker):
            evidence = json.loads(line[len(marker):])
    query = (evidence or {}).get("query_info") or {}
    valid = bool(
        process.returncode == 0
        and evidence
        and int(evidence["seed"]) == seed
        and evidence.get("af_query_only")
        and int(evidence.get("rollout_steps", -1)) == 0
        and query.get("final_bilateral_contact")
        and query.get("contact_ratio", 0.0) >= 0.70
        and query.get("ee_return_error_m", 1.0) <= 0.001
    )
    tail = process.stdout.splitlines()[-20:]
    return valid, evidence, tail, process.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        type=Path,
        help="additional root-map or JSONL files whose actual_seed values must not be reused",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="comma-separated task subset",
    )
    args = parser.parse_args()
    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    if not tasks or not set(tasks).issubset(TASKS):
        raise SystemExit(f"invalid task subset: {tasks}")
    repo = Path(__file__).resolve().parents[1]
    parent = json.loads(PARENT_ROOT_MAP.read_text(encoding="utf-8"))
    forbidden_seeds = {
        int(value["actual_seed"]) for value in parent.get("roots", {}).values()
    }
    for exclude_path in args.exclude:
        if not exclude_path.exists():
            continue
        if exclude_path.suffix == ".jsonl":
            for line in exclude_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("actual_seed") is not None:
                    forbidden_seeds.add(int(row["actual_seed"]))
        else:
            try:
                extra = json.loads(exclude_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for value in extra.get("roots", {}).values():
                if value.get("actual_seed") is not None:
                    forbidden_seeds.add(int(value["actual_seed"]))
    data = (
        json.loads(args.out.read_text(encoding="utf-8"))
        if args.out.exists()
        else {
            "schema_id": "AF_ROBOTWIN_FRESH_OUTCOME_BLIND_ROOT_MAP_V1",
            "tasks": list(tasks),
            "frictions_qualified": sorted(MUS),
            "roots_per_task": ROOTS_PER_TASK,
            "parent_root_map_sha256": sha256(PARENT_ROOT_MAP),
            "downstream_outcomes_opened_during_selection": False,
            "roots": {},
        }
    )
    new_count = 0
    if set(data.get("tasks", [])) != set(tasks):
        raise RuntimeError("existing fresh-root map task scope mismatch")
    for task_index, task in enumerate(tasks):
        for slot in range(ROOTS_PER_TASK):
            key = f"{task}|fresh_slot={slot}"
            if key in data["roots"]:
                continue
            if args.max_new and new_count >= args.max_new:
                return 0
            base_seed = 2_100_000 + task_index * 1_000_000 + slot * 100_000
            accepted = None
            for offset in range(50):
                seed = base_seed + offset
                if seed in forbidden_seeds:
                    continue
                print(f"FRESH_ROOT_CANDIDATE task={task} slot={slot} seed={seed}", flush=True)
                checks = []
                failed = False
                for index, mu in enumerate(MUS):
                    valid, evidence, tail, returncode = run_query(
                        repo, task, slot, seed, mu, expert=(index == 0)
                    )
                    query = (evidence or {}).get("query_info") or {}
                    checks.append(
                        {
                            "friction": mu,
                            "valid": valid,
                            "returncode": returncode,
                            "query_summary": {
                                field: query.get(field)
                                for field in (
                                    "samples",
                                    "contact_ratio",
                                    "final_bilateral_contact",
                                    "ee_return_error_m",
                                    "actor_return_error_m",
                                    "relative_slip_path_m",
                                )
                            },
                            "error_tail": None if valid else tail,
                        }
                    )
                    print(
                        f"FRESH_ROOT_QUERY task={task} slot={slot} seed={seed} mu={mu} valid={valid}",
                        flush=True,
                    )
                    if not valid:
                        failed = True
                        break
                if not failed:
                    accepted = {"actual_seed": seed, "checks": checks}
                    break
            if accepted is None:
                raise RuntimeError(f"no fresh query-qualified root for {key}")
            data["roots"][key] = accepted
            atomic_write(args.out, data)
            new_count += 1
            print(
                f"FRESH_ROOT_ACCEPTED {len(data['roots'])}/{len(tasks) * ROOTS_PER_TASK} "
                f"task={task} slot={slot} seed={accepted['actual_seed']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
