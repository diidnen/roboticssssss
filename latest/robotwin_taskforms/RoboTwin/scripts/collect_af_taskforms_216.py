#!/usr/bin/env python3
"""Resume-safe 216-branch ActiveForcing task-form collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


TASKS = ("handover_mic", "dump_bin_bigbin")
MUS = (0.25, 0.55, 0.85)
FORCES = (3, 4, 5)
ROOT_SLOTS = tuple(range(12))
CHECKPOINT = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "checkpoints/pi0_robotwin_30000/30000"
)
DEFAULT_OUT = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "experiments/af_taskforms_216_v1"
)
ROOT_MAP_NAME = "root_map.json"


def split_for(slot: int) -> str:
    if slot < 8:
        return "train"
    if slot < 10:
        return "validation"
    return "test"


def branch_key(task: str, slot: int, mu: float, force: int) -> str:
    return f"{task}|slot={slot}|mu={mu:.2f}|force={force}"


def append_jsonl(path: Path, value: dict) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.add(json.loads(line)["branch_key"])
    return keys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-new", type=int, default=0, help="0 means all remaining")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    evaluator = repo / "scripts/eval_policy_xpolicylab.py"
    pi0_adapter = repo / "XPolicyLab/policy/Pi_0/model.py"
    pi0_policy = repo / "XPolicyLab/policy/Pi_0/openpi/src/openpi/policies/policy.py"
    args.out.mkdir(parents=True, exist_ok=True)
    records_path = args.out / "branches.jsonl"
    completed = existing_keys(records_path)
    root_map_path = args.out / ROOT_MAP_NAME
    if not root_map_path.exists():
        raise SystemExit(f"exact root map is required: {root_map_path}")
    root_map = json.loads(root_map_path.read_text(encoding="utf-8"))
    if len(root_map.get("roots", {})) != len(TASKS) * len(ROOT_SLOTS):
        raise SystemExit("exact root map is incomplete")
    manifest = {
        "schema_id": "AF_ROBOTWIN_TASKFORMS_216_V1",
        "tasks": list(TASKS),
        "frictions": list(MUS),
        "candidate_forces_n": list(FORCES),
        "root_slots": list(ROOT_SLOTS),
        "split_rule": "slots 0-7 train, 8-9 validation, 10-11 test",
        "branch_count": len(TASKS) * len(MUS) * len(FORCES) * len(ROOT_SLOTS),
        "query": {
            "schema_id": "ROBOTWIN_SHEAR_QUERY_OBSERVABLES_V1",
            "force_n": 4.0,
            "displacement_m": 0.002,
        },
        "checkpoint": str(CHECKPOINT),
        "checkpoint_metadata_sha256": sha256(CHECKPOINT / "_CHECKPOINT_METADATA"),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "evaluator_sha256": sha256(evaluator),
        "pi0_adapter_sha256": sha256(pi0_adapter),
        "pi0_policy_sha256": sha256(pi0_policy),
        "common_random_numbers": {
            "enabled": True,
            "episode_rng_seed": "actual_seed",
            "reset_hook": "Pi_0.Model.prepare_case -> Policy.reset_rng",
        },
        "root_map": str(root_map_path),
        "root_map_sha256": sha256(root_map_path),
        "resume_record": str(records_path),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    new_count = 0
    total = manifest["branch_count"]
    for task in TASKS:
        for slot in ROOT_SLOTS:
            for mu in MUS:
                for force in FORCES:
                    key = branch_key(task, slot, mu, force)
                    exact_seed = int(
                        root_map["roots"][f"{task}|slot={slot}"]["actual_seed"]
                    )
                    if key in completed:
                        continue
                    if args.max_new and new_count >= args.max_new:
                        return 0
                    additional = ",".join(
                        [
                            f"ckpt_name={CHECKPOINT}",
                            "action_type=joint",
                            "activeforcing_enabled=true",
                            "af_dynamic_evaluator=false",
                            "af_supplied_grasp=true",
                            "af_query_enabled=true",
                            "af_query_force_n=4",
                            "af_query_displacement_m=0.002",
                            f"af_contact_friction={mu}",
                            f"af_force_limit_n={force}",
                            f"start_seed={exact_seed}",
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
                        "false",
                        "--frequency",
                        "30",
                        "--additional_info",
                        additional,
                    ]
                    print(
                        f"AF_BRANCH_START {len(completed) + 1}/{total} {key}",
                        flush=True,
                    )
                    started = time.time()
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
                    assert process.stdout is not None
                    for line in process.stdout:
                        marker = "ACTIVEFORCING_EVIDENCE "
                        if line.startswith(marker):
                            evidence = json.loads(line[len(marker) :])
                            continue
                        if line.startswith("step:"):
                            match = re.search(r"(\d+) / (\d+)", line)
                            if match and int(match.group(1)) % 25 == 0:
                                print(
                                    f"AF_BRANCH_PROGRESS key={key} "
                                    f"step={match.group(1)}/{match.group(2)}",
                                    flush=True,
                                )
                            continue
                        print(line, end="", flush=True)
                    returncode = process.wait()
                    if returncode != 0 or evidence is None:
                        print(
                            f"AF_BRANCH_RETRYABLE_ERROR key={key} rc={returncode} "
                            f"evidence={evidence is not None}",
                            flush=True,
                        )
                        return 2
                    if int(evidence["seed"]) != exact_seed:
                        print(
                            f"AF_BRANCH_ROOT_MISMATCH key={key} expected={exact_seed} "
                            f"actual={evidence['seed']}",
                            flush=True,
                        )
                        return 3
                    query = evidence.get("query_info") or {}
                    valid = bool(
                        query.get("final_bilateral_contact")
                        and query.get("contact_ratio", 0.0) >= 0.70
                        and query.get("ee_return_error_m", 1.0) <= 0.001
                    )
                    record = {
                        "branch_key": key,
                        "task": task,
                        "root_slot": slot,
                        "actual_seed": evidence["seed"],
                        "split": split_for(slot),
                        "friction": mu,
                        "force_n": force,
                        "valid": valid,
                        "duration_s": time.time() - started,
                        "evidence": evidence,
                    }
                    append_jsonl(records_path, record)
                    completed.add(key)
                    new_count += 1
                    print(
                        f"AF_BRANCH_DONE {len(completed)}/{total} key={key} "
                        f"valid={valid} success={evidence['success']} "
                        f"steps={evidence['rollout_steps']} "
                        f"contact_ratio={(evidence.get('dynamic_metrics') or {}).get('contact_ratio')}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
