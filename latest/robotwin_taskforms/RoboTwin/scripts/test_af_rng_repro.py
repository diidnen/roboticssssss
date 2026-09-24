#!/usr/bin/env python3
"""Verify common-random-number replay for one ActiveForcing branch."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


CHECKPOINT = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "checkpoints/pi0_robotwin_30000/30000"
)


def digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_once(repo: Path, repeat: int) -> dict:
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
            "af_contact_friction=0.55",
            "af_force_limit_n=4",
            "start_seed=100000",
            "strict_seed=true",
        ]
    )
    command = [
        sys.executable,
        str(repo / "scripts/eval_policy_xpolicylab.py"),
        "--task_name", "handover_mic",
        "--env_cfg_type", "arx_x5",
        "--policy_name", "Pi_0",
        "--host", "localhost",
        "--port", "6001",
        "--protocol", "ws",
        "--seed", "0",
        "--test_num", "1",
        "--expert_check", "false",
        "--frequency", "30",
        "--additional_info", additional,
    ]
    process = subprocess.run(
        command,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "ROBOTWIN_SUPPRESS_EVAL_CONFIG": "1"},
    )
    marker = "ACTIVEFORCING_EVIDENCE "
    evidence = None
    for line in process.stdout.splitlines():
        if line.startswith(marker):
            evidence = json.loads(line[len(marker):])
    if process.returncode != 0 or evidence is None:
        tail = "\n".join(process.stdout.splitlines()[-30:])
        raise RuntimeError(f"repeat {repeat} failed rc={process.returncode}\n{tail}")
    result = {
        "repeat": repeat,
        "seed": evidence["seed"],
        "success": evidence["success"],
        "rollout_steps": evidence["rollout_steps"],
        "preaction_sequence_sha256": digest(evidence["preaction_sequence"]),
        "preaction_state_sha256": digest(evidence["preaction_state"]),
        "query_sequence_sha256": digest(evidence["query_info"]["trace"]),
        "dynamic_metrics": evidence["dynamic_metrics"],
    }
    print("RNG_REPLAY " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    first = run_once(repo, 1)
    second = run_once(repo, 2)
    exact = first == {**second, "repeat": 1}
    print(f"RNG_REPLAY_EXACT {exact}", flush=True)
    if first["preaction_sequence_sha256"] != second["preaction_sequence_sha256"]:
        print("RNG_REPLAY_ACTION_MISMATCH", flush=True)
        return 2
    if first["preaction_state_sha256"] != second["preaction_state_sha256"]:
        print("RNG_REPLAY_STATE_MISMATCH", flush=True)
        return 3
    if first["success"] != second["success"]:
        print("RNG_REPLAY_OUTCOME_MISMATCH", flush=True)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
