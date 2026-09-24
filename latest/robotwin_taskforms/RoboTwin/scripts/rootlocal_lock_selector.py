#!/usr/bin/env python3
"""Acquire query/motion evidence and lock AF choices before test rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from train_af_taskforms_models import BeliefNet, query_sequence


MUS = np.asarray([0.25, 0.55, 0.85], dtype=np.float32)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def load_belief(report_path: Path) -> list[BeliefNet]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    models = []
    for member in report["belief"]["members"]:
        checkpoint = Path(member["checkpoint"])
        if sha256(checkpoint) != member["sha256"]:
            raise ValueError("belief checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = BeliefNet(input_dim=int(payload.get("input_dim", 15)))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    return models


def acquire(repo: Path, checkpoint: Path, policy_seed: int, mu: float, out: Path) -> dict:
    additional = ",".join(
        [
            f"ckpt_name={checkpoint}",
            "action_type=joint",
            "activeforcing_enabled=true",
            "af_dynamic_evaluator=false",
            "af_supplied_grasp=true",
            "af_query_enabled=true",
            "af_preaction_only=true",
            "af_query_force_n=4",
            "af_query_displacement_m=0.002",
            f"af_contact_friction={mu:g}",
            "af_force_limit_n=4",
            "start_seed=200002",
            f"af_policy_seed={policy_seed}",
            "strict_seed=true",
        ]
    )
    command = [
        sys.executable,
        str(repo / "scripts/eval_policy_xpolicylab.py"),
        "--task_name", "dump_bin_bigbin",
        "--env_cfg_type", "arx_x5",
        "--policy_name", "Pi_0",
        "--host", "localhost",
        "--port", "6001",
        "--protocol", "ws",
        "--seed", "1",
        "--test_num", "1",
        "--expert_check", "false",
        "--frequency", "30",
        "--additional_info", additional,
    ]
    environment = {**os.environ, "ROBOTWIN_SUPPRESS_EVAL_CONFIG": "1"}
    environment["PATH"] = str(repo.parent / "runtime_bin") + os.pathsep + environment.get("PATH", "")
    process = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log = out / f"preaction_mu_{mu:g}_policy_{policy_seed}.log"
    log.write_text(process.stdout, encoding="utf-8")
    evidence = None
    for line in process.stdout.splitlines():
        if line.startswith("ACTIVEFORCING_EVIDENCE "):
            evidence = json.loads(line.split(" ", 1)[1])
    if process.returncode != 0 or evidence is None:
        raise RuntimeError(f"preaction acquisition failed at mu={mu}; see {log}")
    if int(evidence["seed"]) != 200002 or int(evidence["policy_seed"]) != policy_seed:
        raise RuntimeError("seed mismatch in preaction evidence")
    query = evidence["query_info"]
    if not (
        query["final_bilateral_contact"]
        and query["contact_ratio"] >= 0.70
        and query["ee_return_error_m"] <= 0.001
        and evidence["preaction_sequence"] is not None
    ):
        raise RuntimeError("preaction evidence failed frozen qualification")
    return {
        "task": "dump_bin_bigbin",
        "root_seed": 200002,
        "root_slot": 1,
        "policy_seed": policy_seed,
        "friction_for_simulator_setup_only": mu,
        "evidence": evidence,
        "log": str(log),
        "log_sha256": sha256(log),
    }


def predict_mu(record: dict, models: list[BeliefNet]) -> tuple[float, list[float]]:
    adapter = {
        "task": "dump_bin_bigbin",
        "evidence": record["evidence"],
    }
    sequence = torch.tensor(query_sequence(adapter), dtype=torch.float32)
    padded = pad_sequence([sequence], batch_first=True)
    lengths = torch.tensor([len(sequence)], dtype=torch.long)
    with torch.no_grad():
        members = [float(model(padded, lengths)[0]) for model in models]
    return float(np.mean(members)), members


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--belief-report", type=Path, required=True)
    parser.add_argument("--selector-freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(args.selector_freeze.read_text(encoding="utf-8"))
    model_path = Path(freeze["model"])
    if sha256(model_path) != freeze["model_sha256"]:
        raise ValueError("selector model hash mismatch")
    selector = json.loads(model_path.read_text(encoding="utf-8"))
    models = load_belief(args.belief_report)
    decisions = []
    for mu in MUS:
        record = acquire(args.repo, args.checkpoint, args.policy_seed, float(mu), args.out)
        prediction, members = predict_mu(record, models)
        band = float(MUS[np.abs(MUS - prediction).argmin()])
        selected = float(selector["curves"][f"{band:.2f}"]["selected_force_n"])
        decisions.append(
            {
                "simulator_friction_for_posthoc_audit_only": float(mu),
                "belief_members": members,
                "predicted_mu": prediction,
                "predicted_mu_band": band,
                "selected_force_n": selected,
                "preaction_record": record,
            }
        )
    lock = {
        "schema_id": "AF_DUMP_ROOTLOCAL_TEST_SELECTION_LOCK_V1",
        "created_unix_s": time.time(),
        "task": "dump_bin_bigbin",
        "root_seed": 200002,
        "root_slot": 1,
        "policy_seed": args.policy_seed,
        "selector_freeze": str(args.selector_freeze),
        "selector_freeze_sha256": sha256(args.selector_freeze),
        "decisions": decisions,
        "outcomes_opened_before_lock": False,
    }
    lock_path = args.out / "selection_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"refusing to overwrite existing lock {lock_path}")
    atomic_json(lock_path, lock)
    print(json.dumps({"lock": str(lock_path), "sha256": sha256(lock_path), "decisions": decisions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
