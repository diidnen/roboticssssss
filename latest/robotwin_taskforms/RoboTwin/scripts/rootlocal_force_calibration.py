#!/usr/bin/env python3
"""Resumable, paired force calibration for one dump_bin_bigbin root."""

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


DEFAULT_CASES = (
    (0.85, 0.5),
    (0.85, 1.0),
    (0.85, 1.5),
    (0.85, 2.0),
    (0.85, 2.5),
    (0.85, 3.0),
    (0.25, 1.0),
    (0.25, 1.5),
    (0.25, 2.0),
    (0.25, 2.5),
    (0.25, 3.0),
    (0.25, 3.5),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["branch_key"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def run_case(
    repo: Path,
    checkpoint: Path,
    root_seed: int,
    root_slot: int,
    policy_seed: int,
    friction: float,
    force: float,
    out_dir: Path,
) -> dict:
    additional = ",".join(
        [
            f"ckpt_name={checkpoint}",
            "action_type=joint",
            "activeforcing_enabled=true",
            "af_dynamic_evaluator=false",
            "af_supplied_grasp=true",
            "af_query_enabled=true",
            "af_query_force_n=4",
            "af_query_displacement_m=0.002",
            f"af_contact_friction={friction:g}",
            f"af_force_limit_n={force:g}",
            f"start_seed={root_seed}",
            f"af_policy_seed={policy_seed}",
            "strict_seed=true",
        ]
    )
    command = [
        sys.executable,
        str(repo / "scripts/eval_policy_xpolicylab.py"),
        "--task_name",
        "dump_bin_bigbin",
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
        str(root_slot),
        "--test_num",
        "1",
        "--expert_check",
        "false",
        "--frequency",
        "30",
        "--additional_info",
        additional,
    ]
    environment = {**os.environ, "ROBOTWIN_SUPPRESS_EVAL_CONFIG": "1"}
    environment["PATH"] = str(repo.parent / "runtime_bin") + os.pathsep + environment.get("PATH", "")
    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    evidence = None
    transcript: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        transcript.append(line)
        if line.startswith("ACTIVEFORCING_EVIDENCE "):
            evidence = json.loads(line.split(" ", 1)[1])
        elif line.startswith("step:"):
            match = re.search(r"(\d+) / (\d+)", line)
            if match and int(match.group(1)) % 50 == 0:
                print(
                    f"CAL_PROGRESS mu={friction:g} F={force:g} "
                    f"seed={policy_seed} {match.group(1)}/{match.group(2)}",
                    flush=True,
                )
    returncode = process.wait()
    log_path = out_dir / f"mu_{friction:g}_force_{force:g}_policy_{policy_seed}.log"
    log_path.write_text("".join(transcript), encoding="utf-8")
    if returncode != 0 or evidence is None:
        raise RuntimeError(
            f"invalid rollout mu={friction:g} F={force:g} seed={policy_seed} "
            f"returncode={returncode}; see {log_path}"
        )
    if int(evidence.get("seed", -1)) != root_seed:
        raise RuntimeError(f"root substitution detected: {evidence.get('seed')} != {root_seed}")
    if int(evidence.get("policy_seed", -1)) != policy_seed:
        raise RuntimeError(f"policy seed mismatch: {evidence.get('policy_seed')} != {policy_seed}")
    return {
        "schema_id": "AF_DUMP_ROOTLOCAL_FORCE_CALIBRATION_V1",
        "branch_key": (
            f"dump_bin_bigbin|root={root_seed}|mu={friction:g}|"
            f"force={force:g}|policy={policy_seed}"
        ),
        "task": "dump_bin_bigbin",
        "root_slot": root_slot,
        "actual_seed": root_seed,
        "policy_seed": policy_seed,
        "friction": friction,
        "force_n": force,
        "success": bool(evidence["success"]),
        "duration_s": time.time() - started,
        "evidence": evidence,
        "log_path": str(log_path),
        "log_sha256": sha256(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root-seed", type=int, default=200002)
    parser.add_argument("--root-slot", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=60200002)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case as friction,force. May be repeated; default is the frozen coarse panel.",
    )
    args = parser.parse_args()
    cases = DEFAULT_CASES if not args.case else tuple(
        tuple(float(value) for value in item.split(",", 1)) for item in args.case
    )
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "branches.jsonl"
    completed = load_completed(output)
    with output.open("a", encoding="utf-8") as stream:
        for friction, force in cases:
            branch_key = (
                f"dump_bin_bigbin|root={args.root_seed}|mu={friction:g}|"
                f"force={force:g}|policy={args.policy_seed}"
            )
            if branch_key in completed:
                print(f"CAL_SKIP {branch_key}", flush=True)
                continue
            print(f"CAL_START {branch_key}", flush=True)
            record = run_case(
                args.repo,
                args.checkpoint,
                args.root_seed,
                args.root_slot,
                args.policy_seed,
                friction,
                force,
                args.out,
            )
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            completed.add(branch_key)
            metrics = record["evidence"].get("dynamic_metrics") or {}
            print(
                f"CAL_DONE {branch_key} success={record['success']} "
                f"steps={record['evidence']['rollout_steps']} "
                f"contact={metrics.get('contact_ratio')} "
                f"measured={metrics.get('measured_force_mean_n')}",
                flush=True,
            )
    manifest = {
        "schema_id": "AF_DUMP_ROOTLOCAL_FORCE_CALIBRATION_MANIFEST_V1",
        "scope": "one fixed root; no cross-root generalization",
        "task": "dump_bin_bigbin",
        "root_seed": args.root_seed,
        "root_slot": args.root_slot,
        "policy_seed": args.policy_seed,
        "cases": [{"friction": mu, "force_n": force} for mu, force in cases],
        "checkpoint": str(args.checkpoint),
        "repo": str(args.repo),
        "output": str(output),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
