#!/usr/bin/env python3
"""Resume-safe extension of the sealed 216 rows to a nine-force 648-row grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


TASKS = ("handover_mic", "dump_bin_bigbin")
MUS = (0.25, 0.55, 0.85)
FORCES = (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0)
PARENT_FORCES = (3.0, 4.0, 5.0)
NEW_FORCES = tuple(force for force in FORCES if force not in PARENT_FORCES)
ROOT_SLOTS = tuple(range(12))
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
CHECKPOINT = BASE / "checkpoints/pi0_robotwin_30000/30000"
DEFAULT_PARENT = BASE / "experiments/af_taskforms_216_v1"
DEFAULT_OUT = BASE / "experiments/af_taskforms_forcegrid648_v1"
FFMPEG_DIR = BASE / "runtime_bin"
EXPECTED_PARENT_SHA256 = "234082d3843a65ea32a82768ff0b9c80080d175beaffb93747991f2cb1681dc1"
EXPECTED_ROOT_MAP_SHA256 = "2d54f08fb1f9243c110b830017780fb1eff4aaada6281bf337c2cbc9d183c844"


def split_for(slot: int) -> str:
    if slot < 8:
        return "train"
    if slot < 10:
        return "validation"
    return "test"


def force_token(force: float) -> str:
    text = f"{float(force):.2f}".rstrip("0").rstrip(".")
    return text


def branch_key(task: str, slot: int, mu: float, force: float) -> str:
    return f"{task}|slot={slot}|mu={mu:.2f}|force={force_token(force)}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, value: dict) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluator_env() -> dict[str, str]:
    env = dict(os.environ)
    env["ROBOTWIN_SUPPRESS_EVAL_CONFIG"] = "1"
    env["PATH"] = str(FFMPEG_DIR) + os.pathsep + env.get("PATH", "")
    return env


def normalize_key(row: dict) -> tuple[str, int, float, float]:
    return (
        str(row["task"]),
        int(row["root_slot"]),
        round(float(row["friction"]), 8),
        round(float(row["force_n"]), 8),
    )


def initialize_from_parent(parent: Path, out: Path) -> tuple[Path, Path, list[dict]]:
    parent_records = parent / "branches.jsonl"
    parent_root_map = parent / "root_map.json"
    if sha256(parent_records) != EXPECTED_PARENT_SHA256:
        raise SystemExit("sealed parent records hash mismatch")
    if sha256(parent_root_map) != EXPECTED_ROOT_MAP_SHA256:
        raise SystemExit("sealed root-map hash mismatch")
    parent_rows = read_rows(parent_records)
    if len(parent_rows) != 216:
        raise SystemExit(f"sealed parent must contain 216 rows, found {len(parent_rows)}")
    parent_keys = {normalize_key(row) for row in parent_rows}
    expected_parent = {
        (task, slot, mu, force)
        for task in TASKS
        for slot in ROOT_SLOTS
        for mu in MUS
        for force in PARENT_FORCES
    }
    if parent_keys != expected_parent:
        raise SystemExit("sealed parent support mismatch")

    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "parent_216.snapshot.jsonl"
    root_map_copy = out / "root_map.json"
    records = out / "branches.jsonl"
    if not snapshot.exists():
        shutil.copyfile(parent_records, snapshot)
    if sha256(snapshot) != EXPECTED_PARENT_SHA256:
        raise SystemExit("parent snapshot hash mismatch")
    if not root_map_copy.exists():
        shutil.copyfile(parent_root_map, root_map_copy)
    if sha256(root_map_copy) != EXPECTED_ROOT_MAP_SHA256:
        raise SystemExit("copied root-map hash mismatch")
    if not records.exists():
        shutil.copyfile(snapshot, records)

    rows = read_rows(records)
    row_map = {normalize_key(row): row for row in rows}
    if len(row_map) != len(rows):
        raise SystemExit("duplicate rows already present in resume file")
    snapshot_map = {normalize_key(row): row for row in parent_rows}
    for key, original in snapshot_map.items():
        if key not in row_map or row_map[key] != original:
            raise SystemExit(f"parent row changed or missing in resume file: {key}")
    return records, root_map_copy, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-new", type=int, default=0, help="0 means all remaining")
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="comma-separated task subset and order; defaults to the full frozen order",
    )
    args = parser.parse_args()
    active_tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    if not active_tasks or len(set(active_tasks)) != len(active_tasks):
        raise SystemExit("--tasks must name at least one unique task")
    if not set(active_tasks).issubset(TASKS):
        raise SystemExit(f"--tasks must be a subset of {TASKS}, got {active_tasks}")

    repo = Path(__file__).resolve().parents[1]
    evaluator = repo / "scripts/eval_policy_xpolicylab.py"
    pi0_adapter = repo / "XPolicyLab/policy/Pi_0/model.py"
    pi0_policy = repo / "XPolicyLab/policy/Pi_0/openpi/src/openpi/policies/policy.py"
    records_path, root_map_path, rows = initialize_from_parent(args.parent, args.out)
    completed = {normalize_key(row) for row in rows}
    root_map = json.loads(root_map_path.read_text(encoding="utf-8"))
    if len(root_map.get("roots", {})) != len(TASKS) * len(ROOT_SLOTS):
        raise SystemExit("exact root map is incomplete")

    manifest = {
        "schema_id": "AF_ROBOTWIN_TASKFORMS_FORCEGRID648_V1",
        "tasks": list(TASKS),
        "active_collection_tasks": list(active_tasks),
        "frictions": list(MUS),
        "candidate_forces_n": list(FORCES),
        "new_forces_n": list(NEW_FORCES),
        "root_slots": list(ROOT_SLOTS),
        "split_rule": "slots 0-7 train, 8-9 validation, 10-11 reused diagnostic test",
        "branch_count": len(TASKS) * len(MUS) * len(FORCES) * len(ROOT_SLOTS),
        "parent": {
            "records": str(args.parent / "branches.jsonl"),
            "records_sha256": EXPECTED_PARENT_SHA256,
            "copied_rows": 216,
            "previous_test_outcomes_opened": True,
        },
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
        "collector_sha256": sha256(Path(__file__)),
        "common_random_numbers": {
            "enabled": True,
            "episode_rng_seed": "actual_seed",
            "reset_hook": "Pi_0.Model.prepare_case -> Policy.reset_rng",
            "execution": "sequential single-client collection",
        },
        "root_map": str(root_map_path),
        "root_map_sha256": sha256(root_map_path),
        "resume_record": str(records_path),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    new_count = 0
    total = int(manifest["branch_count"])
    for task in active_tasks:
        for slot in ROOT_SLOTS:
            for mu in MUS:
                for force in NEW_FORCES:
                    normalized = (task, slot, mu, force)
                    if normalized in completed:
                        continue
                    if args.max_new and new_count >= args.max_new:
                        return 0
                    exact_seed = int(root_map["roots"][f"{task}|slot={slot}"]["actual_seed"])
                    key = branch_key(task, slot, mu, force)
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
                            f"af_force_limit_n={force_token(force)}",
                            f"start_seed={exact_seed}",
                            "strict_seed=true",
                        ]
                    )
                    command = [
                        sys.executable,
                        str(evaluator),
                        "--task_name", task,
                        "--env_cfg_type", "arx_x5",
                        "--policy_name", "Pi_0",
                        "--host", "localhost",
                        "--port", "6001",
                        "--protocol", "ws",
                        "--seed", str(slot),
                        "--test_num", "1",
                        "--expert_check", "false",
                        "--frequency", "30",
                        "--additional_info", additional,
                    ]
                    print(f"AF648_BRANCH_START {len(completed) + 1}/{total} {key}", flush=True)
                    started = time.time()
                    process = subprocess.Popen(
                        command,
                        cwd=repo,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env=evaluator_env(),
                    )
                    evidence = None
                    assert process.stdout is not None
                    for line in process.stdout:
                        marker = "ACTIVEFORCING_EVIDENCE "
                        if line.startswith(marker):
                            evidence = json.loads(line[len(marker):])
                            continue
                        if line.startswith("step:"):
                            match = re.search(r"(\d+) / (\d+)", line)
                            if match and int(match.group(1)) % 50 == 0:
                                print(
                                    f"AF648_BRANCH_PROGRESS key={key} "
                                    f"step={match.group(1)}/{match.group(2)}",
                                    flush=True,
                                )
                            continue
                        print(line, end="", flush=True)
                    returncode = process.wait()
                    if returncode != 0 or evidence is None:
                        print(
                            f"AF648_BRANCH_RETRYABLE_ERROR key={key} rc={returncode} "
                            f"evidence={evidence is not None}",
                            flush=True,
                        )
                        return 2
                    if int(evidence["seed"]) != exact_seed:
                        print(
                            f"AF648_BRANCH_ROOT_MISMATCH key={key} expected={exact_seed} "
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
                        "lineage": "new_forcegrid648_v1",
                    }
                    append_jsonl(records_path, record)
                    completed.add(normalized)
                    new_count += 1
                    print(
                        f"AF648_BRANCH_DONE {len(completed)}/{total} key={key} "
                        f"valid={valid} success={evidence['success']} "
                        f"steps={evidence['rollout_steps']} "
                        f"contact_ratio={(evidence.get('dynamic_metrics') or {}).get('contact_ratio')}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
