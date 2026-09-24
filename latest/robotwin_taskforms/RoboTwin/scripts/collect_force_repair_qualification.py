#!/usr/bin/env python3
"""Repeat endpoint branches with fixed environment seed and independent policy RNG."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import sys
import time


FORCES = (3.0, 5.0)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def key(row: dict) -> tuple[str, int, float]:
    return (row["task"], int(row["root_slot"]), round(float(row["friction"]), 8))


def classify(groups: dict[tuple[str, int, float], list[dict]]) -> dict[str, list[tuple[str, int, float]]]:
    result = {"EASY": [], "TRANSITION": [], "UNRESCUED": []}
    for context, rows in sorted(groups.items()):
        outcomes = {float(r["force_n"]): bool((r.get("evidence") or {}).get("success", False)) for r in rows}
        if outcomes.get(3.0) and outcomes.get(5.0):
            result["EASY"].append(context)
        elif outcomes.get(5.0) and not outcomes.get(3.0):
            result["TRANSITION"].append(context)
        else:
            result["UNRESCUED"].append(context)
    return result


def run_one(repo: Path, checkpoint: Path, reference: dict, force: float, repeat: int, out_dir: Path) -> dict:
    task, slot, mu = reference["task"], int(reference["root_slot"]), float(reference["friction"])
    root_seed = int(reference["actual_seed"])
    policy_seed = root_seed + repeat * 10_000_000 + int(force * 100)
    additional = ",".join([
        f"ckpt_name={checkpoint}", "action_type=joint", "activeforcing_enabled=true",
        "af_dynamic_evaluator=false", "af_supplied_grasp=true", "af_query_enabled=true",
        "af_query_force_n=4", "af_query_displacement_m=0.002", f"af_contact_friction={mu}",
        f"af_force_limit_n={force:g}", f"start_seed={root_seed}", f"af_policy_seed={policy_seed}",
        "strict_seed=true",
    ])
    cmd = [sys.executable, str(repo / "scripts/eval_policy_xpolicylab.py"),
        "--task_name", task, "--env_cfg_type", "arx_x5", "--policy_name", "Pi_0",
        "--host", "localhost", "--port", "6001", "--protocol", "ws", "--seed", str(slot),
        "--test_num", "1", "--expert_check", "false", "--frequency", "30",
        "--additional_info", additional]
    started = time.time()
    env = {**os.environ, "ROBOTWIN_SUPPRESS_EVAL_CONFIG": "1"}
    runtime_bin = repo.parent / "runtime_bin"
    env["PATH"] = str(runtime_bin) + os.pathsep + env.get("PATH", "")
    proc = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    evidence = None
    log_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        log_lines.append(line.rstrip())
        if line.startswith("ACTIVEFORCING_EVIDENCE "):
            evidence = json.loads(line.split(" ", 1)[1])
        elif line.startswith("step:"):
            match = re.search(r"(\d+) / (\d+)", line)
            if match and int(match.group(1)) % 50 == 0:
                print(f"REPAIR_PROGRESS {task} slot={slot} mu={mu} F={force} repeat={repeat} {match.group(1)}/{match.group(2)}", flush=True)
    rc = proc.wait()
    if rc != 0 or evidence is None:
        raise RuntimeError(f"rollout failed task={task} slot={slot} mu={mu} F={force} repeat={repeat} rc={rc} tail={log_lines[-20:]}")
    if int(evidence.get("seed", -1)) != root_seed:
        raise RuntimeError(
            f"environment seed mismatch task={task} slot={slot} mu={mu} F={force} "
            f"repeat={repeat}: expected {root_seed}, got {evidence.get('seed')}"
        )
    record = {
        "branch_key": f"{task}|slot={slot}|mu={mu:.2f}|force={force:g}|repeat={repeat}",
        "task": task, "root_slot": slot, "actual_seed": root_seed,
        "policy_seed": policy_seed, "split": "qualification", "friction": mu,
        "force_n": force, "repeat_index": repeat, "valid": True,
        "duration_s": time.time() - started, "evidence": evidence,
        "lineage": "force_supervision_repair_qualification_v1",
        "source_directory": str(out_dir),
    }
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--max-contexts", type=int, default=18)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    source = read_jsonl(args.experiment / "branches.jsonl")
    groups: dict[tuple[str, int, float], list[dict]] = defaultdict(list)
    for row in source:
        groups[key(row)].append(row)
    types = classify(groups)
    selected = types["TRANSITION"][:]
    selected.extend(types["EASY"][: max(0, args.max_contexts - len(selected)) // 2])
    selected.extend(types["UNRESCUED"][: max(0, args.max_contexts - len(selected))])
    selected = selected[:args.max_contexts]
    references = {context: next(r for r in groups[context] if float(r["force_n"]) == 3.0) for context in selected}
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "qualification_branches.jsonl"
    completed = {json.loads(x)["branch_key"] for x in output.read_text(encoding="utf-8").splitlines() if x.strip()} if output.exists() else set()
    with output.open("a", encoding="utf-8") as stream:
        for context in selected:
            reference = references[context]
            for repeat in range(1, args.repeats + 1):
                for force in FORCES:
                    branch_key = f"{reference['task']}|slot={reference['root_slot']}|mu={float(reference['friction']):.2f}|force={force:g}|repeat={repeat}"
                    if branch_key in completed:
                        continue
                    print(f"REPAIR_START {branch_key}", flush=True)
                    record = run_one(args.repo, args.checkpoint, reference, force, repeat, args.out)
                    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed.add(branch_key)
                    print(f"REPAIR_DONE {branch_key} success={record['evidence']['success']}", flush=True)
    manifest = {
        "schema_id": "AF_FORCE_SUPERVISION_REPAIR_QUALIFICATION_V1",
        "selected_contexts": [list(x) for x in selected],
        "category_counts": {k: len(v) for k, v in types.items()},
        "endpoint_forces_n": list(FORCES), "repeats_added_per_force": args.repeats,
        "fixed_environment_seed": True, "independent_policy_seed": True,
        "source_experiment": str(args.experiment), "output": str(output),
    }
    (args.out / "qualification_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
