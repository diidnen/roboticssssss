#!/usr/bin/env python3
"""Fresh-root online comparison for the frozen continuous-force selector."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from evaluate_af_taskforms_forcegrid648 import (
    DENSE_FORCES,
    FIXED_CONTROLS,
    OBSERVED_FORCES,
    choose_force,
    force_curve,
)
from evaluate_af_taskforms_inference import load_models


BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
CHECKPOINT = BASE / "checkpoints/pi0_robotwin_30000/30000"
DEFAULT_EXPERIMENT = BASE / "experiments/af_taskforms_forcegrid648_v1"
FFMPEG_DIR = BASE / "runtime_bin"
MUS = (0.25, 0.55, 0.85)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluator_env() -> dict[str, str]:
    env = dict(os.environ)
    env["ROBOTWIN_SUPPRESS_EVAL_CONFIG"] = "1"
    env["PATH"] = str(FFMPEG_DIR) + os.pathsep + env.get("PATH", "")
    return env


def force_token(force: float) -> str:
    return f"{float(force):.2f}".rstrip("0").rstrip(".")


def run_evidence(repo: Path, task: str, slot: int, seed: int, mu: float, force: float, *, preaction_only: bool):
    additional_fields = [
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
        f"start_seed={seed}",
        "strict_seed=true",
    ]
    if preaction_only:
        additional_fields.append("af_preaction_only=true")
    command = [
        sys.executable,
        str(repo / "scripts/eval_policy_xpolicylab_preaction.py"),
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
        "--additional_info", ",".join(additional_fields),
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
    if process.returncode != 0 or evidence is None:
        tail = "\n".join(process.stdout.splitlines()[-50:])
        raise RuntimeError(
            f"online branch failed task={task} slot={slot} mu={mu} force={force} "
            f"preaction_only={preaction_only} rc={process.returncode}\n{tail}"
        )
    if int(evidence["seed"]) != seed:
        raise RuntimeError("fresh-root seed substitution detected")
    query = evidence.get("query_info") or {}
    if not (
        query.get("final_bilateral_contact")
        and query.get("contact_ratio", 0.0) >= 0.70
        and query.get("ee_return_error_m", 1.0) <= 0.001
    ):
        raise RuntimeError("fresh online query gate failed")
    metrics = evidence.get("dynamic_metrics") or {}
    if not preaction_only:
        for side in ("left", "right"):
            if abs(float(metrics[f"{side}_force_limit_n"]) - force) > 1e-6:
                raise RuntimeError("fresh online force readback mismatch")
    return evidence


def contexts(root_map: dict):
    result = []
    for root_key, value in sorted(root_map["roots"].items()):
        task, slot_text = root_key.split("|fresh_slot=")
        slot = int(slot_text)
        for mu in MUS:
            result.append(
                {
                    "context_id": f"{task}|fresh_slot={slot}|mu={mu:.2f}",
                    "task": task,
                    "fresh_slot": slot,
                    "actual_seed": int(value["actual_seed"]),
                    "friction": mu,
                }
            )
    return result


def summarize(decisions: list[dict]) -> dict:
    fixed = {}
    for force in FIXED_CONTROLS:
        values = [decision["fixed_outcomes"][str(force)] for decision in decisions]
        fixed[str(force)] = {
            "successes": int(sum(values)),
            "contexts": len(values),
            "success_rate": float(np.mean(values)),
            "requested_force_n": force,
        }
    selected_successes = [decision["selected_success"] for decision in decisions]
    selected_success_forces = [
        decision["selected_force_n"]
        for decision in decisions
        if decision["selected_success"]
    ]
    by_task = {}
    for task in sorted({decision["task"] for decision in decisions}):
        rows = [decision for decision in decisions if decision["task"] == task]
        by_task[task] = {
            "contexts": len(rows),
            "selected_successes": sum(row["selected_success"] for row in rows),
            "selected_success_rate": float(np.mean([row["selected_success"] for row in rows])),
            "all_grid_fail_contexts": sum(row["all_grid_fail"] for row in rows),
            "force_selection_failures": sum(row["force_selection_failure"] for row in rows),
        }
    discordance = {}
    for force in FIXED_CONTROLS:
        af_only = sum(
            decision["selected_success"] and not decision["fixed_outcomes"][str(force)]
            for decision in decisions
        )
        fixed_only = sum(
            not decision["selected_success"] and decision["fixed_outcomes"][str(force)]
            for decision in decisions
        )
        discordance[str(force)] = {"activeforcing_only": af_only, "fixed_only": fixed_only}
    return {
        "contexts": len(decisions),
        "activeforcing": {
            "successes": int(sum(selected_successes)),
            "success_rate": float(np.mean(selected_successes)),
            "mean_selected_force_n": float(np.mean([row["selected_force_n"] for row in decisions])),
            "mean_selected_force_on_success_n": (
                float(np.mean(selected_success_forces)) if selected_success_forces else None
            ),
        },
        "fixed": fixed,
        "empirical_grid_oracle": {
            "successes": sum(not row["all_grid_fail"] for row in decisions),
            "success_rate": float(np.mean([not row["all_grid_fail"] for row in decisions])),
            "mean_min_force_on_feasible_n": float(
                np.mean(
                    [row["oracle_min_grid_force_n"] for row in decisions if not row["all_grid_fail"]]
                )
            ) if any(not row["all_grid_fail"] for row in decisions) else None,
        },
        "failure_decomposition": {
            "all_grid_fail_contexts": sum(row["all_grid_fail"] for row in decisions),
            "force_selection_failures": sum(row["force_selection_failure"] for row in decisions),
            "force_sensitive_grid_contexts": sum(row["force_sensitive_grid"] for row in decisions),
        },
        "discordance": discordance,
        "by_task": by_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--fresh-root-map", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--inference-report", type=Path)
    args = parser.parse_args()
    experiment = args.experiment
    root_map_path = args.fresh_root_map or experiment / "fresh_root_map.json"
    training_path = args.training_report or experiment / "models/training_report.json"
    inference_path = args.inference_report or experiment / "dense_inference_report.json"
    selection_path = experiment / "fresh_selection_lock.jsonl"
    selection_manifest_path = experiment / "fresh_selection_lock_manifest.json"
    branch_path = experiment / "fresh_online_branches.jsonl"
    report_path = experiment / "fresh_online_report.json"
    repo = Path(__file__).resolve().parents[1]

    root_map = json.loads(root_map_path.read_text(encoding="utf-8"))
    training_report = json.loads(training_path.read_text(encoding="utf-8"))
    inference_report = json.loads(inference_path.read_text(encoding="utf-8"))
    threshold = float(inference_report["selected_threshold"])
    belief_models, feasibility_models, norm = load_models(training_report)
    expected_contexts = contexts(root_map)
    expected_count = len(root_map.get("roots", {})) * len(MUS)
    if not expected_count or len(expected_contexts) != expected_count:
        raise SystemExit(
            f"expected {expected_count} fresh contexts, found {len(expected_contexts)}"
        )

    if branch_path.exists() and len(read_jsonl(selection_path)) != len(expected_contexts):
        raise SystemExit("outcomes exist before the full selection lock")
    locked = {row["context_id"]: row for row in read_jsonl(selection_path)}
    for context in expected_contexts:
        if context["context_id"] in locked:
            continue
        print(
            f"FRESH_SELECTION_START {len(locked) + 1}/{expected_count} "
            f"{context['context_id']}",
            flush=True,
        )
        evidence = run_evidence(
            repo,
            context["task"],
            context["fresh_slot"],
            context["actual_seed"],
            context["friction"],
            3.0,
            preaction_only=True,
        )
        if not evidence.get("af_preaction_only") or int(evidence.get("rollout_steps", -1)) != 0:
            raise RuntimeError("selection probe executed downstream actions")
        if evidence.get("preaction_sequence") is None or evidence.get("preaction_state") is None:
            raise RuntimeError("selection probe did not expose preaction context")
        reference = {
            "task": context["task"],
            "root_slot": context["fresh_slot"],
            "friction": context["friction"],
            "evidence": evidence,
        }
        raw, projected, posterior = force_curve(
            belief_models, feasibility_models, norm, reference
        )
        selected = choose_force(projected, threshold)
        row = {
            **context,
            "selected_force_n": selected,
            "selected_threshold": threshold,
            "raw_probability_by_dense_force": dict(zip(map(str, DENSE_FORCES), raw)),
            "isotonic_probability_by_dense_force": dict(zip(map(str, DENSE_FORCES), projected)),
            "query_info": evidence["query_info"],
            "preaction_sequence": evidence["preaction_sequence"],
            "preaction_state": evidence["preaction_state"],
            **posterior,
        }
        append_jsonl(selection_path, row)
        locked[context["context_id"]] = row
        print(
            f"FRESH_SELECTION_LOCKED {len(locked)}/{expected_count} "
            f"{context['context_id']} force={selected}",
            flush=True,
        )
    atomic_write(
        selection_manifest_path,
        {
            "schema_id": "AF_ROBOTWIN_FRESH_SELECTION_LOCK_V1",
            "selection_uses_downstream_outcomes": False,
            "contexts": len(locked),
            "selection_lock_sha256": sha256(selection_path),
            "fresh_root_map_sha256": sha256(root_map_path),
            "training_report_sha256": sha256(training_path),
            "inference_report_sha256": sha256(inference_path),
            "selected_threshold": threshold,
            "model_checkpoint_sha256": [
                member["sha256"] for family in ("belief", "feasibility")
                for member in training_report[family]["members"]
            ],
        },
    )

    existing_rows = read_jsonl(branch_path)
    existing = {
        (row["context_id"], round(float(row["force_n"]), 8)): row for row in existing_rows
    }
    if len(existing) != len(existing_rows):
        raise SystemExit("duplicate fresh online branch records")
    total_expected = sum(
        len(set(OBSERVED_FORCES) | {float(row["selected_force_n"])})
        for row in locked.values()
    )
    for context in expected_contexts:
        selection = locked[context["context_id"]]
        forces = sorted(set(OBSERVED_FORCES) | {float(selection["selected_force_n"])})
        for force in forces:
            key = (context["context_id"], round(force, 8))
            if key in existing:
                continue
            print(
                f"FRESH_ONLINE_START {len(existing) + 1}/{total_expected} "
                f"{context['context_id']} force={force_token(force)}",
                flush=True,
            )
            evidence = run_evidence(
                repo,
                context["task"],
                context["fresh_slot"],
                context["actual_seed"],
                context["friction"],
                force,
                preaction_only=False,
            )
            row = {
                **context,
                "force_n": force,
                "is_selected_force": abs(force - float(selection["selected_force_n"])) < 1e-8,
                "success": bool(evidence["success"]),
                "rollout_steps": int(evidence["rollout_steps"]),
                "dynamic_metrics": evidence.get("dynamic_metrics"),
                "query_info": evidence.get("query_info"),
            }
            append_jsonl(branch_path, row)
            existing[key] = row
            print(
                f"FRESH_ONLINE_DONE {len(existing)}/{total_expected} "
                f"{context['context_id']} force={force_token(force)} success={row['success']}",
                flush=True,
            )

    decisions = []
    for context in expected_contexts:
        selection = locked[context["context_id"]]
        force_map = {
            float(row["force_n"]): row
            for row in existing.values()
            if row["context_id"] == context["context_id"]
        }
        selected_force = float(selection["selected_force_n"])
        observed_success = {force: bool(force_map[force]["success"]) for force in OBSERVED_FORCES}
        selected_success = bool(force_map[selected_force]["success"])
        successful_grid = [force for force, success in observed_success.items() if success]
        decisions.append(
            {
                **context,
                "selected_force_n": selected_force,
                "selected_success": selected_success,
                "fixed_outcomes": {
                    str(force): observed_success[force] for force in FIXED_CONTROLS
                },
                "observed_grid_outcomes": {
                    str(force): observed_success[force] for force in OBSERVED_FORCES
                },
                "all_grid_fail": not successful_grid,
                "force_sensitive_grid": bool(successful_grid) and len(successful_grid) < len(OBSERVED_FORCES),
                "force_selection_failure": (not selected_success) and bool(successful_grid),
                "oracle_min_grid_force_n": min(successful_grid) if successful_grid else None,
                "selected_dynamic_metrics": force_map[selected_force]["dynamic_metrics"],
            }
        )
    output = {
        "schema_id": "AF_ROBOTWIN_TASKFORMS_FRESH_CONTINUOUS_ONLINE_REPORT_V1",
        "selection_uses_downstream_outcomes": False,
        "all_selections_locked_before_any_outcome": True,
        "fresh_root_map_sha256": sha256(root_map_path),
        "selection_lock_sha256": sha256(selection_path),
        "online_branches_sha256": sha256(branch_path),
        "summary": summarize(decisions),
        "decisions": decisions,
    }
    atomic_write(report_path, output)
    print("FRESH_ONLINE_COMPLETE " + json.dumps(output["summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
