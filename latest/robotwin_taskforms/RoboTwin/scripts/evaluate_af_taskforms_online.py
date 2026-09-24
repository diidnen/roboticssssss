#!/usr/bin/env python3
"""Online selected-force evaluation on the held-out roots.

Force selection uses only the pre-force query and motion context from the held-out
reference branch. The selected force is then executed by a fresh RoboTwin rollout;
the resulting success label is consumed only after selection for reporting.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from evaluate_af_taskforms_inference import (
    FORCES,
    choose_force,
    force_curve,
    grouped,
    load_models,
)


CHECKPOINT = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "checkpoints/pi0_robotwin_30000/30000"
)


def run_selected(repo: Path, reference: dict, force: int) -> dict:
    task = reference["task"]
    seed = int(reference["actual_seed"])
    mu = float(reference["friction"])
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
        "--seed", str(reference["root_slot"]),
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
        tail = "\n".join(process.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"online evaluation failed task={task} slot={reference['root_slot']} "
            f"force={force} rc={process.returncode}\n{tail}"
        )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    records = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    belief_models, feasibility_models, norm = load_models(report)
    groups = grouped(records, "test")
    decisions = []
    for key, force_map in sorted(groups.items()):
        reference = force_map[3]
        curve, posterior = force_curve(
            belief_models, feasibility_models, norm, reference
        )
        selected = choose_force(curve, float(report.get("selected_threshold", 0.3)))
        evidence = run_selected(repo, reference, selected)
        fixed = {
            str(force): bool(force_map[force]["evidence"]["success"])
            for force in FORCES
        }
        decisions.append(
            {
                "task": key[0],
                "root_slot": key[1],
                "friction_analysis_only": key[2],
                "selected_force_n": selected,
                "selected_success": bool(evidence["success"]),
                "selected_rollout_steps": evidence["rollout_steps"],
                "selected_dynamic_metrics": evidence.get("dynamic_metrics"),
                "probability_by_force": dict(zip(map(str, FORCES), curve)),
                "fixed_outcomes_post_selection": fixed,
                **posterior,
            }
        )
        print(
            "ONLINE_SELECTED "
            + json.dumps(
                {
                    "task": key[0],
                    "slot": key[1],
                    "mu": key[2],
                    "force": selected,
                    "success": bool(evidence["success"]),
                    "steps": evidence["rollout_steps"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    output = {
        "schema_id": "AF_ROBOTWIN_TASKFORMS_ONLINE_INFERENCE_REPORT_V1",
        "selection_uses_test_labels": False,
        "contexts": len(decisions),
        "activeforcing_success_rate": float(
            np.mean([d["selected_success"] for d in decisions])
        ),
        "activeforcing_mean_force_n": float(
            np.mean([d["selected_force_n"] for d in decisions])
        ),
        "decisions": decisions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("ONLINE_INFERENCE_COMPLETE " + json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
