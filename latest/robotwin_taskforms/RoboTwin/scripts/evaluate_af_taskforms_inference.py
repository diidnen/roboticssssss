#!/usr/bin/env python3
"""Select grip force without test labels and evaluate sibling-branch outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_af_taskforms_models import (
    BeliefNet,
    FeasibilityNet,
    query_sequence,
    task_onehot,
)


FORCES = (3, 4, 5)


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_models(report: dict):
    belief = []
    for entry in report["belief"]["members"]:
        checkpoint = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
        model = BeliefNet(checkpoint["input_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        belief.append(model)
    feasibility = []
    for entry in report["feasibility"]["members"]:
        checkpoint = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
        model = FeasibilityNet()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        feasibility.append(model)
    norm = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in report["feasibility"]["normalization"].items()
    }
    return belief, feasibility, norm


def belief_support(models, record: dict) -> list[float]:
    sequence = torch.tensor(query_sequence(record)[None, ...], dtype=torch.float32)
    length = torch.tensor([sequence.shape[1]], dtype=torch.long)
    with torch.no_grad():
        return [float(model(sequence, length)[0]) for model in models]


def force_curve(belief_models, feasibility_models, norm, record: dict) -> tuple[list[float], dict]:
    posterior = belief_support(belief_models, record)
    action = np.asarray(record["evidence"]["preaction_sequence"], dtype=np.float32)
    action = (action - norm["action_mean"]) / norm["action_std"]
    action_tensor = torch.tensor(action[None, ...], dtype=torch.float32)
    state = np.asarray(record["evidence"]["preaction_state"], dtype=np.float32)
    state = (state - norm["state_mean"]) / norm["state_std"]
    probabilities = []
    with torch.no_grad():
        for force in FORCES:
            member_probabilities = []
            for mu in posterior:
                condition = np.asarray(
                    list(state)
                    + [mu, force / 5.0]
                    + task_onehot(record["task"]),
                    dtype=np.float32,
                )
                condition_tensor = torch.tensor(condition[None, ...])
                for model in feasibility_models:
                    member_probabilities.append(
                        float(torch.sigmoid(model(action_tensor, condition_tensor))[0])
                    )
            probabilities.append(float(np.mean(member_probabilities)))
    return probabilities, {
        "posterior_member_means": posterior,
        "posterior_mean": float(np.mean(posterior)),
        "posterior_std": float(np.std(posterior, ddof=1)),
    }


def choose_force(curve: list[float], threshold: float) -> int:
    for force, probability in zip(FORCES, curve):
        if probability >= threshold:
            return force
    return FORCES[-1]


def grouped(records: list[dict], split: str):
    groups = {}
    for record in records:
        if record["valid"] and record["split"] == split:
            key = (record["task"], record["root_slot"], record["friction"])
            groups.setdefault(key, {})[record["force_n"]] = record
    for key, force_map in groups.items():
        if set(force_map) != set(FORCES):
            raise ValueError(f"incomplete force siblings for {key}: {sorted(force_map)}")
    return groups


def evaluate(groups, belief_models, feasibility_models, norm, threshold: float):
    decisions = []
    for key, force_map in sorted(groups.items()):
        reference = force_map[3]
        curve, posterior = force_curve(
            belief_models, feasibility_models, norm, reference
        )
        selected = choose_force(curve, threshold)
        decision = {
            "task": key[0],
            "root_slot": key[1],
            "friction_analysis_only": key[2],
            "probability_by_force": dict(zip(map(str, FORCES), curve)),
            "selected_force_n": selected,
            "selected_success": bool(force_map[selected]["evidence"]["success"]),
            "fixed_outcomes": {
                str(force): bool(force_map[force]["evidence"]["success"])
                for force in FORCES
            },
            **posterior,
        }
        successful = [
            force for force in FORCES if force_map[force]["evidence"]["success"]
        ]
        decision["oracle_min_force_n"] = min(successful) if successful else None
        decisions.append(decision)
    summary = {
        "contexts": len(decisions),
        "activeforcing_success_rate": float(
            np.mean([decision["selected_success"] for decision in decisions])
        ),
        "activeforcing_mean_force_n": float(
            np.mean([decision["selected_force_n"] for decision in decisions])
        ),
        "fixed": {
            str(force): {
                "success_rate": float(
                    np.mean(
                        [decision["fixed_outcomes"][str(force)] for decision in decisions]
                    )
                ),
                "mean_force_n": float(force),
            }
            for force in FORCES
        },
        "oracle_success_rate": float(
            np.mean([decision["oracle_min_force_n"] is not None for decision in decisions])
        ),
        "oracle_mean_force_on_feasible_n": float(
            np.mean(
                [
                    decision["oracle_min_force_n"]
                    for decision in decisions
                    if decision["oracle_min_force_n"] is not None
                ]
            )
        ),
    }
    return summary, decisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.records)
    report = json.loads(args.training_report.read_text())
    belief_models, feasibility_models, norm = load_models(report)

    validation = grouped(records, "validation")
    candidates = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    validation_results = []
    for threshold in candidates:
        summary, _ = evaluate(
            validation, belief_models, feasibility_models, norm, threshold
        )
        validation_results.append({"threshold": threshold, **summary})
    selected = max(
        validation_results,
        key=lambda row: (
            row["activeforcing_success_rate"],
            -row["activeforcing_mean_force_n"],
        ),
    )["threshold"]
    test_summary, decisions = evaluate(
        grouped(records, "test"),
        belief_models,
        feasibility_models,
        norm,
        selected,
    )
    output = {
        "schema_id": "AF_ROBOTWIN_TASKFORMS_INFERENCE_REPORT_V1",
        "threshold_selection_split": "validation",
        "selected_threshold": selected,
        "validation_threshold_sweep": validation_results,
        "test_labels_accessed_after_force_selection": True,
        "test": test_summary,
        "test_decisions": decisions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
