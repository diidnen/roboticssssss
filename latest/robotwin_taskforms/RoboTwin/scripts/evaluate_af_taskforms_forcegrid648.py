#!/usr/bin/env python3
"""Dense continuous-force selection on the 648-row development grid."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_af_taskforms_inference import belief_support, load_models
from train_af_taskforms_models import task_onehot


OBSERVED_FORCES = (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0)
DENSE_FORCES = tuple(float(value) for value in np.round(np.arange(3.0, 5.0001, 0.05), 2))
FIXED_CONTROLS = (3.0, 4.25, 5.0)


def load_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def grouped(records: list[dict], split: str):
    groups: dict[tuple[str, int, float], dict[float, dict]] = defaultdict(dict)
    for record in records:
        if record["valid"] and record["split"] == split:
            key = (record["task"], int(record["root_slot"]), float(record["friction"]))
            groups[key][float(record["force_n"])] = record
    for key, force_map in groups.items():
        if set(force_map) != set(OBSERVED_FORCES):
            raise ValueError(f"incomplete force siblings for {key}: {sorted(force_map)}")
    return groups


def isotonic_increasing(values: list[float]) -> list[float]:
    blocks: list[dict] = []
    for index, value in enumerate(values):
        blocks.append({"sum": float(value), "weight": 1, "indices": [index]})
        while len(blocks) >= 2:
            left = blocks[-2]["sum"] / blocks[-2]["weight"]
            right = blocks[-1]["sum"] / blocks[-1]["weight"]
            if left <= right + 1e-12:
                break
            last = blocks.pop()
            previous = blocks.pop()
            blocks.append(
                {
                    "sum": previous["sum"] + last["sum"],
                    "weight": previous["weight"] + last["weight"],
                    "indices": previous["indices"] + last["indices"],
                }
            )
    result = np.empty(len(values), dtype=np.float64)
    for block in blocks:
        result[block["indices"]] = block["sum"] / block["weight"]
    return result.tolist()


def force_curve(belief_models, feasibility_models, norm, record: dict, forces=DENSE_FORCES):
    posterior = belief_support(belief_models, record)
    action = np.asarray(record["evidence"]["preaction_sequence"], dtype=np.float32)
    action = (action - norm["action_mean"]) / norm["action_std"]
    action_tensor = torch.tensor(action[None, ...], dtype=torch.float32)
    state = np.asarray(record["evidence"]["preaction_state"], dtype=np.float32)
    state = (state - norm["state_mean"]) / norm["state_std"]
    raw = []
    with torch.no_grad():
        for force in forces:
            probabilities = []
            for mu in posterior:
                condition = np.asarray(
                    list(state) + [mu, float(force) / 5.0] + task_onehot(record["task"]),
                    dtype=np.float32,
                )
                condition_tensor = torch.tensor(condition[None, ...])
                for model in feasibility_models:
                    probabilities.append(float(torch.sigmoid(model(action_tensor, condition_tensor))[0]))
            raw.append(float(np.mean(probabilities)))
    projected = isotonic_increasing(raw)
    return raw, projected, {
        "posterior_member_means": posterior,
        "posterior_mean": float(np.mean(posterior)),
        "posterior_std": float(np.std(posterior, ddof=1)),
    }


def choose_force(projected: list[float], threshold: float) -> float:
    for force, probability in zip(DENSE_FORCES, projected):
        if probability >= threshold:
            return force
    return DENSE_FORCES[-1]


def ceiling_observed_force(force: float) -> float:
    for observed in OBSERVED_FORCES:
        if observed + 1e-9 >= force:
            return observed
    return OBSERVED_FORCES[-1]


def evaluate(groups, belief_models, feasibility_models, norm, threshold: float):
    decisions = []
    for key, force_map in sorted(groups.items()):
        reference = force_map[3.0]
        raw, projected, posterior = force_curve(
            belief_models, feasibility_models, norm, reference
        )
        selected = choose_force(projected, threshold)
        proxy_force = ceiling_observed_force(selected)
        fixed = {
            str(force): bool(force_map[force]["evidence"]["success"])
            for force in FIXED_CONTROLS
        }
        observed_outcomes = {
            str(force): bool(force_map[force]["evidence"]["success"])
            for force in OBSERVED_FORCES
        }
        successful = [force for force in OBSERVED_FORCES if observed_outcomes[str(force)]]
        decisions.append(
            {
                "task": key[0],
                "root_slot": key[1],
                "friction_analysis_only": key[2],
                "selected_force_n": selected,
                "offline_ceiling_proxy_force_n": proxy_force,
                "selected_success_ceiling_proxy": bool(
                    force_map[proxy_force]["evidence"]["success"]
                ),
                "raw_probability_by_dense_force": dict(zip(map(str, DENSE_FORCES), raw)),
                "isotonic_probability_by_dense_force": dict(zip(map(str, DENSE_FORCES), projected)),
                "fixed_outcomes": fixed,
                "observed_grid_outcomes": observed_outcomes,
                "oracle_min_observed_force_n": min(successful) if successful else None,
                "all_force_fail": not successful,
                "force_sensitive": bool(successful) and len(successful) < len(OBSERVED_FORCES),
                **posterior,
            }
        )
    success = [decision["selected_success_ceiling_proxy"] for decision in decisions]
    successful_selected_forces = [
        decision["selected_force_n"]
        for decision in decisions
        if decision["selected_success_ceiling_proxy"]
    ]
    feasible = [decision for decision in decisions if not decision["all_force_fail"]]
    regrets = [
        max(0.0, decision["selected_force_n"] - decision["oracle_min_observed_force_n"])
        for decision in feasible
        if decision["selected_success_ceiling_proxy"]
    ]
    summary = {
        "contexts": len(decisions),
        "selected_success_rate_ceiling_proxy": float(np.mean(success)),
        "selected_mean_force_n": float(np.mean([decision["selected_force_n"] for decision in decisions])),
        "selected_mean_force_on_success_n": (
            float(np.mean(successful_selected_forces)) if successful_selected_forces else None
        ),
        "mean_nonnegative_force_regret_on_success_n": float(np.mean(regrets)) if regrets else None,
        "all_force_fail_contexts": sum(decision["all_force_fail"] for decision in decisions),
        "force_sensitive_contexts": sum(decision["force_sensitive"] for decision in decisions),
        "fixed": {
            str(force): {
                "success_rate": float(
                    np.mean([decision["fixed_outcomes"][str(force)] for decision in decisions])
                ),
                "mean_force_n": force,
            }
            for force in FIXED_CONTROLS
        },
        "oracle_observed_grid_success_rate": float(
            np.mean([not decision["all_force_fail"] for decision in decisions])
        ),
        "oracle_mean_min_force_on_feasible_n": (
            float(np.mean([decision["oracle_min_observed_force_n"] for decision in feasible]))
            if feasible
            else None
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
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    belief_models, feasibility_models, norm = load_models(report)

    validation = grouped(records, "validation")
    thresholds = [float(value) for value in np.round(np.arange(0.30, 0.801, 0.05), 2)]
    validation_results = []
    for threshold in thresholds:
        summary, _ = evaluate(validation, belief_models, feasibility_models, norm, threshold)
        validation_results.append({"threshold": threshold, **summary})
    selected_threshold = max(
        validation_results,
        key=lambda row: (
            row["selected_success_rate_ceiling_proxy"],
            -(row["selected_mean_force_on_success_n"] or 99.0),
            -row["selected_mean_force_n"],
        ),
    )["threshold"]
    test_summary, test_decisions = evaluate(
        grouped(records, "test"),
        belief_models,
        feasibility_models,
        norm,
        selected_threshold,
    )
    output = {
        "schema_id": "AF_ROBOTWIN_TASKFORMS_FORCEGRID648_DENSE_INFERENCE_V1",
        "force_model_input": "continuous scalar",
        "observed_force_support_n": list(OBSERVED_FORCES),
        "dense_selection_support_n": {"min": 3.0, "max": 5.0, "step": 0.05},
        "curve_projection": "least-squares isotonic nondecreasing",
        "offline_selected_outcome": "next-higher observed 0.25 N force proxy",
        "threshold_selection_split": "validation",
        "selected_threshold": selected_threshold,
        "validation_threshold_sweep": validation_results,
        "test_labels_accessed_after_force_selection": True,
        "test_previously_opened": True,
        "test": test_summary,
        "test_decisions": test_decisions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
