#!/usr/bin/env python3
"""Train a root-local ActiveForcing belief and continuous-force selector."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from train_af_taskforms_models import (
    BeliefNet,
    FeasibilityNet,
    binary_metrics,
    query_sequence,
    regression_metrics,
)


TRAIN_POLICY_SEEDS = {60200002, 70200002, 80200002, 90200002, 100200002}
VALIDATION_POLICY_SEEDS = {110200002}
CANDIDATE_FORCES = np.round(np.arange(0.5, 8.0001, 0.25), 2).astype(np.float32)
SELECT_THRESHOLD = 0.70


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [row["branch_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate branch keys")
    for row in rows:
        if row.get("task") != "dump_bin_bigbin" or int(row.get("actual_seed", -1)) != 200002:
            raise ValueError("record outside the frozen task/root scope")
        seed = int(row["policy_seed"])
        if seed in TRAIN_POLICY_SEEDS:
            row["split"] = "train"
        elif seed in VALIDATION_POLICY_SEEDS:
            row["split"] = "validation"
        else:
            row["split"] = "unused"
        row["valid"] = True
        if bool(row["success"]) != bool(row["evidence"]["success"]):
            raise ValueError("top-level and evidence success labels disagree")
    return rows


def belief_batch(rows: list[dict], device: str):
    sequences = [torch.tensor(query_sequence(row), dtype=torch.float32) for row in rows]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    labels = torch.tensor([float(row["friction"]) for row in rows], dtype=torch.float32)
    return pad_sequence(sequences, batch_first=True).to(device), lengths.to(device), labels.to(device)


def fit_normalization(rows: list[dict]) -> dict[str, np.ndarray]:
    train = [row for row in rows if row["split"] == "train"]
    actions = np.concatenate(
        [np.asarray(row["evidence"]["preaction_sequence"], dtype=np.float32) for row in train],
        axis=0,
    )
    states = np.asarray([row["evidence"]["preaction_state"] for row in train], dtype=np.float32)
    return {
        "action_mean": actions.mean(axis=0),
        "action_std": np.maximum(actions.std(axis=0), 1e-5),
        "state_mean": states.mean(axis=0),
        "state_std": np.maximum(states.std(axis=0), 1e-5),
    }


def feasibility_batch(rows: list[dict], norm: dict[str, np.ndarray], device: str):
    actions = np.asarray([row["evidence"]["preaction_sequence"] for row in rows], dtype=np.float32)
    actions = (actions - norm["action_mean"]) / norm["action_std"]
    states = np.asarray([row["evidence"]["preaction_state"] for row in rows], dtype=np.float32)
    states = (states - norm["state_mean"]) / norm["state_std"]
    condition = np.asarray(
        [
            list(states[index]) + [float(row["friction"]), float(row["force_n"]) / 8.0, 0.0, 1.0]
            for index, row in enumerate(rows)
        ],
        dtype=np.float32,
    )
    labels = np.asarray([bool(row["evidence"]["success"]) for row in rows], dtype=np.float32)
    return (
        torch.tensor(actions, dtype=torch.float32, device=device),
        torch.tensor(condition, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
    )


def monotone_penalty(model: FeasibilityNet, action: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    # Compare each observed context with the same context at the immediately
    # higher continuous force. This has the same local monotonicity target as
    # a full dense grid but avoids repeating the action GRU 31 times/epoch.
    higher_condition = condition.clone()
    higher_condition[:, 15] = torch.clamp(higher_condition[:, 15] + 0.25 / 8.0, max=1.0)
    current_logits = model(action, condition)
    higher_logits = model(action, higher_condition)
    return torch.relu(current_logits - higher_logits).mean()


def train_belief(rows: list[dict], out: Path, device: str):
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    train_batch = belief_batch(train, device)
    validation_batch = belief_batch(validation, device)
    models = []
    report = {"counts": {"train": len(train), "validation": len(validation)}, "members": []}
    for seed in (41, 42, 43):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = BeliefNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best_state = None
        best_mae = float("inf")
        patience = 0
        best_epoch = 0
        for epoch in range(1200):
            model.train()
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(*train_batch[:2]), train_batch[2])
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    prediction = model(*validation_batch[:2])
                    mae = float(torch.mean(torch.abs(prediction - validation_batch[2])))
                if mae + 1e-6 < best_mae:
                    best_mae = mae
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                if patience >= 35:
                    break
        if best_state is None:
            raise RuntimeError("belief model produced no checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_prediction = model(*validation_batch[:2]).cpu().numpy()
        checkpoint = out / f"belief_member_{seed}.pt"
        torch.save({"state_dict": best_state, "input_dim": 15}, checkpoint)
        report["members"].append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "validation": regression_metrics(val_prediction, validation_batch[2].cpu().numpy()),
                "checkpoint": str(checkpoint),
                "sha256": sha256(checkpoint),
            }
        )
        models.append(model.cpu())
    return models, report


def train_feasibility(rows: list[dict], out: Path, device: str):
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    norm = fit_normalization(rows)
    train_batch = feasibility_batch(train, norm, device)
    validation_batch = feasibility_batch(validation, norm, device)
    positives = float(train_batch[2].sum())
    negatives = float(len(train_batch[2]) - positives)
    pos_weight = torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32, device=device)
    models = []
    report = {
        "counts": {"train": len(train), "validation": len(validation)},
        "continuous_force_support_n": CANDIDATE_FORCES.tolist(),
        "monotone_probability_regularization": 0.15,
        "members": [],
    }
    for seed in (71, 72, 73):
        torch.manual_seed(seed)
        model = FeasibilityNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_state = None
        best_brier = float("inf")
        patience = 0
        best_epoch = 0
        for epoch in range(800):
            model.train()
            optimizer.zero_grad()
            logits = model(train_batch[0], train_batch[1])
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, train_batch[2], pos_weight=pos_weight
            )
            penalty = monotone_penalty(model, train_batch[0], train_batch[1])
            loss = bce + 0.15 * penalty
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    probability = torch.sigmoid(model(validation_batch[0], validation_batch[1]))
                    brier = float(torch.mean((probability - validation_batch[2]) ** 2))
                if brier + 1e-6 < best_brier:
                    best_brier = brier
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                if patience >= 30:
                    break
        if best_state is None:
            raise RuntimeError("feasibility model produced no checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_probability = torch.sigmoid(model(validation_batch[0], validation_batch[1])).cpu().numpy()
        checkpoint = out / f"feasibility_member_{seed}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "force_normalizer_n": 8.0,
                "continuous_force_support_n": CANDIDATE_FORCES.tolist(),
                "monotone_weight": 0.15,
            },
            checkpoint,
        )
        report["members"].append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "validation": binary_metrics(val_probability, validation_batch[2].cpu().numpy()),
                "checkpoint": str(checkpoint),
                "sha256": sha256(checkpoint),
            }
        )
        models.append(model.cpu())
    report["normalization"] = {name: value.tolist() for name, value in norm.items()}
    return models, norm, report


def predict_decision(
    row: dict,
    belief_models: list[BeliefNet],
    feasibility_models: list[FeasibilityNet],
    norm: dict[str, np.ndarray],
) -> dict:
    sequence, length, _ = belief_batch([row], "cpu")
    with torch.no_grad():
        mu_members = [float(model(sequence, length)[0]) for model in belief_models]
    predicted_mu = float(np.clip(np.mean(mu_members), 0.10, 1.20))
    action = np.asarray(row["evidence"]["preaction_sequence"], dtype=np.float32)
    action = (action - norm["action_mean"]) / norm["action_std"]
    state = np.asarray(row["evidence"]["preaction_state"], dtype=np.float32)
    state = (state - norm["state_mean"]) / norm["state_std"]
    actions = torch.tensor(np.repeat(action[None, :, :], len(CANDIDATE_FORCES), axis=0))
    conditions = torch.tensor(
        np.asarray(
            [list(state) + [predicted_mu, float(force) / 8.0, 0.0, 1.0] for force in CANDIDATE_FORCES],
            dtype=np.float32,
        )
    )
    with torch.no_grad():
        member_curves = np.stack(
            [torch.sigmoid(model(actions, conditions)).numpy() for model in feasibility_models]
        )
    curve = member_curves.mean(axis=0)
    feasible = np.flatnonzero(curve >= SELECT_THRESHOLD)
    index = int(feasible[0]) if len(feasible) else len(CANDIDATE_FORCES) - 1
    return {
        "true_friction_for_audit_only": float(row["friction"]),
        "belief_members": mu_members,
        "predicted_friction": predicted_mu,
        "selected_force_n": float(CANDIDATE_FORCES[index]),
        "selected_probability": float(curve[index]),
        "candidate_forces_n": CANDIDATE_FORCES.tolist(),
        "mean_probability_curve": curve.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_records(args.records)
    counts = {
        "train": sum(row["split"] == "train" for row in rows),
        "validation": sum(row["split"] == "validation" for row in rows),
        "unused": sum(row["split"] == "unused" for row in rows),
    }
    if counts["validation"] != 9:
        raise SystemExit(f"expected the frozen nine-row validation panel, found {counts['validation']}")
    belief_models, belief_report = train_belief(rows, args.out, args.device)
    feasibility_models, norm, feasibility_report = train_feasibility(rows, args.out, args.device)
    preview = []
    seen = set()
    for row in [item for item in rows if item["split"] == "validation"]:
        key = (row["policy_seed"], row["friction"])
        if key in seen:
            continue
        seen.add(key)
        preview.append(predict_decision(row, belief_models, feasibility_models, norm))
    report = {
        "schema_id": "AF_DUMP_ROOTLOCAL_CONTINUOUS_TRAINING_V1",
        "scope": "dump_bin_bigbin root 200002 only; no cross-root generalization",
        "records": str(args.records),
        "records_sha256": sha256(args.records),
        "split_policy": {
            "train_policy_seeds": sorted(TRAIN_POLICY_SEEDS),
            "validation_policy_seeds": sorted(VALIDATION_POLICY_SEEDS),
            "test_policy_seeds": [],
        },
        "counts": counts,
        "selection_threshold": SELECT_THRESHOLD,
        "belief": belief_report,
        "feasibility": feasibility_report,
        "validation_decision_preview": preview,
        "validation_used_for_model_selection": True,
        "online_test_outcomes_used": False,
    }
    report_path = args.out / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
