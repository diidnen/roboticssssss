#!/usr/bin/env python3
"""Train root-disjoint friction-belief and full-task feasibility models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


TASKS = ("handover_mic", "dump_bin_bigbin")
MUS = np.asarray([0.25, 0.55, 0.85], dtype=np.float32)


class BeliefNet(nn.Module):
    def __init__(self, input_dim: int = 15):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, 24), nn.ReLU())
        self.gru = nn.GRU(24, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        projected = self.projection(sequences)
        packed = pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(1)


class FeasibilityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_gru = nn.GRU(14, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(18, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, action: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        _, hidden = self.action_gru(action)
        return self.head(torch.cat([hidden[-1], self.condition(condition)], dim=1)).squeeze(1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def task_onehot(task: str) -> list[float]:
    return [float(task == name) for name in TASKS]


def query_sequence(record: dict) -> np.ndarray:
    query = record["evidence"]["query_info"]
    rows = query["trace"]
    ee = np.asarray([row["ee_position"] for row in rows], dtype=np.float32)
    actor = np.asarray([row["actor_position"] for row in rows], dtype=np.float32)
    ee_delta = (ee - ee[0]) / 0.002
    actor_delta = (actor - actor[0]) / 0.002
    relative = actor_delta - ee_delta
    force = np.asarray(
        [[row["measured_single_finger_force_n"] / 5.0] for row in rows],
        dtype=np.float32,
    )
    contact = np.asarray(
        [[float(row["bilateral_contact"])] for row in rows], dtype=np.float32
    )
    context = np.tile(
        np.asarray(task_onehot(record["task"]) + [
            float(query["arm"] == "left"),
            float(query["arm"] == "right"),
        ], dtype=np.float32),
        (len(rows), 1),
    )
    result = np.concatenate([ee_delta, actor_delta, relative, force, contact, context], axis=1)
    if result.shape[1] != 15 or not np.isfinite(result).all():
        raise ValueError("invalid query evidence")
    return result


def belief_rows(records: list[dict]) -> list[dict]:
    chosen = {}
    for record in records:
        if not record["valid"]:
            continue
        key = (record["task"], record["root_slot"], record["friction"])
        chosen.setdefault(key, record)
    return list(chosen.values())


def batch_belief(rows: list[dict], device: str):
    sequences = [torch.tensor(query_sequence(row)) for row in rows]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor([row["friction"] for row in rows], dtype=torch.float32)
    return padded.to(device), lengths.to(device), labels.to(device)


def nearest_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    nearest = MUS[np.abs(prediction[:, None] - MUS[None, :]).argmin(axis=1)]
    return float(np.mean(nearest == target))


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    return {
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "nearest_band_accuracy": nearest_accuracy(prediction, target),
        "prediction_mean": float(np.mean(prediction)),
    }


def train_belief(records: list[dict], out: Path, device: str) -> tuple[list[BeliefNet], dict]:
    rows = belief_rows(records)
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    models = []
    report = {"counts": {key: len(value) for key, value in by_split.items()}, "members": []}
    for member_seed in (41, 42, 43):
        torch.manual_seed(member_seed)
        np.random.seed(member_seed)
        random.seed(member_seed)
        model = BeliefNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        train_batch = batch_belief(by_split["train"], device)
        val_batch = batch_belief(by_split["validation"], device)
        best_state = None
        best_mae = float("inf")
        patience = 0
        for epoch in range(1500):
            model.train()
            optimizer.zero_grad()
            prediction = model(*train_batch[:2])
            loss = nn.functional.mse_loss(prediction, train_batch[2])
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_prediction = model(*val_batch[:2])
                    val_mae = float(torch.mean(torch.abs(val_prediction - val_batch[2])))
                if val_mae + 1e-6 < best_mae:
                    best_mae = val_mae
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= 30:
                    break
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        member_report = {"seed": member_seed, "best_validation_mae": best_mae}
        for split, split_rows in by_split.items():
            batch = batch_belief(split_rows, device)
            with torch.no_grad():
                prediction = model(*batch[:2]).cpu().numpy()
            member_report[split] = regression_metrics(prediction, batch[2].cpu().numpy())
        checkpoint = out / f"belief_member_{member_seed}.pt"
        torch.save({"state_dict": best_state, "input_dim": 15}, checkpoint)
        member_report["checkpoint"] = str(checkpoint)
        member_report["sha256"] = sha256(checkpoint)
        report["members"].append(member_report)
        models.append(model.cpu())
    return models, report


def fit_normalization(records: list[dict]) -> dict:
    train = [row for row in records if row["valid"] and row["split"] == "train"]
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


def feasibility_arrays(rows: list[dict], norm: dict, device: str):
    action = np.asarray(
        [row["evidence"]["preaction_sequence"] for row in rows], dtype=np.float32
    )
    action = (action - norm["action_mean"]) / norm["action_std"]
    state = np.asarray([row["evidence"]["preaction_state"] for row in rows], dtype=np.float32)
    state = (state - norm["state_mean"]) / norm["state_std"]
    condition = np.asarray(
        [
            list(state[index])
            + [row["friction"], row["force_n"] / 5.0]
            + task_onehot(row["task"])
            for index, row in enumerate(rows)
        ],
        dtype=np.float32,
    )
    labels = np.asarray([row["evidence"]["success"] for row in rows], dtype=np.float32)
    return (
        torch.tensor(action, device=device),
        torch.tensor(condition, device=device),
        torch.tensor(labels, device=device),
    )


def binary_metrics(probability: np.ndarray, target: np.ndarray) -> dict:
    eps = 1e-6
    probability = np.clip(probability, eps, 1 - eps)
    prediction = probability >= 0.5
    positives = probability[target == 1]
    negatives = probability[target == 0]
    auc = float(np.mean(positives[:, None] > negatives[None, :])) if len(positives) and len(negatives) else None
    return {
        "nll": float(-np.mean(target * np.log(probability) + (1 - target) * np.log(1 - probability))),
        "brier": float(np.mean((probability - target) ** 2)),
        "accuracy": float(np.mean(prediction == target)),
        "auc": auc,
        "positive_rate": float(np.mean(target)),
    }


def train_feasibility(records: list[dict], out: Path, device: str) -> tuple[list[FeasibilityNet], dict]:
    valid = [
        row
        for row in records
        if row["valid"]
        and row["evidence"].get("preaction_sequence") is not None
        and row["evidence"].get("preaction_state") is not None
    ]
    by_split = {
        split: [row for row in valid if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    norm = fit_normalization(valid)
    models = []
    report = {"counts": {key: len(value) for key, value in by_split.items()}, "members": []}
    for member_seed in (71, 72, 73):
        torch.manual_seed(member_seed)
        model = FeasibilityNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        train_batch = feasibility_arrays(by_split["train"], norm, device)
        val_batch = feasibility_arrays(by_split["validation"], norm, device)
        positives = float(train_batch[2].sum())
        negatives = float(len(train_batch[2]) - positives)
        pos_weight = torch.tensor(negatives / max(positives, 1.0), device=device)
        best_state = None
        best_brier = float("inf")
        patience = 0
        for epoch in range(2000):
            model.train()
            optimizer.zero_grad()
            logits = model(train_batch[0], train_batch[1])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, train_batch[2], pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    probability = torch.sigmoid(model(val_batch[0], val_batch[1]))
                    brier = float(torch.mean((probability - val_batch[2]) ** 2))
                if brier + 1e-6 < best_brier:
                    best_brier = brier
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= 40:
                    break
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        member_report = {"seed": member_seed, "best_validation_brier": best_brier}
        for split, split_rows in by_split.items():
            batch = feasibility_arrays(split_rows, norm, device)
            with torch.no_grad():
                probability = torch.sigmoid(model(batch[0], batch[1])).cpu().numpy()
            member_report[split] = binary_metrics(probability, batch[2].cpu().numpy())
        checkpoint = out / f"feasibility_member_{member_seed}.pt"
        torch.save({"state_dict": best_state}, checkpoint)
        member_report["checkpoint"] = str(checkpoint)
        member_report["sha256"] = sha256(checkpoint)
        report["members"].append(member_report)
        models.append(model.cpu())
    report["normalization"] = {key: value.tolist() for key, value in norm.items()}
    return models, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = read_records(args.records)
    expected = 216
    if len(records) != expected:
        raise SystemExit(f"expected {expected} branches, found {len(records)}")
    keys = [record["branch_key"] for record in records]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate branch keys")
    belief_models, belief_report = train_belief(records, args.out, args.device)
    feasibility_models, feasibility_report = train_feasibility(records, args.out, args.device)
    report = {
        "schema_id": "AF_ROBOTWIN_TASKFORMS_TRAINING_REPORT_V1",
        "records": str(args.records),
        "records_sha256": sha256(args.records),
        "belief": belief_report,
        "feasibility": feasibility_report,
        "root_disjoint": True,
        "test_used_for_model_selection": False,
    }
    report_path = args.out / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
