#!/usr/bin/env python3
"""Train belief and continuous-force feasibility ensembles on the 648-row grid."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_af_taskforms_models import (
    FeasibilityNet,
    binary_metrics,
    feasibility_arrays,
    fit_normalization,
    read_records,
    sha256,
    train_belief,
)


FORCES = (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0)
MONOTONE_WEIGHT = 0.2


def adjacent_pairs(rows: list[dict]) -> torch.Tensor:
    groups: dict[tuple[str, int, float], list[tuple[float, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["task"], int(row["root_slot"]), float(row["friction"]))].append(
            (float(row["force_n"]), index)
        )
    pairs = []
    for values in groups.values():
        ordered = sorted(values)
        if tuple(force for force, _ in ordered) != FORCES:
            raise ValueError("incomplete force siblings while constructing monotonic pairs")
        pairs.extend((ordered[index][1], ordered[index + 1][1]) for index in range(len(ordered) - 1))
    return torch.tensor(pairs, dtype=torch.long)


def monotonic_diagnostics(probability: np.ndarray, rows: list[dict]) -> dict:
    groups: dict[tuple[str, int, float], list[tuple[float, float]]] = defaultdict(list)
    for value, row in zip(probability, rows):
        groups[(row["task"], int(row["root_slot"]), float(row["friction"]))].append(
            (float(row["force_n"]), float(value))
        )
    total = 0
    violations = 0
    maximum_drop = 0.0
    for values in groups.values():
        ordered = [value for _, value in sorted(values)]
        for left, right in zip(ordered, ordered[1:]):
            total += 1
            drop = left - right
            if drop > 1e-6:
                violations += 1
                maximum_drop = max(maximum_drop, drop)
    return {
        "adjacent_pairs": total,
        "violation_count": violations,
        "violation_rate": float(violations / max(total, 1)),
        "maximum_probability_drop": float(maximum_drop),
    }


def train_feasibility(
    records: list[dict], out: Path, device: str, expected_counts: dict[str, int]
):
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
    if {key: len(value) for key, value in by_split.items()} != expected_counts:
        raise ValueError("unexpected root-disjoint split counts")
    norm = fit_normalization(valid)
    train_batch = feasibility_arrays(by_split["train"], norm, device)
    val_batch = feasibility_arrays(by_split["validation"], norm, device)
    pair_index = adjacent_pairs(by_split["train"]).to(device)
    models = []
    report = {
        "counts": expected_counts,
        "monotone_regularization_weight": MONOTONE_WEIGHT,
        "members": [],
    }
    for member_seed in (71, 72, 73):
        torch.manual_seed(member_seed)
        model = FeasibilityNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        positives = float(train_batch[2].sum())
        negatives = float(len(train_batch[2]) - positives)
        pos_weight = torch.tensor(negatives / max(positives, 1.0), device=device)
        best_state = None
        best_brier = float("inf")
        best_epoch = None
        patience = 0
        for epoch in range(2000):
            model.train()
            optimizer.zero_grad()
            logits = model(train_batch[0], train_batch[1])
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, train_batch[2], pos_weight=pos_weight
            )
            monotone_penalty = torch.relu(
                logits[pair_index[:, 0]] - logits[pair_index[:, 1]]
            ).mean()
            loss = bce + MONOTONE_WEIGHT * monotone_penalty
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    probability = torch.sigmoid(model(val_batch[0], val_batch[1]))
                    brier = float(torch.mean((probability - val_batch[2]) ** 2))
                if brier + 1e-6 < best_brier:
                    best_brier = brier
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    }
                    patience = 0
                else:
                    patience += 1
                if patience >= 40:
                    break
        if best_state is None:
            raise RuntimeError("feasibility training did not create a checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        member_report = {
            "seed": member_seed,
            "best_epoch": best_epoch,
            "best_validation_brier": best_brier,
        }
        for split, split_rows in by_split.items():
            batch = feasibility_arrays(split_rows, norm, device)
            with torch.no_grad():
                probability = torch.sigmoid(model(batch[0], batch[1])).cpu().numpy()
            member_report[split] = {
                **binary_metrics(probability, batch[2].cpu().numpy()),
                "predicted_curve": monotonic_diagnostics(probability, split_rows),
            }
        checkpoint = out / f"feasibility_member_{member_seed}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "force_input": "continuous_requested_force_n_divided_by_5",
                "force_support_n": list(FORCES),
                "monotone_weight": MONOTONE_WEIGHT,
            },
            checkpoint,
        )
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-records", type=int, default=648)
    parser.add_argument(
        "--expected-split-counts",
        default="432,108,108",
        help="train,validation,test row counts",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = read_records(args.records)
    if len(records) != args.expected_records:
        raise SystemExit(f"expected {args.expected_records} branches, found {len(records)}")
    count_values = [int(value) for value in args.expected_split_counts.split(",")]
    if len(count_values) != 3 or sum(count_values) != args.expected_records:
        raise SystemExit("invalid --expected-split-counts")
    expected_counts = dict(zip(("train", "validation", "test"), count_values))
    keys = [record["branch_key"] for record in records]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate branch keys")
    belief_models, belief_report = train_belief(records, args.out, args.device)
    _, feasibility_report = train_feasibility(
        records, args.out, args.device, expected_counts
    )
    report = {
        "schema_id": "AF_ROBOTWIN_TASKFORMS_FORCEGRID648_TRAINING_REPORT_V1",
        "records": str(args.records),
        "records_sha256": sha256(args.records),
        "force_support_n": list(FORCES),
        "force_conditioning": "continuous scalar",
        "tasks": sorted({record["task"] for record in records}),
        "belief": belief_report,
        "feasibility": feasibility_report,
        "root_disjoint": True,
        "test_used_for_model_selection": False,
        "reused_test_outcomes_previously_opened": True,
    }
    report_path = args.out / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
