#!/usr/bin/env python3
"""Train the final no-probe continuous feasibility posterior ensemble."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "analysis/results/final_no_probe_continuous_posterior_20260904"
DATA = OUT / "03_true_force_dataset/FINAL_NO_PROBE_TRUE_FORCE_DATASET.csv"
TRAIN = OUT / "05_training"
SEQ_LEN = 8
INPUT_DIM = 16
HIDDEN = 64
FMAX = 8.0
SEEDS = (0, 1, 2)


class NoProbeFeasibility(nn.Module):
    """Same direct GRU + condition MLP + feasibility-head family as old GNP."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM, HIDDEN, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(1, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    def forward(self, sequence: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(sequence)
        c = self.condition(force.reshape(-1, 1))
        return self.head(torch.cat([h[-1], c], dim=-1)).squeeze(-1)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == (y >= 0.5)))


def auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    pos = p[y >= 0.5]
    neg = p[y < 0.5]
    if not len(pos) or not len(neg):
        return None
    # Mann-Whitney formulation, with average ranks for ties.
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and sorted_p[j] == sorted_p[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return float((ranks[y >= 0.5].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if np.any(mask):
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    return {
        "count": int(len(y)),
        "bce_nll": bce(y, p) if len(y) else None,
        "accuracy": accuracy(y, p) if len(y) else None,
        "auroc": auroc(y, p) if len(y) else None,
        "ece_10bin": ece(y, p) if len(y) else None,
        "positive_rate": float(np.mean(y)) if len(y) else None,
    }


def split_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, list[int]], dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(f"{row['root_id']}::{row['context_id']}", []).append(i)
    include_root7703_in_train = os.environ.get("FINAL_PILOT_INCLUDE_ROOT7703", "0") == "1"
    forced_test = [g for g in groups if g.endswith("::root7703") and not include_root7703_in_train]
    forced_train = [g for g in groups if g.endswith("::root7703") and include_root7703_in_train]
    rest = [g for g in groups if g not in forced_test and g not in forced_train]
    rng = random.Random(20260904)
    rng.shuffle(rest)
    n_dev = max(1, round(0.12 * len(rest)))
    n_test = max(1, round(0.12 * len(rest)))
    dev_groups = rest[:n_dev]
    test_groups = rest[n_dev : n_dev + n_test] + forced_test
    train_groups = rest[n_dev + n_test :] + forced_train
    split = {
        "train": [i for g in train_groups for i in groups[g]],
        "dev": [i for g in dev_groups for i in groups[g]],
        "test": [i for g in test_groups for i in groups[g]],
    }
    manifest = {
        "schema": "TRAIN_DEV_TEST_SPLIT_V1",
        "split_unit": "root/context group; no force candidates from one context cross split",
        "seed": 20260904,
        "train_groups": train_groups,
        "dev_groups": dev_groups,
        "test_groups": test_groups,
        "root7703_forced_held_out_test": not include_root7703_in_train,
        "pilot_smoke_include_root7703_in_train": include_root7703_in_train,
        "formal_benchmark": not include_root7703_in_train,
        "counts": {k: len(v) for k, v in split.items()},
    }
    return split, manifest


def predict(model: nn.Module, seq: np.ndarray, force: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(
            torch.from_numpy(seq).float(),
            torch.from_numpy(force / FMAX).float(),
        ).cpu().numpy()


def main() -> None:
    rows = list(csv.DictReader(DATA.open(newline="", encoding="utf-8")))
    sequences = np.asarray([json.loads(r["sequence"]) for r in rows], dtype=np.float32)
    forces = np.asarray([float(r["requested_force_N"]) for r in rows], dtype=np.float32)
    labels = np.asarray([float(r["success_y"]) for r in rows], dtype=np.float32)
    split, split_manifest = split_rows(rows)
    dump(OUT / "TRAIN_DEV_TEST_SPLIT.json", split_manifest)
    normal_mean = sequences[split["train"]].mean(axis=(0, 1))
    normal_std = sequences[split["train"]].std(axis=(0, 1))
    normal_std[normal_std < 1e-6] = 1.0
    sequences_n = (sequences - normal_mean) / normal_std
    dump(TRAIN / "POSTERIOR_TRAINING_CONFIG.json", {
        "schema": "FINAL_NO_PROBE_POSTERIOR_TRAINING_CONFIG_V1",
        "architecture": "GRU(16,64) + candidate-force MLP(1,64) + feasibility head",
        "input": "8-step frozen-arm nominal prefix + task onehot; candidate absolute force F/8",
        "target": "lift + 30-step bilateral hold success",
        "force_domain_N": [1.0, 8.0],
        "loss": "BCEWithLogitsLoss",
        "seeds": list(SEEDS),
        "epochs_max": 180,
        "early_stopping_patience": 25,
        "normalization": {"mean": normal_mean.tolist(), "std": normal_std.tolist()},
        "probe_feature_used": False,
        "friction_feature_used": False,
        "no_requery": True,
        "pilot_smoke_include_root7703_in_train": os.environ.get("FINAL_PILOT_INCLUDE_ROOT7703", "0") == "1",
    })
    ensemble: list[str] = []
    seed_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = NoProbeFeasibility()
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
        tr = split["train"]
        dv = split["dev"]
        fit_indices = list(tr)
        if os.environ.get("FINAL_PILOT_INCLUDE_ROOT7703", "0") == "1":
            root_indices = [i for i, row in enumerate(rows) if row["context_id"] == "root7703"]
            # The one-context pilot has only six frontier observations. Keep
            # every row in the audit split, but balance this context during
            # pilot fitting so its observed boundary is not erased by the
            # historical broad task mixture.
            fit_indices.extend(root_indices * 200)
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(sequences_n[fit_indices]).float(),
                torch.from_numpy(forces[fit_indices] / FMAX).float(),
                torch.from_numpy(labels[fit_indices]).float(),
            ),
            batch_size=64,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        unique_context_indices = []
        seen_contexts = set()
        for i in tr:
            key = f"{rows[i]['root_id']}::{rows[i]['context_id']}"
            if key not in seen_contexts:
                seen_contexts.add(key)
                unique_context_indices.append(i)
        mono_sequence = torch.from_numpy(sequences_n[unique_context_indices]).float()
        mono_force = torch.linspace(1.0, 8.0, 8).float() / FMAX
        best = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        epochs = 0
        for epoch in range(180):
            model.train()
            for xb, fb, yb in train_loader:
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(model(xb, fb), yb)
                loss.backward()
                opt.step()
            # Feasibility is expected to be non-decreasing in candidate force.
            # This regularizes the posterior shape without inserting any
            # observed outcome or minimum-force label into the context.
            model.train()
            seq_rep = mono_sequence[:, None, :, :].expand(-1, len(mono_force), -1, -1).reshape(-1, SEQ_LEN, INPUT_DIM)
            force_rep = mono_force[None, :].expand(len(unique_context_indices), -1).reshape(-1)
            mono_logits = model(seq_rep, force_rep).reshape(len(unique_context_indices), -1)
            mono_penalty = torch.relu(mono_logits[:, :-1] - mono_logits[:, 1:]).mean()
            opt.zero_grad()
            (0.5 * mono_penalty).backward()
            opt.step()
            epochs = epoch + 1
            dev_logits = predict(model, sequences_n[dv], forces[dv])
            dev_loss = bce(labels[dv], 1 / (1 + np.exp(-np.clip(dev_logits, -40, 40))))
            if dev_loss < best - 1e-5:
                best = dev_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 25:
                break
        assert best_state is not None
        model.load_state_dict(best_state)
        path = TRAIN / "checkpoints" / f"FINAL_NO_PROBE_POSTERIOR_seed{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "architecture": {"input_dim": INPUT_DIM, "hidden": HIDDEN, "seq_len": SEQ_LEN},
            "feature_schema": str(OUT / "FINAL_FEATURE_SCHEMA.json"),
            "normalization": {"mean": normal_mean, "std": normal_std},
            "force_normalization": "F/8",
            "seed": seed,
        }, path)
        ensemble.append(str(path))
        seed_results.append({
            "seed": seed,
            "epochs": epochs,
            "best_dev_bce": best,
            "train": metrics(labels[tr], predict(model, sequences_n[tr], forces[tr])),
            "dev": metrics(labels[dv], predict(model, sequences_n[dv], forces[dv])),
            "test": metrics(labels[split["test"]], predict(model, sequences_n[split["test"]], forces[split["test"]])),
        })
    models = []
    for path in ensemble:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = NoProbeFeasibility()
        model.load_state_dict(ckpt["model_state_dict"])
        models.append(model)
    test_i = split["test"]
    ensemble_test = np.mean([predict(m, sequences_n[test_i], forces[test_i]) for m in models], axis=0)
    # Root7703 is intentionally held out; evaluate the posterior curve offline.
    root_seq = sequences_n[next(i for i, r in enumerate(rows) if r["context_id"] == "root7703")]
    root_seq = np.repeat(root_seq[None, ...], 701, axis=0)
    force_grid = np.linspace(1.0, 8.0, 701, dtype=np.float32)
    root_logits = np.mean([
        predict(m, root_seq, force_grid) for m in models
    ], axis=0)
    root_p = 1.0 / (1.0 + np.exp(-np.clip(root_logits, -40, 40)))
    utility = root_p * (1.0 - force_grid / FMAX) + (1.0 - root_p) * (-1.0)
    selected = int(np.argmax(utility))
    monotonic: dict[str, Any] = {}
    violations = []
    for key in sorted({r["context_id"] for r in rows})[:100]:
        i = next(i for i, r in enumerate(rows) if r["context_id"] == key)
        seq = np.repeat(sequences_n[i][None, ...], len(force_grid), axis=0)
        p = 1 / (1 + np.exp(-np.clip(np.mean([predict(m, seq, force_grid) for m in models], axis=0), -40, 40)))
        violations.append(int(np.sum(np.diff(p) < -1e-4)))
    monotonic = {
        "contexts_audited": len(violations),
        "contexts_with_any_adjacent_decrease": int(sum(v > 0 for v in violations)),
        "total_adjacent_decreases": int(sum(violations)),
        "hard_monotonicity_constraint_used": False,
    }
    dump(TRAIN / "POSTERIOR_TRAINING_RESULT.json", {
        "schema": "FINAL_NO_PROBE_POSTERIOR_TRAINING_RESULT_V1",
        "checkpoint_paths": ensemble,
        "checkpoint": ensemble[0],
        "ensemble_size": len(ensemble),
        "dataset": str(DATA),
        "sample_count": len(rows),
        "context_count": len({r["context_id"] for r in rows}),
        "train_dev_test_metrics": seed_results,
        "ensemble_test_metrics": metrics(labels[test_i], ensemble_test),
        "monotonicity_audit": monotonic,
        "status": "VALID_FOR_OFFLINE_SMOKE",
        "probe_feature_used": False,
    })
    (OUT / "FINAL_CHECKPOINT_PATH.txt").write_text(ensemble[0] + "\n", encoding="utf-8")
    dump(OUT / "ROOT7703_POSTERIOR_CURVE.json", {
        "schema": "ROOT7703_POSTERIOR_CURVE_V1",
        "checkpoint_paths": ensemble,
        "context_id": "root7703",
        "force_grid_N": force_grid.tolist(),
        "posterior_success": root_p.tolist(),
        "utility": utility.tolist(),
        "selected_force_N": float(force_grid[selected]),
        "selected_predicted_success_probability": float(root_p[selected]),
        "selected_utility": float(utility[selected]),
        "utility_rule": "p*(1-F/8)+(1-p)*(-1), recovered from prior continuous ActiveForcing integration scaffold",
        "empirical_frontier": "(3.5N, 4.0N]",
        "probe_feature_used": False,
    })
    print(json.dumps({
        "status": "VALID_FOR_OFFLINE_SMOKE",
        "samples": len(rows),
        "test": metrics(labels[test_i], ensemble_test),
        "root7703_selected_force_N": float(force_grid[selected]),
        "root7703_selected_p": float(root_p[selected]),
        "monotonicity": monotonic,
    }, indent=2))


if __name__ == "__main__":
    main()
