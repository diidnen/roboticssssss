#!/usr/bin/env python3
"""Fit and freeze a low-capacity root-local continuous feasibility model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from train_af_taskforms_models import BeliefNet, query_sequence
from torch.nn.utils.rnn import pad_sequence


TRAIN_SEEDS = {60200002, 70200002, 80200002, 90200002, 100200002}
VALIDATION_SEED = 110200002
DEVELOPMENT_ONLY_SEED = 120200002
MUS = np.asarray([0.25, 0.55, 0.85], dtype=np.float32)
CANDIDATE_FORCES = np.round(np.arange(0.5, 8.0001, 0.25), 2)
THRESHOLD = 0.80


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 77 or len({row["branch_key"] for row in rows}) != 77:
        raise ValueError("expected 77 unique development rows")
    if {int(row["actual_seed"]) for row in rows} != {200002}:
        raise ValueError("row outside fixed root")
    return rows


def load_belief_models(report_path: Path) -> list[BeliefNet]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    models = []
    for member in report["belief"]["members"]:
        checkpoint = Path(member["checkpoint"])
        if sha256(checkpoint) != member["sha256"]:
            raise ValueError("belief checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = BeliefNet(input_dim=int(payload.get("input_dim", 15)))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    return models


def predict_mu(row: dict, models: list[BeliefNet]) -> tuple[float, list[float]]:
    sequence = torch.tensor(query_sequence(row), dtype=torch.float32)
    padded = pad_sequence([sequence], batch_first=True)
    lengths = torch.tensor([len(sequence)], dtype=torch.long)
    with torch.no_grad():
        members = [float(model(padded, lengths)[0]) for model in models]
    return float(np.mean(members)), members


def fit_curves(rows: list[dict]) -> dict[str, dict]:
    result = {}
    train = [row for row in rows if int(row["policy_seed"]) in TRAIN_SEEDS]
    for mu in MUS:
        subset = [row for row in train if abs(float(row["friction"]) - float(mu)) < 1e-5]
        force = np.asarray([float(row["force_n"]) for row in subset])
        success = np.asarray([bool(row["success"]) for row in subset], dtype=float)
        model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        model.fit(force, success)
        probability = model.predict(CANDIDATE_FORCES)
        feasible = np.flatnonzero(probability >= THRESHOLD)
        index = int(feasible[0]) if len(feasible) else len(CANDIDATE_FORCES) - 1
        result[f"{float(mu):.2f}"] = {
            "fit_force_thresholds_n": model.X_thresholds_.tolist(),
            "fit_probability_thresholds": model.y_thresholds_.tolist(),
            "candidate_probability": probability.tolist(),
            "selected_force_n": float(CANDIDATE_FORCES[index]),
            "selected_probability": float(probability[index]),
            "training_rows": len(subset),
            "training_positive_rate": float(success.mean()),
        }
    return result


def validation_audit(rows: list[dict], curves: dict[str, dict], models: list[BeliefNet]) -> list[dict]:
    validation = [row for row in rows if matches_seed(row, VALIDATION_SEED)]
    decisions = []
    for true_mu in MUS:
        siblings = [row for row in validation if abs(float(row["friction"]) - float(true_mu)) < 1e-5]
        reference = sorted(siblings, key=lambda row: float(row["force_n"]))[0]
        predicted_mu, members = predict_mu(reference, models)
        band = float(MUS[np.abs(MUS - predicted_mu).argmin()])
        selected = float(curves[f"{band:.2f}"]["selected_force_n"])
        matches = [row for row in siblings if abs(float(row["force_n"]) - selected) < 1e-8]
        decisions.append(
            {
                "true_mu_for_audit_only": float(true_mu),
                "belief_members": members,
                "predicted_mu": predicted_mu,
                "predicted_mu_band": band,
                "selected_force_n": selected,
                "selected_outcome_observed": len(matches) == 1,
                "selected_success": None if len(matches) != 1 else bool(matches[0]["success"]),
            }
        )
    return decisions


def matches_seed(row: dict, seed: int) -> bool:
    return int(row["policy_seed"]) == seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--belief-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.records)
    models = load_belief_models(args.belief_report)
    curves = fit_curves(rows)
    validation = validation_audit(rows, curves, models)
    if not all(item["selected_outcome_observed"] for item in validation):
        raise RuntimeError("frozen threshold selected an unobserved validation force")
    if not all(item["selected_success"] for item in validation):
        raise RuntimeError("root-local selector failed validation success gate")
    model = {
        "schema_id": "AF_DUMP_ROOTLOCAL_ISOTONIC_FEASIBILITY_V1",
        "scope": "dump_bin_bigbin root 200002 only; no cross-root generalization",
        "training_policy_seeds": sorted(TRAIN_SEEDS),
        "validation_policy_seed": VALIDATION_SEED,
        "development_only_policy_seed": DEVELOPMENT_ONLY_SEED,
        "selection_threshold": THRESHOLD,
        "candidate_forces_n": CANDIDATE_FORCES.tolist(),
        "belief_report": str(args.belief_report),
        "belief_report_sha256": sha256(args.belief_report),
        "curves": curves,
        "validation_decisions": validation,
        "validation_successes": sum(bool(item["selected_success"]) for item in validation),
        "validation_trials": len(validation),
        "development_only_rows_used_for_fit_or_selection": 0,
        "future_test_policy_seed": 130200002,
    }
    model_path = args.out / "isotonic_model.json"
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {
        "schema_id": "AF_DUMP_ROOTLOCAL_SELECTOR_FREEZE_V1",
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "selected_force_by_belief_band_n": {
            band: value["selected_force_n"] for band, value in curves.items()
        },
        "selection_threshold": THRESHOLD,
        "candidate_forces_n": CANDIDATE_FORCES.tolist(),
        "test_policy_seed": 130200002,
        "test_outcomes_opened_at_freeze": False,
    }
    freeze_path = args.out / "selection_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
