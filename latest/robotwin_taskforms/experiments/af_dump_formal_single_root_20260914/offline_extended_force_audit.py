#!/usr/bin/env python3
"""Counterfactual off-support evaluation of the frozen feasibility network.

No rollout is executed.  The model weights, normalization, feature and posterior
are unchanged.  Force inputs above 8 N are deliberately extrapolations and are
reported as diagnostics, never as validated probabilities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np


CANDIDATES = np.asarray([5.0, 6.0, 8.0, 10.0, 12.0, 15.0], dtype=float)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p_at(decision: dict, force: float) -> float:
    grid = np.asarray(decision["force_grid_N"], dtype=float)
    index = int(np.argmin(np.abs(grid - force)))
    if not np.isclose(grid[index], force, rtol=0.0, atol=1e-9):
        raise AssertionError((force, grid[index]))
    return float(decision["p_success"][index])


def counts(values):
    return {str(force): int(sum(np.isclose(value, force) for value in values)) for force in CANDIDATES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.v4_root))
    from maxf8_runtime import MaxF8Feasibility

    inputs = json.loads(args.inputs.read_text())
    if not inputs.get("passed") or len(inputs.get("contexts", [])) != 24:
        raise AssertionError("expected 24 verified formal inputs")
    manifest = args.v4_root / "models_v4/feasibility/FEASIBILITY_MANIFEST.json"
    model = MaxF8Feasibility(manifest, manifest_sha256=sha256(manifest), device="cpu")

    # Exact in-support replay before deliberately extending the grid.
    original_grid = model.force_grid.copy()
    rows = []
    max_replay_error = 0.0
    for item in inputs["contexts"]:
        feature = item["feature"]
        posterior = item["posterior"]
        decision = item["original_decision"]
        replay = model.curve(np.asarray(feature["sequence"], np.float32), posterior)
        reference = np.asarray(decision["p_success"], dtype=float)
        if replay.shape != reference.shape:
            raise AssertionError("original curve shape mismatch")
        max_replay_error = max(max_replay_error, float(np.max(np.abs(replay - reference))))

        model.force_grid = CANDIDATES.copy()
        probabilities = model.curve(np.asarray(feature["sequence"], np.float32), posterior).astype(float)
        model.force_grid = original_grid.copy()
        expected_utility_15 = probabilities * (15.0 - CANDIDATES) / 15.0 - (1.0 - probabilities)
        p_peak_index = int(np.argmax(probabilities))
        utility_index = int(np.argmax(expected_utility_15))
        rows.append({
            "context_id": item["context_id"],
            "friction": float(item["friction"]),
            "policy_seed": int(item["policy_seed"]),
            "current_selected_force_N": float(decision["selected_force_N"]),
            "current_selected_probability": float(decision["predicted_success"]),
            "candidate_forces_N": CANDIDATES.tolist(),
            "extrapolated_raw_probability": probabilities.tolist(),
            "expected_utility_if_maxF15": expected_utility_15.tolist(),
            "probability_argmax_force_N": float(CANDIDATES[p_peak_index]),
            "utility_argmax_force_N_if_maxF15": float(CANDIDATES[utility_index]),
            "probability_argmax_is_above_training_support": bool(CANDIDATES[p_peak_index] > 8.0),
            "utility_argmax_is_above_training_support": bool(CANDIDATES[utility_index] > 8.0),
            "overlap_probability_replay_error": {
                str(force): float(abs(probabilities[index] - p_at(decision, force)))
                for index, force in enumerate(CANDIDATES) if force <= 8.0
            },
        })

    if max_replay_error > 1e-6:
        raise AssertionError(f"frozen curve replay mismatch: {max_replay_error}")
    force_mean = float(model.mean[10])
    force_std = float(model.std[10])
    normalized_z = ((CANDIDATES / 8.0) - force_mean) / force_std
    prob_choices = [row["probability_argmax_force_N"] for row in rows]
    utility_choices = [row["utility_argmax_force_N_if_maxF15"] for row in rows]
    output = {
        "version": "FROZEN_FEASIBILITY_OFFSUPPORT_FORCE_AUDIT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "scope": "offline counterfactual only; zero simulator steps; zero label/model/selection-lock changes",
        "warning": "10/12/15 N are outside the model's [0.5, 8] N training support. Raw network outputs there are uncalibrated extrapolations, not estimated real-world success probabilities.",
        "candidate_forces_N": CANDIDATES.tolist(),
        "counterfactual_utility": "p*(15-F)/15-(1-p); diagnostic candidate-set change only",
        "source_hashes": {
            str(args.inputs): sha256(args.inputs),
            str(manifest): sha256(manifest),
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        },
        "exact_original_curve_replay_max_abs_error": max_replay_error,
        "force_feature_normalization": {
            "raw_feature": "force_N / 8",
            "training_normalization_mean": force_mean,
            "training_normalization_std": force_std,
            "candidate_normalized_z": {str(force): float(z) for force, z in zip(CANDIDATES, normalized_z)},
            "training_support_N": [0.5, 8.0],
        },
        "summary": {
            "contexts": len(rows),
            "mean_current_selected_force_N": statistics.fmean(row["current_selected_force_N"] for row in rows),
            "probability_argmax_counts": counts(prob_choices),
            "utility_argmax_counts_if_maxF15": counts(utility_choices),
            "probability_argmax_above_8N": sum(row["probability_argmax_is_above_training_support"] for row in rows),
            "utility_argmax_above_8N_if_maxF15": sum(row["utility_argmax_is_above_training_support"] for row in rows),
            "mean_probability_by_force": {
                str(force): statistics.fmean(row["extrapolated_raw_probability"][index] for row in rows)
                for index, force in enumerate(CANDIDATES)
            },
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(output["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
