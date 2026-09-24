#!/usr/bin/env python3
"""Open HELDOUT labels once after the reduced model-selection lock exists."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

import mass_belief
from mass_feasibility import Network
from time_bounded_mass_common import HERE, load_frozen_subset, read, sha256
from train_time_bounded_mass_feasibility_v1 import (
    all_rows, clean, csv_new, metric, posterior_predictions, prevalence_nll, write,
)


def main() -> None:
    heldout_path = HERE / "TIME_BOUNDED_MASS_HELDOUT_RESULTS.json"
    qualification_path = HERE / "TIME_BOUNDED_MASS_MODEL_QUALIFICATION.json"
    predictions_path = HERE / "TIME_BOUNDED_MASS_HELDOUT_PREDICTIONS.csv"
    if any(path.exists() for path in (heldout_path, qualification_path, predictions_path)):
        raise RuntimeError("HELDOUT has already been opened; refusing a second evaluation")
    subset, subset_sha = load_frozen_subset()
    lock_path = HERE / "TIME_BOUNDED_MASS_FEASIBILITY_CHECKPOINT_SELECTION_LOCK.json"
    manifest_path = HERE / "TIME_BOUNDED_MASS_FEASIBILITY_MANIFEST_PREHELDOUT.json"
    protocol_path = HERE / "TIME_BOUNDED_MASS_ANALYSIS_PROTOCOL.json"
    lock, manifest, protocol = read(lock_path), read(manifest_path), read(protocol_path)
    if lock["time_bounded_subset_sha256"] != subset_sha or manifest["time_bounded_subset_sha256"] != subset_sha:
        raise RuntimeError("selection lock is not bound to the frozen subset")
    if lock.get("heldout_labels_accessed_before_lock") is not False:
        raise RuntimeError("HELDOUT seal violated before checkpoint lock")
    if lock.get("heldout_evaluator_sha256") != sha256(Path(__file__)):
        raise RuntimeError("heldout evaluator changed after checkpoint selection")
    if lock.get("trainer_sha256") != sha256(HERE / "train_time_bounded_mass_feasibility_v1.py"):
        raise RuntimeError("trainer changed after checkpoint selection")
    if lock.get("analysis_protocol_sha256") != sha256(protocol_path):
        raise RuntimeError("analysis protocol changed after checkpoint selection")
    seal_path = HERE / "TIME_BOUNDED_MASS_HELDOUT_SEAL.json"
    if lock.get("heldout_seal_sha256") != sha256(seal_path):
        raise RuntimeError("HELDOUT seal changed after checkpoint selection")
    seal = read(seal_path)
    for row in seal["branches"]:
        branch_path = Path(row["branch_path"])
        for name, expected_hash in row["artifact_sha256"].items():
            if sha256(branch_path / name) != expected_hash:
                raise RuntimeError(f"sealed HELDOUT bytes changed: {branch_path / name}")
    for dependency, expected_hash in lock.get("dependency_sha256", {}).items():
        if sha256(Path(dependency)) != expected_hash:
            raise RuntimeError(f"heldout dependency changed after checkpoint selection: {dependency}")

    heldout = all_rows("HELDOUT")
    if len(heldout) != subset["split_branch_targets"]["HELDOUT"]:
        raise RuntimeError("all 36 frozen HELDOUT branches are required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    normalization_mean = normalization_std = None
    for checkpoint_record in lock["checkpoints"]:
        checkpoint_path = Path(checkpoint_record["path"])
        if sha256(checkpoint_path) != checkpoint_record["sha256"]:
            raise RuntimeError(f"checkpoint changed: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = Network().to(device); model.load_state_dict(payload["state_dict"]); model.eval(); models.append(model)
        mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
        std = np.asarray(payload["normalization_std"], dtype=np.float32)
        if normalization_mean is None:
            normalization_mean, normalization_std = mean, std
        elif not (np.array_equal(normalization_mean, mean) and np.array_equal(normalization_std, std)):
            raise RuntimeError("checkpoint normalization mismatch")
    belief = mass_belief.ContinuousMassBelief(
        HERE / "MASS_BELIEF_MANIFEST.json",
        manifest_sha256=sha256(HERE / "MASS_BELIEF_MANIFEST.json"),
    )
    probability = posterior_predictions(models, heldout, normalization_mean, normalization_std, device, belief)
    heldout_metric = metric([row["y"] for row in heldout], probability)
    heldout_metric["prevalence_NLL"] = prevalence_nll(heldout)
    prediction_rows = []
    groups = defaultdict(list)
    for row, p_success in zip(heldout, probability):
        record = {
            "split": "HELDOUT", "context_id": row["context"]["id"],
            "task": row["context"]["task"], "root": row["context"]["root"],
            "true_mass_kg": row["context"]["mass_kg"], "force_N": row["force"],
            "full_task_success_y": int(row["y"]), "p_success": float(p_success),
            "branch_path": str(row["job"]),
        }
        prediction_rows.append(record)
        groups[(record["task"], record["root"], record["true_mass_kg"])].append(record)
    csv_new(predictions_path, prediction_rows)
    sensitivity_rows = []
    for key, group in sorted(groups.items()):
        by_force = {row["force_N"]: row["p_success"] for row in group}
        sensitivity_rows.append({"task": key[0], "root": key[1], "mass_kg": key[2],
                                 "p5_minus_p3": by_force[5.0] - by_force[3.0]})
    mean_sensitivity = float(np.mean([row["p5_minus_p3"] for row in sensitivity_rows]))
    original_gate = lock["validation_selection"]["original_gate"]
    heldout_better = heldout_metric["NLL"] < heldout_metric["prevalence_NLL"]
    heldout_auroc_ok = heldout_metric["AUROC"] is None or heldout_metric["AUROC"] >= original_gate["HELDOUT_AUROC_min_if_defined"]
    heldout_force_ok = mean_sensitivity >= original_gate["mean_predicted_p5_minus_p3_min"]
    overall_qualified = bool(lock["validation_selection"]["validation_qualified"] and heldout_better and heldout_auroc_ok and heldout_force_ok)
    heldout_result = {
        "schema": "TIME_BOUNDED_MASS_HELDOUT_RESULTS_V1",
        "opened_once_after_selection_lock": True,
        "selection_lock_sha256": sha256(lock_path),
        "analysis_protocol_sha256": sha256(protocol_path),
        "subset_sha256": subset_sha,
        "metrics": heldout_metric,
        "mean_predicted_p5_minus_p3": mean_sensitivity,
        "force_sensitivity_by_context": sensitivity_rows,
        "predictions_csv": str(predictions_path),
        "predictions_csv_sha256": sha256(predictions_path),
        "heldout_labels_used_for_checkpoint_or_threshold_selection": False,
    }
    write(heldout_path, heldout_result)
    qualification = {
        "schema": "TIME_BOUNDED_MASS_MODEL_QUALIFICATION_V1",
        "qualified": overall_qualified,
        "status": "PASS" if overall_qualified else "FAIL",
        "validation_selection": lock["validation_selection"],
        "heldout_metrics": heldout_metric,
        "heldout_NLL_better_than_prevalence": heldout_better,
        "heldout_AUROC_gate": heldout_auroc_ok,
        "heldout_mean_p5_minus_p3": mean_sensitivity,
        "heldout_force_sensitivity_gate": heldout_force_ok,
        "label_valid_fraction": 1.0,
        "fine_force_calibration_claim": "NOT_TESTED",
        "continuous_force_response_claim": "LIMITED_BY_COARSE_GRID",
        "heldout_access_count": 1,
        "outcome_used_for_model_selection": False,
        "source_sha256": {str(Path(__file__)): sha256(Path(__file__)), str(protocol_path): sha256(protocol_path)},
    }
    write(qualification_path, qualification)
    print(json.dumps(clean(qualification), indent=2))


if __name__ == "__main__":
    main()
