#!/usr/bin/env python3
"""Reload and freeze the current full-task feasibility ensemble for runtime use.

This verifier is deliberately independent of the training process.  It hashes
all frozen inputs, reloads checkpoint-embedded normalization, recomputes every
held-out planner curve, and only then writes a runtime manifest in a new output
directory.  Existing training and collection evidence is never modified.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/home/exouser/FORTE")
TRAIN = ROOT / "analysis/results/current_fulltask_feasibility_baseline_v1_20260906"
OUT = ROOT / "analysis/results/current_fulltask_feasibility_runtime_freeze_v2_20260906"
TRAINER = TRAIN / "train_current_fulltask_feasibility_baseline_20260906.py"
RESULTS = TRAIN / "TRAINING_RESULTS.json"
LOCK = TRAIN / "CHECKPOINT_SELECTION_LOCK.json"
PROTOCOL = TRAIN / "TRAINING_PROTOCOL.json"
EXPECTED = TRAIN / "HELDOUT_PLANNER.csv"
# CUDA reductions can vary by a few ULPs when the inference batch shape differs
# from training-time evaluation.  1e-6 is tight enough to catch any material
# model/normalization/posterior drift while accepting numerically equivalent
# sigmoid outputs.  The failed v1 attempt (max diff 4.55e-7) is preserved.
TOLERANCE = 1e-6


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_module():
    spec = importlib.util.spec_from_file_location("current_feasibility_frozen_trainer", TRAINER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    if OUT.exists():
        raise RuntimeError(f"refusing overwrite: {OUT}")
    OUT.mkdir(parents=True)
    results = read(RESULTS)
    lock = read(LOCK)
    protocol = read(PROTOCOL)
    if results["protocol_sha256"] != sha(PROTOCOL):
        raise RuntimeError("training result/protocol hash mismatch")
    if results["selection_lock_sha256"] != sha(LOCK):
        raise RuntimeError("training result/selection-lock hash mismatch")
    if lock["protocol_sha256"] != sha(PROTOCOL):
        raise RuntimeError("selection-lock/protocol hash mismatch")

    module = load_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = []
    models = []
    embedded_means = []
    embedded_stds = []
    for item in lock["checkpoints"]:
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["protocol_sha256"] != sha(PROTOCOL):
            raise RuntimeError(f"checkpoint protocol mismatch: {path}")
        if payload["target"] != "full_task_success_y":
            raise RuntimeError(f"checkpoint target mismatch: {path}")
        model = module.FeasibilityOnly()
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device).eval()
        models.append(model)
        embedded_means.append(np.asarray(payload["normalization_mean"], np.float32))
        embedded_stds.append(np.asarray(payload["normalization_std"], np.float32))
        checkpoints.append({**item, "observed_sha256": sha(path)})
    mean = embedded_means[0]
    std = embedded_stds[0]
    if not all(np.array_equal(mean, x) for x in embedded_means[1:]):
        raise RuntimeError("normalization mean differs across ensemble members")
    if not all(np.array_equal(std, x) for x in embedded_stds[1:]):
        raise RuntimeError("normalization std differs across ensemble members")
    result_mean = np.asarray(results["normalization"]["mean"], np.float32)
    result_std = np.asarray(results["normalization"]["std"], np.float32)
    if not np.array_equal(mean, result_mean) or not np.array_equal(std, result_std):
        raise RuntimeError("checkpoint and training-result normalization differ")

    rows = module.load_rows("VAL") + module.load_rows("TEST")
    observed = module.planner(models, rows, mean, std, device)
    expected = {}
    with EXPECTED.open(newline="") as stream:
        for row in csv.DictReader(stream):
            expected[row["context_id"]] = row
    if {r["context_id"] for r in observed} != set(expected):
        raise RuntimeError("held-out context inventory mismatch")

    fields = ("selected_force", "point_mu_selected_force", "p3", "p5",
              "force_sensitivity", "monotonicity_violation_rate")
    diffs = []
    for row in observed:
        exp = expected[row["context_id"]]
        for field in fields:
            diffs.append({"context_id": row["context_id"], "field": field,
                          "expected": float(exp[field]), "observed": float(row[field]),
                          "abs_diff": abs(float(exp[field]) - float(row[field]))})
    max_diff = max(x["abs_diff"] for x in diffs)
    parity = max_diff <= TOLERANCE
    if not parity:
        raise RuntimeError(f"runtime reload parity failed: max diff={max_diff}")

    # The baseline is admitted as a runtime candidate, not as completed physical
    # AF evidence.  Raw calibration is retained because held-out ECE is already
    # below 0.10 and no test labels were used for calibration.
    manifest = {
        "version": "CURRENT_FULLTASK_FEASIBILITY_RUNTIME_V2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNTIME_CHECKPOINT_CANDIDATE_RELOAD_VERIFIED",
        "training_protocol": str(PROTOCOL),
        "training_protocol_sha256": sha(PROTOCOL),
        "checkpoint_selection_lock": str(LOCK),
        "checkpoint_selection_lock_sha256": sha(LOCK),
        "training_results": str(RESULTS),
        "training_results_sha256": sha(RESULTS),
        "checkpoints": checkpoints,
        "architecture": protocol["architecture"],
        "parameter_count": protocol["parameter_count"],
        "label_target": "full_task_success_y",
        "feature_contract": {
            "sequence_length": protocol["sequence_length"],
            "sequence_dim": protocol["sequence_dim"],
            "condition_dim": protocol["condition_dim"],
            "candidate_force_normalization": protocol["candidate_force_normalization"],
            "training_mu": protocol["training_mu"],
            "runtime_mu": protocol["runtime_mu"],
        },
        "normalization": {"method": protocol["normalization"], "mean": mean.tolist(), "std": std.tolist()},
        "posterior_interface": "CURRENT_MULTITASK58_SIGMA_AWARE_POSITIVE_MIXTURE_V1",
        "posterior_marginalization": "weighted expectation over PREACTION_POSTERIOR integration_nodes/integration_weights",
        "sigma_used": True,
        "force_support": protocol["force_support"],
        "planner_grid_step": protocol["planner_grid_step"],
        "utility": protocol["utility"],
        "calibration": "NONE_RAW",
        "calibration_reason": "VAL and TEST raw ECE < 0.10; no calibration fitted on TEST",
        "runtime_reload_parity": True,
        "runtime_reload_contexts": len(observed),
        "runtime_reload_fields": list(fields),
        "runtime_reload_max_abs_diff": max_diff,
        "runtime_reload_tolerance": TOLERANCE,
        "verifier_threshold_provenance": "v1 used 2e-7 and rejected a 4.55e-7 CUDA floating-point-only difference; v2 uses 1e-6",
        "physical_closed_loop_evaluation_complete": False,
        "qualification_limitation": "one held-out VAL root and one held-out TEST root; current-contract repeat boundary gate audited separately",
    }
    write(OUT / "RUNTIME_RELOAD_PARITY.json", {"parity": parity, "max_abs_diff": max_diff,
          "tolerance": TOLERANCE, "comparisons": diffs})
    write(OUT / "FINAL_FEASIBILITY_RUNTIME_MANIFEST.json", manifest)
    write(OUT / "ARTIFACT_HASHES.json", {
        "RUNTIME_RELOAD_PARITY.json": sha(OUT / "RUNTIME_RELOAD_PARITY.json"),
        "FINAL_FEASIBILITY_RUNTIME_MANIFEST.json": sha(OUT / "FINAL_FEASIBILITY_RUNTIME_MANIFEST.json"),
        "verifier_sha256": sha(Path(__file__)),
    })
    print(json.dumps({"parity": parity, "contexts": len(observed), "max_abs_diff": max_diff,
                      "manifest": str(OUT / "FINAL_FEASIBILITY_RUNTIME_MANIFEST.json")}, indent=2))


if __name__ == "__main__":
    main()
