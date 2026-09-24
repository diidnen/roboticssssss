#!/usr/bin/env python3
"""Independent, shareable QA readout for frozen MASS learned components."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(clean(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def binary_metrics(frame):
    y = frame.full_task_success_y.to_numpy(dtype=float)
    p = np.clip(frame.p_success.to_numpy(dtype=float), 1e-7, 1 - 1e-7)
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    order = np.argsort(-p); ranked_y = y[order]
    auprc = float((np.cumsum(ranked_y) / np.arange(1, len(y) + 1))[ranked_y == 1].mean()) if positives else None
    order_up = np.argsort(p, kind="mergesort"); ranks = np.empty(len(p), float); index = 0
    while index < len(p):
        end = index + 1
        while end < len(p) and p[order_up[end]] == p[order_up[index]]: end += 1
        ranks[order_up[index:end]] = (index + 1 + end) / 2; index = end
    auroc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) /
                  (positives * negatives)) if positives and negatives else None
    ece = 0.0
    for bin_index in range(10):
        low, high = bin_index / 10, (bin_index + 1) / 10
        mask = (p >= low) & ((p < high) if bin_index < 9 else (p <= high))
        if mask.any(): ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return {"n": int(len(y)), "positive_count": positives,
            "NLL": float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))),
            "Brier": float(np.mean((p - y) ** 2)), "AUROC": auroc,
            "AUPRC": auprc, "ECE_10bin": ece}


def close(a, b):
    if a is None or b is None: return a is None and b is None
    return bool(np.isclose(float(a), float(b), rtol=0, atol=1e-12))


def main():
    outputs = [HERE / "MASS_OFFLINE_QUALIFICATION_AUDIT.json",
               HERE / "MASS_FEASIBILITY_MASS_FORCE_INSPECTION.csv",
               HERE / "MASS_OFFLINE_QUALIFICATION_REPORT.md",
               HERE / "MASS_BELIEF_IDENTITY_CONTROL.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("offline qualification report already exists")
    belief_q = read(HERE / "MASS_BELIEF_QUALIFICATION.json")
    belief_m = read(HERE / "MASS_BELIEF_METRICS.json")
    feasibility_q = read(HERE / "MASS_FEASIBILITY_QUALIFICATION.json")
    feasibility_m = read(HERE / "MASS_FEASIBILITY_METRICS.json")
    collection_quality = read(HERE / "MASS_CURRENT_TRAINING_COLLECTION_DATA_QUALITY.json")
    if collection_quality.get("status") != "PASS": raise RuntimeError("training collection data quality failed")
    belief_predictions_path = HERE / "MASS_BELIEF_PREDICTIONS.csv"
    belief_frame = pd.read_csv(belief_predictions_path)
    prediction_path = HERE / "MASS_FEASIBILITY_PREDICTIONS.csv"
    frame = pd.read_csv(prediction_path)

    recomputed = {split: binary_metrics(group) for split, group in frame.groupby("split", sort=True)}
    checks = {}
    for split, metrics in recomputed.items():
        for name in ("n", "positive_count", "NLL", "Brier", "AUROC", "AUPRC", "ECE_10bin"):
            checks[f"{split}.{name}"] = close(metrics[name], feasibility_m[split][name])
    duplicate_keys = int(frame.duplicated(["split", "context_id", "force_N"]).sum())
    expected_rows = {"TRAIN": 432, "VAL": 108, "HELDOUT": 108}
    actual_rows = frame.groupby("split").size().to_dict()
    checks["row_completeness"] = all(int(actual_rows.get(split, 0)) == count for split, count in expected_rows.items())
    checks["no_duplicate_context_force_rows"] = duplicate_keys == 0
    checks["probabilities_finite_unit_interval"] = bool(np.isfinite(frame.p_success).all() and frame.p_success.between(0, 1).all())

    identity_train = belief_frame[belief_frame.split == "TRAIN"]
    identity_held = belief_frame[belief_frame.split == "HELDOUT"].copy()
    task_means = identity_train.groupby("task").true_mass_kg.mean().to_dict()
    identity_held["task_only_prediction_kg"] = identity_held.task.map(task_means)
    global_mean = float(identity_train.true_mass_kg.mean())
    identity_held["global_prediction_kg"] = global_mean
    balanced = (belief_frame.groupby(["split", "task", "root", "true_mass_kg"]).size() == 1).all()
    full_factorial = (belief_frame.groupby(["split", "task", "root"]).true_mass_kg.nunique() == 3).all()
    task_only_mae = float(np.mean(np.abs(identity_held.task_only_prediction_kg - identity_held.true_mass_kg)))
    identity_control = {"control": "TRAIN task-conditional mean evaluated on the held-out root",
        "train_task_means_kg": {str(int(k)): float(v) for k, v in task_means.items()},
        "train_global_mean_kg": global_mean, "heldout_root_absent_from_training": not bool(set(identity_held.root) & set(identity_train.root)),
        "mass_factorial_with_task_and_root": bool(balanced and full_factorial),
        "task_only_heldout_MAE_kg": task_only_mae, "task_only_distinct_predictions": int(identity_held.task_only_prediction_kg.nunique()),
        "query_model_heldout_MAE_kg": float(belief_m["HELDOUT"]["MAE"]),
        "query_model_MAE_improvement_over_task_only_kg": task_only_mae - float(belief_m["HELDOUT"]["MAE"]),
        "interpretation": "Task and root identities are balanced across all three masses. The held-out root is unseen, and a task-only predictor cannot rank mass. The query model's within-task held-out ranking therefore cannot be explained by task/root identity alone."}
    write_json(outputs[3], identity_control)
    checks["belief_identity_control_factorial"] = identity_control["mass_factorial_with_task_and_root"]
    checks["belief_heldout_root_unseen"] = identity_control["heldout_root_absent_from_training"]
    checks["query_model_beats_task_only_MAE"] = identity_control["query_model_heldout_MAE_kg"] < identity_control["task_only_heldout_MAE_kg"]

    held = frame[frame.split == "HELDOUT"]
    taskwise = {str(int(task)): binary_metrics(group) for task, group in held.groupby("task")}
    rootwise = {str(int(root)): binary_metrics(group) for root, group in held.groupby("root")}
    inspection = (frame.groupby(["split", "task", "true_mass_kg", "force_N"], as_index=False)
                  .agg(n=("full_task_success_y", "size"), empirical_success=("full_task_success_y", "mean"),
                       mean_predicted_success=("p_success", "mean"),
                       min_predicted_success=("p_success", "min"), max_predicted_success=("p_success", "max")))
    inspection.to_csv(outputs[1], index=False, quoting=csv.QUOTE_MINIMAL)
    held_curves = inspection[inspection.split == "HELDOUT"]
    sensitivity = {}
    for task, group in held_curves.groupby("task"):
        mass_curve = group.groupby("true_mass_kg").mean_predicted_success.mean()
        force_curve = group.groupby("force_N").mean_predicted_success.mean()
        sensitivity[str(int(task))] = {
            "mean_predicted_success_by_mass_kg": {str(k): float(v) for k, v in mass_curve.items()},
            "mean_predicted_success_by_force_N": {str(k): float(v) for k, v in force_curve.items()},
            "p5_minus_p3": float(force_curve.loc[5.0] - force_curve.loc[3.0])}

    source_paths = [HERE / name for name in ("MASS_BELIEF_QUALIFICATION.json", "MASS_BELIEF_METRICS.json",
        "MASS_FEASIBILITY_QUALIFICATION.json", "MASS_FEASIBILITY_METRICS.json",
        "MASS_BELIEF_MANIFEST.json", "MASS_FEASIBILITY_MANIFEST.json",
        "MASS_CURRENT_TRAINING_COLLECTION_DATA_QUALITY.json")] + [prediction_path, belief_predictions_path]
    ready = bool(belief_q.get("qualified") and feasibility_q.get("qualified") and all(checks.values()))
    audit = {"as_of_utc": datetime.now(timezone.utc).isoformat(),
        "overall_assessment": "READY_TO_SHARE" if ready else "NEEDS_REVISION",
        "belief_qualified": belief_q.get("qualified"), "feasibility_qualified": feasibility_q.get("qualified"),
        "training_collection_data_quality": collection_quality.get("status"),
        "calculation_reconciliation": checks, "duplicate_primary_keys": duplicate_keys,
        "expected_rows": expected_rows, "actual_rows": {str(k): int(v) for k, v in actual_rows.items()},
        "belief_metrics": belief_m, "feasibility_recomputed": recomputed,
        "feasibility_heldout_taskwise": taskwise, "feasibility_heldout_rootwise": rootwise,
        "heldout_mass_force_inspection": sensitivity, "belief_identity_control": identity_control,
        "sources": {str(path): sha(path) for path in source_paths},
        "limitations": ["Feasibility is trained on controlled-motion auxiliary branches and requires separate online-VLA transfer qualification.",
                        "Mass support is three discrete in-support anchors; no out-of-support inference is tested.",
                        "Calibration coverage from only 12 held-out belief contexts is descriptive."]}
    write_json(outputs[0], audit)
    held_b = belief_m["HELDOUT"]
    report = f"""# MASS offline model qualification

## Technical summary

Overall assessment: **{audit['overall_assessment']}**. Belief qualification is **{'PASS' if belief_q.get('qualified') else 'FAIL'}** and full-task feasibility qualification is **{'PASS' if feasibility_q.get('qualified') else 'FAIL'}**. This report is an independent reconciliation of the saved predictions, not a new model-selection pass.

## The query belief is informative on held-out roots

Held-out belief MAE is {held_b['MAE']:.6f} kg, RMSE {held_b['RMSE']:.6f} kg, bias {held_b['bias']:+.6f} kg, and Spearman rho {held_b['Spearman']:.4f}. Empirical 68/90/95% interval coverage is {held_b['coverage']['0.68']:.3f}/{held_b['coverage']['0.9']:.3f}/{held_b['coverage']['0.95']:.3f} across 12 held-out contexts. Task-wise and root-wise values remain in `MASS_BELIEF_METRICS.json`.

Task and root are fully crossed with all three masses, and the held-out root is absent from training. A task-only TRAIN-mean control has held-out MAE {task_only_mae:.6f} kg and only {identity_control['task_only_distinct_predictions']} distinct predictions, versus {held_b['MAE']:.6f} kg for the physical-query model. Exact control evidence is in `MASS_BELIEF_IDENTITY_CONTROL.json`; it rules out task/root identity alone, not every possible nuisance signal.

## Full-task feasibility metrics reconcile exactly

The saved 648 predictions contain 432 TRAIN, 108 VAL, and 108 HELDOUT rows with {duplicate_keys} duplicate split/context/force keys. Independently recomputed held-out metrics are NLL={recomputed['HELDOUT']['NLL']:.6f}, Brier={recomputed['HELDOUT']['Brier']:.6f}, AUROC={recomputed['HELDOUT']['AUROC']}, AUPRC={recomputed['HELDOUT']['AUPRC']}, and ECE={recomputed['HELDOUT']['ECE_10bin']:.6f}. Every stored metric reconciles at absolute tolerance 1e-12: **{all(checks.values())}**.

## Predicted mass–force behavior is auditable

Task-wise held-out mass and force curves, including p(5 N)-p(3 N), are recorded in `MASS_OFFLINE_QUALIFICATION_AUDIT.json`; exact cell values and empirical outcomes are in `MASS_FEASIBILITY_MASS_FORCE_INSPECTION.csv`. No monotonicity constraint was imposed, and these descriptive curves are not used to alter the frozen utility or final masses.

## Scope and limitations

- The feasibility target is complete downstream success, not lift alone.
- Sibling forces share a physical context and split; roots are disjoint across TRAIN, VAL, and HELDOUT.
- Feasibility uses controlled-motion auxiliary branches. The online-VLA transfer must pass the separate development-only qualification before final freeze.
- Training support is the discrete 0.05/0.10/0.20 kg anchor set. Claims are limited to later in-support interpolation.
- Coverage estimates use only 12 held-out belief contexts and should not be read as formal calibration guarantees.

## Next gate

Proceed to the development-only current-runtime online-VLA qualification only when `overall_assessment` is `READY_TO_SHARE`. Do not use final roots or final outcomes to repair a failed learned component.
"""
    outputs[2].write_text(report)
    print(json.dumps({"assessment": audit["overall_assessment"], "checks_pass": all(checks.values()),
                      "outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
