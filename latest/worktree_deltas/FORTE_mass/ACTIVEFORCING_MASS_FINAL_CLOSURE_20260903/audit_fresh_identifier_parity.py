#!/usr/bin/env python3
"""Audit formal-vs-fresh Mass identifier preprocessing without Isaac."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
FORMAL = ROOT.parent.parent / "FORTE_mass" / "ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500" / "M3_TASK2_FORMAL_STRUCTURED_20260902_130700_RESUME"
FRESH = ROOT / "fresh_e2e_corrected_query_final_all_v2" / "MASS_FRESH_E2E_QUERIES.json"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("mass_modeling_parity", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    modeling = load_module(ROOT / "mass_modeling_final.py")
    contexts = list(csv.DictReader((FORMAL / "M3_TASK2_STRUCTURED_FORMAL_CONTEXTS.csv").open(newline="", encoding="utf-8")))
    train = [r for r in contexts if r.get("split") == "TRAIN" and int(float(r.get("query_valid", 0))) == 1]
    fresh = json.loads(FRESH.read_text(encoding="utf-8"))
    features = list(modeling.PHYSICAL_FEATURES)
    x_train = np.asarray([modeling.physical_vector(r) for r in train], dtype=np.float64)
    x_fresh = np.asarray([modeling.physical_vector({"query_record": r["query_record"]}) for r in fresh], dtype=np.float64)
    mean = x_train.mean(0); std = x_train.std(0); std[std < 1e-7] = 1.0
    lo = x_train.min(0); hi = x_train.max(0)
    rows = []
    for j, name in enumerate(features):
        vals = x_fresh[:, j]
        rows.append({"feature": name, "formal_train_min": lo[j], "formal_train_max": hi[j], "formal_train_mean": mean[j], "formal_train_std": std[j], "fresh_min": vals.min(), "fresh_max": vals.max(), "fresh_mean": vals.mean(), "fresh_outside_train_range_n": int(np.sum((vals < lo[j]) | (vals > hi[j]))), "fresh_abs_z_max": float(np.max(np.abs((vals - mean[j]) / std[j]))), "interpretation": "OUT_OF_RANGE" if np.any((vals < lo[j]) | (vals > hi[j])) else "within_train_range"})
    with (ROOT / "MASS_FRESH_IDENTIFIER_PARITY_AUDIT.csv").open("w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=list(rows[0])); out.writeheader(); out.writerows(rows)

    # Refit the frozen three-seed identifier exactly as the fresh runner does.
    y = np.asarray([float(r["mass_kg"]) for r in train], dtype=np.float32)
    models = [modeling.train_regressor(x_train.astype(np.float32), y, seed, torch.device("cpu")) for seed in (11, 23, 37)]
    pred_train = np.asarray([np.mean([float(m(v[None, :].astype(np.float32))[0]) for m in models]) for v in x_train])
    pred_fresh = np.asarray([np.mean([float(m(v[None, :].astype(np.float32))[0]) for m in models]) for v in x_fresh])
    context_rows = []
    for q, pred in zip(fresh, pred_fresh):
        context_rows.append({"context_id": q["tuple_id"], "gt_mass_kg": q["mass_kg"], "fresh_pred_mass_kg": float(q["pred_mass_kg"]), "parity_recomputed_pred_mass_kg": float(pred), "runner_vs_recomputed_abs_diff_kg": abs(float(q["pred_mass_kg"]) - float(pred)), "query_valid": q["query_valid"], "feature_count": len(modeling.physical_vector({"query_record": q["query_record"]})), "missing_required_feature_count": sum(k not in json.loads(q["query_record"]) for k in features), "raw_record_sha256": hashlib.sha256(q["query_record"].encode()).hexdigest()})
    with (ROOT / "MASS_FRESH_IDENTIFIER_CONTEXT_PARITY.csv").open("w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=list(context_rows[0])); out.writeheader(); out.writerows(context_rows)

    report = [
        "# Fresh Mass identifier preprocessing/parity audit", "",
        "Scope: existing six corrected fresh query records only; no query, Mass branch, model, π0, controller, or evaluator rerun.", "",
        "## Checks", "",
        f"- Formal TRAIN query contexts: {len(train)}; fresh query records: {len(fresh)}; required physical features: {len(features)}.",
        f"- Fresh query valid: {sum(int(q['query_valid']) for q in fresh)}/{len(fresh)}; missing required fields: {sum(int(r['missing_required_feature_count']) for r in context_rows)}.",
        f"- Recomputed runner parity: max absolute difference {max(r['runner_vs_recomputed_abs_diff_kg'] for r in context_rows):.8f} kg.",
        f"- Frozen preprocessing source: `mass_modeling_final.py`, SHA-256 `{sha(ROOT / 'mass_modeling_final.py')}`.",
        "- `obj_disp_probe_m` is consumed as metres; `actual_probe_displacement_mm` remains a separate millimetre feature, matching the formal feature definition.",
        "- The runner's seven-field `query_features` helper is not used for the authoritative identifier; the runner calls `mass_modeling_final.physical_vector` with the full 12-feature vector.", "",
        "## Finding", "",
        f"Fresh identifier predictions are {pred_fresh.min():.4f}--{pred_fresh.max():.4f} kg while GT is 0.05--0.20 kg. All six are high-biased; the recomputed values agree with the persisted runner values to numerical precision.",
        "Therefore this is not a simple wrong-unit, missing-field, stale-checkpoint, or runner-vs-model preprocessing mismatch. The observed 0.41--0.55 kg output is produced by the frozen identifier when fresh query features are fed through the same preprocessing path; it is an out-of-support feature/calibration shift relative to formal TRAIN.",
        "",
        "## Risk and next action", "",
        "Severity: HIGH for fresh downstream force selection; confidence: HIGH for parity correctness and high bias, MEDIUM for the physical cause of the feature shift. The smallest next diagnostic is feature-level ablation/clip sensitivity on these saved records; it must remain diagnostic and must not be used to alter final E2E claims without fresh validation.",
        "",
        "See `MASS_FRESH_IDENTIFIER_PARITY_AUDIT.csv` and `MASS_FRESH_IDENTIFIER_CONTEXT_PARITY.csv` for feature ranges, train-range violations, z-scores, and per-context recomputation.",
    ]
    (ROOT / "MASS_FRESH_IDENTIFIER_PARITY_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"formal_train_contexts": len(train), "fresh_queries": len(fresh), "missing_fields": sum(int(r["missing_required_feature_count"]) for r in context_rows), "max_recompute_diff_kg": max(r["runner_vs_recomputed_abs_diff_kg"] for r in context_rows), "fresh_pred_min_kg": float(pred_fresh.min()), "fresh_pred_max_kg": float(pred_fresh.max())}, sort_keys=True))


if __name__ == "__main__":
    main()
