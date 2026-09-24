#!/usr/bin/env python3
"""Attribute hard-search failures to H8 representation, world model, or evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_direct_worldmodel_verifier as base


def predict(model, x: np.ndarray, xm: np.ndarray, xs: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        z = model(torch.tensor((x - xm) / xs, dtype=torch.float32, device=device))
        return torch.sigmoid(z).cpu().numpy()


def load_evaluator(path: Path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = base.OutcomeEvaluator().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, np.asarray(ck["x_mean"], np.float32), np.asarray(ck["x_std"], np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    args = ap.parse_args()
    out = args.artifact
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnp = base.load("gnp_failure_attr", base.FORTE / "gnp_style_continuous.py")
    tpi = base.load("tpi_failure_attr", gnp.TPI_CODE)
    cf = base.load("cf_failure_attr", gnp.CF_CODE)
    fixed_mod = base.load("fixed_failure_attr", base.FORTE / "train_fixed_scene_exact.py")
    train, dev, _, _ = fixed_mod.build_population(base.FIXED, gnp, tpi, cf)
    y_train = np.asarray([t.outcome for t in train], np.float32)
    y_dev = np.asarray([t.outcome for t in dev], np.float32)
    real_train = np.stack([base.trajectory_feature(t.state[1:base.H+1]) for t in train])
    real_dev = np.stack([base.trajectory_feature(t.state[1:base.H+1]) for t in dev])
    score_rows = []
    error_rows = []
    training_rows = []
    matrix_rows = []
    for seed in base.SEEDS:
        world_path = base.FIXED / f"FIXED_SCENE_JOINT_seed{seed}.pt"
        world_ck = torch.load(world_path, map_location=device, weights_only=False)
        norm = tuple(np.asarray(world_ck["normalization"][k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
        world = tpi.ShortHorizonPhysicsGRU(17, 54, base.H).to(device)
        world.load_state_dict(world_ck["physics_state_dict"])
        world.eval()
        pred_train_state = [cf.pred_state(world, t, t.force, norm, tpi, device) for t in train]
        pred_dev_state = [cf.pred_state(world, t, t.force, norm, tpi, device) for t in dev]
        pred_train = np.stack([base.trajectory_feature(x) for x in pred_train_state])
        pred_dev = np.stack([base.trajectory_feature(x) for x in pred_dev_state])

        pred_eval, pxm, pxs = load_evaluator(out / f"TRAJECTORY_ONLY_OUTCOME_EVALUATOR_seed{seed}.pt", device)
        real_eval, rxm, rxs, hist = base.fit_evaluator(real_train, y_train, seed, device)
        for row in hist:
            training_rows.append({"evaluator_train_input": "REAL_H8", **row})
        channels = {
            "PRED_EVAL_ON_PRED": predict(pred_eval, pred_dev, pxm, pxs, device),
            "PRED_EVAL_ON_REAL": predict(pred_eval, real_dev, pxm, pxs, device),
            "REAL_EVAL_ON_REAL": predict(real_eval, real_dev, rxm, rxs, device),
            "REAL_EVAL_ON_PRED": predict(real_eval, pred_dev, rxm, rxs, device),
        }
        for name, probs in channels.items():
            matrix_rows.append({"seed": seed, "input_evaluator": name, **base.probability_metrics(y_dev, probs)})
        ys = norm[3]
        for i, tr in enumerate(dev):
            row = {
                "seed": seed,
                "branch_id": tr.branch_id,
                "context_id": tr.context_id,
                "task": int(tr.task),
                "force_N": float(tr.force),
                "repeat": 1 if "_R1_" in tr.branch_id else 2,
                "actual_success": int(tr.outcome),
            }
            row.update({name.lower(): float(probs[i]) for name, probs in channels.items()})
            score_rows.append(row)
            pred = pred_dev_state[i]
            actual = tr.state[1:base.H+1]
            mask = tr.mask[1:base.H+1]
            standardized = np.abs(pred - actual) / np.maximum(ys, 1e-6)
            valid_count = np.maximum(mask.sum(0), 1.0)
            channel_mae = (standardized * mask).sum(0) / valid_count
            erow = {
                "seed": seed,
                "branch_id": tr.branch_id,
                "context_id": tr.context_id,
                "force_N": float(tr.force),
                "repeat": row["repeat"],
                "actual_success": int(tr.outcome),
                "trajectory_standardized_MAE": float((standardized * mask).sum() / max(mask.sum(), 1.0)),
            }
            for j, name in enumerate(tpi.STATE_NAMES):
                erow[f"mae_{name}"] = float(channel_mae[j])
            error_rows.append(erow)
    scores = pd.DataFrame(score_rows)
    errors = pd.DataFrame(error_rows)
    ensemble = scores.groupby(["branch_id", "context_id", "task", "force_N", "repeat", "actual_success"], as_index=False).agg(
        pred_eval_on_pred=("pred_eval_on_pred", "mean"),
        pred_eval_on_real=("pred_eval_on_real", "mean"),
        real_eval_on_real=("real_eval_on_real", "mean"),
        real_eval_on_pred=("real_eval_on_pred", "mean"),
    )
    err_ensemble = errors.groupby(["branch_id", "context_id", "force_N", "repeat", "actual_success"], as_index=False).mean(numeric_only=True)
    ensemble = ensemble.merge(err_ensemble, on=["branch_id", "context_id", "force_N", "repeat", "actual_success"], validate="one_to_one")
    for name in ["PRED_EVAL_ON_PRED", "PRED_EVAL_ON_REAL", "REAL_EVAL_ON_REAL", "REAL_EVAL_ON_PRED"]:
        col = name.lower()
        matrix_rows.append({"seed": "ensemble", "input_evaluator": name, **base.probability_metrics(y_dev, ensemble[col].to_numpy())})

    cell = ensemble.groupby(["context_id", "task", "force_N"], as_index=False).agg(
        actual_success_rate=("actual_success", "mean"),
        pred_eval_on_pred=("pred_eval_on_pred", "mean"),
        pred_eval_on_real=("pred_eval_on_real", "mean"),
        real_eval_on_real=("real_eval_on_real", "mean"),
        real_eval_on_pred=("real_eval_on_pred", "mean"),
        trajectory_standardized_MAE=("trajectory_standardized_MAE", "mean"),
    )
    channel_cols = [c for c in ensemble.columns if c.startswith("mae_")]
    channel_cell = ensemble.groupby(["context_id", "force_N"], as_index=False)[channel_cols].mean()
    cell = cell.merge(channel_cell, on=["context_id", "force_N"], validate="one_to_one")
    hard = pd.read_csv(out / "DIRECT_WORLD_MODEL_HARD_SEARCH_SELECTIONS.csv")
    chosen = hard[hard.policy == "HARD_SEARCH_WITH_EXPLICIT_MAX_FALLBACK"][
        ["context_id", "direct_proposal_force_N", "selected_force_N", "evaluator_binary_success", "actual_success_rate"]
    ]
    gates = pd.read_csv(out / "DIRECT_WORLD_MODEL_HARD_SEARCH_GATE.csv")
    relevant_contexts = set(gates.loc[gates.any_higher_binary_success == 0, "context_id"])
    relevant_contexts.update(chosen.loc[chosen.actual_success_rate < 1, "context_id"])
    case_rows = []
    for cid in sorted(relevant_contexts):
        cmeta = chosen[chosen.context_id == cid].iloc[0]
        forces = cell[cell.context_id == cid].sort_values("force_N")
        selected = forces.iloc[(forces.force_N - float(cmeta.selected_force_N)).abs().argmin()]
        any_success = bool((forces.actual_success_rate > 0).any())
        all_repeat_agree = bool(not ((forces.actual_success_rate > 0) & (forces.actual_success_rate < 1)).any())
        y = float(selected.actual_success_rate)
        prr = float(selected.real_eval_on_real)
        rrp = float(selected.real_eval_on_pred)
        ppp = float(selected.pred_eval_on_pred)
        if 0.0 < y < 1.0:
            attribution = "STOCHASTIC_REPEAT_AMBIGUITY"
        elif y == 1.0 and ppp <= 0.5:
            if prr <= 0.5:
                attribution = "H8_REAL_TRAJECTORY_REPRESENTATION_OR_REAL_EVALUATOR"
            elif rrp <= 0.5:
                attribution = "WORLD_MODEL_PREDICTED_TRAJECTORY_SHIFT"
            else:
                attribution = "PREDICTED_TRAJECTORY_EVALUATOR_FALSE_NEGATIVE"
        elif y == 0.0 and ppp > 0.5:
            if prr > 0.5:
                attribution = "H8_REAL_TRAJECTORY_REPRESENTATION_OR_REAL_EVALUATOR"
            elif rrp > 0.5:
                attribution = "WORLD_MODEL_PREDICTED_TRAJECTORY_SHIFT"
            else:
                attribution = "PREDICTED_TRAJECTORY_EVALUATOR_FALSE_POSITIVE"
        elif not bool(cmeta.evaluator_binary_success) and any_success:
            attribution = "SEARCH_NO_VALID_DESPITE_REAL_SUCCESS_ELSEWHERE"
        elif not any_success:
            attribution = "FORCE_SUPPORT_HAS_NO_OBSERVED_SUCCESS"
        else:
            attribution = "NO_FINAL_ERROR"
        top = sorted(((float(selected[c]), c.removeprefix("mae_")) for c in channel_cols), reverse=True)[:3]
        case_rows.append({
            "context_id": cid,
            "task": int(selected.task),
            "selected_force_N": float(selected.force_N),
            "selected_actual_success_rate": y,
            "any_observed_success_in_support": int(any_success),
            "repeat_deterministic_over_support": int(all_repeat_agree),
            "pred_eval_on_pred": ppp,
            "real_eval_on_real": prr,
            "real_eval_on_pred": rrp,
            "pred_eval_on_real": float(selected.pred_eval_on_real),
            "trajectory_standardized_MAE": float(selected.trajectory_standardized_MAE),
            "top_error_channels": json.dumps(top),
            "attribution": attribution,
        })
    write = {
        "WORLD_MODEL_FAILURE_SCORE_MATRIX.csv": pd.DataFrame(matrix_rows),
        "WORLD_MODEL_FAILURE_BRANCH_FORENSIC.csv": ensemble,
        "WORLD_MODEL_FAILURE_CELL_FORENSIC.csv": cell,
        "WORLD_MODEL_FAILURE_CASE_ATTRIBUTION.csv": pd.DataFrame(case_rows),
        "REAL_TRAJECTORY_EVALUATOR_TRAINING.csv": pd.DataFrame(training_rows),
    }
    for name, df in write.items():
        df.to_csv(out / name, index=False)
    matrix = pd.DataFrame(matrix_rows)
    ens = matrix[matrix.seed.astype(str) == "ensemble"].set_index("input_evaluator")
    cases = pd.DataFrame(case_rows)
    counts = cases.attribution.value_counts().to_dict()
    report = "# World-model hard-search failure attribution\n\n"
    report += "This forensic separates H8 outcome observability, predicted-trajectory error, and evaluator error using the 2x2 matrix of REAL/PRED trajectory inputs and REAL/PRED-trained evaluators. No controller or frozen model was changed.\n\n"
    report += "| Input / evaluator | AUROC | Brier | NLL |\n|---|---:|---:|---:|\n"
    labels = {
        "REAL_EVAL_ON_REAL": "REAL trajectory → REAL-trained evaluator",
        "REAL_EVAL_ON_PRED": "PRED trajectory → REAL-trained evaluator",
        "PRED_EVAL_ON_REAL": "REAL trajectory → PRED-trained evaluator",
        "PRED_EVAL_ON_PRED": "PRED trajectory → PRED-trained evaluator",
    }
    for key, label in labels.items():
        r = ens.loc[key]
        report += f"| {label} | {r.AUROC:.3f} | {r.Brier:.3f} | {r.NLL:.3f} |\n"
    report += f"\nCase attribution counts: `{json.dumps(counts, sort_keys=True)}`. See `WORLD_MODEL_FAILURE_CASE_ATTRIBUTION.csv` for exact contexts and top trajectory-error channels.\n"
    (out / "WORLD_MODEL_FAILURE_ATTRIBUTION_REPORT.md").write_text(report, encoding="utf-8")
    paths = [out / x for x in list(write) + ["WORLD_MODEL_FAILURE_ATTRIBUTION_REPORT.md"]]
    (out / "WORLD_MODEL_FAILURE_ATTRIBUTION_SHA256SUMS.txt").write_text(
        "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in paths), encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", "device": str(device), "matrix": matrix[matrix.seed.astype(str) == "ensemble"].to_dict("records"), "attribution_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
