#!/usr/bin/env python3
"""Freeze OLD Direct calibration and boundary metrics on observed TRAIN/DEV."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "BOUNDARY_AWARE_FULL_TASK_FEASIBILITY_DATA_CLOSURE_20260902"
DATA = OUT / "AUTHORITATIVE_720_BOUNDARY_LABELS.csv"
EPS = 1e-6
N_BINS = 10


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = p.astype(float)
    y = y.astype(int)
    brier = float(np.mean((p - y) ** 2))
    p_nll = np.clip(p, EPS, 1 - EPS)
    nll = float(-np.mean(y * np.log(p_nll) + (1 - y) * np.log(1 - p_nll)))
    bins = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
    ece = 0.0
    for b in range(N_BINS):
        mask = bins == b
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    if len(np.unique(y)) == 2:
        # Mann-Whitney form of AUROC with average ranks for ties.
        ranks = pd.Series(p).rank(method="average").to_numpy(float)
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    else:
        auc = math.nan
    return {"n": len(y), "positive_rate": float(y.mean()), "brier": brier, "ece_10_equal_width": ece, "nll": nll, "auroc": auc}


def main() -> None:
    d = pd.read_csv(DATA)
    d["p"] = d.p_D_OOF_ensemble.astype(float)
    d["y"] = d.full_task_success.astype(int)
    rows = []
    for task, q in d.groupby("task", sort=True):
        rows.append({"scope": "TASK", "task": int(task), **metrics(q.y.to_numpy(), q.p.to_numpy())})
    rows.append({"scope": "POOLED", "task": "ALL", **metrics(d.y.to_numpy(), d.p.to_numpy())})
    task_metrics = [r for r in rows if r["scope"] == "TASK"]
    rows.append({
        "scope": "MACRO_TASK", "task": "MACRO", "n": sum(r["n"] for r in task_metrics),
        **{k: float(np.nanmean([r[k] for r in task_metrics])) for k in ["positive_rate", "brier", "ece_10_equal_width", "nll", "auroc"]},
    })
    pd.DataFrame(rows).to_csv(OUT / "OLD_DIRECT_CALIBRATION_BASELINE.csv", index=False)

    bin_rows = []
    for scope, q in [(str(int(t)), x) for t, x in d.groupby("task", sort=True)] + [("ALL", d)]:
        q = q.copy()
        q["bin"] = np.minimum((q.p * N_BINS).astype(int), N_BINS - 1)
        for b in range(N_BINS):
            z = q[q.bin == b]
            bin_rows.append({
                "task": scope, "bin": b, "lower": b / N_BINS, "upper": (b + 1) / N_BINS,
                "n": len(z), "mean_predicted_probability": z.p.mean() if len(z) else np.nan,
                "empirical_success_rate": z.y.mean() if len(z) else np.nan,
                "calibration_gap": (z.p.mean() - z.y.mean()) if len(z) else np.nan,
            })
    pd.DataFrame(bin_rows).to_csv(OUT / "OLD_DIRECT_RELIABILITY_BINS.csv", index=False)

    pair_rows = []
    boundary_rows = []
    for (task, cid), q in d.groupby(["task", "context_id"], sort=True):
        cells = q.groupby("force_N", as_index=False).agg(empirical_success_rate=("y", "mean"), p=("p", "mean"), repeats=("y", "size"))
        fails = cells[cells.empirical_success_rate == 0]
        succs = cells[cells.empirical_success_rate == 1]
        for f in fails.itertuples():
            for s in succs[succs.force_N > f.force_N].itertuples():
                pair_rows.append({
                    "task": int(task), "context_id": cid, "failing_low_force_N": f.force_N,
                    "successful_high_force_N": s.force_N, "p_fail_low": f.p, "p_success_high": s.p,
                    "probability_gap_high_minus_low": s.p - f.p,
                    "ranking_correct": float(s.p > f.p), "ranking_tie": float(s.p == f.p),
                })
        # Boundary MAE is defined only for a clean unanimous empirical bracket.
        high_fail = float(fails.force_N.max()) if len(fails) else np.nan
        low_succ = float(succs.force_N.min()) if len(succs) else np.nan
        clean = bool(np.isfinite(high_fail) and np.isfinite(low_succ) and high_fail < low_succ)
        predicted_safe = cells[cells.p >= 0.5]
        predicted_transition = float(predicted_safe.force_N.min()) if len(predicted_safe) else np.nan
        empirical_mid = (high_fail + low_succ) / 2 if clean else np.nan
        boundary_rows.append({
            "task": int(task), "context_id": cid, "empirical_boundary_identifiable": int(clean),
            "empirical_highest_unanimous_failure_N": high_fail,
            "empirical_lowest_unanimous_success_N": low_succ,
            "empirical_transition_midpoint_N": empirical_mid,
            "predicted_transition_p_ge_0p5_N": predicted_transition,
            "boundary_mae_N": abs(predicted_transition - empirical_mid) if clean and np.isfinite(predicted_transition) else np.nan,
            "predicted_transition_missing": int(not np.isfinite(predicted_transition)),
        })
    pairs = pd.DataFrame(pair_rows)
    boundaries = pd.DataFrame(boundary_rows)
    pairs.to_csv(OUT / "OLD_DIRECT_BOUNDARY_PAIR_DIAGNOSTICS.csv", index=False)
    boundaries.to_csv(OUT / "OLD_DIRECT_BOUNDARY_MAE.csv", index=False)

    ranking_rows = []
    for task in [0, 1, 5, 6]:
        p = pairs[pairs.task == task]
        b = boundaries[(boundaries.task == task) & (boundaries.empirical_boundary_identifiable == 1)]
        ranking_rows.append({
            "scope": "TASK", "task": task, "ordered_failure_success_pairs": len(p),
            "force_ranking_accuracy": p.ranking_correct.mean() if len(p) else np.nan,
            "mean_probability_gap": p.probability_gap_high_minus_low.mean() if len(p) else np.nan,
            "negative_or_zero_probability_gap_rate": (p.probability_gap_high_minus_low <= 0).mean() if len(p) else np.nan,
            "clean_boundary_contexts": len(b), "boundary_mae_N": b.boundary_mae_N.mean() if len(b) else np.nan,
        })
    rt = pd.DataFrame(ranking_rows)
    macro = {"scope": "MACRO_TASK", "task": "MACRO", "ordered_failure_success_pairs": int(rt.ordered_failure_success_pairs.sum())}
    for col in ["force_ranking_accuracy", "mean_probability_gap", "negative_or_zero_probability_gap_rate", "boundary_mae_N"]:
        macro[col] = float(rt[col].mean())
    macro["clean_boundary_contexts"] = int(rt.clean_boundary_contexts.sum())
    rt = pd.concat([rt, pd.DataFrame([macro])], ignore_index=True)
    rt.to_csv(OUT / "OLD_DIRECT_BOUNDARY_METRICS.csv", index=False)

    t6 = d[(d.task == 6) & (d.force_N >= 3.01) & (d.force_N <= 3.06)].copy()
    t6[["branch_id", "context_id", "root_id", "friction_band", "mu_GT", "force_N", "repeat", "p", "y", "local_lift_success", "failure_stage_audited"]].to_csv(OUT / "TASK6_3P01_3P06_OLD_DIRECT.csv", index=False)

    calib = pd.DataFrame(rows)
    md = [
        "# OLD Direct boundary-calibration baseline", "",
        "Status: `FROZEN_BEFORE_NEW_BOUNDARY_DATA`", "",
        "All metrics use frozen root-held-out OOF `p_D_OOF_ensemble` and observed TRAIN/DEV outcomes. ECE uses 10 equal-width bins. NLL clips probabilities only for numerical evaluation at 1e-6. Force ranking compares every unanimously failing lower-force cell with every unanimously successful higher-force cell in the same context.", "",
        "Boundary MAE is deliberately conservative: it is reported only when a context has a clean unanimous bracket. The empirical point is the midpoint between highest unanimous failure and lowest unanimous success; the predicted transition is the lowest candidate with p>=0.5. Censored or overlapping contexts are excluded, not imputed.", "",
        "| Task | Brier | ECE | NLL | AUROC | Ranking accuracy | Mean p gap | Clean boundaries | Boundary MAE (N) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cal_by = calib[calib.scope == "TASK"].set_index("task")
    rank_by = rt[rt.scope == "TASK"].set_index("task")
    for task in [0, 1, 5, 6]:
        c, r = cal_by.loc[task], rank_by.loc[task]
        fmt = lambda x: "NA" if pd.isna(x) else f"{x:.4f}"
        md.append(f"| {task} | {fmt(c.brier)} | {fmt(c.ece_10_equal_width)} | {fmt(c.nll)} | {fmt(c.auroc)} | {fmt(r.force_ranking_accuracy)} | {fmt(r.mean_probability_gap)} | {int(r.clean_boundary_contexts)} | {fmt(r.boundary_mae_N)} |")
    md += [
        "", "## Task6 low-force diagnosis", "",
        f"The exact 3.01–3.06 N slice contains {len(t6)} branches, {int((t6.y == 0).sum())} full-task failures, and mean predicted success {t6.p.mean():.4f}. This is the preregistered slice for testing whether boundary augmentation lowers false confidence; no threshold or training change has been made.", "",
        "These baseline metrics do not establish that new data repairs Direct. They freeze the comparison target for OLD DATA ONLY versus OLD + BOUNDARY DATA.", "",
        "Task1 retains the 140/180 reconstructed full-task label caveat.",
    ]
    (OUT / "DIRECT_BASELINE_BOUNDARY_REPORT.md").write_text("\n".join(md) + "\n")
    (OUT / "DIRECT_BASELINE_METRIC_DEFINITIONS.json").write_text(json.dumps({
        "status": "FROZEN_PRE_BOUNDARY_COLLECTION", "ece": "10 equal-width probability bins",
        "nll_clip": EPS, "ranking": "unanimous fail lower force vs unanimous success higher force within context",
        "boundary_empirical": "midpoint(highest unanimous failure, lowest unanimous success), clean brackets only",
        "boundary_predicted": "lowest archived candidate with p_D >= 0.5", "task1_caveat": "140/180 reconstructed full-task labels",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
