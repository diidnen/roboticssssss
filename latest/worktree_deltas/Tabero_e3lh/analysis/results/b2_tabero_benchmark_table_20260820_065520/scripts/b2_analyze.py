#!/usr/bin/env python3
"""B2 offline analysis: F*_full, Fixed Robust, ΔF, task classification. No Isaac."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)

OFFICIAL = [0, 1, 2, 3, 5, 6, 7, 8, 9]
OBJECTS = {
    0: "alphabet_soup",
    1: "cream_cheese",
    2: "salad_dressing",
    3: "bbq_sauce",
    5: "tomato_sauce",
    6: "butter",
    7: "milk",
    8: "chocolate_pudding",
    9: "orange_juice",
}
MUS = [0.2, 0.5, 1.0]
MU_LABEL = {0.2: "low", 0.5: "mid", 1.0: "high"}
TAUS = (0.6, 0.8, 0.9)
PRIMARY_TAU = 0.8


def _fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_rows(*names):
    rows = []
    seen = set()
    for name in names:
        p = OUT / name
        if not p.exists() or p.stat().st_size == 0:
            continue
        with p.open() as f:
            for row in csv.DictReader(f):
                tid = row.get("trial_id")
                if tid:
                    if tid in seen:
                        continue
                    seen.add(tid)
                rows.append(row)
    return rows


def cell_sr(rows, task_id, mu, F, min_n=1):
    sub = [
        r
        for r in rows
        if int(float(r["task_id"])) == int(task_id)
        and abs(_fnum(r["friction"]) - mu) < 1e-6
        and abs(_fnum(r["force"]) - F) < 1e-6
    ]
    if len(sub) < min_n:
        return None, 0
    sr = sum(int(float(r["full_task_success"])) for r in sub) / len(sub)
    return sr, len(sub)


def forces_for_task(rows, task_id):
    fs = sorted({_fnum(r["force"]) for r in rows if int(float(r["task_id"])) == int(task_id)})
    return [f for f in fs if math.isfinite(f)]


def fstar(rows, task_id, mu, tau=PRIMARY_TAU):
    fs = forces_for_task(rows, task_id)
    qualified = []
    detail = {}
    for F in fs:
        sr, n = cell_sr(rows, task_id, mu, F)
        detail[F] = {"sr": sr, "n": n}
        if sr is not None and sr >= tau:
            qualified.append(F)
    if not qualified:
        return None, detail
    return min(qualified), detail


def classify_task(fstar_by_mu, robust, oracle_mean, ceiling, data_ready, traj_ok, mu_ok, n_cells):
    if not data_ready:
        return "BLOCKED"
    if not traj_ok:
        return "BLOCKED"
    if not mu_ok:
        return "BLOCKED"
    if n_cells == 0:
        return "BLOCKED"
    vals = [fstar_by_mu[m] for m in MUS]
    if all(v is None for v in vals):
        return "NO_FEASIBLE_RANGE"
    if any(v is None for v in vals):
        # some mu have no qualifying F
        known = [v for v in vals if v is not None]
        if known and max(known) != min(known):
            return "POSITIVE"  # still a friction-dependent gap, with a hole
        return "NO_FEASIBLE_RANGE"
    lo, mid, hi = vals
    if lo != hi:
        if robust is not None and oracle_mean is not None and (robust - oracle_mean) >= 0.5:
            return "POSITIVE"
        return "LOW_DECISION_VALUE" if abs(lo - hi) > 0 else "FIXED_ROBUST"
    # all F* equal
    fs = lo
    if ceiling is not None and ceiling >= 0.8 and fs == min(forces_available := [x for x in [lo, mid, hi] if x is not None]):
        # if F* equals the lowest tested force that was in the set... handled below
        pass
    if all(v == vals[0] for v in vals):
        # if that common F* is the lowest force that was tested AND SR at that force is high, FIXED_LOW
        return "FIXED_ROBUST"  # refined in caller with min tested force
    return "FIXED_ROBUST"


def main():
    rows = load_rows(
        "TASK1_ORACLE_SCAN.csv",
        "TASK7_ORACLE_SCAN.csv",
        "ALL_TASK_CHEAP_SCAN.csv",
        "POSITIVE_TASK_EXPANDED.csv",
    )
    # provenance
    prov_path = OUT / "TASK_DATA_PROVENANCE.csv"
    prov = {}
    if prov_path.exists():
        with prov_path.open() as f:
            for r in csv.DictReader(f):
                prov[int(r["task_id"])] = r

    oracles = []
    table = []
    fstar_by_task = {}
    robust_by_task = {}
    oracle_mean_by_task = {}
    saving_by_task = {}
    positive, fixed_low, fixed_robust, blocked = [], [], [], []
    eligible = []

    for tid in OFFICIAL:
        trows = [r for r in rows if int(float(r["task_id"])) == tid]
        fs = forces_for_task(trows, tid)
        data_ready = True
        if prov.get(tid):
            data_ready = prov[tid].get("local_path") not in ("", "MISSING") and prov[tid].get("reset", "PASS") != "FAIL"
        mu_ok = True
        if trows:
            mu_ok = all(int(float(r.get("friction_override_ok", "1"))) == 1 for r in trows)
        # trajectory reliability: some cell with full_task_success
        traj_ok = any(int(float(r["full_task_success"])) == 1 for r in trows) if trows else False
        fstar_mu = {}
        details = {}
        for mu in MUS:
            fst, det = fstar(trows, tid, mu, PRIMARY_TAU)
            fstar_mu[mu] = fst
            details[mu] = det
            for tau in TAUS:
                fst_t, _ = fstar(trows, tid, mu, tau)
                oracles.append(
                    {
                        "task_id": tid,
                        "object": OBJECTS[tid],
                        "friction": mu,
                        "tau": tau,
                        "Fstar_full": "" if fst_t is None else fst_t,
                        "status": "NO_QUALIFYING_FORCE_IN_TESTED_RANGE" if fst_t is None else "OK",
                        "n_forces_tested": len(fs),
                        "forces": ",".join(str(x) for x in fs),
                    }
                )
        # Fixed Robust: min F s.t. min_mu SR(F,mu) >= tau, using only mus that have data
        robust = None
        for F in fs:
            srs = []
            ok_all = True
            for mu in MUS:
                sr, n = cell_sr(trows, tid, mu, F)
                if sr is None:
                    ok_all = False
                    break
                srs.append(sr)
            if ok_all and min(srs) >= PRIMARY_TAU:
                robust = F
                break
        known_f = [fstar_mu[m] for m in MUS if fstar_mu[m] is not None]
        oracle_mean = float(np.mean(known_f)) if len(known_f) == 3 else (float(np.mean(known_f)) if known_f else None)
        dF = None if (robust is None or oracle_mean is None) else float(robust - oracle_mean)
        f_range = None if len(known_f) < 2 else float(max(known_f) - min(known_f))
        # ceiling: max full SR over cells
        ceiling = None
        if trows:
            by = defaultdict(list)
            for r in trows:
                by[(_fnum(r["friction"]), _fnum(r["force"]))].append(int(float(r["full_task_success"])))
            ceiling = max(sum(v) / len(v) for v in by.values()) if by else None

        n_cells = len({(_fnum(r["friction"]), _fnum(r["force"])) for r in trows})
        if not trows:
            cls = "BLOCKED"
        elif not traj_ok:
            cls = "BLOCKED"
        elif not mu_ok:
            cls = "BLOCKED"
        elif all(fstar_mu[m] is None for m in MUS):
            cls = "NO_FEASIBLE_RANGE"
        elif all(fstar_mu[m] is not None for m in MUS) and fstar_mu[0.2] != fstar_mu[1.0]:
            cls = "POSITIVE" if (dF is not None and dF >= 0.5) or (f_range is not None and f_range >= 1.0) else "LOW_DECISION_VALUE"
        elif all(fstar_mu[m] is not None for m in MUS) and fstar_mu[0.2] == fstar_mu[0.5] == fstar_mu[1.0]:
            common = fstar_mu[0.2]
            min_f = min(fs) if fs else None
            if min_f is not None and abs(common - min_f) < 1e-9:
                cls = "FIXED_LOW"
            else:
                cls = "FIXED_ROBUST"
        else:
            cls = "NO_FEASIBLE_RANGE"

        # eligibility
        elig = (
            cls == "POSITIVE"
            and data_ready
            and traj_ok
            and mu_ok
            and robust is not None
            and dF is not None
            and dF >= 0.5
            and f_range is not None
            and f_range >= 1.0
        )

        row = {
            "task_id": tid,
            "object": OBJECTS[tid],
            "data_ready": int(bool(trows) or data_ready),
            "n_trials": len(trows),
            "n_cells": n_cells,
            "low_mu_Fstar": "" if fstar_mu[0.2] is None else fstar_mu[0.2],
            "mid_mu_Fstar": "" if fstar_mu[0.5] is None else fstar_mu[0.5],
            "high_mu_Fstar": "" if fstar_mu[1.0] is None else fstar_mu[1.0],
            "fixed_robust": "" if robust is None else robust,
            "oracle_mean_F": "" if oracle_mean is None else round(oracle_mean, 3),
            "delta_F": "" if dF is None else round(dF, 3),
            "Fstar_range": "" if f_range is None else f_range,
            "full_sr_ceiling": "" if ceiling is None else round(ceiling, 3),
            "classification": cls,
            "eligible_main": int(elig),
            "traj_ok": int(traj_ok),
            "mu_override_ok": int(mu_ok),
        }
        table.append(row)
        fstar_by_task[str(tid)] = {str(m): fstar_mu[m] for m in MUS}
        robust_by_task[str(tid)] = robust
        oracle_mean_by_task[str(tid)] = oracle_mean
        saving_by_task[str(tid)] = dF
        if cls == "POSITIVE":
            positive.append(tid)
        elif cls == "FIXED_LOW":
            fixed_low.append(tid)
        elif cls in ("FIXED_ROBUST", "LOW_DECISION_VALUE"):
            fixed_robust.append(tid)
        elif cls == "BLOCKED":
            blocked.append(tid)
        if elig:
            eligible.append(tid)

    with (OUT / "FULLTASK_FORCE_ORACLES.csv").open("w", newline="") as f:
        if oracles:
            w = csv.DictWriter(f, fieldnames=list(oracles[0].keys()))
            w.writeheader()
            w.writerows(oracles)
        else:
            f.write("task_id,status\n")

    fields = list(table[0].keys())
    with (OUT / "TABERO_PHYSICS_DECISION_TABLE.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)

    md = [
        "# TABERO_PHYSICS_DECISION_TABLE",
        "",
        "Primary τ = 0.8. F* is min F with full-task SR ≥ τ. Empty F* = NO_QUALIFYING_FORCE_IN_TESTED_RANGE.",
        "",
        "| Task | Object | Data ready | Low μ F* | Mid μ F* | High μ F* | Fixed Robust | Oracle Mean F | ΔF | Full SR ceiling | Classification |",
        "| ---- | ------ | ---------: | -------: | -------: | --------: | -----------: | ------------: | -: | --------------: | -------------- |",
    ]
    for r in table:
        md.append(
            f"| {r['task_id']} | {r['object']} | {r['data_ready']} | {r['low_mu_Fstar']} | {r['mid_mu_Fstar']} | "
            f"{r['high_mu_Fstar']} | {r['fixed_robust']} | {r['oracle_mean_F']} | {r['delta_F']} | "
            f"{r['full_sr_ceiling']} | {r['classification']} |"
        )
    (OUT / "TABERO_PHYSICS_DECISION_TABLE.md").write_text("\n".join(md) + "\n")

    # plots
    def _plot_sr(task_id, fname):
        trows = [r for r in rows if int(float(r["task_id"])) == task_id]
        if not trows:
            return
        fs = forces_for_task(trows, task_id)
        plt.figure(figsize=(6.2, 4.2))
        for mu in MUS:
            ys, xs = [], []
            for F in fs:
                sr, n = cell_sr(trows, task_id, mu, F)
                if sr is None:
                    continue
                xs.append(F)
                ys.append(sr)
            if xs:
                plt.plot(xs, ys, marker="o", label=f"μ={mu:g}")
        plt.axhline(0.8, color="gray", ls="--", lw=1, label="τ=0.8")
        plt.xlabel("Force (N)")
        plt.ylabel("Full-task success rate")
        plt.title(f"Task {task_id} {OBJECTS[task_id]}")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS / fname, dpi=140)
        plt.close()

    _plot_sr(1, "success_vs_force_task1.png")
    _plot_sr(7, "success_vs_force_task7.png")

    # fstar heatmap-like bars
    tids = [r["task_id"] for r in table if r["n_trials"] > 0]
    if tids:
        x = np.arange(len(tids))
        width = 0.25
        plt.figure(figsize=(8.5, 4.2))
        for i, mu in enumerate(MUS):
            ys = []
            for tid in tids:
                v = fstar_by_task[str(tid)][str(mu)]
                ys.append(np.nan if v is None else v)
            plt.bar(x + (i - 1) * width, ys, width, label=f"μ={mu:g} ({MU_LABEL[mu]})")
        plt.xticks(x, [f"{t}\n{OBJECTS[t]}" for t in tids], fontsize=8)
        plt.ylabel("F*_full (N)  τ=0.8")
        plt.title("Full-task minimum sufficient force")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS / "fstar_by_task_and_friction.png", dpi=140)
        plt.close()

        plt.figure(figsize=(8.5, 4.2))
        rob = [robust_by_task[str(t)] if robust_by_task[str(t)] is not None else np.nan for t in tids]
        ora = [oracle_mean_by_task[str(t)] if oracle_mean_by_task[str(t)] is not None else np.nan for t in tids]
        plt.bar(x - 0.2, rob, 0.4, label="Fixed Robust")
        plt.bar(x + 0.2, ora, 0.4, label="Oracle mean F*")
        plt.xticks(x, [f"{t}\n{OBJECTS[t]}" for t in tids], fontsize=8)
        plt.ylabel("Force (N)")
        plt.title("Fixed Robust vs Oracle mean force")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS / "robust_vs_oracle_force.png", dpi=140)
        plt.close()

        plt.figure(figsize=(8.5, 4.2))
        rng = []
        sav = []
        for t in tids:
            r = next(z for z in table if z["task_id"] == t)
            rng.append(np.nan if r["Fstar_range"] == "" else float(r["Fstar_range"]))
            sav.append(np.nan if r["delta_F"] == "" else float(r["delta_F"]))
        plt.bar(x - 0.2, rng, 0.4, label="F* range (max-min)")
        plt.bar(x + 0.2, sav, 0.4, label="ΔF robust-oracle")
        plt.xticks(x, [f"{t}\n{OBJECTS[t]}" for t in tids], fontsize=8)
        plt.ylabel("Newtons")
        plt.title("Decision-value summary")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS / "task_decision_value_summary.png", dpi=140)
        plt.close()

    summary = {
        "n_rows": len(rows),
        "positive": positive,
        "fixed_low": fixed_low,
        "fixed_robust": fixed_robust,
        "blocked": blocked,
        "eligible": eligible,
        "fstar_by_task": fstar_by_task,
        "robust_by_task": robust_by_task,
        "oracle_mean_by_task": oracle_mean_by_task,
        "saving_by_task": saving_by_task,
        "table": table,
    }
    (OUT / "ANALYSIS_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("positive", "fixed_low", "fixed_robust", "blocked", "eligible")}, indent=2))


if __name__ == "__main__":
    main()
