#!/usr/bin/env python3
"""Phase A: full-task F* from the 4/5/6 N grid. Calibration-split success table only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
P3 = Path("/home/exouser/Tabero/analysis/results/p3_probe_belief_decision_20260819_104513")
MUS = (0.2, 0.5, 1.0)
FORCES = (4.0, 5.0, 6.0)
TAUS = (0.6, 0.8, 0.9)


def _f(x, d=np.nan):
    try:
        if x in ("", "None", None):
            return d
        return float(x)
    except Exception:
        return d


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = OUT / "FULLTASK_FORCE_GRID.csv"
    rows = list(csv.DictReader(p.open()))
    grid = []
    sr = {}
    lift = {}
    for mu in MUS:
        for F in FORCES:
            rs = [r for r in rows if abs(_f(r["friction"]) - mu) < 1e-6 and abs(_f(r["chosen_force"]) - F) < 1e-6]
            # grid method encodes force in chosen_force / f_cmd_final
            if not rs:
                rs = [r for r in rows if abs(_f(r["friction"]) - mu) < 1e-6 and abs(_f(r["f_cmd_final"]) - F) < 1e-6]
            n = len(rs)
            rec = {
                "friction": mu, "force_N": F, "n": n,
                "pick_sr": float(np.mean([_f(r["pick_success"]) for r in rs])) if n else np.nan,
                "lift_sr": float(np.mean([_f(r["lift_success"]) for r in rs])) if n else np.nan,
                "transport_sr": float(np.mean([_f(r["transport_retention"]) for r in rs])) if n else np.nan,
                "place_sr": float(np.mean([_f(r["place_success"]) for r in rs])) if n else np.nan,
                "full_sr": float(np.mean([_f(r["full_task_success"]) for r in rs])) if n else np.nan,
                "drop_rate": float(np.mean([_f(r["dropped"]) for r in rs])) if n else np.nan,
                "timeout_rate": float(np.mean([_f(r["timeout"]) for r in rs])) if n else np.nan,
                "mean_force": float(np.nanmean([_f(r["mean_force"]) for r in rs])) if n else np.nan,
                "peak_force": float(np.nanmean([_f(r["peak_force"]) for r in rs])) if n else np.nan,
            }
            grid.append(rec)
            sr[(mu, F)] = rec["full_sr"]
            lift[(mu, F)] = rec["lift_sr"]

    with (OUT / "FULLTASK_FORCE_GRID_SUMMARY.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
        w.writeheader(); w.writerows(grid)

    def fstar(tau, table):
        out = {}
        for mu in MUS:
            ok = [F for F in FORCES if table[(mu, F)] >= tau]
            out[str(mu)] = float(min(ok)) if ok else float(max(FORCES))
        return out

    fstar_08 = fstar(0.8, sr)
    robust = max(fstar_08.values())
    mapping = {
        "tau_primary": 0.8,
        "fstar_by_friction": fstar_08,
        "fstar_by_tau": {str(t): fstar(t, sr) for t in TAUS},
        "lift_fstar_tau0.8": fstar(0.8, lift),
        "fixed_robust": robust,
        "n_seeds": 20,
        "forces_N": list(FORCES),
    }
    (OUT / "FSTAR_FULL.json").write_text(json.dumps(mapping, indent=2) + "\n")

    lines = [
        "# Full-task force oracle (P3-R2 Phase A)",
        "",
        "F*_full(μ) = min { F ∈ {4,5,6} : P(full pick-place success | F, μ) ≥ τ }",
        "Primary τ = 0.8. Lift-based F* is reported only as a contrast; it is **not** the action target.",
        "",
        "| μ | 4 N full | 5 N full | 6 N full | F*_full (τ=0.8) | F*_full (τ=0.6) | F*_full (τ=0.9) | F*_lift (τ=0.8) |",
        "| - | -------: | -------: | -------: | --------------: | --------------: | --------------: | --------------: |",
    ]
    f06 = fstar(0.6, sr); f09 = fstar(0.9, sr); fl = fstar(0.8, lift)
    for mu in MUS:
        lines.append(
            f"| {mu:g} | {sr[(mu,4)]:.2f} | {sr[(mu,5)]:.2f} | {sr[(mu,6)]:.2f} | "
            f"{fstar_08[str(mu)]:g} | {f06[str(mu)]:g} | {f09[str(mu)]:g} | {fl[str(mu)]:g} |"
        )
    lines += [
        "",
        f"FIXED_ROBUST (covers all μ at τ=0.8) = **{robust:g} N**",
        "",
        "Belief success table below uses **even seeds only** (calibration). Eval seeds are not used to set τ.",
    ]
    (OUT / "FULLTASK_FORCE_ORACLE.md").write_text("\n".join(lines) + "\n")

    # calibration success table (even seeds)
    table = {}
    for F in FORCES:
        table[str(int(F))] = {}
        for mu in MUS:
            rs = [
                r for r in rows
                if abs(_f(r["friction"]) - mu) < 1e-6
                and abs(_f(r["f_cmd_final"]) - F) < 1e-6
                and int(float(r["seed_idx"])) % 2 == 0
            ]
            table[str(int(F))][str(mu)] = float(np.mean([_f(r["full_task_success"]) for r in rs])) if rs else 0.0

    p3 = json.loads((P3 / "BELIEF_MODEL.json").read_text())
    model = {
        "signal": "z_imbalance_peak",
        "probe": "A",
        "means": p3["means"],
        "stds": p3["stds"],
        "prior": p3["prior"],
        "tau": 0.8,
        "calib_seeds": "even_phaseA_grid",
        "heldout_seeds": "odd",
        "likelihood_source": "P3_FROZEN",
        "success_table": table,
        "success_table_source": "P3R2_PHASE_A_EVEN_SEEDS_FULL_TASK",
        "force_candidates": [int(F) for F in FORCES],
        "decision_rule": "min F in candidates s.t. sum_b P(full_success|F,mu) >= 0.8",
        "BELIEF_RULE_UPDATED_FOR_FULL_TASK_ORACLE": True,
        "fstar_full": fstar_08,
    }
    (OUT / "BELIEF_MODEL.json").write_text(json.dumps(model, indent=2) + "\n")

    plot = OUT / "plots"
    plot.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(FORCES))
    w = 0.22
    for i, mu in enumerate(MUS):
        ys = [sr[(mu, F)] for F in FORCES]
        ax.bar(x + (i - 1) * w, ys, width=w, label=f"μ={mu:g}")
    ax.axhline(0.8, ls="--", c="gray", lw=0.8, label="τ=0.8")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{F:g} N" for F in FORCES])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("full-task SR")
    ax.legend(fontsize=8)
    ax.set_title("Full-task SR vs grip command")
    fig.tight_layout()
    fig.savefig(plot / "full_sr_vs_force_by_friction.png", dpi=140)
    fig.savefig(plot / "full_sr_vs_force_by_friction.pdf")
    plt.close(fig)
    print(json.dumps(mapping, indent=2))
    print("success_table_even", table)


if __name__ == "__main__":
    main()
