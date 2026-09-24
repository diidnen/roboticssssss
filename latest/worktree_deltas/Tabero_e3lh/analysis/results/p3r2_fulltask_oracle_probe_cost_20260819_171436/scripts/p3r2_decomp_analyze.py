#!/usr/bin/env python3
"""P3-R2 Phase B: probe vs belief SR decomposition + plots."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
PLOT = OUT / "plots"
PLOT.mkdir(exist_ok=True)
MUS = (0.2, 0.5, 1.0)


def _f(x, d=np.nan):
    try:
        if x in ("", "None", None):
            return d
        return float(x)
    except Exception:
        return d


def write_csv(name, rows):
    if not rows:
        (OUT / name).write_text("")
        return
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sr_by_mu(rows, method):
    out = {}
    for mu in MUS:
        rs = [r for r in rows if r["method"] == method and abs(_f(r["friction"]) - mu) < 1e-6]
        n = len(rs)
        out[str(mu)] = {
            "n": n,
            "full": float(np.mean([_f(r["full_task_success"]) for r in rs])) if n else np.nan,
            "lift": float(np.mean([_f(r["lift_success"]) for r in rs])) if n else np.nan,
            "chosen": float(np.nanmean([_f(r["chosen_force"]) for r in rs])) if n else np.nan,
            "mean_force": float(np.nanmean([_f(r["mean_force"]) for r in rs])) if n else np.nan,
            "peak_force": float(np.nanmean([_f(r["peak_force"]) for r in rs])) if n else np.nan,
            "integrated": float(np.nanmean([_f(r["integrated_force"]) for r in rs])) if n else np.nan,
            "timeout": float(np.mean([_f(r["timeout"]) for r in rs])) if n else np.nan,
        }
    return out


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fstar = json.loads((OUT / "FSTAR_FULL.json").read_text())
    fmap = {float(k): float(v) for k, v in fstar["fstar_by_friction"].items()}
    robust = float(fstar["fixed_robust"])

    grid = list(csv.DictReader((OUT / "FULLTASK_FORCE_GRID.csv").open()))
    decomp = list(csv.DictReader((OUT / "DECOMP_RESULTS.csv").open())) if (OUT / "DECOMP_RESULTS.csv").exists() else []

    # Reconstruct no-probe methods from the grid using F*_full / robust
    reconstructed = []
    for r in grid:
        mu = _f(r["friction"])
        F = _f(r["f_cmd_final"])
        if abs(F - robust) < 1e-6:
            rr = dict(r); rr["method"] = "fixed_robust"; reconstructed.append(rr)
        if abs(F - fmap[mu]) < 1e-6:
            rr = dict(r); rr["method"] = "no_probe_oracle"; reconstructed.append(rr)

    all_rows = reconstructed + decomp
    by = defaultdict(list)
    for r in all_rows:
        by[r["method"]].append(r)

    write_csv("BASELINE_FIXED_ROBUST.csv", by.get("fixed_robust", []))
    write_csv("NO_PROBE_ORACLE.csv", by.get("no_probe_oracle", []))
    write_csv("PROBE_GT_FORCE.csv", by.get("probe_gt", []))
    write_csv("PROBE_BELIEF.csv", by.get("probe_belief", []))

    summaries = {m: sr_by_mu(all_rows, m) for m in ("fixed_robust", "no_probe_oracle", "probe_gt", "probe_belief")}
    (OUT / "DECOMP_SUMMARY.json").write_text(json.dumps(summaries, indent=2) + "\n")

    def overall(method, key="full"):
        vals = [summaries[method][str(mu)][key] for mu in MUS]
        return float(np.nanmean(vals))

    d_probe = overall("no_probe_oracle") - overall("probe_gt")
    d_belief = overall("probe_gt") - overall("probe_belief")
    decomp_rows = [{
        "no_probe_oracle_overall_full_sr": overall("no_probe_oracle"),
        "probe_gt_overall_full_sr": overall("probe_gt"),
        "probe_belief_overall_full_sr": overall("probe_belief"),
        "fixed_robust_overall_full_sr": overall("fixed_robust"),
        "probe_cost_sr_gap": d_probe,
        "belief_cost_sr_gap": d_belief,
        "probe_cost_dominates": int(d_probe > d_belief and d_probe >= 0.03),
        "belief_error_dominates": int(d_belief > d_probe and d_belief >= 0.03),
    }]
    for mu in MUS:
        decomp_rows[0][f"probe_cost_mu{mu:g}"] = summaries["no_probe_oracle"][str(mu)]["full"] - summaries["probe_gt"][str(mu)]["full"]
        decomp_rows[0][f"belief_cost_mu{mu:g}"] = summaries["probe_gt"][str(mu)]["full"] - summaries["probe_belief"][str(mu)]["full"]
    write_csv("PROBE_COST_DECOMPOSITION.csv", decomp_rows)

    errors = []
    for r in by.get("probe_belief", []):
        mu = _f(r["friction"])
        F = _f(r["chosen_force"])
        Fstar = fmap[mu]
        errors.append({
            "seed_idx": r["seed_idx"], "gt_mu": mu, "z_used": r.get("z_used", ""),
            "posterior_mu02": r.get("posterior_mu02", ""),
            "chosen_force": F, "fstar_full": Fstar,
            "exact": int(abs(F - Fstar) < 1e-6),
            "under_force": int(F < Fstar - 1e-6),
            "over_force": int(F > Fstar + 1e-6),
            "full_task_success": r["full_task_success"],
        })
    write_csv("DECISION_ERRORS.csv", errors)
    if errors:
        print("decision acc", float(np.mean([e["exact"] for e in errors])),
              "under", float(np.mean([e["under_force"] for e in errors])),
              "over", float(np.mean([e["over_force"] for e in errors])))

    # plots
    methods = ["fixed_robust", "no_probe_oracle", "probe_gt", "probe_belief"]
    labels = ["fixed robust", "no-probe oracle", "probe+GT", "probe+belief"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(MUS)); w = 0.18
    for i, m in enumerate(methods):
        ys = [summaries[m][str(mu)]["full"] for mu in MUS]
        ax.bar(x + (i - 1.5) * w, ys, width=w, label=labels[i])
    ax.set_xticks(x); ax.set_xticklabels([f"μ={mu:g}" for mu in MUS])
    ax.set_ylim(0, 1.05); ax.set_ylabel("full-task SR"); ax.legend(fontsize=8)
    ax.set_title("Probe vs belief decomposition")
    fig.tight_layout()
    fig.savefig(PLOT / "probe_cost_decomposition.png", dpi=140)
    fig.savefig(PLOT / "probe_cost_decomposition.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for i, m in enumerate(methods):
        ys = [summaries[m][str(mu)]["chosen"] for mu in MUS]
        ax.plot(MUS, ys, marker="o", label=labels[i])
    ax.set_xticks(MUS); ax.set_ylabel("mean selected force (N)")
    ax.set_xlabel("friction"); ax.legend(fontsize=8)
    ax.set_title("Selected force by method")
    fig.tight_layout()
    fig.savefig(PLOT / "selected_force_by_method.png", dpi=140)
    fig.savefig(PLOT / "selected_force_by_method.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for m, lab, mk in zip(methods, labels, "os^D"):
        ax.scatter([overall(m, "chosen")], [overall(m, "full")], marker=mk, s=70, label=lab)
    ax.set_xlabel("mean selected force (N)"); ax.set_ylabel("mean full-task SR")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8)
    ax.set_title("Success vs force")
    fig.tight_layout()
    fig.savefig(PLOT / "success_force_tradeoff.png", dpi=140)
    fig.savefig(PLOT / "success_force_tradeoff.pdf")
    plt.close(fig)

    distp = OUT / "PROBE_DISTURBANCE.csv"
    if distp.exists() and distp.stat().st_size > 0:
        drows = list(csv.DictReader(distp.open()))
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for succ, c, lab in [(1, "C2", "full success"), (0, "C3", "full fail")]:
            xs = [_f(r["pose_shift_m"]) * 1000 for r in drows if int(float(r["full_task_success"])) == succ]
            ax.hist(xs, bins=12, alpha=0.55, color=c, label=lab)
        ax.set_xlabel("probe object pose shift (mm)")
        ax.set_ylabel("count")
        ax.legend()
        ax.set_title("Probe pose shift vs full-task outcome")
        fig.tight_layout()
        fig.savefig(PLOT / "probe_object_pose_shift.png", dpi=140)
        fig.savefig(PLOT / "probe_object_pose_shift.pdf")
        plt.close(fig)

    print("probe_cost", d_probe, "belief_cost", d_belief)
    print(json.dumps({m: {k: summaries[m][k]["full"] for k in summaries[m]} for m in summaries}, indent=2))


if __name__ == "__main__":
    main()
