#!/usr/bin/env python3
"""P3 full-downstream tables, plots, verdict inputs. No neural nets."""
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
METHODS = ("fixed4", "fixed6", "oracle", "probe_belief")


def _f(x, d=np.nan):
    try:
        if x in ("", "None", None):
            return d
        return float(x)
    except Exception:
        return d


def _rows(path):
    p = OUT / path
    if not p.exists() or p.stat().st_size == 0:
        return []
    return list(csv.DictReader(p.open()))


def write_csv(name, rows):
    if not rows:
        (OUT / name).write_text("")
        return
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sr_table(rows, method):
    out = {}
    for mu in MUS:
        rs = [r for r in rows if r["method"] == method and abs(_f(r["friction"]) - mu) < 1e-6]
        n = len(rs)
        out[str(mu)] = {
            "n": n,
            "pick": float(np.mean([_f(r["pick_success"]) for r in rs])) if n else np.nan,
            "lift": float(np.mean([_f(r["lift_success"]) for r in rs])) if n else np.nan,
            "full": float(np.mean([_f(r["full_task_success"]) for r in rs])) if n else np.nan,
            "mean_force": float(np.nanmean([_f(r["mean_force"]) for r in rs])) if n else np.nan,
            "peak_force": float(np.nanmean([_f(r["peak_force"]) for r in rs])) if n else np.nan,
            "chosen": float(np.nanmean([_f(r["chosen_force"]) for r in rs])) if n else np.nan,
        }
    return out


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _rows("FULL_DOWNSTREAM_RESULTS.csv")
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["method"]].append(r)

    write_csv("BASELINE_FIXED4.csv", by_m.get("fixed4", []))
    write_csv("BASELINE_FIXED6.csv", by_m.get("fixed6", []))
    write_csv("BASELINE_ORACLE_PHYSICS.csv", by_m.get("oracle", []))
    write_csv("PROBE_BELIEF_RESULTS.csv", by_m.get("probe_belief", []))

    fd = []
    for r in by_m.get("probe_belief", []):
        gt = _f(r["friction"])
        F = _f(r["chosen_force"])
        oracle = 6.0 if abs(gt - 0.2) < 1e-6 else 4.0
        fd.append({
            "seed_idx": r["seed_idx"], "gt_mu": gt, "z_used": r.get("z_used", ""),
            "posterior_mu02": r.get("posterior_mu02", ""),
            "posterior_mu05": r.get("posterior_mu05", ""),
            "posterior_mu10": r.get("posterior_mu10", ""),
            "p_succ_4N": r.get("p_succ_4N", ""),
            "chosen_force": F, "oracle_force": oracle,
            "correct": int(abs(F - oracle) < 1e-6),
            "full_task_success": r["full_task_success"],
            "probe_failure": r.get("probe_failure", ""),
            "probe_duration_s": r.get("probe_duration_s", ""),
            "probe_path_mm": r.get("probe_path_mm", ""),
            "obj_disp_probe_m": r.get("obj_disp_probe_m", ""),
        })
    write_csv("FORCE_DECISIONS.csv", fd)

    cost = []
    pb = by_m.get("probe_belief", [])
    if pb:
        cost.append({
            "n": len(pb),
            "probe_failure_rate": float(np.mean([_f(r["probe_failure"]) for r in pb])),
            "mean_probe_duration_s": float(np.nanmean([_f(r["probe_duration_s"]) for r in pb])),
            "mean_probe_path_mm": float(np.nanmean([_f(r["probe_path_mm"]) for r in pb])),
            "mean_obj_disp_probe_mm": float(np.nanmean([_f(r["obj_disp_probe_m"]) * 1000 for r in pb])),
            "contact_lost_rate": float(np.mean([_f(r.get("contact_lost_probe", 0)) for r in pb])),
            "extra_task_time_s_vs_fixed6": float(
                np.nanmean([_f(r["steps"]) for r in pb]) * 0.05
                - np.nanmean([_f(r["steps"]) for r in by_m.get("fixed6", pb)]) * 0.05
            ) if by_m.get("fixed6") else np.nan,
        })
    write_csv("PROBE_COST_ANALYSIS.csv", cost)

    summaries = {m: sr_table(rows, m) for m in METHODS}
    (OUT / "DOWNSTREAM_SUMMARY.json").write_text(json.dumps(summaries, indent=2) + "\n")

    # plots
    if rows:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        x = np.arange(len(MUS))
        w = 0.18
        for i, m in enumerate(METHODS):
            ys = [summaries[m][str(mu)]["full"] for mu in MUS]
            ax.bar(x + (i - 1.5) * w, ys, width=w, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([f"μ={mu:g}" for mu in MUS])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("full task SR")
        ax.legend(fontsize=8)
        ax.set_title("P3 downstream success")
        fig.tight_layout()
        fig.savefig(PLOT / "downstream_success.png", dpi=140)
        fig.savefig(PLOT / "downstream_success.pdf")
        plt.close(fig)

        if fd:
            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            for mu, c in [(0.2, "C3"), (0.5, "C0"), (1.0, "C2")]:
                xs = [_f(r["chosen_force"]) for r in fd if abs(_f(r["gt_mu"]) - mu) < 1e-6]
                ax.scatter(np.random.default_rng(1).normal(mu, 0.03, size=len(xs)), xs, c=c, s=28, label=f"μ={mu:g}")
            ax.set_yticks([4, 6])
            ax.set_xlabel("GT friction (label only)")
            ax.set_ylabel("selected force (N)")
            ax.legend()
            ax.set_title("Probe+belief selected force")
            fig.tight_layout()
            fig.savefig(PLOT / "selected_force_by_friction.png", dpi=140)
            fig.savefig(PLOT / "selected_force_by_friction.pdf")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            for mu, c in [(0.2, "C3"), (0.5, "C0"), (1.0, "C2")]:
                xs = [_f(r["posterior_mu02"]) for r in fd if abs(_f(r["gt_mu"]) - mu) < 1e-6]
                ax.scatter(np.random.default_rng(2).normal(mu, 0.03, size=len(xs)), xs, c=c, s=28, label=f"μ={mu:g}")
            ax.axhline(0.2, ls="--", c="gray", lw=0.8, label="τ=0.8 ⇔ P(low)≤0.2")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("GT friction")
            ax.set_ylabel("P(μ=0.20 | z)")
            ax.legend(fontsize=8)
            ax.set_title("Posterior low-friction probability")
            fig.tight_layout()
            fig.savefig(PLOT / "posterior_by_friction.png", dpi=140)
            fig.savefig(PLOT / "posterior_by_friction.pdf")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        for m, mk in [("fixed4", "o"), ("fixed6", "s"), ("oracle", "^"), ("probe_belief", "D")]:
            fmean = np.nanmean([summaries[m][str(mu)]["chosen"] for mu in MUS])
            srmean = np.nanmean([summaries[m][str(mu)]["full"] for mu in MUS])
            ax.scatter([fmean], [srmean], marker=mk, s=70, label=m)
        ax.set_xlabel("mean selected grip command (N)")
        ax.set_ylabel("mean full-task SR")
        ax.set_xlim(3.5, 6.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.set_title("Force vs success trade-off")
        fig.tight_layout()
        fig.savefig(PLOT / "force_vs_success_tradeoff.png", dpi=140)
        fig.savefig(PLOT / "force_vs_success_tradeoff.pdf")
        plt.close(fig)

    print("eval analyze n", len(rows), {m: len(by_m[m]) for m in METHODS})


if __name__ == "__main__":
    main()
