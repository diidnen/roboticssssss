#!/usr/bin/env python3
"""P3 probe screening + optional discrete-Bayes calibration. No neural nets."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
PLOT = OUT / "plots"
PLOT.mkdir(exist_ok=True)

SIGS = [
    "z_marker_tang_peak", "z_marker_tang_mean", "z_marker_mean_peak", "z_marker_unloading",
    "z_marker_asym_peak", "z_marker_vel_abs_peak", "z_hm_mean",
    "z_f_meas_mean", "z_imbalance_peak", "z_ftan_peak",
]
PRIV = ["z_rel_xy_peak", "z_rel_z_min"]


def _f(x, d=np.nan):
    try:
        if x in ("", "None", None):
            return d
        return float(x)
    except Exception:
        return d


def auc_score(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    def auc(p, n):
        order = np.concatenate([n, p])
        r = np.argsort(np.argsort(order)) + 1
        n1, n0 = len(p), len(n)
        return float((r[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

    return max(auc(pos, neg), auc(-pos, -neg))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    if s < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / s)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = OUT / "PROBE_SCREENING_RESULTS.csv"
    if not p.exists():
        print("no screening csv")
        return
    rows = list(csv.DictReader(p.open()))
    by_probe = defaultdict(list)
    for r in rows:
        by_probe[r["probe"]].append(r)

    sep = []
    best = None
    for probe, rs in by_probe.items():
        low = [r for r in rs if abs(_f(r["friction"]) - 0.2) < 1e-6]
        high = [r for r in rs if abs(_f(r["friction"]) - 0.2) > 1e-6]
        fail = np.mean([_f(r["probe_failure"]) for r in rs])
        for sig in SIGS + PRIV:
            lv = [_f(r[sig]) for r in low]
            hv = [_f(r[sig]) for r in high]
            rec = {
                "probe": probe, "signal": sig,
                "deployable": int(sig in SIGS),
                "n_low": int(np.isfinite(lv).sum()), "n_nonlow": int(np.isfinite(hv).sum()),
                "low_mean": float(np.nanmean(lv)), "nonlow_mean": float(np.nanmean(hv)),
                "low_std": float(np.nanstd(lv, ddof=1)) if np.isfinite(lv).sum() > 1 else np.nan,
                "nonlow_std": float(np.nanstd(hv, ddof=1)) if np.isfinite(hv).sum() > 1 else np.nan,
                "auc": auc_score(lv, hv),
                "cohens_d": cohens_d(lv, hv),
                "probe_failure_rate": fail,
            }
            sep.append(rec)
            if rec["deployable"] and np.isfinite(rec["auc"]):
                if best is None or rec["auc"] > best["auc"] or (
                    rec["auc"] == best["auc"] and abs(rec["cohens_d"]) > abs(best["cohens_d"])
                ):
                    best = rec

    if sep:
        with (OUT / "PROBE_SIGNAL_SEPARABILITY.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sep[0].keys()))
            w.writeheader(); w.writerows(sep)

    # plots
    if rows and "z_marker_tang_peak" in rows[0]:
        fig, axes = plt.subplots(1, max(len(by_probe), 1), figsize=(4.2 * max(len(by_probe), 1), 3.8), squeeze=False)
        for ax, probe in zip(axes[0], sorted(by_probe)):
            for mu, c in [(0.2, "C3"), (0.5, "C0"), (1.0, "C2")]:
                xs = [_f(r["z_marker_tang_peak"]) for r in by_probe[probe] if abs(_f(r["friction"]) - mu) < 1e-6]
                ax.scatter(np.random.default_rng(0).normal(mu, 0.02, size=len(xs)), xs, c=c, label=f"μ={mu:g}", s=22)
            ax.set_title(f"probe {probe}: marker_tang peak")
            ax.set_xlabel("friction")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(PLOT / "probe_signal_by_friction.png", dpi=140)
        fig.savefig(PLOT / "probe_signal_by_friction.pdf")
        plt.close(fig)

        # example trajectories if step file exists
        sp = OUT / "PROBE_STEP_TRAJECTORIES.csv"
        if sp.exists():
            steps = list(csv.DictReader(sp.open()))
            fig, ax = plt.subplots(figsize=(7.4, 4.0))
            for mu, c in [(0.2, "C3"), (0.5, "C0"), (1.0, "C2")]:
                # seed 0 if present
                tid = f"p3_screen_A_s0_mu{mu:g}"
                ts = [r for r in steps if r["trial_id"] == tid and r["phase"].startswith("probe")]
                if not ts:
                    continue
                t0 = _f(ts[0]["t_s"])
                ax.plot([_f(r["t_s"]) - t0 for r in ts], [_f(r["marker_tangential"]) for r in ts], c=c, label=f"μ={mu:g}")
            ax.set_xlabel("probe time (s)")
            ax.set_ylabel("marker tangential")
            ax.legend()
            ax.set_title("Probe A marker trajectories (seed 0)")
            fig.tight_layout()
            fig.savefig(PLOT / "probe_marker_trajectories.png", dpi=140)
            fig.savefig(PLOT / "probe_marker_trajectories.pdf")
            plt.close(fig)

    qualified = bool(
        best is not None
        and best["auc"] >= 0.80
        and abs(best["cohens_d"]) >= 0.8
        and best["probe_failure_rate"] <= 0.1
        and best["n_low"] >= 8
    )
    print("best", best)
    print("qualified", qualified)

    # Calibrate Gaussian likelihood on even seeds of the best probe+signal
    if best is not None and best["n_low"] >= 6:
        probe, sig = best["probe"], best["signal"]
        rs = [r for r in by_probe[probe]]
        calib = [r for r in rs if int(float(r["seed_idx"])) % 2 == 0]
        held = [r for r in rs if int(float(r["seed_idx"])) % 2 == 1]
        means, stds = {}, {}
        for mu in (0.2, 0.5, 1.0):
            xs = [_f(r[sig]) for r in calib if abs(_f(r["friction"]) - mu) < 1e-6]
            xs = [x for x in xs if np.isfinite(x)]
            means[str(mu)] = float(np.mean(xs)) if xs else 0.0
            stds[str(mu)] = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.1
            stds[str(mu)] = max(stds[str(mu)], 1e-3)
        model = {
            "signal": sig, "probe": probe,
            "means": means, "stds": stds,
            "prior": {"0.2": 1 / 3, "0.5": 1 / 3, "1.0": 1 / 3},
            "tau": 0.8, "theta": 0.5,
            "calib_seeds": "even", "heldout_seeds": "odd",
            "success_table": {"4": {"0.2": 0.0, "0.5": 1.0, "1.0": 1.0}, "6": {"0.2": 1.0, "0.5": 1.0, "1.0": 1.0}},
        }
        (OUT / "BELIEF_MODEL.json").write_text(json.dumps(model, indent=2) + "\n")
        (OUT / "BELIEF_MODEL_SPEC.md").write_text(
            "\n".join([
                "# Discrete Bayesian belief (P3)",
                "",
                "State: φ ∈ {μ=0.20, 0.50, 1.00}",
                "Prior: uniform 1/3 (not the experimental sampling prior).",
                f"Observation z = `{sig}` from probe {probe} (scalar).",
                "Likelihood: independent Gaussian per μ, fit on **even** seeds only.",
                "Posterior: b1(φ) ∝ N(z; μφ, σφ) b0(φ)",
                "Decision (primary): choose min F∈{4,6} s.t. Σ_μ b(μ) P(success|F,μ) ≥ 0.8",
                "with P(success|4,0.20)=0, P(success|4,≥0.50)=1, P(success|6,·)=1.",
                "Equivalent: choose 4 N iff P(μ=0.20|z) ≤ 0.2, else 6 N.",
                "Secondary: θ-rule choose 6 N iff P(μ=0.20|z)>0.5.",
                "GT friction is a label only; it is not an input at decision time.",
            ]) + "\n"
        )
        # held-out posterior quality
        def gauss_post(z):
            un = {}
            for mu in (0.2, 0.5, 1.0):
                s = stds[str(mu)]
                like = np.exp(-0.5 * ((z - means[str(mu)]) / s) ** 2) / (s * np.sqrt(2 * np.pi))
                un[mu] = like / 3.0
            zsum = sum(un.values()) + 1e-18
            return {mu: un[mu] / zsum for mu in un}

        held_rows = []
        correct = []
        for r in held:
            z = _f(r[sig])
            if not np.isfinite(z):
                continue
            post = gauss_post(z)
            gt = _f(r["friction"])
            p4 = 1.0 - post[0.2]
            F = 4.0 if p4 >= 0.8 else 6.0
            oracle_F = 6.0 if abs(gt - 0.2) < 1e-6 else 4.0
            held_rows.append({
                "seed_idx": r["seed_idx"], "gt_mu": gt, "z": z,
                "post_low": post[0.2], "post_mid": post[0.5], "post_high": post[1.0],
                "chosen_F": F, "oracle_F": oracle_F, "correct": int(F == oracle_F),
            })
            correct.append(int(F == oracle_F))
        if held_rows:
            with (OUT / "BELIEF_HELDOUT_RESULTS.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(held_rows[0].keys()))
                w.writeheader(); w.writerows(held_rows)
            with (OUT / "BELIEF_CALIBRATION.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["mu", "n", "mean_z", "std_z"])
                w.writeheader()
                for mu in (0.2, 0.5, 1.0):
                    xs = [_f(r[sig]) for r in calib if abs(_f(r["friction"]) - mu) < 1e-6]
                    w.writerow({"mu": mu, "n": len(xs), "mean_z": float(np.nanmean(xs)), "std_z": float(np.nanstd(xs, ddof=1) if len(xs) > 1 else np.nan)})
            print("heldout decision acc", float(np.mean(correct)) if correct else None, "n", len(correct))

        gate = {
            "P3_PROBE_SIGNAL_QUALIFIED": qualified,
            "best_probe": None if best is None else best["probe"],
            "best_signal": None if best is None else best["signal"],
            "auc": None if best is None else best["auc"],
            "d": None if best is None else best["cohens_d"],
        }
        (OUT / "SCREEN_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
        print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
