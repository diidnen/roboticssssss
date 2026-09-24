#!/usr/bin/env python3
"""D1: offline audit of frozen P3 Probe A telemetry for 4/5/6 N resolution.

Feature selection on even seeds only. Held-out odd seeds evaluated once.
No neural nets. No new probe.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
PLOT = OUT / "plots"
PLOT.mkdir(exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)

P3 = Path("/home/exouser/Tabero/analysis/results/p3_probe_belief_decision_20260819_104513")
TRAJ = P3 / "PROBE_STEP_TRAJECTORIES.csv"
P3_BELIEF = P3 / "BELIEF_MODEL.json"

MUS = (0.2, 0.5, 1.0)
DT = 0.05
FSTAR = {0.2: 6.0, 0.5: 5.0, 1.0: 4.0}
SUCC = {
    4.0: {0.2: 0.00, 0.5: 0.55, 1.0: 1.00},
    5.0: {0.2: 0.00, 0.5: 0.95, 1.0: 1.00},
    6.0: {0.2: 0.80, 0.5: 1.00, 1.0: 1.00},
}
FORCES = (4.0, 5.0, 6.0)
# Frozen P3 imbalance Gaussians (even screening seeds)
P3_MEAN = {0.2: 0.5622988700866699, 0.5: 0.8634264469146729, 1.0: 1.159930419921875}
P3_STD = {0.2: 0.11864785105282566, 0.5: 0.1833187067922804, 1.0: 0.21181580260148877}


def _f(x, d=np.nan):
    try:
        if x in ("", "None", None):
            return d
        return float(x)
    except Exception:
        return d


def auc_score(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    def _auc(p, n):
        order = np.concatenate([n, p])
        r = np.argsort(np.argsort(order)) + 1
        n1, n0 = len(p), len(n)
        return float((r[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

    a = _auc(pos, neg)
    b = _auc(-pos, -neg)
    return max(a, b), a  # best-direction AUC, raw (pos>neg)


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    if s < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / s)


def slope(y):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return np.nan
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def hyst(fwd, back):
    a = np.asarray(fwd, float); b = np.asarray(back, float)
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    return float(np.mean(np.abs(a[:n] - b[:n][::-1])))


def stats(y):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return dict(peak=np.nan, mean=np.nan, std=np.nan, integ=np.nan, resid=np.nan, deriv_peak=np.nan)
    d = np.diff(y) if len(y) > 1 else np.array([0.0])
    return dict(
        peak=float(np.max(y)), mean=float(np.mean(y)), std=float(np.std(y, ddof=1) if len(y) > 1 else 0.0),
        integ=float(np.sum(y) * DT), resid=float(y[-1] - y[0]),
        deriv_peak=float(np.max(np.abs(d))),
    )


def gauss_pdf(x, m, s):
    s = max(s, 1e-4)
    return float(np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)))


def posterior(zs, means, stds, prior=None):
    """zs, means, stds: dict mu -> value or dict mu -> dict of named stats for NB."""
    prior = prior or {mu: 1 / 3 for mu in MUS}
    un = {}
    for mu in MUS:
        like = 1.0
        if isinstance(zs, dict) and any(isinstance(v, dict) for v in means.values()):
            pass
        for name, z in zs.items():
            like *= gauss_pdf(z, means[name][mu], stds[name][mu])
        un[mu] = like * prior[mu]
    zsum = sum(un.values()) + 1e-18
    return {mu: un[mu] / zsum for mu in MUS}


def decide(post, tau=0.8):
    p = {}
    for F in FORCES:
        p[F] = sum(post[mu] * SUCC[F][mu] for mu in MUS)
    ok = [F for F in FORCES if p[F] >= tau]
    F = min(ok) if ok else 6.0
    return F, p


def load_trials():
    by = defaultdict(lambda: defaultdict(list))
    with TRAJ.open() as f:
        for row in csv.DictReader(f):
            if row["method"] != "probe_belief" or row["probe"] != "A":
                continue
            if row["phase"] not in ("hold", "probe_out", "probe_back", "probe_hold"):
                continue
            key = (int(float(row["seed_idx"])), float(row["friction"]))
            by[key][row["phase"]].append(row)
    trials = []
    for (seed, mu), phases in by.items():
        for ph in phases:
            phases[ph] = sorted(phases[ph], key=lambda r: int(r["step"]))
        hold, out, back, phold = phases["hold"], phases["probe_out"], phases["probe_back"], phases["probe_hold"]
        probe = out + back + phold

        def col(rows, name):
            return np.array([_f(r[name]) for r in rows], float)

        rec = {"seed_idx": seed, "friction": mu, "split": "calib" if seed % 2 == 0 else "heldout"}
        series = {
            "imb": "imbalance", "fmeas": "f_meas", "ftan": "f_tangential",
            "mtang": "marker_tangential", "mmean": "marker_mean", "masym": "marker_asym",
            "mvel": "marker_vel", "hm": "hm_mean",
        }
        for pref, colname in series.items():
            yp = col(probe, colname)
            yo, yb, yh, yhold = col(out, colname), col(back, colname), col(phold, colname), col(hold, colname)
            st = stats(yp)
            rec[f"{pref}_peak"] = st["peak"]
            rec[f"{pref}_mean"] = st["mean"]
            rec[f"{pref}_std"] = st["std"]
            rec[f"{pref}_integ"] = st["integ"]
            rec[f"{pref}_resid"] = st["resid"]
            rec[f"{pref}_deriv_peak"] = st["deriv_peak"]
            rec[f"{pref}_load_slope"] = slope(yo)
            rec[f"{pref}_unload_slope"] = slope(yb)
            rec[f"{pref}_hyst"] = hyst(yo, yb)
            rec[f"{pref}_hold_end"] = float(yhold[-1]) if len(yhold) else np.nan
            rec[f"{pref}_probe_end"] = float(yh[-1]) if len(yh) else np.nan
            rec[f"{pref}_residual_vs_hold"] = (
                float(np.mean(yh) - np.mean(yhold[-5:])) if (len(yh) and len(yhold)) else np.nan
            )
            rec[f"{pref}_recovery"] = float(yh[-1] - yh[0]) if len(yh) else np.nan
        rec["z_imbalance_peak"] = rec["imb_peak"]  # alias
        rec["fstar"] = FSTAR[mu]
        trials.append(rec)
    return trials


DEPLOYABLE = None  # filled after seeing keys


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = load_trials()
    keys = [k for k in trials[0] if k not in ("seed_idx", "friction", "split", "fstar") and not k.startswith("z_")]
    # drop privileged-looking if any slipped in; all constructed from force/tactile
    global DEPLOYABLE
    DEPLOYABLE = keys

    calib = [t for t in trials if t["split"] == "calib"]
    held = [t for t in trials if t["split"] == "heldout"]
    with (OUT / "CALIBRATION_SPLIT.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0].keys())); w.writeheader(); w.writerows(calib)
    with (OUT / "HELDOUT_SPLIT.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0].keys())); w.writeheader(); w.writerows(held)

    screen = []
    best_mid = None
    for sig in DEPLOYABLE:
        bymu = {mu: [_f(t[sig]) for t in calib if abs(t["friction"] - mu) < 1e-6] for mu in MUS}
        low = bymu[0.2]; notlow = bymu[0.5] + bymu[1.0]
        mid, high = bymu[0.5], bymu[1.0]
        auc_ln, raw_ln = auc_score(low, notlow)
        auc_mh, raw_mh = auc_score(mid, high)
        rec = {
            "signal": sig, "deployable": 1,
            "n_calib": len(calib),
            "mu02_mean": float(np.nanmean(low)), "mu02_std": float(np.nanstd(low, ddof=1)),
            "mu05_mean": float(np.nanmean(mid)), "mu05_std": float(np.nanstd(mid, ddof=1)),
            "mu10_mean": float(np.nanmean(high)), "mu10_std": float(np.nanstd(high, ddof=1)),
            "mu02_median": float(np.nanmedian(low)), "mu05_median": float(np.nanmedian(mid)), "mu10_median": float(np.nanmedian(high)),
            "low_vs_notlow_auc": auc_ln, "low_vs_notlow_raw_auc": raw_ln,
            "mid_vs_high_auc": auc_mh, "mid_vs_high_raw_auc": raw_mh,
            "low_vs_notlow_d": cohens_d(low, notlow),
            "mid_vs_high_d": cohens_d(mid, high),
        }
        screen.append(rec)
        if np.isfinite(auc_mh) and (best_mid is None or auc_mh > best_mid["mid_vs_high_auc"] or (
            auc_mh == best_mid["mid_vs_high_auc"] and abs(rec["mid_vs_high_d"]) > abs(best_mid["mid_vs_high_d"])
        )):
            best_mid = rec

    screen.sort(key=lambda r: (-(r["mid_vs_high_auc"] if np.isfinite(r["mid_vs_high_auc"]) else -1),))
    with (OUT / "SIGNAL_SCREENING.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(screen[0].keys())); w.writeheader(); w.writerows(screen)
    mid_rows = [r for r in screen]
    with (OUT / "MID_VS_HIGH_SEPARABILITY.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mid_rows[0].keys())); w.writeheader(); w.writerows(mid_rows)

    z1 = "imb_peak"
    z2 = best_mid["signal"] if best_mid else "mtang_hyst"
    use_second = bool(best_mid and best_mid["mid_vs_high_auc"] >= 0.75 and z2 != z1)

    def fit_gauss(split_rows, names):
        means, stds = {}, {}
        for name in names:
            means[name], stds[name] = {}, {}
            for mu in MUS:
                xs = [_f(t[name]) for t in split_rows if abs(t["friction"] - mu) < 1e-6]
                xs = [x for x in xs if np.isfinite(x)]
                means[name][mu] = float(np.mean(xs)) if xs else 0.0
                stds[name][mu] = max(float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.1, 1e-3)
        return means, stds

    # B0: frozen P3 1D imbalance, full-task table, tau=0.8
    def b0_post(z):
        un = {mu: gauss_pdf(z, P3_MEAN[mu], P3_STD[mu]) / 3 for mu in MUS}
        s = sum(un.values()) + 1e-18
        return {mu: un[mu] / s for mu in MUS}

    names_ref = [z1, z2] if use_second else [z1]
    means_c, stds_c = fit_gauss(calib, names_ref)
    # keep z1 likelihood frozen from P3 for stage-1 compatibility
    means_c["imb_peak"] = dict(P3_MEAN)
    stds_c["imb_peak"] = dict(P3_STD)

    def refined_post(t, names):
        zs = {n: _f(t[n]) for n in names}
        return posterior(zs, means_c, stds_c)

    def eval_rule(rows, kind):
        out = []
        for t in rows:
            if kind == "b0":
                post = b0_post(_f(t["imb_peak"]))
            elif kind == "refined":
                post = refined_post(t, names_ref)
            elif kind == "oracle":
                post = {mu: 1.0 if abs(t["friction"] - mu) < 1e-6 else 0.0 for mu in MUS}
            elif kind == "fixed6":
                post = {0.2: 1.0, 0.5: 0.0, 1.0: 0.0}  # dummy; force forced
            F, ps = decide(post, tau=0.8)
            if kind == "fixed6":
                F = 6.0
            if kind == "oracle":
                F = FSTAR[t["friction"]]
            fstar = FSTAR[t["friction"]]
            rec = {
                "seed_idx": t["seed_idx"], "gt_mu": t["friction"], "split": t["split"],
                "z1": _f(t["imb_peak"]), "z2": _f(t[z2]),
                "post02": post[0.2], "post05": post[0.5], "post10": post[1.0],
                "chosen_F": F, "fstar": fstar,
                "exact": int(abs(F - fstar) < 1e-6),
                "under": int(F < fstar - 1e-6),
                "over": int(F > fstar + 1e-6),
            }
            out.append(rec)
        return out

    b0_h = eval_rule(held, "b0")
    ref_h = eval_rule(held, "refined")
    # tau sensitivity on calib only (not used to pick tau)
    tau_sens = []
    for tau in (0.6, 0.8, 0.9):
        acc = und = ov = []
        Fs = []
        for t in calib:
            post = refined_post(t, names_ref)
            F, _ = decide(post, tau=tau)
            fstar = FSTAR[t["friction"]]
            tau_sens.append({"tau": tau, "seed_idx": t["seed_idx"], "gt_mu": t["friction"], "chosen_F": F,
                             "exact": int(abs(F - fstar) < 1e-6), "under": int(F < fstar - 1e-6), "over": int(F > fstar + 1e-6)})

    def summ(rows):
        n = len(rows)
        return {
            "n": n,
            "exact": float(np.mean([r["exact"] for r in rows])) if n else np.nan,
            "under": float(np.mean([r["under"] for r in rows])) if n else np.nan,
            "over": float(np.mean([r["over"] for r in rows])) if n else np.nan,
            "mean_F": float(np.mean([r["chosen_F"] for r in rows])) if n else np.nan,
            "by_mu": {
                str(mu): {
                    "n": sum(1 for r in rows if abs(r["gt_mu"] - mu) < 1e-6),
                    "exact": float(np.mean([r["exact"] for r in rows if abs(r["gt_mu"] - mu) < 1e-6])),
                    "under": float(np.mean([r["under"] for r in rows if abs(r["gt_mu"] - mu) < 1e-6])),
                    "over": float(np.mean([r["over"] for r in rows if abs(r["gt_mu"] - mu) < 1e-6])),
                    "mean_F": float(np.mean([r["chosen_F"] for r in rows if abs(r["gt_mu"] - mu) < 1e-6])),
                    "counts": {4: sum(1 for r in rows if abs(r["gt_mu"]-mu)<1e-6 and abs(r["chosen_F"]-4)<1e-6),
                               5: sum(1 for r in rows if abs(r["gt_mu"]-mu)<1e-6 and abs(r["chosen_F"]-5)<1e-6),
                               6: sum(1 for r in rows if abs(r["gt_mu"]-mu)<1e-6 and abs(r["chosen_F"]-6)<1e-6)},
                } for mu in MUS
            },
        }

    s_b0, s_ref = summ(b0_h), summ(ref_h)
    s_fix = {"exact": 1 / 3, "under": 0.0, "over": 2 / 3, "mean_F": 6.0}
    s_ora = {"exact": 1.0, "under": 0.0, "over": 0.0, "mean_F": 5.0}

    with (OUT / "CURRENT_BELIEF_DECISIONS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(b0_h[0].keys())); w.writeheader(); w.writerows(b0_h)
    with (OUT / "REFINED_BELIEF_DECISIONS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ref_h[0].keys())); w.writeheader(); w.writerows(ref_h)
    cmp_rows = [
        {"method": "current_p3_imbalance_1d", **{k: s_b0[k] for k in ("n", "exact", "under", "over", "mean_F")}},
        {"method": "refined", **{k: s_ref[k] for k in ("n", "exact", "under", "over", "mean_F")}},
        {"method": "fixed6", "n": len(held), "exact": s_fix["exact"], "under": s_fix["under"], "over": s_fix["over"], "mean_F": 6.0},
        {"method": "oracle", "n": len(held), "exact": 1.0, "under": 0.0, "over": 0.0, "mean_F": 5.0},
    ]
    with (OUT / "DECISION_ERROR_COMPARISON.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys())); w.writeheader(); w.writerows(cmp_rows)

    gate = (
        s_ref["under"] <= 0.05
        and s_ref["exact"] > 0.38 + 1e-9
        and s_ref["over"] < 0.60 - 1e-9
        and s_ref["mean_F"] < 5.4
    )
    # also require mid-high AUC on calib actually usable
    mid_ok = bool(best_mid and best_mid["mid_vs_high_auc"] >= 0.8)

    spec = [
        "# Refined belief (D1)",
        "",
        "Probe unchanged: P3 Probe A, +2 mm base-Y out/back, 4 N.",
        "Likelihood: Gaussian naive Bayes. Feature selection on **even seeds only**.",
        "Held-out odd seeds evaluated once.",
        f"Stage-1 / z1 = `imb_peak` (frozen P3 Gaussian).",
        f"Best calib mid-vs-high signal z2 = `{z2}` (AUC={best_mid['mid_vs_high_auc'] if best_mid else None}).",
        f"Second statistic used: **{use_second}**.",
        f"Statistics in refined likelihood: {names_ref}",
        "Success table: P3-R2 full-task empirical (not lift).",
        "Decision: min F∈{4,5,6} s.t. Σ b(μ) P_full(success|F,μ) ≥ 0.8 (τ not tuned on held-out).",
        f"Offline gate passed: {gate}",
        f"Calib mid-vs-high AUC>=0.8: {mid_ok}",
    ]
    (OUT / "REFINED_BELIEF_SPEC.md").write_text("\n".join(spec) + "\n")
    (OUT / "SIGNAL_CANDIDATES.md").write_text(
        "\n".join([
            "# Signal candidates (hand-designed, probe window only)",
            "",
            "Sources: `f_meas`, `imbalance`, `f_tangential`, GelSight `marker_*`, `hm_mean`.",
            "Windows: probe_out (10), probe_back (10), probe_hold (5); hold used only for residual-vs-hold.",
            "Statistics per channel: peak, mean, std, integral, residual, |Δ| peak, load slope, unload slope,",
            "forward/return hysteresis, residual vs hold, recovery over probe_hold.",
            "Privileged object pose is not used as a decision feature.",
            f"N calib even seeds = {len(calib)}; N held-out odd seeds = {len(held)} (probe_belief Probe A).",
            "",
            f"Best calib LOW vs NOT-LOW: `{max(screen, key=lambda r: r['low_vs_notlow_auc'] or 0)['signal']}`",
            f"Best calib MID vs HIGH: `{z2}` AUC={best_mid['mid_vs_high_auc'] if best_mid else None} d={best_mid['mid_vs_high_d'] if best_mid else None}",
        ]) + "\n"
    )

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, sig, title in (
        (axes[0], "imb_peak", "imbalance peak"),
        (axes[1], z2, f"best mid-high: {z2}"),
    ):
        for mu, c in ((0.2, "C3"), (0.5, "C0"), (1.0, "C2")):
            xs = [_f(t[sig]) for t in calib if abs(t["friction"] - mu) < 1e-6]
            ax.scatter(np.random.default_rng(0).normal(mu, 0.02, len(xs)), xs, c=c, s=28, label=f"μ={mu:g}")
        ax.set_title(title + " (calib)")
        ax.set_xlabel("friction"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(PLOT / "signal_by_friction.png", dpi=140); fig.savefig(PLOT / "signal_by_friction.pdf"); plt.close(fig)

    top = screen[:12]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.barh([r["signal"] for r in reversed(top)], [r["mid_vs_high_auc"] for r in reversed(top)])
    ax.axvline(0.8, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("calib MID vs HIGH AUC"); ax.set_xlim(0.45, 1.02)
    ax.set_title("Mid vs high separability (even seeds)")
    fig.tight_layout(); fig.savefig(PLOT / "mid_vs_high_auc.png", dpi=140); fig.savefig(PLOT / "mid_vs_high_auc.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for mu, c in ((0.2, "C3"), (0.5, "C0"), (1.0, "C2")):
        xs = [r["post02"] for r in ref_h if abs(r["gt_mu"] - mu) < 1e-6]
        ax.scatter(np.random.default_rng(1).normal(mu, 0.03, len(xs)), xs, c=c, s=28, label=f"μ={mu:g}")
    ax.set_ylabel("P(μ=0.20|z)"); ax.set_xlabel("GT μ"); ax.legend(); ax.set_ylim(-0.05, 1.05)
    ax.set_title("Refined posterior low (held-out)")
    fig.tight_layout(); fig.savefig(PLOT / "posterior_by_friction.png", dpi=140); fig.savefig(PLOT / "posterior_by_friction.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for mu, c in ((0.2, "C3"), (0.5, "C0"), (1.0, "C2")):
        xs = [r["chosen_F"] for r in ref_h if abs(r["gt_mu"] - mu) < 1e-6]
        ax.scatter(np.random.default_rng(2).normal(mu, 0.03, len(xs)), xs, c=c, s=32, label=f"μ={mu:g}")
    ax.set_yticks([4, 5, 6]); ax.set_title("Refined selected force (held-out)"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOT / "selected_force_by_friction.png", dpi=140); fig.savefig(PLOT / "selected_force_by_friction.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labs = ["current P3", "refined", "fixed 6 N", "oracle"]
    exacts = [s_b0["exact"], s_ref["exact"], s_fix["exact"], 1.0]
    unders = [s_b0["under"], s_ref["under"], 0.0, 0.0]
    overs = [s_b0["over"], s_ref["over"], s_fix["over"], 0.0]
    x = np.arange(4); w = 0.25
    ax.bar(x - w, exacts, w, label="exact")
    ax.bar(x, unders, w, label="under")
    ax.bar(x + w, overs, w, label="over")
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("Held-out decision errors")
    fig.tight_layout(); fig.savefig(PLOT / "exact_under_over_errors.png", dpi=140); fig.savefig(PLOT / "exact_under_over_errors.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(labs, [s_b0["mean_F"], s_ref["mean_F"], 6.0, 5.0], color=["C0", "C1", "C3", "C2"])
    ax.set_ylabel("mean selected force (N)"); ax.set_ylim(3.5, 6.5)
    ax.set_title("Held-out mean selected force")
    fig.tight_layout(); fig.savefig(PLOT / "mean_selected_force.png", dpi=140); fig.savefig(PLOT / "mean_selected_force.pdf"); plt.close(fig)

    status = "D1_NO_ADDITIONAL_DECISION_SIGNAL"
    if mid_ok and gate:
        status = "D1_FINE_FORCE_DECISION_QUALIFIED"
    elif mid_ok and s_ref["over"] < s_b0["over"] - 0.05 and s_ref["under"] <= 0.05:
        status = "D1_SECOND_STATISTIC_WEAK" if s_ref["exact"] <= 0.38 else "D1_FINE_FORCE_DECISION_QUALIFIED"
    elif best_mid and best_mid["mid_vs_high_auc"] < 0.8:
        status = "D1_PROBE_SUPPORTS_COARSE_NOT_FINE_FORCE_DECISION"
    elif use_second and best_mid["mid_vs_high_auc"] >= 0.75:
        status = "D1_SECOND_STATISTIC_WEAK"

    summary = {
        "status": status,
        "z1": z1, "z2": z2, "use_second": use_second,
        "best_mid_vs_high_auc_calib": None if best_mid is None else best_mid["mid_vs_high_auc"],
        "best_low_vs_notlow_auc_calib": max(r["low_vs_notlow_auc"] for r in screen),
        "b0": s_b0, "refined": s_ref, "gate": gate, "mid_ok": mid_ok,
        "n_calib": len(calib), "n_heldout": len(held),
        "names_ref": names_ref,
    }
    (OUT / "OFFLINE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    print("top mid-high", [(r["signal"], round(r["mid_vs_high_auc"], 3), round(r["mid_vs_high_d"], 3)) for r in screen[:8]])
    print("top low-notlow", sorted([(r["signal"], round(r["low_vs_notlow_auc"], 3)) for r in screen], key=lambda x: -x[1])[:6])


if __name__ == "__main__":
    main()
