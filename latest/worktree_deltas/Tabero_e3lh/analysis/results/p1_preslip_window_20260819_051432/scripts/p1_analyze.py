#!/usr/bin/env python3
"""P1 analysis: pre-slip window. Diagnostic only. No learned method.

Lead times are sampled at T_slip^low - Δt for failing μ=0.20 episodes and at the
matched lift-relative time on μ=0.50 / 1.00 successes. Times before T0 (hold /
late close) are kept — they are valid matched times, not skipped.
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

LEADS_MS = [1000, 750, 500, 300, 200, 100, 50]
DT = 0.05
DEPLOYABLE = [
    "f_meas", "dF", "imbalance", "f_tangential", "contact_force", "d_pred", "gripper_open",
    "marker_mean", "marker_max", "marker_std", "marker_asym", "marker_tangential",
    "marker_vel", "marker_acc", "hm_mean", "hm_max", "ee_vz",
]
PRIVILEGED = ["rel_z", "v_rel_z", "rel_xy", "obj_vz", "obj_wz"]
ALL_SIG = DEPLOYABLE + PRIVILEGED
USEFUL_AUC, USEFUL_D = 0.70, 0.50
WEAK_AUC, WEAK_D = 0.65, 0.35
# Practical floors so rank-AUC on ~0.04 N / 0.4% marker offsets cannot manufacture a window.
PRACTICAL = {
    "f_meas": 0.40, "dF": 2.0, "imbalance": 0.15, "f_tangential": 0.40, "contact_force": 0.40,
    "d_pred": 0.002, "gripper_open": 0.002, "ee_vz": 0.01,
    "marker_mean": 0.20, "marker_max": 1.0, "marker_std": 0.20, "marker_asym": 0.20,
    "marker_tangential": 0.10, "marker_vel": 1.0, "marker_acc": 20.0,
    "hm_mean": 0.15, "hm_max": 0.15,
    "rel_z": 0.001, "v_rel_z": 0.005, "rel_xy": 0.002, "obj_vz": 0.005, "obj_wz": 0.15,
}


def _f(x, d=np.nan):
    try:
        if x in ("", "None", None):
            return d
        return float(x)
    except Exception:
        return d


def load_events():
    return list(csv.DictReader((OUT / "EVENT_TIMES.csv").open()))


def load_steps():
    p = OUT / "STEP_TRAJECTORIES.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(p.open()))


def enrich(rows):
    """Add contact_force and marker_acc; rows mutated in place, grouped by trial."""
    by = defaultdict(list)
    for r in rows:
        r["contact_force"] = float(
            np.hypot(np.hypot(_f(r["fL_x"]), _f(r["fL_y"])), _f(r["fL_z"]))
            + np.hypot(np.hypot(_f(r["fR_x"]), _f(r["fR_y"])), _f(r["fR_z"]))
        )
        by[r["trial_id"]].append(r)
    for tid, rs in by.items():
        rs.sort(key=lambda r: int(r["step"]))
        prev_vel = None
        prev_t = None
        for r in rs:
            t = _f(r["t_s"])
            vel = _f(r["marker_vel"])
            if prev_vel is None or prev_t is None or t <= prev_t:
                r["marker_acc"] = 0.0
            else:
                r["marker_acc"] = (vel - prev_vel) / max(t - prev_t, 1e-6)
            prev_vel, prev_t = vel, t
    return by


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    if s < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / s)


def overlap_coef(a, b, bins=20):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi - lo < 1e-12:
        return 1.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, edges = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    return float(np.minimum(ha, hb).dot(np.diff(edges)))


def auc_score(fail, succ):
    """ROC-AUC taking the better of signal and -signal (direction unknown a priori)."""
    fail, succ = np.asarray(fail, float), np.asarray(succ, float)
    fail, succ = fail[np.isfinite(fail)], succ[np.isfinite(succ)]
    if len(fail) == 0 or len(succ) == 0:
        return np.nan

    def auc(pos, neg):
        n1, n0 = len(pos), len(neg)
        order = np.concatenate([neg, pos])
        r = np.argsort(np.argsort(order)) + 1
        up = r[n0:].sum() - n1 * (n1 + 1) / 2.0
        return float(up / (n1 * n0))

    return max(auc(fail, succ), auc(-fail, -succ))


def iqr(xs):
    xs = np.asarray(xs, float)
    xs = xs[np.isfinite(xs)]
    if len(xs) < 2:
        return np.nan
    return float(np.percentile(xs, 75) - np.percentile(xs, 25))


def sample_at_t(steps, t_s, key):
    if not steps:
        return np.nan
    ts = np.array([_f(r["t_s"]) for r in steps])
    j = int(np.argmin(np.abs(ts - t_s)))
    if abs(ts[j] - t_s) > 0.08:
        return np.nan
    return _f(steps[j].get(key))


def dump_split(steps_all, force_keys, motion_keys, tac_keys):
    def dump(name, keys):
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in steps_all:
                w.writerow({k: r.get(k, "") for k in keys})

    dump("FORCE_TRAJECTORIES.csv", force_keys)
    dump("MOTION_TRAJECTORIES.csv", motion_keys)
    dump("TACTILE_SUMMARIES.csv", tac_keys)


def logistic_auc(x, y, seed_idx, train_even=True):
    """1-feature logistic diagnostic, split by seed parity to avoid seed leakage."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    s = np.asarray(seed_idx, int)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, s = x[m], y[m], s[m]
    if len(x) < 8:
        return np.nan, np.nan
    tr = (s % 2 == 0) if train_even else (s % 2 == 1)
    te = ~tr
    if tr.sum() < 4 or te.sum() < 4 or len(np.unique(y[tr])) < 2:
        return np.nan, np.nan
    # standardise on train
    mu, sd = x[tr].mean(), x[tr].std() + 1e-8
    xt = (x[tr] - mu) / sd
    yt = y[tr]
    w = 0.0
    b = 0.0
    for _ in range(80):
        z = w * xt + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        w -= 0.2 * ((p - yt) * xt).mean()
        b -= 0.2 * (p - yt).mean()
    xe = (x[te] - mu) / sd
    z = w * xe + b
    scores = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
    pos, neg = scores[y[te] == 1], scores[y[te] == 0]
    return auc_score(pos, neg) if len(pos) and len(neg) else np.nan, float(w)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = load_events()
    steps_all = load_steps()
    by_tid = enrich(steps_all)

    force_keys = [
        "trial_id", "seed_idx", "friction", "phase", "step", "t_s", "t_from_lift",
        "f_cmd", "f_meas", "dF", "fL_x", "fL_y", "fL_z", "fR_x", "fR_y", "fR_z",
        "imbalance", "f_tangential", "contact_force", "contact",
    ]
    motion_keys = [
        "trial_id", "seed_idx", "friction", "phase", "step", "t_s", "t_from_lift",
        "ee_x", "ee_y", "ee_z", "ee_vx", "ee_vy", "ee_vz",
        "obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz", "obj_wx", "obj_wy", "obj_wz",
        "rel_x", "rel_y", "rel_z", "v_rel_x", "v_rel_y", "v_rel_z", "rel_xy",
        "gt_slip", "gt_micro", "gt_loss",
    ]
    tac_keys = [
        "trial_id", "seed_idx", "friction", "phase", "step", "t_s", "t_from_lift",
        "marker_mean", "marker_max", "marker_std", "marker_asym", "marker_tangential",
        "marker_vel", "marker_acc", "hm_mean", "hm_max", "hm_std", "rgb_mean", "tactile_ok",
    ]
    if steps_all:
        dump_split(steps_all, force_keys, motion_keys, tac_keys)

    by_seed = defaultdict(dict)
    for e in events:
        by_seed[int(float(e["seed_idx"]))][_f(e["friction"])] = e
    matched = []
    for s, d in sorted(by_seed.items()):
        if 0.2 in d and 0.5 in d and 1.0 in d:
            matched.append(s)
    rows_m = []
    for s in matched:
        rec = {"seed_idx": s}
        for mu in (0.2, 0.5, 1.0):
            e = by_seed[s][mu]
            rec[f"mu{mu:g}_fail"] = int(float(e["future_fail"]))
            rec[f"mu{mu:g}_lift"] = int(float(e["lift_success"]))
            rec[f"mu{mu:g}_tslip"] = e.get("t_slip")
            rec[f"mu{mu:g}_tloss"] = e.get("t_loss")
            rec[f"mu{mu:g}_tlift"] = e.get("t_lift0")
            rec[f"mu{mu:g}_tmicro"] = e.get("t_micro")
        rows_m.append(rec)
    if rows_m:
        with (OUT / "MATCHED_EPISODES.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_m[0].keys()))
            w.writeheader()
            w.writerows(rows_m)

    fail_rate = {}
    for mu in (0.2, 0.5, 1.0):
        es = [e for e in events if abs(_f(e["friction"]) - mu) < 1e-6]
        fail_rate[mu] = float(np.mean([_f(e["future_fail"]) for e in es])) if es else np.nan

    def finite_ms(key, mu=0.2):
        xs = [
            _f(e[key]) * 1000
            for e in events
            if abs(_f(e["friction"]) - mu) < 1e-6 and e.get(key) not in ("", "None", None)
        ]
        return [x for x in xs if np.isfinite(x)]

    slip_to_loss = finite_ms("loss_minus_slip_s")
    lift_to_slip = finite_ms("slip_minus_lift_s")
    lift_to_loss = finite_ms("loss_minus_lift_s")
    lift_to_micro = finite_ms("t_micro")  # absolute; convert below
    micro_from_lift = []
    for e in events:
        if abs(_f(e["friction"]) - 0.2) > 1e-6:
            continue
        t0, tm = _f(e.get("t_lift0")), _f(e.get("t_micro"))
        if np.isfinite(t0) and np.isfinite(tm):
            micro_from_lift.append((tm - t0) * 1000)

    def mean_curve(mu, key, tmin=-1.0, tmax=2.0):
        xs = np.arange(tmin, tmax + 1e-9, DT)
        acc = [[] for _ in xs]
        for e in events:
            if abs(_f(e["friction"]) - mu) > 1e-6:
                continue
            for r in by_tid.get(e["trial_id"], []):
                t = _f(r["t_from_lift"])
                if t < tmin or t > tmax:
                    continue
                j = int(round((t - tmin) / DT))
                if 0 <= j < len(acc):
                    acc[j].append(_f(r[key]))
        mean = np.array([np.nanmean(a) if a else np.nan for a in acc])
        lo = np.array([np.nanpercentile(a, 25) if len(a) >= 3 else np.nan for a in acc])
        hi = np.array([np.nanpercentile(a, 75) if len(a) >= 3 else np.nan for a in acc])
        return xs, mean, lo, hi

    def plot_aligned(key, fname, ylabel, privileged=False):
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        for mu, c in [(0.2, "C3"), (0.5, "C0"), (1.0, "C2")]:
            xs, m, lo, hi = mean_curve(mu, key)
            ax.plot(xs, m, color=c, label=f"μ={mu:g}")
            ax.fill_between(xs, lo, hi, color=c, alpha=0.15)
        if lift_to_slip:
            ax.axvline(np.median(lift_to_slip) / 1000.0, ls="--", c="C3", lw=0.9, label="median T_slip (μ=0.2)")
        if lift_to_loss:
            ax.axvline(np.median(lift_to_loss) / 1000.0, ls=":", c="k", lw=0.9, label="median T_loss (μ=0.2)")
        if micro_from_lift:
            ax.axvline(np.median(micro_from_lift) / 1000.0, ls="-.", c="C1", lw=0.8, label="median T_micro (μ=0.2)")
        ax.axvline(0.0, ls=":", c="gray", lw=0.7)
        ax.set_xlabel("time from lift onset T0 (s)")
        ax.set_ylabel(ylabel + ("  [privileged]" if privileged else ""))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT / f"{fname}.png", dpi=140)
        fig.savefig(PLOT / f"{fname}.pdf")
        plt.close(fig)

    if steps_all:
        plot_aligned("f_meas", "aligned_force", "measured squeeze (N)")
        plot_aligned("imbalance", "aligned_force_imbalance", "left/right force imbalance (N)")
        plot_aligned("contact_force", "aligned_contact_force", "contact force ||fL||+||fR|| (N)")
        plot_aligned("rel_z", "aligned_relative_motion", "rel_z (m)", privileged=True)
        plot_aligned("v_rel_z", "aligned_rel_vz", "v_rel_z (m/s)", privileged=True)
        plot_aligned("marker_mean", "aligned_marker_shear", "mean marker displacement (vs rest)")
        plot_aligned("marker_vel", "aligned_marker_velocity", "marker velocity")
        plot_aligned("marker_tangential", "aligned_marker_tangential", "tangential marker displacement")
        plot_aligned("hm_mean", "aligned_height_map", "height-map mean")

        # example matched triplet seed 0 if present
        if 0 in matched:
            fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.0), sharex=True)
            for ax, key, ylab in [
                (axes[0], "f_meas", "squeeze N"),
                (axes[1], "marker_mean", "marker mean"),
                (axes[2], "rel_z", "rel_z m [priv]"),
            ]:
                for mu, c in [(0.2, "C3"), (0.5, "C0"), (1.0, "C2")]:
                    e = by_seed[0][mu]
                    st = by_tid[e["trial_id"]]
                    t = np.array([_f(r["t_from_lift"]) for r in st])
                    y = np.array([_f(r[key]) for r in st])
                    m = t >= -0.5
                    ax.plot(t[m], y[m], color=c, label=f"μ={mu:g}", lw=1.2)
                tslip = _f(by_seed[0][0.2].get("t_slip")) - _f(by_seed[0][0.2].get("t_lift0"))
                tloss = _f(by_seed[0][0.2].get("t_loss")) - _f(by_seed[0][0.2].get("t_lift0"))
                ax.axvline(tslip, ls="--", c="C3", lw=0.8)
                if np.isfinite(tloss):
                    ax.axvline(tloss, ls=":", c="k", lw=0.8)
                ax.set_ylabel(ylab)
                ax.grid(True, alpha=0.3)
            axes[0].legend(fontsize=8)
            axes[-1].set_xlabel("t from T0 (s), seed 0 matched triplet")
            fig.tight_layout()
            fig.savefig(PLOT / "example_matched_triplet.png", dpi=140)
            fig.savefig(PLOT / "example_matched_triplet.pdf")
            plt.close(fig)

    # ---- lead-time separability: future-fail (μ=0.2) vs future-success (0.5+1.0) ----
    sep_rows = []
    for lead_ms in LEADS_MS:
        lead_s = lead_ms / 1000.0
        for sig in ALL_SIG:
            fail_v, succ_v = [], []
            fail_seeds, succ_seeds = [], []
            n_pre_t0 = 0
            for s in matched:
                e_low = by_seed[s][0.2]
                t0 = _f(e_low.get("t_lift0"))
                tslip = _f(e_low.get("t_slip"))
                if not np.isfinite(t0) or not np.isfinite(tslip):
                    continue
                t_query = tslip - lead_s
                if t_query < t0:
                    n_pre_t0 += 1
                fv = sample_at_t(by_tid.get(e_low["trial_id"], []), t_query, sig)
                fail_v.append(fv)
                fail_seeds.append(s)
                for mu in (0.5, 1.0):
                    e_ok = by_seed[s][mu]
                    t0s = _f(e_ok.get("t_lift0"))
                    if not np.isfinite(t0s):
                        continue
                    t_ok = t0s + (tslip - t0 - lead_s)
                    succ_v.append(sample_at_t(by_tid.get(e_ok["trial_id"], []), t_ok, sig))
                    succ_seeds.append(s)
            fail_a = np.asarray(fail_v, float)
            succ_a = np.asarray(succ_v, float)
            d = cohens_d(fail_a, succ_a)
            ov = overlap_coef(fail_a, succ_a)
            auc = auc_score(fail_a, succ_a)
            y = np.concatenate([np.ones(len(fail_a)), np.zeros(len(succ_a))])
            x = np.concatenate([fail_a, succ_a])
            seeds_cat = np.concatenate([np.asarray(fail_seeds), np.asarray(succ_seeds)])
            log_auc, _ = logistic_auc(x, y, seeds_cat)
            sep_rows.append({
                "task": "future_fail_vs_success",
                "lead_ms": lead_ms,
                "lead_steps": int(round(lead_s / DT)),
                "signal": sig,
                "deployable": int(sig in DEPLOYABLE),
                "n_fail": int(np.isfinite(fail_a).sum()),
                "n_succ": int(np.isfinite(succ_a).sum()),
                "n_query_before_T0": n_pre_t0,
                "fail_mean": float(np.nanmean(fail_a)) if len(fail_a) else np.nan,
                "succ_mean": float(np.nanmean(succ_a)) if len(succ_a) else np.nan,
                "fail_std": float(np.nanstd(fail_a, ddof=1)) if np.isfinite(fail_a).sum() > 1 else np.nan,
                "succ_std": float(np.nanstd(succ_a, ddof=1)) if np.isfinite(succ_a).sum() > 1 else np.nan,
                "fail_median": float(np.nanmedian(fail_a)) if len(fail_a) else np.nan,
                "succ_median": float(np.nanmedian(succ_a)) if len(succ_a) else np.nan,
                "fail_iqr": iqr(fail_a),
                "succ_iqr": iqr(succ_a),
                "cohens_d": d,
                "overlap": ov,
                "auc": auc,
                "logistic_auc_seed_split": log_auc,
            })

    # ---- friction identification at same matched times: 0.2 vs 0.5, 0.2 vs 1.0, 0.5 vs 1.0 ----
    friction_rows = []
    for lead_ms in LEADS_MS:
        lead_s = lead_ms / 1000.0
        for sig in ALL_SIG:
            groups = {0.2: [], 0.5: [], 1.0: []}
            for s in matched:
                e_low = by_seed[s][0.2]
                t0 = _f(e_low.get("t_lift0"))
                tslip = _f(e_low.get("t_slip"))
                if not np.isfinite(t0) or not np.isfinite(tslip):
                    continue
                dt_from_t0 = tslip - t0 - lead_s
                for mu in (0.2, 0.5, 1.0):
                    e = by_seed[s][mu]
                    t0s = _f(e.get("t_lift0"))
                    if not np.isfinite(t0s):
                        continue
                    groups[mu].append(sample_at_t(by_tid.get(e["trial_id"], []), t0s + dt_from_t0, sig))
            for a, b, name in [(0.2, 0.5, "mu0.2_vs_0.5"), (0.2, 1.0, "mu0.2_vs_1.0"), (0.5, 1.0, "mu0.5_vs_1.0")]:
                friction_rows.append({
                    "task": name,
                    "lead_ms": lead_ms,
                    "signal": sig,
                    "deployable": int(sig in DEPLOYABLE),
                    "auc": auc_score(groups[a], groups[b]),
                    "cohens_d": cohens_d(groups[a], groups[b]),
                    "overlap": overlap_coef(groups[a], groups[b]),
                    "mean_a": float(np.nanmean(groups[a])) if groups[a] else np.nan,
                    "mean_b": float(np.nanmean(groups[b])) if groups[b] else np.nan,
                })

    if sep_rows:
        with (OUT / "LEAD_TIME_SEPARABILITY.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sep_rows[0].keys()))
            w.writeheader()
            w.writerows(sep_rows)
        with (OUT / "SIGNAL_EFFECT_SIZES.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sep_rows[0].keys()))
            w.writeheader()
            w.writerows(sep_rows)
    if friction_rows:
        with (OUT / "FRICTION_IDENTIFICATION.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(friction_rows[0].keys()))
            w.writeheader()
            w.writerows(friction_rows)

    def practical_ok(r, sig=None):
        sig = sig or r.get("signal")
        floor = PRACTICAL.get(sig, 0.0)
        if "fail_mean" in r and "succ_mean" in r:
            delta = abs(_f(r["fail_mean"]) - _f(r["succ_mean"]))
        else:
            delta = abs(_f(r.get("mean_a")) - _f(r.get("mean_b")))
        return np.isfinite(delta) and delta >= floor

    def best_at(lead, deployable=True, rows=None, require_practical=True):
        rows = rows if rows is not None else sep_rows
        cand = [r for r in rows if r["lead_ms"] == lead and r.get("deployable") == int(deployable)]
        cand = [r for r in cand if np.isfinite(r.get("auc", np.nan))]
        if require_practical:
            cand = [r for r in cand if practical_ok(r)]
        if not cand:
            return None
        return max(cand, key=lambda r: (r["auc"], abs(r["cohens_d"] if np.isfinite(r["cohens_d"]) else 0)))

    def meets(r, auc_min, dmin):
        return (
            r is not None
            and np.isfinite(r["auc"])
            and r["auc"] >= auc_min
            and abs(r["cohens_d"]) >= dmin
            and practical_ok(r)
        )

    # Primary scientific target: μ=0.2 fail vs μ=0.5 success (the 4 N decision boundary).
    # Pooled 0.5+1.0 inflates hold-phase AUC because μ=1.0 is slightly different at rest.
    bound_rows = []
    for r in friction_rows:
        if r["task"] != "mu0.2_vs_0.5":
            continue
        bound_rows.append({
            "lead_ms": r["lead_ms"],
            "signal": r["signal"],
            "deployable": r["deployable"],
            "auc": r["auc"],
            "cohens_d": r["cohens_d"],
            "fail_mean": r["mean_a"],
            "succ_mean": r["mean_b"],
            "overlap": r["overlap"],
        })

    if bound_rows:
        with (OUT / "DECISION_BOUNDARY_SEPARABILITY.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bound_rows[0].keys()))
            w.writeheader()
            w.writerows(bound_rows)

    def earliest(deployable, auc_min=USEFUL_AUC, dmin=USEFUL_D, rows=None, signals=None):
        use = bound_rows if rows is None else rows
        if signals is not None:
            use = [r for r in use if r["signal"] in signals]
        for lead in LEADS_MS:
            b = best_at(lead, deployable, rows=use)
            if meets(b, auc_min, dmin):
                return lead, b
        return None, None

    e_dep, b_dep = earliest(True)
    e_priv, b_priv = earliest(False, signals=["rel_z", "v_rel_z", "rel_xy"])
    e_dep_weak, b_dep_weak = earliest(True, WEAK_AUC, WEAK_D)
    e_priv_weak, b_priv_weak = earliest(False, WEAK_AUC, WEAK_D, signals=["rel_z", "v_rel_z", "rel_xy"])
    e_dep_pooled, b_dep_pooled = earliest(True, rows=sep_rows)

    if sep_rows:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for sig, lab, ls in [
            ("f_meas", "squeeze (0.2 vs 0.5)", "-"),
            ("marker_mean", "marker mean (0.2 vs 0.5)", "-"),
            ("marker_vel", "marker vel (0.2 vs 0.5)", "-"),
            ("hm_mean", "height-map (0.2 vs 0.5)", "-"),
            ("rel_z", "rel_z priv (0.2 vs 0.5)", "--"),
            ("v_rel_z", "v_rel_z priv (0.2 vs 0.5)", "--"),
        ]:
            xs, ys, xf, yf = [], [], [], []
            for lead in LEADS_MS:
                rr = [r for r in bound_rows if r["signal"] == sig and r["lead_ms"] == lead]
                if rr and np.isfinite(rr[0]["auc"]):
                    xs.append(lead)
                    ys.append(rr[0]["auc"])
                    if practical_ok(rr[0]) and abs(rr[0]["cohens_d"]) >= USEFUL_D:
                        xf.append(lead)
                        yf.append(rr[0]["auc"])
            ax.plot(xs, ys, marker="o", ls=ls, alpha=0.35, label=None)
            ax.plot(xf, yf, marker="o", ls=ls, label=lab)
        ax.axhline(0.7, ls="--", c="gray", lw=0.8, label="AUC=0.70")
        ax.invert_xaxis()
        ax.set_xlabel("lead time before T_slip (ms)")
        ax.set_ylabel("single-signal ROC-AUC (μ=0.2 fail vs μ=0.5 success)")
        ax.set_ylim(0.4, 1.02)
        ax.legend(fontsize=8, ncol=2)
        ax.set_title("Solid = practical gap; faint = rank-AUC only (tiny offsets ignored)")
        fig.tight_layout()
        fig.savefig(PLOT / "lead_time_auc.png", dpi=140)
        fig.savefig(PLOT / "lead_time_auc.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        dep, priv = [], []
        for lead in LEADS_MS:
            bd = best_at(lead, True, rows=bound_rows)
            bp = best_at(lead, False, rows=[r for r in bound_rows if r["signal"] in ("rel_z", "v_rel_z", "rel_xy")])
            dep.append(bd["auc"] if bd and practical_ok(bd) else np.nan)
            priv.append(bp["auc"] if bp and practical_ok(bp) else np.nan)
        ax.plot(LEADS_MS, dep, marker="o", label="best deployable (practical)")
        ax.plot(LEADS_MS, priv, marker="s", label="best privileged rel motion (practical)")
        ax.axhline(0.7, ls="--", c="gray")
        ax.invert_xaxis()
        ax.set_xlabel("lead ms before T_slip")
        ax.set_ylabel("best single-signal AUC")
        ax.set_ylim(0.4, 1.02)
        ax.legend(fontsize=8)
        ax.set_title("Pre-slip window: 4 N fail vs μ=0.5; tiny hold-phase rank-AUC omitted")
        fig.tight_layout()
        fig.savefig(PLOT / "preslip_window_summary.png", dpi=140)
        fig.savefig(PLOT / "preslip_window_summary.pdf")
        plt.close(fig)

    def useful(group, lead_need=200, auc_min=USEFUL_AUC, rows=None):
        rows = bound_rows if rows is None else rows
        leads = [r for r in rows if r["signal"] in group and r["lead_ms"] >= lead_need]
        return any(
            np.isfinite(r["auc"])
            and r["auc"] >= auc_min
            and abs(r.get("cohens_d", 0) or 0) >= USEFUL_D
            and practical_ok(r)
            for r in leads
        )

    # classification (do not manufacture a window)
    status = "P1_NO_USEFUL_PRESLIP_WINDOW"
    if e_dep is not None and e_dep >= 200:
        status = "P1_STRONG_PRESLIP_WINDOW_FOUND"
    elif e_dep_weak is not None and e_dep_weak >= 200:
        # 200 ms with only weak AUC is still late-ish / weak
        status = "P1_WEAK_LATE_PRESLIP_SIGNAL"
    elif (e_dep is not None and e_dep < 200) or (e_dep_weak is not None and e_dep_weak < 200):
        status = "P1_WEAK_LATE_PRESLIP_SIGNAL"
    elif e_priv is not None and e_priv >= 50 and (e_dep is None or (e_dep_weak or 0) < 50):
        status = "P1_WEAK_LATE_PRESLIP_SIGNAL"

    tactile_on = any(_f(e.get("tactile")) == 1 for e in events)

    # friction-identifiable? best deployable AUC for 0.2 vs 0.5 at >=200 ms
    def friction_ident(lead_min=200):
        def best_task(task):
            cand = [
                r for r in friction_rows
                if r["lead_ms"] >= lead_min and r["deployable"] == 1 and r["task"] == task
                and np.isfinite(r["auc"]) and practical_ok(r)
            ]
            return max((r["auc"] for r in cand), default=np.nan)
        best_fail_mid = best_task("mu0.2_vs_0.5")
        best_mid_high = best_task("mu0.5_vs_1.0")
        # 3-way ID at the earliest deployable failure-prediction lead, if any
        TAC = {"marker_mean", "marker_vel", "hm_mean"}
        if e_dep is not None:
            mid_at_edep = max(
                (
                    r["auc"] for r in friction_rows
                    if r["lead_ms"] == e_dep and r["deployable"] == 1 and r["task"] == "mu0.5_vs_1.0"
                    and r["signal"] in TAC
                    and np.isfinite(r["auc"]) and practical_ok(r)
                ),
                default=np.nan,
            )
            if np.isfinite(best_fail_mid) and best_fail_mid >= 0.7:
                if np.isfinite(mid_at_edep) and mid_at_edep >= 0.7:
                    return "YES"
                return "PARTIAL"
        if np.isfinite(best_fail_mid) and best_fail_mid >= 0.7:
            return "PARTIAL"
        return "NO"

    fric_id = friction_ident(200)

    primary = [
        "Primary comparison is μ=0.2 vs μ=0.5 (4 N decision boundary), not pooled success with μ=1.0.",
        "Hold-phase (500–1000 ms lead) rank-AUC can look high from tiny offsets / μ=1.0; those are rejected by practical floors.",
    ]
    if b_dep:
        primary.append(
            f"best deployable at {e_dep} ms: {b_dep['signal']} AUC={b_dep['auc']:.3f} d={b_dep['cohens_d']:.2f} "
            f"fail_mean={b_dep.get('fail_mean')} succ_mean={b_dep.get('succ_mean')}"
        )
    if b_priv:
        primary.append(
            f"best privileged at {e_priv} ms: {b_priv['signal']} AUC={b_priv['auc']:.3f} d={b_priv['cohens_d']:.2f}"
        )
    primary.append(f"μ=0.20 future-fail rate={fail_rate.get(0.2)}; μ=0.50={fail_rate.get(0.5)}; μ=1.00={fail_rate.get(1.0)}")
    primary.append(
        f"median lift→slip={np.median(lift_to_slip) if lift_to_slip else None} ms; "
        f"slip→loss={np.median(slip_to_loss) if slip_to_loss else None} ms"
    )
    if e_dep_pooled is not None:
        primary.append(
            f"pooled (0.5+1.0) deployable would have claimed {e_dep_pooled} ms via "
            f"{None if not b_dep_pooled else b_dep_pooled['signal']}; rejected as μ=1.0 / tiny-offset artifact"
        )

    (OUT / "DEPLOYABLE_VS_PRIVILEGED_SIGNALS.md").write_text(
        "\n".join([
            "# Deployable vs privileged signals",
            "",
            "Privileged object pose / velocity / relative motion are **SIMULATOR_PRIVILEGED_SIGNAL**.",
            "They help understanding but are not future policy observations unless a real estimator exists.",
            "",
            "## Deployable (force / tactile / proprioception)",
            ", ".join(DEPLOYABLE),
            "",
            "## Privileged (simulator object state)",
            ", ".join(PRIVILEGED),
            "",
            "## What counts as useful",
            "Primary label is **future 4 N failure**, operationalized as μ=0.20 vs μ=0.50",
            "(the hidden-friction decision region). Pooled success (0.50+1.00) is reported but not used",
            "to declare a window, because μ=1.00 is slightly different even during hold and inflates rank-AUC.",
            "",
            "A signal is useful only if AUC≥0.70, |Cohen's d|≥0.50, **and** the mean gap exceeds a practical floor",
            "(e.g. 0.40 N for squeeze, 0.20 for marker mean). High AUC on a 0.04 N or 0.4% marker offset is not a window.",
            "",
            f"Tactile enabled in collection: {tactile_on}",
            f"Best deployable useful lead (μ=0.2 vs 0.5): {e_dep} ms"
            + (f" via `{b_dep['signal']}` AUC={b_dep['auc']:.3f} d={b_dep['cohens_d']:.2f}" if b_dep else ""),
            f"Best privileged useful lead: {e_priv} ms"
            + (f" via `{b_priv['signal']}` AUC={b_priv['auc']:.3f} d={b_priv['cohens_d']:.2f}" if b_priv else ""),
            f"Weak deployable lead: {e_dep_weak} ms",
            f"Pooled-success artifact lead (not used): {e_dep_pooled} ms",
            "",
            "Single-signal ROC-AUC and 1-feature logistic (even/odd seed split) are **diagnostic probes**, not a learned method.",
            "GT friction and future fail/success are analysis labels only; they are not features.",
        ])
        + "\n"
    )

    steps_before_slip = None if e_dep is None else int(round(e_dep / 50.0))
    steps_before_loss = None
    if e_dep is not None and slip_to_loss:
        steps_before_loss = int(round((e_dep + float(np.median(slip_to_loss))) / 50.0))

    verdict = {
        "status": status,
        "method_change": "NONE",
        "task": "libero_object_1",
        "force_N": 4,
        "frictions": [0.2, 0.5, 1.0],
        "matched_seed_count": len(matched),
        "n_events": len(events),
        "tactile_enabled": tactile_on,
        "low_friction_future_failure_rate": fail_rate.get(0.2),
        "mid_friction_future_failure_rate": fail_rate.get(0.5),
        "high_friction_future_failure_rate": fail_rate.get(1.0),
        "gt_slip_defined": True,
        "gt_slip_rule": "F1R2/R1-v1: rel_z<-8mm OR v_rel_z<-0.05 for 2 steps OR rel_xy>15mm OR contact loss on lift OR dropped",
        "gt_contact_loss_defined": True,
        "median_slip_to_loss_ms": float(np.median(slip_to_loss)) if slip_to_loss else None,
        "median_lift_to_slip_ms": float(np.median(lift_to_slip)) if lift_to_slip else None,
        "median_lift_to_loss_ms": float(np.median(lift_to_loss)) if lift_to_loss else None,
        "median_lift_to_micro_ms": float(np.median(micro_from_lift)) if micro_from_lift else None,
        "deployable_signals_tested": DEPLOYABLE,
        "privileged_signals_tested": PRIVILEGED,
        "primary_comparison": "mu0.2_fail_vs_mu0.5_success",
        "q1_500ms_before_slip_deployable_separates": bool(e_dep is not None and e_dep >= 500),
        "q2_200ms_before_slip_deployable_separates": bool(e_dep is not None and e_dep >= 200),
        "q3_only_50ms": bool(e_dep is not None and e_dep < 150),
        "q4_only_at_or_after_T_slip": bool(e_dep is None),
        "earliest_useful_deployable_lead_ms": e_dep,
        "earliest_useful_privileged_lead_ms": e_priv,
        "earliest_weak_deployable_lead_ms": e_dep_weak,
        "pooled_success_artifact_lead_ms": e_dep_pooled,
        "earliest_useful_deployable_lead_steps": steps_before_slip,
        "control_steps_before_contact_loss_if_act_at_deployable_lead": steps_before_loss,
        "best_deployable_signal": None if b_dep is None else {
            "signal": b_dep["signal"], "auc": b_dep["auc"], "d": b_dep["cohens_d"], "lead_ms": e_dep,
        },
        "best_privileged_signal": None if b_priv is None else {
            "signal": b_priv["signal"], "auc": b_priv["auc"], "d": b_priv["cohens_d"], "lead_ms": e_priv,
        },
        "force_signal_useful": useful(["f_meas", "dF", "imbalance", "f_tangential", "contact_force"]),
        "tactile_signal_useful": useful(["marker_mean", "marker_max", "marker_vel", "marker_tangential", "marker_asym", "marker_acc"]),
        "motion_signal_useful": useful(["rel_z", "v_rel_z", "rel_xy"], lead_need=50),
        "future_failure_predictable_before_gross_slip": bool(e_dep is not None and e_dep >= 200),
        "friction_identifiable_before_gross_slip": fric_id,
        "online_adaptation_feasible_from_current_observations": bool(e_dep is not None and e_dep >= 200),
        "potential_intervention_window_ms": None if e_dep is None else {
            "from_signal_to_T_slip_ms": e_dep,
            "from_signal_to_T_loss_ms": None if not slip_to_loss else e_dep + float(np.median(slip_to_loss)),
            "note": "NOT a rescue experiment. Only the theoretical time left if F* were changed at that instant.",
        },
        "primary_evidence": primary,
        "limitations": [
            "Single-signal AUC / 1-feature logistic is a diagnostic, not a trained method.",
            "Object pose/velocity/relative motion are privileged simulator state.",
            "GT slip is the F1R2 8 mm / -0.05 m/s rule; not retuned to manufacture a window.",
            "Scripted 4 N grasp, not Tabero π0.",
            "Lead times longer than lift→slip (~475 ms) fall into the hold phase; that is reported, not dropped.",
            "Pooled μ=0.5+1.0 success can inflate hold-phase rank-AUC; primary verdict uses μ=0.2 vs μ=0.5 plus practical mean-gap floors.",
            "The tactile cue at ~300 ms is incipient-slip unloading after lift onset, not a pre-lift friction ID during hold.",
        ],
    }
    (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2, default=float) + "\n")
    print("status", status, "matched", len(matched), "tactile", tactile_on)
    print("fail rates", fail_rate)
    print("median lift->slip ms", verdict["median_lift_to_slip_ms"], "slip->loss", verdict["median_slip_to_loss_ms"])
    print("earliest deployable", e_dep, None if not b_dep else b_dep["signal"], "priv", e_priv)


if __name__ == "__main__":
    main()
