#!/usr/bin/env python3
"""D2: hierarchical 6N veto + mid/high split. METHOD_CHANGE=NONE."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
D1 = Path("/home/exouser/Tabero/analysis/results/d1_force_decision_resolution_20260819_223557")
PLOT = OUT / "plots"
PLOT.mkdir(exist_ok=True)

FSTAR = {0.2: 6.0, 0.5: 5.0, 1.0: 4.0}
Z1, Z2 = "imb_peak", "ftan_hyst"


def _f(x, d=np.nan):
    try:
        if x in ("", "None", None):
            return d
        return float(x)
    except Exception:
        return d


def load_split(name):
    p = D1 / name
    return list(csv.DictReader(p.open()))


def decide(row, th_low, th_hyst):
    z1, z2 = _f(row[Z1]), _f(row[Z2])
    # Stage 1: low friction veto — small imbalance => must 6N
    if z1 <= th_low:
        return 6.0, "stage1_low_veto", z1, z2
    # Stage 2: large hysteresis => high friction => 4N; else 5N (ambiguous -> 5N)
    if z2 > th_hyst:
        return 4.0, "stage2_high", z1, z2
    return 5.0, "stage2_mid", z1, z2


def metrics(rows, th_low, th_hyst):
    recs = []
    for r in rows:
        F, rule, z1, z2 = decide(r, th_low, th_hyst)
        mu = _f(r["friction"])
        fs = FSTAR[mu]
        recs.append({
            "seed_idx": r["seed_idx"], "gt_mu": mu, "split": r.get("split", ""),
            "z1": z1, "z2": z2, "rule": rule,
            "chosen_F": F, "fstar": fs,
            "exact": int(abs(F - fs) < 1e-6),
            "under": int(F < fs - 1e-6),
            "over": int(F > fs + 1e-6),
        })
    n = len(recs)
    low = [x for x in recs if abs(x["gt_mu"] - 0.2) < 1e-6]
    miss_low = sum(1 for x in low if x["rule"] != "stage1_low_veto")
    return recs, {
        "n": n,
        "exact": float(np.mean([x["exact"] for x in recs])) if n else np.nan,
        "under": float(np.mean([x["under"] for x in recs])) if n else np.nan,
        "over": float(np.mean([x["over"] for x in recs])) if n else np.nan,
        "mean_F": float(np.mean([x["chosen_F"] for x in recs])) if n else np.nan,
        "low_miss_rate": miss_low / max(len(low), 1),
        "low_miss_n": miss_low,
    }


def pick_thresholds(calib):
    low_z1 = sorted(_f(r[Z1]) for r in calib if abs(_f(r["friction"]) - 0.2) < 1e-6)
    mid_z2 = sorted(_f(r[Z2]) for r in calib if abs(_f(r["friction"]) - 0.5) < 1e-6)
    high_z2 = sorted(_f(r[Z2]) for r in calib if abs(_f(r["friction"]) - 1.0) < 1e-6)

    # Stage 1: imb <= th => LOW => 6N. Conservative: th = max(low) so all μ=0.20 vetoed.
    th_low = float(max(low_z1)) if low_z1 else 0.75

    # Stage 2: hyst > th => HIGH => 4N. Conservative: th below min(high) so ambiguous mid stays 5N.
    # Prefer 5N when ambiguous: set th at midpoint between max(mid) and min(high), biased toward high.
    mid_max = float(max(mid_z2)) if mid_z2 else 0.003
    high_min = float(min(high_z2)) if high_z2 else 0.02
    th_hyst = float(0.5 * (mid_max + high_min))  # midpoint; all mid <= mid_max < th typically

    # Verify calib: zero missed low
    _, m = metrics(calib, th_low, th_hyst)
    if m["low_miss_n"] > 0:
        th_low = float(max(low_z1)) + 1e-6

    return {
        "stage1_signal": Z1,
        "stage1_rule": "if imb_peak <= theta_low then FORCE=6N and STOP",
        "theta_low": th_low,
        "stage1_calib_low_imb_min": float(min(low_z1)),
        "stage1_calib_low_imb_max": float(max(low_z1)),
        "stage2_signal": Z2,
        "stage2_rule": "else if ftan_hyst > theta_mid_high then 4N else 5N",
        "theta_mid_high": th_hyst,
        "stage2_calib_mid_hyst_max": mid_max,
        "stage2_calib_high_hyst_min": high_min,
        "calib_split": "even_seeds_probe_belief_A",
        "eval_split": "odd_seeds_probe_belief_A",
    }


def b0_decide(row):
    z = _f(row[Z1])
    means = {0.2: 0.5622988700866699, 0.5: 0.8634264469146729, 1.0: 1.159930419921875}
    stds = {0.2: 0.11864785105282566, 0.5: 0.1833187067922804, 1.0: 0.21181580260148877}
    succ = {4: {0.2: 0.0, 0.5: 0.55, 1.0: 1.0}, 5: {0.2: 0.0, 0.5: 0.95, 1.0: 1.0}, 6: {0.2: 0.8, 0.5: 1.0, 1.0: 1.0}}
    un = {}
    for mu in (0.2, 0.5, 1.0):
        s = stds[mu]
        like = np.exp(-0.5 * ((z - means[mu]) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        un[mu] = like / 3
    zsum = sum(un.values()) + 1e-18
    post = {mu: un[mu] / zsum for mu in un}
    p4 = 1 - post[0.2]
    F = 4.0 if p4 >= 0.8 else 6.0
    return F


def summarize_method(recs):
    n = len(recs)
    by = {}
    for mu in (0.2, 0.5, 1.0):
        xs = [r for r in recs if abs(r["gt_mu"] - mu) < 1e-6]
        counts = {4: 0, 5: 0, 6: 0}
        for r in xs:
            counts[int(r["chosen_F"])] = counts.get(int(r["chosen_F"]), 0) + 1
        by[str(mu)] = {
            "n": len(xs),
            "exact": float(np.mean([r["exact"] for r in xs])) if xs else np.nan,
            "under": float(np.mean([r["under"] for r in xs])) if xs else np.nan,
            "over": float(np.mean([r["over"] for r in xs])) if xs else np.nan,
            "mean_F": float(np.mean([r["chosen_F"] for r in xs])) if xs else np.nan,
            "counts": counts,
        }
    return {
        "n": n,
        "exact": float(np.mean([r["exact"] for r in recs])) if n else np.nan,
        "under": float(np.mean([r["under"] for r in recs])) if n else np.nan,
        "over": float(np.mean([r["over"] for r in recs])) if n else np.nan,
        "mean_F": float(np.mean([r["chosen_F"] for r in recs])) if n else np.nan,
        "by_mu": by,
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calib = load_split("CALIBRATION_SPLIT.csv")
    held = load_split("HELDOUT_SPLIT.csv")
    th = pick_thresholds(calib)
    (OUT / "CALIBRATION_THRESHOLDS.json").write_text(json.dumps(th, indent=2) + "\n")

    spec = [
        "# Hierarchical force decision (D2)",
        "",
        "Frozen probe: P3 A (+2 mm base-Y, 4 N grasp). No joint NB.",
        "",
        "## Stage 1 — 6 N safety veto",
        f"Signal: `{Z1}` (peak left/right force imbalance during probe).",
        f"Rule: if `imb_peak <= {th['theta_low']:.6g}` → **6 N** and STOP.",
        "Rationale: low μ shows smaller probe imbalance; conservative θ = max(calib low μ).",
        "",
        "## Stage 2 — Mid vs high (only if NOT stage-1 LOW)",
        f"Signal: `{Z2}` (mean |F_tan_forward − F_tan_return| during probe).",
        f"Rule: if `ftan_hyst > {th['theta_mid_high']:.6g}` → **4 N**; else **5 N**.",
        "Ambiguous → 5 N (conservative for under-force).",
        "",
        f"Thresholds fit on **even** calibration seeds only.",
        f"Held-out **odd** seeds evaluated once.",
    ]
    (OUT / "HIERARCHICAL_RULE_SPEC.md").write_text("\n".join(spec) + "\n")

    hier_recs, hier_m = metrics(held, th["theta_low"], th["theta_mid_high"])

    # baselines on same held rows
    b0_recs, b1_recs, b2_recs, b4_recs = [], [], [], []
    for r in held:
        mu = _f(r["friction"])
        fs = FSTAR[mu]
        for tag, F in [
            ("current_p3", b0_decide(r)),
            ("fixed6", 6.0),
            ("oracle", fs),
        ]:
            rec = {
                "seed_idx": r["seed_idx"], "gt_mu": mu, "chosen_F": F, "fstar": fs,
                "exact": int(abs(F - fs) < 1e-6),
                "under": int(F < fs - 1e-6),
                "over": int(F > fs + 1e-6),
            }
            if tag == "current_p3":
                b0_recs.append(rec)
            elif tag == "fixed6":
                b1_recs.append(rec)
            else:
                b4_recs.append(rec)
    # D1 joint NB from D1 heldout file
    d1h = list(csv.DictReader((D1 / "REFINED_BELIEF_DECISIONS.csv").open()))
    for r in d1h:
        b2_recs.append({
            "seed_idx": r["seed_idx"], "gt_mu": _f(r["gt_mu"]),
            "chosen_F": _f(r["chosen_F"]), "fstar": _f(r["fstar"]),
            "exact": int(r["exact"]), "under": int(r["under"]), "over": int(r["over"]),
        })

    with (OUT / "OFFLINE_HELDOUT_DECISIONS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hier_recs[0].keys()))
        w.writeheader()
        w.writerows(hier_recs)

    cmp_rows = []
    for name, recs in [
        ("fixed6", b1_recs), ("current_p3_1d", b0_recs), ("d1_joint_nb", b2_recs),
        ("hierarchical", hier_recs), ("oracle", b4_recs),
    ]:
        s = summarize_method(recs)
        cmp_rows.append({"method": name, **{k: s[k] for k in ("n", "exact", "under", "over", "mean_F")}})
    with (OUT / "BASELINE_COMPARISON.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader()
        w.writerows(cmp_rows)

    err = []
    for r in hier_recs:
        err.append({**r, "method": "hierarchical"})
    for r in b0_recs:
        err.append({**r, "method": "current_p3", "rule": "nb1d"})
    with (OUT / "OFFLINE_ERROR_ANALYSIS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(err[0].keys()))
        w.writeheader()
        w.writerows(err)

    gate = (
        hier_m["under"] == 0
        and hier_m["low_miss_rate"] == 0
        and hier_m["over"] <= 0.15
        and hier_m["mean_F"] <= 5.2
        and hier_m["exact"] >= 0.85
    )

    # plots
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for mu, c in ((0.2, "C3"), (0.5, "C0"), (1.0, "C2")):
        xs = [_f(r[Z1]) for r in calib + held if abs(_f(r["friction"]) - mu) < 1e-6]
        ax.scatter(np.random.default_rng(0).normal(mu, 0.02, len(xs)), xs, c=c, s=24, label=f"μ={mu:g}")
    ax.axhline(th["theta_low"], ls="--", c="C3", lw=0.9, label=f"θ_low={th['theta_low']:.3f}")
    ax.set_xlabel("friction"); ax.set_ylabel("imb_peak"); ax.legend(fontsize=7)
    ax.set_title("Stage 1: 6N veto threshold")
    fig.tight_layout()
    fig.savefig(PLOT / "stage1_low_veto.png", dpi=140)
    fig.savefig(PLOT / "stage1_low_veto.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for mu, c in ((0.5, "C0"), (1.0, "C2")):
        xs = [_f(r[Z2]) for r in calib + held if abs(_f(r["friction"]) - mu) < 1e-6]
        ax.scatter(np.random.default_rng(1).normal(mu, 0.02, len(xs)), xs, c=c, s=24, label=f"μ={mu:g}")
    ax.axhline(th["theta_mid_high"], ls="--", c="gray", lw=0.9, label=f"θ={th['theta_mid_high']:.4f}")
    ax.set_xlabel("friction"); ax.set_ylabel("ftan_hyst"); ax.legend(fontsize=7)
    ax.set_title("Stage 2: mid (5N) vs high (4N)")
    fig.tight_layout()
    fig.savefig(PLOT / "stage2_mid_high.png", dpi=140)
    fig.savefig(PLOT / "stage2_mid_high.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for mu, c in ((0.2, "C3"), (0.5, "C0"), (1.0, "C2")):
        xs = [r["chosen_F"] for r in hier_recs if abs(r["gt_mu"] - mu) < 1e-6]
        ax.scatter(np.random.default_rng(2).normal(mu, 0.03, len(xs)), xs, c=c, s=32, label=f"μ={mu:g}")
    ax.set_yticks([4, 5, 6]); ax.set_title("Hierarchical selected force (held-out)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT / "selected_force_by_friction.png", dpi=140)
    fig.savefig(PLOT / "selected_force_by_friction.pdf")
    plt.close(fig)

    labs = ["current P3", "D1 NB", "hier", "fixed6", "oracle"]
    ms = [summarize_method(b0_recs), summarize_method(b2_recs), summarize_method(hier_recs),
          summarize_method(b1_recs), summarize_method(b4_recs)]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(5); w = 0.25
    ax.bar(x - w, [m["exact"] for m in ms], w, label="exact")
    ax.bar(x, [m["under"] for m in ms], w, label="under")
    ax.bar(x + w, [m["over"] for m in ms], w, label="over")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=15); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("Held-out decision errors")
    fig.tight_layout()
    fig.savefig(PLOT / "under_over_exact.png", dpi=140)
    fig.savefig(PLOT / "under_over_exact.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(labs, [m["mean_F"] for m in ms])
    ax.set_ylabel("mean selected F (N)"); ax.set_ylim(3.5, 6.5)
    ax.set_title("Mean selected force (held-out)")
    fig.tight_layout()
    fig.savefig(PLOT / "mean_selected_force.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for lab, m, mk in zip(labs, ms, "os^D*"):
        ax.scatter([m["mean_F"]], [m["exact"]], s=70, marker=mk, label=lab)
    ax.set_xlabel("mean F"); ax.set_ylabel("exact accuracy"); ax.legend(fontsize=8)
    ax.set_title("Force vs exact (held-out)")
    fig.tight_layout()
    fig.savefig(PLOT / "success_force_tradeoff.png", dpi=140)
    plt.close(fig)

    summary = {
        "thresholds": th,
        "heldout_hierarchical": hier_m,
        "heldout_hierarchical_by_mu": summarize_method(hier_recs)["by_mu"],
        "gate_passed": gate,
        "comparison": cmp_rows,
    }
    (OUT / "OFFLINE_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
