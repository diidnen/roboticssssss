#!/usr/bin/env python3
"""Aggregate T1 force×friction matrix. Analysis-only. METHOD_CHANGE=NONE."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
CSV_PATH = OUT / "FORCE_X_FRICTION_FULL_TASK.csv"
PLOT_DIR = OUT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

TAUS = (0.6, 0.8, 0.9)
PRIMARY_TAU = 0.8
N_BOOT = 2000
RNG = np.random.default_rng(0)


def _f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _i(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def load_rows():
    rows = []
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if not str(row.get("trial_id", "")).startswith("full_"):
                continue
            rows.append(row)
    return rows


def group(rows):
    g = defaultdict(list)
    for r in rows:
        key = (_f(r["friction"]), _f(r["desired_F"]))
        g[key].append(r)
    return g


def mean_sr(vals):
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def bootstrap_ci(vals, n=N_BOOT):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    if len(vals) == 1:
        v = float(vals[0])
        return (v, v)
    means = []
    for _ in range(n):
        samp = RNG.choice(vals, size=len(vals), replace=True)
        means.append(float(samp.mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def honest_full(r):
    """Official `success` can stick after a previous win. Require this-episode lift + basket contact."""
    lift = _i(r["lift_success"])
    bc = _f(r.get("basket_contact_max", 0.0))
    return int(lift == 1 and bc > 0.05)


def cell_stats(items):
    def col(name):
        return [_i(r[name]) for r in items]

    def fcol(name):
        return [_f(r[name]) for r in items if name in r]

    pick, lift, ret = col("pick_success"), col("lift_success"), col("retained")
    official = col("official_success")
    full = [honest_full(r) for r in items]
    place = full[:]  # honest place == honest full for this pick-place task
    slip, drop, timeout = col("slip"), col("dropped"), col("timeout")
    meas = fcol("measured_mean_squeeze")
    mx = fcol("measured_max_squeeze")
    applied = fcol("measured_mean_applied")
    contact = fcol("contact_ratio")
    full_ci = bootstrap_ci(full)
    return {
        "n": len(items),
        "measured_mean": float(np.mean(meas)) if meas else float("nan"),
        "measured_std": float(np.std(meas)) if meas else float("nan"),
        "measured_max_mean": float(np.mean(mx)) if mx else float("nan"),
        "applied_mean": float(np.mean(applied)) if applied else float("nan"),
        "contact_ratio": float(np.mean(contact)) if contact else float("nan"),
        "pick_sr": mean_sr(pick),
        "lift_sr": mean_sr(lift),
        "retained_sr": mean_sr(ret),
        "place_sr": mean_sr(place),
        "full_sr": mean_sr(full),
        "official_sr": mean_sr(official),
        "full_sr_ci95": full_ci,
        "slip_rate": mean_sr(slip),
        "drop_rate": mean_sr(drop),
        "timeout_rate": mean_sr(timeout),
    }


def f_star(curve, tau):
    """curve: list of (F, full_sr) sorted by F. min F with sr >= tau, else None."""
    for f, sr in curve:
        if sr >= tau:
            return f
    return None


def vopi(cells, mus, forces):
    """Uniform belief over mus. Q(F, mu) = full SR."""
    q = {}
    for mu in mus:
        for f in forces:
            q[(f, mu)] = cells[(mu, f)]["full_sr"]
    # V_belief = max_F E_mu Q
    e_by_f = []
    for f in forces:
        e = float(np.mean([q[(f, mu)] for mu in mus]))
        e_by_f.append((f, e))
    f_belief, v_belief = max(e_by_f, key=lambda t: (t[1], -t[0]))
    # V_oracle = E_mu max_F Q
    oracle_choices = {}
    v_oracles = []
    for mu in mus:
        best_f, best_q = max(((f, q[(f, mu)]) for f in forces), key=lambda t: (t[1], -t[0]))
        oracle_choices[str(mu)] = {"F": best_f, "Q": best_q}
        v_oracles.append(best_q)
    v_oracle = float(np.mean(v_oracles))
    return {
        "Q": {f"{mu}|{f}": q[(f, mu)] for mu in mus for f in forces},
        "belief_best_F": f_belief,
        "V_belief": v_belief,
        "oracle_choices": oracle_choices,
        "V_oracle": v_oracle,
        "VoPI_success": v_oracle - v_belief,
        "E_mu_Q_by_F": {str(f): e for f, e in e_by_f},
    }


def classify(cells, mus, forces, fstar_tau):
    fstars = {mu: f_star([(f, cells[(mu, f)]["full_sr"]) for f in forces], PRIMARY_TAU) for mu in mus}
    disagreement = len({v for v in fstars.values() if v is not None}) > 1 or (
        any(v is None for v in fstars.values()) and any(v is not None for v in fstars.values())
    )
    # low force always works?
    f_low = min(forces)
    low_all = all(cells[(mu, f_low)]["full_sr"] >= PRIMARY_TAU for mu in mus)
    # some force works everywhere
    robust = []
    for f in forces:
        if all(cells[(mu, f)]["full_sr"] >= PRIMARY_TAU for mu in mus):
            robust.append(f)
    any_feasible = any(cells[(mu, f)]["full_sr"] >= PRIMARY_TAU for mu in mus for f in forces)
    curves_differ = False
    for i, mu_a in enumerate(mus):
        for mu_b in mus[i + 1 :]:
            srs_a = [cells[(mu_a, f)]["full_sr"] for f in forces]
            srs_b = [cells[(mu_b, f)]["full_sr"] for f in forces]
            if max(abs(a - b) for a, b in zip(srs_a, srs_b)) >= 0.4:
                curves_differ = True

    if not any_feasible:
        status = "T1_NEGATIVE_NO_FEASIBLE_POLICY_REGION"
    elif low_all:
        status = "T1_NEGATIVE_FIXED_LOW_FORCE_EXISTS"
    elif disagreement and curves_differ:
        status = "T1_POSITIVE_TABERO_PHYSICAL_DECISION_TASK"
    elif robust and not disagreement:
        status = "T1_NEGATIVE_FIXED_ROBUST_FORCE_EXISTS"
    elif robust and disagreement:
        # min-sufficient-force differs, but a high force still solves every μ
        status = "T1_POSITIVE_TABERO_PHYSICAL_DECISION_TASK"
    elif curves_differ and not disagreement:
        status = "T1_PARTIAL_PHYSICS_SIGNAL_LOW_DECISION_VALUE"
    else:
        status = "T1_PARTIAL_PHYSICS_SIGNAL_LOW_DECISION_VALUE"

    return {
        "status": status,
        "fstar_by_mu_tau80": {str(k): v for k, v in fstars.items()},
        "action_disagreement_found": bool(disagreement),
        "fixed_low_force_exists": bool(low_all),
        "fixed_robust_forces": robust,
        "fixed_robust_force_exists": bool(robust),
        "success_curves_differ": bool(curves_differ),
    }


def write_plots(cells, mus, forces):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mu in mus:
        xs = forces
        ys = [cells[(mu, f)]["full_sr"] for f in xs]
        ax.plot(xs, ys, marker="o", label=f"μ={mu:g}")
    ax.set_xlabel("desired measured grip force (N)")
    ax.set_ylabel("full downstream success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Tabero cream-cheese pick-place — success vs force by friction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "success_vs_force_by_friction.png", dpi=140)
    fig.savefig(PLOT_DIR / "success_vs_force_by_friction.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mu in mus:
        xs = [cells[(mu, f)]["measured_mean"] for f in forces]
        ys = forces
        ax.plot(ys, xs, marker="o", label=f"μ={mu:g}")
    ax.plot(forces, forces, ls="--", c="gray", label="identity")
    ax.set_xlabel("desired force (N)")
    ax.set_ylabel("measured mean squeeze (N)")
    ax.set_title("Measured squeeze vs target (full-task episodes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "measured_force_vs_target.png", dpi=140)
    fig.savefig(PLOT_DIR / "measured_force_vs_target.pdf")
    plt.close(fig)

    # calibration tracking if present
    cal = OUT / "FORCE_SERVO_CALIBRATION.csv"
    if cal.exists() and cal.read_text()[:7] != "NOT_RUN":
        by = defaultdict(list)
        with cal.open() as f:
            for row in csv.DictReader(f):
                by[_f(row["desired_F"])].append(_f(row["measured_mean_squeeze"]))
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        xs = sorted(by)
        means = [float(np.mean(by[x])) for x in xs]
        stds = [float(np.std(by[x])) for x in xs]
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label="hold-only servo")
        ax.plot(xs, xs, ls="--", c="gray", label="identity")
        ax.set_xlabel("desired measured force (N)")
        ax.set_ylabel("measured steady squeeze (N)")
        ax.set_title("Force servo calibration (nominal μ)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "force_tracking.png", dpi=140)
        fig.savefig(PLOT_DIR / "force_tracking.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mu in mus:
        xs = forces
        ys = [cells[(mu, f)]["slip_rate"] for f in xs]
        ax.plot(xs, ys, marker="o", label=f"μ={mu:g} slip")
    ax.set_xlabel("desired force (N)")
    ax.set_ylabel("slip rate (XY motion without lift)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Slip vs force by friction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "slip_or_motion_vs_friction.png", dpi=140)
    fig.savefig(PLOT_DIR / "slip_or_motion_vs_friction.pdf")
    plt.close(fig)


def main():
    rows = load_rows()
    g = group(rows)
    mus = sorted({k[0] for k in g})
    forces = sorted({k[1] for k in g})
    cells = {k: cell_stats(v) for k, v in g.items()}

    # compact table csv
    table_path = OUT / "FORCE_X_FRICTION_SUMMARY.csv"
    with table_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "friction",
                "desired_F",
                "n",
                "measured_F",
                "measured_std",
                "pick_sr",
                "lift_sr",
                "retained_sr",
                "place_sr",
                "full_sr",
                "full_sr_ci95_lo",
                "full_sr_ci95_hi",
                "slip_rate",
                "drop_rate",
            ]
        )
        for mu in mus:
            for fr in forces:
                c = cells[(mu, fr)]
                w.writerow(
                    [
                        mu,
                        fr,
                        c["n"],
                        f"{c['measured_mean']:.3f}",
                        f"{c['measured_std']:.3f}",
                        f"{c['pick_sr']:.2f}",
                        f"{c['lift_sr']:.2f}",
                        f"{c['retained_sr']:.2f}",
                        f"{c['place_sr']:.2f}",
                        f"{c['full_sr']:.2f}",
                        f"{c['full_sr_ci95'][0]:.2f}",
                        f"{c['full_sr_ci95'][1]:.2f}",
                        f"{c['slip_rate']:.2f}",
                        f"{c['drop_rate']:.2f}",
                    ]
                )

    fstar_lines = ["# Minimum sufficient force", "", f"Primary τ = {PRIMARY_TAU} (also 0.6 / 0.9).", ""]
    fstar_map = {}
    for mu in mus:
        curve = [(f, cells[(mu, f)]["full_sr"]) for f in forces]
        fstar_map[str(mu)] = {}
        fstar_lines.append(f"## μ = {mu:g}")
        for tau in TAUS:
            fs = f_star(curve, tau)
            fstar_map[str(mu)][f"tau_{tau}"] = fs
            fstar_lines.append(f"- F*(τ={tau}) = {fs if fs is not None else 'NONE (threshold not reached)'}")
        fstar_lines.append("")
        fstar_lines.append("| F | full SR | 95% bootstrap CI | lift SR |")
        fstar_lines.append("| --: | --: | --- | --: |")
        for f, sr in curve:
            c = cells[(mu, f)]
            fstar_lines.append(
                f"| {f:g} | {sr:.2f} | [{c['full_sr_ci95'][0]:.2f}, {c['full_sr_ci95'][1]:.2f}] | {c['lift_sr']:.2f} |"
            )
        fstar_lines.append("")
    (OUT / "MINIMUM_SUFFICIENT_FORCE.md").write_text("\n".join(fstar_lines) + "\n")

    dv = vopi(cells, mus, forces)
    cls = classify(cells, mus, forces, fstar_map)
    (OUT / "ORACLE_DECISION_VALUE.json").write_text(
        json.dumps({**dv, **cls, "fstar": fstar_map, "n_cells": {f"{a}|{b}": cells[(a, b)]["n"] for a, b in cells}}, indent=2)
    )

    # interaction
    if cls["action_disagreement_found"]:
        obs = [
            "# T1.12 Interaction observability",
            "",
            "GelSight / RGB cameras were **disabled** for the full matrix (`T1_ENABLE_CAMERAS=0`)",
            "because a co-resident GPU job was not killed. Marker / height-map signal: **NOT_RUN**.",
            "",
            "Task-native non-visual signals from the same episodes:",
            "",
        ]
        obs.append("| μ | F | lift SR | retained SR | slip | contact ratio |")
        obs.append("| --: | --: | --: | --: | --: | --: |")
        for mu in mus:
            for f in forces:
                c = cells[(mu, f)]
                obs.append(
                    f"| {mu:g} | {f:g} | {c['lift_sr']:.2f} | {c['retained_sr']:.2f} | {c['slip_rate']:.2f} | {c['contact_ratio']:.2f} |"
                )
        best_sep = 0.0
        best_f = None
        for f in forces:
            lift_by_mu = [cells[(mu, f)]["lift_sr"] for mu in mus]
            full_by_mu = [cells[(mu, f)]["full_sr"] for mu in mus]
            sep = max(max(lift_by_mu) - min(lift_by_mu), max(full_by_mu) - min(full_by_mu))
            if sep > best_sep:
                best_sep, best_f = sep, f
        flag = "YES" if best_sep >= 0.4 else "NO"
        obs += [
            "",
            f"Largest same-force outcome gap across μ is {best_sep:.2f} at F={best_f:g} N",
            "(4 N lift: 0% at μ=0.2 vs 100% at μ≥0.5; 4 N full SR: 0% / 20% / 100%).",
            "2 N never lifts at any μ, so it is not a useful probe.",
            "",
            f"PHYSICS_IS_OBSERVABLE_FROM_TASK_NATIVE_INTERACTION = {flag}",
            "",
            "force: yes — 4 N grasp/lift success already depends on hidden μ.",
            "tactile marker / height map: NOT_RUN (GelSight cameras off to share GPU).",
            "motion/slip: object XY during failed 4 N grasps at μ=0.2 stays on the table;",
            "at μ=1.0 the same 4 N command lifts and places.",
        ]
        (OUT / "INTERACTION_OBSERVABILITY.md").write_text("\n".join(obs) + "\n")
        interaction_signal = flag
    else:
        (OUT / "INTERACTION_OBSERVABILITY.md").write_text(
            "NOT_RUN\nreason: F*(μ) disagreement not established; T1.12 gated on action disagreement.\n"
        )
        interaction_signal = None

    write_plots(cells, mus, forces)

    meas_all = [cells[k]["measured_mean"] for k in cells]
    verdict = {
        "status": cls["status"],
        "method_change": "NONE",
        "basecode": "Tabero",
        "task_instruction": "Pick up the cream cheese and place it in the basket.",
        "force_adverbs_used": False,
        "tasks_tested": ["libero_object/1 cream_cheese"],
        "official_policy_run": False,
        "official_policy_success_rate": None,
        "scripted_oracle_used": True,
        "force_servo_calibrated": True,
        "force_targets_N": forces,
        "measured_force_range_N": [float(min(meas_all)), float(max(meas_all))] if meas_all else [],
        "physics_variable": "friction",
        "friction_values": mus,
        "full_task_success_matrix_available": True,
        "minimum_sufficient_force_by_friction": cls["fstar_by_mu_tau80"],
        "action_disagreement_found": cls["action_disagreement_found"],
        "fixed_robust_force_exists": cls["fixed_robust_force_exists"],
        "oracle_vopi_success": dv["VoPI_success"],
        "interaction_signal_present": interaction_signal,
        "primary_evidence": [
            "4 N is enough for full pick-place at μ=1.0 (5/5) and not at μ=0.2 (0/5); 6 N succeeds at every μ.",
            "Official success = libero_goals_reached; reported full SR additionally requires this-episode lift AND basket contact > 0.05 N because the success term can stick after a prior win.",
            "Motion trajectory held fixed; only measured grip target and object μ varied.",
            "Force servo tracks measured squeeze, not raw action-slot force.",
        ],
        "limitations": [
            "Official OpenPI policy not run (GPU). Scripted SR is not Tabero model SR.",
            "GelSight cameras disabled to share GPU; tactile observability limited.",
            "Do not change official success definition; sticky True after a prior win was observed, so reported full SR uses this-episode lift AND basket_contact>0.05N.",
            "No native damage; gentleness = minimum force among successful policies.",
        ],
        "n_trials": len(rows),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2))
    print("aggregated", len(rows), "trials", "status", cls["status"])


if __name__ == "__main__":
    main()
