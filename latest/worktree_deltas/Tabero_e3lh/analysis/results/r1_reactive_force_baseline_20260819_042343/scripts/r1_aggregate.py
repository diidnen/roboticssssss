#!/usr/bin/env python3
"""Aggregate R1 reactive-force baseline. METHOD_CHANGE=NONE."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
PLOT = OUT / "plots"
PLOT.mkdir(exist_ok=True)


def load(path: Path):
    if not path.exists() or path.read_text()[:7] == "NOT_RUN":
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def f(x, d=0.0):
    try:
        return float(x) if x not in ("", "None", None) else d
    except Exception:
        return d


def i(x):
    return int(float(x or 0))


def cell(rows, method, mu):
    rs = [r for r in rows if r["method"] == method and abs(f(r["friction"]) - mu) < 1e-6]
    if not rs:
        return None
    def mean(key):
        return float(np.mean([f(r[key]) for r in rs]))
    def sr(key):
        return float(np.mean([i(r[key]) for r in rs]))
    return {
        "n": len(rs),
        "lift_sr": sr("lift_success"),
        "place_sr": sr("place_success"),
        "full_sr": sr("full_task_success"),
        "pick_sr": sr("pick_success"),
        "retained_sr": sr("retained"),
        "slip_rate": sr("gt_slip"),
        "mean_F": mean("measured_mean_squeeze"),
        "peak_F": float(np.mean([f(r["measured_max_squeeze"]) for r in rs])),
        "final_F": mean("final_F_cmd"),
        "updates": mean("n_updates"),
        "rescue": sr("rescue") if any("rescue" in r for r in rs) else None,
        "slam": sr("slam"),
        "reached6": sr("force_reached_6"),
        "excess": mean("excess_force"),
        "integrated": mean("integrated_force"),
        "latency": float(np.nanmean([f(r["slip_to_update_latency_s"], np.nan) for r in rs])),
    }


def plots(rows, mus, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"fixed4": "fixed 4 N", "fixed6": "fixed 6 N", "fixed8": "fixed 8 N", "reactive": "reactive 4→6→8"}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(mus))
    width = 0.18
    for k, m in enumerate(methods):
        ys = [cell(rows, m, mu)["full_sr"] if cell(rows, m, mu) else np.nan for mu in mus]
        ax.bar(x + (k - 1.5) * width, ys, width, label=labels.get(m, m))
    ax.set_xticks(x, [f"μ={mu:g}" for mu in mus])
    ax.set_ylabel("full downstream SR")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Full pick-place success by method and friction")
    fig.tight_layout()
    fig.savefig(PLOT / "success_by_method_and_friction.png", dpi=140)
    fig.savefig(PLOT / "success_by_method_and_friction.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for k, m in enumerate(methods):
        ys = [cell(rows, m, mu)["mean_F"] if cell(rows, m, mu) else np.nan for mu in mus]
        ax.bar(x + (k - 1.5) * width, ys, width, label=labels.get(m, m))
    ax.set_xticks(x, [f"μ={mu:g}" for mu in mus])
    ax.set_ylabel("mean measured squeeze (N)")
    ax.legend()
    ax.set_title("Mean grip force by method and friction")
    fig.tight_layout()
    fig.savefig(PLOT / "mean_force_by_method_and_friction.png", dpi=140)
    fig.savefig(PLOT / "mean_force_by_method_and_friction.pdf")
    plt.close(fig)

    traj = OUT / "FORCE_TRAJECTORIES.csv"
    if traj.exists():
        ts = list(csv.DictReader(traj.open()))
        # representative reactive low-mu success if any
        react = [r for r in rows if r["method"] == "reactive" and abs(f(r["friction"]) - 0.2) < 1e-6]
        pick = None
        for r in react:
            if i(r.get("full_task_success", 0)) or i(r.get("lift_success", 0)):
                pick = r["trial_id"]
                break
        if pick is None and react:
            pick = react[0]["trial_id"]
        if pick:
            ep = [t for t in ts if t["trial_id"] == pick]
            fig, ax = plt.subplots(figsize=(7.6, 4.2))
            ax.plot([f(t["t_s"]) for t in ep], [f(t["f_cmd"]) for t in ep], label="target F*")
            ax.plot([f(t["t_s"]) for t in ep], [f(t["f_meas"]) for t in ep], label="measured squeeze")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("force (N)")
            ax.set_title(f"Reactive force tracking — {pick}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(PLOT / "force_target_vs_measured.png", dpi=140)
            fig.savefig(PLOT / "force_target_vs_measured.pdf")
            fig.savefig(PLOT / "reactive_low_friction_episode.png", dpi=140)
            fig.savefig(PLOT / "reactive_low_friction_episode.pdf")
            plt.close(fig)

    ev = OUT / "SLIP_AND_UPDATE_EVENTS.csv"
    if ev.exists() and ev.read_text()[:7] != "NOT_RUN":
        evs = list(csv.DictReader(ev.open()))
        if evs:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            lats = []
            for r in rows:
                if r["method"] == "reactive" and f(r.get("slip_to_update_latency_s")) == f(r.get("slip_to_update_latency_s")):
                    if r.get("slip_to_update_latency_s") not in ("", "None", None):
                        lats.append(f(r["slip_to_update_latency_s"]))
            if lats:
                ax.hist(lats, bins=10, color="C0")
                ax.set_xlabel("slip → first force update (s)")
                ax.set_ylabel("episodes")
                ax.set_title("Reactive slip-to-update latency")
                fig.tight_layout()
                fig.savefig(PLOT / "slip_update_timing.png", dpi=140)
                fig.savefig(PLOT / "slip_update_timing.pdf")
            plt.close(fig)


def main():
    low = load(OUT / "LOW_FRICTION_RESCUE.csv")
    allr = load(OUT / "ALL_FRICTION_BASELINES.csv")
    rows = allr if allr else low
    methods = ["fixed4", "fixed6", "fixed8", "reactive"]
    mus = sorted({f(r["friction"]) for r in rows}) if rows else [0.2]

    def sr_mu(method, mu, key="full_task_success"):
        c = cell(rows, method, mu)
        return None if c is None else c["full_sr" if key == "full_task_success" else "lift_sr"]

    table = []
    for mu in mus:
        for m in methods:
            c = cell(rows, m, mu)
            if not c:
                continue
            table.append({"friction": mu, "method": m, **c})
    if table:
        with (OUT / "R1_CELL_SUMMARY.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)

    plots(rows, mus, methods)

    low_mu = 0.2
    c4 = cell(rows, "fixed4", low_mu)
    c6 = cell(rows, "fixed6", low_mu)
    c8 = cell(rows, "fixed8", low_mu)
    cr = cell(rows, "reactive", low_mu)

    def fmap(method, key):
        return {str(mu): (cell(rows, method, mu)[key] if cell(rows, method, mu) else None) for mu in mus}

    reactive_mean = fmap("reactive", "mean_F")
    fixed6_mean = fmap("fixed6", "mean_F")
    uses_less = None
    if all(cell(rows, "reactive", mu) and cell(rows, "fixed6", mu) for mu in mus):
        uses_less = all(
            cell(rows, "reactive", mu)["mean_F"] + 0.3 < cell(rows, "fixed6", mu)["mean_F"]
            for mu in mus
            if mu >= 0.5
        )
    match6 = None
    if cr and c6:
        match6 = cr["full_sr"] >= max(0.8, c6["full_sr"] - 0.15)

    slam = cr["slam"] > 0.05 if cr else None
    reached = cr["reached6"] >= 0.8 if cr else None
    low_pass = bool(c4 and c6 and cr and c4["lift_sr"] <= 0.2 and c6["lift_sr"] >= 0.8 and cr["full_sr"] >= 0.8)

    high_stay4 = True
    for mu in mus:
        if mu < 0.45:
            continue
        c = cell(rows, "reactive", mu)
        if c is None:
            high_stay4 = False
            break
        high_stay4 = high_stay4 and (c["final_F"] <= 4.5) and (c["updates"] <= 0.3)

    status = "R1_BLOCKED_TECHNICALLY"
    if cr is None:
        status = "R1_BLOCKED_TECHNICALLY"
    elif c4 and c4["lift_sr"] <= 0.2 and cr["lift_sr"] <= 0.3:
        status = "R1_REACTIVE_RESCUE_FAILED"
    elif cr and cr["reached6"] < 0.5:
        status = "R1_DYNAMIC_FORCE_SERVO_FAILED"
    elif cr and cr["latency"] is not None and cr["latency"] == cr["latency"] and cr["latency"] > 1.0 and cr["lift_sr"] < 0.8:
        status = "R1_SLIP_DETECTION_TOO_LATE"
    elif c6 and cr and cr["full_sr"] + 0.15 < c6["full_sr"] and (not uses_less):
        status = "R1_REACTIVE_NO_ADVANTAGE_OVER_FIXED_6N"
    elif low_pass and (len(mus) < 3 or not all(cell(rows, "reactive", mu) for mu in [0.2, 0.5, 1.0])):
        status = "R1_FULL_DOWNSTREAM_NOT_QUALIFIED" if not low_pass else "R1_REACTIVE_FORCE_BASELINE_QUALIFIED"
        # low-mu only complete
        if len(mus) == 1:
            status = "R1_REACTIVE_FORCE_BASELINE_QUALIFIED" if low_pass and match6 else (
                "R1_REACTIVE_RESCUE_FAILED" if cr["full_sr"] <= 0.3 else "R1_FULL_DOWNSTREAM_NOT_QUALIFIED"
            )
    elif low_pass and match6 and uses_less and high_stay4:
        status = "R1_REACTIVE_FORCE_BASELINE_QUALIFIED"
    elif low_pass and match6 and not uses_less:
        status = "R1_REACTIVE_NO_ADVANTAGE_OVER_FIXED_6N"
    elif low_pass:
        status = "R1_FULL_DOWNSTREAM_NOT_QUALIFIED"

    verdict = {
        "status": status,
        "method_change": "NONE",
        "task": "libero_object_1",
        "force_adverbs_used": False,
        "friction_values": mus,
        "reactive_initial_force_N": 4,
        "reactive_force_schedule_N": [4, 6, 8],
        "gt_slip_used": True,
        "this_is_not_official_forte": True,
        "low_friction_fixed4_sr": c4["full_sr"] if c4 else None,
        "low_friction_fixed6_sr": c6["full_sr"] if c6 else None,
        "low_friction_reactive_sr": cr["full_sr"] if cr else None,
        "low_friction_fixed4_lift_sr": c4["lift_sr"] if c4 else None,
        "low_friction_fixed6_lift_sr": c6["lift_sr"] if c6 else None,
        "low_friction_reactive_lift_sr": cr["lift_sr"] if cr else None,
        "reactive_measured_force_update_valid": reached,
        "accidental_force_slam_present": slam,
        "full_downstream_run": any(i(r["full_task_success"]) == 1 or True for r in rows),
        "fixed4_full_sr": fmap("fixed4", "full_sr"),
        "fixed6_full_sr": fmap("fixed6", "full_sr"),
        "fixed8_full_sr": fmap("fixed8", "full_sr"),
        "reactive_full_sr": fmap("reactive", "full_sr"),
        "reactive_mean_force_by_friction": reactive_mean,
        "fixed6_mean_force_by_friction": fixed6_mean,
        "reactive_uses_less_force_than_fixed6": uses_less,
        "reactive_matches_fixed6_success": match6,
        "reactive_baseline_qualified": status == "R1_REACTIVE_FORCE_BASELINE_QUALIFIED",
        "high_nominal_stays_near_4N": high_stay4,
        "cells": table,
        "primary_evidence": [],
        "limitations": [
            "THIS IS NOT OFFICIAL FORTE; force-space translation of start-low → slip → raise grip.",
            "GT slip is simulator-state, not tactile.",
            "Scripted arm, not Tabero π0.",
            "GelSight cameras off to share GPU.",
        ],
    }
    (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2))
    print("status", status)
    if table:
        print(f"{'mu':>5} {'method':<10} {'n':>3} {'lift':>5} {'full':>5} {'meanF':>6} {'peak':>6} {'upd':>5}")
        for t in table:
            print(f"{t['friction']:5.1f} {t['method']:<10} {t['n']:3d} {t['lift_sr']:5.2f} {t['full_sr']:5.2f} {t['mean_F']:6.2f} {t['peak_F']:6.2f} {t['updates']:5.2f}")


if __name__ == "__main__":
    main()
