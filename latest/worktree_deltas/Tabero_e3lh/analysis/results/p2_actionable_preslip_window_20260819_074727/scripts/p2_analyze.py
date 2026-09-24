#!/usr/bin/env python3
"""P2 analysis: rescue curve vs oracle command lead. Diagnostic only."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
PLOT = OUT / "plots"
PLOT.mkdir(exist_ok=True)
P1_WARN_MS = 300
SR_GATE = 0.8
DT = 0.05


def _f(x, d=np.nan):
    try:
        if x in ("", "None", None):
            return d
        return float(x)
    except Exception:
        return d


def load_results():
    p = OUT / "INTERVENTION_RESULTS.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(p.open()))


def sr(rows, key="full_task_success"):
    xs = [_f(r[key]) for r in rows]
    xs = [x for x in xs if np.isfinite(x)]
    if not xs:
        return np.nan, 0, 0
    return float(np.mean(xs)), int(sum(xs)), len(xs)


def cond_lead(r):
    c = r["cond"]
    if c == "fixed4":
        return "fixed4"
    if c == "fixed6":
        return "fixed6"
    if c == "lift0":
        return "lift0"
    try:
        return int(float(r["lead_ms"]))
    except Exception:
        return r.get("lead_ms")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_results()
    by = defaultdict(list)
    for r in rows:
        by[cond_lead(r)].append(r)

    base4 = by.get("fixed4", [])
    base6 = by.get("fixed6", [])
    # BASELINE_EVENT_TIMES
    if base4:
        fields = ["seed_idx", "t_lift0", "t_micro", "t_slip", "t_loss", "lift_success", "full_task_success"]
        with (OUT / "BASELINE_EVENT_TIMES.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sorted(base4, key=lambda x: int(float(x["seed_idx"]))):
                w.writerow({k: r.get(k, "") for k in fields})

    # INTERVENTION_SCHEDULE + FORCE_EFFECTIVE
    sched, feff = [], []
    for r in rows:
        if r["cond"] in ("fixed4", "fixed6"):
            continue
        sched.append({
            "trial_id": r["trial_id"], "seed_idx": r["seed_idx"], "cond": r["cond"],
            "lead_ms": r.get("lead_ms"), "t_slip_ref": r.get("t_slip_ref"),
            "t_cmd_sched": r.get("t_cmd_sched"), "t_cmd_actual": r.get("t_cmd_actual"),
            "oracle_timing": r.get("oracle_timing"),
        })
        feff.append({
            "trial_id": r["trial_id"], "seed_idx": r["seed_idx"], "cond": r["cond"],
            "lead_ms": r.get("lead_ms"),
            "t_cmd_actual": r.get("t_cmd_actual"),
            "t_force_rise": r.get("t_force_rise"),
            "t_reach_5N": r.get("t_reach_5N"),
            "t_force_effective": r.get("t_force_effective"),
            "t_reach_6band": r.get("t_reach_6band"),
            "cmd_lead_vs_ref_slip_ms": r.get("cmd_lead_vs_ref_slip_ms"),
            "effective_lead_vs_ref_slip_ms": r.get("effective_lead_vs_ref_slip_ms"),
            "peak_force": r.get("peak_force"),
            "slam": r.get("slam"),
            "reached6_before_ref_slip": r.get("reached6_before_ref_slip"),
            "case_b_force_ok_still_fail": r.get("case_b_force_ok_still_fail"),
        })
    if sched:
        with (OUT / "INTERVENTION_SCHEDULE.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sched[0].keys()))
            w.writeheader(); w.writerows(sched)
    if feff:
        with (OUT / "FORCE_EFFECTIVE_TIMES.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(feff[0].keys()))
            w.writeheader(); w.writerows(feff)

    numeric_leads = sorted([k for k in by if isinstance(k, int)], reverse=True)
    rescue_rows = []

    def add_row(label, lead_ms, steps, rs):
        full, nf, n = sr(rs, "full_task_success")
        lift, nl, n2 = sr(rs, "lift_success")
        eff_before = np.mean([_f(r["reached6_before_ref_slip"]) for r in rs]) if rs else np.nan
        peak = np.median([_f(r["peak_force"]) for r in rs]) if rs else np.nan
        slam = np.mean([_f(r["slam"]) for r in rs]) if rs else np.nan
        case_b = np.mean([_f(r["case_b_force_ok_still_fail"]) for r in rs]) if rs else np.nan
        cmd_to_eff = []
        cmd_to_rise = []
        cmd_to_6 = []
        for r in rs:
            tc, te, tr, t6 = _f(r.get("t_cmd_actual")), _f(r.get("t_force_effective")), _f(r.get("t_force_rise")), _f(r.get("t_reach_6band"))
            if np.isfinite(tc) and np.isfinite(tr):
                cmd_to_rise.append((tr - tc) * 1000)
            if np.isfinite(tc) and np.isfinite(te):
                cmd_to_eff.append((te - tc) * 1000)
            if np.isfinite(tc) and np.isfinite(t6):
                cmd_to_6.append((t6 - tc) * 1000)
        rescue_rows.append({
            "condition": label,
            "lead_ms": lead_ms,
            "control_steps": steps,
            "n": n,
            "lift_sr": lift, "lift_k": nl,
            "full_sr": full, "full_k": nf,
            "frac_6N_before_ref_slip": eff_before,
            "median_peak_N": peak,
            "slam_rate": slam,
            "case_b_rate": case_b,
            "median_cmd_to_rise_ms": float(np.median(cmd_to_rise)) if cmd_to_rise else np.nan,
            "median_cmd_to_5p5N_ms": float(np.median(cmd_to_eff)) if cmd_to_eff else np.nan,
            "median_cmd_to_6band_ms": float(np.median(cmd_to_6)) if cmd_to_6 else np.nan,
        })

    if base4:
        add_row("fixed4", None, None, base4)
    if base6:
        add_row("fixed6", None, None, base6)
    if by.get("lift0"):
        add_row("lift0", "lift0", None, by["lift0"])
    for lead in numeric_leads:
        add_row(f"lead_{lead}", lead, int(round(lead / 50.0)), by[lead])

    if rescue_rows:
        with (OUT / "RESCUE_CURVE.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rescue_rows[0].keys()))
            w.writeheader(); w.writerows(rescue_rows)

    full_by, lift_by = {}, {}
    for r in rescue_rows:
        if isinstance(r["lead_ms"], int):
            full_by[str(r["lead_ms"])] = r["full_sr"]
            lift_by[str(r["lead_ms"])] = r["lift_sr"]

    # Latest actionable = smallest lead_ms (closest to / after slip) with full SR>=0.8.
    nmin = 5 if (len(base4) >= 5) else 3
    ok_leads = []
    for lead in numeric_leads:
        full, _, n = sr(by[lead])
        if n >= nmin and np.isfinite(full) and full >= SR_GATE:
            ok_leads.append(lead)
    actionable = min(ok_leads) if ok_leads else None
    if actionable is None:
        for lead in sorted(numeric_leads, reverse=True):
            full, _, n = sr(by[lead])
            if n >= 3 and np.isfinite(full) and full >= SR_GATE:
                actionable = lead
                break

    # effective-force actionable: bin episodes by effective lead
    eff_bins = defaultdict(list)
    for r in rows:
        if r["cond"] in ("fixed4", "fixed6"):
            continue
        el = _f(r.get("effective_lead_vs_ref_slip_ms"))
        if not np.isfinite(el):
            continue
        # round to nearest 50 ms
        b = int(round(el / 50.0) * 50)
        eff_bins[b].append(r)

    latest_eff = None
    for b in sorted(eff_bins.keys()):  # most late first (negative)
        full, _, n = sr(eff_bins[b])
        if n >= 3 and full >= SR_GATE:
            latest_eff = b
            break

    # plots
    leads_plot = [k for k in numeric_leads]
    if leads_plot:
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        xs = leads_plot
        ys = [sr(by[k])[0] for k in xs]
        yl = [sr(by[k], "lift_success")[0] for k in xs]
        ax.plot(xs, ys, marker="o", label="full SR")
        ax.plot(xs, yl, marker="s", label="lift SR")
        ax.axhline(0.8, ls="--", c="gray", label="SR=0.8")
        ax.axvline(P1_WARN_MS, ls=":", c="C1", label="P1 marker ~300 ms")
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("oracle command lead before baseline T_slip (ms)")
        ax.set_ylabel("success rate")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Rescue rate vs command lead (ORACLE timing)")
        fig.tight_layout()
        fig.savefig(PLOT / "rescue_rate_vs_command_lead.png", dpi=140)
        fig.savefig(PLOT / "rescue_rate_vs_command_lead.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        bx = sorted(eff_bins.keys(), reverse=True)
        bys = [sr(eff_bins[b])[0] for b in bx]
        ax.plot(bx, bys, marker="o")
        ax.axhline(0.8, ls="--", c="gray")
        ax.axvline(P1_WARN_MS, ls=":", c="C1")
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("effective-force lead before baseline T_slip (ms)")
        ax.set_ylabel("full SR")
        ax.set_title("Rescue rate vs when measured force first ≥5.5 N")
        fig.tight_layout()
        fig.savefig(PLOT / "rescue_rate_vs_effective_force_lead.png", dpi=140)
        fig.savefig(PLOT / "rescue_rate_vs_effective_force_lead.pdf")
        plt.close(fig)

    # representative trajectories from FORCE_TRAJECTORIES if present
    traj_p = OUT / "FORCE_TRAJECTORIES.csv"
    if traj_p.exists():
        traj = list(csv.DictReader(traj_p.open()))
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        shown = set()
        for want in [("fixed4", "C3"), ("fixed6", "C2"), ("lead_300", "C0"), ("lead_0", "C1")]:
            pass
        # pick seed 0 if present
        for cond, color, lab in [
            ("fixed4", "C3", "fixed 4 N"),
            ("fixed6", "C2", "fixed 6 N"),
            ("intv", None, None),
        ]:
            pass
        def plot_tid(tid, color, lab):
            ts = [r for r in traj if r["trial_id"] == tid]
            if not ts:
                return
            t0 = None
            for r in ts:
                if r["phase"] == "lift":
                    t0 = _f(r["t_s"]); break
            if t0 is None:
                t0 = _f(ts[0]["t_s"])
            ax.plot([_f(r["t_s"]) - t0 for r in ts], [_f(r["f_meas"]) for r in ts], color=color, label=lab, lw=1.2)

        s0 = [r for r in rows if int(float(r["seed_idx"])) == 0]
        mapping = []
        for r in s0:
            if r["cond"] == "fixed4":
                mapping.append((r["trial_id"], "C3", "fixed 4 N"))
            elif r["cond"] == "fixed6":
                mapping.append((r["trial_id"], "C2", "fixed 6 N"))
            elif r["cond"] == "lift0":
                mapping.append((r["trial_id"], "C0", "4→6 at T0"))
            elif str(r.get("lead_ms")) == "300":
                mapping.append((r["trial_id"], "C4", "4→6 at 300 ms"))
            elif str(r.get("lead_ms")) == "0":
                mapping.append((r["trial_id"], "C1", "4→6 at T_slip"))
        for tid, c, lab in mapping:
            plot_tid(tid, c, lab)
        ax.axhline(6.0, ls="--", c="gray", lw=0.8)
        ax.set_xlim(-0.5, 2.5)
        ax.set_xlabel("time from lift onset (s)")
        ax.set_ylabel("measured squeeze (N)")
        ax.legend(fontsize=8)
        ax.set_title("Representative force traces (seed 0)")
        fig.tight_layout()
        fig.savefig(PLOT / "representative_force_trajectories.png", dpi=140)
        fig.savefig(PLOT / "representative_force_trajectories.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        def plot_rel(tid, color, lab):
            ts = [r for r in traj if r["trial_id"] == tid]
            if not ts:
                return
            t0 = None
            for r in ts:
                if r["phase"] == "lift":
                    t0 = _f(r["t_s"]); break
            if t0 is None:
                return
            ax.plot([_f(r["t_s"]) - t0 for r in ts], [_f(r["rel_z"]) for r in ts], color=color, label=lab, lw=1.2)
        for tid, c, lab in mapping:
            plot_rel(tid, c, lab)
        ax.axhline(-0.008, ls="--", c="gray", lw=0.8, label="GT slip 8 mm")
        ax.set_xlim(-0.5, 2.0)
        ax.set_xlabel("time from lift onset (s)")
        ax.set_ylabel("rel_z (m)  [privileged]")
        ax.legend(fontsize=8)
        ax.set_title("Representative object relative motion (seed 0)")
        fig.tight_layout()
        fig.savefig(PLOT / "representative_object_motion.png", dpi=140)
        fig.savefig(PLOT / "representative_object_motion.pdf")
        plt.close(fig)

    # alignment plot
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.axvline(P1_WARN_MS, color="C1", lw=2, label="P1 marker warning ~300 ms")
    if actionable is not None:
        ax.axvline(actionable, color="C2", lw=2, ls="--", label=f"P2 latest actionable cmd ~{actionable} ms")
    ax.set_xlim(1100, -150)
    ax.set_yticks([])
    ax.set_xlabel("ms before baseline T_slip  (left = earlier)")
    ax.legend(fontsize=8)
    ax.set_title("P1 information time vs P2 required action time")
    fig.tight_layout()
    fig.savefig(PLOT / "p1_signal_vs_p2_action_window.png", dpi=140)
    fig.savefig(PLOT / "p1_signal_vs_p2_action_window.pdf")
    plt.close(fig)

    f4, k4, n4 = sr(base4)
    f6, k6, n6 = sr(base6)
    slams = [_f(r["slam"]) for r in rows if r["cond"] not in ("fixed4",)]
    slam_any = bool(np.nanmean(slams) > 0.05) if slams else False
    peaks = [_f(r["peak_force"]) for r in rows]
    accidental_slam = bool(np.nanmax(peaks) >= 10.0) if peaks else False

    # timing dependence: early leads high, late leads low
    early = [sr(by[k])[0] for k in numeric_leads if isinstance(k, int) and k >= 300]
    late = [sr(by[k])[0] for k in numeric_leads if isinstance(k, int) and k <= 100]
    timing_dep = bool(early and late and np.nanmean(early) - np.nanmean(late) >= 0.3)

    p1_early_enough = None
    if actionable is not None:
        p1_early_enough = bool(P1_WARN_MS >= actionable)  # warning at least as early as required action
        # If actionable is 300, warning=300 is enough. If actionable is 500, warning too late.

    # classification
    status = "P2_BLOCKED_TECHNICALLY"
    if n4 >= 3 and f4 < 0.3 and n6 >= 3 and f6 >= 0.8:
        if actionable is not None and p1_early_enough and timing_dep and not accidental_slam:
            # need 300ms itself competitive
            full300 = sr(by.get(300, []))[0]
            if np.isfinite(full300) and full300 >= 0.8:
                status = "P2_ACTIONABLE_WINDOW_CONFIRMED"
            elif actionable <= 200:
                status = "P2_ACTIONABLE_WINDOW_CONFIRMED"
            else:
                status = "P2_WINDOW_EXISTS_BUT_P1_SIGNAL_TOO_LATE"
        elif actionable is not None and not p1_early_enough:
            if actionable >= 750:
                status = "P2_ONLY_VERY_EARLY_INTERVENTION_WORKS"
            else:
                status = "P2_WINDOW_EXISTS_BUT_P1_SIGNAL_TOO_LATE"
        elif actionable is None:
            # maybe force never reaches 6N in time
            lat = [r["median_cmd_to_5p5N_ms"] for r in rescue_rows if np.isfinite(_f(r.get("median_cmd_to_5p5N_ms")))]
            if lat and np.nanmedian(lat) > 250:
                status = "P2_FORCE_UPDATE_LATENCY_DOMINATES"
            else:
                status = "P2_NO_RECOVERABLE_PRESLIP_WINDOW"
        elif accidental_slam:
            status = "P2_FORCE_UPDATE_LATENCY_DOMINATES"
    elif n4 < 3 or n6 < 3:
        status = "P2_BLOCKED_TECHNICALLY"

    align = [
        "# P1 vs P2 window alignment",
        "",
        "P2 uses **ORACLE_INTERVENTION_TIMING** (baseline T_slip). Not GelSight-triggered.",
        "",
        f"- P1 deployable marker warning: **{P1_WARN_MS} ms** before T_slip",
        f"- P2 latest command lead with full SR≥{SR_GATE}: **{actionable} ms**",
        f"- P2 latest effective-force lead with SR≥{SR_GATE}: **{latest_eff} ms**",
        f"- P1 signal early enough to intervene: **{p1_early_enough}**",
        "",
        f"Fixed 4 N full SR: {k4}/{n4} = {f4}",
        f"Fixed 6 N full SR: {k6}/{n6} = {f6}",
        "",
        "If T_warning ≥ T_required_action_lead, task-native online adaptation is physically feasible.",
        "If T_warning < T_required_action_lead, the marker cue is visible but too late; prefer probe / pre-contact prior.",
        "",
        "## Rescue curve",
    ]
    for r in rescue_rows:
        align.append(
            f"- {r['condition']}: lift {r['lift_k']}/{r['n']} full {r['full_k']}/{r['n']} "
            f"6N-before-slip={r['frac_6N_before_ref_slip']} peak={r['median_peak_N']}"
        )
    (OUT / "P1_P2_WINDOW_ALIGNMENT.md").write_text("\n".join(align) + "\n")

    latency_rows = [r for r in rescue_rows if np.isfinite(_f(r.get("median_cmd_to_5p5N_ms")))]
    med_rise = float(np.nanmedian([_f(r["median_cmd_to_rise_ms"]) for r in latency_rows])) if latency_rows else None
    med_eff = float(np.nanmedian([_f(r["median_cmd_to_5p5N_ms"]) for r in latency_rows])) if latency_rows else None
    med_6 = float(np.nanmedian([_f(r["median_cmd_to_6band_ms"]) for r in latency_rows])) if latency_rows else None

    verdict = {
        "status": status,
        "method_change": "NONE",
        "task": "libero_object_1",
        "friction": 0.2,
        "initial_force_N": 4,
        "intervention_force_N": 6,
        "oracle_intervention_timing": True,
        "gelsight_triggered": False,
        "baseline_fixed4_full_sr": f4,
        "baseline_fixed4": f"{k4}/{n4}",
        "fixed6_full_sr": f6,
        "fixed6": f"{k6}/{n6}",
        "lead_times_ms": numeric_leads,
        "full_sr_by_lead_time": full_by,
        "lift_sr_by_lead_time": lift_by,
        "force_tracking_after_update_valid": bool(med_eff is not None and med_eff < 2000),
        "median_cmd_to_rise_ms": med_rise,
        "median_cmd_to_5p5N_ms": med_eff,
        "median_cmd_to_6band_ms": med_6,
        "latest_actionable_command_lead_ms": actionable,
        "latest_actionable_effective_force_lead_ms": latest_eff,
        "p1_marker_warning_lead_ms": P1_WARN_MS,
        "p1_signal_early_enough_to_intervene": p1_early_enough,
        "clear_timing_dependence": timing_dep,
        "accidental_force_slam_present": accidental_slam,
        "primary_evidence": [
            f"fixed4 full={k4}/{n4}",
            f"fixed6 full={k6}/{n6}",
            f"latest actionable command lead={actionable} ms",
            f"P1 warning 300 ms early enough={p1_early_enough}",
        ],
        "limitations": [
            "Intervention time is oracle-aligned to baseline T_slip, not a deployable detector.",
            "Scripted pick-place, not Tabero π0.",
            "Honest full success = this-episode lift AND basket_contact>0.05 N.",
            "GT slip is the P1 8 mm rule; not moved post-hoc.",
        ],
    }
    (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2, default=float) + "\n")
    print("status", status)
    print("fixed4", k4, n4, f4, "fixed6", k6, n6, f6)
    print("actionable_cmd", actionable, "eff", latest_eff, "p1_enough", p1_early_enough)
    for r in rescue_rows:
        print(r["condition"], "full", r["full_k"], "/", r["n"], "lift", r["lift_k"], "/", r["n"])


if __name__ == "__main__":
    main()
