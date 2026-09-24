#!/usr/bin/env python3
"""Finalize B4 FORTE-inspired reactive baseline artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(os.environ.get("B4_OUT", Path(__file__).resolve().parents[1]))
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)
TASKS = [0, 1, 2, 5, 6]
TASK_OBJECTS = {
    0: "alphabet soup",
    1: "cream cheese",
    2: "salad dressing",
    5: "tomato sauce",
    6: "butter",
    7: "milk",
}
FSTAR = {
    0: {0.2: 5.0, 0.5: 4.0, 1.0: 3.0},
    1: {0.2: 6.0, 0.5: 5.0, 1.0: 3.0},
    2: {0.2: 8.0, 0.5: 3.0, 1.0: 3.0},
    5: {0.2: 5.0, 0.5: 4.0, 1.0: 3.0},
    6: {0.2: 4.0, 0.5: 3.0, 1.0: 3.0},
}
FIXED_ROBUST = {0: 5.0, 1: 6.0, 2: 8.0, 5: 5.0, 6: 4.0}
METHOD_LABEL = {
    "fixed_low": "Fixed Low",
    "fixed_robust": "Fixed Robust",
    "gt_minforce": "GT-MinForce",
    "forte_gt_reactive": "FORTE-GT",
    "early_gt_reference": "Early-GT Ref",
}
B2R2 = Path("/home/exouser/Tabero/analysis/results/b2r2_tabero_task_breadth_20260821_073931")
B2R2_FILES = {
    0: "TASK0_SCAN.csv",
    1: "TASK1_FROZEN_REFERENCE_SCAN.csv",
    2: "TASK2_SCAN.csv",
    5: "TASK5_SCAN.csv",
    6: "TASK6_SCAN.csv",
}


def read_csvs(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(OUT.glob(pattern)):
        if path.exists() and path.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_df(path: Path, df: pd.DataFrame):
    if df.empty:
        path.write_text("status\nNOT_RUN\n")
    else:
        df.to_csv(path, index=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd="/home/exouser/Tabero", text=True).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def _num(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def prep(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "task_id",
        "seed_idx",
        "friction",
        "initial_force_N",
        "final_force_target_N",
        "fstar_N",
        "fixed_robust_N",
        "pick_success",
        "lift_success",
        "transport_success",
        "place_success",
        "full_success",
        "mean_force_N",
        "peak_force_N",
        "integrated_force_Ns",
        "number_of_updates",
        "gt_slip_onset_s",
        "first_update_s",
        "recovery_latency_s",
        "time_to_sufficient_force_s",
        "obj_displacement_before_first_update_m",
        "obj_displacement_before_sufficient_force_m",
    ]:
        _num(df, col)
    return df


def normalize_reference_rows() -> pd.DataFrame:
    rows = []
    for task, fname in B2R2_FILES.items():
        path = B2R2 / fname
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        for _, r in raw.iterrows():
            mu = round(float(r["friction"]), 1)
            force = float(r["force"] if "force" in raw.columns and not pd.isna(r.get("force")) else r["chosen_force"])
            method = None
            if abs(force - 3.0) < 1e-6:
                method = "fixed_low"
            if abs(force - FIXED_ROBUST[task]) < 1e-6:
                method = "fixed_robust" if method is None else method
            if abs(force - FSTAR[task][mu]) < 1e-6:
                # Keep GT-MinForce as a separate copy of the same episode row.
                rows.append(_ref_row(task, r, mu, force, "gt_minforce", "canonical_gt_minforce"))
            if method is not None:
                rows.append(_ref_row(task, r, mu, force, method, method))
    if not rows:
        return pd.DataFrame()
    return prep(pd.DataFrame(rows))


def _ref_row(task: int, r: pd.Series, mu: float, force: float, method: str, rule: str) -> dict:
    full = r.get("full_success", r.get("full_task_success", 0))
    mean_force = r.get("mean_force_N", r.get("mean_force", r.get("mean_grip_force", np.nan)))
    peak_force = r.get("peak_force_N", r.get("peak_force", np.nan))
    integrated = r.get("integrated_force_Ns", r.get("integrated_force", np.nan))
    return {
        "trial_id": f"b2r2ref_t{task}_{method}_{r.get('trial_id', '')}",
        "task_id": task,
        "object": r.get("object", TASK_OBJECTS[task]),
        "instruction": r.get("instruction", ""),
        "method": method,
        "baseline_name": method,
        "official_forte_reproduction": 0,
        "seed_idx": r.get("seed_idx", np.nan),
        "friction": mu,
        "friction_static_applied": r.get("friction_static_applied", r.get("friction_applied", np.nan)),
        "friction_dynamic_applied": r.get("friction_dynamic_applied", r.get("friction_applied", np.nan)),
        "initial_force_N": force,
        "final_force_target_N": force,
        "fstar_N": FSTAR[task][mu],
        "fixed_robust_N": FIXED_ROBUST[task],
        "decision_rule": rule,
        "force_ladder_N": "",
        "force_target_trajectory": f"hold_{force:g}",
        "update_times_s": "",
        "reached_force_times_s": "",
        "pick_success": r.get("pick_success", np.nan),
        "lift_success": r.get("lift_success", np.nan),
        "transport_success": r.get("transport_success", r.get("transport_retention", np.nan)),
        "place_success": r.get("place_success", np.nan),
        "full_success": full,
        "official_success": r.get("official_success", np.nan),
        "dropped": r.get("dropped", np.nan),
        "timeout": r.get("timeout", np.nan),
        "lost_in_transit": r.get("lost_in_transit", np.nan),
        "basket_contact_max": r.get("basket_contact_max", np.nan),
        "mean_force_N": mean_force,
        "peak_force_N": peak_force,
        "integrated_force_Ns": integrated,
        "integrated_target_Ns": np.nan,
        "number_of_updates": 0,
        "gt_slip_onset_s": np.nan,
        "early_gt_onset_s": np.nan,
        "first_update_s": np.nan,
        "first_event_latency_s": np.nan,
        "early_event_latency_s": np.nan,
        "slip_to_first_update_s": np.nan,
        "recovery_latency_s": np.nan,
        "time_to_sufficient_force_s": np.nan,
        "obj_displacement_before_first_update_m": np.nan,
        "obj_displacement_before_sufficient_force_m": np.nan,
        "gt_slip_used": 0,
        "early_gt_used": 0,
        "slam_20N": 0,
        "steps": r.get("steps", np.nan),
        "t_episode_s": r.get("t_episode_s", np.nan),
        "reference_source": str(B2R2),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["task_id", "object", "friction", "method"], dropna=False)
    rows = []
    for key, sub in g:
        task_id, obj, mu, method = key
        rows.append(
            {
                "task_id": int(task_id),
                "object": obj,
                "friction": float(mu),
                "method": method,
                "method_label": METHOD_LABEL.get(method, method),
                "n": int(len(sub)),
                "full_sr": float(sub["full_success"].mean()),
                "pick_sr": float(sub["pick_success"].mean()),
                "lift_sr": float(sub["lift_success"].mean()),
                "place_sr": float(sub["place_success"].mean()),
                "mean_force_N": float(sub["mean_force_N"].mean()),
                "peak_force_N": float(sub["peak_force_N"].mean()),
                "integrated_force_Ns": float(sub["integrated_force_Ns"].mean()),
                "mean_updates": float(sub["number_of_updates"].mean()) if "number_of_updates" in sub else 0.0,
                "slip_event_rate": float(sub["gt_slip_onset_s"].notna().mean()) if "gt_slip_onset_s" in sub else 0.0,
                "mean_first_event_latency_s": float(sub["first_event_latency_s"].mean()) if "first_event_latency_s" in sub else np.nan,
                "mean_recovery_latency_s": float(sub["recovery_latency_s"].mean()) if "recovery_latency_s" in sub else np.nan,
                "mean_final_target_N": float(sub["final_force_target_N"].mean()),
                "mean_fstar_N": float(sub["fstar_N"].mean()),
                "mean_fixed_robust_N": float(sub["fixed_robust_N"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["task_id", "friction", "method"])


def task_classifications(main: pd.DataFrame, summary: pd.DataFrame, dynamic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        s = summary[summary.task_id == task]
        forte = s[s.method == "forte_gt_reactive"]
        robust = s[s.method == "fixed_robust"]
        oracle = s[s.method == "gt_minforce"]
        if forte.empty or robust.empty or oracle.empty:
            classification = "REACTIVE_FORCE_SERVO_BLOCKED"
            reason = "missing B4 main rows"
        else:
            forte_sr = float(forte.full_sr.mean())
            robust_sr = float(robust.full_sr.mean())
            forte_force = float(forte.mean_force_N.mean())
            robust_force = float(robust.mean_force_N.mean())
            low_forte = float(forte[forte.friction == 0.2].full_sr.mean())
            low_robust = float(robust[robust.friction == 0.2].full_sr.mean())
            dyn = dynamic[dynamic.task_id == task] if not dynamic.empty and "task_id" in dynamic else pd.DataFrame()
            dynamic_failed = bool((not dyn.empty) and ("servo_ok" in dyn) and (pd.to_numeric(dyn["servo_ok"], errors="coerce").fillna(0).min() == 0))
            formal_task = main[(main.task_id == task) & (main.method == "forte_gt_reactive")]
            formal_force_ok = bool(
                (not formal_task.empty)
                and (
                    pd.to_numeric(formal_task["peak_force_N"], errors="coerce").max() >= max(3.5, FIXED_ROBUST[task] - 0.5)
                    or (
                        pd.to_numeric(formal_task["final_force_target_N"], errors="coerce").max() > 3.0
                        and pd.to_numeric(formal_task["mean_force_N"], errors="coerce").max() > 3.5
                    )
                )
            )
            servo_blocked = dynamic_failed and not formal_force_ok
            if servo_blocked:
                classification = "REACTIVE_FORCE_SERVO_BLOCKED"
                reason = "dynamic force-servo audit did not pass for every ladder transition"
            elif forte_sr >= robust_sr - 0.05 and forte_force < robust_force - 0.25 and low_forte >= low_robust - 0.10:
                classification = "REACTIVE_STRONG"
                reason = "SR near Fixed Robust while mean force is materially lower"
            elif low_forte < low_robust - 0.20:
                classification = "REACTIVE_TOO_LATE"
                reason = "low-friction rescue falls well below Fixed Robust despite GT slip"
            elif forte_sr >= robust_sr - 0.20:
                classification = "REACTIVE_PARTIAL"
                reason = "aggregate SR is close enough to matter but not a strong matched-success baseline"
            else:
                classification = "REACTIVE_TOO_LATE"
                reason = "aggregate SR is far below Fixed Robust"
        f_task = forte if not forte.empty else pd.DataFrame()
        r_task = robust if not robust.empty else pd.DataFrame()
        o_task = oracle if not oracle.empty else pd.DataFrame()
        rows.append(
            {
                "task_id": task,
                "object": TASK_OBJECTS[task],
                "classification": classification,
                "reason": reason,
                "forte_mean_full_sr": float(f_task.full_sr.mean()) if not f_task.empty else np.nan,
                "fixed_robust_mean_sr": float(r_task.full_sr.mean()) if not r_task.empty else np.nan,
                "forte_low_mu_sr": float(f_task[f_task.friction == 0.2].full_sr.mean()) if not f_task.empty else np.nan,
                "forte_mid_mu_sr": float(f_task[f_task.friction == 0.5].full_sr.mean()) if not f_task.empty else np.nan,
                "forte_high_mu_sr": float(f_task[f_task.friction == 1.0].full_sr.mean()) if not f_task.empty else np.nan,
                "forte_mean_force_N": float(f_task.mean_force_N.mean()) if not f_task.empty else np.nan,
                "fixed_robust_mean_force_N": float(r_task.mean_force_N.mean()) if not r_task.empty else np.nan,
                "gt_minforce_mean_force_N": float(o_task.mean_force_N.mean()) if not o_task.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, value: str, title: str, ylabel: str, path_base: Path):
    pivot = df.pivot_table(index="task_id", columns="method_label", values=value, aggfunc="mean")
    order = [m for m in ["Fixed Low", "FORTE-GT", "Fixed Robust", "GT-MinForce"] if m in pivot.columns]
    pivot = pivot[order]
    ax = pivot.plot(kind="bar", figsize=(9.5, 4.8), color=["#9ca3af", "#2563eb", "#d97706", "#059669"][: len(order)])
    ax.set_title(title)
    ax.set_xlabel("Task")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".png"), dpi=180)
    plt.savefig(path_base.with_suffix(".svg"))
    plt.close()


def make_plots(main: pd.DataFrame, summary: pd.DataFrame, traj: pd.DataFrame, classif: pd.DataFrame):
    task_method = summary.groupby(["task_id", "method_label"], as_index=False).agg(
        full_sr=("full_sr", "mean"),
        mean_force_N=("mean_force_N", "mean"),
        mean_updates=("mean_updates", "mean"),
    )
    plot_bar(task_method, "full_sr", "Full Success Rate By Task And Method", "Full SR", PLOTS / "sr_by_task_method")
    plot_bar(task_method, "mean_force_N", "Mean Grip Force By Task And Method", "Mean force (N)", PLOTS / "mean_force_by_task_method")
    plot_bar(task_method[task_method.method_label == "FORTE-GT"], "mean_updates", "FORTE-GT Update Count By Task", "Mean updates", PLOTS / "update_count_by_task")

    comp = classif[["task_id", "forte_mean_force_N", "fixed_robust_mean_force_N", "gt_minforce_mean_force_N"]].copy()
    comp = comp.rename(columns={
        "forte_mean_force_N": "FORTE-GT",
        "fixed_robust_mean_force_N": "Fixed Robust",
        "gt_minforce_mean_force_N": "GT-MinForce",
    })
    ax = comp.set_index("task_id").plot(kind="bar", figsize=(9.5, 4.8), color=["#2563eb", "#d97706", "#059669"])
    ax.set_title("Reactive vs Robust vs Oracle Mean Force")
    ax.set_xlabel("Task")
    ax.set_ylabel("Mean force (N)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTS / "reactive_vs_robust_vs_oracle.png", dpi=180)
    plt.savefig(PLOTS / "reactive_vs_robust_vs_oracle.svg")
    plt.close()

    if not traj.empty:
        tr = traj[(traj["method"] == "forte_gt_reactive") & (pd.to_numeric(traj["friction"], errors="coerce") == 0.2)].copy()
        if not tr.empty:
            tr["seed_idx"] = pd.to_numeric(tr["seed_idx"], errors="coerce")
            tr = tr[tr["seed_idx"].isin(sorted(tr["seed_idx"].dropna().unique())[:3])]
            fig, axes = plt.subplots(len(TASKS), 1, figsize=(10, 10), sharex=True)
            for ax, task in zip(axes, TASKS):
                sub = tr[tr.task_id == task]
                for seed, ss in sub.groupby("seed_idx"):
                    ss = ss.sort_values("t_s")
                    ax.plot(ss["t_s"], ss["force_target"], alpha=0.45, linewidth=1.1, label=f"s{int(seed)} target")
                    ax.plot(ss["t_s"], ss["measured_force"], alpha=0.35, linewidth=0.9, linestyle="--")
                ax.set_title(f"task{task} {TASK_OBJECTS[task]} low-mu trajectories")
                ax.set_ylabel("N")
                ax.grid(alpha=0.2)
            axes[-1].set_xlabel("time (s)")
            plt.tight_layout()
            plt.savefig(PLOTS / "force_trajectories_low_mu.png", dpi=180)
            plt.savefig(PLOTS / "force_trajectories_low_mu.svg")
            plt.close()

    timing = summary[summary.method == "forte_gt_reactive"].copy()
    if not timing.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        x = np.arange(len(TASKS))
        vals = []
        for task in TASKS:
            task_rows = main[(main.task_id == task) & (main.method == "forte_gt_reactive")]
            vals.append(pd.to_numeric(task_rows["recovery_latency_s"], errors="coerce").mean())
        ax.bar([str(t) for t in TASKS], vals, color="#2563eb")
        ax.set_title("Slip To Recovery Timing")
        ax.set_xlabel("Task")
        ax.set_ylabel("Mean recovery latency after first update (s)")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(PLOTS / "slip_to_recovery_timing.png", dpi=180)
        plt.savefig(PLOTS / "slip_to_recovery_timing.svg")
        plt.close()


def write_specs():
    (OUT / "FORTE_INSPIRED_SPEC.md").write_text(
        """# FORTE-Inspired Reactive Baseline Spec

Baseline name: `FORTE-INSPIRED REACTIVE`.

This is not an official FORTE reproduction. Official FORTE uses its own fin-ray gripper and a position decrement controller. B4 tests only the information/control principle in Tabero:

```text
start low
-> detect slip/instability
-> increase calibrated force target
```

Main method:

```text
FORTE_GT_REACTIVE
initial force = 3 N
force ladder = 3 -> 4 -> 5 -> 6 -> 8 N
slip source = privileged GT slip/instability
```

All fixed and reactive methods use the same scripted/oracle downstream arm trajectory, neutral task instruction, friction setting, initial seed, and Tabero force servo. The force decision/control rule is the main variable.

The optional `EARLY_GT_REFERENCE` is not FORTE. It is an oracle timing reference using privileged micro-instability.
"""
    )
    (OUT / "GT_SLIP_DEFINITION.md").write_text(
        """# GT Slip Definition

Main B4 detector: frozen P1/F1R2/R1-v1 gross GT slip rule.

```text
GT_SLIP = true during hold/lift/transit/over-basket if any condition holds:

relative z loss < -8 mm
relative downward velocity < -0.05 m/s for 2 consecutive control steps
relative xy motion > 15 mm
finger contact deterioration/loss after contact had existed
drop flag
```

The rule does not read hidden friction or canonical F*. It is privileged simulator state and is therefore an upper-bound slip detector for the FORTE-inspired reactive baseline.

Optional early reference:

```text
EARLY_GT = relative z loss < -2 mm or relative downward velocity < -0.015 m/s
```

`EARLY_GT_REFERENCE` is explicitly not FORTE and is not the primary baseline.
"""
    )


def main():
    b4_main = prep(read_csvs("TASK*_MAIN_EPISODES.csv"))
    refs = normalize_reference_rows()
    main = prep(pd.concat([b4_main, refs], ignore_index=True, sort=False)) if not refs.empty else b4_main
    if b4_main.empty:
        raise SystemExit("No TASK*_MAIN_EPISODES.csv files found.")
    traj = prep(read_csvs("TASK*_MAIN_STEP_TRAJECTORIES.csv"))
    early = prep(read_csvs("TASK*_EARLY_GT_EPISODES.csv"))
    negative = prep(read_csvs("TASK*_NEGATIVE_EPISODES.csv"))
    dynamic = read_csvs("DYNAMIC_FORCE_SERVO_AUDIT_TASK*.csv")
    if not dynamic.empty:
        for col in ["task_id", "servo_ok", "settling_time_s", "peak_measured_after_N", "episode_peak_force_N"]:
            _num(dynamic, col)

    write_df(OUT / "DYNAMIC_FORCE_SERVO_AUDIT.csv", dynamic)
    for task in TASKS:
        tdf = main[main.task_id == task].copy()
        write_df(OUT / f"TASK{task}_FORTE.csv", tdf)

    refs = {
        "FIXED_LOW_REFERENCE.csv": main[main.method == "fixed_low"],
        "FIXED_ROBUST_REFERENCE.csv": main[main.method == "fixed_robust"],
        "GT_MINFORCE_REFERENCE.csv": main[main.method == "gt_minforce"],
    }
    for name, df in refs.items():
        write_df(OUT / name, df)
    write_df(OUT / "OPTIONAL_EARLY_GT_REFERENCE.csv", early)
    write_df(OUT / "OPTIONAL_NEGATIVE_CONTROL.csv", negative)

    summary = summarize(main)
    write_df(OUT / "FORTE_MAIN_RESULTS.csv", summary)
    classif = task_classifications(main, summary, dynamic)
    write_df(OUT / "TASK_LEVEL_CLASSIFICATION.csv", classif)
    make_plots(main, summary, traj, classif)
    write_specs()

    forte = main[main.method == "forte_gt_reactive"]
    robust = main[main.method == "fixed_robust"]
    oracle = main[main.method == "gt_minforce"]
    mean_full_sr = float(forte.full_success.mean())
    fixed_robust_mean_sr = float(robust.full_success.mean())
    mean_force = float(forte.mean_force_N.mean())
    fixed_robust_mean_force = float(robust.mean_force_N.mean())
    gt_minforce_mean_force = float(oracle.mean_force_N.mean())
    sr_gap_ratio = None if fixed_robust_mean_sr == 0 else mean_full_sr / fixed_robust_mean_sr
    force_gap_ratio = None if gt_minforce_mean_force == 0 else mean_force / gt_minforce_mean_force
    strong = classif[classif.classification == "REACTIVE_STRONG"].task_id.astype(int).tolist()
    partial = classif[classif.classification == "REACTIVE_PARTIAL"].task_id.astype(int).tolist()
    too_late = classif[classif.classification == "REACTIVE_TOO_LATE"].task_id.astype(int).tolist()
    servo_blocked = classif[classif.classification == "REACTIVE_FORCE_SERVO_BLOCKED"].task_id.astype(int).tolist()
    if servo_blocked and len(servo_blocked) == len(TASKS):
        status = "B4_DYNAMIC_FORCE_SERVO_BLOCKED"
    elif len(strong) == len(TASKS):
        status = "B4_FORTE_REACTIVE_STRONG_BASELINE"
    elif len(too_late) >= 2:
        status = "B4_POST_SLIP_REACTION_TOO_LATE_MULTI_TASK"
    elif strong or partial or too_late:
        status = "B4_TASK_DEPENDENT_REACTIVE_BASELINE"
    else:
        status = "B4_FORTE_REACTIVE_PARTIAL"

    task_results = {}
    for task in TASKS:
        task_rows = summary[summary.task_id == task]
        task_results[str(task)] = {
            "object": TASK_OBJECTS[task],
            "classification": classif[classif.task_id == task].classification.iloc[0],
            "frictions": {
                str(mu): {
                    m: {
                        "n": int(row.n),
                        "full_sr": float(row.full_sr),
                        "mean_force_N": float(row.mean_force_N),
                        "mean_updates": float(row.mean_updates),
                    }
                    for m, row in task_rows[task_rows.friction == mu].set_index("method").iterrows()
                }
                for mu in sorted(task_rows.friction.dropna().unique())
            },
        }

    verdict = {
        "status": status,
        "method_change": "NONE",
        "benchmark_tasks": TASKS,
        "baseline_name": "FORTE_INSPIRED_GT_REACTIVE",
        "official_forte_reproduction": False,
        "initial_force_N": 3,
        "force_ladder_N": [3, 4, 5, 6, 8],
        "gt_slip_used": True,
        "task_results": task_results,
        "mean_full_sr": mean_full_sr,
        "mean_force_N": mean_force,
        "fixed_robust_mean_sr": fixed_robust_mean_sr,
        "fixed_robust_mean_force_N": fixed_robust_mean_force,
        "gt_minforce_mean_force_N": gt_minforce_mean_force,
        "sr_gap_to_fixed_robust": sr_gap_ratio,
        "sr_delta_to_fixed_robust": mean_full_sr - fixed_robust_mean_sr,
        "force_gap_to_gt_minforce": force_gap_ratio,
        "tasks_reactive_strong": strong,
        "tasks_reactive_partial": partial,
        "tasks_reactive_too_late": too_late,
        "tasks_force_servo_blocked": servo_blocked,
        "task2_low_mu_sr": float(summary[(summary.task_id == 2) & (summary.friction == 0.2) & (summary.method == "forte_gt_reactive")].full_sr.mean()),
        "post_slip_reaction_too_late_multitask": len(too_late) >= 2,
        "primary_evidence": [
            "B4 main matched scripted downstream FORTE-GT episodes, N=20 per task/friction cell",
            "FORTE-GT uses privileged gross slip and force ladder 3->4->5->6->8",
            "Fixed Low, Fixed Robust, and GT-MinForce references combine B4 fixed-force supplement rows with frozen canonical B2-R2 rows when the protocol matched",
        ],
        "limitations": [
            "FORTE-inspired baseline is not official FORTE hardware reproduction",
            "GT slip is privileged and therefore an upper-bound detector",
            "Reference rows may exceed N=20 in cells where B4 supplement rows and frozen canonical B2-R2 rows both exist",
            "Task2 measured-force telemetry contains near-zero rows in frozen references and B4 mid/high conditions; use task2 target/update telemetry and SR for force-decision interpretation",
            "SWEEP_ERROR_task0_dynamic.json is retained from the initial sandbox no-CUDA attempt; final dynamic audit artifacts were produced after GPU execution",
            "No learned or deployable tactile slip detector was trained",
            "No OURS/probe/belief method was implemented in B4",
        ],
    }
    (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2))

    env = {
        "tabero_git_head": git(["git", "rev-parse", "HEAD"]),
        "tabero_git_status_short": git(["git", "status", "--short"]),
        "result_dir": str(OUT),
        "python": os.sys.executable,
        "script_sha256": {
            "b4_eval.py": sha256(OUT / "scripts" / "b4_eval.py"),
            "b4_finalize.py": sha256(OUT / "scripts" / "b4_finalize.py"),
        },
        "benchmark_tasks": TASKS,
        "candidate_forces_N": [3, 4, 5, 6, 8],
        "gt_slip_definition": "P1/F1R2/R1-v1 gross slip: rel_z<-8mm or v_rel_z<-0.05m/s for 2 steps or rel_xy>15mm or contact loss/drop",
    }
    (OUT / "ENV_PROVENANCE.json").write_text(json.dumps(env, indent=2))

    write_markdown(summary, classif, verdict)


def fmt(x, nd=3):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return f"{float(x):.{nd}f}"


def write_markdown(summary: pd.DataFrame, classif: pd.DataFrame, verdict: dict):
    lines = [
        "# B4 FORTE-Inspired Reactive Baseline",
        "",
        "Status: `{}`.".format(verdict["status"]),
        "",
        "`METHOD_CHANGE = NONE`. This is not an official FORTE reproduction. B4 tests the FORTE-style principle `start low -> detect GT slip/instability -> increase calibrated Tabero force target`.",
        "",
        "## Technical Summary",
        "",
        f"- Aggregate FORTE-GT full SR: {fmt(verdict['mean_full_sr'])}; Fixed Robust full SR: {fmt(verdict['fixed_robust_mean_sr'])}; SR ratio: {fmt(verdict['sr_gap_to_fixed_robust'])}.",
        f"- Aggregate FORTE-GT mean force: {fmt(verdict['mean_force_N'])} N; Fixed Robust mean force: {fmt(verdict['fixed_robust_mean_force_N'])} N; GT-MinForce mean force: {fmt(verdict['gt_minforce_mean_force_N'])} N.",
        f"- Reactive strong tasks: {verdict['tasks_reactive_strong']}; partial tasks: {verdict['tasks_reactive_partial']}; too-late tasks: {verdict['tasks_reactive_too_late']}.",
        f"- FORTE-GT was newly run in B4 at N=20 per task/friction cell; Fixed Low, Fixed Robust, and GT-MinForce rows combine B4 fixed-force supplement rows with frozen canonical B2-R2 references where protocol matched.",
        f"- QA note: task2 measured-force telemetry has near-zero rows in frozen references and B4 mid/high conditions; task2 force behavior should be read from target/update telemetry and SR.",
        f"- QA note: `SWEEP_ERROR_task0_dynamic.json` is the initial sandbox no-CUDA attempt; final dynamic audit files were produced by the later GPU run.",
        "",
        "## Main Results",
        "",
        "| task | object | low-mu SR | mid-mu SR | high-mu SR | FORTE mean F | robust SR | robust mean F | GT-MinForce mean F | classification |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in classif.sort_values("task_id").iterrows():
        task = int(row.task_id)
        forte = summary[(summary.task_id == task) & (summary.method == "forte_gt_reactive")]
        robust = summary[(summary.task_id == task) & (summary.method == "fixed_robust")]
        low = forte[forte.friction == 0.2].full_sr.mean()
        mid = forte[forte.friction == 0.5].full_sr.mean()
        high = forte[forte.friction == 1.0].full_sr.mean()
        lines.append(
            f"| {task} | {row.object} | {fmt(low,2)} | {fmt(mid,2)} | {fmt(high,2)} | "
            f"{fmt(row.forte_mean_force_N,2)} | {fmt(robust.full_sr.mean(),2)} | {fmt(row.fixed_robust_mean_force_N,2)} | "
            f"{fmt(row.gt_minforce_mean_force_N,2)} | `{row.classification}` |"
        )
    lines += [
        "",
        "## Low-Friction Rescue",
        "",
        "| task | start F | needed F | mean updates | rescue SR | mean time to sufficient F |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        low = summary[(summary.task_id == task) & (summary.method == "forte_gt_reactive") & (summary.friction == 0.2)]
        if low.empty:
            continue
        task_main = pd.read_csv(OUT / f"TASK{task}_FORTE.csv")
        low_ep = task_main[(task_main.method == "forte_gt_reactive") & (pd.to_numeric(task_main.friction) == 0.2)]
        lines.append(
            f"| {task} | 3 | {FSTAR[task][0.2]:g} | {fmt(low.mean_updates.iloc[0],2)} | "
            f"{fmt(low.full_sr.iloc[0],2)} | {fmt(pd.to_numeric(low_ep['time_to_sufficient_force_s'], errors='coerce').mean(),2)} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The B4 classification is task-level, not just aggregate. A task is `REACTIVE_STRONG` only when FORTE-GT stays near Fixed Robust success while materially lowering force. A task is `REACTIVE_TOO_LATE` when low-friction rescue remains far below Fixed Robust even with privileged GT slip.",
        "",
        "Task2 salad dressing is the stress test because the low-friction condition needs 8 N while mid/high friction need only 3 N. The reactive ladder has to climb 3->4->5->6->8 after instability; the task2 low-mu row in `FORTE_MAIN_RESULTS.csv` records whether that happens early enough.",
        "",
        "## Files",
        "",
        "- `FORTE_MAIN_RESULTS.csv` / `FORTE_MAIN_RESULTS.md`",
        "- `TASK_LEVEL_CLASSIFICATION.csv`",
        "- `FINAL_VERDICT.json`",
        "- `TASK{0,1,2,5,6}_FORTE.csv`",
        "- `DYNAMIC_FORCE_SERVO_AUDIT.csv`",
        "- `plots/`",
        "",
        "## Next Step",
        "",
        "Freeze B4, run B5 Tabero Neutral, then freeze the baseline landscape before implementing final OURS.",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "FORTE_MAIN_RESULTS.md").write_text(text)
    (OUT / "README.md").write_text(text)


if __name__ == "__main__":
    main()
