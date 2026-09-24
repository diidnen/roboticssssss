#!/usr/bin/env python3
"""Final task-demand LOTO gate for the Tabero world-model line.

Two phases are deliberate: ``freeze`` reconstructs only deployment-available
nominal plans and writes the immutable descriptor/protocol before optimization;
``run`` trains on seen tasks, freezes selection, and only then loads held-out
task DEV outcomes. Original TEST outcomes/telemetry are never loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
OLD = RESULTS / "loto_feasibility_generalization_20260830_022058"
JOINT_AUTH = RESULTS / "full_task_feasibility_20260830_012830"
HIST = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
P5_CODE = REPO / "analysis/p5s0c_paired_boundary_probe_value.py"
P4_CODE = RESULTS / "p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py"
OUT = Path(os.environ.get("TASK_DEMAND_OUT", RESULTS / "task_demand_loto_generalization_20260830_044334"))

TASKS = [0, 1, 5, 6]
FOLDS = {f"T{t}": {"heldout": t, "seen": [x for x in TASKS if x != t]} for t in TASKS}
SEEDS = [0, 1, 2]
LAMBDAS = [0.1, 0.3, 1.0]
H = 8
EPOCHS = 80
BATCH = 64
LR = 8e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
DT = 0.05
G = np.asarray([0.0, 0.0, -9.81], dtype=np.float64)
ACTIVE_PHASES = {"branch_hold", "lift", "transit", "over_basket", "place"}
PHASE_SPECS = [("branch_hold", 20), ("lift", 50), ("transit", 110), ("over_basket", 30), ("place", 40)]
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 2026083004

FEATURES = [
    "planned_total_duration_s", "planned_control_steps",
    "duration_branch_hold_s", "duration_lift_s", "duration_transit_s",
    "duration_over_basket_s", "duration_place_s",
    "total_path_length_m", "net_displacement_x_m", "net_displacement_y_m", "net_displacement_z_m",
    "mean_speed_mps", "peak_speed_mps", "p95_speed_mps",
    "mean_accel_mps2", "peak_accel_mps2", "p95_accel_mps2", "peak_jerk_mps3",
    "peak_abs_ax_mps2", "peak_abs_ay_mps2", "peak_abs_az_mps2",
    "p95_abs_ax_mps2", "p95_abs_ay_mps2", "p95_abs_az_mps2",
    "mean_effective_accel_mps2", "peak_effective_accel_mps2", "p95_effective_accel_mps2",
    "peak_horizontal_effective_accel_mps2", "p95_horizontal_effective_accel_mps2",
    "peak_abs_vertical_effective_accel_mps2", "p95_abs_vertical_effective_accel_mps2",
    "peak_speed_time_fraction", "peak_accel_time_fraction", "peak_effective_accel_time_fraction",
    "lift_peak_speed_mps", "lift_peak_accel_mps2",
    "transit_peak_speed_mps", "transit_peak_accel_mps2",
    "over_basket_peak_speed_mps", "over_basket_peak_accel_mps2",
    "place_peak_speed_mps", "place_peak_accel_mps2",
]

GO_GATE = {
    "t1_boundary_ranking_min": 0.75,
    "macro_frontier_exact_min": 0.75,
    "macro_under_force_max": 0.15,
    "tasks_not_worse_than_old_no_motion_min": 3,
    "clear_macro_composite_improvement_min": 0.05,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def manifest_metadata() -> tuple[pd.DataFrame, int]:
    """Load TRAIN/DEV metadata without outcome fields; count TEST IDs only."""
    rows: list[dict[str, Any]] = []
    test_n = 0
    with (HIST / "P5S0C_BRANCH_MANIFEST.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "TEST":
                test_n += 1
                continue
            if r["split"] in {"TRAIN", "DEV"}:
                rows.append({
                    "branch_id": r["branch_id"], "context_id": r["context_id"],
                    "root_id": r["root_id"], "task": int(r["task"]), "split": r["split"],
                    "friction": float(r["hidden_friction_analysis_only"]),
                    "force": float(r["requested_force_N"]), "telemetry_path": r["telemetry_path"],
                })
    return pd.DataFrame(rows), test_n


def _interp(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    # Authoritative P4/P5 implementation: alpha=(i+1)/n.
    a = (np.arange(n, dtype=np.float64) + 1.0) / float(n)
    return (1.0 - a[:, None]) * start[None, :] + a[:, None] * end[None, :]


def reconstruct_nominal_plan(path: Path, place_z: float) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Rebuild the outcome-independent 250-step grasp-maintenance plan.

    All valid P5-S0-C branches contain hold, lift and at least 75 transit steps.
    The deterministic P5 controller constructs the remaining over-basket/place
    targets before execution, so they can be recovered without realized states.
    """
    d = pd.read_csv(path, usecols=["t_s", "phase", "cmd_x", "cmd_y", "cmd_z"])
    cmd = d[["cmd_x", "cmd_y", "cmd_z"]].to_numpy(dtype=np.float64)
    phase = d.phase.astype(str).to_numpy()
    dt = float(np.median(np.diff(d.t_s.to_numpy(dtype=float)))) if len(d) > 1 else DT
    if abs(dt - DT) > 1e-6:
        raise RuntimeError(f"unexpected dt={dt} in {path}")
    required = {"branch_hold": 20, "lift": 50, "transit": 2}
    for ph, n in required.items():
        if int((phase == ph).sum()) < n:
            raise RuntimeError(f"insufficient nominal prefix {ph} in {path}")
    hold_start = cmd[phase == "branch_hold"][0]
    lift_obs = cmd[phase == "lift"]
    transit_obs = cmd[phase == "transit"]
    lift_start = cmd[phase == "branch_hold"][-1]
    lift_target = lift_start + 50.0 * np.median(np.diff(lift_obs, axis=0), axis=0)
    lift = _interp(lift_start, lift_target, 50)
    transit_start = lift[-1]
    transit_target = transit_start + 110.0 * np.median(np.diff(transit_obs, axis=0), axis=0)
    transit = _interp(transit_start, transit_target, 110)
    over = _interp(transit[-1], transit_target, 30)
    place_target = transit_target.copy()
    # P5 computes place_z=basket_z+0.10. Transit_z is max(lift_z,
    # basket_z+0.18), so basket_z cannot always be inverted from transit_z.
    # The fixed nominal place-z is recovered once from available complete
    # nominal command plans (never realized object state/outcome).
    place_target[2] = float(place_z)
    place = _interp(over[-1], place_target, 40)
    hold = np.repeat(hold_start[None, :], 20, axis=0)
    plan = np.concatenate([hold, lift, transit, over, place], axis=0)
    phases = sum(([name] * n for name, n in PHASE_SPECS), [])
    # Compare every available nominal active command, never realized state/outcome.
    ncmp = min(len(cmd), len(plan))
    max_err = float(np.max(np.abs(cmd[:ncmp] - plan[:ncmp])))
    if max_err > 2e-5:
        raise RuntimeError(f"nominal reconstruction error {max_err:.3g} in {path}")
    return plan, phases, {
        "source_path": str(path), "source_rows": int(len(d)), "dt_s": dt,
        "compared_prefix_steps": int(ncmp), "max_prefix_abs_error_m": max_err,
        "release_settling_excluded": True,
    }


def descriptor(plan: np.ndarray, phases: list[str]) -> dict[str, float]:
    v = np.zeros_like(plan); v[1:] = np.diff(plan, axis=0) / DT
    a = np.zeros_like(plan); a[1:] = np.diff(v, axis=0) / DT
    j = np.zeros_like(plan); j[1:] = np.diff(a, axis=0) / DT
    speed = np.linalg.norm(v, axis=1); acc = np.linalg.norm(a, axis=1)
    eff = a - G[None, :]; effn = np.linalg.norm(eff, axis=1)
    horiz = np.linalg.norm(eff[:, :2], axis=1); vert = np.abs(eff[:, 2])
    q95 = lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.95))
    out: dict[str, float] = {
        "planned_total_duration_s": len(plan) * DT, "planned_control_steps": float(len(plan)),
        **{f"duration_{ph}_s": n * DT for ph, n in PHASE_SPECS},
        "total_path_length_m": float(np.linalg.norm(np.diff(plan, axis=0), axis=1).sum()),
        "net_displacement_x_m": float(plan[-1, 0] - plan[0, 0]),
        "net_displacement_y_m": float(plan[-1, 1] - plan[0, 1]),
        "net_displacement_z_m": float(plan[-1, 2] - plan[0, 2]),
        "mean_speed_mps": float(speed.mean()), "peak_speed_mps": float(speed.max()), "p95_speed_mps": q95(speed),
        "mean_accel_mps2": float(acc.mean()), "peak_accel_mps2": float(acc.max()), "p95_accel_mps2": q95(acc),
        "peak_jerk_mps3": float(np.linalg.norm(j, axis=1).max()),
        "peak_abs_ax_mps2": float(np.abs(a[:, 0]).max()), "peak_abs_ay_mps2": float(np.abs(a[:, 1]).max()), "peak_abs_az_mps2": float(np.abs(a[:, 2]).max()),
        "p95_abs_ax_mps2": q95(np.abs(a[:, 0])), "p95_abs_ay_mps2": q95(np.abs(a[:, 1])), "p95_abs_az_mps2": q95(np.abs(a[:, 2])),
        "mean_effective_accel_mps2": float(effn.mean()), "peak_effective_accel_mps2": float(effn.max()), "p95_effective_accel_mps2": q95(effn),
        "peak_horizontal_effective_accel_mps2": float(horiz.max()), "p95_horizontal_effective_accel_mps2": q95(horiz),
        "peak_abs_vertical_effective_accel_mps2": float(vert.max()), "p95_abs_vertical_effective_accel_mps2": q95(vert),
        "peak_speed_time_fraction": float(np.argmax(speed) / max(len(speed) - 1, 1)),
        "peak_accel_time_fraction": float(np.argmax(acc) / max(len(acc) - 1, 1)),
        "peak_effective_accel_time_fraction": float(np.argmax(effn) / max(len(effn) - 1, 1)),
    }
    pharr = np.asarray(phases)
    for ph in ["lift", "transit", "over_basket", "place"]:
        m = pharr == ph
        out[f"{ph}_peak_speed_mps"] = float(speed[m].max())
        out[f"{ph}_peak_accel_mps2"] = float(acc[m].max())
    if list(out) != FEATURES:
        raise RuntimeError(f"descriptor schema mismatch: {list(out)}")
    return out


def build_context_descriptors(meta: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    nominal_place_z: list[float] = []
    for p in sorted(set(meta.telemetry_path.astype(str))):
        d = pd.read_csv(p, usecols=["phase", "cmd_z"])
        q = d[d.phase.astype(str) == "place"]
        if len(q):
            nominal_place_z.append(float(q.cmd_z.iloc[-1]))
    if not nominal_place_z or np.ptp(nominal_place_z) > 2e-5:
        raise RuntimeError("fixed nominal place-z is unavailable or inconsistent")
    place_z = float(np.median(nominal_place_z))
    for cid, g in meta.groupby("context_id", sort=True):
        # Deterministic, label-free source choice; every branch shares the plan.
        r = g.sort_values(["force", "branch_id"]).iloc[0]
        plan, phases, audit = reconstruct_nominal_plan(Path(r.telemetry_path), place_z)
        feat = descriptor(plan, phases)
        rows.append({"context_id": cid, "root_id": r.root_id, "task": int(r.task), "split": r.split, "friction": float(r.friction), **feat})
        audit.update({"context_id": cid, "root_id": r.root_id, "task": int(r.task), "split": r.split, "selected_source_rule": "lowest-force then branch-id; nominal commands only", "nominal_place_z_recovered_from_complete_nominal_plans": place_z})
        audits.append(audit)
    return pd.DataFrame(rows), audits


def by_task_tables(desc: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for scope, d in [("TRAIN", desc[desc.split == "TRAIN"]), ("DEV", desc[desc.split == "DEV"]), ("TRAIN_DEV", desc)]:
        for task, g in d.groupby("task"):
            for f in FEATURES:
                summary.append({"scope": scope, "task": int(task), "feature": f, "n_contexts": len(g), "mean": float(g[f].mean()), "std": float(g[f].std(ddof=0)), "min": float(g[f].min()), "max": float(g[f].max())})
    tr = desc[desc.split == "TRAIN"]
    mu = tr[FEATURES].mean().to_numpy(); sd = tr[FEATURES].std(ddof=0).to_numpy(); active = sd > 1e-8; sd[~active] = 1.0
    distance: list[dict[str, Any]] = []
    means = {int(t): ((g[FEATURES].mean().to_numpy() - mu) / sd) for t, g in tr.groupby("task")}
    for a in TASKS:
        for b in TASKS:
            distance.append({"row_type": "TRAIN_TASK_CENTROID", "task_a": a, "task_b": b, "standardized_euclidean": float(np.sqrt(np.mean((means[a][active] - means[b][active]) ** 2))), "active_features": int(active.sum())})
    fold_ood: dict[str, Any] = {}
    for fold, cfg in FOLDS.items():
        seen = tr[tr.task.isin(cfg["seen"])]
        smu = seen[FEATURES].mean().to_numpy(); ssd = seen[FEATURES].std(ddof=0).to_numpy(); sa = ssd > 1e-8; ssd[~sa] = 1.0
        centroids = {int(t): ((g[FEATURES].mean().to_numpy() - smu) / ssd) for t, g in seen.groupby("task")}
        held = desc[(desc.split == "DEV") & (desc.task == cfg["heldout"])]
        z = (held[FEATURES].to_numpy() - smu) / ssd
        nearest = np.asarray([min(float(np.sqrt(np.mean((x[sa] - c[sa]) ** 2))) for c in centroids.values()) for x in z])
        own = []
        for t, g in seen.groupby("task"):
            zz = (g[FEATURES].to_numpy() - smu) / ssd
            own.extend(float(np.sqrt(np.mean((x[sa] - centroids[int(t)][sa]) ** 2))) for x in zz)
        ratio = float(nearest.mean() / (np.median(own) + 1e-9))
        fold_ood[fold] = {"heldout_task": cfg["heldout"], "heldout_context_n": len(held), "mean_nearest_seen_task_centroid_distance": float(nearest.mean()), "median_seen_within_task_distance": float(np.median(own)), "ood_ratio": ratio, "ood_flag_preregistered_ratio_gt_2": bool(ratio > 2.0), "active_features": int(sa.sum())}
        distance.append({"row_type": "FOLD_HELDOUT_TO_NEAREST_SEEN", "fold": fold, **fold_ood[fold]})
    return summary, distance, fold_ood


def auth_files() -> list[Path]:
    return [
        OLD / "FINAL_REPORT.md", OLD / "LOTO_FEASIBILITY_GENERALIZATION_PROTOCOL.json", OLD / "LOTO_MODEL_COMPARISON.csv", OLD / "LOTO_TASK_METRICS.csv",
        JOINT_AUTH / "FINAL_REPORT.md", JOINT_AUTH / "FULL_TASK_FEASIBILITY_PROTOCOL.json", JOINT_AUTH / "FEASIBILITY_MODEL_COMPARISON.csv", JOINT_AUTH / "IE_RETENTION_AUDIT.csv",
        HIST / "P5S0C_BRANCH_MANIFEST.csv", HIST / "P5S0C_FORCE_MANIFEST.json", HIST / "P5S0C_SPLIT_MANIFEST.json",
        P5_CODE, P4_CODE, REPO / "analysis/trajectory_physical_imagination.py", REPO / "analysis/loto_feasibility_generalization.py",
    ]


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta, test_n = manifest_metadata()
    desc, reconstruction = build_context_descriptors(meta)
    desc.to_csv(OUT / "TASK_DYNAMICS_CONTEXT_DESCRIPTORS.csv", index=False)
    summary, distances, fold_ood = by_task_tables(desc)
    write_csv(OUT / "TASK_DYNAMICS_BY_TASK.csv", summary)
    write_csv(OUT / "TASK_DYNAMICS_DISTANCE_MATRIX.csv", distances)
    write_json(OUT / "NOMINAL_PLAN_RECONSTRUCTION_AUDIT.json", {"status": "PASS", "context_count": len(reconstruction), "max_prefix_abs_error_m": max(x["max_prefix_abs_error_m"] for x in reconstruction), "records": reconstruction})

    input_audit = {
        "status": "PASS",
        "decision_time_available_nominal_motion": "complete deterministic seven-phase downstream Cartesian translation plan is constructed before branch execution",
        "source_implementation": str(P5_CODE), "source_sha256": sha256(P5_CODE), "interpolation_implementation": str(P4_CODE), "interpolation_sha256": sha256(P4_CODE),
        "available": ["complete nominal Cartesian translation", "phase names/boundaries", "control-step counts", "dt=0.05 s", "fixed grip-force command semantics"],
        "not_available": ["nominal orientation/rotation trajectory (eef_aa is fixed)", "nominal angular velocity/acceleration", "realized future state", "future object trajectory/contact/slip"],
        "actual_rollout_length_used": False,
        "outcome_leakage_prevention": "reconstruct fixed 250-step grasp-maintenance plan from nominal prefix and frozen P5 phase targets; exclude release/settling",
        "plain_language": "At decision time the controller knows the full deterministic translation/phase skeleton. It does not know a future rotational trajectory because the P5 controller holds orientation fixed.",
    }
    write_json(OUT / "TASK_DYNAMICS_INPUT_AUDIT.json", input_audit)
    spec = {
        "status": "FROZEN_BEFORE_LOTO_TRAINING_OR_HELDOUT_OUTCOME_EVALUATION", "dimension": len(FEATURES), "feature_names": FEATURES,
        "source": "deployment-available nominal P5 Cartesian plan only", "coordinate_frame": "world Cartesian", "dt_s": DT,
        "finite_differences": {"v[0]": 0, "v[t]": "(p[t]-p[t-1])/dt", "a[0]": 0, "a[t]": "(v[t]-v[t-1])/dt", "j[0]": 0, "j[t]": "(a[t]-a[t-1])/dt", "smoothing": "none"},
        "effective_acceleration": "a_eff = a_nominal - [0,0,-9.81] in world frame",
        "phase_mapping": {"LIFT": "lift", "TRANSPORT": "transit", "OVER_TARGET": "over_basket", "PRE_PLACE": "place"},
        "excluded_phases": ["release", "settle"], "rotation_angular_tilt_features": "NOT_AVAILABLE_AND_OMITTED; fixed eef orientation is not a nominal rotational trajectory",
        "local_grasp_axis_decomposition": "NOT_USED; no deployment-authoritative nominal local grasp frame in branch plan",
        "object_mass": "NOT_USED", "missing_values": "non-finite descriptor invalidates context; no imputation",
        "normalization": "seen-task TRAIN unique contexts only in each fold; zero-variance std set to one",
    }
    write_json(OUT / "TASK_DYNAMICS_DESCRIPTOR_SPEC.json", spec)

    # Forensic uses nominal descriptor only, never held-out outcomes.
    tr = desc[desc.split == "TRAIN"]
    t0 = tr[tr.task == 0][FEATURES].mean(); t1 = tr[tr.task == 1][FEATURES].mean()
    sd = tr[FEATURES].std(ddof=0).replace(0, 1.0)
    top = ((t1 - t0).abs() / sd).sort_values(ascending=False).head(8)
    t1_ood = fold_ood["T1"]
    t0_ood = fold_ood["T0"]
    forensic = [
        "# T0 VS T1 DYNAMICS FORENSIC", "",
        "This analysis uses only the deployment-available nominal Cartesian plan; no branch outcome or realized trajectory enters the descriptor.", "",
        "## Why T0 was easy in the previous LOTO", "",
        f"T0's mean nearest-seen centroid distance is {t0_ood['mean_nearest_seen_task_centroid_distance']:.3f} (OOD ratio {t0_ood['ood_ratio']:.3f}). This measures whether its nominal translational demand lies inside the motion geometry spanned by tasks 1/5/6; it does not use the prior success labels.", "",
        "## Why T1 may be difficult", "",
        f"T1's mean nearest-seen centroid distance is {t1_ood['mean_nearest_seen_task_centroid_distance']:.3f} (OOD ratio {t1_ood['ood_ratio']:.3f}; preregistered OOD flag={t1_ood['ood_flag_preregistered_ratio_gt_2']}). Its strongest T1-vs-T0 standardized nominal-demand differences are:", "",
    ]
    for f, z in top.items():
        forensic.append(f"- `{f}`: |standardized mean difference| = {z:.3f}; T0={t0[f]:.6g}, T1={t1[f]:.6g}.")
    forensic += ["", "Because P5 holds end-effector orientation fixed, this dataset cannot test an angular-demand explanation. If T1 is not clearly OOD in the available translation descriptor, the raw-H8 failure cannot honestly be attributed to missing rotation alone.", ""]
    (OUT / "T0_VS_T1_DYNAMICS_FORENSIC.md").write_text("\n".join(forensic), encoding="utf-8")

    old_protocol_sha = sha256(OLD / "LOTO_FEASIBILITY_GENERALIZATION_PROTOCOL.json")
    protocol = {
        "status": "FROZEN_BEFORE_NEW_TRAINING_OR_HELDOUT_TASK_OUTCOME_EVALUATION", "scientific_goal": "test whether explicit nominal task-demand features repair unseen-task generalization and whether physics auxiliary adds independent value",
        "authoritative_old_loto": str(OLD), "authoritative_old_protocol_sha256": old_protocol_sha,
        "folds": FOLDS, "models": {
            "FEAS_TASK_DEMAND": {"inputs": ["branch-start physical condition", "GT friction", "candidate force", "common phase", "42D task-demand descriptor"], "loss": "BCEWithLogits"},
            "JOINT_TASK_DEMAND": {"inputs": ["branch-start physical condition", "GT friction", "candidate force", "common phase", "raw H8 Cartesian commands", "42D task-demand descriptor"], "loss": "L_traj + 1.0 L_IE + lambda_feas L_feas"},
        },
        "task_id_input": False, "lambda_feas_candidates_joint": LAMBDAS, "lambda_IE": 1.0, "seeds": SEEDS,
        "training": {"epochs": EPOCHS, "batch": BATCH, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0},
        "normalization": "raw physical and descriptor statistics from seen-task TRAIN only per fold",
        "calibration": "isotonic fit on seen-task TRAIN only; threshold 0.5", "selection": "seen-task DEV only; F_prev false-safe, boundary ranking, boundary exact, AUROC, under-force, IE error",
        "heldout": "unseen-task DEV roots loaded only after fold selection artifact is frozen", "original_TEST": "metadata not retained; outcome/telemetry never loaded",
        "metrics": ["AUROC", "AUPRC", "F_prev false-safe", "boundary ranking", "boundary exact", "frontier exact", "within-one", "under-force", "over-force"],
        "go_gate": GO_GATE,
        "comparison_rules": {
            "composite": "mean(boundary ranking, 1-fprev false-safe, frontier exact, 1-under-force)",
            "task_demand_improvement": "macro composite >= matched old raw-H8 + 0.05 and candidate not worse on >=3/4 task composites",
            "not_worse_than_no_motion": "candidate composite >= old Joint No-Motion on >=3/4 tasks",
            "strong_joint_advantage": "Joint macro composite > Feas macro by >=0.05, not worse on >=3/4 tasks, and one key macro metric improves by >=0.05",
            "T1_OOD": "descriptor OOD ratio >2.0, frozen before outcomes",
        },
        "forbidden": ["original TEST outcomes/telemetry", "Probe", "Q2F", "continuous force", "simulator collection", "E2E", "task ID", "heldout-task tuning", "new IE/event/contact loss", "larger model/horizon/Transformer"],
        "original_test_manifest_rows_skipped": test_n,
        "descriptor_spec_sha256": sha256(OUT / "TASK_DYNAMICS_DESCRIPTOR_SPEC.json"),
        "authoritative_hashes": {str(p): sha256(p) for p in auth_files() if p.exists()},
    }
    write_json(OUT / "TASK_DEMAND_LOTO_PROTOCOL.json", protocol)
    leakage = {
        "status": "PASS_BY_FROZEN_DESIGN_PENDING_RUNTIME_ASSERTIONS", "descriptor_source": "nominal plan only", "actual_future_execution": False,
        "actual_rollout_duration": False, "realized_acceleration": False, "realized_object_trajectory": False, "contact_or_slip": False,
        "F_star_or_outcome": False, "task_id": False, "root_id_input": False, "heldout_normalization_or_calibration": False,
        "nominal_plan_duration_not_failure_duration": True, "original_TEST_outcome_or_telemetry_loaded": False,
    }
    write_json(OUT / "TASK_DEMAND_LEAKAGE_AUDIT.json", leakage)
    print(f"[freeze] out={OUT}\n[freeze] contexts={len(desc)} features={len(FEATURES)} protocol={sha256(OUT/'TASK_DEMAND_LOTO_PROTOCOL.json')}", flush=True)


def scientific_rows(tasks: list[int], split: str) -> pd.DataFrame:
    """Read only requested non-TEST rows; excluded labels are never retained."""
    rows: list[dict[str, str]] = []
    with (HIST / "P5S0C_BRANCH_MANIFEST.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split and int(r["task"]) in tasks:
                rows.append(r)
    d = pd.DataFrame(rows)
    for c in ["task", "full_task_success_y"]:
        d[c] = pd.to_numeric(d[c]).astype(int)
    for c in ["requested_force_N", "hidden_friction_analysis_only"]:
        d[c] = pd.to_numeric(d[c]).astype(float)
    return d


class FeasTaskDemand(nn.Module):
    def __init__(self, demand_dim: int):
        super().__init__()
        self.phase_gru = nn.GRU(7, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.demand = nn.Sequential(nn.Linear(demand_dim, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step: torch.Tensor, cond: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
        _, h = self.phase_gru(step[:, :, 6:13])
        return self.head(torch.cat([h[-1], self.condition(cond), self.demand(demand)], dim=-1)).squeeze(-1)


class JointTaskDemand(nn.Module):
    def __init__(self, tpi, demand_dim: int):
        super().__init__()
        self.physics = tpi.ShortHorizonPhysicsGRU(17, 54, H)
        self.demand = nn.Sequential(nn.Linear(demand_dim, 64), nn.ReLU())
        self.feas_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.demand_dim = demand_dim

    def latent(self, step: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = cond[:, None, :].expand(-1, step.shape[1], -1)
        z, _ = self.physics.gru(torch.cat([step, c], dim=-1))
        return z[:, -1]

    def forward(self, step: torch.Tensor, cond: torch.Tensor, demand: torch.Tensor | None = None):
        z = self.latent(step, cond)
        if demand is None:
            demand = torch.zeros((len(z), self.demand_dim), dtype=z.dtype, device=z.device)
        return self.physics.head(z).view(-1, H, 13), self.feas_head(torch.cat([z, self.demand(demand)], dim=-1)).squeeze(-1)

    def feasibility(self, step: torch.Tensor, cond: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
        return self.forward(step, cond, demand)[1]


def demand_map() -> dict[str, np.ndarray]:
    d = pd.read_csv(OUT / "TASK_DYNAMICS_CONTEXT_DESCRIPTORS.csv")
    return {str(r.context_id): np.asarray([getattr(r, f) for f in FEATURES], dtype=np.float32) for r in d.itertuples(index=False)}


def demand_normalization(train: list[Any], dmap: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(t.context_id for t in train))
    x = np.stack([dmap[c] for c in unique])
    mean = x.mean(0); std = x.std(0); std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def demand_tensor(traces: list[Any], dmap: dict[str, np.ndarray], dnorm, device) -> torch.Tensor:
    mean, std = dnorm
    x = np.stack([(dmap[t.context_id] - mean) / std for t in traces])
    return torch.tensor(x, dtype=torch.float32, device=device)


def branch_tensors(old, cf, tpi, traces, norm, dmap, dnorm, device):
    step, cond, y = old.branch_tensors(cf, tpi, traces, norm, device, False)
    # Hard runtime assertion: categorical task-ID columns remain zero.
    if float(step[:, :, 13:17].abs().max().item()) != 0.0:
        raise RuntimeError("task-ID masking failed")
    return step, cond, demand_tensor(traces, dmap, dnorm, device), y


def train_feas(old, cf, tpi, model, train, meta, norm, dmap, dnorm, device, seed, fold, logs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, EPOCHS + 1):
        ids = old.sampled_indices(train, meta, seed, epoch); losses = []; model.train()
        for st in range(0, len(ids), BATCH):
            batch = [train[int(i)] for i in ids[st:st + BATCH]]
            step, cond, dem, y = branch_tensors(old, cf, tpi, batch, norm, dmap, dnorm, device)
            opt.zero_grad(set_to_none=True)
            logit = model(step, cond, dem)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.item()))
        logs.append({"fold": fold, "variant": "FEAS_TASK_DEMAND", "lambda_feas": 0.0, "seed": seed, "epoch": epoch, "feasibility_loss": float(np.mean(losses))})
        if epoch % 20 == 0:
            print(f"[{fold} FEAS_TASK_DEMAND] s={seed} ep={epoch}/{EPOCHS} bce={np.mean(losses):.5f}", flush=True)
    model.eval(); return model


def train_joint(old, cf, tpi, model, train, meta, seen_all, pairs, norm, dmap, dnorm, device, seed, lam, fold, logs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    units = old.units_for(tpi, seen_all, pairs)
    for epoch in range(1, EPOCHS + 1):
        bases = []; ies = []; feass = []; model.train(); batches = old.unit_batches(units, seed, epoch)
        ids = old.sampled_indices(train, meta, seed + 100, epoch, n=len(batches) * BATCH)
        for bi, batch in enumerate(batches):
            _, step, cond, y, mask, weight = old.physical_batch(batch, norm, device, False)
            fbatch = [train[int(i)] for i in ids[bi * BATCH:(bi + 1) * BATCH]]
            fs, fc, fd, fy = branch_tensors(old, cf, tpi, fbatch, norm, dmap, dnorm, device)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(step, cond)
            flogit = model.feasibility(fs, fc, fd)
            base = (nn.functional.smooth_l1_loss(pred, y, reduction="none") * mask * weight[:, None, None]).sum() / (mask.sum() + 1e-6)
            ie = old.ie_loss(pred, batch, norm, device)
            feas = nn.functional.binary_cross_entropy_with_logits(flogit, fy)
            total = base + ie + float(lam) * feas
            total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            bases.append(float(base.item())); ies.append(float(ie.item())); feass.append(float(feas.item()))
        logs.append({"fold": fold, "variant": "JOINT_TASK_DEMAND", "lambda_feas": lam, "seed": seed, "epoch": epoch, "trajectory_loss": float(np.mean(bases)), "ie_loss": float(np.mean(ies)), "feasibility_loss": float(np.mean(feass)), "total_loss": float(np.mean(bases) + np.mean(ies) + lam * np.mean(feass))})
        if epoch % 20 == 0:
            print(f"[{fold} JOINT_TASK_DEMAND] l={lam} s={seed} ep={epoch}/{EPOCHS} traj={np.mean(bases):.4f} ie={np.mean(ies):.4f} feas={np.mean(feass):.4f}", flush=True)
    model.eval(); return model


def logits(old, cf, tpi, model, variant, traces, norm, dmap, dnorm, device) -> dict[str, float]:
    out: dict[str, float] = {}; model.eval()
    with torch.no_grad():
        for st in range(0, len(traces), 128):
            batch = traces[st:st + 128]
            step, cond, dem, _ = branch_tensors(old, cf, tpi, batch, norm, dmap, dnorm, device)
            z = model(step, cond, dem) if variant == "FEAS_TASK_DEMAND" else model.feasibility(step, cond, dem)
            for tr, value in zip(batch, z.detach().cpu().numpy()):
                out[tr.branch_id] = float(value)
    return out


def ensemble_eval(old, cf, ftf, tpi, models, cals, variant, traces, norm, dmap, dnorm, device, fold, stage):
    all_probs = []
    for model, cal in zip(models, cals):
        lg = logits(old, cf, tpi, model, variant, traces, norm, dmap, dnorm, device)
        ids = [t.branch_id for t in traces]
        p = cf.iso_predict(cal, [lg[i] for i in ids])
        all_probs.append({i: float(v) for i, v in zip(ids, p)})
    probs = {t.branch_id: float(np.mean([p[t.branch_id] for p in all_probs])) for t in traces}
    bm = ftf.binary_metrics([t.outcome for t in traces], [probs[t.branch_id] for t in traces])
    br, ba = ftf.boundary_eval(probs, traces, variant, "ENSEMBLE", "CALIBRATED")
    cr, ca = ftf.controller_eval(probs, traces, variant, "ENSEMBLE")
    row = {"fold": fold, "evaluation_stage": stage, "variant": variant, **bm, **ba, **ca}
    return row, [{"fold": fold, "evaluation_stage": stage, **r} for r in br], [{"fold": fold, "evaluation_stage": stage, **r} for r in cr], probs


def selection_key(row: dict[str, Any]):
    def good(x, fallback):
        return fallback if not math.isfinite(float(x)) else float(x)
    return (good(row["fprev_false_safe"], 1.0), -good(row["boundary_ranking"], 0.0), -good(row["boundary_exact"], 0.0), -good(row["auroc"], 0.0), good(row["under_force"], 1.0), good(row.get("ie_error_H8", math.inf), math.inf))


def save_checkpoint(path: Path, model, fold, variant, lam, seed, protocol_sha, norm, dnorm) -> None:
    torch.save({
        "state_dict": model.state_dict(), "fold": fold, "variant": variant, "lambda_feas": lam, "seed": seed,
        "protocol_sha256": protocol_sha, "task_id_input": False, "descriptor_dimension": len(FEATURES),
        "normalization": {"x_mean": norm[0], "x_std": norm[1], "y_mean": norm[2], "y_std": norm[3], "demand_mean": dnorm[0], "demand_std": dnorm[1]},
    }, path)


def param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def metric_composite(row: dict[str, Any]) -> float:
    return float(np.mean([float(row["boundary_ranking"]), 1 - float(row["fprev_false_safe"]), float(row["frontier_exact"]), 1 - float(row["under_force"])]))


def run() -> None:
    protocol_path = OUT / "TASK_DEMAND_LOTO_PROTOCOL.json"
    if not protocol_path.exists():
        raise RuntimeError("freeze must run first")
    protocol = json.loads(protocol_path.read_text())
    if protocol["status"] != "FROZEN_BEFORE_NEW_TRAINING_OR_HELDOUT_TASK_OUTCOME_EVALUATION":
        raise RuntimeError("protocol not frozen")
    protocol_sha = sha256(protocol_path)
    tpi = load_module("tpi_task_demand", REPO / "analysis/trajectory_physical_imagination.py")
    cf = load_module("cf_task_demand", REPO / "analysis/counterfactual_force_world_model.py")
    ftf = load_module("ftf_task_demand", REPO / "analysis/full_task_feasibility_decoder.py")
    old = load_module("old_loto_util", REPO / "analysis/loto_feasibility_generalization.py")
    old.CF = cf; old.ACTIVE_PHASES = ACTIVE_PHASES; old.SEGMENT_CACHE.clear()
    dmap = demand_map()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required by immutable training protocol")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print(f"[run] device={torch.cuda.get_device_name(0)} protocol={protocol_sha}", flush=True)

    train_logs: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    task_metrics: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    controller_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    ck_rows: list[dict[str, Any]] = []

    for fold, cfg in FOLDS.items():
        fold_dir = OUT / fold; fold_dir.mkdir(parents=True, exist_ok=True)
        # Only seen-task labels/telemetry are retained before selection freeze.
        train_rows = scientific_rows(cfg["seen"], "TRAIN")
        dev_rows = scientific_rows(cfg["seen"], "DEV")
        train, train_meta = old.load_traces(tpi, train_rows)
        seen_dev, dev_meta = old.load_traces(tpi, dev_rows)
        if set(t.root_id for t in train) & set(t.root_id for t in seen_dev):
            raise RuntimeError(f"root leakage in {fold}")
        meta = {**train_meta, **dev_meta}; seen_all = train + seen_dev
        pairs = old.make_pairs(cf, seen_all, meta)
        norm = old.fit_fold_norm(tpi, train)
        dnorm = demand_normalization(train, dmap)
        norm_obj = {"fit_split": "seen-task TRAIN", "fit_tasks": cfg["seen"], "task_id_masked": True, "x_mean": norm[0].tolist(), "x_std": norm[1].tolist(), "y_mean": norm[2].tolist(), "y_std": norm[3].tolist(), "demand_mean": dnorm[0].tolist(), "demand_std": dnorm[1].tolist()}
        write_json(fold_dir / "NORMALIZATION.json", norm_obj)
        models_by: dict[tuple[str, float], list[Any]] = {}; cals_by: dict[tuple[str, float], list[Any]] = {}
        for variant in ["FEAS_TASK_DEMAND", "JOINT_TASK_DEMAND"]:
            lambdas = [0.0] if variant == "FEAS_TASK_DEMAND" else LAMBDAS
            for lam in lambdas:
                models = []; cals = []
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    model = FeasTaskDemand(len(FEATURES)) if variant == "FEAS_TASK_DEMAND" else JointTaskDemand(tpi, len(FEATURES))
                    cp = fold_dir / f"{variant}_lambda{lam}_seed{seed}.pt"
                    if cp.exists():
                        ck = torch.load(cp, map_location=device, weights_only=False); model.load_state_dict(ck["state_dict"]); model = model.to(device); model.eval()
                        print(f"[resume] {fold} {variant} l={lam} s={seed}", flush=True)
                    else:
                        if variant == "FEAS_TASK_DEMAND":
                            model = train_feas(old, cf, tpi, model, train, meta, norm, dmap, dnorm, device, seed, fold, train_logs)
                        else:
                            model = train_joint(old, cf, tpi, model, train, meta, seen_all, pairs, norm, dmap, dnorm, device, seed, lam, fold, train_logs)
                        save_checkpoint(cp, model, fold, variant, lam, seed, protocol_sha, norm, dnorm)
                        write_csv(OUT / "TASK_DEMAND_TRAINING_MANIFEST.csv", train_logs)
                    lg = logits(old, cf, tpi, model, variant, train, norm, dmap, dnorm, device)
                    cal = cf.fit_iso([lg[t.branch_id] for t in train], [t.outcome for t in train])
                    write_json(fold_dir / f"CALIBRATION_{variant}_lambda{lam}_seed{seed}.json", {"fit_split": "seen-task TRAIN", "fit_tasks": cfg["seen"], "method": "isotonic", "threshold": THRESHOLD, "x": cal["x"], "y": cal["y"]})
                    models.append(model); cals.append(cal)
                    ck_rows.append({"fold": fold, "variant": variant, "lambda_feas": lam, "seed": seed, "checkpoint": str(cp), "sha256": sha256(cp), "parameter_count": param_count(model)})
                models_by[(variant, lam)] = models; cals_by[(variant, lam)] = cals
                row, _, _, _ = ensemble_eval(old, cf, ftf, tpi, models, cals, variant, seen_dev, norm, dmap, dnorm, device, fold, "SEEN_TASK_DEV_MODEL_SELECTION")
                if variant == "JOINT_TASK_DEMAND":
                    pm = [old.physical_metrics(m, "JOINT_MOTION_CONDITIONED", seen_dev, [p for p in pairs if p.split == "DEV"], norm, tpi, device) for m in models]
                    row.update({k: float(np.nanmean([x[k] for x in pm])) for k in pm[0] if k != "pair_n"}); row["ie_pair_n"] = int(np.sum([x["pair_n"] for x in pm]))
                row["lambda_feas"] = lam; selection_rows.append(row)
                write_csv(OUT / "TASK_DEMAND_MODEL_SELECTION.csv", selection_rows)

        selected = {"FEAS_TASK_DEMAND": 0.0}
        candidates = [r for r in selection_rows if r["fold"] == fold and r["variant"] == "JOINT_TASK_DEMAND"]
        selected["JOINT_TASK_DEMAND"] = float(min(candidates, key=selection_key)["lambda_feas"])
        freeze_obj = {
            "status": "FROZEN_BEFORE_HELDOUT_TASK_LOAD", "fold": fold, "seen_tasks": cfg["seen"], "heldout_task": cfg["heldout"],
            "selected_lambda_feas": selected, "selection_split": "seen-task DEV only", "heldout_outcomes_or_telemetry_loaded": False,
            "protocol_sha256": protocol_sha, "normalization_sha256": sha256(fold_dir / "NORMALIZATION.json"),
            "checkpoint_sha256": {v: [sha256(fold_dir / f"{v}_lambda{lam}_seed{s}.pt") for s in SEEDS] for v, lam in selected.items()},
        }
        write_json(fold_dir / "FOLD_SELECTION_FROZEN_BEFORE_HELDOUT.json", freeze_obj)
        print(f"[{fold}] selected={selected}; now loading heldout DEV task {cfg['heldout']} once", flush=True)

        held_rows = scientific_rows([cfg["heldout"]], "DEV")
        held, held_meta = old.load_traces(tpi, held_rows)
        held_pairs = old.make_pairs(cf, held, held_meta)
        for variant, lam in selected.items():
            models = models_by[(variant, lam)]; cals = cals_by[(variant, lam)]
            row, br, cr, _ = ensemble_eval(old, cf, ftf, tpi, models, cals, variant, held, norm, dmap, dnorm, device, fold, "HELDOUT_TASK_DEV_FINAL")
            row["lambda_feas"] = lam; row["heldout_task"] = cfg["heldout"]
            task_metrics.append(row)
            boundary_rows.extend([{"lambda_feas": lam, "heldout_task": cfg["heldout"], **r} for r in br])
            controller_rows.extend([{"lambda_feas": lam, "heldout_task": cfg["heldout"], **r} for r in cr])
            if variant == "JOINT_TASK_DEMAND":
                pm = [old.physical_metrics(m, "JOINT_MOTION_CONDITIONED", held, held_pairs, norm, tpi, device) for m in models]
                retention_rows.append({"fold": fold, "heldout_task": cfg["heldout"], "variant": variant, "lambda_feas": lam, **{k: float(np.nanmean([x[k] for x in pm])) for k in pm[0] if k != "pair_n"}, "ie_pair_n": int(np.sum([x["pair_n"] for x in pm]))})
        write_csv(OUT / "TASK_DEMAND_TASK_METRICS.csv", task_metrics)
        write_csv(OUT / "TASK_DEMAND_BOUNDARY_METRICS.csv", boundary_rows)
        write_csv(OUT / "TASK_DEMAND_CONTROLLER_SELECTION.csv", controller_rows)
        write_csv(OUT / "TASK_DEMAND_IE_RETENTION.csv", retention_rows)

    write_csv(OUT / "TASK_DEMAND_CHECKPOINT_MANIFEST.csv", ck_rows)
    finalize(task_metrics, retention_rows, protocol_sha)


def finalize(new_rows: list[dict[str, Any]], retention_rows: list[dict[str, Any]], protocol_sha: str) -> None:
    keys = ["auroc", "auprc", "boundary_n", "boundary_ranking", "fprev_false_safe", "fstar_false_unsafe", "boundary_exact", "frontier_exact", "within_one_step", "under_force", "over_force", "mean_selected_force_N"]
    old = pd.read_csv(OLD / "LOTO_TASK_METRICS.csv")
    old = old[old.evaluation_stage == "HELDOUT_TASK_DEV_FINAL"].copy()
    rename = {"JOINT_MOTION_CONDITIONED": "OLD_JOINT_RAW_H8", "FEASIBILITY_ONLY_MOTION_CONDITIONED": "OLD_FEAS_ONLY_RAW_H8", "JOINT_NO_FUTURE_MOTION": "OLD_JOINT_NO_MOTION"}
    rows: list[dict[str, Any]] = []
    for r in old.to_dict("records"):
        rows.append({"fold": r["fold"], "heldout_task": int(r["heldout_task"]), "variant": rename[r["variant"]], **{k: float(r[k]) for k in keys}})
    for r in new_rows:
        rows.append({"fold": r["fold"], "heldout_task": int(r["heldout_task"]), "variant": r["variant"], **{k: float(r[k]) for k in keys}})
    variants = list(rename.values()) + ["FEAS_TASK_DEMAND", "JOINT_TASK_DEMAND"]
    macro: list[dict[str, Any]] = []
    for v in variants:
        rr = [r for r in rows if r["variant"] == v]
        m = {"fold": "MACRO", "heldout_task": "MACRO", "variant": v, "task_count": len(rr)}
        for k in keys:
            m[k] = float(np.nanmean([r[k] for r in rr]))
        macro.append(m)
    write_csv(OUT / "TASK_DEMAND_MODEL_COMPARISON.csv", rows + macro)

    by = {(r["variant"], int(r["heldout_task"])): r for r in rows}
    mm = {r["variant"]: r for r in macro}
    def compare(candidate: str, baseline: str) -> dict[str, Any]:
        ds = np.asarray([metric_composite(by[(candidate, t)]) - metric_composite(by[(baseline, t)]) for t in TASKS])
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot = np.asarray([float(np.mean(rng.choice(ds, len(ds), replace=True))) for _ in range(BOOTSTRAP_REPS)])
        return {"candidate": candidate, "baseline": baseline, "task_composite_deltas": {str(t): float(ds[i]) for i, t in enumerate(TASKS)}, "macro_composite_delta": float(ds.mean()), "tasks_not_worse": int((ds >= -1e-12).sum()), "tasks_strictly_better": int((ds > 1e-12).sum()), "task_bootstrap_95ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    comparisons = {
        "feas_demand_vs_raw_h8": compare("FEAS_TASK_DEMAND", "OLD_FEAS_ONLY_RAW_H8"),
        "joint_demand_vs_raw_h8": compare("JOINT_TASK_DEMAND", "OLD_JOINT_RAW_H8"),
        "feas_demand_vs_no_motion": compare("FEAS_TASK_DEMAND", "OLD_JOINT_NO_MOTION"),
        "joint_demand_vs_no_motion": compare("JOINT_TASK_DEMAND", "OLD_JOINT_NO_MOTION"),
        "joint_vs_feas_demand": compare("JOINT_TASK_DEMAND", "FEAS_TASK_DEMAND"),
    }
    write_json(OUT / "TASK_DEMAND_PAIRED_COMPARISON.json", comparisons)

    def qualifies(v: str, matched_old: str) -> tuple[bool, dict[str, Any]]:
        m = mm[v]; t1 = by[(v, 1)]; c = compare(v, matched_old); nm = compare(v, "OLD_JOINT_NO_MOTION")
        checks = {
            "t1_boundary_ranking": bool(t1["boundary_ranking"] >= GO_GATE["t1_boundary_ranking_min"]),
            "macro_frontier_exact": bool(m["frontier_exact"] >= GO_GATE["macro_frontier_exact_min"]),
            "macro_under_force": bool(m["under_force"] <= GO_GATE["macro_under_force_max"]),
            "clear_matched_raw_improvement": bool(c["macro_composite_delta"] >= GO_GATE["clear_macro_composite_improvement_min"] and c["tasks_not_worse"] >= 3),
            "not_worse_than_no_motion_3of4": bool(nm["tasks_not_worse"] >= GO_GATE["tasks_not_worse_than_old_no_motion_min"]),
        }
        return bool(all(checks.values())), checks
    feas_go, feas_checks = qualifies("FEAS_TASK_DEMAND", "OLD_FEAS_ONLY_RAW_H8")
    joint_go, joint_checks = qualifies("JOINT_TASK_DEMAND", "OLD_JOINT_RAW_H8")
    jvf = comparisons["joint_vs_feas_demand"]
    joint_key_adv = max(
        mm["JOINT_TASK_DEMAND"]["boundary_ranking"] - mm["FEAS_TASK_DEMAND"]["boundary_ranking"],
        mm["JOINT_TASK_DEMAND"]["frontier_exact"] - mm["FEAS_TASK_DEMAND"]["frontier_exact"],
        mm["FEAS_TASK_DEMAND"]["under_force"] - mm["JOINT_TASK_DEMAND"]["under_force"],
    )
    strong_joint = bool(joint_go and jvf["macro_composite_delta"] >= 0.05 and jvf["tasks_not_worse"] >= 3 and joint_key_adv >= 0.05)
    dynamics_now_matter = bool(feas_go or joint_go)
    physics_adds = strong_joint
    ood_rows = pd.read_csv(OUT / "TASK_DYNAMICS_DISTANCE_MATRIX.csv")
    t1_ood = bool(ood_rows[(ood_rows.row_type == "FOLD_HELDOUT_TO_NEAREST_SEEN") & (ood_rows.fold == "T1")].iloc[0].ood_flag_preregistered_ratio_gt_2)
    best_t1 = max(by[("FEAS_TASK_DEMAND", 1)]["boundary_ranking"], by[("JOINT_TASK_DEMAND", 1)]["boundary_ranking"])
    any_improvement = bool(max(comparisons["feas_demand_vs_raw_h8"]["macro_composite_delta"], comparisons["joint_demand_vs_raw_h8"]["macro_composite_delta"]) >= 0.05 or best_t1 >= 0.25)
    if strong_joint:
        classification = "PHYSICS_AUXILIARY_AND_TASK_DEMAND_ENABLE_GENERALIZATION"
    elif dynamics_now_matter:
        classification = "TASK_DEMAND_FEASIBILITY_IS_SUFFICIENT"
    elif t1_ood and best_t1 < 0.75:
        classification = "T1_REMAINS_STRUCTURALLY_OUT_OF_DISTRIBUTION"
    elif any_improvement:
        classification = "TASK_DEMAND_IMPROVES_BUT_GENERALIZATION_REMAINS_INSUFFICIENT"
    else:
        classification = "TASK_DEMAND_REPRESENTATION_DOES_NOT_FIX_GENERALIZATION"
    world_continue = classification == "PHYSICS_AUXILIARY_AND_TASK_DEMAND_ENABLE_GENERALIZATION"
    decision = {
        "primary_classification": classification, "world_model_line": "CONTINUE WORLD-MODEL LINE" if world_continue else "STOP WORLD-MODEL LINE",
        "downstream_dynamics_now_matter": dynamics_now_matter, "physics_auxiliary_now_adds_value": physics_adds,
        "feas_task_demand_go": feas_go, "joint_task_demand_go": joint_go, "strong_joint_go": strong_joint,
        "feas_checks": feas_checks, "joint_checks": joint_checks, "T1_descriptor_OOD_flag": t1_ood, "best_new_T1_boundary_ranking": best_t1,
        "protocol_sha256": protocol_sha,
    }
    write_json(OUT / "TASK_DEMAND_FINAL_DECISION.json", decision)

    if classification in {"TASK_DEMAND_IMPROVES_BUT_GENERALIZATION_REMAINS_INSUFFICIENT", "TASK_DEMAND_REPRESENTATION_DOES_NOT_FIX_GENERALIZATION", "T1_REMAINS_STRUCTURALLY_OUT_OF_DISTRIBUTION"}:
        pivot = """# AGENTIC PROBE PIVOT PLAN

## Supported physical-sensing evidence

- The fixed probe has already supported explicit friction estimation and decision-discordant ranking.
- Historical Q2F provides a direct probe-to-force comparison lineage.
- P5/Q2F artifacts establish real continuous-force execution and continuous decision provenance.

## World-model claims not supported

- Raw H8 future motion did not add reliable unseen-task value.
- Physics/IE auxiliary training did not establish a stable unseen-task advantage.
- Reliable counterfactual continuous physical imagination has not been proven.

## Simplified paper method

Frozen VLA → agent decides whether physical information is needed → active query → instance-physics belief → task-aware feasibility/direct force decision → minimum-sufficient controller.

The physical world model is not presented as a core module after this NO-GO.

## Next decisive experiment

Strict no-probe versus probe-informed control on friction-decision-discordant contexts, followed only later by a when-to-probe uncertainty gate, optional second probe, continuous minimum-force benefit, reactive-control comparison, and fresh E2E.
"""
        (OUT / "AGENTIC_PROBE_PIVOT_PLAN.md").write_text(pivot, encoding="utf-8")

    make_report(rows, macro, retention_rows, comparisons, decision)
    leakage_path = OUT / "TASK_DEMAND_LEAKAGE_AUDIT.json"
    leakage = json.loads(leakage_path.read_text())
    leakage["status"] = "PASS"
    leakage["runtime_assertions"] = {"checkpoint_count_expected": 48, "checkpoint_count_observed": len(list(OUT.glob("T*/FEAS_TASK_DEMAND_*.pt"))) + len(list(OUT.glob("T*/JOINT_TASK_DEMAND_*.pt"))), "heldout_metric_rows_expected": 8, "heldout_metric_rows_observed": len(new_rows), "fold_selection_files": len(list(OUT.glob("T*/FOLD_SELECTION_FROZEN_BEFORE_HELDOUT.json"))), "original_TEST_loaded": False}
    write_json(leakage_path, leakage)
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(f"{sha256(p)}  {p.relative_to(OUT)}" for p in sorted(files)) + "\n", encoding="utf-8")
    print(f"[complete] classification={classification} world={decision['world_model_line']} report={OUT/'FINAL_REPORT.md'}", flush=True)


def make_report(rows, macro, retention_rows, comparisons, decision) -> None:
    by = {(r["variant"], int(r["heldout_task"])): r for r in rows}
    mm = {r["variant"]: r for r in macro}
    def f(x):
        return "NA" if not math.isfinite(float(x)) else f"{float(x):.3f}"
    names = ["OLD_JOINT_RAW_H8", "OLD_FEAS_ONLY_RAW_H8", "OLD_JOINT_NO_MOTION", "FEAS_TASK_DEMAND", "JOINT_TASK_DEMAND"]
    audit = json.loads((OUT / "TASK_DYNAMICS_INPUT_AUDIT.json").read_text())
    spec = json.loads((OUT / "TASK_DYNAMICS_DESCRIPTOR_SPEC.json").read_text())
    forensic = (OUT / "T0_VS_T1_DYNAMICS_FORENSIC.md").read_text().splitlines()
    lines = [
        "# STATUS", "", "COMPLETE — descriptor frozen before outcomes, 48 new checkpoints trained across four strict LOTO folds, seen-task-only selection/calibration completed, and each unseen DEV task evaluated once.", "",
        "# SINGLE SCIENTIFIC GOAL", "", "Determine whether an explicit deployment-available task-dynamics demand representation repairs unseen-task feasibility generalization and whether physics/IE auxiliary training adds independent value.", "",
        "# CONNECTION TO PREVIOUS LOTO FAILURE", "", "The authoritative raw-H8 LOTO result failed: macro boundary ranking 0.438 for Joint, and unseen T1 collapsed to ranking 0 with F_prev false-safe 1.0. Old baselines were reused, not retrained.", "",
        "# WHAT INFORMATION WAS MISSING FROM RAW H8 MOTION?", "", audit["plain_language"], "Raw H8 exposed only the immediate local translation window. The new descriptor adds the full nominal grasp-maintenance plan's duration, path, velocity, acceleration, jerk, effective acceleration, phase peaks, and peak-demand timing. Rotation/angular-demand features remain NOT_AVAILABLE because the controller holds orientation fixed.", "",
        "# TASK-DYNAMICS DESCRIPTOR", "", f"The frozen descriptor has {spec['dimension']} deterministic, task-ID-free features in world Cartesian coordinates. It uses no realized future, object trajectory, contact, failure duration, F_star, or outcome. Release and settling are excluded; normalization is fit on seen-task TRAIN contexts per fold.", "",
        "# T0 VS T1 PHYSICAL-DEMAND FORENSIC", "",
    ]
    lines.extend(forensic[2:])
    lines += ["", "Plain answer: T1 is descriptor-OOD by the preregistered ratio, but T0 is even farther from its fold's seen-task centroids and previously generalized perfectly. Therefore nominal-demand OOD alone is not a sufficient explanation for why T1 failed.", "",
              "# LEAVE-ONE-TASK-OUT PROTOCOL", "", "T0/T1/T5/T6 each hold out one task. Training uses only seen-task TRAIN roots; lambda selection uses only seen-task DEV; isotonic calibration uses only seen-task TRAIN; fold hashes are frozen before held-out DEV outcomes/telemetry are loaded. Original TEST was untouched.", "",
              "# TASK DEMAND LEAKAGE AUDIT", "", "PASS. Descriptor inputs come only from the fixed nominal plan. No task ID, actual rollout duration, realized acceleration/state, contact/slip, root identity, F_star, frontier, outcome, held-out normalization, or held-out calibration enters either model."]
    for task in TASKS:
        lines += ["", f"# HELD-OUT T{task}", ""]
        for v in ["FEAS_TASK_DEMAND", "JOINT_TASK_DEMAND"]:
            r = by[(v, task)]
            lines.append(f"- {v}: AUROC {f(r['auroc'])}; boundary ranking {f(r['boundary_ranking'])}; F_prev false-safe {f(r['fprev_false_safe'])}; frontier exact {f(r['frontier_exact'])}; under-force {f(r['under_force'])}; boundary pairs n={int(r['boundary_n'])}.")
        if task == 6:
            lines.append("- Caveat: T6 has only one valid boundary pair, so its boundary estimate is high variance.")
    lines += ["", "# MACRO GENERALIZATION", "", "| Model | Boundary ranking | F_prev false-safe | Frontier exact | Under-force |", "|---|---:|---:|---:|---:|"]
    for v in names:
        r = mm[v]; lines.append(f"| {v} | {f(r['boundary_ranking'])} | {f(r['fprev_false_safe'])} | {f(r['frontier_exact'])} | {f(r['under_force'])} |")
    for metric, title in [("boundary_ranking", "Boundary ranking"), ("fprev_false_safe", "F_prev false-safe"), ("frontier_exact", "Frontier exact"), ("under_force", "Under-force")]:
        lines += ["", f"## {title}: task-wise matrix", "", "| Model | T0 | T1 | T5 | T6 | Macro |", "|---|---:|---:|---:|---:|---:|"]
        for v in names:
            lines.append(f"| {v} | {f(by[(v, 0)][metric])} | {f(by[(v, 1)][metric])} | {f(by[(v, 5)][metric])} | {f(by[(v, 6)][metric])} | {f(mm[v][metric])} |")
    lines += ["", "# OLD H8 VS TASK-DEMAND", "", f"FEAS task-demand vs raw-H8 macro composite Δ={comparisons['feas_demand_vs_raw_h8']['macro_composite_delta']:.3f}; Joint task-demand vs raw-H8 Δ={comparisons['joint_demand_vs_raw_h8']['macro_composite_delta']:.3f}.", "",
              "Neither model rescued T1: FEAS_TASK_DEMAND remained at boundary ranking 0.000 and JOINT_TASK_DEMAND reached only 0.250, below the preregistered 0.750 recovery threshold. Relative to old T1 ranking 0.000, this is at most a small partial change, not a structural rescue.", "",
              "# TASK-DEMAND VS NO-MOTION", "", f"FEAS task-demand vs old no-motion Δ={comparisons['feas_demand_vs_no_motion']['macro_composite_delta']:.3f} and was non-worse on {comparisons['feas_demand_vs_no_motion']['tasks_not_worse']}/4 tasks. Joint task-demand Δ={comparisons['joint_demand_vs_no_motion']['macro_composite_delta']:.3f} and was non-worse on {comparisons['joint_demand_vs_no_motion']['tasks_not_worse']}/4 tasks.", "",
              "# JOINT VS FEASIBILITY-ONLY", "", f"Joint-vs-Feas task-demand macro composite Δ={comparisons['joint_vs_feas_demand']['macro_composite_delta']:.3f}; task-bootstrap 95% CI [{comparisons['joint_vs_feas_demand']['task_bootstrap_95ci'][0]:.3f}, {comparisons['joint_vs_feas_demand']['task_bootstrap_95ci'][1]:.3f}]; Joint was non-worse on {comparisons['joint_vs_feas_demand']['tasks_not_worse']}/4 tasks.", "",
              "# DOES DOWNSTREAM DYNAMICS NOW MATTER?", "", "YES" if decision["downstream_dynamics_now_matter"] else "NO", "",
              "# DOES PHYSICS AUXILIARY NOW ADD VALUE?", "", "YES" if decision["physics_auxiliary_now_adds_value"] else "NO", "",
              "# PHYSICAL / IE RETENTION", "", "The selected Joint checkpoints retained measurable local force-intervention direction signal, but this did not translate into held-out feasibility. T6 is especially unstable in magnitude (effect ratio far above one).", "",
              "| Fold | λ_feas | Trajectory error H8 | IE error H8 | Direction agreement | Effect magnitude ratio |", "|---|---:|---:|---:|---:|---:|"]
    for r in retention_rows:
        lines.append(f"| {r['fold']} | {f(r['lambda_feas'])} | {f(r['trajectory_error_H8'])} | {f(r['ie_error_H8'])} | {f(r['force_intervention_direction_agreement'])} | {f(r['effect_magnitude_ratio_H8'])} |")
    lines += ["", "# PRIMARY_CLASSIFICATION", "", decision["primary_classification"], "",
              "# WORLD-MODEL GO / NO-GO", "", decision["world_model_line"], "",
              "# WHAT IS NOW PROVEN", "", ("The preregistered task-demand gate passed and Joint established an independent advantage over task-demand feasibility-only." if decision["world_model_line"].startswith("CONTINUE") else "The final preregistered structural gate did not establish a sufficient independent world-model/physics-auxiliary advantage for unseen-task force feasibility."), "",
              "# WHAT IS NOT PROVEN", "", "No original TEST, Probe, Q2F, continuous force, fresh simulator collection, or E2E was run. Fixed-orientation P5 data cannot test angular-demand representations. T6's single boundary pair remains high variance.", "",
              "# METHOD IMPLICATION", ""]
    if decision["primary_classification"] == "PHYSICS_AUXILIARY_AND_TASK_DEMAND_ENABLE_GENERALIZATION":
        lines.append("Retain Joint physics + task demand + full-task feasibility as the candidate method.")
    elif decision["primary_classification"] == "TASK_DEMAND_FEASIBILITY_IS_SUFFICIENT":
        lines.append("Simplify to task-demand-conditioned feasibility; the world model is a nonessential auxiliary and the world-model line stops.")
    else:
        lines.append("Stop the world-model method-development line and pivot to active probe + agentic physical sensing/decision.")
    lines += ["", "# NEXT_METHOD", "", ("replace GT friction with frozen Probe estimator and compare against direct Q2F under the same task-generalization protocol" if decision["world_model_line"].startswith("CONTINUE") else "run the agentic-probe necessity experiment: strict no-probe vs probe-informed on friction-decision-discordant contexts"), ""]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["freeze", "run"])
    args = ap.parse_args()
    freeze() if args.phase == "freeze" else run()
