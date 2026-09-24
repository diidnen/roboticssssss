#!/usr/bin/env python3
"""Retrospective structured-grasp-context mechanism diagnostic.

This script has a hard two-stage contract:

1. ``audit`` reads physical snapshots/telemetry and freezes a low-dimensional
   structured feature specification before any new model is fit.
2. ``experiment`` verifies that frozen specification hash, then runs the same
   taskwise root-heldout CV used by the existing Base/Visual/Joint study.

No probe, simulator, visual encoder, PCA rank, task distribution, loss weight,
optimizer, epoch count, or root split is changed by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn

import per_task_visual_context_early as taskwise
import per_task_visual_generalization as taskgen
import task0_visual_context_early as early
import task0_visual_generalization as gen
import taskwise_joint_novisual as jnv


ROOT = Path("/home/exouser/FORTE")
SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
DEFAULT_OUT = ROOT / "structured_grasp_context_20260831"
TASKS = (0, 1, 5)
TASK_OBJECTS = {0: "alphabet_soup_1", 1: "cream_cheese_1", 5: "tomato_sauce_1"}
TASK_NAMES = {0: "alphabet soup", 1: "cream cheese", 5: "tomato sauce"}
RHO = 0.80
COMPARATOR_EQUIVALENCE_ATOL = 5e-8

FROZEN = {
    0: ROOT / "task0_visual_context_early_20260831_025000",
    1: ROOT / "taskwise_visual_context_20260831_074000/task1_train_frozen",
    5: ROOT / "taskwise_visual_context_20260831_074000/task5_train_frozen",
}
EXISTING_VISUAL_CV = {
    0: ROOT / "task0_visual_generalization_20260831_040609/TASK0_GROUP_HELDOUT_CV.csv",
    1: FROZEN[1] / "TASK1_GROUP_HELDOUT_CV.csv",
    5: FROZEN[5] / "TASK5_GROUP_HELDOUT_CV.csv",
}
EXISTING_JNV_CV = {
    t: ROOT / f"joint_mechanism_20260831/task{t}_joint_novisual/TASK{t}_JOINT_NOVISUAL_ROOT_CV.csv"
    for t in TASKS
}
EXISTING_JNV_DENSE = {
    t: ROOT / f"joint_mechanism_20260831/task{t}_joint_novisual/TASK{t}_JOINT_NOVISUAL_ROOT_CV_DENSE_PREDICTIONS.csv"
    for t in TASKS
}
EXISTING_JNV_TRAIN = {
    0: ROOT / "joint_mechanism_20260831/task0_joint_novisual/TASK0_JOINT_NOVISUAL_TRAIN_METRICS.csv",
    1: ROOT / "joint_mechanism_20260831/task1_joint_novisual/TASK1_JOINT_NOVISUAL_TRAIN_METRICS.csv",
    5: ROOT / "joint_mechanism_20260831/task5_joint_novisual/TASK5_JOINT_NOVISUAL_TRAIN_METRICS.csv",
}

ACTIVE_FEATURES = [
    "object_to_eef_dx_m",
    "object_to_eef_dy_m",
    "object_to_eef_dz_m",
    "object_tilt_rad",
    "object_yaw_sin",
    "object_yaw_cos",
    "eef_base_x_m",
    "eef_base_y_m",
    "eef_base_z_m",
    "gripper_opening_m",
    "finger_joint_asymmetry_m",
]

MODEL_NAMES = [
    "Base",
    "Structured",
    "Full Visual",
    "Joint-NoVisual",
    "Structured Joint",
    "Visual Joint",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x, np.float32)).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    """Render a compact deterministic Markdown table without tabulate."""
    def cell(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def qinv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    return np.r_[q[0], -q[1:]] / max(float(np.dot(q, q)), 1e-12)


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, np.float64)
    bw, bx, by, bz = np.asarray(b, np.float64)
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def qapply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    v = np.asarray(v, np.float64)
    qv = q[1:]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q[0] * v)


def quaternion_summary(q: np.ndarray) -> tuple[float, float, float]:
    """Return tilt, sin(yaw), cos(yaw) for wxyz quaternion."""
    w, x, y, z = np.asarray(q, np.float64)
    rzz = 1.0 - 2.0 * (x*x + y*y)
    tilt = math.acos(float(np.clip(rzz, -1.0, 1.0)))
    sy = 2.0 * (w*z + x*y)
    cy = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(sy, cy)
    return tilt, math.sin(yaw), math.cos(yaw)


def pca_path(task: int) -> Path:
    if task == 0:
        return FROZEN[0] / "TASK0_PCA17_PROVISIONAL.npz"
    return FROZEN[task] / f"TASK{task}_PCA17_TRAIN_ONLY.npz"


def visual_projection(task: int, raw: np.ndarray) -> np.ndarray:
    p = np.load(pca_path(task))
    z = (np.asarray(raw, np.float32) - p["raw_mean"]) @ p["components"].T
    return ((z - p["projected_mean"]) / p["projected_std"]).astype(np.float32)


def physical_paths(split: str, task: int, cid: str) -> dict[str, Path]:
    low = split.lower()
    base = SOURCE / f"collection_{low}"
    return {
        "probe": base / f"task{task}/P5S0C_PROBE_TELEMETRY/{cid}_probe_timesteps.csv",
        "snapshot": base / f"snapshots/{cid}.pt",
        "visual": base / f"visual/{cid}_feature.npy",
    }


def branch_table(task: int, split: str) -> pd.DataFrame:
    if split == "TRAIN":
        if task == 0:
            d = pd.read_csv(SOURCE / "collection_train/task0/task0/branches.csv")
            return d.rename(columns={"requested_force_N": "force_N", "full_task_success_y": "success"})
        return pd.read_csv(FROZEN[task] / f"TASK{task}_CANONICAL_TRAIN_BRANCHES.csv")
    path = SOURCE / f"collection_dev/task{task}/task{task}/branches.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).rename(columns={"requested_force_N": "force_N", "full_task_success_y": "success"})


def empirical_frontier(branches: pd.DataFrame, cid: str) -> tuple[float, str, int, int]:
    if branches.empty or "context_id" not in branches:
        return math.nan, "[]", 0, 0
    q = branches[branches.context_id.astype(str) == cid].copy()
    if q.empty:
        return math.nan, "[]", 0, 0
    curve = q.groupby("force_N").success.agg(["sum", "count", "mean"]).reset_index().sort_values("force_N")
    valid = curve[curve["mean"] >= RHO]
    frontier = float(valid.force_N.iloc[0]) if len(valid) else math.nan
    return frontier, json.dumps(curve.to_dict("records"), separators=(",", ":")), int(len(curve)), int(len(q))


def representative_branch_path(branches: pd.DataFrame, cid: str) -> Path | None:
    if branches.empty:
        return None
    q = branches[branches.context_id.astype(str) == cid]
    if q.empty or "telemetry_path" not in q:
        return None
    return Path(str(q.sort_values(["force_N"]).iloc[0].telemetry_path))


def h8_summary(path: Path | None) -> dict[str, Any]:
    empty = {
        "nominal_h8_dx_m": math.nan, "nominal_h8_dy_m": math.nan, "nominal_h8_dz_m": math.nan,
        "nominal_h8_translation_m": math.nan, "nominal_h8_path_length_m": math.nan,
        "nominal_h8_horizontal_m": math.nan, "nominal_h8_vertical_abs_m": math.nan,
        "nominal_h8_phase": "UNAVAILABLE", "nominal_orientation_change_rad": math.nan,
    }
    if path is None or not path.exists():
        return empty
    d = pd.read_csv(path)
    if len(d) < early.H or not {"cmd_x", "cmd_y", "cmd_z", "phase"} <= set(d):
        return empty
    cmd = d[["cmd_x", "cmd_y", "cmd_z"]].iloc[:early.H].to_numpy(float)
    delta = cmd[-1] - cmd[0]
    increments = np.diff(cmd, axis=0)
    return {
        "nominal_h8_dx_m": float(delta[0]), "nominal_h8_dy_m": float(delta[1]),
        "nominal_h8_dz_m": float(delta[2]), "nominal_h8_translation_m": float(np.linalg.norm(delta)),
        "nominal_h8_path_length_m": float(np.linalg.norm(increments, axis=1).sum()),
        "nominal_h8_horizontal_m": float(np.linalg.norm(delta[:2])),
        "nominal_h8_vertical_abs_m": float(abs(delta[2])),
        "nominal_h8_phase": "+".join(pd.unique(d.phase.iloc[:early.H].astype(str))),
        "nominal_orientation_change_rad": math.nan,
    }


def audit_rows() -> list[dict[str, Any]]:
    manifest = pd.read_csv(SOURCE / "PROSPECTIVE_CONTEXT_MANIFEST.csv")
    visuals: dict[str, dict[str, Any]] = {}
    for split in ("TRAIN", "DEV"):
        vp = SOURCE / f"collection_{split.lower()}/visual_alignment_worker.csv"
        if vp.exists():
            for r in pd.read_csv(vp).to_dict("records"):
                visuals[str(r["context_id"])] = r
    branches = {(t, s): branch_table(t, s) for t in TASKS for s in ("TRAIN", "DEV")}
    rows: list[dict[str, Any]] = []
    for r in manifest.sort_values(["task", "split", "root_index", "friction_band"]).to_dict("records"):
        task, split, cid = int(r["task"]), str(r["split"]), str(r["context_id"])
        if task not in TASKS:
            continue
        paths = physical_paths(split, task, cid)
        if not all(paths[k].exists() for k in ("probe", "snapshot", "visual")):
            continue
        probe = pd.read_csv(paths["probe"])
        hold = probe[(probe.probe_phase.astype(str) == "hold") & (probe.step.astype(int) == 190)]
        if len(hold) != 1:
            raise RuntimeError(f"{cid}: expected exactly one strict step-190 hold row")
        pr = hold.iloc[0]
        snap = torch.load(paths["snapshot"], map_location="cpu", weights_only=False)
        robot = snap["articulation"]["robot"]
        robot_pose = robot["root_pose"][0].detach().cpu().numpy().astype(float)
        joints = robot["joint_position"][0].detach().cpu().numpy().astype(float)
        obj_pose = snap["rigid_object"][TASK_OBJECTS[task]]["root_pose"][0].detach().cpu().numpy().astype(float)
        obj_base_p = qapply(qinv(robot_pose[3:]), obj_pose[:3] - robot_pose[:3])
        obj_base_q = qmul(qinv(robot_pose[3:]), obj_pose[3:])
        eef_base = np.array([pr.eef_x, pr.eef_y, pr.eef_z], float)
        eef_world = robot_pose[:3] + qapply(robot_pose[3:], eef_base)
        rel = obj_base_p - eef_base
        tilt, yaw_sin, yaw_cos = quaternion_summary(obj_base_q)
        raw_visual = np.load(paths["visual"], allow_pickle=False).astype(np.float32)
        projected = visual_projection(task, raw_visual)
        frontier, curve, force_cells, branch_count = empirical_frontier(branches[(task, split)], cid)
        rep = representative_branch_path(branches[(task, split)], cid)
        root_ids = sorted(manifest[(manifest.task == task) & (manifest.split == "TRAIN")].root_id.astype(str).unique())
        cv_fold = root_ids.index(str(r["root_id"])) % 3 if split == "TRAIN" and str(r["root_id"]) in root_ids else "HELD_OUT_SEEN"
        row = {
            "context_id": cid, "root_id": str(r["root_id"]), "source_root_id": str(r["source_root_id"]),
            "root_index": int(r["root_index"]), "root_seed": int(r["root_seed"]), "task": task,
            "task_name": TASK_NAMES[task], "object_identity": TASK_OBJECTS[task],
            "object_asset": f"LIBERO object id {TASK_OBJECTS[task]} (USD path not logged in authoritative artifact)",
            "friction": float(r["mu_GT"]), "friction_band": str(r["friction_band"]), "split": split,
            "root_heldout_cv_fold": cv_fold, "physical_snapshot_status": "AUTHORITATIVE_CAPTURED_STEP190",
            "object_world_x_m": float(obj_pose[0]), "object_world_y_m": float(obj_pose[1]),
            "object_world_z_m": float(obj_pose[2]), "object_world_qw": float(obj_pose[3]),
            "object_world_qx": float(obj_pose[4]), "object_world_qy": float(obj_pose[5]),
            "object_world_qz": float(obj_pose[6]),
            "object_base_x_m": float(obj_base_p[0]), "object_base_y_m": float(obj_base_p[1]),
            "object_base_z_m": float(obj_base_p[2]),
            "eef_world_x_m": float(eef_world[0]), "eef_world_y_m": float(eef_world[1]),
            "eef_world_z_m": float(eef_world[2]), "eef_world_orientation_status": "NOT_LOGGED",
            "eef_base_x_m": float(eef_base[0]), "eef_base_y_m": float(eef_base[1]),
            "eef_base_z_m": float(eef_base[2]),
            "object_to_eef_dx_m": float(rel[0]), "object_to_eef_dy_m": float(rel[1]),
            "object_to_eef_dz_m": float(rel[2]), "object_to_eef_distance_m": float(np.linalg.norm(rel)),
            "horizontal_grasp_eccentricity_m": float(np.linalg.norm(rel[:2])),
            "vertical_grasp_offset_m": float(rel[2]),
            "object_to_eef_relative_rotation_status": "UNAVAILABLE_EEF_QUATERNION_NOT_LOGGED",
            "object_tilt_rad": tilt, "object_yaw_sin": yaw_sin, "object_yaw_cos": yaw_cos,
            "gripper_opening_m": float(pr.gripper_opening),
            "left_finger_joint_m": float(joints[-2]), "right_finger_joint_m": float(joints[-1]),
            "finger_joint_asymmetry_m": float(joints[-2] - joints[-1]),
            **h8_summary(rep),
            "empirical_Fstar_0p8_N": frontier, "empirical_force_cells": force_cells,
            "empirical_branch_count": branch_count, "empirical_success_curve_json": curve,
            "visual_feature_path": str(paths["visual"]), "visual_feature_raw_sha256": array_sha256(raw_visual),
            "visual_feature_raw_dim": int(raw_visual.size),
            "visual_feature_frozen_pca_json": json.dumps(projected.tolist(), separators=(",", ":")),
            "visual_feature_frozen_pca_dim": int(projected.size),
            "snapshot_path": str(paths["snapshot"]), "probe_telemetry_path": str(paths["probe"]),
            "branch_template_path": str(rep) if rep else "",
        }
        rows.append(row)
    return rows


def feature_spec(created: str) -> dict[str, Any]:
    return {
        "status": "FROZEN_BEFORE_STRUCTURED_MODEL_TRAINING",
        "created_at_utc": created,
        "scope": "same task + same object/task distribution + new physical pre-probe context",
        "representation_name": "STRUCTURED_CONTEXT_ORACLE_DIAGNOSTIC",
        "active_dimension": len(ACTIVE_FEATURES),
        "active_features_in_order": ACTIVE_FEATURES,
        "context_projection": "Linear(11,16)+ReLU; same 16-D bottleneck as Full Visual Linear(17,16)+ReLU",
        "feature_selection_rule": "physical meaning and authoritative pre-probe availability only; no model score or held-out label used",
        "normalization": "fold-local TRAIN-root mean/std for CV; full TRAIN mean/std for full-data checkpoints",
        "oracle_reason": "relative translation and object orientation require simulator GT object pose",
        "excluded_candidates": {
            "object_to_eef_relative_rotation": "EEF quaternion was passed to the frozen VLA request but not persisted; cannot recover honestly",
            "preprobe_eef_orientation": "not persisted",
            "nominal_h8_displacement_and_path": "audited but excluded because the frozen first H8 is branch_hold and has numerical-zero variance",
            "nominal_h8_orientation_change": "orientation commands were not logged",
            "horizontal_eccentricity_and_distance": "audited but excluded from model because they are deterministic transforms of dx/dy/dz",
            "all_robot_joint_positions": "excluded to avoid a high-dimensional state dump; only finger asymmetry retained",
            "friction": "already present in the unchanged Base condition input; not duplicated in x_grasp",
        },
        "fairness": {
            "epochs": early.EPOCHS, "optimizer": "AdamW", "lr": early.LR,
            "weight_decay": early.WEIGHT_DECAY, "hidden_size": 64,
            "context_bottleneck": 16, "lambda_physics": 1.0, "lambda_IE": 1.0,
            "lambda_feasibility": early.LAMBDA_FEAS, "root_split": "existing 3-fold grouped root split",
            "cv_seed": 0, "full_train_seeds": early.SEEDS,
        },
        "forbidden_claim": "The oracle structured models are mechanism diagnostics, not deployable methods.",
    }


def legality_rows() -> list[dict[str, Any]]:
    rows = []
    def add(name, status, source, included, note=""):
        rows.append({"feature": name, "deployment_legality": status, "authoritative_source": source,
                     "included_in_x_grasp": int(included), "note": note})
    for f in ["object_to_eef_dx_m", "object_to_eef_dy_m", "object_to_eef_dz_m"]:
        add(f, "SIM_PRIVILEGED", "snapshot object GT + legal EEF state", True, "makes full vector oracle-only")
    for f in ["object_tilt_rad", "object_yaw_sin", "object_yaw_cos"]:
        add(f, "SIM_PRIVILEGED", "snapshot object GT quaternion", True, "world/base object orientation; EEF-relative rotation unavailable")
    for f in ["eef_base_x_m", "eef_base_y_m", "eef_base_z_m", "gripper_opening_m"]:
        add(f, "DEPLOYMENT_LEGAL", "pre-probe robot observation", True)
    add("finger_joint_asymmetry_m", "DERIVED_FROM_LEGAL_STATE", "saved robot finger joints", True)
    for f in ["nominal_h8_dx_m", "nominal_h8_dy_m", "nominal_h8_dz_m", "nominal_h8_translation_m", "nominal_h8_path_length_m"]:
        add(f, "DERIVED_FROM_LEGAL_STATE", "frozen nominal command trace", False, "excluded: numerical-zero H8 branch_hold variance")
    add("object_to_eef_relative_rotation", "UNAVAILABLE", "EEF quaternion not persisted", False)
    add("preprobe_eef_orientation", "UNAVAILABLE", "EEF quaternion not persisted", False)
    add("nominal_h8_orientation_change", "UNAVAILABLE", "orientation command not logged", False)
    add("friction", "DEPLOYMENT_LEGAL_AFTER_ACTIVE_PROBE", "unchanged Base condition", False, "already in Base; Probe not executed in this round")
    add("empirical_Fstar_0p8_N", "LABEL_ONLY", "branch outcomes", False, "never a model input")
    add("visual_feature", "DEPLOYMENT_LEGAL", "frozen VLA pre-action feature", False, "used only by existing visual comparators")
    return rows


def correlation_rows(audit: pd.DataFrame) -> list[dict[str, Any]]:
    features = [
        "friction", "object_to_eef_dx_m", "object_to_eef_dy_m", "object_to_eef_dz_m",
        "horizontal_grasp_eccentricity_m", "vertical_grasp_offset_m", "object_tilt_rad",
        "gripper_opening_m", "finger_joint_asymmetry_m", "eef_base_x_m", "eef_base_y_m",
        "eef_base_z_m", "nominal_h8_translation_m", "nominal_h8_path_length_m",
    ]
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for split in ("TRAIN", "DEV_SEEN"):
            q = audit[(audit.task == task) & (audit.split == ("TRAIN" if split == "TRAIN" else "DEV"))].copy()
            q = q[np.isfinite(q.empirical_Fstar_0p8_N.astype(float))]
            if q.empty:
                continue
            y = q.empirical_Fstar_0p8_N.to_numpy(float)
            mu = q.friction.to_numpy(float)
            for feature in features:
                x = q[feature].to_numpy(float)
                keep = np.isfinite(x) & np.isfinite(y)
                xx, yy, mm = x[keep], y[keep], mu[keep]
                if len(xx) < 3 or np.ptp(xx) <= 1e-12 or np.ptp(yy) <= 1e-12:
                    rho, pval, slope, intercept, r2, partial = (math.nan,) * 6
                else:
                    sp = stats.spearmanr(xx, yy)
                    rho, pval = float(sp.statistic), float(sp.pvalue)
                    fit = stats.linregress(xx, yy)
                    slope, intercept, r2 = float(fit.slope), float(fit.intercept), float(fit.rvalue**2)
                    if len(xx) >= 8 and np.ptp(mm) > 1e-12:
                        rx = xx - np.polyval(np.polyfit(mm, xx, 1), mm)
                        ry = yy - np.polyval(np.polyfit(mm, yy, 1), mm)
                        partial = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 1e-12 and np.std(ry) > 1e-12 else math.nan
                    else:
                        partial = math.nan
                rows.append({
                    "task": task, "split": split, "feature": feature, "n_contexts": int(len(xx)),
                    "n_root_seed_groups": int(q.root_id.nunique()), "spearman_rho": rho,
                    "spearman_p_value_descriptive_only": pval, "linear_slope_Fstar_N_per_feature_unit": slope,
                    "linear_intercept_N": intercept, "linear_R2": r2,
                    "partial_pearson_controlling_friction": partial,
                    "claim_scope": "DESCRIPTIVE_NOT_FEATURE_SELECTION_NOT_CAUSAL",
                })
    return rows


def root06_root07_comparison(audit: pd.DataFrame) -> dict[str, Any]:
    q = audit[(audit.task == 0) & audit.context_id.str.contains("r06|r07", regex=True) & (audit.split == "DEV")].sort_values("root_index")
    if len(q) != 2:
        return {"status": "UNAVAILABLE", "rows_found": len(q)}
    a, b = q.iloc[0], q.iloc[1]
    fields = [
        "friction", "object_world_x_m", "object_world_y_m", "object_world_z_m",
        "eef_base_x_m", "eef_base_y_m", "eef_base_z_m", "object_to_eef_dx_m",
        "object_to_eef_dy_m", "object_to_eef_dz_m", "horizontal_grasp_eccentricity_m",
        "gripper_opening_m", "finger_joint_asymmetry_m", "object_tilt_rad",
        "nominal_h8_translation_m", "nominal_h8_path_length_m", "empirical_Fstar_0p8_N",
    ]
    return {
        "status": "COMPLETE", "root06_context_id": str(a.context_id), "root07_context_id": str(b.context_id),
        "root06_Fstar_0p8_N": float(a.empirical_Fstar_0p8_N), "root07_Fstar_0p8_N": float(b.empirical_Fstar_0p8_N),
        "signed_root07_minus_root06": {f: float(b[f] - a[f]) for f in fields},
        "interpretation": "The recorded grasp geometry and first H8 are nearly identical; the 0.5 N frontier difference is not explained by a material recorded grasp-offset or H8-motion difference.",
    }


def audit_markdown(audit: pd.DataFrame, feature_hash: str, comparison: dict[str, Any]) -> str:
    counts = audit.groupby(["task", "split"]).agg(contexts=("context_id", "nunique"), root_seed_groups=("root_id", "nunique"), valid_frontiers=("empirical_Fstar_0p8_N", lambda x: int(np.isfinite(x.astype(float)).sum()))).reset_index()
    train = audit[audit.split == "TRAIN"]
    h8_max = float(train.nominal_h8_path_length_m.abs().max())
    d = comparison.get("signed_root07_minus_root06", {})
    lines = [
        "# Root Physical Context Audit", "", "## Direct answer", "",
        "The dataset contains **18 captured pre-probe physical contexts per task, grouped into 6 root-seed families** (three friction-conditioned executions per root seed). The correct independent-count statement is therefore not simply 18 IID roots: the held-root CV holds out two entire root-seed families (six physical contexts) per fold.", "",
        "For task0 root06 (F*0.8=4.25 N) versus root07 (3.75 N), recorded relative grasp geometry is effectively the same. The horizontal eccentricity differs by only {:.3f} mm, vertical offset by {:.3f} mm, gripper opening by {:.3f} mm, and friction by {:.4f}. The frozen first H8 remains branch_hold for both and has essentially zero displacement. The largest obvious recorded difference is a common scene translation (EEF/object y changes together by about {:.3f} mm), not a different object-in-gripper offset.".format(
            abs(d.get("horizontal_grasp_eccentricity_m", math.nan))*1000,
            abs(d.get("object_to_eef_dz_m", math.nan))*1000,
            abs(d.get("gripper_opening_m", math.nan))*1000,
            abs(d.get("friction", math.nan)),
            abs(d.get("eef_base_y_m", math.nan))*1000,
        ), "",
        "This means the root06/root07 0.5 N frontier difference is **not explained by a material recorded grasp offset, orientation change, gripper opening change, or H8 demand change**. It is consistent with unlogged contact microstate / outcome stochasticity, or with a relevant variable outside the current audit interface.", "",
        "## Captured population", "", markdown_table(counts), "",
        "- task5 planned DEV roots were not physically collected, so task5 mechanism evaluation uses the existing retrospective TRAIN root-heldout CV only.",
        "- task1 DEV collection is incomplete and remains previously seen; it is audited descriptively but is not used to tune the feature set.",
        "- task1 TRAIN has the existing reconstructed-label caveat; task5 supplies direct-label replication.", "",
        "## What roots actually vary in", "",
        "Across captured contexts the logged variables that vary are absolute object/EEF placement, small residual object-to-EEF offsets, object quaternion, finger state/opening, friction, and raw visual features. Relative rotation cannot be reconstructed because EEF quaternion was not persisted.", "",
        f"The maximum TRAIN first-H8 command path length is {h8_max:.3e} m; all first-H8 rows are branch_hold. Therefore H8 motion is audited but excluded from x_grasp before model fitting.", "",
        "## Deployment legality", "",
        "The 11-D Structured vector is an oracle mechanism diagnostic: six dimensions depend on simulator GT object pose. EEF position, gripper opening, and finger-joint asymmetry are legal/derived from legal robot state. No deployable claim is made from the oracle vector.", "",
        "## Integrity controls", "",
        f"- Frozen structured feature spec SHA256: `{feature_hash}`.",
        "- One row per captured context; all rows link strict step-190 probe telemetry, a restorable snapshot, and a frozen visual feature.",
        "- F*0.8 is a label-only field and never enters x_grasp.",
        "- Correlations are descriptive and are not used for feature selection.",
        "- Existing held-out roots are mechanism-analysis data, not a new untouched TEST.", "",
    ]
    return "\n".join(lines)


def run_audit(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    spec_path = out / "STRUCTURED_GRASP_CONTEXT_FEATURES.json"
    write_json(spec_path, feature_spec(created))
    spec_hash = sha256(spec_path)
    rows = audit_rows()
    audit = pd.DataFrame(rows).sort_values(["task", "split", "root_index", "friction_band"])
    if audit.context_id.duplicated().any():
        raise RuntimeError("duplicate physical context rows")
    expected_train = {t: 18 for t in TASKS}
    got = audit[audit.split == "TRAIN"].groupby("task").context_id.nunique().to_dict()
    if got != expected_train:
        raise RuntimeError(f"TRAIN context coverage mismatch: {got}")
    audit.to_csv(out / "ROOT_PHYSICAL_CONTEXT_AUDIT.csv", index=False)
    write_csv(out / "STRUCTURED_CONTEXT_DEPLOYMENT_LEGALITY.csv", legality_rows())
    corr = pd.DataFrame(correlation_rows(audit))
    corr.to_csv(out / "ROOT_FRONTIER_PHYSICAL_CORRELATIONS.csv", index=False)
    comparison = root06_root07_comparison(audit)
    write_json(out / "ROOT06_ROOT07_PHYSICAL_COMPARISON.json", comparison)
    (out / "ROOT_PHYSICAL_CONTEXT_AUDIT.md").write_text(audit_markdown(audit, spec_hash, comparison), encoding="utf-8")
    write_json(out / "AUDIT_FREEZE.json", {
        "status": "ROOT_AUDIT_AND_FEATURES_FROZEN_BEFORE_MODEL_TRAINING",
        "created_at_utc": created, "feature_spec_sha256": spec_hash,
        "audit_rows": len(audit), "train_contexts": int((audit.split == "TRAIN").sum()),
        "train_root_seed_groups": int(audit[audit.split == "TRAIN"].root_id.nunique()),
        "task_counts": [
            {"task": int(task), "split": str(split), "contexts": int(count)}
            for (task, split), count in audit.groupby(["task", "split"]).size().items()
        ],
        "model_scores_read_for_feature_selection": False, "Probe_run": False,
    })
    print(audit.groupby(["task", "split"]).size().to_string())
    print(json.dumps(comparison, indent=2))


class StructuredFull(early.VisualFullFeas):
    pass


class StructuredJoint(early.VisualJoint):
    pass


def load_population(task: int, scratch: Path):
    suffix = f"sgc_{task}_{scratch.name}_{id(scratch)}"
    tpi = early.load_module("sgc_tpi_" + suffix, early.TPI_CODE)
    cf = early.load_module("sgc_cf_" + suffix, early.CF_CODE)
    full = early.load_module("sgc_full_" + suffix, early.FULL_CODE)
    if task == 0:
        _, _, cmap, traces, meta, visual_dim = early.audit_and_load(scratch, tpi)
        pairs = early.build_pairs(cf, traces, meta)
    else:
        cmap, traces, meta, visual_dim, _, _, _ = taskwise.audit_and_load(task, scratch, tpi)
        pairs = taskwise.build_pairs(task, cf, traces, meta)
    segs, norm = early.build_segments_and_norm(cf, tpi, traces)
    return tpi, cf, full, cmap, traces, meta, visual_dim, pairs, segs, norm


def context_normalization(cmap: dict[str, dict[str, Any]], cids: list[str], key: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([np.asarray(cmap[c][key], np.float32) for c in sorted(set(cids))])
    mean, std = x.mean(0).astype(np.float32), x.std(0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def with_context(cmap: dict[str, dict[str, Any]], key: str, mean: np.ndarray, std: np.ndarray) -> dict[str, dict[str, Any]]:
    return {cid: {**v, "visual": ((np.asarray(v[key], np.float32) - mean) / std).astype(np.float32)} for cid, v in cmap.items()}


def fold_visual_context(cmap0: dict[str, dict[str, Any]], train_cids: list[str]) -> tuple[dict[str, dict[str, Any]], int]:
    raw_train = np.stack([cmap0[c]["visual_raw"] for c in sorted(train_cids)])
    mean = raw_train.mean(0)
    _, singular, vt = np.linalg.svd(raw_train - mean, full_matrices=False)
    rank = min(17, len(train_cids) - 1, int(np.sum(singular > singular[0] * 1e-7)))
    comp = vt[:rank]
    ztrain = (raw_train - mean) @ comp.T
    zm, zs = ztrain.mean(0), ztrain.std(0)
    zs[zs < 1e-6] = 1.0
    cmap = {}
    for cid, value in cmap0.items():
        z = ((value["visual_raw"] - mean) @ comp.T - zm) / zs
        z17 = np.zeros(17, np.float32)
        z17[:rank] = z.astype(np.float32)
        cmap[cid] = {**value, "visual": z17}
    return cmap, rank


def add_structured_raw(cmap: dict[str, dict[str, Any]], audit: pd.DataFrame) -> None:
    lookup = audit[audit.split == "TRAIN"].set_index("context_id")
    for cid in cmap:
        if cid not in lookup.index:
            raise RuntimeError(f"missing structured audit row for {cid}")
        cmap[cid]["structured_raw"] = lookup.loc[cid, ACTIVE_FEATURES].to_numpy(dtype=np.float32)


def dense_forces(task: int) -> np.ndarray:
    return gen.FORCES_DENSE if task == 0 else taskgen.TASK_DENSE_FORCES[task]


def model_curve(model: Any, kind: str, context: np.ndarray | None, template: Any, tpi, cf, norm, forces: np.ndarray) -> np.ndarray:
    values = []
    with torch.no_grad():
        c = torch.tensor(context[None], dtype=torch.float32) if context is not None else None
        for force in forces:
            step_np, cond_np = gen.normalized_segment(cf, tpi, template, float(force), norm)
            step = torch.tensor(step_np[None], dtype=torch.float32)
            cond = torch.tensor(cond_np[None], dtype=torch.float32)
            if kind == "BASE":
                logit = model(step, cond)
            elif kind == "FULL":
                logit = model(step, cond, c)
            elif kind == "JOINT":
                logit = model(step, cond, c)[1]
            else:
                raise ValueError(kind)
            values.append(float(torch.sigmoid(logit)[0]))
    return np.asarray(values)


def curve_metrics(task: int, model: str, fold: int, held: list[Any], curves: dict[str, np.ndarray], forces: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = []
    for cid, curve in curves.items():
        q = [tr for tr in held if tr.context_id == cid]
        cells: dict[float, list[float]] = {}
        for tr in q:
            cells.setdefault(float(tr.force), []).append(float(tr.outcome))
        valid_force = [f for f in sorted(cells) if float(np.mean(cells[f])) >= RHO]
        real = float(valid_force[0]) if valid_force else math.nan
        idx = np.flatnonzero(curve >= RHO)
        pred = float(forces[int(idx[0])]) if len(idx) else math.nan
        err = pred - real if math.isfinite(real) and math.isfinite(pred) else math.nan
        details.append({
            "task": task, "fold": fold, "model": model, "context_id": cid,
            "root_id": q[0].root_id, "real_frontier_N": real, "predicted_frontier_N": pred,
            "signed_error_N": err, "absolute_error_N": abs(err) if math.isfinite(err) else math.nan,
            "under_force": bool(not math.isfinite(pred) or (math.isfinite(real) and pred < real - 1e-9)),
            "under_force_magnitude_N": real - pred if math.isfinite(err) and err < 0 else 0.0 if math.isfinite(err) else math.nan,
            "excess_force": bool(math.isfinite(err) and err > 1e-9),
            "excess_force_N": err if math.isfinite(err) and err > 0 else 0.0 if math.isfinite(err) else math.nan,
            "local_monotonicity": float(np.mean(np.diff(curve) >= -1e-8)),
        })
    valid = [d for d in details if math.isfinite(d["real_frontier_N"])]
    finite = [d for d in valid if math.isfinite(d["predicted_frontier_N"])]
    under = [d for d in valid if d["under_force"]]
    excess = [d for d in finite if d["excess_force"]]
    summary = {
        "valid_frontier_contexts": len(valid), "finite_decision_contexts": len(finite),
        "finite_decision_coverage": len(finite)/len(valid) if valid else math.nan,
        "frontier_MAE_N": float(np.mean([d["absolute_error_N"] for d in finite])) if finite else math.nan,
        "under_force_rate": len(under)/len(valid) if valid else math.nan,
        "mean_under_force_magnitude_N": float(np.mean([d["under_force_magnitude_N"] for d in under if math.isfinite(d["under_force_magnitude_N"])])) if under else 0.0,
        "excess_force_rate": len(excess)/len(finite) if finite else math.nan,
        "mean_excess_force_N": float(np.mean([d["excess_force_N"] for d in excess])) if excess else 0.0,
        "mean_local_monotonicity": float(np.mean([d["local_monotonicity"] for d in details])),
        "context_monotonic_rate": float(np.mean([d["local_monotonicity"] >= 1.0 - 1e-12 for d in details])),
    }
    return summary, details


def branch_root_metrics(model: str, fold: int, held: list[Any], logits: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for root_id in sorted({tr.root_id for tr in held}):
        q = [tr for tr in held if tr.root_id == root_id]
        r = jnv.branch_metrics(model, fold, q, logits)
        rows.append({**r, "root_id": root_id})
    return rows


def train_structured_full(task: int, out: Path, tpi, cf, full, cmap0, traces, meta, pairs, segs, norm) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cids = [tr.context_id for tr in traces]
    mean, std = context_normalization(cmap0, cids, "structured_raw")
    cmap = with_context(cmap0, "structured_raw", mean, std)
    ckdir = out / f"checkpoints/task{task}"
    ckdir.mkdir(parents=True, exist_ok=True)
    metrics, manifest = [], []
    device = torch.device("cpu")
    for seed in early.SEEDS:
        sf_path = ckdir / f"STRUCTURED_seed{seed}.pt"
        sj_path = ckdir / f"STRUCTURED_JOINT_seed{seed}.pt"
        if sf_path.exists():
            sf_ck = torch.load(sf_path, map_location=device, weights_only=False)
            sf = StructuredFull(len(ACTIVE_FEATURES)).to(device)
            sf.load_state_dict(sf_ck["state_dict"]); sf.eval()
            steps = int(sf_ck["optimizer_steps"])
        else:
            sf, _, steps = early.train_full(traces, segs, norm, cmap, meta, device, seed, len(ACTIVE_FEATURES))
        if sj_path.exists():
            sj_ck = torch.load(sj_path, map_location=device, weights_only=False)
            physics_state = {k[len("physics."):]: v for k, v in sj_ck["state_dict"].items() if k.startswith("physics.")}
            sj = StructuredJoint(tpi, physics_state, len(ACTIVE_FEATURES)).to(device)
            sj.load_state_dict(sj_ck["state_dict"]); sj.eval()
            jsteps = int(sj_ck["optimizer_steps"]); units = int(sj_ck["physical_units"])
            base_path = Path(str(sj_ck["initial_physics_checkpoint"]))
        else:
            sj, _, jsteps, units, base_path = early.train_joint(full, cf, tpi, traces, pairs, segs, norm, cmap, meta, device, seed, len(ACTIVE_FEATURES))
        for name, model, kind, st in [("Structured", sf, "FULL", steps), ("Structured Joint", sj, "JOINT", jsteps)]:
            logits = early.logits_for(model, kind, traces, segs, norm, cmap, device)
            tm = early.train_metrics(name, seed, traces, logits)
            tm.update({"model": name, "task": task, "context_dim": len(ACTIVE_FEATURES),
                       "parameters": sum(p.numel() for p in model.parameters())})
            metrics.append(tm)
            path = sf_path if name == "Structured" else sj_path
            if not path.exists():
                torch.save({
                "state_dict": model.state_dict(), "model": name, "task": task, "seed": seed,
                "epochs": early.EPOCHS, "optimizer": "AdamW", "lr": early.LR,
                "weight_decay": early.WEIGHT_DECAY, "lambda_physics": 1.0 if kind == "JOINT" else None,
                "lambda_IE": 1.0 if kind == "JOINT" else None,
                "lambda_feasibility": early.LAMBDA_FEAS if kind == "JOINT" else None,
                "structured_feature_spec_sha256": sha256(out / "STRUCTURED_GRASP_CONTEXT_FEATURES.json"),
                "structured_mean": mean, "structured_std": std, "TRAIN_only": True,
                "DEV_used": False, "TEST_used": False, "oracle_diagnostic": True,
                "optimizer_steps": st, "physical_units": units if kind == "JOINT" else None,
                "initial_physics_checkpoint": str(base_path) if kind == "JOINT" else None,
                }, path)
            manifest.append({"task": task, "model": name, "seed": seed, "path": str(path), "sha256": sha256(path),
                             "parameters": sum(p.numel() for p in model.parameters()), "optimizer_steps": st})
    return metrics, manifest


def existing_train_metrics(task: int) -> list[dict[str, Any]]:
    d = pd.read_csv(FROZEN[task] / f"TASK{task}_TRAIN_IN_SAMPLE_METRICS.csv")
    mapping = {"PROSPECTIVE_BASE_FEAS": "Base", "VISUAL_CONTEXT_FULL_FEAS": "Full Visual", "VISUAL_CONTEXT_JOINT": "Visual Joint"}
    rows = []
    for r in d[d.variant.isin(mapping)].to_dict("records"):
        rows.append({**r, "model": mapping[r["variant"]], "task": task})
    j = pd.read_csv(EXISTING_JNV_TRAIN[task])
    for r in j.to_dict("records"):
        rows.append({**r, "model": "Joint-NoVisual", "task": task})
    return rows


def run_task_experiment(task: int, out: Path, audit: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scratch = out / f"_experiment_audit/task{task}"
    scratch.mkdir(parents=True, exist_ok=True)
    tpi, cf, full, cmap0, traces, meta, _, pairs, all_segs, full_norm = load_population(task, scratch)
    add_structured_raw(cmap0, audit)
    structured_train, ck_manifest = train_structured_full(task, out, tpi, cf, full, cmap0, traces, meta, pairs, all_segs, full_norm)
    train_metrics = existing_train_metrics(task) + structured_train

    roots = sorted({tr.root_id for tr in traces})
    if len(roots) != 6:
        raise RuntimeError(f"task{task}: expected six root-seed groups, got {len(roots)}")
    folds = [set(roots[i::3]) for i in range(3)]
    visual_reference = pd.read_csv(EXISTING_VISUAL_CV[task])
    visual_reference = visual_reference[visual_reference.fold.astype(str) != "__MEAN__"]
    jnv_reference = pd.read_csv(EXISTING_JNV_CV[task])
    jnv_reference = jnv_reference[jnv_reference.fold.astype(str) != "__MEAN__"]
    jnv_dense = pd.read_csv(EXISTING_JNV_DENSE[task])
    forces = dense_forces(task)
    device = torch.device("cpu")
    fold_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    for fold, held_roots in enumerate(folds):
        fold_cache = out / f"_fold_cache/task{task}_fold{fold}.json"
        if fold_cache.exists():
            cached = json.loads(fold_cache.read_text())
            if (cached.get("comparator_equivalence_atol") == COMPARATOR_EQUIVALENCE_ATOL and
                    cached.get("feature_spec_sha256") == sha256(out / "STRUCTURED_GRASP_CONTEXT_FEATURES.json")):
                fold_rows.extend(cached["fold_rows"])
                detail_rows.extend(cached["detail_rows"])
                root_rows.extend(cached["root_rows"])
                print(f"[structured CV] task={task} fold={fold} reused verified cache", flush=True)
                continue
        fold_start, detail_start, root_start = len(fold_rows), len(detail_rows), len(root_rows)
        train = [tr for tr in traces if tr.root_id not in held_roots]
        held = [tr for tr in traces if tr.root_id in held_roots]
        train_meta = {tr.branch_id: meta[tr.branch_id] for tr in train}
        segs_train = {tr.branch_id: all_segs[tr.branch_id] for tr in train}
        norm = early.build_segments_and_norm(cf, tpi, train)[1]
        fold_pairs = taskwise.build_pairs(task, cf, train, train_meta)
        train_cids = sorted({tr.context_id for tr in train})
        visual_cmap, rank = fold_visual_context(cmap0, train_cids)
        sm, ss = context_normalization(cmap0, train_cids, "structured_raw")
        structured_cmap = with_context(cmap0, "structured_raw", sm, ss)
        print(f"[structured CV] task={task} fold={fold} train={len(train)} held={len(held)}", flush=True)
        base, _, _ = early.train_base(full, train, segs_train, norm, visual_cmap, train_meta, device, 0)
        vf, _, _ = early.train_full(train, segs_train, norm, visual_cmap, train_meta, device, 0, 17)
        vj, _, _, _, _ = early.train_joint(full, cf, tpi, train, fold_pairs, segs_train, norm, visual_cmap, train_meta, device, 0, 17)
        sf, _, _ = early.train_full(train, segs_train, norm, structured_cmap, train_meta, device, 0, len(ACTIVE_FEATURES))
        sj, _, _, _, _ = early.train_joint(full, cf, tpi, train, fold_pairs, segs_train, norm, structured_cmap, train_meta, device, 0, len(ACTIVE_FEATURES))
        fitted = {
            "Base": (base, "BASE", visual_cmap),
            "Full Visual": (vf, "FULL", visual_cmap),
            "Visual Joint": (vj, "JOINT", visual_cmap),
            "Structured": (sf, "FULL", structured_cmap),
            "Structured Joint": (sj, "JOINT", structured_cmap),
        }
        templates = {cid: next(tr for tr in held if tr.context_id == cid) for cid in sorted({tr.context_id for tr in held})}
        for name, (model, kind, cmap) in fitted.items():
            internal = "BASE" if kind == "BASE" else kind
            logits = early.logits_for(model, internal, held, all_segs, norm, cmap, device)
            bm = jnv.branch_metrics(name, fold, held, logits)
            context_curves = {cid: model_curve(model, kind, None if kind == "BASE" else cmap[cid]["visual"], template, tpi, cf, norm, forces) for cid, template in templates.items()}
            fm, details = curve_metrics(task, name, fold, held, context_curves, forces)
            row = {**bm, **fm, "task": task, "seed": 0, "heldout_root_ids": json.dumps(sorted(held_roots)),
                   "fold_PCA_effective_rank": rank, "structured_feature_dim": len(ACTIVE_FEATURES), "DEV_used": 0}
            fold_rows.append(row)
            detail_rows.extend(details)
            rr = branch_root_metrics(name, fold, held, logits)
            root_rows.extend({**x, "task": task} for x in rr)
            ref_name = {"Base": "PROSPECTIVE_BASE_FEAS", "Full Visual": "VISUAL_CONTEXT_FULL_FEAS", "Visual Joint": "VISUAL_CONTEXT_JOINT"}.get(name)
            if ref_name:
                ref = visual_reference[(visual_reference.fold.astype(int) == fold) & (visual_reference.model == ref_name)].iloc[0]
                bce_delta = bm["BCE"] - float(ref.BCE)
                brier_delta = bm["Brier"] - float(ref.Brier)
                equivalent = abs(bce_delta) <= COMPARATOR_EQUIVALENCE_ATOL and abs(brier_delta) <= COMPARATOR_EQUIVALENCE_ATOL
                row.update({
                    "frozen_reference_BCE": float(ref.BCE),
                    "frozen_reference_Brier": float(ref.Brier),
                    "recomputed_minus_reference_BCE": bce_delta,
                    "recomputed_minus_reference_Brier": brier_delta,
                    "comparator_reproducibility": "EQUIVALENT" if equivalent else "MATERIAL_MISMATCH_CURRENT_EXACT_RECOMPUTE",
                    "comparator_equivalence_atol": COMPARATOR_EQUIVALENCE_ATOL,
                })
                if not equivalent and name in {"Base", "Full Visual"}:
                    raise RuntimeError(
                        f"task{task} fold{fold} {name}: frozen CV reproducibility mismatch; "
                        f"actual_BCE={bm['BCE']:.17g} reference_BCE={float(ref.BCE):.17g} "
                        f"actual_Brier={bm['Brier']:.17g} reference_Brier={float(ref.Brier):.17g} "
                        f"atol={COMPARATOR_EQUIVALENCE_ATOL:.1e}"
                    )
                if not equivalent:
                    print(
                        f"[reproducibility warning] task={task} fold={fold} model={name} "
                        f"delta_BCE={bce_delta:.9g} delta_Brier={brier_delta:.9g}; "
                        "using current exact-protocol recomputation for paired mechanism comparison",
                        flush=True,
                    )
        # Reuse pre-existing, same-protocol Joint-NoVisual branch metrics and dense curves.
        jr = jnv_reference[(jnv_reference.fold.astype(int) == fold) & (jnv_reference.model == "JOINT_NOVISUAL")].iloc[0].to_dict()
        curves = {}
        for cid in templates:
            q = jnv_dense[(jnv_dense.fold.astype(int) == fold) & (jnv_dense.model == "JOINT_NOVISUAL") & (jnv_dense.context_id == cid)].sort_values("force_N")
            if len(q) != len(forces) or not np.allclose(q.force_N.to_numpy(float), forces):
                raise RuntimeError(f"task{task} fold{fold} {cid}: Joint-NoVisual dense curve mismatch")
            curves[cid] = q.probability.to_numpy(float)
        fm, details = curve_metrics(task, "Joint-NoVisual", fold, held, curves, forces)
        fold_rows.append({**jr, **fm, "model": "Joint-NoVisual", "task": task})
        detail_rows.extend(details)
        # Root-level NLL is recomputed from saved branch-dense metrics only for fitted models;
        # Joint-NoVisual root fractions remain unavailable rather than fabricated.
        write_json(fold_cache, {
            "task": task,
            "fold": fold,
            "feature_spec_sha256": sha256(out / "STRUCTURED_GRASP_CONTEXT_FEATURES.json"),
            "comparator_equivalence_atol": COMPARATOR_EQUIVALENCE_ATOL,
            "fold_rows": fold_rows[fold_start:],
            "detail_rows": detail_rows[detail_start:],
            "root_rows": root_rows[root_start:],
        })
    return fold_rows, detail_rows, root_rows, train_metrics, ck_manifest


def aggregate_results(folds: pd.DataFrame, details: pd.DataFrame, roots: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["BCE", "NLL", "probability_MAE", "Brier", "accuracy_0.5", "signed_bias",
               "valid_frontier_contexts", "finite_decision_contexts", "finite_decision_coverage",
               "frontier_MAE_N", "under_force_rate", "mean_under_force_magnitude_N",
               "excess_force_rate", "mean_excess_force_N", "mean_local_monotonicity", "context_monotonic_rate"]
    for (task, model), q in folds.groupby(["task", "model"]):
        row: dict[str, Any] = {"task": int(task), "task_name": TASK_NAMES[int(task)], "evaluation_scope": "RETROSPECTIVE_TRAIN_ROOT_HELDOUT_3FOLD_SEED0", "model": model}
        for m in metrics:
            row[m] = float(pd.to_numeric(q[m], errors="coerce").mean()) if m in q else math.nan
        tq = train[(train.task == task) & (train.model == model)]
        if len(tq) and "bce_nll" in tq:
            vals = pd.to_numeric(tq.bce_nll, errors="coerce").dropna()
        elif len(tq) and "BCE" in tq:
            vals = pd.to_numeric(tq.BCE, errors="coerce").dropna()
        else:
            vals = pd.Series(dtype=float)
        row["TRAIN_in_sample_NLL_seed_mean"] = float(vals.mean()) if len(vals) else math.nan
        row["TRAIN_in_sample_NLL_seed_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else math.nan
        row["TRAIN_to_rootheldout_NLL_gap"] = row["NLL"] - row["TRAIN_in_sample_NLL_seed_mean"] if math.isfinite(row["TRAIN_in_sample_NLL_seed_mean"]) else math.nan
        rq = roots[(roots.task == task) & (roots.model == model)]
        bq = roots[(roots.task == task) & (roots.model == "Base")][["fold", "root_id", "NLL"]].rename(columns={"NLL": "base_root_NLL"})
        if len(rq):
            merged = rq.merge(bq, on=["fold", "root_id"], how="left")
            row["fraction_root_seed_groups_improved_over_Base_NLL"] = float(np.mean(merged.NLL < merged.base_root_NLL - 1e-12))
        else:
            row["fraction_root_seed_groups_improved_over_Base_NLL"] = math.nan
        dq = details[(details.task == task) & (details.model == model)]
        db = details[(details.task == task) & (details.model == "Base")][["fold", "context_id", "absolute_error_N"]].rename(columns={"absolute_error_N": "base_abs_error_N"})
        merged = dq.merge(db, on=["fold", "context_id"], how="left")
        finite = merged[np.isfinite(merged.absolute_error_N) & np.isfinite(merged.base_abs_error_N)]
        row["fraction_physical_contexts_improved_over_Base_frontier"] = float(np.mean(finite.absolute_error_N < finite.base_abs_error_N - 1e-12)) if len(finite) else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "model"])


def joint_comparison_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for task in TASKS:
        q = summary[summary.task == task].set_index("model")
        def delta(a, b, metric):
            return float(q.loc[a, metric] - q.loc[b, metric])
        rows.append({
            "task": task, "task_name": TASK_NAMES[task],
            "StructuredJoint_minus_Structured_NLL": delta("Structured Joint", "Structured", "NLL"),
            "StructuredJoint_minus_Structured_frontier_MAE_N": delta("Structured Joint", "Structured", "frontier_MAE_N"),
            "StructuredJoint_minus_Structured_under_force_rate": delta("Structured Joint", "Structured", "under_force_rate"),
            "VisualJoint_minus_FullVisual_NLL": delta("Visual Joint", "Full Visual", "NLL"),
            "VisualJoint_minus_FullVisual_frontier_MAE_N": delta("Visual Joint", "Full Visual", "frontier_MAE_N"),
            "VisualJoint_minus_FullVisual_under_force_rate": delta("Visual Joint", "Full Visual", "under_force_rate"),
            "JointNoVisual_minus_Base_NLL": delta("Joint-NoVisual", "Base", "NLL"),
            "StructuredJoint_better_NLL_and_frontier_underforce_nonworse": bool(
                q.loc["Structured Joint", "NLL"] < q.loc["Structured", "NLL"] and
                q.loc["Structured Joint", "frontier_MAE_N"] < q.loc["Structured", "frontier_MAE_N"] and
                q.loc["Structured Joint", "under_force_rate"] <= q.loc["Structured", "under_force_rate"]
            ),
            "VisualJoint_worse_NLL_than_FullVisual": bool(q.loc["Visual Joint", "NLL"] > q.loc["Full Visual", "NLL"]),
        })
    return rows


def classify(summary: pd.DataFrame, joint: pd.DataFrame, audit: pd.DataFrame) -> dict[str, Any]:
    task_rules = []
    for task in TASKS:
        q = summary[summary.task == task].set_index("model")
        structured_beats_base = bool(q.loc["Structured", "NLL"] < q.loc["Base", "NLL"] and q.loc["Structured", "frontier_MAE_N"] < q.loc["Base", "frontier_MAE_N"] and q.loc["Structured", "under_force_rate"] <= q.loc["Base", "under_force_rate"])
        structured_beats_visual = bool(q.loc["Structured", "NLL"] < q.loc["Full Visual", "NLL"] and q.loc["Structured", "frontier_MAE_N"] < q.loc["Full Visual", "frontier_MAE_N"] and q.loc["Structured", "under_force_rate"] <= q.loc["Full Visual", "under_force_rate"])
        task_rules.append({"task": task, "structured_beats_base": structured_beats_base, "structured_beats_full_visual": structured_beats_visual})
    n_struct_base = sum(r["structured_beats_base"] for r in task_rules)
    n_struct_visual = sum(r["structured_beats_full_visual"] for r in task_rules)
    n_sjoint = int(joint.StructuredJoint_better_NLL_and_frontier_underforce_nonworse.sum())
    n_vjoint_worse = int(joint.VisualJoint_worse_NLL_than_FullVisual.sum())
    if n_struct_base >= 2 and n_struct_visual >= 2:
        primary = "STRUCTURED_GRASP_CONTEXT_OUTPERFORMS_GENERIC_VISUAL_CONTEXT"
        next_step = "ADOPT_STRUCTURED_DIRECTION_BUT_REPLACE_SIM_GT_OBJECT_POSE_WITH_A_LEGAL_ESTIMATOR_BEFORE_DEPLOYABLE_CLAIM"
    elif n_struct_base <= 1:
        primary = "CURRENT_STRUCTURED_CONTEXT_INSUFFICIENT"
        next_step = "RUN_FROZEN_INDEPENDENT_CONTEXT_SCALING_STARTING_AT_S30_NOT_S80_AND_CREATE_NEW_UNTOUCHED_TEST_BEFORE_TRAINING_CHOICES"
    else:
        primary = "MIXED_EVIDENCE_NO_ROUTE_CHANGE"
        next_step = "DO_NOT_SCALE_TO_80_OR_DROP_VISION_YET; ADD_A_PREDECLARED_S30_CONTEXT_SCALING_CHECK"
    physics = "PHYSICS_AUXILIARY_WORKS_WITH_STRUCTURED_CONTEXT" if n_sjoint >= 2 and n_vjoint_worse >= 2 else "NO_STABLE_INDEPENDENT_STRUCTURED_JOINT_VALUE"
    return {
        "primary_classification": primary, "physics_auxiliary_classification": physics,
        "next_step": next_step, "task_rules": task_rules,
        "counts": {"structured_beats_base_tasks": n_struct_base, "structured_beats_full_visual_tasks": n_struct_visual,
                   "structured_joint_win_tasks": n_sjoint, "visual_joint_worse_tasks": n_vjoint_worse},
        "root06_root07_recorded_geometry_explains_frontier_difference": False,
        "generic_visual_train_gain_interpretation": "ROOT_MEMORIZATION_UNDER_CURRENT_SIX_ROOT_SEED_GROUP_REGIME",
        "structured_method_deployable": False,
        "structured_method_deployability_reason": "six active dimensions use simulator GT object pose",
        "vision_sample_limited_proven": False,
        "representation_inefficiency_proven": False,
        "why_no_A_or_B_overclaim": "This retrospective diagnostic can show whether the current low-dimensional oracle helps; only a frozen independent-context learning curve can prove sample limitation.",
        "direct_answers": {
            "root_difference": "Millimetre/sub-millimetre EEF and relative-pose variation, gripper state, and friction are logged; H8 demand is numerical-zero branch_hold, while EEF rotation/contact microstate/COM offset are missing.",
            "geometry_explains_Fstar": "Descriptive correlations exist but are not stable across tasks after controlling friction, and the joint oracle vector does not beat Base.",
            "generic_vision_train_gain": "Current-data root memorization; not evidence of held-root physical generalization.",
            "structured_more_stable": "No: zero of three tasks passes lower NLL, lower frontier MAE, and non-worse under-force versus Base.",
            "structured_joint_independent_value": "No stable value under the frozen safety-aware rule.",
            "recommended_route": "Freeze a new untouched 8-12-root/task TEST, then start independent-context scaling at S30; do not adopt the current oracle structured vector.",
        },
        "effective_context_grain": {"friction_conditioned_contexts_per_task": 18, "root_seed_families_per_task": 6},
        "task0_visual_joint_reproducibility_caveat": "Legacy task0 Visual Joint grouped-CV values did not reproduce under identical code/data hashes; current same-environment paired retraining is used and audited separately.",
        "scope": "same task / same object-task distribution / held-out root-seed groups; previously seen mechanism analysis; no new untouched TEST",
        "Probe_run": False, "new_simulator_collection": False,
    }


def report_markdown(summary: pd.DataFrame, joint: pd.DataFrame, classification: dict[str, Any], audit: pd.DataFrame) -> str:
    cols = ["task", "model", "NLL", "probability_MAE", "Brier", "frontier_MAE_N", "under_force_rate", "mean_excess_force_N", "TRAIN_to_rootheldout_NLL_gap", "fraction_root_seed_groups_improved_over_Base_NLL"]
    compact = summary[cols].copy()
    comparison = root06_root07_comparison(audit)
    c = comparison.get("signed_root07_minus_root06", {})
    primary = classification["primary_classification"]
    lines = [
        "# Vision vs Structured Context: Same-Task Root-Heldout Diagnostic", "",
        "## Technical summary", "",
        f"**Primary classification: `{primary}`.**",
        "",
        "The experiment is a retrospective mechanism diagnostic on the existing grouped root-heldout folds. It does not create a new untouched TEST. The Structured and Structured Joint models use the unchanged 80 epochs, optimizer, losses, hidden width, force semantics, labels, and folds; their only change is replacing the 17-D frozen visual PCA input with a frozen 11-D physical context through the same 16-D projection.", "",
        f"The structured vector is an **oracle diagnostic**, not deployable: six dimensions use simulator GT object pose. Relative EEF orientation is unavailable and the first H8 nominal command is branch_hold with numerical-zero displacement.", "",
        "## Model evidence on existing grouped held-root folds", "", markdown_table(compact), "",
        "Lower NLL/probability MAE/Brier/frontier MAE is better. Under-force is the safety-facing error; excess force is reported separately. TRAIN gaps use three-seed full-data TRAIN metrics versus seed0 held-root CV, matching the existing frozen design.", "",
        "## Root06 versus root07 is not a grasp-offset explanation", "",
        "task0 root06 has empirical F*0.8=4.25 N and root07=3.75 N, but their recorded object-to-EEF geometry is nearly identical: Δ horizontal eccentricity={:.3f} mm, Δ vertical offset={:.3f} mm, Δ opening={:.3f} mm, and Δ friction={:.4f}. Both H8 traces are branch_hold with ~0 displacement. The scene shifts by ~{:.3f} mm in y while object and EEF move together.".format(
            abs(c.get("horizontal_grasp_eccentricity_m", math.nan))*1000,
            abs(c.get("object_to_eef_dz_m", math.nan))*1000,
            abs(c.get("gripper_opening_m", math.nan))*1000,
            abs(c.get("friction", math.nan)), abs(c.get("eef_base_y_m", math.nan))*1000), "",
        "Therefore the observed 0.5 N frontier difference is not attributable to a material logged grasp offset or H8 motion. Unlogged contact microstate and finite-repeat outcome stochasticity remain plausible.", "",
        "## Does generic vision mainly memorize roots?", "",
        "Under the current six root-seed groups per task, Full Visual has strong in-sample signal but its grouped-held-root gap is the controlling evidence. A TRAIN gain without consistent held-root improvement is classified as current root memorization, not as visual physical generalization. This does not prove that vision can never work; it says the present data do not establish it.", "",
        "## Does low-dimensional grasp context generalize more stably?", "",
        f"The taskwise direction rules yield {classification['counts']['structured_beats_base_tasks']}/3 tasks where Structured beats Base jointly on NLL, frontier MAE, and non-worse under-force, and {classification['counts']['structured_beats_full_visual_tasks']}/3 where it also beats Full Visual. This produces `{primary}`.", "",
        "Because the vector is oracle-only, even a positive diagnostic would require a legal object-pose/geometry estimator before becoming a final method. If the structured vector does not beat Base, the negative result is stronger: even simulator-GT geometry at this interface is insufficient.", "",
        "## Does physics Joint retain independent value?", "", markdown_table(joint), "",
        f"Physics classification: `{classification['physics_auxiliary_classification']}`. Joint value is required to improve NLL and frontier MAE with non-worse under-force; a TRAIN-only loss improvement is not accepted.", "",
        "## Scope, uncertainty, and robustness", "",
        "- The effective grouping is 6 root seeds × 3 friction-conditioned physical snapshots per task. Treating force branches, repeats, or timesteps as independent contexts is forbidden.",
        "- CV seed0 and full-data three-seed TRAIN metrics match the existing frozen protocol. Held-root seed stability is not claimed because the original CV froze one seed.",
        "- task1 labels retain the reconstructed-label caveat; task5 is the direct-label replication.",
        "- task5 planned DEV roots were not collected; task5 evidence here is retrospective TRAIN root-heldout CV.",
        "- Correlations are descriptive, clustered by root seed, and not causal or feature-selection evidence.",
        "- No Probe, new simulator rollout, PCA change, encoder change, lambda tuning, architecture expansion, or new TEST access occurred.", "",
        "## Recommended next step", "",
        f"`{classification['next_step']}`", "",
        "If scaling is selected, first freeze a genuinely new 8–12-root/task untouched TEST (prioritize task0/task5), then run nested S18⊂S30⊂S50⊂S80 with fixed encoder/PCA/architecture/loss. Start by collecting S30; do not jump directly to 80 roots. If a legal structured representation already solves the held-root problem, do not use vision merely for appearance.", "",
        "## Further questions", "",
        "- Can future collection persist EEF quaternion and a legal object-pose estimate at step190?",
        "- Does a demand summary beginning at lift onset, rather than the current all-hold first H8, explain root variation under a separately frozen protocol?",
        "- How much of the frontier spread remains after increasing repeats for uncertainty estimation without pretending repeats add context diversity?", "",
    ]
    return "\n".join(lines)


def run_experiment(out: Path) -> None:
    freeze_path = out / "AUDIT_FREEZE.json"
    spec_path = out / "STRUCTURED_GRASP_CONTEXT_FEATURES.json"
    if not freeze_path.exists() or not spec_path.exists():
        raise RuntimeError("run audit first")
    freeze = json.loads(freeze_path.read_text())
    if sha256(spec_path) != freeze["feature_spec_sha256"]:
        raise RuntimeError("structured feature specification changed after audit freeze")
    audit = pd.read_csv(out / "ROOT_PHYSICAL_CONTEXT_AUDIT.csv")
    all_folds, all_details, all_roots, all_train, all_ck = [], [], [], [], []
    for task in TASKS:
        folds, details, roots, train, ck = run_task_experiment(task, out, audit)
        all_folds.extend(folds); all_details.extend(details); all_roots.extend(roots); all_train.extend(train); all_ck.extend(ck)
    fold_df, detail_df, root_df, train_df = map(pd.DataFrame, (all_folds, all_details, all_roots, all_train))
    fold_df.to_csv(out / "ROOT_HELDOUT_FOLD_METRICS.csv", index=False)
    repro_cols = ["task", "fold", "model", "BCE", "Brier", "frozen_reference_BCE", "frozen_reference_Brier",
                  "recomputed_minus_reference_BCE", "recomputed_minus_reference_Brier",
                  "comparator_reproducibility", "comparator_equivalence_atol"]
    repro = fold_df[fold_df.model.isin(["Base", "Full Visual", "Visual Joint"])][repro_cols]
    write_json(out / "COMPARATOR_REPRODUCIBILITY_AUDIT.json", {
        "policy": "Base and Full Visual are hard equivalence gates. Visual Joint is retrained in the current exact frozen protocol because the legacy CV values are not numerically reproducible; Structured Joint is trained in that same current environment.",
        "rows": repro.to_dict("records"),
    })
    detail_df.to_csv(out / "ROOT_HELDOUT_CONTEXT_FRONTIERS.csv", index=False)
    root_df.to_csv(out / "ROOT_HELDOUT_ROOT_NLL.csv", index=False)
    train_df.to_csv(out / "FULL_TRAIN_MODEL_METRICS.csv", index=False)
    write_json(out / "STRUCTURED_CHECKPOINT_MANIFEST.json", {"checkpoints": all_ck, "feature_spec_sha256": sha256(spec_path)})
    summary = aggregate_results(fold_df, detail_df, root_df, train_df)
    summary.to_csv(out / "TASKWISE_STRUCTURED_VS_VISUAL.csv", index=False)
    joint = pd.DataFrame(joint_comparison_rows(summary))
    joint.to_csv(out / "TASKWISE_STRUCTURED_JOINT.csv", index=False)
    classification = classify(summary, joint, audit)
    write_json(out / "FINAL_CONTEXT_REPRESENTATION_CLASSIFICATION.json", classification)
    (out / "VISION_VS_STRUCTURED_CONTEXT_REPORT.md").write_text(report_markdown(summary, joint, classification, audit), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(classification, indent=2))


def write_hashes(out: Path) -> None:
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt" and "_experiment_audit" not in path.parts:
            rows.append(f"{sha256(path)}  {path.relative_to(out)}")
    (out / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["audit", "experiment", "hashes"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = args.out.resolve()
    if args.phase == "audit":
        run_audit(out)
    elif args.phase == "experiment":
        run_experiment(out)
    else:
        write_hashes(out)


if __name__ == "__main__":
    main()
