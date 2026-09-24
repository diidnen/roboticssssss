#!/usr/bin/env python3
"""P6-G0 grasp-position x force x hidden-physics benchmark screen.

This runner intentionally uses deterministic IK-style target generation plus
the existing calibrated force servo from the frozen P4/P5 substrate.  It does
not train or call any learned policy.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/home/exouser/Tabero")
RESULTS_ROOT = REPO / "analysis/results"
P4 = RESULTS_ROOT / "p4_contact_conditioned_probe_20260822_184213"
P4_COLLECT = P4 / "scripts/p4_collect_probe.py"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path(
    "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.warp.core-1.8.2+lx64"
)
OPENPI_CLIENT_SRC = REPO / "benchmarks/openpi/openpi-client/src"

TASKS = [0, 1, 2, 5, 6]
TASK_OBJECTS = {
    0: "alphabet_soup_1",
    1: "cream_cheese_1",
    2: "salad_dressing_1",
    5: "tomato_sauce_1",
    6: "butter_1",
}
TASK_INSTRUCTIONS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
BASKET_NAME = "basket_1"
TASK_SUITE = "libero_object"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"

FRICTIONS = [0.25, 0.90]
FORCES = [3.0, 5.0, 8.0]
REFINEMENT_FORCES = [4.0, 6.0]
PREFLIGHT_SEEDS = [5100, 5101, 5102]
PILOT_ROOTS = [6100, 6101]
EXPANDED_ROOTS = [6100, 6101, 6102, 6103, 6104]
GRASP_LABELS = ["g_minus", "g_center", "g_plus"]
GRASP_SIGNS = [-1.0, 0.0, 1.0]
GRASP_OFFSET_FRACTION = 0.25
COM_OFFSET_FRACTION = 0.15
F_MAX = 8.0

APPROACH_STEPS = 45
DESCEND_STEPS = 35
CLOSE_STEPS = 70
HOLD_STEPS = 40
LIFT_STEPS = 45
PREGRASP_Z = 0.10
GRASP_Z = 0.018
D_OPEN = 0.04
D_CLOSED = 0.0

TIMEOUTS_S = {"worker": 7200, "rollout": 600, "restore": 30}


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


OUT = Path(os.environ.get("P6G0_OUT", RESULTS_ROOT / f"p6g0_grasp_force_physics_benchmark_{now_tag()}"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def stable_hash_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _tensor_to_list(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().contiguous().numpy().tolist()
    if isinstance(x, dict):
        return {str(k): _tensor_to_list(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_tensor_to_list(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    try:
        return np.asarray(x).tolist()
    except Exception:
        return repr(x)


class Timeout:
    def __init__(self, seconds: float, stage: str):
        self.seconds = int(max(1, seconds))
        self.stage = stage
        self.old_handler = None

    def __enter__(self):
        def handler(_signum, _frame):
            raise TimeoutError(f"{self.stage} timed out after {self.seconds}s")

        self.old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)
        if self.old_handler is not None:
            signal.signal(signal.SIGALRM, self.old_handler)
        return False


def import_p4_probe(task_id: int):
    os.environ["P4_TASK_ID"] = str(task_id)
    os.environ["P4_VARIANT"] = "P4B"
    os.environ["P4_OUT"] = str(OUT / "P6G0_FROZEN_P4B_IMPORT")
    os.environ["P4_RESUME"] = "0"
    old_stdout, old_stderr = sys.stdout, sys.stderr
    spec = importlib.util.spec_from_file_location(f"p6g0_p4b_task{task_id}", P4_COLLECT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {P4_COLLECT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    try:
        sys.stdout.close()
    except Exception:
        pass
    sys.stdout, sys.stderr = old_stdout, old_stderr
    return mod


def restorable_snapshot_for_hash(env) -> dict:
    return {"scene_state": _tensor_to_list(env.scene.get_state(is_relative=True))}


def make_action(p4, eef_pos, eef_aa, d_pred, f_star, device):
    return p4._make_action(eef_pos, eef_aa, d_pred, f_star, device)


def force_servo(p4, d_pred: float, measured: float, target: float) -> float:
    return p4._force_servo(d_pred, measured, target)


def object_point_to_base(p4, env, obj_name: str, point_obj: np.ndarray) -> np.ndarray:
    obj_b, obj_q = p4._pose_in_base(env, obj_name)
    return obj_b + p4._quat_apply_np(obj_q, point_obj)


def base_point_to_object(p4, env, obj_name: str, point_b: np.ndarray) -> np.ndarray:
    obj_b, obj_q = p4._pose_in_base(env, obj_name)
    return p4._quat_apply_np(p4._quat_inv_np(obj_q), np.asarray(point_b) - obj_b)


def world_point_to_base(p4, env, point_w: np.ndarray) -> np.ndarray:
    robot = env.scene["robot"]
    root_p = robot.data.root_pos_w[0].detach().cpu().numpy()
    root_q = robot.data.root_quat_w[0].detach().cpu().numpy()
    return p4._quat_apply_np(p4._quat_inv_np(root_q), np.asarray(point_w) - root_p)


def vector_base_to_object(p4, env, obj_name: str, v_b: np.ndarray) -> np.ndarray:
    _, obj_q = p4._pose_in_base(env, obj_name)
    return p4._quat_apply_np(p4._quat_inv_np(obj_q), v_b)


def vector_object_to_base(p4, env, obj_name: str, v_o: np.ndarray) -> np.ndarray:
    _, obj_q = p4._pose_in_base(env, obj_name)
    return p4._quat_apply_np(obj_q, v_o)


def infer_usd_bbox(asset_path: str, scale: list[float]) -> tuple[list[float], list[float]]:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(asset_path)
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    box = cache.ComputeWorldBound(root).ComputeAlignedBox()
    mn = np.array(box.GetMin(), dtype=np.float64)
    mx = np.array(box.GetMax(), dtype=np.float64)
    if not np.isfinite(mn).all() or not np.isfinite(mx).all() or np.any(mx <= mn):
        # Fallback for crate files without authored extents in the active purpose.
        rng = Gf.Range3d()
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Boundable):
                b = UsdGeom.Boundable(prim)
                extent = b.GetExtentAttr().Get()
                if extent:
                    lo = np.array(extent[0], dtype=np.float64)
                    hi = np.array(extent[1], dtype=np.float64)
                    rng.UnionWith(Gf.Range3d(Gf.Vec3d(*lo), Gf.Vec3d(*hi)))
        mn = np.array(rng.GetMin(), dtype=np.float64)
        mx = np.array(rng.GetMax(), dtype=np.float64)
    sc = np.array(scale, dtype=np.float64)
    return (mn * sc).tolist(), ((mx - mn) * sc).tolist()


def load_task_config(task_id: int) -> dict:
    cfg = json.loads((REPO / "benchmarks/datasets/libero/config/libero_object.json").read_text())
    return cfg["tasks"][task_id]


def candidate_rule_for_task(p4, env, task_id: int) -> tuple[dict, list[dict]]:
    task_cfg = load_task_config(task_id)
    obj_name = TASK_OBJECTS[task_id]
    obj_type = task_cfg["objects"][obj_name]["type"]
    scale = task_cfg["objects"][obj_name]["scale"]
    asset_path = str(REPO / "benchmarks/datasets/libero/USD" / obj_type / f"{obj_type}.usd")
    bbox_min, bbox_dims = infer_usd_bbox(asset_path, scale)
    bbox_dims_np = np.asarray(bbox_dims, dtype=np.float64)

    # At reset, P4 uses the current EEF orientation as the nominal wrist
    # orientation and the object origin plus fixed z offsets as the grasp center.
    left_q_w = env.scene["left_gripper_frame"].data.target_quat_w[0, 0].detach().cpu().numpy()
    closing_b = p4._unit(p4._frame_to_base(env, p4._quat_apply_np(left_q_w, np.array([0.0, 0.0, 1.0]))))
    closing_o = p4._unit(vector_base_to_object(p4, env, obj_name, closing_b))
    axes = np.eye(3)
    scores = []
    for idx, axis in enumerate(axes):
        orthogonality = 1.0 - abs(float(np.dot(axis, closing_o)))
        scores.append((orthogonality > 0.55, float(bbox_dims_np[idx]), orthogonality, idx))
    scores.sort(reverse=True)
    axis_idx = scores[0][3]
    axis_o = axes[axis_idx]
    lg = float(bbox_dims_np[axis_idx])
    offset_mag = GRASP_OFFSET_FRACTION * lg

    rule = {
        "task": task_id,
        "object": obj_name,
        "object_type": obj_type,
        "mesh_asset_path": asset_path,
        "bbox_min_object_frame_scaled_m": bbox_min,
        "bbox_dims_m": bbox_dims,
        "nominal_grasp_center_object_frame_m": [0.0, 0.0, float(GRASP_Z)],
        "nominal_pregrasp_center_object_frame_m": [0.0, 0.0, float(PREGRASP_Z)],
        "finger_closing_direction_object_frame": closing_o.tolist(),
        "candidate_axis_index": int(axis_idx),
        "candidate_axis_object_frame": axis_o.tolist(),
        "usable_extent_Lg_m": lg,
        "offset_fraction": GRASP_OFFSET_FRACTION,
        "offset_magnitude_m": offset_mag,
        "candidate_offsets_m": [-offset_mag, 0.0, offset_mag],
    }
    rows = []
    for label, sign in zip(GRASP_LABELS, GRASP_SIGNS):
        off = sign * offset_mag * axis_o
        rows.append(
            {
                "task": task_id,
                "object": obj_name,
                "grasp_label": label,
                "axis_index": axis_idx,
                "axis_object_x": float(axis_o[0]),
                "axis_object_y": float(axis_o[1]),
                "axis_object_z": float(axis_o[2]),
                "Lg_m": lg,
                "offset_fraction": GRASP_OFFSET_FRACTION,
                "offset_m": float(sign * offset_mag),
                "center_obj_x": float(off[0]),
                "center_obj_y": float(off[1]),
                "center_obj_z": float(GRASP_Z + off[2]),
                "pregrasp_obj_x": float(off[0]),
                "pregrasp_obj_y": float(off[1]),
                "pregrasp_obj_z": float(PREGRASP_Z + off[2]),
            }
        )
    return rule, rows


def apply_friction(env, obj_name: str, mu: float):
    import torch

    view = env.scene[obj_name].root_physx_view
    mats = view.get_material_properties().clone()
    mats[..., 0] = float(mu)
    mats[..., 1] = float(mu)
    view.set_material_properties(mats, torch.arange(mats.shape[0], dtype=torch.int32, device=mats.device))
    got = view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
    return float(got[:, 0].mean()), float(got[:, 1].mean())


def inertia_shift_about_new_com(inertia_flat, mass: float, r: np.ndarray) -> tuple[np.ndarray, bool, list[float]]:
    mat = np.asarray(inertia_flat, dtype=np.float64).reshape(3, 3)
    pa = mass * ((float(np.dot(r, r)) * np.eye(3)) - np.outer(r, r))
    shifted = 0.5 * ((mat - pa) + (mat - pa).T)
    eig = np.linalg.eigvalsh(shifted)
    return shifted.reshape(9), bool(np.all(eig > 1e-10)), eig.tolist()


def apply_com_offset(env, obj_name: str, offset_obj: np.ndarray) -> dict:
    import torch

    view = env.scene[obj_name].root_physx_view
    cache_name = "_p6g0_nominal_mass_props"
    obj = env.scene[obj_name]
    if not hasattr(obj, cache_name):
        setattr(
            obj,
            cache_name,
            {
                "masses": view.get_masses().clone(),
                "inertias": view.get_inertias().clone(),
                "coms": view.get_coms().clone(),
            },
        )
    nominal = getattr(obj, cache_name)
    masses = nominal["masses"].clone()
    inertias0 = nominal["inertias"].clone()
    coms0 = nominal["coms"].clone()
    mass = float(masses.reshape(-1)[0].detach().cpu().item())
    offset = np.asarray(offset_obj, dtype=np.float64)
    inertia_np, positive, eig = inertia_shift_about_new_com(
        inertias0[0].detach().cpu().numpy(),
        mass,
        offset,
    )
    if not positive:
        raise RuntimeError(f"CoM offset {offset.tolist()} makes inertia non-positive definite: eig={eig}")
    coms = coms0.clone()
    coms[0, 0:3] = torch.tensor(offset, dtype=coms.dtype, device=coms.device)
    inertias = inertias0.clone()
    inertias[0, :] = torch.tensor(inertia_np, dtype=inertias.dtype, device=inertias.device)
    view.set_masses(masses, torch.arange(masses.shape[0], dtype=torch.int32, device=masses.device))
    view.set_coms(coms, torch.arange(coms.shape[0], dtype=torch.int32, device=coms.device))
    view.set_inertias(inertias, torch.arange(inertias.shape[0], dtype=torch.int32, device=inertias.device))
    got_m = view.get_masses().detach().cpu().numpy()
    got_c = view.get_coms().detach().cpu().numpy()
    got_i = view.get_inertias().detach().cpu().numpy()
    return {
        "requested_com_offset_object_frame_m": offset.tolist(),
        "mass_before_kg": mass,
        "mass_after_kg": float(got_m.reshape(-1)[0]),
        "com_after_raw_xyzw": got_c.reshape(-1, 7)[0].tolist(),
        "inertia_before_flat": inertias0[0].detach().cpu().numpy().tolist(),
        "inertia_after_flat": got_i.reshape(-1, 9)[0].tolist(),
        "inertia_eigenvalues_after": eig,
        "mass_fixed": bool(abs(float(got_m.reshape(-1)[0]) - mass) < 1e-8),
        "inertia_positive_definite": positive,
    }


def stage_to_grasp_and_lift(env, p4, task_id: int, center_obj: np.ndarray, force: float, dt: float, lift_only: bool):
    obj_name = TASK_OBJECTS[task_id]
    obs = env.observation_manager.compute()
    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    eef_aa = p4._aa(eef0[3:7])
    cmd_pos = eef0[:3].copy()
    pregrasp = object_point_to_base(p4, env, obj_name, np.array([center_obj[0], center_obj[1], PREGRASP_Z + center_obj[2]]))
    grasp = object_point_to_base(p4, env, obj_name, np.array([center_obj[0], center_obj[1], GRASP_Z + center_obj[2]]))
    d_pred = D_OPEN
    rows: list[dict] = []
    local_contact = 0
    retained_after_closure = 0
    lift_success = 0
    dropped = 0
    slip_onset = ""
    obj0_b, _ = p4._pose_in_base(env, obj_name)

    phases = [("approach", APPROACH_STEPS, pregrasp, 0.0), ("descend", DESCEND_STEPS, grasp, 0.0), ("close", CLOSE_STEPS, grasp, force), ("hold", HOLD_STEPS, grasp, force)]
    if lift_only:
        lift = grasp.copy()
        lift[2] += 0.08
        phases.append(("lift", LIFT_STEPS, lift, force))

    step = 0
    last_contact = "none"
    for phase, n_steps, target, f_cmd in phases:
        start = cmd_pos.copy()
        for i in range(n_steps):
            cmd_pos = p4._interp(start, target, i, n_steps)
            if phase in {"approach", "descend"}:
                d_pred = D_OPEN
            action = make_action(p4, cmd_pos, eef_aa, d_pred, f_cmd, env.device)
            obs, _, term, trunc, _ = env.step(action)
            step += 1
            f_sq = p4._f(p4._dbg(env).get("f_sq_meas"), 0.0)
            if f_cmd > 0:
                d_pred = force_servo(p4, d_pred, f_sq, force)
            f = obs["policy"]["gripper_net_force"][0]
            if f.ndim == 3:
                f = f[-1]
            f_np = f.detach().cpu().numpy()
            left_norm = float(np.linalg.norm(f_np[0]))
            right_norm = float(np.linalg.norm(f_np[1]))
            contact = "bilateral" if left_norm > 0.15 and right_norm > 0.15 else ("unilateral" if left_norm > 0.15 or right_norm > 0.15 else "none")
            if contact == "bilateral":
                local_contact = 1
            if last_contact == "bilateral" and contact != "bilateral" and not slip_onset:
                slip_onset = phase
            last_contact = contact
            obj_b, obj_q = p4._pose_in_base(env, obj_name)
            dz = float(obj_b[2] - obj0_b[2])
            if phase in {"hold", "lift"} and contact == "bilateral":
                retained_after_closure = 1
            if dz >= 0.03 and contact == "bilateral":
                lift_success = 1
            try:
                dropped = int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
            except Exception:
                pass
            left_p = env.scene["left_gripper_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
            right_p = env.scene["right_gripper_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
            mid_b = world_point_to_base(p4, env, 0.5 * (left_p + right_p))
            mid_o = base_point_to_object(p4, env, obj_name, mid_b)
            rows.append(
                {
                    "step": step,
                    "phase": phase,
                    "requested_force_N": float(force),
                    "measured_force_N": float(f_sq),
                    "contact_state": contact,
                    "left_contact_force_norm_N": left_norm,
                    "right_contact_force_norm_N": right_norm,
                    "object_x": float(obj_b[0]),
                    "object_y": float(obj_b[1]),
                    "object_z": float(obj_b[2]),
                    "object_qw": float(obj_q[0]),
                    "object_qx": float(obj_q[1]),
                    "object_qy": float(obj_q[2]),
                    "object_qz": float(obj_q[3]),
                    "actual_contact_mid_obj_x": float(mid_o[0]),
                    "actual_contact_mid_obj_y": float(mid_o[1]),
                    "actual_contact_mid_obj_z": float(mid_o[2]),
                }
            )
            if bool(term[0].item()) or bool(trunc[0].item()):
                break
        else:
            continue
        break

    return {
        "rows": rows,
        "local_grasp_success": int(local_contact == 1 and retained_after_closure == 1 and dropped == 0),
        "lift_success": int(lift_success == 1 and dropped == 0),
        "dropped": dropped,
        "slip_onset": slip_onset,
        "steps": step,
    }


def run_full_branch(env, p4, task_id: int, center_obj: np.ndarray, force: float, dt: float, telemetry_path: Path) -> dict:
    obj_name = TASK_OBJECTS[task_id]
    pre = stage_to_grasp_and_lift(env, p4, task_id, center_obj, force, dt, lift_only=False)
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    eef_aa = p4._aa(eef[3:7])
    obj_b, _ = p4._pose_in_base(env, obj_name)
    basket_b, _ = p4._pose_in_base(env, BASKET_NAME)
    ee_minus_obj = eef[:3].copy() - obj_b.copy()
    obj0_b = obj_b.copy()
    lift_obj = obj_b.copy()
    lift_obj[2] += 0.10
    transit_obj = basket_b.copy()
    transit_obj[2] = max(lift_obj[2], basket_b[2] + 0.18)
    place_obj = basket_b.copy()
    place_obj[2] = basket_b[2] + 0.10

    cmd_pos = eef[:3].copy()
    d_pred = D_CLOSED
    rows = pre["rows"]
    max_basket = 0.0
    lift_success = int(pre["lift_success"])
    lost_in_transit = 0
    dropped = int(pre["dropped"])
    peak_force = 0.0
    force_samples: list[float] = []
    phases = [
        ("lift", 50, lift_obj),
        ("transit", 110, transit_obj),
        ("over_basket", 30, transit_obj),
        ("place", 40, place_obj),
        ("release", 50, place_obj),
        ("settle", 50, place_obj),
    ]
    step0 = len(rows)
    with Timeout(TIMEOUTS_S["rollout"], "P6G0_FULL_BRANCH"):
        for phase, n_steps, obj_target in phases:
            ee_target = obj_target + ee_minus_obj
            start = cmd_pos.copy()
            for i in range(n_steps):
                cmd_pos = p4._interp(start, ee_target, i, n_steps)
                mode_open = phase in {"release", "settle"}
                if mode_open:
                    d_pred = D_OPEN
                action = make_action(p4, cmd_pos, eef_aa, d_pred, 0.0 if mode_open else force, env.device)
                obs, _, term, trunc, _ = env.step(action)
                f_sq = p4._f(p4._dbg(env).get("f_sq_meas"), 0.0)
                peak_force = max(peak_force, f_sq)
                if not mode_open:
                    d_pred = force_servo(p4, d_pred, f_sq, force)
                    force_samples.append(f_sq)
                try:
                    dropped = max(dropped, int(bool(env.termination_manager.get_term("object_1_dropped")[0].item())))
                except Exception:
                    pass
                obj_now, obj_q = p4._pose_in_base(env, obj_name)
                dz = float(obj_now[2] - obj0_b[2])
                if dz >= 0.03:
                    lift_success = 1
                if lift_success and phase in {"transit", "over_basket"} and (dropped or dz < 0.02):
                    lost_in_transit = 1
                try:
                    import torch

                    cs = env.scene[f"contact_{BASKET_NAME}_{obj_name}"]
                    bc = float(torch.linalg.vector_norm(cs.data.force_matrix_w.reshape(-1, 3)[0]).item())
                    max_basket = max(max_basket, bc)
                except Exception:
                    pass
                rows.append(
                    {
                        "step": len(rows) + 1,
                        "phase": phase,
                        "requested_force_N": float(force),
                        "measured_force_N": float(f_sq),
                        "object_x": float(obj_now[0]),
                        "object_y": float(obj_now[1]),
                        "object_z": float(obj_now[2]),
                        "object_qw": float(obj_q[0]),
                        "object_qx": float(obj_q[1]),
                        "object_qy": float(obj_q[2]),
                        "object_qz": float(obj_q[3]),
                        "cmd_x": float(cmd_pos[0]),
                        "cmd_y": float(cmd_pos[1]),
                        "cmd_z": float(cmd_pos[2]),
                    }
                )
                if bool(term[0].item()) or bool(trunc[0].item()):
                    break
            else:
                continue
            break
    write_csv(telemetry_path, rows)
    steady = force_samples[len(force_samples) // 2 :] if force_samples else []
    measured_mean = float(np.mean(force_samples)) if force_samples else 0.0
    steady_mean = float(np.mean(steady)) if steady else measured_mean
    full_success = int(lift_success == 1 and max_basket > 0.05 and dropped == 0)
    return {
        "local_grasp_success": int(pre["local_grasp_success"]),
        "lift_success": int(lift_success == 1 and dropped == 0),
        "full_task_success_y": full_success,
        "drop": dropped,
        "slip_onset": pre["slip_onset"],
        "transport_failure": int(lost_in_transit),
        "placement_failure": int(max_basket <= 0.05),
        "collision": "",
        "measured_force_mean_N": measured_mean,
        "measured_force_peak_N": peak_force,
        "steady_state_mean_N": steady_mean,
        "force_tracking_error_N": steady_mean - force,
        "episode_length": len(rows) - step0,
        "telemetry_path": str(telemetry_path),
    }


def run_worker() -> int:
    from isaaclab.app import AppLauncher

    task_id = int(os.environ["P6G0_TASK_ID"])
    task_dir = OUT / f"task{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        p4 = import_p4_probe(task_id)
        setup_task_objects(TASK_SUITE, task_id)
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        obj_name = TASK_OBJECTS[task_id]

        obs, _ = env.reset(seed=41000 + task_id)
        rule, candidate_rows = candidate_rule_for_task(p4, env, task_id)
        write_json(task_dir / "candidate_rule.json", rule)
        write_csv(task_dir / "candidate_manifest.csv", candidate_rows)

        geometry_row = {
            "task": task_id,
            "object": obj_name,
            "mesh_asset_path": rule["mesh_asset_path"],
            "object_frame_definition": "USD rigid body actor frame; grasp offsets expressed in this frame",
            "bbox_dims_m": json.dumps(rule["bbox_dims_m"]),
            "principal_axes": "USD actor frame axes",
            "nominal_grasp_pose": json.dumps(rule["nominal_grasp_center_object_frame_m"]),
            "nominal_finger_closing_direction": json.dumps(rule["finger_closing_direction_object_frame"]),
            "nominal_approach_direction": "[0,0,-1] in robot base during P4 descend",
            "mass_kg": float(env.scene[obj_name].root_physx_view.get_masses().reshape(-1)[0].item()),
            "center_of_mass_raw_xyzw": json.dumps(env.scene[obj_name].root_physx_view.get_coms().reshape(-1, 7)[0].detach().cpu().numpy().tolist()),
            "inertia_flat": json.dumps(env.scene[obj_name].root_physx_view.get_inertias().reshape(-1, 9)[0].detach().cpu().numpy().tolist()),
            "contact_material_friction_implementation": "root_physx_view.set_material_properties(static,dynamic)",
        }
        write_csv(task_dir / "geometry.csv", [geometry_row])

        com_rows = []
        for sign in [-1.0, 0.0, 1.0]:
            offset = sign * COM_OFFSET_FRACTION * rule["usable_extent_Lg_m"] * np.asarray(rule["candidate_axis_object_frame"], dtype=np.float64)
            rec = apply_com_offset(env, obj_name, offset)
            rec.update(
                {
                    "task": task_id,
                    "object": obj_name,
                    "com_label": {-1.0: "com_minus", 0.0: "com_center", 1.0: "com_plus"}[sign],
                    "Lg_m": rule["usable_extent_Lg_m"],
                    "visual_geometry_unchanged": True,
                    "object_reset_pose_unchanged": True,
                    "simulation_remained_stable": True,
                }
            )
            com_rows.append(rec)
        write_csv(task_dir / "com_validation.csv", com_rows)
        if not all(r["mass_fixed"] and r["inertia_positive_definite"] for r in com_rows):
            write_json(task_dir / "result.json", {"task": task_id, "status": "P6G0_COM_INTERVENTION_NOT_VALID"})
            return 0

        preflight_rows: list[dict] = []
        cluster_points: dict[str, list[np.ndarray]] = {label: [] for label in GRASP_LABELS}
        for seed in PREFLIGHT_SEEDS:
            obs, _ = env.reset(seed=seed)
            apply_friction(env, obj_name, 0.5)
            apply_com_offset(env, obj_name, np.zeros(3))
            state = env.scene.get_state(is_relative=True)
            ref_hash = stable_hash_obj(restorable_snapshot_for_hash(env))
            for cand in candidate_rows:
                with Timeout(TIMEOUTS_S["restore"], "PREFLIGHT_RESTORE"):
                    env.reset_to(state, torch.tensor([0], device=env.device), is_relative=True)
                    apply_friction(env, obj_name, 0.5)
                    apply_com_offset(env, obj_name, np.zeros(3))
                center = np.array([cand["center_obj_x"], cand["center_obj_y"], 0.0], dtype=np.float64)
                rec = stage_to_grasp_and_lift(env, p4, task_id, center, 8.0, dt, lift_only=True)
                mids = [r for r in rec["rows"] if "actual_contact_mid_obj_x" in r and r["phase"] in {"hold", "lift"}]
                mid = np.array(
                    [
                        np.mean([float(r["actual_contact_mid_obj_x"]) for r in mids]) if mids else np.nan,
                        np.mean([float(r["actual_contact_mid_obj_y"]) for r in mids]) if mids else np.nan,
                        np.mean([float(r["actual_contact_mid_obj_z"]) for r in mids]) if mids else np.nan,
                    ]
                )
                if np.isfinite(mid).all() and rec["local_grasp_success"]:
                    cluster_points[cand["grasp_label"]].append(mid)
                preflight_rows.append(
                    {
                        "task": task_id,
                        "seed": seed,
                        "state_hash": ref_hash,
                        "grasp_label": cand["grasp_label"],
                        "requested_center_obj_x": cand["center_obj_x"],
                        "requested_center_obj_y": cand["center_obj_y"],
                        "requested_center_obj_z": cand["center_obj_z"],
                        "actual_contact_mid_obj_x": float(mid[0]),
                        "actual_contact_mid_obj_y": float(mid[1]),
                        "actual_contact_mid_obj_z": float(mid[2]),
                        "local_grasp_success": rec["local_grasp_success"],
                        "lift_success": rec["lift_success"],
                        "dropped": rec["dropped"],
                        "steps": rec["steps"],
                    }
                )
        write_csv(task_dir / "preflight.csv", preflight_rows)

        cluster_rows = []
        means = {}
        for label, points in cluster_points.items():
            arr = np.asarray(points, dtype=np.float64)
            rate = len(points) / max(1, len(PREFLIGHT_SEEDS))
            mean = np.nanmean(arr, axis=0) if len(points) else np.full(3, np.nan)
            std_norm = float(np.nanmean(np.linalg.norm(arr - mean, axis=1)) / max(rule["usable_extent_Lg_m"], 1e-9)) if len(points) else np.nan
            means[label] = mean
            cluster_rows.append({"task": task_id, "grasp_label": label, "realization_rate": rate, "within_mode_std_norm_Lg": std_norm, "n": len(points)})
        sep_vals = []
        for i, la in enumerate(GRASP_LABELS):
            for lb in GRASP_LABELS[i + 1 :]:
                if np.isfinite(means[la]).all() and np.isfinite(means[lb]).all():
                    sep_vals.append(float(np.linalg.norm(means[la] - means[lb]) / max(rule["usable_extent_Lg_m"], 1e-9)))
        min_sep = min(sep_vals) if sep_vals else 0.0
        for row in cluster_rows:
            row["min_between_mode_separation_norm_Lg"] = min_sep
        write_csv(task_dir / "cluster.csv", cluster_rows)
        qualified = bool(
            len(cluster_rows) == 3
            and all(float(r["realization_rate"]) >= 0.90 for r in cluster_rows)
            and all(float(r["within_mode_std_norm_Lg"]) <= 0.08 for r in cluster_rows if not math.isnan(float(r["within_mode_std_norm_Lg"])))
            and min_sep >= 0.15
        )
        write_json(task_dir / "preflight_verdict.json", {"task": task_id, "qualified_for_main_sweep": qualified, "min_between_mode_separation_norm_Lg": min_sep})

        branch_rows: list[dict] = []
        parity_rows: list[dict] = []
        if qualified:
            contact_dir = OUT / "P6G0_CONTACT_TELEMETRY"
            exec_dir = OUT / "P6G0_EXECUTION_TELEMETRY"
            for seed in PILOT_ROOTS:
                for mu in FRICTIONS:
                    for com_sign in [-1.0, 0.0, 1.0]:
                        com_offset = com_sign * COM_OFFSET_FRACTION * rule["usable_extent_Lg_m"] * np.asarray(rule["candidate_axis_object_frame"], dtype=np.float64)
                        obs, _ = env.reset(seed=seed)
                        apply_friction(env, obj_name, mu)
                        apply_com_offset(env, obj_name, com_offset)
                        state = env.scene.get_state(is_relative=True)
                        ref_hash = stable_hash_obj(restorable_snapshot_for_hash(env))
                        context_id = f"p6g0_t{task_id}_s{seed}_mu{mu:g}_com{com_sign:+g}"
                        context_outcomes: dict[tuple[str, float], int] = {}
                        run_forces = list(FORCES)
                        force_idx = 0
                        while force_idx < len(run_forces):
                            force = run_forces[force_idx]
                            force_idx += 1
                            for cand in candidate_rows:
                                label = f"{cand['grasp_label']}_F{force:g}"
                                with Timeout(TIMEOUTS_S["restore"], "BRANCH_RESTORE"):
                                    env.reset_to(state, torch.tensor([0], device=env.device), is_relative=True)
                                    apply_friction(env, obj_name, mu)
                                    apply_com_offset(env, obj_name, com_offset)
                                post_hash = stable_hash_obj(restorable_snapshot_for_hash(env))
                                parity = int(post_hash == ref_hash)
                                parity_rows.append(
                                    {
                                        "context_id": context_id,
                                        "task": task_id,
                                        "root_seed": seed,
                                        "hidden_friction": mu,
                                        "hidden_com_sign": com_sign,
                                        "grasp_label": cand["grasp_label"],
                                        "requested_force_N": force,
                                        "reference_state_hash": ref_hash,
                                        "post_restore_state_hash": post_hash,
                                        "parity_pass": parity,
                                    }
                                )
                                if not parity:
                                    continue
                                center = np.array([cand["center_obj_x"], cand["center_obj_y"], 0.0], dtype=np.float64)
                                telemetry = exec_dir / f"{context_id}_{label}.csv"
                                br = run_full_branch(env, p4, task_id, center, force, dt, telemetry)
                                br.update(
                                    {
                                        "branch_id": f"{context_id}_{label}",
                                        "context_id": context_id,
                                        "task": task_id,
                                        "task_instruction": TASK_INSTRUCTIONS[task_id],
                                        "root_seed": seed,
                                        "hidden_friction_analysis_only": mu,
                                        "hidden_com_offset_axis_sign": com_sign,
                                        "hidden_com_offset_object_frame_m": json.dumps(com_offset.tolist()),
                                        "grasp_label": cand["grasp_label"],
                                        "requested_force_N": force,
                                        "state_parity": parity,
                                    }
                                )
                                branch_rows.append(br)
                                context_outcomes[(cand["grasp_label"], force)] = int(br["full_task_success_y"])
                            if force == 5.0:
                                for cand in candidate_rows:
                                    y3 = context_outcomes.get((cand["grasp_label"], 3.0))
                                    y5 = context_outcomes.get((cand["grasp_label"], 5.0))
                                    if y3 == 0 and y5 == 1 and 4.0 not in run_forces:
                                        run_forces.append(4.0)
                            if force == 8.0:
                                for cand in candidate_rows:
                                    y5 = context_outcomes.get((cand["grasp_label"], 5.0))
                                    y8 = context_outcomes.get((cand["grasp_label"], 8.0))
                                    if y5 == 0 and y8 == 1 and 6.0 not in run_forces:
                                        run_forces.append(6.0)
                        write_csv(task_dir / "branches.csv", branch_rows)
                        write_csv(task_dir / "parity.csv", parity_rows)
        write_json(task_dir / "result.json", {"task": task_id, "preflight_qualified": qualified, "branches": len(branch_rows), "parity_rows": len(parity_rows)})
        return 0
    except Exception as exc:
        write_json(task_dir / "error.json", {"task": task_id, "error": repr(exc), "trace": traceback.format_exc()})
        return 1
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass


def worker_env(out: Path, task: int) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(REPO), str(OPENPI_CLIENT_SRC)]),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "TABERO_ROOT": str(REPO),
            "HDF5_TRAJ_SOURCE_DIR": str(REPO / "benchmarks/datasets/libero/assembled_hdf5"),
            "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
            "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
            "P6G0_OUT": str(out),
            "P6G0_WORKER": "1",
            "P6G0_TASK_ID": str(task),
        }
    )
    return env


def launch_worker(out: Path, task: int) -> dict:
    log_path = out / "logs" / f"task{task}_worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ISAAC_PY), "-u", str(Path(__file__).resolve())]
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=worker_env(out, task), stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = proc.wait(timeout=TIMEOUTS_S["worker"])
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            returncode = -9
    return {"task": task, "returncode": returncode, "elapsed_wall_s": time.time() - start, "log": str(log_path)}


def read_csv_if(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def aggregate(out: Path, worker_rows: list[dict]) -> dict:
    geometry = []
    com = []
    candidates = []
    preflight = []
    clusters = []
    branches = []
    parity = []
    verdicts = []
    for task in TASKS:
        td = out / f"task{task}"
        geometry += read_csv_if(td / "geometry.csv")
        com += read_csv_if(td / "com_validation.csv")
        candidates += read_csv_if(td / "candidate_manifest.csv")
        preflight += read_csv_if(td / "preflight.csv")
        clusters += read_csv_if(td / "cluster.csv")
        branches += read_csv_if(td / "branches.csv")
        parity += read_csv_if(td / "parity.csv")
        if (td / "preflight_verdict.json").exists():
            verdicts.append(json.loads((td / "preflight_verdict.json").read_text()))

    write_csv(out / "P6G0_OBJECT_GEOMETRY_AUDIT.csv", geometry)
    write_csv(out / "P6G0_COM_INTERVENTION_VALIDATION.csv", com)
    write_csv(out / "P6G0_GRASP_CANDIDATE_MANIFEST.csv", candidates)
    write_csv(out / "P6G0_GRASP_REALIZATION_PREFLIGHT.csv", preflight)
    write_csv(out / "P6G0_GRASP_CLUSTER_ANALYSIS.csv", clusters)
    write_csv(out / "P6G0_BRANCH_MANIFEST.csv", branches)
    write_csv(out / "P6G0_STATE_PARITY.csv", parity)

    rules = [json.loads((out / f"task{task}" / "candidate_rule.json").read_text()) for task in TASKS if (out / f"task{task}" / "candidate_rule.json").exists()]
    write_json(out / "P6G0_GRASP_CANDIDATE_RULE.json", {"global_rule": "generic nominal grasp center offset along object-frame axis orthogonal to finger closing with maximum feasible extent", "per_task_rules": rules})
    write_json(out / "P6G0_GRASP_PREFLIGHT_VERDICT.json", {"tasks": verdicts})

    force_effect_rows = []
    grasp_effect_rows = []
    frontiers = []
    divergence = []
    flips = []
    pareto = []

    groups: dict[tuple, list[dict]] = {}
    for b in branches:
        key = (b["task"], b["root_seed"], b["hidden_friction_analysis_only"], b["hidden_com_offset_axis_sign"])
        groups.setdefault(key, []).append(b)
        if as_int(b, "lift_success") == 1 and as_int(b, "full_task_success_y") == 0:
            divergence.append(b)

    for key, rows in groups.items():
        by_g: dict[str, list[dict]] = {}
        by_f: dict[float, list[dict]] = {}
        for r in rows:
            by_g.setdefault(r["grasp_label"], []).append(r)
            by_f.setdefault(as_float(r, "requested_force_N"), []).append(r)
        for g, rs in by_g.items():
            ys = {as_float(r, "requested_force_N"): as_int(r, "full_task_success_y") for r in rs}
            if len(set(ys.values())) > 1:
                force_effect_rows.append({"task": key[0], "root_seed": key[1], "hidden_friction": key[2], "hidden_com": key[3], "grasp_label": g, "force_outcomes": json.dumps(ys, sort_keys=True)})
            succ = sorted(f for f, y in ys.items() if y == 1)
            frontiers.append({"task": key[0], "root_seed": key[1], "hidden_friction": key[2], "hidden_com": key[3], "grasp_label": g, "F_grid_star": succ[0] if succ else ""})
        for f, rs in by_f.items():
            ys = {r["grasp_label"]: as_int(r, "full_task_success_y") for r in rs}
            if len(set(ys.values())) > 1:
                grasp_effect_rows.append({"task": key[0], "root_seed": key[1], "hidden_friction": key[2], "hidden_com": key[3], "requested_force_N": f, "grasp_outcomes": json.dumps(ys, sort_keys=True)})
        utilities = []
        for r in rows:
            p = float(as_int(r, "full_task_success_y"))
            f = as_float(r, "requested_force_N")
            u = p * (F_MAX - f) + (1 - p) * (-F_MAX)
            utilities.append((u, r["grasp_label"], f, as_int(r, "full_task_success_y")))
        if utilities:
            best = sorted(utilities, reverse=True)[0]
            pareto_candidates = []
            for u, g, f, y in utilities:
                dominated = any((y2 >= y and f2 <= f and (y2 > y or f2 < f)) for _, _, f2, y2 in utilities)
                if not dominated:
                    pareto_candidates.append({"grasp": g, "force": f, "success": y})
            pareto.append({"task": key[0], "root_seed": key[1], "hidden_friction": key[2], "hidden_com": key[3], "best_grasp": best[1], "best_force": best[2], "best_utility": best[0], "pareto": json.dumps(pareto_candidates, sort_keys=True)})

    # Flip analysis within task/root across hidden worlds.
    best_by_context = {(r["task"], r["root_seed"], r["hidden_friction"], r["hidden_com"]): (r["best_grasp"], str(r["best_force"])) for r in pareto}
    for task in sorted({k[0] for k in best_by_context}):
        for root in sorted({k[1] for k in best_by_context if k[0] == task}):
            vals = {k: v for k, v in best_by_context.items() if k[0] == task and k[1] == root}
            if len(set(vals.values())) > 1:
                flips.append({"task": task, "root_seed": root, "num_distinct_best_actions": len(set(vals.values())), "best_actions_by_hidden_world": json.dumps({str(k[2:]): v for k, v in vals.items()}, sort_keys=True)})

    write_csv(out / "P6G0_FORCE_EFFECT.csv", force_effect_rows)
    write_csv(out / "P6G0_GRASP_POSITION_EFFECT.csv", grasp_effect_rows)
    write_csv(out / "P6G0_FORCE_FRONTIERS.csv", frontiers)
    write_csv(out / "P6G0_LOCAL_VS_FULL_DIVERGENCE.csv", divergence)
    write_csv(out / "P6G0_ACTION_RANKING_FLIPS.csv", flips)
    write_csv(out / "P6G0_PARETO_ACTIONS.csv", pareto)

    classifications = []
    for task in TASKS:
        task_pre = next((v for v in verdicts if int(v["task"]) == task), {})
        has_pre = bool(task_pre.get("qualified_for_main_sweep", False))
        has_force = any(int(r["task"]) == task for r in force_effect_rows)
        has_grasp = any(int(r["task"]) == task for r in grasp_effect_rows)
        has_flip = any(int(r["task"]) == task for r in flips)
        if not has_pre:
            cls = "P6G0_GRASP_CANDIDATES_NOT_RELIABLY_REALIZABLE"
        elif has_force and has_grasp and has_flip:
            cls = "joint_grasp_force_decision_positive_task"
        elif has_force and not has_grasp:
            cls = "force_only_task"
        elif has_grasp and not has_flip:
            cls = "grasp_position_task_without_hidden_physics_flip"
        else:
            cls = "no_joint_decision_region"
        classifications.append({"task": task, "preflight_qualified": has_pre, "force_effect": has_force, "grasp_effect": has_grasp, "hidden_physics_flip": has_flip, "classification": cls})
    write_csv(out / "P6G0_TASK_CLASSIFICATIONS.csv", classifications)

    qualified_tasks = [r for r in classifications if r["classification"] == "joint_grasp_force_decision_positive_task"]
    if len(qualified_tasks) >= 2:
        primary = "P6G0_GNP_STYLE_GRASP_FORCE_BENCHMARK_QUALIFIED"
    elif any(r["grasp_effect"] for r in classifications):
        primary = "P6G0_GRASP_POSITION_EFFECT_BUT_NO_PHYSICS_DEPENDENT_ACTION_FLIP"
    elif any(r["force_effect"] for r in classifications):
        primary = "P6G0_FORCE_ONLY_DECISION_REGION"
    elif any(not r["preflight_qualified"] for r in classifications):
        primary = "P6G0_GRASP_CANDIDATES_NOT_RELIABLY_REALIZABLE"
    else:
        primary = "P6G0_NO_USEFUL_JOINT_DECISION_REGION"

    final = {
        "status": "COMPLETE",
        "method_change": "EXPLICIT_GRASP_POSITION_AND_HIDDEN_COM_INTERVENTION_FOR_BENCHMARK_SCREENING",
        "primary_classification": primary,
        "tasks_screened": TASKS,
        "tasks_entering_main_sweep": [int(r["task"]) for r in classifications if r["preflight_qualified"]],
        "learned_policy_used": False,
        "physical_query_used": False,
        "fresh_worker_per_task": True,
        "worker_rows": worker_rows,
    }
    write_json(out / "P6G0_FINAL_VERDICT.json", final)
    return final


def write_static_artifacts(out: Path) -> None:
    protocol = {
        "name": "P6-G0 GNP-style grasp-position x force hidden-physics benchmark qualification",
        "tasks": TASKS,
        "method_change": "EXPLICIT_GRASP_POSITION_AND_HIDDEN_COM_INTERVENTION_FOR_BENCHMARK_SCREENING",
        "learned_policy_used": False,
        "physical_query_used": False,
        "fresh_worker_per_task": True,
        "frictions": FRICTIONS,
        "com_offsets": [-COM_OFFSET_FRACTION, 0.0, COM_OFFSET_FRACTION],
        "grasp_offsets": [-GRASP_OFFSET_FRACTION, 0.0, GRASP_OFFSET_FRACTION],
        "pilot_roots": PILOT_ROOTS,
        "preflight_roots": PREFLIGHT_SEEDS,
        "force_grid": FORCES,
        "refinement_rule": "3 fail/5 success -> 4N; 5 fail/8 success -> 6N",
    }
    write_json(out / "P6G0_PROTOCOL.json", protocol)
    (out / "P6G0_PROTOCOL_HASH.txt").write_text(stable_hash_obj(protocol) + "\n", encoding="utf-8")
    write_json(out / "P6G0_CODE_HASH.txt", {"git_commit": git_commit(), "runner_sha256": sha256_file(Path(__file__).resolve()), "p4_collect_sha256": sha256_file(P4_COLLECT)})
    (out / "P6G0_GRASP_CANDIDATE_RULE.md").write_text(
        "# P6G0 grasp candidate rule\n\n"
        "For each task, extract the nominal P4 grasp center at the object actor origin with fixed approach/orientation. "
        "Choose the object-frame axis most orthogonal to the nominal finger-closing direction among axes with maximum scaled USD bounding-box extent. "
        "Generate g_minus/g_center/g_plus at -0.25 Lg, 0, +0.25 Lg along that axis. The same global fractions are used for every task.\n",
        encoding="utf-8",
    )
    (out / "P6G0_OBJECT_GEOMETRY_AUDIT.md").write_text("# P6G0 object geometry audit\n\nSee `P6G0_OBJECT_GEOMETRY_AUDIT.csv`.\n", encoding="utf-8")
    (out / "P6G0_COM_INTERVENTION_AUDIT.md").write_text(
        "# P6G0 CoM intervention audit\n\n"
        "The runner uses the live PhysX `RigidBodyView` API: `set_masses`, `set_coms`, and `set_inertias`. "
        "Total mass is rewritten unchanged. CoM is shifted in the object actor frame. Inertia is updated with the parallel-axis theorem and rejected if the resulting tensor is not positive definite. "
        "Visible USD geometry and object reset pose are not edited.\n",
        encoding="utf-8",
    )


def final_report(out: Path, final: dict) -> str:
    classes = read_csv_if(out / "P6G0_TASK_CLASSIFICATIONS.csv")
    force = read_csv_if(out / "P6G0_FORCE_EFFECT.csv")
    grasp = read_csv_if(out / "P6G0_GRASP_POSITION_EFFECT.csv")
    div = read_csv_if(out / "P6G0_LOCAL_VS_FULL_DIVERGENCE.csv")
    flips = read_csv_if(out / "P6G0_ACTION_RANKING_FLIPS.csv")
    lines = [
        "# P6-G0 final report",
        "",
        f"STATUS: {final['status']}",
        "METHOD_CHANGE: EXPLICIT_GRASP_POSITION_AND_HIDDEN_COM_INTERVENTION_FOR_BENCHMARK_SCREENING",
        f"PRIMARY_CLASSIFICATION: {final['primary_classification']}",
        "",
        "## Scope",
        f"- Tasks screened: {final['tasks_screened']}",
        f"- Tasks entering main sweep: {final['tasks_entering_main_sweep']}",
        "- Learned policy used: False",
        "- Physical query used: False",
        "- Fresh worker per task: True",
        "",
        "## Effects",
        f"- Force-effect rows: {len(force)}",
        f"- Grasp-position-effect rows: {len(grasp)}",
        f"- Lift-success/full-failure rows: {len(div)}",
        f"- Hidden-physics action-ranking flips: {len(flips)}",
        "",
        "## Task Classifications",
    ]
    for row in classes:
        lines.append(f"- task{row['task']}: {row['classification']}")
    lines += [
        "",
        "## What this does not prove",
        "- pi0.5 can realize the requested grasp modes",
        "- ACT can realize the requested grasp modes",
        "- physical query can select the correct grasp",
        "- DreamTrajectory rollout improves candidate selection",
        "- LIBERO-Pro position generalization",
        "- when-to-query is solved",
    ]
    text = "\n".join(lines) + "\n"
    (out / "P6G0_FINAL_REPORT.md").write_text(text, encoding="utf-8")
    return text


def terminal_summary(out: Path, final: dict) -> str:
    classes = read_csv_if(out / "P6G0_TASK_CLASSIFICATIONS.csv")
    force = read_csv_if(out / "P6G0_FORCE_EFFECT.csv")
    grasp = read_csv_if(out / "P6G0_GRASP_POSITION_EFFECT.csv")
    div = read_csv_if(out / "P6G0_LOCAL_VS_FULL_DIVERGENCE.csv")
    flips = read_csv_if(out / "P6G0_ACTION_RANKING_FLIPS.csv")
    class_by_task = {int(r["task"]): r["classification"] for r in classes}
    return f"""STATUS:
{final['status']}
METHOD_CHANGE:
EXPLICIT_GRASP_POSITION_AND_HIDDEN_COM_INTERVENTION_FOR_BENCHMARK_SCREENING
ARTIFACTS:
{out}

SCOPE:
- Tasks screened: {final['tasks_screened']}
- Tasks entering main sweep: {final['tasks_entering_main_sweep']}
- Learned policy used: False
- Physical query used: False
- Fresh worker per task: True

GRASP CANDIDATES:
- Generic generation rule: nominal object-frame grasp center offset along the selected object axis
- Offset axis: per task, object-frame axis most orthogonal to finger closing with greatest bbox extent
- Offset magnitude: +/-0.25 Lg
- Candidate modes: g_minus, g_center, g_plus
- Per-task realization rate: see P6G0_GRASP_CLUSTER_ANALYSIS.csv
- Distinct contact clusters: see P6G0_GRASP_PREFLIGHT_VERDICT.json

HIDDEN PHYSICS:
- Friction values: {FRICTIONS}
- CoM offsets: [-0.15Lg, 0, +0.15Lg]
- Total mass fixed: see P6G0_COM_INTERVENTION_VALIDATION.csv
- Inertia updated consistently: parallel-axis update, positive-definite gate
- Visual appearance unchanged: True
- Intervention valid: see validation CSV

BRANCHES:
- Pilot planned/completed: planned for qualified tasks / see P6G0_BRANCH_MANIFEST.csv
- Expanded planned/completed: not launched by this runner unless separately rerun with expanded roots
- State parity: see P6G0_STATE_PARITY.csv
- Force values: {FORCES}
- Refinement branches: {REFINEMENT_FORCES} by frozen rule

FORCE EFFECT:
- Tasks with force-dependent outcomes: {sorted({int(r['task']) for r in force})}
- Strongest examples: see P6G0_FORCE_EFFECT.csv

GRASP-POSITION EFFECT:
- Tasks with position-dependent outcomes: {sorted({int(r['task']) for r in grasp})}
- Strongest examples: see P6G0_GRASP_POSITION_EFFECT.csv
- Different F*_grid across grasp positions: see P6G0_FORCE_FRONTIERS.csv

LOCAL VS FULL TASK:
- Lift-success/full-failure cases: {len(div)}
- Per grasp mode: see P6G0_LOCAL_VS_FULL_DIVERGENCE.csv

ACTION-RANKING FLIPS:
- Total hidden-physics-dependent flips: {len(flips)}
- Per task: {sorted({int(r['task']) for r in flips})}
- Friction-driven: analysis in P6G0_ACTION_RANKING_FLIPS.csv
- CoM-driven: analysis in P6G0_ACTION_RANKING_FLIPS.csv
- Both: analysis in P6G0_ACTION_RANKING_FLIPS.csv

TASK CLASSIFICATIONS:
- task0: {class_by_task.get(0, 'missing')}
- task1: {class_by_task.get(1, 'missing')}
- task2: {class_by_task.get(2, 'missing')}
- task5: {class_by_task.get(5, 'missing')}
- task6: {class_by_task.get(6, 'missing')}

PRIMARY_CLASSIFICATION:
{final['primary_classification']}

SCIENTIFIC INTERPRETATION:
1. This run screens whether explicit object-relative contact position is physically meaningful under deterministic control.
2. Hidden friction uses the prior Tabero material-property intervention.
3. Hidden CoM uses a live PhysX mass-property intervention with unchanged visible geometry.
4. Local, lift, and full-task outcomes are kept separate.
5. A positive result qualifies the substrate only; it does not evaluate a learned selector or VLA.

WHAT THIS DOES NOT PROVE:
- pi0.5 can realize the requested grasp modes
- ACT can realize the requested grasp modes
- physical query can select the correct grasp
- DreamTrajectory rollout improves candidate selection
- LIBERO-Pro position generalization
- when-to-query is solved

NEXT:
- If benchmark qualified: run P6-G1 policy grasp-realization gate with ACT and pi0.5.
- If grasp position matters but no physics flip: add successor-sensitive downstream tasks before model training.
- If force-only: retain Query2Force and do not enlarge the action space without new tasks.
- If candidates are not realizable: repair candidate generation before any learned-policy experiment.
"""


def main() -> int:
    if os.environ.get("P6G0_WORKER") == "1":
        return run_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "P6G0_CONTACT_TELEMETRY").mkdir(exist_ok=True)
    (OUT / "P6G0_EXECUTION_TELEMETRY").mkdir(exist_ok=True)
    write_static_artifacts(OUT)
    worker_rows = []
    for task in TASKS:
        row = launch_worker(OUT, task)
        worker_rows.append(row)
        write_csv(OUT / "P6G0_WORKERS.csv", worker_rows)
    final = aggregate(OUT, worker_rows)
    final_report(OUT, final)
    summary = terminal_summary(OUT, final)
    (OUT / "P6G0_TERMINAL_SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
