#!/usr/bin/env python3
"""P4 contact-conditioned shear probe collection.

This script lives inside the P4 result directory and does not modify Tabero
core, benchmarks, canonical force labels, or prior result folders.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TABERO = Path("/home/exouser/Tabero")
OUT = Path(os.environ.get("P4_OUT", Path(__file__).resolve().parents[1])).resolve()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)

TASK_OBJECTS = {
    0: "alphabet_soup_1",
    1: "cream_cheese_1",
    2: "salad_dressing_1",
    3: "bbq_sauce_1",
    5: "tomato_sauce_1",
    6: "butter_1",
    7: "milk_1",
    8: "chocolate_pudding_1",
    9: "orange_juice_1",
}
BASKET_NAME = "basket_1"
TASK_SUITE = "libero_object"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"

TASK_ID = int(os.environ["P4_TASK_ID"])
OBJ_NAME = TASK_OBJECTS[TASK_ID]
VARIANT = os.environ.get("P4_VARIANT", "P4B").upper()
N_SEEDS = int(os.environ.get("P4_N_SEEDS", "5"))
MUS = [float(x) for x in os.environ.get("P4_MUS", "0.2,0.5,1.0").split(",")]
SEED_OFFSET = int(os.environ.get("P4_SEED_OFFSET", "0"))
RESUME = os.environ.get("P4_RESUME", "1") not in ("0", "false", "False")

APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS, HOLD_STEPS = 45, 35, 70, 40
PREGRASP_Z, GRASP_Z = 0.10, 0.018
SERVO_STEP, SERVO_DEADBAND = 0.0006, 0.4
D_OPEN, D_CLOSED = 0.04, 0.0
POST_HOLD_STEPS = int(os.environ.get("P4_POST_HOLD_STEPS", "5"))
RETURN_STEPS = int(os.environ.get("P4_RETURN_STEPS", "10"))

P4A_FORCE_N = float(os.environ.get("P4A_FORCE_N", "4.0"))
P4B_BASE_FORCE_N = float(os.environ.get("P4B_BASE_FORCE_N", "3.0"))
P4B_PRELOAD_STEP_N = float(os.environ.get("P4B_PRELOAD_STEP_N", "0.5"))
P4B_PRELOAD_CAP_N = float(os.environ.get("P4B_PRELOAD_CAP_N", "4.5"))

FIXED_AMP_M = float(os.environ.get("P4_FIXED_AMP_MM", "2.0")) / 1000.0
ADAPTIVE_STEP_M = float(os.environ.get("P4_ADAPTIVE_STEP_MM", "0.2")) / 1000.0
MAX_DISP_M = float(os.environ.get("P4_MAX_DISP_MM", "2.0")) / 1000.0
RHO_IMPULSE_TARGET = float(os.environ.get("P4_RHO_IMPULSE_TARGET", "0.012"))
RHO_CAP = float(os.environ.get("P4_RHO_CAP", "0.08"))
RELATIVE_NORMAL_ALPHA = float(os.environ.get("P4_RELATIVE_NORMAL_ALPHA", "0.55"))
FINGER_FORCE_MIN_N = float(os.environ.get("P4_FINGER_FORCE_MIN_N", "0.15"))
CONTACT_FORCE_EPS_N = float(os.environ.get("P4_CONTACT_FORCE_EPS_N", "0.20"))
MARKER_NORM_STOP = float(os.environ.get("P4_MARKER_NORM_STOP", "0.12"))
MAX_OUT_STEPS = int(os.environ.get("P4_MAX_OUT_STEPS", "25"))
MAJOR_DISP_M = float(os.environ.get("P4_MAJOR_DISP_M", "0.01"))
ROT_MAJOR_RAD = float(os.environ.get("P4_ROT_MAJOR_RAD", "0.35"))

if VARIANT not in {"P4A", "P4B"}:
    raise SystemExit(f"Unsupported P4_VARIANT={VARIANT}; expected P4A or P4B")

os.chdir(TABERO)
sys.path.insert(0, str(TABERO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(TABERO / "benchmarks/datasets/libero/assembled_hdf5")
os.environ.setdefault("LIBERO_CONFIG_DIR", str(TABERO / "benchmarks/datasets/libero/config"))
os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(TABERO / "benchmarks/datasets/libero/USD"))

log_path = OUT / "logs" / f"collect_{VARIANT.lower()}_task{TASK_ID}.log"
sys.stdout = open(log_path, "a", buffering=1)
sys.stderr = sys.stdout


def _aa(q):
    w, x, y, z = q
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    if s < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (np.array([x, y, z], dtype=np.float32) / s) * angle


def _quat_angle(q0, q1) -> float:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    dot = abs(float(np.dot(q0, q1)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def _f(x, default=0.0):
    if x is None:
        return default
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return float(x.reshape(-1)[0].detach().cpu().item())
    except Exception:
        pass
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return default


def _pose_in_base(env, name):
    import isaaclab.utils.math as math_utils

    robot, obj = env.scene["robot"], env.scene[name]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )
    return pos_b[0].detach().cpu().numpy(), quat_b[0].detach().cpu().numpy()


def _unit(v, default=None):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n > 1e-9:
        return (v / n).astype(np.float32)
    if default is None:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(default, dtype=np.float32)


def _quat_apply_np(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, x, y, z = q
    qv = np.array([x, y, z])
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def _quat_inv_np(q):
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]]) / max(float(np.dot(q, q)), 1e-12)


def _frame_to_base(env, vec_w):
    robot = env.scene["robot"]
    root_q = robot.data.root_quat_w[0].detach().cpu().numpy()
    return _quat_apply_np(_quat_inv_np(root_q), vec_w)


def _current_contact_frame(env):
    left_q_w = env.scene["left_gripper_frame"].data.target_quat_w[0, 0].detach().cpu().numpy()
    n_w = _unit(_quat_apply_np(left_q_w, np.array([0.0, 0.0, 1.0])))
    x_w = _unit(_quat_apply_np(left_q_w, np.array([1.0, 0.0, 0.0])))
    y_w = _unit(_quat_apply_np(left_q_w, np.array([0.0, 1.0, 0.0])))
    n_b = _unit(_frame_to_base(env, n_w), default=[0.0, 0.0, 1.0])
    x_b = _unit(_frame_to_base(env, x_w), default=[1.0, 0.0, 0.0])
    y_b = _unit(_frame_to_base(env, y_w), default=[0.0, 1.0, 0.0])

    obj_b, _ = _pose_in_base(env, OBJ_NAME)
    basket_b, _ = _pose_in_base(env, BASKET_NAME)
    transport = np.asarray(basket_b - obj_b, dtype=np.float32)
    transport[2] = 0.0
    tangent = transport - float(np.dot(transport, n_b)) * n_b
    source = "projected_object_to_basket_transport"
    if np.linalg.norm(tangent) < 1e-4:
        tangent = y_b - float(np.dot(y_b, n_b)) * n_b
        source = "secondary_left_fingertip_y"
    if np.linalg.norm(tangent) < 1e-4:
        tangent = x_b - float(np.dot(x_b, n_b)) * n_b
        source = "secondary_left_fingertip_x"
    if np.linalg.norm(tangent) < 1e-4:
        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32) - n_b[0] * n_b
        source = "base_x_fallback_projected"
    tangent = _unit(tangent, default=[1.0, 0.0, 0.0])
    binormal = _unit(np.cross(n_b, tangent), default=[0.0, 0.0, 1.0])
    return n_b, tangent, binormal, source, obj_b, basket_b


def _make_action(eef_pos, eef_aa, d_pred, f_star, device):
    import torch

    a = torch.zeros((1, 13), device=device)
    a[0, 0:3] = torch.tensor(np.asarray(eef_pos, dtype=np.float32), device=device)
    a[0, 3:6] = torch.tensor(np.asarray(eef_aa, dtype=np.float32), device=device)
    a[0, 6] = float(d_pred)
    a[0, 9] = 0.5 * f_star
    a[0, 12] = 0.5 * f_star
    return a


def _force_servo(d_pred, f_meas, f_star):
    err = f_star - f_meas
    if abs(err) <= SERVO_DEADBAND:
        return float(np.clip(d_pred, D_CLOSED, D_OPEN))
    d_pred = d_pred - SERVO_STEP if err > 0 else d_pred + SERVO_STEP
    return float(np.clip(d_pred, D_CLOSED, D_OPEN))


def _interp(start, end, i, n):
    a = (i + 1) / max(n, 1)
    return (1 - a) * start + a * end


NOMINAL_MAT = None


def _apply_friction(env, name, mu):
    global NOMINAL_MAT
    import torch

    view = env.scene[name].root_physx_view
    if NOMINAL_MAT is None:
        NOMINAL_MAT = view.get_material_properties().clone()
    mats = NOMINAL_MAT.clone()
    mats[..., 0] = float(mu)
    mats[..., 1] = float(mu)
    view.set_material_properties(mats, torch.arange(mats.shape[0], dtype=torch.int32))
    got = view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
    return float(got[:, 0].mean())


def _dbg(env):
    try:
        return env.action_manager.get_term("arm_action").debug_info or {}
    except Exception:
        return {}


def _tactile_summaries(obs):
    out = {
        "marker_mean": 0.0,
        "marker_tangential": 0.0,
        "tactile_ok": 0,
    }
    try:
        mm = obs["policy"]["gripper_marker_motion"][0].detach().cpu().numpy()
        cur, init = mm[:, 1], mm[:, 0]
        disp = cur - init
        mag = np.linalg.norm(disp, axis=-1)
        out["marker_mean"] = float(mag.mean())
        tang_vec = disp.mean(axis=1)
        out["marker_tangential"] = float(np.linalg.norm(tang_vec, axis=-1).mean())
        out["tactile_ok"] = 1
    except Exception:
        pass
    return out


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _append_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    if not rows:
        return
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for row in rows:
            w.writerow(row)


@dataclass
class ProbeStep:
    trial_id: str
    task_id: int
    object_id: str
    seed_idx: int
    friction: float
    timestamp: int
    probe_variant: str
    probe_phase: str
    step: int
    t_s: float
    force_target: float
    measured_squeeze: float
    target_normal_force: float
    measured_fn: float
    measured_ft: float
    ft_over_fn: float
    left_fx: float
    left_fy: float
    left_fz: float
    right_fx: float
    right_fy: float
    right_fz: float
    force_imbalance: float
    force_imbalance_ratio: float
    gripper_opening: float
    contact_normal_x: float
    contact_normal_y: float
    contact_normal_z: float
    contact_tangent_x: float
    contact_tangent_y: float
    contact_tangent_z: float
    commanded_tangent_increment_mm: float
    accumulated_displacement_mm: float
    marker_motion: float
    marker_tangential: float
    marker_velocity: float
    marker_loading_unloading: float
    contact_left: int
    contact_right: int
    contact_state: str
    stop_trigger: str
    eef_x: float
    eef_y: float
    eef_z: float
    object_x_priv: float
    object_y_priv: float
    object_z_priv: float
    object_qw_priv: float
    object_qx_priv: float
    object_qy_priv: float
    object_qz_priv: float
    tactile_ok: int


def _existing_done(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open() as f:
        return {r["trial_id"] for r in csv.DictReader(f)}


def run_probe_episode(env, *, seed_idx: int, mu: float, trial_id: str, dt: float):
    try:
        obs, _ = env.reset(seed=int(seed_idx))
    except TypeError:
        import torch

        torch.manual_seed(int(seed_idx))
        np.random.seed(int(seed_idx))
        obs, _ = env.reset()
    applied = _apply_friction(env, OBJ_NAME, mu)

    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    ee_pos = eef0[:3].copy()
    eef_aa = _aa(eef0[3:7])
    obj0_b, _ = _pose_in_base(env, OBJ_NAME)
    pregrasp = obj0_b.copy()
    pregrasp[2] += PREGRASP_Z
    grasp = obj0_b.copy()
    grasp[2] += GRASP_Z

    d_pred = D_OPEN
    cmd_pos = ee_pos.copy()
    step = 0
    rows: list[ProbeStep] = []

    preload_target = P4A_FORCE_N if VARIANT == "P4A" else P4B_BASE_FORCE_N
    stop_trigger = "completed"
    probe_t0 = probe_t1 = None
    obj_at_probe0 = None
    obj_q_at_probe0 = None
    obj_disp_probe = 0.0
    obj_rot_probe = 0.0
    contact_lost_probe = 0
    probe_failure = 0
    dropped = 0
    major_disturbance = 0
    accumulated_m = 0.0
    rho_impulse = 0.0
    preprobe_fn = np.nan
    preprobe_marker = np.nan
    preprobe_ft = np.nan
    prev_marker = None
    marker_at_forward0 = None

    f_hist: list[float] = []
    fn_hist: list[float] = []
    ft_hist: list[float] = []
    rho_hist: list[float] = []
    imb_hist: list[float] = []
    imb_ratio_hist: list[float] = []
    mk_hist: list[float] = []
    mk_tang_hist: list[float] = []
    mk_vel_hist: list[float] = []
    grip_hist: list[float] = []
    disp_hist: list[float] = []

    contact_normal, tangent, binormal, tangent_source, obj_b, basket_b = _current_contact_frame(env)

    def observe_and_record(phase, is_probe, inc_mm):
        nonlocal step, d_pred, dropped, probe_t0, probe_t1, obj_at_probe0, obj_q_at_probe0
        nonlocal obj_disp_probe, obj_rot_probe, contact_lost_probe, probe_failure, major_disturbance
        nonlocal prev_marker, marker_at_forward0, stop_trigger, rho_impulse, contact_normal, tangent

        force_cmd = preload_target if phase not in ("approach", "descend") else 0.0
        action = _make_action(cmd_pos, eef_aa, d_pred, force_cmd, env.device)
        obs2, _, term, trunc, _ = env.step(action)
        step += 1
        t_s = step * dt

        f = obs2["policy"]["gripper_net_force"][0]
        if f.ndim == 3:
            f = f[-1]
        fL = f[0].detach().cpu().numpy().astype(np.float32)
        fR = f[1].detach().cpu().numpy().astype(np.float32)
        f_sq = _f(_dbg(env).get("f_sq_meas"), 0.0)
        fn = float(2.0 * min(abs(fL[2]), abs(fR[2])))
        ft = float(np.linalg.norm(np.array([fL[0] + fR[0], fL[1] + fR[1]], dtype=np.float32)))
        rho = float(ft / max(fn, 1e-6))
        imb = float(abs(np.linalg.norm(fL) - np.linalg.norm(fR)))
        imb_ratio = float(imb / max(np.linalg.norm(fL) + np.linalg.norm(fR), 1e-6))
        left_contact = int(np.linalg.norm(fL) > FINGER_FORCE_MIN_N and abs(fL[2]) > FINGER_FORCE_MIN_N)
        right_contact = int(np.linalg.norm(fR) > FINGER_FORCE_MIN_N and abs(fR[2]) > FINGER_FORCE_MIN_N)
        contact_state = "bilateral" if left_contact and right_contact else ("unilateral" if left_contact or right_contact else "none")
        try:
            gripper_opening = _f(obs2["policy"]["gripper_pos"][0], np.nan)
        except Exception:
            gripper_opening = np.nan

        if phase not in ("approach", "descend"):
            d_pred = _force_servo(d_pred, f_sq, preload_target)

        tac = _tactile_summaries(obs2)
        marker_vel = 0.0 if prev_marker is None else (tac["marker_mean"] - prev_marker) / dt
        prev_marker = tac["marker_mean"]
        if marker_at_forward0 is None and is_probe:
            marker_at_forward0 = tac["marker_mean"]
        marker_du = 0.0 if marker_at_forward0 is None else tac["marker_mean"] - marker_at_forward0

        obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
        obj_q = env.scene[OBJ_NAME].data.root_quat_w[0].detach().cpu().numpy()
        ee = obs2["policy"]["eef_pose"][0].detach().cpu().numpy()[:3]
        try:
            dropped = int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
        except Exception:
            pass

        if is_probe:
            if probe_t0 is None:
                probe_t0 = t_s
                obj_at_probe0 = obj_p.copy()
                obj_q_at_probe0 = obj_q.copy()
            probe_t1 = t_s
            f_hist.append(f_sq)
            fn_hist.append(fn)
            ft_hist.append(ft)
            rho_hist.append(rho)
            rho_impulse += rho * dt
            imb_hist.append(imb)
            imb_ratio_hist.append(imb_ratio)
            mk_hist.append(tac["marker_mean"])
            mk_tang_hist.append(tac["marker_tangential"])
            mk_vel_hist.append(marker_vel)
            grip_hist.append(gripper_opening)
            disp_hist.append(accumulated_m * 1000.0)
            obj_disp_probe = max(obj_disp_probe, float(np.linalg.norm(obj_p - obj_at_probe0)))
            obj_rot_probe = max(obj_rot_probe, _quat_angle(obj_q_at_probe0, obj_q))
            if contact_state != "bilateral" or fn < CONTACT_FORCE_EPS_N or dropped:
                contact_lost_probe = 1
                probe_failure = 1
                if stop_trigger == "completed":
                    stop_trigger = "hard_contact_loss"
            if obj_disp_probe > MAJOR_DISP_M or obj_rot_probe > ROT_MAJOR_RAD:
                major_disturbance = 1

        rows.append(
            ProbeStep(
                trial_id=trial_id,
                task_id=TASK_ID,
                object_id=OBJ_NAME,
                seed_idx=seed_idx,
                friction=mu,
                timestamp=step,
                probe_variant=VARIANT,
                probe_phase=phase,
                step=step,
                t_s=t_s,
                force_target=float(force_cmd),
                measured_squeeze=f_sq,
                target_normal_force=float(preload_target),
                measured_fn=fn,
                measured_ft=ft,
                ft_over_fn=rho,
                left_fx=float(fL[0]),
                left_fy=float(fL[1]),
                left_fz=float(fL[2]),
                right_fx=float(fR[0]),
                right_fy=float(fR[1]),
                right_fz=float(fR[2]),
                force_imbalance=imb,
                force_imbalance_ratio=imb_ratio,
                gripper_opening=gripper_opening,
                contact_normal_x=float(contact_normal[0]),
                contact_normal_y=float(contact_normal[1]),
                contact_normal_z=float(contact_normal[2]),
                contact_tangent_x=float(tangent[0]),
                contact_tangent_y=float(tangent[1]),
                contact_tangent_z=float(tangent[2]),
                commanded_tangent_increment_mm=float(inc_mm),
                accumulated_displacement_mm=float(accumulated_m * 1000.0),
                marker_motion=tac["marker_mean"],
                marker_tangential=tac["marker_tangential"],
                marker_velocity=marker_vel,
                marker_loading_unloading=marker_du,
                contact_left=left_contact,
                contact_right=right_contact,
                contact_state=contact_state,
                stop_trigger=stop_trigger,
                eef_x=float(ee[0]),
                eef_y=float(ee[1]),
                eef_z=float(ee[2]),
                object_x_priv=float(obj_p[0]),
                object_y_priv=float(obj_p[1]),
                object_z_priv=float(obj_p[2]),
                object_qw_priv=float(obj_q[0]),
                object_qx_priv=float(obj_q[1]),
                object_qy_priv=float(obj_q[2]),
                object_qz_priv=float(obj_q[3]),
                tactile_ok=tac["tactile_ok"],
            )
        )
        return obs2, bool(term[0].item()) or bool(trunc[0].item()), contact_state, fn, ft, rho, tac["marker_mean"]

    terminated = False
    for phase, n_steps, target in [
        ("approach", APPROACH_STEPS, pregrasp),
        ("descend", DESCEND_STEPS, grasp),
        ("close", CLOSE_STEPS, grasp),
        ("hold", HOLD_STEPS, grasp),
    ]:
        start = cmd_pos.copy()
        for i in range(n_steps):
            cmd_pos = _interp(start, target, i, n_steps)
            if phase in ("approach", "descend"):
                d_pred = D_OPEN
            obs, terminated, contact_state, fn, _, _, _ = observe_and_record(phase, False, 0.0)
            if VARIANT == "P4B" and phase == "hold" and i in (14, 24, 34):
                if contact_state != "bilateral" or fn < 0.65 * preload_target:
                    preload_target = min(P4B_PRELOAD_CAP_N, preload_target + P4B_PRELOAD_STEP_N)
            if terminated:
                break
        if terminated:
            break

    preprobe_fn = float(np.mean([r.measured_fn for r in rows if r.probe_phase == "hold"][-10:])) if rows else np.nan
    preprobe_marker = float(np.mean([r.marker_motion for r in rows if r.probe_phase == "hold"][-10:])) if rows else np.nan
    preprobe_ft = float(np.mean([r.measured_ft for r in rows if r.probe_phase == "hold"][-10:])) if rows else np.nan

    contact_normal, tangent, binormal, tangent_source, obj_b, basket_b = _current_contact_frame(env)
    probe_start = grasp.copy()
    cmd_pos = probe_start.copy()

    if not terminated:
        if VARIANT == "P4A":
            target = probe_start + tangent * FIXED_AMP_M
            start = cmd_pos.copy()
            for i in range(10):
                prev = cmd_pos.copy()
                cmd_pos = _interp(start, target, i, 10)
                accumulated_m = float(np.linalg.norm(cmd_pos - probe_start))
                obs, terminated, _, fn, _, rho, marker = observe_and_record(
                    "probe_out", True, float(np.linalg.norm(cmd_pos - prev) * 1000.0)
                )
                if stop_trigger != "completed" or terminated:
                    break
                if np.isfinite(preprobe_fn) and fn < RELATIVE_NORMAL_ALPHA * preprobe_fn:
                    stop_trigger = "relative_normal_drop"
                    break
                if rho >= RHO_CAP:
                    stop_trigger = "shear_ratio_cap"
                    break
                if np.isfinite(preprobe_marker) and abs(marker - preprobe_marker) / max(abs(preprobe_marker), 1e-6) > MARKER_NORM_STOP:
                    stop_trigger = "marker_motion_budget"
                    break
        else:
            for _ in range(MAX_OUT_STEPS):
                if accumulated_m >= MAX_DISP_M:
                    stop_trigger = "max_displacement_cap"
                    break
                inc = min(ADAPTIVE_STEP_M, MAX_DISP_M - accumulated_m)
                cmd_pos = cmd_pos + tangent * inc
                accumulated_m += inc
                obs, terminated, _, fn, _, rho, marker = observe_and_record("probe_out", True, inc * 1000.0)
                if stop_trigger != "completed" or terminated:
                    break
                if np.isfinite(preprobe_fn) and fn < RELATIVE_NORMAL_ALPHA * preprobe_fn:
                    stop_trigger = "relative_normal_drop"
                    break
                if rho >= RHO_CAP:
                    stop_trigger = "shear_ratio_cap"
                    break
                if rho_impulse >= RHO_IMPULSE_TARGET:
                    stop_trigger = "normalized_shear_impulse"
                    break
                if np.isfinite(preprobe_marker) and abs(marker - preprobe_marker) / max(abs(preprobe_marker), 1e-6) > MARKER_NORM_STOP:
                    stop_trigger = "marker_motion_budget"
                    break

        return_start = cmd_pos.copy()
        for i in range(RETURN_STEPS):
            cmd_pos = _interp(return_start, probe_start, i, RETURN_STEPS)
            accumulated_m = float(np.linalg.norm(cmd_pos - probe_start))
            obs, terminated, _, _, _, _, _ = observe_and_record("probe_back", True, 0.0)
            if terminated:
                break
        if not terminated:
            for _ in range(POST_HOLD_STEPS):
                cmd_pos = probe_start.copy()
                accumulated_m = 0.0
                obs, terminated, _, _, _, _, _ = observe_and_record("probe_hold", True, 0.0)
                if terminated:
                    break

    def peak(xs):
        return float(np.max(xs)) if xs else np.nan

    def mean(xs):
        return float(np.mean(xs)) if xs else np.nan

    def last(xs):
        return float(xs[-1]) if xs else np.nan

    def hyst(xs):
        return float(xs[-1] - xs[0]) if len(xs) >= 2 else np.nan

    def slope(xs):
        return float((xs[-1] - xs[0]) / max(len(xs) - 1, 1) / dt) if len(xs) >= 2 else np.nan

    half = max(1, len(ft_hist) // 2)
    rec = {
        "trial_id": trial_id,
        "probe_source": "REAL_CONTACT_CONDITIONED_PROBE",
        "probe_id": VARIANT,
        "probe_variant": VARIANT,
        "task_id": TASK_ID,
        "object_id": OBJ_NAME,
        "seed_idx": seed_idx,
        "friction": mu,
        "friction_applied": applied,
        "target_preload_N": preload_target,
        "preload_rule": "fixed_4N" if VARIANT == "P4A" else "start_3N_minimal_bilateral_contact_cap4.5N",
        "contact_frame_source": tangent_source,
        "contact_normal_x": float(contact_normal[0]),
        "contact_normal_y": float(contact_normal[1]),
        "contact_normal_z": float(contact_normal[2]),
        "contact_tangent_x": float(tangent[0]),
        "contact_tangent_y": float(tangent[1]),
        "contact_tangent_z": float(tangent[2]),
        "contact_binormal_x": float(binormal[0]),
        "contact_binormal_y": float(binormal[1]),
        "contact_binormal_z": float(binormal[2]),
        "object_to_basket_x": float(basket_b[0] - obj_b[0]),
        "object_to_basket_y": float(basket_b[1] - obj_b[1]),
        "object_to_basket_z": float(basket_b[2] - obj_b[2]),
        "actual_probe_displacement_mm": peak(disp_hist),
        "probe_path_mm": float(sum(abs(x) for x in np.diff([0.0] + disp_hist))) if disp_hist else 0.0,
        "probe_duration_s": None if probe_t0 is None else (probe_t1 - probe_t0 + dt),
        "stop_trigger": stop_trigger,
        "probe_failure": probe_failure,
        "contact_lost_probe": contact_lost_probe,
        "dropped": dropped,
        "obj_disp_probe_m": obj_disp_probe,
        "obj_rot_probe_rad": obj_rot_probe,
        "major_disturbance": major_disturbance,
        "tactile_ok": int(any(r.tactile_ok for r in rows)),
        "preprobe_normal_force": preprobe_fn,
        "preprobe_marker_motion": preprobe_marker,
        "preprobe_tangential_force": preprobe_ft,
        "f_meas_mean": mean(f_hist),
        "normal_force_mean": mean(fn_hist),
        "normal_force_peak": peak(fn_hist),
        "normal_force_hyst": hyst(fn_hist),
        "normal_loading_slope": slope(fn_hist[:half]),
        "normal_unloading_slope": slope(fn_hist[half:]),
        "ftan_mean": mean(ft_hist),
        "ftan_peak": peak(ft_hist),
        "ftan_hyst": hyst(ft_hist),
        "rho_mean": mean(rho_hist),
        "rho_peak": peak(rho_hist),
        "rho_hyst": hyst(rho_hist),
        "rho_impulse": rho_impulse,
        "imb_mean": mean(imb_hist),
        "imb_peak": peak(imb_hist),
        "imb_ratio_mean": mean(imb_ratio_hist),
        "imb_ratio_peak": peak(imb_ratio_hist),
        "marker_mean": mean(mk_hist),
        "marker_peak": peak(mk_hist),
        "marker_tangential_mean": mean(mk_tang_hist),
        "marker_tangential_peak": peak(mk_tang_hist),
        "marker_hyst": hyst(mk_hist),
        "marker_unloading": last(mk_hist) - mean(mk_hist) if mk_hist else np.nan,
        "marker_vel_abs_peak": float(np.max(np.abs(mk_vel_hist))) if mk_vel_hist else np.nan,
        "residual_marker_displacement": last(mk_hist),
        "gripper_opening_mean": mean(grip_hist),
    }
    print(
        f"summary {trial_id} task={TASK_ID} mu={mu} variant={VARIANT} fail={probe_failure} "
        f"ret={1-contact_lost_probe} stop={stop_trigger} disp_mm={rec['actual_probe_displacement_mm']:.3f} "
        f"rho_peak={rec['rho_peak']:.4f}",
        flush=True,
    )
    return rows, rec


def main():
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(TASK_SUITE, TASK_ID)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = float(os.environ.get("P4_EPISODE_S", "25"))
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        print("env ready", {"task": TASK_ID, "object": OBJ_NAME, "dt": dt, "n_seeds": N_SEEDS, "variant": VARIANT}, flush=True)

        meta = {
            "task_id": TASK_ID,
            "object_id": OBJ_NAME,
            "dt": dt,
            "n_seeds": N_SEEDS,
            "mus": MUS,
            "variant": VARIANT,
            "method_change": "NONE",
            "probe_rule": "contact_conditioned_shear",
            "task_specific_probe_lookup": False,
        }
        (OUT / f"{VARIANT}_TASK{TASK_ID}_COLLECT_META.json").write_text(json.dumps(meta, indent=2) + "\n")

        task_csv = OUT / f"{VARIANT}_TASK{TASK_ID}_PROBE.csv"
        episodes_csv = OUT / f"{VARIANT}_EPISODES.csv"
        steps_csv = OUT / f"{VARIANT}_TIMESTEPS.csv"
        done = _existing_done(task_csv) if RESUME else set()
        task_rows: list[dict] = []
        if RESUME and task_csv.exists() and task_csv.stat().st_size > 0:
            with task_csv.open() as f:
                task_rows.extend(list(csv.DictReader(f)))
        step_fields = list(ProbeStep.__dataclass_fields__.keys())

        for s in range(SEED_OFFSET, SEED_OFFSET + N_SEEDS):
            for mu in MUS:
                trial_id = f"p4_{VARIANT.lower()}_t{TASK_ID}_s{s}_mu{mu:g}"
                if trial_id in done:
                    print("skip", trial_id, flush=True)
                    continue
                rows, rec = run_probe_episode(env, seed_idx=s, mu=mu, trial_id=trial_id, dt=dt)
                task_rows.append(rec)
                fields = list(rec.keys())
                _write_csv(task_csv, task_rows, fields)
                _append_csv(episodes_csv, [rec], fields)
                _append_csv(steps_csv, [asdict(r) for r in rows], step_fields)
                done.add(trial_id)

        os._exit(0)
    except Exception as exc:
        print("EXCEPTION", repr(exc), flush=True)
        print(traceback.format_exc(), flush=True)
        (OUT / f"{VARIANT}_TASK{TASK_ID}_COLLECT_ERROR.json").write_text(
            json.dumps({"error": repr(exc), "trace": traceback.format_exc()}, indent=2)
        )
        try:
            simulation_app.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
