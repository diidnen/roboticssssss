#!/usr/bin/env python3
"""P4-R2 contact-feasible multi-primitive probe collection.

Result-scoped implementation. It does not modify Tabero core, labels,
candidate forces, controller calibration, benchmark tasks, or checkpoints.
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
OUT = Path(os.environ.get("P4R2_OUT", Path(__file__).resolve().parents[1])).resolve()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "P4R2_TIMESTEP_TELEMETRY").mkdir(exist_ok=True)

TASK_OBJECTS = {0: "alphabet_soup_1", 1: "cream_cheese_1", 2: "salad_dressing_1", 5: "tomato_sauce_1", 6: "butter_1"}
BASKET_NAME = "basket_1"
TASK_SUITE = "libero_object"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"

TASK_ID = int(os.environ["P4R2_TASK_ID"])
OBJ_NAME = TASK_OBJECTS[TASK_ID]
PRIMITIVE = os.environ.get("P4R2_PRIMITIVE", "S").upper()
SPLIT = os.environ.get("P4R2_SPLIT", "development").lower()
N_SEEDS = int(os.environ.get("P4R2_N_SEEDS", "5"))
MUS = [float(x) for x in os.environ.get("P4R2_MUS", "0.2,0.5,1.0").split(",")]
SEED_OFFSET = int(os.environ.get("P4R2_SEED_OFFSET", "1000"))
RESUME = os.environ.get("P4R2_RESUME", "1") not in ("0", "false", "False")

APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS, HOLD_STEPS = 45, 35, 70, 40
PREGRASP_Z, GRASP_Z = 0.10, 0.018
SERVO_STEP, SERVO_DEADBAND = 0.0006, 0.4
D_OPEN, D_CLOSED = 0.04, 0.0
POST_HOLD_STEPS = int(os.environ.get("P4R2_POST_HOLD_STEPS", "5"))
RETURN_STEPS = int(os.environ.get("P4R2_RETURN_STEPS", "10"))

BASE_FORCE_N = float(os.environ.get("P4R2_BASE_FORCE_N", "3.0"))
PRELOAD_STEP_N = float(os.environ.get("P4R2_PRELOAD_STEP_N", "0.5"))
PRELOAD_CAP_N = float(os.environ.get("P4R2_PRELOAD_CAP_N", "4.5"))
S_STEP_M = float(os.environ.get("P4R2_S_STEP_MM", "0.1")) / 1000.0
L_STEP_M = float(os.environ.get("P4R2_L_STEP_MM", "0.1")) / 1000.0
S_MAX_DISP_M = float(os.environ.get("P4R2_S_MAX_DISP_MM", "2.0")) / 1000.0
L_MAX_DISP_M = float(os.environ.get("P4R2_L_MAX_DISP_MM", "1.2")) / 1000.0
S_RHO_IMPULSE_TARGET = float(os.environ.get("P4R2_S_RHO_IMPULSE_TARGET", "0.010"))
L_LOAD_IMPULSE_TARGET = float(os.environ.get("P4R2_L_LOAD_IMPULSE_TARGET", "0.060"))
RHO_CAP = float(os.environ.get("P4R2_RHO_CAP", "0.08"))
RELATIVE_NORMAL_ALPHA = float(os.environ.get("P4R2_RELATIVE_NORMAL_ALPHA", "0.55"))
FINGER_FORCE_MIN_N = float(os.environ.get("P4R2_FINGER_FORCE_MIN_N", "0.15"))
CONTACT_FORCE_EPS_N = float(os.environ.get("P4R2_CONTACT_FORCE_EPS_N", "0.20"))
MARKER_NORM_STOP = float(os.environ.get("P4R2_MARKER_NORM_STOP", "0.12"))
MAX_OUT_STEPS = int(os.environ.get("P4R2_MAX_OUT_STEPS", "25"))
MAJOR_DISP_M = float(os.environ.get("P4R2_MAJOR_DISP_M", "0.01"))
ROT_MAJOR_RAD = float(os.environ.get("P4R2_ROT_MAJOR_RAD", "0.35"))
L_IMBALANCE_CAP = float(os.environ.get("P4R2_L_IMBALANCE_CAP", "0.75"))
L_SUPPORT_LOAD_BUDGET_N = float(os.environ.get("P4R2_L_SUPPORT_LOAD_BUDGET_N", "1.25"))

S_LEAKAGE_MAX = float(os.environ.get("P4R2_S_LEAKAGE_MAX", "0.10"))
S_ALIGNMENT_MIN = float(os.environ.get("P4R2_S_ALIGNMENT_MIN", "0.10"))
S_NULL_CONDITION_MAX = float(os.environ.get("P4R2_S_NULL_CONDITION_MAX", "10000000.0"))
PRELOAD_BALANCE_MAX = float(os.environ.get("P4R2_PRELOAD_BALANCE_MAX", "0.65"))
PRELOAD_FN_MIN_FACTOR = float(os.environ.get("P4R2_PRELOAD_FN_MIN_FACTOR", "0.65"))
S_PREPROBE_FT_MIN_N = float(os.environ.get("P4R2_S_PREPROBE_FT_MIN_N", "0.01"))

S_INFO_RHO_IMPULSE_MIN = float(os.environ.get("P4R2_S_INFO_RHO_IMPULSE_MIN", "0.0025"))
S_INFO_MARKER_TANGENTIAL_MIN = float(os.environ.get("P4R2_S_INFO_MARKER_TANGENTIAL_MIN", "0.015"))
L_INFO_LOAD_IMPULSE_MIN = float(os.environ.get("P4R2_L_INFO_LOAD_IMPULSE_MIN", "0.015"))
L_INFO_FORCE_DELTA_MIN_N = float(os.environ.get("P4R2_L_INFO_FORCE_DELTA_MIN_N", "0.08"))

if PRIMITIVE not in {"S", "L"}:
    raise SystemExit(f"Unsupported P4R2_PRIMITIVE={PRIMITIVE}; expected S or L")

os.chdir(TABERO)
sys.path.insert(0, str(TABERO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(TABERO / "benchmarks/datasets/libero/assembled_hdf5")
os.environ.setdefault("LIBERO_CONFIG_DIR", str(TABERO / "benchmarks/datasets/libero/config"))
os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(TABERO / "benchmarks/datasets/libero/USD"))

log_path = OUT / "logs" / f"collect_{SPLIT}_{PRIMITIVE.lower()}_task{TASK_ID}.log"
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
        robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w, obj.data.root_quat_w
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


def _frame_axis_base(env, frame_name: str, axis):
    q_w = env.scene[frame_name].data.target_quat_w[0, 0].detach().cpu().numpy()
    return _unit(_frame_to_base(env, _quat_apply_np(q_w, np.asarray(axis, dtype=np.float64))))


def _transport_direction(env):
    obj_b, _ = _pose_in_base(env, OBJ_NAME)
    basket_b, _ = _pose_in_base(env, BASKET_NAME)
    transport = np.asarray(basket_b - obj_b, dtype=np.float32)
    transport[2] = 0.0
    return _unit(transport, default=[1.0, 0.0, 0.0]), obj_b, basket_b


def _contact_geometry(env):
    n_left = _frame_axis_base(env, "left_gripper_frame", [0.0, 0.0, 1.0])
    n_right = _frame_axis_base(env, "right_gripper_frame", [0.0, 0.0, 1.0])
    left_x = _frame_axis_base(env, "left_gripper_frame", [1.0, 0.0, 0.0])
    left_y = _frame_axis_base(env, "left_gripper_frame", [0.0, 1.0, 0.0])
    downstream, obj_b, basket_b = _transport_direction(env)

    nmat = np.vstack([n_left.astype(np.float64), n_right.astype(np.float64)])
    gram = nmat @ nmat.T
    try:
        sv = np.linalg.svd(nmat, compute_uv=False)
        condition = float(sv[0] / max(sv[-1], 1e-6))
    except Exception:
        condition = float("inf")
    projector = np.eye(3) - nmat.T @ np.linalg.inv(gram + 1e-4 * np.eye(2)) @ nmat
    candidate = projector @ downstream
    source = "bilateral_nullspace_projected_downstream"
    if np.linalg.norm(candidate) < 1e-5:
        candidate = np.cross(n_left, n_right)
        source = "bilateral_normal_cross"
    if np.linalg.norm(candidate) < 1e-5:
        candidate = projector @ left_y
        source = "bilateral_nullspace_left_y"
    if np.linalg.norm(candidate) < 1e-5:
        candidate = projector @ left_x
        source = "bilateral_nullspace_left_x"
    shear = _unit(candidate, default=[1.0, 0.0, 0.0])
    leakage_left = float(abs(np.dot(n_left, shear)))
    leakage_right = float(abs(np.dot(n_right, shear)))
    return {
        "n_left": n_left,
        "n_right": n_right,
        "shear": shear,
        "support": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "downstream": downstream,
        "obj_b": obj_b,
        "basket_b": basket_b,
        "source": source,
        "leakage_left": leakage_left,
        "leakage_right": leakage_right,
        "max_normal_leakage": max(leakage_left, leakage_right),
        "downstream_alignment": float(abs(np.dot(shear, downstream))),
        "null_condition": condition,
        "bilateral_normal_consistency": float(abs(np.dot(n_left, n_right))),
    }


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
    return float(np.clip(d_pred - SERVO_STEP if err > 0 else d_pred + SERVO_STEP, D_CLOSED, D_OPEN))


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
    out = {"marker_mean": 0.0, "marker_tangential": 0.0, "tactile_ok": 0}
    try:
        mm = obs["policy"]["gripper_marker_motion"][0].detach().cpu().numpy()
        disp = mm[:, 1] - mm[:, 0]
        out["marker_mean"] = float(np.linalg.norm(disp, axis=-1).mean())
        out["marker_tangential"] = float(np.linalg.norm(disp.mean(axis=1), axis=-1).mean())
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
    split: str
    task_id: int
    object_id: str
    seed_idx: int
    friction: float
    timestamp: int
    primitive: str
    selector_output: str
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
    left_normal_x: float
    left_normal_y: float
    left_normal_z: float
    right_normal_x: float
    right_normal_y: float
    right_normal_z: float
    commanded_dir_x: float
    commanded_dir_y: float
    commanded_dir_z: float
    downstream_dir_x: float
    downstream_dir_y: float
    downstream_dir_z: float
    commanded_increment_mm: float
    accumulated_displacement_mm: float
    eef_displacement_mm: float
    object_displacement_m: float
    object_rotation_rad: float
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


def _selector(geom, preprobe_fn, preprobe_ft, preprobe_imb_ratio, contact_state):
    reasons = []
    s_feasible = True
    if contact_state != "bilateral":
        s_feasible = False
        reasons.append("preprobe_not_bilateral")
    if not np.isfinite(preprobe_fn) or preprobe_fn < PRELOAD_FN_MIN_FACTOR * BASE_FORCE_N:
        s_feasible = False
        reasons.append("weak_preload")
    if np.isfinite(preprobe_imb_ratio) and preprobe_imb_ratio > PRELOAD_BALANCE_MAX:
        s_feasible = False
        reasons.append("preload_imbalance")
    if not np.isfinite(preprobe_ft) or preprobe_ft < S_PREPROBE_FT_MIN_N:
        s_feasible = False
        reasons.append("low_preprobe_tangential_contact")
    if geom["max_normal_leakage"] > S_LEAKAGE_MAX:
        s_feasible = False
        reasons.append("normal_leakage")
    if geom["downstream_alignment"] < S_ALIGNMENT_MIN:
        s_feasible = False
        reasons.append("low_downstream_alignment")
    if geom["null_condition"] > S_NULL_CONDITION_MAX and geom["bilateral_normal_consistency"] < 0.95:
        s_feasible = False
        reasons.append("unstable_nullspace")
    if s_feasible:
        return "S", "selected_shear", True
    l_feasible = contact_state == "bilateral" and np.isfinite(preprobe_fn) and preprobe_fn >= CONTACT_FORCE_EPS_N
    if l_feasible:
        return "L", "fallback_support_transfer_after_" + "+".join(reasons), False
    return "NO_FEASIBLE_PROBE", "neither_feasible_" + "+".join(reasons), False


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
    eef_aa = _aa(eef0[3:7])
    obj0_b, _ = _pose_in_base(env, OBJ_NAME)
    pregrasp = obj0_b.copy()
    pregrasp[2] += PREGRASP_Z
    grasp = obj0_b.copy()
    grasp[2] += GRASP_Z

    d_pred = D_OPEN
    cmd_pos = eef0[:3].copy()
    step = 0
    rows: list[ProbeStep] = []
    preload_target = BASE_FORCE_N
    stop_trigger = "completed"
    probe_t0 = probe_t1 = None
    obj_at_probe0 = obj_q_at_probe0 = eef_at_probe0 = None
    obj_disp_probe = obj_rot_probe = eef_disp_probe = 0.0
    contact_lost_probe = probe_failure = dropped = major_disturbance = return_completed = primitive_infeasible = 0
    accumulated_m = rho_impulse = load_impulse = support_force_delta_peak = 0.0
    prev_marker = marker_at_forward0 = None
    selector_output = "UNSET"

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
    load_hist: list[float] = []

    geom = _contact_geometry(env)
    motion_dir = geom["shear"] if PRIMITIVE == "S" else geom["support"]
    probe_start = grasp.copy()

    def observe_and_record(phase, is_probe, inc_mm):
        nonlocal step, d_pred, dropped, probe_t0, probe_t1, obj_at_probe0, obj_q_at_probe0, eef_at_probe0
        nonlocal obj_disp_probe, obj_rot_probe, eef_disp_probe, contact_lost_probe, probe_failure, major_disturbance
        nonlocal prev_marker, marker_at_forward0, stop_trigger, rho_impulse, load_impulse, support_force_delta_peak
        nonlocal geom, motion_dir

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

        geom = _contact_geometry(env)
        if PRIMITIVE == "S":
            motion_dir = geom["shear"]
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
                eef_at_probe0 = ee.copy()
            probe_t1 = t_s
            f_hist.append(f_sq)
            fn_hist.append(fn)
            ft_hist.append(ft)
            rho_hist.append(rho)
            rho_impulse += rho * dt
            baseline_fn = fn_hist[0] if fn_hist else fn
            vertical_load_delta = max(0.0, float(fn - baseline_fn))
            load_impulse += vertical_load_delta * dt
            support_force_delta_peak = max(support_force_delta_peak, vertical_load_delta)
            load_hist.append(vertical_load_delta)
            imb_hist.append(imb)
            imb_ratio_hist.append(imb_ratio)
            mk_hist.append(tac["marker_mean"])
            mk_tang_hist.append(tac["marker_tangential"])
            mk_vel_hist.append(marker_vel)
            grip_hist.append(gripper_opening)
            disp_hist.append(accumulated_m * 1000.0)
            obj_disp_probe = max(obj_disp_probe, float(np.linalg.norm(obj_p - obj_at_probe0)))
            obj_rot_probe = max(obj_rot_probe, _quat_angle(obj_q_at_probe0, obj_q))
            eef_disp_probe = max(eef_disp_probe, float(np.linalg.norm(ee - eef_at_probe0)))
            if contact_state != "bilateral" or fn < CONTACT_FORCE_EPS_N or dropped:
                contact_lost_probe = 1
                probe_failure = 1
                if stop_trigger == "completed":
                    stop_trigger = "hard_contact_loss"
            if obj_disp_probe > MAJOR_DISP_M or obj_rot_probe > ROT_MAJOR_RAD:
                major_disturbance = 1

        rows.append(
            ProbeStep(
                trial_id, SPLIT, TASK_ID, OBJ_NAME, seed_idx, mu, step, PRIMITIVE, selector_output, phase, step, t_s,
                float(force_cmd), f_sq, float(preload_target), fn, ft, rho,
                float(fL[0]), float(fL[1]), float(fL[2]), float(fR[0]), float(fR[1]), float(fR[2]),
                imb, imb_ratio, gripper_opening,
                float(geom["n_left"][0]), float(geom["n_left"][1]), float(geom["n_left"][2]),
                float(geom["n_right"][0]), float(geom["n_right"][1]), float(geom["n_right"][2]),
                float(motion_dir[0]), float(motion_dir[1]), float(motion_dir[2]),
                float(geom["downstream"][0]), float(geom["downstream"][1]), float(geom["downstream"][2]),
                float(inc_mm), float(accumulated_m * 1000.0), float(eef_disp_probe * 1000.0),
                obj_disp_probe, obj_rot_probe, tac["marker_mean"], tac["marker_tangential"], marker_vel, marker_du,
                left_contact, right_contact, contact_state, stop_trigger,
                float(ee[0]), float(ee[1]), float(ee[2]), float(obj_p[0]), float(obj_p[1]), float(obj_p[2]),
                float(obj_q[0]), float(obj_q[1]), float(obj_q[2]), float(obj_q[3]), tac["tactile_ok"],
            )
        )
        return obs2, bool(term[0].item()) or bool(trunc[0].item()), contact_state, fn, ft, rho, tac["marker_mean"], imb_ratio

    terminated = False
    latest_contact_state, latest_fn, latest_imb_ratio = "none", np.nan, np.nan
    for phase, n_steps, target in [("approach", APPROACH_STEPS, pregrasp), ("descend", DESCEND_STEPS, grasp), ("close", CLOSE_STEPS, grasp), ("hold", HOLD_STEPS, grasp)]:
        start = cmd_pos.copy()
        for i in range(n_steps):
            cmd_pos = _interp(start, target, i, n_steps)
            if phase in ("approach", "descend"):
                d_pred = D_OPEN
            obs, terminated, latest_contact_state, latest_fn, _, _, _, latest_imb_ratio = observe_and_record(phase, False, 0.0)
            if phase == "hold" and i in (14, 24, 34):
                if latest_contact_state != "bilateral" or latest_fn < 0.65 * preload_target:
                    preload_target = min(PRELOAD_CAP_N, preload_target + PRELOAD_STEP_N)
            if terminated:
                break
        if terminated:
            break

    hold_rows = [r for r in rows if r.probe_phase == "hold"]
    preprobe_fn = float(np.mean([r.measured_fn for r in hold_rows[-10:]])) if hold_rows else np.nan
    preprobe_marker = float(np.mean([r.marker_motion for r in hold_rows[-10:]])) if hold_rows else np.nan
    preprobe_ft = float(np.mean([r.measured_ft for r in hold_rows[-10:]])) if hold_rows else np.nan
    preprobe_imb_ratio = float(np.mean([r.force_imbalance_ratio for r in hold_rows[-10:]])) if hold_rows else np.nan
    preload_noise_fn = float(np.std([r.measured_fn for r in hold_rows[-10:]])) if len(hold_rows) >= 2 else 0.0
    preload_noise_ft = float(np.std([r.measured_ft for r in hold_rows[-10:]])) if len(hold_rows) >= 2 else 0.0
    preload_noise_marker = float(np.std([r.marker_motion for r in hold_rows[-10:]])) if len(hold_rows) >= 2 else 0.0

    geom = _contact_geometry(env)
    selector_output, selector_reason, selector_s_feasible = _selector(geom, preprobe_fn, preprobe_ft, preprobe_imb_ratio, latest_contact_state)
    motion_dir = geom["shear"] if PRIMITIVE == "S" else geom["support"]
    probe_start = grasp.copy()
    cmd_pos = probe_start.copy()

    if PRIMITIVE == "S" and not selector_s_feasible:
        primitive_infeasible = 1
        probe_failure = 1
        stop_trigger = "primitive_s_infeasible_preprobe"
    if PRIMITIVE == "L" and selector_output == "NO_FEASIBLE_PROBE":
        primitive_infeasible = 1
        probe_failure = 1
        stop_trigger = "primitive_l_infeasible_preprobe"

    if not terminated and not primitive_infeasible:
        max_disp = S_MAX_DISP_M if PRIMITIVE == "S" else L_MAX_DISP_M
        step_m = S_STEP_M if PRIMITIVE == "S" else L_STEP_M
        for _ in range(MAX_OUT_STEPS):
            if accumulated_m >= max_disp:
                stop_trigger = "max_displacement_cap"
                break
            if PRIMITIVE == "S":
                geom = _contact_geometry(env)
                if geom["max_normal_leakage"] > S_LEAKAGE_MAX * 1.5:
                    stop_trigger = "online_shear_geometry_invalid"
                    break
                motion_dir = geom["shear"]
            inc = min(step_m, max_disp - accumulated_m)
            cmd_pos = cmd_pos + motion_dir * inc
            accumulated_m += inc
            obs, terminated, _, fn, _, rho, marker, imb_ratio = observe_and_record("probe_out", True, inc * 1000.0)
            if stop_trigger != "completed" or terminated:
                break
            if np.isfinite(preprobe_fn) and fn < RELATIVE_NORMAL_ALPHA * preprobe_fn:
                stop_trigger = "relative_normal_drop"
                break
            if rho >= RHO_CAP:
                stop_trigger = "shear_ratio_cap"
                break
            if PRIMITIVE == "S" and rho_impulse >= S_RHO_IMPULSE_TARGET:
                stop_trigger = "normalized_shear_impulse"
                break
            if PRIMITIVE == "L":
                if imb_ratio >= L_IMBALANCE_CAP:
                    stop_trigger = "support_transfer_imbalance_cap"
                    break
                if support_force_delta_peak >= L_SUPPORT_LOAD_BUDGET_N:
                    stop_trigger = "support_load_budget"
                    break
                if load_impulse >= L_LOAD_IMPULSE_TARGET:
                    stop_trigger = "support_load_impulse"
                    break
            if np.isfinite(preprobe_marker) and abs(marker - preprobe_marker) / max(abs(preprobe_marker), 1e-6) > MARKER_NORM_STOP:
                stop_trigger = "marker_motion_budget"
                break
        return_start = cmd_pos.copy()
        for i in range(RETURN_STEPS):
            cmd_pos = _interp(return_start, probe_start, i, RETURN_STEPS)
            accumulated_m = float(np.linalg.norm(cmd_pos - probe_start))
            obs, terminated, _, _, _, _, _, _ = observe_and_record("probe_back", True, 0.0)
            if terminated:
                break
        if not terminated:
            return_completed = 1
            for _ in range(POST_HOLD_STEPS):
                cmd_pos = probe_start.copy()
                accumulated_m = 0.0
                obs, terminated, _, _, _, _, _, _ = observe_and_record("probe_hold", True, 0.0)
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
    informative_s = PRIMITIVE == "S" and rho_impulse >= max(S_INFO_RHO_IMPULSE_MIN, 5.0 * preload_noise_ft / max(preprobe_fn, 1e-6) * dt) and peak(mk_tang_hist) >= max(S_INFO_MARKER_TANGENTIAL_MIN, 5.0 * preload_noise_marker)
    informative_l = PRIMITIVE == "L" and load_impulse >= L_INFO_LOAD_IMPULSE_MIN and support_force_delta_peak >= max(L_INFO_FORCE_DELTA_MIN_N, 5.0 * preload_noise_fn)
    informative = bool(informative_s or informative_l)
    safe = bool(not dropped and not major_disturbance and not contact_lost_probe and return_completed and not terminated and not primitive_infeasible)
    qualified = bool(safe and informative)

    rec = {
        "trial_id": trial_id, "split": SPLIT, "probe_source": "P4R2_CONTACT_FEASIBLE_MULTI_PRIMITIVE_PROBE_COMPOSER",
        "primitive": PRIMITIVE, "task_id": TASK_ID, "object_id": OBJ_NAME, "seed_idx": seed_idx,
        "friction": mu, "friction_applied": applied, "selector_output": selector_output,
        "selector_reason": selector_reason, "selector_score_max_normal_leakage": geom["max_normal_leakage"],
        "selector_score_downstream_alignment": geom["downstream_alignment"], "selector_score_null_condition": geom["null_condition"],
        "selector_score_preprobe_fn": preprobe_fn, "selector_score_preprobe_imbalance_ratio": preprobe_imb_ratio,
        "selector_s_feasible": int(selector_s_feasible), "primitive_infeasible": primitive_infeasible,
        "target_preload_N": preload_target, "preload_rule": "start_3N_minimal_bilateral_contact_cap4.5N",
        "motion_source": geom["source"] if PRIMITIVE == "S" else "gravity_opposing_support_transfer",
        "left_normal_x": float(geom["n_left"][0]), "left_normal_y": float(geom["n_left"][1]), "left_normal_z": float(geom["n_left"][2]),
        "right_normal_x": float(geom["n_right"][0]), "right_normal_y": float(geom["n_right"][1]), "right_normal_z": float(geom["n_right"][2]),
        "commanded_dir_x": float(motion_dir[0]), "commanded_dir_y": float(motion_dir[1]), "commanded_dir_z": float(motion_dir[2]),
        "downstream_dir_x": float(geom["downstream"][0]), "downstream_dir_y": float(geom["downstream"][1]), "downstream_dir_z": float(geom["downstream"][2]),
        "object_to_basket_x": float(geom["basket_b"][0] - geom["obj_b"][0]), "object_to_basket_y": float(geom["basket_b"][1] - geom["obj_b"][1]), "object_to_basket_z": float(geom["basket_b"][2] - geom["obj_b"][2]),
        "normal_leakage_left": geom["leakage_left"], "normal_leakage_right": geom["leakage_right"], "max_normal_leakage": geom["max_normal_leakage"],
        "bilateral_normal_consistency": geom["bilateral_normal_consistency"], "patch_center_available": False, "patch_area_available": False,
        "patch_edge_margin_available": False, "patch_margin_proxy": float(preprobe_marker),
        "actual_probe_displacement_mm": peak(disp_hist), "probe_path_mm": float(sum(abs(x) for x in np.diff([0.0] + disp_hist))) if disp_hist else 0.0,
        "actual_eef_displacement_mm": float(eef_disp_probe * 1000.0), "probe_duration_s": None if probe_t0 is None else (probe_t1 - probe_t0 + dt),
        "stop_trigger": stop_trigger, "probe_failure": probe_failure, "contact_lost_probe": contact_lost_probe,
        "hard_contact_loss_timestamp": next((r.t_s for r in rows if r.stop_trigger == "hard_contact_loss"), np.nan),
        "dropped": dropped, "obj_disp_probe_m": obj_disp_probe, "obj_rot_probe_rad": obj_rot_probe, "major_disturbance": major_disturbance,
        "return_completed": return_completed, "tactile_ok": int(any(r.tactile_ok for r in rows)),
        "preprobe_normal_force": preprobe_fn, "preprobe_marker_motion": preprobe_marker, "preprobe_tangential_force": preprobe_ft,
        "preload_noise_fn": preload_noise_fn, "preload_noise_ft": preload_noise_ft, "preload_noise_marker": preload_noise_marker,
        "f_meas_mean": mean(f_hist), "normal_force_mean": mean(fn_hist), "normal_force_peak": peak(fn_hist), "normal_force_hyst": hyst(fn_hist),
        "normal_loading_slope": slope(fn_hist[:half]), "normal_unloading_slope": slope(fn_hist[half:]), "ftan_mean": mean(ft_hist), "ftan_peak": peak(ft_hist),
        "ftan_hyst": hyst(ft_hist), "rho_mean": mean(rho_hist), "rho_peak": peak(rho_hist), "rho_hyst": hyst(rho_hist), "rho_impulse": rho_impulse,
        "support_load_impulse": load_impulse, "support_force_delta_peak": support_force_delta_peak, "imb_mean": mean(imb_hist), "imb_peak": peak(imb_hist),
        "imb_ratio_mean": mean(imb_ratio_hist), "imb_ratio_peak": peak(imb_ratio_hist), "marker_mean": mean(mk_hist), "marker_peak": peak(mk_hist),
        "marker_tangential_mean": mean(mk_tang_hist), "marker_tangential_peak": peak(mk_tang_hist), "marker_hyst": hyst(mk_hist),
        "marker_unloading": last(mk_hist) - mean(mk_hist) if mk_hist else np.nan, "marker_vel_abs_peak": float(np.max(np.abs(mk_vel_hist))) if mk_vel_hist else np.nan,
        "residual_marker_displacement": last(mk_hist), "gripper_opening_mean": mean(grip_hist), "safe": int(safe), "informative": int(informative),
        "qualified": int(qualified), "task_specific_probe_lookup": False, "gt_hidden_physics_input": False,
    }
    print(f"summary {trial_id} task={TASK_ID} mu={mu} primitive={PRIMITIVE} selector={selector_output} safe={safe} info={informative} qual={qualified} stop={stop_trigger}", flush=True)
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
        env_cfg.episode_length_s = float(os.environ.get("P4R2_EPISODE_S", "25"))
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        print("env ready", {"task": TASK_ID, "object": OBJ_NAME, "dt": dt, "n_seeds": N_SEEDS, "primitive": PRIMITIVE, "split": SPLIT}, flush=True)

        meta = {"task_id": TASK_ID, "object_id": OBJ_NAME, "dt": dt, "n_seeds": N_SEEDS, "seed_offset": SEED_OFFSET, "mus": MUS, "primitive": PRIMITIVE, "split": SPLIT, "method_change": "CONTACT_FEASIBLE_MULTI_PRIMITIVE_PROBE_COMPOSER_ONLY", "task_specific_probe_lookup": False, "gt_hidden_physics_input": False}
        (OUT / f"P4R2_{SPLIT}_{PRIMITIVE}_TASK{TASK_ID}_COLLECT_META.json").write_text(json.dumps(meta, indent=2) + "\n")
        task_csv = OUT / f"P4R2_{SPLIT}_{PRIMITIVE}_TASK{TASK_ID}_PROBE.csv"
        episodes_csv = OUT / f"P4R2_{SPLIT}_{PRIMITIVE}_EPISODES.csv"
        steps_csv = OUT / "P4R2_TIMESTEP_TELEMETRY" / f"P4R2_{SPLIT}_{PRIMITIVE}_TASK{TASK_ID}_TIMESTEPS.csv"
        done = _existing_done(task_csv) if RESUME else set()
        task_rows: list[dict] = []
        if RESUME and task_csv.exists() and task_csv.stat().st_size > 0:
            with task_csv.open() as f:
                task_rows.extend(list(csv.DictReader(f)))
        step_fields = list(ProbeStep.__dataclass_fields__.keys())
        for s in range(SEED_OFFSET, SEED_OFFSET + N_SEEDS):
            for mu in MUS:
                trial_id = f"p4r2_{SPLIT}_{PRIMITIVE.lower()}_t{TASK_ID}_s{s}_mu{mu:g}"
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
        (OUT / f"P4R2_{SPLIT}_{PRIMITIVE}_TASK{TASK_ID}_COLLECT_ERROR.json").write_text(json.dumps({"error": repr(exc), "trace": traceback.format_exc()}, indent=2))
        try:
            simulation_app.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
