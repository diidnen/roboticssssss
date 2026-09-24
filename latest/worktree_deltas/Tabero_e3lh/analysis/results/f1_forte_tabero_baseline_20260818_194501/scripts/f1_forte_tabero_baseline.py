#!/usr/bin/env python3
"""F1 Tabero: calibrated force servo + FORTE-style reactive baseline.

METHOD_CHANGE=NONE. Analysis-only. No Tabero source edits. No learned controllers.
ARM_POLICY=SCRIPTED.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

TABERO = Path("/home/exouser/Tabero")
OUT = Path(os.environ.get("F1_OUT", Path(__file__).resolve().parents[1]))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

sys.stdout = open(OUT / "logs" / "f1_tabero.log", "w", buffering=1)
sys.stderr = sys.stdout

os.chdir(TABERO)
sys.path.insert(0, str(TABERO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(TABERO / "benchmarks/datasets/libero/assembled_hdf5")
os.environ.setdefault("LIBERO_CONFIG_DIR", str(TABERO / "benchmarks/datasets/libero/config"))
os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(TABERO / "benchmarks/datasets/libero/USD"))

ENV_ID = os.environ.get("F1_ENV_ID", "Isaac-Libero-Franka-Hybrid-Tactile-v0")
TASK_SUITE = os.environ.get("F1_TASK_SUITE", "libero_object")
TASK_ID = int(os.environ.get("F1_TASK_ID", "1"))
OBJ_NAME = os.environ.get("F1_OBJ", "cream_cheese_1")
BASKET_NAME = os.environ.get("F1_BASKET", "basket_1")
STAGE = os.environ.get("F1_STAGE", "all")  # servo | explore | baselines | all

FORCE_TARGETS = [float(x) for x in os.environ.get("F1_FORCES", "1,2,3,4,6,8").split(",")]
FRICTION_VALUES = [float(x) for x in os.environ.get("F1_FRICTIONS", "0.2,0.5,1.0").split(",")]
N_SEEDS_SERVO = int(os.environ.get("F1_N_SEEDS_SERVO", "5"))
N_SEEDS = int(os.environ.get("F1_N_SEEDS", "5"))
DO_PLACE = os.environ.get("F1_DO_PLACE", "0") == "1"

APPROACH_STEPS = int(os.environ.get("F1_APPROACH_STEPS", "45"))
DESCEND_STEPS = int(os.environ.get("F1_DESCEND_STEPS", "35"))
CLOSE_STEPS = int(os.environ.get("F1_CLOSE_STEPS", "70"))
HOLD_STEPS = int(os.environ.get("F1_HOLD_STEPS", "40"))
LIFT_STEPS = int(os.environ.get("F1_LIFT_STEPS", "40"))
TRANSIT_STEPS = int(os.environ.get("F1_TRANSIT_STEPS", "50"))
PLACE_STEPS = int(os.environ.get("F1_PLACE_STEPS", "40"))
RELEASE_STEPS = int(os.environ.get("F1_RELEASE_STEPS", "25"))

PREGRASP_Z = float(os.environ.get("F1_PREGRASP_Z", "0.10"))
GRASP_Z = float(os.environ.get("F1_GRASP_Z", "0.018"))
LIFT_Z = float(os.environ.get("F1_LIFT_Z", "0.12"))
PLACE_CLEAR_Z = float(os.environ.get("F1_PLACE_CLEAR_Z", "0.12"))
PLACE_Z = float(os.environ.get("F1_PLACE_Z", "0.04"))

SERVO_STEP = float(os.environ.get("F1_SERVO_STEP", "0.0006"))
SERVO_DEADBAND = float(os.environ.get("F1_SERVO_DEADBAND", "0.4"))
D_OPEN = 0.04
D_CLOSED = 0.0
DT = 1.0 / 60.0  # logging approx; env dt read if available

LIFT_SUCCESS_Z = 0.03
F_INIT = float(os.environ.get("F1_F_INIT", "2.0"))
F_HIGH = float(os.environ.get("F1_F_HIGH", "8.0"))
F_INC = float(os.environ.get("F1_F_INC", "2.0"))
F_MAX = float(os.environ.get("F1_F_MAX", "8.0"))
SLIP_COOLDOWN_STEPS = int(os.environ.get("F1_SLIP_COOLDOWN", "12"))
REL_Z_SLIP = 0.008
V_REL_SLIP = -0.05
XY_SLIP = 0.015
MARKER_D_THRESH = float(os.environ.get("F1_MARKER_D", "15.0"))
IMB_THRESH = float(os.environ.get("F1_IMB", "1.5"))


def _axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    if s < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (np.array([x, y, z], dtype=np.float32) / s) * angle


def _to_float(x, default: float = 0.0) -> float:
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


def _pose_in_base(env, name: str):
    import isaaclab.utils.math as math_utils

    robot = env.scene["robot"]
    obj = env.scene[name]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w, obj.data.root_quat_w
    )
    return pos_b[0].detach().cpu().numpy(), quat_b[0].detach().cpu().numpy()


def _make_action(eef_pos, eef_aa, d_pred, desired_squeeze_N, device):
    import torch

    action = torch.zeros((1, 13), device=device)
    action[0, 0:3] = torch.tensor(np.asarray(eef_pos, dtype=np.float32), device=device)
    action[0, 3:6] = torch.tensor(np.asarray(eef_aa, dtype=np.float32), device=device)
    action[0, 6] = float(d_pred)
    half = float(desired_squeeze_N) * 0.5
    action[0, 9] = half
    action[0, 12] = half
    return action


def _read_debug(env):
    try:
        return env.action_manager.get_term("arm_action").debug_info or {}
    except Exception:
        return {}


def force_servo(d_pred: float, f_meas: float, f_star: float) -> float:
    err = f_star - f_meas
    if abs(err) <= SERVO_DEADBAND:
        return float(np.clip(d_pred, D_CLOSED, D_OPEN))
    if err > 0:
        d_pred = d_pred - SERVO_STEP
    else:
        d_pred = d_pred + SERVO_STEP
    return float(np.clip(d_pred, D_CLOSED, D_OPEN))


NOMINAL_MAT = None


def _read_friction(env, name: str) -> dict:
    view = env.scene[name].root_physx_view
    mats = view.get_material_properties()
    arr = mats.detach().cpu().numpy()
    return {
        "shape": list(arr.shape),
        "static": arr.reshape(-1, 3)[:, 0].tolist(),
        "dynamic": arr.reshape(-1, 3)[:, 1].tolist(),
        "restitution": arr.reshape(-1, 3)[:, 2].tolist(),
        "static_mean": float(arr.reshape(-1, 3)[:, 0].mean()),
        "dynamic_mean": float(arr.reshape(-1, 3)[:, 1].mean()),
    }


def _apply_friction(env, name: str, mu: float) -> dict:
    global NOMINAL_MAT
    import torch

    view = env.scene[name].root_physx_view
    if NOMINAL_MAT is None:
        NOMINAL_MAT = view.get_material_properties().clone()
    mats = NOMINAL_MAT.clone()
    mats[..., 0] = mu
    mats[..., 1] = mu
    n_env = mats.shape[0]
    indices = torch.arange(n_env, dtype=torch.int32)
    view.set_material_properties(mats, indices)
    got = view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
    return {
        "requested_mu": mu,
        "applied_static_mean": float(got[:, 0].mean()),
        "applied_dynamic_mean": float(got[:, 1].mean()),
        "nominal_static_mean": float(NOMINAL_MAT.detach().cpu().numpy().reshape(-1, 3)[:, 0].mean()),
    }


def _interp(start, end, i, n):
    a = (i + 1) / n
    return (1 - a) * start + a * end


@dataclass
class StepRow:
    trial_id: str
    stage: str
    controller: str
    seed_idx: int
    friction: float
    desired_F: float
    phase: str
    step: int
    d_pred: float
    f_sq_meas: float
    applied: float
    gripper_open: float
    ee_x: float
    ee_y: float
    ee_z: float
    obj_x: float
    obj_y: float
    obj_z: float
    obj_dz: float
    rel_z: float
    rel_xy: float
    v_rel_z: float
    marker: float
    marker_d: float
    imb: float
    contact: int
    gt_slip: int
    tactile_slip: int
    dropped: int
    official_success: int
    n_force_inc: int


def run_episode(
    env,
    *,
    stage: str,
    seed_idx: int,
    mu: float,
    desired_F: float,
    controller: str,
    do_lift: bool,
    do_place: bool,
    trial_id: str,
    log_steps: bool,
):
    from benchmarks.common.metrics import compute_contact_force_metrics_from_lr_forces

    obs, _ = env.reset()
    fr_info = _apply_friction(env, OBJ_NAME, mu)

    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    ee_pos = eef0[:3].copy()
    eef_aa = _axis_angle_from_quat(eef0[3:7])
    obj0_w = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
    obj0_b, _ = _pose_in_base(env, OBJ_NAME)
    basket_b, _ = _pose_in_base(env, BASKET_NAME)

    pregrasp = obj0_b.copy()
    pregrasp[2] += PREGRASP_Z
    grasp = obj0_b.copy()
    grasp[2] += GRASP_Z
    lift = grasp.copy()
    lift[2] += LIFT_Z
    transit = basket_b.copy()
    transit[2] = lift[2]
    if transit[2] < basket_b[2] + PLACE_CLEAR_Z:
        transit[2] = basket_b[2] + PLACE_CLEAR_Z
    place = basket_b.copy()
    place[2] = basket_b[2] + PLACE_Z

    d_pred = D_OPEN
    cmd_pos = ee_pos.copy()
    rows: list[StepRow] = []
    fL_hold, fR_hold = [], []
    step = 0
    pick_success = 0
    lift_success = 0
    retained = 1
    placed = 0
    official_success = 0
    dropped = 0
    timed_out = 0
    peak_z = obj0_w[2]
    max_obj_xy = 0.0
    f_cmd = float(desired_F)
    n_inc = 0
    n_gt_slip = 0
    n_tac_slip = 0
    cooldown = 0
    grasp_rel = None
    prev_rel_z = None
    vneg_run = 0
    marker_prev = None
    contact_ever = 0
    f_trace = []
    settle_step = None
    peak_force = 0.0
    lost_contact = 0

    def phases():
        yield "approach", APPROACH_STEPS, pregrasp, False
        yield "descend", DESCEND_STEPS, grasp, False
        yield "close", CLOSE_STEPS, grasp, True
        yield "hold", HOLD_STEPS, grasp, True
        if do_lift:
            yield "lift", LIFT_STEPS, lift, True
        if do_place:
            yield "transit", TRANSIT_STEPS, transit, True
            yield "place", PLACE_STEPS, place, True
            yield "release", RELEASE_STEPS, place, False

    for phase, n_steps, target, servo_on in phases():
        start = cmd_pos.copy()
        for i in range(n_steps):
            cmd_pos = _interp(start, target, i, n_steps)
            if not servo_on:
                d_pred = D_OPEN if phase in ("approach", "descend", "release") else d_pred
            action = _make_action(cmd_pos, eef_aa, d_pred, f_cmd if servo_on else 0.0, env.device)
            obs, rew, term, trunc, info = env.step(action)
            step += 1
            if cooldown > 0:
                cooldown -= 1

            pol = obs["policy"]
            f = pol["gripper_net_force"][0]
            if f.ndim == 3:
                f = f[-1]
            fL = f[0].detach().cpu().numpy()
            fR = f[1].detach().cpu().numpy()
            dbg = _read_debug(env)
            f_sq = _to_float(dbg.get("f_sq_meas"), 0.0)
            f_app_vec = dbg.get("F_app_meas_b")
            applied = 0.0
            if f_app_vec is not None:
                try:
                    import torch as _t

                    if isinstance(f_app_vec, _t.Tensor):
                        applied = float(_t.linalg.vector_norm(f_app_vec.reshape(-1, 3)[0]).item())
                except Exception:
                    applied = 0.0
            if servo_on:
                d_pred = force_servo(d_pred, f_sq, f_cmd)

            obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
            ee = pol["eef_pose"][0].detach().cpu().numpy()
            gp = float(abs(pol["gripper_pos"][0, 0].detach().cpu().item()))
            try:
                mm = float(np.linalg.norm(pol["gripper_marker_motion"][0].detach().cpu().numpy()))
            except Exception:
                mm = 0.0
            marker_d = 0.0 if marker_prev is None else abs(mm - marker_prev)
            marker_prev = mm
            imb = float(abs(np.linalg.norm(fL) - np.linalg.norm(fR)))
            contact = int((np.abs(fL).sum() + np.abs(fR).sum()) > 1e-3)
            if contact:
                contact_ever = 1
            try:
                dropped = int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
            except Exception:
                pass
            try:
                official_success = int(bool(env.termination_manager.get_term("success")[0].item()))
            except Exception:
                pass
            grasp_flag = 0
            try:
                grasp_flag = int(bool(obs["subtask_terms"]["grasp_1"][0].item()))
            except Exception:
                pass

            rel = obj_p - ee[:3]
            if phase == "close" and i == n_steps - 1:
                grasp_rel = rel.copy()
            if grasp_rel is None and phase in ("hold", "lift", "transit") and contact:
                grasp_rel = rel.copy()
            rel_z = float(rel[2] - grasp_rel[2]) if grasp_rel is not None else 0.0
            rel_xy = float(np.linalg.norm((rel - grasp_rel)[:2])) if grasp_rel is not None else 0.0
            v_rel_z = 0.0
            if prev_rel_z is not None:
                v_rel_z = (rel_z - prev_rel_z) / max(DT, 1e-6)
            prev_rel_z = rel_z
            if v_rel_z < V_REL_SLIP:
                vneg_run += 1
            else:
                vneg_run = 0

            gt_slip = 0
            if servo_on and phase in ("hold", "lift", "transit"):
                if dropped:
                    gt_slip = 1
                if grasp_rel is not None and rel_z < -REL_Z_SLIP:
                    gt_slip = 1
                if phase == "lift" and vneg_run >= 2:
                    gt_slip = 1
                if grasp_rel is not None and rel_xy > XY_SLIP:
                    gt_slip = 1
                if contact_ever and (not contact) and phase in ("lift", "transit"):
                    gt_slip = 1
                    lost_contact = 1
            tactile_slip = int((marker_d > MARKER_D_THRESH) or (imb > IMB_THRESH))
            if gt_slip:
                n_gt_slip += 1
            if tactile_slip:
                n_tac_slip += 1

            if controller == "forte_oracle" and servo_on and phase in ("lift", "transit"):
                if gt_slip and cooldown == 0 and f_cmd < F_MAX - 1e-6:
                    f_cmd = min(F_MAX, f_cmd + F_INC)
                    n_inc += 1
                    cooldown = SLIP_COOLDOWN_STEPS
            if controller == "forte_tactile" and servo_on and phase in ("lift", "transit"):
                if tactile_slip and cooldown == 0 and f_cmd < F_MAX - 1e-6:
                    f_cmd = min(F_MAX, f_cmd + F_INC)
                    n_inc += 1
                    cooldown = SLIP_COOLDOWN_STEPS

            obj_dz = float(obj_p[2] - obj0_w[2])
            obj_xy = float(np.linalg.norm(obj_p[:2] - obj0_w[:2]))
            max_obj_xy = max(max_obj_xy, obj_xy)
            peak_z = max(peak_z, float(obj_p[2]))
            peak_force = max(peak_force, f_sq)
            if phase in ("hold", "lift", "transit"):
                fL_hold.append(fL)
                fR_hold.append(fR)
                f_trace.append(f_sq)
                if settle_step is None and abs(f_sq - f_cmd) <= SERVO_DEADBAND and contact:
                    settle_step = step
            if grasp_flag or (contact and obj_dz > 0.01):
                pick_success = 1
            if obj_dz >= LIFT_SUCCESS_Z:
                lift_success = 1
            if phase in ("lift", "transit", "place") and (dropped or obj_dz < 0.01):
                retained = 0
            if official_success:
                placed = 1

            if log_steps:
                rows.append(
                    StepRow(
                        trial_id=trial_id,
                        stage=stage,
                        controller=controller,
                        seed_idx=seed_idx,
                        friction=mu,
                        desired_F=desired_F,
                        phase=phase,
                        step=step,
                        d_pred=d_pred,
                        f_sq_meas=f_sq,
                        applied=applied,
                        gripper_open=gp,
                        ee_x=float(ee[0]),
                        ee_y=float(ee[1]),
                        ee_z=float(ee[2]),
                        obj_x=float(obj_p[0]),
                        obj_y=float(obj_p[1]),
                        obj_z=float(obj_p[2]),
                        obj_dz=obj_dz,
                        rel_z=rel_z,
                        rel_xy=rel_xy,
                        v_rel_z=v_rel_z,
                        marker=mm,
                        marker_d=marker_d,
                        imb=imb,
                        contact=contact,
                        gt_slip=gt_slip,
                        tactile_slip=tactile_slip,
                        dropped=dropped,
                        official_success=official_success,
                        n_force_inc=n_inc,
                    )
                )
            if bool(term[0].item()) or bool(trunc[0].item()):
                timed_out = int(bool(trunc[0].item()) and not official_success)
                break
        else:
            continue
        break

    hold_rows_f = f_trace
    if fL_hold:
        m = compute_contact_force_metrics_from_lr_forces(np.stack(fL_hold), np.stack(fR_hold))
        mean_sq, max_sq, contact_ratio, mean_app = m.squeeze_mean, m.squeeze_max, m.contact_ratio, m.external_norm_mean
    else:
        mean_sq = max_sq = contact_ratio = mean_app = 0.0
    last_dz = rows[-1].obj_dz if rows else float(env.scene[OBJ_NAME].data.root_pos_w[0, 2].item() - obj0_w[2])
    std_sq = float(np.std(hold_rows_f)) if hold_rows_f else 0.0
    overshoot = max(0.0, peak_force - desired_F)
    summary = {
        "trial_id": trial_id,
        "stage": stage,
        "controller": controller,
        "seed_idx": seed_idx,
        "friction": mu,
        "desired_F": desired_F,
        "final_cmd_F": f_cmd,
        "measured_mean_squeeze": mean_sq,
        "measured_max_squeeze": max_sq,
        "measured_std_squeeze": std_sq,
        "tracking_error": abs(mean_sq - desired_F) if mean_sq else desired_F,
        "overshoot": overshoot,
        "settling_steps": settle_step if settle_step is not None else -1,
        "measured_mean_applied": mean_app,
        "contact_ratio": contact_ratio,
        "contact_maintained": int(contact_ratio >= 0.8 and not lost_contact),
        "pick_success": pick_success,
        "lift_success": lift_success,
        "retained": retained if do_lift else int(not dropped),
        "place_success": placed if do_place else 0,
        "full_task_success": official_success if do_place else 0,
        "official_success": official_success,
        "dropped": dropped,
        "n_gt_slip_steps": n_gt_slip,
        "n_tac_slip_steps": n_tac_slip,
        "gt_slip_episode": int(n_gt_slip > 0 or (do_lift and lift_success == 0) or dropped),
        "n_force_increases": n_inc,
        "timeout": timed_out,
        "steps": step,
        "obj_peak_z": peak_z,
        "obj_disp_xy": max_obj_xy,
        "final_obj_dz": last_dz,
        "d_pred_final": d_pred,
        "friction_applied": fr_info.get("applied_static_mean", mu),
        "nominal_friction": fr_info.get("nominal_static_mean"),
        "instruction": "Pick up the cream cheese and place it in the basket.",
        "force_adverbs_used": False,
        "arm_policy": "SCRIPTED",
    }
    print(
        f"summary {trial_id} ctrl={controller} meas={mean_sq:.2f} cmdF={f_cmd:.1f} "
        f"lift={lift_success} drop={dropped} gt_slip={n_gt_slip} inc={n_inc}",
        flush=True,
    )
    return rows, summary


def _write_csv(path: Path, dicts: list[dict], not_run_reason: str | None = None):
    if not dicts:
        path.write_text(f"NOT_RUN\nreason: {not_run_reason or 'empty'}\n")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
        w.writeheader()
        for d in dicts:
            w.writerow(d)


def _write_steps(path: Path, rows: list[StepRow]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main():
    t0 = time.time()
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
        env_cfg.episode_length_s = float(os.environ.get("F1_EPISODE_S", "40"))
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        print("env ready", flush=True)

        obs, _ = env.reset()
        fr_obj = _read_friction(env, OBJ_NAME)
        (OUT / "FRICTION_RAW.json").write_text(json.dumps({"object": fr_obj, "obj_name": OBJ_NAME}, indent=2))
        nominal_mu = float(fr_obj["static_mean"])
        frictions = list(FRICTION_VALUES)

        servo_summ, servo_rows = [], []
        explore_summ, explore_rows = [], []
        slip_rows = []
        base_summ = []

        run_servo = STAGE in ("servo", "all")
        run_explore = STAGE in ("explore", "all")
        run_base = STAGE in ("baselines", "all")

        if run_servo:
            for f_star in FORCE_TARGETS:
                for s in range(N_SEEDS_SERVO):
                    tid = f"servo_f{f_star:g}_s{s}"
                    print(f"=== {tid} ===", flush=True)
                    rows, summ = run_episode(
                        env,
                        stage="servo",
                        seed_idx=s,
                        mu=nominal_mu,
                        desired_F=f_star,
                        controller="fixed",
                        do_lift=False,
                        do_place=False,
                        trial_id=tid,
                        log_steps=True,
                    )
                    servo_rows.extend(rows)
                    servo_summ.append(summ)
            _write_csv(OUT / "TABERO_FORCE_SERVO_CALIBRATION.csv", servo_summ)
            _write_steps(OUT / "TABERO_FORCE_SERVO_CALIBRATION_STEPS.csv", servo_rows)

        if run_explore:
            explore_forces = [f for f in FORCE_TARGETS if f >= 2.0] or FORCE_TARGETS
            for mu in frictions:
                for f_star in explore_forces:
                    for s in range(N_SEEDS):
                        tid = f"fxmu_mu{mu:g}_f{f_star:g}_s{s}"
                        print(f"=== {tid} ===", flush=True)
                        log = f_star == 2.0  # slip audit at low force
                        rows, summ = run_episode(
                            env,
                            stage="explore",
                            seed_idx=s,
                            mu=mu,
                            desired_F=f_star,
                            controller="fixed",
                            do_lift=True,
                            do_place=DO_PLACE,
                            trial_id=tid,
                            log_steps=log,
                        )
                        explore_summ.append(summ)
                        if log:
                            slip_rows.extend(rows)
                            explore_rows.extend(rows)
            _write_csv(OUT / "FORCE_X_FRICTION_EXPLORATION.csv", explore_summ)
            _write_steps(OUT / "TABERO_SLIP_SIGNAL_AUDIT.csv", slip_rows)

        if run_base:
            # reuse explore for fixed-low / fixed-high if present; still run reactive
            for mu in frictions:
                for ctrl, f0, name in [
                    ("fixed", F_INIT, "low"),
                    ("fixed", F_HIGH, "high"),
                    ("forte_oracle", F_INIT, "oracle"),
                    ("forte_tactile", F_INIT, "tactile"),
                ]:
                    for s in range(N_SEEDS):
                        tid = f"base_{name}_mu{mu:g}_s{s}"
                        print(f"=== {tid} ===", flush=True)
                        rows, summ = run_episode(
                            env,
                            stage="baseline",
                            seed_idx=s,
                            mu=mu,
                            desired_F=f0,
                            controller=ctrl,
                            do_lift=True,
                            do_place=DO_PLACE,
                            trial_id=tid,
                            log_steps=False,
                        )
                        base_summ.append(summ)
            low = [d for d in base_summ if d["controller"] == "fixed" and d["desired_F"] == F_INIT]
            high = [d for d in base_summ if d["controller"] == "fixed" and d["desired_F"] == F_HIGH]
            oracle = [d for d in base_summ if d["controller"] == "forte_oracle"]
            tac = [d for d in base_summ if d["controller"] == "forte_tactile"]
            _write_csv(OUT / "BASELINE_FIXED_LOW.csv", low)
            _write_csv(OUT / "BASELINE_FIXED_HIGH.csv", high)
            _write_csv(OUT / "BASELINE_FORTE_ORACLE_SLIP.csv", oracle)
            _write_csv(OUT / "BASELINE_FORTE_TACTILE.csv", tac)

        meta = {
            "env_id": ENV_ID,
            "task_suite": TASK_SUITE,
            "task_id": TASK_ID,
            "instruction": "Pick up the cream cheese and place it in the basket.",
            "force_adverbs_used": False,
            "arm_policy": "SCRIPTED",
            "stage": STAGE,
            "force_targets": FORCE_TARGETS,
            "frictions": frictions,
            "nominal_friction": nominal_mu,
            "n_servo": len(servo_summ),
            "n_explore": len(explore_summ),
            "n_baseline": len(base_summ),
            "elapsed_s": time.time() - t0,
            "do_place": DO_PLACE,
        }
        (OUT / "SWEEP_META.json").write_text(json.dumps(meta, indent=2))
        print("done", json.dumps(meta), flush=True)
        try:
            env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass
        os._exit(0)
    except Exception as e:
        print("EXCEPTION", repr(e), flush=True)
        print(traceback.format_exc(), flush=True)
        (OUT / "SWEEP_ERROR.json").write_text(json.dumps({"error": repr(e), "trace": traceback.format_exc()}, indent=2))
        try:
            simulation_app.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
