#!/usr/bin/env python3
"""R1: FORTE-inspired reactive FORCE baseline (not official FORTE).

METHOD_CHANGE=NONE. Analysis-only. Same calibrated opening servo for
FIXED_4N / FIXED_6N / FIXED_8N / REACTIVE_4_TO_6_TO_8.
Force update = change of numeric servo target 4 → 6 → 8 N on GT_SLIP.
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
OUT = Path(os.environ.get("R1_OUT", Path(__file__).resolve().parents[1]))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

sys.stdout = open(OUT / "logs" / "r1_isaac.log", "w", buffering=1)
sys.stderr = sys.stdout

os.chdir(TABERO)
sys.path.insert(0, str(TABERO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(TABERO / "benchmarks/datasets/libero/assembled_hdf5")
os.environ.setdefault("LIBERO_CONFIG_DIR", str(TABERO / "benchmarks/datasets/libero/config"))
os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(TABERO / "benchmarks/datasets/libero/USD"))

ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
TASK_SUITE = "libero_object"
TASK_ID = 1
OBJ_NAME = "cream_cheese_1"
BASKET_NAME = "basket_1"

STAGE = os.environ.get("R1_STAGE", "low_mu")  # low_mu | all_mu
N_SEEDS = int(os.environ.get("R1_N_SEEDS", "10"))
FRICTIONS = [float(x) for x in os.environ.get("R1_FRICTIONS", "0.2").split(",")]
METHODS = os.environ.get("R1_METHODS", "fixed4,fixed6,fixed8,reactive").split(",")
DO_PLACE = os.environ.get("R1_DO_PLACE", "1") == "1"
LOG_TRAJ = os.environ.get("R1_LOG_TRAJ", "1") == "1"

APPROACH_STEPS = 45
DESCEND_STEPS = 35
CLOSE_STEPS = 70
HOLD_STEPS = 40
LIFT_STEPS = int(os.environ.get("R1_LIFT_STEPS", "50"))
TRANSIT_STEPS = 110
OVER_BASKET_STEPS = 30
PLACE_STEPS = 40
RELEASE_STEPS = 50
SETTLE_PLACE_STEPS = 50

PREGRASP_Z = 0.10
GRASP_Z = 0.018
LIFT_Z = 0.16
PLACE_CLEAR_Z = 0.18
PLACE_Z = 0.10

SERVO_STEP = 0.0006
SERVO_DEADBAND = 0.4
D_OPEN = 0.04
D_CLOSED = 0.0

LIFT_SUCCESS_Z = 0.03
# v1 used F1R2 8 mm / -0.05 m/s (too late; see DIAGNOSIS_V1.md).
# v2: one global earlier-onset tweak, not per-friction.
REL_Z_SLIP = float(os.environ.get("R1_REL_Z_SLIP", "0.004"))
V_REL_SLIP = float(os.environ.get("R1_V_REL_SLIP", "-0.02"))
VNEG_RUN_N = int(os.environ.get("R1_VNEG_RUN", "1"))
XY_SLIP = 0.015
SLAM_N = 20.0

SETTLE_TOL_N = 0.8
SETTLE_MAX_STEPS = 20
OBSERVE_STEPS = 10
SLIP_PERSIST_COUNT = 3
SCHEDULE = [4.0, 6.0, 8.0]
F_STAR_MU = {0.2: 6.0, 0.5: 4.0, 1.0: 4.0}

METHOD_F = {"fixed4": 4.0, "fixed6": 6.0, "fixed8": 8.0, "reactive": 4.0}


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


def _interp(start, end, i, n):
    a = (i + 1) / n
    return (1 - a) * start + a * end


NOMINAL_MAT = None


def _apply_friction(env, name: str, mu: float) -> dict:
    global NOMINAL_MAT
    import torch

    view = env.scene[name].root_physx_view
    if NOMINAL_MAT is None:
        NOMINAL_MAT = view.get_material_properties().clone()
    mats = NOMINAL_MAT.clone()
    mats[..., 0] = mu
    mats[..., 1] = mu
    indices = torch.arange(mats.shape[0], dtype=torch.int32)
    view.set_material_properties(mats, indices)
    got = view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
    return {
        "requested_mu": mu,
        "applied_static_mean": float(got[:, 0].mean()),
        "applied_dynamic_mean": float(got[:, 1].mean()),
        "nominal_static_mean": float(NOMINAL_MAT.detach().cpu().numpy().reshape(-1, 3)[:, 0].mean()),
    }


def _write_csv(path: Path, dicts: list[dict]):
    if not dicts:
        path.write_text("NOT_RUN\nreason: empty\n")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
        w.writeheader()
        for d in dicts:
            w.writerow(d)


def _append_csv(path: Path, dicts: list[dict], fieldnames: list[str]):
    if not dicts:
        return
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        for d in dicts:
            w.writerow(d)


@dataclass
class StepRow:
    trial_id: str
    method: str
    seed_idx: int
    friction: float
    phase: str
    step: int
    t_s: float
    f_cmd: float
    f_meas: float
    d_pred: float
    obj_dz: float
    rel_z: float
    rel_xy: float
    v_rel_z: float
    contact: int
    gt_slip: int
    sm_state: str
    n_updates: int


class ReactiveSM:
    """Deterministic 4→6→8 force-target machine. Same params for every μ."""

    def __init__(self):
        self.f_cmd = SCHEDULE[0]
        self.idx = 0
        self.state = "track"
        self.settle_left = 0
        self.observe_left = 0
        self.slip_in_observe = 0
        self.n_updates = 0
        self.events: list[dict] = []

    def _raise(self, t_s, f_meas, rel_z, rel_xy, phase, reason):
        if self.idx >= len(SCHEDULE) - 1:
            return
        before = self.f_cmd
        self.idx += 1
        self.f_cmd = SCHEDULE[self.idx]
        self.n_updates += 1
        self.state = "settle"
        self.settle_left = SETTLE_MAX_STEPS
        self.observe_left = 0
        self.slip_in_observe = 0
        self.events.append(
            {
                "t_s": t_s,
                "phase": phase,
                "reason": reason,
                "force_before": before,
                "force_after": self.f_cmd,
                "f_meas_at_update": f_meas,
                "rel_z": rel_z,
                "rel_xy": rel_xy,
            }
        )

    def step(self, gt_slip, t_s, f_meas, rel_z, rel_xy, phase, allow_update):
        if self.state == "settle":
            self.settle_left -= 1
            if abs(f_meas - self.f_cmd) <= SETTLE_TOL_N or self.settle_left <= 0:
                self.state = "observe"
                self.observe_left = OBSERVE_STEPS
                self.slip_in_observe = 0
            return
        if self.state == "observe":
            if gt_slip:
                self.slip_in_observe += 1
            self.observe_left -= 1
            if self.observe_left <= 0:
                if allow_update and self.slip_in_observe >= SLIP_PERSIST_COUNT:
                    self._raise(t_s, f_meas, rel_z, rel_xy, phase, "slip_persists")
                else:
                    self.state = "track"
            return
        if allow_update and gt_slip:
            self._raise(t_s, f_meas, rel_z, rel_xy, phase, "gt_slip")


def run_episode(env, *, method: str, seed_idx: int, mu: float, trial_id: str, dt: float, do_place: bool):
    from benchmarks.common.metrics import compute_contact_force_metrics_from_lr_forces

    obs, _ = env.reset()
    fr_info = _apply_friction(env, OBJ_NAME, mu)

    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    ee_pos = eef0[:3].copy()
    eef_aa = _axis_angle_from_quat(eef0[3:7])
    obj0_w = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
    obj0_b, _ = _pose_in_base(env, OBJ_NAME)
    basket_b, _ = _pose_in_base(env, BASKET_NAME)

    pregrasp = obj0_b.copy(); pregrasp[2] += PREGRASP_Z
    grasp = obj0_b.copy(); grasp[2] += GRASP_Z
    lift = grasp.copy(); lift[2] += LIFT_Z
    transit = basket_b.copy(); transit[2] = max(lift[2], basket_b[2] + PLACE_CLEAR_Z)
    place = basket_b.copy(); place[2] = basket_b[2] + PLACE_Z

    d_pred = D_OPEN
    cmd_pos = ee_pos.copy()
    sm = ReactiveSM() if method == "reactive" else None
    f_cmd = METHOD_F[method]
    rows: list[StepRow] = []
    events: list[dict] = []
    fL_hold, fR_hold = [], []
    f_trace = []
    step = 0
    pick_success = lift_success = 0
    lost_in_transit = 0
    retained_over_basket = 0
    official_success = dropped = timed_out = 0
    max_obj_xy = 0.0
    peak_z = obj0_w[2]
    peak_force = 0.0
    max_basket_contact = 0.0
    grasp_rel = None
    prev_rel_z = None
    vneg_run = 0
    contact_ever = 0
    gt_onset = None
    slip_update_latency = None
    force_reached_6 = 0
    after_update_F = []
    excess = 0.0
    f_star_mu = F_STAR_MU.get(round(mu, 2), 6.0)

    def phases():
        yield "approach", APPROACH_STEPS, pregrasp, "open"
        yield "descend", DESCEND_STEPS, grasp, "open"
        yield "close", CLOSE_STEPS, grasp, "track"
        yield "hold", HOLD_STEPS, grasp, "track"
        yield "lift", LIFT_STEPS, lift, "track"
        if do_place:
            yield "transit", TRANSIT_STEPS, transit, "freeze"
            yield "over_basket", OVER_BASKET_STEPS, transit, "freeze"
            yield "place", PLACE_STEPS, place, "freeze"
            yield "release", RELEASE_STEPS, place, "open"
            yield "settle", SETTLE_PLACE_STEPS, place, "open"

    for phase, n_steps, target, servo_mode in phases():
        start = cmd_pos.copy()
        for i in range(n_steps):
            cmd_pos = _interp(start, target, i, n_steps)
            if servo_mode == "open":
                d_pred = D_OPEN
            cur_f = sm.f_cmd if sm is not None else f_cmd
            action = _make_action(cmd_pos, eef_aa, d_pred, cur_f if servo_mode != "open" else 0.0, env.device)
            obs, rew, term, trunc, info = env.step(action)
            step += 1
            t_s = step * dt

            pol = obs["policy"]
            f = pol["gripper_net_force"][0]
            if f.ndim == 3:
                f = f[-1]
            fL = f[0].detach().cpu().numpy()
            fR = f[1].detach().cpu().numpy()
            dbg = _read_debug(env)
            f_sq = _to_float(dbg.get("f_sq_meas"), 0.0)
            if servo_mode == "track":
                d_pred = force_servo(d_pred, f_sq, cur_f)
            # After a reactive raise during freeze phases, re-enable tracking until settled.
            if sm is not None and servo_mode == "freeze" and sm.state == "settle":
                d_pred = force_servo(d_pred, f_sq, sm.f_cmd)

            obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
            ee = pol["eef_pose"][0].detach().cpu().numpy()
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
            if grasp_rel is None and phase in ("hold", "lift") and contact:
                grasp_rel = rel.copy()
            rel_z = float(rel[2] - grasp_rel[2]) if grasp_rel is not None else 0.0
            rel_xy = float(np.linalg.norm((rel - grasp_rel)[:2])) if grasp_rel is not None else 0.0
            v_rel_z = 0.0
            if prev_rel_z is not None:
                v_rel_z = (rel_z - prev_rel_z) / max(dt, 1e-6)
            prev_rel_z = rel_z
            if v_rel_z < V_REL_SLIP:
                vneg_run += 1
            else:
                vneg_run = 0

            gt_slip = 0
            if phase in ("hold", "lift", "transit", "over_basket"):
                if dropped:
                    gt_slip = 1
                if grasp_rel is not None and rel_z < -REL_Z_SLIP:
                    gt_slip = 1
                if phase == "lift" and vneg_run >= VNEG_RUN_N:
                    gt_slip = 1
                if grasp_rel is not None and rel_xy > XY_SLIP:
                    gt_slip = 1
                if contact_ever and (not contact) and phase in ("lift", "transit", "over_basket"):
                    gt_slip = 1
            if gt_slip and gt_onset is None:
                gt_onset = t_s

            allow_update = phase in ("lift", "transit", "over_basket")
            n_upd_before = sm.n_updates if sm is not None else 0
            if sm is not None:
                sm.step(gt_slip, t_s, f_sq, rel_z, rel_xy, phase, allow_update)
                if sm.n_updates > n_upd_before:
                    if slip_update_latency is None and gt_onset is not None:
                        slip_update_latency = t_s - gt_onset
                    ev = sm.events[-1]
                    ev.update({"trial_id": trial_id, "method": method, "seed_idx": seed_idx, "friction": mu})
                    events.append(ev)

            basket_contact = 0.0
            try:
                import torch as _t

                cs = env.scene[f"contact_{BASKET_NAME}_{OBJ_NAME}"]
                basket_contact = float(_t.linalg.vector_norm(cs.data.force_matrix_w.reshape(-1, 3)[0]).item())
            except Exception:
                basket_contact = 0.0
            max_basket_contact = max(max_basket_contact, basket_contact)

            obj_dz = float(obj_p[2] - obj0_w[2])
            obj_xy = float(np.linalg.norm(obj_p[:2] - obj0_w[:2]))
            max_obj_xy = max(max_obj_xy, obj_xy)
            peak_z = max(peak_z, float(obj_p[2]))
            peak_force = max(peak_force, f_sq)
            cur_f = sm.f_cmd if sm is not None else f_cmd
            if cur_f >= 6.0 - 1e-6 and f_sq >= 5.0:
                force_reached_6 = 1
            if sm is not None and sm.n_updates > 0:
                after_update_F.append(f_sq)
            if phase in ("hold", "lift", "transit", "over_basket"):
                fL_hold.append(fL)
                fR_hold.append(fR)
                f_trace.append(f_sq)
                excess += max(f_sq - f_star_mu, 0.0) * dt

            if grasp_flag or (contact and obj_dz > 0.01):
                pick_success = 1
            if obj_dz >= LIFT_SUCCESS_Z:
                lift_success = 1
            if lift_success and phase in ("transit", "over_basket") and (dropped or obj_dz < 0.02):
                lost_in_transit = 1
            if phase == "over_basket":
                retained_over_basket = int((not dropped) and obj_dz >= LIFT_SUCCESS_Z)

            if LOG_TRAJ:
                rows.append(
                    StepRow(
                        trial_id=trial_id,
                        method=method,
                        seed_idx=seed_idx,
                        friction=mu,
                        phase=phase,
                        step=step,
                        t_s=t_s,
                        f_cmd=cur_f,
                        f_meas=f_sq,
                        d_pred=d_pred,
                        obj_dz=obj_dz,
                        rel_z=rel_z,
                        rel_xy=rel_xy,
                        v_rel_z=v_rel_z,
                        contact=contact,
                        gt_slip=gt_slip,
                        sm_state=sm.state if sm is not None else "fixed",
                        n_updates=sm.n_updates if sm is not None else 0,
                    )
                )
            if bool(term[0].item()) or bool(trunc[0].item()):
                timed_out = int(bool(trunc[0].item()) and not official_success)
                break
        else:
            if phase == "lift" and lift_success == 0 and do_place:
                print(f"early_abort_no_lift {trial_id}", flush=True)
                break
            continue
        break

    if fL_hold:
        m = compute_contact_force_metrics_from_lr_forces(np.stack(fL_hold), np.stack(fR_hold))
        mean_sq, max_sq, contact_ratio, mean_app = m.squeeze_mean, m.squeeze_max, m.contact_ratio, m.external_norm_mean
    else:
        mean_sq = max_sq = contact_ratio = mean_app = 0.0
    honest_full = int(lift_success == 1 and max_basket_contact > 0.05)
    retained = int(lift_success and not lost_in_transit and retained_over_basket) if do_place else int(lift_success)
    final_f = sm.f_cmd if sm is not None else f_cmd
    n_updates = sm.n_updates if sm is not None else 0
    slam = int(peak_force >= SLAM_N)
    rescue = 0
    if method == "reactive":
        rescue = int(
            n_updates >= 1
            and force_reached_6
            and lift_success
            and (honest_full if do_place else True)
            and not slam
        )
    summary = {
        "trial_id": trial_id,
        "stage": STAGE,
        "method": method,
        "seed_idx": seed_idx,
        "friction": mu,
        "initial_F": METHOD_F[method],
        "final_F_cmd": final_f,
        "measured_mean_squeeze": mean_sq,
        "measured_max_squeeze": max(peak_force, max_sq),
        "measured_mean_applied": mean_app,
        "contact_ratio": contact_ratio,
        "mean_F_after_update": float(np.mean(after_update_F)) if after_update_F else None,
        "pick_success": pick_success,
        "lift_success": lift_success,
        "retained": retained,
        "place_success": honest_full if do_place else 0,
        "full_task_success": honest_full if do_place else 0,
        "official_success": official_success,
        "dropped": dropped,
        "gt_slip": int(gt_onset is not None),
        "gt_slip_onset_s": gt_onset,
        "n_updates": n_updates,
        "slip_to_update_latency_s": slip_update_latency,
        "force_reached_6": force_reached_6,
        "rescue": rescue,
        "slam": slam,
        "timeout": timed_out,
        "steps": step,
        "obj_peak_z": peak_z,
        "obj_disp_xy": max_obj_xy,
        "basket_contact_max": max_basket_contact,
        "excess_force": excess,
        "integrated_force": float(sum(f_trace) * dt) if f_trace else 0.0,
        "friction_applied": fr_info.get("applied_static_mean", mu),
        "d_pred_final": d_pred,
    }
    print(
        f"summary {trial_id} method={method} meas={mean_sq:.2f} cmd={final_f:.1f} "
        f"upd={n_updates} lift={lift_success} full={summary['full_task_success']} "
        f"slip={summary['gt_slip']} onset={gt_onset} reached6={force_reached_6} slam={slam} rescue={rescue}",
        flush=True,
    )
    return rows, summary, events


def main():
    from isaaclab.app import AppLauncher

    enable_cameras = os.environ.get("R1_ENABLE_CAMERAS", "0") not in ("0", "false", "False")
    app_launcher = AppLauncher(headless=True, enable_cameras=enable_cameras, num_envs=1)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(TASK_SUITE, TASK_ID)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = float(os.environ.get("R1_EPISODE_S", "45"))
        if not enable_cameras:
            for name in ("eye_in_hand_cam", "agentview_cam", "gsmini_left", "gsmini_right"):
                if hasattr(env_cfg.scene, name):
                    delattr(env_cfg.scene, name)
            pol = getattr(env_cfg.observations, "policy", None)
            if pol is not None and hasattr(pol, "gripper_marker_motion"):
                delattr(pol, "gripper_marker_motion")
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        print("env ready", "dt", dt, flush=True)

        frictions = FRICTIONS
        if STAGE == "all_mu":
            frictions = [0.2, 0.5, 1.0]
        methods = [m.strip() for m in METHODS if m.strip()]

        if STAGE == "low_mu":
            csv_path = OUT / "LOW_FRICTION_RESCUE.csv"
            full_path = OUT / "FULL_DOWNSTREAM_RESULTS.csv"
        elif STAGE == "low_mu_v2":
            csv_path = OUT / "LOW_FRICTION_RESCUE_V2.csv"
            full_path = OUT / "FULL_DOWNSTREAM_V2.csv"
        else:
            csv_path = OUT / "ALL_FRICTION_BASELINES.csv"
            full_path = OUT / "FULL_DOWNSTREAM_RESULTS.csv"
        summaries = []
        done = set()
        for p in (csv_path, full_path):
            if p.exists() and p.read_text()[:7] != "NOT_RUN":
                try:
                    with p.open() as f:
                        for row in csv.DictReader(f):
                            done.add(row["trial_id"])
                            if p == csv_path:
                                summaries.append(row)
                except Exception:
                    pass
        print(f"resume {len(done)} ids", flush=True)

        traj_fields = list(StepRow.__dataclass_fields__.keys())
        ev_fields = [
            "trial_id", "method", "seed_idx", "friction", "t_s", "phase", "reason",
            "force_before", "force_after", "f_meas_at_update", "rel_z", "rel_xy",
        ]

        for mu in frictions:
            for method in methods:
                for s in range(N_SEEDS):
                    tid = f"{STAGE}_{method}_mu{mu:g}_s{s}"
                    if tid in done:
                        print(f"skip {tid}", flush=True)
                        continue
                    print(f"=== {tid} ===", flush=True)
                    rows, summ, events = run_episode(
                        env, method=method, seed_idx=s, mu=mu, trial_id=tid, dt=dt, do_place=DO_PLACE
                    )
                    summaries.append(summ)
                    _write_csv(csv_path, summaries)
                    if DO_PLACE:
                        _write_csv(full_path, summaries)
                    if LOG_TRAJ and rows:
                        _append_csv(OUT / "FORCE_TRAJECTORIES.csv", [asdict(r) for r in rows], traj_fields)
                    if events:
                        _append_csv(OUT / "SLIP_AND_UPDATE_EVENTS.csv", events, ev_fields)

        meta = {
            "stage": STAGE,
            "methods": methods,
            "frictions": frictions,
            "n_seeds": N_SEEDS,
            "do_place": DO_PLACE,
            "n_summaries": len(summaries),
            "this_is_not_official_forte": True,
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
