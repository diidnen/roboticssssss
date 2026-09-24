#!/usr/bin/env python3
"""M1-R2 real task-specific Probe A collection.

METHOD_CHANGE=NONE. This script only runs the frozen 4N, base-Y +2mm,
return, stabilize probe on one requested task and writes telemetry into the
M1-R2 result directory. It does not modify Tabero core, benchmark or oracles.
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
OUT = Path(os.environ.get("M1R2_OUT", Path(__file__).resolve().parents[1])).resolve()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)

TASK_OBJECTS = {
    0: "alphabet_soup_1",
    1: "cream_cheese_1",
    2: "salad_dressing_1",
    5: "tomato_sauce_1",
    6: "butter_1",
}
BASKET_NAME = "basket_1"
TASK_SUITE = "libero_object"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"

TASK_ID = int(os.environ["M1R2_TASK_ID"])
OBJ_NAME = TASK_OBJECTS[TASK_ID]
N_SEEDS = int(os.environ.get("M1R2_N_SEEDS", "20"))
MUS = [float(x) for x in os.environ.get("M1R2_MUS", "0.2,0.5,1.0").split(",")]
SEED_OFFSET = int(os.environ.get("M1R2_SEED_OFFSET", "0"))
RESUME = os.environ.get("M1R2_RESUME", "1") not in ("0", "false", "False")

AMP_M = float(os.environ.get("M1R2_AMP_MM", "2")) / 1000.0
PROBE_HALF = int(os.environ.get("M1R2_PROBE_HALF_STEPS", "10"))
POST_HOLD = int(os.environ.get("M1R2_POST_HOLD_STEPS", "5"))
APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS, HOLD_STEPS = 45, 35, 70, 40
PREGRASP_Z, GRASP_Z = 0.10, 0.018
SERVO_STEP, SERVO_DEADBAND = 0.0006, 0.4
D_OPEN, D_CLOSED = 0.04, 0.0
F_PROBE = 4.0
MAJOR_DISP_M = float(os.environ.get("M1R2_MAJOR_DISP_M", "0.01"))
ROT_MAJOR_RAD = float(os.environ.get("M1R2_MAJOR_ROT_RAD", "0.35"))

os.chdir(TABERO)
sys.path.insert(0, str(TABERO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(TABERO / "benchmarks/datasets/libero/assembled_hdf5")
os.environ.setdefault("LIBERO_CONFIG_DIR", str(TABERO / "benchmarks/datasets/libero/config"))
os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(TABERO / "benchmarks/datasets/libero/USD"))

log_path = OUT / "logs" / f"collect_task{TASK_ID}.log"
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


def _tactile_summaries(env, obs):
    out = {
        "marker_mean": 0.0,
        "marker_max": 0.0,
        "marker_std": 0.0,
        "marker_asym": 0.0,
        "marker_tangential": 0.0,
        "marker_tangential_abs": 0.0,
        "hm_mean": 0.0,
        "hm_max": 0.0,
        "hm_std": 0.0,
        "tactile_ok": 0,
    }
    try:
        mm = obs["policy"]["gripper_marker_motion"][0].detach().cpu().numpy()
        cur, init = mm[:, 1], mm[:, 0]
        disp = cur - init
        mag = np.linalg.norm(disp, axis=-1)
        out["marker_mean"] = float(mag.mean())
        out["marker_max"] = float(mag.max()) if mag.size else 0.0
        out["marker_std"] = float(mag.std())
        left = mag[0].mean() if mag.shape[0] > 0 else 0.0
        right = mag[1].mean() if mag.shape[0] > 1 else 0.0
        out["marker_asym"] = float(abs(left - right))
        tang_vec = disp.mean(axis=1)
        out["marker_tangential"] = float(np.linalg.norm(tang_vec, axis=-1).mean())
        out["marker_tangential_abs"] = float(np.abs(disp[..., 1]).mean())
        out["tactile_ok"] = 1
    except Exception:
        pass
    try:
        vals = []
        maxs = []
        stds = []
        for side in ("gsmini_left", "gsmini_right"):
            sen = env.scene[side]
            hm = sen.data.output.get("height_map")
            if hm is None:
                continue
            arr = hm[0].detach().cpu().numpy()
            vals.append(float(arr.mean()))
            maxs.append(float(arr.max()))
            stds.append(float(arr.std()))
        if vals:
            out["hm_mean"] = float(np.mean(vals))
            out["hm_max"] = float(np.max(maxs))
            out["hm_std"] = float(np.mean(stds))
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
    probe_phase: str
    step: int
    t_s: float
    force_target: float
    measured_squeeze: float
    left_fx: float
    left_fy: float
    left_fz: float
    right_fx: float
    right_fy: float
    right_fz: float
    normal_force: float
    tangential_force: float
    force_imbalance: float
    gripper_opening: float
    marker_motion: float
    marker_tangential_motion: float
    marker_tangential_abs: float
    marker_velocity: float
    marker_asymmetry: float
    height_map_mean: float
    height_map_max: float
    height_map_std: float
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
    rel_x_priv: float
    rel_y_priv: float
    rel_z_priv: float
    tactile_ok: int


def _existing_done(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open() as f:
        return {r["trial_id"] for r in csv.DictReader(f)}


def run_probe_episode(env, *, seed_idx: int, mu: float, trial_id: str, dt: float, tactile: bool):
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
    obj0_w = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
    obj0_q = env.scene[OBJ_NAME].data.root_quat_w[0].detach().cpu().numpy()
    obj0_b, _ = _pose_in_base(env, OBJ_NAME)
    pregrasp = obj0_b.copy()
    pregrasp[2] += PREGRASP_Z
    grasp = obj0_b.copy()
    grasp[2] += GRASP_Z

    d_pred = D_OPEN
    cmd_pos = ee_pos.copy()
    last_cmd = cmd_pos.copy()
    step = 0
    rows: list[ProbeStep] = []

    probe_t0 = probe_t1 = None
    obj_at_probe0 = None
    obj_q_at_probe0 = None
    obj_disp_probe = 0.0
    obj_rot_probe = 0.0
    probe_path_mm = 0.0
    contact_lost_probe = 0
    probe_failure = 0
    dropped = 0
    major_disturbance = 0
    grasp_rel = None
    prev_marker = None
    f_hist: list[float] = []
    normal_hist: list[float] = []
    ftan_hist: list[float] = []
    imb_hist: list[float] = []
    mk_hist: list[float] = []
    mk_tang_hist: list[float] = []
    mk_vel_hist: list[float] = []
    hm_hist: list[float] = []
    grip_hist: list[float] = []
    preprobe_normal = np.nan
    preprobe_marker = np.nan
    preprobe_ftan = np.nan

    phases = [
        ("approach", APPROACH_STEPS, pregrasp, "open", False),
        ("descend", DESCEND_STEPS, grasp, "open", False),
        ("close", CLOSE_STEPS, grasp, "track", False),
        ("hold", HOLD_STEPS, grasp, "track", False),
        ("probe_out", PROBE_HALF, grasp + np.array([0.0, AMP_M, 0.0]), "track", True),
        ("probe_back", PROBE_HALF, grasp, "track", True),
        ("probe_hold", POST_HOLD, grasp, "track", True),
    ]

    for phase, n_steps, target, mode, is_probe in phases:
        start = cmd_pos.copy()
        if phase == "probe_out":
            preprobe_normal = float(np.mean(normal_hist[-max(1, min(10, len(normal_hist))):])) if normal_hist else np.nan
            preprobe_marker = float(np.mean(mk_hist[-max(1, min(10, len(mk_hist))):])) if mk_hist else np.nan
            preprobe_ftan = float(np.mean(ftan_hist[-max(1, min(10, len(ftan_hist))):])) if ftan_hist else np.nan
        for i in range(n_steps):
            cmd_pos = _interp(start, target, i, n_steps)
            if is_probe:
                probe_path_mm += float(np.linalg.norm(cmd_pos - last_cmd) * 1000.0)
            last_cmd = cmd_pos.copy()
            if mode == "open":
                d_pred = D_OPEN
            action = _make_action(cmd_pos, eef_aa, d_pred, F_PROBE if mode != "open" else 0.0, env.device)
            obs, _, term, trunc, _ = env.step(action)
            step += 1
            t_s = step * dt

            f = obs["policy"]["gripper_net_force"][0]
            if f.ndim == 3:
                f = f[-1]
            fL = f[0].detach().cpu().numpy().astype(np.float32)
            fR = f[1].detach().cpu().numpy().astype(np.float32)
            f_sq = _f(_dbg(env).get("f_sq_meas"), 0.0)
            normal_force = float(abs(fL[2]) + abs(fR[2]))
            tangential_force = float(np.linalg.norm(np.concatenate([fL[:2], fR[:2]])))
            imbalance = float(abs(np.linalg.norm(fL) - np.linalg.norm(fR)))
            try:
                gripper_opening = _f(obs["policy"]["gripper_pos"][0], np.nan)
            except Exception:
                gripper_opening = np.nan

            if mode == "track":
                d_pred = _force_servo(d_pred, f_sq, F_PROBE)

            tac = _tactile_summaries(env, obs) if tactile else {
                "marker_mean": 0.0,
                "marker_max": 0.0,
                "marker_std": 0.0,
                "marker_asym": 0.0,
                "marker_tangential": 0.0,
                "marker_tangential_abs": 0.0,
                "hm_mean": 0.0,
                "hm_max": 0.0,
                "hm_std": 0.0,
                "tactile_ok": 0,
            }
            marker_vel = 0.0 if prev_marker is None else (tac["marker_mean"] - prev_marker) / dt
            prev_marker = tac["marker_mean"]

            obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
            obj_q = env.scene[OBJ_NAME].data.root_quat_w[0].detach().cpu().numpy()
            ee = obs["policy"]["eef_pose"][0].detach().cpu().numpy()[:3]
            rel = obj_p - ee
            if phase == "close" and i == n_steps - 1:
                grasp_rel = rel.copy()
            rel_d = rel - grasp_rel if grasp_rel is not None else np.zeros(3)

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
                normal_hist.append(normal_force)
                ftan_hist.append(tangential_force)
                imb_hist.append(imbalance)
                mk_hist.append(tac["marker_mean"])
                mk_tang_hist.append(tac["marker_tangential"])
                mk_vel_hist.append(marker_vel)
                hm_hist.append(tac["hm_mean"])
                grip_hist.append(gripper_opening)
                obj_disp_probe = max(obj_disp_probe, float(np.linalg.norm(obj_p - obj_at_probe0)))
                obj_rot_probe = max(obj_rot_probe, _quat_angle(obj_q_at_probe0, obj_q))
                contact = int((np.abs(fL).sum() + np.abs(fR).sum()) > 1e-3)
                if not contact or dropped:
                    contact_lost_probe = 1
                    probe_failure = 1
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
                    probe_phase=phase,
                    step=step,
                    t_s=t_s,
                    force_target=F_PROBE if mode != "open" else 0.0,
                    measured_squeeze=f_sq,
                    left_fx=float(fL[0]),
                    left_fy=float(fL[1]),
                    left_fz=float(fL[2]),
                    right_fx=float(fR[0]),
                    right_fy=float(fR[1]),
                    right_fz=float(fR[2]),
                    normal_force=normal_force,
                    tangential_force=tangential_force,
                    force_imbalance=imbalance,
                    gripper_opening=gripper_opening,
                    marker_motion=tac["marker_mean"],
                    marker_tangential_motion=tac["marker_tangential"],
                    marker_tangential_abs=tac["marker_tangential_abs"],
                    marker_velocity=marker_vel,
                    marker_asymmetry=tac["marker_asym"],
                    height_map_mean=tac["hm_mean"],
                    height_map_max=tac["hm_max"],
                    height_map_std=tac["hm_std"],
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
                    rel_x_priv=float(rel_d[0]),
                    rel_y_priv=float(rel_d[1]),
                    rel_z_priv=float(rel_d[2]),
                    tactile_ok=tac["tactile_ok"],
                )
            )
            if bool(term[0].item()) or bool(trunc[0].item()):
                break
        else:
            continue
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

    half = max(1, len(normal_hist) // 2)
    rec = {
        "trial_id": trial_id,
        "probe_source": "REAL_TASK_SPECIFIC_PROBE",
        "probe_id": "ProbeA_4N_baseY_plus2mm_return",
        "task_id": TASK_ID,
        "object_id": OBJ_NAME,
        "seed_idx": seed_idx,
        "friction": mu,
        "friction_applied": applied,
        "probe_force_N": F_PROBE,
        "probe_amp_mm": AMP_M * 1000.0,
        "probe_path_mm": probe_path_mm,
        "probe_duration_s": None if probe_t0 is None else (probe_t1 - probe_t0 + dt),
        "probe_failure": probe_failure,
        "contact_lost_probe": contact_lost_probe,
        "dropped": dropped,
        "obj_disp_probe_m": obj_disp_probe,
        "obj_rot_probe_rad": obj_rot_probe,
        "major_disturbance": major_disturbance,
        "tactile_ok": int(any(r.tactile_ok for r in rows)),
        "preprobe_normal_force": preprobe_normal,
        "preprobe_marker_motion": preprobe_marker,
        "preprobe_tangential_force": preprobe_ftan,
        "f_meas_mean": mean(f_hist),
        "normal_force_mean": mean(normal_hist),
        "normal_force_peak": peak(normal_hist),
        "normal_force_hyst": hyst(normal_hist),
        "normal_loading_slope": slope(normal_hist[:half]),
        "normal_unloading_slope": slope(normal_hist[half:]),
        "ftan_mean": mean(ftan_hist),
        "ftan_peak": peak(ftan_hist),
        "ftan_hyst": hyst(ftan_hist),
        "imb_mean": mean(imb_hist),
        "imb_peak": peak(imb_hist),
        "marker_mean": mean(mk_hist),
        "marker_peak": peak(mk_hist),
        "marker_tangential_mean": mean(mk_tang_hist),
        "marker_tangential_peak": peak(mk_tang_hist),
        "marker_hyst": hyst(mk_hist),
        "marker_unloading": last(mk_hist) - mean(mk_hist) if mk_hist else np.nan,
        "marker_vel_abs_peak": float(np.max(np.abs(mk_vel_hist))) if mk_vel_hist else np.nan,
        "residual_marker_displacement": last(mk_hist),
        "height_map_mean": mean(hm_hist),
        "gripper_opening_mean": mean(grip_hist),
        "delta_normal_over_baseline": (mean(normal_hist) - preprobe_normal) / max(abs(preprobe_normal), 1e-6)
        if np.isfinite(preprobe_normal)
        else np.nan,
        "delta_ftan_over_baseline": (mean(ftan_hist) - preprobe_ftan) / max(abs(preprobe_ftan), 1e-6)
        if np.isfinite(preprobe_ftan)
        else np.nan,
        "delta_marker_over_baseline": (mean(mk_hist) - preprobe_marker) / max(abs(preprobe_marker), 1e-6)
        if np.isfinite(preprobe_marker)
        else np.nan,
    }
    print(
        f"summary {trial_id} task={TASK_ID} mu={mu} fail={probe_failure} "
        f"disp_mm={obj_disp_probe * 1000:.3f} ftan={rec['ftan_peak']:.4f} imb={rec['imb_peak']:.4f}",
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
        env_cfg.episode_length_s = float(os.environ.get("M1R2_EPISODE_S", "25"))
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        print("env ready", {"task": TASK_ID, "object": OBJ_NAME, "dt": dt, "n_seeds": N_SEEDS}, flush=True)

        meta = {
            "task_id": TASK_ID,
            "object_id": OBJ_NAME,
            "dt": dt,
            "n_seeds": N_SEEDS,
            "mus": MUS,
            "probe": "A",
            "probe_force_N": F_PROBE,
            "probe_amp_mm": AMP_M * 1000.0,
            "probe_half_steps": PROBE_HALF,
            "post_hold_steps": POST_HOLD,
            "method_change": "NONE",
        }
        (OUT / f"TASK{TASK_ID}_COLLECT_META.json").write_text(json.dumps(meta, indent=2) + "\n")

        task_csv = OUT / f"TASK{TASK_ID}_PROBE.csv"
        episodes_csv = OUT / "REAL_PROBE_EPISODES.csv"
        steps_csv = OUT / "REAL_PROBE_TIMESTEPS.csv"
        done = _existing_done(task_csv) if RESUME else set()
        task_rows: list[dict] = []
        if RESUME and task_csv.exists() and task_csv.stat().st_size > 0:
            with task_csv.open() as f:
                task_rows.extend(list(csv.DictReader(f)))
        step_fields = list(ProbeStep.__dataclass_fields__.keys())

        for s in range(SEED_OFFSET, SEED_OFFSET + N_SEEDS):
            for mu in MUS:
                trial_id = f"m1r2_t{TASK_ID}_probeA_s{s}_mu{mu:g}"
                if trial_id in done:
                    print("skip", trial_id, flush=True)
                    continue
                rows, rec = run_probe_episode(env, seed_idx=s, mu=mu, trial_id=trial_id, dt=dt, tactile=True)
                task_rows.append(rec)
                fields = list(rec.keys())
                _write_csv(task_csv, task_rows, fields)
                _append_csv(episodes_csv, [rec], fields)
                _append_csv(steps_csv, [asdict(r) for r in rows], step_fields)
                done.add(trial_id)

        # Isaac camera annotator teardown can abort during interpreter cleanup on
        # this headless stack. Data has already been flushed after every episode,
        # so exit the per-task process directly.
        os._exit(0)
    except Exception as exc:
        print("EXCEPTION", repr(exc), flush=True)
        print(traceback.format_exc(), flush=True)
        (OUT / f"TASK{TASK_ID}_COLLECT_ERROR.json").write_text(
            json.dumps({"error": repr(exc), "trace": traceback.format_exc()}, indent=2)
        )
        try:
            simulation_app.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
