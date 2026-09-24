#!/usr/bin/env python3
"""P1 collector: 4 N matched-friction lift telemetry. METHOD_CHANGE=NONE.

Not a method. Records force / tactile / motion through lift until loss or stable lift.
GT_SLIP uses F1R2 / R1-v1 gross-slip rule (8 mm, v_rel < -0.05 for 2 steps).
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
OUT = Path(os.environ.get("P1_OUT", Path(__file__).resolve().parents[1]))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

sys.stdout = open(OUT / "logs" / "p1_isaac.log", "w", buffering=1)
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

N_SEEDS = int(os.environ.get("P1_N_SEEDS", "20"))
FORCE_N = float(os.environ.get("P1_FORCE", "4"))
FRICTIONS = [float(x) for x in os.environ.get("P1_FRICTIONS", "0.2,0.5,1.0").split(",")]

APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS, HOLD_STEPS = 45, 35, 70, 40
LIFT_STEPS = int(os.environ.get("P1_LIFT_STEPS", "50"))
PREGRASP_Z, GRASP_Z, LIFT_Z = 0.10, 0.018, 0.16

SERVO_STEP, SERVO_DEADBAND = 0.0006, 0.4
D_OPEN, D_CLOSED = 0.04, 0.0
LIFT_SUCCESS_Z = 0.03

# Gross slip = F1R2 / R1 v1 (do not retune to manufacture a window)
REL_Z_SLIP, V_REL_SLIP, VNEG_RUN_N, XY_SLIP = 0.008, -0.05, 2, 0.015
# Micro-event anchor only
REL_Z_MICRO, V_REL_MICRO = 0.002, -0.015
LOSS_F_N, LOSS_STREAK = 0.3, 3


def _aa(q):
    w, x, y, z = q
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    if s < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (np.array([x, y, z], dtype=np.float32) / s) * angle


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


def _make_action(eef_pos, eef_aa, d_pred, f_star, device):
    import torch

    a = torch.zeros((1, 13), device=device)
    a[0, 0:3] = torch.tensor(np.asarray(eef_pos, dtype=np.float32), device=device)
    a[0, 3:6] = torch.tensor(np.asarray(eef_aa, dtype=np.float32), device=device)
    a[0, 6] = float(d_pred)
    a[0, 9] = 0.5 * f_star
    a[0, 12] = 0.5 * f_star
    return a


def force_servo(d_pred, f_meas, f_star):
    err = f_star - f_meas
    if abs(err) <= SERVO_DEADBAND:
        return float(np.clip(d_pred, D_CLOSED, D_OPEN))
    d_pred = d_pred - SERVO_STEP if err > 0 else d_pred + SERVO_STEP
    return float(np.clip(d_pred, D_CLOSED, D_OPEN))


def _interp(start, end, i, n):
    a = (i + 1) / n
    return (1 - a) * start + a * end


NOMINAL_MAT = None


def _apply_friction(env, name, mu):
    global NOMINAL_MAT
    import torch

    view = env.scene[name].root_physx_view
    if NOMINAL_MAT is None:
        NOMINAL_MAT = view.get_material_properties().clone()
    mats = NOMINAL_MAT.clone()
    mats[..., 0] = mu
    mats[..., 1] = mu
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
        "marker_mean": 0.0, "marker_max": 0.0, "marker_std": 0.0, "marker_asym": 0.0,
        "marker_tangential": 0.0, "hm_mean": 0.0, "hm_max": 0.0, "hm_std": 0.0, "rgb_mean": 0.0,
        "tactile_ok": 0,
    }
    try:
        mm = obs["policy"]["gripper_marker_motion"][0].detach().cpu().numpy()
        # (S=2, T=2, M, 2)  T: init, current
        cur, init = mm[:, 1], mm[:, 0]
        disp = cur - init
        mag = np.linalg.norm(disp, axis=-1)  # (S, M)
        out["marker_mean"] = float(mag.mean())
        out["marker_max"] = float(mag.max()) if mag.size else 0.0
        out["marker_std"] = float(mag.std())
        left, right = mag[0].mean() if mag.shape[0] > 0 else 0.0, mag[1].mean() if mag.shape[0] > 1 else 0.0
        out["marker_asym"] = float(abs(left - right))
        out["marker_tangential"] = float(np.linalg.norm(disp.mean(axis=1), axis=-1).mean())
        out["tactile_ok"] = 1
    except Exception:
        pass
    for side, key_hm, key_rgb in (("gsmini_left", "hm_mean", "rgb_mean"),):
        try:
            sen = env.scene[side]
            hm = sen.data.output.get("height_map")
            if hm is not None:
                arr = hm[0].detach().cpu().numpy()
                out["hm_mean"] = float(arr.mean())
                out["hm_max"] = float(arr.max())
                out["hm_std"] = float(arr.std())
            rgb = sen.data.output.get("tactile_rgb")
            if rgb is not None:
                out["rgb_mean"] = float(rgb[0].float().mean().cpu().item())
        except Exception:
            pass
    try:
        sen_r = env.scene["gsmini_right"]
        hm = sen_r.data.output.get("height_map")
        if hm is not None:
            arr = hm[0].detach().cpu().numpy()
            out["hm_mean"] = 0.5 * (out["hm_mean"] + float(arr.mean()))
            out["hm_max"] = max(out["hm_max"], float(arr.max()))
    except Exception:
        pass
    return out


def _append(path: Path, dicts, fields):
    if not dicts:
        return
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for d in dicts:
            w.writerow(d)


@dataclass
class Step:
    trial_id: str
    seed_idx: int
    friction: float
    phase: str
    step: int
    t_s: float
    t_from_lift: float
    f_cmd: float
    f_meas: float
    dF: float
    fL_x: float
    fL_y: float
    fL_z: float
    fR_x: float
    fR_y: float
    fR_z: float
    imbalance: float
    f_tangential: float
    d_pred: float
    gripper_open: float
    ee_x: float
    ee_y: float
    ee_z: float
    ee_vx: float
    ee_vy: float
    ee_vz: float
    obj_x: float
    obj_y: float
    obj_z: float
    obj_vx: float
    obj_vy: float
    obj_vz: float
    obj_wx: float
    obj_wy: float
    obj_wz: float
    rel_x: float
    rel_y: float
    rel_z: float
    v_rel_x: float
    v_rel_y: float
    v_rel_z: float
    rel_xy: float
    contact: int
    gt_slip: int
    gt_micro: int
    gt_loss: int
    dropped: int
    marker_mean: float
    marker_max: float
    marker_std: float
    marker_asym: float
    marker_tangential: float
    marker_vel: float
    hm_mean: float
    hm_max: float
    hm_std: float
    rgb_mean: float
    tactile_ok: int


def run_episode(env, *, seed_idx: int, mu: float, trial_id: str, dt: float, tactile: bool):
    import torch

    try:
        obs, _ = env.reset(seed=int(seed_idx))
    except TypeError:
        torch.manual_seed(int(seed_idx))
        np.random.seed(int(seed_idx))
        obs, _ = env.reset()
    applied = _apply_friction(env, OBJ_NAME, mu)

    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    ee_pos = eef0[:3].copy()
    eef_aa = _aa(eef0[3:7])
    obj0_w = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
    obj0_b, _ = _pose_in_base(env, OBJ_NAME)
    pregrasp = obj0_b.copy(); pregrasp[2] += PREGRASP_Z
    grasp = obj0_b.copy(); grasp[2] += GRASP_Z
    lift = grasp.copy(); lift[2] += LIFT_Z

    d_pred = D_OPEN
    cmd_pos = ee_pos.copy()
    prev_ee = ee_pos.copy()
    prev_f = None
    prev_rel = None
    prev_marker = None
    grasp_rel = None
    vneg = 0
    contact_ever = 0
    high_f_ever = 0
    loss_streak = 0
    t_lift0 = None
    t_micro = t_slip = t_loss = None
    lift_success = 0
    dropped = 0
    rows: list[Step] = []
    step = 0
    f_cmd = FORCE_N

    phases = [
        ("approach", APPROACH_STEPS, pregrasp, "open"),
        ("descend", DESCEND_STEPS, grasp, "open"),
        ("close", CLOSE_STEPS, grasp, "track"),
        ("hold", HOLD_STEPS, grasp, "track"),
        ("lift", LIFT_STEPS, lift, "track"),
    ]

    for phase, n_steps, target, mode in phases:
        start = cmd_pos.copy()
        for i in range(n_steps):
            cmd_pos = _interp(start, target, i, n_steps)
            if mode == "open":
                d_pred = D_OPEN
            action = _make_action(cmd_pos, eef_aa, d_pred, f_cmd if mode != "open" else 0.0, env.device)
            obs, rew, term, trunc, info = env.step(action)
            step += 1
            t_s = step * dt
            if phase == "lift" and t_lift0 is None:
                t_lift0 = t_s

            pol = obs["policy"]
            f = pol["gripper_net_force"][0]
            if f.ndim == 3:
                f = f[-1]
            fL = f[0].detach().cpu().numpy()
            fR = f[1].detach().cpu().numpy()
            f_sq = _f(_dbg(env).get("f_sq_meas"), 0.0)
            if mode == "track":
                d_pred = force_servo(d_pred, f_sq, f_cmd)
            dF = 0.0 if prev_f is None else (f_sq - prev_f) / dt
            prev_f = f_sq
            if f_sq > 2.0:
                high_f_ever = 1

            obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
            try:
                obj_v = env.scene[OBJ_NAME].data.root_lin_vel_w[0].detach().cpu().numpy()
                obj_w = env.scene[OBJ_NAME].data.root_ang_vel_w[0].detach().cpu().numpy()
            except Exception:
                obj_v = np.zeros(3); obj_w = np.zeros(3)
            ee = pol["eef_pose"][0].detach().cpu().numpy()
            ee_xyz = ee[:3]
            ee_v = (ee_xyz - prev_ee) / dt
            prev_ee = ee_xyz.copy()
            gp = float(abs(pol["gripper_pos"][0, 0].detach().cpu().item()))
            contact = int((np.abs(fL).sum() + np.abs(fR).sum()) > 1e-3)
            if contact:
                contact_ever = 1
            try:
                dropped = int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
            except Exception:
                pass

            rel = obj_p - ee_xyz
            if phase == "close" and i == n_steps - 1:
                grasp_rel = rel.copy()
            if grasp_rel is None and phase in ("hold", "lift") and contact:
                grasp_rel = rel.copy()
            rel_d = rel - grasp_rel if grasp_rel is not None else np.zeros(3)
            v_rel = np.zeros(3) if prev_rel is None else (rel_d - prev_rel) / dt
            prev_rel = rel_d.copy()
            if v_rel[2] < V_REL_SLIP:
                vneg += 1
            else:
                vneg = 0

            gt_slip = 0
            if phase in ("hold", "lift"):
                if dropped or (grasp_rel is not None and rel_d[2] < -REL_Z_SLIP):
                    gt_slip = 1
                if phase == "lift" and vneg >= VNEG_RUN_N:
                    gt_slip = 1
                if grasp_rel is not None and np.linalg.norm(rel_d[:2]) > XY_SLIP:
                    gt_slip = 1
                if contact_ever and (not contact) and phase == "lift":
                    gt_slip = 1
            gt_micro = 0
            if phase == "lift" and grasp_rel is not None:
                if rel_d[2] < -REL_Z_MICRO or v_rel[2] < V_REL_MICRO:
                    gt_micro = 1

            if high_f_ever and f_sq < LOSS_F_N and phase == "lift":
                loss_streak += 1
            else:
                loss_streak = 0
            gt_loss = int(dropped or (phase == "lift" and contact_ever and not contact and loss_streak >= 1) or loss_streak >= LOSS_STREAK)

            if gt_micro and t_micro is None:
                t_micro = t_s
            if gt_slip and t_slip is None:
                t_slip = t_s
            if gt_loss and t_loss is None:
                t_loss = t_s

            obj_dz = float(obj_p[2] - obj0_w[2])
            if obj_dz >= LIFT_SUCCESS_Z:
                lift_success = 1

            tac = _tactile_summaries(env, obs) if tactile else {
                "marker_mean": 0.0, "marker_max": 0.0, "marker_std": 0.0, "marker_asym": 0.0,
                "marker_tangential": 0.0, "hm_mean": 0.0, "hm_max": 0.0, "hm_std": 0.0, "rgb_mean": 0.0,
                "tactile_ok": 0,
            }
            mvel = 0.0 if prev_marker is None else (tac["marker_mean"] - prev_marker) / dt
            prev_marker = tac["marker_mean"]

            imb = float(abs(np.linalg.norm(fL) - np.linalg.norm(fR)))
            ftan = float(np.linalg.norm(np.concatenate([fL[:2], fR[:2]])))

            rows.append(
                Step(
                    trial_id=trial_id, seed_idx=seed_idx, friction=mu, phase=phase, step=step, t_s=t_s,
                    t_from_lift=-1.0 if t_lift0 is None else t_s - t_lift0,
                    f_cmd=f_cmd, f_meas=f_sq, dF=dF,
                    fL_x=float(fL[0]), fL_y=float(fL[1]), fL_z=float(fL[2]),
                    fR_x=float(fR[0]), fR_y=float(fR[1]), fR_z=float(fR[2]),
                    imbalance=imb, f_tangential=ftan, d_pred=d_pred, gripper_open=gp,
                    ee_x=float(ee_xyz[0]), ee_y=float(ee_xyz[1]), ee_z=float(ee_xyz[2]),
                    ee_vx=float(ee_v[0]), ee_vy=float(ee_v[1]), ee_vz=float(ee_v[2]),
                    obj_x=float(obj_p[0]), obj_y=float(obj_p[1]), obj_z=float(obj_p[2]),
                    obj_vx=float(obj_v[0]), obj_vy=float(obj_v[1]), obj_vz=float(obj_v[2]),
                    obj_wx=float(obj_w[0]), obj_wy=float(obj_w[1]), obj_wz=float(obj_w[2]),
                    rel_x=float(rel_d[0]), rel_y=float(rel_d[1]), rel_z=float(rel_d[2]),
                    v_rel_x=float(v_rel[0]), v_rel_y=float(v_rel[1]), v_rel_z=float(v_rel[2]),
                    rel_xy=float(np.linalg.norm(rel_d[:2])),
                    contact=contact, gt_slip=gt_slip, gt_micro=gt_micro, gt_loss=gt_loss, dropped=dropped,
                    marker_mean=tac["marker_mean"], marker_max=tac["marker_max"], marker_std=tac["marker_std"],
                    marker_asym=tac["marker_asym"], marker_tangential=tac["marker_tangential"], marker_vel=mvel,
                    hm_mean=tac["hm_mean"], hm_max=tac["hm_max"], hm_std=tac["hm_std"], rgb_mean=tac["rgb_mean"],
                    tactile_ok=tac["tactile_ok"],
                )
            )
            if bool(term[0].item()) or bool(trunc[0].item()):
                break
        else:
            continue
        break

    future_fail = int(lift_success == 0)
    ev = {
        "trial_id": trial_id, "seed_idx": seed_idx, "friction": mu, "force_N": FORCE_N,
        "friction_applied": applied, "lift_success": lift_success, "future_fail": future_fail,
        "t_lift0": t_lift0, "t_micro": t_micro, "t_slip": t_slip, "t_loss": t_loss,
        "slip_minus_lift_s": None if (t_slip is None or t_lift0 is None) else t_slip - t_lift0,
        "loss_minus_slip_s": None if (t_loss is None or t_slip is None) else t_loss - t_slip,
        "loss_minus_lift_s": None if (t_loss is None or t_lift0 is None) else t_loss - t_lift0,
        "dropped": dropped, "steps": step, "tactile": int(tactile and any(r.tactile_ok for r in rows)),
    }
    print(
        f"summary {trial_id} lift={lift_success} fail={future_fail} "
        f"T0={t_lift0} Tmicro={t_micro} Tslip={t_slip} Tloss={t_loss}",
        flush=True,
    )
    return rows, ev


def main():
    from isaaclab.app import AppLauncher

    enable_cameras = os.environ.get("P1_ENABLE_CAMERAS", "1") not in ("0", "false", "False")
    app_launcher = AppLauncher(headless=True, enable_cameras=enable_cameras, num_envs=1)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(TASK_SUITE, TASK_ID)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = float(os.environ.get("P1_EPISODE_S", "25"))
        tactile = enable_cameras
        if not enable_cameras:
            for name in ("eye_in_hand_cam", "agentview_cam", "gsmini_left", "gsmini_right"):
                if hasattr(env_cfg.scene, name):
                    delattr(env_cfg.scene, name)
            pol = getattr(env_cfg.observations, "policy", None)
            if pol is not None and hasattr(pol, "gripper_marker_motion"):
                delattr(pol, "gripper_marker_motion")
            tactile = False
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        print("env ready", "dt", dt, "tactile", tactile, flush=True)
        (OUT / "COLLECT_META.json").write_text(json.dumps({"dt": dt, "tactile": tactile, "n_seeds": N_SEEDS, "force": FORCE_N, "frictions": FRICTIONS}, indent=2))

        ev_path = OUT / "EVENT_TIMES.csv"
        done = set()
        events = []
        if ev_path.exists() and ev_path.stat().st_size > 0:
            with ev_path.open() as f:
                for row in csv.DictReader(f):
                    done.add(row["trial_id"])
                    events.append(row)
        print("resume", len(done), flush=True)
        fields = list(Step.__dataclass_fields__.keys())
        ev_fields = [
            "trial_id", "seed_idx", "friction", "force_N", "friction_applied", "lift_success", "future_fail",
            "t_lift0", "t_micro", "t_slip", "t_loss", "slip_minus_lift_s", "loss_minus_slip_s",
            "loss_minus_lift_s", "dropped", "steps", "tactile",
        ]

        for s in range(N_SEEDS):
            for mu in FRICTIONS:
                tid = f"p1_s{s}_mu{mu:g}"
                if tid in done:
                    print("skip", tid, flush=True)
                    continue
                print("===", tid, "===", flush=True)
                rows, ev = run_episode(env, seed_idx=s, mu=mu, trial_id=tid, dt=dt, tactile=tactile)
                events.append(ev)
                with ev_path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=ev_fields)
                    w.writeheader()
                    for e in events:
                        w.writerow(e)
                _append(OUT / "STEP_TRAJECTORIES.csv", [asdict(r) for r in rows], fields)

        print("done n_events", len(events), flush=True)
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
