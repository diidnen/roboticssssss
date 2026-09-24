#!/usr/bin/env python3
"""Gate S1 oracle scripted force sweep — no learned policy, METHOD_CHANGE=NONE."""
from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

TABERO = Path("/home/exouser/Tabero")
OUT = Path(os.environ.get("GATE_S1_OUT", TABERO / "analysis/results/gate_s1_safety_substrate_latest"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)

sys.stdout = open(OUT / "logs" / "gate_s1_oracle.log", "w", buffering=1)
sys.stderr = sys.stdout

os.chdir(TABERO)
sys.path.insert(0, str(TABERO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(TABERO / "benchmarks/datasets/libero/assembled_hdf5")
os.environ.setdefault("LIBERO_CONFIG_DIR", str(TABERO / "benchmarks/datasets/libero/config"))
os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(TABERO / "benchmarks/datasets/libero/USD"))

ENV_ID = os.environ.get("GATE_S1_ENV_ID", "Isaac-Libero-Franka-Hybrid-Tactile-v0")
TASK_SUITE = os.environ.get("GATE_S1_TASK_SUITE", "libero_object")
TASK_ID = int(os.environ.get("GATE_S1_TASK_ID", "1"))
OBJ_NAME = os.environ.get("GATE_S1_OBJ", "cream_cheese_1")

FORCE_TARGETS = [float(x) for x in os.environ.get(
    "GATE_S1_FORCES", "0.5,1,2,4,6,8,12"
).split(",")]
N_SEEDS = int(os.environ.get("GATE_S1_N_SEEDS", "3"))
MASS_SCALES = [float(x) for x in os.environ.get(
    "GATE_S1_MASS_SCALES", "0.5,1.0,2.0"
).split(",")]
RUN_PHYSICS = os.environ.get("GATE_S1_RUN_PHYSICS", "1") == "1"

APPROACH_STEPS = int(os.environ.get("GATE_S1_APPROACH_STEPS", "50"))
DESCEND_STEPS = int(os.environ.get("GATE_S1_DESCEND_STEPS", "40"))
PRELOAD_STEPS = int(os.environ.get("GATE_S1_PRELOAD_STEPS", "20"))
HOLD_STEPS = int(os.environ.get("GATE_S1_HOLD_STEPS", "50"))
LIFT_STEPS = int(os.environ.get("GATE_S1_LIFT_STEPS", "40"))

PREGRASP_Z = float(os.environ.get("GATE_S1_PREGRASP_Z", "0.10"))
GRASP_Z = float(os.environ.get("GATE_S1_GRASP_Z", "0.018"))
LIFT_Z = float(os.environ.get("GATE_S1_LIFT_Z", "0.08"))
SLIP_XY_THRESH = float(os.environ.get("GATE_S1_SLIP_XY_M", "0.02"))
LIFT_SUCCESS_Z = float(os.environ.get("GATE_S1_LIFT_SUCCESS_Z", "0.03"))


@dataclass
class TrialRecord:
    trial_id: str
    task_suite: str
    task_id: int
    obj_name: str
    seed_idx: int
    mass_scale: float
    nominal_mass_kg: float | None
    target_squeeze_N: float
    phase: str
    step: int
    f_sq_target_N: float
    f_sq_meas_N: float
    f_sq_meas_raw_N: float
    fL_meas_z: float
    fR_meas_z: float
    applied_force_norm_N: float
    gripper_open_m: float
    ee_x: float
    ee_y: float
    ee_z: float
    obj_x: float
    obj_y: float
    obj_z: float
    obj_dx: float
    obj_dy: float
    obj_dz: float
    marker_motion_mean: float
    contact: int
    dropped: int
    success: int


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
        arr = np.asarray(x).reshape(-1)
        return float(arr[0])
    except Exception:
        return default


def _object_pose_in_base(env, obj_name: str):
    import isaaclab.utils.math as math_utils

    robot = env.scene["robot"]
    obj = env.scene[obj_name]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )
    return pos_b[0].detach().cpu().numpy(), quat_b[0].detach().cpu().numpy()


def _make_action(eef_pos, eef_aa, gripper_open, target_squeeze_N, device):
    import torch

    action = torch.zeros((1, 13), device=device)
    action[0, 0:3] = torch.tensor(eef_pos, device=device, dtype=torch.float32)
    action[0, 3:6] = torch.tensor(eef_aa, device=device, dtype=torch.float32)
    action[0, 6] = gripper_open
    half = float(target_squeeze_N) * 0.5
    action[0, 9] = half
    action[0, 12] = half
    return action


def _read_debug(env):
    try:
        term = env.action_manager.get_term("arm_action")
        return term.debug_info or {}
    except Exception:
        return {}


def _marker_motion_mean(obs) -> float:
    try:
        gmm = obs["policy"]["gripper_marker_motion"]
        arr = gmm[0].detach().cpu().numpy()
        return float(np.linalg.norm(arr))
    except Exception:
        return 0.0


NOMINAL_MASS_TENSOR = None
NOMINAL_INERTIA_TENSOR = None


def _apply_mass_scale(env, obj_name: str, scale: float) -> float:
    global NOMINAL_MASS_TENSOR, NOMINAL_INERTIA_TENSOR
    import torch

    obj = env.scene[obj_name]
    view = obj.root_physx_view
    if NOMINAL_MASS_TENSOR is None:
        NOMINAL_MASS_TENSOR = view.get_masses().clone()
        NOMINAL_INERTIA_TENSOR = view.get_inertias().clone()
    indices = torch.tensor([0], dtype=torch.int32)
    masses = (NOMINAL_MASS_TENSOR * scale).clone()
    inertias = (NOMINAL_INERTIA_TENSOR * scale).clone()
    view.set_masses(masses, indices)
    view.set_inertias(inertias, indices)
    return float(NOMINAL_MASS_TENSOR[0, 0].item())


def run_trial(env, seed_idx: int, mass_scale: float, target_squeeze_N: float, trial_prefix: str):
    import torch
    from benchmarks.common.metrics import compute_contact_force_metrics_from_lr_forces

    obs, _ = env.reset()
    nominal_mass = _apply_mass_scale(env, OBJ_NAME, mass_scale)

    # ForcePositionAction / DiffIK expect EEF pose in the *robot base* frame.
    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    ee_pos0 = eef0[:3].copy()
    eef_aa = _axis_angle_from_quat(eef0[3:7])

    obj0_w = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
    obj0_b, _ = _object_pose_in_base(env, OBJ_NAME)

    pregrasp = obj0_b.copy()
    pregrasp[2] += PREGRASP_Z
    grasp = obj0_b.copy()
    grasp[2] += GRASP_Z
    lift = grasp.copy()
    lift[2] += LIFT_Z

    if trial_prefix.endswith("_s0"):
        print(
            f"pose_debug {trial_prefix} ee_base={ee_pos0} obj_base={obj0_b} obj_world={obj0_w}",
            flush=True,
        )

    rows: list[TrialRecord] = []
    # ForcePositionAction uses action[6] as predicted opening d_pred, then only
    # a small squeeze_kp correction. Oracle must close d_pred to make contact.
    phase_plan = (
        [("approach", APPROACH_STEPS, pregrasp, 0.04, 0.0)]
        + [("descend", DESCEND_STEPS, grasp, 0.04, 0.0)]
        + [("close", PRELOAD_STEPS, grasp, 0.0, 0.5)]
        + [("hold", HOLD_STEPS, grasp, 0.0, target_squeeze_N)]
        + [("lift", LIFT_STEPS, lift, 0.0, target_squeeze_N)]
    )

    step_global = 0
    fL_hold, fR_hold = [], []
    current_cmd_pos = ee_pos0.copy()

    for phase, n_steps, target_pos, grip_open, f_target in phase_plan:
        start_pos = current_cmd_pos.copy()
        start_grip = 0.04 if phase in ("approach", "descend") else (0.04 if phase == "close" else grip_open)
        end_grip = grip_open
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            pos = (1 - alpha) * start_pos + alpha * target_pos
            current_cmd_pos = pos
            grip_now = (1 - alpha) * start_grip + alpha * end_grip if phase == "close" else end_grip

            action = _make_action(pos, eef_aa, grip_now, f_target, env.device)
            obs, rew, term, trunc, info = env.step(action)
            step_global += 1

            pol = obs["policy"]
            f = pol["gripper_net_force"][0]
            if f.ndim == 3:
                f = f[-1]
            fL = f[0].detach().cpu().numpy()
            fR = f[1].detach().cpu().numpy()

            dbg = _read_debug(env)
            f_sq_meas = _to_float(dbg.get("f_sq_meas"), 0.0)
            f_sq_meas_raw = _to_float(dbg.get("f_sq_meas_raw"), f_sq_meas)
            f_sq_target = _to_float(dbg.get("f_sq_pred_eff", dbg.get("f_sq_pred")), f_target)
            f_app_vec = dbg.get("F_app_meas_b")
            if f_app_vec is None:
                f_app = 0.0
            else:
                try:
                    import torch as _t

                    if isinstance(f_app_vec, _t.Tensor):
                        f_app = float(_t.linalg.vector_norm(f_app_vec.reshape(-1, 3)[0]).item())
                    else:
                        f_app = float(np.linalg.norm(np.asarray(f_app_vec).reshape(-1)[:3]))
                except Exception:
                    f_app = 0.0

            obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
            ee_obs = pol["eef_pose"][0].detach().cpu().numpy()
            # gripper_pos is (q_left, -q_right); use |q_left| as opening.
            gp = float(abs(pol["gripper_pos"][0, 0].detach().cpu().item()))

            contact = int((np.abs(fL).sum() + np.abs(fR).sum()) > 1e-3)
            dropped = 0
            try:
                dropped = int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
            except Exception:
                pass
            success = 0
            try:
                success = int(bool(env.termination_manager.get_term("success")[0].item()))
            except Exception:
                pass

            if phase in ("hold", "lift"):
                fL_hold.append(fL)
                fR_hold.append(fR)

            rows.append(
                TrialRecord(
                    trial_id=trial_prefix,
                    task_suite=TASK_SUITE,
                    task_id=TASK_ID,
                    obj_name=OBJ_NAME,
                    seed_idx=seed_idx,
                    mass_scale=mass_scale,
                    nominal_mass_kg=nominal_mass,
                    target_squeeze_N=target_squeeze_N,
                    phase=phase,
                    step=step_global,
                    f_sq_target_N=f_sq_target,
                    f_sq_meas_N=f_sq_meas,
                    f_sq_meas_raw_N=f_sq_meas_raw,
                    fL_meas_z=float(fL[2]),
                    fR_meas_z=float(fR[2]),
                    applied_force_norm_N=f_app,
                    gripper_open_m=float(gp),
                    ee_x=float(ee_obs[0]),
                    ee_y=float(ee_obs[1]),
                    ee_z=float(ee_obs[2]),
                    obj_x=float(obj_p[0]),
                    obj_y=float(obj_p[1]),
                    obj_z=float(obj_p[2]),
                    obj_dx=float(obj_p[0] - obj0_w[0]),
                    obj_dy=float(obj_p[1] - obj0_w[1]),
                    obj_dz=float(obj_p[2] - obj0_w[2]),
                    marker_motion_mean=_marker_motion_mean(obs),
                    contact=contact,
                    dropped=dropped,
                    success=success,
                )
            )

            if term[0] or trunc[0]:
                break
        if term[0] or trunc[0]:
            break

    summary = {
        "trial_id": trial_prefix,
        "seed_idx": seed_idx,
        "mass_scale": mass_scale,
        "target_squeeze_N": target_squeeze_N,
        "nominal_mass_kg": nominal_mass,
        "steps": step_global,
    }
    if fL_hold:
        m = compute_contact_force_metrics_from_lr_forces(np.stack(fL_hold), np.stack(fR_hold))
        hold = [r for r in rows if r.phase == "hold"]
        lift_rows = [r for r in rows if r.phase == "lift"]
        last = lift_rows[-1] if lift_rows else hold[-1] if hold else rows[-1]
        obj_disp_xy = float(np.linalg.norm([last.obj_dx, last.obj_dy]))
        lift_z = float(lift_rows[-1].obj_dz) if lift_rows else float("nan")
        ee_err = float(np.linalg.norm(np.array([rows[-1].ee_x, rows[-1].ee_y, rows[-1].ee_z]) - current_cmd_pos))
        lifted = int(lift_z >= LIFT_SUCCESS_Z) if lift_rows else 0
        summary.update(
            {
                "hold_contact_ratio": m.contact_ratio,
                "hold_mean_squeeze_N": m.squeeze_mean,
                "hold_max_squeeze_N": m.squeeze_max,
                "hold_mean_applied_N": m.external_norm_mean,
                "hold_max_applied_N": m.external_norm_max,
                "obj_disp_xy_m": obj_disp_xy,
                "obj_disp_z_m": last.obj_dz,
                "obj_lift_z_m": lift_z,
                "lifted": lifted,
                "ee_final_tracking_err_m": ee_err,
                "slip": int(obj_disp_xy > SLIP_XY_THRESH),
                "dropped": last.dropped,
                "success": last.success,
                "marker_motion_mean_hold": float(np.mean([r.marker_motion_mean for r in hold])) if hold else 0.0,
                "grasp_stable": int(
                    m.contact_ratio > 0.3
                    and last.dropped == 0
                    and lifted == 1
                ),
            }
        )
    else:
        summary.update({"grasp_stable": 0, "hold_contact_ratio": 0.0, "lifted": 0})

    print(
        f"summary {trial_prefix} contact={summary.get('hold_contact_ratio')} "
        f"fsq={summary.get('hold_mean_squeeze_N')} lifted={summary.get('lifted')} "
        f"grip={rows[-1].gripper_open_m:.4f} ee=({rows[-1].ee_x:.3f},{rows[-1].ee_y:.3f},{rows[-1].ee_z:.3f})",
        flush=True,
    )
    return rows, summary


def main():
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=True,
        enable_cameras=os.environ.get("GATE_S1_ENABLE_CAMERAS", "1") == "1",
        num_envs=1,
    )
    simulation_app = app_launcher.app

    all_rows: list[TrialRecord] = []
    summaries = []

    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(TASK_SUITE, TASK_ID)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        # Scripted grasp+lift exceeds the default 8s Libero timeout (160 steps).
        env_cfg.episode_length_s = float(os.environ.get("GATE_S1_EPISODE_S", "30"))
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        print("env ready episode_length_s", env_cfg.episode_length_s, flush=True)
        print("env ready", flush=True)

        # nominal mass from sim
        obs, _ = env.reset()
        nominal_mass = float(env.scene[OBJ_NAME].root_physx_view.get_masses()[0, 0].item())
        print(f"nominal_mass_kg={nominal_mass}", flush=True)

        trial_counter = 0
        for mass_scale in ([1.0] if not RUN_PHYSICS else MASS_SCALES):
            for f_target in FORCE_TARGETS:
                for seed_idx in range(N_SEEDS):
                    trial_counter += 1
                    tid = f"m{mass_scale:g}_f{f_target:g}_s{seed_idx}"
                    print(f"=== trial {tid} ===", flush=True)
                    rows, summary = run_trial(env, seed_idx, mass_scale, f_target, tid)
                    all_rows.extend(rows)
                    summaries.append(summary)

        # write CSVs
        sweep_path = OUT / "FORCE_SWEEP_RESULTS.csv"
        with sweep_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(asdict(r))

        cal_path = OUT / "FORCE_CALIBRATION.csv"
        with cal_path.open("w", newline="") as f:
            fieldnames = list(summaries[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for s in summaries:
                w.writerow(s)

        if RUN_PHYSICS and len(MASS_SCALES) > 1:
            phys_path = OUT / "PHYSICS_SWEEP_RESULTS.csv"
            with phys_path.open("w", newline="") as f:
                fieldnames = list(summaries[0].keys())
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for s in summaries:
                    w.writerow(s)
        else:
            (OUT / "PHYSICS_SWEEP_RESULTS.csv").write_text("NOT_RUN\nreason: mass_scale sweep disabled or single scale only\n")

        meta = {
            "env_id": ENV_ID,
            "task_suite": TASK_SUITE,
            "task_id": TASK_ID,
            "obj_name": OBJ_NAME,
            "force_targets": FORCE_TARGETS,
            "mass_scales": MASS_SCALES if RUN_PHYSICS else [1.0],
            "n_seeds": N_SEEDS,
            "nominal_mass_kg": nominal_mass,
            "n_trials": len(summaries),
            "n_rows": len(all_rows),
        }
        (OUT / "SWEEP_META.json").write_text(json.dumps(meta, indent=2))

        # quick calibration monotonicity on nominal mass hold summaries
        nom = [s for s in summaries if s["mass_scale"] == 1.0]
        by_f = {}
        for s in nom:
            by_f.setdefault(s["target_squeeze_N"], []).append(s.get("hold_mean_squeeze_N", 0.0))
        cal = {str(k): {"mean_meas": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in sorted(by_f.items())}
        (OUT / "FORCE_CALIBRATION_SUMMARY.json").write_text(json.dumps(cal, indent=2))
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
