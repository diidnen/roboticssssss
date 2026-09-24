#!/usr/bin/env python3
"""T1 oracle: calibrated grip-force servo + full pick-place × friction.

METHOD_CHANGE=NONE. Analysis-only controller instrumentation. No Tabero source edits.
POLICY_DOWNSTREAM_REPRODUCTION is not performed here (see ENV_PROVENANCE).
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
OUT = Path(os.environ.get("T1_OUT", TABERO / "analysis/results/t1_tabero_downstream_decision_value_latest"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

sys.stdout = open(OUT / "logs" / "t1_oracle.log", "w", buffering=1)
sys.stderr = sys.stdout

os.chdir(TABERO)
sys.path.insert(0, str(TABERO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(TABERO / "benchmarks/datasets/libero/assembled_hdf5")
os.environ.setdefault("LIBERO_CONFIG_DIR", str(TABERO / "benchmarks/datasets/libero/config"))
os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(TABERO / "benchmarks/datasets/libero/USD"))

ENV_ID = os.environ.get("T1_ENV_ID", "Isaac-Libero-Franka-Hybrid-Tactile-v0")
TASK_SUITE = os.environ.get("T1_TASK_SUITE", "libero_object")
TASK_ID = int(os.environ.get("T1_TASK_ID", "1"))
OBJ_NAME = os.environ.get("T1_OBJ", "cream_cheese_1")
BASKET_NAME = os.environ.get("T1_BASKET", "basket_1")
STAGE = os.environ.get("T1_STAGE", "calib")  # calib | full | both

FORCE_TARGETS = [float(x) for x in os.environ.get("T1_FORCES", "2,4,6,8,10,12").split(",")]
FRICTION_VALUES = [float(x) for x in os.environ.get("T1_FRICTIONS", "0.2,0.5,1.0").split(",")]
N_SEEDS_CALIB = int(os.environ.get("T1_N_SEEDS_CALIB", "3"))
N_SEEDS_FULL = int(os.environ.get("T1_N_SEEDS_FULL", "5"))

APPROACH_STEPS = int(os.environ.get("T1_APPROACH_STEPS", "45"))
DESCEND_STEPS = int(os.environ.get("T1_DESCEND_STEPS", "35"))
CLOSE_STEPS = int(os.environ.get("T1_CLOSE_STEPS", "70"))
HOLD_STEPS = int(os.environ.get("T1_HOLD_STEPS", "40"))
LIFT_STEPS = int(os.environ.get("T1_LIFT_STEPS", "35"))
TRANSIT_STEPS = int(os.environ.get("T1_TRANSIT_STEPS", "110"))
OVER_BASKET_STEPS = int(os.environ.get("T1_OVER_BASKET_STEPS", "30"))
PLACE_STEPS = int(os.environ.get("T1_PLACE_STEPS", "40"))
RELEASE_STEPS = int(os.environ.get("T1_RELEASE_STEPS", "50"))
SETTLE_STEPS = int(os.environ.get("T1_SETTLE_STEPS", "50"))

PREGRASP_Z = float(os.environ.get("T1_PREGRASP_Z", "0.10"))
GRASP_Z = float(os.environ.get("T1_GRASP_Z", "0.018"))
LIFT_Z = float(os.environ.get("T1_LIFT_Z", "0.16"))
PLACE_CLEAR_Z = float(os.environ.get("T1_PLACE_CLEAR_Z", "0.18"))
PLACE_Z = float(os.environ.get("T1_PLACE_Z", "0.10"))

SERVO_STEP = float(os.environ.get("T1_SERVO_STEP", "0.0006"))
SERVO_DEADBAND = float(os.environ.get("T1_SERVO_DEADBAND", "0.4"))
D_OPEN = 0.04
D_CLOSED = 0.0

LIFT_SUCCESS_Z = 0.03
SLIP_XY = 0.03
LOW_FORCE_GATE_N = 10.0


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
    """Non-learned gripper-opening servo toward measured squeeze F*."""
    err = f_star - f_meas
    if abs(err) <= SERVO_DEADBAND:
        return float(np.clip(d_pred, D_CLOSED, D_OPEN))
    if err > 0:
        d_pred = d_pred - SERVO_STEP
    else:
        d_pred = d_pred + SERVO_STEP
    return float(np.clip(d_pred, D_CLOSED, D_OPEN))


NOMINAL_MAT = None  # (1, S, 3) static, dynamic, restitution


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
    # materials: (..., 3) = static, dynamic, restitution
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


@dataclass
class StepRow:
    trial_id: str
    stage: str
    seed_idx: int
    friction: float
    desired_F: float
    phase: str
    step: int
    d_pred: float
    f_sq_meas: float
    f_sq_raw: float
    applied: float
    gripper_open: float
    ee_x: float
    ee_y: float
    ee_z: float
    obj_x: float
    obj_y: float
    obj_z: float
    obj_dz: float
    marker: float
    contact: int
    dropped: int
    official_success: int
    grasp_flag: int
    xy_to_basket: float
    z_to_basket: float
    basket_contact: float


def _interp(start, end, i, n):
    a = (i + 1) / n
    return (1 - a) * start + a * end


def run_episode(env, *, stage: str, seed_idx: int, mu: float, desired_F: float, do_place: bool, trial_id: str):
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
    transit = basket_b.copy(); transit[2] = lift[2]
    if transit[2] < basket_b[2] + PLACE_CLEAR_Z:
        transit[2] = basket_b[2] + PLACE_CLEAR_Z
    place = basket_b.copy(); place[2] = basket_b[2] + PLACE_Z

    if seed_idx == 0:
        print(f"pose {trial_id} ee={ee_pos} obj={obj0_b} basket={basket_b} fr={fr_info}", flush=True)

    d_pred = D_OPEN
    cmd_pos = ee_pos.copy()
    rows: list[StepRow] = []
    fL_hold, fR_hold = [], []
    step = 0
    pick_success = 0
    lift_success = 0
    lost_in_transit = 0
    retained_over_basket = 0
    placed = 0
    official_success = 0
    dropped = 0
    timed_out = 0
    max_obj_xy = 0.0
    peak_z = obj0_w[2]
    max_basket_contact = 0.0
    xy_to_basket_final = 0.0
    z_to_basket_final = 0.0
    obj_dz_over_basket = 0.0

    def phases():
        # servo_mode: open | track | freeze
        yield "approach", APPROACH_STEPS, pregrasp, "open", 0.0
        yield "descend", DESCEND_STEPS, grasp, "open", 0.0
        yield "close", CLOSE_STEPS, grasp, "track", desired_F
        yield "hold", HOLD_STEPS, grasp, "track", desired_F
        if do_place:
            yield "lift", LIFT_STEPS, lift, "freeze", desired_F
            yield "transit", TRANSIT_STEPS, transit, "freeze", desired_F
            yield "over_basket", OVER_BASKET_STEPS, transit, "freeze", desired_F
            yield "place", PLACE_STEPS, place, "freeze", desired_F
            yield "release", RELEASE_STEPS, place, "open", 0.0
            yield "settle", SETTLE_STEPS, place, "open", 0.0

    for phase, n_steps, target, servo_mode, f_star in phases():
        start = cmd_pos.copy()
        for i in range(n_steps):
            cmd_pos = _interp(start, target, i, n_steps)
            if servo_mode == "open":
                d_pred = D_OPEN
            action = _make_action(cmd_pos, eef_aa, d_pred, f_star if servo_mode != "open" else 0.0, env.device)
            obs, rew, term, trunc, info = env.step(action)
            step += 1

            pol = obs["policy"]
            f = pol["gripper_net_force"][0]
            if f.ndim == 3:
                f = f[-1]
            fL = f[0].detach().cpu().numpy()
            fR = f[1].detach().cpu().numpy()
            dbg = _read_debug(env)
            f_sq = _to_float(dbg.get("f_sq_meas"), 0.0)
            f_raw = _to_float(dbg.get("f_sq_meas_raw"), f_sq)
            f_app_vec = dbg.get("F_app_meas_b")
            applied = 0.0
            if f_app_vec is not None:
                try:
                    import torch as _t

                    if isinstance(f_app_vec, _t.Tensor):
                        applied = float(_t.linalg.vector_norm(f_app_vec.reshape(-1, 3)[0]).item())
                except Exception:
                    applied = 0.0
            if servo_mode == "track":
                d_pred = force_servo(d_pred, f_sq, f_star)

            obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
            ee = pol["eef_pose"][0].detach().cpu().numpy()
            gp = float(abs(pol["gripper_pos"][0, 0].detach().cpu().item()))
            try:
                mm = float(np.linalg.norm(pol["gripper_marker_motion"][0].detach().cpu().numpy()))
            except Exception:
                mm = 0.0
            contact = int((np.abs(fL).sum() + np.abs(fR).sum()) > 1e-3)
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

            basket_w = env.scene[BASKET_NAME].data.root_pos_w[0].detach().cpu().numpy()
            xy_to_basket = float(np.linalg.norm(obj_p[:2] - basket_w[:2]))
            z_to_basket = float(abs(obj_p[2] - basket_w[2]))
            basket_contact = 0.0
            try:
                import torch as _t

                cs = env.scene[f"contact_{BASKET_NAME}_{OBJ_NAME}"]
                basket_contact = float(_t.linalg.vector_norm(cs.data.force_matrix_w.reshape(-1, 3)[0]).item())
            except Exception:
                basket_contact = 0.0
            max_basket_contact = max(max_basket_contact, basket_contact)
            xy_to_basket_final = xy_to_basket
            z_to_basket_final = z_to_basket

            obj_dz = float(obj_p[2] - obj0_w[2])
            obj_xy = float(np.linalg.norm(obj_p[:2] - obj0_w[:2]))
            max_obj_xy = max(max_obj_xy, obj_xy)
            peak_z = max(peak_z, float(obj_p[2]))

            if phase in ("hold", "lift", "transit", "over_basket"):
                fL_hold.append(fL)
                fR_hold.append(fR)
            if grasp_flag or (contact and obj_dz > 0.01):
                pick_success = 1
            if obj_dz >= LIFT_SUCCESS_Z:
                lift_success = 1
            if lift_success and phase in ("transit", "over_basket") and (dropped or obj_dz < 0.02):
                lost_in_transit = 1
            if phase == "over_basket":
                obj_dz_over_basket = obj_dz
                retained_over_basket = int((not dropped) and obj_dz >= LIFT_SUCCESS_Z)
            if official_success:
                placed = 1

            rows.append(
                StepRow(
                    trial_id=trial_id,
                    stage=stage,
                    seed_idx=seed_idx,
                    friction=mu,
                    desired_F=desired_F,
                    phase=phase,
                    step=step,
                    d_pred=d_pred,
                    f_sq_meas=f_sq,
                    f_sq_raw=f_raw,
                    applied=applied,
                    gripper_open=gp,
                    ee_x=float(ee[0]),
                    ee_y=float(ee[1]),
                    ee_z=float(ee[2]),
                    obj_x=float(obj_p[0]),
                    obj_y=float(obj_p[1]),
                    obj_z=float(obj_p[2]),
                    obj_dz=obj_dz,
                    marker=mm,
                    contact=contact,
                    dropped=dropped,
                    official_success=official_success,
                    grasp_flag=grasp_flag,
                    xy_to_basket=xy_to_basket,
                    z_to_basket=z_to_basket,
                    basket_contact=basket_contact,
                )
            )
            if bool(term[0].item()) or bool(trunc[0].item()):
                timed_out = int(bool(trunc[0].item()) and not official_success)
                break
        else:
            # Skip transport/place if the object never left the table.
            if phase == "lift" and lift_success == 0 and do_place:
                print(f"early_abort_no_lift {trial_id}", flush=True)
                break
            continue
        break

    hold_rows = [r for r in rows if r.phase in ("hold", "lift", "transit")]
    if fL_hold:
        m = compute_contact_force_metrics_from_lr_forces(np.stack(fL_hold), np.stack(fR_hold))
        mean_sq, max_sq, contact_ratio, mean_app = m.squeeze_mean, m.squeeze_max, m.contact_ratio, m.external_norm_mean
    else:
        mean_sq = max_sq = contact_ratio = mean_app = 0.0
    last = rows[-1] if rows else None
    slip = int(max_obj_xy > SLIP_XY and lift_success == 0)
    summary = {
        "trial_id": trial_id,
        "stage": stage,
        "seed_idx": seed_idx,
        "friction": mu,
        "desired_F": desired_F,
        "measured_mean_squeeze": mean_sq,
        "measured_max_squeeze": max_sq,
        "measured_mean_applied": mean_app,
        "contact_ratio": contact_ratio,
        "tracking_error": abs(mean_sq - desired_F) if mean_sq else desired_F,
        "pick_success": pick_success,
        "lift_success": lift_success,
        "retained": (int(lift_success and not lost_in_transit and retained_over_basket) if do_place else int(lift_success and not dropped)),
        "place_success": placed if do_place else 0,
        "full_task_success": official_success if do_place else 0,
        "official_success": official_success,
        "dropped": dropped,
        "slip": slip,
        "timeout": timed_out,
        "steps": step,
        "obj_peak_z": peak_z,
        "obj_disp_xy": max_obj_xy,
        "final_obj_dz": last.obj_dz if last else 0.0,
        "xy_to_basket_final": xy_to_basket_final,
        "z_to_basket_final": z_to_basket_final,
        "basket_contact_max": max_basket_contact,
        "obj_dz_over_basket": obj_dz_over_basket,
        "lost_in_transit": lost_in_transit,
        "marker_mean": float(np.mean([r.marker for r in hold_rows])) if hold_rows else 0.0,
        "d_pred_final": last.d_pred if last else D_OPEN,
        "friction_applied": fr_info.get("applied_static_mean", mu),
        "nominal_friction": fr_info.get("nominal_static_mean"),
    }
    print(
        f"summary {trial_id} meas={mean_sq:.2f} pick={pick_success} lift={lift_success} "
        f"ret={summary['retained']} place={summary['place_success']} full={official_success} "
        f"drop={dropped} xy={max_obj_xy:.3f} basket_xy={xy_to_basket_final:.3f} "
        f"basket_z={z_to_basket_final:.3f} bc={max_basket_contact:.3f}",
        flush=True,
    )
    return rows, summary


def _write_csv(path: Path, dicts: list[dict]):
    if not dicts:
        path.write_text("NOT_RUN\nreason: empty\n")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
        w.writeheader()
        for d in dicts:
            w.writerow(d)


def main():
    from isaaclab.app import AppLauncher

    enable_cameras = os.environ.get("T1_ENABLE_CAMERAS", "1") not in ("0", "false", "False")
    app_launcher = AppLauncher(headless=True, enable_cameras=enable_cameras, num_envs=1)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(TASK_SUITE, TASK_ID)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = float(os.environ.get("T1_EPISODE_S", "40"))
        if not enable_cameras:
            # Oracle uses pose + finger force, not images. Drop cameras / GelSight
            # render sensors so Isaac can share the GPU with a co-resident job.
            for name in ("eye_in_hand_cam", "agentview_cam", "gsmini_left", "gsmini_right"):
                if hasattr(env_cfg.scene, name):
                    delattr(env_cfg.scene, name)
                    print(f"disabled_sensor {name}", flush=True)
            pol = getattr(env_cfg.observations, "policy", None)
            if pol is not None and hasattr(pol, "gripper_marker_motion"):
                delattr(pol, "gripper_marker_motion")
                print("disabled_obs gripper_marker_motion", flush=True)
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        print("env ready", flush=True)

        obs, _ = env.reset()
        fr_obj = _read_friction(env, OBJ_NAME)
        fr_audit = {"object": fr_obj, "obj_name": OBJ_NAME}
        try:
            robot = env.scene["robot"]
            rm = robot.root_physx_view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
            fr_audit["robot_material_n_shapes"] = int(rm.shape[0])
            fr_audit["robot_static_mean"] = float(rm[:, 0].mean())
            fr_audit["robot_static_min"] = float(rm[:, 0].min())
            fr_audit["robot_static_max"] = float(rm[:, 0].max())
        except Exception as e:
            fr_audit["robot_material_error"] = repr(e)
        (OUT / "FRICTION_RAW.json").write_text(json.dumps(fr_audit, indent=2))
        print("friction_audit", json.dumps(fr_audit), flush=True)

        nominal_mu = float(fr_obj["static_mean"])
        # If user passed default grid, replace middle with actual nominal.
        frictions = list(FRICTION_VALUES)
        if STAGE in ("full", "both") and 0.5 in frictions and abs(nominal_mu - 0.5) > 0.05:
            frictions = sorted({frictions[0], round(nominal_mu, 3), frictions[-1]})

        calib_summ, calib_rows = [], []
        if STAGE in ("calib", "both"):
            for f_star in FORCE_TARGETS:
                for s in range(N_SEEDS_CALIB):
                    tid = f"calib_f{f_star:g}_s{s}"
                    print(f"=== {tid} ===", flush=True)
                    rows, summ = run_episode(
                        env, stage="calib", seed_idx=s, mu=nominal_mu, desired_F=f_star, do_place=False, trial_id=tid
                    )
                    calib_rows.extend(rows)
                    calib_summ.append(summ)
            _write_csv(OUT / "FORCE_SERVO_CALIBRATION.csv", calib_summ)
            if calib_rows:
                with (OUT / "FORCE_SERVO_CALIBRATION_STEPS.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(asdict(calib_rows[0]).keys()))
                    w.writeheader()
                    for r in calib_rows:
                        w.writerow(asdict(r))

            # calibration gate
            by_f = {}
            for s in calib_summ:
                by_f.setdefault(s["desired_F"], []).append(s["measured_mean_squeeze"])
            cal_table = []
            for f, vs in sorted(by_f.items()):
                cal_table.append(
                    {
                        "desired_N": f,
                        "measured_steady_N": float(np.mean(vs)),
                        "std": float(np.std(vs)),
                        "error": abs(float(np.mean(vs)) - f),
                        "n": len(vs),
                    }
                )
            (OUT / "FORCE_SERVO_CALIBRATION_SUMMARY.json").write_text(json.dumps(cal_table, indent=2))
            meas = [c["measured_steady_N"] for c in cal_table]
            ordered = all(meas[i + 1] + 0.3 >= meas[i] for i in range(len(meas) - 1)) if len(meas) > 1 else False
            low_ok = min(meas) <= LOW_FORCE_GATE_N if meas else False
            separated = (max(meas) - min(meas) >= 3.0) if meas else False
            gate = {
                "ordered": ordered,
                "low_force_regime_achieved": low_ok,
                "separated": separated,
                "min_measured": min(meas) if meas else None,
                "max_measured": max(meas) if meas else None,
                "pass": bool(ordered and low_ok and separated),
            }
            (OUT / "FORCE_SERVO_GATE.json").write_text(json.dumps(gate, indent=2))
            print("CALIB_GATE", json.dumps(gate), flush=True)
            if STAGE == "both" and not gate["pass"]:
                print("calibration gate failed; skipping full matrix", flush=True)
                STAGE_EFFECTIVE = "calib"
            else:
                STAGE_EFFECTIVE = STAGE
        else:
            STAGE_EFFECTIVE = STAGE
            gate = {"pass": True}

        full_summ = []
        full_csv = OUT / "FORCE_X_FRICTION_FULL_TASK.csv"
        done_ids = set()
        if full_csv.exists() and full_csv.read_text()[:7] != "NOT_RUN":
            try:
                with full_csv.open() as f:
                    for row in csv.DictReader(f):
                        full_summ.append(row)
                        done_ids.add(row["trial_id"])
                print(f"resume {len(done_ids)} trials from CSV", flush=True)
            except Exception:
                full_summ, done_ids = [], set()
        if STAGE_EFFECTIVE in ("full", "both") and (STAGE == "full" or gate.get("pass", False)):
            for mu in frictions:
                for f_star in FORCE_TARGETS:
                    for s in range(N_SEEDS_FULL):
                        tid = f"full_mu{mu:g}_f{f_star:g}_s{s}"
                        if tid in done_ids:
                            print(f"skip {tid}", flush=True)
                            continue
                        print(f"=== {tid} ===", flush=True)
                        rows, summ = run_episode(
                            env, stage="full", seed_idx=s, mu=mu, desired_F=f_star, do_place=True, trial_id=tid
                        )
                        full_summ.append(summ)
                        _write_csv(full_csv, full_summ)
        else:
            if not full_csv.exists():
                full_csv.write_text(
                    "NOT_RUN\nreason: calibration gate failed or stage=calib only\n"
                )

        meta = {
            "env_id": ENV_ID,
            "task_suite": TASK_SUITE,
            "task_id": TASK_ID,
            "instruction": "Pick up the cream cheese and place it in the basket.",
            "force_adverbs_used": False,
            "stage": STAGE,
            "force_targets": FORCE_TARGETS,
            "frictions": frictions,
            "nominal_friction": nominal_mu,
            "n_calib": len(calib_summ),
            "n_full": len(full_summ),
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
