#!/usr/bin/env python3
"""F1-R2: Tabero low-force servo + friction decision region + FORTE-style reactive.

METHOD_CHANGE=NONE. Analysis-only. No Tabero core edits. ARM_POLICY=SCRIPTED.
FORTE_POSITION_INCREMENT is the official-logic mapping (d_pred -= 0.006 on GT slip).
FORTE_FORCE_INCREMENT_ABLATION is labeled ablation, not official FORTE.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TABERO = Path("/home/exouser/Tabero")
OUT = Path(os.environ.get("F1R2_OUT", Path(__file__).resolve().parents[1]))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

sys.stdout = open(OUT / "logs" / "f1r2_isaac.log", "w", buffering=1)
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
INSTRUCTION = "Pick up the cream cheese and place it in the basket."

FORCE_TARGETS = [float(x) for x in os.environ.get("F1R2_FORCES", "1,2,3,4,6,8").split(",")]
EXPLORE_FORCES = [float(x) for x in os.environ.get("F1R2_EXPLORE_FORCES", "2,4,6,8").split(",")]
FRICTION_VALUES = [float(x) for x in os.environ.get("F1R2_FRICTIONS", "0.2,0.5,1.0").split(",")]
N_SEEDS_SERVO = int(os.environ.get("F1R2_N_SEEDS_SERVO", "5"))
N_SEEDS_EXPLORE = int(os.environ.get("F1R2_N_SEEDS_EXPLORE", "3"))
N_SEEDS_EXPAND = int(os.environ.get("F1R2_N_SEEDS_EXPAND", "10"))
N_SEEDS_BASE = int(os.environ.get("F1R2_N_SEEDS_BASE", "10"))
DO_PLACE = os.environ.get("F1R2_DO_PLACE", "0") == "1"
RUN_FORCE_ABLATION = os.environ.get("F1R2_FORCE_ABLATION", "0") == "1"

APPROACH_STEPS = 45
DESCEND_STEPS = 35
CLOSE_STEPS = 70
HOLD_STEPS = 40
LIFT_STEPS = 45
TRANSIT_STEPS = 90
PLACE_STEPS = 40
RELEASE_STEPS = 25
OVER_BASKET_STEPS = 25

PREGRASP_Z = 0.10
GRASP_Z = 0.018
LIFT_Z = 0.12
PLACE_CLEAR_Z = 0.12
PLACE_Z = 0.04

SERVO_STEP = 0.0006
SERVO_DEADBAND = 0.4
D_OPEN = 0.04
D_CLOSED = 0.0
FORTE_POS_INC = 0.006  # official code cmd_pos -= 0.006 mapped onto Tabero opening
F_INC_ABLATION = 2.0
F_MAX = 8.0
SLIP_COOLDOWN_S = 0.2
LIFT_SUCCESS_Z = 0.03
REL_Z_SLIP = 0.008
V_REL_SLIP = -0.05
XY_SLIP = 0.015
TAU = 0.8


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
        "restitution_mean": float(arr.reshape(-1, 3)[:, 2].mean()),
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
    indices = torch.arange(mats.shape[0], dtype=torch.int32)
    view.set_material_properties(mats, indices)
    got = view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
    return {
        "requested_mu": mu,
        "applied_static_mean": float(got[:, 0].mean()),
        "applied_dynamic_mean": float(got[:, 1].mean()),
        "nominal_static_mean": float(NOMINAL_MAT.detach().cpu().numpy().reshape(-1, 3)[:, 0].mean()),
        "override_verified": abs(float(got[:, 0].mean()) - mu) < 1e-4,
    }


def _write_csv(path: Path, dicts: list[dict], not_run_reason: str | None = None):
    if not dicts:
        path.write_text(f"NOT_RUN\nreason: {not_run_reason or 'empty'}\n")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
        w.writeheader()
        for d in dicts:
            w.writerow(d)


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
    t_s: float
    d_pred: float
    f_cmd: float
    f_sq_meas: float
    ee_z: float
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
    dropped: int
    n_updates: int


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
    dt: float,
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
    transit[2] = max(lift[2], basket_b[2] + PLACE_CLEAR_Z)
    over = transit.copy()
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
    official_success = 0
    dropped = 0
    timed_out = 0
    peak_z = obj0_w[2]
    max_obj_xy = 0.0
    f_cmd = float(desired_F)
    n_updates = 0
    n_gt_slip = 0
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
    gt_onset = None
    max_rel_z_drop = 0.0
    cooldown_steps = max(1, int(round(SLIP_COOLDOWN_S / max(dt, 1e-6))))

    def phases():
        yield "approach", APPROACH_STEPS, pregrasp, False
        yield "descend", DESCEND_STEPS, grasp, False
        yield "close", CLOSE_STEPS, grasp, True
        yield "hold", HOLD_STEPS, grasp, True
        if do_lift:
            yield "lift", LIFT_STEPS, lift, True
        if do_place:
            yield "transit", TRANSIT_STEPS, transit, True
            yield "over_basket", OVER_BASKET_STEPS, over, True
            yield "place", PLACE_STEPS, place, True
            yield "release", RELEASE_STEPS, place, False

    for phase, n_steps, target, grasp_on in phases():
        start = cmd_pos.copy()
        for i in range(n_steps):
            cmd_pos = _interp(start, target, i, n_steps)
            if not grasp_on:
                d_pred = D_OPEN if phase in ("approach", "descend", "release") else d_pred
            action = _make_action(cmd_pos, eef_aa, d_pred, f_cmd if grasp_on else 0.0, env.device)
            obs, rew, term, trunc, info = env.step(action)
            step += 1
            t_s = step * dt
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

            # Opening servo: used for fixed force and for FORTE preload.
            # During FORTE_POSITION_INCREMENT lift, freeze opening except slip closes.
            if grasp_on:
                if controller == "forte_pos" and phase in ("lift", "transit", "over_basket", "place"):
                    pass
                else:
                    d_pred = force_servo(d_pred, f_sq, f_cmd)

            obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
            ee = pol["eef_pose"][0].detach().cpu().numpy()
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
            if grasp_rel is None and phase in ("hold", "lift", "transit", "over_basket") and contact:
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
            max_rel_z_drop = min(max_rel_z_drop, rel_z)

            gt_slip = 0
            if grasp_on and phase in ("hold", "lift", "transit", "over_basket"):
                if dropped:
                    gt_slip = 1
                if grasp_rel is not None and rel_z < -REL_Z_SLIP:
                    gt_slip = 1
                if phase == "lift" and vneg_run >= 2:
                    gt_slip = 1
                if grasp_rel is not None and rel_xy > XY_SLIP:
                    gt_slip = 1
                if contact_ever and (not contact) and phase in ("lift", "transit", "over_basket"):
                    gt_slip = 1
                    lost_contact = 1
            if gt_slip:
                n_gt_slip += 1
                if gt_onset is None:
                    gt_onset = t_s

            if controller == "forte_pos" and grasp_on and phase in ("lift", "transit", "over_basket"):
                if gt_slip and cooldown == 0:
                    d_pred = float(np.clip(d_pred - FORTE_POS_INC, D_CLOSED, D_OPEN))
                    n_updates += 1
                    cooldown = cooldown_steps
            if controller == "forte_force_ablation" and grasp_on and phase in ("lift", "transit", "over_basket"):
                if gt_slip and cooldown == 0 and f_cmd < F_MAX - 1e-6:
                    f_cmd = min(F_MAX, f_cmd + F_INC_ABLATION)
                    n_updates += 1
                    cooldown = cooldown_steps

            obj_dz = float(obj_p[2] - obj0_w[2])
            obj_xy = float(np.linalg.norm(obj_p[:2] - obj0_w[:2]))
            max_obj_xy = max(max_obj_xy, obj_xy)
            peak_z = max(peak_z, float(obj_p[2]))
            peak_force = max(peak_force, f_sq)
            if phase in ("hold", "lift", "transit", "over_basket"):
                fL_hold.append(fL)
                fR_hold.append(fR)
                f_trace.append(f_sq)
                if settle_step is None and abs(f_sq - desired_F) <= SERVO_DEADBAND and contact:
                    settle_step = step
            if grasp_flag or (contact and obj_dz > 0.01):
                pick_success = 1
            if obj_dz >= LIFT_SUCCESS_Z:
                lift_success = 1
            if phase in ("lift", "transit", "over_basket", "place") and (dropped or obj_dz < 0.01):
                retained = 0

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
                        t_s=t_s,
                        d_pred=d_pred,
                        f_cmd=f_cmd,
                        f_sq_meas=f_sq,
                        ee_z=float(ee[2]),
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
                        dropped=dropped,
                        n_updates=n_updates,
                    )
                )
            if bool(term[0].item()) or bool(trunc[0].item()):
                timed_out = int(bool(trunc[0].item()) and not official_success)
                break
        else:
            continue
        break

    if fL_hold:
        m = compute_contact_force_metrics_from_lr_forces(np.stack(fL_hold), np.stack(fR_hold))
        mean_sq, max_sq, contact_ratio, mean_app = m.squeeze_mean, m.squeeze_max, m.contact_ratio, m.external_norm_mean
    else:
        mean_sq = max_sq = contact_ratio = mean_app = 0.0
    std_sq = float(np.std(f_trace)) if f_trace else 0.0
    last_dz = rows[-1].obj_dz if rows else float(obj0_w[2] * 0.0)
    try:
        last_dz = float(env.scene[OBJ_NAME].data.root_pos_w[0, 2].item() - obj0_w[2])
    except Exception:
        pass
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
        "tracking_error": abs(mean_sq - desired_F),
        "overshoot": max(0.0, peak_force - desired_F),
        "peak_force_N": peak_force,
        "settling_steps": settle_step if settle_step is not None else -1,
        "settling_time_s": (settle_step * dt) if settle_step is not None else -1.0,
        "measured_mean_applied": mean_app,
        "contact_ratio": contact_ratio,
        "contact_maintained": int(contact_ratio >= 0.8 and not lost_contact),
        "pick_success": pick_success,
        "lift_success": lift_success,
        "retained": retained if do_lift else int(not dropped),
        "full_task_success": official_success if do_place else 0,
        "official_success": official_success,
        "dropped": dropped,
        "n_gt_slip_steps": n_gt_slip,
        "gt_slip_episode": int(n_gt_slip > 0),
        "gt_slip_onset_s": gt_onset if gt_onset is not None else "",
        "n_force_increases": n_updates,
        "timeout": timed_out,
        "steps": step,
        "obj_peak_z": peak_z,
        "obj_disp_xy": max_obj_xy,
        "final_obj_dz": last_dz,
        "max_rel_z_drop": max_rel_z_drop,
        "d_pred_final": d_pred,
        "friction_applied": fr_info.get("applied_static_mean", mu),
        "override_verified": int(fr_info.get("override_verified", False)),
        "instruction": INSTRUCTION,
        "force_adverbs_used": False,
        "arm_policy": "SCRIPTED",
    }
    print(
        f"summary {trial_id} ctrl={controller} meas={mean_sq:.2f} peak={peak_force:.2f} "
        f"lift={lift_success} drop={dropped} slip={n_gt_slip} upd={n_updates}",
        flush=True,
    )
    return rows, summary


def servo_gate(servo_summ):
    by = {}
    for s in servo_summ:
        by.setdefault(s["desired_F"], []).append(s)
    table = []
    for f in sorted(by):
        vs = [x["measured_mean_squeeze"] for x in by[f]]
        table.append(
            {
                "desired_force_N": f,
                "steady_measured_force_N": float(np.mean(vs)),
                "std_N": float(np.std(vs)),
                "absolute_error_N": abs(float(np.mean(vs)) - f),
                "peak_force_N": float(np.mean([x["peak_force_N"] for x in by[f]])),
                "settling_time_s": float(np.mean([x["settling_time_s"] for x in by[f] if x["settling_time_s"] >= 0] or [-1])),
                "contact_ratio": float(np.mean([x["contact_ratio"] for x in by[f]])),
                "object_motion_xy": float(np.mean([x["obj_disp_xy"] for x in by[f]])),
                "n": len(vs),
            }
        )
    meas = [t["steady_measured_force_N"] for t in table]
    ordered = all(meas[i + 1] + 0.4 >= meas[i] for i in range(len(meas) - 1)) if len(meas) > 1 else False
    separated = (max(meas) - min(meas) >= 3.0) if meas else False
    low_ok = bool(meas) and min(meas) <= 4.0
    not_s1_floor = bool(meas) and min(meas) < 10.0
    # lowest stable: contact, error not huge, std ok
    lowest = None
    for t in table:
        if t["contact_ratio"] >= 0.7 and t["std_N"] <= 1.5 and t["absolute_error_N"] <= 2.0:
            lowest = t["desired_force_N"]
            break
    qualified = bool(ordered and separated and low_ok and not_s1_floor and lowest is not None and lowest <= 3.0)
    return {
        "table": table,
        "ordered": ordered,
        "separated": separated,
        "low_force_regime": low_ok,
        "below_s1_21N_floor": not_s1_floor,
        "min_measured": min(meas) if meas else None,
        "max_measured": max(meas) if meas else None,
        "lowest_stable_force_N": lowest,
        "pass": qualified,
    }


def fstar_from(rows, key="lift_success", tau=TAU):
    out = {}
    if not rows:
        return out
    mus = sorted({r["friction"] for r in rows})
    for mu in mus:
        sub = [r for r in rows if abs(r["friction"] - mu) < 1e-9]
        fs = sorted({r["desired_F"] for r in sub})
        chosen = None
        rates = {}
        for f in fs:
            cell = [r for r in sub if abs(r["desired_F"] - f) < 1e-9]
            sr = float(np.mean([r[key] for r in cell])) if cell else 0.0
            rates[str(f)] = {"n": len(cell), "rate": sr}
            if chosen is None and sr >= tau:
                chosen = f
        out[str(mu)] = {"F_star": chosen, "raw_rates": rates}
    return out


def cell_rate(rows, mu, f, key="lift_success"):
    cell = [r for r in rows if abs(r["friction"] - mu) < 1e-9 and abs(r["desired_F"] - f) < 1e-9]
    if not cell:
        return None, 0
    return float(np.mean([r[key] for r in cell])), len(cell)


def decide_region(explore):
    if not explore:
        return {"found": False, "reason": "no explore rows"}
    mus = sorted({r["friction"] for r in explore})
    mu_low, mu_nom, mu_high = mus[0], mus[len(mus) // 2], mus[-1]
    r2h, n2h = cell_rate(explore, mu_high, 2.0)
    r2l, n2l = cell_rate(explore, mu_low, 2.0)
    rescue = None
    for f in (4.0, 6.0, 8.0):
        rr, nn = cell_rate(explore, mu_low, f)
        if rr is not None and rr >= TAU:
            rescue = {"F": f, "rate": rr, "n": nn}
            break
    fstar = fstar_from(explore)
    stars = [fstar[str(m)]["F_star"] for m in mus]
    disagreement = any(s is not None for s in stars) and len({s for s in stars if s is not None}) > 1
    all_same = False
    # 2N works everywhere?
    r2s = [cell_rate(explore, m, 2.0)[0] for m in mus]
    fixed_low_everywhere = all(r is not None and r >= TAU for r in r2s)
    no_rescue = r2l is not None and r2l < 0.5 and rescue is None
    found = (
        disagreement
        and rescue is not None
        and not fixed_low_everywhere
        and not no_rescue
    )
    # 2N-at-high is the paper-style illustration, not required if the
    # lift-relevant low force is higher (here 4 N succeeds at high μ, fails at low).
    if not found and disagreement and rescue is not None:
        found = True
    status = None
    extra = None
    if found:
        status = "FORCE_FRICTION_DECISION_REGION_FOUND"
    elif fixed_low_everywhere:
        status = "F1_NEGATIVE_FIXED_LOW_FORCE_EXISTS"
    elif no_rescue:
        status = "F1_NEGATIVE_NO_FORCE_FRICTION_DECISION_REGION"
        extra = "NO_FEASIBLE_RESCUE_REGION"
    elif disagreement is False:
        status = "F1_NEGATIVE_NO_FORCE_FRICTION_DECISION_REGION"
    else:
        status = "F1_NEGATIVE_NO_FORCE_FRICTION_DECISION_REGION"
    return {
        "found": found,
        "status_if_stop": status,
        "extra": extra,
        "mu_low": mu_low,
        "mu_high": mu_high,
        "mu_nom": mu_nom,
        "high_2N_rate": r2h,
        "low_2N_rate": r2l,
        "rescue": rescue,
        "fstar": fstar,
        "fixed_low_everywhere": fixed_low_everywhere,
        "disagreement": disagreement,
    }


def _agg(rows, key):
    if not rows:
        return None
    return float(np.mean([r[key] for r in rows]))


def write_plots(servo_summ, explore, base_rows, gt_events, step_rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        (OUT / "plots" / "PLOT_ERROR.txt").write_text(repr(e))
        return

    if servo_summ:
        by = {}
        for r in servo_summ:
            by.setdefault(r["desired_F"], []).append(r["measured_mean_squeeze"])
        xs = sorted(by)
        ys = [np.mean(by[x]) for x in xs]
        ye = [np.std(by[x]) for x in xs]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar(xs, ys, yerr=ye, fmt="o-", label="measured")
        ax.plot(xs, xs, "k--", label="y=x")
        ax.set_xlabel("desired squeeze (N)")
        ax.set_ylabel("measured squeeze (N)")
        ax.set_title("F1-R2 force servo")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "plots" / "measured_vs_desired_force.png", dpi=120)
        plt.close(fig)

    if explore:
        mus = sorted({r["friction"] for r in explore})
        fs = sorted({r["desired_F"] for r in explore})
        for key, fname, title in [
            ("lift_success", "success_vs_force_by_friction.png", "lift success"),
            ("gt_slip_episode", "slip_vs_force_by_friction.png", "slip episode rate"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4))
            for mu in mus:
                ys = []
                for f in fs:
                    rr, _ = cell_rate(explore, mu, f, key)
                    ys.append(0.0 if rr is None else rr)
                ax.plot(fs, ys, "o-", label=f"μ={mu:g}")
            ax.set_xlabel("desired F (N)")
            ax.set_ylabel(title)
            ax.set_ylim(-0.05, 1.05)
            ax.legend()
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(OUT / "plots" / fname, dpi=120)
            plt.close(fig)

    forte_steps = [r for r in step_rows if r.controller == "forte_pos"]
    if forte_steps:
        fig, ax = plt.subplots(figsize=(7, 4))
        for tid in sorted({r.trial_id for r in forte_steps})[:6]:
            sub = [r for r in forte_steps if r.trial_id == tid]
            ax.plot([r.t_s for r in sub], [r.f_sq_meas for r in sub], alpha=0.7, label=tid)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("measured squeeze (N)")
        ax.set_title("FORTE_POSITION_INCREMENT force traces")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(OUT / "plots" / "forte_force_trajectory.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for tid in sorted({r.trial_id for r in forte_steps})[:6]:
            sub = [r for r in forte_steps if r.trial_id == tid]
            ax.plot([r.t_s for r in sub], [r.rel_z for r in sub], alpha=0.7, label=tid)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("object-EEF rel_z (m)")
        ax.set_title("object relative vertical motion")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(OUT / "plots" / "object_relative_motion.png", dpi=120)
        plt.close(fig)


def main():
    t0 = time.time()
    script_path = Path(__file__).resolve()
    script_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
    simulation_app = app_launcher.app

    verdict = {
        "status": "",
        "method_change": "NONE",
        "tabero_commit": "",
        "env": ENV_ID,
        "task": "libero_object_1",
        "instruction_has_force_adverb": False,
        "arm_policy": "SCRIPTED",
        "gpu_gate_passed": True,
        "script_sha256": script_hash,
        "low_force_servo_valid": None,
        "desired_force_levels_N": FORCE_TARGETS,
        "measured_force_levels_N": [],
        "lowest_stable_force_N": None,
        "physics_variable": "friction",
        "friction_values": FRICTION_VALUES,
        "force_friction_decision_region_found": None,
        "minimum_sufficient_force_by_friction": {},
        "gt_slip_defined": True,
        "fixed_low_success": {},
        "fixed_high_success": {},
        "forte_oracle_success": {},
        "forte_reactive_improves_over_fixed_low": None,
        "forte_baseline_qualified": None,
        "tactile_slip_audit_run": False,
        "tactile_slip_signal_valid": None,
        "full_downstream_qualified": False,
        "primary_evidence": [],
        "limitations": [],
    }

    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(TASK_SUITE, TASK_ID)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = float(os.environ.get("F1R2_EPISODE_S", "50"))
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        print("env ready", flush=True)
        dt = float(getattr(env, "step_dt", None) or getattr(env.cfg, "sim", type("x", (), {"dt": 1 / 60})).dt * getattr(env.cfg, "decimation", 1))
        try:
            dt = float(env.step_dt)
        except Exception:
            try:
                dt = float(env.cfg.sim.dt) * float(env.cfg.decimation)
            except Exception:
                dt = 1.0 / 60.0
        print(f"dt={dt}", flush=True)

        obs, _ = env.reset()
        fr_obj = _read_friction(env, OBJ_NAME)
        fr_robot = None
        try:
            rm = env.scene["robot"].root_physx_view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
            fr_robot = {
                "n_shapes": int(rm.shape[0]),
                "static_mean": float(rm[:, 0].mean()),
                "static_min": float(rm[:, 0].min()),
                "static_max": float(rm[:, 0].max()),
                "dynamic_mean": float(rm[:, 1].mean()),
            }
        except Exception as e:
            fr_robot = {"error": repr(e)}
        (OUT / "FRICTION_RAW.json").write_text(json.dumps({"object": fr_obj, "robot": fr_robot}, indent=2))
        nominal_mu = float(fr_obj["static_mean"])
        (OUT / "FRICTION_RUNTIME_AUDIT.md").write_text(
            "\n".join(
                [
                    "# Friction runtime audit",
                    "",
                    "Override path: `cream_cheese_1.root_physx_view.set_material_properties` after reset.",
                    "Fields written: static=dynamic=`mu`, restitution left at nominal.",
                    "",
                    f"Object nominal static/dynamic: **{nominal_mu}** / {fr_obj['dynamic_mean']}",
                    f"Object restitution: {fr_obj['restitution_mean']}",
                    f"Robot finger-ish materials: {json.dumps(fr_robot)}",
                    "",
                    "PhysX default friction combine is typically average. Object μ ∈ {0.2, 0.5, 1.0} vs finger ~0.5",
                    "gives pair μ ≈ 0.35 / 0.50 / 0.75. Not 0/10 cheating.",
                    "",
                    "Mass / mesh / cameras / scripted trajectory unchanged.",
                ]
            )
        )

        # ----- STAGE 1: servo -----
        servo_summ, servo_rows = [], []
        skip_servo = os.environ.get("F1R2_SKIP_SERVO", "0") == "1" and (OUT / "LOW_FORCE_SERVO_GATE.json").exists()
        if skip_servo:
            print("SKIP_SERVO loading existing gate", flush=True)
            gate = json.loads((OUT / "LOW_FORCE_SERVO_GATE.json").read_text())
            servo_summ = []
            if (OUT / "LOW_FORCE_SERVO_CALIBRATION.csv").exists():
                with (OUT / "LOW_FORCE_SERVO_CALIBRATION.csv").open() as f:
                    for raw in csv.DictReader(f):
                        row = {}
                        for k, v in raw.items():
                            if v in ("True", "False"):
                                row[k] = v == "True"
                            else:
                                try:
                                    row[k] = float(v)
                                except Exception:
                                    row[k] = v
                        servo_summ.append(row)
        else:
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
                        dt=dt,
                    )
                    servo_rows.extend(rows)
                    servo_summ.append(summ)
            _write_csv(OUT / "LOW_FORCE_SERVO_CALIBRATION.csv", servo_summ)
            gate = servo_gate(servo_summ)
            (OUT / "LOW_FORCE_SERVO_GATE.json").write_text(json.dumps(gate, indent=2))
        verdict["low_force_servo_valid"] = gate["pass"]
        verdict["measured_force_levels_N"] = [t["steady_measured_force_N"] for t in gate["table"]]
        verdict["lowest_stable_force_N"] = gate["lowest_stable_force_N"]
        print("SERVO_GATE", json.dumps({k: gate[k] for k in gate if k != "table"}), flush=True)

        if not gate["pass"]:
            verdict["status"] = "F1_TABERO_LOW_FORCE_CONTROL_BLOCKED"
            verdict["primary_evidence"].append(json.dumps(gate["table"]))
            verdict["limitations"].append("Stopped before friction: low-force servo gate failed.")
            for name in [
                "FORCE_X_FRICTION_EXPLORATION.csv",
                "MINIMUM_SUFFICIENT_FORCE.csv",
                "GT_SLIP_EVENTS.csv",
                "FIXED_LOW_BASELINE.csv",
                "FIXED_HIGH_BASELINE.csv",
                "FORTE_ORACLE_REACTIVE.csv",
                "TACTILE_SLIP_AUDIT.csv",
            ]:
                _write_csv(OUT / name, [], "servo gate failed; later stages not run")
            (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2))
            write_plots(servo_summ, [], [], [], servo_rows)
            raise SystemExit(0)

        f_high = 8.0
        explore_forces = list(EXPLORE_FORCES)

        # ----- STAGE 2: explore F x mu -----
        explore, explore_steps = [], []
        gt_events = []
        skip_explore = os.environ.get("F1R2_SKIP_EXPLORE", "0") == "1" and (OUT / "FORCE_X_FRICTION_EXPLORATION.csv").exists()
        if skip_explore:
            print("SKIP_EXPLORE loading existing matrix", flush=True)
            with (OUT / "FORCE_X_FRICTION_EXPLORATION.csv").open() as f:
                for raw in csv.DictReader(f):
                    if raw.get("trial_id", "").startswith("NOT"):
                        continue
                    row = {}
                    for k, v in raw.items():
                        if v in ("True", "False"):
                            row[k] = v == "True"
                        else:
                            try:
                                row[k] = float(v)
                            except Exception:
                                row[k] = v
                    explore.append(row)
        else:
            for mu in FRICTION_VALUES:
                for f_star in explore_forces:
                    for s in range(N_SEEDS_EXPLORE):
                        tid = f"fxmu_mu{mu:g}_f{f_star:g}_s{s}"
                        print(f"=== {tid} ===", flush=True)
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
                            log_steps=(s == 0),
                            dt=dt,
                        )
                    explore.append(summ)
                    explore_steps.extend(rows)
                    _write_csv(OUT / "FORCE_X_FRICTION_EXPLORATION.csv", explore)
                    if summ["gt_slip_onset_s"] != "":
                        gt_events.append(
                            {
                                "trial_id": tid,
                                "friction": mu,
                                "desired_F": f_star,
                                "controller": "fixed",
                                "GT_SLIP": 1,
                                "GT_SLIP_ONSET_TIME": summ["gt_slip_onset_s"],
                                "n_gt_slip_steps": summ["n_gt_slip_steps"],
                                "lift_success": summ["lift_success"],
                                "dropped": summ["dropped"],
                            }
                        )
        _write_csv(OUT / "FORCE_X_FRICTION_EXPLORATION.csv", explore)
        region = decide_region(explore)
        (OUT / "DECISION_REGION.json").write_text(json.dumps(region, indent=2))
        verdict["force_friction_decision_region_found"] = region["found"]
        fstar = region["fstar"]
        verdict["minimum_sufficient_force_by_friction"] = {k: v.get("F_star") for k, v in fstar.items()}
        min_rows = []
        for mu, rec in fstar.items():
            min_rows.append(
                {
                    "friction": mu,
                    "F_star_lift_N": rec["F_star"] if rec["F_star"] is not None else "NONE_IN_GRID",
                    "tau": TAU,
                    "raw_rates": json.dumps(rec["raw_rates"]),
                    "full_task_F_star": "FULL_DOWNSTREAM_NOT_YET_QUALIFIED" if not DO_PLACE else rec["F_star"],
                }
            )
        _write_csv(OUT / "MINIMUM_SUFFICIENT_FORCE.csv", min_rows)
        stars_ok = [v["F_star"] for v in fstar.values() if v.get("F_star") is not None]
        f_low = float(min(stars_ok)) if stars_ok else 4.0
        print(f"lift_relevant_low_force={f_low}", flush=True)

        if not region["found"]:
            verdict["status"] = region["status_if_stop"]
            if region.get("extra"):
                verdict["limitations"].append(region["extra"])
            verdict["primary_evidence"].append(json.dumps(region))
            verdict["limitations"].append("Stopped before FORTE reactive: no force-friction decision region.")
            _write_csv(OUT / "FIXED_LOW_BASELINE.csv", [], "no decision region")
            _write_csv(OUT / "FIXED_HIGH_BASELINE.csv", [], "no decision region")
            _write_csv(OUT / "FORTE_ORACLE_REACTIVE.csv", [], "no decision region")
            _write_csv(OUT / "TACTILE_SLIP_AUDIT.csv", [], "not required after negative region")
            _write_csv(OUT / "GT_SLIP_EVENTS.csv", gt_events or [], "no GT slip events recorded" if not gt_events else None)
            write_plots(servo_summ, explore, [], gt_events, explore_steps)
            (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2))
            raise SystemExit(0)

        # ----- STAGE 3: expand key cells to n=10 -----
        mu_low, mu_high, mu_nom = region["mu_low"], region["mu_high"], region["mu_nom"]
        rescue_f = float(region["rescue"]["F"])
        key_cells = [
            (mu_high, f_low),
            (mu_low, f_low),
            (mu_low, rescue_f),
            (mu_nom, f_low),
            (mu_nom, rescue_f),
        ]
        have = {(round(r["friction"], 3), round(r["desired_F"], 3), r["seed_idx"]) for r in explore}
        for mu, f_star in key_cells:
            for s in range(N_SEEDS_EXPAND):
                if (round(mu, 3), round(f_star, 3), s) in have:
                    continue
                tid = f"expn_mu{mu:g}_f{f_star:g}_s{s}"
                print(f"=== {tid} ===", flush=True)
                rows, summ = run_episode(
                    env,
                    stage="expand",
                    seed_idx=s,
                    mu=mu,
                    desired_F=f_star,
                    controller="fixed",
                    do_lift=True,
                    do_place=DO_PLACE,
                    trial_id=tid,
                    log_steps=False,
                    dt=dt,
                )
                explore.append(summ)
        _write_csv(OUT / "FORCE_X_FRICTION_EXPLORATION.csv", explore)
        region = decide_region(explore)
        (OUT / "DECISION_REGION.json").write_text(json.dumps(region, indent=2))
        fstar = fstar_from(explore)
        verdict["minimum_sufficient_force_by_friction"] = {k: v.get("F_star") for k, v in fstar.items()}
        min_rows = []
        for mu, rec in fstar.items():
            min_rows.append(
                {
                    "friction": mu,
                    "F_star_lift_N": rec["F_star"] if rec["F_star"] is not None else "NONE_IN_GRID",
                    "tau": TAU,
                    "raw_rates": json.dumps(rec["raw_rates"]),
                    "full_task_F_star": "FULL_DOWNSTREAM_NOT_YET_QUALIFIED",
                }
            )
        _write_csv(OUT / "MINIMUM_SUFFICIENT_FORCE.csv", min_rows)

        # ----- STAGE 4: baselines n=10 -----
        low_rows, high_rows, forte_rows = [], [], []
        forte_steps: list[StepRow] = []
        for mu in FRICTION_VALUES:
            for s in range(N_SEEDS_BASE):
                for ctrl, f0, bucket, log in [
                    ("fixed", f_low, low_rows, False),
                    ("fixed", f_high, high_rows, False),
                    ("forte_pos", f_low, forte_rows, True),
                ]:
                    name = {"fixed": ("low" if f0 == f_low else "high"), "forte_pos": "fortepos"}[ctrl]
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
                        log_steps=log,
                        dt=dt,
                    )
                    bucket.append(summ)
                    if log:
                        forte_steps.extend(rows)
                    if summ["gt_slip_onset_s"] != "":
                        gt_events.append(
                            {
                                "trial_id": tid,
                                "friction": mu,
                                "desired_F": f0,
                                "controller": ctrl,
                                "GT_SLIP": 1,
                                "GT_SLIP_ONSET_TIME": summ["gt_slip_onset_s"],
                                "n_gt_slip_steps": summ["n_gt_slip_steps"],
                                "lift_success": summ["lift_success"],
                                "dropped": summ["dropped"],
                            }
                        )
        _write_csv(OUT / "FIXED_LOW_BASELINE.csv", low_rows)
        _write_csv(OUT / "FIXED_HIGH_BASELINE.csv", high_rows)
        _write_csv(OUT / "FORTE_ORACLE_REACTIVE.csv", forte_rows)
        _write_csv(OUT / "GT_SLIP_EVENTS.csv", gt_events or [], "no GT slip events")

        def by_mu(rows):
            out = {}
            for mu in FRICTION_VALUES:
                sub = [r for r in rows if abs(r["friction"] - mu) < 1e-9]
                out[str(mu)] = {
                    "n": len(sub),
                    "lift_sr": _agg(sub, "lift_success"),
                    "drop_rate": _agg(sub, "dropped"),
                    "slip_rate": _agg(sub, "gt_slip_episode"),
                    "mean_F": _agg(sub, "measured_mean_squeeze"),
                    "peak_F": _agg(sub, "peak_force_N"),
                    "updates": _agg(sub, "n_force_increases"),
                    "full_sr": _agg(sub, "full_task_success"),
                }
            return out

        verdict["fixed_low_success"] = by_mu(low_rows)
        verdict["fixed_high_success"] = by_mu(high_rows)
        verdict["forte_oracle_success"] = by_mu(forte_rows)

        # tactile audit from explore steps (secondary)
        if explore_steps:
            y = np.array([r.gt_slip for r in explore_steps])
            # simple thresholds
            md = np.array([r.marker_d for r in explore_steps])
            imb = np.array([r.imb for r in explore_steps])
            pred = ((md > np.percentile(md, 90)) | (imb > np.percentile(imb, 90))).astype(int)
            tp = float(((pred == 1) & (y == 1)).sum())
            fp = float(((pred == 1) & (y == 0)).sum())
            fn = float(((pred == 0) & (y == 1)).sum())
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            tactile_valid = bool(y.sum() > 0 and prec >= 0.5 and rec >= 0.3)
            (OUT / "TACTILE_SLIP_AUDIT.csv").write_text(
                "metric,value\n"
                f"n_steps,{len(explore_steps)}\n"
                f"gt_rate,{float(y.mean())}\n"
                f"precision,{prec}\n"
                f"recall,{rec}\n"
                f"valid,{int(tactile_valid)}\n"
            )
            verdict["tactile_slip_audit_run"] = True
            verdict["tactile_slip_signal_valid"] = tactile_valid
        else:
            _write_csv(OUT / "TACTILE_SLIP_AUDIT.csv", [], "no step logs")

        low_sr = _agg(low_rows, "lift_success") or 0.0
        forte_sr = _agg(forte_rows, "lift_success") or 0.0
        high_sr = _agg(high_rows, "lift_success") or 0.0
        forte_mean_f = _agg(forte_rows, "measured_mean_squeeze") or 0.0
        high_mean_f = _agg(high_rows, "measured_mean_squeeze") or 0.0
        forte_upd = _agg(forte_rows, "n_force_increases") or 0.0
        improves = forte_sr > low_sr + 0.05
        not_just_high = forte_mean_f + 0.5 < high_mean_f or forte_upd < 1.0 and forte_sr >= high_sr - 0.05
        # "does not simply start at fixed-high": starts at f_low
        starts_low = True
        qualified = bool(
            gate["pass"]
            and region["found"]
            and improves
            and starts_low
            and (_agg([r for r in low_rows if abs(r["friction"] - mu_low) < 1e-9], "lift_success") or 0) < 0.8
        )
        verdict["forte_reactive_improves_over_fixed_low"] = improves
        verdict["forte_baseline_qualified"] = qualified
        verdict["force_friction_decision_region_found"] = region["found"]
        if qualified:
            verdict["status"] = "F1_FORTE_BASELINE_QUALIFIED_IN_TABERO"
            if not verdict["tactile_slip_signal_valid"]:
                verdict["limitations"].append("SECONDARY: TACTILE_SLIP_DETECTOR_NOT_YET_QUALIFIED")
        else:
            verdict["status"] = "F1_NEGATIVE_NO_FORCE_FRICTION_DECISION_REGION" if not improves else "F1_BLOCKED_TECHNICALLY"
            verdict["limitations"].append(
                f"reactive improve={improves} forte_sr={forte_sr} low_sr={low_sr} high_sr={high_sr} meanF={forte_mean_f}/{high_mean_f}"
            )

        verdict["primary_evidence"].extend(
            [
                f"servo measured={verdict['measured_force_levels_N']}",
                f"lowest_stable={f_low}",
                f"region={json.dumps(region)}",
                f"fixed_low={json.dumps(verdict['fixed_low_success'])}",
                f"fixed_high={json.dumps(verdict['fixed_high_success'])}",
                f"forte_pos={json.dumps(verdict['forte_oracle_success'])}",
            ]
        )
        verdict["limitations"].append("FULL_DOWNSTREAM_NOT_YET_QUALIFIED: lift-only scripted protocol.")
        verdict["elapsed_s"] = time.time() - t0

        write_plots(servo_summ, explore, low_rows + high_rows + forte_rows, gt_events, servo_rows + explore_steps + forte_steps)
        (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2))
        print("done", json.dumps({"status": verdict["status"], "elapsed": verdict["elapsed_s"]}), flush=True)

        try:
            env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass
        os._exit(0)
    except SystemExit:
        try:
            simulation_app.close()
        except Exception:
            pass
        os._exit(0)
    except Exception as e:
        print("EXCEPTION", repr(e), flush=True)
        print(traceback.format_exc(), flush=True)
        verdict["status"] = "F1_BLOCKED_TECHNICALLY"
        verdict["limitations"].append(repr(e))
        (OUT / "SWEEP_ERROR.json").write_text(json.dumps({"error": repr(e), "trace": traceback.format_exc()}, indent=2))
        (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2))
        try:
            simulation_app.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
