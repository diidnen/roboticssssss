#!/usr/bin/env python3
"""B4: FORTE-inspired reactive baseline on frozen 5-task Tabero benchmark.

This is not an official FORTE reproduction. It uses Tabero's calibrated force
servo and compares the FORTE-style information/control principle:

    start low -> detect GT slip/instability -> increment force target.

METHOD_CHANGE=NONE. This script writes only inside the B4 result directory.
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
OUT = Path(os.environ.get("B4_OUT", Path(__file__).resolve().parents[1]))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

LOG_NAME = os.environ.get("B4_LOG_NAME", "b4_eval_isaac.log")
sys.stdout = open(OUT / "logs" / LOG_NAME, "w", buffering=1)
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
BASKET_NAME = "basket_1"

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
TASK_INSTRUCTIONS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    3: "pick up the bbq sauce and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
    7: "pick up the milk and place it in the basket",
    8: "pick up the chocolate pudding and place it in the basket",
    9: "pick up the orange juice and place it in the basket",
}
FSTAR = {
    0: {0.2: 5.0, 0.5: 4.0, 1.0: 3.0},
    1: {0.2: 6.0, 0.5: 5.0, 1.0: 3.0},
    2: {0.2: 8.0, 0.5: 3.0, 1.0: 3.0},
    5: {0.2: 5.0, 0.5: 4.0, 1.0: 3.0},
    6: {0.2: 4.0, 0.5: 3.0, 1.0: 3.0},
    7: {0.2: 3.0, 0.5: 3.0, 1.0: 3.0},
}
FIXED_ROBUST = {0: 5.0, 1: 6.0, 2: 8.0, 5: 5.0, 6: 4.0, 7: 3.0}
POSITIVE_TASKS = [0, 1, 2, 5, 6]
FORCE_LADDER = [3.0, 4.0, 5.0, 6.0, 8.0]
INITIAL_FORCE = 3.0

TASK_ID = int(os.environ.get("B4_TASK_ID", "1"))
OBJ_NAME = TASK_OBJECTS[TASK_ID]
INSTRUCTION = TASK_INSTRUCTIONS[TASK_ID]
for banned in ("gently", "softly", "firmly", "tightly", "slippery", "low friction", "high friction"):
    assert banned not in INSTRUCTION.lower(), f"force/physics hint leaked in instruction: {banned}"

MODE = os.environ.get("B4_MODE", "main")  # dynamic | smoke | main | early_gt | negative
N_SEEDS = int(os.environ.get("B4_N_SEEDS", "20"))
SEED0 = int(os.environ.get("B4_SEED0", "0"))
MUS = [float(x) for x in os.environ.get("B4_MUS", "0.2,0.5,1.0").split(",") if x.strip()]
METHODS = [m.strip() for m in os.environ.get("B4_METHODS", "fixed_low,fixed_robust,gt_minforce,forte_gt_reactive").split(",") if m.strip()]
LOG_TRAJ = os.environ.get("B4_LOG_TRAJ", "1") not in ("0", "false", "False")
LOG_EVERY = int(os.environ.get("B4_LOG_EVERY", "1"))
TRAJ_METHODS = {m.strip() for m in os.environ.get("B4_TRAJ_METHODS", "forte_gt_reactive,early_gt_reference").split(",") if m.strip()}
TRAJ_MUS = {float(x) for x in os.environ.get("B4_TRAJ_MUS", "0.2").split(",") if x.strip()}

# Same scripted downstream as B2/B2-R2/E2E.
APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS, HOLD_STEPS = 45, 35, 70, 40
LIFT_STEPS = 50
TRANSIT_STEPS, OVER_BASKET_STEPS, PLACE_STEPS = 110, 30, 40
RELEASE_STEPS, SETTLE_PLACE_STEPS = 50, 50
PREGRASP_Z, GRASP_Z, LIFT_Z = 0.10, 0.018, 0.16
PLACE_CLEAR_Z, PLACE_Z = 0.18, 0.10
SERVO_STEP, SERVO_DEADBAND = 0.0006, 0.4
D_OPEN, D_CLOSED = 0.04, 0.0
LIFT_SUCCESS_Z = 0.03

# Frozen P1/F1R2/R1-v1 gross GT slip definition.
REL_Z_SLIP, V_REL_SLIP, VNEG_RUN_N, XY_SLIP = 0.008, -0.05, 2, 0.015

# Optional EARLY-GT reference: privileged micro-instability, not FORTE.
REL_Z_EARLY, V_REL_EARLY, VNEG_EARLY_N = 0.002, -0.015, 1

SETTLE_TOL_N = 0.8
SETTLE_MAX_STEPS = 20
SLAM_N = 20.0
NOMINAL_MAT = None


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
    d_pred = d_pred - SERVO_STEP if err > 0 else d_pred + SERVO_STEP
    return float(np.clip(d_pred, D_CLOSED, D_OPEN))


def _interp(start, end, i, n):
    a = (i + 1) / max(n, 1)
    return (1 - a) * start + a * end


def _apply_friction(env, name: str, mu: float) -> dict:
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
    return {
        "requested_mu": mu,
        "applied_static_mean": float(got[:, 0].mean()),
        "applied_dynamic_mean": float(got[:, 1].mean()),
        "nominal_static_mean": float(NOMINAL_MAT.detach().cpu().numpy().reshape(-1, 3)[:, 0].mean()),
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    if fields is None and rows:
        fields = list(rows[0].keys())
    if fields is None:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _append_csv(path: Path, rows: list[dict], fields: list[str]):
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
class StepRow:
    trial_id: str
    task_id: int
    object: str
    method: str
    seed_idx: int
    friction: float
    phase: str
    step: int
    t_s: float
    force_target: float
    measured_force: float
    d_pred: float
    obj_dz: float
    rel_z: float
    rel_xy: float
    v_rel_z: float
    contact: int
    gt_slip: int
    early_gt: int
    state: str
    n_updates: int


class ReactiveForceLadder:
    def __init__(self, *, early: bool = False):
        self.force = INITIAL_FORCE
        self.idx = 0
        self.state = "track"
        self.settle_left = 0
        self.n_updates = 0
        self.events: list[dict] = []
        self.early = early
        self.pending_event_idx: int | None = None

    def _raise(self, *, t_s, f_meas, rel_z, rel_xy, phase, reason):
        if self.idx >= len(FORCE_LADDER) - 1:
            return
        before = self.force
        self.idx += 1
        self.force = FORCE_LADDER[self.idx]
        self.n_updates += 1
        self.state = "settle"
        self.settle_left = SETTLE_MAX_STEPS
        self.events.append(
            {
                "event_index": self.n_updates,
                "t_update_s": t_s,
                "t_reached_s": None,
                "phase": phase,
                "reason": reason,
                "force_before": before,
                "force_after": self.force,
                "f_meas_at_update": f_meas,
                "rel_z": rel_z,
                "rel_xy": rel_xy,
            }
        )
        self.pending_event_idx = len(self.events) - 1

    def step(self, *, event_flag, t_s, f_meas, rel_z, rel_xy, phase, allow_update):
        if self.pending_event_idx is not None and abs(f_meas - self.force) <= SETTLE_TOL_N:
            self.events[self.pending_event_idx]["t_reached_s"] = t_s
            self.pending_event_idx = None
        if self.state == "settle":
            self.settle_left -= 1
            if abs(f_meas - self.force) <= SETTLE_TOL_N or self.settle_left <= 0:
                self.state = "track"
            return
        if allow_update and event_flag:
            self._raise(t_s=t_s, f_meas=f_meas, rel_z=rel_z, rel_xy=rel_xy, phase=phase, reason="early_gt" if self.early else "gt_slip")


def method_initial_force(method: str, mu: float) -> tuple[float, str]:
    if method == "fixed_low":
        return INITIAL_FORCE, "fixed_low_3N"
    if method == "fixed_robust":
        return FIXED_ROBUST[TASK_ID], "fixed_robust_task_force"
    if method == "gt_minforce":
        return FSTAR[TASK_ID][round(mu, 2)], "canonical_gt_minforce"
    if method in ("forte_gt_reactive", "early_gt_reference"):
        return INITIAL_FORCE, "reactive_start_3N"
    raise ValueError(f"unknown method {method}")


def run_episode(env, *, method: str, seed_idx: int, mu: float, trial_id: str, dt: float):
    try:
        obs, _ = env.reset(seed=int(seed_idx))
    except TypeError:
        import torch

        torch.manual_seed(int(seed_idx))
        np.random.seed(int(seed_idx))
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
    initial_f, decision_rule = method_initial_force(method, mu)
    sm = ReactiveForceLadder(early=(method == "early_gt_reference")) if method in ("forte_gt_reactive", "early_gt_reference") else None
    f_cmd = initial_f

    rows: list[StepRow] = []
    event_rows: list[dict] = []
    step = 0
    pick_success = lift_success = 0
    transport_retention = 0
    lost_in_transit = 0
    place_success = 0
    dropped = timed_out = official_success = 0
    max_basket_contact = 0.0
    peak_force = 0.0
    active_force_integral = 0.0
    active_force_samples: list[float] = []
    target_integral = 0.0
    grasp_rel = None
    prev_rel_z = None
    vneg_run = 0
    vneg_early_run = 0
    contact_ever = 0
    t_lift0 = None
    t_gt_slip = None
    t_early_gt = None
    t_first_update = None
    t_recovery = None
    obj_disp_before_first_update = None
    obj_disp_before_sufficient_force = None
    first_sufficient_t = None
    fstar = FSTAR[TASK_ID][round(mu, 2)]

    def phases():
        yield "approach", APPROACH_STEPS, pregrasp, "open"
        yield "descend", DESCEND_STEPS, grasp, "open"
        yield "close", CLOSE_STEPS, grasp, "track"
        yield "hold", HOLD_STEPS, grasp, "track"
        yield "lift", LIFT_STEPS, lift, "track"
        yield "transit", TRANSIT_STEPS, transit, "freeze"
        yield "over_basket", OVER_BASKET_STEPS, transit, "freeze"
        yield "place", PLACE_STEPS, place, "freeze"
        yield "release", RELEASE_STEPS, place, "open"
        yield "settle", SETTLE_PLACE_STEPS, place, "open"

    for phase, n_steps, target, servo_mode in phases():
        start = cmd_pos.copy()
        if phase == "lift":
            t_lift0 = (step + 1) * dt
        for i in range(n_steps):
            cmd_pos = _interp(start, target, i, n_steps)
            if servo_mode == "open":
                d_pred = D_OPEN
            cur_target = sm.force if sm is not None else f_cmd
            action = _make_action(cmd_pos, eef_aa, d_pred, cur_target if servo_mode != "open" else 0.0, env.device)
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

            if servo_mode == "track" or (sm is not None and sm.state == "settle" and servo_mode == "freeze"):
                d_pred = force_servo(d_pred, f_sq, cur_target)

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
            try:
                grasp_flag = int(bool(obs["subtask_terms"]["grasp_1"][0].item()))
            except Exception:
                grasp_flag = 0

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
            if v_rel_z < V_REL_EARLY:
                vneg_early_run += 1
            else:
                vneg_early_run = 0

            gt_slip = 0
            early_gt = 0
            if phase in ("hold", "lift", "transit", "over_basket"):
                if dropped:
                    gt_slip = 1
                    early_gt = 1
                if grasp_rel is not None and rel_z < -REL_Z_SLIP:
                    gt_slip = 1
                if phase == "lift" and vneg_run >= VNEG_RUN_N:
                    gt_slip = 1
                if grasp_rel is not None and rel_xy > XY_SLIP:
                    gt_slip = 1
                if contact_ever and (not contact) and phase in ("lift", "transit", "over_basket"):
                    gt_slip = 1
                if grasp_rel is not None and rel_z < -REL_Z_EARLY:
                    early_gt = 1
                if phase == "lift" and vneg_early_run >= VNEG_EARLY_N:
                    early_gt = 1
            if gt_slip and t_gt_slip is None:
                t_gt_slip = t_s
            if early_gt and t_early_gt is None:
                t_early_gt = t_s

            allow_update = phase in ("lift", "transit", "over_basket")
            before_updates = sm.n_updates if sm is not None else 0
            if sm is not None:
                event_flag = early_gt if method == "early_gt_reference" else gt_slip
                sm.step(event_flag=event_flag, t_s=t_s, f_meas=f_sq, rel_z=rel_z, rel_xy=rel_xy, phase=phase, allow_update=allow_update)
                if sm.n_updates > before_updates:
                    ev = dict(sm.events[-1])
                    ev.update({"trial_id": trial_id, "task_id": TASK_ID, "object": OBJ_NAME, "method": method, "seed_idx": seed_idx, "friction": mu})
                    event_rows.append(ev)
                    if t_first_update is None:
                        t_first_update = t_s
                        obj_disp_before_first_update = float(np.linalg.norm(obj_p - obj0_w))

            cur_target = sm.force if sm is not None else f_cmd
            if first_sufficient_t is None and cur_target >= fstar - 1e-6 and f_sq >= max(0.0, fstar - SETTLE_TOL_N):
                first_sufficient_t = t_s
                obj_disp_before_sufficient_force = float(np.linalg.norm(obj_p - obj0_w))

            basket_contact = 0.0
            try:
                import torch as _t

                cs = env.scene[f"contact_{BASKET_NAME}_{OBJ_NAME}"]
                basket_contact = float(_t.linalg.vector_norm(cs.data.force_matrix_w.reshape(-1, 3)[0]).item())
            except Exception:
                pass
            max_basket_contact = max(max_basket_contact, basket_contact)

            obj_dz = float(obj_p[2] - obj0_w[2])
            if grasp_flag or (contact and obj_dz > 0.01):
                pick_success = 1
            if obj_dz >= LIFT_SUCCESS_Z:
                lift_success = 1
                if t_recovery is None and t_first_update is not None:
                    t_recovery = t_s
            if lift_success and phase in ("transit", "over_basket") and (dropped or obj_dz < 0.02):
                lost_in_transit = 1
            if phase == "over_basket" and lift_success and (not dropped) and obj_dz >= LIFT_SUCCESS_Z:
                transport_retention = 1
            if max_basket_contact > 0.05:
                place_success = 1

            peak_force = max(peak_force, f_sq)
            if servo_mode != "open":
                active_force_integral += f_sq * dt
                target_integral += cur_target * dt
                active_force_samples.append(f_sq)

            log_this_traj = LOG_TRAJ and method in TRAJ_METHODS and round(mu, 3) in {round(x, 3) for x in TRAJ_MUS}
            if log_this_traj and (step % LOG_EVERY == 0 or gt_slip or early_gt):
                rows.append(
                    StepRow(
                        trial_id=trial_id,
                        task_id=TASK_ID,
                        object=OBJ_NAME,
                        method=method,
                        seed_idx=seed_idx,
                        friction=mu,
                        phase=phase,
                        step=step,
                        t_s=t_s,
                        force_target=cur_target,
                        measured_force=f_sq,
                        d_pred=d_pred,
                        obj_dz=obj_dz,
                        rel_z=rel_z,
                        rel_xy=rel_xy,
                        v_rel_z=v_rel_z,
                        contact=contact,
                        gt_slip=gt_slip,
                        early_gt=early_gt,
                        state=sm.state if sm is not None else "fixed",
                        n_updates=sm.n_updates if sm is not None else 0,
                    )
                )

            if bool(term[0].item()) or bool(trunc[0].item()):
                try:
                    timed_out = int(bool(env.termination_manager.get_term("time_out")[0].item()))
                except Exception:
                    timed_out = int(bool(trunc[0].item()))
                break
        else:
            continue
        break

    full_success = int(lift_success == 1 and place_success == 1 and dropped == 0)
    final_target = sm.force if sm is not None else f_cmd
    update_times = []
    reached_times = []
    update_path = []
    if sm is not None:
        for ev in sm.events:
            update_times.append(ev["t_update_s"])
            reached_times.append(ev["t_reached_s"])
            update_path.append(f"{ev['force_before']:g}->{ev['force_after']:g}@{ev['t_update_s']:.2f}")

    summary = {
        "trial_id": trial_id,
        "task_id": TASK_ID,
        "object": OBJ_NAME,
        "instruction": INSTRUCTION,
        "method": method,
        "baseline_name": "FORTE_INSPIRED_GT_REACTIVE" if method == "forte_gt_reactive" else method,
        "official_forte_reproduction": 0,
        "seed_idx": seed_idx,
        "friction": mu,
        "friction_static_applied": fr_info["applied_static_mean"],
        "friction_dynamic_applied": fr_info["applied_dynamic_mean"],
        "initial_force_N": initial_f,
        "final_force_target_N": final_target,
        "fstar_N": fstar,
        "fixed_robust_N": FIXED_ROBUST[TASK_ID],
        "decision_rule": decision_rule,
        "force_ladder_N": "3;4;5;6;8",
        "force_target_trajectory": ";".join(update_path) if update_path else f"hold_{initial_f:g}",
        "update_times_s": ";".join("" if t is None else f"{t:.4f}" for t in update_times),
        "reached_force_times_s": ";".join("" if t is None else f"{t:.4f}" for t in reached_times),
        "pick_success": pick_success,
        "lift_success": lift_success,
        "transport_success": transport_retention,
        "place_success": place_success,
        "full_success": full_success,
        "official_success": official_success,
        "dropped": dropped,
        "timeout": timed_out,
        "lost_in_transit": lost_in_transit,
        "basket_contact_max": max_basket_contact,
        "mean_force_N": float(np.mean(active_force_samples)) if active_force_samples else 0.0,
        "peak_force_N": peak_force,
        "integrated_force_Ns": active_force_integral,
        "integrated_target_Ns": target_integral,
        "number_of_updates": sm.n_updates if sm is not None else 0,
        "gt_slip_onset_s": t_gt_slip,
        "early_gt_onset_s": t_early_gt,
        "first_update_s": t_first_update,
        "first_event_latency_s": None if t_gt_slip is None or t_lift0 is None else t_gt_slip - t_lift0,
        "early_event_latency_s": None if t_early_gt is None or t_lift0 is None else t_early_gt - t_lift0,
        "slip_to_first_update_s": None if t_gt_slip is None or t_first_update is None else t_first_update - t_gt_slip,
        "recovery_latency_s": None if t_recovery is None or t_first_update is None else t_recovery - t_first_update,
        "time_to_sufficient_force_s": first_sufficient_t,
        "obj_displacement_before_first_update_m": obj_disp_before_first_update,
        "obj_displacement_before_sufficient_force_m": obj_disp_before_sufficient_force,
        "gt_slip_used": int(method == "forte_gt_reactive"),
        "early_gt_used": int(method == "early_gt_reference"),
        "slam_20N": int(peak_force >= SLAM_N),
        "steps": step,
        "t_episode_s": step * dt,
    }
    print(
        f"summary {trial_id} task={TASK_ID} mu={mu} method={method} finalF={final_target} "
        f"updates={summary['number_of_updates']} pick={pick_success} lift={lift_success} "
        f"place={place_success} full={full_success} meanF={summary['mean_force_N']:.2f}",
        flush=True,
    )
    return [asdict(r) for r in rows], summary, event_rows


def run_dynamic_audit(env, *, seed_idx: int, mu: float, dt: float):
    try:
        obs, _ = env.reset(seed=int(seed_idx))
    except TypeError:
        import torch

        torch.manual_seed(int(seed_idx))
        np.random.seed(int(seed_idx))
        obs, _ = env.reset()
    _apply_friction(env, OBJ_NAME, mu)
    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    ee_pos = eef0[:3].copy()
    eef_aa = _axis_angle_from_quat(eef0[3:7])
    obj0_w = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
    obj0_b, _ = _pose_in_base(env, OBJ_NAME)
    pregrasp = obj0_b.copy(); pregrasp[2] += PREGRASP_Z
    grasp = obj0_b.copy(); grasp[2] += GRASP_Z
    d_pred = D_OPEN
    cmd_pos = ee_pos.copy()
    step = 0
    peak_force = 0.0
    contact_retention = 1
    rows = []

    def do_step(target_pos, target_force, mode):
        nonlocal obs, d_pred, step, peak_force, contact_retention
        if mode == "open":
            d_pred = D_OPEN
        action = _make_action(target_pos, eef_aa, d_pred, target_force if mode != "open" else 0.0, env.device)
        obs, rew, term, trunc, info = env.step(action)
        step += 1
        f_sq = _to_float(_read_debug(env).get("f_sq_meas"), 0.0)
        if mode == "track":
            d_pred = force_servo(d_pred, f_sq, target_force)
        peak_force = max(peak_force, f_sq)
        obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
        if step > APPROACH_STEPS + DESCEND_STEPS and float(obj_p[2] - obj0_w[2]) < -0.02:
            contact_retention = 0
        return f_sq

    for phase, n, target, force, mode in [
        ("approach", APPROACH_STEPS, pregrasp, 0.0, "open"),
        ("descend", DESCEND_STEPS, grasp, 0.0, "open"),
        ("close3", CLOSE_STEPS, grasp, 3.0, "track"),
    ]:
        start = cmd_pos.copy()
        for i in range(n):
            cmd_pos = _interp(start, target, i, n)
            do_step(cmd_pos, force, mode)

    for before, after in zip(FORCE_LADDER[:-1], FORCE_LADDER[1:]):
        t_change = (step + 1) * dt
        reached = None
        max_f = 0.0
        samples = []
        for _ in range(35):
            f_sq = do_step(grasp, after, "track")
            samples.append(f_sq)
            max_f = max(max_f, f_sq)
            if reached is None and abs(f_sq - after) <= SETTLE_TOL_N:
                reached = step * dt
        rows.append(
            {
                "task_id": TASK_ID,
                "object": OBJ_NAME,
                "seed_idx": seed_idx,
                "friction": mu,
                "force_before_N": before,
                "force_after_N": after,
                "target_change_time_s": t_change,
                "reached_time_s": reached,
                "settling_time_s": None if reached is None else reached - t_change,
                "mean_measured_after_N": float(np.mean(samples)) if samples else 0.0,
                "peak_measured_after_N": max_f,
                "episode_peak_force_N": peak_force,
                "contact_retention": contact_retention,
                "servo_ok": int(reached is not None and peak_force < SLAM_N and contact_retention == 1),
            }
        )
    print(f"dynamic_audit task={TASK_ID} rows={len(rows)} peak={peak_force:.2f}", flush=True)
    return rows


def plan_jobs():
    jobs = []
    if MODE == "smoke":
        methods = ["fixed_low", "fixed_robust", "forte_gt_reactive"]
        mus = [0.2]
        n = N_SEEDS
    elif MODE == "early_gt":
        methods = ["early_gt_reference"]
        mus = MUS
        n = N_SEEDS
    elif MODE == "negative":
        methods = ["forte_gt_reactive"]
        mus = MUS
        n = N_SEEDS
    else:
        methods = METHODS
        mus = MUS
        n = N_SEEDS
    for method in methods:
        for mu in mus:
            for s in range(SEED0, SEED0 + n):
                tid = f"b4_t{TASK_ID}_{method}_s{s}_mu{mu:g}"
                jobs.append((method, s, mu, tid))
    return jobs


def main():
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False, num_envs=1)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(TASK_SUITE, TASK_ID)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = float(os.environ.get("B4_EPISODE_S", "45"))
        for name in ("eye_in_hand_cam", "agentview_cam", "gsmini_left", "gsmini_right"):
            if hasattr(env_cfg.scene, name):
                delattr(env_cfg.scene, name)
        pol = getattr(env_cfg.observations, "policy", None)
        if pol is not None and hasattr(pol, "gripper_marker_motion"):
            delattr(pol, "gripper_marker_motion")
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        print(
            f"env ready mode={MODE} dt={dt} task={TASK_ID} obj={OBJ_NAME} "
            f"seed0={SEED0} n={N_SEEDS} methods={METHODS} mus={MUS}",
            flush=True,
        )

        if MODE == "dynamic":
            path = OUT / f"DYNAMIC_FORCE_SERVO_AUDIT_TASK{TASK_ID}.csv"
            rows = run_dynamic_audit(env, seed_idx=SEED0, mu=0.2, dt=dt)
            _write_csv(path, rows)
        else:
            suffix = {
                "smoke": "SMOKE",
                "main": "MAIN",
                "early_gt": "EARLY_GT",
                "negative": "NEGATIVE",
            }.get(MODE, MODE.upper())
            summary_path = OUT / f"TASK{TASK_ID}_{suffix}_EPISODES.csv"
            traj_path = OUT / f"TASK{TASK_ID}_{suffix}_STEP_TRAJECTORIES.csv"
            events_path = OUT / f"TASK{TASK_ID}_{suffix}_FORCE_UPDATE_EVENTS.csv"

            summaries = []
            done = set()
            if summary_path.exists() and summary_path.stat().st_size > 0:
                with summary_path.open() as f:
                    for row in csv.DictReader(f):
                        summaries.append(row)
                        done.add(row["trial_id"])
            for method, seed_idx, mu, trial_id in plan_jobs():
                if trial_id in done:
                    print("skip", trial_id, flush=True)
                    continue
                print("===", trial_id, "===", flush=True)
                traj, summary, events = run_episode(env, method=method, seed_idx=seed_idx, mu=mu, trial_id=trial_id, dt=dt)
                summaries.append(summary)
                _write_csv(summary_path, summaries, list(summary.keys()))
                if traj:
                    _append_csv(traj_path, traj, list(traj[0].keys()))
                if events:
                    _append_csv(events_path, events, list(events[0].keys()))
            print(f"done summaries={len(summaries)} path={summary_path}", flush=True)

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
        (OUT / f"SWEEP_ERROR_task{TASK_ID}_{MODE}.json").write_text(json.dumps({"error": repr(e), "trace": traceback.format_exc()}, indent=2))
        try:
            simulation_app.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
