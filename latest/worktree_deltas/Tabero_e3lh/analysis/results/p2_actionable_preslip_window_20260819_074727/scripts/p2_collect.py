#!/usr/bin/env python3
"""P2: oracle-timed 4N→6N intervention. METHOD_CHANGE=NONE.

Not a method. Not GelSight-triggered. Timing is ORACLE_INTERVENTION_TIMING
aligned to this-session baseline T_slip (μ=0.20, fixed 4 N).
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
OUT = Path(os.environ.get("P2_OUT", Path(__file__).resolve().parents[1]))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

sys.stdout = open(OUT / "logs" / "p2_isaac.log", "w", buffering=1)
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

N_SEEDS = int(os.environ.get("P2_N_SEEDS", "5"))
MU = 0.2
F0, F1 = 4.0, 6.0
# lead_ms; negative = after T_slip. "lift0" is a named condition.
LEAD_MS = [int(x) for x in os.environ.get("P2_LEADS", "500,400,300,250,200,150,100,0,-100").split(",") if x]
DO_FIXED6 = os.environ.get("P2_FIXED6", "1") == "1"
DO_LIFT0 = os.environ.get("P2_LIFT0", "1") == "1"
DO_PLACE = True
LOG_TRAJ = os.environ.get("P2_LOG_TRAJ", "1") == "1"

APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS, HOLD_STEPS = 45, 35, 70, 40
LIFT_STEPS = 50
TRANSIT_STEPS, OVER_BASKET_STEPS, PLACE_STEPS = 110, 30, 40
RELEASE_STEPS, SETTLE_PLACE_STEPS = 50, 50
PREGRASP_Z, GRASP_Z, LIFT_Z = 0.10, 0.018, 0.16
PLACE_CLEAR_Z, PLACE_Z = 0.18, 0.10

SERVO_STEP, SERVO_DEADBAND = 0.0006, 0.4
D_OPEN, D_CLOSED = 0.04, 0.0
LIFT_SUCCESS_Z = 0.03
# P1 / F1R2 / R1-v1 gross slip — do not retune
REL_Z_SLIP, V_REL_SLIP, VNEG_RUN_N, XY_SLIP = 0.008, -0.05, 2, 0.015
LOSS_F_N, LOSS_STREAK = 0.3, 3
F_EFF = 5.5
SLAM_N = 10.0
SETTLE_AFTER_BUMP = 20


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


def _rewrite(path: Path, dicts, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in dicts:
            w.writerow(d)


@dataclass
class Step:
    trial_id: str
    cond: str
    seed_idx: int
    phase: str
    step: int
    t_s: float
    f_cmd: float
    f_meas: float
    d_pred: float
    rel_z: float
    v_rel_z: float
    obj_dz: float
    contact: int
    gt_slip: int
    gt_loss: int
    intervened: int


def run_episode(
    env,
    *,
    seed_idx: int,
    trial_id: str,
    cond: str,
    dt: float,
    t_slip_ref: float | None,
    t_lift0_ref: float | None,
    lead_ms: int | None,
):
    try:
        obs, _ = env.reset(seed=int(seed_idx))
    except TypeError:
        import torch

        torch.manual_seed(int(seed_idx))
        np.random.seed(int(seed_idx))
        obs, _ = env.reset()
    applied = _apply_friction(env, OBJ_NAME, MU)

    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    ee_pos = eef0[:3].copy()
    eef_aa = _aa(eef0[3:7])
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
    if cond == "fixed6":
        f_cmd = F1
        t_cmd_sched = None
    elif cond == "fixed4":
        f_cmd = F0
        t_cmd_sched = None
    elif cond == "lift0":
        f_cmd = F0
        t_cmd_sched = t_lift0_ref  # may be None; then first lift step
    else:
        f_cmd = F0
        t_cmd_sched = None if t_slip_ref is None or lead_ms is None else (t_slip_ref - lead_ms / 1000.0)

    step = 0
    t_lift0 = t_micro = t_slip = t_loss = None
    t_cmd_actual = None
    t_rise = t_5 = t_55 = t_6band = None
    f_at_cmd = None
    peak_force = 0.0
    bump_left = 0
    intervened = 0
    grasp_rel = None
    prev_rel = None
    vneg = 0
    contact_ever = high_f_ever = 0
    loss_streak = 0
    lift_success = pick_success = dropped = 0
    lost_in_transit = 0
    retained_over_basket = 0
    official_success = 0
    max_basket = 0.0
    rows: list[Step] = []

    phases = [
        ("approach", APPROACH_STEPS, pregrasp, "open"),
        ("descend", DESCEND_STEPS, grasp, "open"),
        ("close", CLOSE_STEPS, grasp, "track"),
        ("hold", HOLD_STEPS, grasp, "track"),
        ("lift", LIFT_STEPS, lift, "track"),
        ("transit", TRANSIT_STEPS, transit, "freeze"),
        ("over_basket", OVER_BASKET_STEPS, transit, "freeze"),
        ("place", PLACE_STEPS, place, "freeze"),
        ("release", RELEASE_STEPS, place, "open"),
        ("settle", SETTLE_PLACE_STEPS, place, "open"),
    ]

    for phase, n_steps, target, mode in phases:
        start = cmd_pos.copy()
        for i in range(n_steps):
            cmd_pos = _interp(start, target, i, n_steps)
            if mode == "open":
                d_pred = D_OPEN
            t_before = (step + 1) * dt
            if phase == "lift" and t_lift0 is None:
                # t_s after this step will be t_before
                pass

            # oracle command: switch 4→6 at scheduled time (or first lift step for lift0)
            if cond not in ("fixed4", "fixed6") and intervened == 0 and mode != "open":
                fire = False
                if cond == "lift0" and phase == "lift" and i == 0:
                    fire = True
                elif t_cmd_sched is not None and t_before + 1e-9 >= t_cmd_sched and phase in (
                    "close", "hold", "lift", "transit", "over_basket"
                ):
                    fire = True
                if fire:
                    f_cmd = F1
                    intervened = 1
                    t_cmd_actual = t_before
                    bump_left = SETTLE_AFTER_BUMP

            action = _make_action(cmd_pos, eef_aa, d_pred, f_cmd if mode != "open" else 0.0, env.device)
            obs, rew, term, trunc, info = env.step(action)
            step += 1
            t_s = step * dt
            if phase == "lift" and t_lift0 is None:
                t_lift0 = t_s
                if cond == "lift0" and t_cmd_actual is None and intervened:
                    t_cmd_actual = t_s

            f = obs["policy"]["gripper_net_force"][0]
            if f.ndim == 3:
                f = f[-1]
            fL = f[0].detach().cpu().numpy()
            fR = f[1].detach().cpu().numpy()
            f_sq = _f(_dbg(env).get("f_sq_meas"), 0.0)
            peak_force = max(peak_force, f_sq)
            if f_sq > 2.0:
                high_f_ever = 1

            do_track = mode == "track" or (mode == "freeze" and bump_left > 0)
            if do_track:
                d_pred = force_servo(d_pred, f_sq, f_cmd)
                if bump_left > 0:
                    bump_left -= 1
                    if abs(f_sq - f_cmd) <= 0.8:
                        bump_left = 0

            if intervened and t_cmd_actual is not None:
                if f_at_cmd is None:
                    f_at_cmd = f_sq
                if t_rise is None and f_at_cmd is not None and f_sq >= f_at_cmd + 0.3:
                    t_rise = t_s
                if t_5 is None and f_sq >= 5.0:
                    t_5 = t_s
                if t_55 is None and f_sq >= F_EFF:
                    t_55 = t_s
                if t_6band is None and abs(f_sq - F1) <= 0.8:
                    t_6band = t_s

            obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
            ee = obs["policy"]["eef_pose"][0].detach().cpu().numpy()[:3]
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
                if int(bool(obs["subtask_terms"]["grasp_1"][0].item())):
                    pick_success = 1
            except Exception:
                pass

            rel = obj_p - ee
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
            if phase in ("hold", "lift", "transit", "over_basket"):
                if dropped or (grasp_rel is not None and rel_d[2] < -REL_Z_SLIP):
                    gt_slip = 1
                if phase == "lift" and vneg >= VNEG_RUN_N:
                    gt_slip = 1
                if grasp_rel is not None and np.linalg.norm(rel_d[:2]) > XY_SLIP:
                    gt_slip = 1
                if contact_ever and (not contact) and phase in ("lift", "transit"):
                    gt_slip = 1
            if high_f_ever and f_sq < LOSS_F_N and phase in ("lift", "transit"):
                loss_streak += 1
            else:
                loss_streak = 0
            gt_loss = int(
                dropped
                or (phase in ("lift", "transit") and contact_ever and not contact and loss_streak >= 1)
                or loss_streak >= LOSS_STREAK
            )
            if phase == "lift" and grasp_rel is not None and t_micro is None:
                if rel_d[2] < -0.002 or v_rel[2] < -0.015:
                    t_micro = t_s
            if gt_slip and t_slip is None:
                t_slip = t_s
            if gt_loss and t_loss is None:
                t_loss = t_s

            obj_dz = float(obj_p[2] - obj0_w[2])
            if obj_dz >= 0.01 and contact:
                pick_success = 1
            if obj_dz >= LIFT_SUCCESS_Z:
                lift_success = 1
            if lift_success and phase in ("transit", "over_basket") and (dropped or obj_dz < 0.02):
                lost_in_transit = 1
            if phase == "over_basket":
                retained_over_basket = int((not dropped) and obj_dz >= LIFT_SUCCESS_Z)

            try:
                import torch as _t

                cs = env.scene[f"contact_{BASKET_NAME}_{OBJ_NAME}"]
                bc = float(_t.linalg.vector_norm(cs.data.force_matrix_w.reshape(-1, 3)[0]).item())
                max_basket = max(max_basket, bc)
            except Exception:
                pass

            if LOG_TRAJ:
                rows.append(
                    Step(
                        trial_id=trial_id, cond=cond, seed_idx=seed_idx, phase=phase, step=step, t_s=t_s,
                        f_cmd=f_cmd, f_meas=f_sq, d_pred=d_pred,
                        rel_z=float(rel_d[2]), v_rel_z=float(v_rel[2]), obj_dz=obj_dz,
                        contact=contact, gt_slip=gt_slip, gt_loss=gt_loss, intervened=intervened,
                    )
                )
            if bool(term[0].item()) or bool(trunc[0].item()):
                break
        else:
            continue
        break

    honest_full = int(lift_success == 1 and max_basket > 0.05)
    retained = int(lift_success and not lost_in_transit and retained_over_basket)
    t_eff = t_55
    cmd_lead_ms = None
    if t_cmd_actual is not None and t_slip_ref is not None:
        cmd_lead_ms = (t_slip_ref - t_cmd_actual) * 1000.0
    eff_lead_ms = None
    if t_eff is not None and t_slip_ref is not None:
        eff_lead_ms = (t_slip_ref - t_eff) * 1000.0
    reached6_before_slip = int(t_eff is not None and t_slip_ref is not None and t_eff < t_slip_ref)
    reached6_before_loss = int(t_eff is not None and (t_loss is None or t_eff < t_loss))
    # Case B: 6N effective before baseline slip but still failed
    case_b = int(cond not in ("fixed4",) and honest_full == 0 and reached6_before_slip == 1)

    rec = {
        "trial_id": trial_id,
        "cond": cond,
        "seed_idx": seed_idx,
        "friction": MU,
        "lead_ms": "" if lead_ms is None and cond != "lift0" else ("lift0" if cond == "lift0" else lead_ms),
        "oracle_timing": int(cond not in ("fixed4", "fixed6")),
        "t_lift0": t_lift0,
        "t_micro": t_micro,
        "t_slip": t_slip,
        "t_loss": t_loss,
        "t_slip_ref": t_slip_ref,
        "t_cmd_sched": t_cmd_sched,
        "t_cmd_actual": t_cmd_actual,
        "t_force_rise": t_rise,
        "t_reach_5N": t_5,
        "t_force_effective": t_eff,
        "t_reach_6band": t_6band,
        "cmd_lead_vs_ref_slip_ms": cmd_lead_ms,
        "effective_lead_vs_ref_slip_ms": eff_lead_ms,
        "f_at_cmd": f_at_cmd,
        "peak_force": peak_force,
        "slam": int(peak_force >= SLAM_N),
        "intervened": intervened,
        "reached6_before_ref_slip": reached6_before_slip,
        "reached6_before_this_loss": reached6_before_loss,
        "case_b_force_ok_still_fail": case_b,
        "pick_success": pick_success,
        "lift_success": lift_success,
        "grasp_retained": retained,
        "transport_success": int(lift_success and not lost_in_transit),
        "basket_contact_max": max_basket,
        "place_success": honest_full,
        "full_task_success": honest_full,
        "official_success": official_success,
        "dropped": dropped,
        "friction_applied": applied,
        "steps": step,
    }
    print(
        f"summary {trial_id} cond={cond} lift={lift_success} full={honest_full} "
        f"T0={t_lift0} Tslip={t_slip} Tcmd={t_cmd_actual} Teff={t_eff} peak={peak_force:.2f} caseB={case_b}",
        flush=True,
    )
    return rows, rec


RESULT_FIELDS = [
    "trial_id", "cond", "seed_idx", "friction", "lead_ms", "oracle_timing",
    "t_lift0", "t_micro", "t_slip", "t_loss", "t_slip_ref", "t_cmd_sched", "t_cmd_actual",
    "t_force_rise", "t_reach_5N", "t_force_effective", "t_reach_6band",
    "cmd_lead_vs_ref_slip_ms", "effective_lead_vs_ref_slip_ms",
    "f_at_cmd", "peak_force", "slam", "intervened",
    "reached6_before_ref_slip", "reached6_before_this_loss", "case_b_force_ok_still_fail",
    "pick_success", "lift_success", "grasp_retained", "transport_success",
    "basket_contact_max", "place_success", "full_task_success", "official_success",
    "dropped", "friction_applied", "steps",
]


def plan_jobs():
    jobs = []
    for s in range(N_SEEDS):
        jobs.append(("fixed4", s, None))
        if DO_FIXED6:
            jobs.append(("fixed6", s, None))
        if DO_LIFT0:
            jobs.append(("lift0", s, None))
        for lead in LEAD_MS:
            jobs.append(("intv", s, lead))
    return jobs


def main():
    from isaaclab.app import AppLauncher

    enable_cameras = os.environ.get("P2_ENABLE_CAMERAS", "0") not in ("0", "false", "False")
    app_launcher = AppLauncher(headless=True, enable_cameras=enable_cameras, num_envs=1)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(TASK_SUITE, TASK_ID)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = float(os.environ.get("P2_EPISODE_S", "45"))
        if not enable_cameras:
            for name in ("eye_in_hand_cam", "agentview_cam", "gsmini_left", "gsmini_right"):
                if hasattr(env_cfg.scene, name):
                    delattr(env_cfg.scene, name)
            pol = getattr(env_cfg.observations, "policy", None)
            if pol is not None and hasattr(pol, "gripper_marker_motion"):
                delattr(pol, "gripper_marker_motion")
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        print("env ready dt", dt, "n_seeds", N_SEEDS, "leads", LEAD_MS, flush=True)
        (OUT / "COLLECT_META.json").write_text(json.dumps({
            "dt": dt, "n_seeds": N_SEEDS, "leads": LEAD_MS, "mu": MU, "f0": F0, "f1": F1,
            "oracle_timing": True, "cameras": enable_cameras, "slip_rule": "P1_8mm",
        }, indent=2) + "\n")

        res_path = OUT / "INTERVENTION_RESULTS.csv"
        results = []
        done = set()
        if res_path.exists() and res_path.stat().st_size > 0:
            with res_path.open() as f:
                for row in csv.DictReader(f):
                    done.add(row["trial_id"])
                    results.append(row)
        print("resume", len(done), flush=True)

        # per-seed baseline T_slip from completed fixed4, else from this session as we go
        tslip = {}
        tlift = {}
        for r in results:
            if r["cond"] == "fixed4":
                tslip[int(float(r["seed_idx"]))] = float(r["t_slip"]) if r.get("t_slip") not in ("", "None", None) else None
                tlift[int(float(r["seed_idx"]))] = float(r["t_lift0"]) if r.get("t_lift0") not in ("", "None", None) else None

        step_fields = list(Step.__dataclass_fields__.keys())
        jobs = plan_jobs()
        for cond, s, lead in jobs:
            if cond == "intv":
                tid = f"p2_s{s}_lead{lead}"
            else:
                tid = f"p2_s{s}_{cond}"
            if tid in done:
                print("skip", tid, flush=True)
                continue
            if cond in ("intv", "lift0") and s not in tslip:
                print("NEED_BASELINE_FIRST", tid, flush=True)
                continue
            print("===", tid, "===", flush=True)
            rows, rec = run_episode(
                env,
                seed_idx=s,
                trial_id=tid,
                cond=cond,
                dt=dt,
                t_slip_ref=tslip.get(s),
                t_lift0_ref=tlift.get(s),
                lead_ms=lead,
            )
            if cond == "fixed4":
                tslip[s] = rec["t_slip"]
                tlift[s] = rec["t_lift0"]
            results.append(rec)
            done.add(tid)
            _rewrite(res_path, results, RESULT_FIELDS)
            if LOG_TRAJ and rows:
                _append(OUT / "FORCE_TRAJECTORIES.csv", [asdict(r) for r in rows], step_fields)

        print("done n", len(results), flush=True)
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
