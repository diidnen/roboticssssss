#!/usr/bin/env python3
"""B2 Tabero benchmark qualification: scripted full-task force x friction scan.

Isolated from D2. METHOD_CHANGE=NONE. No probe, no VLA, no DeliGrasp/FORTE.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

TABERO = Path("/home/exouser/Tabero")
OUT = Path(os.environ.get("B2_OUT", Path(__file__).resolve().parents[1]))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

sys.stdout = open(OUT / "logs" / os.environ.get("B2_LOG_NAME", "b2_isaac.log"), "w", buffering=1)
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
TASK_ID = int(os.environ.get("B2_TASK_ID", "1"))
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
OBJ_NAME = TASK_OBJECTS[TASK_ID]
INSTRUCTION = TASK_INSTRUCTIONS[TASK_ID]
assert "gently" not in INSTRUCTION and "firmly" not in INSTRUCTION
assert "softly" not in INSTRUCTION and "tightly" not in INSTRUCTION

CSV_NAME = os.environ.get("B2_CSV", f"TASK{TASK_ID}_ORACLE_SCAN.csv")
N_SEEDS = int(os.environ.get("B2_N_SEEDS", "5"))
SEED0 = int(os.environ.get("B2_SEED0", "0"))
MUS = [float(x) for x in os.environ.get("B2_MUS", "0.2,0.5,1.0").split(",") if x.strip()]
FORCES = [float(x) for x in os.environ.get("B2_FORCES", "3,4,5,6,8").split(",") if x.strip()]
CELL_SPEC = os.environ.get("B2_CELL_SPEC", "").strip()

# Same scripted downstream as T1/P3 qualification (no probe).
APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS, HOLD_STEPS = 45, 35, 70, 40
LIFT_STEPS = 50
TRANSIT_STEPS, OVER_BASKET_STEPS, PLACE_STEPS = 110, 30, 40
RELEASE_STEPS, SETTLE_PLACE_STEPS = 50, 50
PREGRASP_Z, GRASP_Z, LIFT_Z = 0.10, 0.018, 0.16
PLACE_CLEAR_Z, PLACE_Z = 0.18, 0.10
SERVO_STEP, SERVO_DEADBAND = 0.0006, 0.4
D_OPEN, D_CLOSED = 0.04, 0.0
LIFT_SUCCESS_Z = 0.03
NOMINAL_MAT = None
NOMINAL_MU = None


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
    a = (i + 1) / max(n, 1)
    return (1 - a) * start + a * end


def _read_mu(env, name):
    got = env.scene[name].root_physx_view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
    return float(got[:, 0].mean()), float(got[:, 1].mean())


def _apply_friction(env, name, mu):
    global NOMINAL_MAT, NOMINAL_MU
    import torch

    view = env.scene[name].root_physx_view
    if NOMINAL_MAT is None:
        NOMINAL_MAT = view.get_material_properties().clone()
        NOMINAL_MU = float(NOMINAL_MAT.detach().cpu().numpy().reshape(-1, 3)[:, 0].mean())
    mats = NOMINAL_MAT.clone()
    mats[..., 0] = mu
    mats[..., 1] = mu
    view.set_material_properties(mats, torch.arange(mats.shape[0], dtype=torch.int32))
    got_s, got_d = _read_mu(env, name)
    ok = abs(got_s - mu) < 0.02 and abs(got_d - mu) < 0.02
    return got_s, got_d, ok, float(NOMINAL_MU)


def _dbg(env):
    try:
        return env.action_manager.get_term("arm_action").debug_info or {}
    except Exception:
        return {}


def _rewrite(path: Path, dicts, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in dicts:
            w.writerow(d)


def run_episode(env, *, seed_idx, mu, trial_id, f_cmd, dt):
    try:
        obs, _ = env.reset(seed=int(seed_idx))
    except TypeError:
        import torch

        torch.manual_seed(int(seed_idx))
        np.random.seed(int(seed_idx))
        obs, _ = env.reset()
    applied_s, applied_d, mu_ok, nominal_mu = _apply_friction(env, OBJ_NAME, mu)

    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    ee_pos = eef0[:3].copy()
    eef_aa = _aa(eef0[3:7])
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
    place = basket_b.copy()
    place[2] = basket_b[2] + PLACE_Z

    d_pred = D_OPEN
    cmd_pos = ee_pos.copy()
    step = 0
    dropped = lift_success = pick_success = 0
    lost_in_transit = timed_out = 0
    max_basket = 0.0
    peak_force = 0.0
    integrated_force = 0.0
    term_reason = ""
    force_samples = []

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
            action = _make_action(cmd_pos, eef_aa, d_pred, f_cmd if mode != "open" else 0.0, env.device)
            obs, rew, term, trunc, info = env.step(action)
            step += 1
            f_sq = _f(_dbg(env).get("f_sq_meas"), 0.0)
            peak_force = max(peak_force, f_sq)
            integrated_force += f_sq * dt
            if mode == "track":
                d_pred = force_servo(d_pred, f_sq, f_cmd)
                force_samples.append(f_sq)
            try:
                dropped = int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
            except Exception:
                pass
            obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
            obj_dz = float(obj_p[2] - obj0_w[2])
            if obj_dz >= 0.01:
                pick_success = 1
            if obj_dz >= LIFT_SUCCESS_Z:
                lift_success = 1
            if lift_success and phase in ("transit", "over_basket") and (dropped or obj_dz < 0.02):
                lost_in_transit = 1
            try:
                import torch as _t

                cs = env.scene[f"contact_{BASKET_NAME}_{OBJ_NAME}"]
                bc = float(_t.linalg.vector_norm(cs.data.force_matrix_w.reshape(-1, 3)[0]).item())
                max_basket = max(max_basket, bc)
            except Exception:
                pass
            if bool(term[0].item()) or bool(trunc[0].item()):
                try:
                    if bool(env.termination_manager.get_term("time_out")[0].item()):
                        timed_out = 1
                        term_reason = "time_out"
                except Exception:
                    term_reason = "term"
                break
        else:
            continue
        break

    honest_full = int(lift_success == 1 and max_basket > 0.05 and dropped == 0)
    rec = {
        "trial_id": trial_id,
        "phase": os.environ.get("B2_PHASE", "A"),
        "task_id": TASK_ID,
        "task_suite": TASK_SUITE,
        "object": OBJ_NAME,
        "instruction": INSTRUCTION,
        "seed_idx": seed_idx,
        "friction": mu,
        "friction_static_applied": applied_s,
        "friction_dynamic_applied": applied_d,
        "friction_override_ok": int(mu_ok),
        "nominal_mu": nominal_mu,
        "force": f_cmd,
        "pick_success": pick_success,
        "lift_success": lift_success,
        "transport_retention": int(lift_success == 1 and lost_in_transit == 0),
        "place_success": int(max_basket > 0.05),
        "full_task_success": honest_full,
        "basket_contact_max": max_basket,
        "dropped": dropped,
        "timeout": timed_out,
        "lost_in_transit": lost_in_transit,
        "term_reason": term_reason,
        "peak_force": peak_force,
        "mean_grip_force": float(np.mean(force_samples)) if force_samples else 0.0,
        "integrated_force": integrated_force,
        "steps": step,
        "t_episode_s": step * dt,
    }
    print(
        f"summary {trial_id} task={TASK_ID} mu={mu} F={f_cmd} pick={pick_success} lift={lift_success} "
        f"trans={rec['transport_retention']} place={rec['place_success']} full={honest_full} "
        f"mu_ok={int(mu_ok)} timeout={timed_out}",
        flush=True,
    )
    return rec


def _parse_cell_spec(spec):
    """Parse compact transition cells: '0.2:4,5;0.5:3,4;1.0:3'."""
    cells = []
    if not spec:
        return cells
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        mu_s, force_s = chunk.split(":", 1)
        mu = float(mu_s)
        for f_s in force_s.split(","):
            f_s = f_s.strip()
            if f_s:
                cells.append((float(f_s), mu))
    return cells


def plan_jobs():
    jobs = []
    cells = _parse_cell_spec(CELL_SPEC)
    if not cells:
        cells = [(F, mu) for F in FORCES for mu in MUS]
    for F, mu in cells:
        for s in range(SEED0, SEED0 + N_SEEDS):
            tid = f"b2_t{TASK_ID}_F{F:g}_s{s}_mu{mu:g}"
            jobs.append((F, s, mu, tid))
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
        env_cfg.episode_length_s = float(os.environ.get("B2_EPISODE_S", "45"))
        for name in ("eye_in_hand_cam", "agentview_cam", "gsmini_left", "gsmini_right"):
            if hasattr(env_cfg.scene, name):
                delattr(env_cfg.scene, name)
        pol = getattr(env_cfg.observations, "policy", None)
        if pol is not None and hasattr(pol, "gripper_marker_motion"):
            delattr(pol, "gripper_marker_motion")
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        print(
            f"env ready dt={dt} task={TASK_ID} obj={OBJ_NAME} instruction={INSTRUCTION!r} "
            f"n_seeds={N_SEEDS} seed0={SEED0} forces={FORCES} mus={MUS} cell_spec={CELL_SPEC!r}",
            flush=True,
        )
        res_path = OUT / CSV_NAME
        results, done = [], set()
        if res_path.exists() and res_path.stat().st_size > 0:
            with res_path.open() as f:
                for row in csv.DictReader(f):
                    done.add(row["trial_id"])
                    results.append(row)
        print("resume", len(done), flush=True)
        for F, s, mu, tid in plan_jobs():
            if tid in done:
                print("skip", tid, flush=True)
                continue
            print("===", tid, "===", flush=True)
            rec = run_episode(env, seed_idx=s, mu=mu, trial_id=tid, f_cmd=F, dt=dt)
            results.append(rec)
            _rewrite(res_path, results, list(rec.keys()))
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
        (OUT / f"SWEEP_ERROR_task{TASK_ID}.json").write_text(
            json.dumps({"error": repr(e), "trace": traceback.format_exc()}, indent=2)
        )
        try:
            simulation_app.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
