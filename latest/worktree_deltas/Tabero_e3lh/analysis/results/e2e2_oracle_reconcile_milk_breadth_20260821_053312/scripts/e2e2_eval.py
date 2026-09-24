#!/usr/bin/env python3
"""E2E-2 isolated evaluator.

All runs share the same scripted full-task arm policy.  This copy adds a
force-cell scan mode for canonical F* reconciliation while preserving the
frozen D2 provisional rule unchanged. METHOD_CHANGE=NONE.
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
OUT = Path(os.environ.get("E2E_OUT", Path(__file__).resolve().parents[1]))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

LOG_NAME = os.environ.get("E2E_LOG_NAME", "e2e_eval_isaac.log")
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
TASK_ID = int(os.environ.get("E2E_TASK_ID", "1"))
BASKET_NAME = "basket_1"

TASK_OBJECTS = {
    0: "alphabet_soup_1", 1: "cream_cheese_1", 2: "salad_dressing_1",
    3: "bbq_sauce_1", 5: "tomato_sauce_1", 6: "butter_1",
    7: "milk_1", 8: "chocolate_pudding_1", 9: "orange_juice_1",
}

OBJ_NAME = TASK_OBJECTS[TASK_ID]
N_SEEDS = int(os.environ.get("E2E_N_SEEDS", "3"))
SEED0 = int(os.environ.get("E2E_SEED0", "50"))
MUS = [float(x) for x in os.environ.get("E2E_MUS", "0.2,0.5,1.0").split(",")]
METHODS = [m.strip() for m in os.environ.get("E2E_METHODS", "fixed_low,fixed_robust,oracle,provisional_ours").split(",")]
MODE = os.environ.get("E2E_MODE", "methods")
FORCES = [float(x) for x in os.environ.get("E2E_FORCES", "3,4,5,6,8").split(",") if x.strip()]

# Scripted trajectory params (same as B2/D2)
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

# Probe params (frozen D2)
AMP_M = 0.002
PROBE_HALF = 10
POST_HOLD = 5

# --- Task-specific configs (loaded from B2/D2) ---

def load_task_config():
    """Load oracle force mapping and fixed robust force for current task."""
    cfg_path = OUT / "BENCHMARK_TASKS.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        task_key = str(TASK_ID)
        if task_key in cfg.get("tasks", {}):
            return cfg["tasks"][task_key]
    # Fallback: task 1 known values
    if TASK_ID == 1:
        return {
            "fstar": {"0.2": 6.0, "0.5": 5.0, "1.0": 4.0},
            "fixed_robust": 6.0,
            "fixed_low": 3.0,
            "force_candidates": [3, 4, 5, 6],
        }
    if TASK_ID == 7:
        return {
            "fstar": {"0.2": 3.0, "0.5": 3.0, "1.0": 3.0},
            "fixed_robust": 3.0,
            "fixed_low": 3.0,
            "force_candidates": [3, 4, 5, 6, 8],
        }
    return None


def load_hier_thresholds():
    d2 = Path("/home/exouser/Tabero/analysis/results/d2_hierarchical_force_decision_20260820_054605")
    p = d2 / "CALIBRATION_THRESHOLDS.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"theta_low": 0.724223, "theta_mid_high": 0.011378}


def load_deligrasp_cache():
    p = OUT / "DELIGRASP_CACHE.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


# --- Force decision functions ---

def decide_fixed_low(task_cfg, mu):
    return float(task_cfg.get("fixed_low", 3.0)), "fixed_low", {}

def decide_fixed_robust(task_cfg, mu):
    return float(task_cfg.get("fixed_robust", 6.0)), "fixed_robust", {}

def decide_oracle(task_cfg, mu):
    fstar = task_cfg["fstar"]
    F = float(fstar.get(str(mu), fstar.get(f"{mu:g}", 6.0)))
    return F, "oracle", {}

def decide_deligrasp(task_cfg, mu, obj_name, cache):
    key = f"{obj_name}"
    if key in cache:
        F = float(cache[key].get("force", task_cfg.get("fixed_robust", 6.0)))
        return F, "deligrasp_semantic", cache[key]
    return float(task_cfg.get("fixed_robust", 6.0)), "deligrasp_no_cache", {}

def decide_hierarchical(z_imb, z_hyst, hier_th):
    th_low = float(hier_th["theta_low"])
    th_hyst = float(hier_th["theta_mid_high"])
    if z_imb <= th_low:
        return 6.0, "stage1_low_veto"
    if z_hyst > th_hyst:
        return 4.0, "stage2_high"
    return 5.0, "stage2_mid"

def decide_forte_gt(mu, task_cfg):
    """FORTE-inspired: start at lowest force, step up if slip detected (GT)."""
    fstar = task_cfg["fstar"]
    target = float(fstar.get(str(mu), fstar.get(f"{mu:g}", 6.0)))
    candidates = sorted(task_cfg.get("force_candidates", [3, 4, 5, 6]))
    return candidates[0], "forte_start_low", {"target": target, "candidates": candidates}


# --- Utilities ---

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


# --- Main episode runner ---

def run_episode(env, *, seed_idx, mu, trial_id, method, dt, task_cfg, hier_th, deligrasp_cache, force_override=None):
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
    obj0_b, _ = _pose_in_base(env, OBJ_NAME)
    basket_b, _ = _pose_in_base(env, BASKET_NAME)
    pregrasp = obj0_b.copy(); pregrasp[2] += PREGRASP_Z
    grasp = obj0_b.copy(); grasp[2] += GRASP_Z
    lift = grasp.copy(); lift[2] += LIFT_Z
    transit = basket_b.copy(); transit[2] = max(lift[2], basket_b[2] + PLACE_CLEAR_Z)
    place = basket_b.copy(); place[2] = basket_b[2] + PLACE_Z

    d_pred = D_OPEN
    cmd_pos = ee_pos.copy()
    want_probe = method in ("provisional_ours",)
    f_cmd = 4.0  # initial grasp force for probe

    step = 0
    dropped = lift_success = pick_success = 0
    lost_in_transit = timed_out = 0
    max_basket = 0.0
    peak_force = 0.0
    integrated_force = 0.0
    term_reason = ""
    decided = 0
    chosen_F = f_cmd
    decision_rule = ""
    decision_info = {}
    z_imb_peak = np.nan
    z_ftan_hyst = np.nan
    imb_hist = []
    ftan_out, ftan_back = [], []
    path_mm = 0.0
    last_cmd = ee_pos.copy()
    t_probe0 = t_probe1 = None

    # FORTE reactive state
    forte_info = {}
    forte_slip_detected = False
    forte_escalations = 0

    # Pre-decide for non-probe methods.
    if force_override is not None:
        chosen_F = float(force_override)
        decision_rule = f"force_cell_{chosen_F:g}N"
        decision_info = {}
        decided = 1
    elif method == "fixed_low":
        chosen_F, decision_rule, decision_info = decide_fixed_low(task_cfg, mu)
        decided = 1
    elif method == "fixed_robust":
        chosen_F, decision_rule, decision_info = decide_fixed_robust(task_cfg, mu)
        decided = 1
    elif method == "oracle":
        chosen_F, decision_rule, decision_info = decide_oracle(task_cfg, mu)
        decided = 1
    elif method == "deligrasp_style":
        chosen_F, decision_rule, decision_info = decide_deligrasp(task_cfg, mu, OBJ_NAME, deligrasp_cache)
        decided = 1
    elif method == "forte_gt":
        chosen_F, decision_rule, forte_info = decide_forte_gt(mu, task_cfg)
        f_cmd = chosen_F
        decided = 0  # will escalate reactively

    if decided:
        f_cmd = chosen_F

    def phases():
        yield "approach", APPROACH_STEPS, pregrasp, "open", None
        yield "descend", DESCEND_STEPS, grasp, "open", None
        yield "close", CLOSE_STEPS, grasp, "track", None
        yield "hold", HOLD_STEPS, grasp, "track", None
        if want_probe:
            g = grasp.copy()
            tgt = g.copy(); tgt[1] += AMP_M
            yield "probe_out", PROBE_HALF, tgt, "track", "probe"
            yield "probe_back", PROBE_HALF, g, "track", "probe"
            yield "probe_hold", POST_HOLD, g, "track", "probe"
        yield "lift", LIFT_STEPS, lift, "track", None
        yield "transit", TRANSIT_STEPS, transit, "freeze", None
        yield "over_basket", OVER_BASKET_STEPS, transit, "freeze", None
        yield "place", PLACE_STEPS, place, "freeze", None
        yield "release", RELEASE_STEPS, place, "open", None
        yield "settle", SETTLE_PLACE_STEPS, place, "open", None

    for phase, n_steps, target, mode, tag in phases():
        start = cmd_pos.copy()
        # Hierarchical decision at lift
        if phase == "lift" and method == "provisional_ours" and not decided:
            z_imb_peak = float(np.max(imb_hist)) if imb_hist else 0.0
            if ftan_out and ftan_back:
                n = min(len(ftan_out), len(ftan_back))
                z_ftan_hyst = float(np.mean(np.abs(np.array(ftan_out[:n]) - np.array(ftan_back[:n][::-1]))))
            else:
                z_ftan_hyst = 0.0
            chosen_F, decision_rule = decide_hierarchical(z_imb_peak, z_ftan_hyst, hier_th)
            f_cmd = chosen_F
            decided = 1

        # FORTE: finalize decided after lift completes (force may keep escalating)
        if phase == "transit" and method == "forte_gt" and not decided:
            chosen_F = f_cmd
            decided = 1

        for i in range(n_steps):
            cmd_pos = _interp(start, target, i, n_steps)
            if tag == "probe":
                path_mm += float(np.linalg.norm(cmd_pos - last_cmd) * 1000.0)
            last_cmd = cmd_pos.copy()
            if mode == "open":
                d_pred = D_OPEN
            action = _make_action(cmd_pos, eef_aa, d_pred, f_cmd if mode != "open" else 0.0, env.device)
            obs, rew, term, trunc, info = env.step(action)
            step += 1
            f = obs["policy"]["gripper_net_force"][0]
            if f.ndim == 3:
                f = f[-1]
            fL = f[0].detach().cpu().numpy()
            fR = f[1].detach().cpu().numpy()
            f_sq = _f(_dbg(env).get("f_sq_meas"), 0.0)
            peak_force = max(peak_force, f_sq)
            integrated_force += f_sq * dt
            if mode == "track":
                d_pred = force_servo(d_pred, f_sq, f_cmd)
            imb = float(abs(np.linalg.norm(fL) - np.linalg.norm(fR)))
            ftan = float(np.linalg.norm(np.concatenate([fL[:2], fR[:2]])))

            if tag == "probe":
                if t_probe0 is None:
                    t_probe0 = step * dt
                t_probe1 = step * dt
                imb_hist.append(imb)
                if phase == "probe_out":
                    ftan_out.append(ftan)
                elif phase == "probe_back":
                    ftan_back.append(ftan)

            # FORTE GT reactive: detect slip during lift, escalate repeatedly
            if method == "forte_gt" and phase == "lift":
                obj_p_forte = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
                obj_dz_forte = float(obj_p_forte[2] - obj0_w[2])
                # slip: object not rising with gripper
                if obj_dz_forte < 0.01 and i > 5:
                    candidates = forte_info.get("candidates", [3, 4, 5, 6])
                    cur_idx = candidates.index(int(f_cmd)) if int(f_cmd) in candidates else len(candidates)-1
                    if cur_idx < len(candidates) - 1:
                        f_cmd = float(candidates[cur_idx + 1])
                        forte_escalations += 1
                        chosen_F = f_cmd

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
    fstar = task_cfg["fstar"]
    fs = float(fstar.get(str(mu), fstar.get(f"{mu:g}", 6.0)))

    rec = {
        "trial_id": trial_id, "task_id": TASK_ID, "object": OBJ_NAME,
        "method": method, "info_source": _info_source(method),
        "seed_idx": seed_idx, "friction": mu, "friction_applied": applied,
        "chosen_force": chosen_F, "fstar": fs,
        "decision_rule": decision_rule,
        "exact_force": int(abs(chosen_F - fs) < 1e-6),
        "under_force": int(chosen_F < fs - 1e-6),
        "over_force": int(chosen_F > fs + 1e-6),
        "pick_success": pick_success, "lift_success": lift_success,
        "transport_retention": int(lift_success == 1 and lost_in_transit == 0),
        "place_success": int(max_basket > 0.05),
        "full_task_success": honest_full,
        "basket_contact_max": max_basket,
        "dropped": dropped, "timeout": timed_out, "term_reason": term_reason,
        "lost_in_transit": lost_in_transit,
        "peak_force": peak_force, "mean_force": integrated_force / max(step * dt, 1e-6),
        "integrated_force": integrated_force,
        "steps": step, "t_episode_s": step * dt,
        "probe_used": int(want_probe),
        "probe_duration_s": (t_probe1 - t_probe0 + dt) if t_probe0 is not None else 0.0,
        "probe_path_mm": path_mm,
        "z_imb_peak": z_imb_peak, "z_ftan_hyst": z_ftan_hyst,
        "forte_escalations": forte_escalations,
    }
    print(
        f"summary {trial_id} method={method} mu={mu} F={chosen_F} "
        f"full={honest_full} rule={decision_rule}",
        flush=True,
    )
    return rec


def _info_source(method):
    m = {
        "fixed_low": "none", "fixed_robust": "none", "oracle": "GT physics",
        "deligrasp_style": "semantic prior", "forte_gt": "post-slip",
        "provisional_ours": "pre-lift probe", "tabero_neutral": "language",
        "force_cell": "controlled force scan",
    }
    return m.get(method, "unknown")


def _rewrite(path, dicts, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in dicts:
            w.writerow(d)


def _parse_cell_spec():
    spec = os.environ.get("E2E_CELL_SPEC", "").strip()
    if not spec:
        return {mu: list(FORCES) for mu in MUS}
    cells = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        mu_s, forces_s = part.split(":", 1)
        mu = float(mu_s.strip())
        cells[mu] = [float(x) for x in forces_s.split(",") if x.strip()]
    return cells


def plan_jobs():
    jobs = []
    if MODE == "force_cells":
        cells = _parse_cell_spec()
        for s in range(SEED0, SEED0 + N_SEEDS):
            for mu in MUS:
                for F in cells.get(mu, []):
                    tid = f"e2e2_t{TASK_ID}_F{F:g}_s{s}_mu{mu:g}"
                    jobs.append(("force_cell", s, mu, tid, F))
        return jobs
    for method in METHODS:
        for s in range(SEED0, SEED0 + N_SEEDS):
            for mu in MUS:
                tid = f"e2e_t{TASK_ID}_{method}_s{s}_mu{mu:g}"
                jobs.append((method, s, mu, tid, None))
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
        env_cfg.episode_length_s = 45.0
        for name in ("eye_in_hand_cam", "agentview_cam", "gsmini_left", "gsmini_right"):
            if hasattr(env_cfg.scene, name):
                delattr(env_cfg.scene, name)
        pol = getattr(env_cfg.observations, "policy", None)
        if pol is not None and hasattr(pol, "gripper_marker_motion"):
            delattr(pol, "gripper_marker_motion")
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        print(f"env ready dt={dt} task={TASK_ID} obj={OBJ_NAME} mode={MODE} methods={METHODS} n_seeds={N_SEEDS}", flush=True)

        task_cfg = load_task_config()
        if task_cfg is None:
            print("ERROR: no task config available", flush=True)
            os._exit(1)
        hier_th = load_hier_thresholds()
        deligrasp_cache = load_deligrasp_cache()

        csv_name = os.environ.get("E2E_CSV_NAME")
        if not csv_name:
            csv_name = f"TASK{TASK_ID}_FORCE_SCAN.csv" if MODE == "force_cells" else f"E2E_TASK{TASK_ID}_RESULTS.csv"
        res_path = OUT / csv_name
        results, done = [], set()
        if res_path.exists() and res_path.stat().st_size > 0:
            with res_path.open() as f:
                for row in csv.DictReader(f):
                    done.add(row["trial_id"])
                    results.append(row)
        print("resume", len(done), flush=True)

        for method, s, mu, tid, force_override in plan_jobs():
            if tid in done:
                print("skip", tid, flush=True)
                continue
            print("===", tid, "===", flush=True)
            rec = run_episode(
                env, seed_idx=s, mu=mu, trial_id=tid, method=method, dt=dt,
                task_cfg=task_cfg, hier_th=hier_th, deligrasp_cache=deligrasp_cache,
                force_override=force_override,
            )
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
        (OUT / "SWEEP_ERROR.json").write_text(json.dumps({"error": repr(e), "trace": traceback.format_exc()}, indent=2))
        try:
            simulation_app.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
