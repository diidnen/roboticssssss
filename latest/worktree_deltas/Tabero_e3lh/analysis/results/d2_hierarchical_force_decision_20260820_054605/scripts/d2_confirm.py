#!/usr/bin/env python3
"""D2 confirmation: frozen Probe A + hierarchical 6N veto vs baselines. METHOD_CHANGE=NONE."""
from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

TABERO = Path("/home/exouser/Tabero")
OUT = Path(os.environ.get("D2_OUT", Path(__file__).resolve().parents[1]))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

sys.stdout = open(OUT / "logs" / "d2_confirm_isaac.log", "w", buffering=1)
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

MODE = "confirm"
N_SEEDS = int(os.environ.get("D2_N_SEEDS", "20"))
SEED0 = int(os.environ.get("D2_SEED0", "30"))
MUS = [0.2, 0.5, 1.0]
METHODS = [m.strip() for m in os.environ.get("D2_METHODS", "fixed6,current_p3,hierarchical,oracle").split(",") if m.strip()]
AMP_M = 0.002
PROBE_HALF = 10
POST_HOLD = 5

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


def _tactile_summaries(env, obs):
    out = {
        "marker_mean": 0.0, "marker_tangential": 0.0, "marker_asym": 0.0, "tactile_ok": 0,
    }
    try:
        mm = obs["policy"]["gripper_marker_motion"][0].detach().cpu().numpy()
        cur, init = mm[:, 1], mm[:, 0]
        disp = cur - init
        mag = np.linalg.norm(disp, axis=-1)
        out["marker_mean"] = float(mag.mean())
        left = mag[0].mean() if mag.shape[0] > 0 else 0.0
        right = mag[1].mean() if mag.shape[0] > 1 else 0.0
        out["marker_asym"] = float(abs(left - right))
        out["marker_tangential"] = float(np.linalg.norm(disp.mean(axis=1), axis=-1).mean())
        out["tactile_ok"] = 1
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


def _rewrite(path: Path, dicts, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in dicts:
            w.writerow(d)


def load_json(name, default=None):
    p = OUT / name
    if not p.exists():
        return default
    return json.loads(p.read_text())


def _pdf(z, m, s):
    s = max(float(s), 1e-4)
    return float(np.exp(-0.5 * ((z - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)))


def decide_hierarchical(z_imb, z_hyst, th_low, th_hyst):
    if z_imb <= th_low:
        return 6.0, "stage1_low_veto"
    if z_hyst > th_hyst:
        return 4.0, "stage2_high"
    return 5.0, "stage2_mid"


def decide_nb(zs, model, names):
    mus = [0.2, 0.5, 1.0]
    prior = {float(k): float(v) for k, v in model["prior"].items()}
    unnorm = {}
    for mu in mus:
        like = 1.0
        for name in names:
            m = float(model["means"][name][str(mu)])
            s = float(model["stds"][name][str(mu)])
            like *= _pdf(zs[name], m, s)
        unnorm[mu] = like * prior.get(mu, 1 / 3)
    zsum = sum(unnorm.values()) + 1e-18
    post = {mu: unnorm[mu] / zsum for mu in mus}
    table = model["success_table"]
    forces = [float(x) for x in model.get("force_candidates", [4, 5, 6])]
    tau = float(model.get("tau", 0.8))
    p_succ = {}
    for F in forces:
        key = str(int(F))
        row = table.get(key) or {}
        p_succ[F] = sum(post[mu] * float(row.get(str(mu), 0.0)) for mu in mus)
    feasible = [F for F in sorted(forces) if p_succ[F] >= tau]
    Fstar = min(feasible) if feasible else max(forces)
    return Fstar, post, p_succ


def fstar_full(mu, mapping):
    return float(mapping[str(mu)] if str(mu) in mapping else mapping[f"{mu:g}"])


def snapshot(tag, obj_p, obj_q, ee, d_pred, f_sq, imb, fL, fR):
    return {
        "tag": tag,
        "obj_x": float(obj_p[0]), "obj_y": float(obj_p[1]), "obj_z": float(obj_p[2]),
        "obj_qw": float(obj_q[0]), "obj_qx": float(obj_q[1]), "obj_qy": float(obj_q[2]), "obj_qz": float(obj_q[3]),
        "eef_x": float(ee[0]), "eef_y": float(ee[1]), "eef_z": float(ee[2]),
        "d_pred": float(d_pred), "f_meas": float(f_sq), "imbalance": float(imb),
        "fL_n": float(np.linalg.norm(fL)), "fR_n": float(np.linalg.norm(fR)),
    }


def run_episode(env, *, seed_idx, mu, trial_id, method, f_fixed, dt, tactile, belief_model, fstar_map, hier_th):
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
    basket_b, _ = _pose_in_base(env, BASKET_NAME)
    pregrasp = obj0_b.copy(); pregrasp[2] += PREGRASP_Z
    grasp = obj0_b.copy(); grasp[2] += GRASP_Z
    lift = grasp.copy(); lift[2] += LIFT_Z
    transit = basket_b.copy(); transit[2] = max(lift[2], basket_b[2] + PLACE_CLEAR_Z)
    place = basket_b.copy(); place[2] = basket_b[2] + PLACE_Z

    d_pred = D_OPEN
    cmd_pos = ee_pos.copy()
    want_probe = True  # all confirmation methods use frozen Probe A
    f_cmd = 4.0

    step = 0
    grasp_rel = None
    dropped = lift_success = pick_success = 0
    lost_in_transit = timed_out = 0
    max_basket = 0.0
    peak_force = 0.0
    integrated_force = 0.0
    t_probe0 = t_probe1 = None
    probe_fail = contact_lost_probe = 0
    obj_disp_probe = 0.0
    obj_at_probe0 = None
    path_mm = 0.0
    last_cmd = cmd_pos.copy()
    f_hist, imb_hist, mk_tang = [], [], []
    ftan_out, ftan_back = [], []
    snaps = {}
    chosen_F = f_cmd
    post = {}
    p_succ = {}
    z_used = np.nan
    decision_rule = ""
    decided = 0
    term_reason = ""

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
        if phase == "lift" and not decided:
            z_imb = float(np.max(imb_hist)) if imb_hist else 0.0
            z_hyst = np.nan
            if ftan_out and ftan_back:
                n = min(len(ftan_out), len(ftan_back))
                z_hyst = float(np.mean(np.abs(np.array(ftan_out[:n]) - np.array(ftan_back[:n][::-1]))))
            z_used = z_imb
            if method == "fixed6":
                chosen_F = 6.0
            elif method == "oracle":
                chosen_F = fstar_full(mu, fstar_map)
            elif method == "current_p3" and belief_model is not None:
                chosen_F, post, p_succ = decide_nb({"imb_peak": z_imb}, belief_model, ["imb_peak"])
            elif method == "hierarchical" and hier_th is not None:
                th_low = float(hier_th["theta_low"])
                th_hyst = float(hier_th["theta_mid_high"])
                chosen_F, decision_rule = decide_hierarchical(z_imb, z_hyst, th_low, th_hyst)
                z_used = z_hyst
            else:
                chosen_F = 6.0
            f_cmd = chosen_F
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
            tac = _tactile_summaries(env, obs) if tactile else {"marker_mean": 0.0, "marker_tangential": 0.0, "marker_asym": 0.0, "tactile_ok": 0}
            obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
            obj_q = env.scene[OBJ_NAME].data.root_quat_w[0].detach().cpu().numpy()
            ee = obs["policy"]["eef_pose"][0].detach().cpu().numpy()[:3]
            rel = obj_p - ee
            if phase == "close" and i == n_steps - 1:
                grasp_rel = rel.copy()
            try:
                dropped = int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
            except Exception:
                pass
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
            if tag == "probe":
                if t_probe0 is None:
                    t_probe0 = step * dt
                t_probe1 = step * dt
                f_hist.append(f_sq); imb_hist.append(imb); mk_tang.append(tac["marker_tangential"])
                if phase == "probe_out":
                    ftan_out.append(ftan)
                elif phase == "probe_back":
                    ftan_back.append(ftan)
                if obj_at_probe0 is None:
                    obj_at_probe0 = obj_p.copy()
                obj_disp_probe = max(obj_disp_probe, float(np.linalg.norm(obj_p - obj_at_probe0)))
                if (np.abs(fL).sum() + np.abs(fR).sum()) <= 1e-3 or dropped:
                    contact_lost_probe = 1
                    probe_fail = 1
            last_end = i == n_steps - 1
            if last_end and phase == "hold":
                snaps["before_probe"] = snapshot("before_probe", obj_p, obj_q, ee, d_pred, f_sq, imb, fL, fR)
            if last_end and phase == "probe_hold":
                snaps["after_probe"] = snapshot("after_probe", obj_p, obj_q, ee, d_pred, f_sq, imb, fL, fR)
            if phase == "lift" and i == 0:
                snaps["before_lift"] = snapshot("before_lift", obj_p, obj_q, ee, d_pred, f_sq, imb, fL, fR)
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
    bp = snaps.get("before_probe") or {}
    ap = snaps.get("after_probe") or {}
    pose_shift = np.nan
    rot_shift = np.nan
    if bp and ap:
        pose_shift = float(np.linalg.norm([ap["obj_x"] - bp["obj_x"], ap["obj_y"] - bp["obj_y"], ap["obj_z"] - bp["obj_z"]]))
        # quaternion angle
        q0 = np.array([bp["obj_qw"], bp["obj_qx"], bp["obj_qy"], bp["obj_qz"]])
        q1 = np.array([ap["obj_qw"], ap["obj_qx"], ap["obj_qy"], ap["obj_qz"]])
        rot_shift = float(2 * np.arccos(np.clip(abs(np.dot(q0, q1)), 0, 1)))
    rec = {
        "trial_id": trial_id, "mode": MODE, "method": method,
        "seed_idx": seed_idx, "friction": mu, "friction_applied": applied,
        "f_cmd_initial": 4.0 if method == "probe_belief" else f_cmd,
        "f_cmd_final": f_cmd, "chosen_force": chosen_F,
        "z_imbalance_peak": float(np.max(imb_hist)) if imb_hist else np.nan,
        "z_ftan_hyst": float(np.mean(np.abs(np.array(ftan_out[:min(len(ftan_out), len(ftan_back))]) - np.array(ftan_back[:min(len(ftan_out), len(ftan_back))][::-1])))) if (ftan_out and ftan_back) else np.nan,
        "z_used": z_used, "decision_rule": decision_rule, "tactile_ok": int(tactile),
        "probe": "A" if want_probe else "",
        "probe_duration_s": None if t_probe0 is None else (t_probe1 - t_probe0 + dt),
        "probe_path_mm": path_mm if want_probe else 0.0,
        "probe_failure": probe_fail, "contact_lost_probe": contact_lost_probe,
        "obj_disp_probe_m": obj_disp_probe,
        "pose_shift_m": pose_shift, "rot_shift_rad": rot_shift,
        "d_pred_before_probe": bp.get("d_pred", ""), "d_pred_after_probe": ap.get("d_pred", ""),
        "imb_before_probe": bp.get("imbalance", ""), "imb_after_probe": ap.get("imbalance", ""),
        "f_before_probe": bp.get("f_meas", ""), "f_after_probe": ap.get("f_meas", ""),
        "posterior_mu02": post.get(0.2, ""), "posterior_mu05": post.get(0.5, ""), "posterior_mu10": post.get(1.0, ""),
        "p_succ_chosen": p_succ.get(chosen_F, "") if p_succ else "",
        "pick_success": pick_success, "lift_success": lift_success,
        "transport_retention": int(lift_success == 1 and lost_in_transit == 0),
        "place_success": int(max_basket > 0.05),
        "full_task_success": honest_full, "basket_contact_max": max_basket,
        "dropped": dropped, "timeout": timed_out, "term_reason": term_reason,
        "lost_in_transit": lost_in_transit,
        "peak_force": peak_force, "mean_force": integrated_force / max(step * dt, 1e-6),
        "integrated_force": integrated_force, "steps": step,
        "t_episode_s": step * dt,
        "t_lift_start": (APPROACH_STEPS + DESCEND_STEPS + CLOSE_STEPS + HOLD_STEPS + (2 * PROBE_HALF + POST_HOLD if want_probe else 0)) * dt,
    }
    print(
        f"summary {trial_id} method={method} mu={mu} lift={lift_success} full={honest_full} "
        f"F={rec['chosen_force']} timeout={timed_out} pfail={probe_fail}",
        flush=True,
    )
    dist_row = None
    if want_probe and bp and ap:
        dist_row = {
            "trial_id": trial_id, "method": method, "seed_idx": seed_idx, "friction": mu,
            "full_task_success": honest_full, "chosen_force": chosen_F,
            **{f"bp_{k}": v for k, v in bp.items()},
            **{f"ap_{k}": v for k, v in ap.items()},
            "pose_shift_m": pose_shift, "rot_shift_rad": rot_shift,
        }
        bl = snaps.get("before_lift") or {}
        for k, v in bl.items():
            dist_row[f"bl_{k}"] = v
    return rec, dist_row


def plan_jobs():
    jobs = []
    for method in METHODS:
        for k in range(N_SEEDS):
            s = SEED0 + k
            for mu in MUS:
                jobs.append((method, None, s, mu, f"d2_{method}_s{s}_mu{mu:g}"))
    return jobs


def main():
    from isaaclab.app import AppLauncher

    enable_cameras = os.environ.get("D2_ENABLE_CAMERAS", "0") not in ("0", "false", "False")
    app_launcher = AppLauncher(headless=True, enable_cameras=enable_cameras, num_envs=1)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(TASK_SUITE, TASK_ID)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = float(os.environ.get("D2_EPISODE_S", "45"))
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
        print("env ready dt", dt, "mode", MODE, "tactile", tactile, "n_seeds", N_SEEDS, flush=True)
        fstar_map = {"0.2": 6.0, "0.5": 5.0, "1.0": 4.0}
        belief = load_json("BELIEF_MODEL.json")
        hier_th = load_json("CALIBRATION_THRESHOLDS.json")
        csv_name = "FULLTASK_CONFIRMATION.csv"
        res_path = OUT / csv_name
        results, done = [], set()
        if res_path.exists() and res_path.stat().st_size > 0:
            with res_path.open() as f:
                for row in csv.DictReader(f):
                    done.add(row["trial_id"])
                    results.append(row)
        print("resume", len(done), flush=True)
        dist_rows = []
        for method, F, s, mu, tid in plan_jobs():
            if tid in done:
                print("skip", tid, flush=True)
                continue
            print("===", tid, "===", flush=True)
            rec, dist = run_episode(
                env, seed_idx=s, mu=mu, trial_id=tid, method=method, f_fixed=F,
                dt=dt, tactile=tactile, belief_model=belief, fstar_map=fstar_map, hier_th=hier_th,
            )
            results.append(rec)
            _rewrite(res_path, results, list(rec.keys()))
            if dist:
                dist_rows.append(dist)
                _append(OUT / "PROBE_DISTURBANCE.csv", [dist], list(dist.keys()))
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
