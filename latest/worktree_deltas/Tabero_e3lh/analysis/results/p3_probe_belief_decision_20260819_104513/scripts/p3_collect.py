#!/usr/bin/env python3
"""P3: pre-lift probe + discrete Bayes 4/6 N decision. METHOD_CHANGE=NONE.

Not a learned method. Always-probe PoC. Same probe for every μ.
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
OUT = Path(os.environ.get("P3_OUT", Path(__file__).resolve().parents[1]))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

sys.stdout = open(OUT / "logs" / "p3_isaac.log", "w", buffering=1)
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

MODE = os.environ.get("P3_MODE", "screen")  # screen | eval
N_SEEDS = int(os.environ.get("P3_N_SEEDS", "10"))
MUS = [0.2, 0.5, 1.0]
PROBES = os.environ.get("P3_PROBES", "A,B").split(",")
METHODS = os.environ.get("P3_METHODS", "fixed4,fixed6,oracle,probe_belief").split(",")
CHOSEN_PROBE = os.environ.get("P3_CHOSEN_PROBE", "A")
AMP_M = float(os.environ.get("P3_AMP_MM", "2")) / 1000.0
PROBE_HALF = int(os.environ.get("P3_PROBE_HALF_STEPS", "10"))
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
F4, F6 = 4.0, 6.0


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
        "marker_tangential": 0.0, "hm_mean": 0.0, "hm_max": 0.0, "hm_std": 0.0,
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
        out["marker_tangential"] = float(np.linalg.norm(disp.mean(axis=1), axis=-1).mean())
        out["tactile_ok"] = 1
    except Exception:
        pass
    try:
        for side in ("gsmini_left", "gsmini_right"):
            sen = env.scene[side]
            hm = sen.data.output.get("height_map")
            if hm is None:
                continue
            arr = hm[0].detach().cpu().numpy()
            if out["hm_mean"] == 0.0:
                out["hm_mean"] = float(arr.mean())
                out["hm_max"] = float(arr.max())
                out["hm_std"] = float(arr.std())
            else:
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


def _rewrite(path: Path, dicts, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in dicts:
            w.writerow(d)


def load_belief():
    p = OUT / "BELIEF_MODEL.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def decide_force(z, model, rule="principled"):
    """Return F, posterior dict, P_succ_4, P_succ_6."""
    mus = [0.2, 0.5, 1.0]
    prior = {float(k): float(v) for k, v in model["prior"].items()}
    means = {float(k): float(v) for k, v in model["means"].items()}
    stds = {float(k): max(float(v), 1e-4) for k, v in model["stds"].items()}
    like = {}
    for mu in mus:
        s = stds[mu]
        like[mu] = float(np.exp(-0.5 * ((z - means[mu]) / s) ** 2) / (s * np.sqrt(2 * np.pi)))
    unnorm = {mu: like[mu] * prior.get(mu, 1 / 3) for mu in mus}
    zsum = sum(unnorm.values()) + 1e-18
    post = {mu: unnorm[mu] / zsum for mu in mus}
    p_succ = {
        4.0: post[0.2] * 0.0 + post[0.5] * 1.0 + post[1.0] * 1.0,
        6.0: 1.0,
    }
    tau = float(model.get("tau", 0.8))
    if rule == "theta":
        th = float(model.get("theta", 0.5))
        F = 6.0 if post[0.2] > th else 4.0
    else:
        F = 4.0 if p_succ[4.0] >= tau else 6.0
    return F, post, p_succ[4.0], p_succ[6.0]


@dataclass
class Step:
    trial_id: str
    method: str
    probe: str
    seed_idx: int
    friction: float
    phase: str
    step: int
    t_s: float
    f_cmd: float
    f_meas: float
    imbalance: float
    f_tangential: float
    marker_mean: float
    marker_tangential: float
    marker_asym: float
    marker_vel: float
    hm_mean: float
    rel_xy: float
    rel_z: float
    tactile_ok: int


def run_episode(env, *, seed_idx, mu, trial_id, method, probe_name, dt, tactile, do_place, belief_model):
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
    if method == "fixed6":
        f_cmd = F6
    elif method == "oracle":
        f_cmd = F6 if abs(mu - 0.2) < 1e-6 else F4
    else:
        f_cmd = F4  # fixed4, screen, probe_belief until decision

    want_probe = method in ("screen", "probe_belief")
    step = 0
    grasp_rel = None
    prev_rel = None
    prev_marker = None
    dropped = lift_success = pick_success = 0
    lost_in_transit = 0
    max_basket = 0.0
    peak_force = 0.0
    integrated_force = 0.0
    t_probe0 = t_probe1 = None
    probe_fail = 0
    contact_lost_probe = 0
    obj_disp_probe = 0.0
    obj_at_probe0 = None
    path_mm = 0.0
    last_cmd = cmd_pos.copy()
    f_hist, imb_hist, ftan_hist = [], [], []
    mk_mean, mk_tang, mk_asym, mk_vel, hm_hist = [], [], [], [], []
    relxy_h, relz_h = [], []
    hold_marker = None
    chosen_F = f_cmd
    post = {}
    p4 = p6 = np.nan
    z_used = np.nan
    decided = 0
    rows: list[Step] = []

    def phases():
        yield "approach", APPROACH_STEPS, pregrasp, "open", None
        yield "descend", DESCEND_STEPS, grasp, "open", None
        yield "close", CLOSE_STEPS, grasp, "track", None
        yield "hold", HOLD_STEPS, grasp, "track", None
        if want_probe:
            g = grasp.copy()
            if probe_name == "A":
                tgt = g.copy(); tgt[1] += AMP_M
            else:
                tgt = g.copy(); tgt[2] += AMP_M
            yield "probe_out", PROBE_HALF, tgt, "track", "probe"
            yield "probe_back", PROBE_HALF, g, "track", "probe"
            yield "probe_hold", POST_HOLD, g, "track", "probe"
        if method == "screen":
            return
        yield "lift", LIFT_STEPS, lift, "track", None
        if do_place:
            yield "transit", TRANSIT_STEPS, transit, "freeze", None
            yield "over_basket", OVER_BASKET_STEPS, transit, "freeze", None
            yield "place", PLACE_STEPS, place, "freeze", None
            yield "release", RELEASE_STEPS, place, "open", None
            yield "settle", SETTLE_PLACE_STEPS, place, "open", None

    for phase, n_steps, target, mode, tag in phases():
        start = cmd_pos.copy()
        # After probe_hold, choose force for probe_belief before lift
        if phase == "lift" and method == "probe_belief" and not decided:
            sig = (belief_model or {}).get("signal", "z_imbalance_peak")
            peaks = {
                "z_marker_tang_peak": mk_tang,
                "z_marker_tang_mean": mk_tang,
                "z_marker_mean_peak": mk_mean,
                "z_marker_unloading": mk_mean,
                "z_marker_asym_peak": mk_asym,
                "z_marker_vel_abs_peak": mk_vel,
                "z_hm_mean": hm_hist,
                "z_f_meas_mean": f_hist,
                "z_imbalance_peak": imb_hist,
                "z_ftan_peak": ftan_hist,
            }
            xs = peaks.get(sig, imb_hist)
            if sig == "z_marker_tang_mean" or sig == "z_hm_mean" or sig == "z_f_meas_mean":
                z_used = float(np.mean(xs)) if xs else 0.0
            elif sig == "z_marker_unloading":
                z_used = float((hold_marker - np.mean(xs)) if (hold_marker is not None and xs) else 0.0)
            elif sig == "z_marker_vel_abs_peak":
                z_used = float(np.max(np.abs(xs))) if xs else 0.0
            else:
                z_used = float(np.max(xs)) if xs else 0.0
            if belief_model is not None:
                chosen_F, post, p4, p6 = decide_force(z_used, belief_model, rule="principled")
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
            t_s = step * dt
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
            tac = _tactile_summaries(env, obs) if tactile else {
                "marker_mean": 0.0, "marker_max": 0.0, "marker_std": 0.0, "marker_asym": 0.0,
                "marker_tangential": 0.0, "hm_mean": 0.0, "hm_max": 0.0, "hm_std": 0.0, "tactile_ok": 0,
            }
            mvel = 0.0 if prev_marker is None else (tac["marker_mean"] - prev_marker) / dt
            prev_marker = tac["marker_mean"]
            obj_p = env.scene[OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
            ee = obs["policy"]["eef_pose"][0].detach().cpu().numpy()[:3]
            rel = obj_p - ee
            if phase == "close" and i == n_steps - 1:
                grasp_rel = rel.copy()
            rel_d = rel - grasp_rel if grasp_rel is not None else np.zeros(3)
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
                    t_probe0 = t_s
                t_probe1 = t_s
                f_hist.append(f_sq); imb_hist.append(imb); ftan_hist.append(ftan)
                mk_mean.append(tac["marker_mean"]); mk_tang.append(tac["marker_tangential"])
                mk_asym.append(tac["marker_asym"]); mk_vel.append(mvel); hm_hist.append(tac["hm_mean"])
                relxy_h.append(float(np.linalg.norm(rel_d[:2]))); relz_h.append(float(rel_d[2]))
                if obj_at_probe0 is None:
                    obj_at_probe0 = obj_p.copy()
                obj_disp_probe = max(obj_disp_probe, float(np.linalg.norm(obj_p - obj_at_probe0)))
                contact = int((np.abs(fL).sum() + np.abs(fR).sum()) > 1e-3)
                if not contact or dropped:
                    contact_lost_probe = 1
                    probe_fail = 1
            if phase == "hold" and i == n_steps - 1:
                hold_marker = tac["marker_mean"]
            rows.append(
                Step(
                    trial_id=trial_id, method=method, probe=probe_name or "", seed_idx=seed_idx, friction=mu,
                    phase=phase, step=step, t_s=t_s, f_cmd=f_cmd, f_meas=f_sq,
                    imbalance=imb, f_tangential=ftan, marker_mean=tac["marker_mean"],
                    marker_tangential=tac["marker_tangential"], marker_asym=tac["marker_asym"],
                    marker_vel=mvel, hm_mean=tac["hm_mean"],
                    rel_xy=float(np.linalg.norm(rel_d[:2])), rel_z=float(rel_d[2]),
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

    honest_full = int(do_place and lift_success == 1 and max_basket > 0.05)
    rec = {
        "trial_id": trial_id, "mode": MODE, "method": method, "probe": probe_name or "",
        "seed_idx": seed_idx, "friction": mu, "friction_applied": applied,
        "f_cmd_initial": F4 if method != "fixed6" else F6,
        "f_cmd_final": f_cmd, "chosen_force": chosen_F if method == "probe_belief" else f_cmd,
        "z_marker_tang_peak": peak(mk_tang),
        "z_marker_tang_mean": mean(mk_tang),
        "z_marker_mean_peak": peak(mk_mean),
        "z_marker_unloading": (hold_marker - mean(mk_mean)) if (hold_marker is not None and mk_mean) else np.nan,
        "z_marker_asym_peak": peak(mk_asym),
        "z_marker_vel_abs_peak": float(np.max(np.abs(mk_vel))) if mk_vel else np.nan,
        "z_hm_mean": mean(hm_hist),
        "z_f_meas_mean": mean(f_hist),
        "z_imbalance_peak": peak(imb_hist),
        "z_ftan_peak": peak(ftan_hist),
        "z_rel_xy_peak": peak(relxy_h),
        "z_rel_z_min": float(np.min(relz_h)) if relz_h else np.nan,
        "tactile_ok": int(any(r.tactile_ok for r in rows)),
        "probe_duration_s": None if t_probe0 is None else (t_probe1 - t_probe0 + dt),
        "probe_amp_mm": AMP_M * 1000.0 if want_probe else 0.0,
        "probe_path_mm": path_mm if want_probe else 0.0,
        "probe_failure": probe_fail,
        "contact_lost_probe": contact_lost_probe,
        "obj_disp_probe_m": obj_disp_probe,
        "posterior_mu02": post.get(0.2, ""),
        "posterior_mu05": post.get(0.5, ""),
        "posterior_mu10": post.get(1.0, ""),
        "p_succ_4N": p4, "p_succ_6N": p6,
        "z_used": z_used,
        "pick_success": pick_success, "lift_success": lift_success,
        "full_task_success": honest_full, "basket_contact_max": max_basket,
        "dropped": dropped, "peak_force": peak_force, "mean_force": integrated_force / max(step * dt, 1e-6),
        "integrated_force": integrated_force, "steps": step,
        "t_probe0": t_probe0, "t_lift_start": (APPROACH_STEPS + DESCEND_STEPS + CLOSE_STEPS + HOLD_STEPS + (2 * PROBE_HALF + POST_HOLD if want_probe else 0)) * dt,
    }
    print(
        f"summary {trial_id} method={method} mu={mu} lift={lift_success} full={honest_full} "
        f"F={rec['chosen_force']} z_tang={rec['z_marker_tang_peak']} pfail={probe_fail} tac={rec['tactile_ok']}",
        flush=True,
    )
    return rows, rec


RESULT_FIELDS = None  # filled from first rec


def plan_jobs():
    jobs = []
    if MODE == "screen":
        for probe in PROBES:
            probe = probe.strip()
            if not probe:
                continue
            for s in range(N_SEEDS):
                for mu in MUS:
                    jobs.append(("screen", probe, s, mu, f"p3_screen_{probe}_s{s}_mu{mu:g}"))
    else:
        for method in METHODS:
            method = method.strip()
            if not method:
                continue
            pr = CHOSEN_PROBE if method == "probe_belief" else ""
            for s in range(N_SEEDS):
                for mu in MUS:
                    jobs.append((method, pr, s, mu, f"p3_{method}_s{s}_mu{mu:g}"))
    return jobs


def main():
    from isaaclab.app import AppLauncher

    enable_cameras = os.environ.get("P3_ENABLE_CAMERAS", "1") not in ("0", "false", "False")
    app_launcher = AppLauncher(headless=True, enable_cameras=enable_cameras, num_envs=1)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(TASK_SUITE, TASK_ID)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = float(os.environ.get("P3_EPISODE_S", "45" if MODE == "eval" else "25"))
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
        (OUT / "COLLECT_META.json").write_text(json.dumps({
            "dt": dt, "mode": MODE, "tactile": tactile, "n_seeds": N_SEEDS, "probes": PROBES,
            "amp_mm": AMP_M * 1000, "probe_half_steps": PROBE_HALF,
        }, indent=2) + "\n")
        belief = load_belief() if MODE == "eval" else None
        csv_name = "PROBE_SCREENING_RESULTS.csv" if MODE == "screen" else "FULL_DOWNSTREAM_RESULTS.csv"
        res_path = OUT / csv_name
        results, done = [], set()
        if res_path.exists() and res_path.stat().st_size > 0:
            with res_path.open() as f:
                for row in csv.DictReader(f):
                    done.add(row["trial_id"])
                    results.append(row)
        print("resume", len(done), flush=True)
        step_fields = list(Step.__dataclass_fields__.keys())
        for method, probe, s, mu, tid in plan_jobs():
            if tid in done:
                print("skip", tid, flush=True)
                continue
            print("===", tid, "===", flush=True)
            rows, rec = run_episode(
                env, seed_idx=s, mu=mu, trial_id=tid, method=method, probe_name=probe or None,
                dt=dt, tactile=tactile, do_place=(MODE == "eval"), belief_model=belief,
            )
            results.append(rec)
            _rewrite(res_path, results, list(rec.keys()))
            if rows:
                _append(OUT / "PROBE_STEP_TRAJECTORIES.csv", [asdict(r) for r in rows], step_fields)
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
