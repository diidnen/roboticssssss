#!/usr/bin/env python3
"""Qualification for a mass-sensitive multi-waypoint transport structure.

The frozen P4-B query, force controller, masses, and candidate force grid are
unchanged.  Only the downstream task geometry is extended to a reproducible
turning/acceleration path before the existing basket placement.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

TABERO = Path("/home/exouser/Tabero")
P4_PATH = Path("/home/exouser/FORTE_mass/frozen_p4_mass_probe.py")
P5_PATH = TABERO / "analysis/p5s0a_true_matched_dataset.py"
TASKS = {
    0: "alphabet_soup_1", 1: "cream_cheese_1", 2: "salad_dressing_1",
    3: "bbq_sauce_1", 5: "tomato_sauce_1", 6: "butter_1",
    7: "milk_1", 8: "chocolate_pudding_1", 9: "orange_juice_1",
}
MASS_BANDS = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}
if os.environ.get("MASS_BANDS_PROFILE") == "TASK0_VALID_RANGE_V1":
    # Development-only recovery profile for task0: its 0.20 kg probe was
    # query-invalid, so this narrower three-band range is frozen before the
    # new development root and is not applied to the task2 formal protocol.
    MASS_BANDS = {"LOW": 0.025, "MID": 0.05, "HIGH": 0.10}
FORCES = [0.5, 1.0, 1.5, 2.5, 4.0]
BASKET = "basket_1"


class NoopLogger:
    def emit(self, *args, **kwargs):
        return None


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or ["status"])
        w.writeheader(); w.writerows(rows)


def add_mass_setter(p4, obj_name: str):
    original = p4._apply_friction
    state = {}
    def apply(env, name, mu):
        out = original(env, name, mu)
        import torch
        view = env.scene[obj_name].root_physx_view
        if not state:
            state["mass"] = view.get_masses().clone()
            state["inertia"] = view.get_inertias().clone()
            state["nominal"] = float(state["mass"][0, 0].item())
        ratio = float(os.environ["MASS_QUERY_CURRENT_KG"]) / state["nominal"]
        ids = torch.arange(1, dtype=torch.int32)
        view.set_masses(state["mass"] * ratio, ids)
        view.set_inertias(state["inertia"] * ratio, ids)
        return out
    p4._apply_friction = apply
    return state


def structured_branch(env, p4, *, task, force, context_id, seed, band, dt, out_path):
    import torch
    obj = TASKS[task]
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    aa = p4._aa(eef[3:7]); cmd = eef[:3].copy(); obj0 = env.scene[obj].data.root_pos_w[0].detach().cpu().numpy()
    basket, _ = p4._pose_in_base(env, BASKET)
    lift = cmd.copy(); lift[2] = max(lift[2] + 0.142, cmd[2] + 0.08)
    transit_z = max(lift[2], basket[2] + 0.18)
    center = basket.copy(); center[2] = transit_z
    # A fixed turning/acceleration path. It is deliberately independent of
    # mass, force outcome, or test data.
    waypoints = [
        center + np.array([0.00, 0.18, 0.00]),
        center + np.array([0.00, -0.18, 0.00]),
        center + np.array([0.14, 0.00, 0.00]),
        center,
    ]
    phases = [("hold", 20, cmd.copy(), "track"), ("lift", 60, lift, "track")]
    for i, target in enumerate(waypoints):
        phases.append((f"turn_{i}", 60, target, "freeze"))
    place = basket.copy(); place[2] = basket[2] + 0.10
    phases += [("place", 40, place, "freeze"), ("release", 50, place, "open"), ("settle", 50, place, "open")]
    d_pred = p4.D_CLOSED; step = 0; lift_success = 0; dropped = 0; lost = 0; max_basket = 0.0; peak = 0.0; forces = []; rows = []
    for phase, n, target, mode in phases:
        start = cmd.copy()
        for i in range(n):
            cmd = p4._interp(start, target, i, n)
            if mode == "open": d_pred = p4.D_OPEN
            action = p4._make_action(cmd, aa, d_pred, force if mode != "open" else 0.0, env.device)
            obs, _, term, trunc, _ = env.step(action); step += 1
            f = p4._f(p4._dbg(env).get("f_sq_meas"), 0.0); peak = max(peak, f)
            if mode != "open": forces.append(f)
            if mode == "track": d_pred = p4._force_servo(d_pred, f, force)
            pos = env.scene[obj].data.root_pos_w[0].detach().cpu().numpy(); dz = float(pos[2] - obj0[2])
            if dz >= 0.01: lift_success = 1
            try: dropped = int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
            except Exception: pass
            if lift_success and phase.startswith("turn_") and (dropped or dz < 0.02): lost = 1
            try:
                cs = env.scene[f"contact_{BASKET}_{obj}"]
                max_basket = max(max_basket, float(torch.linalg.vector_norm(cs.data.force_matrix_w.reshape(-1, 3)[0]).item()))
            except Exception: pass
            rows.append({"context_id": context_id, "task": task, "seed": seed, "mass_band": band, "force": force, "step": step, "phase": phase, "measured_force_N": f, "object_z_delta": dz})
            if bool(term[0].item()) or bool(trunc[0].item()): break
    # A valid full-task outcome must retain the object through the complete
    # transport path.  Basket contact alone is insufficient: a dropped object
    # can later collide with the basket and create a false placement success.
    transport_retention = int(lift_success and not lost and not dropped)
    success = int(transport_retention and max_basket > 0.05)
    write_csv(out_path, rows)
    vals = forces[len(forces)//2:] if forces else [0.0]
    return {"context_id": context_id, "task_id": task, "root_seed": seed, "mass_band": band, "mass_kg": MASS_BANDS[band], "requested_force_N": force, "pick_success": int(lift_success > 0), "lift_success": lift_success, "transport_retention": transport_retention, "place_success": int(max_basket > 0.05), "full_task_success_y": success, "dropped": dropped, "lost_in_transit": lost, "failure_stage": "turning_transport" if lost else ("placement" if not success else ""), "steps": step, "measured_force_mean_N": float(np.mean(vals)), "measured_force_peak_N": peak, "path_structure": "LONG_TRANSPORT_TURNING_ACCELERATION_V1"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", type=int, choices=sorted(TASKS), required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--roots", type=int, nargs="+", required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); task = a.task; obj = TASKS[task]
    os.environ.update({
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
        "P4_TASK_ID": str(task), "P4_VARIANT": "P4B",
        "P4_OUT": str(a.out / "P4B_IMPORT"), "P4_RESUME": "0",
        "P4_EPISODE_S": "45",
        "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
    })
    manifest = {"status": "RUNNING", "task": task, "object": obj, "query": "frozen_P4B", "fixed_friction": 0.5, "mass_bands_kg": MASS_BANDS, "forces_N": FORCES, "roots": a.roots, "path_structure": "LONG_TRANSPORT_TURNING_ACCELERATION_V1", "vla_retraining": False, "controller_law_changed": False, "qualification_only": True}
    (a.out / f"M2_STRUCTURED_TASK{task}_PROTOCOL.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        setup_task_objects("libero_object", task); cfg = parse_env_cfg("Isaac-Libero-Franka-Hybrid-Tactile-v0", device="cuda:0", num_envs=1); cfg.episode_length_s = 45.0
        env = gym.make("Isaac-Libero-Franka-Hybrid-Tactile-v0", cfg=cfg).unwrapped; dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        p4 = load_module(P4_PATH, f"mass_structured_p4_{task}"); p5 = load_module(P5_PATH, f"mass_structured_p5_{task}"); p5.TASK_OBJECTS.update(TASKS); p5.TASK_INSTRUCTIONS.update({k: f"pick up the {v.rsplit('_', 1)[0].replace('_', ' ')} and place it in the basket" for k, v in TASKS.items()}); nominal = add_mass_setter(p4, obj)
        contexts = []; branches = []
        for root in a.roots:
            for band, mass in MASS_BANDS.items():
                os.environ["MASS_QUERY_CURRENT_KG"] = str(mass); cid = f"structured_t{task}_root{root}_{band.lower()}"; env.reset(seed=root); probe, rec = p4.run_probe_episode(env, seed_idx=root, mu=0.5, trial_id=cid, dt=dt); valid = int(rec.get("probe_failure", 0) == 0 and rec.get("contact_lost_probe", 0) == 0 and rec.get("dropped", 0) == 0); snap = copy.deepcopy(env.scene.get_state(is_relative=True)); contexts.append({"context_id": cid, "task_id": task, "root_seed": root, "mass_band": band, "mass_kg": mass, "friction": 0.5, "query_valid": valid, "query_stop": rec.get("stop_trigger", ""), "path_structure": manifest["path_structure"]})
                if valid:
                    for force in FORCES:
                        env.reset_to(copy.deepcopy(snap), torch.tensor([0], device=env.device), is_relative=True); branches.append(structured_branch(env, p4, task=task, force=force, context_id=cid, seed=root, band=band, dt=dt, out_path=a.out / "telemetry" / f"{cid}_F{force:g}.csv"))
                write_csv(a.out / f"M2_STRUCTURED_TASK{task}_CONTEXTS.csv", contexts); write_csv(a.out / f"M2_STRUCTURED_TASK{task}_BRANCHES.csv", branches)
        manifest.update({"status": "COMPLETED", "contexts": len(contexts), "branches": len(branches), "dt_s": dt, "nominal_mass_kg": nominal.get("nominal")}); (a.out / f"M2_STRUCTURED_TASK{task}_PROTOCOL.json").write_text(json.dumps(manifest, indent=2) + "\n"); os._exit(0)
    except Exception as exc:
        manifest.update({"status": "ERROR", "error": repr(exc)}); (a.out / f"M2_STRUCTURED_TASK{task}_PROTOCOL.json").write_text(json.dumps(manifest, indent=2) + "\n"); os._exit(3)


if __name__ == "__main__": main()
