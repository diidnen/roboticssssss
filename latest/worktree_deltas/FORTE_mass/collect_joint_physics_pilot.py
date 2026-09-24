#!/usr/bin/env python3
"""Small 3x3 friction/mass identifiability pilot using frozen P4-B."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

TABERO = Path("/home/exouser/Tabero")
P4_PATH = Path("/home/exouser/FORTE_mass/frozen_p4_mass_probe.py")
TASKS = {2: "salad_dressing_1"}
MUS = [0.25, 0.50, 0.75]
MASSES = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jsonable(v):
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy().tolist()
    except Exception:
        pass
    if isinstance(v, dict): return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jsonable(x) for x in v]
    if isinstance(v, (np.integer, np.floating, np.bool_)): return v.item()
    return v


def add_mass_setter(p4, obj_name):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-roots", type=int, nargs="+", default=[8300, 8301])
    ap.add_argument("--test-roots", type=int, nargs="+", default=[8302])
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    task = 2; obj = TASKS[task]
    os.environ.update({
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
        "P4_TASK_ID": str(task), "P4_VARIANT": "P4B",
        "P4_OUT": str(a.out / "P4B_IMPORT"), "P4_RESUME": "0",
        "P4_EPISODE_S": "45",
        "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
    })
    protocol = {"status": "RUNNING", "query": "frozen_P4B", "task_id": task,
                "friction_grid": MUS, "mass_bands_kg": MASSES,
                "train_roots": a.train_roots, "test_roots": a.test_roots,
                "downstream_outcomes_read": False,
                "force_choice_proxy": "quantize(1.0 + 2.0*friction + 5.0*mass_kg, [0.5,1,1.5,2.5,4])"}
    pfile = a.out / "JOINT_PHYSICS_PILOT_PROTOCOL.json"; pfile.write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []; exit_code = 3
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False, num_envs=1).app
        import gymnasium as gym
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        import tac_manip.tasks  # noqa: F401
        from tac_manip.utils.task_configs import setup_task_objects
        setup_task_objects("libero_object", task)
        cfg = parse_env_cfg("Isaac-Libero-Franka-Hybrid-Tactile-v0", device="cuda:0", num_envs=1); cfg.episode_length_s = 45.0
        env = gym.make("Isaac-Libero-Franka-Hybrid-Tactile-v0", cfg=cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        p4 = load_module(P4_PATH, "joint_pilot_p4"); state = add_mass_setter(p4, obj)
        for split, roots in (("TRAIN", a.train_roots), ("TEST", a.test_roots)):
            for root in roots:
                for band, mass in MASSES.items():
                    for mu in MUS:
                        os.environ["MASS_QUERY_CURRENT_KG"] = str(mass)
                        trial = f"joint_{split.lower()}_root{root}_mu{mu:g}_{band.lower()}"
                        env.reset(seed=int(root))
                        probe, rec = p4.run_probe_episode(env, seed_idx=int(root), mu=mu, trial_id=trial, dt=dt)
                        rec = jsonable(rec)
                        rec.update({"trial_id": trial, "split": split, "root_seed": root,
                                    "friction": mu, "mass_band": band, "mass_kg": mass,
                                    "query_valid": int(rec.get("probe_failure", 0) == 0 and rec.get("contact_lost_probe", 0) == 0 and rec.get("dropped", 0) == 0),
                                    "query_rows": len(probe), "query_record_json": json.dumps(rec, sort_keys=True)})
                        rows.append(rec)
                        with (a.out / "JOINT_PHYSICS_PILOT_TIMESTEPS.csv").open("a", newline="", encoding="utf-8") as f:
                            step_rows = [jsonable(asdict(x)) for x in probe]
                            if step_rows:
                                fields = list(step_rows[0]); w = csv.DictWriter(f, fieldnames=fields)
                                if f.tell() == 0: w.writeheader()
                                w.writerows(step_rows)
                        with (a.out / "JOINT_PHYSICS_PILOT_EPISODES.csv").open("w", newline="", encoding="utf-8") as f:
                            fields = list(rows[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        protocol.update({"status": "COMPLETED", "episodes": len(rows), "dt_s": dt,
                         "nominal_mass_kg": state.get("nominal"), "source_sha256": hashlib.sha256(P4_PATH.read_bytes()).hexdigest()})
        exit_code = 0
    except Exception as exc:
        protocol.update({"status": "ERROR", "error": repr(exc)})
    pfile.write_text(json.dumps(protocol, indent=2) + "\n")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(exit_code)


if __name__ == "__main__": main()
