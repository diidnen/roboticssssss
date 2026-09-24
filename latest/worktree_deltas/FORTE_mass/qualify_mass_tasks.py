#!/usr/bin/env python3
"""Small mass x force qualification for candidate LIBERO object tasks.

Uses the frozen P4-B contact query followed by the existing scripted downstream
controller. This is a qualification screen, not the formal dataset and not a
VLA retraining path. It only writes to the mass-extension timestamp directory.
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
FORTE = Path("/home/exouser/FORTE")
P4_PATH = Path("/home/exouser/FORTE_mass/frozen_p4_mass_probe.py")
P5_PATH = TABERO / "analysis/p5s0a_true_matched_dataset.py"
# Candidate set recovered from the authoritative LIBERO object task manifest.
# Qualification is intentionally broader than the two initially prioritized
# tasks; promotion is decided from measured outcomes, not from task name.
TASKS = {
    0: "alphabet_soup_1", 1: "cream_cheese_1", 2: "salad_dressing_1",
    3: "bbq_sauce_1", 5: "tomato_sauce_1", 6: "butter_1",
    7: "milk_1", 8: "chocolate_pudding_1", 9: "orange_juice_1",
}
MASS_BANDS = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}
# Development force grid selected before the next qualification run from the
# 0.49--1.96 N weight range of the LOW--HIGH masses plus a 4 N margin.
FORCES = [0.5, 1.0, 1.5, 2.5, 4.0]


class NoopLogger:
    def emit(self, *args, **kwargs):
        return None


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_p4(task: int, out: Path):
    os.environ["P4_TASK_ID"] = str(task)
    os.environ["P4_VARIANT"] = "P4B"
    os.environ["P4_OUT"] = str(out / "P4B_IMPORT")
    os.environ["P4_RESUME"] = "0"
    return load_module(P4_PATH, f"mass_qual_p4_task{task}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def add_mass_setter(p4, obj_name: str):
    original = p4._apply_friction
    state = {}

    def apply(env, name, mu):
        applied = original(env, name, mu)
        import torch
        view = env.scene[obj_name].root_physx_view
        if not state:
            state["mass"] = view.get_masses().clone()
            state["inertia"] = view.get_inertias().clone()
            state["nominal"] = float(state["mass"][0, 0].item())
        requested = float(os.environ["MASS_QUERY_CURRENT_KG"])
        ratio = requested / state["nominal"]
        view.set_masses(state["mass"] * ratio, torch.arange(1, dtype=torch.int32))
        view.set_inertias(state["inertia"] * ratio, torch.arange(1, dtype=torch.int32))
        return applied

    p4._apply_friction = apply
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=sorted(TASKS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--roots", type=int, nargs=2, required=True)
    args = ap.parse_args()
    task, obj = int(args.task), TASKS[int(args.task)]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(TABERO / "benchmarks/datasets/libero/assembled_hdf5")
    os.environ.setdefault("LIBERO_CONFIG_DIR", str(TABERO / "benchmarks/datasets/libero/config"))
    os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(TABERO / "benchmarks/datasets/libero/USD"))
    os.environ["P4_EPISODE_S"] = "45"

    manifest = {
        "status": "RUNNING", "task": task, "object": obj,
        "query": "frozen_P4B", "fixed_friction": 0.5,
        "mass_bands_kg": MASS_BANDS, "forces_N": FORCES,
        "roots": [int(x) for x in args.roots],
        "qualification_only": True, "downstream_controller": "existing_scripted_branch_controller",
        "vla_retraining": False, "outcome_used_for_query_design": False,
    }
    (out / f"M2_TASK{task}_PROTOCOL.json").write_text(json.dumps(manifest, indent=2) + "\n")
    app = env = None
    exit_code = 3
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects("libero_object", task)
        cfg = parse_env_cfg("Isaac-Libero-Franka-Hybrid-Tactile-v0", device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        env = gym.make("Isaac-Libero-Franka-Hybrid-Tactile-v0", cfg=cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        p4 = load_p4(task, out)
        p5 = load_module(P5_PATH, f"mass_qual_p5_task{task}")
        # The frozen P5 helper predates the broader LIBERO recovery map and
        # only contains the four friction-core task IDs. Its downstream
        # controller is task-agnostic apart from object/instruction lookup, so
        # extend those lookups locally without changing controller code.
        p5.TASK_OBJECTS.update(TASKS)
        p5.TASK_INSTRUCTIONS.update({
            k: f"pick up the {v.rsplit('_', 1)[0].replace('_', ' ')} and place it in the basket"
            for k, v in TASKS.items()
        })
        state = add_mass_setter(p4, obj)
        logger = NoopLogger()
        context_rows: list[dict] = []
        branch_rows: list[dict] = []
        for root in args.roots:
            for band, mass in MASS_BANDS.items():
                os.environ["MASS_QUERY_CURRENT_KG"] = str(mass)
                context_id = f"m2_qual_t{task}_root{int(root)}_{band.lower()}"
                probe_rows, probe_rec = p4.run_probe_episode(env, seed_idx=int(root), mu=0.5, trial_id=context_id, dt=dt)
                probe_valid = int(probe_rec.get("probe_failure", 0) == 0 and probe_rec.get("contact_lost_probe", 0) == 0 and probe_rec.get("dropped", 0) == 0)
                state_snapshot = copy.deepcopy(env.scene.get_state(is_relative=True))
                context = {
                    "context_id": context_id, "task_id": task, "task_name": f"libero_object_{task}", "object": obj,
                    "root_seed": int(root), "mass_band": band, "mass_kg": float(mass), "friction": 0.5,
                    "probe_valid": probe_valid, "probe_rows": len(probe_rows), "probe_stop": probe_rec.get("stop_trigger", ""),
                    "probe_displacement_mm": probe_rec.get("actual_probe_displacement_mm", ""),
                    "normal_force_mean": probe_rec.get("normal_force_mean", ""), "rho_impulse": probe_rec.get("rho_impulse", ""),
                    "mass_nominal_kg": state.get("nominal", ""), "candidate_branches": 0,
                }
                if probe_valid:
                    for force in FORCES:
                        env.reset_to(copy.deepcopy(state_snapshot), torch.tensor([0], device=env.device), is_relative=True)
                        branch = p5.downstream_branch(env, p4, task_id=task, force=force,
                            branch_label=f"QUAL_F{force:g}", context_id=context_id, split="QUAL",
                            seed=int(root), friction=0.5, dt=dt, logger=logger, telemetry_path=None)
                        branch.update({"context_id": context_id, "task_id": task, "task_name": f"libero_object_{task}",
                                       "object": obj, "root_seed": int(root), "mass_band": band, "mass_kg": float(mass),
                                       "friction": 0.5, "requested_force_N": float(force), "probe_valid": probe_valid})
                        branch_rows.append(branch)
                    context["candidate_branches"] = len(FORCES)
                context_rows.append(context)
                write_csv(out / f"M2_TASK{task}_QUAL_CONTEXTS.csv", context_rows)
                write_csv(out / f"M2_TASK{task}_QUAL_BRANCHES.csv", branch_rows)
        manifest.update({"status": "COMPLETED", "context_rows": len(context_rows), "branch_rows": len(branch_rows), "dt_s": dt})
        (out / f"M2_TASK{task}_PROTOCOL.json").write_text(json.dumps(manifest, indent=2) + "\n")
        exit_code = 0
    except Exception as exc:
        manifest.update({"status": "ERROR", "error": repr(exc)})
        (out / f"M2_TASK{task}_PROTOCOL.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
