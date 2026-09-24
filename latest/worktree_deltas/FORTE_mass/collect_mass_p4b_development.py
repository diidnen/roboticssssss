#!/usr/bin/env python3
"""Collect a mass-controlled P4-B query matrix on development roots.

This adapter imports the frozen P4-B implementation read-only, applies an
absolute mass and recomputed inertia after each reset, and writes only to the
caller-supplied mass-extension directory.  It does not read downstream
outcomes and does not alter the friction-line collector or its results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


TABERO = Path("/home/exouser/Tabero")
P4_PATH = TABERO / "analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
TASKS = {0: "alphabet_soup_1", 5: "tomato_sauce_1"}
MASS_BANDS = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}


def load_p4(task: int, out: Path):
    os.environ["P4_TASK_ID"] = str(task)
    os.environ["P4_VARIANT"] = "P4B"
    os.environ["P4_OUT"] = str(out / "P4B_IMPORT")
    os.environ["P4_RESUME"] = "0"
    name = f"mass_p4b_task{task}"
    spec = importlib.util.spec_from_file_location(name, P4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {P4_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def tensor_json(x: Any) -> Any:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
    except Exception:
        pass
    if isinstance(x, dict):
        return {str(k): tensor_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tensor_json(v) for v in x]
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    return x


def add_mass_setter(p4, obj_name: str):
    """Wrap frozen friction application with absolute mass/inertia setting."""
    original_friction = p4._apply_friction
    state: dict[str, Any] = {}

    def apply(env, name, mu):
        applied = original_friction(env, name, mu)
        import torch
        view = env.scene[obj_name].root_physx_view
        if "mass" not in state:
            state["mass"] = view.get_masses().clone()
            state["inertia"] = view.get_inertias().clone()
            state["nominal_mass"] = float(state["mass"][0, 0].item())
        requested = float(os.environ["MASS_QUERY_CURRENT_KG"])
        ratio = requested / max(state["nominal_mass"], 1e-12)
        view.set_masses(state["mass"] * ratio, torch.arange(1, dtype=torch.int32))
        view.set_inertias(state["inertia"] * ratio, torch.arange(1, dtype=torch.int32))
        got = float(view.get_masses()[0, 0].item())
        if not np.isfinite(got) or abs(got - requested) > 5e-4:
            raise RuntimeError(f"mass set verification failed: requested={requested} got={got}")
        return applied

    p4._apply_friction = apply
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=sorted(TASKS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dev-roots", type=int, nargs="+", required=True)
    ap.add_argument("--heldout-roots", type=int, nargs="+", required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    task = int(args.task)
    obj_name = TASKS[task]
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(TABERO / "benchmarks/datasets/libero/assembled_hdf5")
    os.environ.setdefault("LIBERO_CONFIG_DIR", str(TABERO / "benchmarks/datasets/libero/config"))
    os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(TABERO / "benchmarks/datasets/libero/USD"))
    os.environ["P4_EPISODE_S"] = "25"

    manifest = {
        "status": "RUNNING",
        "task": task,
        "object": obj_name,
        "query": "P4B_frozen_contact_conditioned_shear",
        "friction": 0.5,
        "mass_bands_kg": MASS_BANDS,
        "development_roots": [int(x) for x in args.dev_roots],
        "heldout_roots": [int(x) for x in args.heldout_roots],
        "downstream_outcomes_read": False,
        "mass_setting": "absolute mass with inertia scaled by same ratio after reset",
    }
    (out / f"M1_TASK{task}_PROTOCOL.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    app = env = None
    exit_code = 3
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects("libero_object", task)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = 25.0
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        p4 = load_p4(task, out)
        mass_state = add_mass_setter(p4, obj_name)
        all_episode_rows: list[dict[str, Any]] = []
        all_step_rows: list[dict[str, Any]] = []
        trial_index = 0
        for split, roots in (("DEV", args.dev_roots), ("HELDOUT", args.heldout_roots)):
            for root in roots:
                for band, mass in MASS_BANDS.items():
                    trial_id = f"m1_p4b_t{task}_root{int(root)}_{band.lower()}"
                    os.environ["MASS_QUERY_CURRENT_KG"] = str(mass)
                    rows, rec = p4.run_probe_episode(env, seed_idx=int(root), mu=0.5, trial_id=trial_id, dt=dt)
                    episode = dict(rec)
                    episode.update({
                        "trial_id": trial_id,
                        "task_id": task,
                        "object_id": obj_name,
                        "split": split,
                        "mass_band": band,
                        "mass_kg": float(mass),
                        "nominal_mass_kg": float(mass_state.get("nominal_mass", float("nan"))),
                        "friction_fixed": 0.5,
                        "query_dt_s": dt,
                        "query_valid": int(rec.get("probe_failure", 0) == 0 and rec.get("dropped", 0) == 0),
                    })
                    all_episode_rows.append(episode)
                    for row in rows:
                        sr = asdict(row) if hasattr(row, "__dataclass_fields__") else tensor_json(row)
                        sr.update({"split": split, "mass_band": band, "mass_kg": float(mass), "friction_fixed": 0.5})
                        all_step_rows.append(sr)
                    trial_index += 1
                    write_csv(out / f"M1_TASK{task}_P4B_EPISODES.csv", all_episode_rows)
                    write_csv(out / f"M1_TASK{task}_P4B_TIMESTEPS.csv", all_step_rows)
                    manifest.update({"completed_trials": trial_index, "dt_s": dt, "nominal_mass_kg": mass_state.get("nominal_mass")})
                    (out / f"M1_TASK{task}_PROGRESS.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest.update({"status": "COMPLETED", "episode_rows": len(all_episode_rows), "timestep_rows": len(all_step_rows)})
        (out / f"M1_TASK{task}_PROTOCOL.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        exit_code = 0
    except Exception as exc:
        manifest.update({"status": "ERROR", "error": repr(exc), "trace": traceback.format_exc()})
        (out / f"M1_TASK{task}_PROTOCOL.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        exit_code = 3
    finally:
        # Isaac camera/Kit destruction is known to raise a pybind11 weakref
        # abort after data is flushed.  The frozen P4 collector uses the same
        # process-level exit policy; do not call env.close()/app.close().
        pass

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
