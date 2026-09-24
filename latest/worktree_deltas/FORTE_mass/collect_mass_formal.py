#!/usr/bin/env python3
"""Formal Hidden-Mass branch collector.

Run one task per Isaac process after M2 qualification. Root families are kept
within one split; every sibling force branch is restored from one post-query
state. This collector never trains or reads downstream outcomes while making
the query/identifier decision.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
# Frozen candidate set for the mass line; selected before formal test roots.
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
    return load_module(P4_PATH, f"mass_formal_p4_task{task}")


def write_csv(path: Path, rows: list[dict]) -> None:
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


def jsonable(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().tolist()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def capture_query_rgb(env, out: Path, context_id: str) -> dict:
    """Persist camera RGB at the end of the frozen query when available.

    Camera sensors are not part of the P4-B policy observation group, so they
    are recorded as an auxiliary artifact rather than fed back into the
    frozen query or downstream controller. Missing cameras are explicit.
    """
    result = {"status": "UNAVAILABLE", "frames": {}}
    try:
        import torch
        for name in ("agentview_cam", "eye_in_hand_cam"):
            try:
                camera = env.scene[name]
                rgb = camera.data.output.get("rgb")
                if rgb is None:
                    continue
                arr = rgb[0].detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb[0])
                if arr.dtype.kind == "f" and float(np.nanmax(arr)) <= 1.5:
                    arr = np.clip(arr * 255.0, 0, 255)
                arr = np.asarray(arr, dtype=np.uint8)
                path = out / "query_rgb" / f"{context_id}_{name}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, arr)
                result["frames"][name] = {"path": str(path), "shape": list(arr.shape), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            except Exception:
                continue
        if result["frames"]:
            result["status"] = "CAPTURED"
    except Exception as exc:
        result["error"] = repr(exc)
    return result


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
    ap.add_argument("--train-roots", type=int, nargs="+", required=True)
    ap.add_argument("--test-roots", type=int, nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    task, obj = int(args.task), TASKS[int(args.task)]
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(TABERO / "benchmarks/datasets/libero/assembled_hdf5")
    os.environ.setdefault("LIBERO_CONFIG_DIR", str(TABERO / "benchmarks/datasets/libero/config"))
    os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(TABERO / "benchmarks/datasets/libero/USD"))
    os.environ["P4_EPISODE_S"] = "45"
    manifest = {
        "status": "RUNNING", "task_id": task, "task_name": f"libero_object_{task}", "object": obj,
        "query": "P4-B frozen contact-conditioned shear", "fixed_friction": 0.5,
        "mass_bands_kg": MASS_BANDS, "candidate_forces_N": FORCES,
        "train_roots": args.train_roots, "test_roots": args.test_roots, "repeats": args.repeats,
        "sibling_force_branches_stay_together": True, "downstream_outcome_used_for_query": False,
        "vla_retrained": False, "controller_law_changed": False,
    }
    protocol = out / f"M3_TASK{task}_FORMAL_PROTOCOL.json"
    protocol.write_text(json.dumps(manifest, indent=2) + "\n")
    app = env = None; exit_code = 3
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
        p5 = load_module(P5_PATH, f"mass_formal_p5_task{task}")
        # Keep the frozen downstream controller unchanged while extending its
        # local object/instruction lookup to the recovered LIBERO tasks.
        p5.TASK_OBJECTS.update(TASKS)
        p5.TASK_INSTRUCTIONS.update({
            k: f"pick up the {v.rsplit('_', 1)[0].replace('_', ' ')} and place it in the basket"
            for k, v in TASKS.items()
        })
        mass_state = add_mass_setter(p4, obj)
        contexts: list[dict] = []; branches: list[dict] = []; timesteps: list[dict] = []; query_obs: list[dict] = []
        for split, roots in (("TRAIN", args.train_roots), ("TEST", args.test_roots)):
            for root in roots:
                for band, mass in MASS_BANDS.items():
                    os.environ["MASS_QUERY_CURRENT_KG"] = str(mass)
                    context_id = f"mass_formal_{split.lower()}_t{task}_root{int(root)}_{band.lower()}"
                    obs0, _ = env.reset(seed=int(root))
                    initial_hash = stable_hash(env.scene.get_state(is_relative=True))
                    probe_rows, probe_rec = p4.run_probe_episode(env, seed_idx=int(root), mu=0.5, trial_id=context_id, dt=dt)
                    valid = int(probe_rec.get("probe_failure", 0) == 0 and probe_rec.get("contact_lost_probe", 0) == 0 and probe_rec.get("dropped", 0) == 0)
                    post_query_state = copy.deepcopy(env.scene.get_state(is_relative=True))
                    post_query_hash = stable_hash(post_query_state)
                    obs = env.observation_manager.compute()
                    rgb_observation = capture_query_rgb(env, out, context_id)
                    query_obs.append({"context_id": context_id, "task_id": task, "root_seed": int(root), "split": split, "mass_band": band, "mass_kg": mass, "initial_state_hash": initial_hash, "post_query_state_hash": post_query_hash, "policy_observation": jsonable(obs.get("policy", {})), "rgb_observation": rgb_observation, "rgb_used_by_identifier": False})
                    for step in probe_rows:
                        row = jsonable(step); row.update({"context_id": context_id, "task_id": task, "split": split, "mass_band": band, "mass_kg": mass, "friction_fixed": 0.5})
                        timesteps.append(row)
                    context = {"context_id": context_id, "task_id": task, "task_name": f"libero_object_{task}", "object": obj, "root_family": f"t{task}_root{int(root)}", "root_seed": int(root), "split": split, "mass_band": band, "mass_kg": mass, "friction": 0.5, "initial_state_hash": initial_hash, "post_query_state_hash": post_query_hash, "query_history_rows": len(probe_rows), "query_valid": valid, "query_record": json.dumps(probe_rec, sort_keys=True), "candidate_forces": len(FORCES), "repeats": args.repeats}
                    contexts.append(context)
                    if valid:
                        for force in FORCES:
                            for repeat in range(args.repeats):
                                env.reset_to(copy.deepcopy(post_query_state), torch.tensor([0], device=env.device), is_relative=True)
                                branch = p5.downstream_branch(env, p4, task_id=task, force=force, branch_label=f"FORMAL_F{force:g}_R{repeat}", context_id=context_id, split=split, seed=int(root), friction=0.5, dt=dt, logger=NoopLogger(), telemetry_path=None)
                                branch.update({"context_id": context_id, "task_id": task, "task_name": f"libero_object_{task}", "object": obj, "root_family": f"t{task}_root{int(root)}", "root_seed": int(root), "split": split, "mass_band": band, "mass_kg": mass, "friction": 0.5, "repeat": repeat, "initial_state_hash": initial_hash, "post_query_state_hash": post_query_hash, "query_valid": valid, "measured_force": branch.get("measured_force_mean_N", "")})
                                branches.append(branch)
                    write_csv(out / f"M3_TASK{task}_FORMAL_CONTEXTS.csv", contexts)
                    write_csv(out / f"M3_TASK{task}_FORMAL_BRANCHES.csv", branches)
                    write_csv(out / f"M3_TASK{task}_FORMAL_QUERY_TIMESTEPS.csv", timesteps)
                    (out / f"M3_TASK{task}_FORMAL_QUERY_OBSERVATIONS.json").write_text(json.dumps(query_obs, indent=2) + "\n")
        manifest.update({"status": "COMPLETED", "dt_s": dt, "contexts": len(contexts), "query_timesteps": len(timesteps), "branches": len(branches), "nominal_mass_kg": mass_state.get("nominal"), "source_sha256": hashlib.sha256(P4_PATH.read_bytes()).hexdigest()})
        protocol.write_text(json.dumps(manifest, indent=2) + "\n"); exit_code = 0
    except Exception as exc:
        manifest.update({"status": "ERROR", "error": repr(exc)})
        protocol.write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        pass
    sys.stdout.flush(); sys.stderr.flush(); os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
