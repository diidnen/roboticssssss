#!/usr/bin/env python3
"""Formal Hidden-Mass collection for qualified structured transport tasks.

This collector preserves the frozen P4-B query and the frozen force-setpoint
controller.  Each sibling-force branch is restored from one post-query state;
the query never reads branch outcomes.  The task path is the qualification
path LONG_TRANSPORT_TURNING_ACCELERATION_V1.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
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
STRUCTURED_PATH = Path("/home/exouser/FORTE_mass/qualify_mass_structure.py")
TASKS = {0: "alphabet_soup_1", 1: "cream_cheese_1", 2: "salad_dressing_1", 3: "bbq_sauce_1", 5: "tomato_sauce_1", 6: "butter_1", 7: "milk_1", 8: "chocolate_pudding_1", 9: "orange_juice_1"}
MASS_BANDS = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}
FORCES = [0.5, 1.0, 1.5, 2.5, 4.0]
PATH_STRUCTURE = "LONG_TRANSPORT_TURNING_ACCELERATION_V1"


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


def jsonable(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().tolist()
    except Exception:
        pass
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)): return value.item()
    return value


def stable_hash(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"]); w.writeheader(); w.writerows(rows)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def capture_query_rgb(env, out: Path, context_id: str):
    result = {"status": "UNAVAILABLE", "frames": {}}
    try:
        import torch
        for name in ("agentview_cam", "eye_in_hand_cam"):
            try:
                rgb = env.scene[name].data.output.get("rgb")
                if rgb is None: continue
                arr = rgb[0].detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb[0])
                if arr.dtype.kind == "f" and float(np.nanmax(arr)) <= 1.5: arr = np.clip(arr * 255.0, 0, 255)
                arr = np.asarray(arr, dtype=np.uint8)
                path = out / "query_rgb" / f"{context_id}_{name}.npy"; path.parent.mkdir(parents=True, exist_ok=True); np.save(path, arr)
                result["frames"][name] = {"path": str(path), "shape": list(arr.shape), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            except Exception: continue
        if result["frames"]: result["status"] = "CAPTURED"
    except Exception as exc: result["error"] = repr(exc)
    return result


def add_mass_setter(p4, obj_name):
    original = p4._apply_friction; state = {}
    def apply(env, name, mu):
        out = original(env, name, mu)
        import torch
        view = env.scene[obj_name].root_physx_view
        if not state:
            state["mass"] = view.get_masses().clone(); state["inertia"] = view.get_inertias().clone(); state["nominal"] = float(state["mass"][0, 0].item())
        ratio = float(os.environ["MASS_QUERY_CURRENT_KG"]) / state["nominal"]
        ids = torch.arange(1, dtype=torch.int32); view.set_masses(state["mass"] * ratio, ids); view.set_inertias(state["inertia"] * ratio, ids)
        return out
    p4._apply_friction = apply
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=sorted(TASKS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-roots", type=int, nargs="+", required=True)
    ap.add_argument("--test-roots", type=int, nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--resume-from", type=Path, default=None)
    a = ap.parse_args(); task = a.task; obj = TASKS[task]; a.out.mkdir(parents=True, exist_ok=True)
    for k, v in {"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "P4_TASK_ID": str(task), "P4_VARIANT": "P4B", "P4_OUT": str(a.out / "P4B_IMPORT"), "P4_RESUME": "0", "P4_EPISODE_S": "45", "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"), "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"), "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD")}.items(): os.environ[k] = v
    manifest = {"status": "RUNNING", "task_id": task, "task_name": f"libero_object_{task}", "object": obj, "query": "frozen_P4B", "fixed_friction": 0.5, "mass_bands_kg": MASS_BANDS, "candidate_forces_N": FORCES, "train_roots": a.train_roots, "test_roots": a.test_roots, "repeats": a.repeats, "path_structure": PATH_STRUCTURE, "sibling_force_branches_stay_together": True, "downstream_outcome_used_by_query": False, "vla_retrained": False, "controller_law_changed": False, "resume_from": str(a.resume_from) if a.resume_from else None}
    protocol = a.out / f"M3_TASK{task}_STRUCTURED_FORMAL_PROTOCOL.json"; protocol.write_text(json.dumps(manifest, indent=2) + "\n")
    exit_code = 3
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        setup_task_objects("libero_object", task); cfg = parse_env_cfg("Isaac-Libero-Franka-Hybrid-Tactile-v0", device="cuda:0", num_envs=1); cfg.episode_length_s = 45.0
        env = gym.make("Isaac-Libero-Franka-Hybrid-Tactile-v0", cfg=cfg).unwrapped; dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        p4 = load_module(P4_PATH, f"mass_structured_formal_p4_t{task}"); p5 = load_module(P5_PATH, f"mass_structured_formal_p5_t{task}"); structured = load_module(STRUCTURED_PATH, f"mass_structured_formal_path_t{task}")
        p5.TASK_OBJECTS.update(TASKS); p5.TASK_INSTRUCTIONS.update({k: f"pick up the {v.rsplit('_', 1)[0].replace('_', ' ')} and place it in the basket" for k, v in TASKS.items()})
        mass_state = add_mass_setter(p4, obj); contexts = []; branches = []; query_steps = []; observations = []
        if a.resume_from:
            old_context = next(a.resume_from.glob("M3_TASK*_STRUCTURED_FORMAL_CONTEXTS.csv"), None)
            old_branch = next(a.resume_from.glob("M3_TASK*_STRUCTURED_FORMAL_BRANCHES.csv"), None)
            old_steps = next(a.resume_from.glob("M3_TASK*_STRUCTURED_FORMAL_QUERY_TIMESTEPS.csv"), None)
            old_obs = next(a.resume_from.glob("M3_TASK*_STRUCTURED_FORMAL_QUERY_OBSERVATIONS.json"), None)
            if not old_context or not old_branch:
                raise FileNotFoundError("resume directory lacks formal context/branch CSV")
            contexts = read_csv(old_context); branches = read_csv(old_branch)
            if old_steps: query_steps = read_csv(old_steps)
            if old_obs: observations = json.loads(old_obs.read_text())
            manifest["reused_contexts"] = len(contexts); manifest["reused_branches"] = len(branches)
        existing_context_ids = {str(r.get("context_id")) for r in contexts}
        for split, roots in (("TRAIN", a.train_roots), ("TEST", a.test_roots)):
            for root in roots:
                for band, mass in MASS_BANDS.items():
                    os.environ["MASS_QUERY_CURRENT_KG"] = str(mass); cid = f"mass_structured_{split.lower()}_t{task}_root{root}_{band.lower()}"
                    if cid in existing_context_ids: continue
                    env.reset(seed=int(root)); initial_state = copy.deepcopy(env.scene.get_state(is_relative=True)); initial_hash = stable_hash(initial_state)
                    probe_rows, probe_rec = p4.run_probe_episode(env, seed_idx=int(root), mu=0.5, trial_id=cid, dt=dt); valid = int(probe_rec.get("probe_failure", 0) == 0 and probe_rec.get("contact_lost_probe", 0) == 0 and probe_rec.get("dropped", 0) == 0); post = copy.deepcopy(env.scene.get_state(is_relative=True)); post_hash = stable_hash(post)
                    obs = env.observation_manager.compute(); rgb = capture_query_rgb(env, a.out, cid); observations.append({"context_id": cid, "task_id": task, "root_seed": root, "split": split, "mass_band": band, "mass_kg": mass, "initial_state_hash": initial_hash, "post_query_state_hash": post_hash, "policy_observation": jsonable(obs.get("policy", {})), "rgb_observation": rgb, "rgb_used_by_identifier": False})
                    for pr in probe_rows:
                        q = asdict(pr) if hasattr(pr, "__dataclass_fields__") else dict(pr)
                        q = jsonable(q); q.update({"context_id": cid, "task_id": task, "split": split, "mass_band": band, "mass_kg": mass, "friction_fixed": 0.5}); query_steps.append(q)
                    contexts.append({"context_id": cid, "task_id": task, "task_name": f"libero_object_{task}", "object": obj, "root_family": f"t{task}_root{root}", "root_seed": root, "split": split, "mass_band": band, "mass_kg": mass, "friction": 0.5, "initial_state_hash": initial_hash, "post_query_state_hash": post_hash, "query_history_rows": len(probe_rows), "query_valid": valid, "query_record": json.dumps(probe_rec, sort_keys=True), "candidate_forces": len(FORCES), "repeats": a.repeats, "path_structure": PATH_STRUCTURE})
                    if valid:
                        for force in FORCES:
                            for repeat in range(a.repeats):
                                env.reset_to(copy.deepcopy(post), torch.tensor([0], device=env.device), is_relative=True); telemetry = a.out / "telemetry" / f"{cid}_F{force:g}_R{repeat}.csv"
                                br = structured.structured_branch(env, p4, task=task, force=force, context_id=cid, seed=int(root), band=band, dt=dt, out_path=telemetry)
                                br.update({"context_id": cid, "task_id": task, "task_name": f"libero_object_{task}", "object": obj, "root_family": f"t{task}_root{root}", "root_seed": root, "split": split, "mass_band": band, "mass_kg": mass, "friction": 0.5, "repeat": repeat, "initial_state_hash": initial_hash, "post_query_state_hash": post_hash, "query_valid": valid, "measured_force": br.get("measured_force_mean_N", ""), "path_structure": PATH_STRUCTURE}); branches.append(br)
                    write_csv(a.out / f"M3_TASK{task}_STRUCTURED_FORMAL_CONTEXTS.csv", contexts); write_csv(a.out / f"M3_TASK{task}_STRUCTURED_FORMAL_BRANCHES.csv", branches); write_csv(a.out / f"M3_TASK{task}_STRUCTURED_FORMAL_QUERY_TIMESTEPS.csv", query_steps); (a.out / f"M3_TASK{task}_STRUCTURED_FORMAL_QUERY_OBSERVATIONS.json").write_text(json.dumps(observations, indent=2) + "\n")
        manifest.update({"status": "COMPLETED", "dt_s": dt, "contexts": len(contexts), "query_timesteps": len(query_steps), "branches": len(branches), "nominal_mass_kg": mass_state.get("nominal"), "source_sha256": hashlib.sha256(P4_PATH.read_bytes()).hexdigest()}); protocol.write_text(json.dumps(manifest, indent=2) + "\n"); exit_code = 0
    except Exception as exc:
        manifest.update({"status": "ERROR", "error": repr(exc)}); protocol.write_text(json.dumps(manifest, indent=2) + "\n")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(exit_code)


if __name__ == "__main__": main()
