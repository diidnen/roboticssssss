#!/usr/bin/env python3
"""Capture a reproducible established grasp from an official LIBERO replay.

This executable runs one task per Isaac process.  It replays joint-space
demonstration actions only until the predeclared bilateral-contact handoff,
then serializes the relative scene state.  It never supplies downstream
actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/home/exouser/Tabero")
DATA = Path("/media/volume/newdata/exouser/task_form_coverage_20260910/assembled_hdf5")
OUT = REPO / "analysis/results/task_form_coverage_20260910/supplied_grasps"
ENV_ID = "Isaac-Libero-Franka-Replay-Camera-Tactile-v0"
CONTACT_N = 0.15
STABLE_STEPS = 5
TASKS = {
    "book": {
        "suite": "libero_10", "task": 5, "object": "black_book_1", "target": "desk_caddy_1",
        "glob": "libero_10_task5_*book*demo.hdf5",
    },
    "wine": {
        "suite": "libero_goal", "task": 9, "object": "wine_bottle_1", "target": "wine_rack_1",
        "glob": "libero_goal_task9_*wine*bottle*rack*demo.hdf5",
    },
    "bowl": {
        "suite": "libero_spatial", "task": 4, "object": "akita_black_bowl_1", "target": "plate_1",
        "glob": "libero_spatial_task4_*bowl*drawer*plate*demo.hdf5",
    },
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value: Any) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def clean(value: Any) -> Any:
    if hasattr(value, "detach"):
        return clean(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=sorted(TASKS))
    parser.add_argument("--demo-id", type=int, default=0)
    args = parser.parse_args()
    task = TASKS[args.candidate]
    matches = sorted(DATA.glob(task["glob"]))
    if len(matches) != 1:
        raise RuntimeError(f"expected one official HDF5 for {args.candidate}, got {matches}")
    source = matches[0]
    destination = OUT / args.candidate / f"demo_{args.demo_id}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {destination}")
    destination.mkdir(parents=True)

    os.environ.update({
        "HDF5_TRAJ_SOURCE_DIR": str(DATA),
        "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
        "TASK_SUITE": task["suite"], "TASK_ID": str(task["task"]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
    })
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab.utils.datasets import HDF5DatasetFileHandler
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(task["suite"], task["task"])
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 60.0
        cfg.terminations.time_out = None
        cfg.observations.policy.concatenate_terms = False
        cfg.sim.physx.enable_ccd = True
        sensor_name = f"contact_grasp_{task['object']}"
        if hasattr(cfg.scene, sensor_name):
            getattr(cfg.scene, sensor_name).max_contact_data_count_per_prim = 128
        env = gym.make(ENV_ID, cfg=cfg).unwrapped

        ds = HDF5DatasetFileHandler(); ds.open(str(source))
        names = sorted(ds.get_episode_names(), key=lambda x: int(str(x).split("_")[-1]))
        chosen = next((name for name in names if int(str(name).split("_")[-1]) == args.demo_id), None)
        if chosen is None:
            raise ValueError(f"demo {args.demo_id} unavailable: {names}")
        episode = ds.load_episode(chosen, env.device)
        actions = episode.data.get("actions")
        if actions is None or tuple(actions.shape)[1:] != (8,) or env.action_space.shape[-1] != 8:
            raise RuntimeError(f"joint replay mismatch: actions={getattr(actions, 'shape', None)}, env={env.action_space.shape}")
        env.reset()
        env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)

        stable = 0
        telemetry = []
        handoff = None
        for index in range(actions.shape[0]):
            obs, _, terminated, truncated, _ = env.step(actions[index:index + 1])
            sensor = env.scene[sensor_name]
            matrix = sensor.data.force_matrix_w[0].detach().cpu().numpy().reshape(-1, 1, 3)
            if matrix.shape[0] < 2:
                raise RuntimeError(f"bilateral sensor body axis missing: {matrix.shape}")
            left = matrix[0].sum(axis=0)
            right = matrix[1].sum(axis=0)
            bilateral = np.linalg.norm(left) >= CONTACT_N and np.linalg.norm(right) >= CONTACT_N
            all_force = obs["policy"]["gripper_net_force"][0].detach().cpu().numpy()
            if all_force.ndim == 3:
                all_force = all_force[-1]
            all_left = float(np.linalg.norm(all_force[0]))
            all_right = float(np.linalg.norm(all_force[1]))
            native_grasp = bool(obs["subtask_terms"]["grasp_1"][0].detach().cpu().item())
            closed_command = float(actions[index, -1].detach().cpu()) < 0.0
            stable = stable + 1 if bilateral and closed_command else 0
            row = {
                "step": index + 1, "left_object_force_norm_N": float(np.linalg.norm(left)),
                "right_object_force_norm_N": float(np.linalg.norm(right)), "bilateral": bool(bilateral),
                "left_all_contact_force_norm_N": all_left, "right_all_contact_force_norm_N": all_right,
                "native_grasp_1": native_grasp,
                "closed_command": bool(closed_command), "stable_run": stable,
                "object_pose": clean(env.scene[task["object"]].data.root_state_w[0, :7]),
                "robot_joint_position": clean(env.scene["robot"].data.joint_pos[0]),
                "gripper_joint_position": clean(env.scene["robot"].data.joint_pos[0, -2:]),
            }
            telemetry.append(row)
            if stable >= STABLE_STEPS:
                state = env.scene.get_state(is_relative=True)
                snapshot = destination / "SUPPLIED_GRASP_SCENE_STATE.pt"
                torch.save({
                    "scene_state": state,
                    "handoff_step": index + 1,
                    "source_action": actions[index].detach().cpu(),
                    "source_action_prefix": actions[:index + 1].detach().cpu(),
                    "task": task,
                }, snapshot)
                handoff = {**row, "snapshot": str(snapshot), "snapshot_sha256": sha(snapshot)}
                break
            if bool(terminated[0]) or bool(truncated[0]):
                break
        if handoff is None:
            failure = {
                "status": "FAILED_NO_FIVE_STEP_STABLE_BILATERAL_GRASP",
                "task": task, "source_hdf5": str(source), "source_hdf5_sha256": sha(source),
                "demo_id": args.demo_id, "episode_name": str(chosen),
                "source_actions_shape": list(actions.shape), "contact_threshold_N": CONTACT_N,
                "stable_steps_required": STABLE_STEPS,
                "maximum_observed_stable_run": max((row["stable_run"] for row in telemetry), default=0),
                "pre_handoff_action_source": "OFFICIAL_LIBERO_DEMONSTRATION_REPLAY",
                "post_handoff_action_source": "NONE_IN_THIS_EXECUTABLE",
            }
            (destination / "SUPPLIED_GRASP_FAILURE.json").write_text(json.dumps(clean(failure), indent=2, sort_keys=True) + "\n")
            (destination / "REPLAY_TELEMETRY.json").write_text(json.dumps(clean(telemetry), indent=2) + "\n")
            print(json.dumps(clean(failure), indent=2), flush=True)
            sys.stdout.flush(); sys.stderr.flush(); os._exit(2)
        metadata = {
            "status": "CAPTURED_NOT_YET_HYBRID_RESTORE_QUALIFIED",
            "task": task, "source_hdf5": str(source), "source_hdf5_sha256": sha(source),
            "demo_id": args.demo_id, "episode_name": str(chosen),
            "source_actions_shape": list(actions.shape),
            "source_action_prefix_sha256": array_sha(actions[:handoff["step"]].detach().cpu().numpy()),
            "handoff": handoff, "contact_threshold_N": CONTACT_N,
            "stable_steps_required": STABLE_STEPS,
            "pre_handoff_action_source": "OFFICIAL_LIBERO_DEMONSTRATION_REPLAY",
            "post_handoff_action_source": "NONE_IN_THIS_EXECUTABLE",
        }
        (destination / "SUPPLIED_GRASP_METADATA.json").write_text(json.dumps(clean(metadata), indent=2, sort_keys=True) + "\n")
        (destination / "REPLAY_TELEMETRY.json").write_text(json.dumps(clean(telemetry), indent=2) + "\n")
        print(json.dumps(clean(metadata), indent=2), flush=True)
        # Isaac Sim 5.1 can abort while destroying camera/Replicator objects.
        # All immutable evidence has been closed and flushed at this point;
        # terminate the isolated one-task process without invoking that path.
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
