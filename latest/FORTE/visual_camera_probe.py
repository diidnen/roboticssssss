#!/usr/bin/env python3
"""One-environment camera API smoke test for the prospective visual collector."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
simulation_app = app_launcher.app
print("after_app", flush=True)

try:
    import gymnasium as gym
    print("after_gym", flush=True)
    import tac_manip.tasks  # noqa: F401
    print("after_tasks", flush=True)
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tac_manip.utils.task_configs import setup_task_objects

    print("before_setup", flush=True)
    try:
        setup_task_objects("libero_object", 0)
    except BaseException as exc:
        print("setup_exception", type(exc).__name__, repr(exc), flush=True)
        raise
    print("after_setup", flush=True)
    try:
        env_cfg = parse_env_cfg("Isaac-Libero-Franka-Hybrid-Tactile-v0", device="cuda:0", num_envs=1)
        print("after_cfg", flush=True)
        env_cfg.episode_length_s = 45.0
        env = gym.make("Isaac-Libero-Franka-Hybrid-Tactile-v0", cfg=env_cfg).unwrapped
        print("after_make", flush=True)
        env.reset(seed=5100)
        print("after_reset", flush=True)
    except BaseException as exc:
        print("env_exception", type(exc).__name__, repr(exc), flush=True)
        raise
    env.sim.render()
    print("scene_keys", list(env.scene.keys()))
    print("sensor_keys", list(getattr(env.scene, "sensors", {}).keys()))
    for name in ["agentview_cam", "eye_in_hand_cam"]:
        sensor = env.scene[name]
        print("sensor", name, type(sensor).__name__)
        print("data_output_keys", list(sensor.data.output.keys()))
        for key, value in sensor.data.output.items():
            print("output", name, key, tuple(value.shape), str(value.dtype), value.device)
    out = Path("/tmp/visual_camera_probe")
    out.mkdir(parents=True, exist_ok=True)
    for name, filename in [("agentview_cam", "agentview.npy"), ("eye_in_hand_cam", "wrist.npy")]:
        arr = env.scene[name].data.output["rgb"][0].detach().cpu().numpy()
        import numpy as np
        np.save(out / filename, arr)
        print("saved", name, arr.shape, arr.dtype, int(arr.min()), int(arr.max()))
finally:
    simulation_app.close()
