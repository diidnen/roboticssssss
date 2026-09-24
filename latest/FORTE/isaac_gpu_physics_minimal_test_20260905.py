#!/usr/bin/env python3
"""Minimal no-camera Isaac GPU-physics smoke for runtime diagnosis only."""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ.setdefault("TABERO_ROOT", "/home/exouser/Tabero")
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=False, num_envs=1)
    app = launcher.app
    env = None
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        task_id = 0
        env_id = "Isaac-Cartpole-v0"
        import isaaclab_tasks  # noqa: F401
        cfg = parse_env_cfg(env_id, device="cuda:0", num_envs=1)
        cfg.recorders = {}
        cfg.episode_length_s = 10.0
        env = gym.make(env_id, cfg=cfg).unwrapped
        obs, _ = env.reset(seed=5050)
        action = torch.zeros((1, env.action_space.shape[0]), device=env.device, dtype=torch.float32)
        terminated = truncated = False
        for _ in range(10):
            obs, _, term, trunc, _ = env.step(action)
            terminated = bool(term[0].item())
            truncated = bool(trunc[0].item())
        result = {
            "status": "PASS",
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "env_device": str(env.device),
            "env_id": env_id,
            "camera_enabled": False,
            "physics_steps": 10,
            "terminated": terminated,
            "truncated": truncated,
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "python": sys.executable, "error": repr(exc)}, sort_keys=True))
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
