#!/usr/bin/env python3
"""Load the frozen Isaac task scene and exit without reset or env.step."""

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
BASE = ROOT / "analysis/results"
TABERO = Path("/home/exouser/Tabero")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        ROOT,
        BASE / "current_runtime_recovery_v2_20260905",
        BASE / "current_runtime_core_snapshot_v2_20260905",
        BASE / "current_runtime_sensor_repair_v3_candidate_20260905",
        BASE / "current_multitask58_loader_candidate_20260905",
        BASE / "current_runtime_branch_execution_v6_20260905",
    ):
        sys.path.insert(0, str(path))

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    evidence = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": 0,
        "explicit_reset_calls": 0,
        "explicit_step_calls": 0,
        "physics_outcomes_generated": 0,
    }
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        module_path = TABERO / "analysis/p5s0c_paired_boundary_probe_value.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("preflight_task_sensors", module_path)
        p5 = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = p5
        spec.loader.exec_module(p5)
        p5.P4_COLLECT = ROOT / "activeforcing_current_probe.py"
        p4 = p5.import_p4_probe(0)

        setup_task_objects("libero_object", 0)
        cfg = parse_env_cfg(p4.ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        getattr(cfg.scene, "contact_grasp_" + p4.OBJ_NAME).max_contact_data_count_per_prim = 128

        free_before, total = torch.cuda.mem_get_info(0)
        env = gym.make(p4.ENV_ID, cfg=cfg).unwrapped
        torch.cuda.synchronize(0)
        free_after, _ = torch.cuda.mem_get_info(0)
        evidence.update(
            {
                "status": "PASS",
                "environment_id": p4.ENV_ID,
                "object": p4.OBJ_NAME,
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_total_mib": total // 1048576,
                "cuda_free_before_mib": free_before // 1048576,
                "cuda_free_after_scene_load_mib": free_after // 1048576,
                "scene_load_delta_mib": (free_before - free_after) // 1048576,
            }
        )
        return 0
    except Exception:
        evidence.update({"status": "FAIL", "error": traceback.format_exc()})
        return 1
    finally:
        if env is not None:
            env.close()
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
