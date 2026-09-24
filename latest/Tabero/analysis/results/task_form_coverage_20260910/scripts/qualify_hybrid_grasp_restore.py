#!/usr/bin/env python3
"""Transfer a demonstrated grasp into the authoritative Hybrid controller.

The replay snapshot contains only physical scene state.  This executable loads
that state into the current 13D Hybrid environment, holds the current EEF pose
while the unchanged controller establishes Fixed-5 tracking, then captures the
full exposed runtime/controller/cache/RNG state and verifies exact restore.
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
FORTE = Path("/home/exouser/FORTE")
ROOT = REPO / "analysis/results/task_form_coverage_20260910"
DATA = Path("/media/volume/newdata/exouser/task_form_coverage_20260910/assembled_hdf5")
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
TASKS = {
    "book": {"suite": "libero_10", "task": 5, "object": "black_book_1", "target": "desk_caddy_1"},
    "wine": {"suite": "libero_goal", "task": 9, "object": "wine_bottle_1", "target": "wine_rack_1"},
    "bowl": {"suite": "libero_spatial", "task": 4, "object": "akita_black_bowl_1", "target": "plate_1"},
}
FORCE_N = 5.0
MU = 0.6
CONTACT_N = 0.15
STABLE_STEPS = 5
MAX_HOLD_STEPS = 40


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def clean(value: Any) -> Any:
    if hasattr(value, "detach"): return clean(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def move_tensors(value: Any, device: Any) -> Any:
    """Match the established restore path: scene states must be device-local."""
    if hasattr(value, "to") and hasattr(value, "detach"):
        return value.detach().to(device=device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def axis_angle(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float); q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:]
    angle = 2 * np.arccos(np.clip(w, -1, 1)); scale = np.linalg.norm(xyz)
    return np.zeros(3, np.float32) if scale < 1e-8 else (xyz / scale * angle).astype(np.float32)


def set_object_friction(env, object_name: str, mu: float, torch) -> list:
    asset = env.scene[object_name]; props = asset.root_physx_view.get_material_properties().clone()
    props[..., 0] = mu; props[..., 1] = mu
    # Isaac's PhysX view expects a CPU int32 tensor of environment indices;
    # Python lists can stall the renderer callback path.
    asset.root_physx_view.set_material_properties(
        props, torch.arange(props.shape[0], dtype=torch.int32)
    )
    return clean(asset.root_physx_view.get_material_properties())


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("candidate", choices=sorted(TASKS)); args = parser.parse_args()
    task = TASKS[args.candidate]
    replay_dir = ROOT / "supplied_grasps" / args.candidate
    direct = replay_dir / "SUPPLIED_GRASP_SCENE_STATE.pt"
    if direct.exists(): source = direct
    else:
        candidates = sorted(replay_dir.glob("demo_*/SUPPLIED_GRASP_SCENE_STATE.pt"))
        if not candidates: raise FileNotFoundError(f"no captured replay grasp for {args.candidate}")
        source = candidates[0]
    out = replay_dir / "hybrid_fixed5"
    if out.exists(): raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    os.environ.update({
        "HDF5_TRAJ_SOURCE_DIR": str(DATA),
        "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
        "TASK_SUITE": task["suite"], "TASK_ID": str(task["task"]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
    })
    sys.path.insert(0, str(FORTE))
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        from activeforcing_execution_snapshot import capture, restore, max_difference

        setup_task_objects(task["suite"], task["task"])
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1); cfg.episode_length_s = 45.0
        sensor_name = f"contact_grasp_{task['object']}"
        getattr(cfg.scene, sensor_name).max_contact_data_count_per_prim = 128
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        saved = torch.load(source, map_location="cpu", weights_only=False)
        env.reset(seed=190000 + task["task"])
        ids = torch.tensor([0], device=env.device)
        print("HYBRID_RESTORE_MARKER=BEFORE_SCENE_RESET", flush=True)
        env.scene.reset_to(move_tensors(saved["scene_state"], env.device), ids, is_relative=True)
        print("HYBRID_RESTORE_MARKER=AFTER_SCENE_RESET", flush=True)
        env.scene.write_data_to_sim(); env.sim.forward()
        print("HYBRID_RESTORE_MARKER=AFTER_SIM_FORWARD", flush=True)
        materials = set_object_friction(env, task["object"], MU, torch)
        print("HYBRID_RESTORE_MARKER=AFTER_FRICTION", flush=True)
        telemetry = []; stable = 0
        for step in range(1, MAX_HOLD_STEPS + 1):
            print(f"HYBRID_RESTORE_MARKER=HOLD_STEP_BEGIN_{step}", flush=True)
            obs = env.observation_manager.compute()["policy"]
            eef = obs["eef_pose"][0].detach().cpu().numpy()
            action = np.zeros(13, np.float32); action[:3] = eef[:3]; action[3:6] = axis_angle(eef[3:7])
            action[6] = float(saved["source_action"][-1]); action[9] = FORCE_N / 2; action[12] = FORCE_N / 2
            obs2, _, terminated, truncated, _ = env.step(torch.as_tensor(action, device=env.device).reshape(1, 13))
            print(f"HYBRID_RESTORE_MARKER=HOLD_STEP_END_{step}", flush=True)
            matrix = env.scene[sensor_name].data.force_matrix_w[0].sum(dim=1).detach().cpu().numpy()
            left, right = float(np.linalg.norm(matrix[0])), float(np.linalg.norm(matrix[1]))
            bilateral = left >= CONTACT_N and right >= CONTACT_N
            stable = stable + 1 if bilateral else 0
            telemetry.append({"step": step, "left_object_force_norm_N": left, "right_object_force_norm_N": right,
                "bilateral": bilateral, "stable_run": stable, "object_pose": clean(env.scene[task["object"]].data.root_state_w[0, :7]),
                "robot_joint_position": clean(env.scene["robot"].data.joint_pos[0]),
                "controller_debug": clean(getattr(env.action_manager.get_term("arm_action"), "debug_info", {}))})
            if stable >= STABLE_STEPS: break
            if bool(terminated[0]) or bool(truncated[0]): break
        if stable < STABLE_STEPS:
            result = {"status": "FAILED_HYBRID_FIXED5_BILATERAL_HOLD", "task": task, "source": str(source),
                "source_sha256": sha(source), "steps": len(telemetry), "maximum_stable_run": max(x["stable_run"] for x in telemetry)}
            (out / "HYBRID_RESTORE_FAILURE.json").write_text(json.dumps(clean(result), indent=2, sort_keys=True) + "\n")
            (out / "HYBRID_HOLD_TELEMETRY.json").write_text(json.dumps(clean(telemetry), indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True); os._exit(2)
        full = capture(env); snapshot = out / "SUPPLIED_ESTABLISHED_GRASP.pt"; torch.save(full, snapshot)
        restore(env, full); restored = capture(env); difference = max_difference(full, restored)
        result = {"status": "QUALIFIED_SUPPLIED_ESTABLISHED_GRASP", "task": task,
            "source_replay_snapshot": str(source), "source_replay_snapshot_sha256": sha(source),
            "hybrid_snapshot": str(snapshot), "hybrid_snapshot_sha256": sha(snapshot),
            "force_setpoint_N": FORCE_N, "object_side_friction": MU, "material_properties": materials,
            "hold_steps": len(telemetry), "stable_bilateral_tail": STABLE_STEPS,
            "restore_max_exposed_difference": difference, "restore_exact": bool(difference == 0),
            "post_handoff_arm_motion": "ZERO_DISPLACEMENT_EEF_HOLD_ONLY; no downstream continuation",
            "controller": "AUTHORITATIVE_CURRENT_FORCE_POSITION_ACTION"}
        (out / "HYBRID_RESTORE_QUALIFICATION.json").write_text(json.dumps(clean(result), indent=2, sort_keys=True) + "\n")
        (out / "HYBRID_HOLD_TELEMETRY.json").write_text(json.dumps(clean(telemetry), indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    finally:
        app.close()


if __name__ == "__main__": raise SystemExit(main())
