#!/usr/bin/env python3
"""Development-only frozen-VLA Fixed-5 gate from a supplied established grasp.

There is deliberately no scripted post-handoff motion, no ActiveForcing model,
and no task remapping.  A supplied Hybrid snapshot is restored, the unchanged
force-position controller receives a fixed 5 N target, and arm motion is from
the frozen online VLA at the native ten-step re-query cadence.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/exouser/Tabero")
FORTE = Path("/home/exouser/FORTE")
AF = FORTE / "online_vla_restore_20260907"
ROOT = REPO / "analysis/results/task_form_coverage_20260910"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
HORIZON = 350
FORCE_N = 5.0
MU = 0.6
TAIL = 20

TASKS = {
    "book": {
        "suite": "libero_10", "task": 5, "task_id": "libero_10/task5",
        "instruction": "pick up the book and place it in the back compartment of the caddy",
        "object": "black_book_1", "target": "desk_caddy_1", "release_required": True,
    },
    "wine": {
        "suite": "libero_goal", "task": 9, "task_id": "libero_goal/task9",
        "instruction": "put the wine bottle on the rack",
        "object": "wine_bottle_1", "target": "wine_rack_1", "release_required": True,
    },
}


def clean(x):
    if hasattr(x, "detach"):
        return clean(x.detach().cpu().numpy())
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(clean(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def set_friction(env, name: str, mu: float, torch):
    view = env.scene[name].root_physx_view
    props = view.get_material_properties().clone()
    props[..., :2] = float(mu)
    view.set_material_properties(props, torch.arange(props.shape[0], dtype=torch.int32))
    got = view.get_material_properties().detach().cpu().numpy()
    if not np.allclose(got[..., :2], mu, rtol=0, atol=1e-6):
        raise RuntimeError("object friction readback mismatch")
    return got


def quat_apply(q, v):
    q = np.asarray(q, dtype=np.float64); v = np.asarray(v, dtype=np.float64)
    w, x, y, z = q; qv = np.array([x, y, z])
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def force_snapshot(env, object_name: str) -> dict:
    """Same filtered-sensor bilateral normal-force measurement as P5."""
    mat = env.scene[f"contact_grasp_{object_name}"].data.force_matrix_w
    if mat is None or mat.ndim != 4 or mat.shape[1:] != (2, mat.shape[2], 3):
        raise RuntimeError(f"invalid filtered contact matrix: {None if mat is None else tuple(mat.shape)}")
    world = mat[0].sum(dim=1).detach().cpu().numpy().astype(float)
    left_q = env.scene["left_gripper_frame"].data.target_quat_w[0, 0].detach().cpu().numpy()
    right_q = env.scene["right_gripper_frame"].data.target_quat_w[0, 0].detach().cpu().numpy()
    li = quat_apply(np.array([left_q[0], -left_q[1], -left_q[2], -left_q[3]]), world[0])
    ri = quat_apply(np.array([right_q[0], -right_q[1], -right_q[2], -right_q[3]]), world[1])
    ln, rn = abs(float(li[2])), abs(float(ri[2]))
    return {"F_obj_left_world": world[0].tolist(), "F_obj_right_world": world[1].tolist(),
            "F_obj_left_normal_n": ln, "F_obj_right_normal_n": rn,
            "F_obj_bilateral_n": 2.0 * min(ln, rn)}


def scalar(value, default=0.0) -> float:
    try:
        return float(np.asarray(value.detach().cpu() if hasattr(value, "detach") else value).reshape(-1)[0])
    except Exception:
        return float(default)


def force_servo(command: float, measured: float, target: float) -> float:
    # Exact frozen current-controller constants: SERVO_STEP=.0006,
    # SERVO_DEADBAND=.4, D_CLOSED=0, D_OPEN=.04.
    if abs(target - measured) <= .4:
        return float(np.clip(command, 0.0, .04))
    return float(np.clip(command - .0006 if target > measured else command + .0006, 0.0, .04))


def controller_debug(env) -> dict:
    return getattr(env.action_manager.get_term("arm_action"), "debug_info", {}) or {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=sorted(TASKS))
    parser.add_argument("root", type=int)
    parser.add_argument("--port", type=int, default=18885)
    parser.add_argument("--repair-tag", default="")
    args = parser.parse_args()
    task = TASKS[args.candidate]
    suffix = f"_{args.repair_tag}" if args.repair_tag else ""
    job = ROOT / "development_fixed5" / args.candidate / f"root_{args.root}{suffix}"
    if job.exists():
        raise FileExistsError(f"refusing to overwrite {job}")
    job.mkdir(parents=True)
    source = ROOT / "supplied_grasps" / args.candidate / "hybrid_fixed5" / "SUPPLIED_ESTABLISHED_GRASP.pt"
    if not source.is_file():
        raise FileNotFoundError(source)

    os.environ.update({
        "HDF5_TRAJ_SOURCE_DIR": "/media/volume/newdata/exouser/task_form_coverage_20260910/assembled_hdf5",
        "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
        "TASK_SUITE": task["suite"], "TASK_ID": str(task["task"]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
    })
    sys.path[:0] = [str(REPO), str(AF), str(FORTE), str(REPO / "analysis")]
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    try:
        print("FIXED5_MARKER=AFTER_APP", flush=True)
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        from activeforcing_execution_snapshot import restore
        from arbitration import Arbitration
        print("FIXED5_MARKER=AFTER_IMPORTS", flush=True)

        runtime = import_file("task_form_frozen_runtime", AF / "runtime.py")
        print("FIXED5_MARKER=AFTER_RUNTIME_IMPORT", flush=True)
        runtime.INSTRUCTIONS = {task["task"]: task["instruction"]}
        setup_task_objects(task["suite"], task["task"])
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        getattr(cfg.scene, f"contact_grasp_{task['object']}").max_contact_data_count_per_prim = 128
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        print("FIXED5_MARKER=AFTER_ENV", flush=True)
        env.reset(seed=args.root)
        saved = torch.load(source, map_location="cpu", weights_only=False)
        # The qualified supplied snapshot includes its validated mu=.6
        # material state. Set it before strict exposed-state restoration so
        # restore can detect, rather than mask, a material mismatch.
        materials = set_friction(env, task["object"], MU, torch)
        restore(env, saved)
        print("FIXED5_MARKER=AFTER_RESTORE", flush=True)
        plan = {"id": f"{task['task_id'].replace('/', '_')}_r{args.root}", "task": task["task"],
                "object": task["object"], "target": task["target"], "root": args.root}
        # `_gripper_abs_cmd` records the raw normalized replay command (-1
        # means closed). The authoritative downstream arbitration consumes the
        # controller's physical aperture command, saved as `_last_d_cmd`.
        handoff = scalar(saved["objects"]["arm_action"].get("_last_d_cmd"), 0.0)
        if not 0.0 <= handoff <= .04:
            raise ValueError(f"invalid restored physical handoff command {handoff}")
        arb = Arbitration(FORCE_N, handoff)
        build, Buffer, adapter_sha = runtime.observation_builder()
        buffer = Buffer()
        client = runtime.OnlineClient(job, plan, args.port)
        chunk, proof = runtime.request_chunk(env, plan, client, build, buffer, 1)
        print("FIXED5_MARKER=AFTER_FIRST_RPC", flush=True)

        original_compute, original_reset = env.termination_manager.compute, env._reset_idx
        terminals = []
        def compute():
            original_compute()
            # Preserve only physical failure termination. Semantic success is
            # observed live and must not reset the scene before its 20-step
            # stability contract is evaluated.
            failures = torch.zeros_like(env.termination_manager.terminated)
            for name in env.termination_manager.active_terms:
                cfg_term = env.termination_manager.get_term_cfg(name)
                if name != "success" and not cfg_term.time_out:
                    failures.logical_or_(env.termination_manager.get_term(name))
            env.termination_manager._terminated_buf.copy_(failures)
            return env.termination_manager.dones
        def reset(ids):
            if len(ids):
                terminals.append(True)
        env.termination_manager.compute, env._reset_idx = compute, reset
        trace, error = [], None
        try:
            for step in range(1, HORIZON + 1):
                if step > 1 and (step - 1) % runtime.REPLAN_STEPS == 0:
                    chunk, proof = runtime.request_chunk(env, plan, client, build, buffer, step)
                action, arbitration = arb.action(chunk[(step - 1) % runtime.REPLAN_STEPS])
                env.step(torch.as_tensor(action, device=env.device).reshape(1, 13))
                contact = force_snapshot(env, task["object"])
                native = bool(env.termination_manager.get_term("success")[0])
                dropped = bool(env.termination_manager.get_term("object_1_dropped")[0])
                measured = float(contact["F_obj_bilateral_n"])
                trace.append({"step": step, "request_id": proof["request_id"], "action": action.tolist(),
                    "native_goal_reached": native, "grasp_retention_failure": dropped,
                    "release_intent": arbitration["vla_release_intent"],
                    "unheld": max(np.linalg.norm(contact["F_obj_left_world"]), np.linalg.norm(contact["F_obj_right_world"])) <= .05,
                    "measured_bilateral_squeeze_N": measured, "target_object_contact": contact,
                    "controller_debug": clean(controller_debug(env)), "physical_reset_requested": bool(terminals)})
                arb.feedback(scalar(controller_debug(env).get("f_sq_meas"), 0.0), arbitration["vla_release_intent"], force_servo)
                if terminals:
                    break
        except Exception as exc:
            import traceback
            error = {"error": repr(exc), "traceback": traceback.format_exc()}
        finally:
            env.termination_manager.compute, env._reset_idx = original_compute, original_reset

        tail = trace[-TAIL:]
        valid = error is None and len(trace) == HORIZON
        native_stable = len(tail) == TAIL and all(x["native_goal_reached"] for x in tail)
        released = (not task["release_required"]) or (len(tail) == TAIL and all(x["release_intent"] and x["unheld"] for x in tail))
        dropped = any(x["grasp_retention_failure"] for x in trace)
        success = bool(valid and native_stable and released and not dropped)
        failure = None if success else (
            "VLA_DOWNSTREAM_EXECUTION_FAILURE" if error else
            "GRASP_RETENTION_FAILURE" if dropped else
            "RELEASE_FAILURE" if native_stable and not released else
            "TERMINAL_GEOMETRIC_FAILURE")
        write(job / "ACTION_TRACE.json", trace)
        write(job / "FIXED5_DOWNSTREAM_RESULT.json", {
            "status": "QUALIFIED" if success else "NOT_QUALIFIED", "task": task, "root": args.root,
            "method": "Fixed-5", "force_setpoint_N": FORCE_N, "object_side_friction": MU,
            "source_snapshot": str(source), "source_snapshot_sha256": sha(source),
            "material_properties": materials, "horizon": HORIZON, "steps": len(trace),
            "online_vla_requery_steps": runtime.REPLAN_STEPS, "online_vla_rpc_calls": client.calls,
            "frozen_vla_checkpoint_sha256": client.metadata["checkpoint_sha256"],
            "observation_adapter_sha256": adapter_sha, "native_terminal_stable_last20": native_stable,
            "release_required": task["release_required"], "release_and_unheld_last20": released, "grasp_retention_failure": dropped,
            "full_task_success_y": int(success) if valid else None, "label_valid": valid,
            "failure_class": failure, "error": error,
            "post_handoff_action_source": "ONLINE_FROZEN_VLA_ONLY",
            "scripted_post_handoff_actions": False,
        })
        print(json.dumps({"candidate": args.candidate, "root": args.root, "success": success,
                          "steps": len(trace), "failure": failure}, indent=2), flush=True)
        return 0 if valid else 2
    except BaseException as exc:
        import traceback
        write(job / "STARTUP_OR_RUNTIME_ERROR.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        print(f"FIXED5_FATAL={exc!r}", flush=True)
        return 2
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
