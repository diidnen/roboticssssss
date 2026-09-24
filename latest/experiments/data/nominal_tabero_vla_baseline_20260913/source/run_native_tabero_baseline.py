#!/usr/bin/env python3
"""Audited nominal frozen-Tabero-pi0 baseline.

The policy receives only the task prompt and live Tabero observations.  There is
no ActiveForcing query/probe and no action arbitration: every one of the first
13 postprocessed pi0 action dimensions is submitted to the environment exactly
as returned by the frozen policy server.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


TABERO = Path("/home/exouser/Tabero")
FORTE = Path("/home/exouser/FORTE")
SNAPSHOT = TABERO / "analysis/results/task_form_coverage_20260910/runtime/frozen_vla_server/SOURCE_SNAPSHOT"
RUNTIME_DIR = FORTE / "online_vla_restore_20260907"
HELPER = TABERO / "analysis/p6g1r1_controller_grasp_vla_handoff.py"
CHECKPOINT = Path("/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999")
EXPECTED_CHECKPOINT_SHA256 = "0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17"
TASKS = {
    1: {
        "object": "cream_cheese_1",
        "target": "basket_1",
        "prompt": "pick up the cream cheese and place it in the basket",
    },
    6: {
        "object": "butter_1",
        "target": "basket_1",
        "prompt": "pick up the butter and place it in the basket",
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean(value):
    if hasattr(value, "detach"):
        return clean(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(clean(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def set_friction(env, name: str, mu: float, torch):
    view = env.scene[name].root_physx_view
    props = view.get_material_properties().clone()
    props[..., :2] = float(mu)
    view.set_material_properties(props, torch.arange(props.shape[0], dtype=torch.int32))
    got = view.get_material_properties().detach().cpu().numpy()
    if not np.allclose(got[..., :2], mu, rtol=0.0, atol=1e-6):
        raise RuntimeError("object friction readback mismatch")
    return got


def basket_contact(env, target: str, obj: str) -> float:
    import torch

    try:
        matrix = env.scene[f"contact_{target}_{obj}"].data.force_matrix_w
        return float(torch.linalg.vector_norm(matrix.reshape(-1, 3)[0]).item())
    except Exception:
        return 0.0


def is_dropped(env) -> bool:
    try:
        return bool(env.termination_manager.get_term("object_1_dropped")[0].item())
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=18895)
    parser.add_argument("--task", type=int, choices=sorted(TASKS), default=1)
    parser.add_argument("--root", type=int, default=5108)
    parser.add_argument("--friction", type=float, default=0.5077986605848918)
    parser.add_argument("--horizon", type=int, default=350)
    parser.add_argument("--replan-steps", type=int, default=10)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    task = TASKS[args.task]
    protocol = {
        "method": "Nominal Frozen VLA / Native VLA",
        "task": args.task,
        "root_seed": args.root,
        "friction": args.friction,
        "prompt": task["prompt"],
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "policy_config": "pi0_lora_tacfield_tabero",
        "probe": False,
        "activeforcing": False,
        "force_or_gripper_override": False,
        "submitted_action": "raw postprocessed pi0 action[:, :13], byte-identical float32 copy",
        "commanded_force_N": None,
        "horizon": args.horizon,
        "replan_steps": args.replan_steps,
        "success_rule": "lift_dz>=0.03m and basket_contact>0.05N and no drop",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_new(out / "PROTOCOL.json", protocol)
    (out / "PROTOCOL.sha256").write_text(stable_hash(protocol) + "\n", encoding="utf-8")
    (out / "RUNNER.sha256").write_text(sha_file(Path(__file__).resolve()) + "\n", encoding="utf-8")

    os.environ.update({
        "HDF5_TRAJ_SOURCE_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5",
        "LIBERO_CONFIG_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/config",
        "LIBERO_ASSETS_DATA_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/USD",
        "TASK_SUITE": "libero_object",
        "TASK_ID": str(args.task),
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
    })
    sys.path[:0] = [
        str(SNAPSHOT), str(RUNTIME_DIR), str(TABERO), str(TABERO / "analysis"),
        str(FORTE), str(TABERO / "benchmarks/openpi/openpi-client/src"),
    ]

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    trace = []
    chunks = []
    error = None
    try:
        import torch

        helper = load_module("native_vla_helper", HELPER)
        runtime = load_module("native_vla_runtime", RUNTIME_DIR / "runtime.py")
        env, p6, p4 = helper.import_env_modules(args.task)
        env.reset(seed=args.root)
        friction_readback = set_friction(env, task["object"], args.friction, torch)
        helper.p6g1.settle_root_before_hash(env, p6, p4, helper.ROOT_SETTLE_STEPS)
        root_hash = stable_hash(env.scene.get_state(is_relative=True))
        initial_position, _ = helper.obj_pose(p4, env, args.task)
        initial_z = float(initial_position[2])

        plan = {
            "id": f"native_t{args.task}_root{args.root}_mu{args.friction:.9f}",
            "task": args.task,
            "object": task["object"],
            "target": task["target"],
            "root": args.root,
        }
        client = runtime.OnlineClient(out, plan, args.port)
        build, Buffer, adapter_sha = runtime.observation_builder()
        tactile = Buffer()
        write_new(out / "ROOT_AND_INTERFACE.json", {
            "root_state_sha256": root_hash,
            "friction_readback": friction_readback,
            "observation_adapter_sha256": adapter_sha,
            "prompt": task["prompt"],
        })

        done = False
        for start_step in range(1, args.horizon + 1, args.replan_steps):
            obs_now = env.observation_manager.compute()
            cameras = {
                name: env.scene[name].data.output["rgb"][0].detach().cpu().numpy().copy()
                for name in ("agentview_cam", "eye_in_hand_cam")
            }
            payload = build(env, obs_now, task["prompt"], tactile)
            payload = {k: v for k, v in payload.items() if isinstance(k, str)}
            action_chunk, proof = client.infer(payload, start_step, cameras)
            chunk13 = np.ascontiguousarray(action_chunk[:, :13], dtype=np.float32)
            chunks.append(chunk13.copy())
            for offset in range(min(args.replan_steps, args.horizon - start_step + 1)):
                step = start_step + offset
                raw_action = np.ascontiguousarray(chunk13[offset], dtype=np.float32)
                submitted_sha = hashlib.sha256(raw_action.tobytes()).hexdigest()
                obs, _, terminated, truncated, _ = env.step(
                    torch.from_numpy(raw_action.copy()).reshape(1, 13).to(env.device)
                )
                position, _ = helper.obj_pose(p4, env, args.task)
                bc = basket_contact(env, task["target"], task["object"])
                dropped = is_dropped(env)
                snap = helper.p6g1.contact_snapshot(
                    env, p6, p4, obs, args.task, plan["id"], "native_vla", step,
                    float(env.cfg.sim.dt) * int(env.cfg.decimation), 0,
                )
                native_success = False
                try:
                    native_success = bool(env.termination_manager.get_term("success")[0].item())
                except Exception:
                    pass
                trace.append({
                    "step": step,
                    "request_id": proof["request_id"],
                    "chunk_index": offset,
                    "submitted_action": raw_action.tolist(),
                    "submitted_action_bytes_sha256": submitted_sha,
                    "raw_gripper_action": float(raw_action[6]),
                    "raw_force_slots": raw_action[7:13].tolist(),
                    "object_position": position.tolist(),
                    "object_lift_dz_m": float(position[2] - initial_z),
                    "basket_contact_N": bc,
                    "measured_squeeze_N": float(snap["measured_squeeze_N"]),
                    "contact_state": snap["contact_state"],
                    "drop": dropped,
                    "native_success_term": native_success,
                    "terminated": bool(terminated[0].item()),
                    "truncated": bool(truncated[0].item()),
                })
                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    done = True
                    break
            if done:
                break

        np.savez_compressed(out / "POLICY_CHUNKS_RAW_EXECUTED.npz", **{
            f"chunk_{i:03d}": value for i, value in enumerate(chunks)
        })
        write_new(out / "ACTION_TRACE.json", trace)
        max_lift = max((row["object_lift_dz_m"] for row in trace), default=0.0)
        max_basket = max((row["basket_contact_N"] for row in trace), default=0.0)
        dropped_any = any(row["drop"] for row in trace)
        result = {
            "completed": True,
            "label_valid": bool(trace),
            "full_task_success_y": int(max_lift >= 0.03 and max_basket > 0.05 and not dropped_any),
            "native_success_observed": int(any(row["native_success_term"] for row in trace)),
            "lift_success": int(max_lift >= 0.03),
            "placement_contact_success": int(max_basket > 0.05),
            "drop": int(dropped_any),
            "steps": len(trace),
            "rpc_count": len(chunks),
            "max_lift_dz_m": max_lift,
            "max_basket_contact_N": max_basket,
            "mean_measured_squeeze_N": float(np.mean([row["measured_squeeze_N"] for row in trace])) if trace else None,
            "peak_measured_squeeze_N": max((row["measured_squeeze_N"] for row in trace), default=None),
            "commanded_force_N": None,
            "probe_used": False,
            "activeforcing_used": False,
            "action_override_used": False,
            "checkpoint_sha256": client.metadata["checkpoint_sha256"],
            "all_rpc_verified": client.calls == len(chunks),
            "error": None,
        }
        write_new(out / "RESULT.json", result)
        print("NATIVE_VLA_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        error = {"error": repr(exc), "traceback": traceback.format_exc(), "steps": len(trace)}
        write_new(out / "ERROR.json", error)
        print("NATIVE_VLA_ERROR=" + json.dumps(error, sort_keys=True), flush=True)
        return 1
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
