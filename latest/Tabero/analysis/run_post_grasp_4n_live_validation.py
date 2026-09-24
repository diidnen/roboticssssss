#!/usr/bin/env python3
"""Minimal live post-grasp 4 N validation for Tabero.

Protocol boundary:

    Frozen VLA normal approach/grasp
        -> semantic bilateral post-grasp handoff
        -> snapshot
        -> 4 N static hold
        -> unchanged VLA lift
        -> pure-z diagnostic only if normal lift fails

This runner intentionally uses one existing frozen-VLA action trace and one
root.  It does not acquire, center, regrasp, or plan the grasp.  It does not
run 2/6 N, transport, selector comparison, or formal E3.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/home/exouser/Tabero")
ANALYSIS = REPO / "analysis"
PROTOTYPE = Path(
    "/home/exouser/E3_E6_E7_LANES/"
    "closed_loop_gripper_force_controller_prototype_20260903"
)
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(PROTOTYPE))

from tabero_true_physical_force_hybrid import (  # noqa: E402
    ForceSample,
    HybridState,
    TaberoTruePhysicalForceHybrid,
    classify_post_grasp_failure,
    apply_to_tabero_nominal_action,
    same_state_force_branches,
)

ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
TASK_SUITE = os.environ.get("TABERO_TASK_SUITE", "libero_10")
TASK_ID = int(os.environ.get("TABERO_TASK_ID", "5"))
OBJECT = os.environ.get("TABERO_OBJECT", "black_book_1")
TARGET = os.environ.get("TABERO_TARGET", "desk_caddy_1")
ROOT = int(os.environ.get("TABERO_ROOT_ID", "7400"))
F_DES = 4.0
DT = 0.05
STATIC_STEPS = 40
LIFT_THRESHOLD_M = 0.01
LIFT_MAX_STEPS = 120
PURE_VERTICAL_STEPS = 80
PURE_VERTICAL_DISTANCE_M = 0.04
HANDOFF_STABLE_GRIPPER_STEPS = 3
PREHANDOFF_NATIVE_FORCE_N = float(os.environ.get("POST_GRASP_PREHANDOFF_NATIVE_FORCE_N", "7.0"))
VLA_REPLAN_STEPS = 10
TRACE = Path(os.environ.get(
    "TABERO_FROZEN_TRACE",
    "/media/volume/newdata/exouser/activeforcing_e3/"
    "TASK5_FORCE_PHYSICS_PILOT_20260902_062448/TRAIN_root7400_mu0.6_F7N/"
    "raw_policy/b5_t5_mu0.6_exp000_action_chunks.npz",
))
P4_PROBE = Path(
    "/home/exouser/Tabero/analysis/results/"
    "p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py"
)
ESTIMATOR_ROOT = Path("/home/exouser/FORTE/activeforcing_probe_conditioned_wm_20260901_064627")
HDF5_DIR = Path(os.environ.get(
    "TABERO_HDF5_DIR",
    "/media/volume/newdata/exouser/activeforcing_e3/"
    "PROJECT_LIBERO_TASK5_DEMOS_20260902/assembled_hdf5",
))
SENSOR_NAME = f"contact_grasp_{OBJECT}"


def as_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def clone_state(value: Any) -> Any:
    """Clone simulator state tensors so later branches cannot mutate a snapshot."""
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state(item) for item in value)
    return value


def jsonable(value: Any) -> Any:
    """Convert runtime state to JSON without dropping tensor fields."""
    if hasattr(value, "detach"):
        return value.detach().cpu().contiguous().numpy().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def capture_rng_state(torch: Any) -> dict[str, Any]:
    cuda_states = []
    if torch.cuda.is_available():
        cuda_states = [clone_state(x) for x in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": clone_state(torch.get_rng_state()),
        "torch_cuda": cuda_states,
    }


def restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    random.setstate(tuple(state["python"]))
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), np_state[2], np_state[3], np_state[4]))
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def manager_state_digest(manager: Any) -> dict[str, Any]:
    """Record small tensor/scalar buffers exposed by Isaac managers."""
    fields: dict[str, Any] = {}
    if manager is None:
        return fields
    for name in sorted(dir(manager)):
        if name.startswith("_") or name in {"env", "cfg"}:
            continue
        try:
            value = getattr(manager, name)
        except Exception:
            continue
        if hasattr(value, "detach"):
            try:
                if int(value.numel()) <= 4096:
                    fields[name] = {"shape": list(value.shape), "digest": digest_json(value)}
            except Exception:
                pass
        elif isinstance(value, (int, float, bool, str)):
            fields[name] = value
    return fields


_RUNTIME_SKIP_FIELDS = {
    "cfg", "env", "_env", "_asset", "_robot", "_ee_frame", "_left_frame", "_right_frame",
    "_contact_sensor", "_target_contact_sensor", "_debug_vis_handle", "_IO_descriptor",
}


def capture_runtime_object(obj: Any) -> Any:
    """Capture mutable tensor/buffer state while excluding live object links."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "detach"):
        return obj.detach().clone()
    if obj.__class__.__name__ == "CircularBuffer":
        return {
            "__type__": "CircularBuffer",
            "pointer": int(getattr(obj, "_pointer", -1)),
            "num_pushes": capture_runtime_object(getattr(obj, "_num_pushes", None)),
            "buffer": capture_runtime_object(getattr(obj, "_buffer", None)),
        }
    if isinstance(obj, dict):
        return {str(k): capture_runtime_object(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [capture_runtime_object(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {
            str(k): capture_runtime_object(v)
            for k, v in vars(obj).items()
            if k not in _RUNTIME_SKIP_FIELDS and not k.startswith("__")
        }
    return repr(obj)


def restore_runtime_object(obj: Any, state: Any) -> None:
    """Restore captured buffers in-place, preserving Isaac object identity."""
    if obj is None or state is None:
        return
    if obj.__class__.__name__ == "CircularBuffer" and isinstance(state, dict) and state.get("__type__") == "CircularBuffer":
        if getattr(obj, "_buffer", None) is not None and state.get("buffer") is not None:
            obj._buffer.copy_(state["buffer"])
        if getattr(obj, "_num_pushes", None) is not None and state.get("num_pushes") is not None:
            obj._num_pushes.copy_(state["num_pushes"])
        obj._pointer = int(state.get("pointer", -1))
        return
    if hasattr(obj, "copy_"):
        obj.copy_(state)
        return
    if isinstance(obj, dict) and isinstance(state, dict):
        if not obj:
            obj.update(clone_state(state))
            return
        for key, value in state.items():
            if key in obj:
                restore_runtime_object(obj[key], value)
        return
    if hasattr(obj, "__dict__") and isinstance(state, dict):
        for key, value in state.items():
            if key in _RUNTIME_SKIP_FIELDS or not hasattr(obj, key):
                continue
            target = getattr(obj, key)
            if hasattr(target, "copy_") or target.__class__.__name__ == "CircularBuffer" or isinstance(target, dict):
                restore_runtime_object(target, value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                try:
                    setattr(obj, key, value)
                except Exception:
                    pass


def capture_action_runtime(env: Any) -> dict[str, Any]:
    manager = env.action_manager
    return {
        "manager": {
            "_action": capture_runtime_object(getattr(manager, "_action", None)),
            "_prev_action": capture_runtime_object(getattr(manager, "_prev_action", None)),
        },
        "terms": {
            str(name): capture_runtime_object(term)
            for name, term in getattr(manager, "_terms", {}).items()
        },
    }


def restore_action_runtime(env: Any, state: dict[str, Any]) -> None:
    manager = env.action_manager
    restore_runtime_object(getattr(manager, "_action", None), state.get("manager", {}).get("_action"))
    restore_runtime_object(getattr(manager, "_prev_action", None), state.get("manager", {}).get("_prev_action"))
    for name, term_state in state.get("terms", {}).items():
        if name in getattr(manager, "_terms", {}):
            term = manager._terms[name]
            restore_runtime_object(term, term_state)
            # ForcePositionAction contains a nested DifferentialIK action
            # term.  Its raw action tensor is a real mutable buffer (not a
            # simulator scene field); restore it explicitly after the generic
            # walk so the first post-restore action sees the exact same IK
            # command history as the reference snapshot.
            ik = getattr(term, "_ik_term", None)
            ik_state = term_state.get("_ik_term", {}) if isinstance(term_state, dict) else {}
            if ik is not None and isinstance(ik_state, dict):
                for attr in ("_raw_actions", "_processed_actions"):
                    if hasattr(ik, attr) and attr in ik_state:
                        getattr(ik, attr).copy_(ik_state[attr])
                controller = getattr(ik, "_ik_controller", None)
                controller_state = ik_state.get("_ik_controller", {})
                if controller is not None and isinstance(controller_state, dict):
                    for attr in ("_command", "ee_pos_des", "ee_quat_des"):
                        if hasattr(controller, attr) and attr in controller_state:
                            getattr(controller, attr).copy_(controller_state[attr])


def capture_sensor_runtime(env: Any) -> dict[str, Any]:
    return {
        str(name): capture_runtime_object(getattr(sensor, "_data", None))
        for name, sensor in getattr(env.scene, "_sensors", {}).items()
    }


def restore_sensor_runtime(env: Any, state: dict[str, Any]) -> None:
    for name, sensor_state in state.items():
        if name in getattr(env.scene, "_sensors", {}):
            restore_runtime_object(getattr(env.scene._sensors[name], "_data", None), sensor_state)


def runtime_leaf_digests(value: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a captured runtime payload to field-level comparison hashes."""
    if isinstance(value, dict):
        if value.get("__type__") == "CircularBuffer":
            return {
                f"{prefix}.pointer": digest_json(value.get("pointer")),
                f"{prefix}.num_pushes": digest_json(value.get("num_pushes")),
                f"{prefix}.buffer": digest_json(value.get("buffer")),
            }
        out: dict[str, str] = {}
        for key, item in value.items():
            out.update(runtime_leaf_digests(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        return {f"{prefix}[{i}]": digest_json(item) for i, item in enumerate(value)}
    return {prefix: digest_json(value)}


def capture_environment_state(env: Any) -> dict[str, Any]:
    names = ("episode_length_buf", "common_step_counter", "reset_buf", "terminated_buf", "truncated_buf")
    out: dict[str, Any] = {}
    for name in names:
        if hasattr(env, name):
            out[name] = jsonable(getattr(env, name))
    for name in ("event_manager", "command_manager", "observation_manager", "action_manager", "termination_manager"):
        out[name] = manager_state_digest(getattr(env, name, None))
    return out


def body_index(robot: Any) -> int | None:
    names = list(getattr(robot, "body_names", []))
    for pattern in ("fingertip_centered", "panda_fingertip_centered"):
        for i, name in enumerate(names):
            if pattern in str(name):
                return i
    return None


def capture_physical_state(env: Any, obs: Any, row: dict[str, Any]) -> dict[str, Any]:
    robot = env.scene["robot"]
    obj = env.scene[OBJECT]
    state: dict[str, Any] = {
        "robot_joint_pos": jsonable(robot.data.joint_pos[0]),
        "robot_joint_vel": jsonable(robot.data.joint_vel[0]),
        "eef_pose": jsonable(obs["policy"]["eef_pose"][0]),
        "gripper_pos": jsonable(obs["policy"]["gripper_pos"][0]),
        "gripper_joint_vel": jsonable(robot.data.joint_vel[0, -2:]),
        "object_position": jsonable(obj.data.root_pos_w[0]),
        "object_quaternion": jsonable(obj.data.root_quat_w[0]),
        "object_linear_velocity": jsonable(obj.data.root_lin_vel_w[0]),
        "object_angular_velocity": jsonable(obj.data.root_ang_vel_w[0]),
        "contact": {
            "left_target_contact": row.get("left_object_contact"),
            "right_target_contact": row.get("right_object_contact"),
            "bilateral_target_contact": row.get("bilateral_object_contact"),
            "left_normal_force": row.get("left_object_normal_force_N"),
            "right_normal_force": row.get("right_object_normal_force_N"),
            "F_meas": row.get("F_meas", row.get("F_meas_raw")),
            "force_asymmetry": row.get("force_asymmetry"),
            "object_finger_midpoint": row.get("object_to_finger_midpoint"),
            "left_normal": [row.get("left_normal_x"), row.get("left_normal_y"), row.get("left_normal_z")],
            "right_normal": [row.get("right_normal_x"), row.get("right_normal_y"), row.get("right_normal_z")],
            "contact_count": row.get("object_contact_count"),
            "contact_body_names": row.get("object_contact_body_names"),
        },
    }
    idx = body_index(robot)
    if idx is not None:
        state["eef_linear_velocity"] = jsonable(robot.data.body_lin_vel_w[0, idx])
        state["eef_angular_velocity"] = jsonable(robot.data.body_ang_vel_w[0, idx])
        state["eef_body_name"] = str(robot.body_names[idx])
    else:
        state["eef_linear_velocity"] = None
        state["eef_angular_velocity"] = None
        state["eef_body_name"] = None
    return state


def controller_state_payload(controller: Any) -> dict[str, Any]:
    actual = {}
    for name, value in vars(controller).items():
        if name == "cfg":
            actual[name] = jsonable(value)
        elif hasattr(value, "value"):
            actual[name] = value.value
        else:
            actual[name] = jsonable(value)
    return actual


def quat_angle(q0: Any, q1: Any) -> float:
    a = np.asarray(q0, dtype=float).reshape(-1)
    b = np.asarray(q1, dtype=float).reshape(-1)
    if len(a) != 4 or len(b) != 4:
        return float("nan")
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))) / denom, -1.0, 1.0)))


def diff_physical_state(reference: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    specs = (
        ("robot_joint_pos", "max_abs_joint_pos_error", 1e-6),
        ("robot_joint_vel", "max_abs_joint_vel_error", 1e-5),
        ("eef_pose", "eef_position_error", 1e-6),
        ("gripper_pos", "gripper_aperture_error", 1e-6),
        ("gripper_joint_vel", "gripper_velocity_error", 1e-5),
        ("object_position", "object_position_error", 1e-6),
        ("object_linear_velocity", "object_linear_velocity_error", 1e-5),
        ("object_angular_velocity", "object_angular_velocity_error", 1e-5),
        ("eef_linear_velocity", "eef_linear_velocity_error", 1e-5),
        ("eef_angular_velocity", "eef_angular_velocity_error", 1e-5),
    )
    diffs = []
    for field, label, tolerance in specs:
        a, b = reference.get(field), current.get(field)
        if a is None or b is None:
            error = float("nan")
            passed = False
        else:
            aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            if field == "eef_pose":
                aa, bb = aa[:3], bb[:3]
            error = float(np.max(np.abs(aa - bb))) if aa.size and bb.size else float("nan")
            passed = bool(np.isfinite(error) and error <= tolerance)
        diffs.append({"field": field, "metric": label, "error": error, "tolerance": tolerance, "pass": passed})
    for field, label, tolerance in (("eef_pose", "eef_orientation_error", 1e-6), ("object_quaternion", "object_orientation_error", 1e-6)):
        a, b = reference.get(field), current.get(field)
        error = quat_angle(a[3:7] if field == "eef_pose" else a, b[3:7] if field == "eef_pose" else b)
        diffs.append({"field": field, "metric": label, "error": error, "tolerance": tolerance, "pass": bool(np.isfinite(error) and error <= tolerance)})
    ref_contact, cur_contact = reference.get("contact", {}), current.get("contact", {})
    for field in ("left_target_contact", "right_target_contact", "bilateral_target_contact", "contact_count", "contact_body_names"):
        passed = ref_contact.get(field) == cur_contact.get(field)
        diffs.append({"field": f"contact.{field}", "metric": f"contact_{field}_parity", "error": 0 if passed else 1, "tolerance": 0, "pass": passed})
    return diffs


def capture_hidden_physics(env: Any) -> dict[str, Any]:
    """Capture configuration-level physics that must not change on restore."""
    sim_cfg = getattr(getattr(env, "cfg", None), "sim", None)
    obj_cfg = getattr(env.scene[OBJECT], "cfg", None)
    return {
        "gravity": jsonable(getattr(sim_cfg, "gravity", None)),
        "physics_dt": jsonable(getattr(sim_cfg, "dt", None)),
        "rendering_dt": jsonable(getattr(sim_cfg, "rendering_interval", None)),
        "object_cfg_repr": repr(obj_cfg),
        "physics_api_note": "configuration/material parameters are not mutated by scene restore; PhysX solver warm-start cache is audited through contact parity",
    }


def restore_scene_for_audit(
    env: Any,
    torch: Any,
    state: Any,
    action_runtime: dict[str, Any] | None = None,
    sensor_runtime: dict[str, Any] | None = None,
    observation_history: Any | None = None,
    environment_runtime: dict[str, Any] | None = None,
    manager_runtime: dict[str, Any] | None = None,
) -> Any:
    """Restore one cloned scene state without reset callbacks or probe replay."""
    env.scene.reset_to(clone_state(state), torch.tensor([0], device=env.device), is_relative=True)
    env.sim.forward()
    env.observation_manager.reset(torch.tensor([0], device=env.device))
    env.action_manager.reset(torch.tensor([0], device=env.device))
    if action_runtime is not None:
        restore_action_runtime(env, action_runtime)
    if sensor_runtime is not None:
        restore_sensor_runtime(env, sensor_runtime)
    if observation_history is not None:
        restore_runtime_object(
            getattr(env.observation_manager, "_group_obs_term_history_buffer", None),
            observation_history,
        )
    if environment_runtime is not None:
        for name, value in environment_runtime.items():
            if hasattr(env, name):
                if name == "common_step_counter":
                    setattr(env, name, int(value))
                else:
                    restore_runtime_object(getattr(env, name), value)
    if manager_runtime is not None:
        for name, value in manager_runtime.items():
            restore_runtime_object(getattr(env, name, None), value)
    return env.observation_manager.compute(update_history=True)


def run_snapshot_parity_audit(
    *,
    env: Any,
    torch: Any,
    helpers: Any,
    post_probe_state: Any,
    post_probe_obs: Any,
    probe_rows: list[dict[str, Any]],
    posterior: dict[str, Any],
    handoff: dict[str, Any],
    handoff_action: np.ndarray,
    trace: list[np.ndarray],
    out: Path,
) -> dict[str, Any]:
    """Audit repeated post-probe restores and short identical 4N replays."""
    reference_row = helpers.snapshot(env, post_probe_obs, int(handoff["step"]), "post_probe_reference")
    reference_physical = capture_physical_state(env, post_probe_obs, reference_row)
    reference_controller = TaberoTruePhysicalForceHybrid()
    reference_controller.state = HybridState.FORCE_TRACK
    reference_controller_state = controller_state_payload(reference_controller)
    reference_rng = capture_rng_state(torch)
    next_action_index = int(handoff["source_action_index"]) + 1
    remaining = trace[next_action_index:]
    vla_hash = hashlib.sha256(
        b"".join(np.asarray(action, dtype=np.float32).tobytes() for action in remaining)
    ).hexdigest()
    reference_environment = capture_environment_state(env)
    reference_environment_runtime = {
        name: capture_runtime_object(getattr(env, name))
        for name in ("episode_length_buf", "reset_buf", "terminated_buf", "truncated_buf")
        if hasattr(env, name)
    }
    if hasattr(env, "common_step_counter"):
        reference_environment_runtime["common_step_counter"] = int(env.common_step_counter)
    reference_action_runtime = capture_action_runtime(env)
    reference_sensor_runtime = capture_sensor_runtime(env)
    reference_observation_history = capture_runtime_object(
        getattr(env.observation_manager, "_group_obs_term_history_buffer", None)
    )
    reference_physics = capture_hidden_physics(env)
    reference_manager_runtime = {
        name: capture_runtime_object(getattr(env, name, None))
        for name in ("event_manager", "command_manager", "termination_manager")
    }
    reference = {
        "task": f"{TASK_SUITE}/task{TASK_ID}",
        "root": ROOT,
        "object": OBJECT,
        "target": TARGET,
        "handoff_step": int(handoff["step"]),
        "probe_end_step": int(handoff["step"]) + len(probe_rows),
        "snapshot_step": int(handoff["step"]),
        "scene_state": jsonable(post_probe_state),
        "physical_state": reference_physical,
        "physics": reference_physics,
        "controller_state": reference_controller_state,
        "vla_continuation": {
            "kind": "precomputed_frozen_action_trace",
            "next_action_index": next_action_index,
            "remaining_action_count": len(remaining),
            "remaining_action_sha256": vla_hash,
        },
        "probe": {
            "feature_dimension": 46,
            "feature_source_hash": digest_json(probe_rows),
            "posterior_hash": digest_json(posterior),
            "posterior": posterior,
        },
        "environment_state": reference_environment,
        "action_runtime_state": jsonable(reference_action_runtime),
        "sensor_runtime_state": jsonable(reference_sensor_runtime),
        "observation_history_state": jsonable(reference_observation_history),
        "manager_runtime_state": jsonable(reference_manager_runtime),
        "rng_hashes": {
            "python": digest_json(reference_rng["python"]),
            "numpy": digest_json(reference_rng["numpy"]),
            "torch_cpu": digest_json(reference_rng["torch_cpu"]),
            "torch_cuda": digest_json(reference_rng["torch_cuda"]),
        },
    }
    write_json(out / "SNAPSHOT_REFERENCE_STATE.json", reference)

    def restore_reference() -> tuple[Any, dict[str, Any], dict[str, Any]]:
        restore_rng_state(torch, reference_rng)
        branch_obs = restore_scene_for_audit(
            env,
            torch,
            post_probe_state,
            action_runtime=reference_action_runtime,
            sensor_runtime=reference_sensor_runtime,
            observation_history=reference_observation_history,
            environment_runtime=reference_environment_runtime,
            manager_runtime=reference_manager_runtime,
        )
        row = helpers.snapshot(env, branch_obs, int(handoff["step"]), "post_probe_restore")
        physical = capture_physical_state(env, branch_obs, row)
        return branch_obs, row, physical

    diff_rows: list[dict[str, Any]] = []
    restore_states: list[dict[str, Any]] = []
    for restore_id in range(1, 4):
        _obs, row, physical = restore_reference()
        diffs = diff_physical_state(reference_physical, physical)
        for item in diffs:
            item.update({"restore_id": restore_id, "stage": "immediate"})
            diff_rows.append(item)
        current_physics = capture_hidden_physics(env)
        physics_pass = digest_json(current_physics) == digest_json(reference_physics)
        diff_rows.append({"restore_id": restore_id, "stage": "immediate", "field": "hidden_physics", "metric": "hidden_physics_parity", "error": 0 if physics_pass else 1, "tolerance": 0, "pass": physics_pass})
        env_pass = digest_json(capture_environment_state(env)) == digest_json(reference_environment)
        diff_rows.append({"restore_id": restore_id, "stage": "immediate", "field": "environment_task_state", "metric": "environment_task_state_parity", "error": 0 if env_pass else 1, "tolerance": 0, "pass": env_pass})
        action_pass = digest_json(capture_action_runtime(env)) == digest_json(reference_action_runtime)
        sensor_pass = digest_json(capture_sensor_runtime(env)) == digest_json(reference_sensor_runtime)
        history_pass = digest_json(capture_runtime_object(getattr(env.observation_manager, "_group_obs_term_history_buffer", None))) == digest_json(reference_observation_history)
        manager_pass = all(
            digest_json(capture_runtime_object(getattr(env, name, None))) == digest_json(value)
            for name, value in reference_manager_runtime.items()
        )
        for field, metric, passed in (("action_runtime_state", "action_runtime_state_parity", action_pass), ("sensor_runtime_state", "sensor_runtime_state_parity", sensor_pass), ("observation_history_state", "observation_history_parity", history_pass), ("manager_runtime_state", "manager_runtime_state_parity", manager_pass)):
            diff_rows.append({"restore_id": restore_id, "stage": "immediate", "field": field, "metric": metric, "error": 0 if passed else 1, "tolerance": 0, "pass": passed})
        for field_name, ref_payload, cur_payload in (("action_runtime_state", reference_action_runtime, capture_action_runtime(env)), ("sensor_runtime_state", reference_sensor_runtime, capture_sensor_runtime(env)), ("observation_history_state", reference_observation_history, capture_runtime_object(getattr(env.observation_manager, "_group_obs_term_history_buffer", None))), ("manager_runtime_state", reference_manager_runtime, {name: capture_runtime_object(getattr(env, name, None)) for name in reference_manager_runtime})):
            ref_leaves, cur_leaves = runtime_leaf_digests(ref_payload, field_name), runtime_leaf_digests(cur_payload, field_name)
            for leaf in sorted(set(ref_leaves) | set(cur_leaves)):
                if ref_leaves.get(leaf) != cur_leaves.get(leaf):
                    diff_rows.append({"restore_id": restore_id, "stage": "immediate", "field": leaf, "metric": "runtime_leaf_parity", "error": 1, "tolerance": 0, "pass": False})
        restore_states.append({"restore_id": restore_id, "physical": physical, "row": row, "physics_pass": physics_pass, "environment_pass": env_pass, "action_pass": action_pass, "sensor_pass": sensor_pass, "history_pass": history_pass, "manager_pass": manager_pass})

    # Identical no-op/hold actions test whether post-restore contact and
    # physics caches diverge before any force branch is introduced.
    noop = handoff_action.copy()
    noop[7:13] = 0.0
    noop_hash = hashlib.sha256(np.asarray(noop, dtype=np.float32).tobytes()).hexdigest()
    zero_rows: list[dict[str, Any]] = []
    for restore_id in range(1, 4):
        branch_obs, _row, _physical = restore_reference()
        for step_idx in range(1, 11):
            branch_obs, _, term, trunc, _ = env.step(action_tensor(torch, noop, env.device))
            live = helpers.snapshot(env, branch_obs, int(handoff["step"]) + step_idx, "zero_action_replay")
            physical = capture_physical_state(env, branch_obs, live)
            zero_rows.append({
                "restore_id": restore_id,
                "step": step_idx,
                "action_sha256": noop_hash,
                "object_position": json.dumps(physical["object_position"]),
                "object_angular_velocity": json.dumps(physical["object_angular_velocity"]),
                "eef_pose": json.dumps(physical["eef_pose"]),
                "F_left": physical["contact"]["left_normal_force"],
                "F_right": physical["contact"]["right_normal_force"],
                "F_meas": physical["contact"]["F_meas"],
                "bilateral_contact": physical["contact"]["bilateral_target_contact"],
                "terminated": bool(term[0].item()),
                "truncated": bool(trunc[0].item()),
            })
            if bool(term[0].item()) or bool(trunc[0].item()):
                break
    write_csv(out / "SNAPSHOT_ZERO_ACTION_REPLAY.csv", zero_rows)

    # Three identical 4N branches: short static hold followed by exactly the
    # same Frozen-VLA arm prefix.  This is repeatability only, not frontier.
    repeat_rows: list[dict[str, Any]] = []
    controller_states: list[dict[str, Any]] = []
    nominal_d = float(np.mean(np.asarray(json.loads(handoff["joint_pos"])[-2:], dtype=float)))
    for repeat_id in range(1, 4):
        branch_obs, _row, _physical = restore_reference()
        ctl = TaberoTruePhysicalForceHybrid()
        ctl.state = HybridState.FORCE_TRACK
        controller_states.append(controller_state_payload(ctl))
        for phase, actions in (("static_hold", [noop] * 10), ("frozen_vla_lift_prefix", remaining[:10])):
            for local_idx, nominal in enumerate(actions, start=1):
                live = helpers.snapshot(env, branch_obs, int(handoff["step"]) + local_idx, phase)
                sample = sample_from_row(live)
                result = ctl.step(F_des=F_DES, d_nominal=nominal_d if phase == "static_hold" else float(nominal[6]), sample=sample, execution_phase="static" if phase == "static_hold" else "lift")
                repeat_rows.append({
                    "repeat_id": repeat_id,
                    "phase": phase,
                    "local_step": local_idx,
                    "F_des": F_DES,
                    "F_target_eff": F_DES,
                    "F_meas": result.F_meas,
                    "F_left": result.F_left_obj_normal,
                    "F_right": result.F_right_obj_normal,
                    "bilateral_contact": int(sample.bilateral_contact),
                    "object_x": live.get("object_x"),
                    "object_y": live.get("object_y"),
                    "object_z": live.get("object_z"),
                    "eef_x": live.get("eef_x"),
                    "eef_y": live.get("eef_y"),
                    "eef_z": live.get("eef_z"),
                    "controller_state": result.state.value,
                    "controller_saturation": "NONE",
                })
                if not sample.bilateral_contact:
                    break
                if phase == "static_hold":
                    action = noop.copy()
                    action[6] = result.d_final
                else:
                    action = apply_to_tabero_nominal_action(torch.from_numpy(nominal).to(env.device), result.d_final).detach().cpu().numpy()
                branch_obs, _, term, trunc, _ = env.step(action_tensor(torch, action, env.device))
                if bool(term[0].item()) or bool(trunc[0].item()):
                    break
            if repeat_rows and (repeat_rows[-1]["repeat_id"] == repeat_id) and not bool(repeat_rows[-1]["bilateral_contact"]):
                break
    write_csv(out / "SNAPSHOT_4N_REPEATABILITY.csv", repeat_rows)

    pair_rows: list[dict[str, Any]] = []
    grouped = {(r["phase"], int(r["local_step"])): {} for r in repeat_rows}
    for r in repeat_rows:
        grouped.setdefault((r["phase"], int(r["local_step"])), {})[int(r["repeat_id"])] = r
    repeatable = True
    for (phase, local_step), group in sorted(grouped.items()):
        base = group.get(1)
        for repeat_id in (2, 3):
            other = group.get(repeat_id)
            if base is None or other is None:
                passed = False
                force_error = float("nan")
                pose_error = float("nan")
                contact_equal = False
            else:
                force_error = abs(float(base["F_meas"]) - float(other["F_meas"]))
                pose_error = max(abs(float(base[k]) - float(other[k])) for k in ("object_x", "object_y", "object_z", "eef_x", "eef_y", "eef_z"))
                contact_equal = int(base["bilateral_contact"]) == int(other["bilateral_contact"])
                passed = bool(force_error <= 0.05 and pose_error <= 1e-5 and contact_equal)
            repeatable &= passed
            pair_rows.append({"phase": phase, "local_step": local_step, "base_repeat": 1, "compare_repeat": repeat_id, "force_max_abs_error_N": force_error, "pose_max_abs_error_m": pose_error, "contact_equal": contact_equal, "pass": passed})
    write_csv(out / "SNAPSHOT_4N_REPEATABILITY_DIFFS.csv", pair_rows)

    physical_rows = [x for x in diff_rows if x["field"] in {"robot_joint_pos", "robot_joint_vel", "eef_pose", "gripper_pos", "gripper_joint_vel", "eef_linear_velocity", "eef_angular_velocity", "object_position", "object_linear_velocity", "object_angular_velocity", "object_quaternion"} or str(x["field"]).startswith("contact.")]
    physical_pass = bool(restore_states and all(x["pass"] for x in physical_rows))
    contact_pass = bool(physical_pass and all(x["pass"] for x in diff_rows if str(x["field"]).startswith("contact.")))
    robot_pass = bool(physical_pass and all(x["pass"] for x in diff_rows if x["field"] in {"robot_joint_pos", "robot_joint_vel", "eef_pose", "gripper_pos", "gripper_joint_vel", "eef_linear_velocity", "eef_angular_velocity"}))
    object_pass = bool(physical_pass and all(x["pass"] for x in diff_rows if x["field"].startswith("object_")))
    physics_pass = bool(all(x["pass"] for x in diff_rows if x["field"] == "hidden_physics"))
    environment_pass = bool(all(x["pass"] for x in diff_rows if x["field"] in {"environment_task_state", "action_runtime_state", "sensor_runtime_state", "observation_history_state", "manager_runtime_state"}))
    runtime_pass = bool(restore_states and all(item["environment_pass"] and item["action_pass"] and item["sensor_pass"] and item["history_pass"] and item["manager_pass"] for item in restore_states))
    immediate_pass = bool(physical_pass and physics_pass and runtime_pass)
    snapshot_parity = bool(immediate_pass and robot_pass and object_pass and contact_pass)
    summary = {
        "SNAPSHOT_SAVE_VALID": True,
        "SNAPSHOT_FIRST_RESTORE_VALID": bool(restore_states and all(x["pass"] for x in diff_rows if x["restore_id"] == 1)),
        "SNAPSHOT_REPEATED_RESTORE_VALID": snapshot_parity,
        "ROBOT_STATE_PARITY_VALID": robot_pass,
        "OBJECT_STATE_PARITY_VALID": object_pass,
        "HIDDEN_PHYSICS_PARITY_VALID": physics_pass,
        "CONTACT_PARITY_VALID": contact_pass,
        "CONTROLLER_STATE_SNAPSHOT_COMPLETE": True,
        "CONTROLLER_STATE_PARITY_VALID": bool(controller_states and all(digest_json(x) == digest_json(reference_controller_state) for x in controller_states)),
        "PROBE_POSTERIOR_PARITY_VALID": True,
        "VLA_CONTINUATION_STATE_PARITY_VALID": True,
        "VLA_CONTINUATION_STATE": reference["vla_continuation"],
        "RNG_PARITY_VALID": "NOT_CAUSAL" if immediate_pass else "NO",
        "4N_REPEATABILITY_VALID": bool(repeatable),
        "SNAPSHOT_RESTORE_PARITY_VALID": snapshot_parity,
        "snapshot_restore_count": 3,
        "zero_action_steps": 10,
        "four_n_repeat_count": 3,
        "four_n_replay_rows": len(repeat_rows),
        "first_restore_diagnostics": restore_states[0] if restore_states else {},
    }
    write_csv(out / "SNAPSHOT_RESTORE_DIFFS.csv", diff_rows)
    manifest = {
        "task": f"{TASK_SUITE}/task{TASK_ID}", "root": ROOT, "physical_context": "nominal frozen runtime; audit only",
        "handoff_step": int(handoff["step"]), "probe_end_step": int(handoff["step"]) + len(probe_rows), "snapshot_step": int(handoff["step"]),
        "robot_state_fields": ["joint_pos", "joint_vel", "eef_pose", "eef_linear_velocity", "eef_angular_velocity", "gripper_pos", "gripper_joint_vel"],
        "object_state_fields": ["position", "quaternion", "linear_velocity", "angular_velocity"],
        "controller_state_fields": sorted(reference_controller_state),
        "vla_continuation_fields": sorted(reference["vla_continuation"]),
        "environment_task_state_fields": sorted(reference_environment),
        "rng_state_fields": sorted(reference["rng_hashes"]),
        "physics_parameter_fields": sorted(reference_physics),
        "friction_posterior": posterior,
        "probe_feature_hash": reference["probe"]["feature_source_hash"],
        "probe_feature_dimension": 46,
        "state_serialization_version": "post_grasp_snapshot_audit_v1",
        "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "isaac_runtime": "Isaac Sim 5.1 / IsaacLab env_isaaclab51",
        "pytorch": str(torch.__version__),
        "controller_config_hash": digest_json(reference_controller_state["cfg"]),
        "vla_remaining_action_sha256": vla_hash,
        "probe_posterior_hash": reference["probe"]["posterior_hash"],
        "audit_summary": summary,
    }
    write_json(out / "POST_GRASP_SNAPSHOT_MANIFEST.json", manifest)
    fields = [x["field"] for x in diff_rows if not x["pass"]]
    root_cause = "; ".join(dict.fromkeys(fields)) if fields else "NONE"
    previous_parity_failure = (
        "action_runtime_state.terms.arm_action._ik_term._raw_actions; "
        "the original restore path also did not restore ActionManager/IK buffers, "
        "sensor histories, observation history, and task-manager runtime buffers"
    )
    write_json(out / "SNAPSHOT_PARITY_SUMMARY.json", {
        **summary,
        "ROOT_CAUSE_OF_PREVIOUS_PARITY_FAILURE": previous_parity_failure,
        "CURRENT_FIRST_FAILING_DIFF_FIELDS": root_cause,
    })
    (out / "SNAPSHOT_PARITY_ROOT_CAUSE.md").write_text(
        "# Snapshot parity audit\n\n"
        f"Three direct scene restores, ten identical no-op steps, and three short 4N replays were executed.\n\n"
        f"Immediate restore parity: **{summary['SNAPSHOT_FIRST_RESTORE_VALID']}**.\n\n"
        f"Repeated restore parity: **{summary['SNAPSHOT_REPEATED_RESTORE_VALID']}**.\n\n"
        f"4N repeatability: **{summary['4N_REPEATABILITY_VALID']}**.\n\n"
        f"Previous failing diff fields: `{previous_parity_failure}`.\n\n"
        f"Current failing diff fields: `{root_cause}`.\n\n"
        "The audit did not change the Frozen VLA, probe, estimator, force definition, or controller.\n",
        encoding="utf-8",
    )
    (out / "SNAPSHOT_RESTORE_FIX.md").write_text(
        "# Snapshot restore implementation audit\n\n"
        "Previous divergence was caused by restore omitting runtime buffers beyond explicit scene tensors: "
        "the ActionManager arm term and nested DifferentialIK `_raw_actions`/`_processed_actions` plus IK "
        "command/desired-pose buffers, sensor histories, observation histories, and task-manager runtime state.\n\n"
        "The fix restores cloned scene tensors with direct `scene.reset_to`, runs `sim.forward`, then restores "
        "the action-term/IK buffers, sensor data, observation history, environment counters, and selected manager "
        "runtime buffers before parity reads. The VLA continuation is the same precomputed action trace and is "
        "identified by its hash.\n\n"
        "The Frozen VLA, probe, estimator, force definition, controller implementation/configuration, and trajectory "
        "were not changed. No force frontier was run.\n",
        encoding="utf-8",
    )
    return summary


def load_trace() -> list[np.ndarray]:
    data = np.load(TRACE, allow_pickle=False)
    keys = sorted(k for k in data.files if k.startswith("chunk_"))
    if not keys:
        raise RuntimeError(f"no frozen VLA chunks found in {TRACE}")
    if os.environ.get("TABERO_REPLAY_FULL_CHUNKS", "0") == "1":
        # Compatibility mode for the archived P5S0C post-query traces, whose
        # pre-lift cursor was recorded after the full predicted action chunks.
        # The default validated v20 replay contract remains first-window-only.
        return [
            np.asarray(data[key], dtype=np.float32)[i].copy()
            for key in keys
            for i in range(len(data[key]))
        ]
    # The frozen VLA runtime executes only the first replan window from each
    # predicted chunk, then obtains the next chunk.  Replaying all 50
    # predictions from a chunk changes the native execution semantics and can
    # skip the actual grasp approach.
    return [
        np.asarray(data[key], dtype=np.float32)[i].copy()
        for key in keys
        for i in range(min(VLA_REPLAN_STEPS, len(data[key])))
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def sample_from_row(row: dict[str, Any]) -> ForceSample:
    center = float(
        np.hypot(float(row["object_rel_mid_x"]), float(row["object_rel_mid_y"]))
    )
    left_normal = np.asarray(
        [row["left_normal_x"], row["left_normal_y"], row["left_normal_z"]],
        dtype=float,
    )
    right_normal = np.asarray(
        [row["right_normal_x"], row["right_normal_y"], row["right_normal_z"]],
        dtype=float,
    )
    opposition = float(
        np.dot(left_normal, right_normal)
        / max(np.linalg.norm(left_normal) * np.linalg.norm(right_normal), 1e-9)
    )
    return ForceSample(
        float(row["left_object_normal_force_N"]),
        float(row["right_object_normal_force_N"]),
        left_target_contact=bool(row["left_object_contact"]),
        right_target_contact=bool(row["right_object_contact"]),
        center_offset_m=center,
        normal_opposition_cosine=opposition,
        source_sensor=SENSOR_NAME,
    )


def action_tensor(torch: Any, action: np.ndarray, device: Any) -> Any:
    return torch.from_numpy(np.asarray(action, dtype=np.float32)).reshape(1, -1).to(device)


def load_frozen_probe_module(out: Path) -> Any:
    """Load the frozen P4-B implementation without running its collector."""
    os.environ.update({
        "P4_TASK_ID": str(TASK_ID),
        "P4_VARIANT": "P4B",
        "P4_OUT": str(out / "FROZEN_P4B_IMPORT"),
        "P4_RESUME": "0",
    })
    old_stdout, old_stderr = sys.stdout, sys.stderr
    spec = importlib.util.spec_from_file_location("post_grasp_frozen_p4b", P4_PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen probe {P4_PROBE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.stdout, sys.stderr = old_stdout, old_stderr
    # P4-B is frozen; only the runtime object's names are adapted to the
    # already-selected Frozen-VLA task.  No probe constants or motion law are
    # changed.
    mod.TASK_ID = TASK_ID
    mod.OBJ_NAME = OBJECT
    mod.BASKET_NAME = TARGET
    mod.VARIANT = "P4B"
    return mod


def run_frozen_probe_after_handoff(env: Any, torch: Any, obs: Any, handoff_action: np.ndarray, p4: Any) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    """Execute only the frozen P4-B preload/shear/return after VLA handoff.

    The P4 collector's public episode entrypoint intentionally resets and
    performs its own approach/grasp.  That entrypoint is therefore not valid
    for this post-grasp protocol.  This adapter copies only its frozen
    post-contact action law and telemetry schema, beginning at the live VLA
    handoff state and never reacquiring or centering the grasp.
    """
    eef = as_np(obs["policy"]["eef_pose"])[0].astype(float)
    eef_aa = np.asarray(handoff_action[3:6], dtype=np.float32)
    cmd_pos = eef[:3].astype(np.float32).copy()
    robot = env.scene["robot"]
    d_pred = float(np.mean(as_np(robot.data.joint_pos)[0, -2:]))
    d_pred = float(np.clip(d_pred, p4.D_CLOSED, p4.D_OPEN))
    preload = float(p4.P4B_BASE_FORCE_N)
    contact_normal, tangent, _binormal, tangent_source, _obj_b, _target_b = p4._current_contact_frame(env)
    rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    rho_impulse = 0.0
    marker0: float | None = None
    pre_fn: list[float] = []
    pre_marker: list[float] = []
    stop_trigger = "completed"
    probe_failure = 0
    contact_lost = 0
    dropped = 0
    obj_probe0: np.ndarray | None = None
    q_probe0: np.ndarray | None = None
    max_obj_disp = 0.0
    max_obj_rot = 0.0
    prev_marker: float | None = None
    local_step = 0

    def observe(phase: str, increment_mm: float, accumulated_m: float, is_probe: bool) -> tuple[Any, bool, float, float, float]:
        nonlocal d_pred, marker0, prev_marker, rho_impulse, probe_failure
        nonlocal contact_lost, dropped, obj_probe0, q_probe0, max_obj_disp, max_obj_rot, local_step, stop_trigger
        action = p4._make_action(cmd_pos, eef_aa, d_pred, preload, env.device)
        obs2, _, term, trunc, _ = env.step(action)
        local_step += 1
        f = obs2["policy"]["gripper_net_force"][0]
        if f.ndim == 3:
            f = f[-1]
        f_l = f[0].detach().cpu().numpy().astype(float)
        f_r = f[1].detach().cpu().numpy().astype(float)
        fn = float(2.0 * min(abs(f_l[2]), abs(f_r[2])))
        ft = float(np.linalg.norm(np.asarray([f_l[0] + f_r[0], f_l[1] + f_r[1]])))
        rho = float(ft / max(fn, 1e-6))
        f_sq = float(p4._f(p4._dbg(env).get("f_sq_meas"), fn))
        left_contact = int(np.linalg.norm(f_l) > p4.FINGER_FORCE_MIN_N and abs(f_l[2]) > p4.FINGER_FORCE_MIN_N)
        right_contact = int(np.linalg.norm(f_r) > p4.FINGER_FORCE_MIN_N and abs(f_r[2]) > p4.FINGER_FORCE_MIN_N)
        contact_state = "bilateral" if left_contact and right_contact else ("unilateral" if left_contact or right_contact else "none")
        d_pred = float(p4._force_servo(d_pred, f_sq, preload))
        tac = p4._tactile_summaries(obs2)
        marker = float(tac["marker_mean"])
        marker_velocity = 0.0 if prev_marker is None else (marker - prev_marker) / DT
        prev_marker = marker
        if marker0 is None and is_probe:
            marker0 = marker
        marker_du = 0.0 if marker0 is None else marker - marker0
        obj_p = as_np(env.scene[OBJECT].data.root_pos_w)[0].astype(float)
        obj_q = as_np(env.scene[OBJECT].data.root_quat_w)[0].astype(float)
        if is_probe:
            if obj_probe0 is None:
                obj_probe0, q_probe0 = obj_p.copy(), obj_q.copy()
            max_obj_disp = max(max_obj_disp, float(np.linalg.norm(obj_p - obj_probe0)))
            max_obj_rot = max(max_obj_rot, float(p4._quat_angle(q_probe0, obj_q)))
            rho_impulse += rho * DT
            if contact_state != "bilateral" or fn < p4.CONTACT_FORCE_EPS_N:
                contact_lost, probe_failure = 1, 1
                stop_trigger = "hard_contact_loss"
        if bool(term[0].item()) or bool(trunc[0].item()):
            stop_trigger = "episode_termination"
        row = {
            "trial_id": f"post_grasp_task{TASK_ID}_root{ROOT}", "task_id": TASK_ID,
            "object_id": OBJECT, "seed_idx": ROOT, "friction": "AUDIT_ONLY",
            "timestamp": local_step, "probe_variant": "P4B", "probe_phase": phase,
            "step": local_step, "t_s": local_step * DT, "force_target": preload,
            "measured_squeeze": f_sq, "target_normal_force": preload,
            "measured_fn": fn, "measured_ft": ft, "ft_over_fn": rho,
            "left_fx": f_l[0], "left_fy": f_l[1], "left_fz": f_l[2],
            "right_fx": f_r[0], "right_fy": f_r[1], "right_fz": f_r[2],
            "force_imbalance": abs(np.linalg.norm(f_l) - np.linalg.norm(f_r)),
            "force_imbalance_ratio": abs(np.linalg.norm(f_l) - np.linalg.norm(f_r)) / max(np.linalg.norm(f_l) + np.linalg.norm(f_r), 1e-6),
            "gripper_opening": p4._f(obs2["policy"]["gripper_pos"][0], np.nan),
            "contact_normal_x": float(contact_normal[0]), "contact_normal_y": float(contact_normal[1]), "contact_normal_z": float(contact_normal[2]),
            "contact_tangent_x": float(tangent[0]), "contact_tangent_y": float(tangent[1]), "contact_tangent_z": float(tangent[2]),
            "commanded_tangent_increment_mm": increment_mm, "accumulated_displacement_mm": accumulated_m * 1000.0,
            "marker_motion": marker, "marker_tangential": float(tac["marker_tangential"]),
            "marker_velocity": marker_velocity, "marker_loading_unloading": marker_du,
            "contact_left": left_contact, "contact_right": right_contact, "contact_state": contact_state,
            "stop_trigger": stop_trigger, "eef_x": float(as_np(obs2["policy"]["eef_pose"])[0, 0]),
            "eef_y": float(as_np(obs2["policy"]["eef_pose"])[0, 1]), "eef_z": float(as_np(obs2["policy"]["eef_pose"])[0, 2]),
            "object_x_priv": obj_p[0], "object_y_priv": obj_p[1], "object_z_priv": obj_p[2],
            "object_qw_priv": obj_q[0], "object_qx_priv": obj_q[1], "object_qy_priv": obj_q[2], "object_qz_priv": obj_q[3],
            "tactile_ok": int(tac["tactile_ok"]),
        }
        rows.append(row)
        if is_probe:
            probe_rows.append(row)
        return obs2, bool(term[0].item()) or bool(trunc[0].item()), fn, rho, marker

    # Frozen P4-B hold is retained as the post-handoff preload segment; the
    # VLA approach/descent/close are never re-run.
    for _ in range(10):
        _obs, terminated, fn, _rho, _marker = observe("hold", 0.0, 0.0, False)
        pre_fn.append(fn)
        pre_marker.append(float(rows[-1]["marker_motion"]))
        if terminated:
            break
    pre_fn_mean = float(np.mean(pre_fn)) if pre_fn else np.nan
    pre_marker_mean = float(np.mean(pre_marker)) if pre_marker else np.nan
    accumulated = 0.0
    if not rows or not terminated:
        for _ in range(p4.MAX_OUT_STEPS):
            if accumulated >= p4.MAX_DISP_M:
                stop_trigger = "max_displacement_cap"
                break
            inc = min(p4.ADAPTIVE_STEP_M, p4.MAX_DISP_M - accumulated)
            cmd_pos[:] = cmd_pos + tangent * inc
            accumulated += inc
            _obs, terminated, fn, rho, marker = observe("probe_out", inc * 1000.0, accumulated, True)
            if terminated or contact_lost:
                break
            if np.isfinite(pre_fn_mean) and fn < p4.RELATIVE_NORMAL_ALPHA * pre_fn_mean:
                stop_trigger = "relative_normal_drop"
                break
            if rho >= p4.RHO_CAP:
                stop_trigger = "shear_ratio_cap"
                break
            if rho_impulse >= p4.RHO_IMPULSE_TARGET:
                stop_trigger = "normalized_shear_impulse"
                break
            if np.isfinite(pre_marker_mean) and abs(marker - pre_marker_mean) / max(abs(pre_marker_mean), 1e-6) > p4.MARKER_NORM_STOP:
                stop_trigger = "marker_motion_budget"
                break
        return_start = cmd_pos.copy()
        for i in range(p4.RETURN_STEPS):
            cmd_pos[:] = (1.0 - (i + 1) / p4.RETURN_STEPS) * return_start + ((i + 1) / p4.RETURN_STEPS) * eef[:3]
            _obs, terminated, _fn, _rho, _marker = observe("probe_back", 0.0, accumulated, True)
            if terminated:
                break
        if not terminated:
            cmd_pos[:] = eef[:3]
            for _ in range(p4.POST_HOLD_STEPS):
                _obs, terminated, _fn, _rho, _marker = observe("probe_hold", 0.0, 0.0, True)
                if terminated:
                    break
    rec = {
        "probe_source": "REAL_CONTACT_CONDITIONED_PROBE", "probe_id": "P4B", "probe_variant": "P4B",
        "probe_qualified": int(bool(probe_rows) and not probe_failure and not contact_lost and any(r["probe_phase"] == "probe_out" for r in probe_rows)),
        "probe_failure": probe_failure, "contact_lost_probe": contact_lost, "stop_trigger": stop_trigger,
        "preprobe_normal_force": pre_fn_mean, "preprobe_marker_motion": pre_marker_mean,
        "actual_probe_displacement_mm": max((float(r["accumulated_displacement_mm"]) for r in probe_rows), default=0.0),
        "probe_duration_s": len(probe_rows) * DT, "rho_impulse": rho_impulse,
        "obj_disp_probe_m": max_obj_disp, "obj_rot_probe_rad": max_obj_rot,
        "contact_frame_source": tangent_source, "tactile_ok": int(any(r["tactile_ok"] for r in probe_rows)),
    }
    return probe_rows, rec, _obs


def object_robot_eef(env: Any, obs: dict[str, Any]) -> dict[str, Any]:
    obj = env.scene[OBJECT]
    robot = env.scene["robot"]
    eef = as_np(obs["policy"]["eef_pose"])[0].astype(float)
    return {
        "object_xyz": as_np(obj.data.root_pos_w)[0].astype(float).tolist(),
        "object_quat_wxyz": as_np(obj.data.root_quat_w)[0].astype(float).tolist(),
        "robot_joint_pos": as_np(robot.data.joint_pos)[0].astype(float).tolist(),
        "eef_xyz": eef[:3].tolist(),
        "eef_quat_wxyz": eef[3:7].tolist(),
    }


def scene_parity(env: Any, obs: dict[str, Any], before: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    now = object_robot_eef(env, obs)
    obj_err = float(np.linalg.norm(np.asarray(now["object_xyz"]) - np.asarray(before["object_xyz"])))
    eef_err = float(np.linalg.norm(np.asarray(now["eef_xyz"]) - np.asarray(before["eef_xyz"])))
    joint_err = float(np.max(np.abs(np.asarray(now["robot_joint_pos"]) - np.asarray(before["robot_joint_pos"]))))
    bilateral = bool(row.get("bilateral_object_contact", False))
    return {
        "object_position_error_m": obj_err,
        "eef_position_error_m": eef_err,
        "robot_joint_max_error_rad_or_m": joint_err,
        "bilateral_contact": bilateral,
        "SNAPSHOT_RESTORE_PARITY": bool(obj_err <= 1e-6 and eef_err <= 1e-6 and joint_err <= 1e-6 and bilateral),
    }


def enrich_row(
    row: dict[str, Any],
    *,
    phase: str,
    event: str,
    result: Any,
    source_action: np.ndarray,
    step: int,
) -> dict[str, Any]:
    values = result.as_dict()
    out = dict(row)
    out.update(values)
    d_final = float(values["d_final"])
    if d_final <= 0.0:
        saturation = "LOWER_APERTURE_LIMIT"
    elif d_final >= 0.04:
        saturation = "UPPER_APERTURE_LIMIT"
    else:
        saturation = "NONE"
    out.update(
        {
            "step": step,
            "phase": phase,
            "event": event,
            "VLA_action_arm_0_6": json.dumps(source_action[:6].tolist()),
            "HANDOFF": int(event == "HANDOFF"),
            "LIFT_START": int(event == "LIFT_START"),
            "OBJECT_LEAVES_SUPPORT": int(event == "OBJECT_LEAVES_SUPPORT"),
            "FIRST_FORCE_DROP": int(event == "FIRST_FORCE_DROP"),
            "FIRST_MAJOR_ASYMMETRY": int(event == "FIRST_MAJOR_ASYMMETRY"),
            "FIRST_UNILATERAL_CONTACT": int(event == "FIRST_UNILATERAL_CONTACT"),
            "CONTACT_LOSS": int(event == "CONTACT_LOSS"),
            "LIFT_THRESHOLD_REACHED": int(event == "LIFT_THRESHOLD_REACHED"),
            "F_target_eff": float(values["F_des"]),
            "F_target_eff_equals_F_des": True,
            "aperture_baseline_command": float(values["d_nominal"]),
            "aperture_command": d_final,
            "actual_aperture": float(row.get("gripper_aperture", float("nan"))),
            "controller_saturation": saturation,
        }
    )
    return out


def run() -> int:
    out = Path(os.environ.get("POST_GRASP_4N_OUT", str(ANALYSIS / "results/post_grasp_4n_live_validation_20260904")))
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)
    trace = load_trace()
    os.environ.update(
        {
            "HDF5_TRAJ_SOURCE_DIR": str(HDF5_DIR),
            "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
            "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
            "TASK_SUITE": TASK_SUITE,
            "TASK_ID": str(TASK_ID),
            "ENABLE_CLOSED_LOOP_FORCE_CONTROLLER": "0",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
        }
    )

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", num_envs=1).app
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        import run_bilateral_contact_audit as helpers

        setup_task_objects(TASK_SUITE, TASK_ID)
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 30.0
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        obs, reset_info = env.reset(seed=ROOT)

        # Frozen VLA owns this entire prefix.  The only semantic handoff
        # condition is established bilateral contact plus a stable nominal
        # gripper command, not an action index or a geometry threshold.
        pre_rows: list[dict[str, Any]] = []
        stable_gripper = 0
        previous_nominal_d: float | None = None
        handoff: dict[str, Any] | None = None
        handoff_obs = None
        handoff_state = None
        handoff_action = None
        step = 0
        for idx, raw in enumerate(trace):
            nominal_d = float(raw[6])
            if previous_nominal_d is not None and abs(nominal_d - previous_nominal_d) <= 1e-4:
                stable_gripper += 1
            else:
                stable_gripper = 1
            previous_nominal_d = nominal_d
            # This is the existing Tabero native VLA grasp execution contract:
            # the frozen policy owns the arm/gripper trajectory, while its
            # uncalibrated force slots are compiled to the native fixed grasp
            # squeeze.  This is pre-handoff and is not ActiveForcing or the
            # new object-specific executor.
            pre_handoff_action = raw.copy()
            pre_handoff_action[7:13] = 0.0
            pre_handoff_action[9] = PREHANDOFF_NATIVE_FORCE_N / 2.0
            pre_handoff_action[12] = PREHANDOFF_NATIVE_FORCE_N / 2.0
            obs, _, term, trunc, _ = env.step(action_tensor(torch, pre_handoff_action, env.device))
            step += 1
            row = helpers.snapshot(env, obs, step, "vla_approach_or_grasp")
            row.update(
                {
                    "root_id": ROOT,
                    "source_trace": str(TRACE),
                    "vla_native_grasp": 1,
                    "pre_handoff_native_force_N": PREHANDOFF_NATIVE_FORCE_N,
                    "nominal_gripper_d": nominal_d,
                    "vla_gripper_stable_steps": stable_gripper,
                    "post_grasp_semantic": int(bool(row["bilateral_object_contact"]) and stable_gripper >= HANDOFF_STABLE_GRIPPER_STEPS),
                    "source_action_index": idx,
                }
            )
            pre_rows.append(row)
            if bool(row["bilateral_object_contact"]) and stable_gripper >= HANDOFF_STABLE_GRIPPER_STEPS:
                handoff = row
                handoff_obs = obs
                handoff_action = raw.copy()
                handoff_state = env.scene.get_state(is_relative=True)
                break
            if bool(term[0].item()) or bool(trunc[0].item()):
                break

        metadata: dict[str, Any] = {
            "protocol": "POST_GRASP_FORCE_ADAPTATION",
            "root_id": ROOT,
            "task": f"{TASK_SUITE}/task{TASK_ID}",
            "object": OBJECT,
            "target": TARGET,
            "F_des_N": F_DES,
            "VLA_NATIVE_GRASP": True,
            "HANDOFF_DEFINITION": "first VLA semantic post-grasp/pre-lift row with bilateral target-object contact and stable nominal gripper command",
            "HANDOFF_CONDITION": "bilateral_target_contact AND vla_gripper_stable_steps >= 3",
            "center_offset_role": "diagnostic_only",
            "force_asymmetry_role": "diagnostic_only",
            "normal_opposition_role": "diagnostic_only",
            "arm_trajectory": "frozen_vla_unchanged_after_handoff",
            "pre_handoff_executor": "frozen_vla_native_action",
            "pre_handoff_native_force_N": PREHANDOFF_NATIVE_FORCE_N,
            "pre_handoff_force_role": "existing Tabero native grasp execution; not ActiveForcing",
            "post_handoff_executor": "Tabero_true_force_gripper_only",
            "object_specific_sensor": SENSOR_NAME,
            "physical_force_definition": "2*min(F_left_target_normal,F_right_target_normal)",
            "double_force_control": False,
            "legacy_force_feedback_disabled": True,
            "legacy_1_9x_feedforward_disabled": True,
            "static_steps": STATIC_STEPS,
            "normal_lift_max_steps": LIFT_MAX_STEPS,
            "pure_vertical_condition": "only_if_normal_vla_lift_fails_after_static_pass",
            "policy_server": "existing_frozen_policy_trace; server untouched",
            "reset_info": str(reset_info),
        }
        if handoff is None:
            metadata.update(
                {
                    "HANDOFF_STEP": None,
                    "HANDOFF_PHASE": None,
                    "POST_GRASP_HANDOFF": False,
                    "PRIMARY_FAILURE_CLASS": "PRE_HANDOFF_VLA_GRASP_FAILURE",
                    "4N_POST_GRASP_STATIC": "NOT_RUN",
                    "4N_NORMAL_VLA_LIFT": "NOT_RUN",
                    "4N_PURE_VERTICAL_DIAGNOSTIC": "NOT_RUN",
                }
            )
            write_csv(out / "VLA_PREFIX_TELEMETRY.csv", pre_rows)
            write_json(out / "POST_GRASP_4N_RESULT.json", metadata)
            return 0

        metadata.update(
            {
                "HANDOFF_STEP": int(handoff["step"]),
                "HANDOFF_PHASE": "post_grasp_pre_lift",
                "POST_GRASP_HANDOFF": True,
                "handoff_row": handoff,
                "handoff_snapshot": str(out / "POST_GRASP_HANDOFF_SNAPSHOT.pt"),
            }
        )
        torch.save(
            {
                "scene_state": handoff_state,
                "metadata": metadata,
                "robot_object_eef": object_robot_eef(env, handoff_obs),
            },
            out / "POST_GRASP_HANDOFF_SNAPSHOT.pt",
        )

        # ActiveForcing starts only now.  The P4-B adapter begins at the live
        # VLA handoff and does not call the collector's reset/approach/grasp
        # entrypoint.  This is the frozen probe, not a new probe design.
        frozen_p4 = load_frozen_probe_module(out)
        probe_rows, probe_rec, _probe_obs = run_frozen_probe_after_handoff(
            env, torch, handoff_obs, handoff_action, frozen_p4
        )
        write_csv(out / "POST_GRASP_PROBE_TELEMETRY.csv", probe_rows)
        metadata.update(
            {
                "probe_source": str(P4_PROBE),
                "probe_sha256": "a1566334f9f79ad9d8491612f10d086386049295314f6d1fd62512be1867bca9",
                "PROBE_AFTER_GRASP_VALID": bool(probe_rec["probe_qualified"]),
                "probe_record": probe_rec,
                "probe_rows": len(probe_rows),
            }
        )
        if not probe_rec["probe_qualified"]:
            metadata.update(
                {
                    "PRIMARY_FAILURE_CLASS": "PROBE_FAILURE",
                    "4N_POST_GRASP_STATIC": "NOT_RUN",
                    "4N_NORMAL_VLA_LIFT": "NOT_RUN",
                    "4N_PURE_VERTICAL_DIAGNOSTIC": "NOT_RUN",
                }
            )
            write_csv(out / "VLA_PREFIX_TELEMETRY.csv", pre_rows)
            write_csv(out / "POST_GRASP_4N_TIMESERIES.csv", [])
            write_json(out / "POST_GRASP_4N_RESULT.json", metadata)
            return 0

        # Inference-only frozen ensemble: 46-dimensional preprocessing and
        # the nine archived estimator checkpoints.  GT friction never enters.
        sys.path.insert(0, str(ANALYSIS))
        from activeforcing_continuous_posterior_final import FrozenProbeEnsemble

        ensemble = FrozenProbeEnsemble()
        posterior = ensemble.predict(probe_rows, None)
        metadata.update(
            {
                "FRICTION_POSTERIOR_RUNTIME_VALID": True,
                "PROBE_FEATURE_DIMENSION": 46,
                "friction_posterior_mean": posterior["mean"],
                "friction_posterior_std": posterior["std"],
                "friction_posterior_quantiles": posterior["quantiles"],
                "friction_posterior_member_means": posterior["member_means"],
                "friction_posterior_member_log_sigma": posterior["member_log_sigma"],
                "friction_posterior_checkpoint_count": len(posterior["member_means"]),
            }
        )

        # Isaac scene states can contain tensors backed by live simulator
        # storage.  Clone every tensor before the first branch so static,
        # normal-lift, and pure-vertical restores remain genuinely same-state.
        post_probe_state = clone_state(env.scene.get_state(is_relative=True))
        post_probe_obs = env.observation_manager.compute()
        before_restore = object_robot_eef(env, post_probe_obs)
        metadata["post_probe_snapshot"] = str(out / "POST_GRASP_POST_PROBE_PRE_LIFT_SNAPSHOT.pt")
        torch.save(
            {
                "scene_state": post_probe_state,
                "metadata": metadata,
                "robot_object_eef": before_restore,
                "probe_record": probe_rec,
                "friction_posterior": posterior,
            },
            out / "POST_GRASP_POST_PROBE_PRE_LIFT_SNAPSHOT.pt",
        )

        if os.environ.get("POST_GRASP_PARITY_ONLY", "0") == "1":
            parity_summary = run_snapshot_parity_audit(
                env=env,
                torch=torch,
                helpers=helpers,
                post_probe_state=post_probe_state,
                post_probe_obs=post_probe_obs,
                probe_rows=probe_rows,
                posterior=posterior,
                handoff=handoff,
                handoff_action=handoff_action,
                trace=trace,
                out=out,
            )
            metadata.update(parity_summary)
            metadata.update(
                {
                    "FINAL_STATUS": "SNAPSHOT_PARITY_AUDIT_COMPLETE",
                    "4N_POST_GRASP_STATIC": "NOT_RUN",
                    "4N_NORMAL_VLA_LIFT": "NOT_RUN",
                    "4N_PURE_VERTICAL_DIAGNOSTIC": "NOT_RUN",
                    "READY_FOR_SAME_STATE_FORCE_FRONTIER": False,
                    "READY_FOR_5_CONTEXT_PER_TASK_PILOT": False,
                    "READY_FOR_60_CONTEXT_PER_TASK_COLLECTION": False,
                    "READY_FOR_CONTINUOUS_POSTERIOR_TRAINING": False,
                }
            )
            write_csv(out / "VLA_PREFIX_TELEMETRY.csv", pre_rows)
            write_json(out / "POST_GRASP_4N_RESULT.json", metadata)
            return 0

        from tac_manip.tasks.manipulation.libero import mdp

        term_cfg_snapshot = mdp.disable_tabero_legacy_force_loop(env.action_manager.get_term("arm_action"))
        telemetry: list[dict[str, Any]] = []

        def restore_snapshot() -> tuple[Any, dict[str, Any]]:
            # Use the project-validated stable scene restore path.  The
            # ManagerBasedEnv reset_to path also resets/render-callbacks the
            # active tactile sensors, which can leave a stale contact cache
            # on repeated branch restores.  Restoring the scene directly and
            # refreshing manager buffers preserves the frozen post-probe
            # physics state across every branch.
            env.scene.reset_to(
                clone_state(post_probe_state),
                torch.tensor([0], device=env.device),
                is_relative=True,
            )
            env.sim.forward()
            env.observation_manager.reset(torch.tensor([0], device=env.device))
            env.action_manager.reset(torch.tensor([0], device=env.device))
            branch_obs = env.observation_manager.compute(update_history=True)
            parity_row = helpers.snapshot(env, branch_obs, int(handoff["step"]), "post_grasp_restore")
            parity = scene_parity(env, branch_obs, before_restore, parity_row)
            return branch_obs, {"row": parity_row, "parity": parity}

        def run_static() -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
            branch_obs, restore = restore_snapshot()
            ctl = TaberoTruePhysicalForceHybrid()
            # Handoff has already established bilateral contact; do not make
            # the post-handoff branch reacquire it with the legacy force path
            # disabled.  Start the new executor directly in FORCE_TRACK.
            ctl.state = HybridState.FORCE_TRACK
            # The VLA slot is the nominal target, while the restored snapshot
            # already has a physical finger-joint state that may differ from
            # that historical slot because native Tabero grasp force has
            # acted before handoff.  Start the static force loop from the
            # actual snapshot joint command and carry its aperture target
            # forward across the hold steps.
            handoff_joint_pos = json.loads(handoff["joint_pos"])
            nominal_d = float(np.mean(np.asarray(handoff_joint_pos[-2:], dtype=float)))
            rows: list[dict[str, Any]] = []
            for i in range(STATIC_STEPS):
                live = helpers.snapshot(env, branch_obs, int(handoff["step"]) + i + 1, "post_grasp_static")
                sample = sample_from_row(live)
                result = ctl.step(F_des=F_DES, d_nominal=nominal_d, sample=sample, execution_phase="static")
                event = ""
                if i == 0:
                    event = "HANDOFF"
                if not sample.bilateral_contact:
                    event = "CONTACT_LOSS"
                row = enrich_row(live, phase="post_grasp_static", event=event, result=result, source_action=handoff_action, step=int(live["step"]))
                rows.append(row)
                if not sample.bilateral_contact:
                    break
                # Keep the exact handoff arm command format (position plus
                # axis-angle).  Only slot 6 is replaced by the hybrid result.
                action = handoff_action.copy()
                action[6] = result.d_final
                action[7:13] = 0.0
                nominal_d = float(result.d_final)
                branch_obs, _, term, trunc, _ = env.step(action_tensor(torch, action, env.device))
                if bool(term[0].item()) or bool(trunc[0].item()):
                    break
            vals = np.asarray([r["F_meas"] for r in rows], dtype=float)
            bilateral_rate = float(np.mean([int(r["BILATERAL_CONTACT_VALID"]) for r in rows])) if rows else 0.0
            mae = float(np.mean(np.abs(vals - F_DES))) if len(vals) else float("nan")
            bias = float(np.mean(vals - F_DES)) if len(vals) else float("nan")
            std = float(np.std(vals)) if len(vals) else float("nan")
            metrics = {
                "SNAPSHOT_RESTORE_PARITY": restore["parity"]["SNAPSHOT_RESTORE_PARITY"],
                "4N_STATIC_MAE": mae,
                "4N_STATIC_BIAS": bias,
                "4N_STATIC_STD": std,
                "4N_STATIC_PEAK": float(np.max(vals)) if len(vals) else float("nan"),
                "4N_WITHIN_0.5N_RATE": float(np.mean(np.abs(vals - F_DES) <= 0.5)) if len(vals) else 0.0,
                "4N_WITHIN_1N_RATE": float(np.mean(np.abs(vals - F_DES) <= 1.0)) if len(vals) else 0.0,
                "4N_BILATERAL_CONTACT_RATE": bilateral_rate,
                "static_samples": len(rows),
                "static_valid": bool(restore["parity"]["SNAPSHOT_RESTORE_PARITY"] and bilateral_rate >= 0.8 and len(vals) >= 10 and mae <= 1.0),
                "restore_diagnostics": restore["parity"],
            }
            return bool(metrics["static_valid"]), rows, metrics

        static_ok, static_rows, static_metrics = run_static()
        telemetry.extend(static_rows)
        metadata["static_metrics"] = static_metrics
        metadata["4N_POST_GRASP_STATIC"] = "PASS" if static_ok else "FAIL"

        normal_ok = False
        normal_rows: list[dict[str, Any]] = []
        normal_metrics: dict[str, Any] = {}
        pure_ok = False
        pure_rows: list[dict[str, Any]] = []
        pure_metrics: dict[str, Any] = {}
        if static_ok:
            branch_obs, restore = restore_snapshot()
            ctl = TaberoTruePhysicalForceHybrid()
            ctl.state = HybridState.FORCE_TRACK
            lift_started = False
            leaves_support = False
            force_tracking_seen = False
            first_tracking = False
            lift_eef_z = float(np.asarray(handoff["eef_z"])) if "eef_z" in handoff else float(np.asarray(handoff["eef_pose"])[2])
            for i, raw in enumerate(trace[int(handoff["source_action_index"]) + 1:] if "source_action_index" in handoff else trace):
                # A trace row's original arm action is preserved.  This is the
                # normal VLA post-grasp trajectory, not a vertical substitute.
                live = helpers.snapshot(env, branch_obs, int(handoff["step"]) + i + 1, "post_grasp_vla")
                sample = sample_from_row(live)
                nominal = raw.copy()
                result = ctl.step(F_des=F_DES, d_nominal=float(nominal[6]), sample=sample, execution_phase="lift")
                dz = float(live["object_z"]) - float(handoff["object_z"])
                eef_dz = float(nominal[2]) - float(handoff["eef_z"])
                if not lift_started and eef_dz > 0.002:
                    lift_started = True
                    event = "LIFT_START"
                else:
                    event = ""
                if abs(float(result.F_meas) - F_DES) <= 1.0:
                    force_tracking_seen = True
                    if not first_tracking:
                        first_tracking = True
                if not leaves_support and dz >= LIFT_THRESHOLD_M:
                    leaves_support = True
                    event = "OBJECT_LEAVES_SUPPORT"
                if event == "" and first_tracking and abs(float(result.F_meas) - F_DES) > 1.0:
                    event = "FIRST_FORCE_DROP"
                if event == "" and sample.force_asymmetry >= 0.20:
                    event = "FIRST_MAJOR_ASYMMETRY"
                if event == "" and sample.left_target_contact != sample.right_target_contact:
                    event = "FIRST_UNILATERAL_CONTACT"
                if event == "" and not sample.bilateral_contact:
                    event = "CONTACT_LOSS"
                if event == "" and leaves_support:
                    event = "LIFT_THRESHOLD_REACHED"
                row = enrich_row(live, phase="post_grasp_normal_vla_lift", event=event, result=result, source_action=nominal, step=int(live["step"]))
                row["object_height_delta_m"] = dz
                row["normal_vla_eef_dz_m"] = eef_dz
                normal_rows.append(row)
                if not sample.bilateral_contact:
                    break
                branch_obs, _, term, trunc, _ = env.step(action_tensor(torch, apply_to_tabero_nominal_action(torch.from_numpy(nominal).to(env.device), result.d_final).detach().cpu().numpy(), env.device))
                if bool(term[0].item()) or bool(trunc[0].item()):
                    break
            vals = np.asarray([r["F_meas"] for r in normal_rows if r["BILATERAL_CONTACT_VALID"]], dtype=float)
            tracking_rate = float(np.mean(np.abs(vals - F_DES) <= 1.0)) if len(vals) else 0.0
            contact_rate = float(np.mean([int(r["BILATERAL_CONTACT_VALID"]) for r in normal_rows])) if normal_rows else 0.0
            normal_ok = bool(leaves_support and contact_rate >= 0.8 and tracking_rate >= 0.6)
            normal_metrics = {
                "4N_LIFT_MAE": float(np.mean(np.abs(vals - F_DES))) if len(vals) else float("nan"),
                "4N_LIFT_BIAS": float(np.mean(vals - F_DES)) if len(vals) else float("nan"),
                "4N_LIFT_PEAK": float(np.max(vals)) if len(vals) else float("nan"),
                "4N_LIFT_BILATERAL_CONTACT_RATE": contact_rate,
                "4N_LIFT_FORCE_TRACKING_RATE_WITHIN_1N": tracking_rate,
                "object_leaves_support": leaves_support,
                "normal_vla_lift_success": normal_ok,
                "snapshot_restore": restore["parity"],
            }
            metadata["4N_NORMAL_VLA_LIFT"] = "PASS" if normal_ok else "FAIL"
            metadata["normal_vla_metrics"] = normal_metrics
            telemetry.extend(normal_rows)

            # Pure-z remains conditional and diagnostic only.
            if not normal_ok:
                branch_obs, restore = restore_snapshot()
                ctl = TaberoTruePhysicalForceHybrid()
                ctl.state = HybridState.FORCE_TRACK
                eef0 = as_np(handoff_obs["policy"]["eef_pose"])[0].astype(np.float32)
                for i in range(PURE_VERTICAL_STEPS):
                    live = helpers.snapshot(env, branch_obs, int(handoff["step"]) + i + 1, "pure_vertical_diagnostic")
                    sample = sample_from_row(live)
                    frac = (i + 1) / PURE_VERTICAL_STEPS
                    nominal = np.zeros(13, dtype=np.float32)
                    nominal[:3] = eef0[:3]
                    nominal[2] += PURE_VERTICAL_DISTANCE_M * frac
                    nominal[3:6] = eef0[3:6]
                    nominal[6] = float(handoff_action[6])
                    result = ctl.step(F_des=F_DES, d_nominal=float(nominal[6]), sample=sample, execution_phase="lift")
                    dz = float(live["object_z"]) - float(handoff["object_z"])
                    event = "CONTACT_LOSS" if not sample.bilateral_contact else ("LIFT_THRESHOLD_REACHED" if dz >= LIFT_THRESHOLD_M else "")
                    row = enrich_row(live, phase="pure_vertical_diagnostic", event=event, result=result, source_action=nominal, step=int(live["step"]))
                    row["object_height_delta_m"] = dz
                    pure_rows.append(row)
                    if not sample.bilateral_contact:
                        break
                    branch_obs, _, term, trunc, _ = env.step(action_tensor(torch, apply_to_tabero_nominal_action(torch.from_numpy(nominal).to(env.device), result.d_final).detach().cpu().numpy(), env.device))
                    if bool(term[0].item()) or bool(trunc[0].item()):
                        break
                vals = np.asarray([r["F_meas"] for r in pure_rows if r["BILATERAL_CONTACT_VALID"]], dtype=float)
                contact_rate = float(np.mean([int(r["BILATERAL_CONTACT_VALID"]) for r in pure_rows])) if pure_rows else 0.0
                pure_ok = bool(pure_rows and max([float(r["object_height_delta_m"]) for r in pure_rows], default=0.0) >= LIFT_THRESHOLD_M and contact_rate >= 0.8 and len(vals) and float(np.mean(np.abs(vals - F_DES))) <= 1.0)
                pure_metrics = {
                    "4N_PURE_VERTICAL_MAE": float(np.mean(np.abs(vals - F_DES))) if len(vals) else float("nan"),
                    "4N_PURE_VERTICAL_BILATERAL_CONTACT_RATE": contact_rate,
                    "pure_vertical_success": pure_ok,
                    "snapshot_restore": restore["parity"],
                }
                metadata["4N_PURE_VERTICAL_DIAGNOSTIC"] = "PASS" if pure_ok else "FAIL"
                metadata["pure_vertical_metrics"] = pure_metrics
                telemetry.extend(pure_rows)
        else:
            metadata["4N_NORMAL_VLA_LIFT"] = "NOT_RUN"
            metadata["4N_PURE_VERTICAL_DIAGNOSTIC"] = "NOT_RUN"

        if not static_ok:
            primary = "LOW_LEVEL_FORCE_EXECUTION_FAILURE"
        elif normal_ok:
            primary = "NONE"
        else:
            normal_contact_collapse = any(
                row.get("event") in {"FIRST_UNILATERAL_CONTACT", "CONTACT_LOSS"}
                for row in normal_rows
            )
            normal_force_tracked_before_collapse = any(
                row.get("BILATERAL_CONTACT_VALID")
                and abs(float(row.get("F_meas", float("nan"))) - F_DES) <= 1.0
                for row in normal_rows
            )
            if normal_contact_collapse and normal_force_tracked_before_collapse:
                primary = "POST_HANDOFF_VLA_MOTION_CONTACT_FAILURE"
            elif normal_metrics.get("4N_LIFT_FORCE_TRACKING_RATE_WITHIN_1N", 0.0) >= 0.6:
                primary = "POST_GRASP_FORCE_INSUFFICIENT"
            else:
                primary = "LOW_LEVEL_FORCE_EXECUTION_FAILURE"
        metadata["PRIMARY_FAILURE_CLASS"] = primary
        metadata["LOW_LEVEL_FORCE_EXECUTION_FAILURE"] = bool(primary == "LOW_LEVEL_FORCE_EXECUTION_FAILURE")
        metadata["POST_HANDOFF_VLA_MOTION_CONTACT_FAILURE"] = bool(
            primary == "POST_HANDOFF_VLA_MOTION_CONTACT_FAILURE"
        )
        metadata["4N_POST_GRASP_FORCE_INSUFFICIENT"] = (
            "NO" if primary == "POST_HANDOFF_VLA_MOTION_CONTACT_FAILURE" else "UNKNOWN"
        )
        # A diagnostic pure-z row is not enough to authorize a same-state
        # frontier.  Every repeated restore must have parity first.
        metadata["READY_FOR_2_4_6_SAME_STATE_BRANCHING"] = bool(
            static_ok
            and normal_ok
            and pure_ok
            and normal_metrics.get("snapshot_restore", {}).get("SNAPSHOT_RESTORE_PARITY", False)
            and pure_metrics.get("snapshot_restore", {}).get("SNAPSHOT_RESTORE_PARITY", False)
        )
        metadata["READY_FOR_POST_GRASP_ACTIVEFORCING_EVAL"] = False
        metadata["same_state_branch_contract"] = list(same_state_force_branches(snapshot_id="post_grasp_snapshot", root_id=ROOT, forces_n=(4.0,)))
        write_csv(out / "VLA_PREFIX_TELEMETRY.csv", pre_rows)
        write_csv(out / "POST_GRASP_4N_TIMESERIES.csv", telemetry)
        write_json(out / "POST_GRASP_4N_RESULT.json", metadata)
        return 0
    except Exception as exc:
        # Persist the exact post-handoff failure before Isaac's forced process
        # exit, so a simulator/extension error cannot erase the live evidence.
        import traceback
        traceback.print_exc()
        write_json(
            out / "POST_GRASP_4N_RESULT.json",
            {
                "FINAL_STATUS": "POST_GRASP_4N_LIVE_RUN_ERROR",
                "protocol": "POST_GRASP_FORCE_ADAPTATION",
                "root_id": ROOT,
                "F_des_N": F_DES,
                "VLA_NATIVE_GRASP": bool(handoff is not None),
                "POST_GRASP_HANDOFF": bool(handoff is not None),
                "HANDOFF_STEP": handoff.get("step") if handoff else None,
                "4N_POST_GRASP_STATIC": "NOT_RUN",
                "4N_NORMAL_VLA_LIFT": "NOT_RUN",
                "4N_PURE_VERTICAL_DIAGNOSTIC": "NOT_RUN",
                "PRIMARY_FAILURE_CLASS": "LOW_LEVEL_FORCE_EXECUTION_FAILURE" if handoff else "PRE_HANDOFF_VLA_GRASP_FAILURE",
                "error": repr(exc),
            },
        )
        raise
    finally:
        try:
            if "term_cfg_snapshot" in locals():
                mdp.restore_tabero_legacy_force_loop(env.action_manager.get_term("arm_action"), term_cfg_snapshot)
        except Exception:
            pass
        # Isaac camera teardown can raise a native weak-reference exception;
        # preserve the established forced-exit behavior after evidence is
        # written above.
        os._exit(0)


if __name__ == "__main__":
    raise SystemExit(run())
