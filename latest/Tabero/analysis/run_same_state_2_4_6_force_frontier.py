#!/usr/bin/env python3
"""Same-state 2/4/6 N post-grasp force frontier.

This runner consumes the already audited Task5/root7400 post-probe/pre-lift
snapshot.  It deliberately does not execute the VLA prefix, grasp, or probe.
Each branch restores the same serialized scene and runtime buffers, starts the
same gripper-only executor, changes only ``F_des``, settles briefly, and then
replays the same frozen VLA continuation trace.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/home/exouser/Tabero")
ANALYSIS = REPO / "analysis"
PROTOTYPE = Path("/home/exouser/E3_E6_E7_LANES/closed_loop_gripper_force_controller_prototype_20260903")
SNAPSHOT_DIR = Path(os.environ.get(
    "POST_GRASP_SNAPSHOT_DIR",
    str(ANALYSIS / "results/post_grasp_snapshot_parity_audit_20260904_v20"),
))
SNAPSHOT_PT = Path(os.environ.get(
    "POST_GRASP_SNAPSHOT_PT",
    str(SNAPSHOT_DIR / "POST_GRASP_POST_PROBE_PRE_LIFT_SNAPSHOT.pt"),
))
REFERENCE_JSON = Path(os.environ.get(
    "POST_GRASP_REFERENCE_JSON",
    str(SNAPSHOT_DIR / "SNAPSHOT_REFERENCE_STATE.json"),
))
BASE_RESULT = Path(os.environ.get(
    "POST_GRASP_BASE_RESULT",
    str(SNAPSHOT_DIR / "POST_GRASP_4N_RESULT.json"),
))
OUT_DEFAULT = Path(os.environ.get(
    "POST_GRASP_OUT_DEFAULT",
    str(ANALYSIS / "results/post_grasp_force_frontier_2_4_6_20260904"),
))
FORCES_DEFAULT = (2.0, 4.0, 6.0)
STATIC_STEPS = 40
DT = 0.05
LIFT_THRESHOLD_M = 0.01
FORCE_TOLERANCE_N = 1.0

sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(PROTOTYPE))
import run_post_grasp_4n_live_validation as base  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def quat_wxyz_to_axis_angle(quat: Any) -> np.ndarray:
    """Convert Isaac wxyz quaternion to the 3-vector action representation."""
    q = np.asarray(quat, dtype=float).reshape(-1)[:4]
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    if q[0] < 0.0:
        q = -q
    angle = 2.0 * np.arctan2(float(np.linalg.norm(q[1:])), float(np.clip(q[0], -1.0, 1.0)))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (q[1:] / max(float(np.linalg.norm(q[1:])), 1e-12) * angle).astype(np.float32)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def move_tensors(value: Any, torch: Any, device: Any) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: move_tensors(item, torch, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, torch, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, torch, device) for item in value)
    return value


def restore_disk_runtime(target: Any, state: Any, torch: Any) -> None:
    """Restore JSON-serialized runtime state into live Isaac objects."""
    if target is None or state is None:
        return
    if torch.is_tensor(target):
        if torch.is_tensor(state):
            source = state.to(device=target.device, dtype=target.dtype)
        else:
            source = torch.as_tensor(state, device=target.device, dtype=target.dtype)
        target.copy_(source)
        return
    if target.__class__.__name__ == "CircularBuffer" and isinstance(state, dict):
        if getattr(target, "_buffer", None) is not None and state.get("buffer") is not None:
            restore_disk_runtime(target._buffer, state["buffer"], torch)
        if getattr(target, "_num_pushes", None) is not None and state.get("num_pushes") is not None:
            restore_disk_runtime(target._num_pushes, state["num_pushes"], torch)
        if "pointer" in state:
            target._pointer = int(state["pointer"])
        return
    if isinstance(target, dict) and isinstance(state, dict):
        for key, value in state.items():
            if key in target:
                restore_disk_runtime(target[key], value, torch)
        return
    if isinstance(target, list) and isinstance(state, list):
        for dst, src in zip(target, state):
            restore_disk_runtime(dst, src, torch)
        return
    if hasattr(target, "__dict__") and isinstance(state, dict):
        for key, value in state.items():
            if key.startswith("__") or not hasattr(target, key):
                continue
            dst = getattr(target, key)
            if torch.is_tensor(dst) or isinstance(dst, (dict, list)) or dst.__class__.__name__ == "CircularBuffer" or hasattr(dst, "__dict__"):
                restore_disk_runtime(dst, value, torch)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                try:
                    setattr(target, key, value)
                except Exception:
                    pass


def restore_all_runtime(env: Any, ref: dict[str, Any], torch: Any) -> None:
    """Restore the same mutable runtime buffers used by the v20 parity audit."""
    action = ref["action_runtime_state"]
    manager = env.action_manager
    restore_disk_runtime(getattr(manager, "_action", None), action["manager"]["_action"], torch)
    restore_disk_runtime(getattr(manager, "_prev_action", None), action["manager"]["_prev_action"], torch)
    for name, state in action.get("terms", {}).items():
        term = getattr(manager, "_terms", {}).get(name)
        if term is None:
            continue
        restore_disk_runtime(term, state, torch)
        ik = getattr(term, "_ik_term", None)
        ik_state = state.get("_ik_term", {}) if isinstance(state, dict) else {}
        if ik is not None and isinstance(ik_state, dict):
            for attr in ("_raw_actions", "_processed_actions"):
                if hasattr(ik, attr) and attr in ik_state:
                    restore_disk_runtime(getattr(ik, attr), ik_state[attr], torch)
            controller = getattr(ik, "_ik_controller", None)
            controller_state = ik_state.get("_ik_controller", {})
            if controller is not None and isinstance(controller_state, dict):
                for attr in ("_command", "ee_pos_des", "ee_quat_des"):
                    if hasattr(controller, attr) and attr in controller_state:
                        restore_disk_runtime(getattr(controller, attr), controller_state[attr], torch)

    for name, state in ref.get("sensor_runtime_state", {}).items():
        sensor = getattr(env.scene, "_sensors", {}).get(name)
        if sensor is not None:
            restore_disk_runtime(getattr(sensor, "_data", None), state, torch)
    restore_disk_runtime(
        getattr(env.observation_manager, "_group_obs_term_history_buffer", None),
        ref.get("observation_history_state"),
        torch,
    )
    for name, state in ref.get("manager_runtime_state", {}).items():
        restore_disk_runtime(getattr(env, name, None), state, torch)

    env_state = ref.get("environment_state", {})
    for name in ("episode_length_buf", "reset_buf", "terminated_buf", "truncated_buf"):
        if name in env_state and hasattr(env, name):
            restore_disk_runtime(getattr(env, name), env_state[name], torch)
    if "common_step_counter" in env_state and hasattr(env, "common_step_counter"):
        env.common_step_counter = int(env_state["common_step_counter"])


def safe_term(env: Any, name: str) -> bool:
    try:
        value = env.termination_manager.get_term(name)
        return bool(value[0].item())
    except Exception:
        return False


def row_force(row: dict[str, Any]) -> float:
    return 2.0 * min(float(row["left_object_normal_force_N"]), float(row["right_object_normal_force_N"]))


def physical_branch_row(
    *,
    live: dict[str, Any],
    result: Any,
    force: float,
    branch: str,
    phase: str,
    action_index: int,
    event: str,
    snapshot_xyz: np.ndarray,
    env: Any,
    torch: Any,
) -> dict[str, Any]:
    out = base.enrich_row(
        live,
        phase=phase,
        event=event,
        result=result,
        source_action=np.asarray(live.get("_source_action", np.zeros(13)), dtype=np.float32),
        step=int(live["step"]),
    )
    obj = np.asarray([live["object_x"], live["object_y"], live["object_z"]], dtype=float)
    q = np.asarray([live["object_qw"], live["object_qx"], live["object_qy"], live["object_qz"]], dtype=float)
    out.update(
        {
            "branch": branch,
            "F_des": force,
            "F_target_eff": force,
            "force_error": float(result.force_error),
            "F_meas": float(result.F_meas),
            "F_left_obj": float(result.F_left_obj_normal),
            "F_right_obj": float(result.F_right_obj_normal),
            "bilateral_contact": int(bool(live["bilateral_object_contact"])),
            "force_asymmetry": float(result.force_asymmetry),
            "aperture_command": float(result.d_final),
            "controller_saturation": out["controller_saturation"],
            "object_position": json.dumps(obj.tolist()),
            "object_orientation": json.dumps(q.tolist()),
            "object_height": float(live["object_z"]),
            "object_height_delta_m": float(live["object_z"] - snapshot_xyz[2]),
            "object_slip": int(not bool(live["bilateral_object_contact"])),
            # snapshot/sensor rows may carry numpy scalar values (typically
            # float32); normalize them before JSON encoding so one telemetry
            # row cannot abort the whole frontier run.
            "EE_motion": json.dumps([float(live["eef_x"]), float(live["eef_y"]), float(live["eef_z"])]),
            "VLA_action_index": action_index,
            "trajectory_cursor": action_index,
            "gripper_aperture": float(live["gripper_aperture"]),
            "lift_success_so_far": int(live["object_z"] - snapshot_xyz[2] >= LIFT_THRESHOLD_M),
            "transport_retention_so_far": int(live["object_z"] - snapshot_xyz[2] >= LIFT_THRESHOLD_M and bool(live["bilateral_object_contact"]) and not safe_term(env, "object_1_dropped")),
            "placement_success_so_far": int(safe_term(env, "success")),
            "remaining_task_success_so_far": int(safe_term(env, "success")),
            "source_sensor": base.SENSOR_NAME,
            "controller_state": result.state.value,
            "snapshot_id": "Task5_root7400_post_probe_pre_lift_v20",
            "snapshot_restore_parity_required": True,
        }
    )
    return out


def branch_restore(env: Any, torch: Any, scene_state: Any, ref: dict[str, Any], helpers: Any) -> tuple[Any, dict[str, Any]]:
    env.scene.reset_to(base.clone_state(scene_state), torch.tensor([0], device=env.device), is_relative=True)
    env.sim.forward()
    env.observation_manager.reset(torch.tensor([0], device=env.device))
    env.action_manager.reset(torch.tensor([0], device=env.device))
    if ref.get("action_runtime_state"):
        restore_all_runtime(env, ref, torch)
    obs = env.observation_manager.compute(update_history=True)
    row = helpers.snapshot(env, obs, int(ref.get("snapshot_step", os.environ.get("POST_GRASP_SNAPSHOT_STEP", "206"))), "post_probe_restore")
    return obs, base.capture_physical_state(env, obs, row)


def parity_row(reference: dict[str, Any], current: dict[str, Any], branch: str) -> dict[str, Any]:
    diffs = base.diff_physical_state(reference, current)
    return {
        "branch": branch,
        "snapshot_restore_parity": int(all(item["pass"] for item in diffs)),
        "max_position_error_m": max(
            [float(item["error"]) for item in diffs if item["metric"] in {"eef_position_error", "object_position_error"}]
            or [float("nan")]
        ),
        "max_joint_error": max([float(item["error"]) for item in diffs if "joint" in item["metric"]] or [float("nan")]),
        "orientation_errors": json.dumps({item["metric"]: item["error"] for item in diffs if "orientation" in item["metric"]}),
        "first_failed_fields": ";".join(item["field"] for item in diffs if not item["pass"]),
    }


def add_controller_trace(
    row: dict[str, Any],
    *,
    result: Any,
    ctl: Any,
    action: np.ndarray,
    phase: str,
    snapshot_aperture: float,
    takeover_aperture: float,
    previous: dict[str, Any],
    env: Any,
) -> dict[str, Any]:
    """Attach only fields observable at the hybrid/action boundary.

    This is an audit adapter, not a control change.  Fields that the current
    controller does not expose are explicitly marked rather than invented.
    """
    corr = float(result.force_correction)
    cfg = result_gate_cfg = getattr(ctl, "cfg", None)
    close_limit = float(getattr(result_gate_cfg, "close_step_limit_m", float("nan")))
    open_limit = float(getattr(result_gate_cfg, "open_step_limit_m", float("nan")))
    deadband = float(getattr(result_gate_cfg, "deadband_n", float("nan")))
    internal = {
        "state": getattr(getattr(ctl, "state", None), "value", str(getattr(ctl, "state", "NOT_EXPOSED"))),
        "bilateral_streak": getattr(ctl, "_bilateral_streak", "NOT_EXPOSED"),
        "filtered_force_state": getattr(ctl, "_filtered_force", "NOT_EXPOSED"),
        "grasp_gate": getattr(getattr(ctl, "_grasp_gate", None), "as_dict", lambda: "NOT_EXPOSED")(),
    }
    row.update(
        {
            "physics_step": int(getattr(env, "common_step_counter", row.get("step", -1))),
            "task_phase": phase,
            "F_target_raw": float(result.F_des),
            "F_target_eff": float(result.F_des),
            "F_meas_raw": float(result.F_meas_raw),
            "F_meas_filtered": float(result.F_meas),
            "d_snapshot": float(snapshot_aperture),
            "d_takeover": float(takeover_aperture),
            "d_pred": float(result.d_nominal),
            "delta_d_force": corr,
            "d_cmd": float(result.d_final),
            "d_actual": float(row.get("gripper_aperture", float("nan"))),
            "close_rate_limit_active": int(np.isfinite(close_limit) and corr < 0.0 and abs(corr) >= close_limit - 1e-12),
            "open_rate_limit_active": int(np.isfinite(open_limit) and corr > 0.0 and abs(corr) >= open_limit - 1e-12),
            "deadband_active": int(bool(result.force_loop_active) and np.isfinite(deadband) and abs(float(result.force_error)) < deadband),
            "close_saturation": int(float(result.d_final) <= float(getattr(result_gate_cfg, "aperture_min_m", -1.0))),
            "open_saturation": int(float(result.d_final) >= float(getattr(result_gate_cfg, "aperture_max_m", 1.0))),
            "controller_enabled": 1,
            "force_track_enabled": int(bool(result.force_loop_active)),
            "handoff_flag": int(row.get("HANDOFF", 0)),
            "contact_latch": int(bool(result.gate.bilateral_contact_valid)),
            "previous_force_error": previous.get("force_error", "NOT_AVAILABLE"),
            "previous_measured_force": previous.get("F_meas", "NOT_AVAILABLE"),
            "previous_aperture_command": previous.get("aperture_command", "NOT_AVAILABLE"),
            "action_manager_gripper_input": float(action[6]),
            "legacy_gripper_input": json.dumps(np.asarray(action[7:13], dtype=np.float32).tolist()),
            "final_gripper_command": float(action[6]),
            "VLA_nominal_aperture": float(action[6]) if phase == "post_grasp_vla" else float(result.d_nominal),
            "force_controller_aperture": float(result.d_final),
            "gripper_command_owner": "hybrid_force_controller",
            "controller_internal_state": json.dumps(internal, default=str, sort_keys=True),
        }
    )
    previous.update(
        {
            "force_error": float(result.force_error),
            "F_meas": float(result.F_meas),
            "aperture_command": float(result.d_final),
        }
    )
    return row


def run() -> int:
    out = Path(os.environ.get("POST_GRASP_FRONTIER_OUT", str(OUT_DEFAULT)))
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)
    if not SNAPSHOT_PT.exists():
        raise FileNotFoundError(f"snapshot artifact is missing: {SNAPSHOT_PT}")

    ref = json.loads(REFERENCE_JSON.read_text(encoding="utf-8")) if REFERENCE_JSON.exists() else {}
    force_text = os.environ.get("POST_GRASP_FORCE_LIST", "")
    force_values = tuple(float(x.strip()) for x in force_text.split(",") if x.strip()) if force_text.strip() else FORCES_DEFAULT
    if not force_values:
        raise ValueError("POST_GRASP_FORCE_LIST must contain at least one force")
    base_result = json.loads(BASE_RESULT.read_text(encoding="utf-8")) if BASE_RESULT.exists() else {}
    trace = base.load_trace()
    import torch
    raw_snapshot = torch.load(SNAPSHOT_PT, map_location="cpu", weights_only=False) if "vla_continuation" not in ref else None
    scene_snapshot_only = isinstance(raw_snapshot, dict) and "scene_state" not in raw_snapshot
    next_index = int(ref.get("vla_continuation", {}).get("next_action_index", os.environ.get("POST_GRASP_NEXT_ACTION_INDEX", "206")))
    remaining = trace[next_index:]
    actual_trace_hash = hashlib.sha256(b"".join(np.asarray(action, dtype=np.float32).tobytes() for action in remaining)).hexdigest()
    expected_trace_hash = ref.get("vla_continuation", {}).get("remaining_action_sha256", actual_trace_hash)
    if actual_trace_hash != expected_trace_hash:
        raise RuntimeError(f"Frozen VLA continuation hash mismatch: {actual_trace_hash} != {expected_trace_hash}")
    if ref.get("handoff_step") is not None and next_index != int(ref["handoff_step"]):
        raise RuntimeError("snapshot/handoff step contract mismatch")
    handoff_action = np.asarray(trace[next_index - 1], dtype=np.float32).copy()
    snapshot_id = os.environ.get(
        "POST_GRASP_SNAPSHOT_ID",
        f"Task{base.TASK_ID}_root{base.ROOT}_post_probe_pre_lift",
    )

    os.environ.update(
        {
            "HDF5_TRAJ_SOURCE_DIR": str(base.HDF5_DIR),
            "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
            "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
            "TASK_SUITE": base.TASK_SUITE,
            "TASK_ID": str(base.TASK_ID),
            "ENABLE_CLOSED_LOOP_FORCE_CONTROLLER": "0",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
        }
    )
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", num_envs=1).app
    env = None
    term_cfg_snapshot = None
    exit_code = 0
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        from tac_manip.tasks.manipulation.libero import mdp
        import run_bilateral_contact_audit as helpers

        setup_task_objects(base.TASK_SUITE, base.TASK_ID)
        cfg = parse_env_cfg(base.ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 30.0
        env = gym.make(base.ENV_ID, cfg=cfg).unwrapped
        env.reset(seed=base.ROOT)
        snap = torch.load(SNAPSHOT_PT, map_location="cpu", weights_only=False)
        scene_state = move_tensors(snap if scene_snapshot_only else snap["scene_state"], torch, env.device)
        # Live restored measurements are the authoritative parity baseline.
        term_cfg_snapshot = mdp.disable_tabero_legacy_force_loop(env.action_manager.get_term("arm_action"))

        all_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        parity_rows: list[dict[str, Any]] = []
        reference_live: dict[str, Any] | None = None
        branch_contract = list(base.same_state_force_branches(snapshot_id=snapshot_id, root_id=base.ROOT, forces_n=force_values))

        for force in force_values:
            branch = f"{int(force)}N"
            branch_obs, restored_physical = branch_restore(env, torch, scene_state, ref, helpers)
            if reference_live is None:
                reference_live = restored_physical
            parity = parity_row(reference_live, restored_physical, branch)
            parity_rows.append(parity)
            snapshot_xyz = np.asarray(restored_physical["object_position"], dtype=float)
            if scene_snapshot_only:
                # Scene-only post-query snapshots do not contain the action
                # manager's last VLA command.  Preserve the restored physical
                # hand pose for static settling; the post-settle continuation
                # remains the same frozen trace for every force branch.
                eef_now = np.asarray(branch_obs["policy"]["eef_pose"][0].detach().cpu().numpy(), dtype=np.float32)
                handoff_action[:3] = eef_now[:3]
                handoff_action[3:6] = quat_wxyz_to_axis_angle(eef_now[3:7])
            ctl = base.TaberoTruePhysicalForceHybrid()
            ctl.state = base.HybridState.FORCE_TRACK
            pre_target_controller_state = base.controller_state_payload(ctl)
            snapshot_aperture = float(np.mean(np.asarray(restored_physical["gripper_pos"], dtype=float)))
            if base_result.get("handoff_row", {}).get("joint_pos"):
                handoff_joint_pos = json.loads(base_result["handoff_row"]["joint_pos"])
                nominal_d = float(np.mean(np.asarray(handoff_joint_pos[-2:], dtype=float)))
            elif scene_snapshot_only:
                nominal_d = snapshot_aperture
            else:
                nominal_d = float(handoff_action[6])
            takeover_aperture = nominal_d
            previous_trace: dict[str, Any] = {}
            post_target_controller_state = base.controller_state_payload(ctl)
            rows: list[dict[str, Any]] = []
            static_vals: list[float] = []
            static_bilateral: list[int] = []
            static_tracking: list[int] = []
            terminated = False

            # Static realization happens in this same restored branch; there
            # is no second restore between settling and VLA continuation.
            for i in range(STATIC_STEPS):
                live = helpers.snapshot(env, branch_obs, int(ref.get("snapshot_step", os.environ.get("POST_GRASP_SNAPSHOT_STEP", "206"))) + i + 1, "post_grasp_static")
                sample = base.sample_from_row(live)
                result = ctl.step(F_des=force, d_nominal=nominal_d, sample=sample, execution_phase="static")
                action = handoff_action.copy()
                action[6] = result.d_final
                action[7:13] = 0.0
                live["_source_action"] = action.tolist()
                event = "HANDOFF" if i == 0 else ("CONTACT_LOSS" if not sample.bilateral_contact else "")
                row = physical_branch_row(live=live, result=result, force=force, branch=branch, phase="post_grasp_static", action_index=next_index, event=event, snapshot_xyz=snapshot_xyz, env=env, torch=torch)
                add_controller_trace(row, result=result, ctl=ctl, action=action, phase="post_grasp_static", snapshot_aperture=snapshot_aperture, takeover_aperture=takeover_aperture, previous=previous_trace, env=env)
                rows.append(row)
                static_vals.append(float(result.F_meas))
                static_bilateral.append(int(sample.bilateral_contact))
                static_tracking.append(int(sample.bilateral_contact and abs(result.F_meas - force) <= FORCE_TOLERANCE_N))
                nominal_d = float(result.d_final)
                branch_obs, _, term, trunc, _ = env.step(base.action_tensor(torch, action, env.device))
                terminated = bool(term[0].item()) or bool(trunc[0].item())
                if terminated or not sample.bilateral_contact:
                    break

            static_mean = float(np.mean(static_vals)) if static_vals else float("nan")
            static_mae = float(np.mean(np.abs(np.asarray(static_vals) - force))) if static_vals else float("nan")
            force_realization_valid = bool(
                parity["snapshot_restore_parity"]
                and len(static_vals) >= 10
                and float(np.mean(static_bilateral)) >= 0.8
                and static_mae <= FORCE_TOLERANCE_N
            )

            lift_success = False
            contact_loss_after_tracking = False
            force_tracking_seen = bool(static_tracking and max(static_tracking) == 1)
            vla_motion_disturbance = False
            dropped = False
            success_seen = False
            max_height_delta = 0.0
            last_action_index = next_index - 1
            if force_realization_valid and not terminated:
                for i, raw in enumerate(remaining):
                    action_index = next_index + i
                    live = helpers.snapshot(env, branch_obs, int(ref.get("snapshot_step", os.environ.get("POST_GRASP_SNAPSHOT_STEP", "206"))) + STATIC_STEPS + i + 1, "post_grasp_vla")
                    sample = base.sample_from_row(live)
                    nominal = np.asarray(raw, dtype=np.float32).copy()
                    result = ctl.step(F_des=force, d_nominal=float(nominal[6]), sample=sample, execution_phase="lift")
                    dz = float(live["object_z"] - snapshot_xyz[2])
                    max_height_delta = max(max_height_delta, dz)
                    if dz >= LIFT_THRESHOLD_M:
                        lift_success = True
                    if lift_success and not sample.bilateral_contact:
                        contact_loss_after_tracking = True
                    if lift_success and float(np.linalg.norm(nominal[:3] - handoff_action[:3])) > 0.002:
                        vla_motion_disturbance = True
                    dropped = dropped or safe_term(env, "object_1_dropped")
                    success_seen = success_seen or safe_term(env, "success")
                    live["_source_action"] = nominal.tolist()
                    event = ""
                    if i == 0 and float(nominal[2] - live["eef_z"]) > 0.002:
                        event = "LIFT_START"
                    if lift_success and not any(r.get("OBJECT_LEAVES_SUPPORT") for r in rows):
                        event = "OBJECT_LEAVES_SUPPORT"
                    if not sample.bilateral_contact:
                        event = "CONTACT_LOSS"
                    if success_seen:
                        event = "REMAINING_TASK_SUCCESS"
                    final_action = base.apply_to_tabero_nominal_action(
                        torch.from_numpy(nominal).to(env.device), result.d_final
                    ).detach().cpu().numpy()
                    row = physical_branch_row(live=live, result=result, force=force, branch=branch, phase="post_grasp_vla", action_index=action_index, event=event, snapshot_xyz=snapshot_xyz, env=env, torch=torch)
                    add_controller_trace(row, result=result, ctl=ctl, action=final_action, phase="post_grasp_vla", snapshot_aperture=snapshot_aperture, takeover_aperture=takeover_aperture, previous=previous_trace, env=env)
                    row.update(
                        {
                            "object_slip": int(contact_loss_after_tracking or dropped),
                            "lift_success_so_far": int(lift_success),
                            "transport_retention_so_far": int(lift_success and not contact_loss_after_tracking and not dropped),
                            "placement_success_so_far": int(success_seen),
                            "remaining_task_success_so_far": int(success_seen),
                            "vla_action_arm_0_6": json.dumps(nominal[:6].tolist()),
                        }
                    )
                    rows.append(row)
                    last_action_index = action_index
                    if not sample.bilateral_contact or bool(term[0].item()) or bool(trunc[0].item()) or success_seen:
                        break
                    branch_obs, _, term, trunc, _ = env.step(base.action_tensor(torch, final_action, env.device))
                    success_seen = success_seen or bool(term[0].item()) or safe_term(env, "success")
                    dropped = dropped or safe_term(env, "object_1_dropped")
                    if success_seen:
                        break
                    if bool(term[0].item()) or bool(trunc[0].item()):
                        break

            transport_retention = bool(lift_success and not contact_loss_after_tracking and not dropped)
            placement_success = bool(success_seen)
            remaining_success = bool(success_seen)
            if not force_realization_valid:
                failure = "LOW_LEVEL_FORCE_EXECUTION_FAILURE"
            elif remaining_success:
                failure = "NONE"
            elif contact_loss_after_tracking and vla_motion_disturbance:
                failure = "POST_HANDOFF_VLA_MOTION_CONTACT_FAILURE"
            else:
                failure = "POST_GRASP_FORCE_INSUFFICIENT"
            summary = {
                "branch": branch,
                "F_des": force,
                "F_target_eff": force,
                "F_meas": static_mean,
                "F_left_obj": float(np.mean([r["F_left_obj"] for r in rows if r["phase"] == "post_grasp_static"])) if static_vals else float("nan"),
                "F_right_obj": float(np.mean([r["F_right_obj"] for r in rows if r["phase"] == "post_grasp_static"])) if static_vals else float("nan"),
                "force_error": static_mean - force if np.isfinite(static_mean) else float("nan"),
                "bilateral_contact": int(bool(static_bilateral and np.mean(static_bilateral) >= 0.8)),
                "force_asymmetry": float(np.mean([r["force_asymmetry"] for r in rows if r["phase"] == "post_grasp_static"])) if static_vals else float("nan"),
                "aperture_command": float(rows[-1]["aperture_command"]) if rows else float("nan"),
                "controller_saturation": ";".join(sorted({str(r["controller_saturation"]) for r in rows})),
                "object_position_final": rows[-1]["object_position"] if rows else "",
                "object_orientation_final": rows[-1]["object_orientation"] if rows else "",
                "object_height_final": rows[-1]["object_height"] if rows else float("nan"),
                "object_height_delta_max_m": max_height_delta,
                "object_slip": int(contact_loss_after_tracking or dropped),
                "EE_motion_final": rows[-1]["EE_motion"] if rows else "",
                "VLA_action_index_start": next_index,
                "VLA_action_index_end": last_action_index,
                "VLA_continuation_action_count": len(remaining),
                "VLA_continuation_sha256": actual_trace_hash,
                "gripper_aperture_final": rows[-1]["gripper_aperture"] if rows else float("nan"),
                "lift_success": int(lift_success),
                "transport_retention": int(transport_retention),
                "placement_success": int(placement_success),
                "remaining_task_success": int(remaining_success),
                "force_realization_valid": int(force_realization_valid),
                "static_mean_force_N": static_mean,
                "static_mae_N": static_mae,
                "static_bilateral_rate": float(np.mean(static_bilateral)) if static_bilateral else 0.0,
                "static_tracking_rate_within_1N": float(np.mean(static_tracking)) if static_tracking else 0.0,
                "static_sample_count": len(static_vals),
                "contact_loss_after_tracking": int(contact_loss_after_tracking),
                "vla_motion_disturbance": int(vla_motion_disturbance),
                "dropped": int(dropped),
                "failure_type": failure,
                "SNAPSHOT_RESTORE_PARITY_VALID": bool(parity["snapshot_restore_parity"]),
                "VLA_CONTINUATION_PARITY_VALID": bool(actual_trace_hash == expected_trace_hash),
                "snapshot_id": snapshot_id,
                "branch_only_change": "F_des",
                "PRE_TARGET_INJECTION_CONTROLLER_STATE": pre_target_controller_state,
                "POST_TARGET_INJECTION_CONTROLLER_STATE": post_target_controller_state,
                "FIRST_TRACE_STEP": rows[0].get("step") if rows else None,
                "LAST_TRACE_STEP": rows[-1].get("step") if rows else None,
            }
            summaries.append(summary)
            all_rows.extend(rows)
            write_csv(out / f"{branch}_TIMESERIES.csv", rows)
            print(json.dumps({"branch": branch, "force_realization_valid": force_realization_valid, "static_mean_N": static_mean, "lift_success": lift_success, "remaining_task_success": remaining_success, "failure_type": failure}, sort_keys=True), flush=True)

        write_csv(out / "SNAPSHOT_BRANCH_PARITY.csv", parity_rows)
        write_csv(out / "POST_GRASP_FORCE_FRONTIER_TIMESERIES.csv", all_rows)
        write_csv(out / "POST_GRASP_FORCE_FRONTIER_SUMMARY.csv", summaries)

        valid = [s for s in summaries if s["force_realization_valid"] and s["SNAPSHOT_RESTORE_PARITY_VALID"] and s["VLA_CONTINUATION_PARITY_VALID"]]
        outcomes = {float(s["F_des"]): bool(s["remaining_task_success"]) for s in valid}
        force_dependent = "NO"
        if len({int(v) for v in outcomes.values()}) > 1:
            force_dependent = "YES"
        elif valid and len({s["failure_type"] for s in valid}) > 1:
            force_dependent = "PARTIAL"
        successes = sorted(force for force, outcome in outcomes.items() if outcome)
        failures = sorted(force for force, outcome in outcomes.items() if not outcome)
        if successes and failures:
            # For the ordered 2/4/6 grid, the interval uses the largest tested
            # failure below the first tested success.
            lower = max(force for force in failures if force < min(successes)) if any(force < min(successes) for force in failures) else min(successes)
            frontier_interval = f"({lower:g}N, {min(successes):g}N]"
        elif valid and all(outcomes.values()):
            frontier_interval = "F_star <= 2N (FRONTIER_BELOW_TEST_RANGE)"
        elif valid and not any(outcomes.values()):
            frontier_interval = "not identified: all tested branches failed"
        else:
            frontier_interval = "not identified: insufficient valid branches"
        if force_dependent == "YES":
            frontier_validated = "YES"
        elif force_dependent == "PARTIAL":
            frontier_validated = "PARTIAL"
        else:
            frontier_validated = "NO"

        final = {
            "FINAL_STATUS": "POST_GRASP_FORCE_FRONTIER_COMPLETE",
            "AUDITED_TASK": f"libero_10/task{base.TASK_ID}",
            "AUDITED_ROOT": base.ROOT,
            "SNAPSHOT_ID": snapshot_id,
            "SNAPSHOT_SOURCE": str(SNAPSHOT_PT),
            "SNAPSHOT_REFERENCE_SOURCE": str(REFERENCE_JSON),
            "SNAPSHOT_RESTORE_PARITY_VALID": "YES" if all(p["snapshot_restore_parity"] for p in parity_rows) else "NO",
            "VLA_CONTINUATION_PARITY_VALID": "YES" if actual_trace_hash == expected_trace_hash else "NO",
            "VLA_CONTINUATION_ACTION_SHA256": actual_trace_hash,
            "VLA_CONTINUATION_ACTION_COUNT": len(remaining),
            "BRANCH_ONLY_CHANGE": "F_des",
            "2N_FORCE_REALIZATION_VALID": "YES" if next((s["force_realization_valid"] for s in summaries if s["F_des"] == 2.0), 0) else "NO",
            "4N_FORCE_REALIZATION_VALID": "YES" if next((s["force_realization_valid"] for s in summaries if s["F_des"] == 4.0), 0) else "NO",
            "6N_FORCE_REALIZATION_VALID": "YES" if next((s["force_realization_valid"] for s in summaries if s["F_des"] == 6.0), 0) else "NO",
            "2N_STATIC_MEAN_FORCE": next((s["static_mean_force_N"] for s in summaries if s["F_des"] == 2.0), float("nan")),
            "4N_STATIC_MEAN_FORCE": next((s["static_mean_force_N"] for s in summaries if s["F_des"] == 4.0), float("nan")),
            "6N_STATIC_MEAN_FORCE": next((s["static_mean_force_N"] for s in summaries if s["F_des"] == 6.0), float("nan")),
            "2N_LIFT_SUCCESS": "YES" if next((s["lift_success"] for s in summaries if s["F_des"] == 2.0), 0) else "NO",
            "4N_LIFT_SUCCESS": "YES" if next((s["lift_success"] for s in summaries if s["F_des"] == 4.0), 0) else "NO",
            "6N_LIFT_SUCCESS": "YES" if next((s["lift_success"] for s in summaries if s["F_des"] == 6.0), 0) else "NO",
            "2N_REMAINING_TASK_SUCCESS": "YES" if next((s["remaining_task_success"] for s in summaries if s["F_des"] == 2.0), 0) else "NO",
            "4N_REMAINING_TASK_SUCCESS": "YES" if next((s["remaining_task_success"] for s in summaries if s["F_des"] == 4.0), 0) else "NO",
            "6N_REMAINING_TASK_SUCCESS": "YES" if next((s["remaining_task_success"] for s in summaries if s["F_des"] == 6.0), 0) else "NO",
            "2N_FAILURE_TYPE": next((s["failure_type"] for s in summaries if s["F_des"] == 2.0), "NOT_RUN"),
            "4N_FAILURE_TYPE": next((s["failure_type"] for s in summaries if s["F_des"] == 4.0), "NOT_RUN"),
            "6N_FAILURE_TYPE": next((s["failure_type"] for s in summaries if s["F_des"] == 6.0), "NOT_RUN"),
            "FORCE_DEPENDENT_OUTCOME": force_dependent,
            "POST_GRASP_FORCE_FRONTIER_VALIDATED": frontier_validated,
            "FRONTIER_INTERVAL": frontier_interval,
            "FORCE_REALIZATION_RULE": "static samples >=10, bilateral rate >=0.8, MAE <=1.0N, and restore parity",
            "FORMAL_FRONTIER_TRAJECTORY": "Frozen VLA original normal remaining trajectory",
            "PURE_VERTICAL_DIAGNOSTIC": "NOT_RUN",
            "POST_HANDOFF_VLA_MOTION_CONTACT_FAILURE": any(s["failure_type"] == "POST_HANDOFF_VLA_MOTION_CONTACT_FAILURE" for s in summaries),
            "LOW_LEVEL_FORCE_EXECUTION_FAILURE": any(s["failure_type"] == "LOW_LEVEL_FORCE_EXECUTION_FAILURE" for s in summaries),
            "READY_FOR_5_CONTEXT_PER_TASK_PILOT": frontier_validated == "YES",
            "READY_FOR_CONTINUOUS_POSTERIOR_TRAINING": frontier_validated == "YES",
            "same_state_branch_contract": branch_contract,
            "provenance": {
                "previous_validated_snapshot_restore_parity": base_result.get("SNAPSHOT_RESTORE_PARITY_VALID"),
                "previous_validated_4n_repeatability": base_result.get("4N_REPEATABILITY_VALID"),
                "probe_sha256": base_result.get("probe_sha256"),
                "probe_feature_dimension": base_result.get("PROBE_FEATURE_DIMENSION"),
                "friction_posterior_runtime_valid": base_result.get("FRICTION_POSTERIOR_RUNTIME_VALID"),
                "frozen_vla_remaining_action_sha256": expected_trace_hash,
            },
        }
        write_json(out / "POST_GRASP_FORCE_FRONTIER_RESULT.json", final)
        write_json(out / "POST_GRASP_FORCE_FRONTIER_PROTOCOL.json", {
            "protocol": "RUN SAME-STATE 2N / 4N / 6N POST-GRASP FORCE FRONTIER",
            "task": f"Task{base.TASK_ID}",
            "root": base.ROOT,
            "snapshot": snapshot_id,
            "forces_N": list(force_values),
            "only_change": "F_des",
            "approach_grasp_probe_rerun": False,
            "pure_vertical": False,
            "continuation": "same frozen VLA action sequence, cursor, phase semantics",
        })
        return 0
    except Exception as exc:
        exit_code = 1
        (out / "POST_GRASP_FORCE_FRONTIER_ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(out / "POST_GRASP_FORCE_FRONTIER_RESULT.json", {"FINAL_STATUS": "POST_GRASP_FORCE_FRONTIER_RUNTIME_ERROR", "error": repr(exc), "snapshot": str(SNAPSHOT_PT)})
    finally:
        try:
            if env is not None and term_cfg_snapshot is not None:
                from tac_manip.tasks.manipulation.libero import mdp
                mdp.restore_tabero_legacy_force_loop(env.action_manager.get_term("arm_action"), term_cfg_snapshot)
        except Exception:
            pass
        # Isaac camera teardown can raise a native weak-reference exception;
        # terminate after all evidence is flushed, matching the validated
        # runtime runner's teardown policy.
        os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(run())
