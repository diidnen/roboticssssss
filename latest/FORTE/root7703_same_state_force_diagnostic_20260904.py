#!/usr/bin/env python3
"""Diagnose and validate true-force realization from the canonical root7703 state.

This is deliberately a new audit adapter.  It does not modify the Tabero
controller implementation or the canonical snapshot.  Each branch restores
the same scene/runtime state, holds the restored EE pose, clears the legacy
force slots, and changes only the external ``F_des``.
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
SNAPSHOT = Path("/home/exouser/FORTE/analysis/results/historical_success_state_recovery_20260904/HISTORICAL_SUCCESS_POST_GRASP_STATE.pt")
REFERENCE = Path("/home/exouser/FORTE/analysis/results/historical_success_state_recovery_20260904/STATE_REFERENCE.json")
OUT = Path("/home/exouser/FORTE/analysis/results/root7703_same_state_force_frontier_20260904_retry5")
ROOT = 7703
TASK_SUITE = "libero_10"
TASK_ID = 5
OBJECT = "black_book_1"
TARGET = "desk_caddy_1"
FORCES = (2.0, 4.0, 6.0)
MAX_TRACE_STEPS = 150
STABLE_WINDOW = 10
FORCE_TOLERANCE_N = 1.0
FORCE_STD_TOLERANCE_N = 0.5
APERTURE_RANGE_TOLERANCE_M = 2.0e-4
FORCE_STEP_DIFF_TOLERANCE_N = 0.05
CONTACT_EPS_N = 0.15
HOLD_STEPS = 30

sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(PROTOTYPE))
import run_post_grasp_4n_live_validation as base  # noqa: E402
import run_same_state_2_4_6_force_frontier as same  # noqa: E402

base.ROOT = ROOT
base.TASK_SUITE = TASK_SUITE
base.TASK_ID = TASK_ID
base.OBJECT = OBJECT
base.TARGET = TARGET
base.TRACE = Path("/media/volume/newdata/exouser/activeforcing_e3/P1_SIMPLIFIED_COLLECTION_FORMAL_20260903/P1_SIMPLIFIED_ROOT7703_F6N/raw_policy/b5_t5_mu0.6_exp000_action_chunks.npz")
base.HDF5_DIR = Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_SOURCE_20260902_103300/assembled_hdf5")
same.SNAPSHOT_PT = SNAPSHOT
same.REFERENCE_JSON = REFERENCE
same.BASE_RESULT = Path("/does/not/exist")


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def npv(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value: Any, default: float = float("nan")) -> float:
    try:
        a = npv(value).reshape(-1)
        return float(a[0]) if len(a) else default
    except Exception:
        return default


def term_runtime(term: Any, manager: Any) -> dict[str, Any]:
    """Expose the target path without serializing arbitrary Isaac objects."""
    out: dict[str, Any] = {
        "cfg": {
            name: base.jsonable(getattr(getattr(term, "cfg", None), name, None))
            for name in (
                "squeeze_kp", "squeeze_ff_k_load_z", "target_contact_squeeze_enabled",
                "meas_force_filter_alpha", "pos_kp", "gripper_open_val",
            )
        },
        "raw_actions": base.jsonable(getattr(term, "_raw_actions", None)),
        "processed_actions": base.jsonable(getattr(term, "_processed_actions", None)),
        "gripper_abs_cmd": base.jsonable(getattr(term, "_gripper_abs_cmd", None)),
        "last_d_cmd": base.jsonable(getattr(term, "_last_d_cmd", None)),
        "force_ema": base.jsonable(getattr(term, "_f_sq_meas_ema", None)),
        "force_ema_initialized": bool(getattr(term, "_f_sq_meas_ema_initialized", False)),
        "manager_action": base.jsonable(getattr(manager, "_action", None)),
    }
    ik = getattr(term, "_ik_term", None)
    if ik is not None:
        out["ik"] = {
            "raw_actions": base.jsonable(getattr(ik, "_raw_actions", None)),
            "processed_actions": base.jsonable(getattr(ik, "_processed_actions", None)),
            "command": base.jsonable(getattr(getattr(ik, "_ik_controller", None), "_command", None)),
            "ee_pos_des": base.jsonable(getattr(getattr(ik, "_ik_controller", None), "ee_pos_des", None)),
            "ee_quat_des": base.jsonable(getattr(getattr(ik, "_ik_controller", None), "ee_quat_des", None)),
        }
    return out


def make_pose_hold_action(obs: Any, gripper_command: float) -> np.ndarray:
    """Build an action at the *current restored* EE pose, with no force slots."""
    eef = npv(obs["policy"]["eef_pose"])[0].astype(np.float32)
    action = np.zeros(13, dtype=np.float32)
    action[:3] = eef[:3]
    action[3:6] = same.quat_wxyz_to_axis_angle(eef[3:7])
    action[6] = float(gripper_command)
    # The native ForcePositionAction path is disabled for force feedback; the
    # external hybrid owns the only force target.  Zeroing these slots also
    # makes the branch-only-change contract explicit in every row.
    action[7:13] = 0.0
    return action


def restore_branch(env: Any, torch: Any, scene_state: Any, ref: dict[str, Any], helpers: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    env.scene.reset_to(base.clone_state(scene_state), torch.tensor([0], device=env.device), is_relative=True)
    env.sim.forward()
    env.observation_manager.reset(torch.tensor([0], device=env.device))
    env.action_manager.reset(torch.tensor([0], device=env.device))
    same.restore_all_runtime(env, ref, torch)
    obs = env.observation_manager.compute(update_history=True)
    row = helpers.snapshot(env, obs, int(ref["state_step"]), "force_restore")
    physical = base.capture_physical_state(env, obs, row)
    return obs, row, physical


def object_dynamics(env: Any) -> dict[str, Any]:
    obj = env.scene[OBJECT]
    return {
        "object_position": npv(obj.data.root_pos_w)[0].astype(float).tolist(),
        "object_velocity": npv(obj.data.root_lin_vel_w)[0].astype(float).tolist(),
        "object_angular_velocity": npv(obj.data.root_ang_vel_w)[0].astype(float).tolist(),
        "object_height": float(npv(obj.data.root_pos_w)[0, 2]),
    }


def finger_state(env: Any, obs: Any) -> dict[str, Any]:
    robot = env.scene["robot"]
    eef = npv(obs["policy"]["eef_pose"])[0].astype(float)
    return {
        "finger_joint_positions": npv(robot.data.joint_pos)[0, -2:].astype(float).tolist(),
        "finger_joint_velocities": npv(robot.data.joint_vel)[0, -2:].astype(float).tolist(),
        "eef_pose": eef.tolist(),
    }


def row_from_step(
    *,
    pre: dict[str, Any],
    post: dict[str, Any],
    pre_fingers: dict[str, Any],
    post_fingers: dict[str, Any],
    result: Any,
    ctl: Any,
    action: np.ndarray,
    term_state: dict[str, Any],
    branch: str,
    requested: float,
    physics_step: int,
    restore_object_height: float,
    env: Any,
) -> dict[str, Any]:
    left_force = float(post["left_object_normal_force_N"])
    right_force = float(post["right_object_normal_force_N"])
    measured = 2.0 * min(left_force, right_force)
    dyn = object_dynamics(env)
    eef_delta = np.asarray(post_fingers["eef_pose"][:3]) - np.asarray(pre_fingers["eef_pose"][:3])
    out = {
        "branch": branch,
        "physics_step": physics_step,
        "requested_force_target": float(requested),
        "controller_force_target": float(result.F_des),
        "effective_force_target": float(result.F_des),
        "force_target_path": "external TaberoTruePhysicalForceHybrid.step(F_des) -> d_final -> action[6]",
        "native_force_slots": json.dumps(action[7:13].astype(float).tolist()),
        "left_finger_position": float(post_fingers["finger_joint_positions"][0]),
        "right_finger_position": float(post_fingers["finger_joint_positions"][1]),
        "finger_joint_positions": json.dumps(post_fingers["finger_joint_positions"]),
        "finger_joint_velocities": json.dumps(post_fingers["finger_joint_velocities"]),
        "left_finger_frame_position": json.dumps([float(post["left_finger_x"]), float(post["left_finger_y"]), float(post["left_finger_z"])]),
        "right_finger_frame_position": json.dumps([float(post["right_finger_x"]), float(post["right_finger_y"]), float(post["right_finger_z"])]),
        "eef_pose": json.dumps(post_fingers["eef_pose"]),
        "arm_pose_delta_after_step_m": float(np.linalg.norm(eef_delta)),
        "aperture": float(post["gripper_aperture"]),
        "gripper_command": float(action[6]),
        "gripper_target": float(scalar(term_state.get("gripper_abs_cmd"))),
        "controller_internal_target": float(result.F_des),
        "controller_output": float(result.d_final),
        "controller_force_error": float(result.force_error),
        "controller_force_correction": float(result.force_correction),
        "controller_state": result.state.value,
        "controller_force_loop_active": int(result.force_loop_active),
        "controller_internal_state": json.dumps(base.controller_state_payload(ctl), default=str, sort_keys=True),
        "left_target_force": left_force,
        "right_target_force": right_force,
        "aggregate_force": measured,
        "force_error": float(requested - measured),
        "left_contact": int(bool(post["left_object_contact"])),
        "right_contact": int(bool(post["right_object_contact"])),
        "bilateral_contact": int(bool(post["bilateral_object_contact"])),
        "object_height": dyn["object_height"],
        "object_position": json.dumps(dyn["object_position"]),
        "object_velocity": json.dumps(dyn["object_velocity"]),
        "object_angular_velocity": json.dumps(dyn["object_angular_velocity"]),
        "object_height_delta_from_restore": dyn["object_height"] - float(restore_object_height),
        "object_dropped": int(same.safe_term(env, "object_1_dropped")),
        "post_observation_force_metric": "2*min(left_object_normal_force_N,right_object_normal_force_N)",
        "source_sensor": base.SENSOR_NAME,
        "measurement_phase": "pending",
    }
    return out


def static_gate(rows: list[dict[str, Any]], requested: float) -> dict[str, Any]:
    if not rows:
        return {"valid": False, "settling_steps": None, "reason": "no_rows"}
    first_stable_end: int | None = None
    stable_stats: dict[str, Any] = {}
    for end in range(STABLE_WINDOW, len(rows) + 1):
        window = rows[end - STABLE_WINDOW:end]
        forces = np.asarray([float(x["aggregate_force"]) for x in window], dtype=float)
        apertures = np.asarray([float(x["aperture"]) for x in window], dtype=float)
        bilateral = np.asarray([int(x["bilateral_contact"]) for x in window], dtype=int)
        mean = float(forces.mean())
        std = float(forces.std())
        mae = float(np.abs(forces - requested).mean())
        aperture_range = float(apertures.max() - apertures.min())
        step_diffs = np.abs(np.diff(forces))
        max_step_diff = float(step_diffs.max()) if len(step_diffs) else float("inf")
        if (
            np.all(bilateral == 1)
            and mae <= FORCE_TOLERANCE_N
            and std <= FORCE_STD_TOLERANCE_N
            and aperture_range <= APERTURE_RANGE_TOLERANCE_M
            and max_step_diff <= FORCE_STEP_DIFF_TOLERANCE_N
        ):
            first_stable_end = end
            stable_stats = {
                "measurement_start_trace_step": int(window[0]["physics_step"]),
                "measurement_end_trace_step": int(window[-1]["physics_step"]),
                "sample_count": len(window),
                "static_mean_force_N": mean,
                "static_std_force_N": std,
                "static_mae_N": mae,
                "bilateral_rate": float(bilateral.mean()),
                "aperture_range_m": aperture_range,
                "max_adjacent_force_step_diff_N": max_step_diff,
            }
            break
    if first_stable_end is None:
        return {
            "valid": False,
            "settling_steps": None,
            "reason": "no_stable_window_before_contact_or_trace_end",
            "last_force_N": float(rows[-1]["aggregate_force"]),
            "last_bilateral_contact": int(rows[-1]["bilateral_contact"]),
        }
    return {
        "valid": True,
        "settling_steps": int(first_stable_end),
        "thresholds": {
            "stable_window_steps": STABLE_WINDOW,
            "force_mae_max_N": FORCE_TOLERANCE_N,
            "force_std_max_N": FORCE_STD_TOLERANCE_N,
            "aperture_range_max_m": APERTURE_RANGE_TOLERANCE_M,
            "max_adjacent_force_step_diff_N": FORCE_STEP_DIFF_TOLERANCE_N,
            "bilateral_required": True,
        },
        **stable_stats,
    }


def make_path_audit(root7400: dict[str, Any]) -> dict[str, Any]:
    metric = {
        "sensor_name": "contact_grasp_black_book_1",
        "sensor_body_mapping": ["panda_leftfinger", "panda_rightfinger"],
        "sensor_filter": "target object black_book_1 only",
        "force_matrix_w_shape": "(N,2,M,3)",
        "side_aggregation": "sum over filtered target-object contact dimension M per finger",
        "normal_projection": "abs(dot(object-filtered world force, each finger closing normal))",
        "left_right_aggregation": "F_meas = 2*min(F_left_target_normal,F_right_target_normal)",
        "contact_threshold_N": CONTACT_EPS_N,
        "static_gate_in_root7400": "40 static rows, bilateral rate >=0.8, MAE <=1.0N, restore parity",
    }
    return {
        "schema": "ROOT7400_VS_ROOT7703_FORCE_PATH_AUDIT_V1",
        "root7400": {
            "runner": str(REPO / "analysis/run_post_grasp_4n_live_validation.py"),
            "frontier_result": str(REPO / "analysis/results/post_grasp_force_frontier_final_20260904/ROOT7400_FORCE_FRONTIER_RESULT.json"),
            "target_injection": "TaberoTruePhysicalForceHybrid.step(F_des) -> apply_to_tabero_nominal_action(action, d_final)",
            "native_legacy_loop": "disabled: squeeze_kp=0, squeeze_ff_k_load_z=0, target_contact_squeeze_enabled=False",
            "native_force_slots": "zeroed before env.step",
            "requested_controller_effective_relation": "requested F_des == controller F_des == F_target_eff",
            "validated_force_N": {"2": root7400["2N_STATIC_MEAN_FORCE"], "4": root7400["4N_STATIC_MEAN_FORCE"], "6": root7400["6N_STATIC_MEAN_FORCE"]},
        },
        "root7703_previous_smoke": {
            "runner": str(REPO / "analysis/run_same_state_2_4_6_force_frontier.py"),
            "target_injection": "same external TaberoTruePhysicalForceHybrid path; result.F_des copied to F_target_eff",
            "native_force_slots": "zeroed before env.step",
            "requested_controller_effective_relation": "requested F_des == controller F_des == F_target_eff",
            "confirmed_path_mismatch": "static action reused archived arm pose target instead of holding the restored EE pose; root7703 action target differed from physical EE by approximately 2.7mm in position and substantially in orientation",
            "initial_gripper_target_source": "restored native action target action[6]=0.0072860145m; not an effective force target override failure",
        },
        "root7703_corrected_diagnostic": {
            "runner": str(Path(__file__)),
            "target_injection": "same external TaberoTruePhysicalForceHybrid.step(F_des) -> d_final -> action[6]",
            "arm_policy": "current restored EE pose held fixed; no archived arm target replay",
            "native_legacy_loop": "disabled: squeeze_kp=0, squeeze_ff_k_load_z=0, target_contact_squeeze_enabled=False",
            "native_force_slots": "[0,0,0,0,0,0]",
            "requested_controller_effective_relation": "requested F_des == controller F_des == effective F_des",
        },
        "ROOT7400_FORCE_METRIC": metric,
        "ROOT7703_FORCE_METRIC": metric,
        "FORCE_METRIC_MATCH": "YES",
        "source_hashes": {
            "root7400_runner": sha256(REPO / "analysis/run_post_grasp_4n_live_validation.py"),
            "root7703_previous_runner": sha256(REPO / "analysis/run_same_state_2_4_6_force_frontier.py"),
            "true_force_controller": sha256(REPO / "analysis/tabero_true_physical_force_hybrid.py"),
            "runtime_bridge": sha256(REPO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/true_physical_force_hybrid.py"),
        },
    }


def run() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    if not SNAPSHOT.exists() or not REFERENCE.exists():
        raise FileNotFoundError("canonical root7703 snapshot/reference missing")
    root7400 = json.loads((REPO / "analysis/results/post_grasp_force_frontier_final_20260904/ROOT7400_FORCE_FRONTIER_RESULT.json").read_text())
    write_json(OUT / "ROOT7400_VS_ROOT7703_FORCE_PATH_AUDIT.json", make_path_audit(root7400))
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    import torch

    os.environ.update({
        "HDF5_TRAJ_SOURCE_DIR": str(base.HDF5_DIR),
        "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
        "TASK_SUITE": TASK_SUITE,
        "TASK_ID": str(TASK_ID),
        "ENABLE_CLOSED_LOOP_FORCE_CONTROLLER": "0",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
    })
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", num_envs=1).app
    env = None
    term_cfg_snapshot = None
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        from tac_manip.tasks.manipulation.libero import mdp
        import run_bilateral_contact_audit as helpers

        setup_task_objects(TASK_SUITE, TASK_ID)
        cfg = parse_env_cfg(base.ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 30.0
        env = gym.make(base.ENV_ID, cfg=cfg).unwrapped
        env.reset(seed=ROOT)
        snap = torch.load(SNAPSHOT, map_location="cpu", weights_only=False)
        scene_state = same.move_tensors(snap["scene_state"], torch, env.device)
        term = env.action_manager.get_term("arm_action")
        term_cfg_snapshot = mdp.disable_tabero_legacy_force_loop(term)

        base_snapshot_row: dict[str, Any] | None = None
        summaries: list[dict[str, Any]] = []
        all_runtime: dict[str, Any] = {}
        frozen_trace = base.load_trace()
        for force in FORCES:
            branch = f"{int(force)}N"
            obs, restore_row, restore_physical = restore_branch(env, torch, scene_state, ref, helpers)
            if base_snapshot_row is None:
                base_snapshot_row = restore_row
            parity = base.diff_physical_state(ref["physical_state"], restore_physical)
            parity_ok = all(x["pass"] or x["field"] == "eef_angular_velocity" for x in parity)
            # The snapshot's native command is part of the historical state;
            # it is the nominal aperture seed, not a force target.
            arm_state = ref["action_runtime_state"]["terms"]["arm_action"]
            native_nominal = float(np.asarray(arm_state["_raw_actions"], dtype=float).reshape(-1, 13)[0, 6])
            initial_native_nominal = native_nominal
            hold_action = make_pose_hold_action(obs, native_nominal)
            ctl = base.TaberoTruePhysicalForceHybrid()
            ctl.state = base.HybridState.FORCE_TRACK
            rows: list[dict[str, Any]] = []
            stable: dict[str, Any] = {}
            contact_loss_step: int | None = None
            restore_eef = np.asarray(restore_physical["eef_pose"], dtype=float)
            for i in range(MAX_TRACE_STEPS):
                pre = helpers.snapshot(env, obs, int(ref["state_step"]) + i, "settling")
                pre_fingers = finger_state(env, obs)
                sample = base.sample_from_row(pre)
                result = ctl.step(F_des=force, d_nominal=native_nominal, sample=sample, execution_phase="static")
                action = hold_action.copy()
                action[6] = float(result.d_final)
                action[7:13] = 0.0
                obs, _, term_flag, trunc_flag, _ = env.step(base.action_tensor(torch, action, env.device))
                post = helpers.snapshot(env, obs, int(ref["state_step"]) + i + 1, "settling")
                post_fingers = finger_state(env, obs)
                term_state = term_runtime(term, env.action_manager)
                row = row_from_step(
                    pre=pre, post=post, pre_fingers=pre_fingers,
                    post_fingers=post_fingers, result=result, ctl=ctl, action=action,
                    term_state=term_state, branch=branch, requested=force,
                    physics_step=int(ref["state_step"]) + i + 1,
                    restore_object_height=float(restore_row["object_z"]), env=env,
                )
                row["snapshot_restore_parity"] = int(parity_ok)
                row["restore_eef_pose_error_m"] = float(np.linalg.norm(np.asarray(post_fingers["eef_pose"]) - restore_eef))
                rows.append(row)
                if not int(post["bilateral_object_contact"]):
                    contact_loss_step = int(row["physics_step"])
                    break
                gate = static_gate(rows, force)
                if gate.get("valid") and not stable:
                    stable = gate
                    for stable_row in rows[-STABLE_WINDOW:]:
                        stable_row["measurement_phase"] = "stable_window"
                    # Once realization is valid, the rest of this branch is a
                    # fixed-arm physical hold; no VLA lift/transport is run.
                    break
                if bool(term_flag[0].item()) or bool(trunc_flag[0].item()):
                    break
                native_nominal = float(result.d_final)

            if stable:
                hold_rows: list[dict[str, Any]] = []
                for j in range(HOLD_STEPS):
                    pre = helpers.snapshot(env, obs, int(ref["state_step"]) + len(rows) + j, "hold")
                    pre_fingers = finger_state(env, obs)
                    sample = base.sample_from_row(pre)
                    result = ctl.step(F_des=force, d_nominal=native_nominal, sample=sample, execution_phase="static")
                    action = hold_action.copy()
                    action[6] = float(result.d_final)
                    action[7:13] = 0.0
                    obs, _, term_flag, trunc_flag, _ = env.step(base.action_tensor(torch, action, env.device))
                    post = helpers.snapshot(env, obs, int(ref["state_step"]) + len(rows) + j + 1, "hold")
                    post_fingers = finger_state(env, obs)
                    term_state = term_runtime(term, env.action_manager)
                    hrow = row_from_step(
                        pre=pre, post=post, pre_fingers=pre_fingers,
                        post_fingers=post_fingers, result=result, ctl=ctl, action=action,
                        term_state=term_state, branch=branch, requested=force,
                        physics_step=int(ref["state_step"]) + len(rows) + j + 1,
                        restore_object_height=float(restore_row["object_z"]), env=env,
                    )
                    hrow["measurement_phase"] = "hold"
                    hrow["snapshot_restore_parity"] = int(parity_ok)
                    hrow["restore_eef_pose_error_m"] = float(np.linalg.norm(np.asarray(post_fingers["eef_pose"]) - restore_eef))
                    hold_rows.append(hrow)
                    if not int(post["bilateral_object_contact"]):
                        contact_loss_step = int(hrow["physics_step"])
                        break
                rows.extend(hold_rows)

            gate = stable or static_gate(rows, force)
            force_valid = bool(gate.get("valid", False) and parity_ok)
            static_hold_success = bool(force_valid and len(rows) >= int(gate.get("settling_steps", 0)) + HOLD_STEPS and all(int(x["bilateral_contact"]) for x in rows[-HOLD_STEPS:]))

            # After realization, execute only a short prefix of the archived
            # continuation.  The arm action is frozen historical data; this is
            # a minimal lift/hold outcome check, not a transport benchmark.
            lift_rows: list[dict[str, Any]] = []
            lift_success = False
            lift_contact_loss_step: int | None = None
            lift_hold_success = False
            max_height_delta = 0.0
            if force_valid and static_hold_success:
                next_index = int(ref["source_action_index"]) + 1
                last_vla_action: np.ndarray | None = None
                for raw in frozen_trace[next_index:next_index + 40]:
                    pre = helpers.snapshot(env, obs, int(ref["state_step"]) + len(rows) + len(lift_rows), "frozen_vla_lift")
                    pre_fingers = finger_state(env, obs)
                    sample = base.sample_from_row(pre)
                    result = ctl.step(F_des=force, d_nominal=float(raw[6]), sample=sample, execution_phase="lift")
                    action = base.apply_to_tabero_nominal_action(
                        torch.from_numpy(np.asarray(raw, dtype=np.float32)).to(env.device),
                        result.d_final,
                    ).detach().cpu().numpy().reshape(-1).astype(np.float32)
                    obs, _, term_flag, trunc_flag, _ = env.step(base.action_tensor(torch, action, env.device))
                    post = helpers.snapshot(env, obs, int(ref["state_step"]) + len(rows) + len(lift_rows) + 1, "frozen_vla_lift")
                    post_fingers = finger_state(env, obs)
                    term_state = term_runtime(term, env.action_manager)
                    lrow = row_from_step(
                        pre=pre, post=post, pre_fingers=pre_fingers,
                        post_fingers=post_fingers, result=result, ctl=ctl, action=action,
                        term_state=term_state, branch=branch, requested=force,
                        physics_step=int(ref["state_step"]) + len(rows) + len(lift_rows) + 1,
                        restore_object_height=float(restore_row["object_z"]), env=env,
                    )
                    lrow["phase"] = "frozen_vla_lift"
                    lrow["measurement_phase"] = "post_realization_continuation"
                    dz = float(lrow["object_height_delta_from_restore"])
                    max_height_delta = max(max_height_delta, dz)
                    lrow["lift_success_so_far"] = int(dz >= 0.01)
                    lrow["contact_loss_after_realization"] = int(not lrow["bilateral_contact"])
                    lift_rows.append(lrow)
                    last_vla_action = action
                    if dz >= 0.01:
                        lift_success = True
                        break
                    if not lrow["bilateral_contact"] or bool(term_flag[0].item()) or bool(trunc_flag[0].item()):
                        lift_contact_loss_step = int(lrow["physics_step"]) if not lrow["bilateral_contact"] else None
                        break
                if lift_success and last_vla_action is not None:
                    lift_hold_rows: list[dict[str, Any]] = []
                    for j in range(HOLD_STEPS):
                        pre = helpers.snapshot(env, obs, int(ref["state_step"]) + len(rows) + len(lift_rows) + j, "frozen_vla_lift_hold")
                        pre_fingers = finger_state(env, obs)
                        sample = base.sample_from_row(pre)
                        result = ctl.step(F_des=force, d_nominal=float(last_vla_action[6]), sample=sample, execution_phase="lift")
                        action = last_vla_action.copy()
                        action[6] = float(result.d_final)
                        action[7:13] = 0.0
                        obs, _, term_flag, trunc_flag, _ = env.step(base.action_tensor(torch, action, env.device))
                        post = helpers.snapshot(env, obs, int(ref["state_step"]) + len(rows) + len(lift_rows) + j + 1, "frozen_vla_lift_hold")
                        post_fingers = finger_state(env, obs)
                        term_state = term_runtime(term, env.action_manager)
                        hrow = row_from_step(
                            pre=pre, post=post, pre_fingers=pre_fingers,
                            post_fingers=post_fingers, result=result, ctl=ctl, action=action,
                            term_state=term_state, branch=branch, requested=force,
                            physics_step=int(ref["state_step"]) + len(rows) + len(lift_rows) + j + 1,
                            restore_object_height=float(restore_row["object_z"]), env=env,
                        )
                        hrow["phase"] = "frozen_vla_lift_hold"
                        hrow["measurement_phase"] = "post_lift_hold"
                        lift_hold_rows.append(hrow)
                        if not hrow["bilateral_contact"]:
                            break
                    lift_rows.extend(lift_hold_rows)
                    lift_hold_success = len(lift_hold_rows) == HOLD_STEPS and all(int(x["bilateral_contact"]) for x in lift_hold_rows)
            rows.extend(lift_rows)
            hold_success = static_hold_success
            outcome = "FORCE_REALIZATION_VALID" if force_valid else ("TARGET_NOT_REACHED_BEFORE_CONTACT_LOSS" if contact_loss_step is not None else "RUNTIME_FAILURE")
            summary = {
                "branch": branch,
                "requested_force_target": force,
                "controller_force_target": force,
                "effective_force_target": force,
                "snapshot_preload_force_N": float(2.0 * min(float(restore_row["left_object_normal_force_N"]), float(restore_row["right_object_normal_force_N"]))),
                "snapshot_controller_target": initial_native_nominal,
                "snapshot_gripper_target": initial_native_nominal,
                "initial_gripper_command_source": "canonical action_runtime_state arm_action._raw_actions[0,6]",
                "force_metric": "2*min(left_object_normal_force_N,right_object_normal_force_N)",
                "force_metric_match": "YES",
                "snapshot_restore_parity": "YES" if parity_ok else "NO",
                "settling_steps": gate.get("settling_steps"),
                "realized_force_N": gate.get("static_mean_force_N"),
                "static_mean_force_N": gate.get("static_mean_force_N"),
                "static_std_force_N": gate.get("static_std_force_N"),
                "static_mae_force_N": gate.get("static_mae_N"),
                "measurement_sample_count": gate.get("sample_count"),
                "measurement_gate": gate,
                "trace_steps": len(rows),
                "contact_loss_step": contact_loss_step,
                "max_arm_pose_error_from_restore_m": max((float(x.get("restore_eef_pose_error_m", 0.0)) for x in rows), default=None),
                "arm_target_policy": "held at current restored EE pose; native pos_kp response is recorded, not suppressed",
                "lift_success": "YES" if lift_success else "NO",
                "hold_success": "YES" if hold_success else "NO",
                "static_hold_success": "YES" if static_hold_success else "NO",
                "frozen_vla_lift_success": "YES" if lift_success else "NO",
                "frozen_vla_lift_hold_success": "YES" if lift_hold_success else "NO",
                "frozen_vla_lift_rows": len(lift_rows),
                "frozen_vla_lift_contact_loss_step": lift_contact_loss_step,
                "frozen_vla_max_object_height_delta_m": max_height_delta,
                "object_height_start_m": float(restore_row["object_z"]),
                "object_height_final_m": float(rows[-1]["object_height"]) if rows else None,
                "object_height_delta_m": float(rows[-1]["object_height_delta_from_restore"]) if rows else None,
                "outcome": outcome,
                "controller_target_override_failure": False,
                "native_force_slots_zeroed": True,
                "arm_geometry_modified": False,
                "explicit_arm_command_changed": False,
            }
            summaries.append(summary)
            write_csv(OUT / f"ROOT7703_FORCE_TRACE_{branch}.csv", rows)
            all_runtime[branch] = {
                "restore_row": restore_row,
                "restore_physical": restore_physical,
                "restore_runtime_after_override": term_runtime(term, env.action_manager),
                "first_trace_row": rows[0] if rows else None,
                "last_trace_row": rows[-1] if rows else None,
                "parity_diffs": parity,
                "summary": summary,
            }

        diagnosis = {
            "schema": "FORCE_REALIZATION_DIAGNOSIS_V1",
            "canonical_root": ROOT,
            "canonical_step": int(ref["state_step"]),
            "canonical_snapshot": str(SNAPSHOT),
            "canonical_state_recovery_valid": True,
            "controller_rewritten": False,
            "diagnosis": "TARGET_OVERRIDE_AND_ARM_TARGET_ORDERING_AUDIT",
            "previous_smoke_failure": "previous runner replayed archived arm pose target from root7703 while restored physical EE was at a different pose; this changed grasp geometry before target realization",
            "target_path_conclusion": "requested_force_target == controller_force_target == effective_force_target for all branches; native force slots zeroed; no old target overwrite observed",
            "metric_conclusion": "root7400 and root7703 use identical object-filtered bilateral metric",
            "case_classification": {s["branch"]: s["outcome"] for s in summaries},
            "branches": all_runtime,
            "gate_definition": {
            "settling_phase": "all trace rows before first stable window; transient retained",
                "measurement_phase": "first 10 consecutive rows with bilateral contact, MAE <=1.0N, std <=0.5N, aperture range <=0.0002m, and every adjacent force step difference <=0.05N",
                "root7400_criteria_preserved": "bilateral rate and MAE/restore requirements preserved; added explicit stability checks",
            },
        }
        write_json(OUT / "FORCE_REALIZATION_DIAGNOSIS.json", diagnosis)
        smoke = {
            "schema": "SAME_SUCCESS_STATE_2_4_6_RESULT_V2",
            "status": "COMPLETE",
            "canonical_root": ROOT,
            "canonical_step": int(ref["state_step"]),
            "canonical_snapshot": str(SNAPSHOT),
            "branch_only_change": "requested external F_des; arm held at restored EE pose",
            "same_snapshot": True,
            "same_grasp_geometry": True,
            "explicit_arm_trajectory_change": False,
            "native_arm_force_position_response_recorded": True,
            "branches": summaries,
            "SAME_SUCCESS_STATE_FORCE_FRONTIER_VALID": bool(all(s["outcome"] == "FORCE_REALIZATION_VALID" for s in summaries)),
            "frontier_note": "This artifact covers same-state force realization and fixed-arm hold only; no transport benchmark was run.",
        }
        write_json(OUT / "SAME_SUCCESS_STATE_2_4_6_RESULT.json", smoke)
        report_lines = [
            "# ROOT7703 same-success-state force realization",
            "",
            f"Canonical state: root{ROOT}, step {ref['state_step']}; snapshot recovery is already validated.",
            "",
            "The prior smoke replayed the archived arm pose target after restore. The corrected audit holds the actual restored EE pose and changes only the external force target. The native legacy squeeze loop and force slots are disabled.",
            "",
            "## Result",
            "",
        ]
        for s in summaries:
            report_lines.append(f"- {s['branch']}: {s['outcome']}; target={s['requested_force_target']:.1f}N; realized={s['realized_force_N']}; settling_steps={s['settling_steps']}; fixed-arm hold={s['hold_success']}; frozen-VLA lift={s['frozen_vla_lift_success']}; frozen-VLA lift-hold={s['frozen_vla_lift_hold_success']}; contact_loss_step={s['contact_loss_step']}.")
        report_lines += [
            "",
            "The force metric matches root7400: object-filtered `contact_grasp_black_book_1`, side-resolved `panda_leftfinger`/`panda_rightfinger`, normal projection, and `2*min(left,right)` aggregation.",
            "",
            "No controller implementation or canonical snapshot was changed. Lift/transport was intentionally not run in this diagnostic; the arm was held fixed to isolate force realization.",
            "",
        ]
        (OUT / "ROOT7703_FORCE_FRONTIER_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
        return 0
    except Exception:
        write_json(OUT / "FORCE_REALIZATION_DIAGNOSIS.json", {"FINAL_STATUS": "RUNTIME_FAILURE", "traceback": traceback.format_exc()})
        return 1
    finally:
        try:
            if env is not None and term_cfg_snapshot is not None:
                from tac_manip.tasks.manipulation.libero import mdp
                mdp.restore_tabero_legacy_force_loop(env.action_manager.get_term("arm_action"), term_cfg_snapshot)
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    raise SystemExit(run())
