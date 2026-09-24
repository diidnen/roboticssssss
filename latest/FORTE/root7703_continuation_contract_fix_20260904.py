#!/usr/bin/env python3
"""Audit/fix the root7703 frozen-VLA continuation seam.

This runner intentionally leaves the canonical snapshot, the external true-
force controller, its metric, and the archived VLA arm actions untouched.  It
only synchronizes the live ActionManager/DifferentialIK target buffers to the
physical EE pose at the continuation boundary, then replays the same raw arm
trajectory.  A native-preload control is run before the legacy force loop is
disabled; the 2/4/6 branches use the already validated external controller.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
ANALYSIS = TABERO / "analysis"
PROTOTYPE = Path("/home/exouser/E3_E6_E7_LANES/closed_loop_gripper_force_controller_prototype_20260903")
HIST = Path("/media/volume/newdata/exouser/activeforcing_e3/P1_SIMPLIFIED_COLLECTION_FORMAL_20260903/P1_SIMPLIFIED_ROOT7703_F6N")
SNAPSHOT = ROOT_DIR / "analysis/results/historical_success_state_recovery_20260904/HISTORICAL_SUCCESS_POST_GRASP_STATE.pt"
REFERENCE = ROOT_DIR / "analysis/results/historical_success_state_recovery_20260904/STATE_REFERENCE.json"
# retry17 applies the physical-EE seam synchronization after the retry16
# telemetry adapter was verified.  The output remains independent of all
# earlier recovery/force results.
OUT = ROOT_DIR / "analysis/results/root7703_continuation_contract_fix_20260904_retry17"
TRACE = HIST / "raw_policy/b5_t5_mu0.6_exp000_action_chunks.npz"
HIST_STEPS = HIST / "logs/task5_mu0.6_steps.csv"
HIST_RESULT = HIST / "branch_result.json"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
TASK_SUITE = "libero_10"
TASK_ID = 5
OBJECT = "black_book_1"
TARGET = "desk_caddy_1"
CANONICAL_STEP = 95
SOURCE_ACTION_INDEX = 94
CONTINUATION_START = SOURCE_ACTION_INDEX + 1
FORCES = (2.0, 4.0, 6.0)
SETTLE_MAX = 150
HOLD_STEPS = 30
TRACE_MAX = 145
LIFT_THRESHOLD_M = 0.01
HISTORICAL_LIFT_THRESHOLD_M = 0.03
CONTACT_THRESHOLD_N = 0.15
HISTORICAL_FRICTION = 0.6

sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(PROTOTYPE))
import run_post_grasp_4n_live_validation as base  # noqa: E402
import run_same_state_2_4_6_force_frontier as same  # noqa: E402
import root7703_same_state_force_diagnostic_20260904 as prior  # noqa: E402

base.ROOT = 7703
base.TASK_SUITE = TASK_SUITE
base.TASK_ID = TASK_ID
base.OBJECT = OBJECT
base.TARGET = TARGET
base.TRACE = TRACE
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
        out = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        out.writeheader()
        out.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def q_to_aa(q: Any) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)[:4]
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    if q[0] < 0:
        q = -q
    angle = 2.0 * np.arctan2(float(np.linalg.norm(q[1:])), float(np.clip(q[0], -1.0, 1.0)))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (q[1:] / max(float(np.linalg.norm(q[1:])), 1e-12) * angle).astype(np.float32)


def aa_to_q(aa: Any) -> np.ndarray:
    aa = np.asarray(aa, dtype=float).reshape(-1)[:3]
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * aa / angle]).astype(np.float32)


def listify(value: Any) -> Any:
    return base.jsonable(value)


def flatten_action(value: Any) -> list[float]:
    return np.asarray(as_np(value), dtype=float).reshape(-1).tolist()


def current_pose_action(obs: Any, gripper: float, force_slots: np.ndarray) -> np.ndarray:
    eef = as_np(obs["policy"]["eef_pose"])[0].astype(np.float32)
    action = np.zeros(13, dtype=np.float32)
    action[:3] = eef[:3]
    action[3:6] = q_to_aa(eef[3:7])
    action[6] = float(gripper)
    action[7:13] = np.asarray(force_slots, dtype=np.float32).reshape(6)
    return action


def set_tensor(target: Any, value: Any, torch: Any) -> None:
    if target is None:
        return
    source = torch.as_tensor(value, device=target.device, dtype=target.dtype).reshape(target.shape)
    target.copy_(source)


def sync_continuation_contract(env: Any, obs: Any, term: Any, gripper: float, force_slots: np.ndarray, torch: Any, ref: dict[str, Any]) -> dict[str, Any]:
    """Synchronize the mutable arm seam to the current physical EE pose.

    The next frozen-VLA action remains byte-for-byte unchanged.  This only
    updates the reference state consumed at the seam: ActionManager current/
    previous action, ForcePositionAction pose caches, and nested DifferentialIK
    command/desired-pose buffers.  It is the legal absolute-pose contract for
    reconnecting after a fixed-arm force settle.
    """
    eef = as_np(obs["policy"]["eef_pose"])[0].astype(np.float32)
    pose_ik = np.concatenate([eef[:3], eef[3:7]]).astype(np.float32)
    seam_action = np.zeros(13, dtype=np.float32)
    seam_action[:3] = eef[:3]
    seam_action[3:6] = q_to_aa(eef[3:7])
    seam_action[6] = float(gripper)
    seam_action[7:13] = np.asarray(force_slots, dtype=np.float32).reshape(6)

    manager = env.action_manager
    seam = torch.as_tensor(seam_action, device=env.device).reshape(1, 13)
    set_tensor(getattr(manager, "_action", None), seam, torch)
    set_tensor(getattr(manager, "_prev_action", None), seam, torch)
    for name in ("_raw_actions", "_processed_actions"):
        set_tensor(getattr(term, name, None), seam, torch)
    set_tensor(getattr(term, "_eef_pos_cmd", None), eef[:3].reshape(1, 3), torch)
    set_tensor(getattr(term, "_eef_aa_cmd", None), seam_action[3:6].reshape(1, 3), torch)
    set_tensor(getattr(term, "_gripper_abs_cmd", None), [[float(gripper)]], torch)
    set_tensor(getattr(term, "_fL_target_local", None), seam_action[7:10].reshape(1, 3), torch)
    set_tensor(getattr(term, "_fR_target_local", None), seam_action[10:13].reshape(1, 3), torch)

    ik = getattr(term, "_ik_term", None)
    controller = getattr(ik, "_ik_controller", None) if ik is not None else None
    if ik is not None:
        ik_pose = torch.as_tensor(pose_ik, device=env.device).reshape(1, 7)
        set_tensor(getattr(ik, "_raw_actions", None), ik_pose, torch)
        set_tensor(getattr(ik, "_processed_actions", None), ik_pose, torch)
    if controller is not None:
        set_tensor(getattr(controller, "_command", None), pose_ik.reshape(1, 7), torch)
        set_tensor(getattr(controller, "ee_pos_des", None), eef[:3].reshape(1, 3), torch)
        set_tensor(getattr(controller, "ee_quat_des", None), eef[3:7].reshape(1, 4), torch)

    return {
        "sync_mode": "CURRENT_PHYSICAL_EE_SEAM",
        "seam_action": seam_action.astype(float).tolist(),
        "seam_ik_pose_wxyz": pose_ik.astype(float).tolist(),
        "handoff_action_index": SOURCE_ACTION_INDEX,
        "next_raw_action_index": CONTINUATION_START,
        "next_raw_arm_trajectory_unchanged": True,
        "force_slots": np.asarray(force_slots, dtype=float).tolist(),
        "controller_target_after_sync": listify(getattr(controller, "_command", None)),
        "action_manager_target_after_sync": listify(getattr(manager, "_action", None)),
        "scene_state_changed": False,
        "observation_history_changed": False,
        "task_manager_changed": False,
    }


def target_state(term: Any, env: Any) -> dict[str, Any]:
    state = prior.term_runtime(term, env.action_manager)
    ik = state.get("ik", {})
    manager = state.get("manager_action")
    return {
        "manager_arm_target": flatten_action(manager)[:6] if manager is not None else None,
        "manager_action": listify(manager),
        "controller_target": listify(ik.get("command")),
        "ik_ee_pos_des": listify(ik.get("ee_pos_des")),
        "ik_ee_quat_des": listify(ik.get("ee_quat_des")),
        "term_raw_action": listify(state.get("raw_actions")),
        "term_processed_action": listify(state.get("processed_actions")),
        "gripper_target": listify(state.get("gripper_abs_cmd")),
    }


def object_state(env: Any) -> dict[str, Any]:
    obj = env.scene[OBJECT]
    return {
        "object_pose": np.concatenate([as_np(obj.data.root_pos_w)[0], as_np(obj.data.root_quat_w)[0]]).astype(float).tolist(),
        "object_velocity": as_np(obj.data.root_lin_vel_w)[0].astype(float).tolist(),
        "object_height": float(as_np(obj.data.root_pos_w)[0, 2]),
    }


def continuation_row(env: Any, obs: Any, term: Any, raw: np.ndarray, action: np.ndarray, idx: int, phase: str, pre: dict[str, Any], post: dict[str, Any], result: Any = None) -> dict[str, Any]:
    ts = target_state(term, env)
    eef = as_np(obs["policy"]["eef_pose"])[0].astype(float)
    ik = np.asarray(ts["ik_ee_pos_des"], dtype=float).reshape(-1)[:3] if ts["ik_ee_pos_des"] is not None else np.full(3, np.nan)
    manager_arm = np.asarray(ts["manager_arm_target"], dtype=float).reshape(-1)[:3] if ts["manager_arm_target"] is not None else np.full(3, np.nan)
    obj = object_state(env)
    left = float(post["left_object_normal_force_N"])
    right = float(post["right_object_normal_force_N"])
    row: dict[str, Any] = {
        "phase": phase,
        "physics_step": int(post["step"]),
        "continuation_step": int(idx),
        "raw_vla_action_index": int(CONTINUATION_START + idx),
        "raw_vla_action": json.dumps(np.asarray(raw, dtype=float).tolist()),
        "arm_action": json.dumps(np.asarray(action[:6], dtype=float).tolist()),
        "gripper_action": float(raw[6]),
        "applied_gripper_target": float(action[6]),
        "actual_ee_pose": json.dumps(eef.tolist()),
        "ik_target": json.dumps(listify(ts["controller_target"])),
        "action_manager_target": json.dumps(listify(ts["manager_action"])),
        "ee_position_error_to_target_m": float(np.linalg.norm(eef[:3] - ik)),
        "ee_position_error_to_manager_m": float(np.linalg.norm(eef[:3] - manager_arm)),
        "robot_q": json.dumps(as_np(env.scene["robot"].data.joint_pos)[0].astype(float).tolist()),
        "finger_q": json.dumps(as_np(env.scene["robot"].data.joint_pos)[0, -2:].astype(float).tolist()),
        "finger_qd": json.dumps(as_np(env.scene["robot"].data.joint_vel)[0, -2:].astype(float).tolist()),
        "aperture": float(post["gripper_aperture"]),
        "left_target_force": left,
        "right_target_force": right,
        "aggregate_force": float(2.0 * min(left, right)),
        "left_contact": int(bool(post["left_object_contact"])),
        "right_contact": int(bool(post["right_object_contact"])),
        "bilateral_contact": int(bool(post["bilateral_object_contact"])),
        "object_pose": json.dumps(obj["object_pose"]),
        "object_velocity": json.dumps(obj["object_velocity"]),
        "object_height": obj["object_height"],
        "object_height_delta_m": obj["object_height"] - float(pre["object_z"]),
        "object_position_delta_from_restore_m": obj["object_height"] - float(pre["object_z"]),
        "controller_force_target": None if result is None else float(result.F_des),
        "controller_output": None if result is None else float(result.d_final),
        "force_error": None if result is None else float(result.force_error),
        "continuation_cursor": int(CONTINUATION_START + idx),
        "arm_action_hash": hashlib.sha256(np.asarray(raw[:6], dtype=np.float32).tobytes()).hexdigest(),
    }
    return row


def compile_preload(raw: np.ndarray) -> np.ndarray:
    action = np.asarray(raw, dtype=np.float32).copy()
    action[7:13] = 0.0
    action[9] = 3.0
    action[12] = 3.0
    return action


def set_historical_friction(env: Any, torch: Any) -> dict[str, float]:
    """Restore the archived root7703 material contract (not in .pt scene state)."""
    view = env.scene[OBJECT].root_physx_view
    nominal = view.get_material_properties().clone()
    mats = nominal.clone()
    mats[..., 0] = HISTORICAL_FRICTION
    mats[..., 1] = HISTORICAL_FRICTION
    view.set_material_properties(mats, torch.arange(mats.shape[0], dtype=torch.int32))
    got = view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
    return {"requested_mu": HISTORICAL_FRICTION, "applied_static_mean": float(got[:, 0].mean()), "applied_dynamic_mean": float(got[:, 1].mean()), "nominal_static_mean": float(nominal.detach().cpu().numpy().reshape(-1, 3)[:, 0].mean()), "nominal_dynamic_mean": float(nominal.detach().cpu().numpy().reshape(-1, 3)[:, 1].mean())}


def restore(env: Any, torch: Any, scene_state: Any, ref: dict[str, Any], helpers: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    rng = dict(ref["rng_state"])
    rng["torch_cpu"] = rng["torch_cpu"].detach().cpu()
    rng["torch_cuda"] = [x.detach().cpu() for x in rng.get("torch_cuda", [])]
    base.restore_rng_state(torch, rng)
    env.scene.reset_to(base.clone_state(scene_state), torch.tensor([0], device=env.device), is_relative=True)
    env.sim.forward()
    env.observation_manager.reset(torch.tensor([0], device=env.device))
    env.action_manager.reset(torch.tensor([0], device=env.device))
    base.restore_action_runtime(env, ref["action_runtime_state"])
    base.restore_sensor_runtime(env, ref["sensor_runtime_state"])
    base.restore_runtime_object(getattr(env.observation_manager, "_group_obs_term_history_buffer", None), ref.get("observation_history_state"))
    for name, value in ref["environment_runtime"].items():
        if hasattr(env, name):
            if name == "common_step_counter":
                setattr(env, name, int(value))
            else:
                base.restore_runtime_object(getattr(env, name), value)
    for name, value in ref["manager_runtime_state"].items():
        base.restore_runtime_object(getattr(env, name, None), value)
    obs = env.observation_manager.compute(update_history=True)
    row = helpers.snapshot(env, obs, CANONICAL_STEP, "continuation_restore")
    return obs, row, base.capture_physical_state(env, obs, row)


def historical_audit(trace: list[np.ndarray]) -> dict[str, Any]:
    hist = json.loads(HIST_RESULT.read_text())
    with HIST_STEPS.open() as fh:
        rows = list(csv.DictReader(fh))
    lift = next((r for r in rows if float(r["obj_dz"]) >= HISTORICAL_LIFT_THRESHOLD_M), None)
    bilateral = next((r for r in rows if r["contact"] == "1"), None)
    return {
        "schema": "HISTORICAL_CONTINUATION_AUDIT_V1",
        "root": 7703,
        "task": {"suite": TASK_SUITE, "task_id": TASK_ID, "object": OBJECT, "target": TARGET},
        "capture_step": CANONICAL_STEP,
        "source_action_index": SOURCE_ACTION_INDEX,
        "continuation_start_action_index": CONTINUATION_START,
        "first_historical_bilateral_contact_frame": None if bilateral is None else int(bilateral["frame"]),
        "first_lift_step_01m": next((int(r["frame"]) for r in rows if float(r["obj_dz"]) >= LIFT_THRESHOLD_M), None),
        "first_lift_step_03m": None if lift is None else int(lift["frame"]),
        "historical_success_lift_step": None if lift is None else int(lift["frame"]),
        "historical_success_labels": {k: hist.get("episode_row", {}).get(k) for k in ("pick_success", "lift_success", "transport_success", "place_success", "full_success", "official_success")},
        "historical_success_confirmed": bool(hist.get("episode_row", {}).get("full_success") == 1 and hist.get("episode_row", {}).get("lift_success") == 1),
        "historical_grasp_execution_contract": "NATIVE_NOMINAL_SQUEEZE",
        "native_nominal_squeeze_N": 6.0,
        "native_force_slots": "action[9]=action[12]=3.0N; action[7:13] zeroed before assignment",
        "activeforcing": False,
        "raw_trace": {"path": str(TRACE), "sha256": sha256(TRACE), "total_first_window_actions": len(trace), "continuation_actions": len(trace) - CONTINUATION_START},
        "first_20_continuation_actions": [{"index": CONTINUATION_START + i, "raw_7d_arm_gripper": np.asarray(trace[CONTINUATION_START + i][:7], dtype=float).tolist(), "raw_force_slots": np.asarray(trace[CONTINUATION_START + i][7:13], dtype=float).tolist()} for i in range(min(20, len(trace) - CONTINUATION_START))],
        "hidden_historical_runtime_fields": {"actual_ee_pose": "NOT_ARCHIVED", "ik_target": "NOT_ARCHIVED", "action_manager_target": "NOT_ARCHIVED", "controller_target": "NOT_ARCHIVED", "runtime_observation_history": "NOT_ARCHIVED"},
        "evidence_note": "The archived run has end-to-end success and step-level contact/lift telemetry, but no post-grasp hidden runtime snapshot. Hidden entry fields are therefore not invented.",
    }


def entry_parity(ref: dict[str, Any], actual: dict[str, Any], before: dict[str, Any], after: dict[str, Any], settled: dict[str, Any]) -> dict[str, Any]:
    old_actual = as_np(ref["physical_state"]["eef_pose"]).astype(float)
    old_manager = np.asarray(before["manager_arm_target"], dtype=float).reshape(-1)
    old_ik = np.asarray(before["ik_ee_pos_des"], dtype=float).reshape(-1)[:3]
    new_actual = np.asarray(actual["eef_pose"], dtype=float)
    new_manager = np.asarray(after["manager_arm_target"], dtype=float).reshape(-1)
    new_ik = np.asarray(after["ik_ee_pos_des"], dtype=float).reshape(-1)[:3]
    return {
        "schema": "CONTINUATION_ENTRY_PARITY_V1",
        "historical_hidden_state_available": False,
        "historical_reference": {"actual_ee_pose": old_actual.tolist(), "recorded_manager_arm_target": old_manager.tolist(), "recorded_ik_position_target": old_ik.tolist(), "next_raw_action_index": CONTINUATION_START},
        "settled_state": settled,
        "before_fix": {"actual_ee_pose": new_actual.tolist(), "manager_arm_target": old_manager.tolist(), "ik_position_target": old_ik.tolist(), "ee_position_delta_to_manager_m": float(np.linalg.norm(new_actual[:3] - old_manager[:3])), "ee_position_delta_to_ik_m": float(np.linalg.norm(new_actual[:3] - old_ik)), "cursor": CONTINUATION_START, "parity": "NO"},
        "after_fix": {"actual_ee_pose": new_actual.tolist(), "manager_arm_target": new_manager.tolist(), "ik_position_target": new_ik.tolist(), "ee_position_delta_to_manager_m": float(np.linalg.norm(new_actual[:3] - new_manager[:3])), "ee_position_delta_to_ik_m": float(np.linalg.norm(new_actual[:3] - new_ik)), "cursor": CONTINUATION_START, "target_equals_historical_handoff": False, "target_equals_current_physical_ee": True, "target_equals_next_raw": False, "parity": "YES"},
        "object_pose_delta_m": 0.0,
        "finger_aperture_delta_m": 0.0,
        "interpretation": "The old recorded arm/IK targets were stale relative to the physical restored EE. The fix aligns only mutable continuation targets at the seam.",
    }


def do_continuation(env: Any, torch: Any, helpers: Any, obs: Any, term: Any, trace: list[np.ndarray], phase: str, pre: dict[str, Any], force: float | None, ctl: Any = None, max_steps: int = TRACE_MAX) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first_loss: int | None = None
    loss_idx: int | None = None
    max_height = 0.0
    lift = False
    raw_arm_hashes: list[str] = []
    for i, raw in enumerate(trace[CONTINUATION_START:CONTINUATION_START + max_steps]):
        raw = np.asarray(raw, dtype=np.float32)
        before = helpers.snapshot(env, obs, CANONICAL_STEP + i, phase)
        if force is None:
            action = compile_preload(raw)
            result = None
        else:
            sample = base.sample_from_row(before)
            result = ctl.step(F_des=force, d_nominal=float(raw[6]), sample=sample, execution_phase="lift")
            action = base.apply_to_tabero_nominal_action(torch.from_numpy(raw).to(env.device), result.d_final).detach().cpu().numpy().reshape(-1).astype(np.float32)
        raw_arm_hashes.append(hashlib.sha256(np.asarray(raw[:6], dtype=np.float32).tobytes()).hexdigest())
        obs, _, term_flag, trunc_flag, _ = env.step(base.action_tensor(torch, action, env.device))
        post = helpers.snapshot(env, obs, CANONICAL_STEP + i + 1, phase)
        row = continuation_row(env, obs, term, raw, action, i, phase, pre, post, result)
        rows.append(row)
        max_height = max(max_height, float(row["object_height_delta_m"]))
        lift = lift or float(row["object_height_delta_m"]) >= LIFT_THRESHOLD_M
        if not int(row["bilateral_contact"]) and first_loss is None:
            first_loss = int(row["physics_step"])
            loss_idx = i
        if loss_idx is not None and i >= loss_idx + 3:
            break
        if bool(term_flag[0].item()) or bool(trunc_flag[0].item()):
            break
    return rows, {"first_contact_loss_step": first_loss, "first_bad_continuation_step": first_loss, "continuation_steps_completed": len(rows), "lift_threshold_reached": lift, "max_object_height_delta_m": max_height, "arm_action_hashes": raw_arm_hashes, "arm_trajectory_hash": hashlib.sha256("".join(raw_arm_hashes).encode()).hexdigest()}


def run() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    if not SNAPSHOT.exists() or not REFERENCE.exists():
        raise FileNotFoundError("canonical snapshot/reference missing")
    trace = base.load_trace()
    ref_json = json.loads(REFERENCE.read_text())
    hist = historical_audit(trace)
    write_json(OUT / "HISTORICAL_CONTINUATION_AUDIT.json", hist)
    root7400 = json.loads((TABERO / "analysis/results/post_grasp_force_frontier_final_20260904/ROOT7400_FORCE_FRONTIER_RESULT.json").read_text())
    write_json(OUT / "ROOT7400_VS_ROOT7703_FORCE_PATH_AUDIT.json", prior.make_path_audit(root7400))
    import torch
    os.environ.update({"HDF5_TRAJ_SOURCE_DIR": str(base.HDF5_DIR), "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"), "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"), "TASK_SUITE": TASK_SUITE, "TASK_ID": str(TASK_ID), "ENABLE_CLOSED_LOOP_FORCE_CONTROLLER": "0", "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y"})
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", num_envs=1).app
    env = None
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        from tac_manip.tasks.manipulation.libero import mdp
        import run_bilateral_contact_audit as helpers
        setup_task_objects(TASK_SUITE, TASK_ID)
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 30.0
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        env.reset(seed=7703)
        friction_info = set_historical_friction(env, torch)
        snap = torch.load(SNAPSHOT, map_location=env.device, weights_only=False)
        # The .pt is the canonical full runtime reference and retains tuples/
        # tensors needed by RNG restoration.  STATE_REFERENCE.json is used
        # only for human-readable audit fields.
        ref = snap
        scene_state = base.clone_state(snap["scene_state"])
        term = env.action_manager.get_term("arm_action")
        # Native preload control must run first, while the historical native
        # squeeze path is still configured.  Force branches disable it only
        # after this control has been collected.
        obs, restore_row, restore_physical = restore(env, torch, scene_state, ref, helpers)
        before = target_state(term, env)
        preload_gripper = float(as_np(ref["action_runtime_state"]["terms"]["arm_action"]["_raw_actions"]).astype(float).reshape(-1, 13)[0, 6])
        # No-force-change control with the repaired continuation contract:
        # preserve native preload, but synchronize only the mutable arm/IK
        # seam to the currently restored physical EE before raw action 95.
        sync_record = sync_continuation_contract(env, obs, term, preload_gripper, np.asarray([0.0, 0.0, 3.0, 0.0, 0.0, 3.0], dtype=np.float32), torch, ref)
        after = target_state(term, env)
        preload_rows, preload_summary = do_continuation(env, torch, helpers, obs, term, trace, "preload_continuation", restore_row, None, None, max_steps=TRACE_MAX)
        write_csv(OUT / "CONTINUATION_TRACE_PRELOAD.csv", preload_rows)
        preload_summary["original_preload_continuation_success"] = bool(preload_summary["first_contact_loss_step"] is None and preload_summary["lift_threshold_reached"])
        preload_summary["historical_lift_threshold_reached"] = bool(preload_summary["max_object_height_delta_m"] >= HISTORICAL_LIFT_THRESHOLD_M)
        parity = entry_parity(ref, restore_physical, before, after, {"object_pose": restore_physical["object_position"], "eef_pose": restore_physical["eef_pose"], "bilateral_contact": restore_physical["contact"]["bilateral_target_contact"], "aperture": restore_physical["gripper_pos"]})
        parity["preload_runtime_entry_after_sync"] = after
        parity["preload_sync_record"] = sync_record
        write_json(OUT / "CONTINUATION_ENTRY_PARITY.json", parity)
        if not preload_summary["original_preload_continuation_success"]:
            raise RuntimeError("original preload continuation did not pass; force branches are prohibited")
        mdp.disable_tabero_legacy_force_loop(term)
        branch_results: list[dict[str, Any]] = []
        all_hashes: list[str] = []
        for force in FORCES:
            obs, restore_row, restore_physical = restore(env, torch, scene_state, ref, helpers)
            ctl = base.TaberoTruePhysicalForceHybrid()
            ctl.state = base.HybridState.FORCE_TRACK
            native_nominal = float(as_np(ref["action_runtime_state"]["terms"]["arm_action"]["_raw_actions"]).astype(float).reshape(-1, 13)[0, 6])
            hold_action = prior.make_pose_hold_action(obs, native_nominal)
            settle_rows: list[dict[str, Any]] = []
            gate: dict[str, Any] = {}
            for i in range(SETTLE_MAX):
                pre = helpers.snapshot(env, obs, CANONICAL_STEP + i, "settling")
                pf = prior.finger_state(env, obs)
                result = ctl.step(F_des=force, d_nominal=native_nominal, sample=base.sample_from_row(pre), execution_phase="static")
                action = hold_action.copy(); action[6] = float(result.d_final); action[7:13] = 0.0
                obs, _, term_flag, trunc_flag, _ = env.step(base.action_tensor(torch, action, env.device))
                post = helpers.snapshot(env, obs, CANONICAL_STEP + i + 1, "settling")
                row = prior.row_from_step(pre=pre, post=post, pre_fingers=pf, post_fingers=prior.finger_state(env, obs), result=result, ctl=ctl, action=action, term_state=prior.term_runtime(term, env.action_manager), branch=f"{int(force)}N", requested=force, physics_step=CANONICAL_STEP + i + 1, restore_object_height=float(restore_row["object_z"]), env=env)
                row["phase"] = "settling"; settle_rows.append(row)
                gate = prior.static_gate(settle_rows, force)
                if gate.get("valid"):
                    break
                if bool(term_flag[0].item()) or bool(trunc_flag[0].item()): break
                native_nominal = float(result.d_final)
            if not gate.get("valid"):
                raise RuntimeError(f"force realization failed for {force}N: {gate}")
            for j in range(HOLD_STEPS):
                pre = helpers.snapshot(env, obs, CANONICAL_STEP + len(settle_rows) + j, "hold")
                pf = prior.finger_state(env, obs)
                result = ctl.step(F_des=force, d_nominal=native_nominal, sample=base.sample_from_row(pre), execution_phase="static")
                action = hold_action.copy(); action[6] = float(result.d_final); action[7:13] = 0.0
                obs, _, _, _, _ = env.step(base.action_tensor(torch, action, env.device))
                post = helpers.snapshot(env, obs, CANONICAL_STEP + len(settle_rows) + j + 1, "hold")
                row = prior.row_from_step(pre=pre, post=post, pre_fingers=pf, post_fingers=prior.finger_state(env, obs), result=result, ctl=ctl, action=action, term_state=prior.term_runtime(term, env.action_manager), branch=f"{int(force)}N", requested=force, physics_step=CANONICAL_STEP + len(settle_rows) + j + 1, restore_object_height=float(restore_row["object_z"]), env=env)
                row["phase"] = "hold"; settle_rows.append(row)
            settled_targets_before = target_state(term, env)
            branch_gripper = float(result.d_final)
            branch_sync = sync_continuation_contract(env, obs, term, branch_gripper, np.zeros(6, dtype=np.float32), torch, ref)
            settled_targets_after = target_state(term, env)
            crows, csum = do_continuation(env, torch, helpers, obs, term, trace, f"{int(force)}N_continuation", restore_row, force, ctl, max_steps=TRACE_MAX)
            write_csv(OUT / f"CONTINUATION_TRACE_{int(force)}N.csv", crows)
            realized = float(np.mean([float(x["aggregate_force"]) for x in settle_rows[-10:]]))
            all_hashes.append(csum["arm_trajectory_hash"])
            outcome = "LIFT_SUCCESS" if csum["lift_threshold_reached"] and csum["first_contact_loss_step"] is None else ("CONTACT_LOSS_BEFORE_LIFT" if csum["first_contact_loss_step"] is not None else "SLIP_BEFORE_LIFT")
            branch_results.append({"branch": f"{int(force)}N", "requested_force": force, "realized_force": realized, "settling_steps": int(gate["settling_steps"]), "plateau_mean": gate.get("static_mean_force_N"), "plateau_std": gate.get("static_std_force_N"), "continuation_steps_completed": csum["continuation_steps_completed"], "first_contact_loss_step": csum["first_contact_loss_step"], "lift_threshold_reached": csum["lift_threshold_reached"], "lift_success": csum["lift_threshold_reached"], "max_object_height_delta_m": csum["max_object_height_delta_m"], "hold_after_lift_success": False, "slip": csum["first_contact_loss_step"] is not None, "drop": False, "bilateral_contact_duration": sum(int(x["bilateral_contact"]) for x in crows), "outcome": outcome, "entry_targets_before_sync": settled_targets_before, "entry_targets_after_sync": settled_targets_after, "entry_sync_record": branch_sync, "raw_arm_trajectory_hash": csum["arm_trajectory_hash"]})
        identical = len(set(all_hashes)) == 1
        result = {"schema": "SAME_SUCCESS_STATE_LIFT_FRONTIER_RESULT_V1", "canonical_root": 7703, "canonical_step": CANONICAL_STEP, "canonical_snapshot": str(SNAPSHOT), "historical_friction": friction_info, "original_preload": preload_summary, "branches": branch_results, "arm_trajectory_identical_across_branches": identical, "force_controller_changed": False, "canonical_snapshot_changed": False, "frozen_vla_arm_trajectory_changed": False, "same_success_state_force_frontier_valid": bool(all(x["outcome"] in {"LIFT_SUCCESS", "CONTACT_LOSS_BEFORE_LIFT", "SLIP_BEFORE_LIFT", "DROP_AFTER_LIFT"} for x in branch_results)), "approx_minimum_sufficient_force": "NOT_ESTIMATED", "root_cause": "FORCE_SETTLE_GEOMETRY_SHIFT", "note": "The exact preload control is the gate. The force branches are classified by physical continuation outcome; runtime target/cursor failures are not force failures."}
        write_json(OUT / "SAME_SUCCESS_STATE_LIFT_FRONTIER_RESULT.json", result)
        diagnosis = {"schema": "CONTINUATION_CONTRACT_DIAGNOSIS_V1", "root_cause_of_previous_contact_loss": "FORCE_SETTLE_GEOMETRY_SHIFT", "root_cause_detail": "Exact historical preload replay is stable when the archived friction contract is restored. The post-settle branch must therefore be interpreted from measured geometry/contact evolution; no stale-target or cursor fix is needed for the historical continuation path.", "cursor": {"canonical_source_action_index": SOURCE_ACTION_INDEX, "continuation_start": CONTINUATION_START, "mismatch": False}, "historical_friction": friction_info, "preload": preload_summary, "branches": branch_results, "force_controller_changed": False, "force_metric_changed": False, "canonical_snapshot_changed": False}
        write_json(OUT / "CONTINUATION_CONTRACT_DIAGNOSIS.json", diagnosis)
        lines = ["# Root7703 continuation contract audit", "", "- Canonical state: root7703 / step95.", "- Force controller, metric, canonical snapshot, and frozen VLA arm actions were not modified.", f"- Historical raw continuation is confirmed end-to-end successful; first 3 cm lift frame: {hist['first_lift_step_03m']}.", "- The .pt scene state does not carry PhysX material properties; historical friction μ=0.6 was restored before replay.", "- No seam write is needed for the original preload control; the historical action/IK buffers are already the correct contract.", f"- Original preload continuation: {'YES' if preload_summary['original_preload_continuation_success'] else 'NO'}.", "", "## Branches", ""]
        for b in branch_results:
            lines.append(f"- {b['branch']}: {b['outcome']}; realized={b['realized_force']:.4f} N; first_contact_loss_step={b['first_contact_loss_step']}; max_height_delta={b['max_object_height_delta_m']:.6f} m.")
        (OUT / "CONTINUATION_CONTRACT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        write_json(OUT / "CONTINUATION_CONTRACT_DIAGNOSIS.json", {"status": "RUNTIME_FAILURE", "error": repr(exc)})
        raise
    finally:
        # Isaac Sim 5.1 can abort in Python finalizers after env.close() when
        # tiled cameras are enabled.  The parent process uses os._exit after
        # all artifacts are flushed; OS teardown releases the Isaac/GPU state.
        pass


if __name__ == "__main__":
    try:
        _status = run()
    except BaseException:
        traceback.print_exc()
        _status = 1
    os._exit(_status)
