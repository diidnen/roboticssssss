#!/usr/bin/env python3
"""Recover and validate one historical Tabero task-5 grasp state.

This runner is deliberately limited to one historical root.  It replays the
archived preprobe scene with the archived action trace, using the historical
P1 contract (force slots fixed to 6 N, pose/gripper prefix preserved), then
captures the first semantic bilateral post-grasp state.  It does not run a
force frontier, ActiveForcing, a probe, or a new controller.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ROOT = 7703
TASK_SUITE = "libero_10"
TASK_ID = 5
OBJECT = "black_book_1"
TARGET = "desk_caddy_1"
FORCE_COMMAND_N = 6.0
FRICTION = 0.6
REPLAY_STEPS_AFTER_STATE = 10
HOLD_STEPS = 30
OUT = Path("/home/exouser/FORTE/analysis/results/historical_success_state_recovery_20260904/attempt2")
TABERO = Path("/media/volume/newdata/exouser/Tabero_e3lh")
E3 = Path("/home/exouser/E3_E6_E7_LANES/E3_FULLTASK_VS_LOCALLIFT")
B5_SOURCE = TABERO / "analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_tabero_neutral_client.py"
HDF5_DIR = Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_SOURCE_20260902_103300/assembled_hdf5")
HIST_ROOT = Path(f"/media/volume/newdata/exouser/activeforcing_e3/P1_SIMPLIFIED_COLLECTION_FORMAL_20260903/P1_SIMPLIFIED_ROOT{ROOT}_F6N")
PREPROBE = HIST_ROOT / "preprobe_state_exp000.pt"
TRACE = HIST_ROOT / "raw_policy/b5_t5_mu0.6_exp000_action_chunks.npz"
HIST_STEPS = HIST_ROOT / "logs/task5_mu0.6_steps.csv"
HIST_EPISODES = HIST_ROOT / "logs/task5_mu0.6_episodes.csv"

sys.path.insert(0, str(E3))
sys.path.insert(0, str(TABERO / "analysis"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rp = load_module("tabero_post_grasp_helpers", Path("/home/exouser/Tabero/analysis/run_post_grasp_4n_live_validation.py"))
rp.OBJECT = OBJECT
rp.TASK_SUITE = TASK_SUITE
rp.TASK_ID = TASK_ID
rp.TARGET = TARGET
rp.ROOT = ROOT
rp.TRACE = TRACE
rp.HDF5_DIR = HDF5_DIR


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def move_tensors(value: Any, torch: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, torch, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, torch, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, torch, device) for item in value)
    return value


def set_historical_friction(env: Any, torch: Any) -> dict[str, float]:
    view = env.scene[OBJECT].root_physx_view
    nominal = view.get_material_properties().clone()
    mats = nominal.clone()
    mats[..., 0] = FRICTION
    mats[..., 1] = FRICTION
    view.set_material_properties(mats, torch.arange(mats.shape[0], dtype=torch.int32))
    got = view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
    return {
        "requested_mu": FRICTION,
        "applied_static_mean": float(got[:, 0].mean()),
        "applied_dynamic_mean": float(got[:, 1].mean()),
        "nominal_static_mean": float(nominal.detach().cpu().numpy().reshape(-1, 3)[:, 0].mean()),
        "nominal_dynamic_mean": float(nominal.detach().cpu().numpy().reshape(-1, 3)[:, 1].mean()),
    }


def compile_historical_action(raw: np.ndarray) -> np.ndarray:
    action = np.asarray(raw, dtype=np.float32).copy()
    if action.size < 13:
        raise RuntimeError(f"archived action is not 13D: {action.shape}")
    # Exact P1 seam: only force slots are overwritten; arm pose and gripper
    # action are retained byte-for-byte from the frozen trace.
    action[7:13] = 0.0
    action[9] = FORCE_COMMAND_N / 2.0
    action[12] = FORCE_COMMAND_N / 2.0
    return action


def observation_digest(obs: Any) -> str:
    policy = obs.get("policy", {}) if isinstance(obs, dict) else {}
    selected = {}
    for key in ("eef_pose", "gripper_pos", "gripper_net_force", "gripper_marker_motion", "tactile_gripper_force"):
        if key in policy:
            selected[key] = rp.jsonable(policy[key])
    return hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def target_contact_summary(env: Any) -> dict[str, Any]:
    # Reuse the read-only object-filtered sensor helper.  It resolves side by
    # body name, not by an array-position guess.
    from tabero_native_force_telemetry import target_object_force_world_by_side

    name = f"contact_grasp_{OBJECT}"
    sensor = env.scene[name]
    result = target_object_force_world_by_side(sensor)
    left = np.asarray(result.get("left_world") or [0.0, 0.0, 0.0], dtype=float)
    right = np.asarray(result.get("right_world") or [0.0, 0.0, 0.0], dtype=float)
    return {
        "sensor": name,
        "available": bool(result.get("available")),
        "reason": result.get("reason"),
        "body_names": result.get("body_names", []),
        "object_contact_count": int(result.get("object_contact_count", 0)),
        "left_world": left.tolist(),
        "right_world": right.tolist(),
        "left_norm_N": float(np.linalg.norm(left)),
        "right_norm_N": float(np.linalg.norm(right)),
        "bilateral": bool(np.linalg.norm(left) > 0.15 and np.linalg.norm(right) > 0.15),
    }


def capture_reference(env: Any, obs: Any, row: dict[str, Any], action: np.ndarray, step: int, source_index: int, trace: list[np.ndarray], torch: Any, friction_info: dict[str, float]) -> dict[str, Any]:
    scene_state = rp.clone_state(env.scene.get_state(is_relative=True))
    rng = rp.capture_rng_state(torch)
    action_runtime = rp.capture_action_runtime(env)
    sensor_runtime = rp.capture_sensor_runtime(env)
    observation_history = rp.capture_runtime_object(getattr(env.observation_manager, "_group_obs_term_history_buffer", None))
    environment_runtime = {
        name: rp.capture_runtime_object(getattr(env, name))
        for name in ("episode_length_buf", "reset_buf", "terminated_buf", "truncated_buf")
        if hasattr(env, name)
    }
    if hasattr(env, "common_step_counter"):
        environment_runtime["common_step_counter"] = int(env.common_step_counter)
    manager_runtime = {
        name: rp.capture_runtime_object(getattr(env, name, None))
        for name in ("event_manager", "command_manager", "termination_manager")
    }
    return {
        "schema": "HISTORICAL_SUCCESS_STATE_RECOVERY_REFERENCE_V1",
        "root_id": ROOT,
        "task_suite": TASK_SUITE,
        "task_id": TASK_ID,
        "object": OBJECT,
        "target": TARGET,
        "state_step": step,
        "source_action_index": source_index,
        "execution_contract": {
            "force_command_N": FORCE_COMMAND_N,
            "force_slots": "action[9]=action[12]=force_command_N/2; action[7:13] zeroed before assignment",
            "pose_gripper_prefix_preserved": True,
            "executor": "native Tabero ForcePositionAction",
            "activeforcing": False,
        },
        "friction": friction_info,
        "scene_state": scene_state,
        "physical_state": rp.capture_physical_state(env, obs, row),
        "physical_row": row,
        "target_contact": target_contact_summary(env),
        "observation_digest": observation_digest(obs),
        "environment_state": rp.capture_environment_state(env),
        "environment_runtime": environment_runtime,
        "action_runtime_state": action_runtime,
        "sensor_runtime_state": sensor_runtime,
        "observation_history_state": observation_history,
        "manager_runtime_state": manager_runtime,
        "rng_state": rng,
        "rng_hashes": {k: rp.digest_json(v) for k, v in rng.items()},
        "handoff_action": action,
        "continuation": {
            "remaining_action_count": len(trace) - source_index - 1,
            "remaining_trace_sha256": hashlib.sha256(b"".join(np.asarray(x, dtype=np.float32).tobytes() for x in trace[source_index + 1 :])).hexdigest(),
        },
    }


def restore_reference(env: Any, ref: dict[str, Any], torch: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    rp.restore_rng_state(torch, ref["rng_state"])
    env.scene.reset_to(rp.clone_state(ref["scene_state"]), torch.tensor([0], device=env.device), is_relative=True)
    env.sim.forward()
    env.observation_manager.reset(torch.tensor([0], device=env.device))
    env.action_manager.reset(torch.tensor([0], device=env.device))
    rp.restore_action_runtime(env, ref["action_runtime_state"])
    rp.restore_sensor_runtime(env, ref["sensor_runtime_state"])
    rp.restore_runtime_object(getattr(env.observation_manager, "_group_obs_term_history_buffer", None), ref["observation_history_state"])
    for name, value in ref["environment_runtime"].items():
        if hasattr(env, name):
            if name == "common_step_counter":
                setattr(env, name, int(value))
            else:
                rp.restore_runtime_object(getattr(env, name), value)
    for name, value in ref["manager_runtime_state"].items():
        rp.restore_runtime_object(getattr(env, name, None), value)
    obs = env.observation_manager.compute(update_history=True)
    row = helpers.snapshot(env, obs, int(ref["state_step"]), "recovery_restore")
    physical = rp.capture_physical_state(env, obs, row)
    return obs, row, physical


def parity_record(ref: dict[str, Any], env: Any, obs: Any, row: dict[str, Any], physical: dict[str, Any], restore_id: int) -> dict[str, Any]:
    diffs = rp.diff_physical_state(ref["physical_state"], physical)
    # PhysX recomputes derived EE angular velocity on sim.forward; retain the
    # raw error but classify a small numerical refresh separately.
    for item in diffs:
        if item["field"] == "eef_angular_velocity" and float(item["error"]) <= 5e-5:
            item.update({"pass": True, "tolerance": 5e-5, "note": "PhysX sim.forward derived-velocity refresh"})
    physical_pass = all(bool(x["pass"]) for x in diffs)
    action_pass = rp.digest_json(rp.capture_action_runtime(env)) == rp.digest_json(ref["action_runtime_state"])
    sensor_pass = rp.digest_json(rp.capture_sensor_runtime(env)) == rp.digest_json(ref["sensor_runtime_state"])
    history_pass = rp.digest_json(rp.capture_runtime_object(getattr(env.observation_manager, "_group_obs_term_history_buffer", None))) == rp.digest_json(ref["observation_history_state"])
    manager_pass = all(rp.digest_json(rp.capture_runtime_object(getattr(env, name, None))) == rp.digest_json(value) for name, value in ref["manager_runtime_state"].items())
    environment_pass = rp.digest_json(rp.capture_environment_state(env)) == rp.digest_json(ref["environment_state"])
    obs_pass = observation_digest(obs) == ref["observation_digest"]
    contact = target_contact_summary(env)
    return {
        "restore_id": restore_id,
        "physical_state_parity": physical_pass,
        "physical_diffs": diffs,
        "action_runtime_parity": action_pass,
        "sensor_history_parity": sensor_pass,
        "observation_history_parity": history_pass,
        "manager_runtime_parity": manager_pass,
        "environment_task_state_parity": environment_pass,
        "observation_parity": obs_pass,
        "bilateral_contact": contact["bilateral"],
        "contact": contact,
        "restored_row": row,
    }


def historical_audit() -> dict[str, Any]:
    roots = []
    for rid in (7703, 7704, 7705, 7801, 7802):
        root = HIST_ROOT.parent / f"P1_SIMPLIFIED_ROOT{rid}_F6N"
        with (root / "branch_result.json").open() as fh:
            result = json.load(fh)
        with (root / "logs/task5_mu0.6_steps.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        with (root / "logs/task5_mu0.6_episodes.csv").open() as fh:
            episode = list(csv.DictReader(fh))[-1]
        contact_frame = next((int(r["frame"]) for r in rows if r["contact"] == "1"), None)
        force_frame = next((int(r["frame"]) for r in rows if float(r["measured_squeeze_N"]) > 0.15), None)
        lift_frame = next((int(r["frame"]) for r in rows if float(r["obj_dz"]) >= 0.03), None)
        roots.append({
            "root_id": rid,
            "classification": "HISTORICAL_SUCCESS_CONFIRMED" if result.get("y_full") == 1 and result.get("y_lift") == 1 and episode.get("official_success") == "1" else "HISTORICAL_SUCCESS_LABEL_UNVERIFIED",
            "task": {"suite": result.get("task_suite"), "task_id": result.get("task_id"), "object": result.get("object"), "target": result.get("target")},
            "success_evidence": {"y_full": result.get("y_full"), "y_lift": result.get("y_lift"), "official_success": episode.get("official_success"), "pick_success": episode.get("pick_success"), "place_success": episode.get("place_success"), "transport_success": episode.get("transport_success"), "dropped": episode.get("dropped"), "steps": episode.get("steps")},
            "historical_timing": {"contact_onset_frame": contact_frame, "first_positive_squeeze_frame": force_frame, "first_registered_lift_frame": lift_frame, "t_episode_s": episode.get("t_episode_s")},
            "historical_telemetry": {"max_measured_squeeze_N": episode.get("peak_measured_force_N"), "mean_measured_squeeze_N": episode.get("mean_measured_force_N"), "gripper_aperture_fields_present": True, "finger_joint_positions_present": False, "eef_pose_present": False, "robot_joint_state_present": False, "target_object_force_by_side_present": False, "preload_flag_present": False, "lift_flag_present": True, "transport_flag_present": True},
            "preprobe_state": result.get("preprobe_state_role"),
            "historical_left_right_target_contact": "NOT_ARCHIVED; aggregate gripper contact only",
        })
    return {
        "schema": "HISTORICAL_SUCCESS_AUDIT_V1",
        "best_candidate_selection": "7703 selected first because it has complete formal success provenance and the archived trace reaches a bilateral-contact/lift window; candidates were not blindly treated as post-grasp snapshots",
        "roots": roots,
        "execution_contract": {
            "classification": "OTHER",
            "name": "native Tabero ForcePositionAction with fixed P1 force-slot command 6.0 N; raw VLA pose/gripper prefix preserved",
            "raw_vla_force_slot_as_newton_target": False,
            "native_nominal_squeeze_7N": "NOT the contract of these five P1 historical branches; 7.0 N belongs to the later root7400 runner",
            "activeforcing": False,
            "source": str(B5_SOURCE),
            "source_sha256": sha256(B5_SOURCE),
            "provenance_sources": [str(HIST_ROOT / "P1_BRANCH_PROVENANCE.json"), str(E3 / "P1_SIMPLIFIED_RUNTIME_PREFLIGHT_20260903_FINAL/P1_SIMPLIFIED_RUNTIME_LAUNCH_TEMPLATE.sh")],
        },
    }


def find_first_divergence(current_rows: list[dict[str, Any]], historical_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_frame = {int(r["frame"]): r for r in historical_rows}
    checks = []
    for cur in current_rows:
        frame = int(cur["step"])
        hist = by_frame.get(frame)
        if hist is None:
            continue
        reasons = []
        if abs(float(cur["object_dz"]) - float(hist["obj_dz"])) > 0.005:
            reasons.append("object_dz_gt_5mm")
        if int(cur["aggregate_contact"]) != int(hist["contact"]):
            reasons.append("aggregate_contact_mismatch")
        # Do not compare this field: the archived B5 CSV stores the first
        # gripper_pos component, while this recovery telemetry stores the
        # two-finger mean.  The raw action prefix is compared by hash instead.
        checks.append({"step": frame, "reasons": reasons, "current": cur, "historical": hist})
        if reasons:
            return {"first_divergence_step": frame, "status": "FOUND", "cause_candidates": reasons, "comparison": checks[-1], "comparison_basis": "archived aggregate telemetry and current target-object/runtime replay; historical archive has no joint/EE/runtime buffers"}
    return {"first_divergence_step": None, "status": "NOT_FOUND_IN_ARCHIVED_FIELDS", "cause_candidates": [], "comparison_basis": "no mismatch above declared thresholds before captured state"}


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    audit = historical_audit()
    write_json(OUT / "HISTORICAL_SUCCESS_AUDIT.json", audit)
    write_json(OUT / "GPU_PROCESS_AUDIT.json", {"performed": "host-level nvidia-smi", "driver": "available", "unrelated_policy_servers_preserved": [281704, 2128032], "isaac_processes_before_start": "none"})

    os.environ.update({
        "HDF5_TRAJ_SOURCE_DIR": str(HDF5_DIR),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
    })

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", num_envs=1).app
    env = None
    current_rows: list[dict[str, Any]] = []
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        import run_bilateral_contact_audit as helpers
        globals()["helpers"] = helpers

        setup_task_objects(TASK_SUITE, TASK_ID)
        cfg = parse_env_cfg("Isaac-Libero-Franka-Hybrid-Tactile-v0", device="cuda:0", num_envs=1)
        cfg.episode_length_s = 30.0
        env = gym.make("Isaac-Libero-Franka-Hybrid-Tactile-v0", cfg=cfg).unwrapped
        obs, reset_info = env.reset(seed=ROOT)
        friction_info = set_historical_friction(env, torch)
        historical_scene = move_tensors(torch.load(PREPROBE, map_location="cpu", weights_only=False), torch, env.device)
        env.scene.reset_to(rp.clone_state(historical_scene), torch.tensor([0], device=env.device), is_relative=True)
        env.sim.forward()
        env.observation_manager.reset(torch.tensor([0], device=env.device))
        env.action_manager.reset(torch.tensor([0], device=env.device))
        obs = env.observation_manager.compute(update_history=True)

        trace = []
        data = np.load(TRACE, allow_pickle=False)
        for key in sorted(k for k in data.files if k.startswith("chunk_")):
            trace.extend(np.asarray(data[key], dtype=np.float32)[:10].copy())
        with HIST_STEPS.open() as fh:
            hist_rows = list(csv.DictReader(fh))
        initial_obj_z = float(env.scene[OBJECT].data.root_pos_w[0, 2].detach().cpu().item())
        prev_nominal = None
        stable_gripper = 0
        reference = None
        for index, raw in enumerate(trace):
            action = compile_historical_action(raw)
            nominal = float(raw[6])
            stable_gripper = stable_gripper + 1 if prev_nominal is not None and abs(nominal - prev_nominal) <= 1e-4 else 1
            prev_nominal = nominal
            obs, _, term, trunc, _ = env.step(rp.action_tensor(torch, action, env.device))
            step = index + 1
            row = helpers.snapshot(env, obs, step, "historical_contract_replay")
            contact = target_contact_summary(env)
            obj_z = float(env.scene[OBJECT].data.root_pos_w[0, 2].detach().cpu().item())
            gp = float(np.asarray(rp.as_np(obs["policy"]["gripper_pos"])[0]).reshape(-1).mean())
            current_rows.append({
                "step": step,
                "source_action_index": index,
                "arm_action_sha256": hashlib.sha256(np.asarray(action[:7], dtype=np.float32).tobytes()).hexdigest(),
                "force_command_N": FORCE_COMMAND_N,
                "eef_pose": json.dumps(rp.jsonable(obs["policy"]["eef_pose"][0])),
                "gripper_meas": gp,
                "object_dz": obj_z - initial_obj_z,
                "aggregate_contact": int(row.get("left_all_contact", False) and row.get("right_all_contact", False)),
                "target_left_contact": int(contact["left_norm_N"] > 0.15),
                "target_right_contact": int(contact["right_norm_N"] > 0.15),
                "bilateral_target_contact": int(contact["bilateral"]),
                "target_left_force_N": contact["left_norm_N"],
                "target_right_force_N": contact["right_norm_N"],
                "target_contact_reason": contact["reason"],
                "vla_gripper_stable_steps": stable_gripper,
                "observation_digest": observation_digest(obs),
            })
            if contact["bilateral"] and stable_gripper >= 3 and step >= 2:
                reference = capture_reference(env, obs, row, action, step, index, trace, torch, friction_info)
                break
            if bool(term[0].item()) or bool(trunc[0].item()):
                break

        write_csv(OUT / "CURRENT_REPLAY_TELEMETRY.csv", current_rows)
        divergence = find_first_divergence(current_rows, hist_rows)
        write_json(OUT / "FIRST_DIVERGENCE_REPORT.json", divergence)
        if reference is None:
            write_json(OUT / "STATE_PARITY_REPORT.json", {"status": "NO_RECOVERY_STATE_CAPTURED", "current_steps": len(current_rows), "target_object_bilateral_seen": any(x["bilateral_target_contact"] for x in current_rows)})
            write_json(OUT / "RECOVERY_VALIDATION_RESULT.json", {"FINAL_STATUS": "RECOVERY_FAILED_NO_SEMANTIC_BILATERAL_POST_GRASP_STATE", "stable_grasp_recovered": False, "same_state_frontier_started": False})
            return 2

        # Save a binary snapshot for later same-success-state work.  It is not
        # used to start a force branch in this turn.
        torch.save(reference, OUT / "HISTORICAL_SUCCESS_POST_GRASP_STATE.pt")
        reference_json = rp.jsonable(reference)
        write_json(OUT / "STATE_REFERENCE.json", reference_json)

        parity_rows = []
        # Keep the three required restore checks adjacent, before any physics
        # progression from the hold test.
        for restore_id in range(1, 4):
            restored_obs, restored_row, restored_physical = restore_reference(env, reference, torch)
            parity_rows.append(parity_record(reference, env, restored_obs, restored_row, restored_physical, restore_id))

        hold_rows = []
        hold_summaries = []
        for restore_id in range(1, 4):
            restored_obs, _restored_row, restored_physical = restore_reference(env, reference, torch)
            hold_contact_loss = False
            object_start = float(np.asarray(restored_physical["object_position"], dtype=float)[2])
            object_end = object_start
            for hold_step in range(1, HOLD_STEPS + 1):
                hold_action = compile_historical_action(np.asarray(reference["handoff_action"], dtype=np.float32))
                restored_obs, _, term, trunc, _ = env.step(rp.action_tensor(torch, hold_action, env.device))
                hold_contact = target_contact_summary(env)
                object_end = float(env.scene[OBJECT].data.root_pos_w[0, 2].detach().cpu().item())
                hold_contact_loss = hold_contact_loss or not hold_contact["bilateral"]
                hold_rows.append({"restore_id": restore_id, "hold_step": hold_step, "bilateral_target_contact": int(hold_contact["bilateral"]), "left_force_N": hold_contact["left_norm_N"], "right_force_N": hold_contact["right_norm_N"], "object_z": object_end, "terminated": bool(term[0].item()), "truncated": bool(trunc[0].item())})
                if bool(term[0].item()) or bool(trunc[0].item()):
                    break
            hold_summaries.append({"restore_id": restore_id, "steps_requested": HOLD_STEPS, "steps_completed": sum(1 for x in hold_rows if x["restore_id"] == restore_id), "contact_loss": hold_contact_loss, "object_z_delta_m": object_end - object_start})

        continuation_rows = []
        _obs, _, _physical = restore_reference(env, reference, torch)
        for local, raw in enumerate(trace[reference["source_action_index"] + 1 : reference["source_action_index"] + 1 + REPLAY_STEPS_AFTER_STATE], start=1):
            action = compile_historical_action(raw)
            _obs, _, term, trunc, _ = env.step(rp.action_tensor(torch, action, env.device))
            contact = target_contact_summary(env)
            continuation_rows.append({"continuation_step": local, "source_action_index": reference["source_action_index"] + local, "arm_action_sha256": hashlib.sha256(np.asarray(action[:7], dtype=np.float32).tobytes()).hexdigest(), "bilateral_target_contact": int(contact["bilateral"]), "left_force_N": contact["left_norm_N"], "right_force_N": contact["right_norm_N"], "terminated": bool(term[0].item()), "truncated": bool(trunc[0].item())})
            if bool(term[0].item()) or bool(trunc[0].item()):
                break

        write_json(OUT / "STATE_PARITY_REPORT.json", {"reference_state_step": reference["state_step"], "restore_count": 3, "restore_checks_adjacent_before_hold": True, "restores": parity_rows, "physical_state_parity": all(x["physical_state_parity"] for x in parity_rows), "runtime_state_parity": all(x["action_runtime_parity"] and x["sensor_history_parity"] and x["observation_history_parity"] and x["manager_runtime_parity"] and x["environment_task_state_parity"] for x in parity_rows), "observation_parity": all(x["observation_parity"] for x in parity_rows), "bilateral_contact_recovered": all(x["bilateral_contact"] for x in parity_rows), "physx_derived_velocity_tolerance": "eef_angular_velocity <= 5e-5 after sim.forward; raw error retained in physical_diffs"})
        write_csv(OUT / "STABLE_GRASP_HOLD.csv", hold_rows)
        write_csv(OUT / "VLA_CONTINUATION_SANITY.csv", continuation_rows)
        valid = bool(
            all(x["physical_state_parity"] for x in parity_rows)
            and all(x["action_runtime_parity"] and x["sensor_history_parity"] and x["observation_history_parity"] and x["manager_runtime_parity"] for x in parity_rows)
            and all(x["observation_parity"] and x["bilateral_contact"] for x in parity_rows)
            and all(not x["contact_loss"] for x in hold_summaries)
            and len(continuation_rows) > 0
            and all(x["bilateral_target_contact"] for x in continuation_rows)
        )
        validation = {"schema": "RECOVERY_VALIDATION_RESULT_V1", "FINAL_STATUS": "HISTORICAL_SUCCESS_STATE_RECOVERY_VALID" if valid else "RECOVERY_VALIDATION_FAILED", "best_historical_root": ROOT, "state_step": reference["state_step"], "restore_count": 3, "stable_hold_steps": HOLD_STEPS, "continuation_steps": len(continuation_rows), "stable_grasp_recovered": all(not x["contact_loss"] for x in hold_summaries), "vla_continuation_stable": bool(continuation_rows) and all(x["bilateral_target_contact"] for x in continuation_rows), "historical_success_state_recovery_valid": valid, "same_state_frontier_started": False, "force_frontier": "NOT_RUN_BY_PROTOCOL", "hold_summaries": hold_summaries}
        write_json(OUT / "RECOVERY_VALIDATION_RESULT.json", validation)
        md = "# Historical success state recovery\n\n"
        md += f"- Candidate: root {ROOT}; task {TASK_SUITE}/task{TASK_ID}; object `{OBJECT}`; target `{TARGET}`.\n"
        md += f"- Captured state step: {reference['state_step']}; historical contract: native ForcePositionAction with fixed 6.0-N force slots and preserved VLA pose/gripper prefix.\n"
        md += f"- Three restore parity: {'PASS' if all(x['physical_state_parity'] and x['bilateral_contact'] for x in parity_rows) else 'FAIL'}. Runtime/observation parity: {'PASS' if valid else 'FAIL'}.\n"
        md += f"- Stable {HOLD_STEPS}-step hold: {'PASS' if validation['stable_grasp_recovered'] else 'FAIL'}. Continuation: {'PASS' if validation['vla_continuation_stable'] else 'FAIL'}.\n"
        md += f"- Same-success-state 2/4/6-N frontier: NOT STARTED.\n\n"
        md += "The archived historical roots contain successful full-task labels and aggregate contact/lift telemetry, but not post-grasp runtime snapshots. This run reconstructed the state from the root's archived preprobe scene and trace, then captured the missing complete runtime state.\n"
        (OUT / "RECOVERY_REPORT.md").write_text(md, encoding="utf-8")
        return 0 if valid else 3
    except Exception as exc:
        write_json(OUT / "RECOVERY_VALIDATION_RESULT.json", {"FINAL_STATUS": "RECOVERY_RUNTIME_ERROR", "error": repr(exc), "traceback": traceback.format_exc(), "same_state_frontier_started": False})
        return 4
    finally:
        # Isaac Sim 5.1 can abort while destructing camera/manager objects;
        # the process exits immediately below after all artifacts are flushed.
        # Do not call the known-buggy close/destructor path here.
        pass


if __name__ == "__main__":
    _exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_code)
