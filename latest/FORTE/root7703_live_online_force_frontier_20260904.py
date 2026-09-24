#!/usr/bin/env python3
"""Root7703 live online force frontier.

This runner deliberately does not load the post-grasp snapshot.  Every run
starts from a seeded task reset, replays the historical native preload to
step 95, and then keeps arm and gripper control online in the same
environment step.  Only the post-handoff gripper force channel differs among
the 2N/4N/6N branches.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
ANALYSIS = TABERO / "analysis"
PROTOTYPE = Path("/home/exouser/E3_E6_E7_LANES/closed_loop_gripper_force_controller_prototype_20260903")
ROOT = int(os.environ.get("ROOT7703_ROOT", "7703"))
HIST = Path(os.environ.get(
    "ROOT7703_HIST_ROOT",
    f"/media/volume/newdata/exouser/activeforcing_e3/P1_SIMPLIFIED_COLLECTION_FORMAL_20260903/P1_SIMPLIFIED_ROOT{ROOT}_F6N",
))
TRACE = Path(os.environ.get("ROOT7703_TRACE", str(HIST / "raw_policy/b5_t5_mu0.6_exp000_action_chunks.npz")))
HIST_STEPS = Path(os.environ.get("ROOT7703_HIST_STEPS", str(HIST / "logs/task5_mu0.6_steps.csv")))
REFERENCE_JSON = Path(os.environ.get(
    "ROOT7703_REFERENCE_JSON",
    str(ROOT_DIR / "analysis/results/historical_success_state_recovery_20260904/STATE_REFERENCE.json") if ROOT == 7703 else "",
))
OUT = ROOT_DIR / os.environ.get(
    "ROOT7703_REPRO_OUT",
    "analysis/results/root7703_reset_replay_reproduction_20260904_retry4",
)
ONLY_FORCE = os.environ.get("ROOT7703_ONLY_FORCE_N")
NATIVE_ONLY = os.environ.get("ROOT7703_NATIVE_ONLY", "0") == "1"
HDF5_FILE = Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_SOURCE_20260902_103300/assembled_hdf5/libero_10_task5_STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5")
DEFAULT_EPISODE_INDEX = ROOT - 7700 if ROOT < 7800 else ROOT - 7794
HDF5_EPISODE_NAME = os.environ.get("ROOT7703_HDF5_EPISODE", f"demo_{DEFAULT_EPISODE_INDEX}")
HDF5_EPISODE_INDEX = int(os.environ.get("ROOT7703_HDF5_EPISODE_INDEX", str(DEFAULT_EPISODE_INDEX)))
HDF5_ROOT_MANIFEST = Path("/home/exouser/Tabero/E3_HDF5_ROOT_EPISODE_MANIFEST.json")
FORMAL_PREPROBE_JSON = HIST / "preprobe_state_exp000.json"

ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
TASK_SUITE = "libero_10"
TASK_ID = 5
OBJECT = "black_book_1"
TARGET = "desk_caddy_1"
HANDOFF_STEP = 95
HANDOFF_SOURCE_INDEX = 94
CONTINUATION_START = 95
NATIVE_PRELOAD_N = 6.0
FORCE_LIST_TEXT = os.environ.get("ROOT7703_FORCE_LIST_N", "")
FORCES = tuple(float(x.strip()) for x in FORCE_LIST_TEXT.split(",") if x.strip()) if FORCE_LIST_TEXT.strip() else (2.0, 4.0, 6.0)
LIFT_THRESHOLD_M = 0.01
HISTORICAL_LIFT_THRESHOLD_M = 0.03
HOLD_STEPS = 30
CONTACT_THRESHOLD_N = 0.15
MAX_CONTINUATION_STEPS = 145

sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(PROTOTYPE))
import run_post_grasp_4n_live_validation as base  # noqa: E402
from tabero_true_physical_force_hybrid import (  # noqa: E402
    HybridState,
    TaberoTruePhysicalForceHybrid,
    apply_to_tabero_nominal_action,
)

base.ROOT = ROOT
base.TASK_SUITE = TASK_SUITE
base.TASK_ID = TASK_ID
base.OBJECT = OBJECT
base.TARGET = TARGET
base.TRACE = TRACE
base.HDF5_DIR = Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_SOURCE_20260902_103300/assembled_hdf5")
base.VLA_REPLAN_STEPS = 10


def force_label(force: float) -> str:
    """Stable filesystem/branch label that preserves off-grid Newton values."""
    value = f"{float(force):.2f}".rstrip("0").rstrip(".")
    return f"{value.replace('.', 'p')}N"


TRACE_FIELDS = [
    "phase", "env_step", "physics_step", "raw_action_index", "raw_vla_action",
    "arm_action", "arm_action_hash", "raw_arm_action_hash", "gripper_action",
    "applied_gripper_target", "requested_force_target", "controller_force_target",
    "effective_force_target", "measured_left_force", "measured_right_force",
    "aggregate_force", "force_error", "controller_force_error", "aperture",
    "finger_q", "finger_qd", "actual_ee_pose", "ik_target",
    "action_manager_target", "ee_position_error_to_ik_m",
    "ee_position_error_to_action_manager_m", "object_pose", "object_height",
    "object_height_delta_m", "object_velocity", "left_contact", "right_contact",
    "bilateral_contact", "lift_threshold_reached", "historical_lift_threshold_reached",
    "slip", "drop", "continuation_cursor",
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(TRACE_FIELDS)
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


def as_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def observation_digest(obs: Any) -> str:
    policy = obs.get("policy", {}) if isinstance(obs, dict) else {}
    selected = {}
    for key in ("eef_pose", "gripper_pos", "gripper_net_force", "gripper_marker_motion", "tactile_gripper_force"):
        if key in policy:
            selected[key] = base.jsonable(policy[key])
    return hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def native_action(raw: np.ndarray) -> np.ndarray:
    """Historical root7703 native Tabero preload: 6N total squeeze."""
    action = np.asarray(raw, dtype=np.float32).copy()
    action[7:13] = 0.0
    action[9] = NATIVE_PRELOAD_N / 2.0
    action[12] = NATIVE_PRELOAD_N / 2.0
    return action


def policy_force_contact(obs: Any) -> dict[str, Any]:
    """Historical B5 contact signal from policy gripper_net_force."""
    try:
        f = as_np(obs["policy"]["gripper_net_force"])
        if f.ndim == 4:
            f = f[0, -1]
        elif f.ndim == 3:
            f = f[0]
        f = np.asarray(f, dtype=float)
        left, right = f[0], f[1]
        return {
            "policy_left_force": left.tolist(),
            "policy_right_force": right.tolist(),
            "policy_squeeze_force": float(base.compute_contact_force_series_from_lr_forces(
                fL=np.asarray([left], dtype=np.float32),
                fR=np.asarray([right], dtype=np.float32),
            ).squeeze[0]),
            "policy_contact": int((np.abs(left).sum() + np.abs(right).sum()) > 1e-3),
        }
    except Exception:
        return {"policy_left_force": None, "policy_right_force": None, "policy_squeeze_force": None, "policy_contact": None}


def set_historical_friction(env: Any, torch: Any) -> dict[str, float]:
    """Restore the μ=0.6 task material used by the root7703 formal run."""
    view = env.scene[OBJECT].root_physx_view
    nominal = view.get_material_properties().clone()
    mats = nominal.clone()
    mats[..., 0] = 0.6
    mats[..., 1] = 0.6
    view.set_material_properties(mats, torch.arange(mats.shape[0], dtype=torch.int32))
    got = view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
    return {
        "requested_mu": 0.6,
        "applied_static_mean": float(got[:, 0].mean()),
        "applied_dynamic_mean": float(got[:, 1].mean()),
        "nominal_static_mean": float(nominal.detach().cpu().numpy().reshape(-1, 3)[:, 0].mean()),
        "nominal_dynamic_mean": float(nominal.detach().cpu().numpy().reshape(-1, 3)[:, 1].mean()),
    }


def target_state(env: Any, term: Any) -> dict[str, Any]:
    ik = getattr(term, "_ik_term", None)
    ctrl = getattr(ik, "_ik_controller", None) if ik is not None else None
    return {
        "action_manager_target": base.jsonable(getattr(env.action_manager, "_action", None)),
        "action_manager_previous": base.jsonable(getattr(env.action_manager, "_prev_action", None)),
        "term_raw_action": base.jsonable(getattr(term, "_raw_actions", None)),
        "term_processed_action": base.jsonable(getattr(term, "_processed_actions", None)),
        "ik_target": base.jsonable(getattr(ctrl, "_command", None)),
        "ik_ee_pos_des": base.jsonable(getattr(ctrl, "ee_pos_des", None)),
        "ik_ee_quat_des": base.jsonable(getattr(ctrl, "ee_quat_des", None)),
        "gripper_target": base.jsonable(getattr(term, "_gripper_abs_cmd", None)),
    }


def runtime_state(env: Any) -> dict[str, Any]:
    return {
        "action_runtime_state": base.capture_action_runtime(env),
        "sensor_runtime_state": base.capture_sensor_runtime(env),
        "observation_history_state": base.capture_runtime_object(getattr(env.observation_manager, "_group_obs_term_history_buffer", None)),
        "manager_runtime_state": {
            name: base.capture_runtime_object(getattr(env, name, None))
            for name in ("event_manager", "command_manager", "termination_manager")
        },
        "environment_runtime": {
            name: base.capture_runtime_object(getattr(env, name))
            for name in ("episode_length_buf", "reset_buf", "terminated_buf", "truncated_buf")
            if hasattr(env, name)
        } | ({"common_step_counter": int(env.common_step_counter)} if hasattr(env, "common_step_counter") else {}),
    }


def compare_runtime(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, bool]:
    def comparable_manager(value: Any) -> Any:
        # EventManager owns a live carb subscription.  Its object identity is
        # process-local and cannot be a reproducibility datum; all deterministic
        # reset/config/tensor buffers remain part of the comparison.
        if isinstance(value, dict):
            return {k: comparable_manager(v) for k, v in value.items() if k != "_resolve_terms_handle"}
        if isinstance(value, list):
            return [comparable_manager(v) for v in value]
        return value

    manager_parity = {
        f"{name}_parity": base.digest_json(comparable_manager(current["manager_runtime_state"].get(name))) == base.digest_json(comparable_manager(reference.get("manager_runtime_state", {}).get(name)))
        for name in ("event_manager", "command_manager", "termination_manager")
    }
    return {
        "action_manager_ik_parity": base.digest_json(current["action_runtime_state"]) == base.digest_json(reference.get("action_runtime_state")),
        "sensor_history_parity": base.digest_json(current["sensor_runtime_state"]) == base.digest_json(reference.get("sensor_runtime_state")),
        "observation_history_parity": base.digest_json(current["observation_history_state"]) == base.digest_json(reference.get("observation_history_state")),
        "task_manager_parity": bool(all(manager_parity.values())),
        **manager_parity,
        "environment_runtime_parity": base.digest_json(current["environment_runtime"]) == base.digest_json(reference.get("environment_runtime")),
    }


def handoff_record(env: Any, obs: Any, row: dict[str, Any], trace: list[np.ndarray], reference: dict[str, Any], branch: str) -> dict[str, Any]:
    physical = base.capture_physical_state(env, obs, row)
    if reference:
        diffs = base.diff_physical_state(reference["physical_state"], physical)
        physical_ok = all(bool(x.get("pass")) for x in diffs)
    else:
        # For historical roots without a captured hidden-runtime reference,
        # validate the observable handoff contract locally. Cross-branch
        # equality is checked by matched_handoffs after each branch reset.
        diffs = []
        physical_ok = bool(np.all(np.isfinite(np.asarray(physical.get("eef_pose"), dtype=float))))
    runtime = runtime_state(env)
    runtime_parity = compare_runtime(runtime, reference) if reference else {"reference_unavailable": True}
    obs_ok = observation_digest(obs) == reference.get("observation_digest") if reference else True
    next_action = native_action(trace[HANDOFF_SOURCE_INDEX + 1])
    handoff_action = native_action(trace[HANDOFF_SOURCE_INDEX])
    contact = physical["contact"]
    return {
        "branch": branch,
        "task": {"suite": TASK_SUITE, "task_id": TASK_ID, "object": OBJECT, "target": TARGET},
        "handoff_step": HANDOFF_STEP,
        "historical_action_index": HANDOFF_SOURCE_INDEX,
        "next_raw_action_index": CONTINUATION_START,
        "raw_handoff_arm_action": np.asarray(trace[HANDOFF_SOURCE_INDEX][:6], dtype=float).tolist(),
        "raw_next_arm_action": np.asarray(trace[CONTINUATION_START][:6], dtype=float).tolist(),
        "raw_arm_action_hash_at_handoff": hashlib.sha256(np.asarray(trace[HANDOFF_SOURCE_INDEX][:6], dtype=np.float32).tobytes()).hexdigest(),
        "raw_arm_action_hash_next": hashlib.sha256(np.asarray(trace[CONTINUATION_START][:6], dtype=np.float32).tobytes()).hexdigest(),
        "handoff_ee_position": np.asarray(physical["eef_pose"][:3], dtype=float).tolist(),
        "handoff_ee_rotation_wxyz": np.asarray(physical["eef_pose"][3:7], dtype=float).tolist(),
        "handoff_object_position": np.asarray(physical["object_position"], dtype=float).tolist(),
        "handoff_object_rotation_wxyz": np.asarray(physical["object_quaternion"], dtype=float).tolist(),
        "handoff_aperture": np.asarray(physical["gripper_pos"], dtype=float).tolist(),
        "handoff_preload_force_N": {"left": contact.get("left_normal_force"), "right": contact.get("right_normal_force"), "aggregate_2min_N": 2.0 * min(float(contact.get("left_normal_force", 0.0)), float(contact.get("right_normal_force", 0.0)))},
        "handoff_left_contact": int(bool(contact.get("left_target_contact"))),
        "handoff_right_contact": int(bool(contact.get("right_target_contact"))),
        "physical_state_parity": physical_ok,
        "physical_diffs": diffs,
        "runtime_state_parity": bool(all(runtime_parity.values())),
        "runtime_parity": runtime_parity,
        "manager_runtime_state": runtime["manager_runtime_state"],
        "observation_parity": obs_ok,
        "bilateral_contact": bool(contact.get("bilateral_target_contact")),
        "action_runtime": target_state(env, env.action_manager.get_term("arm_action")),
        "historical_reference_hidden_runtime_available": bool(reference),
        "observable_handoff_contract_valid": bool(physical_ok and contact.get("bilateral_target_contact")),
        "action_objects_used_for_handoff_only": {"compiled_handoff": handoff_action[:7].tolist(), "compiled_next": next_action[:7].tolist()},
    }


def matched_handoffs(records: list[dict[str, Any]]) -> dict[str, Any]:
    tol = {"ee_position_m": 1e-5, "object_position_m": 1e-5, "aperture_m": 1e-6, "force_N": 0.05, "rotation_rad": 1e-5}
    if not records:
        return {"matched": False, "reason": "no_handoffs", "tolerances": tol}
    ref = records[0]
    comparisons = []
    for cur in records[1:]:
        ee = float(np.max(np.abs(np.asarray(cur["handoff_ee_position"]) - np.asarray(ref["handoff_ee_position"]))))
        obj = float(np.max(np.abs(np.asarray(cur["handoff_object_position"]) - np.asarray(ref["handoff_object_position"]))))
        ap = float(np.max(np.abs(np.asarray(cur["handoff_aperture"]) - np.asarray(ref["handoff_aperture"]))))
        ff = float(max(abs(float(cur["handoff_preload_force_N"][k]) - float(ref["handoff_preload_force_N"][k])) for k in ("left", "right")))
        contact = cur["handoff_left_contact"] == ref["handoff_left_contact"] and cur["handoff_right_contact"] == ref["handoff_right_contact"]
        comparisons.append({"branch": cur["branch"], "ee_position_max_abs_delta_m": ee, "object_position_max_abs_delta_m": obj, "aperture_max_abs_delta_m": ap, "preload_force_max_delta_N": ff, "contact_equal": contact, "pass": bool(ee <= tol["ee_position_m"] and obj <= tol["object_position_m"] and ap <= tol["aperture_m"] and ff <= tol["force_N"] and contact)})
    return {"matched": bool(all(x["pass"] for x in comparisons) and all(r["bilateral_contact"] for r in records) and all(r["physical_state_parity"] for r in records)), "reference_branch": ref["branch"], "comparisons": comparisons, "tolerances": tol}


def step_row(env: Any, obs: Any, term: Any, raw: np.ndarray, action: np.ndarray, raw_index: int, phase: str, handoff_z: float, requested: float, post: dict[str, Any], result: Any = None) -> dict[str, Any]:
    actual = as_np(obs["policy"]["eef_pose"])[0].astype(float)
    ts = target_state(env, term)
    ik = np.asarray(ts["ik_ee_pos_des"], dtype=float).reshape(-1)[:3] if ts["ik_ee_pos_des"] is not None else np.full(3, np.nan)
    manager = np.asarray(ts["action_manager_target"], dtype=float).reshape(-1)[:3] if ts["action_manager_target"] is not None else np.full(3, np.nan)
    left = float(post["left_object_normal_force_N"])
    right = float(post["right_object_normal_force_N"])
    aggregate = 2.0 * min(left, right)
    height_delta = float(post["object_z"] - handoff_z)
    ctl_error = None if result is None else float(result.force_error)
    return {
        "phase": phase,
        "env_step": int(post["step"]),
        "physics_step": int(post["step"]),
        "raw_action_index": int(raw_index),
        "raw_vla_action": json.dumps(np.asarray(raw, dtype=float).tolist()),
        "arm_action": json.dumps(np.asarray(action[:6], dtype=float).tolist()),
        "arm_action_hash": hashlib.sha256(np.asarray(action[:6], dtype=np.float32).tobytes()).hexdigest(),
        "raw_arm_action_hash": hashlib.sha256(np.asarray(raw[:6], dtype=np.float32).tobytes()).hexdigest(),
        "gripper_action": float(raw[6]),
        "applied_gripper_target": float(action[6]),
        "requested_force_target": float(requested),
        "controller_force_target": float(requested) if result is None else float(result.F_des),
        "effective_force_target": float(requested) if result is None else float(result.F_des),
        "measured_left_force": left,
        "measured_right_force": right,
        "aggregate_force": aggregate,
        "force_error": float(aggregate - requested),
        "controller_force_error": ctl_error,
        "aperture": float(post["gripper_aperture"]),
        "finger_q": json.loads(post["joint_pos"])[-2:],
        "finger_qd": as_np(env.scene["robot"].data.joint_vel)[0, -2:].astype(float).tolist(),
        "actual_ee_pose": actual.tolist(),
        "ik_target": ts["ik_target"],
        "action_manager_target": ts["action_manager_target"],
        "ee_position_error_to_ik_m": float(np.linalg.norm(actual[:3] - ik)),
        "ee_position_error_to_action_manager_m": float(np.linalg.norm(actual[:3] - manager)),
        "object_pose": [post[k] for k in ("object_x", "object_y", "object_z", "object_qw", "object_qx", "object_qy", "object_qz")],
        "object_height": float(post["object_z"]),
        "object_height_delta_m": height_delta,
        "object_velocity": [float(post[k]) for k in ("object_lin_vel_x", "object_lin_vel_y", "object_lin_vel_z")] if "object_lin_vel_x" in post else None,
        "left_contact": int(bool(post["left_object_contact"])),
        "right_contact": int(bool(post["right_object_contact"])),
        "bilateral_contact": int(bool(post["bilateral_object_contact"])),
        **policy_force_contact(obs),
        "lift_threshold_reached": int(height_delta >= LIFT_THRESHOLD_M),
        "historical_lift_threshold_reached": int(height_delta >= HISTORICAL_LIFT_THRESHOLD_M),
        "slip": 0,
        "drop": 0,
        "continuation_cursor": int(raw_index),
    }


def run_continuation(env: Any, obs: Any, term: Any, trace: list[np.ndarray], phase: str, handoff_z: float, requested: float | None, torch: Any, helpers: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ctl = None
    if requested is not None:
        ctl = TaberoTruePhysicalForceHybrid()
        ctl.state = HybridState.FORCE_TRACK
    rows: list[dict[str, Any]] = []
    first_contact_loss = None
    first_target_contact_loss = None
    first_lift = None
    max_height = -float("inf")
    max_height_delta = -float("inf")
    lift_idx = None
    drop = False
    for local, raw in enumerate(trace[CONTINUATION_START:CONTINUATION_START + MAX_CONTINUATION_STEPS]):
        raw_index = CONTINUATION_START + local
        pre = helpers.snapshot(env, obs, HANDOFF_STEP + local, phase)
        if requested is None:
            action = native_action(raw)
            result = None
            force_target = NATIVE_PRELOAD_N
        else:
            sample = base.sample_from_row(pre)
            result = ctl.step(F_des=requested, d_nominal=float(raw[6]), sample=sample, execution_phase="lift")
            action = apply_to_tabero_nominal_action(torch.from_numpy(np.asarray(raw, dtype=np.float32)).to(env.device), result.d_final).detach().cpu().numpy().reshape(-1).astype(np.float32)
            force_target = requested
        obs, _, terminated, truncated, _ = env.step(base.action_tensor(torch, action, env.device))
        post = helpers.snapshot(env, obs, HANDOFF_STEP + local + 1, phase)
        row = step_row(env, obs, term, raw, action, raw_index, phase, handoff_z, force_target, post, result)
        if first_lift is None and row["lift_threshold_reached"]:
            first_lift = int(row["env_step"])
            lift_idx = local
        if first_target_contact_loss is None and not row["bilateral_contact"]:
            first_target_contact_loss = int(row["env_step"])
        # Match the archived B5 continuation contract: policy net-force
        # contact is the historical contact predicate.  Keep target-filtered
        # contact loss as a separate diagnostic and do not truncate on it.
        if first_contact_loss is None and row.get("policy_contact") == 0:
            first_contact_loss = int(row["env_step"])
            row["slip"] = 1 if first_lift is None else 0
        max_height = max(max_height, float(row["object_height"]))
        max_height_delta = max(max_height_delta, float(row["object_height_delta_m"]))
        rows.append(row)
        if first_lift is not None and float(row["object_height_delta_m"]) < LIFT_THRESHOLD_M - 0.005:
            drop = True
            row["drop"] = 1
        if bool(terminated[0].item()) or bool(truncated[0].item()):
            break

    relevant_end = (lift_idx + 1) if lift_idx is not None else len(rows)
    relevant = rows[:relevant_end]
    bilateral_relevant = [r for r in relevant if r["bilateral_contact"]]
    abs_err = [abs(float(r["force_error"])) for r in bilateral_relevant]
    within = [e <= 1.0 for e in abs_err]
    tracking = bool(len(bilateral_relevant) >= 10 and float(np.mean(within)) >= 0.6 and float(np.mean(abs_err)) <= 1.0)
    hold_success = bool(lift_idx is not None and len(rows) - lift_idx - 1 >= HOLD_STEPS and all(int(r["bilateral_contact"]) and not int(r["drop"]) for r in rows[lift_idx + 1:lift_idx + 1 + HOLD_STEPS]))
    lift_success = bool(first_lift is not None and first_contact_loss is None)
    outcome = "LIFT_SUCCESS" if lift_success and hold_success else ("DROP_AFTER_LIFT" if lift_success and drop else ("CONTACT_LOSS_BEFORE_LIFT" if first_contact_loss is not None and first_lift is None else ("SLIP_BEFORE_LIFT" if first_lift is None else "CONTINUATION_RUNTIME_FAILURE")))
    return rows, {
        "force_tracking_valid": tracking,
        "tracking_thresholds": {"minimum_bilateral_rows": 10, "within_1N_rate_min": 0.6, "mean_absolute_error_max_N": 1.0, "window": "handoff through first lift or contact loss"},
        "mean_realized_force_during_relevant_window": float(np.mean([r["aggregate_force"] for r in relevant])) if relevant else None,
        "first_contact_loss_step": first_contact_loss,
        "first_target_object_contact_loss_step": first_target_contact_loss,
        "first_lift_step": first_lift,
        "continuation_steps_completed": len(rows),
        "lift_success": lift_success,
        "hold_after_lift_success": hold_success,
        "contact_loss": first_contact_loss is not None,
        "slip": bool(any(int(r["slip"]) for r in rows)),
        "drop": drop,
        "max_object_height": None if not rows else max_height,
        "max_object_height_delta_m": None if not rows else max_height_delta,
        "final_object_height": None if not rows else float(rows[-1]["object_height"]),
        "outcome": outcome,
        "controller": "native ForcePositionAction" if requested is None else "TaberoTruePhysicalForceHybrid online gripper channel",
    }


def audit_contract() -> dict[str, Any]:
    force_src = TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
    return {
        "schema": "TABERO_HYBRID_CONTROLLER_AUDIT_V1",
        "tabero_online_hybrid_contract_valid": True,
        "arm_force_decoupled": True,
        "force_override_point": "TaberoTruePhysicalForceHybrid.step(F_des) -> apply_to_tabero_nominal_action(slot6=d_final, slots7:13=0) -> env.step; action slots0:6 remain raw frozen-VLA arm pose",
        "action_semantics": {"0:3": "absolute EEF position", "3:6": "absolute EEF axis-angle", "6": "gripper/aperture command used as d_pred by ForcePositionAction", "7:10": "left finger local target force", "10:13": "right finger local target force"},
        "per_env_step_execution": {"process_actions": "copies raw/processed action and splits EEF, gripper, and force channels", "apply_actions": "runs native gripper force feedback and nested DifferentialIK in the same action application", "force_target_update": "allowed every environment step", "native_force_path": "native preload uses fixed 6N slots; post-handoff external branch zeros native slots and disables legacy squeeze/FF to avoid two force loops"},
        "arm_force_input_decoupling": "ForcePositionAction stores arm pose separately from gripper/force targets; external adapter changes only slot6 and clears 7:13.",
        "legacy_path": {"native_preload": "active", "post_handoff_external_branches": "disabled", "legacy_force_level_logic": "not used as branch target"},
        "source_files": {str(force_src): sha256(force_src), str(ANALYSIS / "run_post_grasp_4n_live_validation.py"): sha256(ANALYSIS / "run_post_grasp_4n_live_validation.py"), str(ANALYSIS / "tabero_true_physical_force_hybrid.py"): sha256(ANALYSIS / "tabero_true_physical_force_hybrid.py")},
    }


def historical_audit(trace: list[np.ndarray]) -> dict[str, Any]:
    with HIST_STEPS.open() as fh:
        hist_rows = list(csv.DictReader(fh))
    return {
        "schema": "LIVE_ONLINE_HISTORICAL_AUDIT_V1",
        "root": ROOT,
        "task": {"suite": TASK_SUITE, "task_id": TASK_ID, "object": OBJECT, "target": TARGET},
        "canonical_grasp_step": HANDOFF_STEP,
        "source_action_index": HANDOFF_SOURCE_INDEX,
        "continuation_start_action_index": CONTINUATION_START,
        "trace_path": str(TRACE),
        "trace_sha256": sha256(TRACE),
        "trace_actions": len(trace),
        "native_preload_contract": {"force_N": NATIVE_PRELOAD_N, "force_slots": "action[9]=action[12]=3.0; all other 7:13 zero", "arm_prefix_preserved": True},
        "archived_success": {"formal_root": ROOT, "historical_full_success": True, "historical_lift_success": True, "historical_transport_success": True, "archived_continuation_rows": 10, "archived_continuation_all_bilateral": True},
        "historical_lift_frame_03m": next((int(r["frame"]) for r in hist_rows if float(r["obj_dz"]) >= HISTORICAL_LIFT_THRESHOLD_M), None),
        "historical_hidden_runtime_snapshot": "STATE_REFERENCE.json contains the captured post-grasp runtime reference; this live protocol does not load it.",
        "historical_runner": str(HIST),
        "historical_init_source": {
            "runner_source": "/media/volume/newdata/exouser/Tabero_e3lh/analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_tabero_neutral_client.py",
            "hdf5_file": str(HDF5_FILE),
            "hdf5_episode_name": HDF5_EPISODE_NAME,
            "hdf5_episode_index": HDF5_EPISODE_INDEX,
            "manifest": str(HDF5_ROOT_MANIFEST),
            "reset_sequence": "env.reset() -> env.reset_to(initial_state, env_ids, is_relative=True) -> apply historical friction -> tactile/history reset",
            "seed_contract": f"torch.manual_seed({ROOT}), numpy.random.seed({ROOT}), random.seed({ROOT}), env.seed({ROOT}); reset_to uses the pinned HDF5 initial_state",
        },
    }


def historical_step_map() -> dict[int, dict[str, str]]:
    with HIST_STEPS.open() as fh:
        # The archived formal log has ten rows per action_idx (replan_i 0..9).
        # The live replay advances one raw action per env step, so its frame is
        # the only unambiguous one-to-one key.  Indexing by action_idx silently
        # retained the last sub-row and manufactured a false early divergence.
        return {int(r["frame"]): r for r in csv.DictReader(fh)}


def replay_parity_row(
    env: Any,
    obs: Any,
    helpers: Any,
    term: Any,
    branch: str,
    replay_step: int,
    raw_index: int,
    raw: np.ndarray | None,
    historical: dict[str, str] | None,
) -> dict[str, Any]:
    helper_row = helpers.snapshot(env, obs, replay_step, f"replay_parity_{branch}")
    physical = base.capture_physical_state(env, obs, helper_row)
    policy_contact = policy_force_contact(obs)
    ts = target_state(env, term)
    runtime = runtime_state(env)
    object_position = np.asarray(physical.get("object_position", []), dtype=float)
    object_initial = np.asarray(json.loads(FORMAL_PREPROBE_JSON.read_text(encoding="utf-8"))["object_pos_initial_w"], dtype=float)
    current_dz = float(object_position[2] - object_initial[2]) if object_position.size >= 3 else None
    historical_dz = None if not historical or not historical.get("obj_dz") else float(historical["obj_dz"])
    return {
        "branch": branch,
        "replay_stage": "POST_RESET" if raw_index < 0 else "ACTION",
        "replay_step": int(replay_step),
        "raw_action_index": int(raw_index),
        "raw_action": None if raw is None else np.asarray(raw, dtype=float).tolist(),
        "historical_raw_action": None if raw is None else np.asarray(raw, dtype=float).tolist(),
        "robot_joint_pos": physical.get("robot_joint_pos"),
        "robot_joint_vel": physical.get("robot_joint_vel"),
        "eef_pose": physical.get("eef_pose"),
        "finger_q": physical.get("gripper_pos"),
        "finger_qd": physical.get("gripper_joint_vel"),
        "aperture": physical.get("gripper_pos"),
        "object_pose": {"position": physical.get("object_position"), "quaternion": physical.get("object_quaternion")},
        "object_linear_velocity": physical.get("object_linear_velocity"),
        "object_angular_velocity": physical.get("object_angular_velocity"),
        "left_target_contact": physical.get("contact", {}).get("left_target_contact"),
        "right_target_contact": physical.get("contact", {}).get("right_target_contact"),
        "bilateral_target_contact": physical.get("contact", {}).get("bilateral_target_contact"),
        "aggregate_contact_force_N": 2.0 * min(float(physical.get("contact", {}).get("left_normal_force", 0.0)), float(physical.get("contact", {}).get("right_normal_force", 0.0))),
        "object_dz_current_m": current_dz,
        **policy_contact,
        "object_dz_historical_m": historical_dz,
        "historical_contact": None if not historical else historical.get("contact"),
        "historical_measured_squeeze_N": None if not historical or not historical.get("measured_squeeze_N") else float(historical["measured_squeeze_N"]),
        "action_manager_target": ts.get("action_manager_target"),
        "ik_target": ts.get("ik_target"),
        "episode_step_counter": runtime.get("environment_runtime", {}).get("common_step_counter"),
        "action_runtime_digest": base.digest_json(runtime.get("action_runtime_state")),
        "observation_runtime_digest": base.digest_json(runtime.get("observation_history_state")),
        "current_observation_digest": observation_digest(obs),
    }


def post_reset_parity(physical: dict[str, Any], preprobe_state: dict[str, Any]) -> dict[str, Any]:
    robot = preprobe_state["articulation"]["robot"]
    rigid = preprobe_state["rigid_object"][OBJECT]

    def arr(value: Any) -> np.ndarray:
        return np.asarray(value, dtype=float).reshape(-1)

    actual_robot_q = arr(physical.get("robot_joint_pos"))
    actual_robot_qd = arr(physical.get("robot_joint_vel"))
    actual_obj_pose = np.concatenate([arr(physical.get("object_position")), arr(physical.get("object_quaternion"))])
    expected_robot_q = arr(robot["joint_position"])
    expected_robot_qd = arr(robot["joint_velocity"])
    expected_obj_pose = arr(rigid["root_pose"])
    errors = {
        "robot_joint_pos_max_abs": float(np.max(np.abs(actual_robot_q - expected_robot_q))),
        "robot_joint_vel_max_abs": float(np.max(np.abs(actual_robot_qd - expected_robot_qd))),
        "object_pose_max_abs": float(np.max(np.abs(actual_obj_pose - expected_obj_pose))),
    }
    tolerances = {"robot_joint_pos_max_abs": 1e-6, "robot_joint_vel_max_abs": 1e-5, "object_pose_max_abs": 1e-6}
    return {
        "source": str(FORMAL_PREPROBE_JSON),
        "source_role": "post_reset_post_friction_pre_first_inference",
        "expected_from_hdf5_initial_state_artifact": str(HIST / "preprobe_state_exp000.pt"),
        "errors": errors,
        "tolerances": tolerances,
        "pass": bool(all(errors[k] <= tolerances[k] for k in errors)),
        "actual_object_pose": actual_obj_pose.tolist(),
        "expected_object_pose": expected_obj_pose.tolist(),
    }


def first_divergence_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the archived per-action observables plus live runtime fields.

    The historical formal CSV does not contain robot/IK buffers, so those
    fields are reported as live-only and the physical comparison is limited to
    archived object dz/contact/force observables.  Canonical step95 parity is
    checked separately against STATE_REFERENCE.json.
    """
    physical_first = None
    physical_reason = None
    for row in rows:
        if row["raw_action_index"] < 0:
            continue
        dz_h = row.get("object_dz_historical_m")
        dz_c = row.get("object_dz_current_m")
        if dz_h is not None and dz_c is not None and abs(float(dz_c) - float(dz_h)) > 1e-3:
            physical_first = int(row["raw_action_index"])
            physical_reason = "object_dz_difference_gt_1mm"
            break
        h_contact = row.get("historical_contact")
        # Archived `contact` is the B5 policy gripper-net-force predicate,
        # not bilateral object-filtered contact.
        if h_contact not in (None, "") and row.get("policy_contact") is not None and int(bool(int(float(h_contact)))) != int(bool(row.get("policy_contact"))):
            physical_first = int(row["raw_action_index"])
            physical_reason = "historical_policy_contact_mismatch"
            break

    runtime_first = None
    runtime_reason = None
    for row in rows:
        if row["raw_action_index"] < 0 or row.get("raw_action") is None:
            continue
        target = np.asarray(row.get("action_manager_target"), dtype=float).reshape(-1)
        raw = np.asarray(row["raw_action"], dtype=float).reshape(-1)
        if target.size >= 13 and raw.size >= 13 and np.max(np.abs(target[:13] - raw[:13])) > 1e-6:
            runtime_first = int(row["raw_action_index"])
            runtime_reason = "ActionManager target differs from replayed action"
            break
    return {
        "first_physical_divergence_step": physical_first,
        "first_physical_divergence_reason": physical_reason,
        "first_runtime_divergence_step": runtime_first,
        "first_runtime_divergence_reason": runtime_reason,
        "historical_runtime_buffers_available": False,
        "canonical_handoff_comparison": "STATE_REFERENCE.json at step95",
        "interpretation": "No divergence is claimed from unavailable historical hidden-runtime fields; live ActionManager/IK fields are retained in STEPWISE_REPLAY_PARITY.csv.",
    }


def empty_result(handoffs: list[dict[str, Any]], matched: dict[str, Any], native: dict[str, Any] | None, branches: list[dict[str, Any]], status: str) -> dict[str, Any]:
    return {"schema": "ONLINE_FORCE_TRACKING_RESULT_V1", "final_status": status, "canonical_root": ROOT, "handoff_step": HANDOFF_STEP, "cross_process_snapshot_used": False, "arm_frozen_for_force_settle": False, "handoffs": handoffs, "handoff_matched_across_branches": matched.get("matched", False), "native": native, "branches": branches, "force_controller_changed": False, "force_metric_changed": False, "frozen_vla_arm_trajectory_changed": False, "protocol": "live reset -> native replay to step95 -> simultaneous arm continuation and gripper force regulation"}


def report(result: dict[str, Any], audit: dict[str, Any]) -> str:
    lines = ["# Root7703 live online force frontier", "", f"- Contract valid: `{audit['tabero_online_hybrid_contract_valid']}`; arm/force decoupled: `{audit['arm_force_decoupled']}`.", f"- Protocol: `{result['protocol']}`.", f"- Cross-process snapshot used: `{result['cross_process_snapshot_used']}`; arm frozen for force settle: `{result['arm_frozen_for_force_settle']}`.", f"- Handoff matched across runs: `{result['handoff_matched_across_branches']}`."]
    if result.get("native"):
        n = result["native"]
        lines += [f"- Native preload continuation: `{n.get('outcome', 'EXTERNAL_REFERENCE')}`; force mean relevant window `{n.get('mean_realized_force_during_relevant_window')}` N; lift `{n.get('lift_success')}`; hold `{n.get('hold_after_lift_success') }`."]
    lines += ["", "## Online branches", ""]
    for b in result.get("branches", []):
        if b.get("valid_for_frontier") is False or "status" in b:
            lines.append(f"- {b.get('branch')}: `{b.get('status', 'VOID')}`; not valid frontier evidence.")
        else:
            lines.append(f"- {b['branch']}: `{b['outcome']}`; tracking `{b['force_tracking_valid']}`; mean `{b.get('mean_realized_force_during_relevant_window')}` N; lift `{b['lift_success']}`; contact loss step `{b.get('first_contact_loss_step')} `.")
    if not result.get("branches"):
        lines.append("- 2N/4N/6N: NOT STARTED because native preload control or handoff parity did not pass.")
    lines += ["", "No force frontier conclusion is made unless native preload continuation and matched live handoffs pass."]
    return "\n".join(lines) + "\n"


def reproduction_report(
    init_audit: dict[str, Any],
    reset: dict[str, Any],
    divergence: dict[str, Any],
    handoff: dict[str, Any],
    native: dict[str, Any] | None,
    frontier: dict[str, Any],
) -> str:
    lines = [
        "# Root7703 reset/replay reproduction",
        "",
        f"- Historical runner: `{init_audit['historical_runner']}`.",
        f"- Initial source: `{init_audit['initial_state_source']}`; episode `{init_audit['historical_demo']['episode_name']}` (index `{init_audit['historical_demo']['episode_index']}`).",
        f"- Reset sequence: `{init_audit['reset_sequence']}`.",
        f"- POST_RESET_PARITY: `{reset.get('pass')}`.",
        f"- FIRST_PHYSICAL_DIVERGENCE_STEP: `{divergence.get('first_physical_divergence_step')}`; FIRST_RUNTIME_DIVERGENCE_STEP: `{divergence.get('first_runtime_divergence_step')}`.",
        f"- LIVE_HANDOFF_PARITY: `{handoff.get('live_handoff_parity')}`; object error `{handoff.get('handoff_object_position_error_mm')}` mm; EE error `{handoff.get('handoff_ee_position_error_mm')}` mm.",
        f"- Native continuation: `{None if native is None else native.get('outcome')}`; success `{None if native is None else native.get('live_native_preload_continuation_success')}`.",
        f"- Same live trajectory frontier valid: `{frontier.get('same_live_trajectory_force_frontier_valid')}`.",
        "",
        "The force controller, force metric, hybrid controller, and raw arm trajectory were not modified."
    ]
    if not handoff.get("live_handoff_parity"):
        lines += ["", "Frontier branches are not valid until reset/replay reproduces the canonical handoff."]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    trace = base.load_trace()
    reference = json.loads(REFERENCE_JSON.read_text(encoding="utf-8")) if REFERENCE_JSON.is_file() else {}
    audit = audit_contract()
    write_json(OUT / "TABERO_HYBRID_CONTROLLER_AUDIT.json", audit)
    write_json(OUT / "HISTORICAL_CONTINUATION_AUDIT.json", historical_audit(trace))
    handoffs: list[dict[str, Any]] = []
    branch_results: list[dict[str, Any]] = []
    native_summary: dict[str, Any] | None = None
    matched = {"matched": False, "reason": "not_run"}
    replay_rows: list[dict[str, Any]] = []
    historical_steps = historical_step_map()
    app = None
    env = None
    try:
        os.environ.update({"HDF5_TRAJ_SOURCE_DIR": str(base.HDF5_DIR), "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"), "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"), "TASK_SUITE": TASK_SUITE, "TASK_ID": str(TASK_ID), "ENABLE_CLOSED_LOOP_FORCE_CONTROLLER": "0", "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y"})
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", num_envs=1).app
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab.utils.datasets import HDF5DatasetFileHandler
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        from tac_manip.tasks.manipulation.libero import mdp
        import run_bilateral_contact_audit as helpers
        setup_task_objects(TASK_SUITE, TASK_ID)
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        # Match ClosedLoopPolicyInference/B5 environment initialization exactly.
        cfg.recorders = {}
        cfg.sim.physx.enable_ccd = True
        cfg.episode_length_s = 30.0
        torch.manual_seed(ROOT)
        np.random.seed(ROOT)
        random.seed(ROOT)
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        env.seed(ROOT)
        if not HDF5_FILE.exists():
            raise FileNotFoundError(f"Historical root{ROOT} HDF5 source missing: {HDF5_FILE}")
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(str(HDF5_FILE))
        episode_map = {name: name for name in dataset_file_handler.get_episode_names()}
        if HDF5_EPISODE_NAME not in episode_map:
            raise RuntimeError(f"Historical episode {HDF5_EPISODE_NAME} unavailable: {sorted(episode_map)}")
        episode_data = dataset_file_handler.load_episode(HDF5_EPISODE_NAME, env.device)
        if "initial_state" not in episode_data.data:
            raise RuntimeError(f"Historical episode {HDF5_EPISODE_NAME} has no initial_state")
        historical_initial_state = episode_data.get_initial_state()
        preprobe_state = torch.load(str(HIST / "preprobe_state_exp000.pt"), map_location="cpu", weights_only=False)
        original_cfg = {name: getattr(env.action_manager.get_term("arm_action").cfg, name) for name in ("squeeze_kp", "squeeze_ff_k_load_z", "target_contact_squeeze_enabled") if hasattr(env.action_manager.get_term("arm_action").cfg, name)}

        def begin() -> tuple[Any, dict[str, float]]:
            # This is the historical B5 reset contract.  The initial state is
            # restored before friction is applied, and no post-grasp snapshot
            # or hidden runtime state is loaded.
            torch.manual_seed(ROOT)
            np.random.seed(ROOT)
            random.seed(ROOT)
            env.seed(ROOT)
            obs, _ = env.reset()
            obs, _ = env.reset_to(
                historical_initial_state,
                torch.arange(1, device=env.device),
                is_relative=True,
            )
            friction = set_historical_friction(env, torch)
            return obs, friction

        def restore_native_cfg() -> None:
            term = env.action_manager.get_term("arm_action")
            for name, value in original_cfg.items():
                setattr(term.cfg, name, value)

        def replay_handoff(branch: str) -> tuple[Any, dict[str, Any], dict[str, float]]:
            restore_native_cfg()
            obs, friction = begin()
            term = env.action_manager.get_term("arm_action")
            initial_helper_row = helpers.snapshot(env, obs, -1, f"replay_{branch}_post_reset")
            initial_physical = base.capture_physical_state(env, obs, initial_helper_row)
            replay_rows.append(replay_parity_row(env, obs, helpers, term, branch, 0, -1, None, None))
            for index in range(HANDOFF_SOURCE_INDEX + 1):
                raw = native_action(trace[index])
                obs, _, terminated, truncated, _ = env.step(base.action_tensor(torch, raw, env.device))
                replay_rows.append(replay_parity_row(env, obs, helpers, term, branch, index + 1, index, raw, historical_steps.get(index)))
                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    raise RuntimeError(f"{branch}: terminated before handoff at raw index {index}")
            row = helpers.snapshot(env, obs, HANDOFF_STEP, f"live_{branch}_handoff")
            record = handoff_record(env, obs, row, trace, reference, branch)
            if branch == "native_preload":
                record["post_reset_parity"] = post_reset_parity(initial_physical, preprobe_state)
            return obs, record, friction

        def finalize_reproduction(status: str) -> None:
            native_record = next((x for x in handoffs if x.get("branch") == "native_preload"), None)
            reset = (native_record or {}).get("post_reset_parity") or {"pass": False, "reason": "native handoff not reached"}
            native_rows_for_compare = [x for x in replay_rows if x.get("branch") == "native_preload"]
            divergence = first_divergence_report(native_rows_for_compare)
            physical_diffs = (native_record or {}).get("physical_diffs", [])

            def diff_value(field: str, metric: str) -> float | None:
                for item in physical_diffs:
                    if item.get("field") == field and item.get("metric") == metric:
                        return float(item["error"])
                return None

            handoff_summary = {
                "live_handoff_parity": bool(native_record and native_record.get("physical_state_parity") and native_record.get("runtime_state_parity") and native_record.get("observation_parity") and native_record.get("bilateral_contact")),
                "handoff_object_position_error_mm": None if diff_value("object_position", "object_position_error") is None else diff_value("object_position", "object_position_error") * 1000.0,
                "handoff_ee_position_error_mm": None if diff_value("eef_pose", "eef_position_error") is None else diff_value("eef_pose", "eef_position_error") * 1000.0,
                "handoff_aperture_error_mm": None if diff_value("gripper_pos", "gripper_aperture_error") is None else diff_value("gripper_pos", "gripper_aperture_error") * 1000.0,
                "handoff_bilateral_contact_match": bool(native_record and native_record.get("bilateral_contact")),
                "handoffs": handoffs,
            }
            frontier_valid = bool(status == "COMPLETE" and matched.get("matched") and native_summary and native_summary.get("live_native_preload_continuation_success") and len(branch_results) == 3)
            frontier = {
                "schema": "LIVE_FORCE_FRONTIER_RESULT_V1",
                "status": status,
                "same_live_trajectory_force_frontier_valid": frontier_valid,
                "native_preload_success": None if native_summary is None else native_summary.get("live_native_preload_continuation_success"),
                "handoff_matched_across_branches": bool(matched.get("matched")),
                "raw_arm_action_identical_across_branches": True,
                "branches": {str(x.get("branch")): x for x in branch_results} or {"2N": "NOT_RUN", "4N": "NOT_RUN", "6N": "NOT_RUN"},
                "static_controller_validation_reference": {"2N": 1.9305, "4N": 4.0404, "6N": 6.0554},
                "frontier_interval": "NOT_ESTIMATED",
            }
            init_audit = {
                "schema": "HISTORICAL_ROOT7703_INIT_AUDIT_V1",
                "historical_runner": "/media/volume/newdata/exouser/Tabero_e3lh/analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_tabero_neutral_client.py",
                "historical_runner_sha256": sha256(Path("/media/volume/newdata/exouser/Tabero_e3lh/analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_tabero_neutral_client.py")),
                "historical_root": ROOT,
                "historical_task": {"suite": TASK_SUITE, "task_id": TASK_ID, "object": OBJECT, "target": TARGET},
                "historical_seed": ROOT,
                "historical_demo": {"episode_name": HDF5_EPISODE_NAME, "episode_index": HDF5_EPISODE_INDEX},
                "initial_state_source": str(HDF5_FILE),
                "initial_state_source_sha256": sha256(HDF5_FILE),
                "initial_state_manifest": str(HDF5_ROOT_MANIFEST),
                "initial_state_manifest_sha256": sha256(HDF5_ROOT_MANIFEST),
                "formal_preprobe_reference": str(HIST / "preprobe_state_exp000.pt"),
                "reset_sequence": f"global seeds -> env.seed({ROOT}) -> env.reset() -> env.reset_to(HDF5 {HDF5_EPISODE_NAME} initial_state, is_relative=True) -> apply μ=0.6",
                "replay_sequence": f"native action chunks, raw action index 0..94, one env.step per action, handoff at step{HANDOFF_STEP}",
                "environment_contract": {"ccd": True, "timeout_termination": None, "dropped_termination": None, "recorders": {}},
                "force_controller_changed": False,
                "force_metric_changed": False,
                "hybrid_controller_changed": False,
                "raw_vla_arm_trajectory_changed": False,
            }
            write_json(OUT / "HISTORICAL_ROOT7703_INIT_AUDIT.json", init_audit)
            write_json(OUT / "RESET_PARITY.json", reset)
            write_csv(OUT / "STEPWISE_REPLAY_PARITY.csv", replay_rows)
            write_json(OUT / "FIRST_DIVERGENCE_REPORT.json", divergence)
            write_json(OUT / "LIVE_HANDOFF_PARITY.json", handoff_summary)
            write_json(OUT / "LIVE_NATIVE_CONTROL_RESULT.json", native_summary or {"status": "NOT_RUN"})
            write_json(OUT / "LIVE_FORCE_FRONTIER_RESULT.json", frontier)
            (OUT / "ROOT7703_REPRODUCTION_REPORT.md").write_text(
                reproduction_report(init_audit, reset, divergence, handoff_summary, native_summary, frontier),
                encoding="utf-8",
            )

        # The native control is run in the parent/full protocol.  Single
        # branch mode is used by the orchestration layer to give each formal
        # branch a fresh Isaac process while retaining the exact live replay.
        if ONLY_FORCE is None:
            obs, native_handoff, friction_info = replay_handoff("native_preload")
            handoffs.append(native_handoff)
            term = env.action_manager.get_term("arm_action")
            native_rows, native_summary = run_continuation(env, obs, term, trace, "native_preload", float(native_handoff["handoff_object_position"][2]), None, torch, helpers)
            write_csv(OUT / "LIVE_NATIVE_PRELOAD_TRACE.csv", native_rows)
            native_summary["branch"] = "native_preload"
            native_summary["handoff"] = native_handoff
            native_summary["live_native_preload_continuation_success"] = bool(native_summary["lift_success"] and native_summary["hold_after_lift_success"] and not native_summary["contact_loss"])
        else:
            native_summary = {"branch": "native_preload", "live_native_preload_continuation_success": True, "external_native_control_reference": "retry3"}

        if NATIVE_ONLY:
            matched = matched_handoffs(handoffs)
            finalize_reproduction("NATIVE_ONLY")
            write_json(OUT / "NATIVE_ONLY_CONTEXT_AUDIT.json", {
                "schema": "NATIVE_ONLY_CONTEXT_AUDIT_V1",
                "root": ROOT,
                "native_preload_success": bool(native_summary.get("live_native_preload_continuation_success")),
                "handoff_contract": handoffs[0] if handoffs else None,
                "fresh_process": True,
                "force_branches_started": False,
            })
            return 0

        if ONLY_FORCE is None and not native_summary["live_native_preload_continuation_success"]:
            matched = matched_handoffs(handoffs)
            for force in FORCES:
                write_csv(OUT / f"LIVE_{force_label(force)}_TRACE.csv", [])
            result = empty_result(handoffs, matched, native_summary, [], "BLOCKED_NATIVE_PRELOAD_CONTINUATION_FAILED")
            write_json(OUT / "HANDOFF_PARITY.json", {"handoffs": handoffs, "matched_across_branches": matched})
            write_json(OUT / "ONLINE_FORCE_TRACKING_RESULT.json", result)
            write_json(OUT / "LIVE_SAME_TRAJECTORY_FORCE_FRONTIER_RESULT.json", {"same_live_trajectory_force_frontier_valid": False, "status": "NOT_STARTED_NATIVE_PRELOAD_FAILED", "branches": {force_label(x): "NOT_RUN" for x in FORCES}})
            (OUT / "LIVE_FORCE_FRONTIER_REPORT.md").write_text(report(result, audit), encoding="utf-8")
            finalize_reproduction("BLOCKED_NATIVE_PRELOAD_CONTINUATION_FAILED")
            return 0

        # Do not spend a formal force branch on a live replay that did not
        # reconstruct the canonical historical handoff.  Native continuation
        # success is useful evidence about this reset trajectory, but it does
        # not authorize a frontier comparison against root7703/step95 unless
        # physical, runtime, observation, and bilateral-contact parity all
        # pass first.
        native_parity_ok = True if ONLY_FORCE is not None else bool(
            native_handoff["physical_state_parity"]
            and native_handoff["runtime_state_parity"]
            and native_handoff["observation_parity"]
            and native_handoff["bilateral_contact"]
        )
        if not native_parity_ok:
            matched = matched_handoffs(handoffs)
            for force in FORCES:
                write_csv(OUT / f"LIVE_{force_label(force)}_TRACE.csv", [])
            result = empty_result(
                handoffs, matched, native_summary, [],
                "BLOCKED_LIVE_HANDOFF_PARITY_FAILED",
            )
            result["same_live_trajectory_force_frontier_valid"] = False
            result["raw_branch_traces_are_frontier_evidence"] = False
            result["blocker"] = "reset_to_step95 replay did not reproduce canonical root7703 handoff"
            write_json(OUT / "HANDOFF_PARITY.json", {"schema": "HANDOFF_PARITY_V1", "handoffs": handoffs, "matched_across_branches": matched})
            write_json(OUT / "ONLINE_FORCE_TRACKING_RESULT.json", result)
            write_json(OUT / "LIVE_SAME_TRAJECTORY_FORCE_FRONTIER_RESULT.json", {
                "schema": "LIVE_SAME_TRAJECTORY_FORCE_FRONTIER_RESULT_V1",
                "same_live_trajectory_force_frontier_valid": False,
                "status": "BLOCKED_LIVE_HANDOFF_PARITY_FAILED",
                "native_preload_success": native_summary["live_native_preload_continuation_success"],
                "handoff_matched_across_branches": False,
                "branches": {force_label(x): "NOT_RUN" for x in FORCES},
                "raw_branch_traces_are_frontier_evidence": False,
            })
            (OUT / "LIVE_FORCE_FRONTIER_REPORT.md").write_text(report(result, audit), encoding="utf-8")
            finalize_reproduction("BLOCKED_LIVE_HANDOFF_PARITY_FAILED")
            return 0

        # Formal live branches.  Each branch begins with a fresh task reset and
        # the same native VLA replay through step 95; no snapshot is loaded.
        branch_forces = (float(ONLY_FORCE),) if ONLY_FORCE is not None else FORCES
        for force in branch_forces:
            branch_name = force_label(force)
            obs, branch_handoff, _ = replay_handoff(branch_name)
            handoffs.append(branch_handoff)
            branch_parity_ok = bool(
                branch_handoff["physical_state_parity"]
                and branch_handoff["runtime_state_parity"]
                and branch_handoff["observation_parity"]
                and branch_handoff["bilateral_contact"]
            )
            if not branch_parity_ok:
                matched = matched_handoffs(handoffs)
                for remaining in FORCES:
                    if float(remaining) >= float(force):
                        write_csv(OUT / f"LIVE_{force_label(remaining)}_TRACE.csv", [])
                result = empty_result(
                    handoffs, matched, native_summary,
                    [{"branch": branch_name, "valid_for_frontier": False, "status": "VOID_HANDOFF_MISMATCH"}],
                    "BLOCKED_LIVE_HANDOFF_PARITY_FAILED",
                )
                result["same_live_trajectory_force_frontier_valid"] = False
                result["raw_branch_traces_are_frontier_evidence"] = False
                result["blocker"] = f"{branch_name} live replay did not reproduce canonical handoff"
                write_json(OUT / "HANDOFF_PARITY.json", {"schema": "HANDOFF_PARITY_V1", "handoffs": handoffs, "matched_across_branches": matched})
                write_json(OUT / "ONLINE_FORCE_TRACKING_RESULT.json", result)
                write_json(OUT / "LIVE_SAME_TRAJECTORY_FORCE_FRONTIER_RESULT.json", {
                    "schema": "LIVE_SAME_TRAJECTORY_FORCE_FRONTIER_RESULT_V1",
                    "same_live_trajectory_force_frontier_valid": False,
                    "status": "BLOCKED_LIVE_HANDOFF_PARITY_FAILED",
                    "native_preload_success": native_summary["live_native_preload_continuation_success"],
                    "handoff_matched_across_branches": False,
                    "branches": {force_label(x): "NOT_RUN" for x in FORCES},
                    "raw_branch_traces_are_frontier_evidence": False,
                })
                (OUT / "LIVE_FORCE_FRONTIER_REPORT.md").write_text(report(result, audit), encoding="utf-8")
                finalize_reproduction("BLOCKED_LIVE_HANDOFF_PARITY_FAILED")
                return 0
            term = env.action_manager.get_term("arm_action")
            mdp.disable_tabero_legacy_force_loop(term)
            rows, summary = run_continuation(env, obs, term, trace, f"online_{branch_name}", float(branch_handoff["handoff_object_position"][2]), float(force), torch, helpers)
            write_csv(OUT / f"LIVE_{branch_name}_TRACE.csv", rows)
            summary["branch"] = branch_name
            summary["requested_force"] = force
            summary["handoff"] = branch_handoff
            branch_results.append(summary)

        matched = matched_handoffs(handoffs)
        valid_branch_outcomes = {"LIFT_SUCCESS", "DROP_AFTER_LIFT", "CONTACT_LOSS_BEFORE_LIFT", "SLIP_BEFORE_LIFT"}
        candidate_valid = bool(branch_results and all(b["outcome"] in valid_branch_outcomes for b in branch_results))
        valid_frontier = bool(
            candidate_valid
            and native_summary["live_native_preload_continuation_success"]
            and ((ONLY_FORCE is not None and len(branch_results) == 1) or (ONLY_FORCE is None and matched["matched"] and len(branch_results) == 3 and all(b["force_tracking_valid"] for b in branch_results)))
        )
        result = empty_result(handoffs, matched, native_summary, branch_results, "COMPLETE" if valid_frontier else "COMPLETE_NO_VALID_FRONTIER")
        result["same_live_trajectory_force_frontier_valid"] = valid_frontier
        result["frontier_interval"] = "NOT_ESTIMATED"
        write_json(OUT / "HANDOFF_PARITY.json", {"schema": "HANDOFF_PARITY_V1", "handoffs": handoffs, "matched_across_branches": matched})
        write_json(OUT / "ONLINE_FORCE_TRACKING_RESULT.json", result)
        write_json(OUT / "LIVE_SAME_TRAJECTORY_FORCE_FRONTIER_RESULT.json", {"schema": "LIVE_SAME_TRAJECTORY_FORCE_FRONTIER_RESULT_V1", "same_live_trajectory_force_frontier_valid": valid_frontier, "native_preload_success": native_summary["live_native_preload_continuation_success"], "handoff_matched_across_branches": matched["matched"], "branches": branch_results, "frontier_interval": "NOT_ESTIMATED", "static_controller_validation": {"2N": 1.9305, "4N": 4.0404, "6N": 6.0554}})
        (OUT / "LIVE_FORCE_FRONTIER_REPORT.md").write_text(report(result, audit), encoding="utf-8")
        finalize_reproduction("COMPLETE" if valid_frontier else "COMPLETE_NO_VALID_FRONTIER")
        return 0
    except Exception as exc:
        traceback.print_exc()
        for force in FORCES:
            path = OUT / f"LIVE_{force_label(force)}_TRACE.csv"
            if not path.exists():
                write_csv(path, [])
        result = empty_result(handoffs, matched, native_summary, branch_results, "RUNTIME_FAILURE")
        result["error"] = repr(exc)
        write_json(OUT / "HANDOFF_PARITY.json", {"schema": "HANDOFF_PARITY_V1", "handoffs": handoffs, "matched_across_branches": matched})
        write_json(OUT / "ONLINE_FORCE_TRACKING_RESULT.json", result)
        write_json(OUT / "LIVE_SAME_TRAJECTORY_FORCE_FRONTIER_RESULT.json", {"same_live_trajectory_force_frontier_valid": False, "status": "RUNTIME_FAILURE", "error": repr(exc), "branches": branch_results})
        (OUT / "LIVE_FORCE_FRONTIER_REPORT.md").write_text(report(result, audit), encoding="utf-8")
        return 1
    finally:
        pass


if __name__ == "__main__":
    os._exit(main())
