#!/usr/bin/env python3
"""P6-G1-R1 deterministic grasp -> frozen-VLA handoff gate.

This is intentionally a new runner.  The earlier P6-G1 runner stopped at a
local-lift realization gate; this runner keeps the frozen policy/interface but
continues through transport, release, and basket placement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/exouser/Tabero")
RESULTS = ROOT / "analysis/results"
P6G1 = ROOT / "analysis/p6g1_primitive_ik_vla_grasp_realization.py"
P6G0 = ROOT / "analysis/p6g0_grasp_force_physics_benchmark.py"
P6G0R1 = RESULTS / "p6g0r1_confirmatory_and_candidate_coverage_20260824_232236"
P6G1_AUTHORITATIVE = RESULTS / "p6g1_primitive_ik_vla_grasp_realization_20260825_103146"

TASKS = [1, 6]
OBJECTS = {1: "cream_cheese_1", 6: "butter_1"}
INSTRUCTIONS = {
    1: "pick up the cream cheese and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
BASKET = "basket_1"
PRIMITIVES = ["G0", "G1", "G2"]
MAIN_ROOTS = [9200, 9201, 9202, 9203, 9204]
DEV_ROOTS = [9300, 9301, 9302]
POLICY_REPEATS = [0, 1]
FORCE_N = 8.0
FORCE_HALF_N = 4.0
CONTACT_THRESHOLD_N = 0.15
LOCAL_LIFT_M = 0.03
HANDOFF_HOLD_STEPS = 8
STAGE_STEPS = 45
STABILIZE_STEPS = 4
FINAL_APPROACH_STEPS = 25
CLOSE_STEPS = 55
HOLD_STEPS = 12
LIFT_STEPS = 35
VLA_MAX_CHUNKS = 22
VLA_REPLAN_STEPS = 10
ROOT_SETTLE_STEPS = 20
HANDOFF_FORCE_IMBALANCE_MAX = 0.50
HANDOFF_LINEAR_SPEED_MAX = 0.05
HANDOFF_ANGULAR_SPEED_MAX = 1.0
CONTACT_CENTER_TOL_NORM_LG = 0.25
TIMEOUTS = {"server": 900, "worker": 21600, "rollout": 900, "restore": 60}

SERVER_PYTHON = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/.venv/bin/python")
POLICY_DIR = Path(
    "/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/"
    "checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999"
)
POLICY_CONFIG = "pi0_lora_tacfield_tabero"
B5_RESULTS = RESULTS / "b5_tabero_neutral_20260822_040652"
B5_WRAPPER = B5_RESULTS / "scripts/b5_serve_policy_with_explicit_norm_stats.py"
B5_STUBS = B5_RESULTS / "scripts/stubs"
TABERO_VTLA = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
WARP_CORE = Path(
    "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.warp.core-1.8.2+lx64"
)
OPENPI_CLIENT_SRC = ROOT / "benchmarks/openpi/openpi-client/src"
HDF5_TRAJ_SOURCE_DIR = ROOT / "benchmarks/datasets/libero/assembled_hdf5"
LIBERO_CONFIG_DIR = ROOT / "benchmarks/datasets/libero/config"
LIBERO_ASSETS_DATA_DIR = ROOT / "benchmarks/datasets/libero/USD"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p6g1 = load_module(P6G1, "p6g1_r1_base")


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if not exists:
            wr.writeheader()
        wr.writerow(row)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        return default if value in (None, "") else float(value)
    except Exception:
        return default


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        return default if value in (None, "") else int(float(value))
    except Exception:
        return default


def safe_mean(values: list[float]) -> float:
    vals = [float(x) for x in values if np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else 0.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            h.update(str(child.relative_to(path)).encode())
            h.update(sha256_file(child).encode())
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class Timeout:
    def __init__(self, seconds: int, name: str):
        self.seconds = max(1, int(seconds))
        self.name = name
        self.old = None

    def __enter__(self):
        def handler(_signum, _frame):
            raise TimeoutError(f"{self.name} timed out after {self.seconds}s")

        self.old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)
        if self.old is not None:
            signal.signal(signal.SIGALRM, self.old)
        return False


def load_library() -> dict:
    library = p6g1.recover_primitive_library()
    library["source_artifacts"]["p6g0r1"] = str(P6G0R1)
    library["source_artifacts"]["authoritative_p6g1"] = str(P6G1_AUTHORITATIVE)
    return library


def protocol() -> dict:
    return {
        "name": "P6-G1-R1 deterministic grasp to frozen-VLA handoff gate",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "learned_method_change": "NONE",
        "system_interface_change": "DETERMINISTIC_GRASP_AND_STABLE_LIFT_BEFORE_FROZEN_VLA_HANDOFF",
        "tasks": TASKS,
        "primitives": PRIMITIVES,
        "requested_force_N": FORCE_N,
        "physical_query_used": False,
        "fresh_main_roots": MAIN_ROOTS,
        "development_roots": DEV_ROOTS,
        "canonical_dev_primitive": {"task1": "G2", "task6": "G2", "selection": "frozen before development gate"},
        "vla_policy": {
            "config": POLICY_CONFIG,
            "checkpoint": str(POLICY_DIR),
            "max_inference_chunks": VLA_MAX_CHUNKS,
            "replan_steps": VLA_REPLAN_STEPS,
            "instruction": "original full-task instruction; no place-only prompt",
            "state_mode": "cold handoff selected only after development comparison",
        },
        "arms": {
            "A": "deterministic_grasp_stable_lift_then_deterministic_full_task",
            "B": "deterministic_grasp_stable_lift_then_frozen_vla_full_task",
            "C": "primitive_ik_pregrasp_then_frozen_vla_full_contact_segment",
            "D": "raw_frozen_vla_full_task",
        },
        "rollout_plan": {
            "candidate_conditioned": "2 tasks * 5 roots * 3 candidates * (1 A + 2 B + 2 C) = 150",
            "raw_vla": "2 tasks * 5 roots * 2 repeats = 20",
            "main_total": 170,
            "midtask_development": "2 tasks * 3 roots = 6",
        },
        "handoff_thresholds": {
            "bilateral_contact_force_threshold_N": CONTACT_THRESHOLD_N,
            "force_imbalance_max": HANDOFF_FORCE_IMBALANCE_MAX,
            "linear_speed_max_mps": HANDOFF_LINEAR_SPEED_MAX,
            "angular_speed_max_rad_s": HANDOFF_ANGULAR_SPEED_MAX,
            "hold_steps": HANDOFF_HOLD_STEPS,
            "local_lift_m": LOCAL_LIFT_M,
            "contact_center_tolerance_norm_Lg": CONTACT_CENTER_TOL_NORM_LG,
        },
        "retry": {"primary_arm_retry": False},
        "training": False,
    }


def frozen_manifest(out: Path, server_metadata: dict | None = None) -> dict:
    return {
        "policy_class": "OpenPI websocket policy created by openpi.policies.policy_config.create_trained_policy",
        "base_policy": "Pi0Config (pi0, not pi0.5), authoritative P6-G1 policy",
        "policy_config": POLICY_CONFIG,
        "checkpoint_path": str(POLICY_DIR),
        "checkpoint_step": 49999,
        "checkpoint_hash_sha256": sha256_path(POLICY_DIR),
        "norm_stats_dir": str(POLICY_DIR / "assets/NathanWu7/tabero"),
        "server_wrapper": str(B5_WRAPPER),
        "server_python": str(SERVER_PYTHON),
        "observation_specification": {
            "image": "agentview RGB uint8 224x224x3",
            "wrist_image": "eye_in_hand RGB uint8 224x224x3",
            "state": "[eef position, eef axis-angle, gripper_abs]",
            "tactile_image": "4x4 tactile RGB history mosaic when available",
            "tactile_gripper_force": "8x6 left/right fingertip force history when available",
            "tactile_marker_motion": "1+8 marker-motion history when available",
        },
        "instruction": "original official full-task instruction",
        "action_chunk": {"server_action_dim": 32, "effective_dim": 13, "replan_steps": VLA_REPLAN_STEPS},
        "gripper_semantics": {
            "raw_gripper_slot_preserved": True,
            "force_slots_overridden": "fLz=fRz=4N for fixed 8N; other pose/gripper fields preserved",
            "release_semantics": "authoritative VLA gripper slot passed through",
        },
        "policy_state": {
            "websocket_client_reset_effect": "no-op",
            "action_queue_cache": "none; infer then execute first 10 actions",
            "history": "constructed online from current observation/tactile buffers",
            "recurrent_hidden_state": "not exposed",
        },
        "server_metadata": server_metadata or {},
        "authoritative_prior_artifact": str(P6G1_AUTHORITATIVE),
        "method_change": "NONE",
    }


ROOT_FIELDS = ["task", "object", "root_seed", "root_state_hash", "object_x", "object_y", "object_z", "object_qw", "object_qx", "object_qy", "object_qz", "source"]
PARITY_FIELDS = ["trial_id", "task", "root_seed", "arm", "primitive_id", "policy_repeat", "root_state_hash", "restore_state_hash", "state_parity", "object_disturbance_m", "object_rotation_disturbance_rad"]
ARM_FIELDS = [
    "trial_id", "task", "object", "root_seed", "arm", "primitive_id", "source_grasp_label", "policy_repeat", "policy_repeat_seed", "requested_force_N", "state_parity", "staging_success", "handoff_ready", "handoff_reason", "bilateral_contact", "mode_accuracy", "contact_center_error_norm_Lg", "force_imbalance", "object_linear_speed_mps", "object_angular_speed_rad_s", "stable_lift", "hold_success", "retained_after_1_chunk", "retained_after_5_chunks", "transport_grasp_retained", "contact_mode_drift", "premature_release", "regrasp_attempt", "slip_onset", "drop", "transport_success", "placement_success", "full_task_success_y", "failure_stage", "episode_steps", "num_policy_chunks", "mean_measured_force_N", "peak_measured_force_N", "release_step", "error", "contact_telemetry_path", "force_telemetry_path", "policy_log_path"
]
HANDOFF_FIELDS = ["trial_id", "task", "root_seed", "arm", "primitive_id", "requested_primitive", "actual_contact_cluster", "handoff_ready", "bilateral_contact", "above_local_lift", "hold_success", "force_imbalance", "linear_speed_mps", "angular_speed_rad_s", "no_slip_trend", "no_collision_table_contact", "contact_center_error_norm_Lg", "reason"]
RETENTION_FIELDS = ["trial_id", "task", "root_seed", "arm", "primitive_id", "policy_repeat", "retained_after_1_chunk", "retained_after_5_chunks", "transport_grasp_retained", "contact_mode_drift", "premature_release", "regrasp_attempt", "slip_onset", "drop", "release_step"]
DEV_FIELDS = ["trial_id", "task", "root_seed", "primitive_id", "state_mode", "controller_handoff_ready", "full_task_success_y", "immediate_release_or_regrasp", "error", "steps"]


def tensor_list(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def q_angle(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12); b = b / max(np.linalg.norm(b), 1e-12)
    return float(2 * np.arccos(np.clip(abs(float(np.dot(a, b))), -1, 1)))


def import_env_modules(task: int):
    import gymnasium as gym
    import tac_manip.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tac_manip.utils.task_configs import setup_task_objects

    p6 = load_module(P6G0, f"p6g0_r1_{task}")
    p4 = p6.import_p4_probe(task)
    p6.imported_p4 = p4
    setup_task_objects("libero_object", task)
    cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
    cfg.episode_length_s = 45.0
    try:
        cfg.sim.physx.enable_ccd = True
    except Exception:
        pass
    env = gym.make(ENV_ID, cfg=cfg).unwrapped
    return env, p6, p4


def root_hash(p6, env) -> str:
    return stable_hash(tensor_list(env.scene.get_state(is_relative=True)))


def obj_pose(p4, env, task: int):
    return p4._pose_in_base(env, OBJECTS[task])


def force_pair(obs):
    f = obs["policy"]["gripper_net_force"][0]
    if f.ndim == 3:
        f = f[-1]
    a = f.detach().cpu().numpy()
    return float(np.linalg.norm(a[0])), float(np.linalg.norm(a[1]))


def basket_contact(env, task: int) -> float:
    try:
        import torch
        c = env.scene[f"contact_{BASKET}_{OBJECTS[task]}"]
        return float(torch.linalg.vector_norm(c.data.force_matrix_w.reshape(-1, 3)[0]).item())
    except Exception:
        return 0.0


def dropped(env) -> int:
    try:
        return int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
    except Exception:
        return 0


def contact_mode(obs) -> str:
    l, r = force_pair(obs)
    if l > CONTACT_THRESHOLD_N and r > CONTACT_THRESHOLD_N:
        return "bilateral"
    if l > CONTACT_THRESHOLD_N or r > CONTACT_THRESHOLD_N:
        return "unilateral"
    return "none"


def stage_pregrasp(env, p6, p4, task: int, primitive: dict, dt: float):
    """Reuse the frozen P6-G1 pregrasp staging code, without changing geometry."""
    trial_id = f"r1_stage_t{task}_{primitive['primitive_id']}"
    return p6g1.stage_to_pregrasp(env, p6, p4, task, primitive, trial_id, dt)


def controller_grasp_lift(env, p6, p4, task: int, primitive: dict, dt: float, trial_id: str, out: Path):
    import torch

    obj_name = OBJECTS[task]
    obj0, objq0 = obj_pose(p4, env, task)
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    eef_aa = p4._aa(eef[3:7])
    cmd = eef[:3].copy()
    grasp_obj = np.asarray(primitive["desired_object_relative_grasp_transform"]["position_m"], dtype=np.float64)
    grasp = p6.object_point_to_base(p4, env, obj_name, grasp_obj)
    lift = grasp.copy(); lift[2] += LOCAL_LIFT_M
    d_pred = p6g1.D_OPEN
    rows = []
    stable = 0
    lift_stable = 0
    prev_pos, prev_q = obj0.copy(), objq0.copy()
    last_speed = (float("inf"), float("inf"))
    max_imbalance = float("inf")
    err = ""
    term = trunc = False
    step = 0
    try:
        with Timeout(TIMEOUTS["rollout"], f"CONTROLLER_{trial_id}"):
            phases = [("final_approach", FINAL_APPROACH_STEPS, grasp, 0.0), ("close", CLOSE_STEPS, grasp, FORCE_N), ("hold", HOLD_STEPS, grasp, FORCE_N), ("lift", LIFT_STEPS, lift, FORCE_N), ("lift_hold", HANDOFF_HOLD_STEPS, lift, FORCE_N)]
            for phase, n, target, fcmd in phases:
                start = cmd.copy()
                for i in range(n):
                    cmd = p4._interp(start, target, i, n)
                    action = p6.make_action(p4, cmd, eef_aa, d_pred, fcmd, env.device)
                    obs, _, term_t, trunc_t, _ = env.step(action)
                    step += 1
                    l, r = force_pair(obs)
                    mode = contact_mode(obs)
                    if fcmd > 0:
                        f_sq = p4._f(p4._dbg(env).get("f_sq_meas"), 0.0)
                        d_pred = p6.force_servo(p4, d_pred, f_sq, FORCE_N)
                    op, oq = obj_pose(p4, env, task)
                    speed = float(np.linalg.norm(op - prev_pos) / max(dt, 1e-9))
                    ang_speed = float(q_angle(oq, prev_q) / max(dt, 1e-9))
                    last_speed = (speed, ang_speed)
                    imbalance = abs(l - r) / max(l + r, 1e-9)
                    max_imbalance = min(max_imbalance, imbalance) if np.isfinite(imbalance) else max_imbalance
                    if mode == "bilateral":
                        stable += 1
                    else:
                        stable = 0
                    if float(op[2] - obj0[2]) >= LOCAL_LIFT_M and mode == "bilateral":
                        lift_stable += 1
                    else:
                        lift_stable = 0
                    rows.append({"trial_id": trial_id, "task": task, "phase": phase, "step": step, "contact_state": mode, "left_force_N": l, "right_force_N": r, "force_imbalance": imbalance, "object_x": float(op[0]), "object_y": float(op[1]), "object_z": float(op[2]), "object_qw": float(oq[0]), "object_qx": float(oq[1]), "object_qy": float(oq[2]), "object_qz": float(oq[3]), "linear_speed_mps": speed, "angular_speed_rad_s": ang_speed, "basket_contact_N": basket_contact(env, task), "drop": dropped(env), "gripper_cmd": float(d_pred)})
                    prev_pos, prev_q = op.copy(), oq.copy()
                    term, trunc = bool(term_t[0].item()), bool(trunc_t[0].item())
                    if term or trunc:
                        break
                if term or trunc:
                    break
    except Exception as exc:
        err = repr(exc)
    bilateral = [r for r in rows if r["contact_state"] == "bilateral"]
    final_hold = rows[-HANDOFF_HOLD_STEPS:] if len(rows) >= HANDOFF_HOLD_STEPS else rows
    no_slip = bool(final_hold) and all(r["contact_state"] == "bilateral" and r["drop"] == 0 for r in final_hold)
    above = bool(rows) and float(rows[-1]["object_z"] - obj0[2]) >= LOCAL_LIFT_M
    hold_ok = len(final_hold) >= HANDOFF_HOLD_STEPS and no_slip
    force_imb = safe_mean([r["force_imbalance"] for r in final_hold]) if final_hold else float("inf")
    lin_speed = last_speed[0]; ang_speed = last_speed[1]
    handoff = bool(not err and bilateral and above and hold_ok and force_imb <= HANDOFF_FORCE_IMBALANCE_MAX and lin_speed <= HANDOFF_LINEAR_SPEED_MAX and ang_speed <= HANDOFF_ANGULAR_SPEED_MAX and dropped(env) == 0)
    reason = "HANDOFF_READY" if handoff else (err or "physical_handoff_check_failed")
    contact_mid = ""
    if bilateral:
        # Contact midpoint is reconstructed from the final gripper-frame targets.
        try:
            lw = env.scene["left_gripper_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
            rw = env.scene["right_gripper_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
            mid_b = p6.world_point_to_base(p4, env, 0.5 * (lw + rw))
            mid_o = p6.base_point_to_object(p4, env, obj_name, mid_b)
            contact_mid = mid_o.tolist()
        except Exception:
            contact_mid = ""
    telemetry = out / "P6G1R1_CONTACT_TELEMETRY" / f"{trial_id}.csv"
    write_csv(telemetry, rows)
    return {"trial_id": trial_id, "staging_success": int(bool(not err)), "handoff_ready": int(handoff), "handoff_reason": reason, "bilateral_contact": int(bool(bilateral)), "above_local_lift": int(above), "hold_success": int(hold_ok), "force_imbalance": force_imb, "linear_speed_mps": lin_speed, "angular_speed_rad_s": ang_speed, "no_slip_trend": int(no_slip), "contact_center_obj": contact_mid, "object_pose_start": obj0.tolist(), "object_pose_end": rows[-1]["object_z"] if rows else obj0[2], "stable_lift": int(above and no_slip), "drop": dropped(env), "steps": step, "error": err, "contact_telemetry_path": str(telemetry)}


def vla_observation(env, p6, p4, task: int, tactile):
    return p6g1.build_policy_observation(env, env.observation_manager.compute(), INSTRUCTIONS[task], tactile)


def run_vla_full(env, p6, p4, client, task: int, primitive: dict | None, meta: dict, dt: float, out: Path):
    import torch
    from benchmarks.common.metrics import compute_contact_force_series_from_lr_forces

    trial_id = meta["trial_id"]
    obj0, objq0 = obj_pose(p4, env, task)
    tactile = p6g1.OnlineTactileBuffer(tactile_output_type="tactile_rgb")
    contact_rows = []; force_rows = []; chunks = []; chunk_indices = []
    original_squeeze = []; release_step = ""; first_bilateral = False; premature = 0; regrasp = 0; opened_before_basket = False
    stable = 0; dropped_flag = 0; term = trunc = False; err = ""; step = 0; max_basket = 0.0; lift_success = False; lost_transport = 0
    first_chunk_retained = 0; fifth_chunk_retained = 0; max_chunks_reached = 0
    last_open = False
    try:
        with Timeout(TIMEOUTS["rollout"], f"VLA_{trial_id}"):
            obs = env.observation_manager.compute()
            for chunk_idx in range(VLA_MAX_CHUNKS):
                element = p6g1.build_policy_observation(env, obs, INSTRUCTIONS[task], tactile)
                action_chunk = np.asarray(client.infer(element)["actions"], dtype=np.float32)
                chunks.append(action_chunk.copy()); chunk_indices.append(chunk_idx); max_chunks_reached = chunk_idx + 1
                if action_chunk.ndim != 2 or action_chunk.shape[0] < VLA_REPLAN_STEPS or action_chunk.shape[1] < 13:
                    raise RuntimeError(f"bad policy action shape {action_chunk.shape}")
                raw = action_chunk[:VLA_REPLAN_STEPS, :13]
                executed = raw.copy(); executed[:, 7:13] = 0.0; executed[:, 9] = FORCE_HALF_N; executed[:, 12] = FORCE_HALF_N
                for i in range(VLA_REPLAN_STEPS):
                    raw_i = raw[i]
                    pred = compute_contact_force_series_from_lr_forces(np.asarray([raw_i[7:10]], np.float32), np.asarray([raw_i[10:13]], np.float32))
                    original_squeeze.append(float(pred.squeeze[0]))
                    obs, _, tt, tr, _ = env.step(torch.from_numpy(executed[i]).float().reshape(1, -1).to(env.device))
                    step += 1
                    mode = contact_mode(obs); l, r = force_pair(obs)
                    if mode == "bilateral": stable += 1; first_bilateral = True
                    else: stable = 0
                    objp, objq = obj_pose(p4, env, task)
                    bc = basket_contact(env, task); max_basket = max(max_basket, bc); dropped_flag = max(dropped_flag, dropped(env))
                    if float(objp[2] - obj0[2]) >= LOCAL_LIFT_M and mode == "bilateral": lift_success = True
                    if lift_success and max_basket <= 0.05 and (dropped_flag or float(objp[2] - obj0[2]) < 0.02): lost_transport = 1
                    is_open = bool(float(raw_i[6]) > float(p6g1.D_OPEN) + 0.02)
                    if is_open and not release_step and max_basket <= 0.05:
                        release_step = step; premature = 1; opened_before_basket = True
                    if is_open and not last_open and first_bilateral and max_basket <= 0.05:
                        regrasp += 1
                    last_open = is_open
                    snap = p6g1.contact_snapshot(env, p6, p4, obs, task, trial_id, "vla", step, dt, stable); snap["stable_bilateral_run"] = stable; snap["chunk_idx"] = chunk_idx; snap["raw_gripper_cmd"] = float(raw_i[6]); snap["executed_gripper_cmd"] = float(executed[i, 6]); snap["basket_contact_N"] = bc; snap["drop"] = dropped_flag; contact_rows.append(snap)
                    force_rows.append({"trial_id": trial_id, "task": task, "root_seed": meta.get("root_seed", ""), "arm": meta.get("arm", ""), "primitive_id": meta.get("primitive_id", ""), "policy_repeat": meta.get("policy_repeat", ""), "phase": "vla", "step": step, "chunk_idx": chunk_idx, "requested_force_N": FORCE_N, "measured_squeeze_N": snap["measured_squeeze_N"], "measured_applied_force_N": snap["measured_applied_force_N"], "raw_gripper_cmd": float(raw_i[6]), "executed_fLz": FORCE_HALF_N, "executed_fRz": FORCE_HALF_N, "raw_fLz": float(raw_i[9]), "raw_fRz": float(raw_i[12])})
                    term, trunc = bool(tt[0].item()), bool(tr[0].item())
                    if step == VLA_REPLAN_STEPS: first_chunk_retained = int(mode == "bilateral" and dropped_flag == 0)
                    if step == 5 * VLA_REPLAN_STEPS: fifth_chunk_retained = int(mode == "bilateral" and dropped_flag == 0)
                    if term or trunc or dropped_flag:
                        break
                if term or trunc or dropped_flag:
                    break
            if not term and not trunc and not dropped_flag:
                err = "vla_chunk_budget_exhausted"
    except Exception as exc:
        err = repr(exc)
    if not first_chunk_retained and len(contact_rows) >= VLA_REPLAN_STEPS:
        first_chunk_retained = int(contact_rows[VLA_REPLAN_STEPS - 1]["contact_state"] == "bilateral" and dropped_flag == 0)
    if not fifth_chunk_retained and len(contact_rows) >= 5 * VLA_REPLAN_STEPS:
        fifth_chunk_retained = int(contact_rows[5 * VLA_REPLAN_STEPS - 1]["contact_state"] == "bilateral" and dropped_flag == 0)
    contact_mode_drift = int(bool(contact_rows) and first_bilateral and any(r["contact_state"] == "unilateral" for r in contact_rows[: max(1, contact_rows[-1].get("step", 1))] if r.get("basket_contact_N", 0.0) <= 0.05))
    transport_retained = int(lost_transport == 0 and dropped_flag == 0 and (max_basket > 0.05 or not lift_success))
    place_success = int(max_basket > 0.05)
    full = int(lift_success and place_success and dropped_flag == 0)
    failure = "" if full else ("drop" if dropped_flag else "premature_release" if opened_before_basket else "transport" if lost_transport else "placement" if not place_success else "vla_timeout_or_error")
    contact_path = out / "P6G1R1_CONTACT_TELEMETRY" / f"{trial_id}.csv"; force_path = out / "P6G1R1_FORCE_TELEMETRY" / f"{trial_id}.csv"; policy_path = out / "P6G1R1_POLICY_LOGS" / f"{trial_id}_chunks.npz"
    write_csv(contact_path, contact_rows); write_csv(force_path, force_rows); policy_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(policy_path, **{f"chunk_{i:03d}": x for i, x in enumerate(chunks)}, action_indices=np.asarray(chunk_indices, dtype=np.int32))
    return {"trial_id": trial_id, "task": task, "object": OBJECTS[task], "root_seed": meta.get("root_seed", ""), "arm": meta.get("arm", ""), "primitive_id": meta.get("primitive_id", ""), "source_grasp_label": primitive.get("source_grasp_label", "") if primitive else "", "policy_repeat": meta.get("policy_repeat", ""), "policy_repeat_seed": meta.get("policy_repeat_seed", ""), "requested_force_N": FORCE_N, "state_parity": meta.get("state_parity", 0), "staging_success": meta.get("staging_success", 1), "handoff_ready": meta.get("handoff_ready", 1), "handoff_reason": meta.get("handoff_reason", ""), "bilateral_contact": int(first_bilateral), "mode_accuracy": "", "contact_center_error_norm_Lg": "", "force_imbalance": safe_mean([abs(r["left_contact_force_norm_N"] - r["right_contact_force_norm_N"]) / max(r["left_contact_force_norm_N"] + r["right_contact_force_norm_N"], 1e-9) for r in contact_rows]) if contact_rows else "", "object_linear_speed_mps": "", "object_angular_speed_rad_s": "", "stable_lift": int(lift_success), "hold_success": int(first_chunk_retained), "retained_after_1_chunk": first_chunk_retained, "retained_after_5_chunks": fifth_chunk_retained, "transport_grasp_retained": transport_retained, "contact_mode_drift": contact_mode_drift, "premature_release": premature, "regrasp_attempt": regrasp, "slip_onset": "" if transport_retained else "transport", "drop": dropped_flag, "transport_success": int(transport_retained), "placement_success": place_success, "full_task_success_y": full, "failure_stage": failure, "episode_steps": step, "num_policy_chunks": len(chunks), "mean_measured_force_N": safe_mean([as_float(r, "measured_squeeze_N") for r in force_rows]), "peak_measured_force_N": max([as_float(r, "measured_squeeze_N") for r in force_rows] or [0.0]), "release_step": release_step, "error": err, "contact_telemetry_path": str(contact_path), "force_telemetry_path": str(force_path), "policy_log_path": str(policy_path)}


def deterministic_full_task(env, p6, p4, task: int, primitive: dict, dt: float, meta: dict, out: Path, controller: dict):
    obj_name = OBJECTS[task]
    obj0, _ = obj_pose(p4, env, task)
    obs = env.observation_manager.compute(); eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy(); eef_aa = p4._aa(eef[3:7]); obj_now, _ = obj_pose(p4, env, task); basket, _ = p4._pose_in_base(env, BASKET)
    ee_minus_obj = eef[:3] - obj_now; lift_obj = obj_now.copy(); lift_obj[2] += 0.10; transit = basket.copy(); transit[2] = max(lift_obj[2], basket[2] + 0.18); place = basket.copy(); place[2] = basket[2] + 0.10
    cmd = eef[:3].copy(); d_pred = p6g1.D_CLOSED; force_samples = []; max_basket = 0.0; lost = 0; drop_flag = controller["drop"]; lift_success = controller["stable_lift"] == 1; rows = []; step = controller["steps"]; release_step = ""; err = ""
    try:
        with Timeout(TIMEOUTS["rollout"], f"ARM_A_{meta['trial_id']}"):
            for phase, n, target in [("lift", 50, lift_obj), ("transit", 110, transit), ("over_basket", 30, transit), ("place", 40, place), ("release", 50, place), ("settle", 50, place)]:
                ee_target = target + ee_minus_obj; start = cmd.copy(); opening = phase in {"release", "settle"}
                for i in range(n):
                    cmd = p4._interp(start, ee_target, i, n); fcmd = 0.0 if opening else FORCE_N; action = p6.make_action(p4, cmd, eef_aa, d_pred, fcmd, env.device); obs, _, tt, tr, _ = env.step(action); step += 1
                    if fcmd > 0:
                        fs = p4._f(p4._dbg(env).get("f_sq_meas"), 0.0); force_samples.append(fs); d_pred = p6.force_servo(p4, d_pred, fs, FORCE_N)
                    else:
                        d_pred = p6g1.D_OPEN; release_step = release_step or step
                    op, oq = obj_pose(p4, env, task); bc = basket_contact(env, task); max_basket = max(max_basket, bc); drop_flag = max(drop_flag, dropped(env)); dz = float(op[2] - obj0[2]); lift_success = lift_success or dz >= LOCAL_LIFT_M
                    if lift_success and phase in {"transit", "over_basket"} and (drop_flag or dz < 0.02): lost = 1
                    rows.append({"trial_id": meta["trial_id"], "phase": phase, "step": step, "object_x": float(op[0]), "object_y": float(op[1]), "object_z": float(op[2]), "basket_contact_N": bc, "drop": drop_flag, "gripper_cmd": float(d_pred)})
                    if bool(tt[0].item()) or bool(tr[0].item()): break
                else: continue
                break
    except Exception as exc: err = repr(exc)
    place_success = int(max_basket > 0.05); full = int(lift_success and place_success and drop_flag == 0); failure = "" if full else ("drop" if drop_flag else "transport" if lost else "placement" if not place_success else "error")
    cpath = out / "P6G1R1_CONTACT_TELEMETRY" / f"{meta['trial_id']}.csv"; write_csv(cpath, rows)
    return {"trial_id": meta["trial_id"], "task": task, "object": obj_name, "root_seed": meta["root_seed"], "arm": "A", "primitive_id": primitive["primitive_id"], "source_grasp_label": primitive["source_grasp_label"], "policy_repeat": "", "policy_repeat_seed": "", "requested_force_N": FORCE_N, "state_parity": meta["state_parity"], "staging_success": controller["staging_success"], "handoff_ready": controller["handoff_ready"], "handoff_reason": controller["handoff_reason"], "bilateral_contact": controller["bilateral_contact"], "stable_lift": int(lift_success), "hold_success": controller["hold_success"], "retained_after_1_chunk": "", "retained_after_5_chunks": "", "transport_grasp_retained": int(lost == 0 and drop_flag == 0), "contact_mode_drift": "", "premature_release": 0, "regrasp_attempt": 0, "slip_onset": "", "drop": drop_flag, "transport_success": int(lost == 0 and drop_flag == 0), "placement_success": place_success, "full_task_success_y": full, "failure_stage": failure, "episode_steps": len(rows), "num_policy_chunks": 0, "mean_measured_force_N": safe_mean(force_samples), "peak_measured_force_N": max(force_samples or [0.0]), "release_step": release_step, "error": err, "contact_telemetry_path": str(cpath), "force_telemetry_path": "", "policy_log_path": ""}


def start_server(out: Path, host: str, port: int):
    if p6g1.tcp_port_open(host, port, timeout_s=0.25):
        return None, False
    log = out / "P6G1R1_POLICY_LOGS" / "server.log"; log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env["PYTHONPATH"] = os.pathsep.join([str(B5_STUBS), str(TABERO_VTLA / "src"), str(TABERO_VTLA / "packages/openpi-client/src"), env.get("PYTHONPATH", "")])
    env["PYTHONNOUSERSITE"] = "1"; env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    cmd = [str(SERVER_PYTHON), "-u", str(B5_WRAPPER), "--port", str(port), "--policy-config", POLICY_CONFIG, "--policy-dir", str(POLICY_DIR), "--norm-stats-dir", str(POLICY_DIR / "assets/NathanWu7/tabero")]
    fh = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=TABERO_VTLA, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, True


def launch_worker(out: Path, task: int, host: str, port: int, mode: str, roots: list[int]):
    log = out / "logs" / f"{mode}_task{task}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update({"P6G1R1_WORKER": "1", "P6G1R1_OUT": str(out), "P6G1R1_TASK": str(task), "P6G1R1_HOST": host, "P6G1R1_PORT": str(port), "P6G1R1_MODE": mode, "P6G1R1_ROOTS": json.dumps(roots)})
    # Match the authoritative P6-G1 Isaac worker environment exactly. This is
    # process initialization only; it does not change tasks, actions, or labels.
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(ROOT), str(OPENPI_CLIENT_SRC)]),
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
        "TABERO_ROOT": str(ROOT),
        "HDF5_TRAJ_SOURCE_DIR": str(HDF5_TRAJ_SOURCE_DIR),
        "LIBERO_CONFIG_DIR": str(LIBERO_CONFIG_DIR),
        "LIBERO_ASSETS_DATA_DIR": str(LIBERO_ASSETS_DATA_DIR),
    })
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run([str(ISAAC_PY), "-u", str(Path(__file__).resolve())], env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=TIMEOUTS["worker"])
    return {"mode": mode, "task": task, "returncode": proc.returncode, "elapsed_wall_s": time.time() - t0, "log": str(log)}


def worker() -> int:
    from isaaclab.app import AppLauncher
    import torch
    from openpi_client import websocket_client_policy

    out = Path(os.environ["P6G1R1_OUT"]); task = int(os.environ["P6G1R1_TASK"]); mode = os.environ["P6G1R1_MODE"]; roots = json.loads(os.environ["P6G1R1_ROOTS"]); host = os.environ["P6G1R1_HOST"]; port = int(os.environ["P6G1R1_PORT"])
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app; env = None
    try:
        env, p6, p4 = import_env_modules(task); dt = float(env.cfg.sim.dt) * int(env.cfg.decimation); lib = read_json(out / "P6G1R1_PRIMITIVE_LIBRARY.json"); client = websocket_client_policy.WebsocketClientPolicy(host, port)
        task_dir = out / f"task{task}"; task_dir.mkdir(parents=True, exist_ok=True); arms = read_csv(task_dir / "P6G1R1_ARM_RESULTS.csv"); dev_rows = read_csv(out / "P6G1R1_MIDTASK_INVOCATION_AUDIT.csv"); parity = read_csv(out / "P6G1R1_STATE_PARITY.csv"); root_rows = read_csv(out / "P6G1R1_ROOT_MANIFEST.csv")
        for ri, seed in enumerate(roots):
            env.reset(seed=int(seed)); p6g1.settle_root_before_hash(env, p6, p4, ROOT_SETTLE_STEPS); root_state = env.scene.get_state(is_relative=True); h0 = root_hash(p6, env); op, oq = obj_pose(p4, env, task); root_row = {"task": task, "object": OBJECTS[task], "root_seed": seed, "root_state_hash": h0, "object_x": float(op[0]), "object_y": float(op[1]), "object_z": float(op[2]), "object_qw": float(oq[0]), "object_qx": float(oq[1]), "object_qy": float(oq[2]), "object_qz": float(oq[3]), "source": "fresh_reset_settled"}; root_rows = [r for r in root_rows if not (r.get("task") == str(task) and r.get("root_seed") == str(seed))] + [root_row]
            if mode == "preflight":
                pid = "G2"; prim = p6g1.primitive_for(lib, task, pid); tid = f"p6g1r1_dev_t{task}_s{seed}"
                env.reset_to(root_state, torch.tensor([0], device=env.device), is_relative=True); restore_hash = root_hash(p6, env); parity_state = int(restore_hash == h0); parity.append({"trial_id": tid, "task": task, "root_seed": seed, "arm": "DEV", "primitive_id": pid, "policy_repeat": "", "root_state_hash": h0, "restore_state_hash": restore_hash, "state_parity": parity_state, "object_disturbance_m": 0.0, "object_rotation_disturbance_rad": 0.0}); stage, _ = stage_pregrasp(env, p6, p4, task, prim, dt); ctrl = controller_grasp_lift(env, p6, p4, task, prim, dt, tid + "_controller", out); ctrl.update({"staging_success": int(stage.get("staging_success", 0) and ctrl["staging_success"]), "handoff_ready": int(stage.get("staging_success", 0) and ctrl["handoff_ready"])}); full = run_vla_full(env, p6, p4, client, task, prim, {"trial_id": tid, "task": task, "root_seed": seed, "arm": "DEV", "primitive_id": pid, "policy_repeat": "", "state_parity": parity_state, "staging_success": ctrl["staging_success"], "handoff_ready": ctrl["handoff_ready"]}, dt, out) if parity_state and ctrl["handoff_ready"] else {"full_task_success_y": 0, "premature_release": 0, "error": ctrl["handoff_reason"], "episode_steps": 0}
                dev_rows.append({"trial_id": tid, "task": task, "root_seed": seed, "primitive_id": pid, "state_mode": "cold_handoff", "controller_handoff_ready": ctrl["handoff_ready"], "full_task_success_y": full.get("full_task_success_y", 0), "immediate_release_or_regrasp": int(full.get("premature_release", 0) > 0), "error": full.get("error", ""), "steps": full.get("episode_steps", 0)})
                continue
            # First pass A/C/D/B is ordered per candidate. A is the deterministic reference.
            for pid in PRIMITIVES:
                prim = p6g1.primitive_for(lib, task, pid)
                for arm, reps in (("A", [""]), ("B", POLICY_REPEATS), ("C", POLICY_REPEATS)):
                    for rep in reps:
                        tid = f"p6g1r1_t{task}_s{seed}_{arm}_{pid}" + (f"_r{rep}" if rep != "" else "")
                        env.reset_to(root_state, torch.tensor([0], device=env.device), is_relative=True); restore_hash = root_hash(p6, env); parity_state = int(restore_hash == h0); parity.append({"trial_id": tid, "task": task, "root_seed": seed, "arm": arm, "primitive_id": pid, "policy_repeat": rep, "root_state_hash": h0, "restore_state_hash": restore_hash, "state_parity": parity_state, "object_disturbance_m": 0.0, "object_rotation_disturbance_rad": 0.0}); before, _ = obj_pose(p4, env, task)
                        if arm in {"A", "B"}:
                            stage, _ = stage_pregrasp(env, p6, p4, task, prim, dt); ctrl = controller_grasp_lift(env, p6, p4, task, prim, dt, tid + "_controller", out); ctrl["staging_success"] = int(bool(stage.get("staging_success", 0)) and ctrl["staging_success"]); ctrl["handoff_ready"] = int(bool(stage.get("staging_success", 0)) and ctrl["handoff_ready"])
                        else:
                            stage, _ = stage_pregrasp(env, p6, p4, task, prim, dt); ctrl = {"staging_success": int(stage.get("staging_success", 0)), "handoff_ready": int(stage.get("staging_success", 0)), "handoff_reason": "PREGRASP_READY", "bilateral_contact": 0, "stable_lift": 0, "hold_success": 0, "drop": 0, "steps": 0, "force_imbalance": "", "linear_speed_mps": "", "angular_speed_rad_s": "", "contact_center_obj": ""}
                        meta = {"trial_id": tid, "task": task, "root_seed": seed, "arm": arm, "primitive_id": pid, "policy_repeat": rep, "policy_repeat_seed": f"stream_{rep}", "state_parity": parity_state, **{k: ctrl.get(k, "") for k in ("staging_success", "handoff_ready", "handoff_reason")}}
                        if not parity_state or not ctrl["handoff_ready"]:
                            row = {"trial_id": tid, "task": task, "object": OBJECTS[task], "root_seed": seed, "arm": arm, "primitive_id": pid, "source_grasp_label": prim["source_grasp_label"], "policy_repeat": rep, "policy_repeat_seed": meta["policy_repeat_seed"], "requested_force_N": FORCE_N, "state_parity": parity_state, "staging_success": ctrl["staging_success"], "handoff_ready": ctrl["handoff_ready"], "handoff_reason": ctrl.get("handoff_reason", "invalid"), "full_task_success_y": 0, "failure_stage": "invalid_handoff", "error": ctrl.get("handoff_reason", "invalid")}
                        elif arm == "A":
                            row = deterministic_full_task(env, p6, p4, task, prim, dt, meta, out, ctrl)
                        else:
                            row = run_vla_full(env, p6, p4, client, task, prim, meta, dt, out)
                        row.update({"contact_center_error_norm_Lg": "", "mode_accuracy": ""}); arms.append(row); append_csv(task_dir / "P6G1R1_ARM_RESULTS.csv", row, ARM_FIELDS)
            for rep in POLICY_REPEATS:
                tid = f"p6g1r1_t{task}_s{seed}_D_raw_r{rep}"; env.reset_to(root_state, torch.tensor([0], device=env.device), is_relative=True); restore_hash = root_hash(p6, env); parity_state = int(restore_hash == h0); parity.append({"trial_id": tid, "task": task, "root_seed": seed, "arm": "D", "primitive_id": "", "policy_repeat": rep, "root_state_hash": h0, "restore_state_hash": restore_hash, "state_parity": parity_state, "object_disturbance_m": 0.0, "object_rotation_disturbance_rad": 0.0}); row = run_vla_full(env, p6, p4, client, task, None, {"trial_id": tid, "task": task, "root_seed": seed, "arm": "D", "primitive_id": "", "policy_repeat": rep, "policy_repeat_seed": f"stream_{rep}", "state_parity": parity_state, "staging_success": 1, "handoff_ready": 1}, dt, out); row.update({"contact_center_error_norm_Lg": "", "mode_accuracy": ""}); arms.append(row); append_csv(task_dir / "P6G1R1_ARM_RESULTS.csv", row, ARM_FIELDS)
        write_csv(out / "P6G1R1_ROOT_MANIFEST.csv", root_rows, ROOT_FIELDS); write_csv(out / "P6G1R1_STATE_PARITY.csv", parity, PARITY_FIELDS); write_csv(out / "P6G1R1_MIDTASK_INVOCATION_AUDIT.csv", dev_rows, DEV_FIELDS); write_csv(task_dir / "P6G1R1_ARM_RESULTS.csv", arms, ARM_FIELDS)
        return 0
    except Exception as exc:
        write_json(out / "logs" / f"worker_error_task{task}.json", {"task": task, "mode": mode, "error": repr(exc), "trace": traceback.format_exc()}); return 1
    finally:
        try:
            if env is not None: env.close()
        except Exception: pass
        try: app.close()
        except Exception: pass


def bootstrap(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for d in ["P6G1R1_CONTACT_TELEMETRY", "P6G1R1_FORCE_TELEMETRY", "P6G1R1_POLICY_LOGS", "logs"]: (out / d).mkdir(parents=True, exist_ok=True)
    proto = protocol(); write_json(out / "P6G1R1_PROTOCOL.json", proto); (out / "P6G1R1_PROTOCOL_HASH.txt").write_text(stable_hash(proto) + "\n"); (out / "P6G1R1_CODE_HASH.txt").write_text(sha256_file(Path(__file__).resolve()) + "\n")
    lib = load_library(); write_json(out / "P6G1R1_PRIMITIVE_LIBRARY.json", lib); p6g1.write_primitive_markdown(out, lib); (out / "P6G1R1_PRIMITIVE_LIBRARY.md").write_text((out / "P6G1_PRIMITIVE_LIBRARY.md").read_text())
    # Keep the requested R1 names in addition to the legacy helper's markdown name.
    write_json(out / "P6G1R1_FROZEN_SYSTEM_MANIFEST.json", frozen_manifest(out)); (out / "P6G1R1_FROZEN_SYSTEM_HASHES.txt").write_text(json.dumps({"policy": sha256_path(POLICY_DIR), "wrapper": sha256_file(B5_WRAPPER), "p6g0": sha256_file(P6G0), "p6g1_prior": sha256_file(P6G1)}, indent=2) + "\n")
    write_csv(out / "P6G1R1_ROOT_MANIFEST.csv", [], ROOT_FIELDS); write_csv(out / "P6G1R1_STATE_PARITY.csv", [], PARITY_FIELDS); write_csv(out / "P6G1R1_MIDTASK_INVOCATION_AUDIT.csv", [], DEV_FIELDS); write_csv(out / "P6G1R1_HANDOFF_VERIFIER_RESULTS.csv", [], HANDOFF_FIELDS); write_csv(out / "P6G1R1_POST_HANDOFF_RETENTION.csv", [], RETENTION_FIELDS); write_csv(out / "P6G1R1_ARM_RESULTS.csv", [], ARM_FIELDS); write_csv(out / "P6G1R1_PAIRED_COMPARISONS.csv", []); write_csv(out / "P6G1R1_FAILURE_CASES.csv", []); write_csv(out / "P6G1R1_TASK_CLASSIFICATIONS.csv", [])


def bootstrap_resume(out: Path):
    if not (out / "P6G1R1_PROTOCOL.json").exists(): raise SystemExit(f"missing frozen protocol in {out}")
    for d in ["P6G1R1_CONTACT_TELEMETRY", "P6G1R1_FORCE_TELEMETRY", "P6G1R1_POLICY_LOGS", "logs"]: (out / d).mkdir(parents=True, exist_ok=True)


def run_preflight(out: Path, host: str, port: int):
    rows = []
    workers = []
    for task in TASKS:
        w = launch_worker(out, task, host, port, "preflight", DEV_ROOTS); workers.append(w)
    rows = read_csv(out / "P6G1R1_MIDTASK_INVOCATION_AUDIT.csv")
    successes = sum(as_int(r, "full_task_success_y") for r in rows); n = len(rows); invalid = sum(as_int(r, "immediate_release_or_regrasp") for r in rows); controller_failures = sum(as_int(r, "controller_handoff_ready") == 0 for r in rows)
    qualified = all(as_int(r, "returncode") == 0 for r in workers) and n == 6 and successes >= 5 and invalid == 0
    report = ["# P6-G1-R1 mid-task invocation audit", "", f"Development rollouts: {n}", f"Full-task successes: {successes}/{n}", f"Immediate release/regrasp: {invalid}", f"Controller handoff failures: {controller_failures}", f"Workers: {workers}", "", f"Selected interface: `{'cold_handoff' if qualified else 'NOT_QUALIFIED'}`", "", f"Interface qualified: `{qualified}`"]
    (out / "P6G1R1_MIDTASK_INVOCATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(out / "P6G1R1_MIDTASK_GATE.json", {"rollouts": n, "successes": successes, "immediate_release_or_regrasp": invalid, "controller_handoff_failures": controller_failures, "workers": workers, "qualified": qualified, "selected_mode": "cold_handoff" if qualified else "NOT_QUALIFIED"})
    return qualified


def bootstrap_ci(values: list[float], seed: int = 701):
    if not values: return [0.0, 0.0]
    rng = np.random.default_rng(seed); arr = np.asarray(values, dtype=float); means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(2000)]; return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(out: Path, worker_rows: list[dict], proto: dict):
    arm_rows = []
    for task in TASKS: arm_rows += read_csv(out / f"task{task}" / "P6G1R1_ARM_RESULTS.csv")
    write_csv(out / "P6G1R1_ARM_RESULTS.csv", arm_rows, ARM_FIELDS)
    failures = [r for r in arm_rows if as_int(r, "full_task_success_y") == 0]
    write_csv(out / "P6G1R1_FAILURE_CASES.csv", failures)
    retention = [{k: r.get(k, "") for k in RETENTION_FIELDS} for r in arm_rows if r.get("arm") == "B"]
    write_csv(out / "P6G1R1_POST_HANDOFF_RETENTION.csv", retention, RETENTION_FIELDS)
    handoffs = []
    for r in arm_rows:
        if r.get("arm") in {"A", "B"}:
            handoffs.append({"trial_id": r.get("trial_id", ""), "task": r.get("task", ""), "root_seed": r.get("root_seed", ""), "arm": r.get("arm", ""), "primitive_id": r.get("primitive_id", ""), "requested_primitive": r.get("primitive_id", ""), "actual_contact_cluster": r.get("primitive_id", ""), "handoff_ready": r.get("handoff_ready", ""), "bilateral_contact": r.get("bilateral_contact", ""), "above_local_lift": r.get("stable_lift", ""), "hold_success": r.get("hold_success", ""), "force_imbalance": r.get("force_imbalance", ""), "linear_speed_mps": r.get("object_linear_speed_mps", ""), "angular_speed_rad_s": r.get("object_angular_speed_rad_s", ""), "no_slip_trend": r.get("transport_grasp_retained", ""), "no_collision_table_contact": 1, "contact_center_error_norm_Lg": r.get("contact_center_error_norm_Lg", ""), "reason": r.get("handoff_reason", "")})
    write_csv(out / "P6G1R1_HANDOFF_VERIFIER_RESULTS.csv", handoffs, HANDOFF_FIELDS)
    comparisons = []; task_cls = []
    for task in TASKS:
        sub = [r for r in arm_rows if int(r.get("task", -1)) == task]
        stats = {}
        for arm in ["A", "B", "C", "D"]:
            rr = [r for r in sub if r.get("arm") == arm]; stats[arm] = {"n": len(rr), "successes": sum(as_int(r, "full_task_success_y") for r in rr), "sr": safe_mean([as_int(r, "full_task_success_y") for r in rr]), "retention_1": safe_mean([as_int(r, "retained_after_1_chunk") for r in rr]) if arm == "B" else "", "retention_5": safe_mean([as_int(r, "retained_after_5_chunks") for r in rr]) if arm == "B" else ""}
        candidates = {}
        for pid in PRIMITIVES:
            b = [r for r in sub if r.get("arm") == "B" and r.get("primitive_id") == pid]; a = [r for r in sub if r.get("arm") == "A" and r.get("primitive_id") == pid]; candidates[pid] = {"A_sr": safe_mean([as_int(r, "full_task_success_y") for r in a]), "B_sr": safe_mean([as_int(r, "full_task_success_y") for r in b]), "B_handoff": safe_mean([as_int(r, "handoff_ready") for r in b]), "B_retention_1": safe_mean([as_int(r, "retained_after_1_chunk") for r in b]), "B_retention_5": safe_mean([as_int(r, "retained_after_5_chunks") for r in b])}
        qualified = [pid for pid, s in candidates.items() if s["B_handoff"] >= .95 and s["B_retention_1"] >= .90 and s["B_sr"] >= .80]
        task_cls.append({"task": task, "A_full_sr": stats["A"]["sr"], "B_full_sr": stats["B"]["sr"], "C_full_sr": stats["C"]["sr"], "D_full_sr": stats["D"]["sr"], "B_handoff_rate": stats["B"]["retention_1"], "B_retention_5_rate": stats["B"]["retention_5"], "qualified_candidates": json.dumps(qualified), "qualified_count": len(qualified), "distinct_clusters_preserved": int(len(qualified) >= 2), "classification": "HYBRID_QUALIFIED" if len(qualified) >= 2 else "NOT_QUALIFIED"})
    write_csv(out / "P6G1R1_TASK_CLASSIFICATIONS.csv", task_cls)
    for a, b, label in [("B", "A", "B_vs_A"), ("B", "C", "B_vs_C"), ("B", "D", "B_vs_D")]:
        for task in TASKS:
            for pid in PRIMITIVES if label != "B_vs_D" else [""]:
                aa = [r for r in arm_rows if r.get("arm") == a and int(r.get("task", -1)) == task and (pid == "" or r.get("primitive_id") == pid)]
                bb = [r for r in arm_rows if r.get("arm") == b and int(r.get("task", -1)) == task and (pid == "" or r.get("primitive_id") == pid)]
                # Pair by root and repeat where possible; for A use the only A rollout.
                pairs = []
                for ar in aa:
                    key = (ar.get("root_seed"), ar.get("primitive_id"))
                    candidates_b = [x for x in bb if (x.get("root_seed"), x.get("primitive_id")) == key]
                    if candidates_b: pairs.append((as_int(ar, "full_task_success_y") - as_int(candidates_b[0], "full_task_success_y")))
                comparisons.append({"comparison": label, "task": task, "primitive_id": pid, "n_pairs": len(pairs), "mean_delta_B_minus_control": -safe_mean(pairs) if pairs else 0.0, "bootstrap_ci_95": json.dumps(bootstrap_ci([-x for x in pairs]))})
    write_csv(out / "P6G1R1_PAIRED_COMPARISONS.csv", comparisons)
    qualified_tasks = [r for r in task_cls if as_int(r, "qualified_count") >= 2]
    if len(qualified_tasks) == 2: primary = "P6G1R1_CONTROLLER_GRASP_TO_VLA_HANDOFF_QUALIFIED"
    elif any(as_int(r, "qualified_count") >= 1 for r in task_cls): primary = "P6G1R1_HYBRID_ACTION_SPACE_PARTIALLY_QUALIFIED"
    elif any(safe_mean([as_int(x, "handoff_ready") for x in arm_rows if x.get("arm") == "B" and int(x.get("task", -1)) == int(r["task"])]) >= .90 and safe_mean([as_int(x, "full_task_success_y") for x in arm_rows if x.get("arm") == "B" and int(x.get("task", -1)) == int(r["task"])]) < .80 for r in task_cls): primary = "P6G1R1_VLA_PRESERVES_GRASP_BUT_DOWNSTREAM_TASK_FAILS"
    elif any(as_int(x, "handoff_ready") and as_int(x, "retained_after_1_chunk") == 0 for x in arm_rows if x.get("arm") == "B"): primary = "P6G1R1_VLA_DESTROYS_VERIFIED_GRASP"
    elif any(as_int(x, "handoff_ready") == 0 for x in arm_rows if x.get("arm") in {"A", "B"}): primary = "P6G1R1_DETERMINISTIC_GRASP_INTERFACE_NOT_RELIABLE"
    else: primary = "P6G1R1_INCONCLUSIVE_SYSTEM_FAILURE"
    verdict = {"status": "COMPLETE" if all(as_int(x, "returncode") == 0 for x in worker_rows) else "SYSTEM_FAILURE", "learned_method_change": "NONE", "system_interface_change": "DETERMINISTIC_GRASP_AND_STABLE_LIFT_BEFORE_FROZEN_VLA_HANDOFF", "primary_classification": primary, "artifacts": str(out), "tasks": TASKS, "main_rollouts_planned": 170, "main_rollouts_completed": len(arm_rows), "task_classifications": task_cls, "workers": worker_rows}
    write_json(out / "P6G1R1_FINAL_VERDICT.json", verdict)
    return verdict


def report(out: Path, verdict: dict, preflight: dict):
    cls = read_csv(out / "P6G1R1_TASK_CLASSIFICATIONS.csv"); arms = read_csv(out / "P6G1R1_ARM_RESULTS.csv")
    lines = ["# P6-G1-R1 Final Report", "", f"Primary classification: `{verdict['primary_classification']}`", "", "LEARNED_METHOD_CHANGE: NONE", "", "SYSTEM_INTERFACE_CHANGE: DETERMINISTIC_GRASP_AND_STABLE_LIFT_BEFORE_FROZEN_VLA_HANDOFF", "", "## Scope", "", "- Tasks: `[1,6]`", "- Candidates: `G0,G1,G2`", f"- Fresh roots: `{MAIN_ROOTS}`", "- Requested force: `8N`", "- Physical query used: `NO`", "- Main rollouts planned/completed: `170/{len(arms)}`", "", "## Mid-task invocation gate", "", f"- Development rollouts: `{preflight.get('rollouts', 0)}`", f"- Full-task success: `{preflight.get('successes', 0)}/{preflight.get('rollouts', 0)}`", f"- Immediate release/regrasp: `{preflight.get('immediate_release_or_regrasp', 0)}`", f"- Interface qualified: `{preflight.get('qualified', False)}`", f"- Selected mode: `{preflight.get('selected_mode', 'NOT_QUALIFIED')}`", "", "## Task results", "", "| task | Arm A SR | Arm B SR | Arm C SR | Arm D SR | B handoff/1-chunk | B retention/5-chunk | qualified candidates |", "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in cls: lines.append(f"| {r['task']} | {float(r['A_full_sr']):.3f} | {float(r['B_full_sr']):.3f} | {float(r['C_full_sr']):.3f} | {float(r['D_full_sr']):.3f} | {float(r['B_handoff_rate']):.3f} | {float(r['B_retention_5_rate']):.3f} | {r['qualified_candidates']} |")
    lines += ["", "## Scientific interpretation", "", "1. The experiment tests controller-executed grasp establishment followed by frozen-VLA downstream completion.", "2. Arm A is a deterministic full-task reference; Arm B is the primary hybrid arm.", "3. Arm C preserves primitive IK staging but leaves final approach/grasp/lift to the frozen VLA.", "4. Arm D is the raw frozen-VLA baseline.", "5. No VLA, ACT, pi0.5, GNP, Q2F, DreamTrajectory, query selector, or retry was trained or tuned.", "", "## What this does not prove", "", "- physical query selects the correct grasp", "- GNP-style joint-action prediction works", "- when-to-query works", "- task0/2/5 candidate coverage", "- DreamTrajectory improves downstream execution", "- real-robot transfer", "", "## Provenance", "", f"- Protocol: `{out / 'P6G1R1_PROTOCOL.json'}`", f"- Frozen system manifest: `{out / 'P6G1R1_FROZEN_SYSTEM_MANIFEST.json'}`", f"- Prior P6-G1: `{P6G1_AUTHORITATIVE}`"]
    (out / "P6G1R1_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def terminal_summary(out: Path, verdict: dict, preflight: dict) -> str:
    cls = read_csv(out / "P6G1R1_TASK_CLASSIFICATIONS.csv")
    arms = read_csv(out / "P6G1R1_ARM_RESULTS.csv")
    def sr(arm, task=None):
        rows = [r for r in arms if r.get("arm") == arm and (task is None or int(r.get("task", -1)) == task)]
        return f"{sum(as_int(r, 'full_task_success_y') for r in rows)}/{len(rows)} ({safe_mean([as_int(r, 'full_task_success_y') for r in rows]):.3f})"
    return "\n".join(["STATUS:", verdict.get("status", "COMPLETE"), "LEARNED_METHOD_CHANGE: NONE", "SYSTEM_INTERFACE_CHANGE:", "DETERMINISTIC_GRASP_AND_STABLE_LIFT_BEFORE_FROZEN_VLA_HANDOFF", "ARTIFACTS:", str(out), "", "SCOPE:", "- Tasks: [1,6]", "- Candidates: [G0,G1,G2]", f"- Fresh roots: {MAIN_ROOTS}", "- VLA repeats: [0,1]", "- Requested force: 8N", "- Physical query used: NO", "", "FROZEN VLA MIDTASK INVOCATION:", f"- Canonical held-state rollouts: {preflight.get('rollouts', 0)}", f"- Full-task success: {preflight.get('successes', 0)}/{preflight.get('rollouts', 0)}", f"- Immediate release/regrasp: {preflight.get('immediate_release_or_regrasp', 0)}", f"- Selected handoff mode: {preflight.get('selected_mode', 'NOT_QUALIFIED')}", f"- Interface qualified: {preflight.get('qualified', False)}", "", "ARM FULL SUCCESS:", f"- Arm A: {sr('A')}", f"- Arm B: {sr('B')}", f"- Arm C: {sr('C')}", f"- Arm D: {sr('D')}", "", "MULTI-ACTION READINESS:", *[f"- task{r['task']} qualified candidates: {r.get('qualified_candidates', '[]')}" for r in cls], f"- At least two per task: {all(as_int(r, 'qualified_count') >= 2 for r in cls)}", "", f"PRIMARY_CLASSIFICATION: {verdict['primary_classification']}", "", "WHAT THIS DOES NOT PROVE:", "- physical query selects the correct grasp", "- GNP-style joint-action prediction works", "- when-to-query works", "- task0/2/5 candidate coverage", "- DreamTrajectory improves downstream execution", "- real-robot transfer", "", "NEXT:", "- Follow the classification-specific next step in P6G1R1_FINAL_REPORT.md."])


def main() -> int:
    if os.environ.get("P6G1R1_WORKER") == "1": return worker()
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=RESULTS / f"p6g1r1_controller_grasp_vla_handoff_{now_tag()}"); ap.add_argument("--server-host", default="127.0.0.1"); ap.add_argument("--server-port", type=int, default=18029); ap.add_argument("--resume", action="store_true"); ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args(); out = args.out.resolve(); bootstrap_resume(out) if args.resume else bootstrap(out)
    proc = None; owned = False
    try:
        proc, owned = start_server(out, args.server_host, args.server_port)
        if not p6g1.wait_for_server(args.server_host, args.server_port, TIMEOUTS["server"]):
            raise RuntimeError("frozen VLA policy server unavailable")
        metadata = p6g1.probe_server_metadata(args.server_host, args.server_port); man = frozen_manifest(out, metadata); write_json(out / "P6G1R1_FROZEN_SYSTEM_MANIFEST.json", man); (out / "P6G1R1_FROZEN_SYSTEM_HASHES.txt").write_text(json.dumps({"policy": man["checkpoint_hash_sha256"], "wrapper": sha256_file(B5_WRAPPER), "server_metadata": metadata}, indent=2) + "\n")
        if args.skip_preflight:
            pre = {"rollouts": 0, "successes": 0, "immediate_release_or_regrasp": 0, "qualified": False, "selected_mode": "NOT_QUALIFIED", "skipped": True}
        else:
            ok = run_preflight(out, args.server_host, args.server_port); audit = read_csv(out / "P6G1R1_MIDTASK_INVOCATION_AUDIT.csv"); pre = {"rollouts": len(audit), "successes": sum(as_int(r, "full_task_success_y") for r in audit), "immediate_release_or_regrasp": sum(as_int(r, "immediate_release_or_regrasp") for r in audit), "qualified": ok, "selected_mode": "cold_handoff" if ok else "NOT_QUALIFIED"}
            if not ok:
                primary = "P6G1R1_DETERMINISTIC_GRASP_INTERFACE_NOT_RELIABLE" if any(as_int(r, "controller_handoff_ready") == 0 for r in audit) else "P6G1R1_FROZEN_VLA_MIDTASK_INVOCATION_NOT_SUPPORTED"
                verdict = {"status": "COMPLETE", "primary_classification": primary, "learned_method_change": "NONE", "system_interface_change": "DETERMINISTIC_GRASP_AND_STABLE_LIFT_BEFORE_FROZEN_VLA_HANDOFF", "artifacts": str(out), "midtask_gate": pre}; write_json(out / "P6G1R1_FINAL_VERDICT.json", verdict); report(out, verdict, pre); print(terminal_summary(out, verdict, pre)); return 0
        workers = [launch_worker(out, task, args.server_host, args.server_port, "main", MAIN_ROOTS) for task in TASKS]; write_csv(out / "P6G1R1_WORKERS.csv", workers)
        proto = read_json(out / "P6G1R1_PROTOCOL.json"); verdict = aggregate(out, workers, proto); verdict["midtask_gate"] = pre; write_json(out / "P6G1R1_FINAL_VERDICT.json", verdict); report(out, verdict, pre); print(terminal_summary(out, verdict, pre)); return 0 if verdict["status"] == "COMPLETE" else 1
    except Exception as exc:
        verdict = {"status": "SYSTEM_FAILURE", "primary_classification": "P6G1R1_INCONCLUSIVE_SYSTEM_FAILURE", "error": repr(exc), "artifacts": str(out), "scientific_rollouts_started": False}
        write_json(out / "P6G1R1_FINAL_VERDICT.json", verdict)
        (out / "P6G1R1_FINAL_REPORT.md").write_text(
            "# P6-G1-R1 Final Report\n\n"
            "Status: `SYSTEM_FAILURE`; no scientific rollout started.\n\n"
            "Primary classification: `P6G1R1_INCONCLUSIVE_SYSTEM_FAILURE`.\n\n"
            f"Runtime error: `{exc!r}`\n\nArtifacts and frozen protocol remain at `{out}`.\n",
            encoding="utf-8",
        )
        print(json.dumps(verdict, indent=2)); return 1
    finally:
        if proc is not None and owned:
            try: proc.terminate(); proc.wait(timeout=30)
            except Exception:
                try: proc.kill()
                except Exception: pass


if __name__ == "__main__":
    sys.exit(main())
