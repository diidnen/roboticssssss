#!/usr/bin/env python3
"""P6-G1 primitive/IK-conditioned frozen-VLA grasp realization gate.

This runner is deliberately local to the P6-G1 experiment.  It reuses the
qualified P6-G0 grasp candidates, stages the end effector analytically, and
invokes the frozen OpenPI/Tabero policy only after the staged handoff.
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
import socket
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/home/exouser/Tabero")
RESULTS_ROOT = REPO / "analysis/results"
P6G0_RESULTS = RESULTS_ROOT / "p6g0_grasp_force_physics_benchmark_20260824_182251"
P6G0R1_RESULTS = RESULTS_ROOT / "p6g0r1_confirmatory_and_candidate_coverage_20260824_232236"
P6G0_SCRIPT = REPO / "analysis/p6g0_grasp_force_physics_benchmark.py"
B5_RESULTS = RESULTS_ROOT / "b5_tabero_neutral_20260822_040652"
B5_SERVER_WRAPPER = B5_RESULTS / "scripts/b5_serve_policy_with_explicit_norm_stats.py"
B5_STUBS = B5_RESULTS / "scripts/stubs"
SERVER_PYTHON = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/.venv/bin/python")
TABERO_VTLA = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
TABERO_VTLA_SRC = TABERO_VTLA / "src"
TABERO_VTLA_CLIENT_SRC = TABERO_VTLA / "packages/openpi-client/src"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path(
    "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.warp.core-1.8.2+lx64"
)
OPENPI_CLIENT_SRC = REPO / "benchmarks/openpi/openpi-client/src"
HDF5_DIR = REPO / "benchmarks/datasets/libero/assembled_hdf5"
CONFIG_DIR = REPO / "benchmarks/datasets/libero/config"
ASSETS_DIR = REPO / "benchmarks/datasets/libero/USD"

POLICY_CONFIG = "pi0_lora_tacfield_tabero"
POLICY_DIR = Path(
    "/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/"
    "checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999"
)
NORM_STATS_DIR = POLICY_DIR / "assets/NathanWu7/tabero"
POLICY_STEP = 49999

TASKS = [1, 6]
TASK_OBJECTS = {1: "cream_cheese_1", 6: "butter_1"}
TASK_NAMES = {1: "cream cheese", 6: "butter"}
TASK_INSTRUCTIONS = {
    1: "pick up the cream cheese and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
BASKET_NAME = "basket_1"
TASK_SUITE = "libero_object"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"

FRESH_ROOTS = list(range(9100, 9110))
POLICY_REPEATS = [0, 1]
PRIMITIVE_IDS = ["G0", "G1", "G2"]
SOURCE_LABEL_ORDER = ["g_minus", "g_center", "g_plus"]

FIXED_FORCE_N = 8.0
FORCE_SLOT_HALF_N = FIXED_FORCE_N / 2.0
ROOT_SETTLE_STEPS = 20
STAGE_STEPS = 45
STABILIZE_STEPS = 4
REFERENCE_DESCEND_STEPS = 35
REFERENCE_CLOSE_STEPS = 70
REFERENCE_HOLD_STEPS = 40
REFERENCE_LIFT_STEPS = 45
VLA_MAX_INFERENCE_STEPS = 22
VLA_REPLAN_STEPS = 10
LOCAL_LIFT_Z_M = 0.03
LOCAL_LIFT_HOLD_STEPS = 8
CONTACT_FORCE_THRESHOLD_N = 0.15
PREGRASP_POS_TOL_M = 0.005
PREGRASP_ORI_TOL_RAD = math.radians(5.0)
OBJECT_DISTURB_POS_TOL_M = 0.003
OBJECT_DISTURB_ORI_TOL_RAD = math.radians(5.0)
D_OPEN = 0.04
D_CLOSED = 0.0

DIST_LAMBDA_P = 1.0
DIST_LAMBDA_R = 0.25
DIST_LAMBDA_D = 0.50
MIN_DISTINCT_SEPARATION_NORM = 0.15

TIMEOUTS_S = {
    "server_start": 900,
    "worker": 21600,
    "rollout": 900,
    "restore": 30,
}


ROOT_MANIFEST_FIELDS = [
    "task",
    "object",
    "root_index",
    "root_seed",
    "root_state_hash",
    "object_pose_base_x",
    "object_pose_base_y",
    "object_pose_base_z",
    "object_pose_base_qw",
    "object_pose_base_qx",
    "object_pose_base_qy",
    "object_pose_base_qz",
    "source",
]

PARITY_FIELDS = [
    "trial_id",
    "task",
    "root_seed",
    "arm",
    "primitive_id",
    "policy_repeat",
    "root_state_hash",
    "restore_state_hash",
    "state_parity",
    "object_pose_before_stage_x",
    "object_pose_before_stage_y",
    "object_pose_before_stage_z",
    "object_pose_handoff_x",
    "object_pose_handoff_y",
    "object_pose_handoff_z",
    "object_disturbance_m",
    "object_rotation_disturbance_rad",
]

STAGING_FIELDS = [
    "trial_id",
    "task",
    "root_seed",
    "arm",
    "primitive_id",
    "policy_repeat",
    "stage_attempted",
    "ik_converged",
    "collision_free",
    "no_fingertip_contact_before_handoff",
    "pregrasp_position_error_m",
    "pregrasp_orientation_error_rad",
    "object_disturbance_m",
    "object_rotation_disturbance_rad",
    "staging_success",
    "handoff_valid",
    "steps",
    "error",
]

ROLLOUT_FIELDS = [
    "trial_id",
    "task",
    "object",
    "root_seed",
    "arm",
    "primitive_id",
    "source_grasp_label",
    "policy_repeat",
    "policy_repeat_seed",
    "prompt",
    "requested_force_N",
    "state_parity",
    "staging_success",
    "handoff_valid",
    "bilateral_contact",
    "first_stable_bilateral_step",
    "local_grasp_success",
    "short_lift_success",
    "drop",
    "contact_loss",
    "collision",
    "policy_timeout",
    "terminated",
    "truncated",
    "steps",
    "num_policy_chunks",
    "mean_measured_force_N",
    "peak_measured_force_N",
    "mean_original_vla_force_slot_N",
    "peak_original_vla_force_slot_N",
    "object_lift_dz_m",
    "object_rotation_rad",
    "left_contact_obj_x",
    "left_contact_obj_y",
    "left_contact_obj_z",
    "right_contact_obj_x",
    "right_contact_obj_y",
    "right_contact_obj_z",
    "contact_mid_obj_x",
    "contact_mid_obj_y",
    "contact_mid_obj_z",
    "finger_closing_obj_x",
    "finger_closing_obj_y",
    "finger_closing_obj_z",
    "wrist_obj_x",
    "wrist_obj_y",
    "wrist_obj_z",
    "wrist_obj_qw",
    "wrist_obj_qx",
    "wrist_obj_qy",
    "wrist_obj_qz",
    "grasp_depth_m",
    "gripper_width_m",
    "contact_telemetry_path",
    "force_telemetry_path",
    "policy_log_path",
    "error",
]

CONTACT_FIELDS = [
    "trial_id",
    "task",
    "root_seed",
    "arm",
    "primitive_id",
    "policy_repeat",
    "phase",
    "step",
    "t_s",
    "contact_state",
    "stable_bilateral_run",
    "left_contact_force_norm_N",
    "right_contact_force_norm_N",
    "measured_squeeze_N",
    "measured_applied_force_N",
    "left_contact_obj_x",
    "left_contact_obj_y",
    "left_contact_obj_z",
    "right_contact_obj_x",
    "right_contact_obj_y",
    "right_contact_obj_z",
    "contact_mid_obj_x",
    "contact_mid_obj_y",
    "contact_mid_obj_z",
    "finger_closing_obj_x",
    "finger_closing_obj_y",
    "finger_closing_obj_z",
    "wrist_obj_x",
    "wrist_obj_y",
    "wrist_obj_z",
    "wrist_obj_qw",
    "wrist_obj_qx",
    "wrist_obj_qy",
    "wrist_obj_qz",
    "grasp_depth_m",
    "gripper_width_m",
    "object_x",
    "object_y",
    "object_z",
    "object_qw",
    "object_qx",
    "object_qy",
    "object_qz",
]

FORCE_FIELDS = [
    "trial_id",
    "task",
    "root_seed",
    "arm",
    "primitive_id",
    "policy_repeat",
    "phase",
    "step",
    "requested_force_N",
    "measured_squeeze_N",
    "measured_applied_force_N",
    "original_vla_fLx",
    "original_vla_fLy",
    "original_vla_fLz",
    "original_vla_fRx",
    "original_vla_fRy",
    "original_vla_fRz",
    "executed_fLx",
    "executed_fLy",
    "executed_fLz",
    "executed_fRx",
    "executed_fRy",
    "executed_fRz",
    "gripper_cmd",
]


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    active_fields = fields
    if exists:
        with path.open(newline="", encoding="utf-8") as existing_fh:
            header = next(csv.reader(existing_fh), None)
        if header:
            active_fields = header
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=active_fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in active_fields})
        fh.flush()
        os.fsync(fh.fileno())


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def dedupe_rows(rows: list[dict], key_fn) -> list[dict]:
    order: list[Any] = []
    by_key: dict[Any, dict] = {}
    for row in rows:
        key = key_fn(row)
        if key not in by_key:
            order.append(key)
        by_key[key] = row
    return [by_key[key] for key in order]


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
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = str(file_path.relative_to(path)).replace(os.sep, "/")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(file_path).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def stable_hash_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_status(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=path, text=True).strip()
    except Exception:
        return "UNKNOWN"


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        val = row.get(key, default)
        if val in ("", None):
            return default
        return float(val)
    except Exception:
        return default


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        val = row.get(key, default)
        if val in ("", None):
            return default
        return int(float(val))
    except Exception:
        return default


def safe_mean(vals: list[float]) -> float:
    vals = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


class Timeout:
    def __init__(self, seconds: float, stage: str):
        self.seconds = int(max(1, seconds))
        self.stage = stage
        self.old_handler = None

    def __enter__(self):
        def handler(_signum, _frame):
            raise TimeoutError(f"{self.stage} timed out after {self.seconds}s")

        self.old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)
        if self.old_handler is not None:
            signal.signal(signal.SIGALRM, self.old_handler)
        return False


def recover_primitive_library() -> dict:
    manifest_path = P6G0_RESULTS / "P6G0_GRASP_CANDIDATE_MANIFEST.csv"
    rows = [r for r in read_csv(manifest_path) if int(r["task"]) in TASKS]
    by_task: dict[int, list[dict]] = {task: [] for task in TASKS}
    for row in rows:
        by_task[int(row["task"])].append(row)

    primitives: list[dict] = []
    for task in TASKS:
        task_rows = sorted(by_task[task], key=lambda r: (as_float(r, "offset_m"), r.get("grasp_label", "")))
        if len(task_rows) != 3:
            raise RuntimeError(f"Expected 3 P6-G0 candidates for task {task}, got {len(task_rows)}")
        for idx, row in enumerate(task_rows):
            pid = PRIMITIVE_IDS[idx]
            axis = [as_float(row, "axis_object_x"), as_float(row, "axis_object_y"), as_float(row, "axis_object_z")]
            center = [as_float(row, "center_obj_x"), as_float(row, "center_obj_y"), as_float(row, "center_obj_z")]
            pregrasp = [
                as_float(row, "pregrasp_obj_x"),
                as_float(row, "pregrasp_obj_y"),
                as_float(row, "pregrasp_obj_z"),
            ]
            primitives.append(
                {
                    "primitive_id": pid,
                    "task": task,
                    "object": TASK_OBJECTS[task],
                    "source_grasp_label": row["grasp_label"],
                    "source_manifest_row": row,
                    "deterministic_sort_key": {"offset_m": as_float(row, "offset_m"), "source_label": row["grasp_label"]},
                    "desired_object_relative_grasp_transform": {
                        "position_m": center,
                        "orientation": "runtime nominal P4/P6 reset end-effector wrist orientation",
                    },
                    "desired_object_relative_pregrasp_transform": {
                        "position_m": pregrasp,
                        "orientation": "same runtime nominal wrist orientation as grasp",
                    },
                    "approach_direction_object_frame": [0.0, 0.0, -1.0],
                    "finger_closing_direction_object_frame": axis,
                    "grasp_depth_m": center[2],
                    "gripper_width_m": D_OPEN,
                    "usable_extent_Lg_m": as_float(row, "Lg_m"),
                    "expected_contact_points": {
                        "source": "P6-G0 did not save true mesh contact points; P6-G1 records gripper-frame contact proxies at runtime",
                        "expected_contact_center_object_frame_m": center,
                    },
                    "handoff_condition": {
                        "ik_converged": True,
                        "no_fingertip_contact_before_handoff": True,
                        "pregrasp_position_error_m_lte": PREGRASP_POS_TOL_M,
                        "pregrasp_orientation_error_rad_lte": PREGRASP_ORI_TOL_RAD,
                        "object_disturbance_m_lte": OBJECT_DISTURB_POS_TOL_M,
                    },
                    "local_stop_predicate": {
                        "lift_height_m": LOCAL_LIFT_Z_M,
                        "hold_steps": LOCAL_LIFT_HOLD_STEPS,
                        "or_timeout_policy_steps": VLA_MAX_INFERENCE_STEPS * VLA_REPLAN_STEPS,
                    },
                    "semantic_description_after_geometry_inspection": (
                        f"object-frame offset {as_float(row, 'offset_m'):+.6f} m along axis {axis}"
                    ),
                }
            )
    return {
        "source_artifacts": {
            "p6g0_manifest": str(manifest_path),
            "p6g0r1_track_a": str(P6G0R1_RESULTS / "TRACK_A/P6G0R1A_COMPLETED_BRANCHES.csv"),
        },
        "id_policy": "deterministic object-frame sort by signed offset_m ascending: G0,G1,G2",
        "semantic_labels_sent_to_vla": False,
        "tasks": TASKS,
        "primitives": primitives,
    }


def primitive_for(library: dict, task: int, primitive_id: str) -> dict:
    for primitive in library["primitives"]:
        if int(primitive["task"]) == int(task) and primitive["primitive_id"] == primitive_id:
            return primitive
    raise KeyError((task, primitive_id))


def write_primitive_markdown(out: Path, library: dict) -> None:
    lines = [
        "# P6-G1 Primitive Library",
        "",
        f"Source manifest: `{library['source_artifacts']['p6g0_manifest']}`",
        "",
        "IDs are frozen by signed object-frame offset sorting. Semantic labels are not sent to the VLA.",
        "",
        "| task | object | primitive | P6-G0 label | offset_m | grasp center obj m | pregrasp obj m |",
        "| ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for primitive in library["primitives"]:
        row = primitive["source_manifest_row"]
        lines.append(
            "| {task} | {obj} | {pid} | {src} | {off:.6f} | {center} | {pre} |".format(
                task=primitive["task"],
                obj=primitive["object"],
                pid=primitive["primitive_id"],
                src=primitive["source_grasp_label"],
                off=as_float(row, "offset_m"),
                center=json.dumps(primitive["desired_object_relative_grasp_transform"]["position_m"]),
                pre=json.dumps(primitive["desired_object_relative_pregrasp_transform"]["position_m"]),
            )
        )
    (out / "P6G1_PRIMITIVE_LIBRARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def protocol_obj() -> dict:
    return {
        "name": "P6-G1 primitive/IK-conditioned frozen-VLA grasp realization gate",
        "created_at_utc": utc_iso(),
        "learned_method_change": "NONE",
        "system_interface_change": "PRIMITIVE_IK_STAGING_BEFORE_FROZEN_VLA",
        "tasks": TASKS,
        "fresh_roots": FRESH_ROOTS,
        "primitive_ids": PRIMITIVE_IDS,
        "rollout_plan": {
            "reference_per_task": len(FRESH_ROOTS) * len(PRIMITIVE_IDS),
            "primitive_conditioned_vla_per_task": len(FRESH_ROOTS) * len(PRIMITIVE_IDS) * len(POLICY_REPEATS),
            "raw_vla_per_task": len(FRESH_ROOTS) * len(POLICY_REPEATS),
            "total": len(TASKS)
            * (
                len(FRESH_ROOTS) * len(PRIMITIVE_IDS)
                + len(FRESH_ROOTS) * len(PRIMITIVE_IDS) * len(POLICY_REPEATS)
                + len(FRESH_ROOTS) * len(POLICY_REPEATS)
            ),
        },
        "force_contract": {
            "requested_grip_force_N": FIXED_FORCE_N,
            "vla_force_override": "preserve policy pose and gripper scalar; replace force slots [7:13] with fLz=fRz=4N",
            "original_vla_force_slots_recorded": True,
        },
        "staging": {
            "root_settle_steps_before_s0_hash": ROOT_SETTLE_STEPS,
            "stage_steps": STAGE_STEPS,
            "stabilize_steps": STABILIZE_STEPS,
            "pregrasp_position_tolerance_m": PREGRASP_POS_TOL_M,
            "pregrasp_orientation_tolerance_deg": math.degrees(PREGRASP_ORI_TOL_RAD),
            "object_disturbance_tolerance_m": OBJECT_DISTURB_POS_TOL_M,
        },
        "vla": {
            "max_inference_steps": VLA_MAX_INFERENCE_STEPS,
            "replan_steps": VLA_REPLAN_STEPS,
            "local_lift_z_m": LOCAL_LIFT_Z_M,
            "local_lift_hold_steps": LOCAL_LIFT_HOLD_STEPS,
            "instruction": "official original task instruction, identical across G0/G1/G2",
            "semantic_grasp_labels_in_prompt": False,
        },
        "classification_distance": {
            "lambda_p": DIST_LAMBDA_P,
            "lambda_R": DIST_LAMBDA_R,
            "lambda_d": DIST_LAMBDA_D,
            "position_normalizer": "P6-G0 usable_extent_Lg_m",
            "orientation_distance": "quaternion geodesic radians",
            "depth": "object-frame contact midpoint z",
        },
        "qualification": {
            "primitive": {
                "ik_staging_success_gte": 0.95,
                "requested_mode_accuracy_gte": 0.80,
                "bilateral_contact_gte": 0.80,
                "short_lift_success_gte": 0.80,
            },
            "dominant_cluster_fraction_lte": 0.70,
            "distinct_cluster_min_separation_norm_gte": MIN_DISTINCT_SEPARATION_NORM,
        },
        "decision_discipline": {
            "training": False,
            "candidate_transform_tuning_after_outcomes": False,
            "hidden_physics_input": False,
            "physical_query": False,
            "full_task_success_required": False,
        },
    }


def policy_manifest_obj(policy_hash: str | None = None, server_metadata: dict | None = None) -> dict:
    return {
        "policy_class": "OpenPI websocket policy created by openpi.policies.policy_config.create_trained_policy",
        "base_policy": "Pi0Config (pi0, not pi0.5)",
        "policy_config": POLICY_CONFIG,
        "checkpoint_path": str(POLICY_DIR),
        "checkpoint_step": POLICY_STEP,
        "checkpoint_hash_sha256": policy_hash,
        "norm_stats_dir": str(NORM_STATS_DIR),
        "server_wrapper": str(B5_SERVER_WRAPPER),
        "server_python": str(SERVER_PYTHON),
        "authoritative_downstream_artifact": str(B5_RESULTS / "FINAL_VERDICT.json"),
        "observation_specification": {
            "image": "agentview RGB uint8 224x224x3",
            "wrist_image": "eye_in_hand RGB uint8 224x224x3",
            "state": "[eef_x,eef_y,eef_z,eef_axis_angle_x,eef_axis_angle_y,eef_axis_angle_z,gripper_abs]",
            "tactile_image": "4x4 tactile_rgb history mosaic when available",
            "tactile_gripper_force": "8x6 left/right fingertip force history when available",
            "tactile_marker_motion": "1+8 marker-motion history when available",
        },
        "instruction_format": "plain official LIBERO task instruction in key 'prompt'; no force adverbs and no grasp labels",
        "action_specification": {
            "server_action_dim": "32D padded action chunk",
            "executed_effective_dim": 13,
            "slots": "[x,y,z,rx,ry,rz,gripper,fLx,fLy,fLz,fRx,fRy,fRz]",
            "control_mode": "tactile / Isaac-Libero-Franka-Hybrid-Tactile-v0",
        },
        "action_chunk_horizon": {
            "server_chunk_min_required": VLA_REPLAN_STEPS,
            "client_replan_steps": VLA_REPLAN_STEPS,
            "local_max_inference_steps": VLA_MAX_INFERENCE_STEPS,
        },
        "force_output_semantics": {
            "policy_force_slots": "left/right finger local-frame force vectors",
            "executed_force_contract": "force slots overridden to fLz=fRz=4N for fixed 8N squeeze; pose and gripper scalar preserved",
            "original_force_slots_logged": True,
        },
        "policy_state": {
            "websocket_client_reset_effect": "no-op in openpi_client.WebsocketClientPolicy",
            "action_queue_cache_in_client": "none; each replan calls infer and executes first replan_steps",
            "observation_history": "constructed online after reset/staging via tactile/force buffers",
            "recurrent_hidden_state": "not exposed by websocket interface",
        },
        "server_metadata": server_metadata or {},
        "method_change": "NONE",
    }


def write_agent_interface(out: Path, library: dict, stats: dict | None = None) -> None:
    stats = stats or {}
    entries = []
    for primitive in library["primitives"]:
        key = f"task{primitive['task']}_{primitive['primitive_id']}"
        entries.append(
            {
                "primitive_id": primitive["primitive_id"],
                "task": primitive["task"],
                "object": primitive["object"],
                "source_grasp_label": primitive["source_grasp_label"],
                "object_relative_pregrasp_transform": primitive["desired_object_relative_pregrasp_transform"],
                "object_relative_grasp_transform": primitive["desired_object_relative_grasp_transform"],
                "approach_direction_object_frame": primitive["approach_direction_object_frame"],
                "handoff_condition": primitive["handoff_condition"],
                "expected_realized_cluster": stats.get(key, {}).get("expected_realized_cluster", ""),
                "validity_statistics": stats.get(key, {}),
                "force_contract_field": {
                    "name": "requested_force_N",
                    "fixed_value_for_P6G1": FIXED_FORCE_N,
                    "future_agent_may_set": True,
                },
                "vla_instruction": TASK_INSTRUCTIONS[int(primitive["task"])],
                "local_stop_predicate": primitive["local_stop_predicate"],
            }
        )
    payload = {
        "name": "P6G1_AGENT_ACTION_INTERFACE",
        "learned_method_change": "NONE",
        "system_interface_change": "PRIMITIVE_IK_STAGING_BEFORE_FROZEN_VLA",
        "semantic_grasp_labels_sent_to_vla": False,
        "actions": entries,
    }
    write_json(out / "P6G1_AGENT_ACTION_INTERFACE.json", payload)
    lines = [
        "# P6-G1 Agent Action Interface",
        "",
        "No agent is implemented in P6-G1. This is the frozen machine-readable primitive interface for future work.",
        "",
        "| task | object | primitive | force contract | VLA instruction | validity |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        valid = entry["validity_statistics"].get("classification", "pending")
        lines.append(
            f"| {entry['task']} | {entry['object']} | {entry['primitive_id']} | "
            f"{FIXED_FORCE_N:g}N in P6-G1 | {entry['vla_instruction']} | {valid} |"
        )
    (out / "P6G1_AGENT_ACTION_INTERFACE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def server_env() -> dict:
    env = os.environ.copy()
    py_parts = [str(B5_STUBS), str(TABERO_VTLA_SRC), str(TABERO_VTLA_CLIENT_SRC)]
    if env.get("PYTHONPATH"):
        py_parts.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(py_parts),
            "PYTHONNOUSERSITE": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    return env


def start_policy_server(out: Path, port: int) -> subprocess.Popen | None:
    if tcp_port_open("127.0.0.1", port, timeout_s=0.25):
        return None
    logs = out / "P6G1_POLICY_LOGS"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(SERVER_PYTHON),
        "-u",
        str(B5_SERVER_WRAPPER),
        "--port",
        str(port),
        "--policy-config",
        POLICY_CONFIG,
        "--policy-dir",
        str(POLICY_DIR),
        "--norm-stats-dir",
        str(NORM_STATS_DIR),
    ]
    stdout = (logs / "server_stdout.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=TABERO_VTLA, env=server_env(), stdout=stdout, stderr=subprocess.STDOUT)
    return proc


def tcp_port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def wait_for_server(host: str, port: int, timeout_s: float) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        if tcp_port_open(host, port, timeout_s=2.0):
            return True
        time.sleep(5)
    return False


def probe_server_metadata(host: str, port: int) -> dict:
    script = (
        "import json;"
        "from openpi_client import websocket_client_policy as w;"
        f"c=w.WebsocketClientPolicy('{host}', {int(port)});"
        "print(json.dumps(c.get_server_metadata(), sort_keys=True, default=str))"
    )
    env = server_env()
    out = subprocess.check_output(
        [str(SERVER_PYTHON), "-c", script],
        cwd=TABERO_VTLA,
        env=env,
        text=True,
        timeout=60,
    )
    return json.loads(out.strip().splitlines()[-1])


def worker_env(out: Path, task: int, server_host: str, server_port: int, roots_per_task: int) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(REPO), str(OPENPI_CLIENT_SRC)]),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "TABERO_ROOT": str(REPO),
            "HDF5_TRAJ_SOURCE_DIR": str(HDF5_DIR),
            "LIBERO_CONFIG_DIR": str(CONFIG_DIR),
            "LIBERO_ASSETS_DATA_DIR": str(ASSETS_DIR),
            "P6G1_OUT": str(out),
            "P6G1_WORKER": "1",
            "P6G1_TASK_ID": str(task),
            "P6G1_SERVER_HOST": server_host,
            "P6G1_SERVER_PORT": str(server_port),
            "P6G1_ROOTS_PER_TASK": str(roots_per_task),
            "P6G0_OUT": str(out / "P6G1_FROZEN_P4B_IMPORT"),
            "USE_RELATIVE_MODE": "False",
        }
    )
    return env


def launch_worker(out: Path, task: int, server_host: str, server_port: int, roots_per_task: int) -> dict:
    log_path = out / "logs" / f"task{task}_worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ISAAC_PY), "-u", str(Path(__file__).resolve())]
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=worker_env(out, task, server_host, server_port, roots_per_task),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            returncode = proc.wait(timeout=TIMEOUTS_S["worker"])
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            returncode = -9
    return {"task": task, "returncode": returncode, "elapsed_wall_s": time.time() - start, "log": str(log_path)}


def _tensor_to_list(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().contiguous().numpy().tolist()
    if isinstance(x, dict):
        return {str(k): _tensor_to_list(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_tensor_to_list(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    try:
        return np.asarray(x).tolist()
    except Exception:
        return repr(x)


def import_p6g0():
    spec = importlib.util.spec_from_file_location("p6g1_imported_p6g0", P6G0_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {P6G0_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def q_inv(q: np.ndarray) -> np.ndarray:
    q = q_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def q_angle(a: np.ndarray, b: np.ndarray) -> float:
    a = q_normalize(a)
    b = q_normalize(b)
    dot = abs(float(np.dot(a, b)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def pose_in_base(p6, env, obj_name: str) -> tuple[np.ndarray, np.ndarray]:
    return p6.imported_p4._pose_in_base(env, obj_name) if hasattr(p6, "imported_p4") else p6._pose_in_base(env, obj_name)


def root_state_hash(p6, env) -> str:
    return stable_hash_obj({"scene_state": _tensor_to_list(env.scene.get_state(is_relative=True))})


def object_pose_row(p6, env, obj_name: str, prefix: str = "") -> dict:
    pos, quat = p6.imported_p4._pose_in_base(env, obj_name)
    return {
        f"{prefix}x": float(pos[0]),
        f"{prefix}y": float(pos[1]),
        f"{prefix}z": float(pos[2]),
        f"{prefix}qw": float(quat[0]),
        f"{prefix}qx": float(quat[1]),
        f"{prefix}qy": float(quat[2]),
        f"{prefix}qz": float(quat[3]),
    }


def stage_to_pregrasp(env, p6, p4, task: int, primitive: dict, trial_id: str, dt: float) -> tuple[dict, Any]:
    import torch

    obj_name = TASK_OBJECTS[task]
    obs = env.observation_manager.compute()
    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    desired_q = eef0[3:7].copy()
    eef_aa = p4._aa(desired_q)
    cmd_pos = eef0[:3].copy()
    pre_obj = np.asarray(primitive["desired_object_relative_pregrasp_transform"]["position_m"], dtype=np.float64)
    target = p6.object_point_to_base(p4, env, obj_name, pre_obj)
    obj0, objq0 = p4._pose_in_base(env, obj_name)
    max_left = 0.0
    max_right = 0.0
    terminated = truncated = False
    error = ""
    steps = 0

    try:
        with Timeout(TIMEOUTS_S["rollout"], f"STAGING_{trial_id}"):
            for phase, n_steps in (("stage", STAGE_STEPS), ("stabilize", STABILIZE_STEPS)):
                start = cmd_pos.copy()
                phase_target = target.copy()
                for i in range(n_steps):
                    if phase == "stage":
                        cmd_pos = p4._interp(start, phase_target, i, n_steps)
                    else:
                        cmd_pos = phase_target.copy()
                    action = p6.make_action(p4, cmd_pos, eef_aa, D_OPEN, 0.0, env.device)
                    obs, _, term, trunc, _ = env.step(action)
                    steps += 1
                    f = obs["policy"]["gripper_net_force"][0]
                    if f.ndim == 3:
                        f = f[-1]
                    f_np = f.detach().cpu().numpy()
                    max_left = max(max_left, float(np.linalg.norm(f_np[0])))
                    max_right = max(max_right, float(np.linalg.norm(f_np[1])))
                    terminated = bool(term[0].item())
                    truncated = bool(trunc[0].item())
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
    except Exception as exc:
        error = repr(exc)

    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    obj1, objq1 = p4._pose_in_base(env, obj_name)
    pos_err = float(np.linalg.norm(eef[:3] - target))
    ori_err = q_angle(eef[3:7], desired_q)
    obj_dist = float(np.linalg.norm(obj1 - obj0))
    obj_rot = q_angle(objq1, objq0)
    no_contact = int(max_left <= CONTACT_FORCE_THRESHOLD_N and max_right <= CONTACT_FORCE_THRESHOLD_N)
    ik_ok = int(pos_err <= PREGRASP_POS_TOL_M and ori_err <= PREGRASP_ORI_TOL_RAD and not error)
    collision_free = int(no_contact and not terminated and not truncated)
    handoff_valid = int(
        ik_ok
        and collision_free
        and obj_dist <= OBJECT_DISTURB_POS_TOL_M
        and obj_rot <= OBJECT_DISTURB_ORI_TOL_RAD
    )
    row = {
        "trial_id": trial_id,
        "task": task,
        "root_seed": "",
        "arm": "",
        "primitive_id": primitive["primitive_id"],
        "policy_repeat": "",
        "stage_attempted": 1,
        "ik_converged": ik_ok,
        "collision_free": collision_free,
        "no_fingertip_contact_before_handoff": no_contact,
        "pregrasp_position_error_m": pos_err,
        "pregrasp_orientation_error_rad": ori_err,
        "object_disturbance_m": obj_dist,
        "object_rotation_disturbance_rad": obj_rot,
        "staging_success": handoff_valid,
        "handoff_valid": handoff_valid,
        "steps": steps,
        "error": error,
    }
    return row, obs


def settle_root_before_hash(env, p6, p4, steps: int = ROOT_SETTLE_STEPS):
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    eef_aa = p4._aa(eef[3:7])
    cmd_pos = eef[:3].copy()
    for _ in range(int(steps)):
        action = p6.make_action(p4, cmd_pos, eef_aa, D_OPEN, 0.0, env.device)
        obs, _, term, trunc, _ = env.step(action)
        if bool(term[0].item()) or bool(trunc[0].item()):
            break
    return env.observation_manager.compute()


def _to_uint8_rgb(img):
    import torch

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _pad_history_front(items: list[np.ndarray], target_len: int) -> list[np.ndarray]:
    if len(items) >= target_len:
        return items[-target_len:]
    return [items[0]] * (target_len - len(items)) + items


class OnlineTactileBuffer:
    def __init__(self, tactile_output_type: str = "tactile_rgb") -> None:
        self.tactile_sensors = ("gsmini_left", "gsmini_right")
        self.tactile_output_type = tactile_output_type
        self.reset()

    def reset(self) -> None:
        self._left_frames: deque[np.ndarray] = deque(maxlen=8)
        self._right_frames: deque[np.ndarray] = deque(maxlen=8)
        self._force_hist: deque[np.ndarray] = deque(maxlen=8)
        self._marker_hist: deque[np.ndarray] = deque(maxlen=8)
        self._marker_init: np.ndarray | None = None

    def update_force(self, obs: dict) -> None:
        import torch

        policy_obs = obs.get("policy", {}) if isinstance(obs, dict) else {}
        if not isinstance(policy_obs, dict) or "gripper_net_force" not in policy_obs:
            return
        gnf = policy_obs["gripper_net_force"]
        if isinstance(gnf, torch.Tensor):
            gnf = gnf.detach().cpu().numpy()
        gnf = np.asarray(gnf)
        gnf0 = np.squeeze(gnf, axis=0)
        inst = gnf0.reshape(-1, 2, 3)[0].reshape(6).astype(np.float32)
        self._force_hist.append(inst)

    def update_tactile_frames(self, env, env_id: int = 0) -> None:
        left_name, right_name = self.tactile_sensors
        left_img = env.unwrapped.scene.sensors[left_name].data.output[self.tactile_output_type][env_id]
        right_img = env.unwrapped.scene.sensors[right_name].data.output[self.tactile_output_type][env_id]
        self._left_frames.append(_to_uint8_rgb(left_img))
        self._right_frames.append(_to_uint8_rgb(right_img))

    def update_marker_motion(self, obs: dict) -> None:
        import torch

        policy_obs = obs.get("policy", {}) if isinstance(obs, dict) else {}
        if not isinstance(policy_obs, dict) or "gripper_marker_motion" not in policy_obs:
            return
        gmm = policy_obs["gripper_marker_motion"]
        if isinstance(gmm, torch.Tensor):
            gmm = gmm.detach().cpu().numpy()
        gmm0 = np.squeeze(np.asarray(gmm), axis=0)
        if gmm0.ndim != 4:
            return
        init_pos = gmm0[:, 0, :, :].reshape(-1, 2).astype(np.float32)
        curr_pos = gmm0[:, 1, :, :].reshape(-1, 2).astype(np.float32)
        if self._marker_init is None:
            self._marker_init = init_pos
        self._marker_hist.append(curr_pos)

    def get_force_history(self) -> np.ndarray | None:
        if not self._force_hist:
            return None
        return np.stack(_pad_history_front(list(self._force_hist), 8), axis=0).astype(np.float32)

    def get_tactile_image(self) -> np.ndarray | None:
        if not self._left_frames or not self._right_frames:
            return None
        import cv2

        left_hist = _pad_history_front(list(self._left_frames), 8)
        right_hist = _pad_history_front(list(self._right_frames), 8)
        canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        cell_h, cell_w = 56, 56
        for k in range(8):
            r, c = divmod(k, 2)
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            canvas[y0:y1, x0:x1] = cv2.resize(left_hist[k], (cell_w, cell_h))
            x0, x1 = (c + 2) * cell_w, (c + 3) * cell_w
            canvas[y0:y1, x0:x1] = cv2.resize(right_hist[k], (cell_w, cell_h))
        return canvas

    def get_marker_motion(self) -> np.ndarray | None:
        if self._marker_init is None or not self._marker_hist:
            return None
        hist = _pad_history_front(list(self._marker_hist), 8)
        out = np.zeros((9, self._marker_init.shape[0], 2), dtype=np.float32)
        out[0] = self._marker_init
        out[1:] = np.stack(hist, axis=0)
        return out


def _add_bytes_key_aliases(d: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in d:
            d[key.encode("utf-8")] = d[key]


def build_policy_observation(env, obs: dict, prompt: str, tactile_buf: OnlineTactileBuffer):
    from benchmarks.openpi.env import quat2axisangle, resize_frames_with_padding

    rgbs = []
    for cam_name in ("agentview_cam", "eye_in_hand_cam"):
        cam = env.unwrapped.scene[cam_name]
        rgb = cam.data.output["rgb"]
        rgb = resize_frames_with_padding(rgb, (224, 224, 3), bgr_conversion=False, pad_img=True)
        rgbs.append(rgb)

    eef_pose = np.squeeze(obs["policy"]["eef_pose"].detach().cpu().numpy(), axis=0)
    pos = eef_pose[:3]
    axis_angle = quat2axisangle(eef_pose[3:7].copy())
    gripper_pos = np.squeeze(obs["policy"]["gripper_pos"].detach().cpu().numpy(), axis=0)
    gripper_scalar = np.array([np.asarray(gripper_pos).reshape(-1)[0]], dtype=np.float32)
    state = np.concatenate((pos, axis_angle, gripper_scalar), axis=0).astype(np.float32)

    tactile_buf.update_force(obs)
    try:
        tactile_buf.update_tactile_frames(env, env_id=0)
    except Exception:
        pass
    tactile_buf.update_marker_motion(obs)

    element = {
        "image": _to_uint8_rgb(np.squeeze(rgbs[0], axis=0)),
        "wrist_image": _to_uint8_rgb(np.squeeze(rgbs[1], axis=0)),
        "state": state,
        "observation/image": _to_uint8_rgb(np.squeeze(rgbs[0], axis=0)),
        "observation/wrist_image": _to_uint8_rgb(np.squeeze(rgbs[1], axis=0)),
        "observation/state": state,
        "prompt": prompt,
    }
    tac_img = tactile_buf.get_tactile_image()
    tac_force = tactile_buf.get_force_history()
    tac_mm = tactile_buf.get_marker_motion()
    if tac_img is not None:
        element["tactile_image"] = tac_img
        element["observation/tactile_image"] = tac_img
    if tac_force is not None:
        element["tactile_gripper_force"] = tac_force
        element["observation/tactile_gripper_force"] = tac_force
    if tac_mm is not None:
        element["tactile_marker_motion"] = tac_mm
        element["observation/tactile_marker_motion"] = tac_mm
    _add_bytes_key_aliases(
        element,
        (
            "image",
            "wrist_image",
            "state",
            "prompt",
            "tactile_image",
            "tactile_gripper_force",
            "tactile_marker_motion",
            "observation/image",
            "observation/wrist_image",
            "observation/state",
            "observation/tactile_image",
            "observation/tactile_gripper_force",
            "observation/tactile_marker_motion",
        ),
    )
    return element


def contact_snapshot(env, p6, p4, obs: dict, task: int, trial_id: str, phase: str, step: int, dt: float, stable_run: int) -> dict:
    from benchmarks.common.metrics import compute_contact_force_series_from_lr_forces

    obj_name = TASK_OBJECTS[task]
    f = obs["policy"]["gripper_net_force"][0]
    if f.ndim == 3:
        f = f[-1]
    f_np = f.detach().cpu().numpy().astype(np.float32)
    f_left = f_np[0]
    f_right = f_np[1]
    left_norm = float(np.linalg.norm(f_left))
    right_norm = float(np.linalg.norm(f_right))
    contact_state = (
        "bilateral"
        if left_norm > CONTACT_FORCE_THRESHOLD_N and right_norm > CONTACT_FORCE_THRESHOLD_N
        else ("unilateral" if left_norm > CONTACT_FORCE_THRESHOLD_N or right_norm > CONTACT_FORCE_THRESHOLD_N else "none")
    )
    series = compute_contact_force_series_from_lr_forces(
        np.asarray([f_left], dtype=np.float32),
        np.asarray([f_right], dtype=np.float32),
    )
    left_w = env.scene["left_gripper_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
    right_w = env.scene["right_gripper_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
    left_b = p6.world_point_to_base(p4, env, left_w)
    right_b = p6.world_point_to_base(p4, env, right_w)
    left_o = p6.base_point_to_object(p4, env, obj_name, left_b)
    right_o = p6.base_point_to_object(p4, env, obj_name, right_b)
    mid_o = 0.5 * (left_o + right_o)
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    wrist_o = p6.base_point_to_object(p4, env, obj_name, eef[:3])
    _, obj_q = p4._pose_in_base(env, obj_name)
    wrist_q_o = q_mul(q_inv(obj_q), eef[3:7])
    left_q_w = env.scene["left_gripper_frame"].data.target_quat_w[0, 0].detach().cpu().numpy()
    closing_b = p4._unit(p4._frame_to_base(env, p4._quat_apply_np(left_q_w, np.array([0.0, 0.0, 1.0]))))
    closing_o = p6.vector_base_to_object(p4, env, obj_name, closing_b)
    obj_b, obj_q = p4._pose_in_base(env, obj_name)
    return {
        "trial_id": trial_id,
        "task": task,
        "phase": phase,
        "step": int(step),
        "t_s": float(step * dt),
        "contact_state": contact_state,
        "stable_bilateral_run": stable_run,
        "left_contact_force_norm_N": left_norm,
        "right_contact_force_norm_N": right_norm,
        "measured_squeeze_N": float(series.squeeze[0]),
        "measured_applied_force_N": float(series.external_norm[0]),
        "left_contact_obj_x": float(left_o[0]),
        "left_contact_obj_y": float(left_o[1]),
        "left_contact_obj_z": float(left_o[2]),
        "right_contact_obj_x": float(right_o[0]),
        "right_contact_obj_y": float(right_o[1]),
        "right_contact_obj_z": float(right_o[2]),
        "contact_mid_obj_x": float(mid_o[0]),
        "contact_mid_obj_y": float(mid_o[1]),
        "contact_mid_obj_z": float(mid_o[2]),
        "finger_closing_obj_x": float(closing_o[0]),
        "finger_closing_obj_y": float(closing_o[1]),
        "finger_closing_obj_z": float(closing_o[2]),
        "wrist_obj_x": float(wrist_o[0]),
        "wrist_obj_y": float(wrist_o[1]),
        "wrist_obj_z": float(wrist_o[2]),
        "wrist_obj_qw": float(wrist_q_o[0]),
        "wrist_obj_qx": float(wrist_q_o[1]),
        "wrist_obj_qy": float(wrist_q_o[2]),
        "wrist_obj_qz": float(wrist_q_o[3]),
        "grasp_depth_m": float(mid_o[2]),
        "gripper_width_m": float(np.linalg.norm(left_o - right_o)),
        "object_x": float(obj_b[0]),
        "object_y": float(obj_b[1]),
        "object_z": float(obj_b[2]),
        "object_qw": float(obj_q[0]),
        "object_qx": float(obj_q[1]),
        "object_qy": float(obj_q[2]),
        "object_qz": float(obj_q[3]),
    }


def summarize_contact_rows(rows: list[dict], obj0: np.ndarray, objq0: np.ndarray) -> dict:
    bilateral_rows = [r for r in rows if r.get("contact_state") == "bilateral"]
    stable_rows = [r for r in bilateral_rows if as_int(r, "stable_bilateral_run") >= 3]
    chosen = stable_rows[0] if stable_rows else (bilateral_rows[0] if bilateral_rows else None)
    force_vals = [as_float(r, "measured_squeeze_N") for r in rows]
    object_z = [as_float(r, "object_z") for r in rows]
    lift_dz = (max(object_z) - float(obj0[2])) if object_z else 0.0
    obj_rot = 0.0
    if rows:
        last = rows[-1]
        obj_rot = q_angle(np.array([as_float(last, "object_qw"), as_float(last, "object_qx"), as_float(last, "object_qy"), as_float(last, "object_qz")]), objq0)
    out = {
        "bilateral_contact": int(bool(bilateral_rows)),
        "first_stable_bilateral_step": as_int(chosen, "step", -1) if chosen else "",
        "local_grasp_success": int(bool(bilateral_rows)),
        "short_lift_success": int(lift_dz >= LOCAL_LIFT_Z_M and bool(bilateral_rows)),
        "mean_measured_force_N": safe_mean(force_vals) if force_vals else 0.0,
        "peak_measured_force_N": max(force_vals) if force_vals else 0.0,
        "object_lift_dz_m": float(lift_dz),
        "object_rotation_rad": float(obj_rot),
    }
    if chosen:
        for key in [
            "left_contact_obj_x",
            "left_contact_obj_y",
            "left_contact_obj_z",
            "right_contact_obj_x",
            "right_contact_obj_y",
            "right_contact_obj_z",
            "contact_mid_obj_x",
            "contact_mid_obj_y",
            "contact_mid_obj_z",
            "finger_closing_obj_x",
            "finger_closing_obj_y",
            "finger_closing_obj_z",
            "wrist_obj_x",
            "wrist_obj_y",
            "wrist_obj_z",
            "wrist_obj_qw",
            "wrist_obj_qx",
            "wrist_obj_qy",
            "wrist_obj_qz",
            "grasp_depth_m",
            "gripper_width_m",
        ]:
            out[key] = as_float(chosen, key)
    return out


def write_rollout_telemetry(
    out: Path,
    trial_id: str,
    contact_rows: list[dict],
    force_rows: list[dict],
    policy_chunks: list[np.ndarray],
    chunk_indices: list[int],
) -> tuple[str, str, str]:
    contact_path = out / "P6G1_CONTACT_TELEMETRY" / f"{trial_id}.csv"
    force_path = out / "P6G1_FORCE_TELEMETRY" / f"{trial_id}.csv"
    policy_path = out / "P6G1_POLICY_LOGS" / f"{trial_id}_chunks.npz"
    write_csv(contact_path, contact_rows, CONTACT_FIELDS)
    write_csv(force_path, force_rows, FORCE_FIELDS)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    if policy_chunks:
        payload = {f"chunk_{i:03d}": arr for i, arr in enumerate(policy_chunks)}
        payload["action_indices"] = np.asarray(chunk_indices, dtype=np.int32)
        np.savez_compressed(policy_path, **payload)
    else:
        np.savez_compressed(policy_path, action_indices=np.asarray([], dtype=np.int32))
    return str(contact_path), str(force_path), str(policy_path)


def run_reference_rollout(env, p6, p4, task: int, primitive: dict, trial_meta: dict, dt: float) -> dict:
    import torch

    out = Path(os.environ["P6G1_OUT"])
    obj_name = TASK_OBJECTS[task]
    trial_id = trial_meta["trial_id"]
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    eef_aa = p4._aa(eef[3:7])
    cmd_pos = eef[:3].copy()
    grasp_obj = np.asarray(primitive["desired_object_relative_grasp_transform"]["position_m"], dtype=np.float64)
    grasp = p6.object_point_to_base(p4, env, obj_name, grasp_obj)
    lift = grasp.copy()
    lift[2] += 0.08
    obj0, objq0 = p4._pose_in_base(env, obj_name)
    d_pred = D_OPEN
    contact_rows: list[dict] = []
    force_rows: list[dict] = []
    stable_run = 0
    step = 0
    dropped = 0
    terminated = truncated = False
    error = ""
    try:
        with Timeout(TIMEOUTS_S["rollout"], f"REFERENCE_{trial_id}"):
            phases = [
                ("descend", REFERENCE_DESCEND_STEPS, grasp, 0.0),
                ("close", REFERENCE_CLOSE_STEPS, grasp, FIXED_FORCE_N),
                ("hold", REFERENCE_HOLD_STEPS, grasp, FIXED_FORCE_N),
                ("lift", REFERENCE_LIFT_STEPS, lift, FIXED_FORCE_N),
            ]
            for phase, n_steps, target, f_cmd in phases:
                start = cmd_pos.copy()
                for i in range(n_steps):
                    cmd_pos = p4._interp(start, target, i, n_steps)
                    if phase == "descend":
                        d_pred = D_OPEN
                    action = p6.make_action(p4, cmd_pos, eef_aa, d_pred, f_cmd, env.device)
                    obs, _, term, trunc, _ = env.step(action)
                    step += 1
                    f_sq = p4._f(p4._dbg(env).get("f_sq_meas"), 0.0)
                    if f_cmd > 0:
                        d_pred = p6.force_servo(p4, d_pred, f_sq, FIXED_FORCE_N)
                    snap = contact_snapshot(env, p6, p4, obs, task, trial_id, phase, step, dt, stable_run)
                    if snap["contact_state"] == "bilateral":
                        stable_run += 1
                    else:
                        stable_run = 0
                    snap["stable_bilateral_run"] = stable_run
                    snap.update({k: trial_meta.get(k, "") for k in ("root_seed", "arm", "primitive_id", "policy_repeat")})
                    contact_rows.append(snap)
                    force_rows.append(
                        {
                            "trial_id": trial_id,
                            "task": task,
                            "root_seed": trial_meta.get("root_seed", ""),
                            "arm": "O",
                            "primitive_id": primitive["primitive_id"],
                            "policy_repeat": "",
                            "phase": phase,
                            "step": step,
                            "requested_force_N": FIXED_FORCE_N,
                            "measured_squeeze_N": snap["measured_squeeze_N"],
                            "measured_applied_force_N": snap["measured_applied_force_N"],
                            "executed_fLz": FORCE_SLOT_HALF_N if f_cmd > 0 else 0.0,
                            "executed_fRz": FORCE_SLOT_HALF_N if f_cmd > 0 else 0.0,
                            "gripper_cmd": float(d_pred),
                        }
                    )
                    try:
                        dropped = max(dropped, int(bool(env.termination_manager.get_term("object_1_dropped")[0].item())))
                    except Exception:
                        pass
                    terminated = bool(term[0].item())
                    truncated = bool(trunc[0].item())
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
    except Exception as exc:
        error = repr(exc)

    contact_path, force_path, policy_path = write_rollout_telemetry(out, trial_id, contact_rows, force_rows, [], [])
    summary = summarize_contact_rows(contact_rows, obj0, objq0)
    summary.update(
        {
            "trial_id": trial_id,
            "task": task,
            "object": obj_name,
            "root_seed": trial_meta["root_seed"],
            "arm": "O",
            "primitive_id": primitive["primitive_id"],
            "source_grasp_label": primitive["source_grasp_label"],
            "policy_repeat": "",
            "policy_repeat_seed": "",
            "prompt": "",
            "requested_force_N": FIXED_FORCE_N,
            "state_parity": trial_meta.get("state_parity", 0),
            "staging_success": trial_meta.get("staging_success", 0),
            "handoff_valid": trial_meta.get("handoff_valid", 0),
            "drop": dropped,
            "contact_loss": 0,
            "collision": "",
            "policy_timeout": 0,
            "terminated": int(terminated),
            "truncated": int(truncated),
            "steps": step,
            "num_policy_chunks": 0,
            "mean_original_vla_force_slot_N": "",
            "peak_original_vla_force_slot_N": "",
            "contact_telemetry_path": contact_path,
            "force_telemetry_path": force_path,
            "policy_log_path": policy_path,
            "error": error,
        }
    )
    return summary


def run_vla_rollout(env, p6, p4, client, task: int, primitive: dict | None, trial_meta: dict, dt: float) -> dict:
    import torch

    out = Path(os.environ["P6G1_OUT"])
    obj_name = TASK_OBJECTS[task]
    trial_id = trial_meta["trial_id"]
    prompt = TASK_INSTRUCTIONS[task]
    obs = env.observation_manager.compute()
    obj0, objq0 = p4._pose_in_base(env, obj_name)
    tactile_buf = OnlineTactileBuffer(tactile_output_type="tactile_rgb")
    contact_rows: list[dict] = []
    force_rows: list[dict] = []
    policy_chunks: list[np.ndarray] = []
    chunk_indices: list[int] = []
    original_force_squeeze: list[float] = []
    stable_run = 0
    step = 0
    terminated = truncated = False
    dropped = 0
    timeout = 0
    error = ""
    lift_hold = 0
    local_done = False
    try:
        from benchmarks.common.metrics import compute_contact_force_series_from_lr_forces

        with Timeout(TIMEOUTS_S["rollout"], f"VLA_{trial_id}"):
            for action_idx in range(VLA_MAX_INFERENCE_STEPS):
                element = build_policy_observation(env, obs, prompt, tactile_buf)
                action_chunk = np.asarray(client.infer(element)["actions"], dtype=np.float32)
                policy_chunks.append(action_chunk.copy())
                chunk_indices.append(action_idx)
                if action_chunk.shape[0] < VLA_REPLAN_STEPS or action_chunk.shape[1] < 13:
                    raise RuntimeError(f"Policy returned action_chunk shape {action_chunk.shape}, expected at least ({VLA_REPLAN_STEPS},13)")
                raw = action_chunk[:VLA_REPLAN_STEPS, :13].astype(np.float32)
                executed = raw.copy()
                executed[:, 7:13] = 0.0
                executed[:, 9] = FORCE_SLOT_HALF_N
                executed[:, 12] = FORCE_SLOT_HALF_N
                action = torch.from_numpy(executed).float()
                for i in range(action.shape[0]):
                    obs, _, term, trunc, _ = env.step(action[i].reshape([1, -1]).to(env.device))
                    step += 1
                    raw_i = raw[i]
                    pred_series = compute_contact_force_series_from_lr_forces(
                        np.asarray([raw_i[7:10]], dtype=np.float32),
                        np.asarray([raw_i[10:13]], dtype=np.float32),
                    )
                    original_force_squeeze.append(float(pred_series.squeeze[0]))
                    snap = contact_snapshot(env, p6, p4, obs, task, trial_id, "vla", step, dt, stable_run)
                    if snap["contact_state"] == "bilateral":
                        stable_run += 1
                    else:
                        stable_run = 0
                    snap["stable_bilateral_run"] = stable_run
                    snap.update({k: trial_meta.get(k, "") for k in ("root_seed", "arm", "primitive_id", "policy_repeat")})
                    contact_rows.append(snap)
                    force_rows.append(
                        {
                            "trial_id": trial_id,
                            "task": task,
                            "root_seed": trial_meta.get("root_seed", ""),
                            "arm": trial_meta["arm"],
                            "primitive_id": trial_meta.get("primitive_id", ""),
                            "policy_repeat": trial_meta.get("policy_repeat", ""),
                            "phase": "vla",
                            "step": step,
                            "requested_force_N": FIXED_FORCE_N,
                            "measured_squeeze_N": snap["measured_squeeze_N"],
                            "measured_applied_force_N": snap["measured_applied_force_N"],
                            "original_vla_fLx": float(raw_i[7]),
                            "original_vla_fLy": float(raw_i[8]),
                            "original_vla_fLz": float(raw_i[9]),
                            "original_vla_fRx": float(raw_i[10]),
                            "original_vla_fRy": float(raw_i[11]),
                            "original_vla_fRz": float(raw_i[12]),
                            "executed_fLx": 0.0,
                            "executed_fLy": 0.0,
                            "executed_fLz": FORCE_SLOT_HALF_N,
                            "executed_fRx": 0.0,
                            "executed_fRy": 0.0,
                            "executed_fRz": FORCE_SLOT_HALF_N,
                            "gripper_cmd": float(executed[i, 6]),
                        }
                    )
                    try:
                        dropped = max(dropped, int(bool(env.termination_manager.get_term("object_1_dropped")[0].item())))
                    except Exception:
                        pass
                    obj_now, _ = p4._pose_in_base(env, obj_name)
                    if float(obj_now[2] - obj0[2]) >= LOCAL_LIFT_Z_M and stable_run >= 1:
                        lift_hold += 1
                    else:
                        lift_hold = 0
                    if lift_hold >= LOCAL_LIFT_HOLD_STEPS:
                        local_done = True
                    terminated = bool(term[0].item())
                    truncated = bool(trunc[0].item())
                    if local_done or terminated or truncated or dropped:
                        break
                if local_done or terminated or truncated or dropped:
                    break
            if not local_done and not terminated and not truncated:
                timeout = 1
    except Exception as exc:
        error = repr(exc)

    contact_path, force_path, policy_path = write_rollout_telemetry(out, trial_id, contact_rows, force_rows, policy_chunks, chunk_indices)
    summary = summarize_contact_rows(contact_rows, obj0, objq0)
    summary.update(
        {
            "trial_id": trial_id,
            "task": task,
            "object": obj_name,
            "root_seed": trial_meta["root_seed"],
            "arm": trial_meta["arm"],
            "primitive_id": primitive["primitive_id"] if primitive else "",
            "source_grasp_label": primitive["source_grasp_label"] if primitive else "",
            "policy_repeat": trial_meta["policy_repeat"],
            "policy_repeat_seed": trial_meta["policy_repeat_seed"],
            "prompt": prompt,
            "requested_force_N": FIXED_FORCE_N,
            "state_parity": trial_meta.get("state_parity", 0),
            "staging_success": trial_meta.get("staging_success", 1 if primitive is None else 0),
            "handoff_valid": trial_meta.get("handoff_valid", 1 if primitive is None else 0),
            "drop": dropped,
            "contact_loss": int(bool(summary["bilateral_contact"]) and not contact_rows[-1]["contact_state"] == "bilateral") if contact_rows else 0,
            "collision": "",
            "policy_timeout": timeout,
            "terminated": int(terminated),
            "truncated": int(truncated),
            "steps": step,
            "num_policy_chunks": len(policy_chunks),
            "mean_original_vla_force_slot_N": safe_mean(original_force_squeeze) if original_force_squeeze else "",
            "peak_original_vla_force_slot_N": max(original_force_squeeze) if original_force_squeeze else "",
            "contact_telemetry_path": contact_path,
            "force_telemetry_path": force_path,
            "policy_log_path": policy_path,
            "error": error,
        }
    )
    return summary


def run_worker() -> int:
    from isaaclab.app import AppLauncher

    out = Path(os.environ["P6G1_OUT"])
    task = int(os.environ["P6G1_TASK_ID"])
    server_host = os.environ["P6G1_SERVER_HOST"]
    server_port = int(os.environ["P6G1_SERVER_PORT"])
    roots_per_task = int(os.environ.get("P6G1_ROOTS_PER_TASK", str(len(FRESH_ROOTS))))
    roots = FRESH_ROOTS[:roots_per_task]
    task_dir = out / f"task{task}"
    task_dir.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        from openpi_client import websocket_client_policy as _websocket_client_policy

        p6 = import_p6g0()
        p4 = p6.import_p4_probe(task)
        p6.imported_p4 = p4
        setup_task_objects(TASK_SUITE, task)
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 30.0
        try:
            cfg.sim.physx.enable_ccd = True
        except Exception:
            pass
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        obj_name = TASK_OBJECTS[task]
        library = read_json(out / "P6G1_PRIMITIVE_LIBRARY.json")
        primitives = [primitive_for(library, task, pid) for pid in PRIMITIVE_IDS]
        client = _websocket_client_policy.WebsocketClientPolicy(server_host, server_port)
        write_json(task_dir / "policy_server_metadata.json", client.get_server_metadata())

        root_rows: list[dict] = dedupe_rows(
            read_csv(task_dir / "root_manifest.csv"),
            lambda r: (r.get("task", ""), r.get("root_seed", "")),
        )
        parity_rows: list[dict] = dedupe_rows(read_csv(task_dir / "parity.csv"), lambda r: r.get("trial_id", ""))
        staging_rows: list[dict] = dedupe_rows(read_csv(task_dir / "staging.csv"), lambda r: r.get("trial_id", ""))
        rollout_rows: list[dict] = dedupe_rows(read_csv(task_dir / "rollouts.csv"), lambda r: r.get("trial_id", ""))
        existing_rollouts = {r.get("trial_id", "") for r in rollout_rows}

        for root_index, root_seed in enumerate(roots):
            obs, _ = env.reset(seed=int(root_seed))
            obs = settle_root_before_hash(env, p6, p4, ROOT_SETTLE_STEPS)
            root_state = env.scene.get_state(is_relative=True)
            h0 = root_state_hash(p6, env)
            obj0, objq0 = p4._pose_in_base(env, obj_name)
            root_row = {
                "task": task,
                "object": obj_name,
                "root_index": root_index,
                "root_seed": root_seed,
                "root_state_hash": h0,
                "object_pose_base_x": float(obj0[0]),
                "object_pose_base_y": float(obj0[1]),
                "object_pose_base_z": float(obj0[2]),
                "object_pose_base_qw": float(objq0[0]),
                "object_pose_base_qx": float(objq0[1]),
                "object_pose_base_qy": float(objq0[2]),
                "object_pose_base_qz": float(objq0[3]),
                "source": f"fresh_env_reset_seed_no_hdf5_replay_after_{ROOT_SETTLE_STEPS}_settle_steps",
            }
            if str(root_seed) not in {str(r.get("root_seed", "")) for r in root_rows}:
                root_rows.append(root_row)
                append_csv(task_dir / "root_manifest.csv", root_row, ROOT_MANIFEST_FIELDS)

            for primitive in primitives:
                for arm, repeats in (("O", [""]), ("V", POLICY_REPEATS)):
                    for repeat in repeats:
                        trial_id = (
                            f"p6g1_t{task}_s{root_seed}_{arm}_{primitive['primitive_id']}"
                            if arm == "O"
                            else f"p6g1_t{task}_s{root_seed}_{arm}_{primitive['primitive_id']}_r{repeat}"
                        )
                        if trial_id in existing_rollouts:
                            continue
                        with Timeout(TIMEOUTS_S["restore"], f"RESTORE_{trial_id}"):
                            env.reset_to(root_state, torch.tensor([0], device=env.device), is_relative=True)
                        restore_hash = root_state_hash(p6, env)
                        state_parity = int(restore_hash == h0)
                        before_obj, before_q = p4._pose_in_base(env, obj_name)
                        stage_row, obs_after_stage = stage_to_pregrasp(env, p6, p4, task, primitive, trial_id, dt)
                        stage_row.update(
                            {
                                "root_seed": root_seed,
                                "arm": arm,
                                "policy_repeat": repeat,
                            }
                        )
                        staging_rows = [r for r in staging_rows if r.get("trial_id") != trial_id]
                        staging_rows.append(stage_row)
                        append_csv(task_dir / "staging.csv", stage_row, STAGING_FIELDS)
                        handoff_obj, handoff_q = p4._pose_in_base(env, obj_name)
                        parity_row = {
                            "trial_id": trial_id,
                            "task": task,
                            "root_seed": root_seed,
                            "arm": arm,
                            "primitive_id": primitive["primitive_id"],
                            "policy_repeat": repeat,
                            "root_state_hash": h0,
                            "restore_state_hash": restore_hash,
                            "state_parity": state_parity,
                            "object_pose_before_stage_x": float(before_obj[0]),
                            "object_pose_before_stage_y": float(before_obj[1]),
                            "object_pose_before_stage_z": float(before_obj[2]),
                            "object_pose_handoff_x": float(handoff_obj[0]),
                            "object_pose_handoff_y": float(handoff_obj[1]),
                            "object_pose_handoff_z": float(handoff_obj[2]),
                            "object_disturbance_m": float(np.linalg.norm(handoff_obj - before_obj)),
                            "object_rotation_disturbance_rad": q_angle(handoff_q, before_q),
                        }
                        parity_rows = [r for r in parity_rows if r.get("trial_id") != trial_id]
                        parity_rows.append(parity_row)
                        append_csv(task_dir / "parity.csv", parity_row, PARITY_FIELDS)
                        trial_meta = {
                            "trial_id": trial_id,
                            "root_seed": root_seed,
                            "arm": arm,
                            "primitive_id": primitive["primitive_id"],
                            "policy_repeat": repeat,
                            "policy_repeat_seed": f"server_stream_repeat_{repeat}" if arm == "V" else "",
                            "state_parity": state_parity,
                            "staging_success": stage_row["staging_success"],
                            "handoff_valid": stage_row["handoff_valid"],
                        }
                        if not state_parity or not int(stage_row["handoff_valid"]):
                            rollout = {
                                "trial_id": trial_id,
                                "task": task,
                                "object": obj_name,
                                "root_seed": root_seed,
                                "arm": arm,
                                "primitive_id": primitive["primitive_id"],
                                "source_grasp_label": primitive["source_grasp_label"],
                                "policy_repeat": repeat,
                                "policy_repeat_seed": trial_meta["policy_repeat_seed"],
                                "prompt": TASK_INSTRUCTIONS[task] if arm == "V" else "",
                                "requested_force_N": FIXED_FORCE_N,
                                "state_parity": state_parity,
                                "staging_success": stage_row["staging_success"],
                                "handoff_valid": stage_row["handoff_valid"],
                                "bilateral_contact": 0,
                                "local_grasp_success": 0,
                                "short_lift_success": 0,
                                "drop": 0,
                                "policy_timeout": 0,
                                "steps": 0,
                                "num_policy_chunks": 0,
                                "error": "state parity or handoff invalid",
                            }
                        elif arm == "O":
                            rollout = run_reference_rollout(env, p6, p4, task, primitive, trial_meta, dt)
                        else:
                            rollout = run_vla_rollout(env, p6, p4, client, task, primitive, trial_meta, dt)
                        rollout_rows = [r for r in rollout_rows if r.get("trial_id") != trial_id]
                        rollout_rows.append(rollout)
                        existing_rollouts.add(trial_id)
                        append_csv(task_dir / "rollouts.csv", rollout, ROLLOUT_FIELDS)
                        append_csv(task_dir / "actual_contacts.csv", {k: rollout.get(k, "") for k in rollout}, ROLLOUT_FIELDS)

            for repeat in POLICY_REPEATS:
                trial_id = f"p6g1_t{task}_s{root_seed}_R_raw_r{repeat}"
                if trial_id in existing_rollouts:
                    continue
                with Timeout(TIMEOUTS_S["restore"], f"RESTORE_{trial_id}"):
                    env.reset_to(root_state, torch.tensor([0], device=env.device), is_relative=True)
                restore_hash = root_state_hash(p6, env)
                state_parity = int(restore_hash == h0)
                before_obj, before_q = p4._pose_in_base(env, obj_name)
                parity_row = {
                    "trial_id": trial_id,
                    "task": task,
                    "root_seed": root_seed,
                    "arm": "R",
                    "primitive_id": "",
                    "policy_repeat": repeat,
                    "root_state_hash": h0,
                    "restore_state_hash": restore_hash,
                    "state_parity": state_parity,
                    "object_pose_before_stage_x": float(before_obj[0]),
                    "object_pose_before_stage_y": float(before_obj[1]),
                    "object_pose_before_stage_z": float(before_obj[2]),
                    "object_pose_handoff_x": float(before_obj[0]),
                    "object_pose_handoff_y": float(before_obj[1]),
                    "object_pose_handoff_z": float(before_obj[2]),
                    "object_disturbance_m": 0.0,
                    "object_rotation_disturbance_rad": 0.0,
                }
                parity_rows = [r for r in parity_rows if r.get("trial_id") != trial_id]
                parity_rows.append(parity_row)
                append_csv(task_dir / "parity.csv", parity_row, PARITY_FIELDS)
                if state_parity:
                    trial_meta = {
                        "trial_id": trial_id,
                        "root_seed": root_seed,
                        "arm": "R",
                        "primitive_id": "",
                        "policy_repeat": repeat,
                        "policy_repeat_seed": f"server_stream_repeat_{repeat}",
                        "state_parity": state_parity,
                        "staging_success": 1,
                        "handoff_valid": 1,
                    }
                    rollout = run_vla_rollout(env, p6, p4, client, task, None, trial_meta, dt)
                else:
                    rollout = {
                        "trial_id": trial_id,
                        "task": task,
                        "object": obj_name,
                        "root_seed": root_seed,
                        "arm": "R",
                        "primitive_id": "",
                        "policy_repeat": repeat,
                        "policy_repeat_seed": f"server_stream_repeat_{repeat}",
                        "prompt": TASK_INSTRUCTIONS[task],
                        "requested_force_N": FIXED_FORCE_N,
                        "state_parity": state_parity,
                        "staging_success": 1,
                        "handoff_valid": 1,
                        "bilateral_contact": 0,
                        "local_grasp_success": 0,
                        "short_lift_success": 0,
                        "drop": 0,
                        "policy_timeout": 0,
                        "steps": 0,
                        "num_policy_chunks": 0,
                        "error": "state parity invalid",
                    }
                rollout_rows = [r for r in rollout_rows if r.get("trial_id") != trial_id]
                rollout_rows.append(rollout)
                existing_rollouts.add(trial_id)
                append_csv(task_dir / "rollouts.csv", rollout, ROLLOUT_FIELDS)
                append_csv(task_dir / "actual_contacts.csv", {k: rollout.get(k, "") for k in rollout}, list(rollout.keys()))

        root_rows = dedupe_rows(root_rows, lambda r: (r.get("task", ""), r.get("root_seed", "")))
        parity_rows = dedupe_rows(parity_rows, lambda r: r.get("trial_id", ""))
        staging_rows = dedupe_rows(staging_rows, lambda r: r.get("trial_id", ""))
        rollout_rows = dedupe_rows(rollout_rows, lambda r: r.get("trial_id", ""))
        write_csv(task_dir / "root_manifest.csv", root_rows, ROOT_MANIFEST_FIELDS)
        write_csv(task_dir / "parity.csv", parity_rows, PARITY_FIELDS)
        write_csv(task_dir / "staging.csv", staging_rows, STAGING_FIELDS)
        write_csv(task_dir / "rollouts.csv", rollout_rows, ROLLOUT_FIELDS)
        write_csv(task_dir / "actual_contacts.csv", rollout_rows, ROLLOUT_FIELDS)
        write_json(task_dir / "result.json", {"task": task, "status": "COMPLETE", "rollouts": len(rollout_rows)})
        return 0
    except Exception as exc:
        write_json(task_dir / "error.json", {"task": task, "error": repr(exc), "trace": traceback.format_exc()})
        return 1
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass


def collect_task_csv(out: Path, name: str) -> list[dict]:
    rows: list[dict] = []
    for task in TASKS:
        rows += read_csv(out / f"task{task}" / name)
    return rows


def reference_clusters(out: Path, rollouts: list[dict]) -> tuple[list[dict], dict]:
    clusters: list[dict] = []
    cluster_map: dict[tuple[int, str], dict] = {}
    for task in TASKS:
        library = read_json(out / "P6G1_PRIMITIVE_LIBRARY.json")
        for pid in PRIMITIVE_IDS:
            primitive = primitive_for(library, task, pid)
            rows = [
                r
                for r in rollouts
                if int(r.get("task", -1)) == task and r.get("arm") == "O" and r.get("primitive_id") == pid
            ]
            valid = [
                r
                for r in rows
                if r.get("contact_mid_obj_x", "") not in ("", None) and as_int(r, "bilateral_contact") == 1
            ]
            centers = np.asarray(
                [[as_float(r, "contact_mid_obj_x"), as_float(r, "contact_mid_obj_y"), as_float(r, "contact_mid_obj_z")] for r in valid],
                dtype=np.float64,
            )
            depths = np.asarray([as_float(r, "grasp_depth_m") for r in valid], dtype=np.float64)
            quats = np.asarray(
                [[as_float(r, "wrist_obj_qw"), as_float(r, "wrist_obj_qx"), as_float(r, "wrist_obj_qy"), as_float(r, "wrist_obj_qz")] for r in valid],
                dtype=np.float64,
            )
            if len(centers):
                centroid = np.mean(centers, axis=0)
                covariance = np.cov(centers.T).tolist() if len(centers) > 1 else np.zeros((3, 3)).tolist()
                ref_q = q_normalize(np.mean(np.asarray([q_normalize(q) for q in quats]), axis=0))
                depth = float(np.mean(depths))
                within = float(np.mean(np.linalg.norm(centers - centroid, axis=1)) / max(primitive["usable_extent_Lg_m"], 1e-9))
            else:
                centroid = np.full(3, np.nan)
                covariance = np.full((3, 3), np.nan).tolist()
                ref_q = np.array([np.nan, np.nan, np.nan, np.nan])
                depth = float("nan")
                within = float("nan")
            row = {
                "task": task,
                "object": TASK_OBJECTS[task],
                "primitive_id": pid,
                "source_grasp_label": primitive["source_grasp_label"],
                "n_reference_rollouts": len(rows),
                "n_valid_contacts": len(valid),
                "reference_lift_success_rate": safe_mean([as_int(r, "short_lift_success") for r in rows]) if rows else 0.0,
                "centroid_contact_mid_obj_x": float(centroid[0]),
                "centroid_contact_mid_obj_y": float(centroid[1]),
                "centroid_contact_mid_obj_z": float(centroid[2]),
                "centroid_wrist_obj_qw": float(ref_q[0]),
                "centroid_wrist_obj_qx": float(ref_q[1]),
                "centroid_wrist_obj_qy": float(ref_q[2]),
                "centroid_wrist_obj_qz": float(ref_q[3]),
                "centroid_grasp_depth_m": depth,
                "within_reference_cluster_std_norm_Lg": within,
                "contact_center_covariance_json": json.dumps(covariance),
            }
            clusters.append(row)
            cluster_map[(task, pid)] = row
    return clusters, cluster_map


def classify_rollout(row: dict, cluster_map: dict, library: dict) -> dict:
    task = int(row["task"])
    if row.get("contact_mid_obj_x", "") in ("", None) or as_int(row, "bilateral_contact") == 0:
        return {
            "trial_id": row["trial_id"],
            "task": task,
            "arm": row["arm"],
            "requested_primitive": row.get("primitive_id", ""),
            "realized_cluster": "NONE",
            "classification_accuracy": 0 if row["arm"] == "V" else "",
            "distance_to_nearest": "",
            "distance_margin": "",
            "confidence": 0.0,
        }
    p = np.asarray([as_float(row, "contact_mid_obj_x"), as_float(row, "contact_mid_obj_y"), as_float(row, "contact_mid_obj_z")])
    q = np.asarray([as_float(row, "wrist_obj_qw"), as_float(row, "wrist_obj_qx"), as_float(row, "wrist_obj_qy"), as_float(row, "wrist_obj_qz")])
    depth = as_float(row, "grasp_depth_m")
    distances = []
    for pid in PRIMITIVE_IDS:
        cluster = cluster_map.get((task, pid), {})
        if cluster.get("centroid_contact_mid_obj_x", "") in ("", None):
            continue
        c = np.asarray(
            [
                as_float(cluster, "centroid_contact_mid_obj_x", float("nan")),
                as_float(cluster, "centroid_contact_mid_obj_y", float("nan")),
                as_float(cluster, "centroid_contact_mid_obj_z", float("nan")),
            ]
        )
        cq = np.asarray(
            [
                as_float(cluster, "centroid_wrist_obj_qw", float("nan")),
                as_float(cluster, "centroid_wrist_obj_qx", float("nan")),
                as_float(cluster, "centroid_wrist_obj_qy", float("nan")),
                as_float(cluster, "centroid_wrist_obj_qz", float("nan")),
            ]
        )
        if not np.isfinite(c).all() or not np.isfinite(cq).all():
            continue
        primitive = primitive_for(library, task, pid)
        lg = max(float(primitive["usable_extent_Lg_m"]), 1e-9)
        dp = float(np.linalg.norm(p - c) / lg)
        dR = q_angle(q, cq)
        dd = abs(depth - as_float(cluster, "centroid_grasp_depth_m")) / lg
        d = DIST_LAMBDA_P * dp + DIST_LAMBDA_R * dR + DIST_LAMBDA_D * dd
        distances.append((d, pid, dp, dR, dd))
    distances.sort()
    if not distances:
        realized = "NONE"
        d0 = margin = ""
        conf = 0.0
    else:
        d0, realized, _, _, _ = distances[0]
        margin = (distances[1][0] - d0) if len(distances) > 1 else float("inf")
        conf = float(margin / max(distances[1][0], 1e-9)) if len(distances) > 1 else 1.0
    requested = row.get("primitive_id", "")
    return {
        "trial_id": row["trial_id"],
        "task": task,
        "root_seed": row.get("root_seed", ""),
        "arm": row["arm"],
        "requested_primitive": requested,
        "realized_cluster": realized,
        "classification_accuracy": int(realized == requested) if row["arm"] == "V" else "",
        "distance_to_nearest": d0,
        "distance_margin": margin,
        "confidence": conf,
    }


def mutual_information(requests: list[str], realized: list[str]) -> float:
    if not requests:
        return 0.0
    req_vals = sorted(set(requests))
    real_vals = sorted(set(realized))
    n = float(len(requests))
    mi = 0.0
    for r in req_vals:
        pr = requests.count(r) / n
        for z in real_vals:
            pz = realized.count(z) / n
            prz = sum(1 for a, b in zip(requests, realized) if a == r and b == z) / n
            if prz > 0 and pr > 0 and pz > 0:
                mi += prz * math.log(prz / (pr * pz), 2)
    return float(mi)


def aggregate_and_classify(out: Path, worker_rows: list[dict]) -> dict:
    library = read_json(out / "P6G1_PRIMITIVE_LIBRARY.json")
    root_manifest = collect_task_csv(out, "root_manifest.csv")
    parity = collect_task_csv(out, "parity.csv")
    staging = collect_task_csv(out, "staging.csv")
    rollouts = collect_task_csv(out, "rollouts.csv")
    write_csv(out / "P6G1_ROOT_MANIFEST.csv", root_manifest, ROOT_MANIFEST_FIELDS)
    write_csv(out / "P6G1_STATE_PARITY.csv", parity, PARITY_FIELDS)
    write_csv(out / "P6G1_STAGING_RESULTS.csv", staging, STAGING_FIELDS)
    write_csv(out / "P6G1_ROLLOUT_RESULTS.csv", rollouts, ROLLOUT_FIELDS)
    write_csv(out / "P6G1_ACTUAL_CONTACTS.csv", rollouts, ROLLOUT_FIELDS)

    clusters, cluster_map = reference_clusters(out, rollouts)
    write_csv(out / "P6G1_REFERENCE_GRASP_CLUSTERS.csv", clusters)

    mode_rows = [classify_rollout(r, cluster_map, library) for r in rollouts if r.get("arm") in {"V", "R"}]
    write_csv(out / "P6G1_MODE_CLASSIFICATION.csv", mode_rows)

    task_rows = []
    canon_rows = []
    interface_stats: dict[str, dict] = {}
    task_class_by_task: dict[int, str] = {}
    mi_by_task: dict[int, float] = {}
    collapse_by_task: dict[int, bool] = {}
    for task in TASKS:
        task_v = [r for r in rollouts if int(r.get("task", -1)) == task and r.get("arm") == "V"]
        task_modes = [m for m in mode_rows if int(m.get("task", -1)) == task and m.get("arm") == "V"]
        task_r_modes = [m for m in mode_rows if int(m.get("task", -1)) == task and m.get("arm") == "R"]
        requests = [m["requested_primitive"] for m in task_modes if m["realized_cluster"] != "NONE"]
        realized = [m["realized_cluster"] for m in task_modes if m["realized_cluster"] != "NONE"]
        mi = mutual_information(requests, realized)
        mi_by_task[task] = mi
        realized_counts = {pid: realized.count(pid) for pid in PRIMITIVE_IDS}
        total_realized = max(1, sum(realized_counts.values()))
        dominant_fraction = max(realized_counts.values() or [0]) / total_realized
        raw_realized = [m["realized_cluster"] for m in task_r_modes]
        raw_dist = {pid: raw_realized.count(pid) for pid in PRIMITIVE_IDS + ["NONE"]}
        collapse = dominant_fraction > 0.70
        collapse_by_task[task] = collapse

        ref_task = [c for c in clusters if int(c["task"]) == task]
        centers = {
            c["primitive_id"]: np.asarray(
                [
                    as_float(c, "centroid_contact_mid_obj_x", float("nan")),
                    as_float(c, "centroid_contact_mid_obj_y", float("nan")),
                    as_float(c, "centroid_contact_mid_obj_z", float("nan")),
                ]
            )
            for c in ref_task
        }
        seps = []
        for i, pa in enumerate(PRIMITIVE_IDS):
            for pb in PRIMITIVE_IDS[i + 1 :]:
                if np.isfinite(centers.get(pa, np.full(3, np.nan))).all() and np.isfinite(centers.get(pb, np.full(3, np.nan))).all():
                    lg = max(float(primitive_for(library, task, pa)["usable_extent_Lg_m"]), 1e-9)
                    seps.append(float(np.linalg.norm(centers[pa] - centers[pb]) / lg))
        min_sep = min(seps) if seps else 0.0
        within_vals = [as_float(c, "within_reference_cluster_std_norm_Lg", float("nan")) for c in ref_task]
        within = safe_mean(within_vals)
        qualified_modes = 0
        mode_stats = {}
        for pid in PRIMITIVE_IDS:
            rows_pid = [r for r in task_v if r.get("primitive_id") == pid]
            modes_pid = [m for m in task_modes if m.get("requested_primitive") == pid]
            staging_pid = [
                r
                for r in staging
                if int(r.get("task", -1)) == task and r.get("arm") == "V" and r.get("primitive_id") == pid
            ]
            ik_rate = safe_mean([as_int(r, "staging_success") for r in staging_pid]) if staging_pid else 0.0
            acc = safe_mean([as_int(m, "classification_accuracy") for m in modes_pid]) if modes_pid else 0.0
            bilat = safe_mean([as_int(r, "bilateral_contact") for r in rows_pid]) if rows_pid else 0.0
            lift = safe_mean([as_int(r, "short_lift_success") for r in rows_pid]) if rows_pid else 0.0
            qualifies = bool(ik_rate >= 0.95 and acc >= 0.80 and bilat >= 0.80 and lift >= 0.80)
            qualified_modes += int(qualifies)
            mode_stats[pid] = {"ik": ik_rate, "accuracy": acc, "bilateral": bilat, "lift": lift, "qualifies": qualifies}
            interface_stats[f"task{task}_{pid}"] = {
                "expected_realized_cluster": pid,
                "ik_staging_success_rate": ik_rate,
                "requested_mode_accuracy": acc,
                "bilateral_contact_rate": bilat,
                "short_lift_success_rate": lift,
                "classification": "qualified" if qualifies else "not_qualified",
            }

        distinct = bool(min_sep > max(within, 0.0) and min_sep >= MIN_DISTINCT_SEPARATION_NORM)
        if qualified_modes == 3 and distinct and not collapse:
            task_cls = "THREE_MODE_QUALIFIED"
        elif qualified_modes >= 2 and distinct:
            task_cls = "TWO_MODE_PARTIAL"
        elif collapse:
            task_cls = "CANONICAL_GRASP_COLLAPSE"
        else:
            task_cls = "NOT_QUALIFIED"
        task_class_by_task[task] = task_cls
        task_rows.append(
            {
                "task": task,
                "object": TASK_OBJECTS[task],
                "G0_accuracy": mode_stats["G0"]["accuracy"],
                "G0_lift_rate": mode_stats["G0"]["lift"],
                "G0_qualified": int(mode_stats["G0"]["qualifies"]),
                "G1_accuracy": mode_stats["G1"]["accuracy"],
                "G1_lift_rate": mode_stats["G1"]["lift"],
                "G1_qualified": int(mode_stats["G1"]["qualifies"]),
                "G2_accuracy": mode_stats["G2"]["accuracy"],
                "G2_lift_rate": mode_stats["G2"]["lift"],
                "G2_qualified": int(mode_stats["G2"]["qualifies"]),
                "num_qualified_modes": qualified_modes,
                "dominant_cluster_fraction": dominant_fraction,
                "mutual_information_bits": mi,
                "between_mode_min_sep_norm_Lg": min_sep,
                "within_mode_variation_norm_Lg": within,
                "raw_vla_distribution_json": json.dumps(raw_dist, sort_keys=True),
                "classification": task_cls,
            }
        )
        canon_rows.append(
            {
                "task": task,
                "object": TASK_OBJECTS[task],
                "requested_primitive_affects_realized_grasp": int(mi > 0.05 and not collapse),
                "mutual_information_bits": mi,
                "dominant_cluster_fraction": dominant_fraction,
                "collapse_detected": int(collapse),
                "realized_distribution_json": json.dumps(realized_counts, sort_keys=True),
                "raw_vla_distribution_json": json.dumps(raw_dist, sort_keys=True),
            }
        )

    write_csv(out / "P6G1_TASK_CLASSIFICATIONS.csv", task_rows)
    write_csv(out / "P6G1_CANONICALIZATION_ANALYSIS.csv", canon_rows)
    write_agent_interface(out, library, interface_stats)

    worker_failed = any(int(r.get("returncode", 1)) != 0 for r in worker_rows)
    vla_errors = [r for r in rollouts if r.get("arm") in {"V", "R"} and r.get("error")]
    vla_attempts = [r for r in rollouts if r.get("arm") in {"V", "R"}]
    staging_rate = safe_mean([as_int(r, "staging_success") for r in staging]) if staging else 0.0
    if worker_failed:
        primary = "P6G1_INCONCLUSIVE_SYSTEM_FAILURE"
    elif vla_attempts and len(vla_errors) == len(vla_attempts):
        primary = "P6G1_FROZEN_VLA_INVOCATION_INTEGRATION_FAILED"
    elif staging_rate < 0.95:
        primary = "P6G1_PRIMITIVE_STAGING_NOT_RELIABLE"
    elif all(task_class_by_task.get(t) == "THREE_MODE_QUALIFIED" for t in TASKS):
        primary = "P6G1_PRIMITIVE_IK_VLA_THREE_MODE_REALIZATION_QUALIFIED"
    elif all(task_class_by_task.get(t) in {"THREE_MODE_QUALIFIED", "TWO_MODE_PARTIAL"} for t in TASKS) and any(
        task_class_by_task.get(t) == "THREE_MODE_QUALIFIED" for t in TASKS
    ):
        primary = "P6G1_PRIMITIVE_IK_VLA_ACTION_SPACE_QUALIFIED"
    elif any(task_class_by_task.get(t) in {"THREE_MODE_QUALIFIED", "TWO_MODE_PARTIAL"} for t in TASKS):
        primary = "P6G1_PRIMITIVE_IK_VLA_REALIZATION_PARTIAL"
    elif all(collapse_by_task.get(t, False) for t in TASKS):
        primary = "P6G1_VLA_CANONICAL_GRASP_COLLAPSE"
    else:
        primary = "P6G1_PRIMITIVE_IK_VLA_REALIZATION_PARTIAL"

    verdict = {
        "status": "COMPLETE" if not worker_failed else "SYSTEM_FAILURE",
        "primary_classification": primary,
        "learned_method_change": "NONE",
        "system_interface_change": "PRIMITIVE_IK_STAGING_BEFORE_FROZEN_VLA",
        "artifacts": str(out),
        "tasks": TASKS,
        "rollouts_planned": protocol_obj()["rollout_plan"]["total"],
        "rollouts_completed": len(rollouts),
        "staging_success_rate": staging_rate,
        "task_classifications": {str(r["task"]): r["classification"] for r in task_rows},
        "mutual_information_bits": {str(k): v for k, v in mi_by_task.items()},
        "worker_rows": worker_rows,
        "optional_policies": {"ACT": "not run; no frozen runnable local adapter selected", "pi0.5": "not run; no frozen runnable local adapter selected"},
    }
    write_json(out / "P6G1_FINAL_VERDICT.json", verdict)
    write_report(out, verdict, task_rows, clusters, canon_rows)
    if primary in {
        "P6G1_PRIMITIVE_IK_VLA_THREE_MODE_REALIZATION_QUALIFIED",
        "P6G1_PRIMITIVE_IK_VLA_ACTION_SPACE_QUALIFIED",
    }:
        write_successor_protocols(out)
    return verdict


def write_successor_protocols(out: Path) -> None:
    (out / "P6G2_STRICT_MATCHED_JOINT_ACTION_DATASET_PROTOCOL.md").write_text(
        """# P6-G2 Strict-Matched Joint Action Dataset Protocol

For each hidden-physics root, collect one physical query, save the post-query state, then branch G0/G1/G2 across a frozen force grid. Labels must include local grasp, short lift, and full task success. Splits are root-level and hide friction/CoM from the student. The learning target is P(Y_full=1 | e_physical, tau, g, F), with a GNP-style outcome-set teacher, deployable probe-evidence student, uncertainty-aware planner, and no post-outcome candidate tuning.
""",
        encoding="utf-8",
    )
    (out / "P6G3_DREAMTRAJECTORY_LOCAL_REFINEMENT_INTERFACE.md").write_text(
        """# P6-G3 DreamTrajectory Local Refinement Interface

The GNP-style planner selects coarse primitive g plus force F. IK stages to the selected primitive. The frozen VLA generates the nominal contact-rich action chunk. A DreamTrajectory GRU may roll out local perturbations around that selected chunk and choose the most realizable local variant. DreamTrajectory must not choose among coarse G0/G1/G2 modes in the first version.
""",
        encoding="utf-8",
    )


def write_report(out: Path, verdict: dict, task_rows: list[dict], clusters: list[dict], canon_rows: list[dict]) -> None:
    lines = [
        "# P6-G1 Final Report",
        "",
        f"Primary classification: `{verdict['primary_classification']}`",
        "",
        "`LEARNED_METHOD_CHANGE: NONE`",
        "",
        "`SYSTEM_INTERFACE_CHANGE: PRIMITIVE_IK_STAGING_BEFORE_FROZEN_VLA`",
        "",
        "## Scope",
        "",
        f"- Tasks: `{TASKS}`",
        f"- Primitive count: `{len(PRIMITIVE_IDS)}`",
        f"- Fresh roots per task: `{len(FRESH_ROOTS)}`",
        f"- Requested force: `{FIXED_FORCE_N:g}N`",
        "- Physical query used: `NO`",
        "",
        "## Task Classifications",
        "",
        "| task | object | G0 acc/lift | G1 acc/lift | G2 acc/lift | qualified modes | dominant cluster | classification |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task']} | {row['object']} | {float(row['G0_accuracy']):.3f}/{float(row['G0_lift_rate']):.3f} | "
            f"{float(row['G1_accuracy']):.3f}/{float(row['G1_lift_rate']):.3f} | "
            f"{float(row['G2_accuracy']):.3f}/{float(row['G2_lift_rate']):.3f} | "
            f"{row['num_qualified_modes']} | {float(row['dominant_cluster_fraction']):.3f} | {row['classification']} |"
        )
    lines += [
        "",
        "## Reference Clusters",
        "",
        "Reference clusters are computed only from Arm O deterministic rollouts.",
        "",
        "## Interpretation",
        "",
        "1. This run tests realization of already qualified object-relative grasp candidates, not candidate selection.",
        "2. The VLA receives the same original task instruction for G0/G1/G2; staging is the only primitive signal.",
        "3. Force is fixed to an 8N contract by overwriting force slots while preserving VLA pose and gripper command.",
        "4. Raw VLA rollouts estimate the natural canonical grasp distribution without primitive staging.",
        "5. Full basket placement is intentionally outside this local realization gate.",
        "",
        "## What This Does Not Prove",
        "",
        "- the agent can select the correct primitive",
        "- physical query can infer the best primitive",
        "- GNP-style joint-action prediction works",
        "- DreamTrajectory improves local action execution",
        "- task0/2/5 candidate coverage",
        "- when-to-query is solved",
    ]
    (out / "P6G1_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def terminal_summary(out: Path, verdict: dict) -> str:
    task_rows = read_csv(out / "P6G1_TASK_CLASSIFICATIONS.csv")
    clusters = read_csv(out / "P6G1_REFERENCE_GRASP_CLUSTERS.csv")
    canon = read_csv(out / "P6G1_CANONICALIZATION_ANALYSIS.csv")
    staging = read_csv(out / "P6G1_STAGING_RESULTS.csv")

    def task_row(task: int) -> dict:
        return next((r for r in task_rows if int(r["task"]) == task), {})

    def cluster_line(task: int) -> str:
        parts = []
        for pid in PRIMITIVE_IDS:
            c = next((r for r in clusters if int(r["task"]) == task and r["primitive_id"] == pid), {})
            parts.append(
                f"{pid}=({as_float(c, 'centroid_contact_mid_obj_x', float('nan')):.3f},"
                f"{as_float(c, 'centroid_contact_mid_obj_y', float('nan')):.3f},"
                f"{as_float(c, 'centroid_contact_mid_obj_z', float('nan')):.3f})"
            )
        return "; ".join(parts)

    staging_success = safe_mean([as_int(r, "staging_success") for r in staging]) if staging else 0.0
    collision_free = safe_mean([as_int(r, "collision_free") for r in staging]) if staging else 0.0
    pos_err = safe_mean([as_float(r, "pregrasp_position_error_m") for r in staging]) if staging else float("nan")
    ori_err = safe_mean([as_float(r, "pregrasp_orientation_error_rad") for r in staging]) if staging else float("nan")
    disturb = safe_mean([1 - as_int(r, "handoff_valid") for r in staging]) if staging else 0.0
    mi_vals = [as_float(r, "mutual_information_bits") for r in canon]
    collapse = any(as_int(r, "collapse_detected") for r in canon)

    t1 = task_row(1)
    t6 = task_row(6)
    return f"""STATUS:
{verdict.get('status', 'COMPLETE')}
LEARNED_METHOD_CHANGE: NONE
SYSTEM_INTERFACE_CHANGE:
PRIMITIVE_IK_STAGING_BEFORE_FROZEN_VLA
ARTIFACTS:
{out}

SCOPE:
- Tasks: {TASKS}
- Primitive count: {len(PRIMITIVE_IDS)}
- Fresh roots: {len(FRESH_ROOTS)}
- Primary frozen policy: {POLICY_CONFIG} @ {POLICY_DIR}
- Optional policies: ACT not run; pi0.5 not run
- Requested force: {FIXED_FORCE_N:g}N
- Physical query used: NO

PRIMITIVES:
- Source artifact: {P6G0_RESULTS / 'P6G0_GRASP_CANDIDATE_MANIFEST.csv'}
- G0 geometry: signed offset min from P6-G0 manifest
- G1 geometry: center offset from P6-G0 manifest
- G2 geometry: signed offset max from P6-G0 manifest
- Task/object lookup used at runtime: {TASK_OBJECTS}
- Semantic labels used: NO

IK STAGING:
- Success rate: {staging_success:.3f}
- Collision-free rate: {collision_free:.3f}
- Mean position error: {pos_err:.6f} m
- Mean orientation error: {math.degrees(ori_err) if np.isfinite(ori_err) else float('nan'):.3f} deg
- Object disturbance rate: {disturb:.3f}

REFERENCE CLUSTERS:
- task1 G0/G1/G2: {cluster_line(1)}
- task6 G0/G1/G2: {cluster_line(6)}
- Between-mode separation: task1={as_float(t1, 'between_mode_min_sep_norm_Lg', float('nan')):.3f}, task6={as_float(t6, 'between_mode_min_sep_norm_Lg', float('nan')):.3f}
- Within-mode variation: task1={as_float(t1, 'within_mode_variation_norm_Lg', float('nan')):.3f}, task6={as_float(t6, 'within_mode_variation_norm_Lg', float('nan')):.3f}

FROZEN VLA REALIZATION:

task1:
- G0 mode accuracy / lift rate: {as_float(t1, 'G0_accuracy', 0):.3f} / {as_float(t1, 'G0_lift_rate', 0):.3f}
- G1 mode accuracy / lift rate: {as_float(t1, 'G1_accuracy', 0):.3f} / {as_float(t1, 'G1_lift_rate', 0):.3f}
- G2 mode accuracy / lift rate: {as_float(t1, 'G2_accuracy', 0):.3f} / {as_float(t1, 'G2_lift_rate', 0):.3f}
- Number of qualified modes: {t1.get('num_qualified_modes', '')}
- Dominant-cluster fraction: {as_float(t1, 'dominant_cluster_fraction', 0):.3f}
- Classification: {t1.get('classification', '')}

task6:
- G0 mode accuracy / lift rate: {as_float(t6, 'G0_accuracy', 0):.3f} / {as_float(t6, 'G0_lift_rate', 0):.3f}
- G1 mode accuracy / lift rate: {as_float(t6, 'G1_accuracy', 0):.3f} / {as_float(t6, 'G1_lift_rate', 0):.3f}
- G2 mode accuracy / lift rate: {as_float(t6, 'G2_accuracy', 0):.3f} / {as_float(t6, 'G2_lift_rate', 0):.3f}
- Number of qualified modes: {t6.get('num_qualified_modes', '')}
- Dominant-cluster fraction: {as_float(t6, 'dominant_cluster_fraction', 0):.3f}
- Classification: {t6.get('classification', '')}

RAW VLA:
- task1 natural grasp distribution: {t1.get('raw_vla_distribution_json', '')}
- task6 natural grasp distribution: {t6.get('raw_vla_distribution_json', '')}

CANONICALIZATION:
- Requested primitive affects realized grasp: {not collapse}
- Mutual information: {safe_mean(mi_vals):.3f} bits
- Collapse detected: {collapse}

OPTIONAL POLICIES:
- ACT: not run
- pi0.5: not run

PRIMARY_CLASSIFICATION:
{verdict['primary_classification']}

SCIENTIFIC INTERPRETATION:
1. P6-G1 used only the qualified task1/task6 P6-G0 grasp candidates.
2. The frozen VLA received no semantic grasp labels.
3. Arm O reference clusters define the mode targets; Arm V outcomes never define targets.
4. Arm R records the raw VLA canonical grasp distribution.
5. The verdict is a realization-gate classification, not a planner or GNP result.

WHAT THIS DOES NOT PROVE:
- the agent can select the correct primitive
- physical query can infer the best primitive
- GNP-style joint-action prediction works
- DreamTrajectory improves local action execution
- task0/2/5 candidate coverage
- when-to-query is solved

NEXT:
- If action space qualifies:
  generate the strict-matched physical-evidence x primitive x force dataset.
- If VLA canonicalizes all modes:
  add explicit strategy conditioning or retain deterministic controller execution.
- If staging fails:
  repair the primitive-to-pregrasp interface before changing the VLA.
"""


def prepare_artifacts(out: Path, roots_per_task: int) -> tuple[dict, str]:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("P6G1_CONTACT_TELEMETRY", "P6G1_FORCE_TELEMETRY", "P6G1_POLICY_LOGS", "logs"):
        (out / name).mkdir(parents=True, exist_ok=True)
    protocol = protocol_obj()
    if roots_per_task != len(FRESH_ROOTS):
        protocol["fresh_roots"] = FRESH_ROOTS[:roots_per_task]
        protocol["rollout_plan"]["note"] = "non-default roots_per_task debug override"
    write_json(out / "P6G1_PROTOCOL.json", protocol)
    protocol_hash = stable_hash_obj(protocol)
    (out / "P6G1_PROTOCOL_HASH.txt").write_text(protocol_hash + "\n", encoding="utf-8")
    code_hash = sha256_file(Path(__file__).resolve())
    (out / "P6G1_CODE_HASH.txt").write_text(code_hash + "\n", encoding="utf-8")
    library = recover_primitive_library()
    write_json(out / "P6G1_PRIMITIVE_LIBRARY.json", library)
    write_primitive_markdown(out, library)
    write_agent_interface(out, library)
    write_csv(out / "P6G1_ROOT_MANIFEST.csv", [], ROOT_MANIFEST_FIELDS)
    write_csv(out / "P6G1_STATE_PARITY.csv", [], PARITY_FIELDS)
    write_csv(out / "P6G1_STAGING_RESULTS.csv", [], STAGING_FIELDS)
    write_csv(out / "P6G1_REFERENCE_GRASP_CLUSTERS.csv", [])
    write_csv(out / "P6G1_ROLLOUT_RESULTS.csv", [], ROLLOUT_FIELDS)
    write_csv(out / "P6G1_ACTUAL_CONTACTS.csv", [], ROLLOUT_FIELDS)
    write_csv(out / "P6G1_MODE_CLASSIFICATION.csv", [])
    write_csv(out / "P6G1_CANONICALIZATION_ANALYSIS.csv", [])
    write_csv(out / "P6G1_TASK_CLASSIFICATIONS.csv", [])
    return library, code_hash


def prepare_resume_artifacts(out: Path, roots_per_task: int) -> None:
    if not out.exists():
        raise SystemExit(f"Cannot resume; output directory does not exist: {out}")
    for name in ("P6G1_CONTACT_TELEMETRY", "P6G1_FORCE_TELEMETRY", "P6G1_POLICY_LOGS", "logs"):
        (out / name).mkdir(parents=True, exist_ok=True)

    protocol_path = out / "P6G1_PROTOCOL.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        existing_roots = protocol.get("fresh_roots", FRESH_ROOTS)
        if len(existing_roots) != roots_per_task:
            raise SystemExit(
                f"Cannot resume {out}: protocol has {len(existing_roots)} roots per task, "
                f"but --roots-per-task requested {roots_per_task}."
            )
    else:
        protocol = protocol_obj()
        if roots_per_task != len(FRESH_ROOTS):
            protocol["fresh_roots"] = FRESH_ROOTS[:roots_per_task]
            protocol["rollout_plan"]["note"] = "non-default roots_per_task debug override"
        write_json(protocol_path, protocol)
        (out / "P6G1_PROTOCOL_HASH.txt").write_text(stable_hash_obj(protocol) + "\n", encoding="utf-8")

    code_hash = sha256_file(Path(__file__).resolve())
    (out / "P6G1_CODE_HASH.txt").write_text(code_hash + "\n", encoding="utf-8")
    library_path = out / "P6G1_PRIMITIVE_LIBRARY.json"
    if library_path.exists():
        library = read_json(library_path)
    else:
        library = recover_primitive_library()
        write_json(library_path, library)
    write_primitive_markdown(out, library)
    write_agent_interface(out, library)


def main() -> int:
    if os.environ.get("P6G1_WORKER") == "1":
        return run_worker()

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / f"p6g1_primitive_ik_vla_grasp_realization_{now_tag()}")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=18019)
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--roots-per-task", type=int, default=len(FRESH_ROOTS))
    parser.add_argument("--tasks", nargs="+", type=int, default=TASKS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if sorted(args.tasks) != TASKS:
        raise SystemExit("P6-G1 scope is fixed to tasks [1, 6].")
    roots_per_task = max(1, min(int(args.roots_per_task), len(FRESH_ROOTS)))
    out = args.out.resolve()
    if args.resume:
        prepare_resume_artifacts(out, roots_per_task)
    else:
        prepare_artifacts(out, roots_per_task)

    policy_hash = ""
    try:
        policy_hash = sha256_path(POLICY_DIR)
    except Exception as exc:
        policy_hash = f"HASH_FAILED:{exc!r}"
    (out / "P6G1_FROZEN_POLICY_HASH.txt").write_text(str(policy_hash) + "\n", encoding="utf-8")
    write_json(out / "P6G1_FROZEN_POLICY_MANIFEST.json", policy_manifest_obj(str(policy_hash)))

    server_proc = None
    server_started_by_this_run = False
    server_metadata: dict = {}
    worker_rows: list[dict] = []
    try:
        if not args.no_start_server:
            server_proc = start_policy_server(out, args.server_port)
            server_started_by_this_run = server_proc is not None
        if not wait_for_server(args.server_host, args.server_port, TIMEOUTS_S["server_start"]):
            verdict = {
                "status": "POLICY_SERVER_UNAVAILABLE",
                "primary_classification": "P6G1_FROZEN_VLA_INVOCATION_INTEGRATION_FAILED",
                "learned_method_change": "NONE",
                "system_interface_change": "PRIMITIVE_IK_STAGING_BEFORE_FROZEN_VLA",
                "artifacts": str(out),
                "error": f"Policy server unavailable at {args.server_host}:{args.server_port}",
            }
            write_json(out / "P6G1_FINAL_VERDICT.json", verdict)
            write_report(out, verdict, [], [], [])
            print(terminal_summary(out, verdict))
            return 1
        try:
            server_metadata = probe_server_metadata(args.server_host, args.server_port)
        except Exception as exc:
            server_metadata = {"metadata_probe_error": repr(exc)}
        write_json(out / "P6G1_FROZEN_POLICY_MANIFEST.json", policy_manifest_obj(str(policy_hash), server_metadata))

        for task in TASKS:
            row = launch_worker(out, task, args.server_host, args.server_port, roots_per_task)
            worker_rows.append(row)
            append_csv(out / "P6G1_WORKERS.csv", row, ["task", "returncode", "elapsed_wall_s", "log"])

        verdict = aggregate_and_classify(out, worker_rows)
        print(terminal_summary(out, verdict))
        return 0 if verdict.get("status") == "COMPLETE" else 1
    finally:
        if server_proc is not None and server_started_by_this_run:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=30)
            except Exception:
                try:
                    server_proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
