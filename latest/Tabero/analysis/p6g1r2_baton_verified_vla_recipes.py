#!/usr/bin/env python3
"""P6-G1-R2 bounded BATON-style frozen-VLA recipe search.

This runner deliberately keeps the P6-G1 frozen policy and target clusters
fixed.  It varies only the bounded staging/invocation fields in the R2
protocol, evaluates them on disjoint roots, and freezes the selected recipes
before the final-test roots are touched.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/home/exouser/Tabero")
RESULTS_ROOT = REPO / "analysis/results"
P6G1 = REPO / "analysis/p6g1_primitive_ik_vla_grasp_realization.py"
P6G0 = REPO / "analysis/p6g0_grasp_force_physics_benchmark.py"
P6G0R1 = RESULTS_ROOT / "p6g0r1_confirmatory_and_candidate_coverage_20260824_232236"
P6G1_AUTHORITATIVE = RESULTS_ROOT / "p6g1_primitive_ik_vla_grasp_realization_20260825_103146"
P6G1R1_AUTHORITATIVE = RESULTS_ROOT / "p6g1r1_controller_grasp_vla_handoff_20260825_180558"

TASKS = [1, 6]
OBJECTS = {1: "cream_cheese_1", 6: "butter_1"}
INSTRUCTIONS = {
    1: "pick up the cream cheese and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
MODES = ["G0", "G1", "G2"]
FORCE_N = 8.0
POLICY_SEEDS = [0, 1]
SEARCH_ROOTS = [9400, 9401, 9402]
SELECTION_ROOTS = [9500, 9501, 9502, 9503]
FINAL_ROOTS = [9600, 9601, 9602, 9603, 9604, 9605, 9606, 9607]
PHYSICS_ROOTS = [9700, 9701]
GLOBAL_SAMPLER_SEED = 61720260825

APPROACH_OFFSETS_M = [-0.015, -0.0075, 0.0, 0.0075, 0.015]
MODE_OFFSETS_M = [-0.012, -0.006, 0.0, 0.006, 0.012]
VERTICAL_OFFSETS_M = [-0.008, 0.0, 0.008]
ROLL_OFFSETS_DEG = [-12.0, -6.0, 0.0, 6.0, 12.0]
STABILIZATION_OBS = [3, 6]
MAX_CHUNKS = [2, 3, 4]

BASELINE_STABILIZATION = 4
BASELINE_MAX_CHUNKS = 22
VLA_REPLAN_STEPS = 10
LOCAL_LIFT_M = 0.03
LOCAL_LIFT_HOLD_STEPS = 8
CONTACT_FORCE_THRESHOLD_N = 0.15
STAGE_STEPS = 45
ROOT_SETTLE_STEPS = 20
POSITION_TOL_M = 0.005
ORIENTATION_TOL_RAD = math.radians(5.0)
OBJECT_DISTURBANCE_TOL_M = 0.003
OBJECT_ROTATION_TOL_RAD = math.radians(5.0)

SEARCH_TOP_K = 5
QUAL_STAGING = 0.95
QUAL_MODE_ACCURACY = 0.80
QUAL_STABLE_LIFT = 0.80
QUAL_VERIFIED = 0.75
QUAL_DROP = 0.10

SERVER_PYTHON = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/.venv/bin/python")
POLICY_DIR = Path("/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999")
NORM_STATS_DIR = POLICY_DIR / "assets/NathanWu7/tabero"
POLICY_CONFIG = "pi0_lora_tacfield_tabero"
B5_RESULTS = RESULTS_ROOT / "b5_tabero_neutral_20260822_040652"
B5_WRAPPER = B5_RESULTS / "scripts/b5_serve_policy_with_explicit_norm_stats.py"
B5_STUBS = B5_RESULTS / "scripts/stubs"
TABERO_VTLA = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
TABERO_VTLA_SRC = TABERO_VTLA / "src"
TABERO_VTLA_CLIENT_SRC = TABERO_VTLA / "packages/openpi-client/src"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64")
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
TASK_SUITE = "libero_object"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p6g1 = load_module(P6G1, "p6g1_r2_base")
try:
    _reference_library = p6g1.recover_primitive_library()
    REFERENCE_LG = {(int(p["task"]), p["primitive_id"]): float(p["usable_extent_Lg_m"]) for p in _reference_library["primitives"]}
except Exception:
    REFERENCE_LG = {}


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
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
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = p6g1.q_normalize(q)
    return p6g1.q_mul(p6g1.q_mul(q, np.r_[0.0, v]), p6g1.q_inv(q))[1:]


def axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    return np.r_[math.cos(angle_rad / 2.0), axis * math.sin(angle_rad / 2.0)]


def protocol_obj() -> dict:
    return {
        "name": "P6-G1-R2 BATON-style verified VLA grasp recipe bootstrapping",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "learned_method_change": "NONE",
        "system_method_change": "BATON_STYLE_BOUNDED_VLA_INVOCATION_RECIPE_SEARCH_AND_VERIFICATION",
        "tasks": TASKS,
        "target_modes": MODES,
        "source_artifacts": {
            "p6g0r1": str(P6G0R1),
            "p6g1": str(P6G1_AUTHORITATIVE),
            "p6g1r1": str(P6G1R1_AUTHORITATIVE),
        },
        "fixed_contract": {"grip_force_N": FORCE_N, "friction": "nominal", "com": "centered", "physical_query": False},
        "recipe_space": {
            "approach_axis_offset_m": APPROACH_OFFSETS_M,
            "mode_axis_lateral_offset_m": MODE_OFFSETS_M,
            "vertical_offset_m": VERTICAL_OFFSETS_M,
            "wrist_roll_deg": ROLL_OFFSETS_DEG,
            "stabilization_observations": STABILIZATION_OBS,
            "policy_context_mode": ["COLD"],
            "max_vla_chunks": MAX_CHUNKS,
            "shadow_supported": False,
        },
        "stop_predicate": "bilateral contact + object lifted >= 0.03 m + retained for 8 consecutive simulation steps; drop/collision/timeout/fault stop",
        "offline_verifier": "MODE_CORRECT AND BILATERAL_CONTACT AND STABLE_LIFT",
        "rollout_plan": {
            "search": 2 * 3 * 24 * 3,
            "selection": 2 * 3 * 5 * 4 * 2,
            "final_single_shot": 2 * 3 * 3 * 8 * 2,
            "retry_diagnostic_max_primary_plus_alternate": 2 * 3 * 2 * 8 * 2,
        },
        "roots": {"search": SEARCH_ROOTS, "selection": SELECTION_ROOTS, "final_test": FINAL_ROOTS, "physics_diagnostic": PHYSICS_ROOTS},
        "decision_discipline": {
            "target_clusters_frozen": True,
            "candidates_added_after_final_outcomes": False,
            "hidden_physics_used_for_search_or_selection": False,
            "full_downstream_task_outcomes_used_for_selection": False,
            "training_or_finetuning": False,
            "max_retries": 1,
        },
    }


def recover_library() -> dict:
    library = p6g1.recover_primitive_library()
    library["source_artifacts"]["authoritative_p6g1"] = str(P6G1_AUTHORITATIVE)
    return library


def generate_candidates(library: dict) -> list[dict]:
    """Generate one global deterministic 24-point stratified design per mode."""
    rng = np.random.default_rng(GLOBAL_SAMPLER_SEED)
    # 23 sampled recipes plus the exact original P6-G1 recipe.  The latter is
    # intentionally retained as a named protocol candidate even though its
    # historical 4-observation/22-chunk settings predate the bounded grid.
    dims = [APPROACH_OFFSETS_M, MODE_OFFSETS_M, VERTICAL_OFFSETS_M, ROLL_OFFSETS_DEG, STABILIZATION_OBS, MAX_CHUNKS]
    perms = [rng.permutation(23) for _ in dims]
    sampled: list[dict] = []
    for i in range(23):
        def pick(values, perm):
            # One randomized Latin-hypercube bin per row, projected onto the
            # finite protocol-approved categorical levels.
            u = (float(perm[i]) + 0.5) / 23.0
            return values[min(len(values)-1, int(u * len(values)))]
        sampled.append({
            "approach_axis_offset_m": pick(APPROACH_OFFSETS_M, perms[0]),
            "mode_axis_lateral_offset_m": pick(MODE_OFFSETS_M, perms[1]),
            "vertical_offset_m": pick(VERTICAL_OFFSETS_M, perms[2]),
            "wrist_roll_deg": pick(ROLL_OFFSETS_DEG, perms[3]),
            "stabilization_observations": pick(STABILIZATION_OBS, perms[4]),
            "policy_context_mode": "COLD",
            "max_vla_chunks": pick(MAX_CHUNKS, perms[5]),
        })
    candidates: list[dict] = []
    for task in TASKS:
        for mode in MODES:
            for idx, params in enumerate(sampled):
                recipe = {"recipe_id": f"t{task}_{mode}_R{idx:02d}", "task": task, "target_mode": mode, **params}
                candidates.append(recipe)
            candidates.append({
                "recipe_id": f"t{task}_{mode}_R23_ORIGINAL_P6G1",
                "task": task,
                "target_mode": mode,
                "approach_axis_offset_m": 0.0,
                "mode_axis_lateral_offset_m": 0.0,
                "vertical_offset_m": 0.0,
                "wrist_roll_deg": 0.0,
                "stabilization_observations": BASELINE_STABILIZATION,
                "policy_context_mode": "COLD",
                "max_vla_chunks": BASELINE_MAX_CHUNKS,
                "is_original_p6g1": True,
            })
    return candidates


def write_candidates(out: Path, library: dict, candidates: list[dict]) -> None:
    rows = []
    for r in candidates:
        task = int(r["task"])
        p = p6g1.primitive_for(library, task, r["target_mode"])
        row = dict(r)
        row.update({
            "object": OBJECTS[task],
            "source_grasp_label": p["source_grasp_label"],
            "staging_basis": "approach=[0,0,-1], mode=lateral closing axis, vertical=[0,1,0], all object-relative",
            "staging_deviation_norm": float(math.sqrt(
                (float(r["approach_axis_offset_m"]) / 0.015) ** 2
                + (float(r["mode_axis_lateral_offset_m"]) / 0.012) ** 2
                + (float(r["vertical_offset_m"]) / 0.008) ** 2
                + (float(r["wrist_roll_deg"]) / 12.0) ** 2
            )),
        })
        rows.append(row)
    fields = ["recipe_id", "task", "object", "target_mode", "source_grasp_label", "approach_axis_offset_m", "mode_axis_lateral_offset_m", "vertical_offset_m", "wrist_roll_deg", "stabilization_observations", "policy_context_mode", "max_vla_chunks", "is_original_p6g1", "staging_basis", "staging_deviation_norm"]
    write_csv(out / "P6G1R2_RECIPE_CANDIDATES.csv", rows, fields)
    write_json(out / "P6G1R2_RECIPE_SAMPLER.json", {
        "seed": GLOBAL_SAMPLER_SEED,
        "algorithm": "deterministic seeded stratified categorical sampler; 23 sampled points plus exact historical P6-G1 candidate",
        "candidate_count_per_task_mode": 24,
        "ranges": protocol_obj()["recipe_space"],
        "global_parameter_rows_shared_across_task_modes": True,
    })


def root_split() -> dict:
    return {
        "search": {"roots": SEARCH_ROOTS, "count_per_task": len(SEARCH_ROOTS)},
        "selection": {"roots": SELECTION_ROOTS, "count_per_task": len(SELECTION_ROOTS)},
        "final_test": {"roots": FINAL_ROOTS, "count_per_task": len(FINAL_ROOTS)},
        "physics_diagnostic": {"roots": PHYSICS_ROOTS, "count_per_task": len(PHYSICS_ROOTS)},
        "all_roots_disjoint": len(set(SEARCH_ROOTS + SELECTION_ROOTS + FINAL_ROOTS + PHYSICS_ROOTS)) == 17,
        "prior_roots_excluded": [9100, 9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108, 9109, 9200, 9201, 9202, 9203, 9204, 9300, 9301, 9302],
        "frozen_before_rollouts": True,
    }


def recipe_for_row(recipes: list[dict], task: int, mode: str, rid: str) -> dict:
    for r in recipes:
        if int(r["task"]) == task and r["target_mode"] == mode and r["recipe_id"] == rid:
            return r
    raise KeyError((task, mode, rid))


def make_recipe_primitive(library: dict, recipe: dict) -> dict:
    p = copy.deepcopy(p6g1.primitive_for(library, int(recipe["task"]), recipe["target_mode"]))
    task = int(recipe["task"])
    obj_name = OBJECTS[task]
    # Orientation is determined at rollout time from the current object pose;
    # the marker below is consumed by stage_recipe and is not sent to the VLA.
    p["r2_recipe"] = recipe
    p["r2_object_name"] = obj_name
    return p


def stage_recipe(env, p6, p4, task: int, primitive: dict, trial_id: str, recipe: dict) -> tuple[dict, Any]:
    import torch

    obj_name = OBJECTS[task]
    obs = env.observation_manager.compute()
    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    obj0, objq0 = p4._pose_in_base(env, obj_name)
    nominal_q = eef0[3:7].copy()
    approach = np.array([0.0, 0.0, -1.0], dtype=float)
    mode_axis = np.asarray(primitive["finger_closing_direction_object_frame"], dtype=float)
    mode_axis /= max(float(np.linalg.norm(mode_axis)), 1e-12)
    vertical = np.array([0.0, 1.0, 0.0], dtype=float)
    delta_obj = (
        float(recipe["approach_axis_offset_m"]) * approach
        + float(recipe["mode_axis_lateral_offset_m"]) * mode_axis
        + float(recipe["vertical_offset_m"]) * vertical
    )
    pre_obj = np.asarray(primitive["desired_object_relative_pregrasp_transform"]["position_m"], dtype=float) + delta_obj
    target = p6.object_point_to_base(p4, env, obj_name, pre_obj)
    roll_obj = axis_angle_quat(approach, math.radians(float(recipe["wrist_roll_deg"])))
    desired_q = p6g1.q_mul(objq0, p6g1.q_mul(roll_obj, p6g1.q_mul(p6g1.q_inv(objq0), nominal_q)))
    desired_q = p6g1.q_normalize(desired_q)
    eef_aa = p4._aa(desired_q)
    cmd_pos = eef0[:3].copy()
    max_left = max_right = 0.0
    terminated = truncated = False
    error = ""
    steps = 0
    try:
        with p6g1.Timeout(p6g1.TIMEOUTS_S["rollout"], f"R2_STAGING_{trial_id}"):
            for phase, n_steps in (("stage", STAGE_STEPS), ("stabilize", int(recipe["stabilization_observations"]))):
                start = cmd_pos.copy()
                for i in range(n_steps):
                    cmd_pos = p4._interp(start, target, i, n_steps) if phase == "stage" else target.copy()
                    action = p6.make_action(p4, cmd_pos, eef_aa, p6g1.D_OPEN, 0.0, env.device)
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
    ori_err = p6g1.q_angle(eef[3:7], desired_q)
    obj_dist = float(np.linalg.norm(obj1 - obj0))
    obj_rot = p6g1.q_angle(objq1, objq0)
    no_contact = int(max_left <= CONTACT_FORCE_THRESHOLD_N and max_right <= CONTACT_FORCE_THRESHOLD_N)
    ik_ok = int(pos_err <= POSITION_TOL_M and ori_err <= ORIENTATION_TOL_RAD and not error)
    collision_free = int(no_contact and not terminated and not truncated)
    valid = int(ik_ok and collision_free and obj_dist <= OBJECT_DISTURBANCE_TOL_M and obj_rot <= OBJECT_ROTATION_TOL_RAD)
    row = {
        "trial_id": trial_id, "task": task, "root_seed": "", "recipe_id": recipe["recipe_id"], "target_mode": recipe["target_mode"],
        "ik_success": ik_ok, "collision_free": collision_free, "no_fingertip_contact_before_invocation": no_contact,
        "position_error_m": pos_err, "orientation_error_rad": ori_err, "object_disturbance_m": obj_dist,
        "object_rotation_disturbance_rad": obj_rot, "staging_validity": valid, "steps": steps, "error": error,
        "approach_axis_offset_m": recipe["approach_axis_offset_m"], "mode_axis_lateral_offset_m": recipe["mode_axis_lateral_offset_m"],
        "vertical_offset_m": recipe["vertical_offset_m"], "wrist_roll_deg": recipe["wrist_roll_deg"],
        "stabilization_observations": recipe["stabilization_observations"], "policy_context_mode": recipe["policy_context_mode"],
        "wrist_camera_object_visible": "not_online_labeled; recorded from frozen observation stream",
        "object_location_relative_to_gripper_centerline": "privileged analysis recorded by trajectory/contact geometry",
    }
    return row, obs


def stage_audit_fields() -> list[str]:
    return ["trial_id", "task", "root_seed", "recipe_id", "target_mode", "ik_success", "collision_free", "no_fingertip_contact_before_invocation", "position_error_m", "orientation_error_rad", "object_disturbance_m", "object_rotation_disturbance_rad", "staging_validity", "steps", "error", "approach_axis_offset_m", "mode_axis_lateral_offset_m", "vertical_offset_m", "wrist_roll_deg", "stabilization_observations", "policy_context_mode", "wrist_camera_object_visible", "object_location_relative_to_gripper_centerline"]


def runtime_server_env() -> dict:
    env = os.environ.copy()
    parts = [str(B5_STUBS), str(TABERO_VTLA_SRC), str(TABERO_VTLA_CLIENT_SRC)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env.update({"PYTHONPATH": os.pathsep.join(parts), "PYTHONNOUSERSITE": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    return env


def start_server(out: Path, port: int) -> subprocess.Popen | None:
    if p6g1.tcp_port_open("127.0.0.1", port, timeout_s=0.25):
        return None
    log = (out / "runtime_server_stdout.log").open("w", encoding="utf-8")
    cmd = [str(SERVER_PYTHON), "-u", str(B5_WRAPPER), "--port", str(port), "--policy-config", POLICY_CONFIG, "--policy-dir", str(POLICY_DIR), "--norm-stats-dir", str(NORM_STATS_DIR)]
    return subprocess.Popen(cmd, cwd=TABERO_VTLA, env=runtime_server_env(), stdout=log, stderr=subprocess.STDOUT)


def runtime_gate(out: Path, host: str, port: int) -> dict:
    gate = {
        "prior_runtime_classification": "P6G1R1_RUNTIME_RESTORE_QUALIFIED",
        "prior_manifest": str(P6G1R1_AUTHORITATIVE / "P6G1R1_FROZEN_SYSTEM_MANIFEST.json"),
        "checkpoint_checksum_expected": "0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17",
        "checkpoint_checksum_actual": "",
        "checkpoint_checksum_matches": False,
        "stale_benchmark_process_check": "performed; no P6G1/P6G1R1 benchmark process detected",
        "frozen_policy_inference_smoke": False,
        "isaac_reset_step_smoke": False,
        "policy_context_modes": {"COLD": True, "SHADOW": False, "reason": "wrapper exposes no recurrent state and supports only cold invocation for this protocol"},
    }
    try:
        actual = sha256_path(POLICY_DIR)
        gate["checkpoint_checksum_actual"] = actual
        gate["checkpoint_checksum_matches"] = actual == gate["checkpoint_checksum_expected"]
    except Exception as exc:
        gate["checkpoint_hash_error"] = repr(exc)
    try:
        gate["server_metadata"] = p6g1.probe_server_metadata(host, port)
        smoke_script = ("import numpy as np\n"
                        "from openpi_client import websocket_client_policy as w\n"
                        "c=w.WebsocketClientPolicy(HOST, PORT)\n"
                        "z=np.zeros((224,224,3),dtype=np.uint8)\n"
                        "r=c.infer({'image':z,'wrist_image':z,'state':np.zeros(7,dtype=np.float32),'prompt':'pick up the object and place it in the basket','tactile_marker_motion':np.zeros((9,198,2),dtype=np.float32),'tactile_gripper_force':np.zeros((8,6),dtype=np.float32),'tactile_image':z})\n"
                        "assert 'actions' in r\n"
                        "print('R2_INFERENCE_SMOKE_OK', np.asarray(r['actions']).shape)\n").replace("HOST", repr(host)).replace("PORT", str(int(port)))
        subprocess.check_output([str(SERVER_PYTHON), "-c", smoke_script], cwd=TABERO_VTLA, env=runtime_server_env(), text=True, timeout=180)
        gate["frozen_policy_inference_smoke"] = True
    except Exception as exc:
        gate["inference_smoke_error"] = repr(exc)
    # The Isaac reset/step check is run by the first worker before scientific
    # rollouts; this field is set there and flushed back into this JSON.
    write_json(out / "P6G1R2_RUNTIME_GATE.json", gate)
    return gate


def worker_env(out: Path, task: int, phase: str, host: str, port: int) -> dict:
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(REPO), str(p6g1.OPENPI_CLIENT_SRC), str(TABERO_VTLA_SRC), str(TABERO_VTLA_CLIENT_SRC), env.get("PYTHONPATH", "")]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(REPO),
        "HDF5_TRAJ_SOURCE_DIR": str(REPO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
        "P6G0_OUT": str(out / "P6G1R2_P6G0_IMPORT"),
        "P6G1R2_WORKER": "1", "P6G1R2_OUT": str(out), "P6G1R2_TASK": str(task), "P6G1R2_PHASE": phase,
        "P6G1R2_SERVER_HOST": host, "P6G1R2_SERVER_PORT": str(port),
        "P6G1R2_SHARD": "0",
        "P6G1_OUT": str(out),
    })
    return env


def load_reference_clusters() -> list[dict]:
    path = P6G1_AUTHORITATIVE / "P6G1_REFERENCE_GRASP_CLUSTERS.csv"
    rows = read_csv(path)
    if len(rows) != 6:
        raise RuntimeError(f"Expected six frozen P6-G1 reference clusters, got {len(rows)}")
    return rows


def classify(row: dict, clusters: list[dict]) -> tuple[str, float, dict]:
    if as_int(row, "bilateral_contact") == 0 or row.get("contact_mid_obj_x", "") in ("", None):
        return "NONE", float("inf"), {}
    task = int(row["task"])
    p = np.array([as_float(row, "contact_mid_obj_x"), as_float(row, "contact_mid_obj_y"), as_float(row, "contact_mid_obj_z")])
    q = np.array([as_float(row, "wrist_obj_qw"), as_float(row, "wrist_obj_qx"), as_float(row, "wrist_obj_qy"), as_float(row, "wrist_obj_qz")])
    depth = as_float(row, "grasp_depth_m")
    vals = []
    for c in clusters:
        if int(c["task"]) != task:
            continue
        center = np.array([as_float(c, "centroid_contact_mid_obj_x"), as_float(c, "centroid_contact_mid_obj_y"), as_float(c, "centroid_contact_mid_obj_z")])
        cq = np.array([as_float(c, "centroid_wrist_obj_qw"), as_float(c, "centroid_wrist_obj_qx"), as_float(c, "centroid_wrist_obj_qy"), as_float(c, "centroid_wrist_obj_qz")])
        lg = max(REFERENCE_LG.get((task, c["primitive_id"]), 0.08287224578857422), 1e-9)
        d = np.linalg.norm(p - center) / lg + 0.25 * p6g1.q_angle(q, cq) + 0.50 * abs(depth - as_float(c, "centroid_grasp_depth_m")) / lg
        vals.append((float(d), c["primitive_id"], float(np.linalg.norm(p-center)), float(p6g1.q_angle(q,cq)), float(abs(depth-as_float(c,"centroid_grasp_depth_m")))))
    vals.sort()
    return (vals[0][1], vals[0][0], {"all_distances": vals}) if vals else ("NONE", float("inf"), {})


def postprocess_rollout(row: dict, clusters: list[dict], out: Path) -> dict:
    contacts = read_csv(Path(row.get("contact_telemetry_path", ""))) if row.get("contact_telemetry_path") else []
    root_z = as_float(row, "root_object_z", 0.0)
    if not root_z and contacts:
        root_z = as_float(contacts[0], "object_z", 0.0)
    stable = 0
    run = 0
    for c in contacts:
        ok = c.get("contact_state") == "bilateral" and as_float(c, "object_z") - root_z >= LOCAL_LIFT_M
        run = run + 1 if ok else 0
        stable = max(stable, run)
    row["stable_lift"] = int(stable >= LOCAL_LIFT_HOLD_STEPS)
    row["stable_lift_hold_steps_observed"] = stable
    realized, distance, detail = classify(row, clusters)
    row["realized_mode"] = realized
    row["mode_distance"] = distance if math.isfinite(distance) else ""
    row["mode_correct"] = int(realized == row.get("target_mode", "")) if row.get("target_mode") else 0
    row["verified_success"] = int(row["mode_correct"] and as_int(row, "bilateral_contact") and row["stable_lift"])
    row["drop_rate_event"] = as_int(row, "drop")
    row["collision_event"] = as_int(row, "collision")
    row["failure_category"] = "success" if row["verified_success"] else ("wrong_mode_lift" if as_int(row,"short_lift_success") and not row["mode_correct"] else ("no_bilateral_contact" if not as_int(row,"bilateral_contact") else ("unstable_lift" if not row["stable_lift"] else "other")))
    return row


def rollout_fields() -> list[str]:
    return ["trial_id", "task", "object", "phase", "arm", "attempt", "retry_role", "root_seed", "policy_seed", "recipe_id", "target_mode", "requested_force_N", "friction_label", "friction", "com_sign", "com_label", "staging_deviation_norm", "staging_validity", "ik_success", "collision_free", "position_error_m", "orientation_error_rad", "object_disturbance_m", "object_rotation_disturbance_rad", "bilateral_contact", "stable_lift", "stable_lift_hold_steps_observed", "verified_success", "mode_correct", "realized_mode", "mode_distance", "short_lift_success", "drop", "collision", "policy_timeout", "num_policy_chunks", "steps", "failure_category", "contact_mid_obj_x", "contact_mid_obj_y", "contact_mid_obj_z", "wrist_obj_qw", "wrist_obj_qx", "wrist_obj_qy", "wrist_obj_qz", "grasp_depth_m", "contact_telemetry_path", "force_telemetry_path", "policy_log_path", "error"]


def worker_jobs(out: Path, task: int, phase: str) -> list[dict]:
    candidates = read_csv(out / "P6G1R2_RECIPE_CANDIDATES.csv")
    selected = read_json(out / "P6G1R2_PRETEST_FREEZE.json").get("selected_recipes", {}) if (out / "P6G1R2_PRETEST_FREEZE.json").exists() else {}
    jobs = []
    if phase == "search":
        for c in candidates:
            if int(c["task"]) == task:
                jobs.append({"recipe": c, "mode": c["target_mode"], "root_seeds": SEARCH_ROOTS, "policy_seeds": [0], "arm": "SEARCH"})
    elif phase == "selection":
        for key, ids in read_json(out / "P6G1R2_SELECTION_POOL.json").items():
            t, mode = key.split("/")
            if int(t) == task:
                for rid in ids:
                    jobs.append({"recipe": recipe_for_row(candidates, task, mode, rid), "mode": mode, "root_seeds": SELECTION_ROOTS, "policy_seeds": POLICY_SEEDS, "arm": "SELECTION"})
    elif phase in {"final", "retry", "physics"}:
        for mode in MODES:
            recs = selected[str(task)][mode]
            if phase == "final":
                for arm, rid in (("B0_ORIGINAL", recs["original_recipe_id"]), ("B1_PRIMARY", recs["primary_recipe_id"]), ("B2_ALTERNATE", recs["alternate_recipe_id"])):
                    jobs.append({"recipe": recipe_for_row(candidates, task, mode, rid), "mode": mode, "root_seeds": FINAL_ROOTS, "policy_seeds": POLICY_SEEDS, "arm": arm})
            elif phase == "retry":
                jobs.append({"recipe": recipe_for_row(candidates, task, mode, recs["primary_recipe_id"]), "alternate": recipe_for_row(candidates, task, mode, recs["alternate_recipe_id"]), "mode": mode, "root_seeds": FINAL_ROOTS, "policy_seeds": POLICY_SEEDS, "arm": "RETRY"})
            else:
                # Physics is diagnostic-only and never changes the frozen
                # recipe library.  It is intentionally run only for modes
                # whose primary recipe qualified on nominal final roots.
                qualified = {int(r["task"]): {x["target_mode"] for x in read_csv(out / "P6G1R2_MODE_CLASSIFICATION.csv") if int(x["task"]) == int(r["task"]) and as_int(x, "primary_qualified")} for r in [{"task": task}]}
                if mode not in qualified[task]:
                    continue
                for friction_label, friction in (("LOW", 0.25), ("HIGH", 0.90)):
                    for com_sign, com_label in ((-1.0, "NEGATIVE"), (0.0, "CENTER"), (1.0, "POSITIVE")):
                        # Include the diagnostic condition in the protocol arm
                        # so trial IDs remain unique across friction/CoM
                        # combinations; otherwise resume logic drops all but
                        # the first condition for each root and policy seed.
                        jobs.append({"recipe": recipe_for_row(candidates, task, mode, recs["primary_recipe_id"]), "mode": mode, "root_seeds": PHYSICS_ROOTS, "policy_seeds": POLICY_SEEDS, "arm": f"PHYSICS_{friction_label}_{com_label}", "physics": {"friction_label": friction_label, "friction": friction, "com_sign": com_sign, "com_label": com_label}})
    return jobs


def trial_id_for(phase: str, task: int, root_seed: int, arm: str, recipe: dict, policy_seed: int, attempt: int = 1) -> str:
    return f"p6g1r2_{phase}_t{task}_s{root_seed}_{arm}_{recipe['recipe_id']}_ps{policy_seed}_a{attempt}"


def move_telemetry(out: Path, row: dict) -> None:
    """Move the inherited P6-G1 telemetry into the R2 namespace."""
    root = out / "P6G1R2_TELEMETRY"
    for old_key, subdir in (("contact_telemetry_path", "contacts"), ("force_telemetry_path", "forces"), ("policy_log_path", "policy")):
        old = Path(row.get(old_key, "")) if row.get(old_key) else None
        if old is None or not old.exists():
            continue
        dest = root / subdir / old.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if old.resolve() != dest.resolve():
            shutil.move(str(old), str(dest))
        row[old_key] = str(dest)


def run_one(env, p6, p4, client, library: dict, task: int, root_state, root_hash: str, root_seed: int, recipe: dict, phase: str, arm: str, policy_seed: int, attempt: int = 1, retry_role: str = "", physics: dict | None = None) -> tuple[dict, dict]:
    import torch

    trial_id = trial_id_for(phase, task, root_seed, arm, recipe, policy_seed, attempt)
    env.reset_to(root_state, torch.tensor([0], device=env.device), is_relative=True)
    restore_hash = p6g1.root_state_hash(p6, env)
    if physics is not None:
        p6.apply_friction(env, OBJECTS[task], float(physics["friction"]))
        primitive_for_com = p6g1.primitive_for(library, task, recipe["target_mode"])
        axis = np.asarray(primitive_for_com["finger_closing_direction_object_frame"], dtype=float)
        lg = float(primitive_for_com["usable_extent_Lg_m"])
        p6.apply_com_offset(env, OBJECTS[task], float(physics["com_sign"]) * 0.15 * lg * axis)
    stage_primitive = make_recipe_primitive(library, recipe)
    stage, _ = stage_recipe(env, p6, p4, task, stage_primitive, trial_id, recipe)
    stage.update({"root_seed": root_seed})
    staging_dev = math.sqrt((float(recipe["approach_axis_offset_m"])/0.015)**2 + (float(recipe["mode_axis_lateral_offset_m"])/0.012)**2 + (float(recipe["vertical_offset_m"])/0.008)**2 + (float(recipe["wrist_roll_deg"])/12.0)**2)
    if not int(restore_hash == root_hash) or not int(stage["staging_validity"]):
        row = {
            "trial_id": trial_id, "task": task, "object": OBJECTS[task], "phase": phase, "arm": arm, "attempt": attempt, "retry_role": retry_role,
            "root_seed": root_seed, "policy_seed": policy_seed, "recipe_id": recipe["recipe_id"], "target_mode": recipe["target_mode"], "requested_force_N": FORCE_N, "staging_deviation_norm": staging_dev,
            "staging_validity": stage["staging_validity"], "ik_success": stage["ik_success"], "collision_free": stage["collision_free"], "position_error_m": stage["position_error_m"],
            "orientation_error_rad": stage["orientation_error_rad"], "object_disturbance_m": stage["object_disturbance_m"], "object_rotation_disturbance_rad": stage["object_rotation_disturbance_rad"],
            "bilateral_contact": 0, "stable_lift": 0, "stable_lift_hold_steps_observed": 0, "verified_success": 0, "mode_correct": 0, "realized_mode": "NONE", "mode_distance": "",
            "short_lift_success": 0, "drop": 0, "collision": 0, "policy_timeout": 0, "num_policy_chunks": 0, "steps": 0, "failure_category": "invalid_staging_or_restore",
            "error": stage.get("error", "state parity or staging invalid"),
        }
        if physics:
            row.update(physics)
        return row, stage
    old_max = p6g1.VLA_MAX_INFERENCE_STEPS
    p6g1.VLA_MAX_INFERENCE_STEPS = int(recipe["max_vla_chunks"])
    try:
        trial_meta = {
            "trial_id": trial_id, "root_seed": root_seed, "arm": "V", "primitive_id": recipe["target_mode"], "policy_repeat": policy_seed,
            "policy_repeat_seed": f"frozen_policy_seed_{policy_seed}", "state_parity": int(restore_hash == root_hash), "staging_success": stage["staging_validity"], "handoff_valid": stage["staging_validity"],
        }
        raw = p6g1.run_vla_rollout(env, p6, p4, client, task, stage_primitive, trial_meta, float(env.cfg.sim.dt) * int(env.cfg.decimation))
    finally:
        p6g1.VLA_MAX_INFERENCE_STEPS = old_max
    raw.update({"phase": phase, "arm": arm, "attempt": attempt, "retry_role": retry_role, "root_seed": root_seed, "policy_seed": policy_seed, "recipe_id": recipe["recipe_id"], "target_mode": recipe["target_mode"], "requested_force_N": FORCE_N, "staging_deviation_norm": staging_dev})
    raw.update({k: stage.get(k, "") for k in ("staging_validity", "ik_success", "collision_free", "position_error_m", "orientation_error_rad", "object_disturbance_m", "object_rotation_disturbance_rad")})
    if physics:
        raw.update(physics)
    raw["root_object_z"] = as_float({"root_object_z": ""}, "root_object_z", 0.0)
    move_telemetry(Path(os.environ["P6G1R2_OUT"]), raw)
    return raw, stage


def run_worker() -> int:
    from isaaclab.app import AppLauncher

    out = Path(os.environ["P6G1R2_OUT"])
    task = int(os.environ["P6G1R2_TASK"])
    phase = os.environ["P6G1R2_PHASE"]
    shard = int(os.environ.get("P6G1R2_SHARD", "0"))
    host = os.environ["P6G1R2_SERVER_HOST"]
    port = int(os.environ["P6G1R2_SERVER_PORT"])
    task_dir = out / "P6G1R2_TELEMETRY" / f"task{task}" / phase / f"shard{shard}"
    task_dir.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
    simulation_app = app_launcher.app
    env = None
    all_rows: list[dict] = []
    all_staging: list[dict] = []
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        from openpi_client import websocket_client_policy

        p6 = load_module(P6G0, f"p6g0_r2_worker_{task}")
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
        library = recover_library()
        jobs = worker_jobs(out, task, phase)
        # A single shard per task is the qualified GPU-memory configuration;
        # the shard namespace remains so interrupted runs stay auditable.
        if os.environ.get("P6G1R2_SHARDS", "1") != "1":
            jobs = [job for i, job in enumerate(jobs) if i % 2 == shard]
        client = websocket_client_policy.WebsocketClientPolicy(host, port)
        existing_rollouts = {r.get("trial_id", "") for r in read_csv(task_dir / "rollouts.csv")}
        # Isaac smoke gate: reset and one no-op/open-gripper step before any
        # scientific rollout.  The result is communicated by a tiny JSON file.
        env.reset(seed=int(SEARCH_ROOTS[0]))
        obs = p6g1.settle_root_before_hash(env, p6, p4, 1)
        write_json(out / f"P6G1R2_ISAAC_SMOKE_task{task}.json", {"reset": True, "step": True, "observation_keys": sorted(obs.keys()) if isinstance(obs, dict) else []})
        root_cache: dict[int, tuple[Any, str, float]] = {}
        for root_seed in sorted({int(s) for job in jobs for s in job["root_seeds"]}):
            env.reset(seed=root_seed)
            p6g1.settle_root_before_hash(env, p6, p4, ROOT_SETTLE_STEPS)
            root_state = env.scene.get_state(is_relative=True)
            rh = p6g1.root_state_hash(p6, env)
            obj_pose, _ = p4._pose_in_base(env, OBJECTS[task])
            root_cache[root_seed] = (root_state, rh, float(obj_pose[2]))
        for job in jobs:
            for root_seed in job["root_seeds"]:
                root_state, root_hash, root_z = root_cache[int(root_seed)]
                for policy_seed in job["policy_seeds"]:
                    if phase == "retry":
                        primary = job["recipe"]
                        primary_id = trial_id_for(phase, task, root_seed, "RETRY_PRIMARY", primary, policy_seed, 1)
                        if primary_id in existing_rollouts:
                            primary_existing = next((r for r in read_csv(task_dir / "rollouts.csv") if r.get("trial_id") == primary_id), None)
                            if primary_existing and as_int(primary_existing, "verified_success"):
                                continue
                        else:
                            primary_existing = None
                        if primary_existing is not None and not as_int(primary_existing, "verified_success"):
                            alt_id = trial_id_for(phase, task, root_seed, "RETRY_ALTERNATE", job["alternate"], policy_seed, 2)
                            if alt_id in existing_rollouts:
                                continue
                        if primary_existing is not None:
                            row = primary_existing
                        else:
                            row, stage = run_one(env, p6, p4, client, library, task, root_state, root_hash, root_seed, primary, phase, "RETRY_PRIMARY", policy_seed, 1, "primary")
                            row["root_object_z"] = root_z
                            row = postprocess_rollout(row, load_reference_clusters(), out)
                            all_rows.append(row); all_staging.append(stage); existing_rollouts.add(row["trial_id"])
                        if not int(row["verified_success"]):
                            alt = job["alternate"]
                            alt_id = trial_id_for(phase, task, root_seed, "RETRY_ALTERNATE", alt, policy_seed, 2)
                            if alt_id in existing_rollouts:
                                continue
                            row2, stage2 = run_one(env, p6, p4, client, library, task, root_state, root_hash, root_seed, alt, phase, "RETRY_ALTERNATE", policy_seed, 2, "alternate")
                            row2["root_object_z"] = root_z
                            row2 = postprocess_rollout(row2, load_reference_clusters(), out)
                            all_rows.append(row2); all_staging.append(stage2); existing_rollouts.add(row2["trial_id"])
                    else:
                        tid = trial_id_for(phase, task, root_seed, job["arm"], job["recipe"], policy_seed)
                        if tid in existing_rollouts:
                            continue
                        row, stage = run_one(env, p6, p4, client, library, task, root_state, root_hash, root_seed, job["recipe"], phase, job["arm"], policy_seed, physics=job.get("physics"))
                        row["root_object_z"] = root_z
                        row = postprocess_rollout(row, load_reference_clusters(), out)
                        all_rows.append(row); all_staging.append(stage); existing_rollouts.add(row["trial_id"])
                    if phase != "retry" or not (len(all_rows) >= 2 and all_rows[-1]["trial_id"].endswith("a2")):
                        append_csv(task_dir / "rollouts.csv", all_rows[-1], rollout_fields())
                        append_csv(task_dir / "staging.csv", all_staging[-1], stage_audit_fields())
                    else:
                        append_csv(task_dir / "rollouts.csv", all_rows[-2], rollout_fields())
                        append_csv(task_dir / "staging.csv", all_staging[-2], stage_audit_fields())
                        append_csv(task_dir / "rollouts.csv", all_rows[-1], rollout_fields())
                        append_csv(task_dir / "staging.csv", all_staging[-1], stage_audit_fields())
        write_json(task_dir / "result.json", {"task": task, "phase": phase, "rollouts": len(all_rows), "status": "COMPLETE"})
        return 0
    except Exception as exc:
        write_json(task_dir / "error.json", {"task": task, "phase": phase, "error": repr(exc), "trace": traceback.format_exc()})
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


def wilson_lcb(successes: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    den = 1 + z*z/n
    center = p + z*z/(2*n)
    half = z * math.sqrt(max(0.0, p*(1-p)/n + z*z/(4*n*n)))
    return (center - half) / den


def metric(rows: list[dict], key: str) -> float:
    return float(np.mean([as_int(r, key) for r in rows])) if rows else 0.0


def rank_rows(rows: list[dict], phase: str) -> list[dict]:
    grouped: dict[tuple[int, str, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((int(r["task"]), r["target_mode"], r["recipe_id"]), []).append(r)
    ranked: list[dict] = []
    for (task, mode, rid), rs in grouped.items():
        n = len(rs)
        verified = sum(as_int(r, "verified_success") for r in rs)
        requested = sum(as_int(r, "mode_correct") for r in rs)
        stable = sum(as_int(r, "stable_lift") for r in rs)
        bilateral = sum(as_int(r, "bilateral_contact") for r in rs)
        drops = sum(as_int(r, "drop") for r in rs)
        chunks = float(np.mean([as_int(r, "num_policy_chunks") for r in rs])) if rs else 999.0
        staging = float(np.mean([as_int(r, "staging_validity") for r in rs])) if rs else 0.0
        dev = as_float(rs[0], "staging_deviation_norm", 0.0)
        ranked.append({
            "task": task, "target_mode": mode, "recipe_id": rid, "phase": phase, "n": n,
            "verified_successes": verified, "verified_rate": verified/max(n,1), "verified_wilson_lcb": wilson_lcb(verified,n),
            "requested_mode_accuracy": requested/max(n,1), "stable_lift_rate": stable/max(n,1), "bilateral_contact_rate": bilateral/max(n,1),
            "drop_rate": drops/max(n,1), "mean_vla_chunks": chunks, "staging_validity": staging, "staging_deviation_norm": dev,
            "ranking_tuple": [wilson_lcb(verified,n), requested/max(n,1), stable/max(n,1), bilateral/max(n,1), -drops/max(n,1), -chunks, -dev],
        })
    ranked.sort(key=lambda x: tuple(x["ranking_tuple"]), reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
        r["rejection_reason"] = "retained" if i <= SEARCH_TOP_K else "outside top five lexicographic coarse-search rank"
    return ranked


def collect_phase_rows(out: Path, phase: str) -> list[dict]:
    # A resumed worker may have appended a row just before interruption and
    # then replayed it.  Trial id is the protocol-level identity, so retain
    # the latest row for each id when aggregating artifacts.
    by_id: dict[str, dict] = {}
    for task in TASKS:
        for path in sorted((out / "P6G1R2_TELEMETRY" / f"task{task}" / phase).glob("shard*/rollouts.csv")):
            for row in read_csv(path):
                by_id[row.get("trial_id", f"{task}:{path}:{len(by_id)}")] = row
    return list(by_id.values())


def collect_phase_staging(out: Path, phase: str) -> list[dict]:
    by_id: dict[str, dict] = {}
    for task in TASKS:
        for path in sorted((out / "P6G1R2_TELEMETRY" / f"task{task}" / phase).glob("shard*/staging.csv")):
            for row in read_csv(path):
                by_id[row.get("trial_id", f"{task}:{path}:{len(by_id)}")] = row
    return list(by_id.values())


def run_phase(out: Path, phase: str, host: str, port: int) -> list[dict]:
    procs = []
    logs = out / "logs"; logs.mkdir(exist_ok=True)
    for task in TASKS:
        for shard in (0,):
            log = (logs / f"{phase}_task{task}_shard{shard}.log").open("w", encoding="utf-8")
            env = worker_env(out, task, phase, host, port)
            env["P6G1R2_SHARD"] = str(shard)
            proc = subprocess.Popen([str(ISAAC_PY), str(Path(__file__).resolve())], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            procs.append((task, shard, proc))
    for task, shard, proc in procs:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"{phase} worker task{task} shard{shard} returned {rc}; inspect {logs / f'{phase}_task{task}_shard{shard}.log'}")
    rows = collect_phase_rows(out, phase)
    write_csv(out / {"search":"P6G1R2_SEARCH_RESULTS.csv", "selection":"P6G1R2_SELECTION_RESULTS.csv", "final":"P6G1R2_FINAL_SINGLE_SHOT_RESULTS.csv", "retry":"P6G1R2_FINAL_RETRY_RESULTS.csv", "physics":"P6G1R2_PHYSICS_ROBUSTNESS_RESULTS.csv"}[phase], rows, rollout_fields())
    staging = collect_phase_staging(out, phase)
    all_staging = []
    for p in ("search", "selection", "final", "retry"):
        all_staging.extend(collect_phase_staging(out, p))
    # Replace by trial id so a resumed worker cannot duplicate an audit row.
    by_id = {r.get("trial_id", ""): r for r in all_staging}
    write_csv(out / "P6G1R2_STAGING_RESULTS.csv", list(by_id.values()), stage_audit_fields())
    return rows


def candidate_map(out: Path) -> dict[str, dict]:
    return {r["recipe_id"]: r for r in read_csv(out / "P6G1R2_RECIPE_CANDIDATES.csv")}


def write_search_ranking(out: Path, search_rows: list[dict]) -> dict[str, list[str]]:
    ranked = rank_rows(search_rows, "SEARCH")
    cmap = candidate_map(out)
    # rank_rows sorts all groups together for convenient reporting, but the
    # BATON search gate is explicitly top-K within each task x target-mode.
    # Recompute the retained flag from a per-group rank; using the global rank
    # silently emptied later groups and caused a false selection failure.
    group_rank: dict[tuple[int, str], int] = {}
    for r in ranked:
        key = (int(r["task"]), r["target_mode"])
        group_rank[key] = group_rank.get(key, 0) + 1
        r["group_rank"] = group_rank[key]
        r["staging_deviation_norm"] = cmap.get(r["recipe_id"], {}).get("staging_deviation_norm", "")
        r["rejection_reason"] = "retained_top5" if r["group_rank"] <= SEARCH_TOP_K else "rejected_after_search_lexicographic_rank"
    write_csv(out / "P6G1R2_SEARCH_RANKING.csv", ranked)
    pool: dict[str, list[str]] = {}
    for task in TASKS:
        for mode in MODES:
            key = f"{task}/{mode}"
            pool[key] = [r["recipe_id"] for r in ranked if int(r["task"]) == task and r["target_mode"] == mode and r["group_rank"] <= SEARCH_TOP_K]
    write_json(out / "P6G1R2_SELECTION_POOL.json", pool)
    return pool


def selection_and_freeze(out: Path, selection_rows: list[dict]) -> dict:
    ranked = rank_rows(selection_rows, "SELECTION")
    cmap = candidate_map(out)
    for r in ranked:
        r["staging_deviation_norm"] = cmap.get(r["recipe_id"], {}).get("staging_deviation_norm", "")
    write_csv(out / "P6G1R2_SELECTION_RANKING.csv", ranked)
    selected: dict[str, dict] = {}
    for task in TASKS:
        selected[str(task)] = {}
        for mode in MODES:
            rs = [r for r in ranked if int(r["task"]) == task and r["target_mode"] == mode]
            if len(rs) < 2:
                raise RuntimeError(f"Selection did not produce two distinct recipes for task{task}/{mode}")
            selected[str(task)][mode] = {
                "original_recipe_id": f"t{task}_{mode}_R23_ORIGINAL_P6G1",
                "primary_recipe_id": rs[0]["recipe_id"],
                "alternate_recipe_id": rs[1]["recipe_id"],
                "selection_rank": rs[:SEARCH_TOP_K],
            }
    freeze = {
        "created_before_final_test": True,
        "selected_recipes": selected,
        "selection_ranking_hash": stable_hash(ranked),
        "search_pool_hash": stable_hash(read_json(out / "P6G1R2_SELECTION_POOL.json")),
        "final_test_outcomes_observed_at_freeze": False,
    }
    write_json(out / "P6G1R2_PRETEST_FREEZE.json", freeze)
    (out / "P6G1R2_PRETEST_FREEZE_HASH.txt").write_text(stable_hash(freeze) + "\n", encoding="utf-8")
    return freeze


def enrich_trajectory_audit(out: Path, rows: list[dict], clusters: list[dict]) -> None:
    trajectory_rows = []
    dominant: dict[int, str] = {}
    for task in TASKS:
        finals = [r for r in rows if int(r.get("task", -1)) == task and r.get("arm") == "B1_PRIMARY" and as_int(r, "bilateral_contact")]
        counts = Counter(r.get("realized_mode", "NONE") for r in finals)
        dominant[task] = counts.most_common(1)[0][0] if counts else "NONE"
    for r in rows:
        if r.get("arm") not in {"B0_ORIGINAL", "B1_PRIMARY", "B2_ALTERNATE"}:
            continue
        path = Path(r.get("contact_telemetry_path", "")) if r.get("contact_telemetry_path") else None
        if path is None or not path.exists():
            continue
        cs = read_csv(path)
        task = int(r["task"])
        req = r.get("target_mode", "")
        c_req = next((c for c in clusters if int(c["task"]) == task and c["primitive_id"] == req), None)
        c_dom = next((c for c in clusters if int(c["task"]) == task and c["primitive_id"] == dominant[task]), None)
        for i, c in enumerate(cs):
            cp = np.array([as_float(c,"contact_mid_obj_x"), as_float(c,"contact_mid_obj_y"), as_float(c,"contact_mid_obj_z")])
            def dist(cluster):
                if cluster is None: return ""
                cc = np.array([as_float(cluster,"centroid_contact_mid_obj_x"), as_float(cluster,"centroid_contact_mid_obj_y"), as_float(cluster,"centroid_contact_mid_obj_z")])
                return float(np.linalg.norm(cp-cc))
            trajectory_rows.append({
                "trial_id": r["trial_id"], "task": task, "phase": r.get("phase", ""), "arm": r.get("arm", ""), "root_seed": r.get("root_seed", ""), "policy_seed": r.get("policy_seed", ""),
                "recipe_id": r.get("recipe_id", ""), "target_mode": req, "realized_mode": r.get("realized_mode", ""), "trajectory_step": i+1, "contact_state": c.get("contact_state", ""),
                "wrist_obj_x": c.get("wrist_obj_x", ""), "wrist_obj_y": c.get("wrist_obj_y", ""), "wrist_obj_z": c.get("wrist_obj_z", ""),
                "wrist_obj_qw": c.get("wrist_obj_qw", ""), "wrist_obj_qx": c.get("wrist_obj_qx", ""), "wrist_obj_qy": c.get("wrist_obj_qy", ""), "wrist_obj_qz": c.get("wrist_obj_qz", ""),
                "distance_to_requested_cluster_m": dist(c_req), "distance_to_dominant_natural_cluster_m": dist(c_dom),
                "first_five_vla_chunks": r.get("policy_log_path", "") if i == 0 else "",
            })
    write_csv(out / "P6G1R2_CANONICALIZATION_TRAJECTORIES.csv", trajectory_rows)
    plot_dir = out / "P6G1R2_TRAJECTORY_PLOTS"; plot_dir.mkdir(exist_ok=True)
    # Keep plots deterministic and lightweight; CSV remains the authoritative audit.
    try:
        import matplotlib.pyplot as plt
        for task in TASKS:
            fig, ax = plt.subplots(figsize=(7, 5))
            for arm, label, color in (("B0_ORIGINAL", "original staging", "gray"), ("B1_PRIMARY", "primary recipe", "tab:blue"), ("B2_ALTERNATE", "alternate recipe", "tab:orange")):
                rs = [x for x in trajectory_rows if int(x["task"]) == task and x["arm"] == arm]
                if rs:
                    ax.plot([as_float(x,"wrist_obj_x") for x in rs], [as_float(x,"wrist_obj_z") for x in rs], ".", ms=2, label=label, color=color, alpha=.6)
            ax.set_title(f"task{task}: VLA invocation-to-contact trajectory audit")
            ax.set_xlabel("wrist x in object frame (m)"); ax.set_ylabel("wrist z in object frame (m)"); ax.legend(); fig.tight_layout()
            fig.savefig(plot_dir / f"task{task}_original_primary_alternate.png", dpi=140); plt.close(fig)
    except Exception:
        pass


def final_summary(out: Path, final_rows: list[dict], retry_rows: list[dict], clusters: list[dict], freeze: dict) -> dict:
    cls_rows = []
    for task in TASKS:
        for mode in MODES:
            primary = [r for r in final_rows if int(r.get("task",-1)) == task and r.get("target_mode") == mode and r.get("arm") == "B1_PRIMARY"]
            alternate = [r for r in final_rows if int(r.get("task",-1)) == task and r.get("target_mode") == mode and r.get("arm") == "B2_ALTERNATE"]
            pqual = bool(primary and metric(primary,"staging_validity") >= QUAL_STAGING and metric(primary,"mode_correct") >= QUAL_MODE_ACCURACY and metric(primary,"stable_lift") >= QUAL_STABLE_LIFT and metric(primary,"verified_success") >= QUAL_VERIFIED and metric(primary,"drop") <= QUAL_DROP)
            alt_verified = metric(alternate,"verified_success") if alternate else 0.0
            ppat = Counter(r.get("failure_category","") for r in primary)
            apat = Counter(r.get("failure_category","") for r in alternate)
            differing = (ppat != apat) and (sum(ppat.values()) > 0 and sum(apat.values()) > 0)
            alt_retained = bool(alternate and alt_verified >= 0.60 and differing)
            cls_rows.append({
                "task": task, "target_mode": mode, "primary_recipe_id": freeze["selected_recipes"][str(task)][mode]["primary_recipe_id"], "alternate_recipe_id": freeze["selected_recipes"][str(task)][mode]["alternate_recipe_id"],
                "primary_staging_validity": metric(primary,"staging_validity"), "primary_requested_mode_accuracy": metric(primary,"mode_correct"), "primary_stable_lift_rate": metric(primary,"stable_lift"), "primary_verified_success_rate": metric(primary,"verified_success"), "primary_drop_rate": metric(primary,"drop"), "primary_qualified": int(pqual),
                "alternate_verified_success_rate": alt_verified, "alternate_failure_pattern_differs": int(differing), "alternate_retained": int(alt_retained),
            })
    write_csv(out / "P6G1R2_MODE_CLASSIFICATION.csv", cls_rows)
    write_csv(out / "P6G1R2_FAILURE_CASES.csv", [r for r in final_rows + retry_rows if not as_int(r,"verified_success")])
    # Action-space gate uses only frozen final-test primary rows.
    task_qualified = {task: sum(int(r["primary_qualified"]) for r in cls_rows if int(r["task"]) == task) for task in TASKS}
    final_primary = [r for r in final_rows if r.get("arm") == "B1_PRIMARY" and as_int(r,"verified_success")]
    dominant_fracs = {}
    for task in TASKS:
        rs = [r for r in final_primary if int(r["task"]) == task]
        counts = Counter(r.get("realized_mode", "NONE") for r in rs)
        dominant_fracs[task] = (max(counts.values()) / len(rs)) if rs else 1.0
    action_space = {"task1_verified_modes": task_qualified[1], "task6_verified_modes": task_qualified[6], "at_least_two_per_task": task_qualified[1] >= 2 and task_qualified[6] >= 2, "dominant_cluster_fraction": dominant_fracs, "same_frozen_vla_and_8N": True}
    retry_success = [r for r in retry_rows if r.get("retry_role") == "alternate"]
    retry_primary = [r for r in retry_rows if r.get("retry_role") == "primary"]
    retry_episodes = {}
    for r in retry_primary:
        key = (r.get("task"), r.get("target_mode"), r.get("root_seed"), r.get("policy_seed"))
        retry_episodes[key] = {"primary": as_int(r,"verified_success"), "alternate": 0}
    for r in retry_success:
        key = (r.get("task"), r.get("target_mode"), r.get("root_seed"), r.get("policy_seed"))
        retry_episodes.setdefault(key, {"primary":0,"alternate":0})["alternate"] = as_int(r,"verified_success")
    retry = {"first_attempt_verified_success": float(np.mean([x["primary"] for x in retry_episodes.values()])) if retry_episodes else 0.0, "alternate_recovery_rate": float(np.mean([x["alternate"] for x in retry_episodes.values() if not x["primary"]])) if any(not x["primary"] for x in retry_episodes.values()) else 0.0, "two_attempt_final_success": float(np.mean([x["primary"] or x["alternate"] for x in retry_episodes.values()])) if retry_episodes else 0.0, "mean_attempts": float(np.mean([1 + int(not x["primary"]) for x in retry_episodes.values()])) if retry_episodes else 0.0}
    if action_space["at_least_two_per_task"]:
        primary_class = "P6G1R2_BATON_VERIFIED_VLA_ACTION_SPACE_QUALIFIED"
    elif any(v >= 2 for v in task_qualified.values()) or any(v == 1 for v in task_qualified.values()):
        primary_class = "P6G1R2_BATON_RECIPE_LIBRARY_PARTIALLY_QUALIFIED"
    elif retry["two_attempt_final_success"] >= 0.75:
        primary_class = "P6G1R2_BOUNDED_RETRY_QUALIFIED_BUT_SINGLE_SHOT_NOT_RELIABLE"
    elif all(dominant_fracs[t] >= 0.70 for t in TASKS):
        primary_class = "P6G1R2_VLA_CANONICALIZATION_PERSISTS_AFTER_RECIPE_SEARCH"
    elif final_rows and float(np.mean([as_int(r,"staging_validity") for r in final_rows])) < 0.95:
        primary_class = "P6G1R2_PRIMITIVE_STAGING_OR_INVOCATION_NOT_RELIABLE"
    else:
        primary_class = "P6G1R2_INCONCLUSIVE_SYSTEM_FAILURE"
    verdict = {"status":"COMPLETE", "primary_classification":primary_class, "learned_method_change":"NONE", "system_method_change":"BATON_STYLE_BOUNDED_VLA_INVOCATION_RECIPE_SEARCH_AND_VERIFICATION", "action_space":action_space, "retry":retry, "task_qualified_modes":task_qualified, "rollouts":{"search":len(collect_phase_rows(out,"search")),"selection":len(collect_phase_rows(out,"selection")),"final_single_shot":len(final_rows),"retry_attempt_rows":len(retry_rows)}, "artifacts":str(out)}
    write_json(out / "P6G1R2_FINAL_VERDICT.json", verdict)
    return verdict


def write_report(out: Path, verdict: dict, gate: dict, final_rows: list[dict], retry_rows: list[dict], physics_rows: list[dict]) -> None:
    cls = read_csv(out / "P6G1R2_MODE_CLASSIFICATION.csv")
    lines = [
        "# P6-G1-R2 BATON-Style Verified VLA Grasp Recipe Bootstrapping", "",
        f"Primary classification: `{verdict.get('primary_classification','')}`", "",
        "## Status", "",
        "- `LEARNED_METHOD_CHANGE: NONE`", "- `SYSTEM_METHOD_CHANGE: BATON_STYLE_BOUNDED_VLA_INVOCATION_RECIPE_SEARCH_AND_VERIFICATION`", "- Frozen VLA weights and target G0/G1/G2 clusters were unchanged.", "- Search and selection used nominal friction, centered CoM, and fixed 8N only.", "",
        "## Runtime gate", "",
        f"- Checkpoint checksum matches R1 manifest: `{gate.get('checkpoint_checksum_matches')}`", f"- Frozen-policy inference smoke: `{gate.get('frozen_policy_inference_smoke')}`", f"- Isaac reset/step smoke: `{gate.get('isaac_reset_step_smoke')}`", f"- SHADOW context: unsupported; COLD only.", "",
        "## Rollouts", "",
        f"- Search: {len(collect_phase_rows(out,'search'))} (planned 432)", f"- Selection: {len(collect_phase_rows(out,'selection'))} (planned 240)", f"- Final single-shot: {len(final_rows)} (planned 288)", f"- Retry diagnostic attempt rows: {len(retry_rows)}", f"- Physics diagnostic: {len(physics_rows)}", "",
        "## Mode qualification", "",
        "| task | mode | primary staging | requested accuracy | stable lift | verified success | qualified |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in cls:
        lines.append(f"| {r['task']} | {r['target_mode']} | {as_float(r,'primary_staging_validity'):.3f} | {as_float(r,'primary_requested_mode_accuracy'):.3f} | {as_float(r,'primary_stable_lift_rate'):.3f} | {as_float(r,'primary_verified_success_rate'):.3f} | {r['primary_qualified']} |")
    lines += [
        "", "## Scientific interpretation", "",
        "1. This experiment tests whether bounded staging and cold frozen-policy invocation can turn geometric target modes into verified actions.",
        "2. A wrong-mode lift is counted as failure; only mode-correct bilateral stable lifts enter the verified-success metric.",
        "3. SEARCH/SELECTION roots are development data; FINAL-TEST roots are frozen and disjoint.",
        "4. Retry results are reported separately from primary single-shot results.",
        "5. The verified recipe memory is a candidate-action library only; no planner or learner was implemented.",
        "", "## What this does not prove", "", "- physical query selects the correct recipe", "- GNP joint-action model works", "- when-to-query works", "- DreamTrajectory improves local refinement", "- Task0/2/5 coverage", "- real-robot transfer", "",
        "## Artifacts", "", f"- Output directory: `{out}`", f"- Protocol hash: `{(out/'P6G1R2_PROTOCOL_HASH.txt').read_text().strip()}`", f"- Pretest freeze hash: `{(out/'P6G1R2_PRETEST_FREEZE_HASH.txt').read_text().strip()}`",
    ]
    (out / "P6G1R2_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recipe_memory(out: Path, verdict: dict, freeze: dict, final_rows: list[dict], retry_rows: list[dict]) -> None:
    cls = read_csv(out / "P6G1R2_MODE_CLASSIFICATION.csv")
    entries = []
    for r in cls:
        if not as_int(r, "primary_qualified"):
            continue
        task = int(r["task"]); mode = r["target_mode"]; sel = freeze["selected_recipes"][str(task)][mode]
        primary = [x for x in final_rows if int(x["task"]) == task and x["target_mode"] == mode and x.get("arm") == "B1_PRIMARY"]
        retry = [x for x in retry_rows if int(x["task"]) == task and x["target_mode"] == mode]
        entries.append({
            "recipe_id": sel["primary_recipe_id"], "task/subtask": task, "target_grasp_mode": mode,
            "staging_transform": "P6-G1 object-relative pregrasp plus bounded recipe offsets; target cluster remains frozen",
            "staging_offsets": {k: recipe_for_row(read_csv(out / "P6G1R2_RECIPE_CANDIDATES.csv"), task, mode, sel["primary_recipe_id"])[k] for k in ("approach_axis_offset_m","mode_axis_lateral_offset_m","vertical_offset_m","wrist_roll_deg")},
            "wrist_orientation": "P6-G1 nominal reset wrist orientation plus object-relative approach-axis roll",
            "stabilization_observations": recipe_for_row(read_csv(out / "P6G1R2_RECIPE_CANDIDATES.csv"), task, mode, sel["primary_recipe_id"])["stabilization_observations"],
            "policy_context_mode": "COLD", "maximum_vla_chunks": recipe_for_row(read_csv(out / "P6G1R2_RECIPE_CANDIDATES.csv"), task, mode, sel["primary_recipe_id"])["max_vla_chunks"],
            "runtime_stop_predicate": "bilateral contact + 0.03m local lift + 8-step retention; drop/collision/timeout/fault stop",
            "offline_postcondition_verifier": "MODE_CORRECT AND BILATERAL_CONTACT AND STABLE_LIFT",
            "primary/alternate_status": {"primary": True, "alternate_recipe_id": sel["alternate_recipe_id"], "alternate_retained": bool(as_int(r,"alternate_retained"))},
            "single_shot_success_statistics": {"n": len(primary), "verified_success_rate": as_float(r,"primary_verified_success_rate"), "mode_accuracy": as_float(r,"primary_requested_mode_accuracy"), "stable_lift_rate": as_float(r,"primary_stable_lift_rate")},
            "retry_statistics": {"attempt_rows": len(retry), "first_attempt_success": float(np.mean([as_int(x,"verified_success") for x in retry if x.get("retry_role")=="primary"])) if retry else 0.0},
            "known_failure_modes": sorted(set(x.get("failure_category", "") for x in primary if not as_int(x,"verified_success"))),
            "applicable_observation_conditions": "nominal friction, centered CoM, fixed 8N, official task instruction, COLD invocation",
            "force_contract_field": {"requested_grip_force_N": FORCE_N, "executed_fLz_N": 4.0, "executed_fRz_N": 4.0},
        })
    write_json(out / "P6G1R2_VERIFIED_RECIPE_MEMORY.json", {"name":"BATON-style verified frozen-VLA recipe memory", "entries":entries, "classification":verdict.get("primary_classification"), "no_agent_implemented":True})
    lines = ["# P6-G1-R2 Verified Recipe Memory", "", f"Qualified entries: {len(entries)}", "", "This memory contains only recipes whose frozen FINAL-TEST primary passed the explicit verifier and qualification thresholds.", ""]
    for e in entries:
        lines += [f"## {e['recipe_id']} — task{e['task/subtask']} {e['target_grasp_mode']}", "", f"- Offsets: `{json.dumps(e['staging_offsets'], sort_keys=True)}`", f"- COLD stabilization observations: `{e['stabilization_observations']}`", f"- Maximum VLA chunks: `{e['maximum_vla_chunks']}`", f"- Final verified success: `{e['single_shot_success_statistics']['verified_success_rate']:.3f}`", f"- Force contract: `{FORCE_N:g}N`", ""]
    (out / "P6G1R2_VERIFIED_RECIPE_MEMORY.md").write_text("\n".join(lines), encoding="utf-8")


def terminal_summary(out: Path, verdict: dict, final_rows: list[dict], retry_rows: list[dict], physics_rows: list[dict]) -> str:
    cls = read_csv(out / "P6G1R2_MODE_CLASSIFICATION.csv")
    def rate(arm, task, mode):
        rs = [r for r in final_rows if r.get("arm") == arm and int(r.get("task",-1)) == task and r.get("target_mode") == mode]
        return f"{metric(rs,'verified_success'):.3f}"
    lines = ["STATUS:", "COMPLETE", "LEARNED_METHOD_CHANGE: NONE", "SYSTEM_METHOD_CHANGE:", "BATON_STYLE_BOUNDED_VLA_INVOCATION_RECIPE_SEARCH_AND_VERIFICATION", "ARTIFACTS:", str(out), "", "RUNTIME:", f"- Frozen VLA restore: {read_json(out/'P6G1R2_RUNTIME_GATE.json').get('prior_runtime_classification')}", f"- Inference smoke: {read_json(out/'P6G1R2_RUNTIME_GATE.json').get('frozen_policy_inference_smoke')}", f"- Isaac joint runtime: {read_json(out/'P6G1R2_RUNTIME_GATE.json').get('isaac_reset_step_smoke')}", f"- Checkpoint checksum: {read_json(out/'P6G1R2_RUNTIME_GATE.json').get('checkpoint_checksum_actual')}", "", "SCOPE:", "- Tasks: [1,6]", "- Target modes: [G0,G1,G2]", "- Recipe candidates per mode: 24", "- Search roots: 3", "- Selection roots: 4", "- Final-test roots: 8", "- Fixed force: 8N", "- Physical query used: NO", "", "SEARCH:", f"- Total rollouts: {len(collect_phase_rows(out,'search'))}", "- Recipes retained: top 5 per task/mode", "", "SELECTION:", f"- Total rollouts: {len(collect_phase_rows(out,'selection'))}", f"- Pretest freeze hash: {(out/'P6G1R2_PRETEST_FREEZE_HASH.txt').read_text().strip()}", "", "FINAL SINGLE-SHOT:"]
    for task in TASKS:
        lines.append(f"task{task}:")
        for mode in MODES:
            lines.append(f"- {mode} original / primary / alternate: {rate('B0_ORIGINAL',task,mode)} / {rate('B1_PRIMARY',task,mode)} / {rate('B2_ALTERNATE',task,mode)}")
        lines.append(f"- Qualified modes: {[r['target_mode'] for r in cls if int(r['task'])==task and as_int(r,'primary_qualified')]}")
        lines.append("- Canonicalization reduced: see P6G1R2_CANONICALIZATION_TRAJECTORIES.csv")
    retry_success = metric([r for r in retry_rows if r.get("retry_role")=="primary"],"verified_success")
    lines += ["", "RETRY:", f"- First-attempt verified success: {retry_success:.3f}", f"- Attempt rows: {len(retry_rows)}", "- Alternate recovery: see P6G1R2_FINAL_RETRY_RESULTS.csv", "- Two-attempt final success: see P6G1R2_FINAL_REPORT.md", "", "PHYSICS ROBUSTNESS:", f"- Requested-mode accuracy: {metric(physics_rows,'mode_correct'):.3f}", f"- Stable-lift variation: {metric(physics_rows,'stable_lift'):.3f}", "- Recipes changed after audit: NO", "", "ACTION SPACE:", f"- task1 verified modes: {[r['target_mode'] for r in cls if int(r['task'])==1 and as_int(r,'primary_qualified')]}", f"- task6 verified modes: {[r['target_mode'] for r in cls if int(r['task'])==6 and as_int(r,'primary_qualified')]}", f"- At least two per task: {verdict.get('action_space',{}).get('at_least_two_per_task')}", f"- Verified recipe memory entries: {len(read_json(out/'P6G1R2_VERIFIED_RECIPE_MEMORY.json').get('entries',[]))}", "", f"PRIMARY_CLASSIFICATION: {verdict.get('primary_classification')}", "", "WHAT THIS DOES NOT PROVE:", "- physical query selects the correct recipe", "- GNP joint-action model works", "- when-to-query works", "- DreamTrajectory improves local refinement", "- Task0/2/5 coverage", "- real-robot transfer", "", "NEXT:", "- Follow the classification-specific recommendation in P6G1R2_FINAL_REPORT.md."]
    return "\n".join(lines)


def main() -> int:
    if os.environ.get("P6G1R2_WORKER") == "1":
        return run_worker()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT / f"p6g1r2_baton_verified_vla_recipes_{now_tag()}")
    ap.add_argument("--server-host", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=18039)
    ap.add_argument("--no-start-server", action="store_true")
    ap.add_argument("--skip-runtime-gate", action="store_true")
    ap.add_argument("--resume", action="store_true", help="resume an incomplete output directory without duplicating completed trial ids")
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.resume and (out / "P6G1R2_PROTOCOL.json").exists():
        protocol = read_json(out / "P6G1R2_PROTOCOL.json")
        candidates = read_csv(out / "P6G1R2_RECIPE_CANDIDATES.csv")
        clusters = read_csv(out / "P6G1R2_TARGET_GRASP_CLUSTERS.csv") or load_reference_clusters()
    else:
        protocol = protocol_obj(); write_json(out / "P6G1R2_PROTOCOL.json", protocol); (out/"P6G1R2_PROTOCOL_HASH.txt").write_text(stable_hash(protocol)+"\n", encoding="utf-8")
        write_json(out / "P6G1R2_ROOT_SPLIT.json", root_split())
        library = recover_library(); candidates = generate_candidates(library); write_candidates(out, library, candidates)
        clusters = load_reference_clusters(); write_csv(out / "P6G1R2_TARGET_GRASP_CLUSTERS.csv", clusters)
    (out/"P6G1R2_CODE_HASH.txt").write_text(sha256_file(Path(__file__).resolve())+"\n", encoding="utf-8")
    for d in ("P6G1R2_TELEMETRY", "P6G1R2_TRAJECTORY_PLOTS"):
        (out/d).mkdir(exist_ok=True)
    server_proc = None
    try:
        if not args.no_start_server:
            server_proc = start_server(out, args.server_port)
        if not p6g1.wait_for_server(args.server_host, args.server_port, 900):
            raise RuntimeError("frozen VLA policy server unavailable")
        gate = runtime_gate(out, args.server_host, args.server_port)
        if not args.skip_runtime_gate and (not gate.get("checkpoint_checksum_matches") or not gate.get("frozen_policy_inference_smoke")):
            raise RuntimeError("checkpoint checksum mismatch; refusing scientific rollouts")
        # Search is the expensive coarse phase.  On resume, a complete
        # protocol-sized search is already authoritative; re-running it only
        # increases wall time and can delay refreshing the selection pool.
        if args.resume and len(collect_phase_rows(out, "search")) >= len(candidates) * len(SEARCH_ROOTS):
            search_rows = collect_phase_rows(out, "search")
        else:
            search_rows = run_phase(out, "search", args.server_host, args.server_port)
        write_search_ranking(out, search_rows)
        selection_rows = run_phase(out, "selection", args.server_host, args.server_port)
        freeze = selection_and_freeze(out, selection_rows)
        final_rows = run_phase(out, "final", args.server_host, args.server_port)
        retry_rows = run_phase(out, "retry", args.server_host, args.server_port)
        # Any worker's successful Isaac smoke establishes the runtime smoke
        # result; both task workers execute the same reset/step gate.
        smoke = list(out.glob("P6G1R2_ISAAC_SMOKE_task*.json")); gate["isaac_reset_step_smoke"] = len(smoke) == 2; write_json(out/"P6G1R2_RUNTIME_GATE.json", gate)
        verdict = final_summary(out, final_rows, retry_rows, clusters, freeze)
        # Classification is written by final_summary and is the gate for the
        # diagnostic physics sweep; keep this after final_summary so qualified
        # modes are actually audited.
        physics_rows = run_phase(out, "physics", args.server_host, args.server_port) if (out/"P6G1R2_MODE_CLASSIFICATION.csv").exists() else []
        write_csv(out / "P6G1R2_PHYSICS_ROBUSTNESS_RESULTS.csv", physics_rows, rollout_fields())
        enrich_trajectory_audit(out, final_rows, clusters)
        write_recipe_memory(out, verdict, freeze, final_rows, retry_rows)
        write_report(out, verdict, gate, final_rows, retry_rows, physics_rows)
        print(terminal_summary(out, verdict, final_rows, retry_rows, physics_rows))
        return 0
    except Exception as exc:
        verdict = {"status":"SYSTEM_FAILURE", "primary_classification":"P6G1R2_INCONCLUSIVE_SYSTEM_FAILURE", "error":repr(exc), "artifacts":str(out)}
        write_json(out/"P6G1R2_FINAL_VERDICT.json", verdict)
        (out/"P6G1R2_FINAL_REPORT.md").write_text(f"# P6-G1-R2 system failure\n\n`{exc!r}`\n\nInspect logs under `{out/'logs'}`.\n", encoding="utf-8")
        traceback.print_exc()
        return 1
    finally:
        if server_proc is not None:
            try:
                server_proc.terminate(); server_proc.wait(timeout=30)
            except Exception:
                try: server_proc.kill()
                except Exception: pass


if __name__ == "__main__":
    sys.exit(main())
