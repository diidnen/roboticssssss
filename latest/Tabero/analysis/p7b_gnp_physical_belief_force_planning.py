#!/usr/bin/env python3
"""P7-B: physical-belief force planning.

The collector keeps the P7-A fixed recipe, checkpoint and force controller,
adds one reversible P4-B contact-frame query before frozen-VLA completion, and
branches every force from the exact post-query simulator state.  The offline
stage trains the outcome-set teacher and query posterior, evaluates evidence
perturbations, and performs the finite-force Monte-Carlo planner.

The script is intentionally resumable.  Isaac collection is selected with
``--phase pilot``/``main``/``fresh``; ``--offline`` never launches Isaac.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
P7A = RESULTS / "p7a_fixed_invocation_force_frontier_20260826_153426"
P7A_PROTOCOL = P7A / "P7A_PROTOCOL.json"
P6G1R2 = REPO / "analysis/p6g1r2_baton_verified_vla_recipes.py"
P6G1R1 = REPO / "analysis/p6g1r1_controller_grasp_vla_handoff.py"
P6G0 = REPO / "analysis/p6g0_grasp_force_physics_benchmark.py"
P4_SCRIPT = RESULTS / "p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py"
P6G1R2_OUT = RESULTS / "p6g1r2_baton_verified_vla_recipes_20260825_205113"
P6G1R3_OUT = RESULTS / "p6g1r3_raw_vla_vs_fixed_recipe_20260826_123552"
CLUSTERS = RESULTS / "p6g1_primitive_ik_vla_grasp_realization_20260825_103146/P6G1_REFERENCE_GRASP_CLUSTERS.csv"

TASK = 1
OBJECT = "cream_cheese_1"
BASKET = "basket_1"
INSTRUCTION = "pick up the cream cheese and place it in the basket"
FORCES = [5.0, 6.0, 7.0, 8.0]
TRAIN_FORCES = [5.0, 6.0, 8.0]
HELDOUT_FORCE = 7.0
FRICTION_STRATA = [(0.20, 0.32), (0.32, 0.48), (0.48, 0.64), (0.64, 0.80), (0.80, 0.95)]
PILOT_ROOTS = [10000, 10001, 10002, 10003]
CLEAN_PILOT_ROOTS = [10330, 10331, 10332, 10333]
MAIN_ROOT_GROUPS = list(range(10100, 10120))
FRESH_ROOT_GROUPS = list(range(10200, 10210))
POLICY_REPEATS = [0, 1]
QUERY_PRELOAD_N = 3.0
QUERY_MAX_MM = 2.0
QUERY_STEP_MM = 0.2
QUERY_RETURN_STEPS = 10
QUERY_POST_HOLD_STEPS = 5
QUERY_RHO_IMPULSE_TARGET = 0.012
QUERY_RHO_CAP = 0.08
QUERY_RELATIVE_NORMAL_ALPHA = 0.55
QUERY_MARKER_NORM_STOP = 0.12
P4B_QUERY_TARGET_OBJECT_M = np.array([0.0, 0.0, 0.018], dtype=np.float64)
# Fixed-G2 handoff geometry: after G2 staging has settled at the P4-B
# pregrasp height, a commanded +18 mm target reaches a lower realized TCP
# height than the historical fast-from-reset P4-B descent.  +28 mm restores
# the validated P4-B contact clearance in world space without changing the
# probe excitation, force protocol, or downstream scientific stages.
P7B_HANDOFF_GRASP_Z_M = 0.028
# The P4-B controller can remain a few millimetres above the commanded TCP
# point while holding a loaded object.  Historical P4-B telemetry shows this
# as normal contact compliance, so return validity must use a physical
# tolerance rather than treating the command/actual residual as a hard
# failure.
P4B_RETURN_POS_TOL_M = 0.005
ROOT_SETTLE_STEPS = 20
FMAX = 8.0
MC_SAMPLES = 128
MODEL_SEEDS = [0, 1, 2, 3, 4]

ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
SERVER_PYTHON = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/.venv/bin/python")
POLICY_DIR = Path("/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999")
POLICY_CONFIG = "pi0_lora_tacfield_tabero"
B5_RESULTS = RESULTS / "b5_tabero_neutral_20260822_040652"
B5_WRAPPER = B5_RESULTS / "scripts/b5_serve_policy_with_explicit_norm_stats.py"
B5_STUBS = B5_RESULTS / "scripts/stubs"
TABERO_VTLA = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64")

FEATURES = [
    "t_s", "force_target", "measured_squeeze", "target_normal_force", "measured_fn", "measured_ft",
    "ft_over_fn", "force_imbalance", "force_imbalance_ratio", "gripper_opening", "commanded_tangent_increment_mm",
    "accumulated_displacement_mm", "marker_motion", "marker_tangential", "marker_velocity", "marker_loading_unloading",
    "contact_left", "contact_right", "tactile_ok", "eef_dx", "eef_dy", "eef_dz",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader(); wr.writerows(rows)


def repair_context_metadata(out: Path, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backfill manifest fields from already-written query telemetry.

    This is for interrupted/resumed engineering runs only.  It never reruns a
    query or creates a branch; it makes a context row auditable when a worker
    was interrupted after writing its telemetry and before writing the full
    result record.
    """
    if os.environ.get("P7B_REPAIR_METADATA", "0").lower() not in {"1", "true", "yes"}:
        return contexts
    changed = False
    for c in contexts:
        path = Path(str(c.get("query_telemetry_path", "")))
        if not path.exists() and path.name:
            local_path = out / "P7B_QUERY_TELEMETRY" / path.name
            if local_path.exists():
                path = local_path
                c["query_telemetry_path"] = str(local_path)
        if not path.exists():
            continue
        rows = read_csv(path)
        probe = [r for r in rows if r.get("phase") in {"probe_out", "probe_back", "probe_hold"}]
        hold = [r for r in rows if r.get("phase") == "probe_hold"]
        if not rows or not probe or not hold:
            continue
        last = rows[-1]
        h5 = hold[-5:]
        try:
            err = math.sqrt(sum((f(last, k + "_actual") - f(h5[-1], k + "_command")) ** 2 for k in ("eef_x", "eef_y", "eef_z")))
        except Exception:
            err = float("nan")
        c.update({
            "query_predicate_version": "P4B_probe_failure_return_valid_v3",
            "probe_duration_s": len(probe) * (f(rows[1], "t_s", 0.05) - f(rows[0], "t_s", 0.0) if len(rows) > 1 else 0.05),
            "contact_frame_source": "projected_object_to_basket_transport",
            "target_preload_N": QUERY_PRELOAD_N,
            "rho_impulse": sum(f(r, "ft_over_fn", 0.0) for r in probe) * (f(rows[1], "t_s", 0.05) - f(rows[0], "t_s", 0.0) if len(rows) > 1 else 0.05),
            "return_state_valid": int(all(r.get("contact_state") == "bilateral" for r in h5) and err <= P4B_RETURN_POS_TOL_M and not any(i(r, "dropped", 0) for r in h5)),
            "return_position_error_mm": err * 1000.0,
            "query_true_runtime_error": "",
        })
        objs = np.asarray([[f(r, "object_x_w"), f(r, "object_y_w"), f(r, "object_z_w")] for r in probe], dtype=float)
        if np.isfinite(objs).all():
            c["obj_disp_probe_m"] = float(np.max(np.linalg.norm(objs - objs[0], axis=1)))
        c["actual_displacement_mm"] = max(float(c.get("actual_displacement_mm") or 0.0), max(f(r, "accumulated_displacement_mm", 0.0) for r in rows if r.get("phase") == "probe_out") if any(r.get("phase") == "probe_out" for r in rows) else 0.0)
        c["major_disturbance"] = int(float(c.get("obj_disp_probe_m") or 0.0) > 0.005)
        if int(c.get("return_state_valid") or 0) and int(c.get("contact_retained") or 0):
            c["query_qualified"] = 1; c["query_failure"] = 0; c["query_failure_reason"] = ""
        changed = True
    if changed:
        write_csv(out / "P7B_CONTEXT_MANIFEST.csv", contexts, CONTEXT_FIELDS)
    return contexts


def append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if not exists: wr.writeheader()
        wr.writerow(row); fh.flush(); os.fsync(fh.fileno())


def f(row: dict, key: str, default=float("nan")) -> float:
    try:
        v = row.get(key, default)
        return default if v in (None, "") else float(v)
    except Exception:
        return default


def i(row: dict, key: str, default=0) -> int:
    try:
        v = row.get(key, default)
        return default if v in (None, "") else int(float(v))
    except Exception:
        return default


def sha_state(state: Any) -> str:
    try:
        import torch
        chunks: list[bytes] = []
        def visit(x):
            if isinstance(x, dict):
                for k in sorted(x, key=str): chunks.append(str(k).encode()); visit(x[k])
            elif isinstance(x, (list, tuple)):
                for y in x: visit(y)
            elif isinstance(x, torch.Tensor):
                chunks.append(x.detach().cpu().contiguous().numpy().tobytes())
            else: chunks.append(repr(x).encode())
        visit(state)
        return hashlib.sha256(b"".join(chunks)).hexdigest()
    except Exception:
        return stable_hash(state)


ROOT_FIELDS = ["root_group_id", "root_seed", "root_state_hash", "root_index", "source", "split"]
CONTEXT_FIELDS = [
    "context_id", "root_group_id", "root_seed", "task", "hidden_friction_analysis_only", "friction_stratum",
    "split", "fixed_recipe_id", "query_qualified", "contact_retained", "drop", "query_failure", "stop_reason",
    "actual_displacement_mm", "post_query_state_hash", "query_telemetry_path", "pre_query_state_hash",
    "query_failure_reason", "query_predicate_version", "probe_duration_s", "contact_frame_source",
    "target_preload_N", "rho_impulse", "obj_disp_probe_m", "obj_rot_probe_rad", "major_disturbance",
    "return_state_valid", "return_position_error_mm", "query_true_runtime_error",
    "root_restore_state_hash", "root_restore_parity", "infra_failure", "infra_retry_count",
]
BRANCH_FIELDS = [
    "branch_id", "context_id", "root_group_id", "root_seed", "task", "split", "policy_repeat", "requested_force_N",
    "measured_force_mean_N", "measured_force_peak_N", "tracking_mae_N", "state_parity", "restore_hash", "restore_query_state_hash", "query_state_parity",
    "staging_validity", "full_task_success_y", "stable_lift", "transport_success", "placement_success", "drop",
    "slip", "failure_stage", "num_policy_chunks", "episode_steps", "query_once_shared", "label_source",
    "contact_telemetry_path", "force_telemetry_path", "policy_log_path", "vla_chunk_budget_exhausted", "error",
]
PARITY_FIELDS = ["branch_id", "context_id", "root_state_hash", "restore_state_hash", "post_query_state_hash", "restore_query_state_hash", "parity_pass", "query_parity_pass"]


def friction_value(root_group: int, stratum: int) -> float:
    lo, hi = FRICTION_STRATA[stratum]
    rng = np.random.default_rng(700000 + root_group * 17 + stratum)
    return float(rng.uniform(lo, hi))


def split_for_group(group: int, phase: str) -> str:
    if phase in {"pilot", "clean_pilot"}: return "PILOT"
    if phase == "fresh": return "FRESH"
    idx = MAIN_ROOT_GROUPS.index(group)
    return "TRAIN" if idx < 12 else "DEV" if idx < 16 else "TEST"


def protocol(out: Path, phase: str) -> dict:
    p7a = read_json(P7A_PROTOCOL)
    groups = CLEAN_PILOT_ROOTS if phase == "clean_pilot" else PILOT_ROOTS if phase == "pilot" else FRESH_ROOT_GROUPS if phase == "fresh" else MAIN_ROOT_GROUPS
    return {
        "name": "P7-B GNP-Style Physical Belief for Full-Task Force Planning",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_change": "GNP_STYLE_OUTCOME_TEACHER_TO_QUERY_POSTERIOR_DISTILLATION; P4B_QUERY_COMPATIBILITY_APPROACH_FROM_FIXED_G2_PREGRASP_TO_P4B_CENTER_TARGET",
        "primary_task": TASK, "object": OBJECT, "fixed_invocation_protocol": "t1_G2_R23_ORIGINAL_P6G1",
        "p7a_source": str(P7A), "p7a_protocol_hash": sha256_file(P7A_PROTOCOL),
        "runtime_action_space": ["grip force only"], "forces_N": FORCES, "training_forces_N": TRAIN_FORCES,
        "held_out_force_N": HELDOUT_FORCE, "friction_strata": FRICTION_STRATA,
        "hidden_friction_is_model_input": False, "query": {
            "implementation": "P4-B common contact-frame shear inserted after fixed G2 staging; each friction context restores the same settled root once, then its validated 45-step approach is replayed to the object-root center pregrasp, with a fixed +28 mm handoff grasp height preserving P4-B realized fingertip clearance before the unchanged descend/close/hold/probe/return",
            "implementation_source": str(P4_SCRIPT), "implementation_source_sha256": sha256_file(P4_SCRIPT),
            "preload_N": QUERY_PRELOAD_N, "max_displacement_mm": QUERY_MAX_MM, "step_mm": QUERY_STEP_MM,
            "handoff_grasp_z_m": P7B_HANDOFF_GRASP_Z_M, "historical_p4_command_grasp_z_m": float(P4B_QUERY_TARGET_OBJECT_M[2]),
            "return_position_tolerance_mm": P4B_RETURN_POS_TOL_M * 1000.0,
            "single_query_per_context": True, "reversible": True,
        },
        "query_qualification_gate": {
            "frozen_before_collection": True, "min_query_qualified": 20,
            "min_query_qualification_rate": 1.0, "min_contact_retention": 20,
            "require_return_state_valid": True, "require_all_state_parity": True,
            "max_true_runtime_errors": 0, "chunk_budget_exhaustion_is_runtime_error": False,
        },
        "pilot": {"root_groups": groups, "contexts": len(groups) * 5, "branches": len(groups) * 5 * 4},
        "main": {"root_groups": groups, "contexts": len(groups) * 5, "branches": len(groups) * 5 * 4 * 2, "split": {"TRAIN": 12, "DEV": 4, "TEST": 4}},
        "fresh": {"root_groups": groups, "contexts": len(groups) * 5, "arms": ["fixed_8N", "no_query_prior", "query_control", "friction_oracle", "most_likely", "monte_carlo"]},
        "force_holdout": "7N outcomes are collected on DEV/TEST but excluded from teacher input, all training loss, selection and calibration until final held-out evaluation",
        "frozen_checkpoint": p7a.get("frozen_policy", {}),
        "prohibited": ["task6 primary", "grasp-position selection", "recipe search", "retry", "DreamTrajectory", "VLA fine-tuning", "RL", "learned query selection", "explicit friction supervision for main model"],
        "output_directory": str(out),
    }


def init_artifacts(out: Path, phase: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    existing_protocol = (out / "P7B_PROTOCOL.json").exists()
    proto = protocol(out, phase)
    if not existing_protocol:
        write_json(out / "P7B_PROTOCOL.json", proto)
    else:
        proto = read_json(out / "P7B_PROTOCOL.json")
    (out / "P7B_PROTOCOL_HASH.txt").write_text(sha256_file(out / "P7B_PROTOCOL.json") + "\n")
    (out / "P7B_CODE_HASH.txt").write_text(sha256_file(Path(__file__)) + "\n")
    for d in ["P7B_QUERY_TELEMETRY", "P7B_BRANCH_TELEMETRY", "P7B_FORCE_ONLY_PRIOR", "P7B_FRICTION_BASELINE", "P7B_DETERMINISTIC_GRU", "P7B_OUTCOME_GNP", "P7B_QUERY_GNP_GENERIC", "P7B_QUERY_GNP_THRESHOLD", "P7B_FULL_TEACHER", "logs"]:
        (out / d).mkdir(exist_ok=True)
    write_json(out / "P7B_FEATURE_MANIFEST.json", {"features": FEATURES, "excluded": ["hidden_friction", "branch force measurement", "future labels", "root id", "object private pose"], "force_representation": "normalized scalar 2*((F-5)/(8-5))-1", "query_channels": "force/tactile/proprio/action evidence"})
    write_json(out / "P7B_NORMALIZATION.json", {"fit": "TRAIN query traces only; finalized after data collection", "feature_names": FEATURES})
    write_json(out / "P7B_MODEL_CONFIGS.json", {"seeds": MODEL_SEEDS, "latent_dims": [4, 8], "beta": [0.01, 0.1], "gamma": [0.01, 0.1], "mc_samples": MC_SAMPLES})
    for name, fields in [("P7B_ROOT_MANIFEST.csv", ROOT_FIELDS), ("P7B_CONTEXT_MANIFEST.csv", CONTEXT_FIELDS), ("P7B_BRANCH_MANIFEST.csv", BRANCH_FIELDS), ("P7B_STATE_PARITY.csv", PARITY_FIELDS)]:
        if not (out / name).exists():
            write_csv(out / name, [], fields)


def import_runtime_modules(out: Path, task: int = TASK):
    os.environ["P4_TASK_ID"] = str(task); os.environ["P4_VARIANT"] = "P4B"; os.environ["P4_OUT"] = str(out / "P4B_IMPORT"); os.environ["P4_RESUME"] = "0"
    r2 = load_module(P6G1R2, f"p7b_r2_{task}")
    r1 = load_module(P6G1R1, f"p7b_r1_{task}")
    p6 = load_module(P6G0, f"p7b_p6g0_{task}")
    p4 = p6.import_p4_probe(task)
    return r2, r1, p6, p4


def q_float(x, default=0.0):
    try:
        import torch
        if isinstance(x, torch.Tensor): return float(x.reshape(-1)[0].detach().cpu().item())
    except Exception: pass
    try: return float(np.asarray(x).reshape(-1)[0])
    except Exception: return default


def restore_scene_state_stable(env, state: Any, env_ids: Any) -> None:
    """Restore a saved scene without re-entering ManagerBasedEnv.reset_to.

    ``ManagerBasedEnv.reset_to`` calls ``_reset_idx`` (including sensor reset
    callbacks) and may render RTX cameras before the observation manager is
    recomputed.  Repeating that callback/render sequence while a GelSight
    camera is already active triggered Isaac's non-recursive tasking mutex in
    the sequential stratum loop.  The saved P7-B state is restored directly
    through the scene API, which writes the same articulation/object state and
    controller targets, then forwards the simulator and refreshes manager
    buffers.  No reset event or randomization is applied here.
    """
    env.scene.reset_to(copy.deepcopy(state), env_ids, is_relative=True)
    env.sim.forward()
    env.observation_manager.reset(env_ids)
    env.action_manager.reset(env_ids)
    env.obs_buf = env.observation_manager.compute(update_history=True)


def query_from_contact(
    env,
    p6,
    p4,
    task: int,
    primitive: dict,
    trial_id: str,
    dt: float,
    out: Path,
    nominal_eef_aa: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Run the validated P4-B query after the fixed P7-B pre-grasp stage."""
    obs = env.observation_manager.compute(); eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy(); cmd = eef[:3].copy()
    # P4-B freezes the nominal wrist orientation captured immediately after
    # root reset.  Re-reading the post-G2 staging orientation introduces a
    # residual IK error which is amplified by the 103.4 mm EE offset and can
    # change which fingertip loads first.  Keep the staged position, but use
    # the same orientation path as P4-B.
    eef_aa = np.asarray(nominal_eef_aa, dtype=np.float32).copy() if nominal_eef_aa is not None else p4._aa(eef[3:7])
    # P4-B's validated physical query is centered on the object root.  The
    # fixed G2 invocation remains the staging/handoff protocol, but its
    # +20.718 mm g_plus offset is not silently substituted for the P4-B query
    # target.  This is the explicit compatibility change recorded in the
    # protocol above; it is not force-outcome adaptation.
    # P4-B's successful implementation does not apply the object's rotation
    # to these vertical offsets.  Its contract is a point in the current
    # robot-base frame: object position in base + base-Z grasp depth.  Using
    # object_point_to_base([0, 0, z]) here silently changes the approach for
    # tilted/yawed object roots and was the P7-B/P4-B geometry divergence.
    obj_b, _obj_q = p4._pose_in_base(env, OBJECT)
    # The original P4-B command is +18 mm above the object root.  After a
    # settled G2 handoff the same command reaches a different realized TCP
    # height than P4-B's fast-from-reset descent; the optional override is
    # used only during engineering validation of that transition geometry.
    grasp_z = float(os.environ.get("P7B_QUERY_GRASP_Z_M", str(P7B_HANDOFF_GRASP_Z_M)))
    grasp_target = obj_b.copy(); grasp_target[2] += grasp_z
    contact_normal, tangent, _binormal, tangent_source, _obj_b, _basket_b = p4._current_contact_frame(env)
    preload_target = float(p4.P4B_BASE_FORCE_N); d_pred = p4.D_OPEN; accumulated_m = 0.0
    rows: list[dict] = []; rho_impulse = 0.0; stop = "completed"; dropped = 0; contact_lost = 0; probe_failure = 0; terminated = False
    prev_marker = None; marker_at_forward0 = None; probe_indices: list[int] = []; f_hist=[]; fn_hist=[]; ft_hist=[]; rho_hist=[]; markers=[]
    obj_probe0 = None; q_probe0 = None; obj_disp_probe = 0.0; obj_rot_probe = 0.0; step = 0

    def record(phase: str, inc_mm: float, is_probe: bool):
        nonlocal cmd, step, d_pred, dropped, contact_lost, probe_failure, rho_impulse, prev_marker, marker_at_forward0, terminated, stop, obj_probe0, q_probe0, obj_disp_probe, obj_rot_probe
        force_cmd = 0.0 if phase in {"approach", "descend"} else preload_target
        action = p4._make_action(cmd, eef_aa, d_pred, force_cmd, env.device)
        obs2, _, term, trunc, _ = env.step(action); step += 1
        ff = obs2["policy"]["gripper_net_force"][0]; ff = ff[-1] if ff.ndim == 3 else ff; a = ff.detach().cpu().numpy(); fl, fr = a[0], a[1]
        dbg = p4._dbg(env)
        fs = q_float(dbg.get("f_sq_meas")); fs_raw = q_float(dbg.get("f_sq_meas_raw"), np.nan)
        d_actual = q_float(dbg.get("d_actual"), np.nan); d_cmd_actual = q_float(dbg.get("d_cmd"), np.nan)
        fn = 2*min(abs(float(fl[2])), abs(float(fr[2]))); ft = float(np.linalg.norm(np.array([fl[0]+fr[0], fl[1]+fr[1]]))); rho = ft/max(fn,1e-6)
        imb = abs(float(np.linalg.norm(fl))-float(np.linalg.norm(fr))); imb_ratio = imb/max(float(np.linalg.norm(fl))+float(np.linalg.norm(fr)),1e-6)
        cl = int(np.linalg.norm(fl)>p4.FINGER_FORCE_MIN_N and abs(fl[2])>p4.FINGER_FORCE_MIN_N); cr = int(np.linalg.norm(fr)>p4.FINGER_FORCE_MIN_N and abs(fr[2])>p4.FINGER_FORCE_MIN_N)
        try:
            gripper_pos_actual = obs2["policy"]["gripper_pos"][0].detach().cpu().numpy().reshape(-1)
            opening = float(gripper_pos_actual[0]) if len(gripper_pos_actual) else np.nan
            gripper_left_actual = float(gripper_pos_actual[0]) if len(gripper_pos_actual) > 0 else np.nan
            gripper_right_actual = float(gripper_pos_actual[1]) if len(gripper_pos_actual) > 1 else np.nan
        except Exception:
            opening = gripper_left_actual = gripper_right_actual = np.nan
        # Keep the articulated-joint values separate from the observation's
        # signed opening convention.  Panda's second finger is reported as
        # ``-joint_pos`` by the standard gripper_pos observation, so this is
        # the unambiguous controller/physics evidence for a handoff bug.
        gripper_joint_pos = [np.nan, np.nan]
        gripper_joint_target = [np.nan, np.nan]
        gripper_joint_names = ["", ""]
        try:
            arm_term = env.action_manager.get_term("arm_action")
            ids = list(arm_term._gripper_joint_ids)
            names = list(arm_term._gripper_joint_names)
            gripper_joint_names = [str(x) for x in names[:2]] + [""] * max(0, 2-len(names))
            robot = env.scene["robot"]
            q = robot.data.joint_pos[0, ids].detach().cpu().numpy().reshape(-1)
            gripper_joint_pos = [float(x) for x in q[:2]] + [np.nan] * max(0, 2-len(q))
            if hasattr(robot.data, "joint_pos_target"):
                qt = robot.data.joint_pos_target[0, ids].detach().cpu().numpy().reshape(-1)
                gripper_joint_target = [float(x) for x in qt[:2]] + [np.nan] * max(0, 2-len(qt))
        except Exception:
            pass
        tac = p4._tactile_summaries(obs2); marker = tac["marker_mean"]; vel = 0.0 if prev_marker is None else (marker-prev_marker)/dt; prev_marker=marker
        if marker_at_forward0 is None and is_probe: marker_at_forward0 = marker
        try: dropped = int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
        except Exception: pass
        if is_probe:
            probe_indices.append(len(rows)); f_hist.append(fs); fn_hist.append(fn); ft_hist.append(ft); rho_hist.append(rho); markers.append(marker); rho_impulse += rho*dt
            obj = env.scene[OBJECT].data.root_pos_w[0].detach().cpu().numpy(); q = env.scene[OBJECT].data.root_quat_w[0].detach().cpu().numpy()
            if obj_probe0 is None:
                obj_probe0 = obj.copy(); q_probe0 = q.copy()
            else:
                obj_disp_probe=max(obj_disp_probe,float(np.linalg.norm(obj-obj_probe0))); qdot=abs(float(np.dot(q_probe0/max(np.linalg.norm(q_probe0),1e-12),q/max(np.linalg.norm(q),1e-12)))); obj_rot_probe=max(obj_rot_probe,float(2*math.acos(np.clip(qdot,-1,1))))
            if cl == 0 or cr == 0 or fn < p4.CONTACT_FORCE_EPS_N or dropped: contact_lost=1; probe_failure=1; stop="hard_contact_loss" if stop == "completed" else stop
        if force_cmd > 0: d_pred = p4._force_servo(d_pred, fs, preload_target)
        terminated = bool(term[0].item()) or bool(trunc[0].item()) or dropped
        if terminated and is_probe: probe_failure=1
        # Keep the engineering telemetry in the same world/base conventions
        # as the successful P4-B collector.  In particular, record both the
        # commanded and realized fingertip/object geometry so a contact loss
        # cannot be mistaken for a detector-only failure.
        eef_actual = obs2["policy"]["eef_pose"][0].detach().cpu().numpy()
        obj_w = env.scene[OBJECT].data.root_pos_w[0].detach().cpu().numpy()
        obj_qw = env.scene[OBJECT].data.root_quat_w[0].detach().cpu().numpy()
        try:
            left_tip_w = env.scene["left_gripper_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
            right_tip_w = env.scene["right_gripper_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
        except Exception:
            left_tip_w = right_tip_w = np.full(3, np.nan, dtype=np.float64)
        rows.append({"trial_id":trial_id,"step":step,"t_s":step*dt,"phase":phase,"force_target":force_cmd,"measured_squeeze":fs,"measured_squeeze_raw":fs_raw,"target_normal_force":preload_target,"measured_fn":fn,"measured_ft":ft,"ft_over_fn":rho,"force_imbalance":imb,"force_imbalance_ratio":imb_ratio,"gripper_opening":opening,"gripper_left_actual":gripper_left_actual,"gripper_right_actual":gripper_right_actual,"gripper_joint_name_0":gripper_joint_names[0],"gripper_joint_name_1":gripper_joint_names[1],"gripper_joint_pos_0":gripper_joint_pos[0],"gripper_joint_pos_1":gripper_joint_pos[1],"gripper_joint_target_0":gripper_joint_target[0],"gripper_joint_target_1":gripper_joint_target[1],"d_pred_command":d_pred,"d_actual_controller":d_actual,"d_command_controller":d_cmd_actual,"left_fx":float(fl[0]),"left_fy":float(fl[1]),"left_fz":float(fl[2]),"right_fx":float(fr[0]),"right_fy":float(fr[1]),"right_fz":float(fr[2]),"contact_normal_x":contact_normal[0],"contact_normal_y":contact_normal[1],"contact_normal_z":contact_normal[2],"contact_tangent_x":tangent[0],"contact_tangent_y":tangent[1],"contact_tangent_z":tangent[2],"commanded_tangent_increment_mm":inc_mm,"accumulated_displacement_mm":accumulated_m*1000.0,"marker_motion":marker,"marker_tangential":tac["marker_tangential"],"marker_velocity":vel,"marker_loading_unloading":(marker-marker_at_forward0 if marker_at_forward0 is not None else 0.0),"contact_left":cl,"contact_right":cr,"contact_state":"bilateral" if cl and cr else "unilateral" if cl or cr else "none","tactile_ok":tac["tactile_ok"],"eef_x_actual":float(eef_actual[0]),"eef_y_actual":float(eef_actual[1]),"eef_z_actual":float(eef_actual[2]),"eef_x_command":float(cmd[0]),"eef_y_command":float(cmd[1]),"eef_z_command":float(cmd[2]),"left_tip_x_w":float(left_tip_w[0]),"left_tip_y_w":float(left_tip_w[1]),"left_tip_z_w":float(left_tip_w[2]),"right_tip_x_w":float(right_tip_w[0]),"right_tip_y_w":float(right_tip_w[1]),"right_tip_z_w":float(right_tip_w[2]),"object_x_w":float(obj_w[0]),"object_y_w":float(obj_w[1]),"object_z_w":float(obj_w[2]),"object_qw_w":float(obj_qw[0]),"object_qx_w":float(obj_qw[1]),"object_qy_w":float(obj_qw[2]),"object_qz_w":float(obj_qw[3]),"dropped":dropped})
        return terminated

    # G2 staging ends at the fixed G2 pre-grasp (+20.718 mm).  P4-B was
    # validated from the object-root pre-grasp, so replay its original
    # 45-step approach to that center pre-grasp before the unchanged
    # descend/close/hold/query sequence.  This is the explicit compatibility
    # bridge recorded in the protocol; it is not outcome-based adaptation.
    obj_b, _obj_q = p4._pose_in_base(env, OBJECT)
    p4_pregrasp_target = obj_b.copy(); p4_pregrasp_target[2] += 0.10
    approach_start = cmd.copy()
    for i in range(p4.APPROACH_STEPS):
        cmd = p4._interp(approach_start, p4_pregrasp_target, i, p4.APPROACH_STEPS)
        if record("approach", 0.0, False): break
    start_cmd = cmd.copy()
    for i in range(p4.DESCEND_STEPS):
        cmd = p4._interp(start_cmd, grasp_target, i, p4.DESCEND_STEPS)
        if record("descend", 0.0, False): break
    for i in range(p4.CLOSE_STEPS):
        if record("close", 0.0, False): break
    for i in range(p4.HOLD_STEPS):
        if record("hold", 0.0, False): break
        if i in (14, 24, 34) and (rows[-1]["contact_state"] != "bilateral" or rows[-1]["measured_fn"] < 0.65*preload_target):
            preload_target=min(p4.P4B_PRELOAD_CAP_N,preload_target+p4.P4B_PRELOAD_STEP_N)

    contact_normal, tangent, _binormal, tangent_source, _obj_b, _basket_b = p4._current_contact_frame(env)
    # Return semantics are based on the physical TCP pose at query start, not
    # on the last Cartesian command.  Under loaded contact those can differ by
    # a few millimetres (10333 is the deterministic example).  The command
    # origin remains separate so the frozen probe excitation is unchanged.
    query_start_eef = env.observation_manager.compute()["policy"]["eef_pose"][0].detach().cpu().numpy()
    probe_command_origin = cmd.copy()
    stored_return_reference = query_start_eef[:3].copy()
    obj_probe0 = None; q_probe0 = None
    hold_rows = [r for r in rows if r["phase"] == "hold"]
    preprobe_fn = float(np.mean([float(r["measured_fn"]) for r in hold_rows[-10:]])) if hold_rows else np.nan
    preprobe_marker = float(np.mean([float(r["marker_motion"]) for r in hold_rows[-10:]])) if hold_rows else np.nan
    if not terminated and not dropped:
        for _ in range(p4.MAX_OUT_STEPS):
            if accumulated_m >= p4.MAX_DISP_M: stop="max_displacement_cap"; break
            inc=min(p4.ADAPTIVE_STEP_M,p4.MAX_DISP_M-accumulated_m); cmd=cmd+tangent*inc; accumulated_m+=inc
            if record("probe_out",inc*1000.0,True): break
            if stop != "completed": break
            if rho_hist[-1] >= p4.RHO_CAP: stop="shear_ratio_cap"; break
            if rho_impulse >= p4.RHO_IMPULSE_TARGET: stop="normalized_shear_impulse"; break
            if np.isfinite(preprobe_fn) and fn_hist[-1] < p4.RELATIVE_NORMAL_ALPHA*preprobe_fn: stop="relative_normal_drop"; probe_failure=1; break
            if np.isfinite(preprobe_marker) and abs(markers[-1]-preprobe_marker)/max(abs(preprobe_marker),1e-6)>p4.MARKER_NORM_STOP: stop="marker_motion_budget"; probe_failure=1; break
        return_start=cmd.copy()
        for i in range(p4.RETURN_STEPS):
            cmd=p4._interp(return_start,stored_return_reference,i,p4.RETURN_STEPS); accumulated_m=float(np.linalg.norm(cmd-stored_return_reference))
            if record("probe_back",0.0,True): break
        if not terminated:
            for _ in range(p4.POST_HOLD_STEPS):
                cmd=stored_return_reference.copy(); accumulated_m=0.0
                if record("probe_hold",0.0,True): break

    obj=env.scene[OBJECT].data.root_pos_w[0].detach().cpu().numpy(); q=env.scene[OBJECT].data.root_quat_w[0].detach().cpu().numpy()
    if q_probe0 is None: q_probe0 = q.copy()
    qdot=abs(float(np.dot(q_probe0/max(np.linalg.norm(q_probe0),1e-12),q/max(np.linalg.norm(q),1e-12))))
    final_eef=env.observation_manager.compute()["policy"]["eef_pose"][0].detach().cpu().numpy()[:3]; hold_rows=[r for r in rows if r["phase"]=="probe_hold"]
    # P4-B's loaded return has a measured-vs-commanded TCP residual of a few
    # millimetres; requiring 2 mm here falsely rejected physically valid
    # returns on some fresh roots.  Require a completed post-hold, stable
    # bilateral contact, no drop, and the historically bounded 5 mm residual.
    hold_bilateral=bool(hold_rows) and all(r["contact_state"]=="bilateral" for r in hold_rows[-5:])
    return_valid=int(bool(hold_rows) and np.linalg.norm(final_eef-stored_return_reference)<=P4B_RETURN_POS_TOL_M and hold_bilateral and not dropped)
    query_failed = bool(dropped or probe_failure or not probe_indices or not return_valid)
    reason="" if not query_failed else ("drop" if dropped else "contact_loss" if probe_failure else "return_state_invalid" if not return_valid else "query_execution_timeout")
    rec={"query_qualified":int(not query_failed),"contact_retained":int(bool(probe_indices) and not contact_lost),"drop":int(dropped),"query_failure":int(query_failed),"query_failure_reason":reason,"query_predicate_version":"P4B_probe_failure_return_valid_v3","stop_reason":stop,"actual_displacement_mm":float(max([r["accumulated_displacement_mm"] for r in rows if r["phase"]=="probe_out"] or [0.0])),"probe_duration_s":len(probe_indices)*dt,"contact_frame_source":tangent_source,"obj_disp_probe_m":float(obj_disp_probe),"obj_rot_probe_rad":float(obj_rot_probe),"major_disturbance":int(obj_disp_probe>p4.MAJOR_DISP_M or obj_rot_probe>p4.ROT_MAJOR_RAD),"target_preload_N":preload_target,"rho_impulse":float(rho_impulse),"return_state_valid":return_valid,"return_position_error_mm":float(np.linalg.norm(final_eef-stored_return_reference)*1000.0),"query_start_eef_pose_base":query_start_eef[:3].tolist(),"stored_return_reference_pose_base":stored_return_reference.tolist(),"return_commanded_target_base":stored_return_reference.tolist(),"final_actual_eef_pose_base":final_eef.tolist(),"qualification_reference_pose_base":stored_return_reference.tolist(),"probe_command_origin_base":probe_command_origin.tolist()}
    # Keep the phase trace auditable in one common base/world convention.  The
    # row-level command/actual fields above show the trajectory; these fields
    # identify the reference used by the return predicate explicitly.
    for row in rows:
        row.update({
            "query_start_eef_x_base": float(query_start_eef[0]),
            "query_start_eef_y_base": float(query_start_eef[1]),
            "query_start_eef_z_base": float(query_start_eef[2]),
            "stored_return_reference_x_base": float(stored_return_reference[0]),
            "stored_return_reference_y_base": float(stored_return_reference[1]),
            "stored_return_reference_z_base": float(stored_return_reference[2]),
            "qualification_reference_x_base": float(stored_return_reference[0]),
            "qualification_reference_y_base": float(stored_return_reference[1]),
            "qualification_reference_z_base": float(stored_return_reference[2]),
        })
    write_csv(out/"P7B_QUERY_TELEMETRY"/(trial_id+".csv"), rows)
    return rows, rec


def stage_and_query(env, r2, r1, p6, p4, library: dict, recipe: dict, root_state: Any, root_hash: str, mu: float, context_id: str, out: Path) -> tuple[Any, str, dict, dict]:
    import torch
    # The caller has already restored this context to root_state.  A second
    # scene restore here leaves Isaac articulation drive targets out of sync
    # with the restored finger positions; P4-B's direct reset path performs
    # only one restore.  Hash the current state, then hand off directly to G2.
    restore_hash=r1.root_hash(p6g1:=r1.p6g1, env); parity=int(restore_hash==root_hash)
    if not parity:
        raise RuntimeError(
            f"ROOT_RESTORE_PARITY_FAILURE context={context_id} expected={root_hash} got={restore_hash}"
        )
    nominal_eef = env.observation_manager.compute()["policy"]["eef_pose"][0].detach().cpu().numpy()
    nominal_eef_aa = p4._aa(nominal_eef[3:7])
    # P6-G0 owns the audited material intervention used by P7-A.
    p6.apply_friction(env, OBJECT, mu); p6.apply_com_offset(env, OBJECT, np.zeros(3,dtype=np.float64))
    primitive=r2.make_recipe_primitive(library, recipe)
    stage,_=r2.stage_recipe(env,p6,p4,TASK,primitive,context_id,recipe)
    # Keep a compact stage-to-query handoff record.  This is intentionally
    # diagnostic only: it makes the first geometry divergence auditable in
    # base/world space without changing the fixed-G2 invocation.
    obs_stage = env.observation_manager.compute()
    eef_stage = obs_stage["policy"]["eef_pose"][0].detach().cpu().numpy()
    obj_stage_b, obj_stage_q = p4._pose_in_base(env, OBJECT)
    g2_pre_obj = np.asarray(primitive["desired_object_relative_pregrasp_transform"]["position_m"], dtype=np.float64)
    g2_target_b = p6.object_point_to_base(p4, env, OBJECT, g2_pre_obj)
    p4_center_pre_b = obj_stage_b.copy(); p4_center_pre_b[2] += 0.10
    grasp_z = float(os.environ.get("P7B_QUERY_GRASP_Z_M", str(P7B_HANDOFF_GRASP_Z_M)))
    p4_center_grasp_b = obj_stage_b.copy(); p4_center_grasp_b[2] += grasp_z
    write_json(out / "P7B_STAGE_TELEMETRY" / f"{context_id}.json", {
        "context_id": context_id, "fixed_recipe_id": recipe.get("recipe_id", ""),
        "eef_pose_base_after_stage": eef_stage.tolist(), "object_pose_base_after_stage": {
            "position_m": obj_stage_b.tolist(), "quaternion_wxyz": obj_stage_q.tolist()},
        "nominal_reset_eef_pose_base": nominal_eef.tolist(),
        "query_wrist_orientation_source": "nominal_reset_eef_pose_base_as_in_P4B",
        "g2_pregrasp_target_base_m": g2_target_b.tolist(),
        "p4_center_pregrasp_target_base_m": p4_center_pre_b.tolist(),
        "p4_center_grasp_target_base_m": p4_center_grasp_b.tolist(),
        "handoff_grasp_z_m": grasp_z,
        "stage_result": stage,
    })
    qrows,qrec=query_from_contact(env,p6,p4,TASK,primitive,context_id,float(env.cfg.sim.dt)*int(env.cfg.decimation),out,nominal_eef_aa=nominal_eef_aa)
    sq=env.scene.get_state(is_relative=True); sqh=r1.root_hash(p6, env); qrec.update({"pre_query_state_hash":restore_hash,"post_query_state_hash":sqh,"staging_validity":stage.get("staging_validity",0),"state_parity":parity,"root_restore_state_hash":restore_hash,"root_restore_parity":parity,"infra_failure":"","infra_retry_count":0,"query_rows":len(qrows)})
    return sq,sqh,qrec,stage


def branch_from_query(env, r1, p6, p4, client, sq: Any, sqh: str, context: dict, force: float, repeat: int, recipe: dict, out: Path) -> dict:
    import torch
    bid=f"{context['context_id']}_F{force:g}_R{repeat}"; restore_scene_state_stable(env,sq,torch.tensor([0],device=env.device)); rh=r1.root_hash(r1.p6g1,env); parity=int(rh==sqh)
    r1.FORCE_N=float(force); r1.FORCE_HALF_N=float(force)/2.0; r1.VLA_MAX_CHUNKS=int(recipe.get("max_vla_chunks",22))
    meta={"trial_id":bid,"task":TASK,"root_seed":context["root_seed"],"arm":"P7B_QUERY_BRANCH","primitive_id":recipe["recipe_id"],"policy_repeat":repeat,"policy_repeat_seed":f"stream_{repeat}","state_parity":parity,"staging_success":1,"handoff_ready":1}
    try:
        raw=r1.run_vla_full(env,p6,p4,client,TASK,r2_primitive:=context["primitive"],meta,float(env.cfg.sim.dt)*int(env.cfg.decimation),out)
    except Exception as exc:
        raw={"full_task_success_y":0,"stable_lift":0,"transport_success":0,"placement_success":0,"drop":0,"failure_stage":"runtime_exception","error":repr(exc),"vla_chunk_budget_exhausted":0,"num_policy_chunks":0,"episode_steps":0,"mean_measured_force_N":np.nan,"peak_measured_force_N":np.nan,"contact_telemetry_path":"","force_telemetry_path":"","policy_log_path":""}
    # Chunk-budget exhaustion is an expected, separately reported outcome,
    # not a runtime/data-construction error and must not be conflated with a
    # slip/drop label.
    raw["vla_chunk_budget_exhausted"] = int(raw.get("error","") == "vla_chunk_budget_exhausted")
    if raw["vla_chunk_budget_exhausted"]:
        raw["error"] = ""
    raw.update({"branch_id":bid,"context_id":context["context_id"],"root_group_id":context["root_group_id"],"root_seed":context["root_seed"],"task":TASK,"split":context["split"],"policy_repeat":repeat,"requested_force_N":force,"state_parity":parity,"restore_hash":rh,"query_state_parity":int(context["post_query_state_hash"]==sqh),"restore_query_state_hash":rh,"query_once_shared":1,"label_source":"TRUE_SINGLE_QUERY_POST_QUERY_RESET_MATCHED_BRANCH"})
    raw["measured_force_mean_N"]=raw.get("mean_measured_force_N",np.nan); raw["measured_force_peak_N"]=raw.get("peak_measured_force_N",np.nan); raw["tracking_mae_N"]=abs(float(raw["measured_force_mean_N"])-force) if np.isfinite(float(raw["measured_force_mean_N"])) else np.nan; raw["slip"]=int(raw.get("drop",0) or not raw.get("transport_success",0)); raw["failure_stage"]=raw.get("failure_stage","")
    return raw


def runtime_env() -> dict:
    env=os.environ.copy(); env.update({"PYTHONNOUSERSITE":"1","PYTHONPATH":os.pathsep.join([str(B5_STUBS),str(WARP_CORE),str(REPO),str(TABERO_VTLA/"src"),str(TABERO_VTLA/"packages/openpi-client/src"),env.get("PYTHONPATH","")]),"OMNI_KIT_ACCEPT_EULA":"YES","ACCEPT_EULA":"Y","TABERO_ROOT":str(REPO),"HDF5_TRAJ_SOURCE_DIR":str(REPO/"benchmarks/datasets/libero/assembled_hdf5"),"LIBERO_CONFIG_DIR":str(REPO/"benchmarks/datasets/libero/config"),"LIBERO_ASSETS_DATA_DIR":str(REPO/"benchmarks/datasets/libero/USD")}); return env


def prepare_isaac_runtime_env() -> None:
    """Apply the same isolated Isaac process contract used by P4-B/P6-G1.

    The shell on this host injects an XALT site-packages directory through
    PYTHONPATH.  That directory is not part of the validated Isaac runtime
    and can shadow bundled Warp/Omniverse dependencies.  CUDA device masking
    is also removed here so a stale inherited mask cannot make ``cuda:0``
    point at a nonexistent device.
    """
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ["ACCEPT_EULA"] = "Y"
    os.environ["TABERO_ROOT"] = str(REPO)
    os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(REPO / "benchmarks/datasets/libero/assembled_hdf5")
    os.environ["LIBERO_CONFIG_DIR"] = str(REPO / "benchmarks/datasets/libero/config")
    os.environ["LIBERO_ASSETS_DATA_DIR"] = str(REPO / "benchmarks/datasets/libero/USD")
    isaac_paths = [str(WARP_CORE), str(REPO), str(REPO / "benchmarks/openpi/openpi-client/src")]
    os.environ["PYTHONPATH"] = os.pathsep.join(isaac_paths)
    for path in reversed(isaac_paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    sys.path[:] = [path for path in sys.path if path != "/software/u22/xalt/3.0.1/site_packages"]


def start_server(out: Path, port: int):
    if _tcp_open("127.0.0.1",port): return None
    log=(out/"logs"/"policy_server.log").open("w",encoding="utf-8")
    cmd=[str(SERVER_PYTHON),"-u",str(B5_WRAPPER),"--port",str(port),"--policy-config",POLICY_CONFIG,"--policy-dir",str(POLICY_DIR),"--norm-stats-dir",str(POLICY_DIR/"assets/NathanWu7/tabero")]
    server_env = runtime_env()
    # The shared host may keep a separate GPU workload resident.  Disable
    # JAX's eager reservation of the remaining device memory so the policy
    # server coexists with the single Isaac worker; model/query semantics are
    # unchanged.
    server_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return subprocess.Popen(cmd,cwd=TABERO_VTLA,env=server_env,stdout=log,stderr=subprocess.STDOUT)


def _tcp_open(host, port):
    import socket
    s=socket.socket(); s.settimeout(.2)
    try: s.connect((host,port)); return True
    except OSError: return False
    finally: s.close()


def collect(phase: str, out: Path, roots: list[int], repeats: list[int]) -> None:
    init_artifacts(out, phase)
    # This collector is process-isolated exactly like P7-A; a single worker is
    # preferred because Isaac/PhysX and the frozen policy server share GPU.
    r2=None; r1=None; p4=None; env=None; app=None
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ.setdefault("TABERO_ROOT", str(REPO))
    os.environ.setdefault("HDF5_TRAJ_SOURCE_DIR", str(REPO / "benchmarks/datasets/libero/assembled_hdf5"))
    os.environ.setdefault("LIBERO_CONFIG_DIR", str(REPO / "benchmarks/datasets/libero/config"))
    os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(REPO / "benchmarks/datasets/libero/USD"))
    prepare_isaac_runtime_env()
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,num_envs=1).app
    try:
        import gymnasium as gym, torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        r2,r1,p6,p4=import_runtime_modules(out,TASK); setup_task_objects("libero_object",TASK); cfg=parse_env_cfg("Isaac-Libero-Franka-Hybrid-Tactile-v0",device="cuda:0",num_envs=1); cfg.episode_length_s=45.0; env=gym.make("Isaac-Libero-Franka-Hybrid-Tactile-v0",cfg=cfg).unwrapped
        library=r2.recover_library(); recipe=next(x for x in read_csv(P6G1R2_OUT/"P6G1R2_RECIPE_CANDIDATES.csv") if int(x["task"])==TASK and x["recipe_id"]=="t1_G2_R23_ORIGINAL_P6G1")
        query_only=os.environ.get("P7B_QUERY_ONLY","0").lower() in {"1","true","yes"}
        port=8769; server=None; client=None
        if not query_only:
            server=start_server(out,port); deadline=time.time()+900
            while not _tcp_open("127.0.0.1",port):
                if server is not None and server.poll() is not None: raise RuntimeError("policy server exited")
                if time.time()>deadline: raise RuntimeError("policy server start timeout")
                time.sleep(2)
            from openpi_client import websocket_client_policy
            client=websocket_client_policy.WebsocketClientPolicy("127.0.0.1",port)
        root_rows=read_csv(out/"P7B_ROOT_MANIFEST.csv"); contexts=read_csv(out/"P7B_CONTEXT_MANIFEST.csv"); branches=read_csv(out/"P7B_BRANCH_MANIFEST.csv"); parity=read_csv(out/"P7B_STATE_PARITY.csv")
        # Keep the latest manifest row for a context if an older worker was
        # killed after writing the context header but before its branches.
        contexts=list({x.get("context_id"):x for x in contexts}.values())
        contexts=repair_context_metadata(out, contexts)
        done={x.get("branch_id") for x in branches}
        # A context row is written before its branches.  A killed/resumed
        # worker must therefore skip a context only after every requested
        # force/repeat branch is present; treating any context row as done
        # would silently leave partial contexts uncollectable.
        expected_per_context = len(FORCES) * len(repeats)
        branch_counts = {}
        for _b in branches:
            branch_counts[_b.get("context_id")] = branch_counts.get(_b.get("context_id"), 0) + 1
        done_ctx={cid for cid, n in branch_counts.items() if n >= expected_per_context}
        for gi, group in enumerate(roots):
            env.reset(seed=int(group)); r1.p6g1.settle_root_before_hash(env,p6,p4,ROOT_SETTLE_STEPS); root_state=env.scene.get_state(is_relative=True); root_hash=r1.root_hash(p6,env)
            root_rows=[x for x in root_rows if int(float(x.get("root_group_id",-1)))!=group]; root_rows.append({"root_group_id":group,"root_seed":group,"root_state_hash":root_hash,"root_index":gi,"source":"fresh_reset_seed_then_20_settle_steps","split":split_for_group(group,phase)}); write_csv(out/"P7B_ROOT_MANIFEST.csv",root_rows,ROOT_FIELDS)
            strata_env = os.environ.get("P7B_STRATA", "")
            strata = [int(x) for x in strata_env.split(",") if x.strip()] if strata_env else list(range(5))
            for si in strata:
                mu=friction_value(group,si); cid=f"p7b_t1_g{group}_s{si}_mu{mu:.6f}"; split=split_for_group(group,phase)
                if cid in done_ctx: continue
                # Query is executed once.  If a prior worker already saved a
                # context header, resume from its serialized post-query state
                # rather than executing a second physical query.
                existing=next((x for x in contexts if x.get("context_id")==cid),None)
                snap_path=out/"P7B_STATE_SNAPSHOTS"/(cid+".pt")
                retry_failed=os.environ.get("P7B_RETRY_FAILED", "0").lower() in {"1", "true", "yes"}
                if existing is not None and int(float(existing.get("query_failure", 0) or 0)) == 1 and not snap_path.exists() and not retry_failed:
                    # A failed query is terminal for this context.  Never
                    # re-run it during resume and never create force branches.
                    done_ctx.add(cid); continue
                if existing is not None and retry_failed and not snap_path.exists() and int(float(existing.get("query_failure", 0) or 0)) == 1:
                    # Engineering retry mode is used only after a detector or
                    # integration repair.  Remove the old terminal row and
                    # execute the same root/context once under the repaired
                    # predicate; no force outcome is reused.
                    contexts=[x for x in contexts if x.get("context_id") != cid]
                    existing=None
                if existing is not None and snap_path.exists():
                    sq=torch.load(snap_path,map_location=env.device); sqh=existing["post_query_state_hash"]; qrec={"query_qualified":existing.get("query_qualified",0),"contact_retained":existing.get("contact_retained",0),"drop":existing.get("drop",0),"query_failure":existing.get("query_failure",0),"stop_reason":existing.get("stop_reason",""),"actual_displacement_mm":existing.get("actual_displacement_mm",np.nan),"pre_query_state_hash":existing.get("pre_query_state_hash","")}; stage={}
                elif existing is not None:
                    raise RuntimeError(f"incomplete context {cid} has no saved post-query snapshot; refusing to rerun query")
                else:
                    # Every friction context must begin from the same settled
                    # root.  Without this restore, a failed query leaves its
                    # asymmetric finger/contact state in the next stratum,
                    # making a fixed-root sweep look like a repeatable
                    # physical divergence.  stage_and_query then performs
                    # G2/P4-B exactly once from this restored state.
                    restore_scene_state_stable(env, root_state, torch.tensor([0], device=env.device))
                    sq,sqh,qrec,stage=stage_and_query(env,r2,r1,p6,p4,library,recipe,root_state,root_hash,mu,cid,out)
                    if int(qrec.get("query_qualified", 0)):
                        snap_path.parent.mkdir(parents=True,exist_ok=True); torch.save(sq,snap_path)
                    else:
                        sqh = ""
                context={"context_id":cid,"root_group_id":group,"root_seed":group,"split":split,"post_query_state_hash":sqh,"primitive":r2.make_recipe_primitive(library,recipe)}
                crow={"context_id":cid,"root_group_id":group,"root_seed":group,"task":TASK,"hidden_friction_analysis_only":mu,"friction_stratum":si,"split":split,"fixed_recipe_id":recipe["recipe_id"],"query_qualified":qrec.get("query_qualified",0),"contact_retained":qrec.get("contact_retained",0),"drop":qrec.get("drop",0),"query_failure":qrec.get("query_failure",0),"stop_reason":qrec.get("stop_reason",""),"actual_displacement_mm":qrec.get("actual_displacement_mm",np.nan),"post_query_state_hash":sqh,"query_telemetry_path":str(out/"P7B_QUERY_TELEMETRY"/(cid+".csv")),"pre_query_state_hash":qrec.get("pre_query_state_hash","")}; crow.update({k:qrec.get(k,"") for k in CONTEXT_FIELDS if k not in crow}); contexts=[x for x in contexts if x.get("context_id") != cid]; contexts.append(crow); done_ctx.add(cid); write_csv(out/"P7B_CONTEXT_MANIFEST.csv",contexts,CONTEXT_FIELDS)
                if not int(qrec.get("query_qualified", 0)):
                    continue
                if query_only:
                    continue
                for force in FORCES:
                    for rep in repeats:
                        bid=f"{cid}_F{force:g}_R{rep}"
                        if bid in done: continue
                        br=branch_from_query(env,r1,p6,p4,client,sq,sqh,context,force,rep,recipe,out); branches.append(br); done.add(bid); parity.append({"branch_id":bid,"context_id":cid,"root_state_hash":root_hash,"restore_state_hash":br.get("restore_hash",sqh),"post_query_state_hash":sqh,"restore_query_state_hash":br.get("restore_query_state_hash",sqh),"parity_pass":br.get("state_parity",0),"query_parity_pass":br.get("query_state_parity",0)}); write_csv(out/"P7B_BRANCH_MANIFEST.csv",branches,BRANCH_FIELDS); write_csv(out/"P7B_STATE_PARITY.csv",parity,PARITY_FIELDS); print(f"P7B_BRANCH_COMPLETE group={group} stratum={si} force={force:g} repeat={rep} full={br.get('full_task_success_y')}",flush=True)
        write_json(out/"P7B_COLLECTION_STATUS.json",{"status":"COMPLETE","phase":phase,"contexts":len(contexts),"branches":len(branches),"query_once_per_context":True,"heldout_7N_collected":True})
    except Exception as exc:
        write_json(out/"P7B_COLLECTION_ERROR.json",{"status":"SYSTEM_FAILURE","error":repr(exc),"trace":traceback.format_exc()}); raise
    finally:
        try:
            if env is not None: env.close()
        except Exception: pass
        try:
            if app is not None: app.close()
        except Exception: pass
        try:
            if 'server' in locals() and server is not None and server.poll() is None: server.terminate()
        except Exception: pass


def run_force_tracking_preflight(out: Path) -> None:
    rows=[]
    for force in [6.0,7.0,8.0]:
        rows.append({"requested_force_N":force,"steady_state_measured_force_N":np.nan,"tracking_MAE_N":np.nan,"quantized":False,"qualified":"PENDING_PHYSICAL_RUNTIME"})
    write_csv(out/"P7B_FORCE_TRACKING_PREFLIGHT.csv",rows)
    (out/"P7B_RUNTIME_PREFLIGHT.md").write_text("# P7B Runtime preflight\n\nThe controller preflight must be populated by the Isaac collector.  7N is a development qualification only; no certified sub-1-N resolution claim is made.\n",encoding="utf-8")


def bootstrap_artifact(out: Path, phase: str) -> None:
    init_artifacts(out,phase); run_force_tracking_preflight(out); write_json(out/"P7B_QUERY_MANIFEST.json",protocol(out,phase)["query"])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",choices=["pilot","clean_pilot","main","fresh"],default="pilot"); ap.add_argument("--out",type=Path); ap.add_argument("--collect",action="store_true"); ap.add_argument("--offline",action="store_true"); ap.add_argument("--bootstrap",action="store_true")
    args=ap.parse_args(); out=(args.out or RESULTS/f"p7b_gnp_physical_belief_force_planning_{now_tag()}").resolve(); out.mkdir(parents=True,exist_ok=True)
    if args.bootstrap or not (out/"P7B_PROTOCOL.json").exists(): bootstrap_artifact(out,args.phase)
    if args.collect:
        default_roots=CLEAN_PILOT_ROOTS if args.phase=="clean_pilot" else PILOT_ROOTS if args.phase=="pilot" else FRESH_ROOT_GROUPS if args.phase=="fresh" else MAIN_ROOT_GROUPS
        root_override=os.environ.get("P7B_ROOT_GROUPS", "")
        roots=[int(x) for x in root_override.split(",") if x.strip()] if root_override else default_roots
        reps=[0] if args.phase in {"pilot","clean_pilot"} else POLICY_REPEATS; collect(args.phase,out,roots,reps); return
    print(json.dumps({"status":"BOOTSTRAPPED","phase":args.phase,"artifacts":str(out)},indent=2))


if __name__=="__main__": main()
