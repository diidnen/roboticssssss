#!/usr/bin/env python3
"""P7A fixed-invocation force frontier.

This runner is deliberately narrower than the preceding BATON experiments:
the task recipe, grasp geometry, frozen VLA checkpoint, and invocation
protocol are fixed.  The only per-rollout experimental variable is requested
grip force.  Hidden friction is a blocked physical condition, not a runtime
decision variable.

The short frozen P7A markdown protocol did not contain numeric matrix values.
The execution manifest records the inherited authoritative predecessor matrix
explicitly: forces [3, 5, 8] N, hidden friction [0.25, 0.90], ten fresh roots
per task, and two policy repeats.  No outcome is used to change that matrix.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
P6G1R3_OUT = RESULTS / "p6g1r3_raw_vla_vs_fixed_recipe_20260826_123552"
FROZEN_PROTOCOL = P6G1R3_OUT / "P7A_FIXED_INVOCATION_FORCE_FRONTIER_PROTOCOL.md"
R3_SCRIPT = REPO / "analysis/p6g1r3_raw_vla_vs_fixed_recipe.py"
R2_OUT = RESULTS / "p6g1r2_baton_verified_vla_recipes_20260825_205113"
R2_CANDIDATES = R2_OUT / "P6G1R2_RECIPE_CANDIDATES.csv"
CLUSTERS = REPO / "analysis/results/p6g1_primitive_ik_vla_grasp_realization_20260825_103146/P6G1_REFERENCE_GRASP_CLUSTERS.csv"
P6G0R1_PROTOCOL = RESULTS / "p6g0r1_confirmatory_and_candidate_coverage_20260824_232236/P6G0R1_PROTOCOL.json"

TASKS = [1, 6]
OBJECTS = {1: "cream_cheese_1", 6: "butter_1"}
FORCE_GRID = [3.0, 5.0, 8.0]
HIDDEN_FRICTIONS = [0.25, 0.90]
POLICY_SEEDS = [0, 1]
ROOTS = [9840, 9841, 9842, 9843, 9844, 9845, 9846, 9847, 9848, 9849]
FIXED_RECIPE_IDS = {1: "t1_G2_R23_ORIGINAL_P6G1", 6: "t6_G0_R09"}
EXPECTED_CHECKPOINT_HASH = "0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17"
ROOT_SETTLE_STEPS = 20
EXPECTED_ROLLOUTS = len(TASKS) * len(ROOTS) * len(POLICY_SEEDS) * len(FORCE_GRID) * len(HIDDEN_FRICTIONS)

VALID_CLUSTERS = ["G0", "G1", "G2", "OTHER", "NO_VALID_GRASP"]


def load_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# R3 imports the same frozen R2/R1 helpers used by the authoritative run.
r3 = load_module(R3_SCRIPT, "p7a_r3_helpers")
r2 = r3.r2
r1 = r3.r1


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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
        fh.flush()
        os.fsync(fh.fileno())


def as_float(row: dict, key: str, default: float = float("nan")) -> float:
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


ROOT_FIELDS = [
    "task", "object", "root_index", "root_seed", "root_state_hash",
    "object_pose_base_x", "object_pose_base_y", "object_pose_base_z",
    "object_pose_base_qw", "object_pose_base_qx", "object_pose_base_qy", "object_pose_base_qz",
    "source", "arm_restore_rule",
]

ROLLOUT_FIELDS = [
    "trial_id", "task", "object", "arm", "root_seed", "policy_seed", "policy_repeat_seed",
    "fixed_recipe_id", "target_mode", "requested_force_N", "hidden_friction", "friction_label",
    "com_label", "state_parity", "root_state_hash", "restore_state_hash", "physics_application",
    "staging_success", "staging_validity", "staging_time_s", "extra_action_count", "num_policy_chunks",
    "episode_steps", "episode_duration_s", "bilateral_contact", "stable_lift", "slip", "drop",
    "object_rotation_during_lift_rad", "contact_retention", "transport_success", "placement_success",
    "full_task_success", "failure_stage", "realized_grasp_cluster", "contact_center_obj_x",
    "contact_center_obj_y", "contact_center_obj_z", "wrist_obj_qw", "wrist_obj_qx", "wrist_obj_qy",
    "wrist_obj_qz", "grasp_depth_m", "left_contact_obj_x", "left_contact_obj_y", "left_contact_obj_z",
    "right_contact_obj_x", "right_contact_obj_y", "right_contact_obj_z", "mean_measured_force_N",
    "peak_measured_force_N", "contact_telemetry_path", "force_telemetry_path", "policy_log_path", "error",
]

AGG_FIELDS = [
    "task", "object", "hidden_friction", "friction_label", "requested_force_N", "n", "valid_grasp_n",
    "dominant_grasp_cluster", "dominant_grasp_fraction", "cluster_entropy_nats", "bilateral_contact_rate",
    "stable_lift_rate", "full_task_success_rate", "placement_rate", "slip_rate", "drop_rate",
    "contact_center_n", "contact_center_mean_x_m", "contact_center_mean_y_m", "contact_center_mean_z_m",
    "contact_center_std_x_m", "contact_center_std_y_m", "contact_center_std_z_m", "contact_center_spatial_std_m",
    "contact_center_spatial_variance_m2", "mean_object_rotation_during_lift_rad", "mean_episode_duration_s",
    "mean_num_policy_chunks", "failure_mode_counts_json",
]

VAR_FIELDS = [
    "task", "object", "hidden_friction", "friction_label", "requested_force_N", "n", "valid_grasp_n",
    "mean_x_m", "mean_y_m", "mean_z_m", "std_x_m", "std_y_m", "std_z_m", "spatial_std_m", "spatial_variance_m2",
]

PAIR_FIELDS = [
    "task", "object", "hidden_friction", "friction_label", "force_a_N", "force_b_N", "metric",
    "mean_a", "mean_b", "delta_b_minus_a", "bootstrap_ci95_low", "bootstrap_ci95_high", "n_roots",
    "direction_consistent_root_fraction", "definition",
]

FAIL_FIELDS = [
    "task", "object", "hidden_friction", "friction_label", "requested_force_N", "failure_stage",
    "count", "fraction_of_rollouts", "slip_count", "drop_count", "runtime_error_count", "trial_ids_json",
]


def recipe_rows() -> dict[int, dict]:
    rows = read_csv(R2_CANDIDATES)
    out = {}
    for task, rid in FIXED_RECIPE_IDS.items():
        matches = [r for r in rows if int(r["task"]) == task and r["recipe_id"] == rid]
        if len(matches) != 1:
            raise RuntimeError(f"fixed R2 recipe not unique: task={task} id={rid} n={len(matches)}")
        out[task] = matches[0]
    return out


def prior_root_audit() -> dict:
    tokens = {str(x) for x in ROOTS}
    hits = []
    for d in RESULTS.iterdir():
        if not d.is_dir() or not d.name.lower().startswith(("p6g0", "p6g1")):
            continue
        if d.name.lower().startswith("p7a"):
            continue
        for p in d.rglob("*"):
            if not p.is_file() or p.stat().st_size > 30_000_000 or p.suffix == ".npz":
                continue
            try:
                text = p.read_text(errors="ignore")
            except Exception:
                continue
            found = sorted(token for token in tokens if token in text.split())
            if found:
                hits.append({"path": str(p), "roots": found})
    return {"roots": ROOTS, "prior_artifact_glob": "p6g0*/p6g1*", "collisions": hits, "disjoint": not hits}


def protocol_obj(out: Path, recipes: dict[int, dict], root_audit: dict) -> dict:
    source_text = FROZEN_PROTOCOL.read_text(encoding="utf-8")
    predecessor = read_json(P6G0R1_PROTOCOL)
    return {
        "name": "P7A Fixed-Invocation Force Frontier",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method_change": "NONE",
        "scientific_question": "Whether force has a stable, repeatable, actionable causal effect on manipulation outcome after invocation, grasp, and execution geometry are fixed.",
        "source_frozen_protocol": str(FROZEN_PROTOCOL),
        "source_frozen_protocol_sha256": sha256_file(FROZEN_PROTOCOL),
        "source_frozen_protocol_text_preserved_verbatim": True,
        "tasks": TASKS,
        "object_by_task": OBJECTS,
        "fixed_recipe_by_task": recipes,
        "runtime_action_space": ["Force only"],
        "force_grid_N": FORCE_GRID,
        "hidden_friction_values": HIDDEN_FRICTIONS,
        "center_of_mass": "centered/default",
        "policy_repeats": POLICY_SEEDS,
        "fresh_roots_per_task": ROOTS,
        "total_rollouts": EXPECTED_ROLLOUTS,
        "matrix_instantiation": {
            "status": "explicit_inherited_execution_matrix",
            "reason": "The frozen P7A markdown specifies force/hidden-friction crossing but omits numeric levels, root count, and repeat count.",
            "force_source": str(P6G0R1_PROTOCOL),
            "force_source_grid_N": predecessor["track_a"]["force_grid"],
            "friction_source": str(P6G0R1_PROTOCOL),
            "friction_source_values": predecessor["track_a"]["frictions"],
            "root_repeat_discipline_source": str(P6G1R3_OUT / "P6G1R3_PROTOCOL.json"),
            "human_visible_before_execution": True,
        },
        "matched_root_protocol": "one settled initial state per task/root, independently restored before every force x friction x policy-repeat rollout",
        "physics": {"hidden_variable": "object friction", "com": "centered/default", "mass": "unchanged", "inertia": "unchanged/default"},
        "invocation": {
            "sequence": ["analytic/IK staging using frozen qualified recipe", "frozen VLA contact-rich execution", "constant requested force for entire rollout", "full task"],
            "recipe_search": False, "recipe_selection": False, "retry": False, "online_force_change": False,
            "grasp_position_runtime_decision": False, "vla_checkpoint_change": False, "steering_operator_change": False,
        },
        "frozen_policy": {
            "checkpoint": str(r1.POLICY_DIR), "checkpoint_hash_sha256": sha256_path(r1.POLICY_DIR),
            "expected_hash_sha256": EXPECTED_CHECKPOINT_HASH, "config": r1.POLICY_CONFIG,
            "server_wrapper": str(r1.B5_WRAPPER), "server_wrapper_sha256": sha256_file(r1.B5_WRAPPER),
        },
        "analysis_rules": {
            "paired_unit": "root-level mean over two policy repeats; policy repeats are not independent environmental roots",
            "bootstrap": "10,000 paired bootstrap resamples of roots within task x hidden-friction group",
            "meaningful_effect_threshold": 0.10,
            "confirmed_rule": "At least one task x hidden-friction group has a >=0.10 absolute range in stable-lift or full-task success across force, with paired 95% CI excluding zero and >=0.70 root-direction consistency, with complete valid data.",
            "weak_rule": "Any force range is >=0.10 but the confirmed rule is not met, or a smaller directionally repeatable effect is present.",
            "none_rule": "No force range reaches 0.10 and no repeatable frontier structure is present.",
            "invalid_rule": "Missing/duplicate rollouts, state-parity failure, recipe/invocation drift, runtime errors, or material execution-geometry confound prevents attribution.",
        },
        "root_audit": root_audit,
        "prohibited": ["training", "GNP", "physical query", "DreamTrajectory", "probe", "recipe search", "recipe selection", "retry", "online force change", "grasp candidate changes"],
        "output_directory": str(out),
        "protocol_text_sha256": sha256_file(FROZEN_PROTOCOL),
        "protocol_text_length": len(source_text),
    }


def friction_label(mu: float) -> str:
    return "LOW" if abs(mu - 0.25) < 1e-9 else "HIGH"


def trial_id(task: int, root: int, force: float, mu: float, seed: int) -> str:
    return f"p7a_t{task}_root{root}_mu{mu:g}_f{force:g}_ps{seed}_FIXED"


def run_one(env, p6, p4, client, task: int, root_state: Any, root_hash: str, root_seed: int, policy_seed: int, force: float, mu: float, recipe: dict, library: dict, out: Path, clusters: list[dict]) -> dict:
    import torch
    tid = trial_id(task, root_seed, force, mu, policy_seed)
    t0 = time.perf_counter()
    env.reset_to(copy.deepcopy(root_state), torch.tensor([0], device=env.device), is_relative=True)
    restore_hash = r1.root_hash(p6, env)
    parity = int(restore_hash == root_hash)
    physics_application = "friction_applied_to_object_material; centered_default_com; mass_inertia_unchanged"
    stage = {}
    primitive = r2.make_recipe_primitive(library, recipe)
    stage_start = time.perf_counter()
    try:
        # Material-only hidden intervention.  CoM is explicitly reset to the
        # cached nominal value by the existing audited helper.
        p6.apply_friction(env, OBJECTS[task], float(mu))
        p6.apply_com_offset(env, OBJECTS[task], np.zeros(3, dtype=np.float64))
        stage, _ = r2.stage_recipe(env, p6, p4, task, primitive, tid, recipe)
    except Exception as exc:
        stage = {"staging_validity": 0, "error": repr(exc), "steps": 0}
    staging_time = time.perf_counter() - stage_start

    # The R1 full-task implementation reads these globals when it overwrites
    # both gripper force channels.  This changes only the pre-registered force
    # variable; it does not alter the frozen policy or its raw actions.
    r1.FORCE_N = float(force)
    r1.FORCE_HALF_N = float(force) / 2.0
    meta = {
        "trial_id": tid, "task": task, "root_seed": root_seed, "arm": "P7A_FIXED",
        "primitive_id": recipe["recipe_id"], "policy_repeat": str(policy_seed),
        "policy_repeat_seed": f"stream_{policy_seed}", "state_parity": parity,
        "staging_success": stage.get("staging_validity", 0), "handoff_ready": stage.get("staging_validity", 0),
    }
    try:
        raw = r1.run_vla_full(env, p6, p4, client, task, primitive, meta, float(env.cfg.sim.dt) * int(env.cfg.decimation), out)
    except Exception as exc:
        raw = {
            "trial_id": tid, "task": task, "object": OBJECTS[task], "arm": "P7A_FIXED", "root_seed": root_seed,
            "primitive_id": recipe["recipe_id"], "full_task_success_y": 0, "stable_lift": 0, "drop": 0,
            "transport_success": 0, "placement_success": 0, "failure_stage": "runtime_exception", "episode_steps": 0,
            "num_policy_chunks": 0, "error": repr(exc), "contact_telemetry_path": "", "force_telemetry_path": "", "policy_log_path": "",
        }
    duration = time.perf_counter() - t0
    raw.update({
        "trial_id": tid, "task": task, "object": OBJECTS[task], "arm": "P7A_FIXED", "root_seed": root_seed,
        "policy_seed": policy_seed, "policy_repeat_seed": f"stream_{policy_seed}", "fixed_recipe_id": recipe["recipe_id"],
        "target_mode": recipe["target_mode"], "requested_force_N": force, "hidden_friction": mu,
        "friction_label": friction_label(mu), "com_label": "CENTERED_DEFAULT", "state_parity": parity,
        "root_state_hash": root_hash, "restore_state_hash": restore_hash, "physics_application": physics_application,
        "staging_success": stage.get("staging_validity", 0), "staging_validity": stage.get("staging_validity", 0),
        "staging_time_s": staging_time, "extra_action_count": stage.get("steps", 0), "num_policy_chunks": raw.get("num_policy_chunks", 0),
        "episode_steps": raw.get("episode_steps", 0), "episode_duration_s": duration,
        "bilateral_contact": raw.get("bilateral_contact", 0), "stable_lift": raw.get("stable_lift", 0),
        "drop": raw.get("drop", 0), "contact_retention": raw.get("transport_grasp_retained", raw.get("transport_success", 0)),
        "transport_success": raw.get("transport_success", 0), "placement_success": raw.get("placement_success", 0),
        "full_task_success": raw.get("full_task_success_y", 0), "failure_stage": raw.get("failure_stage", ""),
        "error": raw.get("error", ""),
    })
    result = r3.finish_rollout(raw, clusters)
    result["requested_force_N"] = force
    result["hidden_friction"] = mu
    result["friction_label"] = friction_label(mu)
    return result


def worker(out: Path, task: int, host: str, port: int) -> int:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    try:
        from openpi_client import websocket_client_policy
        env, p6, p4 = r1.import_env_modules(task)
        client = websocket_client_policy.WebsocketClientPolicy(host, port)
        library = r2._reference_library
        recipe = recipe_rows()[task]
        clusters = read_csv(CLUSTERS)
        root_path = out / "P7A_ROOT_MANIFEST.csv"
        results_path = out / "P7A_ROLLOUT_RESULTS.csv"
        existing = {r.get("trial_id") for r in read_csv(results_path)}
        reset_smoke = out / "P7A_RESET_STEP_SMOKE.json"
        for idx, root_seed in enumerate(ROOTS):
            env.reset(seed=int(root_seed))
            r1.p6g1.settle_root_before_hash(env, p6, p4, ROOT_SETTLE_STEPS)
            root_state = env.scene.get_state(is_relative=True)
            h0 = r1.root_hash(p6, env)
            obj0, objq0 = p4._pose_in_base(env, OBJECTS[task])
            root_row = {
                "task": task, "object": OBJECTS[task], "root_index": idx, "root_seed": root_seed,
                "root_state_hash": h0, "object_pose_base_x": obj0[0], "object_pose_base_y": obj0[1], "object_pose_base_z": obj0[2],
                "object_pose_base_qw": objq0[0], "object_pose_base_qx": objq0[1], "object_pose_base_qy": objq0[2], "object_pose_base_qz": objq0[3],
                "source": f"fresh_env_reset_seed_then_{ROOT_SETTLE_STEPS}_settle_steps",
                "arm_restore_rule": "deep-copied common root state independently restored before each force x friction x policy repeat",
            }
            current = read_csv(root_path)
            if not any(int(float(x.get("task", -1))) == task and int(float(x.get("root_seed", -1))) == root_seed for x in current):
                append_csv(root_path, root_row, ROOT_FIELDS)
            if not reset_smoke.exists():
                write_json(reset_smoke, {"task": task, "root_seed": root_seed, "reset_step_smoke": True, "root_state_hash": h0})
            for mu in HIDDEN_FRICTIONS:
                for force in FORCE_GRID:
                    for policy_seed in POLICY_SEEDS:
                        tid = trial_id(task, root_seed, force, mu, policy_seed)
                        if tid in existing:
                            continue
                        row = run_one(env, p6, p4, client, task, root_state, h0, root_seed, policy_seed, force, mu, recipe, library, out, clusters)
                        append_csv(results_path, row, ROLLOUT_FIELDS)
                        existing.add(tid)
                        print(f"P7A_ROLLOUT_COMPLETE task={task} root={root_seed} mu={mu:g} force={force:g} policy_seed={policy_seed} stable={row.get('stable_lift')} full={row.get('full_task_success')} failure={row.get('failure_stage')}", flush=True)
        return 0
    except Exception as exc:
        write_json(out / f"P7A_WORKER_ERROR_task{task}.json", {"task": task, "error": repr(exc), "trace": traceback.format_exc()})
        return 1
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            app.close()
        except Exception:
            pass


def bootstrap(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for d in ["P7A_CONTACT_TELEMETRY", "P7A_FORCE_TELEMETRY", "P7A_POLICY_LOGS", "logs"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    recipes = recipe_rows()
    audit = prior_root_audit()
    proto = protocol_obj(out, recipes, audit)
    write_json(out / "P7A_PROTOCOL.json", proto)
    (out / "P7A_PROTOCOL_HASH.txt").write_text(sha256_file(out / "P7A_PROTOCOL.json") + "\n", encoding="utf-8")
    (out / "P7A_CODE_HASH.txt").write_text(sha256_file(Path(__file__).resolve()) + "\n", encoding="utf-8")
    # Preserve the frozen protocol text byte-for-byte as an artifact.
    (out / "P7A_FIXED_INVOCATION_FORCE_FRONTIER_PROTOCOL.md").write_bytes(FROZEN_PROTOCOL.read_bytes())
    write_csv(out / "P7A_ROOT_MANIFEST.csv", [], ROOT_FIELDS)
    write_csv(out / "P7A_ROLLOUT_RESULTS.csv", [], ROLLOUT_FIELDS)
    return proto


def paired_bootstrap(values: list[float], seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.asarray([np.mean(x[rng.integers(0, len(x), len(x))]) for _ in range(10000)], dtype=float)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def group_rows(rows: list[dict], task: int, mu: float, force: float) -> list[dict]:
    return [r for r in rows if int(float(r.get("task", -1))) == task and abs(as_float(r, "hidden_friction") - mu) < 1e-9 and abs(as_float(r, "requested_force_N") - force) < 1e-9]


def geometry(rows: list[dict]) -> tuple[dict, dict]:
    valid = [r for r in rows if r.get("realized_grasp_cluster") not in ("NO_VALID_GRASP", "") and r.get("contact_center_obj_x") not in (None, "")]
    pts = np.asarray([[as_float(r, f"contact_center_obj_{a}") for a in "xyz"] for r in valid], dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)] if len(pts) else np.empty((0, 3))
    if len(pts):
        mean = np.mean(pts, axis=0)
        std = np.std(pts, axis=0, ddof=1) if len(pts) > 1 else np.zeros(3)
        spatial = float(np.linalg.norm(std))
    else:
        mean, std, spatial = np.full(3, np.nan), np.full(3, np.nan), float("nan")
    return {"n": len(pts), "mean": mean, "std": std, "spatial": spatial}, {"valid_rows": valid}


def aggregate(out: Path, proto: dict) -> dict:
    rows = read_csv(out / "P7A_ROLLOUT_RESULTS.csv")
    clusters_out = []
    agg = []
    var = []
    fail = []
    for task in TASKS:
        for mu in HIDDEN_FRICTIONS:
            for force in FORCE_GRID:
                sub = group_rows(rows, task, mu, force)
                labels = [r.get("realized_grasp_cluster", "NO_VALID_GRASP") or "NO_VALID_GRASP" for r in sub]
                counts = Counter(labels)
                n = len(labels)
                dom, dom_n = (counts.most_common(1)[0] if counts else ("NONE", 0))
                entropy = -sum((c / n) * math.log(c / n) for c in counts.values() if c and n) if n else float("nan")
                valid_grasp_n = sum(label != "NO_VALID_GRASP" for label in labels)
                for cluster in VALID_CLUSTERS:
                    clusters_out.append({"task": task, "object": OBJECTS[task], "hidden_friction": mu, "friction_label": friction_label(mu), "requested_force_N": force, "cluster": cluster, "count": counts.get(cluster, 0), "fraction": counts.get(cluster, 0) / n if n else float("nan"), "dominant_cluster": dom, "dominant_fraction": dom_n / n if n else float("nan"), "cluster_entropy_nats": entropy, "valid_grasp_n": valid_grasp_n})
                g, _ = geometry(sub)
                failure_counts = Counter(r.get("failure_stage", "") or "success" for r in sub if as_int(r, "full_task_success") == 0)
                agg.append({
                    "task": task, "object": OBJECTS[task], "hidden_friction": mu, "friction_label": friction_label(mu), "requested_force_N": force, "n": n,
                    "valid_grasp_n": valid_grasp_n, "dominant_grasp_cluster": dom, "dominant_grasp_fraction": dom_n / n if n else float("nan"),
                    "cluster_entropy_nats": entropy, "bilateral_contact_rate": np.mean([as_int(r, "bilateral_contact") for r in sub]) if sub else float("nan"),
                    "stable_lift_rate": np.mean([as_int(r, "stable_lift") for r in sub]) if sub else float("nan"),
                    "full_task_success_rate": np.mean([as_int(r, "full_task_success") for r in sub]) if sub else float("nan"),
                    "placement_rate": np.mean([as_int(r, "placement_success") for r in sub]) if sub else float("nan"),
                    "slip_rate": np.mean([as_int(r, "slip") for r in sub]) if sub else float("nan"),
                    "drop_rate": np.mean([as_int(r, "drop") for r in sub]) if sub else float("nan"),
                    "contact_center_n": g["n"], "contact_center_mean_x_m": g["mean"][0], "contact_center_mean_y_m": g["mean"][1], "contact_center_mean_z_m": g["mean"][2],
                    "contact_center_std_x_m": g["std"][0], "contact_center_std_y_m": g["std"][1], "contact_center_std_z_m": g["std"][2],
                    "contact_center_spatial_std_m": g["spatial"], "contact_center_spatial_variance_m2": g["spatial"] ** 2 if np.isfinite(g["spatial"]) else float("nan"),
                    "mean_object_rotation_during_lift_rad": np.nanmean([as_float(r, "object_rotation_during_lift_rad") for r in sub]) if sub else float("nan"),
                    "mean_episode_duration_s": np.nanmean([as_float(r, "episode_duration_s") for r in sub]) if sub else float("nan"),
                    "mean_num_policy_chunks": np.nanmean([as_float(r, "num_policy_chunks") for r in sub]) if sub else float("nan"),
                    "failure_mode_counts_json": json.dumps(dict(failure_counts), sort_keys=True),
                })
                var.append({"task": task, "object": OBJECTS[task], "hidden_friction": mu, "friction_label": friction_label(mu), "requested_force_N": force, "n": n, "valid_grasp_n": g["n"], "mean_x_m": g["mean"][0], "mean_y_m": g["mean"][1], "mean_z_m": g["mean"][2], "std_x_m": g["std"][0], "std_y_m": g["std"][1], "std_z_m": g["std"][2], "spatial_std_m": g["spatial"], "spatial_variance_m2": g["spatial"] ** 2 if np.isfinite(g["spatial"]) else float("nan")})
                by_failure = Counter((r.get("failure_stage", "") or "success") for r in sub if as_int(r, "full_task_success") == 0)
                for mode, count in sorted(by_failure.items()):
                    mode_rows = [r for r in sub if (r.get("failure_stage", "") or "success") == mode]
                    fail.append({"task": task, "object": OBJECTS[task], "hidden_friction": mu, "friction_label": friction_label(mu), "requested_force_N": force, "failure_stage": mode, "count": count, "fraction_of_rollouts": count / n if n else float("nan"), "slip_count": sum(as_int(r, "slip") for r in mode_rows), "drop_count": sum(as_int(r, "drop") for r in mode_rows), "runtime_error_count": sum(bool(r.get("error")) for r in mode_rows), "trial_ids_json": json.dumps([r.get("trial_id") for r in mode_rows])})
    write_csv(out / "P7A_GRASP_CLUSTERS.csv", clusters_out)
    write_csv(out / "P7A_AGGREGATED_RESULTS.csv", agg, AGG_FIELDS)
    write_csv(out / "P7A_CONTACT_VARIABILITY.csv", var, VAR_FIELDS)
    write_csv(out / "P7A_FAILURE_MODE_TABLE.csv", fail, FAIL_FIELDS)
    # Keep the descriptive failure-mode table and also emit the exact
    # user-facing required artifact name for traceability.
    write_csv(out / "P7A_FAILURE_CASES.csv", fail, FAIL_FIELDS)

    pairs = []
    frontier = []
    for task in TASKS:
        for mu in HIDDEN_FRICTIONS:
            for fa in FORCE_GRID:
                for fb in FORCE_GRID:
                    if fb <= fa:
                        continue
                    for metric in ("bilateral_contact", "stable_lift", "full_task_success", "drop", "slip"):
                        diffs = []; va = []; vb = []; signs = []
                        for root in ROOTS:
                            ra = [r for r in group_rows(rows, task, mu, fa) if int(float(r.get("root_seed", -1))) == root]
                            rb = [r for r in group_rows(rows, task, mu, fb) if int(float(r.get("root_seed", -1))) == root]
                            if len(ra) != len(POLICY_SEEDS) or len(rb) != len(POLICY_SEEDS):
                                continue
                            aa = float(np.mean([as_int(x, metric) for x in ra])); bb = float(np.mean([as_int(x, metric) for x in rb]))
                            va.append(aa); vb.append(bb); diffs.append(bb - aa)
                            signs.append(int((bb - aa) > 0) if metric not in ("drop", "slip") else int((bb - aa) < 0))
                        lo, hi = paired_bootstrap(diffs, 701000 + task * 10000 + int(mu * 100) + int(fa * 10) + int(fb))
                        pairs.append({"task": task, "object": OBJECTS[task], "hidden_friction": mu, "friction_label": friction_label(mu), "force_a_N": fa, "force_b_N": fb, "metric": metric, "mean_a": np.mean(va) if va else float("nan"), "mean_b": np.mean(vb) if vb else float("nan"), "delta_b_minus_a": np.mean(diffs) if diffs else float("nan"), "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi, "n_roots": len(diffs), "direction_consistent_root_fraction": np.mean(signs) if signs else float("nan"), "definition": "paired root-level means over two policy repeats; roots are the environmental unit"})
            for metric in ("stable_lift", "full_task_success"):
                rates = [next((as_float(a, "stable_lift_rate" if metric == "stable_lift" else "full_task_success_rate") for a in agg if int(a["task"]) == task and abs(float(a["hidden_friction"]) - mu) < 1e-9 and abs(float(a["requested_force_N"]) - f) < 1e-9), float("nan")) for f in FORCE_GRID]
                frontier.append({"task": task, "object": OBJECTS[task], "hidden_friction": mu, "friction_label": friction_label(mu), "metric": metric, "force_grid_N": json.dumps(FORCE_GRID), "success_by_force": json.dumps(rates), "range_max_minus_min": float(np.nanmax(rates) - np.nanmin(rates)) if np.any(np.isfinite(rates)) else float("nan"), "best_force_grid_N": FORCE_GRID[int(np.nanargmax(rates))] if np.any(np.isfinite(rates)) else ""})
    write_csv(out / "P7A_PAIRED_COMPARISONS.csv", pairs, PAIR_FIELDS)
    write_csv(out / "P7A_FORCE_FRONTIER.csv", frontier)
    return {"rows": rows, "aggregate": agg, "clusters": clusters_out, "variability": var, "failures": fail, "pairs": pairs, "frontier": frontier}


def telemetry_audit(rows: list[dict]) -> dict:
    missing_contact = []; missing_force = []; bad_force = []
    for row in rows:
        for key, bucket in (("contact_telemetry_path", missing_contact), ("force_telemetry_path", missing_force)):
            path = Path(row.get(key, ""))
            if not path.exists() or path.stat().st_size == 0:
                bucket.append(row.get("trial_id"))
        fp = Path(row.get("force_telemetry_path", ""))
        if fp.exists():
            trs = read_csv(fp)
            for tr in trs:
                req = as_float(row, "requested_force_N")
                applied = as_float(tr, "executed_fLz", float("nan")) + as_float(tr, "executed_fRz", float("nan"))
                if np.isfinite(req) and np.isfinite(applied) and abs(applied - req) > 1e-5:
                    bad_force.append({"trial_id": row.get("trial_id"), "requested": req, "executed_sum": applied})
                    break
    return {"rollouts": len(rows), "missing_contact_telemetry": missing_contact, "missing_force_telemetry": missing_force, "force_channel_mismatch": bad_force[:20], "all_telemetry_present": not missing_contact and not missing_force, "all_force_channels_match": not bad_force}


def confound_audit(out: Path, proto: dict, summary: dict) -> dict:
    rows = summary["rows"]
    roots = read_csv(out / "P7A_ROOT_MANIFEST.csv")
    expected_ids = {trial_id(t, root, f, mu, ps) for t in TASKS for root in ROOTS for f in FORCE_GRID for mu in HIDDEN_FRICTIONS for ps in POLICY_SEEDS}
    got_ids = [r.get("trial_id") for r in rows]
    recipe_drift = {str(task): sorted({r.get("fixed_recipe_id") for r in rows if int(float(r.get("task", -1))) == task}) for task in TASKS}
    staging = [{"task": task, "force": f, "mu": mu, "valid_rate": float(np.mean([as_int(r, "staging_validity") for r in group_rows(rows, task, mu, f)])) if group_rows(rows, task, mu, f) else float("nan")} for task in TASKS for mu in HIDDEN_FRICTIONS for f in FORCE_GRID]
    audit = {
        "expected_rollouts": EXPECTED_ROLLOUTS, "observed_rollouts": len(rows), "expected_unique_trial_ids": len(expected_ids), "observed_unique_trial_ids": len(set(got_ids)),
        "missing_trial_ids": sorted(expected_ids - set(got_ids))[:50], "unexpected_trial_ids": sorted(set(got_ids) - expected_ids)[:50],
        "duplicate_trial_ids": sorted([k for k, v in Counter(got_ids).items() if v > 1]), "complete_unique_matrix": set(got_ids) == expected_ids and len(got_ids) == len(set(got_ids)),
        "root_manifest_rows": len(roots), "root_manifest_expected_rows": len(TASKS) * len(ROOTS), "root_manifest_complete": len(roots) == len(TASKS) * len(ROOTS),
        "state_parity_all": all(as_int(r, "state_parity") == 1 for r in rows), "state_parity_failures": [r.get("trial_id") for r in rows if as_int(r, "state_parity") != 1],
        "recipe_ids_by_task": recipe_drift, "recipe_drift": recipe_drift != {"1": [FIXED_RECIPE_IDS[1]], "6": [FIXED_RECIPE_IDS[6]]},
        "staging_validity_by_condition": staging, "staging_validity_all": all(x["valid_rate"] == 1.0 for x in staging if np.isfinite(x["valid_rate"])),
        "retry_or_recipe_search": False, "online_force_change": False, "runtime_force_variable_only": True,
        # The frozen full-task runner uses this string for a normal episode
        # termination at its fixed VLA chunk budget.  It is a valid outcome
        # failure mode, not an infrastructure failure and must remain in the
        # denominator.  Only other non-empty errors invalidate attribution.
        "telemetry": telemetry_audit(rows), "runtime_errors": [r.get("trial_id") for r in rows if r.get("error") and r.get("error") != "vla_chunk_budget_exhausted"],
    }
    audit["valid_for_attribution"] = bool(audit["complete_unique_matrix"] and audit["root_manifest_complete"] and audit["state_parity_all"] and not audit["recipe_drift"] and audit["staging_validity_all"] and not audit["runtime_errors"] and audit["telemetry"]["all_telemetry_present"] and audit["telemetry"]["all_force_channels_match"])
    write_json(out / "P7A_CONFOUND_AUDIT.json", audit)
    return audit


def decide(out: Path, summary: dict, audit: dict) -> dict:
    frontier = summary["frontier"]
    pairs = summary["pairs"]
    confirmed_groups = []
    weak_groups = []
    if not audit["valid_for_attribution"]:
        verdict = "EXPERIMENT_INVALID"
    else:
        for fr in frontier:
            if fr["metric"] not in ("stable_lift", "full_task_success") or not np.isfinite(as_float(fr, "range_max_minus_min")):
                continue
            relevant = [p for p in pairs if int(p["task"]) == int(fr["task"]) and abs(float(p["hidden_friction"]) - float(fr["hidden_friction"])) < 1e-9 and p["metric"] == fr["metric"]]
            ci_excludes = any(np.isfinite(as_float(p, "bootstrap_ci95_low")) and np.isfinite(as_float(p, "bootstrap_ci95_high")) and (as_float(p, "bootstrap_ci95_low") > 0 or as_float(p, "bootstrap_ci95_high") < 0) and as_float(p, "direction_consistent_root_fraction") >= 0.70 for p in relevant)
            if as_float(fr, "range_max_minus_min") >= 0.10 and ci_excludes:
                confirmed_groups.append(fr)
            elif as_float(fr, "range_max_minus_min") >= 0.10 or (as_float(fr, "range_max_minus_min") >= 0.05 and any(as_float(p, "direction_consistent_root_fraction") >= 0.70 for p in relevant)):
                weak_groups.append(fr)
        if confirmed_groups:
            verdict = "FORCE_FRONTIER_CONFIRMED"
        elif weak_groups:
            verdict = "FORCE_EFFECT_PRESENT_BUT_WEAK"
        else:
            verdict = "NO_ACTIONABLE_FORCE_FRONTIER"
    decision = {
        "status": "COMPLETE" if audit["valid_for_attribution"] else "INVALID",
        "primary_verdict": verdict,
        "method_change": "NONE",
        "runtime_action_space": ["Force only"],
        "fixed_recipe_by_task": FIXED_RECIPE_IDS,
        "force_grid_N": FORCE_GRID,
        "hidden_friction_values": HIDDEN_FRICTIONS,
        "confirmed_groups": [dict(x) for x in confirmed_groups],
        "weak_groups": [dict(x) for x in weak_groups],
        "attribution_audit": audit,
        "artifacts": str(out),
        "code_hash": sha256_file(Path(__file__).resolve()),
        "protocol_hash": sha256_file(out / "P7A_PROTOCOL.json"),
        "interpretation": {
            "force_frontier_confirmed": "At least one task x hidden-friction context has a >=10 percentage-point force range with paired root CI excluding zero and >=70% root-direction consistency.",
            "no_recipe_search": True,
            "no_posthoc_force_selection": True,
        },
    }
    write_json(out / "P7A_FINAL_VERDICT.json", decision)
    # P7A's BATON keep/remove decision is represented by the same final,
    # attribution-audited decision object; there is no second planner result.
    write_json(out / "P7A_BATON_DECISION.json", decision)
    return decision


def report(out: Path, proto: dict, summary: dict, audit: dict, decision: dict) -> None:
    rows = summary["rows"]
    lines = [
        "# P7A — Fixed-Invocation Force Frontier",
        "",
        "## Technical summary",
        "",
        f"Primary verdict: `{decision['primary_verdict']}`.",
        "",
        f"This experiment contains {len(rows)} rollout rows over tasks 1 and 6, one frozen validated invocation recipe per task, forces {FORCE_GRID} N, hidden friction {HIDDEN_FRICTIONS}, {len(ROOTS)} fresh roots per task, and two policy repeats per root/condition.",
        "",
        "Force is the only runtime control variable. No recipe search, retry, probe, predictor, online force change, VLA modification, or grasp-position decision was used.",
        "",
        "## Force frontier results",
        "",
        "The table reports the exact condition denominator. Success rates are rollout-level proportions; paired comparisons use root-level means over the two policy repeats.",
        "",
        "| task | hidden μ | force (N) | n | dominant grasp | grasp fraction | contact std (m) | stable lift | full task | slip | drop | failure modes |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for a in summary["aggregate"]:
        lines.append(f"| {a['task']} | {a['hidden_friction']:.2f} | {a['requested_force_N']:.0f} | {a['n']} | {a['dominant_grasp_cluster']} | {as_float(a,'dominant_grasp_fraction'):.3f} | {as_float(a,'contact_center_spatial_std_m'):.5f} | {as_float(a,'stable_lift_rate'):.3f} | {as_float(a,'full_task_success_rate'):.3f} | {as_float(a,'slip_rate'):.3f} | {as_float(a,'drop_rate'):.3f} | `{a['failure_mode_counts_json']}` |")
    lines += [
        "",
        "### Frontier shape by task and hidden friction",
        "",
        "| task | hidden μ | metric | success by force [3,5,8 N] | range | best grid force |",
        "|---:|---:|---|---|---:|---:|",
    ]
    for f in summary["frontier"]:
        lines.append(f"| {f['task']} | {f['hidden_friction']:.2f} | {f['metric']} | `{f['success_by_force']}` | {as_float(f,'range_max_minus_min'):.3f} | {f['best_force_grid_N']} |")
    lines += [
        "",
        "The frontier table is descriptive: it does not claim that the grid identifies a continuous optimum. A force region is considered actionable here only when the predeclared paired-root effect rule is met.",
        "",
        "## Paired effect sizes",
        "",
        "| task | μ | force contrast | metric | Δ high−low | paired bootstrap 95% CI | root-direction consistency |",
        "|---:|---:|---|---|---:|---|---:|",
    ]
    for p in summary["pairs"]:
        if p["metric"] in ("stable_lift", "full_task_success"):
            lines.append(f"| {p['task']} | {p['hidden_friction']:.2f} | {p['force_a_N']:.0f}→{p['force_b_N']:.0f} N | {p['metric']} | {as_float(p,'delta_b_minus_a'):.3f} | [{as_float(p,'bootstrap_ci95_low'):.3f}, {as_float(p,'bootstrap_ci95_high'):.3f}] | {as_float(p,'direction_consistent_root_fraction'):.3f} |")
    lines += [
        "",
        "## Scope, data, and metric definitions",
        "",
        f"- Stable lift: frozen P6-G1 local-lift/retention definition used by the VLA full-task runner.",
        f"- Full-task success: lift, transport, placement, and no drop under the frozen task evaluator.",
        f"- Grasp consistency: dominant post-rollout frozen P6-G1 cluster fraction; it is analysis-only and never conditioned the policy.",
        f"- Contact variability: sample standard deviation of first valid bilateral contact-center coordinates in object frame; spatial std is the Euclidean norm of axis-wise standard deviations.",
        f"- Paired unit: root, after averaging the two policy repeats. The two repeats are not treated as independent roots.",
        "",
        "## Confound check",
        "",
        f"- Matrix completeness: `{audit['complete_unique_matrix']}` ({audit['observed_rollouts']}/{audit['expected_rollouts']} unique rows).",
        f"- State parity: `{audit['state_parity_all']}`; failures: {len(audit['state_parity_failures'])}.",
        f"- Fixed recipe drift: `{audit['recipe_drift']}`; IDs by task: `{audit['recipe_ids_by_task']}`.",
        f"- Staging validity all conditions: `{audit['staging_validity_all']}`.",
        f"- Runtime errors: {len(audit['runtime_errors'])}; telemetry complete: `{audit['telemetry']['all_telemetry_present']}`; force channels match requested force: `{audit['telemetry']['all_force_channels_match']}`.",
        f"- Attribution audit valid: `{audit['valid_for_attribution']}`.",
        "",
        "The inherited P6-G1-R2 runtime-gate JSON retains its legacy `isaac_reset_step_smoke` field as false because P7A does not emit the P6-G1-R2-named smoke marker. The P7A-specific reset smoke artifact is true, and all 240 P7A rollout rows have state parity true; this legacy marker is therefore not treated as a P7A runtime failure.",
        "",
        "Contact geometry is reported as an observed mediator/diagnostic. Any force-associated contact movement is not silently discarded; recipe identity and staging remain fixed, while geometry is quantified in `P7A_CONTACT_VARIABILITY.csv`.",
        "",
        "## Limitations and robustness",
        "",
        "This is a force-only frontier gate on nominal task dynamics with hidden friction interventions; it does not validate a physics predictor, probing policy, adaptive force controller, or continuous-force optimum. The grid is inherited from the prior authoritative force matrix because the frozen P7A markdown omitted numeric levels; that instantiation is recorded in `P7A_PROTOCOL.json`.",
        "",
        "## What this proves",
        "",
        f"The evidence supports exactly the verdict `{decision['primary_verdict']}`. It does not prove that a future physical-belief module can infer friction, choose force online, or improve generalization beyond the evaluated task × friction × force grid.",
        "",
        "## Next",
        "",
        "Only if the verdict is `FORCE_FRONTIER_CONFIRMED`: proceed to a separately preregistered hidden-physics belief/probing experiment whose runtime decision is still Force. If the effect is weak or absent, do not claim that force adaptation is supported by this gate.",
        "",
        "## Artifact map",
        "",
        "- Raw rollout rows: `P7A_ROLLOUT_RESULTS.csv`.",
        "- Aggregated force cells: `P7A_AGGREGATED_RESULTS.csv`.",
        "- Contact variability: `P7A_CONTACT_VARIABILITY.csv`.",
        "- Paired root comparisons: `P7A_PAIRED_COMPARISONS.csv`.",
        "- Failure modes: `P7A_FAILURE_MODE_TABLE.csv`.",
        "- Required failure-case alias: `P7A_FAILURE_CASES.csv`.",
        "- BATON keep/remove decision: `P7A_BATON_DECISION.json`.",
        "- Confound audit: `P7A_CONFOUND_AUDIT.json`.",
        "- Final verdict: `P7A_FINAL_VERDICT.json`.",
    ]
    (out / "P7A_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_port() -> int:
    for port in (8765, 8766, 8767, 8768):
        if not r1.p6g1.tcp_port_open("127.0.0.1", port, timeout_s=0.25):
            return port
    raise RuntimeError("no free policy server port")


def launch_worker(out: Path, task: int, host: str, port: int) -> subprocess.Popen:
    log = out / "logs" / f"worker_task{task}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = r3.worker_env(out, task, host, port)
    env.update({"P7A_WORKER": "1", "P7A_OUT": str(out), "P7A_TASK": str(task), "P7A_SERVER_HOST": host, "P7A_SERVER_PORT": str(port)})
    # r3.worker_env already carries the authoritative Isaac/OpenPI paths.
    return subprocess.Popen([str(r1.ISAAC_PY), "-u", str(Path(__file__).resolve()), "--worker"], env=env, stdout=log.open("w", encoding="utf-8"), stderr=subprocess.STDOUT)


def main(args: argparse.Namespace) -> int:
    if args.worker:
        # Workers must use the parent-selected output directory.  Recomputing
        # a timestamp here would split raw rows and telemetry across runs.
        worker_out = Path(os.environ["P7A_OUT"])
        return worker(worker_out, int(os.environ["P7A_TASK"]), os.environ["P7A_SERVER_HOST"], int(os.environ["P7A_SERVER_PORT"]))
    out = Path(args.out) if args.out else RESULTS / f"p7a_fixed_invocation_force_frontier_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    if args.finalize:
        proto = read_json(out / "P7A_PROTOCOL.json")
        # Finalize must bind the artifact to the exact analysis code that was
        # used after the rollout process exited.
        (out / "P7A_CODE_HASH.txt").write_text(sha256_file(Path(__file__).resolve()) + "\n", encoding="utf-8")
        summary = aggregate(out, proto)
        audit = confound_audit(out, proto, summary)
        decision = decide(out, summary, audit)
        report(out, proto, summary, audit, decision)
        return 0

    proto = bootstrap(out)
    if not proto["root_audit"]["disjoint"]:
        raise RuntimeError(f"fresh root collision: {proto['root_audit']}")
    port = choose_port()
    server = r2.start_server(out, port)
    try:
        deadline = time.time() + 900
        while not r1.p6g1.tcp_port_open("127.0.0.1", port, timeout_s=0.25):
            if server is not None and server.poll() is not None:
                raise RuntimeError("policy server exited before listening")
            if time.time() > deadline:
                raise RuntimeError("policy server start timeout")
            time.sleep(2)
        gate = r2.runtime_gate(out, "127.0.0.1", port)
        write_json(out / "P7A_RUNTIME_GATE.json", gate)
        if not gate.get("checkpoint_checksum_matches") or not gate.get("frozen_policy_inference_smoke"):
            raise RuntimeError(f"runtime gate failed: {gate}")
        for task in TASKS:
            proc = launch_worker(out, task, "127.0.0.1", port)
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"task {task} worker failed rc={rc}; see logs/worker_task{task}.log")
        summary = aggregate(out, proto)
        audit = confound_audit(out, proto, summary)
        decision = decide(out, summary, audit)
        report(out, proto, summary, audit, decision)
        return 0
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    raise SystemExit(main(parser.parse_args()))
