#!/usr/bin/env python3
"""P6-G1-R3: raw frozen VLA versus one fixed verified invocation recipe.

This is the final BATON keep/remove ablation.  It deliberately contains no
recipe search, retries, online grasp choice, GNP, physical query, or training.
The two arms restore the same settled root state before each rollout.
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
R2_SCRIPT = REPO / "analysis/p6g1r2_baton_verified_vla_recipes.py"
R1_SCRIPT = REPO / "analysis/p6g1r1_controller_grasp_vla_handoff.py"
R2_OUT = RESULTS / "p6g1r2_baton_verified_vla_recipes_20260825_205113"
R2_CANDIDATES = R2_OUT / "P6G1R2_RECIPE_CANDIDATES.csv"
R2_CLUSTERS = RESULTS / "p6g1_primitive_ik_vla_grasp_realization_20260825_103146" / "P6G1_REFERENCE_GRASP_CLUSTERS.csv"

TASKS = [1, 6]
OBJECTS = {1: "cream_cheese_1", 6: "butter_1"}
FORCE_N = 8.0
POLICY_SEEDS = [0, 1]
ROOTS = [9800, 9801, 9802, 9803, 9804, 9805, 9806, 9807, 9808, 9809]
FIXED_RECIPE_IDS = {1: "t1_G2_R23_ORIGINAL_P6G1", 6: "t6_G0_R09"}
EXPECTED_CHECKPOINT_HASH = "0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17"
NO_RETRY = True
NO_SEARCH = True
ROOT_SETTLE_STEPS = 20

P6G0 = REPO / "analysis/p6g0_grasp_force_physics_benchmark.py"
SERVER_PYTHON = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/.venv/bin/python")
POLICY_DIR = Path("/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999")
NORM_STATS_DIR = POLICY_DIR / "assets/NathanWu7/tabero"
B5_WRAPPER = RESULTS / "b5_tabero_neutral_20260822_040652/scripts/b5_serve_policy_with_explicit_norm_stats.py"
B5_STUBS = RESULTS / "b5_tabero_neutral_20260822_040652/scripts/stubs"
TABERO_VTLA = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
TABERO_VTLA_SRC = TABERO_VTLA / "src"
TABERO_VTLA_CLIENT_SRC = TABERO_VTLA / "packages/openpi-client/src"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r2 = load_module(R2_SCRIPT, "p6g1r3_r2")
r1 = load_module(R1_SCRIPT, "p6g1r3_r1")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, default)
        return default if v in (None, "") else float(v)
    except Exception:
        return default


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        v = row.get(key, default)
        return default if v in (None, "") else int(float(v))
    except Exception:
        return default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(sha256_file(p).encode())
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


ROOT_FIELDS = ["task", "object", "root_index", "root_seed", "root_state_hash", "object_pose_base_x", "object_pose_base_y", "object_pose_base_z", "object_pose_base_qw", "object_pose_base_qx", "object_pose_base_qy", "object_pose_base_qz", "source", "arm_restore_rule"]
ROLLOUT_FIELDS = [
    "trial_id", "task", "object", "arm", "root_seed", "policy_seed", "policy_repeat_seed", "fixed_recipe_id", "target_mode", "requested_force_N", "friction_label", "com_label", "state_parity", "restore_state_hash", "staging_success", "staging_validity", "staging_time_s", "extra_action_count", "num_policy_chunks", "episode_steps", "episode_duration_s", "bilateral_contact", "stable_lift", "slip", "drop", "object_rotation_during_lift_rad", "contact_retention", "transport_success", "placement_success", "full_task_success", "failure_stage", "realized_grasp_cluster", "contact_center_obj_x", "contact_center_obj_y", "contact_center_obj_z", "wrist_obj_qw", "wrist_obj_qx", "wrist_obj_qy", "wrist_obj_qz", "grasp_depth_m", "left_contact_obj_x", "left_contact_obj_y", "left_contact_obj_z", "right_contact_obj_x", "right_contact_obj_y", "right_contact_obj_z", "contact_telemetry_path", "force_telemetry_path", "policy_log_path", "error"]
CLUSTER_FIELDS = ["task", "arm", "cluster", "count", "fraction", "dominant_cluster", "dominant_fraction", "cluster_entropy_nats", "valid_grasp_count", "no_valid_grasp_count"]
VAR_FIELDS = ["task", "arm", "n", "contact_center_std_x_m", "contact_center_std_y_m", "contact_center_std_z_m", "contact_center_spatial_std_m", "wrist_orientation_std_deg", "grasp_depth_std_m", "contact_center_mean_x_m", "contact_center_mean_y_m", "contact_center_mean_z_m"]
PAIR_FIELDS = ["task", "metric", "arm_a_raw", "arm_b_fixed", "delta_b_minus_a", "bootstrap_ci95_low", "bootstrap_ci95_high", "n_roots", "direction_consistent_root_fraction", "definition"]
FAIL_FIELDS = ["trial_id", "task", "arm", "root_seed", "policy_seed", "failure_stage", "error", "full_task_success", "stable_lift", "drop", "contact_retention", "realized_grasp_cluster"]


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
    # Audit exact root tokens in prior P6-G0/P6-G1/R1/R2 artifacts.  R3 roots
    # are rejected if any prior artifact contains them, including manifests.
    tokens = {str(x) for x in ROOTS}
    hits = []
    for d in RESULTS.iterdir():
        if not d.is_dir() or not d.name.lower().startswith(("p6g0", "p6g1")) or d.name.lower().startswith("p6g1r3"):
            continue
        for p in d.rglob("*"):
            if not p.is_file() or p.stat().st_size > 30_000_000:
                continue
            try:
                txt = p.read_text(errors="ignore")
            except Exception:
                continue
            found = sorted(t for t in tokens if t in txt.split())
            if found:
                hits.append({"path": str(p), "roots": found})
    if hits:
        raise RuntimeError(f"R3 roots collide with prior artifact: {hits[:5]}")
    return {"roots": ROOTS, "prior_artifact_glob": "p6g0*/p6g1*", "collisions": [], "disjoint": True}


def protocol_obj(out: Path, recipes: dict[int, dict], root_audit: dict) -> dict:
    return {
        "name": "P6-G1-R3 Raw Frozen VLA vs Fixed Verified Recipe",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method_change": "NONE",
        "question": "Whether one fixed verified invocation recipe materially improves execution reliability over raw frozen VLA when grasp position is not a runtime variable.",
        "tasks": TASKS,
        "object_by_task": OBJECTS,
        "fresh_roots_per_task": ROOTS,
        "policy_seeds": POLICY_SEEDS,
        "total_rollouts": len(TASKS) * len(ROOTS) * len(POLICY_SEEDS) * 2,
        "matched_root_protocol": "one settled initial state per root, independently restored before Raw and Fixed arms",
        "physics": {"friction": "nominal", "center_of_mass": "centered/default", "force_target_N": FORCE_N},
        "arm_A_raw_frozen_vla": {"sequence": ["task reset/root restore", "frozen VLA full task", "fixed 8N force wrapper"], "staging": False, "target_grasp_conditioning": False, "retry": False},
        "arm_B_fixed_verified_recipe": {"sequence": ["analytic/IK staging", "frozen R2 qualified invocation", "frozen VLA full task", "fixed 8N force wrapper"], "recipes": recipes, "retry": False, "recipe_search": False},
        "frozen_policy": {"checkpoint": str(r2.POLICY_DIR), "checkpoint_hash_sha256": sha256_path(r2.POLICY_DIR), "expected_hash_sha256": EXPECTED_CHECKPOINT_HASH, "config": r2.POLICY_CONFIG, "server_wrapper": str(r2.B5_WRAPPER)},
        "analysis_only_grasp_labels": ["G0", "G1", "G2", "OTHER", "NO_VALID_GRASP"],
        "cluster_definitions": str(R2_CLUSTERS),
        "root_audit": root_audit,
        "prohibited": ["training", "new recipe search", "GNP", "physical query", "DreamTrajectory", "online force change", "changed grasp candidates", "retry"],
        "decision_criteria": {"removal": {"raw_dominant_fraction_each_task_min": 0.75, "stable_lift_abs_gap_max": 0.10, "full_task_abs_gap_max": 0.10, "fixed_variability_reduction_max": 0.30}, "retention": {"stable_lift_or_full_task_improvement_min": 0.10, "or_variability_reduction_min": 0.30, "consistency_fraction": 0.70}},
        "output_directory": str(out),
    }


def worker_env(out: Path, task: int, host: str, port: int) -> dict:
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(REPO), str(r2.p6g1.OPENPI_CLIENT_SRC), str(TABERO_VTLA_SRC), str(TABERO_VTLA_CLIENT_SRC), env.get("PYTHONPATH", "")]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(REPO),
        "HDF5_TRAJ_SOURCE_DIR": str(REPO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
        "P6G1R3_WORKER": "1", "P6G1R3_OUT": str(out), "P6G1R3_TASK": str(task),
        "P6G1R3_SERVER_HOST": host, "P6G1R3_SERVER_PORT": str(port),
    })
    return env


def telemetry_contact(row: dict) -> dict:
    path = Path(row.get("contact_telemetry_path", ""))
    rows = read_csv(path)
    if not rows:
        return {}
    # First bilateral sample is the frozen contact definition used for grasp
    # classification. If absent, keep the best available geometric sample for
    # diagnostics but classify as NO_VALID_GRASP.
    bilateral = next((x for x in rows if x.get("contact_state") == "bilateral"), None)
    x = bilateral or next((x for x in rows if x.get("contact_mid_obj_x", "") not in ("", None)), rows[-1])
    keys = ["left_contact_obj_x", "left_contact_obj_y", "left_contact_obj_z", "right_contact_obj_x", "right_contact_obj_y", "right_contact_obj_z", "contact_mid_obj_x", "contact_mid_obj_y", "contact_mid_obj_z", "wrist_obj_qw", "wrist_obj_qx", "wrist_obj_qy", "wrist_obj_qz", "grasp_depth_m"]
    out = {k: x.get(k, "") for k in keys}
    # R3 output calls the bilateral midpoint a contact center; retain the
    # frozen telemetry spelling too because R2 cluster classification consumes
    # contact_mid_obj_* directly.
    for axis in ("x", "y", "z"):
        out[f"contact_center_obj_{axis}"] = out[f"contact_mid_obj_{axis}"]
    lift_rows = [z for z in rows if z.get("object_z", "") not in ("", None)]
    if len(lift_rows) > 1:
        start_q = np.asarray([as_float(lift_rows[0], f"object_q{q}") for q in ("w", "x", "y", "z")])
        peak = max(lift_rows, key=lambda z: as_float(z, "object_z"))
        peak_q = np.asarray([as_float(peak, f"object_q{q}") for q in ("w", "x", "y", "z")])
        qdot = float(np.dot(start_q / max(np.linalg.norm(start_q), 1e-12), peak_q / max(np.linalg.norm(peak_q), 1e-12)))
        out["object_rotation_during_lift_rad"] = float(2 * math.acos(float(np.clip(abs(qdot), -1.0, 1.0))))
    else:
        out["object_rotation_during_lift_rad"] = ""
    return out


def finish_rollout(base: dict, cluster_rows: list[dict]) -> dict:
    geom = telemetry_contact(base)
    base.update(geom)
    try:
        cl, dist, _ = r2.classify({**base, "bilateral_contact": base.get("bilateral_contact", 0)}, cluster_rows)
    except Exception:
        cl = "NONE"
    base["realized_grasp_cluster"] = "NO_VALID_GRASP" if cl in ("NONE", "") else cl
    base["slip"] = int(as_int(base, "drop") or not as_int(base, "contact_retention"))
    if base.get("object_rotation_during_lift_rad", "") in ("", None):
        base["object_rotation_during_lift_rad"] = as_float(base, "object_angular_speed_rad_s")
    return base


def run_one_arm(env, p6, p4, client, task: int, root_state: Any, root_hash: str, root_seed: int, policy_seed: int, arm: str, recipe: dict | None, library: dict, out: Path, cluster_rows: list[dict]) -> dict:
    import torch
    trial_id = f"p6g1r3_t{task}_root{root_seed}_ps{policy_seed}_{'RAW' if arm == 'A_RAW' else 'FIXED'}"
    t0 = time.perf_counter()
    env.reset_to(copy.deepcopy(root_state), torch.tensor([0], device=env.device), is_relative=True)
    restore_hash = r1.root_hash(p6, env)
    parity = int(restore_hash == root_hash)
    stage = {}
    stage_start = time.perf_counter()
    primitive = None
    if arm == "B_FIXED":
        primitive = r2.make_recipe_primitive(library, recipe)
        stage, _ = r2.stage_recipe(env, p6, p4, task, primitive, trial_id, recipe)
    staging_time = time.perf_counter() - stage_start if arm == "B_FIXED" else 0.0
    meta = {"trial_id": trial_id, "task": task, "root_seed": root_seed, "arm": arm, "primitive_id": recipe["recipe_id"] if recipe else "", "policy_repeat": str(policy_seed), "policy_repeat_seed": f"stream_{policy_seed}", "state_parity": parity, "staging_success": stage.get("staging_validity", 1) if arm == "B_FIXED" else 1, "handoff_ready": stage.get("staging_validity", 1) if arm == "B_FIXED" else 1}
    try:
        result = r1.run_vla_full(env, p6, p4, client, task, primitive, meta, float(env.cfg.sim.dt) * int(env.cfg.decimation), out)
    except Exception as exc:
        result = {"trial_id": trial_id, "task": task, "object": OBJECTS[task], "arm": arm, "root_seed": root_seed, "primitive_id": meta["primitive_id"], "policy_repeat": str(policy_seed), "full_task_success_y": 0, "stable_lift": 0, "drop": 0, "transport_success": 0, "placement_success": 0, "failure_stage": "runtime_exception", "episode_steps": 0, "num_policy_chunks": 0, "error": repr(exc), "contact_telemetry_path": "", "force_telemetry_path": "", "policy_log_path": ""}
    duration = time.perf_counter() - t0
    result.update({"trial_id": trial_id, "task": task, "object": OBJECTS[task], "arm": arm, "root_seed": root_seed, "policy_seed": policy_seed, "policy_repeat_seed": f"stream_{policy_seed}", "fixed_recipe_id": recipe["recipe_id"] if recipe else "", "target_mode": recipe["target_mode"] if recipe else "", "requested_force_N": FORCE_N, "friction_label": "NOMINAL", "com_label": "CENTERED_DEFAULT", "state_parity": parity, "restore_state_hash": restore_hash, "staging_success": stage.get("staging_validity", 1) if arm == "B_FIXED" else 1, "staging_validity": stage.get("staging_validity", 1) if arm == "B_FIXED" else 1, "staging_time_s": staging_time, "extra_action_count": stage.get("steps", 0) if arm == "B_FIXED" else 0, "num_policy_chunks": result.get("num_policy_chunks", 0), "episode_steps": result.get("episode_steps", 0), "episode_duration_s": duration, "bilateral_contact": result.get("bilateral_contact", 0), "stable_lift": result.get("stable_lift", 0), "drop": result.get("drop", 0), "contact_retention": result.get("transport_grasp_retained", result.get("transport_success", 0)), "transport_success": result.get("transport_success", 0), "placement_success": result.get("placement_success", 0), "full_task_success": result.get("full_task_success_y", 0), "failure_stage": result.get("failure_stage", ""), "error": result.get("error", "")})
    return finish_rollout(result, cluster_rows)


def worker(out: Path, task: int, host: str, port: int) -> int:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    try:
        # Isaac/Omniverse imports must happen after SimulationApp exists.
        from openpi_client import websocket_client_policy
        env, p6, p4 = r1.import_env_modules(task)
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        client = websocket_client_policy.WebsocketClientPolicy(host, port)
        library = r2._reference_library
        recipe = recipe_rows()[task]
        clusters = read_csv(R2_CLUSTERS)
        root_path = out / "P6G1R3_ROOT_MANIFEST.csv"
        results_path = out / "P6G1R3_ROLLOUT_RESULTS.csv"
        existing = {r.get("trial_id") for r in read_csv(results_path)}
        for idx, root_seed in enumerate(ROOTS):
            env.reset(seed=int(root_seed))
            r1.p6g1.settle_root_before_hash(env, p6, p4, ROOT_SETTLE_STEPS)
            root_state = env.scene.get_state(is_relative=True)
            h0 = r1.root_hash(p6, env)
            obj0, objq0 = p4._pose_in_base(env, OBJECTS[task])
            root_row = {"task": task, "object": OBJECTS[task], "root_index": idx, "root_seed": root_seed, "root_state_hash": h0, "object_pose_base_x": obj0[0], "object_pose_base_y": obj0[1], "object_pose_base_z": obj0[2], "object_pose_base_qw": objq0[0], "object_pose_base_qx": objq0[1], "object_pose_base_qy": objq0[2], "object_pose_base_qz": objq0[3], "source": f"fresh_env_reset_seed_then_{ROOT_SETTLE_STEPS}_settle_steps", "arm_restore_rule": "deep-copied root state independently restored before each arm/policy seed"}
            current = read_csv(root_path)
            if not any(int(float(x.get("task", -1))) == task and int(float(x.get("root_seed", -1))) == root_seed for x in current):
                append_csv(root_path, root_row, ROOT_FIELDS)
            for policy_seed in POLICY_SEEDS:
                for arm, recipe_arg in (("A_RAW", None), ("B_FIXED", recipe)):
                    tid = f"p6g1r3_t{task}_root{root_seed}_ps{policy_seed}_{'RAW' if arm == 'A_RAW' else 'FIXED'}"
                    if tid in existing:
                        continue
                    row = run_one_arm(env, p6, p4, client, task, root_state, h0, root_seed, policy_seed, arm, recipe_arg, library, out, clusters)
                    append_csv(results_path, row, ROLLOUT_FIELDS)
                    existing.add(tid)
                    print(f"R3_ROLLOUT_COMPLETE task={task} root={root_seed} policy_seed={policy_seed} arm={arm} full={row.get('full_task_success')} cluster={row.get('realized_grasp_cluster')}", flush=True)
        return 0
    except Exception as exc:
        write_json(out / f"P6G1R3_WORKER_ERROR_task{task}.json", {"task": task, "error": repr(exc), "trace": traceback.format_exc()})
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


def bootstrap_ci(values: list[float], seed: int = 61720260825) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.asarray([np.mean(x[rng.integers(0, len(x), len(x))]) for _ in range(10000)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def metric_value(row: dict, metric: str) -> float:
    return as_float(row, metric)


def summarize(out: Path, protocol: dict) -> dict:
    rows = read_csv(out / "P6G1R3_ROLLOUT_RESULTS.csv")
    if len(rows) != 80:
        raise RuntimeError(f"R3 requires 80 unique rollouts, found {len(rows)}")
    keys = [(int(r["task"]), int(r["root_seed"]), int(r["policy_seed"]), r["arm"]) for r in rows]
    if len(set(keys)) != 80 or any(as_int(r, "state_parity") != 1 for r in rows):
        raise RuntimeError("R3 matched-root/parity audit failed")
    clusters = read_csv(R2_CLUSTERS)
    cluster_rows = []; var_rows = []
    variability = {}
    for task in TASKS:
        for arm in ("A_RAW", "B_FIXED"):
            sub = [r for r in rows if int(r["task"]) == task and r["arm"] == arm]
            labels = [r.get("realized_grasp_cluster", "NO_VALID_GRASP") for r in sub]
            cnt = Counter(labels); n = len(labels); probs = [v / n for v in cnt.values()]
            ent = -sum(p * math.log(p) for p in probs if p > 0)
            dom, dom_n = cnt.most_common(1)[0]
            for lab, c in sorted(cnt.items()):
                cluster_rows.append({"task": task, "arm": arm, "cluster": lab, "count": c, "fraction": c / n, "dominant_cluster": dom, "dominant_fraction": dom_n / n, "cluster_entropy_nats": ent, "valid_grasp_count": sum(x not in ("NO_VALID_GRASP",) for x in labels), "no_valid_grasp_count": sum(x == "NO_VALID_GRASP" for x in labels)})
            pts = np.asarray([[as_float(r, "contact_center_obj_x"), as_float(r, "contact_center_obj_y"), as_float(r, "contact_center_obj_z")] for r in sub if r.get("contact_center_obj_x", "") not in ("", None) and r.get("realized_grasp_cluster") != "NO_VALID_GRASP"], dtype=float)
            qs = np.asarray([[as_float(r, "wrist_obj_qw"), as_float(r, "wrist_obj_qx"), as_float(r, "wrist_obj_qy"), as_float(r, "wrist_obj_qz")] for r in sub if r.get("wrist_obj_qw", "") not in ("", None) and r.get("realized_grasp_cluster") != "NO_VALID_GRASP"], dtype=float)
            depths = np.asarray([as_float(r, "grasp_depth_m") for r in sub if r.get("grasp_depth_m", "") not in ("", None) and r.get("realized_grasp_cluster") != "NO_VALID_GRASP"], dtype=float)
            meanp = pts.mean(axis=0) if len(pts) else np.full(3, np.nan)
            stdp = pts.std(axis=0, ddof=1) if len(pts) > 1 else np.full(3, np.nan)
            spatial = float(np.sqrt(np.sum(stdp ** 2))) if np.all(np.isfinite(stdp)) else float("nan")
            if len(qs):
                q0 = qs[0] / max(np.linalg.norm(qs[0]), 1e-12)
                ang = [2 * math.degrees(math.acos(np.clip(abs(float(np.dot(q / max(np.linalg.norm(q), 1e-12), q0))), -1.0, 1.0))) for q in qs]
                qstd = float(np.std(ang, ddof=1)) if len(ang) > 1 else 0.0
            else:
                qstd = float("nan")
            dstd = float(np.std(depths, ddof=1)) if len(depths) > 1 else float("nan")
            variability[(task, arm)] = {"spatial": spatial, "n": len(pts)}
            var_rows.append({"task": task, "arm": arm, "n": len(pts), "contact_center_std_x_m": stdp[0] if len(pts) > 1 else "", "contact_center_std_y_m": stdp[1] if len(pts) > 1 else "", "contact_center_std_z_m": stdp[2] if len(pts) > 1 else "", "contact_center_spatial_std_m": spatial if np.isfinite(spatial) else "", "wrist_orientation_std_deg": qstd if np.isfinite(qstd) else "", "grasp_depth_std_m": dstd if np.isfinite(dstd) else "", "contact_center_mean_x_m": meanp[0] if len(pts) else "", "contact_center_mean_y_m": meanp[1] if len(pts) else "", "contact_center_mean_z_m": meanp[2] if len(pts) else ""})
    write_csv(out / "P6G1R3_GRASP_CLUSTERS.csv", cluster_rows, CLUSTER_FIELDS)
    write_csv(out / "P6G1R3_CONTACT_VARIABILITY.csv", var_rows, VAR_FIELDS)

    pair_rows = []
    for task in TASKS:
        for metric in ("stable_lift", "full_task_success", "contact_retention", "episode_duration_s"):
            diffs = []
            root_consistency = []
            for root in ROOTS:
                a = [r for r in rows if int(r["task"]) == task and int(r["root_seed"]) == root and r["arm"] == "A_RAW"]
                b = [r for r in rows if int(r["task"]) == task and int(r["root_seed"]) == root and r["arm"] == "B_FIXED"]
                da = np.mean([metric_value(x, metric) for x in a]); db = np.mean([metric_value(x, metric) for x in b]); d = float(db - da); diffs.append(d); root_consistency.append(int(d > 0) if metric != "episode_duration_s" else int(d < 0))
            lo, hi = bootstrap_ci(diffs, seed=61720260825 + task)
            pair_rows.append({"task": task, "metric": metric, "arm_a_raw": np.mean([metric_value(r, metric) for r in rows if int(r["task"]) == task and r["arm"] == "A_RAW"]), "arm_b_fixed": np.mean([metric_value(r, metric) for r in rows if int(r["task"]) == task and r["arm"] == "B_FIXED"]), "delta_b_minus_a": np.mean(diffs), "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi, "n_roots": len(ROOTS), "direction_consistent_root_fraction": np.mean(root_consistency), "definition": "root-level mean across two policy seeds; paired bootstrap resamples the 10 environmental roots"})
        raw_sp = variability[(task, "A_RAW")]["spatial"]; fix_sp = variability[(task, "B_FIXED")]["spatial"]
        reduction = 1.0 - fix_sp / raw_sp if raw_sp > 0 and np.isfinite(raw_sp) and np.isfinite(fix_sp) else float("nan")
        pooled_means = {}
        for arm in ("A_RAW", "B_FIXED"):
            pts_all = np.asarray([[as_float(r, "contact_center_obj_x"), as_float(r, "contact_center_obj_y"), as_float(r, "contact_center_obj_z")] for r in rows if int(r["task"]) == task and r["arm"] == arm and r.get("contact_center_obj_x", "") not in ("", None) and r.get("realized_grasp_cluster") != "NO_VALID_GRASP"], dtype=float)
            pooled_means[arm] = pts_all.mean(axis=0) if len(pts_all) else np.full(3, np.nan)
        root_spreads = {"A_RAW": [], "B_FIXED": []}
        for root in ROOTS:
            for arm in ("A_RAW", "B_FIXED"):
                pts = np.asarray([[as_float(r, "contact_center_obj_x"), as_float(r, "contact_center_obj_y"), as_float(r, "contact_center_obj_z")] for r in rows if int(r["task"]) == task and int(r["root_seed"]) == root and r["arm"] == arm and r.get("contact_center_obj_x", "") not in ("", None) and r.get("realized_grasp_cluster") != "NO_VALID_GRASP"], dtype=float)
                root_spreads[arm].append(float(np.sqrt(np.mean(np.sum((pts - pooled_means[arm]) ** 2, axis=1)))) if len(pts) and np.all(np.isfinite(pooled_means[arm])) else float("nan"))
        paired_spread = [b - a for a, b in zip(root_spreads["A_RAW"], root_spreads["B_FIXED"]) if np.isfinite(a) and np.isfinite(b)]
        spread_lo, spread_hi = bootstrap_ci(paired_spread, seed=61720260825 + task + 100) if paired_spread else (float("nan"), float("nan"))
        spread_consistency = float(np.mean(np.asarray(paired_spread) < 0)) if paired_spread else float("nan")
        pair_rows.append({"task": task, "metric": "contact_center_spatial_std", "arm_a_raw": raw_sp, "arm_b_fixed": fix_sp, "delta_b_minus_a": fix_sp - raw_sp, "bootstrap_ci95_low": spread_lo, "bootstrap_ci95_high": spread_hi, "n_roots": len(paired_spread), "direction_consistent_root_fraction": spread_consistency, "definition": "pooled 3D sample std norm across valid bilateral grasp contacts; CI uses paired root-level dispersion around the arm pooled centroid"})
        pair_rows.append({"task": task, "metric": "contact_variance_reduction", "arm_a_raw": 0.0, "arm_b_fixed": reduction, "delta_b_minus_a": reduction, "bootstrap_ci95_low": (-spread_hi / raw_sp if raw_sp > 0 and np.isfinite(spread_hi) else ""), "bootstrap_ci95_high": (-spread_lo / raw_sp if raw_sp > 0 and np.isfinite(spread_lo) else ""), "n_roots": len(paired_spread), "direction_consistent_root_fraction": spread_consistency, "definition": "1 - fixed-recipe pooled spatial std / raw pooled spatial std; CI transformed from paired contact-spread delta"})
    write_csv(out / "P6G1R3_PAIRED_COMPARISONS.csv", pair_rows, PAIR_FIELDS)

    fail_rows = [{k: r.get(k, "") for k in FAIL_FIELDS} for r in rows if as_int(r, "full_task_success") == 0]
    write_csv(out / "P6G1R3_FAILURE_CASES.csv", fail_rows, FAIL_FIELDS)

    def stat(task: int, arm: str, metric: str) -> float:
        return float(np.mean([metric_value(r, metric) for r in rows if int(r["task"]) == task and r["arm"] == arm]))
    decisions = {}
    for task in TASKS:
        raw_dom = max([as_float(x, "dominant_fraction") for x in cluster_rows if int(x["task"]) == task and x["arm"] == "A_RAW"] or [0.0])
        stable_gap = abs(stat(task, "B_FIXED", "stable_lift") - stat(task, "A_RAW", "stable_lift"))
        full_gap = abs(stat(task, "B_FIXED", "full_task_success") - stat(task, "A_RAW", "full_task_success"))
        raw_sp = variability[(task, "A_RAW")]["spatial"]; fix_sp = variability[(task, "B_FIXED")]["spatial"]
        reduction = 1 - fix_sp / raw_sp if raw_sp > 0 and np.isfinite(raw_sp) and np.isfinite(fix_sp) else 0.0
        root_lift = [metric_value(next(r for r in rows if int(r["task"]) == task and int(r["root_seed"]) == root and r["arm"] == "B_FIXED" and int(r["policy_seed"]) == ps), "stable_lift") - metric_value(next(r for r in rows if int(r["task"]) == task and int(r["root_seed"]) == root and r["arm"] == "A_RAW" and int(r["policy_seed"]) == ps), "stable_lift") for root in ROOTS for ps in POLICY_SEEDS]
        root_full = [metric_value(next(r for r in rows if int(r["task"]) == task and int(r["root_seed"]) == root and r["arm"] == "B_FIXED" and int(r["policy_seed"]) == ps), "full_task_success") - metric_value(next(r for r in rows if int(r["task"]) == task and int(r["root_seed"]) == root and r["arm"] == "A_RAW" and int(r["policy_seed"]) == ps), "full_task_success") for root in ROOTS for ps in POLICY_SEEDS]
        decisions[str(task)] = {"raw_dominant_fraction": raw_dom, "stable_lift_abs_gap": stable_gap, "full_task_abs_gap": full_gap, "contact_center_variability_reduction": reduction, "fixed_stable_lift_improvement": stat(task, "B_FIXED", "stable_lift") - stat(task, "A_RAW", "stable_lift"), "fixed_full_task_improvement": stat(task, "B_FIXED", "full_task_success") - stat(task, "A_RAW", "full_task_success"), "root_direction_consistency": {"stable_lift": float(np.mean(np.asarray(root_lift) > 0)), "full_task": float(np.mean(np.asarray(root_full) > 0))}}
    remove = all(x["raw_dominant_fraction"] >= .75 and x["stable_lift_abs_gap"] <= .10 and x["full_task_abs_gap"] <= .10 and x["contact_center_variability_reduction"] < .30 for x in decisions.values())
    retain_adv = all(max(x["fixed_stable_lift_improvement"], x["fixed_full_task_improvement"], x["contact_center_variability_reduction"]) >= .10 for x in decisions.values())
    retain_consistent = all(max(x["root_direction_consistency"]["stable_lift"], x["root_direction_consistency"]["full_task"]) >= .70 for x in decisions.values())
    if remove:
        classification = "P6G1R3_RAW_VLA_CANONICAL_GRASP_SUFFICIENT_BATON_REMOVABLE"
        baton_removal = "YES"
    elif retain_adv and retain_consistent:
        classification = "P6G1R3_FIXED_RECIPE_USEFUL_AS_EXECUTION_SUBSTRATE"
        baton_removal = "NO"
    else:
        classification = "P6G1R3_RAW_VLA_EXECUTION_VARIABILITY_TOO_HIGH"
        baton_removal = "NO"
    decision = {"baton_removal": baton_removal, "primary_classification": classification, "criteria": {"removal": remove, "retention_advantage_all_tasks": retain_adv, "retention_consistency_all_tasks": retain_consistent}, "per_task": decisions, "n_rollouts": len(rows), "no_recipe_search": True, "no_retry": True}
    write_json(out / "P6G1R3_BATON_DECISION.json", decision)
    return {"rows": rows, "clusters": cluster_rows, "variability": var_rows, "pairs": pair_rows, "decision": decision}


def report(out: Path, protocol: dict, summary: dict) -> None:
    d = summary["decision"]
    next_name = "P7A_RAW_VLA_FORCE_FRONTIER_PROTOCOL.md" if d["baton_removal"] == "YES" else "P7A_FIXED_INVOCATION_FORCE_FRONTIER_PROTOCOL.md"
    rows = summary["rows"]
    lines = ["# P6-G1-R3 — Raw Frozen VLA vs Fixed Verified Recipe", "", f"PRIMARY_CLASSIFICATION: {d['primary_classification']}", f"BATON_REMOVAL: {d['baton_removal']}", "", "## Protocol audit", "", f"- Tasks: {TASKS}", f"- Fresh roots per task: {len(ROOTS)} ({ROOTS[0]}–{ROOTS[-1]})", f"- Policy repeats: {POLICY_SEEDS}", f"- Rollouts: {len(rows)} / required 80", "- Nominal friction; centered/default CoM; fixed 8N", "- Recipe search: NO", "- Retry: NO", "- Runtime grasp-position decision: NO", "", "## Metrics", "", "| task | arm | dominant cluster | dominant fraction | entropy (nats) | bilateral contact | stable lift | drop | placement | full task | duration (s) |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for arm in ("A_RAW", "B_FIXED"):
            cr = [x for x in summary["clusters"] if int(x["task"]) == task and x["arm"] == arm]
            vr = next(x for x in summary["variability"] if int(x["task"]) == task and x["arm"] == arm)
            sub = [r for r in rows if int(r["task"]) == task and r["arm"] == arm]
            dom = max(cr, key=lambda x: as_float(x, "fraction"))
            lines.append(f"| {task} | {arm} | {dom['cluster']} | {as_float(dom,'fraction'):.3f} | {as_float(dom,'cluster_entropy_nats'):.3f} | {np.mean([as_int(x,'bilateral_contact') for x in sub]):.3f} | {np.mean([as_int(x,'stable_lift') for x in sub]):.3f} | {np.mean([as_int(x,'drop') for x in sub]):.3f} | {np.mean([as_int(x,'placement_success') for x in sub]):.3f} | {np.mean([as_int(x,'full_task_success') for x in sub]):.3f} | {np.mean([as_float(x,'episode_duration_s') for x in sub]):.2f} |")
    lines += ["", "## Contact variability", "", "See `P6G1R3_CONTACT_VARIABILITY.csv` for spatial, wrist-orientation, and depth standard deviations. The paired contact-center effects and 95% root bootstrap intervals are in `P6G1R3_PAIRED_COMPARISONS.csv`.", "", "## Scientific interpretation", "", "1. This is a matched-root ablation of a fixed invocation substrate, not a recipe-search or grasp-position planning experiment.", "2. Raw grasp labels are post-rollout analysis only; Arm A was never conditioned on G0/G1/G2.", "3. The decision uses the prespecified removal criteria, with paired root bootstrap intervals in `P6G1R3_PAIRED_COMPARISONS.csv`.", "4. Grasp position remains outside the runtime action space; the only planned physical runtime variable for the next gate is force.", "", f"## Next protocol", "", f"Create and run `{next_name}`: one fixed invocation choice (Raw VLA if removable, otherwise one validated recipe per task) crossed with hidden friction and grip force, with no recipe search, selection, or retry."]
    (out / "P6G1R3_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "P6G1R3_FINAL_VERDICT.json", {"status": "COMPLETE", "primary_classification": d["primary_classification"], "baton_removal": d["baton_removal"], "final_method_decision": "Raw Frozen VLA" if d["baton_removal"] == "YES" else "One fixed validated invocation protocol", "runtime_action_space": ["Force only"], "next_protocol": next_name, "artifacts": str(out), "protocol_hash": sha256_file(out / "P6G1R3_PROTOCOL.json"), "code_hash": sha256_file(Path(__file__).resolve()), "audit": {"rollouts": len(rows), "unique_trial_ids": len({r['trial_id'] for r in rows}), "state_parity_all": all(as_int(r, 'state_parity') == 1 for r in rows), "recipe_search": False, "retry": False}})
    if d["baton_removal"] == "YES":
        (out / next_name).write_text("# P7A — Raw Frozen VLA Force Frontier Protocol\n\nUse Raw Frozen VLA canonical grasp with no BATON layer, crossed with hidden friction and grip force. Keep task reset, fresh matched roots, centered/default CoM, and no retry or recipe search. Estimate `P(Y_full=1 | task, F, hidden physics)` and `F*_grid(mu)`. Runtime action space: force only.\n", encoding="utf-8")
    else:
        (out / next_name).write_text("# P7A — Fixed Invocation Force Frontier Protocol\n\nUse exactly one frozen validated invocation recipe per task, crossed with hidden friction and grip force. No recipe search, selection, retry, online grasp-position decision, or online force change. Estimate `P(Y_full=1 | task, F, hidden physics)` and `F*_grid(mu)`. Runtime action space: force only.\n", encoding="utf-8")


def choose_port() -> int:
    for p in (8765, 8766, 8767):
        if not r2.p6g1.tcp_port_open("127.0.0.1", p, timeout_s=.25):
            return p
    raise RuntimeError("no free policy server port")


def main(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else RESULTS / f"p6g1r3_raw_vla_vs_fixed_recipe_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    out.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        protocol = read_json(out / "P6G1R3_PROTOCOL.json")
        rows = read_csv(out / "P6G1R3_ROLLOUT_RESULTS.csv")
        rewritten = []
        clusters = read_csv(R2_CLUSTERS)
        for row in rows:
            rewritten.append(finish_rollout(row, clusters))
        write_csv(out / "P6G1R3_ROLLOUT_RESULTS.csv", rewritten, ROLLOUT_FIELDS)
        (out / "P6G1R3_CODE_HASH.txt").write_text(sha256_file(Path(__file__).resolve()) + "\n", encoding="utf-8")
        summary = summarize(out, protocol)
        report(out, protocol, summary)
        return 0
    root_audit = prior_root_audit()
    recipes = recipe_rows()
    protocol = protocol_obj(out, recipes, root_audit)
    write_json(out / "P6G1R3_PROTOCOL.json", protocol)
    (out / "P6G1R3_PROTOCOL_HASH.txt").write_text(sha256_file(out / "P6G1R3_PROTOCOL.json") + "\n", encoding="utf-8")
    (out / "P6G1R3_CODE_HASH.txt").write_text(sha256_file(Path(__file__).resolve()) + "\n", encoding="utf-8")
    write_csv(out / "P6G1R3_ROOT_MANIFEST.csv", [], ROOT_FIELDS)
    write_csv(out / "P6G1R3_ROLLOUT_RESULTS.csv", [], ROLLOUT_FIELDS)
    port = choose_port()
    server = r2.start_server(out, port)
    try:
        deadline = time.time() + 900
        while not r2.p6g1.tcp_port_open("127.0.0.1", port, timeout_s=.25):
            if server is not None and server.poll() is not None:
                raise RuntimeError("policy server exited before listening")
            if time.time() > deadline:
                raise RuntimeError("policy server start timeout")
            time.sleep(2)
        gate = r2.runtime_gate(out, "127.0.0.1", port)
        write_json(out / "P6G1R3_RUNTIME_GATE.json", gate)
        if not gate.get("checkpoint_checksum_matches") or not gate.get("frozen_policy_inference_smoke"):
            raise RuntimeError(f"runtime gate failed: {gate}")
        for task in TASKS:
            log = (out / f"worker_task{task}.log").open("w", encoding="utf-8")
            proc = subprocess.Popen([str(ISAAC_PY), "-u", str(Path(__file__).resolve()), "--worker"], env=worker_env(out, task, "127.0.0.1", port), stdout=log, stderr=subprocess.STDOUT)
            rc = proc.wait()
            log.close()
            if rc != 0:
                raise RuntimeError(f"task {task} worker failed rc={rc}; see worker_task{task}.log")
        summary = summarize(out, protocol)
        report(out, protocol, summary)
        return 0
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ns = ap.parse_args()
    if ns.worker:
        raise SystemExit(worker(Path(os.environ["P6G1R3_OUT"]), int(os.environ["P6G1R3_TASK"]), os.environ["P6G1R3_SERVER_HOST"], int(os.environ["P6G1R3_SERVER_PORT"])) )
    raise SystemExit(main(ns))
