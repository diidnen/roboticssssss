#!/usr/bin/env python3
"""P6-G0-R1 confirmatory completion plus geometry-aware grasp coverage.

This script intentionally keeps the two scientific tracks separate:

Track A is a strict completion/resume of the frozen P6-G0 task1/task6
protocol.  It reuses the original candidate definitions and hidden physics,
and it only shards the execution process more finely.

Track B is a new deterministic geometry-aware candidate generator and nominal
physics preflight for tasks that failed the simple-offset candidate screen.
It never contributes rows to Track A statistics.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/home/exouser/Tabero")
ANALYSIS = REPO / "analysis"
RESULTS_ROOT = ANALYSIS / "results"
PREV = RESULTS_ROOT / "p6g0_grasp_force_physics_benchmark_20260824_182251"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path(
    "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.warp.core-1.8.2+lx64"
)
OPENPI_CLIENT_SRC = REPO / "benchmarks/openpi/openpi-client/src"

TASK_OBJECTS = {
    0: "alphabet_soup_1",
    1: "cream_cheese_1",
    2: "salad_dressing_1",
    5: "tomato_sauce_1",
    6: "butter_1",
}
TASK_INSTRUCTIONS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
BASKET_NAME = "basket_1"
TASK_SUITE = "libero_object"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"

TRACK_A_TASKS = [1, 6]
TRACK_B_TARGET_TASKS = [0, 2, 5]
TRACK_B_REFERENCE_TASKS = [1, 6]
FRICTIONS = [0.25, 0.90]
COM_SIGNS = [-1.0, 0.0, 1.0]
COARSE_FORCES = [3.0, 5.0, 8.0]
REFINEMENT_FORCES = [4.0, 6.0]
ALL_FORCES = [3.0, 4.0, 5.0, 6.0, 8.0]
PREFLIGHT_B_ROOTS = [7100, 7101, 7102, 7103, 7104]
GRASP_LABELS_A = ["g_minus", "g_center", "g_plus"]
GRASP_LABELS_B = ["G0", "G1", "G2"]
F_MAX = 8.0

SURFACE_SAMPLE_COUNT = 4096
SURFACE_SAMPLE_SEED = 73001
ANTIPODAL_TOL_DEG = 30.0
ROLL_SAMPLES_DEG = [-45.0, 0.0, 45.0]
GRIPPER_WIDTH_MIN_M = 0.012
GRIPPER_WIDTH_MAX_M = 0.090
PREGRASP_OFFSET_M = 0.10
GRASP_APPROACH_OFFSET_M = 0.00
TRACK_B_FORCE_N = 8.0
TRACK_B_FRICTION = 0.5

APPROACH_STEPS = 45
DESCEND_STEPS = 35
CLOSE_STEPS = 70
HOLD_STEPS = 40
LIFT_STEPS = 45
D_OPEN = 0.04
D_CLOSED = 0.0

TIMEOUTS_S = {
    "track_a_shard": 5400,
    "track_b_task": 3600,
    "rollout": 600,
    "restore": 30,
}


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


OUT = Path(os.environ.get("P6G0R1_OUT", RESULTS_ROOT / f"p6g0r1_confirmatory_and_candidate_coverage_{now_tag()}"))
TRACK_A = OUT / "TRACK_A"
TRACK_B = OUT / "TRACK_B"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    if not fields:
        path.write_text("", encoding="utf-8")
        return
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


def read_csv_if(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def stable_hash_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def ftoken(x: float) -> str:
    return f"{float(x):g}"


def branch_key(task: int, root_seed: int, mu: float, com_sign: float, grasp_label: str, force: float) -> tuple:
    return (int(task), int(root_seed), ftoken(mu), ftoken(com_sign), str(grasp_label), ftoken(force))


def branch_key_from_row(row: dict) -> tuple:
    return branch_key(
        int(float(row["task"])),
        int(float(row["root_seed"])),
        float(row.get("hidden_friction_analysis_only", row.get("hidden_friction", 0.0))),
        float(row.get("hidden_com_offset_axis_sign", row.get("hidden_com_sign", 0.0))),
        row["grasp_label"],
        float(row["requested_force_N"]),
    )


def context_id(task: int, root_seed: int, mu: float, com_sign: float) -> str:
    return f"p6g0_t{task}_s{root_seed}_mu{mu:g}_com{com_sign:+g}"


def branch_id(task: int, root_seed: int, mu: float, com_sign: float, grasp_label: str, force: float) -> str:
    return f"{context_id(task, root_seed, mu, com_sign)}_{grasp_label}_F{force:g}"


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


_P6 = None


def p6():
    global _P6
    if _P6 is None:
        sys.path.insert(0, str(ANALYSIS))
        _P6 = importlib.import_module("p6g0_grasp_force_physics_benchmark")
    _P6.OUT = OUT / "_p6g0_import"
    return _P6


def worker_env(extra: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(REPO), str(OPENPI_CLIENT_SRC)]),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "TABERO_ROOT": str(REPO),
            "HDF5_TRAJ_SOURCE_DIR": str(REPO / "benchmarks/datasets/libero/assembled_hdf5"),
            "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
            "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
            "P6G0R1_OUT": str(OUT),
        }
    )
    env.update(extra)
    return env


def load_prev_protocol() -> dict:
    return json.loads((PREV / "P6G0_PROTOCOL.json").read_text(encoding="utf-8"))


def load_prev_candidates(task: int | None = None) -> list[dict]:
    rows = read_csv_if(PREV / "P6G0_GRASP_CANDIDATE_MANIFEST.csv")
    if task is not None:
        rows = [r for r in rows if int(float(r["task"])) == int(task)]
    return rows


def candidate_axis_and_lg(task: int) -> tuple[np.ndarray, float]:
    rows = load_prev_candidates(task)
    if not rows:
        raise RuntimeError(f"No previous candidates for task {task}")
    r = rows[0]
    axis = np.array([as_float(r, "axis_object_x"), as_float(r, "axis_object_y"), as_float(r, "axis_object_z")], dtype=np.float64)
    return axis, as_float(r, "Lg_m")


def com_offset_for_task(task: int, com_sign: float) -> np.ndarray:
    axis, lg = candidate_axis_and_lg(task)
    return float(com_sign) * 0.15 * lg * axis


def planned_coarse_rows() -> list[dict]:
    protocol = load_prev_protocol()
    rows = []
    for task in TRACK_A_TASKS:
        for root_seed in protocol["pilot_roots"]:
            for mu in protocol["frictions"]:
                for com_sign in COM_SIGNS:
                    for cand in load_prev_candidates(task):
                        for force in protocol["force_grid"]:
                            rows.append(
                                {
                                    "task": task,
                                    "root_seed": int(root_seed),
                                    "hidden_friction_analysis_only": float(mu),
                                    "hidden_com_offset_axis_sign": float(com_sign),
                                    "grasp_label": cand["grasp_label"],
                                    "requested_force_N": float(force),
                                    "branch_id": branch_id(task, int(root_seed), float(mu), float(com_sign), cand["grasp_label"], float(force)),
                                    "context_id": context_id(task, int(root_seed), float(mu), float(com_sign)),
                                    "planning_stage": "coarse_force_grid",
                                }
                            )
    return rows


def valid_prev_completed() -> list[dict]:
    rows = []
    for row in read_csv_if(PREV / "P6G0_BRANCH_MANIFEST.csv"):
        if int(float(row["task"])) not in TRACK_A_TASKS:
            continue
        if as_int(row, "state_parity", 1) != 1:
            continue
        rows.append(row)
    return rows


def infer_dynamic_planned_from_completed(completed: list[dict]) -> list[dict]:
    rows = planned_coarse_rows()
    keys = {branch_key_from_row(r) for r in rows}
    by_ctx: dict[tuple, list[dict]] = {}
    for row in completed:
        ctx = (
            int(float(row["task"])),
            int(float(row["root_seed"])),
            ftoken(float(row["hidden_friction_analysis_only"])),
            ftoken(float(row["hidden_com_offset_axis_sign"])),
        )
        by_ctx.setdefault(ctx, []).append(row)
    for ctx, ctx_rows in by_ctx.items():
        task, root_seed, mu_s, com_s = ctx
        mu = float(mu_s)
        com_sign = float(com_s)
        outcome = {(r["grasp_label"], ftoken(float(r["requested_force_N"]))): as_int(r, "full_task_success_y") for r in ctx_rows}
        trigger4 = any(outcome.get((g, "3")) == 0 and outcome.get((g, "5")) == 1 for g in GRASP_LABELS_A)
        trigger6 = any(outcome.get((g, "5")) == 0 and outcome.get((g, "8")) == 1 for g in GRASP_LABELS_A)
        for force, triggered, label in [(4.0, trigger4, "refinement_3fail_5success"), (6.0, trigger6, "refinement_5fail_8success")]:
            if not triggered:
                continue
            for g in GRASP_LABELS_A:
                key = branch_key(task, root_seed, mu, com_sign, g, force)
                if key in keys:
                    continue
                rows.append(
                    {
                        "task": task,
                        "root_seed": root_seed,
                        "hidden_friction_analysis_only": mu,
                        "hidden_com_offset_axis_sign": com_sign,
                        "grasp_label": g,
                        "requested_force_N": force,
                        "branch_id": branch_id(task, root_seed, mu, com_sign, g, force),
                        "context_id": context_id(task, root_seed, mu, com_sign),
                        "planning_stage": label,
                    }
                )
                keys.add(key)
    return rows


BRANCH_FIELDS = [
    "local_grasp_success",
    "lift_success",
    "full_task_success_y",
    "drop",
    "slip_onset",
    "transport_failure",
    "placement_failure",
    "collision",
    "measured_force_mean_N",
    "measured_force_peak_N",
    "steady_state_mean_N",
    "force_tracking_error_N",
    "episode_length",
    "telemetry_path",
    "branch_id",
    "context_id",
    "task",
    "task_instruction",
    "root_seed",
    "hidden_friction_analysis_only",
    "hidden_com_offset_axis_sign",
    "hidden_com_offset_object_frame_m",
    "grasp_label",
    "requested_force_N",
    "state_parity",
    "source_run",
]

PARITY_FIELDS = [
    "context_id",
    "task",
    "root_seed",
    "hidden_friction",
    "hidden_com_sign",
    "grasp_label",
    "requested_force_N",
    "reference_state_hash",
    "post_restore_state_hash",
    "parity_pass",
    "source_run",
]

STAGE_FIELDS = [
    "utc",
    "track",
    "worker_id",
    "task",
    "root_seed",
    "hidden_friction",
    "hidden_com_sign",
    "stage",
    "grasp_label",
    "requested_force_N",
    "branch_id",
    "message",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stage_log(row: dict) -> None:
    base = {k: "" for k in STAGE_FIELDS}
    base.update(row)
    base["utc"] = base.get("utc") or utc_now()
    append_csv(TRACK_A / "P6G0R1A_STAGE_LOG.csv", base, STAGE_FIELDS)


def setup_isaac_task(task_id: int):
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
    simulation_app = app_launcher.app
    import gymnasium as gym
    import tac_manip.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tac_manip.utils.task_configs import setup_task_objects

    pp = p6()
    p4 = pp.import_p4_probe(task_id)
    setup_task_objects(TASK_SUITE, task_id)
    cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
    cfg.episode_length_s = 45.0
    env = gym.make(ENV_ID, cfg=cfg).unwrapped
    dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
    return simulation_app, env, p4, dt


def close_isaac(simulation_app, env) -> None:
    try:
        if env is not None:
            env.close()
    except Exception:
        pass
    try:
        if simulation_app is not None:
            simulation_app.close()
    except Exception:
        pass


def run_track_a_shard() -> int:
    import torch

    task = int(os.environ["P6G0R1_TASK"])
    root_seed = int(os.environ["P6G0R1_ROOT"])
    mu = float(os.environ["P6G0R1_MU"])
    com_sign = float(os.environ["P6G0R1_COM"])
    worker_id = f"t{task}_s{root_seed}_mu{mu:g}_com{com_sign:+g}"
    shard_dir = TRACK_A / "shards" / worker_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    branch_path = shard_dir / "branches.csv"
    parity_path = shard_dir / "parity.csv"
    stage_log({"track": "A", "worker_id": worker_id, "task": task, "root_seed": root_seed, "hidden_friction": mu, "hidden_com_sign": com_sign, "stage": "worker_start"})

    simulation_app = None
    env = None
    try:
        simulation_app, env, p4, dt = setup_isaac_task(task)
        pp = p6()
        obj_name = TASK_OBJECTS[task]
        candidates = load_prev_candidates(task)
        com_offset = com_offset_for_task(task, com_sign)
        obs, _ = env.reset(seed=root_seed)
        pp.apply_friction(env, obj_name, mu)
        pp.apply_com_offset(env, obj_name, com_offset)
        state = env.scene.get_state(is_relative=True)
        ref_hash = stable_hash_obj(pp.restorable_snapshot_for_hash(env))
        ctx_id = context_id(task, root_seed, mu, com_sign)
        completed = {branch_key_from_row(r): r for r in read_csv_if(branch_path)}
        context_outcomes = {
            (r["grasp_label"], ftoken(float(r["requested_force_N"]))): as_int(r, "full_task_success_y")
            for r in completed.values()
        }
        run_forces = list(COARSE_FORCES)
        force_idx = 0
        exec_dir = TRACK_A / "P6G0R1A_EXECUTION_TELEMETRY"
        exec_dir.mkdir(parents=True, exist_ok=True)
        while force_idx < len(run_forces):
            force = float(run_forces[force_idx])
            force_idx += 1
            for cand in candidates:
                g = cand["grasp_label"]
                key = branch_key(task, root_seed, mu, com_sign, g, force)
                bid = branch_id(task, root_seed, mu, com_sign, g, force)
                if key in completed:
                    context_outcomes[(g, ftoken(force))] = as_int(completed[key], "full_task_success_y")
                    stage_log({"track": "A", "worker_id": worker_id, "task": task, "root_seed": root_seed, "hidden_friction": mu, "hidden_com_sign": com_sign, "stage": "branch_skip_completed", "grasp_label": g, "requested_force_N": force, "branch_id": bid})
                    continue
                stage_log({"track": "A", "worker_id": worker_id, "task": task, "root_seed": root_seed, "hidden_friction": mu, "hidden_com_sign": com_sign, "stage": "branch_start", "grasp_label": g, "requested_force_N": force, "branch_id": bid})
                with Timeout(TIMEOUTS_S["restore"], "P6G0R1A_RESTORE"):
                    env.reset_to(state, torch.tensor([0], device=env.device), is_relative=True)
                    pp.apply_friction(env, obj_name, mu)
                    pp.apply_com_offset(env, obj_name, com_offset)
                post_hash = stable_hash_obj(pp.restorable_snapshot_for_hash(env))
                parity = int(post_hash == ref_hash)
                parity_row = {
                    "context_id": ctx_id,
                    "task": task,
                    "root_seed": root_seed,
                    "hidden_friction": mu,
                    "hidden_com_sign": com_sign,
                    "grasp_label": g,
                    "requested_force_N": force,
                    "reference_state_hash": ref_hash,
                    "post_restore_state_hash": post_hash,
                    "parity_pass": parity,
                    "source_run": "P6G0R1A",
                }
                append_csv(parity_path, parity_row, PARITY_FIELDS)
                if not parity:
                    stage_log({"track": "A", "worker_id": worker_id, "task": task, "root_seed": root_seed, "hidden_friction": mu, "hidden_com_sign": com_sign, "stage": "branch_parity_failed", "grasp_label": g, "requested_force_N": force, "branch_id": bid})
                    continue
                center = np.array([as_float(cand, "center_obj_x"), as_float(cand, "center_obj_y"), 0.0], dtype=np.float64)
                telemetry = exec_dir / f"{bid}.csv"
                br = pp.run_full_branch(env, p4, task, center, force, dt, telemetry)
                br.update(
                    {
                        "branch_id": bid,
                        "context_id": ctx_id,
                        "task": task,
                        "task_instruction": TASK_INSTRUCTIONS[task],
                        "root_seed": root_seed,
                        "hidden_friction_analysis_only": mu,
                        "hidden_com_offset_axis_sign": com_sign,
                        "hidden_com_offset_object_frame_m": json.dumps(com_offset.tolist()),
                        "grasp_label": g,
                        "requested_force_N": force,
                        "state_parity": parity,
                        "source_run": "P6G0R1A",
                    }
                )
                append_csv(branch_path, br, BRANCH_FIELDS)
                completed[key] = br
                context_outcomes[(g, ftoken(force))] = as_int(br, "full_task_success_y")
                stage_log({"track": "A", "worker_id": worker_id, "task": task, "root_seed": root_seed, "hidden_friction": mu, "hidden_com_sign": com_sign, "stage": "branch_done", "grasp_label": g, "requested_force_N": force, "branch_id": bid, "message": f"y={br['full_task_success_y']}"})
            if force == 5.0:
                if any(context_outcomes.get((g, "3")) == 0 and context_outcomes.get((g, "5")) == 1 for g in GRASP_LABELS_A):
                    if 4.0 not in run_forces:
                        run_forces.append(4.0)
                        stage_log({"track": "A", "worker_id": worker_id, "task": task, "root_seed": root_seed, "hidden_friction": mu, "hidden_com_sign": com_sign, "stage": "refinement_added", "requested_force_N": 4.0, "message": "3 fail / 5 success"})
            if force == 8.0:
                if any(context_outcomes.get((g, "5")) == 0 and context_outcomes.get((g, "8")) == 1 for g in GRASP_LABELS_A):
                    if 6.0 not in run_forces:
                        run_forces.append(6.0)
                        stage_log({"track": "A", "worker_id": worker_id, "task": task, "root_seed": root_seed, "hidden_friction": mu, "hidden_com_sign": com_sign, "stage": "refinement_added", "requested_force_N": 6.0, "message": "5 fail / 8 success"})
        write_json(shard_dir / "result.json", {"status": "complete", "worker_id": worker_id, "branches": len(read_csv_if(branch_path))})
        stage_log({"track": "A", "worker_id": worker_id, "task": task, "root_seed": root_seed, "hidden_friction": mu, "hidden_com_sign": com_sign, "stage": "worker_done"})
        return 0
    except Exception as exc:
        write_json(shard_dir / "error.json", {"worker_id": worker_id, "error": repr(exc), "trace": traceback.format_exc()})
        stage_log({"track": "A", "worker_id": worker_id, "task": task, "root_seed": root_seed, "hidden_friction": mu, "hidden_com_sign": com_sign, "stage": "worker_error", "message": repr(exc)})
        return 1
    finally:
        close_isaac(simulation_app, env)


def quaternion_from_matrix(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = math.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def unit(v: np.ndarray, default: list[float] | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n > 1e-9:
        return v / n
    return np.asarray(default if default is not None else [0.0, 0.0, 0.0], dtype=np.float64)


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    angle = 2.0 * math.acos(float(np.clip(w, -1.0, 1.0)))
    s = math.sqrt(max(1e-12, 1.0 - w * w))
    if s < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (np.array([x, y, z], dtype=np.float32) / s) * angle


def axis_angle_from_basis(closing_b: np.ndarray, approach_up_b: np.ndarray) -> np.ndarray:
    z_axis = unit(closing_b, [0.0, 1.0, 0.0])
    x_axis = np.asarray(approach_up_b, dtype=np.float64)
    x_axis = unit(x_axis - float(np.dot(x_axis, z_axis)) * z_axis, [1.0, 0.0, 0.0])
    y_axis = unit(np.cross(z_axis, x_axis), [0.0, 1.0, 0.0])
    x_axis = unit(np.cross(y_axis, z_axis), [1.0, 0.0, 0.0])
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return quat_to_axis_angle(quaternion_from_matrix(rot))


def usd_bbox_and_mesh(asset_path: str, scale: list[float]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(asset_path)
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    box = cache.ComputeWorldBound(root).ComputeAlignedBox()
    mn = np.array(box.GetMin(), dtype=np.float64)
    mx = np.array(box.GetMax(), dtype=np.float64)
    sc = np.array(scale, dtype=np.float64)
    meshes = []
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not indices:
            continue
        mat = xcache.GetLocalToWorldTransform(prim)
        arr = []
        for pnt in pts:
            hp = mat.Transform(Gf.Vec3d(float(pnt[0]), float(pnt[1]), float(pnt[2])))
            arr.append([float(hp[0]) * sc[0], float(hp[1]) * sc[1], float(hp[2]) * sc[2]])
        arr = np.asarray(arr, dtype=np.float64)
        idx = 0
        tris = []
        for c in counts:
            face = list(indices[idx : idx + c])
            idx += c
            if len(face) < 3:
                continue
            for j in range(1, len(face) - 1):
                tris.append(arr[[face[0], face[j], face[j + 1]]])
        if tris:
            meshes.append({"prim_path": str(prim.GetPath()), "triangles": np.asarray(tris, dtype=np.float64)})
    return mn * sc, (mx - mn) * sc, meshes


def sample_surface_points(meshes: list[dict], bbox_min: np.ndarray, bbox_dims: np.ndarray, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    tris = []
    for m in meshes:
        if len(m["triangles"]):
            tris.append(m["triangles"])
    if tris:
        tri = np.concatenate(tris, axis=0)
    else:
        lo = bbox_min
        hi = bbox_min + bbox_dims
        corners = np.array(
            [
                [lo[0], lo[1], lo[2]],
                [hi[0], lo[1], lo[2]],
                [hi[0], hi[1], lo[2]],
                [lo[0], hi[1], lo[2]],
                [lo[0], lo[1], hi[2]],
                [hi[0], lo[1], hi[2]],
                [hi[0], hi[1], hi[2]],
                [lo[0], hi[1], hi[2]],
            ],
            dtype=np.float64,
        )
        faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
        tri = []
        for a, b, c, d in faces:
            tri.append(corners[[a, b, c]])
            tri.append(corners[[a, c, d]])
        tri = np.asarray(tri, dtype=np.float64)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(normals, axis=1)
    valid = areas > 1e-12
    tri = tri[valid]
    normals = normals[valid]
    areas = areas[valid]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    probs = areas / max(float(np.sum(areas)), 1e-12)
    idx = rng.choice(np.arange(len(tri)), size=n, replace=True, p=probs)
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    pts = (1 - r1)[:, None] * tri[idx, 0] + (r1 * (1 - r2))[:, None] * tri[idx, 1] + (r1 * r2)[:, None] * tri[idx, 2]
    ns = normals[idx]
    return {"points": pts, "normals": ns, "mesh_triangle_count": int(len(tri))}


def nearest_sample(points: np.ndarray, normals: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    d = np.linalg.norm(points - target[None, :], axis=1)
    i = int(np.argmin(d))
    return points[i], normals[i], float(d[i])


def geometry_candidates(task: int, env, p4) -> tuple[list[dict], list[dict], dict, dict]:
    pp = p6()
    cfg = pp.load_task_config(task)
    obj_name = TASK_OBJECTS[task]
    obj_type = cfg["objects"][obj_name]["type"]
    scale = cfg["objects"][obj_name]["scale"]
    asset_path = str(REPO / "benchmarks/datasets/libero/USD" / obj_type / f"{obj_type}.usd")
    bbox_min, bbox_dims, meshes = usd_bbox_and_mesh(asset_path, scale)
    surf = sample_surface_points(meshes, bbox_min, bbox_dims, SURFACE_SAMPLE_COUNT, SURFACE_SAMPLE_SEED + task)
    pts = surf["points"]
    ns = surf["normals"]
    extent = float(max(bbox_dims))
    table_normal_obj = unit(pp.vector_base_to_object(p4, env, obj_name, np.array([0.0, 0.0, 1.0])), [0.0, 0.0, 1.0])
    vertical_axis = int(np.argmax(np.abs(table_normal_obj)))
    horizontal_axes = [i for i in range(3) if i != vertical_axis]
    if len(horizontal_axes) < 2:
        horizontal_axes = [0, 1]
    span_order = sorted(horizontal_axes, key=lambda i: float(bbox_dims[i]), reverse=True)
    close_order = sorted(horizontal_axes, key=lambda i: float(bbox_dims[i]))
    raw = []
    cand_idx = 0
    tol_cos = math.cos(math.radians(ANTIPODAL_TOL_DEG))
    z_axis = vertical_axis
    z_levels = [0.42, 0.58, 0.74]
    offsets = [-0.28, 0.0, 0.28]
    for close_axis in close_order:
        span_axis = next((a for a in span_order if a != close_axis), span_order[0])
        width = float(bbox_dims[close_axis])
        for z_frac in z_levels:
            for off_frac in offsets:
                center = bbox_min + 0.5 * bbox_dims
                center[span_axis] = bbox_min[span_axis] + (0.5 + off_frac) * bbox_dims[span_axis]
                center[z_axis] = bbox_min[z_axis] + z_frac * bbox_dims[z_axis]
                axis = np.zeros(3, dtype=np.float64)
                axis[close_axis] = 1.0
                p_l_theory = center - 0.5 * width * axis
                p_r_theory = center + 0.5 * width * axis
                p_l, n_l, dl = nearest_sample(pts, ns, p_l_theory)
                p_r, n_r, dr = nearest_sample(pts, ns, p_r_theory)
                closing = unit(p_r - p_l, axis)
                actual_width = float(np.linalg.norm(p_r - p_l))
                anti_l = float(np.dot(n_l, -closing))
                anti_r = float(np.dot(n_r, closing))
                edge_margin = min(
                    float(np.min((center - bbox_min) / np.maximum(bbox_dims, 1e-9))),
                    float(np.min((bbox_min + bbox_dims - center) / np.maximum(bbox_dims, 1e-9))),
                )
                table_clearance = float(np.dot(center - bbox_min, table_normal_obj))
                rejection = []
                if actual_width < GRIPPER_WIDTH_MIN_M or actual_width > GRIPPER_WIDTH_MAX_M:
                    rejection.append("width_outside_gripper_limits")
                if anti_l < tol_cos or anti_r < tol_cos:
                    rejection.append("antipodal_normal_tolerance")
                if table_clearance < 0.015:
                    rejection.append("insufficient_table_clearance")
                if edge_margin < 0.10:
                    rejection.append("insufficient_edge_margin")
                if abs(float(np.dot(closing, table_normal_obj))) > 0.45:
                    rejection.append("closing_axis_too_vertical")
                score = (
                    2.0 * max(0.0, anti_l)
                    + 2.0 * max(0.0, anti_r)
                    + 1.5 * min(edge_margin, 0.5)
                    + 0.6 * min(table_clearance / max(extent, 1e-9), 1.0)
                    - 0.4 * (dl + dr) / max(extent, 1e-9)
                    - 0.2 * abs(actual_width - 0.055) / 0.055
                )
                raw.append(
                    {
                        "task": task,
                        "object": obj_name,
                        "raw_candidate_id": f"raw_{cand_idx:04d}",
                        "valid": int(not rejection),
                        "rejection_reason": ";".join(rejection),
                        "score": score,
                        "center_obj_x": float(center[0]),
                        "center_obj_y": float(center[1]),
                        "center_obj_z": float(center[2]),
                        "left_contact_obj_x": float(p_l[0]),
                        "left_contact_obj_y": float(p_l[1]),
                        "left_contact_obj_z": float(p_l[2]),
                        "right_contact_obj_x": float(p_r[0]),
                        "right_contact_obj_y": float(p_r[1]),
                        "right_contact_obj_z": float(p_r[2]),
                        "left_normal_obj": json.dumps(n_l.tolist()),
                        "right_normal_obj": json.dumps(n_r.tolist()),
                        "closing_dir_obj": json.dumps(closing.tolist()),
                        "approach_dir_obj": json.dumps(table_normal_obj.tolist()),
                        "wrist_roll_deg": 0.0,
                        "grasp_depth_m": GRASP_APPROACH_OFFSET_M,
                        "gripper_width_m": actual_width,
                        "antipodal_left": anti_l,
                        "antipodal_right": anti_r,
                        "edge_margin_proxy": edge_margin,
                        "table_clearance_m": table_clearance,
                        "nearest_surface_error_m": dl + dr,
                    }
                )
                cand_idx += 1
    if sum(1 for r in raw if int(r["valid"]) == 1) < 3:
        for close_axis in close_order:
            span_axis = next((a for a in span_order if a != close_axis), span_order[0])
            width = float(bbox_dims[close_axis])
            axis = np.zeros(3, dtype=np.float64)
            axis[close_axis] = 1.0
            for off_frac in offsets:
                center = bbox_min + 0.5 * bbox_dims
                center[span_axis] = bbox_min[span_axis] + (0.5 + off_frac) * bbox_dims[span_axis]
                center[z_axis] = bbox_min[z_axis] + 0.60 * bbox_dims[z_axis]
                p_l = center - 0.5 * width * axis
                p_r = center + 0.5 * width * axis
                closing = axis.copy()
                edge_margin = min(
                    float(np.min((center - bbox_min) / np.maximum(bbox_dims, 1e-9))),
                    float(np.min((bbox_min + bbox_dims - center) / np.maximum(bbox_dims, 1e-9))),
                )
                table_clearance = float(np.dot(center - bbox_min, table_normal_obj))
                rejection = []
                if width < GRIPPER_WIDTH_MIN_M or width > GRIPPER_WIDTH_MAX_M:
                    rejection.append("width_outside_gripper_limits")
                if table_clearance < 0.015:
                    rejection.append("insufficient_table_clearance")
                if edge_margin < 0.10:
                    rejection.append("insufficient_edge_margin")
                score = 4.0 + 1.5 * min(edge_margin, 0.5) + 0.6 * min(table_clearance / max(extent, 1e-9), 1.0) - 0.2 * abs(width - 0.055) / 0.055
                raw.append(
                    {
                        "task": task,
                        "object": obj_name,
                        "raw_candidate_id": f"bbox_proxy_{cand_idx:04d}",
                        "valid": int(not rejection),
                        "rejection_reason": ";".join(rejection),
                        "score": score,
                        "center_obj_x": float(center[0]),
                        "center_obj_y": float(center[1]),
                        "center_obj_z": float(center[2]),
                        "left_contact_obj_x": float(p_l[0]),
                        "left_contact_obj_y": float(p_l[1]),
                        "left_contact_obj_z": float(p_l[2]),
                        "right_contact_obj_x": float(p_r[0]),
                        "right_contact_obj_y": float(p_r[1]),
                        "right_contact_obj_z": float(p_r[2]),
                        "left_normal_obj": json.dumps((-axis).tolist()),
                        "right_normal_obj": json.dumps(axis.tolist()),
                        "closing_dir_obj": json.dumps(closing.tolist()),
                        "approach_dir_obj": json.dumps(table_normal_obj.tolist()),
                        "wrist_roll_deg": 0.0,
                        "grasp_depth_m": GRASP_APPROACH_OFFSET_M,
                        "gripper_width_m": width,
                        "antipodal_left": 1.0,
                        "antipodal_right": 1.0,
                        "edge_margin_proxy": edge_margin,
                        "table_clearance_m": table_clearance,
                        "nearest_surface_error_m": 0.0,
                    }
                )
                cand_idx += 1
    valid = [r for r in raw if int(r["valid"]) == 1]
    valid.sort(key=lambda r: (-float(r["score"]), r["center_obj_x"], r["center_obj_y"], r["center_obj_z"], r["closing_dir_obj"]))
    quality = valid[: max(3, min(len(valid), 18))]
    selected = []
    if quality:
        selected.append(quality[0])
        while len(selected) < min(3, len(quality)):
            best = None
            best_d = -1.0
            for r in quality:
                if r in selected:
                    continue
                feat = candidate_feature(r, extent)
                d = min(float(np.linalg.norm(feat - candidate_feature(s, extent))) for s in selected)
                key = (d, float(r["score"]))
                if d > best_d or (abs(d - best_d) < 1e-12 and best is not None and float(r["score"]) > float(best["score"])):
                    best = r
                    best_d = d
            if best is None:
                break
            selected.append(best)
    selected.sort(key=lambda r: (float(r["center_obj_x"]), float(r["center_obj_y"]), float(r["center_obj_z"]), r["closing_dir_obj"]))
    selected_rows = []
    for i, r in enumerate(selected[:3]):
        out = dict(r)
        out["candidate_id"] = f"G{i}"
        out["selection_rank"] = i
        out["human_readable_label"] = "geometry_medoid"
        selected_rows.append(out)
    audit = {
        "task": task,
        "object": obj_name,
        "mesh_asset_path": asset_path,
        "collision_mesh": "USD collision/proxy mesh when authored; visual mesh fallback otherwise",
        "visual_mesh": "USD render/default mesh",
        "object_frame": "USD rigid actor frame; candidates expressed in object frame",
        "bbox_min_object_frame_m": json.dumps(bbox_min.tolist()),
        "bbox_dims_m": json.dumps(bbox_dims.tolist()),
        "surface_sample_count": SURFACE_SAMPLE_COUNT,
        "mesh_triangle_count": surf["mesh_triangle_count"],
        "surface_normals": "sampled triangle normals, fallback bbox normals if no mesh triangles",
        "table_relative_pose": json.dumps(p4._pose_in_base(env, obj_name)[0].tolist()),
        "gripper_width_limits_m": json.dumps([GRIPPER_WIDTH_MIN_M, GRIPPER_WIDTH_MAX_M]),
        "finger_geometry": "parallel-jaw tactile gripper; action width proxy D_OPEN/D_CLOSED reused from P4/P6",
        "nominal_approach_direction": "base +z pregrasp, descend toward object along table normal",
        "antipodal_tolerance_deg": ANTIPODAL_TOL_DEG,
    }
    rule = {
        "surface_samples": SURFACE_SAMPLE_COUNT,
        "sampling_seed_base": SURFACE_SAMPLE_SEED,
        "antipodal_tolerance_deg": ANTIPODAL_TOL_DEG,
        "roll_samples_deg": ROLL_SAMPLES_DEG,
        "gripper_width_limits_m": [GRIPPER_WIDTH_MIN_M, GRIPPER_WIDTH_MAX_M],
        "selection": "quality-filtered deterministic farthest-point selection over object-frame center, closing direction, approach direction, roll, depth, width",
        "task_object_lookup_used": False,
        "hidden_physics_used": False,
        "full_task_outcomes_used": False,
    }
    return raw, selected_rows, audit, rule


def candidate_feature(row: dict, extent: float) -> np.ndarray:
    center = np.array([as_float(row, "center_obj_x"), as_float(row, "center_obj_y"), as_float(row, "center_obj_z")], dtype=np.float64) / max(extent, 1e-9)
    closing = np.array(json.loads(row["closing_dir_obj"]), dtype=np.float64)
    approach = np.array(json.loads(row["approach_dir_obj"]), dtype=np.float64)
    return np.concatenate([center, closing, approach, [as_float(row, "wrist_roll_deg") / 180.0, as_float(row, "grasp_depth_m") / max(extent, 1e-9), as_float(row, "gripper_width_m") / max(extent, 1e-9)]])


B_RESULT_FIELDS = [
    "task",
    "root_seed",
    "candidate_id",
    "state_hash",
    "requested_center_obj_x",
    "requested_center_obj_y",
    "requested_center_obj_z",
    "requested_left_contact_obj",
    "requested_right_contact_obj",
    "requested_closing_dir_obj",
    "requested_approach_dir_obj",
    "requested_wrist_roll_deg",
    "grasp_depth_m",
    "gripper_width_m",
    "actual_contact_mid_obj_x",
    "actual_contact_mid_obj_y",
    "actual_contact_mid_obj_z",
    "actual_left_tip_obj",
    "actual_right_tip_obj",
    "actual_wrist_pose",
    "bilateral_contact",
    "object_retained",
    "short_lift_success",
    "collision",
    "table_contact",
    "ik_failure",
    "dropped",
    "steps",
]


def run_track_b_attempt(env, p4, task: int, cand: dict, force: float, dt: float) -> dict:
    pp = p6()
    obj_name = TASK_OBJECTS[task]
    center_obj = np.array([as_float(cand, "center_obj_x"), as_float(cand, "center_obj_y"), as_float(cand, "center_obj_z")], dtype=np.float64)
    closing_o = np.array(json.loads(cand["closing_dir_obj"]), dtype=np.float64)
    closing_b = pp.vector_object_to_base(p4, env, obj_name, closing_o)
    approach_b = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    eef_aa = axis_angle_from_basis(closing_b, approach_b)
    grasp = pp.object_point_to_base(p4, env, obj_name, center_obj)
    pregrasp = grasp + PREGRASP_OFFSET_M * approach_b
    lift = grasp + 0.08 * approach_b
    obs = env.observation_manager.compute()
    eef0 = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    cmd_pos = eef0[:3].copy()
    d_pred = D_OPEN
    phases = [("approach", APPROACH_STEPS, pregrasp, 0.0), ("descend", DESCEND_STEPS, grasp, 0.0), ("close", CLOSE_STEPS, grasp, force), ("hold", HOLD_STEPS, grasp, force), ("lift", LIFT_STEPS, lift, force)]
    local_contact = 0
    retained = 0
    lift_success = 0
    dropped = 0
    ik_failure = 0
    rows = []
    obj0_b, _ = p4._pose_in_base(env, obj_name)
    step = 0
    max_eef_error = 0.0
    for phase, n_steps, target, f_cmd in phases:
        start = cmd_pos.copy()
        for i in range(n_steps):
            cmd_pos = p4._interp(start, target, i, n_steps)
            if phase in {"approach", "descend"}:
                d_pred = D_OPEN
            action = pp.make_action(p4, cmd_pos, eef_aa, d_pred, f_cmd, env.device)
            obs, _, term, trunc, _ = env.step(action)
            step += 1
            f_sq = p4._f(p4._dbg(env).get("f_sq_meas"), 0.0)
            if f_cmd > 0:
                d_pred = pp.force_servo(p4, d_pred, f_sq, force)
            f = obs["policy"]["gripper_net_force"][0]
            if f.ndim == 3:
                f = f[-1]
            f_np = f.detach().cpu().numpy()
            left_norm = float(np.linalg.norm(f_np[0]))
            right_norm = float(np.linalg.norm(f_np[1]))
            contact = "bilateral" if left_norm > 0.15 and right_norm > 0.15 else ("unilateral" if left_norm > 0.15 or right_norm > 0.15 else "none")
            if contact == "bilateral":
                local_contact = 1
            obj_b, obj_q = p4._pose_in_base(env, obj_name)
            dz = float(obj_b[2] - obj0_b[2])
            if phase in {"hold", "lift"} and contact == "bilateral":
                retained = 1
            if dz >= 0.03 and contact == "bilateral":
                lift_success = 1
            try:
                dropped = max(dropped, int(bool(env.termination_manager.get_term("object_1_dropped")[0].item())))
            except Exception:
                pass
            ee_now = obs["policy"]["eef_pose"][0].detach().cpu().numpy()[:3]
            max_eef_error = max(max_eef_error, float(np.linalg.norm(ee_now - cmd_pos)))
            left_p = env.scene["left_gripper_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
            right_p = env.scene["right_gripper_frame"].data.target_pos_w[0, 0].detach().cpu().numpy()
            mid_b = pp.world_point_to_base(p4, env, 0.5 * (left_p + right_p))
            mid_o = pp.base_point_to_object(p4, env, obj_name, mid_b)
            rows.append((phase, mid_o, left_p, right_p, obj_b, obj_q))
            if bool(term[0].item()) or bool(trunc[0].item()):
                break
        else:
            continue
        break
    if max_eef_error > 0.065:
        ik_failure = 1
    mids = [r[1] for r in rows if r[0] in {"hold", "lift"}]
    mid = np.nanmean(np.asarray(mids, dtype=np.float64), axis=0) if mids else np.full(3, np.nan)
    left_w = rows[-1][2] if rows else np.full(3, np.nan)
    right_w = rows[-1][3] if rows else np.full(3, np.nan)
    left_o = pp.base_point_to_object(p4, env, obj_name, pp.world_point_to_base(p4, env, left_w)) if np.isfinite(left_w).all() else np.full(3, np.nan)
    right_o = pp.base_point_to_object(p4, env, obj_name, pp.world_point_to_base(p4, env, right_w)) if np.isfinite(right_w).all() else np.full(3, np.nan)
    actual_pose = obs["policy"]["eef_pose"][0].detach().cpu().numpy().tolist()
    return {
        "actual_contact_mid_obj_x": float(mid[0]),
        "actual_contact_mid_obj_y": float(mid[1]),
        "actual_contact_mid_obj_z": float(mid[2]),
        "actual_left_tip_obj": json.dumps(left_o.tolist()),
        "actual_right_tip_obj": json.dumps(right_o.tolist()),
        "actual_wrist_pose": json.dumps(actual_pose),
        "bilateral_contact": int(local_contact == 1),
        "object_retained": int(retained == 1 and dropped == 0),
        "short_lift_success": int(lift_success == 1 and dropped == 0),
        "collision": "",
        "table_contact": int(dropped == 1),
        "ik_failure": ik_failure,
        "dropped": dropped,
        "steps": step,
    }


def run_track_b_worker() -> int:
    import torch

    task = int(os.environ["P6G0R1_TASK"])
    task_dir = TRACK_B / f"task{task}"
    task_dir.mkdir(parents=True, exist_ok=True)
    simulation_app = None
    env = None
    try:
        simulation_app, env, p4, dt = setup_isaac_task(task)
        pp = p6()
        obj_name = TASK_OBJECTS[task]
        obs, _ = env.reset(seed=42000 + task)
        raw, selected, audit, rule = geometry_candidates(task, env, p4)
        write_csv(task_dir / "raw_candidates.csv", raw)
        write_csv(task_dir / "selected_candidates.csv", selected)
        write_csv(task_dir / "geometry_audit.csv", [audit])
        write_json(task_dir / "selection_rule.json", rule)
        manifest = []
        results = []
        if task in TRACK_B_TARGET_TASKS:
            for root_seed in PREFLIGHT_B_ROOTS:
                obs, _ = env.reset(seed=root_seed)
                pp.apply_friction(env, obj_name, TRACK_B_FRICTION)
                pp.apply_com_offset(env, obj_name, np.zeros(3))
                state = env.scene.get_state(is_relative=True)
                ref_hash = stable_hash_obj(pp.restorable_snapshot_for_hash(env))
                for cand in selected:
                    manifest.append({"task": task, "root_seed": root_seed, "candidate_id": cand["candidate_id"], "force_N": TRACK_B_FORCE_N, "friction": TRACK_B_FRICTION, "centered_com": True})
                    with Timeout(TIMEOUTS_S["restore"], "P6G0R1B_RESTORE"):
                        env.reset_to(state, torch.tensor([0], device=env.device), is_relative=True)
                        pp.apply_friction(env, obj_name, TRACK_B_FRICTION)
                        pp.apply_com_offset(env, obj_name, np.zeros(3))
                    rec = run_track_b_attempt(env, p4, task, cand, TRACK_B_FORCE_N, dt)
                    row = {
                        "task": task,
                        "root_seed": root_seed,
                        "candidate_id": cand["candidate_id"],
                        "state_hash": ref_hash,
                        "requested_center_obj_x": cand["center_obj_x"],
                        "requested_center_obj_y": cand["center_obj_y"],
                        "requested_center_obj_z": cand["center_obj_z"],
                        "requested_left_contact_obj": json.dumps([cand["left_contact_obj_x"], cand["left_contact_obj_y"], cand["left_contact_obj_z"]]),
                        "requested_right_contact_obj": json.dumps([cand["right_contact_obj_x"], cand["right_contact_obj_y"], cand["right_contact_obj_z"]]),
                        "requested_closing_dir_obj": cand["closing_dir_obj"],
                        "requested_approach_dir_obj": cand["approach_dir_obj"],
                        "requested_wrist_roll_deg": cand["wrist_roll_deg"],
                        "grasp_depth_m": cand["grasp_depth_m"],
                        "gripper_width_m": cand["gripper_width_m"],
                    }
                    row.update(rec)
                    results.append(row)
                    append_csv(task_dir / "realization_results.csv", row, B_RESULT_FIELDS)
        write_csv(task_dir / "preflight_manifest.csv", manifest)
        write_json(task_dir / "result.json", {"task": task, "raw_candidates": len(raw), "selected_candidates": len(selected), "realization_attempts": len(results), "status": "complete"})
        return 0
    except Exception as exc:
        write_json(task_dir / "error.json", {"task": task, "error": repr(exc), "trace": traceback.format_exc()})
        return 1
    finally:
        close_isaac(simulation_app, env)


def launch_worker(mode: str, extra: dict[str, str], timeout_s: float, log_name: str) -> dict:
    log_path = OUT / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ISAAC_PY), "-u", str(Path(__file__).resolve())]
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=worker_env(extra), stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            returncode = -9
    row = {"mode": mode, "returncode": returncode, "elapsed_wall_s": time.time() - start, "log": str(log_path)}
    row.update(extra)
    return row


WORKER_FIELDS = ["mode", "returncode", "elapsed_wall_s", "log", "P6G0R1_TRACK_A_SHARD", "P6G0R1_TRACK_B_WORKER", "P6G0R1_TASK", "P6G0R1_ROOT", "P6G0R1_MU", "P6G0R1_COM"]


def compare_planned_completed(planned: list[dict], completed: list[dict]) -> tuple[list[dict], list[dict]]:
    completed_keys = {branch_key_from_row(r) for r in completed}
    already = []
    missing = []
    for row in planned:
        if branch_key_from_row(row) in completed_keys:
            already.append(row)
        else:
            missing.append(row)
    return already, missing


def track_a_shards_to_run(missing: list[dict]) -> list[tuple[int, int, float, float]]:
    contexts = sorted(
        {
            (int(float(r["task"])), int(float(r["root_seed"])), float(r["hidden_friction_analysis_only"]), float(r["hidden_com_offset_axis_sign"]))
            for r in missing
        }
    )
    return contexts


def collect_track_a_r1_rows() -> tuple[list[dict], list[dict]]:
    branches = []
    parity = []
    for shard in sorted((TRACK_A / "shards").glob("*/branches.csv")):
        branches += read_csv_if(shard)
    for shard in sorted((TRACK_A / "shards").glob("*/parity.csv")):
        parity += read_csv_if(shard)
    return branches, parity


def dedupe_branches(rows: list[dict]) -> tuple[list[dict], int]:
    out = {}
    dup = 0
    for r in rows:
        k = branch_key_from_row(r)
        if k in out:
            dup += 1
            if str(r.get("source_run")) == "P6G0R1A":
                out[k] = r
        else:
            out[k] = r
    return list(out.values()), dup


def analyze_track_a(branches: list[dict]) -> dict:
    force_effect_rows = []
    grasp_effect_rows = []
    frontiers = []
    divergence = []
    flips = []
    pareto = []
    groups: dict[tuple, list[dict]] = {}
    for b in branches:
        key = (b["task"], b["root_seed"], b["hidden_friction_analysis_only"], b["hidden_com_offset_axis_sign"])
        groups.setdefault(key, []).append(b)
        if as_int(b, "lift_success") == 1 and as_int(b, "full_task_success_y") == 0:
            divergence.append(b)
    for key, rows in groups.items():
        by_g: dict[str, list[dict]] = {}
        by_f: dict[float, list[dict]] = {}
        for r in rows:
            by_g.setdefault(r["grasp_label"], []).append(r)
            by_f.setdefault(as_float(r, "requested_force_N"), []).append(r)
        for g, rs in by_g.items():
            ys = {as_float(r, "requested_force_N"): as_int(r, "full_task_success_y") for r in rs}
            if len(set(ys.values())) > 1:
                force_effect_rows.append({"task": key[0], "root_seed": key[1], "hidden_friction": key[2], "hidden_com": key[3], "grasp_label": g, "force_outcomes": json.dumps(ys, sort_keys=True)})
            succ = sorted(f for f, y in ys.items() if y == 1)
            frontiers.append({"task": key[0], "root_seed": key[1], "hidden_friction": key[2], "hidden_com": key[3], "grasp_label": g, "F_grid_star": succ[0] if succ else ""})
        for f, rs in by_f.items():
            ys = {r["grasp_label"]: as_int(r, "full_task_success_y") for r in rs}
            if len(set(ys.values())) > 1:
                grasp_effect_rows.append({"task": key[0], "root_seed": key[1], "hidden_friction": key[2], "hidden_com": key[3], "requested_force_N": f, "grasp_outcomes": json.dumps(ys, sort_keys=True)})
        utilities = []
        for r in rows:
            y = as_int(r, "full_task_success_y")
            f = as_float(r, "requested_force_N")
            u = y * (F_MAX - f) + (1 - y) * (-F_MAX)
            utilities.append((u, r["grasp_label"], f, y))
        if utilities:
            best = sorted(utilities, reverse=True)[0]
            pareto_candidates = []
            for u, g, f, y in utilities:
                dominated = any((y2 >= y and f2 <= f and (y2 > y or f2 < f)) for _, _, f2, y2 in utilities)
                if not dominated:
                    pareto_candidates.append({"grasp": g, "force": f, "success": y})
            pareto.append({"task": key[0], "root_seed": key[1], "hidden_friction": key[2], "hidden_com": key[3], "best_grasp": best[1], "best_force": best[2], "best_utility": best[0], "pareto": json.dumps(pareto_candidates, sort_keys=True)})
    best_by_context = {(r["task"], r["root_seed"], r["hidden_friction"], r["hidden_com"]): (r["best_grasp"], str(r["best_force"])) for r in pareto}
    for task in sorted({k[0] for k in best_by_context}):
        for root in sorted({k[1] for k in best_by_context if k[0] == task}):
            vals = {k: v for k, v in best_by_context.items() if k[0] == task and k[1] == root}
            if len(set(vals.values())) > 1:
                flips.append({"task": task, "root_seed": root, "num_distinct_best_actions": len(set(vals.values())), "best_actions_by_hidden_world": json.dumps({str(k[2:]): v for k, v in vals.items()}, sort_keys=True)})
    return {
        "force": force_effect_rows,
        "grasp": grasp_effect_rows,
        "frontiers": frontiers,
        "divergence": divergence,
        "flips": flips,
        "pareto": pareto,
    }


def classify_track_a(planned: list[dict], completed: list[dict], duplicate_count: int, worker_rows: list[dict], analysis: dict) -> dict:
    _, missing = compare_planned_completed(planned, completed)
    matrix_complete = len(missing) == 0
    timed_out = [r for r in worker_rows if int(r["returncode"]) == -9]
    classes = {}
    for task in TRACK_A_TASKS:
        task_s = str(task)
        roots = sorted({r["root_seed"] for r in completed if str(r["task"]) == task_s})
        force_roots = sorted({r["root_seed"] for r in analysis["force"] if str(r["task"]) == task_s})
        grasp_roots = sorted({r["root_seed"] for r in analysis["grasp"] if str(r["task"]) == task_s})
        flip_roots = sorted({r["root_seed"] for r in analysis["flips"] if str(r["task"]) == task_s})
        task_missing = [r for r in missing if int(float(r["task"])) == task]
        if task_missing:
            cls = "P6G0R1A_CONFIRMATORY_SWEEP_NOT_COMPLETE"
        elif force_roots and grasp_roots and flip_roots and len(flip_roots) >= 2:
            cls = "P6G0R1A_TASK_EFFECTS_REPLICATED_MULTIPLE_ROOTS"
        elif force_roots and grasp_roots and flip_roots:
            cls = "P6G0R1A_TASK_EFFECTS_REPLICATED_SINGLE_ROOT"
        elif force_roots or grasp_roots:
            cls = "P6G0R1A_TASK_EFFECTS_WEAKENED"
        else:
            cls = "P6G0R1A_TASK_NO_ROBUST_ACTION_FLIP"
        classes[task] = {
            "task": task,
            "matrix_complete": not task_missing,
            "force_effect": bool(force_roots),
            "force_effect_roots": force_roots,
            "grasp_position_effect": bool(grasp_roots),
            "grasp_position_effect_roots": grasp_roots,
            "lift_full_divergence": any(str(r["task"]) == task_s for r in analysis["divergence"]),
            "action_ranking_flips": bool(flip_roots),
            "flip_roots": flip_roots,
            "classification": cls,
        }
    if not matrix_complete:
        verdict = "P6G0R1A_CONFIRMATORY_SWEEP_NOT_COMPLETE"
    elif any(int(r["returncode"]) not in {0} for r in worker_rows) and not completed:
        verdict = "P6G0R1A_INCONCLUSIVE_SYSTEM_FAILURE"
    elif all(classes[t]["force_effect"] and classes[t]["grasp_position_effect"] and classes[t]["action_ranking_flips"] and len(classes[t]["flip_roots"]) >= 2 for t in TRACK_A_TASKS):
        verdict = "P6G0R1A_CONFIRMATORY_SWEEP_COMPLETE_EFFECTS_REPLICATED"
    elif any(classes[t]["action_ranking_flips"] for t in TRACK_A_TASKS):
        verdict = "P6G0R1A_CONFIRMATORY_SWEEP_COMPLETE_EFFECTS_WEAKENED"
    else:
        verdict = "P6G0R1A_CONFIRMATORY_SWEEP_COMPLETE_NO_ROBUST_ACTION_FLIP"
    return {
        "verdict": verdict,
        "matrix_complete": matrix_complete,
        "missing_count": len(missing),
        "duplicate_rows": duplicate_count,
        "timeouts": len(timed_out),
        "task_classes": classes,
    }


def aggregate_track_b(worker_rows: list[dict]) -> dict:
    audits = []
    raw = []
    selected = []
    manifests = []
    results = []
    rules = []
    for task in TRACK_B_TARGET_TASKS + TRACK_B_REFERENCE_TASKS:
        td = TRACK_B / f"task{task}"
        audits += read_csv_if(td / "geometry_audit.csv")
        raw += read_csv_if(td / "raw_candidates.csv")
        selected += read_csv_if(td / "selected_candidates.csv")
        manifests += read_csv_if(td / "preflight_manifest.csv")
        results += read_csv_if(td / "realization_results.csv")
        if (td / "selection_rule.json").exists():
            rules.append(json.loads((td / "selection_rule.json").read_text(encoding="utf-8")))
    write_csv(TRACK_B / "P6G0R1B_OBJECT_GEOMETRY_AUDIT.csv", audits)
    write_csv(TRACK_B / "P6G0R1B_RAW_GRASP_CANDIDATES.csv", raw)
    write_csv(TRACK_B / "P6G0R1B_SELECTED_GRASP_CANDIDATES.csv", selected)
    write_json(TRACK_B / "P6G0R1B_CANDIDATE_SELECTION_RULE.json", {"global_rule": rules[0] if rules else {}, "per_task_rules_identical": True})
    (TRACK_B / "P6G0R1B_CANDIDATE_SELECTION_RULE.md").write_text(
        "# P6-G0-R1B candidate selection rule\n\n"
        "The generator samples object surfaces, proposes antipodal parallel-jaw contact pairs with one global angular tolerance, "
        "filters by gripper width, table clearance, edge margin, and vertical-closing rejection, scores only geometry/kinematics proxies, "
        "then selects up to three diverse representatives with deterministic farthest-point selection in object-frame grasp-feature space.\n",
        encoding="utf-8",
    )
    write_csv(TRACK_B / "P6G0R1B_PREFLIGHT_MANIFEST.csv", manifests)
    write_csv(TRACK_B / "P6G0R1B_REALIZATION_RESULTS.csv", results, B_RESULT_FIELDS)
    cluster_rows = []
    task_classes = []
    for task in TRACK_B_TARGET_TASKS:
        task_results = [r for r in results if int(float(r["task"])) == task]
        extent = max(json.loads(next((a["bbox_dims_m"] for a in audits if int(float(a["task"])) == task), "[1,1,1]")))
        qualified_candidates = []
        means = {}
        for cid in GRASP_LABELS_B:
            rs = [r for r in task_results if r["candidate_id"] == cid]
            pts = np.array(
                [[as_float(r, "actual_contact_mid_obj_x", np.nan), as_float(r, "actual_contact_mid_obj_y", np.nan), as_float(r, "actual_contact_mid_obj_z", np.nan)] for r in rs if as_int(r, "bilateral_contact") and as_int(r, "short_lift_success")],
                dtype=np.float64,
            )
            mean = np.nanmean(pts, axis=0) if len(pts) else np.full(3, np.nan)
            means[cid] = mean
            within = float(np.nanmean(np.linalg.norm(pts - mean, axis=1)) / max(extent, 1e-9)) if len(pts) else math.nan
            ik_rate = sum(1 - as_int(r, "ik_failure") for r in rs) / max(1, len(rs))
            bilat_rate = sum(as_int(r, "bilateral_contact") for r in rs) / max(1, len(rs))
            lift_rate = sum(as_int(r, "short_lift_success") for r in rs) / max(1, len(rs))
            qualified = bool(ik_rate >= 0.8 and bilat_rate >= 0.8 and lift_rate >= 0.8 and (math.isnan(within) or within <= 0.08))
            if qualified:
                qualified_candidates.append(cid)
            cluster_rows.append(
                {
                    "task": task,
                    "candidate_id": cid,
                    "n": len(rs),
                    "ik_success_rate": ik_rate,
                    "bilateral_contact_rate": bilat_rate,
                    "short_lift_success_rate": lift_rate,
                    "within_candidate_contact_std_norm_extent": within,
                    "actual_contact_mean_obj": json.dumps(mean.tolist()),
                    "individual_qualified": qualified,
                }
            )
        sep_vals = []
        for i, ca in enumerate(GRASP_LABELS_B):
            for cb in GRASP_LABELS_B[i + 1 :]:
                if np.isfinite(means[ca]).all() and np.isfinite(means[cb]).all():
                    sep_vals.append(float(np.linalg.norm(means[ca] - means[cb]) / max(extent, 1e-9)))
        min_sep = min(sep_vals) if sep_vals else 0.0
        for row in cluster_rows:
            if int(row["task"]) == task:
                row["min_between_candidate_separation_norm_extent"] = min_sep
        if len(qualified_candidates) >= 3 and min_sep >= 0.15:
            cls = "THREE_MODE_GEOMETRY_CANDIDATES_QUALIFIED"
        elif len(qualified_candidates) >= 2 and min_sep >= 0.15:
            cls = "TWO_MODE_GEOMETRY_CANDIDATES_PARTIALLY_QUALIFIED"
        elif len(qualified_candidates) >= 2:
            cls = "GEOMETRY_CANDIDATES_REACHABLE_BUT_NOT_DISTINCT"
        elif min_sep >= 0.15 and any(as_int(r, "bilateral_contact") for r in task_results):
            cls = "GEOMETRY_CANDIDATES_DISTINCT_BUT_NOT_STABLE"
        else:
            cls = "GEOMETRY_AWARE_GENERATOR_NOT_COVERING"
        task_classes.append(
            {
                "task": task,
                "raw_valid_candidates": sum(1 for r in raw if int(float(r["task"])) == task and as_int(r, "valid") == 1),
                "selected_candidates": sum(1 for r in selected if int(float(r["task"])) == task),
                "qualified_candidates": len(qualified_candidates),
                "min_between_candidate_separation_norm_extent": min_sep,
                "realization_rate": sum(as_int(r, "bilateral_contact") for r in task_results) / max(1, len(task_results)),
                "short_lift_rate": sum(as_int(r, "short_lift_success") for r in task_results) / max(1, len(task_results)),
                "classification": cls,
            }
        )
    write_csv(TRACK_B / "P6G0R1B_ACTUAL_CONTACT_CLUSTERS.csv", cluster_rows)
    write_csv(TRACK_B / "P6G0R1B_TASK_CLASSIFICATIONS.csv", task_classes)
    three = sum(1 for r in task_classes if r["classification"] == "THREE_MODE_GEOMETRY_CANDIDATES_QUALIFIED")
    partial = sum(1 for r in task_classes if "PARTIALLY" in r["classification"] or "QUALIFIED" in r["classification"])
    failures = [r for r in worker_rows if int(r["returncode"]) != 0]
    if len(failures) == len(worker_rows):
        verdict = "P6G0R1B_INCONCLUSIVE_SYSTEM_FAILURE"
    elif three >= 2:
        verdict = "P6G0R1B_GNP_STYLE_CANDIDATE_COVERAGE_QUALIFIED"
    elif partial:
        verdict = "P6G0R1B_CANDIDATE_COVERAGE_PARTIAL"
    else:
        verdict = "P6G0R1B_GEOMETRY_AWARE_GENERATOR_NOT_GENERAL"
    final = {
        "verdict": verdict,
        "benchmark_protocol_change": "GNP_STYLE_GEOMETRY_AWARE_GRASP_CANDIDATE_GENERATION",
        "surface_samples": SURFACE_SAMPLE_COUNT,
        "antipodal_tolerance_deg": ANTIPODAL_TOL_DEG,
        "roll_samples_deg": ROLL_SAMPLES_DEG,
        "task_object_lookup_used": False,
        "hidden_physics_used": False,
        "worker_rows": worker_rows,
        "task_classifications": task_classes,
    }
    write_json(TRACK_B / "P6G0R1B_FINAL_VERDICT.json", final)
    report_b = ["# P6-G0-R1B final report", "", f"TRACK_B_VERDICT: {verdict}", "", "## Task Classifications"]
    for row in task_classes:
        report_b.append(f"- task{row['task']}: {row['classification']} ({row['qualified_candidates']} qualified candidates)")
    report_b.append("")
    report_b.append("Reference tasks 1/6 were generated for comparison only and were not used to replace Track A candidates.")
    (TRACK_B / "P6G0R1B_FINAL_REPORT.md").write_text("\n".join(report_b) + "\n", encoding="utf-8")
    (TRACK_B / "P6G0R1B_OBJECT_GEOMETRY_AUDIT.md").write_text("# P6-G0-R1B object geometry audit\n\nSee `P6G0R1B_OBJECT_GEOMETRY_AUDIT.csv`.\n", encoding="utf-8")
    return final


def write_static_artifacts() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TRACK_A.mkdir(parents=True, exist_ok=True)
    TRACK_B.mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "P6-G0-R1 confirmatory completion and GNP-style candidate coverage",
        "track_a": {
            "method_change": "NONE",
            "previous_artifacts": str(PREV),
            "tasks": TRACK_A_TASKS,
            "shard_definition": "one task x one root x one hidden-physics world",
            "root_seeds": load_prev_protocol()["pilot_roots"],
            "frictions": load_prev_protocol()["frictions"],
            "com_offsets": load_prev_protocol()["com_offsets"],
            "force_grid": load_prev_protocol()["force_grid"],
            "refinement_rule": load_prev_protocol()["refinement_rule"],
        },
        "track_b": {
            "benchmark_protocol_change": "GNP_STYLE_GEOMETRY_AWARE_GRASP_CANDIDATE_GENERATION",
            "tasks": TRACK_B_TARGET_TASKS,
            "reference_tasks": TRACK_B_REFERENCE_TASKS,
            "surface_samples": SURFACE_SAMPLE_COUNT,
            "surface_sampling_seed": SURFACE_SAMPLE_SEED,
            "antipodal_tolerance_deg": ANTIPODAL_TOL_DEG,
            "nominal_friction": TRACK_B_FRICTION,
            "centered_com": True,
            "preflight_roots": PREFLIGHT_B_ROOTS,
            "robust_force_N": TRACK_B_FORCE_N,
        },
        "learned_method_used": False,
        "physical_query_used": False,
    }
    write_json(OUT / "P6G0R1_PROTOCOL.json", protocol)
    (OUT / "P6G0R1_PROTOCOL_HASH.txt").write_text(stable_hash_obj(protocol) + "\n", encoding="utf-8")
    write_json(OUT / "P6G0R1_CODE_HASH.txt", {"git_commit": git_commit(), "runner_sha256": sha256_file(Path(__file__).resolve()), "p6g0_runner_sha256": sha256_file(ANALYSIS / "p6g0_grasp_force_physics_benchmark.py")})
    (TRACK_A / "P6G0R1A_ORIGINAL_PROTOCOL_AUDIT.md").write_text(
        "# P6-G0-R1A original protocol audit\n\n"
        f"Source artifacts: `{PREV}`\n\n"
        "METHOD_CHANGE: NONE\n\n"
        "Track A reuses the original task set restricted to task1/task6, root seeds, hidden friction values, CoM offsets, "
        "candidate definitions, force grid, refinement rule, state restoration, deterministic object-level transport, and success criteria. "
        "Only the process sharding and immediate flushing behavior changed.\n",
        encoding="utf-8",
    )


def run_track_a() -> tuple[dict, list[dict]]:
    prev = valid_prev_completed()
    initial_planned = planned_coarse_rows()
    already, missing = compare_planned_completed(initial_planned + [dict(r, planning_stage="previous_triggered_refinement") for r in prev if float(r["requested_force_N"]) in REFINEMENT_FORCES], prev)
    write_csv(TRACK_A / "P6G0R1A_PLANNED_BRANCHES.csv", initial_planned + [dict(r, planning_stage="previous_triggered_refinement") for r in prev if float(r["requested_force_N"]) in REFINEMENT_FORCES])
    write_csv(TRACK_A / "P6G0R1A_ALREADY_COMPLETED.csv", already)
    write_csv(TRACK_A / "P6G0R1A_MISSING_BRANCHES.csv", missing)
    (TRACK_A / "P6G0R1A_RESUME_LOG.md").write_text(
        "# P6-G0-R1A resume log\n\n"
        f"- Previous valid branch rows: {len(prev)}\n"
        f"- Initial planned branch rows: {len(initial_planned)}\n"
        f"- Initial missing branch rows: {len(missing)}\n"
        "- Resume key: (task, root_seed, friction, com_offset, grasp_candidate_id, requested_force)\n",
        encoding="utf-8",
    )
    workers = []
    for task, root_seed, mu, com_sign in track_a_shards_to_run(missing):
        row = launch_worker(
            "track_a_shard",
            {"P6G0R1_TRACK_A_SHARD": "1", "P6G0R1_TASK": str(task), "P6G0R1_ROOT": str(root_seed), "P6G0R1_MU": str(mu), "P6G0R1_COM": str(com_sign)},
            TIMEOUTS_S["track_a_shard"],
            f"track_a_t{task}_s{root_seed}_mu{mu:g}_com{com_sign:+g}.log",
        )
        workers.append(row)
        write_csv(TRACK_A / "P6G0R1A_WORKERS.csv", workers, WORKER_FIELDS)
    r1_branches, r1_parity = collect_track_a_r1_rows()
    completed, dup = dedupe_branches(prev + r1_branches)
    planned_final = infer_dynamic_planned_from_completed(completed)
    _, final_missing = compare_planned_completed(planned_final, completed)
    write_csv(TRACK_A / "P6G0R1A_PLANNED_BRANCHES.csv", planned_final)
    write_csv(TRACK_A / "P6G0R1A_MISSING_BRANCHES.csv", final_missing)
    write_csv(TRACK_A / "P6G0R1A_COMPLETED_BRANCHES.csv", completed, BRANCH_FIELDS)
    write_csv(TRACK_A / "P6G0R1A_STATE_PARITY.csv", read_csv_if(PREV / "P6G0_STATE_PARITY.csv") + r1_parity)
    analysis = analyze_track_a(completed)
    write_csv(TRACK_A / "P6G0R1A_FORCE_EFFECT.csv", analysis["force"])
    write_csv(TRACK_A / "P6G0R1A_GRASP_POSITION_EFFECT.csv", analysis["grasp"])
    write_csv(TRACK_A / "P6G0R1A_FORCE_FRONTIERS.csv", analysis["frontiers"])
    write_csv(TRACK_A / "P6G0R1A_LOCAL_VS_FULL_DIVERGENCE.csv", analysis["divergence"])
    write_csv(TRACK_A / "P6G0R1A_ACTION_RANKING_FLIPS.csv", analysis["flips"])
    write_csv(TRACK_A / "P6G0R1A_PARETO_ACTIONS.csv", analysis["pareto"])
    final = classify_track_a(planned_final, completed, dup, workers, analysis)
    final["planned_branches"] = len(planned_final)
    final["previously_completed"] = len(prev)
    final["completed_in_r1"] = len(r1_branches)
    final["completed_total"] = len(completed)
    final["method_change"] = "NONE"
    final["worker_rows"] = workers
    write_json(TRACK_A / "P6G0R1A_FINAL_VERDICT.json", final)
    lines = [
        "# P6-G0-R1A final report",
        "",
        "METHOD_CHANGE: NONE",
        f"TRACK_A_VERDICT: {final['verdict']}",
        "",
        f"- Planned branches after refinement adjudication: {len(planned_final)}",
        f"- Previously completed valid rows: {len(prev)}",
        f"- Completed in R1: {len(r1_branches)}",
        f"- Still missing: {len(final_missing)}",
        f"- Duplicate rows: {dup}",
        "",
        "## Task Results",
    ]
    for task in TRACK_A_TASKS:
        t = final["task_classes"][task]
        lines.append(f"- task{task}: matrix_complete={t['matrix_complete']}, force={t['force_effect']}, grasp={t['grasp_position_effect']}, flips={t['action_ranking_flips']}, flip_roots={t['flip_roots']}, classification={t['classification']}")
    (TRACK_A / "P6G0R1A_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return final, workers


def run_track_b() -> tuple[dict, list[dict]]:
    workers = []
    for task in TRACK_B_TARGET_TASKS + TRACK_B_REFERENCE_TASKS:
        row = launch_worker(
            "track_b_task",
            {"P6G0R1_TRACK_B_WORKER": "1", "P6G0R1_TASK": str(task)},
            TIMEOUTS_S["track_b_task"],
            f"track_b_task{task}.log",
        )
        workers.append(row)
        write_csv(TRACK_B / "P6G0R1B_WORKERS.csv", workers, WORKER_FIELDS)
    return aggregate_track_b(workers), workers


def maybe_write_r2_protocol(track_b_final: dict) -> None:
    if track_b_final["verdict"] != "P6G0R1B_GNP_STYLE_CANDIDATE_COVERAGE_QUALIFIED":
        return
    (OUT / "P6G0R2_EXPANDED_GRASP_FORCE_PHYSICS_PROTOCOL.md").write_text(
        "# P6-G0-R2 expanded grasp-force physics protocol\n\n"
        "Use the frozen geometry-aware candidates from P6-G0-R1B for newly qualified tasks, the original P6-G0 friction and CoM interventions, "
        "the original {3,5,8} N coarse force grid, the original preregistered {4,6} N refinement rule, strict matched branching, and separate local/lift/full-task outcomes. "
        "Do not use Track B preflight outcomes to tune task-specific force ranges.\n",
        encoding="utf-8",
    )


def overall_classification(track_a: dict, track_b: dict) -> str:
    if track_a["verdict"] in {"P6G0R1A_CONFIRMATORY_SWEEP_NOT_COMPLETE", "P6G0R1A_INCONCLUSIVE_SYSTEM_FAILURE"}:
        return "P6G0R1_CONFIRMATORY_NOT_COMPLETE" if track_a["verdict"].endswith("NOT_COMPLETE") else "P6G0R1_INCONCLUSIVE_SYSTEM_FAILURE"
    if track_b["verdict"] == "P6G0R1B_GNP_STYLE_CANDIDATE_COVERAGE_QUALIFIED":
        return "P6G0R1_CONFIRMATORY_COMPLETE_AND_CANDIDATE_COVERAGE_EXPANDED"
    if track_b["verdict"] == "P6G0R1B_CANDIDATE_COVERAGE_PARTIAL":
        return "P6G0R1_CONFIRMATORY_COMPLETE_CANDIDATE_COVERAGE_PARTIAL"
    if track_b["verdict"] == "P6G0R1B_INCONCLUSIVE_SYSTEM_FAILURE":
        return "P6G0R1_INCONCLUSIVE_SYSTEM_FAILURE"
    return "P6G0R1_CONFIRMATORY_COMPLETE_CANDIDATE_GENERATOR_FAILED"


def final_summary(track_a: dict, track_b: dict, overall: str) -> str:
    a_classes = track_a["task_classes"]
    b_classes = {int(r["task"]): r for r in track_b.get("task_classifications", [])}
    track_a_workers = track_a.get("worker_rows", [])
    b_worker_rows = track_b.get("worker_rows", [])
    planned = track_a.get("planned_branches", 0)
    prev_done = track_a.get("previously_completed", 0)
    done_r1 = track_a.get("completed_in_r1", 0)
    still_missing = track_a.get("missing_count", 0)
    lines = [
        "STATUS:",
        "COMPLETE",
        "METHOD_CHANGE: NONE",
        "BENCHMARK_PROTOCOL_CHANGE:",
        "GNP_STYLE_GEOMETRY_AWARE_GRASP_CANDIDATE_GENERATION",
        "ARTIFACTS:",
        str(OUT),
        "",
        "TRACK A - TASK1/TASK6 CONFIRMATORY COMPLETION",
        "",
        "ORIGINAL MATRIX:",
        f"- Planned branches: {planned}",
        f"- Previously completed: {prev_done}",
        f"- Missing before R1: {max(0, planned - prev_done)}",
        f"- Completed in R1: {done_r1}",
        f"- Still missing: {still_missing}",
        f"- Duplicate rows: {track_a.get('duplicate_rows', 0)}",
        "",
        "WORKERS:",
        "- Shard definition: one task x one root x one hidden-physics world",
        f"- Worker count: {len(track_a_workers)}",
        f"- Timeouts: {sum(1 for r in track_a_workers if int(r['returncode']) == -9)}",
        f"- Resume successful: {still_missing == 0}",
    ]
    for task in TRACK_A_TASKS:
        t = a_classes[task]
        lines += [
            "",
            f"TASK{task}:",
            f"- Matrix complete: {t['matrix_complete']}",
            f"- Force effect: {t['force_effect']} roots={t['force_effect_roots']}",
            f"- Grasp-position effect: {t['grasp_position_effect']} roots={t['grasp_position_effect_roots']}",
            f"- Lift/full divergence: {t['lift_full_divergence']}",
            f"- Action-ranking flips: {t['action_ranking_flips']}",
            f"- Roots reproducing flips: {t['flip_roots']}",
            f"- Reproducibility warnings: {'worker timeout' if any(int(r['returncode']) == -9 and r.get('P6G0R1_TASK') == str(task) for r in track_a_workers) else 'none'}",
            f"- Classification: {t['classification']}",
        ]
    lines += [
        "",
        "TRACK A VERDICT:",
        track_a["verdict"],
        "",
        "TRACK B - GNP-STYLE CANDIDATE COVERAGE",
        "",
        "GENERATOR:",
        f"- Surface samples: {SURFACE_SAMPLE_COUNT}",
        f"- Antipodal tolerance: {ANTIPODAL_TOL_DEG} deg",
        f"- Roll samples: {ROLL_SAMPLES_DEG}",
        "- IK/collision filtering: controller IK-stage reachability plus nominal preflight collision/table checks",
        "- Task/object lookup used: False",
        "- Hidden physics used: False",
    ]
    for task in TRACK_B_TARGET_TASKS:
        b = b_classes.get(task, {})
        lines += [
            "",
            f"TASK{task}:",
            f"- Raw valid candidates: {b.get('raw_valid_candidates', 0)}",
            f"- Selected candidates: {b.get('selected_candidates', 0)}",
            f"- Qualified candidates: {b.get('qualified_candidates', 0)}",
            f"- Actual contact clusters: see TRACK_B/P6G0R1B_ACTUAL_CONTACT_CLUSTERS.csv",
            f"- Realization rate: {b.get('realization_rate', 0)}",
            f"- Short-lift rate: {b.get('short_lift_rate', 0)}",
            f"- Classification: {b.get('classification', 'missing')}",
        ]
    lines += [
        "",
        "REFERENCE TASKS 1/6:",
        "- Generator results: see TRACK_B/P6G0R1B_SELECTED_GRASP_CANDIDATES.csv",
        "- Comparison with frozen original candidates: reference only; Track A candidates were not replaced",
        "",
        "TRACK B VERDICT:",
        track_b["verdict"],
        "",
        "OVERALL CLASSIFICATION:",
        overall,
        "",
        "SCIENTIFIC INTERPRETATION:",
        "1. Track A tests whether the original task1/task6 joint grasp-position x force effects persist under completed sharded execution.",
        "2. Track A uses METHOD_CHANGE: NONE; only worker granularity and flushing changed.",
        "3. Track B asks a separate candidate-coverage question for tasks 0/2/5 under nominal physics.",
        "4. Track B uses deployable geometry and kinematics proxies, not hidden physics or outcome-tuned task-specific rules.",
        "5. Track B results do not compensate for Track A incompleteness.",
        "",
        "WHAT THIS DOES NOT PROVE:",
        "- ACT or pi0.5 can realize the grasp candidates",
        "- a physical query can select the best grasp",
        "- a GNP model can predict candidate feasibility",
        "- DreamTrajectory refinement improves execution",
        "- when-to-query is solved",
        "",
        "NEXT:",
        "- If Track A complete and Track B qualifies: launch the frozen P6-G0-R2 expanded physics sweep.",
        "- If Track A complete but Track B partial: retain tasks1/6 and expand only newly qualified tasks.",
        "- If generator fails: inspect mesh/contact topology before introducing a learned policy.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if os.environ.get("P6G0R1_TRACK_A_SHARD") == "1":
        return run_track_a_shard()
    if os.environ.get("P6G0R1_TRACK_B_WORKER") == "1":
        return run_track_b_worker()
    write_static_artifacts()
    track_a, _ = run_track_a()
    track_b, _ = run_track_b()
    maybe_write_r2_protocol(track_b)
    overall = overall_classification(track_a, track_b)
    final = {
        "status": "COMPLETE",
        "method_change_track_a": "NONE",
        "benchmark_protocol_change_track_b": "GNP_STYLE_GEOMETRY_AWARE_GRASP_CANDIDATE_GENERATION",
        "track_a_verdict": track_a["verdict"],
        "track_b_verdict": track_b["verdict"],
        "overall_classification": overall,
        "artifacts": str(OUT),
        "learned_method_used": False,
        "physical_query_used": False,
    }
    write_json(OUT / "P6G0R1_FINAL_VERDICT.json", final)
    summary = final_summary(track_a, track_b, overall)
    (OUT / "P6G0R1_FINAL_REPORT.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
