#!/usr/bin/env python3
"""P5-S0-A four-task true-matched GNP-style dataset generation.

Dataset generation only. No model training or force-sufficiency scoring.

The orchestrator runs one Isaac worker process per task to avoid cross-task Kit
environment reuse. Each worker executes frozen P4-B probes once per physical
context, saves the post-probe scene state, restores that same state before
split-specific downstream force branches, and records state parity.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/home/exouser/Tabero")
RESULTS_ROOT = REPO / "analysis/results"
P4 = RESULTS_ROOT / "p4_contact_conditioned_probe_20260822_184213"
P4_COLLECT = P4 / "scripts/p4_collect_probe.py"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path(
    "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.warp.core-1.8.2+lx64"
)
OPENPI_CLIENT_SRC = REPO / "benchmarks/openpi/openpi-client/src"

TASKS = [0, 1, 5, 6]
TASK_OBJECTS = {0: "alphabet_soup_1", 1: "cream_cheese_1", 5: "tomato_sauce_1", 6: "butter_1"}
TASK_INSTRUCTIONS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
BASKET_NAME = "basket_1"
TASK_SUITE = "libero_object"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
EXPECTED_P4B_HASH = "a1566334f9f79ad9d8491612f10d086386049295314f6d1fd62512be1867bca9"
FRICTION_SAMPLER_SEED = 2026082305
FRICTION_VALUES = [float(x) for x in np.random.default_rng(FRICTION_SAMPLER_SEED).uniform(0.2, 1.0, size=5)]
TRAIN_FRICTIONS = FRICTION_VALUES[:3]
DEV_FRICTION = FRICTION_VALUES[3]
TEST_FRICTION = FRICTION_VALUES[4]
CONTEXT_SEEDS = [4100, 4101]
TRAIN_FORCES = [3.0, 4.0, 5.0, 6.0, 8.0]
SEEN_FORCE_VALUES = [3.0, 4.0, 5.0, 6.0, 8.0]
HELDOUT_FORCE_VALUES = [3.5, 4.5, 5.5, 6.5, 7.0, 7.5]
DEV_TEST_FORCES = SEEN_FORCE_VALUES + HELDOUT_FORCE_VALUES
SAME_FORCE_REPLAY_FORCE = 5.0

TIMEOUTS_S = {
    "probe": 180.0,
    "snapshot": 20.0,
    "branch_restore": 20.0,
    "full_task_rollout": 600.0,
    "worker": 7200.0,
}


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


OUT = Path(os.environ.get("P5S0A_OUT", RESULTS_ROOT / f"p5s0a_true_matched_dataset_{now_tag()}"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fields = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def friction_split(mu: float) -> str:
    for train_mu in TRAIN_FRICTIONS:
        if abs(mu - train_mu) < 1e-12:
            return "TRAIN"
    if abs(mu - DEV_FRICTION) < 1e-12:
        return "DEV"
    if abs(mu - TEST_FRICTION) < 1e-12:
        return "TEST"
    raise ValueError(f"Unknown friction value {mu}")


def forces_for_split(split: str) -> list[float]:
    if split == "TRAIN":
        return TRAIN_FORCES
    if split in {"DEV", "TEST"}:
        return DEV_TEST_FORCES
    raise ValueError(f"Unknown split {split}")


def force_role(force: float, split: str) -> str:
    if split == "TRAIN":
        return "TRAIN_SEEN_FORCE"
    if force in SEEN_FORCE_VALUES:
        return "EVAL_SEEN_FORCE"
    return "EVAL_HELDOUT_FORCE"


def context_id_for(task_id: int, seed: int, mu: float, split: str) -> str:
    return f"p5s0a_{split.lower()}_t{task_id}_s{seed}_mu{mu:.6f}"


class StageLogger:
    def __init__(
        self,
        out_dir: Path,
        task: int,
        seed: int,
        dt: float | None = None,
        context_id: str = "",
        split: str = "",
        friction: float | None = None,
    ):
        self.path = out_dir / "P5S0A_STAGE_LOG.csv"
        self.task = task
        self.seed = seed
        self.context_id = context_id
        self.split = split
        self.friction = friction
        self.t0 = time.time()
        self.dt = dt

    def emit(self, stage: str, env=None, p4=None, reason: str = "", robot_phase: str = "") -> None:
        snap = telemetry_snapshot(env, p4, self.dt) if env is not None and p4 is not None else {}
        row = {
            "context_id": self.context_id,
            "task": self.task,
            "seed": self.seed,
            "split": self.split,
            "hidden_friction_analysis_only": "" if self.friction is None else self.friction,
            "stage": stage,
            "wall_clock_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_wall_s": round(time.time() - self.t0, 6),
            "environment_step": snap.get("environment_step", ""),
            "simulation_time_s": snap.get("simulation_time_s", ""),
            "robot_phase": robot_phase,
            "object_pose": snap.get("object_pose", ""),
            "gripper_state": snap.get("gripper_state", ""),
            "bilateral_contact_flag": snap.get("bilateral_contact_flag", ""),
            "left_normal_force_N": snap.get("left_normal_force_N", ""),
            "right_normal_force_N": snap.get("right_normal_force_N", ""),
            "stop_failure_reason": reason,
        }
        append_csv(self.path, row)
        print(f"[P5S0A] task={self.task} stage={stage} elapsed={row['elapsed_wall_s']} reason={reason}", flush=True)


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


def stable_hash_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def restorable_snapshot_for_hash(env) -> dict:
    return {"scene_state": _tensor_to_list(env.scene.get_state(is_relative=True))}


def audit_snapshot_for_hash(env, p4) -> dict:
    obs = env.observation_manager.compute()
    dbg = p4._dbg(env)
    out = {
        "scene_state": _tensor_to_list(env.scene.get_state(is_relative=True)),
        "policy_obs": {
            "eef_pose": _tensor_to_list(obs["policy"].get("eef_pose")),
            "gripper_pos": _tensor_to_list(obs["policy"].get("gripper_pos")),
            "arm_joint_pos": _tensor_to_list(obs["policy"].get("arm_joint_pos")),
            "gripper_net_force": _tensor_to_list(obs["policy"].get("gripper_net_force")),
        },
        "action_debug": {
            k: _tensor_to_list(v)
            for k, v in dbg.items()
            if k in {"f_sq_meas", "f_sq_pred", "f_sq_meas_raw", "d_pred", "d_cmd", "d_actual"}
        },
    }
    return out


def telemetry_snapshot(env, p4, dt: float | None = None) -> dict:
    if env is None:
        return {}
    obj_name = TASK_OBJECTS[int(p4.TASK_ID)]
    try:
        obs = env.observation_manager.compute()
        obj_p = env.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy()
        obj_q = env.scene[obj_name].data.root_quat_w[0].detach().cpu().numpy()
        gp = obs["policy"]["gripper_pos"][0].detach().cpu().numpy()
        f = obs["policy"]["gripper_net_force"][0]
        if f.ndim == 3:
            f = f[-1]
        f_np = f.detach().cpu().numpy()
        left_norm = float(abs(f_np[0][2]))
        right_norm = float(abs(f_np[1][2]))
        step = int(getattr(env, "common_step_counter", 0))
        return {
            "environment_step": step,
            "simulation_time_s": "" if dt is None else round(step * dt, 6),
            "object_pose": json.dumps({"p": obj_p.tolist(), "q": obj_q.tolist()}, separators=(",", ":")),
            "gripper_state": json.dumps(gp.tolist(), separators=(",", ":")),
            "bilateral_contact_flag": int(left_norm > 0.15 and right_norm > 0.15),
            "left_normal_force_N": left_norm,
            "right_normal_force_N": right_norm,
        }
    except Exception as exc:
        return {"stop_failure_reason": repr(exc)}


def import_p4_probe(task_id: int):
    os.environ["P4_TASK_ID"] = str(task_id)
    os.environ["P4_VARIANT"] = "P4B"
    os.environ["P4_OUT"] = str(OUT / "P5S0A_FROZEN_P4B_IMPORT")
    os.environ["P4_RESUME"] = "0"
    old_stdout, old_stderr = sys.stdout, sys.stderr
    spec = importlib.util.spec_from_file_location(f"p5s0a_p4b_task{task_id}", P4_COLLECT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {P4_COLLECT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    try:
        sys.stdout.close()
    except Exception:
        pass
    sys.stdout, sys.stderr = old_stdout, old_stderr
    return mod


def downstream_branch(
    env,
    p4,
    *,
    task_id: int,
    force: float,
    branch_label: str,
    context_id: str,
    split: str,
    seed: int,
    friction: float,
    dt: float,
    logger: StageLogger,
    telemetry_path: Path | None = None,
    label_source: str = "TRUE_POST_PROBE_RESET_MATCHED_BRANCH",
) -> dict:
    obj_name = TASK_OBJECTS[task_id]
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    eef_aa = p4._aa(eef[3:7])
    cmd_pos = eef[:3].copy()
    obj0_w = env.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy()
    basket_b, _ = p4._pose_in_base(env, BASKET_NAME)

    lift = cmd_pos.copy()
    lift[2] = max(lift[2] + 0.142, cmd_pos[2] + 0.08)
    transit = basket_b.copy()
    transit[2] = max(lift[2], basket_b[2] + 0.18)
    place = basket_b.copy()
    place[2] = basket_b[2] + 0.10

    phases = [
        ("branch_hold", 20, cmd_pos.copy(), "track"),
        ("lift", 50, lift, "track"),
        ("transit", 110, transit, "freeze"),
        ("over_basket", 30, transit, "freeze"),
        ("place", 40, place, "freeze"),
        ("release", 50, place, "open"),
        ("settle", 50, place, "open"),
    ]
    d_pred = p4.D_CLOSED
    step = 0
    dropped = lift_success = pick_success = lost_in_transit = timed_out = 0
    max_basket = peak_force = integrated_force = 0.0
    force_samples: list[float] = []
    trajectory_rows: list[dict] = []
    term_reason = ""

    logger.emit(f"{branch_label}_EXECUTION_STARTED", env, p4, robot_phase="full_task")
    with Timeout(TIMEOUTS_S["full_task_rollout"], f"{branch_label}_FULL_TASK"):
        for phase, n_steps, target, mode in phases:
            start = cmd_pos.copy()
            for i in range(n_steps):
                cmd_pos = p4._interp(start, target, i, n_steps)
                if mode == "open":
                    d_pred = p4.D_OPEN
                action = p4._make_action(cmd_pos, eef_aa, d_pred, force if mode != "open" else 0.0, env.device)
                obs, _, term, trunc, _ = env.step(action)
                step += 1
                f_sq = p4._f(p4._dbg(env).get("f_sq_meas"), 0.0)
                peak_force = max(peak_force, f_sq)
                integrated_force += f_sq * dt
                if mode == "track":
                    d_pred = p4._force_servo(d_pred, f_sq, force)
                if mode != "open":
                    force_samples.append(f_sq)
                try:
                    dropped = int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
                except Exception:
                    pass
                obj_p = env.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy()
                obj_q = env.scene[obj_name].data.root_quat_w[0].detach().cpu().numpy()
                gp = obs["policy"]["gripper_pos"][0].detach().cpu().numpy()
                obj_dz = float(obj_p[2] - obj0_w[2])
                trajectory_rows.append(
                    {
                        "context_id": context_id,
                        "branch_label": branch_label,
                        "task": task_id,
                        "split": split,
                        "seed": seed,
                        "hidden_friction_analysis_only": float(friction),
                        "requested_force_N": float(force),
                        "step": step,
                        "t_s": step * dt,
                        "phase": phase,
                        "mode": mode,
                        "measured_force_N": float(f_sq),
                        "cmd_x": float(cmd_pos[0]),
                        "cmd_y": float(cmd_pos[1]),
                        "cmd_z": float(cmd_pos[2]),
                        "object_x_analysis_only": float(obj_p[0]),
                        "object_y_analysis_only": float(obj_p[1]),
                        "object_z_analysis_only": float(obj_p[2]),
                        "object_qw_analysis_only": float(obj_q[0]),
                        "object_qx_analysis_only": float(obj_q[1]),
                        "object_qy_analysis_only": float(obj_q[2]),
                        "object_qz_analysis_only": float(obj_q[3]),
                        "gripper_pos_0": float(gp[0]) if len(gp) > 0 else "",
                        "gripper_pos_1": float(gp[1]) if len(gp) > 1 else "",
                    }
                )
                if obj_dz >= 0.01:
                    pick_success = 1
                if obj_dz >= 0.03:
                    lift_success = 1
                if lift_success and phase in ("transit", "over_basket") and (dropped or obj_dz < 0.02):
                    lost_in_transit = 1
                try:
                    import torch

                    cs = env.scene[f"contact_{BASKET_NAME}_{obj_name}"]
                    bc = float(torch.linalg.vector_norm(cs.data.force_matrix_w.reshape(-1, 3)[0]).item())
                    max_basket = max(max_basket, bc)
                except Exception:
                    pass
                if bool(term[0].item()) or bool(trunc[0].item()):
                    try:
                        if bool(env.termination_manager.get_term("time_out")[0].item()):
                            timed_out = 1
                            term_reason = "time_out"
                    except Exception:
                        term_reason = "term"
                    break
            else:
                continue
            break

    success = int(lift_success == 1 and max_basket > 0.05 and dropped == 0)
    measured_mean = float(np.mean(force_samples)) if force_samples else 0.0
    steady = force_samples[len(force_samples) // 2 :] if force_samples else []
    steady_state_mean = float(np.mean(steady)) if steady else measured_mean
    tracking_mae = float(np.mean([abs(x - float(force)) for x in steady])) if steady else abs(measured_mean - float(force))
    if telemetry_path is not None:
        write_csv(telemetry_path, trajectory_rows)
    rec = {
        "branch_id": f"{context_id}_{branch_label}_F{force:g}",
        "context_id": context_id,
        "task": task_id,
        "task_instruction": TASK_INSTRUCTIONS[task_id],
        "split": split,
        "seed": seed,
        "hidden_friction_analysis_only": float(friction),
        "branch_label": branch_label,
        "force_role": force_role(float(force), split) if label_source == "TRUE_POST_PROBE_RESET_MATCHED_BRANCH" else "SAME_FORCE_REPLAY_DIAGNOSTIC",
        "requested_force_N": float(force),
        "measured_force_trajectory_json": json.dumps(force_samples),
        "measured_force_mean_N": measured_mean,
        "measured_force_peak_N": float(peak_force),
        "steady_state_mean_N": steady_state_mean,
        "force_tracking_mae_N": tracking_mae,
        "force_tracking_error_N": steady_state_mean - float(force),
        "integrated_force": float(integrated_force),
        "pick_success": pick_success,
        "lift_success": lift_success,
        "transport_retention": int(lift_success == 1 and lost_in_transit == 0),
        "place_success": int(max_basket > 0.05),
        "full_task_success_y": success,
        "failure_reason": "" if success else (term_reason or "full_task_failure"),
        "dropped": dropped,
        "lost_in_transit": lost_in_transit,
        "timeout": timed_out,
        "steps": step,
        "t_episode_s": step * dt,
        "failure_stage": "" if success else "full_task_rollout",
        "episode_length": step,
        "label_source": label_source,
        "telemetry_path": str(telemetry_path) if telemetry_path is not None else "",
    }
    logger.emit(f"{branch_label}_EXECUTION_COMPLETE", env, p4, reason=rec["failure_reason"], robot_phase="full_task")
    return rec


def derive_p4b_probe_quality(p4, probe_rows, probe_rec: dict) -> dict:
    phases = {r.probe_phase for r in probe_rows}
    contact_retained = int(1 - int(probe_rec.get("contact_lost_probe", 0)))
    drop = int(probe_rec.get("dropped", 0))
    disturbance = int(probe_rec.get("major_disturbance", 0))
    return_completed = int("probe_back" in phases and "probe_hold" in phases)
    rho_impulse = float(probe_rec.get("rho_impulse", 0.0) or 0.0)
    marker_tangent = float(probe_rec.get("marker_tangential_peak", 0.0) or 0.0)
    informative = int(
        rho_impulse >= float(getattr(p4, "RHO_IMPULSE_TARGET", 0.012))
        or marker_tangent >= 0.015
        or str(probe_rec.get("stop_trigger", "")) == "normalized_shear_impulse"
    )
    safe = int(
        int(probe_rec.get("probe_failure", 0)) == 0
        and contact_retained == 1
        and drop == 0
        and disturbance == 0
        and return_completed == 1
    )
    return {
        "probe_qualified": int(safe == 1 and informative == 1),
        "contact_retained": contact_retained,
        "drop": drop,
        "disturbance": disturbance,
        "return_completed": return_completed,
        "informative": informative,
        "safe": safe,
    }


def worker_main() -> int:
    from isaaclab.app import AppLauncher

    task_id = int(os.environ["P5S0A_TASK_ID"])
    task_dir = OUT / f"task{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    boot_logger = StageLogger(OUT, task_id, -1)
    stage_logger = boot_logger
    boot_logger.emit("RESET_STARTED", reason="worker_start")

    app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        p4 = import_p4_probe(task_id)
        setup_task_objects(TASK_SUITE, task_id)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = 45.0
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        boot_logger.dt = dt
        boot_logger.emit("RESET_COMPLETE", env, p4, robot_phase="env_ready")

        probe_dir = OUT / "P5S0A_PROBE_TELEMETRY"
        branch_dir = OUT / "P5S0A_BRANCH_TELEMETRY"
        context_rows: list[dict] = []
        branch_rows: list[dict] = []
        parity_rows: list[dict] = []
        replay_rows: list[dict] = []
        replay_done = False

        for mu in FRICTION_VALUES:
            split = friction_split(mu)
            for seed in CONTEXT_SEEDS:
                context_id = context_id_for(task_id, seed, mu, split)
                stage_logger = StageLogger(OUT, task_id, seed, dt=dt, context_id=context_id, split=split, friction=mu)
                stage_logger.emit("RESET_STARTED", env, p4, robot_phase="context_start")
                stage_logger.emit("VLA_APPROACH_STARTED", env, p4, robot_phase="approach")
                stage_logger.emit("CONTACT_ESTABLISHMENT_STARTED", env, p4, robot_phase="preload")
                with Timeout(TIMEOUTS_S["probe"], "PROBE"):
                    probe_rows, probe_rec = p4.run_probe_episode(env, seed_idx=seed, mu=mu, trial_id=context_id, dt=dt)

                phases = {r.probe_phase for r in probe_rows}
                stage_logger.emit("PRELIFT_STATE_REACHED", env, p4, robot_phase="pre_probe")
                if "hold" in phases:
                    stage_logger.emit("BILATERAL_CONTACT_ESTABLISHED", env, p4, robot_phase="hold")
                stage_logger.emit("PROBE_STARTED", env, p4, robot_phase="probe")
                if "probe_out" in phases:
                    stage_logger.emit("PROBE_FORWARD_COMPLETE", env, p4, reason=str(probe_rec.get("stop_trigger", "")), robot_phase="probe_out")
                if "probe_back" in phases:
                    stage_logger.emit("PROBE_RETURN_STARTED", env, p4, robot_phase="probe_back")
                stage_logger.emit("PROBE_COMPLETED", env, p4, reason=str(probe_rec.get("stop_trigger", "")), robot_phase="probe_hold")

                write_csv(probe_dir / f"{context_id}_probe_timesteps.csv", [asdict(r) for r in probe_rows])

                quality = derive_p4b_probe_quality(p4, probe_rows, probe_rec)
                probe_qualified = int(quality["probe_qualified"])
                context_status = "CONTEXT_READY_FOR_BRANCHING" if probe_qualified else "CONTEXT_INVALID_PROBE"
                stage_logger.emit("POST_PROBE_SNAPSHOT_STARTED", env, p4, reason=context_status)
                with Timeout(TIMEOUTS_S["snapshot"], "SNAPSHOT"):
                    sq = env.scene.get_state(is_relative=True)
                    reference_snapshot = restorable_snapshot_for_hash(env)
                    reference_audit_snapshot = audit_snapshot_for_hash(env, p4)
                    hq = stable_hash_obj(reference_snapshot)
                    hq_audit = stable_hash_obj(reference_audit_snapshot)
                stage_logger.emit("POST_PROBE_SNAPSHOT_SAVED", env, p4, reason=hq)

                expected_branches = len(forces_for_split(split)) if probe_qualified else 0
                context_row = {
                    "context_id": context_id,
                    "task": task_id,
                    "task_instruction": TASK_INSTRUCTIONS[task_id],
                    "split": split,
                    "hidden_friction_analysis_only": float(mu),
                    "seed": seed,
                    "probe_implementation": "P4-B common contact-frame shear",
                    "probe_qualified": probe_qualified,
                    "contact_retained": quality["contact_retained"],
                    "drop": quality["drop"],
                    "disturbance": quality["disturbance"],
                    "probe_completion": int("probe_out" in phases),
                    "return_completed": quality["return_completed"],
                    "informative": quality["informative"],
                    "safe": quality["safe"],
                    "probe_quality_derivation": "derived_from_existing_P4B_fields_without_changing_probe",
                    "stop_reason": probe_rec.get("stop_trigger", ""),
                    "actual_displacement_mm": float(probe_rec.get("actual_probe_displacement_mm", np.nan)),
                    "post_probe_state_hash": hq,
                    "post_probe_audit_hash": hq_audit,
                    "status": context_status,
                    "planned_primary_branches": expected_branches,
                    "completed_primary_branches": 0,
                    "probe_rerun_between_branches": False,
                    "probe_telemetry_path": str(probe_dir / f"{context_id}_probe_timesteps.csv"),
                }

                completed_for_context = 0
                if probe_qualified:
                    for force in forces_for_split(split):
                        force_tag = str(force).replace(".", "p")
                        label = f"BRANCH_F{force_tag}"
                        pre_hash = stable_hash_obj(restorable_snapshot_for_hash(env))
                        pre_audit_hash = stable_hash_obj(audit_snapshot_for_hash(env, p4))
                        stage_logger.emit(f"{label}_RESTORE_STARTED", env, p4, reason=f"pre_hash={pre_hash}")
                        with Timeout(TIMEOUTS_S["branch_restore"], f"{label}_RESTORE"):
                            env.reset_to(sq, torch.tensor([0], device=env.device), is_relative=True)
                            post_hash = stable_hash_obj(restorable_snapshot_for_hash(env))
                            post_audit_hash = stable_hash_obj(audit_snapshot_for_hash(env, p4))
                        parity_pass = int(post_hash == hq)
                        parity_row = {
                            "context_id": context_id,
                            "branch_label": label,
                            "task": task_id,
                            "split": split,
                            "seed": seed,
                            "hidden_friction_analysis_only": float(mu),
                            "requested_force_N": float(force),
                            "is_replay_diagnostic": 0,
                            "pre_restore_hash": pre_hash,
                            "post_restore_hash": post_hash,
                            "reference_post_probe_hash": hq,
                            "pre_restore_audit_hash": pre_audit_hash,
                            "post_restore_audit_hash": post_audit_hash,
                            "reference_post_probe_audit_hash": hq_audit,
                            "audit_hash_parity": int(post_audit_hash == hq_audit),
                            "parity_pass": parity_pass,
                            "field_level_differences": "" if parity_pass else "restorable scene_state hash mismatch",
                            "audit_differences": "" if post_audit_hash == hq_audit else "recomputed observation/action-debug hash differs; excluded from PASS parity and documented in hash scope",
                        }
                        parity_rows.append(parity_row)
                        stage_logger.emit(f"{label}_RESTORE_VERIFIED", env, p4, reason="PASS" if parity_pass else "FAIL")
                        if not parity_pass:
                            continue
                        telemetry_path = branch_dir / f"{context_id}_{label}_F{force:g}_trajectory.csv"
                        br = downstream_branch(
                            env,
                            p4,
                            task_id=task_id,
                            force=force,
                            branch_label=label,
                            context_id=context_id,
                            split=split,
                            seed=seed,
                            friction=mu,
                            dt=dt,
                            logger=stage_logger,
                            telemetry_path=telemetry_path,
                        )
                        br["post_probe_state_hash"] = hq
                        br["state_parity"] = parity_pass
                        branch_rows.append(br)
                        completed_for_context += 1

                    if not replay_done:
                        pair: list[dict] = []
                        for replay_idx in [1, 2]:
                            label = f"REPLAY_{replay_idx}_F{str(SAME_FORCE_REPLAY_FORCE).replace('.', 'p')}"
                            pre_hash = stable_hash_obj(restorable_snapshot_for_hash(env))
                            stage_logger.emit(f"{label}_RESTORE_STARTED", env, p4, reason=f"pre_hash={pre_hash}")
                            with Timeout(TIMEOUTS_S["branch_restore"], f"{label}_RESTORE"):
                                env.reset_to(sq, torch.tensor([0], device=env.device), is_relative=True)
                                post_hash = stable_hash_obj(restorable_snapshot_for_hash(env))
                            parity_pass = int(post_hash == hq)
                            stage_logger.emit(f"{label}_RESTORE_VERIFIED", env, p4, reason="PASS" if parity_pass else "FAIL")
                            if not parity_pass:
                                replay_rows.append(
                                    {
                                        "context_id": context_id,
                                        "task": task_id,
                                        "split": split,
                                        "seed": seed,
                                        "hidden_friction_analysis_only": float(mu),
                                        "replay_index": replay_idx,
                                        "requested_force_N": SAME_FORCE_REPLAY_FORCE,
                                        "state_parity": parity_pass,
                                        "full_task_success_y": "",
                                        "failure_reason": "STATE_PARITY_FAIL",
                                    }
                                )
                                continue
                            telemetry_path = branch_dir / f"{context_id}_{label}_trajectory.csv"
                            rr = downstream_branch(
                                env,
                                p4,
                                task_id=task_id,
                                force=SAME_FORCE_REPLAY_FORCE,
                                branch_label=label,
                                context_id=context_id,
                                split=split,
                                seed=seed,
                                friction=mu,
                                dt=dt,
                                logger=stage_logger,
                                telemetry_path=telemetry_path,
                                label_source="SAME_FORCE_REPLAY_DIAGNOSTIC",
                            )
                            rr["state_parity"] = parity_pass
                            rr["post_probe_state_hash"] = hq
                            rr["replay_index"] = replay_idx
                            pair.append(rr)
                            replay_rows.append(rr)
                        if len(pair) == 2:
                            outcome_disagree = int(pair[0]["full_task_success_y"] != pair[1]["full_task_success_y"])
                            length_delta = abs(int(pair[0]["episode_length"]) - int(pair[1]["episode_length"]))
                            for row in replay_rows[-2:]:
                                row["same_force_outcome_disagreement"] = outcome_disagree
                                row["same_force_episode_length_delta"] = length_delta
                        replay_done = True

                context_row["completed_primary_branches"] = completed_for_context
                context_row["strict_matched"] = int(
                    probe_qualified == 1
                    and completed_for_context == expected_branches
                    and all(
                        int(r["parity_pass"]) == 1
                        for r in parity_rows
                        if r["context_id"] == context_id and int(r["is_replay_diagnostic"]) == 0
                    )
                )
                context_rows.append(context_row)
                stage_logger.emit("CONTEXT_COMPLETE", env, p4, reason="PASS" if context_row["strict_matched"] else context_status)
                write_csv(task_dir / "context.csv", context_rows)
                write_csv(task_dir / "branches.csv", branch_rows)
                write_csv(task_dir / "parity.csv", parity_rows)
                write_csv(task_dir / "same_force_replay.csv", replay_rows)

        pass_task = bool(
            len(context_rows) == len(FRICTION_VALUES) * len(CONTEXT_SEEDS)
            and all(int(r["strict_matched"]) == 1 for r in context_rows if int(r["probe_qualified"]) == 1)
        )
        write_json(
            task_dir / "result.json",
            {
                "task": task_id,
                "pass": pass_task,
                "contexts": len(context_rows),
                "probe_qualified_contexts": sum(int(r["probe_qualified"]) for r in context_rows),
                "strict_matched_contexts": sum(int(r["strict_matched"]) for r in context_rows),
                "primary_branches": len(branch_rows),
                "primary_parity_passes": sum(int(r["parity_pass"]) for r in parity_rows if int(r.get("is_replay_diagnostic", 0)) == 0),
                "same_force_replay_rows": len(replay_rows),
            },
        )
        return 0
    except Exception as exc:
        stage = classify_exception_stage(str(exc))
        write_json(
            task_dir / "error.json",
            {"task": task_id, "error": repr(exc), "trace": traceback.format_exc(), "stalled_stage": stage},
        )
        try:
            stage_logger.emit("CONTEXT_COMPLETE", env, import_p4_probe(task_id) if env is not None else None, reason=f"FAIL:{stage}")
        except Exception:
            pass
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


def classify_exception_stage(message: str) -> str:
    m = message.upper()
    if "PROBE" in m:
        return "STALL_DURING_PROBE"
    if "SNAPSHOT" in m:
        return "STALL_DURING_SNAPSHOT"
    if "RESTORE" in m:
        return "STALL_DURING_BRANCH_RESTORE"
    if "FULL_TASK" in m:
        return "STALL_DURING_FULL_TASK"
    if "CUDA" in m or "GPU" in m:
        return "GPU_OR_SIMULATION_STALL"
    return "OTHER"


def worker_env(out: Path, task: int) -> dict:
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
            "P5S0A_OUT": str(out),
            "P5S0A_WORKER": "1",
            "P5S0A_TASK_ID": str(task),
        }
    )
    return env


def launch_worker(out: Path, task: int) -> dict:
    log_path = out / "logs" / f"task{task}_worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ISAAC_PY), "-u", str(Path(__file__).resolve())]
    start = time.time()
    start_iso = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=worker_env(out, task), stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = proc.wait(timeout=TIMEOUTS_S["worker"])
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
    return {
        "task": task,
        "worker_pid": proc.pid,
        "process_start_utc": start_iso,
        "returncode": returncode,
        "elapsed_wall_s": time.time() - start,
        "log": str(log_path),
    }


def read_task_rows(out: Path) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    contexts: list[dict] = []
    branches: list[dict] = []
    parity: list[dict] = []
    replay: list[dict] = []
    results: list[dict] = []
    for task in TASKS:
        task_dir = out / f"task{task}"
        if (task_dir / "context.csv").exists() and (task_dir / "context.csv").stat().st_size > 0:
            with (task_dir / "context.csv").open() as fh:
                contexts.extend(list(csv.DictReader(fh)))
        if (task_dir / "branches.csv").exists() and (task_dir / "branches.csv").stat().st_size > 0:
            with (task_dir / "branches.csv").open() as fh:
                branches.extend(list(csv.DictReader(fh)))
        if (task_dir / "parity.csv").exists() and (task_dir / "parity.csv").stat().st_size > 0:
            with (task_dir / "parity.csv").open() as fh:
                parity.extend(list(csv.DictReader(fh)))
        if (task_dir / "same_force_replay.csv").exists() and (task_dir / "same_force_replay.csv").stat().st_size > 0:
            with (task_dir / "same_force_replay.csv").open() as fh:
                replay.extend(list(csv.DictReader(fh)))
        result = {"task": task, "pass": False, "reason": "missing_result"}
        if (task_dir / "result.json").exists():
            result.update(json.loads((task_dir / "result.json").read_text()))
        if (task_dir / "error.json").exists():
            result.update(json.loads((task_dir / "error.json").read_text()))
        results.append(result)
    return contexts, branches, parity, replay, results


def write_static_artifacts(out: Path) -> None:
    p4_hash = sha256_file(P4_COLLECT)
    if p4_hash != EXPECTED_P4B_HASH:
        raise RuntimeError(f"P4-B probe hash mismatch: expected {EXPECTED_P4B_HASH}, got {p4_hash}")

    protocol = {
        "name": "P5-S0-A Four-Task True-Matched GNP-Style Dataset Generation",
        "scope": "FOUR_TASK_DEVELOPMENT_DATASET",
        "method_change": "NONE",
        "tasks": TASKS,
        "task2_used": False,
        "contexts": {
            "friction_values": 5,
            "seeds_per_task_friction": len(CONTEXT_SEEDS),
            "planned_physical_contexts": len(TASKS) * len(FRICTION_VALUES) * len(CONTEXT_SEEDS),
            "train_contexts": len(TASKS) * len(TRAIN_FRICTIONS) * len(CONTEXT_SEEDS),
            "dev_contexts": len(TASKS) * len(CONTEXT_SEEDS),
            "test_contexts": len(TASKS) * len(CONTEXT_SEEDS),
        },
        "forces": {
            "train_forces_N": TRAIN_FORCES,
            "dev_test_forces_N": DEV_TEST_FORCES,
            "heldout_force_values_N": HELDOUT_FORCE_VALUES,
            "same_force_replay_force_N": SAME_FORCE_REPLAY_FORCE,
            "planned_primary_branches": 296,
        },
        "probe": "P4-B common contact-frame shear",
        "no_model_training": True,
        "timeouts_s": TIMEOUTS_S,
        "process_isolation_fix": "one Isaac worker process per task",
        "state_hash": "env.scene.get_state(is_relative=True)",
    }
    write_json(out / "P5S0A_PROTOCOL.json", protocol)
    (out / "P5S0A_PROTOCOL_HASH.txt").write_text(stable_hash_obj(protocol) + "\n", encoding="utf-8")
    write_json(
        out / "P5S0A_CODE_HASH.txt",
        {"git_commit": git_commit(), "runner_sha256": sha256_file(Path(__file__).resolve()), "p4b_collect_sha256": p4_hash},
    )
    frozen_probe = {
        "implementation": "P4-B common contact-frame shear",
        "implementation_path": str(P4_COLLECT),
        "task_specific_parameters": False,
        "variant": "P4B",
        "global_parameters": {
            "P4B_BASE_FORCE_N": 3.0,
            "P4B_PRELOAD_STEP_N": 0.5,
            "P4B_PRELOAD_CAP_N": 4.5,
            "P4_FIXED_AMP_MM": 2.0,
            "P4_ADAPTIVE_STEP_MM": 0.2,
            "P4_MAX_DISP_MM": 2.0,
            "P4_RHO_IMPULSE_TARGET": 0.012,
            "P4_RHO_CAP": 0.08,
            "P4_RELATIVE_NORMAL_ALPHA": 0.55,
            "P4_RETURN_STEPS": 10,
            "P4_POST_HOLD_STEPS": 5,
        },
        "code_hash": p4_hash,
    }
    write_json(out / "P5S0A_FROZEN_PROBE.json", frozen_probe)

    write_json(
        out / "P5S0A_FRICTION_MANIFEST.json",
        {
            "sampling_seed": FRICTION_SAMPLER_SEED,
            "sampler": "numpy.random.default_rng(seed).uniform(0.2, 1.0, size=5)",
            "sampled_values": FRICTION_VALUES,
            "train_friction": TRAIN_FRICTIONS,
            "dev_friction": DEV_FRICTION,
            "test_friction": TEST_FRICTION,
            "analysis_only_not_model_input": True,
        },
    )
    seed_contexts = [
        {
            "context_id": context_id_for(task, seed, mu, friction_split(mu)),
            "task": task,
            "seed": seed,
            "split": friction_split(mu),
            "hidden_friction_analysis_only": float(mu),
        }
        for task in TASKS
        for mu in FRICTION_VALUES
        for seed in CONTEXT_SEEDS
    ]
    write_json(out / "P5S0A_SEED_MANIFEST.json", {"context_seeds": CONTEXT_SEEDS, "planned_contexts": seed_contexts})
    write_json(
        out / "P5S0A_SPLIT_MANIFEST.json",
        {
            "split_rule": "friction-level split; all branches from each context remain in the same split",
            "TRAIN": [r for r in seed_contexts if r["split"] == "TRAIN"],
            "DEV": [r for r in seed_contexts if r["split"] == "DEV"],
            "TEST": [r for r in seed_contexts if r["split"] == "TEST"],
        },
    )
    write_json(
        out / "P5S0A_FORCE_MANIFEST.json",
        {
            "train_force_values_N": TRAIN_FORCES,
            "seen_force_values_N": SEEN_FORCE_VALUES,
            "heldout_force_values_N": HELDOUT_FORCE_VALUES,
            "dev_test_force_values_N": DEV_TEST_FORCES,
            "same_force_replay_force_N": SAME_FORCE_REPLAY_FORCE,
            "offgrid_is_development_only": True,
        },
    )
    (out / "P5S0A_STATE_HASH_SCOPE.md").write_text(
        """# P5-S0-A State Hash Scope

Included in the PASS/FAIL reference and post-restore hash:
- `env.scene.get_state(is_relative=True)` recursively serialized, including IsaacLab scene state available for `env.reset_to`.

Recorded as non-PASS audit hashes:
- Recomputed policy observation fields: `eef_pose`, `gripper_pos`, `arm_joint_pos`, `gripper_net_force`.
- Force/action debug fields when exposed: `f_sq_meas`, `f_sq_pred`, `f_sq_meas_raw`, `d_pred`, `d_cmd`, `d_actual`.

Excluded or unresolved:
- Python, Torch, NumPy, PhysX, and task RNG internal states after the post-probe point.
- Internal force-controller integrator/state not present in `action_manager.debug_info`.
- Tactile sensor history buffers beyond the recomputed observation tensors exposed by the policy group.
- Renderer/GPU/RTX state and asynchronous sensor pipeline internals.
- Wall-clock time and log process state.

Matching risk:
`env.reset_to` is treated as authoritative for restorable simulator scene state. Recomputed observations and action debug values can differ immediately after restore even when scene state matches; those audit hashes are reported but do not define the S0-A PASS parity criterion.
""",
        encoding="utf-8",
    )
    (out / "P5S0A_DATASET_SCHEMA.md").write_text(
        """# P5-S0-A Dataset Schema

Logical unit:
- `context_id`
- task descriptor and instruction
- split
- seed
- hidden friction as analysis-only simulator configuration
- one complete P4-B probe trace in `P5S0A_PROBE_TELEMETRY/`
- one post-probe state hash
- branch rows from the identical restored post-probe state

Primary learning sample:
`(context_id, probe_trace, task/context descriptor, requested_force_N, full_task_success_y)`.

Fields prohibited as future model input:
- `hidden_friction_analysis_only`
- object GT state and trajectory fields marked `analysis_only`
- future outcome fields
- canonical force frontier or inferred labels

All primary branches from a context remain in one split. Same-force replay rows are diagnostics only and are not independent training labels.
""",
        encoding="utf-8",
    )


def build_task_result_rows(contexts: list[dict], branches: list[dict], parity: list[dict], results: list[dict]) -> list[dict]:
    by_context_task = {int(r["task"]): r for r in contexts if r.get("task", "") != ""}
    branches_by_task: dict[int, list[dict]] = {task: [] for task in TASKS}
    for row in branches:
        branches_by_task[int(row["task"])].append(row)
    parity_by_task: dict[int, list[dict]] = {task: [] for task in TASKS}
    for row in parity:
        parity_by_task[int(row["task"])].append(row)
    result_by_task = {int(r["task"]): r for r in results}
    rows = []
    for task in TASKS:
        ctx = by_context_task.get(task, {})
        brs = branches_by_task.get(task, [])
        prs = parity_by_task.get(task, [])
        res = result_by_task.get(task, {})
        pass_task = bool(res.get("pass", False))
        rows.append(
            {
                "task": task,
                "verdict": "PASS" if pass_task else "FAIL",
                "probe_qualified": ctx.get("probe_qualified", False),
                "post_probe_hash": ctx.get("post_probe_state_hash", ""),
                "branch_count": len(brs),
                "parity_pass_count": sum(int(float(p.get("parity_pass", 0))) for p in prs),
                "branch_A_parity": next((p.get("parity_pass", "") for p in prs if p.get("branch_label") == "BRANCH_A"), ""),
                "branch_B_parity": next((p.get("parity_pass", "") for p in prs if p.get("branch_label") == "BRANCH_B"), ""),
                "branch_A_outcome": next((b.get("full_task_success_y", "") for b in brs if "BRANCH_A" in b.get("branch_id", "")), ""),
                "branch_B_outcome": next((b.get("full_task_success_y", "") for b in brs if "BRANCH_B" in b.get("branch_id", "")), ""),
                "failure_reason": res.get("reason", res.get("stalled_stage", "")),
            }
        )
    return rows


def write_task1_forensic(out: Path, worker_records: list[dict], results: list[dict]) -> None:
    task1 = next((r for r in results if int(r["task"]) == 1), {"task": 1, "pass": False, "reason": "missing"})
    stalled = task1.get("stalled_stage", "NOT_STALLED" if task1.get("pass") else "OTHER")
    root_cause = (
        "Previous task1 stall was localized to cross-task Isaac/Kit environment reuse in one process. "
        "R1 uses one fresh Isaac worker process per task as the minimum engineering fix."
    )
    if not task1.get("pass") and "error" in task1:
        root_cause += f" Current task1 worker failed with: {task1.get('error')}"
    obj = {
        "task": 1,
        "stalled_stage": stalled,
        "root_cause": root_cause,
        "engineering_fix": "Process-isolated per-task Isaac worker; global P4B probe and force branches unchanged.",
        "scientific_method_changed": False,
        "worker_record": next((w for w in worker_records if int(w["task"]) == 1), {}),
    }
    write_json(out / "TASK1_STALL_FORENSIC.json", obj)
    (out / "TASK1_STALL_FORENSIC.md").write_text(
        f"""# Task1 Stall Forensic

Stalled stage: `{obj['stalled_stage']}`

Root cause: {obj['root_cause']}

Engineering fix: {obj['engineering_fix']}

Scientific method changed: `False`

Notes: the prior four-task preflight completed task0, then task1 did not advance to a context row after the second task environment was created in the same Isaac process. R1 therefore launches each task in a fresh worker process and aggregates artifacts only after worker exit.
""",
        encoding="utf-8",
    )


def write_final(out: Path, task_rows: list[dict], contexts: list[dict], branches: list[dict], parity: list[dict]) -> None:
    passed = [int(r["task"]) for r in task_rows if r["verdict"] == "PASS"]
    failed = [int(r["task"]) for r in task_rows if r["verdict"] != "PASS"]
    if len(passed) == 4:
        classification = "P5S0A_FOUR_TASK_STRICT_MATCHED_BRANCHING_QUALIFIED"
        status = "PASS"
        ready = "YES"
    elif passed:
        classification = "P5S0A_STRICT_MATCHING_PARTIALLY_QUALIFIED"
        status = "PARTIAL"
        ready = "NO"
    else:
        classification = "P5S0A_STRICT_MATCHING_NOT_QUALIFIED"
        status = "FAIL"
        ready = "NO"
    verdict = {
        "STATUS": status,
        "METHOD_CHANGE": "NONE",
        "PRIMARY_CLASSIFICATION": classification,
        "tasks_passed": passed,
        "tasks_failed": failed,
        "contexts": len(contexts),
        "branches": len(branches),
        "state_parity_rows": len(parity),
        "true_matched_dataset_ready": ready == "YES",
        "task2_used": False,
        "model_training_run": False,
    }
    write_json(out / "P5S0A_FINAL_VERDICT.json", verdict)

    proposed = ""
    if ready == "YES":
        proposed_obj = {
            "name": "P5-S0-A TRUE MATCHED DATASET",
            "do_not_launch_without_explicit_instruction": True,
            "tasks": TASKS,
            "hidden_friction_values": 5,
            "seeds_per_task_friction": 2,
            "physical_contexts": 40,
            "force_branches_per_context": 5,
            "estimated_full_task_branches": 200,
        }
        write_json(out / "P5S0A_TRUE_MATCHED_DATASET_PROTOCOL_PROPOSED.json", proposed_obj)
        proposed = "\nProposed next run written to `P5S0A_TRUE_MATCHED_DATASET_PROTOCOL_PROPOSED.json`.\n"

    task_block = "\n".join(
        [
            f"""task{int(r['task'])}:
- probe qualified: {r['probe_qualified']}
- post-probe hash: {r['post_probe_hash']}
- branch A parity: {r['branch_A_parity']}
- branch B parity: {r['branch_B_parity']}
- branch A full-task outcome: {r['branch_A_outcome']}
- branch B full-task outcome: {r['branch_B_outcome']}
- PASS/FAIL: {r['verdict']}"""
            for r in task_rows
        ]
    )
    report = f"""# P5-S0-A Final Report

STATUS:
{status}

METHOD_CHANGE: NONE

ARTIFACTS:
{out}

FROZEN PROBE:
- implementation: P4-B common contact-frame shear
- task-specific parameters: False
- code hash: {sha256_file(P4_COLLECT)}

TASK1 FORENSIC:
- stalled stage: see TASK1_STALL_FORENSIC.json
- root cause: prior cross-task Isaac process reuse; R1 uses per-task worker isolation
- engineering fix: one fresh Isaac worker per task
- scientific method changed: False

STRICT MATCHING:

{task_block}

STATE PARITY:
- fields included in PASS hash: restorable `env.scene.get_state(is_relative=True)`
- fields recorded as audit hash: eef/gripper/joint/force policy observations, exposed force-controller debug fields
- fields excluded: RNG internals, hidden PhysX/controller/tactile buffers not exposed/restored by IsaacLab, renderer state
- unresolved matching risks: see P5S0A_STATE_HASH_SCOPE.md

OVERALL:
- tasks passed: {passed}
- tasks failed: {failed}
- primary classification: {classification}

NEXT:
- TRUE_MATCHED_DATASET_READY: {ready}
- proposed next run: {'P5-S0-A TRUE MATCHED DATASET protocol only' if ready == 'YES' else 'Fix failed task gate before dataset generation'}
{proposed}
"""
    (out / "P5S0A_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(
        f"""STATUS:
{status}
METHOD_CHANGE: NONE
ARTIFACTS:
{out}

FROZEN PROBE:
- implementation: P4-B common contact-frame shear
- task-specific parameters: False
- code hash: {sha256_file(P4_COLLECT)}

TASK1 FORENSIC:
- stalled stage: see TASK1_STALL_FORENSIC.json
- root cause: prior cross-task Isaac process reuse; R1 uses per-task worker isolation
- engineering fix: one fresh Isaac worker per task
- scientific method changed: False

STRICT MATCHING:

{task_block}

STATE PARITY:
- fields included in PASS hash: restorable `env.scene.get_state(is_relative=True)`
- fields recorded as audit hash: eef/gripper/joint/force policy observations, exposed force-controller debug fields
- fields excluded: RNG internals, hidden PhysX/controller/tactile buffers not exposed/restored by IsaacLab, renderer state
- unresolved matching risks: see P5S0A_STATE_HASH_SCOPE.md

OVERALL:
- tasks passed: {passed}
- tasks failed: {failed}
- primary classification: {classification}

NEXT:
- TRUE_MATCHED_DATASET_READY: {ready}
- proposed next run: {'P5-S0-A TRUE MATCHED DATASET protocol only' if ready == 'YES' else 'Fix failed task gate before dataset generation'}
""",
        flush=True,
    )


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def label_distribution_rows(branches: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for split in ["ALL", "TRAIN", "DEV", "TEST"]:
        for task in ["ALL"] + [str(t) for t in TASKS]:
            subset = [b for b in branches if (split == "ALL" or b.get("split") == split) and (task == "ALL" or int(b.get("task", -1)) == int(task))]
            if not subset:
                continue
            pos = sum(as_int(b, "full_task_success_y") for b in subset)
            mixed = 0
            for cid in sorted({b["context_id"] for b in subset}):
                ys = {as_int(b, "full_task_success_y") for b in subset if b["context_id"] == cid}
                mixed += int(ys == {0, 1})
            rows.append(
                {
                    "split": split,
                    "task": task,
                    "branches": len(subset),
                    "positive_labels": pos,
                    "negative_labels": len(subset) - pos,
                    "contexts_with_both_success_and_failure": mixed,
                    "success_fraction": pos / len(subset),
                }
            )
    return rows


def monotonicity_rows(branches: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for cid in sorted({b["context_id"] for b in branches}):
        ctx = sorted([b for b in branches if b["context_id"] == cid], key=lambda r: as_float(r, "requested_force_N"))
        violations = []
        for i, lo in enumerate(ctx):
            for hi in ctx[i + 1 :]:
                if as_int(lo, "full_task_success_y") == 1 and as_int(hi, "full_task_success_y") == 0:
                    violations.append(f"{as_float(lo, 'requested_force_N'):g}>{as_float(hi, 'requested_force_N'):g}")
        rows.append(
            {
                "context_id": cid,
                "task": ctx[0].get("task", "") if ctx else "",
                "split": ctx[0].get("split", "") if ctx else "",
                "hidden_friction_analysis_only": ctx[0].get("hidden_friction_analysis_only", "") if ctx else "",
                "force_outcomes": json.dumps([(as_float(b, "requested_force_N"), as_int(b, "full_task_success_y")) for b in ctx]),
                "monotonicity_violation": int(bool(violations)),
                "violation_pairs_lower_success_higher_failure": ";".join(violations),
            }
        )
    return rows


def measured_force_rows(branches: list[dict]) -> list[dict]:
    rows: list[dict] = []
    keys = sorted({(b.get("split", ""), int(b.get("task", -1)), as_float(b, "requested_force_N")) for b in branches})
    for split, task, force in keys:
        subset = [b for b in branches if b.get("split") == split and int(b.get("task", -1)) == task and abs(as_float(b, "requested_force_N") - force) < 1e-9]
        rows.append(
            {
                "split": split,
                "task": task,
                "requested_force_N": force,
                "branches": len(subset),
                "mean_measured_force_mean_N": float(np.mean([as_float(b, "measured_force_mean_N") for b in subset])) if subset else "",
                "mean_steady_state_N": float(np.mean([as_float(b, "steady_state_mean_N") for b in subset])) if subset else "",
                "mean_peak_force_N": float(np.mean([as_float(b, "measured_force_peak_N") for b in subset])) if subset else "",
                "mean_tracking_mae_N": float(np.mean([as_float(b, "force_tracking_mae_N") for b in subset])) if subset else "",
            }
        )
    return rows


def trajectory_delta(path_a: str, path_b: str, fields: list[str]) -> tuple[int, float, float]:
    if not path_a or not path_b or not Path(path_a).exists() or not Path(path_b).exists():
        return 0, float("nan"), float("nan")
    with Path(path_a).open() as fa, Path(path_b).open() as fb:
        ra = list(csv.DictReader(fa))
        rb = list(csv.DictReader(fb))
    n = min(len(ra), len(rb))
    if n == 0:
        return 0, float("nan"), float("nan")
    vals = []
    final_vals = []
    for i in range(n):
        for field in fields:
            vals.append(abs(as_float(ra[i], field, float("nan")) - as_float(rb[i], field, float("nan"))))
    for field in fields:
        final_vals.append(abs(as_float(ra[n - 1], field, float("nan")) - as_float(rb[n - 1], field, float("nan"))))
    clean = [v for v in vals if not np.isnan(v)]
    clean_final = [v for v in final_vals if not np.isnan(v)]
    return n, float(np.mean(clean)) if clean else float("nan"), float(np.mean(clean_final)) if clean_final else float("nan")


def replay_summary(replay: list[dict]) -> tuple[list[dict], str, int]:
    rows: list[dict] = []
    disagreement_tasks = set()
    for task in TASKS:
        task_rows = [r for r in replay if int(r.get("task", -1)) == task]
        if len(task_rows) < 2:
            rows.append({"task": task, "status": "MISSING_REPLAY"})
            continue
        a, b = task_rows[0], task_rows[1]
        outcome_disagree = int(as_int(a, "full_task_success_y") != as_int(b, "full_task_success_y"))
        if outcome_disagree:
            disagreement_tasks.add(task)
        force_n, force_mean_delta, force_final_delta = trajectory_delta(a.get("telemetry_path", ""), b.get("telemetry_path", ""), ["measured_force_N"])
        obj_n, obj_mean_delta, obj_final_delta = trajectory_delta(
            a.get("telemetry_path", ""),
            b.get("telemetry_path", ""),
            ["object_x_analysis_only", "object_y_analysis_only", "object_z_analysis_only"],
        )
        grip_n, grip_mean_delta, grip_final_delta = trajectory_delta(a.get("telemetry_path", ""), b.get("telemetry_path", ""), ["gripper_pos_0", "gripper_pos_1"])
        rows.append(
            {
                "task": task,
                "status": "OK",
                "context_id": a.get("context_id", ""),
                "requested_force_N": SAME_FORCE_REPLAY_FORCE,
                "replay_1_success": a.get("full_task_success_y", ""),
                "replay_2_success": b.get("full_task_success_y", ""),
                "outcome_disagreement": outcome_disagree,
                "episode_length_delta": abs(as_int(a, "episode_length") - as_int(b, "episode_length")),
                "mean_force_delta_N": abs(as_float(a, "measured_force_mean_N") - as_float(b, "measured_force_mean_N")),
                "steady_state_force_delta_N": abs(as_float(a, "steady_state_mean_N") - as_float(b, "steady_state_mean_N")),
                "trajectory_common_steps": min(force_n, obj_n, grip_n),
                "measured_force_trajectory_mean_abs_delta_N": force_mean_delta,
                "measured_force_trajectory_final_abs_delta_N": force_final_delta,
                "object_xyz_trajectory_mean_abs_delta_m": obj_mean_delta,
                "object_xyz_trajectory_final_abs_delta_m": obj_final_delta,
                "gripper_trajectory_mean_abs_delta": grip_mean_delta,
                "gripper_trajectory_final_abs_delta": grip_final_delta,
                "replay_1_telemetry": a.get("telemetry_path", ""),
                "replay_2_telemetry": b.get("telemetry_path", ""),
            }
        )
    md = "# P5-S0-A Same-Force Replay Diagnostic\n\n"
    for row in rows:
        md += f"task{row['task']}: status={row.get('status', 'OK')}, outcome_disagreement={row.get('outcome_disagreement', '')}, episode_length_delta={row.get('episode_length_delta', '')}, mean_force_delta_N={row.get('mean_force_delta_N', '')}\n"
    md += "\nReplay rows are diagnostics only and are not independent training labels.\n"
    return rows, md, len(disagreement_tasks)


def detect_context_leakage(contexts: list[dict]) -> bool:
    by_context: dict[str, set[str]] = {}
    for row in contexts:
        by_context.setdefault(row["context_id"], set()).add(row["split"])
    return any(len(splits) > 1 for splits in by_context.values())


def build_p5s0a_task_rows(contexts: list[dict], branches: list[dict], parity: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for task in TASKS:
        c = [r for r in contexts if int(r.get("task", -1)) == task]
        b = [r for r in branches if int(r.get("task", -1)) == task]
        p = [r for r in parity if int(r.get("task", -1)) == task and as_int(r, "is_replay_diagnostic") == 0]
        pos = sum(as_int(r, "full_task_success_y") for r in b)
        rows.append(
            {
                "task": task,
                "planned_contexts": len(FRICTION_VALUES) * len(CONTEXT_SEEDS),
                "completed_contexts": len(c),
                "probe_qualified_contexts": sum(as_int(r, "probe_qualified") for r in c),
                "strict_matched_contexts": sum(as_int(r, "strict_matched") for r in c),
                "planned_primary_branches": sum(as_int(r, "planned_primary_branches") for r in c),
                "completed_primary_branches": len(b),
                "primary_state_parity_rows": len(p),
                "primary_state_parity_passes": sum(as_int(r, "parity_pass") for r in p),
                "positive_labels": pos,
                "negative_labels": len(b) - pos,
            }
        )
    return rows


def write_p5s0a_final(
    out: Path,
    contexts: list[dict],
    branches: list[dict],
    parity: list[dict],
    replay: list[dict],
    results: list[dict],
    worker_records: list[dict],
) -> None:
    task_rows = build_p5s0a_task_rows(contexts, branches, parity)
    write_csv(out / "P5S0A_TASK_RESULTS.csv", task_rows)
    write_csv(out / "P5S0A_FORCE_LABEL_DISTRIBUTION.csv", label_distribution_rows(branches))
    mono_rows = monotonicity_rows(branches)
    write_csv(out / "P5S0A_MONOTONICITY_DIAGNOSTIC.csv", mono_rows)
    force_rows = measured_force_rows(branches)
    write_csv(out / "P5S0A_MEASURED_FORCE_DIAGNOSTIC.csv", force_rows)
    replay_diag_rows, replay_md, replay_disagreement_task_count = replay_summary(replay)
    write_csv(out / "P5S0A_SAME_FORCE_REPLAY.csv", replay)
    write_csv(out / "P5S0A_SAME_FORCE_REPLAY_SUMMARY.csv", replay_diag_rows)
    (out / "P5S0A_REPLAY_DIAGNOSTIC.md").write_text(replay_md, encoding="utf-8")

    primary_parity = [r for r in parity if as_int(r, "is_replay_diagnostic") == 0]
    planned_contexts = len(TASKS) * len(FRICTION_VALUES) * len(CONTEXT_SEEDS)
    planned_primary_branches = sum(as_int(r, "planned_primary_branches") for r in contexts)
    completed = len(contexts)
    qualified = sum(as_int(r, "probe_qualified") for r in contexts)
    strict = sum(as_int(r, "strict_matched") for r in contexts)
    positive = sum(as_int(r, "full_task_success_y") for r in branches)
    negative = len(branches) - positive
    mixed_contexts = sum(1 for cid in {b["context_id"] for b in branches} if {as_int(b, "full_task_success_y") for b in branches if b["context_id"] == cid} == {0, 1})
    mono_violations = sum(as_int(r, "monotonicity_violation") for r in mono_rows)
    task2_used = any(int(r.get("task", -1)) == 2 for r in contexts + branches)
    leakage = detect_context_leakage(contexts)
    strict_matching_valid = bool(primary_parity and len(primary_parity) == len(branches) and all(as_int(r, "parity_pass") == 1 for r in primary_parity))
    replay_stable = replay_disagreement_task_count < 2
    all_tasks_represented = all(any(int(r.get("task", -1)) == task for r in contexts) for task in TASKS)
    no_proxy_labels = all(r.get("label_source") == "TRUE_POST_PROBE_RESET_MATCHED_BRANCH" for r in branches)
    ready = bool(completed == planned_contexts and all_tasks_represented and strict_matching_valid and no_proxy_labels and not task2_used and not leakage and replay_stable and positive > 0 and negative > 0 and mixed_contexts >= 2 and qualified > 0)
    any_system_failure = any(int(w.get("returncode", 0)) != 0 and w.get("timeout") for w in worker_records)
    if ready:
        status = "PASS"
        classification = "P5S0A_TRUE_MATCHED_DATASET_QUALIFIED"
    elif not replay_stable:
        status = "FAIL"
        classification = "P5S0A_SIMULATOR_REPLAY_INSTABILITY"
    elif any_system_failure:
        status = "FAIL"
        classification = "P5S0A_INCONCLUSIVE_SYSTEM_FAILURE"
    elif completed > 0 and strict_matching_valid and positive > 0 and negative > 0:
        status = "PARTIAL"
        classification = "P5S0A_DATASET_PARTIALLY_QUALIFIED"
    else:
        status = "FAIL"
        classification = "P5S0A_DATASET_NOT_QUALIFIED"

    readiness = {
        "Q2F_MODEL_TRAINING_READY": ready,
        "all_four_tasks_represented": all_tasks_represented,
        "planned_contexts": planned_contexts,
        "completed_contexts": completed,
        "probe_qualified_contexts": qualified,
        "strict_matched_contexts": strict,
        "planned_primary_branches": planned_primary_branches,
        "completed_primary_branches": len(branches),
        "primary_state_parity_rate": (sum(as_int(r, "parity_pass") for r in primary_parity) / len(primary_parity)) if primary_parity else 0.0,
        "proxy_labels_used": not no_proxy_labels,
        "canonical_force_frontier_used_as_label": False,
        "task2_used": task2_used,
        "cross_split_context_leakage": leakage,
        "replay_stability_acceptable": replay_stable,
        "same_force_replay_disagreement_tasks": replay_disagreement_task_count,
        "positive_labels": positive,
        "negative_labels": negative,
        "contexts_with_both_success_and_failure": mixed_contexts,
        "monotonicity_violation_contexts": mono_violations,
        "primary_classification": classification,
    }
    write_json(out / "P5S0A_DATASET_READINESS.json", readiness)
    write_json(out / "P5S0A_FINAL_VERDICT.json", {"STATUS": status, "METHOD_CHANGE": "NONE", "PRIMARY_CLASSIFICATION": classification, **readiness})

    branch_split_counts = {s: sum(1 for r in branches if r.get("split") == s) for s in ["TRAIN", "DEV", "TEST"]}
    offgrid_executed = sorted({as_float(r, "requested_force_N") for r in branches if r.get("force_role") == "EVAL_HELDOUT_FORCE"})
    force_summary = {
        "mean_tracking_mae_N": float(np.mean([as_float(r, "force_tracking_mae_N") for r in branches])) if branches else 0.0,
        "mean_steady_state_N": float(np.mean([as_float(r, "steady_state_mean_N") for r in branches])) if branches else 0.0,
        "mean_peak_N": float(np.mean([as_float(r, "measured_force_peak_N") for r in branches])) if branches else 0.0,
    }
    per_task_label = {str(t): {"positive": sum(as_int(b, "full_task_success_y") for b in branches if int(b.get("task", -1)) == t), "negative": sum(1 - as_int(b, "full_task_success_y") for b in branches if int(b.get("task", -1)) == t)} for t in TASKS}
    replay_lines = "\n".join([f"- task{r['task']}: disagreement={r.get('outcome_disagreement', '')}, length_delta={r.get('episode_length_delta', '')}" for r in replay_diag_rows])

    summary = f"""STATUS:
{status}
METHOD_CHANGE: NONE
ARTIFACTS:
{out}

SCOPE:
- Tasks: {TASKS}
- Task2 used: {task2_used}
- Probe: P4-B common contact-frame shear
- Fresh worker per task: True

PHYSICS:
- Friction values: {FRICTION_VALUES}
- TRAIN friction: {TRAIN_FRICTIONS}
- DEV friction: {DEV_FRICTION}
- TEST friction: {TEST_FRICTION}

CONTEXTS:
- Planned: {planned_contexts}
- Completed: {completed}
- Probe qualified: {qualified}
- Strict matched: {strict}
- Per task: {task_rows}

BRANCHES:
- Planned: {planned_primary_branches}
- Completed: {len(branches)}
- State parity: {sum(as_int(r, 'parity_pass') for r in primary_parity)}/{len(primary_parity)}
- TRAIN branches: {branch_split_counts['TRAIN']}
- DEV branches: {branch_split_counts['DEV']}
- TEST branches: {branch_split_counts['TEST']}

FORCES:
- TRAIN force values: {TRAIN_FORCES}
- Held-out force values: {HELDOUT_FORCE_VALUES}
- Requested off-grid targets executed: {offgrid_executed}
- Measured-force tracking summary: {force_summary}

SAME-FORCE REPLAY:
{replay_lines}
- Any outcome disagreement: {replay_disagreement_task_count > 0}

LABELS:
- Positive: {positive}
- Negative: {negative}
- Per-task positive/negative: {per_task_label}
- Contexts with both success and failure: {mixed_contexts}
- Monotonicity violations: {mono_violations}

DATA VALIDITY:
- Proxy labels used: {not no_proxy_labels}
- Canonical F* used as label: False
- Cross-split context leakage: {leakage}
- Strict matching valid: {strict_matching_valid}
- Replay stability acceptable: {replay_stable}

PRIMARY_CLASSIFICATION:
{classification}

SCIENTIFIC_INTERPRETATION:
1. This run generated true matched probe/force/outcome data only; no model was trained.
2. Friction is recorded solely as analysis-only simulator configuration and split control.
3. Off-grid forces are development interpolation labels, not controller-resolution claims.
4. The dataset is ready for Q2F comparison only if the readiness gate above is true.

NEXT:
- Q2F_MODEL_TRAINING_READY: {'YES' if ready else 'NO'}
- Recommended next experiment:
  P5-S0-B TRUE-MATCHED Q2F MODEL COMPARISON
"""
    (out / "P5S0A_FINAL_REPORT.md").write_text("# P5-S0-A Final Report\n\n" + summary, encoding="utf-8")
    print(summary, flush=True)


def orchestrator_main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    write_static_artifacts(OUT)
    worker_records: list[dict] = []

    for task in TASKS:
        try:
            rec = launch_worker(OUT, task)
        except subprocess.TimeoutExpired as exc:
            rec = {
                "task": task,
                "worker_pid": "",
                "process_start_utc": "",
                "returncode": 124,
                "elapsed_wall_s": TIMEOUTS_S["worker"],
                "log": str(OUT / "logs" / f"task{task}_worker.log"),
                "timeout": True,
            }
            write_json(OUT / f"task{task}" / "error.json", {"task": task, "stalled_stage": "GPU_OR_SIMULATION_STALL", "error": repr(exc), "trace": ""})
        worker_records.append(rec)
        write_json(OUT / "P5S0A_WORKER_RECORDS.json", worker_records)

    contexts, branches, parity, replay, results = read_task_rows(OUT)
    write_csv(OUT / "P5S0A_CONTEXT_MANIFEST.csv", contexts)
    write_csv(OUT / "P5S0A_BRANCH_MANIFEST.csv", branches)
    write_csv(OUT / "P5S0A_STATE_PARITY.csv", parity)
    write_p5s0a_final(OUT, contexts, branches, parity, replay, results, worker_records)
    return 0


def main() -> int:
    if os.environ.get("P5S0A_WORKER") == "1":
        return worker_main()
    return orchestrator_main()


if __name__ == "__main__":
    raise SystemExit(main())
