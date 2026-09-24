#!/usr/bin/env python3
"""P5-S0-R1 four-task strict matched branching gate.

Data-generation validity only. No model training or force-sufficiency scoring.

The orchestrator runs one Isaac worker process per task to avoid cross-task Kit
environment reuse. Each worker executes one frozen P4-B probe exactly once,
saves the post-probe scene state, restores that same state before two downstream
force branches, and records state parity.
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
FORCES = [3.0, 6.0]
SEED = 3000
FRICTION = 0.6443245385975112

TIMEOUTS_S = {
    "probe": 180.0,
    "snapshot": 20.0,
    "branch_restore": 20.0,
    "full_task_rollout": 120.0,
    "worker": 480.0,
}


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


OUT = Path(os.environ.get("P5S0R1_OUT", RESULTS_ROOT / f"p5s0r1_four_task_strict_matching_{now_tag()}"))


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


class StageLogger:
    def __init__(self, out_dir: Path, task: int, seed: int, dt: float | None = None):
        self.path = out_dir / "P5S0R1_STAGE_LOG.csv"
        self.task = task
        self.seed = seed
        self.t0 = time.time()
        self.dt = dt

    def emit(self, stage: str, env=None, p4=None, reason: str = "", robot_phase: str = "") -> None:
        snap = telemetry_snapshot(env, p4, self.dt) if env is not None and p4 is not None else {}
        row = {
            "task": self.task,
            "seed": self.seed,
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
        print(f"[P5S0R1] task={self.task} stage={stage} elapsed={row['elapsed_wall_s']} reason={reason}", flush=True)


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
    os.environ["P4_OUT"] = str(OUT / "P5S0R1_FROZEN_P4B_IMPORT")
    os.environ["P4_RESUME"] = "0"
    old_stdout, old_stderr = sys.stdout, sys.stderr
    spec = importlib.util.spec_from_file_location(f"p5s0r1_p4b_task{task_id}", P4_COLLECT)
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


def downstream_branch(env, p4, *, task_id: int, force: float, branch_label: str, context_id: str, dt: float, logger: StageLogger) -> dict:
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
                obj_dz = float(obj_p[2] - obj0_w[2])
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
    rec = {
        "branch_id": f"{context_id}_{branch_label}_F{force:g}",
        "context_id": context_id,
        "task": task_id,
        "requested_force_N": float(force),
        "measured_force_trajectory_json": json.dumps(force_samples),
        "measured_force_mean_N": measured_mean,
        "measured_force_peak_N": float(peak_force),
        "steady_state_mean_N": measured_mean,
        "force_tracking_error_N": measured_mean - float(force),
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
        "label_source": "TRUE_POST_PROBE_RESET_MATCHED_BRANCH",
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

    task_id = int(os.environ["P5S0R1_TASK_ID"])
    task_dir = OUT / f"task{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    stage_logger = StageLogger(OUT, task_id, SEED)
    stage_logger.emit("RESET_STARTED", reason="worker_start")

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
        stage_logger.dt = dt
        stage_logger.emit("RESET_COMPLETE", env, p4, robot_phase="env_ready")

        context_id = f"p5s0r1_t{task_id}_s{SEED}_mu{FRICTION:.6f}"
        stage_logger.emit("VLA_APPROACH_STARTED", env, p4, robot_phase="approach")
        stage_logger.emit("CONTACT_ESTABLISHMENT_STARTED", env, p4, robot_phase="preload")
        with Timeout(TIMEOUTS_S["probe"], "PROBE"):
            probe_rows, probe_rec = p4.run_probe_episode(env, seed_idx=SEED, mu=FRICTION, trial_id=context_id, dt=dt)

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

        raw_dir = OUT / "P5S0R1_RAW_TELEMETRY"
        write_csv(raw_dir / f"{context_id}_probe_timesteps.csv", [asdict(r) for r in probe_rows])

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

        context_row = {
            "context_id": context_id,
            "task": task_id,
            "task_instruction": TASK_INSTRUCTIONS[task_id],
            "hidden_friction_analysis_only": FRICTION,
            "seed": SEED,
            "probe_implementation": "P4-B common contact-frame shear",
            "probe_qualified": probe_qualified,
            "contact_retained": quality["contact_retained"],
            "drop": quality["drop"],
            "disturbance": quality["disturbance"],
            "return_completed": quality["return_completed"],
            "informative": quality["informative"],
            "safe": quality["safe"],
            "probe_quality_derivation": "derived_from_existing_P4B_fields_without_changing_probe",
            "stop_reason": probe_rec.get("stop_trigger", ""),
            "actual_displacement_mm": float(probe_rec.get("actual_probe_displacement_mm", np.nan)),
            "post_probe_state_hash": hq,
            "post_probe_audit_hash": hq_audit,
            "status": context_status,
            "probe_rerun_between_branches": False,
        }
        write_csv(task_dir / "context.csv", [context_row])

        branch_rows: list[dict] = []
        parity_rows: list[dict] = []
        if not probe_qualified:
            write_csv(task_dir / "branches.csv", branch_rows)
            write_csv(task_dir / "parity.csv", parity_rows)
            stage_logger.emit("CONTEXT_COMPLETE", env, p4, reason=context_status)
            write_json(task_dir / "result.json", {"task": task_id, "pass": False, "reason": context_status})
            return 0

        for label, force in zip(["BRANCH_A", "BRANCH_B"], FORCES):
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
            br = downstream_branch(env, p4, task_id=task_id, force=force, branch_label=label, context_id=context_id, dt=dt, logger=stage_logger)
            br["post_probe_state_hash"] = hq
            branch_rows.append(br)
            write_csv(raw_dir / f"{br['branch_id']}_force_trace.csv", [{"i": i, "measured_force_N": x} for i, x in enumerate(json.loads(br["measured_force_trajectory_json"]))])

        write_csv(task_dir / "branches.csv", branch_rows)
        write_csv(task_dir / "parity.csv", parity_rows)
        pass_task = bool(
            probe_qualified
            and len(branch_rows) == 2
            and len(parity_rows) == 2
            and all(int(r["parity_pass"]) == 1 for r in parity_rows)
        )
        stage_logger.emit("CONTEXT_COMPLETE", env, p4, reason="PASS" if pass_task else "FAIL")
        write_json(
            task_dir / "result.json",
            {
                "task": task_id,
                "pass": pass_task,
                "probe_qualified": bool(probe_qualified),
                "post_probe_hash": hq,
                "branches": len(branch_rows),
                "parity_passes": sum(int(r["parity_pass"]) for r in parity_rows),
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
            "P5S0R1_OUT": str(out),
            "P5S0R1_WORKER": "1",
            "P5S0R1_TASK_ID": str(task),
        }
    )
    return env


def launch_worker(out: Path, task: int) -> dict:
    log_path = out / "logs" / f"task{task}_worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ISAAC_PY), "-u", str(Path(__file__).resolve())]
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO, env=worker_env(out, task), stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUTS_S["worker"])
    return {"task": task, "returncode": proc.returncode, "elapsed_wall_s": time.time() - start, "log": str(log_path)}


def read_task_rows(out: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    contexts: list[dict] = []
    branches: list[dict] = []
    parity: list[dict] = []
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
        result = {"task": task, "pass": False, "reason": "missing_result"}
        if (task_dir / "result.json").exists():
            result.update(json.loads((task_dir / "result.json").read_text()))
        if (task_dir / "error.json").exists():
            result.update(json.loads((task_dir / "error.json").read_text()))
        results.append(result)
    return contexts, branches, parity, results


def write_static_artifacts(out: Path) -> None:
    protocol = {
        "name": "P5-S0-R1 Four-Task Strict Matched Branching Gate",
        "method_change": "NONE",
        "tasks": TASKS,
        "task2_used": False,
        "contexts_per_task": 1,
        "force_branches_per_context": 2,
        "forces_N": FORCES,
        "hidden_friction_analysis_only": FRICTION,
        "seed": SEED,
        "probe": "P4-B common contact-frame shear",
        "no_model_training": True,
        "timeouts_s": TIMEOUTS_S,
        "process_isolation_fix": "one Isaac worker process per task",
    }
    write_json(out / "P5S0R1_PROTOCOL.json", protocol)
    write_json(
        out / "P5S0R1_CODE_HASH.txt",
        {
            "git_commit": git_commit(),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "p4b_collect_sha256": sha256_file(P4_COLLECT),
        },
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
        "code_hash": sha256_file(P4_COLLECT),
    }
    write_json(out / "P5S0R1_FROZEN_PROBE.json", frozen_probe)
    (out / "P5S0R1_STATE_HASH_SCOPE.md").write_text(
        """# P5-S0-R1 State Hash Scope

Included in the PASS/FAIL reference and post-restore hash:
- `env.scene.get_state(is_relative=True)` recursively serialized, including IsaacLab scene state available for `env.reset_to`.

Recorded as non-PASS audit hashes:
- Recomputed policy observation fields: `eef_pose`, `gripper_pos`, `arm_joint_pos`, `gripper_net_force`.
- Force/action debug fields when exposed: `f_sq_meas`, `f_sq_pred`, `f_sq_meas_raw`, `d_pred`, `d_cmd`, `d_actual`.

Recorded in stage logs but not necessarily hash-restored as independent simulator internals:
- object pose/quaternion from scene bodies,
- bilateral contact flag from left/right local normal force,
- gripper state from policy observation,
- environment step and derived simulation time.

Excluded or unresolved:
- Python, Torch, NumPy, PhysX, and task RNG internal states after the post-probe point.
- Internal force-controller integrator/state not present in `action_manager.debug_info`.
- Tactile sensor history buffers beyond the recomputed observation tensors exposed by the policy group.
- Renderer/GPU/RTX state and asynchronous sensor pipeline internals.
- Wall-clock time and log process state.

Matching risk:
`env.reset_to` is treated as authoritative for restorable simulator scene state. Recomputed observations and action debug values can differ immediately after restore even when scene state matches; those audit hashes are reported but do not define the R1 PASS parity criterion. Fields that IsaacLab does not expose or restore are documented above and remain unresolved matching risks for future hardening.
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
        classification = "P5S0R1_FOUR_TASK_STRICT_MATCHED_BRANCHING_QUALIFIED"
        status = "PASS"
        ready = "YES"
    elif passed:
        classification = "P5S0R1_STRICT_MATCHING_PARTIALLY_QUALIFIED"
        status = "PARTIAL"
        ready = "NO"
    else:
        classification = "P5S0R1_STRICT_MATCHING_NOT_QUALIFIED"
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
    write_json(out / "P5S0R1_FINAL_VERDICT.json", verdict)

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
    report = f"""# P5-S0-R1 Final Report

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
- unresolved matching risks: see P5S0R1_STATE_HASH_SCOPE.md

OVERALL:
- tasks passed: {passed}
- tasks failed: {failed}
- primary classification: {classification}

NEXT:
- TRUE_MATCHED_DATASET_READY: {ready}
- proposed next run: {'P5-S0-A TRUE MATCHED DATASET protocol only' if ready == 'YES' else 'Fix failed task gate before dataset generation'}
{proposed}
"""
    (out / "P5S0R1_FINAL_REPORT.md").write_text(report, encoding="utf-8")
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
- unresolved matching risks: see P5S0R1_STATE_HASH_SCOPE.md

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


def orchestrator_main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    write_static_artifacts(OUT)
    worker_records: list[dict] = []

    # Task1 first for forensic localization, then remaining tasks.
    for task in [1, 0, 5, 6]:
        try:
            rec = launch_worker(OUT, task)
        except subprocess.TimeoutExpired as exc:
            rec = {"task": task, "returncode": 124, "elapsed_wall_s": TIMEOUTS_S["worker"], "log": str(OUT / "logs" / f"task{task}_worker.log"), "timeout": True}
            write_json(OUT / f"task{task}" / "error.json", {"task": task, "stalled_stage": "GPU_OR_SIMULATION_STALL", "error": repr(exc), "trace": ""})
        worker_records.append(rec)
        write_json(OUT / "P5S0R1_WORKER_RECORDS.json", worker_records)

    contexts, branches, parity, results = read_task_rows(OUT)
    write_csv(OUT / "P5S0R1_CONTEXT_MANIFEST.csv", contexts)
    write_csv(OUT / "P5S0R1_BRANCH_MANIFEST.csv", branches)
    write_csv(OUT / "P5S0R1_STATE_PARITY.csv", parity)
    task_rows = build_task_result_rows(contexts, branches, parity, results)
    write_csv(OUT / "P5S0R1_TASK_RESULTS.csv", task_rows)
    write_task1_forensic(OUT, worker_records, results)
    write_final(OUT, task_rows, contexts, branches, parity)
    return 0


def main() -> int:
    if os.environ.get("P5S0R1_WORKER") == "1":
        return worker_main()
    return orchestrator_main()


if __name__ == "__main__":
    raise SystemExit(main())
