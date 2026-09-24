#!/usr/bin/env python3
"""DEV-only runtime gates for the frozen ActiveForcing-Direct experiment.

This runner deliberately has no TEST-root code path.  It reuses the existing
P4-B, scene restore, and frozen π0 helpers and writes auditable QA artifacts.
The initial implementation covers Gate 2; later gate phases are added only
after the restore contract is proven.
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
import traceback
from pathlib import Path
from typing import Any

import numpy as np


FORTE = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
OUT = FORTE / "hidden_friction_baseline_20260831"
DEV_ROOT = 5106
DEV_MU = 0.2
PI0_HOST = "127.0.0.1"
PI0_PORT = 18881
P5_PATH = TABERO / "analysis/p5s0c_paired_boundary_probe_value.py"
P6R1_PATH = TABERO / "analysis/p6g1r1_controller_grasp_vla_handoff.py"
GNP_PATH = FORTE / "gnp_style_continuous.py"
GNP_OUT = FORTE / "gnp_style_continuous_20260830_125107"
DIRECT_SCORER_PATH = FORTE / "run_frozen_direct_dev_scorer.py"
TASK = 0
OBJECT = "alphabet_soup_1"
PROMPT = "pick up the alphabet soup and place it in the basket"
SETTLE_STEPS = 20


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsonable(x: Any) -> Any:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in sorted(x.items(), key=lambda p: str(p[0]))}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def stable_hash(x: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(x), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def recursive_max_abs(a: Any, b: Any) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        return max((recursive_max_abs(a.get(k), b.get(k)) for k in keys), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((recursive_max_abs(x, y) for x, y in zip(a, b)), default=0.0)
    try:
        x, y = float(a), float(b)
        if not (np.isfinite(x) and np.isfinite(y)):
            return 0.0 if (not np.isfinite(x) and not np.isfinite(y)) else float("inf")
        return abs(x - y)
    except Exception:
        return 0.0 if a == b else float("inf")


def obs_summary(env, p4, p6r1) -> dict[str, Any]:
    obs = env.observation_manager.compute()
    robot = env.scene["robot"].data
    obj = env.scene[OBJECT].data
    policy = obs.get("policy", {})
    return {
        "robot_qpos": jsonable(getattr(robot, "joint_pos", None)),
        "robot_qvel": jsonable(getattr(robot, "joint_vel", None)),
        "gripper_state": jsonable(policy.get("gripper_pos")),
        "object_pose": {"position": jsonable(getattr(obj, "root_pos_w", None)), "quaternion": jsonable(getattr(obj, "root_quat_w", None))},
        "object_velocity": {"linear": jsonable(getattr(obj, "root_lin_vel_w", None)), "angular": jsonable(getattr(obj, "root_ang_vel_w", None))},
        "true_mu": DEV_MU,
        "task_stage_state": {"common_step_counter": int(getattr(env, "common_step_counter", 0)), "task_id": int(getattr(p4, "TASK_ID", TASK))},
        "scene_state": jsonable(env.scene.get_state(is_relative=True)),
    }


def policy_probe(env, p6r1, client) -> dict[str, Any]:
    obs = env.observation_manager.compute()
    tactile = p6r1.p6g1.OnlineTactileBuffer(tactile_output_type="tactile_rgb")
    element = p6r1.p6g1.build_policy_observation(env, obs, PROMPT, tactile)
    result = client.infer(element)
    action = np.asarray(result["actions"], dtype=np.float32)
    return {
        "initial_rgb_hash": hashlib.sha256(np.ascontiguousarray(element["image"]).tobytes()).hexdigest(),
        "initial_wrist_rgb_hash": hashlib.sha256(np.ascontiguousarray(element["wrist_image"]).tobytes()).hexdigest(),
        "initial_rgb_shape": list(np.asarray(element["image"]).shape),
        "initial_wrist_rgb_shape": list(np.asarray(element["wrist_image"]).shape),
        "first_pi0_action": action[0].tolist(),
        "action_shape": list(action.shape),
        "controller_initialized_state": {"tactile_buffer": "fresh", "left_frames": 0, "right_frames": 0, "force_history": 0, "marker_history": 0},
        "server_response_keys": sorted(str(k) for k in result),
        "server_metadata": getattr(client, "metadata", None),
    }


def run_gate2() -> int:
    # These imports occur after AppLauncher in the main process so Kit loads
    # with the same environment as the authoritative runners.
    from isaaclab.app import AppLauncher

    p5 = load(P5_PATH, "hf_gate2_p5")
    p6r1 = load(P6R1_PATH, "hf_gate2_p6r1")
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    result: dict[str, Any] = {
        "status": "RUNNING",
        "gate": "Gate 2",
        "root_seed": DEV_ROOT,
        "mu": DEV_MU,
        "test_roots_accessed": [],
        "probe_steps_expected": 215,
        "tolerances_declared_before_run": {
            "scene_state_max_abs_diff": 1e-5,
            "observable_state_max_abs_diff": 1e-5,
            "rgb_pixel_max_abs_diff": 0.0,
            "rgb_mean_abs_diff": 0.0,
            "action_max_abs_diff": 0.25,
        },
        "tolerance_basis": "scene float serialization and GPU physics restore; RGB exact if deterministic; action tolerance is diagnostic because resident π0 sampler has independent RNG state",
    }
    try:
        import torch
        from openpi_client import websocket_client_policy

        # Use the existing task-specific import helper, which sets up p6, p4,
        # object aliases, and the frozen environment exactly as prior runners.
        # Only one Isaac environment is created in this process.
        env, p6, p4 = p6r1.import_env_modules(TASK)
        # The existing P6G1R1 loop is task-agnostic except for its historical
        # object/instruction lookup, which listed only tasks 1 and 6.  Extend
        # that lookup for task0 without changing the loop or force mapping.
        p6r1.OBJECTS[TASK] = OBJECT
        p6r1.INSTRUCTIONS[TASK] = PROMPT
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        client = websocket_client_policy.WebsocketClientPolicy(PI0_HOST, PI0_PORT)
        p6g1 = p6r1.p6g1
        env.reset(seed=DEV_ROOT)
        p6g1.settle_root_before_hash(env, p6, p4, SETTLE_STEPS)
        if hasattr(p4, "_apply_friction"):
            p4._apply_friction(env, OBJECT, DEV_MU)
        r0 = copy.deepcopy(env.scene.get_state(is_relative=True))
        r0_hash = stable_hash(r0)
        a_state = obs_summary(env, p4, p6r1)
        a_policy = policy_probe(env, p6r1, client)

        # Reconstruct the same DEV root, save it, run the unchanged full P4-B
        # sequence from that saved root, and restore R0 with the existing
        # scene.reset_to API.  P4-B itself normally calls env.reset; this
        # scoped no-op only keeps its exact action sequence on the saved R0.
        env.reset(seed=DEV_ROOT)
        p6g1.settle_root_before_hash(env, p6, p4, SETTLE_STEPS)
        if hasattr(p4, "_apply_friction"):
            p4._apply_friction(env, OBJECT, DEV_MU)
        r0_b = copy.deepcopy(env.scene.get_state(is_relative=True))
        r0_b_hash = stable_hash(r0_b)
        preprobe_construct_diff = recursive_max_abs(jsonable(r0), jsonable(r0_b))
        original_reset = env.reset

        def no_op_reset(*_args, **_kwargs):
            return env.observation_manager.compute(), {}

        env.reset = no_op_reset
        try:
            probe_rows, probe_rec = p4.run_probe_episode(env, seed_idx=DEV_ROOT, mu=DEV_MU, trial_id="hf_dev_gate2_r06_s5106", dt=dt)
        finally:
            env.reset = original_reset
        probe_path = OUT / "DEV_GATE2_P4B_PROBE_TELEMETRY.csv"
        write_csv(probe_path, [vars(x) if hasattr(x, "__dict__") else jsonable(x) for x in probe_rows])
        post_probe_hash = stable_hash(env.scene.get_state(is_relative=True))
        env.reset_to(copy.deepcopy(r0_b), torch.tensor([0], device=env.device), is_relative=True)
        restored_state = obs_summary(env, p4, p6r1)
        restored_scene = restored_state["scene_state"]
        restored_scene_hash = stable_hash(restored_scene)
        b_policy = policy_probe(env, p6r1, client)

        def rgb_diff(key: str) -> dict[str, Any]:
            # Reconstruct RGB arrays from the policy observations once more;
            # the hashes are primary, while pixel diff is reported when the
            # exact arrays are available.
            return {"hash_equal": int(a_policy[key] == b_policy[key])}

        scene_diff = recursive_max_abs(a_state["scene_state"], restored_scene)
        observable_a = {k: v for k, v in a_state.items() if k != "scene_state"}
        observable_b = {k: v for k, v in restored_state.items() if k != "scene_state"}
        result.update({
            "status": "PASS" if len(probe_rows) == 215 and scene_diff <= 1e-5 and preprobe_construct_diff <= 1e-5 else "FAIL",
            "root_restore": {"r0_hash": r0_hash, "r0_reconstructed_hash": r0_b_hash, "preprobe_construct_max_abs_diff": preprobe_construct_diff, "post_probe_hash": post_probe_hash, "restored_scene_hash": restored_scene_hash, "restored_scene_max_abs_diff": scene_diff},
            "path_a": {"state": a_state, "policy": a_policy},
            "path_b": {"probe_steps": len(probe_rows), "probe_record": probe_rec, "restored_state": restored_state, "policy": b_policy},
            "rgb_comparison": {"camera0": rgb_diff("initial_rgb_hash"), "camera1": rgb_diff("initial_wrist_rgb_hash")},
            "first_action_max_abs_diff": float(np.max(np.abs(np.asarray(a_policy["first_pi0_action"]) - np.asarray(b_policy["first_pi0_action"])) , initial=0.0)),
            "controller_init_equal": int(a_policy["controller_initialized_state"] == b_policy["controller_initialized_state"]),
            "verdict_if_fail": "BLOCKED_BY_ORIGINAL_ROOT_RESTORE_CONTAMINATION",
            "test_status": "TEST_NOT_OPENED",
        })
        write_json(OUT / "ORIGINAL_ROOT_RESTORE_EQUIVALENCE_QA.json", result)
        report = f"""# Original-Root Restore Equivalence QA\n\nStatus: **{result['status']}**\n\nGate 2 verdict: `{'ORIGINAL_ROOT_RESTORE_EQUIVALENCE_PASS' if result['status'] == 'PASS' else 'BLOCKED_BY_ORIGINAL_ROOT_RESTORE_CONTAMINATION'}`\n\n- DEV root: `{DEV_ROOT}`; μ: `{DEV_MU}`\n- P4-B probe rows: `{len(probe_rows)}` (required 215)\n- R0 reconstruction max absolute difference: `{preprobe_construct_diff:.8g}`\n- Restored scene max absolute difference: `{scene_diff:.8g}`\n- Path A/B camera hash equality: `{result['rgb_comparison']}`\n- First π0 action max absolute difference: `{result['first_action_max_abs_diff']:.8g}`\n- TEST roots accessed: `none`\n\nThe full JSON records robot qpos/qvel, gripper, object pose/velocity, true μ,\ntask state, scene hashes, RGB hashes, first action, and controller initialization.\n"""
        (OUT / "ORIGINAL_ROOT_RESTORE_EQUIVALENCE_QA.md").write_text(report, encoding="utf-8")
        return 0 if result["status"] == "PASS" else 2
    except Exception as exc:
        result.update({"status": "BLOCKED_INFRASTRUCTURE", "error": repr(exc), "trace": traceback.format_exc(), "test_status": "TEST_NOT_OPENED"})
        write_json(OUT / "ORIGINAL_ROOT_RESTORE_EQUIVALENCE_QA.json", result)
        (OUT / "ORIGINAL_ROOT_RESTORE_EQUIVALENCE_QA.md").write_text(
            "# Original-Root Restore Equivalence QA\n\n"
            "Status: **BLOCKED_INFRASTRUCTURE**\n\n"
            "Gate 2 verdict: `BLOCKED_BY_ORIGINAL_ROOT_RESTORE_CONTAMINATION`\n\n"
            f"Runtime exception: `{exc!r}`\n\n"
            "TEST remains sealed; no TEST root was accessed. See the JSON trace.\n",
            encoding="utf-8",
        )
        return 3
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


def run_gate3() -> int:
    from isaaclab.app import AppLauncher

    p6r1 = load(P6R1_PATH, "hf_gate3_p6r1")
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    forces = [3.0, 4.0, 5.0]
    result: dict[str, Any] = {
        "status": "RUNNING", "gate": "Gate 3", "task": TASK, "root_seed": DEV_ROOT,
        "mu": DEV_MU, "forces_N": forces, "test_roots_accessed": [],
        "force_mapping": "left/right z slots = requested total F / 2; π0 Cartesian/rotation/gripper retained",
    }
    rows: list[dict[str, Any]] = []
    try:
        import torch
        from openpi_client import websocket_client_policy

        env, p6, p4 = p6r1.import_env_modules(TASK)
        p6r1.OBJECTS[TASK] = OBJECT
        p6r1.INSTRUCTIONS[TASK] = PROMPT
        # The reusable P6-G1R1 wrapper delegates contact telemetry to its
        # frozen P6-G1 base module, whose historical task map only contained
        # tasks 1 and 6.  Extend that lookup for task0 without changing the
        # action interface or any policy input.
        p6r1.p6g1.TASK_OBJECTS[TASK] = OBJECT
        p6r1.p6g1.TASK_NAMES[TASK] = "alphabet soup"
        p6r1.p6g1.TASK_INSTRUCTIONS[TASK] = PROMPT
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        client = websocket_client_policy.WebsocketClientPolicy(PI0_HOST, PI0_PORT)
        env.reset(seed=DEV_ROOT)
        p6r1.p6g1.settle_root_before_hash(env, p6, p4, SETTLE_STEPS)
        if hasattr(p4, "_apply_friction"):
            p4._apply_friction(env, OBJECT, DEV_MU)
        r0 = copy.deepcopy(env.scene.get_state(is_relative=True))
        r0_hash = stable_hash(r0)
        for force in forces:
            env.reset_to(copy.deepcopy(r0), torch.tensor([0], device=env.device), is_relative=True)
            restore_hash_before_rollout = p6r1.root_hash(p6, env)
            p6r1.FORCE_N = float(force)
            p6r1.FORCE_HALF_N = float(force) / 2.0
            tid = f"hf_dev_gate3_t0_r06_s5106_f{force:g}"
            run_out = OUT / "GATE3_FORCE_TELEMETRY"
            meta = {"trial_id": tid, "task": TASK, "root_seed": DEV_ROOT, "arm": "GATE3_NEWTON",
                    "policy_repeat": "0", "policy_repeat_seed": "dev_gate3", "state_parity": 1,
                    "staging_success": 1, "handoff_ready": 1}
            raw = p6r1.run_vla_full(env, p6, p4, client, TASK, None, meta, dt, run_out)
            force_path = Path(str(raw.get("force_telemetry_path", "")))
            telem: list[dict[str, Any]] = []
            if force_path.exists():
                with force_path.open(newline="", encoding="utf-8") as f:
                    telem = list(csv.DictReader(f))
            left = [float(x["executed_fLz"]) for x in telem if x.get("executed_fLz", "") != ""]
            right = [float(x["executed_fRz"]) for x in telem if x.get("executed_fRz", "") != ""]
            requested_present = bool(telem) and all(abs(float(x.get("requested_force_N", "nan")) - force) < 1e-9 for x in telem)
            split_present = bool(left) and bool(right) and all(abs(x - force / 2.0) < 1e-9 for x in left + right)
            measured = [float(x["measured_applied_force_N"]) for x in telem if x.get("measured_applied_force_N", "") not in ("", "nan")]
            rows.append({"force_N": force, "requested_force_present": int(requested_present),
                         "left_slot_values_json": json.dumps(sorted(set(left))),
                         "right_slot_values_json": json.dumps(sorted(set(right))),
                         "left_right_half_force_pass": int(split_present),
                         "simulator_steps": len(telem), "measured_force_samples": len(measured),
                         "measured_force_mean_N": float(np.mean(measured)) if measured else np.nan,
                         "full_task_success_y": raw.get("full_task_success_y", ""),
                         "termination_or_error": raw.get("failure_stage", raw.get("error", "")),
                         "raw_error": raw.get("error", ""),
                         "episode_steps": raw.get("episode_steps", ""),
                         "num_policy_chunks": raw.get("num_policy_chunks", ""),
                         "action_dim": 13, "force_telemetry_path": str(force_path),
                         "restore_hash_before_rollout": restore_hash_before_rollout,
                         "restore_parity_before_rollout": int(restore_hash_before_rollout == r0_hash),
                         "root_restore_hash_after_rollout": p6r1.root_hash(p6, env), "root_hash": r0_hash})
        ok = all(r["requested_force_present"] and r["left_right_half_force_pass"] and r["simulator_steps"] > 0 and r["measured_force_samples"] > 0 for r in rows)
        result.update({"status": "PASS" if ok else "BLOCKED_BY_TASK0_FORCE_INTERFACE", "r0_hash": r0_hash, "results": rows,
                       "outcome_is_not_used_for_method_change": True, "test_status": "TEST_NOT_OPENED"})
        write_csv(OUT / "TASK0_FORCE_TO_PI0_DEV_MICROTEST.csv", rows)
        write_json(OUT / "TASK0_FORCE_TO_PI0_DEV_MICROTEST.json", result)
        (OUT / "TASK0_FORCE_TO_PI0_DEV_QA.md").write_text(
            "# Task0 Newton Force → Frozen π0 DEV QA\n\n"
            f"Status: **{result['status']}**\n\n"
            f"Gate 3 verdict: `{'TASK0_NEWTON_FORCE_TO_PI0_PASS' if ok else 'BLOCKED_BY_TASK0_FORCE_INTERFACE'}`\n\n"
            + "\n".join(f"- {r['force_N']:.1f} N: requested={r['requested_force_present']}, half-slots={r['left_right_half_force_pass']}, simulator_steps={r['simulator_steps']}, measured_samples={r['measured_force_samples']}, full={r['full_task_success_y']}" for r in rows)
            + "\n\nThe force outcomes are infrastructure telemetry only and were not used to alter any frozen method parameter. TEST roots were not accessed.\n",
            encoding="utf-8",
        )
        return 0 if ok else 2
    except Exception as exc:
        result.update({"status": "BLOCKED_BY_TASK0_FORCE_INTERFACE", "error": repr(exc), "trace": traceback.format_exc(), "results": rows, "test_status": "TEST_NOT_OPENED"})
        write_json(OUT / "TASK0_FORCE_TO_PI0_DEV_MICROTEST.json", result)
        (OUT / "TASK0_FORCE_TO_PI0_DEV_QA.md").write_text(
            "# Task0 Newton Force → Frozen π0 DEV QA\n\nStatus: **BLOCKED_BY_TASK0_FORCE_INTERFACE**\n\n"
            f"Runtime exception: `{exc!r}`\n\nTEST remains sealed.\n", encoding="utf-8")
        return 3
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


def run_gate4() -> int:
    """Run the first real DEV ActiveForcing-Direct closed loop.

    The Direct stack is loaded from the frozen method-development output and
    is queried only on DEV context data.  The simulator branch remains the
    existing P4-B probe followed by exact scene restore and the existing
    frozen π0 downstream loop.
    """
    from isaaclab.app import AppLauncher

    p5 = load(P5_PATH, "hf_gate4_p5")
    p6r1 = load(P6R1_PATH, "hf_gate4_p6r1")
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    candidates = [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
    result: dict[str, Any] = {
        "status": "RUNNING", "gate": "Gate 4", "task": TASK,
        "root_seed": DEV_ROOT, "probe_steps_expected": 215,
        "candidate_forces_N": candidates, "online_threshold": 0.5,
        "selection_probability": "raw ensemble mean of frozen Direct feasibility logits after sigmoid",
        "fallback": "minimum candidate with p_success >= 0.5; if none passes, frozen max-force fallback 5.00 N",
        "test_roots_accessed": [], "method": "ActiveForcing-Direct",
    }
    try:
        import torch
        from openpi_client import websocket_client_policy

        env, p6, p4 = p6r1.import_env_modules(TASK)
        p6r1.OBJECTS[TASK] = OBJECT; p6r1.INSTRUCTIONS[TASK] = PROMPT
        p6r1.p6g1.TASK_OBJECTS[TASK] = OBJECT
        p6r1.p6g1.TASK_NAMES[TASK] = "alphabet soup"
        p6r1.p6g1.TASK_INSTRUCTIONS[TASK] = PROMPT
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        client = websocket_client_policy.WebsocketClientPolicy(PI0_HOST, PI0_PORT)

        # Freeze a fresh DEV root and execute the unchanged 215-step P4-B
        # probe from that root.  The no-op reset is scoped to P4-B because the
        # root is already constructed here; it does not alter the probe.
        env.reset(seed=DEV_ROOT)
        p6r1.p6g1.settle_root_before_hash(env, p6, p4, SETTLE_STEPS)
        if hasattr(p4, "_apply_friction"):
            p4._apply_friction(env, OBJECT, DEV_MU)
        r0 = copy.deepcopy(env.scene.get_state(is_relative=True))
        r0_hash = stable_hash(r0)
        original_reset = env.reset

        def no_op_reset(*_args, **_kwargs):
            return env.observation_manager.compute(), {}

        env.reset = no_op_reset
        try:
            probe_rows, probe_rec = p4.run_probe_episode(
                env, seed_idx=DEV_ROOT, mu=DEV_MU,
                trial_id="hf_dev_gate4_r06_s5106", dt=dt,
            )
        finally:
            env.reset = original_reset
        probe_path = OUT / "DEV_GATE4_P4B_PROBE_TELEMETRY.csv"
        write_csv(probe_path, [vars(x) if hasattr(x, "__dict__") else jsonable(x) for x in probe_rows])

        # Reuse the exact frozen Direct code in its compatible scientific
        # Python environment.  The scorer reads only frozen DEV context/model
        # files and returns the nine candidate scores; it does not touch Isaac
        # or any TEST artifact.
        score_proc = subprocess.run(
            ["/usr/bin/python3", str(DIRECT_SCORER_PATH)],
            cwd=str(FORTE), capture_output=True, text=True, timeout=900,
        )
        if score_proc.returncode != 0:
            raise RuntimeError(f"frozen Direct scorer failed: {score_proc.stderr[-4000:]}")
        score_bundle = json.loads(score_proc.stdout)
        score_rows = score_bundle["candidate_scores"]
        passing = [r for r in score_rows if r["passes_threshold"]]
        selected = float(score_bundle["selected_force_N"])
        fallback_used = bool(score_bundle["fallback_used"])

        env.reset_to(copy.deepcopy(r0), torch.tensor([0], device=env.device), is_relative=True)
        restored_state = jsonable(env.scene.get_state(is_relative=True))
        restore_diff = recursive_max_abs(jsonable(r0), restored_state)
        restore_hash = stable_hash(restored_state)
        p6r1.FORCE_N = selected; p6r1.FORCE_HALF_N = selected / 2.0
        tid = "hf_dev_gate4_t0_r06_s5106_direct"
        meta = {"trial_id": tid, "task": TASK, "root_seed": DEV_ROOT,
                "arm": "ACTIVEFORCING_DIRECT", "policy_repeat": "0",
                "policy_repeat_seed": "dev_gate4", "state_parity": int(restore_diff <= 1e-5),
                "staging_success": 1, "handoff_ready": 1}
        raw = p6r1.run_vla_full(env, p6, p4, client, TASK, None, meta, dt, OUT / "GATE4_DIRECT_TELEMETRY")
        force_path = Path(str(raw.get("force_telemetry_path", "")))
        force_rows: list[dict[str, Any]] = []
        if force_path.exists():
            with force_path.open(newline="", encoding="utf-8") as f:
                force_rows = list(csv.DictReader(f))
        probe_phases = {
            str(getattr(x, "probe_phase", "")) if hasattr(x, "probe_phase") else str(x.get("probe_phase", ""))
            for x in probe_rows
        }
        probe_ok = bool(
            (len(probe_rows) == 215 or {"probe_back", "probe_hold"} <= probe_phases)
            and int(probe_rec.get("probe_failure", 0)) == 0
            and int(probe_rec.get("contact_lost_probe", 0)) == 0
            and int(probe_rec.get("dropped", 0)) == 0
        )
        result.update({
            "status": "PASS" if probe_ok and len(score_rows) == 9 and raw.get("episode_steps", 0) > 0 and len(force_rows) > 0 else "BLOCKED_BY_DIRECT_DEV_CLOSED_LOOP",
            "probe": {"rows": len(probe_rows), "required_nominal_rows": 215, "frozen_safety_stop_allowed": True, "completed_phases": sorted(probe_phases), "record": probe_rec, "telemetry_path": str(probe_path), "probe_runtime_qualification": int(probe_ok)},
            "direct_context": score_bundle["direct_context"],
            "candidate_scores": score_rows, "threshold": 0.5, "selected_force_N": selected, "fallback_used": fallback_used,
            "root_restore": {"r0_hash": r0_hash, "restored_hash": restore_hash, "scene_max_abs_diff": restore_diff, "tolerance": 1e-5},
            "execution": {"raw": raw, "force_telemetry_rows": len(force_rows), "force_telemetry_path": str(force_path), "requested_force_N": selected, "measured_force_samples": len(force_rows)},
            "outcome_is_not_used_for_method_change": True, "test_status": "TEST_NOT_OPENED",
        })
        write_json(OUT / "ACTIVEFORCING_DIRECT_DEV_CLOSED_LOOP.json", result)
        write_csv(OUT / "ACTIVEFORCING_DIRECT_DEV_CANDIDATE_SCORES.csv", score_rows)
        report = ["# ActiveForcing-Direct DEV Closed Loop", "", f"Status: **{result['status']}**", "", "Gate 4 verdict: `ACTIVEFORCING_DIRECT_DEV_CLOSED_LOOP_PASS`" if result["status"] == "PASS" else "Gate 4 verdict: `BLOCKED_BY_DIRECT_DEV_CLOSED_LOOP`", "", f"- Probe rows: `{len(probe_rows)}` / 215", f"- Direct context: `{score_bundle['direct_context']['context_id']}`", "- Threshold: `p_success >= 0.5`", f"- Selected force: `{selected:.2f} N`", f"- Fallback used: `{fallback_used}`", f"- Restore max abs diff: `{restore_diff:.8g}`", f"- π0 episode steps: `{raw.get('episode_steps', 0)}`", f"- Force telemetry rows: `{len(force_rows)}`", "", "Candidate scores:", ""]
        report += [f"- `{r['candidate_force_N']:.2f} N -> {r['raw_probability']:.8f}`" for r in score_rows]
        report += ["", "DEV outcome is infrastructure-only; no frozen method parameter was changed. TEST roots were not accessed."]
        (OUT / "ACTIVEFORCING_DIRECT_DEV_CLOSED_LOOP.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        return 0 if result["status"] == "PASS" else 2
    except Exception as exc:
        result.update({"status": "BLOCKED_BY_DIRECT_DEV_CLOSED_LOOP", "error": repr(exc), "trace": traceback.format_exc(), "test_status": "TEST_NOT_OPENED"})
        write_json(OUT / "ACTIVEFORCING_DIRECT_DEV_CLOSED_LOOP.json", result)
        (OUT / "ACTIVEFORCING_DIRECT_DEV_CLOSED_LOOP.md").write_text(
            "# ActiveForcing-Direct DEV Closed Loop\n\nStatus: **BLOCKED_BY_DIRECT_DEV_CLOSED_LOOP**\n\n"
            f"Runtime exception: `{exc!r}`\n\nTEST remains sealed; no TEST root was accessed.\n", encoding="utf-8")
        return 3
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["2", "3", "4"], required=True)
    args = ap.parse_args()
    return run_gate2() if args.gate == "2" else run_gate3() if args.gate == "3" else run_gate4()


if __name__ == "__main__":
    raise SystemExit(main())
