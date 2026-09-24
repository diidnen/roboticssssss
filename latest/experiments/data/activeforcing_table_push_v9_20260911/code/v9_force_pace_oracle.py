"""V9 exact-state force-and-pace oracle gate for table-push ActiveForcing.

This is deliberately an oracle causal test, not a deployable learned selector.
At every eligible pi0 replan it branches the same next five actions from the
same simulator/controller state under a frozen 4x3 grid of XY pace scales and
torque-safe task-axis force floors.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

CLIENT = "/home/exouser/SoftVTBench/openpi/upstream/packages/openpi-client/src"
CLIENT_DEPS = "/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/client_venv/lib/python3.10/site-packages"
V5_CODE = "/media/volume/data/exouser/activeforcing_table_push_v5_20260910/code"
V6_CODE = "/media/volume/data/exouser/activeforcing_table_push_v6_20260910/code"
OUT = Path(os.environ.get("AF_V9_OUT", "/media/volume/data/exouser/activeforcing_table_push_v9_20260911"))
for path in (CLIENT_DEPS, CLIENT, V5_CODE, V6_CODE):
    sys.path.insert(0, path)

from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from r2h_branch import array_hash, build_env, contact_force
from v6_controller import TorqueSafeMinimumForceAdapter

POLICY_PORT = 8015
TABLE_MU = 1.2
CHUNK = 5
MAX_STEPS = 310
STABILIZE = 10
ALPHAS = [0.25, 0.5, 0.75, 1.0]
FORCES = [None, 30.0, 40.0]
CANDIDATES = [(alpha, floor_n) for alpha in ALPHAS for floor_n in FORCES]
GOAL_CENTER = np.asarray([-0.05, 0.21], dtype=float)
ACTION_XY_MIN = 0.01


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def axis_angle(quaternion):
    q = np.asarray(quaternion, dtype=float).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    return np.zeros(3) if denominator < 1e-9 else q[:3] * 2.0 * np.arccos(q[3]) / denominator


def prepared_observation(obs):
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224)
    )
    state = np.r_[obs["robot0_eef_pos"], axis_angle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    return image, wrist, state


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def clone_simple(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (str, bytes, int, float, bool, type(None), np.generic)):
        return copy.deepcopy(value)
    if isinstance(value, (list, tuple, dict)):
        try:
            return copy.deepcopy(value)
        except Exception:
            return None
    return None


def object_state_snapshot(obj):
    if obj is None:
        return {}
    result = {}
    for key, value in obj.__dict__.items():
        cloned = clone_simple(value)
        if cloned is not None:
            result[key] = cloned
    return result


def object_state_restore(obj, state):
    if obj is None:
        return
    for key, value in state.items():
        setattr(obj, key, copy.deepcopy(value))


def control_snapshot(env):
    c = env.robots[0].controller
    robot = env.robots[0]
    inner = env.env
    sim_fields = {}
    # MjSimState.flatten() intentionally omits solver warm-start and control
    # buffers. They are causally relevant when replaying a contact-rich chunk.
    for key in (
        "ctrl", "qacc_warmstart", "qacc", "qfrc_applied", "xfrc_applied",
        "mocap_pos", "mocap_quat", "act", "userdata",
    ):
        value = getattr(env.sim.data, key, None)
        if value is not None:
            sim_fields[key] = np.asarray(value).copy()
    return {
        "controller": object_state_snapshot(c),
        "robot": object_state_snapshot(robot),
        "interpolator_pos": object_state_snapshot(getattr(c, "interpolator_pos", None)),
        "interpolator_ori": object_state_snapshot(getattr(c, "interpolator_ori", None)),
        "sim_data": sim_fields,
        "env_timestep": getattr(inner, "timestep", None),
        "env_done": getattr(inner, "done", None),
    }


def restore_all(env, sim_state, control_state):
    c = env.robots[0].controller
    env.sim.set_state_from_flattened(np.asarray(sim_state, dtype=float))
    env.sim.forward()
    object_state_restore(c, control_state["controller"])
    object_state_restore(env.robots[0], control_state["robot"])
    object_state_restore(getattr(c, "interpolator_pos", None), control_state["interpolator_pos"])
    object_state_restore(getattr(c, "interpolator_ori", None), control_state["interpolator_ori"])
    for key, value in control_state["sim_data"].items():
        target = getattr(env.sim.data, key, None)
        if target is not None and np.asarray(target).shape == np.asarray(value).shape:
            target[...] = value
    if control_state["env_timestep"] is not None:
        env.env.timestep = control_state["env_timestep"]
    if control_state["env_done"] is not None:
        env.env.done = control_state["env_done"]
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()


def run_chunk(env, actions, floor_n, alpha):
    pre_xy = env.sim.data.qpos[30:32].copy()
    pre_dist = float(np.linalg.norm(pre_xy - GOAL_CENTER))
    adapter = None
    rows = []
    clips = 0
    min_alpha = 1.0
    success = False
    try:
        for local_step, nominal_action in enumerate(np.asarray(actions, dtype=float)):
            action = nominal_action.copy()
            action[:2] *= float(alpha)
            pre_force, pre_contacts, _ = contact_force(env.sim.model, env.sim.data)
            norm = float(np.linalg.norm(action[:2]))
            eligible = bool(pre_contacts) and norm >= ACTION_XY_MIN and floor_n is not None
            direction = action[:2] / norm if norm >= ACTION_XY_MIN else None
            if eligible and adapter is None:
                adapter = TorqueSafeMinimumForceAdapter(env, direction, True, floor_n)
            if adapter is not None:
                adapter.trace = []
                adapter.enabled = eligible
                if eligible:
                    adapter.minimum_axis_force_n = float(floor_n)
                    adapter.direction = np.asarray([direction[0], direction[1], 0.0])
            obs, reward, done, info = env.step(action.tolist())
            post_force, post_contacts, _ = contact_force(env.sim.model, env.sim.data)
            trace = [] if adapter is None else list(adapter.trace)
            clips += sum(int(x.get("clip_count", 0)) for x in trace)
            min_alpha = min(min_alpha, min((float(x.get("safety_alpha", 1.0)) for x in trace), default=1.0))
            success = success or bool(env.check_success())
            rows.append({
                "local_step": local_step,
                "alpha": float(alpha),
                "nominal_action": nominal_action.tolist(),
                "executed_action": action.tolist(),
                "pre_contact": bool(pre_contacts),
                "post_contact": bool(post_contacts),
                "eligible": eligible,
                "direction_xy": None if direction is None else direction.tolist(),
                "pre_force_world": pre_force.tolist(),
                "post_force_world": post_force.tolist(),
                "plate_xy": env.sim.data.qpos[30:32].copy().tolist(),
                "reward": float(reward),
                "controller_trace": trace,
            })
        post_xy = env.sim.data.qpos[30:32].copy()
        post_dist = float(np.linalg.norm(post_xy - GOAL_CENTER))
        displacement = post_xy - pre_xy
        goal_dir = (GOAL_CENTER - pre_xy) / max(pre_dist, 1e-12)
        along = float(displacement @ goal_dir)
        lateral = float(abs(displacement[0] * goal_dir[1] - displacement[1] * goal_dir[0]))
        contact_count = int(sum(r["post_contact"] for r in rows))
        return {
            "candidate_floor_n": floor_n,
            "candidate_alpha": float(alpha),
            "pre_plate_xy": pre_xy.tolist(),
            "post_plate_xy": post_xy.tolist(),
            "goal_distance_before_m": pre_dist,
            "goal_distance_after_m": post_dist,
            "goal_progress_m": pre_dist - post_dist,
            "along_goal_displacement_m": along,
            "lateral_displacement_m": lateral,
            "contact_count": contact_count,
            "final_contact": bool(rows[-1]["post_contact"]),
            "native_success": success,
            "total_actuator_clips": clips,
            "minimum_safety_alpha": min_alpha,
            "final_state_sha256": array_hash(env.get_sim_state()),
            "final_qpos_sha256": array_hash(env.sim.data.qpos.copy()),
            "final_state": env.get_sim_state().copy().tolist(),
            "final_qpos": env.sim.data.qpos.copy().tolist(),
            "final_qvel": env.sim.data.qvel.copy().tolist(),
            "rows": rows,
        }
    finally:
        if adapter is not None:
            adapter.close()


def candidate_rank(row):
    # Strict lexicographic physical objective. The final tie-break prefers the
    # lower intervention, with stock ordered below every explicit floor.
    floor_cost = 0.0 if row["candidate_floor_n"] is None else float(row["candidate_floor_n"])
    return (
        int(row["native_success"]),
        int(row["final_contact"]),
        int(row["contact_count"]),
        float(row["goal_progress_m"]),
        float(row["along_goal_displacement_m"]),
        -float(row["lateral_displacement_m"]),
        -floor_cost,
        float(row["candidate_alpha"]),
    )


def main():
    run = OUT / "force_pace_oracle_root0_mu1p2"
    if run.exists():
        raise RuntimeError(f"immutable run exists: {run}")
    run.mkdir(parents=True)
    env, obs, table = build_env()
    env.sim.model.geom_friction[table] = [TABLE_MU, 0.005, 0.0001]
    client = WebsocketClientPolicy(host="127.0.0.1", port=POLICY_PORT)
    metadata = client.get_server_metadata()
    plan = collections.deque()
    inference_rows, oracle_rows, execution_rows = [], [], []
    stable_contact = 0
    armed = False
    replan_index = 0
    success = False
    first_success_step = None
    error = None
    video = cv2.VideoWriter(str(run / "oracle.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (224, 224))
    print(json.dumps({
        "event": "V9_START",
        "alphas": ALPHAS,
        "forces_n": FORCES,
        "candidate_count": len(CANDIDATES),
        "table_mu": TABLE_MU,
        "chunk": CHUNK,
        "maximum_steps": MAX_STEPS,
    }), flush=True)

    def infer(current_obs, step):
        image, wrist, state = prepared_observation(current_obs)
        seed = 8000 + replan_index
        request = {
            "observation/image": image,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": "push the plate to the front of the stove",
            "_activeforcing_episode_start_seed": seed,
        }
        actions = np.asarray(client.infer(request)["actions"], dtype=float)
        inference_rows.append({
            "step": step,
            "replan_index": replan_index,
            "seed": seed,
            "image_sha256": array_hash(image),
            "wrist_sha256": array_hash(wrist),
            "state_sha256": array_hash(state),
            "actions_sha256": array_hash(actions),
        })
        return actions[:CHUNK]

    try:
        for step in range(MAX_STEPS):
            frame, _, _ = prepared_observation(obs)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            pre_force, pre_contacts, _ = contact_force(env.sim.model, env.sim.data)
            stable_contact = stable_contact + 1 if pre_contacts else 0
            armed = armed or stable_contact >= 3

            if step < STABILIZE:
                action = np.r_[np.zeros(6), -1.0]
                selected = None
                selected_alpha = 1.0
                source = "stabilize"
            else:
                if not plan:
                    chunk = infer(obs, step)
                    replan_index += 1
                    sim_state = env.get_sim_state().copy()
                    cstate = control_snapshot(env)
                    branch_rows = []
                    if armed:
                        for alpha, floor_n in CANDIDATES:
                            restore_all(env, sim_state, cstate)
                            branch_rows.append(run_chunk(env, chunk, floor_n, alpha))
                        # Replay stock twice from the same full saved state. Any
                        # difference invalidates all counterfactual comparisons.
                        stock_reference = next(
                            r for r in branch_rows
                            if r["candidate_floor_n"] is None and r["candidate_alpha"] == 1.0
                        )
                        restore_all(env, sim_state, cstate)
                        stock_repeat = run_chunk(env, chunk, None, 1.0)
                        parity_state = float(np.max(np.abs(np.asarray(stock_reference["final_state"]) - np.asarray(stock_repeat["final_state"]))))
                        parity_qpos = float(np.max(np.abs(np.asarray(stock_reference["final_qpos"]) - np.asarray(stock_repeat["final_qpos"]))))
                        parity_qvel = float(np.max(np.abs(np.asarray(stock_reference["final_qvel"]) - np.asarray(stock_repeat["final_qvel"]))))
                        exact_parity = stock_reference["final_state_sha256"] == stock_repeat["final_state_sha256"]
                        if not exact_parity:
                            raise RuntimeError(f"same-state stock replay parity failed: state={parity_state:.3e}, qpos={parity_qpos:.3e}, qvel={parity_qvel:.3e}")
                        valid = [r for r in branch_rows if r["total_actuator_clips"] == 0]
                        if not valid:
                            raise RuntimeError("every force candidate clipped an actuator")
                        chosen = max(valid, key=candidate_rank)
                        selected = chosen["candidate_floor_n"]
                        selected_alpha = chosen["candidate_alpha"]
                    else:
                        selected = None
                        selected_alpha = 1.0
                    restore_all(env, sim_state, cstate)
                    oracle_rows.append({
                        "step": step,
                        "replan_index": replan_index - 1,
                        "armed": armed,
                        "pre_contact": bool(pre_contacts),
                        "chunk_sha256": array_hash(chunk),
                        "selected_floor_n": selected,
                        "selected_alpha": selected_alpha,
                        "branches": branch_rows,
                        "stock_repeat_parity": {
                            "exact": exact_parity if branch_rows else None,
                            "state_max_abs_diff": parity_state if branch_rows else None,
                            "qpos_max_abs_diff": parity_qpos if branch_rows else None,
                            "qvel_max_abs_diff": parity_qvel if branch_rows else None,
                        },
                    })
                    plan.extend([(a, selected_alpha, selected) for a in chunk])
                nominal_action, selected_alpha, selected = plan.popleft()
                action = np.asarray(nominal_action, dtype=float).copy()
                action[:2] *= float(selected_alpha)
                source = "pi0_force_pace_oracle_chunk" if (selected is not None or selected_alpha != 1.0) else "pi0_stock_chunk"

            adapter = None
            norm = float(np.linalg.norm(action[:2]))
            force_intervention = selected is not None and bool(pre_contacts) and norm >= ACTION_XY_MIN
            pace_intervention = step >= STABILIZE and float(selected_alpha) != 1.0
            if force_intervention:
                direction = action[:2] / norm
                adapter = TorqueSafeMinimumForceAdapter(env, direction, True, selected)
            obs_next, reward, done, info = env.step(np.asarray(action, dtype=float).tolist())
            post_force, post_contacts, _ = contact_force(env.sim.model, env.sim.data)
            trace = [] if adapter is None else list(adapter.trace)
            if adapter is not None:
                adapter.close()
            success_now = bool(env.check_success())
            if success_now and not success:
                first_success_step = step
            success = success or success_now
            execution_rows.append({
                "step": step,
                "source": source,
                "nominal_action_sha256": array_hash(action if step < STABILIZE else nominal_action),
                "executed_action_sha256": array_hash(action),
                "selected_alpha": float(selected_alpha),
                "selected_floor_n": selected,
                "intervention": bool(force_intervention or pace_intervention),
                "force_intervention": force_intervention,
                "pace_intervention": pace_intervention,
                "pre_contact": bool(pre_contacts),
                "post_contact": bool(post_contacts),
                "pre_force_world": pre_force.tolist(),
                "post_force_world": post_force.tolist(),
                "plate_position": env.sim.data.qpos[30:33].copy().tolist(),
                "reward": float(reward),
                "native_success_now": success_now,
                "controller_trace": trace,
            })
            obs = obs_next
            if step % 25 == 0 or success_now:
                print(json.dumps({"event": "progress", "step": step, "success": success, "armed": armed, "contact": bool(post_contacts), "plate_xy": env.sim.data.qpos[30:32].copy().tolist(), "latest_selected_alpha": selected_alpha, "latest_selected_floor_n": selected}), flush=True)
            if success:
                break
    except Exception as exc:
        error = repr(exc)
    finally:
        video.release()
        env.close()

    (run / "inference.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inference_rows))
    (run / "oracle_branches.jsonl").write_text("".join(json.dumps(x) + "\n" for x in oracle_rows))
    (run / "execution.jsonl").write_text("".join(json.dumps(x) + "\n" for x in execution_rows))
    trace = [t for r in execution_rows for t in r["controller_trace"]]
    selected_combos = [
        {"alpha": r["selected_alpha"], "floor_n": r["selected_floor_n"]}
        for r in oracle_rows if r["branches"]
    ]
    final_xy = execution_rows[-1]["plate_position"][:2] if execution_rows else None
    initial_xy = execution_rows[0]["plate_position"][:2] if execution_rows else None
    all_stock_replays_exact = bool(selected_combos) and all(
        r["stock_repeat_parity"]["exact"] for r in oracle_rows if r["branches"]
    )
    total_clips = sum(int(t.get("clip_count", 0)) for t in trace)
    checkpoint_hash_matches = metadata.get("checkpoint_tree_sha256") == "92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103"
    validity_checks = {
        "error_free": error is None,
        "all_repeated_stock_alpha1_branches_byte_exact": all_stock_replays_exact,
        "executed_actuator_clips_zero": total_clips == 0,
        "frozen_pi0_checkpoint_hash_matches": checkpoint_hash_matches,
    }
    gate_pass = bool(success and all(validity_checks.values()))
    result = {
        "gate": "V9_RECEDING_HORIZON_FORCE_AND_PACE_ORACLE",
        "classification": "FORCE_PACE_ORACLE_RECOVERS_TASK" if gate_pass else ("INVALID_EXECUTION" if not all(validity_checks.values()) else "FORCE_PACE_ORACLE_DOES_NOT_RECOVER_TASK"),
        "pass": gate_pass,
        "validity_checks": validity_checks,
        "error": error,
        "native_success": success,
        "first_success_step": first_success_step,
        "steps": len(execution_rows),
        "table_slide_friction": TABLE_MU,
        "candidate_alpha": ALPHAS,
        "candidate_minimum_axis_force_n": FORCES,
        "candidate_count": len(CANDIDATES),
        "oracle_replans": len(oracle_rows),
        "active_oracle_replans": len(selected_combos),
        "selected_combinations": selected_combos,
        "unique_selected_combinations": sorted({(x["alpha"], x["floor_n"]) for x in selected_combos}, key=str),
        "intervention_steps": sum(int(r["intervention"]) for r in execution_rows),
        "pace_intervention_steps": sum(int(r["pace_intervention"]) for r in execution_rows),
        "force_intervention_steps": sum(int(r["force_intervention"]) for r in execution_rows),
        "contact_steps": sum(int(r["post_contact"]) for r in execution_rows),
        "total_actuator_clips": total_clips,
        "minimum_safety_alpha": min((float(t.get("safety_alpha", 1.0)) for t in trace), default=1.0),
        "initial_plate_xy": initial_xy,
        "final_plate_xy": final_xy,
        "final_goal_distance_m": None if final_xy is None else float(np.linalg.norm(np.asarray(final_xy) - GOAL_CENTER)),
        "server_metadata": metadata,
        "claim_boundary": "simulator-oracle force-and-pace causal gate; not deployable learned ActiveForcing",
    }
    atomic_json(run / "RESULT.json", result)
    atomic_json(OUT / "V9_RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if error is not None:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
