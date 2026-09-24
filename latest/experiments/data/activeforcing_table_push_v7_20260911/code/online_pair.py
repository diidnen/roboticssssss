"""Lockstep paired online pi0 validation of V6 minimum-force ActiveForcing."""
import collections
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
V7_CODE = "/media/volume/data/exouser/activeforcing_table_push_v7_20260911/code"
for path in (CLIENT_DEPS, CLIENT, V5_CODE, V6_CODE, V7_CODE):
    sys.path.insert(0, path)

from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from r2h_branch import array_hash, build_env, contact_force
from v6_controller import TorqueSafeMinimumForceAdapter

OUT = Path(V7_CODE).parent
CONTRACT = json.loads((OUT / "ONLINE_CONTRACT.json").read_text())
TARGET_N = float(os.environ.get("AF_TARGET_N", CONTRACT["activeforcing_minimum_force_target_n"]))
RUN_KIND = os.environ.get("AF_RUN_KIND", "formal")
TABLE_MU = float(os.environ.get("AF_TABLE_MU", CONTRACT["table_friction"][0]))


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def axis_angle(quaternion):
    q = np.asarray(quaternion, dtype=float).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if denominator < 1e-9 else q[:3] * 2.0 * np.arccos(q[3]) / denominator


def direction_from_chunk(chunk):
    actions = np.asarray(chunk, dtype=float)
    selected = [i for i, action in enumerate(actions) if np.linalg.norm(action[:2]) >= 0.01][:15]
    if not selected:
        raise ValueError("no usable XY direction in fresh chunk")
    aggregate = np.median(actions[selected, :2], axis=0)
    norm = float(np.linalg.norm(aggregate))
    if norm < 0.01:
        raise ValueError("degenerate median direction")
    return {
        "chunk_sha256": array_hash(actions),
        "selected_indices": selected,
        "aggregate_xy": aggregate.tolist(),
        "direction_world_xy": (aggregate / norm).tolist(),
    }


class CausalPI:
    def __init__(self, target):
        self.target = float(target)
        self.ema = None
        self.integral = 0.0
        self.command = None

    def reset(self):
        self.ema = None
        self.integral = 0.0
        self.command = None

    def observe(self, measured, contact):
        if not contact:
            self.reset()
            return
        cfg = CONTRACT["controller"]
        alpha = cfg["ema_alpha"]
        self.ema = measured if self.ema is None else alpha * measured + (1.0 - alpha) * self.ema
        error = self.target - self.ema
        proposed_integral = float(np.clip(self.integral + error, -cfg["integral_cap"], cfg["integral_cap"]))
        raw = self.target + cfg["kp"] * error + cfg["ki"] * proposed_integral
        low, high = cfg["axis_command_limits_n"]
        clipped = float(np.clip(raw, low, high))
        if clipped == raw or (clipped == low and error > 0) or (clipped == high and error < 0):
            self.integral = proposed_integral
        if self.command is None:
            self.command = clipped
        else:
            ramp = cfg["ramp_n"]
            self.command = float(np.clip(clipped, self.command - ramp, self.command + ramp))


def prepared_observation(obs):
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224)
    )
    state = np.r_[obs["robot0_eef_pos"], axis_angle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    return image, wrist, state


def main():
    r2 = json.loads((Path(V6_CODE).parent / "R2_RESULT.json").read_text())
    if not r2["pass"]:
        raise RuntimeError("V6 R2 prerequisite did not pass")
    run = OUT / "online_pair" if RUN_KIND == "formal" else OUT / "online_calibration" / f"mu_{TABLE_MU:.2f}_target_{TARGET_N:.1f}N"
    if run.exists():
        raise RuntimeError(f"immutable online pair exists: {run}")
    run.mkdir(parents=True)
    env_b, obs_b, table_b = build_env()
    env_h, obs_h, table_h = build_env()
    env_b.sim.model.geom_friction[table_b, 0] = TABLE_MU
    env_h.sim.model.geom_friction[table_h, 0] = TABLE_MU
    client = WebsocketClientPolicy(host="127.0.0.1", port=CONTRACT["policy_port"])
    metadata = client.get_server_metadata()
    shared_plan, plan_b, plan_h = collections.deque(), collections.deque(), collections.deque()
    inference_rows = []
    direction = None
    direction_record = None
    direction_step = None
    stable = 0
    post_replan_index = 0
    pi = CausalPI(TARGET_N)
    adapter = None
    rows_b, rows_h, pair_rows = [], [], []
    success_b = success_h = False
    first_success_b = first_success_h = None
    writers = {
        "untouched": cv2.VideoWriter(str(run / "untouched.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (224, 224)),
        "hybrid": cv2.VideoWriter(str(run / "hybrid.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (224, 224)),
    }

    def infer(obs, step, branch, reason, seed=None):
        image, wrist, state = prepared_observation(obs)
        request = {
            "observation/image": image,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": CONTRACT["task_language"],
        }
        if seed is not None:
            request["_activeforcing_episode_start_seed"] = int(seed)
        actions = np.asarray(client.infer(request)["actions"], dtype=float)
        inference_rows.append(
            {
                "step": step,
                "branch": branch,
                "reason": reason,
                "seed": seed,
                "image_sha256": array_hash(image),
                "wrist_sha256": array_hash(wrist),
                "state_sha256": array_hash(state),
                "actions_sha256": array_hash(actions),
            }
        )
        return actions

    error = None
    try:
        if array_hash(env_b.get_sim_state()) != array_hash(env_h.get_sim_state()):
            raise RuntimeError("initial physical states are not exact")
        for step in range(CONTRACT["maximum_steps"]):
            frame_b, _, _ = prepared_observation(obs_b)
            frame_h, _, _ = prepared_observation(obs_h)
            writers["untouched"].write(cv2.cvtColor(frame_b, cv2.COLOR_RGB2BGR))
            writers["hybrid"].write(cv2.cvtColor(frame_h, cv2.COLOR_RGB2BGR))
            state_b = env_b.get_sim_state().copy()
            state_h = env_h.get_sim_state().copy()
            qpos_b, qpos_h = env_b.sim.data.qpos.copy(), env_h.sim.data.qpos.copy()
            qvel_b, qvel_h = env_b.sim.data.qvel.copy(), env_h.sim.data.qvel.copy()
            force_b_pre, contacts_b_pre, raw_b_pre = contact_force(env_b.sim.model, env_b.sim.data)
            force_h_pre, contacts_h_pre, raw_h_pre = contact_force(env_h.sim.model, env_h.sim.data)
            if direction is None and bool(contacts_b_pre) != bool(contacts_h_pre):
                raise RuntimeError("pre-intervention contact mismatch")
            stable = stable + 1 if contacts_b_pre else 0

            if step < CONTRACT["stabilization_steps"]:
                action_b = action_h = np.r_[np.zeros(6), -1.0]
                source_b = source_h = "shared_stabilize"
            elif direction is None:
                if not shared_plan:
                    seed = CONTRACT["initial_policy_seed"] if step == CONTRACT["stabilization_steps"] else None
                    shared_plan.extend(infer(obs_b, step, "shared", "pre_intervention_replan", seed)[: CONTRACT["chunk_execution_length"]])
                action_b = action_h = np.asarray(shared_plan.popleft(), dtype=float)
                source_b = source_h = "shared_online_pi0"
            else:
                if not plan_b and not plan_h:
                    seed = CONTRACT["post_intervention_common_seed_base"] + post_replan_index
                    plan_b.extend(infer(obs_b, step, "untouched", "post_intervention_replan", seed)[: CONTRACT["chunk_execution_length"]])
                    plan_h.extend(infer(obs_h, step, "hybrid", "post_intervention_replan", seed)[: CONTRACT["chunk_execution_length"]])
                    post_replan_index += 1
                action_b = np.asarray(plan_b.popleft(), dtype=float)
                action_h = np.asarray(plan_h.popleft(), dtype=float)
                source_b = source_h = "branch_online_pi0_common_noise"

            if direction is None and stable >= 3:
                action_norm = float(np.linalg.norm(action_h[:2]))
                if action_norm < 0.01:
                    raise RuntimeError("third-contact nominal action has degenerate XY direction")
                direction = np.asarray(action_h[:2], dtype=float) / action_norm
                direction_record = {
                    "source": "third-contact current nominal pi0 XY action",
                    "nominal_action_sha256": array_hash(action_h),
                    "raw_xy": np.asarray(action_h[:2], dtype=float).tolist(),
                    "direction_world_xy": direction.tolist(),
                }
                direction_step = step
                common = list(shared_plan)
                shared_plan.clear()
                plan_b.extend(common)
                plan_h.extend(common)
                source_b = source_h = "shared_motion_direction_arming_action_untouched"

            current_norm = float(np.linalg.norm(action_h[:2]))
            current_direction = np.asarray(action_h[:2], dtype=float) / current_norm if current_norm >= 0.01 else None

            intervention = bool(
                direction is not None and step > direction_step and contacts_h_pre and pi.command is not None and current_direction is not None
            )
            command = float(pi.command) if intervention else None
            if adapter is None and intervention:
                adapter = TorqueSafeMinimumForceAdapter(env_h, current_direction, True, command)
            elif adapter is not None:
                adapter.enabled = intervention
                if intervention:
                    adapter.minimum_axis_force_n = command
                    adapter.direction = np.asarray([current_direction[0], current_direction[1], 0.0], dtype=float)
                adapter.trace = []
            if not contacts_h_pre or current_direction is None:
                pi.reset()

            obs_b_next, reward_b, done_b, info_b = env_b.step(action_b.tolist())
            obs_h_next, reward_h, done_h, info_h = env_h.step(action_h.tolist())
            force_b_post, contacts_b_post, raw_b_post = contact_force(env_b.sim.model, env_b.sim.data)
            force_h_post, contacts_h_post, raw_h_post = contact_force(env_h.sim.model, env_h.sim.data)
            aligned_b = None if current_direction is None else float(force_b_post[:2] @ current_direction)
            aligned_h = None if current_direction is None else float(force_h_post[:2] @ current_direction)
            if direction is not None and current_direction is not None:
                pi.observe(aligned_h, bool(contacts_h_post))
            controller_trace = [] if adapter is None else list(adapter.trace)
            success_b_now = bool(env_b.check_success())
            success_h_now = bool(env_h.check_success())
            if success_b_now and not success_b:
                first_success_b = step
            if success_h_now and not success_h:
                first_success_h = step
            success_b |= success_b_now
            success_h |= success_h_now

            common_b = {
                "step": step,
                "source": source_b,
                "pre_state_sha256": array_hash(state_b),
                "pre_qpos": qpos_b.tolist(),
                "pre_qvel": qvel_b.tolist(),
                "nominal_action": action_b.tolist(),
                "nominal_action_sha256": array_hash(action_b),
                "pre_contact": bool(contacts_b_pre),
                "post_contact": bool(contacts_b_post),
                "pre_force_world": force_b_pre.tolist(),
                "post_force_world": force_b_post.tolist(),
                "post_aligned_force_n": aligned_b,
                "nominal_action_direction_xy": None if current_direction is None else current_direction.tolist(),
                "plate_position": env_b.sim.data.qpos[30:33].copy().tolist(),
                "reward": float(reward_b),
                "native_success_now": success_b_now,
            }
            common_h = {
                "step": step,
                "source": source_h,
                "pre_state_sha256": array_hash(state_h),
                "pre_qpos": qpos_h.tolist(),
                "pre_qvel": qvel_h.tolist(),
                "nominal_action": action_h.tolist(),
                "nominal_action_sha256": array_hash(action_h),
                "pre_contact": bool(contacts_h_pre),
                "post_contact": bool(contacts_h_post),
                "pre_force_world": force_h_pre.tolist(),
                "post_force_world": force_h_post.tolist(),
                "post_aligned_force_n": aligned_h,
                "nominal_action_direction_xy": None if current_direction is None else current_direction.tolist(),
                "plate_position": env_h.sim.data.qpos[30:33].copy().tolist(),
                "intervention": intervention,
                "minimum_axis_command_n": command,
                "controller_trace": controller_trace,
                "reward": float(reward_h),
                "native_success_now": success_h_now,
            }
            rows_b.append(common_b)
            rows_h.append(common_h)
            if direction_step is None or step <= direction_step + 1:
                pair_rows.append(
                    {
                        "step": step,
                        "state_exact": array_hash(state_b) == array_hash(state_h),
                        "qpos_max_abs_diff": float(np.max(np.abs(qpos_b - qpos_h))),
                        "qvel_max_abs_diff": float(np.max(np.abs(qvel_b - qvel_h))),
                        "action_exact": array_hash(action_b) == array_hash(action_h),
                        "action_max_abs_diff": float(np.max(np.abs(action_b - action_h))),
                        "hybrid_intervention": intervention,
                    }
                )
            obs_b, obs_h = obs_b_next, obs_h_next
            if success_b and success_h:
                break
    except Exception as exc:
        error = repr(exc)
    finally:
        if adapter is not None:
            adapter.close()
        for writer in writers.values():
            writer.release()
        env_b.close()
        env_h.close()

    (run / "untouched.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows_b))
    (run / "hybrid.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows_h))
    (run / "inference.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inference_rows))
    first_intervention = next((row["step"] for row in rows_h if row["intervention"]), None)
    paired_prefix = [row for row in pair_rows if first_intervention is None or row["step"] <= first_intervention]
    trace = [item for row in rows_h for item in row["controller_trace"]]
    checks = {
        "error_free": error is None,
        "exact_pre_intervention_pair": bool(paired_prefix) and all(row["state_exact"] and row["action_exact"] for row in paired_prefix),
        "hybrid_native_success": success_h,
        "minimum_intervention_steps": sum(row["intervention"] for row in rows_h) >= CONTRACT["pass_gate"]["minimum_intervention_steps"],
        "finite_torque": all(item.get("finite", False) for item in trace),
        "zero_actuator_clips": sum(item.get("clip_count", 0) for item in trace) == 0,
        "stock_axis_never_reduced": all(
            item["applied_axis_force_n"] + 1e-10 >= item["stock_axis_force_n"]
            for item in trace if item.get("mode") == "torque_safe_minimum_axis_force"
        ),
        "all_intervention_directions_match_nominal_actions": all(
            row["nominal_action_direction_xy"] is not None
            and np.allclose(
                np.asarray(row["nominal_action_direction_xy"]),
                np.asarray(row["nominal_action"][:2]) / np.linalg.norm(row["nominal_action"][:2]),
                atol=0.0,
                rtol=1e-14,
            )
            for row in rows_h if row["intervention"]
        ),
    }
    result = {
        "gate": CONTRACT["gate"],
        "contract_version": CONTRACT["version"],
        "run_kind": RUN_KIND,
        "target_n": TARGET_N,
        "table_slide_friction": TABLE_MU,
        "pass": all(checks.values()),
        "checks": checks,
        "server_metadata": metadata,
        "steps": len(rows_h),
        "direction_step": direction_step,
        "direction_record": direction_record,
        "first_intervention_step": first_intervention,
        "intervention_steps": sum(row["intervention"] for row in rows_h),
        "untouched_native_success": success_b,
        "hybrid_native_success": success_h,
        "untouched_first_success_step": first_success_b,
        "hybrid_first_success_step": first_success_h,
        "outcome_class": "force_control_recovery" if success_h and not success_b else "no_regression_both_success" if success_h else "hybrid_failed",
        "total_actuator_clips": sum(item.get("clip_count", 0) for item in trace),
        "minimum_safety_alpha": min((item.get("safety_alpha", 1.0) for item in trace), default=1.0),
        "final_plate_position_untouched": rows_b[-1]["plate_position"] if rows_b else None,
        "final_plate_position_hybrid": rows_h[-1]["plate_position"] if rows_h else None,
        "pairing_rows": pair_rows,
        "error": error,
    }
    result_path = OUT / "ONLINE_RESULT.json" if RUN_KIND == "formal" else run / "RESULT.json"
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)
    if error is not None:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
