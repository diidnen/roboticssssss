"""One fresh-process paired online pi0 rollout for V5 table-friction ActiveForcing."""
import collections
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
for path in (CLIENT_DEPS, CLIENT):
    if path not in sys.path:
        sys.path.insert(0, path)

from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

CODE = Path("/media/volume/data/exouser/activeforcing_table_push_v5_20260910/code")
sys.path.insert(0, str(CODE))
from r2h_branch import HybridForceAdapter, array_hash, build_env, contact_force

OUT = CODE.parent
CONTRACT = json.loads((OUT / "A3_ONLINE_CONTRACT.json").read_text())


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


def direction_from_fresh_chunk(chunk):
    actions = np.asarray(chunk, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("fresh action chunk must be [T,7]")
    selected = [i for i, action in enumerate(actions) if np.linalg.norm(action[:2]) >= 0.01][:15]
    if not selected:
        raise ValueError("fresh chunk has no usable XY direction")
    aggregate = np.median(actions[selected, :2], axis=0)
    norm = float(np.linalg.norm(aggregate))
    if norm < 0.01:
        raise ValueError("fresh chunk median direction is degenerate")
    return {
        "raw_chunk": actions.tolist(),
        "chunk_sha256": array_hash(actions),
        "selected_indices": selected,
        "aggregate_xy": aggregate.tolist(),
        "direction_world_xy": (aggregate / norm).tolist(),
        "min_norm": 0.01,
    }


def observation_hashes(observation):
    return {key: array_hash(observation[key]) for key in sorted(observation)}


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

    def observe_completed_step(self, measured_force, contact):
        if not contact:
            self.reset()
            return
        cfg = CONTRACT["controller"]
        alpha = float(cfg["ema_alpha"])
        self.ema = measured_force if self.ema is None else alpha * measured_force + (1.0 - alpha) * self.ema
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
            ramp = float(cfg["ramp_n"])
            self.command = float(np.clip(clipped, self.command - ramp, self.command + ramp))


def main():
    mode = sys.argv[1]
    if mode not in CONTRACT["branches"]:
        raise ValueError(mode)
    run = OUT / "online" / mode
    if run.exists():
        raise RuntimeError(f"immutable online branch exists: {run}")
    run.mkdir(parents=True)

    env, obs, table = build_env()
    model, data = env.sim.model, env.sim.data
    task = "push the plate to the front of the stove"
    client = WebsocketClientPolicy(host="127.0.0.1", port=int(CONTRACT["policy_port"]))
    server_metadata = client.get_server_metadata()
    plan = collections.deque()
    pi = CausalPI(CONTRACT["hybrid_target_n"])
    adapter = None
    stable = 0
    direction = None
    direction_record = None
    direction_step = None
    inference_index = 0
    rows = []
    inference_rows = []
    writer = cv2.VideoWriter(
        str(run / "agentview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (224, 224)
    )

    def infer(current_obs, step, episode_start=False, reason="replan"):
        nonlocal inference_index
        image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(current_obs["agentview_image"][::-1, ::-1]), 224, 224)
        )
        wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(current_obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224)
        )
        state = np.r_[
            current_obs["robot0_eef_pos"],
            axis_angle(current_obs["robot0_eef_quat"]),
            current_obs["robot0_gripper_qpos"],
        ]
        request = {
            "observation/image": image,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": task,
        }
        if episode_start:
            request["_activeforcing_episode_start_seed"] = int(CONTRACT["policy_seed"])
        answer = client.infer(request)
        actions = np.asarray(answer["actions"], dtype=float)
        inference_rows.append(
            {
                "inference_index": inference_index,
                "step": step,
                "reason": reason,
                "episode_start_seed": int(CONTRACT["policy_seed"]) if episode_start else None,
                "image_sha256": array_hash(image),
                "wrist_sha256": array_hash(wrist),
                "state_sha256": array_hash(state),
                "actions_sha256": array_hash(actions),
                "actions_shape": list(actions.shape),
            }
        )
        inference_index += 1
        return actions

    error = None
    try:
        before = model.geom_friction[table].copy()
        model.geom_friction[table] = [
            CONTRACT["table_slide_friction"],
            CONTRACT["table_torsional_friction"],
            CONTRACT["table_rolling_friction"],
        ]
        readback = model.geom_friction[table].copy()

        for step in range(int(CONTRACT["maximum_steps"])):
            frame = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224)
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            pre_state = env.get_sim_state().copy()
            pre_qpos = data.qpos.copy()
            pre_qvel = data.qvel.copy()
            pre_force, pre_identities, pre_raw = contact_force(model, data)
            stable = stable + 1 if pre_identities else 0

            if step < int(CONTRACT["stabilization_steps"]):
                nominal = np.r_[np.zeros(6), -1.0]
                source = "stabilize"
            else:
                if not plan:
                    chunk = infer(obs, step, episode_start=(step == CONTRACT["stabilization_steps"]), reason="replan")
                    plan.extend(chunk[: int(CONTRACT["policy_chunk_execution_length"])])
                nominal = np.asarray(plan.popleft(), dtype=float)
                source = "online_pi0"

            if direction is None and stable >= 3:
                plan.clear()
                fresh = infer(obs, step, reason="fresh_after_third_contact")
                direction_record = direction_from_fresh_chunk(fresh)
                direction = np.asarray(direction_record["direction_world_xy"], dtype=float)
                direction_step = step
                plan.extend(fresh[: int(CONTRACT["policy_chunk_execution_length"])])
                nominal = np.asarray(plan.popleft(), dtype=float)
                source = "fresh_pi0_after_third_contact_first_action_untouched"

            intervention = bool(
                mode == "hybrid"
                and direction is not None
                and direction_step is not None
                and step > direction_step
                and pre_identities
                and pi.command is not None
            )
            command = float(pi.command) if intervention else None
            if mode == "hybrid" and direction is not None:
                if adapter is None and intervention:
                    adapter = HybridForceAdapter(env, direction, True, command)
                elif adapter is not None:
                    adapter.enabled = intervention
                    adapter.axis_command_n = command if intervention else adapter.axis_command_n
                    adapter.trace = []
            if not pre_identities:
                pi.reset()

            post_obs, reward, done, info = env.step(nominal.tolist())
            post_force, post_identities, post_raw = contact_force(model, data)
            post_aligned = None if direction is None else float(post_force[:2] @ direction)
            if direction is not None:
                pi.observe_completed_step(post_aligned, bool(post_identities))
            controller_trace = [] if adapter is None else list(adapter.trace)
            native_success = bool(env.check_success())
            rows.append(
                {
                    "step": step,
                    "source": source,
                    "pre_state_sha256": array_hash(pre_state),
                    "pre_qpos_sha256": array_hash(pre_qpos),
                    "pre_qvel_sha256": array_hash(pre_qvel),
                    "pre_qpos": pre_qpos.tolist(),
                    "pre_qvel": pre_qvel.tolist(),
                    "pre_observation_hashes": observation_hashes(obs),
                    "nominal_action": nominal.tolist(),
                    "nominal_action_sha256": array_hash(nominal),
                    "pre_contact": bool(pre_identities),
                    "pre_contact_identities": pre_identities,
                    "pre_raw_contacts": pre_raw,
                    "pre_force_world_on_plate": pre_force.tolist(),
                    "stable_contact_steps": stable,
                    "direction_step": direction_step,
                    "direction_world_xy": None if direction is None else direction.tolist(),
                    "intervention": intervention,
                    "absolute_axis_command_n": command,
                    "controller_trace": controller_trace,
                    "post_contact": bool(post_identities),
                    "post_contact_identities": post_identities,
                    "post_raw_contacts": post_raw,
                    "post_force_world_on_plate": post_force.tolist(),
                    "post_aligned_force_n": post_aligned,
                    "post_state_sha256": array_hash(env.get_sim_state().copy()),
                    "post_qpos_sha256": array_hash(data.qpos.copy()),
                    "post_qvel_sha256": array_hash(data.qvel.copy()),
                    "reward": float(reward),
                    "done": bool(done),
                    "native_success": native_success,
                }
            )
            obs = post_obs
            if native_success:
                break
    except Exception as exc:
        error = repr(exc)
    finally:
        if adapter is not None:
            adapter.close()
        writer.release()
        native_final = bool(env.check_success()) if error is None else False
        env.close()

    (run / "telemetry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run / "inference.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inference_rows))
    all_controller_rows = [item for row in rows for item in row["controller_trace"]]
    receipt = {
        "gate": CONTRACT["gate"],
        "mode": mode,
        "contract_version": CONTRACT["version"],
        "task_language": task,
        "root": CONTRACT["root"],
        "policy_seed": CONTRACT["policy_seed"],
        "server_metadata": server_metadata,
        "table_friction_before": before.tolist(),
        "table_friction_readback": readback.tolist(),
        "target_n": CONTRACT["hybrid_target_n"] if mode == "hybrid" else None,
        "steps": len(rows),
        "direction_step": direction_step,
        "direction_record": direction_record,
        "first_intervention_step": next((row["step"] for row in rows if row["intervention"]), None),
        "intervention_steps": sum(row["intervention"] for row in rows),
        "all_torques_finite": all(item.get("finite", False) for item in all_controller_rows),
        "total_actuator_clips": sum(item.get("clip_count", 0) for item in all_controller_rows),
        "native_success": native_final,
        "error": error,
    }
    atomic_json(run / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    if error is not None:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
