"""Default-physics live pi0 sanity rollout for LIBERO Goal task 7."""
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

LIBERO = "/media/volume/newdata/exouser/flowdagger_e960a/bundle/e959c_bootstrap_resolved_r0_20260817T082827Z/flowdagger/flowdagger_pi05/openpi/third_party/libero"
CLIENT = "/home/exouser/SoftVTBench/openpi/upstream/packages/openpi-client/src"
CLIENT_DEPS = "/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/client_venv/lib/python3.10/site-packages"
for path in (CLIENT_DEPS, CLIENT, LIBERO):
    sys.path.insert(0, path)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

OUT = Path("/media/volume/data/exouser/activeforcing_knob_damping_v1_20260911")
DAMPING = float(os.environ.get("KNOB_DAMPING", "1.0"))


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def axis_angle(quaternion):
    q = np.asarray(quaternion, dtype=float).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if denominator < 1e-9 else q[:3] * 2.0 * np.arccos(q[3]) / denominator


def main():
    run = OUT / "noquery_sweep" / f"damping_{DAMPING:g}_root0_seed1000"
    if run.exists():
        raise RuntimeError(f"immutable run exists: {run}")
    run.mkdir(parents=True)
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, weights_only=False, **{k: v for k, v in kwargs.items() if k != "weights_only"})
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(7)
    init = suite.get_task_init_states(7)[0]
    env = OffScreenRenderEnv(
        bddl_file_name=Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(7)
    env.reset()
    obs = env.set_init_state(init)
    model = env.sim.model
    joint_metadata = []
    for joint_id in range(model.njnt):
        name = str(model.joint_id2name(joint_id))
        if "stove" in name.lower() or "knob" in name.lower():
            dof = int(model.jnt_dofadr[joint_id])
            joint_metadata.append(
                {
                    "joint_id": joint_id,
                    "name": name,
                    "type": int(model.jnt_type[joint_id]),
                    "qpos_address": int(model.jnt_qposadr[joint_id]),
                    "dof_address": dof,
                    "damping": float(model.dof_damping[dof]),
                    "frictionloss": float(model.dof_frictionloss[dof]),
                }
            )
    knob = next(item for item in joint_metadata if item["name"] == "flat_stove_1_button")
    model.dof_damping[knob["dof_address"]] = DAMPING
    damping_readback = float(model.dof_damping[knob["dof_address"]])
    client = WebsocketClientPolicy(host="127.0.0.1", port=8015)
    metadata = client.get_server_metadata()
    plan = collections.deque()
    rows = []
    writer = cv2.VideoWriter(str(run / "agentview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (224, 224))
    success = False
    first_success = None
    error = None
    try:
        for step in range(310):
            image = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
            wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if step < 10:
                action = np.r_[np.zeros(6), -1.0]
                source = "stabilize"
            else:
                if not plan:
                    state = np.r_[obs["robot0_eef_pos"], axis_angle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
                    request = {"observation/image": image, "observation/wrist_image": wrist, "observation/state": state, "prompt": task.language}
                    if step == 10:
                        request["_activeforcing_episode_start_seed"] = 1000
                    chunk = np.asarray(client.infer(request)["actions"], dtype=float)
                    plan.extend(chunk[:5])
                    print(f"[KNOB BASELINE] inference step={step} chunk={digest(chunk)[:12]}", flush=True)
                action = np.asarray(plan.popleft(), dtype=float)
                source = "online_pi0"
            pre_qpos = env.sim.data.qpos.copy()
            obs, reward, done, info = env.step(action.tolist())
            success_now = bool(env.check_success())
            rows.append(
                {
                    "step": step,
                    "source": source,
                    "nominal_action": action.tolist(),
                    "nominal_action_sha256": digest(action),
                    "pre_qpos_sha256": digest(pre_qpos),
                    "post_qpos": env.sim.data.qpos.copy().tolist(),
                    "reward": float(reward),
                    "native_success": success_now,
                }
            )
            if step % 50 == 0:
                print(f"[KNOB BASELINE] step={step} success={success_now}", flush=True)
            if success_now:
                success = True
                first_success = step
                break
    except Exception as exc:
        error = repr(exc)
    finally:
        writer.release()
        env.close()
    (run / "telemetry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = {
        "task_id": 7,
        "task_language": task.language,
        "root": 0,
        "policy_seed": 1000,
        "server_metadata": metadata,
        "joint_metadata": joint_metadata,
        "knob_damping_requested": DAMPING,
        "knob_damping_readback": damping_readback,
        "steps": len(rows),
        "native_success": success,
        "first_success_step": first_success,
        "error": error,
    }
    (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
