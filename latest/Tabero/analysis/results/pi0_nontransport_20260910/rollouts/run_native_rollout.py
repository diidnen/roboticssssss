"""One online frozen-policy LIBERO rollout. No action replay or expert policy."""
import collections
import hashlib
import json
import math
import pathlib
import sys
import time

import cv2
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

OUT = pathlib.Path(__file__).resolve().parent
TASK_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
INIT_INDEX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
HORIZON = 300
WAIT_STEPS = 10
REPLAN_STEPS = 5
SEED = 7
CHECKPOINT = "/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero"
TREE_SHA256 = "92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103"

def quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def state_identity(x):
    x = np.ascontiguousarray(np.asarray(x))
    return {"index": INIT_INDEX, "sha256": hashlib.sha256(x.tobytes()).hexdigest(), "shape": list(x.shape), "dtype": str(x.dtype)}

def main():
    original = torch.load
    torch.load = lambda *a, **kw: original(*a, weights_only=False, **kw)
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(TASK_ID)
    states = benchmark.get_benchmark_dict()["libero_goal"]().get_task_init_states(TASK_ID)
    init = states[INIT_INDEX]
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(SEED)
    rollout_dir = OUT / f"task{TASK_ID}_init{INIT_INDEX}_native_pi0_libero"
    rollout_dir.mkdir(exist_ok=True)
    receipt = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "policy_config":"pi0_libero", "checkpoint_path":CHECKPOINT,"checkpoint_tree_sha256":TREE_SHA256,"task_suite":"libero_goal","task_id":TASK_ID,"language":task.language,"init_state":state_identity(init),"horizon":HORIZON,"stabilization_dummy_steps":WAIT_STEPS,"replan_steps":REPLAN_STEPS,"action_source":"online websocket frozen pi0_libero only after stabilization","success_rule":"native env.step done flag; final wrapped env.check_success recorded","actions":[],"inference_receipts":[]}
    writer = None
    try:
        env.reset(); obs = env.set_init_state(init)
        writer = cv2.VideoWriter(str(rollout_dir / "agentview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (224,224))
        client = WebsocketClientPolicy("127.0.0.1", 8010)
        plan = collections.deque(); done = False
        for step in range(HORIZON + WAIT_STEPS):
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
            wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if step < WAIT_STEPS:
                action = np.array([0.0]*6 + [-1.0], dtype=float)
                source = "stabilization_dummy"
            else:
                if not plan:
                    element = {"observation/image":img,"observation/wrist_image":wrist,"observation/state":np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),"prompt":str(task.language)}
                    t0 = time.monotonic(); answer = client.infer(element); elapsed = time.monotonic()-t0
                    chunk = np.asarray(answer["actions"])
                    if len(chunk) < REPLAN_STEPS: raise RuntimeError("policy action chunk shorter than frozen cadence")
                    plan.extend(chunk[:REPLAN_STEPS])
                    receipt["inference_receipts"].append({"env_step":step,"latency_seconds":elapsed,"chunk_shape":list(chunk.shape)})
                action = np.asarray(plan.popleft(), dtype=float); source = "online_pi0_libero"
            obs, reward, done, info = env.step(action.tolist())
            receipt["actions"].append({"env_step":step,"source":source,"action":action.tolist(),"reward":float(reward),"done":bool(done)})
            if done: break
        receipt.update({"finished_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),"steps_executed":len(receipt["actions"]),"success":bool(done),"final_native_check_success":bool(env.check_success())})
    except Exception as exc:
        receipt.update({"error":repr(exc),"success":False})
    finally:
        if writer is not None: writer.release()
        env.close()
    np.save(rollout_dir / "actions.npy", np.array([x["action"] for x in receipt["actions"]], dtype=float))
    (rollout_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k:receipt.get(k) for k in ("success","steps_executed","error","final_native_check_success")}, indent=2))
    return 0 if "error" not in receipt else 1

if __name__ == "__main__": sys.exit(main())
