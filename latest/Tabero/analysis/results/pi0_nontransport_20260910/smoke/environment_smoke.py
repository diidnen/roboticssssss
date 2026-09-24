import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

OUT = pathlib.Path(__file__).resolve().parents[1]
TASKS = {0: "open the middle drawer of the cabinet", 5: "push the plate to the front of the stove", 7: "turn on the stove"}

def init_digest(state):
    arr = np.ascontiguousarray(np.asarray(state))
    return {"sha256": hashlib.sha256(arr.tobytes()).hexdigest(), "shape": list(arr.shape), "dtype": str(arr.dtype)}

def main():
    # The supplied, trusted LIBERO init-state files predate PyTorch 2.6's safe default.
    # This applies only to their read-only load; it is not policy/checkpoint loading.
    original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: original_torch_load(*args, weights_only=False, **kwargs)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    result = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "suite": "libero_goal", "tasks": []}
    for task_id, required_language in TASKS.items():
        task = suite.get_task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        entry = {"task_id": task_id, "language": task.language, "required_language": required_language, "language_matches": task.language == required_language, "bddl": str(bddl), "init_states": []}
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(7)
        try:
            reset_obs = env.reset()
            states = suite.get_task_init_states(task_id)
            entry["init_state_count"] = len(states)
            obs = env.set_init_state(states[0])
            entry["init_states"].append(init_digest(states[0]))
            entry["reset_keys"] = sorted(obs.keys())
            entry["agentview_shape"] = list(obs["agentview_image"].shape)
            entry["wrist_shape"] = list(obs["robot0_eye_in_hand_image"].shape)
            np.save(OUT / "smoke" / f"task{task_id}_agentview.npy", obs["agentview_image"])
            np.save(OUT / "smoke" / f"task{task_id}_wrist.npy", obs["robot0_eye_in_hand_image"])
            entry["native_success_predicate"] = "OffScreenRenderEnv.check_success() -> wrapped task env._check_success()"
            entry["success_at_init"] = bool(env.check_success())
            entry["rendered"] = True
            entry["valid"] = entry["language_matches"] and not entry["success_at_init"]
        except Exception as exc:
            entry["valid"] = False
            entry["error"] = repr(exc)
        finally:
            env.close()
        result["tasks"].append(entry)
    result["passed"] = all(x["valid"] for x in result["tasks"])
    (OUT / "smoke" / "environment_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2

if __name__ == "__main__":
    sys.exit(main())
