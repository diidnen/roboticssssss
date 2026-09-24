"""Fresh-process one-step knob yaw-torque interface branch."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

LIBERO = "/media/volume/newdata/exouser/flowdagger_e960a/bundle/e959c_bootstrap_resolved_r0_20260817T082827Z/flowdagger/flowdagger_pi05/openpi/third_party/libero"
V5_CODE = "/media/volume/data/exouser/activeforcing_table_push_v5_20260910/code"
CODE = Path("/media/volume/data/exouser/activeforcing_knob_damping_v2_20260911/code")
for path in (LIBERO, V5_CODE, str(CODE)):
    sys.path.insert(0, path)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from r2h_branch import array_hash
from knob_yaw_controller import TorqueSafeMinimumYawTorqueAdapter

OUT = CODE.parent
SOURCE = Path("/media/volume/data/exouser/activeforcing_knob_damping_v1_20260911/noquery_sweep/damping_256_root0_seed1000")


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def build_env():
    old_load = torch.load
    torch.load = lambda *args, **kwargs: old_load(*args, weights_only=False, **{k: v for k, v in kwargs.items() if k != "weights_only"})
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(7)
    env = OffScreenRenderEnv(bddl_file_name=Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file, camera_heights=256, camera_widths=256)
    env.seed(7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(7)[0])
    env.sim.model.dof_damping[36] = 256.0
    return env, obs


def main():
    kind, command = sys.argv[1], float(sys.argv[2])
    output = OUT / "r2" / f"{kind}_{command:+.0f}Nm.json"
    if output.exists():
        raise RuntimeError(f"immutable branch exists: {output}")
    rows = [json.loads(line) for line in (SOURCE / "telemetry.jsonl").read_text().splitlines()]
    actions = np.asarray([row["nominal_action"] for row in rows], dtype=float)
    step = 90
    env, obs = build_env()
    for action in actions[:step]:
        obs, _, _, _ = env.step(action.tolist())
    pre_state = env.get_sim_state().copy()
    adapter = None
    yaw_sign = np.sign(actions[step, 5])
    if kind != "untouched":
        adapter = TorqueSafeMinimumYawTorqueAdapter(env, yaw_sign, kind == "minimum", command)
    obs, reward, done, info = env.step(actions[step].tolist())
    post_state = env.get_sim_state().copy()
    trace = [] if adapter is None else list(adapter.trace)
    arm_indices = env.robots[0]._ref_joint_actuator_indexes
    document = {
        "kind": kind,
        "minimum_yaw_torque_nm": command,
        "step": step,
        "yaw_sign": float(yaw_sign),
        "pre_state_sha256": array_hash(pre_state),
        "post_state_sha256": array_hash(post_state),
        "post_qpos": env.sim.data.qpos.copy().tolist(),
        "post_qvel": env.sim.data.qvel.copy().tolist(),
        "knob_post_qpos": float(env.sim.data.qpos[40]),
        "knob_post_qvel": float(env.sim.data.qvel[36]),
        "final_ctrl": env.sim.data.ctrl[arm_indices].copy().tolist(),
        "adapter_trace": trace,
        "finite_torque": all(item.get("finite", False) for item in trace) if trace else True,
        "total_clip_count": sum(item.get("clip_count", 0) for item in trace),
        "stock_yaw_never_reduced": all(item["applied_yaw_torque_nm"] + 1e-10 >= item["stock_yaw_torque_nm"] for item in trace if item.get("mode") == "torque_safe_minimum_world_yaw_torque"),
    }
    if adapter is not None:
        adapter.close()
    env.close()
    atomic_json(output, document)


if __name__ == "__main__":
    main()
