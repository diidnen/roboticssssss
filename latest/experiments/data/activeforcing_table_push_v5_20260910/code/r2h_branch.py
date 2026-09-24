"""One fresh-process V5 hybrid-force R2H branch; no policy inference."""
import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

LIBERO = "/media/volume/newdata/exouser/flowdagger_e960a/bundle/e959c_bootstrap_resolved_r0_20260817T082827Z/flowdagger/flowdagger_pi05/openpi/third_party/libero"
sys.path.insert(0, LIBERO)

import mujoco
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import nullspace_torques, opspace_matrices, orientation_error

OUT = Path("/media/volume/data/exouser/activeforcing_table_push_v5_20260910")
V2 = Path("/media/volume/data/exouser/activeforcing_table_push_v2_20260910/calibration/calibration_root0_mu1.2_residual_plus0.00")
OSC_SHA256 = "cfa62a0e719bc53ef0c701efa66f7b3e2272d4fca2150ec73c05bb145eba85cb"
ROBOT_GEOMS = {
    "gripper0_hand_collision",
    "gripper0_finger1_collision",
    "gripper0_finger1_pad_collision",
    "gripper0_finger2_collision",
    "gripper0_finger2_pad_collision",
}


def array_hash(value):
    arr = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def observation_hash(obs):
    payload = b"".join(
        np.ascontiguousarray(np.asarray(obs[key])).tobytes() for key in sorted(obs)
    )
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def contact_force(model, data):
    plate_geoms = {
        i for i in range(model.ngeom) if str(model.geom_id2name(i)).startswith("plate_1_")
    }
    total = np.zeros(3)
    identities, raw = [], []
    for index in range(data.ncon):
        contact = data.contact[index]
        name1 = model.geom_id2name(contact.geom1)
        name2 = model.geom_id2name(contact.geom2)
        relevant = (name1 in ROBOT_GEOMS and contact.geom2 in plate_geoms) or (
            name2 in ROBOT_GEOMS and contact.geom1 in plate_geoms
        )
        if not relevant:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model._model, data._data, index, wrench)
        force = contact.frame.reshape(3, 3).T @ wrench[:3]
        force = force if contact.geom2 in plate_geoms else -force
        total += force
        identities.append([name1, name2])
        raw.append(
            {
                "index": index,
                "geom1": name1,
                "geom2": name2,
                "wrench_contact": wrench.tolist(),
                "force_world_on_plate": force.tolist(),
            }
        )
    return total, identities, raw


def controller_goal_hash(env):
    controller = env.robots[0].controller
    values = []
    for key in ("goal_pos", "goal_ori", "goal_ori_quat", "goal_vel", "goal_torque"):
        if hasattr(controller, key):
            values.append(np.asarray(getattr(controller, key)).ravel())
    return array_hash(np.concatenate(values) if values else np.zeros(0))


class HybridForceAdapter:
    """Instance-local copy of installed OSC math for enabled hybrid mode."""

    def __init__(self, env, direction_xy, enabled, axis_command_n):
        self.env = env
        self.controller = env.robots[0].controller
        self.original = self.controller.run_controller
        self.direction = np.asarray([direction_xy[0], direction_xy[1], 0.0], dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.enabled = bool(enabled)
        self.axis_command_n = float(axis_command_n)
        self.trace = []

        def run(_controller):
            if not self.enabled:
                base = np.asarray(self.original(), dtype=float)
                self.trace.append(
                    {
                        "mode": "disabled_original_bound_method",
                        "base_torque": base.tolist(),
                        "raw_returned_torque": base.tolist(),
                        "disabled_raw_equals_base_exact": True,
                        "finite": bool(np.isfinite(base).all()),
                        "shape": list(base.shape),
                        "clip_count": 0,
                    }
                )
                return base

            c = self.controller
            c.update()
            if c.interpolator_pos is not None:
                desired_pos = (
                    c.interpolator_pos.get_interpolated_goal()
                    if c.interpolator_pos.order == 1
                    else None
                )
            else:
                desired_pos = np.array(c.goal_pos)
            if desired_pos is None:
                raise RuntimeError("unsupported position interpolator")

            if c.interpolator_ori is not None:
                c.relative_ori = orientation_error(c.ee_ori_mat, c.ori_ref)
                ori_error = c.interpolator_ori.get_interpolated_goal()
            else:
                ori_error = orientation_error(np.array(c.goal_ori), c.ee_ori_mat)

            position_error = desired_pos - c.ee_pos
            desired_force = position_error * np.asarray(c.kp[:3]) + (-c.ee_pos_vel) * np.asarray(c.kd[:3])
            desired_torque = ori_error * np.asarray(c.kp[3:6]) + (-c.ee_ori_vel) * np.asarray(c.kd[3:6])
            lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
                c.mass_matrix, c.J_full, c.J_pos, c.J_ori
            )
            if c.uncoupling:
                stock_force = lambda_pos @ desired_force
                stock_orientation_torque = lambda_ori @ desired_torque
            else:
                stock_wrench = lambda_full @ np.concatenate([desired_force, desired_torque])
                stock_force = stock_wrench[:3]
                stock_orientation_torque = stock_wrench[3:]

            stock_axis = float(self.direction @ stock_force)
            hybrid_force = stock_force + self.direction * (self.axis_command_n - stock_axis)
            hybrid_wrench = np.concatenate([hybrid_force, stock_orientation_torque])
            task_torque = c.J_full.T @ hybrid_wrench
            gravity = np.asarray(c.torque_compensation, dtype=float)
            nullspace = nullspace_torques(
                c.mass_matrix, nullspace_matrix, c.initial_joint, c.joint_pos, c.joint_vel
            )
            raw = task_torque + gravity + nullspace
            c.torques = raw
            Controller.run_controller(c)

            low, high = env.robots[0].torque_limits
            self.trace.append(
                {
                    "mode": "hybrid_axis_replacement",
                    "desired_force_pre_decoupling": desired_force.tolist(),
                    "stock_decoupled_force": stock_force.tolist(),
                    "stock_axis_force_n": stock_axis,
                    "axis_command_n": self.axis_command_n,
                    "hybrid_decoupled_force": hybrid_force.tolist(),
                    "hybrid_axis_force_n": float(self.direction @ hybrid_force),
                    "orientation_decoupled_torque": stock_orientation_torque.tolist(),
                    "J_pos_world": c.J_pos.tolist(),
                    "J_full": c.J_full.tolist(),
                    "task_torque": task_torque.tolist(),
                    "gravity_compensation": gravity.tolist(),
                    "nullspace_torque": nullspace.tolist(),
                    "raw_returned_torque": raw.tolist(),
                    "finite": bool(np.isfinite(raw).all()),
                    "shape": list(raw.shape),
                    "clip_count": int(np.sum((raw < low) | (raw > high))),
                    "min_margin": float(np.min(np.minimum(raw - low, high - raw))),
                }
            )
            return raw

        self.controller.run_controller = types.MethodType(run, self.controller)

    def close(self):
        self.controller.run_controller = self.original


def build_env():
    original_load = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.pop("weights_only", None)
        return original_load(*args, weights_only=False, **kwargs)

    torch.load = compatible_load
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(5)
    init = suite.get_task_init_states(5)[0]
    env = OffScreenRenderEnv(
        bddl_file_name=Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(7)
    env.reset()
    obs = env.set_init_state(init)
    table = next(
        i for i in range(env.sim.model.ngeom) if env.sim.model.geom_id2name(i) == "table_collision"
    )
    env.sim.model.geom_friction[table] = [1.2, 0.005, 0.0001]
    return env, obs, table


def main():
    step = int(sys.argv[1])
    kind = sys.argv[2]
    command_n = float(sys.argv[3])
    out = OUT / "branches" / f"s{step}_{kind}_{command_n:+.0f}N.json"
    if out.exists():
        raise RuntimeError(f"immutable branch exists: {out}")

    telemetry = [json.loads(line) for line in (V2 / "telemetry.jsonl").read_text().splitlines()]
    executed = np.asarray([row["executed_action"] for row in telemetry], dtype=float)
    nominal = np.asarray([row["nominal_action"] for row in telemetry], dtype=float)
    receipt = json.loads((V2 / "receipt.json").read_text())
    direction_xy = np.asarray(receipt["direction_record"]["direction_world_xy"], dtype=float)

    env, obs, table = build_env()
    for action in executed[:step]:
        obs, _, _, _ = env.step(action.tolist())

    pre_state = env.get_sim_state().copy()
    pre_qpos = env.sim.data.qpos.copy()
    pre_qvel = env.sim.data.qvel.copy()
    pre_goal_hash = controller_goal_hash(env)
    pre_force, pre_contacts, pre_raw = contact_force(env.sim.model, env.sim.data)
    adapter = None
    if kind != "untouched":
        adapter = HybridForceAdapter(
            env,
            direction_xy,
            enabled=(kind == "hybrid"),
            axis_command_n=command_n,
        )

    post_obs, reward, done, info = env.step(nominal[step].tolist())
    post_state = env.get_sim_state().copy()
    post_qpos = env.sim.data.qpos.copy()
    post_qvel = env.sim.data.qvel.copy()
    post_force, post_contacts, post_raw = contact_force(env.sim.model, env.sim.data)
    arm_indices = env.robots[0]._ref_joint_actuator_indexes
    final_ctrl = env.sim.data.ctrl[arm_indices].copy()
    low, high = env.robots[0].torque_limits
    trace = [] if adapter is None else adapter.trace

    document = {
        "step": step,
        "kind": kind,
        "axis_command_n": command_n,
        "source": "V2 zero-residual official fine-tuned pi0 fixed action prefix; replay only",
        "source_osc_sha256": OSC_SHA256,
        "new_task_outcomes": 0,
        "seed": 7,
        "table_friction": env.sim.model.geom_friction[table].tolist(),
        "pre_state": {
            "hash": array_hash(pre_state),
            "array": pre_state.tolist(),
            "qpos": pre_qpos.tolist(),
            "qvel": pre_qvel.tolist(),
        },
        "post_state": {
            "hash": array_hash(post_state),
            "array": post_state.tolist(),
            "qpos": post_qpos.tolist(),
            "qvel": post_qvel.tolist(),
        },
        "pre_controller_goal_hash": pre_goal_hash,
        "post_controller_goal_hash": controller_goal_hash(env),
        "observation_hash_pre": observation_hash(obs),
        "observation_hash_post": observation_hash(post_obs),
        "nominal_action": nominal[step].tolist(),
        "nominal_action_sha256": array_hash(nominal[step]),
        "direction_xy": direction_xy.tolist(),
        "direction_hash": array_hash(direction_xy),
        "no_replan": True,
        "pre_force_world_on_plate": pre_force.tolist(),
        "pre_contacts": pre_contacts,
        "pre_raw_contacts": pre_raw,
        "post_force_world_on_plate": post_force.tolist(),
        "post_contacts": post_contacts,
        "post_raw_contacts": post_raw,
        "post_aligned_force_n": float(post_force[:2] @ direction_xy),
        "final_arm_ctrl": final_ctrl.tolist(),
        "final_arm_ctrl_hash": array_hash(final_ctrl),
        "final_arm_ctrl_min_margin": float(np.min(np.minimum(final_ctrl - low, high - final_ctrl))),
        "adapter_installed": adapter is not None,
        "adapter_enabled": kind == "hybrid",
        "adapter_trace": trace,
        "reward": float(reward),
        "done": bool(done),
    }
    if adapter is not None:
        adapter.close()
    env.close()
    atomic_json(out, document)


if __name__ == "__main__":
    main()
