import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

LIBERO = "/media/volume/newdata/exouser/flowdagger_e960a/bundle/e959c_bootstrap_resolved_r0_20260817T082827Z/flowdagger/flowdagger_pi05/openpi/third_party/libero"
sys.path.insert(0, LIBERO)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

BASE = Path("/media/volume/data/exouser/activeforcing_knob_damping_v1_20260911/baseline_default_root0_seed1000")
rows = [json.loads(line) for line in (BASE / "telemetry.jsonl").read_text().splitlines()]
actions = [row["nominal_action"] for row in rows]
old_load = torch.load
torch.load = lambda *args, **kwargs: old_load(*args, weights_only=False, **{k: v for k, v in kwargs.items() if k != "weights_only"})
suite = benchmark.get_benchmark_dict()["libero_goal"]()
task = suite.get_task(7)
env = OffScreenRenderEnv(
    bddl_file_name=Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
    camera_heights=256,
    camera_widths=256,
)
env.seed(7)
env.reset()
env.set_init_state(suite.get_task_init_states(7)[0])
model, data = env.sim.model, env.sim.data
robot = {i for i in range(model.ngeom) if str(model.geom_id2name(i)).startswith("gripper0_")}
print("JOINTS")
for joint_id in range(model.njnt):
    name = str(model.joint_id2name(joint_id))
    if "stove" in name.lower() or "button" in name.lower():
        print(joint_id, name, int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]))
print("BODIES")
for body_id in range(model.nbody):
    name = str(model.body_id2name(body_id))
    if "stove" in name.lower() or "button" in name.lower():
        print(body_id, name)
print("GEOMS")
for geom_id in range(model.ngeom):
    name = str(model.geom_id2name(geom_id))
    body_name = str(model.body_id2name(model.geom_bodyid[geom_id]))
    if "stove" in name.lower() or "button" in name.lower() or "stove" in body_name.lower() or "button" in body_name.lower():
        print(geom_id, name, body_name)
print("CONTACTS")
for step, action in enumerate(actions):
    env.step(action)
    if 45 <= step <= 90:
        found = []
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if contact.geom1 not in robot and contact.geom2 not in robot:
                continue
            name1, name2 = model.geom_id2name(contact.geom1), model.geom_id2name(contact.geom2)
            if "stove" not in str(name1).lower() and "stove" not in str(name2).lower() and "button" not in str(name1).lower() and "button" not in str(name2).lower():
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model._model, data._data, contact_id, wrench)
            found.append((name1, name2, wrench.tolist()))
        if found:
            print(step, "qpos", float(data.qpos[40]), "qvel", float(data.qvel[36]), "action", action, "contacts", found)
env.close()
