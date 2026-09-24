"""Deterministic state/contact reconstruction of already-recorded actions.

This deliberately makes no policy connection and is not a new rollout.  It is
only a measurement pass over the exact action arrays retained with the three
completed online-policy receipts.
"""
import hashlib
import json
import pathlib
import sys
import time

import mujoco
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SRC = pathlib.Path("/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/rollouts")
OUT = pathlib.Path("/media/volume/data/exouser/pi0_nontransport_20260910/audit")
TARGET_TERMS = {0: ("cabinet", "drawer"), 5: ("plate",), 7: ("stove", "knob")}

def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def name(model, typ, idx):
    # robosuite exposes a compatibility wrapper rather than raw MjModel.
    method = {mujoco.mjtObj.mjOBJ_GEOM: "geom_id2name",
              mujoco.mjtObj.mjOBJ_BODY: "body_id2name",
              mujoco.mjtObj.mjOBJ_JOINT: "joint_id2name"}[typ]
    x = getattr(model, method)(int(idx))
    return x if x is not None else f"{typ.name}:{idx}"

def main():
    original = torch.load
    torch.load = lambda *a, **kw: original(*a, weights_only=False, **kw)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    report = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "measurement": "deterministic replay of archived raw actions; no policy inference/rollout",
              "tasks": []}
    for tid in (0, 5, 7):
        task = suite.get_task(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        actions = np.load(SRC / f"task{tid}_init0_native_pi0_libero/actions.npy")
        receipt = json.loads((SRC / f"task{tid}_init0_native_pi0_libero/receipt.json").read_text())
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(7); env.reset(); env.set_init_state(suite.get_task_init_states(tid)[0])
        m, d = env.sim.model, env.sim.data
        geom_names = [name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
        body_names = [name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        joint_names = [name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        terms = TARGET_TERMS[tid]
        target_geoms = [i for i,n in enumerate(geom_names) if any(t in n.lower() for t in terms)]
        robot_geoms = [i for i,n in enumerate(geom_names) if any(t in n.lower() for t in ("gripper", "finger", "robot0"))]
        target_bodies = [i for i,n in enumerate(body_names) if any(t in n.lower() for t in terms)]
        target_joints = [i for i,n in enumerate(joint_names) if any(t in n.lower() for t in terms)]
        start_pos = {body_names[i]: d.xpos[i].tolist() for i in target_bodies}
        start_qpos = {joint_names[i]: float(d.qpos[m.jnt_qposadr[i]]) for i in target_joints}
        contacts=[]; frames=[]
        for step, action in enumerate(actions):
            obs, reward, done, info = env.step(action.tolist())
            hit=[]
            for ci in range(d.ncon):
                c=d.contact[ci]; a,b=int(c.geom1),int(c.geom2)
                if (a in robot_geoms and b in target_geoms) or (b in robot_geoms and a in target_geoms):
                    hit.append([geom_names[a], geom_names[b]])
            if hit: contacts.append({"step":step,"pairs":hit})
            frames.append({"step":step,
              "target_positions":{body_names[i]:d.xpos[i].tolist() for i in target_bodies},
              "target_qpos":{joint_names[i]:float(d.qpos[m.jnt_qposadr[i]]) for i in target_joints}})
        end_pos = {body_names[i]: d.xpos[i].tolist() for i in target_bodies}
        end_qpos = {joint_names[i]: float(d.qpos[m.jnt_qposadr[i]]) for i in target_joints}
        changes=[]
        for bn in start_pos:
            changes.append({"body":bn,"translation_m":float(np.linalg.norm(np.array(end_pos[bn])-np.array(start_pos[bn])))})
        for jn in start_qpos:
            changes.append({"joint":jn,"qpos_delta":end_qpos[jn]-start_qpos[jn]})
        entry={"task_id":tid,"language":task.language,"archived_action_sha256":sha(actions),
          "receipt_action_count":len(receipt["actions"]),"model_targets":{"geom_names":[geom_names[i] for i in target_geoms],"body_names":[body_names[i] for i in target_bodies],"joint_names":[joint_names[i] for i in target_joints]},
          "robot_target_contacts":contacts,"robot_target_contact_steps":[x["step"] for x in contacts],"start": {"positions":start_pos,"qpos":start_qpos},"end":{"positions":end_pos,"qpos":end_qpos},"net_changes":changes,"native_success_after_reconstruction":bool(env.check_success())}
        (OUT / f"task{tid}_contact_motion_reconstruction.json").write_text(json.dumps(entry,indent=2)+"\n")
        report["tasks"].append(entry)
        env.close()
    (OUT/"CONTACT_MOTION_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
if __name__ == "__main__": main()
