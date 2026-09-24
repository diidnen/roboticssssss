#!/usr/bin/env python3
"""One-shot contact seek diagnostic for Gate S1 protocol tuning."""
import os, sys
from pathlib import Path
import numpy as np

TABERO = Path("/home/exouser/Tabero")
os.chdir(TABERO)
sys.path.insert(0, str(TABERO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(TABERO / "benchmarks/datasets/libero/assembled_hdf5")
os.environ.setdefault("LIBERO_CONFIG_DIR", str(TABERO / "benchmarks/datasets/libero/config"))
os.environ.setdefault("LIBERO_ASSETS_DATA_DIR", str(TABERO / "benchmarks/datasets/libero/USD"))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import tac_manip.tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tac_manip.utils.task_configs import setup_task_objects

ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
setup_task_objects("libero_object", 1)
env = gym.make(ENV_ID, cfg=parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)).unwrapped
obs, _ = env.reset()
obj = env.scene["cream_cheese_1"].data.root_pos_w[0].cpu().numpy()
ee = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
quat = env.scene["ee_frame"].data.target_quat_w[0, 0].cpu().numpy()
print("obj", obj)
print("ee0", ee)
print("quat", quat)

w,x,y,z = quat
angle = 2*np.arccos(np.clip(w,-1,1))
s = np.sqrt(max(1e-12,1-w*w))
aa = np.array([x,y,z])/s*angle if s>1e-6 else np.zeros(3)

def act(pos, g=0.01, f=0.0):
    a = torch.zeros((1,13), device=env.device)
    a[0,:3] = torch.tensor(pos, device=env.device)
    a[0,3:6] = torch.tensor(aa, device=env.device)
    a[0,6] = g
    h = f*0.5
    a[0,9] = h
    a[0,12] = h
    return a

# approach xy
target = obj.copy(); target[2] = obj[2] + 0.08
for i in range(40):
    alpha = (i+1)/40
    pos = (1-alpha)*ee + alpha*target
    obs,_,term,trunc,_ = env.step(act(pos, 0.04, 0))
    if term[0] or trunc[0]: break

contact_z = None
for dz in np.linspace(0.08, 0.0, 40):
    pos = obj.copy(); pos[2] = obj[2] + dz
    obs,_,term,trunc,_ = env.step(act(pos, 0.008, 0))
    dbg = env.action_manager.get_term("arm_action").debug_info
    fsq = float(dbg["f_sq_meas_raw"][0])
    f = obs["policy"]["gripper_net_force"][0,-1].cpu().numpy()
    print(f"dz={dz:.3f} fsq={fsq:.3f} fL={f[0]} fR={f[1]} gp={obs['policy']['gripper_pos'][0].mean().item():.4f}")
    if fsq > 0.2 and contact_z is None:
        contact_z = dz

print("contact_z_est", contact_z)
env.close()
simulation_app.close()
