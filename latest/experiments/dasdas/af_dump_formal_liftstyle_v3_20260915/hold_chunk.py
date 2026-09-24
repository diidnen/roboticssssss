"""Scripted 8x64 hold-chunk used by v3 training and this online selector."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


OLD = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "experiments/af_dump_original_restore_20260912"
)
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))
from native_original_motion_features import build_features


def actor_velocity(task) -> np.ndarray:
    entity = task.deskbin.actor if hasattr(task.deskbin, "actor") else task.deskbin
    import sapien

    component = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    if component is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(component.linear_velocity, dtype=np.float64).reshape(-1)[:3]


def finger_joints(task) -> np.ndarray:
    return np.asarray(
        [float(task.robot.get_left_gripper_val()), float(task.robot.get_right_gripper_val())],
        dtype=np.float64,
    )


def build_hold_feature(task) -> dict:
    arm = str(getattr(task, "_af_grasp_arm_tag", "left") or "left")
    contact = task.get_actor_gripper_contact_forces(task.deskbin, arm)
    opening = (
        float(task.robot.get_left_gripper_val())
        if arm == "left"
        else float(task.robot.get_right_gripper_val())
    )
    force = float(contact.get("single_finger_normal_force_n") or 0.0)
    raw = [
        {
            "object_id": "deskbin",
            "gripper_opening": opening,
            "left_fx": 0.0,
            "left_fy": 0.0,
            "left_fz": force,
            "right_fx": 0.0,
            "right_fy": 0.0,
            "right_fz": force,
        }
    ]
    ee = np.asarray(task.get_arm_pose(arm)[:3], dtype=np.float64)
    chunk = np.repeat(ee.reshape(1, 3), 8, axis=0)
    feature = build_features(raw, finger_joints(task), actor_velocity(task), chunk)
    feature["motion_acquisition"] = "scripted_eef_hold_interp_8_like_lift_v5"
    feature["source"] = "SCRIPTED_HOLD_CHUNK_DUMP_SHARED_PREFIX"
    feature["grasp_arm"] = arm
    return feature
