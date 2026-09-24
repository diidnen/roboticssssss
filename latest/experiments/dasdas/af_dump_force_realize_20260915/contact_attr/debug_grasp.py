"""Debug-only left grasp after the official right→left handover.

Official scripted_establish_grasp, VLA, assets, and original squeeze inner
are not modified. Left contact_point_id / grasp_dis / optional TCP-local-X
roll come from debug_grasp.json.

TCP roll keeps the grasp translation and spins fingers about the gripper
X axis used by RoboTwin's -0.12 m TCP offset (closing/approach).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import transforms3d as t3d

from envs.utils import ArmTag
from envs.utils.action import Action

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "debug_grasp.json"


def load_debug_grasp(path: Path | None = None) -> dict:
    path = Path(path or DEFAULT_CONFIG)
    if not path.exists():
        return {
            "left_contact_point_id": 1,
            "left_grasp_dis": 0.0,
            "tcp_roll_about_local_x_deg": 0.0,
            "note": "baseline wall pinch; not opposing",
        }
    return json.loads(path.read_text())


def write_debug_grasp(payload: dict, path: Path | None = None) -> Path:
    path = Path(path or DEFAULT_CONFIG)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _rotate_pose_local_x(pose, deg: float):
    pose = list(pose)
    p = pose[:3]
    q = pose[3:]
    delta = t3d.quaternions.axangle2quat([1.0, 0.0, 0.0], float(np.radians(deg)))
    qn = t3d.quaternions.qmult(q, delta)
    return list(p) + [float(x) for x in qn]


def _translate_pose_local(pose, xyz) -> list:
    pose = list(pose)
    p = np.asarray(pose[:3], dtype=float)
    q = pose[3:]
    rot = t3d.quaternions.quat2mat(q)
    p = p + rot @ np.asarray(xyz, dtype=float)
    return [float(x) for x in p] + [float(x) for x in q]


def _rotate_inverse(quat_wxyz, vec):
    w, x, y, z = [float(v) for v in quat_wxyz]
    qvec = np.array([x, y, z], dtype=float)
    v = np.asarray(vec, dtype=float)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def centering_lateral_y(grasp_pose, obj_pose, max_abs: float = 0.08) -> dict:
    """Shift TCP along local Y so its object-xz sits on the bin axis.

    Official side-face contact points hover over one wall. The finger-open
    axis is TCP local Y; a small Y offset is what puts the closing axis
    through the bin instead of through that wall's 5 mm thickness.
    """
    p = np.asarray(grasp_pose[:3], dtype=float)
    rot = t3d.quaternions.quat2mat(grasp_pose[3:])
    y_world = rot[:, 1]
    tcp_obj = _rotate_inverse(obj_pose.q, p - np.asarray(obj_pose.p, dtype=float))
    y_obj = _rotate_inverse(obj_pose.q, y_world)
    axis_xz = np.array([y_obj[0], y_obj[2]], dtype=float)
    tcp_xz = np.array([tcp_obj[0], tcp_obj[2]], dtype=float)
    denom = float(np.dot(axis_xz, axis_xz)) + 1e-12
    raw = float(-np.dot(axis_xz, tcp_xz) / denom)
    clipped = float(np.clip(raw, -abs(max_abs), abs(max_abs)))
    return {
        "raw_m": raw,
        "clipped_m": clipped,
        "tcp_obj": [float(x) for x in tcp_obj],
        "y_obj": [float(x) for x in y_obj],
        "x_obj": [float(x) for x in _rotate_inverse(obj_pose.q, rot[:, 0])],
    }


def _left_grasp_actions(
    task,
    cid: int,
    grasp_dis: float,
    tcp_roll_deg: float,
    lateral_y_m: float = 0.0,
    center_over_bin: bool = False,
):
    place = ArmTag("left")
    chosen = task.choose_grasp_pose(
        task.deskbin,
        arm_tag=place,
        pre_dis=0.08,
        target_dis=float(grasp_dis),
        contact_point_id=int(cid),
    )
    if not chosen or chosen[0] is None or chosen[1] is None:
        raise RuntimeError(f"choose_grasp_pose failed for contact_point_id={cid}")
    pre, grasp = chosen
    if abs(float(tcp_roll_deg)) > 1e-9:
        pre = _rotate_pose_local_x(pre, tcp_roll_deg)
        grasp = _rotate_pose_local_x(grasp, tcp_roll_deg)
    info = None
    extra = float(lateral_y_m)
    if center_over_bin:
        info = centering_lateral_y(grasp, task.deskbin.get_pose())
        extra = extra + float(info["clipped_m"])
    if abs(extra) > 1e-9:
        pre = _translate_pose_local(pre, [0.0, extra, 0.0])
        grasp = _translate_pose_local(grasp, [0.0, extra, 0.0])
    actions = (
        [
            Action(place, "move", target_pose=pre),
            Action(place, "close", target_gripper_pos=0.0),
        ]
        if pre == grasp
        else [
            Action(place, "move", target_pose=pre),
            Action(place, "move", target_pose=grasp, constraint_pose=[1, 1, 1, 0, 0, 0]),
            Action(place, "close", target_gripper_pos=0.0),
        ]
    )
    return actions, extra, info


def scripted_establish_grasp_debug(
    task,
    left_contact_point_id: int = 1,
    left_grasp_dis: float = 0.0,
    tcp_roll_about_local_x_deg: float = 0.0,
    tcp_lateral_y_m: float = 0.0,
    contact_yaw_about_bin_up_deg: float = 0.0,
    center_over_bin: bool = False,
) -> str:
    """Same handover as v4; left grasp uses debug id / grasp_dis / TCP roll.

    contact_yaw_about_bin_up_deg is accepted for older configs but ignored;
    TCP-local-X roll is the orientation fix that keeps translation.
    """
    del contact_yaw_about_bin_up_deg
    deskbin_pose = task.deskbin.get_pose().p
    grasp_arm = ArmTag("left" if deskbin_pose[0] < 0 else "right")
    place_arm = ArmTag("left")
    cid = int(left_contact_point_id)
    roll = float(tcp_roll_about_local_x_deg)
    extra = float(tcp_lateral_y_m)
    info = None
    if grasp_arm == "right":
        task.move(
            task.grasp_actor(task.deskbin, arm_tag=grasp_arm, pre_grasp_dis=0.08, contact_point_id=3)
        )
        task.move(task.move_by_displacement(grasp_arm, z=0.08, move_axis="arm"))
        task.move(
            task.place_actor(
                task.deskbin,
                target_pose=task.middle_pose,
                arm_tag=grasp_arm,
                pre_dis=0.08,
                dis=0.01,
            )
        )
        task.move(task.move_by_displacement(grasp_arm, z=0.1, move_axis="arm"))
        left_actions, extra, info = _left_grasp_actions(
            task, cid, left_grasp_dis, roll, tcp_lateral_y_m, center_over_bin
        )
        task.move(task.back_to_origin(grasp_arm), (place_arm, left_actions))
    else:
        left_actions, extra, info = _left_grasp_actions(
            task, cid, left_grasp_dis, roll, tcp_lateral_y_m, center_over_bin
        )
        task.move((place_arm, left_actions))
    task._af_grasp_arm_tag = "left"
    task._af_debug_grasp = {
        "left_contact_point_id": cid,
        "left_grasp_dis": float(left_grasp_dis),
        "tcp_roll_about_local_x_deg": roll,
        "tcp_lateral_y_m": float(extra),
        "center_over_bin": bool(center_over_bin),
        "centering": info,
        "handover": str(grasp_arm) == "right",
    }
    return "left"