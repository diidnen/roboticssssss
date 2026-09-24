"""Dump established grasp, copied from v3 collection (lift analog).

Puts the bin in the left hand. Transfer and dump remainder stay for π0.
"""
from __future__ import annotations


GRASP_FORCE_N = 12.0
QUERY_FORCE_N = 4.0
QUERY_DISPLACEMENT_M = 0.012


def scripted_establish_grasp(task) -> str:
    from envs.utils import ArmTag

    deskbin_pose = task.deskbin.get_pose().p
    grasp_arm = ArmTag("left" if deskbin_pose[0] < 0 else "right")
    place_arm = ArmTag("left")
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
        task.move(
            task.back_to_origin(grasp_arm),
            task.grasp_actor(task.deskbin, arm_tag=place_arm, pre_grasp_dis=0.08, contact_point_id=1),
        )
    else:
        task.move(
            task.grasp_actor(task.deskbin, arm_tag=place_arm, pre_grasp_dis=0.08, contact_point_id=1)
        )
    task._af_grasp_arm_tag = "left"
    contact_positions = task.get_gripper_actor_contact_position("063_tabletrashbin")
    if not contact_positions or not task.is_left_gripper_close():
        raise RuntimeError("lift-clone established grasp did not keep left-hand contact")
    return "left"
