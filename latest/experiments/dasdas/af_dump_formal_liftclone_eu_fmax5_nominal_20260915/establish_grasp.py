"""Dump established grasp, lift analog.

Lift keeps the P4 grasp and never re-runs grasp motion. Dump maps that as:
if P4 (or live squeeze) still has bilateral contact, only raise the 12 N
force cap; do not call grasp_actor. The RoboTwin contact_position +
is_left_gripper_close heuristic misses P4 finger links and will re-grasp
into a live hold.
"""
from __future__ import annotations


GRASP_FORCE_N = 12.0
QUERY_FORCE_N = 4.0
QUERY_DISPLACEMENT_M = 0.012
HOLD_FORCE_N = 0.25


def _arm_tag(task) -> str:
    arm = getattr(task, "_af_grasp_arm_tag", None)
    if arm in ("left", "right"):
        return arm
    pose = task.deskbin.get_pose().p
    return "left" if pose[0] < 0 else "right"


def _gripper_val(task, arm: str) -> float:
    if arm == "left":
        return float(task.robot.get_left_gripper_val())
    return float(task.robot.get_right_gripper_val())


def live_hold(task, arm: str | None = None) -> dict:
    arm = arm or _arm_tag(task)
    contact = task.get_actor_gripper_contact_forces(task.deskbin, arm)
    force = float(contact.get("single_finger_normal_force_n") or 0.0)
    holding = bool(contact.get("bilateral_contact")) and force > HOLD_FORCE_N
    return {
        "arm": arm,
        "holding": holding,
        "bilateral_contact": bool(contact.get("bilateral_contact")),
        "single_finger_normal_force_n": force,
        "gripper_val": _gripper_val(task, arm),
    }


def p4_row_holding(p4_raw) -> bool:
    if not p4_raw:
        return False
    last = p4_raw[-1]
    return (
        last.get("contact_state") == "bilateral"
        and int(last.get("contact_left") or 0) == 1
        and int(last.get("contact_right") or 0) == 1
        and last.get("dropped") in (0, None, False)
    )


def scripted_establish_grasp(task, p4_raw=None) -> str:
    from envs.utils import ArmTag

    arm = _arm_tag(task)
    live = live_hold(task, arm)
    p4_held = p4_row_holding(p4_raw)
    if live["holding"] or p4_held:
        task._af_grasp_arm_tag = arm
        task._af_established_grasp = {
            "skipped_script": True,
            "reason": "p4_or_live_bilateral_hold_lift_analog",
            "p4_row_holding": p4_held,
            "live": live,
            "arm": arm,
            "grasp_force_N": GRASP_FORCE_N,
        }
        return arm

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
    after = live_hold(task, "left")
    if not after["holding"]:
        raise RuntimeError("lift-clone established grasp did not keep left-hand contact")
    task._af_established_grasp = {
        "skipped_script": False,
        "reason": "scripted_12N_grasp_from_open_state",
        "p4_row_holding": p4_held,
        "live": after,
        "arm": "left",
        "grasp_force_N": GRASP_FORCE_N,
    }
    return "left"
