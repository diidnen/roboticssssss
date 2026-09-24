"""Lift/Tabero squeeze inner loop on dump remainder.

F is a force REFERENCE, not a per-finger actuator cap. Stock gripper effort
limits stay at the robot default. Exact original ForcePositionAction inner
gains: EMA 0.2, ff 0.9 above 1 N, kp 0.0008, deadzone 0.25 N.

Official dump / 18/19 / Fig.B are not modified.
"""
from __future__ import annotations

import numpy as np

from original_squeeze_inner import OriginalSqueezeInner

SETTLE_STEPS = 150
REALIZE_TOL = 0.35
HOLD_WINDOW = 25
ARM = "left"


def capture_stock_drives(task, arm: str = ARM):
    joints = getattr(task.robot, f"{arm}_gripper")
    return [(joint, joint.stiffness, joint.damping, joint.force_limit, joint.drive_mode) for joint, _, _ in joints]


def restore_stock_drives(stock) -> None:
    for joint, stiffness, damping, limit, mode in stock:
        joint.set_drive_properties(stiffness, damping, limit, mode)


def _bilateral_squeeze(task) -> float:
    contact = task.get_actor_gripper_contact_forces(task.deskbin, ARM)
    return 2.0 * float(contact["single_finger_normal_force_n"])


def _actual_aperture_m(task) -> float:
    joints = getattr(task.robot, f"{ARM}_gripper")
    entity = getattr(task.robot, f"{ARM}_entity")
    active = list(entity.get_active_joints())
    positions = [abs(float(entity.qpos[active.index(joint)])) for joint, _, _ in joints]
    return float(np.clip(np.mean(positions) if positions else 0.0, 0.0, 0.04))


def _write_aperture(task, command_m: float) -> None:
    scale = getattr(task.robot, f"{ARM}_gripper_scale")
    span = float(scale[1] - scale[0])
    if abs(span) < 1e-9:
        raise RuntimeError("gripper_scale is degenerate")
    normalized = (float(command_m) - float(scale[0])) / span
    task.robot.set_gripper(float(np.clip(normalized, 0.0, 1.0)), ARM, gripper_eps=0.0)


class LiftSqueezeHold:
    """Original inner squeeze loop. Commanded dump F is single-finger; inner uses 2F bilateral."""

    def __init__(self, task, force_n: float, stock):
        self.task = task
        self.force_n = float(force_n)
        self.force_ref = 2.0 * self.force_n
        self.stock = stock
        self.inner = OriginalSqueezeInner()
        self.last = None

    def step(self) -> dict:
        restore_stock_drives(self.stock)
        measured = _bilateral_squeeze(self.task)
        outer = _actual_aperture_m(self.task)
        command = self.inner.step(measured, self.force_ref, outer)
        _write_aperture(self.task, command)
        contact = self.task.get_actor_gripper_contact_forces(self.task.deskbin, ARM)
        self.last = {
            "single_finger_n": float(contact["single_finger_normal_force_n"]),
            "bilateral_n": measured,
            "bilateral_contact": bool(contact["bilateral_contact"]),
            "outer_aperture_m": outer,
            "inner_aperture_m": command,
            "force_ref_bilateral_n": self.force_ref,
        }
        return self.last


def install_lift_squeeze(task, holder: LiftSqueezeHold) -> None:
    if not getattr(task, "_af_orig_scene_step", None):
        task._af_orig_scene_step = task.scene.step

        def stepped():
            if getattr(task, "_af_lift_inner_active", False) and getattr(task, "_af_lift_holder", None) is not None:
                task._af_lift_holder.step()
            return task._af_orig_scene_step()

        task.scene.step = stepped
    task._af_lift_holder = holder
    task._af_lift_inner_active = True


def uninstall_lift_squeeze(task) -> None:
    task._af_lift_inner_active = False
    task._af_lift_holder = None


def realize_commanded_force(task, force_n: float, stock, steps: int = SETTLE_STEPS) -> dict:
    holder = LiftSqueezeHold(task, force_n, stock)
    install_lift_squeeze(task, holder)
    task.af_force_limit_n = float(force_n)
    rows = []
    for _ in range(int(steps)):
        task.scene.step()
        rows.append(dict(holder.last or {}))
    window = rows[-HOLD_WINDOW:]
    singles = [float(row.get("single_finger_n") or 0.0) for row in window]
    mean = float(np.mean(singles))
    frac = float(np.mean([bool(row.get("bilateral_contact")) for row in window]))
    last = rows[-1]
    rel = abs(mean - float(force_n)) / max(float(force_n), 1e-6)
    return {
        "commanded_force_N": float(force_n),
        "controller": "lift_original_squeeze_inner",
        "pre_motion_squeeze_n": float(last.get("single_finger_n") or 0.0),
        "pre_motion_squeeze_mean_n": mean,
        "pre_motion_bilateral": bool(last.get("bilateral_contact")),
        "pre_motion_bilateral_frac": frac,
        "pre_motion_rel_error": rel,
        "force_realized_before_motion": bool(frac >= 0.8 and rel <= REALIZE_TOL),
        "settle_steps": int(steps),
        "pre_motion_inner_aperture_m": last.get("inner_aperture_m"),
        "force_ref_bilateral_n": 2.0 * float(force_n),
    }
