"""Isaac Lab bridge for the minimal Tabero true-force hybrid.

The controller itself lives in ``analysis.tabero_true_physical_force_hybrid``
so it remains unit-testable without Isaac Sim.  This file is the deliberately
small runtime bridge: it reads the object-filtered contact sensor, preserves
Tabero's nominal arm action, and disables the old squeeze correction path.

The expected sensor configuration is::

    ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/<target_object>"],
    )

The force matrix is therefore ``(N, 2, filtered_target_bodies, 3)`` and does
not include table/basket contacts.
"""

from __future__ import annotations

from typing import Any

import torch

from .terminations import object_filtered_gripper_force_local


def validate_object_specific_sensor(
    sensor: Any,
    *,
    expected_target_prim_path: str | None = None,
    expected_body_names: tuple[str, str] = (
        "panda_leftfinger",
        "panda_rightfinger",
    ),
) -> dict[str, Any]:
    """Fail closed unless the sensor is the intended bilateral object sensor."""

    cfg = getattr(sensor, "cfg", None)
    filters = tuple(getattr(cfg, "filter_prim_paths_expr", ()) or ())
    if expected_target_prim_path is not None and not any(
        expected_target_prim_path in str(path) for path in filters
    ):
        raise RuntimeError(
            "object-specific sensor filter does not contain the expected "
            f"target prim {expected_target_prim_path!r}: {filters!r}"
        )
    body_names = tuple(getattr(sensor, "body_names", ()) or ())
    if body_names and body_names != expected_body_names:
        raise RuntimeError(
            f"object-specific sensor body order mismatch: expected "
            f"{expected_body_names!r}, got {body_names!r}"
        )
    return {
        "target_filter_prim_paths_expr": filters,
        "body_names": body_names or expected_body_names,
        "left_body": expected_body_names[0],
        "right_body": expected_body_names[1],
        "force_matrix_required": "(N,2,M,3)",
    }


def object_specific_bilateral_force_sample(
    env: Any,
    *,
    target_contact_sensor_name: str,
    center_offset_m: torch.Tensor,
    left_normal_w: torch.Tensor | None = None,
    right_normal_w: torch.Tensor | None = None,
    contact_force_min_n: float = 0.15,
    expected_target_prim_path: str | None = None,
) -> dict[str, torch.Tensor]:
    """Read object-specific bilateral force and gate inputs for all envs.

    ``center_offset_m`` must be computed from the target object's position and
    the finger midpoint in the grasp plane.  Passing it explicitly avoids
    silently treating a world-axis offset as a grasp-axis offset.

    If explicit geometric contact normals are unavailable, the normalized
    object-filtered reaction-force vectors are used as a conservative runtime
    proxy.  Their dot product must be close to -1 for opposing contacts.
    """

    sensor = env.scene[target_contact_sensor_name]
    validate_object_specific_sensor(
        sensor, expected_target_prim_path=expected_target_prim_path
    )
    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None or force_matrix_w.ndim != 4:
        shape = None if force_matrix_w is None else tuple(force_matrix_w.shape)
        raise RuntimeError(
            f"{target_contact_sensor_name!r} must expose force_matrix_w with "
            f"shape (N,2,M,3); got {shape}"
        )
    if force_matrix_w.shape[1] != 2 or force_matrix_w.shape[-1] != 3:
        raise RuntimeError(
            f"{target_contact_sensor_name!r} must expose (N,2,M,3), "
            f"got {tuple(force_matrix_w.shape)}"
        )

    # This is the same validated target-object path used by the existing
    # object_filtered_gripper_force_local() helper.  Never substitute the
    # all-contact contact_gripper/net_forces_w signal here.
    local = object_filtered_gripper_force_local(
        env,
        contact_sensor_name=target_contact_sensor_name,
        left_gripper_frame_name="left_gripper_frame",
        right_gripper_frame_name="right_gripper_frame",
    )
    left_local = local[:, 0, :]
    right_local = local[:, 1, :]
    left_normal = torch.abs(left_local[:, 2])
    right_normal = torch.abs(right_local[:, 2])

    force_lr_w = force_matrix_w.sum(dim=2)
    left_contact = torch.linalg.vector_norm(force_lr_w[:, 0, :], dim=-1) >= contact_force_min_n
    right_contact = torch.linalg.vector_norm(force_lr_w[:, 1, :], dim=-1) >= contact_force_min_n

    if left_normal_w is None or right_normal_w is None:
        left_direction = torch.nn.functional.normalize(force_lr_w[:, 0, :], dim=-1, eps=1e-9)
        right_direction = torch.nn.functional.normalize(force_lr_w[:, 1, :], dim=-1, eps=1e-9)
    else:
        left_direction = torch.nn.functional.normalize(left_normal_w, dim=-1, eps=1e-9)
        right_direction = torch.nn.functional.normalize(right_normal_w, dim=-1, eps=1e-9)
    opposition = (left_direction * right_direction).sum(dim=-1)

    return {
        "F_left_obj_normal": left_normal,
        "F_right_obj_normal": right_normal,
        "F_meas": 2.0 * torch.minimum(left_normal, right_normal),
        "F_mean": 0.5 * (left_normal + right_normal),
        "F_sum": left_normal + right_normal,
        "force_asymmetry": torch.abs(left_normal - right_normal).div(
            torch.clamp(left_normal + right_normal, min=1e-9)
        ),
        "left_target_contact": left_contact,
        "right_target_contact": right_contact,
        "normal_opposition_cosine": opposition,
        "center_offset_m": center_offset_m,
        "object_specific_sensor": torch.ones_like(left_normal, dtype=torch.bool),
    }


def disable_tabero_legacy_force_loop(action_term: Any) -> dict[str, Any]:
    """Apply the Option-B single-loop contract to a ForcePositionAction.

    The action term still owns the nominal gripper position target and the
    robot PD command.  Its old force-error correction and 1.9x target
    feed-forward are disabled; the external hybrid controller is the only
    physical-force correction source.
    """

    required = {
        "squeeze_kp": 0.0,
        "squeeze_ff_k_load_z": 0.0,
        "target_contact_squeeze_enabled": False,
    }
    previous: dict[str, Any] = {}
    for name, value in required.items():
        if not hasattr(action_term.cfg, name):
            raise AttributeError(f"Tabero action term is missing {name!r}")
        previous[name] = getattr(action_term.cfg, name)
        setattr(action_term.cfg, name, value)
    return {
        "previous": previous,
        "DOUBLE_FORCE_CONTROL": False,
        "legacy_force_feedback": "disabled",
        "legacy_feedforward": "disabled",
        "active_feedback_source": "object-specific bilateral F_meas",
    }


def restore_tabero_legacy_force_loop(action_term: Any, snapshot: dict[str, Any]) -> None:
    """Restore a term modified by :func:`disable_tabero_legacy_force_loop`."""

    for name, value in snapshot.get("previous", {}).items():
        setattr(action_term.cfg, name, value)


def hybrid_nominal_action(action: torch.Tensor, d_final: torch.Tensor | float) -> torch.Tensor:
    """Return a Tabero action with only gripper aperture physically corrected."""

    if action.ndim not in (1, 2) or action.shape[-1] != 13:
        raise ValueError(f"expected Tabero action shape (13,) or (N,13), got {tuple(action.shape)}")
    out = action.clone()
    out[..., 6] = d_final
    # Prevent policy force slots from reaching ForcePositionAction as a second
    # aperture controller.  Arm pose slots 0:6 are untouched.
    out[..., 7:13] = 0.0
    return out


__all__ = [
    "validate_object_specific_sensor",
    "object_specific_bilateral_force_sample",
    "disable_tabero_legacy_force_loop",
    "restore_tabero_legacy_force_loop",
    "hybrid_nominal_action",
]
