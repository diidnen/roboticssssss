"""Frozen-candidate feature and terminal-contract primitives.

This module contains no outcome-dependent thresholds and performs no task-set
admission.  The final admitted task order is supplied explicitly by a hashed
manifest after Fixed-5 qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ORIGINAL_TASK_ORDER = (
    "libero_object/task0", "libero_object/task1",
    "libero_object/task5", "libero_object/task6",
)
CANDIDATE_ORDER = (
    "libero_10/task5", "libero_goal/task9", "libero_spatial/task4",
)
BACKUP_CANDIDATE_ORDER = ("libero_goal/task0",)
MOTION_DIM = 6
CONDITION_DIM = 54


def expanded_shape(task_order: Iterable[str]) -> tuple[int, int, int]:
    order = tuple(task_order)
    if order[:4] != ORIGINAL_TASK_ORDER or len(set(order)) != len(order):
        raise ValueError("task order must preserve the four authoritative IDs and be unique")
    sequence_dim = MOTION_DIM + len(order)
    return 8, sequence_dim + CONDITION_DIM, sequence_dim


def encode_task(task_id: str, task_order: Iterable[str]) -> np.ndarray:
    order = tuple(task_order)
    code = np.zeros(len(order), dtype=np.float32)
    try:
        code[order.index(task_id)] = 1.0
    except ValueError as exc:
        raise ValueError(f"unregistered task: {task_id}") from exc
    return code


def expand_original_phasefree(x64: np.ndarray, task_id: str, task_order: Iterable[str]) -> np.ndarray:
    """Expand an authoritative 8x64 row without changing motion/condition values."""
    x64 = np.asarray(x64, dtype=np.float32)
    order = tuple(task_order)
    _, total, sequence = expanded_shape(order)
    if x64.shape != (8, 64) or task_id not in ORIGINAL_TASK_ORDER:
        raise ValueError("expected an original-task phase-free 8x64 row")
    expected_old = np.zeros(4, dtype=np.float32)
    expected_old[ORIGINAL_TASK_ORDER.index(task_id)] = 1.0
    if not np.array_equal(x64[:, 6:10], np.broadcast_to(expected_old, (8, 4))):
        raise ValueError("original task one-hot does not match declared task")
    out = np.zeros((8, total), dtype=np.float32)
    out[:, :6] = x64[:, :6]
    out[:, 6:sequence] = encode_task(task_id, order)
    out[:, sequence:] = x64[:, 10:]
    return out


def make_new_phasefree(motion6: np.ndarray, condition54: np.ndarray, task_id: str,
                       task_order: Iterable[str]) -> np.ndarray:
    motion6 = np.asarray(motion6, dtype=np.float32)
    condition54 = np.asarray(condition54, dtype=np.float32)
    order = tuple(task_order)
    _, total, sequence = expanded_shape(order)
    if motion6.shape != (8, 6) or condition54.shape != (8, 54):
        raise ValueError("expected motion (8,6) and condition (8,54)")
    out = np.empty((8, total), dtype=np.float32)
    out[:, :6] = motion6
    out[:, 6:sequence] = encode_task(task_id, order)
    out[:, sequence:] = condition54
    if not np.isfinite(out).all():
        raise ValueError("nonfinite feasibility input")
    return out


@dataclass(frozen=True)
class TerminalContract:
    task_id: str
    native_hold_steps: int = 20
    clearance_required: bool = False
    release_required: bool = True


TERMINAL_CONTRACTS = {
    "libero_10/task5": TerminalContract("libero_10/task5", release_required=True),
    "libero_goal/task9": TerminalContract("libero_goal/task9", release_required=True),
    "libero_spatial/task4": TerminalContract("libero_spatial/task4", clearance_required=True),
    # Backup pulling form. Admission additionally requires a handle-specific
    # friction/squeeze audit; terminal release is not part of the native
    # drawer-open goal.
    "libero_goal/task0": TerminalContract("libero_goal/task0", release_required=False),
}


def full_task_outcome(task_id: str, trace: list[dict], *, execution_error: str | None = None) -> dict:
    """Apply the predeclared candidate evaluator to simulator-native telemetry."""
    contract = TERMINAL_CONTRACTS[task_id]
    if execution_error or not trace:
        return {"label_valid": False, "full_task_success_y": None,
                "failure_class": "OTHER", "reason": "EXECUTION_INVALID"}
    dropped = any(bool(row.get("grasp_retention_failure")) for row in trace)
    reset = any(bool(row.get("unexplained_reset")) for row in trace)
    native_tail = trace[-contract.native_hold_steps:]
    native_stable = len(native_tail) == contract.native_hold_steps and all(bool(row.get("native_goal_reached")) for row in native_tail)
    clearance = (not contract.clearance_required) or any(bool(row.get("drawer_clearance_reached")) for row in trace)
    released = (not contract.release_required) or (
        len(native_tail) == contract.native_hold_steps
        and all(bool(row.get("release_intent")) and bool(row.get("unheld")) for row in native_tail)
    )
    success = native_stable and clearance and released and not dropped and not reset
    if success:
        failure = None
    elif dropped:
        failure = "GRASP_RETENTION_FAILURE"
    elif not clearance:
        failure = "TASK_CONTACT_OR_INTERACTION_FAILURE"
    elif not native_stable:
        failure = "TERMINAL_GEOMETRIC_FAILURE"
    elif not released:
        failure = "RELEASE_FAILURE"
    else:
        failure = "OTHER"
    return {
        "label_valid": True, "full_task_success_y": int(success),
        "native_terminal_stable": native_stable, "drawer_clearance_reached": clearance,
        "release_satisfied": released, "grasp_retention_failure": dropped,
        "failure_class": failure,
    }
