"""Minimal Tabero + true physical target-object force hybrid.

This module is deliberately independent of Isaac Lab so that its control and
state-machine contract can be tested without starting a simulator.  The live
adapter is the small function :func:`apply_to_tabero_nominal_action`:

    frozen VLA action -> this adapter -> Tabero ForcePositionAction

The adapter preserves the six arm fields, replaces only the nominal gripper
field, and clears the legacy per-finger force slots.  The Tabero action term
must be configured with ``squeeze_kp=0`` and ``squeeze_ff_k_load_z=0`` when
this adapter is used.  Thus there is exactly one force-error-to-aperture
correction loop.

The force sample contract is intentionally explicit.  ``F_left_obj_normal``
and ``F_right_obj_normal`` are object-filtered, inward-normal magnitudes, not
the all-contact gripper force and not a controller target.  The bilateral
measurement is always::

    F_meas = 2 * min(F_left_obj_normal, F_right_obj_normal)

No force tracking or lift permission is granted before all grasp-gate checks
pass.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Literal, Sequence


class HybridState(str, Enum):
    VLA_APPROACH = "VLA_APPROACH"
    ACQUIRE_BILATERAL = "ACQUIRE_BILATERAL"
    VALIDATE_GRASP = "VALIDATE_GRASP"
    FORCE_TRACK = "FORCE_TRACK"
    LIFT_TRANSPORT = "LIFT_TRANSPORT"
    RELEASE = "RELEASE"
    GRASP_INVALID = "GRASP_INVALID"
    CONTACT_LOSS = "CONTACT_LOSS"


POST_GRASP_PHASES = frozenset(
    {
        "grasp_completed",
        "post_grasp",
        "pre_lift",
        "branch_hold_end",
    }
)


@dataclass(frozen=True)
class PostGraspHandoff:
    """A semantic VLA-to-force-executor handoff, never a fixed step number."""

    step: int
    condition: str
    definition: str
    bilateral_target_contact: bool
    F_left_obj: float | None = None
    F_right_obj: float | None = None
    force_asymmetry: float | None = None
    center_offset_m: float | None = None
    normal_opposition_cosine: float | None = None

    @property
    def valid(self) -> bool:
        return self.bilateral_target_contact

    @property
    def failure_class(self) -> str | None:
        return None if self.valid else "PRE_HANDOFF_VLA_GRASP_FAILURE"

    def as_dict(self) -> dict[str, Any]:
        return {
            "HANDOFF_DEFINITION": self.definition,
            "HANDOFF_STEP": self.step,
            "HANDOFF_CONDITION": self.condition,
            "BILATERAL_TARGET_CONTACT": self.bilateral_target_contact,
            "F_left_obj": self.F_left_obj,
            "F_right_obj": self.F_right_obj,
            "force_asymmetry": self.force_asymmetry,
            "center_offset_m": self.center_offset_m,
            "normal_opposition_cosine": self.normal_opposition_cosine,
        }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass"}
    return bool(value)


def find_post_grasp_handoff(rows: Sequence[dict[str, Any]]) -> PostGraspHandoff | None:
    """Find the first semantic post-grasp handoff in a VLA rollout.

    A row qualifies only when its phase is an explicit post-grasp/pre-lift
    semantic or it carries ``grasp_completed``/``post_grasp``.  Bilateral
    target-object contact is the minimum validity check.  Center offset and
    force asymmetry are recorded, never used to discard the episode.
    """

    definition = (
        "first VLA semantic grasp-completed/post-grasp/pre-lift/lift-start "
        "row with bilateral target-object contact"
    )
    first_semantic_without_contact: PostGraspHandoff | None = None
    for row in rows:
        phase = str(row.get("phase", "")).strip().lower().replace("-", "_")
        explicit = phase in POST_GRASP_PHASES or any(
            _as_bool(row.get(key, False))
            for key in ("grasp_completed", "post_grasp")
        )
        if not explicit:
            continue
        bilateral = _as_bool(
            row.get("bilateral_target_contact", row.get("bilateral_object_contact", False))
        )
        evidence = PostGraspHandoff(
            step=int(row.get("step", row.get("t", 0))),
            condition=phase or "explicit_grasp_transition",
            definition=definition,
            bilateral_target_contact=bilateral,
            F_left_obj=_optional_float(row.get("F_left_obj", row.get("F_left_obj_normal"))),
            F_right_obj=_optional_float(row.get("F_right_obj", row.get("F_right_obj_normal"))),
            force_asymmetry=_optional_float(row.get("force_asymmetry")),
            center_offset_m=_optional_float(row.get("center_offset_m")),
            normal_opposition_cosine=_optional_float(row.get("normal_opposition_cosine")),
        )
        if bilateral:
            return evidence
        if first_semantic_without_contact is None:
            first_semantic_without_contact = evidence
    return first_semantic_without_contact


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def classify_post_grasp_failure(
    *,
    handoff_valid: bool,
    force_tracks_target: bool,
    remaining_task_failed: bool,
    contact_lost_after_tracking: bool = False,
    vla_motion_disturbance: bool = False,
) -> str:
    """Assign the required failure class with handoff-first precedence."""

    if not handoff_valid:
        return "PRE_HANDOFF_VLA_GRASP_FAILURE"
    if contact_lost_after_tracking and vla_motion_disturbance:
        return "POST_HANDOFF_VLA_MOTION_CONTACT_FAILURE"
    if not force_tracks_target:
        return "LOW_LEVEL_FORCE_EXECUTION_FAILURE"
    if remaining_task_failed:
        return "POST_GRASP_FORCE_INSUFFICIENT"
    return "NONE"


def same_state_force_branches(
    *, snapshot_id: str, root_id: int | str, forces_n: Sequence[float]
) -> tuple[dict[str, Any], ...]:
    """Describe fair branches: same VLA snapshot, only ``F_des`` changes."""

    if len(forces_n) == 0:
        raise ValueError("at least one physical force branch is required")
    return tuple(
        {
            "snapshot_id": snapshot_id,
            "root_id": root_id,
            "F_des": float(force),
            "branch_only_change": "F_des",
            "arm_trajectory": "frozen_vla_continues_unchanged",
            "grasp_source": "frozen_vla_post_grasp_snapshot",
        }
        for force in forces_n
    )


@dataclass(frozen=True)
class ForceSample:
    """One object-specific bilateral force observation.

    The force values should be computed from the filtered contact sensor's
    left/right target-object rows.  ``center_offset_m`` is the norm of the
    object's offset from the finger midpoint in the chosen grasp plane.
    ``normal_opposition_cosine`` is the dot product of the two *outward*
    contact-normal directions; a valid opposing pair is close to -1.
    """

    left_obj_normal_n: float
    right_obj_normal_n: float
    left_target_contact: bool
    right_target_contact: bool
    center_offset_m: float
    normal_opposition_cosine: float
    source_sensor: str = "contact_grasp_<target_object>"

    def __post_init__(self) -> None:
        values = (
            self.left_obj_normal_n,
            self.right_obj_normal_n,
            self.center_offset_m,
            self.normal_opposition_cosine,
        )
        if not all(isfinite(float(v)) for v in values):
            raise ValueError("force sample contains non-finite values")
        if self.left_obj_normal_n < 0 or self.right_obj_normal_n < 0:
            raise ValueError("normal force magnitudes must be non-negative")

    @property
    def f_meas_n(self) -> float:
        return 2.0 * min(self.left_obj_normal_n, self.right_obj_normal_n)

    @property
    def f_mean_n(self) -> float:
        return 0.5 * (self.left_obj_normal_n + self.right_obj_normal_n)

    @property
    def f_sum_n(self) -> float:
        return self.left_obj_normal_n + self.right_obj_normal_n

    @property
    def force_asymmetry(self) -> float:
        return abs(self.left_obj_normal_n - self.right_obj_normal_n) / max(
            self.f_sum_n, 1e-9
        )

    @property
    def bilateral_contact(self) -> bool:
        return self.left_target_contact and self.right_target_contact

    @classmethod
    def from_local_force_vectors(
        cls,
        left_force_local: Sequence[float],
        right_force_local: Sequence[float],
        *,
        left_target_contact: bool,
        right_target_contact: bool,
        center_offset_m: float,
        normal_opposition_cosine: float,
        source_sensor: str = "contact_grasp_<target_object>",
        normal_axis_index: int = 2,
    ) -> "ForceSample":
        """Build a sample from object-filtered finger-local force vectors.

        The current Tabero convention uses the local z component for closing
        direction.  The absolute component is used because the two local
        frames have opposite signs for the environment reaction force.
        """

        left = abs(float(left_force_local[normal_axis_index]))
        right = abs(float(right_force_local[normal_axis_index]))
        return cls(
            left,
            right,
            left_target_contact=left_target_contact,
            right_target_contact=right_target_contact,
            center_offset_m=center_offset_m,
            normal_opposition_cosine=normal_opposition_cosine,
            source_sensor=source_sensor,
        )


@dataclass(frozen=True)
class GraspGateConfig:
    """Post-grasp diagnostics; bilateral contact is the only hard handoff gate."""

    center_offset_max_m: float = 0.005
    force_asymmetry_max: float = 0.20
    normal_opposition_cosine_min: float = 0.95
    contact_force_min_n: float = 0.15
    bilateral_acquire_steps: int = 3


@dataclass(frozen=True)
class HybridConfig:
    """Starting controller parameters requested for the prototype."""

    kp_m_per_n: float = 0.0002
    deadband_n: float = 0.25
    open_step_limit_m: float = 0.00006
    close_step_limit_m: float = 0.00015
    filter_alpha: float = 0.50
    aperture_min_m: float = 0.0
    aperture_max_m: float = 0.04
    grasp_gate: GraspGateConfig = field(default_factory=GraspGateConfig)
    # Feed-forward is an actuator bias, never a replacement physical target.
    feedforward_mode: Literal["disabled", "actuator_compensation"] = "disabled"
    feedforward_bias_m_per_n: float = 0.0
    contact_loss_reacquire: bool = True
    # Contact-loss recovery is deliberately short and reversible.  These are
    # environment/policy-step hysteresis counts, matching the cadence at
    # which ``step`` is called by the live runner.
    contact_loss_hysteresis_steps: int = 1
    contact_recovery_hysteresis_steps: int = 1
    # Safety bounds for the temporary reacquire nudge.  They are not normal
    # force targets and do not replace the object-filtered F_des/F_meas loop.
    reacquire_cumulative_close_limit_m: float = 0.00045
    reacquire_force_safety_limit_n: float = 8.0


@dataclass(frozen=True)
class GateResult:
    centering_valid: bool
    bilateral_contact_valid: bool
    force_balance_valid: bool
    normal_opposition_valid: bool
    valid_grasp: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "GRASP_CENTERING_VALID": self.centering_valid,
            "BILATERAL_CONTACT_VALID": self.bilateral_contact_valid,
            "FORCE_BALANCE_VALID": self.force_balance_valid,
            "CONTACT_NORMALS_OPPOSING": self.normal_opposition_valid,
            "VALID_GRASP": self.valid_grasp,
            "gate_reason": self.reason,
        }


@dataclass(frozen=True)
class HybridStep:
    state: HybridState
    F_des: float
    F_left_obj_normal: float
    F_right_obj_normal: float
    F_meas_raw: float
    F_meas: float
    F_mean: float
    F_sum: float
    force_asymmetry: float
    d_nominal: float
    force_correction: float
    feedforward_correction: float
    d_final: float
    force_error: float
    filter_initialized: bool
    allow_lift: bool
    force_loop_active: bool
    gate: GateResult
    double_force_control: bool = False
    reacquire_nudge: bool = False
    reacquire_close_applied_m: float = 0.0
    contact_loss_streak: int = 0
    contact_recovery_streak: int = 0
    reacquire_force_safety_clamped: bool = False

    def as_dict(self) -> dict[str, Any]:
        out = {
            "execution_state": self.state.value,
            "F_des": self.F_des,
            "F_left_obj_normal": self.F_left_obj_normal,
            "F_right_obj_normal": self.F_right_obj_normal,
            "F_meas_raw": self.F_meas_raw,
            "F_meas": self.F_meas,
            "F_mean": self.F_mean,
            "F_sum": self.F_sum,
            "force_asymmetry": self.force_asymmetry,
            "d_nominal": self.d_nominal,
            "force_correction": self.force_correction,
            "feedforward_correction": self.feedforward_correction,
            "d_final": self.d_final,
            "force_error": self.force_error,
            "allow_lift": self.allow_lift,
            "force_loop_active": self.force_loop_active,
            "DOUBLE_FORCE_CONTROL": self.double_force_control,
            "reacquire_nudge": self.reacquire_nudge,
            "reacquire_close_applied_m": self.reacquire_close_applied_m,
            "contact_loss_streak": self.contact_loss_streak,
            "contact_recovery_streak": self.contact_recovery_streak,
            "reacquire_force_safety_clamped": self.reacquire_force_safety_clamped,
        }
        out.update(self.gate.as_dict())
        return out


def evaluate_grasp(sample: ForceSample, cfg: GraspGateConfig) -> GateResult:
    """Evaluate the post-grasp handoff.

    Scope boundary: centering, asymmetry, and normal opposition are recorded
    context variables.  Only bilateral target-object contact is required to
    start force adaptation; an imperfect but force-dependent grasp remains a
    valid posterior context.
    """

    bilateral = (
        sample.left_target_contact
        and sample.right_target_contact
        and sample.left_obj_normal_n >= cfg.contact_force_min_n
        and sample.right_obj_normal_n >= cfg.contact_force_min_n
    )
    centering = sample.center_offset_m <= cfg.center_offset_max_m
    balance = sample.force_asymmetry <= cfg.force_asymmetry_max
    opposition = (
        sample.normal_opposition_cosine <= -cfg.normal_opposition_cosine_min
    )
    valid = bilateral
    if valid:
        reason = "post_grasp_bilateral_target_contact"
    elif not bilateral:
        reason = "missing_bilateral_target_object_contact"
    elif not centering:
        reason = "object_outside_centering_threshold"
    elif not opposition:
        reason = "contact_normals_not_opposing"
    else:
        reason = "bilateral_force_asymmetry_above_threshold"
    return GateResult(centering, bilateral, balance, opposition, valid, reason)


class TaberoTruePhysicalForceHybrid:
    """Stateful gripper-only physical-force executor.

    ``step`` is called once per policy/environment action step.  The arm
    action is never accepted by this controller and therefore cannot be
    modified here.  Use :func:`apply_to_tabero_nominal_action` to merge the
    returned aperture target into the original Tabero action.
    """

    def __init__(self, cfg: HybridConfig | None = None) -> None:
        self.cfg = cfg or HybridConfig()
        if not 0 < self.cfg.filter_alpha <= 1:
            raise ValueError("filter_alpha must be in (0, 1]")
        if self.cfg.contact_loss_hysteresis_steps < 1:
            raise ValueError("contact_loss_hysteresis_steps must be >= 1")
        if self.cfg.contact_recovery_hysteresis_steps < 1:
            raise ValueError("contact_recovery_hysteresis_steps must be >= 1")
        if self.cfg.reacquire_cumulative_close_limit_m < 0:
            raise ValueError("reacquire_cumulative_close_limit_m must be >= 0")
        if self.cfg.reacquire_force_safety_limit_n < 0:
            raise ValueError("reacquire_force_safety_limit_n must be >= 0")
        self.state = HybridState.VLA_APPROACH
        self._bilateral_streak = 0
        self._contact_loss_streak = 0
        self._contact_recovery_streak = 0
        self._reacquire_close_accum_m = 0.0
        self._filtered_force: float | None = None
        self._grasp_gate: GateResult | None = None

    @property
    def grasp_gate(self) -> GateResult | None:
        return self._grasp_gate

    def reset(self) -> None:
        self.state = HybridState.VLA_APPROACH
        self._bilateral_streak = 0
        self._contact_loss_streak = 0
        self._contact_recovery_streak = 0
        self._reacquire_close_accum_m = 0.0
        self._filtered_force = None
        self._grasp_gate = None

    def _filter(self, raw: float) -> tuple[float, bool]:
        if self._filtered_force is None:
            self._filtered_force = raw
            return raw, True
        alpha = self.cfg.filter_alpha
        self._filtered_force = alpha * raw + (1.0 - alpha) * self._filtered_force
        return self._filtered_force, True

    def step(
        self,
        *,
        F_des: float,
        d_nominal: float,
        sample: ForceSample,
        execution_phase: Literal["approach", "static", "lift", "transport", "release"] = "static",
    ) -> HybridStep:
        """Advance the state machine and return the gripper-only command."""

        if F_des < 0 or not isfinite(float(F_des)):
            raise ValueError("F_des must be a finite non-negative physical force")
        if not isfinite(float(d_nominal)):
            raise ValueError("d_nominal must be finite")

        raw = sample.f_meas_n
        filtered, initialized = self._filter(raw)
        gate = evaluate_grasp(sample, self.cfg.grasp_gate)
        self._grasp_gate = gate

        # A loss is entered only after the configured short hysteresis.  Once
        # in CONTACT_LOSS, bilateral contact must be present for the recovery
        # hysteresis before normal force tracking is re-armed.
        recovered_this_step = False
        if self.state in {HybridState.FORCE_TRACK, HybridState.LIFT_TRANSPORT}:
            if sample.bilateral_contact:
                self._contact_loss_streak = 0
            else:
                self._contact_loss_streak += 1
                if self._contact_loss_streak >= self.cfg.contact_loss_hysteresis_steps:
                    self.state = HybridState.CONTACT_LOSS
                    self._contact_recovery_streak = 0
        elif self.state == HybridState.CONTACT_LOSS:
            self._contact_loss_streak = 0
            if sample.bilateral_contact:
                self._contact_recovery_streak += 1
                if self._contact_recovery_streak >= self.cfg.contact_recovery_hysteresis_steps:
                    # The first recovered bilateral sample becomes the new
                    # physical measurement reference.  This prevents stale
                    # zero/unilateral measurements from causing an impulse on
                    # the first normal tracking step.
                    self._filtered_force = raw
                    filtered = raw
                    initialized = True
                    self._contact_recovery_streak = 0
                    self._reacquire_close_accum_m = 0.0
                    self.state = HybridState.FORCE_TRACK
                    recovered_this_step = True
            else:
                self._contact_recovery_streak = 0

        if execution_phase == "release":
            self.state = HybridState.RELEASE
            self._bilateral_streak = 0
        elif self.state == HybridState.VLA_APPROACH:
            if sample.bilateral_contact:
                self.state = HybridState.ACQUIRE_BILATERAL
                self._bilateral_streak = 1
            else:
                self.state = HybridState.VLA_APPROACH
        elif self.state == HybridState.GRASP_INVALID:
            # Recover only through a fresh bilateral post-grasp handoff.  The
            # geometry diagnostics are intentionally not part of this gate.
            if sample.bilateral_contact:
                self.state = HybridState.ACQUIRE_BILATERAL
                self._bilateral_streak = 1
        elif self.state == HybridState.ACQUIRE_BILATERAL:
            if sample.bilateral_contact:
                self._bilateral_streak += 1
                if self._bilateral_streak >= self.cfg.grasp_gate.bilateral_acquire_steps:
                    self.state = HybridState.VALIDATE_GRASP
            else:
                self._bilateral_streak = 0
        if self.state == HybridState.VALIDATE_GRASP:
            # Scope boundary: bilateral target-object contact is the minimum
            # post-grasp handoff condition. Centering, normal opposition, and
            # asymmetry remain diagnostics for explaining required force.
            if gate.bilateral_contact_valid:
                self.state = HybridState.FORCE_TRACK
            else:
                self.state = HybridState.GRASP_INVALID
        # The recovery transition above intentionally happens before this
        # execution-phase promotion, so a live lift/transport caller reports
        # LIFT_TRANSPORT while the normal force loop is active again.
        if (
            self.state in {HybridState.FORCE_TRACK, HybridState.LIFT_TRANSPORT}
            and sample.bilateral_contact
            and execution_phase in {"lift", "transport"}
        ):
            self.state = HybridState.LIFT_TRANSPORT

        tracking = self.state in {HybridState.FORCE_TRACK, HybridState.LIFT_TRANSPORT}
        allow_lift = tracking and gate.bilateral_contact_valid and sample.bilateral_contact
        error = float(F_des) - filtered
        feedback = 0.0
        ff = 0.0
        reacquire_nudge = False
        reacquire_close_applied_m = 0.0
        reacquire_force_safety_clamped = False
        if tracking:
            if abs(error) >= self.cfg.deadband_n:
                # Tabero aperture convention: negative delta closes, positive
                # delta opens.  This preserves the validated low-force sign.
                feedback = -self.cfg.kp_m_per_n * error
            if self.cfg.feedforward_mode == "actuator_compensation":
                ff = -self.cfg.feedforward_bias_m_per_n * float(F_des)
        elif self.state == HybridState.CONTACT_LOSS and self.cfg.contact_loss_reacquire:
            # A bounded reacquisition nudge is not an F_meas=0 force error.
            # Stop closing if one finger is already carrying a large load, or
            # once the small cumulative reacquire budget is exhausted.
            reacquire_nudge = True
            unilateral_peak = max(sample.left_obj_normal_n, sample.right_obj_normal_n)
            if unilateral_peak >= self.cfg.reacquire_force_safety_limit_n:
                reacquire_force_safety_clamped = True
            else:
                remaining = max(
                    0.0,
                    self.cfg.reacquire_cumulative_close_limit_m
                    - self._reacquire_close_accum_m,
                )
                reacquire_close_applied_m = min(
                    self.cfg.close_step_limit_m,
                    remaining,
                )
                feedback = -reacquire_close_applied_m

        correction = feedback + ff
        if correction < 0:
            correction = max(correction, -self.cfg.close_step_limit_m)
        else:
            correction = min(correction, self.cfg.open_step_limit_m)
        d_final = min(
            self.cfg.aperture_max_m,
            max(self.cfg.aperture_min_m, float(d_nominal) + correction),
        )
        if self.state == HybridState.CONTACT_LOSS and correction < 0.0:
            self._reacquire_close_accum_m += -float(correction)
        elif tracking:
            # A normal tracking step supersedes any reacquire-only state.
            self._reacquire_close_accum_m = 0.0
        return HybridStep(
            state=self.state,
            F_des=float(F_des),
            F_left_obj_normal=sample.left_obj_normal_n,
            F_right_obj_normal=sample.right_obj_normal_n,
            F_meas_raw=raw,
            F_meas=filtered,
            F_mean=sample.f_mean_n,
            F_sum=sample.f_sum_n,
            force_asymmetry=sample.force_asymmetry,
            d_nominal=float(d_nominal),
            force_correction=correction,
            feedforward_correction=ff,
            d_final=d_final,
            force_error=error,
            filter_initialized=initialized,
            allow_lift=allow_lift,
            force_loop_active=tracking,
            gate=gate,
            reacquire_nudge=reacquire_nudge,
            reacquire_close_applied_m=float(-correction if reacquire_nudge and correction < 0.0 else 0.0),
            contact_loss_streak=self._contact_loss_streak,
            contact_recovery_streak=self._contact_recovery_streak,
            reacquire_force_safety_clamped=reacquire_force_safety_clamped,
        )


def apply_to_tabero_nominal_action(nominal_action: Any, d_final: float) -> Any:
    """Merge only aperture into a Tabero 13-D action.

    Slots 0:6 (EEF position/orientation) are preserved byte-for-byte for
    array/tensor actions.  Slot 6 is the final aperture target.  Slots 7:13
    are cleared so Tabero's old per-finger force targets cannot form a second
    force controller.  The caller must also disable ``ForcePositionAction``'s
    legacy squeeze feedback/feed-forward fields as documented below.
    """

    out = nominal_action.clone() if hasattr(nominal_action, "clone") else deepcopy(nominal_action)
    ndim = int(getattr(out, "ndim", 1))
    if ndim == 1:
        out[6] = d_final
        try:
            out[7:13] = 0.0
        except TypeError:
            out[7:13] = [0.0] * 6
    elif ndim == 2:
        out[:, 6] = d_final
        out[:, 7:13] = 0.0
    else:
        raise ValueError(f"expected a 1-D or 2-D Tabero action, got ndim={ndim}")
    return out


def tabero_single_force_loop_config() -> dict[str, Any]:
    """Return the required Option-B native Tabero configuration contract."""

    return {
        "squeeze_kp": 0.0,
        "squeeze_ff_k_load_z": 0.0,
        "target_contact_squeeze_enabled": False,
        "gripper_action": None,
        "DOUBLE_FORCE_CONTROL": False,
        "legacy_force_feedback": "disabled; nominal aperture only",
        "physical_feedback_source": "object-specific bilateral force_matrix_w",
    }


def make_object_force_sample(
    *,
    left_local_force: Sequence[float],
    right_local_force: Sequence[float],
    left_target_contact: bool,
    right_target_contact: bool,
    center_offset_m: float,
    normal_opposition_cosine: float,
    sensor_name: str,
) -> ForceSample:
    """Explicit construction point for the live ``contact_grasp_<object>`` bridge."""

    return ForceSample.from_local_force_vectors(
        left_local_force,
        right_local_force,
        left_target_contact=left_target_contact,
        right_target_contact=right_target_contact,
        center_offset_m=center_offset_m,
        normal_opposition_cosine=normal_opposition_cosine,
        source_sensor=sensor_name,
    )
