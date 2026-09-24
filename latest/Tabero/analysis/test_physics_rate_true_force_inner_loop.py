"""Deterministic checks for the optional FORTE physics-rate force loop.

These tests deliberately avoid importing Isaac Lab.  They validate the
time-base conversion and the source-level isolation/lifecycle contract before
the same code is exercised in fresh GPU physics processes.
"""

from __future__ import annotations

import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
RUNNER = Path("/home/exouser/FORTE/root7703_probe_activeforcing_smoke_20260904.py")


def bounded_correction(
    error_n: float,
    *,
    dt_s: float,
    kp_m_per_n_s: float = 0.004,
    deadzone_n: float = 0.25,
    open_rate_mps: float = 0.0024,
    close_rate_mps: float = 0.0030,
) -> float:
    if abs(error_n) < deadzone_n:
        return 0.0
    raw = -kp_m_per_n_s * error_n * dt_s
    return min(open_rate_mps * dt_s, max(-close_rate_mps * dt_s, raw))


def adaptive_release(
    force_excess_n: float,
    raw_force_n: float,
    opening_backlog_m: float,
    *,
    dt_s: float = 1 / 60,
    target_n: float = 4.0,
    slope_n_per_m: float = 9400.0,
    alpha: float = 0.20,
    backlog_soft_start_m: float = 0.0012124786153435707,
    backlog_limit_m: float = 0.0020110527984797955,
    deadzone_n: float = 0.25,
    guard_width_n: float = 0.75,
    open_rate_mps: float = 0.0030,
) -> float:
    if force_excess_n < deadzone_n:
        return 0.0
    backlog_scale = min(
        1.0,
        max(
            0.0,
            (backlog_limit_m - opening_backlog_m)
            / (backlog_limit_m - backlog_soft_start_m),
        ),
    )
    contact_scale = min(
        1.0,
        max(0.0, (raw_force_n - target_n - deadzone_n) / guard_width_n),
    )
    proposed = alpha * force_excess_n / slope_n_per_m
    return min(proposed * backlog_scale * contact_scale, open_rate_mps * dt_s)


def one_step_prediction(
    *,
    d_actual: float,
    d_command: float,
    candidate_release: float,
    force_n: float,
    beta_follow: float = 0.0023732437353127787,
    slope_n_per_m: float = 9400.0,
) -> tuple[float, float, float]:
    baseline_actual = d_actual + beta_follow * (d_command - d_actual)
    candidate_actual = d_actual + beta_follow * (
        d_command + candidate_release - d_actual
    )
    candidate_force = force_n - slope_n_per_m * (
        candidate_actual - d_actual
    )
    return baseline_actual, candidate_actual, candidate_force


def friction_utilization(tangential_n: float, normal_n: float, mu_safe: float) -> float:
    return tangential_n / max(mu_safe * normal_n, 1.0e-6)


def velocity_resolved_command(
    *,
    force_meas_n: float,
    force_des_n: float,
    actual_aperture_m: float,
    actual_velocity_mps: float = 0.0,
    dt_s: float = 1 / 60,
    slope_n_per_m: float = 9400.0,
    settling_time_s: float = 0.20,
    deadzone_n: float = 0.25,
    open_rate_mps: float = 0.0024,
    close_rate_mps: float = 0.0030,
    safety_scale: float = 1.0,
) -> tuple[float, float, float]:
    error = force_meas_n - force_des_n
    requested = 0.0 if abs(error) < deadzone_n else error / (
        slope_n_per_m * settling_time_s
    )
    nominal = min(open_rate_mps, max(-close_rate_mps, requested))
    if nominal * actual_velocity_mps > 0.0 and abs(actual_velocity_mps) > abs(nominal):
        nominal = 0.0
    safe = nominal * safety_scale if nominal > 0.0 else nominal
    return nominal, safe, actual_aperture_m + safe * dt_s


def offset_servo_command(
    *,
    force_meas_n: float,
    force_des_n: float,
    actual_aperture_m: float,
    eq_offset_m: float,
    dt_s: float = 1 / 60,
    slope_n_per_m: float = 9400.0,
    settling_time_s: float = 0.20,
    deadzone_n: float = 0.25,
    release_rate_mps: float = 0.0024,
    close_rate_mps: float = 0.0030,
    safety_scale: float = 1.0,
    eq_min_m: float = -0.0057,
    eq_max_m: float = -0.0019,
) -> tuple[float, float, float]:
    error = force_meas_n - force_des_n
    velocity = 0.0 if abs(error) < deadzone_n else error / (
        slope_n_per_m * settling_time_s
    )
    velocity = min(release_rate_mps, max(-close_rate_mps, velocity))
    safe_velocity = velocity * safety_scale if velocity > 0.0 else velocity
    next_offset = min(eq_max_m, max(eq_min_m, eq_offset_m + safe_velocity * dt_s))
    return safe_velocity, next_offset, actual_aperture_m + next_offset


class PhysicsRateTrueForceInnerLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")

    def test_correction_direction(self) -> None:
        # error = desired - measured; positive aperture motion opens.
        self.assertGreater(bounded_correction(3.0 - 6.0, dt_s=1 / 60), 0.0)
        self.assertLess(bounded_correction(3.0 - 1.0, dt_s=1 / 60), 0.0)

    def test_temporal_gain_preserved_for_one_second(self) -> None:
        error_n = 0.3  # below rate limiting and above the 0.25 N deadzone
        old = 20 * (-0.0002 * error_n)
        new = 60 * bounded_correction(error_n, dt_s=1 / 60)
        self.assertAlmostEqual(old, new, places=12)
        self.assertAlmostEqual(abs(old / error_n), 0.004, places=12)

    def test_filter_time_constant_preserved(self) -> None:
        alpha_reference = 0.5
        reference_dt = 0.05
        physics_dt = 1 / 60
        alpha_physics = 1 - (1 - alpha_reference) ** (physics_dt / reference_dt)
        retained_after_one_second = (1 - alpha_physics) ** 60
        retained_reference = (1 - alpha_reference) ** 20
        self.assertAlmostEqual(retained_after_one_second, retained_reference, places=12)
        self.assertAlmostEqual(
            -reference_dt / math.log(1 - alpha_reference),
            0.07213475204444818,
            places=12,
        )

    def test_stateful_correction_accumulates(self) -> None:
        d_force_cmd = 0.020
        fixed_vla_d_pred = 0.010
        for _ in range(10):
            d_force_cmd += bounded_correction(3.0 - 6.0, dt_s=1 / 60)
        self.assertGreater(d_force_cmd, 0.020)
        self.assertNotAlmostEqual(d_force_cmd, fixed_vla_d_pred, places=8)

    def test_feature_flag_is_default_off_and_forte_explicitly_enables(self) -> None:
        self.assertIn("authoritative_true_force_inner_loop_enabled: bool = False", self.source)
        self.assertIn("term.enable_authoritative_true_force_inner_loop(", self.runner)
        self.assertIn("elif self.cfg.squeeze_kp != 0.0:", self.source)

    def test_episode_reset_clears_runtime_state(self) -> None:
        reset_body = self.source.split("def reset(", 1)[1].split(
            "def enable_authoritative_true_force_inner_loop", 1
        )[0]
        for required in (
            "self._authoritative_true_force_enabled = False",
            "self.cfg.authoritative_true_force_inner_loop_enabled = False",
            "self._authoritative_true_force_filter = None",
            "self._authoritative_true_force_d_cmd_initialized[ids] = False",
            "self._authoritative_true_force_reacquire_accum[ids] = 0.0",
        ):
            self.assertIn(required, reset_body)

    def test_handoff_initializes_from_applied_target_not_contact_displaced_position(self) -> None:
        enable_body = self.source.split(
            "def enable_authoritative_true_force_inner_loop", 1
        )[1].split("def disable_authoritative_true_force_inner_loop", 1)[0]
        command_initialization = enable_body.split(
            "self._authoritative_true_force_d_cmd_initialized[:] = True", 1
        )[0]
        self.assertIn(
            "self._authoritative_true_force_d_cmd[:] = self._last_d_cmd",
            command_initialization,
        )
        self.assertNotIn(
            "self._robot.data.joint_pos[:, self._gripper_joint_ids].mean",
            command_initialization,
        )

    def test_recovery_uses_previous_loss_state_and_current_aperture(self) -> None:
        was_lost_at = self.source.index(
            "was_lost = self._authoritative_true_force_loss_streak > 0"
        )
        streak_update_at = self.source.index(
            "self._authoritative_true_force_loss_streak = torch.where(",
            was_lost_at,
        )
        recovered_at = self.source.index("recovered = bilateral & was_lost")
        resync_at = self.source.index(
            "self._authoritative_true_force_d_cmd[recovered] = torch.minimum("
        )
        self.assertLess(was_lost_at, streak_update_at)
        self.assertLess(streak_update_at, recovered_at)
        self.assertLess(recovered_at, resync_at)

    def test_production_formula_is_dt_normalized_and_stateful(self) -> None:
        for pattern in (
            r"kp_step\s*=.*authoritative_true_force_kp_m_per_n_s\) \* dt",
            r"open_step\s*=.*authoritative_true_force_open_rate_mps\) \* dt",
            r"close_step\s*=.*authoritative_true_force_close_rate_mps\) \* dt",
            r"d_command_unclamped\s*=\s*d_previous_command \+ correction",
        ):
            self.assertRegex(self.source, re.compile(pattern))

    def test_raw_force_only_guards_opening(self) -> None:
        self.assertIn(
            "authoritative_true_force_raw_open_guard_enabled: bool = True",
            self.source,
        )
        self.assertIn("(normal_correction > 0.0)", self.source)
        self.assertIn(
            "normal_correction = torch.where(\n                raw_open_guard,",
            self.source,
        )

    def test_adaptive_release_default_isolated_from_native_path(self) -> None:
        self.assertIn(
            "authoritative_true_force_adaptive_release_enabled: bool = False",
            self.source,
        )
        self.assertIn("ROOT7703_ADAPTIVE_RELEASE_ENABLED", self.runner)

    def test_offset_corrected_opening_backlog(self) -> None:
        # Contact compliance leaves actual aperture 5.5 mm above target.  The
        # static offset is not pending opening; only unequal progress is.
        cmd0, actual0 = 0.0065, 0.0120
        cmd, actual = 0.0075, 0.0122
        direct = cmd - actual
        backlog = max((cmd - cmd0) - (actual - actual0), 0.0)
        self.assertLess(direct, 0.0)
        self.assertAlmostEqual(backlog, 0.0008, places=12)

    def test_adaptive_release_is_backlog_suppressed(self) -> None:
        unblocked = adaptive_release(3.0, 7.0, 0.0)
        half_blocked = adaptive_release(
            3.0, 7.0, (0.0012124786153435707 + 0.0020110527984797955) / 2.0
        )
        blocked = adaptive_release(3.0, 7.0, 0.0020110527984797955)
        self.assertGreater(unblocked, half_blocked)
        self.assertGreater(half_blocked, blocked)
        self.assertEqual(blocked, 0.0)

    def test_small_backlog_does_not_suppress_release(self) -> None:
        no_backlog = adaptive_release(3.0, 7.0, 0.0)
        p50_backlog = adaptive_release(3.0, 7.0, 0.0012124786153435707)
        self.assertAlmostEqual(no_backlog, p50_backlog, places=12)

    def test_adaptive_release_contact_guard_is_continuous(self) -> None:
        self.assertEqual(adaptive_release(1.0, 4.25, 0.0), 0.0)
        middle = adaptive_release(1.0, 4.625, 0.0)
        stable = adaptive_release(1.0, 5.0, 0.0)
        self.assertGreater(middle, 0.0)
        self.assertLess(middle, stable)

    def test_adaptive_release_respects_final_rate_ceiling(self) -> None:
        release = adaptive_release(20.0, 24.0, 0.0)
        self.assertLessEqual(release, 0.0030 / 60.0)

    def test_contact_recovery_restarts_release_reference(self) -> None:
        recovery = self.source.split("if bool(torch.any(recovered)):", 1)[1].split(
            "self._authoritative_true_force_recovery_streak", 1
        )[0]
        self.assertIn(
            "self._authoritative_true_force_release_active[recovered] = False",
            recovery,
        )
        self.assertIn(
            "self._authoritative_true_force_release_actual_reference[recovered]",
            recovery,
        )

    def test_contact_safety_supervisor_default_is_shadow_isolated(self) -> None:
        self.assertIn(
            "authoritative_true_force_contact_safety_supervisor_enabled: bool = False",
            self.source,
        )
        self.assertIn(
            "authoritative_true_force_contact_safety_shadow_only: bool = True",
            self.source,
        )
        self.assertIn(
            "if safety_supervisor_applied:",
            self.source,
        )

    def test_object_filtered_per_finger_decomposition_is_recorded(self) -> None:
        self.assertIn(
            "left_true_tangential = torch.linalg.vector_norm(",
            self.source,
        )
        self.assertIn('"left_true_local_fx"', self.source)
        self.assertIn('"right_true_local_fz"', self.source)

    def test_friction_utilization_uses_conservative_mu(self) -> None:
        self.assertAlmostEqual(friction_utilization(0.5, 2.0, 0.5), 0.5)
        self.assertGreater(friction_utilization(1.1, 2.0, 0.5), 1.0)

    def test_candidate_release_improves_predicted_force_vs_no_release(self) -> None:
        baseline, candidate, force = one_step_prediction(
            d_actual=0.012,
            d_command=0.007,
            candidate_release=0.00005,
            force_n=7.0,
        )
        self.assertGreater(candidate, baseline)
        baseline_force = 7.0 - 9400.0 * (baseline - 0.012)
        self.assertLess(force, baseline_force)

    def test_contact_preservation_reflex_is_bounded(self) -> None:
        dt = 1 / 60
        per_step = 0.0006 * dt
        cumulative = 0.0
        for _ in range(20):
            cumulative += min(per_step, max(0.00005 - cumulative, 0.0))
        self.assertLessEqual(per_step, 0.00001 + 1e-12)
        self.assertAlmostEqual(cumulative, 0.00005, places=12)

    def test_weak_side_prediction_can_trigger_critical_before_loss(self) -> None:
        weak_force = 2.37
        weak_derivative_nps = -7.64
        projected = weak_force + weak_derivative_nps * 0.05
        self.assertLess(projected, 4.0 / 2.0)
        self.assertGreater(weak_force, 0.15)

    def test_reflex_critical_threshold_matches_confirmed_warning_boundary(self) -> None:
        critical = self.source.split("critical_safety =", 1)[1].split(
            "reflex_enabled =", 1
        )[0]
        self.assertIn(
            "< 0.5 * self._authoritative_true_force_target",
            critical,
        )

    def test_velocity_resolved_feature_is_default_off(self) -> None:
        self.assertIn(
            "authoritative_true_force_velocity_resolved_enabled: bool = False",
            self.source,
        )
        self.assertIn("ROOT7703_VELOCITY_RESOLVED_ENABLED", self.runner)

    def test_velocity_gain_is_dimensionally_derived(self) -> None:
        expected = 1.0 / (9400.0 * 0.20)
        self.assertAlmostEqual(expected, 0.0005319148936170213, places=15)
        self.assertIn(
            "velocity_gain = 1.0 / max(\n                velocity_slope * velocity_settling_time",
            self.source,
        )

    def test_velocity_force_direction(self) -> None:
        opening, _, _ = velocity_resolved_command(
            force_meas_n=7.0, force_des_n=4.0, actual_aperture_m=0.010
        )
        closing, _, _ = velocity_resolved_command(
            force_meas_n=2.0, force_des_n=4.0, actual_aperture_m=0.010
        )
        self.assertGreater(opening, 0.0)
        self.assertLess(closing, 0.0)

    def test_velocity_position_target_has_only_one_step_backlog(self) -> None:
        nominal, safe, target = velocity_resolved_command(
            force_meas_n=20.0, force_des_n=4.0, actual_aperture_m=0.010
        )
        self.assertAlmostEqual(nominal, 0.0024, places=12)
        self.assertAlmostEqual(target - 0.010, safe / 60.0, places=12)
        self.assertLessEqual(target - 0.010, 0.0024 / 60.0 + 1e-12)
        self.assertIn(
            "d_command_unclamped = actual_now + correction",
            self.source,
        )

    def test_actual_velocity_overspeed_brakes_position_lead(self) -> None:
        nominal, safe, target = velocity_resolved_command(
            force_meas_n=7.0,
            force_des_n=4.0,
            actual_aperture_m=0.010,
            actual_velocity_mps=0.0020,
        )
        self.assertEqual(nominal, 0.0)
        self.assertEqual(safe, 0.0)
        self.assertEqual(target, 0.010)

    def test_velocity_safety_filter_only_scales_opening(self) -> None:
        opening, opening_safe, _ = velocity_resolved_command(
            force_meas_n=7.0,
            force_des_n=4.0,
            actual_aperture_m=0.010,
            safety_scale=0.25,
        )
        closing, closing_safe, _ = velocity_resolved_command(
            force_meas_n=2.0,
            force_des_n=4.0,
            actual_aperture_m=0.010,
            safety_scale=0.0,
        )
        self.assertAlmostEqual(opening_safe, 0.25 * opening, places=15)
        self.assertEqual(closing_safe, closing)

    def test_velocity_mode_neutralizes_rejected_actuator_predictor(self) -> None:
        self.assertIn(
            "if velocity_resolved_enabled or offset_servo_enabled:",
            self.source,
        )
        self.assertIn("predicted_actual_no_release = actual_now", self.source)
        self.assertIn("predicted_actual_candidate = actual_now", self.source)

    def test_offset_servo_feature_is_default_off(self) -> None:
        self.assertIn(
            "authoritative_true_force_offset_servo_enabled: bool = False",
            self.source,
        )
        self.assertIn("ROOT7703_OFFSET_SERVO_ENABLED", self.runner)

    def test_offset_servo_handoff_preserves_loaded_target(self) -> None:
        actual = 0.012146800756454468
        applied_target = 0.0065455143339931965
        eq = applied_target - actual
        _, next_eq, target = offset_servo_command(
            force_meas_n=4.0,
            force_des_n=4.0,
            actual_aperture_m=actual,
            eq_offset_m=eq,
            eq_min_m=eq,
        )
        self.assertAlmostEqual(next_eq, eq, places=15)
        self.assertAlmostEqual(target, applied_target, places=15)

    def test_offset_servo_force_direction_changes_preload(self) -> None:
        eq = -0.004
        _, released, _ = offset_servo_command(
            force_meas_n=7.0,
            force_des_n=4.0,
            actual_aperture_m=0.012,
            eq_offset_m=eq,
        )
        _, closed, _ = offset_servo_command(
            force_meas_n=2.0,
            force_des_n=4.0,
            actual_aperture_m=0.012,
            eq_offset_m=eq,
        )
        self.assertGreater(released, eq)
        self.assertLess(closed, eq)

    def test_offset_servo_is_bounded_and_safety_only_scales_release(self) -> None:
        release_v, release_eq, _ = offset_servo_command(
            force_meas_n=20.0,
            force_des_n=4.0,
            actual_aperture_m=0.012,
            eq_offset_m=-0.002,
            safety_scale=0.0,
        )
        close_v, close_eq, _ = offset_servo_command(
            force_meas_n=0.0,
            force_des_n=4.0,
            actual_aperture_m=0.012,
            eq_offset_m=-0.00569,
            safety_scale=0.0,
        )
        self.assertEqual(release_v, 0.0)
        self.assertEqual(release_eq, -0.002)
        self.assertLess(close_v, 0.0)
        self.assertEqual(close_eq, -0.0057)

    def test_offset_servo_uses_actual_plus_preserved_equilibrium(self) -> None:
        self.assertIn(
            "d_command_unclamped = actual_now + eq_offset_clamped",
            self.source,
        )
        self.assertIn(
            "eq_offset_unclamped = eq_offset_previous + correction",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
