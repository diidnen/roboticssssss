from __future__ import annotations

import unittest

from analysis.tabero_true_physical_force_hybrid import (
    ForceSample,
    GraspGateConfig,
    HybridConfig,
    HybridState,
    TaberoTruePhysicalForceHybrid,
    apply_to_tabero_nominal_action,
    classify_post_grasp_failure,
    evaluate_grasp,
    find_post_grasp_handoff,
    same_state_force_branches,
    tabero_single_force_loop_config,
)


def valid_sample(left: float = 2.0, right: float = 2.0) -> ForceSample:
    return ForceSample(
        left,
        right,
        left_target_contact=True,
        right_target_contact=True,
        center_offset_m=0.003,
        normal_opposition_cosine=-0.99,
        source_sensor="contact_grasp_black_book_1",
    )


class TruePhysicalForceHybridTests(unittest.TestCase):
    def test_handoff_uses_semantic_phase_and_not_centering_gate(self) -> None:
        rows = [
            {"step": 10, "phase": "approach", "bilateral_object_contact": 0},
            {
                "step": 25,
                "phase": "pre_lift",
                "bilateral_object_contact": 1,
                "F_left_obj": 1.0,
                "F_right_obj": 0.5,
                "force_asymmetry": 0.333,
                "center_offset_m": 0.019,
                "normal_opposition_cosine": -0.99,
            },
        ]
        handoff = find_post_grasp_handoff(rows)
        self.assertIsNotNone(handoff)
        assert handoff is not None
        self.assertEqual(handoff.step, 25)
        self.assertEqual(handoff.condition, "pre_lift")
        self.assertTrue(handoff.valid)
        self.assertIsNone(handoff.failure_class)
        self.assertAlmostEqual(handoff.center_offset_m, 0.019)

        later_contact = find_post_grasp_handoff(
            [
                {"step": 25, "phase": "pre_lift", "bilateral_object_contact": 0},
                {"step": 26, "phase": "pre_lift", "bilateral_object_contact": 1},
            ]
        )
        self.assertIsNotNone(later_contact)
        assert later_contact is not None
        self.assertEqual(later_contact.step, 26)
        self.assertTrue(later_contact.valid)

        no_grasp = find_post_grasp_handoff(
            [{"step": 25, "phase": "pre_lift", "bilateral_object_contact": 0}]
        )
        self.assertIsNotNone(no_grasp)
        assert no_grasp is not None
        self.assertEqual(no_grasp.failure_class, "PRE_HANDOFF_VLA_GRASP_FAILURE")

    def test_failure_taxonomy_is_handoff_first(self) -> None:
        self.assertEqual(
            classify_post_grasp_failure(
                handoff_valid=False,
                force_tracks_target=False,
                remaining_task_failed=True,
            ),
            "PRE_HANDOFF_VLA_GRASP_FAILURE",
        )
        self.assertEqual(
            classify_post_grasp_failure(
                handoff_valid=True,
                force_tracks_target=False,
                remaining_task_failed=True,
            ),
            "LOW_LEVEL_FORCE_EXECUTION_FAILURE",
        )
        self.assertEqual(
            classify_post_grasp_failure(
                handoff_valid=True,
                force_tracks_target=True,
                remaining_task_failed=True,
            ),
            "POST_GRASP_FORCE_INSUFFICIENT",
        )
        self.assertEqual(
            classify_post_grasp_failure(
                handoff_valid=True,
                force_tracks_target=True,
                remaining_task_failed=True,
                contact_lost_after_tracking=True,
                vla_motion_disturbance=True,
            ),
            "POST_HANDOFF_VLA_MOTION_CONTACT_FAILURE",
        )

    def test_same_state_branches_change_only_physical_force(self) -> None:
        branches = same_state_force_branches(
            snapshot_id="root7_handoff", root_id=7, forces_n=(2, 4, 6)
        )
        self.assertEqual([b["F_des"] for b in branches], [2.0, 4.0, 6.0])
        self.assertEqual({b["snapshot_id"] for b in branches}, {"root7_handoff"})
        self.assertEqual({b["branch_only_change"] for b in branches}, {"F_des"})
        self.assertTrue(all(b["arm_trajectory"] == "frozen_vla_continues_unchanged" for b in branches))

    def test_bilateral_measurement_and_saved_components(self) -> None:
        sample = valid_sample(left=1.8, right=2.2)
        self.assertAlmostEqual(sample.f_meas_n, 3.6)
        self.assertAlmostEqual(sample.f_mean_n, 2.0)
        self.assertAlmostEqual(sample.f_sum_n, 4.0)
        self.assertAlmostEqual(sample.force_asymmetry, 0.1)
        self.assertAlmostEqual(
            ForceSample.from_local_force_vectors(
                (0.0, 0.0, -1.8),
                (0.0, 0.0, 2.2),
                left_target_contact=True,
                right_target_contact=True,
                center_offset_m=0.003,
                normal_opposition_cosine=-0.99,
            ).f_meas_n,
            3.6,
        )

    def test_gate_reports_each_failure_independently(self) -> None:
        gate = GraspGateConfig()
        result = evaluate_grasp(valid_sample(), gate)
        self.assertTrue(result.valid_grasp)
        self.assertTrue(result.centering_valid)
        self.assertTrue(result.bilateral_contact_valid)
        self.assertTrue(result.force_balance_valid)
        self.assertTrue(result.normal_opposition_valid)

        bad = ForceSample(
            1.0,
            3.0,
            left_target_contact=True,
            right_target_contact=False,
            center_offset_m=0.020,
            normal_opposition_cosine=0.0,
        )
        result = evaluate_grasp(bad, gate)
        self.assertFalse(result.valid_grasp)
        self.assertFalse(result.centering_valid)
        self.assertFalse(result.bilateral_contact_valid)
        self.assertFalse(result.force_balance_valid)
        self.assertFalse(result.normal_opposition_valid)

    def test_missing_bilateral_handoff_never_allows_lift_or_force_tracking(self) -> None:
        ctl = TaberoTruePhysicalForceHybrid()
        bad = ForceSample(
            0.0,
            3.0,
            left_target_contact=True,
            right_target_contact=False,
            center_offset_m=0.020,
            normal_opposition_cosine=-0.99,
        )
        for _ in range(4):
            out = ctl.step(F_des=4.0, d_nominal=0.02, sample=bad, execution_phase="lift")
        self.assertIn(
            out.state,
            {HybridState.VLA_APPROACH, HybridState.GRASP_INVALID, HybridState.VALIDATE_GRASP},
        )
        self.assertFalse(out.allow_lift)
        self.assertFalse(out.force_loop_active)
        self.assertEqual(out.force_correction, 0.0)

    def test_imperfect_vla_grasp_is_tracked_and_geometry_is_diagnostic(self) -> None:
        ctl = TaberoTruePhysicalForceHybrid()
        imperfect = ForceSample(
            1.0,
            3.0,
            left_target_contact=True,
            right_target_contact=True,
            center_offset_m=0.020,
            normal_opposition_cosine=-0.99,
        )
        for _ in range(3):
            out = ctl.step(F_des=4.0, d_nominal=0.02, sample=imperfect, execution_phase="static")
        self.assertEqual(out.state, HybridState.FORCE_TRACK)
        self.assertTrue(out.allow_lift)
        self.assertFalse(out.gate.centering_valid)
        self.assertFalse(out.gate.force_balance_valid)

    def test_valid_gate_then_force_loop_preserves_sign_and_limits(self) -> None:
        ctl = TaberoTruePhysicalForceHybrid()
        for _ in range(3):
            out = ctl.step(F_des=4.0, d_nominal=0.02, sample=valid_sample(1.0, 1.0))
        self.assertEqual(out.state, HybridState.FORCE_TRACK)
        self.assertTrue(out.allow_lift)

        low = ctl.step(F_des=4.0, d_nominal=0.02, sample=valid_sample(1.0, 1.0))
        self.assertLess(low.force_correction, 0.0)  # low force -> close
        self.assertLessEqual(abs(low.force_correction), 0.00015)

        ctl.step(F_des=4.0, d_nominal=0.02, sample=valid_sample(3.0, 3.0))
        high = ctl.step(F_des=4.0, d_nominal=0.02, sample=valid_sample(3.0, 3.0))
        self.assertGreater(high.force_correction, 0.0)  # high force -> open
        self.assertLessEqual(high.force_correction, 0.00006)

    def test_contact_loss_is_not_zero_force_error(self) -> None:
        ctl = TaberoTruePhysicalForceHybrid()
        for _ in range(3):
            ctl.step(F_des=4.0, d_nominal=0.02, sample=valid_sample())
        lost = ForceSample(
            0.0,
            0.0,
            left_target_contact=False,
            right_target_contact=False,
            center_offset_m=0.003,
            normal_opposition_cosine=-0.99,
        )
        out = ctl.step(F_des=4.0, d_nominal=0.02, sample=lost, execution_phase="lift")
        self.assertEqual(out.state, HybridState.CONTACT_LOSS)
        self.assertFalse(out.allow_lift)
        self.assertFalse(out.force_loop_active)
        self.assertEqual(out.force_error, 4.0 - out.F_meas)
        self.assertEqual(out.force_correction, -0.00015)

    def test_contact_loss_recovers_and_reenables_force_feedback(self) -> None:
        ctl = TaberoTruePhysicalForceHybrid()
        ctl.state = HybridState.FORCE_TRACK
        ctl.step(F_des=4.0, d_nominal=0.02, sample=valid_sample(2.0, 2.0))

        lost = ForceSample(
            0.0,
            0.0,
            left_target_contact=False,
            right_target_contact=False,
            center_offset_m=0.003,
            normal_opposition_cosine=-0.99,
        )
        lost_out = ctl.step(F_des=4.0, d_nominal=0.02, sample=lost, execution_phase="lift")
        self.assertEqual(lost_out.state, HybridState.CONTACT_LOSS)
        self.assertTrue(lost_out.reacquire_nudge)
        self.assertFalse(lost_out.force_loop_active)

        # Bilateral recovery must be reversible.  A high recovered force is
        # intentional here: normal feedback should open, not keep closing.
        recovered = ctl.step(
            F_des=4.0,
            d_nominal=0.02,
            sample=valid_sample(3.0, 3.0),
            execution_phase="static",
        )
        self.assertEqual(recovered.state, HybridState.FORCE_TRACK)
        self.assertTrue(recovered.force_loop_active)
        self.assertFalse(recovered.reacquire_nudge)
        self.assertEqual(recovered.reacquire_close_applied_m, 0.0)
        self.assertTrue(recovered.filter_initialized)
        self.assertGreater(recovered.force_correction, 0.0)

    def test_reacquire_is_cumulative_bounded_and_has_unilateral_safety_guard(self) -> None:
        cfg = HybridConfig(
            reacquire_cumulative_close_limit_m=0.0003,
            reacquire_force_safety_limit_n=2.0,
        )
        ctl = TaberoTruePhysicalForceHybrid(cfg)
        ctl.state = HybridState.CONTACT_LOSS
        unilateral_high = ForceSample(
            2.1,
            0.0,
            left_target_contact=True,
            right_target_contact=False,
            center_offset_m=0.003,
            normal_opposition_cosine=-0.99,
        )
        safe = ctl.step(F_des=4.0, d_nominal=0.02, sample=unilateral_high)
        self.assertTrue(safe.reacquire_nudge)
        self.assertTrue(safe.reacquire_force_safety_clamped)
        self.assertEqual(safe.force_correction, 0.0)

        low = ForceSample(
            0.0,
            0.0,
            left_target_contact=False,
            right_target_contact=False,
            center_offset_m=0.003,
            normal_opposition_cosine=-0.99,
        )
        first = ctl.step(F_des=4.0, d_nominal=0.02, sample=low)
        second = ctl.step(F_des=4.0, d_nominal=0.02, sample=low)
        third = ctl.step(F_des=4.0, d_nominal=0.02, sample=low)
        self.assertEqual(first.force_correction, -0.00015)
        self.assertEqual(second.force_correction, -0.00015)
        self.assertEqual(third.force_correction, 0.0)

    def test_action_adapter_changes_only_gripper_and_clears_force_slots(self) -> None:
        nominal = [float(i) for i in range(13)]
        out = apply_to_tabero_nominal_action(nominal, 0.021)
        self.assertEqual(out[:6], nominal[:6])
        self.assertEqual(out[6], 0.021)
        self.assertEqual(out[7:13], [0.0] * 6)
        self.assertEqual(nominal[6:], [6.0] + [float(i) for i in range(7, 13)])

        import torch

        tensor_action = torch.arange(26, dtype=torch.float32).reshape(2, 13)
        tensor_out = apply_to_tabero_nominal_action(tensor_action, 0.019)
        self.assertTrue(torch.equal(tensor_out[:, :6], tensor_action[:, :6]))
        self.assertTrue(torch.all(tensor_out[:, 6] == 0.019))
        self.assertTrue(torch.all(tensor_out[:, 7:13] == 0.0))

    def test_legacy_feedforward_is_not_physical_target(self) -> None:
        cfg = HybridConfig(
            feedforward_mode="actuator_compensation",
            feedforward_bias_m_per_n=0.000001,
        )
        ctl = TaberoTruePhysicalForceHybrid(cfg)
        for _ in range(3):
            out = ctl.step(F_des=4.0, d_nominal=0.02, sample=valid_sample())
        self.assertAlmostEqual(out.force_error, 4.0 - out.F_meas)
        self.assertAlmostEqual(out.F_des, 4.0)
        self.assertNotEqual(out.feedforward_correction, 0.0)

    def test_single_loop_contract(self) -> None:
        cfg = tabero_single_force_loop_config()
        self.assertFalse(cfg["DOUBLE_FORCE_CONTROL"])
        self.assertEqual(cfg["squeeze_kp"], 0.0)
        self.assertEqual(cfg["squeeze_ff_k_load_z"], 0.0)
        self.assertIsNone(cfg["gripper_action"])


if __name__ == "__main__":
    unittest.main()
