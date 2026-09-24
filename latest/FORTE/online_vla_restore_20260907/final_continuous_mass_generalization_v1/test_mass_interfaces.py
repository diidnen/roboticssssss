import unittest
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

import mass_belief
from mass_feasibility import MassFeasibility
import mass_online_worker
from mass_online_variants import MassMethod


class ConstantModel(torch.nn.Module):
    def forward(self, step, condition):
        # A deterministic probability response to both force and mass columns.
        return 2.0 * condition[:, 0] + 3.0 * condition[:, 1]


class MassInterfaceTests(unittest.TestCase):
    def test_positive_mixture_quadrature_uses_sigma_and_positive_support(self):
        posterior = mass_belief.positive.PositivePosterior(
            (0.05, 0.10, 0.20), (np.log(0.01), np.log(0.02), np.log(0.03)), (1 / 3,) * 3
        )
        nodes, weights, qa = mass_belief.integration(posterior)
        self.assertTrue(np.all(nodes > 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=9)
        self.assertTrue(qa["sigma_used"])
        self.assertGreater(len(nodes), 3)

    def test_feasibility_consumes_phasefree_force_and_mass_columns(self):
        runtime = MassFeasibility.__new__(MassFeasibility)
        runtime.mean = np.zeros(64, np.float32)
        runtime.std = np.ones(64, np.float32)
        runtime.models = [ConstantModel(), ConstantModel(), ConstantModel()]
        runtime.device = torch.device("cpu")
        runtime.force_grid = np.asarray([3.0, 4.0, 5.0])
        posterior = {"interface": "CURRENT_MASS58_SIGMA_AWARE_POSITIVE_MIXTURE_V1",
                     "candidate_actions_executed": 0, "hidden_mass_used": False,
                     "integration_nodes": [0.08, 0.16], "integration_weights": [0.25, 0.75]}
        curve = runtime.curve(np.zeros((8, 64), np.float32), posterior)
        self.assertEqual(curve.shape, (3,))
        self.assertTrue(np.all(np.diff(curve) > 0))

    def test_af_rejects_hidden_mass_flag(self):
        runtime = MassFeasibility.__new__(MassFeasibility)
        runtime.mean = np.zeros(64, np.float32); runtime.std = np.ones(64, np.float32)
        runtime.models = [ConstantModel()] * 3; runtime.device = torch.device("cpu")
        runtime.force_grid = np.asarray([4.0])
        posterior = {"interface": "CURRENT_MASS58_SIGMA_AWARE_POSITIVE_MIXTURE_V1",
                     "candidate_actions_executed": 0, "hidden_mass_used": True,
                     "integration_nodes": [0.1], "integration_weights": [1.0]}
        with self.assertRaisesRegex(ValueError, "hidden mass"):
            runtime.curve(np.zeros((8, 64), np.float32), posterior)

    def test_online_motion_descriptor_exact_current_parity(self):
        source = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT")
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location("authoritative_continuous_worker", source / "worker.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        raw = [{"left_fx": "0.1", "left_fy": "0.2", "left_fz": "1.1",
                "right_fx": "0.3", "right_fy": "0.4", "right_fz": "1.2",
                "gripper_opening": "0.02", "object_id": "alphabet_soup_1"}]
        joints = np.asarray([0.0] * 7 + [0.02, -0.02], np.float32)
        saved = {"state": {"articulation": {"robot": {"joint_position": joints}},
                           "rigid_object": {"alphabet_soup_1": {"root_velocity": np.asarray([.01, .02, .03, 0, 0, 0])}}}}
        chunk = np.arange(50 * 13, dtype=np.float32).reshape(50, 13) / 1000
        expected = module.make_sequence(raw, saved, 0, chunk)
        actual = mass_online_worker.make_sequence(raw, saved, 0, chunk)
        np.testing.assert_array_equal(actual, expected)

    def test_af_planner_constructor_rejects_true_mass(self):
        with self.assertRaisesRegex(ValueError, "must not enter"):
            MassMethod("ACTIVEFORCING_MASS", 0.1)
        with self.assertRaisesRegex(ValueError, "must not enter"):
            MassMethod("FIXED_4", 0.1)


if __name__ == "__main__":
    unittest.main()
