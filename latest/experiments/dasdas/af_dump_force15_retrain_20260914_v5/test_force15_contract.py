from __future__ import annotations

import unittest

import numpy as np

from force15_utility import expected_utility, realized_squeeze, select_force


CALIBRATION = {
    "fit_split": "TRAIN",
    "feasibility_input": False,
    "command_knots_N": [0.5, 5.0, 8.0, 10.0, 12.0, 15.0],
    "realized_squeeze_knots_N": [0.5, 5.2, 8.0, 8.7, 8.7, 8.8],
}


class Force15UtilityTests(unittest.TestCase):
    def test_realization_is_interpolated_and_saturating(self):
        value = realized_squeeze(np.asarray([8.0, 9.0, 12.0, 15.0]), CALIBRATION)
        np.testing.assert_allclose(value, [8.0, 8.35, 8.7, 8.8])

    def test_probability_can_outweigh_realized_force_cost(self):
        grid = np.asarray([0.5, 5.0, 8.0, 10.0, 12.0, 15.0])
        probability = np.asarray([0.05, 0.35, 0.15, 0.55, 0.75, 0.70])
        result = select_force(grid, probability, [0.5, 15.0], CALIBRATION)
        self.assertEqual(result["selected_force_N"], 12.0)

    def test_tie_break_prefers_lower_command(self):
        grid = np.asarray([0.5, 5.0, 8.0, 10.0, 12.0, 15.0])
        probability = np.asarray([0.0, 0.0, 0.0, 0.0, 0.75, 0.75])
        result = select_force(grid, probability, [0.5, 15.0], CALIBRATION)
        self.assertEqual(result["selected_force_N"], 12.0)

    def test_utility_rejects_post_support_force(self):
        with self.assertRaises(ValueError):
            expected_utility(np.asarray([0.5]), np.asarray([15.1]), CALIBRATION)


if __name__ == "__main__":
    unittest.main()
