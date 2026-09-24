import unittest
import numpy as np
from max_force_utility import expected_utility, select_force


class MaxForceUtilityTests(unittest.TestCase):
    def test_endpoint_semantics(self):
        self.assertEqual(float(expected_utility(1, 8, 8)), 0)
        self.assertEqual(float(expected_utility(1, 0, 8)), 1)
        np.testing.assert_array_equal(expected_utility([0, 0, 0], [0, 5, 8], 8), [-1, -1, -1])

    def test_old_maximum_equivalence(self):
        p = np.array([0.1, 0.5, 0.9]); f = np.array([3, 4, 5])
        np.testing.assert_allclose(expected_utility(p, f, 5), p*(5-f)/5-(1-p))

    def test_eight_newtons_can_win_and_ties_choose_low(self):
        result = select_force([0.5, 5, 8], [0, 0.44, 0.69], [0.5, 8])
        self.assertEqual(result['selected_force_N'], 8)
        self.assertEqual(select_force([0.5, 8], [0, 0], [0.5, 8])['selected_force_N'], 0.5)

    def test_invalid_inputs(self):
        for args in [(0.5, 8, 5), (1.1, 2, 8), (0.5, 1, 0)]:
            with self.assertRaises(ValueError): expected_utility(*args)


if __name__ == '__main__': unittest.main()
