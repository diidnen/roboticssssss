import unittest

import numpy as np

import activeforcing_e7_continuous_repair as e7


class CandidateGeneratorTests(unittest.TestCase):
    def test_uniform_is_random_not_linspace_and_reproducible(self):
        a = e7.generate_candidates("UNIFORM_CONTINUOUS", 3.4, 5.0, 10, 123)
        b = e7.generate_candidates("UNIFORM_CONTINUOUS", 3.4, 5.0, 10, 123)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.allclose(a, np.linspace(3.4, 5.0, 10)))

    def test_stratified_has_one_draw_per_stratum(self):
        lo, hi, k = 3.4, 5.0, 10
        x = e7.generate_candidates("STRATIFIED_CONTINUOUS", lo, hi, k, 456)
        edges = np.linspace(lo, hi, k + 1)
        self.assertTrue(np.all(x >= edges[:-1]))
        self.assertTrue(np.all(x <= edges[1:]))

    def test_fixed_grid_uses_frozen_taskwise_support(self):
        x = e7.generate_candidates("FIXED_GRID", 4.8, 6.0, 10, 1, task=1)
        self.assertTrue(set(np.unique(x)).issubset(set(e7.FIXED_GRIDS[1])))
        self.assertIn(6.0, x)

    def test_expected_utility_tie_breaks_to_lower_force(self):
        model = e7.MonotoneDirect({0: np.array([0.0, 0.05, 0.0])}, 0.0, False, 1.0)
        force, _, _ = e7.select_candidates(model, 0, np.array([3.0, 3.0]), np.array([0.5]))
        self.assertEqual(force, 3.0)

    def test_proposal_scores_only_candidates_inside_same_budget(self):
        class CountingModel:
            def __init__(self):
                self.batch_sizes = []

            def probability(self, task, force, friction):
                self.batch_sizes.append(len(np.asarray(force)))
                return np.full(len(np.asarray(force)), 0.5)

        model = CountingModel()
        x = e7.generate_candidates("PROPOSAL_GUIDED", 3.4, 5.0, 10, 789, model, 0, 0.45)
        self.assertEqual(len(x), 10)
        self.assertEqual(model.batch_sizes, [5])


if __name__ == "__main__":
    unittest.main()
