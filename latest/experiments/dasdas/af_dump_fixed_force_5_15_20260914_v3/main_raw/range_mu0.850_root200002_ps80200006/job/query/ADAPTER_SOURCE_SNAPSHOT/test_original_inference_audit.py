"""Synthetic validator tests only; never counted as experimental results."""
from copy import deepcopy
import unittest
import numpy as np
from audit_original_inference_results import check_decision


def fixture():
    grid = np.round(np.arange(.5, 8.0001, .05), 8)
    probability = np.zeros_like(grid)
    # A nonconstant curve exercises the original utility calculation.
    probability[10] = 1. / (2. - grid[10]/5.)
    probability[30] = 1. / (2. - grid[30]/5.)
    utility = probability*(5.-grid)/5. - (1.-probability)
    chosen = int(np.argmax(utility))
    return {'force_grid_N': grid.tolist(), 'p_success': probability.tolist(),
            'expected_utility': utility.tolist(), 'selected_force_N': float(grid[chosen]),
            'utility': float(utility[chosen]), 'predicted_success': float(probability[chosen]),
            'utility_normalization_N': 5., 'posterior_sigma_used': True}


class DecisionAuditTests(unittest.TestCase):
    def test_accept_exact_rule(self):
        check_decision(fixture())

    def test_exact_tie_uses_first_force(self):
        value = fixture(); n = len(value['force_grid_N'])
        value.update(p_success=[0.]*n, expected_utility=[-1.]*n,
                     selected_force_N=.5, predicted_success=0., utility=-1.)
        check_decision(value)
        value['selected_force_N'] = .55
        with self.assertRaises(ValueError): check_decision(value)

    def test_reject_changed_choice(self):
        value = fixture(); value['selected_force_N'] = 8.
        with self.assertRaises(ValueError): check_decision(value)

    def test_reject_renormalized_utility(self):
        value = fixture(); grid = np.asarray(value['force_grid_N']); p = np.asarray(value['p_success'])
        value['expected_utility'] = (p*(8.-grid)/8. - (1.-p)).tolist()
        with self.assertRaises(AssertionError): check_decision(value)

    def test_reject_missing_sigma(self):
        value = fixture(); value['posterior_sigma_used'] = False
        with self.assertRaises(ValueError): check_decision(value)

    def test_reject_invalid_curve(self):
        for invalid in (float('nan'), -0.1, 1.1):
            value = deepcopy(fixture()); value['p_success'][0] = invalid
            with self.assertRaises(ValueError): check_decision(value)


if __name__ == '__main__':
    unittest.main()
