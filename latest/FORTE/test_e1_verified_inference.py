import unittest
import numpy as np
import e1_verified_inference as rt


class InferenceContractTests(unittest.TestCase):
    def test_low_force_tie(self):
        # Both candidates have utility exactly -1.
        self.assertEqual(rt.select([5,3],[0,0],5)['selected_setpoint'],3)

    def test_actual_utility(self):
        result=rt.select([3,4,5],[.4,.99,1],5)
        self.assertEqual(result['selected_setpoint'],4)
        self.assertAlmostEqual(result['utility'],.188)

    def test_nonfinite_rejected(self):
        for fs,ps in [([3],[np.nan]),([np.inf],[.9]),([3],[1.1]),([],[])]:
            with self.assertRaises(ValueError): rt.select(fs,ps,5)

    def test_shortened_probe_not_admitted(self):
        rows=[{'probe_phase':'hold'}]*190+[{'probe_phase':'probe_out'}]+[{'probe_phase':'probe_back'}]*15
        self.assertFalse(rt.validate_live_probe(rows,[.4,.5,.6])['admitted'])

    def test_nonpositive_member_not_admitted(self):
        rows=[{'probe_phase':'hold'}]*190+[{'probe_phase':'probe_out'}]*10+[{'probe_phase':'probe_back'}]*15
        self.assertFalse(rt.validate_live_probe(rows,[-.01,.5,.6])['admitted'])
        self.assertFalse(rt.validate_live_probe(rows,[np.nan,.5,.6])['admitted'])

    def test_checkpoint_name_cannot_silently_fallback(self):
        with self.assertRaises(ValueError):rt.probabilities(np.zeros((1,8,71)),0,'typo')


if __name__=='__main__': unittest.main()
