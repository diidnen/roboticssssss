import unittest
from copy import deepcopy
import numpy as np
import audit_v4_inference
import infer_v4
import run_v4_inference
from max_force_utility import select_force


class OnlineBindingsTests(unittest.TestCase):
    def test_original_execution_and_queue_bindings(self):
        self.assertTrue(callable(infer_v4.bind()['qualify']))
        self.assertTrue(callable(run_v4_inference.bind()))

    def test_independent_audit_uses_maxf8(self):
        grid=np.round(np.arange(.5,8.0001,.05),8)
        probability=np.linspace(.05,.8,len(grid))
        decision=select_force(grid,probability,[.5,8]);decision['posterior_sigma_used']=True
        check=audit_v4_inference.bind()['check_decision']
        check(decision)
        wrong=deepcopy(decision);wrong['utility_normalization_N']=5.
        with self.assertRaises(ValueError):check(wrong)
        wrong=deepcopy(decision)
        wrong['expected_utility']=(probability*(5-grid)/5-(1-probability)).tolist()
        with self.assertRaises(AssertionError):check(wrong)

    def test_independent_audit_rejects_bad_selection(self):
        grid=np.round(np.arange(.5,8.0001,.05),8)
        decision=select_force(grid,np.zeros(len(grid)),[.5,8]);decision['posterior_sigma_used']=True
        check=audit_v4_inference.bind()['check_decision'];check(decision)
        decision['selected_force_N']=8.
        with self.assertRaises(ValueError):check(decision)


if __name__=='__main__':unittest.main()
