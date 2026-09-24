import unittest
from types import SimpleNamespace
import torch
from run_full_horizon_matched_pilot_20260905 import (
    defer_success_only, full_horizon_place_success, EXPECTED_PHASE_STEPS)


class Manager:
    active_terms=['success','object_1_dropped','custom_timeout']
    def __init__(self, success, dropped, timeout):
        self.values={k:torch.tensor(v) for k,v in zip(self.active_terms,[success,dropped,timeout])}
        self._terminated_buf=self.values['success'] | self.values['object_1_dropped']
        self._truncated_buf=self.values['custom_timeout'].clone()
    @property
    def terminated(self):return self._terminated_buf
    @property
    def dones(self):return self._terminated_buf | self._truncated_buf
    def get_term(self,name):return self.values[name]
    def get_term_cfg(self,name):return SimpleNamespace(time_out=name=='custom_timeout')


class FullHorizonTests(unittest.TestCase):
    def test_success_only_deferred_but_drop_timeout_preserved(self):
        manager=Manager([True,True,True,False],[False,True,False,False],[False,False,True,False])
        raw={k:v.clone() for k,v in manager.values.items()}
        self.assertEqual(defer_success_only(manager).tolist(),[False,True,True,False])
        for k in raw:self.assertTrue(torch.equal(raw[k],manager.values[k]))

    def test_intermediate_contact_cannot_count_as_full_place(self):
        self.assertEqual(full_horizon_place_success({'branch_hold':20,'lift':50,'transit':86},True,2.),0)

    def test_complete_phases_alone_not_sufficient(self):
        self.assertEqual(full_horizon_place_success(EXPECTED_PHASE_STEPS,False,2.),0)
        self.assertEqual(full_horizon_place_success(EXPECTED_PHASE_STEPS,True,.05),0)
        self.assertEqual(full_horizon_place_success(EXPECTED_PHASE_STEPS,True,.06),1)


if __name__=='__main__':unittest.main()
