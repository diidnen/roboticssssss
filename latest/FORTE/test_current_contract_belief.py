import copy
import csv
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from current_contract_belief_features import ProbeEvidence
from current_contract_physical_belief import FrictionMember,batch,verify_decision_prefix

BASE=Path('/home/exouser/FORTE/analysis/results/preaction_decision_contract_20260905/corrected_friction_stop/high')


class EvidenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (BASE/'RAW_PROBE.csv').open() as f:cls.raw=list(csv.DictReader(f))
        cls.rb=json.loads((BASE/'CONTACT_PATCH_READBACK.json').read_text())

    def test_runtime_saved_parity(self):
        e=ProbeEvidence();np.testing.assert_array_equal(e.load(BASE),e.rows(self.raw,self.rb))

    def test_labels_private_state_ids_do_not_enter(self):
        raw=copy.deepcopy(self.raw);rb=copy.deepcopy(self.rb)
        for r in raw:r.update(friction=999,seed_idx=-10,trial_id='changed',full_task_success_y=0)
        for r in rb:r.update(object_position=[999]*3,object_velocity=[-999]*3,material_properties=[999],root_id='bad')
        e=ProbeEvidence();np.testing.assert_array_equal(e.rows(raw,rb),e.rows(self.raw,self.rb))

    def test_friction_observation_is_used(self):
        rb=copy.deepcopy(self.rb);rb[-1]['patches']['friction'][0]['sum_friction_vector_world'][0]+=.1
        e=ProbeEvidence();a=e.rows(self.raw,rb);b=e.rows(self.raw,self.rb)
        self.assertAlmostEqual(float(a[-1,46]-b[-1,46]),.1,places=6)

    def test_future_observations_do_not_change_prefix(self):
        e=ProbeEvidence();a=e.rows(self.raw,self.rb)
        for end in [1,80,190,191,len(a)]:np.testing.assert_array_equal(a[:end],e.rows(self.raw[:end],self.rb[:end]))

    def test_stale_step_rejected(self):
        rb=copy.deepcopy(self.rb);rb[-1]['step']-=1
        with self.assertRaises(ValueError):ProbeEvidence().rows(self.raw,rb)

    def test_branch_input_rejected(self):
        raw=copy.deepcopy(self.raw);raw[-1]['probe_phase']='branch_hold'
        with self.assertRaises(ValueError):ProbeEvidence().rows(raw,self.rb)

    def test_wrong_pair_order_rejected(self):
        rb=copy.deepcopy(self.rb);rb[-1]['patches']['friction'].reverse()
        with self.assertRaises(ValueError):ProbeEvidence().rows(self.raw,rb)

    def test_padding_not_a_model_observation(self):
        torch.manual_seed(0);m=FrictionMember();m.eval()
        a=np.random.default_rng(0).normal(size=(3,58)).astype(np.float32)
        b=np.random.default_rng(1).normal(size=(8,58)).astype(np.float32)
        with torch.no_grad():
            p=m(*batch([a]))[0];q=m(*batch([a,b]))[0]
        torch.testing.assert_close(p[0],q[0])

    def test_variable_probe_steps_allowed_but_completion_required(self):
        verify_decision_prefix(self.raw)
        with self.assertRaises(ValueError):verify_decision_prefix(self.raw[:191])

    def test_contact_loss_refuses_deployment(self):
        raw=copy.deepcopy(self.raw);raw[-1]['contact_left']=0
        with self.assertRaises(ValueError):verify_decision_prefix(raw)

    def test_mixed_context_refuses_deployment(self):
        raw=copy.deepcopy(self.raw);raw[-1]['trial_id']='other'
        with self.assertRaises(ValueError):verify_decision_prefix(raw)

    def test_reordered_phases_refuse_deployment(self):
        raw=copy.deepcopy(self.raw);raw[0],raw[-1]=raw[-1],raw[0]
        with self.assertRaises(ValueError):verify_decision_prefix(raw)


if __name__=='__main__':unittest.main()
