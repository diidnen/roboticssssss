import tempfile
import unittest
from pathlib import Path
import numpy as np
from activeforcing_decision_state import ContractError, DecisionClock, collection_gate, matched_training_input


class DecisionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.snapshot = Path(self.temp.name) / 'snapshot.bin'
        self.snapshot.write_bytes(b'immutable test snapshot')
        self.clock = DecisionClock()
        self.clock.stepped()
        self.kw = dict(context_id='test', task=0, snapshot_path=self.snapshot,
                       execution_contract_hash='test-only', state=np.arange(13),
                       mask=np.ones(13), commands=np.zeros((8,3)),
                       phases=['branch_hold']*8, posterior_support=[.3,.5,.9],
                       posterior_weights=[1/3]*3)

    def artifact(self):
        self.clock.finish_probe()
        return self.clock.extract(**self.kw)

    def test_reject_before_probe_completion(self):
        with self.assertRaises(ContractError): self.clock.extract(**self.kw)

    def test_reject_execution_without_snapshot_and_features(self):
        self.clock.finish_probe()
        with self.assertRaises(ContractError): self.clock.begin_execution()

    def test_candidate_parity_all_supports(self):
        self.assertEqual(self.artifact().candidate_parity()['MAX_FEATURE_DIFF'], 0.)

    def test_runtime_training_same_builder(self):
        a = self.artifact()
        label=dict(context_id=a.context_id,label_source='CURRENT_MATCHED_FULL_TASK_BRANCH',
            decision_snapshot_sha256=a.snapshot_sha256,execution_contract_hash=a.execution_contract_hash,
            candidate_F=4.,restore_execution_verified=True,lift_success=1,place_success=1,dropped=0,
            full_task_success_y=1,post_action_state='MUST_NOT_BE_READ')
        training,y=matched_training_input(a,4.,.5,label)
        np.testing.assert_array_equal(training,a.input(4.,.5))
        self.assertEqual(y,1)
        label['label_source']='OLD720'
        with self.assertRaises(ContractError):matched_training_input(a,4.,.5,label)

    def test_reject_post_action_extraction(self):
        self.artifact(); self.clock.begin_execution(); self.clock.stepped()
        with self.assertRaises(ContractError): self.clock.extract(**self.kw)

    def test_reject_step_during_planning(self):
        self.artifact()
        with self.assertRaises(ContractError): self.clock.stepped()

    def test_reject_mutated_snapshot(self):
        a = self.artifact(); self.snapshot.write_bytes(b'changed')
        with self.assertRaises(ContractError): a.input(3., .5)

    def test_no_hidden_state_alias(self):
        a = self.artifact(); x = a.input(3., .5); x[:,19] = 99
        self.assertEqual(a.input(3., .5)[0,19], 0.)

    def test_force_support_frozen(self):
        a = self.artifact()
        with self.assertRaises(ContractError): a.input(6., .5)

    def test_nonpositive_posterior_rejected(self):
        self.kw['posterior_support'] = [-1., .5, .9]
        with self.assertRaises(ContractError): self.artifact()

    def test_collection_gate_fails_closed(self):
        self.assertFalse(collection_gate(probe_validated=True, posterior_validated=False,
            decision_parity=True, execution_restore_validated=False,
            empirical_boundary_validated=False)['ready'])


if __name__ == '__main__': unittest.main()
