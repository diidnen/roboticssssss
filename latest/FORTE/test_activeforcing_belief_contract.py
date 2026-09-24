import unittest
from pathlib import Path
import numpy as np
import torch
import e1_verified_inference as rt
import activeforcing_e2e_task0_smoke_20260905 as smoke
from activeforcing_belief_contract import require_input_support

ROOT=rt.ROOT
OUT=ROOT/'analysis/results/physical_belief_transfer_root_cause_20260905'


class BeliefContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.old=rt.module('pre_guard_test_reference',OUT/'activeforcing_e2e_task0_smoke.before_belief_guard.py')
        cls.cid='p5s0c_train_t0_r00_s5100_high_mu0.940189'
        cls.path=smoke.OUT/'PROBE_TELEMETRY'/f'{cls.cid}_repeat1.csv'
        cls.after=smoke.posterior_from_probe(cls.path)

    def test_means_and_selector_support_unchanged(self):
        before=self.old.posterior_from_probe(self.path)
        for k in ['member_means','member_log_sigma','posterior_samples_mu','mean','std']:
            np.testing.assert_array_equal(before[k],self.after[k])

    def test_uncertainty_reconstructed_not_tuned(self):
        p=self.after
        expected=(np.var(p['member_means'],ddof=1)+np.mean(np.exp(2*np.asarray(p['member_log_sigma']))))*p['uncertainty_interval_scale']**2
        self.assertAlmostEqual(p['calibrated_total_std']**2,expected)
        self.assertIn('epistemic',p['std_semantics'])

    def test_current_probe_refused_by_actual_entrypoint(self):
        with self.assertRaisesRegex(ValueError,'BLOCKED_UNVERIFIED_PROBE_INPUT'):
            smoke.infer_curves(self.cid,self.after,probe_path=self.path)

    def test_diagnostic_cannot_write_historical_output(self):
        with self.assertRaisesRegex(ValueError,'separate output'):
            smoke.infer_curves(self.cid,self.after,probe_path=self.path,diagnostic_only=True)

    def test_unannotated_posterior_not_silently_approved(self):
        with self.assertRaises(ValueError):require_input_support({'member_means':[.3,.5,.7]})

    def test_historical_supported_probe_admitted_for_input_only(self):
        path=Path('/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_PROBE_TELEMETRY')/f'{self.cid}_probe_timesteps.csv'
        p=smoke.posterior_from_probe(path)
        require_input_support(p)
        self.assertFalse(p['final_method_deployment_approved'])


if __name__=='__main__':unittest.main()
