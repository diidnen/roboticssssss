"""Prevent changed outcomes or incomplete rechecks bypassing final gates."""
import copy
import tempfile
import unittest
from pathlib import Path
from common import sha
from prepare_final_online_vla import replacement_server_admitted
from review_current_server36 import transfer_gates


class CurrentServerAdmission(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.ready=self.root/'ready.json';self.ready.write_text('{}')
        self.metadata={'checkpoint_sha256':'checkpoint'}
        self.parity=dict(role='COMPLETE_CURRENT_SERVER36_QUALIFICATION_REVIEW',passed=True,
            reviewed_branches=36,combined_primary_branches=72,numerical_diagnostic_requests=1400,
            all_four_force_boundaries_valid=True,VLA_SEED_MATCHING_VALID=True,
            server_ready_sha256=sha(self.ready),checkpoint_sha256='checkpoint',
            source_qualification=str(self.root),same_current_server_instance=True,
            VLA_POSTPROBE_BEHAVIOR_VALID=True,no_outcome_retry=True,no_parameters_changed=True,
            all_original_outcomes_reproduced=False,
            transfer=dict(passed=True,original_thresholds_reused_without_relaxation=True,
                gates={k:True for k in ('primary_complete','both_outcomes_observed','auroc_at_least_0_70',
                    'brier_at_most_0_22','ece5_at_most_0_20','af_within_two_successes_of_fixed5',
                    'af_not_constant_force','force_boundary_all_four_tasks')}))

    def valid(self,p):
        return replacement_server_admitted(p,self.root,self.ready,self.metadata)

    def test_changed_outcome_requires_complete_current_qualification(self):
        self.assertTrue(self.valid(self.parity))
        for n in (14,35):
            p=copy.deepcopy(self.parity);p['reviewed_branches']=n
            self.assertFalse(self.valid(p))

    def test_failed_transfer_cannot_be_hidden_by_top_level_pass(self):
        p=copy.deepcopy(self.parity);p['transfer']['gates']['brier_at_most_0_22']=False
        self.assertFalse(self.valid(p))
        p=copy.deepcopy(self.parity);p['transfer']['gates'].pop('ece5_at_most_0_20')
        self.assertFalse(self.valid(p))

    def test_server_and_seed_binding_required(self):
        for k,v in (('server_ready_sha256','changed'),('VLA_SEED_MATCHING_VALID',False),
                    ('same_current_server_instance',False),('no_outcome_retry',False)):
            p=copy.deepcopy(self.parity);p[k]=v;self.assertFalse(self.valid(p))

    def test_failed_strict_parity_still_rejected(self):
        p=copy.deepcopy(self.parity);p.update(role='PHYSICAL_RESTART_QUALIFICATION_ADMISSION_REVIEW',
            reviewed_branches=14,all_original_outcomes_and_forces_reproduced=False)
        self.assertFalse(self.valid(p))

    def test_original_actual_dev_report_passes_unchanged_thresholds(self):
        from common import HERE,read
        report=read(HERE/'INDEPENDENT_QUALIFICATION_FINAL_REVIEW.json')
        report=copy.deepcopy(report)
        # The old report includes four separate Fixed4 boundary additions.
        report['probabilities']['rows']=[r for r in report['probabilities']['rows'] if r['method']!='FIXED_4']
        self.assertTrue(transfer_gates(report)['passed'])
        report['probabilities']['main_probability']['Brier']=.23
        self.assertFalse(transfer_gates(report)['passed'])


if __name__=='__main__':unittest.main()
