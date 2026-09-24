"""No-physics negative controls for entry to untouched final test roots."""
import json
import tempfile
import unittest
from pathlib import Path
import run_final_online_vla as runner
import prepare_final_online_vla as freezer
from common import sha


class FinalAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.manifest = {k: True for k in runner.REQUIRED_GATES}
        self.manifest.update(FINAL_RUNTIME_USES_ONLINE_VLA=True, final_runtime_frozen=True,
            EXISTING_FEASIBILITY_VLA_TRANSFER='PASS', DOWNSTREAM_ACTION_SOURCE='ONLINE_VLA',
            source_hashes={str(Path(runner.__file__).resolve()): sha(runner.__file__)})
        # Synthetic identifiers exist only inside a temporary fixture; no
        # simulator or policy client can be constructed by validate().
        objects={0:'alphabet_soup_1',1:'cream_cheese_1',5:'tomato_sauce_1',6:'butter_1'}
        self.contexts = [dict(root=r, task=t, band=b, id=f't{t}_r{r}_{b.lower()}',object=objects[t],target='basket_1',mu=.5)
            for r in (1, 2, 3, 4) for t in (0, 1, 5, 6) for b in ('LOW','MID','HIGH')]
        self.save()

    def save(self):
        for name in ('FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json','CANDIDATE_RUNTIME_MANIFEST.json'):
            (self.out/name).write_text(json.dumps(self.manifest))
        (self.out/'DEV_PLAN.json').write_text(json.dumps({'contexts': self.contexts}))
        (self.out/'FINAL_FRESH_ROOT_PLAN.json').write_text(json.dumps({
            'contexts':self.contexts,'roots':[1,2,3,4],'root_selection_used_outcomes':False}))

    def test_complete_fixture_validates_without_physics(self):
        self.assertEqual(len(runner.validate(self.out)[1]), 48)

    def test_replay_and_scripted_rejected(self):
        for source in ('SCRIPTED','VLA_REPLAY'):
            with self.subTest(source=source):
                self.manifest['DOWNSTREAM_ACTION_SOURCE']=source;self.save()
                with self.assertRaisesRegex(RuntimeError,'forbidden'):
                    runner.validate(self.out)

    def test_missing_postprobe_gate_rejected(self):
        del self.manifest['VLA_POSTPROBE_BEHAVIOR_VALID'];self.save()
        with self.assertRaisesRegex(RuntimeError,'gate incomplete'):
            runner.validate(self.out)

    def test_pending_transfer_rejected(self):
        self.manifest['EXISTING_FEASIBILITY_VLA_TRANSFER']='PENDING';self.save()
        with self.assertRaisesRegex(RuntimeError,'not been admitted'):
            runner.validate(self.out)

    def test_changed_source_rejected(self):
        self.manifest['source_hashes'][str(Path(runner.__file__).resolve())]='0'*64;self.save()
        with self.assertRaisesRegex(RuntimeError,'changed'):
            runner.validate(self.out)

    def test_duplicate_context_rejected(self):
        self.contexts[-1]=dict(self.contexts[0]);self.save()
        with self.assertRaisesRegex(RuntimeError,'coverage or identity'):
            runner.validate(self.out)

    def test_worker_manifest_divergence_rejected(self):
        (self.out/'CANDIDATE_RUNTIME_MANIFEST.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'diverges'):
            runner.validate(self.out)

    def test_cross_task_object_misrouting_rejected_before_physics(self):
        self.contexts[0]['object']='butter_1';self.save()
        with self.assertRaisesRegex(RuntimeError,'object/target'):
            runner.validate(self.out)

    def test_noncanonical_context_rejected_before_physics(self):
        self.contexts[0]['id']='renamed_context';self.save()
        with self.assertRaisesRegex(RuntimeError,'Noncanonical'):
            runner.validate(self.out)

    def test_freeze_refuses_incomplete_qualification_without_creating_output(self):
        admission={k:True for k in runner.REQUIRED_GATES}
        admission.update(qualification_directory=str(self.out),EXISTING_FEASIBILITY_VLA_TRANSFER='PASS',completed_primary_branches=27)
        (self.out/'admission.json').write_text(json.dumps(admission))
        (self.out/'fresh.json').write_text('{}')
        destination=self.out/'new_final'
        with self.assertRaisesRegex(RuntimeError,'qualification incomplete'):
            freezer.prepare(self.out,self.out/'admission.json',self.out/'fresh.json',destination)
        self.assertFalse(destination.exists())

    def test_freeze_refuses_exposed_root_without_creating_output(self):
        proof=self.out/'proof.json';proof.write_text('{}')
        evidence={str(proof):sha(proof)}
        admission={k:True for k in runner.REQUIRED_GATES}
        admission.update(qualification_directory=str(self.out),EXISTING_FEASIBILITY_VLA_TRANSFER='PASS',completed_primary_branches=36,evidence_sha256=evidence)
        fresh=dict(FRESH_ROOT_NONEXPOSURE_VERIFIED=True,root_selection_used_outcomes=False,
            physics_branches_started_before_freeze=0,evidence_sha256=evidence,roots=[1,2,3,4],excluded_root_or_seed_ids=[1])
        (self.out/'admission.json').write_text(json.dumps(admission))
        (self.out/'fresh.json').write_text(json.dumps(fresh))
        destination=self.out/'new_final'
        with self.assertRaisesRegex(RuntimeError,'untouched roots'):
            freezer.prepare(self.out,self.out/'admission.json',self.out/'fresh.json',destination)
        self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
