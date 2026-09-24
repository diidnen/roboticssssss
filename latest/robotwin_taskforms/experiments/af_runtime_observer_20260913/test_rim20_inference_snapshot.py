"""Synthetic backup tests; do not access live experiment paths."""
import contextlib
import io
import json
import tarfile
import unittest
from test_completed_rim20_backup import BackupTests
import pack_rim20_inference_snapshot as snapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.fixture = BackupTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.restore = self.fixture.restore
        self.write = self.fixture.write
        models = self.restore / 'original_models_v3_rim20'
        complete = {'completed': True, 'test_groups_executed': 0}
        for part, name in [('belief', 'BELIEF'), ('feasibility', 'FEASIBILITY')]:
            path = models / part / (name + '_MANIFEST.json')
            self.write(path, {'synthetic': True})
            complete[part + '_manifest_sha256'] = snapshot.digest(path)
            self.write(models / part / 'CHECKPOINT_SELECTION_LOCK.json', {'checkpoints': [0, 1, 2]})
        self.write(models / 'TRAINING_COMPLETE.json', complete)
        self.complete = complete

    def test_closed_models_before_test(self):
        files, cases, branches = snapshot.select(self.restore)
        self.assertTrue(any(p.name == 'TRAINING_COMPLETE.json' for p in files))
        self.assertEqual((cases, branches), ([], []))

    def test_sealed_terminal_only_and_archive_manifest(self):
        context = {'id': 'test_synthetic', 'split': 'TEST'}
        dataset = self.fixture.dataset
        contexts = snapshot.read(dataset / 'CONTEXTS.json') + [context]
        self.write(dataset / 'CONTEXTS.json', contexts)
        job = self.restore / 'original_inference_v3_rim20/test_synthetic/job'
        self.write(job / 'PREACTION_AF_DECISION.json', {'synthetic': True})
        self.write(job / 'PREACTION_SELECTION_LOCK.json', {
            'context': context, 'candidate_actions_executed': 0, 'labels_read': False,
            'model_manifests': {p: self.complete[p + '_manifest_sha256'] for p in ('belief', 'feasibility')},
            'artifact_hashes': {'PREACTION_AF_DECISION.json': snapshot.digest(job / 'PREACTION_AF_DECISION.json')}})
        self.write(job / 'query/qualification.json', {'completed': True})
        self.write(job / 'branch_0_3N/result.json', {'completed': True, 'success': False})
        self.write(job / 'branch_1_1N/live.json', {'in_progress': True})
        with contextlib.redirect_stdout(io.StringIO()) as output:
            snapshot.pack(self.restore)
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt['terminal_test_branches'], 1)
        with tarfile.open(receipt['archive']) as tar:
            manifest = json.load(tar.extractfile('ARCHIVE_MANIFEST.json'))
            self.assertTrue(manifest['partial_snapshot_not_final_result'])
            self.assertFalse(any('branch_1_' in name for name in manifest['files']))
            for name, entry in manifest['files'].items():
                data = tar.extractfile(name).read()
                self.assertEqual(entry['sha256'], snapshot.hashlib.sha256(data).hexdigest())

    def test_changed_training_manifest_rejected(self):
        self.write(self.restore / 'original_models_v3_rim20/belief/BELIEF_MANIFEST.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'Training manifest changed'):
            snapshot.select(self.restore)

    def test_context_scoping_and_unknown_context_rejection(self):
        contexts = [{'id': 'test_a', 'split': 'TEST'}, {'id': 'test_b', 'split': 'TEST'}]
        self.write(self.fixture.dataset / 'CONTEXTS.json', contexts)
        # Invalid/unsealed evidence in an unselected case must not be read or archived.
        self.write(self.restore / 'original_inference_v3_rim20/test_b/job/PREACTION_SELECTION_LOCK.json', {})
        files, cases, branches = snapshot.select(self.restore, 'test_a')
        self.assertEqual((cases, branches), ([], []))
        self.assertFalse(any('test_b' in p.parts for p in files))
        self.assertTrue(any(p.name == 'TRAINING_COMPLETE.json' for p in files))
        with self.assertRaisesRegex(ValueError, 'not a frozen TEST context'):
            snapshot.select(self.restore, 'missing')


if __name__ == '__main__':
    unittest.main(defaultTest='SnapshotTests')
