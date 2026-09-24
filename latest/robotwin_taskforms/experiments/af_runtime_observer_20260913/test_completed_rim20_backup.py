"""Synthetic tests only; never opens the live scientific dataset."""
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import pack_completed_rim20_run as backup


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='af-backup-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.restore = self.base / 'restore'
        self.observer = self.base / 'observer'
        self.destination = self.base / 'final.tar.gz'
        self.dataset = self.restore / 'original_rootlocal_dataset_v3_rim20'
        gates = {
            'original_rootlocal_dataset_v3_rim20/COLLECTION_COMPLETE.json':
                {'completed': True, 'total_rows': 128, 'contexts': list(range(16))},
            'original_models_v3_rim20/TRAINING_COMPLETE.json': {'completed': True},
            'original_inference_v3_rim20/FINAL_INFERENCE_RESULTS.json':
                {'completed': True, 'paired_rollouts': 24, 'independent_query_settings': 4},
            'original_inference_v3_rim20/INDEPENDENT_FINAL_RESULT_AUDIT.json': {'passed': True},
            'original_pipeline_v3_rim20/RIM20_PIPELINE_AUDITED.json': {'completed': True},
            'original_rim20_v3_driver/COMPLETE.json': {'completed': True},
        }
        for name, data in gates.items():
            self.write(self.restore / name, data)
        self.write(self.observer / 'driver736766_capture/OBSERVER_EXIT.json', {'closed': True})
        contexts = [{'id': 'c' + str(i), 'split': 'TRAIN' if i < 12 else 'VAL'} for i in range(16)]
        self.write(self.dataset / 'CONTEXTS.json', contexts)
        self.archives = []
        for context in contexts:
            archive = self.base / ('af_dump_rim20_v3_' + context['id'] + '_20260913.tar.gz')
            archive.write_bytes(b'synthetic raw archive')
            self.archives.append(archive)
            self.write(archive.with_suffix('.receipt.json'), {
                'archive': str(archive), 'size': archive.stat().st_size,
                'sha256': backup.digest(archive)})

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def pack(self):
        with contextlib.redirect_stdout(io.StringIO()):
            backup.pack(self.restore, self.observer, self.destination)

    def test_complete_manifest_checksums_and_no_overwrite(self):
        self.pack()
        receipt = backup.read(self.destination.with_suffix('.receipt.json'))
        self.assertEqual(receipt['sha256'], backup.digest(self.destination))
        with tarfile.open(self.destination) as tar:
            manifest = json.load(tar.extractfile('ARCHIVE_MANIFEST.json'))
            self.assertEqual(len(manifest['raw_groups_separately_archived']), 16)
            for name, metadata in manifest['files'].items():
                content = tar.extractfile(name).read()
                self.assertEqual(metadata['size'], len(content))
                self.assertEqual(metadata['sha256'], backup.hashlib.sha256(content).hexdigest())
        with self.assertRaises(FileExistsError):
            self.pack()

    def test_incomplete_no_archive(self):
        self.write(self.restore / 'original_models_v3_rim20/TRAINING_COMPLETE.json', {'completed': False})
        with self.assertRaises(ValueError):
            self.pack()
        self.assertFalse(self.destination.exists())

    def test_corrupt_group_no_archive(self):
        content = self.archives[0].read_bytes()
        self.archives[0].write_bytes(b'X' + content[1:])
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.pack()
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
