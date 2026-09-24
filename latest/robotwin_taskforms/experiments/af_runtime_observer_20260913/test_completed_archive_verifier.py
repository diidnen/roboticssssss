import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from verify_completed_rim20_archive import read_members


class ArchiveVerifierTests(unittest.TestCase):
    def make(self, directory, mode='valid'):
        path = Path(directory) / 'test.tar.gz'
        name = '../escape' if mode == 'unsafe' else 'tree/evidence.txt'
        content = b'raw evidence'
        manifest = {'files': {name: {'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}}}
        if mode == 'corrupt':
            manifest['files'][name]['sha256'] = '0' * 64
        with tarfile.open(path, 'w:gz') as archive:
            entries = [(name, content)]
            if mode == 'duplicate':
                entries.append((name, content))
            entries.append(('ARCHIVE_MANIFEST.json', json.dumps(manifest).encode()))
            for member, data in entries:
                info = tarfile.TarInfo(member)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return path

    def test_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, files, _ = read_members(self.make(directory))
            self.assertEqual(manifest['files'], files)

    def test_rejections(self):
        for mode in ['unsafe', 'duplicate', 'corrupt']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    read_members(self.make(directory, mode))


if __name__ == '__main__':
    unittest.main()
