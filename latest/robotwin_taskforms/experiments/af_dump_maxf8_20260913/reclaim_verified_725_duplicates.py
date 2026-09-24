"""Remove only the exact verified duplicate source archive and its 37 transport parts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from assemble_v4_shards import sha

HERE = Path(__file__).resolve().parent
ARCHIVES = Path('/media/volume/newdata/exouser/af_dump_maxf8_archives_20260913')
NAME = 'af_dump_maxf8_train_mu0.725_ps40200002_full_group_20260913T055239Z.tar.gz'
HASH = '54236888cb408a7270df599ff9d441908a88c586f9d609bcf603fe3c35cd79a9'
MANIFEST_HASH = 'e984a6c38f19171e56147007cd1225bf51a113bf2403b5de4a613227f00b05b3'


def main(apply):
    proof_path = HERE / 'ANVIL_BACKUP_AUDIT_12.json'
    proof = json.loads(proof_path.read_text())
    expected = proof['archives']
    if (not proof['passed'] or len(expected) != 1 or not expected[0]['passed'] or
            expected[0]['archive'] != NAME or expected[0]['sha256'] != HASH or
            expected[0]['files_verified'] != 718):
        raise ValueError('Missing exact independent Anvil proof')
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(proof['finished_utc'])).total_seconds()
    if not 0 <= age < 1800:
        raise ValueError('Require fresh Anvil verification')
    raw = HERE / 'additional_data_v1/groups/train_mu0.725_ps40200002_full_group/attempt_001'
    if not (raw / 'ACCEPTED.json').is_file() or not (raw / 'job/query/original_raw_rows.json').is_file():
        raise ValueError('Original accepted raw data must remain')
    directory = ARCHIVES / (NAME + '.parts_v1')
    manifest_path = directory / 'MANIFEST.json'
    if sha(manifest_path) != MANIFEST_HASH or directory.is_symlink() or directory.resolve().parent != ARCHIVES:
        raise ValueError('Unexpected source shard directory')
    manifest = json.loads(manifest_path.read_text())
    if len(manifest['parts']) != 37 or manifest['source_receipt']['sha256'] != HASH:
        raise ValueError('Unexpected source manifest')
    targets = [(ARCHIVES / NAME, expected[0]['bytes'], HASH)]
    for index, row in enumerate(manifest['parts']):
        if row['name'] != 'part_%04d.bin' % index:
            raise ValueError('Noncanonical source part')
        targets.append((directory / row['name'], row['size'], row['sha256']))
    for path, size, digest in targets:
        if path.is_symlink() or path.resolve().parent not in [ARCHIVES, directory]:
            raise ValueError('Unsafe exact deletion target')
        if path.stat().st_size != size or sha(path) != digest:
            raise ValueError('Source changed; stop before deletion')
    result = {'apply': apply, 'bytes': sum(size for _, size, _ in targets),
              'targets_verified': len(targets), 'proof_sha256': sha(proof_path),
              'helper_sha256': sha(Path(__file__)), 'removed': [],
              'original_raw_and_anvil_copies_retained': True,
              'utc': datetime.now(timezone.utc).isoformat()}
    record = HERE / 'SHARD_SOURCE_DUPLICATES_RECLAIMED_01.json'
    if apply:
        if record.exists():
            raise ValueError('Never repeat this reclaim')
        for path, _, _ in targets:
            path.unlink()
            result['removed'].append(str(path))
        with record.open('x') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    main(parser.parse_args().apply)
