"""Storage-only repair: verified duplicate cleanup and transparent fresh output directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
BASE = Path('/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments')
OUTPUT = BASE / 'af_dump_maxf8_confirmation_storage_20260913'
ALIAS = Path('/media/volume/newdata/exouser/af_dump_maxf8_confirmation_20260913')
NAMES = ['af_dump_original_full_controller_20260913_v1.tar.gz'] + [
    'af_dump_original_train_mu0.' + mu + '_root200002_20260913.tar.gz'
    for mu in ['650', '700', '750', '800', '850']]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(apply):
    proof_path = HERE / 'OUTPUT_SPACE_DUPLICATE_PROOF_01.json'
    proof = json.loads(proof_path.read_text())
    if (not proof['passed'] or not proof['source_and_anvil_sha256_match'] or
            proof['source_gzip_test_exit'] != 0 or proof['anvil_gzip_test_exit'] != 0):
        raise ValueError('Incomplete independent backup proof')
    if [row['name'] for row in proof['rows']] != NAMES:
        raise ValueError('Exact cleanup inventory changed')
    if OUTPUT.exists() or OUTPUT.is_symlink() or ALIAS.exists() or ALIAS.is_symlink():
        raise ValueError('Output paths already exist; inspect, never replace')
    if BASE.resolve() != BASE or OUTPUT.parent != BASE or ALIAS.parent.resolve() != ALIAS.parent:
        raise ValueError('Unexpected path resolution')
    rows = proof['rows']
    for row in rows:
        path = Path(row['source'])
        if path != BASE / row['name'] or path.is_symlink() or path.resolve().parent != BASE:
            raise ValueError('Unsafe exact duplicate target')
        expected_anvil = '/anvil/projects/x-cis250966/tabero-transfer/jetstream-activeforcing-20260912/' + row['name']
        if row['anvil'] != expected_anvil or path.stat().st_size != row['size'] or sha(path) != row['sha256']:
            raise ValueError('Source or backup receipt changed')
    result = {'apply': apply, 'verified_duplicate_bytes': sum(row['size'] for row in rows),
              'proof_sha256': sha(proof_path), 'source_sha256': sha(Path(__file__)),
              'removed': [], 'scientific_source_changes': False,
              'utc': datetime.now(timezone.utc).isoformat()}
    if apply:
        record = HERE / 'OUTPUT_STORAGE_APPLIED_01.json'
        if record.exists():
            raise ValueError('Already applied; never repeat')
        for row in rows:
            Path(row['source']).unlink()
            result['removed'].append(row['source'])
        OUTPUT.mkdir()
        if shutil.disk_usage(OUTPUT).free < 6 * 1024**3:
            raise RuntimeError('Insufficient capacity after verified cleanup; output alias not created')
        ALIAS.symlink_to(OUTPUT, target_is_directory=True)
        if ALIAS.resolve() != OUTPUT or ALIAS.stat().st_dev != OUTPUT.stat().st_dev:
            raise RuntimeError('Output alias binding failed')
        # The supervisor's existing mkdir(exist_ok=True) and disk_usage guard
        # operate on the same physical directory through this alias.
        ALIAS.mkdir(exist_ok=True)
        result.update(output_alias=str(ALIAS), physical_output=str(OUTPUT),
                      available_bytes=shutil.disk_usage(ALIAS).free,
                      original_raw_and_anvil_copies_retained=True)
        with record.open('x') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    main(parser.parse_args().apply)
