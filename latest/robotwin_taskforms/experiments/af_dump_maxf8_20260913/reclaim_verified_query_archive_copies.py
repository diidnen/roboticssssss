"""Reclaim exact Anvil-verified duplicate tar files, retaining all raw data and receipts."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ARCHIVES = Path('/media/volume/newdata/exouser/af_dump_maxf8_archives_20260913')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main(apply):
    inventory = json.loads((HERE/'VERIFIED_QUERY_ARCHIVE_DUPLICATES_01.json').read_text())['archives']
    proof = json.loads((HERE/'ANVIL_QUERY_DUPLICATES_RECHECK_01.json').read_text())
    if proof['passed'] is not True or len(inventory) != 12:
        raise ValueError('Missing full destination proof')
    verified = {row['archive']: row for row in proof['archives']}
    if set(verified) != {r['archive'] for r in inventory}:
        raise ValueError('Destination coverage mismatch')
    targets = []
    for row in inventory:
        name = row['archive']
        path = ARCHIVES/name
        if path.is_symlink() or path.resolve().parent != ARCHIVES.resolve():
            raise ValueError('Unsafe exact target')
        if not name.startswith('af_dump_maxf8_train_') or '_query_only_' not in name or not name.endswith('.tar.gz'):
            raise ValueError('Outside authorized duplicate scope')
        remote = verified[name]
        if remote['passed'] is not True or remote['sha256'] != row['sha256'] or remote['bytes'] != row['bytes']:
            raise ValueError('Destination differs from saved evidence')
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Source duplicate changed')
        receipt = json.loads(path.with_suffix('.receipt.json').read_text())
        if receipt['sha256'] != row['sha256']:
            raise ValueError('Source receipt mismatch')
        targets.append(path)
    report = {'validated': True, 'applied': apply, 'exact_targets': [str(p) for p in targets],
              'reclaim_bytes': sum(r['bytes'] for r in inventory),
              'anvil_proof_sha256': sha(HERE/'ANVIL_QUERY_DUPLICATES_RECHECK_01.json'),
              'raw_data_and_anvil_copies_retained': True, 'utc': datetime.now(timezone.utc).isoformat()}
    if apply:
        receipt = HERE/'QUERY_DUPLICATE_RECLAIM_01.json'
        if receipt.exists():
            raise ValueError('Reclaim already recorded')
        for path in targets:
            path.unlink()
        with receipt.open('x') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    main(parser.parse_args().apply)
