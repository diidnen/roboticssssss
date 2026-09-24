"""Read-only destination audit of 16 full V3 raw archives, not partial snapshots."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_one(base, expected):
    archive = base / expected['archive']
    if archive.resolve().parent != base or archive.is_symlink():
        raise ValueError('Archive outside selected backup directory')
    if archive.stat().st_size != expected['size'] or sha(archive) != expected['sha256']:
        raise ValueError('Source/destination archive mismatch: ' + archive.name)
    receipt = archive.with_suffix('.receipt.json')
    if sha(receipt) != expected['receipt_sha256']:
        raise ValueError('Source/destination receipt mismatch')
    observed, manifest, labels, group_audit = {}, None, {}, None
    prefix = 'original_rootlocal_dataset_v3_rim20/groups/' + expected['context'] + '/attempt_001/job/'
    with tarfile.open(archive, mode='r|gz') as tar:
        for member in tar:
            if not member.isfile() or member.name in observed:
                raise ValueError('Unexpected/duplicate archive member')
            keep = (member.name == 'ARCHIVE_MANIFEST.json'
                    or (member.name.startswith(prefix + 'branch_') and member.name.endswith('/result.json'))
                    or member.name == prefix + 'INDEPENDENT_GROUP_AUDIT.json')
            chunks = []
            h = hashlib.sha256()
            size = 0
            stream = tar.extractfile(member)
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
                size += len(chunk)
                if keep:
                    chunks.append(chunk)
            if size != member.size:
                raise ValueError('Truncated member')
            if member.name == 'ARCHIVE_MANIFEST.json':
                if manifest is not None:
                    raise ValueError('Duplicate manifest')
                manifest = json.loads(b''.join(chunks))
                continue
            observed[member.name] = {'sha256': h.hexdigest(), 'size': size}
            if keep:
                value = json.loads(b''.join(chunks))
                if member.name.endswith('/INDEPENDENT_GROUP_AUDIT.json'):
                    group_audit = value
                else:
                    labels[member.name] = value
    if manifest is None or manifest['files'] != observed:
        raise ValueError('Archive content differs from per-file manifest')
    if manifest['context']['id'] != expected['context']:
        raise ValueError('Wrong context archive')
    if manifest['validated_terminal_group'] is not True or manifest['live_groups_included'] is not False:
        raise ValueError('Partial or unqualified archive')
    if group_audit is None or group_audit['passed'] is not True or group_audit['branches'] != 8:
        raise ValueError('Missing independent raw group audit')
    if len(labels) != 8 or any(row.get('completed') is not True for row in labels.values()):
        raise ValueError('Missing terminal branch labels')
    return {'context': expected['context'], 'archive': archive.name, 'sha256': expected['sha256'],
            'receipt_sha256': expected['receipt_sha256'], 'files_verified': len(observed),
            'terminal_labels': len(labels), 'bytes': expected['size'], 'passed': True}


def main(base, expectations):
    base = base.resolve()
    expected = json.loads(expectations.read_text())
    contexts = [row['context'] for row in expected['groups']]
    if len(contexts) != 16 or len(set(contexts)) != 16:
        raise ValueError('Expected exactly16 unique full groups')
    rows = []
    for row in expected['groups']:
        checked = verify_one(base, row)
        rows.append(checked)
        print('GROUP_VERIFIED ' + json.dumps(checked), flush=True)
    result = {'passed': True, 'groups': rows, 'terminal_labels': sum(r['terminal_labels'] for r in rows),
              'expectations_sha256': sha(expectations), 'verifier_sha256': sha(Path(__file__)),
              'collection_complete_sha256': expected['collection_complete_sha256'],
              'scope': 'full raw V3 TRAIN/VAL archives only; models/TEST and Anvil execution not claimed',
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    print('DATASET_BACKUP_AUDIT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backup_dir', type=Path)
    parser.add_argument('expectations', type=Path)
    args = parser.parse_args()
    main(args.backup_dir, args.expectations)
