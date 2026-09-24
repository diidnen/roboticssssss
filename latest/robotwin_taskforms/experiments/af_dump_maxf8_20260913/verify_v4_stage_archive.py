"""Verify destination bytes and completion gates for model/confirmation archives."""
import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
from datetime import datetime, timezone


def verify(base, name, expected_sha):
    sys.path.insert(0, str(base.parent / 'runtime_observer_20260913'))
    from verify_completed_rim20_archive import read_members, sha
    archive = base / name
    if archive.resolve().parent != base.resolve():
        raise ValueError('Archive outside destination')
    receipt = json.loads(archive.with_suffix('.receipt.json').read_text())
    if receipt['sha256'] != expected_sha or sha(archive) != expected_sha or archive.stat().st_size != receipt['size']:
        raise ValueError('Destination/source receipt mismatch')
    manifest, files, payloads = read_members(archive)
    if len(files) != receipt['files'] or manifest['stage'] != receipt['stage'] or manifest['case'] != receipt['case']:
        raise ValueError('Receipt/manifest mismatch')
    if manifest['stage'] == 'models':
        prefix = 'af_dump_maxf8_20260913/models_v4/'
        complete = payloads[prefix + 'TRAINING_COMPLETE.json']
        if not complete['completed'] or complete['test_groups_executed'] != 0:
            raise ValueError('Invalid model completion')
        if files[prefix + 'TRAINING_PROTOCOL.json']['sha256'] != complete['protocol_sha256']:
            raise ValueError('Training protocol mismatch')
        for part in ['belief', 'feasibility']:
            model = prefix + part + '/'
            if files[model + part.upper() + '_MANIFEST.json']['sha256'] != complete[part + '_manifest_sha256']:
                raise ValueError('Model manifest mismatch')
            lock = payloads[model + 'CHECKPOINT_SELECTION_LOCK.json']
            if len(lock['checkpoints']) != 3:
                raise ValueError('Incomplete ensemble')
            for checkpoint in lock['checkpoints']:
                member = model + PurePosixPath(checkpoint['path']).name
                if files[member]['sha256'] != checkpoint['sha256']:
                    raise ValueError('Checkpoint hash mismatch')
    elif manifest['stage'] == 'confirmation-case':
        case = manifest['case']
        prefix = 'confirmation/inference_v4/'
        names = [prefix + 'INFERENCE_PROTOCOL.json', prefix + case + '/PROCESS_EXIT.json',
                 prefix + case + '/job/AF_INFERENCE_RESULT.json']
        selected = {}
        with tarfile.open(archive, 'r|gz') as stream:
            for member in stream:
                if member.name in names:
                    selected[member.name] = json.load(stream.extractfile(member))
        protocol, exited, result = [selected[n] for n in names]
        if exited['exit_code'] != 0 or files[names[2]]['sha256'] != exited['result_sha256']:
            raise ValueError('Invalid closed-case result')
        context = next(c for c in protocol['contexts'] if c['id'] == case)
        if result['context'] != context or sorted(r['method'] for r in result['outcomes']) != sorted(protocol['methods']):
            raise ValueError('Incomplete paired context')
    else:
        raise ValueError('Unsupported stage')
    return {'passed': True, 'archive': name, 'sha256': expected_sha, 'files_verified': len(files),
            'stage': manifest['stage'], 'case': manifest['case'],
            'scientific_audit_not_replaced': True, 'verified_utc': datetime.now(timezone.utc).isoformat(),
            'verifier_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base', type=Path)
    parser.add_argument('name')
    parser.add_argument('expected_sha')
    args = parser.parse_args()
    print(json.dumps(verify(args.base, args.name, args.expected_sha)), flush=True)
