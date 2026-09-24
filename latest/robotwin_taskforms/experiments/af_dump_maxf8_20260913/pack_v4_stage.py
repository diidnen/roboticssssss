"""Archive only terminal V4 models or a closed confirmation context; no experiment writes."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
CONFIRMATION = Path('/media/volume/newdata/exouser/af_dump_maxf8_confirmation_20260913/inference_v4')
ARCHIVES = Path('/media/volume/newdata/exouser/af_dump_maxf8_archives_20260913')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def select(stage, case):
    sources = {p: HERE.name + '/' + p.name for p in HERE.glob('*.py')}
    if stage == 'models':
        if case is not None:
            raise ValueError('Models stage has no case')
        root = HERE / 'models_v4'
        complete = read(root / 'TRAINING_COMPLETE.json')
        if not complete['completed'] or complete['test_groups_executed'] != 0:
            raise ValueError('Invalid training completion')
        for key, name in [('belief_manifest_sha256', 'belief/BELIEF_MANIFEST.json'),
                          ('feasibility_manifest_sha256', 'feasibility/FEASIBILITY_MANIFEST.json')]:
            if sha(root / name) != complete[key]:
                raise ValueError('Training manifest mismatch')
        for path in root.rglob('*'):
            if path.is_file():
                sources[path] = HERE.name + '/' + str(path.relative_to(HERE))
    elif stage == 'confirmation-case':
        if not case or Path(case).name != case or case in {'.', '..'}:
            raise ValueError('Require one exact context id')
        root = CONFIRMATION / case
        if root.resolve().parent != CONFIRMATION.resolve():
            raise ValueError('Case escapes confirmation directory')
        protocol = read(CONFIRMATION / 'INFERENCE_PROTOCOL.json')
        context = next(c for c in protocol['contexts'] if c['id'] == case)
        receipt = read(root / 'PROCESS_EXIT.json')
        result = root / 'job/AF_INFERENCE_RESULT.json'
        if receipt['exit_code'] != 0 or sha(result) != receipt['result_sha256']:
            raise ValueError('Case not closed with matching result')
        value = read(result)
        if value['context'] != context:
            raise ValueError('Result context mismatch')
        if sorted(r['method'] for r in value['outcomes']) != sorted(protocol['methods']):
            raise ValueError('Incomplete paired methods')
        for path in root.rglob('*'):
            if path.is_file():
                sources[path] = 'confirmation/inference_v4/' + str(path.relative_to(CONFIRMATION))
        sources[CONFIRMATION / 'INFERENCE_PROTOCOL.json'] = 'confirmation/inference_v4/INFERENCE_PROTOCOL.json'
    else:
        raise ValueError('Unknown stage')
    return sources


def main(stage, case):
    sources = select(stage, case)
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = ARCHIVES / ('af_dump_maxf8_' + stage + '_' + (case or 'all') + '_' + stamp + '.tar.gz')
    manifest = {'stage': stage, 'case': case, 'partial_run': True, 'files': {}}
    with destination.open('xb') as output, tarfile.open(fileobj=output, mode='w:gz') as archive:
        for path, name in sorted(sources.items()):
            if path.is_symlink():
                raise ValueError('Symlink requires explicit handling')
            before = path.stat()
            digest = sha(path)
            info = tarfile.TarInfo(name)
            info.size = before.st_size
            with path.open('rb') as stream:
                archive.addfile(info, stream)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha(path) != digest:
                raise ValueError('Source changed during backup')
            manifest['files'][name] = {'size': info.size, 'sha256': digest}
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo('ARCHIVE_MANIFEST.json')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    receipt = {'archive': str(destination), 'stage': stage, 'case': case,
               'size': destination.stat().st_size, 'sha256': sha(destination), 'files': len(sources)}
    with destination.with_suffix('.receipt.json').open('x') as output:
        json.dump(receipt, output, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['models', 'confirmation-case'])
    parser.add_argument('--case')
    args = parser.parse_args()
    main(args.stage, args.case)
