"""Archive terminal V3 models/TEST and link all 16 separately saved raw groups.

Backup-only utility outside the frozen scientific source tree. Refuses partial
runs, absent observer closure, symlinks, changed input files, and overwrites.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(restore, observer):
    names = ('original_models_v3_rim20', 'original_inference_v3_rim20',
             'original_pipeline_v3_rim20', 'original_rim20_v3_driver')
    files = set()
    for directory in [*(restore / name for name in names), observer]:
        for path in directory.rglob('*'):
            if path.is_symlink():
                raise ValueError('Symlink requires explicit backup handling: ' + str(path))
            if path.is_file() and '__pycache__' not in path.parts:
                files.add(path)
    dataset = restore / 'original_rootlocal_dataset_v3_rim20'
    files.update(p for p in dataset.rglob('*')
                 if p.is_file() and 'groups' not in p.relative_to(dataset).parts
                 and '__pycache__' not in p.parts)
    files.update(restore.glob('*.py'))
    files.update(restore.glob('*.md'))
    return files


def pack(restore, observer, destination):
    restore, observer, destination = (p.resolve() for p in (restore, observer, destination))
    base = restore.parent
    if observer.parent != base or destination.parent != base:
        raise ValueError('Use sibling observer directory and archive under experiments')
    required = {
        restore / 'original_rootlocal_dataset_v3_rim20/COLLECTION_COMPLETE.json': 'completed',
        restore / 'original_models_v3_rim20/TRAINING_COMPLETE.json': 'completed',
        restore / 'original_inference_v3_rim20/FINAL_INFERENCE_RESULTS.json': 'completed',
        restore / 'original_inference_v3_rim20/INDEPENDENT_FINAL_RESULT_AUDIT.json': 'passed',
        restore / 'original_pipeline_v3_rim20/RIM20_PIPELINE_AUDITED.json': 'completed',
        restore / 'original_rim20_v3_driver/COMPLETE.json': 'completed',
    }
    for path, key in required.items():
        if read(path).get(key) is not True:
            raise ValueError('Terminal gate not satisfied: ' + str(path))
    read(observer / 'driver736766_capture/OBSERVER_EXIT.json')
    dataset = restore / 'original_rootlocal_dataset_v3_rim20'
    collection = read(dataset / 'COLLECTION_COMPLETE.json')
    if collection['total_rows'] != 128 or len(collection['contexts']) != 16:
        raise ValueError('Not the complete frozen collection')
    final = read(restore / 'original_inference_v3_rim20/FINAL_INFERENCE_RESULTS.json')
    if final['paired_rollouts'] != 24 or final['independent_query_settings'] != 4:
        raise ValueError('Not the complete frozen TEST')
    groups = []
    receipts = set()
    for context in read(dataset / 'CONTEXTS.json'):
        if context['split'] == 'TEST':
            continue
        receipt_path = base / ('af_dump_rim20_v3_' + context['id'] + '_20260913.tar.receipt.json')
        receipt = read(receipt_path)
        archive = Path(receipt['archive'])
        if archive.parent != base or archive.stat().st_size != receipt['size']:
            raise ValueError('Raw group archive missing or incorrect size')
        if digest(archive) != receipt['sha256']:
            raise ValueError('Raw group archive checksum mismatch')
        receipts.add(receipt_path)
        groups.append({'context': context['id'], 'receipt': receipt})
    if len(groups) != 16:
        raise ValueError('Need all 16 full raw group archives')
    files = inventory(restore, observer) | receipts
    manifest = {'terminal_run': True, 'raw_groups_separately_archived': groups,
                'anvil_destination_verification_still_required': True,
                'anvil_execution_qualification_claimed': False, 'files': {}}
    with destination.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for path in sorted(files):
                if path.is_symlink():
                    raise ValueError('Symlink input: ' + str(path))
                before = path.stat()
                content = path.read_bytes()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Input changed during snapshot: ' + str(path))
                name = str(path.relative_to(base))
                info = tarfile.TarInfo(name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
                manifest['files'][name] = {'sha256': hashlib.sha256(content).hexdigest(), 'size': len(content)}
            if files != inventory(restore, observer) | receipts:
                raise ValueError('Input inventory changed during snapshot')
            content = json.dumps(manifest, indent=2).encode()
            info = tarfile.TarInfo('ARCHIVE_MANIFEST.json')
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    result = {'archive': str(destination), 'size': destination.stat().st_size,
              'sha256': digest(destination), 'files': len(files), 'linked_full_raw_groups': 16,
              'anvil_destination_verification_still_required': True}
    with destination.with_suffix('.receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restore', type=Path)
    parser.add_argument('observer', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    pack(args.restore, args.observer, args.destination)
