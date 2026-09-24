"""Backup-only snapshot of closed model files and sealed terminal TEST evidence.

No experimental imports, inference, selection, auditing writes, or retries.
This partial backup is not scientific completion or a substitute for final audit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
from pack_completed_rim20_run import read, digest


def select(restore, case_id=None):
    models = restore / 'original_models_v3_rim20'
    complete = read(models / 'TRAINING_COMPLETE.json')
    if complete.get('completed') is not True or complete.get('test_groups_executed') != 0:
        raise ValueError('Training must be terminal and locked before TEST')
    for part, name in [('belief', 'BELIEF'), ('feasibility', 'FEASIBILITY')]:
        if digest(models / part / (name + '_MANIFEST.json')) != complete[part + '_manifest_sha256']:
            raise ValueError('Training manifest changed')
        if len(read(models / part / 'CHECKPOINT_SELECTION_LOCK.json')['checkpoints']) != 3:
            raise ValueError('Missing original model selection lock')
    files = set(p for p in models.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    dataset = restore / 'original_rootlocal_dataset_v3_rim20'
    if read(dataset / 'COLLECTION_COMPLETE.json')['total_rows'] != 128:
        raise ValueError('Incomplete collection')
    files.update(p for p in dataset.glob('*.json') if p.is_file())
    files.update(p for p in (dataset / 'SOURCE_SNAPSHOT').rglob('*') if p.is_file())
    files.update(restore.glob('*.py'))
    files.update(restore.glob('*.md'))
    inference = restore / 'original_inference_v3_rim20'
    if (inference / 'INFERENCE_PROTOCOL.json').exists():
        files.add(inference / 'INFERENCE_PROTOCOL.json')
    contexts = read(dataset / 'CONTEXTS.json')
    if case_id is not None and case_id not in [c['id'] for c in contexts if c['split'] == 'TEST']:
        raise ValueError('Requested context is not a frozen TEST context')
    sealed_cases, terminal_branches = [], []
    for context in contexts:
        if context['split'] != 'TEST':
            continue
        if case_id is not None and context['id'] != case_id:
            continue
        case = inference / context['id']
        job = case / 'job'
        seal_path = job / 'PREACTION_SELECTION_LOCK.json'
        if not seal_path.exists():
            continue
        seal = read(seal_path)
        if seal['context'] != context or seal['candidate_actions_executed'] != 0 or seal['labels_read']:
            raise ValueError('Unqualified selection lock')
        if seal['model_manifests'] != {p: complete[p + '_manifest_sha256'] for p in ('belief', 'feasibility')}:
            raise ValueError('Changed TEST model lock')
        for name, expected in seal['artifact_hashes'].items():
            path = job / name
            if digest(path) != expected:
                raise ValueError('Changed sealed artifact')
            files.add(path)
        if read(job / 'query/qualification.json').get('completed') is not True:
            raise ValueError('Query not terminal')
        files.add(seal_path)
        files.update(p for p in (job / 'query').rglob('*') if p.is_file())
        files.update(p for p in case.glob('*.json') if p.name != 'PROCESS_EXIT.json')
        sealed_cases.append(context['id'])
        for terminal in sorted(job.glob('branch_*/result.json')):
            if read(terminal).get('completed') is not True:
                continue
            terminal_branches.append(str(terminal.parent.relative_to(restore)))
            files.update(p for p in terminal.parent.rglob('*') if p.is_file())
        # Entire case is immutable only once its process has exited successfully.
        exit_path = case / 'PROCESS_EXIT.json'
        if exit_path.exists() and read(exit_path)['exit_code'] == 0:
            files.update(p for p in case.rglob('*') if p.is_file())
    return files, sealed_cases, terminal_branches


def pack(restore, case_id=None):
    restore = restore.resolve()
    files, cases, branches = select(restore, case_id)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = restore.parent / f'af_dump_rim20_model_test_snapshot_{len(branches):02d}_{stamp}.tar.gz'
    metadata = {'partial_snapshot_not_final_result': True, 'independent_final_audit_still_required': True,
                'closed_training_tree_included': True, 'live_unsealed_evidence_excluded': True,
                'requested_test_context': case_id,
                'sealed_test_contexts': cases, 'terminal_test_branches': branches, 'files': {}}
    with destination.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for path in sorted(files):
                if path.is_symlink():
                    raise ValueError('Unexpected symlink')
                before = path.stat()
                content = path.read_bytes()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Input changed during backup')
                name = str(path.relative_to(restore))
                info = tarfile.TarInfo(name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
                metadata['files'][name] = {'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
            content = json.dumps(metadata, indent=2).encode()
            info = tarfile.TarInfo('ARCHIVE_MANIFEST.json')
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    result = {'archive': str(destination), 'size': destination.stat().st_size, 'sha256': digest(destination),
              'requested_test_context': case_id,
              'terminal_test_branches': len(branches), 'files': len(files), 'partial_snapshot_not_final_result': True}
    with destination.with_suffix('.receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('restore', type=Path)
    parser.add_argument('--case', dest='case_id', help='Only this frozen TEST context; model tree always included')
    args = parser.parse_args()
    pack(args.restore, args.case_id)
