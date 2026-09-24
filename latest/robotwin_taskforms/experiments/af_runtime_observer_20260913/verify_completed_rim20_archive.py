"""Read-only final destination coverage check; never executes an experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_members(archive):
    observed, payloads, manifest = {}, {}, None
    keep_names = {'TRAINING_COMPLETE.json', 'COLLECTION_COMPLETE.json',
                  'CHECKPOINT_SELECTION_LOCK.json', 'FINAL_INFERENCE_RESULTS.json',
                  'INDEPENDENT_FINAL_RESULT_AUDIT.json', 'RIM20_PIPELINE_AUDITED.json',
                  'COMPLETE.json', 'OBSERVER_EXIT.json', 'ANVIL_FULL_DATASET_AUDIT.json'}
    with tarfile.open(archive, mode='r|gz') as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or member.name in observed:
                raise ValueError('Unsafe or duplicate archive member')
            keep = path.name in keep_names or member.name == 'ARCHIVE_MANIFEST.json'
            h, size, chunks = hashlib.sha256(), 0, []
            stream = tar.extractfile(member)
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
                size += len(chunk)
                if keep:
                    chunks.append(chunk)
            if size != member.size:
                raise ValueError('Truncated archive member')
            if member.name == 'ARCHIVE_MANIFEST.json':
                if manifest is not None:
                    raise ValueError('Duplicate manifest')
                manifest = json.loads(b''.join(chunks))
            else:
                observed[member.name] = {'sha256': h.hexdigest(), 'size': size}
                if keep:
                    payloads[member.name] = json.loads(b''.join(chunks))
    if manifest is None or manifest['files'] != observed:
        raise ValueError('Archive files differ from per-file manifest')
    return manifest, observed, payloads


def verify(archive, expected_sha, restore_name='af_dump_original_restore_20260912'):
    archive = archive.resolve()
    receipt = json.loads(archive.with_suffix('.receipt.json').read_text())
    if receipt['sha256'] != expected_sha or archive.stat().st_size != receipt['size'] or sha(archive) != expected_sha:
        raise ValueError('Source/destination final archive mismatch')
    manifest, files, payloads = read_members(archive)
    if manifest.get('terminal_run') is not True:
        raise ValueError('Not a terminal final archive')
    root = restore_name + '/'
    models = root + 'original_models_v3_rim20/'
    inference = root + 'original_inference_v3_rim20/'
    observer = 'af_runtime_observer_20260913/'
    required = {
        root + 'original_rootlocal_dataset_v3_rim20/COLLECTION_COMPLETE.json': 'completed',
        models + 'TRAINING_COMPLETE.json': 'completed',
        inference + 'FINAL_INFERENCE_RESULTS.json': 'completed',
        inference + 'INDEPENDENT_FINAL_RESULT_AUDIT.json': 'passed',
        root + 'original_pipeline_v3_rim20/RIM20_PIPELINE_AUDITED.json': 'completed',
        root + 'original_rim20_v3_driver/COMPLETE.json': 'completed',
        observer + 'ANVIL_FULL_DATASET_AUDIT.json': 'passed',
    }
    for name, key in required.items():
        if payloads[name].get(key) is not True:
            raise ValueError('Missing completion gate: ' + name)
    payloads[observer + 'driver736766_capture/OBSERVER_EXIT.json']
    for name in ['stdout.log', 'stderr.log']:
        files[observer + 'driver736766_capture/' + name]
    collection = payloads[root + 'original_rootlocal_dataset_v3_rim20/COLLECTION_COMPLETE.json']
    if collection['total_rows'] != 128 or len(collection['contexts']) != 16:
        raise ValueError('Incomplete collection metadata')
    training = payloads[models + 'TRAINING_COMPLETE.json']
    if training['test_groups_executed'] != 0:
        raise ValueError('Invalid training-before-TEST lock')
    if files[models + 'TRAINING_PROTOCOL.json']['sha256'] != training['protocol_sha256']:
        raise ValueError('Training protocol mismatch')
    checkpoints = 0
    for part, name in [('belief', 'BELIEF'), ('feasibility', 'FEASIBILITY')]:
        if files[models + part + '/' + name + '_MANIFEST.json']['sha256'] != training[part + '_manifest_sha256']:
            raise ValueError('Model manifest mismatch')
        lock = payloads[models + part + '/CHECKPOINT_SELECTION_LOCK.json']
        if len(lock['checkpoints']) != 3:
            raise ValueError('Missing original model ensemble members')
        for checkpoint in lock['checkpoints']:
            member = models + part + '/' + PurePosixPath(checkpoint['path']).name
            if files[member]['sha256'] != checkpoint['sha256']:
                raise ValueError('Checkpoint bytes differ from lock')
            checkpoints += 1
    final = payloads[inference + 'FINAL_INFERENCE_RESULTS.json']
    audit = payloads[inference + 'INDEPENDENT_FINAL_RESULT_AUDIT.json']
    if final['independent_query_settings'] != 4 or final['paired_rollouts'] != 24 or len(final['cases']) != 4:
        raise ValueError('Incorrect TEST denominator')
    if audit['paired_rollouts_audited'] != 24 or audit['final_result_sha256'] != files[inference + 'FINAL_INFERENCE_RESULTS.json']['sha256']:
        raise ValueError('Final independent audit does not cover results')
    if audit['audit_source_sha256'] != files[root + 'audit_original_inference_results.py']['sha256']:
        raise ValueError('Final audit source missing or changed')
    source_restore = PurePosixPath(final['cases'][0]['context']['collection_protocol_path']).parent.parent
    for source, expected in audit['raw_audit_hashes'].items():
        member = root + str(PurePosixPath(source).relative_to(source_restore))
        if files[member]['sha256'] != expected:
            raise ValueError('Raw audit reference missing or changed')
    methods = ['ActiveForcing', 'Fixed-1N', 'Fixed-3N', 'Fixed-5N', 'Fixed-6N', 'Fixed-8N']
    branches = 0
    for case in final['cases']:
        context = case['context']
        if context['root'] != 200002 or context['task'] != 'dump_bin_bigbin':
            raise ValueError('Unexpected TEST scope')
        if [r['method'] for r in case['outcomes']] != methods:
            raise ValueError('Missing paired method')
        job = inference + context['id'] + '/job/'
        for name in ['qualification.json', 'original_raw_rows.json', 'patch_readbacks.json']:
            files[job + 'query/' + name]
        for i, row in enumerate(case['outcomes']):
            branch = job + f"branch_{i}_{row['force_N']:g}N/"
            if files[branch + 'result.json']['sha256'] != row['result_sha256']:
                raise ValueError('Raw result not preserved')
            for name in ['physics_trace.jsonl.gz', 'arbitration.json', 'policy_receipts.json',
                         'chunk_000.npy', 'original_motion_feature.json', 'INDEPENDENT_BRANCH_AUDIT.json']:
                files[branch + name]
            branches += 1
    prior = payloads[observer + 'ANVIL_FULL_DATASET_AUDIT.json']
    if prior['terminal_labels'] != 128 or len(prior['groups']) != 16:
        raise ValueError('Incomplete raw dataset backup audit')
    linked = manifest['raw_groups_separately_archived']
    if len(linked) != 16 or {x['context'] for x in linked} != {x['context'] for x in prior['groups']}:
        raise ValueError('Incomplete linked raw dataset archives')
    for entry in linked:
        expected = entry['receipt']
        raw = archive.parent / PurePosixPath(expected['archive']).name
        previous = next(x for x in prior['groups'] if x['context'] == entry['context'])
        if expected['sha256'] != previous['sha256'] or raw.stat().st_size != expected['size'] or sha(raw) != expected['sha256']:
            raise ValueError('Linked raw archive missing or changed')
    result = {'passed': True, 'archive': archive.name, 'archive_sha256': expected_sha,
              'archived_files_verified': len(files), 'model_checkpoints_verified': checkpoints,
              'paired_rollouts_verified': branches, 'raw_collection_labels_preserved': 128,
              'linked_raw_archives_verified': 16, 'final_result_sha256': audit['final_result_sha256'],
              'verifier_sha256': sha(Path(__file__)),
              'scope': 'complete artifact preservation; native execution on Anvil not claimed',
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('expected_sha256')
    args = parser.parse_args()
    verify(args.archive, args.expected_sha256)
