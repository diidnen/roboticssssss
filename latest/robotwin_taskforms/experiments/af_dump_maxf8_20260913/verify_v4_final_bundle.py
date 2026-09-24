"""Read-only complete V4 archive graph and original reused-data preservation audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import sys

OVERLAY = 'af_dump_maxf8_final_evidence_20260913T091728Z.tar.gz'
OVERLAY_SHA = '6ca676f022e70de48297e6a8ec20e22486e4ab738665fa8db2e943d8b7742d90'
OLD_FINAL = 'af_dump_rim20_FINAL_COMPLETE_20260913T031900Z.tar.gz'
OLD_SHA = '7a4e0021260ad8eb62793d944a57809339f14b4b873be1202961ed33380fc404'
ROOT = 'af_dump_maxf8_20260913/'
INF = 'confirmation/inference_v4/'

def main(base):
    sys.path.insert(0,str(base.parent/'runtime_observer_20260913'))
    from verify_completed_rim20_archive import read_members, sha, verify
    overlay = base/OVERLAY
    if sha(overlay) != OVERLAY_SHA:
        raise ValueError('Final evidence overlay changed')
    manifest, overlay_files, payloads = read_members(overlay)
    links = manifest['linked_archives']
    if len(links) != 33 or len({Path(r['archive']).name for r in links}) != 33:
        raise ValueError('Incomplete unique archive graph')
    files = {}
    counts = {'collection':0,'confirmation-case':0,'models':0}
    total_files = 0
    receipts = []
    for expected in links:
        name = PurePosixPath(expected['archive']).name
        archive = base/name
        if archive.is_symlink() or archive.resolve().parent != base.resolve():
            raise ValueError('Unsafe linked archive')
        receipt = json.loads(archive.with_suffix('.receipt.json').read_text())
        if receipt != expected or archive.stat().st_size != expected['size'] or sha(archive) != expected['sha256']:
            raise ValueError('Linked source/destination identity mismatch: '+name)
        linked_manifest, members, linked_payloads = read_members(archive)
        if len(members) != expected['files'] or linked_manifest['case'] != expected['case']:
            raise ValueError('Linked member coverage mismatch')
        stage = expected.get('stage','collection')
        counts[stage] += 1
        if stage == 'collection':
            group = ROOT+'additional_data_v1/groups/'+expected['case']+'/attempt_001/'
            for suffix in ['ACCEPTED.json','PROCESS_EXIT.json','job/query/original58_engineering.npy','job/query/original_raw_rows.json']:
                if group+suffix not in members:
                    raise ValueError('Missing collected evidence')
        files.update(members)
        total_files += len(members)
        receipts.append(dict(archive=name,sha256=expected['sha256'],files_verified=len(members)))
        print('VERIFIED_LINK '+name,flush=True)
    if counts != {'collection':24,'confirmation-case':8,'models':1}:
        raise ValueError('Missing required archive type')
    files.update(overlay_files)
    final = payloads[INF+'FINAL_INFERENCE_RESULTS.json']
    audit = payloads[INF+'INDEPENDENT_FINAL_RESULT_AUDIT.json']
    complete = payloads[ROOT+'confirmation_driver_v1/COMPLETE.json']
    if not final['completed'] or not audit['passed'] or not complete['completed']:
        raise ValueError('Missing scientific completion gate')
    if final['paired_rollouts'] != 48 or final['decision_contexts'] != 8 or audit['paired_rollouts_audited'] != 48:
        raise ValueError('Incorrect validation scope')
    if audit['final_result_sha256'] != files[INF+'FINAL_INFERENCE_RESULTS.json']['sha256'] or complete['final_audit_sha256'] != files[INF+'INDEPENDENT_FINAL_RESULT_AUDIT.json']['sha256']:
        raise ValueError('Final result/audit digest mismatch')
    if audit['audit_source_sha256'] != files[ROOT+'audit_v4_inference.py']['sha256']:
        raise ValueError('Executed independent audit source not preserved')
    for original, expected in audit['raw_audit_hashes'].items():
        suffix = original.split('/inference_v4/',1)[1]
        if files[INF+suffix]['sha256'] != expected:
            raise ValueError('Raw audit link missing')
    method_names = ['ActiveForcing','Fixed-1N','Fixed-3N','Fixed-5N','Fixed-6N','Fixed-8N']
    summaries = {name:[] for name in method_names}
    for case in final['cases']:
        context = case['context']
        if context['root'] != 200002 or context['task'] != 'dump_bin_bigbin' or case['utility_normalization_N'] != 8:
            raise ValueError('Wrong scientific scope')
        if [row['method'] for row in case['outcomes']] != method_names:
            raise ValueError('Missing paired method')
        for i,row in enumerate(case['outcomes']):
            branch = INF+context['id']+'/job/branch_%d_%gN/'%(i,row['force_N'])
            if files[branch+'result.json']['sha256'] != row['result_sha256']:
                raise ValueError('Raw outcome missing')
            for name in ['physics_trace.jsonl.gz','arbitration.json','policy_receipts.json','chunk_000.npy','original_motion_feature.json','INDEPENDENT_BRANCH_AUDIT.json']:
                if branch+name not in files:
                    raise ValueError('Missing raw trajectory evidence')
            utility = (8-row['force_N'])/8 if row['success'] else -1
            if abs(utility-row['original_utility']) > 1e-10:
                raise ValueError('Incorrect maxF utility')
            summaries[row['method']].append(row)
    training = payloads[ROOT+'models_v4/TRAINING_COMPLETE.json']
    if not training['completed'] or training['test_groups_executed'] != 0 or training['utility_normalization_N'] != 8:
        raise ValueError('Training gate mismatch')
    weights = 0
    for part in ['belief','feasibility']:
        prefix = ROOT+'models_v4/'+part+'/'
        if files[prefix+part.upper()+'_MANIFEST.json']['sha256'] != training[part+'_manifest_sha256']:
            raise ValueError('Trained manifest mismatch')
        checkpoints = payloads[prefix+'CHECKPOINT_SELECTION_LOCK.json']['checkpoints']
        if len(checkpoints) != 3:
            raise ValueError('Missing ensemble member')
        for checkpoint in checkpoints:
            if files[prefix+PurePosixPath(checkpoint['path']).name]['sha256'] != checkpoint['sha256']:
                raise ValueError('Missing checkpoint bytes')
            weights += 1
    original = verify(base.parent/OLD_FINAL,OLD_SHA)
    result = dict(passed=True,overlay_sha256=OVERLAY_SHA,linked_archives_verified=receipts,
                  archive_counts=counts,linked_members_verified=total_files,overlay_members_verified=len(overlay_files),
                  model_weights_verified=weights,paired_rollouts_verified=48,decision_contexts=8,
                  original_reused_data_audit=original,final_result_sha256=audit['final_result_sha256'],
                  summary=[dict(method=name,successes=sum(r['success'] for r in rows),n=len(rows),
                                mean_command_N=sum(r['force_N'] for r in rows)/len(rows),
                                mean_utility=sum(r['original_utility'] for r in rows)/len(rows)) for name,rows in summaries.items()],
                  native_execution_on_anvil_claimed=False,verifier_sha256=sha(Path(__file__)),
                  completed_utc=datetime.now(timezone.utc).isoformat())
    print('FINAL_BUNDLE_AUDIT '+json.dumps(result),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base',type=Path)
    main(parser.parse_args().base)
