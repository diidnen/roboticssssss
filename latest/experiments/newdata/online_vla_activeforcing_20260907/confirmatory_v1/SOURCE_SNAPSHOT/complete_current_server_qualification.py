"""Complete the second, 36-branch dev qualification without rerunning outcomes."""
import argparse
import os
import shutil
from pathlib import Path
from common import HERE, read, write, sha

METHODS = ('FIXED_3', 'FIXED_5', 'ACTIVEFORCING')

def freeze(out, qualification, boundary, release, ready):
    prior = read(boundary/'CANDIDATE_RUNTIME_MANIFEST.json')
    for p, digest in prior['source_hashes'].items():
        if sha(p) != digest:
            raise RuntimeError('Existing qualified source changed: '+p)
    metadata = read(ready)
    sources = [boundary, release]
    reused = {}
    references = {}
    for source in sources:
        done = read(source/'RECHECK_EXECUTION_COMPLETE.json')
        for row in done['rows']:
            job = Path(row['job'])
            if read(job/'VLA_SERVER_METADATA.json') != metadata:
                raise RuntimeError('Reuse requires exact current server instance')
            if job.name in reused:
                raise RuntimeError('Duplicate branch')
            reused[job.name] = job
            references[row['context']] = source/'references'/row['context']
    if len(reused) != 14:
        raise RuntimeError('Expected exactly 14 existing outcomes, including failure')
    out.mkdir(exist_ok=False)
    snap = out/'SOURCE_SNAPSHOT'
    shutil.copytree(boundary/'SOURCE_SNAPSHOT', snap)
    shutil.copy2(HERE/Path(__file__).name, snap/Path(__file__).name)
    shutil.copy2(qualification/'DEV_PLAN.json', out/'DEV_PLAN.json')
    for name in ('branches', 'references', 'initial_chunk_identity'):
        (out/name).mkdir()
    for name, job in reused.items():
        (out/'branches'/name).symlink_to(job, target_is_directory=True)
    for cid, ref in references.items():
        (out/'references'/cid).symlink_to(ref, target_is_directory=True)
        candidates = [j for j in reused.values() if j.name.startswith(cid+'__')]
        identities = [read(j/'INITIAL_ONLINE_CHUNK_IDENTITY.json') for j in candidates]
        if not all(x == identities[0] for x in identities):
            raise RuntimeError('Existing same-X pairing invalid')
        write(out/'initial_chunk_identity'/(cid+'.json'), identities[0])
    plans = read(out/'DEV_PLAN.json')['contexts']
    # Complete the changed context first, then every remaining cell, once each.
    plans = sorted(plans, key=lambda p: (p['id'] != 't5_r5100_mid', p['id']))
    pending = [dict(context=p['id'], method=m) for p in plans for m in METHODS
               if p['id']+'__'+m not in reused]
    if len(pending) != 22:
        raise RuntimeError('Only the missing 22 branches may execute')
    protocol = dict(version='CURRENT_SERVER_COMPLETE36_V1',
        reason='Task5 MID Fixed3 changed from success to valid placement failure after server restart. Complete all cells under current server; do not require byte-identical cross-process outcomes.',
        old_primary_branches=36, current_primary_branches=36,
        reused_current_branches=14, new_branches=22, combined_primary_max=72,
        separate_old_fixed4_boundary_branches=4, total_dev_downstream_max=76,
        final_roots_used=False, no_outcome_retry=True, no_model_or_runtime_tuning=True,
        qualification=str(qualification), server_ready=str(ready), server_ready_sha256=sha(ready),
        reuse={name:str(job) for name,job in reused.items()}, pending=pending,
        changed_failure_retained='t5_r5100_mid__FIXED_3',
        acceptance='Use original complete36 transfer thresholds and all provenance/behavior/probe/feature/force-boundary gates. Outcome parity with old server is diagnostic, not a substitute for current-server qualification.',
        threshold_source=str(HERE/'PRIMARY36_TRANSFER_SCREEN.json'),
        threshold_source_sha256=sha(HERE/'PRIMARY36_TRANSFER_SCREEN.json'))
    write(out/'COMPLETION_PROTOCOL.json', protocol)
    source_hashes={p:d for p,d in prior['source_hashes'].items()
                   if not Path(p).is_relative_to(boundary/'SOURCE_SNAPSHOT')}
    source_hashes.update({str(p):sha(p) for p in snap.glob('*.py')})
    source_hashes[str(out/'COMPLETION_PROTOCOL.json')]=sha(out/'COMPLETION_PROTOCOL.json')
    prior.update(version='CURRENT_SERVER_COMPLETE36_V1', source_hashes=source_hashes,
        selected_contexts=[p['id'] for p in plans], selected_methods=list(METHODS),
        max_additional_branches=22, total_completed_qualification_plus_recheck_max=76,
        reason=protocol['reason'])
    write(out/'CANDIDATE_RUNTIME_MANIFEST.json', prior)

def run(out):
    from launch import launch
    from audit_rollout import audit
    from audit_geometric_label import audit as geometry
    protocol=read(out/'COMPLETION_PROTOCOL.json')
    manifest=read(out/'CANDIDATE_RUNTIME_MANIFEST.json')
    if sha(__file__) != manifest['source_hashes'].get(str(Path(__file__).resolve())):
        raise RuntimeError('Run frozen coordinator only')
    write(out/'COMPLETION_QUEUE_STARTED.json', dict(pid=os.getpid(),new_branches=22))
    plans=read(out/'DEV_PLAN.json')['contexts']
    for item in protocol['pending']:
        cid=item['context'];method=item['method']
        index=next(i for i,p in enumerate(plans) if p['id']==cid)
        launch(out,index,'REFERENCE')
        ref=out/'references'/cid
        old=Path(protocol['qualification'])/'references'/cid
        checks={name:read(ref/name)==read(old/name) for name in ('CONTACT_PATCH_READBACK.json','PREACTION_POSTERIOR.json')}
        checks['RAW_PROBE.csv']=(ref/'RAW_PROBE.csv').read_bytes()==(old/'RAW_PROBE.csv').read_bytes()
        if not all(checks.values()):raise RuntimeError('Common established grasp/probe no longer matches')
        launch(out,index,method)
        job=out/'branches'/(cid+'__'+method)
        p=audit(job);g=geometry(job)
        if not p['passed'] or not g['passed']:raise RuntimeError('Invalid online evidence')
        if read(job/'VLA_SERVER_METADATA.json')!=read(protocol['server_ready']):
            raise RuntimeError('Policy server instance changed')
        print('CURRENT_SERVER_BRANCH_DONE',cid,method,p['outcome']['full_task_success_y'],flush=True)
    write(out/'CURRENT_SERVER36_EXECUTION_COMPLETE.json',dict(branches=36,new_branches=22,
        reused_branches=14,final_roots_used=False,review_required=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--freeze',action='store_true');p.add_argument('--qualification',type=Path)
    p.add_argument('--boundary',type=Path);p.add_argument('--release',type=Path)
    p.add_argument('--server-ready',type=Path);a=p.parse_args()
    if a.freeze:freeze(a.out,a.qualification,a.boundary,a.release,a.server_ready)
    else:run(a.out)
