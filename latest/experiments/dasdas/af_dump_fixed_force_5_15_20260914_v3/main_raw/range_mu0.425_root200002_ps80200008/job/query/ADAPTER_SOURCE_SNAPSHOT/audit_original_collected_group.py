"""Independent raw-evidence audit, not a replay of a collector's pass boolean."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from rootlocal_collection_contract import read,write,sha,verify_runtime
from native_original_evidence_adapter import measured_patch_record


def same_or_write(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError('Existing audit no longer matches raw evidence: '+str(path))
    else:write(path,value)


def audit_query(query):
    query=Path(query);raw=read(query/'original_raw_rows.json');patches=read(query/'patch_readbacks.json')
    if len(raw)!=len(patches):raise ValueError('Query row alignment failed')
    max_error=0.;raw_sources={}
    for row,patch in zip(raw,patches):
        capture_path=query/f"query_{row['step']:04d}.json"
        capture=read(capture_path);observed=capture['contacts']
        reconstructed=measured_patch_record(observed,patch['prestep_com_velocities_world'],
            patch['measurement_provenance']['physics_dt_s'],int(row['step']))
        if reconstructed['patches']!=patch['patches']:
            raise ValueError('Saved true-friction/normal record differs from actual native-cache force balance')
        if reconstructed['finger_joints']!=patch['finger_joints'] or reconstructed['aperture_m']!=patch['aperture_m']:
            raise ValueError('Actual finger state differs from saved conditioning')
        raw_sources[str(capture_path)]=sha(capture_path)
    for p in [query/'patch_readbacks.json',query/'original_raw_rows.json']:
        raw_sources[str(p)]=sha(p)
    result={'passed':True,'query_rows':len(raw),'all_native_cache_contact_balance_rebuilds_exact':True,
            'actual_finger_joint_readback_exact':True,'source_hashes':raw_sources,
            'audit_source_sha256':sha(__file__)}
    same_or_write(query/'INDEPENDENT_NATIVE_FORCE_REBUILD.json',result)
    return result


def audit_branch(branch):
    branch=Path(branch);result=read(branch/'result.json');actions=read(branch/'arbitration.json')
    if not result['completed'] or result['success']!=result['official_final_check']:raise ValueError('Ambiguous outcome')
    if result['native_actions']!=len(actions):raise ValueError('Action count mismatch')
    if not result['success'] and result['native_actions']!=result['native_horizon']:
        raise ValueError('Early failure shortcut, not full-task horizon')
    if result['old_irrecoverable_shortcut_used']:raise ValueError('Forbidden old failure predicate')
    chunks=sorted(branch.glob('chunk_*.npy'));policy=read(branch/'policy_receipts.json')
    if len(chunks)!=result['chunks'] or len(chunks)!=len(policy):raise ValueError('Missing native chunks')
    sequences=[]
    for path,receipt in zip(chunks,policy):
        array=np.load(path,allow_pickle=False)
        if array.ndim!=2 or array.shape[1]!=14 or not np.isfinite(array).all():raise ValueError('Invalid native action array')
        if hashlib.sha256(array.tobytes()).hexdigest()!=receipt['actions_sha256']:raise ValueError('Chunk changed')
        sequences.append(array)
    executed=np.concatenate(sequences)[:len(actions)]
    np.testing.assert_array_equal(executed,np.asarray([a['native_action14'] for a in actions],np.float32))
    force=result['force_setpoint_bilateral_n']
    for action in actions:
        if action['raw_vla_arm_command']!=action['final_arm_command']:raise ValueError('VLA arm command changed')
        if action['selected_force_setpoint']!=force:raise ValueError('Candidate force changed mid-rollout')
        if action['active_force_setpoint']!=(0. if action['vla_release_intent'] else force):raise ValueError('Release arbitration mismatch')
        if action['vla_gripper_override_after_af_handoff']:raise ValueError('VLA overwrote AF gripper command')
    if sum(a['vla_release_intent'] for a in actions)!=result['release_actions']:raise ValueError('Release count mismatch')
    digest=hashlib.sha256();forces=[];previous=None;worst_force=0.;worst_filter=0.
    query=branch.parent/'query'
    ema=np.float32(read(query/'original_squeeze_inner_trace.json')[-1]['measured_filtered_N'])
    count=0
    with gzip.open(branch/'physics_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for count,line in enumerate(stream,1):
            encoded=line.rstrip('\n');digest.update(encoded.encode());row=json.loads(encoded)
            if row['physics_step']!=count:raise ValueError('Noncontiguous physical steps')
            contact=row['contact'];axis=np.asarray(contact['normal_axis_world'],float)
            projected=[]
            for finger in contact['fingers']:
                # Independently sum raw contact impulses in recorded native
                # body ordering; no reported aggregate is trusted here.
                total=np.zeros(3)
                for point in finger['points']:
                    sign=1 if point['finger_body_index']==0 else -1
                    total+=sign*np.asarray(point['impulse_ns'],float)/.004
                error=float(np.max(abs(total-np.asarray(finger['force_world_n']))))
                worst_force=max(worst_force,error)
                projected.append(float(np.dot(total,axis)))
            measured=2*min(map(abs,projected))
            worst_force=max(worst_force,abs(measured-contact['measured_squeeze_n']))
            if worst_force>1e-8:raise ValueError('Measured squeeze not reproducible from actual raw impulses')
            inner=row['original_squeeze_inner']
            if inner is None:raise ValueError('Missing original inner-force controller trace')
            if previous is not None and inner['measured_raw_N']!=previous:
                raise ValueError('Inner feedback not sourced from previous native physical contact')
            ema=np.float32(np.float32(ema)*np.float32(.8)+np.float32(inner['measured_raw_N'])*np.float32(.2))
            worst_filter=max(worst_filter,abs(float(ema)-inner['measured_filtered_N']))
            if worst_filter>1e-5:raise ValueError('Original EMA update drift')
            if inner['force_reference_N'] not in (0.,force):raise ValueError('Unplanned inner-force reference')
            previous=contact['measured_squeeze_n'];forces.append(previous)
    if not count or count!=result['physics_steps'] or digest.hexdigest()!=result['trace_sha256']:
        raise ValueError('Raw trace count/hash mismatch')
    if float(np.mean(forces))!=result['measured_mean_squeeze_n'] or float(np.max(forces))!=result['measured_max_squeeze_n']:
        raise ValueError('Reported measured-force statistics differ from raw trace')
    files=[branch/'result.json',branch/'arbitration.json',branch/'physics_trace.jsonl.gz',
           branch/'policy_receipts.json',branch/'original_motion_feature.json',*chunks]
    report={'passed':True,'physics_steps':count,'actions':len(actions),
            'max_raw_impulse_force_error_N':worst_force,'max_original_EMA_error_N':worst_filter,
            'raw_actions_and_arm_pass_through_exact':True,'raw_trace_hash_recomputed':digest.hexdigest(),
            'official_full_horizon_label_consistent':True,
            'source_hashes':{str(p):sha(p) for p in files},'audit_source_sha256':sha(__file__)}
    same_or_write(branch/'INDEPENDENT_BRANCH_AUDIT.json',report)
    return report


def audit_group(job,allow_partial=False):
    job=Path(job);query=audit_query(job/'query')
    branches=sorted(p.parent for p in job.glob('branch_*/result.json'))
    results=[audit_branch(p) for p in branches]
    summary=read(job/'online_qualification.json');context=read(job/'CONTEXT_LOCK.json')
    verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
    if not allow_partial:
        if len(branches)!=8 or len(summary['first_chunk_hashes'])!=8:raise ValueError('Incomplete group')
        if len(set(summary['first_chunk_hashes']))!=1:raise ValueError('Unpaired first policy chunk')
        if len({sha(p/'original_motion_feature.json') for p in branches})!=1:raise ValueError('Candidate feature mismatch')
        report={'passed':True,'context_id':context['id'],'branches':len(results),
                'query_rows':query['query_rows'],'branch_audit_hashes':{str(p):sha(p/'INDEPENDENT_BRANCH_AUDIT.json') for p in branches},
                'query_audit_sha256':sha(job/'query/INDEPENDENT_NATIVE_FORCE_REBUILD.json'),
                'runtime_manifest_sha256':context['runtime_manifest_sha256']}
        same_or_write(job/'INDEPENDENT_GROUP_AUDIT.json',report)
    print(json.dumps({'group':str(job),'completed_branches_audited':len(results),
                      'query_native_force_rows_rebuilt':query['query_rows'],'complete_group_required':not allow_partial}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('job',type=Path);ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args();audit_group(args.job,args.allow_partial)
