"""Independent read-only join of real server receipts and executed arm actions."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from common import read, sha, array_sha, payload_sha, INSTRUCTIONS, TABERO

def audit(job):
    job=Path(job);result=read(job/'BRANCH_RESULT.json');traces=read(job/'BRANCH_TRACE.json')
    meta=read(job/'VLA_SERVER_METADATA.json')
    server_records={}
    for line in (Path(meta['server_log_dir'])/'INFERENCE.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['request_id'] in server_records:raise RuntimeError('Repeated server request id')
        server_records[r['request_id']]=r
    errors=[];latencies=[];chunks={};used=set()
    final_manifest=job.parent.parent/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json'
    if final_manifest.exists():
        frozen=read(final_manifest)
        expected_server=read(frozen['POLICY_SERVER_VERSION']['readiness_evidence'])
        if meta!=expected_server:
            errors.append('actual server metadata/process differs from final frozen instance')
    adapter=read(job/'VLA_OBSERVATION_ADAPTER.json')
    if adapter['source_sha256']!=sha(TABERO/'analysis/p6g1_primitive_ik_vla_grasp_realization.py'):
        errors.append('historical observation preprocessing source changed')
    process=read(job/'PROCESS_EXIT.json')
    if process['exit_code']!=0 or not (process.get('completion') or {}).get('logical_success',False):
        errors.append('worker process did not exit successfully with complete evidence')
    command=read(job/'PROCESS.json')['command']
    origin=Path(command[command.index('--job')+1]).resolve()
    expected_worker=origin.name+':'+hashlib.sha256(str(origin).encode()).hexdigest()[:12]
    plan_identity=result['plan']
    expected_context=f"t{plan_identity['task']}_r{plan_identity['root']}_{plan_identity['band'].lower()}"
    if plan_identity['id']!=expected_context:errors.append('result context ID does not match task/root/band')
    decision=read(job/'PLANNER_DECISION.json');expected_force=float(decision['executed_force_N'])
    if result['force']!=expected_force:errors.append('result force does not match planner execution')
    if not origin.name.endswith('__'+decision['method']):errors.append('method does not match original worker identity')
    if decision['method']=='ACTIVEFORCING' and decision['selected_force_N']!=expected_force:
        errors.append('AF executed force differs from its selection')
    if decision['method'].startswith('FIXED_') and float(decision['method'][6:])!=expected_force:
        errors.append('fixed baseline force changed')
    for n,t in enumerate(traces,1):
        if n!=t['branch_step']:errors.append('step sequence')
        request_step=1+((n-1)//10)*10;index=(n-1)%10
        if request_step not in chunks:
            p=job/'RPC'/f'{request_step:04d}.json';receipt=read(p);data=np.load(p.with_suffix('.npz'))
            server=server_records.get(receipt['request_id'])
            if not server or any(receipt[k]!=v for k,v in server.items()):errors.append('missing/mismatched real server receipt')
            if receipt['checkpoint_sha256']!=meta['checkpoint_sha256']:errors.append('checkpoint mismatch')
            if not receipt['model_inference_called'] or receipt['downstream_action_source']!='ONLINE_VLA':errors.append('not online')
            if receipt['context_id']!=expected_context or receipt['worker_id']!=expected_worker or receipt['step']!=request_step:
                errors.append('inference context/worker/step does not match this branch')
            if receipt['instruction']!=INSTRUCTIONS[plan_identity['task']]:errors.append('wrong task instruction')
            if not receipt['client_start_ns']<=receipt['server_start_ns']<=receipt['server_end_ns']<=receipt['client_received_ns']:
                errors.append('inference timestamp is outside rollout request')
            if sha(p.with_suffix('.npz'))!=receipt['artifact_sha256']:errors.append('RPC artifact hash')
            raw=data['raw_vla_action'];post=data['postprocessed_vla_action']
            if array_sha(raw)!=receipt['raw_model_action_sha256'] or array_sha(post)!=receipt['action_sha256']:errors.append('action hash')
            payload={k.removeprefix('payload__'):data[k] for k in data.files if k.startswith('payload__')}
            payload['prompt']=receipt['instruction']
            if payload_sha(payload)!=receipt['observation_sha256']:errors.append('observation hash')
            chunks[request_step]=(receipt,post);latencies.append(receipt['latency_ms'])
        receipt,post=chunks[request_step];used.add(receipt['request_id'])
        if 'action_submit_ns' in t and not receipt['client_received_ns']<=t['action_submit_ns']<=t['action_completed_ns']:
            errors.append('action executed before current online inference response')
        if t['request_id']!=receipt['request_id'] or t['chunk_index']!=index:errors.append('action routing')
        action=np.asarray(t['action'],np.float32);policy=post[index].astype(np.float32)
        if not np.array_equal(action[:6],policy[:6]):errors.append('arm source not VLA')
        if not np.array_equal(np.asarray(t['raw_vla_arm_command'],np.float32),policy[:6]) or np.float32(t['raw_vla_gripper_command'])!=policy[6]:
            errors.append('raw VLA command telemetry differs from actual inference')
        if t['selected_force_setpoint']!=expected_force:errors.append('per-step selected force mismatch')
        if t['active_force_setpoint']!=(0. if t['vla_release_intent'] else expected_force):
            errors.append('active squeeze force is not AF/fixed setpoint outside release')
        f=t['active_force_setpoint']
        if not np.array_equal(action[7:13],np.array([0,0,f/2,0,0,f/2],np.float32)):errors.append('force override')
        if sha(t['raw_observation_path'])!=t['raw_observation_sha256']:errors.append('per-step observation missing/corrupt')
    sequence=np.load(job/'PREACTION_SEQUENCE.npy')
    if decision.get('feasibility_input_shape')==[8,64] and sequence.shape!=(8,64):
        errors.append('phase-free feature schema mismatch')
    plan=chunks[1][1][:8,:3].astype(np.float32).astype(np.float64)
    if not np.array_equal(sequence[:,:3],(plan-plan[0]).astype(np.float32)):errors.append('feasibility displacement not online chunk')
    if not np.array_equal(sequence[:,3:6],np.vstack([np.zeros((1,3)),np.diff(plan,axis=0)]).astype(np.float32)):errors.append('feasibility increments not online chunk')
    if read(job/'SOURCE_HASHES_BEFORE.json')!=read(job/'SOURCE_HASHES_AFTER.json'):errors.append('source changed during rollout')
    if not read(job/'POSTPROBE_EQUALITY.json')['passed']:errors.append('state equality')
    return {'passed':not errors,'errors':errors,'steps':len(traces),'real_online_requests':len(used),
        'latency_median_ms':float(np.median(latencies)),'latency_max_ms':float(max(latencies)),
        'outcome':result['outcome'],'checkpoint_sha256':meta['checkpoint_sha256'],
        'downstream_action_source':'ONLINE_VLA' if not errors else 'QUARANTINED',
        'arm_action_verified':not errors,'same_probe_reference_verified':not errors}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('job');args=a.parse_args()
    print(json.dumps(audit(args.job),indent=2))
