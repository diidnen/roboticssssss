"""Independent read-only join of real server receipts and executed arm actions."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import read, sha, array_sha, payload_sha

def audit(job):
    job=Path(job);result=read(job/'BRANCH_RESULT.json');traces=read(job/'BRANCH_TRACE.json')
    meta=read(job/'VLA_SERVER_METADATA.json')
    server_records={}
    for line in (Path(meta['server_log_dir'])/'INFERENCE.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['request_id'] in server_records:raise RuntimeError('Repeated server request id')
        server_records[r['request_id']]=r
    errors=[];latencies=[];chunks={};used=set()
    for n,t in enumerate(traces,1):
        if n!=t['branch_step']:errors.append('step sequence')
        request_step=1+((n-1)//10)*10;index=(n-1)%10
        if request_step not in chunks:
            p=job/'RPC'/f'{request_step:04d}.json';receipt=read(p);data=np.load(p.with_suffix('.npz'))
            server=server_records.get(receipt['request_id'])
            if not server or any(receipt[k]!=v for k,v in server.items()):errors.append('missing/mismatched real server receipt')
            if receipt['checkpoint_sha256']!=meta['checkpoint_sha256']:errors.append('checkpoint mismatch')
            if not receipt['model_inference_called'] or receipt['downstream_action_source']!='ONLINE_VLA':errors.append('not online')
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
        f=t['active_force_setpoint']
        if not np.array_equal(action[7:13],np.array([0,0,f/2,0,0,f/2],np.float32)):errors.append('force override')
        if sha(t['raw_observation_path'])!=t['raw_observation_sha256']:errors.append('per-step observation missing/corrupt')
    sequence=np.load(job/'PREACTION_SEQUENCE.npy')
    decision=read(job/'PLANNER_DECISION.json')
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
