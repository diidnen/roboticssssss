"""No-physics server equivalence check on all archived dev observations.

This is explicitly archived-observation re-inference, never a new rollout or
online manipulation result. No output from this check is fed to a simulator.
"""
import argparse
from pathlib import Path
import numpy as np
from common import read,write,sha,array_sha,payload_sha
from runtime import OnlineClient

def run(qualification,out,ready_path,port,numerical_diagnostic=False):
    ready=read(ready_path)
    for path,digest in ready['source_hashes'].items():
        if sha(path)!=digest:raise RuntimeError('Server source snapshot changed')
    source_jobs=sorted((qualification/'branches').glob('*/PROCESS_EXIT.json'))
    if len(source_jobs)!=40:raise RuntimeError('Require complete40-branch qualified observation corpus')
    out.mkdir(exist_ok=False)
    write(out/'PROTOCOL.json',dict(role='ARCHIVED_DEV_OBSERVATION_REINFERENCE_ONLY',simulator_steps=0,
        new_full_task_labels=0,final_roots_used=False,policy_parameters_changed=False,
        source_qualification=str(qualification),server_ready=str(ready_path),server_ready_sha256=sha(ready_path),
        required_requests=1400,all_requests_required=True,implementation_sha256=sha(__file__),
        cross_process_exactness_required=not numerical_diagnostic,
        diagnostic_only=numerical_diagnostic,does_not_admit_final_physics=True))
    rows=[]
    for process in source_jobs:
        original=process.parent;exit_info=read(process)
        if exit_info['exit_code']!=0 or not exit_info['completion']['logical_success']:
            raise RuntimeError('Unqualified source job')
        plan=read(original/'BRANCH_RESULT.json')['plan']
        job=out/original.name;job.mkdir()
        client=OnlineClient(job,plan,port)
        if client.metadata!=ready:raise RuntimeError('Unexpected policy server')
        requests=sorted((original/'RPC').glob('*.json'))
        for record in requests:
            old=read(record);npz=record.with_suffix('.npz')
            if sha(npz)!=old['artifact_sha256']:raise RuntimeError('Original observation evidence changed')
            with np.load(npz) as a:
                payload={k.removeprefix('payload__'):np.array(a[k],copy=True) for k in a.files if k.startswith('payload__')}
                old_action=np.array(a['postprocessed_vla_action'],copy=True)
            payload['prompt']=old['instruction']
            if payload_sha(payload)!=old['observation_sha256']:raise RuntimeError('Source observation mismatch')
            _,new=client.infer(payload,old['step'],{},observation_origin='ARCHIVED_DEV_OBSERVATION_REINFERENCE_CHECK_NO_PHYSICS')
            keys=('observation_sha256','noise_seed','noise_sha256','checkpoint_sha256',
                  'raw_model_action_sha256','action_sha256','transformed_observation_sha256')
            equal={k:old[k]==new[k] for k in keys}
            row=dict(source_request=str(record),source_request_sha256=sha(record),
                     new_request=str(job/'RPC'/record.name),equal=equal,passed=all(equal.values()))
            with np.load(job/'RPC'/record.with_suffix('.npz').name) as a:
                new_action=np.array(a['postprocessed_vla_action'],copy=True)
            delta=np.abs(new_action-old_action)
            row.update(first10_max_abs_delta_by_action_dimension=delta[:10].max(axis=0).tolist(),
                full50_max_abs_delta_by_action_dimension=delta.max(axis=0).tolist(),
                first10_canonical_open_intent_changes=int(np.count_nonzero((new_action[:10,6]>=.039)!=(old_action[:10,6]>=.039))))
            required=('observation_sha256','noise_seed','noise_sha256','checkpoint_sha256','transformed_observation_sha256')
            if not all(equal[k] for k in required):raise RuntimeError('Policy input/noise/checkpoint changed')
            write(job/(record.stem+'_PARITY.json'),row);rows.append(row)
            if not row['passed'] and not numerical_diagnostic:raise RuntimeError('Server inference differs from qualification; final physics forbidden')
        print('SERVER_PARITY_CONTEXT_METHOD_DONE',original.name,len(rows),flush=True)
        client.client._ws.close()
    if len(rows)!=1400:raise RuntimeError('Incomplete archived inference parity')
    for path,digest in ready['source_hashes'].items():
        if sha(path)!=digest:raise RuntimeError('Server source changed during equivalence check')
    report=dict(role='COMPLETE_SERVER_REQUALIFICATION_NO_PHYSICS',requests=1400,branches=40,
        all_equal=all(r['passed'] for r in rows),raw_and_postprocessed_action_hashes_exact=all(r['passed'] for r in rows),source_qualification=str(qualification),
        server_ready=str(ready_path),server_ready_sha256=sha(ready_path),checkpoint_sha256=ready['checkpoint_sha256'],
        new_server_pid=ready['pid'],source_hashes=ready['source_hashes'],rows=rows,
        simulator_steps=0,new_full_task_labels=0,final_roots_used=False,
        input_noise_checkpoint_preprocessing_parity=True,diagnostic_only=numerical_diagnostic,
        first10_max_abs_delta_by_action_dimension=np.max([r['first10_max_abs_delta_by_action_dimension'] for r in rows],axis=0).tolist(),
        first10_canonical_open_intent_changes=sum(r['first10_canonical_open_intent_changes'] for r in rows),
        interpretation='Archived-observation re-inference numerical diagnostic. Does not itself admit final physics or create full-task labels. Cross-process differences, if any, must be reviewed without pretending byte-exact agreement.')
    write(out/'SERVER_REQUALIFICATION_COMPLETE.json',report)
    print('SERVER_REQUALIFICATION_DIAGNOSTIC_COMPLETE',len(rows),report['all_equal'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--qualification',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--server-ready',type=Path,required=True);p.add_argument('--port',type=int,default=18885)
    p.add_argument('--numerical-diagnostic',action='store_true')
    a=p.parse_args();run(a.qualification,a.out,a.server_ready,a.port,a.numerical_diagnostic)
