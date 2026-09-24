"""Single prospectively locked engineering handoff/replay gate; no AF labels."""
import os
import subprocess
from rootlocal_collection_contract import HERE,BASE,REPO,read,write,sha,now,verify_runtime

def main():
    out=HERE/'canonical_ready_online_gate_v1';out.mkdir(exist_ok=False)
    context=read(HERE/'original_rootlocal_dataset_v1/groups/train_mu0.600_root200002/attempt_001/CONTEXT.json')
    evidence=HERE/'mu060_canonical_ready_diagnostics_v1/INITIALIZATION_COMPARISON.json'
    comparison=read(evidence)['comparisons']
    for key in ['actual_qpos_exact','actual_qvel_exact','original_action13_exact',
                'actual_eef_base_exact','both_probe_qualified','both_raw_observation_sequences_identical']:
        if comparison[key] is not True:raise ValueError('Initialization candidate not qualified: '+key)
    files=[HERE/n for n in ['qualify_canonical_ready_online.py','canonical_open_ready_query.py',
                            'trace_original_query_contact.py','run_canonical_ready_online_gate.py']]
    sources={str(p):sha(p) for p in files}
    write(out/'PROTOCOL.json',{'root':200002,'friction':.6,'policy_seed':30200002,'forces_N':[3.,3.,8.],
        'purpose':'preparation-to-original-query-to-frozen-pi0 handoff and paired duplicate3N replay',
        'source_hashes':sources,'initialization_evidence_sha256':sha(evidence),'formal_data_admission':False,
        'fixed_branch_count':3,'success_not_required_to_accept_a_valid_negative_outcome':True,
        'requires_all_terminal_and_duplicate_full_trace_exact':True,'started_utc':now()})
    verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
    env=os.environ.copy();env.pop('AF_COLLECTION_CONTEXT',None);env.pop('AF_INFERENCE_CONTEXT',None)
    env.update(AF_P4_PHYSICAL_SURFACE_CAMERA='1',AF_ORIGINAL_SQUEEZE_INNER='1',
               AF_ENGINEERING_FORCES='3,3,8',PYTHONUTF8='1',
               PATH=str(BASE/'runtime_bin')+os.pathsep+env.get('PATH','/usr/bin:/bin'))
    command=[str(BASE/'venv_robotwin/bin/python3'),str(HERE/'qualify_canonical_ready_online.py'),
             '--repo',str(REPO),'--out',str(out/'job'),'--native-ft','--friction','.6','--policy-seed','30200002']
    write(out/'PROCESS.json',{'command':command,'cwd':str(REPO),'started_utc':now()})
    with (out/'worker.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/'PID.json',{'pid':process.pid});code=process.wait()
    write(out/'EXIT.json',{'exit_code':code,'finished_utc':now()})
    if code:raise RuntimeError('Engineering online gate process failed; retain records')
    verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
    if any(sha(p)!=s for p,s in sources.items()):raise ValueError('Gate source drift')
    result=read(out/'job/online_qualification.json')
    if len(result['results'])!=3 or any(r['kind']!='done' for r in result['results']):raise ValueError('Incomplete gate')
    if len(result['duplicate_force_replays'])!=1 or not result['duplicate_force_replays'][0]['full_trace_exact']:
        raise ValueError('Same-force paired trace not exact')
    write(out/'COMPLETE.json',{'completed':True,'paired_duplicate_full_trace_exact':True,
        'formal_labels_added':0,'independent_raw_audit_still_required':True,'finished_utc':now()})
    print({'engineering_gate_completed':True,'formal_AF_inference':False},flush=True)

if __name__=='__main__':main()
