"""Paired full-horizon engineering gate; cannot start before all query checks."""
import os
import subprocess
from rootlocal_collection_contract import HERE,BASE,REPO,read,write,sha,now,verify_runtime


def prerequisites():
    qualification=HERE/'rim20_trainval_query_qualification_v1'
    if not read(qualification/'COMPLETE.json')['completed']:raise ValueError('Incomplete query qualification')
    audit=read(qualification/'INDEPENDENT_FULL_QUALIFICATION.json')
    if not audit['all_original_admissions_passed'] or len(audit['cases'])!=17:
        raise ValueError('Missing full original query admissions')
    if audit['formal_labels_added']!=0 or audit['TEST_used']:raise ValueError('Scope changed')
    replay=read(qualification/'REPLAY_AUDIT.json')
    if not replay['raw_rows_byte_equal'] or replay['mismatched_steps']:raise ValueError('Query replay mismatch')
    for source,digest in read(qualification/'PROTOCOL.json')['source_hashes'].items():
        if sha(source)!=digest:raise ValueError('Qualified implementation changed')
    return qualification


def main():
    qualification=prerequisites()  # Before any new directory, job or GPU use.
    dataset=HERE/'original_rootlocal_dataset_v2_canonical'
    lock=read(dataset/'FREEZE_LOCK.json')
    verify_runtime(dataset/'RUNTIME_MANIFEST.json',lock['runtime_sha256'])
    out=HERE/'rim20_online_gate_v1';out.mkdir(exist_ok=False)
    sources={str(HERE/name):sha(HERE/name) for name in ['qualify_rim20_online.py','run_rim20_online_gate.py',
        'diagnose_canonical_rim_alignment.py','canonical_open_ready_query.py','trace_original_query_contact.py']}
    sources.update(read(qualification/'PROTOCOL.json')['source_hashes'])
    evidence={str(qualification/name):sha(qualification/name) for name in
              ['COMPLETE.json','PROTOCOL.json','INDEPENDENT_FULL_QUALIFICATION.json','REPLAY_AUDIT.json']}
    write(out/'PROTOCOL.json',{'root':200002,'friction':.7,'policy_seed':30200002,'offset_m':.02,
        'forces_N':[3.,3.,8.],'source_hashes':sources,'query_qualification_hashes':evidence,
        'formal_data_admission':False,'success_not_required_for_valid_negative':True,
        'requires_all_terminal_and_duplicate_full_trace_exact':True,'started_utc':now()})
    env=os.environ.copy();env.pop('AF_COLLECTION_CONTEXT',None);env.pop('AF_INFERENCE_CONTEXT',None)
    env.update(AF_P4_PHYSICAL_SURFACE_CAMERA='1',AF_ORIGINAL_SQUEEZE_INNER='1',
        AF_ENGINEERING_FORCES='3,3,8',AF_DIAGNOSTIC_RIM_OFFSET_M='.02',PYTHONUTF8='1',
        PATH=str(BASE/'runtime_bin')+os.pathsep+env.get('PATH','/usr/bin:/bin'))
    command=[str(BASE/'venv_robotwin/bin/python3'),str(HERE/'qualify_rim20_online.py'),
             '--repo',str(REPO),'--out',str(out/'job'),'--native-ft','--friction','.7','--policy-seed','30200002']
    write(out/'PROCESS.json',{'command':command,'cwd':str(REPO),'started_utc':now()})
    with (out/'worker.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/'PID.json',{'pid':process.pid});code=process.wait()
    write(out/'EXIT.json',{'exit_code':code,'finished_utc':now()})
    if code:raise RuntimeError('Gate execution failed; retained records')
    verify_runtime(dataset/'RUNTIME_MANIFEST.json',lock['runtime_sha256'])
    if any(sha(path)!=digest for path,digest in {**sources,**evidence}.items()):raise ValueError('Source/evidence drift')
    result=read(out/'job/online_qualification.json')
    if len(result['results'])!=3 or any(row['kind']!='done' for row in result['results']):raise ValueError('Incomplete gate')
    if len(result['duplicate_force_replays'])!=1 or not result['duplicate_force_replays'][0]['full_trace_exact']:
        raise ValueError('Same-force paired trace mismatch')
    write(out/'COMPLETE.json',{'completed':True,'paired_duplicate_full_trace_exact':True,
        'formal_labels_added':0,'independent_raw_audit_still_required':True,'finished_utc':now()})


if __name__=='__main__':main()
