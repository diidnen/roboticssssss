"""Fixed four-case geometry diagnostic; no formal retry or data admission."""
import os
import subprocess
from rootlocal_collection_contract import HERE,BASE,REPO,read,write,sha,now,verify_runtime


def main():
    dataset=HERE/'original_rootlocal_dataset_v2_canonical'
    lock=read(dataset/'FREEZE_LOCK.json')
    failed=dataset/'groups/train_mu0.700_root200002/attempt_001'
    if read(failed/'PROCESS_EXIT.json')['exit_code']!=1:raise ValueError('Expected retained terminal failure')
    out=HERE/'mu070_rim_alignment_diagnostics_v1';out.mkdir(exist_ok=False)
    cases=[('baseline_01',0.),('baseline_02',0.),('shallower_10mm',.01),('shallower_20mm',.02)]
    files=[HERE/n for n in ['MU070_CANONICAL_GEOMETRY_DIAGNOSTIC.md','diagnose_canonical_rim_alignment.py',
           'run_mu070_rim_diagnostics.py','canonical_open_ready_query.py','trace_original_query_contact.py']]
    sources={str(p):sha(p) for p in files}
    write(out/'PROTOCOL.json',{'cases':cases,'root':200002,'friction':.7,'policy_seed':30200002,
        'source_hashes':sources,'frozen_runtime_sha256':lock['runtime_sha256'],'formal_admission':False,
        'query_failures_retained':True,'automatic_adoption_or_retry':False,'started_utc':now()})
    for name,offset in cases:
        verify_runtime(dataset/'RUNTIME_MANIFEST.json',lock['runtime_sha256'])
        if any(sha(p)!=digest for p,digest in sources.items()):raise ValueError('Diagnostic source drift')
        case=out/name;case.mkdir()
        env=os.environ.copy();env.pop('AF_COLLECTION_CONTEXT',None);env.pop('AF_INFERENCE_CONTEXT',None)
        env.update(AF_P4_PHYSICAL_SURFACE_CAMERA='1',AF_ORIGINAL_SQUEEZE_INNER='1',
                   AF_DIAGNOSTIC_RIM_OFFSET_M=str(offset),PYTHONUTF8='1',
                   PATH=str(BASE/'runtime_bin')+os.pathsep+env.get('PATH','/usr/bin:/bin'))
        command=[str(BASE/'venv_robotwin/bin/python3'),str(HERE/'diagnose_canonical_rim_alignment.py'),
                 '--repo',str(REPO),'--out',str(case/'query'),'--native-ft','--friction','.7',
                 '--policy-seed','30200002']
        write(case/'PROCESS.json',{'command':command,'cwd':str(REPO),'offset_m':offset,'started_utc':now()})
        with (case/'worker.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(case/'PID.json',{'pid':process.pid});print('DIAGNOSTIC_START '+name+' '+str(process.pid),flush=True)
            code=process.wait()
        write(case/'EXIT.json',{'exit_code':code,'finished_utc':now()})
        if code:raise RuntimeError('Diagnostic process failure; preserve and inspect '+name)
        result=read(case/'query/original_probe_summary.json')
        print({'case':name,'probe_failure':result['probe_failure'],'diagnostic_only':True},flush=True)
    verify_runtime(dataset/'RUNTIME_MANIFEST.json',lock['runtime_sha256'])
    if any(sha(p)!=digest for p,digest in sources.items()):raise ValueError('Diagnostic source drift')
    write(out/'COMPLETE.json',{'completed':True,'formal_labels_added':0,'independent_audit_required':True,
                             'finished_utc':now()})


if __name__=='__main__':main()
