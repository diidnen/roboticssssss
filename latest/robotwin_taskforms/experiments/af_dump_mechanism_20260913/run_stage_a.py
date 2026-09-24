"""Freeze/start/resume a finite12-branch diagnostic queue, with disk guard."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
OLD=HERE.parent/'af_dump_original_restore_20260912'
V4=HERE.parent/'af_dump_maxf8_20260913'
HISTORY=HERE.parent/'af_dump_maxf8_confirmation_storage_20260913/inference_v4'
sys.path.insert(0,str(OLD))
from rootlocal_collection_contract import read,write,sha,now,verify_runtime
PY=BASE/'venv_robotwin/bin/python3'; REPO=BASE/'RoboTwin'

def freeze():
    path=HERE/'PROTOCOL.json'
    if path.exists():return read(path)
    cases=[]
    for name,forces in [('test_mu0.375_root200002_ps60200002',[5.45,5.45,5.25,5.65,6.,8.]),
                        ('test_mu0.675_root200002_ps50200002',[5.,5.,4.8,5.2,6.,8.])]:
        spec=read(HISTORY/name/'SPEC.json')
        cases.append({'id':name,'forces_N':forces,'methods':['ANCHOR_A','ANCHOR_B','NEIGHBOR_MINUS','NEIGHBOR_PLUS','FIXED6','FIXED8'],
            'historical_spec':str(HISTORY/name/'SPEC.json'),'historical_spec_sha256':sha(HISTORY/name/'SPEC.json'),
            'historical_lock':str(HISTORY/name/'job/PREACTION_SELECTION_LOCK.json'),
            'historical_lock_sha256':sha(HISTORY/name/'job/PREACTION_SELECTION_LOCK.json'),
            'historical_result':str(HISTORY/name/'job/AF_INFERENCE_RESULT.json'),
            'historical_result_sha256':sha(HISTORY/name/'job/AF_INFERENCE_RESULT.json')})
    p={'stage':'POSTHOC_DEVELOPMENT','created_utc':now(),'cases':cases,'planned_rollouts':12,
       'model_training':False,'cross_root':False,'outcome_informed_case_selection':True,
       'source_hashes':{str(HERE/n):sha(HERE/n) for n in ['run_stage_a.py','replicate_force_window.py','EXPERIMENT_CARD.md']},
       'minimum_start_free_bytes':2*1024**3,'disk_reserve_bytes':int(.75*1024**3)}
    write(path,p);return p

def run():
    guard=(HERE/'QUEUE.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=freeze();results=[]
    for case in protocol['cases']:
        for path,h in protocol['source_hashes'].items():
            if sha(path)!=h:raise ValueError('Queue source drift')
        for key in ['historical_spec','historical_lock','historical_result']:
            if sha(case[key])!=case[key+'_sha256']:raise ValueError('Historical artifact drift')
        output=HERE/case['id']
        if output.exists():
            if not (output/'CASE_AUDIT.json').exists():raise RuntimeError('Inspect retained partial attempt before resuming')
            audit=read(output/'CASE_AUDIT.json')
            if not audit['passed']:raise RuntimeError('Prior gate failed')
            results.append(audit);continue
        if shutil.disk_usage(HERE).free<protocol['minimum_start_free_bytes']:raise RuntimeError('Storage gate: retain old data; archive completed outputs before proceeding')
        output.mkdir();spec=read(case['historical_spec'])
        spec.update(replication=case,replication_protocol_sha256=sha(HERE/'PROTOCOL.json'))
        write(output/'SPEC.json',spec)
        c=spec['context'];verify_runtime(c['runtime_manifest_path'],c['runtime_manifest_sha256'])
        cmd=[str(PY),str(HERE/'replicate_force_window.py'),'--repo',str(REPO),'--out',str(output/'job'),
             '--native-ft','--friction',str(c['friction']),'--policy-seed',str(c['policy_seed'])]
        env=os.environ.copy();env.pop('AF_COLLECTION_CONTEXT',None)
        env.update(AF_P4_PHYSICAL_SURFACE_CAMERA='1',AF_ORIGINAL_SQUEEZE_INNER='1',AF_INFERENCE_CONTEXT=str(output/'SPEC.json'),
            PYTHONUTF8='1',PATH=str(BASE/'runtime_bin')+os.pathsep+env.get('PATH','/usr/bin:/bin'))
        write(output/'PROCESS.json',{'command':cmd,'cwd':str(REPO),'started_utc':now(),'spec_sha256':sha(output/'SPEC.json')})
        print('CASE_START '+case['id'],flush=True)
        stopped_for_disk=False
        with (output/'worker.log').open('x') as log:
            p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(output/'PID.json',{'pid':p.pid})
            while p.poll() is None:
                if shutil.disk_usage(HERE).free<protocol['disk_reserve_bytes']:
                    stopped_for_disk=True;os.killpg(p.pid,signal.SIGTERM);p.wait();break
                time.sleep(10)
        write(output/'EXIT.json',{'exit_code':p.returncode,'disk_guard_stop':stopped_for_disk,'finished_utc':now()})
        if p.returncode:raise RuntimeError('Retained worker failure '+case['id'])
        job=output/'job'; result=read(job/'REPLICATION_RESULT.json'); summary=read(job/'online_qualification.json')
        seal=read(job/'PREACTION_SELECTION_LOCK.json'); old=read(case['historical_lock'])
        if seal['state_sha256']!=old['state_sha256'] or seal['first_chunk_sha256']!=old['first_chunk_sha256']:
            raise RuntimeError('Historical state/first chunk mismatch: stop for diagnosis')
        if len(result['outcomes'])!=6:raise ValueError('Missing outcomes')
        duplicates=summary['duplicate_force_replays']
        if not duplicates or not all(d['full_trace_exact'] for d in duplicates):raise RuntimeError('Duplicate force trace mismatch')
        from audit_original_collected_group import audit_query,audit_branch
        qa=audit_query(job/'query')
        audits=[audit_branch(b) for b in sorted(job.glob('branch_*'))]
        # Match historical repeated forces without assuming a method label.
        prior=read(case['historical_result'])['outcomes'];matched=[]
        for row in result['outcomes']:
            for r in prior:
                if row['force_N']==r['force_N']:
                    matched.append({'force':row['force_N'],'old_success':r['success'],'new_success':row['success']})
        audit={'passed':True,'id':case['id'],'paired_rollouts':6,'historical_state_and_chunk_exact':True,
            'duplicate_replays':duplicates,'outcomes':result['outcomes'],'matched_history':matched,
            'query_audit':qa,'branch_audits':audits,'finished_utc':now()}
        write(output/'CASE_AUDIT.json',audit);results.append(audit)
        print('CASE_DONE '+json.dumps({'id':case['id'],'outcomes':result['outcomes']}),flush=True)
    write(HERE/'STAGE_A_COMPLETE.json',{'completed':True,'cases':results,'paired_rollouts':12,'finished_utc':now(),
         'next_stage_not_launched':True,'new_evidence_backup_pending':True})
    print('STAGE_A_COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','start','run']);mode=parser.parse_args().mode
    if mode=='freeze':print(json.dumps(freeze()))
    elif mode=='run':run()
    else:
        freeze()
        with (HERE/'queue.log').open('x') as log:
            p=subprocess.Popen([str(PY),str(__file__),'run'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(HERE/'QUEUE_PID.json',{'pid':p.pid,'started_utc':now()});print(json.dumps({'pid':p.pid,'log':str(HERE/'queue.log')}))
