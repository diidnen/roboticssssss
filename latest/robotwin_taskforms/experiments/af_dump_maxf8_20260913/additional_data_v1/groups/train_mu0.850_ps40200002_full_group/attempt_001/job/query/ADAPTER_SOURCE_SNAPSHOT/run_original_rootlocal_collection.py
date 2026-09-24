"""Exclusive resumable queue; unknown/crashed jobs never become negative labels."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from rootlocal_collection_contract import HERE,BASE,REPO,read,write,sha,now,verify_runtime


def valid_completion(attempt,context):
    exit_path=attempt/'PROCESS_EXIT.json'
    done=attempt/'job/LOGICAL_COMPLETION.json'
    if not exit_path.exists() or not done.exists():return False
    process=read(attempt/'PROCESS.json')
    if read(exit_path)['exit_code']!=0 or process['context']!=context:return False
    receipt=read(done)
    if receipt.get('exit_code')!=0 or receipt.get('rows')!=8 or receipt.get('context_id')!=context['id']:return False
    if receipt['runtime_manifest_sha256']!=context['runtime_manifest_sha256']:return False
    if sha(attempt/'job/COLLECTED_ROWS.json')!=receipt['rows_sha256']:return False
    if sha(attempt/'job/online_qualification.json')!=receipt['summary_sha256']:return False
    for row in read(attempt/'job/COLLECTED_ROWS.json'):
        if sha(Path(row['job'])/'result.json')!=row['result_sha256']:return False
        if sha(Path(row['job'])/'original_motion_feature.json')!=row['feature_sha256']:return False
    return True


def run(dataset,max_groups=None):
    dataset=Path(dataset).resolve()
    guard=(dataset/'COLLECTION_QUEUE.lock').open('a')
    try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another queue holds the live OS lock; refusing duplicate launch')
    # Hold guard throughout the queue process. Child close_fds prevents leakage.
    lock=read(dataset/'FREEZE_LOCK.json')
    if sha(dataset/'CONTEXTS.json')!=lock['contexts_sha256']:raise ValueError('Context schedule changed')
    contexts=[c for c in read(dataset/'CONTEXTS.json') if c['split'] in ('TRAIN','VAL')]
    contexts.sort(key=lambda c:(0 if c['split']=='TRAIN' else 1,-c['friction']))
    finished=[];launched=0
    for context in contexts:
        verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
        group=dataset/'groups'/context['id'];group.mkdir(parents=True,exist_ok=True)
        attempts=sorted(group.glob('attempt_*'))
        valid=[a for a in attempts if valid_completion(a,context)]
        if len(valid)>1:raise ValueError('Multiple valid attempts require explicit duplicate resolution')
        if valid:
            finished.append({'context':context,'attempt':str(valid[0])});continue
        if attempts:
            raise RuntimeError('Unfinished/failed attempt retained; inspect before explicit repair: '+str(group))
        if max_groups is not None and launched>=max_groups:break
        attempt=group/'attempt_001';attempt.mkdir()
        context_path=attempt/'CONTEXT.json';write(context_path,context)
        command=[str(BASE/'venv_robotwin/bin/python3'),str(HERE/'collect_original_rootlocal.py'),
                 '--repo',str(REPO),'--out',str(attempt/'job'),'--native-ft',
                 '--friction',str(context['friction']),'--policy-seed',str(context['policy_seed'])]
        env=os.environ.copy();env.update(AF_P4_PHYSICAL_SURFACE_CAMERA='1',AF_ORIGINAL_SQUEEZE_INNER='1',
            AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP='1',AF_COLLECTION_CONTEXT=str(context_path),
            PYTHONUTF8='1',PATH=str(BASE/'runtime_bin')+os.pathsep+env.get('PATH','/usr/bin:/bin'))
        receipt={'context':context,'command':command,'cwd':str(REPO),'started_utc':now(),
                 'queue_pid':os.getpid(),'queue_source_sha256':sha(__file__),
                 'runtime_manifest_sha256':context['runtime_manifest_sha256']}
        write(attempt/'PROCESS.json',receipt)
        print('COLLECTION_START '+json.dumps(receipt),flush=True)
        with (attempt/'worker.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=True)
            write(attempt/'PROCESS_PID.json',{'pid':process.pid,'started_utc':now()})
            # Each worker branch enforces its own 1800s deadline and cleanup;
            # this wait does not classify a slow but live run as failed.
            code=process.wait()
        write(attempt/'PROCESS_EXIT.json',{'exit_code':code,'finished_utc':now(),'pid':process.pid})
        print('COLLECTION_EXIT '+json.dumps({'context_id':context['id'],'exit_code':code}),flush=True)
        if code or not valid_completion(attempt,context):
            raise RuntimeError('Collection stopped on unknown/invalid terminal receipt: '+str(attempt))
        finished.append({'context':context,'attempt':str(attempt)});launched+=1
    if len(finished)==len(contexts):
        path=dataset/'COLLECTION_COMPLETE.json'
        result={'completed':True,'contexts':finished,'total_rows':len(finished)*8,'finished_utc':now(),
                'test_groups_executed':0,'independent_training_admission_still_required':True}
        if not path.exists():write(path,result)
        print('COLLECTION_COMPLETE '+json.dumps({'groups':len(finished),'rows':len(finished)*8}),flush=True)
    return len(finished)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);ap.add_argument('--max-groups',type=int)
    args=ap.parse_args();run(args.dataset,args.max_groups)
