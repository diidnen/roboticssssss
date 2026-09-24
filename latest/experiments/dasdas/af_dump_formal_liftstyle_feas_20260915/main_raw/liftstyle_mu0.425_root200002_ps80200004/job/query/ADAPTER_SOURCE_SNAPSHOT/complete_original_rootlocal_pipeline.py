"""Finite collection-dependency -> independent audit -> training -> online TEST.

Wait on the live collection's OS lock, not a stale PID or repeated status file.
No training on partial data, no retry/reselection based on TEST outcomes.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
from rootlocal_collection_contract import HERE,BASE,read,write,sha,now


def execute(label,command,out,source_hashes):
    for path,digest in source_hashes.items():
        if sha(path)!=digest:raise ValueError('Frozen pipeline code changed: '+path)
    receipt={'stage':label,'command':command,'started_utc':now(),'source_hashes':source_hashes}
    write(out/(label+'_PROCESS.json'),receipt)
    print('PIPELINE_STAGE_START '+json.dumps(receipt),flush=True)
    with (out/(label+'.log')).open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/(label+'_PID.json'),{'pid':process.pid});code=process.wait()
    write(out/(label+'_EXIT.json'),{'exit_code':code,'finished_utc':now()})
    if code:raise RuntimeError('Pipeline stage failed; raw data retained: '+label)


def run(dataset,models,inference,out):
    dataset,models,inference,out=[Path(p).resolve() for p in (dataset,models,inference,out)]
    out.mkdir(parents=True,exist_ok=False)
    scripts=[HERE/name for name in ['audit_original_collected_group.py','train_original_rootlocal.py',
        'infer_original_rootlocal.py','run_original_rootlocal_inference.py','test_original_rootlocal_training.py',
        'complete_original_rootlocal_pipeline.py']]
    source_hashes={str(p):sha(p) for p in scripts}
    write(out/'PIPELINE_PROTOCOL.json',{'dataset':str(dataset),'models':str(models),'inference':str(inference),
        'collection_freeze_lock_sha256':sha(dataset/'FREEZE_LOCK.json'),'source_hashes':source_hashes,
        'created_utc':now(),'models_selected_before_TEST':True,'automatic_retry_or_model_reselection':False})
    guard=(dataset/'COLLECTION_QUEUE.lock').open('a')
    print('PIPELINE_WAITING_FOR_COLLECTION_OS_LOCK',flush=True)
    fcntl.flock(guard,fcntl.LOCK_EX)
    completed=read(dataset/'COLLECTION_COMPLETE.json')
    if not completed.get('completed') or completed['total_rows']!=128 or len(completed['contexts'])!=16:
        raise ValueError('Collection dependency terminal but incomplete; training prohibited')
    python=str(BASE/'venv_robotwin/bin/python3')
    for index,entry in enumerate(completed['contexts']):
        job=Path(entry['attempt'])/'job'
        execute(f'audit_{index:02d}',[python,str(HERE/'audit_original_collected_group.py'),str(job)],out,source_hashes)
    execute('training_recipe_tests',[python,str(HERE/'test_original_rootlocal_training.py')],out,source_hashes)
    execute('training',[python,str(HERE/'train_original_rootlocal.py'),str(dataset),str(models)],out,source_hashes)
    if not read(models/'TRAINING_COMPLETE.json')['completed']:raise ValueError('Training logical completion absent')
    execute('online_inference',[python,str(HERE/'run_original_rootlocal_inference.py'),str(dataset),str(models),str(inference)],out,source_hashes)
    final=inference/'FINAL_INFERENCE_RESULTS.json'
    if not read(final)['completed']:raise ValueError('Online inference logical completion absent')
    write(out/'PIPELINE_COMPLETE.json',{'completed':True,'final_result':str(final),'sha256':sha(final),
                                      'backup_and_final_independent_audit_still_required':True,'finished_utc':now()})
    print('PIPELINE_INFERENCE_COMPLETE '+str(final),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('dataset','models','inference','out'):ap.add_argument(name,type=Path)
    args=ap.parse_args();run(args.dataset,args.models,args.inference,args.out)
