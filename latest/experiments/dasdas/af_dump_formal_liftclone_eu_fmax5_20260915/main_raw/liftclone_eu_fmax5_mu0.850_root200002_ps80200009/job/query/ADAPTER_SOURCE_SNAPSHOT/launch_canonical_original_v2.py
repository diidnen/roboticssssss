"""One finite original collection -> training -> inference driver, no retries."""
import os
import subprocess
from rootlocal_collection_contract import HERE, BASE, read, write, sha, now, verify_runtime


def main():
    dataset=HERE/'original_rootlocal_dataset_v2_canonical'
    lock=read(dataset/'FREEZE_LOCK.json')
    verify_runtime(dataset/'RUNTIME_MANIFEST.json',lock['runtime_sha256'])
    out=HERE/'original_canonical_v2_driver';out.mkdir(exist_ok=False)
    write(out/'DRIVER_LOCK.json',{'created_utc':now(),'source_sha256':sha(__file__),
        'dataset_freeze_sha256':sha(dataset/'FREEZE_LOCK.json'),'automatic_retry':False,
        'original_models_and_rules_unchanged':True})
    python=str(BASE/'venv_robotwin/bin/python3')
    stages=[('collection',[python,str(HERE/'run_canonical_original_collection.py'),str(dataset)]),
            ('pipeline',[python,str(HERE/'complete_canonical_original_pipeline.py'),str(dataset),
                         str(HERE/'original_models_v2_canonical'),str(HERE/'original_inference_v2_canonical'),
                         str(HERE/'original_pipeline_v2_canonical')])]
    for label,command in stages:
        verify_runtime(dataset/'RUNTIME_MANIFEST.json',lock['runtime_sha256'])
        write(out/(label+'_PROCESS.json'),{'command':command,'started_utc':now(),'cwd':str(HERE)})
        with (out/(label+'.log')).open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(out/(label+'_PID.json'),{'pid':process.pid});print(label+'_START '+str(process.pid),flush=True)
            code=process.wait()
        write(out/(label+'_EXIT.json'),{'exit_code':code,'finished_utc':now()})
        if code:raise RuntimeError('Stopped on stage failure; retain all evidence: '+label)
    write(out/'COMPLETE.json',{'completed':True,'backup_still_required':True,'finished_utc':now()})
    print('CANONICAL_V2_INFERENCE_AUDITED_BACKUP_PENDING',flush=True)


if __name__=='__main__':main()
