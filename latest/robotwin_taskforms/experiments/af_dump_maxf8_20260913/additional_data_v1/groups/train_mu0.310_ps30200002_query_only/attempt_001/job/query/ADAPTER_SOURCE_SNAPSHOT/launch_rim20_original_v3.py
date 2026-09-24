"""Finite original collection -> training -> inference driver; no retries."""
import subprocess
from rootlocal_collection_contract import HERE,BASE,read,write,sha,now,verify_runtime


def main():
    dataset=HERE/'original_rootlocal_dataset_v3_rim20';lock=read(dataset/'FREEZE_LOCK.json')
    manifest=verify_runtime(dataset/'RUNTIME_MANIFEST.json',lock['runtime_sha256'])
    if manifest['initialization_binding']!='CANONICAL_OPEN_READY_RIM20_V3':raise ValueError('Wrong dataset')
    out=HERE/'original_rim20_v3_driver';out.mkdir(exist_ok=False)
    write(out/'DRIVER_LOCK.json',{'created_utc':now(),'source_sha256':sha(__file__),
        'dataset_freeze_sha256':sha(dataset/'FREEZE_LOCK.json'),'automatic_retry':False,
        'original_models_and_rules_unchanged':True})
    python=str(BASE/'venv_robotwin/bin/python3')
    stages=[('collection',[python,str(HERE/'run_rim20_original_collection.py'),str(dataset)]),
            ('pipeline',[python,str(HERE/'complete_rim20_original_pipeline.py'),str(dataset),
                         str(HERE/'original_models_v3_rim20'),str(HERE/'original_inference_v3_rim20'),
                         str(HERE/'original_pipeline_v3_rim20')])]
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
    print('RIM20_V3_INFERENCE_AUDITED_BACKUP_PENDING',flush=True)


if __name__=='__main__':main()
