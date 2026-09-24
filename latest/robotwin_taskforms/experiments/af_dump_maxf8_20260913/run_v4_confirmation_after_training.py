"""Complete trained-model confirmation and independent audit after fixed prerequisites."""
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'af_dump_original_restore_20260912'
sys.path.insert(0,str(OLD))
from rootlocal_collection_contract import read,write,sha,now


def check_plan():
    plan=HERE/'confirmation_plan_v1';lock=read(plan/'FREEZE_LOCK.json')
    if sha(plan/'COLLECTION_PROTOCOL.json')!=lock['protocol_sha256'] or sha(plan/'CONTEXTS.json')!=lock['contexts_sha256']:
        raise ValueError('Confirmation plan changed')
    protocol=read(plan/'COLLECTION_PROTOCOL.json')
    for path,digest in protocol['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('Frozen confirmation source changed '+path)
    return plan


def main():
    out=HERE/'confirmation_driver_v1';out.mkdir(exist_ok=False)
    guard=(out/'QUEUE.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=check_plan()
    write(out/'STAGE_LOCK.json',{'created_utc':now(),'source_sha256':sha(__file__),
          'confirmation_lock_sha256':sha(plan/'FREEZE_LOCK.json')})
    models=HERE/'models_v4'
    while not (models/'TRAINING_COMPLETE.json').exists():
        if (HERE/'training_driver_v1/STOPPED.json').exists():
            raise RuntimeError('Collection/training stopped; confirmation not started')
        exited=HERE/'training_driver_v1/TRAINING_EXIT.json'
        if exited.exists() and read(exited)['exit_code']:
            raise RuntimeError('Training failed; confirmation not started')
        process=read(HERE/'training_supervisor_launch_v1/PROCESS.json')
        proc=Path('/proc')/str(process['pid'])/'cmdline'
        if not proc.exists() or str(HERE/'run_v4_training_after_collection.py').encode() not in proc.read_bytes():
            raise RuntimeError('Training supervisor ended before completion')
        time.sleep(10)
    if not read(models/'TRAINING_COMPLETE.json')['completed']:raise ValueError('Training incomplete')
    check_plan()
    destination=Path('/media/volume/newdata/exouser/af_dump_maxf8_confirmation_20260913')
    destination.mkdir(exist_ok=True)
    if shutil.disk_usage(destination).free<6*1024**3:
        write(out/'WAITING_FOR_ARCHIVE_SPACE.json',{'required_free_bytes':6*1024**3,'started_utc':now(),
             'action':'verify Anvil copies before reclaiming duplicate local archive files only'})
    while shutil.disk_usage(destination).free<6*1024**3:time.sleep(10)
    check_plan()
    python=str(HERE.parent.parent/'venv_robotwin/bin/python3')
    inference=destination/'inference_v4'
    stages=[('online_inference',[python,str(HERE/'run_v4_inference.py'),str(plan),str(models),str(inference)]),
            ('independent_audit',[python,str(HERE/'audit_v4_inference.py'),str(inference)])]
    for name,command in stages:
        check_plan()
        write(out/(name+'_PROCESS.json'),{'command':command,'cwd':str(HERE),'started_utc':now()})
        with (out/(name+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(out/(name+'_PID.json'),{'pid':process.pid});code=process.wait()
        write(out/(name+'_EXIT.json'),{'exit_code':code,'finished_utc':now()})
        if code:raise RuntimeError('Retained stage failure '+name)
    audit=read(inference/'INDEPENDENT_FINAL_RESULT_AUDIT.json')
    if not audit['passed'] or audit['paired_rollouts_audited']!=48:raise ValueError('Incomplete independent audit')
    write(out/'COMPLETE.json',{'completed':True,'final_result_sha256':sha(inference/'FINAL_INFERENCE_RESULTS.json'),
          'final_audit_sha256':sha(inference/'INDEPENDENT_FINAL_RESULT_AUDIT.json'),
          'final_backup_still_required':True,'finished_utc':now()})
    print('V4_CONFIRMED_INFERENCE_BACKUP_PENDING',flush=True)


if __name__=='__main__':main()
