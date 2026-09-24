"""Wait for the accepted complete dataset, then run the frozen training stage once."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'af_dump_original_restore_20260912'
sys.path.insert(0, str(OLD))
from rootlocal_collection_contract import read, write, sha, now


def main():
    out = HERE/'training_driver_v1'; out.mkdir(exist_ok=False)
    guard = (out/'QUEUE.lock').open('a'); fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources = {str(HERE/name):sha(HERE/name) for name in ['train_v4.py','diagnose_existing_models.py',
               'maxf8_runtime.py','max_force_utility.py','run_v4_training_after_collection.py']}
    write(out/'STAGE_LOCK.json', {'sources':sources,'created_utc':now(),
          'collection_lock_sha256':sha(HERE/'additional_data_v1/FREEZE_LOCK.json')})
    while not (HERE/'additional_data_v1/COLLECTION_COMPLETE.json').exists():
        process = read(HERE/'collection_driver_v1/PROCESS.json')
        path = Path('/proc')/str(process['pid'])/'cmdline'
        if not path.exists() or str(HERE/'run_v4_collection.py').encode() not in path.read_bytes():
            write(out/'STOPPED.json',{'reason':'collection exited before complete acceptance','finished_utc':now()})
            raise RuntimeError('Collection needs diagnosis; no partial training')
        time.sleep(10)
    for path,digest in sources.items():
        if sha(path) != digest: raise ValueError('Training source changed after stage lock')
    python = str(HERE.parent.parent/'venv_robotwin/bin/python3')
    with (out/'training.log').open('x') as log:
        process = subprocess.Popen([python,str(HERE/'train_v4.py')],cwd=HERE,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        write(out/'TRAINING_PID.json',{'pid':process.pid,'started_utc':now()})
        code = process.wait()
    write(out/'TRAINING_EXIT.json',{'exit_code':code,'finished_utc':now()})
    if code: raise RuntimeError('Training failure retained for diagnosis')
    if not read(HERE/'models_v4/TRAINING_COMPLETE.json')['completed']: raise ValueError('Training receipt absent')
    print('V4_TRAINING_COMPLETE_INFERENCE_STILL_REQUIRED',flush=True)


if __name__ == '__main__': main()
