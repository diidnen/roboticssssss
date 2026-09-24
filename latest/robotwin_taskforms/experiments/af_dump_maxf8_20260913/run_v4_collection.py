"""Finite sequential queue, original audits, retained failures, bounded disk use."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'af_dump_original_restore_20260912'
BASE = HERE.parent.parent
sys.path.insert(0, str(OLD))
from rootlocal_collection_contract import read, write, sha, now, verify_runtime
from run_original_rootlocal_collection import valid_completion


def main():
    dataset = HERE/'additional_data_v1'
    guard = (dataset/'QUEUE.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = read(dataset/'FREEZE_LOCK.json')
    if sha(dataset/'CONTEXTS.json') != lock['contexts_sha256']: raise ValueError('Schedule changed')
    if sha(dataset/'COLLECTION_PROTOCOL.json') != lock['protocol_sha256']: raise ValueError('Protocol changed')
    finished = []
    for context in read(dataset/'CONTEXTS.json'):
        verify_runtime(context['runtime_manifest_path'], context['runtime_manifest_sha256'])
        attempt = dataset/'groups'/context['id']/'attempt_001'
        if attempt.exists():
            if not (attempt/'ACCEPTED.json').exists(): raise RuntimeError('Inspect retained incomplete attempt '+str(attempt))
            accepted = read(attempt/'ACCEPTED.json')
            for file, digest in accepted['artifact_hashes'].items():
                if sha(file) != digest: raise ValueError('Completed artifact changed')
            finished.append(accepted); continue
        if shutil.disk_usage(dataset).free < 1500*1024**2:
            raise RuntimeError('Disk reserve reached before next group; completed data preserved')
        attempt.mkdir(parents=True); write(attempt/'CONTEXT.json', context)
        command = [str(BASE/'venv_robotwin/bin/python3'), str(HERE/'collect_v4_context.py'),
                   '--repo',str(BASE/'RoboTwin'),'--out',str(attempt/'job'),'--native-ft',
                   '--friction',str(context['friction']),'--policy-seed',str(context['policy_seed'])]
        env = os.environ.copy(); env.pop('AF_INFERENCE_CONTEXT', None)
        env.update(AF_P4_PHYSICAL_SURFACE_CAMERA='1',AF_ORIGINAL_SQUEEZE_INNER='1',
                   AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP='1',AF_COLLECTION_CONTEXT=str(attempt/'CONTEXT.json'),
                   PYTHONUTF8='1',PATH=str(BASE/'runtime_bin')+os.pathsep+env.get('PATH','/usr/bin:/bin'))
        write(attempt/'PROCESS.json', {'context':context,'command':command,'cwd':str(BASE/'RoboTwin'),'started_utc':now()})
        with (attempt/'worker.log').open('x') as log:
            process = subprocess.Popen(command,cwd=BASE/'RoboTwin',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(attempt/'PROCESS_PID.json', {'pid':process.pid,'started_utc':now()})
            print('V4_START '+context['id']+' '+str(process.pid),flush=True)
            code = process.wait()
        write(attempt/'PROCESS_EXIT.json', {'exit_code':code,'finished_utc':now(),'pid':process.pid})
        if code: raise RuntimeError('Retained execution failure '+str(attempt))
        if context['collection_mode'] == 'FULL_GROUP':
            if not valid_completion(attempt, context): raise ValueError('Invalid full-group receipt')
            with (attempt/'audit.log').open('x') as log:
                result = subprocess.run([str(BASE/'venv_robotwin/bin/python3'),str(OLD/'audit_original_collected_group.py'),str(attempt/'job')],
                                        cwd=OLD,stdout=log,stderr=subprocess.STDOUT)
            write(attempt/'AUDIT_EXIT.json', {'exit_code':result.returncode,'finished_utc':now()})
            if result.returncode: raise RuntimeError('Independent raw audit failed '+str(attempt))
            required = [attempt/'job/INDEPENDENT_GROUP_AUDIT.json',attempt/'job/COLLECTED_ROWS.json']
            if not read(required[0])['passed']: raise ValueError('Full audit did not pass')
        else:
            required = [attempt/'job/QUERY_ONLY_COMPLETE.json',attempt/'job/query/INDEPENDENT_NATIVE_FORCE_REBUILD.json']
            if not read(required[0])['completed'] or not read(required[1])['passed']: raise ValueError('Query audit failed')
        required += [attempt/'PROCESS.json',attempt/'PROCESS_EXIT.json',attempt/'job/QUERY_ADMISSION.json',
                     attempt/'job/query/original_raw_rows.json',attempt/'job/query/patch_readbacks.json',
                     attempt/'job/query/original58_engineering.npy',attempt/'job/query/ACTUAL_OBJECT_MATERIAL.json']
        accepted = {'context':context,'attempt':str(attempt),'accepted':True,
                    'artifact_hashes':{str(p):sha(p) for p in required},'finished_utc':now()}
        write(attempt/'ACCEPTED.json',accepted); finished.append(accepted)
        print('V4_ACCEPTED '+context['id'],flush=True)
    write(dataset/'COLLECTION_COMPLETE.json', {'completed':True,'contexts':finished,'new_downstream_labels':64,
                                             'query_only_groups':16,'test_groups_executed':0,'finished_utc':now()})
    print('V4_COLLECTION_COMPLETE',flush=True)


if __name__ == '__main__': main()
