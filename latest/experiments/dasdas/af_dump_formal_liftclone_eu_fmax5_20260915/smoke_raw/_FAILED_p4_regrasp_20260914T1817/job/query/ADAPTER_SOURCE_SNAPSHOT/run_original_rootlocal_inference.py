"""Four prospectively frozen same-root TEST queries after both model locks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
from rootlocal_collection_contract import HERE,BASE,REPO,read,write,sha,now,verify_runtime


def run(dataset,models,out):
    dataset=Path(dataset).resolve();models=Path(models).resolve();out=Path(out).resolve()
    complete=read(models/'TRAINING_COMPLETE.json')
    if not complete.get('completed') or complete['test_groups_executed']!=0:raise ValueError('Invalid training lock')
    source_hashes={str(p):sha(p) for p in [HERE/'infer_original_rootlocal.py',Path(__file__).resolve()]}
    contexts=[c for c in read(dataset/'CONTEXTS.json') if c['split']=='TEST']
    lock=read(dataset/'FREEZE_LOCK.json')
    if len(contexts)!=4 or sha(dataset/'CONTEXTS.json')!=lock['contexts_sha256']:raise ValueError('TEST schedule changed')
    if not out.exists():out.mkdir(parents=True)
    guard=(out/'INFERENCE_QUEUE.lock').open('a')
    try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A live inference queue already holds this OS lock')
    protocol={'dataset':str(dataset),'models':str(models),'contexts':contexts,
              'training_complete_sha256':sha(models/'TRAINING_COMPLETE.json'),'source_hashes':source_hashes,
              'methods':['ActiveForcing','Fixed-1N','Fixed-3N','Fixed-5N','Fixed-6N','Fixed-8N'],
              'test_labels_read_before_protocol':False}
    path=out/'INFERENCE_PROTOCOL.json'
    if path.exists():
        if read(path)!=protocol:raise ValueError('Inference protocol/model changed')
    else:write(path,protocol)
    results=[]
    for context in contexts:
        verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
        for source,digest in source_hashes.items():
            if sha(source)!=digest:raise ValueError('Inference source drift')
        case=out/context['id']
        if case.exists():
            receipt=case/'PROCESS_EXIT.json';result=case/'job/AF_INFERENCE_RESULT.json'
            if not receipt.exists() or read(receipt)['exit_code']!=0 or not result.exists():
                raise RuntimeError('Retained unfinished/failed TEST attempt requires inspection: '+str(case))
            if sha(result)!=read(receipt)['result_sha256']:raise ValueError('TEST result changed')
            if read(result)['context']!=context:raise ValueError('TEST context mismatch')
            results.append(read(result));continue
        case.mkdir()
        spec={'context':context,'models':str(models),
              'belief_manifest_sha256':complete['belief_manifest_sha256'],
              'feasibility_manifest_sha256':complete['feasibility_manifest_sha256']}
        write(case/'SPEC.json',spec)
        command=[str(BASE/'venv_robotwin/bin/python3'),str(HERE/'infer_original_rootlocal.py'),
            '--repo',str(REPO),'--out',str(case/'job'),'--native-ft','--friction',str(context['friction']),
            '--policy-seed',str(context['policy_seed'])]
        env=os.environ.copy();env.update(AF_P4_PHYSICAL_SURFACE_CAMERA='1',AF_ORIGINAL_SQUEEZE_INNER='1',
            AF_INFERENCE_CONTEXT=str(case/'SPEC.json'),PYTHONUTF8='1',
            PATH=str(BASE/'runtime_bin')+os.pathsep+env.get('PATH','/usr/bin:/bin'))
        write(case/'PROCESS.json',{'command':command,'cwd':str(REPO),'spec_sha256':sha(case/'SPEC.json'),
              'inference_protocol_sha256':sha(path),'started_utc':now()})
        print('ORIGINAL_AF_TEST_START '+context['id'],flush=True)
        with (case/'worker.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(case/'PROCESS_PID.json',{'pid':process.pid});code=process.wait()
        result=case/'job/AF_INFERENCE_RESULT.json'
        write(case/'PROCESS_EXIT.json',{'exit_code':code,'finished_utc':now(),
                                       'result_sha256':sha(result) if result.exists() else None})
        if code or not result.exists():raise RuntimeError('TEST failed/unknown; no fabricated outcome')
        results.append(read(result))
    summary=[]
    for method in protocol['methods']:
        rows=[next(r for r in case['outcomes'] if r['method']==method) for case in results]
        summary.append({'method':method,'successes':sum(r['success'] for r in rows),'queries':len(rows),
                        'selected_forces_N':[r['force_N'] for r in rows],
                        'mean_selected_force_N':sum(r['force_N'] for r in rows)/len(rows),
                        'mean_original_utility':sum(r['original_utility'] for r in rows)/len(rows)})
    final={'completed':True,'independent_query_settings':4,'paired_rollouts':24,'summary':summary,
           'cases':results,'inference_protocol_sha256':sha(path),
           'scope':'root 200002 only; no cross-root generalization or damage-safety inference',
           'backup_verification_still_required':True,'finished_utc':now()}
    if not (out/'FINAL_INFERENCE_RESULTS.json').exists():write(out/'FINAL_INFERENCE_RESULTS.json',final)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);ap.add_argument('models',type=Path);ap.add_argument('out',type=Path)
    args=ap.parse_args();run(args.dataset,args.models,args.out)
