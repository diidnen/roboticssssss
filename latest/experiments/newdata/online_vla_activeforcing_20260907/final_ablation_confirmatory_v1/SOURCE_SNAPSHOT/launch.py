"""One-worker dev qualification. Final roots cannot be run by this launcher."""
import argparse
import json
import os
import subprocess
import shutil
import time
from pathlib import Path
from datetime import datetime,timezone
from common import HERE,ROOT,BASE,TABERO,V5,sha,read,write

NVML_QUERY = ['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']
MIN_FREE_MIB = 10000

def _policy_server_health(python):
    client_path = TABERO/'benchmarks/openpi/openpi-client/src'
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(client_path), env.get('PYTHONPATH','')])
    code = (
        "import json; from openpi_client.websocket_client_policy import WebsocketClientPolicy; "
        "p=WebsocketClientPolicy('127.0.0.1',18885); "
        "m=p.get_server_metadata(); p._ws.close(); print(json.dumps(m))"
    )
    proc = subprocess.run([str(python),'-c',code],env=env,text=True,capture_output=True,timeout=30)
    if proc.returncode != 0:
        return False, {'exit_code':proc.returncode,'stdout':proc.stdout.strip(),'stderr':proc.stderr.strip()}
    try:
        metadata = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return False, {'exit_code':proc.returncode,'stdout':proc.stdout.strip(),'stderr':proc.stderr.strip(),'parse_error':repr(exc)}
    expected = read(ROOT/'online_vla_restore_20260907/server_v3_frozen/SERVER_READY.json')
    valid = (
        metadata == expected
        and metadata.get('pid') == 2868465
        and metadata.get('checkpoint_sha256') == '0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17'
        and metadata.get('downstream_action_source') == 'ONLINE_VLA'
    )
    return valid, {'exit_code':proc.returncode,'metadata':metadata,'expected_match':metadata == expected}

def _cuda_runtime_health(python):
    code = (
        "import json,torch; assert torch.cuda.is_available(); "
        "free,total=torch.cuda.mem_get_info(0); "
        "x=torch.arange(4096,device='cuda:0'); y=(x*2+1).sum(); "
        "torch.cuda.synchronize(0); print(json.dumps({'free_mib':free//1048576,'total_mib':total//1048576,'value':int(y.item())}))"
    )
    proc = subprocess.run([str(python),'-c',code],text=True,capture_output=True,timeout=60)
    if proc.returncode != 0:
        return False, {'exit_code':proc.returncode,'stdout':proc.stdout.strip(),'stderr':proc.stderr.strip()}
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return False, {'exit_code':proc.returncode,'stdout':proc.stdout.strip(),'stderr':proc.stderr.strip(),'parse_error':repr(exc)}
    return result['free_mib'] >= MIN_FREE_MIB, {'exit_code':proc.returncode,**result}

def gpu_health(out, python):
    nvml_attempts = []
    nvml = None
    for attempt in range(10):
        proc = subprocess.run(NVML_QUERY,text=True,capture_output=True)
        row = {'attempt':attempt+1,'exit_code':proc.returncode,'stdout':proc.stdout.strip(),'stderr':proc.stderr.strip()}
        nvml_attempts.append(row)
        if proc.returncode == 0:
            try:
                free,util=[int(x.strip()) for x in proc.stdout.split(',')]
                nvml = {'free_mib':free,'utilization_percent':util}
                break
            except Exception as exc:
                row['parse_error'] = repr(exc)
        time.sleep(0.5)

    cuda_valid = None
    cuda_evidence = None
    if nvml is None:
        cuda_valid, cuda_evidence = _cuda_runtime_health(python)
        source = 'CUDA_RUNTIME_FALLBACK'
        resource_valid = cuda_valid
    else:
        source = 'NVML'
        resource_valid = nvml['free_mib'] >= MIN_FREE_MIB and nvml['utilization_percent'] <= 70

    policy_valid, policy_evidence = _policy_server_health(python)
    health = {
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'GPU_HEALTH_SOURCE':source,
        'NVML_FAILURE_DURING_PREFLIGHT':nvml is None or any(x['exit_code'] != 0 for x in nvml_attempts),
        'NVML_ATTEMPTS':nvml_attempts,
        'NVML_RESULT':nvml,
        'CUDA_HEALTH_VALID':cuda_valid,
        'CUDA_EVIDENCE':cuda_evidence,
        'POLICY_SERVER_18885_VALID':policy_valid,
        'POLICY_SERVER_EVIDENCE':policy_evidence,
        'RESOURCE_GATE_VALID':resource_valid,
    }
    with (out/'GPU_HEALTH_GATE_LOG.jsonl').open('a') as handle:
        handle.write(json.dumps(health,sort_keys=True)+'\n')
    if not resource_valid:
        raise RuntimeError('Multi-signal GPU resource gate failed '+json.dumps(health,sort_keys=True))
    if not policy_valid:
        raise RuntimeError('Frozen policy server health gate failed '+json.dumps(health,sort_keys=True))
    return health

def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    source=read(V5/'CONTEXT_PLAN.json')['contexts']
    contexts=[dict(x) for x in source if x['root']==5100]
    if len(contexts)!=12:raise RuntimeError('Require one pre-burned dev root / 12 contexts')
    contexts.sort(key=lambda x:(x['task'],{'LOW':0,'MID':1,'HIGH':2}[x['band']]))
    write(out/'DEV_PLAN.json',{'role':'DEVELOPMENT_ONLY','roots':[5100],'contexts':contexts,
        'methods':['ACTIVEFORCING','FIXED_3','FIXED_5'], 'max_main_branches':36,
        'boundary_additions':'Fixed4 on four preregistered boundary contexts only after first smoke',
        'final_fresh_roots_used':False, 'main_fixed_baselines_use_same_probe':True})
    old=read(V5/'CURRENT_ACTIVEFORCING_RUNTIME_MANIFEST.json')
    # Preserve original evidence; candidate manifest is never the final runtime freeze.
    sources={p:h for p,h in old['source_hashes'].items() if Path(p).suffix=='.py'}
    snapshot=out/'SOURCE_SNAPSHOT';snapshot.mkdir()
    for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    sources.update({str(p):sha(p) for p in snapshot.glob('*.py')})
    sources[str(ROOT/'current_fulltask_feasibility_runtime.py')]=sha(ROOT/'current_fulltask_feasibility_runtime.py')
    write(out/'CANDIDATE_RUNTIME_MANIFEST.json',{'version':'ONLINE_VLA_DEV_CANDIDATE_V1','source_hashes':sources,
        'scripted_v5_development_evidence':True,'scripted_v5_is_final_vla_evidence':False,
        'feasibility_sequence_source':'ONLINE_VLA_ACTION_CHUNK','final_vla_runtime_uses_scripted_prefix':False,
        'first_vla_observation':'restore common post-probe RGB cache; all other model inputs recomputed live and bitwise verified',
        'identical_online_initial_chunk_and_feasibility_X_required':True,'vla_action_replay_forbidden':True,
        'replan_steps':10,'server_chunk_steps':50,'per_method_same_probe':True,
        'label_version':'ONLINE_VLA_350STEP_WHOLE_MESH_RELEASE_SUPPORT_V1',
        'feasibility_transfer':'UNQUALIFIED_ONLINE; phase-free exact frozen-weight column removal verified offline',
        'feasibility_input_shape':[8,64],'phase_representation':'NONE',
        'gripper_rule':'AF servo while VLA requests grasp; canonical .04 / zero force on VLA open intent; raw aperture and six raw force dims masked',
        'final_runtime_frozen':False})

def launch(out,index,method):
    plan=read(out/'DEV_PLAN.json')['contexts'][index]
    if plan['root'] not in (170048,170049):raise RuntimeError('Only the two frozen confirmatory ablation roots are allowed')
    job=out/'references'/plan['id'] if method=='REFERENCE' else out/'branches'/(plan['id']+'__'+method)
    if (job/'WORKER_COMPLETION.json').exists():
        if read(job/'WORKER_COMPLETION.json')['logical_success']:return
        raise RuntimeError('Prior failure retained; explicit new version needed')
    python=Path('/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python')
    health=gpu_health(out,python)
    job.mkdir(parents=True,exist_ok=False)
    write(job/'GPU_HEALTH_PREFLIGHT.json',health)
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',
        PYTHONPATH=os.pathsep.join([str(python.parent.parent/'lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64'),str(out/'SOURCE_SNAPSHOT'),str(ROOT),str(TABERO),str(TABERO/'benchmarks/openpi/openpi-client/src')]),
        TABERO_ROOT=str(TABERO),HDF5_TRAJ_SOURCE_DIR=str(TABERO/'benchmarks/datasets/libero/assembled_hdf5'),
        LIBERO_CONFIG_DIR=str(TABERO/'benchmarks/datasets/libero/config'),LIBERO_ASSETS_DATA_DIR=str(TABERO/'benchmarks/datasets/libero/USD'))
    cmd=[str(python),'-u',str(out/'SOURCE_SNAPSHOT/worker.py'),'--out',str(out),'--job',str(job),'--context',str(index),'--method',method]
    with (job/'WORKER.log').open('x') as f:
        p=subprocess.Popen(cmd,cwd=TABERO,env=env,stdout=f,stderr=subprocess.STDOUT)
        write(job/'PROCESS.json',{'pid':p.pid,'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat(),'gpu_health':health})
        print('START',job,p.pid,flush=True);rc=p.wait()
    completion=read(job/'WORKER_COMPLETION.json') if (job/'WORKER_COMPLETION.json').exists() else None
    write(job/'PROCESS_EXIT.json',{'exit_code':rc,'completion':completion})
    print('END',job,completion,flush=True)
    if not completion or not completion['logical_success']:raise RuntimeError('Qualification stopped on invalid runtime/evidence')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--freeze',action='store_true')
    p.add_argument('--context',type=int,default=0);p.add_argument('--method',default='REFERENCE');a=p.parse_args()
    if a.freeze:freeze(Path(a.out))
    else:launch(Path(a.out),a.context,a.method)
