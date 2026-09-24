"""Read-only identity of the live policy service and its checkpoint/code files.

No reset, reload, weight edit or service restart. This records filesystem and
process provenance, not a claim to have hashed the live JAX device buffers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone
from rootlocal_collection_contract import HERE,BASE,REPO,write


def file_record(path):
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('File changed during hashing')
    return {'sha256':h.hexdigest(),'size':after.st_size,'mtime_ns':after.st_mtime_ns}


def record(pid,out):
    process=Path('/proc')/str(pid)
    command=[x.decode() for x in (process/'cmdline').read_bytes().split(b'\0') if x]
    expected=BASE/'checkpoints/pi0_robotwin_30000/30000'
    if 'ckpt_name='+str(expected) not in command or 'train_config_name=pi0_base_aloha_full_sim_arx-x5_seed_0' not in command:
        raise ValueError('Unexpected live checkpoint/config')
    socket=subprocess.check_output(['ss','-ltnp','( sport = :6001 )'],text=True)
    if f'pid={pid},' not in socket:raise ValueError('Expected service does not own the policy socket')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    policy=REPO/'XPolicyLab/policy/Pi_0';sources=[]
    sources.extend(policy.glob('*.py'));sources.extend(policy.glob('*.yml'))
    sources.extend((policy/'openpi/src').rglob('*.py'))
    sources.extend((REPO/'XPolicyLab/client_server').rglob('*.py'))
    sources.extend([REPO/'XPolicyLab/setup_policy_server.py',REPO/'XPolicyLab/model_template.py',
                    REPO/'XPolicyLab/utils/process_data.py',REPO/'XPolicyLab/utils/checkpoint_resolver.py'])
    source_records={};snapshot=out/'SOURCE_SNAPSHOT';snapshot.mkdir()
    for path in sorted(set(p.resolve() for p in sources if p.is_file())):
        receipt=file_record(path);source_records[str(path)]=receipt
        target=snapshot/receipt['sha256']
        if not target.exists():target.write_bytes(path.read_bytes())
    checkpoint_records={str(p):file_record(p) for p in sorted(expected.rglob('*')) if p.is_file()}
    if not checkpoint_records:raise ValueError('Empty checkpoint tree')
    python=policy/'openpi/.venv/bin/python'
    packages=subprocess.check_output([str(python),'-c',
        'import importlib.metadata,json; print(json.dumps(sorted([(p.metadata["Name"],p.version) for p in importlib.metadata.distributions()])))'],text=True)
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'pid':pid,'command':command,
            'process_cwd':str((process/'cwd').resolve()),'socket_readback':socket,
            'process_start':subprocess.check_output(['ps','-p',str(pid),'-o','lstart='],text=True).strip(),
            'checkpoint_path':str(expected),'checkpoint_files':checkpoint_records,
            'source_files':source_records,'installed_packages':json.loads(packages),
            'service_reset_or_reload_performed':False,'live_device_weights_directly_hashed':False,
            'static_initial_task_name_is_not_policy_prompt':True,
            'task_prompt_source':'Pi_0/model.py encode_obs uses request instruction; self.task_name not used for inference',
            'policy_seed_source':'prepare_case -> policy.reset_rng(policy_seed), independent of force'}
    write(out/'PI0_PROVENANCE.json',result)
    print(json.dumps({'path':str(out/'PI0_PROVENANCE.json'),'checkpoint_files':len(checkpoint_records),
        'checkpoint_bytes':sum(r['size'] for r in checkpoint_records.values()),'source_files':len(source_records)}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pid',required=True,type=int);ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();record(args.pid,args.out)
