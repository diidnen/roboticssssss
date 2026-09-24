"""Finite fixed-geometry TRAIN/VAL engineering gate; no formal labels or TEST."""
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
from itertools import zip_longest
from rootlocal_collection_contract import HERE,BASE,REPO,read,write,sha,now,verify_runtime
from analyze_mu060_highrate import analyze
from check_canonical_contact_order import normalize


def archive(case):
    destination=HERE.parent/('af_dump_rim20_qualification_'+case.name+'_20260913.tar.gz')
    files={p for p in case.rglob('*') if p.is_file()}
    files.update(HERE.glob('*.py'));files.update(HERE.glob('*.md'))
    files.add(case.parent/'PROTOCOL.json')
    manifest={'created_utc':now(),'formal_admission':False,'files':{}}
    with destination.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for p in sorted(files):
                data=p.read_bytes();name=str(p.relative_to(HERE));info=tarfile.TarInfo(name);info.size=len(data)
                tar.addfile(info,io.BytesIO(data));manifest['files'][name]={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
    receipt={'archive':str(destination),'size':destination.stat().st_size,'sha256':sha(destination),'files':len(files)}
    write(destination.with_suffix('.receipt.json'),receipt)
    print('ARCHIVE_READY '+json.dumps(receipt),flush=True)


def main():
    initial=HERE/'mu070_rim_alignment_diagnostics_v1'
    prior=read(initial/'INDEPENDENT_GEOMETRY_AUDIT.json')
    candidate=next(c for c in prior['cases'] if c['case']=='shallower_20mm')
    if candidate['original_probe_failure'] or candidate['highrate_per_pad_below_original_0p15N']:
        raise ValueError('Candidate evidence changed')
    dataset=HERE/'original_rootlocal_dataset_v2_canonical'
    lock=read(dataset/'FREEZE_LOCK.json')
    values=[.7,.6,.3,.35,.4,.45,.5,.55,.65,.75,.8,.85,.325,.475,.625,.775,.7]
    out=HERE/'rim20_trainval_query_qualification_v1';out.mkdir(exist_ok=False)
    names=['RIM20_QUALIFICATION_PROTOCOL.md','run_rim20_qualification.py','diagnose_canonical_rim_alignment.py',
           'canonical_open_ready_query.py','trace_original_query_contact.py','analyze_mu060_highrate.py',
           'audit_original_collected_group.py','check_canonical_contact_order.py','rootlocal_collection_contract.py']
    sources={str(HERE/name):sha(HERE/name) for name in names}
    # Also enforce the source lock from the original diagnostic intervention.
    sources.update(read(initial/'PROTOCOL.json')['source_hashes'])
    write(out/'PROTOCOL.json',{'frictions':values,'offset_m':.02,'root':200002,'policy_seed':30200002,
        'source_hashes':sources,'frozen_runtime_sha256':lock['runtime_sha256'],
        'candidate_evidence_sha256':sha(initial/'INDEPENDENT_GEOMETRY_AUDIT.json'),
        'formal_admission':False,'test_contexts_used':False,'started_utc':now()})
    def verify():
        verify_runtime(dataset/'RUNTIME_MANIFEST.json',lock['runtime_sha256'])
        if any(sha(p)!=digest for p,digest in sources.items()):raise ValueError('Source drift')
    for index,mu in enumerate(values,1):
        verify();case=out/f'{index:02d}_mu{mu:.3f}';case.mkdir()
        env=os.environ.copy();env.pop('AF_COLLECTION_CONTEXT',None);env.pop('AF_INFERENCE_CONTEXT',None)
        env.update(AF_P4_PHYSICAL_SURFACE_CAMERA='1',AF_ORIGINAL_SQUEEZE_INNER='1',
                   AF_DIAGNOSTIC_RIM_OFFSET_M='.02',PYTHONUTF8='1',
                   PATH=str(BASE/'runtime_bin')+os.pathsep+env.get('PATH','/usr/bin:/bin'))
        command=[str(BASE/'venv_robotwin/bin/python3'),str(HERE/'diagnose_canonical_rim_alignment.py'),
                 '--repo',str(REPO),'--out',str(case/'query'),'--native-ft','--friction',str(mu),
                 '--policy-seed','30200002']
        write(case/'PROCESS.json',{'command':command,'cwd':str(REPO),'started_utc':now()})
        with (case/'worker.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(case/'PID.json',{'pid':process.pid});print('QUERY_START '+case.name+' '+str(process.pid),flush=True)
            code=process.wait()
        write(case/'EXIT.json',{'exit_code':code,'finished_utc':now()})
        if code:
            archive(case);raise RuntimeError('Retained process failure '+case.name)
        analyze(case);verify();archive(case)
        summary=read(case/'query/original_probe_summary.json')
        print('QUERY_DONE '+case.name+' '+json.dumps({'probe_failure':summary['probe_failure']}),flush=True)
        if summary['probe_failure']:raise RuntimeError('Retained original query failure '+case.name)
    a=out/'01_mu0.700/query';b=out/'17_mu0.700/query';different=[]
    with gzip.open(a/'DIAGNOSTIC_PHYSICS_TRACE.jsonl.gz','rt') as x,gzip.open(b/'DIAGNOSTIC_PHYSICS_TRACE.jsonl.gz','rt') as y:
        for n,(u,v) in enumerate(zip_longest(x,y),1):
            if u is None or v is None or normalize(json.loads(u))!=normalize(json.loads(v)):different.append(n)
    repeated=sha(a/'original_raw_rows.json')==sha(b/'original_raw_rows.json')
    write(out/'REPLAY_AUDIT.json',{'raw_rows_byte_equal':repeated,'highrate_rows':n,'mismatched_steps':different})
    if not repeated or different:raise RuntimeError('Repeat mismatch retained')
    verify()
    write(out/'COMPLETE.json',{'completed':True,'qualified_queries':17,'formal_labels_added':0,
        'paired_full_horizon_gate_still_required':True,'finished_utc':now()})


if __name__=='__main__':main()
