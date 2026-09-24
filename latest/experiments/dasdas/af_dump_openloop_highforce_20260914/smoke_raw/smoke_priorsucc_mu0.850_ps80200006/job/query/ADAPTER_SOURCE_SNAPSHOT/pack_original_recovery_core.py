"""Small priority backup; explicitly not a substitute for complete raw archives."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from datetime import datetime,timezone
from rootlocal_collection_contract import HERE,read


def pack(dataset):
    dataset=Path(dataset).resolve();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=HERE.parent/f'af_dump_original_recovery_core_{stamp}.tar.gz'
    files=set(HERE.glob('*.py'))|set(HERE.glob('*.md'))
    files.update(p for p in dataset.glob('*.json') if p.is_file())
    files.update(p for p in (dataset/'SOURCE_SNAPSHOT').glob('*') if p.is_file())
    provenance=HERE/'native_pi0_provenance_20260913'
    files.update(p for p in provenance.rglob('*') if p.is_file())
    completed_branches=0;query_groups=[]
    for group in sorted((dataset/'groups').glob('*')):
        for attempt in sorted(group.glob('attempt_*')):
            job=attempt/'job';query=job/'query'
            # A terminal query has a final qualification, all original rows,
            # and closed control/readback files. Exclude live query captures.
            qualification=query/'qualification.json'
            if not qualification.exists() or not read(qualification).get('completed'):continue
            query_groups.append({'context':group.name,'attempt':attempt.name})
            files.update(p for p in attempt.glob('*.json') if p.is_file())
            files.update(p for p in job.glob('*.json') if p.is_file())
            files.update(p for p in query.glob('*.json') if p.is_file())
            files.update(p for p in query.glob('*.npy') if p.is_file())
            for branch in sorted(job.glob('branch_*')):
                terminal=branch/'result.json'
                if not terminal.exists() or not read(terminal).get('completed'):continue
                completed_branches+=1
                files.update(p for p in branch.glob('*.json') if p.is_file())
                files.update(p for p in branch.glob('*.npy') if p.is_file())
    manifest={'scope':'priority recovery core; RAW CAMERA NPZ AND PHYSICS TRACE ARCHIVES STILL REQUIRED',
        'not_a_complete_raw_data_backup':True,'created_utc':stamp,'query_groups':query_groups,
        'terminal_branch_records':completed_branches,'files':{}}
    with destination.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for path in sorted(files):
                data=path.read_bytes();name=str(path.relative_to(HERE));digest=hashlib.sha256(data).hexdigest()
                info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
                manifest['files'][name]={'sha256':digest,'size':len(data)}
            data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('RECOVERY_CORE_MANIFEST.json');info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
    h=hashlib.sha256()
    with destination.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    print(json.dumps({'archive':str(destination),'sha256':h.hexdigest(),'bytes':destination.stat().st_size,
                      'terminal_branch_records':completed_branches,'files':len(files)}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);args=ap.parse_args();pack(args.dataset)
