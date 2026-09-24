"""Full raw backup of closed query and terminal branches while queue continues.

Never include a live branch, label an unfinished group complete, or change the
frozen runtime. Independent force/action reconstruction precedes the snapshot.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
from rootlocal_collection_contract import HERE, read, sha, verify_runtime
from audit_original_collected_group import audit_query,audit_branch


def main(attempt):
    attempt=Path(attempt).resolve();context=read(attempt/'CONTEXT.json')
    manifest=verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
    if manifest.get('initialization_binding')!='CANONICAL_OPEN_READY_V2':raise ValueError('Wrong data version')
    job=attempt/'job';query=job/'query'
    if read(query/'qualification.json')['probe_failure']!=0:raise ValueError('Unqualified formal query')
    audit_query(query)
    branches=sorted(path.parent for path in job.glob('branch_*/result.json'))
    if not branches:raise ValueError('No terminal branches yet')
    for branch in branches:audit_branch(branch)
    dataset=Path(context['collection_protocol_path']).parent
    files=set(path for root in [query,*branches] for path in root.rglob('*') if path.is_file())
    files.update(p for p in attempt.glob('*.json') if p.is_file())
    # Exclude evolving group summaries/logs; these particular locks are closed.
    for name in ['CONTEXT_LOCK.json','QUERY_ADMISSION.json']:
        if (job/name).exists():files.add(job/name)
    files.update(p for p in dataset.glob('*') if p.suffix in ('.json','.md') and p.is_file())
    files.update((dataset/'SOURCE_SNAPSHOT').glob('*'))
    files.update(HERE.glob('*.py'));files.update(HERE.glob('*.md'))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=HERE.parent/f'af_dump_canonical_v2_{context["id"]}_terminal{len(branches):02d}_{stamp}.tar.gz'
    metadata={'context':context,'partial_group_snapshot_not_logical_completion':True,
              'full_raw_query_included':True,'live_branches_excluded':True,
              'terminal_branches':[str(p) for p in branches],'files':{}}
    with destination.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for path in sorted(files):
                data=path.read_bytes();name=str(path.relative_to(HERE));digest=hashlib.sha256(data).hexdigest()
                info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
                metadata['files'][name]={'size':len(data),'sha256':digest}
            data=json.dumps(metadata,indent=2).encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
    receipt={'archive':str(destination),'sha256':sha(destination),'bytes':destination.stat().st_size,
             'terminal_branches':len(branches),'files':len(files),'partial_group_not_complete':True}
    with destination.with_suffix('.receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('attempt',type=Path);args=ap.parse_args();main(args.attempt)
