"""Completed query metadata or full online gate; raw queries archived separately."""
import argparse
import hashlib
import io
import json
import tarfile
from rootlocal_collection_contract import HERE,read,sha,now


def pack(stage):
    if stage=='queries':
        out=HERE/'rim20_trainval_query_qualification_v1'
        if not read(out/'COMPLETE.json')['completed']:raise ValueError('Incomplete query qualification')
        audit=read(out/'INDEPENDENT_FULL_QUALIFICATION.json')
        if not audit['all_original_admissions_passed'] or len(audit['cases'])!=17:raise ValueError('Missing independent admissions')
        files=set(out.glob('*.json'));receipts=[]
        for item in audit['cases']:
            case=out/item['case'];files.update(case.glob('*.json'))
            files.add(case/'query/INDEPENDENT_CAPTURE_AUDIT.json')
            files.add(case/'query/INDEPENDENT_NATIVE_FORCE_REBUILD.json')
            receipt=HERE.parent/('af_dump_rim20_qualification_'+case.name+'_20260913.tar.receipt.json')
            if sha(receipt)!=item['archive_receipt_sha256']:raise ValueError('Raw archive receipt drift')
            receipts.append(read(receipt))
        destination=HERE.parent/'af_dump_rim20_queries_complete_metadata_20260913_v1.tar.gz'
    elif stage=='online':
        out=HERE/'rim20_online_gate_v1'
        if not read(out/'COMPLETE.json')['completed']:raise ValueError('Incomplete online gate')
        if not read(out/'INDEPENDENT_ENGINEERING_AUDIT.json')['passed']:raise ValueError('Missing gate audit')
        files={p for p in out.rglob('*') if p.is_file()};receipts=[]
        destination=HERE.parent/'af_dump_rim20_online_complete_evidence_20260913_v1.tar.gz'
    else:raise ValueError('Unknown evidence stage')
    files.update(HERE.glob('*.py'));files.update(HERE.glob('*.md'))
    manifest={'created_utc':now(),'stage':stage,'formal_AF_inference':False,'files':{},
              'separate_raw_query_archive_receipts':receipts,
              'Anvil_transfer_integrity_requires_separate_verification':True}
    with destination.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for p in sorted(files):
                data=p.read_bytes();name=str(p.relative_to(HERE));info=tarfile.TarInfo(name);info.size=len(data)
                tar.addfile(info,io.BytesIO(data));manifest['files'][name]={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
    receipt={'archive':str(destination),'size':destination.stat().st_size,'sha256':sha(destination),'files':len(files)}
    with destination.with_suffix('.receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['queries','online']);args=ap.parse_args();pack(args.stage)
