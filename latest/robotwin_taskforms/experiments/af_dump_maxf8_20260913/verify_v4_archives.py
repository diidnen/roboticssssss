"""Read-only Anvil archive/receipt and exact per-file coverage verification."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def main(base, names):
    sys.path.insert(0,str(base.parent/'runtime_observer_20260913'))
    from verify_completed_rim20_archive import read_members, sha
    rows=[]
    for name in names:
        archive=base/name
        if archive.resolve().parent!=base.resolve():raise ValueError('Archive must be in destination directory')
        receipt=json.loads(archive.with_suffix('.receipt.json').read_text())
        if sha(archive)!=receipt['sha256'] or archive.stat().st_size!=receipt['size']:
            raise ValueError('Destination differs from source receipt')
        manifest,files,payloads=read_members(archive)
        if len(files)!=receipt['files']:raise ValueError('Receipt/member count mismatch')
        if manifest['case']!=receipt['case']:raise ValueError('Archive case mismatch')
        if receipt['case']:
            prefix='af_dump_maxf8_20260913/additional_data_v1/groups/'+receipt['case']+'/attempt_001/'
            for file in ['ACCEPTED.json','PROCESS_EXIT.json','job/QUERY_ADMISSION.json',
                         'job/query/original_raw_rows.json','job/query/patch_readbacks.json',
                         'job/query/original58_engineering.npy','job/query/INDEPENDENT_NATIVE_FORCE_REBUILD.json']:
                if prefix+file not in files:raise ValueError('Missing accepted-case evidence')
        rows.append({'archive':name,'sha256':receipt['sha256'],'bytes':receipt['size'],
                     'files_verified':len(files),'case':receipt['case'],'passed':True})
    print(json.dumps({'passed':True,'archives':rows,'finished_utc':datetime.now(timezone.utc).isoformat(),
                      'verifier_sha256':sha(Path(__file__))}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('base',type=Path);parser.add_argument('names',nargs='+')
    args=parser.parse_args();main(args.base,args.names)
