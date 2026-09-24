"""Archive all four terminal diagnostic cases, including every failure."""
import hashlib
import io
import json
import tarfile
from rootlocal_collection_contract import HERE,read,sha,now


def main():
    out=HERE/'mu070_rim_alignment_diagnostics_v1'
    if not read(out/'COMPLETE.json')['completed']:raise ValueError('Incomplete diagnostics')
    audit=read(out/'INDEPENDENT_GEOMETRY_AUDIT.json')
    if len(audit['cases'])!=4 or audit['formal_labels_added']!=0:raise ValueError('Wrong audit')
    files={p for p in out.rglob('*') if p.is_file()}
    files.update(HERE.glob('*.py'));files.update(HERE.glob('*.md'))
    destination=HERE.parent/'af_dump_mu070_rim_diagnostics_20260913_v1.tar.gz'
    manifest={'created_utc':now(),'all_four_cases_including_failures':True,'formal_admission':False,'files':{}}
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


if __name__=='__main__':main()
