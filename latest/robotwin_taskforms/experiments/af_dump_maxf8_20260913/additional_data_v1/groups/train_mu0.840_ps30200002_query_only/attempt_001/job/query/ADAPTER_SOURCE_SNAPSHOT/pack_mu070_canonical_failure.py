"""Preserve the full failed v2 query and stopped driver without fake labels."""
import hashlib
import io
import json
import tarfile
from rootlocal_collection_contract import HERE,read,sha,now


def main():
    dataset=HERE/'original_rootlocal_dataset_v2_canonical'
    failed=dataset/'groups/train_mu0.700_root200002/attempt_001'
    if read(failed/'PROCESS_EXIT.json')['exit_code']!=1:raise ValueError('Not a terminal failed attempt')
    if read(failed/'job/query/original_probe_summary.json')['probe_failure']!=1:raise ValueError('Wrong failure type')
    files={p for root in [failed,HERE/'original_canonical_v2_driver'] for p in root.rglob('*') if p.is_file()}
    files.update(p for p in dataset.glob('*') if p.suffix in ('.json','.md') and p.is_file())
    files.update((dataset/'SOURCE_SNAPSHOT').glob('*'));files.update(HERE.glob('*.py'));files.update(HERE.glob('*.md'))
    destination=HERE.parent/'af_dump_mu070_canonical_failure_20260913_v1.tar.gz'
    manifest={'created_utc':now(),'formal_failure_preserved':True,'downstream_labels_added':0,'files':{}}
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
