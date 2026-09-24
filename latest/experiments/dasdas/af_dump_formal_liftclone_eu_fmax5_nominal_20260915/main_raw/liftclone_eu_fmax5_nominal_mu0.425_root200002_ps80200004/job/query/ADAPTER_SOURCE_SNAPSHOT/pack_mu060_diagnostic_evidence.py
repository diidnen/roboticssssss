"""Preserve the failed formal query and both completed diagnostic repeats."""
import hashlib
import io
import json
import tarfile
from rootlocal_collection_contract import HERE, read, sha, now


def main():
    failed=HERE/'original_rootlocal_dataset_v1/groups/train_mu0.600_root200002/attempt_001'
    repeated=HERE/'mu060_contact_diagnostic_repeats_v1'
    if read(failed/'PROCESS_EXIT.json')['exit_code']!=1:raise ValueError('Expected retained failed formal attempt')
    if not read(repeated/'COMPLETE.json')['completed']:raise ValueError('Diagnostic repeats still live')
    roots=[failed,repeated,HERE/'mu060_contact_diagnosis_v1']
    files=set(p for root in roots for p in root.rglob('*') if p.is_file())
    files.update(HERE.glob('*.py')); files.update(HERE.glob('*.md'))
    files.update((HERE/'original_rootlocal_pipeline_v1').glob('*'))
    dataset=HERE/'original_rootlocal_dataset_v1'
    files.update(p for p in dataset.glob('*') if p.suffix in ('.json','.md'))
    files=sorted(p for p in files if p.is_file())
    destination=HERE.parent/'af_dump_mu060_failure_and_diagnostics_20260913_v1.tar.gz'
    manifest={'created_utc':now(),'failed_formal_query_preserved':True,'diagnostic_queries':2,
              'formal_labels_added':0,'files':{}}
    with destination.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for path in files:
                content=path.read_bytes();name=str(path.relative_to(HERE))
                info=tarfile.TarInfo(name);info.size=len(content);tar.addfile(info,io.BytesIO(content))
                manifest['files'][name]={'size':len(content),'sha256':hashlib.sha256(content).hexdigest()}
            content=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(content)
            tar.addfile(info,io.BytesIO(content))
    receipt={'archive':str(destination),'size':destination.stat().st_size,'sha256':sha(destination),'files':len(files)}
    with destination.with_suffix('.receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
