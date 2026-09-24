"""Back up a fully audited, terminal group without including any live worker."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from rootlocal_collection_contract import HERE,read,sha
from run_original_rootlocal_collection import valid_completion


def pack(attempt):
    attempt=Path(attempt).resolve();context=read(attempt/'CONTEXT.json')
    if not valid_completion(attempt,context):raise ValueError('Group not terminal and validated')
    audit=read(attempt/'job/INDEPENDENT_GROUP_AUDIT.json')
    if not audit['passed'] or audit['branches']!=8:raise ValueError('Raw evidence audit incomplete')
    dataset=Path(context['collection_protocol_path']).parent
    destination=HERE.parent/(f"af_dump_original_{context['id']}_20260913.tar.gz")
    files=set(p for p in attempt.rglob('*') if p.is_file())
    files.update(p for p in (dataset/'SOURCE_SNAPSHOT').glob('*') if p.is_file())
    files.update(p for p in dataset.glob('*') if p.suffix in ('.json','.md') and p.is_file())
    files.update(HERE.glob('*.py'));files.update(HERE.glob('*.md'))
    manifest={'context':context,'validated_terminal_group':True,'live_groups_included':False,'files':{}}
    with destination.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for path in sorted(files):
                content=path.read_bytes();name=str(path.relative_to(HERE))
                info=tarfile.TarInfo(name);info.size=len(content);tar.addfile(info,io.BytesIO(content))
                manifest['files'][name]={'sha256':hashlib.sha256(content).hexdigest(),'size':len(content)}
            content=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(content)
            tar.addfile(info,io.BytesIO(content))
    h=hashlib.sha256()
    with destination.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    result={'archive':str(destination),'size':destination.stat().st_size,'sha256':h.hexdigest(),'files':len(files)}
    with destination.with_suffix('.receipt.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('attempt',type=Path);args=ap.parse_args();pack(args.attempt)
