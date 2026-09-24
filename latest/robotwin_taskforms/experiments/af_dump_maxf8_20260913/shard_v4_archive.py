"""Split an already closed, checksummed backup into independently verified transport parts."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ARCHIVES=Path('/media/volume/newdata/exouser/af_dump_maxf8_archives_20260913')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def main(name):
    source=ARCHIVES/name
    if source.resolve().parent!=ARCHIVES.resolve() or source.is_symlink() or not name.endswith('.tar.gz'):
        raise ValueError('Require exact archive basename')
    receipt=json.loads(source.with_suffix('.receipt.json').read_text())
    if sha(source)!=receipt['sha256'] or source.stat().st_size!=receipt['size']:
        raise ValueError('Closed source differs from receipt')
    destination=ARCHIVES/(name+'.parts_v1')
    destination.mkdir(exist_ok=False)
    parts=[]
    with source.open('rb') as stream:
        index=0
        for data in iter(lambda:stream.read(16*1024*1024),b''):
            part=destination/('part_%04d.bin'%index)
            with part.open('xb') as out:out.write(data)
            parts.append({'name':part.name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            index+=1
    manifest={'source_receipt':receipt,'parts':parts,'created_utc':datetime.now(timezone.utc).isoformat(),
              'transport_only':True,'scientific_artifacts_unchanged':True}
    with (destination/'MANIFEST.json').open('x') as out:json.dump(manifest,out,indent=2)
    print(json.dumps({'directory':str(destination),'manifest_sha256':sha(destination/'MANIFEST.json'),
                      'archive_sha256':receipt['sha256'],'parts':len(parts),'bytes':receipt['size']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');main(p.parse_args().name)
