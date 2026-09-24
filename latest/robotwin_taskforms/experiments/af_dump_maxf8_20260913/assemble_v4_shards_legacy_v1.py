"""Validate transport shards and reconstruct original bytes without replacing existing archives."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def main(directory,expected_manifest,assemble):
    directory=directory.resolve();manifest_path=directory/'MANIFEST.json'
    if sha(manifest_path)!=expected_manifest:raise ValueError('Manifest differs from source')
    manifest=json.loads(manifest_path.read_text());parts=manifest['parts']
    verified=[];missing=[]
    for index,row in enumerate(parts):
        if row['name']!='part_%04d.bin'%index:raise ValueError('Noncanonical part order')
        path=directory/row['name']
        if path.is_symlink() or path.resolve().parent!=directory:raise ValueError('Unsafe part')
        if not path.exists() or path.stat().st_size!=row['size']:
            missing.append(row['name']);continue
        if sha(path)!=row['sha256']:raise ValueError('Corrupt part '+row['name'])
        verified.append(row['name'])
    receipt=manifest['source_receipt'];destination=directory.parent/Path(receipt['archive']).name
    if sum(row['size'] for row in parts)!=receipt['size']:raise ValueError('Invalid total size')
    result={'verified_parts':verified,'pending_parts':missing,'assembled':False,
            'source_manifest_sha256':expected_manifest,'utc':datetime.now(timezone.utc).isoformat()}
    if assemble:
        if missing:raise ValueError('Incomplete transport')
        if destination.exists() or destination.with_suffix('.receipt.json').exists():
            raise ValueError('Never replace existing archive/receipt')
        temporary=directory/(destination.name+'.assembling')
        h=hashlib.sha256();count=0
        with temporary.open('xb') as output:
            for row in parts:
                with (directory/row['name']).open('rb') as stream:
                    for data in iter(lambda:stream.read(1024*1024),b''):
                        h.update(data);count+=len(data);output.write(data)
        if count!=receipt['size'] or h.hexdigest()!=receipt['sha256']:
            raise ValueError('Reconstructed archive differs from source')
        # Hard-link creation is atomic and refuses to overwrite any existing path.
        destination.hardlink_to(temporary)
        with destination.with_suffix('.receipt.json').open('x') as output:json.dump(receipt,output,indent=2)
        temporary.unlink()
        result.update(assembled=True,archive=str(destination),sha256=h.hexdigest(),bytes=count,
                      independent_archive_audit_still_required=True)
        with (directory/'ASSEMBLED.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('expected_manifest')
    p.add_argument('--assemble',action='store_true');a=p.parse_args();main(a.directory,a.expected_manifest,a.assemble)
