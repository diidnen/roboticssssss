"""Full terminal raw preservation of the canonical-ready intervention."""
import hashlib
import io
import json
import tarfile
from rootlocal_collection_contract import HERE, read, sha, now


def main():
    roots=[HERE/'mu060_canonical_ready_diagnostics_v1',HERE/'canonical_ready_online_gate_v1']
    for root in roots:
        if not read(root/'COMPLETE.json')['completed']:raise ValueError('Nonterminal evidence')
    if not read(roots[1]/'INDEPENDENT_ENGINEERING_AUDIT.json')['passed']:raise ValueError('Unaudited gate')
    files=set(p for root in roots for p in root.rglob('*') if p.is_file())
    files.update(HERE.glob('*.py'));files.update(HERE.glob('*.md'))
    destination=HERE.parent/'af_dump_canonical_ready_evidence_20260913_v1.tar.gz'
    manifest={'created_utc':now(),'formal_labels_added':0,'files':{}}
    with destination.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for path in sorted(files):
                content=path.read_bytes();name=str(path.relative_to(HERE))
                info=tarfile.TarInfo(name);info.size=len(content);tar.addfile(info,io.BytesIO(content))
                manifest['files'][name]={'size':len(content),'sha256':hashlib.sha256(content).hexdigest()}
            content=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(content)
            tar.addfile(info,io.BytesIO(content))
    receipt={'archive':str(destination),'size':destination.stat().st_size,'sha256':sha(destination),'files':len(files)}
    with destination.with_suffix('.receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
