"""Archive completed controller evidence without touching live collection jobs."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent


def main():
    directories=['original_p4_full_squeeze_v4','original_online_camera_5_8_8_v2',
                 'original_online_full_squeeze_reference_v3','original_online_full_squeeze_lowrange_v1']
    for name in directories:
        folder=HERE/name
        receipt=folder/('online_qualification.json' if name.startswith('original_online') else 'qualification.json')
        data=json.loads(receipt.read_text())
        if name.startswith('original_online'):
            if len(data['results'])!=3 or any(r['kind']!='done' for r in data['results']):
                raise ValueError('Not a completed run: '+name)
        elif not data['completed']:raise ValueError('Not a completed query: '+name)
    files=set()
    for name in directories:files.update(p for p in (HERE/name).rglob('*') if p.is_file())
    for suffix in ('*.py','*.cpp','*.md','*.so'):files.update(HERE.glob(suffix))
    for pattern in ('original_p4_full_squeeze_v*.log','original_online_full_squeeze*.log'):
        files.update(HERE.glob(pattern))
    # Include immutable dataset locks, not partial live query/rollout outputs.
    dataset=HERE/'original_rootlocal_dataset_v1'
    files.update(p for p in dataset.glob('*') if p.is_file() and p.suffix in ('.json','.md'))
    files.update(p for p in (dataset/'SOURCE_SNAPSHOT').glob('*') if p.is_file())
    archive=HERE.parent/'af_dump_original_full_controller_20260913_v1.tar.gz'
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'completed engineering and prospective dataset freeze',
              'completed_directories':directories,'live_collection_rows_excluded':True,'files':{}}
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for path in sorted(files):
                content=path.read_bytes();name=str(path.relative_to(HERE))
                info=tarfile.TarInfo(name);info.size=len(content)
                tar.addfile(info,io.BytesIO(content))
                manifest['files'][name]={'size':len(content),'sha256':hashlib.sha256(content).hexdigest()}
            content=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(content)
            tar.addfile(info,io.BytesIO(content))
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest.hexdigest(),
                      'files':len(files)},indent=2))


if __name__=='__main__':main()
