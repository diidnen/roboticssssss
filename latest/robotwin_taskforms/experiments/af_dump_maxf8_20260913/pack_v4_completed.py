"""Backup closed V4 metadata or one independently accepted case, never live traces."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
HERE = Path(__file__).resolve().parent
ARCHIVES = Path('/media/volume/newdata/exouser/af_dump_maxf8_archives_20260913')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(case):
    files = set(HERE.glob('*.py')) | set(HERE.glob('*.md'))
    files.update((HERE/'additional_data_v1').glob('*.json'))
    if case:
        path = HERE/'additional_data_v1/groups'/case/'attempt_001'
        if not path.resolve().is_relative_to(HERE/'additional_data_v1/groups'):
            raise ValueError('Invalid case path')
        accepted=json.loads((path/'ACCEPTED.json').read_text())
        if not accepted['accepted']: raise ValueError('Case not accepted')
        for name,digest in accepted['artifact_hashes'].items():
            if sha(Path(name))!=digest:raise ValueError('Accepted artifact changed')
        files.update(p for p in path.rglob('*') if p.is_file())
    else:
        for name,gate in [('belief_convergence_400_v1','DIAGNOSTIC_RESULT.json'),
                          ('utility_only_runtime_v1','RUNTIME_PARITY.json')]:
            if not (HERE/name/gate).exists():raise ValueError('Metadata stage incomplete')
            files.update(p for p in (HERE/name).rglob('*') if p.is_file())
        files.add(HERE/'CORRECTED_SEALED_CURVES.json')
        plan=HERE/'confirmation_plan_v1'
        if plan.exists(): files.update(plan.glob('*.json'))
        for name in ['training_driver_v1/STAGE_LOCK.json','confirmation_driver_v1/STAGE_LOCK.json',
                     'collection_driver_v1/PROCESS.json','training_supervisor_launch_v1/PROCESS.json',
                     'confirmation_supervisor_launch_v1/PROCESS.json','VERIFIED_REDUNDANT_ARCHIVES.json']:
            path=HERE/name
            if path.exists(): files.add(path)
    ARCHIVES.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=ARCHIVES/('af_dump_maxf8_'+(case or 'metadata')+'_'+stamp+'.tar.gz')
    manifest={'case':case,'partial_run':True,'files':{}}
    with destination.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for path in sorted(files):
            if path.is_symlink():raise ValueError('Symlink needs explicit backup handling')
            before=path.stat(); data=path.read_bytes();after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed')
            name=HERE.name+'/'+str(path.relative_to(HERE));info=tarfile.TarInfo(name);info.size=len(data)
            archive.addfile(info,io.BytesIO(data));manifest['files'][name]={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    result={'archive':str(destination),'size':destination.stat().st_size,'sha256':sha(destination),'files':len(files),'case':case}
    with destination.with_suffix('.receipt.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case');main(parser.parse_args().case)
