"""Create a scoped archive of new sources/diagnostics, never alter raw results."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).parent
target=root.parent/'af_dump_sensor_bridge_evidence_20260912.tar.gz'
if target.exists():raise FileExistsError(target)
files=[]
for item in sorted(root.iterdir()):
    if item.is_file() and item.suffix in ('.py','.cpp','.md','.json','.log'):
        files.append(item)
    elif item.is_dir() and (item.name.startswith('force_') or item.name in
                            ('tactile_acquisition_v1','tactile_acquisition_v2','tactile_dynamic_v1')):
        files.extend(p for p in sorted(item.rglob('*')) if p.is_file() and not p.is_symlink())
manifest={str(p.relative_to(root)):{'size':p.stat().st_size,
          'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
manifest_file=root/'SENSOR_BRIDGE_EVIDENCE_MANIFEST.json'
manifest_file.write_text(json.dumps({'scope':'engineering diagnostics; NOT formal AF task results',
    'files':manifest},indent=2))
with tarfile.open(target,'w:gz') as archive:
    for path in files+[manifest_file]:
        archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
print(json.dumps({'archive':str(target),'size':target.stat().st_size,
                  'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'files':len(files)+1}))
