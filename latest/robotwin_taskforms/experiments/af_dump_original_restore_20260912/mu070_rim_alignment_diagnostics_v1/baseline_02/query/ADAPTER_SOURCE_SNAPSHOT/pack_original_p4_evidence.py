"""Snapshot only completed engineering evidence and exact current source bytes."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import argparse

ROOT=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--stage',choices=('original_p4','online_camera'),default='original_p4')
args=parser.parse_args()
NAME=('af_dump_original_p4_handoff_20260913' if args.stage=='original_p4'
      else 'af_dump_online_camera_engineering_20260913_v1')
target=ROOT.parent/(NAME+'.tar.gz')
if target.exists():raise FileExistsError(target)
directories=('fork_physics_replay_v1','fork_native_actions_v1','fork_observation_bridge_v1',
             'native_cartesian_kinematics_v1','tactile_original_math_replay_v2',
             'tactile_dynamic_original_math_v2','original_p4_native_v1',
             'force_canonical_actor_ids_v12','force_canonical_shape_ids_v13')
if args.stage=='online_camera':
    directories=('original_online_forks_v1','collision_surface_camera_v1',
                 'collision_surface_camera_v2','collision_surface_camera_v3')
files=[p for p in sorted(ROOT.iterdir()) if p.is_file() and p.suffix in ('.py','.cpp','.md','.json')]
for name in directories:
    path=ROOT/name
    if not path.is_dir():raise FileNotFoundError(path)
    files.extend(p for p in sorted(path.rglob('*')) if p.is_file() and not p.is_symlink())
manifest={'scope':'completed original P4, sensor and fork engineering; NOT formal AF inference',
          'stage':args.stage,'included_completed_directories':list(directories),
          'active_online_run_excluded':('original_online_forks_v1' if args.stage=='original_p4'
                                       else 'original_online_camera_5_8_8_v2'),'files':{}}
with tarfile.open(target,'w:gz') as archive:
    for path in files:
        data=path.read_bytes();relative=str(path.relative_to(ROOT))
        manifest['files'][relative]={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        entry=tarfile.TarInfo(relative);entry.size=len(data)
        archive.addfile(entry,io.BytesIO(data))
    data=json.dumps(manifest,indent=2).encode()
    entry=tarfile.TarInfo('P4_HANDOFF_ARCHIVE_MANIFEST.json');entry.size=len(data)
    archive.addfile(entry,io.BytesIO(data))
receipt={'archive':str(target),'bytes':target.stat().st_size,'files':len(files),
         'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
(ROOT/(NAME+'_receipt.json')).write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
