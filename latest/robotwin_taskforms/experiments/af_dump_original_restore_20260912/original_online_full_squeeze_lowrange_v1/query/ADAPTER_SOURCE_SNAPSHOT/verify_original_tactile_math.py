"""Signed-angle regression and offline replay; preserves every old raw frame."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from original_tactile_core import OriginalTactileCore,euler_xyz_from_quat,ORIGINAL_MATH

root=Path(__file__).parent
out=root/'tactile_original_math_replay_v2'
out.mkdir(exist_ok=False)
rng=np.random.default_rng(912)
rotations=Rotation.random(1024,random_state=rng)
quats=rotations.as_quat()[:,[3,0,1,2]]
angles=np.stack([x.numpy() for x in euler_xyz_from_quat(torch.tensor(quats,dtype=torch.float64))],axis=1)
reconstructed=Rotation.from_euler('xyz',angles).as_matrix()
error=float(np.max(abs(reconstructed-rotations.as_matrix())))
negative=Rotation.from_euler('z',-.3).as_quat()[[3,0,1,2]][None]
yaw=float(euler_xyz_from_quat(torch.tensor(negative))[2][0])
assert abs(yaw+.3)<1e-12 and error<1e-12
core=OriginalTactileCore()
folder=root/'tactile_dynamic_v1'
record=json.loads((folder/'qualification.json').read_text())
rows=[]
for capture in record['captures']:
    tag=capture['tag']
    with np.load(folder/(tag+'_sensor.npz')) as data:
        raw=data['collision_axial_depth_m']
        depth=np.where((raw>=.024)&(raw<=.029),raw,.029)*1000
        markers=core.observe(depth,data['relative_positions_m'],data['relative_quaternions_wxyz'],
                             source='PHYSX_COLLISION_DEPTH_DIAGNOSTIC_NOT_QUALIFIED')
        old=data['collision_markers_diagnostic']
        row={'tag':tag,'marker_max_delta_from_old_provisional_math_px':float(abs(markers-old).max()),
             'deformation_max_px':np.linalg.norm(markers[:,1]-markers[:,0],axis=-1).max(axis=1).tolist()}
    np.savez_compressed(out/(tag+'_markers.npz'),markers=markers)
    rows.append(row)
report={'scope':'unchanged saved real sensor inputs, corrected original math bridge; NOT AF task data',
        'formal_collection_gate_passed':False,'rotation_roundtrip_error':error,'negative_yaw':yaw,
        'original_math_source':str(ORIGINAL_MATH),
        'original_math_sha256':hashlib.sha256(ORIGINAL_MATH.read_bytes()).hexdigest(),
        'source_hashes':core.source_hashes,'frames':rows}
(out/'qualification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'frames':len(rows),'rotation_roundtrip_error':error,'negative_yaw':yaw,
    'max_delta_px':max(row['marker_max_delta_from_old_provisional_math_px'] for row in rows)}))
