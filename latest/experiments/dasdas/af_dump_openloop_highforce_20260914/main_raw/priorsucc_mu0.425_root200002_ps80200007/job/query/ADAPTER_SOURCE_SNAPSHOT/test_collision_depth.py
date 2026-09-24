"""Analytical raycast tests; no task data and no changes to production scene."""
import json
from pathlib import Path
import numpy as np
import sapien
from scipy.spatial.transform import Rotation
import af_native_joint_readback as native

scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[.001, .01, .01])
target = builder.build_static('depth_target')
component = target.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[.001, .01, .01])
occluder = builder.build_static('excluded_actor')
rows = []
for angle in (0., .7, 1.8):
    rotation = Rotation.from_euler('z', angle)
    matrix = rotation.as_matrix()
    shift = np.array([.23, -.15, .44])
    quat = rotation.as_quat()[[3,0,1,2]]
    target.set_pose(sapien.Pose(matrix @ np.array([.029,0,0]) + shift, quat))
    occluder.set_pose(sapien.Pose(matrix @ np.array([.015,0,0]) + shift, quat))
    local = np.array([[1,0,0], [1,.1,.1], [1,1,0]], dtype=float)
    lengths = np.linalg.norm(local,axis=1)
    directions = (local/lengths[:,None]) @ matrix.T
    origins = np.tile(shift,(3,1))
    before = scene.physx_system.pack()
    hit = native.target_rays(component, origins, directions, .08)
    expected = np.array([.028, .028*lengths[1], -1])
    error = float(np.max(abs(hit-expected)))
    rows.append({'angle_rad': angle, 'hit_distances_m': hit.tolist(), 'expected_m': expected.tolist(),
                 'max_error_m': error, 'physics_unchanged': before == scene.physx_system.pack(),
                 'excluded_nearer_actor': True})
report = {'scope': 'analytical geometry UNIT TEST; not AF collection', 'rows': rows,
          'passed': all(r['max_error_m'] < 1e-6 and r['physics_unchanged'] for r in rows)}
Path(__file__).with_name('COLLISION_DEPTH_UNIT_TEST.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert report['passed']
