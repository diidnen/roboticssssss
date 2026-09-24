import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

root = Path(__file__).parent
report = []
for tag in ('scripted_grasp', 'preload_250', 'preload_500', 'open_negative_control'):
    folder = root / 'tactile_acquisition_v1'
    record = json.loads((folder / (tag + '.json')).read_text())
    arrays = np.load(folder / (tag + '_sensor.npz'))
    for i, finger in enumerate(record['contacts']['fingers']):
        pose = record['frames'][i]['finger_pose']
        rotation = Rotation.from_quat(np.array(pose[3:])[[1,2,3,0]]).as_matrix()
        points = [(np.array(p['position_world'])-pose[:3]) @ rotation for p in finger['points'] if p['is_target']]
        physical = arrays['collision_axial_depth_m'][i]
        render = arrays['depth_mm'][i] / 1000
        both = (physical > .024) & (physical < .029) & (render < .029)
        report.append({'tag': tag, 'finger': finger['finger'], 'target_contacts_local': [p.tolist() for p in points],
            'normal_squeeze_n': record['contacts']['measured_squeeze_n'],
            'physical_min_positive_mm': float(physical[physical>0].min()*1000) if (physical>0).any() else None,
            'render_min_mm': float(render.min()*1000),
            'median_render_minus_collision_mm': float(np.median(render[both]-physical[both])*1000) if both.any() else None})
print(json.dumps([{k:v for k,v in r.items() if k != 'target_contacts_local'} for r in report], indent=2))
(root / 'TACTILE_GEOMETRY_REVIEW.json').write_text(json.dumps(report, indent=2))

sys.path.insert(0, str(root / 'sensor_dependencies'))
from pxr import Usd, UsdGeom
stage = Usd.Stage.Open('/home/exouser/Tabero/source/tac_manip/tac_manip/assets/data/Robots/Franka_gsmini/physx_rigid_gelpads.usd')
attributes = []
for prim in stage.Traverse():
    if 'gel' not in str(prim.GetPath()).lower(): continue
    values = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
              if any(k in a.GetName().lower() for k in ('physics', 'extent', 'xformop'))}
    if values: attributes.append({'path': str(prim.GetPath()), 'attributes': values})
(root / 'ORIGINAL_GEL_PHYSICS.json').write_text(json.dumps(attributes, indent=2))
cache = UsdGeom.XformCache()
frames = []
for side in ('left', 'right'):
    gel = stage.GetPrimAtPath('/panda/gelpad_' + side)
    cam = stage.GetPrimAtPath('/panda/gelsight_mini_case_' + side + '/Camera')
    transform = cache.GetLocalToWorldTransform(gel) * cache.GetLocalToWorldTransform(cam).GetInverse()
    frames.append({'side': side, 'gel_in_opengl_camera': [list(r) for r in transform]})
(root / 'ORIGINAL_GEL_CAMERA_FRAMES.json').write_text(json.dumps(frames, indent=2))
