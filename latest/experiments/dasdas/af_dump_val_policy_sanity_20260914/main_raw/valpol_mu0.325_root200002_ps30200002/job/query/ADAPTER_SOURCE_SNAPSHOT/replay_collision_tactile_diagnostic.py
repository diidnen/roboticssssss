"""Compare depth-acquisition sources on saved REAL scene frames, no AF labels.

Collision raycasts interrogate the actual simulator geometry. They are not
claimed equivalent to the original renderer or approved as model input here.
"""
from pathlib import Path
import json
import numpy as np
from original_tactile_core import OriginalTactileCore

root = Path(__file__).parent
folder = root / 'tactile_acquisition_v1'
core = OriginalTactileCore()
rows = []
for tag in ('scripted_grasp', 'preload_250', 'preload_500', 'open_negative_control'):
    data = np.load(folder / (tag + '_sensor.npz'))
    raw = data['collision_axial_depth_m']
    depth = np.where((raw >= .024) & (raw <= .029), raw, .029) * 1000
    markers = core.observe(depth, data['relative_positions_m'], data['relative_quaternions_wxyz'],
                           source='PHYSX_COLLISION_DEPTH_DIAGNOSTIC_NOT_QUALIFIED')
    row = {'tag': tag, 'depth_min_mm': depth.min(axis=(1,2)).tolist(),
           'marker_displacement_max_px': np.linalg.norm(markers[:,1]-markers[:,0],axis=-1).max(axis=1).tolist(),
           'visible_contact_fraction': (depth < 28.5).mean(axis=(1,2)).tolist(),
           'formal_collection_gate_passed': False}
    rows.append(row)
    np.savez_compressed(folder / (tag + '_collision_marker_diagnostic.npz'), markers=markers, depth_mm=depth)
    print(json.dumps(row), flush=True)
(root / 'COLLISION_TACTILE_DIAGNOSTIC.json').write_text(json.dumps(rows, indent=2))
