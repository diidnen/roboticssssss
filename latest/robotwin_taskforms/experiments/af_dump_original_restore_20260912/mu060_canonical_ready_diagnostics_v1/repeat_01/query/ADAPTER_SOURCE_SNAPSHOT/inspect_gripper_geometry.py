import json
from pathlib import Path
import numpy as np
import trimesh

base = Path('/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin/assets/embodiments/ARX-X5/meshes')
for finger in ('link7', 'link8'):
    mesh = trimesh.load(base / (finger + '.STL'), force='mesh')
    print(json.dumps({'name': finger, 'bounds': mesh.bounds.tolist(), 'centroid': mesh.centroid.tolist(),
                      'vertex_quantiles': np.quantile(mesh.vertices, [0,.1,.5,.9,1], axis=0).tolist()}))
