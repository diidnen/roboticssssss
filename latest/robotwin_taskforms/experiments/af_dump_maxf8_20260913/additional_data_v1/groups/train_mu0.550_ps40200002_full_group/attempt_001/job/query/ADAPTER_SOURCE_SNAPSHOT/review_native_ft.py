"""Review provisional native readback, without producing AF inputs or labels."""
import argparse
import json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('directory', type=Path)
args = ap.parse_args()
results = []
for path in sorted(args.directory.glob('hold_*.json')):
    trace = json.loads(path.read_text())
    measurements = []
    for row in trace[len(trace)//2:]:
        for f in row['fingers']:
            ft = f['native_joint_readback']
            total = np.array(ft['unqualified_contact_balance_world_n'])
            normal = np.array(f['target_normal_vector_world_n'])
            tangent = total - normal
            norm = np.linalg.norm(normal)
            axis = normal / norm if norm > 1e-8 else np.zeros(3)
            measurements.append([norm, np.linalg.norm(total), np.linalg.norm(tangent),
                                 abs(np.dot(tangent, axis)),
                                 np.linalg.norm(ft['com_linear_acceleration'])])
    x = np.asarray(measurements)
    results.append({'file': path.name, 'tail_finger_samples': len(x),
                    'columns': ['normal_impulse_force_norm', 'provisional_total_balance_norm',
                                'provisional_tangent_norm', 'normal_component_error', 'com_acceleration_norm'],
                    'mean': x.mean(0).tolist(), 'median': np.median(x, axis=0).tolist(),
                    'p95': np.percentile(x, 95, axis=0).tolist(),
                    'not_yet_qualified_as_original_af_sensor': True})
report = {'scope': 'provisional sensor engineering only', 'rows': results}
(args.directory / 'FT_READBACK_REVIEW.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
