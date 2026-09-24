import json
from pathlib import Path
import numpy as np

root = Path(__file__).parent
folder = root / 'tactile_dynamic_v1'
data = json.loads((folder / 'dynamic_contacts.json').read_text())
qualification = json.loads((folder / 'qualification.json').read_text())
summary = {'scope': 'engineering readout, NOT full-task/AF result',
           'completed': qualification['completed'],
           'sensor_frames': len(qualification['captures']), 'physics_steps': len(data),
           'formal_collection_gate_passed': False, 'phases': []}
for phase in dict.fromkeys(row['phase'] for row in data):
    rows = [r for r in data if r['phase'] == phase]
    squeeze = np.array([r['measured_squeeze_n'] for r in rows])
    discrepancy, other_load = [], []
    for row in rows:
        normal = np.array(row['normal_axis_world'])
        for finger in row['fingers']:
            balance = np.array(finger['native_joint_readback']['unqualified_contact_balance_world_n'])
            projected = np.array(finger['force_world_n'])
            discrepancy.append(abs(np.dot(balance-projected, normal)))
            other_load.append(sum(np.linalg.norm(p['impulse_ns']) / .004
                                  for p in finger['points'] if not p['is_target']))
    summary['phases'].append({'phase': phase, 'samples': len(rows),
        'squeeze_mean_n': float(squeeze.mean()), 'squeeze_std_n': float(squeeze.std()),
        'bilateral_positive_force_fraction': float(np.mean(squeeze > .2)),
        'normal_balance_discrepancy_p95_n': float(np.percentile(discrepancy,95)),
        'non_target_impulse_force_max_n': float(max(other_load))})
negative = qualification['captures'][-1]
summary['open_marker_max_px'] = negative['collision_marker_displacement_max_px']
summary['all_captures_preserve_physics'] = all(r['physics_unchanged_by_capture'] for r in qualification['captures'])
(root / 'TACTILE_DYNAMIC_REVIEW.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
