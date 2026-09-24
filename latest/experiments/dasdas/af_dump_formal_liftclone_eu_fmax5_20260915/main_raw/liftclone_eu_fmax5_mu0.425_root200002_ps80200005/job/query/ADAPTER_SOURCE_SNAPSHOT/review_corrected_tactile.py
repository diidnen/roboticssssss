"""Compact readout for engineering gates, never manufactures AF task labels."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).parent
folder=root/'tactile_dynamic_original_math_v2'
report=json.loads((folder/'qualification.json').read_text())
trace=json.loads((folder/'dynamic_contacts.json').read_text())
phases={}
for phase in dict.fromkeys(r['phase'] for r in trace):
    rows=[r for r in trace if r['phase']==phase]
    valid=[r for r in rows if 'native_aggregate_patch_readback' in r]
    friction=[np.linalg.norm(f['sum_friction_vector_world']) for r in valid
              for f in r['native_aggregate_patch_readback']['patches']['friction']]
    phases[phase]={'physics_samples':len(rows),'qualified_aggregate_readouts':len(valid),
                   'measured_squeeze_mean_n':float(np.mean([r['measured_squeeze_n'] for r in rows])),
                   'per_finger_friction_norm_p95_n':float(np.percentile(friction,95)) if friction else None}
open_fr=report['open_force_negative_control']['patches']['friction']
open_error=max(abs(np.array([r['sum_friction_vector_world'] for r in open_fr])).ravel())
open_capture=report['captures'][-1]
summary={'scope':'engineering ONLY; no training or inference score',
         'completed':report['completed'],'frames':len(report['captures']),'phases':phases,
         'readout_rejections':[r['patch_readback_rejected'] for r in trace if 'patch_readback_rejected' in r],
         'open_force_component_abs_max_n':float(open_error),
         'open_force_negative_control_passed':bool(open_error<.001),
         'open_collision_marker_max_px':open_capture['collision_marker_displacement_max_px'],
         'formal_collection_gate_passed':False}
(folder/'ENGINEERING_READOUT.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
