"""Independently recompute online label from saved poses/contact/action traces.

Uses authored whole-mesh geometry and observed release, never a downstream
phase annotation. Does not overwrite rollout labels or native drop flags.
"""
import argparse,json
from pathlib import Path
import numpy as np
from common import BASE,read,sha
from drop_diagnostics import old,diagnose

def audit(job):
    job=Path(job);result=read(job/'BRANCH_RESULT.json');trace=read(job/'BRANCH_TRACE.json')
    meta=read(job/'DECISION_METADATA.json');reported=result['outcome'];task=result['plan']['task']
    info=next(r for r in read(BASE/'current_runtime_recovery_v2_20260905/TASK_GEOMETRY_MANIFEST.json')['tasks'] if r['task']==task)
    if sha(Path(info['label_geometry']))!=info['label_sha256']:raise RuntimeError('Authored geometry changed')
    geom=old.label_module.geom
    with np.load(info['label_geometry']) as f:vertices=f['vertices'];region=geom.Region(f['basket_from_site'],f['half_size'])
    checks=[geom.containment(region,geom.transform(t['basket_pose_w']),geom.transform(t['object_pose_w']),vertices) for t in trace[-50:]]
    inside=len(checks)==50 and all(x['whole_mesh_inside_authored_obb'] for x in checks)
    opened=len(trace)>=20 and all(t['vla_release_intent'] and t['final_gripper_command']==float(np.float32(.04)) for t in trace[-20:])
    unheld=len(trace)>=20 and all(max(t['finger_object_contact_norms_N'])<=.05 for t in trace[-20:])
    support=bool(trace and trace[-1]['basket_contact_force_N']>.05)
    lift=any(t['object_position_m'][2]-meta['object_pose'][2]>=.03 for t in trace)
    drop=any(t['terminations'].get('object_1_dropped',False) for t in trace)
    timeout=any(t['terminations'].get('time_out',False) for t in trace)
    reset=any(t['physical_reset_requested'] for t in trace)
    complete=len(trace)==350
    valid=bool(trace) and result['error'] is None and (complete or reset)
    expected=dict(label_valid=valid,full_task_success_y=int(lift and not drop and not timeout and not reset and complete and inside and opened and unheld and support) if valid else None,
        lift_success=int(lift),place_success=int(inside and opened and unheld and support),inside_last50=inside,opened_last20=opened,unheld_last20=unheld,final_support_contact=support,dropped=int(drop))
    mismatches={k:{'reported':reported.get(k),'recomputed':v} for k,v in expected.items() if reported.get(k)!=v}
    measured=diagnose(job)
    return dict(passed=not mismatches,mismatches=mismatches,recomputed=expected,scripted_downstream_phase_used=False,
        geometry_source_sha256=info['label_sha256'],trace_sha256=sha(job/'BRANCH_TRACE.json'),
        implementation_sha256=sha(Path(__file__)),label_overwritten=False,
        independent_measured_drop=measured['measured_drop_outside_target'],
        primary_drop_flag_scope='Frozen native terminal flag preserved. Measured pre-release detachment is reported separately; never equate native false with no physical drop.',
        potential_recovery_case=bool(expected['full_task_success_y'] and measured['measured_drop_outside_target']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');a=p.parse_args();print(json.dumps(audit(a.job),indent=2))
