"""Read-only VLA-aware measured loss audit using frozen physical thresholds.

The original native dropped flag only detects some falls. A grounded object
left on the table is not necessarily a native terminal. These derived records
never overwrite historical outcomes or inject a scripted execution phase.
"""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from common import BASE,read,sha

SOURCE=BASE/'current_runtime_grasp_drop_candidate_20260905/measured_grasp_drop.py'
spec=importlib.util.spec_from_file_location('vla_measured_drop_original_thresholds',SOURCE)
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)

def diagnose(job):
    job=Path(job);traces=read(job/'BRANCH_TRACE.json');result=read(job/'BRANCH_RESULT.json')
    plan=result['plan'];decision=read(job/'DECISION_METADATA.json');z0=decision['object_pose'][2]
    # Only a mask for the inherited proof's pre-release domain, never a motion
    # schedule. All pre-release observations get the same accepted dummy name.
    domain=[{**t,'phase':'release' if t['vla_release_intent'] else 'lift'} for t in traces]
    proof=old.proof(domain,z0)
    info=next(x for x in read(BASE/'current_runtime_recovery_v2_20260905/TASK_GEOMETRY_MANIFEST.json')['tasks'] if x['task']==plan['task'])
    geom=old.label_module.geom
    with np.load(info['label_geometry']) as f:vertices=f['vertices'];region=geom.Region(f['basket_from_site'],f['half_size'])
    disjoint=False;checks=[]
    if proof['proven']:
        for frame in proof['proof_frames']:
            t=traces[frame['step']-1]
            g=geom.containment(region,geom.transform(t['basket_pose_w']),geom.transform(t['object_pose_w']),vertices)
            lo=np.asarray(g['mesh_min_site'])[:2];hi=np.asarray(g['mesh_max_site'])[:2]
            separate=bool(np.any((hi < -region.half_size[:2]) | (lo > region.half_size[:2])))
            checks.append({'step':t['branch_step'],'whole_mesh_xy_disjoint':separate})
        disjoint=all(x['whole_mesh_xy_disjoint'] for x in checks)
    pre=[t for t in traces if not t['vla_release_intent']]
    first_release=next((t['branch_step'] for t in traces if t['vla_release_intent']),None)
    proper_prefix=traces[:first_release-1] if first_release is not None else traces
    slip=np.asarray([t['object_relative_translation_m'] for t in proper_prefix])
    return {'version':'ONLINE_VLA_RELEASE_MASK_MEASURED_DROP_DIAGNOSTIC_V1','source_sha256':sha(SOURCE),
        'label_overwritten':False,'physical_actions_changed':False,'scripted_execution_phase_used':False,
        'native_drop':result['outcome'].get('dropped'), 'measured_drop_outside_target':bool(proof['proven'] and disjoint),
        'physical_proof':proof,'target_exclusion_checks':checks,
        'first_vla_release_step':first_release,'pre_release_relative_motion_max_m':float(np.linalg.norm(slip-slip[0],axis=1).max()) if len(slip) else None,
        'max_measured_bilateral_squeeze':max(t['measured_bilateral_squeeze'] for t in traces),
        'mean_pre_release_bilateral_squeeze':float(np.mean([t['measured_bilateral_squeeze'] for t in proper_prefix])) if proper_prefix else None,
        'force_causality_alone_proven':False}

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('job');a=p.parse_args();print(json.dumps(diagnose(a.job),indent=2))
