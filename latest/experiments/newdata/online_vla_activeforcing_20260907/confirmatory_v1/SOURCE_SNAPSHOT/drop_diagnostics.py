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

def causal_drop_proof(trace,z0):
    """Same frozen measurements/thresholds; raw VLA open intent ends domain.

    There is deliberately no phase field, phase builder or synthetic stage.
    Proof uses current and past observations only, reported after the fifth
    proving frame, never backdated to a future-informed event time.
    """
    no={'proven':False,'contract':'ONLINE_OBSERVABLE_PRE_RELEASE_CONTACT_LOSS_V2','reason':'INSUFFICIENT_MEASURED_PROOF'}
    if not trace or not np.isfinite(z0):return no
    peak=float(z0);prior=False;window=[]
    for expected,t in enumerate(trace,1):
        if t.get('branch_step')!=expected:return {**no,'reason':'INVALID_BRANCH_CLOCK'}
        if t['vla_release_intent']:break
        z=float(t['object_position_m'][2]);contact=np.asarray(t['finger_object_contact_norms_N'],float)
        relative=np.asarray(t['object_relative_translation_m'],float)
        if contact.shape!=(2,) or relative.shape!=(3,) or not np.isfinite(np.r_[z,contact,relative]).all() or np.any(contact<0):
            return {**no,'reason':'INVALID_MEASURED_VALUES'}
        if t.get('physical_reset_requested'):return {**no,'reason':'NATIVE_TERMINAL'}
        peak=max(peak,z)
        if z-z0>=old.LIFT_M and np.all(contact>=old.PRIOR_CONTACT_N):prior=True
        distance=float(np.linalg.norm(relative));fall=peak-z
        qualifies=prior and peak-z0>=old.LIFT_M and fall>=old.FALL_FROM_PEAK_M and np.all(contact<=old.CONTACT_ZERO_N) and distance>old.DETACHED_DISTANCE_M
        frame=dict(step=expected,object_height_m=z,height_fall_from_peak_m=fall,normal_contact_norms_N=contact.tolist(),object_relative_distance_m=distance)
        window=window+[frame] if qualifies else []
        if len(window)>=old.CONTACT_WINDOW:
            return dict(proven=True,contract=no['contract'],event='MEASURED_PRE_RELEASE_GRASP_DROP',event_step=expected,
                window_start_step=window[-old.CONTACT_WINDOW]['step'],initial_object_height_m=z0,peak_object_height_m=peak,
                lift_height_m=peak-z0,prior_lifted_bilateral_contact=True,proof_frames=window[-old.CONTACT_WINDOW:],
                force_causality_proven=False,native_root_height_drop_flag_used=False,causal=True)
    return no

def diagnose(job):
    job=Path(job);traces=read(job/'BRANCH_TRACE.json');result=read(job/'BRANCH_RESULT.json')
    plan=result['plan'];decision=read(job/'DECISION_METADATA.json');z0=decision['object_pose'][2]
    proof=causal_drop_proof(traces,z0)
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
    return {'version':'ONLINE_VLA_OBSERVABLE_MEASURED_DROP_DIAGNOSTIC_V2','threshold_source_sha256':sha(SOURCE),
        'implementation_sha256':sha(Path(__file__)),'synthetic_phase_fields_used':False,
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
