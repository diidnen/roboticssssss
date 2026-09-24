"""Read-only causal event diagnostics; no scripted downstream stage input."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from common import read,sha

def probe_shift(job):
    raw=list(csv.DictReader((job/'RAW_PROBE.csv').open()))
    rec=read(job/'CONTACT_PATCH_READBACK.json')
    begin=next(i for i,r in enumerate(raw) if r['probe_phase'].startswith('probe_'))
    # Labels select the physical probe interval only. They are not downstream
    # stage features or downstream failure-stage annotations.
    a,b=rec[begin-1],rec[-1]
    pos=lambda r:np.asarray(r['object_position']).reshape(-1)[:3]
    eef=lambda r:np.asarray(r['eef_pose']).reshape(-1)[:3]
    quat=lambda r:np.asarray(r['object_quaternion']).reshape(-1)[:4]
    dot=abs(float(np.dot(quat(a),quat(b))/(np.linalg.norm(quat(a))*np.linalg.norm(quat(b)))))
    return dict(pre_probe_step=a['step'],post_probe_step=b['step'],object_translation_m=float(np.linalg.norm(pos(b)-pos(a))),
        object_rotation_rad=float(2*np.arccos(np.clip(dot,0,1))),eef_translation_m=float(np.linalg.norm(eef(b)-eef(a))),
        aperture_change_m=b['aperture_m']-a['aperture_m'],
        bilateral_force_before=a['target_object_force']['F_obj_bilateral_n'],
        bilateral_force_after=b['target_object_force']['F_obj_bilateral_n'],
        image_difference='UNAVAILABLE: pre-probe images were not saved by historical frozen core',
        tactile_state_difference='Force/contact differences measured; full pre-probe model tactile image input not archived',
        interpretation='Physical perturbation magnitude only; cannot by itself prove model distribution-shift acceptability.')

def audit(job):
    job=Path(job);tr=read(job/'BRANCH_TRACE.json');result=read(job/'BRANCH_RESULT.json')
    dec=read(job/'DECISION_METADATA.json');z0=dec['object_pose'][2]
    events=[];open_far=[];regrasps=[];anomalies=[];was_open=False;ever_lifted=False
    previous=None
    for t in tr:
        pos=np.asarray(t['object_position_m']);target=np.asarray(t['basket_pose_w'])[:3]
        relative=np.asarray(t['object_relative_translation_m']);contact=np.asarray(t['normal_force_N'])
        lifted=pos[2]-z0>=.03;ever_lifted|=lifted
        near=float(np.linalg.norm(pos[:2]-target[:2]))<.12
        held=bool(np.all(contact>=.15));opened=bool(t['vla_release_intent'])
        # A current/past-observation event state, not LIFT/TRANSPORT/etc labels.
        state=('TARGET_SUPPORTED_RELEASE' if opened and near and t['basket_contact_force_N']>.05 else
               'OPEN_INTENT_NEAR_TARGET' if opened and near else
               'OPEN_INTENT_AWAY_FROM_TARGET' if opened else
               'ELEVATED_BILATERAL_CONTACT' if lifted and held else
               'BILATERAL_CONTACT' if held else
               'CONTACT_LOST_AFTER_ELEVATION' if ever_lifted else 'NO_BILATERAL_CONTACT')
        if not events or events[-1]['state']!=state:events.append(dict(step=t['branch_step'],state=state))
        if opened and not near:open_far.append(t['branch_step'])
        if was_open and not opened and np.linalg.norm(relative)<.12 and not near:
            regrasps.append(t['branch_step'])
        arm=np.asarray(t['raw_vla_arm_command'])
        if previous is not None:
            displacement=float(np.linalg.norm(arm[:3]-previous[:3]))
            if displacement>.10:anomalies.append(dict(step=t['branch_step'],translation_jump_m=displacement))
        previous=arm;was_open=opened
    rpc=[read(p) for p in sorted((job/'RPC').glob('*.json'))]
    return dict(job=str(job),context=result['plan']['id'],method=read(job/'PLANNER_DECISION.json')['method'],
        outcome=result['outcome'],probe_shift=probe_shift(job),events=events,
        VLA_FAILURE_STAGE_LABEL_CAUSAL=True,scripted_downstream_stage_used=False,
        repeated_grasp_candidate_steps=regrasps,regrasp_candidate_present=bool(regrasps),
        early_regrasp_candidate_present=any(s<=50 for s in regrasps),
        raw_open_intent_away_target_steps=open_far,
        VLA_ACTION_ANOMALY_RATE=len(anomalies)/max(1,len(tr)-1),action_anomalies=anomalies,
        diagnostic_thresholds='Regrasp candidate: raw open->close, object within12cm of gripper, outside12cm target XY; requires visual review. Translation anomaly >10cm per .05s command transition. No invented semantic stage.',
        VLA_POLICY_STOCHASTIC=True,COMMON_RANDOM_NUMBERS_USED=True,
        policy_seeds=[dict(step=r['step'],noise_seed=r['noise_seed'],noise_sha256=r['noise_sha256'],
            action_sha256=r['action_sha256'],observation_sha256=r['observation_sha256']) for r in rpc],
        source_hashes={str(job/'BRANCH_TRACE.json'):sha(job/'BRANCH_TRACE.json'),str(Path(__file__)):sha(Path(__file__))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');a=p.parse_args();print(json.dumps(audit(a.job),indent=2))
