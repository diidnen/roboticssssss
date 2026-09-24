"""Read-only stability consistency diagnostic, NOT a replacement label.

Compare instantaneous simulator velocity with measured 20 Hz pose evolution.
Net pose motion cannot rule out sub-frame oscillation; a disagreement is an
adjudication flag, never permission to silently promote a frozen failed label.
"""
import numpy as np
from activeforcing_placement_contract import CONFIG,rotation


def audit_stability(trace,dt=.05):
    count=CONFIG['stable_window_steps']
    if not np.isfinite(dt) or dt<=0 or len(trace)<count+1:raise ValueError('Missing time window')
    tail=trace[-count-1:]
    steps=np.asarray([t['episode_step'] for t in tail])
    if not np.all(np.diff(steps)==1):raise ValueError('Non-contiguous observation clock')
    poses=np.asarray([t['object_pose_w'] for t in tail],float)
    if poses.shape!=(count+1,7) or not np.isfinite(poses).all():raise ValueError('Invalid poses')
    q=poses[:,3:];norm=np.linalg.norm(q,axis=1)
    if np.any(norm<1e-8):raise ValueError('Invalid quaternion')
    q=q/norm[:,None]
    speed=np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1)/dt
    angular=2*np.arccos(np.clip(np.abs((q[:-1]*q[1:]).sum(axis=1)),0,1))/dt
    relative=np.array([(np.array(t['object_pose_w'][:3])-np.array(t['basket_pose_w'][:3]))@
        rotation(t['basket_pose_w'][3:]) for t in tail[1:]])
    position_range=float(np.linalg.norm(np.ptp(relative,axis=0)))
    raw_linear=np.array([np.linalg.norm(t['object_velocity_w'][:3]) for t in tail[1:]])
    raw_angular=np.array([np.linalg.norm(t['object_velocity_w'][3:]) for t in tail[1:]])
    if not np.isfinite(raw_linear).all() or not np.isfinite(raw_angular).all():raise ValueError('Invalid raw velocity')
    pose_stable=bool(speed.max()<=CONFIG['maximum_linear_speed_m_s'] and angular.max()<=CONFIG['maximum_angular_speed_rad_s']
        and position_range<=CONFIG['maximum_relative_position_range_m'])
    raw_stable=bool(raw_linear.max()<=CONFIG['maximum_linear_speed_m_s'] and raw_angular.max()<=CONFIG['maximum_angular_speed_rad_s']
        and position_range<=CONFIG['maximum_relative_position_range_m'])
    return {'pose_clock_dt_s':dt,'pose_intervals':count,'max_pose_derived_linear_speed_m_s':float(speed.max()),
        'max_pose_derived_angular_speed_rad_s':float(angular.max()),'max_raw_linear_speed_m_s':float(raw_linear.max()),
        'max_raw_angular_speed_rad_s':float(raw_angular.max()),'relative_position_range_m':position_range,
        'pose_motion_within_frozen_numeric_thresholds':pose_stable,'raw_velocity_within_frozen_thresholds':raw_stable,
        'instantaneous_velocity_only_rejection':bool(pose_stable and not raw_stable),
        'label_rewritten':False,'new_success_label':None,
        'caveat':'20Hz net pose movement is not an independent measurement of within-step high-frequency oscillation; discrepancy requires label-contract adjudication'}
