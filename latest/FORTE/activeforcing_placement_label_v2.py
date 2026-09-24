"""Offline pilot label repair: geometric placement with observed pose stability.

v1's instantaneous PhysX velocity gate conflicts with nearly stationary pose
traces in sustained basket contact. v2 keeps every numeric tolerance and the
geometry/release/window requirements, but measures net motion from successive
poses on the verified observation clock. Raw v1 labels are NEVER overwritten.

This is a transparently post-start label-semantic repair, NOT a claim that v2
was pre-registered before all pilot observations or is admitted for training.
"""
from activeforcing_placement_contract import PHASE_COUNTS
from activeforcing_placement_stability_audit import audit_stability


def evaluate(trace,phase_counts,geometry):
    motion=audit_stability(trace)
    tail=trace[-20:]
    inside=all(geometry.containment(t['object_pose_w'],t['basket_pose_w'])['inside'] for t in tail)
    unheld=all(max(t['finger_object_contact_norms_N'])<=.05 for t in tail)
    complete=phase_counts==PHASE_COUNTS
    stable=motion['pose_motion_within_frozen_numeric_thresholds']
    return {'label_version':'GEOMETRIC_POSE_WINDOW_V2','place_success':int(complete and inside and unheld and stable),
        'all_phases_completed':complete,'all_final_window_inside':inside,'unheld':unheld,'settled_pose_window':stable,
        'motion':motion,'numeric_thresholds_changed':False,'physics_or_inputs_changed':False,
        'raw_v1_overwritten':False,'pre_registered_before_entire_pilot':False,'formal_training_admitted':False,
        'definition':'completed full horizon AND current-basket collision-surface containment over final20 AND no finger contact AND fixed-threshold pose-window stability'}
