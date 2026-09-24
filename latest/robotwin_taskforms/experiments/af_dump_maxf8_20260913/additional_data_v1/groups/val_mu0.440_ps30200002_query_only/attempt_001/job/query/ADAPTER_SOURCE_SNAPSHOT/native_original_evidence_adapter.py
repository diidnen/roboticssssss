"""Measurement bridge to the unchanged original 58D builder; not an AF model.

Legacy force channels are NORMAL vector projections, never full contact force.
Only the extra friction channels use qualified Newton/Euler contact balance.
No mass/friction labels, outcomes, roots or candidate forces enter this bridge.
Native force-cache mechanics need known link inertial parameters internally;
they are not exported as belief inputs. This module does not certify tactile
mounting, query phase timing, or branch replay: those remain separate gates.
"""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from native_contact_balance import contact_force

SOURCE = Path('/home/exouser/FORTE/current_contract_belief_features.py')
NORMAL_CONTACT_TOLERANCE_N = 1e-6


def measured_patch_record(observation, prestep_velocities_world, physics_dt, step):
    """Export genuine normal samples and measured aggregate friction per finger.

SAPIEN exposes normal contact impulses but not individual friction patches.
The unchanged 58D schema consumes their per-finger SUM, which is measurable by
force balance for leaf fingers contacting ONLY the target. No fake patch count,
position, per-patch tangential vector, or known coefficient is synthesized.
"""
    if physics_dt <= 0 or step < 1 or len(observation['fingers']) != 2:
        raise ValueError('Invalid two-finger sample')
    if np.asarray(prestep_velocities_world).shape != (2, 3):
        raise ValueError('Actual pre-step COM velocities are required')
    normal_rows, friction_rows = [], []
    for side, (finger, velocity) in enumerate(zip(observation['fingers'], prestep_velocities_world)):
        points = finger['points']
        other_load = sum(np.linalg.norm(p['impulse_ns']) / physics_dt
                         for p in points if not p['is_target'])
        if other_load > NORMAL_CONTACT_TOLERANCE_N:
            raise ValueError('Cannot attribute aggregate friction: loaded non-target finger contact')
        native = finger['native_joint_readback']
        full = contact_force(native, velocity, physics_dt)
        target_normal = np.asarray(finger['target_normal_vector_world_n'], float)
        measured_friction = full - target_normal
        if not np.isfinite(measured_friction).all():
            raise ValueError('Nonfinite measured friction')
        normal_magnitudes = [[float(np.linalg.norm(p['impulse_ns']) / physics_dt)]
                             for p in points if p['is_target']]
        normal_rows.append({'pair': [side, 0], 'normal_forces': normal_magnitudes})
        friction_rows.append({'pair': [side, 0],
                              'sum_friction_vector_world': measured_friction.tolist()})
    joints = np.asarray(observation['actual_finger_joint_m'], float)
    if joints.shape != (2,) or not np.isfinite(joints).all():
        raise ValueError('Missing actual finger joint positions')
    return {'step': int(step), 'patches': {'normal': normal_rows, 'friction': friction_rows},
            'finger_joints': joints.tolist(), 'aperture_m': float(joints.sum()),
            'measurement_provenance': {
                'normal': 'native_target_contact_normal_impulse_divided_by_physics_dt',
                'friction': 'qualified_leaf_contact_balance_minus_normal_vector',
                'friction_aggregation': 'measured_per_finger_sum_not_resolved_individual_patches',
                'loaded_other_contact_rejected': True,
                'physics_dt_s': float(physics_dt)}}


def normal_force_local(observation, gripper_rotations_world):
    """Exact world-to-local convention used by original observation function.

Caller supplies the declared sensor frame rotations, NOT camera pose by
assumption. The original offsets differ between left and right fingers.
"""
    rotations = np.asarray(gripper_rotations_world, float)
    if rotations.shape != (2,3,3) or not np.isfinite(rotations).all():
        raise ValueError('Two declared gripper-frame rotations are required')
    if not np.allclose(rotations.transpose(0,2,1)@rotations, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError('Nonorthonormal sensor frames')
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-6, rtol=0):
        raise ValueError('Improper sensor rotation')
    return np.stack([rotation.T @ np.asarray(finger['force_world_n'], float)
                     for finger, rotation in zip(observation['fingers'], rotations)]).astype(np.float32)


def original_evidence(raw, readback):
    """Fail closed before original builder's historical missing-value fallback."""
    spec=importlib.util.spec_from_file_location('_native_unchanged_original58', SOURCE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder=module.ProbeEvidence()
    # Inspect the same imported legacy builder's feature names; numerical
    # channels must all be actually observed, not silently filled with zeros.
    mandatory=[name for name in builder.legacy.feature_names
               if '=' not in name and name not in
               ('probe_phase_unknown','contact_state_unknown','eef_dx','eef_dy','eef_dz')]
    for index,row in enumerate(raw,1):
        for key in mandatory + ['eef_x','eef_y','eef_z']:
            if key not in row or row[key] == '' or not np.isfinite(float(row[key])):
                raise ValueError(f'Missing real observation at step {index}: {key}')
        if row.get('tactile_ok') not in (1,1.,'1','1.0'):
            raise ValueError('Unqualified/missing tactile observations cannot enter original AF')
    return builder.rows(raw,readback)
