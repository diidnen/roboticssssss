"""Real dump sensor motion qualification. Not original-P4 protocol/training data.

Keeps the task, root and physical assets; probes acquisition at fixed 4 N.
Depth provenance remains explicit, and no full-task success labels are emitted.
"""
from copy import deepcopy
import json
import numpy as np
from original_tactile_core import OriginalTactileCore
from qualify_tactile_acquisition import Acquisition
import qualify_native_interfaces as base
from native_original_evidence_adapter import measured_patch_record


def qualify(env, out):
    out.mkdir(parents=True, exist_ok=False)
    arm = env._af_grasp_arm_tag
    acquisition = Acquisition(env, arm)
    start_snapshot = base.Snapshot(env)
    servo, constants = base.original_servo()
    scale = getattr(env.robot, arm + '_gripper_scale')
    joints = getattr(env.robot, arm + '_gripper')
    command = float(joints[0][0].drive_target[0])
    env.robot.set_gripper_force_limit(2., arm)
    report = {'scope': 'actual dump dynamic sensor engineering qualification',
              'root': 200002, 'friction': .55, 'total_squeeze_target_n': 4.,
              'formal_collection_gate_passed': False, 'task_success_evaluated': False,
              'original_P4_protocol_claimed': False, 'servo_constants': constants,
              'physics_dt_s': env.physics_timestep, 'sensor_period_s': 10*env.physics_timestep,
              'original_tactile_source_hashes': acquisition.core.source_hashes,
              'captures': [], 'motion_segments': []}
    all_contacts = []
    old_callback = getattr(env, 'on_physics_step', None)
    step = 0
    phase = 'preload'
    original_scene_step = env.scene.step
    prestep_velocities = None

    def step_with_velocity():
        nonlocal prestep_velocities
        prestep_velocities = [np.asarray(j.child_link.linear_velocity).copy() for j,_,_ in joints]
        return original_scene_step()

    def after_step():
        nonlocal command, step
        if callable(old_callback): old_callback()
        step += 1
        observed = base.contacts(env, arm)
        observed.update(step=step, phase=phase)
        observed['prestep_com_velocities_world'] = [v.tolist() for v in prestep_velocities]
        try:
            observed['native_aggregate_patch_readback'] = measured_patch_record(
                observed, prestep_velocities, env.physics_timestep, step)
        except ValueError as exc:
            observed['patch_readback_rejected'] = str(exc)
        all_contacts.append(observed)
        if step % 5 == 0:
            command = servo(command, observed['measured_squeeze_n'], 4.)
            env.robot.set_gripper((command-scale[0])/(scale[1]-scale[0]), arm, gripper_eps=0)
        if step % 10 == 0:
            report['captures'].append(acquisition.capture(out, f'{phase}_{step:05d}'))
            (out / 'progress.json').write_text(json.dumps({'phase': phase, 'steps': step,
                'sensor_frames': len(report['captures']), 'formal_collection_gate_passed': False}))

    env.on_physics_step = after_step
    env.scene.step = step_with_velocity
    try:
        for _ in range(250):
            env.scene.step()
            after_step()
        # Fixed geometric tangent, perpendicular to the closing axis. This is
        # an instrumentation excitation, NOT a replacement learned query.
        normal = np.asarray(base.contacts(env, arm)['normal_axis_world'])
        tangent = np.array([0., 0., 1.])
        tangent -= np.dot(tangent, normal)*normal
        if np.linalg.norm(tangent) < 1e-4:
            tangent = np.array([1.,0.,0.])-normal[0]*normal
        tangent /= np.linalg.norm(tangent)
        report['diagnostic_tangent_world'] = tangent.tolist()
        origin = np.array(env.get_arm_pose(arm), copy=True)
        for name, displacement in [('out', .002), ('back', 0.)]:
            phase = name
            target = origin.copy()
            target[:3] += tangent*displacement
            ok = env.move(env.move_to_pose(arm, target))
            report['motion_segments'].append({'name': name, 'planned': bool(ok),
                'target_pose': target.tolist(), 'actual_pose': np.asarray(env.get_arm_pose(arm)).tolist()})
            if not ok: raise RuntimeError('Diagnostic motion planner failed: ' + name)
        phase = 'post_hold'
        for _ in range(100):
            env.scene.step()
            after_step()
        # Real open-state negative control; servo must not close the fingers.
        env.on_physics_step = old_callback
        env.robot.set_gripper_force_limit(5., arm)
        env.robot.set_gripper((.04-scale[0])/(scale[1]-scale[0]), arm, gripper_eps=0)
        for _ in range(250): env.scene.step()
        report['captures'].append(acquisition.capture(out, 'open_negative_control'))
        open_contact = base.contacts(env,arm)
        report['open_force_negative_control'] = measured_patch_record(
            open_contact,prestep_velocities,env.physics_timestep,step+1)
        report['completed'] = True
    except BaseException as exc:
        report['completed'] = False
        report['error'] = repr(exc)
        raise
    finally:
        env.scene.step = original_scene_step
        env.on_physics_step = old_callback
        (out / 'dynamic_contacts.json').write_text(json.dumps(all_contacts))
        start_snapshot.restore()
        report['initial_physics_restored'] = True
        (out / 'qualification.json').write_text(json.dumps(report, indent=2))
        print('DYNAMIC_SENSOR_QUALIFICATION ' + json.dumps({k:v for k,v in report.items() if k != 'captures'}), flush=True)


if __name__ == '__main__':
    base.qualify = qualify
    base.main()
