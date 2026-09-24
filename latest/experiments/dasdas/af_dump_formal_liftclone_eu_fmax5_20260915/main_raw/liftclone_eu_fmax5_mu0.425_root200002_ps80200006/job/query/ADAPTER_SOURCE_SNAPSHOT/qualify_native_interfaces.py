#!/usr/bin/env python3
"""Engineering qualification, NOT an AF training or task-success experiment.

Reuse the original P4 aperture servo verbatim (AST extraction), measure real
PhysX contact impulses and actual qpos, and test native physics snapshots.
No tactile features, outcomes or AF predictions are synthesized.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys
import numpy as np
import torch
import sapien

P4 = Path('/home/exouser/Tabero/analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py')

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def vec(x):
    return np.asarray(x).tolist()

def original_servo():
    tree = ast.parse(P4.read_text())
    wanted = {'SERVO_STEP', 'SERVO_DEADBAND', 'D_OPEN', 'D_CLOSED'}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assigned = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
            if assigned & wanted:
                if not assigned <= wanted:
                    raise RuntimeError('Unexpected source assignment')
                body.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == '_force_servo':
            body.append(node)
    namespace = {'np': np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(P4), 'exec'), namespace)
    return namespace['_force_servo'], {k: namespace[k] for k in sorted(wanted)}

def readback(env):
    entities = []
    for e in env.scene.get_all_actors():
        p = e.get_pose()
        row = {'name': e.get_name(), 'pose': vec(np.r_[p.p, p.q])}
        for name in ('get_velocity', 'get_angular_velocity'):
            if hasattr(e, name): row[name] = vec(getattr(e, name)())
        body = e.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if body is not None:
            row['native_com_linear_velocity'] = vec(body.linear_velocity)
            row['native_angular_velocity'] = vec(body.angular_velocity)
        entities.append(row)
    articulations = []
    for e in env.scene.get_all_articulations():
        p = e.get_pose()
        row = {'name': e.get_name(), 'pose': vec(np.r_[p.p, p.q])}
        for name in ('get_qpos', 'get_qvel', 'get_qf', 'get_root_linear_velocity', 'get_root_angular_velocity'):
            row[name] = vec(getattr(e, name)())
        row['joints'] = [{'name': j.name, 'target': vec(j.drive_target),
                          'velocity_target': vec(j.drive_velocity_target)} for j in e.get_active_joints()]
        if os.environ.get('AF_DIAGNOSTIC_CHECK_LINK_STATE') == '1':
            row['native_links'] = [{'name': link.name,
                 'pose': vec(np.r_[link.pose.p,link.pose.q]),
                 'com_linear_velocity':vec(link.linear_velocity),
                 'angular_velocity':vec(link.angular_velocity)} for link in e.get_links()]
        articulations.append(row)
    return {'actors': entities, 'articulations': articulations}

def contacts(env, arm):
    joints = getattr(env.robot, arm + '_gripper')
    # Both finger vectors expressed in one orthonormal frame. The closing axis
    # is the ARX-X5 URDF parent link6 Y axis (the two joint axes are +/-Y).
    parent = joints[0][0].parent_link
    rotation = parent.pose.to_transformation_matrix()[:3, :3]
    normal_axis = rotation[:, 1]
    actor_entity = env.deskbin.actor
    fingers = [j.child_link for j, _, _ in joints]
    records = []
    for finger in fingers:
        total = np.zeros(3)
        target_total = np.zeros(3)
        normal = np.zeros(3)
        tangent = np.zeros(3)
        points = []
        for c in env.scene.get_contacts():
            entities = [b.entity for b in c.bodies]
            idx = next((i for i, e in enumerate(entities) if e == finger.entity), None)
            if idx is None:
                continue
            target = entities[1 - idx] == actor_entity
            for p in c.points:
                impulse = np.asarray(p.impulse, dtype=float)
                # Raw body ordering, impulse and normal retained for independent
                # sign verification; squeeze is insensitive to global sign.
                force = (1 if idx == 0 else -1) * impulse / env.physics_timestep
                n = np.asarray(p.normal, dtype=float)
                fn = np.dot(force, n) * n
                total += force
                if target:
                    target_total += force
                    normal += fn
                    tangent += force - fn
                points.append({'other': entities[1 - idx].name, 'is_target': target,
                               'finger_body_index': idx, 'impulse_ns': vec(impulse),
                               'normal': vec(n), 'position_world': vec(p.position),
                               'separation_m': float(p.separation)})
        records.append({'finger': finger.name, 'force_world_n': vec(total),
                        'target_force_world_n': vec(target_total),
                        'target_normal_vector_world_n': vec(normal),
                        'unqualified_tangential_impulse_residual_world_n': vec(tangent), 'points': points})
        if getattr(env, '_qualification_native_ft', False):
            import af_native_joint_readback
            records[-1]['native_joint_readback'] = af_native_joint_readback.read_link(finger)
    projected = [float(np.dot(r['force_world_n'], normal_axis)) for r in records]
    target_projected = [float(np.dot(r['target_force_world_n'], normal_axis)) for r in records]
    entity = getattr(env.robot, arm + '_entity')
    active = list(entity.get_active_joints())
    positions = [float(entity.qpos[active.index(j)]) for j, _, _ in joints]
    return {'fingers': records, 'normal_axis_world': vec(normal_axis),
            'measured_squeeze_n': 2 * min(map(abs, projected)),
            'target_measured_squeeze_n': 2 * min(map(abs, target_projected)),
            'actual_finger_joint_m': positions, 'sum_actual_joints_m': sum(positions)}

class Snapshot:
    def __init__(self, env):
        self.env = env
        self.canonical_kinematics = os.environ.get('AF_DIAGNOSTIC_CANONICAL_KINEMATICS') == '1'
        if self.canonical_kinematics:
            self.update_kinematics()
        self.physics = env.scene.physx_system.pack()
        self.poses = env.scene.pack_poses()
        self.obs = deepcopy(env.now_obs)
        self.np_rng = np.random.get_state()
        self.py_rng = random.getstate()
        self.torch_rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        self.robot_fields = {k: deepcopy(v) for k, v in vars(env.robot).items()
                             if k.endswith('_gripper_val') or k.endswith('_force_limit_n')}
        self.joints = []
        self.qf = []
        # SAPIEN 3's legacy Entity wrapper has no get_velocity. Capture the
        # actual dynamic component velocities explicitly, not just actor poses.
        self.rigid_velocities = [(b, np.array(b.linear_velocity).copy(),
                                  np.array(b.angular_velocity).copy())
                                 for b in env.scene.physx_system.get_rigid_dynamic_components()
                                 if not b.kinematic]
        for art in env.scene.get_all_articulations():
            self.qf.append((art, np.array(art.get_qf()).copy()))
            for j in art.get_active_joints():
                self.joints.append((j, np.array(j.drive_target).copy(),
                                    np.array(j.drive_velocity_target).copy(),
                                    j.stiffness, j.damping, j.force_limit, j.drive_mode))
        self.readback = readback(env)

    def update_kinematics(self):
        import af_native_joint_readback as native
        for art in self.env.scene.get_all_articulations():
            native.update_kinematics(art.get_links()[0])

    def restore(self):
        env = self.env
        env.scene.physx_system.unpack(self.physics)
        env.scene.unpack_poses(self.poses)
        for body, linear, angular in self.rigid_velocities:
            body.set_linear_velocity(linear)
            body.set_angular_velocity(angular)
        for art, qf in self.qf:
            art.set_qf(qf)
        for j, target, velocity, stiffness, damping, limit, mode in self.joints:
            j.set_drive_properties(stiffness, damping, limit, mode)
            j.set_drive_target(target)
            j.set_drive_velocity_target(velocity)
        for k, v in self.robot_fields.items(): setattr(env.robot, k, deepcopy(v))
        env.now_obs = deepcopy(self.obs)
        np.random.set_state(self.np_rng)
        random.setstate(self.py_rng)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None: torch.cuda.set_rng_state_all(self.cuda_rng)
        if self.canonical_kinematics:
            self.update_kinematics()
        actual = readback(env)
        if actual != self.readback:
            mismatches = []
            for group in self.readback:
                for i, (left, right) in enumerate(zip(self.readback[group], actual[group])):
                    for key in left:
                        if left[key] != right[key]:
                            mismatches.append({'group': group, 'index': i, 'key': key,
                                               'before': left[key], 'restored': right[key]})
            raise RuntimeError('Native snapshot state mismatch: ' + json.dumps(mismatches))

class QualificationDone(BaseException):
    pass

def qualify(env, out):
    out.mkdir(parents=True, exist_ok=False)
    servo, constants = original_servo()
    arm = env._af_grasp_arm_tag
    joints = getattr(env.robot, arm + '_gripper')
    before = contacts(env, arm)
    (out / 'initial_contacts.json').write_text(json.dumps(before, indent=2))
    snap = Snapshot(env)
    (out / 'physx_snapshot.bin').write_bytes(snap.physics)
    (out / 'scene_poses.bin').write_bytes(snap.poses)
    report = {'scope': 'engineering stationary hold; not full-task AF', 'root': 200002,
              'friction': 0.55, 'original_p4_sha256': hashlib.sha256(P4.read_bytes()).hexdigest(),
              'servo_constants': constants, 'physics_dt': env.physics_timestep,
              'state_sha256': digest(snap.readback), 'branches': [],
              'tactile_features_available': False, 'task_success_evaluated': False}
    scale = getattr(env.robot, arm + '_gripper_scale')
    handoff = float(joints[0][0].drive_target[0])
    report['handoff_command_m'] = handoff
    # The old scripted grasp commands -0.01 m despite a 0 m URDF joint limit.
    # Apply the original servo's own clipping once at the common 4 N preload
    # reference, rather than mislabel that old command as an original handoff.
    report['native_scripted_handoff_inside_original_support'] = 0 <= handoff <= constants['D_OPEN']
    handoff = servo(handoff, before['measured_squeeze_n'], 4.)
    report['qualification_initial_servo_command_m'] = handoff
    # The native scene is held fixed. This cadence is an explicit engineering
    # probe, not a claim of original rollout-time equivalence.
    servo_stride = 5
    report['qualification_servo_period_s'] = servo_stride * env.physics_timestep
    for branch_id, force in enumerate((3., 3., 4., 5.)):
        snap.restore()
        command = handoff
        env.robot.set_gripper_force_limit(force / 2, arm)
        trace = []
        for step in range(500):
            if step % servo_stride == 0:
                normalized = (command - scale[0]) / (scale[1] - scale[0])
                env.robot.set_gripper(normalized, arm, gripper_eps=0)
            env.scene.step()
            observed = contacts(env, arm)
            observed.update({'step': step + 1, 'command_m': command,
                             'object_pose': vec(np.r_[env.deskbin.get_pose().p, env.deskbin.get_pose().q])})
            trace.append(observed)
            if (step + 1) % servo_stride == 0:
                command = servo(command, observed['measured_squeeze_n'], force)
        (out / f'hold_{branch_id}_{force:g}N.json').write_text(json.dumps(trace))
        forces = np.array([r['measured_squeeze_n'] for r in trace[250:]])
        row = {'branch_id': branch_id, 'total_squeeze_setpoint_n': force,
               'per_finger_drive_cap_n': force / 2, 'trace_sha256': digest(trace),
               'tail_mean_squeeze_n': float(forces.mean()), 'tail_std_squeeze_n': float(forces.std()),
               'tail_in_original_deadband_fraction': float(np.mean(abs(forces - force) <= constants['SERVO_DEADBAND'])),
               'tail_contact_fraction': float(np.mean(forces > 0)), 'end_state_sha256': digest(readback(env))}
        report['branches'].append(row)
        (out / 'qualification.json').write_text(json.dumps(report, indent=2))
        print('NATIVE_INTERFACE_BRANCH ' + json.dumps(row), flush=True)
    report['same_force_exact_trace_replay'] = report['branches'][0]['trace_sha256'] == report['branches'][1]['trace_sha256']
    report['same_force_exact_end_state'] = report['branches'][0]['end_state_sha256'] == report['branches'][1]['end_state_sha256']
    snap.restore()
    report['snapshot_restores_exposed_state_exactly'] = True
    (out / 'qualification.json').write_text(json.dumps(report, indent=2))
    print('NATIVE_INTERFACE_DONE ' + json.dumps(report), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--native-ft', action='store_true')
    ap.add_argument('--friction', type=float, default=.55)
    ap.add_argument('--policy-seed', type=int, default=140200002)
    args = ap.parse_args()
    args.repo = args.repo.resolve()
    args.out = args.out.resolve()
    if not np.isfinite(args.friction) or args.friction<=0:
        raise ValueError('Positive finite object-side friction required')
    if os.environ.get('AF_DIAGNOSTIC_ENHANCED_DETERMINISM') == '1':
        original_scene_config = sapien.SceneConfig
        def deterministic_scene_config(*a, **kw):
            cfg = original_scene_config(*a, **kw)
            cfg.enable_enhanced_determinism = True
            return cfg
        sapien.SceneConfig = deterministic_scene_config
    sys.path.insert(0, str(args.repo))
    from envs.dump_bin_bigbin import dump_bin_bigbin
    original = dump_bin_bigbin.supply_activeforcing_grasp
    def after_grasp(self):
        if os.environ.get('AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP') == '1':
            self._af_grasp_arm_tag = 'left' if self.deskbin.get_pose().p[0] < 0 else 'right'
        else:
            original(self)
        self._qualification_native_ft = args.native_ft
        self._af_qualification_policy_seed = args.policy_seed
        qualify(self, args.out)
        raise QualificationDone()
    dump_bin_bigbin.supply_activeforcing_grasp = after_grasp
    checkpoint = args.repo.parent / 'checkpoints/pi0_robotwin_30000/30000'
    additional = ','.join([f'ckpt_name={checkpoint}', 'action_type=joint', 'activeforcing_enabled=true',
        'af_dynamic_evaluator=false', 'af_supplied_grasp=true', 'af_query_enabled=false',
        'af_preaction_only=true', f'af_contact_friction={args.friction}', 'af_force_limit_n=3',
        'start_seed=200002', f'af_policy_seed={args.policy_seed}', 'strict_seed=true'])
    sys.argv = [str(args.repo / 'scripts/eval_policy_xpolicylab.py'), '--task_name', 'dump_bin_bigbin',
                '--env_cfg_type', 'arx_x5', '--policy_name', 'Pi_0', '--host', 'localhost', '--port', '6001',
                '--protocol', 'ws', '--seed', '1', '--test_num', '1', '--expert_check', 'false',
                '--frequency', '30', '--additional_info', additional]
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    except QualificationDone:
        pass

if __name__ == '__main__':
    main()
