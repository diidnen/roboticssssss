"""Discriminate real friction from instrumentation bias on one actual dump state."""
import gzip
import os
import json
from pathlib import Path
import numpy as np
import sapien
import qualify_native_interfaces as base
from native_contact_balance import contact_force


def qualify(env, out):
    out.mkdir(parents=True, exist_ok=False)
    arm = env._af_grasp_arm_tag
    servo, constants = base.original_servo()
    joints = getattr(env.robot, arm + '_gripper')
    scale = getattr(env.robot, arm + '_gripper_scale')
    command = float(joints[0][0].drive_target[0])
    env.robot.set_gripper_force_limit(2.,arm)
    for step in range(500):
        if step%5 == 0:
            command = servo(command,base.contacts(env,arm)['measured_squeeze_n'],4.)
            env.robot.set_gripper((command-scale[0])/(scale[1]-scale[0]),arm,gripper_eps=0)
        env.scene.step()
    snapshot = base.Snapshot(env)
    canonical_state = dict(snapshot.readback)
    canonical_state['actors'] = sorted(canonical_state['actors'], key=lambda r:r['name'])
    canonical_state['articulations'] = sorted(canonical_state['articulations'], key=lambda r:r['name'])
    (out/'canonical_initial_state.json').write_text(json.dumps(canonical_state,indent=2))
    bodies = [j.child_link for j,_,_ in joints]
    bodies.append(env.deskbin.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent))
    materials = []
    for body in bodies:
        for shape in body.get_collision_shapes():
            original = shape.get_physical_material()
            zero = env.scene.create_physical_material(0.,0.,original.restitution)
            materials.append((shape,original,zero))
    report = {'scope':'same-state actual-task sensor diagnostic; NOT AF trial', 'root':200002,
              'formal_collection_gate_passed':False,'state_sha256':base.digest(snapshot.readback),
              'branches':[], 'materials_before':[{'static':m.static_friction,'dynamic':m.dynamic_friction,
              'restitution':m.restitution} for _,m,_ in materials]}
    cold = os.environ.get('AF_DIAGNOSTIC_COLD_SOLVER_RESET') == '1'
    report['canonical_cold_solver_reset'] = cold
    report['canonical_initial_state_sha256'] = base.digest(canonical_state)
    report['enhanced_determinism'] = bool(env.scene.physx_system.get_config().enable_enhanced_determinism)
    try:
        branch_names = os.environ.get('AF_DIAGNOSTIC_BRANCHES', 'original,zero_friction,original_replay').split(',')
        fresh = os.environ.get('AF_DIAGNOSTIC_FRESH_PREFIX') == '1'
        if fresh and (branch_names != ['original'] or cold):
            raise ValueError('Fresh-prefix test must not restore/mutate an existing branch')
        for name in branch_names:
            if not fresh:
                for shape,original,zero in materials: shape.set_physical_material(original)
            if cold:
                import af_native_joint_readback as native
                report['cold_reset_receipt'] = native.cold_solver_reset(joints[0][0].child_link,
                    flush=os.environ.get('AF_DIAGNOSTIC_FLUSH_BUFFERS') == '1',
                    canonical_ids=os.environ.get('AF_DIAGNOSTIC_CANONICAL_ACTOR_IDS') == '1',
                    canonical_shapes=os.environ.get('AF_DIAGNOSTIC_CANONICAL_SHAPE_IDS') == '1')
                report.setdefault('all_cold_reset_receipts',[]).append(report['cold_reset_receipt'])
            if not fresh: snapshot.restore()
            if os.environ.get('AF_DIAGNOSTIC_WAKE_ALL') == '1':
                for body in env.scene.physx_system.get_rigid_dynamic_components():
                    if not body.kinematic: body.wake_up()
                for art in env.scene.get_all_articulations():
                    art.get_links()[0].wake_up()
                report['canonical_wake_all'] = True
            if name=='zero_friction':
                for shape,original,zero in materials: shape.set_physical_material(zero)
            records=[]
            for step in range(30):
                velocities=[np.asarray(j.child_link.get_linear_velocity()).copy() for j,_,_ in joints]
                env.scene.step()
                row=base.contacts(env,arm)
                row['step']=step
                row['exposed_state']=base.readback(env)
                for finger,velocity in zip(row['fingers'],velocities):
                    finger['prestep_com_velocity_world']=velocity.tolist()
                    finger['damping_corrected_contact_world_n']=contact_force(finger['native_joint_readback'],velocity,env.physics_timestep).tolist()
                art=getattr(env.robot,arm+'_entity')
                row['qf']=np.asarray(art.get_qf()).tolist()
                records.append(row)
            with gzip.open(out/(name+'.json.gz'),'wt') as f: json.dump(records,f)
            errors=[]
            other=[]
            closing=[]
            normal_angles=[]
            corrected_errors=[]
            for row in records:
                axis=np.array(row['normal_axis_world'])
                for finger in row['fingers']:
                    full=np.array(finger['native_joint_readback']['unqualified_contact_balance_world_n'])
                    normal=np.array(finger['force_world_n'])
                    residual=full-normal
                    errors.append(residual)
                    corrected_errors.append(np.asarray(finger['damping_corrected_contact_world_n'])-normal)
                    closing.append(np.dot(residual,axis))
                    other.append(sum(np.linalg.norm(p['impulse_ns'])/.004 for p in finger['points'] if not p['is_target']))
                    for point in finger['points']:
                        if point['is_target'] and np.linalg.norm(point['impulse_ns'])/.004>.01:
                            normal_angles.append(float(np.degrees(np.arccos(np.clip(abs(np.dot(point['normal'],axis)),0,1)))))
            errors=np.array(errors)
            corrected_errors=np.asarray(corrected_errors)
            result={'name':name,'trace_sha256':base.digest(records),
                    'full_minus_normal_abs_max_n':abs(errors).max(axis=0).tolist(),
                    'corrected_full_minus_normal_abs_max_n':abs(corrected_errors).max(axis=0).tolist(),
                    'full_minus_normal_norm_p95_n':float(np.percentile(np.linalg.norm(errors,axis=1),95)),
                    'closing_component_abs_p95_n':float(np.percentile(abs(np.array(closing)),95)),
                    'loaded_normal_angle_range_degrees':[min(normal_angles),max(normal_angles)] if normal_angles else None,
                    'other_contact_force_max_n':max(other),
                    'squeeze_mean_n':float(np.mean([r['measured_squeeze_n'] for r in records]))}
            report['branches'].append(result)
            (out/'qualification.json').write_text(json.dumps(report,indent=2))
            print('ZERO_FRICTION_DISCRIMINATION '+json.dumps(result),flush=True)
        report['original_replay_exact']=(report['branches'][0]['trace_sha256']==report['branches'][-1]['trace_sha256']) if len(report['branches'])>1 else None
        zero=next((b for b in report['branches'] if b['name']=='zero_friction'),None)
        report['zero_friction_full_vector_matches_normal']=max(zero['corrected_full_minus_normal_abs_max_n'])<.001 if zero else None
    finally:
        for shape,original,zero in materials: shape.set_physical_material(original)
        snapshot.restore()
        report['materials_and_exposed_physics_restored']=True
        (out/'qualification.json').write_text(json.dumps(report,indent=2))
        print('ZERO_FRICTION_DIAGNOSTIC_DONE '+json.dumps(report),flush=True)


if __name__=='__main__':
    base.qualify=qualify
    base.main()
