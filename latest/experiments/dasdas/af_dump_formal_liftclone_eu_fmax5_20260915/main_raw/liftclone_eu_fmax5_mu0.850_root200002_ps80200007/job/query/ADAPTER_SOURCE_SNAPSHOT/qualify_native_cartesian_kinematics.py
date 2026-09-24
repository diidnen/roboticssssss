"""FK/IK checked against actual native arm-pose API, no physics rollout labels."""
import json
import numpy as np
import sapien
import qualify_native_interfaces as base
from native_cartesian_kinematics import NativeCartesianKinematics


def qualify(env,out):
    out.mkdir(parents=True,exist_ok=False)
    arm=env._af_grasp_arm_tag
    fk=NativeCartesianKinematics(env,arm)
    snapshot=base.Snapshot(env)
    original=np.asarray(fk.art.get_qpos()).copy()
    tests=[]
    try:
        for joint_index in [None]+fk.arm_indices:
            q=original.copy()
            if joint_index is not None:q[joint_index]+=.01
            fk.art.set_qpos(q)
            actual=env.get_arm_pose(arm)
            expected=sapien.Pose(actual[:3],actual[3:]).to_transformation_matrix()
            predicted=fk.command_pose_world(q).to_transformation_matrix()
            tests.append({'perturbed_joint':joint_index,'matrix_max_error':float(abs(predicted-expected).max()),
                          'position_error_m':float(np.linalg.norm(predicted[:3,3]-expected[:3,3]))})
        fk.art.set_qpos(original)
        now=fk.command_pose_world(original)
        goal=sapien.Pose(now.p+np.array([0,0,.001]),now.q)
        solved=fk.solve_command_pose_world(goal)
        inverse_error=float(np.max(abs(fk.command_pose_world(solved).to_transformation_matrix()-goal.to_transformation_matrix())))
        chunk=np.load(out.parent/'fork_native_actions_v1/FIRST_NATIVE_ACTION_CHUNK.npy')
        before=base.readback(env)
        xyz=fk.future_xyz_base(chunk)
        np.save(out/'REAL_CHUNK_CARTESIAN_XYZ_BASE.npy',xyz,allow_pickle=False)
        reader_unchanged=before==base.readback(env)
        passed=all(t['matrix_max_error']<1e-5 for t in tests) and inverse_error<1e-4 and reader_unchanged
        report={'scope':'native FK/IK engineering only','arm':arm,'ee_joint':fk.ee.name,
            'formal_collection_gate_passed':False,'tests':tests,'ik_pose_max_error':inverse_error,
            'future_chunk_fk_preserves_physics':reader_unchanged,'xyz_shape':list(xyz.shape),
            'xyz_motion_span_m':np.ptp(xyz,axis=0).tolist(),'passed':passed}
        (out/'qualification.json').write_text(json.dumps(report,indent=2))
        print('NATIVE_CARTESIAN_QUALIFICATION '+json.dumps(report),flush=True)
        assert passed
    finally:snapshot.restore()


if __name__=='__main__':
    base.qualify=qualify
    base.main()
