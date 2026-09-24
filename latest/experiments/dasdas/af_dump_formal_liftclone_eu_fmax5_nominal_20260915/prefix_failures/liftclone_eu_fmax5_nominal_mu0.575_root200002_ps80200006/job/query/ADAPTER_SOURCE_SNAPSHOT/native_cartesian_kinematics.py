"""Real native forward/inverse kinematics; never substitute joint angles for xyz."""
import numpy as np
import sapien


class NativeCartesianKinematics:
    def __init__(self,env,arm):
        if arm not in ('left','right'):raise ValueError(arm)
        self.env,self.arm=env,arm
        self.art=getattr(env.robot,arm+'_entity')
        self.model=self.art.create_pinocchio_model()
        self.ee=getattr(env.robot,arm+'_ee')
        active=list(self.art.get_active_joints())
        self.arm_indices=[active.index(j) for j in getattr(env.robot,arm+'_arm_joints')]
        if len(self.arm_indices)!=6:raise ValueError('Expected six native arm joints')
        self.rotation_offset=(np.asarray(getattr(env.robot,arm+'_global_trans_matrix')) @
                              np.asarray(getattr(env.robot,arm+'_delta_matrix')))
        self.ee_offset_m=float(getattr(env.robot,arm+'_gripper_bias'))-.12

    def joint_pose_base(self,qpos):
        self.model.compute_forward_kinematics(np.asarray(qpos,float))
        return self.model.get_link_pose(self.ee.child_link.index)*self.ee.pose_in_child

    def command_pose_base(self,qpos):
        # Same frame and gripper_bias convention as Robot._trans_endpose(False).
        joint=self.joint_pose_base(qpos).to_transformation_matrix()
        rotation=joint[:3,:3]@self.rotation_offset
        matrix=np.eye(4)
        matrix[:3,:3]=rotation
        matrix[:3,3]=joint[:3,3]+rotation@np.array([self.ee_offset_m,0,0])
        return sapien.Pose(matrix)

    def command_pose_world(self,qpos):
        return self.art.get_pose()*self.command_pose_base(qpos)

    def future_xyz_base(self,chunk):
        chunk=np.asarray(chunk,float)
        if chunk.ndim!=2 or chunk.shape[1]!=14 or not np.isfinite(chunk).all():
            raise ValueError('Expected actual native 14D joint-action chunk')
        offset=0 if self.arm=='left' else 7
        current=np.asarray(self.art.get_qpos(),float)
        positions=[]
        for action in chunk:
            q=current.copy();q[self.arm_indices]=action[offset:offset+6]
            positions.append(self.command_pose_base(q).p)
        return np.asarray(positions,np.float32)

    def solve_command_pose_world(self,pose,initial_qpos=None):
        pose=pose if isinstance(pose,sapien.Pose) else sapien.Pose(pose[:3],pose[3:])
        base_pose=self.art.get_pose().inv()*pose
        command=base_pose.to_transformation_matrix()
        joint=np.eye(4)
        joint[:3,:3]=command[:3,:3]@self.rotation_offset.T
        joint[:3,3]=command[:3,3]-command[:3,:3]@np.array([self.ee_offset_m,0,0])
        child=sapien.Pose(joint)*self.ee.pose_in_child.inv()
        initial=np.asarray(self.art.get_qpos() if initial_qpos is None else initial_qpos,float)
        mask=np.zeros_like(initial);mask[self.arm_indices]=1
        q,ok,error=self.model.compute_inverse_kinematics(self.ee.child_link.index,child,
            initial_qpos=initial,active_qmask=mask,eps=1e-5,max_iterations=200,dt=.1,damp=1e-6)
        if not ok:raise RuntimeError('Native Cartesian IK did not converge: '+str(error))
        if not np.allclose(np.asarray(q)[mask==0],initial[mask==0],rtol=0,atol=1e-9):
            raise RuntimeError('IK moved an inactive joint')
        return np.asarray(q,float)
