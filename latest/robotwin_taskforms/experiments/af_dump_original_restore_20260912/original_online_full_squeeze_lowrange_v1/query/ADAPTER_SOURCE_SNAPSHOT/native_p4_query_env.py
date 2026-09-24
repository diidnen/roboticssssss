"""Explicit native embodiment/measurement adapter for the original P4 function.

Engineering acquisition only. Collision depth is retained with its diagnostic
provenance; this file does not certify it for formal AF training.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation
import qualify_native_interfaces as base
from qualify_tactile_acquisition import Acquisition
from native_cartesian_kinematics import NativeCartesianKinematics
from native_original_evidence_adapter import measured_patch_record, normal_force_local


def tensor(x):
    return torch.as_tensor(np.asarray(x).copy(),dtype=torch.float32)


class NativeSceneView:
    def __init__(self, bridge): self.bridge=bridge
    def __getitem__(self,name):
        bridge=self.bridge
        if name=='robot': pose=bridge.kin.art.get_pose()
        elif name=='left_gripper_frame':
            rotation=bridge.force_rotations()[0]
            quat=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
            return SimpleNamespace(data=SimpleNamespace(target_quat_w=tensor(quat)[None,None]))
        else: pose=getattr(bridge.native,name).get_pose()
        return SimpleNamespace(data=SimpleNamespace(root_pos_w=tensor(pose.p)[None],
                                                     root_quat_w=tensor(pose.q)[None]))


class NativeP4QueryEnv:
    def __init__(self,env,out):
        self.native,self.out=env,out
        self.arm=env._af_grasp_arm_tag
        self.kin=NativeCartesianKinematics(env,self.arm)
        self.joints=getattr(env.robot,self.arm+'_gripper')
        self.scale=getattr(env.robot,self.arm+'_gripper_scale')
        self.device='cpu'
        self.scene=NativeSceneView(self)
        self.step_count=0
        self.physics_steps=0
        self.elapsed_s=0.
        self.last_dt_s=0.
        self.readbacks=[]
        self.control_records=[]
        self.captures=[]
        self.debug={}
        self.action_manager=SimpleNamespace(get_term=lambda name:SimpleNamespace(debug_info=self.debug))
        # Exact original minimum-height predicate; native world coordinates are
        # logged explicitly rather than silently redefining the drop threshold.
        self.termination_manager=SimpleNamespace(get_term=lambda name:
            torch.tensor([float(env.deskbin.get_pose().p[2]) < -.05]))
        self.stock_drives=[(j,j.stiffness,j.damping,j.force_limit,j.drive_mode) for j,_,_ in self.joints]
        self.contact_id=1 if self.arm=='left' else 3
        self.geometry=[]
        self._did_reset=False
        self.inner=None
        self.inner_trace=[]
        if os.environ.get('AF_ORIGINAL_SQUEEZE_INNER')=='1':
            from original_squeeze_inner import OriginalSqueezeInner
            self.inner=OriginalSqueezeInner()
            env._af_original_squeeze_inner=self.inner

    def ready(self):
        from envs.utils import ArmTag
        self.arm_tag=ArmTag(self.arm)
        start=self.native.get_grasp_pose(self.native.deskbin,self.arm_tag,
                                        contact_point_id=self.contact_id,pre_dis=.12)
        if start is None or not np.isfinite(start).all(): raise RuntimeError('Missing native rim annotation')
        # Ready configuration only; no close command and no scripted grasp.
        self.native.robot.set_gripper((.04-self.scale[0])/(self.scale[1]-self.scale[0]),self.arm,gripper_eps=0)
        ok=self.native.move(self.native.move_to_pose(self.arm_tag,start))
        if not ok: raise RuntimeError('Native open ready-pose planning failed')
        self.fixed_orientation=np.asarray(self.native.get_arm_pose(self.arm)[3:])
        self.depth_source='PHYSX_COLLISION_DEPTH_DIAGNOSTIC_NOT_QUALIFIED'
        if os.environ.get('AF_P4_PHYSICAL_SURFACE_CAMERA')=='1':
            qualification=Path(__file__).parent/'collision_surface_camera_v3/qualification.json'
            if not json.loads(qualification.read_text()).get('passed'):
                raise RuntimeError('Physical-surface camera has not passed independent qualification')
            from collision_surface_render_acquisition import CollisionSurfaceRenderAcquisition
            self.acquisition=CollisionSurfaceRenderAcquisition(self.native,self.arm)
            self.depth_source='SAPIEN_RENDER_CAMERA_DEPTH_ON_EXACT_PHYSICAL_SURFACE'
        else:
            self.acquisition=Acquisition(self.native,self.arm)
        self.initial_observation=self.observe(record=False)

    def geometry_pose(self, pre_dis):
        pose=self.native.get_grasp_pose(self.native.deskbin,self.arm_tag,
                                       contact_point_id=self.contact_id,pre_dis=pre_dis)
        if pose is None or not np.isfinite(pose).all(): raise RuntimeError('Native grasp geometry unavailable')
        # The original query holds initial orientation fixed; reject a binding
        # that would require a hidden orientation intervention mid-query.
        angle=2*np.arccos(np.clip(abs(np.dot(np.asarray(pose[3:]),self.fixed_orientation)),0,1))
        if angle>.01: raise RuntimeError('Rim geometry requires changed query orientation: '+str(angle))
        actual=sapien.Pose(pose[:3],self.fixed_orientation)
        base_pose=self.kin.art.get_pose().inv()*actual
        self.geometry.append({'step':self.step_count,'pre_dis_m':pre_dis,
                              'native_world_pose':list(pose),'bound_base_xyz':base_pose.p.tolist()})
        return base_pose.p.copy()

    def bind_geometry(self,original_pregrasp,original_grasp):
        self.original_centroid_targets={'pregrasp':original_pregrasp.tolist(),'grasp':original_grasp.tolist()}
        return self.geometry_pose(.10),self.geometry_pose(0.)

    def refresh_grasp(self): return self.geometry_pose(0.)

    def pose_in_base(self,name):
        pose=self.kin.art.get_pose().inv()*getattr(self.native,name).get_pose()
        return pose.p.copy(),pose.q.copy()

    def verify_friction(self,name,mu):
        actor=getattr(self.native,name).actor
        body=actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        values=[(float(shape.physical_material.static_friction),
                 float(shape.physical_material.dynamic_friction)) for shape in body.collision_shapes]
        if not values or not np.allclose(values,mu,rtol=0,atol=1e-6):
            raise RuntimeError('Requested object friction not applied: '+str(values))
        self.material_readback=values
        return float(np.mean(values))

    def force_rotations(self):
        rotations=[]
        for i,(joint,_,_) in enumerate(self.joints):
            sign=1 if i==0 else -1
            # Explicit CAD pad frame: x along finger, z into gap. This is NOT
            # assumed to equal the camera frame; its rotation is declared here.
            local=np.array([[1,0,0],[0,0,-sign],[0,sign,0.]])
            rotation=joint.child_link.pose.to_transformation_matrix()[:3,:3]@local
            closing=joint.parent_link.pose.to_transformation_matrix()[:3,1]
            if abs(float(rotation[:,2]@closing))<.9999:
                raise RuntimeError('Declared tactile-force frame not aligned to actual closing axis')
            rotations.append(rotation)
        return np.asarray(rotations)

    def reset(self,seed):
        if self._did_reset or int(seed)!=200002:
            raise RuntimeError('Query adapter only wraps this live root once; no unqualified restore')
        self._did_reset=True
        return self.initial_observation,{}

    def observe(self,record):
        observed=base.contacts(self.native,self.arm)
        rotations=self.force_rotations()
        normal=normal_force_local(observed,rotations)
        self.debug['f_sq_meas']=(self.inner.last['measured_filtered_N'] if self.inner is not None and self.inner.last
                                else float(2*min(abs(normal[:,2]))))
        tag='reset' if not record else f'query_{self.step_count:04d}'
        capture=self.acquisition.capture(self.out,tag)
        self.captures.append(capture)
        with np.load(self.out/(tag+'_sensor.npz')) as data:
            marker_key='markers' if self.depth_source.startswith('SAPIEN_RENDER') else 'collision_markers_diagnostic'
            markers=data[marker_key].copy()
        if not np.isfinite(markers).all(): raise RuntimeError('Nonfinite real tactile observation')
        actual_pose=self.kin.art.get_pose().inv()*sapien.Pose(
            self.native.get_arm_pose(self.arm)[:3],self.native.get_arm_pose(self.arm)[3:])
        if record:
            readback=measured_patch_record(observed,self.prestep_velocities,
                                          self.native.physics_timestep,self.step_count)
            readback['elapsed_s']=self.elapsed_s
            readback['interval_s']=self.last_dt_s
            readback['prestep_com_velocities_world']=[v.tolist() for v in self.prestep_velocities]
            readback['sensor_depth_provenance']=self.depth_source
            self.readbacks.append(readback)
        return {'policy':{'eef_pose':tensor(np.r_[actual_pose.p,actual_pose.q])[None],
                          'gripper_net_force':tensor(normal)[None],
                          'gripper_pos':tensor(observed['actual_finger_joint_m'])[None],
                          'gripper_marker_motion':tensor(markers)[None]}}

    def step(self,action):
        value=action.detach().cpu().numpy().reshape(-1)
        if value.shape!=(13,) or not np.isfinite(value).all(): raise RuntimeError('Invalid original P4 action')
        force=float(value[9]+value[12])
        if not np.isclose(value[9],value[12]): raise RuntimeError('Expected bilateral original force targets')
        quat=Rotation.from_rotvec(value[3:6]).as_quat()[[3,0,1,2]]
        target=self.kin.art.get_pose()*sapien.Pose(value[:3],quat)
        q=self.kin.solve_command_pose_world(target)
        dt=float(self.native.physics_timestep)
        # Nearest future native step, not a fake nominal timestamp. Alternates
        # 13/12 (52/48 ms) for a 4 ms simulator and a nominal 50 ms protocol.
        next_total=int(np.floor((self.step_count+1)*.05/dt+.5))
        n=next_total-self.physics_steps
        if n<1: raise RuntimeError('Invalid query clock')
        self.last_dt_s=n*dt
        self.native.robot.set_arm_joints(q[self.kin.arm_indices],np.zeros(6),self.arm)
        if force>0 and self.inner is None:
            self.native.robot.set_gripper_force_limit(force/2,self.arm)
        else:
            # Zero squeeze denotes open-position intent; a zero actuator cap
            # would prevent opening. Restore original native position drives.
            for joint,stiffness,damping,limit,mode in self.stock_drives:
                joint.set_drive_properties(stiffness,damping,limit,mode)
        self.native.robot.set_gripper((float(value[6])-self.scale[0])/(self.scale[1]-self.scale[0]),
                                     self.arm,gripper_eps=0)
        for _ in range(n):
            if self.inner is not None:
                measured=base.contacts(self.native,self.arm)['measured_squeeze_n']
                actual_command=self.inner.step(measured,force,float(value[6]))
                self.native.robot.set_gripper((actual_command-self.scale[0])/(self.scale[1]-self.scale[0]),
                                             self.arm,gripper_eps=0)
                self.inner_trace.append({'physics_step':len(self.inner_trace)+1,**self.inner.last})
            self.prestep_velocities=[np.asarray(j.child_link.linear_velocity).copy() for j,_,_ in self.joints]
            self.native.scene.step()
        self.native._af_original_outer_command=float(value[6])
        self.step_count+=1
        self.physics_steps=next_total
        self.elapsed_s=self.physics_steps*dt
        obs=self.observe(record=True)
        self.control_records.append({'step':self.step_count,'elapsed_s':self.elapsed_s,
            'interval_s':self.last_dt_s,'native_steps':n,'original_action13':value.tolist(),
            'joint_targets':q[self.kin.arm_indices].tolist(),
            'actual_eef_base':obs['policy']['eef_pose'][0].tolist(),
            'force_frame_rotations_world':self.force_rotations().tolist(),
            'measured_squeeze_n':self.debug['f_sq_meas']})
        return obs,torch.zeros(1),self.termination_manager.get_term('object_1_dropped'),torch.tensor([False]),{}

    def save(self):
        for name,value in [('patch_readbacks',self.readbacks),('native_controls',self.control_records),
                           ('geometry_binding',self.geometry),('original_squeeze_inner_trace',self.inner_trace)]:
            (self.out/(name+'.json')).write_text(json.dumps(value,indent=2))
