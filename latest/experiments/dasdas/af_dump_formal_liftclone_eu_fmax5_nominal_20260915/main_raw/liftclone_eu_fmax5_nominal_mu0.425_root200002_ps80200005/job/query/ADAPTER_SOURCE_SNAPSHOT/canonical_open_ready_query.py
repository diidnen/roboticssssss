"""Engineering-only canonical open-ready initialization, then original P4.

Only replace the native planner-driven preparation move. Never spoof an
observation or initialize a grasp/contact. Frozen runtime files stay unchanged.
"""
import ast
from copy import deepcopy
import hashlib
import inspect
import os
import textwrap
import numpy as np
import sapien
import native_p4_query_env as native
import qualify_original_p4_native as query
import qualify_native_interfaces as base
from trace_original_query_contact import TracedQuery
from rootlocal_collection_contract import write, sha


def bound_ready():
    tree=ast.parse(textwrap.dedent(inspect.getsource(native.NativeP4QueryEnv.ready)))
    baseline=ast.dump(tree)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
              isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ok')
    if ast.unparse(node.value)!='self.native.move(self.native.move_to_pose(self.arm_tag, start))':
        raise ValueError('Native ready source drift')
    original=deepcopy(node.value)
    node.value=ast.parse('self.canonicalize_open_ready(start)',mode='eval').body
    executable=deepcopy(tree);node.value=original
    if ast.dump(tree)!=baseline:raise ValueError('Undeclared ready binding')
    namespace=dict(native.__dict__)
    exec(compile(ast.fix_missing_locations(executable),native.__file__,'exec'),namespace)
    return namespace['ready']


class CanonicalOpenReadyQuery(TracedQuery):
    ready=bound_ready()

    def canonicalize_open_ready(self,start):
        contact=base.contacts(self.native,self.arm)
        if contact['target_measured_squeeze_n']>1e-8:
            raise ValueError('Canonical initialization must be before grasp/contact')
        object_before=np.r_[self.native.deskbin.get_pose().p,self.native.deskbin.get_pose().q].copy()
        art=self.kin.art; active=list(art.get_active_joints())
        # Deterministic configured position targets, not noisy settled qpos.
        seed=np.asarray([float(j.drive_target[0]) for j in active],float)
        seed[self.kin.arm_indices]=np.asarray(getattr(self.native.robot,self.arm+'_homestate'),float)
        for joint,mult,offset in self.joints:seed[active.index(joint)]=.04*mult+offset
        target=self.kin.solve_command_pose_world(sapien.Pose(start[:3],start[3:]),initial_qpos=seed)
        limits=np.asarray(art.get_qlimits())
        if np.any(target<limits[:,0]-1e-6) or np.any(target>limits[:,1]+1e-6):
            raise ValueError('Canonical open configuration outside native joint limits')
        art.set_qpos(target);art.set_qvel(np.zeros_like(target))
        for joint,value in zip(active,target):
            joint.set_drive_target(float(value));joint.set_drive_velocity_target(0.)
        self.native.robot._entity_qf(art)
        actual=np.asarray(art.get_qpos())
        np.testing.assert_allclose(actual,target,rtol=0,atol=2e-7)
        np.testing.assert_array_equal(art.get_qvel(),np.zeros_like(target))
        np.testing.assert_array_equal(np.r_[self.native.deskbin.get_pose().p,self.native.deskbin.get_pose().q],object_before)
        pose=self.kin.command_pose_world(actual)
        if np.linalg.norm(pose.p-np.asarray(start[:3]))>2e-5:
            raise ValueError('Canonical ready FK target mismatch')
        write(self.out/'CANONICAL_OPEN_READY_INITIALIZATION.json',{
            'diagnostic_only':True,'native_ready_source_sha256':sha(native.__file__),
            'only_preparation_move_replaced':True,'original_P4_and_squeeze_unchanged':True,
            'actual_observations_not_spoofed':True,'physics_steps_during_initialization':0,
            'grasp_supplied':False,'object_pose_unchanged':True,'native_ready_pose':list(start),
            'configured_seed':seed.tolist(),'target_qpos':target.tolist(),'actual_qpos':actual.tolist(),
            'actual_qvel':np.asarray(art.get_qvel()).tolist(),'actual_ready_world_pose':np.r_[pose.p,pose.q].tolist(),
            'actual_qpos_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
            'full_solver_state_replay_claimed':False})
        return True


if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    query.NativeP4QueryEnv=CanonicalOpenReadyQuery
    base.qualify=query.qualify
    base.main()
