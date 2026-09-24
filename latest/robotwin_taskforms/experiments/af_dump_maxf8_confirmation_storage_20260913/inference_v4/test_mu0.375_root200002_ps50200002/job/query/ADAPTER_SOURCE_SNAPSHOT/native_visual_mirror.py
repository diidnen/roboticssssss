"""CPU-worker state -> parent native renderer. Not a physics snapshot restore.

The renderer parent must NEVER be stepped or used as a fresh fork source after
mirroring. All physics workers must originate from the unchanged live boundary.
Force/contact readings must come from the worker, NOT this visual mirror.
"""
from copy import deepcopy
import numpy as np
from qualify_native_interfaces import readback,digest


def capture_visual_packet(env):
    joints=[]
    for art in env.scene.get_all_articulations():
        joints.append({'name':art.name,'qf':np.asarray(art.get_qf()).copy(),
            'joints':[{'name':j.name,'target':np.asarray(j.drive_target).copy(),
                       'velocity':np.asarray(j.drive_velocity_target).copy(),
                       'stiffness':j.stiffness,'damping':j.damping,'limit':j.force_limit,'mode':j.drive_mode}
                      for j in art.get_active_joints()]})
    return {'physics':env.scene.physx_system.pack(),'poses':env.scene.pack_poses(),
            'articulations':joints,'robot_fields':{k:deepcopy(v) for k,v in vars(env.robot).items()
                      if k.endswith('_gripper_val') or k.endswith('_force_limit_n')},
            'take_action_cnt':env.take_action_cnt,'eval_success':env.eval_success,
            'state':readback(env)}


def apply_visual_packet(env,packet):
    env.scene.physx_system.unpack(packet['physics'])
    env.scene.unpack_poses(packet['poses'])
    arts=list(env.scene.get_all_articulations())
    if len(arts)!=len(packet['articulations']):raise ValueError('Visual mirror topology changed')
    for art,record in zip(arts,packet['articulations']):
        if art.name!=record['name']:raise ValueError('Visual articulation identity mismatch')
        art.set_qf(record['qf'])
        joints=list(art.get_active_joints())
        if len(joints)!=len(record['joints']):raise ValueError('Visual joint topology changed')
        for joint,row in zip(joints,record['joints']):
            if joint.name!=row['name']:raise ValueError('Visual joint ordering changed')
            joint.set_drive_properties(row['stiffness'],row['damping'],row['limit'],row['mode'])
            joint.set_drive_target(row['target']);joint.set_drive_velocity_target(row['velocity'])
    for name,value in packet['robot_fields'].items():setattr(env.robot,name,deepcopy(value))
    env.take_action_cnt=packet['take_action_cnt'];env.eval_success=packet['eval_success']
    if readback(env)!=packet['state']:raise RuntimeError('Visual mirror differs from worker exposed state')


def native_observation_from_packet(env,packet):
    apply_visual_packet(env,packet)
    before=readback(env)
    observation=env.get_obs()
    if readback(env)!=before:raise RuntimeError('Parent observation render changed physics')
    return observation


def observation_identity(observation):
    import hashlib
    images={}
    for name,row in observation['observation'].items():
        if 'rgb' in row:
            a=np.asarray(row['rgb'])
            images[name]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
    return {'images':images,'proprio_sha256':digest({
        'joint_vector':np.asarray(observation['joint_action']['vector']).tolist(),
        'endpose':observation.get('endpose',{})})}
