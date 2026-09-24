"""Explicit execution-state capture/restore, without episode reset events.

Restores exposed state, NOT undocumented PhysX solver/contact warm-start state.
An uninterrupted-versus-restored action replay must independently qualify it.
No controller algorithm, gains, action, or contact measurement is changed.
"""
import random
import numpy as np
import torch


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if value is None or type(value) in (int, float, bool, str):
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(clone_value(v) for v in value)
    if isinstance(value, dict):
        return {k:clone_value(v) for k,v in value.items()}
    raise TypeError(type(value).__name__)


def attributes(obj):
    result={}
    for name,value in vars(obj).items():
        try:result[name]=clone_value(value)
        except TypeError:pass
    return result


def restored_value(current,saved,device):
    if isinstance(saved,torch.Tensor):
        target_device=current.device if isinstance(current,torch.Tensor) else device
        value=saved.to(target_device)
        if isinstance(current,torch.Tensor) and current.shape==value.shape and current.dtype==value.dtype:
            current.copy_(value);return current
        return value.clone()
    if isinstance(saved,np.ndarray):return saved.copy()
    if isinstance(saved,dict):
        existing=current if isinstance(current,dict) else {}
        return {k:restored_value(existing.get(k),v,device) for k,v in saved.items()}
    if isinstance(saved,(list,tuple)):
        existing=current if isinstance(current,(list,tuple)) else []
        return type(saved)(restored_value(existing[i] if i<len(existing) else None,v,device) for i,v in enumerate(saved))
    return saved


def apply_attributes(obj,saved,device):
    for name,value in saved.items():
        setattr(obj,name,restored_value(getattr(obj,name,None),value,device))


def object_registry(env):
    objects={}
    def add(name,obj):
        if obj is None:return
        objects[name]=obj
        # TimestampedBuffer/CircularBuffer data, no recursion into env/assets.
        for key,value in vars(obj).items():
            if type(value).__name__ in ('TimestampedBuffer','CircularBuffer'):
                objects[name+'/'+key]=value
    for name in ('action_manager','termination_manager','reward_manager','command_manager','event_manager','observation_manager'):
        add(name,getattr(env,name))
    arm=env.action_manager.get_term('arm_action')
    add('arm_action',arm);add('arm_ik',arm._ik_term)
    add('arm_ik_controller',getattr(arm._ik_term,'_ik_controller',None))
    for name,asset in {**env.scene.articulations,**env.scene.rigid_objects}.items():
        add('asset/'+name,asset);add('asset_data/'+name,asset.data)
        for key,actuator in getattr(asset,'actuators',{}).items():add('actuator/'+name+'/'+key,actuator)
    for name,sensor in env.scene.sensors.items():
        if type(sensor).__name__ in ('ContactSensor','FrameTransformer'):
            _=sensor.data
            add('sensor/'+name,sensor);add('sensor_data/'+name,sensor._data)
    return objects


ENV_FIELDS=('episode_length_buf','common_step_counter','_sim_step_counter',
            'reset_buf','reset_terminated','reset_time_outs','reward_buf')


def capture(env):
    if env.num_envs!=1:raise ValueError('Only single-environment snapshot restoration is qualified')
    registry=object_registry(env)
    robot=env.scene['robot']
    properties={}
    for name,asset in {**env.scene.articulations,**env.scene.rigid_objects}.items():
        view=asset.root_physx_view
        properties[name]={'material':clone_value(view.get_material_properties())}
    return {'state':clone_value(env.scene.get_state(is_relative=True)),
        'objects':{k:attributes(v) for k,v in registry.items()},
        'environment':{k:clone_value(getattr(env,k)) for k in ENV_FIELDS if hasattr(env,k)},
        'joint_targets':{k:clone_value(getattr(robot.data,k)) for k in ('joint_pos_target','joint_vel_target','joint_effort_target')},
        'materials':properties,
        'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
               'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
        'excluded':['unexposed PhysX contact/solver warm-start state','renderer internal state'],
        'restore_method':'scene.reset_to + exposed action/sensor/manager/cache/target/RNG state; NO env.reset_to'}


def restore(env,saved):
    # Do NOT call env.reset_to: it performs episode reset/randomization and
    # destroys controller latch/filter/history and episode time.
    registry=object_registry(env)
    if set(registry)!=set(saved['objects']):raise ValueError('Snapshot object layout changed')
    ids=torch.tensor([0],device=env.device)
    state=restored_value(None,saved['state'],env.device)
    env.scene.reset_to(state,ids,is_relative=True)
    robot=env.scene['robot']
    for key,method in [('joint_pos_target','set_joint_position_target'),
                       ('joint_vel_target','set_joint_velocity_target'),
                       ('joint_effort_target','set_joint_effort_target')]:
        getattr(robot,method)(saved['joint_targets'][key].to(env.device))
    env.scene.write_data_to_sim()
    env.sim.forward()  # kinematic update only; no candidate/simulation step
    for name,values in saved['objects'].items():apply_attributes(registry[name],values,env.device)
    apply_attributes(env,saved['environment'],env.device)
    # No reset event runs, so materials must remain equal, not be silently fixed.
    for name,values in saved['materials'].items():
        actual=env.scene[name].root_physx_view.get_material_properties().cpu()
        if not torch.equal(actual,values['material']):raise ValueError('Material changed during restore: '+name)
    random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy'])
    torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])


def max_difference(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b):return float('inf')
        return max([max_difference(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(tuple,list)):
        if len(a)!=len(b):return float('inf')
        return max([max_difference(x,y) for x,y in zip(a,b)] or [0.])
    if isinstance(a,(torch.Tensor,np.ndarray)):
        x=np.asarray(a.cpu() if isinstance(a,torch.Tensor) else a)
        y=np.asarray(b.cpu() if isinstance(b,torch.Tensor) else b)
        if x.shape!=y.shape:return float('inf')
        if x.size==0:return 0.
        if x.dtype.kind not in 'biufc':return 0. if np.array_equal(x,y) else float('inf')
        x=x.astype(float);y=y.astype(float)
        if not np.array_equal(np.isnan(x),np.isnan(y)) or not np.array_equal(np.isinf(x),np.isinf(y)):
            return float('inf')
        finite=np.isfinite(x)&np.isfinite(y)
        if not np.array_equal(x[~finite],y[~finite],equal_nan=True):return float('inf')
        return float(np.max(np.abs(x[finite]-y[finite]))) if finite.any() else 0.
    if type(a) in (float,int,bool):
        if np.isnan(a) and np.isnan(b):return 0.
        if a==b:return 0.
        return abs(float(a)-float(b))
    return 0. if a==b else float('inf')
