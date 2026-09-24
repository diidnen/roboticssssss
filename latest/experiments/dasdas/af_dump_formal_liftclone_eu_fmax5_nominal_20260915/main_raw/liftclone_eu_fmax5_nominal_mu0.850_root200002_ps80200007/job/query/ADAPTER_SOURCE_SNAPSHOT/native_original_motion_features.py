"""Original 8x64 feasibility features, with native sensor/task-name bindings.

Task slot 0 is NEWLY bound to dump_bin_bigbin for a new training run. This is
not permission to load LIBERO task-0 weights or relabel dump as alphabet soup.
The original four-dimensional capacity and all feature equations are retained.
"""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np

SOURCE=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT/worker.py')
TASK_BINDING={'id':'DUMP_ROOTLOCAL_NEW_TRAINING_NATIVE_TASK_BINDING_V1',
              'slots':['dump_bin_bigbin',None,None,None],
              'requires_fresh_task_bound_feasibility_weights':True,
              'original_LIBERO_checkpoint_task_slots_compatible':False}


def sequence_builder():
    import sys
    sys.path.insert(0,str(SOURCE.parent))
    original=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='make_sequence')
    fn=deepcopy(original)
    edits=[]
    for node in ast.walk(fn):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='joints':
            edits.append((node,'value',node.value))
            node.value=ast.parse("np.asarray(saved['native_actual_finger_joints']).reshape(-1)",mode='eval').body
        elif isinstance(node,ast.Compare) and ast.unparse(node.left)=='len(joints)':
            if ast.unparse(node)!='len(joints) != 9': raise RuntimeError('Original joint schema drift')
            edits.append((node,'comparators',node.comparators));node.comparators=[ast.Constant(2)]
        elif isinstance(node,ast.Call) and ast.unparse(node)=='(0, 1, 5, 6).index(task)':
            edits.append((node,'func',node.func))
            node.func=ast.parse("TASK_BINDING['slots'].index",mode='eval').body
    if len(edits)!=3: raise RuntimeError('Unexpected original feature bindings: '+str(len(edits)))
    executable=deepcopy(fn)
    for node,key,value in reversed(edits): setattr(node,key,value)
    if ast.dump(fn)!=ast.dump(original):raise RuntimeError('Undeclared feature equation change')
    namespace={'np':np,'TASK_BINDING':TASK_BINDING}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[executable],type_ignores=[])),str(SOURCE),'exec'),namespace)
    return namespace['make_sequence']


def build_features(raw,actual_finger_joints,object_velocity_world,cartesian_chunk_base):
    joints=np.asarray(actual_finger_joints,float)
    velocity=np.asarray(object_velocity_world,float)
    chunk=np.asarray(cartesian_chunk_base,float)
    if joints.shape!=(2,) or velocity.shape!=(3,) or chunk.ndim!=2 or chunk.shape[0]<8 or chunk.shape[1]!=3:
        raise ValueError('Actual two-finger state, measured object velocity and >=8 real FK actions required')
    if not all(np.isfinite(x).all() for x in (joints,velocity,chunk)) or not raw:
        raise ValueError('Missing/nonfinite original-meaning measurement')
    saved={'native_actual_finger_joints':joints,'state':{'rigid_object':{
        raw[-1]['object_id']:{'root_velocity':velocity}}}}
    x=sequence_builder()(raw,saved,'dump_bin_bigbin',chunk)
    if x.shape!=(8,64) or not np.isfinite(x).all():raise RuntimeError('Invalid original sequence')
    return {'sequence':x.tolist(),'source':'ONLINE_VLA_ACTION_CHUNK','candidate_actions_executed':0,
            'task_binding':TASK_BINDING,'motion_acquisition':'real native joint-action forward kinematics in robot base',
            'original_worker_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'original_feature_equations_preserved':True}


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(SOURCE.parent))
    # Numerical binding-only unit test, NOT production observations or labels.
    fn=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='make_sequence')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),ns)
    rng=np.random.default_rng(1309)
    worst=0.
    for _ in range(100):
        joints=rng.uniform(.005,.025,2);velocity=rng.normal(size=3);chunk=rng.normal(size=(50,3))
        row={'object_id':'synthetic_unit_test_only','gripper_opening':joints[0]}
        row.update({side+'_f'+axis:float(rng.normal()) for side in ('left','right') for axis in 'xyz'})
        saved={'state':{'articulation':{'robot':{'joint_position':np.r_[np.zeros(7),joints]}},
                         'rigid_object':{row['object_id']:{'root_velocity':velocity}}}}
        expected=ns['make_sequence']([row],saved,0,chunk)
        actual=np.asarray(build_features([row],joints,velocity,chunk)['sequence'],np.float32)
        assert np.array_equal(expected,actual)
        worst=max(worst,float(abs(expected-actual).max()))
    print('ORIGINAL_MOTION_EQUATION_PARITY_PASSED',100,worst)
