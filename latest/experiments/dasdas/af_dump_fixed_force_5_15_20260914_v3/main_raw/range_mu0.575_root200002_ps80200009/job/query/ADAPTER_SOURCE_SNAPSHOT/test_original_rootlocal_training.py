"""Recipe parity tests on synthetic arrays; never create experimental labels."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import torch
from rootlocal_collection_contract import FORTE,HERE
from train_original_rootlocal import BELIEF_TRAINER,FEAS_TRAINER,recipe


def function(tree,name):
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


def main():
    original=ast.parse(BELIEF_TRAINER.read_text(encoding='utf-8'))
    native=ast.parse((HERE/'train_original_rootlocal.py').read_text(encoding='utf-8'))
    oldloop=next(n for n in ast.walk(function(original,'train')) if isinstance(n,ast.For)
                 and ast.unparse(n.target)=='epoch')
    newloop=next(n for n in ast.walk(function(native,'train_belief')) if isinstance(n,ast.For)
                 and ast.unparse(n.target)=='epoch')
    if ast.dump(oldloop)!=ast.dump(newloop):
        raise AssertionError('Belief optimizer/update/selection loop differs from original source')
    normfn=function(original,'normalize');namespace={'np':np}
    exec(compile(ast.Module(body=[normfn],type_ignores=[]),str(BELIEF_TRAINER),'exec'),namespace)
    rng=np.random.default_rng(913);rows=[]
    for i,length in enumerate((209,212,207,211)):
        x=rng.normal(size=(length,58)).astype(np.float32);x[:,0]=.3;x[:,1]=0
        rows.append({'root':200002,'split':'TRAIN' if i<3 else 'VAL','raw':x})
    expected=namespace['normalize'](deepcopy(rows))
    joined=np.concatenate([r['raw'] for r in rows if r['split']=='TRAIN'])
    mean=joined.mean(0);std=np.maximum(joined.std(0),1e-6)
    np.testing.assert_array_equal(mean,np.asarray(expected['mean'],np.float32))
    np.testing.assert_array_equal(std,np.asarray(expected['std'],np.float32))
    original_feas=ast.parse(FEAS_TRAINER.read_text(encoding='utf-8'))
    fn=function(original_feas,'fit_normalization');namespace={'np':np}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(FEAS_TRAINER),'exec'),namespace)
    values=rng.normal(size=(13,8,64)).astype(np.float32);values[:,:,0]=0;values[:,:,1]=.3
    oldmean,oldstd=namespace['fit_normalization']([{'x':x} for x in values])
    mean=values.mean(axis=(0,1)).astype(np.float32);std=values.std(axis=(0,1)).astype(np.float32);std[std<1e-6]=1.
    np.testing.assert_array_equal(mean,oldmean);np.testing.assert_array_equal(std,oldstd)
    assert recipe()['seeds']==[0,1,2] and recipe()['updates']==80
    from infer_original_rootlocal import bound_execution_module
    binding=bound_execution_module().inference_binding_receipt
    assert binding['original_AST_recovered_exactly'] and binding['control_and_physics_changed'] is False
    from infer_original_rootlocal import restore_initial_camera_cache
    from native_visual_mirror import observation_identity
    from qualify_native_interfaces import digest
    from types import SimpleNamespace
    common={'observation':{'head':{'rgb':np.zeros((4,4,3),np.uint8)}},
            'joint_action':{'vector':np.arange(14,dtype=np.float32)},'endpose':{}}
    live=deepcopy(common);live['observation']['head']['rgb'][:]=123
    env=SimpleNamespace(_af_locked_native_action_count=0,_af_locked_state_sha256=digest({'unit_only':1}),
        _af_locked_observation_identity=observation_identity(common),
        _af_locked_camera_cache={'head':common['observation']['head']['rgb']})
    packet={'take_action_cnt':0,'state':{'unit_only':1}}
    restored=restore_initial_camera_cache(env,packet,lambda e,p:deepcopy(live))
    assert observation_identity(restored)==observation_identity(common)
    later=restore_initial_camera_cache(env,{**packet,'take_action_cnt':1},lambda e,p:deepcopy(live))
    assert observation_identity(later)==observation_identity(live)
    changed=deepcopy(live);changed['joint_action']['vector'][0]=9
    try:restore_initial_camera_cache(env,packet,lambda e,p:changed)
    except ValueError:pass
    else:raise AssertionError('Changed live nonimage input was accepted')
    try:restore_initial_camera_cache(env,{**packet,'state':{'unit_only':2}},lambda e,p:deepcopy(live))
    except ValueError:pass
    else:raise AssertionError('Wrong pre-action physical state was accepted')
    print('ORIGINAL_BELIEF_FULL_UPDATE_LOOP_AST_EXACT')
    print('ORIGINAL_BELIEF_AND_FEASIBILITY_DISTINCT_NORMALIZATION_PARITY_PASSED')
    print('ORIGINAL_ONLINE_INFERENCE_HARNESS_AST_BINDING_PASSED')
    print('ORIGINAL_INITIAL_CAMERA_CACHE_ONLY_AND_LIVE_PROPRIO_GUARDS_PASSED')


if __name__=='__main__':main()
