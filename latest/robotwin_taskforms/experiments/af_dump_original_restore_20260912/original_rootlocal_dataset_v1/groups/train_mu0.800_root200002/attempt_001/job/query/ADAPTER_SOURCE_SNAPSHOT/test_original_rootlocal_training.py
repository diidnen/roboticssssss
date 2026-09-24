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
    print('ORIGINAL_BELIEF_FULL_UPDATE_LOOP_AST_EXACT')
    print('ORIGINAL_BELIEF_AND_FEASIBILITY_DISTINCT_NORMALIZATION_PARITY_PASSED')
    print('ORIGINAL_ONLINE_INFERENCE_HARNESS_AST_BINDING_PASSED')


if __name__=='__main__':main()
