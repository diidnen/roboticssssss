"""Original native online execution with the user-corrected maxF8 selector/scoring."""
import ast
from copy import deepcopy
from pathlib import Path
import os
import sys
HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'af_dump_original_restore_20260912'
sys.path.insert(0,str(OLD))
import infer_original_rootlocal as original
from maxf8_runtime import MaxF8Feasibility
from max_force_utility import expected_utility
from rim20_formal_binding import install


def bind():
    tree=ast.parse((OLD/'infer_original_rootlocal.py').read_text())
    functions=[deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef)]
    baseline=ast.dump(ast.Module(body=functions,type_ignores=[]))
    changes=[]
    for function in functions:
        if function.name!='qualify':continue
        for node in ast.walk(function):
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='utility':
                changes.append((node,node.value))
                node.value=ast.parse('float(expected_utility(y, force, 8.0))',mode='eval').body
    if len(changes)!=1:raise ValueError('Unexpected original outcome-scoring layout')
    executable=deepcopy(functions)
    for node,value in changes:node.value=value
    if ast.dump(ast.Module(body=functions,type_ignores=[]))!=baseline:raise ValueError('Unexpected native inference modification')
    namespace=dict(original.__dict__)
    namespace.update(__file__=str(Path(__file__).resolve()),NativeOriginalFeasibility=MaxF8Feasibility,expected_utility=expected_utility)
    original_write=namespace['write']
    def tagged_write(path,value):
        if Path(path).name=='AF_INFERENCE_RESULT.json':
            value={**value,'utility_normalization_N':8.0,
                   'utility_definition':'p*(maxF-F)/maxF-(1-p)',
                   'original_execution_functions_preserved_except_outcome_utility':True}
        original_write(path,value)
    namespace['write']=tagged_write
    exec(compile(ast.fix_missing_locations(ast.Module(body=executable,type_ignores=[])),str(__file__),'exec'),namespace)
    return namespace


if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    install(); namespace=bind(); original.base.qualify=namespace['qualify'];original.base.main()
