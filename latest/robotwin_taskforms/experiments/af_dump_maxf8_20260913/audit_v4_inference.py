"""Independent raw audit with maxF8 arithmetic and eight paired decision contexts."""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'af_dump_original_restore_20260912'
sys.path.insert(0,str(OLD))
import audit_original_inference_results as original


def bind():
    tree=ast.parse((OLD/'audit_original_inference_results.py').read_text())
    functions=[deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef)]
    baseline=ast.dump(ast.Module(body=functions,type_ignores=[]));edits=[]
    for function in functions:
        if function.name=='check_decision':
            for node in ast.walk(function):
                if isinstance(node,ast.Constant) and isinstance(node.value,float) and node.value==5.0:
                    edits.append((node,'value',node.value));node.value=8.0
        elif function.name=='audit':
            for node in ast.walk(function):
                if isinstance(node,ast.Compare) and ast.unparse(node.left)=='len(contexts)':
                    previous=node.comparators[0]
                    if ast.literal_eval(previous)!=4:raise ValueError('Original case-count check changed')
                    edits.append((node,'comparators',node.comparators));node.comparators=[ast.Constant(8)]
                elif isinstance(node,ast.Compare) and ast.unparse(node.left)=="final['paired_rollouts']":
                    if ast.literal_eval(node.comparators[0])!=24:raise ValueError('Original rollout-count check changed')
                    edits.append((node,'comparators',node.comparators));node.comparators=[ast.Constant(48)]
                elif isinstance(node,ast.Dict):
                    for i,key in enumerate(node.keys):
                        if isinstance(key,ast.Constant) and key.value=='original_utility':
                            values=list(node.values)
                            edits.append((node,'values',values))
                            node.values=list(values)
                            node.values[i]=ast.parse('y*(8.0-force)/8.0 + (1-y)*-1.0',mode='eval').body
    if len(edits)!=6:raise ValueError('Unexpected independent-audit correction count')
    executable=deepcopy(functions)
    for node,key,value in edits:setattr(node,key,value)
    if ast.dump(ast.Module(body=functions,type_ignores=[]))!=baseline:raise ValueError('Unexpected raw-audit change')
    namespace=dict(original.__dict__);namespace['__file__']=str(Path(__file__).resolve())
    exec(compile(ast.fix_missing_locations(ast.Module(body=executable,type_ignores=[])),str(__file__),'exec'),namespace)
    return namespace


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('inference',type=Path)
    bind()['audit'](parser.parse_args().inference)
