"""Original queue, maxF8 implementation and the predeclared two policy seeds."""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import shutil
import sys
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'af_dump_original_restore_20260912'
sys.path.insert(0,str(OLD))
import run_original_rootlocal_inference as original


def bind():
    tree=ast.parse((OLD/'run_original_rootlocal_inference.py').read_text())
    function=deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run'))
    baseline=ast.dump(function);edits=[]
    for node in ast.walk(function):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='source_hashes':
            edits.append((node,'value',node.value));node.value=ast.parse('_v4_sources()',mode='eval').body
        elif isinstance(node,ast.Constant) and node.value=='infer_original_rootlocal.py':
            edits.append((node,'value',node.value));node.value=str(HERE/'infer_v4.py')
        elif isinstance(node,ast.Compare) and ast.unparse(node.left)=='len(contexts)':
            if ast.literal_eval(node.comparators[0])!=4:raise ValueError('Original context-count drift')
            edits.append((node,'comparators',node.comparators));node.comparators=[ast.Constant(8)]
        elif isinstance(node,ast.Dict):
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='paired_rollouts':
                    if ast.literal_eval(node.values[i])!=24:raise ValueError('Original rollout-count drift')
                    previous=list(node.values);edits.append((node,'values',previous));node.values=list(previous);node.values[i]=ast.Constant(48)
    if len(edits)!=5:raise ValueError('Unexpected queue binding count')
    executable=deepcopy(function)
    for node,key,value in edits:setattr(node,key,value)
    if ast.dump(function)!=baseline:raise ValueError('Unexpected queue algorithm change')
    namespace=dict(original.__dict__);namespace['__file__']=str(Path(__file__).resolve())
    namespace['_v4_sources']=lambda:{str(HERE/name):original.sha(HERE/name) for name in
        ['run_v4_inference.py','infer_v4.py','maxf8_runtime.py','max_force_utility.py','audit_v4_inference.py']}
    oldwrite=namespace['write']
    def tagged_write(path,value):
        if Path(path).name=='FINAL_INFERENCE_RESULTS.json':
            value={**value,'utility_normalization_N':8.0,'physical_friction_settings':4,
                   'decision_contexts':8,'policy_repeats_per_friction':2}
        oldwrite(path,value)
    namespace['write']=tagged_write
    exec(compile(ast.fix_missing_locations(ast.Module(body=[executable],type_ignores=[])),str(__file__),'exec'),namespace)
    return namespace['run']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['dataset','models','out']:parser.add_argument(name,type=Path)
    args=parser.parse_args()
    if shutil.disk_usage(args.out.parent).free<6*1024**3:raise RuntimeError('Reserve 6GiB before starting complete confirmation')
    bind()(args.dataset,args.models,args.out)
