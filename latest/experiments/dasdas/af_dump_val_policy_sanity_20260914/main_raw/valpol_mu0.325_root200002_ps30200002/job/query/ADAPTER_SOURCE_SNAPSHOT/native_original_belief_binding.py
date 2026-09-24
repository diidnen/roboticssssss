"""Native task registry binding for the otherwise unchanged original loader.

No old LIBERO checkpoint may be relabeled as a dump checkpoint. Three newly
trained 58D members with explicit native task/protocol provenance are required.
All original preprocessing, predictions, Gaussian-mixture conditioning and
32/64 quadrature checks execute from the original loader unchanged.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
import torch
from native_original_motion_features import TASK_BINDING

SOURCE=Path('/home/exouser/FORTE/analysis/results/current_multitask58_loader_candidate_20260905/continuous_belief.py')


def native_loader_module():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'));original=ast.dump(tree)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ContinuousBelief')
    rows=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='rows')
    edits=[]
    for node in ast.walk(rows):
        if isinstance(node,ast.SetComp) and ast.unparse(node.elt)=="int(r['task_id'])":
            edits.append((node,'elt',node.elt));node.elt=ast.parse("str(r['task_id'])",mode='eval').body
        elif isinstance(node,ast.Set) and {ast.literal_eval(x) for x in node.elts}=={0,1,5,6}:
            edits.append((node,'elts',node.elts));node.elts=[ast.Constant('dump_bin_bigbin')]
    if len(edits)!=2:raise RuntimeError('Original belief task guard structure changed')
    executable=deepcopy(tree)
    for node,key,value in reversed(edits):setattr(node,key,value)
    if ast.dump(tree)!=original:raise RuntimeError('Undeclared belief algorithm change')
    module=ModuleType('_native_original_continuous_belief_bound')
    module.__file__=str(SOURCE)
    sys.modules[module.__name__]=module
    exec(compile(ast.fix_missing_locations(executable),str(SOURCE),'exec'),module.__dict__)
    module.native_binding_receipt={'task_binding':TASK_BINDING,'changed_nodes':'task registry guards only',
        'original_AST_recovered_exactly':True,
        'original_loader_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return module


def load_native_belief(path,*,manifest_sha256):
    path=Path(path);content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=manifest_sha256:raise ValueError('Native belief manifest changed')
    manifest=json.loads(content)
    module=native_loader_module()
    if manifest.get('native_task_binding')!=TASK_BINDING:
        raise ValueError('Explicit new-task registry is required; legacy LIBERO task aliases are forbidden')
    if manifest.get('native_loader_binding')!=module.native_binding_receipt:
        raise ValueError('Native loader binding source/provenance mismatch')
    if manifest.get('qualified_tasks')!=['dump_bin_bigbin']:
        raise ValueError('Only this native task binding is permitted')
    for entry in manifest.get('checkpoints',[]):
        data=Path(entry['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Native checkpoint changed')
        checkpoint=torch.load(entry['path'],map_location='cpu',weights_only=True)
        if checkpoint.get('native_task_binding')!=TASK_BINDING:
            raise ValueError('Old or unbound checkpoint cannot be loaded for dump')
        if checkpoint.get('root_scope')!=[200002] or not checkpoint.get('training_protocol_sha256'):
            raise ValueError('Missing root-local original training provenance')
    class NativeTaggedBelief(module.ContinuousBelief):
        def rows(self,*args,**kwargs):
            result=super().rows(*args,**kwargs)
            result['native_task_binding']=deepcopy(TASK_BINDING)
            result['root_scope']=[200002]
            return result
    return NativeTaggedBelief(path,manifest_sha256=manifest_sha256)


if __name__=='__main__':
    module=native_loader_module()
    print(json.dumps(module.native_binding_receipt,indent=2))
    legacy=Path('/home/exouser/FORTE/analysis/results/current_matched_runtime_collection_v5_20260906/BELIEF_MANIFEST.json')
    try:load_native_belief(legacy,manifest_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest())
    except ValueError as exc:
        if 'new-task registry' not in str(exc):raise
        print('LEGACY_CHECKPOINT_RELABELING_REJECTED')
    else:raise AssertionError('Unqualified legacy task model accepted')
