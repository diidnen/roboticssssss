"""Six declared diagnostic force branches, all original V4 execution rules."""
import ast
from copy import deepcopy
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
V4=HERE.parent/'af_dump_maxf8_20260913'
sys.path.insert(0,str(V4))
import infer_v4

def bind():
    n=infer_v4.bind()
    source=Path(infer_v4.original.__file__)
    fn=deepcopy(next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='locked_query'))
    baseline=ast.dump(fn); edits=[]
    for node in ast.walk(fn):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='forces':
            edits.append((node,'value',node.value));node.value=ast.parse("_declared_forces(spec)",mode='eval').body
        if isinstance(node,ast.Dict):
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='branch_methods':
                    old=list(node.values);edits.append((node,'values',old));node.values=list(old)
                    node.values[i]=ast.parse("spec['replication']['methods']",mode='eval').body
    if len(edits)!=2:raise ValueError('Original locked-query layout changed')
    executable=deepcopy(fn)
    for node,key,value in edits:setattr(node,key,value)
    if ast.dump(fn)!=baseline:raise ValueError('Unexpected execution edit')
    def forces(spec):
        rp=spec['replication']; protocol=n['read'](HERE/'PROTOCOL.json')
        if rp not in protocol['cases']:raise ValueError('Undeclared diagnostic case')
        if n['sha'](HERE/'PROTOCOL.json')!=spec['replication_protocol_sha256']:raise ValueError('Plan drift')
        for p,h in protocol['source_hashes'].items():
            if n['sha'](p)!=h:raise ValueError('Source drift: '+p)
        values=rp['forces_N']
        if len(values)!=6 or len(rp['methods'])!=6 or not all(.5<=f<=8 for f in values):raise ValueError('Wrong branch schedule')
        return values
    n['_declared_forces']=forces
    oldwrite=n['write']
    def write(path,value):
        if Path(path).name=='AF_INFERENCE_RESULT.json':
            value={**value,'formal_AF_inference':False,'scientific_stage':'POSTHOC_DEVELOPMENT_FORCE_WINDOW',
                   'historical_outcomes_informed_case_and_force_selection':True,
                   'original_models_unchanged':True,'original_execution_unchanged':True,
                   'utility_normalization_N':8.,'utility_definition':'p*(8-F)/8-(1-p)',
                   'scope':'two selected historical contexts; not unbiased method confirmation',
                   'diagnostic_source_sha256':n['sha'](__file__)}
            path=Path(path).with_name('REPLICATION_RESULT.json')
        oldwrite(path,value)
    n['write']=write
    exec(compile(ast.fix_missing_locations(ast.Module(body=[executable],type_ignores=[])),__file__,'exec'),n)
    return n

if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    infer_v4.install();ns=bind();infer_v4.original.base.qualify=ns['qualify'];infer_v4.original.base.main()
