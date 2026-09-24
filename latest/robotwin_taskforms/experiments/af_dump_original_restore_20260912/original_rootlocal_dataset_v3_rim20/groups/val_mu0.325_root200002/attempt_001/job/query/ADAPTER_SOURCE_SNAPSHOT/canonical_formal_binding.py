"""Qualified preparation-only binding; original query/control code unchanged."""
import ast
from copy import deepcopy
import inspect
import os
from pathlib import Path
import textwrap
import canonical_open_ready_query as candidate
import qualify_original_p4_native as query
from rootlocal_collection_contract import HERE, read, sha, write, verify_runtime


def formal_ready_method():
    tree=ast.parse(textwrap.dedent(inspect.getsource(candidate.CanonicalOpenReadyQuery.canonicalize_open_ready)))
    baseline=ast.dump(tree);edits=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Dict):
            for index,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='diagnostic_only':
                    if ast.unparse(node.values[index])!='True':raise ValueError('Candidate receipt drift')
                    edits.append((node,index,node.values[index]));node.values[index]=ast.Constant(False)
    if len(edits)!=1:raise ValueError('Unexpected canonical receipt edit count')
    executable=deepcopy(tree)
    for node,index,old in edits:node.values[index]=old
    if ast.dump(tree)!=baseline:raise ValueError('Undeclared canonical algorithm change')
    namespace=dict(candidate.__dict__)
    exec(compile(ast.fix_missing_locations(executable),__file__,'exec'),namespace)
    return namespace['canonicalize_open_ready']


class FormalCanonicalQuery(candidate.CanonicalOpenReadyQuery):
    canonicalize_open_ready=formal_ready_method()

    def __init__(self,env,out):
        collection=os.environ.get('AF_COLLECTION_CONTEXT');inference=os.environ.get('AF_INFERENCE_CONTEXT')
        if bool(collection)==bool(inference):raise ValueError('Exactly one formal context is required')
        context=read(collection) if collection else read(inference)['context']
        manifest=verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
        if manifest.get('initialization_binding')!='CANONICAL_OPEN_READY_V2':raise ValueError('Wrong runtime version')
        for path,digest in manifest['canonical_engineering_gate_hashes'].items():
            if sha(path)!=digest:raise ValueError('Canonical qualification evidence changed')
        super().__init__(env,out)
        write(Path(out)/'FORMAL_INITIALIZATION_BINDING.json',{
            'binding':'CANONICAL_OPEN_READY_V2','context':context,
            'candidate_algorithm_source_sha256':sha(candidate.__file__),
            'binding_source_sha256':sha(__file__),
            'algorithm_AST_recovered_exactly':True,'only_candidate_receipt_diagnostic_flag_changed':True,
            'original_P4_and_squeeze_unchanged':True})


def install():
    query.NativeP4QueryEnv=FormalCanonicalQuery

