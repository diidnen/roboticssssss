"""Conditional v3 geometry binding; original candidate physics is unchanged."""
import ast
from copy import deepcopy
import inspect
import os
from pathlib import Path
import textwrap
import diagnose_canonical_rim_alignment as candidate
import canonical_open_ready_query as ready_candidate
from canonical_formal_binding import formal_ready_method
import qualify_original_p4_native as query
from rootlocal_collection_contract import read,sha,write,verify_runtime


class FormalReadyReceipt(ready_candidate.CanonicalOpenReadyQuery):
    # Existing exact-AST binding changes only the canonical receipt's flag.
    canonicalize_open_ready=formal_ready_method()


def formal_rim_class():
    # Compile the whole class to preserve the zero-argument super() class cell.
    tree=ast.parse(textwrap.dedent(inspect.getsource(candidate.AlignedDiagnosticQuery)))
    baseline=ast.dump(tree);edits=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Dict):
            for index,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='diagnostic_only':
                    if ast.unparse(node.values[index])!='True':raise ValueError('Rim receipt drift')
                    edits.append((node,index,node.values[index]));node.values[index]=ast.Constant(False)
    if len(edits)!=1:raise ValueError('Unexpected receipt binding count')
    executable=deepcopy(tree)
    for node,index,previous in edits:node.values[index]=previous
    if ast.dump(tree)!=baseline:raise ValueError('Undeclared rim algorithm change')
    namespace=dict(candidate.__dict__)
    namespace['CanonicalOpenReadyQuery']=FormalReadyReceipt
    exec(compile(ast.fix_missing_locations(executable),__file__,'exec'),namespace)
    return namespace['AlignedDiagnosticQuery']


BoundRimQuery=formal_rim_class()


class FormalRim20Query(BoundRimQuery):
    def __init__(self,env,out):
        collection=os.environ.get('AF_COLLECTION_CONTEXT');inference=os.environ.get('AF_INFERENCE_CONTEXT')
        if bool(collection)==bool(inference):raise ValueError('Exactly one formal context required')
        context=read(collection) if collection else read(inference)['context']
        manifest=verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
        if manifest.get('initialization_binding')!='CANONICAL_OPEN_READY_RIM20_V3':raise ValueError('Wrong runtime version')
        if manifest.get('native_rim_pre_dis_offset_m')!=.02:raise ValueError('Wrong fixed geometry')
        for path,digest in manifest['rim20_engineering_gate_hashes'].items():
            if sha(path)!=digest:raise ValueError('Qualified engineering evidence changed')
        if os.environ.get('AF_DIAGNOSTIC_RIM_OFFSET_M')!='.02':raise ValueError('Wrong executed geometry')
        super().__init__(env,out)
        write(Path(out)/'FORMAL_INITIALIZATION_BINDING.json',{
            'binding':'CANONICAL_OPEN_READY_RIM20_V3','context':context,'pre_dis_offset_m':.02,
            'candidate_algorithm_source_sha256':sha(candidate.__file__),'binding_source_sha256':sha(__file__),
            'candidate_algorithm_AST_recovered_exactly':True,'only_candidate_receipt_flags_changed':True,
            'original_P4_and_squeeze_unchanged':True,'same_geometry_for_TRAIN_VAL_TEST':True})


def install():
    os.environ['AF_DIAGNOSTIC_RIM_OFFSET_M']='.02'
    query.NativeP4QueryEnv=FormalRim20Query
