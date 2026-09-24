"""Preserve a complete v2 group; only change the old packer's archive prefix."""
import argparse
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import textwrap
import pack_original_collected_group as original
from rootlocal_collection_contract import read


def bound_pack():
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.pack)));baseline=ast.dump(tree);edits=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='af_dump_original_':
            edits.append(node);node.value='af_dump_canonical_v2_'
    if len(edits)!=1:raise ValueError('Archive basename source drift')
    executable=deepcopy(tree)
    for node in edits:node.value='af_dump_original_'
    if ast.dump(tree)!=baseline:raise ValueError('Undeclared archive content change')
    namespace=dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(executable),__file__,'exec'),namespace)
    return namespace['pack']


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('attempt',type=Path);args=ap.parse_args()
    context=read(args.attempt/'CONTEXT.json');manifest=read(context['runtime_manifest_path'])
    if manifest.get('initialization_binding')!='CANONICAL_OPEN_READY_V2':raise ValueError('Not canonical v2 data')
    bound_pack()(args.attempt)
