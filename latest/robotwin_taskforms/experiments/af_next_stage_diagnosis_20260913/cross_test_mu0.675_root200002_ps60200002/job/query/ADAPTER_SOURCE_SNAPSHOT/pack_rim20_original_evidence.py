"""Original full/partial evidence packers with explicit non-colliding v3 names."""
import argparse
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import textwrap
import pack_original_collected_group as full
import pack_canonical_terminal_snapshot as partial
from rootlocal_collection_contract import read,verify_runtime


def bind(function,module,replacements):
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    baseline=ast.dump(tree);edits=[];counts={key:0 for key in replacements}
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in replacements:
            old=node.value;edits.append((node,old));counts[old]+=1;node.value=replacements[old]
    if any(count!=1 for count in counts.values()):raise ValueError('Unexpected packer binding count')
    executable=deepcopy(tree)
    for node,old in edits:node.value=old
    if ast.dump(tree)!=baseline:raise ValueError('Undeclared archive content change')
    namespace=dict(module.__dict__)
    exec(compile(ast.fix_missing_locations(executable),__file__,'exec'),namespace)
    return namespace[function.__name__]


pack_full=bind(full.pack,full,{'af_dump_original_':'af_dump_rim20_v3_'})
pack_partial=bind(partial.main,partial,{'CANONICAL_OPEN_READY_V2':'CANONICAL_OPEN_READY_RIM20_V3',
                                     'af_dump_canonical_v2_':'af_dump_rim20_v3_'})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('attempt',type=Path);ap.add_argument('--partial',action='store_true')
    args=ap.parse_args();attempt=args.attempt.resolve();context=read(attempt/'CONTEXT.json')
    manifest=verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
    if manifest.get('initialization_binding')!='CANONICAL_OPEN_READY_RIM20_V3':raise ValueError('Not v3 data')
    (pack_partial if args.partial else pack_full)(attempt)
