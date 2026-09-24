"""Reuse every original raw gate check with only the diagnostic output path bound."""
import ast
from copy import deepcopy
import inspect
import textwrap
import audit_canonical_online_gate as original


def bound_audit():
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.main)))
    baseline=ast.dump(tree);edits=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='canonical_ready_online_gate_v1':
            edits.append(node);node.value='rim20_online_gate_v1'
    if len(edits)!=1:raise ValueError('Unexpected audit binding count')
    executable=deepcopy(tree)
    for node in edits:node.value='canonical_ready_online_gate_v1'
    if ast.dump(tree)!=baseline:raise ValueError('Undeclared audit change')
    namespace=dict(original.__dict__);namespace['__file__']=__file__
    exec(compile(ast.fix_missing_locations(executable),__file__,'exec'),namespace)
    return namespace['main']


if __name__=='__main__':bound_audit()()
