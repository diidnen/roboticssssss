"""Exact original queue functions with only the child entrypoint changed."""
import ast
from copy import deepcopy
import inspect
import textwrap


def bind(module,old,new,expected_count=1):
    tree=ast.parse(textwrap.dedent(inspect.getsource(module.run)))
    baseline=ast.dump(tree);edits=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value==old:
            edits.append(node);node.value=new
    if len(edits)!=expected_count:raise ValueError(f'Expected {expected_count} entrypoint/source-lock bindings, found {len(edits)}')
    executable=deepcopy(tree)
    for node in edits:node.value=old
    if ast.dump(tree)!=baseline:raise ValueError('Undeclared queue change')
    namespace=dict(module.__dict__)
    exec(compile(ast.fix_missing_locations(executable),__file__,'exec'),namespace)
    return namespace['run']
