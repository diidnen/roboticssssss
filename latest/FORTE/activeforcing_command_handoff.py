"""Preserve the probe action command across handoff, not measured aperture.

Adapts only the initialization in the existing full-task runner. Its phase
schedule, force servo, arm trajectory and release actions remain unchanged.
"""
import ast
import hashlib
import inspect
import json
import numpy as np


def command_from_probe(last_action,controller_command,closed,opened):
    action=np.asarray(last_action,float)
    if action.shape!=(1,13) or not np.isfinite(action).all():raise ValueError('Invalid probe action')
    value=float(action[0,6])
    if not closed<=value<=opened:raise ValueError('Probe command outside controller bounds')
    if abs(value-float(controller_command))>1e-8:raise ValueError('Probe/controller command mismatch')
    return value


def adapt_branch_runner(module):
    original=inspect.getsource(module.downstream_branch)
    tree=ast.parse(original);function=tree.body[0]
    matches=[i for i,node in enumerate(function.body) if isinstance(node,ast.Try) and 'gripper_ids' in ast.unparse(node)]
    if len(matches)!=1:raise ValueError('Unrecognized historical handoff initialization')
    index=matches[0];before=ast.dump(function.body[index],include_attributes=False)
    function.body[index]=ast.parse('d_pred = float(handoff_cmd)').body[0]
    function.name='downstream_branch_current_handoff'
    function.args.kwonlyargs.append(ast.arg(arg='handoff_cmd'));function.args.kw_defaults.append(None)
    ast.fix_missing_locations(tree)
    source=ast.unparse(tree)
    def semantic(node):
        if isinstance(node,ast.AST):
            return {'node':type(node).__name__,**{k:semantic(v) for k,v in ast.iter_fields(node)
                if not (k=='type_params' and not v)}}
        if isinstance(node,list):return [semantic(v) for v in node]
        return node
    # ast.unparse whitespace/tuple-parentheses changed between Python3.10
    # (host) and 3.11 (Isaac). Compare structure, not pretty-printed text.
    semantic_hash=hashlib.sha256(json.dumps(semantic(tree),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    namespace=dict(vars(module));exec(compile(tree,'<command_continuity_branch>','exec'),namespace)
    return namespace[function.name],{'original_source_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'adapted_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'adapted_semantic_ast_sha256':semantic_hash,
        'changed_top_level_body_nodes':1,'removed_initialization_ast':before,
        'change':'Measured-joint initialization replaced by checked last-probe action command parameter',
        'unchanged':'All subsequent branch phase/action/controller/servo code',
        'source':source}
