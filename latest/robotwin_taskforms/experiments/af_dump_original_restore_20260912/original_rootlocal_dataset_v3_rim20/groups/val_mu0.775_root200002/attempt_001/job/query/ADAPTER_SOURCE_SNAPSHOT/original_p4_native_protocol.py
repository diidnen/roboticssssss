"""Original P4 bodies with auditable, reversible geometry/clock binding only."""
import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType
import numpy as np

SOURCE = Path('/home/exouser/FORTE/activeforcing_current_probe.py')
FUNCTIONS = {'_aa', '_quat_angle', '_f', '_unit', '_quat_apply_np', '_quat_inv_np',
             '_frame_to_base', '_current_contact_frame', '_make_action', '_force_servo',
             '_interp', '_dbg', '_tactile_summaries', 'run_probe_episode'}


def load_protocol():
    tree = ast.parse(SOURCE.read_text())
    body = []
    constants = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and
                x.id == 'APPROACH_STEPS' for x in ast.walk(node.targets[0])):
            constants = True
        if constants and isinstance(node, ast.If):
            constants = False
        if constants and isinstance(node, ast.Assign):
            body.append(deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            body.append(deepcopy(node))
        elif isinstance(node, ast.ClassDef) and node.name == 'ProbeStep':
            body.append(deepcopy(node))
    original = ast.dump(ast.Module(body=deepcopy(body), type_ignores=[]))
    episode = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == 'run_probe_episode')
    geometry_hook = ast.parse('pregrasp, grasp = env.bind_geometry(pregrasp, grasp)').body[0]
    index = next(i for i,n in enumerate(episode.body) if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'd_pred')
    episode.body.insert(index, geometry_hook)
    loop = next(n for n in episode.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
                and any(isinstance(x,ast.Name) and x.id=='phase' for x in n.target.elts))
    refresh_hook = ast.parse("if phase == 'descend':\n    grasp[:] = env.refresh_grasp()\n    target = grasp").body[0]
    loop.body.insert(0, refresh_hook)
    observer = next(n for n in episode.body if isinstance(n,ast.FunctionDef) and n.name=='observe_and_record')
    replacements = []
    for node in ast.walk(observer):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id == 't_s':
            replacements.append((node,'value',deepcopy(node.value)))
            node.value = ast.parse('env.elapsed_s',mode='eval').body
        # Only the two measured-time uses inside the observer. The outer
        # historical summary remains unchanged and is not a belief input.
        if isinstance(node,ast.BinOp) and isinstance(node.right,ast.Name) and node.right.id=='dt':
            replacements.append((node,'right',deepcopy(node.right)))
            node.right = ast.parse('env.last_dt_s',mode='eval').body
    # t_s's old value may have been traversed before replacement; require the
    # two intended marker/integral expressions to exist in the live AST.
    active_clock_uses = sum(isinstance(n,ast.Attribute) and n.attr=='last_dt_s' for n in ast.walk(observer))
    if active_clock_uses != 2:
        raise RuntimeError('Original observer clock structure drifted: '+str(active_clock_uses))
    adapted = ast.Module(body=body,type_ignores=[])
    executable = deepcopy(adapted)
    # Exact AST equality after removing ONLY declared hooks and undoing clock
    # expressions proves all original decisions/equations/defaults survived.
    episode.body.remove(geometry_hook)
    loop.body.remove(refresh_hook)
    for node,key,value in reversed(replacements): setattr(node,key,value)
    if ast.dump(adapted) != original:
        raise RuntimeError('Undeclared P4 source change')
    module = ModuleType('_original_p4_native_protocol')
    sys.modules[module.__name__] = module
    module.__dict__.update(np=np, os=os, dataclass=dataclass,
        TASK_ID='dump_bin_bigbin', OBJ_NAME='deskbin', BASKET_NAME='dustbin', VARIANT='P4B')
    exec(compile(ast.fix_missing_locations(executable), str(SOURCE), 'exec'),module.__dict__)
    module._pose_in_base = lambda env,name: env.pose_in_base(name)
    module._apply_friction = lambda env,name,mu: env.verify_friction(name,mu)
    module.binding_receipt = {
        'original_source':str(SOURCE), 'original_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'original_AST_recovered_exactly':True, 'actual_dt_expression_count':active_clock_uses,
        'geometry_hooks':['initial native rim pregrasp/grasp','refresh rim grasp before descent'],
        'unchanged_functions':sorted(FUNCTIONS), 'task_binding':'dump_bin_bigbin (not a LIBERO task ID)',
        'formal_collection_gate_passed':False}
    return module


if __name__=='__main__':
    import json
    print(json.dumps(load_protocol().binding_receipt,indent=2))
