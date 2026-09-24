"""Locked original AF decision before any paired TEST candidate is executed.

The frozen native full-task harness is reused with only branch-count, declared
force-support and descriptive scope bindings. Physics/control code is unchanged.
"""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType
import numpy as np
import torch
import qualify_native_interfaces as base
from rootlocal_collection_contract import HERE,read,write,sha,verify_runtime
from qualify_original_p4_native import qualify as original_query
from native_original_belief_binding import load_native_belief
from native_original_feasibility_binding import NativeOriginalFeasibility
from native_original_motion_features import build_features
from native_cartesian_kinematics import NativeCartesianKinematics
from native_visual_mirror import observation_identity
from admit_original_query import admit


def bound_execution_module():
    source=HERE/'qualify_original_online_forks.py'
    tree=ast.parse(source.read_text(encoding='utf-8'));baseline=ast.dump(tree);edits=[]
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qualify')
    for node in ast.walk(fn):
        if not isinstance(node,ast.Assign) or not isinstance(node.targets[0],ast.Name):continue
        name=node.targets[0].id
        if name=='max_branches':
            if ast.unparse(node.value)!='8 if collection is not None else 3':raise RuntimeError('Harness drift')
            edits.append((node,node.value));node.value=ast.Constant(6)
        elif name=='support':
            edits.append((node,node.value));node.value=ast.parse('(.5,8.)',mode='eval').body
        elif name=='scope':
            edits.append((node,node.value));node.value=ast.Constant('locked original learned AF and paired fixed-force TEST inference')
    if len(edits)!=3:raise RuntimeError('Unexpected inference harness binding')
    executable=deepcopy(tree)
    for node,value in edits:node.value=value
    if ast.dump(tree)!=baseline:raise RuntimeError('Undeclared harness/physics change')
    module=ModuleType('_original_native_paired_inference_harness');module.__file__=str(source)
    sys.modules[module.__name__]=module
    exec(compile(ast.fix_missing_locations(executable),str(source),'exec'),module.__dict__)
    module.inference_binding_receipt={'source_sha256':sha(source),'original_AST_recovered_exactly':True,
        'only_changes':['six paired branches','declared support [.5,8]','descriptive scope'],
        'control_and_physics_changed':False}
    return module


def context_and_models():
    spec=read(os.environ['AF_INFERENCE_CONTEXT'])
    context=spec['context'];models=Path(spec['models'])
    if context['split']!='TEST' or context['root']!=200002:raise ValueError('Only planned same-root TEST is allowed')
    dataset=Path(context['collection_protocol_path']).parent
    lock=read(dataset/'FREEZE_LOCK.json')
    if sha(dataset/'CONTEXTS.json')!=lock['contexts_sha256'] or context not in read(dataset/'CONTEXTS.json'):
        raise ValueError('Unplanned TEST context')
    verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
    completed=read(models/'TRAINING_COMPLETE.json')
    if not completed['completed']:raise ValueError('Training incomplete')
    for part,key in [('belief','belief_manifest_sha256'),('feasibility','feasibility_manifest_sha256')]:
        manifest=models/part/('BELIEF_MANIFEST.json' if part=='belief' else 'FEASIBILITY_MANIFEST.json')
        if sha(manifest)!=completed[key] or sha(manifest)!=spec[key]:raise ValueError('Deployment checkpoint manifest changed')
        selection=read(models/part/'CHECKPOINT_SELECTION_LOCK.json')
        if len(selection['checkpoints'])!=3:raise ValueError('Missing pre-TEST checkpoint-selection lock')
    return spec,context,models


def locked_query(env,out):
    spec,context,models=context_and_models()
    if env.af_contact_friction!=context['friction'] or env._af_qualification_policy_seed!=context['policy_seed']:
        raise ValueError('Native TEST setup differs from plan')
    if os.environ.get('AF_P4_PHYSICAL_SURFACE_CAMERA')!='1' or os.environ.get('AF_ORIGINAL_SQUEEZE_INNER')!='1':
        raise ValueError('Formal sensor and complete original squeeze loop required')
    original_query(env,out)
    import sapien
    body=env.deskbin.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    write(out/'ACTUAL_OBJECT_MATERIAL.json',{'shape_materials':[
        [float(s.physical_material.static_friction),float(s.physical_material.dynamic_friction)] for s in body.collision_shapes]})
    write(out/'ONLINE_QUERY_ADMISSION.json',admit(out,require_material=True))
    initial=base.readback(env);initial_count=env.take_action_cnt
    from scripts.eval_policy_xpolicylab import (build_policy_client,close_policy_client,
        normalize_action_chunk,robotwin_obs_to_xpolicylab,xpolicylab_action_to_robotwin)
    client=build_policy_client({'protocol':'ws','host':'localhost','port':6001,
                               'evaluation_id':'original-af-preaction-lock','trial_id':context['id']})
    try:
        client.call(func_name='prepare_case',obs={'task_name':'dump_bin_bigbin','seed':200002,
            'policy_seed':context['policy_seed'],'instruction':env.get_instruction(),'action_type':'joint'})
        client.call(func_name='reset')
        observation=env.get_obs()
        payload=robotwin_obs_to_xpolicylab(observation,instruction=env.get_instruction(),env_idx=0,frequency=30,task_env=env)
        client.call(func_name='update_obs',obs=payload)
        response=normalize_action_chunk(client.call(func_name='get_action'));actions=[]
        for action in response:
            flat,kind=xpolicylab_action_to_robotwin(action,action_type='joint',current_observation=observation)
            if kind!='qpos':raise ValueError('Unexpected policy action type')
            actions.append(flat)
        actions=np.asarray(actions,np.float32)
    finally:close_policy_client(client)
    raw=read(out/'original_raw_rows.json');patch=read(out/'patch_readbacks.json')
    kin=NativeCartesianKinematics(env,env._af_grasp_arm_tag)
    feature=build_features(raw,patch[-1]['finger_joints'],np.asarray(body.linear_velocity),kin.future_xyz_base(actions))
    torch.set_num_threads(1)
    belief=load_native_belief(models/'belief/BELIEF_MANIFEST.json',manifest_sha256=spec['belief_manifest_sha256'])
    posterior=belief.rows(raw,patch,runtime_manifest_sha256=context['runtime_manifest_sha256'])
    posterior.update(candidate_actions_executed=0)
    feasibility=NativeOriginalFeasibility(models/'feasibility/FEASIBILITY_MANIFEST.json',
        manifest_sha256=spec['feasibility_manifest_sha256'],device='cpu')
    decision=feasibility.select(feature,posterior)
    if base.readback(env)!=initial or env.take_action_cnt!=initial_count:
        raise ValueError('Decision construction changed physical state/action count')
    serial={k:v for k,v in posterior.items() if k not in ('continuous_posterior','raw_features','normalized_features')}
    serial={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in serial.items()}
    parent=out.parent
    write(parent/'PREACTION_POSTERIOR.json',serial);write(parent/'PREACTION_FEATURE.json',feature)
    np.save(parent/'PREACTION_PI0_CHUNK.npy',actions)
    write(parent/'PREACTION_AF_DECISION.json',decision)
    forces=[float(decision['selected_force_N']),1.,3.,5.,6.,8.]
    seal={'context_id':context['id'],'root':200002,'candidate_actions_executed':0,
          'labels_read':False,'state_sha256':base.digest(initial),
          'first_chunk_sha256':hashlib.sha256(actions.tobytes()).hexdigest(),
          'observation_identity':observation_identity(observation),'forces_N':forces,
          'branch_methods':['ActiveForcing','Fixed-1N','Fixed-3N','Fixed-5N','Fixed-6N','Fixed-8N'],
          'model_manifests':{'belief':spec['belief_manifest_sha256'],'feasibility':spec['feasibility_manifest_sha256']},
          'artifact_hashes':{name:sha(parent/name) for name in ['PREACTION_POSTERIOR.json','PREACTION_FEATURE.json',
                                                             'PREACTION_PI0_CHUNK.npy','PREACTION_AF_DECISION.json']},
          'context':context,'inference_source_sha256':sha(__file__)}
    write(parent/'PREACTION_SELECTION_LOCK.json',seal)
    env._af_preacton_selection_lock=seal
    os.environ['AF_ENGINEERING_FORCES']=','.join(str(f) for f in forces)


def qualify(env,out):
    module=bound_execution_module();module.qualify_query=locked_query
    module.qualify(env,out)
    spec,context,models=context_and_models()
    seal=read(out/'PREACTION_SELECTION_LOCK.json');summary=read(out/'online_qualification.json')
    if len(summary['results'])!=6 or any(r['kind']!='done' for r in summary['results']):raise ValueError('Incomplete paired online TEST')
    if summary['first_chunk_hashes']!=[seal['first_chunk_sha256']]*6:
        raise ValueError('Execution first chunk differs from pre-action selection chunk')
    if summary['common_handoff_state_sha256']!=seal['state_sha256']:raise ValueError('Handoff changed after selection')
    for name,digest in seal['artifact_hashes'].items():
        if sha(out/name)!=digest:raise ValueError('Pre-action lock artifacts changed')
    feature=read(out/'PREACTION_FEATURE.json');rows=[]
    for index,(method,force,record) in enumerate(zip(seal['branch_methods'],seal['forces_N'],summary['results'])):
        branch=out/f'branch_{index}_{force:g}N';value=record['result']
        if read(branch/'original_motion_feature.json')!=feature:raise ValueError('Candidate feature differs from sealed feature')
        if value['force_setpoint_bilateral_n']!=force or value['success']!=value['official_final_check']:
            raise ValueError('Execution force or official success mismatch')
        y=int(value['success']);utility=y*(5.-force)/5.+(1-y)*-1.
        rows.append({'method':method,'force_N':force,'success':y,'original_utility':utility,
                     'actual_mean_force_N':value['measured_mean_squeeze_n'],
                     'native_actions':value['native_actions'],'result_sha256':sha(branch/'result.json')})
    write(out/'AF_INFERENCE_RESULT.json',{'completed':True,'formal_AF_inference':True,'context':context,
        'selection_lock_sha256':sha(out/'PREACTION_SELECTION_LOCK.json'),'outcomes':rows,
        'paired_rollouts':6,'independent_query_settings':1,'harness_binding':module.inference_binding_receipt,
        'scope':'same-root friction-conditioned task experiment; no cross-root or damage-safety claim'})


if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    base.qualify=qualify
    base.main()
