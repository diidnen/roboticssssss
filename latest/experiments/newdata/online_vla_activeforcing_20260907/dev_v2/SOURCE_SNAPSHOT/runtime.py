"""Actual online VLA downstream execution, with no waypoint executor import."""
import ast
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path
import numpy as np
from common import HERE, ROOT, BASE, TABERO, INSTRUCTIONS, read, write, sha, clean, array_sha, payload_sha
from arbitration import Arbitration

HORIZON = 350
REPLAN_STEPS = 10
LABEL_VERSION = 'ONLINE_VLA_350STEP_WHOLE_MESH_RELEASE_SUPPORT_V1'

def observation_builder():
    source = TABERO/'analysis/p6g1_primitive_ik_vla_grasp_realization.py'
    names = {'_to_uint8_rgb','_pad_history_front','OnlineTactileBuffer','_add_bytes_key_aliases','build_policy_observation'}
    tree = ast.parse(source.read_text())
    selected = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if len(selected)!=len(names): raise RuntimeError('Historical observation adapter changed')
    ns = {'np':np,'deque':deque}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
    return ns['build_policy_observation'],ns['OnlineTactileBuffer'],sha(source)

class OnlineClient:
    def __init__(self, job, plan, port):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy('127.0.0.1',port)
        self.metadata = self.client.get_server_metadata()
        if self.metadata.get('downstream_action_source')!='ONLINE_VLA' or not self.metadata.get('checkpoint_loaded'):
            raise RuntimeError('Non-provenance VLA server forbidden')
        expected='0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17'
        if self.metadata['checkpoint_sha256']!=expected: raise RuntimeError('Wrong VLA checkpoint')
        self.job=Path(job);self.plan=plan;self.worker=self.job.name;self.calls=0
        (self.job/'RPC').mkdir(exist_ok=False)
        write(self.job/'VLA_SERVER_METADATA.json',self.metadata)

    def infer(self, payload, step, raw_cameras):
        self.calls+=1
        if step!=1+(self.calls-1)*REPLAN_STEPS: raise RuntimeError('Native replan schedule violation')
        rid=f'{self.worker}:{step:04d}'
        digest=payload_sha(payload)
        seed=int(hashlib.sha256(f"v1:{self.plan['id']}:{step}".encode()).hexdigest()[:8],16)
        audit={'request_id':rid,'worker_id':self.worker,'context_id':self.plan['id'],
            'step':step,'observation_sha256':digest,'noise_seed':seed,
            'instruction':INSTRUCTIONS[self.plan['task']]}
        start=time.time_ns()
        response=self.client.infer({**payload,'_audit':audit})
        received=time.time_ns();proof=response['provenance']
        for key in audit:
            if proof[key]!=audit[key]:raise RuntimeError('RPC routing or observation mismatch: '+key)
        if not proof['model_inference_called'] or proof['downstream_action_source']!='ONLINE_VLA':
            raise RuntimeError('Online inference absent')
        if not start<=proof['server_start_ns']<=proof['server_end_ns']<=received:
            raise RuntimeError('Response not generated during current synchronous request')
        if proof['checkpoint_sha256']!=self.metadata['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
        action=np.asarray(response['actions']);raw=np.asarray(response['raw_model_actions'])
        if action.shape!=(50,13) or raw.shape!=(50,32):raise RuntimeError(f'Unexpected action shape {action.shape}/{raw.shape}')
        if array_sha(action)!=proof['action_sha256'] or array_sha(raw)!=proof['raw_model_action_sha256']:
            raise RuntimeError('Action RPC digest mismatch')
        if not np.isfinite(action).all() or not np.isfinite(raw).all():raise RuntimeError('Nonfinite model action')
        dest=self.job/'RPC'/f'{step:04d}'
        # No pickled data. Raw full cameras and canonical preprocessing payload.
        arrays={f'payload__{k}':v for k,v in payload.items() if isinstance(v,np.ndarray)}
        arrays.update(raw_vla_action=raw,postprocessed_vla_action=action,**raw_cameras)
        with dest.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        write(dest.with_suffix('.json'),{**proof,'client_start_ns':start,'client_received_ns':received,
            'latency_ms':(received-start)/1e6,'artifact_sha256':sha(dest.with_suffix('.npz')),
            'executed_chunk_indices':list(range(REPLAN_STEPS)), 'discarded_chunk_indices':list(range(REPLAN_STEPS,50))})
        return action.astype(np.float32),proof

def observation(env,p4,p5,plan,geom):
    import torch
    obj=env.scene[plan['object']];target=env.scene[plan['target']];robot=env.scene['robot']
    obs=env.observation_manager.compute();eef=np.asarray(clean(obs['policy']['eef_pose'][0]),float)
    robotpose=np.asarray(clean(robot.data.root_state_w[0,:7]),float)
    worldgripper=geom.transform(robotpose)@geom.transform(eef)
    objectpose=np.asarray(clean(obj.data.root_state_w[0,:7]),float)
    relative=geom.inverse(worldgripper)@geom.transform(objectpose)
    contact=p5.target_object_force_snapshot(env,p4,plan['object']);dbg=p4._dbg(env)
    matrix=env.scene['contact_'+plan['target']+'_'+plan['object']].data.force_matrix_w
    normals=[contact['F_obj_left_normal_n'],contact['F_obj_right_normal_n']]
    return {'episode_step':int(env.episode_length_buf[0]),'object_pose_w':objectpose.tolist(),
        'object_position_m':objectpose[:3].tolist(),'object_velocity_w':clean(obj.data.root_state_w[0,7:13]),
        'basket_pose_w':clean(target.data.root_state_w[0,:7]),
        'basket_contact_force_N':float(torch.linalg.vector_norm(matrix.reshape(-1,3)[0])),
        'finger_object_contact_norms_N':[float(np.linalg.norm(contact[k])) for k in ('F_obj_left_world','F_obj_right_world')],
        'normal_force_N':normals,'measured_bilateral_squeeze':2*min(normals),
        'gripper_eef_pose_base':eef.tolist(),'gripper_transform_w':worldgripper.tolist(),
        'object_relative_translation_m':relative[:3,3].tolist(),
        'controller_internal_force_reference':p4._f(dbg.get('f_sq_pred_eff'),0.),
        'controller_final_gripper_command':p4._f(dbg.get('d_cmd'),0.),
        'controller_debug':clean(dbg),
        'terminations':{k:bool(env.termination_manager.get_term(k)[0]) for k in env.termination_manager.active_terms}}

def evaluate(trace,vertices,region,geom,z0,*,error=None):
    if not trace: return {'label_valid':False,'reason':'NO_ACTION_EVIDENCE','full_task_success_y':None}
    dropped=any(t['terminations'].get('object_1_dropped',False) for t in trace)
    timeout=any(t['terminations'].get('time_out',False) for t in trace)
    reset=any(t['physical_reset_requested'] for t in trace)
    complete=len(trace)==HORIZON
    if error or (not complete and not reset):
        return {'label_valid':False,'reason':'EXECUTION_ERROR_OR_UNEXPLAINED_SHORT_HORIZON','full_task_success_y':None,'error':error}
    checks=[geom.containment(region,geom.transform(t['basket_pose_w']),geom.transform(t['object_pose_w']),vertices) for t in trace[-50:]]
    inside=len(checks)==50 and all(x['whole_mesh_inside_authored_obb'] for x in checks)
    opened=len(trace)>=20 and all(t['vla_release_intent'] and t['final_gripper_command']==float(np.float32(.04)) for t in trace[-20:])
    unheld=len(trace)>=20 and all(max(t['finger_object_contact_norms_N'])<=.05 for t in trace[-20:])
    support=trace[-1]['basket_contact_force_N']>.05
    lift=any(t['object_position_m'][2]-z0>=.03 for t in trace)
    flags={'NO_LIFT':not lift,'DROP':dropped,'TIMEOUT':timeout,'RESET':reset,'INCOMPLETE_HORIZON':not complete,
        'OUTSIDE_AUTHORED_REGION':not inside,'NOT_RELEASED':not(opened and unheld),'NO_FINAL_SUPPORT_CONTACT':not support}
    return {'label_valid':True,'label_version':LABEL_VERSION,'full_task_success_y':int(not any(flags.values())),
        'lift_success':int(lift),'place_success':int(inside and opened and unheld and support),
        'dropped':int(dropped),'failure_reasons':[k for k,v in flags.items() if v],
        'inside_last50':inside,'opened_last20':opened,'unheld_last20':unheld,'final_support_contact':support,
        'mean_measured_bilateral_squeeze':float(np.mean([t['measured_bilateral_squeeze'] for t in trace if not t['vla_release_intent']])) if any(not t['vla_release_intent'] for t in trace) else None,
        'minimum_containment_margin_m':min(x['minimum_margin_m'] for x in checks)}

def first_chunk(env,plan,job,port):
    client=OnlineClient(job,plan,port)
    build,Buffer,obs_sha=observation_builder();buffer=Buffer()
    write(job/'VLA_OBSERVATION_ADAPTER.json',{'source_sha256':obs_sha,'extraction':'exact five historical definitions; remove duplicate byte aliases only',
        'history':'original chunk-boundary sampling and front padding, initialized at post-probe decision',
        'replan_steps':REPLAN_STEPS,'no_zeros_for_missing_marker_motion':True})
    chunk,proof=request_chunk(env,plan,client,build,buffer,1)
    return client,build,buffer,chunk,proof

def request_chunk(env,plan,client,build,buffer,step):
    obs=env.observation_manager.compute()
    cameras={name:env.scene[name].data.output['rgb'][0].detach().cpu().numpy().copy()
             for name in ('agentview_cam','eye_in_hand_cam')}
    payload=build(env,obs,INSTRUCTIONS[plan['task']],buffer)
    payload={k:v for k,v in payload.items() if isinstance(k,str)}
    if 'tactile_marker_motion' not in payload:raise RuntimeError('Missing real tactile field; zero fallback forbidden')
    return client.infer(payload,step,cameras)

def rollout(env,p4,p5,plan,job,force,handoff,prepared):
    import torch
    import full_task_label as legacy_label
    geom=legacy_label.geom
    taskgeo=next(x for x in read(BASE/'current_runtime_recovery_v2_20260905/TASK_GEOMETRY_MANIFEST.json')['tasks'] if x['task']==plan['task'])
    if sha(taskgeo['label_geometry'])!=taskgeo['label_sha256']:raise RuntimeError('Geometry changed')
    with np.load(taskgeo['label_geometry']) as f: vertices=f['vertices'];region=geom.Region(f['basket_from_site'],f['half_size'])
    client,build,buffer,chunk,proof=prepared
    arb=Arbitration(force,handoff);traces=[];terminals=[];error=None
    original_compute=env.termination_manager.compute;original_reset=env._reset_idx
    z0=float(env.scene[plan['object']].data.root_pos_w[0,2])
    def compute():
        original_compute()
        failures=torch.zeros_like(env.termination_manager.terminated)
        for name in env.termination_manager.active_terms:
            if name!='success' and not env.termination_manager.get_term_cfg(name).time_out:
                failures.logical_or_(env.termination_manager.get_term(name))
        env.termination_manager._terminated_buf.copy_(failures)
        return env.termination_manager.dones
    def reset(ids):
        if len(ids):terminals.append(observation(env,p4,p5,plan,geom))
    env.termination_manager.compute=compute;env._reset_idx=reset
    try:
        for step in range(1,HORIZON+1):
            if step>1 and (step-1)%REPLAN_STEPS==0:chunk,proof=request_chunk(env,plan,client,build,buffer,step)
            index=(step-1)%REPLAN_STEPS;raw=chunk[index]
            observed=env.observation_manager.compute()['policy']
            raw_observation={k:v.detach().cpu().numpy() for k,v in observed.items() if hasattr(v,'detach')}
            raw_observation.update({name:env.scene[name].data.output['rgb'][0].detach().cpu().numpy()
                for name in ('agentview_cam','eye_in_hand_cam')})
            observation_path=job/f'RAW_OBSERVATION_{step:04d}.npz'
            with observation_path.open('xb') as f:np.savez_compressed(f,**raw_observation)
            action,arbitration=arb.action(raw)
            before=int(env.episode_length_buf[0]); env.step(torch.from_numpy(action).reshape(1,13).to(env.device))
            rec=terminals[-1] if terminals else observation(env,p4,p5,plan,geom)
            if not terminals and rec['episode_step']!=before+1:raise RuntimeError('Unexpected episode clock/reset')
            row={'branch_step':step,'request_id':proof['request_id'],'chunk_index':index,'action':action.tolist(),
                 'raw_observation_path':str(observation_path),'raw_observation_sha256':sha(observation_path),
                 'physical_reset_requested':bool(terminals),**arbitration,**rec}
            traces.append(row)
            with (job/'ACTION_TRACE.jsonl').open('a') as f:f.write(json.dumps(clean(row),allow_nan=False)+'\n');f.flush()
            arb.feedback(p4._f(rec['controller_debug'].get('f_sq_meas'),0.),arbitration['vla_release_intent'],p4._force_servo)
            if terminals:break
    except Exception as exc:
        import traceback
        error={'error':repr(exc),'traceback':traceback.format_exc()}
    finally:
        env.termination_manager.compute=original_compute;env._reset_idx=original_reset
    write(job/'BRANCH_TRACE.json',traces)
    outcome=evaluate(traces,vertices,region,geom,z0,error=error)
    result={'plan':plan,'force':force,'steps':len(traces),'rpc_count':client.calls,'outcome':outcome,'error':error,
        'downstream_action_source':'ONLINE_VLA' if traces else 'UNVERIFIED',
        'online_vla_verified':bool(traces) and error is None and client.calls==(len(traces)+REPLAN_STEPS-1)//REPLAN_STEPS,
        'checkpoint_sha256':client.metadata['checkpoint_sha256'],'runtime_label_version':LABEL_VERSION}
    write(job/'BRANCH_RESULT.json',result)
    if error:raise RuntimeError(error['error'])
    return result
