"""Original-P4 handoff -> paired CPU forks -> real frozen pi0 full task.

Engineering only until the complete acquisition/task binding is admitted.
All children fork before the parent mirrors any state. No CPU child invokes
inherited GPU contexts. No old contact-loss shortcut terminates a rollout.
"""
from dataclasses import asdict
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
import numpy as np
import qualify_native_interfaces as base
from qualify_original_p4_native import qualify as qualify_query
from native_original_force_controller import NativeOriginalForceController
from native_original_motion_features import build_features
from native_visual_mirror import capture_visual_packet,native_observation_from_packet,observation_identity
from native_cartesian_kinematics import NativeCartesianKinematics


def child_run(env,pipe,out,force,handoff,stock_drives,support):
    controller=NativeOriginalForceController(env,env._af_grasp_arm_tag,force,handoff,stock_drives,support)
    env._update_render=lambda:None
    env.eval_video_path=None
    env.render_freq=0
    original_step=env.scene.step
    trace_hash=hashlib.sha256()
    physics_steps=0
    forces=[]
    chunk_count=0
    def rpc(kind):
        pipe.send({'kind':kind,'packet':capture_visual_packet(env)})
        response=pipe.recv()
        if response['kind']=='error':raise RuntimeError(response['error'])
        return response
    def get_obs(*a,**kw):
        observation=rpc('observation')['observation']
        env.now_obs=observation
        return observation
    env.get_obs=get_obs
    controller.install()
    try:
        with gzip.open(out/'physics_trace.jsonl.gz','wt') as trace:
            def step():
                nonlocal physics_steps
                controller.before_physics()
                original_step()
                contact=base.contacts(env,env._af_grasp_arm_tag)
                measured=contact['measured_squeeze_n']
                forces.append(measured)
                physics_steps+=1
                row={'physics_step':physics_steps,'contact':contact,'state':base.readback(env),
                     'original_squeeze_inner':controller.inner.last if controller.inner is not None else None}
                encoded=json.dumps(row,sort_keys=True,separators=(',',':'))
                trace_hash.update(encoded.encode());trace.write(encoded+'\n')
                controller.after_physics(measured)
            env.scene.step=step
            while not env.eval_success and env.take_action_cnt<env.step_lim:
                response=rpc('chunk')
                actions=np.asarray(response['actions'],np.float32)
                if actions.ndim!=2 or actions.shape[1]!=14 or not len(actions):raise RuntimeError('Invalid chunk')
                np.save(out/f'chunk_{chunk_count:03d}.npy',actions)
                chunk_count+=1
                for action in actions:
                    controller.before_action(action)
                    # Exact original native arm values and TOPP executor.
                    env.take_action(action.copy(),action_type='qpos')
                    if env.eval_success or env.take_action_cnt>=env.step_lim:break
        result={'completed':True,'success':bool(env.eval_success),'official_final_check':bool(env.check_success()),
                'native_actions':env.take_action_cnt,'native_horizon':env.step_lim,
                'physics_steps':physics_steps,'trace_sha256':trace_hash.hexdigest(),
                'force_setpoint_bilateral_n':force,'measured_mean_squeeze_n':float(np.mean(forces)),
                'measured_max_squeeze_n':float(np.max(forces)),
                'release_actions':sum(r['vla_release_intent'] for r in controller.action_receipts),
                'controller_feedback_ticks':controller.feedback_ticks,'chunks':chunk_count,
                'arbitration_binding':controller.binding_receipt,
                'squeeze_controller_binding':('ORIGINAL_OUTER_AND_ORIGINAL_INNER_POSITION_FEEDBACK'
                    if controller.inner else 'ORIGINAL_OUTER_ONLY_NATIVE_FORCE_CAP_DIAGNOSTIC'),
                'old_irrecoverable_shortcut_used':False,'scope':'engineering, not formal learned AF inference'}
        (out/'arbitration.json').write_text(json.dumps(controller.action_receipts,indent=2))
        (out/'result.json').write_text(json.dumps(result,indent=2))
        pipe.send({'kind':'done','result':result})
    finally:
        controller.uninstall();env.scene.step=original_step


def qualify(env,out):
    out.mkdir(parents=True,exist_ok=False)
    # Capture native open-position actuator settings before query force caps.
    stock=[(j,j.stiffness,j.damping,j.force_limit,j.drive_mode)
           for j,_,_ in getattr(env.robot,env._af_grasp_arm_tag+'_gripper')]
    qualify_query(env,out/'query')
    report=json.loads((out/'query/qualification.json').read_text())
    if not report.get('completed') or report.get('probe_failure'):
        raise RuntimeError('Original P4 did not establish qualified query contact; no downstream branches')
    raw=json.loads((out/'query/original_raw_rows.json').read_text())
    controls=json.loads((out/'query/native_controls.json').read_text())
    handoff=float(controls[-1]['original_action13'][6])
    joints=getattr(env.robot,env._af_grasp_arm_tag+'_gripper')
    inner=getattr(env,'_af_original_squeeze_inner',None)
    if inner is not None:
        if abs(float(env._af_original_outer_command)-handoff)>1e-8:
            raise RuntimeError('Original outer probe command handoff mismatch')
        if abs(float(joints[0][0].drive_target[0])-inner.last['inner_aperture_m'])>1e-7:
            raise RuntimeError('Original inner command differs from actual drive target')
    elif abs(float(joints[0][0].drive_target[0])-handoff)>1e-7:
        raise RuntimeError('Original probe command handoff differs from native drive target')
    initial=base.readback(env)
    initial_hash=base.digest(initial)
    initial_count=env.take_action_cnt
    kin=NativeCartesianKinematics(env,env._af_grasp_arm_tag)
    workers=[]
    forces=tuple(float(x) for x in os.environ.get('AF_ENGINEERING_FORCES','3,3,5').split(','))
    if len(forces)>3 or len(forces)<1 or not all(3<=x<=8 for x in forces):
        raise ValueError('Bounded engineering test accepts 1..3 forces within [3,8] N')
    support=(3.,max(5.,max(forces)))
    # All full-memory children MUST exist before the first parent state mirror.
    for index,force in enumerate(forces):
        branch=out/f'branch_{index}_{force:g}N';branch.mkdir()
        parent,child=mp.Pipe()
        pid=os.fork()
        if pid==0:
            parent.close()
            for old in workers:old['pipe'].close()
            try:
                command=child.recv()
                if command!='start':raise RuntimeError('Expected start')
                child_run(env,child,branch,force,handoff,stock,support)
                os._exit(0)
            except BaseException as exc:
                error={'completed':False,'error':repr(exc),'traceback':traceback.format_exc()}
                (branch/'error.json').write_text(json.dumps(error,indent=2))
                child.send({'kind':'error',**error});os._exit(2)
        child.close()
        workers.append({'pid':pid,'pipe':parent,'force':force,'path':branch})
        if base.readback(env)!=initial or env.take_action_cnt!=initial_count:
            raise RuntimeError('Fork changed shared source boundary')
    from scripts.eval_policy_xpolicylab import (build_policy_client,close_policy_client,
        normalize_action_chunk,robotwin_obs_to_xpolicylab,xpolicylab_action_to_robotwin)
    policy_seed=int(getattr(env,'_af_qualification_policy_seed',140200002))
    summary={'scope':'full native online engineering with original AF force controller',
             'formal_AF_inference':False,'root':200002,'policy_seed':policy_seed,
             'object_side_friction':float(env.af_contact_friction),
             'instruction':env.get_instruction(),'common_handoff_state_sha256':initial_hash,
             'engineering_force_list_N':list(forces),'force_support_N':list(support),
             'handoff_command_m':handoff,'results':[],'first_chunk_hashes':[]}
    try:
        for index,worker in enumerate(workers):
            client=build_policy_client({'protocol':'ws','host':'localhost','port':6001,
                'evaluation_id':'original-af-online-engineering','trial_id':str(index)})
            receipts=[]
            result=None
            try:
                client.call(func_name='prepare_case',obs={'task_name':'dump_bin_bigbin','seed':200002,
                    'policy_seed':policy_seed,'instruction':summary['instruction'],'action_type':'joint'})
                client.call(func_name='reset')
                worker['pipe'].send('start')
                deadline=time.monotonic()+1800
                chunk_count=0
                while time.monotonic()<deadline:
                    if not worker['pipe'].poll(.2):
                        ended,status=os.waitpid(worker['pid'],os.WNOHANG)
                        if ended:
                            worker['pid']=None
                            raise RuntimeError('Worker exited without terminal receipt: '+str(status))
                        continue
                    message=worker['pipe'].recv()
                    if message['kind'] in ('done','error'):
                        result=message;break
                    packet=message['packet']
                    observation=native_observation_from_packet(env,packet)
                    if message['kind']=='observation':
                        worker['pipe'].send({'kind':'observation','observation':observation});continue
                    if message['kind']!='chunk':raise RuntimeError('Unknown worker request')
                    if chunk_count==0 and base.digest(packet['state'])!=initial_hash:
                        raise RuntimeError('Candidate changed state before first policy chunk')
                    payload=robotwin_obs_to_xpolicylab(observation,instruction=summary['instruction'],
                                                      env_idx=0,frequency=30,task_env=env)
                    client.call(func_name='update_obs',obs=payload)
                    response=normalize_action_chunk(client.call(func_name='get_action'))
                    actions=[]
                    for action in response:
                        flat,kind=xpolicylab_action_to_robotwin(action,action_type='joint',current_observation=observation)
                        if kind!='qpos':raise RuntimeError('Native action type changed')
                        actions.append(flat)
                    actions=np.asarray(actions,np.float32)
                    identity=hashlib.sha256(actions.tobytes()).hexdigest()
                    receipts.append({'chunk':chunk_count,'actions_sha256':identity,'observation':observation_identity(observation)})
                    if chunk_count==0:
                        summary['first_chunk_hashes'].append(identity)
                        if identity!=summary['first_chunk_hashes'][0]:raise RuntimeError('Strict policy RNG/action pairing failed')
                        import sapien
                        body=env.deskbin.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                        patch=json.loads((out/'query/patch_readbacks.json').read_text())[-1]
                        feature=build_features(raw,patch['finger_joints'],np.asarray(body.linear_velocity),kin.future_xyz_base(actions))
                        (worker['path']/'original_motion_feature.json').write_text(json.dumps(feature,indent=2))
                    worker['pipe'].send({'kind':'actions','actions':actions})
                    chunk_count+=1
                    print('ORIGINAL_ONLINE_CHUNK '+json.dumps({'branch':index,'force':worker['force'],
                        'chunk':chunk_count,'native_action_count':packet['take_action_cnt']}),flush=True)
                if result is None:raise TimeoutError('Bounded engineering rollout exceeded 30 minutes')
                summary['results'].append(result)
                print('ORIGINAL_ONLINE_BRANCH '+json.dumps(result),flush=True)
                if result['kind']=='error':break
            except BaseException as exc:
                summary['results'].append({'kind':'error','error':repr(exc),'traceback':traceback.format_exc()})
                raise
            finally:
                close_policy_client(client)
                (worker['path']/'policy_receipts.json').write_text(json.dumps(receipts,indent=2))
                (out/'online_qualification.json').write_text(json.dumps(summary,indent=2))
        summary['duplicate_force_replays']=[]
        for i in range(len(summary['results'])):
            for j in range(i):
                left,right=summary['results'][j],summary['results'][i]
                if forces[i]==forces[j] and left['kind']==right['kind']=='done':
                    summary['duplicate_force_replays'].append({'indices':[j,i],'force_N':forces[i],
                        'full_trace_exact':left['result']['trace_sha256']==right['result']['trace_sha256']})
    finally:
        for worker in workers:
            if worker['pid'] is not None:
                ended,_=os.waitpid(worker['pid'],os.WNOHANG)
                if not ended:
                    os.kill(worker['pid'],signal.SIGTERM);os.waitpid(worker['pid'],0)
            worker['pipe'].close()
        (out/'online_qualification.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    base.qualify=qualify
    base.main()
