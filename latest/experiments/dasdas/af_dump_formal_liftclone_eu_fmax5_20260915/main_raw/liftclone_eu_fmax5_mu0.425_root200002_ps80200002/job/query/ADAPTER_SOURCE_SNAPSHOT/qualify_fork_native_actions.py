"""Real frozen-pi0/native TOPP action replay, not AF training or inference score."""
import gzip
import hashlib
import json
import os
import signal
import time
import numpy as np
import qualify_native_interfaces as base


def first_chunk(env,out):
    from scripts.eval_policy_xpolicylab import (build_policy_client,close_policy_client,
        normalize_action_chunk,robotwin_obs_to_xpolicylab,xpolicylab_action_to_robotwin)
    observation=env.get_obs()
    instruction=env.get_instruction()
    payload=robotwin_obs_to_xpolicylab(observation,instruction=instruction,
                                     env_idx=0,frequency=30,task_env=env)
    client=build_policy_client({'protocol':'ws','host':'localhost','port':6001,
        'evaluation_id':'af-original-fork-action-engineering','trial_id':'root200002'})
    try:
        client.call(func_name='prepare_case',obs={'task_name':'dump_bin_bigbin','seed':200002,
            'policy_seed':140200002,'instruction':instruction,'action_type':'joint'})
        client.call(func_name='reset')
        client.call(func_name='update_obs',obs=payload)
        chunk=normalize_action_chunk(client.call(func_name='get_action'))
    finally:close_policy_client(client)
    converted=[]
    for action in chunk:
        flat,kind=xpolicylab_action_to_robotwin(action,action_type='joint',current_observation=observation)
        if kind!='qpos':raise ValueError('Unexpected native action type')
        converted.append(flat)
    array=np.asarray(converted,np.float32)
    if array.ndim!=2 or array.shape[0]<8 or array.shape[1]!=14 or not np.isfinite(array).all():
        raise ValueError('Invalid real pi0 action chunk')
    np.save(out/'FIRST_NATIVE_ACTION_CHUNK.npy',array,allow_pickle=False)
    metadata={'instruction':instruction,'policy_seed':140200002,'shape':list(array.shape),
              'sha256':hashlib.sha256(array.tobytes()).hexdigest(),
              'scope':'locked real chunk; task-name fallback retained for engineering only'}
    (out/'FIRST_CHUNK.json').write_text(json.dumps(metadata,indent=2))
    print('REAL_PI0_CHUNK_LOCKED '+json.dumps(metadata),flush=True)
    return array[:8].copy()


def run_actions(env,arm,actions,path,child):
    original_step=env.scene.step
    original_render=env._update_render
    original_get_obs=env.get_obs
    old_video=env.eval_video_path
    old_render_freq=env.render_freq
    rows=[]
    render_requests=0
    def traced_step():
        original_step()
        rows.append({'state':base.readback(env),'contact':base.contacts(env,arm)})
    def no_child_render():
        nonlocal render_requests
        render_requests+=1
    def unsupported_child_observation(*a,**kw):
        raise RuntimeError('Child requested GPU observation: parent IPC still required')
    env.scene.step=traced_step
    if child:
        env._update_render=no_child_render
        env.get_obs=unsupported_child_observation
        env.eval_video_path=None
        env.render_freq=0
    try:
        for action in actions:env.take_action(action.copy(),action_type='qpos')
    finally:
        env.scene.step=original_step
        env._update_render=original_render
        env.get_obs=original_get_obs
        env.eval_video_path=old_video
        env.render_freq=old_render_freq
        with gzip.open(path,'wt') as stream:json.dump(rows,stream)
    return {'trace_sha256':base.digest(rows),'physics_steps':len(rows),
            'suppressed_child_render_calls':render_requests,'native_eval_success':bool(env.eval_success)}


def qualify(env,out):
    out.mkdir(parents=True,exist_ok=False)
    arm=env._af_grasp_arm_tag
    actions=first_chunk(env,out)
    initial=base.readback(env)
    initial_count=env.take_action_cnt
    report={'scope':'unchanged native action replay only; NOT AF success evaluation',
            'formal_collection_gate_passed':False,'actions':8,'children':[]}
    for index in range(2):
        receipt=out/f'child_{index}.json'
        pid=os.fork()
        if pid==0:
            try:
                result=run_actions(env,arm,actions,out/f'child_{index}_trace.json.gz',True)
                receipt.write_text(json.dumps(result,indent=2))
                os._exit(0)
            except BaseException as exc:
                receipt.write_text(json.dumps({'error':repr(exc)},indent=2))
                os._exit(2)
        deadline=time.monotonic()+60
        status=None
        while time.monotonic()<deadline:
            waited,code=os.waitpid(pid,os.WNOHANG)
            if waited:status=code;break
            time.sleep(.05)
        if status is None:
            os.kill(pid,signal.SIGKILL);os.waitpid(pid,0)
            report['children'].append({'timed_out':True});break
        result=json.loads(receipt.read_text()) if receipt.exists() else {}
        result.update(exit_status=status,parent_state_unchanged=base.readback(env)==initial,
                      parent_action_count_unchanged=env.take_action_cnt==initial_count)
        report['children'].append(result)
        print('FORK_NATIVE_ACTION_CHILD '+json.dumps(result),flush=True)
        if status!=0:break
    if len(report['children'])==2 and all(r.get('exit_status')==0 for r in report['children']):
        report['parent_uninterrupted']=run_actions(env,arm,actions,out/'parent_trace.json.gz',False)
        report['all_children_match_uninterrupted']=all(
            r['trace_sha256']==report['parent_uninterrupted']['trace_sha256'] and r['parent_state_unchanged']
            and r['parent_action_count_unchanged'] for r in report['children'])
    else:report['all_children_match_uninterrupted']=False
    (out/'qualification.json').write_text(json.dumps(report,indent=2))
    print('FORK_NATIVE_ACTION_QUALIFICATION '+json.dumps(report),flush=True)


if __name__=='__main__':
    base.qualify=qualify
    base.main()
