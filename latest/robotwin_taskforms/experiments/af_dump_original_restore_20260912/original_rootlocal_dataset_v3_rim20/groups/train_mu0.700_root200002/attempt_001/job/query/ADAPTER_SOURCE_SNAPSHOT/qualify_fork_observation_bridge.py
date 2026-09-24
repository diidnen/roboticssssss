"""Real native RGB/proprio IPC with CPU-only action workers; no AF task score."""
from copy import deepcopy
import gzip
import json
import multiprocessing as mp
import os
import signal
import time
import numpy as np
import qualify_native_interfaces as base
from qualify_fork_native_actions import first_chunk,run_actions
from native_visual_mirror import capture_visual_packet,native_observation_from_packet,observation_identity


def qualify(env,out):
    out.mkdir(parents=True,exist_ok=False)
    if env.crazy_random_light:raise RuntimeError('Random-light RNG routing not yet implemented')
    arm=env._af_grasp_arm_tag
    actions=first_chunk(env,out)
    initial=base.readback(env)
    report={'scope':'CPU physics/native parent-render IPC engineering only',
            'formal_collection_gate_passed':False,'children':[],'mirror_requests':[]}
    # Reference child runs first; parent does not mirror or step before creating
    # the RPC child. Thus both inherit exactly the same full native state.
    for mode in ('reference','rpc'):
        parent,child=mp.Pipe(duplex=True)
        pid=os.fork()
        if pid==0:
            parent.close()
            try:
                env.eval_video_path=None;env.render_freq=0
                env._update_render=lambda:None
                observation_records=[]
                def get_remote_obs():
                    packet=capture_visual_packet(env)
                    child.send({'kind':'observe','packet':packet})
                    reply=child.recv()
                    if reply['kind']!='observation':raise RuntimeError(reply)
                    env.now_obs=deepcopy(reply['observation'])
                    observation_records.append(observation_identity(env.now_obs))
                    return env.now_obs
                if mode=='rpc':env.get_obs=get_remote_obs
                else:
                    def unsupported_reference_obs(*a,**kw):
                        raise RuntimeError('Reference child observation boundary needs parent routing')
                    env.get_obs=unsupported_reference_obs
                rows=[]
                old_step=env.scene.step
                def traced_step():
                    old_step()
                    rows.append({'state':base.readback(env),'contact':base.contacts(env,arm)})
                env.scene.step=traced_step
                if mode=='rpc':get_remote_obs()
                for action in actions:
                    env.take_action(action.copy(),action_type='qpos')
                    if mode=='rpc':get_remote_obs()
                with gzip.open(out/(mode+'_trace.json.gz'),'wt') as stream:json.dump(rows,stream)
                child.send({'kind':'done','result':{'mode':mode,'physics_steps':len(rows),
                    'trace_sha256':base.digest(rows),'observations':observation_records}})
                child.close();os._exit(0)
            except BaseException as exc:
                try:child.send({'kind':'error','error':repr(exc)})
                finally:
                    child.close();os._exit(2)
        child.close()
        deadline=time.monotonic()+120
        result=None
        while time.monotonic()<deadline:
            if parent.poll(.2):
                try:message=parent.recv()
                except EOFError:break
                if message['kind']=='observe':
                    packet=message['packet']
                    try:observation=native_observation_from_packet(env,packet)
                    except BaseException as exc:
                        result={'kind':'error','error':'Parent mirror: '+repr(exc)}
                        parent.send(result)
                        break
                    # This is a real observation, never zero-filled or replayed
                    # from another physical state.
                    report['mirror_requests'].append({'action_count':packet['take_action_cnt'],
                        'state_sha256':base.digest(packet['state']),'identity':observation_identity(observation)})
                    parent.send({'kind':'observation','observation':observation})
                else:
                    result=message;break
            waited,status=os.waitpid(pid,os.WNOHANG)
            if waited:
                pid=None;break
        if pid is not None:
            if result is None:
                os.kill(pid,signal.SIGKILL)
            _,status=os.waitpid(pid,0)
        parent.close()
        if result is None:result={'kind':'error','error':'worker timeout/exit without result'}
        report['children'].append(result)
        if mode=='reference' and base.readback(env)!=initial:
            raise RuntimeError('Reference child changed parent before RPC fork')
        (out/'qualification.json').write_text(json.dumps(report,indent=2))
        print('FORK_OBSERVATION_CHILD '+json.dumps({k:v for k,v in result.items() if k!='result'}),flush=True)
        if result['kind']!='done':break
    report['exact_physics_with_real_observation_ipc']=(len(report['children'])==2 and
        all(r['kind']=='done' for r in report['children']) and
        report['children'][0]['result']['trace_sha256']==report['children'][1]['result']['trace_sha256'])
    (out/'qualification.json').write_text(json.dumps(report,indent=2))
    print('FORK_OBSERVATION_QUALIFICATION '+json.dumps({
        'exact_physics_with_real_observation_ipc':report['exact_physics_with_real_observation_ipc'],
        'real_mirror_requests':len(report['mirror_requests'])}),flush=True)


if __name__=='__main__':
    base.qualify=qualify
    base.main()
