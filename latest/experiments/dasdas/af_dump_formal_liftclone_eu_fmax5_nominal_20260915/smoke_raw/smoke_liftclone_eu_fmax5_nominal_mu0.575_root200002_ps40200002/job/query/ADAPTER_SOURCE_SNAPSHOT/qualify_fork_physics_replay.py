"""Bounded CPU-physics-only fork test; never use inherited GPU contexts.

Question: does an OS copy of the full native process preserve the contact/solver
history that public snapshots exclude? Children do only CPU scene.step and
native readback, then os._exit (no inherited renderer/CUDA teardown). This is
NOT a production rollout worker or permission to render/infer after fork.
"""
import gzip
import json
import os
import signal
import time
import numpy as np
import qualify_native_interfaces as base
from native_original_evidence_adapter import measured_patch_record


def record_physics(env,arm,count):
    joints=getattr(env.robot,arm+'_gripper')
    rows=[]
    for step in range(1,count+1):
        velocities=[np.asarray(j.child_link.linear_velocity).copy() for j,_,_ in joints]
        env.scene.step()
        row=base.contacts(env,arm)
        row['state']=base.readback(env)
        row['aggregate_force']=measured_patch_record(row,velocities,env.physics_timestep,step)
        rows.append(row)
    return rows


def qualify(env,out):
    out.mkdir(parents=True,exist_ok=False)
    arm=env._af_grasp_arm_tag
    joints=getattr(env.robot,arm+'_gripper')
    scale=getattr(env.robot,arm+'_gripper_scale')
    servo,_=base.original_servo()
    command=float(joints[0][0].drive_target[0])
    env.robot.set_gripper_force_limit(2.,arm)
    for step in range(500):
        if step%5==0:
            command=servo(command,base.contacts(env,arm)['measured_squeeze_n'],4.)
            env.robot.set_gripper((command-scale[0])/(scale[1]-scale[0]),arm,gripper_eps=0)
        env.scene.step()
    initial=base.readback(env)
    report={'scope':'CPU-only full-memory fork diagnostic, no CUDA/render/policy in children',
            'formal_collection_gate_passed':False,'initial_state_sha256':base.digest(initial),
            'steps_per_branch':200,'children':[]}
    for index in range(3):
        destination=out/f'fork_{index}.json.gz'
        pid=os.fork()
        if pid==0:
            try:
                if base.readback(env)!=initial:raise RuntimeError('Fork did not preserve exposed state')
                rows=record_physics(env,arm,200)
                with gzip.open(destination,'wt') as stream:json.dump(rows,stream)
                os._exit(0)
            except BaseException as exc:
                (out/f'fork_{index}_error.json').write_text(json.dumps({'error':repr(exc)}))
                os._exit(2)
        deadline=time.monotonic()+30
        status=None
        while time.monotonic()<deadline:
            waited,code=os.waitpid(pid,os.WNOHANG)
            if waited:
                status=code;break
            time.sleep(.05)
        if status is None:
            # Only the exact child created above is stopped on timeout.
            os.kill(pid,signal.SIGKILL)
            os.waitpid(pid,0)
            report['children'].append({'index':index,'timed_out':True})
            break
        result={'index':index,'exit_status':status,'parent_state_unchanged':base.readback(env)==initial}
        if status==0:
            with gzip.open(destination,'rt') as stream:result['trace_sha256']=base.digest(json.load(stream))
        report['children'].append(result)
        print('CPU_FORK_CHILD '+json.dumps(result),flush=True)
        if status!=0 or not result['parent_state_unchanged']:break
    if len(report['children'])==3 and all(r.get('exit_status')==0 for r in report['children']):
        baseline=record_physics(env,arm,200)
        with gzip.open(out/'parent_uninterrupted.json.gz','wt') as stream:json.dump(baseline,stream)
        report['uninterrupted_trace_sha256']=base.digest(baseline)
        report['all_children_match_uninterrupted']=all(
            r['trace_sha256']==report['uninterrupted_trace_sha256'] for r in report['children'])
    else:
        report['all_children_match_uninterrupted']=False
    (out/'qualification.json').write_text(json.dumps(report,indent=2))
    print('CPU_FORK_QUALIFICATION '+json.dumps(report),flush=True)


if __name__=='__main__':
    base.qualify=qualify
    base.main()
