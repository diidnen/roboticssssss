"""Reference-only camera/tactile capture; no downstream physics or VLA calls.

Run in a separately frozen diagnostic directory after the active single-worker
queue. Wrap only the readout hook; underlying core, probe, actions and controller
stay unchanged. Observations are copied after the existing simulation step.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from common import read,write,sha,clean
import worker
from summarize_probe_observation import summarize

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--job',required=True)
    parser.add_argument('--context',type=int,required=True);parser.add_argument('--method',default='REFERENCE')
    parser.add_argument('--port',type=int,default=18885);args=parser.parse_args()
    if args.method!='REFERENCE':raise RuntimeError('Probe diagnostic cannot execute downstream branch')
    worker.setup_paths()
    import measurement_hooks
    original=measurement_hooks.patches;counter=0;paths=[];job=Path(args.job)
    def patches(env,plan,dt):
        nonlocal counter
        result=original(env,plan,dt);counter+=1
        # Preserve the fixed contract's established-grasp tail and all physical
        # probe frames; actual phase transition is read afterwards from CSV.
        if counter>=180:
            values={f'rgb__{name}':env.scene[name].data.output['rgb'][0].detach().cpu().numpy().copy()
                    for name in ('agentview_cam','eye_in_hand_cam')}
            obs=getattr(env,'obs_buf',{}).get('policy',{})
            values.update({f'policy__{k}':v.detach().cpu().numpy().copy() for k,v in obs.items() if hasattr(v,'detach')})
            path=job/f'PROBE_OBSERVATION_{counter:04d}.npz'
            with path.open('xb') as f:np.savez_compressed(f,**values)
            paths.append(dict(step=counter,path=str(path),sha256=sha(path)))
        return result
    measurement_hooks.patches=patches
    original_write=worker.write
    def completion_write(path,value):
        if Path(path).name=='WORKER_COMPLETION.json' and value.get('logical_success'):
            # Isaac app.close may terminate the interpreter; summarize before
            # writing completion and before worker.run reaches its finally.
            summarize(job)
        return original_write(path,value)
    worker.write=completion_write
    write(job/'PROBE_DIAGNOSTIC_INSTRUMENTATION.json',dict(role='REFERENCE_ONLY_SENSOR_READOUT_DIAGNOSTIC',
        implementation_sha256=sha(Path(__file__)),extra_physics_steps=0,downstream_actions=0,
        original_measurement_hook=str(original.__module__)+'.'+original.__name__,captures='Copy current camera and cached policy observations after existing steps >=180'))
    rc=worker.run(args)
    if rc:raise RuntimeError('Probe diagnostic failed; retain incomplete output')

if __name__=='__main__':main()
