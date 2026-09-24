"""User-authorized three-context probe-only diagnosis, no task branches.

The P4 source and parameters are immutable. Instrumentation reads returned
observations and simulator state; never recomputes policy observations,
changes actions, or takes additional simulation steps.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT=Path('/home/exouser/FORTE')
TAB=Path('/home/exouser/Tabero')
OUT=ROOT/'analysis/results/probe_only_instrumented_20260905'
sys.path.insert(0,str(ROOT))
import activeforcing_e2e_task0_smoke_20260905 as smoke

EXPECTED={smoke.P4_PATH:'a1566334f9f79ad9d8491612f10d086386049295314f6d1fd62512be1867bca9',
          smoke.P5_PATH:'7396b899429be3e3c298217dfab29c01afee03594e6ed8b909b4df04554353a2'}
PHASE_PARAMETERS={'APPROACH_STEPS':45,'DESCEND_STEPS':35,'CLOSE_STEPS':70,'HOLD_STEPS':40,
 'POST_HOLD_STEPS':5,'RETURN_STEPS':10,'P4B_BASE_FORCE_N':3.,'P4B_PRELOAD_STEP_N':.5,'P4B_PRELOAD_CAP_N':4.5,
 'ADAPTIVE_STEP_M':.0002,'MAX_DISP_M':.002,'RHO_IMPULSE_TARGET':.012,'RHO_CAP':.08,
 'RELATIVE_NORMAL_ALPHA':.55,'MAX_OUT_STEPS':25,'SERVO_STEP':.0006,'SERVO_DEADBAND':.4}

def jsonable(x):
    if hasattr(x,'detach'):return x.detach().cpu().numpy().tolist()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    return str(x)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(value),indent=2)+'\n')

def check_sources():
    for p,h in EXPECTED.items():
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError(f'Frozen source changed: {p}')

def material(view):
    return jsonable(view.get_material_properties())

def worker(band):
    import torch
    check_sources()
    job=OUT/'jobs'/band
    job.mkdir(parents=True,exist_ok=True)
    if (job/'PROBE_STARTED.json').exists():raise RuntimeError('Refusing a second physical probe in the same job')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,num_envs=1).app
    env=None;records=[];errors=[]
    try:
        import gymnasium as gym
        import tac_manip.tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        p5=smoke.load_module('instrumented_p5_readonly',smoke.P5_PATH);p5.OUT=job
        plans=smoke.contexts();plan=next(p for p in plans if str(p['friction_band']).lower()==band)
        cid=plan['context_id'];p4=p5.import_p4_probe(0)
        for k,v in PHASE_PARAMETERS.items():
            if not np.isclose(float(getattr(p4,k)),v,rtol=0,atol=1e-12):raise RuntimeError(f'P4 parameter drift: {k}')
        setup_task_objects(p5.TASK_SUITE,0)
        cfg=parse_env_cfg(smoke.ENV_ID,device='cuda:0',num_envs=1);cfg.episode_length_s=45.
        env=gym.make(smoke.ENV_ID,cfg=cfg).unwrapped
        env.reset(seed=int(plan['root_seed']))
        objname=p4.OBJ_NAME;obj=env.scene[objname];robot=env.scene['robot']
        write(job/'BEFORE_PROBE.json',{'context':plan,'object_material':material(obj.root_physx_view),
            'robot_material':material(robot.root_physx_view),'object_masses':jsonable(obj.root_physx_view.get_masses()),
            'action_term_config':str(cfg.actions),'scene_sensor_names':list(env.scene.sensors),
            'instrumentation':'read returned observations only; no observation_manager.compute; no additional env.step'})
        original_step=env.step
        def observed_step(action):
            result=original_step(action)
            step=len(records)+1
            item={'step':step,'action':jsonable(action),'policy_local_force':jsonable(result[0]['policy']['gripper_net_force']),
                  'eef_pose':jsonable(result[0]['policy']['eef_pose']), 'gripper_pos':jsonable(result[0]['policy']['gripper_pos'])}
            # Optional sensors are never allowed to alter or abort the fixed probe.
            try:
                item['target_object_force']=p5.target_object_force_snapshot(env,p4,objname)
                item['all_contact_world']=jsonable(env.scene['contact_gripper'].data.net_forces_w)
                item['left_frame_quat_w']=jsonable(env.scene['left_gripper_frame'].data.target_quat_w)
                item['right_frame_quat_w']=jsonable(env.scene['right_gripper_frame'].data.target_quat_w)
                item['object_position_w']=jsonable(obj.data.root_pos_w)
                item['object_quaternion_w']=jsonable(obj.data.root_quat_w)
                item['object_velocity_w']=jsonable(obj.data.root_lin_vel_w)
                item['finger_joint_positions']=jsonable(robot.data.joint_pos[0,-2:])
                item['action_debug']=jsonable(p4._dbg(env))
                if step in [1,80,150,190,191,200,206,215]:
                    item['object_material']=material(obj.root_physx_view)
                    item['robot_material']=material(robot.root_physx_view)
                    item['object_masses']=jsonable(obj.root_physx_view.get_masses())
            except Exception as exc:
                item['readback_error']=repr(exc);errors.append({'step':step,'error':repr(exc)})
            records.append(item)
            return result
        env.step=observed_step
        write(job/'PROBE_STARTED.json',{'context_id':cid,'unix_time':time.time(),'one_physical_probe':True})
        rows,rec=p4.run_probe_episode(env,seed_idx=int(plan['root_seed']),mu=float(plan['hidden_friction_analysis_only']),trial_id=cid,dt=smoke.DT)
        env.step=original_step
        raw=[vars(r) if hasattr(r,'__dict__') else dict(r) for r in rows]
        smoke.write_csv(job/'RAW_PROBE.csv',raw)
        write(job/'CONTACT_READBACK.json',records)
        quality=p5.derive_p4b_probe_quality(p4,rows,rec)
        state=smoke.clone_cpu(env.scene.get_state(is_relative=True))
        torch.save({'state':state,'context_id':cid,'task':0},job/'POST_PROBE_STATE.pt')
        write(job/'RESULT.json',{'status':'PROBE_ONLY_COMPLETE','context':plan,'probe_record':rec,'probe_quality':quality,
            'raw_rows':len(raw),'observed_steps':len(records),'instrumentation_errors':errors,
            'post_object_material':material(obj.root_physx_view),'post_robot_material':material(robot.root_physx_view),
            'post_object_masses':jsonable(obj.root_physx_view.get_masses()),'task_branches':0,'training':False,
            'controller_changed':False,'probe_changed':False,'utility_changed':False,
            'probe_out_steps':sum(r['probe_phase']=='probe_out' for r in raw)})
        check_sources()
        print(f'PROBE_ONLY_COMPLETE {band} rows={len(raw)} stop={rec.get("stop_trigger")}',flush=True)
        return 0
    except Exception as exc:
        write(job/'ERROR.json',{'error':repr(exc),'traceback':traceback.format_exc(),'observed_steps':len(records)})
        if records:write(job/'PARTIAL_CONTACT_READBACK.json',records)
        return 1
    finally:
        if env is not None:
            try:env.close()
            except Exception:pass
        try:app.close()
        except Exception:pass

def launch():
    check_sources()
    if (OUT/'LAUNCH_MANIFEST.json').exists():raise RuntimeError('Use existing jobs; no automatic repeated probes')
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'LAUNCH_MANIFEST.json',{'authorization':'User approved three instrumented probe-only diagnostics after blocked audit',
        'contexts':['high','mid','low'],'single_probe_per_context':True,'task_branches':0,'training':False,
        'parameters':PHASE_PARAMETERS,'frozen_sources':{str(p):h for p,h in EXPECTED.items()},
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'current_execution_contract':str(smoke.VALIDATION/'CURRENT_EXECUTION_CONTRACT.json')})
    results=[]
    for band in ['high','mid','low']:
        job=OUT/'jobs'/band;job.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy()
        env.update({'PYTHONNOUSERSITE':'1','PYTHONPATH':os.pathsep.join([str(smoke.WARP_CORE),str(TAB),str(smoke.OPENPI)]),
            'OMNI_KIT_ACCEPT_EULA':'YES','ACCEPT_EULA':'Y','TABERO_ROOT':str(TAB),'P5S0C_OUT':str(job),
            'HDF5_TRAJ_SOURCE_DIR':str(TAB/'benchmarks/datasets/libero/assembled_hdf5'),
            'LIBERO_CONFIG_DIR':str(TAB/'benchmarks/datasets/libero/config'),
            'LIBERO_ASSETS_DATA_DIR':str(TAB/'benchmarks/datasets/libero/USD')})
        cmd=[str(smoke.ISAAC_PY),'-u',str(Path(__file__).resolve()),'--worker',band]
        print(f'START_PROBE_ONLY {band}',flush=True)
        with (job/'WORKER_STDOUT.log').open('x') as log:
            p=subprocess.Popen(cmd,cwd=TAB,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(job/'PROCESS.json',{'pid':p.pid,'command':cmd,'start_time':time.time()})
            rc=p.wait()
        results.append({'band':band,'returncode':rc})
        write(OUT/'PROCESS_RESULTS.json',results)
        print(f'END_PROBE_ONLY {band} rc={rc}',flush=True)
        if rc:break
    check_sources()
    return 0 if len(results)==3 and all(x['returncode']==0 for x in results) else 1

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['high','mid','low']);ap.add_argument('--launch',action='store_true')
    args=ap.parse_args()
    if args.worker:raise SystemExit(worker(args.worker))
    if args.launch:raise SystemExit(launch())
    ap.error('explicit --launch or --worker required')
