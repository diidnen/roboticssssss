"""Gated current-contract investigation. This entrypoint only runs probes.

No task-branch launcher is exposed until the probe/belief gate has passed.
Friction patches are read separately: ContactSensor net_forces_w is NORMAL
contact force, not total contact force and not the friction force.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np

ROOT = Path('/home/exouser/FORTE')
sys.path.insert(0, str(ROOT))
import activeforcing_e2e_task0_smoke_20260905 as smoke
from activeforcing_decision_state import DecisionClock, digest, sha256
from activeforcing_feasibility_features import from_saved_probe
from probe_only_instrumented_20260905 import jsonable

OUT = ROOT/'analysis/results/preaction_decision_contract_20260905'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+'\n')


def patch_data(sensor, dt):
    """Copy normal buffers before friction API reuses counts/start storage."""
    view = sensor.contact_physx_view
    force, points, normals, separation, counts, starts = [x.detach().cpu().numpy().copy()
        for x in view.get_contact_data(dt)]
    normal_pairs = []
    for index in np.ndindex(counts.shape):
        start, count = int(starts[index]), int(counts[index])
        if count and start + count >= view.max_contact_data_count:
            raise RuntimeError('Contact patch buffer possibly saturated; cannot trust force readback')
        ids = slice(start, start+count)
        normal_pairs.append({'pair':index,'count':count,'normal_forces':force[ids],
            'points':points[ids],'normals':normals[ids],'separations':separation[ids],
            'sum_normal_vector_world':(force[ids]*normals[ids]).sum(0)})
    friction, fpoints, fcounts, fstarts = [x.detach().cpu().numpy().copy()
        for x in view.get_friction_data(dt)]
    friction_pairs = []
    for index in np.ndindex(fcounts.shape):
        start, count = int(fstarts[index]), int(fcounts[index])
        if count and start + count >= view.max_contact_data_count:
            raise RuntimeError('Friction patch buffer possibly saturated')
        ids = slice(start, start+count)
        friction_pairs.append({'pair':index,'count':count,'friction_forces_world':friction[ids],
            'points':fpoints[ids],'sum_friction_vector_world':friction[ids].sum(0)})
    return {'normal':normal_pairs,'friction':friction_pairs,
            'physics_dt':dt, 'capacity':view.max_contact_data_count}


def worker(band, run_name, repaired_stop=False):
    import torch
    job=OUT/run_name/band
    job.mkdir(parents=True,exist_ok=True)
    if (job/'STARTED.json').exists(): raise RuntimeError('Refusing overwrite or repeated physical trial')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,num_envs=1).app
    env=None; records=[]; clock=DecisionClock()
    try:
        import gymnasium as gym
        import tac_manip.tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        p5=smoke.load_module('preaction_p5',smoke.P5_PATH);p5.OUT=job
        plan=next(x for x in smoke.contexts() if x['friction_band'].lower()==band)
        probe_source=ROOT/'activeforcing_current_probe.py' if repaired_stop else smoke.P4_PATH
        p5.P4_COLLECT=probe_source
        p4=p5.import_p4_probe(0);cid=plan['context_id']
        setup_task_objects(p5.TASK_SUITE,0)
        cfg=parse_env_cfg(smoke.ENV_ID,device='cuda:0',num_envs=1);cfg.episode_length_s=45.
        # Readback capacity only; no changed material, solver, gains or action.
        sensor_cfg=getattr(cfg.scene,'contact_grasp_'+p4.OBJ_NAME)
        sensor_cfg.max_contact_data_count_per_prim=128
        contract={'env_id':smoke.ENV_ID,'p4_source':str(probe_source),
            'p4_sha256':sha256(probe_source),'p5_sha256':sha256(smoke.P5_PATH),
            'actions':str(cfg.actions),'physics_dt':cfg.sim.dt,'decimation':cfg.decimation,
            'preaction_state':'POST_PROBE_AFTER_RETURN_AND_HOLD',
            'force_domain':[3.,5.],'force_semantics':'CONTINUOUS_GRASP_FORCE_SETPOINT',
            'full_task_success_y':'lift_success AND place_success AND NOT dropped',
            'candidate_actions_in_probe':0,'task_branches':0,
            'repaired_stop_diagnostic':repaired_stop,
            'sensor_semantics':'normal and friction patches recorded separately, raw policy input UNCHANGED'}
        write(job/'EXECUTION_CONTRACT.json',contract)
        env=gym.make(smoke.ENV_ID,cfg=cfg).unwrapped
        env.reset(seed=int(plan['root_seed']))
        original_step=env.step
        def observed_step(action):
            if clock.phase=='DECISION_STATE': raise RuntimeError('No env.step allowed during planning')
            result=original_step(action);clock.stepped()
            obj=env.scene[p4.OBJ_NAME]
            records.append({'step':clock.step,'action':jsonable(action),
                'policy_local_normal_projection':jsonable(result[0]['policy']['gripper_net_force']),
                'eef_pose':jsonable(result[0]['policy']['eef_pose']),
                'finger_joints':jsonable(env.scene['robot'].data.joint_pos[0,-2:]),
                'aperture_m':float(env.scene['robot'].data.joint_pos[0,-2:].sum()),
                'object_position':jsonable(obj.data.root_pos_w),
                'object_velocity':jsonable(obj.data.root_lin_vel_w),
                'object_quaternion':jsonable(obj.data.root_quat_w),
                'target_object_force':p5.target_object_force_snapshot(env,p4,p4.OBJ_NAME),
                'patches':jsonable(patch_data(env.scene['contact_grasp_'+p4.OBJ_NAME],float(cfg.sim.dt))),
                'controller_debug':jsonable(p4._dbg(env))})
            return result
        env.step=observed_step
        write(job/'STARTED.json',{'context':plan,'stage':'PROBE_DIAGNOSTIC_ONLY'})
        kwargs={}
        if repaired_stop:
            from activeforcing_probe_friction_contract import FrictionProbeBudget
            kwargs['termination_signal']=FrictionProbeBudget(records,p4._quat_apply_np)
        rows,rec=p4.run_probe_episode(env,seed_idx=int(plan['root_seed']),
            mu=float(plan['hidden_friction_analysis_only']),trial_id=cid,dt=smoke.DT,**kwargs)
        clock.finish_probe()
        raw=[vars(x) for x in rows]
        smoke.write_csv(job/'RAW_PROBE.csv',raw)
        write(job/'CONTACT_PATCH_READBACK.json',records)
        state=smoke.clone_cpu(env.scene.get_state(is_relative=True))
        # Capture additional controller state but do NOT claim solver-exact restoration.
        arm=env.action_manager.get_term('arm_action')
        controller_state={k:smoke.clone_cpu(v) for k,v in vars(arm).items()
            if isinstance(v,(torch.Tensor,int,float,bool,str))}
        snapshot=job/'DECISION_STATE.pt'
        torch.save({'state':state,'context_id':cid,'controller_tensor_state':controller_state,
            'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
            'observation_step':clock.step,'candidate_actions_already_executed':0,
            'full_execution_restore_validated':False},snapshot)
        last=raw[-1]
        commands=[{'phase':'branch_hold',**{f'cmd_{a}':last[f'eef_{a}'] for a in 'xyz'}}]*8
        smoke.write_csv(job/'PREACTION_NOMINAL_COMMANDS.csv',commands)
        posterior=smoke.posterior_from_probe(job/'RAW_PROBE.csv')
        write(job/'POSTERIOR.json',posterior)
        canonical,prov=from_saved_probe(cid,job/'RAW_PROBE.csv',job/'PREACTION_NOMINAL_COMMANDS.csv',
            0,3.,posterior['member_means'][0],state_path=snapshot)
        a=clock.extract(context_id=cid,task=0,snapshot_path=snapshot,
            execution_contract_hash=digest(contract),state=canonical[0,19:32],mask=canonical[0,32:45],
            commands=[[c['cmd_'+v] for v in 'xyz'] for c in commands],phases=['branch_hold']*8,
            posterior_support=posterior['member_means'],posterior_weights=[1/3]*3)
        np.testing.assert_array_equal(canonical,a.input(3.,posterior['member_means'][0]))
        write(job/'DECISION_FEATURES.json',{'decision':asdict(a),'feature_provenance':prov,
            'parity':a.candidate_parity(), 'live_clock_step_after_planning':clock.step,
            'EXTRACTION_BEFORE_CANDIDATE_ACTION':True,'EXECUTION_CONTRACT_PARITY':'NOT_YET_PHYSICALLY_VALIDATED'})
        write(job/'RESULT.json',{'context':plan,'probe_record':rec,'steps':clock.step,
            'outward_steps':sum(x['probe_phase']=='probe_out' for x in raw),
            'posterior_mean':posterior['mean'],'posterior_support':posterior['member_means'],
            'decision_feature_parity':a.candidate_parity(),'task_branches':0,
            'physical_diagnostic_executed':True,'pilot_physics_executed':False,
            'controller_changed':False,'utility_changed':False,'raw_belief_features_changed':False})
        print('PROBE_CONTRACT_DIAGNOSTIC_COMPLETE',band,flush=True)
        return 0
    except Exception as exc:
        write(job/'ERROR.json',{'error':repr(exc),'traceback':traceback.format_exc(),'step':clock.step})
        if records:write(job/'PARTIAL_READBACK.json',records)
        return 1
    finally:
        if env is not None:env.close()
        app.close()


def launch(bands, run_name, repaired_stop=False):
    for band in bands:
        job=OUT/run_name/band;job.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy()
        env.update({'PYTHONNOUSERSITE':'1','PYTHONPATH':os.pathsep.join([str(smoke.WARP_CORE),str(smoke.TABERO),str(smoke.OPENPI)]),
            'OMNI_KIT_ACCEPT_EULA':'YES','ACCEPT_EULA':'Y','TABERO_ROOT':str(smoke.TABERO),'P5S0C_OUT':str(job),
            'HDF5_TRAJ_SOURCE_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/assembled_hdf5'),
            'LIBERO_CONFIG_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/config'),
            'LIBERO_ASSETS_DATA_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/USD')})
        cmd=[str(smoke.ISAAC_PY),'-u',str(Path(__file__).resolve()),'--worker',band,'--run-name',run_name]
        if repaired_stop:cmd.append('--repaired-stop')
        with (job/'WORKER.log').open('x') as log:
            process=subprocess.Popen(cmd,cwd=smoke.TABERO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(job/'PROCESS.json',{'pid':process.pid,'command':cmd})
            print('START',band,process.pid,flush=True)
            code=process.wait()
        print('END',band,code,flush=True)
        if code:return code
    return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['high','mid','low'])
    ap.add_argument('--launch',nargs='+',choices=['high','mid','low'])
    ap.add_argument('--run-name',default='normal_vs_friction_readback')
    ap.add_argument('--repaired-stop',action='store_true')
    args=ap.parse_args()
    if args.worker:raise SystemExit(worker(args.worker,args.run_name,args.repaired_stop))
    if args.launch:raise SystemExit(launch(args.launch,args.run_name,args.repaired_stop))
    ap.error('Explicit probe-only mode required')
