"""Authorized task0 probe-only corpus. No branch/feasibility API is called."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
OUT=ROOT/'analysis/results/current_contract_physical_belief_20260905'
sys.path.insert(0,str(ROOT))
import activeforcing_e2e_task0_smoke_20260905 as smoke
from activeforcing_preact_action_probe_pilot_20260905 import patch_data, jsonable
from activeforcing_probe_friction_contract import FrictionProbeBudget
from current_contract_belief_features import ProbeEvidence
from activeforcing_decision_state import DecisionClock


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')


def freeze():
    if (OUT/'PROTOCOL.json').exists():raise RuntimeError('Protocol already frozen; use --collect')
    OUT.mkdir(parents=True,exist_ok=True)
    p5=smoke.load_module('belief_plan_only',smoke.P5_PATH)
    plans=[]
    for p in p5.context_plan_for_task(0):
        p=dict(p);p['historical_split']=p['split']
        p['split']='DIAGNOSTIC' if p['root_index']==0 else ('VAL' if p['split']=='DEV' else p['split'])
        p['data_path']=str(ROOT/'analysis/results/preaction_decision_contract_20260905/corrected_friction_stop'/p['friction_band'].lower()) if p['split']=='DIAGNOSTIC' else str(OUT/'data'/p['context_id'])
        plans.append(p)
    groups={s:sorted({p['root_seed'] for p in plans if p['split']==s}) for s in ['TRAIN','VAL','TEST','DIAGNOSTIC']}
    assert sum(map(len,groups.values()))==len(set(v for x in groups.values() for v in x))==12
    paths=[ROOT/'activeforcing_current_probe.py',ROOT/'activeforcing_probe_friction_contract.py',
        ROOT/'current_contract_belief_features.py',ROOT/'collect_current_contract_belief_20260905.py',
        ROOT/'activeforcing_preact_action_probe_pilot_20260905.py',smoke.P5_PATH,
        Path('/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py')]
    protocol={'scope':'task0 probe-only physical-belief development, NOT four-task or feasibility qualification',
        'authorization':'User approved root-grouped probe-only collection and physical-belief retraining',
        'contexts':plans,'root_split':groups,'seed_count':3,'training_seeds':[0,1,2],
        'architecture':'Linear(input_dim,16)+ReLU -> GRU(16,16) -> mu/log_sigma heads',
        'epochs':80,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,
        'gradient_clip':1.,'loss':'Gaussian NLL (without constant) + 0.05*MAE',
        'sampling':'per-epoch bootstrap TRAIN roots with replacement, all contexts of each sampled root',
        'normalization':'TRAIN only, global unpadded sample/time rows, all channels, std floor 1e-6',
        'checkpoint_rule':'each seed lowest VAL Gaussian NLL; equal-weight ensemble of ALL three selected seeds',
        'posterior_interface':'unchanged three member means + member log_sigma; empirical member support remains unchanged',
        'calibration':'NONE; report raw uncertainty; no calibration search',
        'eligibility_rule':{'scope':'task0 candidate only','both_val_test':'MAE and mixture NLL better than TRAIN Gaussian prior',
            'positive_member_means':True,'test_probe_admission_fraction_min':.9,
            'test_complete_root_high_mid_low_order_fraction_min':.75,'test_90_interval_coverage_min':.75},
        'diagnostic_root_excluded_from_training_and_selection':True,
        'task_branches_allowed':False,'feasibility_labels_allowed':False,'max_parallel_physics_workers':2,
        'probe_quality_rule':'no probe failure/contact loss/drop/major disturbance; return10 + hold5 completed; final bilateral',
        'sha256':{str(p):sha(p) for p in paths}}
    write(OUT/'PROTOCOL.json',protocol)
    write(OUT/'SPLIT_MANIFEST.json',{'root_split':groups,'context_counts':{s:sum(p['split']==s for p in plans) for s in groups},'ROOT_LEAKAGE':False})
    write(OUT/'FEATURE_SCHEMA.json',ProbeEvidence().schema())
    print(json.dumps({'new_physical_probes':33,'reused_diagnostic_probes':3,'root_split':groups},indent=2))


def check_protocol():
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    for p,h in protocol['sha256'].items():
        if sha(p)!=h:raise RuntimeError('Frozen source changed: '+p)
    return protocol


def worker(cid):
    import torch
    protocol=check_protocol();plan=next(p for p in protocol['contexts'] if p['context_id']==cid)
    if plan['split']=='DIAGNOSTIC':raise RuntimeError('Reuse locked diagnostic; no new run')
    job=Path(plan['data_path']);job.mkdir(parents=True,exist_ok=True)
    if (job/'STARTED.json').exists():raise RuntimeError('No silent repeated physical run')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,num_envs=1).app
    env=None;records=[];clock=DecisionClock()
    try:
        import gymnasium as gym
        import tac_manip.tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        p5=smoke.load_module('belief_probe_only_p5',smoke.P5_PATH);p5.OUT=job;p5.P4_COLLECT=ROOT/'activeforcing_current_probe.py'
        p4=p5.import_p4_probe(0)
        setup_task_objects(p5.TASK_SUITE,0)
        cfg=parse_env_cfg(smoke.ENV_ID,device='cuda:0',num_envs=1);cfg.episode_length_s=45.
        getattr(cfg.scene,'contact_grasp_'+p4.OBJ_NAME).max_contact_data_count_per_prim=128
        write(job/'RUNTIME_CONFIG.json',{'actions':str(cfg.actions),'physics_dt':cfg.sim.dt,'decimation':cfg.decimation,
            'creation_seed':cfg.seed,'context_reset_seed':plan['root_seed'],'protocol_sha256':sha(OUT/'PROTOCOL.json')})
        env=gym.make(smoke.ENV_ID,cfg=cfg).unwrapped;env.reset(seed=int(plan['root_seed']))
        original_step=env.step
        def observed_step(action):
            if clock.phase!='PROBE':raise RuntimeError('No actions allowed after probe')
            result=original_step(action);clock.stepped();obj=env.scene[p4.OBJ_NAME]
            records.append({'step':clock.step,'action':jsonable(action),
                'policy_local_normal_projection':jsonable(result[0]['policy']['gripper_net_force']),
                'eef_pose':jsonable(result[0]['policy']['eef_pose']),
                'finger_joints':jsonable(env.scene['robot'].data.joint_pos[0,-2:]),
                'aperture_m':float(env.scene['robot'].data.joint_pos[0,-2:].sum()),
                'object_position':jsonable(obj.data.root_pos_w),'object_velocity':jsonable(obj.data.root_lin_vel_w),
                'object_quaternion':jsonable(obj.data.root_quat_w),
                'target_object_force':p5.target_object_force_snapshot(env,p4,p4.OBJ_NAME),
                'patches':jsonable(patch_data(env.scene['contact_grasp_'+p4.OBJ_NAME],float(cfg.sim.dt))),
                'controller_debug':jsonable(p4._dbg(env))})
            return result
        env.step=observed_step
        write(job/'STARTED.json',{'context_id':cid,'time':time.time(),'task_branches':0})
        rows,rec=p4.run_probe_episode(env,seed_idx=int(plan['root_seed']),mu=float(plan['hidden_friction_analysis_only']),
            trial_id=cid,dt=smoke.DT,termination_signal=FrictionProbeBudget(records,p4._quat_apply_np))
        clock.finish_probe();raw=[vars(r) for r in rows]
        smoke.write_csv(job/'RAW_PROBE.csv',raw);write(job/'CONTACT_PATCH_READBACK.json',records)
        state=smoke.clone_cpu(env.scene.get_state(is_relative=True));arm=env.action_manager.get_term('arm_action')
        torch.save({'state':state,'context_id':cid,'observation_step':clock.step,'candidate_actions_already_executed':0,
            'controller_tensor_state':{k:smoke.clone_cpu(v) for k,v in vars(arm).items() if isinstance(v,(torch.Tensor,int,float,bool,str))},
            'full_execution_restore_validated':False},job/'DECISION_STATE.pt')
        x=ProbeEvidence().rows(raw,records)
        # Both paths must use exactly the same observations and schema.
        np.testing.assert_array_equal(x,ProbeEvidence().load(job))
        np.save(job/'RAW_FEATURES.npy',x)
        reasons=[k for k in ['probe_failure','contact_lost_probe','dropped','major_disturbance'] if int(rec.get(k,0))]
        for phase,n in [('probe_back',10),('probe_hold',5)]:
            if sum(r['probe_phase']==phase for r in raw)!=n:reasons.append('incomplete_'+phase)
        if not (int(raw[-1]['contact_left']) and int(raw[-1]['contact_right'])):reasons.append('final_nonbilateral')
        mats=env.scene[p4.OBJ_NAME].root_physx_view.get_material_properties().detach().cpu().numpy().reshape(-1,3)
        if not np.allclose(mats[:,:2],float(plan['hidden_friction_analysis_only']),rtol=0,atol=1e-6):raise RuntimeError('Applied-mu label/readback mismatch')
        write(job/'RESULT.json',{'context':plan,'probe_record':rec,'steps':clock.step,
            'outward_steps':sum(r['probe_phase']=='probe_out' for r in raw),'probe_qualified':not reasons,
            'exclusion_reasons':reasons,'applied_object_static_dynamic_mu':mats[:,:2].tolist(),
            'training_target_mu':float(plan['hidden_friction_analysis_only']),'target_source':'explicit scenario setting verified by simulator readback; never a model input',
            'feature_shape':list(x.shape),'TRAIN_RUNTIME_FEATURE_PARITY':True,'MAX_FEATURE_DIFF':0.,
            'candidate_actions_executed':0,'task_branches':0,'feasibility_labels':0,
            'sha256':{name:sha(job/name) for name in ['RAW_PROBE.csv','CONTACT_PATCH_READBACK.json','DECISION_STATE.pt','RAW_FEATURES.npy']}})
        check_protocol();print('PROBE_ONLY_COMPLETE',cid,len(raw),not reasons,flush=True)
        return 0
    except Exception as exc:
        write(job/'ERROR.json',{'error':repr(exc),'traceback':traceback.format_exc(),'step':clock.step})
        if records:write(job/'PARTIAL_READBACK.json',records)
        return 1
    finally:
        if env is not None:env.close()
        app.close()


def launch_one(plan):
    job=Path(plan['data_path']);job.mkdir(parents=True,exist_ok=True)
    if (job/'RESULT.json').exists():return {'context_id':plan['context_id'],'status':'ALREADY_COMPLETE'}
    if (job/'STARTED.json').exists() or (job/'WORKER.log').exists():raise RuntimeError('Interrupted run requires inspection: '+str(job))
    env=os.environ.copy();env.update({'PYTHONNOUSERSITE':'1',
        'PYTHONPATH':os.pathsep.join([str(smoke.WARP_CORE),str(smoke.TABERO),str(smoke.OPENPI)]),
        'OMNI_KIT_ACCEPT_EULA':'YES','ACCEPT_EULA':'Y','TABERO_ROOT':str(smoke.TABERO),'P5S0C_OUT':str(job),
        'HDF5_TRAJ_SOURCE_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/assembled_hdf5'),
        'LIBERO_CONFIG_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/config'),
        'LIBERO_ASSETS_DATA_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/USD')})
    cmd=[str(smoke.ISAAC_PY),'-u',str(Path(__file__).resolve()),'--worker',plan['context_id']]
    with (job/'WORKER.log').open('x') as stream:
        proc=subprocess.Popen(cmd,cwd=smoke.TABERO,env=env,stdout=stream,stderr=subprocess.STDOUT)
        write(job/'PROCESS.json',{'pid':proc.pid,'command':cmd,'start_time':time.time()})
        print('START',plan['context_id'],proc.pid,flush=True)
        rc=proc.wait(timeout=900)
    print('END',plan['context_id'],rc,flush=True)
    return {'context_id':plan['context_id'],'returncode':rc}


def collect():
    protocol=check_protocol();plans=[p for p in protocol['contexts'] if p['split']!='DIAGNOSTIC']
    # Acquisition is fixed in advance. Test outcomes do not influence training.
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(launch_one,plans):
            results.append(result);write(OUT/'COLLECTION_PROGRESS.json',results)
    check_protocol()
    if any(r.get('returncode',0) for r in results):raise RuntimeError('Some acquisition jobs failed; inspect, do not train silently')
    write(OUT/'COLLECTION_COMPLETE.json',{'new_probes':33,'diagnostic_reused':3,'task_branches':0})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--collect',action='store_true');ap.add_argument('--worker')
    a=ap.parse_args()
    if a.freeze:freeze()
    elif a.collect:collect()
    elif a.worker:raise SystemExit(worker(a.worker))
    else:ap.error('Explicit mode required')
