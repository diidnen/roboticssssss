"""Gated snapshot replay diagnostic and, only on PASS, task0 matched pilot.

No training, utility/force-domain edits, posterior-interface edits or controller edits.
First uninterrupted 4N hold20 is a RESTORE DIAGNOSTIC, not a task label.
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

ROOT=Path('/home/exouser/FORTE');sys.path.insert(0,str(ROOT))
import activeforcing_e2e_task0_smoke_20260905 as smoke
from activeforcing_preact_action_probe_pilot_20260905 import patch_data,jsonable
from activeforcing_probe_friction_contract import FrictionProbeBudget
from activeforcing_decision_state import DecisionClock,digest,sha256
from current_contract_belief_features import ProbeEvidence
OUT=ROOT/'analysis/results/current_contract_restore_and_matched_pilot_20260905'
RUN_NAME=''
BELIEF=ROOT/'analysis/results/current_contract_physical_belief_20260905/PHYSICAL_BELIEF_CANDIDATE_MANIFEST.json'
CONTROLLER=smoke.TABERO/'source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py'
TOLERANCES={'joint_position_m':1e-5,'object_position_m':1e-5,
    'object_quaternion_component':1e-5,'object_velocity':1e-4,
    'normal_force_N':.01,'d_command_m':1e-5,'f_filter_N':.01}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')


def check_sources():
    manifest=json.loads(BELIEF.read_text())
    if not manifest['task0_candidate_qualified']:raise ValueError('Belief candidate not qualified')
    for p,h in manifest['source_hashes'].items():
        if sha256(p)!=h:raise ValueError('Frozen source changed: '+p)
    return manifest


def freeze():
    check_sources();path=OUT/'PROTOCOL.json'
    if path.exists():raise ValueError('Protocol already frozen')
    write(path,{'scope':'task0 restore diagnosis, then gated matched pilot only; no full collection',
        'contexts':smoke.contexts(),'forces':[3.,4.,5.],
        'restore_audit_force':4.,'restore_audit_steps':20,'restored_replays':2,
        'handoff':'preserve last probe raw gripper action command; NOT actual joint position or D_CLOSED',
        'trajectory_tolerances':TOLERANCES,'same_initial_state_tolerance':1e-7,
        'first_gate':'uninterrupted hold vs two full-state restored holds; all channels within frozen tolerances',
        'posterior_semantics':'UNCHANGED three member means; sigma integration not silently added',
        'posterior_calibration_validated':False,'pilot_is_empirical_boundary_not_planner_benchmark':True,
        'full_dataset_collection_allowed':False,'full_task_success_y':'lift_success AND place_success AND NOT dropped',
        'label_source':'CURRENT_MATCHED_FULL_TASK_BRANCH','branch_runner':str(smoke.P5_PATH),
        'sha256':{str(p):sha256(p) for p in [Path(__file__),ROOT/'activeforcing_execution_snapshot.py',
            ROOT/'activeforcing_decision_state.py',ROOT/'activeforcing_feasibility_features.py',smoke.P5_PATH,CONTROLLER,BELIEF]}})


def read_protocol():
    protocol=json.loads((OUT/'PROTOCOL.json').read_text());check_sources()
    for p,h in protocol['sha256'].items():
        if sha256(p)!=h:raise ValueError('Protocol source changed: '+p)
    return protocol


def readout(env,p4):
    obj=env.scene[p4.OBJ_NAME].root_physx_view
    robot=env.scene['robot'].root_physx_view
    patches=patch_data(env.scene['contact_grasp_'+p4.OBJ_NAME],float(env.cfg.sim.dt))
    dbg=p4._dbg(env)
    transform=obj.get_transforms()[0].detach().cpu().numpy().copy()
    return {'joint_position_m':robot.get_dof_positions()[0].detach().cpu().numpy().copy(),
        'object_position_m':transform[:3],'object_quaternion_component':transform[3:],
        'object_velocity':obj.get_velocities()[0].detach().cpu().numpy().copy(),
        'normal_force_N':[sum(float(f[0]) for f in p['normal_forces']) for p in patches['normal']],
        'd_command_m':float(p4._f(dbg.get('d_cmd'),0.)),
        'f_filter_N':float(p4._f(dbg.get('f_sq_meas'),0.)),
        'episode_step':int(env.episode_length_buf[0].item())}


def compare_trace(a,b):
    if len(a)!=len(b):return {'passed':False,'reason':'step count mismatch'}
    errors={k:float(np.max(np.abs(np.asarray([r[k] for r in a])-np.asarray([r[k] for r in b])))) for k in TOLERANCES}
    return {'passed':all(np.isfinite(v) and v<=TOLERANCES[k] for k,v in errors.items()),
        'max_abs_error':errors,'tolerances':TOLERANCES,
        'episode_step_equal':[r['episode_step'] for r in a]==[r['episode_step'] for r in b]}


def build_features(job,plan):
    # Host process: old Isaac NumPy ABI must not load host-trained checkpoints.
    from infer_current_contract_belief import infer_saved_decision
    from activeforcing_feasibility_features import from_saved_probe
    posterior=infer_saved_decision(BELIEF,job)
    write(job/'POSTERIOR.json',posterior)
    raw=smoke.load_module('feature_csv_module',ROOT/'activeforcing_feasibility_features.py').rows(job/'RAW_PROBE.csv')
    last=raw[-1];commands=[{'phase':'branch_hold',**{f'cmd_{k}':last['eef_'+k] for k in 'xyz'}} for _ in range(8)]
    smoke.write_csv(job/'PREACTION_COMMANDS.csv',commands)
    x,provenance=from_saved_probe(plan['context_id'],job/'RAW_PROBE.csv',job/'PREACTION_COMMANDS.csv',0,3.,posterior['member_means'][0],state_path=job/'DECISION_STATE.pt')
    clock=DecisionClock();clock.step=len(raw);clock.finish_probe()
    d=clock.extract(context_id=plan['context_id'],task=0,snapshot_path=job/'DECISION_STATE.pt',
        execution_contract_hash=sha256(OUT/'PROTOCOL.json'),state=x[0,19:32],mask=x[0,32:45],
        commands=[[float(c['cmd_'+k]) for k in 'xyz'] for c in commands],phases=['branch_hold']*8,
        posterior_support=posterior['member_means'],posterior_weights=[1/3]*3)
    np.testing.assert_array_equal(x,d.input(3.,posterior['member_means'][0]))
    tensors=np.stack([d.input(f,mu) for f in [3.,4.,5.] for mu in posterior['member_means']])
    np.save(job/'FROZEN_PREACTION_FEATURES.npy',tensors)
    write(job/'DECISION_FEATURES.json',{'decision':asdict(d),'parity':d.candidate_parity(),
        'provenance':provenance,'no_candidate_executed_before_extraction':True,
        'feature_array_sha256':sha256(job/'FROZEN_PREACTION_FEATURES.npy')})


def worker(band,allow_pilot):
    import torch
    from activeforcing_execution_snapshot import capture,restore,max_difference
    protocol=read_protocol();plan=next(p for p in protocol['contexts'] if p['friction_band'].lower()==band)
    job=OUT/band;job.mkdir(parents=True,exist_ok=True)
    if (job/'STARTED.json').exists():raise RuntimeError('No overwrite of a physical trial')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,num_envs=1).app
    env=None;records=[];mode={'name':'PROBE'};terminal=[]
    try:
        import gymnasium as gym
        import tac_manip.tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        p5=smoke.load_module('current_matched_p5',smoke.P5_PATH);p5.OUT=job
        p5.P4_COLLECT=ROOT/'activeforcing_current_probe.py';p4=p5.import_p4_probe(0)
        setup_task_objects(p5.TASK_SUITE,0)
        cfg=parse_env_cfg(smoke.ENV_ID,device='cuda:0',num_envs=1);cfg.episode_length_s=45.
        getattr(cfg.scene,'contact_grasp_'+p4.OBJ_NAME).max_contact_data_count_per_prim=128
        env=gym.make(smoke.ENV_ID,cfg=cfg).unwrapped;env.reset(seed=int(plan['root_seed']))
        original_step=env.step;original_reset=env._reset_idx
        def observed_reset(ids):
            if mode['name'] in ('RESTORE_AUDIT','BRANCH') and len(ids):
                terminal.append({'mode':mode['name'],'before_auto_reset':readout(env,p4),
                    'terms':{k:jsonable(env.termination_manager.get_term(k)) for k in env.termination_manager.active_terms}})
            return original_reset(ids)
        env._reset_idx=observed_reset
        def observed_step(action):
            if mode['name']=='DECISION_STATE':raise RuntimeError('Action during hypothetical planning')
            result=original_step(action)
            if mode['name']=='PROBE':
                obj=env.scene[p4.OBJ_NAME]
                records.append({'step':len(records)+1,'action':jsonable(action),
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
        write(job/'STARTED.json',{'context':plan,'scope':'probe + restore audit; pilot only if gate passes'})
        rows,rec=p4.run_probe_episode(env,seed_idx=int(plan['root_seed']),mu=float(plan['hidden_friction_analysis_only']),
            trial_id=plan['context_id'],dt=smoke.DT,termination_signal=FrictionProbeBudget(records,p4._quat_apply_np))
        raw=[vars(r) for r in rows]
        if any(rec.get(k,0) for k in ('probe_failure','contact_lost_probe','dropped','major_disturbance')):
            raise RuntimeError('Probe admission failed')
        mode['name']='DECISION_STATE'
        smoke.write_csv(job/'RAW_PROBE.csv',raw);write(job/'CONTACT_PATCH_READBACK.json',records)
        np.save(job/'RAW_BELIEF_FEATURES.npy',ProbeEvidence().rows(raw,records))
        frozen=capture(env)
        frozen.update(context_id=plan['context_id'],observation_step=len(raw),candidate_actions_already_executed=0,
            controller_tensor_state=frozen['objects']['arm_action'],full_execution_restore_validated=False)
        torch.save(frozen,job/'DECISION_STATE.pt')
        write(job/'PROBE_RESULT.json',{'context':plan,'record':rec,'total_steps':len(raw),
            'outward_steps':sum(r['probe_phase']=='probe_out' for r in raw)})
        host_env=os.environ.copy();host_env.pop('PYTHONPATH',None);host_env.pop('PYTHONNOUSERSITE',None)
        result=subprocess.run([str(smoke.HOST_PY),str(Path(__file__).resolve()),'--features',band,'--run-name',RUN_NAME],cwd=ROOT,env=host_env,timeout=120,capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Pre-action feature extraction failed: '+result.stdout+result.stderr)
        original_physical=readout(env,p4)
        initial=raw[-1];pos=np.array([initial['eef_'+k] for k in 'xyz'],float)
        obs=env.observation_manager.compute();aa=p4._aa(obs['policy']['eef_pose'][0,3:7].detach().cpu().numpy())
        # Actual joint position is a measurement, not the pending action
        # command. Replacing the latter by the former opened this grasp.
        d0=float(records[-1]['action'][0][6])
        if not p4.D_CLOSED<=d0<=p4.D_OPEN:raise ValueError('Invalid saved gripper action command')
        saved_cmd=float(frozen['objects']['arm_action']['_gripper_abs_cmd'][0,0])
        if abs(d0-saved_cmd)>1e-8:raise ValueError('Probe action/controller handoff command mismatch')
        write(job/'HANDOFF_AUDIT.json',{'last_probe_action_d':d0,'controller_gripper_abs_cmd':saved_cmd,
            'actual_joint_mean':float(env.scene['robot'].data.joint_pos[0,-2:].mean()),
            'first_candidate_action_d':d0,'gripper_command_jump_m':0.,
            'D_CLOSED_hardcode_used':False,'controller_changed':False})
        def audit_hold():
            mode['name']='RESTORE_AUDIT';d=d0;trace=[];actions=[]
            for i in range(20):
                action=p4._make_action(pos,aa,d,4.,env.device)
                _,_,term,trunc,_=env.step(action)
                trace.append(readout(env,p4));actions.append(jsonable(action))
                d=float(p4._force_servo(d,p4._f(p4._dbg(env).get('f_sq_meas'),0.),4.))
                if bool(term[0]) or bool(trunc[0]):raise RuntimeError('Termination during restore hold audit')
            return trace,actions
        reference,reference_actions=audit_hold()
        write(job/'UNINTERRUPTED_HOLD.json',{'trace':reference,'actions':reference_actions})
        checks=[]
        for repeat in range(2):
            mode['name']='RESTORING';restore(env,frozen)
            restored=capture(env)
            before=readout(env,p4)
            initial_errors={k:max_difference(original_physical[k],before[k]) for k in ['joint_position_m','object_position_m','object_quaternion_component','object_velocity']}
            state_errors={k:max_difference(frozen[k],restored[k]) for k in ['state','environment','joint_targets','objects']}
            replay,actions=audit_hold();comparison=compare_trace(reference,replay)
            comparison.update(repeat=repeat,initial_physical_errors=initial_errors,
                exposed_state_errors=state_errors,actions_max_difference=max_difference(reference_actions,actions))
            comparison['passed']=bool(comparison['passed'] and comparison['episode_step_equal'] and
                all(np.isfinite(v) and v<=1e-7 for v in initial_errors.values()) and
                all(np.isfinite(v) and v<=1e-7 for v in state_errors.values()))
            checks.append(comparison)
            write(job/f'RESTORED_HOLD_{repeat}.json',{'comparison':comparison,'trace':replay,'actions':actions})
        gate=all(c['passed'] for c in checks)
        write(job/'RESTORE_GATE.json',{'passed':gate,'checks':checks,'snapshot_sha256':sha256(job/'DECISION_STATE.pt'),
            'qualification_scope':'observed same-action hold20 replay only, not proof of hidden solver bitwise restoration',
            'full_task_execution_parity_validated':False})
        # Full task replay must also be checked before any supervised row can
        # be admitted. This first entrypoint intentionally stops at hold gate.
        write(job/'RESULT.json',{'restore_hold_gate_passed':gate,'allow_pilot_requested':allow_pilot,
            'task_branches':0,'feasibility_labels':0,'next_gate':'same-force full-task replay' if gate else 'diagnose restore mismatch',
            'controller_changed':False,'utility_changed':False,'posterior_interface_changed':False})
        read_protocol();print('RESTORE_GATE',band,gate,flush=True)
        return 0 if gate else 2
    except Exception as exc:
        write(job/'ERROR.json',{'error':repr(exc),'traceback':traceback.format_exc(),'mode':mode['name']})
        return 1
    finally:
        if terminal:write(job/'PRE_RESET_TERMINAL_STATES.json',terminal)
        if env is not None:env.close()
        app.close()


def launch(bands,allow_pilot):
    read_protocol()
    for band in bands:
        job=OUT/band;job.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env.update({'PYTHONNOUSERSITE':'1','PYTHONPATH':os.pathsep.join([str(smoke.WARP_CORE),str(smoke.TABERO),str(smoke.OPENPI)]),
            'OMNI_KIT_ACCEPT_EULA':'YES','ACCEPT_EULA':'Y','TABERO_ROOT':str(smoke.TABERO),'P5S0C_OUT':str(job),
            'HDF5_TRAJ_SOURCE_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/assembled_hdf5'),
            'LIBERO_CONFIG_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/config'),
            'LIBERO_ASSETS_DATA_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/USD')})
        cmd=[str(smoke.ISAAC_PY),'-u',str(Path(__file__).resolve()),'--worker',band,'--run-name',RUN_NAME]
        if allow_pilot:cmd.append('--allow-pilot')
        with (job/'WORKER.log').open('x') as log:
            proc=subprocess.Popen(cmd,cwd=smoke.TABERO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(job/'PROCESS.json',{'pid':proc.pid,'command':cmd});print('START',band,proc.pid,flush=True)
            rc=proc.wait(timeout=1200)
        print('END',band,rc,flush=True)
        if rc:return rc
        # Kit shutdown may mask Python SystemExit; artifacts, not process
        # exit status alone, decide whether the scientific gate passed.
        if (job/'ERROR.json').exists():return 1
        if not (job/'RESULT.json').exists():return 1
        if not json.loads((job/'RESULT.json').read_text())['restore_hold_gate_passed']:return 2
    return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true')
    ap.add_argument('--launch',nargs='+',choices=['high','mid','low']);ap.add_argument('--worker',choices=['high','mid','low'])
    ap.add_argument('--features',choices=['high','mid','low']);ap.add_argument('--allow-pilot',action='store_true')
    ap.add_argument('--run-name',default='')
    a=ap.parse_args()
    if a.run_name:
        if not a.run_name.replace('_','').isalnum():ap.error('Simple run name required')
        RUN_NAME=a.run_name;OUT=OUT/RUN_NAME
    if a.freeze:freeze()
    elif a.features:
        p=next(p for p in read_protocol()['contexts'] if p['friction_band'].lower()==a.features)
        build_features(OUT/a.features,p)
    elif a.worker:raise SystemExit(worker(a.worker,a.allow_pilot))
    elif a.launch:raise SystemExit(launch(a.launch,a.allow_pilot))
    else:ap.error('Explicit gated mode required')
