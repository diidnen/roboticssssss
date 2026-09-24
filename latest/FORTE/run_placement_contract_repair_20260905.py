"""Full-horizon matched pilot with exact pre-action history reconstruction.

Every physical prefix replay is counted. It must reproduce the reference
actions, evidence and exposed decision snapshot before any candidate action.
Replayed observations never replace the reference feasibility input/posterior.
No model training or production posterior/utility/controller edits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np

ROOT=Path('/home/exouser/FORTE');sys.path.insert(0,str(ROOT))
import run_current_contract_restore_pilot_20260905 as base
import activeforcing_e2e_task0_smoke_20260905 as smoke
from activeforcing_command_handoff import command_from_probe
from activeforcing_placement_contract import Geometry, PlacementPath, CONFIG, evaluate_placement, adapt_placement_runner
from activeforcing_probe_friction_contract import FrictionProbeBudget
from activeforcing_preact_action_probe_pilot_20260905 import patch_data,jsonable
from activeforcing_decision_state import sha256
from current_contract_belief_features import ProbeEvidence
HISTORY_ROOT=ROOT/'analysis/results/current_contract_restore_and_matched_pilot_20260905/history_reconstruction'
OUT=HISTORY_ROOT/'placement_contract_repair'
PILOT_RUN_NAME='placement_contract_repair'
HIGH_REFERENCE=HISTORY_ROOT.parent/'command_continuity/high'
EXPECTED_PHASE_STEPS={'branch_hold':20,'lift':50,'transit':110,'over_basket':30,'place':40,'release':50,'settle':50}


def defer_success_only(manager):
    """Preserve raw terms and all failures/timeouts; suppress only success reset."""
    import torch
    if 'success' not in manager.active_terms:
        raise RuntimeError('Expected explicit success termination term')
    failures=torch.zeros_like(manager.terminated)
    for name in manager.active_terms:
        if name != 'success' and not manager.get_term_cfg(name).time_out:
            failures.logical_or_(manager.get_term(name))
    manager._terminated_buf.copy_(failures)
    return manager.dones


def full_horizon_place_success(phase_counts, final_goal, final_contact):
    return int(phase_counts == EXPECTED_PHASE_STEPS and final_goal and final_contact > .05)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')


def reference_dir(band):return HIGH_REFERENCE if band=='high' else HISTORY_ROOT/'references'/band
def job_dir(mode,band,force=4.,repeat=0):
    if mode=='reference':return HISTORY_ROOT/'references'/band
    if mode=='verify':return HISTORY_ROOT/'verification'/band
    return OUT/'branches'/band/f'F{force:g}_R{repeat}'


def freeze():
    base.check_sources()
    if (OUT/'PROTOCOL.json').exists():raise ValueError('Already frozen')
    branch=smoke.load_module('handoff_source_audit',smoke.P5_PATH)
    _,adaptation=adapt_placement_runner(branch)
    write(OUT/'BRANCH_CODE_PROVENANCE.json',adaptation)
    write(OUT/'PROTOCOL.json',{'scope':'bounded repaired-path task0 3x3 pilot plus three 4N repeat gates; not formal training data',
        'restore_method':'FRESH_PROCESS_EXACT_PREFIX_RECONSTRUCTION',
        'not_claimed':'literal restoration of hidden solver state from scene-only snapshot',
        'physical_replayed_probes_counted':True,'new_probe_query_for_selection':False,
        'history_reference_root':str(HISTORY_ROOT),'pilot_run_name':PILOT_RUN_NAME,
        'contexts':smoke.contexts(),'forces':[3.,4.,5.],'full_task_repeat_force':4.,
        'first_gate':'HIGH fresh-process hold20 vs uninterrupted reference; frozen tolerances',
        'prefix_gate':'exact action equality; exact 58D evidence; exact exposed decision state',
        'full_task_gate':'same repaired path 4N repeat; exact action/state comparison; no old-path parity claim',
        'same_action_tolerances':base.TOLERANCES,'maximum_new_physical_prefixes':12,
        'placement_config':CONFIG,'asset_geometry_sha256':sha256(OUT/'ASSET_GEOMETRY.npz'),
        'geometry_source_hashes':json.loads((OUT/'ASSET_GEOMETRY_PROVENANCE.json').read_text())['sources_sha256'],
        'full_horizon_required':True,'expected_phase_steps':EXPECTED_PHASE_STEPS,
        'success_termination':'Record intermediate task-goal satisfaction, defer success reset until all prescribed phases complete; drop/time-out remain immediate',
        'place_success':'final 20 steps geometric containment below rim in current basket frame, unheld, stable; complete phases',
        'full_task_success_y':'lift_success AND place_success AND NOT dropped',
        'goal_geometric_thresholds_changed':True,
        'controller_changed':False,'utility_changed':False,'posterior_interface_changed':False,
        'handoff':'last probe action command, NOT actual finger position and NOT hard-coded zero',
        'old720_labels_used':False,'full_collection_allowed':False,
        'sources_sha256':{str(p):sha256(p) for p in [Path(__file__),ROOT/'activeforcing_placement_contract.py',ROOT/'activeforcing_execution_snapshot.py',
            ROOT/'activeforcing_command_handoff.py',ROOT/'run_current_contract_restore_pilot_20260905.py',
            ROOT/'activeforcing_current_probe.py',ROOT/'activeforcing_probe_friction_contract.py',
            ROOT/'activeforcing_feasibility_features.py',smoke.P5_PATH,base.CONTROLLER,base.BELIEF]},
        'HIGH_reference_snapshot_sha256':sha256(HIGH_REFERENCE/'DECISION_STATE.pt')})


def protocol():
    p=json.loads((OUT/'PROTOCOL.json').read_text());base.check_sources()
    for f,h in p['sources_sha256'].items():
        if sha256(f)!=h:raise ValueError('Frozen pilot source changed: '+f)
    if sha256(OUT/'ASSET_GEOMETRY.npz')!=p['asset_geometry_sha256']:raise ValueError('Geometry changed')
    for f,h in p['geometry_source_hashes'].items():
        if sha256(f)!=h:raise ValueError('Asset/evaluator source changed: '+f)
    return p


def host_features(job,plan,reference):
    base.OUT=OUT
    base.build_features(job,plan)
    if not reference:
        ref=reference_dir(plan['friction_band'].lower())
        np.testing.assert_array_equal(np.load(job/'FROZEN_PREACTION_FEATURES.npy'),np.load(ref/'FROZEN_PREACTION_FEATURES.npy'))
        new=json.loads((job/'POSTERIOR.json').read_text());old=json.loads((ref/'POSTERIOR.json').read_text())
        np.testing.assert_array_equal(new['member_means'],old['member_means'])
    write(job/'FEATURE_GATE.json',{'passed':True,'reference_created':reference,
        'candidate_preact_state_equality':True,'max_feature_difference':0.,
        'reference_feature_source':str(job if reference else reference_dir(plan['friction_band'].lower()))})


def worker(mode,band,force,repeat):
    import torch
    from activeforcing_execution_snapshot import capture,max_difference
    p=protocol();plan=next(c for c in p['contexts'] if c['friction_band'].lower()==band)
    job=job_dir(mode,band,force,repeat);job.mkdir(parents=True,exist_ok=True)
    if (job/'STARTED.json').exists():raise RuntimeError('No overwrite or selective physical repeat')
    ref=reference_dir(band)
    ref_records=json.loads((ref/'CONTACT_PATCH_READBACK.json').read_text()) if mode!='reference' else None
    if mode=='branch':
        gate=json.loads((HISTORY_ROOT/'verification'/band/'RESULT.json').read_text())
        if not gate['passed']:raise ValueError('No passed same-action reconstruction verification')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,num_envs=1).app
    env=None;records=[];branch_steps=[];terminal=[];runtime={'mode':'PROBE','candidate_actions':0}
    try:
        import gymnasium as gym
        import tac_manip.tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        p5=smoke.load_module('matched_history_p5',smoke.P5_PATH);p5.OUT=job
        p5.P4_COLLECT=ROOT/'activeforcing_current_probe.py';p4=p5.import_p4_probe(0)
        setup_task_objects(p5.TASK_SUITE,0)
        cfg=parse_env_cfg(smoke.ENV_ID,device='cuda:0',num_envs=1);cfg.episode_length_s=45.
        getattr(cfg.scene,'contact_grasp_'+p4.OBJ_NAME).max_contact_data_count_per_prim=128
        env=gym.make(smoke.ENV_ID,cfg=cfg).unwrapped;env.reset(seed=int(plan['root_seed']))
        original_step=env.step;original_reset=env._reset_idx
        original_compute=env.termination_manager.compute
        def compute_without_premature_success_reset():
            done=original_compute()
            if runtime['mode']=='BRANCH':
                # Keep the raw success term for audit/final place evaluation,
                # but do not reset/break the branch before release and settle.
                done=defer_success_only(env.termination_manager)
            return done
        env.termination_manager.compute=compute_without_premature_success_reset
        def outcome_observation():
            state=base.readout(env,p4)
            cs=env.scene['contact_'+p5.BASKET_NAME+'_'+p4.OBJ_NAME]
            basket_force=float(torch.linalg.vector_norm(cs.data.force_matrix_w.reshape(-1,3)[0]))
            state.update(basket_contact_force_N=basket_force,
                terminations={k:bool(env.termination_manager.get_term(k)[0]) for k in env.termination_manager.active_terms})
            obj=env.scene[p4.OBJ_NAME];basket=env.scene[p5.BASKET_NAME]
            contact=p5.target_object_force_snapshot(env,p4,p4.OBJ_NAME)
            state.update(object_pose_w=jsonable(obj.data.root_state_w[0,:7]),
                object_velocity_w=jsonable(obj.data.root_state_w[0,7:13]),
                basket_pose_w=jsonable(basket.data.root_state_w[0,:7]),
                basket_velocity_w=jsonable(basket.data.root_state_w[0,7:13]),
                finger_object_contact_norms_N=[float(np.linalg.norm(contact[k])) for k in ('F_obj_left_world','F_obj_right_world')])
            return state
        def before_reset(ids):
            if runtime['mode']=='BRANCH' and len(ids):
                terminal.append({'branch_step':runtime['candidate_actions']+1,'state':outcome_observation()})
            return original_reset(ids)
        env._reset_idx=before_reset
        def observed_step(action):
            if runtime['mode']=='DECISION_STATE':raise RuntimeError('Execution before immutable feature extraction')
            if runtime['mode']=='PROBE' and ref_records is not None:
                index=len(records)
                if index>=len(ref_records):raise RuntimeError('Prefix longer than reference')
                np.testing.assert_array_equal(action.detach().cpu().numpy(),np.array(ref_records[index]['action'],np.float32))
            if runtime['mode']=='BRANCH' and terminal:raise RuntimeError('Extra action after terminal/reset')
            terminal_count=len(terminal)
            result=original_step(action)
            if runtime['mode']=='PROBE':
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
            elif runtime['mode']=='BRANCH':
                runtime['candidate_actions']+=1
                ended=len(terminal)>terminal_count
                record=terminal[-1]['state'] if ended else outcome_observation()
                branch_steps.append({'branch_step':runtime['candidate_actions'],'action':jsonable(action),
                    'captured_before_auto_reset':ended,'success_termination_deferred':bool(record['terminations'].get('success',False)),**record})
                if ended:
                    # Do not let auto-reset erase the control-flow stop flag.
                    obs,reward,_,_,info=result;t=record['terminations']
                    result=(obs,reward,torch.tensor([any(v for k,v in t.items() if k!='time_out')],device=env.device),
                            torch.tensor([t.get('time_out',False)],device=env.device),info)
            return result
        env.step=observed_step
        write(job/'STARTED.json',{'mode':mode,'context':plan,'force':force if mode=='branch' else None,
            'physical_prefix_replay':mode!='reference','new_query_used_for_planning':False})
        rows,rec=p4.run_probe_episode(env,seed_idx=int(plan['root_seed']),mu=float(plan['hidden_friction_analysis_only']),
            trial_id=plan['context_id'],dt=smoke.DT,termination_signal=FrictionProbeBudget(records,p4._quat_apply_np))
        raw=[vars(r) for r in rows]
        if any(rec.get(k,0) for k in ('probe_failure','contact_lost_probe','dropped','major_disturbance')):
            raise RuntimeError('Probe admission failed')
        runtime['mode']='DECISION_STATE'
        smoke.write_csv(job/'RAW_PROBE.csv',raw);write(job/'CONTACT_PATCH_READBACK.json',records)
        x=ProbeEvidence().rows(raw,records);np.save(job/'RAW_BELIEF_FEATURES.npy',x)
        frozen=capture(env)
        frozen.update(context_id=plan['context_id'],observation_step=len(raw),candidate_actions_already_executed=0,
            controller_tensor_state=frozen['objects']['arm_action'],full_execution_restore_validated=False)
        torch.save(frozen,job/'DECISION_STATE.pt')
        prefix_errors={}
        if mode!='reference':
            old=torch.load(ref/'DECISION_STATE.pt',map_location='cpu',weights_only=False)
            prefix_errors={k:max_difference(frozen[k],old[k]) for k in ['state','environment','objects','joint_targets','materials']}
            np.testing.assert_array_equal(x,ProbeEvidence().load(ref))
            if any(v!=0. for v in prefix_errors.values()):raise RuntimeError('Exposed decision state differs from reference: '+str(prefix_errors))
            if len(records)!=len(ref_records):raise RuntimeError('Prefix step count differs')
        write(job/'PREFIX_RECONSTRUCTION_GATE.json',{'passed':True,'reference_created':mode=='reference',
            'action_prefix_exact':True,'belief_feature_exact':True,'exposed_decision_errors':prefix_errors,
            'observations_used_for_new_planning':False,'reference_snapshot':str(ref/'DECISION_STATE.pt'),
            'reference_snapshot_sha256':sha256(ref/'DECISION_STATE.pt'),'candidate_actions':0})
        write(job/'PROBE_RESULT.json',{'record':rec,'outward_steps':sum(r['probe_phase']=='probe_out' for r in raw),'total_steps':len(raw)})
        host_env=os.environ.copy();host_env.pop('PYTHONPATH',None);host_env.pop('PYTHONNOUSERSITE',None)
        cmd=[str(smoke.HOST_PY),str(Path(__file__).resolve()),'--features',mode,'--band',band,'--force',str(force),'--repeat',str(repeat),'--pilot-run-name',PILOT_RUN_NAME]
        done=subprocess.run(cmd,cwd=ROOT,env=host_env,capture_output=True,text=True,timeout=120)
        if done.returncode:raise RuntimeError('Feature parity failed: '+done.stdout+done.stderr)
        d0=command_from_probe(records[-1]['action'],float(frozen['objects']['arm_action']['_gripper_abs_cmd'][0,0]),p4.D_CLOSED,p4.D_OPEN)
        write(job/'HANDOFF.json',{'last_probe_action_command':d0,'first_candidate_gripper_command':d0,
            'actual_joint_mean_not_used':float(env.scene['robot'].data.joint_pos[0,-2:].mean()),'command_jump':0.})
        if mode in ('reference','verify'):
            runtime['mode']='VERIFY_HOLD'
            pos=np.array([raw[-1]['eef_'+k] for k in 'xyz'],float)
            obs=env.observation_manager.compute();aa=p4._aa(obs['policy']['eef_pose'][0,3:7].detach().cpu().numpy())
            d=d0;trace=[];actions=[]
            for i in range(20):
                action=p4._make_action(pos,aa,d,4.,env.device)
                _,_,term,trunc,_=env.step(action)
                trace.append(base.readout(env,p4));actions.append(jsonable(action))
                d=float(p4._force_servo(d,p4._f(p4._dbg(env).get('f_sq_meas'),0.),4.))
                if bool(term[0]) or bool(trunc[0]):raise RuntimeError('Terminated during hold20 reconstruction verification')
            write(job/'UNINTERRUPTED_HOLD.json',{'trace':trace,'actions':actions})
            comparison={'passed':True,'reference_created':True}
            if mode=='verify':
                old=json.loads((ref/'UNINTERRUPTED_HOLD.json').read_text());comparison=base.compare_trace(old['trace'],trace)
                comparison['actions_max_difference']=max_difference(old['actions'],actions)
                comparison['passed']=bool(comparison['passed'] and comparison['episode_step_equal'] and comparison['actions_max_difference']==0.)
            write(job/'RESULT.json',{'passed':comparison['passed'],'mode':mode,'comparison':comparison,
                'task_branches':0,'candidate_hold_actions':20,'feasibility_labels':0})
        else:
            branch,adaptation=adapt_placement_runner(p5)
            expected=json.loads((OUT/'BRANCH_CODE_PROVENANCE.json').read_text())
            if adaptation['original_source_sha256']!=expected['original_source_sha256'] or adaptation['adapted_semantic_ast_sha256']!=expected['adapted_semantic_ast_sha256']:
                raise RuntimeError('Branch implementation changed')
            object_z0=base.readout(env,p4)['object_position_m'][2]
            geometry=Geometry(OUT/'ASSET_GEOMETRY.npz')
            placement_path=PlacementPath(env,p4,p4.OBJ_NAME,p5.BASKET_NAME,geometry)
            runtime['mode']='BRANCH'
            logger=p5.StageLogger(job,0,int(plan['root_seed']),dt=smoke.DT,context_id=plan['context_id'],split='PILOT',friction=plan['hidden_friction_analysis_only'])
            native=branch(env,p4,task_id=0,force=float(force),branch_label=f'MATCHED_F{force:g}_R{repeat}',
                context_id=plan['context_id'],split='PILOT',seed=int(plan['root_seed']),
                friction=float(plan['hidden_friction_analysis_only']),dt=smoke.DT,logger=logger,
                telemetry_path=job/'NATIVE_BRANCH_TRACE_DIAGNOSTIC.csv',label_source='CURRENT_PREACTION_REPAIRED_PLACEMENT_BRANCH',handoff_cmd=d0,placement_path=placement_path)
            write(job/'PLACEMENT_TARGETS.json',placement_path.trace)
            if repeat==1:
                prior_path=job_dir(mode,band,force,0)
                prior=json.loads((prior_path/'BRANCH_TRACE.json').read_text())
                comparison=base.compare_trace(prior,branch_steps)
                comparison['all_branch_fields_max_diff']=max_difference(prior,branch_steps)
                comparison['passed']=bool(comparison['passed'] and comparison['episode_step_equal'] and comparison['all_branch_fields_max_diff']==0.)
                write(job/'FULL_BRANCH_REPLAY_GATE.json',comparison)
                if not comparison['passed']:raise RuntimeError('Repaired path repeat not exact')
            torch.save(capture(env),job/'FINAL_SCENE_STATE.pt')
            write(job/'FINAL_GEOMETRY.json',{'task_goal_params':jsonable(env.termination_manager.get_term_cfg('success').params),
                'rigid_objects':{name:{'root_pose_w':jsonable(asset.data.root_state_w[:,:7]),'velocity':jsonable(asset.data.root_state_w[:,7:13])}
                    for name,asset in env.scene.rigid_objects.items()},
                'scene_sensors':list(env.scene.sensors.keys()),'diagnostic_only':True})
            for sensor_name,sensor in env.scene.sensors.items():
                if hasattr(sensor.data,'output') and 'rgb' in sensor.data.output:
                    from PIL import Image
                    rgb=sensor.data.output['rgb'][0].detach().cpu().numpy()
                    if rgb.dtype!=np.uint8:rgb=np.clip(rgb,0,255).astype(np.uint8)
                    Image.fromarray(rgb[...,:3]).save(job/f'FINAL_{sensor_name}.png')
            lift=int(any(s['object_position_m'][2]-object_z0>=.03 for s in branch_steps))
            import csv
            from collections import Counter
            with (job/'NATIVE_BRANCH_TRACE_DIAGNOSTIC.csv').open() as stream:
                native_trace=list(csv.DictReader(stream))
            phase_counts=dict(Counter(row['phase'] for row in native_trace))
            phases_complete=phase_counts==EXPECTED_PHASE_STEPS
            final_goal=bool(branch_steps[-1]['terminations'].get('success',False))
            final_contact=float(branch_steps[-1]['basket_contact_force_N'])
            placement=evaluate_placement(branch_steps,phase_counts,geometry)
            write(job/'GEOMETRIC_PLACEMENT_EVALUATION.json',placement)
            place=placement['place_success']
            dropped=int(any(s['terminations'].get('object_1_dropped',False) for s in branch_steps))
            y=int(lift and place and not dropped)
            write(job/'BRANCH_TRACE.json',branch_steps);write(job/'NATIVE_RESULT_DIAGNOSTIC.json',native)
            result={'passed':True,'mode':mode,'context_id':plan['context_id'],'context':band,'candidate_F':float(force),'repeat':repeat,
                'lift_success':lift,'place_success':place,'dropped':dropped,'full_task_success_y':y,
                'label_source':'CURRENT_PREACTION_REPAIRED_PLACEMENT_GEOMETRIC_FULL_TASK','outcome_source':'per-step state captured BEFORE any auto-reset',
                'native_runner_label':native['full_task_success_y'],'native_label_agrees':native['full_task_success_y']==y,
                'terminal_reset_count':len(terminal),'steps':len(branch_steps),'task_branches':1,
                'full_horizon_required':True,'phase_counts':phase_counts,'all_phases_completed':phases_complete,
                'ever_task_goal_satisfied':any(s['terminations'].get('success',False) for s in branch_steps),
                'first_task_goal_step':next((s['branch_step'] for s in branch_steps if s['terminations'].get('success',False)),None),
                'final_task_goal_satisfied':final_goal,'final_basket_contact_force_N':final_contact,
                'place_success_definition':placement['definition'],'geometric_evaluation':placement,
                'early_success_termination_deferred':True,'goal_geometric_thresholds_changed':True,
                'label_admitted_for_training':False,'restore_execution_verified':False,
                'label_quarantine_reason':'Pilot only; geometric/image audit and same-force repeat gates required before a collection decision',
                'decision_snapshot_sha256':sha256(ref/'DECISION_STATE.pt'),
                'execution_contract_hash':sha256(OUT/'PROTOCOL.json'),
                'reference_feature_sha256':sha256(ref/'FROZEN_PREACTION_FEATURES.npy'),
                'controller_changed':False,'utility_changed':False,'posterior_interface_changed':False}
            write(job/'RESULT.json',result)
        protocol();print('DONE',mode,band,force,repeat,flush=True)
        return 0
    except Exception as exc:
        write(job/'ERROR.json',{'error':repr(exc),'traceback':traceback.format_exc(),'mode':runtime['mode'],
            'candidate_actions':runtime['candidate_actions']})
        if branch_steps:write(job/'PARTIAL_BRANCH_TRACE.json',branch_steps)
        if 'placement_path' in locals():write(job/'PLACEMENT_TARGETS.json',placement_path.trace)
        return 1
    finally:
        if terminal:write(job/'PRE_RESET_TERMINAL_STATES.json',terminal)
        if env is not None:env.close()
        app.close()


def launch_one(spec):
    mode,band,force,repeat=spec;job=job_dir(mode,band,force,repeat);job.mkdir(parents=True,exist_ok=True)
    if (job/'RESULT.json').exists():
        result=json.loads((job/'RESULT.json').read_text())
        if not result['passed']:raise RuntimeError('Previous failed gate: '+str(job))
        return result
    env=os.environ.copy();env.update({'PYTHONNOUSERSITE':'1','PYTHONPATH':os.pathsep.join([str(smoke.WARP_CORE),str(smoke.TABERO),str(smoke.OPENPI)]),
        'OMNI_KIT_ACCEPT_EULA':'YES','ACCEPT_EULA':'Y','TABERO_ROOT':str(smoke.TABERO),'P5S0C_OUT':str(job),
        'HDF5_TRAJ_SOURCE_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/assembled_hdf5'),
        'LIBERO_CONFIG_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/config'),
        'LIBERO_ASSETS_DATA_DIR':str(smoke.TABERO/'benchmarks/datasets/libero/USD')})
    cmd=[str(smoke.ISAAC_PY),'-u',str(Path(__file__).resolve()),'--worker',mode,'--band',band,'--force',str(force),'--repeat',str(repeat),'--pilot-run-name',PILOT_RUN_NAME]
    with (job/'WORKER.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=smoke.TABERO,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(job/'PROCESS.json',{'pid':proc.pid,'command':cmd});print('START',spec,proc.pid,flush=True)
        rc=proc.wait(timeout=1500)
    if rc or (job/'ERROR.json').exists() or not (job/'RESULT.json').exists():raise RuntimeError('Worker failed: '+str(job))
    result=json.loads((job/'RESULT.json').read_text())
    if not result['passed']:raise RuntimeError('Scientific gate failed: '+str(job))
    print('END',spec,flush=True)
    return result


def run(stage):
    protocol()
    for band in ('high','mid','low'):
        if not json.loads((HISTORY_ROOT/'verification'/band/'RESULT.json').read_text())['passed']:
            raise RuntimeError('Prefix verification gate missing')
    if stage=='gate':jobs=[('branch','high',4.,0)]
    else:
        first=job_dir('branch','high',4.,0)
        if not (first/'RESULT.json').exists():raise RuntimeError('Run and review HIGH4 gate first')
        gate=json.loads((first/'RESULT.json').read_text())
        if not gate['passed'] or not gate['all_phases_completed']:raise RuntimeError('Incomplete path gate')
        jobs=[('branch',b,f,0) for b in ('high','mid','low') for f in (3.,4.,5.)]
        jobs += [('branch',b,4.,1) for b in ('high','mid','low')]
    results=[]
    for offset in range(0,len(jobs),2):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results.extend(list(pool.map(launch_one,jobs[offset:offset+2])))
    write(OUT/(stage.upper()+'_COMPLETE.json'),{'jobs':len(jobs),'results':results})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true')
    ap.add_argument('--run',choices=['gate','pilot'])
    ap.add_argument('--worker',choices=['branch']);ap.add_argument('--features',choices=['branch'])
    ap.add_argument('--band',choices=['high','mid','low']);ap.add_argument('--force',type=float,choices=[3.,4.,5.],default=4.)
    ap.add_argument('--repeat',type=int,choices=[0,1],default=0)
    ap.add_argument('--pilot-run-name',default='placement_contract_repair');a=ap.parse_args()
    if a.pilot_run_name:
        if not a.pilot_run_name.replace('_','').isalnum():ap.error('Simple pilot run name required')
        PILOT_RUN_NAME=a.pilot_run_name;OUT=HISTORY_ROOT/PILOT_RUN_NAME
    if a.freeze:freeze()
    elif a.run:run(a.run)
    elif a.worker:raise SystemExit(worker(a.worker,a.band,a.force,a.repeat))
    elif a.features:
        plan=next(c for c in protocol()['contexts'] if c['friction_band'].lower()==a.band)
        host_features(job_dir(a.features,a.band,a.force,a.repeat),plan,a.features=='reference')
    else:ap.error('Explicit bounded mode required')

