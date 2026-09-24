"""Dev-only current probe + real online VLA + candidate existing feasibility."""
import argparse
import csv
import random
import sys
from pathlib import Path
import numpy as np
from common import ROOT, BASE, V5, HERE, read, write, sha

def setup_paths():
    for path in (ROOT, BASE/'current_runtime_recovery_v2_20260905',
                 BASE/'current_runtime_core_snapshot_v2_20260905',
                 BASE/'current_runtime_sensor_repair_v3_candidate_20260905',
                 BASE/'current_multitask58_loader_candidate_20260905',
                 BASE/'current_runtime_branch_execution_v6_20260905'):
        sys.path.insert(0,str(path))

def make_sequence(raw,saved,eef,p4,task,chunk):
    # Same measured state/mask layout; never generate a scripted prefix.
    from common import clean
    last=raw[-1];state=np.zeros(13,np.float32);mask=np.zeros(13,np.float32);mask[:3]=1
    left=np.array([float(last['left_f'+a]) for a in 'xyz'])
    right=np.array([float(last['right_f'+a]) for a in 'xyz'])
    state[6:10]=[abs(left[2]),abs(right[2]),np.linalg.norm(left[:2]),np.linalg.norm(right[:2])];mask[6:10]=1
    joints=np.asarray(clean(saved['state']['articulation']['robot']['joint_position'])).reshape(-1)
    if len(joints)!=9 or not np.isclose(joints[-2],float(last['gripper_opening']),rtol=0,atol=1e-6):
        raise RuntimeError('Probe/snapshot joint mismatch')
    state[11:13]=[joints[-2],-joints[-1]];mask[11:13]=1
    velocity=np.asarray(clean(saved['state']['rigid_object'][last['object_id']]['root_velocity'])).reshape(-1)[:3]
    state[3:6]=velocity;state[10]=np.linalg.norm(velocity[:2]);mask[3:6]=1;mask[10]=1
    x=np.zeros((8,64),np.float32)
    x[:,6+(0,1,5,6).index(task)]=1
    x[:,12:25]=state;x[:,25:38]=mask;x[:,38:51]=state;x[:,51:64]=mask
    commands=np.asarray(chunk[:8,:3],dtype=np.float64)
    x[:,:3]=commands-commands[0]
    x[:,3:6]=np.vstack([np.zeros((1,3)),np.diff(commands,axis=0)])
    # Seven legacy scripted phase channels have been removed from both input
    # and GRU weights. No implicit phase or missing-phase placeholder remains.
    return x

def run(args):
    out=Path(args.out);job=Path(args.job);plan=read(out/'DEV_PLAN.json')['contexts'][args.context]
    manifest=read(out/'CANDIDATE_RUNTIME_MANIFEST.json')
    for p,h in manifest['source_hashes'].items():
        if sha(p)!=h:raise RuntimeError('Candidate frozen source changed: '+p)
    write(job/'SOURCE_HASHES_BEFORE.json',manifest['source_hashes'])
    setup_paths()
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,num_envs=1).app
    try:
        import torch
        import current_runtime_core as core
        import measurement_hooks
        import continuous_belief
        from activeforcing_execution_snapshot import capture
        from activeforcing_command_handoff import command_from_probe
        from branch_execution import identical,SNAPSHOT_GROUPS
        from phase_free_feasibility import PhaseFreeFeasibility
        from online_ablation_feasibility import AblationFeasibility, METHODS
        import importlib.util
        spec=importlib.util.spec_from_file_location('audited_online_vla_runtime',HERE/'runtime.py')
        runtime=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runtime;spec.loader.exec_module(runtime)
        belief=continuous_belief.ContinuousBelief(V5/'BELIEF_MANIFEST.json',manifest_sha256=sha(V5/'BELIEF_MANIFEST.json'))
        feasibility=AblationFeasibility(args.method) if args.method in METHODS else PhaseFreeFeasibility()
        ref=out/'references'/plan['id']
        def decision(env,p4,p5,liveplan,livejob):
            with (job/'RAW_PROBE.csv').open() as f:raw=list(csv.DictReader(f))
            rb=read(job/'CONTACT_PATCH_READBACK.json')
            saved=torch.load(job/'DECISION_STATE.pt',map_location='cpu',weights_only=False)
            # Before any VLA observation rendering or inference, prove exact
            # decision state and probe prefix shared across all methods.
            live=capture(env)
            eq={k:identical(saved[k],live[k]) for k in SNAPSHOT_GROUPS}
            if args.method!='REFERENCE':
                reference=torch.load(ref/'DECISION_STATE.pt',map_location='cpu',weights_only=False)
                eq.update({'reference_'+k:identical(saved[k],reference[k]) for k in SNAPSHOT_GROUPS})
                eq['probe_rows']=raw==list(csv.DictReader((ref/'RAW_PROBE.csv').open()))
                eq['probe_records']=identical(rb,read(ref/'CONTACT_PATCH_READBACK.json'))
            write(job/'POSTPROBE_EQUALITY.json',{'passed':all(eq.values()),'checks':eq,'candidate_actions_executed':0})
            if not all(eq.values()):raise RuntimeError('Common post-probe state failed')
            py=random.getstate();npy=np.random.get_state();tcpu=torch.get_rng_state();tcuda=torch.cuda.get_rng_state_all()
            pred=belief.rows(raw,rb,runtime_manifest_sha256=sha(V5/'CURRENT_ACTIVEFORCING_RUNTIME_MANIFEST.json'))
            posterior={k:pred[k] for k in ('interface','feature_schema_id','member_means','member_log_sigmas','posterior_moments','integration_qa')}
            posterior.update(integration_nodes=pred['integration_nodes'].tolist(),integration_weights=pred['integration_weights'].tolist(),candidate_actions_executed=0,hidden_friction_used=False)
            write(job/'PREACTION_POSTERIOR.json',posterior)
            random.setstate(py);np.random.set_state(npy);torch.set_rng_state(tcpu);torch.cuda.set_rng_state_all(tcuda)
            if args.method=='REFERENCE':
                runtime.save_reference_observation(env,plan,job)
                return {'reference_saved':True,'candidate_actions_executed':0}
            if posterior!=read(ref/'PREACTION_POSTERIOR.json'):raise RuntimeError('Posterior parity failed')
            prepared=runtime.first_chunk(env,plan,job,args.port,ref)
            chunk=prepared[3]
            x=make_sequence(raw,saved,np.asarray(rb[-1]['eef_pose'],np.float32).reshape(-1),p4,plan['task'],chunk)
            with (job/'PREACTION_SEQUENCE.npy').open('xb') as f:np.save(f,x)
            shared=out/'initial_chunk_identity';shared.mkdir(exist_ok=True)
            canonical=shared/(plan['id']+'.json')
            identity={'payload_sha256':prepared[4]['observation_sha256'],'noise_sha256':prepared[4]['noise_sha256'],
                'online_chunk_sha256':prepared[4]['action_sha256'],'preaction_sequence_sha256':sha(job/'PREACTION_SEQUENCE.npy')}
            if canonical.exists():
                if read(canonical)!=identity:raise RuntimeError('Methods do not share identical online initial chunk / decision X')
            else:write(canonical,identity)
            write(job/'INITIAL_ONLINE_CHUNK_IDENTITY.json',identity)
            decision=feasibility.select(x,posterior)
            force=decision['selected_force_N'] if args.method in ('ACTIVEFORCING',*METHODS) else float(args.method.removeprefix('FIXED_'))
            decision.update(executed_force_N=force,method=args.method,feasibility_sequence_source='ONLINE_VLA_ACTION_CHUNK',
                request_id=prepared[4]['request_id'],phase_channels='REMOVED; exact frozen model column deletion',
                scripted_prefix_used=False,transfer_valid='NOT_YET_TESTED')
            write(job/'PLANNER_DECISION.json',decision)
            d0=command_from_probe(rb[-1]['action'],float(saved['objects']['arm_action']['_gripper_abs_cmd'][0,0]),p4.D_CLOSED,p4.D_OPEN)
            write(job/'HANDOFF.json',{'command':d0,'source':'last physical probe action; not measured aperture','same_probe':True})
            return runtime.rollout(env,p4,p5,plan,job,force,d0,prepared)
        rc=core.run_probe(plan,job,measurement_hooks=measurement_hooks,on_decision=decision,
            runtime_manifest_sha256=sha(V5/'CURRENT_ACTIVEFORCING_RUNTIME_MANIFEST.json'))
        result=read(job/'RESULT.json') if (job/'RESULT.json').exists() else {}
        success=rc==0 and result.get('probe_qualified',False)
        if args.method!='REFERENCE':
            b=read(job/'BRANCH_RESULT.json') if (job/'BRANCH_RESULT.json').exists() else {}
            success=success and b.get('online_vla_verified',False) and b.get('outcome',{}).get('label_valid',False)
        write(job/'WORKER_COMPLETION.json',{'logical_success':bool(success),'method':args.method})
        write(job/'SOURCE_HASHES_AFTER.json',{p:sha(p) for p in manifest['source_hashes']})
        return 0 if success else 2
    finally:app.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--job',required=True)
    p.add_argument('--context',type=int,required=True);p.add_argument('--method',required=True)
    p.add_argument('--port',type=int,default=18885)
    raise SystemExit(run(p.parse_args()))
