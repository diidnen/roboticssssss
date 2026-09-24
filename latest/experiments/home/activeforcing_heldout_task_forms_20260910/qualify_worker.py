"""Fixed-5-only native-reset competence screen, no AF or belief model imports."""
from pathlib import Path
import argparse,ast,json,os,sys,traceback,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import numpy as np
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True);parser.add_argument('--task',type=int,required=True);parser.add_argument('--root',type=int,required=True);parser.add_argument('--mu',type=float,required=True);a=parser.parse_args()
job=P/'qualification'/f'{a.suite}_{a.task}_r{a.root}_mu{a.mu:.2f}';job.mkdir(parents=True,exist_ok=False)
def write(name,x): (job/name).write_text(json.dumps(x,indent=2,default=str)+'\n')
TAB=Path('/home/exouser/Tabero');BASE=Path('/home/exouser/FORTE/analysis/results')
sys.path[:0]=[str(TAB),str(TAB/'benchmarks/openpi/openpi-client/src'),'/home/exouser/FORTE',str(BASE/'current_runtime_sensor_repair_v3_candidate_20260905'),'/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT']
os.environ.update(TASK_SUITE=a.suite,TASK_ID=str(a.task),LIBERO_CONFIG_DIR=str(TAB/'benchmarks/datasets/libero/config'),LIBERO_ASSETS_DATA_DIR=str(TAB/'benchmarks/datasets/libero/USD'))
task=json.loads((TAB/'benchmarks/datasets/libero/config'/f'{a.suite}.json').read_text())['tasks'][a.task]
assert task['task_id']==a.task and len(task['obj_of_interest'])==1 and a.root in [5100,6100]
obj=task['obj_of_interest'][0];plan={'id':job.name,'task':1000+a.task,'object':obj,'root':a.root,'mu':a.mu}
result=dict(plan=plan,instruction=task['language_instruction'],method='FIXED_5',AF_calls=0,role='NATIVE_RESET_VLA_COMPETENCE_SCREEN',valid_outcome=False)
write('STARTED.json',result)
env=None;app=None
try:
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,num_envs=1).app
    import gymnasium as gym,torch,tac_manip.tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    import runtime,common,measurement_hooks
    from arbitration import Arbitration
    common.INSTRUCTIONS[plan['task']]=task['language_instruction']
    cfg=parse_env_cfg('Isaac-Libero-Franka-Hybrid-Tactile-v0',device='cuda:0',num_envs=1)
    cfg.episode_length_s=60
    write('SENSOR_CONFIG.json',measurement_hooks.configure(cfg,plan))
    env=gym.make('Isaac-Libero-Franka-Hybrid-Tactile-v0',cfg=cfg).unwrapped;env.reset(seed=a.root)
    write('SENSOR_BINDING.json',measurement_hooks.bind(env,plan))
    asset=env.scene[obj]
    mats=asset.root_physx_view.get_material_properties().clone();mats[...,:2]=a.mu
    asset.root_physx_view.set_material_properties(mats,torch.tensor([0],dtype=torch.int32))
    actual=asset.root_physx_view.get_material_properties().cpu().numpy()
    assert np.allclose(actual[...,:2],a.mu,atol=1e-6)
    write('PHYSICS.json',{'object_friction':actual.tolist(),'native_mass_unchanged':asset.root_physx_view.get_masses().cpu().tolist(),'mass_intervention':False})
    # Extract exact small sensor/servo functions without importing the collector.
    from types import SimpleNamespace
    from typing import Any
    ns={'np':np,'SERVO_DEADBAND':.4,'SERVO_STEP':.0006,'D_CLOSED':0.,'D_OPEN':.04,'Any':Any}
    for path,names in [(Path('/home/exouser/FORTE/activeforcing_current_probe.py'),{'_force_servo','_quat_apply_np','_unit','_f','_dbg'}),(TAB/'analysis/p5s0c_paired_boundary_probe_value.py',{'target_object_force_snapshot'})]:
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    p4=SimpleNamespace(**ns)
    client=runtime.OnlineClient(job,plan,18891);build,Buffer,digest=runtime.observation_builder();buffer=Buffer()
    arb=Arbitration(5.,.04);traces=[];contacts=0;grasp_step=None;release_seen=False;regrasp=False;terminal_requested=False
    original_compute=env.termination_manager.compute;original_reset=env._reset_idx
    def compute():
        original_compute()
        failures=torch.zeros_like(env.termination_manager.terminated)
        for name in env.termination_manager.active_terms:
            if name!='success' and not env.termination_manager.get_term_cfg(name).time_out:failures.logical_or_(env.termination_manager.get_term(name))
        env.termination_manager._terminated_buf.copy_(failures);return env.termination_manager.dones
    def reset(ids):
        global terminal_requested
        if len(ids):terminal_requested=True
    env.termination_manager.compute=compute;env._reset_idx=reset
    from PIL import Image
    for step in range(1,551):
        if (step-1)%10==0:chunk,proof=runtime.request_chunk(env,plan,client,build,buffer,step)
        action,arbitration=arb.action(chunk[(step-1)%10])
        env.step(torch.from_numpy(action).reshape(1,13).to(env.device))
        contact=ns['target_object_force_snapshot'](env,p4,obj)
        bilateral=bool(contact['target_object_bilateral_contact'] and contact['contact_opposition'])
        contacts=contacts+1 if bilateral else 0
        if grasp_step is None and contacts>=4:grasp_step=step
        if grasp_step is not None and arbitration['vla_release_intent']:release_seen=True
        if release_seen and bilateral and not arbitration['vla_release_intent']:regrasp=True
        dbg=ns['_dbg'](env);arb.feedback(ns['_f'](dbg.get('f_sq_meas'),0.),arbitration['vla_release_intent'],ns['_force_servo'])
        rec=dict(step=step,**arbitration,**contact,native_success=bool(env.termination_manager.get_term('success')[0]),
            object_pose=asset.data.root_state_w[0,:7].cpu().tolist(),eef_pose=env.observation_manager.compute()['policy']['eef_pose'][0].cpu().tolist(),
            grasp_step=grasp_step,regrasp=regrasp,terminal_requested=terminal_requested,request_id=proof['request_id'])
        traces.append(rec)
        with (job/'TRACE.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        if step==1 or step%25==0 or terminal_requested:
            Image.fromarray(env.scene['agentview_cam'].data.output['rgb'][0,:,:,:3].cpu().numpy()).save(job/f'FRAME_{step:04d}.png')
        if terminal_requested or (grasp_step is None and step>=200) or (grasp_step is not None and step>=grasp_step+350):break
    env.termination_manager.compute=original_compute;env._reset_idx=original_reset
    last=traces[-20:]
    semantic=len(last)==20 and all(x['native_success'] for x in last)
    released=len(last)==20 and all(x['vla_release_intent'] and max(np.linalg.norm(x['F_obj_left_world']),np.linalg.norm(x['F_obj_right_world']))<=.05 for x in last)
    success=grasp_step is not None and semantic and released and not regrasp
    result.update(valid_outcome=True,steps=len(traces),rpc_count=client.calls,established_grasp=grasp_step is not None,grasp_step=grasp_step,
        native_semantic_terminal_success=semantic,release_success=released,regrasp=regrasp,full_screen_success=success,
        failure='NONE' if success else 'VLA_PREGRASP_FAILURE' if grasp_step is None else 'REGRASP_OUTSIDE_SCOPE' if regrasp else 'TERMINAL_GEOMETRIC_OR_VLA_FAILURE' if not semantic else 'RELEASE_FAILURE',
        form_evaluator_admission='PENDING_TASK_SPECIFIC_VALIDATION',checkpoint_sha256=client.metadata['checkpoint_sha256'])
except Exception as e:result.update(error=repr(e),traceback=traceback.format_exc(),failure='INFRASTRUCTURE_OR_INTERFACE_FAILURE')
finally:
    write('QUALIFICATION_RESULT.json',result)
    # Isaac 5.1 native scene teardown can abort after a complete result. Exit
    # this isolated worker without callback teardown; OS releases its resources.
    sys.stdout.flush();sys.stderr.flush();os._exit(0 if result['valid_outcome'] else 2)
