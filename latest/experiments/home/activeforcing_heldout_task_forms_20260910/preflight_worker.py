"""Native scene/observation compatibility only. No VLA/AF inference or task actions."""
from pathlib import Path
import argparse,json,os,sys,traceback
p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--task',type=int,required=True);a=p.parse_args()
OUT=Path(__file__).resolve().parent
job=OUT/os.environ.get('AF_PREFLIGHT_DIR','preflight')/f'{a.suite}_{a.task}';job.mkdir(parents=True,exist_ok=False)
TAB=Path('/home/exouser/Tabero')
os.environ.update(TASK_SUITE=a.suite,TASK_ID=str(a.task),LIBERO_CONFIG_DIR=str(TAB/'benchmarks/datasets/libero/config'),LIBERO_ASSETS_DATA_DIR=str(TAB/'benchmarks/datasets/libero/USD'))
sys.path[:0]=[str(TAB),str(TAB/'benchmarks/openpi/openpi-client/src'),'/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT']
def write(x): (job/'PREFLIGHT_RESULT.json').write_text(json.dumps(x,indent=2,default=str)+'\n')
receipt=dict(suite=a.suite,task=a.task,root=5100,role='NATIVE_COMPATIBILITY_ONLY',new_task_AF_calls=0,VLA_calls=0,task_actions=0)
app=None;env=None
try:
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,num_envs=1).app
    import gymnasium as gym
    import tac_manip.tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    import runtime
    cfg=parse_env_cfg('Isaac-Libero-Franka-Hybrid-Tactile-v0',device='cuda:0',num_envs=1)
    receipt['config_loaded']=True
    receipt['actions']=str(cfg.actions)
    env=gym.make('Isaac-Libero-Franka-Hybrid-Tactile-v0',cfg=cfg).unwrapped
    obs,_=env.reset(seed=5100)
    receipt['scene_reset']=True
    build,Buffer,digest=runtime.observation_builder()
    task=json.loads((TAB/'benchmarks/datasets/libero/config'/f'{a.suite}.json').read_text())['tasks'][a.task]
    assert task['task_id']==a.task
    payload=build(env,obs,task['language_instruction'],Buffer())
    receipt['payload_shapes']={k:list(v.shape) for k,v in payload.items() if isinstance(k,str) and hasattr(v,'shape')}
    receipt['required_observations_present']=all(k in payload for k in ['state','image','wrist_image','tactile_marker_motion'])
    receipt['native_terminal_terms']=env.termination_manager.active_terms
    receipt['native_goal_config']=task['goals']
    receipt['scene_rigid_objects']=list(env.scene.rigid_objects)
    receipt['scene_articulations']=list(env.scene.articulations)
    receipt['success_at_reset']=bool(env.termination_manager.get_term('success')[0])
    from PIL import Image
    Image.fromarray(env.scene['agentview_cam'].data.output['rgb'][0,:,:,:3].cpu().numpy()).save(job/'NATIVE_RESET_FRAME.png')
    receipt['status']='SCENE_AND_OBSERVATION_PREFLIGHT_PASSED_NOT_VLA_QUALIFIED'
except Exception as e:
    receipt.update(status='INFRASTRUCTURE_OR_INTERFACE_PREFLIGHT_FAILURE',error=repr(e),traceback=traceback.format_exc())
finally:
    write(receipt)
    if env is not None:env.close()
    if app is not None:app.close()
