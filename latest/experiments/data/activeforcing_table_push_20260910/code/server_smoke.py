"""Real official-policy inference check using native task-5 observation; no env.step."""
import hashlib,json,pathlib,torch,numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
OUT=pathlib.Path('/media/volume/data/exouser/activeforcing_table_push_20260910')
def aa(q):
 q=np.asarray(q).copy();q[3]=np.clip(q[3],-1,1);d=np.sqrt(1-q[3]*q[3]);return np.zeros(3) if d<1e-9 else q[:3]*2*np.arccos(q[3])/d
old=torch.load;torch.load=lambda *a,**kw:old(*a,weights_only=False,**kw)
suite=benchmark.get_benchmark_dict()['libero_goal']();task=suite.get_task(5);env=OffScreenRenderEnv(bddl_file_name=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,camera_heights=256,camera_widths=256);env.seed(7);env.reset();o=env.set_init_state(suite.get_task_init_states(5)[0])
img=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(o['agentview_image'][::-1,::-1]),224,224));w=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(o['robot0_eye_in_hand_image'][::-1,::-1]),224,224))
a=np.asarray(WebsocketClientPolicy('127.0.0.1',8011).infer({'observation/image':img,'observation/wrist_image':w,'observation/state':np.r_[o['robot0_eef_pos'],aa(o['robot0_eef_quat']),o['robot0_gripper_qpos']],'prompt':task.language,'_activeforcing_episode_start_seed':1000})['actions'])
(OUT/'server'/'INFERENCE_SMOKE.json').write_text(json.dumps({'real_inference':True,'task_id':5,'language':task.language,'episode_seed':1000,'action_shape':list(a.shape),'action_sha256':hashlib.sha256(a.tobytes()).hexdigest()})+'\n');env.close()
