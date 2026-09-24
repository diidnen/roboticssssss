"""One online frozen-pi0 Gate-A rollout; no action replay outcome evidence."""
import collections,json,os,pathlib,sys,time,tempfile,math
import cv2,numpy as np,torch,mujoco
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
sys.path.insert(0,str(pathlib.Path(__file__).parent));from push_force import PushForceController
OUT=pathlib.Path('/media/volume/data/exouser/activeforcing_table_push_20260910')
ROBOT={'gripper0_hand_collision','gripper0_finger1_collision','gripper0_finger1_pad_collision','gripper0_finger2_collision','gripper0_finger2_pad_collision'}
def aa(q):
 q=np.asarray(q).copy();q[3]=np.clip(q[3],-1,1);d=np.sqrt(1-q[3]*q[3]);return np.zeros(3) if d<1e-9 else q[:3]*2*np.arccos(q[3])/d
def main():
 idx,mu,target,seed=map(float,sys.argv[1:]);idx=int(idx);seed=int(seed); suffix=os.environ.get('ACTIVEFORCING_RUN_SUFFIX',''); label=f'gateA_init{idx}_mu{mu:g}_F{target:g}_seed{seed}{suffix}';run=OUT/'rollouts'/label;run.mkdir()
 old=torch.load;torch.load=lambda *a,**kw:old(*a,weights_only=False,**kw);suite=benchmark.get_benchmark_dict()['libero_goal']();task=suite.get_task(5);init=suite.get_task_init_states(5)[idx]
 env=OffScreenRenderEnv(bddl_file_name=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,camera_heights=256,camera_widths=256);env.seed(7);env.reset();obs=env.set_init_state(init);m,d=env.sim.model,env.sim.data
 table=next(i for i in range(m.ngeom) if m.geom_id2name(i)=='table_collision');plate={i for i in range(m.ngeom) if str(m.geom_id2name(i)).startswith('plate_1_')};before=m.geom_friction[table].copy();m.geom_friction[table]=[mu,.005,.0001];read=m.geom_friction[table].copy()
 ctl=PushForceController(target);plan=collections.deque();client=WebsocketClientPolicy('127.0.0.1',8011);trace=[];direction=None;stable=0;done=False;writer=cv2.VideoWriter(str(run/'agentview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(224,224))
 try:
  for step in range(310):
   img=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),224,224)); wrist=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),224,224));writer.write(cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
   if step<10: nominal=np.array([0]*6+[-1.]);source='stabilize'
   else:
    if not plan:
     req={'observation/image':img,'observation/wrist_image':wrist,'observation/state':np.r_[obs['robot0_eef_pos'],aa(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']],'prompt':task.language}
     if step==10:req['_activeforcing_episode_start_seed']=seed
     chunk=np.asarray(client.infer(req)['actions']);plan.extend(chunk[:5])
    nominal=np.asarray(plan.popleft(),float);source='online_pi0'
   contacts=[]; total=np.zeros(3)
   for ci in range(d.ncon):
    c=d.contact[ci];n1,n2=m.geom_id2name(c.geom1),m.geom_id2name(c.geom2)
    if ((n1 in ROBOT and c.geom2 in plate) or (n2 in ROBOT and c.geom1 in plate)):
     w=np.zeros(6);mujoco.mj_contactForce(m._model,d._data,ci,w);fw=c.frame.reshape(3,3).T@w[:3]; fp=fw if c.geom2 in plate else -fw;total+=fp;contacts.append({'i':ci,'geom1':n1,'geom2':n2,'wrench_contact':w.tolist(),'force_world_on_plate':fp.tolist()})
   stable=stable+1 if contacts else 0
   if direction is None and stable>=3:
    v=np.r_[nominal[:2],0.]; direction=v/max(np.linalg.norm(v[:2]),1e-9)
   active=direction is not None and stable>=3
   executed,clog=ctl.apply(nominal,direction if direction is not None else [1,0,0],float(total@(direction if direction is not None else [0,0,0])),active)
   obs,reward,done,info=env.step(executed.tolist())
   trace.append({'step':step,'source':source,'nominal_action':nominal.tolist(),'residual_action':(executed-nominal).tolist(),'executed_action':executed.tolist(),'contact_count':len(contacts),'raw_contacts':contacts,'force_sum_world_on_plate':total.tolist(),'push_direction':None if direction is None else direction.tolist(),'controller':clog,'reward':float(reward),'done':bool(done)})
   if done:break
  rec={'gate':'A','run':label,'init':idx,'mu_requested':mu,'table_friction_before':before.tolist(),'table_friction_readback':read.tolist(),'target_push_n':target,'policy_seed':seed,'online_policy_only':True,'success':bool(done),'final_native_check_success':bool(env.check_success()),'steps':len(trace),'direction':None if direction is None else direction.tolist(),'error':None}
 except Exception as e:rec={'gate':'A','run':label,'error':repr(e),'success':False,'infrastructure_failed':True}
 finally:writer.release();env.close()
 (run/'telemetry.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trace));(run/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
if __name__=='__main__':main()
