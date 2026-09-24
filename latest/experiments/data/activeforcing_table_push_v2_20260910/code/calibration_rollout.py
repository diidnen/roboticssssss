"""One A2.1 calibration episode. Fixed residual value, online pi0, no task outcome claim."""
import collections,hashlib,json,pathlib,sys,time
import cv2,numpy as np,torch,mujoco
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from v2_interface import direction_from_fresh_chunk,ResidualRamp
OUT=pathlib.Path('/media/volume/data/exouser/activeforcing_table_push_v2_20260910');ROBOT={'gripper0_hand_collision','gripper0_finger1_collision','gripper0_finger1_pad_collision','gripper0_finger2_collision','gripper0_finger2_pad_collision'}
def aa(q):
 q=np.asarray(q).copy();q[3]=np.clip(q[3],-1,1);n=np.sqrt(1-q[3]*q[3]);return np.zeros(3) if n<1e-9 else q[:3]*2*np.arccos(q[3])/n
def main():
 residual=float(sys.argv[1]); seed=1000;label=f'calibration_root0_mu1.2_residual_{residual:+.2f}'.replace('+','plus').replace('-','minus');run=OUT/'calibration'/label;run.mkdir();assert not (run/'receipt.json').exists()
 old=torch.load;torch.load=lambda *a,**kw:old(*a,weights_only=False,**kw);suite=benchmark.get_benchmark_dict()['libero_goal']();task=suite.get_task(5);init=suite.get_task_init_states(5)[0]
 env=OffScreenRenderEnv(bddl_file_name=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,camera_heights=256,camera_widths=256);env.seed(7);env.reset();obs=env.set_init_state(init);m,d=env.sim.model,env.sim.data;table=next(i for i in range(m.ngeom) if m.geom_id2name(i)=='table_collision');before=m.geom_friction[table].copy();m.geom_friction[table]=[1.2,.005,.0001];read=m.geom_friction[table].copy()
 client=WebsocketClientPolicy('127.0.0.1',8012);plan=collections.deque();stable=0;direction=None;direction_record=None;ramp=ResidualRamp(residual);trace=[];done=False
 try:
  for step in range(180):
   img=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),224,224));wrist=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),224,224))
   def infer(mark=False):
    req={'observation/image':img,'observation/wrist_image':wrist,'observation/state':np.r_[obs['robot0_eef_pos'],aa(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']],'prompt':task.language}
    if mark:req['_activeforcing_episode_start_seed']=seed
    return np.asarray(client.infer(req)['actions'],float)
   if step<10: nominal=np.r_[np.zeros(6),-1.];source='stabilize'
   else:
    if not plan: plan.extend(infer(step==10)[:5])
    nominal=np.asarray(plan.popleft());source='online_pi0'
   contacts=[];total=np.zeros(3)
   for ci in range(d.ncon):
    c=d.contact[ci];n1,n2=m.geom_id2name(c.geom1),m.geom_id2name(c.geom2)
    if ((n1 in ROBOT and c.geom2 in {i for i in range(m.ngeom) if str(m.geom_id2name(i)).startswith('plate_1_')}) or (n2 in ROBOT and c.geom1 in {i for i in range(m.ngeom) if str(m.geom_id2name(i)).startswith('plate_1_')})):
     w=np.zeros(6);mujoco.mj_contactForce(m._model,d._data,ci,w);world=c.frame.reshape(3,3).T@w[:3];onplate=world if c.geom2 in {i for i in range(m.ngeom) if str(m.geom_id2name(i)).startswith('plate_1_')} else -world;total+=onplate;contacts.append({'contact_index':ci,'geom1':n1,'geom2':n2,'wrench_contact':w.tolist(),'frame_world_axes_rows':c.frame.reshape(3,3).tolist(),'sign':'world force is on plate','force_world_on_plate':onplate.tolist()})
   stable=stable+1 if contacts else 0
   if direction is None and stable>=3:
    plan.clear();fresh=infer(False);direction_record=direction_from_fresh_chunk(fresh);direction=np.asarray(direction_record['direction_world_xy']);plan.extend(fresh[:5]);nominal=np.asarray(plan.popleft());source='fresh_pi0_after_third_contact'
   active=direction is not None and stable>=3
   executed,clog=ramp.apply(nominal,direction if direction is not None else [1,0],active)
   obs,reward,done,info=env.step(executed.tolist());projection=None if direction is None else float(total[:2]@direction)
   trace.append({'step':step,'source':source,'nominal_action':nominal.tolist(),'executed_action':executed.tolist(),'residual_action':(executed-nominal).tolist(),'controller':clog,'contact_count':len(contacts),'raw_contacts':contacts,'force_sum_world_on_plate':total.tolist(),'aligned_projection_n':projection,'stable_contact_steps':stable,'active':active,'reward':float(reward),'done':bool(done)})
   if direction is not None and sum(x['active'] for x in trace)>=35:break
  receipt={'gate':'A2.1','run':label,'root':0,'mu_requested':1.2,'table_friction_before':before.tolist(),'table_friction_readback':read.tolist(),'policy_seed':seed,'residual_normalized':residual,'residual_is_not_newtons':True,'direction_record':direction_record,'steps':len(trace),'steady_samples':sum(x['active'] for x in trace),'native_success_ignored_for_calibration':bool(env.check_success()),'error':None}
 except Exception as e:receipt={'gate':'A2.1','run':label,'error':repr(e),'infrastructure_failed':True}
 finally:env.close()
 (run/'telemetry.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trace));(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
