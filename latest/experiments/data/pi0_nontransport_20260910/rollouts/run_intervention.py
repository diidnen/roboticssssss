"""One sequential online frozen-pi0 rollout with exactly one runtime MuJoCo edit."""
import collections, datetime, hashlib, json, math, os, pathlib, sys, tempfile, time
import cv2, numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

OUT=pathlib.Path('/media/volume/data/exouser/pi0_nontransport_20260910/rollouts')
CHECKPOINT='/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero'
TREE='92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103'
SPEC={0:('geom','wooden_cabinet_1_g22',[.2,.3,.1]),5:('geom','table_collision',[2.,.005,.0001]),7:('damping','flat_stove_1_button',5.)}
def axis(q):
 q=np.asarray(q).copy();q[3]=np.clip(q[3],-1,1);d=np.sqrt(1-q[3]*q[3]);return np.zeros(3) if math.isclose(d,0) else q[:3]*2*math.acos(q[3])/d
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def atomic(path,obj):
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.tmp_',suffix='.json'); os.close(fd)
 pathlib.Path(tmp).write_text(json.dumps(obj,indent=2)+'\n'); os.replace(tmp,path)
def geom_id(m,n):
 for i in range(m.ngeom):
  if m.geom_id2name(i)==n:return i
 raise KeyError(n)
def joint_id(m,n):
 for i in range(m.njnt):
  if m.joint_id2name(i)==n:return i
 raise KeyError(n)
def main():
 tid=int(sys.argv[1]); now=datetime.datetime.now(datetime.timezone.utc)
 if now >= datetime.datetime(2026,9,10,15,tzinfo=datetime.timezone.utc): raise RuntimeError('launch deadline passed')
 orig=torch.load; torch.load=lambda *a,**kw: orig(*a,weights_only=False,**kw)
 suite=benchmark.get_benchmark_dict()['libero_goal'](); task=suite.get_task(tid); init=suite.get_task_init_states(tid)[0]
 bddl=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file; env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256);env.seed(7)
 kind,target,after=SPEC[tid]; run=OUT/f'task{tid}_init0_{kind}_intervention'; run.mkdir(exist_ok=False)
 r={'started_utc':now.strftime('%Y-%m-%dT%H:%M:%SZ'),'policy_config':'pi0_libero','checkpoint_path':CHECKPOINT,'checkpoint_tree_sha256':TREE,'task_id':tid,'language':task.language,'init_state':{'index':0,'sha256':digest(init),'shape':list(init.shape),'dtype':str(init.dtype)},'baseline_receipt':f'/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/rollouts/task{tid}_init0_native_pi0_libero/receipt.json','action_source':'online websocket frozen pi0_libero only after stabilization','intervention':{},'actions':[],'inference_receipts':[],'success_rule':'native env.step done flag; final env.check_success()'}
 writer=None
 try:
  env.reset();obs=env.set_init_state(init);m=env.sim.model
  if kind=='geom':
   i=geom_id(m,target); before=m.geom_friction[i].copy().tolist();m.geom_friction[i]=after; readback=m.geom_friction[i].copy().tolist(); prop=f'model.geom_friction[{target}]'
  else:
   j=joint_id(m,target);i=int(m.jnt_dofadr[j]);before=float(m.dof_damping[i]);m.dof_damping[i]=after;readback=float(m.dof_damping[i]);prop=f'model.dof_damping[{target}]'
  r['intervention']={'property':prop,'before':before,'requested_after':after,'readback_after':readback,'one_runtime_property_only':True}
  writer=cv2.VideoWriter(str(run/'agentview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(224,224)); client=WebsocketClientPolicy('127.0.0.1',8010); plan=collections.deque();done=False
  for step in range(310):
   img=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),224,224)); wrist=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),224,224));writer.write(cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
   if step<10: a=np.array([0]*6+[-1.],float);src='stabilization_dummy'
   else:
    if not plan:
     elem={'observation/image':img,'observation/wrist_image':wrist,'observation/state':np.concatenate((obs['robot0_eef_pos'],axis(obs['robot0_eef_quat']),obs['robot0_gripper_qpos'])),'prompt':str(task.language)};t=time.monotonic();ans=client.infer(elem);elapsed=time.monotonic()-t;chunk=np.asarray(ans['actions']);
     if len(chunk)<5:raise RuntimeError('policy action chunk shorter than frozen cadence')
     plan.extend(chunk[:5]);r['inference_receipts'].append({'env_step':step,'latency_seconds':elapsed,'chunk_shape':list(chunk.shape)})
    a=np.asarray(plan.popleft(),float);src='online_pi0_libero'
   obs,reward,done,info=env.step(a.tolist());r['actions'].append({'env_step':step,'source':src,'action':a.tolist(),'reward':float(reward),'done':bool(done)})
   if done:break
  r.update({'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'steps_executed':len(r['actions']),'success':bool(done),'final_native_check_success':bool(env.check_success())})
 except Exception as e:r.update({'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'error':repr(e),'success':False})
 finally:
  if writer:writer.release()
  env.close()
 np.save(run/'actions.npy',np.array([a['action'] for a in r['actions']],float));atomic(run/'receipt.json',r);print(json.dumps({k:r.get(k) for k in ('success','steps_executed','error','final_native_check_success','intervention')},indent=2))
 return 0 if 'error' not in r else 1
if __name__=='__main__':sys.exit(main())
