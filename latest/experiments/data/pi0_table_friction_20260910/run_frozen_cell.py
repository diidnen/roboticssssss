"""One frozen official pi0_libero online rollout for the table-friction grid."""
import collections, datetime, hashlib, json, math, os, pathlib, sys, tempfile, time
import cv2, numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

ROOT=pathlib.Path('/media/volume/data/exouser/pi0_table_friction_20260910')
OUT=ROOT/'rollouts'; CHECKPOINT='/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero'
TREE='92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103'
DEADLINE=datetime.datetime(2026,9,10,15,tzinfo=datetime.timezone.utc)
def digest(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()
def axis(q):
 q=np.asarray(q).copy();q[3]=np.clip(q[3],-1,1);d=np.sqrt(1-q[3]*q[3]);return np.zeros(3) if math.isclose(d,0) else q[:3]*2*math.acos(q[3])/d
def atomic(path,obj):
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.tmp_',suffix='.json');os.close(fd);pathlib.Path(tmp).write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)
def gid(m,name):
 for i in range(m.ngeom):
  if m.geom_id2name(i)==name:return i
 raise KeyError(name)
def plate_body_id(m):
 for i in range(m.nbody):
  if m.body_id2name(i)=='plate_1_main': return i
 raise KeyError('plate_1_main')
def plate_geom_ids(m):
 return {i for i in range(m.ngeom) if str(m.geom_id2name(i)).startswith('plate_1_')}
def main():
 if datetime.datetime.now(datetime.timezone.utc)>=DEADLINE: raise RuntimeError('launch deadline passed')
 idx=int(sys.argv[1]); friction=float(sys.argv[2]); label=str(friction).replace('.','p')
 original=torch.load;torch.load=lambda *a,**kw:original(*a,weights_only=False,**kw)
 suite=benchmark.get_benchmark_dict()['libero_goal']();task=suite.get_task(5);init=suite.get_task_init_states(5)[idx]
 if task.language!='push the plate to the front of the stove': raise RuntimeError('wrong task language')
 bddl=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
 env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256);env.seed(7)
 run=OUT/f'task5_init{idx}_table_mu_{label}';run.mkdir(exist_ok=False)
 r={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'policy_config':'pi0_libero','checkpoint_path':CHECKPOINT,'checkpoint_tree_sha256':TREE,'task_suite':'libero_goal','task_id':5,'language':task.language,'init_state':{'index':idx,'sha256':digest(init),'shape':list(init.shape),'dtype':str(init.dtype)},'horizon':300,'stabilization_dummy_steps':10,'replan_steps':5,'server_rng_behavior':'unknown; frozen server process, no seed API exposed by stock websocket server','action_source':'online websocket frozen pi0_libero only after stabilization','success_rule':'native env.step done flag; final wrapped env.check_success recorded','intervention':{},'actions':[],'inference_receipts':[],'state_trace_schema':'post-step plate_1_main xpos and plate/table collision contacts'}
 writer=None
 try:
  env.reset();obs=env.set_init_state(init);m=env.sim.model;i=gid(m,'table_collision');pb=plate_body_id(m);pg=plate_geom_ids(m);before=m.geom_friction[i].copy().tolist();requested=[friction,.005,.0001];m.geom_friction[i]=requested;readback=m.geom_friction[i].copy().tolist();initial_pos=env.sim.data.body_xpos[pb].copy();trace=[]
  r['intervention']={'property':'model.geom_friction[table_collision][0]','full_property':'model.geom_friction[table_collision]','before':before,'requested_after':requested,'readback_after':readback,'preserved_torsional_rolling':[.005,.0001],'one_runtime_property_only':True}
  if not np.allclose(readback,requested,rtol=0,atol=1e-12):raise RuntimeError('invalid friction property readback')
  writer=cv2.VideoWriter(str(run/'agentview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(224,224));client=WebsocketClientPolicy('127.0.0.1',8010);plan=collections.deque();done=False
  for step in range(310):
   img=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),224,224));wrist=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),224,224));writer.write(cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
   if step<10:a=np.array([0]*6+[-1.],float);src='stabilization_dummy'
   else:
    if not plan:
     e={'observation/image':img,'observation/wrist_image':wrist,'observation/state':np.concatenate((obs['robot0_eef_pos'],axis(obs['robot0_eef_quat']),obs['robot0_gripper_qpos'])),'prompt':str(task.language)};t=time.monotonic();ans=client.infer(e);elapsed=time.monotonic()-t;chunk=np.asarray(ans['actions'])
     if len(chunk)<5:raise RuntimeError('policy action chunk shorter than frozen cadence')
     plan.extend(chunk[:5]);r['inference_receipts'].append({'env_step':step,'latency_seconds':elapsed,'chunk_shape':list(chunk.shape)})
    a=np.asarray(plan.popleft(),float);src='online_pi0_libero'
   obs,reward,done,info=env.step(a.tolist()); contacts=sum(1 for c in env.sim.data.contact[:env.sim.data.ncon] if ((c.geom1==i and c.geom2 in pg) or (c.geom2==i and c.geom1 in pg))); pos=env.sim.data.body_xpos[pb].copy();trace.append({'env_step':step,'plate_xpos':pos.tolist(),'plate_table_contact_points':contacts});r['actions'].append({'env_step':step,'source':src,'action':a.tolist(),'reward':float(reward),'done':bool(done)})
   if done:break
  final_pos=env.sim.data.body_xpos[pb].copy();r.update({'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'steps_executed':len(r['actions']),'success':bool(done),'final_native_check_success':bool(env.check_success()),'plate_metrics':{'body':'plate_1_main','initial_xpos':initial_pos.tolist(),'final_xpos':final_pos.tolist(),'planar_displacement_xy':float(np.linalg.norm((final_pos-initial_pos)[:2])),'planar_delta_xy':(final_pos-initial_pos)[:2].tolist(),'target_progress':'not reconstructed from task internals; see raw authoritative plate state trace','plate_table_contact_steps':sum(x['plate_table_contact_points']>0 for x in trace),'plate_table_contact_point_observations':sum(x['plate_table_contact_points'] for x in trace)}})
 except Exception as e:r.update({'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'error':repr(e),'success':False})
 finally:
  if writer:writer.release()
  env.close()
 np.save(run/'actions.npy',np.array([x['action'] for x in r['actions']],float));(run/'state_trace.jsonl').write_text(''.join(json.dumps(x)+'\\n' for x in locals().get('trace',[])));atomic(run/'receipt.json',r);print(json.dumps({k:r.get(k) for k in ('success','steps_executed','error','final_native_check_success','intervention')},indent=2));return 0 if 'error' not in r else 1
if __name__=='__main__':sys.exit(main())
