"""R1 exact-state counterfactual using one replay and reversible snapshots."""
import copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/media/volume/newdata/exouser/flowdagger_e960a/bundle/e959c_bootstrap_resolved_r0_20260817T082827Z/flowdagger/flowdagger_pi05/openpi/third_party/libero')
import mujoco, torch
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
OUT=Path('/media/volume/data/exouser/activeforcing_table_push_v3_20260910'); V2=Path('/media/volume/data/exouser/activeforcing_table_push_v2_20260910/calibration/calibration_root0_mu1.2_residual_plus0.00')
ROBOT={'gripper0_hand_collision','gripper0_finger1_collision','gripper0_finger1_pad_collision','gripper0_finger2_collision','gripper0_finger2_pad_collision'}
SNAPS=[55,57,59]; TOL=1e-10; MARGIN=.01
def h(x):
 a=np.ascontiguousarray(np.asarray(x));return hashlib.sha256(a.tobytes()).hexdigest()
def obs_hash(o): return hashlib.sha256(b''.join(np.ascontiguousarray(np.asarray(o[k])).tobytes() for k in sorted(o))).hexdigest()
def contacts(m,d):
 pg={i for i in range(m.ngeom) if str(m.geom_id2name(i)).startswith('plate_1_')}; total=np.zeros(3); identities=[];raw=[]
 for i in range(d.ncon):
  c=d.contact[i]; n1,n2=m.geom_id2name(c.geom1),m.geom_id2name(c.geom2)
  if (n1 in ROBOT and c.geom2 in pg) or (n2 in ROBOT and c.geom1 in pg):
   w=np.zeros(6);mujoco.mj_contactForce(m._model,d._data,i,w); f=c.frame.reshape(3,3).T@w[:3]; f=f if c.geom2 in pg else -f
   total+=f;identities.append([n1,n2]);raw.append({'index':i,'geom1':n1,'geom2':n2,'force_world_on_plate':f.tolist(),'wrench_contact':w.tolist()})
 return total,identities,raw
def goals(env):
 c=env.robots[0].controller; vals=[]
 for k in ('goal_pos','goal_ori','goal_ori_quat','goal_vel','goal_torque'):
  if hasattr(c,k): vals.append(np.asarray(getattr(c,k)).ravel())
 return h(np.concatenate(vals) if vals else np.zeros(0))
def build():
 old=torch.load
 def compatible_load(*a,**kw):
  kw.pop('weights_only',None); return old(*a,weights_only=False,**kw)
 torch.load=compatible_load
 suite=benchmark.get_benchmark_dict()['libero_goal'](); task=suite.get_task(5); init=suite.get_task_init_states(5)[0]
 env=OffScreenRenderEnv(bddl_file_name=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,camera_heights=256,camera_widths=256);env.seed(7);env.reset();o=env.set_init_state(init);m,d=env.sim.model,env.sim.data
 table=next(i for i in range(m.ngeom) if m.geom_id2name(i)=='table_collision');m.geom_friction[table]=[1.2,.005,.0001]
 return env,o,task
def state(env,o,nominal,direction):
 m,d=env.sim.model,env.sim.data; f,ids,raw=contacts(m,d)
 return {'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'controller_goal_hash':goals(env),'contact_identities':ids,'pre_force':f,'pre_raw':raw,'observation_hash':obs_hash(o),'nominal':np.asarray(nominal), 'direction':np.asarray(direction)}
def equal(a,b):
 return {'qpos':bool(np.allclose(a['qpos'],b['qpos'],atol=TOL,rtol=0)),'qvel':bool(np.allclose(a['qvel'],b['qvel'],atol=TOL,rtol=0)),'controller_goal':a['controller_goal_hash']==b['controller_goal_hash'],'contacts':a['contact_identities']==b['contact_identities'],'observation':a['observation_hash']==b['observation_hash'],'nominal':bool(np.array_equal(a['nominal'],b['nominal'])),'direction':bool(np.array_equal(a['direction'],b['direction']))}
def capture(env,o,nominal,direction):
 c=env.robots[0].controller
 ctrl={}
 for k,v in c.__dict__.items():
  if isinstance(v,np.ndarray): ctrl[k]=v.copy()
  elif isinstance(v,(int,float,bool,str,type(None),np.number)): ctrl[k]=copy.deepcopy(v)
 observables={}
 for k,v in env.env._observables.items():
  observables[k]={
   '_time_since_last_sample':copy.deepcopy(v._time_since_last_sample),
   '_current_delay':copy.deepcopy(v._current_delay),
   '_current_observed_value':copy.deepcopy(v._current_observed_value),
   '_sampled':copy.deepcopy(v._sampled),
  }
 return {
  'sim':env.get_sim_state().copy(), 'ctrl':ctrl,
  'cur_time':copy.deepcopy(env.env.cur_time),'timestep':copy.deepcopy(env.env.timestep),
  'done':copy.deepcopy(env.env.done),'obs_cache':copy.deepcopy(env.env._obs_cache),
  'observables':observables,'obs':copy.deepcopy(o),'record':state(env,o,nominal,direction),
 }
def restore(env,snap):
 env.set_state(snap['sim']); env.sim.forward()
 env.env.cur_time=copy.deepcopy(snap['cur_time']); env.env.timestep=copy.deepcopy(snap['timestep']); env.env.done=copy.deepcopy(snap['done'])
 env.env._obs_cache=copy.deepcopy(snap['obs_cache'])
 for k,vals in snap['observables'].items():
  ob=env.env._observables[k]
  for name,val in vals.items(): setattr(ob,name,copy.deepcopy(val))
 c=env.robots[0].controller
 for k,v in snap['ctrl'].items(): setattr(c,k,copy.deepcopy(v))
 return copy.deepcopy(snap['obs'])
def main():
 assert not (OUT/'R1_RESULT.json').exists()
 tele=[json.loads(x) for x in (V2/'telemetry.jsonl').read_text().splitlines()]; rec=json.loads((V2/'receipt.json').read_text()); actions=np.asarray([x['executed_action'] for x in tele],float); nominal=np.asarray([x['nominal_action'] for x in tele],float); direction=np.asarray(rec['direction_record']['direction_world_xy'],float)
 (OUT/'canonical'/'v2_zero_required_prefix.json').write_text(json.dumps({'source':str(V2),'not_task_outcome':True,'direction':direction.tolist(),'fresh_chunk_sha256':rec['direction_record']['chunk_sha256'],'fresh_chunk':rec['direction_record']['raw_chunk'],'actions_sha256':h(actions[:max(SNAPS)+1]),'executed_actions':actions[:max(SNAPS)+1].tolist(),'nominal_actions':nominal[:max(SNAPS)+1].tolist(),'snapshots':SNAPS},indent=2)+'\n')
 result={'gate':'R1','source':'V2 zero-residual online official pi0 trajectory, replay diagnostic only','implementation':'single exact replay plus reversible MuJoCo/controller/observable snapshots','new_task_outcomes':0,'snapshot_results':[],'pass':False}
 env,o,task=build(); snapshots={}; start=time.time()
 for step in range(max(SNAPS)+1):
  if step in SNAPS: snapshots[step]=capture(env,o,nominal[step],direction)
  if step<max(SNAPS): o,_,_,_=env.step(actions[step].tolist())
 for s in SNAPS:
  snap=snapshots[s]
  oa=restore(env,snap); a=state(env,oa,nominal[s],direction)
  ob=restore(env,snap); b=state(env,ob,nominal[s],direction)
  parity=equal(a,b)
  parity['matches_canonical_state']=all(equal(a,snap['record']).values())
  if not all(parity.values()): result['snapshot_results'].append({'step':s,'admitted':False,'parity':parity});continue
  # compare V2 saved nominal/direction at same row: exact fixed branch action.
  assert np.array_equal(a['nominal'],nominal[s]); grid=np.array([-.06,-.03,0,.03,.06]); branches=[]
  for r in grid:
   oo=restore(env,snap); z=state(env,oo,nominal[s],direction); act=nominal[s].copy();act[:2]+=r*direction
   safe=bool(np.all(np.abs(act[:2])<1-MARGIN)); assert safe
   post,rew,done,info=env.step(act.tolist()); pf,pi,pr=contacts(env.sim.model,env.sim.data)
   branches.append({'residual_scalar':float(r),'executed_action':act.tolist(),'final_action_safe':safe,'pre_force_world_on_plate':z['pre_force'].tolist(),'post_force_world_on_plate':pf.tolist(),'post_raw_contacts':pr,'post_contact_identities':pi,'contact_retained':bool(pi),'force_projection_post_n':float(pf[:2]@direction)})
  zero=next(x['force_projection_post_n'] for x in branches if x['residual_scalar']==0); inc=[x['force_projection_post_n']-zero for x in branches]; vals=[x['force_projection_post_n'] for x in branches]
  result['snapshot_results'].append({'step':s,'admitted':True,'parity':parity,'branches':branches,'increments_vs_zero_n':inc,'strict_monotonic_post_force':bool(np.all(np.diff(vals)>0)),'all_contact_retained':all(x['contact_retained'] for x in branches)})
 env.close(); result['elapsed_s']=time.time()-start
 valid=[x for x in result['snapshot_results'] if x.get('admitted') and x['strict_monotonic_post_force'] and x['all_contact_retained']]
 result['pass']=len(valid)>=2; result['decision']='R1_PASS' if result['pass'] else 'R1_FAIL_STOP_OSC_CHANNEL_NOT_DEMONSTRATED'; result['rule']='>=2 admitted snapshots with strict post-step order and retained contact'
 (OUT/'R1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
