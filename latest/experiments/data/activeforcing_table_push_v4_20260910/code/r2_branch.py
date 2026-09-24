"""One fresh-process V4 R2 branch; no policy inference or task result claim."""
import hashlib,json,os,sys,tempfile,types
from pathlib import Path
import numpy as np
sys.path.insert(0,'/media/volume/newdata/exouser/flowdagger_e960a/bundle/e959c_bootstrap_resolved_r0_20260817T082827Z/flowdagger/flowdagger_pi05/openpi/third_party/libero')
import mujoco,torch
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
OUT=Path('/media/volume/data/exouser/activeforcing_table_push_v4_20260910');V2=Path('/media/volume/data/exouser/activeforcing_table_push_v2_20260910/calibration/calibration_root0_mu1.2_residual_plus0.00'); ROBOT={'gripper0_hand_collision','gripper0_finger1_collision','gripper0_finger1_pad_collision','gripper0_finger2_collision','gripper0_finger2_pad_collision'}
def h(x):return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()
def atomic(p,x):
 fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.tmp_',suffix='.json');os.close(fd);Path(tmp).write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def contacts(m,d):
 pg={i for i in range(m.ngeom) if str(m.geom_id2name(i)).startswith('plate_1_')};tot=np.zeros(3);ids=[];raw=[]
 for i in range(d.ncon):
  c=d.contact[i];n1,n2=m.geom_id2name(c.geom1),m.geom_id2name(c.geom2)
  if (n1 in ROBOT and c.geom2 in pg) or (n2 in ROBOT and c.geom1 in pg):
   w=np.zeros(6);mujoco.mj_contactForce(m._model,d._data,i,w);f=c.frame.reshape(3,3).T@w[:3];f=f if c.geom2 in pg else -f;tot+=f;ids.append([n1,n2]);raw.append({'index':i,'geom1':n1,'geom2':n2,'wrench_contact':w.tolist(),'force_world_on_plate':f.tolist()})
 return tot,ids,raw
def goalhash(e):
 c=e.robots[0].controller;v=[]
 for k in ('goal_pos','goal_ori','goal_ori_quat','goal_vel','goal_torque'):
  if hasattr(c,k):v.append(np.asarray(getattr(c,k)).ravel())
 return h(np.concatenate(v) if v else np.zeros(0))
def obshash(o):return hashlib.sha256(b''.join(np.ascontiguousarray(np.asarray(o[k])).tobytes() for k in sorted(o))).hexdigest()
class Adapter:
 def __init__(self,e,enabled,F):
  self.e=e;self.c=e.robots[0].controller;self.orig=self.c.run_controller;self.enabled=enabled;self.F=np.asarray(F);self.trace=[]
  def run(_):
   base=np.asarray(self.orig(),float);J=np.asarray(self.c.J_pos,float);add=J.T@self.F if self.enabled else np.zeros_like(base);raw=base+add;lo,hi=e.robots[0].torque_limits
   self.trace.append({'base_torque':base.tolist(),'J_pos_world':J.tolist(),'requested_world_force_n':self.F.tolist(),'added_torque':add.tolist(),'raw_returned_torque':raw.tolist(),'finite':bool(np.isfinite(raw).all()),'shape':list(raw.shape),'disabled_raw_equals_base_exact':bool(np.array_equal(raw,base)) if not self.enabled else None,'ctrlrange_low':lo.tolist(),'ctrlrange_high':hi.tolist(),'clip_count':int(np.sum((raw<lo)|(raw>hi))),'min_margin':float(np.min(np.minimum(raw-lo,hi-raw)))})
   return raw
  self.c.run_controller=types.MethodType(run,self.c)
 def close(self):self.c.run_controller=self.orig
def main():
 step=int(sys.argv[1]);kind=sys.argv[2];force=float(sys.argv[3]);out=OUT/'branches'/f's{step}_{kind}_{force:+.0f}N.json'
 if out.exists():raise RuntimeError('immutable branch exists')
 tele=[json.loads(x) for x in (V2/'telemetry.jsonl').read_text().splitlines()];acts=np.asarray([x['executed_action'] for x in tele]);nom=np.asarray([x['nominal_action'] for x in tele]);rec=json.loads((V2/'receipt.json').read_text());dxy=np.asarray(rec['direction_record']['direction_world_xy']);old=torch.load
 def load(*a,**kw):kw.pop('weights_only',None);return old(*a,weights_only=False,**kw)
 torch.load=load;suite=benchmark.get_benchmark_dict()['libero_goal']();task=suite.get_task(5);init=suite.get_task_init_states(5)[0];env=OffScreenRenderEnv(bddl_file_name=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,camera_heights=256,camera_widths=256);env.seed(7);env.reset();o=env.set_init_state(init);table=next(i for i in range(env.sim.model.ngeom) if env.sim.model.geom_id2name(i)=='table_collision');env.sim.model.geom_friction[table]=[1.2,.005,.0001]
 for a in acts[:step]:o,_,_,_=env.step(a.tolist())
 pre=env.get_sim_state().copy();pf,pid,praw=contacts(env.sim.model,env.sim.data);F=np.r_[dxy*force,0.];adapter=None
 if kind!='untouched':adapter=Adapter(env,kind=='signed',F if kind=='signed' else [0,0,0])
 o2,reward,done,info=env.step(nom[step].tolist());post=env.get_sim_state().copy();ff,fid,fraw=contacts(env.sim.model,env.sim.data);armidx=env.robots[0]._ref_joint_actuator_indexes;ctrl=env.sim.data.ctrl[armidx].copy();lo,hi=env.robots[0].torque_limits
 doc={'step':step,'kind':kind,'requested_force_n':force,'source':'V2 zero residual official pi0 action prefix; fresh-process replay diagnostic only','new_task_outcomes':0,'seed':7,'friction_readback':env.sim.model.geom_friction[table].tolist(),'pre_state':{'hash':h(pre),'array':pre.tolist(),'qpos':env.sim.data.qpos.tolist() if False else pre[:len(env.sim.data.qpos)].tolist(),'qvel':env.sim.data.qvel.tolist()},'post_state':{'hash':h(post),'array':post.tolist(),'qpos':env.sim.data.qpos.tolist(),'qvel':env.sim.data.qvel.tolist()},'controller_goal_hash':goalhash(env),'observation_hash_pre':obshash(o),'observation_hash_post':obshash(o2),'nominal_action':nom[step].tolist(),'nominal_action_sha256':h(nom[step]),'direction':dxy.tolist(),'direction_hash':h(dxy),'no_replan':True,'pre_contacts':pid,'pre_raw_contacts':praw,'post_contacts':fid,'post_raw_contacts':fraw,'post_aligned_force_n':float(ff[:2]@dxy),'final_ctrl':ctrl.tolist(),'final_ctrl_hash':h(ctrl),'ctrl_margin':float(np.min(np.minimum(ctrl-lo,hi-ctrl))),'adapter_installed':adapter is not None,'adapter_trace':[] if adapter is None else adapter.trace}
 if adapter:adapter.close()
 env.close();atomic(out,doc)
if __name__=='__main__':main()
