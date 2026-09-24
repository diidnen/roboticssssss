"""R2 local OSC instance adapter. It never edits robosuite or pi0 actions."""
import copy,json,sys,types
from pathlib import Path
import numpy as np
sys.path.insert(0,'/media/volume/data/exouser/activeforcing_table_push_v3_20260910/code')
import run_r1_replay as r1
OUT=r1.OUT; SNAPS=[55,57]; GRID=np.array([-2.,-1.,0.,1.,2.]) # N, preregistered conservative grid
class CartesianWrenchAdapter:
 """Instance-only post-OSC / pre-SingleArm actuator clipping torque augmentation."""
 def __init__(self,env):
  self.env=env; self.c=env.robots[0].controller; self.original=self.c.run_controller; self.enabled=False; self.force_world=np.zeros(3); self.records=[]
  def wrapped(_self):
   base=np.asarray(self.original(),float); J=np.asarray(self.c.J_pos,float)
   add=J.T@self.force_world if self.enabled else np.zeros_like(base)
   raw=base+add; low,high=env.robots[0].torque_limits; applied=np.clip(raw,low,high)
   self.records.append({'base_torque':base.tolist(),'jacobian_world_pos':J.tolist(),'force_world_requested_n':self.force_world.tolist(),'added_torque':add.tolist(),'raw_torque':raw.tolist(),'ctrl_low':np.asarray(low).tolist(),'ctrl_high':np.asarray(high).tolist(),'finite':bool(np.isfinite(raw).all() and raw.shape==base.shape),'clip_count':int(np.sum((raw<low)|(raw>high))),'min_margin':float(np.min(np.minimum(raw-low,high-raw))),'controller_return_equals_raw':bool(np.allclose(raw,base+add,rtol=0,atol=0))})
   return raw # SingleArm.control applies stock actuator clipping after this return
  self.c.run_controller=types.MethodType(wrapped,self.c)
 def set(self,force_world,enabled): self.force_world=np.asarray(force_world,float);self.enabled=bool(enabled);self.records=[]
 def close(self): self.c.run_controller=self.original
def main():
 assert (OUT/'R1_RESULT.json').exists() and json.loads((OUT/'R1_RESULT.json').read_text())['pass']; assert not (OUT/'R2_RESULT.json').exists()
 tele=[json.loads(x) for x in (r1.V2/'telemetry.jsonl').read_text().splitlines()]; actions=np.asarray([x['executed_action'] for x in tele],float); nominal=np.asarray([x['nominal_action'] for x in tele],float);rec=json.loads((r1.V2/'receipt.json').read_text());direction=np.asarray(rec['direction_record']['direction_world_xy']); env,o,_=r1.build(); snaps={}
 for step in range(max(SNAPS)+1):
  if step in SNAPS: snaps[step]=r1.capture(env,o,nominal[step],direction)
  if step<max(SNAPS):o,_,_,_=env.step(actions[step].tolist())
 adapter=CartesianWrenchAdapter(env); result={'gate':'R2','implementation':'instance-only OSC run_controller wrapper: stock OSC torque + J_pos.T @ world_xy_force immediately before stock SingleArm.control actuator clipping','source_receipt':{'osc':'controllers/osc.py: stock torque = J_full.T @ decoupled_wrench + torque_compensation + nullspace','jacobian':'base_controller.py: J_pos = MuJoCo site translational Jacobian in world coordinates, restricted to qvel indices','clip':'robots/single_arm.py clips returned torques to torque_limits from actuator_ctrlrange before sim.data.ctrl write'},'grid_requested_newtons':GRID.tolist(),'snapshot_results':[],'pass':False,'pi0_nominal_replanned':False,'new_task_outcomes':0}
 for step,snap in snaps.items():
  # exact cloned disabled parity: untouched vs wrapper-disabled, identical nominal bytes.
  oa=r1.restore(env,snap); adapter.set([0,0,0],False); out_a=env.step(nominal[step].tolist()); qa=env.sim.data.qpos.copy();fa,ia,ra=r1.contacts(env.sim.model,env.sim.data); ta=copy.deepcopy(adapter.records)
  ob=r1.restore(env,snap); adapter.set([0,0,0],False); out_b=env.step(nominal[step].tolist()); fb,ib,rb=r1.contacts(env.sim.model,env.sim.data); tb=copy.deepcopy(adapter.records)
  parity={'post_qpos':bool(np.allclose(qa,env.sim.data.qpos,atol=0,rtol=0)),'post_force':bool(np.allclose(fa,fb,atol=0,rtol=0)),'contacts':ia==ib,'torque_trace':ta==tb,'nominal_bytes_identical':nominal[step].tobytes()==nominal[step].copy().tobytes()}
  branches=[]
  for F in GRID:
   oo=r1.restore(env,snap); force=np.r_[direction*F,0.]; adapter.set(force,True); pre=r1.state(env,oo,nominal[step],direction); post,rew,done,info=env.step(nominal[step].tolist()); f,ids,raw=r1.contacts(env.sim.model,env.sim.data); trace=copy.deepcopy(adapter.records)
   assert trace and all(t['finite'] for t in trace)
   branches.append({'requested_xy_force_n':float(F),'force_world_requested_n':force.tolist(),'pi0_nominal_action':nominal[step].tolist(),'pi0_nominal_sha256':r1.h(nominal[step]),'no_replan':True,'pre_force_world_on_plate':pre['pre_force'].tolist(),'post_force_world_on_plate':f.tolist(),'post_raw_contacts':raw,'post_contact_identities':ids,'contact_retained':bool(ids),'post_aligned_force_n':float(f[:2]@direction),'torque_control_substeps':trace})
  vals=np.array([b['post_aligned_force_n'] for b in branches]); sat=max(t['clip_count'] for b in branches for t in b['torque_control_substeps']); result['snapshot_results'].append({'step':step,'disabled_adapter_strict_parity':parity,'branches':branches,'strict_signed_order':bool(np.all(np.diff(vals)>0)),'contact_retained_all':all(b['contact_retained'] for b in branches),'max_actuator_clip_count':sat,'actuator_saturation_dominant':bool(sat>0)})
 adapter.close();env.close();valid=[x for x in result['snapshot_results'] if all(x['disabled_adapter_strict_parity'].values()) and x['strict_signed_order'] and x['contact_retained_all'] and not x['actuator_saturation_dominant']]
 result['pass']=len(valid)>=2;result['decision']='R2_PASS' if result['pass'] else 'R2_FAIL_STOP_BEFORE_A3';result['rule']='two strict clone parity + signed order + retained contact + no actuator clipping snapshots';(OUT/'R2_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
