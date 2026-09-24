import copy,json,sys,types
from pathlib import Path
import numpy as np
sys.path.insert(0,'/media/volume/data/exouser/activeforcing_table_push_v4_20260910/code')
import run_r1_replay as r1
from a3_control import C
O=Path('/media/volume/data/exouser/activeforcing_table_push_v4_20260910'); spec=json.load(open(O/'A3_TRACKING_CONTRACT.json'))
class A:
 def __init__(self,e):
  self.e=e;self.c=e.robots[0].controller;self.o=self.c.run_controller;self.F=np.zeros(3);self.t=[]
  def f(_):
   b=np.asarray(self.o(),float);J=np.asarray(self.c.J_pos,float);add=J.T@self.F;raw=b+add;lo,hi=e.robots[0].torque_limits;self.t.append({'base':b.tolist(),'added':add.tolist(),'raw':raw.tolist(),'finite':bool(np.isfinite(raw).all() and raw.shape==b.shape),'clip_count':int(np.sum((raw<lo)|(raw>hi))),'min_margin':float(np.min(np.minimum(raw-lo,hi-raw)))});return raw
  self.c.run_controller=types.MethodType(f,self.c)
 def close(self):self.c.run_controller=self.o
def one(target,feedback):
 tele=[json.loads(x) for x in open(r1.V2/'telemetry.jsonl')];nom=np.asarray([x['nominal_action'] for x in tele]);acts=np.asarray([x['executed_action'] for x in tele]);rec=json.load(open(r1.V2/'receipt.json'));d=np.asarray(rec['direction_record']['direction_world_xy']);e,o,_=r1.build()
 for x in acts[:57]:o,_,_,_=e.step(x.tolist())
 a=A(e);c=C(target); rows=[]
 for k in range(spec['horizon']):
  u=c.command() if feedback else 0.;a.F=np.r_[d*u,0.];a.t=[];pre=r1.contacts(e.sim.model,e.sim.data)[0];o,_,_,_=e.step(nom[57+k].tolist());f,ids,raw=r1.contacts(e.sim.model,e.sim.data);meas=float(f[:2]@d);rows.append({'k':k,'causal_measurement_index':k-1 if k else None,'requested_force_n':u,'applied_world_force_n':a.F.tolist(),'nominal_hash':r1.h(nom[57+k]),'post_force_n':meas,'contact':bool(ids),'raw_contacts':raw,'torque':copy.deepcopy(a.t)});c.observe(meas,bool(ids)) if feedback else None
 a.close();e.close();return {'target_n':target,'feedback':feedback,'rows':rows,'mae_last5':float(np.mean([abs(x['post_force_n']-target) for x in rows[-5:] if x['contact']]))}
def main():
 assert not (O/'A3_TRACKING_RESULT.json').exists();runs=[one(t,True) for t in spec['targets_n']];base=[one(t,False) for t in spec['targets_n']];real=[np.mean([x['post_force_n'] for x in z['rows'][-5:]]) for z in runs];passed=bool(np.all(np.diff(real)>0) and all(x['mae_last5']<=.8*y['mae_last5'] for x,y in zip(runs,base)));out={'gate':'A3_TRACKING','contract':spec,'feedback_runs':runs,'no_feedback_runs':base,'realized_last5_n':real,'pass':passed,'decision':'A3_TRACKING_PASS' if passed else 'A3_TRACKING_FAIL_STOP_BEFORE_OUTCOME'};(O/'A3_TRACKING_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
