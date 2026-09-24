"""Offline-only validation on successful audit action trace and all V1 traces."""
import json, hashlib
from pathlib import Path
import numpy as np
from v2_interface import direction_from_fresh_chunk
OUT=Path('/media/volume/data/exouser/activeforcing_table_push_v2_20260910/offline/DIRECTION_ESTIMATOR_VALIDATION.json')
V1=Path('/media/volume/data/exouser/activeforcing_table_push_20260910')
SUCCESS=Path('/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/rollouts/task5_init0_native_pi0_libero/actions.npy')
def assess(name,actions,start):
  chunk=actions[start:start+30]
  try:
   x=direction_from_fresh_chunk(chunk)
   later=actions[start+30:,:2]; norms=np.linalg.norm(later,axis=1); dots=(later[norms>=.01]/norms[norms>=.01,None])@np.asarray(x['direction_world_xy'])
   x.update({'name':name,'chunk_start':int(start),'later_count':int(len(dots)),'later_median_dot':None if not len(dots) else float(np.median(dots)),'persistence_positive':bool(len(dots) and np.median(dots)>0)})
   return x
  except Exception as e:return {'name':name,'error':repr(e)}
def main():
 assert not OUT.exists()
 result=[]
 a=np.load(SUCCESS);result.append(assess('successful_audit_trajectory',a,55))
 for p in sorted(V1.glob('rollouts/*/telemetry.jsonl')):
  rows=[json.loads(s) for s in p.read_text().splitlines() if s]; first=next(i for i,x in enumerate(rows) if x['contact_count']>0)
  result.append(assess(p.parent.name,np.asarray([x['nominal_action'] for x in rows]),first+2))
 OUT.write_text(json.dumps({'offline_only':True,'rule':'first 15 nondegenerate XY actions in 30 action window; coordinate median then normalize','cases':result},indent=2)+'\n')
if __name__=='__main__':main()
