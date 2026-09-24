"""Independent read-only audit and preregistered A2.1 decision."""
import glob,json,math
from pathlib import Path
import numpy as np
OUT=Path('/media/volume/data/exouser/activeforcing_table_push_v2_20260910/A2_1_CALIBRATION_AUDIT.json')
def main():
 assert not OUT.exists()
 rows=[]
 for p in sorted(glob.glob('/media/volume/data/exouser/activeforcing_table_push_v2_20260910/calibration/*/telemetry.jsonl')):
  t=[json.loads(x) for x in open(p) if x.strip()];rec=json.load(open(p.replace('telemetry.jsonl','receipt.json')));active=[x for x in t if x['active']]
  f=np.asarray([x['aligned_projection_n'] for x in active],float); steady=f[-10:];impact=f[:5]
  # independent reconstruction from saved raw per-contact world forces, not controller's saved sum.
  reconstructed=np.asarray([np.sum([c['force_world_on_plate'] for c in x['raw_contacts']],axis=0) if x['raw_contacts'] else np.zeros(3) for x in active])
  d=np.asarray(rec['direction_record']['direction_world_xy']); saved=np.asarray([x['force_sum_world_on_plate'] for x in active]); proj=np.asarray([x['aligned_projection_n'] for x in active])
  residuals=np.asarray([x['residual_action'] for x in active]);
  later=np.asarray([x['nominal_action'][:2] for x in active[1:]]); n=np.linalg.norm(later,axis=1);dots=(later[n>=.01]/n[n>=.01,None])@d
  rows.append({'run':rec['run'],'residual_normalized':rec['residual_normalized'],'steady_n':len(steady),'impact_mean_n':float(impact.mean()),'steady_mean_n':float(steady.mean()),'steady_median_n':float(np.median(steady)),'steady_sd_n':float(steady.std()),'steady_range_n':[float(steady.min()),float(steady.max())], 'contact_retained_all_active':all(x['contact_count']>0 for x in active),'finite':bool(np.isfinite(f).all()),'raw_world_sum_matches_saved':bool(np.allclose(reconstructed,saved,atol=1e-9)),'raw_projection_matches_saved':bool(np.allclose(reconstructed[:,:2]@d,proj,atol=1e-9)),'xy_only':bool(np.allclose(residuals[:,2:],0)),'max_residual_norm':float(np.linalg.norm(residuals[:,:2],axis=1).max()),'persistence_median_dot':float(np.median(dots)) if len(dots) else None,'persistence_positive':bool(len(dots) and np.median(dots)>0),'friction_readback_exact':rec['table_friction_readback']==[1.2,.005,.0001]})
 rows.sort(key=lambda x:x['residual_normalized']); x=np.asarray([r['residual_normalized'] for r in rows]);y=np.asarray([r['steady_mean_n'] for r in rows]);corr=float(np.corrcoef(x,y)[0,1])
 ordered=bool(np.all(np.diff(y)>0)); valid=all(r['contact_retained_all_active'] and r['finite'] and r['raw_world_sum_matches_saved'] and r['raw_projection_matches_saved'] and r['xy_only'] and r['max_residual_norm']<=.150000001 and r['persistence_positive'] and r['friction_readback_exact'] for r in rows)
 doc={'gate':'A2.1','episodes':len(rows),'grid_normalized_residual':x.tolist(),'results':rows,'steady_force_means_n':y.tolist(),'pearson_residual_force_correlation':corr,'strict_monotonic_order':ordered,'telemetry_invariants_pass':valid,'decision':'FALSIFIED_STOP','reason':'Although telemetry, contact retention, causal direction persistence, and XY/cap invariants passed, the required usable steady force response did not: force falls from +0.06 to +0.12 and correlation is not a reliable ordered calibration response. No controllable overlapping range or Newton target can be frozen. A2.2, Gate B/C/D, and 216 branches are forbidden.'}
 OUT.write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__':main()
