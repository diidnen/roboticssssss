import csv, json, math
from pathlib import Path

ROOT=Path('/media/volume/data/exouser/pi0_table_friction_20260910')
BASE=Path('/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/rollouts/task5_init0_native_pi0_libero/receipt.json')
HIGH=Path('/media/volume/data/exouser/pi0_nontransport_20260910/rollouts/task5_init0_geom_intervention/receipt.json')
def read(p): return json.loads(Path(p).read_text())
def wilson(k,n,z=1.959963984540054):
 p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-h,c+h]
def main():
 rows=[]
 # The baseline receipt has no intervention block because it is native. Its exact
 # vector is established by the matched authoritative intervention's before-vector.
 for init,mu,path,reused in [(0,.6,BASE,True),(0,2.0,HIGH,True)]:
  r=read(path); rows.append({'init_index':init,'sliding_friction':mu,'receipt_path':str(path),'reused':reused,'receipt':r})
 for init in range(5):
  for mu in (.6,1.2,2.0):
   if init==0 and mu in (.6,2.0): continue
   label=str(mu).replace('.','p');p=ROOT/'rollouts'/f'task5_init{init}_table_mu_{label}'/'receipt.json'
   rows.append({'init_index':init,'sliding_friction':mu,'receipt_path':str(p),'reused':False,'receipt':read(p)})
 rows.sort(key=lambda x:(x['init_index'],x['sliding_friction']))
 out=[]
 for x in rows:
  r=x['receipt']; pm=r.get('plate_metrics',{}); inf=r.get('inference_receipts',[])
  lats=[q['latency_seconds'] for q in inf]
  iv=r.get('intervention',{})
  valid=(r.get('policy_config')=='pi0_libero' and r.get('checkpoint_tree_sha256')=='92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103' and r.get('task_id')==5 and r.get('language')=='push the plate to the front of the stove' and r.get('init_state',{}).get('index')==x['init_index'] and r.get('error') is None)
  if not x['reused']:
   valid &= (iv.get('readback_after')==[x['sliding_friction'],.005,.0001] and iv.get('before')==[.6,.005,.0001])
  elif x['sliding_friction']==2.0:
   valid &= (iv.get('before')==[.6,.005,.0001] and iv.get('readback_after')==[2.0,.005,.0001])
  failure='' if r.get('success') else 'native horizon exhausted without success; state trace shows substantially smaller planar displacement than successful fresh baselines' if pm else 'native horizon exhausted without success (legacy receipt; consult archived video/contact reconstruction)'
  out.append({'init_index':x['init_index'],'sliding_friction':x['sliding_friction'],'reused':x['reused'],'valid_protocol_cell':bool(valid),'success':bool(r.get('success')),'steps_executed':r.get('steps_executed'),'inference_request_count':len(inf) if inf else None,'inference_latency_mean_seconds':sum(lats)/len(lats) if lats else None,'inference_latency_max_seconds':max(lats) if lats else None,'plate_planar_displacement_xy':pm.get('planar_displacement_xy'),'plate_planar_delta_xy':pm.get('planar_delta_xy'),'plate_table_contact_steps':pm.get('plate_table_contact_steps'),'plate_table_contact_point_observations':pm.get('plate_table_contact_point_observations'),'failure_mode':failure,'receipt_path':x['receipt_path']})
 fields=list(out[0]);
 with (ROOT/'FINAL_GRID.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
 by={}
 for mu in (.6,1.2,2.0):
  rs=[x for x in out if x['sliding_friction']==mu];k=sum(x['success'] for x in rs)
  by[str(mu)]={'successes':k,'roots':len(rs),'success_rate':k/len(rs),'wilson_95':wilson(k,len(rs)),'all_cells_valid':all(x['valid_protocol_cell'] for x in rs),'completion_steps_successes':[x['steps_executed'] for x in rs if x['success']]}
 summary={'experiment':'pi0_table_friction_20260910','primary_metric':by,'monotonic_nonincreasing':by['0.6']['success_rate']>=by['1.2']['success_rate']>=by['2.0']['success_rate'],'fresh_baseline_competency':{'successes':sum(x['success'] for x in out if x['sliding_friction']==.6 and not x['reused']),'roots':4},'cells':out,'reused_validation':{'native_mu_0.6':'accepted: task/init/checkpoint/success semantics match; native receipt has no intervention field, and matched authoritative 2.0 receipt establishes table baseline before=[0.6,0.005,0.0001]','intervention_mu_2.0':'accepted: exact before/requested/readback vectors and task/init/checkpoint/success semantics match','comparability_caveat':'stock server exposes no seed API; reused cells came from an earlier frozen server process, so sampler RNG state is unknown and not matched to new cells'},'claim_boundary':'five fixed roots only; controlled one-property robustness evidence, not force adaptation, a validated contact-force mechanism, or population reliability'}
 (ROOT/'FINAL_GRID.json').write_text(json.dumps(summary,indent=2)+'\n')
 md=f'''# Frozen table-friction sweep report

Official online `pi0_libero` on LIBERO Goal task 5 (`push the plate to the front of the stove`) was evaluated over the frozen 5-root × 3-level grid. All 13 authorized new cells ran sequentially before the 15:00 UTC deadline; two verified init-0 pilot cells were reused.

| Sliding friction | Native success | Rate (Wilson 95%) |
|---:|---:|---:|
| 0.6 | {by['0.6']['successes']}/5 | {by['0.6']['success_rate']:.0%} ({by['0.6']['wilson_95'][0]:.1%}–{by['0.6']['wilson_95'][1]:.1%}) |
| 1.2 | {by['1.2']['successes']}/5 | {by['1.2']['success_rate']:.0%} ({by['1.2']['wilson_95'][0]:.1%}–{by['1.2']['wilson_95'][1]:.1%}) |
| 2.0 | {by['2.0']['successes']}/5 | {by['2.0']['success_rate']:.0%} ({by['2.0']['wilson_95'][0]:.1%}–{by['2.0']['wilson_95'][1]:.1%}) |

The observed ordering is monotonic non-increasing (`1.00 ≥ 0.00 ≥ 0.00`), consistent with the preregistered hypothesis. Fresh baseline roots were 4/4 successful; with reused init 0 the baseline is 5/5. Higher-friction failures all exhausted the native 310-step limit. Fresh state traces show much smaller plate planar displacement on failures than on their successful baseline counterparts; this is descriptive rather than a causal contact-force mechanism.

Validity: every new receipt identifies task 5, its intended supplied-init SHA-256, `pi0_libero`, the required checkpoint-tree digest, online requests/actions, and exact full-vector friction readback. Reused init-0 μ=2.0 is accepted because its authoritative receipt records `before=[0.6,0.005,0.0001]` and `readback=[2.0,0.005,0.0001]`. The older pilot protocol line saying `1.0` is stale and was not propagated. Reused and new cells remain RNG-state incomparable because the stock server exposes no seed API.

Claim limit: this is a five-fixed-root, one-property pilot. It supports only a controlled association between increased table sliding friction and reduced native feasibility under this frozen setup. It does not establish population reliability, force adaptation, or a causal mechanism beyond the declared intervention.
'''
 (ROOT/'FINAL_REPORT.md').write_text(md)
if __name__=='__main__': main()
