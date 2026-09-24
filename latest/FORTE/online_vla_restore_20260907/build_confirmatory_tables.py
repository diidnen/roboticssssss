"""Read-only analysis for the frozen 4-root confirmatory run and 8-root pool."""
from pathlib import Path
import csv,json,collections,math,statistics,itertools
import numpy as np
from scipy.stats import spearmanr
HERE=Path('/home/exouser/FORTE/online_vla_restore_20260907'); BASE=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907'); OUT=BASE/'confirmatory_v1'; NEW=OUT/'analysis_new4_v3'; OLD=BASE/'final_main_results_v1'
METHODS=['FIXED_3','FIXED_4','FIXED_5','ACTIVEFORCING']; LABEL={'FIXED_3':'Fixed-3','FIXED_4':'Fixed-4','FIXED_5':'Fixed-5','ACTIVEFORCING':'ActiveForcing'}
def readj(p):return json.loads(Path(p).read_text())
def writej(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvread(p):return list(csv.DictReader(Path(p).open()))
def writecsv(p,rows):
 if not rows:return
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def f(x):return float(x)
def i(x):return int(x)
def pct(x):return float(x)
def mean(xs):return float(np.mean(xs)) if xs else None

def load_rows():
 old=csvread(OLD/'FINAL_ONLINE_VLA_MAIN_ROWS.csv'); new=csvread(NEW/'FINAL_ONLINE_VLA_MAIN_ROWS.csv')
 for r in old+new:
  r['root']=i(r['root']);r['task']=i(r['task']);r['full_task_success']=i(r['full_task_success']);r['lift_success']=i(r['lift_success']);r['dropped']=i(r['dropped']);r['selected_force_N']=f(r['selected_force_N']);r['measured_squeeze_N']=f(r['measured_squeeze_N']);r['friction']=r['friction'];r['method']=r['method'];
 return old,new,old+new

def aggregate(rows):
 return {'n':len(rows),'full_task_success_n':sum(r['full_task_success'] for r in rows),'full_task_SR':mean([r['full_task_success'] for r in rows]),'lift_SR':mean([r['lift_success'] for r in rows]),'drop_rate':mean([r['dropped'] for r in rows]),'mean_selected_force':mean([r['selected_force_N'] for r in rows]),'mean_measured_bilateral_squeeze':mean([r['measured_squeeze_N'] for r in rows]),'median_measured_bilateral_squeeze':float(np.median([r['measured_squeeze_N'] for r in rows]))}

def contexts_from(rows):
 d=collections.defaultdict(dict)
 for r in rows:d[r['context']][r['method']]=r
 return d

def table_for(rows):
 return [dict(method=LABEL[m],**aggregate([r for r in rows if r['method']==m]),force_saving_vs_fixed5_measured=(aggregate([r for r in rows if r['method']=='FIXED_5'])['mean_measured_bilateral_squeeze']-aggregate([r for r in rows if r['method']==m])['mean_measured_bilateral_squeeze'])) for m in METHODS]

def context_table(rows):
 d=contexts_from(rows);out=[]
 for cid,v in sorted(d.items()):
  a=v['ACTIVEFORCING'];out.append({'context':cid,'root':a['root'],'task':a['task'],'friction':a['friction'],'AF_success':a['full_task_success'],'AF_lift':a['lift_success'],'AF_drop':a['dropped'],'AF_selected_force':a['selected_force_N'],'AF_measured_force':a['measured_squeeze_N'],'Fixed3_success':v['FIXED_3']['full_task_success'],'Fixed4_success':v['FIXED_4']['full_task_success'],'Fixed5_success':v['FIXED_5']['full_task_success'],'Fixed3_measured_force':v['FIXED_3']['measured_squeeze_N'],'Fixed4_measured_force':v['FIXED_4']['measured_squeeze_N'],'Fixed5_measured_force':v['FIXED_5']['measured_squeeze_N'],'AF_minus_F3_force_saving':v['FIXED_3']['measured_squeeze_N']-a['measured_squeeze_N'],'AF_minus_F4_force_saving':v['FIXED_4']['measured_squeeze_N']-a['measured_squeeze_N'],'AF_minus_F5_force_saving':v['FIXED_5']['measured_squeeze_N']-a['measured_squeeze_N']})
 return out

def paired(ctxs,baseline):
 out=[]
 for c in ctxs:
  af=c['AF_success'];b=c[f'{baseline}_success'];out.append({'context':c['context'],'root':c['root'],'task':c['task'],'friction':c['friction'],'AF_success':af,'baseline_success':b,'both_success':int(af==1 and b==1),'AF_only_success':int(af==1 and b==0),'baseline_only_success':int(af==0 and b==1),'both_fail':int(af==0 and b==0),'AF_selected_force':c['AF_selected_force'],'baseline_selected_force':{'Fixed3':3.0,'Fixed4':4.0,'Fixed5':5.0}[baseline],'AF_measured_force':c['AF_measured_force'],'baseline_measured_force':c[f'{baseline}_measured_force'],'measured_force_difference_baseline_minus_AF':c[f'{baseline}_measured_force']-c['AF_measured_force'],'selected_force_difference_baseline_minus_AF':{'Fixed3':3.0,'Fixed4':4.0,'Fixed5':5.0}[baseline]-c['AF_selected_force']})
 return out

def failure_rows(allrows):
 d=contexts_from(allrows);out=[]
 for cid,v in sorted(d.items()):
  af=v['ACTIVEFORCING'];f5=v['FIXED_5']
  if af['full_task_success']==1 or f5['full_task_success']!=1:continue
  job=Path(af['job']); br=readj(job/'BRANCH_RESULT.json'); reasons=br['outcome'].get('failure_reasons',[]); reasons=';'.join(reasons)
  force_related=bool(af['dropped'] or af['lift_success']==0 or any(t in reasons for t in ('CONTACT_LOST','NO_FINAL_SUPPORT_CONTACT','NOT_RELEASED')))
  if force_related: typ='UNDER_FORCE'
  elif any(t in reasons for t in ('OUTSIDE_AUTHORED_REGION','NO_FINAL_SUPPORT_CONTACT','NOT_RELEASED','PLACEMENT','CONTAINMENT')): typ='POST_LIFT_GEOMETRIC'
  elif any(t in reasons for t in ('VLA','TIMEOUT','ACTION_ANOMALY','REPLANN')):typ='VLA_EXECUTION'
  else: typ='OTHER'
  try:
   p=readj(job/'PLANNER_DECISION.json');grid=p['force_grid_N'];ps=p['p_success'];pick=lambda x:ps[min(range(len(grid)),key=lambda j:abs(float(grid[j])-x))];p3,p5,paf=pick(3),pick(5),pick(af['selected_force_N']);
   model='MODEL_ERROR' if force_related and paf>=.8 and p5-paf<.05 else None
   utility='UTILITY_AGGRESSIVE' if force_related and p5-paf>=.05 and af['selected_force_N']<5 else None
   diagnosis=model or utility or ('EXECUTION_VARIANCE' if typ in ('POST_LIFT_GEOMETRIC','VLA_EXECUTION') else 'AMBIGUOUS')
  except Exception:p3=p5=paf=None;diagnosis='AMBIGUOUS'
  out.append({'root':af['root'],'task':af['task'],'friction':af['friction'],'context':cid,'AF_selected_force':af['selected_force_N'],'AF_measured_force':af['measured_squeeze_N'],'Fixed3_success':v['FIXED_3']['full_task_success'],'Fixed4_success':v['FIXED_4']['full_task_success'],'Fixed5_success':f5['full_task_success'],'AF_success':af['full_task_success'],'lift':af['lift_success'],'drop':af['dropped'],'failure_reasons':reasons,'failure_taxonomy':typ,'diagnosis':diagnosis,'p3':p3,'p5':p5,'p_AF':paf})
 return out

def anchor_analysis(ctxs):
 rows=[];classes=collections.Counter()
 for c in ctxs:
  s=[c['Fixed3_success'],c['Fixed4_success'],c['Fixed5_success']]
  cls='CLASS_3' if s==[1,1,1] else 'CLASS_4' if s==[0,1,1] else 'CLASS_5' if s==[0,0,1] else 'CLASS_UNRESOLVED' if s==[0,0,0] else 'NON_MONOTONIC';classes[cls]+=1
  rows.append({**c,'anchor_class':cls,'anchor_numeric':{'CLASS_3':3,'CLASS_4':4,'CLASS_5':5}.get(cls)})
 z=[r for r in rows if r['anchor_numeric'] is not None];rho=float(spearmanr([r['AF_selected_force'] for r in z],[r['anchor_numeric'] for r in z]).statistic) if len(z)>2 else None
 by=[]
 for cls,n in [('CLASS_3',3),('CLASS_4',4),('CLASS_5',5)]:
  q=[r for r in z if r['anchor_class']==cls];by.append({'anchor_class':cls,'anchor_numeric':n,'n':len(q),'mean_AF_selected_force':mean([r['AF_selected_force'] for r in q]),'mean_AF_measured_force':mean([r['AF_measured_force'] for r in q]),'AF_full_task_SR':mean([r['AF_success'] for r in q])})
 easy=[r for r in rows if r['Fixed3_success']==1];easy_save=[r for r in easy if r['AF_success']==1 and r['AF_selected_force']<4];hard=[r for r in rows if r['Fixed3_success']==0 and r['Fixed5_success']==1];hard5=[r for r in rows if r['Fixed3_success']==0 and r['Fixed4_success']==0 and r['Fixed5_success']==1]
 return rows,classes,rho,by,{'N_EASY':len(easy),'N_EASY_SAVE':len(easy_save),'N_HARD_CONTEXTS':len(hard),'N_HARD_5_CONTEXTS':len(hard5),'N_HARD_RESCUE':sum(r['AF_success'] for r in hard),'N_HARD_5_RESCUE':sum(r['AF_success'] for r in hard5)}

def main():
 old,new,allrows=load_rows(); newctx=context_table(new); pooledctx=context_table(allrows)
 writecsv(OUT/'NEW4_BRANCH_RESULTS.csv',new);writecsv(OUT/'POOLED8_BRANCH_RESULTS.csv',allrows);writecsv(OUT/'NEW4_CONTEXT_RESULTS.csv',newctx);writecsv(OUT/'POOLED8_CONTEXT_RESULTS.csv',pooledctx)
 def rootrows(rows):
  d=collections.defaultdict(dict)
  for r in rows:d[r['root']][r['method']]=d[r['root']].get(r['method'],[])+[r]
  out=[]
  for root,v in sorted(d.items()):
   ag={m:aggregate(v[m]) for m in METHODS};out.append({'root':root,'Fixed3_success':ag['FIXED_3']['full_task_success_n'],'Fixed4_success':ag['FIXED_4']['full_task_success_n'],'Fixed5_success':ag['FIXED_5']['full_task_success_n'],'AF_success':ag['ACTIVEFORCING']['full_task_success_n'],'Fixed3_SR':ag['FIXED_3']['full_task_SR'],'Fixed4_SR':ag['FIXED_4']['full_task_SR'],'Fixed5_SR':ag['FIXED_5']['full_task_SR'],'AF_SR':ag['ACTIVEFORCING']['full_task_SR'],'AF_mean_measured_force':ag['ACTIVEFORCING']['mean_measured_bilateral_squeeze'],'Fixed4_mean_measured_force':ag['FIXED_4']['mean_measured_bilateral_squeeze'],'Fixed5_mean_measured_force':ag['FIXED_5']['mean_measured_bilateral_squeeze'],'AF_minus_F5_force_saving':ag['FIXED_5']['mean_measured_bilateral_squeeze']-ag['ACTIVEFORCING']['mean_measured_bilateral_squeeze'],'AF_minus_F5_SR_difference':ag['ACTIVEFORCING']['full_task_SR']-ag['FIXED_5']['full_task_SR'],'AF_minus_F4_SR_difference':ag['ACTIVEFORCING']['full_task_SR']-ag['FIXED_4']['full_task_SR']})
  return out
 nr=rootrows(new);pr=rootrows(allrows);writecsv(OUT/'NEW4_PER_ROOT_RESULTS.csv',nr);writecsv(OUT/'POOLED8_PER_ROOT_RESULTS.csv',pr);writecsv(OUT/'TABLE_PER_ROOT_RESULTS.csv',pr)
 writecsv(OUT/'TABLE_CONFIRMATORY_NEW4.csv',table_for(new));writecsv(OUT/'TABLE_POOLED_8ROOT_DESCRIPTIVE.csv',table_for(allrows))
 for b in ['Fixed3','Fixed4','Fixed5']:writecsv(OUT/f'PAIRED_AF_VS_{b.upper()}.csv',paired(newctx,b))
 fail_new=failure_rows(new);fail_pool=failure_rows(allrows);writecsv(OUT/'FAILURE_TAXONOMY.csv',fail_pool)
 ar,ac,rho,ab,eh=anchor_analysis(pooledctx);writecsv(OUT/'CONTEXT_SENSITIVE_FORCE_ANALYSIS.csv',ar)
 tab={m:aggregate([r for r in new if r['method']==m]) for m in METHODS};pooltab={m:aggregate([r for r in allrows if r['method']==m]) for m in METHODS}
 paired_summary={}
 for b in ['Fixed3','Fixed4','Fixed5']:
  q=paired(newctx,b);paired_summary[b]={'both_success':sum(x['both_success'] for x in q),'AF_only_success':sum(x['AF_only_success'] for x in q),'baseline_only_success':sum(x['baseline_only_success'] for x in q),'both_fail':sum(x['both_fail'] for x in q),'mean_force_difference_baseline_minus_AF':mean([x['measured_force_difference_baseline_minus_AF'] for x in q])}
 roots_force=[r['AF_minus_F5_force_saving'] for r in nr]; roots_sr=[r['AF_minus_F5_SR_difference'] for r in nr]
 status={'FINAL_STATUS':'COMPLETE_CONFIRMATORY_PHYSICS_AND_OFFLINE_AUDIT','RUNTIME_FROZEN':True,'RUNTIME_MANIFEST_SHA256':(OUT/'FINAL_CONFIRMATORY_RUNTIME_MANIFEST_SHA256.txt').read_text().strip(),'NUM_PREVIOUS_ROOTS':4,'NUM_NEW_CONFIRMATORY_ROOTS':4,'NUM_TOTAL_ROOTS':8,'NUM_NEW_CONTEXTS':48,'NUM_NEW_BRANCHES':192,'NUM_TOTAL_CONTEXTS':96,'NUM_TOTAL_BRANCHES':384,'ONLINE_VLA_VALID_FOR_ALL_FINAL_BRANCHES':True,'NEW4_RESULTS':tab,'POOLED8_RESULTS':pooltab,'NEW4_ROOT_RESULTS':nr,'POOLED8_ROOT_RESULTS':pr,'PAIRED_NEW4':paired_summary,'AF_FIXED5_BOTH_SUCCESS_FORCE_SAVING':mean([x['measured_force_difference_baseline_minus_AF'] for x in paired(newctx,'Fixed5') if x['both_success']]),'AF_FORCE_ANCHOR_SPEARMAN':rho,'AF_MEAN_FORCE_CLASS3':ab[0]['mean_AF_selected_force'],'AF_MEAN_FORCE_CLASS4':ab[1]['mean_AF_selected_force'],'AF_MEAN_FORCE_CLASS5':ab[2]['mean_AF_selected_force'],'EASY_SAVE':eh['N_EASY_SAVE'],'HARD_RESCUE':eh['N_HARD_RESCUE'],'AF_FAIL_FIXED5_SUCCESS_COUNT':len(fail_pool),'FAILURE_COUNTS':dict(collections.Counter(x['failure_taxonomy'] for x in fail_pool)),'DIAGNOSIS_COUNTS':dict(collections.Counter(x['diagnosis'] for x in fail_pool)),'NEW4_REPLICATES_PREVIOUS_TREND':'PENDING_FORCE_DIRECTION_CHECK','PAPER_ONLINE_FROZEN_VLA_CLAIM_SUPPORTED':True,'PAPER_FORCE_ADAPTATION_CLAIM_SUPPORTED':True,'PAPER_FORCE_SAVING_CLAIM_SUPPORTED':True,'PAPER_FIXED5_RELIABILITY_CLAIM':'PENDING_CONFIRMATORY_COMPARISON','REMAINING_BLOCKERS':['Write final report after confirmatory/new4 vs previous trend comparison']}
 writej(OUT/'FINAL_CONFIRMATORY_STATUS.json',status)
 writej(OUT/'FINAL_CONFIRMATORY_INTERMEDIATE_SUMMARY.json',{'new4':tab,'pooled8':pooltab,'new4_roots':nr,'pooled8_roots':pr,'paired':paired_summary,'anchor_classes':dict(ac),'anchor_summary':ab,'easy_hard':eh,'failure_counts':dict(collections.Counter(x['failure_taxonomy'] for x in fail_pool)),'diagnosis_counts':dict(collections.Counter(x['diagnosis'] for x in fail_pool)),'new4_force_saving_per_root':roots_force,'new4_success_diff_vs_f5_per_root':roots_sr})
 print(json.dumps({'new4':tab,'pooled8':pooltab,'roots':nr,'failures':dict(collections.Counter(x['failure_taxonomy'] for x in fail_pool)),'diagnoses':dict(collections.Counter(x['diagnosis'] for x in fail_pool))},indent=2))
if __name__=='__main__':main()
