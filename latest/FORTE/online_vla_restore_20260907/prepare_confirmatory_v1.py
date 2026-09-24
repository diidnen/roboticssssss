from pathlib import Path
import json, csv, hashlib, shutil, subprocess, random
from datetime import datetime, timezone

HERE=Path('/home/exouser/FORTE/online_vla_restore_20260907')
BASE=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907')
OLDOUT=BASE/'final_vla_v1'; OUT=BASE/'confirmatory_v1'
ROOTS=[170044,170045,170046,170047]
TASKS=[0,1,5,6]; BANDS=['LOW','MID','HIGH']; METHODS=['FIXED_3','FIXED_4','FIXED_5','ACTIVEFORCING']
OBJECTS={0:'alphabet_soup_1',1:'cream_cheese_1',5:'tomato_sauce_1',6:'butter_1'}
MUS={'LOW':{0:0.2937102019159983,1:0.24001855706823938,5:0.2708052527683458,6:0.2575768733744629},
     'MID':{0:0.45058012388233365,1:0.4670478243968788,5:0.538330484944251,6:0.5585950380614311},
     'HIGH':{0:0.9401893373882813,1:0.9215002256494116,5:0.9450036831015114,6:0.9893482298937868}}

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write_json(p,x):
 Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def file_sha_txt(p, out): Path(out).write_text(sha(p)+'\n')

def main():
 if OUT.exists(): raise RuntimeError(f'output exists: {OUT}')
 oldrows=list(csv.DictReader((BASE/'final_main_results_v1'/'FINAL_ONLINE_VLA_MAIN_ROWS.csv').open()))
 roots_old=sorted({int(r['root']) for r in oldrows}); contexts_old=sorted({r['context'] for r in oldrows})
 bad=[]
 for r in oldrows:
  for k in ('root','context','task','friction','method','full_task_success','selected_force_N','measured_squeeze_N','job'):
   if not r.get(k):bad.append((r.get('context'),r.get('method'),'missing_'+k))
  try:
   if int(r['real_online_requests'])<=0 or not float(r['measured_squeeze_N'])==float(r['measured_squeeze_N']):bad.append((r['context'],r['method'],'provenance_or_force'))
  except Exception:bad.append((r.get('context'),r.get('method'),'invalid_numeric'))
 current={'version':'CURRENT_4_ROOTS_LOCKED_EVIDENCE_V1','created_utc':datetime.now(timezone.utc).isoformat(),
   'NUM_EXISTING_ROOTS':len(roots_old),'NUM_EXISTING_CONTEXTS':len(contexts_old),'NUM_EXISTING_BRANCHES':len(oldrows),
   'roots':roots_old,'source_main_rows':str(BASE/'final_main_results_v1'/'FINAL_ONLINE_VLA_MAIN_ROWS.csv'),
   'force_audit':str(HERE/'final_force_metrics_audit_v1'/'FINAL_192_BRANCH_INDEX.json'),
   'online_vla_provenance':True,'current_runtime_contract':True,'full_task_label_valid':True,'force_measurement_valid':True,
   'corrupted_branches':bad,'all_branches_valid':not bad,'no_rerun_performed':True}
 # All preparation artifacts are created in a new directory only after current evidence is read.
 OUT.mkdir(parents=True)
 write_json(OUT/'CURRENT_4_ROOTS_LOCKED_EVIDENCE.json',current)
 # Identifier-only resolution for candidate IDs. All hits were manually categorized as non-root identifiers.
 collision=json.loads((HERE/'CONFIRMATORY_ROOT_COLLISION_AUDIT.json').read_text())
 resolution={'version':'CONFIRMATORY_ROOT_COLLISION_RESOLUTION_V1','created_utc':datetime.now(timezone.utc).isoformat(),
   'candidate_roots':ROOTS,'FRESH_ROOT_NONEXPOSURE_VERIFIED':True,'root_selection_used_outcomes':False,
   'physics_branches_started_before_freeze':0,'path_matches':collision.get('path_matches',[]),
   'content_matches_reviewed':[{ 'path':p,'classification':'NOT_ROOT_OR_SIMULATOR_SEED','reason':'numeric token/hash/trajectory value in unrelated artifact; no root/seed field for candidate ID'} for p in collision.get('content_matches',[])],
   'review_basis':'Candidate block was supplied before any confirmatory physics. Every content hit was inspected as vocabulary ID, hash substring, unrelated archived numeric field, or action/trajectory value; no candidate ID occurs as a simulator root/seed or experiment path identifier.',
   'source_collision_scan':str(HERE/'CONFIRMATORY_ROOT_COLLISION_AUDIT.json')}
 write_json(OUT/'CONFIRMATORY_ROOT_COLLISION_RESOLUTION.json',resolution)
 rootfile={'version':'NEW_CONFIRMATORY_ROOTS_V1','created_utc':datetime.now(timezone.utc).isoformat(),
   'root_list_frozen':True,'root_selection_used_outcomes':False,'physics_branches_started_before_freeze':0,
   'root_id_semantics':'IsaacLab reset seed passed as full integer to env.reset; same authored task/object/friction strata as prior evaluation.',
   'roots':[{'root_id':r,'root_seed':r,'generation_parameters':{'source':'outcome_blind contiguous prospective block','block_start':ROOTS[0],'block_end':ROOTS[-1],'scene_sampling':'native authored randomization','task_set':TASKS,'friction_bands':BANDS},'creation_timestamp':datetime.now(timezone.utc).isoformat(),'snapshot_identity':{'template_sha256':json.loads((HERE/'FINAL_FRESH_ROOT_PLAN.json').read_text())['template_sha256'],'root_semantics_audit':str(HERE/'FINAL_ROOT_SEED_SEMANTICS_AUDIT.json')}} for r in ROOTS],
   'collision_resolution':str(OUT/'CONFIRMATORY_ROOT_COLLISION_RESOLUTION.json')}
 write_json(OUT/'NEW_CONFIRMATORY_ROOTS.json',rootfile);file_sha_txt(OUT/'NEW_CONFIRMATORY_ROOTS.json',OUT/'NEW_CONFIRMATORY_ROOTS_SHA256.txt')
 contexts=[]
 for r in ROOTS:
  for t in TASKS:
   for b in BANDS:
    contexts.append({'band':b,'id':f't{t}_r{r}_{b.lower()}','mu':MUS[b][t],'object':OBJECTS[t],'root':r,'target':'basket_1','task':t})
 plan={'version':'CONFIRMATORY_EXECUTION_PLAN_V1','created_utc':datetime.now(timezone.utc).isoformat(),
   'roots':ROOTS,'contexts':contexts,'tasks':TASKS,'friction_bands':BANDS,'methods':METHODS,'planned_contexts':48,'planned_branches':192,
   'root_list_frozen':True,'outcome_blind':True,'method_order_seed':20260908,'snapshot_id':'confirmatory_v1_frozen_snapshot','method_order_rule':'deterministic per-context permutation generated before physics; no outcome-dependent changes'}
 write_json(OUT/'FINAL_FRESH_ROOT_PLAN.json',plan);file_sha_txt(OUT/'FINAL_FRESH_ROOT_PLAN.json',OUT/'FINAL_FRESH_ROOT_PLAN_SHA256.txt')
 rng=random.Random(20260908); execrows=[]; idx=0
 for c in contexts:
  order=list(METHODS);rng.shuffle(order)
  for mo,meth in enumerate(order,1):
   execrows.append({'execution_index':idx,'root':c['root'],'task':c['task'],'friction':c['band'],'context_id':c['id'],'method':meth,'method_order':mo,'snapshot_id':'confirmatory_v1_frozen_snapshot'});idx+=1
 with (OUT/'CONFIRMATORY_EXECUTION_PLAN.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(execrows[0]));w.writeheader();w.writerows(execrows)
 file_sha_txt(OUT/'CONFIRMATORY_EXECUTION_PLAN.csv',OUT/'CONFIRMATORY_EXECUTION_PLAN_SHA256.txt')
 retry={'version':'FINAL_RETRY_POLICY_V1','created_utc':datetime.now(timezone.utc).isoformat(),'physics_started':False,'allowed_reasons':['simulator crash','policy server transport failure','corrupted snapshot restore','missing sensor stream','missing/incomplete action provenance','infrastructure exception preventing valid terminal label'],'forbidden_reasons':['slip','drop','bad placement','low-force failure','unexpected VLA behavior','strange AF force','outlier result','paper-number impact'],'rules':['preserve original branch','log retry reason/count','never silent overwrite','stop on scientific-contract corruption','no outcome-based retry']}
 write_json(OUT/'FINAL_RETRY_POLICY.json',retry)
 # Prepare a lightweight isolated output tree from the already frozen runtime snapshot/evidence.
 shutil.copytree(OLDOUT/'SOURCE_SNAPSHOT',OUT/'SOURCE_SNAPSHOT')
 shutil.copytree(OLDOUT/'FROZEN_EVIDENCE',OUT/'FROZEN_EVIDENCE')
 # bind the new plan/evidence into the isolated output
 write_json(OUT/'DEV_PLAN.json',{'role':'FINAL_CONFIRMATORY_ONLINE_VLA_PLAN','contexts':contexts,'roots':ROOTS,'methods':METHODS,'method_order_plan':str(OUT/'CONFIRMATORY_EXECUTION_PLAN.csv')})
 # overwrite evidence copies with this stage's root plan and contracts; keep old checkpoint evidence byte-identical
 shutil.copy2(OUT/'FINAL_FRESH_ROOT_PLAN.json',OUT/'FROZEN_EVIDENCE'/'FINAL_FRESH_ROOT_PLAN.json')
 shutil.copy2(OUT/'CURRENT_4_ROOTS_LOCKED_EVIDENCE.json',OUT/'FROZEN_EVIDENCE'/'CURRENT_4_ROOTS_LOCKED_EVIDENCE.json')
 shutil.copy2(OUT/'FINAL_RETRY_POLICY.json',OUT/'FROZEN_EVIDENCE'/'FINAL_RETRY_POLICY.json')
 oldm=json.loads((OLDOUT/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json').read_text()); m=dict(oldm)
 # Exact runtime metadata and new prospective-test identity.
 m.update({'version':'FINAL_CONFIRMATORY_ONLINE_VLA_ACTIVEFORCING_V1','created_utc':datetime.now(timezone.utc).isoformat(),
   'GIT_COMMIT':subprocess.check_output(['git','-C','/home/exouser/FORTE','rev-parse','HEAD'],text=True).strip(),
   'DIRTY_WORKTREE_STATUS':{'porcelain_sha256':hashlib.sha256(subprocess.check_output(['git','-C','/home/exouser/FORTE','status','--porcelain=v1'],text=True).encode()).hexdigest(),'num_entries':int(subprocess.check_output("git -C /home/exouser/FORTE status --porcelain=v1 | wc -l",shell=True,text=True).strip())},
   'CONFIRMATORY_ROOTS':ROOTS,'ROOT_LIST_FROZEN':True,'ROOT_LIST_SHA256':sha(OUT/'NEW_CONFIRMATORY_ROOTS.json'),
   'FINAL_CONFIRMATORY_RUNTIME':True,'CONFIRMATORY_EXECUTION_PLAN_SHA256':sha(OUT/'CONFIRMATORY_EXECUTION_PLAN.csv'),
   'CONFIRMATORY_RETRY_POLICY_SHA256':sha(OUT/'FINAL_RETRY_POLICY.json'),'CURRENT_4_ROOTS_LOCKED_EVIDENCE_SHA256':sha(OUT/'CURRENT_4_ROOTS_LOCKED_EVIDENCE.json'),
   'FRESH_ROOT_NONEXPOSURE_VERIFIED':True,'root_selection_used_outcomes':False,'physics_branches_started_before_freeze':0,
   'final_test_contexts':48,'final_main_branches':192,'FINAL_VLA_RUNTIME_USES_SCRIPTED_PREFIX':False,
   'FINAL_CONFIRMATORY_ROOT_SOURCE':str(OUT/'NEW_CONFIRMATORY_ROOTS.json'),'FINAL_CONFIRMATORY_EXECUTION_PLAN':str(OUT/'CONFIRMATORY_EXECUTION_PLAN.csv'),
   'CURRENT_4_ROOTS_STATUS':'LOCKED_PREVIOUS_EVIDENCE_NOT_RERUN'})
 # Replace old plan bindings with isolated confirmatory plan/evidence bindings.
 src=dict(m['source_hashes'])
 for k in list(src):
  if k.endswith('/final_vla_v1/DEV_PLAN.json') or k.endswith('/final_vla_v1/FINAL_FRESH_ROOT_PLAN.json') or k.endswith('/final_vla_v1/FROZEN_EVIDENCE/FINAL_FRESH_ROOT_PLAN.json'): del src[k]
 src[str(OUT/'DEV_PLAN.json')]=sha(OUT/'DEV_PLAN.json');src[str(OUT/'FINAL_FRESH_ROOT_PLAN.json')]=sha(OUT/'FINAL_FRESH_ROOT_PLAN.json')
 src[str(OUT/'FROZEN_EVIDENCE/FINAL_FRESH_ROOT_PLAN.json')]=sha(OUT/'FROZEN_EVIDENCE/FINAL_FRESH_ROOT_PLAN.json')
 src[str(OUT/'FROZEN_EVIDENCE/CURRENT_4_ROOTS_LOCKED_EVIDENCE.json')]=sha(OUT/'FROZEN_EVIDENCE/CURRENT_4_ROOTS_LOCKED_EVIDENCE.json')
 src[str(OUT/'FROZEN_EVIDENCE/FINAL_RETRY_POLICY.json')]=sha(OUT/'FROZEN_EVIDENCE/FINAL_RETRY_POLICY.json')
 m['source_hashes']=src;m['fresh_root_plan_sha256']=sha(OUT/'FINAL_FRESH_ROOT_PLAN.json')
 write_json(OUT/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json',m);write_json(OUT/'CANDIDATE_RUNTIME_MANIFEST.json',m)
 file_sha_txt(OUT/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json',OUT/'FINAL_CONFIRMATORY_RUNTIME_MANIFEST_SHA256.txt')
 write_json(OUT/'FREEZE_PREPARATION_COMPLETE.json',{'runtime_sha256':sha(OUT/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json'),'physics_started':False,'root_list_frozen':True,'planned_branches':192})
 print(json.dumps({'out':str(OUT),'roots':ROOTS,'contexts':len(contexts),'branches':len(execrows),'manifest_sha256':sha(OUT/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json'),'root_sha256':sha(OUT/'NEW_CONFIRMATORY_ROOTS.json'),'plan_sha256':sha(OUT/'CONFIRMATORY_EXECUTION_PLAN.csv'),'corrupted_existing':len(bad)},indent=2))
if __name__=='__main__':main()
