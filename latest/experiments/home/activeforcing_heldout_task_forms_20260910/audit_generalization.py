"""Reconcile frozen row grain and training/evaluation overlap, without new inference."""
from pathlib import Path
import csv,json,hashlib
from collections import Counter,defaultdict
P=Path(__file__).resolve().parent
B=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907')
F=Path('/home/exouser/FORTE/analysis/results')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=list(csv.DictReader((B/'confirmatory_v1/POOLED8_BRANCH_RESULTS.csv').open()))
assert len(rows)==384 and len({(r['context'],r['method']) for r in rows})==384
contexts={r['context'] for r in rows};assert len(contexts)==96
assert all(len([r for r in rows if r['context']==c])==4 for c in contexts)
training=read(F/'current_fulltask_feasibility_baseline_v1_20260906/TRAINING_PROTOCOL.json')
stage=read(F/'current_matched_stage1_648_v1_20260906/STAGE_MANIFEST.json')
train=[c for c in stage['contexts'] if stage['split_by_context'][c['id']]=='TRAIN']
assert {c['task'] for c in train}=={0,1,5,6}
trainroots={c['root'] for c in train};evalroots={int(r['root']) for r in rows}
assert not trainroots&evalroots
stats={}
for method in ['ACTIVEFORCING','FIXED_3','FIXED_4','FIXED_5']:
    rr=[r for r in rows if r['method']==method]
    stats[method]={'n':len(rr),'successes':sum(int(r['full_task_success']) for r in rr),
        'success_rate':sum(int(r['full_task_success']) for r in rr)/len(rr),
        'mean_measured_squeeze_N':sum(float(r['measured_squeeze_N']) for r in rr)/len(rr)}
support=read(B/'final_continuous_friction_generalization_v1/TRAINING_FRICTION_SUPPORT_AUDIT.json')
unseen=read(B/'final_continuous_friction_generalization_v1/DEV_PLAN.json')['contexts']
checks=[]
for c in unseen:
    key=str(c['task']);bf=support['BELIEF_TRAIN_FRICTION_VALUES_PER_TASK'][key];ff=support['FEASIBILITY_TRAIN_FRICTION_VALUES_PER_TASK'][key]
    checks.append({'id':c['id'],'task':c['task'],'root':c['root'],'mu':c['mu'],
        'absent_belief_train':c['mu'] not in bf,'absent_feasibility_train':c['mu'] not in ff,
        'inside_belief_train_minmax':min(bf)<c['mu']<max(bf),'inside_feasibility_train_minmax':min(ff)<c['mu']<max(ff)})
assert all(all(c[k] for k in ['absent_belief_train','absent_feasibility_train','inside_belief_train_minmax','inside_feasibility_train_minmax']) for c in checks)
assert len(unseen)==48
verified=0
for field in ['belief_training_example_hashes','feasibility_training_row_hashes']:
    for path,digest in support[field].items():assert sha(path)==digest;verified+=1
old=Path('/media/volume/newdata/exouser/Tabero_e3lh/analysis/results/task_demand_loto_generalization_20260830_044334')
historical=read(old/'TASK_DEMAND_FINAL_DECISION.json')
receipt={'main_contexts':96,'main_branches':384,'unique_matched_rows':True,'training_roots':sorted(trainroots),'main_roots':sorted(evalroots),
    'root_overlap':[],'training_tasks':[0,1,5,6],'main_tasks':sorted({int(r['task']) for r in rows}),
    'training_row_hashes_verified':verified,'friction_exact_holdout_checks':checks,'main_FORM_A_results':stats,
    'archived_task_loto_exists':True,'archived_task_loto_current_model':False,'archived_task_loto_decision':historical,
    'current_task_heldout':False,'current_task_form_heldout':False}
(P/'evidence/GENERALIZATION_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
table='| Method | Existing form-A success | Measured squeeze |\n|---|---:|---:|\n'
for k,v in stats.items():table+=f"| {k} | {v['successes']}/96 ({v['success_rate']:.2%}) | {v['mean_measured_squeeze_N']:.3f} N |\n"
(P/'EXISTING_GENERALIZATION_AUDIT.md').write_text('''# Existing generalization audit

The completed main online-VLA evidence is **96 matched contexts / 384 branches / 8 roots**, obtained by pooling two separately frozen 48-context campaigns. The older `final_main_results_v1` is only the first half. Row uniqueness and four-method matching were checked from `confirmatory_v1/POOLED8_BRANCH_RESULTS.csv`.

## A. Root generalization — supported

Feasibility TRAIN roots are 5100,5101,5102,5106 (431 valid rows, 48 contexts); VAL root 6100 and historical TEST root 6103 contain the same task IDs. Main evaluation roots 170040–170047 are disjoint, as are friction-study roots 170052–170053. Fresh-root exposure manifests were frozen before those experiments. This is new reset/context evidence within the same task family, not unseen tasks.

## B. Exact-held-out friction generalization — supported

The completed continuous-friction study contains 4 known tasks × 6 exact-unseen in-support values × 2 fresh roots = 48 contexts, with AF, GT-Physics and Fixed-4 (144 branches). Every test friction was checked absent from both recorded belief and feasibility TRAIN values for that task, and strictly inside their recorded min/max support. Training-row hashes in the archived support audit were rechecked. The variable is object-side material static/dynamic friction, not a measured effective contact-pair coefficient. There is no mass intervention and no new task identity in this evidence.

## C. Object generalization — not demonstrated by the current frozen model's formal evaluations

TRAIN, the pooled main experiment and continuous-friction experiment all manipulate alphabet soup, cream cheese, tomato sauce and butter. Different roots/frictions do not make those unseen objects. The configured Tabero task list includes additional objects, but configured evaluation coverage is not held-out AF evidence.

## D. Task generalization — historical experiment exists; current frozen model evidence does not

An archived August 30 `task_demand_loto_generalization_20260830_044334` experiment explicitly holds each of tasks 0,1,5,6 out of that fold's training and trains different task-demand models. Its `TASK_DEMAND_LOTO_PROTOCOL.json`, training manifest, checkpoint manifest and results are present. Its final decision is `T1_REMAINS_STRUCTURALLY_OUT_OF_DISTRIBUTION`, with both feasibility and joint task-demand go gates false. Thus it would be inaccurate to say no task-held-out work was ever attempted anywhere. It is a different historical architecture/data/runtime, and it does not supply a held-out task result for the authoritative September 6 phase-free ensemble. All four task identities occur in that ensemble's TRAIN set.

## E. Task-form generalization — not previously demonstrated

Both the current evaluations and archived four-task LOTO work remain within lift/transport/place. Scripted task-demand phase descriptors in the historical experiment explicitly cover branch_hold/lift/transit/over_basket/place. Archived generic LIBERO tasks or demonstrations in other projects do not demonstrate transfer by the current frozen AF ensemble.

## Existing-family reference results, never new-form results

'''+table+'''
The squeeze statistic is the archived mean over non-release observations, averaged by branch; it is not the selected setpoint and not the bilateral-contact-conditional statistic. Existing-family paired comparisons are not evidence about new task forms.

## Sources and reproducibility

- `/home/exouser/FORTE/analysis/results/current_fulltask_feasibility_baseline_v1_20260906/TRAINING_PROTOCOL.json` and `current_matched_stage1_648_v1_20260906/STAGE_MANIFEST.json`.
- `/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1/{POOLED8_BRANCH_RESULTS.csv,FINAL_CONFIRMATORY_STATUS.json,FINAL_FRESH_ROOT_PLAN.json}`.
- `/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/{TRAINING_FRICTION_SUPPORT_AUDIT.json,DEV_PLAN.json,ROOT_NONEXPOSURE_AUDIT.json,FINAL_CONTINUOUS_FRICTION_RESULTS.json}`.
- `/media/volume/newdata/exouser/Tabero_e3lh/analysis/results/task_demand_loto_generalization_20260830_044334/`.

`audit_generalization.py` reproduces counts and exact-value overlap checks; `evidence/GENERALIZATION_VERIFICATION.json` contains the per-context audit. Historical LOTO existence must be distinguished from a supported positive generalization claim.
''')
print(json.dumps({'main_contexts':96,'main_branches':384,'verified_training_rows':verified,'exact_heldout_friction_contexts':48,'historical_task_LOTO':True}))
