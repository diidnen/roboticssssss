"""Build the negative-admission report only after all frozen screens finish.

This builder deliberately refuses to finalize if a candidate passes screening;
such a case requires further handoff/evaluator qualification and formal work.
"""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,hashlib,shutil
from collections import Counter
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x): (P/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
tasks=[('libero_10',5,'Book to back compartment','FORM_C_CANDIDATE'),('libero_goal',9,'Wine bottle to rack','FORM_C_CANDIDATE'),('libero_spatial',4,'Bowl from open drawer to plate','FORM_D_CANDIDATE')]
qual=[];allrows=[]
for suite,task,name,form in tasks:
    paths=sorted((P/'qualification').glob(f'{suite}_{task}_*/QUALIFICATION_RESULT.json'))
    outcomes=[read(p) for p in paths]
    assert all(o['valid_outcome'] for o in outcomes),'Invalid attempt needs separate investigation'
    success=sum(o['full_screen_success'] for o in outcomes);fail=len(outcomes)-success
    assert fail>=2 or len(outcomes)==6,'Screen not complete'
    assert success<5,'Candidate passed; do not finalize without downstream qualification'
    qual.append(dict(task_id=f'{suite}:{task}',name=name,task_form=form,attempted=len(outcomes),success=success,failures=fail,
        not_run=6-len(outcomes),best_possible_successes_of_six=success+6-len(outcomes),
        established_grasps=sum(o['established_grasp'] for o in outcomes),qualified=False,
        rejection='FROZEN_NATIVE_RESET_SCREEN_THRESHOLD_UNATTAINABLE',failure_types=dict(Counter(o['failure'] for o in outcomes)),
        evidence=[str(p) for p in paths],supplied_grasp_downstream_competence='NOT_TESTED'))
    for path,o in zip(paths,outcomes):
        allrows.append({'task_id':f'{suite}:{task}','context_id':o['plan']['id'],'root':o['plan']['root'],'mu':o['plan']['mu'],
            'method':'FIXED_5','steps':o['steps'],'native_full_task_screen_success':o['full_screen_success'],
            'established_grasp':o['established_grasp'],'failure':o['failure'],'formal_AF_branch':False,'source':str(path)})
with (P/'FROZEN_VLA_QUALIFICATION_RESULTS.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)

selected={'version':'HELDOUT_TASK_FORM_SET_V1_EMPTY_AFTER_ADMISSION_SCREEN','frozen_utc':datetime.now(timezone.utc).isoformat(),
    'tasks':[],'task_ids':[],'instructions':[],'task_forms':[],'object_overlap_with_training':[],
    'form_overlap_with_training':[],'qualification_results':qual,
    'selection_rule':'No AF outcomes used. Require maintained-grasp scope, faithful runtime/evaluator and >=5/6 Fixed-5 DEV successes; stop after second valid failed native-reset screen. Prefer distinct forms and category C; report D separately.',
    'selection_result':'NO_ADMITTED_TASKS','qualification_scope':'native-reset VLA competence; supplied-grasp downstream competence remains NOT_TESTED',
    'AF_calls_on_new_tasks':0,'feasibility_retraining':False,'new_training_rows':0,
    'formal_contexts':0,'formal_branches':0,'formal_roots':[],
    'excluded_other_candidates':{'libero_goal:0':'articulated handle outside current rigid-object force/probe contract',
        'factory':'different controller/observation contract; frozen VLA support unverified',
        'surface_push':'not maintained-grasp dragging','multi_object_and_hand_mode_composites':'outside single-setpoint maintained-grasp scope'},
    'qualification_protocol_sha256':sha(P/'QUALIFICATION_PROTOCOL.md'),
    'qualification_execution_addendum_sha256':sha(P/'QUALIFICATION_EXECUTION_ADDENDUM.md')}
write('FINAL_HELDOUT_TASK_FORM_SET.json',selected)
(P/'FINAL_HELDOUT_TASK_FORM_SET_SHA256.txt').write_text(sha(P/'FINAL_HELDOUT_TASK_FORM_SET.json')+'  FINAL_HELDOUT_TASK_FORM_SET.json\n')

qt='| Task | Candidate form | Success / attempted | Unrun | Best possible out of six | Established grasps | Decision |\n|---|---|---:|---:|---:|---:|---|\n'
for q in qual:qt+=f"| {q['task_id']} — {q['name']} | {q['task_form']} | {q['success']}/{q['attempted']} | {q['not_run']} | {q['best_possible_successes_of_six']}/6 | {q['established_grasps']} | Not admitted |\n"
(P/'FROZEN_VLA_NEW_TASK_QUALIFICATION.md').write_text('''# Frozen VLA new-task qualification

**No new task passed admission.** These are native-reset Fixed-5 competence screens, not AF tests and not supplied-grasp downstream tests. Each task was stopped after its second valid failure under the rule frozen before VLA calls. Unrun contexts are not counted as failures. The empirical outcome rate is based only on attempted contexts; the 5/6 rule is an operational admission criterion, not a precise population-reliability estimate.

'''+qt+'''
The executed cases used development root 5100 at object-side friction 0.30 and 0.50. Root 6100 and the other friction conditions remain unrun after futility stopping. No fresh formal roots were exposed. The frozen VLA checkpoint digest was checked by the dedicated server and every inference receipt. All arm targets came from online inference at the existing ten-step cadence. Fixed-5 used the existing six-body sensor correction, force servo, and VLA-controlled opening rule. No feasibility or belief model was called.

The observed failure type is available per context in `FROZEN_VLA_QUALIFICATION_RESULTS.csv` and raw `qualification/*/QUALIFICATION_RESULT.json`. In this screen, failure to acquire a grasp within 200 VLA steps is separately labeled `VLA_PREGRASP_FAILURE`; it is not a force-related drop, release failure, geometric insertion failure, or evidence of failed feasibility transfer. There were no observed supplied-grasp downstream trials. The initial-state requirement is stronger than the method's assumed established-grasp handoff, so these negative screens cannot rule out competent downstream execution if a validated grasp is supplied.

Four native scenes passed config/observation preflight. The original extraction reset ignored the task's `joint_pos_range`, visibly leaving the intended drawer closed. Before any extraction VLA outcome, `extraction_reset_compatibility.py` restored the source-specified top joint default to -0.1523311883211136; a separate GPU preflight verified the named top joint and corrected frame. No threshold, force control, model, object root pose, or terminal criterion changed. The original queue was paused before extraction and its already completed book/wine outcomes retained.

Book/caddy and wine/rack native position/contact predicates do not certify compartment placement or target orientation. Even a positive native screen would need a stronger task-specific evaluator before formal admission. Drawer pulling cannot use the current rigid-object contact alias for an articulated handle. Factory insertion/gear/threading definitions exist but have a different controller/observation contract and no verified current-checkpoint qualification. Plate pushing and multi-grasp composites are out of scope.

Preflight teardown aborts occurred after complete scene/observation receipts; these were not task outcomes. The first sandbox preflight had no CUDA access and was repeated with GPU access. These infrastructure records remain in `preflight/` and `preflight_gpu/`. They do not enter the Fixed-5 denominator.

Sources: `QUALIFICATION_PROTOCOL.md`, `QUALIFICATION_EXECUTION_ADDENDUM.md`, `qualification/EXECUTION_FREEZE.json`, `qualification/EXTRACTION_COMPATIBILITY_FREEZE.json`, per-context RPC receipts, and `evidence/QUALIFICATION_QUEUE_COMPATIBILITY_PAUSE.json`.
''')

status={
 'CURRENT_MODEL_HAS_EXPLICIT_TASK_ID':'SUPPORTED',
 'ROOT_GENERALIZATION_ALREADY_EXISTS':'SUPPORTED',
 'HELDOUT_FRICTION_GENERALIZATION_ALREADY_EXISTS':'SUPPORTED',
 'HELDOUT_TASK_GENERALIZATION_PREVIOUSLY_EXISTS':'MIXED',
 'HELDOUT_TASK_FORM_GENERALIZATION_PREVIOUSLY_EXISTS':'NOT_TESTED',
 'NEW_TASK_FORMS_FOUND':'SUPPORTED',
 'FROZEN_VLA_COMPETENT_ON_NEW_FORMS':'NOT_SUPPORTED',
 'FEASIBILITY_ZERO_SHOT_TRANSFER_TO_NEW_FORMS':'NOT_TESTED',
 'FORCE_ADAPTATION_TRANSFER_TO_NEW_FORMS':'NOT_TESTED',
 'RELIABILITY_FORCE_TRADEOFF_TRANSFER_TO_NEW_FORMS':'NOT_TESTED',
 'GT_DECISION_FIDELITY_ON_NEW_FORMS':'NOT_TESTED',
 'GENERALIZATION_BEYOND_LIFT_TRANSPORT_PLACE':'NOT_TESTED',
 'GENERALIZATION_TO_REGRASP_TASKS':'NOT_TESTED',
 'UNIVERSAL_TASK_GENERALIZATION':'NOT_SUPPORTED'}
reasons={
 'CURRENT_MODEL_HAS_EXPLICIT_TASK_ID':'Verified four-way one-hot at phase-free columns 6:10; unseen IDs raise ValueError.',
 'ROOT_GENERALIZATION_ALREADY_EXISTS':'96 matched main contexts across eight fresh roots; TRAIN root overlap absent.',
 'HELDOUT_FRICTION_GENERALIZATION_ALREADY_EXISTS':'48 contexts with exact-unseen in-support friction; same four task identities.',
 'HELDOUT_TASK_GENERALIZATION_PREVIOUSLY_EXISTS':'Historical four-task LOTO experiment exists with mixed/failed gates, but uses a different architecture; no current-model held-out task evaluation.',
 'HELDOUT_TASK_FORM_GENERALIZATION_PREVIOUSLY_EXISTS':'No such current-model evaluation found in audited manifests.',
 'NEW_TASK_FORMS_FOUND':'Native insertion/gear and drawer-pulling definitions exist. This does not mean VLA competence or AF transfer.',
 'FROZEN_VLA_COMPETENT_ON_NEW_FORMS':'No candidate passed the frozen native-reset screen; supplied-grasp downstream competence remains untested.',
 'UNIVERSAL_TASK_GENERALIZATION':'Neither available evidence nor the restricted task-code/interface supports a universal claim.',
 'GENERALIZATION_TO_REGRASP_TASKS':'Outside current single-setpoint maintained-grasp experiment.'}
claims={k:{'status':v,'reason':reasons.get(k,'No new-form AF branches were admitted or executed.')} for k,v in status.items()}
write('HELDOUT_TASK_FORM_CLAIM_AUDIT.json',claims)
ref=read(P/'evidence/GENERALIZATION_VERIFICATION.json')['main_FORM_A_results']
rt='| Form / evidence role | Tasks | Contexts | AF | Fixed-3 | Fixed-4 | Fixed-5 |\n|---|---:|---:|---:|---:|---:|---:|\n'
rt+='| Original A, archived main evaluation | 4 | 96 | '+ ' | '.join(str(ref[k]['successes'])+'/96' for k in ['ACTIVEFORCING','FIXED_3','FIXED_4','FIXED_5'])+' |\n'
rt+='| New C / D candidates, formal experiment | 0 admitted | 0 | N/A | N/A | N/A | N/A |\n'
ct='| Claim | Status |\n|---|---|\n'+''.join(f'| {k} | {v} |\n' for k,v in status.items())
report='''# 1. Executive conclusion

**Held-out task-form transfer remains NOT_TESTED.** The authoritative model does contain an explicit four-way task one-hot. New-form scene definitions exist, but no screened task passed the frozen admission rule. Therefore the final task set is empty and no AF/Fixed formal branches were run. This is an upstream qualification limitation, not a demonstrated failure of feasibility transfer.

The native-reset screen adds a requirement the original method avoids by assuming an established grasp. Its negative outcomes do not answer whether the frozen VLA would succeed from a supplied common grasp. Do not present this report as a completed downstream task-form generalization experiment.

# 2. Current model-input audit

**Explicit task code: YES.** The phase-free `(8,64)` tensor contains six Cartesian command channels, four task-one-hot channels, candidate force, friction, duplicated 13D state and duplicated masks. Object velocity and lateral speed come from the simulator snapshot. Wrist orientation/rotation, target geometry and future VLA replans are not direct feasibility inputs. The three frozen models and pooled TRAIN normalizers are unchanged. `c_task` is paper notation, not a generic implemented semantic embedding. All 64 scalar channels and normalization values are listed in [CURRENT_FEASIBILITY_FEATURES.csv](CURRENT_FEASIBILITY_FEATURES.csv); the full audit is [CURRENT_FEASIBILITY_INPUT_AUDIT.md](CURRENT_FEASIBILITY_INPUT_AUDIT.md).

# 3. What generalization was already present

Root-held-out main evaluation: 96 contexts / 384 branches / eight roots. Exact-held-out in-support friction: 48 contexts / 144 branches / two roots. Both use training task IDs 0,1,5,6 and their same objects. Archived August 30 task LOTO exists, but is a different model/data/runtime with failed generalization gates within the same form. It does not establish current-model task or task-form transfer. See [EXISTING_GENERALIZATION_AUDIT.md](EXISTING_GENERALIZATION_AUDIT.md).

# 4. Available task inventory

The search ledger contains 842 inventory entries: 46 native Tabero config records, five other native task semantics, six identifier-only tasks, and 785 additional archived BDDL definitions/variants. These are availability tiers, not 842 runnable supported tasks. The configured Tabero evaluation list contains nine libero_object tasks. Local assembled HDF5s cover those nine objects plus bowl-to-stove, with 50 demonstrations each; these are not proof of current-checkpoint competence on new forms. Twenty-three Gym environment registrations include controller/sensor variants, which do not increase task-form diversity. See [ALL_AVAILABLE_TASKS.csv](ALL_AVAILABLE_TASKS.csv) and `evidence/task_search_scope.json` for roots, exclusions and search limits.

# 5. Task-form taxonomy

Native definitions provide lift/transport/place (A), insertion/constrained assembly (C), and pulling (D). Reorientation-heavy maintained grasp (B) is only a candidate interpretation of threading/rack tasks; no qualified B instance is established. Plate pushing is not grasped dragging (E). Knob manipulation (F) has unverified hand-control compatibility. Multi-object and close-after-place composites are excluded. Distances/rotations not measured from downstream execution remain unknown. New objects are not new forms. See [TASK_FORM_TAXONOMY.md](TASK_FORM_TAXONOMY.md) and [TASK_FORM_DIVERSITY_MATRIX.csv](TASK_FORM_DIVERSITY_MATRIX.csv).

# 6. VLA qualification

'''+qt+'''
All attempted screens used Fixed-5 and development roots, no AF. The rule was frozen at >=5/6; early rejection after two valid failures leaves four contexts unrun. No denominator of 0/6 is fabricated. Observed pregrasp failures do not identify downstream force sufficiency. See [FROZEN_VLA_NEW_TASK_QUALIFICATION.md](FROZEN_VLA_NEW_TASK_QUALIFICATION.md). The extraction reset compatibility fix was validated and frozen before its first VLA outcome.

# 7. Frozen held-out task-form set

`FINAL_HELDOUT_TASK_FORM_SET.json` contains **zero admitted tasks** and all qualification decisions. Its exact SHA256 is in `FINAL_HELDOUT_TASK_FORM_SET_SHA256.txt`. No task was selected using AF performance. No fresh formal roots were allocated or executed.

# 8. Training-overlap audit

All four original identities appear in current feasibility TRAIN (431 rows, 48 contexts; roots 5100,5101,5102,5106). Candidate book, wine bottle, bowl, drawer-handle and factory objects are absent as manipulated training objects. Their potential results would confound task form with object/native-mass differences and belong to category D. No clean category C task with a training-overlap object and a genuinely new single-grasp form was established. No new labels were added to training; no model was retrained. The explicit task-ID compatibility issue remains unresolved rather than silently removed or aliased.

# 9. Formal protocol

The conditional protocol is recorded in [FORMAL_PROTOCOL.md](FORMAL_PROTOCOL.md): three frozen in-support friction values × two fresh roots × Fixed-3/4/5/AF per qualified task, with common grasp/query/post-query state, identical online VLA cadence and evaluator. No mass intervention. It was **not executed** because admission failed. There is no selected new-task encoding, common P4B handoff, or finalized new-form evaluator to claim as validated.

# 10. Per-task-form results

'''+rt+'''
Only the original-family row is an AF result. Candidate-screen failures cannot populate new-form AF success rates. For B, C, D, E and F, AF performance remains N/A, with zero formal contexts. No form can be said to transfer successfully or fail feasibility transfer from this run.

# 11. Force-requirement adaptation

NOT_TESTED on new forms: no matched Fixed-3/4/5 outcomes exist, so no requirement class or Spearman correlation is computed. The frozen analysis retains unresolved 000 and non-monotonic patterns separately; it does not force a lift-based threshold model onto contact-rich tasks.

# 12. AF vs Fixed-5 / Fixed-4

NOT_TESTED on new forms. The archived original-family AF result is 69/96 versus Fixed-5 71/96 and Fixed-4 65/96. Archived mean non-release measured squeeze is 3.519 N for AF versus 4.689 N for Fixed-5; these numbers must not be reused as transfer results. No new-form paired differences or confidence intervals can be estimated.

# 13. AF vs GT decision diagnostic

NOT_TESTED: new-task feasibility inference was deliberately not called during qualification. AF–GT force MAE is N/A. The formal diagnostic would replace inferred friction with true friction while retaining the same model/context/search; it would remain decision-only and explicitly privileged.

# 14. Generalized failure analysis

The screen failures occurred before an established grasp. They are VLA pregrasp failures under the specified start-state/budget, not AF grasp-retention failures. AF failure counts are zero only because there are zero AF trials; corresponding rates are N/A. No causal conclusion about task contact, release, physical belief or feasibility transfer follows. [GENERALIZED_FAILURE_TAXONOMY.md](GENERALIZED_FAILURE_TAXONOMY.md) defines retention, task contact, VLA execution, terminal geometry, release and other categories without assuming a lift phase.

# 15. Visual task diversity

`HELDOUT_TASK_FORM_VISUAL_AUDIT/` preserves native scene frames, the corrected extraction reset, source hashes and qualification sequences. `TASK_FORM_OVERVIEW_CONTACT_SHEET.png` is explicitly a **candidate scene audit**, not evidence of successful behaviors or an admitted final benchmark. There is no representative successful final-form rollout to show because the final set is empty. The original closed-drawer reset is retained as diagnostic evidence, not a valid extraction scene.

# 16. Supported / mixed / unsupported claims

'''+ct+'''
The claim audit distinguishes existence of historical LOTO work from positive current-model transfer. `NOT_TESTED` must not be rewritten as either success or failure of AF on new forms.

# 17. Main-paper recommendation

**Not main-paper-worthy as evidence of task-form generalization.** Correct the method/input description now: the implementation has an explicit task one-hot and privileged simulator velocity. Define full-task feasibility conceptually as terminal success of the complete downstream frozen-VLA behavior under the maintained grasp; label lift/transport/place/release as the original instantiation. Retain the supported root/friction claims and disclose the observed reliability–force tradeoff within that family.

Answers to the requested questions: Did we truly evaluate AF beyond lift/transport/place? **No.** Are the candidate forms absent from feasibility training? **Yes, as candidate/registered forms.** Did the frozen VLA already know them? **Not established; none passed the native-reset screen.** Did AF adapt force without retraining? **No new-form AF inference or branches were run.** Which forms transferred or failed? **Unknown; none reached the formal test.** Why? **Qualification and runtime/evaluator/handoff gates, including pregrasp failures and categorical unseen-task encoding.** Main-paper-worthy? **No for a transfer claim; useful as a scope/input audit.**

To answer the central scientific question, still required are source-backed supplied-grasp snapshots for genuinely different maintained-grasp tasks, task-specific terminal validation, Fixed-5 qualification from the common post-query handoff, a justified frozen unseen-task encoding, and then one matched AF/Fixed formal test on fresh roots. Do not lower this screen's threshold or rerun its negatives until positive. Such a supplied-grasp study must be prospectively versioned as a different qualification protocol and its distinction disclosed.
'''
(P/'FINAL_HELDOUT_TASK_FORM_GENERALIZATION_REPORT.md').write_text(report)
terminal={
 'CURRENT_EXPLICIT_TASK_CODE':'YES',
 'CURRENT_TASK_CONTEXT_ACTUALLY_IS':'4-way task one-hot + first-8 online Cartesian command summaries + duplicated decision state/masks; simulator velocity included',
 'EXISTING_ROOT_HELDOUT':'YES','EXISTING_FRICTION_HELDOUT':'YES',
 'EXISTING_TASK_HELDOUT':'YES (historical LOTO only; current frozen ensemble NO)','EXISTING_TASK_FORM_HELDOUT':'NO',
 'NUM_AVAILABLE_TASKS':51,'NUM_AVAILABLE_TASKS_SCOPE':'46 native config records + 5 other native tasks; 842 inventory entries across all tiers; not all executable by frozen VLA',
 'NUM_DISTINCT_TASK_FORMS':3,'NUM_DISTINCT_TASK_FORMS_SCOPE':'native intent A/C/D; only original A has admitted current AF evidence',
 'NUM_VLA_QUALIFIED_NEW_FORMS':0,'FINAL_HELDOUT_TASKS':[],'FINAL_HELDOUT_TASK_FORMS':[],
 'FORMS_ABSENT_FROM_FEASIBILITY_TRAINING':['FORM_C insertion/constrained (native definitions)','FORM_D extraction/pulling (native definitions)'],
 'NUM_FRESH_ROOTS':0,'NUM_CONTEXTS':0,'NUM_VALID_BRANCHES':0,
 'QUALIFICATION_SCREEN_VALID_EPISODES':len(allrows),'QUALIFICATION_SCREEN_ESTABLISHED_GRASPS':sum(q['established_grasps'] for q in qual),
 'PER_FORM_RESULTS':{'FORM_A_ARCHIVED_REFERENCE':ref,'NEW_FORMS':{'formal_contexts':0,'AF':'NOT_TESTED','qualification':qual}},
 'AF_FULL_SR':None,'FIXED3_FULL_SR':None,'FIXED4_FULL_SR':None,'FIXED5_FULL_SR':None,
 'AF_MEASURED_SQUEEZE':None,'FIXED5_MEASURED_SQUEEZE':None,'AF_GT_DECISION_FORCE_MAE':None,
 'GENERALIZED_REQUIREMENT_FORCE_SPEARMAN':None,
 **{k:0 for k in ['AF_FAILURES_GRASP_RETENTION','AF_FAILURES_TASK_CONTACT','AF_FAILURES_VLA','AF_FAILURES_TERMINAL_GEOMETRIC','AF_FAILURES_RELEASE','AF_FAILURES_OTHER']},
 'AF_FAILURE_COUNTS_INTERPRETATION':'zero executed AF branches; not zero observed failure rate',
 'HELDOUT_TASK_FORM_GENERALIZATION_CLAIM':'NOT_TESTED','GENERALIZATION_BEYOND_LIFT_TRANSPORT_PLACE':'NOT_TESTED',
 'MAIN_PAPER_WORTHY':'NO',
 'REMAINING_MUST_RUN_TASK_FORM_EXPERIMENTS':'prospectively versioned supplied-grasp/post-query Fixed-5 qualification, frozen task evaluator and unknown-code contract, then one matched fresh-root AF/Fixed trial if tasks qualify'}
write('TERMINAL_SUMMARY.json',terminal)
(P/'TERMINAL_SUMMARY.txt').write_text('\n'.join(k+' = '+('N/A (NOT_TESTED)' if v is None else json.dumps(v) if isinstance(v,(dict,list)) else str(v)) for k,v in terminal.items())+'\n')
print(json.dumps({'new_task_form_AF_status':'NOT_TESTED','qualified_tasks':0,'qualification_episodes':len(allrows),'manifest_sha256':sha(P/'FINAL_HELDOUT_TASK_FORM_SET.json')}))
