import json, shutil, hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
summary=load(H/'offline_feature_audit_v3/PHASE_SENSITIVITY_SUMMARY.json')
parity=load(H/'PHASE_FREE_RUNTIME_PARITY.json')
assert parity['passed'] and summary['results']['PHASE_REMOVED_EXACT']['vs_original']['max_probability_change']==0
p=H/'FEASIBILITY_FEATURE_SCHEMA_AUDIT.json'
shutil.copy2(p,H/'FEASIBILITY_FEATURE_SCHEMA_AUDIT_INITIAL.json')
d=load(p);final=[]
for r in d['FEATURE_RECORDS']:
    if 6<=r['FEATURE_INDEX']<13:continue
    r=dict(r);old=r['FEATURE_INDEX'];r['ORIGINAL_FEATURE_INDEX']=old;r['FEATURE_INDEX']=len(final)
    r.update(ONLINE_VLA_OBSERVABLE=True,PRE_ACTION_CAUSAL=True,SCRIPT_ONLY=False,AVAILABLE_DURING_ONLINE_VLA=True)
    if old<6:
        r['SOURCE_FUNCTION']=str(H/'worker.py')+'::make_sequence'
        r['SOURCE_DATA']='Current post-probe online VLA inference receipt and decoded first8 XYZ action targets; no scripted prefix.'
        r['RECOMMENDED_FINAL_STATUS']='KEEP_ONLINE_CHUNK; VALIDATE_TRANSFER_COVERAGE'
    else:r['RECOMMENDED_FINAL_STATUS']='KEEP_WITH_DOCUMENTED_MEASUREMENT_SCOPE'
    if r.get('DUPLICATES_FEATURE_INDEX') is not None:r['DUPLICATES_FEATURE_INDEX']-=7
    final.append(r)
assert len(final)==64
d.update(FINAL_CANDIDATE_FEATURE_RECORDS=final,FEAS_FEATURE_TOTAL_DIM_ORIGINAL=71,FEAS_FEATURE_TOTAL_DIM=64,
    SEQUENCE_DIM=10,CONDITION_DIM=54,SCRIPT_ONLY_FEATURES_REMOVED=list(range(6,13)),
    SCRIPT_SOURCED_FEATURES_REPLACED_WITH_ONLINE_CHUNK=list(range(6)),
    FINAL_PHASE_REPRESENTATION='NONE',PHASE_PROXY_USES_VLA_CHUNK=False,TRAIN_RUNTIME_PHASE_PARITY=True,
    PHASE_DEPENDENCE_STRONG=False,PHASE_REQUIRED_FOR_CURRENT_FEASIBILITY=False,
    FINAL_INPUT_SCHEMA_SOFTWARE_VALID=True,FINAL_QUALIFIED=False,
    INPUT_REMOVAL_RECEIPT={'offline_summary':str(H/'offline_feature_audit_v3/PHASE_SENSITIVITY_SUMMARY.json'),
        'runtime_parity':str(H/'PHASE_FREE_RUNTIME_PARITY.json'),'original_audit_sha256':sha(H/'FEASIBILITY_FEATURE_SCHEMA_AUDIT_INITIAL.json')},
    CURRENT_SOURCE_HASHES={str(p):sha(p) for p in (H/'worker.py',H/'phase_free_feasibility.py')})
p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
write(H/'GOAL_EXTENSION.json',dict(GOAL_EXTENSION='RESTORE_ONLINE_VLA_AND_REMOVE_SCRIPT_ONLY_FEASIBILITY_DEPENDENCIES',
    master_goal_preserved=True,scripted_pipeline_restarted=False,previous_online_evidence_preserved=True,
    completed=['Full 71D original feature audit','64D phase-free candidate schema','647-row fixed-checkpoint phase sensitivity',
        '3seed x80epoch phase-removed offline ablation','Exact frozen-model phase removal; selected for transfer qualification'],
    remaining=['Complete online dev qualification within72 total branches','Validate all-task arbitration, continuation and postprobe behavior',
        'Reconfirm force boundaries and feasibility probability quality','Only then decide limited VLA-matched data need','Final runtime freeze and4 fresh roots']))
write(H/'OFFLINE_AUDIT_VALIDATION.json',dict(status='SHARE_WITH_CAVEATS',new_physics_runs_for_audit=0,
    original_rows=647,original_split={'TRAIN':431,'VAL':108,'TEST':108},
    verified=['Every consumed feature enumerated','Constant phase checked across all647 rows',
        'Exact phase deletion preserves all647 probabilities and72 force decisions',
        'Actual restored VLA chunk runtime curve parity','Original checkpoints not overwritten'],
    caveats=['PHASE_MASKED uses external missing metadata and neutral mean imputation because original network has no phase-validity input.',
        'Empirical phase permutation is a no-op; channel shuffling reported separately.',
        'Phase-removed retrain retains64 hidden units,54D condition,80epochs,3seeds and same split; parameter count decreases1344 because removed columns are absent; CPU vs original CUDA.',
        'Old scripted TEST is already exposed and cannot establish fresh online VLA performance.',
        'Object velocity is a current simulator-state measurement, not validated real-world perception.',
        'Online motion feature coverage and downstream outcome transfer remain unqualified.'],
    superseded_incomplete_runs=['offline_feature_audit_v1','offline_feature_audit_v2'],
    superseded_reason='Stopped solely to factor identical sequence computations and avoid materializing redundant tensors; no model or quadrature changes. Use completed v3 only.'))
print('Offline audit finalized; final fresh roots remain forbidden pending online qualification.')
