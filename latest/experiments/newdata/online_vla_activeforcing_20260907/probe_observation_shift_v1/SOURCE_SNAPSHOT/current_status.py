"""Read-only evidence aggregation into a clearly provisional stdout status."""
import json
from pathlib import Path
from common import read,CHECKPOINT

H=Path(__file__).resolve().parent
D=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907')
rows=[]
for version in ('dev_v2','dev_v3','dev_v4_phasefree'):
    for p in sorted((D/version/'branches').glob('*/BRANCH_RESULT.json')):
        d=read(p);rows.append(dict(version=version,context=d['plan']['id'],force=d['force'],
            success=d['outcome'].get('full_task_success_y'),online=d['online_vla_verified'],source=str(p)))
status=dict(FINAL_STATUS='IN_PROGRESS; NOT_FINAL_PAPER_EVIDENCE',
    DOWNSTREAM_ACTION_SOURCE='ONLINE_VLA',VLA_CHECKPOINT=str(CHECKPOINT),
    VLA_CHECKPOINT_SHA256='0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17',
    VLA_CHECKPOINT_LOADED='YES',ONLINE_VLA_VERIFIED='YES_DEV_RUNTIME',
    FEAS_FEATURE_TOTAL_DIM=64,SCRIPT_ONLY_FEATURES_FOUND='7 phase;6 historical script-sourced motion channels',
    SCRIPT_ONLY_FEATURES_REMOVED='7 phase;6 motion sources replaced by online VLA chunk',
    PHASE_DEPENDENCE_STRONG='NO',PHASE_REQUIRED_FOR_CURRENT_FEASIBILITY='NO',
    FINAL_PHASE_REPRESENTATION='NONE; exact input-column deletion',PHASE_PROXY_USES_VLA_CHUNK='NO_PROXY_NEEDED',
    TRAIN_RUNTIME_PHASE_PARITY='YES',VLA_ACTION_CHUNK_SIZE=50,VLA_EXECUTION_HORIZON=10,VLA_REQUERY_RATE='2Hz simulated; every10 controlsteps at20Hz',
    VLA_GRIPPER_ARBITRATION_VALID='YES_SOFTWARE_AND_TASK0_DEV; OTHER_TASKS_PENDING',
    ESTABLISHED_GRASP_VLA_CONTINUATION_VALID='TASK0_DEV_SUPPORTED; OTHER_TASKS_PENDING',
    PROBE_INDUCED_VLA_SHIFT_ACCEPTABLE='PENDING; pre/post pose/contact available, pre-probe RGB not yet archived',
    EXISTING_FEASIBILITY_VLA_TRANSFER='PENDING',VLA_MATCHED_RETRAIN_REQUIRED='NOT_YET_DETERMINED',NEW_VLA_MATCHED_ROWS=0,
    FEASIBILITY_RETRAIN_FOR_PHASE_REMOVAL_REQUIRED='NO',
    OFFLINE_RETRAIN_ABLATION='Completed3seeds x80epochs on647scripted rows; not selected as deployment model',
    VLA_REAL_FORCE_BOUNDARY_EXISTS='YES_TASK0_LOW_DEV; OTHER_TASKS_PENDING',
    READY_FOR_FINAL_FRESH_ROOT_VLA_TEST='NO',NUM_FINAL_FRESH_ROOTS_PLANNED=4,NUM_FINAL_FRESH_ROOTS_EXECUTED=0,
    NUM_FINAL_VLA_CONTEXTS=0,NUM_FINAL_VLA_BRANCHES=0,
    AF_VLA_FULL_SR='NOT_RUN',FIXED3_VLA_FULL_SR='NOT_RUN',FIXED4_VLA_FULL_SR='NOT_RUN',FIXED5_VLA_FULL_SR='NOT_RUN',
    AF_VLA_MEASURED_FORCE='NOT_FINAL',FORCE_SAVING_VS_FIXED5='NOT_FINAL',
    TABERO_NEUTRAL_RESULT='NOT_RUN',FORTE_STYLE_RESULT='NOT_RUN',VLA_ABLATION_RESULTS='NOT_RUN',
    SCRIPTED_RESULTS_RETAINED_AS='ENGINEERING_CONTROLLED_MOTION_VALIDATION',PAPER_VLA_CLAIM_SUPPORTED='NO_FINAL_EVALUATION_YET',
    completed_development_branches=rows)
print(json.dumps(status,indent=2))
