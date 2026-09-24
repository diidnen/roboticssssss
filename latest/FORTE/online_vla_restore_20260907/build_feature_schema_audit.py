"""Enumerate every consumed input, including masks and external marginalization."""
import json, hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
R=Path('/home/exouser/FORTE')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=R/'analysis/results/current_runtime_branch_execution_v6_20260905/branch_execution.py'
builder=R/'activeforcing_feasibility_features.py'
runtime=R/'current_fulltask_feasibility_runtime.py'
worker=H/'worker.py'
records=[]
def add(i,name,source,data,script=False,status='KEEP',**extra):
    records.append(dict(FEATURE_NAME=name,FEATURE_INDEX=i,INDEX_BASE=0,
        SOURCE_FUNCTION=source,SOURCE_DATA=data,AVAILABLE_PRE_ACTION=True,
        AVAILABLE_DURING_ONLINE_VLA=not script,SCRIPT_ONLY=script,CAUSAL=True,
        DEPENDS_ON_FUTURE_ACTION=False,RECOMMENDED_FINAL_STATUS=status,
        **extra))
for i,name in enumerate(('chunk_relative_x','chunk_relative_y','chunk_relative_z','chunk_step_dx','chunk_step_dy','chunk_step_dz')):
    add(i,name,[str(old)+'::preaction_features',str(builder)+'::nominal_input',str(worker)+'::make_sequence'],
        'Historical: first eight p4._interp(start,start,i,20) commands. Restored VLA candidate: first eight decoded XYZ targets from current online inference.',
        script=True,status='REPLACE_SOURCE_WITH_CURRENT_ONLINE_VLA_CHUNK; ALREADY IMPLEMENTED IN DEV',
        AVAILABLE_DURING_ONLINE_VLA_AFTER_REPLACEMENT=True,
        DEPENDS_ON_CURRENT_PREDICTED_CHUNK=True,
        FUTURE_ACTION_NOTE='Predicted chunk is known before force selection; no executed future trajectory or outcome is read.',
        FINAL_ONLINE_VLA_OBSERVABLE=True,FINAL_PRE_ACTION_CAUSAL=True,FINAL_SCRIPT_ONLY=False,
        COVERAGE_RISK='Old 647 motion magnitude <=1.1920929e-7. Online motion is outside old support; qualification required.')
for i,p in enumerate(('branch_hold','lift','transit','over_basket','place','release','settle'),6):
    add(i,'script_phase_'+p,[str(old)+'::preaction_features',str(builder)+'::nominal_input'],
        "Hard-coded ['branch_hold']*8. Not measured task progress.",True,'REMOVE_NORMALIZED_GRU_INPUT_COLUMN',
        FINAL_INPUT_PRESENT=False,FINAL_SCRIPT_ONLY=False,
        NOTE='All 647 phase vectors identical; normalized contribution exactly zero. No phase proxy necessary to preserve existing function.')
for i,task in enumerate((0,1,5,6),13):
    add(i,'task_onehot_'+str(task),[str(builder)+'::nominal_input',str(worker)+'::make_sequence'],
        'Known requested task ID mapped from frozen task instruction; not simulator task-stage flag.')
add(17,'candidate_force_div8',[str(runtime)+'::CurrentFeasibility',str(builder)+'::nominal_input'],
    'Enumerated candidate force F in [3,5] N divided by8. Available before execution; not measured realized force.')
add(18,'posterior_quadrature_mu',[str(runtime)+'::CurrentFeasibility'],
    'Runtime: positive continuous posterior integration nodes from current physical probe. Training: simulator friction conditional supervision.',
    NOTE='Training ground truth mu is not a deployment input. Posterior weights integrate model predictions once outside the 71D tensor.',
    RUNTIME_HIDDEN_FRICTION_USED=False)
names=('relative_reference_x','relative_reference_y','relative_reference_z','object_velocity_x','object_velocity_y','object_velocity_z',
    'left_abs_local_normal','right_abs_local_normal','left_local_tangent_norm','right_local_tangent_norm',
    'object_horizontal_speed','first_finger_joint','negative_second_finger_joint')
for base,kind in ((19,'current_state'),(32,'current_validity'),(45,'initial_state'),(58,'initial_validity')):
    for j,n in enumerate(names):
        mask='validity' in kind
        if j<3: data='Zero displacement relative to current pre-action reference; valid constant, not desired waypoint.'
        elif j in (3,4,5,10):data='Current post-probe rigid-object root_velocity; speed is norm(vx,vy). Live Isaac observation captured before action.'
        elif j in (6,7,8,9):data='Last actual probe per-finger gripper-local force vector; abs(Fz) or norm(Fxy).'
        else:data='Current robot snapshot finger joint positions; first joint checked against last probe reading, second sign follows native policy.'
        if mask:data='Validity of '+data+' One only when the actual decision-time source exists; do not fabricate missing readings.'
        add(base+j,kind+'_'+n,[str(old)+'::preaction_features',str(worker)+'::make_sequence'],data,
            DUPLICATES_FEATURE_INDEX=(base+j-26 if base>=45 else None),
            SIMULATOR_STATE_MEASUREMENT_ASSUMPTION=bool(j in (3,4,5,10)),
            REAL_ROBOT_OBSERVABILITY_VERIFIED=False if j in (3,4,5,10) else None,
            NOTE='Object velocity is online available in this simulator, but privileged relative to vision-only deployment; no physical-world velocity estimator has been validated.' if j in (3,4,5,10) else 'No future state or downstream outcome read.')
assert [r['FEATURE_INDEX'] for r in records]==list(range(71))
out=dict(SCHEMA='CURRENT_FULLTASK_8x71_ZERO_BASED',FEAS_FEATURE_TOTAL_DIM=71,SEQUENCE_DIM=17,CONDITION_DIM=54,
    SCRIPT_ONLY_FEATURES_FOUND={'hard_script_phase_indices':list(range(6,13)),
        'historical_script_sourced_motion_indices':list(range(6)),
        'additional_hidden_script_stage_or_waypoint_fields':[],
        'distinction':'Six motion definitions are reusable with causal online chunk; seven scripted phase labels are removed entirely.'},
    PROPOSED_FINAL_TOTAL_DIM=64,PROPOSED_FINAL_SEQUENCE_DIM=10,CONDITION_DIM_UNCHANGED=54,
    SCRIPT_ONLY_FEATURES_REMOVED='7 phase columns pending exact-removal verification and runtime integration; 6 motion sources already replaced in dev runtime.',
    FEATURE_RECORDS=records,
    CONSUMPTION='GRU consumes all 8 timesteps of columns0:17; MLP consumes only timestep0 of columns17:71. Remaining condition repetitions ignored.',
    NO_OTHER_DIRECT_INPUTS=['No downstream outcome','No executed future pose','No desired object pose','No hidden simulator task-stage','No placement waypoint ID','No progress scalar'],
    EXTERNAL_INPUTS=[dict(name='posterior integration nodes and weights',source='58D corrected probe belief -> frozen positive continuous posterior',
        available_pre_action=True,script_only=False,causal=True,note='Variance represented by full node/weight distribution, not a separate tensor sigma dimension.'),
        dict(name='TRAIN-only normalization constants',source='frozen feasibility checkpoints',available_pre_action=True,script_only=False,causal=True),
        dict(name='force support and utility',source='runtime manifest; F=3:0.05:5, p*(5-F)/5-(1-p)',available_pre_action=True,script_only=False,causal=True)],
    OBSERVABILITY_SCOPE='Current simulation runtime. Direct object velocity must be disclosed; real-robot sensor availability is not established.',
    PRE_ACTION_DEFINITION='After completed physical probe and current online VLA chunk inference, before candidate force is executed. Current predicted actions are allowed; future executed state/outcome is forbidden.',
    FINAL_QUALIFIED=False,
    SOURCE_HASHES={str(p):sha(p) for p in (old,builder,runtime,worker,Path(__file__))})
p=H/'FEASIBILITY_FEATURE_SCHEMA_AUDIT.json'
with p.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(p)
