"""Read-only audit of the exact frozen feasibility inputs and split lineage."""
from pathlib import Path
import ast, csv, hashlib, json
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
BASE=Path('/home/exouser/FORTE/analysis/results')
RUNS=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907')
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x): (OUT/name).write_text(json.dumps(x,indent=2)+'\n')

manifest=read(BASE/'current_fulltask_feasibility_runtime_freeze_v2_20260906/FINAL_FEASIBILITY_RUNTIME_MANIFEST.json')
checkpoints=[]
for item in manifest['checkpoints']:
    assert sha(item['path'])==item['sha256']
    ck=torch.load(item['path'],weights_only=False,map_location='cpu')
    checkpoints.append(ck)
assert all(np.array_equal(checkpoints[0]['normalization_mean'],c['normalization_mean']) for c in checkpoints)
keep=list(range(6))+list(range(13,71))
mean=np.asarray(checkpoints[0]['normalization_mean'])[keep]
std=np.asarray(checkpoints[0]['normalization_std'])[keep]
rows=[]
def feature(i,name,source,meaning,deploy,candidate='YES',dependent='YES',identity='NO',hidden='NO'):
    rows.append(dict(index=i,feature_name=name,dimension=1,source=source,physical_meaning=meaning,
        available_at_deployment=deploy,candidate_independent=candidate,task_dependent=dependent,
        explicit_task_identity=identity,hidden_simulator_information=hidden,
        normalization_source='pooled 431 TRAIN rows; checkpoint mean/std; original channel '+str(keep[i]),
        normalization_mean=float(mean[i]),normalization_std=float(std[i])))
for i,a in enumerate('xyz'):
    feature(i,'chunk_relative_'+a,'first online VLA chunk[:8,:3] minus first command','relative planned Cartesian position (m)','YES')
    feature(3+i,'chunk_increment_'+a,'zero first row then np.diff(chunk[:8,:3])','Cartesian command increment (m/step), not velocity or acceleration','YES')
for i,t in enumerate([0,1,5,6]):feature(6+i,'task_onehot_'+str(t),'(0,1,5,6).index(plan[task])','categorical task identity; repeated for eight timesteps','YES for known task; undefined for unseen ID',identity='YES')
feature(10,'candidate_force_div8','force grid 3:0.05:5, overwritten by curve()','candidate squeeze setpoint N / 8','YES','NO','NO')
feature(11,'friction_node','inferred positive mixture integration node; true mu only in training / GT diagnostic','object-side friction conditioning variable','INFERRED','YES','INDIRECT')
state_names=['position_zero_x','position_zero_y','position_zero_z','object_velocity_x','object_velocity_y','object_velocity_z','left_abs_force_z','right_abs_force_z','left_force_xy_norm','right_force_xy_norm','object_lateral_speed','left_finger_joint','negated_right_finger_joint']
for offset,block in [(12,'state'),(38,'duplicate_state')]:
    for j,name in enumerate(state_names):
        if j<3: src='np.zeros(13); never assigned'; meaning='constant zero placeholder, not observed object position'; dep='YES (constant)'; hidden='NO'
        elif j in [3,4,5,10]: src='saved[state][rigid_object][object_id][root_velocity][:3]';meaning='simulator object linear velocity; lateral norm for channel 10';dep='SIMULATOR ONLY in current implementation';hidden='YES'
        elif j<10: src='last RAW_PROBE.csv left_fxyz/right_fxyz';meaning='normal-contact-vector projection proxy; not resolved friction force';dep='sensor analogue; PhysX in this evaluation';hidden='SIMULATED SENSOR'
        else:src='saved robot joint_position[-2:]';meaning='finger joint position m (right sign negated)';dep='YES via encoders';hidden='NO'
        feature(offset+j,block+'_'+name,src,meaning,dep,hidden=hidden,dependent='NO' if j<3 else 'YES')
    for j,name in enumerate(state_names):feature(offset+13+j,block+'_mask_'+name,'builder assignment: all 13 mask channels = 1','availability mask; marks zero position placeholders as present','YES (constant)',dependent='NO')
rows.sort(key=lambda x:x['index']);assert [r['index'] for r in rows]==list(range(64))
with (OUT/'CURRENT_FEASIBILITY_FEATURES.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

# Execute only the pure feature builders against an existing development snapshot.
reference=RUNS/'dev_v4_phasefree/references/t0_r5100_low'
if not reference.exists():reference=RUNS/'final_vla_v1/references/t0_r170040_low'
raw=list(csv.DictReader((reference/'RAW_PROBE.csv').open()))
saved=torch.load(reference/'DECISION_STATE.pt',map_location='cpu',weights_only=False)
def clean(x):
    if hasattr(x,'detach'):return x.detach().cpu().numpy().tolist()
    return x
# Source imports common.clean locally; supply the audited common module.
import sys
sys.path.insert(0,str(RUNS/'final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT'))
builders={}
for name in ['final_vla_v1','confirmatory_v1','final_continuous_friction_generalization_v1']:
    path=RUNS/name/'SOURCE_SNAPSHOT/worker.py';tree=ast.parse(path.read_text())
    fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_sequence')
    ns={'np':np,'clean':clean};exec(compile(ast.Module(body=[fun],type_ignores=[]),str(path),'exec'),ns)
    builders[name]=ns['make_sequence']
chunk=np.arange(50*13,dtype=np.float32).reshape(50,13)*.001
values={k:(f(raw,saved,None,None,0,chunk) if k!='final_continuous_friction_generalization_v1' else f(raw,saved,0,chunk)) for k,f in builders.items()}
assert all(np.array_equal(next(iter(values.values())),v) for v in values.values())
unknown={}
for task in [2,105]:
    try:builders['final_continuous_friction_generalization_v1'](raw,saved,task,chunk)
    except Exception as e:unknown[str(task)]=repr(e)
stage=read(BASE/'current_matched_stage1_648_v1_20260906/STAGE_MANIFEST.json')
contexts={c['id']:c for c in stage['contexts']}
splits={}
for split in ['TRAIN','VAL','TEST']:
    cs=[c for c in contexts.values() if stage['split_by_context'][c['id']]==split]
    splits[split]={'tasks':sorted(set(c['task'] for c in cs)),'roots':sorted(set(c['root'] for c in cs)),'contexts':len(cs)}
write('evidence/MODEL_INPUT_VERIFICATION.json',dict(checkpoint_hashes_verified=True,
    checkpoint_paths=manifest['checkpoints'],normalizers_equal=True,input_shape=[8,64],
    model_shape='GRU(10,64)+MLP(54,64)+head(128,64,1)',
    all_three_builders_exact_equal_on_same_snapshot=True,unknown_task_exceptions=unknown,splits=splits,
    test_fixture=str(reference),synthetic_chunk_used_only_for_pure_builder_equivalence=True,
    physics_started=False,model_inference_on_new_tasks=False))

table='| Columns | Features | Dimension | Source and meaning | Deployment / privileged status |\n|---|---|---:|---|---|\n'
table+='| 0–2 | Relative Cartesian commands | 3 × 8 | First 8 online commands minus command 0 | Available |\n| 3–5 | Cartesian command increments | 3 × 8 | Zero then adjacent differences; no division by dt | Available |\n| 6–9 | **Explicit task one-hot** | 4 × 8 | IDs 0,1,5,6; repeated at every timestep | Known task metadata; unseen ID raises ValueError |\n| 10 | Candidate force / 8 | 1 | Overwritten for each grid candidate | Available; candidate-dependent |\n| 11 | Friction quadrature node | 1 | Posterior integration; true friction in training only | Inferred at AF deployment |\n| 12–24, 38–50 | Two identical state copies | 13 + 13 | Zero positions (3), object velocity (3), contact force proxies (4), lateral speed (1), finger joints (2) | Velocity and speed read privileged simulator state |\n| 25–37, 51–63 | Two all-one masks | 13 + 13 | Constant availability flags, including zero positions | Constant |\n'
text='''# Current feasibility input audit

**Explicit task identity exists: YES.** This conclusion comes from executed source inspection, not paper notation. The exact builder writes `x[:, 6 + (0, 1, 5, 6).index(task)] = 1`. See `CURRENT_FEASIBILITY_FEATURES.csv` for every scalar channel, source, dimension, deployment availability, candidate independence, task dependence, identity status, privileged status, and actual normalization mean/std.

Authoritative lineage: `final_vla_v1` (48 contexts) + `confirmatory_v1` (48) = the completed 96-context main evaluation. `final_continuous_friction_generalization_v1` uses the identical phase-free wrapper SHA256 `456e19e00cc966e80b0dcbfe4af1bd3b3f653221212d43be81df1e8beb1d18cc` and the same three September 6 checkpoints. The pure builders produce bitwise identical tensors on the same saved snapshot and supplied action chunk. Formatting/temporary-variable changes explain different builder AST hashes. No architecture has been modified by this audit.

The effective ensemble contains three `GRU(10,64) + MLP(54,64) + head(128,64,1)` models. The original checkpoints have GRU input width 17; seven constant normalized phase channels are deleted exactly at load. This is a frozen weight migration, not retraining. The 54 conditions are read only at timestep zero; their repetition across eight rows is not extra temporal information.

'''+table+'''
## Answers to the six audit questions

1. **Yes: four-way task one-hot.** No learned standalone task embedding was found, although the GRU learns weights on the categorical channels.
2. There is no `c_task` variable in the audited builder. If the paper uses that symbol for this implementation, it must explicitly include the one-hot and the first-eight-command Cartesian summaries, alongside the duplicated decision state. It cannot be described as a generic task-semantic embedding.
3. Task identity is **not only implicit** in actions/state/language. It is explicitly supplied by `plan['task']`. Language and RGB are supplied to the VLA, not directly to feasibility. Only the first three action coordinates reach feasibility; wrist rotation, gripper intent, and future force channels do not. Arm joint configuration, object orientation/geometry, target location, task contact geometry, and later VLA replans are not direct feasibility features.
4. The feature arithmetic is shared, but ID lookup is restricted to `(0,1,5,6)`. Additional task-specific routes include instruction lookup, object/target selection, asset geometry and grasp initialization, sensor bindings, `current_runtime_core.validate_call`, the belief loader's qualified-task guard, and the basket terminal evaluator. The canonical probe motion/force constants are shared. Task-dependent scene configuration is separate from learned input construction.
5. Feasibility normalization is **one pooled TRAIN-only vector**, shared by all seeds/tasks, fitted on 431 rows over samples and timesteps. Original small-variance channels use std=1. Seven deleted phase entries are removed from both mean and std. The belief loader also uses pooled TRAIN normalization. VLA normalization is explicitly loaded from checkpoint `assets/NathanWu7/tabero`; its directory name differs from the current config's dataset name `NathanWu7/tabero_object_25`. Neither directory name proves the exact training task manifest. No task-indexed normalizer is loaded by the authoritative server.
6. No explicit task-conditioned friction prior is fed to primary AF: it uses the pooled probe ensemble posterior. Learned task-dependent feasibility priors are possible through the one-hot. Task-specific object geometry, scene physics/default masses, target regions and grasp initialization are real preprocessing priors. The September 2 historical protocol lists task-specific Fmax values, but **the September 6 model runtime used here has universal Fmax=5 and force support [3,5]**. The no-query-prior baseline is not the primary AF posterior.

## Deployment caveats and compatibility

Object velocity and lateral speed come directly from the captured simulator `root_velocity`; no deployed visual estimator is present in this path. Contact-vector proxies are simulation sensor readbacks. Neither fact should be hidden by calling every channel deployment-observable. The zero position entries have all-one masks despite containing no measured position. Both state blocks are identical, not pre/post state differences.

Unseen task IDs raise `ValueError: tuple.index(x): x not in tuple` in the unmodified builder. The input audit does not remove, zero, average, or relabel the task code. An all-zero unknown code is dimensionally possible but is an untrained encoding outside the four one-hot vertices; aliasing an unseen form to a seen ID would inject an arbitrary task prior. Neither is an established compatibility contract. This limitation does **not** block Fixed-5 VLA-only qualification, which does not need feasibility. Any eventual AF compatibility mapping must be declared and frozen before AF outcomes, with its extrapolative status reported.

## Primary sources

- `/media/volume/newdata/exouser/online_vla_activeforcing_20260907/{final_vla_v1,confirmatory_v1,final_continuous_friction_generalization_v1}/SOURCE_SNAPSHOT/{worker.py,phase_free_feasibility.py,common.py,runtime.py}`.
- `/home/exouser/FORTE/current_fulltask_feasibility_runtime.py` and `train_current_fulltask_feasibility_baseline_20260906.py`.
- `/home/exouser/FORTE/analysis/results/current_fulltask_feasibility_baseline_v1_20260906/TRAINING_PROTOCOL.json`.
- `/home/exouser/FORTE/analysis/results/current_runtime_recovery_v2_20260905/{branch_execution.py,geometry_grasp_initializer.py}`.
- `/home/exouser/FORTE/analysis/results/current_multitask58_loader_candidate_20260905/continuous_belief.py` and `/home/exouser/FORTE/current_contract_belief_features.py`.
- `/home/exouser/FORTE/online_vla_restore_20260907/server.py`.

Reproduce: run `audit_model.py` with `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python` (NumPy and PyTorch required; no simulator launch). `evidence/MODEL_INPUT_VERIFICATION.json` records checkpoint hashes, builder parity and unsupported-ID exceptions.
'''
(OUT/'CURRENT_FEASIBILITY_INPUT_AUDIT.md').write_text(text)
print(json.dumps({'input_shape':[8,64],'explicit_task_code':True,'parity':True,'splits':splits}))
