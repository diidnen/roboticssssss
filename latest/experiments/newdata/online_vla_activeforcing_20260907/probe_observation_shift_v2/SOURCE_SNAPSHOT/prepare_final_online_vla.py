"""Freeze qualified online runtime only after evidence and fresh-root audits.

No simulator is imported or launched. Qualified worker/arm/force behavior is
copied byte-for-byte. This program does not choose roots or infer gate approval
from incomplete development results.
"""
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timezone
from common import HERE, BASE, V5, TABERO, read, write, sha, checkpoint_inventory, INSTRUCTIONS
from run_final_online_vla import REQUIRED_GATES

RUNTIME_FILES = ('worker.py','runtime.py','arbitration.py','phase_free_feasibility.py','common.py')


def checked_sources(values):
    for path, expected in values.items():
        if sha(path) != expected:
            raise RuntimeError('Evidence/source changed before final freeze: '+path)


def prepare(qualification, admission_path, fresh_path, out):
    qualification, admission_path, fresh_path, out = map(Path,(qualification,admission_path,fresh_path,out))
    if out.exists():
        raise FileExistsError('Unique final runtime destination already exists')
    admission = read(admission_path); fresh = read(fresh_path)
    if Path(admission['qualification_directory']).resolve() != qualification.resolve():
        raise RuntimeError('Qualification identity mismatch')
    if any(admission.get(key) is not True for key in REQUIRED_GATES if key != 'FRESH_ROOT_NONEXPOSURE_VERIFIED'):
        raise RuntimeError('Required scientific gate is incomplete')
    if admission.get('EXISTING_FEASIBILITY_VLA_TRANSFER') not in ('PASS','PARTIAL'):
        raise RuntimeError('No admitted feasibility transfer')
    if admission.get('completed_primary_branches') not in (36,72):
        raise RuntimeError('Development transfer qualification incomplete')
    if not admission.get('evidence_sha256'):
        raise RuntimeError('Gate declarations lack evidence bindings')
    checked_sources(admission['evidence_sha256'])
    if fresh.get('FRESH_ROOT_NONEXPOSURE_VERIFIED') is not True or fresh.get('root_selection_used_outcomes') is not False:
        raise RuntimeError('Fresh-root exposure audit not passed')
    if fresh.get('physics_branches_started_before_freeze') != 0 or not fresh.get('evidence_sha256'):
        raise RuntimeError('Root plan is not prospectively evidence-bound')
    checked_sources(fresh['evidence_sha256'])
    roots = sorted(fresh['roots'])
    if len(set(roots)) != 4 or any(x in fresh['excluded_root_or_seed_ids'] for x in roots):
        raise RuntimeError('Expected four untouched roots outside exposure inventory')
    candidate = read(qualification/'CANDIDATE_RUNTIME_MANIFEST.json')
    checked_sources(candidate['source_hashes'])
    schema = read(HERE/'FEASIBILITY_FEATURE_SCHEMA_AUDIT.json')
    features = schema['FINAL_CANDIDATE_FEATURE_RECORDS']
    if sorted(r['FEATURE_INDEX'] for r in features) != list(range(64)):
        raise RuntimeError('Incomplete final per-dimension input audit')
    if any(r.get('ONLINE_VLA_OBSERVABLE') is not True or r.get('PRE_ACTION_CAUSAL') is not True or r.get('SCRIPT_ONLY') is not False for r in features):
        raise RuntimeError('Noncausal or script-only feature cannot enter final runtime')
    if candidate['feasibility_input_shape'] != [8,64] or candidate['phase_representation'] != 'NONE':
        raise RuntimeError('This freeze path requires the qualified phase-free candidate')
    for name in RUNTIME_FILES:
        if sha(qualification/'SOURCE_SNAPSHOT'/name) != admission['qualified_runtime_sha256'][name]:
            raise RuntimeError('Admission does not bind actual qualified runtime: '+name)
    observation_source = TABERO/'analysis/p6g1_primitive_ik_vla_grasp_realization.py'
    observation_sha = sha(observation_source)
    adapters = list((qualification/'branches').glob('*/VLA_OBSERVATION_ADAPTER.json'))
    if len(adapters)<36 or any(read(p)['source_sha256']!=observation_sha for p in adapters):
        raise RuntimeError('Historical observation adapter not identical across qualified branches')
    ready_path = HERE/'server_v2/SERVER_READY.json'; ready = read(ready_path)
    checked_sources(ready['source_hashes'])
    inventory = checkpoint_inventory()
    if inventory['sha256'] != ready['checkpoint_sha256'] or ready['checkpoint_loaded'] is not True:
        raise RuntimeError('Actual frozen VLA checkpoint not verified')
    belief_path = V5/'BELIEF_MANIFEST.json'; belief = read(belief_path)
    feas_path = BASE/'current_fulltask_feasibility_runtime_freeze_v2_20260906/FINAL_FEASIBILITY_RUNTIME_MANIFEST.json'
    feas = read(feas_path)
    derived_path = HERE/'phase_free_exact_checkpoints/PHASE_FREE_EXACT_MANIFEST.json'
    derived = read(derived_path)
    model_sources = {r['path']:r['sha256'] for r in belief['checkpoints']+feas['checkpoints']+derived['checkpoints']}
    checked_sources(model_sources)
    out.mkdir(parents=True,exist_ok=False); snapshot = out/'SOURCE_SNAPSHOT';snapshot.mkdir()
    for p in (qualification/'SOURCE_SNAPSHOT').glob('*.py'):
        shutil.copy2(p,snapshot/p.name)
    # Current independent auditors and scheduling code never alter the
    # qualified observation, force-selection or physical-execution modules.
    for name in ('run_final_online_vla.py','prepare_final_online_vla.py','build_final_online_results.py','audit_rollout.py',
                 'audit_geometric_label.py','drop_diagnostics.py','behavior_audit.py',
                 'qualification_metrics.py','review_qualification.py','matched_seeds.py'):
        shutil.copy2(HERE/name,snapshot/name)
    for name in RUNTIME_FILES:
        if sha(snapshot/name) != sha(qualification/'SOURCE_SNAPSHOT'/name):
            raise RuntimeError('Qualified runtime behavior changed while copying')
    evidence = out/'FROZEN_EVIDENCE';evidence.mkdir()
    for path in (admission_path,fresh_path,belief_path,feas_path,derived_path,
                 HERE/'FEASIBILITY_FEATURE_SCHEMA_AUDIT.json',HERE/'METRIC_CONTRACT_CANDIDATE.json',ready_path):
        shutil.copy2(path,evidence/path.name)
    write(out/'FINAL_FRESH_ROOT_PLAN.json',fresh)
    write(out/'DEV_PLAN.json',{'role':'FINAL_MAIN_TEST_WORKER_COMPATIBILITY_PLAN',
        'contexts':fresh['contexts'],'roots':roots,'methods':['FIXED_3','FIXED_4','FIXED_5','ACTIVEFORCING']})
    write(evidence/'FINAL_CHECKPOINT_INVENTORY.json',inventory)
    metrics = read(HERE/'METRIC_CONTRACT_CANDIDATE.json')
    metrics.update(version='FINAL_ONLINE_VLA_METRIC_CONTRACT_V1',status='FROZEN_BEFORE_FRESH_ROOT_PHYSICS')
    write(evidence/'FINAL_METRIC_CONTRACT.json',metrics)
    sources = {p:h for p,h in candidate['source_hashes'].items() if not Path(p).is_relative_to(qualification/'SOURCE_SNAPSHOT')}
    sources.update(ready['source_hashes']); sources.update(model_sources)
    sources.update({str(p):sha(p) for p in snapshot.glob('*.py')})
    sources.update({str(p):sha(p) for p in evidence.iterdir()})
    sources.update({str(out/name):sha(out/name) for name in ('DEV_PLAN.json','FINAL_FRESH_ROOT_PLAN.json')})
    sources.update({str(belief_path):sha(belief_path),str(feas_path):sha(feas_path)})
    sources[str(observation_source)] = observation_sha
    client_source = TABERO/'benchmarks/openpi/openpi-client/src/openpi_client'
    if not (client_source/'websocket_client_policy.py').exists():
        raise RuntimeError('Native websocket client source unavailable')
    sources.update({str(p):sha(p) for p in client_source.rglob('*.py')})
    m = dict(candidate)
    m.update({k:True for k in REQUIRED_GATES})
    m.update(version='FINAL_ONLINE_FROZEN_VLA_ACTIVEFORCING_V1',created_utc=datetime.now(timezone.utc).isoformat(),
        final_runtime_frozen=True,FINAL_RUNTIME_USES_ONLINE_VLA=True,DOWNSTREAM_ACTION_SOURCE='ONLINE_VLA',
        VLA_CHECKPOINT=inventory['checkpoint'],VLA_CHECKPOINT_SHA256=inventory['sha256'],
        VLA_CHECKPOINT_HASH_ALGORITHM=inventory['algorithm'],VLA_CHECKPOINT_LOADED=True,
        ONLINE_POLICY_INFERENCE_DURING_ROLLOUT=True,VLA_ACTION_PROVENANCE_VERIFIED=True,
        POLICY_SERVER_VERSION={'backend':ready['backend'],'config':ready['policy_config'],
            'source_hashes':ready['source_hashes'],'readiness_evidence':str(evidence/ready_path.name)},
        TASK_INSTRUCTIONS=INSTRUCTIONS,
        OBS_PREPROCESSING={'source':str(snapshot/'runtime.py'),
            'historical_definitions_source':str(observation_source),'historical_definitions_sha256':observation_sha,
            'semantics':'Historical Tabero image padding224, seven-dimensional pose/gripper state, native tactile marker history; first common RGB cache restored only at identical postprobe state; all later observations live.'},
        ACTION_POSTPROCESSING='Original OpenPI Unnormalize -> AbsoluteActions(first6) -> LiberoForceOutputs(13)',
        ACTION_CHUNKING={'predicted_steps':50,'executed_steps':10,'control_dt_seconds':.05,'requery_control_steps':10},
        VLA_ARM_ACTION_MAPPING='Decoded float32 action[:6] passed through exactly, every executed step',
        VLA_GRIPPER_MASKING_RULE='Raw aperture and six force dimensions retained in logs; AF owns grasp aperture servo and squeeze; raw aperture>=.039 triggers canonical .04 release and zero squeeze, same rule for fixed baselines.',
        PROBE_VERSION={'core':str(BASE/'current_runtime_core_snapshot_v2_20260905/current_runtime_core.py'),
            'same_probe_all_main_methods':True,'established_grasp_only':True},
        PHYSICAL_BELIEF_58D_CHECKPOINTS=belief['checkpoints'],POSTERIOR_INTERFACE=belief['interface'],
        FEASIBILITY_CHECKPOINT={'actually_loaded_source_checkpoints':feas['checkpoints'],
            'frozen_derivation':str(snapshot/'phase_free_feasibility.py'),
            'equivalent_exported_phasefree_checkpoints':derived['checkpoints'],
            'storage_semantics':'Runtime loads original checkpoint and exactly deletes seven normalized-constant GRU input columns; exported64D checkpoints are equivalent state-dict artifacts, not falsely described as files opened by worker.',
            'new_training_steps':0,'input_shape':[8,64]},
        UTILITY=feas['utility'],FORCE_SUPPORT=feas['force_support'],FORCE_GRID_STEP_N=feas['planner_grid_step'],
        CONTROLLER={'source':next(p for p in sources if p.endswith('/force_position_action.py')),
            'servo':'unchanged physical probe P4 force servo','measured_squeeze':'2*min(actual_left_normal,actual_right_normal)'},
        HANDOFF='Identical established-grasp -> same corrected probe -> identical decision snapshot, posterior, common initial observation and newly inferred first chunk; carried last probe command initializes unchanged force servo.',
        PLACEMENT='Original authored whole-mesh basket geometry; VLA arm motion and native open intent; no scripted placement waypoint',
        LABEL_CONTRACT='ONLINE_VLA_350STEP_WHOLE_MESH_RELEASE_SUPPORT_V1',
        METRIC_CONTRACT=str(evidence/'FINAL_METRIC_CONTRACT.json'),
        EXISTING_FEASIBILITY_VLA_TRANSFER=admission['EXISTING_FEASIBILITY_VLA_TRANSFER'],
        feasibility_transfer=admission['EXISTING_FEASIBILITY_VLA_TRANSFER'],
        FEASIBILITY_SEQUENCE_SOURCE='ONLINE_VLA_ACTION_CHUNK',FINAL_VLA_RUNTIME_USES_SCRIPTED_PREFIX=False,
        MAIN_FIXED_BASELINES_USE_SAME_PROBE=True,NOPROBE_IS_SEPARATE_ABLATION=True,
        VLA_POLICY_STOCHASTIC=True,COMMON_RANDOM_NUMBERS_USED=True,
        qualification_source=str(qualification),qualified_runtime_sha256=admission['qualified_runtime_sha256'],
        fresh_root_plan_sha256=sha(out/'FINAL_FRESH_ROOT_PLAN.json'),source_hashes=sources,
        script_only_features_present=False,phase_representation='NONE',
        script_results_role='ENGINEERING_CONTROLLED_MOTION_VALIDATION',
        final_test_contexts=48,final_main_branches=192,parameters_locked=True)
    m['58D_BELIEF_CHECKPOINT'] = belief['checkpoints']
    m['FINAL_PHASE_REPRESENTATION'] = 'NONE'
    write(out/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json',m)
    write(out/'CANDIDATE_RUNTIME_MANIFEST.json',m)
    # validate() normally requires its own frozen script path. Run it through
    # the snapshot module in a fresh interpreter when launching the final queue.
    write(out/'FREEZE_PREPARATION_COMPLETE.json',{'runtime_sha256':sha(out/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json'),
        'worker_runtime_byte_parity':True,'physics_started':False,
        'next_required':'Run frozen snapshot admission validation before final coordinator starts'})
    return m


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--qualification',required=True);p.add_argument('--admission',required=True)
    p.add_argument('--fresh-plan',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    prepare(a.qualification,a.admission,a.fresh_plan,a.out)
    print('FINAL_RUNTIME_FROZEN_NO_PHYSICS_STARTED')
