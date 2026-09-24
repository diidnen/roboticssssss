"""Prospective native dataset locks and non-mutating receipt verification."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
REPO=BASE/'RoboTwin'
FORTE=Path('/home/exouser/FORTE')
SNAPSHOT=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT')
FORCES=[.5,1.,2.,3.,4.,5.,6.,8.]
FRICTIONS={'TRAIN':[.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85],
           'VAL':[.325,.475,.625,.775],'TEST':[.375,.525,.675,.825]}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def now():return datetime.now(timezone.utc).isoformat()


def source_files():
    # Freeze actual executed adapter dependencies and originals. Trainer and
    # queue mechanics may subsequently be added; they get their own locks.
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.so'))
    files += [FORTE/n for n in ['activeforcing_current_probe.py','activeforcing_probe_friction_contract.py',
        'current_contract_belief_features.py','current_contract_physical_belief.py','current_fulltask_feasibility_runtime.py']]
    files += [SNAPSHOT/n for n in ['worker.py','arbitration.py','phase_free_feasibility.py']]
    files += [Path('/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py'),
              Path('/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_tactile_libero_env_cfg.py')]
    for folder in ['envs','scripts','policy/Pi_0']:
        files += list((REPO/folder).rglob('*.py'))
    files += list((REPO/'task_config').rglob('*.yml'))+list((REPO/'task_config').rglob('*.yaml'))
    return sorted(set(p.resolve() for p in files if p.is_file()))


def verify_runtime(path,digest):
    if sha(path)!=digest:raise ValueError('Runtime manifest changed')
    manifest=read(path)
    for name,expected in manifest['source_hashes'].items():
        if sha(name)!=expected:raise ValueError('Runtime source drift: '+name)
    return manifest


def freeze(out):
    from admit_original_query import admit
    admission=admit(HERE/'original_online_full_squeeze_reference_v3/query')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    sources={str(p):sha(p) for p in source_files()}
    snapshot=out/'SOURCE_SNAPSHOT';snapshot.mkdir()
    # Full absolute source names and digests map every content-addressed copy.
    for source,digest in sources.items():
        destination=snapshot/digest
        if not destination.exists():destination.write_bytes(Path(source).read_bytes())
    runtime={'version':'DUMP_ROOT200002_ORIGINAL_AF_NATIVE_V1','created_utc':now(),
             'root_scope':[200002],'source_hashes':sources,'engineering_admission':admission,
             'sensor':'SAPIEN_RENDER_CAMERA_DEPTH_ON_EXACT_PHYSICAL_SURFACE',
             'squeeze':'ORIGINAL_OUTER_AND_ORIGINAL_INNER_POSITION_FEEDBACK',
             'policy_checkpoint':str(BASE/'checkpoints/pi0_robotwin_30000/30000')}
    write(out/'RUNTIME_MANIFEST.json',runtime)
    protocol={'version':'DUMP_ORIGINAL_AF_ROOTLOCAL_COLLECTION_V1','created_utc':now(),
              'root_scope':[200002],'task':'dump_bin_bigbin','policy_seed':30200002,
              'force_support_N':[.5,8.],'forces_N':FORCES,'frictions_by_split':FRICTIONS,
              'split_unit':'complete physical query setting (root, friction); all branches together',
              'cross_root_generalization':False,'original_utility_normalization_N':5.,
              'planned_training_validation_branches':128,'planned_test_rollouts':24,
              'protocol_text_sha256':sha(HERE/'ROOTLOCAL_COLLECTION_PROTOCOL.md'),
              'runtime_manifest_path':str(out/'RUNTIME_MANIFEST.json'),
              'runtime_manifest_sha256':sha(out/'RUNTIME_MANIFEST.json'),
              'test_not_collected_until_both_checkpoint_selection_locks':True}
    write(out/'COLLECTION_PROTOCOL.json',protocol)
    (out/'ROOTLOCAL_COLLECTION_PROTOCOL.md').write_bytes((HERE/'ROOTLOCAL_COLLECTION_PROTOCOL.md').read_bytes())
    contexts=[]
    for split,mus in FRICTIONS.items():
        for index,mu in enumerate(mus):
            context={'id':f'{split.lower()}_mu{mu:.3f}_root200002','split':split,'friction':mu,
                     'root':200002,'policy_seed':30200002,'task':'dump_bin_bigbin',
                     'force_support_N':[.5,8.],'forces_N':FORCES if split!='TEST' else None,
                     'runtime_manifest_path':str(out/'RUNTIME_MANIFEST.json'),
                     'runtime_manifest_sha256':sha(out/'RUNTIME_MANIFEST.json'),
                     'collection_protocol_path':str(out/'COLLECTION_PROTOCOL.json'),
                     'collection_protocol_sha256':sha(out/'COLLECTION_PROTOCOL.json')}
            contexts.append(context)
    write(out/'CONTEXTS.json',contexts)
    write(out/'FREEZE_LOCK.json',{'protocol_sha256':sha(out/'COLLECTION_PROTOCOL.json'),
                                'runtime_sha256':sha(out/'RUNTIME_MANIFEST.json'),
                                'contexts_sha256':sha(out/'CONTEXTS.json')})
    print(json.dumps({'frozen':str(out),'contexts':len(contexts),'TRAIN_VAL_branches':128},indent=2))


def validate_context(context,env):
    if os.environ.get('AF_P4_PHYSICAL_SURFACE_CAMERA')!='1' or os.environ.get('AF_ORIGINAL_SQUEEZE_INNER')!='1':
        raise ValueError('Formal collection cannot use diagnostic sensor/controller fallback')
    verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
    pp=Path(context['collection_protocol_path'])
    if sha(pp)!=context['collection_protocol_sha256']:raise ValueError('Collection protocol changed')
    lock=read(pp.parent/'FREEZE_LOCK.json')
    if sha(pp.parent/'CONTEXTS.json')!=lock['contexts_sha256']:raise ValueError('Context schedule changed')
    if context not in read(pp.parent/'CONTEXTS.json'):raise ValueError('Unplanned context')
    if context['split'] not in ('TRAIN','VAL'):raise ValueError('TEST collection before checkpoint lock forbidden')
    if context['root']!=200002 or context['task']!='dump_bin_bigbin':raise ValueError('Wrong scope')
    if context['forces_N']!=FORCES or context['force_support_N']!=[.5,8.]:raise ValueError('Unplanned forces')
    if float(env.af_contact_friction)!=context['friction'] or env._af_qualification_policy_seed!=context['policy_seed']:
        raise ValueError('Native setup differs from context lock')


def finalize_context(context,out,summary):
    verify_runtime(context['runtime_manifest_path'],context['runtime_manifest_sha256'])
    results=summary['results']
    if len(results)!=len(context['forces_N']) or any(r['kind']!='done' for r in results):
        raise ValueError('Incomplete/unknown branches; not a finished context')
    if len(summary['first_chunk_hashes'])!=len(results) or len(set(summary['first_chunk_hashes']))!=1:
        raise ValueError('First chunk pairing failed')
    feature_paths=sorted(out.glob('branch_*/original_motion_feature.json'))
    if len(feature_paths)!=len(results) or len({sha(p) for p in feature_paths})!=1:
        raise ValueError('Pre-action features differ across paired candidates')
    rows=[]
    for index,(force,record) in enumerate(zip(context['forces_N'],results)):
        result=record['result'];branch=out/f'branch_{index}_{force:g}N'
        if result['force_setpoint_bilateral_n']!=force or not result['completed']:
            raise ValueError('Incomplete/mismatched force branch')
        if result['success']!=result['official_final_check']:
            raise ValueError('Official success checks disagree')
        rows.append({'context_id':context['id'],'task':'dump_bin_bigbin','root':200002,
                     'split':context['split'],'force':force,'true_mu_training_only':context['friction'],
                     'full_task_success_y':int(result['success']),'reference':str(out),
                     'job':str(branch),'result_sha256':sha(branch/'result.json'),
                     'feature_sha256':sha(branch/'original_motion_feature.json')})
    write(out/'COLLECTED_ROWS.json',rows)
    write(out/'LOGICAL_COMPLETION.json',{'completed':True,'exit_code':0,'context_id':context['id'],
         'rows':len(rows),'runtime_manifest_sha256':context['runtime_manifest_sha256'],
         'rows_sha256':sha(out/'COLLECTED_ROWS.json'),'summary_sha256':sha(out/'online_qualification.json'),
         'source_hashes_after_verified':True,'finished_utc':now()})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze']);ap.add_argument('out',type=Path)
    args=ap.parse_args();freeze(args.out)
