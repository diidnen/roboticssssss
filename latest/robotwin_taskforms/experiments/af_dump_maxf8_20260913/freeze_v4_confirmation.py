"""Freeze the prospectively declared same-root confirmation before model completion."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'af_dump_original_restore_20260912'
sys.path.insert(0,str(OLD))
from rootlocal_collection_contract import read,write,sha,now,verify_runtime


def main():
    added=HERE/'additional_data_v1';data_lock=read(added/'FREEZE_LOCK.json')
    if sha(added/'COLLECTION_PROTOCOL.json')!=data_lock['protocol_sha256']:
        raise ValueError('Prospective confirmation specification changed')
    prior=read(added/'COLLECTION_PROTOCOL.json');plan=prior['confirmation']
    if plan['root']!=200002 or plan['new_policy_seeds']!=[50200002,60200002] or plan['rollouts']!=48:
        raise ValueError('Wrong prospective confirmation')
    verify_runtime(added/'RUNTIME_MANIFEST.json',data_lock['runtime_sha256'])
    out=HERE/'confirmation_plan_v1';out.mkdir(exist_ok=False)
    sources={str(HERE/name):sha(HERE/name) for name in ['freeze_v4_confirmation.py','infer_v4.py',
        'run_v4_inference.py','audit_v4_inference.py','test_v4_online_bindings.py','maxf8_runtime.py','max_force_utility.py']}
    protocol={'version':'MAXF8_SAME_ROOT_CONFIRMATION_V1','created_utc':now(),
              'root_scope':[200002],'task':'dump_bin_bigbin','force_support_N':[.5,8.],
              'utility_normalization_N':8.,'prospective_specification':plan,
              'additional_collection_protocol_sha256':data_lock['protocol_sha256'],
              'source_hashes':sources,'physical_friction_settings':4,'decision_contexts':8,
              'paired_rollouts':48,'no_test_based_retraining_or_reselection':True,
              'prior_frictions_known_new_policy_seeds_not_yet_executed':True,
              'runtime_manifest_path':str(added/'RUNTIME_MANIFEST.json'),
              'runtime_manifest_sha256':data_lock['runtime_sha256']}
    write(out/'COLLECTION_PROTOCOL.json',protocol)
    contexts=[]
    for seed in plan['new_policy_seeds']:
        for mu in plan['frictions']:
            contexts.append({'id':f'test_mu{mu:.3f}_root200002_ps{seed}','split':'TEST',
                'friction':mu,'root':200002,'policy_seed':seed,'task':'dump_bin_bigbin',
                'force_support_N':[.5,8.],'forces_N':None,
                'runtime_manifest_path':str(added/'RUNTIME_MANIFEST.json'),'runtime_manifest_sha256':data_lock['runtime_sha256'],
                'collection_protocol_path':str(out/'COLLECTION_PROTOCOL.json'),
                'collection_protocol_sha256':sha(out/'COLLECTION_PROTOCOL.json')})
    write(out/'CONTEXTS.json',contexts)
    write(out/'FREEZE_LOCK.json',{'runtime_sha256':data_lock['runtime_sha256'],
          'protocol_sha256':sha(out/'COLLECTION_PROTOCOL.json'),'contexts_sha256':sha(out/'CONTEXTS.json')})
    print({'frozen':str(out),'contexts':len(contexts),'physical_friction_settings':4,'rollouts':48},flush=True)


if __name__=='__main__':main()
