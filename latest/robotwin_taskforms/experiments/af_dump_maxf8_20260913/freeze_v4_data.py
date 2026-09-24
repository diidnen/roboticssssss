"""Freeze finite added data, split boundaries, and future same-root confirmation seeds."""
from copy import deepcopy
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'af_dump_original_restore_20260912'
sys.path.insert(0, str(OLD))
from rootlocal_collection_contract import read, write, sha, now, verify_runtime


def main():
    prior = OLD/'original_rootlocal_dataset_v3_rim20'
    lock = read(prior/'FREEZE_LOCK.json')
    runtime = deepcopy(verify_runtime(prior/'RUNTIME_MANIFEST.json', lock['runtime_sha256']))
    out = HERE/'additional_data_v1'; out.mkdir(exist_ok=False)
    for name in ['collect_v4_context.py', 'run_v4_collection.py', 'freeze_v4_data.py']:
        runtime['source_hashes'][str(HERE/name)] = sha(HERE/name)
    runtime.update(extension_version='MAXF8_DATA_V4', extension_created_utc=now(),
                   original_runtime_manifest_sha256=lock['runtime_sha256'])
    write(out/'RUNTIME_MANIFEST.json', runtime)
    dense = [0.5, 2.0, 4.0, 5.0, 6.0, 6.5, 7.0, 8.0]
    schedule = []
    for split, mus in [('TRAIN', [.31,.36,.41,.46,.51,.56,.61,.66,.71,.76,.81,.84]),
                       ('VAL', [.34,.44,.64,.74])]:
        for mu in mus: schedule.append((split, mu, 30200002, 'QUERY_ONLY'))
    for split, mus in [('TRAIN', [.85,.70,.55,.725,.575,.425]), ('VAL', [.775,.475])]:
        for mu in mus: schedule.append((split, mu, 40200002, 'FULL_GROUP'))
    old_contexts = read(prior/'CONTEXTS.json')
    splits = {s: {c['friction'] for c in old_contexts if c['split'] == s} for s in ['TRAIN','VAL','TEST']}
    for split, mu, _, _ in schedule:
        if any(mu in values for other, values in splits.items() if other != split):
            raise ValueError('Friction leaked across declared split')
        splits[split].add(mu)
    protocol = {'version': 'MAXF8_ADDITIONAL_DATA_V1', 'created_utc': now(), 'root_scope': [200002],
                'task': 'dump_bin_bigbin', 'force_support_N': [0.5,8.0], 'forces_N': dense,
                'utility_normalization_N': 8.0, 'utility_definition': 'p*(maxF-F)/maxF-(1-p)',
                'reused_formal_data': str(prior), 'reused_labels': 128,
                'new_full_groups': 8, 'new_downstream_labels': 64, 'new_query_only_groups': 16,
                'query_only_labels_not_counted_as_downstream': True,
                'query_feature_duplicates': 'deduplicate within split for belief; preserve every rollout for feasibility',
                'split_unit': 'friction/query setting; all policy seeds and forces retained in same split',
                'no_hidden_mu_in_belief_inputs': True, 'no_outcome_based_resampling': True,
                'confirmation': {'root': 200002, 'frictions': [.375,.525,.675,.825],
                                 'new_policy_seeds': [50200002,60200002],
                                 'methods': ['ActiveForcing','Fixed-1N','Fixed-3N','Fixed-5N','Fixed-6N','Fixed-8N'],
                                 'rollouts': 48, 'new_rollouts_after_training_locks_only': True,
                                 'prior_TEST_not_claimed_untouched': True},
                'runtime_manifest_path': str(out/'RUNTIME_MANIFEST.json'),
                'runtime_manifest_sha256': sha(out/'RUNTIME_MANIFEST.json')}
    write(out/'COLLECTION_PROTOCOL.json', protocol)
    contexts = []
    for split, mu, seed, mode in schedule:
        contexts.append({'id': f'{split.lower()}_mu{mu:.3f}_ps{seed}_{mode.lower()}',
                         'split': split, 'friction': mu, 'root': 200002, 'policy_seed': seed,
                         'task': 'dump_bin_bigbin', 'forces_N': dense, 'force_support_N': [0.5,8.0],
                         'collection_mode': mode, 'runtime_manifest_path': str(out/'RUNTIME_MANIFEST.json'),
                         'runtime_manifest_sha256': sha(out/'RUNTIME_MANIFEST.json'),
                         'collection_protocol_path': str(out/'COLLECTION_PROTOCOL.json'),
                         'collection_protocol_sha256': sha(out/'COLLECTION_PROTOCOL.json')})
    write(out/'CONTEXTS.json', contexts)
    write(out/'FREEZE_LOCK.json', {'runtime_sha256': sha(out/'RUNTIME_MANIFEST.json'),
          'protocol_sha256': sha(out/'COLLECTION_PROTOCOL.json'), 'contexts_sha256': sha(out/'CONTEXTS.json')})
    print({'frozen': str(out), 'new_queries_only': 16, 'new_full_groups': 8, 'new_labels': 64}, flush=True)


if __name__ == '__main__': main()
