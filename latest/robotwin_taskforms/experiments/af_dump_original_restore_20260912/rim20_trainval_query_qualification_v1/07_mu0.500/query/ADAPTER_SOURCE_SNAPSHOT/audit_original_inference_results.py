"""Post-run raw-evidence audit; does not run/reselect models or change TEST.

Reuses the raw impulse/action auditor, not its eight-branch collection wrapper.
The four-context, six-method checks below are independent of the TEST summary.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from rootlocal_collection_contract import read, sha, verify_runtime
from audit_original_collected_group import audit_query, audit_branch, same_or_write

METHODS = ['ActiveForcing', 'Fixed-1N', 'Fixed-3N', 'Fixed-5N', 'Fixed-6N', 'Fixed-8N']


def check_decision(decision):
    grid = np.round(np.arange(.5, 8.0001, .05), 8)
    np.testing.assert_array_equal(decision['force_grid_N'], grid)
    probability = np.asarray(decision['p_success'], dtype=float)
    if probability.shape != grid.shape or not np.isfinite(probability).all():
        raise ValueError('Invalid probability curve')
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('Probability outside [0,1]')
    utility = probability * (5. - grid) / 5. - (1. - probability)
    np.testing.assert_allclose(decision['expected_utility'], utility, rtol=0, atol=1e-12)
    chosen = int(np.argmax(utility))
    if decision['selected_force_N'] != float(grid[chosen]):
        raise ValueError('Original utility/lower-force tie rule not followed')
    np.testing.assert_allclose([decision['utility'], decision['predicted_success']],
                               [utility[chosen], probability[chosen]], rtol=0, atol=1e-12)
    if decision['utility_normalization_N'] != 5. or decision['posterior_sigma_used'] is not True:
        raise ValueError('Original posterior/utility interface changed')


def audit(inference):
    inference = Path(inference).resolve()
    protocol = read(inference / 'INFERENCE_PROTOCOL.json')
    final = read(inference / 'FINAL_INFERENCE_RESULTS.json')
    dataset = Path(protocol['dataset']); models = Path(protocol['models'])
    freeze = read(dataset / 'FREEZE_LOCK.json')
    if sha(dataset / 'CONTEXTS.json') != freeze['contexts_sha256']:
        raise ValueError('Frozen context schedule changed')
    contexts = [c for c in read(dataset / 'CONTEXTS.json') if c['split'] == 'TEST']
    if len(contexts) != 4 or protocol['contexts'] != contexts or protocol['methods'] != METHODS:
        raise ValueError('TEST schedule/method mismatch')
    if final['inference_protocol_sha256'] != sha(inference / 'INFERENCE_PROTOCOL.json'):
        raise ValueError('Inference protocol changed')
    training = read(models / 'TRAINING_COMPLETE.json')
    if not training['completed'] or training['test_groups_executed'] != 0:
        raise ValueError('Invalid training/TEST separation')
    if protocol['training_complete_sha256'] != sha(models / 'TRAINING_COMPLETE.json'):
        raise ValueError('Training lock changed')
    for path, digest in protocol['source_hashes'].items():
        if sha(path) != digest: raise ValueError('Inference source drift')
    cases = []; sources = {}; branch_count = 0
    for context in contexts:
        if context['root'] != 200002 or context['task'] != 'dump_bin_bigbin':
            raise ValueError('Out-of-scope TEST context')
        verify_runtime(context['runtime_manifest_path'], context['runtime_manifest_sha256'])
        case = inference / context['id']; job = case / 'job'
        exit_record = read(case / 'PROCESS_EXIT.json')
        result = read(job / 'AF_INFERENCE_RESULT.json')
        if exit_record['exit_code'] != 0 or exit_record['result_sha256'] != sha(job / 'AF_INFERENCE_RESULT.json'):
            raise ValueError('Missing/changed terminal TEST result')
        if result['context'] != context or not result['completed'] or not result['formal_AF_inference']:
            raise ValueError('Unqualified TEST result')
        seal = read(job / 'PREACTION_SELECTION_LOCK.json')
        if result['selection_lock_sha256'] != sha(job / 'PREACTION_SELECTION_LOCK.json'):
            raise ValueError('Selection lock changed')
        if seal['context'] != context or seal['candidate_actions_executed'] != 0 or seal['labels_read']:
            raise ValueError('Selection was not sealed pre-action')
        if seal['model_manifests'] != {'belief': training['belief_manifest_sha256'],
                                      'feasibility': training['feasibility_manifest_sha256']}:
            raise ValueError('Selected models differ from training lock')
        for name, digest in seal['artifact_hashes'].items():
            if sha(job / name) != digest: raise ValueError('Changed pre-action artifact')
        decision = read(job / 'PREACTION_AF_DECISION.json'); check_decision(decision)
        forces = [decision['selected_force_N'], 1., 3., 5., 6., 8.]
        if seal['branch_methods'] != METHODS or seal['forces_N'] != forces:
            raise ValueError('Unplanned force or method')
        chunk = np.load(job / 'PREACTION_PI0_CHUNK.npy', allow_pickle=False)
        if hashlib.sha256(chunk.tobytes()).hexdigest() != seal['first_chunk_sha256']:
            raise ValueError('Pre-action chunk changed')
        online = read(job / 'online_qualification.json')
        if online['first_chunk_hashes'] != [seal['first_chunk_sha256']] * 6:
            raise ValueError('Unpaired initial actions')
        if online['common_handoff_state_sha256'] != seal['state_sha256']:
            raise ValueError('Pre-action handoff changed')
        if len(online['results']) != 6 or len(result['outcomes']) != 6:
            raise ValueError('Incomplete paired candidates')
        audit_query(job / 'query')
        feature = read(job / 'PREACTION_FEATURE.json')
        rebuilt = []
        for index, (method, force) in enumerate(zip(METHODS, forces)):
            branch = job / f'branch_{index}_{force:g}N'
            audit_branch(branch)
            raw = read(branch / 'result.json')
            if online['results'][index] != {'kind': 'done', 'result': raw}:
                raise ValueError('Summary differs from raw branch result')
            if read(branch / 'original_motion_feature.json') != feature:
                raise ValueError('Unpaired motion feature')
            if raw['force_setpoint_bilateral_n'] != force:
                raise ValueError('Selected and executed force differ')
            y = int(raw['success'])
            rebuilt.append({'method': method, 'force_N': force, 'success': y,
                'original_utility': y*(5.-force)/5. + (1-y)*-1.,
                'actual_mean_force_N': raw['measured_mean_squeeze_n'],
                'native_actions': raw['native_actions'], 'result_sha256': sha(branch / 'result.json')})
            sources[str(branch / 'INDEPENDENT_BRANCH_AUDIT.json')] = sha(branch / 'INDEPENDENT_BRANCH_AUDIT.json')
            branch_count += 1
        if result['outcomes'] != rebuilt:
            raise ValueError('Reported outcomes differ from independently rebuilt outcomes')
        for path in [job / 'AF_INFERENCE_RESULT.json', job / 'PREACTION_SELECTION_LOCK.json',
                     job / 'query/INDEPENDENT_NATIVE_FORCE_REBUILD.json']:
            sources[str(path)] = sha(path)
        cases.append(result)
    summaries = []
    for index, method in enumerate(METHODS):
        rows = [case['outcomes'][index] for case in cases]
        summaries.append({'method': method, 'successes': sum(r['success'] for r in rows),
            'queries': len(rows), 'selected_forces_N': [r['force_N'] for r in rows],
            'mean_selected_force_N': sum(r['force_N'] for r in rows)/len(rows),
            'mean_original_utility': sum(r['original_utility'] for r in rows)/len(rows)})
    if final['cases'] != cases or final['summary'] != summaries:
        raise ValueError('Final aggregation differs from raw paired TEST evidence')
    if not final['completed'] or final['independent_query_settings'] != 4 or final['paired_rollouts'] != 24:
        raise ValueError('Incorrect completion/denominators')
    report = {'passed': True, 'independent_query_settings': 4, 'paired_rollouts_audited': branch_count,
        'final_result_sha256': sha(inference / 'FINAL_INFERENCE_RESULTS.json'),
        'audit_source_sha256': sha(__file__), 'raw_audit_hashes': sources,
        'backup_verification_separate_and_still_required': True}
    same_or_write(inference / 'INDEPENDENT_FINAL_RESULT_AUDIT.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('inference', type=Path)
    audit(parser.parse_args().inference)
