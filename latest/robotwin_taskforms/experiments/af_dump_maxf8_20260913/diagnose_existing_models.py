"""Bounded TRAIN/VAL-only belief convergence diagnosis; no simulation or TEST tuning."""
import argparse
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np

OLD = Path('/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments/af_dump_original_restore_20260912')
sys.path.insert(0, str(OLD))
import train_original_rootlocal as original
from rootlocal_collection_contract import read, write, sha, now


def extended_trainer():
    source = inspect.getsource(original.train_belief)
    tree = ast.parse(source)
    baseline = ast.dump(tree)
    changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 81:
            changes.append((node, node.value)); node.value = 401
        elif isinstance(node, ast.Constant) and node.value == 80:
            changes.append((node, node.value)); node.value = 400
    if len(changes) != 2:
        raise ValueError('Unexpected original training budget layout')
    executable = deepcopy(tree)
    for node, value in changes: node.value = value
    if ast.dump(tree) != baseline: raise ValueError('Training recipe changed beyond budget')
    namespace = dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(executable), str(OLD/'train_original_rootlocal.py'), 'exec'), namespace)
    return namespace['train_belief']


def main(out):
    records, inputs, lock = original.load_dataset(OLD/'original_rootlocal_dataset_v3_rim20')
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'version': 'BELIEF_CONVERGENCE_DIAG_MAX400_V1', 'created_utc': now(),
                'budget_change_only': {'old_updates': 80, 'new_updates': 400},
                'architecture_loss_optimizer_normalization_unchanged': True,
                'selection': 'per seed first minimum VAL Gaussian NLL',
                'no_test_features_or_labels': True, 'root_scope': [200002],
                'input_hashes': inputs, 'dataset_lock': lock,
                'utility_not_used_in_belief_training': True, 'future_utility_maxF_N': 8,
                'script_sha256': sha(__file__), 'original_trainer_sha256': sha(OLD/'train_original_rootlocal.py')}
    write(out/'PROTOCOL.json', protocol)
    print('DIAG_TRAIN_START '+now(), flush=True)
    extended_trainer()(records, out/'belief', out/'PROTOCOL.json', lock['runtime_sha256'])
    history = read(out/'belief/TRAINING_HISTORY.json')
    previous = read(OLD/'original_models_v3_rim20/belief/TRAINING_HISTORY.json')
    prefix = [r for r in history if r['update'] <= 80]
    if prefix != previous:
        raise ValueError('Original first80updates not reproduced exactly')
    comparison = []
    for record in records:
        old = read(OLD/'original_models_v3_rim20/belief'/(record['id']+'_POSTERIOR.json'))
        new = read(out/'belief'/(record['id']+'_POSTERIOR.json'))
        comparison.append({'id': record['id'], 'split': record['split'], 'true_mu': record['y'],
                           'old_mean': old['posterior_moments']['mean'], 'old_std': old['posterior_moments']['std'],
                           'new_mean': new['posterior_moments']['mean'], 'new_std': new['posterior_moments']['std']})
    summary = {}
    for split in ['TRAIN', 'VAL']:
        rows = [r for r in comparison if r['split'] == split]
        summary[split] = {'queries': len(rows),
                          'old_mae': float(np.mean([abs(r['old_mean']-r['true_mu']) for r in rows])),
                          'new_mae': float(np.mean([abs(r['new_mean']-r['true_mu']) for r in rows]))}
    for path, digest in inputs.items():
        if sha(path) != digest: raise ValueError('Input changed during diagnosis')
    result = {'completed': True, 'first80updates_exact': True, 'comparison': comparison, 'summary': summary,
              'selection': read(out/'belief/CHECKPOINT_SELECTION_LOCK.json'),
              'not_formal_online_inference': True, 'finished_utc': now()}
    write(out/'DIAGNOSTIC_RESULT.json', result)
    print(json.dumps({'summary': summary, 'checkpoints': result['selection']['checkpoints'],
                      'first80updates_exact': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('out', type=Path)
    main(parser.parse_args().out)
