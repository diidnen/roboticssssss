#!/usr/bin/env python3
"""Read-only audit of baseline artifacts and dump data; write a new report only."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

BASE = Path('/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911')
FORTE = Path('/home/exouser/FORTE')
REFERENCE = Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / 'original_runtime_audit.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    report = {'scope': 'root 200002 only; engineering diagnostics', 'baseline_manifest': str(REFERENCE / 'CONTINUOUS_FRICTION_RUNTIME_MANIFEST.json')}
    manifest = json.loads(Path(report['baseline_manifest']).read_text())
    report['baseline_manifest_sha256'] = sha(report['baseline_manifest'])
    report['source_hash_checks'] = []
    for path, expected in manifest['source_hashes'].items():
        actual = sha(path) if Path(path).is_file() else None
        report['source_hash_checks'].append({'path': path, 'expected': expected, 'actual': actual, 'passed': actual == expected})
    sys.path.insert(0, str(FORTE))
    sys.path.insert(0, str(REFERENCE / 'SOURCE_SNAPSHOT'))
    report['original_component_loading'] = {}
    try:
        from current_fulltask_feasibility_runtime import CurrentFeasibility
        from phase_free_feasibility import PhaseFreeFeasibility, KEEP_INPUT
        original = CurrentFeasibility(device='cpu')
        phase_free = PhaseFreeFeasibility(device='cpu')
        report['original_component_loading']['feasibility'] = {'passed': True, 'members': len(original.models), 'phase_free_gru_input': phase_free.models[0].command_gru.input_size}
        trainer_path = FORTE / 'analysis/results/current_fulltask_feasibility_baseline_v1_20260906/train_current_fulltask_feasibility_baseline_20260906.py'
        trainer = module('original_af_frozen_trainer', trainer_path)
        saved_rows = trainer.load_rows('VAL')
        row = saved_rows[0]
        job = Path(row['row']['job'])
        posterior_files = [job / 'PREACTION_POSTERIOR.json'] if (job / 'PREACTION_POSTERIOR.json').is_file() else []
        report['parity_source'] = {'job': str(job), 'posterior_files': [str(p) for p in posterior_files]}
        if posterior_files:
            posterior = json.loads(posterior_files[0].read_text())
            a = original.select(row['x'], posterior)
            b = phase_free.select(row['x'][:, KEEP_INPUT], posterior)
            delta = float(np.max(np.abs(np.asarray(a['p_success']) - np.asarray(b['p_success']))))
            report['original_phase_free_parity'] = {'max_probability_delta': delta, 'original_force': a['selected_force_N'], 'phase_free_force': b['selected_force_N'], 'passed': delta <= 1e-6 and a['selected_force_N'] == b['selected_force_N']}
    except Exception as exc:
        report['original_component_loading']['feasibility_error'] = repr(exc)
    try:
        belief_path = FORTE / 'analysis/results/current_multitask58_loader_candidate_20260905/continuous_belief.py'
        belief_mod = module('original_af_continuous_belief', belief_path)
        belief_manifest = FORTE / 'analysis/results/current_matched_runtime_collection_v5_20260906/BELIEF_MANIFEST.json'
        belief = belief_mod.ContinuousBelief(belief_manifest, manifest_sha256=sha(belief_manifest))
        report['original_component_loading']['belief'] = {'passed': True, 'members': len(belief.models), 'input_channels': 58, 'sigma_head_present': all(hasattr(m, 'log_sigma_head') for m in belief.models)}
    except Exception as exc:
        report['original_component_loading']['belief_error'] = repr(exc)
    exp = BASE / 'experiments/af_dump_bin_bigbin_rootlocal_forcecal_20260912'
    paths = [exp / 'branches.jsonl'] + [exp / suffix / 'branches.jsonl' for suffix in ('test_130200002', 'final_140200002', 'final_150200002')]
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    assert all(r['actual_seed'] == 200002 and r['task'] == 'dump_bin_bigbin' for r in rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['policy_seed'], row['friction'])].append(row)
    pairs = []
    for (seed, mu), siblings in sorted(grouped.items()):
        if len(siblings) < 2:
            continue
        state_arrays = np.asarray([r['evidence']['preaction_state'] for r in siblings], dtype=float)
        pairs.append({'policy_seed': seed, 'friction': mu, 'branches': len(siblings),
            'unique_handoff_states': len({canonical_sha(r['evidence']['preaction_state']) for r in siblings}),
            'unique_first_chunks': len({canonical_sha(r['evidence']['preaction_sequence']) for r in siblings}),
            'unique_queries': len({canonical_sha(r['evidence']['query_info']['trace']) for r in siblings}),
            'max_state_spread': float(np.max(np.ptp(state_arrays, axis=0)))})
    report['dump_data'] = {'rows': len(rows), 'independent_roots': 1, 'physical_contexts_root_times_friction': len({r['friction'] for r in rows}),
        'groups': pairs, 'groups_with_state_mismatch': sum(p['unique_handoff_states'] > 1 for p in pairs),
        'groups_with_action_mismatch': sum(p['unique_first_chunks'] > 1 for p in pairs),
        'saved_query_fields': sorted(rows[0]['evidence']['query_info']['trace'][0]),
        'saved_action_shape': list(np.asarray(rows[0]['evidence']['preaction_sequence']).shape),
        'saved_state_dim': len(rows[0]['evidence']['preaction_state']),
        'full_original_probe_rows_present': any('RAW_PROBE' in r['evidence'] for r in rows),
        'original_contact_patch_readback_present': any('CONTACT_PATCH_READBACK' in r['evidence'] for r in rows),
        'branch_file_hashes': {str(p): sha(p) for p in paths}}
    report['confirmed_contract_deviations'] = [
        '58D mean+sigma belief replaced by 15D mean-only model; privileged actor position used in the replacement',
        'positive-support posterior integration replaced by point/friction-band routing',
        'original expected utility replaced by success-probability thresholds',
        'original Cartesian motion/state/mask interface replaced by 14D joint actions and 14D joint state',
        'isotonic replacement ignores motion and state',
        'saved RoboTwin probe lacks original tactile-marker and per-finger patch evidence',
        'root-local train code added monotonic regularization despite repair request disabling it',
        'legacy repaired-model CPU best-state snapshots alias live weights without clone/deepcopy',
        'new early-contact-loss terminal shortcut was not independently validated against the original horizon',
    ]
    report['causal_status'] = 'Implementation/protocol differences confirmed; contribution of each difference to physical failures unresolved. No claim of inherent task failure.'
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps({'report': str(output), 'sha256': sha(output), 'components': report['original_component_loading'],
        'parity': report.get('original_phase_free_parity'), 'rows': len(rows),
        'paired_groups': len(pairs), 'state_mismatches': report['dump_data']['groups_with_state_mismatch'],
        'action_mismatches': report['dump_data']['groups_with_action_mismatch'],
        'source_hash_failures': [r['path'] for r in report['source_hash_checks'] if not r['passed']]}, indent=2))

if __name__ == '__main__':
    main()
