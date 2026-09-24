"""Check corrected runtime against sealed original probabilities with identical weights."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from maxf8_runtime import MaxF8Feasibility, OLD
from rootlocal_collection_contract import read, write, sha, now
from max_force_utility import select_force


def main():
    here = Path(__file__).resolve().parent
    out = here/'utility_only_runtime_v1'; out.mkdir(exist_ok=False)
    oldpath = OLD/'original_models_v3_rim20/feasibility/FEASIBILITY_MANIFEST.json'
    manifest = deepcopy(read(oldpath)); manifest['utility_normalization_N'] = 8.0
    manifest['utility_definition'] = 'p*(maxF-F)/maxF-(1-p)'
    manifest['reused_weights_manifest_sha256'] = sha(oldpath)
    manifest['utility_only_correction'] = True
    for name in ['maxf8_runtime.py', 'max_force_utility.py']:
        manifest['source_hashes'][str(here/name)] = sha(here/name)
    write(out/'FEASIBILITY_MANIFEST.json', manifest)
    torch.set_num_threads(2)
    runtime = MaxF8Feasibility(out/'FEASIBILITY_MANIFEST.json', manifest_sha256=sha(out/'FEASIBILITY_MANIFEST.json'))
    rows = []
    for case in sorted((OLD/'original_inference_v3_rim20').glob('test_mu*_root200002')):
        job = case/'job'; old = read(job/'PREACTION_AF_DECISION.json')
        decision = runtime.select(read(job/'PREACTION_FEATURE.json'), read(job/'PREACTION_POSTERIOR.json'))
        error = float(np.max(np.abs(np.asarray(decision['p_success'])-np.asarray(old['p_success']))))
        if error > 1e-6: raise ValueError('Utility correction changed original model probabilities')
        expected = select_force(old['force_grid_N'], old['p_success'], [0.5, 8])
        if decision['selected_force_N'] != expected['selected_force_N']:
            raise ValueError('Runtime differs from independent curve recomputation')
        rows.append({'id': case.name, 'max_probability_error': error,
                     'old_force_N': old['selected_force_N'], 'corrected_force_N': decision['selected_force_N']})
    write(out/'RUNTIME_PARITY.json', {'passed': True, 'cases': rows, 'weights_unchanged': True,
                                    'new_rollouts': 0, 'finished_utc': now()})
    print(json.dumps(rows), flush=True)


if __name__ == '__main__': main()
