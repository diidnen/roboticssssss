"""Retrospective utility correction on sealed curves; never assigns unexecuted success."""
import argparse
import hashlib
import json
from pathlib import Path
from max_force_utility import select_force


def main(old_inference, out):
    rows = []
    for case in sorted(old_inference.glob('test_mu*_root200002')):
        path = case/'job/PREACTION_AF_DECISION.json'
        old = json.loads(path.read_text())
        corrected = select_force(old['force_grid_N'], old['p_success'], [0.5, 8])
        rows.append({'context': case.name, 'old_force_N': old['selected_force_N'],
                     'source_curve_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'corrected_decision': corrected, 'new_online_outcome': None,
                     'retrospective_only': True})
    with out.open('x') as stream:
        json.dump({'completed': True, 'method': 'same sealed probabilities; maxF8 utility correction only',
                   'online_executions': 0, 'rows': rows}, stream, indent=2, allow_nan=False)
    print(json.dumps([{'context': r['context'], 'old_force_N': r['old_force_N'],
                       'corrected_force_N': r['corrected_decision']['selected_force_N']} for r in rows]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('old_inference', type=Path); parser.add_argument('out', type=Path)
    args = parser.parse_args(); main(args.old_inference, args.out)
