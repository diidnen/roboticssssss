#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    # Repeated identical force first; different force last. No outcomes used.
    for repeat, force in enumerate((3.0, 3.0, 5.0)):
        out = args.out / f'repeat_{repeat}_force_{force:g}'
        log = args.out / f'repeat_{repeat}.log'
        env = dict(os.environ, ROBOTWIN_SUPPRESS_EVAL_CONFIG='1')
        env['PATH'] = str(args.repo.parent / 'runtime_bin') + os.pathsep + env['PATH']
        command = [sys.executable, str(Path(__file__).with_name('probe_replay_worker.py')),
                   '--repo', str(args.repo), '--out', str(out), '--force', str(force)]
        with log.open('x') as stream:
            p = subprocess.run(command, cwd=args.repo, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        evidence = None
        for line in log.read_text().splitlines():
            if line.startswith('ACTIVEFORCING_EVIDENCE '):
                evidence = json.loads(line.split(' ', 1)[1])
        if p.returncode or evidence is None:
            raise RuntimeError(f'Engineering probe failed; inspect {log}')
        handoff = json.loads((out / 'handoff_readback.json').read_text())
        record = {'repeat': repeat, 'force_n': force, 'handoff': handoff, 'evidence': evidence,
                  'physical_state_sha256': handoff['physical_state_sha256'],
                  'query_sha256': digest(evidence['query_info']['trace']),
                  'first_chunk_sha256': digest(evidence['preaction_sequence']),
                  'preaction_state_sha256': digest(evidence['preaction_state']),
                  'downstream_action_steps': evidence['rollout_steps']}
        results.append(record)
        (out / 'preaction_record.json').write_text(json.dumps(record, indent=2))
        print(json.dumps({k: record[k] for k in ('repeat', 'force_n', 'physical_state_sha256', 'query_sha256', 'first_chunk_sha256', 'downstream_action_steps')}), flush=True)
    report = {'scope': 'engineering same-root preaction replay; not training or task-success evaluation', 'repeats': results,
        'same_force_replay_equal': {k: results[0][k] == results[1][k] for k in ('physical_state_sha256', 'query_sha256', 'first_chunk_sha256', 'preaction_state_sha256')},
        'same_force_object_position_delta_m': float(np.linalg.norm(np.array(results[0]['handoff']['object_pose']['p']) - np.array(results[1]['handoff']['object_pose']['p']))),
        'same_force_first_chunk_max_delta': float(np.max(np.abs(np.array(results[0]['evidence']['preaction_sequence']) - np.array(results[1]['evidence']['preaction_sequence']))))}
    (args.out / 'replay_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'repeats'}, indent=2), flush=True)

if __name__ == '__main__':
    main()
