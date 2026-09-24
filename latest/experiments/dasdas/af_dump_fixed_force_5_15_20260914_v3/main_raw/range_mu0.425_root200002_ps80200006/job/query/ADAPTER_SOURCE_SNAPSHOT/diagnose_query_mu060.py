"""Read-only experiment evidence inspection; no threshold/protocol edits."""
import json
from pathlib import Path
import numpy as np
from rootlocal_collection_contract import HERE, read, sha, write
from original_p4_native_protocol import load_protocol


def inspect(mu):
    query = HERE / f'original_rootlocal_dataset_v1/groups/train_mu{mu:.3f}_root200002/attempt_001/job/query'
    protocol = load_protocol()
    rows = read(query / 'original_raw_rows.json')
    controls = read(query / 'native_controls.json')
    inner = read(query / 'original_squeeze_inner_trace.json')
    result = {'friction': mu, 'summary': read(query / 'original_probe_summary.json'),
              'thresholds': {'finger_force_min_N': protocol.FINGER_FORCE_MIN_N,
                             'contact_force_eps_N': protocol.CONTACT_FORCE_EPS_N},
              'probe_rows': [], 'failure_frames': [], 'source_hashes': {}}
    for row in rows:
        if not row['probe_phase'].startswith('probe_'): continue
        result['probe_rows'].append({key: row[key] for key in [
            'step', 'probe_phase', 't_s', 'measured_squeeze', 'measured_fn', 'left_fz', 'right_fz',
            'contact_state', 'stop_trigger', 'force_target', 'gripper_opening', 'marker_motion',
            'object_x_priv', 'object_y_priv', 'object_z_priv']})
        failed = row['contact_state'] != 'bilateral' or row['measured_fn'] < protocol.CONTACT_FORCE_EPS_N
        if failed:
            capture = read(query / f"query_{row['step']:04d}.json")
            end = sum(c['native_steps'] for c in controls[:row['step']])
            result['failure_frames'].append({'row': row, 'capture': capture,
                'control': controls[row['step']-1], 'inner_window': inner[max(0,end-25):end+25]})
    for name in ['original_raw_rows.json', 'original_probe_summary.json', 'patch_readbacks.json',
                 'native_controls.json', 'original_squeeze_inner_trace.json', 'reset.json']:
        result['source_hashes'][str(query/name)] = sha(query/name)
    result['all_failed_frames_dropped'] = False
    return result


def main():
    out = HERE / 'mu060_contact_diagnosis_v1'
    out.mkdir(exist_ok=False)
    results = [inspect(mu) for mu in [.7, .65, .6]]
    write(out/'RAW_COMPARISON.json', {'results': results, 'source_sha256': sha(__file__),
                                   'changes_to_experiment': False})
    for result in results:
        print(json.dumps({'friction': result['friction'], 'thresholds': result['thresholds'],
            'probe_failure': result['summary']['probe_failure'], 'probe_rows': result['probe_rows'],
            'failed_steps': [v['row']['step'] for v in result['failure_frames']]}), flush=True)
    for frame in results[-1]['failure_frames']:
        print('FAILED_FRAME '+json.dumps(frame), flush=True)


if __name__ == '__main__': main()
