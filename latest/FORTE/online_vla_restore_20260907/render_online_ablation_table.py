"""Render complete, verified online development ablations, never fresh-test SR."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(source, destination):
    data = json.loads(source.read_text())
    if (data.get('role') != 'ONLINE_VLA_BURNED_ROOT_ABLATION_EVIDENCE'
            or data.get('ablation_branches') != 96
            or data.get('AF_control_branches') != 24
            or data.get('contexts_per_method') != 24
            or data.get('roots') != [5100, 6200]
            or data.get('scripted_rollouts_included') is not False
            or data.get('final_fresh_roots_used') is not False):
        raise RuntimeError('Complete verified online development evidence required')
    for path, digest in data['source_hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('Ablation source evidence changed: ' + path)
    if sha(data['source_protocol']) != data['source_protocol_sha256']:
        raise RuntimeError('Ablation protocol changed')
    rows = data['rows']
    methods = list(data['main'])
    labels = {'ACTIVEFORCING': 'ActiveForcing', 'COARSE_GRID': 'Coarse Grid',
              'LOCAL_LIFT': 'Local-Lift', 'POSTERIOR_MEAN': 'Posterior Mean',
              'PRIOR_NO_POSTERIOR': 'Prior / No-Posterior'}
    if set(methods) != set(labels):
        raise RuntimeError('Unexpected ablation variant identity')
    if len(methods) != 5 or len(rows) != 120:
        raise RuntimeError('Ablation denominator mismatch')
    output = []
    for method in methods:
        group = [r for r in rows if r['method'] == method]
        if len(group) != 24 or len({r['context'] for r in group}) != 24:
            raise RuntimeError('Missing or duplicate ablation context')
        for row in group:
            if sha(Path(row['job']) / 'BRANCH_RESULT.json') != row['result_sha256']:
                raise RuntimeError('Ablation outcome changed')
        count = sum(r['full_task_success'] for r in group)
        force = sum(r['selected_force_N'] for r in group) / 24
        actual = sum(r['measured_squeeze_N'] for r in group) / 24
        summary = data['main'][method]
        if (count != summary['full_successes']
                or abs(count / 24 - summary['FULL_TASK_SR']) > 1e-12
                or abs(force - summary['MEAN_SELECTED_SETPOINT']) > 1e-12
                or abs(actual - summary['MEASURED_BILATERAL_SQUEEZE']) > 1e-12):
            raise RuntimeError('Ablation summary differs from source rows')
        output.append(dict(method=method,successes=count,contexts=24,
                           selected_force_N=force,measured_squeeze_N=actual))
    destination.mkdir(exist_ok=False)
    with (destination / 'online_dev_ablation_table.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    tex = [r'\begin{table*}[t]', r'\centering',
           r'\caption{Online VLA ablations on two burned development roots (24 contexts per method). These results are separate from the fresh-root main evaluation. The prior variant retains the physical probe.}',
           r'\label{tab:online_dev_ablations}', r'\begin{tabular}{lrrr}', r'\hline',
           r'Method & Full task & Setpoint (N) & Squeeze (N) \\', r'\hline']
    for row in output:
        label = labels[row['method']]
        tex.append(f"{label} & {row['successes']}/24 & {row['selected_force_N']:.2f} & {row['measured_squeeze_N']:.2f} " + r'\\')
    tex.extend([r'\hline', r'\end{tabular}', r'\end{table*}'])
    (destination / 'online_dev_ablation_table.tex').write_text('\n'.join(tex) + '\n')
    (destination / 'INTERPRETATION.md').write_text(
        'Four online variants and matched ActiveForcing controls use the same 24 development contexts. '
        'This table is not additional fresh-test evidence. Prior / No-Posterior retains the probe; '
        'no legal NoProbe experiment is inferred. Local-Lift uses explicitly identified historical '
        'controlled-motion supervision. Counts include original failures without outcome selection.\n')
    artifacts = {p.name: sha(p) for p in destination.iterdir() if p.is_file()}
    (destination / 'RENDER_MANIFEST.json').write_text(json.dumps(dict(
        source=str(source), source_sha256=sha(source), implementation_sha256=sha(__file__),
        artifacts=artifacts, role='ONLINE_DEVELOPMENT_ABLATION_TABLE',
        final_fresh_root_results=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.destination)
