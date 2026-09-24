#!/usr/bin/env python3
import json
from pathlib import Path

out = Path('/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments/af_dump_bin_bigbin_force_repair_20260912_v2')
report = json.loads((out / 'fresh_online_report.json').read_text())
s = report['summary']
lines = [
    '# Fresh test summary',
    '',
    'This report uses two fresh roots and six unseen contexts. AF force selections were locked before any downstream outcome was observed.',
    '',
    f"- Contexts: {s['contexts']}; fresh branches: 54.",
    f"- ActiveForcing: {s['activeforcing']['successes']}/{s['contexts']} = {s['activeforcing']['success_rate']:.3f}; mean selected force {s['activeforcing']['mean_selected_force_n']:.2f} N.",
    f"- Fixed 3 N: {s['fixed']['3.0']['successes']}/{s['contexts']} = {s['fixed']['3.0']['success_rate']:.3f}; fixed 4.25 N: {s['fixed']['4.25']['successes']}/{s['contexts']} = {s['fixed']['4.25']['success_rate']:.3f}; fixed 5 N: {s['fixed']['5.0']['successes']}/{s['contexts']} = {s['fixed']['5.0']['success_rate']:.3f}.",
    f"- Grid oracle: {s['empirical_grid_oracle']['successes']}/{s['contexts']} = {s['empirical_grid_oracle']['success_rate']:.3f}; all-grid-fail contexts: {s['failure_decomposition']['all_grid_fail_contexts']}.",
    f"- Force-sensitive grid contexts: {s['failure_decomposition']['force_sensitive_grid_contexts']}; AF selection failures: {s['failure_decomposition']['force_selection_failures']}.",
    '',
    '## Interpretation',
    '',
    'AF matched the fixed-force baselines (3/6) and did not outperform them. One fresh root succeeded at every tested force and the other failed at every tested force, so the task-level trajectory/contact failure dominated force choice. The single force-sensitive grid context was not rescued by the locked AF selection.',
    '',
    'This is a valid negative result for dump_bin_bigbin: the repaired pipeline, query, force authority, root-disjoint training, and outcome-blind fresh evaluation all ran correctly, but this task does not provide evidence that AF improves success over fixed force.',
]
(out / 'fresh_test_summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
