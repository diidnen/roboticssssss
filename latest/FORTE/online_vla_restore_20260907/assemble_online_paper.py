"""Assemble results-bearing paper sources only after both verified tables exist.

This creates a review candidate, never a goal-completion or submission certificate.
Run after render_online_vla_main and render_online_ablation_table.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_render(directory, result):
    manifest = read(directory / 'RENDER_MANIFEST.json')
    if manifest['source_sha256'] != sha(result):
        raise RuntimeError('Rendered table is not from the supplied results')
    for name, digest in manifest['artifacts'].items():
        path = Path(name)
        if not path.is_absolute():
            path = directory / path
        if sha(path) != digest:
            raise RuntimeError('Rendered artifact changed: ' + str(path))
    return manifest


def assemble(main, ablations, figures, ablation_table, behavior, destination):
    data, dev = read(main), read(ablations)
    if (data.get('role') != 'FINAL_ONLINE_FROZEN_VLA_MAIN_EVIDENCE'
            or data.get('branches') != 192 or data.get('contexts') != 48
            or len(data.get('rows', [])) != 192
            or data.get('all_branch_sources') != 'ONLINE_VLA'
            or data.get('scripted_results_included') is not False):
        raise RuntimeError('Complete verified final online VLA main evidence required')
    if (dev.get('role') != 'ONLINE_VLA_BURNED_ROOT_ABLATION_EVIDENCE'
            or dev.get('ablation_branches') != 96
            or dev.get('AF_control_branches') != 24
            or len(dev.get('rows', [])) != 120
            or dev.get('final_fresh_roots_used') is not False
            or dev.get('scripted_rollouts_included') is not False):
        raise RuntimeError('Complete verified development ablations required')
    if set(data['roots']) & set(dev['roots']):
        raise RuntimeError('Fresh and development roots overlap')
    events = read(behavior)
    if (events.get('role') != 'FINAL_ONLINE_VLA_OBSERVABLE_EVENT_ANALYSIS'
            or events.get('branches') != 192 or len(events.get('rows', [])) != 192
            or events.get('main_results_sha256') != sha(main)
            or events.get('runtime_manifest_sha256') != data['runtime_manifest_sha256']
            or events.get('VLA_FAILURE_STAGE_LABEL_CAUSAL') is not True
            or events.get('scripted_downstream_stage_used') is not False):
        raise RuntimeError('Complete causal final behavior report required')
    if {row['job'] for row in events['rows']} != {row['job'] for row in data['rows']}:
        raise RuntimeError('Behavior report and main contexts differ')
    for row in events['rows']:
        if sha(Path(row['job']) / 'BRANCH_TRACE.json') != row['trace_sha256']:
            raise RuntimeError('Behavior trace evidence changed')
    for report in (data, dev):
        for path, digest in report['source_hashes'].items():
            if sha(path) != digest:
                raise RuntimeError('Underlying evidence changed: ' + path)
    verify_render(figures, main)
    verify_render(ablation_table, ablations)
    here = Path(__file__).resolve().parent
    discussion_source = read(here / 'PAPER_MAIN_DISCUSSION_SOURCE.json')
    if (discussion_source['main_results_sha256'] != sha(main)
            or discussion_source['behavior_report_sha256'] != sha(behavior)
            or discussion_source['discussion_sha256'] != sha(here / 'paper_online_main_discussion.tex')):
        raise RuntimeError('Main discussion does not match supplied evidence')
    inputs = [here / name for name in (
        'paper_online_introduction.tex', 'paper_online_method_section.tex',
        'paper_online_experiment_protocol.tex', 'paper_online_scope_and_limitations.tex',
        'paper_online_main_discussion.tex', 'PAPER_MAIN_DISCUSSION_SOURCE.json',
        'paper_online_ablation_discussion.tex',
        'paper_verified_runtime_references.bib')]
    inputs += [figures / name for name in (
        'main_table.tex', 'main_results_section.tex', 'online_vla_main.pdf',
        'online_vla_force_by_context.pdf', 'main_table.csv')]
    inputs += [ablation_table / name for name in (
        'online_dev_ablation_table.tex', 'online_dev_ablation_table.csv')]
    inputs += [main, ablations, behavior]
    source_hashes = {str(path): sha(path) for path in inputs}
    evidence = [here / name for name in (
        'FEASIBILITY_FEATURE_SCHEMA_AUDIT.json',
        'CURRENT_SERVER36_QUALIFICATION_REVIEW.json',
        'FINAL_VLA_QUALIFICATION_ADMISSION_V2.json',
        'FINAL_FRESH_ROOT_EXPOSURE_RESOLUTION.json',
        'FINAL_ROOT_SEED_SEMANTICS_AUDIT.json',
        'PROBE_SENSOR_DIAGNOSTIC_COMPLETE.json',
        'FINAL_MAIN_FIGURE_VISUAL_REVIEW.json',
        'FINAL_CONTROLLER_REFERENCE_AUDIT.json',
        'FINAL_EVIDENCE_IDENTITY_AUDIT.json',
        'PAPER_ABLATION_DISCUSSION_SOURCE.json',
        'final_regrasp_visual_review_v1/VISUAL_REVIEW.json',
        'final_regrasp_visual_review_v1/SOURCE_FRAMES.json',
        'ONLINE_ABLATION_PROTOCOL_FROZEN_V3.json',
        'ABLATION_INITIAL_OBSERVATION_PARITY_FAILURE.json',
        'ABLATION_REFERENCE_RECOVERY_FIRST_BRANCH_VERIFIED.json',
        'offline_feature_audit_v3/PHASE_SENSITIVITY_SUMMARY.json',
        'phase_free_exact_checkpoints/PHASE_FREE_EXACT_MANIFEST.json',
        'paper_phase_audit_appendix_v1/PHASE_SENSITIVITY_TABLE.csv',
        'paper_phase_audit_appendix_v1/PHASE_SENSITIVITY_APPENDIX.md')]
    final_manifest = Path(data['rows'][0]['job']).parents[1] / 'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json'
    if sha(final_manifest) != data['runtime_manifest_sha256']:
        raise RuntimeError('Final runtime manifest differs from result identity')
    evidence.append(final_manifest)
    ablation_protocol = Path(dev['source_protocol'])
    if sha(ablation_protocol) != dev['source_protocol_sha256']:
        raise RuntimeError('Ablation protocol changed since result aggregation')
    evidence.extend([
        ablation_protocol,
        ablation_protocol.parent / 'HISTORICAL_REFERENCE_RECOVERY.json',
        ablation_protocol.parent / 'CANDIDATE_RUNTIME_MANIFEST.json',
    ])
    source_hashes.update({str(path): sha(path) for path in evidence})
    counts = {method: sum(row['full_task_success'] for row in data['rows']
                         if row['method'] == method) for method in data['main']}
    for method, summary in data['main'].items():
        if abs(counts[method] / 48 - summary['FULL_TASK_SR']) > 1e-12:
            raise RuntimeError('Main success count mismatch')
    af = data['main']['ACTIVEFORCING']
    abstract = (
        'ActiveForcing augments a frozen pretrained vision--language--action policy '
        'with physical probing and posterior-aware grip-force selection. Starting '
        'from a common established grasp, the frozen Tabero $\\pi_0$ policy '
        'generates downstream arm actions online while a feedback controller '
        'realizes the selected squeeze force. Feasibility uses the current '
        'predicted action chunk and observable decision-state features, without '
        'scripted internal phase inputs. '
        f'Across 48 matched contexts from four fresh simulator reset groups, '
        f'ActiveForcing completes {counts["ACTIVEFORCING"]}/48 full tasks; '
        f'Fixed-3, Fixed-4, and Fixed-5 complete {counts["FIXED_3"]}/48, '
        f'{counts["FIXED_4"]}/48, and {counts["FIXED_5"]}/48, respectively. '
        f'The mean AF setpoint is {af["MEAN_SELECTED_SETPOINT"]:.3f} N. '
        'Online development ablations are reported separately. The evaluation '
        'addresses force adaptation after an established grasp without '
        'retraining the VLA; it does not include initial-grasp planning.')
    destination.mkdir(exist_ok=False)
    for path in inputs:
        shutil.copy2(path, destination / path.name)
    support = destination / 'SUPPORTING_EVIDENCE'
    support.mkdir()
    if len({path.name for path in evidence}) != len(evidence):
        raise RuntimeError('Supporting evidence filenames collide')
    for path in evidence:
        shutil.copy2(path, support / path.name)
    (support / 'README.md').write_text(
        'Evidence roles remain separate. Feature and phase audits use historical '
        '647-row controlled-motion data. Qualification and probe diagnostics use '
        'development contexts. The main result files alone contain the four fresh '
        'roots; online ablation files contain burned roots. No scripted success '
        'counts are admitted into online result tables. Recovery records '
        'retain the stopped initial-observation parity failure and the subsequent '
        'prospective recovery of matched historical observations. The unmatched '
        'branch is excluded; cached initial camera observations do not replace '
        'online policy inference or replay actions. The first-branch recovery '
        'receipt is historical and is not a certificate of study completion. '
        'Checkpoint manifests '
        'identify original files and hashes; large checkpoints and raw rollout '
        'arrays remain at their recorded source paths. This directory supports '
        'review but is not a self-contained simulator/model distribution.\n')
    tex = r'''\RequirePackage[T1]{fontenc}
\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb,graphicx}
\title{ActiveForcing: Physical Probing and Grip-Force Adaptation for an Online Frozen Vision--Language--Action Policy}
\author{\IEEEauthorblockN{Anonymous Authors}}
\begin{document}
\maketitle
\begin{abstract}
''' + abstract + r'''
\end{abstract}
\input{paper_online_introduction}
\input{paper_online_method_section}
\input{paper_online_experiment_protocol}
\section{Results}
\input{main_table}
\input{main_results_section}
\input{paper_online_main_discussion}
\begin{figure*}[t]
\centering\includegraphics[width=.95\textwidth]{online_vla_main.pdf}
\caption{Online frozen VLA evaluation. Bars show means over 48 contexts per method; black dots show four root-level means, not confidence intervals.}
\end{figure*}
\begin{figure*}[t]
\centering\includegraphics[width=.95\textwidth]{online_vla_force_by_context.pdf}
\caption{AF selected force by task and friction stratum. Each line follows one fresh reset root. Circles denote full-task success and crosses failure.}
\end{figure*}
\subsection{Online development ablations}
\input{online_dev_ablation_table}
Table~\ref{tab:online_dev_ablations} reports 24 matched development contexts
per method from two burned roots. These results are separate from the final
fresh-root evaluation. The prior variant retains the probe and is not a
NoProbe experiment. Historical supervision for Local-Lift remains explicitly
identified as controlled-motion auxiliary data.
\input{paper_online_ablation_discussion}
\input{paper_online_scope_and_limitations}
\section{Conclusion}
ActiveForcing combines physical probing and posterior-aware force selection
with an online frozen pretrained VLA. Audited execution preserves VLA arm
commands while the force controller realizes the selected squeeze setpoint.
The reported evaluation applies to downstream manipulation from a common
established grasp. Full-task completion, selected force, and measured squeeze
are reported separately so that force reduction is assessed together with
task outcomes.
\bibliographystyle{IEEEtran}
\bibliography{paper_verified_runtime_references}
\end{document}
'''
    (destination / 'manuscript.tex').write_text(tex)
    (destination / 'README.md').write_text(
        'Results-bearing review candidate, not a submission-readiness certificate.\n\n'
        'Compile with the recorded Tectonic toolchain: `tectonic --keep-logs manuscript.tex`. '
        'Inspect every rendered page, table, formula, reference and empirical claim before final handoff. '
        'Verify applicable venue formatting separately. Raw rollout evidence remains at the paths '
        'recorded by the result JSON files. Optional secondary baselines are not inferred.\n')
    (destination / 'ASSEMBLY_MANIFEST.json').write_text(json.dumps(dict(
        role='RESULTS_BEARING_PAPER_REVIEW_CANDIDATE', source_hashes=source_hashes,
        implementation_sha256=sha(__file__), main_branches=192, dev_ablation_branches=96,
        final_runtime_sha256=data['runtime_manifest_sha256'],
        compiled=False, final_visual_review_complete=False, master_goal_complete=False,
        artifacts={str(p.relative_to(destination)): sha(p)
                   for p in destination.rglob('*') if p.is_file()}), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('main', 'ablations', 'figures', 'ablation-table', 'behavior', 'destination'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assemble(args.main, args.ablations, args.figures, args.ablation_table, args.behavior, args.destination)
