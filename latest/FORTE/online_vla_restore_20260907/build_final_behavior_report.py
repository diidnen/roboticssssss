"""Aggregate frozen observable-event diagnostics after complete final execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def build(final, results, destination):
    manifest_path = final / 'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json'
    manifest = read(manifest_path)
    main = read(results)
    if (main.get('role') != 'FINAL_ONLINE_FROZEN_VLA_MAIN_EVIDENCE'
            or main.get('branches') != 192 or len(main.get('rows', [])) != 192
            or main.get('all_branch_sources') != 'ONLINE_VLA'
            or main.get('scripted_results_included') is not False
            or main['runtime_manifest_sha256'] != sha(manifest_path)):
        raise RuntimeError('Complete verified final online evidence required')
    if read(final / 'FINAL_MAIN_EXECUTION_COMPLETE.json')['branches'] != 192:
        raise RuntimeError('Final execution incomplete')
    for path, digest in manifest['source_hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen source changed: ' + path)
    snapshot = final / 'SOURCE_SNAPSHOT'
    # Use exactly the frozen diagnostic implementation, not the mutable working copy.
    sys.path.insert(0, str(snapshot))
    from behavior_audit import audit
    from drop_diagnostics import diagnose
    import behavior_audit
    if Path(behavior_audit.__file__).resolve() != (snapshot / 'behavior_audit.py').resolve():
        raise RuntimeError('Wrong diagnostic module loaded')
    rows = []
    for source_row in main['rows']:
        job = Path(source_row['job'])
        if sha(job / 'BRANCH_RESULT.json') != source_row['result_sha256']:
            raise RuntimeError('Outcome evidence changed')
        admission_path = job / 'FINAL_EVIDENCE_ADMISSION.json'
        if sha(admission_path) != main['source_hashes'][str(admission_path)]:
            raise RuntimeError('Admission evidence changed')
        report = audit(job)
        drop = diagnose(job)
        if not report['VLA_FAILURE_STAGE_LABEL_CAUSAL'] or report['scripted_downstream_stage_used']:
            raise RuntimeError('Noncausal or scripted event analysis')
        trace = read(job / 'BRANCH_TRACE.json')
        rows.append(dict(context=source_row['context'], root=source_row['root'],
            method=source_row['method'], full_task_success=source_row['full_task_success'],
            outcome_failure_reasons=report['outcome']['failure_reasons'],
            observable_events=report['events'],
            last_observable_state=report['events'][-1]['state'] if report['events'] else None,
            early_regrasp_candidate=report['early_regrasp_candidate_present'],
            regrasp_candidate=report['regrasp_candidate_present'],
            regrasp_candidate_steps=report['repeated_grasp_candidate_steps'],
            action_anomalies=report['action_anomalies'], transitions=max(0, len(trace)-1),
            raw_open_away_target_steps=report['raw_open_intent_away_target_steps'],
            measured_drop=drop['measured_drop_outside_target'],
            causal_drop_proof=drop['physical_proof'],
            job=str(job), trace_sha256=sha(job / 'BRANCH_TRACE.json')))
    methods = sorted({row['method'] for row in rows})
    def aggregate(group):
        transitions = sum(row['transitions'] for row in group)
        return dict(branches=len(group),
            VLA_REGRASP_CANDIDATE_RATE=sum(row['regrasp_candidate'] for row in group)/len(group),
            VLA_EARLY_REGRASP_CANDIDATE_RATE=sum(row['early_regrasp_candidate'] for row in group)/len(group),
            VLA_ACTION_ANOMALY_RATE=sum(len(row['action_anomalies']) for row in group)/max(1, transitions),
            action_transition_denominator=transitions,
            failure_reason_counts=dict(Counter(reason for row in group for reason in row['outcome_failure_reasons'])),
            terminal_event_counts=dict(Counter(row['last_observable_state'] for row in group)))
    report = dict(role='FINAL_ONLINE_VLA_OBSERVABLE_EVENT_ANALYSIS', branches=192,
        VLA_FAILURE_STAGE_LABEL_CAUSAL=True, scripted_downstream_stage_used=False,
        main_results=str(results), main_results_sha256=sha(results),
        runtime_manifest_sha256=sha(manifest_path),
        diagnostic_source_sha256=sha(snapshot / 'behavior_audit.py'),
        drop_diagnostic_source_sha256=sha(snapshot / 'drop_diagnostics.py'),
        implementation_sha256=sha(__file__), rows=rows,
        by_method={method:aggregate([row for row in rows if row['method']==method]) for method in methods},
        scope='Observable event histories and frozen geometric failure predicates. Terminal states are not causal attributions. Multiple failure reasons may coexist.',
        regrasp_interpretation='Heuristic candidates require visual review; candidate rate is not a confirmed regrasp-attempt rate.',
        force_failure_causality_proven=False,
        probe_shift_scope='Development pre/post sensor diagnostics and qualification are separate. This report does not infer a learned OOD score or full sensor-distribution equivalence.')
    destination.mkdir(exist_ok=False)
    (destination / 'FINAL_ONLINE_VLA_BEHAVIOR_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('final', 'results', 'destination'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.final, args.results, args.destination)
