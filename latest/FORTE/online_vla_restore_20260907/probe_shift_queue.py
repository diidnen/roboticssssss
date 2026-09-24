"""Bounded reference-only sensor diagnostic after existing qualification exits.

Never starts a second simulator, fresh roots, downstream actions, or training.
"""
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from common import read, write, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--coordinator-pid', type=int, required=True)
    p.add_argument('--reuse-first-reference',type=Path)
    a = p.parse_args()
    q, out = a.qualification, a.diagnostic
    write(out/'DIAGNOSTIC_QUEUE_STARTED.json', {
        'pid': os.getpid(), 'source_sha256': sha(__file__),
        'wait_for_qualification': str(q), 'coordinator_pid': a.coordinator_pid,
        'max_reference_runs': 11 if a.reuse_first_reference else 12,
        'reused_first_reference':str(a.reuse_first_reference) if a.reuse_first_reference else None,
        'downstream_branches': 0, 'fresh_roots': [],
    })
    proc = Path('/proc')/str(a.coordinator_pid)
    while proc.exists():
        command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if str(q/'SOURCE_SNAPSHOT'/'qualify.py') not in command:
            raise RuntimeError('Coordinator PID identity changed; inspect before diagnostic')
        if (out/'STOP_QUEUE.json').exists():
            print('STOPPED_BEFORE_DIAGNOSTIC', flush=True)
            return
        time.sleep(15)
    report = read(q/'QUALIFICATION_REPORT.json')
    if len(report['rows']) != 40 or not report['gates']['primary_complete']:
        raise RuntimeError('Qualification incomplete; diagnostic cannot silently bypass failure')
    completed = list((q/'branches').glob('*/PROCESS_EXIT.json'))
    if len(completed) != 40 or any(read(x)['exit_code'] != 0 for x in completed):
        raise RuntimeError('Qualification process completion invalid')
    plans = read(out/'DEV_PLAN.json')['contexts']
    if len(plans) != 12 or any(x['root'] != 5100 for x in plans):
        raise RuntimeError('Diagnostic scope is twelve burned-root reference probes')
    rows = []
    for index, plan in enumerate(plans):
        if (out/'STOP_QUEUE.json').exists():
            print('STOPPED_BETWEEN_REFERENCE_PROBES', flush=True)
            return
        command = ['/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python', '-u',
                   str(out/'SOURCE_SNAPSHOT'/'launch.py'), '--out', str(out),
                   '--context', str(index), '--method', 'REFERENCE', '--probe-diagnostic']
        if index==0 and a.reuse_first_reference:
            job=a.reuse_first_reference
            if job.name!=plan['id']:
                raise RuntimeError('Wrong reference identity for metadata-only reuse')
            origin=read(job/'PROCESS.json')['command']
            if origin[origin.index('--method')+1]!='REFERENCE' or origin[origin.index('--context')+1]!='0':
                raise RuntimeError('Reused evidence is not the original first reference')
        else:
            subprocess.run(command, check=True)
            job = out/'references'/plan['id']
        reference = q/'references'/plan['id']
        if read(job/'PROCESS_EXIT.json')['exit_code'] != 0:
            raise RuntimeError('Reference diagnostic nonzero exit')
        if not read(job/'WORKER_COMPLETION.json')['logical_success']:
            raise RuntimeError('Reference diagnostic logical failure')
        checks = {}
        for name in ('CONTACT_PATCH_READBACK.json', 'PREACTION_POSTERIOR.json'):
            checks[name] = read(job/name) == read(reference/name)
        checks['RAW_PROBE.csv'] = list(csv.DictReader((job/'RAW_PROBE.csv').open())) == list(csv.DictReader((reference/'RAW_PROBE.csv').open()))
        if not all(checks.values()):
            write(job/'DIAGNOSTIC_REFERENCE_PARITY.json', {'passed': False, 'checks': checks})
            raise RuntimeError('Readout diagnostic changed common physical probe evidence')
        parity={'passed':True,'checks':checks}
        if (job/'DIAGNOSTIC_REFERENCE_PARITY.json').exists():
            if read(job/'DIAGNOSTIC_REFERENCE_PARITY.json')!=parity:
                raise RuntimeError('Existing diagnostic parity disagrees')
        else:write(job/'DIAGNOSTIC_REFERENCE_PARITY.json',parity)
        shift = read(job/'PROBE_OBSERVATION_SHIFT.json')
        rows.append({'context': plan['id'], 'reference_parity': True, 'shift': shift,
                     'original_reference_job':str(job),'reference_reused_without_new_physics':bool(index==0 and a.reuse_first_reference),
                     'artifact_sha256': sha(job/'PROBE_OBSERVATION_SHIFT.json')})
        print('REFERENCE_SENSOR_DIAGNOSTIC_DONE', plan['id'], flush=True)
    write(out/'PROBE_OBSERVATION_SHIFT_REPORT.json', {
        'role': 'DEV_ONLY_PAIRED_SENSOR_OBSERVATION', 'rows': rows,
        'downstream_rollouts': 0, 'final_roots_used': False,
        'interpretation': 'Observed perturbations; behavioral acceptability requires actual online qualification, not image-difference thresholds alone.',
    })


if __name__ == '__main__':
    main()
