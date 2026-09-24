"""192-branch main evaluation, inaccessible without frozen positive gates.

Uses the same qualified worker and online inference path. This driver changes
only root/method scheduling and evidence admission, never method parameters.
"""
import argparse
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from common import ROOT, TABERO, read, write, sha
from audit_rollout import audit
from audit_geometric_label import audit as audit_geometry

METHODS = ('FIXED_3', 'FIXED_4', 'FIXED_5', 'ACTIVEFORCING')
REQUIRED_GATES = (
    'ONLINE_VLA_VERIFIED', 'VLA_ACTION_ARBITRATION_VALID',
    'VLA_POSTPROBE_BEHAVIOR_VALID', 'VLA_COMPATIBLE_FEASIBILITY_INPUTS',
    'VLA_REAL_FORCE_BOUNDARY_EXISTS', 'FULL_TASK_LABEL_VALID',
    'VLA_SEED_MATCHING_VALID', 'FRESH_ROOT_NONEXPOSURE_VERIFIED',
)


def validate(out):
    manifest_path = out/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json'
    m = read(manifest_path)
    if m.get('FINAL_RUNTIME_USES_ONLINE_VLA') is not True or m.get('final_runtime_frozen') is not True:
        raise RuntimeError('No final online VLA runtime freeze')
    if any(m.get(key) is not True for key in REQUIRED_GATES):
        raise RuntimeError('Final VLA qualification gate incomplete')
    if m.get('EXISTING_FEASIBILITY_VLA_TRANSFER') not in ('PASS', 'PARTIAL'):
        raise RuntimeError('Feasibility transfer has not been admitted')
    if m.get('DOWNSTREAM_ACTION_SOURCE') != 'ONLINE_VLA':
        raise RuntimeError('SCRIPTED/VLA_REPLAY forbidden in final table')
    # The historical worker filename is retained unchanged. Its manifest input
    # is an exact compatibility copy of this final freeze, not a second runtime.
    if read(out/'CANDIDATE_RUNTIME_MANIFEST.json') != m:
        raise RuntimeError('Worker manifest diverges from unique final freeze')
    if str(Path(__file__).resolve()) not in m['source_hashes']:
        raise RuntimeError('Final coordinator is not frozen')
    for path, expected in m['source_hashes'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen source/artifact changed: '+path)
    plans = read(out/'DEV_PLAN.json')['contexts']
    official = read(out/'FINAL_FRESH_ROOT_PLAN.json')
    if plans != official['contexts'] or len(plans) != 48:
        raise RuntimeError('Final worker context plan differs')
    roots = sorted({p['root'] for p in plans})
    if len(roots) != 4 or roots != sorted(official['roots']):
        raise RuntimeError('Exactly four frozen fresh root groups required')
    expected = {(r, t, b) for r in roots for t in (0, 1, 5, 6) for b in ('LOW', 'MID', 'HIGH')}
    actual = {(p['root'], p['task'], p['band']) for p in plans}
    if actual != expected or len({p['id'] for p in plans}) != 48:
        raise RuntimeError('Final context coverage or identity is invalid')
    objects = {0:'alphabet_soup_1',1:'cream_cheese_1',5:'tomato_sauce_1',6:'butter_1'}
    for plan in plans:
        if plan['id'] != f"t{plan['task']}_r{plan['root']}_{plan['band'].lower()}":
            raise RuntimeError('Noncanonical final context identifier')
        if plan.get('object') != objects[plan['task']] or plan.get('target') != 'basket_1':
            raise RuntimeError('Final object/target contract changed')
        if not isinstance(plan.get('mu'), (float,int)) or not math.isfinite(plan['mu']) or plan['mu'] <= 0:
            raise RuntimeError('Missing/invalid simulator friction assignment')
    if official['root_selection_used_outcomes'] is not False:
        raise RuntimeError('Outcome-dependent root selection forbidden')
    return m, plans


def launch(out, index, method):
    manifest, plans = validate(out)
    if method not in (*METHODS, 'REFERENCE'):
        raise RuntimeError('Unfrozen primary method')
    # Cross-process bfloat16/XLA numerical variation was measured in dev.
    # Therefore a replacement server, even with the same checkpoint, requires
    # a separate reviewed continuation; never switch instances silently.
    sys.path.insert(0,str(TABERO/'benchmarks/openpi/openpi-client/src'))
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    policy=WebsocketClientPolicy('127.0.0.1',18885)
    actual_server=policy.get_server_metadata();policy._ws.close()
    expected_server=read(manifest['POLICY_SERVER_VERSION']['readiness_evidence'])
    if actual_server!=expected_server:
        raise RuntimeError('Policy server instance changed after final freeze')
    plan = plans[index]
    job = out/'references'/plan['id'] if method == 'REFERENCE' else out/'branches'/(plan['id']+'__'+method)
    if job.exists():
        # No outcome retries. A complete valid job can be resumed only as
        # already-existing evidence, never executed again.
        if not (job/'PROCESS_EXIT.json').exists() or read(job/'PROCESS_EXIT.json')['exit_code'] != 0:
            raise RuntimeError('Incomplete/failed job retained for explicit infrastructure review: '+str(job))
        if not read(job/'WORKER_COMPLETION.json')['logical_success']:
            raise RuntimeError('Logical failure retained')
        return job
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    free, util = [int(x.strip()) for x in gpu.split(',')]
    if free < 10000 or util > 70:
        raise RuntimeError('Single-worker GPU resource gate: '+gpu)
    job.mkdir(parents=True, exist_ok=False)
    python = Path('/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python')
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE='1', OMNI_KIT_ACCEPT_EULA='YES', ACCEPT_EULA='Y',
        PYTHONPATH=os.pathsep.join([str(python.parent.parent/'lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64'), str(out/'SOURCE_SNAPSHOT'), str(ROOT), str(TABERO), str(TABERO/'benchmarks/openpi/openpi-client/src')]),
        TABERO_ROOT=str(TABERO), HDF5_TRAJ_SOURCE_DIR=str(TABERO/'benchmarks/datasets/libero/assembled_hdf5'),
        LIBERO_CONFIG_DIR=str(TABERO/'benchmarks/datasets/libero/config'), LIBERO_ASSETS_DATA_DIR=str(TABERO/'benchmarks/datasets/libero/USD'))
    command = [str(python), '-u', str(out/'SOURCE_SNAPSHOT'/'worker.py'), '--out', str(out), '--job', str(job), '--context', str(index), '--method', method]
    with (job/'WORKER.log').open('x') as f:
        process = subprocess.Popen(command, cwd=TABERO, env=env, stdout=f, stderr=subprocess.STDOUT)
        write(job/'PROCESS.json', {'pid': process.pid, 'command': command,
            'started_utc': datetime.now(timezone.utc).isoformat(), 'gpu_before': gpu,
            'final_manifest_sha256': sha(out/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json')})
        print('START', job, process.pid, flush=True)
        rc = process.wait()
    completion = read(job/'WORKER_COMPLETION.json') if (job/'WORKER_COMPLETION.json').exists() else None
    write(job/'PROCESS_EXIT.json', {'exit_code': rc, 'completion': completion})
    if rc != 0 or not completion or not completion['logical_success']:
        raise RuntimeError('Final queue stopped on infrastructure/evidence failure; no automatic outcome retry')
    return job


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out
    manifest, plans = validate(out)
    write(out/'FINAL_QUEUE_STARTED.json', {'pid': os.getpid(), 'planned_branches': 192,
        'source_sha256': sha(__file__), 'methods': METHODS, 'no_outcome_retry': True})
    rows = []
    for index, plan in enumerate(plans):
        if (out/'STOP_QUEUE.json').exists():
            return
        launch(out, index, 'REFERENCE')
        for method in METHODS:
            if (out/'STOP_QUEUE.json').exists():
                return
            job = launch(out, index, method)
            proof, geometric = audit(job), audit_geometry(job)
            if not proof['passed'] or not geometric['passed']:
                write(job/'FINAL_EVIDENCE_QUARANTINE.json', {'provenance': proof, 'geometry': geometric})
                raise RuntimeError('Invalid branch quarantined; final queue stopped')
            write(job/'FINAL_EVIDENCE_ADMISSION.json', {'provenance': proof, 'geometry': geometric,
                'final_manifest_sha256': sha(out/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json')})
            rows.append({'context': plan['id'], 'method': method, 'job': str(job),
                'result_sha256': sha(job/'BRANCH_RESULT.json'), 'success': proof['outcome']['full_task_success_y']})
        print('FINAL_CONTEXT_DONE', plan['id'], flush=True)
    if len(rows) != 192:
        raise RuntimeError('Incomplete final table')
    write(out/'FINAL_MAIN_EXECUTION_COMPLETE.json', {'rows': rows, 'branches': 192, 'contexts': 48,
        'roots': sorted({p['root'] for p in plans}), 'source': 'ONLINE_VLA',
        'tables_and_paper_validation_still_required': True})


if __name__ == '__main__':
    main()
