"""Exactly two prospective diagnostic repeats; neither becomes formal data."""
import os
import subprocess
from rootlocal_collection_contract import HERE, BASE, REPO, read, write, sha, now, verify_runtime


def main():
    out = HERE/'mu060_contact_diagnostic_repeats_v1'
    out.mkdir(exist_ok=False)
    context = read(HERE/'original_rootlocal_dataset_v1/groups/train_mu0.600_root200002/attempt_001/CONTEXT.json')
    sources = {str(p): sha(p) for p in [HERE/'trace_original_query_contact.py',
        HERE/'run_mu060_contact_diagnostics.py', HERE/'MU060_CONTACT_DIAGNOSTIC_PLAN.md']}
    write(out/'PROTOCOL.json', {'context': context, 'queries': 2, 'formal_admission': False,
        'source_hashes': sources, 'started_utc': now(), 'retry_until_success': False})
    for index in range(2):
        verify_runtime(context['runtime_manifest_path'], context['runtime_manifest_sha256'])
        if any(sha(p)!=s for p,s in sources.items()): raise ValueError('Diagnostic source drift')
        case = out/f'repeat_{index+1:02d}'; case.mkdir()
        env = os.environ.copy()
        env.pop('AF_COLLECTION_CONTEXT', None); env.pop('AF_INFERENCE_CONTEXT', None)
        env.update(AF_P4_PHYSICAL_SURFACE_CAMERA='1', AF_ORIGINAL_SQUEEZE_INNER='1',
            PYTHONUTF8='1', PATH=str(BASE/'runtime_bin')+os.pathsep+env.get('PATH','/usr/bin:/bin'))
        cmd = [str(BASE/'venv_robotwin/bin/python3'), str(HERE/'trace_original_query_contact.py'),
               '--repo', str(REPO), '--out', str(case/'query'), '--native-ft',
               '--friction', '.6', '--policy-seed', '30200002']
        write(case/'PROCESS.json', {'command':cmd, 'cwd':str(REPO), 'started_utc':now()})
        with (case/'worker.log').open('x',encoding='utf-8') as log:
            process = subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(case/'PID.json', {'pid':process.pid}); code=process.wait()
        write(case/'EXIT.json', {'exit_code':code, 'finished_utc':now()})
        if code: raise RuntimeError('Diagnostic process failed; preserve and inspect, no automatic retry')
        verify_runtime(context['runtime_manifest_path'], context['runtime_manifest_sha256'])
        summary=read(case/'query/original_probe_summary.json')
        print({'repeat':index+1,'probe_failure':summary['probe_failure'],'stop_trigger':summary['stop_trigger']},flush=True)
    write(out/'COMPLETE.json', {'completed':True,'diagnostic_queries':2,'formal_labels_added':0,'finished_utc':now()})


if __name__ == '__main__': main()
