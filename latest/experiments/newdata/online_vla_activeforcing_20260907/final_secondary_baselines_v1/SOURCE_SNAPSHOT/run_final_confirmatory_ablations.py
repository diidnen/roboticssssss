"""Single-worker24-context ONLINE ablations, after the main test completes."""
import argparse
import os
from pathlib import Path
from common import read,write,sha
from launch import launch
from audit_rollout import audit
from audit_geometric_label import audit as geometry
from audit_online_ablation import review
from matched_seeds import matched_native_prefixes

def run(out):
    m=read(out/'CANDIDATE_RUNTIME_MANIFEST.json');protocol=read(out/'ONLINE_ABLATION_PROTOCOL.json')
    if sha(__file__)!=m['source_hashes'].get(str(Path(__file__).resolve())):raise RuntimeError('Run frozen coordinator only')
    final=Path(m['source_final_runtime'])
    if read(final/'FINAL_MAIN_EXECUTION_COMPLETE.json')['branches']!=192:
        raise RuntimeError('Main evaluation must finish before ablations')
    for p,d in m['source_hashes'].items():
        if sha(p)!=d:raise RuntimeError('Frozen source changed')
    plans=read(out/'DEV_PLAN.json')['contexts']
    if len(plans)!=24 or sorted({p['root'] for p in plans})!=[170048,170049]:
        raise RuntimeError('Invalid confirmatory ablation roots/coverage')
    write(out/'ABLATION_QUEUE_STARTED.json',dict(pid=os.getpid(),contexts=24,ablation_branches=120, new_AF_controls_max=24))
    rows=[];controls=[]
    for index,plan in enumerate(plans):
        launch(out,index,'REFERENCE');reference=out/'references'/plan['id']
        reused=None
        launch(out,index,'ACTIVEFORCING');control=out/'branches'/(plan['id']+'__ACTIVEFORCING')
        proof=audit(control);geom=geometry(control)
        if not proof['passed'] or not geom['passed']:raise RuntimeError('Invalid online AF comparison')
        expected_server=read(m['POLICY_SERVER_VERSION']['readiness_evidence'])
        if read(control/'VLA_SERVER_METADATA.json')!=expected_server:raise RuntimeError('Control used another server instance')
        identity=read(control/'INITIAL_ONLINE_CHUNK_IDENTITY.json')
        def seeds(job):
            return [{k:r[k] for k in ('step','noise_seed','noise_sha256')} for r in
                    (read(p) for p in sorted((job/'RPC').glob('*.json')))]
        schedules=[seeds(control)]
        c=dict(context=plan['id'],job=str(control),reused=bool(reused),provenance=proof,geometry=geom,
            source_result_sha256=sha(control/'BRANCH_RESULT.json'))
        write(reference/'AF_COMPARISON_CONTROL.json',c);controls.append(c)
        for method in protocol['methods']:
            launch(out,index,method);job=out/'branches'/(plan['id']+'__'+method)
            p=review(job)
            if read(job/'VLA_SERVER_METADATA.json')!=expected_server:raise RuntimeError('Ablation server instance changed')
            if read(job/'INITIAL_ONLINE_CHUNK_IDENTITY.json')!=identity:raise RuntimeError('Ablation/control initialX differs')
            schedules.append(seeds(job))
            if not matched_native_prefixes(schedules):raise RuntimeError('Ablation/control policy seeds differ')
            write(job/'ONLINE_ABLATION_ADMISSION.json',p);rows.append(dict(context=plan['id'],job=str(job),method=method,review=p))
        print('ONLINE_ABLATION_CONTEXT_DONE',plan['id'],len(rows),flush=True)
    if len(rows)!=96 or len(controls)!=24:raise RuntimeError('Incomplete ablation table')
    write(out/'ONLINE_ABLATION_EXECUTION_COMPLETE.json',dict(contexts=24,ablation_branches=96,controls=controls,rows=rows,
        new_AF_controls=sum(not c['reused'] for c in controls),final_fresh_roots_used=False,scripted_results_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
