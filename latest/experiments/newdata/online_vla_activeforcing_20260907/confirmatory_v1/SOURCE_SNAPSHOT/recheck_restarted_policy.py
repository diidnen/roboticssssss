"""Twelve bounded dev rechecks after same-policy server numerical restart.

Four previously declared force-boundary contexts, AF/Fixed3/Fixed5. No final
root, model tuning, outcome-based resampling, or cached downstream action.
"""
import argparse
import os
import shutil
from pathlib import Path
from common import HERE,read,write,sha

CONTEXTS=('t0_r5100_low','t1_r5100_mid','t5_r5100_low','t6_r5100_low')
METHODS=('FIXED_3','FIXED_5','ACTIVEFORCING')

def freeze(qualification,out,ready_path):
    ready=read(ready_path);prior=read(qualification/'CANDIDATE_RUNTIME_MANIFEST.json')
    for p,d in {**prior['source_hashes'],**ready['source_hashes']}.items():
        if sha(p)!=d:raise RuntimeError('Qualified or restarted server source changed')
    out.mkdir(exist_ok=False);snapshot=out/'SOURCE_SNAPSHOT';snapshot.mkdir()
    for p in (qualification/'SOURCE_SNAPSHOT').glob('*.py'):shutil.copy2(p,snapshot/p.name)
    for name in ('recheck_restarted_policy.py','audit_rollout.py','audit_geometric_label.py','drop_diagnostics.py','matched_seeds.py'):
        shutil.copy2(HERE/name,snapshot/name)
    write(out/'DEV_PLAN.json',read(qualification/'DEV_PLAN.json'))
    sources={p:d for p,d in prior['source_hashes'].items() if not Path(p).is_relative_to(qualification/'SOURCE_SNAPSHOT')}
    sources.update(ready['source_hashes']);sources[str(ready_path)]=sha(ready_path)
    sources.update({str(p):sha(p) for p in snapshot.glob('*.py')})
    prior.update(version='SAME_FROZEN_POLICY_RESTART_DEV_RECHECK_V1',source_hashes=sources,
        original_qualification=str(qualification),server_ready=str(ready_path),
        reason='Original server helper source recovered exactly; only equivalent ROOT constant differed. Native outputs vary numerically across server restart with identical observations/noise. Recheck predeclared4 boundary contexts before final freeze.',
        selected_contexts=list(CONTEXTS),selected_methods=list(METHODS),max_additional_branches=12,
        total_completed_qualification_plus_recheck_max=52,parameters_changed=False,
        use_original_qualified_worker_byte_exact=True,final_roots_used=False)
    write(out/'CANDIDATE_RUNTIME_MANIFEST.json',prior)
    write(out/'RECHECK_PROTOCOL.json',dict(contexts=CONTEXTS,methods=METHODS,branches=12,
        source_qualification=str(qualification),new_full_task_label_role='DEV_SERVER_RESTART_REQUALIFICATION',
        no_outcome_retry=True,no_model_tuning=True,final_fresh_roots_used=False))

def run(out):
    from launch import launch
    from audit_rollout import audit
    from audit_geometric_label import audit as geometry
    manifest=read(out/'CANDIDATE_RUNTIME_MANIFEST.json')
    if sha(__file__)!=manifest['source_hashes'].get(str(Path(__file__).resolve())):
        raise RuntimeError('Run only frozen recheck coordinator')
    write(out/'RECHECK_QUEUE_STARTED.json',dict(pid=os.getpid(),branches=12))
    plans=read(out/'DEV_PLAN.json')['contexts'];rows=[]
    for cid in CONTEXTS:
        index=next(i for i,p in enumerate(plans) if p['id']==cid)
        launch(out,index,'REFERENCE')
        original=Path(manifest['original_qualification'])/'references'/cid
        new=out/'references'/cid
        same={name:read(new/name)==read(original/name) for name in ('CONTACT_PATCH_READBACK.json','PREACTION_POSTERIOR.json')}
        same['RAW_PROBE.csv']=(new/'RAW_PROBE.csv').read_bytes()==(original/'RAW_PROBE.csv').read_bytes()
        write(new/'ORIGINAL_QUALIFICATION_PROBE_PARITY.json',dict(passed=all(same.values()),checks=same))
        if not all(same.values()):raise RuntimeError('Restart recheck probe/state no longer matches qualification')
        for method in METHODS:
            launch(out,index,method);job=out/'branches'/(cid+'__'+method)
            proof=audit(job);geom=geometry(job)
            if not proof['passed'] or not geom['passed']:raise RuntimeError('Invalid new-server online evidence')
            old=Path(manifest['original_qualification'])/'branches'/job.name
            old_result=read(old/'BRANCH_RESULT.json');new_result=read(job/'BRANCH_RESULT.json')
            row=dict(context=cid,method=method,job=str(job),provenance=proof,geometry=geom,
                old_job=str(old),same_full_task_outcome=old_result['outcome']['full_task_success_y']==new_result['outcome']['full_task_success_y'],
                old_selected_force=read(old/'PLANNER_DECISION.json')['executed_force_N'],
                new_selected_force=read(job/'PLANNER_DECISION.json')['executed_force_N'])
            write(job/'RECHECK_EVIDENCE.json',row);rows.append(row)
            print('RECHECK_BRANCH_DONE',cid,method,new_result['outcome']['full_task_success_y'],flush=True)
    write(out/'RECHECK_EXECUTION_COMPLETE.json',dict(branches=12,contexts=4,rows=rows,
        original_policy_and_force_method_unchanged=True,final_roots_used=False,
        review_still_required=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--freeze',action='store_true')
    p.add_argument('--qualification',type=Path);p.add_argument('--server-ready',type=Path);a=p.parse_args()
    if a.freeze:freeze(a.qualification,a.out,a.server_ready)
    else:run(a.out)
