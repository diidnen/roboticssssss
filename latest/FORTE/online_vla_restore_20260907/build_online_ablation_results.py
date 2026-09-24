"""Keep small online dev ablations separate from fresh-root main results."""
import argparse
import csv
from pathlib import Path
import numpy as np
from common import read,write,sha
from audit_rollout import audit
from audit_geometric_label import audit as geometry
from audit_online_ablation import review
from matched_seeds import matched_native_prefixes

def build(out,destination):
    done=read(out/'ONLINE_ABLATION_EXECUTION_COMPLETE.json')
    manifest=read(out/'CANDIDATE_RUNTIME_MANIFEST.json')
    protocol=read(out/'ONLINE_ABLATION_PROTOCOL.json')
    if (done['ablation_branches']!=96 or done['contexts']!=24
            or done['final_fresh_roots_used'] or done['scripted_results_used']):
        raise RuntimeError('Only complete online dev ablations may enter this table')
    for p,d in manifest['source_hashes'].items():
        if sha(p)!=d:raise RuntimeError('Ablation source changed')
    plans=read(out/'DEV_PLAN.json')['contexts']
    if len(plans)!=24 or sorted({p['root'] for p in plans})!=[5100,6200]:
        raise RuntimeError('Ablation denominator or development roots changed')
    controls={c['context']:c for c in done['controls']}
    completed={(r['context'],r['method']):r for r in done['rows']}
    expected={(p['id'],m) for p in plans for m in protocol['methods']}
    if set(completed)!=expected or len(done['rows'])!=96 or len(controls)!=24:
        raise RuntimeError('Missing or duplicated variant/context')
    rows=[];evidence={};server=read(manifest['POLICY_SERVER_VERSION']['readiness_evidence'])
    def record(job,plan,method,proof,geom):
        trace=read(job/'BRANCH_TRACE.json');d=read(job/'PLANNER_DECISION.json')
        active=[t for t in trace if not t['vla_release_intent']]
        force=[2*min(t['normal_force_N']) for t in active]
        contact=[2*min(t['normal_force_N']) for t in active if min(t['normal_force_N'])>=.15]
        if not force:raise RuntimeError('Undefined nonrelease force')
        if read(job/'VLA_SERVER_METADATA.json')!=server:raise RuntimeError('Mixed policy instances')
        result=proof['outcome']
        row=dict(context=plan['id'],root=plan['root'],task=plan['task'],friction=plan['band'],method=method,
            full_task_success=result['full_task_success_y'],lift_success=result['lift_success'],
            dropped=int(bool(result['dropped']) or geom['independent_measured_drop']),
            selected_force_N=d['executed_force_N'],measured_squeeze_N=float(np.mean(force)),
            contact_conditional_squeeze_N=float(np.mean(contact)) if contact else None,
            scientific_label=d.get('scientific_label','ActiveForcing'),
            job=str(job),result_sha256=sha(job/'BRANCH_RESULT.json'))
        rows.append(row)
        for name in ('BRANCH_RESULT.json','BRANCH_TRACE.json','PLANNER_DECISION.json','PROCESS_EXIT.json'):
            evidence[str(job/name)]=sha(job/name)
        seeds=[{k:r[k] for k in ('step','noise_seed','noise_sha256')} for r in
               (read(p) for p in sorted((job/'RPC').glob('*.json')))]
        return read(job/'INITIAL_ONLINE_CHUNK_IDENTITY.json'),seeds
    for plan in plans:
        cid=plan['id'];control=Path(controls[cid]['job'])
        p=audit(control);g=geometry(control)
        if not p['passed'] or not g['passed']:raise RuntimeError('Invalid AF control')
        identity,seed=record(control,plan,'ACTIVEFORCING',p,g);schedules=[seed]
        for method in protocol['methods']:
            job=Path(completed[(cid,method)]['job']);r=review(job)
            original=read(job/'ONLINE_ABLATION_ADMISSION.json')
            if r!=original:raise RuntimeError('Ablation review changed since admission')
            other,seeds=record(job,plan,method,r['provenance'],r['geometry'])
            schedules.append(seeds)
            if other!=identity or not matched_native_prefixes(schedules):
                raise RuntimeError('Not a matched current-X/seed comparison')
    def aggregate(group):
        c=[r['contact_conditional_squeeze_N'] for r in group if r['contact_conditional_squeeze_N'] is not None]
        return dict(contexts=len(group),full_successes=sum(r['full_task_success'] for r in group),
            FULL_TASK_SR=float(np.mean([r['full_task_success'] for r in group])),
            LIFT_SR=float(np.mean([r['lift_success'] for r in group])),
            DROP_RATE=float(np.mean([r['dropped'] for r in group])),
            MEAN_SELECTED_SETPOINT=float(np.mean([r['selected_force_N'] for r in group])),
            MEASURED_BILATERAL_SQUEEZE=float(np.mean([r['measured_squeeze_N'] for r in group])),
            CONTACT_CONDITIONAL_SQUEEZE=float(np.mean(c)) if c else None)
    methods=['ACTIVEFORCING',*protocol['methods']]
    report=dict(role='ONLINE_VLA_BURNED_ROOT_ABLATION_EVIDENCE',roots=[5100,6200],contexts_per_method=24,
        ablation_branches=96,AF_control_branches=24,new_AF_controls=done['new_AF_controls'],
        main={m:aggregate([r for r in rows if r['method']==m]) for m in methods},
        by_root={str(root):{m:aggregate([r for r in rows if r['method']==m and r['root']==root]) for m in methods} for root in (5100,6200)},
        rows=rows,source_hashes=evidence,source_protocol=str(out/'ONLINE_ABLATION_PROTOCOL.json'),
        source_protocol_sha256=sha(out/'ONLINE_ABLATION_PROTOCOL.json'),
        scripted_rollouts_included=False,final_fresh_roots_used=False,
        legal_no_probe_tested=False,prior_label='PRIOR / NO-POSTERIOR; physical probe retained',
        source_model_labels='Historical controlled-motion auxiliary supervision remains explicitly scripted, including the Local-Lift model.',
        interpretation='Two burned development roots, matched within context. These ablations are not additional independent final-test evidence or a NoProbe experiment.')
    destination.mkdir(exist_ok=False)
    write(destination/'ONLINE_VLA_ABLATION_RESULTS.json',report)
    with (destination/'ONLINE_VLA_ABLATION_ROWS.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('VERIFIED_ONLINE_ABLATIONS',96,'MATCHED_AF_CONTROLS',24)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args();build(a.out,a.destination)
