"""Bounded 36-branch dev transfer qualification plus four force-boundary rows.

Never launches final roots, training, or further collection. A valid failure
is retained as a negative; evidence/provenance failure stops the entire queue.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from common import read,write,sha
from launch import launch
from audit_rollout import audit

def result_path(out,plan,method):return out/'branches'/(plan['id']+'__'+method)

def summary(out):
    rows=[]
    for job in sorted((out/'branches').glob('*')):
        if not (job/'BRANCH_RESULT.json').exists():continue
        if not (job/'PROCESS_EXIT.json').exists():continue # Still-running branches are not completed evidence.
        a=audit(job)
        if not a['passed']:raise RuntimeError('Invalid evidence '+str(job)+str(a))
        r=read(job/'BRANCH_RESULT.json');d=read(job/'PLANNER_DECISION.json');force=d['executed_force_N']
        i=int(np.argmin(np.abs(np.asarray(d['force_grid_N'])-force)))
        rows.append({'context':r['plan']['id'],'task':r['plan']['task'],'root':r['plan']['root'],
            'band':r['plan']['band'],'method':d['method'],'force':force,'p':d['p_success'][i],
            'y':r['outcome']['full_task_success_y'],'lift':r['outcome']['lift_success'],
            'measured_force':r['outcome']['mean_measured_bilateral_squeeze'],
            'result_path':str(job/'BRANCH_RESULT.json'),'result_sha256':sha(job/'BRANCH_RESULT.json')})
    primary=[r for r in rows if r['method']!='FIXED_4']
    y=np.array([r['y'] for r in primary]);p=np.array([r['p'] for r in primary])
    pos=p[y==1];neg=p[y==0]
    auc=float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:]))) if len(pos) and len(neg) else None
    ece=0.
    for lo,hi in zip(np.arange(0,1,.2),np.arange(.2,1.01,.2)):
        mask=(p>=lo)&(p<hi if hi<.999 else p<=hi)
        if mask.any():ece+=float(mask.mean()*abs(p[mask].mean()-y[mask].mean()))
    method={m:{'n':sum(r['method']==m for r in rows),'successes':sum(r['y'] for r in rows if r['method']==m)} for m in ('ACTIVEFORCING','FIXED_3','FIXED_4','FIXED_5')}
    boundaries={}
    for task in (0,1,5,6):
        pairs=[]
        for cid in sorted({r['context'] for r in rows if r['task']==task}):
            low=[r for r in rows if r['context']==cid and r['force']==3 and r['y']==0]
            high=[r for r in rows if r['context']==cid and r['force']>3 and r['y']==1]
            if low and high:pairs.append({'context':cid,'low':low,'higher':high})
        boundaries[str(task)]={'exists':bool(pairs),'pairs':pairs,'causality':'paired policy/controller evidence; inspect motion/contact traces before causal force claim'}
    af=[r for r in primary if r['method']=='ACTIVEFORCING']
    brier=float(np.mean((p-y)**2)) if len(y) else None
    complete=len(primary)==36 and all(method[m]['n']==12 for m in ('ACTIVEFORCING','FIXED_3','FIXED_5'))
    gate={'primary_complete':complete,'both_outcomes_observed':bool(len(pos) and len(neg)),
        'auroc_at_least_0_70':auc is not None and auc>=.70,'brier_at_most_0_22':brier is not None and brier<=.22,
        'ece5_at_most_0_20':ece<=.20,
        'af_within_two_successes_of_fixed5':method['ACTIVEFORCING']['successes']>=method['FIXED_5']['successes']-2,
        'af_not_constant_force':len({r['force'] for r in af})>1,
        'force_boundary_all_four_tasks':all(v['exists'] for v in boundaries.values())}
    transfer='PASS' if all(gate.values()) else 'PENDING' if not complete else 'PARTIAL' if auc is not None and auc>=.7 else 'FAIL'
    return {'role':'DEV_ONLY_NOT_FINAL_SR','rows':rows,'methods':method,'boundaries':boundaries,
        'brier':brier,'auroc':auc,'ece_5_equal_width_bins':ece,'gates':gate,
        'existing_feasibility_transfer_valid':transfer,
        'one_root_limitation':'engineering transfer screen; not independent-root generalization evidence',
        'thresholds':'engineering thresholds declared during dev integration before bulk qualification; not a preregistered confirmatory test; frozen existing feasibility untouched'}

def main():
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--summarize',action='store_true')
    args=a.parse_args();out=Path(args.out)
    if args.summarize:print(json.dumps(summary(out),indent=2));return
    if (out/'QUALIFICATION_STARTED.json').exists():raise RuntimeError('Exclusive queue claim exists')
    write(out/'QUALIFICATION_STARTED.json',{'coordinator_sha256':sha(__file__),'max_main':36,'max_boundary_extra':4,
        'dev_roots':[5100],'final_roots_used':False,'retraining_allowed_by_this_program':False})
    plans=read(out/'DEV_PLAN.json')['contexts']
    for index,plan in enumerate(plans):
        if (out/'STOP_QUEUE.json').exists():
            print('QUEUE_STOPPED_BETWEEN_CONTEXTS',flush=True);return
        launch(out,index,'REFERENCE')
        for method in ('ACTIVEFORCING','FIXED_3','FIXED_5'):
            if (out/'STOP_QUEUE.json').exists():
                print('QUEUE_STOPPED_BETWEEN_BRANCHES',flush=True);return
            launch(out,index,method)
            proof=audit(result_path(out,plan,method))
            if not proof['passed']:raise RuntimeError('Provenance fail: '+str(proof))
            first=read(result_path(out,plan,method)/'BRANCH_TRACE.json')[0]
            if first['arbitration_version']!='ONLINE_VLA_ARM_AF_SQUEEZE_V2_CANONICAL_OPEN':raise RuntimeError('Unqualified arbitration')
            if not read(result_path(out,plan,method)/'COMMON_OBSERVATION_RESTORE_GATE.json')['passed']:
                raise RuntimeError('Initial VLA observation is not a matched common state')
        if index==0:
            outcomes=[read(result_path(out,plan,m)/'BRANCH_RESULT.json')['outcome']['full_task_success_y'] for m in ('ACTIVEFORCING','FIXED_3','FIXED_5')]
            write(out/'FIRST_CONTEXT_GATE.json',{'outcomes':outcomes,'passed':any(outcomes)})
            if not any(outcomes):raise RuntimeError('First context has no successful VLA rollout: inspect policy/label before broader physics')
        print('CONTEXT_DONE',plan['id'],flush=True)
    for index in (0,4,6,9):
        launch(out,index,'FIXED_4')
        proof=audit(result_path(out,plans[index],'FIXED_4'))
        if not proof['passed']:raise RuntimeError('Boundary provenance failure')
    report=summary(out);write(out/'QUALIFICATION_REPORT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','boundaries')},indent=2),flush=True)

if __name__=='__main__':main()
