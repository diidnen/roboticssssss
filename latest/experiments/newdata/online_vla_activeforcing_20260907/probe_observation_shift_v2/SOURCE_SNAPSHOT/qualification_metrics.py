"""Dev probability/force diagnostics with explicit context denominators."""
import argparse,json
from pathlib import Path
import numpy as np
from common import read

def metric(rows):
    if not rows:return {'n':0}
    y=np.asarray([r['y'] for r in rows],float);p=np.clip([r['p'] for r in rows],1e-7,1-1e-7)
    positives=p[y==1];negatives=p[y==0]
    auc=float(np.mean((positives[:,None]>negatives[None,:])+.5*(positives[:,None]==negatives[None,:]))) if len(positives) and len(negatives) else None
    tp=fp=0;ap=0.
    for value in sorted(set(p),reverse=True):
        labels=y[p==value];added=float(labels.sum());tp+=added;fp+=len(labels)-added
        if len(positives):ap+=added/len(positives)*tp/(tp+fp)
    ece=0.
    for i in range(10):
        mask=(p>=i/10)&(p<(i+1)/10 if i<9 else p<=1)
        if mask.any():ece+=mask.mean()*abs(p[mask].mean()-y[mask].mean())
    return dict(n=len(y),positives=int(y.sum()),NLL=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
        Brier=float(np.mean((y-p)**2)),AUROC=auc,AUPRC=ap if len(positives) else None,ECE_10_equal_width_bins=float(ece))

def review(out):
    out=Path(out);rows=[];curves={}
    for path in sorted((out/'branches').glob('*/BRANCH_RESULT.json')):
        result=read(path);job=path.parent
        if not (job/'PROCESS_EXIT.json').exists():continue
        process=read(job/'PROCESS_EXIT.json')
        if process['exit_code']!=0:raise RuntimeError('Nonzero worker exit cannot enter probability diagnostics')
        if not (job/'WORKER_COMPLETION.json').exists() or not read(job/'WORKER_COMPLETION.json')['logical_success']:continue
        d=read(job/'PLANNER_DECISION.json');force=d['executed_force_N'];grid=np.asarray(d['force_grid_N'])
        index=int(np.argmin(abs(grid-force)));assert abs(grid[index]-force)<1e-7
        plan=result['plan'];cid=plan['id'];curve=np.asarray(d['p_success'])
        rows.append(dict(context=cid,task=plan['task'],root=plan['root'],band=plan['band'],method=d['method'],force=force,
            p=float(curve[index]),y=result['outcome']['full_task_success_y']))
        if cid in curves and not np.allclose(curves[cid]['p'],curve,rtol=0,atol=1e-6):raise RuntimeError('Same decision X produced mismatched probability curves')
        curves[cid]={'p':curve,'p3':float(curve[0]),'p5':float(curve[-1]),'decreasing_adjacent_fraction':float(np.mean(np.diff(curve)<-1e-6))}
    main=[r for r in rows if r['method']!='FIXED_4'];af=[r for r in main if r['method']=='ACTIVEFORCING']
    complete_groups=[]
    for task,root in sorted({(r['task'],r['root']) for r in af}):
        group=[r for r in af if r['task']==task and r['root']==root]
        if len(group)==3:complete_groups.append({'task':task,'root':root,'forces_by_band':{r['band']:r['force'] for r in group},'force_varies':len({r['force'] for r in group})>1})
    return dict(role='DEVELOPMENT_QUALIFICATION_NOT_FINAL_SR',main_probability=metric(main),
        per_task={str(t):metric([r for r in main if r['task']==t]) for t in (0,1,5,6)},
        per_method={m:metric([r for r in main if r['method']==m]) for m in ('ACTIVEFORCING','FIXED_3','FIXED_5')},
        fixed4_boundary_probability=metric([r for r in rows if r['method']=='FIXED_4']),
        AF_contexts=len(af),AF_lower_bound_selection_rate=float(np.mean([r['force']==3 for r in af])) if af else None,
        AF_complete_task_root_groups=complete_groups,
        AF_context_sensitive_force_rate=float(np.mean([g['force_varies'] for g in complete_groups])) if complete_groups else None,
        force_probability_curves={cid:{k:v for k,v in d.items() if k!='p'} for cid,d in curves.items()},rows=rows,
        independent_units='One burned root; context/method rows are paired and correlated. These metrics are engineering qualification, not population estimates.',
        acceptance='Does not independently authorize final tests. Require full qualification, behavior, force boundary, input schema and label gates.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--save');a=p.parse_args();d=review(a.out)
    if a.save:Path(a.save).write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({k:d[k] for k in ('main_probability','per_task','AF_lower_bound_selection_rate','AF_context_sensitive_force_rate')},indent=2))
