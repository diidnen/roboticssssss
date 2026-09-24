#!/usr/bin/env python3
"""Strict nested-root OOF Direct-Only residual on OOF ProbeScalar inputs.

This is a matched follow-up to activeforcing_probe_conditioned_wm_20260901_064627.
It reuses that run's held-root ProbeScalar Direct scores and trains only the
previously frozen Direct-Only utility residual. It never discovers or reads an
untouched TEST namespace and does not train a World Model.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_probe_conditioned_wm as b

ROOT = Path('/home/exouser/FORTE')
BASE = ROOT / 'activeforcing_probe_conditioned_wm_20260901_064627'
TASKS = b.TASKS
FOLDS = b.FOLDS
SEEDS = b.SEEDS
FMAX = b.FMAX


class DirectOnlyResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def task_onehot(tasks):
    ans = np.zeros((len(tasks), len(TASKS)), np.float32)
    for j, task in enumerate(TASKS):
        ans[:, j] = (tasks == task).astype(np.float32)
    return ans


def features(utility, force, fmax, tasks):
    return np.column_stack([utility, force / fmax, task_onehot(tasks)]).astype(np.float32)


def fit_residual(x, target, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mean = x.mean(0).astype(np.float32)
    std = x.std(0).astype(np.float32); std[std < 1e-6] = 1
    xn = ((x - mean) / std).astype(np.float32)
    model = DirectOnlyResidual()
    opt = torch.optim.AdamW(model.parameters(), lr=b.LR, weight_decay=b.WD)
    rng = np.random.default_rng(seed + 99173)
    for _ in range(b.EPOCHS):
        order = rng.permutation(len(x))
        model.train()
        for st in range(0, len(order), b.BATCH):
            ids = order[st:st+b.BATCH]
            opt.zero_grad(set_to_none=True)
            pred = model(torch.tensor(xn[ids], dtype=torch.float32))
            loss = nn.functional.mse_loss(pred, torch.tensor(target[ids], dtype=torch.float32))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
    model.eval()
    return model, mean, std


def predict(model, mean, std, x):
    with torch.no_grad():
        return model(torch.tensor((x-mean)/std, dtype=torch.float32)).numpy().astype(np.float32)


def infer_direct(model, trs, segs, norm, mu_map, h_map):
    s, c, _ = b.tensors(trs, segs, norm, mu_map, h_map, False)
    with torch.no_grad():
        return b.sigmoid(model(s, c).numpy()).astype(np.float32)


def shard(out: Path, outer: int, seed: int):
    target = out / 'shards' / f'fold{outer}_seed{seed}.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(json.dumps({'status':'EXISTS','path':str(target)})); return
    b.TPI, b.CF, b.FULL, cmap, traces, meta, audits, pairs, segs, _ = b.load_pop(out, f'shard{outer}_{seed}')
    md = b.ppv.frame(traces, meta); rf = b.root_fold(md)
    prep = np.load(BASE/'prepared.npz'); raw=prep['raw']; rich=prep['rich']; fold_ctx=prep['fold_ctx']
    _, b.PROBE_META, _ = b.load_raw_probe(traces)
    (mu_map, _), _, _ = b.maps_for_outer(BASE, outer, seed, raw, b.PROBE_META, fold_ctx)
    h_map = {r.context_id:rich[i].astype(np.float32) for i,r in b.PROBE_META.iterrows()}
    train_idx=np.flatnonzero(rf!=outer); held_idx=np.flatnonzero(rf==outer)
    tr_p=np.full(len(train_idx), np.nan, np.float32); loc={g:i for i,g in enumerate(train_idx)}
    # Strict inner cross-fit: each residual-training branch is scored by a
    # Direct model trained on the other inner fold only. Outer roots are absent.
    for j in [f for f in FOLDS if f != outer]:
        fitfold=next(f for f in FOLDS if f not in {outer,j})
        fi=np.flatnonzero(rf==fitfold); vi=np.flatnonzero(rf==j)
        fit=[traces[i] for i in fi]; val=[traces[i] for i in vi]
        ids={t.branch_id for t in fit}; mt={x:meta[x] for x in ids}
        nrm=b.xnorm(fit,segs,mu_map)
        direct=b.train_direct(fit,segs,nrm,mu_map,h_map,mt,seed+11*j,False)
        ll=np.asarray([loc[i] for i in vi]); tr_p[ll]=infer_direct(direct,val,segs,nrm,mu_map,h_map)
    if not np.isfinite(tr_p).all(): raise RuntimeError('inner Direct OOF predictions incomplete')
    task=md.task.to_numpy(int); force=md.force_N.to_numpy(float); success=md.success.to_numpy(float)
    tf=force[train_idx]; tt=task[train_idx]; tm=np.asarray([FMAX[int(x)] for x in tt])
    u=tr_p*((tm-tf)/tm)+(1-tr_p)*-1
    reward=np.where(success[train_idx]>0,(tm-tf)/tm,-1.)
    model,mean,std=fit_residual(features(u,tf,tm,tt),reward-u,seed)
    base=np.load(BASE/'shards'/f'fold{outer}_seed{seed}.npz')
    if not np.array_equal(base['held_idx'],held_idx): raise RuntimeError('base held-index alignment mismatch')
    hu=base['du_scalar'].astype(np.float32); hf=force[held_idx]; ht=task[held_idx]
    hm=np.asarray([FMAX[int(x)] for x in ht])
    delta=predict(model,mean,std,features(hu,hf,hm,ht))
    np.savez_compressed(target,held_idx=held_idx,base_utility=hu,delta=delta,corrected_utility=hu+delta)
    torch.save({'outer_fold':outer,'seed':seed,'state_dict':model.state_dict(),'input_mean':mean,'input_std':std,
                'architecture':'Linear(6,32)->GELU->Linear(32,1)','inputs':'ProbeScalar Direct utility + normalized force + task onehot4',
                'target':'R_real-U_D','loss':'MSE','epochs':b.EPOCHS,'nested_crossfit':True,
                'probe_root_heldout':True,'world_model_used':False,'TEST_used':False},out/'shards'/f'fold{outer}_seed{seed}.pt')
    print(json.dumps({'status':'SHARD_COMPLETE','fold':outer,'seed':seed,'held_branches':len(held_idx)}))


def ensemble_shard(out: Path, outer: int):
    """Exact match to the prior Direct-Only residual protocol.

    First average three nested OOF ProbeScalar Direct probabilities, then fit
    three residual seeds on that single frozen ensemble base.
    """
    target=out/'ensemble_shards'/f'fold{outer}.npz'; target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): print(json.dumps({'status':'EXISTS','path':str(target)})); return
    b.TPI,b.CF,b.FULL,cmap,traces,meta,audits,pairs,segs,_=b.load_pop(out,f'ensemble_shard{outer}')
    md=b.ppv.frame(traces,meta); rf=b.root_fold(md); prep=np.load(BASE/'prepared.npz')
    raw=prep['raw']; rich=prep['rich']; fold_ctx=prep['fold_ctx']; _,b.PROBE_META,_=b.load_raw_probe(traces)
    train_idx=np.flatnonzero(rf!=outer); held_idx=np.flatnonzero(rf==outer); loc={g:i for i,g in enumerate(train_idx)}
    tr_p_seed=np.full((3,len(train_idx)),np.nan,np.float32)
    for base_seed in SEEDS:
        (mu_map,_),_,_=b.maps_for_outer(BASE,outer,base_seed,raw,b.PROBE_META,fold_ctx)
        h_map={r.context_id:rich[i].astype(np.float32) for i,r in b.PROBE_META.iterrows()}
        for j in [f for f in FOLDS if f!=outer]:
            fitfold=next(f for f in FOLDS if f not in {outer,j}); fi=np.flatnonzero(rf==fitfold); vi=np.flatnonzero(rf==j)
            fit=[traces[i] for i in fi]; val=[traces[i] for i in vi]; ids={t.branch_id for t in fit}; mt={x:meta[x] for x in ids}
            nrm=b.xnorm(fit,segs,mu_map); direct=b.train_direct(fit,segs,nrm,mu_map,h_map,mt,base_seed+11*j,False)
            ll=np.asarray([loc[i] for i in vi]); tr_p_seed[base_seed,ll]=infer_direct(direct,val,segs,nrm,mu_map,h_map)
    if not np.isfinite(tr_p_seed).all(): raise RuntimeError('ensemble inner Direct OOF incomplete')
    task=md.task.to_numpy(int); force=md.force_N.to_numpy(float); success=md.success.to_numpy(float)
    tf=force[train_idx]; tt=task[train_idx]; tm=np.asarray([FMAX[int(x)] for x in tt])
    tr_p=tr_p_seed.mean(0); u=tr_p*((tm-tf)/tm)+(1-tr_p)*-1; reward=np.where(success[train_idx]>0,(tm-tf)/tm,-1.)
    x=features(u,tf,tm,tt)
    base_u=np.mean([np.load(BASE/'shards'/f'fold{outer}_seed{s}.npz')['du_scalar'] for s in SEEDS],axis=0).astype(np.float32)
    hf=force[held_idx]; ht=task[held_idx]; hm=np.asarray([FMAX[int(x)] for x in ht]); hx=features(base_u,hf,hm,ht)
    delta=[]
    for residual_seed in SEEDS:
        model,mean,std=fit_residual(x,reward-u,residual_seed); delta.append(predict(model,mean,std,hx))
        torch.save({'outer_fold':outer,'seed':residual_seed,'state_dict':model.state_dict(),'input_mean':mean,'input_std':std,
                    'architecture':'Linear(6,32)->GELU->Linear(32,1)','base':'three-seed nested OOF ProbeScalar Direct ensemble',
                    'target':'R_real-U_D','loss':'MSE','epochs':b.EPOCHS,'nested_crossfit':True,'probe_root_heldout':True,
                    'world_model_used':False,'TEST_used':False},out/'ensemble_shards'/f'fold{outer}_seed{residual_seed}.pt')
    delta=np.stack(delta); np.savez_compressed(target,held_idx=held_idx,base_utility=base_u,delta=delta,corrected_utility=base_u[None,:]+delta)
    print(json.dumps({'status':'ENSEMBLE_SHARD_COMPLETE','fold':outer,'held_branches':len(held_idx)}))


def metrics(q):
    return {'episodes':int(len(q)),'SR':float(q.success.mean()),'under_force':float(q.under_force.mean()),
            'mean_force_N':float(q.selected_force_N.mean()),'excess_force_N':float(q.excess_force_N.mean()),
            'realized_utility':float(q.realized_utility.mean()),'normalized_force_regret':float(q.regret.mean())}


def finalize(out: Path):
    b.TPI,b.CF,b.FULL,cmap,traces,meta,audits,pairs,segs,_=b.load_pop(out,'finalize')
    md=b.ppv.frame(traces,meta); n=len(md)
    md['R_real']=np.where(md.success>0,(md.task.map(FMAX)-md.force_N)/md.task.map(FMAX),-1.)
    base=np.full((3,n),np.nan,np.float32); corrected=np.full((3,n),np.nan,np.float32)
    for fold in FOLDS:
        for seed in SEEDS:
            z=np.load(out/'shards'/f'fold{fold}_seed{seed}.npz'); ids=z['held_idx']
            base[seed,ids]=z['base_utility']; corrected[seed,ids]=z['corrected_utility']
    if not np.isfinite(base).all() or not np.isfinite(corrected).all(): raise RuntimeError('incomplete aggregate')
    rows=[]; decisions=[]
    for label,arr in [('ProbeScalar-Direct',base),('ProbeScalar+Direct-Only-Residual',corrected)]:
        for seed in SEEDS:
            q=b.choose(md,arr[seed]); q['method']=label; q['seed']=str(seed); decisions.append(q)
        q=b.choose(md,arr.mean(0)); q['method']=label; q['seed']='ENSEMBLE_MEAN'; decisions.append(q)
    decisions=pd.concat(decisions,ignore_index=True)
    for (method,seed),q in decisions.groupby(['method','seed']):
        for task in TASKS: rows.append({'method':method,'seed':seed,'scope':f'task{task}',**metrics(q[q.task==task])})
        rows.append({'method':method,'seed':seed,'scope':'POOLED',**metrics(q)})
    table=pd.DataFrame(rows); table.to_csv(out/'PROBE_SCALAR_DIRECT_RESIDUAL_RESULTS.csv',index=False)
    decisions.to_csv(out/'PROBE_SCALAR_DIRECT_RESIDUAL_PER_EPISODE.csv',index=False)
    ens=table[(table.seed=='ENSEMBLE_MEAN')].set_index(['method','scope'])
    sb=ens.loc[('ProbeScalar-Direct','POOLED')]; sr=ens.loc[('ProbeScalar+Direct-Only-Residual','POOLED')]
    seed_gate=[]
    for seed in map(str,SEEDS):
        z=table[(table.seed==seed)&(table.scope=='POOLED')].set_index('method')
        seed_gate.append(bool(z.loc['ProbeScalar+Direct-Only-Residual','realized_utility']>z.loc['ProbeScalar-Direct','realized_utility']))
    summary={'classification':'PROBE_SCALAR_DIRECT_RESIDUAL_IMPROVES_UTILITY' if sr.realized_utility>sb.realized_utility and all(seed_gate) else 'PROBE_SCALAR_DIRECT_RESIDUAL_NOT_STABLE',
             'baseline':{k:float(sb[k]) for k in ['SR','under_force','mean_force_N','excess_force_N','realized_utility']},
             'residual':{k:float(sr[k]) for k in ['SR','under_force','mean_force_N','excess_force_N','realized_utility']},
             'delta':{k:float(sr[k]-sb[k]) for k in ['SR','under_force','mean_force_N','excess_force_N','realized_utility']},
             'utility_improves_all_3_seeds':bool(all(seed_gate)),'per_seed_utility_gate':seed_gate,
             'nested_root_crossfit':True,'probe_root_heldout':True,'world_model_used':False,'untouched_TEST_read':False}
    (out/'PROBE_SCALAR_DIRECT_RESIDUAL_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    lines=['# ProbeScalar Direct-Only Residual Diagnostic','',f"**{summary['classification']}**",'',
           'This matched diagnostic uses fully root-heldout Probe estimates and strict nested Direct/residual cross-fitting. It does not use a World Model or untouched TEST.','',
           '| Method | SR | Under-force | Mean force (N) | Excess force (N) | Realized utility |','|---|---:|---:|---:|---:|---:|',
           f"| ProbeScalar Direct | {sb.SR:.4f} | {sb.under_force:.4f} | {sb.mean_force_N:.4f} | {sb.excess_force_N:.4f} | {sb.realized_utility:.4f} |",
           f"| + Direct-Only residual | {sr.SR:.4f} | {sr.under_force:.4f} | {sr.mean_force_N:.4f} | {sr.excess_force_N:.4f} | {sr.realized_utility:.4f} |",'',
           f"Utility improves in all three seeds: {seed_gate}. Results are OOF development evidence, not untouched TEST evidence."]
    (out/'PROBE_SCALAR_DIRECT_RESIDUAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    req=['PROBE_SCALAR_DIRECT_RESIDUAL_RESULTS.csv','PROBE_SCALAR_DIRECT_RESIDUAL_PER_EPISODE.csv','PROBE_SCALAR_DIRECT_RESIDUAL_SUMMARY.json','PROBE_SCALAR_DIRECT_RESIDUAL_REPORT.md']
    with (out/'SHA256SUMS.txt').open('w') as f:
        for name in req: f.write(hashlib.sha256((out/name).read_bytes()).hexdigest()+'  '+name+'\n')
    print(json.dumps(summary,indent=2))


def finalize_ensemble(out: Path):
    b.TPI,b.CF,b.FULL,cmap,traces,meta,audits,pairs,segs,_=b.load_pop(out,'finalize_ensemble')
    md=b.ppv.frame(traces,meta); md['R_real']=np.where(md.success>0,(md.task.map(FMAX)-md.force_N)/md.task.map(FMAX),-1.); n=len(md)
    base=np.full(n,np.nan,np.float32); corrected=np.full((3,n),np.nan,np.float32)
    for fold in FOLDS:
        z=np.load(out/'ensemble_shards'/f'fold{fold}.npz'); ids=z['held_idx']; base[ids]=z['base_utility']; corrected[:,ids]=z['corrected_utility']
    if not np.isfinite(base).all() or not np.isfinite(corrected).all(): raise RuntimeError('incomplete ensemble aggregate')
    decisions=[]; q=b.choose(md,base); q['method']='ProbeScalar-Direct'; q['seed']='ENSEMBLE_BASE'; decisions.append(q)
    for seed in SEEDS:
        q=b.choose(md,corrected[seed]); q['method']='ProbeScalar+Direct-Only-Residual'; q['seed']=str(seed); decisions.append(q)
    q=b.choose(md,corrected.mean(0)); q['method']='ProbeScalar+Direct-Only-Residual'; q['seed']='ENSEMBLE_MEAN'; decisions.append(q)
    decisions=pd.concat(decisions,ignore_index=True); rows=[]
    for (method,seed),q in decisions.groupby(['method','seed']):
        for task in TASKS: rows.append({'method':method,'seed':seed,'scope':f'task{task}',**metrics(q[q.task==task])})
        rows.append({'method':method,'seed':seed,'scope':'POOLED',**metrics(q)})
    table=pd.DataFrame(rows); table.to_csv(out/'PROBE_SCALAR_DIRECT_RESIDUAL_ENSEMBLE_RESULTS.csv',index=False)
    decisions.to_csv(out/'PROBE_SCALAR_DIRECT_RESIDUAL_ENSEMBLE_PER_EPISODE.csv',index=False)
    pooled=table[table.scope=='POOLED'].set_index(['method','seed']); sb=pooled.loc[('ProbeScalar-Direct','ENSEMBLE_BASE')]; sr=pooled.loc[('ProbeScalar+Direct-Only-Residual','ENSEMBLE_MEAN')]
    seed_gate=[bool(pooled.loc[('ProbeScalar+Direct-Only-Residual',str(s)),'realized_utility']>sb.realized_utility) for s in SEEDS]
    summary={'classification':'PROBE_SCALAR_DIRECT_RESIDUAL_IMPROVES_UTILITY' if sr.realized_utility>sb.realized_utility and all(seed_gate) else 'PROBE_SCALAR_DIRECT_RESIDUAL_NOT_STABLE',
             'protocol':'exact prior protocol: three-seed nested OOF ProbeScalar Direct ensemble, then three residual seeds',
             'baseline':{k:float(sb[k]) for k in ['SR','under_force','mean_force_N','excess_force_N','realized_utility']},
             'residual':{k:float(sr[k]) for k in ['SR','under_force','mean_force_N','excess_force_N','realized_utility']},
             'delta':{k:float(sr[k]-sb[k]) for k in ['SR','under_force','mean_force_N','excess_force_N','realized_utility']},
             'utility_improves_all_3_residual_seeds':bool(all(seed_gate)),'per_seed_utility_gate':seed_gate,
             'nested_root_crossfit':True,'probe_root_heldout':True,'world_model_used':False,'untouched_TEST_read':False}
    (out/'PROBE_SCALAR_DIRECT_RESIDUAL_ENSEMBLE_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    old=pd.read_csv(ROOT/'activeforcing_residual_utility_20260901_055605/NORMAL_POOLED_UTILITY_TABLE.csv')
    oldb=old[(old.method=='P1 Direct Utility')&(old.seed.astype(str)=='ENSEMBLE')&(old.scope=='POOLED')].iloc[0]
    oldr=old[(old.method=='P2 Direct-Only Residual')&(old.seed.astype(str)=='ENSEMBLE_MEAN')&(old.scope=='POOLED')].iloc[0]
    comp=pd.DataFrame([
        {'physics_setting':'Prior authoritative/overlap physics','method':'Direct Utility','SR':oldb.full_task_SR,'under_force':oldb.under_force_rate,'mean_force_N':oldb.mean_force_N,'realized_utility':oldb.mean_realized_utility},
        {'physics_setting':'Prior authoritative/overlap physics','method':'Direct + Direct-Only Residual','SR':oldr.full_task_SR,'under_force':oldr.under_force_rate,'mean_force_N':oldr.mean_force_N,'realized_utility':oldr.mean_realized_utility},
        {'physics_setting':'Fully root-heldout OOF ProbeScalar','method':'Direct Utility','SR':sb.SR,'under_force':sb.under_force,'mean_force_N':sb.mean_force_N,'realized_utility':sb.realized_utility},
        {'physics_setting':'Fully root-heldout OOF ProbeScalar','method':'Direct + Direct-Only Residual','SR':sr.SR,'under_force':sr.under_force,'mean_force_N':sr.mean_force_N,'realized_utility':sr.realized_utility},
    ]); comp.to_csv(out/'AUTHORITATIVE_VS_OOF_PROBE_DIRECT_RESIDUAL.csv',index=False)
    report=f"""# OOF ProbeScalar Direct-Only Residual Report

**Matched result: {summary['classification']}.** With fully root-heldout Probe estimates, Direct-Only residual changes pooled ensemble SR from {sb.SR:.2%} to {sr.SR:.2%}, under-force from {sb.under_force:.2%} to {sr.under_force:.2%}, mean force from {sb.mean_force_N:.3f} N to {sr.mean_force_N:.3f} N, and realized utility from {sb.realized_utility:.4f} to {sr.realized_utility:.4f}.

## Interpretation

This exact matched protocol first ensembles three nested OOF ProbeScalar Direct predictions, then trains three residual seeds. It is directly comparable to the prior Direct-Only residual experiment. Utility seed gates are {seed_gate}. The seed-paired sensitivity is retained separately and is not substituted for this primary comparison.

## Scope and limits

All 24 root families are grouped; Probe, inner Direct and residual do not see an evaluated root. No World Model or untouched TEST is used. This remains pooled TRAIN-root OOF development evidence.
"""; (out/'PROBE_SCALAR_DIRECT_RESIDUAL_ENSEMBLE_REPORT.md').write_text(report)
    req=['PROBE_SCALAR_DIRECT_RESIDUAL_ENSEMBLE_RESULTS.csv','PROBE_SCALAR_DIRECT_RESIDUAL_ENSEMBLE_PER_EPISODE.csv','PROBE_SCALAR_DIRECT_RESIDUAL_ENSEMBLE_SUMMARY.json','AUTHORITATIVE_VS_OOF_PROBE_DIRECT_RESIDUAL.csv','PROBE_SCALAR_DIRECT_RESIDUAL_ENSEMBLE_REPORT.md']
    with (out/'ENSEMBLE_SHA256SUMS.txt').open('w') as f:
        for name in req: f.write(hashlib.sha256((out/name).read_bytes()).hexdigest()+'  '+name+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['shard','finalize','ensemble_shard','finalize_ensemble']); p.add_argument('--out',type=Path,required=True); p.add_argument('--fold',type=int); p.add_argument('--seed',type=int); a=p.parse_args()
    torch.set_num_threads(3); torch.set_num_interop_threads(1)
    a.out.mkdir(parents=True,exist_ok=True)
    if a.phase=='shard': shard(a.out.resolve(),a.fold,a.seed)
    elif a.phase=='finalize': finalize(a.out.resolve())
    elif a.phase=='ensemble_shard': ensemble_shard(a.out.resolve(),a.fold)
    else: finalize_ensemble(a.out.resolve())
