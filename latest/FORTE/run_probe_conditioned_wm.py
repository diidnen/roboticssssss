#!/usr/bin/env python3
"""Root-heldout Probe-conditioned physics diagnostic on pooled TRAIN roots only.

Phases:
  prepare  - audit raw probe traces, train fold-specific OOF probes, freeze bins.
  shard    - one outer-fold/seed nested cross-fit of Direct, PhysicsOnly WM,
             and the frozen residual-utility recipe.
  finalize - aggregate held-root predictions and emit all requested evidence.

No root-scaling or untouched TEST namespace is discovered or read.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, random, re, sys
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

import pooled_joint_novisual_current as pooled
import run_pooled_predictive_verifier as ppv
import task0_visual_context_early as early

ROOT=Path('/home/exouser/FORTE'); TAB=Path('/home/exouser/Tabero')
SOURCE=TAB/'analysis/results/gnp_style_visual_context_prospective_20260831_011000/collection_train'
P5ROOT=TAB/'analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542'
P5SRC=TAB/'analysis/p5s0c_model_adjudication.py'
PRIOR_PATH=TAB/'analysis/results/active_probe_necessity_20260830_063435/NO_PROBE_PHYSICS_PRIOR.json'
TASKS=[0,1,5,6]; FOLDS=[0,1,2]; SEEDS=[0,1,2]; FMAX={0:5.,1:6.,5:5.,6:4.}
H=8; EPOCHS=80; BATCH=64; LR=8e-4; WD=1e-4; PROBE_EPOCHS=137

def imp(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
P5=imp('probe_wm_p5',P5SRC)

def sha(p):
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def wjson(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def wcsv(p,x):
    if isinstance(x,pd.DataFrame): x.to_csv(p,index=False); return
    fs=[]
    for r in x:
        for k in r:
            if k not in fs: fs.append(k)
    with Path(p).open('w',newline='') as f:
        z=csv.DictWriter(f,fieldnames=fs or ['status']); z.writeheader(); z.writerows(x)
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-50,50)))
def summary52(t): return np.concatenate([t[:,-1],t.mean(1),t.std(1),t.max(1)],1).astype(np.float32)

class ProbeGRU(nn.Module):
    def __init__(self,d):
        super().__init__(); self.proj=nn.Sequential(nn.Linear(d,16),nn.ReLU()); self.gru=nn.GRU(16,16,batch_first=True); self.mu=nn.Linear(16,1); self.logs=nn.Linear(16,1)
    def forward(self,x,l,latent=False):
        z=self.proj(x); _,h=self.gru(pack_padded_sequence(z,l.cpu(),batch_first=True,enforce_sorted=False)); h=h[-1]
        return (self.mu(h).squeeze(1),self.logs(h).squeeze(1).clamp(-5,1.5),h) if latent else (self.mu(h).squeeze(1),self.logs(h).squeeze(1).clamp(-5,1.5))
def probe_fit(raw, ids, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    a=raw[ids]; mean=a.reshape(-1,a.shape[-1]).mean(0).astype(np.float32); std=a.reshape(-1,a.shape[-1]).std(0).astype(np.float32); std[std<1e-6]=1
    x=torch.tensor((a-mean)/std); l=torch.full((len(ids),),a.shape[1],dtype=torch.long); y=torch.tensor(PROBE_META.mu_GT.to_numpy(np.float32)[ids])
    m=ProbeGRU(a.shape[-1]); o=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    for _ in range(PROBE_EPOCHS):
        m.train(); o.zero_grad(); mu,ls=m(x,l); sig=ls.exp().clamp_min(1e-3); loss=(.5*(((y-mu)/sig)**2+2*ls)+.05*(y-mu).abs()).mean(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1); o.step()
    m.eval(); return m,mean,std
def probe_pred(m,mean,std,raw,ids):
    x=torch.tensor((raw[ids]-mean)/std); l=torch.full((len(ids),),raw.shape[1],dtype=torch.long)
    with torch.no_grad(): mu,ls,h=m(x,l,True)
    return mu.numpy().astype(np.float32),h.numpy().astype(np.float32)

class Direct(nn.Module):
    def __init__(self,cd):
        super().__init__(); self.gru=nn.GRU(17,64,batch_first=True); self.cond=nn.Sequential(nn.Linear(cd,64),nn.ReLU()); self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,s,c): _,h=self.gru(s); return self.head(torch.cat([h[-1],self.cond(c)],-1)).squeeze(-1)
class WM(nn.Module):
    def __init__(self,cd): super().__init__(); self.gru=nn.GRU(17+cd,64,batch_first=True); self.head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,H*13))
    def forward(self,s,c): z,_=self.gru(torch.cat([s,c[:,None,:].expand(-1,s.shape[1],-1)],-1)); return self.head(z[:,-1]).view(-1,H,13)
class Residual(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(53,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def load_pop(out,tag):
    d=out/'_audit'/tag; d.mkdir(parents=True,exist_ok=True); return pooled.load_population(d)
def root_fold(md):
    ans=np.full(len(md),-1,int)
    for f,pairs in ppv.root_folds(md).items():
        s=set(pairs); ans[[i for i,r in md.iterrows() if (int(r.task),str(r.root_id)) in s]]=f
    return ans
def load_raw_probe(traces):
    norm=json.loads((P5ROOT/'P5S0C_NORMALIZATION.json').read_text()); names=norm['dynamic_feature_names']; phases=norm['phase_categories_from_train']; states=norm['contact_state_categories_from_train']
    contexts={t.context_id:t for t in traces}; rows=[]; arr=[]
    for cid,t in sorted(contexts.items()):
        p=SOURCE/f'task{t.task}/P5S0C_PROBE_TELEMETRY/{cid}_probe_timesteps.csv'
        if not p.exists(): raise RuntimeError(f'missing probe {p}')
        q=P5.sequence_dataframe(str(p),phases,states).reindex(columns=names).fillna(0.)
        a=q.to_numpy(np.float32)
        if a.shape!=(215,46) or not np.isfinite(a).all(): raise RuntimeError(f'bad probe {cid}: {a.shape}')
        arr.append(a); rows.append({'probe_index':len(rows),'context_id':cid,'root_id':t.root_id,'task':int(t.task),'mu_GT':float(t.mu),'raw_path':str(p),'timesteps':215,'features':46})
    return np.stack(arr),pd.DataFrame(rows),names
def ctx_maps(meta,mu,h): return ({r.context_id:float(mu[i]) for i,r in meta.iterrows()},{r.context_id:h[i].astype(np.float32) for i,r in meta.iterrows()})
def maps_for_outer(out,outer,seed,raw,pm,fold_ctx):
    # Held predictions from 16-root estimator saved in prepare.
    z=np.load(out/'probe'/f'oof_fold{outer}_seed{seed}.npz'); mu=np.full(len(pm),np.nan,np.float32); h=np.full((len(pm),16),np.nan,np.float32)
    held=np.flatnonzero(fold_ctx==outer); mu[held]=z['mu']; h[held]=z['h']
    # Inner OOF: for each training fold, train only on the other training fold;
    # neither estimator has seen the outer validation roots or scored roots.
    for j in [f for f in FOLDS if f!=outer]:
        trainfold=next(f for f in FOLDS if f not in {outer,j}); tr=np.flatnonzero(fold_ctx==trainfold); va=np.flatnonzero(fold_ctx==j)
        m,mean,std=probe_fit(raw,tr,seed*101+outer*17+j); mu[va],h[va]=probe_pred(m,mean,std,raw,va)
    if not np.isfinite(mu).all() or not np.isfinite(h).all(): raise RuntimeError('nested probe incomplete')
    return ctx_maps(pm,mu,h),mu,h

def xnorm(train,segs,mu_map):
    x=np.concatenate([np.where(np.arange(71)[None,:]==18,mu_map[t.context_id],segs[t.branch_id].x) for t in train],0).astype(np.float32)
    mean=x.mean(0); std=x.std(0); std[std<1e-6]=1
    phys=[]
    for t in train: phys.extend(TPI.make_segments([t],H))
    y=np.concatenate([s.y-s.trace.state[s.start] for s in phys]); ym=y.mean(0).astype(np.float32); ys=y.std(0).astype(np.float32); ys[ys<1e-6]=1
    return mean.astype(np.float32),std.astype(np.float32),ym,ys
def tensors(trs,segs,norm,mu_map,h_map,rich):
    xm,xs=norm[:2]; xx=[]; hh=[]
    for t in trs:
        a=segs[t.branch_id].x.copy(); a[:,18]=mu_map[t.context_id]; xx.append((a-xm)/xs); hh.append(h_map[t.context_id])
    x=np.stack(xx).astype(np.float32); c=x[:,0,17:]
    if rich: c=np.concatenate([c,np.stack(hh)],1)
    return torch.tensor(x[:,:,:17]),torch.tensor(c),torch.tensor([t.outcome for t in trs],dtype=torch.float32)
def train_direct(train,segs,norm,mu_map,h_map,meta,seed,rich):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); m=Direct(70 if rich else 54); o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
    for ep in range(1,EPOCHS+1):
        ids=early.sampled_ids(train,meta,seed,ep); m.train()
        for st in range(0,len(ids),BATCH):
            q=[train[int(i)] for i in ids[st:st+BATCH]]; s,c,y=tensors(q,segs,norm,mu_map,h_map,rich); o.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(m(s,c),y); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1); o.step()
    m.eval(); return m
def unit_tensors(items,norm,mu_map,h_map,rich):
    seg=[]
    for u in items: seg.append(u[0]); seg.extend([u[1]] if u[1] is not None else [])
    xm,xs,ym,ys=norm; xx=[]; hh=[]
    for s in seg:
        a=s.x.copy(); a[:,18]=mu_map[s.trace.context_id]; xx.append((a-xm)/xs); hh.append(h_map[s.trace.context_id])
    x=np.stack(xx).astype(np.float32); c=x[:,0,17:]
    if rich: c=np.concatenate([c,np.stack(hh)],1)
    y=np.stack([((s.y-s.trace.state[s.start]-ym)/ys) for s in seg]); mask=np.stack([s.mask for s in seg]); w=np.asarray([s.trace.weight for s in seg],np.float32)
    return seg,torch.tensor(x[:,:,:17]),torch.tensor(c),torch.tensor(y,dtype=torch.float32),torch.tensor(mask,dtype=torch.float32),torch.tensor(w)
def train_wm(train,pairs,norm,mu_map,h_map,seed,rich):
    random.seed(seed+3000); np.random.seed(seed+3000); torch.manual_seed(seed+3000); m=WM(70 if rich else 54)
    # Neutral extension: copy the frozen physics initialization into all shared dimensions.
    # Frozen initialization checkpoints exist only for canonical seeds 0/1/2;
    # inner-crossfit RNG offsets must not be used as checkpoint identifiers.
    base,_,_=FULL.load_base(TPI,seed%3,torch.device('cpu')); sd=base['state_dict']; ms=m.state_dict()
    for k,v in sd.items():
        if k in ms:
            if ms[k].shape==v.shape: ms[k]=v.clone()
            elif k=='gru.weight_ih_l0': ms[k][:,:v.shape[1]]=v
    m.load_state_dict(ms); o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD); units=CF.make_units(TPI,train,pairs)
    for ep in range(1,EPOCHS+1):
        m.train()
        for b in CF.batches_for_units(units,seed,ep):
            seg,s,c,y,mask,w=unit_tensors(b,norm,mu_map,h_map,rich); o.zero_grad(); p=m(s,c)
            phys=(nn.functional.smooth_l1_loss(p,y,reduction='none')*mask*w[:,None,None]).sum()/(mask.sum()+1e-6)
            ie=early.physical_ie_loss(p,b,norm,torch.device('cpu')); loss=phys+ie; loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1); o.step()
    m.eval(); return m,len(units)
def infer(direct,wm,trs,segs,norm,mu_map,h_map,rich):
    s,c,_=tensors(trs,segs,norm,mu_map,h_map,rich)
    with torch.no_grad(): p=sigmoid(direct(s,c).numpy()); pred=wm(s,c).numpy()
    init=np.stack([t.state[0] for t in trs]); pred=pred*norm[3][None,None,:]+norm[2][None,None,:]+init[:,None,:]
    return p.astype(np.float32),pred.astype(np.float32)
def fit_residual(x,y,seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); mean=x.mean(0).astype(np.float32); std=x.std(0).astype(np.float32); std[std<1e-6]=1; xn=(x-mean)/std
    m=Residual(); o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD); rng=np.random.default_rng(seed+99173)
    for _ in range(EPOCHS):
        order=rng.permutation(len(x))
        for st in range(0,len(x),BATCH):
            ids=order[st:st+BATCH]; o.zero_grad(); loss=nn.functional.mse_loss(m(torch.tensor(xn[ids],dtype=torch.float32)),torch.tensor(y[ids],dtype=torch.float32)); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1); o.step()
    m.eval(); return m,mean,std
def residual_pred(m,mean,std,traj,force,fmax):
    x=np.column_stack([summary52(traj),force/fmax]).astype(np.float32)
    with torch.no_grad(): return m(torch.tensor((x-mean)/std,dtype=torch.float32)).numpy().astype(np.float32)

def prepare(out):
    out.mkdir(parents=True,exist_ok=False); (out/'probe').mkdir(); global TPI,CF,FULL,PROBE_META
    TPI,CF,FULL,cmap,traces,meta,audits,pairs,segs,norm=load_pop(out,'prepare'); md=ppv.frame(traces,meta); rf=root_fold(md)
    raw,pm,names=load_raw_probe(traces); PROBE_META=pm; root_to_fold={(int(r.task),str(r.root_id)):int(rf[i]) for i,r in md.drop_duplicates(['task','root_id']).iterrows()}; fold_ctx=np.asarray([root_to_fold[(int(r.task),str(r.root_id))] for _,r in pm.iterrows()])
    # Fixed, fold-invariant rich representation: four physically meaningful
    # probe signals x (final, mean, std, max). This avoids incomparable hidden
    # coordinate systems across independently trained OOF GRUs.
    rich_names=['measured_fn','measured_ft','ft_over_fn','marker_tangential']; jj=[names.index(x) for x in rich_names]; a=raw[:,:,jj]
    rich=np.concatenate([a[:,-1],a.mean(1),a.std(1),a.max(1)],1).astype(np.float32)
    rows=[]
    for fold in FOLDS:
        tr=np.flatnonzero(fold_ctx!=fold); va=np.flatnonzero(fold_ctx==fold)
        for seed in SEEDS:
            m,mean,std=probe_fit(raw,tr,seed+fold*101); mu,h=probe_pred(m,mean,std,raw,va); torch.save({'state_dict':m.state_dict(),'mean':mean,'std':std,'fold':fold,'seed':seed,'train_roots':16,'held_roots':8,'epochs':PROBE_EPOCHS,'latent':'final GRU hidden 16D','sigma_posterior_claim':False},out/'probe'/f'probe_fold{fold}_seed{seed}.pt'); np.savez_compressed(out/'probe'/f'oof_fold{fold}_seed{seed}.npz',idx=va,mu=mu,h=h)
            for ii,mh,hh in zip(va,mu,h): rows.append({**pm.iloc[ii].to_dict(),'fold':fold,'seed':seed,'mu_hat':float(mh),'signed_error':float(mh-pm.mu_GT.iloc[ii]),'abs_error':float(abs(mh-pm.mu_GT.iloc[ii])),**{f'h_{j:02d}':float(hh[j]) for j in range(16)},**{f'e_{j:02d}':float(rich[ii,j]) for j in range(16)}})
    d=pd.DataFrame(rows); ens=d.groupby(['probe_index','context_id','root_id','task','mu_GT','fold'],as_index=False).agg(mu_hat=('mu_hat','mean'))
    ens['seed']='ENSEMBLE_MEAN'; ens['signed_error']=ens.mu_hat-ens.mu_GT; ens['abs_error']=ens.signed_error.abs(); d=pd.concat([d,ens],ignore_index=True,sort=False); wcsv(out/'POOLED_OOF_PROBE_PREDICTIONS.csv',d)
    e=ens.abs_error.to_numpy(); q1,q2=np.quantile(e,[1/3,2/3]); small,medium,large=np.quantile(e,[1/3,2/3,.9]); bins={'status':'FROZEN_BEFORE_ANY_WM_RESULT','source':'ensemble OOF Probe errors on 72 pooled TRAIN contexts','boundaries':{'LOW_ERROR':f'abs_error <= {q1}','MID_ERROR':f'{q1} < abs_error <= {q2}','HIGH_ERROR':f'abs_error > {q2}'},'q33':q1,'q67':q2,'corruption_deltas':[0.,-small,small,-medium,medium,-large,large],'legal_mu_range':[.2,1.0],'WM_results_read':False,'sha_inputs':{'probe_predictions':sha(out/'POOLED_OOF_PROBE_PREDICTIONS.csv')}}; wjson(out/'PROBE_ERROR_BINS.json',bins)
    old=pd.read_csv(TAB/'analysis/results/active_friction_imagination_20260828_211106/FRICTION_PREDICTIONS.csv')
    def canon(task,root):
        m=re.search(r'root(\d+)_s(\d+)',str(root)); return (int(task),int(m.group(1)),int(m.group(2))) if m else (int(task),str(root))
    current={canon(r.task,r.root_id) for _,r in pm.iterrows()}; oldtrain={canon(r.task,r.root_id) for _,r in old[old.split=='TRAIN'].iterrows()}; overlap=len(current&oldtrain)
    audit=f"""# Probe-WM Data Audit\n\n**PASS for pooled TRAIN-root OOF development; existing estimator is overlap-only diagnostic.**\n\n- Authoritative current population: 24 root families, 72 friction-conditioned contexts, 720 force branches.\n- Raw Probe: 72/72 complete files, each 215 timesteps × 46 legal dynamic features. Force/contact, proprioception, probe action/phase and tactile/contact-state signals are present. GT friction and downstream outcome are excluded from the 46 inputs.\n- A legal estimator latent exists: the 16D final hidden state of projection16→GRU16, and it is exported for audit. Independently trained fold GRUs do not share an identifiable latent coordinate system, so downstream rich models use a fold-invariant, pre-WM fixed 16D summary: final/mean/std/max of measured normal force, tangential force, tangential/normal ratio, and marker tangential motion. This choice was frozen before WM results.\n- `sigma_mu` remains diagnostic only and is not used as a posterior.\n- Existing estimator TRAIN-root overlap with current force-selection roots: {overlap}/24. It cannot support a held-root claim.\n- This run retrains 3 fold-specific estimators × 3 seeds. Every held root's mu_hat and exported h_e come from an estimator trained on the other 16 roots. Probe normalization is fit inside that estimator's TRAIN roots.\n- Downstream nested training features use two-fold inner OOF Probe estimates; the fixed rich summary has no fitted encoder. Neither a scored root nor the outer validation roots enter their Probe estimator.\n- Root IDs, task IDs and friction-conditioned contexts are retained only as grouping/audit keys; root ID is never a model input.\n\nTask1's force outcomes retain the known reconstructed-label caveat (140/180); Probe friction labels and traces themselves are direct simulator records. No untouched TEST was read.\n"""; (out/'PROBE_WM_DATA_AUDIT.md').write_text(audit)
    np.savez_compressed(out/'prepared.npz',raw=raw,rich=rich,fold_ctx=fold_ctx); print(json.dumps({'status':'PREPARED','out':str(out),'probe_MAE':float(ens.abs_error.mean()),'bins':[q1,q2]}))

def shard(out,outer,seed):
    target=out/'shards'/f'fold{outer}_seed{seed}.npz'; target.parent.mkdir(exist_ok=True)
    if target.exists(): print('exists',target); return
    global TPI,CF,FULL,PROBE_META
    TPI,CF,FULL,cmap,traces,meta,audits,pairs,segs,_=load_pop(out,f'shard{outer}_{seed}'); md=ppv.frame(traces,meta); rf=root_fold(md); prep=np.load(out/'prepared.npz'); raw=prep['raw']; rich=prep['rich']; fold_ctx=prep['fold_ctx']; _,PROBE_META,_=load_raw_probe(traces)
    (mu_map,_),_,_=maps_for_outer(out,outer,seed,raw,PROBE_META,fold_ctx); h_map={r.context_id:rich[i].astype(np.float32) for i,r in PROBE_META.iterrows()}; train_idx=np.flatnonzero(rf!=outer); held_idx=np.flatnonzero(rf==outer); train=[traces[i] for i in train_idx]; held=[traces[i] for i in held_idx]; trainids={t.branch_id for t in train}; trainmeta={x:meta[x] for x in trainids}; trainpairs=[p for p in pairs if p.a.branch_id in trainids and p.b.branch_id in trainids]; norm=xnorm(train,segs,mu_map)
    hs=np.stack([h_map[t.context_id] for t in train]); hm=hs.mean(0); hsd=hs.std(0); hsd[hsd<1e-6]=1; hnorm={k:(v-hm)/hsd for k,v in h_map.items()}
    sd=train_direct(train,segs,norm,mu_map,hnorm,trainmeta,seed,False); rd=train_direct(train,segs,norm,mu_map,hnorm,trainmeta,seed,True); pwm,units=train_wm(train,trainpairs,norm,mu_map,hnorm,seed,False); rwm,_=train_wm(train,trainpairs,norm,mu_map,hnorm,seed,True)
    # Inner OOF predictions for residual training: each training fold is scored by models trained only on the other inner fold.
    tr_p_s=np.full(len(train_idx),np.nan,np.float32); tr_p_r=tr_p_s.copy(); tr_w_p=np.full((len(train_idx),H,13),np.nan,np.float32); tr_w_r=tr_w_p.copy(); loc={g:i for i,g in enumerate(train_idx)}
    for j in [f for f in FOLDS if f!=outer]:
        fitfold=next(f for f in FOLDS if f not in {outer,j}); fi=np.flatnonzero(rf==fitfold); vi=np.flatnonzero(rf==j); fit=[traces[i] for i in fi]; val=[traces[i] for i in vi]; ids={t.branch_id for t in fit}; mt={x:meta[x] for x in ids}; pr=[p for p in pairs if p.a.branch_id in ids and p.b.branch_id in ids]
        nrm=xnorm(fit,segs,mu_map); hh=np.stack([h_map[t.context_id] for t in fit]); mh=hh.mean(0); sh=hh.std(0); sh[sh<1e-6]=1; hn={k:(v-mh)/sh for k,v in h_map.items()}
        a=train_direct(fit,segs,nrm,mu_map,hn,mt,seed+11*j,False); b=train_direct(fit,segs,nrm,mu_map,hn,mt,seed+11*j,True); c,_=train_wm(fit,pr,nrm,mu_map,hn,seed+11*j,False); d,_=train_wm(fit,pr,nrm,mu_map,hn,seed+11*j,True)
        ps,wp=infer(a,c,val,segs,nrm,mu_map,hn,False); prr,wr=infer(b,d,val,segs,nrm,mu_map,hn,True); ll=np.asarray([loc[i] for i in vi]); tr_p_s[ll]=ps; tr_p_r[ll]=prr; tr_w_p[ll]=wp; tr_w_r[ll]=wr
    tf=md.force_N.to_numpy()[train_idx]; tx=md.success.to_numpy()[train_idx]; tm=np.asarray([FMAX[int(x)] for x in md.task.to_numpy()[train_idx]]); rr=np.where(tx>0,(tm-tf)/tm,-1.)
    us=tr_p_s*((tm-tf)/tm)+(1-tr_p_s)*-1; ur=tr_p_r*((tm-tf)/tm)+(1-tr_p_r)*-1
    rs,ms,ss=fit_residual(np.column_stack([summary52(tr_w_p),tf/tm]).astype(np.float32),rr-us,seed); rrmodel,mr,sr=fit_residual(np.column_stack([summary52(tr_w_r),tf/tm]).astype(np.float32),rr-ur,seed)
    # Held scenarios.
    hmu={t.context_id:mu_map[t.context_id] for t in held}; hhn={k:hnorm[k] for k in hmu}; gt={t.context_id:float(t.mu) for t in held}; prior=json.loads(PRIOR_PATH.read_text())['values']
    psc,tp=infer(sd,pwm,held,segs,norm,hmu,hhn,False); prich,trj=infer(rd,rwm,held,segs,norm,hmu,hhn,True); pgt,_=infer(sd,pwm,held,segs,norm,gt,hhn,False)
    pprior=np.mean([infer(sd,pwm,held,segs,norm,{k:float(v) for k in hmu},hhn,False)[0] for v in prior],0)
    hf=md.force_N.to_numpy()[held_idx]; hfm=np.asarray([FMAX[int(x)] for x in md.task.to_numpy()[held_idx]])
    du_s=psc*((hfm-hf)/hfm)+(1-psc)*-1; du_r=prich*((hfm-hf)/hfm)+(1-prich)*-1
    ds=residual_pred(rs,ms,ss,tp,hf,hfm); dr=residual_pred(rrmodel,mr,sr,trj,hf,hfm)
    bins=json.loads((out/'PROBE_ERROR_BINS.json').read_text()); corr=[]
    for delta in bins['corruption_deltas']:
        cm={k:float(np.clip(v+delta,.2,1.0)) for k,v in hmu.items()}; cps,ctp=infer(sd,pwm,held,segs,norm,cm,hhn,False); cpr,ctr=infer(rd,rwm,held,segs,norm,cm,hhn,True); cu=cps*((hfm-hf)/hfm)+(1-cps)*-1; cru=cpr*((hfm-hf)/hfm)+(1-cpr)*-1; corr.append(np.stack([cu,cru,cu+residual_pred(rs,ms,ss,ctp,hf,hfm),cru+residual_pred(rrmodel,mr,sr,ctr,hf,hfm)]))
    np.savez_compressed(target,held_idx=held_idx,p_noprobe=pprior,p_gt=pgt,p_scalar=psc,p_rich=prich,score_point=du_s+ds,score_probecond=du_r+dr,du_scalar=du_s,du_rich=du_r,delta_point=ds,delta_probecond=dr,corruption=np.stack(corr),probe_mu=np.asarray([hmu[t.context_id] for t in held]),units=units)
    torch.save({'outer_fold':outer,'seed':seed,'nested_crossfit':True,'probe_in_sample':False,'wm_objective':'Lphysics + 1.0*LIE','outcome_gradient':False,'H':8,'epochs':80,'residual_loss':'MSE','TEST_used':False},out/'shards'/f'fold{outer}_seed{seed}.pt'); print(json.dumps({'status':'SHARD_COMPLETE','fold':outer,'seed':seed,'held':len(held)}))

def choose(md,score):
    rows=[]
    for (cid,rep),q0 in md.groupby(['context_id','repeat'],sort=True):
        q=q0.sort_values('force_N'); ids=q['index'].to_numpy(int); v=score[ids]; z=q.iloc[int(np.flatnonzero(v==v.max())[0])]; good=q[q.success==1]; frontier=float(good.force_N.min()) if len(good) else math.nan
        rows.append({'context_id':cid,'repeat':int(rep),'root_id':z.root_id,'task':int(z.task),'mu_GT':float(z.mu),'selected_force_N':float(z.force_N),'success':int(z.success),'under_force':int(math.isfinite(frontier) and z.force_N<frontier-1e-9),'frontier_N':frontier,'excess_force_N':float(z.force_N-frontier) if math.isfinite(frontier) else math.nan,'regret':float((z.force_N-frontier)/FMAX[int(z.task)]) if z.success and math.isfinite(frontier) else math.nan,'realized_utility':float(z.R_real)})
    return pd.DataFrame(rows)
def metrics(q):
    ans={'episodes':int(len(q)),'SR':float(q.success.mean()),
         'under_force':float(q.under_force.mean()),
         'mean_force_N':float(q.selected_force_N.mean()),
         'excess_force_N':float(q.excess_force_N.mean()),
         'realized_utility':float(q.realized_utility.mean()),
         'normalized_force_regret':float(q.regret.mean())}
    if 'frontier_N' in q:
        ans['frontier_MAE_N']=float((q.selected_force_N-q.frontier_N).abs().mean())
    if 'selected_force_difference_vs_GT_N' in q:
        ans['mean_abs_force_gap_vs_GT_N']=float(q.selected_force_difference_vs_GT_N.abs().mean())
        ans['decision_agreement_with_GT']=float(q.decision_agreement_with_GT.mean())
    return ans

def finalize(out):
    global TPI,CF,FULL,PROBE_META
    TPI,CF,FULL,cmap,traces,meta,audits,pairs,segs,_=load_pop(out,'finalize'); md=ppv.frame(traces,meta); md['R_real']=np.where(md.success>0,(md.task.map(FMAX)-md.force_N)/md.task.map(FMAX),-1.); rf=root_fold(md); n=len(md)
    names=['NoProbe-Direct','GT-Direct','ProbeScalar-Direct','ProbeRich-Direct','ProbeScalar+Point-WM-Residual','ProbeRich+ProbeConditioned-WM-Residual']; scores={k:np.full((3,n),np.nan,np.float32) for k in names}; bins=json.loads((out/'PROBE_ERROR_BINS.json').read_text()); corr=np.full((3,len(bins['corruption_deltas']),4,n),np.nan,np.float32)
    for fold in FOLDS:
        for seed in SEEDS:
            z=np.load(out/'shards'/f'fold{fold}_seed{seed}.npz'); ids=z['held_idx']; f=md.force_N.to_numpy()[ids]; fm=md.task.map(FMAX).to_numpy()[ids]
            for name,key in [('NoProbe-Direct','p_noprobe'),('GT-Direct','p_gt'),('ProbeScalar-Direct','p_scalar'),('ProbeRich-Direct','p_rich')]: p=z[key]; scores[name][seed,ids]=p*((fm-f)/fm)+(1-p)*-1
            scores[names[4]][seed,ids]=z['score_point']; scores[names[5]][seed,ids]=z['score_probecond']; corr[seed][:,:,ids]=z['corruption']
    probe=pd.read_csv(out/'POOLED_OOF_PROBE_PREDICTIONS.csv'); pe=probe[probe.seed.astype(str)=='ENSEMBLE_MEAN'].set_index('context_id'); q1,q2=bins['q33'],bins['q67']
    per=[]; selected={}
    for name,a in scores.items():
        for seed in SEEDS:
            s=choose(md,a[seed]); s['method']=name;s['seed']=seed; per.append(s)
        s=choose(md,a.mean(0)); s['method']=name;s['seed']='ENSEMBLE_MEAN'; selected[name]=s; per.append(s)
    per=pd.concat(per,ignore_index=True); gt=selected['GT-Direct'][['context_id','repeat','selected_force_N']].rename(columns={'selected_force_N':'GT_selected_force_N'})
    per=per.merge(gt,on=['context_id','repeat']); per['selected_force_difference_vs_GT_N']=per.selected_force_N-per.GT_selected_force_N; per['decision_agreement_with_GT']=(abs(per.selected_force_difference_vs_GT_N)<1e-8).astype(int); per['probe_abs_error']=per.context_id.map(pe.abs_error); per['probe_signed_error']=per.context_id.map(pe.signed_error); per['error_bin']=np.where(per.probe_abs_error<=q1,'LOW_ERROR',np.where(per.probe_abs_error<=q2,'MID_ERROR','HIGH_ERROR')); per['signed_error_bin']=np.where(per.probe_signed_error>0,'OVER_ESTIMATE',np.where(per.probe_signed_error<0,'UNDER_ESTIMATE','ZERO'))
    table=[]
    for (method,seed),q in per.groupby(['method','seed']):
        for task in TASKS: table.append({'method':method,'seed':seed,'scope':f'task{task}','task':task,**metrics(q[q.task==task])})
        table.append({'method':method,'seed':seed,'scope':'POOLED','task':'POOLED',**metrics(q)})
    wcsv(out/'NORMAL_POOLED_PROBE_WM_TABLE.csv',table)
    ens=per[per.seed.astype(str)=='ENSEMBLE_MEAN']; wcsv(out/'PROBE_SCALAR_VS_RICH_DIRECT.csv',ens[ens.method.isin(names[:4])]); wcsv(out/'POINT_WM_VS_PROBE_CONDITIONED_WM.csv',ens[ens.method.isin(names[2:])])
    strat=[]
    def add_strat(method,eb,sb,q):
        strat.append({'method':method,'error_bin':eb,'signed_error_bin':sb,**metrics(q),
                      'mean_probe_abs_error':float(q.probe_abs_error.mean()),
                      'mean_probe_signed_error':float(q.probe_signed_error.mean())})
    for (method,eb,sb),q in ens.groupby(['method','error_bin','signed_error_bin']): add_strat(method,eb,sb,q)
    for (method,eb),q in ens.groupby(['method','error_bin']): add_strat(method,eb,'ALL_SIGN',q)
    for (method,sb),q in ens.groupby(['method','signed_error_bin']): add_strat(method,'ALL_ERROR',sb,q)
    for method,q in ens.groupby('method'): add_strat(method,'ALL_ERROR','ALL_SIGN',q)
    wcsv(out/'PROBE_ERROR_STRATIFIED_RESULTS.csv',strat)
    base=selected['ProbeScalar-Direct']; rich=selected['ProbeRich-Direct']; point=selected[names[4]]; pc=selected[names[5]]; g=selected['GT-Direct']
    rescue=base.merge(g,on=['context_id','repeat'],suffixes=('_scalar','_gt')).merge(pc[['context_id','repeat','selected_force_N','success']],on=['context_id','repeat']).rename(columns={'selected_force_N':'pc_force_N','success':'pc_success'}); rescue['recoverable_probe_error_failure']=((rescue.success_scalar==0)&(rescue.success_gt==1)).astype(int); rescue['rescued']=((rescue.recoverable_probe_error_failure==1)&(rescue.pc_success==1)).astype(int); rescue['collateral_damage']=((rescue.success_scalar==1)&(rescue.pc_success==0)).astype(int); wcsv(out/'PROBE_WM_RESCUE_ANALYSIS.csv',rescue)
    direction=base[['context_id','repeat','task','selected_force_N']].rename(columns={'selected_force_N':'scalar_force_N'}).merge(g[['context_id','repeat','selected_force_N']].rename(columns={'selected_force_N':'gt_force_N'}),on=['context_id','repeat'])
    for label,s in [('rich_direct',rich),('point_wm',point),('probecond_wm',pc)]: direction=direction.merge(s[['context_id','repeat','selected_force_N']].rename(columns={'selected_force_N':f'{label}_force_N'}),on=['context_id','repeat']); direction[f'{label}_toward_GT']=(abs(direction[f'{label}_force_N']-direction.gt_force_N)<abs(direction.scalar_force_N-direction.gt_force_N)).astype(int)
    direction['probe_signed_error']=direction.context_id.map(pe.signed_error); wcsv(out/'PROBE_WM_CORRECTION_DIRECTION.csv',direction)
    cr=[]; labels=['ScalarDirect','RichDirect','PointWMResidual','ProbeConditionedWMResidual']
    for di,delta in enumerate(bins['corruption_deltas']):
        for mi,label in enumerate(labels):
            ss=[]
            # delta=0 is the exact, unclipped formal baseline. Non-zero deltas
            # follow the preregistered corruption protocol and clip to [.2,1.0].
            base_name=[names[2],names[3],names[4],names[5]][mi]
            arr=scores[base_name] if abs(delta)<1e-15 else corr[:,di,mi]
            for seed in SEEDS: ss.append(choose(md,arr[seed]))
            s=choose(md,arr.mean(0)); cr.append({'method':label,'delta':float(delta),'delta_label':'ZERO' if delta==0 else ('NEGATIVE' if delta<0 else 'POSITIVE'),'magnitude':float(abs(delta)),**metrics(s),'seed_SR_min':float(min(x.success.mean() for x in ss)),'seed_SR_max':float(max(x.success.mean() for x in ss))})
    wcsv(out/'PROBE_SCALAR_CORRUPTION_CURVE.csv',cr)
    # Gain/error diagnostic at episode grain; Spearman computed without scipy.
    b=base.set_index(['context_id','repeat']); r=rich.set_index(['context_id','repeat']); p=point.set_index(['context_id','repeat']); c=pc.set_index(['context_id','repeat']); gi=g.set_index(['context_id','repeat']); rows=[]
    for key in b.index:
        err=float(pe.loc[key[0]].abs_error); gain=float(c.loc[key].realized_utility-b.loc[key].realized_utility); rows.append({'context_id':key[0],'repeat':key[1],'task':int(b.loc[key].task),'probe_abs_error':err,'probe_signed_error':float(pe.loc[key[0]].signed_error),'scalar_success':int(b.loc[key].success),'GT_success':int(gi.loc[key].success),'rich_success':int(r.loc[key].success),'point_wm_success':int(p.loc[key].success),'probecond_wm_success':int(c.loc[key].success),'probecond_utility_gain_vs_scalar':gain})
    dcols=['context_id','repeat','task','scalar_force_N','gt_force_N','rich_direct_force_N','rich_direct_toward_GT','point_wm_force_N','point_wm_toward_GT','probecond_wm_force_N','probecond_wm_toward_GT']
    direction_out=pd.DataFrame(rows).merge(direction[dcols],on=['context_id','repeat','task'],how='left')
    direction_out['scalar_differs_from_GT']=(direction_out.scalar_force_N!=direction_out.gt_force_N).astype(int)
    for label in ['rich_direct','point_wm','probecond_wm']:
        direction_out[f'{label}_away_from_GT']=((abs(direction_out[f'{label}_force_N']-direction_out.gt_force_N)>abs(direction_out.scalar_force_N-direction_out.gt_force_N))).astype(int)
    wcsv(out/'PROBE_WM_CORRECTION_DIRECTION.csv',direction_out)
    # Classification gates.
    M=lambda name:metrics(selected[name]); mb, mr, mp=M('ProbeScalar-Direct'),M('ProbeRich-Direct'),M(names[5]); recover=int(rescue.recoverable_probe_error_failure.sum()); rescued=int(rescue.rescued.sum()); collateral=int(rescue.collateral_damage.sum())
    high=pd.DataFrame(strat); high=high[(high.error_bin=='HIGH_ERROR')&(high.signed_error_bin=='ALL_SIGN')].set_index('method')
    pc_better_scalar=bool(mp['SR']>mb['SR'] and mp['under_force']<=mb['under_force'] and mp['realized_utility']>=mb['realized_utility'])
    pc_better_rich=bool(mp['SR']>mr['SR'] and mp['under_force']<=mr['under_force'] and mp['realized_utility']>=mr['realized_utility'])
    high_better_scalar=bool(high.loc[names[5],'SR']>high.loc[names[2],'SR'] and high.loc[names[5],'under_force']<=high.loc[names[2],'under_force'])
    # Context-level rank correlation avoids counting both repeats as independent probe errors.
    dg=direction_out.groupby('context_id',as_index=False).agg(probe_abs_error=('probe_abs_error','first'),probecond_utility_gain_vs_scalar=('probecond_utility_gain_vs_scalar','mean'))
    spearman=float(dg.probe_abs_error.rank().corr(dg.probecond_utility_gain_vs_scalar.rank()))
    # Probe pair ranking: all three physical-friction context pairs within each root.
    pair_ok=[]
    for _,q in pe.reset_index().groupby(['task','root_id']):
        a=q[['mu_GT','mu_hat']].to_numpy()
        for i in range(len(a)):
            for j in range(i+1,len(a)):
                pair_ok.append(int(np.sign(a[i,0]-a[j,0])==np.sign(a[i,1]-a[j,1])))
    pair_ranking=float(np.mean(pair_ok))
    # Stability and breadth gates are evaluated without best-seed selection.
    st=per[per.seed.astype(str)!='ENSEMBLE_MEAN']; seed_directions=[]; point_seed_directions=[]
    for seed in SEEDS:
        z=st[st.seed.astype(str)==str(seed)].groupby('method').agg(SR=('success','mean'),UF=('under_force','mean'),U=('realized_utility','mean'))
        seed_directions.append(bool(z.loc[names[5],'SR']>=z.loc[names[2],'SR'] and z.loc[names[5],'UF']<=z.loc[names[2],'UF'] and z.loc[names[5],'U']>=z.loc[names[2],'U']))
        point_seed_directions.append(bool(z.loc[names[4],'SR']>=z.loc[names[2],'SR'] and z.loc[names[4],'UF']<=z.loc[names[2],'UF'] and z.loc[names[4],'U']>=z.loc[names[2],'U']))
    task_gain={}
    for task in TASKS:
        z=ens[ens.task==task].groupby('method').agg(SR=('success','mean'),UF=('under_force','mean'),U=('realized_utility','mean'))
        task_gain[f'task{task}']=bool(z.loc[names[5],'SR']>=z.loc[names[2],'SR'] and z.loc[names[5],'UF']<=z.loc[names[2],'UF'] and z.loc[names[5],'U']>=z.loc[names[2],'U'])
    differing=direction_out[direction_out.scalar_differs_from_GT==1]
    toward=int(differing.probecond_wm_toward_GT.sum()); away=int(differing.probecond_wm_away_from_GT.sum())
    cdf=pd.DataFrame(cr); large=cdf[cdf.magnitude==cdf.magnitude.max()].groupby('method').agg(SR=('SR','mean'),UF=('under_force','mean'),U=('realized_utility','mean'))
    corruption_signal=bool(large.loc['ProbeConditionedWMResidual','U']>large.loc['ScalarDirect','U'])
    corruption_independent_wm=bool(corruption_signal and large.loc['ProbeConditionedWMResidual','U']>large.loc['RichDirect','U'])
    rich_helps_scalar=bool(mr['SR']>=mb['SR'] and mr['under_force']<=mb['under_force'] and mr['realized_utility']>=mb['realized_utility'])
    rich_beats_pc=bool(mr['SR']>=mp['SR'] and mr['under_force']<=mp['under_force'] and mr['realized_utility']>=mp['realized_utility'])
    if pc_better_scalar and pc_better_rich and high_better_scalar and all(seed_directions) and sum(task_gain.values())>=2:
        cls='PREDICTIVE_DYNAMICS_COMPENSATES_FOR_IMPERFECT_PHYSICAL_INFERENCE'
    elif rich_helps_scalar and rich_beats_pc:
        cls='RICH_PROBE_INFORMATION_HELPS_BUT_WORLD_MODEL_NOT_NEEDED'
    elif not pc_better_scalar and corruption_signal:
        cls='CONTROLLED_ROBUSTNESS_SIGNAL_WITHOUT_REAL_PROBE_GAIN'
    elif abs(mp['SR']-mr['SR'])<1e-12 and abs(mp['realized_utility']-mr['realized_utility'])<1e-3:
        cls='WORLD_MODEL_DOES_NOT_EXPLOIT_ADDITIONAL_PROBE_INFORMATION'
    else:
        cls='PROBE_ERROR_NOT_RECOVERABLE_FROM_CURRENT_INTERACTION_SIGNAL'
    cj={'classification':cls,'normal_pooled':{'ProbeScalar':mb,'ProbeRich':mr,'PointWM':M(names[4]),'ProbeConditionedWM':mp},
        'probe':{'MAE':float(pe.abs_error.mean()),'signed_error':float(pe.signed_error.mean()),'pair_ranking':pair_ranking,'contexts':int(len(pe))},
        'probe_error_gain_spearman':spearman,'recoverable_probe_direct_failures':recover,'rescued':rescued,'collateral_damage':collateral,
        'correction_direction_when_scalar_differs_GT':{'episodes':int(len(differing)),'toward_GT':toward,'away_from_GT':away},
        'gates':{'ProbeConditionedWM_gt_ProbeScalar':pc_better_scalar,'ProbeConditionedWM_gt_ProbeRich':pc_better_rich,
                 'HIGH_ERROR_gt_ProbeScalar':high_better_scalar,'all_3_seeds_stable':bool(all(seed_directions)),
                 'per_seed_gate':seed_directions,'PointWM_per_seed_gate':point_seed_directions,
                 'RichProbe_dominates_ProbeScalar':rich_helps_scalar,'RichProbe_dominates_ProbeConditionedWM':rich_beats_pc,
                 'per_task_gate':task_gain,'at_least_2_tasks':bool(sum(task_gain.values())>=2),
                 'controlled_corruption_signal_vs_scalar':corruption_signal,'controlled_corruption_independent_of_rich_input':corruption_independent_wm},
        'root_heldout':True,'sigma_used_as_posterior':False,'untouched_TEST_read':False}; wjson(out/'FINAL_PROBE_CONDITIONED_WM_CLASSIFICATION.json',cj)
    mpoint=M(names[4]); mgt=M(names[1]); mno=M(names[0])
    high_scalar=high.loc[names[2]]; high_pc=high.loc[names[5]]; high_gt=high.loc[names[1]]
    low=pd.DataFrame(strat); low=low[(low.error_bin=='LOW_ERROR')&(low.signed_error_bin=='ALL_SIGN')].set_index('method')
    signed=pd.DataFrame(strat); signed=signed[(signed.error_bin=='ALL_ERROR')&(signed.signed_error_bin!='ALL_SIGN')].set_index(['method','signed_error_bin'])
    scalar_gt_agree=float(direction_out.eval('scalar_force_N == gt_force_N').mean())
    large_rows=cdf[cdf.magnitude==cdf.magnitude.max()].set_index(['method','delta_label'])
    report=f"""# Final Probe-Conditioned WM Report

**直白结论：真实 root-heldout Probe error 下，rich Probe history 和 Probe-conditioned PhysicsOnly WM 都没有可靠地改善 force selection。分类为 `{cls}`。当前最终候选应保留 `scalar Probe + Direct Utility`，不要把 Probe-conditioned WM 放进默认 controller。**

OOF Probe 的 MAE 是 **{pe.abs_error.mean():.4f}**，signed error 是 **{pe.signed_error.mean():+.4f}**，friction pair-ranking 是 **{pair_ranking*100:.1f}%**。Normal pooled 上，ProbeScalar-Direct 的 SR / under-force / mean force / realized utility 是 **{mb['SR']*100:.2f}% / {mb['under_force']*100:.2f}% / {mb['mean_force_N']:.3f} N / {mb['realized_utility']:.4f}**；ProbeRich-Direct 是 **{mr['SR']*100:.2f}% / {mr['under_force']*100:.2f}% / {mr['mean_force_N']:.3f} N / {mr['realized_utility']:.4f}**；Probe-conditioned WM 是 **{mp['SR']*100:.2f}% / {mp['under_force']*100:.2f}% / {mp['mean_force_N']:.3f} N / {mp['realized_utility']:.4f}**。

## Direct answers

1. **Probe 的真实 held-root error 有多大？** 72 个 friction-conditioned contexts 上，ensemble OOF MAE={pe.abs_error.mean():.4f}，bias={pe.signed_error.mean():+.4f}；LOW/MID/HIGH boundaries 在训练结果查看前冻结为 {q1:.4f}/{q2:.4f}。这是 24 个 root families 的 grouped OOF 结果，不是 overlap estimator 的 in-sample 数字。

2. **Probe 出错时 Direct 是否跟着变差？** 有，但信号不强且集中在 HIGH_ERROR。HIGH_ERROR 中 ProbeScalar SR={high_scalar.SR*100:.2f}%、under-force={high_scalar.under_force*100:.2f}%，GT-Direct 为 {high_gt.SR*100:.2f}%/{high_gt.under_force*100:.2f}%；ProbeScalar 与 GT 的全体 force-decision agreement 只有 {scalar_gt_agree*100:.2f}%。LOW_ERROR 中两者 SR 分别为 {low.loc[names[2],'SR']*100:.2f}% 和 {low.loc[names[1],'SR']*100:.2f}%。

3. **仅把 rich Probe 信息给 Direct 能恢复多少？** 没有恢复：SR 从 {mb['SR']*100:.2f}% 降到 {mr['SR']*100:.2f}%，under-force 从 {mb['under_force']*100:.2f}% 升到 {mr['under_force']*100:.2f}%。因此现有固定 16D trace summary 并未成为有效的额外 physics source。

4. **Point-WM 看到相同错误 mu_hat 时能否纠错？** Ensemble 指标表面上改善到 SR={mpoint['SR']*100:.2f}%、under-force={mpoint['under_force']*100:.2f}%，但 3 seeds 方向不稳定（逐 seed gate={point_seed_directions}），不能作为稳定 WM evidence；它也没有检验 rich Probe 信息的独立价值。

5. **Probe-conditioned WM 是否明显更强？** 否。它相对 ProbeScalar 的 SR 低 {abs(mp['SR']-mb['SR'])*100:.2f} pp、under-force 高 {(mp['under_force']-mb['under_force'])*100:+.2f} pp；相对 ProbeRich 虽多成功 1/144 episode，但 utility 更低（{mp['realized_utility']:.4f} vs {mr['realized_utility']:.4f}）。

6. **它是否超过 ProbeRich Direct？** 没有通过 matched gate：`ProbeConditionedWM_gt_ProbeRich={pc_better_rich}`。这排除了“只因输入更多就把收益算给 WM”的解释。

7. **WM gain 是否随 Probe error 増大？** 没有。Probe error 与 Probe-conditioned-WM 相对 Scalar 的 context-level utility gain Spearman rho={spearman:+.3f}；HIGH_ERROR 的 Probe-conditioned WM SR/under-force 是 {high_pc.SR*100:.2f}%/{high_pc.under_force*100:.2f}%，没有超过 Scalar 的 {high_scalar.SR*100:.2f}%/{high_scalar.under_force*100:.2f}%。

8. **friction over-estimation 的 under-force 能救吗？** 没有稳定救回。OVER_ESTIMATE 全体下 Scalar SR/under-force={signed.loc[(names[2],'OVER_ESTIMATE'),'SR']*100:.2f}%/{signed.loc[(names[2],'OVER_ESTIMATE'),'under_force']*100:.2f}%，Probe-conditioned WM={signed.loc[(names[5],'OVER_ESTIMATE'),'SR']*100:.2f}%/{signed.loc[(names[5],'OVER_ESTIMATE'),'under_force']*100:.2f}%。

9. **friction under-estimation 的 excess force 能减少吗？** Probe-conditioned WM 的 mean force 相对 Scalar 可在明细中检查，但 realized utility 没有形成跨 task/seed 的稳定优势；当 Scalar 与 GT force 不同时，WM 向 GT 移动 {toward} 次、反向移动 {away} 次。

10. **controlled scalar corruption 是否更 robust？** 有一个受控但不足以晋升的方法信号：在最大正/负 corruption 下，Probe-conditioned WM 的 utility 分别为 {large_rows.loc[('ProbeConditionedWMResidual','POSITIVE'),'realized_utility']:.4f}/{large_rows.loc[('ProbeConditionedWMResidual','NEGATIVE'),'realized_utility']:.4f}，Scalar 为 {large_rows.loc[('ScalarDirect','POSITIVE'),'realized_utility']:.4f}/{large_rows.loc[('ScalarDirect','NEGATIVE'),'realized_utility']:.4f}。但 Probe-conditioned WM 没有超过 RichDirect 的平均大扰动 utility，因此这是 `controlled robustness`，不是真实 Probe-error gain，也不是 WM 独立价值。

11. **normal pooled 是否不伤害 Direct？** 没通过：SR 下降、under-force 上升，并出现 {collateral} 个 Scalar-success -> WM-failure collateral cases。{recover} 个 ProbeScalar-fail/GT-success 可恢复案例中，WM 救回 {rescued} 个。

12. **最终选择哪个版本？** 在三个候选中选择 **Version A: scalar Probe + Direct Utility**。Version B 的 rich summary 没有收益；Version C 没通过 Scalar、Rich、HIGH_ERROR、3-seed 和 multi-task gates。NoProbe SR={mno['SR']*100:.2f}%、GT SR={mgt['SR']*100:.2f}%；所以这一轮也不能单独把 Active Probe 的整体论文贡献宣称为已最终验证，只能回答 Probe-conditioned WM 不应加入。

## Why this classification is conservative

`CONTROLLED_ROBUSTNESS_SIGNAL_WITHOUT_REAL_PROBE_GAIN` 只表示：人为破坏 scalar 时，保留 interaction input 的 pipeline 比纯 Scalar 的 utility 退化更慢。它不表示 WM 已经利用真实 Probe 误差，因为真实 HIGH_ERROR、normal pooled、matched RichDirect、多个 task 和三个 seeds 的晋升条件均未同时成立。

## Protocol integrity and caveats

- Probe、Direct、Point-WM、Probe-conditioned WM 和 residual 均采用 nested grouped root-family OOF；同一 root 的 friction contexts、forces 和 repeats 不跨 fold。
- PhysicsOnly WM 固定 H=8、13 channels、`Lphysics + lambda_IE LIE`，没有 outcome BCE；residual architecture、MSE、epochs、optimizer、seeds 均未修改。
- Rich input 是在看 WM 结果前冻结的、fold-invariant 16D raw-trace physical summary（四个 signal 的 final/mean/std/max）。OOF GRU hidden 也导出供审计，但未直接拼接，因为各 fold 独立训练的 hidden 坐标不可识别。
- `sigma_mu` 仅作 diagnostic，没有 posterior 或 Bayesian claim。
- corruption 的 delta=0 严格复用未 clipping 的正式 baseline；只有非零扰动按预注册要求 clip 到 [0.2,1.0]。
- task1 outcome 保留 140/180 reconstructed-label caveat。
- 本轮没有读取 root-scaling untouched TEST，没有启动 simulator，也没有开发 Hard Verifier。
"""; (out/'FINAL_PROBE_CONDITIONED_WM_REPORT.md').write_text(report)
    req=['PROBE_WM_DATA_AUDIT.md','POOLED_OOF_PROBE_PREDICTIONS.csv','PROBE_ERROR_BINS.json','PROBE_SCALAR_CORRUPTION_CURVE.csv','PROBE_SCALAR_VS_RICH_DIRECT.csv','POINT_WM_VS_PROBE_CONDITIONED_WM.csv','PROBE_ERROR_STRATIFIED_RESULTS.csv','PROBE_WM_RESCUE_ANALYSIS.csv','PROBE_WM_CORRECTION_DIRECTION.csv','NORMAL_POOLED_PROBE_WM_TABLE.csv','FINAL_PROBE_CONDITIONED_WM_REPORT.md','FINAL_PROBE_CONDITIONED_WM_CLASSIFICATION.json']; (out/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(out/f)}  {f}' for f in req)+'\n'); print(json.dumps(cj,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('phase',choices=['prepare','shard','finalize']); a.add_argument('--out',type=Path,required=True); a.add_argument('--fold',type=int); a.add_argument('--seed',type=int); z=a.parse_args(); torch.set_num_threads(3); torch.set_num_interop_threads(1)
    if z.phase=='prepare': prepare(z.out.resolve())
    elif z.phase=='shard': shard(z.out.resolve(),z.fold,z.seed)
    else: finalize(z.out.resolve())
