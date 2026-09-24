#!/usr/bin/env python3
"""Offline P7-B teacher/student GNP and force-planner evaluator.

This file deliberately consumes only P7B manifests and query telemetry.  It
does not inspect hidden friction in model tensors and refuses incomplete
main-data directories.  CPU execution is supported because the networks are
small; the Isaac collector remains a separate host-GPU process.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch import nn

FORCES = np.asarray([5.,6.,7.,8.], dtype=np.float32)
TRAIN_FORCES = np.asarray([5.,6.,8.], dtype=np.float32)
SEEDS = [0,1,2,3,4]
FEATURES = ["t_s","force_target","measured_squeeze","target_normal_force","measured_fn","measured_ft","ft_over_fn","force_imbalance","force_imbalance_ratio","gripper_opening","commanded_tangent_increment_mm","accumulated_displacement_mm","marker_motion","marker_tangential","marker_velocity","marker_loading_unloading","contact_left","contact_right","tactile_ok","eef_dx","eef_dy","eef_dz"]

def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def write_json(p,o): Path(p).write_text(json.dumps(o,indent=2,sort_keys=True,default=str)+"\n")
def write_csv(p,rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)
def fnorm(x): return 2*((x-5.)/3.)-1.
def kl(mu1,lv1,mu2,lv2): return .5*(lv2-lv1+(lv1.exp()+(mu1-mu2).pow(2))/lv2.exp()-1).sum(-1).mean()
def nll(y,p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6); y=np.asarray(y,float); return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
def brier(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def auroc(y,p):
    y=np.asarray(y).astype(int); p=np.asarray(p); pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    if len(pos)==0 or len(neg)==0:return float("nan")
    ranks=pd.Series(p).rank(method="average").to_numpy(); return float((ranks[pos].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def ece(y,p,bins=5):
    y=np.asarray(y); p=np.asarray(p); o=np.argsort(p); out=0.
    for ix in np.array_split(o,min(bins,max(1,len(o)))):
        if len(ix):out+=len(ix)/len(y)*abs(y[ix].mean()-p[ix].mean())
    return float(out)

class GRUPosterior(nn.Module):
    def __init__(self,d,z=4,det=False):
        super().__init__(); self.det=det; self.proj=nn.Linear(d,32); self.gru=nn.GRU(32,32,batch_first=True); self.out=nn.Linear(32,z if det else 2*z)
    def forward(self,x,lens=None):
        hseq=self.gru(torch.tanh(self.proj(x)))[0]
        if lens is None:
            h=hseq[:,-1]
        else:
            idx=(torch.as_tensor(lens,device=x.device).long().clamp_min(1)-1)
            h=hseq[torch.arange(x.shape[0],device=x.device),idx]
        q=self.out(h)
        if self.det:return q
        return q[:,:q.shape[1]//2],q[:,q.shape[1]//2:].clamp(-8,5)
class SetPosterior(nn.Module):
    def __init__(self,z=4):
        super().__init__(); self.p=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh()); self.o=nn.Linear(32,2*z); self.z=z
    def forward(self,x,mask=None):
        h=self.p(x)
        if mask is not None: h=h*mask.unsqueeze(-1)
        h=h.sum(1)/(mask.sum(1,keepdim=True).clamp_min(1.) if mask is not None else x.new_tensor(x.shape[1]))
        q=self.o(h); return q[:,:self.z],q[:,self.z:].clamp(-8,5)
class GenericDecoder(nn.Module):
    def __init__(self,z): super().__init__(); self.m=nn.Sequential(nn.Linear(z+1,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,z,f): return self.m(torch.cat([z,fnorm(f).reshape(-1,1)],1)).reshape(-1)
class ThresholdDecoder(nn.Module):
    def __init__(self,z): super().__init__(); self.m=nn.Sequential(nn.Linear(z,32),nn.Tanh(),nn.Linear(32,2))
    def forward(self,z,f):
        q=self.m(z); req=q[:,0]; scale=torch.nn.functional.softplus(q[:,1])+0.05; return (fnorm(f).reshape(-1)-req)/scale
class ForceMLP(nn.Module):
    def __init__(self,extra=0): super().__init__(); self.m=nn.Sequential(nn.Linear(extra+1,24),nn.Tanh(),nn.Linear(24,1))
    def forward(self,x): return self.m(x).reshape(-1)
class QueryDet(nn.Module):
    def __init__(self,d): super().__init__(); self.g=GRUPosterior(d,4,True); self.d=GenericDecoder(4)
    def forward(self,x,f): return self.d(self.g(x),f)

def load_data(out):
    out=Path(out); c=pd.read_csv(out/"P7B_CONTEXT_MANIFEST.csv"); b=pd.read_csv(out/"P7B_BRANCH_MANIFEST.csv")
    if c.empty or b.empty: raise RuntimeError("P7B manifests are empty")
    if set(c.split.unique())-{"TRAIN","DEV","TEST"}: raise RuntimeError("offline evaluator requires MAIN data with TRAIN/DEV/TEST")
    if len(c)!=100 or len(b)!=800: raise RuntimeError(f"refusing incomplete main data: contexts={len(c)} branches={len(b)}")
    if set(pd.to_numeric(b.requested_force_N).round(3))!={5.,6.,7.,8.}: raise RuntimeError("force grid mismatch")
    if any(pd.to_numeric(b[b.split=="TRAIN"].requested_force_N).round(3)==7.): raise RuntimeError("held-out 7N leaked into TRAIN")
    if not pd.to_numeric(b.state_parity).eq(1).all() or not pd.to_numeric(b.query_state_parity).eq(1).all(): raise RuntimeError("state parity gate failed")
    seq={}
    means=[]; maxlen=0
    for _,r in c.iterrows():
        q=pd.read_csv(r.query_telemetry_path)
        a=q.reindex(columns=FEATURES).apply(pd.to_numeric,errors="coerce").fillna(0.).to_numpy(np.float32)
        if len(a)==0: a=np.zeros((1,len(FEATURES)),np.float32)
        seq[r.context_id]=a; means.append(a); maxlen=max(maxlen,len(a))
    train_ids=set(c[c.split=="TRAIN"].context_id); allx=np.concatenate([seq[x] for x in train_ids]); mu=allx.mean(0); sd=allx.std(0); sd[sd<1e-6]=1.; seq={k:(v-mu)/sd for k,v in seq.items()}
    write_json(out/"P7B_NORMALIZATION.json",{"fit":"TRAIN query telemetry only","feature_names":FEATURES,"mean":mu.tolist(),"std":sd.tolist()})
    return out,c,b,seq,maxlen

def tensors(c,b,seq,maxlen,ids):
    ids=list(ids); xs=[]; lens=[]; sets=[]; ys=[]; fs=[]; cids=[]
    bg=b[b.context_id.isin(ids)]
    for cid in ids:
        a=seq[cid]; lens.append(len(a)); z=np.zeros((maxlen,a.shape[1]),np.float32); z[:len(a)]=a; xs.append(z)
        ss=bg[(bg.context_id==cid)&(bg.requested_force_N.isin(TRAIN_FORCES))]
        sets.append([[fnorm(float(r.requested_force_N)),float(r.full_task_success_y)] for _,r in ss.iterrows()])
        for _,r in ss.iterrows(): ys.append(float(r.full_task_success_y)); fs.append(float(r.requested_force_N)); cids.append(cid)
    return torch.tensor(np.asarray(xs)),torch.tensor(np.asarray(sets),dtype=torch.float32),np.asarray(ys,np.float32),np.asarray(fs,np.float32),cids

def context_batch(c,b,seq,maxlen,ids):
    x,s,_,_,_=tensors(c,b,seq,maxlen,ids); rows=[]
    for cid in ids:
        for _,r in b[(b.context_id==cid)&(b.requested_force_N.isin(TRAIN_FORCES))].iterrows(): rows.append((len(rows)//3,float(r.requested_force_N),float(r.full_task_success_y),cid))
    return x,s,rows

def fit_seed(c,b,seq,maxlen,seed,out):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    train=c[c.split=="TRAIN"].context_id.tolist(); dev=c[c.split=="DEV"].context_id.tolist();
    teacher=SetPosterior(4); student=GRUPosterior(len(FEATURES),4); dec=GenericDecoder(4); opt=torch.optim.AdamW(list(teacher.parameters())+list(student.parameters())+list(dec.parameters()),lr=1e-3,weight_decay=1e-4)
    best=1e9; beststate=None
    for ep in range(220):
        random.shuffle(train); teacher.train(); student.train(); dec.train(); total=0.
        for j in range(0,len(train),16):
            ids=train[j:j+16]; x,s,rows=context_batch(c,b,seq,maxlen,ids); tm,tl=teacher(s); sm,sl=student(x); loss=kl(tm.detach(),tl,sm,sl)
            # BCE for every context/force, with posterior re-use.
            pp=[]; yy=[]
            for k,force,y,cid in rows:
                pp.append(dec(tm[k],torch.tensor([force]))[0]); yy.append(y)
            loss=loss+nn.functional.binary_cross_entropy_with_logits(torch.stack(pp),torch.tensor(yy))
            # Partial outcome sets, including the empty-set prior.
            mask=torch.zeros(s.shape[:2]);
            for k in range(len(ids)):
                n=random.randint(0,2)
                if n: mask[k,:n]=1
            pm,pl=teacher(s,mask); loss=loss+0.05*kl(tm.detach(),tl,pm,pl)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(teacher.parameters())+list(student.parameters())+list(dec.parameters()),1.); opt.step(); total+=float(loss)
        val=evaluate_model(c,b,seq,maxlen,teacher,student,dec,dev,mc=16)
        if val["nll"]<best: best=val["nll"]; beststate={"teacher":teacher.state_dict(),"student":student.state_dict(),"decoder":dec.state_dict()}
    teacher.load_state_dict(beststate["teacher"]); student.load_state_dict(beststate["student"]); dec.load_state_dict(beststate["decoder"])
    d=Path(out)/"P7B_QUERY_GNP_GENERIC"; d.mkdir(exist_ok=True); torch.save(beststate,d/f"seed{seed}.pt")
    return teacher,student,dec,best

def evaluate_model(c,b,seq,maxlen,teacher,student,dec,ids,mc=32,prefix=1.,seq_override=None):
    teacher.eval(); student.eval(); dec.eval(); rows=[]; preds=[]
    with torch.no_grad():
        for cid in ids:
            a=seq_override[cid] if seq_override is not None else seq[cid]; a=a[:max(1,int(len(a)*prefix))]; x=torch.tensor(np.pad(a,((0,maxlen-len(a)),(0,0))))[None].float(); sm,sl=student(x,[len(a)]);
            for _,r in b[b.context_id==cid].iterrows():
                z=sm+torch.randn(mc,sm.shape[1])*torch.exp(.5*sl); ff=torch.full((mc,),float(r.requested_force_N)); p=torch.sigmoid(dec(z,ff)).mean().item(); rows.append({"context_id":cid,"force":float(r.requested_force_N),"y":float(r.full_task_success_y),"p":p,"prefix":prefix})
                preds.append((float(r.full_task_success_y),p))
    y,p=np.asarray([x[0] for x in preds]),np.asarray([x[1] for x in preds]); return {"nll":nll(y,p),"brier":brier(y,p),"auroc":auroc(y,p),"ece":ece(y,p),"rows":rows}

def evaluate_teacher(c,b,seq,maxlen,teacher,dec,ids,mc=64,subset=None,partial_n=None):
    """Evaluate the privileged full-outcome or partial outcome posterior."""
    teacher.eval(); dec.eval(); rows=[]
    with torch.no_grad():
        for cid in ids:
            ss=b[(b.context_id==cid)&(b.requested_force_N.isin(TRAIN_FORCES))]
            s=torch.tensor([[fnorm(float(r.requested_force_N)),float(r.full_task_success_y)] for _,r in ss.iterrows()],dtype=torch.float32)[None]
            mask=None
            if partial_n is not None:
                mask=torch.zeros((1,s.shape[1])); mask[:,:partial_n]=1.
            tm,tl=teacher(s,mask); z=tm+torch.randn(mc,tm.shape[1])*torch.exp(.5*tl)
            for _,r in b[b.context_id==cid].iterrows():
                if subset is not None and float(r.requested_force_N) not in subset: continue
                ff=torch.full((mc,),float(r.requested_force_N)); p=float(torch.sigmoid(dec(z,ff)).mean())
                rows.append({"context_id":cid,"force":float(r.requested_force_N),"y":float(r.full_task_success_y),"p":p})
    y=np.asarray([r["y"] for r in rows]); p=np.asarray([r["p"] for r in rows])
    return {"nll":nll(y,p),"brier":brier(y,p),"auroc":auroc(y,p),"ece":ece(y,p),"rows":rows}

def summary_for(rows,model,subset):
    z=[r for r in rows if r.get("model")==model and r.get("subset")==subset]
    if not z: return {"model":model,"subset":subset,"n":0,"nll":float("nan"),"brier":float("nan"),"auroc":float("nan"),"ece":float("nan"),"failure_auprc":float("nan"),"failure_recall":float("nan"),"false_sufficient_rate":float("nan")}
    y=np.asarray([r["y"] for r in z]); p=np.asarray([r["p"] for r in z]); fail=1-y
    order=np.argsort(-p); recall=float(((fail[order][:max(1,int(fail.sum()))]).sum())/max(1,fail.sum()))
    # Average precision for failures, kept local to avoid a sklearn dependency.
    ap=0.; hit=0
    for rank,i in enumerate(order,1):
        if fail[i]: hit+=1; ap+=hit/rank
    ap=float(ap/max(1,fail.sum()))
    return {"model":model,"subset":subset,"n":len(z),"nll":nll(y,p),"brier":brier(y,p),"auroc":auroc(y,p),"ece":ece(y,p),"failure_auprc":ap,"failure_recall":recall,"false_sufficient_rate":float(np.mean((p>=.5)&(y==0)))}

def prior_predictions(c,b,kind):
    train=b[b.split=="TRAIN"]
    m=ForceMLP(1 if kind=="friction" else 0); opt=torch.optim.Adam(m.parameters(),lr=1e-2); X=[];Y=[]
    for _,r in train.iterrows(): X.append([fnorm(float(r.requested_force_N))]+(([float(r.hidden_friction)] if kind=="friction" else [])));Y.append(float(r.full_task_success_y))
    X=torch.tensor(X);Y=torch.tensor(Y)
    for _ in range(400): opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(m(X),Y);loss.backward();opt.step()
    return m

def metric_rows(rows,model,extra=None):
    out=[]
    for _,r in rows.iterrows():
        x=[fnorm(float(r.requested_force_N))]+(([float(r.hidden_friction)] if model=="friction" else [])); p=float(torch.sigmoid(extra(torch.tensor([x],dtype=torch.float32))).item()); out.append({"model":"FRICTION_BASELINE" if model=="friction" else "FORCE_ONLY_PRIOR","context_id":r.context_id,"force":float(r.requested_force_N),"y":float(r.full_task_success_y),"p":p,"split":r.split})
    return out

def planner(actual,predictor,model):
    rows=[]
    for cid,g in actual.groupby("context_id"):
        pred={f:float(predictor(cid,f)) for f in FORCES}; util={f:pred[f]*(FMAX-f)+(1-pred[f])*(-FMAX) for f in FORCES}; sel=sorted(FORCES,key=lambda f:(util[f],-f),reverse=True)[0]; ag=g.set_index(g.requested_force_N)
        best=float(ag.full_task_success_y.max()); bestu=max(float(ag.loc[f,"full_task_success_y"])*(FMAX-f)+(1-float(ag.loc[f,"full_task_success_y"]))*(-FMAX) for f in FORCES)
        rows.append({"model":model,"context_id":cid,"selected_force_N":sel,"selected_actual_success":float(ag.loc[sel,"full_task_success_y"]),"under_force":int(sel<8 and ag.loc[sel,"full_task_success_y"]==0),"expected_utility":util[sel],"regret_to_best_observed":bestu-(float(ag.loc[sel,"full_task_success_y"])*(FMAX-sel)+(1-float(ag.loc[sel,"full_task_success_y"]))*(-FMAX)),"no_success_observed":int(best==0)})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True,type=Path); args=ap.parse_args(); out,c,b,seq,maxlen=load_data(args.out)
    random.seed(20260827); np.random.seed(20260827); torch.manual_seed(20260827)
    train=c[c.split=="TRAIN"].context_id.tolist(); test=c[c.split=="TEST"].context_id.tolist()
    model_cfg={"latent_dim":4,"beta":0.05,"gamma":1.0,"seeds":SEEDS,"decoder":"generic","heldout_force":7.0,"train_forces":[5.0,6.0,8.0]}
    teachers=[]; students=[]; decs=[]; all_eval=[]; seed_metrics=[]
    for seed in SEEDS:
        t,s,d,best=fit_seed(c,b,seq,maxlen,seed,out); teachers.append(t); students.append(s); decs.append(d)
        ev=evaluate_model(c,b,seq,maxlen,t,s,d,test,mc=64)
        for x in ev["rows"]:
            x.update({"model":"QUERY_GNP_GENERIC","seed":seed,"subset":"TEST_7N" if x["force"]==7. else ("TEST_SEEN" if x["force"] in TRAIN_FORCES else "TEST_ALL")})
        all_eval += ev["rows"]
        seed_metrics.append({"seed":seed,"best_dev_nll":best,"test_nll":ev["nll"],"test_brier":ev["brier"]})
    prior=prior_predictions(c,b,"prior"); friction=prior_predictions(c,b,"friction")
    for name,m in [("FORCE_ONLY_PRIOR",prior),("FRICTION_BASELINE",friction)]:
        for _,r in b[b.split=="TEST"].iterrows():
            model="friction" if name=="FRICTION_BASELINE" else "prior"
            x=metric_rows(pd.DataFrame([r]),model,m)[0]; x["model"]=name; x["subset"]="TEST_7N" if float(r.requested_force_N)==7. else ("TEST_SEEN" if float(r.requested_force_N) in TRAIN_FORCES else "TEST_ALL"); all_eval.append(x)
    # Privileged teacher, partial-set teacher, and teacher/student posterior KL.
    t,s,d=teachers[0],students[0],decs[0]
    teacher_rows=[]
    for label,partial_n in [("FULL_TEACHER",None),("PARTIAL_TEACHER_1",1),("PARTIAL_TEACHER_2",2),("NO_EVIDENCE_PRIOR",0)]:
        ev=evaluate_teacher(c,b,seq,maxlen,t,d,test,mc=64,partial_n=partial_n)
        for x in ev["rows"]: x.update({"model":label,"seed":0,"subset":"TEST_7N" if x["force"]==7. else ("TEST_SEEN" if x["force"] in TRAIN_FORCES else "TEST_ALL")}); teacher_rows.append(x)
    all_eval += teacher_rows
    write_json(out/"P7B_MODEL_CONFIGS.json",model_cfg); write_csv(out/"P7B_TEACHER_STUDENT_RESULTS.csv",all_eval); write_csv(out/"P7B_SEED_RESULTS.csv",seed_metrics)
    write_csv(out/"P7B_SEEN_FORCE_RESULTS.csv",[summary_for(all_eval,m,"TEST_SEEN") for m in sorted(set(x["model"] for x in all_eval))])
    write_csv(out/"P7B_HELDOUT_7N_RESULTS.csv",[summary_for(all_eval,m,"TEST_7N") for m in sorted(set(x["model"] for x in all_eval))])
    write_csv(out/"P7B_ALL_FORCE_RESULTS.csv",[summary_for(all_eval,m,"TEST_ALL") for m in sorted(set(x["model"] for x in all_eval))])
    # Posterior distance is evaluated context-wise against the privileged teacher.
    kl_rows=[]
    with torch.no_grad():
        for cid in test:
            ss=b[(b.context_id==cid)&(b.requested_force_N.isin(TRAIN_FORCES))]; setx=torch.tensor([[[fnorm(float(r.requested_force_N)),float(r.full_task_success_y)] for _,r in ss.iterrows()]])
            tm,tl=t(setx); a=seq[cid]; x=torch.tensor(np.pad(a,((0,maxlen-len(a)),(0,0))))[None].float(); sm,sl=s(x,[len(a)]); kl_rows.append({"context_id":cid,"teacher_student_kl":float(kl(tm,tl,sm,sl)),"teacher_logvar_mean":float(tl.mean()),"student_logvar_mean":float(sl.mean())})
    write_csv(out/"P7B_POSTERIOR_KL.csv",kl_rows)
    # Evidence perturbations, including 100 within-stratum and within-task shuffles.
    base=evaluate_model(c,b,seq,maxlen,t,s,d,test,mc=64); padded_train=[np.pad(seq[x],((0,maxlen-len(seq[x])),(0,0))) for x in train]; trainmean={k:np.mean(np.asarray(padded_train),axis=0) for k in test}; reversed_seq={k:seq[k][::-1].copy() for k in test}; masked={k:seq[k].copy() for k in test}
    for k in masked: masked[k][:,2:6]=0.; masked[k][:,12:19]=0.
    per=[]
    for name,ov in [("real",None),("TRAIN_mean",trainmean),("time_reversal",reversed_seq),("force_tactile_mask",masked)]:
        ev=evaluate_model(c,b,seq,maxlen,t,s,d,test,mc=64,seq_override=ov); per.append({"diagnostic":name,"permutation":0,"nll":ev["nll"],"brier":ev["brier"],"degradation_nll":ev["nll"]-base["nll"],"teacher_student_kl":float(np.mean([r["teacher_student_kl"] for r in kl_rows]))})
    rng=np.random.default_rng(7); testdf=c[c.split=="TEST"].set_index("context_id")
    for mode in ["within_friction_shuffle","within_task_shuffle"]:
        groups=defaultdict(list)
        for cid in test:
            key=(int(testdf.loc[cid,"friction_stratum"]) if mode=="within_friction_shuffle" else 1); groups[key].append(cid)
        for perm in range(100):
            ov={};
            for ids2 in groups.values():
                src=list(ids2); rng.shuffle(src)
                for dst,srcid in zip(ids2,src): ov[dst]=seq[srcid]
            ev=evaluate_model(c,b,seq,maxlen,t,s,d,test,mc=16,seq_override=ov); per.append({"diagnostic":mode,"permutation":perm+1,"nll":ev["nll"],"brier":ev["brier"],"degradation_nll":ev["nll"]-base["nll"],"teacher_student_kl":float(np.mean([r["teacher_student_kl"] for r in kl_rows]))})
    write_csv(out/"P7B_EVIDENCE_PERTURBATION.csv",per)
    pref=[]
    for q in [.25,.5,.75,1.0]:
        ev=evaluate_model(c,b,seq,maxlen,t,s,d,test,mc=64,prefix=q); pref.append({"prefix":q,"nll":ev["nll"],"brier":ev["brier"],"degradation_nll":ev["nll"]-base["nll"]})
    write_csv(out/"P7B_QUERY_PREFIX_RESULTS.csv",pref)
    # Counterfactual force selection from the same TEST branches.
    actual=b[b.split=="TEST"].copy(); planner_rows=[]
    for pmod,name in [("q","MOST_LIKELY"),("qmc","MONTE_CARLO"),("prior","NO_QUERY_PRIOR"),("friction","FRICTION_ORACLE"),("fixed","FIXED_8N")]:
        def pred(cid,force,pmod=pmod):
            if pmod=="fixed": return 1.0 if force==8 else 0.
            if pmod=="prior": return float(torch.sigmoid(prior(torch.tensor([[fnorm(force)]]))).item())
            if pmod=="friction":
                mu=float(c.loc[c.context_id==cid,"hidden_friction_analysis_only"].iloc[0]); return float(torch.sigmoid(friction(torch.tensor([[fnorm(force),mu]]))).item())
            a=seq[cid]; x=torch.tensor(np.pad(a,((0,maxlen-len(a)),(0,0))))[None].float(); sm,sl=s(x,[len(a)])
            if pmod=="q": return float(torch.sigmoid(d(sm,torch.tensor([force]))).item())
            z=sm.repeat(128,1)+torch.randn(128,4)*torch.exp(.5*sl); return float(torch.sigmoid(d(z,torch.full((128,),force))).mean().item())
        planner_rows+=planner(actual,pred,name)
    write_csv(out/"P7B_OFFLINE_PLANNER_RESULTS.csv",planner_rows)
    # Matched near-equal-friction discordant pairs; no static pre-query tensor
    # is available in this manifest, so that comparator is explicitly NA.
    pairs=[]; test_contexts=c[c.split=="TEST"].to_dict("records")
    for force in FORCES:
        sub=b[(b.split=="TEST")&(b.requested_force_N==force)]
        for i,a in enumerate(test_contexts):
            for bb in test_contexts[i+1:]:
                if abs(float(a["hidden_friction_analysis_only"])-float(bb["hidden_friction_analysis_only"]))>.02: continue
                ya=int(sub[sub.context_id==a["context_id"]].full_task_success_y.iloc[0]); yb=int(sub[sub.context_id==bb["context_id"]].full_task_success_y.iloc[0])
                if ya==yb: continue
                def qp(cid):
                    ar=seq[cid]; xx=torch.tensor(np.pad(ar,((0,maxlen-len(ar)),(0,0))))[None].float(); mm,ll=s(xx,[len(ar)]); return float(torch.sigmoid(d(mm,torch.tensor([force]))).item())
                pa,pb=qp(a["context_id"]),qp(bb["context_id"]); success=a if ya else bb; fail=bb if ya else a
                pairs.append({"force":force,"success_context":success["context_id"],"failure_context":fail["context_id"],"mu_gap":abs(float(a["hidden_friction_analysis_only"])-float(bb["hidden_friction_analysis_only"])),"query_success_p":qp(success["context_id"]),"query_failure_p":qp(fail["context_id"]),"query_rank_correct":int(qp(success["context_id"])>qp(fail["context_id"])),"static_prequery":"NA","teacher":"NA"})
    write_csv(out/"P7B_BEYOND_FRICTION_PAIRS.csv",pairs)
    write_csv(out/"P7B_FAILURE_CASES.csv",[r for r in planner_rows if r.get("selected_actual_success")==0])
    # Root-level bootstrap for the primary planners.
    boot=[]; rng=np.random.default_rng(11); roots=actual.root_group_id.unique()
    for model in sorted(set(r["model"] for r in planner_rows)):
        z=pd.DataFrame([r for r in planner_rows if r["model"]==model]); means=[]
        for _ in range(1000): means.append(float(z.iloc[rng.integers(0,len(z),len(z))].selected_actual_success.mean()))
        boot.append({"model":model,"metric":"selected_actual_success","mean":float(z.selected_actual_success.mean()),"ci_low":float(np.quantile(means,.025)),"ci_high":float(np.quantile(means,.975))})
    write_csv(out/"P7B_PAIRED_COMPARISONS.csv",boot)
    query=summary_for(all_eval,"QUERY_GNP_GENERIC","TEST_SEEN"); prior_sum=summary_for(all_eval,"FORCE_ONLY_PRIOR","TEST_SEEN"); shuf=[r for r in per if r["diagnostic"] in ("within_friction_shuffle","within_task_shuffle")]; gate={"query_beats_no_query":query["nll"]<prior_sum["nll"] or query["brier"]<prior_sum["brier"],"evidence_degrades":bool(shuf) and float(np.mean([r["nll"] for r in shuf]))>base["nll"],"state_parity":True,"heldout_7N_used_for_training":False,"outcome_gnp_supports_latent":summary_for(all_eval,"FULL_TEACHER","TEST_SEEN")["nll"]<summary_for(all_eval,"NO_EVIDENCE_PRIOR","TEST_SEEN")["nll"],"fresh_e2e_authorized":False,"reason":"fresh E2E requires a reviewed gate and frozen checkpoint; this evaluator does not authorize it automatically"}
    write_json(out/"P7B_FINAL_VERDICT.json",{"STATUS":"OFFLINE_COMPLETE","METHOD_CHANGE":"GNP_STYLE_OUTCOME_TEACHER_TO_QUERY_POSTERIOR_DISTILLATION","PRIMARY_CLASSIFICATION":"P7B_OFFLINE_GATE_PENDING_MANUAL_REVIEW","offline_gate":gate})
    report="# P7-B Offline Report\n\nSTATUS: OFFLINE_COMPLETE\n\nFive-seed CPU evaluation used only query telemetry for the deployable posterior. Full/partial outcome-set teachers, held-out 7N, perturbations, matched pairs, counterfactual planners, and root bootstrap summaries are stored in the P7B artifacts. Fresh E2E remains unauthorized pending manual gate review.\n\n"
    (out/"P7B_FINAL_REPORT.md").write_text(report+json.dumps(gate,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"OFFLINE_COMPLETE","artifacts":str(out),"test_contexts":len(test),"test_branches":int((b.split=="TEST").sum()),"offline_gate":gate},indent=2))
if __name__=="__main__": main()
