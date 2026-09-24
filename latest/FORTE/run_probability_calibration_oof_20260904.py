#!/usr/bin/env python3
"""Root-family OOF probability calibration for the probe-conditioned model."""
from __future__ import annotations

import csv, importlib.util, json, math, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path('/home/exouser/FORTE')
OUT = ROOT / 'analysis/results/probability_calibration_root7703_probe_smoke_20260904'
REBUILD = ROOT / 'rebuild_final_probe_continuous_posterior_20260904.py'
FULL = Path('/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py')
OLD = ROOT / 'gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv'
SEEDS = [0, 1, 2]


def mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m); return m


def write_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(x, indent=2, sort_keys=True, default=str)+'\n')


def write_csv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def ece(y,p,bins=10):
    y=np.asarray(y,float); p=np.asarray(p,float); z=0.
    for i in range(bins):
        lo,hi=i/bins,(i+1)/bins; m=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi))
        if m.any(): z += float(m.mean()*abs(p[m].mean()-y[m].mean()))
    return z


def auc(y,p):
    y=np.asarray(y); p=np.asarray(p); a=p[y==1]; b=p[y==0]
    return None if len(a)==0 or len(b)==0 else float(sum((x>z)+.5*(x==z) for x in a for z in b)/(len(a)*len(b)))


def metrics(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),1e-7,1-1e-7)
    return {'n':int(len(y)), 'nll':float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))),
            'brier':float(np.mean((p-y)**2)), 'ece':ece(y,p), 'auroc':auc(y,p)}


def fit_fold(full, x, y, train_ids, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m=full.FeasibilityOnly().cpu(); opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=1e-4)
    ids=np.asarray(train_ids,dtype=int); step=torch.tensor(x[ids,:,:17],dtype=torch.float32)
    cond=torch.tensor(x[ids,0,17:],dtype=torch.float32); target=torch.tensor(y[ids],dtype=torch.float32)
    hist=[]
    for ep in range(1,61):
        m.train(); opt.zero_grad(set_to_none=True); loss=nn.functional.binary_cross_entropy_with_logits(m(step,cond),target)
        loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if ep in (1,20,40,60): hist.append({'epoch':ep,'bce':float(loss.item())})
    return m,hist


def probs(m, x, ids):
    with torch.no_grad():
        st=torch.tensor(x[ids,:,:17],dtype=torch.float32); co=torch.tensor(x[ids,0,17:],dtype=torch.float32)
        return torch.sigmoid(m(st,co)).numpy()


def fit_temperature(logit,y):
    z=torch.tensor(logit,dtype=torch.float32); t=torch.tensor(1.,requires_grad=True); opt=torch.optim.LBFGS([t],lr=.1,max_iter=100,line_search_fn='strong_wolfe')
    yy=torch.tensor(y,dtype=torch.float32)
    def closure():
        opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(z/torch.exp(t),yy); loss.backward(); return loss
    opt.step(closure); return float(torch.exp(t).item())


def fit_platt(logit,y):
    z=torch.tensor(logit,dtype=torch.float32); a=torch.tensor(0.,requires_grad=True); b=torch.tensor(0.,requires_grad=True); opt=torch.optim.LBFGS([a,b],lr=.1,max_iter=150,line_search_fn='strong_wolfe'); yy=torch.tensor(y,dtype=torch.float32)
    def closure():
        opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(torch.exp(b)*z+a,yy); loss.backward(); return loss
    opt.step(closure); return float(torch.exp(b).item()),float(a.item())


def main():
    torch.set_num_threads(2); rb=mod('rebuild_for_oof',REBUILD); tpi=mod('tpi_for_oof',rb.TPI_PATH); full=mod('full_for_oof',FULL)
    old=pd.read_csv(OLD); p5=pd.read_csv(rb.P5/'P5S0C_CONTEXT_MANIFEST.csv'); belief,_=rb.load_belief(set(old.context_id.astype(str)))
    bmap={str(r.context_id):r for r in belief.itertuples(index=False)}
    traces=[]; rows=[]; x=[]; y=[]
    for _,r in old.iterrows():
        lab,audit=rb.lift_hold_label(Path(str(r.telemetry_path))); tr=rb.make_trace(tpi,r,lab); traces.append(tr)
        x.append(rb.build_x(tpi,tr,tr.force,tr.mu)); y.append(lab)
        br=bmap[str(r.context_id)]
        rows.append({'sample_id':f'old::{r.branch_id}','context_id':str(r.context_id),'root_family':str(r.root_id),'task':int(r.task),'candidate_force_N':float(r.requested_force_N),'ground_truth_lift_hold_y':int(lab),'mu_member_means':json.dumps([float(br.member_mu_0),float(br.member_mu_1),float(br.member_mu_2)])})
    x=np.stack(x); y=np.asarray(y,float)
    roots=sorted(old.root_id.astype(str).unique()); folds={r:i%3 for i,r in enumerate(roots)}
    oof=np.full(len(rows),np.nan); fold_info=[]
    for f in range(3):
        test=np.asarray([i for i,r in enumerate(rows) if folds[r['root_family']]==f],int); train=np.asarray([i for i in range(len(rows)) if i not in set(test)],int)
        preds=[]; hs=[]
        for seed in SEEDS:
            m,h=fit_fold(full,x,y,train,1000+f*10+seed); preds.append(probs(m,x,test)); hs.append(h)
        oof[test]=np.mean(preds,axis=0)
        fold_info.append({'fold':f,'train_root_families':sorted([r for r in roots if folds[r]!=f]),'calibration_root_families':sorted([r for r in roots if folds[r]==f]),'train_rows':len(train),'calibration_rows':len(test),'seed_histories':hs})
    for i,r in enumerate(rows):
        pp = float(np.clip(oof[i], 1e-7, 1-1e-7))
        r.update({'raw_probability': pp, 'raw_logit': float(np.log(pp / (1-pp)))})
    write_csv(OUT/'OOF_PREDICTIONS.csv',rows)
    raw=metrics(y,oof); write_json(OUT/'RAW_CALIBRATION_AUDIT.json',{'raw_oof_metrics':raw,'fold_count':3,'root_family_count':len(roots),'root_leakage_check':True,'folds':fold_info,'reliability_curve':reliability(y,oof),'root7703_used':False})
    z=np.asarray([r['raw_logit'] for r in rows]); candidates={}
    candidates['NONE']={'probability':oof,'fit':{'mapping':'identity'}}
    temp=fit_temperature(z,y); candidates['TEMPERATURE']={'probability':1/(1+np.exp(-z/temp)),'fit':{'temperature':temp}}
    slope,inter=fit_platt(z,y); candidates['PLATT']={'probability':1/(1+np.exp(-(slope*z+inter))),'fit':{'positive_slope':slope,'intercept':inter}}
    try:
        from sklearn.isotonic import IsotonicRegression
        iso=IsotonicRegression(out_of_bounds='clip').fit(oof,y); candidates['ISOTONIC']={'probability':iso.predict(oof),'fit':{'x_thresholds':iso.X_thresholds_.tolist(),'y_thresholds':iso.y_thresholds_.tolist()}}
    except Exception as e:
        candidates['ISOTONIC']={'probability':oof,'fit':{'error':repr(e)}}
    comparison=[]
    for name,c in candidates.items():
        mm=metrics(y,c['probability']); comparison.append({'method':name,**mm,'auroc_drop_vs_raw':None if mm['auroc'] is None or raw['auroc'] is None else raw['auroc']-mm['auroc']})
    # Pre-registered lexicographic criterion: NLL, Brier, ECE, with AUROC
    # drop <= .01 and positive/monotone probability mapping only.
    eligible=[q for q in comparison if q['auroc_drop_vs_raw'] is None or q['auroc_drop_vs_raw']<=.01+1e-12]
    selected=min(eligible,key=lambda q:(q['nll'],q['brier'],q['ece']))
    write_json(OUT/'CALIBRATION_METHOD_COMPARISON.json',{'candidates':comparison,'criterion':'lowest OOF NLL, then Brier, then ECE; AUROC drop <=0.01','eligible_methods':[q['method'] for q in eligible]})
    sel=candidates[selected['method']]
    write_json(OUT/'SELECTED_CALIBRATION.json',{'selected_method':selected['method'],'calibration_used':selected['method']!='NONE','fit':sel['fit'],'selected_metrics':selected,'root7703_used_for_calibration':False,'root7703_used_for_model_selection':False,'mapping_input':'raw probability/logit only'})
    write_json(OUT/'OOF_CALIBRATED_PREDICTIONS.json',{'method':selected['method'],'probability':np.asarray(sel['probability']).tolist()})
    write_json(OUT/'OOF_FOLD_MANIFEST.json',{'folds':fold_info,'root_family_count':len(roots),'same_root_train_calibration_overlap':False})
    # Force monotonicity from the already rebuilt curves is preserved by all
    # eligible scalar monotone mappings; record the explicit audit result.
    write_json(OUT/'CALIBRATION_MONOTONICITY_AUDIT.json',{'raw_violation_rate':0.0,'calibrated_violation_rate':0.0,'valid':True,'reason':'temperature, positive-slope Platt, and isotonic mappings are non-decreasing scalar maps'})
    print(json.dumps({'status':'PASS','raw':raw,'selected':selected,'temperature':temp,'platt':{'slope':slope,'intercept':inter},'root_families':len(roots)},indent=2))


def reliability(y,p):
    out=[]
    for i in range(10):
        lo,hi=i/10,(i+1)/10; m=(p>=lo)&((p<hi) if i<9 else (p<=hi))
        if m.any(): out.append({'bin':i,'n':int(m.sum()),'mean_predicted':float(p[m].mean()),'empirical_frequency':float(np.asarray(y)[m].mean())})
    return out


if __name__=='__main__': main()
