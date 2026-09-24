#!/usr/bin/env python3
"""Evaluate repeated-force models on context-level probabilities."""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from evaluate_af_taskforms_inference import load_models, belief_support
from evaluate_af_taskforms_forcegrid648 import DENSE_FORCES, force_curve
from train_af_taskforms_models import task_onehot

FORCES=(3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0)

def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def groups(rows,split):
    g=defaultdict(list)
    for r in rows:
        if r['valid'] and r['split']==split: g[(r['task'],int(r['root_slot']),float(r['friction']))].append(r)
    return g
def pmean(rs): return float(np.mean([bool(r['evidence']['success']) for r in rs])) if rs else float('nan')
def choose(curve,t):
    for f,p in zip(DENSE_FORCES,curve):
        if p>=t:return f
    return DENSE_FORCES[-1]
def run(gs,bm,fm,norm,t):
    decisions=[]
    for k,rs in sorted(gs.items()):
        ref=sorted(rs,key=lambda r: (float(r['force_n']),int(r.get('repeat_index',0))))[0]
        # The dense curve helper expects one matched record and uses the frozen architecture.
        raw,proj,post=force_curve(bm,fm,norm,ref)
        sel=choose(proj,t); outcome={f:pmean([r for r in rs if abs(float(r['force_n'])-f)<1e-8]) for f in FORCES}
        selected=float(sel)
        # Qualification data are only observed at the endpoint forces.  Do not
        # pool neighboring force bins when the dense model selects an
        # unobserved value; use the nearest observed endpoint for the empirical
        # diagnostic and record the actual selected force separately.
        observed_force = min(FORCES, key=lambda f: abs(float(f) - selected))
        selected_p=pmean([r for r in rs if abs(float(r['force_n'])-observed_force)<1e-8])
        decisions.append({'task':k[0],'root_slot':k[1],'friction':k[2],'selected_force_n':selected,'selected_success_probability':selected_p,'empirical_by_force':{str(f):outcome[f] for f in FORCES},'fixed_3_probability':outcome[3.0],'fixed_4_25_probability':outcome[4.25],'fixed_5_probability':outcome[5.0],**post})
    summ={'contexts':len(decisions),'selected_mean_force_n':float(np.mean([d['selected_force_n'] for d in decisions])) if decisions else None,'selected_mean_success_probability':float(np.mean([d['selected_success_probability'] for d in decisions])) if decisions else None,'fixed_3_mean_success_probability':float(np.mean([d['fixed_3_probability'] for d in decisions])) if decisions else None,'fixed_4_25_mean_success_probability':float(np.mean([d['fixed_4_25_probability'] for d in decisions])) if decisions else None,'fixed_5_mean_success_probability':float(np.mean([d['fixed_5_probability'] for d in decisions])) if decisions else None}
    return summ,decisions
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--records',type=Path,required=True); ap.add_argument('--training-report',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); rows=read(a.records); rep=json.loads(a.training_report.read_text()); bm,fm,norm=load_models(rep); vals=groups(rows,'validation'); tests=groups(rows,'test'); sweep=[]
    for t in np.round(np.arange(.30,.81,.05),2): sweep.append({'threshold':float(t),**run(vals,bm,fm,norm,float(t))[0]})
    chosen=max(sweep,key=lambda x:(x['selected_mean_success_probability'],-x['selected_mean_force_n']))['threshold']; ts,td=run(tests,bm,fm,norm,chosen); out={'schema_id':'AF_FORCE_SUPERVISION_REPAIR_INFERENCE_V1','selected_threshold':chosen,'validation_threshold_sweep':sweep,'test':ts,'test_decisions':td,'test_used_for_model_selection':False}; a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); (a.out.parent/'fresh_test_summary.md').write_text('# Fresh test pending\n\nThis report is reserved for roots never used by training or model selection. The current development test above is root-disjoint but is not the final fresh confirmation.\n'); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
