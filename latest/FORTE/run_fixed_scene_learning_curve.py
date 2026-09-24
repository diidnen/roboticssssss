#!/usr/bin/env python3
"""Run the pre-specified fixed-scene Direct-vs-Joint learning curve."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FORTE = Path("/home/exouser/FORTE")
SEEDS = [0, 1, 2]
FRACTIONS = ["25%", "50%", "75%", "100%"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def write_csv(p, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def subset_traces(build, fraction, curve, all_train, dev):
    chosen = set()
    count_key = {"25%": "3_contexts", "50%": "6_contexts", "75%": "9_contexts", "100%": "12_contexts"}[fraction]
    for values in curve["subsets"].values(): chosen.update(values[count_key])
    train = [x for x in all_train if x.context_id in chosen]
    if not train: raise RuntimeError(f"empty subset {fraction}")
    return train, dev, sorted(chosen)


def ensemble_eval(gnp, models, kind, dev, segs, norm, device):
    per_branch = {}
    for model in models:
        logits = gnp.model_logits(model, kind, dev, segs, norm, device)
        for bid, z in logits.items(): per_branch.setdefault(bid, []).append(float(1.0/(1.0+np.exp(-np.clip(z,-50,50)))))
    rows = []
    for t in dev:
        rows.append({"method": "Direct" if kind == "FEAS" else "Joint", "branch_id": t.branch_id, "context_id": t.context_id, "task": t.task, "mu": t.mu, "force_N": t.force, "repeat": 1 if "_R1_" in t.branch_id else 2, "actual_success": t.outcome, "p_success": float(np.mean(per_branch[t.branch_id])), "seed_ensemble_n": len(per_branch[t.branch_id])})
    results=[]
    for (method,cid), g in pd.DataFrame(rows).groupby(["method","context_id"], sort=True):
        g=g.sort_values("force_N"); passers=g[g.p_success>=0.5]; chosen=passers.iloc[0] if len(passers) else g.iloc[-1]; q=g[abs(g.force_N-float(chosen.force_N))<1e-7]
        results.append({"method":method,"context_id":cid,"task":int(chosen.task),"mu":float(chosen.mu),"selected_force_N":float(chosen.force_N),"threshold":0.5,"fallback_used":int(len(passers)==0),"full_task_success_rate_at_selected_force":float(q.actual_success.mean()),"n_repeats_at_selected_force":len(q)})
    return rows, results


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--audit", type=Path, required=True); args=ap.parse_args(); out=args.audit
    gnp=load("gnp_curve", FORTE/"gnp_style_continuous.py"); tpi=load("tpi_curve", gnp.TPI_CODE); cf=load("cf_curve", gnp.CF_CODE)
    full=load("full_curve", gnp.FULL_CODE)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("A100 CUDA required; CPU fallback forbidden")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    # Use the already audited population builder; this reads only the 720-row
    # common set and its frozen telemetry, never TEST.
    from train_fixed_scene_exact import build_population
    all_train, dev, meta_all, pairs_all = build_population(out, gnp, tpi, cf)
    curve=json.loads((out/"FIXED_SCENE_LEARNING_CURVE_SUBSETS.json").read_text())
    curve_rows=[]; pred_rows=[]; result_rows=[]; train_diag=[]
    for fraction in FRACTIONS:
        train, dev, chosen = subset_traces(build_population, fraction, curve, all_train, dev)
        chosen_set=set(chosen); meta={k:v for k,v in meta_all.items() if k in {x.branch_id for x in train}}
        pairs=[p for p in pairs_all if p.context_id in chosen_set]
        traces=train+dev; segs=gnp.start_segments(cf,tpi,traces); norm=gnp.fit_shared_norm(train, train, {t.branch_id:segs[t.branch_id] for t in train}, tpi)
        direct_models=[]; joint_models=[]
        fraction_dir=out/"learning_curve"/fraction.replace("%","pct"); fraction_dir.mkdir(parents=True,exist_ok=True)
        for seed in SEEDS:
            fm,fh,fs=gnp.train_feas_seed(full,train,segs,norm,meta,device,seed); direct_models.append(fm)
            jm,jh,js,units,base=gnp.train_joint_seed(full,cf,tpi,train,train,pairs,segs,norm,meta,device,seed); joint_models.append(jm)
            write_csv(fraction_dir/f"DIRECT_TRAINING_seed{seed}.csv",fh); write_csv(fraction_dir/f"JOINT_TRAINING_seed{seed}.csv",jh)
            train_diag.append({"fraction":fraction,"method":"Direct","seed":seed,"contexts":len(chosen),"branches":len(train),"final_train_loss":float(fh[-1]["train_bce"]),"epochs":len(fh),"optimizer_steps":int(fs)})
            train_diag.append({"fraction":fraction,"method":"Joint","seed":seed,"contexts":len(chosen),"branches":len(train),"final_train_loss":float(jh[-1]["native_total_loss"]),"final_feasibility_loss":float(jh[-1]["feasibility_loss"]),"final_physics_loss":float(jh[-1]["physics_loss"]),"final_ie_loss":float(jh[-1]["ie_loss"]),"epochs":len(jh),"optimizer_steps":int(js),"physical_units":int(units)})
            torch.save({"method":"ActiveForcing-Direct","fraction":fraction,"seed":seed,"state_dict":fm.state_dict(),"threshold":0.5},fraction_dir/f"DIRECT_seed{seed}.pt")
            torch.save({"method":"ActiveForcing-Joint","fraction":fraction,"seed":seed,"physics_state_dict":jm.physics.state_dict(),"feas_head_state_dict":jm.feas_head.state_dict(),"threshold":0.5},fraction_dir/f"JOINT_seed{seed}.pt")
        for kind, models in [("FEAS",direct_models),("JOINT",joint_models)]:
            pr, rr=ensemble_eval(gnp,models,kind,dev,segs,norm,device)
            for r in pr: r.update({"fraction":fraction,"train_contexts":len(chosen),"train_branches":len(train)})
            for r in rr:
                r.update({"fraction":fraction,"train_contexts":len(chosen),"train_branches":len(train),"dev_contexts":len(rr)})
                curve_rows.append({"fraction":fraction,"train_contexts":len(chosen),"train_branches":len(train),"method":r["method"],"dev_contexts":len(rr),"full_task_DEV_SR":r["full_task_success_rate_at_selected_force"],"fallback_rate":r["fallback_used"],"selected_force_N":r["selected_force_N"]})
            pred_rows.extend(pr); result_rows.extend(rr)
    write_csv(out/"DIRECT_JOINT_LEARNING_CURVE.csv",curve_rows); write_csv(out/"DIRECT_JOINT_LEARNING_CURVE_TRAINING_DIAGNOSTICS.csv",train_diag); write_csv(out/"DIRECT_JOINT_LEARNING_CURVE_PREDICTIONS.csv",pred_rows); write_csv(out/"DIRECT_JOINT_LEARNING_CURVE_RESULTS.csv",result_rows)
    pivot=pd.DataFrame(curve_rows).groupby(["fraction","method"],sort=False).full_task_DEV_SR.mean().unstack()
    md="# Direct vs Joint Fixed-Scene Learning Curve\n\nThe four subsets are nested by complete friction context within each scene family. The same 24 held-out-friction DEV contexts are used at every scale; DEV labels are never used for training or checkpoint selection.\n\n| Data | Direct SR | Joint SR |\n|---|---:|---:|\n"
    for f in FRACTIONS: md += f"| {f} | {float(pivot.loc[f,'Direct']):.3f} | {float(pivot.loc[f,'Joint']):.3f} |\n"
    md += "\nPrimary metric: mean full-task success rate at the minimum force whose ensemble score is >=0.5. This is an offline fixed-scene held-out-friction diagnostic, not TEST evidence. Training budget and architecture are unchanged across scales.\n"
    (out/"DIRECT_JOINT_LEARNING_CURVE.md").write_text(md,encoding="utf-8")
    print(json.dumps({"status":"COMPLETE","out":str(out),"fractions":FRACTIONS,"curve":curve_rows},indent=2))


if __name__=="__main__": main()
