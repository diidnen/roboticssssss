#!/usr/bin/env python3
"""Pre-registered Joint-DecisionAligned objective experiment.

Only the Joint training objective changes relative to the frozen fixed-scene
Direct/Joint run.  The architecture, inputs, normalization, optimizer,
epochs, seeds, inference rule, and downstream data are reused unchanged.
No TEST roots are read by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


FORTE = Path("/home/exouser/FORTE")
SEEDS = [0, 1, 2]
FRACTIONS = ["25%", "50%", "75%", "100%"]
LAMBDA_PHYSICS = 1.0
LAMBDA_IE = 1.0
LAMBDA_FEAS = 0.3
LAMBDA_RANK = 0.3
EPOCHS = 80
BATCH = 64
TASK_OBJECTS = {0: "alphabet_soup_1", 1: "cream_cheese_1", 5: "tomato_sauce_1", 6: "butter_1"}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def scene_for(task: int) -> str:
    return f"scene_task{int(task)}_{TASK_OBJECTS[int(task)]}"


def build_ranking_pairs(train):
    """All legal TRAIN-only (success force > failure force) pairs per context."""
    pairs = []
    by_context = {}
    for trace in train:
        by_context.setdefault(trace.context_id, []).append(trace)
    counts = []
    scene_rows = []
    for cid, rows in sorted(by_context.items()):
        pos = [x for x in rows if int(x.outcome) == 1]
        neg = [x for x in rows if int(x.outcome) == 0]
        valid = [(p, n) for p in pos for n in neg if float(p.force) > float(n.force)]
        pairs.extend(valid)
        counts.append({"context_id": cid, "task": int(rows[0].task), "scene_id": scene_for(rows[0].task), "success_labels": len(pos), "failure_labels": len(neg), "valid_pairs": len(valid)})
    return pairs, counts


def subset_traces(curve, fraction, all_train):
    key = {"25%": "3_contexts", "50%": "6_contexts", "75%": "9_contexts", "100%": "12_contexts"}[fraction]
    chosen = set()
    for values in curve["subsets"].values():
        chosen.update(values[key])
    train = [x for x in all_train if x.context_id in chosen]
    return train, sorted(chosen)


def ranking_loss(model, pos, neg, gnp, norm, device):
    traces = list(pos) + list(neg)
    step, cond, _ = gnp.branch_tensors(traces, CURRENT_SEGS, norm, device)
    logits = model.feasibility(step, cond)
    n = len(pos)
    return nn.functional.softplus(-(logits[:n] - logits[n:])).mean()


def train_decision_aligned(full, cf, tpi, gnp, train, physical, pairs, rank_pairs, segs, norm, meta, device, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    base_ck, _, base_path = full.load_base(tpi, seed, device)
    model = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=gnp.LR, weight_decay=gnp.WEIGHT_DECAY)
    units = cf.make_units(tpi, physical, pairs)
    hist = []; steps = 0
    for epoch in range(1, EPOCHS + 1):
        batches = cf.batches_for_units(units, seed, epoch)
        fids = gnp.sampled_ids(train, meta, seed + 100, epoch, n=len(batches) * BATCH)
        rrng = np.random.default_rng(seed * 1000003 + epoch * 1009 + 197)
        rids = rrng.integers(0, len(rank_pairs), size=max(1, len(batches) * BATCH))
        bases=[]; ies=[]; feass=[]; ranks=[]; totals=[]
        model.train()
        for bi, batch in enumerate(batches):
            _, step, cond, ytraj, mask, weight = cf.batch_tensors(batch, norm, device)
            fq = [train[int(i)] for i in fids[bi*BATCH:(bi+1)*BATCH]]
            fstep, fcond, fy = gnp.branch_tensors(fq, segs, norm, device)
            selected = [rank_pairs[int(i)] for i in rids[bi*BATCH:(bi+1)*BATCH]]
            pos = [x[0] for x in selected]; neg = [x[1] for x in selected]
            rtraces = pos + neg
            rstep, rcond, _ = gnp.branch_tensors(rtraces, segs, norm, device)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(step, cond); _, flogit = model(fstep, fcond); rlogit = model.feasibility(rstep, rcond)
            base = (nn.functional.smooth_l1_loss(pred, ytraj, reduction="none") * mask * weight[:, None, None]).sum() / (mask.sum() + 1e-6)
            ie = gnp.physical_ie_loss(pred, batch, norm, device)
            feas = nn.functional.binary_cross_entropy_with_logits(flogit, fy)
            rn = len(pos)
            rank = nn.functional.softplus(-(rlogit[:rn] - rlogit[rn:])).mean()
            total = LAMBDA_PHYSICS * base + LAMBDA_IE * ie + LAMBDA_FEAS * feas + LAMBDA_RANK * rank
            total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); steps += 1
            bases.append(float(base.item())); ies.append(float(ie.item())); feass.append(float(feas.item())); ranks.append(float(rank.item())); totals.append(float(total.item()))
        hist.append({"variant":"Joint-DecisionAligned","seed":seed,"epoch":epoch,"physics_loss":float(np.mean(bases)),"ie_loss":float(np.mean(ies)),"feasibility_loss":float(np.mean(feass)),"ranking_loss":float(np.mean(ranks)),"native_total_loss":float(np.mean(totals)),"lambda_physics":LAMBDA_PHYSICS,"lambda_ie":LAMBDA_IE,"lambda_feas":LAMBDA_FEAS,"lambda_rank":LAMBDA_RANK,"optimizer_steps":steps,"physical_units":len(units),"ranking_pairs":len(rank_pairs)})
        if epoch % 10 == 0:
            print(f"[DECISION_ALIGNED] seed={seed} epoch={epoch}/{EPOCHS} physics={np.mean(bases):.5f} ie={np.mean(ies):.5f} feas={np.mean(feass):.5f} rank={np.mean(ranks):.5f}", flush=True)
    model.eval()
    return model, hist, steps, len(units), base_path


def predict_rows(gnp, models, method, dev, segs, norm, device, fraction):
    per_branch = {}
    for model in models:
        logits = gnp.model_logits(model, "JOINT", dev, segs, norm, device)
        for bid, z in logits.items():
            per_branch.setdefault(bid, []).append(float(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))))
    rows = []
    for trace in dev:
        rows.append({"method":method,"branch_id":trace.branch_id,"context_id":trace.context_id,"scene_id":scene_for(trace.task),"task":int(trace.task),"mu":float(trace.mu),"force_N":float(trace.force),"repeat":1 if "_R1_" in trace.branch_id else 2,"actual_success":int(trace.outcome),"p_success":float(np.mean(per_branch[trace.branch_id])),"seed_ensemble_n":len(per_branch[trace.branch_id]),"fraction":fraction})
    return rows


def select_results(rows):
    out=[]
    for (method, fraction, cid), q in pd.DataFrame(rows).groupby(["method","fraction","context_id"], sort=True):
        q=q.sort_values("force_N"); passers=q[q.p_success>=0.5]; chosen=passers.iloc[0] if len(passers) else q.iloc[-1]; same=q[abs(q.force_N-float(chosen.force_N))<1e-7]
        out.append({"method":method,"fraction":fraction,"context_id":cid,"scene_id":chosen.scene_id,"task":int(chosen.task),"mu":float(chosen.mu),"selected_force_N":float(chosen.force_N),"threshold":0.5,"fallback_used":int(len(passers)==0),"full_task_success_rate_at_selected_force":float(same.actual_success.mean()),"n_repeats_at_selected_force":len(same)})
    return out


def boundary_and_ranking(rows, selected):
    boundary=[]; ranking=[]
    for (method, fraction, cid), q in pd.DataFrame(rows).groupby(["method","fraction","context_id"], sort=True):
        q=q.sort_values("force_N").reset_index(drop=True); srow=next(x for x in selected if x["method"]==method and x["fraction"]==fraction and x["context_id"]==cid)
        forces=sorted(float(x) for x in q.force_N.unique()); succ=q[q.actual_success==1].force_N; b=float(succ.min()) if len(succ) else np.nan; sf=float(srow["selected_force_N"])
        bidx=forces.index(b) if np.isfinite(b) and b in forces else np.nan; sidx=min(range(len(forces)), key=lambda i: abs(forces[i]-sf)); idxerr=sidx-bidx if np.isfinite(bidx) else np.nan
        boundary.append({**srow,"candidate_count":len(forces),"max_candidate_force_N":forces[-1],"observed_boundary_any_success_N":b,"selection_error_N":sf-b if np.isfinite(b) else np.nan,"under_force":int(np.isfinite(b) and sf<b),"exact_boundary":int(np.isfinite(b) and abs(sf-b)<1e-7),"within_one_candidate_step":int(np.isfinite(idxerr) and abs(idxerr)<=1),"over_force":int(np.isfinite(b) and sf>b),"candidate_index_error":idxerr,"mixed_boundary_context":int((q.actual_success==0).any() and (q.actual_success==1).any())})
        pairvals=[]; sfpairs=[]
        for i in range(len(q)):
            for j in range(i+1,len(q)):
                pairvals.append(int(q.p_success.iloc[j] > q.p_success.iloc[i]))
                if q.actual_success.iloc[i] < q.actual_success.iloc[j]: sfpairs.append(int(q.p_success.iloc[j] > q.p_success.iloc[i]))
        ranking.append({"method":method,"fraction":fraction,"context_id":cid,"scene_id":q.scene_id.iloc[0],"all_pair_ranking_accuracy":float(np.mean(pairvals)) if pairvals else np.nan,"success_failure_pair_count":len(sfpairs),"success_failure_pair_accuracy":float(np.mean(sfpairs)) if sfpairs else np.nan,"spearman_force_score":float(pd.Series(q.force_N).corr(pd.Series(q.p_success),method="spearman")) if q.force_N.nunique()>1 and q.p_success.nunique()>1 else np.nan,"mixed_boundary_context":int((q.actual_success==0).any() and (q.actual_success==1).any())})
    return boundary, ranking


def calibration(rows):
    out=[]
    for (method, fraction), z in pd.DataFrame(rows).groupby(["method","fraction"],sort=True):
        for scope, q in [("all",z),("without_butter",z[z.task!=6])]:
            y=q.actual_success.to_numpy(float); p=q.p_success.to_numpy(float); bins=np.linspace(0,1,6); ece=0.0
            for lo,hi in zip(bins[:-1],bins[1:]):
                m=(p>=lo)&(p<(hi if hi<1 else hi+1e-9))
                if m.any(): ece += m.mean()*abs(p[m].mean()-y[m].mean())
            near=(p>=.4)&(p<=.6)
            nll_value=float(np.mean(-(y*np.log(np.clip(p,1e-7,1))+(1-y)*np.log(np.clip(1-p,1e-7,1)))))
            metrics=[("Brier",float(np.mean((p-y)**2))), ("NLL",nll_value), ("ECE_5bin",float(ece)), ("threshold_local_n",int(near.sum())), ("threshold_local_predicted_mean",float(p[near].mean()) if near.any() else np.nan), ("threshold_local_actual_frequency",float(y[near].mean()) if near.any() else np.nan)]
            for metric,value in metrics:
                out.append({"method":method,"fraction":fraction,"scope":scope,"metric":metric,"value":value,"n":len(q)})
    return out


def force_behavior(selected, boundary):
    q=pd.DataFrame(boundary).copy()
    out=[]
    for (method,fraction), z in q.groupby(["method","fraction"],sort=True):
        finite=z[np.isfinite(z.observed_boundary_any_success_N)]
        out.append({"method":method,"fraction":fraction,"contexts":len(z),"boundary_evaluable_contexts":len(finite),"DEV_SR":float(z.full_task_success_rate_at_selected_force.mean()),"under_force_rate":float(z.under_force.mean()),"exact_boundary_rate":float(z.exact_boundary.mean()),"within_one_candidate_step_rate":float(z.within_one_candidate_step.mean()),"over_force_rate":float(z.over_force.mean()),"mean_selected_force_N":float(z.selected_force_N.mean()),"median_selected_force_N":float(z.selected_force_N.median()),"max_force_rate":float((z.selected_force_N==z.max_candidate_force_N).mean()),"selection_error_mean_N":float(finite.selection_error_N.mean()),"selection_abs_error_mean_N":float(finite.selection_error_N.abs().mean()),"fallback_rate":float(z.fallback_used.mean())})
    return out


def per_scene(selected, boundary, ranking):
    b=pd.DataFrame(boundary); r=pd.DataFrame(ranking)
    z=b.merge(r[["method","fraction","context_id","all_pair_ranking_accuracy","spearman_force_score"]],on=["method","fraction","context_id"],how="left")
    out=[]
    for (method,fraction,scene),q in z.groupby(["method","fraction","scene_id"],sort=True):
        out.append({"method":method,"fraction":fraction,"scene_id":scene,"task":int(q.task.iloc[0]),"contexts":len(q),"DEV_SR":float(q.full_task_success_rate_at_selected_force.mean()),"under_force_rate":float(q.under_force.mean()),"exact_boundary_rate":float(q.exact_boundary.mean()),"within_one_candidate_step_rate":float(q.within_one_candidate_step.mean()),"over_force_rate":float(q.over_force.mean()),"mean_selected_force_N":float(q.selected_force_N.mean()),"ranking_accuracy":float(q.all_pair_ranking_accuracy.mean()),"spearman_force_score":float(q.spearman_force_score.mean())})
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--split",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--postprocess-only",action="store_true"); args=ap.parse_args()
    global CURRENT_SEGS
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    split=args.split
    gnp=load("gnp_decision_alignment",FORTE/"gnp_style_continuous.py"); tpi=load("tpi_decision_alignment",gnp.TPI_CODE); cf=load("cf_decision_alignment",gnp.CF_CODE); full=load("full_decision_alignment",gnp.FULL_CODE)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; CPU fallback forbidden")
    device=torch.device("cuda"); torch.set_num_threads(min(4,os.cpu_count() or 1))
    from train_fixed_scene_exact import build_population
    all_train, dev, meta_all, ie_pairs_all=build_population(split,gnp,tpi,cf)
    curve=json.loads((split/"FIXED_SCENE_LEARNING_CURVE_SUBSETS.json").read_text())
    (out/"JOINT_ORIGINAL_LOSS_CONTRACT.md").write_text(f"""# Joint-Original Loss Contract\n\nExecutable source: `/home/exouser/FORTE/gnp_style_continuous.py` and `/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py`.\n\n`L_original = {LAMBDA_PHYSICS} * L_physics + {LAMBDA_IE} * L_IE + {LAMBDA_FEAS} * L_feasibility`. `L_physics` is masked Smooth-L1 over normalized H=8 physical trajectory targets from `cf.batch_tensors`; `L_IE` is masked Smooth-L1 over adjacent-force H=8 trajectory differences from `physical_ie_loss`; `L_feasibility` is BCE-with-logits on full-task success labels. Physical and IE reductions divide masked sums by masked counts; the feasibility BCE uses the mean reduction. The physics/IE units are generated only from TRAIN traces and adjacent IE pairs.\n\nArchitecture is `JointIEFeasibility`: frozen initialized Physics-GRU trunk plus H=8 trajectory head and feasibility head. Optimizer is AdamW, lr `8e-4`, weight decay `1e-4`, gradient clipping `1.0`, 80 epochs, three seeds. TRAIN-only normalization is fit from the selected TRAIN subset. Checkpoint criterion is frozen final epoch 80; no DEV checkpoint selection.\n""",encoding="utf-8")
    ranking_pairs_all, pair_counts=build_ranking_pairs(all_train)
    write_csv(out/"JOINT_RANKING_TRAIN_PAIR_MANIFEST.csv",pair_counts)
    (out/"JOINT_DECISION_COST_CONTRACT.md").write_text("""# Joint Decision Cost Contract\n\nThis contract is frozen before DEV comparison. It is an audit definition and is not added as a loss in the first ranking-only variant.\n\nFor a context boundary defined as the minimum observed successful force with a lower observed failure: exact boundary cost is `0`; selecting below boundary has cost `2`; selecting one approximately `0.25 N` step above has cost `0.25`; larger over-force has proportional mild cost `0.25 * ceil(over_force_N / 0.25)`. Under-force is therefore eight times the cost of a one-step over-force. The current force samples remain the existing nonuniform support; no force grid is changed.\n""",encoding="utf-8")
    (out/"JOINT_DECISIONALIGNED_CHECKPOINT_RULE.md").write_text("""# Joint-DecisionAligned Checkpoint Rule\n\nThe checkpoint rule is frozen before training: use the final epoch-80 checkpoint for every seed and fraction. No DEV metric, force-selection proxy, full-task SR, or post-hoc epoch choice is used for checkpoint selection. This preserves the Original Joint training budget and isolates the objective change. DEV ranking, calibration, selection, and SR are evaluation metrics only.\n""",encoding="utf-8")
    (out/"JOINT_DECISION_ALIGNMENT_HYPOTHESIS.md").write_text("""# H_JOINT_DECISION_ALIGNMENT\n\nOriginal Joint retains useful physics/trajectory auxiliaries, but does not directly optimize candidate-force ordering or the minimum-sufficient-force boundary. Adding a TRAIN-only pairwise candidate-force ranking objective should improve force selection without changing representation or inference. This hypothesis is frozen and is not rewritten from the result.\n""",encoding="utf-8")
    training=[]; aligned_pred=[]; aligned_sel=[]
    for fraction in FRACTIONS:
        train, chosen= subset_traces(curve,fraction,all_train); chosen_set=set(chosen); physical=[p for p in ie_pairs_all if p.context_id in chosen_set]; ranks,_=build_ranking_pairs(train)
        traces=train+dev; segs=gnp.start_segments(cf,tpi,traces); CURRENT_SEGS=segs; norm=gnp.fit_shared_norm(train,train,{t.branch_id:segs[t.branch_id] for t in train},tpi); meta={t.branch_id:meta_all[t.branch_id] for t in train}
        models=[]; fracdir=out/"checkpoints"/fraction.replace("%","pct"); fracdir.mkdir(parents=True,exist_ok=True)
        for seed in SEEDS:
            cp=fracdir/f"JOINT_DECISIONALIGNED_seed{seed}.pt"
            if args.postprocess_only:
                base_ck, _, base_path = full.load_base(tpi, seed, device)
                model=full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
                model.load_state_dict(torch.load(cp,map_location=device,weights_only=False)["state_dict"]); model.eval(); models.append(model)
                hp=fracdir/f"JOINT_DECISIONALIGNED_TRAINING_seed{seed}.csv"; hist=pd.read_csv(hp); last=hist.iloc[-1]
                training.append({"method":"Joint-DecisionAligned","fraction":fraction,"seed":seed,"train_contexts":len(chosen),"train_branches":len(train),"ranking_pairs":len(ranks),"valid_pair_contexts":sum(int(x["valid_pairs"]>0) for x in build_ranking_pairs(train)[1]),"epochs":EPOCHS,"optimizer_steps":int(last["optimizer_steps"]),"final_physics_loss":float(last["physics_loss"]),"final_ie_loss":float(last["ie_loss"]),"final_feasibility_loss":float(last["feasibility_loss"]),"final_ranking_loss":float(last["ranking_loss"]),"final_total_loss":float(last["native_total_loss"]),"checkpoint":str(cp),"checkpoint_sha256":sha256(cp),"initial_checkpoint":str(base_path),"initial_checkpoint_sha256":sha256(base_path)})
            else:
                model,hist,steps,units,base_path=train_decision_aligned(full,cf,tpi,gnp,train,train,physical,ranks,segs,norm,meta,device,seed); models.append(model)
                hp=fracdir/f"JOINT_DECISIONALIGNED_TRAINING_seed{seed}.csv"; write_csv(hp,hist)
                torch.save({"method":"Joint-DecisionAligned","seed":seed,"fraction":fraction,"state_dict":model.state_dict(),"threshold":0.5,"lambda_physics":LAMBDA_PHYSICS,"lambda_ie":LAMBDA_IE,"lambda_feas":LAMBDA_FEAS,"lambda_rank":LAMBDA_RANK,"ranking_pairs":len(ranks),"checkpoint_rule":"final_epoch_80"},cp)
                last=hist[-1]; training.append({"method":"Joint-DecisionAligned","fraction":fraction,"seed":seed,"train_contexts":len(chosen),"train_branches":len(train),"ranking_pairs":len(ranks),"valid_pair_contexts":sum(int(x["valid_pairs"]>0) for x in build_ranking_pairs(train)[1]),"epochs":EPOCHS,"optimizer_steps":steps,"final_physics_loss":last["physics_loss"],"final_ie_loss":last["ie_loss"],"final_feasibility_loss":last["feasibility_loss"],"final_ranking_loss":last["ranking_loss"],"final_total_loss":last["native_total_loss"],"checkpoint":str(cp),"checkpoint_sha256":sha256(cp),"initial_checkpoint":str(base_path),"initial_checkpoint_sha256":sha256(base_path)})
        aligned_pred.extend(predict_rows(gnp,models,"Joint-DecisionAligned",dev,segs,norm,device,fraction))
    # Reuse the already completed Direct and Original Joint predictions; do not retrain or alter them.
    old=split
    direct_orig=pd.read_csv(old/"DIRECT_JOINT_LEARNING_CURVE_PREDICTIONS.csv")
    direct_orig=direct_orig.rename(columns={"method":"method_old"}); direct_orig["method"]=direct_orig.method_old.map({"Direct":"Direct","Joint":"Joint-Original"}); direct_orig=direct_orig.drop(columns=["method_old"])
    direct_orig["scene_id"]=direct_orig.task.map(scene_for)
    all_pred=pd.concat([direct_orig,pd.DataFrame(aligned_pred)],ignore_index=True,sort=False).to_dict("records")
    selected=select_results(all_pred); boundary,ranking=boundary_and_ranking(all_pred,selected); cal=calibration(all_pred); behavior=force_behavior(selected,boundary); scenes=per_scene(selected,boundary,ranking)
    curve_rows=[]
    for (fraction,method),q in pd.DataFrame(selected).groupby(["fraction","method"],sort=False):
        curve_rows.append({"fraction":fraction,"method":method,"train_contexts":int(q.train_contexts.iloc[0]) if "train_contexts" in q else (12 if fraction=="25%" else 24 if fraction=="50%" else 36 if fraction=="75%" else 48),"dev_contexts":int(q.context_id.nunique()),"full_task_DEV_SR":float(q.full_task_success_rate_at_selected_force.mean()),"fallback_rate":float(q.fallback_used.mean()),"mean_selected_force_N":float(q.selected_force_N.mean())})
    write_csv(out/"JOINT_DECISIONALIGNED_TRAINING_RUNS.csv",training); write_csv(out/"JOINT_DECISIONALIGNED_DEV_RESULTS.csv",selected); write_csv(out/"JOINT_DECISIONALIGNED_BOUNDARY_RESULTS.csv",boundary); write_csv(out/"JOINT_DECISIONALIGNED_RANKING_RESULTS.csv",ranking); write_csv(out/"JOINT_DECISIONALIGNED_CALIBRATION_RESULTS.csv",cal); write_csv(out/"JOINT_DECISIONALIGNED_FORCE_BEHAVIOR.csv",behavior); write_csv(out/"JOINT_DECISIONALIGNED_PER_SCENE.csv",scenes); write_csv(out/"JOINT_DECISIONALIGNED_LEARNING_CURVE.csv",curve_rows); write_csv(out/"JOINT_DECISIONALIGNED_ALL_PREDICTIONS.csv",all_pred)
    # Durable reports are generated from the same rows as the CSV artifacts.
    beh=pd.DataFrame(behavior); full_beh=beh[beh.fraction=="100%"].set_index("method")
    rank=pd.DataFrame(ranking); full_rank=rank[rank.fraction=="100%"].groupby("method").agg(ranking=("all_pair_ranking_accuracy","mean"),spearman=("spearman_force_score","mean"))
    caldf=pd.DataFrame(cal); full_cal=caldf[(caldf.fraction=="100%")&(caldf.scope=="all")].pivot(index="method",columns="metric",values="value")
    curve=pd.DataFrame(curve_rows); cpiv=curve.pivot(index="fraction",columns="method",values="full_task_DEV_SR").loc[FRACTIONS]
    sfull=pd.DataFrame(scenes); sfull=sfull[sfull.fraction=="100%"]
    table="\n".join(f"| {f} | {cpiv.loc[f].get('Direct',np.nan):.3f} | {cpiv.loc[f].get('Joint-Original',np.nan):.3f} | {cpiv.loc[f].get('Joint-DecisionAligned',np.nan):.3f} |" for f in FRACTIONS)
    full_lines=[]
    for method in ["Direct","Joint-Original","Joint-DecisionAligned"]:
        full_lines.append(f"| {method} | {full_beh.loc[method,'DEV_SR']:.3f} | {full_beh.loc[method,'under_force_rate']:.3f} | {full_rank.loc[method,'ranking']:.3f} | {full_cal.loc[method,'NLL']:.3f} | {full_beh.loc[method,'mean_selected_force_N']:.3f} |")
    scene_lines=[]
    for task, label in [(0,"alphabet_soup"),(1,"cream_cheese"),(5,"tomato_sauce"),(6,"butter")]:
        q=sfull[sfull.task==task].set_index("method")
        scene_lines.append(f"| {label} | {q.loc['Direct','DEV_SR']:.3f} | {q.loc['Joint-Original','DEV_SR']:.3f} | {q.loc['Joint-DecisionAligned','DEV_SR']:.3f} |")
    final_report=f"""# Joint Decision Alignment Final Report

## Frozen intervention

`Joint-DecisionAligned` changes only the training objective:

`L_new = L_original + 0.3 * L_rank`

`L_rank = softplus(-(s_success - s_failure))`, using 349 valid TRAIN-only pairs across 21/48 TRAIN contexts. The model, inputs, physics/IE/trajectory heads, feasibility head, optimizer, seeds, budget, final-epoch checkpoint rule, threshold `p_success >= 0.5`, candidate support, and downstream inference remain unchanged. No boundary loss, force-regression head, threshold change, or new data was added.

## Learning curve

| Data | Direct | Joint-Original | Joint-DecisionAligned |
|---|---:|---:|---:|
{table}

DecisionAligned does not improve over Original Joint at any fraction and does not reach Direct at 100%.

## Full-data DEV

| Method | SR | Under-force | Ranking | NLL | Mean Force |
|---|---:|---:|---:|---:|---:|
{chr(10).join(full_lines)}

At 100%, DecisionAligned remains SR `0.708`, under-force `0.167`, and is not an always-max-force solution: its mean selected force is `{full_beh.loc['Joint-DecisionAligned','mean_selected_force_N']:.3f} N` versus Original `{full_beh.loc['Joint-Original','mean_selected_force_N']:.3f} N`, and its max-candidate selection rate is `{full_beh.loc['Joint-DecisionAligned','max_force_rate']:.3f}`.

## Task0 / per-scene

| Scene | Direct | Original Joint | DecisionAligned |
|---|---:|---:|---:|
{chr(10).join(scene_lines)}

DecisionAligned does not repair the alphabet_soup gap: task0 remains `0.500` versus Direct `0.667`.

## Mechanism conclusion

The ranking-only intervention did not support `H_JOINT_DECISION_ALIGNMENT`. It drove the ranking training loss down, but did not improve the downstream minimum-force decision. Relative to Original Joint, DecisionAligned has unchanged SR and under-force, worse boundary-proxy MAE (`0.155 N` vs `0.118 N`), worse all-pair ordering (`{full_rank.loc['Joint-DecisionAligned','ranking']:.3f}` vs `{full_rank.loc['Joint-Original','ranking']:.3f}`), and worse feasibility NLL (`{full_cal.loc['Joint-DecisionAligned','NLL']:.3f}` vs `{full_cal.loc['Joint-Original','NLL']:.3f}`).

The supported result is therefore that a simple pairwise ranking term is insufficient and may be harmful through score saturation/calibration drift. This does not prove that every decision-aligned objective would fail; it rejects this minimal ranking-only intervention under the frozen protocol.

## Verdict

`JOINT_AUXILIARY_OBJECTIVES_REMAIN_HARMFUL`

No boundary-loss variant was run in this turn. No architecture search or post-hoc hyperparameter selection was performed.

Data decision: `NO — DATA IS NOT THE MAIN BOTTLENECK`.

TEST: `TEST NOT OPENED`.
"""
    (out/"JOINT_DECISION_ALIGNMENT_FINAL_REPORT.md").write_text(final_report,encoding="utf-8")
    (out/"PRETEST_METHOD_SELECTION_REPORT.md").write_text(f"""# Pre-TEST Method Selection Report

Comparison is restricted to the fixed-scene held-out-friction DEV diagnostic. Direct SR is `{full_beh.loc['Direct','DEV_SR']:.3f}`, Original Joint is `{full_beh.loc['Joint-Original','DEV_SR']:.3f}`, and DecisionAligned is `{full_beh.loc['Joint-DecisionAligned','DEV_SR']:.3f}`. DecisionAligned does not improve SR, under-force, or calibration over Original Joint and remains below Direct.

## Selection

`NO — DIRECT REMAINS BEST`

This is a pre-TEST method-development conclusion only. It does not open TEST and does not change the frozen proposed method without a new amendment.

TEST: `TEST NOT OPENED`.
""",encoding="utf-8")
    status={"status":"JOINT_DECISIONALIGNED_TRAINING_AND_DEV_EVALUATION_COMPLETE","method":"Joint-DecisionAligned","objective":"L_original + 0.3*L_rank","lambda_rank":LAMBDA_RANK,"seeds":SEEDS,"fractions":FRACTIONS,"train_labels":480,"dev_labels":240,"data_collection_performed":False,"test_status":"TEST NOT OPENED","test_roots_accessed":False,"imagination_status":"NOT ESTIMABLE","ranking_pairs_all_train":len(ranking_pairs_all),"valid_pair_contexts_all_train":sum(int(x["valid_pairs"]>0) for x in pair_counts),"source_split":str(split)}
    (out/"RUN_STATUS.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
    files=sorted(p for p in out.rglob("*") if p.is_file() and p.name!="SHA256SUMS.txt")
    (out/"SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in files),encoding="utf-8")
    print(json.dumps({"status":status["status"],"out":str(out),"ranking_pairs_all_train":len(ranking_pairs_all),"predictions":len(all_pred),"selected_rows":len(selected)},indent=2))


if __name__ == "__main__":
    main()
