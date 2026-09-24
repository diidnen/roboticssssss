#!/usr/bin/env python3
"""Train and evaluate the frozen same-task0 context-count learning curve.

`train` reads TRAIN only and freezes all 60 checkpoints.  `evaluate` refuses to
open TEST until that checkpoint manifest is complete and the atomic TEST commit
passes.  No calibration is applied and no winner/model is selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import task0_visual_context_early as early
import task0_visual_generalization as gen
import task0_joint_novisual_diagnostic as jnv
from task0_context_pipeline_freeze import verify as verify_pipeline_freeze


ROOT = Path("/home/exouser/FORTE")
SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
PROJECT = ROOT / "task0_context_sample_complexity_20260831"
SPLIT = ROOT / "TASK0_CONTEXT_SPLIT_MANIFEST.json"
EARLY = ROOT / "task0_visual_context_early_20260831_025000"
JNV18 = ROOT / "task0_joint_novisual_diagnostic_20260831"
NS = [18, 30, 50, 80]
SEEDS = [0, 1, 2]
MODELS = ["Base", "Residual", "Full Visual", "Joint-NoVisual", "Visual Joint"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); df.to_csv(path, index=False)


def modules():
    tpi = early.load_module("curve_tpi", early.TPI_CODE)
    cf = early.load_module("curve_cf", early.CF_CODE)
    full = early.load_module("curve_full", early.FULL_CODE)
    return tpi, cf, full


def merged_train_tables():
    c1 = pd.read_csv(SOURCE / "collection_train/task0/task0/context.csv")
    b1 = pd.read_csv(SOURCE / "collection_train/task0/task0/branches.csv")
    v1 = pd.read_csv(SOURCE / "collection_train/visual_alignment_worker.csv")
    c2 = pd.read_csv(PROJECT / "collection_train_new/task0/task0/context.csv")
    b2 = pd.read_csv(PROJECT / "collection_train_new/task0/task0/branches.csv")
    v2 = pd.read_csv(PROJECT / "collection_train_new/visual_alignment_worker.csv")
    c = pd.concat([c1, c2], ignore_index=True)
    b = pd.concat([b1, b2], ignore_index=True)
    v = pd.concat([v1, v2], ignore_index=True)
    return c, b, v


def fit_pca17(cids, vmap, out: Path):
    raw = np.stack([np.load(vmap[c]["visual_feature_path"], allow_pickle=False).astype(np.float32) for c in cids])
    mean = raw.mean(0)
    _, singular, vt = np.linalg.svd(raw - mean, full_matrices=False)
    components = vt[:17].astype(np.float32)
    projected = ((raw - mean) @ components.T).astype(np.float32)
    pmean = projected.mean(0).astype(np.float32)
    pstd = projected.std(0).astype(np.float32); pstd[pstd < 1e-6] = 1.0
    np.savez(out, raw_mean=mean, components=components, projected_mean=pmean,
             projected_std=pstd, singular_values=singular)
    return raw, (projected - pmean) / pstd


def transform_pca(paths, pca_path):
    p = np.load(pca_path)
    raw = np.stack([np.load(x, allow_pickle=False).astype(np.float32) for x in paths])
    projected = (raw - p["raw_mean"]) @ p["components"].T
    return raw, ((projected - p["projected_mean"]) / p["projected_std"]).astype(np.float32)


def build_pairs(cf, traces, meta):
    by = {}
    for tr in traces:
        by.setdefault((tr.context_id, meta[tr.branch_id].repeat), []).append(tr)
    pairs = []
    for (cid, repeat), q in sorted(by.items()):
        q = sorted(q, key=lambda tr: meta[tr.branch_id].stratum)
        if len(q) != 5:
            raise RuntimeError(f"expected five strata for {cid}/R{repeat}")
        for a, b in zip(q[:-1], q[1:]):
            ma, mb = meta[a.branch_id], meta[b.branch_id]
            pairs.append(cf.Pair(
                f"curve:{cid}:R{repeat}:S{ma.stratum}_vs_S{mb.stratum}",
                f"curve:{cid}:R{repeat}", "TRAIN", "TASK0_CONTEXT_CURVE",
                cid, a.root_id, 0, ma.friction_band, a.mu, a.force, b.force,
                "adjacent", False, a, b,
            ))
    if len(pairs) != len(by) * 4:
        raise RuntimeError("adjacent IE pair count mismatch")
    return pairs


def load_train_N(n, tpi, cf):
    manifest = json.loads(SPLIT.read_text())
    ids = manifest["nested_TRAIN_context_ids"][f"N{n}"]
    contexts, branches, visual = merged_train_tables()
    contexts = contexts[contexts.context_id.astype(str).isin(ids)].sort_values("context_id")
    branches = branches[branches.context_id.astype(str).isin(ids)].copy()
    visual = visual[visual.context_id.astype(str).isin(ids)].copy()
    if len(contexts) != n or contexts.context_id.nunique() != n or len(visual) != n:
        raise RuntimeError(f"N{n} context/visual mismatch")
    if len(branches) != n * 10:
        raise RuntimeError(f"N{n} branch count != {n*10}")
    vmap = visual.set_index("context_id").to_dict("index")
    pca_path = (EARLY / "TASK0_PCA17_PROVISIONAL.npz") if n == 18 else (PROJECT / f"TASK0_PCA17_N{n}_TRAIN_ONLY.npz")
    cids = contexts.context_id.astype(str).tolist()
    if not pca_path.exists():
        raw, z = fit_pca17(cids, vmap, pca_path)
    else:
        raw, z = transform_pca([Path(vmap[c]["visual_feature_path"]) for c in cids], pca_path)
    cmap = {}
    for row, rv, zv in zip(contexts.itertuples(index=False), raw, z):
        state0, mask0, _ = early.strict_preprobe_state(Path(str(row.probe_telemetry_path)))
        cmap[str(row.context_id)] = {
            "visual_raw": rv, "visual": zv, "preprobe_state": state0,
            "preprobe_mask": mask0, "root_id": str(row.root_id),
            "friction": float(row.hidden_friction_analysis_only),
            "friction_band": str(row.friction_band),
        }
    traces, meta = [], {}
    corrected = {"left_normal_force_N", "right_normal_force_N", "left_tangential_force_N",
                 "right_tangential_force_N", "object_vx_mps", "object_vy_mps", "object_vz_mps"}
    for r in branches.sort_values("branch_id").itertuples(index=False):
        cid = str(r.context_id); path = Path(str(r.telemetry_path)); d = pd.read_csv(path)
        if len(d) < early.H + 1 or not corrected <= set(d.columns):
            raise RuntimeError(f"bad TRAIN telemetry {r.branch_id}")
        state, mask = tpi.state_from(d); state = state.copy(); mask = mask.copy()
        state[0] = cmap[cid]["preprobe_state"]; mask[0] = cmap[cid]["preprobe_mask"]
        force = float(r.requested_force_N); mu = float(r.hidden_friction_analysis_only)
        nominal = tpi.nominal_from(d, 0, force, mu, state, mask)
        tr = tpi.Trace(str(r.branch_id), cid, str(r.root_id), 0, "TRAIN", force, mu,
                       int(r.full_task_success_y), "continuous", path, state, mask, nominal,
                       d.phase.astype(str).tolist(), 1.0, "TASK0_CONTEXT_CURVE")
        s, rep = early.parse_cell(str(r.branch_label)); traces.append(tr)
        meta[tr.branch_id] = early.Meta(tr.branch_id, cid, 0, str(r.root_id),
                                        str(r.friction_band), int(r.full_task_success_y), force, rep, s)
    pairs = build_pairs(cf, traces, meta)
    segs, norm = early.build_segments_and_norm(cf, tpi, traces)
    if n == 18:
        frozen = np.load(EARLY / "TASK0_TRAIN_NORMALIZATION.npz")
        frozen_norm = tuple(
            frozen[k].astype(np.float32)
            for k in ["x_mean", "x_std", "y_mean", "y_std"]
        )
        if not all(np.array_equal(a, b) for a, b in zip(norm, frozen_norm)):
            raise RuntimeError("N18 recomputed normalization differs from authoritative frozen normalization")
        norm = frozen_norm
    return contexts, branches, cmap, traces, meta, pairs, segs, norm, pca_path


def ck_path(n, model, seed):
    slug = model.upper().replace(" ", "_").replace("-", "_")
    return PROJECT / "checkpoints" / f"N{n}_{slug}_seed{seed}.pt"


def save_ck(path, model, seed, state, n, pca_path, norm, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model, "seed": seed, "N": n, "state_dict": state,
        "epochs": 80, "optimizer": "AdamW", "lr": early.LR,
        "weight_decay": early.WEIGHT_DECAY, "PCA_dimension": 17,
        "PCA_path": str(pca_path), "PCA_sha256": sha256(pca_path),
        "normalization": {k: v.tolist() for k, v in zip(["x_mean","x_std","y_mean","y_std"], norm)},
        "TRAIN_only": True, "TEST_used": False, "diagnostic_only": model == "Joint-NoVisual",
        **(extra or {}),
    }, path)


def load_one(path, model, seed, tpi, full, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if model == "Base": m = full.FeasibilityOnly().to(device)
    elif model == "Residual": m = early.VisualResidual(17).to(device)
    elif model == "Full Visual": m = early.VisualFullFeas(17).to(device)
    elif model == "Visual Joint":
        ps = {k[len("physics."):]: v for k,v in ck["state_dict"].items() if k.startswith("physics.")}
        m = early.VisualJoint(tpi, ps, 17).to(device)
    else:
        base_ck, _, _ = full.load_base(tpi, seed, device)
        m = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


def train_all():
    verify_pipeline_freeze()
    train_audit = json.loads((PROJECT / "TASK0_TRAIN_COLLECTION_AUDIT.json").read_text())
    if train_audit.get("status") != "PASS": raise RuntimeError("TRAIN collection audit not PASS")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    tpi, cf, full = modules(); device = torch.device("cpu")
    rows = []
    for n in NS:
        contexts, branches, cmap, traces, meta, pairs, segs, norm, pca_path = load_train_N(n,tpi,cf)
        for seed in SEEDS:
            # Base
            bp = ck_path(n,"Base",seed)
            if bp.exists(): base = load_one(bp,"Base",seed,tpi,full,device)
            elif n == 18:
                src=EARLY/f"PROSPECTIVE_BASE_FEAS_task0_seed{seed}.pt"; q=torch.load(src,map_location=device,weights_only=False)
                base=full.FeasibilityOnly().to(device);base.load_state_dict(q["state_dict"]);base.eval();save_ck(bp,"Base",seed,base.state_dict(),n,pca_path,norm,{"reused_authoritative_N18":str(src),"source_sha256":sha256(src)})
            else:
                base,h,steps=early.train_base(full,traces,segs,norm,cmap,meta,device,seed);save_ck(bp,"Base",seed,base.state_dict(),n,pca_path,norm,{"optimizer_steps":steps})
            rows.append({"N":n,"model":"Base","seed":seed,"checkpoint":str(bp),"sha256":sha256(bp)})
            # Residual
            rp=ck_path(n,"Residual",seed)
            if rp.exists(): residual=load_one(rp,"Residual",seed,tpi,full,device)
            elif n==18:
                src=EARLY/f"VISUAL_INTERCEPT_RESIDUAL_task0_seed{seed}.pt";q=torch.load(src,map_location=device,weights_only=False);residual=early.VisualResidual(17);residual.load_state_dict(q["state_dict"]);residual.eval();save_ck(rp,"Residual",seed,residual.state_dict(),n,pca_path,norm,{"reused_authoritative_N18":str(src),"source_sha256":sha256(src)})
            else:
                residual,h,steps=early.train_residual(base,traces,segs,norm,cmap,meta,device,seed,17);save_ck(rp,"Residual",seed,residual.state_dict(),n,pca_path,norm,{"optimizer_steps":steps})
            rows.append({"N":n,"model":"Residual","seed":seed,"checkpoint":str(rp),"sha256":sha256(rp)})
            # Full
            fp=ck_path(n,"Full Visual",seed)
            if fp.exists(): fullm=load_one(fp,"Full Visual",seed,tpi,full,device)
            elif n==18:
                src=EARLY/f"VISUAL_CONTEXT_FULL_FEAS_task0_seed{seed}.pt";q=torch.load(src,map_location=device,weights_only=False);fullm=early.VisualFullFeas(17);fullm.load_state_dict(q["state_dict"]);fullm.eval();save_ck(fp,"Full Visual",seed,fullm.state_dict(),n,pca_path,norm,{"reused_authoritative_N18":str(src),"source_sha256":sha256(src)})
            else:
                fullm,h,steps=early.train_full(traces,segs,norm,cmap,meta,device,seed,17);save_ck(fp,"Full Visual",seed,fullm.state_dict(),n,pca_path,norm,{"optimizer_steps":steps})
            rows.append({"N":n,"model":"Full Visual","seed":seed,"checkpoint":str(fp),"sha256":sha256(fp)})
            # Joint-NoVisual
            npth=ck_path(n,"Joint-NoVisual",seed)
            if npth.exists(): novis=load_one(npth,"Joint-NoVisual",seed,tpi,full,device)
            elif n==18:
                src=JNV18/f"TASK0_JOINT_NOVISUAL_seed{seed}.pt";q=torch.load(src,map_location=device,weights_only=False);base_ck,_,_=full.load_base(tpi,seed,device);novis=full.JointIEFeasibility(tpi,base_ck["state_dict"]);novis.load_state_dict(q["state_dict"]);novis.eval();save_ck(npth,"Joint-NoVisual",seed,novis.state_dict(),n,pca_path,norm,{"reused_diagnostic_N18":str(src),"source_sha256":sha256(src),"lambda_physics":1.0,"lambda_IE":1.0,"lambda_feasibility":0.3})
            else:
                novis,h,steps,units,base_path=jnv.train_seed(full,cf,tpi,traces,pairs,segs,norm,cmap,meta,seed,device);save_ck(npth,"Joint-NoVisual",seed,novis.state_dict(),n,pca_path,norm,{"optimizer_steps":steps,"physical_units":units,"lambda_physics":1.0,"lambda_IE":1.0,"lambda_feasibility":0.3})
            rows.append({"N":n,"model":"Joint-NoVisual","seed":seed,"checkpoint":str(npth),"sha256":sha256(npth)})
            # Visual Joint
            jp=ck_path(n,"Visual Joint",seed)
            if jp.exists(): joint=load_one(jp,"Visual Joint",seed,tpi,full,device)
            elif n==18:
                src=EARLY/f"VISUAL_CONTEXT_JOINT_task0_seed{seed}.pt";q=torch.load(src,map_location=device,weights_only=False);ps={k[len('physics.'):]:v for k,v in q['state_dict'].items() if k.startswith('physics.')};joint=early.VisualJoint(tpi,ps,17);joint.load_state_dict(q["state_dict"]);joint.eval();save_ck(jp,"Visual Joint",seed,joint.state_dict(),n,pca_path,norm,{"reused_authoritative_N18":str(src),"source_sha256":sha256(src),"lambda_physics":1.0,"lambda_IE":1.0,"lambda_feasibility":0.3})
            else:
                joint,h,steps,units,base_path=early.train_joint(full,cf,tpi,traces,pairs,segs,norm,cmap,meta,device,seed,17);save_ck(jp,"Visual Joint",seed,joint.state_dict(),n,pca_path,norm,{"optimizer_steps":steps,"physical_units":units,"lambda_physics":1.0,"lambda_IE":1.0,"lambda_feasibility":0.3})
            rows.append({"N":n,"model":"Visual Joint","seed":seed,"checkpoint":str(jp),"sha256":sha256(jp)})
            write_json(PROJECT/"TASK0_CONTEXT_CHECKPOINT_MANIFEST_PARTIAL.json",{"status":"IN_PROGRESS","checkpoints":rows,"TEST_used":False})
    if len(rows)!=60: raise RuntimeError("expected 60 checkpoints")
    write_json(PROJECT/"TASK0_CONTEXT_CHECKPOINT_MANIFEST.json",{"status":"ALL_60_FROZEN_BEFORE_TEST_TOUCH","checkpoints":rows,"TEST_used":False,"architecture_loss_lambda_selection_changes":False})
    print(json.dumps({"status":"ALL_60_FROZEN_BEFORE_TEST_TOUCH","checkpoints":60},indent=2))


def load_eval(split, tpi, pca_path):
    if split=="TEST": base=PROJECT/"collection_test_frozen"; c=pd.read_csv(base/"task0/task0/context.csv");b=pd.read_csv(base/"task0/task0/branches.csv");v=pd.read_csv(base/"visual_alignment_worker.csv")
    else: raise ValueError(split)
    c=c.sort_values("context_id");v=v[v.context_id.astype(str).isin(c.context_id.astype(str))];vm=v.set_index("context_id").to_dict("index")
    raw,z=transform_pca([Path(vm[x]["visual_feature_path"]) for x in c.context_id.astype(str)],pca_path)
    cmap={};templates={}
    for row,rv,zv in zip(c.itertuples(index=False),raw,z):
        cid=str(row.context_id);state0,mask0,_=early.strict_preprobe_state(Path(str(row.probe_telemetry_path)));q=b[b.context_id.astype(str)==cid].sort_values(["requested_force_N","branch_label"]);path=Path(str(q.iloc[0].telemetry_path));d=pd.read_csv(path);state,mask=tpi.state_from(d);state=state.copy();mask=mask.copy();state[0]=state0;mask[0]=mask0;mu=float(row.hidden_friction_analysis_only);nom=tpi.nominal_from(d,0,3.0,mu,state,mask);tr=tpi.Trace(f"template:{cid}",cid,str(row.root_id),0,split,3.0,mu,0,"dense",path,state,mask,nom,d.phase.astype(str).tolist(),1.0,"TASK0_FROZEN_TEST")
        cmap[cid]={"visual":zv,"visual_raw":rv,"root_id":str(row.root_id),"friction":mu,"friction_band":str(row.friction_band)};templates[cid]=tr
    return c,b,cmap,templates


def load_models_N(n,tpi,full,device):
    models={}
    for seed in SEEDS:
        base=load_one(ck_path(n,"Base",seed),"Base",seed,tpi,full,device);models[("Base",seed)]=base
        res=load_one(ck_path(n,"Residual",seed),"Residual",seed,tpi,full,device);models[("Residual",seed)]=(base,res)
        for m in ["Full Visual","Joint-NoVisual","Visual Joint"]:models[(m,seed)]=load_one(ck_path(n,m,seed),m,seed,tpi,full,device)
    return models


def predict(models,cmap,templates,norm,tpi,cf,forces):
    rows=[]
    with torch.no_grad():
        for cid in sorted(templates):
            vis=torch.tensor(cmap[cid]["visual"][None],dtype=torch.float32)
            for force in forces:
                sn,cn=gen.normalized_segment(cf,tpi,templates[cid],float(force),norm);step=torch.tensor(sn[None],dtype=torch.float32);cond=torch.tensor(cn[None],dtype=torch.float32)
                for model in MODELS:
                    probs=[]
                    for seed in SEEDS:
                        m=models[(model,seed)]
                        if model=="Base":logit=m(step,cond)
                        elif model=="Residual":logit=m[0](step,cond)+m[1](vis)
                        elif model=="Full Visual":logit=m(step,cond,vis)
                        elif model=="Joint-NoVisual":logit=m(step,cond)[1]
                        else:logit=m(step,cond,vis)[1]
                        probs.append(float(torch.sigmoid(logit)[0]))
                    rows.append({"context_id":cid,"force_N":float(force),"model":model,**{f"seed{s}_prob":probs[s] for s in SEEDS},"ensemble_prob":float(np.mean(probs)),"seed_std":float(np.std(probs,ddof=1))})
    return pd.DataFrame(rows)


def per_root_metrics(n,pred,real,fronts,cmap):
    rows=[];fmap=fronts.set_index("context_id").real_frontier_N.to_dict()
    for model in MODELS:
        for cid,rr in real.groupby("context_id"):
            rr=rr.sort_values("force_N");q=pred[(pred.model==model)&(pred.context_id==cid)].sort_values("force_N")
            for agg,col in [("ENSEMBLE","ensemble_prob")]+[(f"SEED_{s}",f"seed{s}_prob") for s in SEEDS]:
                qr=q[q.force_N.isin(rr.force_N)].sort_values("force_N");p=qr[col].to_numpy(float);y=rr.p_real.to_numpy(float);safe=q[q[col]>=gen.RHO];fp=float(safe.force_N.min()) if len(safe) else math.nan;fr=float(fmap[cid]);err=fp-fr if math.isfinite(fp) and math.isfinite(fr) else math.nan;dense=q[col].to_numpy(float);diff=np.diff(dense)
                rows.append({"N":n,"model":model,"aggregation":agg,"context_id":cid,"root_id":cmap[cid]["root_id"],"friction_band":cmap[cid]["friction_band"],"mu_GT":cmap[cid]["friction"],"probability_MAE":float(np.mean(np.abs(p-y))),"signed_probability_error":float(np.mean(p-y)),"real_frontier_N":fr,"real_frontier_supported":math.isfinite(fr),"predicted_frontier_N":fp,"signed_frontier_error_N":err,"absolute_frontier_error_N":abs(err) if math.isfinite(err) else math.nan,"finite_decision":math.isfinite(fp),"under_force":bool(err<0) if math.isfinite(err) else math.nan,"under_force_magnitude_N":max(0,-err) if math.isfinite(err) else math.nan,"excess_force_N":max(0,err) if math.isfinite(err) else math.nan,"dense_monotonic":bool(np.all(diff>=-1e-8)),"nonmonotonic_steps":int(np.sum(diff< -1e-8)),"safe_to_unsafe_reversals":int(np.sum((dense[:-1]>=gen.RHO)&(dense[1:]<gen.RHO)))})
    return pd.DataFrame(rows)


def evaluate_all():
    verify_pipeline_freeze()
    ck=json.loads((PROJECT/"TASK0_CONTEXT_CHECKPOINT_MANIFEST.json").read_text());commit=json.loads((PROJECT/"TASK0_FROZEN_TEST_COMMIT.json").read_text())
    if ck.get("status")!="ALL_60_FROZEN_BEFORE_TEST_TOUCH" or len(ck["checkpoints"])!=60:raise RuntimeError("all 60 checkpoints must freeze before TEST")
    if commit.get("status")!="ATOMICALLY_COMMITTED_FROZEN_TEST":raise RuntimeError("TEST not atomically committed")
    torch.set_num_threads(1);torch.set_num_interop_threads(1);tpi,cf,full=modules();device=torch.device("cpu")
    summary=[];perroots=[];safety=[];distrows=[]
    for n in NS:
        _,train_branches,train_cmap,train_traces,_,_,_,norm,pca=load_train_N(n,tpi,cf);contexts,test_branches,test_cmap,templates=load_eval("TEST",tpi,pca);models=load_models_N(n,tpi,full,device)
        pred=predict(models,test_cmap,templates,norm,tpi,cf,gen.FORCES_DENSE);real,fronts=gen.real_curves(test_branches);pr=per_root_metrics(n,pred,real,fronts,test_cmap);perroots.append(pr)
        # TRAIN probability gap at observed cells only.
        train_templates={cid:next(x for x in train_traces if x.context_id==cid) for cid in train_cmap};train_forces=sorted(set(train_branches.requested_force_N.astype(float)));tp=predict(models,train_cmap,train_templates,norm,tpi,cf,train_forces);tr,tf=gen.real_curves(train_branches)
        # Distances in frozen representation and raw cosine.
        tz=np.stack([train_cmap[c]["visual"] for c in sorted(train_cmap)]);traw=np.stack([train_cmap[c]["visual_raw"] for c in sorted(train_cmap)]);tcids=sorted(train_cmap)
        for cid in sorted(test_cmap):
            dz=np.linalg.norm(tz-test_cmap[cid]["visual"][None],axis=1);i=int(np.argmin(dz));a=test_cmap[cid]["visual_raw"]/(np.linalg.norm(test_cmap[cid]["visual_raw"])+1e-12);b=traw/(np.linalg.norm(traw,axis=1,keepdims=True)+1e-12);cos=1-b@a
            for model in MODELS:
                for agg in ["ENSEMBLE"]+[f"SEED_{s}" for s in SEEDS]:
                    rr=pr[(pr.context_id==cid)&(pr.model==model)&(pr.aggregation==agg)].iloc[0];distrows.append({"N":n,"context_id":cid,"root_id":test_cmap[cid]["root_id"],"model":model,"aggregation":agg,"nearest_TRAIN_context_id":tcids[i],"nearest_TRAIN_root_id":train_cmap[tcids[i]]["root_id"],"nearest_TRAIN_PCA17_distance":float(dz[i]),"mean_TRAIN_PCA17_distance":float(dz.mean()),"nearest_TRAIN_raw_cosine_distance":float(cos.min()),"probability_MAE":rr.probability_MAE,"absolute_frontier_error_N":rr.absolute_frontier_error_N,"nearest_neighbor_same_friction_band":test_cmap[cid]["friction_band"]==train_cmap[tcids[i]]["friction_band"]})
        for model in MODELS:
            for agg,col in [("ENSEMBLE","ensemble_prob")]+[(f"SEED_{s}",f"seed{s}_prob") for s in SEEDS]:
                q=pr[(pr.model==model)&(pr.aggregation==agg)];tq=tp[tp.model==model];rkey=tr.set_index(["context_id","force_N"]);obs=tq[tq.set_index(["context_id","force_N"]).index.isin(rkey.index)].copy();rr=rkey.loc[list(zip(obs.context_id,obs.force_N))];train_mae=float(np.mean(np.abs(obs[col].to_numpy()-rr.p_real.to_numpy())));base=pr[(pr.model=="Base")&(pr.aggregation==agg)].set_index("context_id")
                improved_prob=sum(float(x.probability_MAE)<float(base.loc[x.context_id].probability_MAE) for x in q.itertuples());front_comparable=[x for x in q.itertuples() if math.isfinite(float(x.absolute_frontier_error_N)) and math.isfinite(float(base.loc[x.context_id].absolute_frontier_error_N))];improved_front=sum(float(x.absolute_frontier_error_N)<float(base.loc[x.context_id].absolute_frontier_error_N) for x in front_comparable)
                ps=[];ys=[]
                for cid,rrr in real.groupby("context_id"):
                    qq=pred[(pred.model==model)&(pred.context_id==cid)&pred.force_N.isin(rrr.force_N)].sort_values("force_N");ps.extend(qq[col]);ys.extend(rrr.sort_values("force_N").p_real)
                ps=np.asarray(ps,float);ys=np.asarray(ys,float)
                evaluable=q[q.real_frontier_supported & q.finite_decision].copy();under=evaluable[evaluable.under_force==True]
                summary.append({"N":n,"model":model,"aggregation":agg,"TEST_probability_MAE":float(np.mean(abs(ps-ys))),"TEST_Brier":float(np.mean((ps-ys)**2)),"TEST_NLL":float(-np.mean(ys*np.log(np.clip(ps,1e-8,1))+(1-ys)*np.log(np.clip(1-ps,1e-8,1)))),"TEST_signed_bias":float(np.mean(ps-ys)),"TEST_frontier_MAE_N":float(q.absolute_frontier_error_N.mean()),"TEST_real_frontier_supported_rate":float(q.real_frontier_supported.mean()),"TEST_frontier_evaluable_roots":len(evaluable),"TEST_under_force_rate":float(evaluable.under_force.mean()) if len(evaluable) else math.nan,"TEST_excess_force_N":float(evaluable.excess_force_N.mean()) if len(evaluable) else math.nan,"TEST_finite_decision_rate":float(q.finite_decision.mean()),"TEST_dense_monotonic_root_fraction":float(q.dense_monotonic.mean()),"TEST_safe_to_unsafe_reversals":int(q.safe_to_unsafe_reversals.sum()),"fraction_TEST_roots_probability_improved_vs_Base":improved_prob/len(q),"TEST_frontier_comparable_roots_vs_Base":len(front_comparable),"fraction_TEST_roots_frontier_improved_vs_Base":improved_front/len(front_comparable) if front_comparable else math.nan,"TRAIN_probability_MAE":train_mae,"TRAIN_to_TEST_probability_MAE_gap":float(np.mean(abs(ps-ys)))-train_mae})
                safety.append({"N":n,"model":model,"aggregation":agg,"TEST_roots":len(q),"real_frontier_supported_roots":int(q.real_frontier_supported.sum()),"frontier_evaluable_roots":len(evaluable),"finite_decision_rate":float(q.finite_decision.mean()),"under_force_rate":float(evaluable.under_force.mean()) if len(evaluable) else math.nan,"mean_under_force_magnitude_N":float(under.under_force_magnitude_N.mean()) if len(under) else 0.0,"mean_excess_force_N":float(evaluable.excess_force_N.mean()) if len(evaluable) else math.nan,"dense_monotonic_root_fraction":float(q.dense_monotonic.mean()),"nonmonotonic_steps":int(q.nonmonotonic_steps.sum()),"safe_to_unsafe_reversals":int(q.safe_to_unsafe_reversals.sum())})
        pred.to_csv(PROJECT/f"TASK0_TEST_PREDICTIONS_N{n}.csv",index=False)
    s=pd.DataFrame(summary);r=pd.concat(perroots,ignore_index=True);sf=pd.DataFrame(safety);dd=pd.DataFrame(distrows)
    write_csv(ROOT/"TASK0_CONTEXT_LEARNING_CURVE.csv",s);write_csv(ROOT/"TASK0_CONTEXT_LEARNING_CURVE_PER_ROOT.csv",r);write_csv(ROOT/"TASK0_CONTEXT_LEARNING_CURVE_SAFETY.csv",sf);write_csv(ROOT/"TASK0_VISUAL_DISTANCE_DIAGNOSTIC.csv",dd)
    write_json(PROJECT/"TASK0_CONTEXT_EVALUATION_COMPLETE.json",{"status":"COMPLETE_NO_SELECTION_OR_CALIBRATION","rows":{"learning_curve":len(s),"per_root":len(r),"safety":len(sf),"distance":len(dd)},"TEST_commit_sha256":sha256(PROJECT/"TASK0_FROZEN_TEST_COMMIT.json")})
    print(s[s.aggregation=="ENSEMBLE"].to_string(index=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("phase",choices=["train","evaluate"]);args=ap.parse_args();train_all() if args.phase=="train" else evaluate_all()
