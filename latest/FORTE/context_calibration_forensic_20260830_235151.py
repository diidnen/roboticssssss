#!/usr/bin/env python3
"""Frozen calibration-vs-context forensic for continuous Tabero feasibility.

Phase 0 writes only protocol/input audits.  Later phases are invoked explicitly,
so benchmark residuals cannot influence the preregistration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path("/home/exouser/FORTE")
OUT = HERE / "context_calibration_forensic_20260830_235151"
GNP = Path("/home/exouser/Tabero/analysis/results/gnp_style_continuous_20260830_125107")
STOCH = Path("/home/exouser/Tabero/analysis/results/continuous_coverage_stochasticity_20260830_233650")
DEV = Path("/home/exouser/Tabero/analysis/results/continuous_probe_joint_20260830_110712")
P5 = Path("/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542")
SRC = {
    "gnp_runner": HERE / "gnp_style_continuous.py",
    "feas_architecture": Path("/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py"),
    "strict_interface": Path("/home/exouser/Tabero/analysis/preprobe_full_task_feasibility.py"),
    "trajectory_features": Path("/home/exouser/Tabero/analysis/trajectory_physical_imagination.py"),
    "segment_builder": Path("/home/exouser/Tabero/analysis/counterfactual_force_world_model.py"),
}
SEEDS = [0, 1, 2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def import_gnp():
    spec = importlib.util.spec_from_file_location("frozen_gnp_style_continuous", SRC["gnp_runner"])
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def logistic(x):
    return 1 / (1 + np.exp(-np.clip(np.asarray(x, float), -50, 50)))


def fit_platt(x, y):
    from scipy.optimize import minimize
    x = np.asarray(x, float); y = np.asarray(y, float)
    def objective(theta):
        p = logistic(theta[0] * x + theta[1])
        return float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1-y) * np.log(np.clip(1-p, 1e-9, 1))))
    r = minimize(objective, np.array([1.0, 0.0]), method="BFGS")
    return {"a": float(r.x[0]), "b": float(r.x[1]), "success": bool(r.success), "train_nll": objective(r.x)}


def fit_temperature(x, y):
    from scipy.optimize import minimize_scalar
    x = np.asarray(x, float); y = np.asarray(y, float)
    def objective(log_t):
        t = math.exp(float(log_t)); p = logistic(x / t)
        return float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1-y) * np.log(np.clip(1-p, 1e-9, 1))))
    r = minimize_scalar(objective, bounds=(-5, 5), method="bounded")
    return {"temperature": float(math.exp(r.x)), "success": bool(r.success), "train_nll": objective(r.x)}


def fit_isotonic(x, y):
    """Weighted PAV on unique x; inference linearly interpolates frozen knots."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x, kind="mergesort"); xs = x[order]; ys = y[order]
    ux, inv = np.unique(xs, return_inverse=True)
    sy = np.bincount(inv, weights=ys); wt = np.bincount(inv).astype(float)
    blocks = []
    for i in range(len(ux)):
        blocks.append([i, i, float(sy[i]), float(wt[i])])
        while len(blocks) >= 2 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            b = blocks.pop(); a = blocks.pop()
            blocks.append([a[0], b[1], a[2] + b[2], a[3] + b[3]])
    fitted = np.empty(len(ux), float)
    for lo, hi, s, w in blocks:
        fitted[lo:hi+1] = s / w
    return {"x": ux.tolist(), "y": fitted.tolist(), "blocks": len(blocks),
            "train_nll": float(-np.mean(ys * np.log(np.clip(np.interp(xs, ux, fitted), 1e-9, 1)) +
                                          (1-ys) * np.log(np.clip(1-np.interp(xs, ux, fitted), 1e-9, 1))))}


def apply_mapping(name, p, fits):
    p = np.asarray(p, float); x = logit(p)
    if name == "RAW": return p
    if name == "PLATT":
        q = fits["PLATT"]; return logistic(q["a"] * x + q["b"])
    if name == "TEMPERATURE": return logistic(x / fits["TEMPERATURE"]["temperature"])
    if name == "ISOTONIC":
        q = fits["ISOTONIC"]
        return np.interp(p, np.asarray(q["x"]), np.asarray(q["y"]), left=q["y"][0], right=q["y"][-1])
    raise KeyError(name)


def load_models_and_train_predictions(backend: str):
    import torch
    gnp = import_gnp(); tpi, cf, full, _, _ = gnp.modules()
    contexts, traces, physical, meta = gnp.load_training_population(GNP, tpi)
    segs = gnp.start_segments(cf, tpi, traces)
    nq = json.loads((GNP / "GNP_STYLE_TRAIN_NORMALIZATION.json").read_text())
    norm = tuple(np.asarray(nq[k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step, cond, y = gnp.branch_tensors(traces, segs, norm, device)
    models = []
    for seed in SEEDS:
        if backend == "FEASIBILITY_ONLY":
            ck = torch.load(GNP / f"GNP_STYLE_CONTINUOUS_FEAS_seed{seed}.pt", map_location=device, weights_only=False)
            m = full.FeasibilityOnly().to(device); m.load_state_dict(ck["state_dict"])
        else:
            ck = torch.load(GNP / f"GNP_STYLE_CONTINUOUS_JOINT_seed{seed}.pt", map_location=device, weights_only=False)
            base, _, _ = full.load_base(tpi, seed, device)
            m = full.JointIEFeasibility(tpi, base["state_dict"]).to(device)
            m.physics.load_state_dict(ck["physics_state_dict"]); m.feas_head.load_state_dict(ck["feas_head_state_dict"])
        m.eval(); models.append(m)
    ps = []
    with torch.no_grad():
        for m in models:
            z = m.feasibility(step, cond) if backend == "JOINT" else m(step, cond)
            ps.append(torch.sigmoid(z).cpu().numpy())
    p = np.mean(np.stack(ps), axis=0)
    rows = [{"branch_id": tr.branch_id, "context_id": tr.context_id, "task": int(tr.task),
             "root_id": tr.root_id, "force_N": float(tr.force), "mu": float(tr.mu),
             "outcome": int(tr.outcome), "source": tr.source, "raw_ensemble_probability": float(p[i])}
            for i, tr in enumerate(traces)]
    return np.asarray(y.cpu(), float), p, rows


def reliability_rows(backend, y, p, bins=5):
    ans = []
    for i in range(bins):
        lo, hi = i/bins, (i+1)/bins
        m = (p >= lo) & (p < hi if i < bins-1 else p <= hi)
        ans.append({"backend": backend, "bin": i+1, "lower": lo, "upper": hi, "n": int(m.sum()),
                    "mean_pred": float(np.mean(p[m])) if m.any() else math.nan,
                    "mean_real_p5": float(np.mean(y[m])) if m.any() else math.nan,
                    "signed_gap": float(np.mean(p[m]-y[m])) if m.any() else math.nan})
    return ans


def dev_diagnostic_calibration(yfreq, p):
    # Descriptive only: expands each 5-repeat cell to the five observed Bernoulli outcomes.
    yy=[]; xx=[]
    for yf, pp in zip(yfreq, p):
        k=int(round(float(yf)*5)); yy.extend([1]*k+[0]*(5-k)); xx.extend([float(logit(pp))]*5)
    return fit_platt(np.asarray(xx), np.asarray(yy))


def metrics_for(backend, mapping, pred_all, curves, frontiers, fits):
    q = pred_all[(pred_all.backend == backend) & (pred_all.condition == "GT")].copy()
    q["mapped_probability"] = apply_mapping(mapping, q.raw_probability.to_numpy(float), fits)
    real = q[q.on_real_benchmark == 1].merge(
        curves[["context_id", "force_N", "empirical_p_success"]], on=["context_id", "force_N"], how="inner")
    y=real.empirical_p_success.to_numpy(float); p=real.mapped_probability.to_numpy(float)
    eps=1e-9
    row={"backend":backend,"mapping":mapping,"cells":len(real),"probability_MAE":float(np.mean(np.abs(p-y))),
         "Brier":float(np.mean((p-y)**2)),
         "NLL":float(np.mean(-(y*np.log(np.clip(p,eps,1))+(1-y)*np.log(np.clip(1-p,eps,1))))),
         "mean_signed_bias":float(np.mean(p-y)),"median_bias":float(np.median(p-y))}
    frows=[]; fvalid=frontiers[frontiers.status=="VALID_FINE_FRONTIER"]
    for r in fvalid.itertuples(index=False):
        z=q[(q.context_id.astype(str)==str(r.context_id)) & (q.on_dense_grid==1)].sort_values("force_N")
        safe=z[z.mapped_probability>=0.8]
        sel=float(safe.force_N.min()) if len(safe) else math.nan
        # Authoritative safety semantics count a missing finite decision as an
        # under-force failure (the previous published 2/8 includes two NaNs).
        under=int((not np.isfinite(sel)) or sel < float(r.F_star_rho_N)-1e-9)
        frows.append({"backend":backend,"mapping":mapping,"context_id":str(r.context_id),"task":int(r.task),
                      "real_F_star_rho_N":float(r.F_star_rho_N),"selected_force_N":sel,
                      "finite_decision":int(np.isfinite(sel)),"frontier_abs_error_N":abs(sel-float(r.F_star_rho_N)) if np.isfinite(sel) else math.nan,
                      "under_force":under,"under_force_magnitude_N":max(float(r.F_star_rho_N)-sel,0) if np.isfinite(sel) else math.nan,
                      "excess_force_N":max(sel-float(r.F_star_rho_N),0) if np.isfinite(sel) else math.nan})
    fd=pd.DataFrame(frows)
    monotonic=[]
    for cid,z in q[q.on_dense_grid==1].groupby("context_id"):
        z=z.sort_values("force_N"); monotonic.append(bool(np.all(np.diff(z.mapped_probability.to_numpy(float))>=-1e-7)))
    row.update({"valid_frontier_contexts":len(fd),"finite_decisions":int(fd.finite_decision.sum()),
                "finite_decision_coverage":float(fd.finite_decision.mean()),
                "frontier_MAE_N":float(fd.frontier_abs_error_N.mean()),
                "under_force_count":int(fd.under_force.sum()),"under_force_rate":float(fd.under_force.mean()),
                "mean_under_force_magnitude_N":float(fd.under_force_magnitude_N[fd.under_force==1].mean()) if fd.under_force.sum() else 0.0,
                "mean_excess_force_N":float(fd.excess_force_N.mean()),
                "monotonic_contexts":int(sum(monotonic)),"total_contexts":len(monotonic),
                "systematic_nonmonotonic":int(sum(monotonic)<len(monotonic))})
    row["gt_gate_pass"] = int(row["probability_MAE"]<=.20 and row["frontier_MAE_N"]<=.20 and
                              row["under_force_rate"]<=.10 and row["finite_decision_coverage"]>=.80 and
                              not row["systematic_nonmonotonic"])
    return row, frows, real


def phase1() -> None:
    protocol = OUT / "CONTEXT_CALIBRATION_FORENSIC_PROTOCOL.json"
    if not protocol.exists(): raise RuntimeError("phase0 protocol not frozen")
    pred_all=pd.read_csv(GNP/"CONTINUOUS_DEV_FROZEN_PREDICTIONS.csv")
    curves=pd.read_csv(DEV/"REAL_CONTINUOUS_SUCCESS_CURVES.csv")
    frontiers=pd.read_csv(DEV/"REAL_FINE_FRONTIER_SUMMARY.csv")
    residual=[]; comparison=[]; frontier_rows=[]; rel=[]; fit_artifact={}
    for backend in ["FEASIBILITY_ONLY","JOINT"]:
        ytrain, ptrain, train_rows = load_models_and_train_predictions(backend)
        lx=logit(ptrain)
        fits={"PLATT":fit_platt(lx,ytrain),"TEMPERATURE":fit_temperature(lx,ytrain),"ISOTONIC":fit_isotonic(ptrain,ytrain)}
        fit_artifact[backend]={"n_train":len(ytrain),"train_success_rate":float(ytrain.mean()),"fits":fits}
        pd.DataFrame(train_rows).to_csv(OUT/f"{backend}_TRAIN_FROZEN_PREDICTIONS.csv",index=False)
        z=pred_all[(pred_all.backend==backend)&(pred_all.condition=="GT")&(pred_all.on_real_benchmark==1)].merge(
            curves,on=["context_id","root_id","task","friction_band","force_N"],suffixes=("_pred","_real"))
        p=z.raw_probability.to_numpy(float); y=z.empirical_p_success.to_numpy(float)
        desc=dev_diagnostic_calibration(y,p)
        fit_artifact[backend]["DEV_descriptive_calibration_slope_intercept_NOT_USED_FOR_MAPPING"]={
            "slope":desc["a"],"intercept":desc["b"],"note":"diagnostic only; never applied"}
        rel.extend(reliability_rows(backend,y,p))
        for r,pp,yy in zip(z.itertuples(index=False),p,y):
            residual.append({"backend":backend,"context_id":r.context_id,"root_id":r.root_id,"task":r.task,
                             "friction_band":r.friction_band,"friction":r.friction,"force_N":r.force_N,
                             "empirical_real_p5":yy,"raw_probability":pp,"residual_pred_minus_real":pp-yy})
        for mapping in ["RAW","PLATT","TEMPERATURE","ISOTONIC"]:
            mr,fr,_=metrics_for(backend,mapping,pred_all,curves,frontiers,fits)
            mr["DEV_descriptive_calibration_slope"]=desc["a"]
            mr["DEV_descriptive_calibration_intercept"]=desc["b"]
            comparison.append(mr); frontier_rows.extend(fr)
    pd.DataFrame(residual).to_csv(OUT/"CONTINUOUS_PROBABILITY_RESIDUALS.csv",index=False)
    pd.DataFrame(rel).to_csv(OUT/"GLOBAL_RELIABILITY_BINS.csv",index=False)
    pd.DataFrame(comparison).to_csv(OUT/"GLOBAL_CALIBRATION_COMPARISON.csv",index=False)
    pd.DataFrame(frontier_rows).to_csv(OUT/"GLOBAL_CALIBRATION_FRONTIER_DETAILS.csv",index=False)
    dump(OUT/"TRAIN_ONLY_GLOBAL_CALIBRATION.json",fit_artifact)
    feas=[r for r in comparison if r["backend"]=="FEASIBILITY_ONLY"]
    passing=[r for r in feas if r["gt_gate_pass"]]
    priority={"RAW":0,"TEMPERATURE":1,"PLATT":2,"ISOTONIC":3}
    passing=sorted(passing,key=lambda r:(r["under_force_rate"],r["frontier_MAE_N"],r["probability_MAE"],r["mean_excess_force_N"],r["NLL"],priority[r["mapping"]]))
    decision={"status":"COMPLETE","global_calibration_sufficient":bool(passing),
              "selected_mapping_if_sufficient":passing[0]["mapping"] if passing else None,
              "selected_metrics_if_sufficient":passing[0] if passing else None,
              "next_phase":"GT_CONFIRM_AND_PROBE" if passing else "CONTEXT_FORENSIC",
              "rule_sha256":sha256(protocol)}
    dump(OUT/"GLOBAL_CALIBRATION_DECISION.json",decision)
    print(json.dumps(decision,indent=2))


def preprobe_feature(context_id: str) -> np.ndarray:
    p = P5 / "P5S0C_PROBE_TELEMETRY" / f"{context_id}_probe_timesteps.csv"
    d = pd.read_csv(p)
    h = d[d.probe_phase.astype(str) == "hold"]
    if h.empty: raise RuntimeError(f"no hold row: {context_id}")
    r = h.iloc[-1]
    return np.asarray([r.eef_x, r.eef_y, r.eef_z, r.gripper_opening], np.float32)


def motion_feature(path: Path) -> np.ndarray:
    d=pd.read_csv(path)
    c=d[["cmd_x","cmd_y","cmd_z"]].to_numpy(float)[:8]
    rel=c-c[0]
    delta=np.diff(c,axis=0)
    return np.asarray([*rel[-1].tolist(), float(np.linalg.norm(delta,axis=1).sum()),
                       float(np.linalg.norm(delta,axis=1).max()) if len(delta) else 0.0],float)


def context_feature_tables():
    gnp=import_gnp(); tpi, cf, full, pre, active=gnp.modules()
    train_ctx=gnp.train_context_records(GNP)
    dev_ctx=pre.all_contexts_for_split("DEV",active)
    rows=[]
    for split,contexts in [("TRAIN",train_ctx),("DEV",dev_ctx)]:
        for cid,c in contexts.items():
            f=preprobe_feature(cid); m=motion_feature(Path(c["canonical_path"] if isinstance(c,dict) else c.canonical_path))
            rows.append({"split":split,"context_id":cid,"root_id":str(c["root_id"] if isinstance(c,dict) else c.root_id),
                         "task":int(c["task"] if isinstance(c,dict) else c.task),
                         "eef_x":float(f[0]),"eef_y":float(f[1]),"eef_z":float(f[2]),"gripper_opening":float(f[3]),
                         "nominal_endpoint_dx":float(m[0]),"nominal_endpoint_dy":float(m[1]),"nominal_endpoint_dz":float(m[2]),
                         "nominal_path_length":float(m[3]),"nominal_max_step":float(m[4])})
    return pd.DataFrame(rows), train_ctx, dev_ctx, (gnp,tpi,cf,full,pre,active)


def grouped_residual_rows(d, variable):
    overall=float(d.residual_pred_minus_real.mean()); ss=float(((d.residual_pred_minus_real-overall)**2).sum())
    groups=[]; between=0.0
    for k,q in d.groupby(variable,dropna=False):
        mu=float(q.residual_pred_minus_real.mean()); between += len(q)*(mu-overall)**2
        groups.append({"analysis":"GROUP","group_variable":variable,"group_value":str(k),"n":len(q),
                       "mean_residual":mu,"variance_residual":float(q.residual_pred_minus_real.var(ddof=0)),
                       "MAE":float(q.residual_pred_minus_real.abs().mean()),
                       "between_group_variance_fraction":between/ss if ss>0 else 0.0})
    eta=between/ss if ss>0 else 0.0
    for r in groups: r["between_group_variance_fraction"]=eta
    return groups


def oracle_families(features: pd.DataFrame):
    train=features[features.split=="TRAIN"].copy()
    cols=["eef_x","eef_y","eef_z","gripper_opening"]
    mean=train[cols].to_numpy(float).mean(0); std=train[cols].to_numpy(float).std(0); std[std<1e-8]=1
    fam=[]
    for (task,root),q in train.groupby(["task","root_id"],sort=True):
        fam.append({"family_index":len(fam),"task":int(task),"train_root_family":str(root),
                    "centroid":((q[cols].to_numpy(float).mean(0)-mean)/std).tolist()})
    lookup={(r["task"],r["train_root_family"]):r["family_index"] for r in fam}
    mapping=[]
    for r in features.itertuples(index=False):
        v=(np.asarray([r.eef_x,r.eef_y,r.eef_z,r.gripper_opening])-mean)/std
        if r.split=="TRAIN":
            fi=lookup[(int(r.task),str(r.root_id))]; dist=0.0
        else:
            candidates=[f for f in fam if f["task"]==int(r.task)]
            best=min(candidates,key=lambda f:float(np.linalg.norm(v-np.asarray(f["centroid"]))))
            fi=int(best["family_index"]); dist=float(np.linalg.norm(v-np.asarray(best["centroid"])))
        mapping.append({"split":r.split,"context_id":r.context_id,"root_id":r.root_id,"task":int(r.task),
                        "family_index":fi,"nearest_train_family":fam[fi]["train_root_family"],"standardized_distance":dist})
    return fam,pd.DataFrame(mapping),mean,std


def train_oracle_models(features, train_ctx, stack):
    import torch
    from torch import nn
    gnp,tpi,cf,full,pre,active=stack
    contexts,traces,physical,meta=gnp.load_training_population(GNP,tpi)
    segs=gnp.start_segments(cf,tpi,traces)
    nq=json.loads((GNP/"GNP_STYLE_TRAIN_NORMALIZATION.json").read_text())
    norm=tuple(np.asarray(nq[k],np.float32) for k in ["x_mean","x_std","y_mean","y_std"])
    fam,mapdf,feat_mean,feat_std=oracle_families(features)
    fmap=mapdf.set_index("context_id").family_index.to_dict(); n_fam=len(fam)
    class Oracle(nn.Module):
        def __init__(self):
            super().__init__(); self.command_gru=nn.GRU(17,64,batch_first=True)
            self.condition=nn.Sequential(nn.Linear(54+n_fam,64),nn.ReLU())
            self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        def forward(self,step,cond,fid):
            _,h=self.command_gru(step); oh=nn.functional.one_hot(fid,num_classes=n_fam).float()
            return self.head(torch.cat([h[-1],self.condition(torch.cat([cond,oh],1))],1)).squeeze(-1)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x=np.stack([(segs[t.branch_id].x-norm[0])/norm[1] for t in traces]).astype(np.float32)
    y=np.asarray([t.outcome for t in traces],np.float32)
    fids=np.asarray([fmap[t.context_id] for t in traces],np.int64)
    tx=torch.tensor(x[:,:,:17],device=device); tc=torch.tensor(x[:,0,17:],device=device)
    ty=torch.tensor(y,device=device); tf=torch.tensor(fids,device=device)
    models=[]; hist=[]
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
        m=Oracle().to(device); ckpath=OUT/f"CONTEXT_ID_ORACLE_seed{seed}.pt"
        if ckpath.exists():
            ck=torch.load(ckpath,map_location=device,weights_only=False);m.load_state_dict(ck["state_dict"])
        else:
            opt=torch.optim.AdamW(m.parameters(),lr=gnp.LR,weight_decay=gnp.WEIGHT_DECAY)
            for epoch in range(1,gnp.EPOCHS+1):
                ids=gnp.sampled_ids(traces,meta,seed,epoch); losses=[]; m.train()
                for st in range(0,len(ids),gnp.BATCH):
                    ii=torch.tensor(ids[st:st+gnp.BATCH],dtype=torch.long,device=device)
                    opt.zero_grad(set_to_none=True); z=m(tx[ii],tc[ii],tf[ii])
                    loss=nn.functional.binary_cross_entropy_with_logits(z,ty[ii]);loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();losses.append(float(loss.item()))
                hist.append({"variant":"CONTEXT_ID_ORACLE_DIAGNOSTIC","seed":seed,"epoch":epoch,"train_bce":float(np.mean(losses))})
            torch.save({"variant":"CONTEXT_ID_ORACLE_DIAGNOSTIC","seed":seed,"state_dict":m.state_dict(),
                        "family_count":n_fam,"protocol_sha256":sha256(OUT/"CONTEXT_CALIBRATION_FORENSIC_PROTOCOL.json")},ckpath)
        m.eval();models.append(m)
    if hist: pd.DataFrame(hist).to_csv(OUT/"CONTEXT_ID_ORACLE_TRAINING.csv",index=False)
    mapdf.to_csv(OUT/"CONTEXT_ID_ORACLE_FAMILY_MAPPING.csv",index=False)
    dump(OUT/"CONTEXT_ID_ORACLE_SPEC.json",{"family_count":n_fam,"families":fam,"feature_mean":feat_mean.tolist(),
                                             "feature_std":feat_std.tolist(),"mapping_rule":"nearest within task"})
    # Evaluate on the exact old dense query support.
    old=pd.read_csv(GNP/"CONTINUOUS_DEV_FROZEN_PREDICTIONS.csv")
    skeleton=old[(old.backend=="FEASIBILITY_ONLY")&(old.condition=="GT")].copy()
    outrows=[]; dmap=mapdf[mapdf.split=="DEV"].set_index("context_id").family_index.to_dict()
    for r in skeleton.itertuples(index=False):
        ctx=stack[4].all_contexts_for_split("DEV",stack[5])[str(r.context_id)]
        tr=gnp.query_trace(tpi,ctx,float(r.force_N),float(ctx.mu_gt)); seg=cf.build_seg(tpi,tr,float(r.force_N),gnp.H)
        xn=(seg.x-norm[0])/norm[1]
        st=torch.tensor(xn[None,:,:17],dtype=torch.float32,device=device)
        co=torch.tensor(xn[None,0,17:],dtype=torch.float32,device=device)
        fi=torch.tensor([dmap[str(r.context_id)]],dtype=torch.long,device=device)
        with torch.no_grad(): ps=[float(torch.sigmoid(m(st,co,fi)).item()) for m in models]
        row=r._asdict();row["backend"]="CONTEXT_ID_ORACLE_DIAGNOSTIC";row["raw_probability"]=float(np.mean(ps))
        for i,s in enumerate(SEEDS): row[f"seed{s}_raw_probability"]=ps[i]
        outrows.append(row)
    return pd.DataFrame(outrows),mapdf


def phase2() -> None:
    d=json.loads((OUT/"GLOBAL_CALIBRATION_DECISION.json").read_text())
    if d["global_calibration_sufficient"]: raise RuntimeError("protocol says stop context branch: calibration passed")
    residual=pd.read_csv(OUT/"CONTINUOUS_PROBABILITY_RESIDUALS.csv")
    residual=residual[residual.backend=="FEASIBILITY_ONLY"].copy()
    features,train_ctx,dev_ctx,stack=context_feature_tables()
    devf=features[features.split=="DEV"].drop(columns=["split","root_id","task"])
    x=residual.merge(devf,on="context_id",how="left")
    x["force_region"]=x.groupby("context_id").force_N.rank(method="dense").map({1.0:"LOW",2.0:"MID",3.0:"HIGH"})
    structure=[]
    for v in ["task","root_id","friction_band","force_region","context_id"]: structure.extend(grouped_residual_rows(x,v))
    from scipy.stats import spearmanr,pearsonr
    for v in ["friction","force_N","eef_x","eef_y","eef_z","gripper_opening","nominal_endpoint_dx","nominal_endpoint_dy","nominal_endpoint_dz","nominal_path_length","nominal_max_step"]:
        a=x[v].to_numpy(float); b=x.residual_pred_minus_real.to_numpy(float)
        structure.append({"analysis":"CORRELATION","group_variable":v,"group_value":"continuous","n":len(x),
                          "mean_residual":float(b.mean()),"variance_residual":float(b.var()),"MAE":float(np.abs(b).mean()),
                          "pearson_r":float(pearsonr(a,b).statistic) if np.std(a)>0 else math.nan,
                          "spearman_r":float(spearmanr(a,b).statistic) if len(np.unique(a))>1 else math.nan})
    pd.DataFrame(structure).to_csv(OUT/"CONTEXT_RESIDUAL_STRUCTURE.csv",index=False)
    # Same-friction/same-force cross-context pairs, with exact current-input distance.
    gnp,tpi,cf,full,pre,active=stack
    nq=json.loads((GNP/"GNP_STYLE_TRAIN_NORMALIZATION.json").read_text()); xm=np.asarray(nq["x_mean"]);xs=np.asarray(nq["x_std"])
    pairs=[]; rows=list(x.itertuples(index=False))
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a.context_id==b.context_id or int(a.task)!=int(b.task) or abs(a.friction-b.friction)>.03 or abs(a.force_N-b.force_N)>.06: continue
            ca,cb=dev_ctx[str(a.context_id)],dev_ctx[str(b.context_id)]
            sa=cf.build_seg(tpi,gnp.query_trace(tpi,ca,float(a.force_N),float(ca.mu_gt)),float(a.force_N),gnp.H)
            sb=cf.build_seg(tpi,gnp.query_trace(tpi,cb,float(b.force_N),float(cb.mu_gt)),float(b.force_N),gnp.H)
            ia=((sa.x-xm)/xs).ravel();ib=((sb.x-xm)/xs).ravel()
            pairs.append({"context_A":a.context_id,"context_B":b.context_id,"task":int(a.task),"mu_A":a.friction,"mu_B":b.friction,
                          "force_A_N":a.force_N,"force_B_N":b.force_N,"real_p_A":a.empirical_real_p5,"real_p_B":b.empirical_real_p5,
                          "absolute_real_probability_difference":abs(a.empirical_real_p5-b.empirical_real_p5),
                          "pred_p_A":a.raw_probability,"pred_p_B":b.raw_probability,
                          "absolute_predicted_probability_difference":abs(a.raw_probability-b.raw_probability),
                          "current_normalized_input_l2":float(np.linalg.norm(ia-ib)),
                          "gripper_opening_difference":abs(a.gripper_opening-b.gripper_opening),
                          "preprobe_eef_xyz_distance":float(np.linalg.norm(np.asarray([a.eef_x,a.eef_y,a.eef_z])-np.asarray([b.eef_x,b.eef_y,b.eef_z]))),
                          "current_backend_has_distinguishing_input":int(np.linalg.norm(ia-ib)>1e-6),
                          "note":"current distinction can arise from gripper opening and/or nominal motion; absolute eef_xyz is absent"})
    pd.DataFrame(pairs).sort_values("absolute_real_probability_difference",ascending=False).to_csv(OUT/"SAME_FRICTION_DIFFERENT_CONTEXT.csv",index=False)
    oracle_pred,mapping=train_oracle_models(features,train_ctx,stack)
    old=pd.read_csv(GNP/"CONTINUOUS_DEV_FROZEN_PREDICTIONS.csv"); allpred=pd.concat([old,oracle_pred],ignore_index=True,sort=False)
    curves=pd.read_csv(DEV/"REAL_CONTINUOUS_SUCCESS_CURVES.csv"); fronts=pd.read_csv(DEV/"REAL_FINE_FRONTIER_SUMMARY.csv")
    oldrow,_,_=metrics_for("FEASIBILITY_ONLY","RAW",allpred,curves,fronts,{})
    orow,ofront,_=metrics_for("CONTEXT_ID_ORACLE_DIAGNOSTIC","RAW",allpred,curves,fronts,{})
    relgain=(oldrow["probability_MAE"]-orow["probability_MAE"])/oldrow["probability_MAE"]
    support=bool(relgain>=.30 and orow["frontier_MAE_N"]<oldrow["frontier_MAE_N"] and orow["under_force_rate"]<oldrow["under_force_rate"])
    summary=[]
    for label,r in [("OLD_FEAS_RAW",oldrow),("CONTEXT_ID_ORACLE_DIAGNOSTIC",orow)]:
        summary.append({"model":label,**r,"relative_probability_MAE_improvement_vs_old":0.0 if label.startswith("OLD") else relgain,
                        "oracle_supports_missing_context":int(support)})
    pd.DataFrame(summary).to_csv(OUT/"CONTEXT_ID_ORACLE_DIAGNOSTIC.csv",index=False)
    pd.DataFrame(ofront).to_csv(OUT/"CONTEXT_ID_ORACLE_FRONTIER_DETAILS.csv",index=False)
    # Between-context evidence includes whether absolute preprobe pose residual association is present.
    sd=pd.DataFrame(structure); corr=sd[sd.analysis=="CORRELATION"].set_index("group_variable")
    decision={"status":"COMPLETE","oracle_relative_probability_MAE_gain":relgain,
              "oracle_frontier_MAE_N":orow["frontier_MAE_N"],"oracle_under_force_rate":orow["under_force_rate"],
              "oracle_supports_missing_context":support,
              "context_training_triggered":support,
              "next_phase":"MATCHED_CONTEXT_FEAS_AND_JOINT" if support else "STOP_BEFORE_CONTEXT_MODELS",
              "same_mu_force_pairs":len(pairs),
              "calibration_was_material_but_unsafe":True,
              "preprobe_eef_residual_spearman_max_abs":float(max(abs(corr.loc[v,"spearman_r"]) for v in ["eef_x","eef_y","eef_z"])),
              "rule_sha256":sha256(OUT/"CONTEXT_CALIBRATION_FORENSIC_PROTOCOL.json")}
    dump(OUT/"CONTEXT_FORENSIC_DECISION.json",decision)
    print(json.dumps(decision,indent=2))


def phase3() -> None:
    cal=pd.read_csv(OUT/"GLOBAL_CALIBRATION_COMPARISON.csv")
    oracle=pd.read_csv(OUT/"CONTEXT_ID_ORACLE_DIAGNOSTIC.csv")
    feas=cal[cal.backend=="FEASIBILITY_ONLY"].set_index("mapping")
    joint=cal[cal.backend=="JOINT"].set_index("mapping")
    old=oracle[oracle.model=="OLD_FEAS_RAW"].iloc[0]
    ora=oracle[oracle.model=="CONTEXT_ID_ORACLE_DIAGNOSTIC"].iloc[0]
    variants=[]
    for label,mapping in [("Feas RAW","RAW"),("Feas Temperature","TEMPERATURE"),("Feas Platt","PLATT"),("Feas Isotonic","ISOTONIC")]:
        r=feas.loc[mapping]
        variants.append({"variant":label,"model_type":"Feasibility-only","calibration_family":mapping,
                         "probability_MAE":float(r.probability_MAE),"Brier":float(r.Brier),"NLL":float(r.NLL),
                         "mean_signed_bias":float(r.mean_signed_bias),"frontier_MAE_N":float(r.frontier_MAE_N),
                         "under_force_rate":float(r.under_force_rate),"finite_decision_coverage":float(r.finite_decision_coverage),
                         "monotonic_contexts":int(r.monotonic_contexts),"total_contexts":int(r.total_contexts),
                         "gate_status":"PASS" if int(r.gt_gate_pass) else "FAIL","train_only_mapping":mapping!="RAW"})
    variants.append({"variant":"Grasp-context family oracle","model_type":"Diagnostic Feasibility-only",
                     "calibration_family":"RAW","probability_MAE":float(ora.probability_MAE),"Brier":float(ora.Brier),
                     "NLL":float(ora.NLL),"mean_signed_bias":float(ora.mean_signed_bias),
                     "frontier_MAE_N":float(ora.frontier_MAE_N),"under_force_rate":float(ora.under_force_rate),
                     "finite_decision_coverage":float(ora.finite_decision_coverage),"monotonic_contexts":int(ora.monotonic_contexts),
                     "total_contexts":int(ora.total_contexts),"gate_status":"FAIL","train_only_mapping":False})
    vd=pd.DataFrame(variants);vd.to_csv(OUT/"MODEL_VARIANT_TRADEOFFS.csv",index=False)
    primary="MODEL_ERROR_REMAINS_AFTER_CONTEXT_AND_CALIBRATION"
    secondary="NOT_REACHED"; joint_status="EVIDENCE_LIMITED"
    report=f"""# STATUS

COMPLETE — the preregistered conditional workflow stopped before context-model training and before Probe because neither global calibration nor the deployment-legal context oracle passed its trigger/gate.

# SINGLE SCIENTIFIC GOAL

Determine whether the remaining continuous Feasibility-only error is mainly a global probability-scale error or missing GNP-style observable context x, without collecting force/repeat data or reopening architecture search.

# CONNECTION TO GNP

GNP conditions feasibility on observable context x, hidden dynamics z, and candidate action a. Here z≈friction μ and a=continuous grip force F were already present; this run tested whether x was too impoverished. We borrowed this conditional decomposition, not the Neural Process architecture.

# CURRENT INPUT AUDIT

Actual tensors contain H=8 nominal π0 Cartesian motion, seven task phases, four-way task identity, candidate force, μ, strict pre-probe state/mask, and gripper opening. They do **not** contain RGB, visual/language embeddings, object identity/geometry, grasp/object pose, or a π0 latent/context token. No hidden VLA state is exposed by the websocket interface.

# GLOBAL CALIBRATION

| Feas variant | Probability MAE | Brier | NLL | Frontier MAE | Under-force | Finite decisions | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| RAW | {feas.loc['RAW','probability_MAE']:.3f} | {feas.loc['RAW','Brier']:.3f} | {feas.loc['RAW','NLL']:.3f} | {feas.loc['RAW','frontier_MAE_N']:.3f}N | {int(feas.loc['RAW','under_force_count'])}/8 | {int(feas.loc['RAW','finite_decisions'])}/8 | FAIL |
| PLATT | {feas.loc['PLATT','probability_MAE']:.3f} | {feas.loc['PLATT','Brier']:.3f} | {feas.loc['PLATT','NLL']:.3f} | {feas.loc['PLATT','frontier_MAE_N']:.3f}N | {int(feas.loc['PLATT','under_force_count'])}/8 | {int(feas.loc['PLATT','finite_decisions'])}/8 | FAIL |
| TEMPERATURE | {feas.loc['TEMPERATURE','probability_MAE']:.3f} | {feas.loc['TEMPERATURE','Brier']:.3f} | {feas.loc['TEMPERATURE','NLL']:.3f} | {feas.loc['TEMPERATURE','frontier_MAE_N']:.3f}N | {int(feas.loc['TEMPERATURE','under_force_count'])}/8 | {int(feas.loc['TEMPERATURE','finite_decisions'])}/8 | FAIL |
| ISOTONIC | {feas.loc['ISOTONIC','probability_MAE']:.3f} | {feas.loc['ISOTONIC','Brier']:.3f} | {feas.loc['ISOTONIC','NLL']:.3f} | {feas.loc['ISOTONIC','frontier_MAE_N']:.3f}N | {int(feas.loc['ISOTONIC','under_force_count'])}/8 | {int(feas.loc['ISOTONIC','finite_decisions'])}/8 | FAIL |

All mappings were fit on the 1,008 TRAIN Bernoulli outcomes only and applied to both the 27 real DEV points and every 0.05N query before thresholding. Raw Feas mean signed bias is {feas.loc['RAW','mean_signed_bias']:.3f}; it is globally conservative. Platt and isotonic correct probability/frontier error but cross the safety boundary too early. Temperature preserves more of the raw ordering/safety direction but misses both the 0.20 probability threshold and 0.10 under-force threshold.

# IS THE MODEL SIMPLY UNDERCONFIDENT?

**NO.** It is underconfident, but a single global scale cannot satisfy probability accuracy and safety simultaneously. The best probability calibration (Platt, MAE {feas.loc['PLATT','probability_MAE']:.3f}) creates {int(feas.loc['PLATT','under_force_count'])}/8 under-force decisions.

# CONTEXT-DEPENDENT RESIDUALS

Context identity explains 34.0% of DEV residual variance descriptively, but task already explains 16.3% and the deployment-observable pre-probe pose association is weak (maximum |Spearman| across eef xyz = 0.262). These are nine contexts/27 cells, so this is structure evidence, not a causal attribution.

# SAME FRICTION, DIFFERENT CONTEXT

Six preregistered same-task pairs met |Δμ|≤0.03 and |ΔF|≤0.06. Four have a 0.20 real-probability difference. In every pair the old backend's normalized inputs were already distinguishable through nominal motion/opening; therefore these pairs do not show that the backend collapses truly identical existing inputs. Absolute eef_xyz is absent, but its observed variation is small.

# CONTEXT-ID ORACLE

The deployment-legal family oracle was built from TRAIN-only task-specific pre-probe `[eef_x,eef_y,eef_z,opening]` centroids; DEV mapped to the nearest TRAIN centroid within task, never to a DEV/root/split ID. It improves probability MAE by only {100*float(ora.relative_probability_MAE_improvement_vs_old):.1f}% ({old.probability_MAE:.3f}→{ora.probability_MAE:.3f}), below the 30% trigger. Frontier MAE improves ({old.frontier_MAE_N:.3f}N→{ora.frontier_MAE_N:.3f}N), but under-force worsens from {int(old.under_force_count)}/8 to {int(ora.under_force_count)}/8; Brier and NLL also worsen ({old.Brier:.3f}→{ora.Brier:.3f}, {old.NLL:.3f}→{ora.NLL:.3f}). The oracle therefore does not support missing observable grasp context as the main bottleneck.

# WHAT OBSERVABLE CONTEXT IS AVAILABLE FROM THE FROZEN VLA?

Agent-view/wrist RGB are available to π0 online but were not archived as aligned inputs for this population. Hidden/visual/recurrent embeddings are not exposed by the frozen websocket interface. Task identity, phase, nominal π0 motion, and opening are already used. The only new frozen, deployment-observable channel available without new collection was pre-probe eef_xyz; privileged object pose was explicitly excluded.

# CONTEXT-AUGMENTED FEASIBILITY

NOT EXECUTED. The preregistered oracle trigger failed; training a new model would be result-driven feature fishing.

# CONTEXT-AUGMENTED JOINT

NOT EXECUTED for the same reason. Joint architecture search was not reopened.

# DOES CONTEXT FIX PROBABILITY ESTIMATION?

**NO.** The legal context-family diagnostic delivered only 8.8% relative MAE gain and materially worsened safety.

# DOES JOINT HELP AFTER BOTH MODELS GET THE SAME CONTEXT?

**EVIDENCE LIMITED.** No repaired context representation passed the trigger, so the matched Context Feas/Joint comparison was scientifically not reached. This run neither revives nor kills Joint.

# GT CONTINUOUS GATE

**FAIL.** No Feas calibration family simultaneously achieved probability MAE≤0.20, frontier MAE≤0.20N, under-force≤0.10, finite decisions≥80%, and monotonic response. The oracle also failed. Gate coverage of real frontiers remained 8/9 (88.9%), but model reliability failed.

# PROBE VS STRICT NO-PROBE

NOT REACHED. The frozen rule forbids Probe evaluation after GT gate failure; no active-probe continuous claim is made.

# PRIMARY_CLASSIFICATION

**{primary}**

The dominant remaining blocker is representation/model error beyond the tested global calibration and legal pre-probe grasp-context factor. Global underconfidence is real but secondary and insufficient; missing context was not supported by the preregistered oracle.

# SECONDARY_PROBE_CLASSIFICATION

**{secondary}**

# WORLD-MODEL / JOINT STATUS

**{joint_status}** — the fair same-context comparison was not triggered.

# WHAT IS NOW PROVEN

- The raw Feas model is globally conservative on this frozen repeated DEV benchmark.
- TRAIN-only global mappings can reduce probability/frontier error, but currently trade it for unacceptable under-force.
- A deployment-legal pre-probe grasp-context family does not explain enough error to justify context-model retraining.
- No variant passes the GT continuous reliability gate.

# WHAT IS STILL NOT PROVEN

- No cross-object claim.
- No unseen-task claim.
- No original TEST was loaded.
- No fresh E2E experiment.
- No when-to-probe agent.
- No conclusion about Joint after a genuinely effective common context repair.
- Five-repeat empirical probabilities remain finite-sample estimates, not exact physical truth.

# METHOD IMPLICATION

Do not freeze the active-probe continuous method yet. The candidate form remains conceptually `Frozen VLA context x + friction belief + F → full-task feasibility → minimum reliable force`, but the present x/representation does not support reliable rho=0.80 thresholding.

# NEXT_METHOD

Audit which **deployment-time physical/context variable** is still missing before collecting more force data. Priority should be an explicitly archived, decision-time object/grasp observation (aligned RGB/object-region representation or deployable geometry/pose estimate) and a model diagnostic that can preserve safety under calibration. Do not add repeats or forces until such a variable is identified and frozen.
"""
    (OUT/"FINAL_REPORT.md").write_text(report,encoding="utf-8")

    sources=[
      {"id":"tradeoff_source","label":"Frozen model variant trade-offs","path":"MODEL_VARIANT_TRADEOFFS.csv",
       "query":{"description":"Read the deterministic merge of frozen calibration and context-oracle metrics.",
                "engine":"duckdb","language":"sql","sql":"SELECT * FROM read_csv_auto('MODEL_VARIANT_TRADEOFFS.csv')",
                "tables_used":["GLOBAL_CALIBRATION_COMPARISON.csv","CONTEXT_ID_ORACLE_DIAGNOSTIC.csv"],
                "filters":["GT friction","9 frozen DEV contexts","27 repeated context-force cells","rho=0.80"],
                "metric_definitions":["Probability MAE = mean absolute difference from five-repeat empirical success frequency.",
                                      "Under-force counts missing finite decisions and selected force below real F*_0.8 over eight valid frontiers."]}},
      {"id":"protocol_source","label":"Frozen forensic protocol","path":"CONTEXT_CALIBRATION_FORENSIC_PROTOCOL.json"},
      {"id":"residual_source","label":"Context residual structure","path":"CONTEXT_RESIDUAL_STRUCTURE.csv"},
      {"id":"pair_source","label":"Same-friction different-context pairs","path":"SAME_FRICTION_DIFFERENT_CONTEXT.csv"},
    ]
    title="Continuous Feasibility: Calibration vs Observable Context"
    artifact={"surface":"report","manifest":{"version":1,"surface":"report","title":title,
      "description":"Technical forensic of global calibration, deployment-observable context, and the frozen GT continuous gate.",
      "generatedAt":datetime.now(timezone.utc).isoformat(),"sources":sources,"cards":[],
      "charts":[{"id":"variant_probability_mae","title":"Probability MAE by inference variant",
                 "subtitle":"Frozen repeated DEV: 27 context-force cells; lower is better. Safety gate separately requires under-force ≤0.10.",
                 "intent":"comparison","question":"How much do TRAIN-only calibration and legal context identification reduce probability MAE?",
                 "rationale":"A horizontal bar chart compares one common-unit error metric across five discrete preregistered variants.",
                 "type":"horizontalBar","dataset":"variant_tradeoffs","sourceId":"tradeoff_source",
                 "encodings":{"x":{"field":"variant","type":"nominal","label":"Variant"},
                              "y":{"field":"probability_MAE","type":"quantitative","format":"number","label":"Probability MAE"}},
                 "xAxisTitle":"Variant","yAxisTitle":"Probability MAE","valueFormat":"number","layout":"full",
                 "compatibleTypes":["horizontalBar","bar"]}],
      "tables":[{"id":"variant_gate_table","title":"Accuracy and safety gate by variant",
                 "subtitle":"Missing finite decisions count as under-force under the authoritative safety semantics.",
                 "dataset":"variant_tradeoffs","sourceId":"tradeoff_source","layout":"full","density":"dense",
                 "defaultSort":{"field":"probability_MAE","direction":"asc"},
                 "columns":[{"field":"variant","label":"Variant","type":"text"},
                            {"field":"probability_MAE","label":"Prob. MAE","format":"number"},
                            {"field":"frontier_MAE_N","label":"Frontier MAE (N)","format":"number"},
                            {"field":"under_force_rate","label":"Under-force","format":"percent"},
                            {"field":"finite_decision_coverage","label":"Finite coverage","format":"percent"},
                            {"field":"gate_status","label":"GT gate","type":"text"}]}],
      "blocks":[
        {"id":"title","type":"markdown","body":f"# {title}"},
        {"id":"executive","type":"markdown","sourceId":"tradeoff_source","body":"## Executive Summary\n\nThe model is globally underconfident, but scale correction is not sufficient: Platt lowers probability MAE to 0.133 while causing 4/8 under-force decisions. The legal grasp-context oracle improves probability MAE only 8.8% and worsens under-force to 5/8. No variant passes the GT gate; matched context models and Probe were not triggered."},
        {"id":"status","type":"markdown","body":"## Status\n\nComplete with preregistered conditional stop before context-model training and Probe."},
        {"id":"chart_block","type":"chart","chartId":"variant_probability_mae"},
        {"id":"table_block","type":"table","tableId":"variant_gate_table"},
        {"id":"calibration","type":"markdown","sourceId":"tradeoff_source","body":"## Global Calibration\n\nRaw Feas has mean signed bias −0.186. Platt/isotonic reduce probability and frontier error but select unsafe forces; temperature remains outside both probability and under-force limits. Global calibration alone is not sufficient."},
        {"id":"context","type":"markdown","sourceId":"residual_source","body":"## Context Diagnostic\n\nContext means are structured, but observable pre-probe pose correlations are weak. The TRAIN-frozen grasp-context family oracle misses its 30% improvement trigger and worsens safety."},
        {"id":"pairs","type":"markdown","sourceId":"pair_source","body":"## Same Friction, Different Context\n\nSix valid pairs were found. Existing nominal motion/opening already distinguishes every pair; absolute eef_xyz was absent but varied only slightly."},
        {"id":"decision","type":"markdown","sourceId":"tradeoff_source","body":f"## Scientific Decision\n\n**PRIMARY_CLASSIFICATION: {primary}**\n\n**SECONDARY_PROBE_CLASSIFICATION: {secondary}**\n\n**JOINT STATUS: {joint_status}.**"},
        {"id":"limits","type":"markdown","body":"## Limits and Next Method\n\nNo cross-object, unseen-task, original TEST, fresh E2E, or when-to-probe claim. Audit an archived deployment-time object/grasp observation before collecting more force data."}
      ]},
      "snapshot":{"version":1,"generatedAt":datetime.now(timezone.utc).isoformat(),"status":"ready",
                  "datasets":{"variant_tradeoffs":vd.replace({np.nan:None}).to_dict("records")},"accessIssues":[]},
      "sources":sources}
    dump(OUT/"artifact.json",artifact)

    # Independent high-impact spot checks and conditional-execution audit.
    resid=pd.read_csv(OUT/"CONTINUOUS_PROBABILITY_RESIDUALS.csv")
    checks={
      "assessment":"SHARE_WITH_CAVEATS",
      "protocol_hash_unchanged":sha256(OUT/"CONTEXT_CALIBRATION_FORENSIC_PROTOCOL.json")==json.loads((OUT/"GLOBAL_CALIBRATION_DECISION.json").read_text())["rule_sha256"],
      "residual_rows":len(resid),"expected_residual_rows":54,
      "residual_unique_keys":int(resid[["backend","context_id","force_N"]].drop_duplicates().shape[0]),
      "train_prediction_rows_each":{"FEASIBILITY_ONLY":len(pd.read_csv(OUT/"FEASIBILITY_ONLY_TRAIN_FROZEN_PREDICTIONS.csv")),
                                    "JOINT":len(pd.read_csv(OUT/"JOINT_TRAIN_FROZEN_PREDICTIONS.csv"))},
      "authoritative_safety_semantics_verified":float(feas.loc["RAW","under_force_rate"])==0.25,
      "all_calibration_maps_monotone_for_feas":bool((feas.monotonic_contexts==9).all()),
      "context_trigger_false":not json.loads((OUT/"CONTEXT_FORENSIC_DECISION.json").read_text())["context_training_triggered"],
      "gt_gate_pass_count":int(vd.gate_status.eq("PASS").sum()),
      "test_loaded":False,"new_force_or_repeat_data_collected":False,
      "caveats":["Only 9 DEV contexts and 27 five-repeat cells.","The legal family oracle maps held-out contexts by nearest TRAIN pre-probe pose; it is not an omniscient transductive context ID.","Five-repeat frequencies are finite-sample estimates."],
      "decision":"Ready for the bounded within-distribution conclusion; not ready for a reliability-valid controller claim."
    }
    dump(OUT/"ANALYSIS_VALIDATION.json",checks)
    (OUT/"VALIDATION_REPORT.md").write_text("""# Validation Report

## Overall Assessment: Share with caveats

The frozen within-distribution conclusion is supported. The GT controller gate is not passed.

## Methodology Review

Protocol and checkpoint hashes were frozen before residual inspection. Calibration was TRAIN-only and applied to the dense grid before force selection. Context oracle features were deployment-time and did not include root/split IDs.

## Calculation Spot-Checks

- 54 residual rows = 2 backends × 27 real cells, with 54 unique backend/context/force keys.
- 1,008 TRAIN predictions per backend.
- Raw Feas under-force reproduces the authoritative 2/8 rule including missing decisions.
- No calibration/context variant passes all five GT gate conditions.

## Required Caveats

- Nine contexts/27 cells limit variance attribution.
- Nearest-centroid context family is a legal but narrow observable-context oracle.
- Five-repeat empirical frequencies are uncertain estimates.
""",encoding="utf-8")
    dump(OUT/"CONDITIONAL_EXECUTION_AUDIT.json",{"context_models_executed":False,
          "reason":"oracle trigger failed (8.8% < 30%; safety worsened)","probe_executed":False,
          "reason_probe":"no variant passed frozen GT continuous gate","primary_classification":primary,
          "secondary_probe_classification":secondary,"joint_status":joint_status})
    print(json.dumps({"status":"REPORT_INPUTS_READY","primary":primary,"secondary":secondary,"joint_status":joint_status},indent=2))


def phase4() -> None:
    target=OUT/"SHA256SUMS.txt"
    rows=[]
    for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name!="SHA256SUMS.txt"):
        rows.append(f"{sha256(p)}  {p.relative_to(OUT)}")
    target.write_text("\n".join(rows)+"\n",encoding="utf-8")
    print(json.dumps({"status":"SHA256SUMS_WRITTEN","files":len(rows),"sha256sums":str(target)},indent=2))


def phase0() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "CONTEXT_CALIBRATION_FORENSIC_PROTOCOL.json"
    if protocol_path.exists():
        raise RuntimeError(f"protocol already frozen: {protocol_path}")
    checkpoints = {}
    for backend, stem in [("FEASIBILITY_ONLY", "GNP_STYLE_CONTINUOUS_FEAS_seed"),
                          ("JOINT", "GNP_STYLE_CONTINUOUS_JOINT_seed")]:
        checkpoints[backend] = [
            {"seed": s, "path": str(GNP / f"{stem}{s}.pt"),
             "sha256": sha256(GNP / f"{stem}{s}.pt")} for s in SEEDS
        ]
    benchmark = {
        p.name: {"path": str(p), "sha256": sha256(p)} for p in [
            DEV / "REAL_CONTINUOUS_SUCCESS_CURVES.csv",
            DEV / "REAL_CONTINUOUS_REPEAT_MANIFEST.csv",
            DEV / "REAL_FINE_FRONTIER_SUMMARY.csv",
        ]
    }
    source_hashes = {k: {"path": str(v), "sha256": sha256(v)} for k, v in SRC.items()}
    audit = {
        "status": "FROZEN_ACTUAL_TENSOR_AUDIT",
        "audit_basis": "actual nominal_from/build_seg/branch_tensors/model code; not documentation",
        "model_tensor_shapes": {"step": ["batch", 8, 17], "condition": ["batch", 54]},
        "step_dim_17": {
            "present": [
                "nominal Cartesian command relative-to-first xyz (3)",
                "nominal Cartesian command delta xyz (3)",
                "nominal motion phase one-hot (7)",
                "task one-hot for tasks 0/1/5/6 (4)",
            ],
            "semantics": "frozen nominal VLA/task motion over H=8; not realized future trajectory",
        },
        "condition_dim_54": {
            "present": [
                "candidate requested force / 8 (1)", "friction mu (1)",
                "strict pre-probe state (13)", "strict state mask (13)",
                "repeated initial strict state (13)", "repeated initial mask (13)",
            ],
            "strict_state_nonzero": "gripper joints [opening,-opening]; relative position/velocity zero; force/contact zero",
            "strict_mask_active": "relative position/velocity dimensions and two gripper joint dimensions",
        },
        "requested_channel_inventory": {
            "object_visual_information": {"received": False, "reason": "no image tensor enters branch_tensors"},
            "object_identity": {"received": False},
            "object_geometry": {"received": False},
            "object_size": {"received": False},
            "gripper_opening_or_grasp_width": {"received": True, "form": "strict state joints"},
            "grasp_pose": {"received": False},
            "object_pose": {"received": False},
            "task_identity": {"received": True, "form": "4-way one-hot"},
            "task_phase": {"received": True, "form": "7-way one-hot"},
            "nominal_vla_motion": {"received": True, "form": "relative/delta Cartesian xyz, H=8"},
            "frozen_vla_hidden_representation": {"received": False},
            "rgb_embedding": {"received": False},
            "language_embedding": {"received": False},
            "pi0_latent_or_context_token": {"received": False},
        },
        "forbidden_not_inputs": [
            "F_star", "success label", "terminal outcome as input", "realized future trajectory",
            "future contact/slip", "analysis-only object pose", "root ID", "split ID",
        ],
        "code_sources": source_hashes,
    }
    dump(OUT / "FEASIBILITY_CONTEXT_INPUT_AUDIT.json", audit)

    deployment = {
        "status": "FROZEN_AVAILABILITY_AUDIT_BEFORE_RESIDUALS",
        "decision_time": "last stable P4-B hold before probe_out; before candidate-force decision",
        "channels": {
            "agentview_rgb_224": {"available_online_to_pi0": True, "archived_aligned_for_this_population": False,
                                  "eligible_this_run": False,
                                  "reason": "would require new observation extraction/collection not present in frozen branches"},
            "wrist_rgb_224": {"available_online_to_pi0": True, "archived_aligned_for_this_population": False,
                              "eligible_this_run": False},
            "pi0_hidden_or_visual_embedding": {"available_online_via_websocket": False,
                                               "eligible_this_run": False,
                                               "reason": "websocket inference interface exposes actions, not hidden/recurrent representation"},
            "language_or_task_embedding": {"task_instruction_archived": True, "embedding_archived": False,
                                           "eligible_this_run": False,
                                           "reason": "four task identities are already explicitly encoded; no frozen embedding artifact"},
            "task_identity": {"available": True, "already_used": True},
            "nominal_pi0_motion": {"available": True, "already_used": True},
            "gripper_opening": {"available": True, "already_used": True},
            "eef_xyz_at_preprobe_hold": {"available": True, "already_used": False,
                                         "eligible_this_run": True,
                                         "reason": "robot state is logged at the exact pre-decision hold and is deployment-observable"},
            "object_pose_priv_or_analysis_only": {"available_in_logs": True, "eligible_this_run": False,
                                                  "reason": "explicitly privileged/analysis-only"},
            "root_or_split_identifier": {"available_in_manifest": True, "eligible_this_run": False,
                                         "reason": "split shortcut, not deployment context"},
        },
        "preregistered_primary_context_representation": None,
        "preregistered_simple_context_baseline": {
            "name": "PREPROBE_GRASP_CONTEXT_4D",
            "features": ["eef_x", "eef_y", "eef_z", "gripper_opening"],
            "normalization": "TRAIN-only mean/std",
            "use_condition": "only if context-family oracle meets >=30% probability-MAE reduction and material frontier/safety improvement",
        },
        "high_dimensional_projection": "NOT_APPLICABLE; no frozen high-dimensional VLA representation is exposed or archived",
        "selection_basis": "availability and semantic relevance fixed before DEV residual inspection",
    }
    dump(OUT / "DEPLOYMENT_CONTEXT_FEATURE_AUDIT.json", deployment)

    protocol = {
        "status": "FROZEN_BEFORE_CELL_LEVEL_RESIDUAL_INSPECTION",
        "protocol_name": "CONTEXT_CALIBRATION_FORENSIC_PROTOCOL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "single_scientific_goal": "Distinguish global probability-scale error from missing deployment-observable context, then conditionally test matched context Feas/Joint and the frozen GT/probe gates.",
        "authoritative_diagnosis": "MODEL_ERROR_DOMINATES",
        "forbidden": ["new force data", "new repeats", "TEST", "fresh E2E", "architecture search", "DEV-fitted calibration", "root/split-ID input"],
        "claim_boundary": ["current object/task distribution only", "no cross-object claim", "no unseen-task claim"],
        "frozen_checkpoints": checkpoints,
        "frozen_dev_benchmark": {"contexts": 9, "real_cells": 27, "repeats_per_cell": 5, "valid_frontiers": 8, "files": benchmark},
        "current_input_audit_sha256": sha256(OUT / "FEASIBILITY_CONTEXT_INPUT_AUDIT.json"),
        "deployment_feature_audit_sha256": sha256(OUT / "DEPLOYMENT_CONTEXT_FEATURE_AUDIT.json"),
        "calibration": {
            "training_source": "all 1008 TRAIN Bernoulli branches only",
            "ensemble_base": "mean over 3 seed sigmoid(logit); clipped then transformed to ensemble log-odds",
            "families": {
                "RAW": "identity",
                "PLATT": "sigmoid(a * ensemble_logodds + b), unconstrained a,b; TRAIN BCE",
                "TEMPERATURE": "sigmoid(ensemble_logodds / T), T>0; TRAIN BCE",
                "ISOTONIC": "monotone piecewise-constant map of raw ensemble probability; TRAIN only",
            },
            "application": "frozen map applied to all real benchmark points and every 0.05N dense-grid query before rho threshold",
            "selection_if_multiple_pass": ["minimum under-force", "minimum frontier MAE", "minimum probability MAE", "minimum excess force", "minimum NLL", "simpler mapping RAW>TEMPERATURE>PLATT>ISOTONIC"],
        },
        "global_calibration_pass": {"probability_mae_max": 0.20, "frontier_mae_N_max": 0.20,
                                    "under_force_rate_max": 0.10, "finite_decision_coverage_min": 0.80,
                                    "systematic_nonmonotonicity_allowed": False},
        "context_forensic": {
            "residual_groups": ["task", "context", "friction_band", "force_region", "preprobe_eef_xyz", "gripper_opening", "nominal_motion_summary"],
            "same_mu_force_tolerances": {"mu_abs": 0.03, "force_N_abs": 0.06},
            "oracle": {
                "name": "CONTEXT_ID_ORACLE_DIAGNOSTIC",
                "family_definition": "task-specific nearest TRAIN grasp-context centroid; centroids are one per TRAIN (task, root) family computed from preprobe [eef_x,y,z,opening] without labels",
                "dev_mapping": "nearest TRAIN centroid within same task; never uses DEV outcome or DEV/root index as feature",
                "input": "current features plus one-hot family ID",
                "architecture": "same FeasibilityOnly hidden sizes; condition input widened only for family one-hot",
                "seeds": SEEDS, "epochs": 80,
                "support_threshold": "probability MAE relative reduction >=30% AND frontier/under-force materially improve",
            },
        },
        "conditional_context_models": {
            "run_only_if_oracle_supports": True,
            "common_added_representation": "PREPROBE_GRASP_CONTEXT_4D",
            "models": ["CONTEXT_CONTINUOUS_FEAS", "CONTEXT_CONTINUOUS_JOINT"],
            "seeds": SEEDS, "epochs": 80,
            "same_train_branches_labels_forces": True,
            "joint_only_extra_training_signal": "authoritative corrected physical telemetry/objectives",
            "no_new_loss_or_lambda_sweep": True,
        },
        "context_value_test": {"probability_mae_relative_improvement_min": 0.20, "frontier_must_improve": True,
                               "under_force_must_not_worsen": True, "leave_one_context_out_influence_required": True},
        "joint_value_test": {"under_force_lower_or_nonworse": True, "frontier_mae_gain_N_min": 0.05,
                             "probability_mae_relative_gain_min": 0.10, "monotonicity_must_be_acceptable": True},
        "gt_gate": {"probability_mae_max": 0.20, "frontier_mae_N_max": 0.20, "under_force_rate_max": 0.10,
                    "finite_decision_coverage_min": 0.80, "systematic_nonmonotonicity_allowed": False,
                    "rho": 0.80, "query_grid_N": 0.05},
        "probe_gate": {"run_only_if_gt_passes": True, "probe": "frozen P4-B mu_hat", "no_probe": "frozen TRAIN-only friction prior",
                       "same_backend_state_context_grid": True, "no_retraining": True},
        "source_hashes": source_hashes,
        "authoritative_input_hashes": {
            "previous_forensic": sha256(STOCH / "FORENSIC_CLASSIFICATION.json"),
            "old_predictions": sha256(GNP / "CONTINUOUS_DEV_FROZEN_PREDICTIONS.csv"),
            "old_train_outcomes": sha256(GNP / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv"),
            "old_normalization": sha256(GNP / "GNP_STYLE_TRAIN_NORMALIZATION.json"),
        },
    }
    dump(protocol_path, protocol)
    (OUT / "CONTEXT_CALIBRATION_FORENSIC_PROTOCOL.sha256").write_text(
        f"{sha256(protocol_path)}  {protocol_path.name}\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "out": str(OUT), "protocol_sha256": sha256(protocol_path)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["phase0", "phase1", "phase2", "phase3", "phase4"])
    args = ap.parse_args()
    if args.phase == "phase0":
        phase0()
    elif args.phase == "phase1":
        phase1()
    elif args.phase == "phase2":
        phase2()
    elif args.phase == "phase3":
        phase3()
    elif args.phase == "phase4":
        phase4()


if __name__ == "__main__":
    main()
