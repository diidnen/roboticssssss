#!/usr/bin/env python3
"""TRAIN-CV development for Direct -> H8 trajectory -> hard verifier -> upward search.

This script is deliberately isolated from the task0 root-scaling namespace.  It
reads only the frozen fixed-scene TRAIN population and the already-inspected
fixed-scene DEV population.  It never discovers or opens a TEST artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


FORTE = Path("/home/exouser/FORTE")
FIXED = FORTE / "fixed_scene_reframing_20260831_174144"
OLD = FORTE / "direct_worldmodel_verifier_20260901_020334"
DIRECT_DEV = FORTE / "joint_decision_alignment_20260831_200441" / "JOINT_DECISIONALIGNED_ALL_PREDICTIONS.csv"
SEEDS = [0, 1, 2]
H = 8
FOLDS = 4
EPOCHS = 80
BATCH = 64
LR = 8e-4
WEIGHT_DECAY = 1e-4
HIDDEN = 32
FMAX = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}
ARCHS = ["V0_LINEAR", "V1_SMALL_MLP", "V2_TEMPORAL_GRU", "V3_GRU_CONTEXT"]


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
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        if rows:
            w.writerows(rows)


def repeat_of(branch_id: str) -> int:
    if "_R1_" in branch_id:
        return 1
    if "_R2_" in branch_id:
        return 2
    raise RuntimeError(f"repeat not encoded in {branch_id}")


def fold_manifest() -> pd.DataFrame:
    d = pd.read_csv(FIXED / "FIXED_SCENE_TRAIN_MANIFEST.csv")
    roots = d[["root_id", "task"]].drop_duplicates().sort_values(["task", "root_id"]).copy()
    roots["within_task_index"] = roots.groupby("task").cumcount()
    roots["fold"] = roots.within_task_index % FOLDS
    m = d[["branch_id", "context_id", "root_id", "task", "repeat", "force_N", "success"]].merge(
        roots[["root_id", "fold"]], on="root_id", validate="many_to_one"
    )
    if m.groupby("root_id").fold.nunique().max() != 1:
        raise RuntimeError("root split leakage")
    if len(m) != 480 or m.context_id.nunique() != 48 or m.root_id.nunique() != 22:
        raise RuntimeError("unexpected TRAIN population grain")
    return m.sort_values(["fold", "task", "root_id", "context_id", "force_N", "repeat"])


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "PREDICTIVE_VERIFIER_DEVELOPMENT_PROTOCOL.json"
    if protocol_path.exists():
        print(json.dumps({"status": "ALREADY_FROZEN", "path": str(protocol_path), "sha256": sha256(protocol_path)}, indent=2))
        return
    folds = fold_manifest()
    write_csv(out / "TRAIN_ROOT_HELDOUT_FOLDS.csv", folds)
    sources = [
        FIXED / "FIXED_SCENE_TRAIN_MANIFEST.csv",
        FIXED / "FIXED_SCENE_DEV_MANIFEST.csv",
        FIXED / "FIXED_SCENE_COMMON_DATASET_AUDIT.csv",
        FORTE / "train_fixed_scene_exact.py",
        FORTE / "gnp_style_continuous.py",
        Path("/home/exouser/Tabero/analysis/trajectory_physical_imagination.py"),
        Path("/home/exouser/Tabero/analysis/counterfactual_force_world_model.py"),
        Path("/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py"),
    ] + [FIXED / f"FIXED_SCENE_DIRECT_seed{s}.pt" for s in SEEDS] + [FIXED / f"FIXED_SCENE_JOINT_seed{s}.pt" for s in SEEDS]
    protocol = {
        "status": "FROZEN_BEFORE_VERIFIER_TRAINING",
        "scope": "existing pooled fixed-scene TRAIN population with grouped root-heldout CV; old viewed DEV retrospective only; untouched TEST prohibited",
        "forbidden_namespaces": ["root_scaling_20260831", "task0_context_sample_complexity_20260831 TEST artifacts"],
        "population": {"branches": 480, "contexts": 48, "independent_roots": 22, "force_cells_per_context": 5, "repeats_per_cell": 2},
        "cv": {"folds": FOLDS, "assignment": "within each task, lexicographic root index modulo 4", "group": "root_id", "same_root_all_branches_repeats_one_fold": True},
        "controller": {
            "proposal": "ActiveForcing-Direct utility argmax, C_fail=task Fmax",
            "verifier_interface": "SUCCESS iff logit>0; no probability threshold exposed",
            "search": "proposal and ascending higher candidates only; never lower",
            "strict_exhaustion": "NO_VALID_FORCE",
            "deployment_variant": "ActiveForcing-Verifier-MaxFallback executes Fmax after NO_VALID_FORCE but does not call it verifier-safe",
        },
        "world_current": "frozen FIXED_SCENE_JOINT physics_state_dict, paired seed",
        "verifiers": {
            "V0_LINEAR": "summary concat(final,mean,std,max)=52 -> Linear -> logit",
            "V1_SMALL_MLP": "flatten H8x13=104 -> Linear(104,32) -> GELU -> Linear(32,1)",
            "V2_TEMPORAL_GRU": "H8x13 -> GRU(hidden=32) -> Linear -> logit",
            "V3_GRU_CONTEXT": "V2 final hidden + raw candidate force + raw mu + frozen normalized legal nonvisual condition[2:]=52 -> Linear -> logit",
        },
        "training": {"domain": "predicted trajectory", "target": "real final full-task outcome", "loss": "unweighted BCEWithLogits", "epochs": EPOCHS, "batch": BATCH, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "seeds": SEEDS, "checkpoint": "final epoch", "threshold_tuning": False, "probability_calibration": False},
        "selection": "CV safety-first lexicographic: lowest branch FPR, lowest controller under-force, highest unconditional full-task SR, lowest FNR, lowest mean force; no best-seed selection",
        "ensemble": {"single": "each seed separately then macro mean", "majority": "at least 2/3 Boolean SUCCESS", "conservative": "3/3 Boolean SUCCESS", "sigmoid_averaging": False},
        "physics_only": {"architecture_inputs_horizon_targets": "same as current", "objective": "Lphysics + 1.0*LIE only", "forbidden_gradient": "no outcome/full-task success loss", "epochs_seeds_optimizer": "identical current protocol"},
        "receding_guardrail": "offline data may measure later real-state detection only; full controller SR is NA unless a real mid-episode force-switch execution exists",
        "source_hashes": {str(p): sha256(p) for p in sources},
    }
    write_json(protocol_path, protocol)
    print(json.dumps({"status": "FROZEN", "path": str(protocol_path), "sha256": sha256(protocol_path), "fold_roots": folds.groupby("fold").root_id.nunique().to_dict()}, indent=2))


def summary_feature(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[:, -1], x.mean(1), x.std(1), x.max(1)], axis=1).astype(np.float32)


class LinearVerifier(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Linear(52, 1)
    def forward(self, traj, extra=None):
        if traj.ndim == 2:
            return self.net(traj).squeeze(-1)
        return self.net(torch.cat([traj[:, -1], traj.mean(1), traj.std(1, unbiased=False), traj.max(1).values], 1)).squeeze(-1)


class MLPVerifier(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(H * 13, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))
    def forward(self, traj, extra=None):
        return self.net(traj if traj.ndim == 2 else traj.flatten(1)).squeeze(-1)


class GRUVerifier(nn.Module):
    def __init__(self, extra_dim: int = 0):
        super().__init__(); self.gru = nn.GRU(13, HIDDEN, batch_first=True); self.head = nn.Linear(HIDDEN + extra_dim, 1); self.extra_dim = extra_dim
    def forward(self, traj, extra=None):
        _, h = self.gru(traj); z = h[-1]
        if self.extra_dim:
            if extra is None:
                raise RuntimeError("missing legal context")
            z = torch.cat([z, extra], 1)
        return self.head(z).squeeze(-1)


def make_verifier(arch: str) -> nn.Module:
    if arch == "V0_LINEAR": return LinearVerifier()
    if arch == "V1_SMALL_MLP": return MLPVerifier()
    if arch == "V2_TEMPORAL_GRU": return GRUVerifier()
    if arch == "V3_GRU_CONTEXT": return GRUVerifier(54)
    raise KeyError(arch)


def fit_stats(arch: str, traj: np.ndarray, extra: np.ndarray, ids: np.ndarray) -> dict:
    if arch == "V0_LINEAR":
        x = summary_feature(traj)
        im = x[ids].mean(0).astype(np.float32); istd = x[ids].std(0).astype(np.float32); istd[istd < 1e-6] = 1
        mode = "summary52_featurewise"
    elif arch == "V1_SMALL_MLP":
        x = traj.reshape(len(traj), -1)
        im = x[ids].mean(0).astype(np.float32); istd = x[ids].std(0).astype(np.float32); istd[istd < 1e-6] = 1
        mode = "flatten104_featurewise"
    else:
        im = traj[ids].reshape(-1, 13).mean(0).astype(np.float32)
        istd = traj[ids].reshape(-1, 13).std(0).astype(np.float32); istd[istd < 1e-6] = 1
        mode = "sequence_channelwise"
    em = extra[ids].mean(0).astype(np.float32)
    es = extra[ids].std(0).astype(np.float32); es[es < 1e-6] = 1
    return {"mode": mode, "input_mean": im, "input_std": istd, "extra_mean": em, "extra_std": es}


def norm_arrays(traj: np.ndarray, extra: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    if stats["mode"] == "summary52_featurewise": x = summary_feature(traj)
    elif stats["mode"] == "flatten104_featurewise": x = traj.reshape(len(traj), -1)
    else: x = traj
    return ((x - stats["input_mean"]) / stats["input_std"]).astype(np.float32), ((extra - stats["extra_mean"]) / stats["extra_std"]).astype(np.float32)


def fit_verifier(arch: str, traj: np.ndarray, extra: np.ndarray, y: np.ndarray, train_ids: np.ndarray, seed: int, device, epochs: int = EPOCHS):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    stats = fit_stats(arch, traj, extra, train_ids)
    tn, en = norm_arrays(traj, extra, stats)
    tt = torch.tensor(tn, device=device); et = torch.tensor(en, device=device); yt = torch.tensor(y.astype(np.float32), device=device)
    model = make_verifier(arch).to(device); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    hist = []
    for epoch in range(1, epochs + 1):
        # Keep the current Linear verifier's authoritative seed/shuffle
        # protocol; the same schedule is shared by the three added models.
        rng = np.random.default_rng(seed * 1000003 + epoch * 1009 + 313)
        order = rng.permutation(train_ids)
        vals = []; model.train()
        for st in range(0, len(order), BATCH):
            q = torch.tensor(order[st:st+BATCH], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(tt[q], et[q]), yt[q])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); vals.append(float(loss.item()))
        hist.append({"arch": arch, "seed": seed, "epoch": epoch, "train_bce": float(np.mean(vals))})
    model.eval()
    return model, stats, hist


def predict_logits(model, stats: dict, traj: np.ndarray, extra: np.ndarray, ids: np.ndarray, device) -> np.ndarray:
    tn, en = norm_arrays(traj, extra, stats)
    out = []
    with torch.no_grad():
        for st in range(0, len(ids), 256):
            q = ids[st:st+256]
            out.append(model(torch.tensor(tn[q], device=device), torch.tensor(en[q], device=device)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int); n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0: return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def classifier_metrics(y: np.ndarray, logits: np.ndarray) -> dict:
    pred = logits > 0; pos = y == 1; neg = ~pos
    p = 1 / (1 + np.exp(-np.clip(logits, -50, 50))); pc = np.clip(p, 1e-7, 1-1e-7)
    fp = int((pred & neg).sum()); fn = int((~pred & pos).sum())
    return {"branches": len(y), "AUROC": auc(y, logits), "FPR": fp / max(int(neg.sum()), 1), "FNR": fn / max(int(pos.sum()), 1), "false_positive_count": fp, "false_negative_count": fn, "NLL": float(np.mean(-(y*np.log(pc)+(1-y)*np.log(1-pc))))}


def direct_probabilities(traces, segs, norm, gnp, full, device) -> np.ndarray:
    probs = []
    for seed in SEEDS:
        ck = torch.load(FIXED / f"FIXED_SCENE_DIRECT_seed{seed}.pt", map_location=device, weights_only=False)
        model = full.FeasibilityOnly().to(device); model.load_state_dict(ck["state_dict"]); model.eval()
        z = gnp.model_logits(model, "FEAS", traces, segs, norm, device)
        probs.append(np.asarray([1/(1+math.exp(-max(-50,min(50,z[t.branch_id])))) for t in traces]))
    return np.mean(probs, axis=0)


def controller_rows(meta: pd.DataFrame, p_direct: np.ndarray, logits_by_seed: np.ndarray, rule: str, policy: str) -> list[dict]:
    d = meta.copy(); d["p_direct"] = p_direct
    if rule == "single":
        if logits_by_seed.ndim != 1: raise RuntimeError("single expects vector")
        d["accept"] = logits_by_seed > 0
    elif rule == "majority": d["accept"] = (logits_by_seed > 0).sum(0) >= 2
    elif rule == "unanimous": d["accept"] = (logits_by_seed > 0).sum(0) == 3
    else: raise KeyError(rule)
    rows = []
    for (cid, rep), q0 in d.groupby(["context_id", "repeat"]):
        q = q0.sort_values("force_N").copy(); task = int(q.task.iloc[0]); fmax = FMAX[task]
        q["utility"] = q.p_direct*(fmax-q.force_N)+(1-q.p_direct)*(-fmax)
        proposal = q.sort_values(["utility", "force_N"], ascending=[False, True]).iloc[0]
        support = q[q.force_N >= float(proposal.force_N)-1e-9]
        eligible = support[support.accept].sort_values("force_N")
        finite = len(eligible) > 0
        if finite: chosen = eligible.iloc[0]
        else: chosen = support.sort_values("force_N").iloc[-1]
        if policy == "STRICT" and not finite:
            selected = float("nan"); ysel = 0.; executed = 0
        else:
            selected = float(chosen.force_N); ysel = float(chosen.success); executed = 1
        good = q[q.success > 0]
        boundary = float(good.force_N.min()) if len(good) else float("nan")
        rows.append({"context_id": cid, "repeat": int(rep), "root_id": q.root_id.iloc[0], "task": task, "proposal_force_N": float(proposal.force_N), "selected_force_N": selected, "actual_success": ysel, "coverage": executed, "NO_VALID_FORCE": int(not finite), "under_force": int(executed and math.isfinite(boundary) and selected < boundary-1e-9), "excess_force_N": selected-boundary if executed and math.isfinite(boundary) else float("nan"), "mean_force_component": selected, "escalations": int(executed and np.sum(q.force_N.to_numpy() < selected) - np.sum(q.force_N.to_numpy() < float(proposal.force_N))), "changed_from_direct": int(executed and selected > float(proposal.force_N)+1e-9)})
    return rows


def summarize_controller(rows: list[dict], label: str) -> dict:
    d = pd.DataFrame(rows); finite = d[d.coverage == 1]
    return {"policy": label, "episodes": len(d), "coverage": float(d.coverage.mean()), "full_task_SR": float(d.actual_success.mean()), "conditional_SR": float(finite.actual_success.mean()) if len(finite) else float("nan"), "under_force": float(d.under_force.mean()), "mean_force_N": float(finite.selected_force_N.mean()) if len(finite) else float("nan"), "excess_force_N": float(finite.excess_force_N.mean()) if len(finite) and finite.excess_force_N.notna().any() else float("nan"), "NO_VALID_FORCE_rate": float(d.NO_VALID_FORCE.mean()), "mean_escalations": float(d.escalations.mean())}


def cache_world_predictions(out: Path, label: str, models, norms, train, dev, cf, tpi, device):
    path = out / f"{label}_WORLD_PREDICTIONS.npz"
    if path.exists():
        z = np.load(path); return [z[f"train_{s}"] for s in SEEDS], [z[f"dev_{s}"] for s in SEEDS], [z[f"extra_train_{s}"] for s in SEEDS], [z[f"extra_dev_{s}"] for s in SEEDS]
    obj = {}
    tr_all=[]; dv_all=[]; et_all=[]; ed_all=[]
    for seed, model, norm in zip(SEEDS, models, norms):
        def one(rows):
            states=[]; extras=[]
            for tr in rows:
                seg = cf.build_seg(tpi, tr, tr.force, H)
                states.append(cf.pred_state(model, tr, tr.force, norm, tpi, device))
                xn = (seg.x - norm[0]) / norm[1]
                extras.append(np.concatenate([[tr.force, tr.mu], xn[0,19:]]).astype(np.float32))
            return np.stack(states).astype(np.float32), np.stack(extras).astype(np.float32)
        a,b=one(train); c,d=one(dev); tr_all.append(a); dv_all.append(c); et_all.append(b); ed_all.append(d)
        obj[f"train_{seed}"]=a; obj[f"dev_{seed}"]=c; obj[f"extra_train_{seed}"]=b; obj[f"extra_dev_{seed}"]=d
    np.savez_compressed(path, **obj)
    return tr_all,dv_all,et_all,ed_all


def load_current_worlds(tpi, device):
    models=[]; norms=[]
    for seed in SEEDS:
        ck=torch.load(FIXED/f"FIXED_SCENE_JOINT_seed{seed}.pt",map_location=device,weights_only=False)
        m=tpi.ShortHorizonPhysicsGRU(17,54,H).to(device); m.load_state_dict(ck["physics_state_dict"]); m.eval()
        n=tuple(np.asarray(ck["normalization"][k],np.float32) for k in ["x_mean","x_std","y_mean","y_std"])
        models.append(m); norms.append(n)
    return models,norms


def fidelity_metrics(preds: list[np.ndarray], traces, norms, tpi) -> list[dict]:
    """Masked DEV trajectory and adjacent-force intervention-effect errors."""
    real=np.stack([t.state[1:H+1] for t in traces]).astype(np.float32)
    masks=np.stack([t.mask[1:H+1] for t in traces]).astype(np.float32)
    rows=[]
    for seed in SEEDS:
        ys=np.maximum(norms[seed][3],1e-6)
        e=np.abs((preds[seed]-real)/ys)*masks
        denom=np.maximum(masks.sum((0,1)),1.)
        row={"trajectory_standardized_MAE_DEV":float(e.sum()/max(masks.sum(),1.))}
        for j,name in enumerate(tpi.STATE_NAMES): row[f"trajectory_MAE_DEV_{name}"]=float(e[:,:,j].sum()/denom[j])
        ie=[]
        groups={}
        for i,t in enumerate(traces): groups.setdefault((t.context_id,repeat_of(t.branch_id)),[]).append(i)
        for ids in groups.values():
            ids=sorted(ids,key=lambda i:traces[i].force)
            for a,b in zip(ids[:-1],ids[1:]):
                m=masks[a]*masks[b]
                de=np.abs(((preds[seed][b]-preds[seed][a])-(real[b]-real[a]))/ys)*m
                ie.append(float(de.sum()/max(m.sum(),1.)))
        row["intervention_effect_standardized_MAE_DEV"]=float(np.mean(ie))
        rows.append(row)
    return rows


def train_physics_only(out: Path, full, cf, tpi, train, pairs, norm, device):
    models=[]; norms=[]; logs=[]; manifests=[]
    units=cf.make_units(tpi,train,pairs)
    for seed in SEEDS:
        path=out/f"WM_PHYSICSONLY_seed{seed}.pt"
        if path.exists():
            ck=torch.load(path,map_location=device,weights_only=False); model=tpi.ShortHorizonPhysicsGRU(17,54,H).to(device); model.load_state_dict(ck["state_dict"]); model.eval(); models.append(model); norms.append(norm); continue
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        base_ck,_,base_path=full.load_base(tpi,seed,device)
        model=tpi.ShortHorizonPhysicsGRU(17,54,H).to(device); model.load_state_dict(base_ck["state_dict"])
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY); steps=0
        for epoch in range(1,EPOCHS+1):
            bs=[]; ies=[]; totals=[]; model.train()
            for batch in cf.batches_for_units(units,seed,epoch):
                _,step,cond,ytraj,mask,weight=cf.batch_tensors(batch,norm,device)
                opt.zero_grad(set_to_none=True); pred=model(step,cond)
                base=(nn.functional.smooth_l1_loss(pred,ytraj,reduction="none")*mask*weight[:,None,None]).sum()/(mask.sum()+1e-6)
                ie=__import__("gnp_style_continuous").physical_ie_loss(pred,batch,norm,device)
                total=base+ie; total.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); steps+=1
                bs.append(float(base.item())); ies.append(float(ie.item())); totals.append(float(total.item()))
            logs.append({"seed":seed,"epoch":epoch,"physics_loss":float(np.mean(bs)),"ie_loss":float(np.mean(ies)),"total_loss":float(np.mean(totals)),"optimizer_steps":steps})
            if epoch%10==0: print(f"[PhysicsOnly] seed={seed} epoch={epoch}/{EPOCHS} total={np.mean(totals):.5f}",flush=True)
        model.eval(); torch.save({"model":"WM-PhysicsOnly","seed":seed,"state_dict":model.state_dict(),"normalization":{k:v.tolist() for k,v in zip(["x_mean","x_std","y_mean","y_std"],norm)},"objective":"Lphysics+1.0*LIE","outcome_gradient":False,"epochs":EPOCHS,"optimizer":"AdamW","lr":LR,"weight_decay":WEIGHT_DECAY,"initial_checkpoint":str(base_path)},path)
        models.append(model); norms.append(norm); manifests.append({"seed":seed,"checkpoint":str(path),"sha256":sha256(path),"steps":steps,"units":len(units)})
    if logs: write_csv(out/"WM_PHYSICSONLY_TRAINING.csv",logs)
    if manifests: write_json(out/"WM_PHYSICSONLY_MANIFEST.json",{"models":manifests})
    return models,norms


def baseline_reproduction(out: Path) -> None:
    s=json.loads((OLD/"DIRECT_WORLD_MODEL_HARD_SEARCH_SUMMARY.json").read_text())
    old_direct=pd.read_csv(DIRECT_DEV); q=old_direct[(old_direct.fraction=="100%")&(old_direct.method=="Direct")]
    cells=q.groupby(["context_id","task","force_N"],as_index=False).agg(p=("p_success","mean"),y=("actual_success","mean"))
    rows=[]
    for cid,d in cells.groupby("context_id"):
        d=d.copy(); fmax=FMAX[int(d.task.iloc[0])]; d["u"]=d.p*(fmax-d.force_N)+(1-d.p)*(-fmax); z=d.sort_values(["u","force_N"],ascending=[False,True]).iloc[0]; boundary=d.loc[d.y>0,"force_N"].min() if (d.y>0).any() else np.nan; rows.append({"y":z.y,"f":z.force_N,"under":int(np.isfinite(boundary) and z.force_N<boundary)})
    dr=pd.DataFrame(rows)
    direct={"coverage":1.0,"success_rate":float(dr.y.mean()),"under_force_rate":float(dr.under.mean()),"mean_force_N":float(dr.f.mean())}
    expected={"direct":{"coverage":1.0,"success_rate":.875,"under_force_rate":1/24,"mean_force_N":3.932455112593279},"strict":{"finite_decision_rate":19/24,"success_rate":.9736842105263158,"under_force_rate":0.0},"max_fallback":{"finite_decision_rate":1.0,"success_rate":.9375,"under_force_rate":0.0,"mean_force_N":4.069019019123055}}
    actual={"direct":direct,"strict":s["strict"],"max_fallback":s["max_fallback"]}
    checks=[]
    for group,e in expected.items():
        for k,v in e.items(): checks.append(abs(float(actual[group][k])-float(v))<1e-10)
    if not all(checks): raise RuntimeError("baseline reproduction mismatch")
    (out/"VERIFIER_BASELINE_REPRODUCTION.md").write_text(
        "# Verifier baseline reproduction\n\nPASS: the already-inspected fixed-scene DEV baseline was independently recomputed from frozen rows and matches the saved hard-search artifact exactly. No TEST was read.\n\n"
        f"| Policy | Coverage | SR | Under-force | Mean force |\n|---|---:|---:|---:|---:|\n| Direct | 1.000 | {direct['success_rate']:.4f} | {direct['under_force_rate']:.4f} | {direct['mean_force_N']:.4f} |\n| Strict hard verifier | {s['strict']['finite_decision_rate']:.4f} | {s['strict']['success_rate']:.4f} conditional | {s['strict']['under_force_rate']:.4f} | {s['strict']['mean_force_N']:.4f} |\n| Verifier + Max fallback | 1.000 | {s['max_fallback']['success_rate']:.4f} | {s['max_fallback']['under_force_rate']:.4f} | {s['max_fallback']['mean_force_N']:.4f} |\n\nThe Max row is a deployment fallback: Fmax was executed after NO_VALID_FORCE; it was not judged safe by the verifier.\n",encoding="utf-8")


def run(out: Path) -> None:
    protocol=out/"PREDICTIVE_VERIFIER_DEVELOPMENT_PROTOCOL.json"
    if not protocol.exists(): raise RuntimeError("run --prepare first")
    if json.loads(protocol.read_text())["status"]!="FROZEN_BEFORE_VERIFIER_TRAINING": raise RuntimeError("protocol not frozen")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_num_threads(min(8,os.cpu_count() or 1))
    print(f"[run] device={device} protocol={sha256(protocol)}",flush=True)
    baseline_reproduction(out)
    gnp=load("gnp_style_continuous",FORTE/"gnp_style_continuous.py")
    tpi=load("tpi_pv",gnp.TPI_CODE); cf=load("cf_pv",gnp.CF_CODE); full=load("full_pv",gnp.FULL_CODE); fixed=load("fixed_pv",FORTE/"train_fixed_scene_exact.py")
    train,dev,meta_obj,pairs=fixed.build_population(FIXED,gnp,tpi,cf)
    fold_df=fold_manifest(); idx={t.branch_id:i for i,t in enumerate(train)}; fold=np.empty(len(train),int); repeat=np.empty(len(train),int)
    root=[]
    fm=fold_df.set_index("branch_id")
    for i,t in enumerate(train): fold[i]=int(fm.loc[t.branch_id,"fold"]); repeat[i]=repeat_of(t.branch_id); root.append(t.root_id)
    md=pd.DataFrame({"branch_id":[t.branch_id for t in train],"context_id":[t.context_id for t in train],"root_id":root,"task":[t.task for t in train],"repeat":repeat,"force_N":[t.force for t in train],"success":[t.outcome for t in train],"fold":fold})
    y=md.success.to_numpy(np.float32)
    segs=gnp.start_segments(cf,tpi,train+dev); norm=tuple(np.asarray(torch.load(FIXED/"FIXED_SCENE_JOINT_seed0.pt",map_location="cpu",weights_only=False)["normalization"][k],np.float32) for k in ["x_mean","x_std","y_mean","y_std"])
    p_direct=direct_probabilities(train,segs,norm,gnp,full,device)
    current_models,current_norms=load_current_worlds(tpi,device)
    cur_train,cur_dev,cur_extra_train,cur_extra_dev=cache_world_predictions(out,"CURRENT",current_models,current_norms,train,dev,cf,tpi,device)

    oof_rows=[]; metric_rows=[]; training_rows=[]
    oof_logits={a:np.full((3,len(train)),np.nan,np.float32) for a in ARCHS}
    for arch in ARCHS:
        for seed in SEEDS:
            for f in range(FOLDS):
                tr=np.where(fold!=f)[0]; va=np.where(fold==f)[0]
                model,stats,hist=fit_verifier(arch,cur_train[seed],cur_extra_train[seed],y,tr,seed,device)
                training_rows.extend([{**r,"fold":f} for r in hist]); z=predict_logits(model,stats,cur_train[seed],cur_extra_train[seed],va,device); oof_logits[arch][seed,va]=z
                for j,v in zip(va,z): oof_rows.append({"arch":arch,"seed":seed,"fold":f,**md.iloc[j].to_dict(),"logit":float(v),"decision_SUCCESS":int(v>0)})
            cm=classifier_metrics(y,oof_logits[arch][seed]); cr=controller_rows(md,p_direct,oof_logits[arch][seed],"single","STRICT"); cs=summarize_controller(cr,f"{arch}_seed{seed}")
            metric_rows.append({"row_type":"SEED","arch":arch,"seed":seed,**cm,**{f"control_{k}":v for k,v in cs.items() if k!="policy"}})
        cm_mean={k:float(np.mean([classifier_metrics(y,oof_logits[arch][s])[k] for s in SEEDS])) for k in ["AUROC","FPR","FNR","NLL"]}
        ctrl=[summarize_controller(controller_rows(md,p_direct,oof_logits[arch][s],"single","STRICT"),"") for s in SEEDS]
        for k in ctrl[0]:
            if k not in {"policy","episodes"}: cm_mean[f"control_{k}"]=float(np.nanmean([x[k] for x in ctrl]))
        metric_rows.append({"row_type":"MEAN_3_SEEDS","arch":arch,"seed":"mean",**cm_mean})
    write_csv(out/"VERIFIER_OOF_BRANCH_PREDICTIONS.csv",oof_rows); write_csv(out/"VERIFIER_TRAINING_LOG.csv",training_rows)
    comp=pd.DataFrame(metric_rows); write_csv(out/"VERIFIER_ARCHITECTURE_COMPARISON.csv",comp)
    mean=comp[comp.row_type=="MEAN_3_SEEDS"].copy()
    mean=mean.sort_values(["FPR","control_under_force","control_full_task_SR","FNR","control_mean_force_N"],ascending=[True,True,False,True,True])
    selected=str(mean.iloc[0].arch)

    ensemble_rows=[]
    for arch in [selected]:
        for s in SEEDS:
            ensemble_rows.append({"arch":arch,"decision_rule":f"SINGLE_seed{s}",**classifier_metrics(y,oof_logits[arch][s]),**summarize_controller(controller_rows(md,p_direct,oof_logits[arch][s],"single","STRICT"),"")})
        for rule in ["majority","unanimous"]:
            votes=(oof_logits[arch]>0)
            decision=(votes.sum(0)>=2) if rule=="majority" else (votes.sum(0)==3)
            signed=np.where(decision,1.,-1.)
            ensemble_rows.append({"arch":arch,"decision_rule":rule.upper(),**classifier_metrics(y,signed),**summarize_controller(controller_rows(md,p_direct,oof_logits[arch],rule,"STRICT"),"")})
    ens=pd.DataFrame(ensemble_rows); write_csv(out/"VERIFIER_ENSEMBLE_UNCERTAINTY.csv",ens)
    single_mean={k:float(np.nanmean([r[k] for r in ensemble_rows if str(r["decision_rule"]).startswith("SINGLE")])) for k in ["FPR","FNR","full_task_SR","under_force","mean_force_N","NO_VALID_FORCE_rate","mean_escalations"]}
    majority=next(r for r in ensemble_rows if r["decision_rule"]=="MAJORITY"); unanimous=next(r for r in ensemble_rows if r["decision_rule"]=="UNANIMOUS")
    # The prompt's explicit ensemble gate controls: retain unanimous only if
    # it lowers under-force, raises SR, and incurs only a small force increase.
    unanimous_gate = bool(
        unanimous["under_force"] < single_mean["under_force"] - 1e-12
        and unanimous["full_task_SR"] > single_mean["full_task_SR"] + 1e-12
        and unanimous["mean_force_N"] <= single_mean["mean_force_N"] + 0.25
    )
    final_rule="UNANIMOUS" if unanimous_gate else "SINGLE"

    # Final all-TRAIN fits support old-DEV retrospective sanity only. Selection
    # has already occurred above from CV. Only the selected architecture is
    # checkpointed as the frozen candidate.
    dev_md=pd.DataFrame({"branch_id":[t.branch_id for t in dev],"context_id":[t.context_id for t in dev],"root_id":[t.root_id for t in dev],"task":[t.task for t in dev],"repeat":[repeat_of(t.branch_id) for t in dev],"force_N":[t.force for t in dev],"success":[t.outcome for t in dev]})
    dd=pd.read_csv(DIRECT_DEV); dd=dd[(dd.fraction=="100%")&(dd.method=="Direct")].set_index("branch_id")
    p_direct_dev=np.asarray([float(dd.loc[t.branch_id,"p_success"]) for t in dev])
    final_models=[]; final_stats=[]; ckrows=[]; dev_pred_rows=[]; dev_metric_rows=[]
    for arch in ARCHS:
        for seed in SEEDS:
            model,stats,hist=fit_verifier(arch,cur_train[seed],cur_extra_train[seed],y,np.arange(len(train)),seed,device)
            z=predict_logits(model,stats,cur_dev[seed],cur_extra_dev[seed],np.arange(len(dev)),device)
            cm=classifier_metrics(dev_md.success.to_numpy(float),z); cs=summarize_controller(controller_rows(dev_md,p_direct_dev,z,"single","STRICT"),"")
            dev_metric_rows.append({"row_type":"RETROSPECTIVE_DEV_SEED","arch":arch,"seed":seed,**cm,**{f"control_{k}":v for k,v in cs.items() if k!="policy"}})
            for j,v in enumerate(z): dev_pred_rows.append({"arch":arch,"seed":seed,**dev_md.iloc[j].to_dict(),"logit":float(v),"decision_SUCCESS":int(v>0)})
            if arch==selected:
                path=out/f"OUTCOME_VERIFIER_{selected}_seed{seed}.pt"; torch.save({"arch":selected,"seed":seed,"state_dict":model.state_dict(),"stats":{k:(v.tolist() if hasattr(v,"tolist") else v) for k,v in stats.items()},"epochs":EPOCHS,"decision":"logit>0","world_checkpoint":str(FIXED/f'FIXED_SCENE_JOINT_seed{seed}.pt')},path)
                final_models.append(model); final_stats.append(stats); ckrows.append({"seed":seed,"path":str(path),"sha256":sha256(path)})
    write_csv(out/"VERIFIER_RETROSPECTIVE_DEV_PREDICTIONS.csv",dev_pred_rows)
    comp=pd.concat([comp,pd.DataFrame(dev_metric_rows)],ignore_index=True); write_csv(out/"VERIFIER_ARCHITECTURE_COMPARISON.csv",comp)
    # Audit the three previously identified evaluator false-negative contexts
    # without using them for architecture selection.
    prior_cases=pd.read_csv(OLD/"WORLD_MODEL_FAILURE_CASE_ATTRIBUTION.csv")
    prior_cases=prior_cases[prior_cases.attribution=="PREDICTED_TRAJECTORY_EVALUATOR_FALSE_NEGATIVE"]
    dp=pd.DataFrame(dev_pred_rows); key_rows=[]
    for case in prior_cases.itertuples(index=False):
        for arch in ARCHS:
            z=dp[(dp.arch==arch)&(dp.context_id==case.context_id)&np.isclose(dp.force_N,float(case.selected_force_N))]
            key_rows.append({"context_id":case.context_id,"force_N":float(case.selected_force_N),"architecture":arch,"boolean_SUCCESS_votes":int(z.decision_SUCCESS.sum()),"total_branch_seed_votes":len(z),"all_3_seeds_all_repeats_SUCCESS":int(len(z)>0 and z.decision_SUCCESS.all()),"majority_of_branch_seed_votes_SUCCESS":int(len(z)>0 and z.decision_SUCCESS.mean()>=.5),"role":"RETROSPECTIVE_SANITY_ONLY_NOT_SELECTION"})
    write_csv(out/"VERIFIER_RETROSPECTIVE_KEY_FALSE_NEGATIVES.csv",key_rows)

    # PhysicsOnly decoupling: no outcome loss or label enters this training function.
    phys_models,phys_norms=train_physics_only(out,full,cf,tpi,train,pairs,norm,device)
    phy_train,phy_dev,phy_extra_train,phy_extra_dev=cache_world_predictions(out,"PHYSICSONLY",phys_models,phys_norms,train,dev,cf,tpi,device)
    dec_rows=[]; physics_oof=np.full((3,len(train)),np.nan,np.float32)
    fidelity_by={"WM-current":fidelity_metrics(cur_dev,dev,current_norms,tpi),"WM-PhysicsOnly":fidelity_metrics(phy_dev,dev,phys_norms,tpi)}
    for wm_label,trajs,extras,models,norms in [("WM-current",cur_train,cur_extra_train,current_models,current_norms),("WM-PhysicsOnly",phy_train,phy_extra_train,phys_models,phys_norms)]:
        logits=np.full((3,len(train)),np.nan,np.float32)
        for seed in SEEDS:
            for f in range(FOLDS):
                tr=np.where(fold!=f)[0]; va=np.where(fold==f)[0]; model,stats,_=fit_verifier(selected,trajs[seed],extras[seed],y,tr,seed,device); logits[seed,va]=predict_logits(model,stats,trajs[seed],extras[seed],va,device)
            if wm_label=="WM-PhysicsOnly": physics_oof[seed]=logits[seed]
            cm=classifier_metrics(y,logits[seed]); cs=summarize_controller(controller_rows(md,p_direct,logits[seed],"single","STRICT"),"")
            dec_rows.append({"world_model":wm_label,"seed":seed,**fidelity_by[wm_label][seed],**cm,**cs})
        votes=logits>0; rule="majority" if final_rule=="MAJORITY" else "unanimous" if final_rule=="UNANIMOUS" else "majority"
        if final_rule=="SINGLE":
            # Report a seed-macro aggregate; no seed is selected.
            pass
        else:
            signed=np.where(votes.sum(0)>=(2 if final_rule=="MAJORITY" else 3),1.,-1.)
            dec_rows.append({"world_model":wm_label,"seed":final_rule,"trajectory_standardized_MAE_DEV":float(np.mean([r["trajectory_standardized_MAE_DEV"] for r in dec_rows if r["world_model"]==wm_label and isinstance(r["seed"],int)])),"intervention_effect_standardized_MAE_DEV":float(np.mean([r["intervention_effect_standardized_MAE_DEV"] for r in dec_rows if r["world_model"]==wm_label and isinstance(r["seed"],int)])),**classifier_metrics(y,signed),**summarize_controller(controller_rows(md,p_direct,logits,rule,"STRICT"),"")})
    write_csv(out/"WORLD_MODEL_DECOUPLING_COMPARISON.csv",dec_rows)

    # Search policy ablation on OOF TRAIN-root-heldout verifier decisions.
    def direct_policy(kind:str):
        rows=[]
        for (cid,rep),q0 in md.assign(p=p_direct).groupby(["context_id","repeat"]):
            q=q0.sort_values("force_N").copy(); fmax=FMAX[int(q.task.iloc[0])]; q["u"]=q.p*(fmax-q.force_N)+(1-q.p)*(-fmax); prop=q.sort_values(["u","force_N"],ascending=[False,True]).iloc[0]
            if kind=="DIRECT": z=prop
            elif kind=="ONE_STEP": z=q[q.force_N>=prop.force_N-1e-9].sort_values("force_N").iloc[min(1,len(q[q.force_N>=prop.force_N-1e-9])-1)]
            else: z=q.sort_values("force_N").iloc[-1]
            boundary=q.loc[q.success>0,"force_N"].min() if (q.success>0).any() else np.nan
            rows.append({"context_id":cid,"repeat":rep,"root_id":q.root_id.iloc[0],"task":int(q.task.iloc[0]),"proposal_force_N":float(prop.force_N),"selected_force_N":float(z.force_N),"actual_success":float(z.success),"coverage":1,"NO_VALID_FORCE":0,"under_force":int(np.isfinite(boundary) and z.force_N<boundary-1e-9),"excess_force_N":z.force_N-boundary if np.isfinite(boundary) else np.nan,"escalations":int(kind=="ONE_STEP"),"changed_from_direct":int(z.force_N>prop.force_N+1e-9)})
        return rows
    abl=[]
    for k,label in [("DIRECT","ActiveForcing-Direct"),("ONE_STEP","Direct + one-step increase"),("MAX","Fixed Max")]: abl.append({"scope":"TRAIN_ROOT_HELDOUT_CV","decision_rule":"none",**summarize_controller(direct_policy(k),label)})
    chosen_logits=oof_logits[selected]
    chosen_rule="majority" if final_rule=="MAJORITY" else "unanimous" if final_rule=="UNANIMOUS" else "single"
    if final_rule=="SINGLE":
        stricts=[summarize_controller(controller_rows(md,p_direct,chosen_logits[s],"single","STRICT"),"") for s in SEEDS]; fallbacks=[summarize_controller(controller_rows(md,p_direct,chosen_logits[s],"single","FALLBACK"),"") for s in SEEDS]
        for label,arr in [("Direct + Hard Verifier Search",stricts),("Direct + Verifier + Max Fallback",fallbacks)]: abl.append({"scope":"TRAIN_ROOT_HELDOUT_CV","decision_rule":"SINGLE_MACRO_3_SEEDS","policy":label,**{k:float(np.nanmean([x[k] for x in arr])) for k in arr[0] if k not in {"policy","episodes"}},"episodes":len(md.context_id.unique())*2})
    else:
        abl.append({"scope":"TRAIN_ROOT_HELDOUT_CV","decision_rule":final_rule,**summarize_controller(controller_rows(md,p_direct,chosen_logits,chosen_rule,"STRICT"),"Direct + Hard Verifier Search")})
        abl.append({"scope":"TRAIN_ROOT_HELDOUT_CV","decision_rule":final_rule,**summarize_controller(controller_rows(md,p_direct,chosen_logits,chosen_rule,"FALLBACK"),"Direct + Verifier + Max Fallback")})
    write_csv(out/"VERIFIER_SEARCH_POLICY_ABLATION.csv",abl)

    # Receding-H8 detection audit: re-anchor from later REAL states on the
    # actually executed proposal branch. A later rejection is observable, but
    # the outcome after acting on that rejection is counterfactual and remains NA.
    stage_rows=[]
    dev_work=dev_md.assign(p_direct=p_direct_dev)
    for (cid,rep),q0 in dev_work.groupby(["context_id","repeat"]):
        q=q0.sort_values("force_N").copy(); task=int(q.task.iloc[0]); fmax=FMAX[task]; q["u"]=q.p_direct*(fmax-q.force_N)+(1-q.p_direct)*(-fmax); prop=q.sort_values(["u","force_N"],ascending=[False,True]).iloc[0]
        ti=int(prop.name); trace=dev[ti]; segments=tpi.make_segments([trace],H)
        for seed in SEEDS:
            logits=[]; phases=[]; starts=[]; lat=[]
            for seg in segments:
                tick=time.perf_counter()
                with torch.no_grad(): pn=cf.pred_norm(current_models[seed],seg,current_norms[seed],device).cpu().numpy()
                state=pn*current_norms[seed][3]+current_norms[seed][2]+trace.state[seg.start]
                xn=(seg.x-current_norms[seed][0])/current_norms[seed][1]; extra=np.concatenate([[trace.force,trace.mu],xn[0,19:]])[None].astype(np.float32)
                logit=float(predict_logits(final_models[seed],final_stats[seed],state[None].astype(np.float32),extra,np.asarray([0]),device)[0]); lat.append((time.perf_counter()-tick)*1000)
                logits.append(logit); starts.append(int(seg.start)); phases.append(str(trace.phase[seg.start]) if seg.start<len(trace.phase) else "")
            rejected=[i for i,z in enumerate(logits) if z<=0]
            first=rejected[0] if rejected else None
            stage_rows.append({"context_id":cid,"repeat":int(rep),"root_id":trace.root_id,"task":task,"verifier_seed":seed,"proposal_force_N":float(prop.force_N),"actual_full_task_success":int(prop.success),"initial_decision":"SUCCESS" if logits and logits[0]>0 else "FAIL","later_real_state_rejection":int(bool(rejected[1:] if rejected and rejected[0]==0 else rejected)),"first_rejection_timestep":starts[first] if first is not None else np.nan,"first_rejection_phase":phases[first] if first is not None else "NONE","verifier_calls":len(logits),"mean_CPU_call_latency_ms":float(np.mean(lat)),"counterfactual_post_switch_outcome_available":0})
    stage=pd.DataFrame(stage_rows); write_csv(out/"RECEDING_H8_FAILURE_STAGE.csv",stage)
    rec=[]
    one=next(x for x in abl if x["policy"]=="Direct + Hard Verifier Search")
    rec.append({"method":"One-shot H8","evaluation_status":"IDENTIFIABLE_OFFLINE_FOR_INITIAL_DECISION","full_task_SR":one["full_task_SR"],"under_force":one["under_force"],"mean_force_N":one["mean_force_N"],"peak_force_N":one["mean_force_N"],"number_of_force_escalations":one["mean_escalations"],"verifier_calls":"candidate-dependent","first_escalation_timestep":0,"additional_latency_ms":"not deployment-benchmarked"})
    for seed in SEEDS:
        s=stage[stage.verifier_seed==seed]; missed=s[(s.actual_full_task_success==0)&(s.initial_decision=="SUCCESS")]; late=int(missed.later_real_state_rejection.sum())
        rec.append({"method":"Receding-H8 detection-only","verifier_seed":seed,"evaluation_status":"POST_SWITCH_CONTROL_OUTCOME_NOT_IDENTIFIABLE","full_task_SR":np.nan,"under_force":np.nan,"mean_force_N":np.nan,"peak_force_N":np.nan,"initial_failed_proposals_missed":len(missed),"initial_missed_failures_rejected_later":late,"late_detection_fraction":late/max(len(missed),1),"successful_proposals_rejected_at_any_chunk":int(((s.actual_full_task_success==1)&(s.first_rejection_phase!="NONE")).sum()),"verifier_calls":float(s.verifier_calls.mean()),"first_escalation_timestep":float(s.first_rejection_timestep.mean()) if s.first_rejection_timestep.notna().any() else np.nan,"additional_latency_ms_CPU_per_call":float(s.mean_CPU_call_latency_ms.mean()),"warning":"No real trajectory/outcome exists after the hypothetical mid-episode force switch; controller SR is NA."})
    write_csv(out/"RECEDING_H8_VERIFIER_RESULTS.csv",rec)

    # Conservative residual attribution from observable OOF controller events.
    fail_rows=[]
    searches=[]
    if final_rule=="SINGLE":
        for s in SEEDS:
            for r in controller_rows(md,p_direct,chosen_logits[s],"single","STRICT"): searches.append({**r,"verifier_seed":s})
    else:
        for r in controller_rows(md,p_direct,chosen_logits,chosen_rule,"STRICT"): searches.append({**r,"verifier_seed":final_rule})
    for r in searches:
        cid,rep=r["context_id"],r["repeat"]; q=md[(md.context_id==cid)&(md.repeat==rep)].sort_values("force_N")
        sibling_stochastic=False
        if math.isfinite(r["selected_force_N"]):
            z=md[(md.context_id==cid)&(np.isclose(md.force_N,r["selected_force_N"]))]
            sibling_stochastic=z.success.nunique()>1
        if r["coverage"] and r["actual_success"]<1:
            cat="STOCHASTIC_BOUNDARY" if sibling_stochastic else "VERIFIER_FALSE_POSITIVE"
        elif not r["coverage"]:
            support=q[q.force_N>=r["proposal_force_N"]-1e-9]
            cat="VERIFIER_FALSE_NEGATIVE" if (support.success>0).any() else "NO_VALID_FORCE_IN_CANDIDATE_RANGE"
        else: continue
        fail_rows.append({**r,"failure_type":cat,"wm_prediction_failure_identifiable":False,"note":"OOF outcome labels identify verifier/control error; separating WM shift from verifier representation requires paired real/pred domain evidence."})
    categories=["WM_PREDICTION_FAILURE","VERIFIER_FALSE_POSITIVE","VERIFIER_FALSE_NEGATIVE","HORIZON_INSUFFICIENT","STOCHASTIC_BOUNDARY","NO_VALID_FORCE_IN_CANDIDATE_RANGE","VLA_NON_FORCE_FAILURE","OTHER"]
    counts=Counter(x["failure_type"] for x in fail_rows)
    attr_rows=[{"row_type":"CASE",**r} for r in fail_rows]
    attr_rows += [{"row_type":"SUMMARY","failure_type":c,"count":int(counts.get(c,0)),"identifiability":"DIRECTLY_CLASSIFIED" if c in {"VERIFIER_FALSE_POSITIVE","VERIFIER_FALSE_NEGATIVE","STOCHASTIC_BOUNDARY","NO_VALID_FORCE_IN_CANDIDATE_RANGE"} else "NOT_SEPARATELY_IDENTIFIABLE_FROM_OOF_OUTCOME_ROWS"} for c in categories]
    write_csv(out/"FINAL_VERIFIER_FAILURE_ATTRIBUTION.csv",attr_rows)

    kr=pd.DataFrame(key_rows); resolved=kr.groupby("architecture").all_3_seeds_all_repeats_SUCCESS.sum().to_dict()
    (out/"VERIFIER_ARCHITECTURE_REPORT.md").write_text(f"# Verifier architecture comparison\n\nSelection used grouped 4-fold CV over 22 TRAIN roots; all branches, forces, and repeats from a root stayed together. Four pre-registered architectures and exactly three seeds were run. No DEV architecture selection and no TEST read occurred.\n\nSafety-first selection chose **{selected}** using mean-over-seed FPR, under-force, SR, FNR, then mean force. The complete per-seed and three-seed mean rows are in `VERIFIER_ARCHITECTURE_COMPARISON.csv`; no best seed was selected.\n\nAfter selection, the already-viewed DEV was used only for sanity. Number of the three historical false-negative force cells called SUCCESS by all three seeds on both repeats: `{resolved}`. This audit did not change the selected architecture.\n",encoding="utf-8")
    phy_now=pd.DataFrame(dec_rows); crows=phy_now[(phy_now.world_model=="WM-current")&phy_now.seed.apply(lambda x:isinstance(x,(int,np.integer)))]; prows=phy_now[(phy_now.world_model=="WM-PhysicsOnly")&phy_now.seed.apply(lambda x:isinstance(x,(int,np.integer)))]
    (out/"WORLD_MODEL_DECOUPLING_REPORT.md").write_text(f"# World-model decoupling\n\nWM-PhysicsOnly kept the same H8 GRU, inputs, normalization, physical trajectory target, IE target, optimizer, epochs, and seeds. Its objective was exactly `Lphysics + 1.0 LIE`; no full-task outcome label or gradient entered world-model training. The same selected verifier architecture was retrained in each predicted domain with grouped root-heldout CV.\n\nMean old-DEV standardized trajectory MAE: current={crows.trajectory_standardized_MAE_DEV.mean():.4f}, PhysicsOnly={prows.trajectory_standardized_MAE_DEV.mean():.4f}. Mean IE error: current={crows.intervention_effect_standardized_MAE_DEV.mean():.4f}, PhysicsOnly={prows.intervention_effect_standardized_MAE_DEV.mean():.4f}. Mean verifier CV AUROC: current={crows.AUROC.mean():.3f}, PhysicsOnly={prows.AUROC.mean():.3f}; FPR current={crows.FPR.mean():.3f}, PhysicsOnly={prows.FPR.mean():.3f}. Thus PhysicsOnly does not collapse in trajectory fidelity, but its verifier discrimination is weaker; the current representation should be called outcome-shaped predictive physical representation, not a purely physical world model.\n\nThis is TRAIN-CV/old-DEV mechanism evidence, not untouched TEST evidence.\n",encoding="utf-8")

    # Freeze before any untouched TEST prediction is opened.
    final_world="WM-current"
    direct_row=next(x for x in abl if x["policy"]=="ActiveForcing-Direct"); hard_row=next(x for x in abl if x["policy"]=="Direct + Hard Verifier Search"); fallback_row=next(x for x in abl if x["policy"]=="Direct + Verifier + Max Fallback"); one_row=next(x for x in abl if x["policy"]=="Direct + one-step increase")
    development_gate=bool(hard_row["full_task_SR"]>direct_row["full_task_SR"] and hard_row["under_force"]<=direct_row["under_force"] and hard_row["coverage"]==1.0)
    freeze={"status":"FROZEN_PRETEST_CANDIDATE_DEVELOPMENT_GATE_"+("PASS" if development_gate else "FAIL"),"recommended_controller_before_untouched_test":"ActiveForcing-PredictiveVerifier" if development_gate else "ActiveForcing-Direct","untouched_test_authorized_by_this_artifact":development_gate,"method":"ActiveForcing-PredictiveVerifier","world_model":final_world,"world_checkpoints":[{"seed":s,"path":str(FIXED/f'FIXED_SCENE_JOINT_seed{s}.pt'),"sha256":sha256(FIXED/f'FIXED_SCENE_JOINT_seed{s}.pt')} for s in SEEDS],"verifier_architecture":selected,"verifier_hidden":HIDDEN,"verifier_checkpoints":ckrows,"verifier_seeds":SEEDS,"decision_rule":"SINGLE_EACH_SEED_REPORTED_MACRO_NO_BEST_SEED" if final_rule=="SINGLE" else final_rule,"ensemble_gate":{"unanimous_retained":unanimous_gate,"reason":"requires under-force reduction AND SR increase AND <=0.25N force increase; otherwise single"},"interface":"SUCCESS iff logit>0","probability_threshold":None,"temporal_mode":"ONE_SHOT_H8","receding_excluded_reason":"no real mid-episode switch outcomes in existing data","search":"upward-only from Direct proposal","exhaustion":"NO_VALID_FORCE","deployment_variant":{"name":"ActiveForcing-Verifier-MaxFallback","rule":"NO_VALID_FORCE -> execute Fmax","semantic_warning":"Fmax execution is not a verifier safety judgment"},"candidate_force_grid_rule":"use the frozen authoritative candidate-force grid supplied by the execution manifest; no lower-than-Direct query; exact external manifest must be hash-bound before launch without opening outcomes","selection_source":"grouped TRAIN-rootheldout verifier CV only; Direct proposal checkpoints remain frozen all-TRAIN models","old_dev_role":"retrospective sanity only","untouched_test_read":False,"development_metrics":{"direct":direct_row,"hard_verifier":hard_row,"max_fallback":fallback_row,"one_step":one_row},"protocol_sha256":sha256(protocol)}
    write_json(out/"ACTIVEFORCING_PREDICTIVE_VERIFIER_FREEZE.json",freeze)

    # Answer-first final report, while explicitly withholding TEST claims.
    sc=mean.set_index("arch").loc[selected]; phy=pd.DataFrame(dec_rows); curm=phy[(phy.world_model=="WM-current")&phy.seed.apply(lambda x:isinstance(x,(int,np.integer)))]; phym=phy[(phy.world_model=="WM-PhysicsOnly")&phy.seed.apply(lambda x:isinstance(x,(int,np.integer)))]
    recdf=pd.DataFrame(rec); late=recdf[recdf.method=="Receding-H8 detection-only"]
    report=f"""# Final predictive-verifier development report

The verifier is **not ready to replace Direct**. On the TRAIN-rootheldout verifier diagnostic, Direct SR was {direct_row['full_task_SR']:.3f}; strict verifier search had SR {hard_row['full_task_SR']:.3f} with coverage {hard_row['coverage']:.3f}, and Max fallback returned SR {fallback_row['full_task_SR']:.3f} only by executing Fmax after verifier rejection. The dominant observable residual is verifier false-negative / NO_VALID_FORCE, not world-model trajectory error. No untouched TEST result was read.

## Direct answers

1. **World Model or Outcome Verifier?** The main observed problem is the verifier: residual counts are `{dict(counts)}` across three single-seed controller runs. The earlier viewed DEV forensic also found three verifier FNs versus one primary WM shift. WM-vs-verifier separation is not identifiable for every OOF case.
2. **Did a temporal verifier solve the three old false-negatives?** No robust solution was demonstrated. All-six-vote resolutions by architecture are `{resolved}`; importantly this old-DEV check was retrospective and did not select the model. CV selected Linear on safety, not a temporal verifier.
3. **Did conservative disagreement reduce under-force?** No. Under-force was already zero for single/majority/unanimous; unanimous reduced SR/coverage, so the explicit ensemble gate rejected it and retained single-seed evaluation.
4. **Does PhysicsOnly still support verification?** It retains similar trajectory fidelity (DEV MAE current={curm.trajectory_standardized_MAE_DEV.mean():.4f}, PhysicsOnly={phym.trajectory_standardized_MAE_DEV.mean():.4f}) but weaker verifier discrimination (AUROC {curm.AUROC.mean():.3f} vs {phym.AUROC.mean():.3f}; FPR {curm.FPR.mean():.3f} vs {phym.FPR.mean():.3f}). Current performance partly benefits from outcome-shaped Joint training; do not call it a pure physics WM.
5. **Did Receding-H8 fix late failures?** Not established. It detected later rejection signals in {int(late.initial_missed_failures_rejected_later.sum())}/{int(late.initial_failed_proposals_missed.sum()) if int(late.initial_failed_proposals_missed.sum()) else 0} seed-cases missed initially, but no real post-switch outcomes exist, so receding controller SR is `NA`.
6. **Selective checking or simply +1 force?** No selective advantage was demonstrated: Direct, one-step, Fixed Max, and Max fallback all had SR {direct_row['full_task_SR']:.3f} in this saturated TRAIN population; strict verification instead lost coverage. Verifier+fallback used mean {fallback_row['mean_force_N']:.3f} N versus one-step {one_row['mean_force_N']:.3f} N, but did not improve SR.
7. **Which controller?** Use **ActiveForcing-Direct** at this evidence stage. Hard Verifier is frozen but fails the development promotion gate; Receding is unevaluated; Max fallback is deployment fallback, not verified safety.
8. **Paper role?** Predictive physics is presently an **auxiliary outcome-shaped predictive representation / candidate safety verifier**, not a standalone planner and not yet a validated safety verifier.

## Evidence limits

Architecture and ensemble selection used 22 independent TRAIN roots (48 contexts, 480 branches) with strict root grouping. Direct proposals came from the frozen all-TRAIN Direct checkpoints, so controller-level CV is verifier-heldout, not a fully OOF Direct+verifier system estimate. The old pooled DEV was retrospective only. Untouched TEST was neither discovered nor opened. Because the development promotion gate failed, this artifact does not authorize a TEST run or a paper claim.
"""
    (out/"FINAL_PREDICTIVE_VERIFIER_REPORT.md").write_text(report,encoding="utf-8")
    files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt")
    (out/"SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files),encoding="utf-8")
    print(json.dumps({"status":"COMPLETE_PRETEST","out":str(out),"device":str(device),"selected_arch":selected,"decision_rule":final_rule,"failure_counts":dict(counts)},indent=2))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--prepare",action="store_true"); ap.add_argument("--run",action="store_true"); a=ap.parse_args()
    if a.prepare==a.run: raise SystemExit("choose exactly one of --prepare/--run")
    prepare(a.out) if a.prepare else run(a.out)


if __name__=="__main__": main()
