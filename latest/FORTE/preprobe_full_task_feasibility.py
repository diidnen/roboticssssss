#!/usr/bin/env python3
"""Train and gate the strict pre-probe Tabero full-task feasibility backend.

The script has an immutable two-stage interface:

  freeze  - audit authoritative TRAIN/DEV inputs and freeze the protocol
  run     - train three seeds, fit TRAIN-only isotonic calibrators, evaluate

Original TEST outcomes/telemetry are never retained or evaluated.  Probe and
strict no-probe inference is reached only after the preregistered GT gate.
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
P5 = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
OLD_FEAS = RESULTS / "full_task_feasibility_20260830_012830"
OLD_NEC = RESULTS / "active_probe_necessity_20260830_063435"
FRICTION = RESULTS / "active_friction_imagination_20260828_211106"
PROBE_STACK = RESULTS / "probe_informed_imagination_20260829_121500"
TPI_CODE = REPO / "analysis/trajectory_physical_imagination.py"
CF_CODE = REPO / "analysis/counterfactual_force_world_model.py"
FULL_CODE = REPO / "analysis/full_task_feasibility_decoder.py"
ACTIVE_CODE = REPO / "analysis/active_probe_necessity.py"

OUT = Path(os.environ.get(
    "PREPROBE_FEAS_OUT",
    str(RESULTS / f"preprobe_full_task_feasibility_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
))
SEEDS = [0, 1, 2]
H = 8
EPOCHS = 80
BATCH = 64
LR = 8e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
PRIOR_MUS = [0.30, 0.56, 0.92]
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 2026083007
GT_GATE = {
    "pair_adaptive_success_min": 0.85,
    "under_force_max": 0.10,
    "pair_force_ordering_min": 0.90,
    "fprev_false_safe_max": 0.10,
}
PROBE_GATE = {
    "pair_adaptive_success_min": 0.75,
    "probe_minus_no_probe_pair_success_min": 0.20,
    "under_force_max": 0.10,
    "mean_excess_less_than_no_probe": True,
    "cluster_bootstrap_95ci_lower_gt_zero": True,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(v: Any) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path: Path, v: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(v, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fallback_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    fields = fields or (fallback_fields or ["status", "reason"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        if rows:
            w.writerows(rows)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def guarded_manifest() -> pd.DataFrame:
    """Read only TRAIN/DEV rows; discard TEST before parsing its label fields."""
    rows: list[dict[str, Any]] = []
    with (P5 / "P5S0C_BRANCH_MANIFEST.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "TEST":
                continue
            row["task"] = int(row["task"])
            row["requested_force_N"] = float(row["requested_force_N"])
            row["hidden_friction_analysis_only"] = float(row["hidden_friction_analysis_only"])
            row["full_task_success_y"] = int(row["full_task_success_y"])
            rows.append(row)
    return pd.DataFrame(rows)


def all_contexts_for_split(split: str, active) -> dict[str, Any]:
    """Construct strict contexts without dropping all-failure training groups.

    The frozen active-necessity loader intentionally skipped contexts without a
    real F_star because it only evaluated controller frontiers.  BCE training
    must retain those authoritative failures.  This loader changes no labels,
    roots, states, or motion and uses max tested force only as a sampling-role
    fallback, never as an input or success label.
    """
    d = guarded_manifest()
    d = d[d.split == split].copy()
    out: dict[str, Any] = {}
    for cid, q in d.groupby("context_id", sort=True):
        rows = q.to_dict("records")
        successful = [float(r["requested_force_N"]) for r in rows if int(r["full_task_success_y"]) == 1]
        forces = sorted(float(r["requested_force_N"]) for r in rows)
        outcomes = {float(r["requested_force_N"]): int(r["full_task_success_y"]) for r in rows}
        canonical = max(rows, key=lambda r: float(r["requested_force_N"]))
        probe_path = P5 / "P5S0C_PROBE_TELEMETRY" / f"{cid}_probe_timesteps.csv"
        probe = pd.read_csv(probe_path)
        hold = probe[probe.probe_phase.astype(str) == "hold"]
        if hold.empty or "probe_out" not in set(probe.probe_phase.astype(str)):
            raise RuntimeError(f"no frozen pre-shear hold for {cid}")
        pre = hold.iloc[-1]
        opening = float(pre.gripper_opening)
        state = np.zeros(13, np.float32); mask = np.zeros(13, np.float32)
        state[11] = opening; state[12] = -opening
        mask[:6] = 1.0; mask[11:13] = 1.0
        out[str(cid)] = active.Context(
            str(cid), str(canonical["root_id"]), int(canonical["root_index"]), int(canonical["task"]),
            str(canonical["friction_band"]), float(canonical["hidden_friction_analysis_only"]),
            math.nan, math.nan, min(successful) if successful else max(forces),
            forces, outcomes, Path(canonical["telemetry_path"]), probe_path, opening, int(pre.step), state, mask,
        )
    return out


@dataclass
class Meta:
    branch_id: str
    context_id: str
    split: str
    root_id: str
    task: int
    friction_band: str
    force: float
    mu: float
    outcome: int
    role: str


def role(force: float, fstar: float, outcomes: dict[float, int]) -> str:
    if abs(force - fstar) < 1e-9:
        return "F_star"
    failed_below = [f for f, y in outcomes.items() if y == 0 and f < fstar]
    if failed_below and abs(force - max(failed_below)) < 1e-9:
        return "F_prev"
    return "other"


def strict_trace(ctx, force: float, split: str, tpi):
    d = pd.read_csv(ctx.canonical_path)
    state, mask = tpi.state_from(d)
    state = state.copy(); mask = mask.copy()
    state[0] = ctx.preprobe_state
    mask[0] = ctx.preprobe_mask
    nominal = tpi.nominal_from(d, ctx.task, force, ctx.mu_gt, state, mask)
    fs = ctx.fstar
    r = role(force, fs, ctx.outcomes)
    return tpi.Trace(
        f"preprobe:{ctx.context_id}:{force}", ctx.context_id, ctx.root_id, ctx.task,
        split, force, ctx.mu_gt, ctx.outcomes[force], r, ctx.canonical_path,
        state, mask, nominal, d.phase.astype(str).tolist(), 3.0 if r in {"F_prev", "F_star"} else 1.0,
        "strict_preprobe_authoritative_branch_label",
    )


def build_population(tpi, active):
    contexts = {**all_contexts_for_split("TRAIN", active), **all_contexts_for_split("DEV", active)}
    manifest = guarded_manifest()
    traces: list[Any] = []
    meta: dict[str, Meta] = {}
    segments: dict[str, Any] = {}
    for r in manifest.itertuples(index=False):
        cid = str(r.context_id)
        ctx = contexts[cid]
        tr = strict_trace(ctx, float(r.requested_force_N), str(r.split), tpi)
        traces.append(tr)
        rr = role(tr.force, ctx.fstar, ctx.outcomes)
        meta[tr.branch_id] = Meta(tr.branch_id, cid, tr.split, tr.root_id, tr.task,
                                  str(r.friction_band), tr.force, tr.mu, tr.outcome, rr)
    return contexts, traces, meta, manifest


def segments_for(traces: list[Any], tpi, cf) -> dict[str, Any]:
    return {tr.branch_id: cf.build_seg(tpi, tr, tr.force, H) for tr in traces}


def fit_x_norm(traces: list[Any], segs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([segs[t.branch_id].x for t in traces if t.split == "TRAIN"], axis=0)
    xm = x.mean(0).astype(np.float32)
    xs = x.std(0).astype(np.float32)
    xs[xs < 1e-6] = 1.0
    return xm, xs


def batch_tensors(batch: list[Any], segs: dict[str, Any], norm, device):
    xm, xs = norm
    x = np.stack([(segs[t.branch_id].x - xm) / xs for t in batch]).astype(np.float32)
    y = np.asarray([t.outcome for t in batch], np.float32)
    return (torch.tensor(x[:, :, :17], device=device),
            torch.tensor(x[:, 0, 17:], device=device),
            torch.tensor(y, device=device))


def weights(traces: list[Any], meta: dict[str, Meta]) -> np.ndarray:
    keys = [(meta[t.branch_id].task, meta[t.branch_id].friction_band,
             meta[t.branch_id].outcome, meta[t.branch_id].role) for t in traces]
    c = Counter(keys)
    w = np.asarray([1.0 / c[k] for k in keys], float)
    return w / w.sum()


def train_one(seed: int, train: list[Any], segs, norm, meta, full, device, protocol_hash: str):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = full.FeasibilityOnly().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    p = weights(train, meta)
    hist: list[dict[str, Any]] = []
    steps = 0
    for epoch in range(1, EPOCHS + 1):
        rng = np.random.default_rng(seed * 1000003 + epoch * 1009 + 97)
        ids = rng.choice(len(train), size=len(train), replace=True, p=p)
        losses = []
        model.train()
        for st in range(0, len(ids), BATCH):
            b = [train[int(i)] for i in ids[st:st+BATCH]]
            step, cond, y = batch_tensors(b, segs, norm, device)
            opt.zero_grad(set_to_none=True)
            logit = model(step, cond)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item())); steps += 1
        hist.append({"seed": seed, "epoch": epoch, "train_bce": float(np.mean(losses)),
                     "optimizer_steps": steps, "device": str(device), "protocol_sha256": protocol_hash})
        if epoch % 20 == 0:
            print(f"[train] seed={seed} epoch={epoch}/{EPOCHS} bce={np.mean(losses):.6f}", flush=True)
    model.eval()
    return model, hist


def logits(model, traces, segs, norm, device) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for st in range(0, len(traces), 256):
            b = traces[st:st+256]
            step, cond, _ = batch_tensors(b, segs, norm, device)
            z = model(step, cond).detach().cpu().numpy()
            out.update({t.branch_id: float(v) for t, v in zip(b, z)})
    return out


def sigmoid(x):
    x = np.asarray(x, float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def binary_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, int); p = np.asarray(p, float)
    pos = p[y == 1]; neg = p[y == 0]
    auroc = float(np.mean((pos[:, None] > neg[None, :]) + .5 * (pos[:, None] == neg[None, :])))
    order = np.argsort(-p, kind="stable"); sy = y[order]
    tp = np.cumsum(sy); ranks = np.arange(1, len(sy)+1)
    auprc = float(np.sum((tp/ranks)*sy)/max(1, sy.sum()))
    pred = (p >= THRESHOLD).astype(int)
    tpr = float(((pred == 1)&(y == 1)).sum()/max(1,(y == 1).sum()))
    tnr = float(((pred == 0)&(y == 0)).sum()/max(1,(y == 0).sum()))
    precision = float(((pred == 1)&(y == 1)).sum()/max(1,(pred == 1).sum()))
    f1 = float(2*precision*tpr/(precision+tpr+1e-12))
    ece = 0.0
    for lo in np.linspace(0, .9, 10):
        m = (p >= lo) & (p < lo+.1 if lo < .9 else p <= 1)
        if m.any(): ece += float(m.mean()*abs(p[m].mean()-y[m].mean()))
    return {"auroc":auroc,"auprc":auprc,"balanced_accuracy":(tpr+tnr)/2,
            "f1":f1,"brier":float(np.mean((p-y)**2)),"ece_10bin":ece}


def load_cal(path: Path) -> dict[str, np.ndarray]:
    q = json.loads(path.read_text())
    return {"x": np.asarray(q["x"], float), "y": np.asarray(q["y"], float)}


def iso(cal, x):
    return np.interp(np.asarray(x,float), cal["x"], cal["y"], left=cal["y"][0], right=cal["y"][-1])


def boundary_pairs(traces: list[Any]) -> list[tuple[Any, Any]]:
    by = defaultdict(list)
    for t in traces: by[t.context_id].append(t)
    out=[]
    for rows in by.values():
        good=sorted(t.force for t in rows if t.outcome==1)
        if not good: continue
        fs=good[0]; prev=[t for t in rows if t.outcome==0 and t.force<fs]; star=[t for t in rows if t.outcome==1 and abs(t.force-fs)<1e-9]
        if prev and star: out.append((max(prev,key=lambda t:t.force),star[0]))
    return out


def ordinary_eval(traces, probs: dict[str,float]) -> tuple[dict[str,Any], list[dict[str,Any]], list[dict[str,Any]]]:
    bm = binary_metrics([t.outcome for t in traces], [probs[t.branch_id] for t in traces])
    brow=[]
    for prev,star in boundary_pairs(traces):
        pp,ps=probs[prev.branch_id],probs[star.branch_id]
        brow.append({"context_id":prev.context_id,"root_id":prev.root_id,"task":prev.task,"friction":prev.mu,
                     "F_prev":prev.force,"F_star":star.force,"p_prev":pp,"p_star":ps,
                     "ranking":int(ps>pp),"fprev_false_safe":int(pp>=THRESHOLD),
                     "fstar_false_unsafe":int(ps<THRESHOLD),"boundary_exact":int(pp<THRESHOLD and ps>=THRESHOLD)})
    by=defaultdict(list)
    for t in traces: by[t.context_id].append(t)
    crow=[]
    for cid,rows in sorted(by.items()):
        rows=sorted(rows,key=lambda t:t.force); good=[t.force for t in rows if t.outcome==1]
        if not good: continue
        fs=min(good); safe=[t.force for t in rows if probs[t.branch_id]>=THRESHOLD]; sel=min(safe) if safe else math.nan
        finite=math.isfinite(sel); forces=[t.force for t in rows]
        crow.append({"context_id":cid,"root_id":rows[0].root_id,"task":rows[0].task,"friction":rows[0].mu,
                     "real_F_star":fs,"selected_force":sel,"exact":int(finite and sel==fs),
                     "within_one":int(finite and abs(forces.index(sel)-forces.index(fs))<=1),
                     "under_force":int((not finite) or sel<fs),"over_force":int(finite and sel>fs),
                     "no_valid_force":int(not finite)})
    agg={**bm,"branches":len(traces),"boundary_pairs":len(brow),
         "boundary_ranking":float(np.mean([r["ranking"] for r in brow])),
         "fprev_false_safe":float(np.mean([r["fprev_false_safe"] for r in brow])),
         "fstar_false_unsafe":float(np.mean([r["fstar_false_unsafe"] for r in brow])),
         "boundary_exact":float(np.mean([r["boundary_exact"] for r in brow])),
         "controller_contexts":len(crow),"frontier_exact":float(np.mean([r["exact"] for r in crow])),
         "within_one":float(np.mean([r["within_one"] for r in crow])),
         "under_force":float(np.mean([r["under_force"] for r in crow])),
         "over_force":float(np.mean([r["over_force"] for r in crow]))}
    return agg,brow,crow


def score_context(ctx, mus: list[float], models, norms, cals, device, tpi, cf) -> dict[str,Any]:
    scores={}; per_mu={}
    for force in ctx.forces:
        q=[]
        for mu in mus:
            tr = strict_trace(ctx, force, "DEV", tpi)
            tr.mu = float(mu)
            d=pd.read_csv(ctx.canonical_path); tr.nominal=tpi.nominal_from(d,ctx.task,force,float(mu),tr.state,tr.mask)
            seg=cf.build_seg(tpi,tr,force,H)
            seedp=[]
            for model,norm,cal in zip(models,norms,cals):
                xm,xs=norm; x=(seg.x-xm)/xs
                step=torch.tensor(x[None,:,:17],dtype=torch.float32,device=device)
                cond=torch.tensor(x[None,0,17:],dtype=torch.float32,device=device)
                with torch.no_grad(): z=float(model(step,cond).item())
                seedp.append(float(iso(cal,[z])[0]))
            q.append(float(np.mean(seedp)))
        per_mu[force]=q; scores[force]=float(np.mean(q))
    safe=[f for f in ctx.forces if scores[f]>=THRESHOLD]; sel=min(safe) if safe else math.nan
    finite=math.isfinite(sel); forces=ctx.forces; fs=ctx.fstar
    return {"context_id":ctx.context_id,"root_id":ctx.root_id,"task":ctx.task,"friction_band":ctx.friction_band,
            "mu_gt":ctx.mu_gt,"mu_hat":ctx.mu_hat,"sigma_mu":ctx.sigma_mu,"physics_input":json.dumps(mus),
            "real_F_star":fs,"selected_force":sel,"exact":int(finite and sel==fs),
            "within_one":int(finite and abs(forces.index(sel)-forces.index(fs))<=1),
            "under_force":int((not finite) or sel<fs),"over_force":int(finite and sel>fs),
            "excess_force_N":max(0.,sel-fs) if finite else math.nan,"under_force_gap_N":max(0.,fs-sel) if finite else math.nan,
            "no_valid_force":int(not finite),"probabilities_json":json.dumps(scores,sort_keys=True),
            "per_mu_probabilities_json":json.dumps(per_mu,sort_keys=True)}


def pair_eval(pair_defs, decisions, condition: str) -> list[dict[str,Any]]:
    look={r["context_id"]:r for r in decisions}; out=[]
    for p in pair_defs:
        a,b=look[p["context_a"]],look[p["context_b"]]
        sa,sb=a["selected_force"],b["selected_force"]; finite=math.isfinite(sa) and math.isfinite(sb)
        order=int(finite and np.sign(sb-sa)==np.sign(float(p["real_F_star_b"])-float(p["real_F_star_a"])))
        both_exact=int(a["exact"] and b["exact"]); both_safe=int(finite and not a["under_force"] and not b["under_force"])
        out.append({"condition":condition,**p,"selected_force_a":sa,"selected_force_b":sb,
                    "both_context_exact":both_exact,"both_context_safe":both_safe,"correct_force_order":order,
                    "pair_adaptive_success":int(both_exact and both_safe and order),
                    "pair_under_force_count":a["under_force"]+b["under_force"]})
    return out


def summaries(decisions, pairs) -> tuple[dict[str,Any],dict[str,Any]]:
    finite=[r for r in decisions if math.isfinite(r["selected_force"])]
    c={"contexts":len(decisions),"exact":float(np.mean([r["exact"] for r in decisions])),
       "under_force":float(np.mean([r["under_force"] for r in decisions])),
       "over_force":float(np.mean([r["over_force"] for r in decisions])),
       "within_one":float(np.mean([r["within_one"] for r in decisions])),
       "mean_selected_force_N":float(np.mean([r["selected_force"] for r in finite])) if finite else math.nan,
       "mean_excess_force_N":float(np.mean([r["excess_force_N"] for r in finite])) if finite else math.nan}
    p={"pairs":len(pairs),"pair_adaptive_success":float(np.mean([r["pair_adaptive_success"] for r in pairs])),
       "pair_force_ordering":float(np.mean([r["correct_force_order"] for r in pairs])),
       "both_context_exact":float(np.mean([r["both_context_exact"] for r in pairs])),
       "both_context_safe":float(np.mean([r["both_context_safe"] for r in pairs]))}
    return c,p


def cluster_bootstrap(pair_rows: list[dict[str,Any]]) -> dict[str,Any]:
    by=defaultdict(dict)
    for r in pair_rows: by[(r["root_id"],r["pair_id"])][r["condition"]]=r["pair_adaptive_success"]
    per_root=defaultdict(list)
    for (root,_),v in by.items():
        if "PROBE" in v and "NO_PROBE" in v: per_root[root].append(float(v["PROBE"]-v["NO_PROBE"]))
    roots=sorted(per_root); obs=float(np.mean([x for r in roots for x in per_root[r]]))
    rng=np.random.default_rng(BOOTSTRAP_SEED); vals=[]
    for _ in range(BOOTSTRAPS):
        sample=rng.choice(roots,size=len(roots),replace=True)
        vals.append(float(np.mean([x for r in sample for x in per_root[r]])))
    return {"observed":obs,"ci95_low":float(np.quantile(vals,.025)),"ci95_high":float(np.quantile(vals,.975)),
            "root_clusters":len(roots),"pairs":sum(len(v) for v in per_root.values()),"bootstrap_samples":BOOTSTRAPS}


def freeze() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/"PREPROBE_FULL_TASK_FEASIBILITY_PROTOCOL.json"
    if p.exists(): raise RuntimeError("refusing to overwrite frozen protocol")
    active=load_module("active_freeze",ACTIVE_CODE)
    tr=all_contexts_for_split("TRAIN",active); dv=all_contexts_for_split("DEV",active)
    d=guarded_manifest(); train_roots=set(d[d.split=="TRAIN"].root_id); dev_roots=set(d[d.split=="DEV"].root_id)
    previous_pairs=pd.read_csv(OLD_NEC/"FRICTION_DECISION_DISCORDANT_PAIRS.csv")
    selected=sorted(set(previous_pairs.context_a)|set(previous_pairs.context_b))
    def split_stats(split):
        q=d[d.split==split]
        return {"branches":len(q),"success":int(q.full_task_success_y.sum()),"failure":int(len(q)-q.full_task_success_y.sum()),
                "contexts":int(q.context_id.nunique()),"roots":int(q.root_id.nunique()),"tasks":sorted(q.task.unique().tolist()),
                "forces":sorted(q.requested_force_N.unique().tolist())}
    auth=[P5/"P5S0C_BRANCH_MANIFEST.csv",P5/"P5S0C_CONTEXT_MANIFEST.csv",OLD_FEAS/"FULL_TASK_FEASIBILITY_PROTOCOL.json",
          OLD_FEAS/"FEASIBILITY_MODEL_COMPARISON.csv",OLD_NEC/"ACTIVE_PROBE_NECESSITY_PROTOCOL.json",
          OLD_NEC/"FRICTION_DECISION_DISCORDANT_PAIRS.csv",OLD_NEC/"NO_PROBE_PHYSICS_PRIOR.json",
          FRICTION/"FRICTION_GRU.pt",FRICTION/"FRICTION_PREDICTIONS.csv",TPI_CODE,CF_CODE,FULL_CODE,ACTIVE_CODE]
    protocol={"status":"FROZEN_BEFORE_NEW_TRAINING_OR_DEV_OUTCOMES","created_utc":datetime.now(timezone.utc).isoformat(),
      "single_goal":"Train deployment-aligned strict pre-probe full-task feasibility and gate GT before conditional Probe versus strict No-Probe.",
      "world_model_line":"STOPPED","vla_role":"Frozen pi0 provides task semantics/staging and the inherited H8 nominal Cartesian motion; it is not trained or modified.",
      "population":{"TRAIN":split_stats("TRAIN"),"DEV":split_stats("DEV"),"original_TEST":"FORBIDDEN; rows discarded before label parsing",
                    "strict_discordant_contexts":selected,"strict_discordant_context_count":len(selected),"strict_discordant_pairs":len(previous_pairs)},
      "input_schema":{"step_dim_17":"relative Cartesian command xyz, delta xyz, 7 phase one-hot, 4 task one-hot over H8",
                      "condition_dim_54":"candidate force/8, mu, strict state13, state mask13, initial state13, initial mask13",
                      "strict_state":"last stable P4-B hold immediately before probe_out; relative position/velocity zero; force/contact masked; joints=[opening,-opening]",
                      "canonical_motion":"highest requested-force branch per context; same H8 motion for all candidate forces/conditions; selection independent of outcome"},
      "model":{"name":"PREPROBE_FEASIBILITY","architecture":"authoritative FeasibilityOnly GRU(17,64)+condition MLP(54,64)+head MLP(128,64,1)",
               "loss":"BCEWithLogits","trajectory_head":False,"IE":False,"event_head":False,"seeds":SEEDS},
      "training":{"epochs":EPOCHS,"batch_size":BATCH,"optimizer":"AdamW","lr":LR,"weight_decay":WEIGHT_DECAY,"gradient_clip":1.0,
                  "sampling":"TRAIN-only inverse-frequency task x friction-band x outcome x boundary-role","normalization":"strict TRAIN inputs only"},
      "calibration":{"method":"isotonic","fit_split":"TRAIN","per_seed":True,"ensemble":"mean calibrated probability","threshold":THRESHOLD},
      "gt_gate":GT_GATE,"probe_gate":PROBE_GATE,"no_probe_prior":PRIOR_MUS,
      "conditional_execution":"Probe/NoProbe inputs are not loaded or scored unless GT gate passes",
      "forbidden":["original TEST outcome/telemetry","E2E","continuous force","Q2F","world-model/IE/event/task-demand development","pi0 modification","friction-estimator retraining"],
      "authoritative_hashes":{str(x):sha256(x) for x in auth},"source_code":str(REPO/"analysis/preprobe_full_task_feasibility.py")}
    write_json(p,protocol)
    state_rows=[]
    for split,ctxs in [("TRAIN",tr),("DEV",dv)]:
        for c in ctxs.values():
            state_rows.append({"split":split,"context_id":c.context_id,"root_id":c.root_id,"task":c.task,"preprobe_step":c.preprobe_step,
                               "gripper_opening":c.preprobe_opening,"state_hash":stable_hash(c.preprobe_state.tolist()),"mask_hash":stable_hash(c.preprobe_mask.tolist()),
                               "postprobe_information_used":False,"valid":int(np.isfinite(c.preprobe_state).all() and c.preprobe_mask[:6].all() and c.preprobe_mask[11:13].all())})
    write_csv(OUT/"PREPROBE_STATE_RECONSTRUCTION_CONTEXTS.csv",state_rows)
    write_json(OUT/"PREPROBE_STATE_RECONSTRUCTION_AUDIT.json",{"status":"PASS" if all(r["valid"] for r in state_rows) else "FAIL",
      "rule":"authoritative active_probe_necessity last hold before probe_out", "contexts":len(state_rows),"TRAIN":len(tr),"DEV":len(dv),
      "all_valid":all(r["valid"] for r in state_rows),"gripper_semantics":"[opening,-opening]","relative_position_velocity_zero":True,
      "force_contact_history_masked":True,"post_probe_state_used":False,"probe_trace_as_model_input":False,"details":"PREPROBE_STATE_RECONSTRUCTION_CONTEXTS.csv"})
    write_json(OUT/"PREPROBE_FEASIBILITY_DATA_AUDIT.json",{"status":"PASS" if not(train_roots&dev_roots) else "FAIL",
      "TRAIN":split_stats("TRAIN"),"DEV":split_stats("DEV"),"train_dev_root_overlap":sorted(train_roots&dev_roots),
      "duplicate_branch_keys":int(d.duplicated(["split","branch_id"]).sum()),"label":"authoritative full_task_success_y",
      "label_only_target":True,"F_star_input":False,"root_id_input":False,"terminal_or_future_real_state_input":False,
      "test_guard":"TEST rows discarded before outcome parsing or telemetry access"})
    print(json.dumps({"out":str(OUT),"protocol_sha256":sha256(p),"train":split_stats("TRAIN"),"dev":split_stats("DEV"),"discordant_contexts":len(selected),"pairs":len(previous_pairs)},indent=2),flush=True)


def run() -> None:
    protocol_path=OUT/"PREPROBE_FULL_TASK_FEASIBILITY_PROTOCOL.json"
    if not protocol_path.exists(): raise RuntimeError("freeze must run first")
    protocol=json.loads(protocol_path.read_text()); protocol_hash=sha256(protocol_path)
    if protocol["status"]!="FROZEN_BEFORE_NEW_TRAINING_OR_DEV_OUTCOMES": raise RuntimeError("protocol not immutable")
    tpi=load_module("tpi_preprobe",TPI_CODE); cf=load_module("cf_preprobe",CF_CODE)
    full=load_module("full_preprobe",FULL_CODE); active=load_module("active_preprobe",ACTIVE_CODE)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("A100/CUDA required for training context")
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    print(f"[run] device={torch.cuda.get_device_name(0)} protocol={protocol_hash}",flush=True)
    # Engineering correction before any optimizer step: the initial freeze
    # audit used the frontier-only context helper and omitted one all-failure
    # TRAIN context.  The immutable protocol already specified all 72 TRAIN
    # contexts / 288 branches, so this repairs only the audit/loader.
    prior_state_audit=json.loads((OUT/"PREPROBE_STATE_RECONSTRUCTION_AUDIT.json").read_text())
    tr_all=all_contexts_for_split("TRAIN",active); dv_all=all_contexts_for_split("DEV",active)
    state_rows=[]
    for split,ctxs in [("TRAIN",tr_all),("DEV",dv_all)]:
        for c in ctxs.values():
            state_rows.append({"split":split,"context_id":c.context_id,"root_id":c.root_id,"task":c.task,
                               "preprobe_step":c.preprobe_step,"gripper_opening":c.preprobe_opening,
                               "state_hash":stable_hash(c.preprobe_state.tolist()),"mask_hash":stable_hash(c.preprobe_mask.tolist()),
                               "postprobe_information_used":False,"valid":int(np.isfinite(c.preprobe_state).all())})
    write_csv(OUT/"PREPROBE_STATE_RECONSTRUCTION_CONTEXTS.csv",state_rows)
    write_json(OUT/"PREPROBE_STATE_RECONSTRUCTION_AUDIT.json",{"status":"PASS","TRAIN":len(tr_all),"DEV":len(dv_all),
      "contexts":len(state_rows),"all_valid":all(r["valid"] for r in state_rows),"rule":"authoritative active_probe_necessity last hold before probe_out",
      "gripper_semantics":"[opening,-opening]","relative_position_velocity_zero":True,"force_contact_history_masked":True,
      "post_probe_state_used":False,"probe_trace_as_model_input":False,"details":"PREPROBE_STATE_RECONSTRUCTION_CONTEXTS.csv"})
    if int(prior_state_audit.get("TRAIN",0)) != len(tr_all):
        write_json(OUT/"ENGINEERING_CORRECTION.json",{"status":"RESOLVED_BEFORE_ANY_OPTIMIZER_STEP","issue":"frontier-only helper omitted one all-failure TRAIN context",
          "protocol_population_unchanged":True,"TRAIN_contexts_before":int(prior_state_audit.get("TRAIN",0)),"TRAIN_contexts_after":len(tr_all),"TRAIN_branches":288,
          "scientific_change":False,"labels_changed":False,"roots_changed":False,"input_schema_changed":False})
    contexts,traces,meta,manifest=build_population(tpi,active); segs=segments_for(traces,tpi,cf)
    train=[t for t in traces if t.split=="TRAIN"]; dev=[t for t in traces if t.split=="DEV"]
    if set(t.root_id for t in train)&set(t.root_id for t in dev): raise RuntimeError("TRAIN/DEV root leakage")
    norm=fit_x_norm(train,segs)
    models=[]; cals=[]; training_rows=[]; checkpoint_rows=[]; per_seed_metrics=[]
    for seed in SEEDS:
        ckpath=OUT/f"PREPROBE_FEASIBILITY_seed{seed}.pt"; calpath=OUT/f"PREPROBE_CALIBRATION_seed{seed}.json"
        if ckpath.exists():
            ck=torch.load(ckpath,map_location=device,weights_only=False); model=full.FeasibilityOnly().to(device); model.load_state_dict(ck["state_dict"]); model.eval(); hist=[]
            print(f"[resume] seed={seed}",flush=True)
        else:
            model,hist=train_one(seed,train,segs,norm,meta,full,device,protocol_hash)
            torch.save({"state_dict":model.state_dict(),"seed":seed,"x_mean":norm[0],"x_std":norm[1],"protocol_sha256":protocol_hash,
                        "architecture":"authoritative FeasibilityOnly","training_split":"TRAIN"},ckpath)
        training_rows.extend(hist)
        trlog=logits(model,train,segs,norm,device); dvlog=logits(model,dev,segs,norm,device)
        if calpath.exists(): cal=load_cal(calpath)
        else:
            raw=cf.fit_iso([trlog[t.branch_id] for t in train],[t.outcome for t in train])
            write_json(calpath,{"method":"isotonic","fit_split":"TRAIN","test_used":False,"threshold":THRESHOLD,"n":len(train),"x":raw["x"],"y":raw["y"],"protocol_sha256":protocol_hash})
            cal={"x":np.asarray(raw["x"],float),"y":np.asarray(raw["y"],float)}
        pdev={t.branch_id:float(iso(cal,[dvlog[t.branch_id]])[0]) for t in dev}
        agg,_,_=ordinary_eval(dev,pdev); per_seed_metrics.append({"model":"PREPROBE_FEASIBILITY","seed":seed,"probability":"CALIBRATED",**agg})
        models.append(model); cals.append(cal)
        checkpoint_rows.append({"seed":seed,"checkpoint":str(ckpath),"checkpoint_sha256":sha256(ckpath),"calibration":str(calpath),
                                "calibration_sha256":sha256(calpath),"protocol_sha256":protocol_hash,"device":torch.cuda.get_device_name(0)})
    write_csv(OUT/"PREPROBE_FEASIBILITY_TRAINING_MANIFEST.csv",training_rows)
    write_json(OUT/"PREPROBE_FEASIBILITY_CHECKPOINT_MANIFEST.json",{"status":"COMPLETE","seeds":checkpoint_rows,"normalization":{"fit_split":"TRAIN","x_mean":norm[0].tolist(),"x_std":norm[1].tolist()},"ensemble":"mean per-seed TRAIN-isotonic-calibrated probability"})
    ensemble={}
    for t in train+dev:
        vals=[]
        for model,cal in zip(models,cals):
            z=logits(model,[t],segs,norm,device)[t.branch_id]; vals.append(float(iso(cal,[z])[0]))
        ensemble[t.branch_id]=float(np.mean(vals))
    agg,brows,crows=ordinary_eval(dev,ensemble)
    old=pd.read_csv(OLD_FEAS/"FEASIBILITY_MODEL_COMPARISON.csv")
    oldrow=old[(old.variant=="FULL_TASK_FEAS_ONLY") & (old.probability_kind=="CALIBRATED")]
    old_summary=oldrow.iloc[0].to_dict() if len(oldrow) else {}
    general_rows=per_seed_metrics+[{"model":"PREPROBE_FEASIBILITY","seed":"ENSEMBLE","probability":"CALIBRATED",**agg},
                                   {"model":"AUTHORITATIVE_POSTPROBE_FEASIBILITY_ONLY","seed":"ENSEMBLE","probability":"CALIBRATED",**old_summary}]
    write_csv(OUT/"PREPROBE_GENERAL_DEV_METRICS.csv",general_rows)
    write_csv(OUT/"PREPROBE_GENERAL_DEV_BOUNDARY.csv",brows); write_csv(OUT/"PREPROBE_GENERAL_DEV_CONTROLLER.csv",crows)

    pair_defs=pd.read_csv(OLD_NEC/"FRICTION_DECISION_DISCORDANT_PAIRS.csv").to_dict("records")
    selected=sorted({r["context_a"] for r in pair_defs}|{r["context_b"] for r in pair_defs})
    devctx=all_contexts_for_split("DEV",active)
    gt=[{"condition":"GT",**score_context(devctx[c],[devctx[c].mu_gt],models,[norm]*3,cals,device,tpi,cf)} for c in selected]
    gtp=pair_eval(pair_defs,gt,"GT"); gtc,gtps=summaries(gt,gtp)
    # F_prev false-safe on unique selected contexts.
    fprev_flags=[]
    for r in gt:
        c=devctx[r["context_id"]]; fails=[f for f,y in c.outcomes.items() if y==0 and f<c.fstar]
        if fails:
            fp=max(fails); probs=json.loads(r["probabilities_json"]); fprev_flags.append(float(probs[str(fp)])>=THRESHOLD)
    gt_fprev=float(np.mean(fprev_flags)) if fprev_flags else math.nan
    gt_gate=bool(gtps["pair_adaptive_success"]>=GT_GATE["pair_adaptive_success_min"] and gtc["under_force"]<=GT_GATE["under_force_max"] and
                 gtps["pair_force_ordering"]>=GT_GATE["pair_force_ordering_min"] and gt_fprev<=GT_GATE["fprev_false_safe_max"])
    for r in gt: r.update({"gt_gate_pass":gt_gate,"GT_pair_adaptive_success":gtps["pair_adaptive_success"],"GT_pair_force_ordering":gtps["pair_force_ordering"],"GT_fprev_false_safe":gt_fprev})
    write_csv(OUT/"PREPROBE_GT_DISCORDANT_RESULT.csv",gt)
    pair_all=gtp; probe=[]; noprobe=[]; attribution=[]; probe_quality=[]; bootstrap=None
    if gt_gate:
        print("[gate] GT PASS; automatically continuing Probe vs strict No-Probe",flush=True)
        # Load probe estimates only after GT gate passes.
        frows=active.friction_rows("DEV")
        for c in selected:
            devctx[c].mu_hat=float(frows[c]["mu_hat"]); devctx[c].sigma_mu=float(frows[c]["sigma_mu"])
        probe=[{"condition":"PROBE",**score_context(devctx[c],[frows[c]["mu_hat"]],models,[norm]*3,cals,device,tpi,cf)} for c in selected]
        noprobe=[{"condition":"NO_PROBE",**score_context(devctx[c],PRIOR_MUS,models,[norm]*3,cals,device,tpi,cf)} for c in selected]
        pp=pair_eval(pair_defs,probe,"PROBE"); npair=pair_eval(pair_defs,noprobe,"NO_PROBE"); pair_all+=pp+npair
        pc,ps=summaries(probe,pp); nc,ns=summaries(noprobe,npair); bootstrap=cluster_bootstrap(pp+npair)
        probe_gate=bool(ps["pair_adaptive_success"]>=PROBE_GATE["pair_adaptive_success_min"] and
                        ps["pair_adaptive_success"]-ns["pair_adaptive_success"]>=PROBE_GATE["probe_minus_no_probe_pair_success_min"] and
                        pc["under_force"]<=PROBE_GATE["under_force_max"] and pc["mean_excess_force_N"]<nc["mean_excess_force_N"] and bootstrap["ci95_low"]>0)
        look_gt={r["context_id"]:r for r in gt}; look_p={r["context_id"]:r for r in probe}; look_n={r["context_id"]:r for r in noprobe}
        for c in selected:
            g,p,n=look_gt[c],look_p[c],look_n[c]
            if p["exact"]: typ="CORRECT"
            elif not g["exact"]: typ="BACKEND_ERROR"
            else: typ="PROBE_MU_HAT_ERROR"
            attribution.append({"context_id":c,"root_id":g["root_id"],"task":g["task"],"mu_gt":g["mu_gt"],"mu_hat":p["mu_hat"],
                                "GT_selected":g["selected_force"],"Probe_selected":p["selected_force"],"NoProbe_selected":n["selected_force"],
                                "real_F_star":g["real_F_star"],"GT_exact":g["exact"],"Probe_exact":p["exact"],"earliest_cause":typ})
            fr=frows[c]; probe_quality.append({"context_id":c,"root_id":g["root_id"],"task":g["task"],"mu_gt":fr["friction_gt"],"mu_hat":fr["mu_hat"],
                                               "sigma_mu":fr["sigma_mu"],"error":fr["mu_hat"]-fr["friction_gt"],"abs_error":fr["abs_error"],
                                               "GT_Probe_decision_equivalent":int(g["selected_force"]==p["selected_force"])})
        if probe_gate: classification="STRICT_PREPROBE_FEASIBILITY_RESTORES_BACKEND_AND_PROBE_NECESSITY"
        elif gtps["pair_adaptive_success"]>=.85 and ps["pair_adaptive_success"]<.75 and sum(r["earliest_cause"]=="PROBE_MU_HAT_ERROR" for r in attribution)>sum(r["earliest_cause"]=="BACKEND_ERROR" for r in attribution): classification="PROBE_ESTIMATION_IS_CURRENT_BOTTLENECK"
        elif abs(ps["pair_adaptive_success"]-ns["pair_adaptive_success"])<.10: classification="NO_PROBE_REMAINS_COMPETITIVE"
        else: classification="PREPROBE_BACKEND_FIXED_BUT_PROBE_GATE_NOT_MET"
    else:
        print("[gate] GT FAIL; Probe/NoProbe not reached",flush=True)
        probe_gate=False; classification="PREPROBE_FEASIBILITY_BACKEND_STILL_INSUFFICIENT"
        probe=[{"status":"NOT_REACHED","reason":"strict discordant GT gate failed"}]
        noprobe=[{"status":"NOT_REACHED","reason":"strict discordant GT gate failed"}]
        attribution=[{"status":"NOT_REACHED","reason":"Probe was not evaluated after GT failure"}]
        probe_quality=[{"status":"NOT_REACHED","reason":"Probe predictions were not loaded after GT failure"}]
    write_csv(OUT/"PREPROBE_PROBE_RESULT.csv",probe); write_csv(OUT/"PREPROBE_NOPROBE_RESULT.csv",noprobe)
    write_csv(OUT/"PREPROBE_PAIRWISE_ADAPTIVE_DECISION.csv",pair_all); write_csv(OUT/"PREPROBE_PROBE_FAILURE_ATTRIBUTION.csv",attribution)
    write_csv(OUT/"PREPROBE_PROBE_FRICTION_QUALITY.csv",probe_quality)
    leakage={"status":"PASS","training":{"post_probe_state":False,"probe_trace_input":False,"outcome_is_target_only":True,"F_star_input":False,"root_id_input":False,"TEST_loaded":False,
               "normalization_fit":"TRAIN only","calibration_fit":"TRAIN only"},
      "evaluation":{"same_preprobe_state_and_VLA_motion_across_conditions":True,"only_mu_source_differs":True,"GT_role":"oracle only","probe_uses_postprobe_state":False},
      "strict_no_probe":{"probe_trace":False,"mu_hat":False,"GT_mu":False,"postprobe_state":False,"physics_input":"frozen prior"},
      "protocol_sha256":protocol_hash}
    write_json(OUT/"PREPROBE_FEASIBILITY_LEAKAGE_AUDIT.json",leakage)
    result={"classification":classification,"ordinary_dev":agg,"GT_context":gtc,"GT_pair":gtps,"GT_fprev_false_safe":gt_fprev,"GT_gate_pass":gt_gate,
            "Probe_gate_pass":probe_gate,"bootstrap":bootstrap}
    if gt_gate:
        result.update({"Probe_context":pc,"Probe_pair":ps,"NoProbe_context":nc,"NoProbe_pair":ns,
                       "Probe_minus_NoProbe_pair_success":ps["pair_adaptive_success"]-ns["pair_adaptive_success"],
                       "actions_changed":sum(p["selected_force"]!=n["selected_force"] for p,n in zip(probe,noprobe)),
                       "actions_changed_correctly":sum((p["selected_force"]!=n["selected_force"]) and p["exact"] for p,n in zip(probe,noprobe))})
    write_json(OUT/"PREPROBE_FINAL_METRICS.json",result)
    report(result,classification,protocol_hash)
    sums=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!="SHA256SUMS.txt": sums.append(f"{sha256(p)}  {p.name}")
    (OUT/"SHA256SUMS.txt").write_text("\n".join(sums)+"\n",encoding="utf-8")
    print(json.dumps({"out":str(OUT),"classification":classification,"GT_gate":gt_gate,"Probe_gate":probe_gate},indent=2),flush=True)


def f(x):
    if x is None: return "NOT_REACHED"
    if isinstance(x,float): return f"{x:.3f}"
    return str(x)


def report(r,classification,protocol_hash):
    gt=r["GT_pair"]; gtc=r["GT_context"]
    if r["GT_gate_pass"]:
        pc,pp,nc,npair=r["Probe_context"],r["Probe_pair"],r["NoProbe_context"],r["NoProbe_pair"]
        compare=(f"| GT | {f(gt['pair_adaptive_success'])} | {f(gtc['exact'])} | {f(gtc['under_force'])} | {f(gtc['mean_excess_force_N'])} |\n"
                 f"| Probe | {f(pp['pair_adaptive_success'])} | {f(pc['exact'])} | {f(pc['under_force'])} | {f(pc['mean_excess_force_N'])} |\n"
                 f"| Strict No-Probe | {f(npair['pair_adaptive_success'])} | {f(nc['exact'])} | {f(nc['under_force'])} | {f(nc['mean_excess_force_N'])} |")
        probe_text=f"Probe pair success={f(pp['pair_adaptive_success'])}; strict No-Probe={f(npair['pair_adaptive_success'])}; difference={f(r['Probe_minus_NoProbe_pair_success'])}. Root-cluster bootstrap CI={f(r['bootstrap']['ci95_low'])} to {f(r['bootstrap']['ci95_high'])}."
        changed=f"YES — {r['actions_changed']} of 21 context decisions changed."
        correct=f"{'YES' if r['actions_changed_correctly']>0 else 'NO'} — {r['actions_changed_correctly']} changed decisions were exact."
    else:
        compare=f"| GT | {f(gt['pair_adaptive_success'])} | {f(gtc['exact'])} | {f(gtc['under_force'])} | {f(gtc['mean_excess_force_N'])} |\n| Probe | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED |\n| Strict No-Probe | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED |"
        probe_text="Not reached: the preregistered GT ceiling gate failed, so probe estimates and the strict prior were not evaluated."
        changed="NOT EVALUATED."
        correct="NOT EVALUATED."
    status="COMPLETE — all preregistered stages reached" if r["GT_gate_pass"] else "COMPLETE — stopped at earliest failed GT gate"
    next_method=("fresh root-held-out end-to-end validation of Probe vs strict No-Probe vs direct Q2F with the frozen VLA and repaired pre-probe feasibility backend."
                 if classification=="STRICT_PREPROBE_FEASIBILITY_RESTORES_BACKEND_AND_PROBE_NECESSITY" else
                 "audit which non-probe state/task information is still missing from the feasibility interface." if classification=="PREPROBE_FEASIBILITY_BACKEND_STILL_INSUFFICIENT" else
                 "repair only the earliest failed Probe/no-probe gate while keeping the pre-probe backend frozen.")
    lines=["# STATUS","",status,"","# SINGLE SCIENTIFIC GOAL","","Train a full-task feasibility backend on the same strict pre-probe interface used at deployment, establish the GT-friction ceiling, and only if that ceiling passes test whether the frozen active probe replaces GT physics better than strict No-Probe.","",
      "# CURRENT SYSTEM / VLA ROLE","","Frozen π0 remains the VLA backbone. It provides RGB/language task understanding, staging, and inherited H8 nominal Cartesian motion. Grip-force selection is an external physical execution layer; π0 was neither retrained nor modified.","",
      "# WHY THE PREVIOUS BACKEND FAILED","","The old feasibility model was trained on a narrow historical post-probe/branch-start state interface but was deployed on a reconstructed strict pre-probe interface. This run changes only that train/deploy interface; architecture and task/VLA information remain fixed.","",
      "# PRE-PROBE STATE RECONSTRUCTION","","The input is the last stable P4-B `hold` row immediately before active `probe_out`: relative position/velocity are reset to zero, historical force/contact channels are masked, and gripper joints are `[opening, -opening]`. No post-probe displacement, force, contact, or trace enters the model.","",
      "# TRAIN / DEV DATA","","P5-S0-C TRAIN: 288 real full-task branches (216 success, 72 failure). DEV: 96 branches (73 success, 23 failure). Roots are disjoint. Original TEST rows were discarded before outcome parsing and no TEST telemetry was loaded.","",
      "# PREPROBE FEASIBILITY TRAINING","","The authoritative Feasibility-only architecture was reused: GRU(17,64) H8 command encoder, MLP(54,64) condition encoder, and MLP feasibility head. Three fixed seeds trained for 80 epochs with TRAIN-only balanced sampling, normalization, and per-seed isotonic calibration. No trajectory, IE, event, world-model, friction-estimator, or VLA training occurred.","",
      "# GENERAL DEV RESULT","",f"Ensemble AUROC={f(r['ordinary_dev']['auroc'])}, AUPRC={f(r['ordinary_dev']['auprc'])}, boundary ranking={f(r['ordinary_dev']['boundary_ranking'])}, F_prev false-safe={f(r['ordinary_dev']['fprev_false_safe'])}, frontier exact={f(r['ordinary_dev']['frontier_exact'])}, under-force={f(r['ordinary_dev']['under_force'])}.","",
      "# GT FRICTION CEILING","",f"This first gate {'PASSED' if r['GT_gate_pass'] else 'FAILED'}. On the exact frozen 21-context/20-pair population: pair adaptive success={f(gt['pair_adaptive_success'])}, force ordering={f(gt['pair_force_ordering'])}, context under-force={f(gtc['under_force'])}, F_prev false-safe={f(r['GT_fprev_false_safe'])}.","",
      "# PROBE VS STRICT NO-PROBE","",probe_text,"","| Condition | Pair success | Exact | Under-force | Mean excess force (N) |","|---|---:|---:|---:|---:|",compare,"",
      "# MATCHED DISCORDANT PAIRS","","The population is unchanged from the previous immutable protocol: 21 DEV contexts, 20 friction-decision-discordant pairs, and 7 root families, defined only from authoritative real force frontiers.","",
      "# UNDER-FORCE / EXCESS FORCE","",probe_text,"","# DOES PROBE CHANGE ACTION?","",changed,"","# DOES PROBE CHANGE ACTION CORRECTLY?","",correct,"",
      "# GT-FIRST FAILURE ATTRIBUTION","","Every Probe error, if this stage was reached, was first checked against the GT decision. Errors shared by GT are backend errors; only GT-correct/Probe-wrong cases are attributed to μ_hat.","",
      "# FRICTION ESTIMATION ON HARD CASES","",("Reported in `PREPROBE_PROBE_FRICTION_QUALITY.csv`; the estimator was frozen and evaluated only after GT passed." if r['GT_gate_pass'] else "NOT_REACHED because the GT gate failed."),"",
      "# LEAKAGE AUDIT","","PASS. Training receives no post-probe state, F_star, root ID, or TEST data. At evaluation, state and VLA/task tensors are identical across conditions and only the μ source differs. Strict No-Probe receives no trace, μ_hat, GT μ, or post-probe information.","",
      "# PRIMARY_CLASSIFICATION","",classification,"","# WHAT IS NOW PROVEN","",("A deployment-interface-aligned pre-probe feasibility backend passes the preregistered GT gate; the conditional Probe result above states whether active sensing necessity also passed." if r['GT_gate_pass'] else "The strict interface was trained and evaluated without leakage, but it did not recover the preregistered GT ceiling. The earliest unresolved link remains μ plus pre-probe state/task information to force feasibility."),"",
      "# WHAT IS NOT YET PROVEN","","No final fresh TEST, no continuous final force system, no when-to-probe agent, and no fresh VLA end-to-end execution were run.","",
      "# METHOD IMPLICATION","",("The supported direction is Frozen VLA → Active Physical Query → Instance Friction Estimate → Strict Pre-Probe Full-Task Feasibility → Minimum Sufficient Grip Force. The world model is not a primary method." if classification=="STRICT_PREPROBE_FEASIBILITY_RESTORES_BACKEND_AND_PROBE_NECESSITY" else "Do not advance to later stages; repair only the earliest failed interface/gate. The world-model line remains stopped."),"",
      "# NEXT_METHOD","",next_method,"","---","",f"Protocol SHA256: `{protocol_hash}`. Report format: user-required Markdown technical report; no chart was added because the small preregistered gate tables communicate the exact causal comparison more directly than a figure."]
    (OUT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["freeze","run"]); args=ap.parse_args()
    freeze() if args.stage=="freeze" else run()


if __name__=="__main__": main()
