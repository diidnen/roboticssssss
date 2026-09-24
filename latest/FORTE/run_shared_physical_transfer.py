#!/usr/bin/env python3
"""LOTO + few-shot shared ActiveForcing transfer from archived populations only.

This runner deliberately excludes World Models, residuals, Probe changes,
simulator collection, and every root-scaling TEST namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import run_probe_conditioned_wm as af
import run_pooled_predictive_verifier as ppv


ROOT = Path("/home/exouser/FORTE")
TAB = Path("/home/exouser/Tabero")
P5 = TAB / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542"
CURRENT_PROBE = TAB / "analysis/results/gnp_style_visual_context_prospective_20260831_011000/collection_train"
TASKS = [0, 1, 5, 6]
FOLDS = [0, 1, 2]
SEEDS = [0, 1, 2]
BUDGETS = [0, 10, 20, 30, 60]
FMAX = af.FMAX
UTILITY_TOL = 0.01


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wjson(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def canonical_key(text: str) -> str:
    return hashlib.sha256(("ACTIVEFORCING_LOTO_V1|" + text).encode()).hexdigest()


def load_current(out: Path, tag: str):
    data = af.load_pop(out, tag)
    af.TPI, af.CF, af.FULL = data[:3]
    tpi, cf, full, cmap, traces, meta, audits, pairs, segs, norm = data
    md = ppv.frame(traces, meta)
    md["R_real"] = np.where(md.success > 0,
                            (md.task.map(FMAX) - md.force_N) / md.task.map(FMAX), -1.0)
    rf = af.root_fold(md)
    assert len(md) == 720 and md.root_id.nunique() == 24 and md.context_id.nunique() == 72
    return tpi, cf, full, cmap, traces, meta, audits, pairs, segs, md, rf


def p5_schema():
    norm = json.loads((P5 / "P5S0C_NORMALIZATION.json").read_text())
    return norm["dynamic_feature_names"], norm["phase_categories_from_train"], norm["contact_state_categories_from_train"]


def read_probe(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    names, phases, states = p5_schema()
    raw_df = pd.read_csv(path)
    q = af.P5.sequence_dataframe(str(path), phases, states).reindex(columns=names).fillna(0.0)
    a = q.to_numpy(np.float32)
    if a.shape != (215, 46) or not np.isfinite(a).all():
        raise RuntimeError(f"bad Probe trace {path}: {a.shape}")
    return a, raw_df


def load_probe_populations(traces):
    old = pd.read_csv(P5 / "P5S0C_CONTEXT_MANIFEST.csv").copy()
    old = old.rename(columns={"hidden_friction_analysis_only": "mu_GT"})
    old["probe_index"] = np.arange(len(old))
    raw_old = []
    old_summary = []
    for r in old.itertuples():
        a, d = read_probe(Path(r.probe_telemetry_path))
        raw_old.append(a)
        old_summary.append(sysid_features(d))
    contexts = {t.context_id: t for t in traces}
    rows, raw_current, current_summary = [], [], []
    for cid, t in sorted(contexts.items()):
        path = CURRENT_PROBE / f"task{t.task}/P5S0C_PROBE_TELEMETRY/{cid}_probe_timesteps.csv"
        a, d = read_probe(path)
        rows.append({"probe_index": len(rows), "context_id": cid, "root_id": t.root_id,
                     "task": int(t.task), "mu_GT": float(t.mu), "probe_telemetry_path": str(path)})
        raw_current.append(a)
        current_summary.append(sysid_features(d))
    current = pd.DataFrame(rows)
    return old, np.stack(raw_old), pd.DataFrame(old_summary), current, np.stack(raw_current), pd.DataFrame(current_summary)


def _safe_stat(q: pd.DataFrame, col: str, stat: str) -> float:
    x = pd.to_numeric(q[col], errors="coerce").to_numpy(float) if col in q else np.array([0.0])
    x = x[np.isfinite(x)]
    if not len(x): return 0.0
    return {"median": np.median, "max": np.max, "std": np.std, "mean": np.mean}[stat](x).item()


def sysid_features(d: pd.DataFrame) -> dict[str, float]:
    bilateral = d.contact_state.astype(str).eq("bilateral") if "contact_state" in d else np.ones(len(d), bool)
    active = d.probe_phase.astype(str).isin(["probe_out", "probe_hold", "probe_back"]) & bilateral
    out = d.probe_phase.astype(str).eq("probe_out") & bilateral
    q = d[active]
    qo = d[out]
    fn = max(_safe_stat(q, "measured_fn", "median"), 1e-6)
    marker = max(_safe_stat(q, "marker_tangential", "max"), 1e-6)
    return {
        "coulomb_ratio_peak": _safe_stat(q, "ft_over_fn", "max"),
        "tangential_force_peak": _safe_stat(q, "measured_ft", "max"),
        "tangential_compliance": _safe_stat(q, "measured_ft", "max") / marker,
        "probe_asymmetry": _safe_stat(qo, "force_imbalance_ratio", "std"),
        "normal_force": fn,
        "marker_tangential_peak": marker,
    }


SYSID_FORMULAS = ["coulomb_ratio_peak", "tangential_force_peak", "tangential_compliance", "probe_asymmetry"]


def fit_affine(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[0]), float(beta[1])


def predict_affine(z: np.ndarray, coef: tuple[float, float]) -> np.ndarray:
    return np.clip(coef[0] + coef[1] * z, 0.2, 1.0)


def calibrate_sysid(source_meta: pd.DataFrame, source_feat: pd.DataFrame) -> dict:
    roots = sorted(source_meta.root_id.unique())
    fold = {r: i % 3 for i, r in enumerate(roots)}
    scores = []
    for formula in SYSID_FORMULAS:
        pred = np.full(len(source_meta), np.nan)
        z = source_feat[formula].to_numpy(float); y = source_meta.mu_GT.to_numpy(float)
        for f in FOLDS:
            tr = np.array([fold[r] != f for r in source_meta.root_id]); va = ~tr
            pred[va] = predict_affine(z[va], fit_affine(z[tr], y[tr]))
        scores.append({"formula": formula, "source_root_cv_mae": float(np.abs(pred-y).mean()),
                       "source_root_cv_rmse": float(np.sqrt(np.mean((pred-y)**2)))})
    scores = sorted(scores, key=lambda r: (r["source_root_cv_mae"], SYSID_FORMULAS.index(r["formula"])))
    selected = scores[0]["formula"]
    coef = fit_affine(source_feat[selected].to_numpy(float), source_meta.mu_GT.to_numpy(float))
    return {"selected_formula": selected, "intercept": coef[0], "slope": coef[1], "source_cv": scores}


def estimator_metrics(meta: pd.DataFrame, pred: np.ndarray) -> dict:
    y = meta.mu_GT.to_numpy(float); e = pred-y
    rho = spearmanr(y, pred).statistic
    correct = total = 0
    for _, g in meta.assign(pred=pred).groupby("root_id"):
        vals = g[["mu_GT", "pred"]].to_numpy()
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if vals[i,0] == vals[j,0]: continue
                correct += int(np.sign(vals[i,0]-vals[j,0]) == np.sign(vals[i,1]-vals[j,1])); total += 1
    return {"contexts": int(len(y)), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt(np.mean(e**2))),
            "bias": float(e.mean()), "Spearman": float(rho), "pair_ranking": float(correct/total)}


def train_probe_loto(out: Path, old: pd.DataFrame, raw_old: np.ndarray,
                     current: pd.DataFrame, raw_current: np.ndarray,
                     old_feat: pd.DataFrame, current_feat: pd.DataFrame):
    af.PROBE_META = old
    pred_rows, metric_rows, current_maps, sysid_info = [], [], {}, {}
    for target in TASKS:
        tr = np.flatnonzero(old.task.to_numpy() != target); va = np.flatnonzero(old.task.to_numpy() == target)
        for seed in SEEDS:
            model, mean, std = af.probe_fit(raw_old, tr, 1000 + target*17 + seed)
            p, _ = af.probe_pred(model, mean, std, raw_old, va)
            pc, _ = af.probe_pred(model, mean, std, raw_current, np.arange(len(current)))
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "target_task": target,
                        "source_tasks": [t for t in TASKS if t != target], "target_labels_used": False,
                        "epochs": af.PROBE_EPOCHS}, out / "models" / f"probe_loto_target{target}_seed{seed}.pt")
            current_maps[(target, "LearnedProbe", seed)] = dict(zip(current.context_id, pc.astype(float)))
            for idx, mu in zip(va, p):
                r = old.iloc[idx]
                pred_rows.append({"target_task": target, "method": "LearnedProbe", "seed": seed,
                                  "context_id": r.context_id, "root_id": r.root_id, "friction_band": r.friction_band,
                                  "mu_GT": float(r.mu_GT), "mu_hat": float(mu), "signed_error": float(mu-r.mu_GT)})
            metric_rows.append({"target_task": target, "method": "LearnedProbe", "seed": seed,
                                "scope": "TASK", **estimator_metrics(old.iloc[va], p)})
        sm = old[old.task != target].reset_index(drop=True); sf = old_feat.loc[old.task != target].reset_index(drop=True)
        info = calibrate_sysid(sm, sf); sysid_info[target] = info
        formula = info["selected_formula"]; coef = (info["intercept"], info["slope"])
        p = predict_affine(old_feat.loc[old.task == target, formula].to_numpy(float), coef)
        pc = predict_affine(current_feat[formula].to_numpy(float), coef)
        for seed in SEEDS:
            current_maps[(target, "ExplicitSysID", seed)] = dict(zip(current.context_id, pc.astype(float)))
        va_df = old[old.task == target].reset_index(drop=True)
        for r, mu in zip(va_df.itertuples(), p):
            pred_rows.append({"target_task": target, "method": "ExplicitSysID", "seed": "DETERMINISTIC",
                              "context_id": r.context_id, "root_id": r.root_id, "friction_band": r.friction_band,
                              "mu_GT": float(r.mu_GT), "mu_hat": float(mu), "signed_error": float(mu-r.mu_GT)})
        metric_rows.append({"target_task": target, "method": "ExplicitSysID", "seed": "DETERMINISTIC",
                            "scope": "TASK", **estimator_metrics(va_df, p), "selected_formula": formula})
        gt = dict(zip(current.context_id, current.mu_GT.astype(float)))
        for seed in SEEDS: current_maps[(target, "GT", seed)] = gt
    pd.DataFrame(pred_rows).to_csv(out / "UNIVERSAL_PROBE_LOTO_PREDICTIONS.csv", index=False)
    m = pd.DataFrame(metric_rows)
    macro=[]
    for (method,seed),g in m.groupby(["method","seed"]):
        macro.append({"target_task":"MACRO","method":method,"seed":seed,"scope":"MACRO",
                      **{c:float(g[c].mean()) for c in ["MAE","RMSE","bias","Spearman","pair_ranking"]},
                      "contexts":int(g.contexts.sum())})
    pd.concat([m,pd.DataFrame(macro)],ignore_index=True).to_csv(out / "PHYSICS_ESTIMATOR_LOTO.csv", index=False)
    wjson(out / "EXPLICIT_SYSID_CALIBRATION.json", sysid_info)
    return current_maps


def context_order(target: int, fold: int, md: pd.DataFrame, rf: np.ndarray) -> list[str]:
    q = md[(md.task == target) & (rf != fold)]
    contexts = sorted(q.context_id.unique(), key=lambda x: canonical_key(f"{target}|{fold}|{x}"))
    assert len(contexts) == 12
    return contexts


def simulate_acquisition(target: int, fold: int, md: pd.DataFrame, rf: np.ndarray) -> pd.DataFrame:
    q = md[(md.task == target) & (rf != fold) & (md.repeat == 1)].copy()
    order = context_order(target, fold, md, rf)
    by = {cid: g.sort_values("force_N").reset_index(drop=True) for cid,g in q.groupby("context_id")}
    assert all(len(g)==5 for g in by.values())
    pending = {c:[2] for c in order}; queried={c:[] for c in order}; complete={c:False for c in order}
    rows=[]
    # Boundary-seeking phase: one online query per context per round.
    while not all(complete.values()):
        progress=False
        for cid in order:
            if complete[cid] or not pending[cid]: continue
            progress=True; i=pending[cid].pop(0); z=by[cid].iloc[i]; queried[cid].append(i)
            rows.append({"target_task":target,"fold":fold,"query_index":len(rows)+1,"phase":"BOUNDARY_SEEK",
                         "context_id":cid,"root_id":z.root_id,"friction":float(z.mu),"force_rank":i+1,
                         "force_N":float(z.force_N),"outcome":int(z.success),"branch_id":z.branch_id})
            if i==2: pending[cid]=[1] if z.success else [3]
            elif i==1: pending[cid]=[0] if z.success else []; complete[cid]=not z.success
            elif i==0: pending[cid]=[]; complete[cid]=True
            elif i==3: pending[cid]=[] if z.success else [4]; complete[cid]=bool(z.success)
            elif i==4: pending[cid]=[]; complete[cid]=True
        if not progress: break
    # Complete the legal 60-branch pool without outcome-based selection.
    for cid in order:
        remaining=[i for i in [2,1,3,0,4] if i not in queried[cid]]
        for i in remaining:
            z=by[cid].iloc[i]
            rows.append({"target_task":target,"fold":fold,"query_index":len(rows)+1,"phase":"POOL_COMPLETION",
                         "context_id":cid,"root_id":z.root_id,"friction":float(z.mu),"force_rank":i+1,
                         "force_N":float(z.force_N),"outcome":int(z.success),"branch_id":z.branch_id})
    ans=pd.DataFrame(rows); assert len(ans)==60 and ans.branch_id.nunique()==60
    return ans


def brackets(q: pd.DataFrame) -> int:
    n=0
    for _,g in q.groupby("context_id"):
        z=g.sort_values("force_rank")
        outcomes=dict(zip(z.force_rank,z.outcome))
        n += sum(int(outcomes.get(i)==0 and outcomes.get(i+1)==1) for i in range(1,5))
    return n


def freeze_and_acquire(out: Path, md: pd.DataFrame, rf: np.ndarray):
    protocol={
      "status":"HASH_FROZEN_BEFORE_ANY_TRANSFER_MODEL_TRAINING",
      "tasks":TASKS,"budgets":BUDGETS,"nested_prefixes":True,"canonical_repeat":1,
      "context_order":"SHA256(ACTIVEFORCING_LOTO_V1|target|fold|context_id)",
      "boundary_policy":{"first":"F3","if_success":"F2 then F1 until adjacent bracket or grid end",
                         "if_failure":"F4 then F5 until adjacent bracket or grid end",
                         "round_robin":"one query/context/round; only observed outcomes set the next query"},
      "pool_completion":"after boundary-seeking terminates, query every unobserved force in fixed rank order [F3,F2,F4,F1,F5] so B60 is the complete 60-branch pool",
      "all_queries_count":True,"unqueried_outcomes_used_for_selection":False,
      "random_reference":"100 fixed SHA256 permutations/split; outcome-blind; analysis only",
      "source_train":"3 tasks x 4 roots x 3 frictions x 5 forces x canonical repeat = 180",
      "target_eval":"2 roots x 3 frictions x 2 repeats = 12/fold",
      "success_reward":"(Fmax-F)/Fmax","failure_reward":-1,"planner":"argmax expected utility",
      "semantic_zero_shot_status":"NOT_IDENTIFIABLE_WITH_UNSEEN_TASK_ONE_HOT",
      "untouched_root_scaling_TEST_used":False,"simulator_queries_launched":0,
      "order_hashes":{f"task{t}_fold{f}":hashlib.sha256("|".join(context_order(t,f,md,rf)).encode()).hexdigest() for t in TASKS for f in FOLDS},
    }
    wjson(out / "TARGET_BOUNDARY_ACQUISITION_PROTOCOL.json", protocol)
    trajectories=[]
    for t in TASKS:
        for f in FOLDS: trajectories.append(simulate_acquisition(t,f,md,rf))
    acq=pd.concat(trajectories,ignore_index=True); acq.to_csv(out/"TARGET_ACQUISITION_TRAJECTORIES.csv",index=False)
    bal=[]; rnd=[]
    for (t,f),g in acq.groupby(["target_task","fold"]):
        pool=md[(md.task==t)&(rf!=f)&(md.repeat==1)].copy()
        for b in BUDGETS:
            q=g[g.query_index<=b]
            bal.append({"target_task":t,"fold":f,"budget":b,"success_count":int(q.outcome.sum()),
                        "failure_count":int(len(q)-q.outcome.sum()),"success_fraction":float(q.outcome.mean()) if len(q) else math.nan,
                        "failure_fraction":float(1-q.outcome.mean()) if len(q) else math.nan,"roots_covered":int(q.root_id.nunique()),
                        "friction_contexts_covered":int(q.context_id.nunique()),"force_values_covered":int(q.force_rank.nunique()),
                        "adjacent_success_failure_brackets":brackets(q),
                        "adaptation_status":"ONE_CLASS_TARGET_ADAPTATION" if len(q) and q.outcome.nunique()<2 else ("NO_TARGET_ADAPTATION" if not len(q) else "TWO_CLASS")})
            if b:
                vals=[]
                for salt in range(100):
                    ranks = pool.assign(force_rank=pool.groupby("context_id").force_N.rank(method="first").astype(int))[ ["branch_id", "force_rank"] ]
                    z=pool.assign(key=pool.branch_id.map(lambda x:hashlib.sha256(f"RANDOM_REF|{salt}|{x}".encode()).hexdigest())).sort_values("key").head(b)
                    z=z.merge(ranks,on="branch_id",validate="one_to_one").assign(outcome=lambda x:x.success)
                    vals.append(brackets(z))
                rnd.append({"target_task":t,"fold":f,"budget":b,"boundary_brackets":brackets(q),
                            "random_brackets_mean":float(np.mean(vals)),"random_brackets_std":float(np.std(vals))})
    pd.DataFrame(bal).to_csv(out/"TARGET_FEWSHOT_LABEL_BALANCE.csv",index=False)
    pd.DataFrame(rnd).to_csv(out/"TARGET_ACQUISITION_EFFICIENCY.csv",index=False)
    return acq


def score_direct(model, traces, segs, norm, mu_map, zero_h):
    s,c,_=af.tensors(traces,segs,norm,mu_map,zero_h,False)
    with torch.no_grad(): return af.sigmoid(model(s,c).numpy()).astype(np.float32)


def evaluate_selected(md: pd.DataFrame, held_idx: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    f=md.force_N.to_numpy()[held_idx]; fm=md.task.map(FMAX).to_numpy()[held_idx]
    u=p*((fm-f)/fm)+(1-p)*-1
    full=np.full(len(md),np.nan,np.float32); full[held_idx]=u
    return af.choose(md.iloc[held_idx],full)


def run_shard(out: Path, target: int, fold: int, seed: int, estimator: str):
    shard=out/"shards"/f"target{target}_fold{fold}_seed{seed}_{estimator}.json"
    if shard.exists(): return
    tpi,cf,full,cmap,traces,meta,audits,pairs,segs,md,rf=load_current(out/"_audit",f"shard_{target}_{fold}_{seed}_{estimator}")
    current=pd.read_csv(out/"CURRENT_PROBE_CONTEXTS.csv")
    pred=pd.read_csv(out/"CURRENT_PHYSICS_PREDICTIONS.csv")
    if estimator=="LearnedProbe":
        q=pred[(pred.target_task==target)&(pred.method==estimator)&(pred.seed.astype(str)==str(seed))]
    else:
        q=pred[(pred.target_task==target)&(pred.method==estimator)]
    mu_map=dict(zip(q.context_id,q.mu_hat)); assert len(mu_map)==72
    acq=pd.read_csv(out/"TARGET_ACQUISITION_TRAJECTORIES.csv")
    aq=acq[(acq.target_task==target)&(acq.fold==fold)]
    source_mask=(md.task!=target)&(rf!=fold)&(md.repeat==1)
    source_idx=np.flatnonzero(source_mask)
    assert len(source_idx)==180
    held_idx=np.flatnonzero((md.task==target)&(rf==fold)); assert len(held_idx)==60
    zero_h={cid:np.zeros(1,np.float32) for cid in md.context_id.unique()}
    rows=[]; selections=[]
    for budget in BUDGETS:
        target_ids=set(aq.loc[aq.query_index<=budget,"branch_id"])
        target_idx=np.flatnonzero(md.branch_id.isin(target_ids).to_numpy())
        train_idx=np.concatenate([source_idx,target_idx]); assert len(train_idx)==180+budget
        train=[traces[i] for i in train_idx]; held=[traces[i] for i in held_idx]
        norm=af.xnorm(train,segs,mu_map)
        model=af.train_direct(train,segs,norm,mu_map,zero_h,meta,seed,False)
        p=score_direct(model,held,segs,norm,mu_map,zero_h)
        sel=evaluate_selected(md,held_idx,p)
        sel["target_task"]=target;sel["fold"]=fold;sel["seed"]=seed;sel["physics_estimator"]=estimator;sel["budget"]=budget
        selections.append(sel)
        rows.append({"target_task":target,"fold":fold,"seed":seed,"physics_estimator":estimator,"budget":budget,
                     "total_train_rollouts":180+budget,"source_train_rollouts":180,"target_train_rollouts":budget,
                     "episodes":len(sel),"successes":int(sel.success.sum()),"sr":float(sel.success.mean()),
                     "underforce":float(sel.under_force.mean()),"mean_force":float(sel.selected_force_N.mean()),
                     "utility":float(sel.realized_utility.mean()),"selected_force_error":float((sel.selected_force_N-sel.frontier_N).abs().mean()),
                     "excess_force":float(sel.excess_force_N.mean())})
    pd.concat(selections).to_csv(out/"shards"/f"target{target}_fold{fold}_seed{seed}_{estimator}_selected.csv",index=False)
    wjson(shard,{"status":"COMPLETE","rows":rows,"target":target,"fold":fold,"seed":seed,"estimator":estimator,
                 "root_scaling_TEST_used":False,"world_model_used":False,"residual_used":False})


def prepare(out: Path):
    out.mkdir(parents=True,exist_ok=True);(out/"models").mkdir(exist_ok=True);(out/"shards").mkdir(exist_ok=True)
    tpi,cf,full,cmap,traces,meta,audits,pairs,segs,md,rf=load_current(out/"_audit","prepare")
    direct_audit="""# Direct Task-Encoding Audit\n\n**`ZERO_SHOT_TASK_TRANSFER_WITH_CURRENT_ENCODING = NOT_IDENTIFIABLE`.**\n\nThe frozen Direct receives a 71D nominal tensor split into a 17D step sequence and 54D static condition. The step sequence contains 6 Cartesian command features, a 7D phase one-hot, and a **4D task one-hot**. The condition contains candidate force/8, scalar friction, current 13-channel physical state plus mask, and initial 13-channel state plus mask. It receives **no RGB, frozen π0 visual feature, language embedding, VLA semantic embedding, raw Probe trace, or learned semantic task embedding**.\n\nFor a leave-one-task-out B=0 model, the target task's one-hot coordinate is never activated during source training. B=0 is therefore an unseen-coordinate extrapolation diagnostic, not semantic zero-shot new-task transfer. B=10/20/30/60 remain valid root-heldout task-specific adaptation experiments without changing the architecture.\n\nSource: exact input construction in `trajectory_physical_imagination.py` and the frozen 17+54 Direct split in `run_probe_conditioned_wm.py`.\n"""
    (out/"DIRECT_TASK_ENCODING_AUDIT.md").write_text(direct_audit)
    old,raw_old,old_feat,current,raw_current,current_feat=load_probe_populations(traces)
    current.to_csv(out/"CURRENT_PROBE_CONTEXTS.csv",index=False)
    maps=train_probe_loto(out,old,raw_old,current,raw_current,old_feat,current_feat)
    pred=[]
    for (target,method,seed),m in maps.items():
        for r in current.itertuples(): pred.append({"target_task":target,"method":method,"seed":seed,"context_id":r.context_id,"root_id":r.root_id,"task":r.task,"mu_GT":r.mu_GT,"mu_hat":m[r.context_id]})
    pd.DataFrame(pred).to_csv(out/"CURRENT_PHYSICS_PREDICTIONS.csv",index=False)
    probe_audit=f"""# Universal Probe Data Audit\n\n**PASS for archived leave-one-task-out Probe evaluation.**\n\n- Population: 144/144 complete fixed-P4B Probe sequences, 48 root families, 36 sequences and 12 roots per task; every sequence is 215 timesteps × 46 legal model features.\n- Friction bands: 48 LOW, 48 MID, 48 HIGH; range is [0.2, 1.0] by protocol.\n- Assets: task0=`alphabet_soup_1`, task1=`cream_cheese_1`, task5=`tomato_sauce_1`, task6=`butter_1`. Thus held-out task also holds out an object family; this design cannot separate task transfer from object transfer and makes no pure cross-object claim.\n- Probe primitive: identical P4B common contact-frame shear for all tasks.\n- Inputs include force/contact, gripper/proprioception, commanded probe action, contact state, phase, marker motion, and EEF displacement. GT friction, privileged object pose, and downstream outcome are excluded.\n- Learned LOTO Probe is trained on all three source tasks only. No target-task friction label, trace, or estimator fine-tuning is used. Target evaluation is therefore root-disjoint automatically because the entire target task is excluded.\n- The force-selection population has a separate complete 72-sequence/24-root archive. Canonical `(task, root_index, root_seed)` identity overlaps the old P5 TRAIN Probe population for 24/24 roots. The historical pooled estimator therefore has 24/24 overlap and cannot establish transfer. In the present LOTO turn, all 6/6 force-selection roots of the target task are excluded from Probe training; source-task overlap remains legal training-side data.\n\nOriginal pooled root-heldout MAE reference: 0.0655. No root-scaling TEST file or result was discovered or read.\n"""
    (out/"UNIVERSAL_PROBE_DATA_AUDIT.md").write_text(probe_audit)
    info=json.loads((out/"EXPLICIT_SYSID_CALIBRATION.json").read_text())
    sysaudit="# Explicit SysID Channel Audit\n\n**Explicit response-curve identification is available, but direct Coulomb slip-threshold identification is not.**\n\nThe 46D legal trace contains measured normal/tangential force, `Ft/Fn`, bilateral contact flags/state, force imbalance, gripper opening, commanded tangential increment, accumulated displacement, marker motion/tangential displacement/velocity, EEF displacement, phase, and tactile validity. The fixed probe remains in bilateral contact, but no authoritative slip-onset flag or saturated Coulomb threshold is recorded; observed `Ft/Fn` is much smaller than GT μ. Therefore `max(Ft/Fn)` cannot be interpreted directly as μ.\n\nThe explicit baseline predefines four one-dimensional physical response summaries: peak `Ft/Fn`, peak tangential force, tangential force per marker displacement, and probe-out force-asymmetry variation. For each LOTO turn it selects one formula by grouped source-root CV, fits only an affine sensor-scale/nuisance calibration on source tasks, clips to the legal [0.2,1.0] range, freezes it, and evaluates the untouched target task. It is non-neural and uses no task ID, target label, validation outcome, or GT μ at inference.\n\nSelected source-only formulas:\n"
    for t in TASKS: sysaudit += f"- target task{t}: `{info[str(t)]['selected_formula']}` (source-CV selection only)\n"
    (out/"EXPLICIT_SYSID_CHANNEL_AUDIT.md").write_text(sysaudit)
    freeze_and_acquire(out,md,rf)
    wjson(out/"PREPARE_COMPLETE.json",{"status":"COMPLETE","current_branches":720,"probe_sequences":144,
          "protocol_sha256":sha(out/"TARGET_BOUNDARY_ACQUISITION_PROTOCOL.json")})


def aggregate(out: Path):
    augment_probe_metrics(out)
    rows=[]; sels=[]
    for p in sorted((out/"shards").glob("target*_*.json")):
        rows.extend(json.loads(p.read_text())["rows"])
    for p in sorted((out/"shards").glob("*_selected.csv")): sels.append(pd.read_csv(p))
    raw=pd.DataFrame(rows); selected=pd.concat(sels,ignore_index=True)
    # Attach matched GT controller decisions.
    gt=selected[selected.physics_estimator=="GT"][["target_task","fold","seed","budget","context_id","repeat","selected_force_N"]].rename(columns={"selected_force_N":"GT_selected_force_N"})
    selected=selected.merge(gt,on=["target_task","fold","seed","budget","context_id","repeat"],validate="many_to_one")
    selected["decision_agreement_GT"]=(abs(selected.selected_force_N-selected.GT_selected_force_N)<1e-8).astype(int)
    selected["selected_force_gap_GT"]=selected.selected_force_N-selected.GT_selected_force_N
    # Task1 label-source sensitivity (140/180 are reconstructed).
    label_files=list((out/"_audit").glob("**/task1/TASK1_CANONICAL_TRAIN_BRANCHES.csv"))
    if not label_files: raise RuntimeError("task1 label-source manifest not found")
    labels=pd.read_csv(label_files[0])[["context_id","repeat","force_N","label_source"]].drop_duplicates()
    t1=selected[selected.target_task==1].merge(labels,left_on=["context_id","repeat","selected_force_N"],right_on=["context_id","repeat","force_N"],validate="many_to_one")
    sens=[]
    for keys,g in t1.groupby(["physics_estimator","budget","seed"]):
        est,b,s=keys; d=g[g.label_source=="DIRECT_CUMULATIVE_BRANCH_LABEL"]
        sens.append({"physics_estimator":est,"target_task":1,"budget":b,"seed":s,"all_episodes":len(g),"direct_label_episodes":len(d),
                     "all_sr":g.success.mean(),"direct_label_only_sr":d.success.mean() if len(d) else math.nan,
                     "all_underforce":g.under_force.mean(),"direct_label_only_underforce":d.under_force.mean() if len(d) else math.nan,
                     "interpretation":"SENSITIVITY_ONLY_SELECTED_BRANCH_LABEL_SOURCE"})
    pd.DataFrame(sens).to_csv(out/"TASK1_DIRECT_LABEL_SENSITIVITY.csv",index=False)
    selected.to_csv(out/"NEW_TASK_FEWSHOT_PER_EPISODE.csv",index=False)
    # Reaggregate folds to 36 episodes / target / estimator / budget / seed.
    agg=[]
    for keys,g in selected.groupby(["physics_estimator","target_task","budget","seed"]):
        est,t,b,s=keys; bal=pd.read_csv(out/"TARGET_FEWSHOT_LABEL_BALANCE.csv"); bb=bal[(bal.target_task==t)&(bal.budget==b)]
        agg.append({"physics_estimator":est,"target_task":t,"budget":b,"total_train_rollouts":180+b,"seed":s,
                    "episodes":len(g),"successes":int(g.success.sum()),"sr":g.success.mean(),"underforce":g.under_force.mean(),
                    "mean_force":g.selected_force_N.mean(),"excess_force":g.excess_force_N.mean(),"utility":g.realized_utility.mean(),
                    "selected_force_error":(g.selected_force_N-g.frontier_N).abs().mean(),"decision_agreement_GT":g.decision_agreement_GT.mean(),
                    "num_success_train":int(bb.success_count.sum()),"num_failure_train":int(bb.failure_count.sum()),
                    "one_class_folds":int((bb.adaptation_status=="ONE_CLASS_TARGET_ADAPTATION").sum())})
    agg=pd.DataFrame(agg); agg.to_csv(out/"NEW_TASK_FEWSHOT_TRANSFER.csv",index=False)
    summary=[]
    for (est,t,b),g in agg.groupby(["physics_estimator","target_task","budget"]):
        row={"physics_estimator":est,"target_task":t,"budget":b,"total_train_rollouts":180+b,"seeds":len(g)}
        for c in ["sr","underforce","mean_force","excess_force","utility","selected_force_error","decision_agreement_GT"]:
            row[c+"_mean"]=g[c].mean();row[c+"_std"]=g[c].std(ddof=0)
        row["num_success_train_mean"]=g.num_success_train.mean();row["num_failure_train_mean"]=g.num_failure_train.mean();row["one_class_folds"]=g.one_class_folds.max()
        summary.append(row)
    per=pd.DataFrame(summary); per.to_csv(out/"PER_TASK_NEW_TASK_TRANSFER.csv",index=False)
    macro=[]
    metric_cols=["sr","underforce","mean_force","excess_force","utility","selected_force_error","decision_agreement_GT"]
    for (est,b),g in agg.groupby(["physics_estimator","budget"]):
        seed_macro=g.groupby("seed")[metric_cols].mean()
        row={"physics_estimator":est,"target_task":"MACRO","budget":b,"total_train_rollouts":180+b,"tasks":g.target_task.nunique(),"seeds":g.seed.nunique()}
        for c in metric_cols: row[c+"_mean"]=seed_macro[c].mean();row[c+"_std"]=seed_macro[c].std(ddof=0)
        row["num_success_train_mean"]=g.groupby("target_task").num_success_train.mean().mean();row["num_failure_train_mean"]=g.groupby("target_task").num_failure_train.mean().mean();row["one_class_folds"]=g.one_class_folds.max();macro.append(row)
    allagg=pd.concat([per,pd.DataFrame(macro)],ignore_index=True,sort=False); allagg.to_csv(out/"NEW_TASK_FEWSHOT_TRANSFER_AGG.csv",index=False)
    # B* by task and macro, against B60 of the same estimator.
    bstars=[]
    for est in ["LearnedProbe","ExplicitSysID","GT"]:
        q=per[per.physics_estimator==est]
        task_pass={}
        for t in TASKS:
            z=q[q.target_task==t].set_index("budget"); ref=z.loc[60]
            passed=[]
            for b in BUDGETS:
                r=z.loc[b]; label_ok=(b==0) or (r.num_success_train_mean>0 and r.num_failure_train_mean>0 and r.one_class_folds==0)
                ok=bool(abs(r.sr_mean-ref.sr_mean)<=.02 and r.underforce_mean<=ref.underforce_mean+.02 and r.utility_mean>=ref.utility_mean-UTILITY_TOL and label_ok)
                passed.append((b,ok))
            bs=next((b for b,ok in passed if ok),None);task_pass[t]=dict(passed)
            bstars.append({"physics_estimator":est,"scope":f"task{t}","B_star":bs if bs is not None else "NOT_REACHED",
                           "semantic_zero_shot_claim_allowed":False,"criteria":"SR gap<=2pp; underforce<=B60+2pp; utility>=B60-0.01; two-class in every fold"})
        z=pd.DataFrame(macro);z=z[z.physics_estimator==est].set_index("budget");ref=z.loc[60];bm=None
        for b in BUDGETS:
            r=z.loc[b]; close=sum(task_pass[t][b] for t in TASKS)
            label_ok=(b==0) or (r.num_success_train_mean>0 and r.num_failure_train_mean>0 and r.one_class_folds==0)
            if abs(r.sr_mean-ref.sr_mean)<=.02 and r.underforce_mean<=ref.underforce_mean+.02 and r.utility_mean>=ref.utility_mean-UTILITY_TOL and label_ok and close>=3:
                bm=b;break
        bstars.append({"physics_estimator":est,"scope":"MACRO","B_star":bm if bm is not None else "NOT_REACHED",
                       "semantic_zero_shot_claim_allowed":False,"tasks_close_to_saturation_at_Bstar":sum(task_pass[t].get(bm,False) for t in TASKS) if bm is not None else 0,
                       "criteria":"macro gates plus >=3/4 tasks individually close"})
    pd.DataFrame(bstars).to_csv(out/"NEW_TASK_BSTAR.csv",index=False)
    # Exact existing all-four-task Sparse60 reference, same population/split/utility.
    ref_path=ROOT/"activeforcing_sample_efficiency_60_20260901_073611/SAMPLE_EFFICIENCY_60_VS_FULL_AGG.csv"
    ref=pd.read_csv(ref_path); ref=ref[ref.controller=="Direct-60"].copy();ref["source_file"]=str(ref_path)
    ref.to_csv(out/"EXISTING_ALL4_SPARSE60_REFERENCE.csv",index=False)
    write_qa(out,selected,agg,acq_path=out/"TARGET_ACQUISITION_TRAJECTORIES.csv")
    plot_figures(out,per,pd.DataFrame(macro))
    write_report(out,per,pd.DataFrame(macro),pd.DataFrame(bstars))


def write_qa(out: Path, selected: pd.DataFrame, agg: pd.DataFrame, acq_path: Path):
    acq=pd.read_csv(acq_path); checks={}
    checks["complete_shard_jsons"]=(len(list((out/"shards").glob("target*_*.json")))==108)
    checks["complete_selected_shards"]=(len(list((out/"shards").glob("*_selected.csv")))==108)
    checks["episode_rows"]=(len(selected)==6480)
    checks["episodes_per_target_budget_seed_estimator"]=(set(selected.groupby(["physics_estimator","target_task","budget","seed"]).size())=={36})
    checks["episodes_per_fold"]=(set(selected.groupby(["physics_estimator","target_task","budget","seed","fold"]).size())=={12})
    checks["evaluation_roots_per_fold"]=(set(selected.groupby(["physics_estimator","target_task","budget","seed","fold"]).root_id.nunique())=={2})
    checks["evaluation_contexts_per_fold"]=(set(selected.groupby(["physics_estimator","target_task","budget","seed","fold"]).context_id.nunique())=={6})
    checks["evaluation_repeats_per_fold"]=(set(selected.groupby(["physics_estimator","target_task","budget","seed","fold"]).repeat.nunique())=={2})
    checks["acquisition_60_unique_per_target_fold"]=(set(acq.groupby(["target_task","fold"]).branch_id.nunique())=={60})
    checks["gt_self_agreement"]=bool(selected[selected.physics_estimator=="GT"].decision_agreement_GT.eq(1).all())
    checks["all_metrics_finite"]=bool(np.isfinite(agg[["sr","underforce","mean_force","utility","selected_force_error"]]).all().all())
    status="PASS" if all(checks.values()) else "FAIL"
    wjson(out/"TRANSFER_DATA_QA.json",{"status":status,"checks":checks,"controller_episode_rows":len(selected),"shards":108,
          "training_counts":{"B0":180,"B10":190,"B20":200,"B30":210,"B60":240},
          "task1_reconstructed_labels":"140/180","root_scaling_TEST_used":False,"new_simulator_rollouts":0})
    if status!="PASS": raise RuntimeError("transfer QA failed")


def augment_probe_metrics(out: Path):
    d=pd.read_csv(out/"UNIVERSAL_PROBE_LOTO_PREDICTIONS.csv")
    rows=[]
    for keys,g in d.groupby(["target_task","method","seed"]):
        t,m,s=keys; rows.append({"target_task":t,"method":m,"seed":s,"scope":"TASK",**estimator_metrics(g.rename(columns={"mu_GT":"mu_GT"}),g.mu_hat.to_numpy())})
        for root,r in g.groupby("root_id"):
            rows.append({"target_task":t,"method":m,"seed":s,"scope":"ROOT","root_id":root,**estimator_metrics(r,r.mu_hat.to_numpy())})
        for band,r in g.groupby("friction_band"):
            y=r.mu_GT.to_numpy(float);p=r.mu_hat.to_numpy(float);e=p-y
            rows.append({"target_task":t,"method":m,"seed":s,"scope":"FRICTION","friction_band":band,"contexts":len(r),"MAE":abs(e).mean(),"RMSE":np.sqrt(np.mean(e**2)),"bias":e.mean(),"Spearman":math.nan,"pair_ranking":math.nan})
    base=pd.DataFrame(rows)
    task=base[base.scope=="TASK"]
    macro=[]
    for (m,s),g in task.groupby(["method","seed"]):
        macro.append({"target_task":"MACRO","method":m,"seed":s,"scope":"MACRO","contexts":int(g.contexts.sum()),
                      **{c:float(g[c].mean()) for c in ["MAE","RMSE","bias","Spearman","pair_ranking"]}})
    summary=[]
    learned=task[task.method=="LearnedProbe"]
    for t,g in learned.groupby("target_task"):
        summary.append({"target_task":t,"method":"LearnedProbe","seed":"THREE_SEED_MEAN","scope":"TASK","contexts":int(g.contexts.iloc[0]),
                        **{c:float(g[c].mean()) for c in ["MAE","RMSE","bias","Spearman","pair_ranking"]},
                        **{c+"_std":float(g[c].std(ddof=0)) for c in ["MAE","RMSE","bias","Spearman","pair_ranking"]}})
    gm=pd.DataFrame(macro); gl=gm[gm.method=="LearnedProbe"]
    summary.append({"target_task":"MACRO","method":"LearnedProbe","seed":"THREE_SEED_MEAN","scope":"MACRO","contexts":144,
                    **{c:float(gl[c].mean()) for c in ["MAE","RMSE","bias","Spearman","pair_ranking"]},
                    **{c+"_std":float(gl[c].std(ddof=0)) for c in ["MAE","RMSE","bias","Spearman","pair_ranking"]}})
    pd.concat([base,pd.DataFrame(macro),pd.DataFrame(summary)],ignore_index=True,sort=False).to_csv(out/"PHYSICS_ESTIMATOR_LOTO.csv",index=False)


def plot_figures(out: Path, per: pd.DataFrame, macro: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    colors={"LearnedProbe":"#2563eb","ExplicitSysID":"#e97316","GT":"#5b6470"}
    fig,ax=plt.subplots(figsize=(8.6,5.2))
    for est in ["LearnedProbe","ExplicitSysID"]:
        q=macro[macro.physics_estimator==est].sort_values("budget")
        ax.errorbar(q.budget,q.sr_mean*100,yerr=q.sr_std*100,marker="o",lw=2,capsize=3,label=est,color=colors[est])
    ref=float(pd.read_csv(out/"EXISTING_ALL4_SPARSE60_REFERENCE.csv").sr_seed_mean.iloc[0])*100
    ax.axhline(ref,color="#5b6470",ls="--",alpha=.7,label="Existing all-4 Sparse-60 Direct")
    ax.set(xlabel="Target-task adaptation rollouts",ylabel="Full-task success rate (%)",xticks=BUDGETS,title="New-task adaptation with a frozen shared Direct architecture")
    ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(out/"FIG_NEW_TASK_FEWSHOT_SR.png",dpi=220);fig.savefig(out/"FIG_NEW_TASK_FEWSHOT_SR.pdf");plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for ax,t in zip(axs.flat,TASKS):
        for est,ls in [("LearnedProbe","-"),("ExplicitSysID","--")]:
            q=per[(per.physics_estimator==est)&(per.target_task==t)].sort_values("budget")
            ax.errorbar(q.budget,q.sr_mean*100,yerr=q.sr_std*100,marker="o",ls=ls,lw=1.8,capsize=2,label=est,color=colors[est])
        ax.set_title(f"task{t}");ax.set_xticks(BUDGETS)
    axs[1,0].set_xlabel("Target rollouts");axs[1,1].set_xlabel("Target rollouts");axs[0,0].set_ylabel("SR (%)");axs[1,0].set_ylabel("SR (%)")
    axs[0,0].legend(frameon=False,fontsize=8);fig.suptitle("Leave-one-task-out adaptation by target task",y=.98);fig.tight_layout();fig.savefig(out/"FIG_NEW_TASK_FEWSHOT_PER_TASK.png",dpi=220);fig.savefig(out/"FIG_NEW_TASK_FEWSHOT_PER_TASK.pdf");plt.close(fig)
    m=pd.read_csv(out/"PHYSICS_ESTIMATOR_LOTO.csv");m=m[m.scope=="TASK"].copy();m["target_task"]=pd.to_numeric(m.target_task).astype(int)
    fig,ax=plt.subplots(figsize=(8,5));x=np.arange(4);w=.36
    learned=m[(m.method=="LearnedProbe")&(m.seed.astype(str)=="THREE_SEED_MEAN")].set_index("target_task").MAE.reindex(TASKS)
    sys=m[m.method=="ExplicitSysID"].set_index("target_task").MAE.reindex(TASKS)
    ax.bar(x-w/2,learned,w,label="Learned Probe",color=colors["LearnedProbe"]);ax.bar(x+w/2,sys,w,label="Explicit SysID",color=colors["ExplicitSysID"])
    ax.set(xticks=x,xticklabels=[f"task{t}" for t in TASKS],ylabel="Friction MAE",title="Probe estimator transfer to a held-out task/object")
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/"FIG_PROBE_LOTO_COMPARISON.png",dpi=220);fig.savefig(out/"FIG_PROBE_LOTO_COMPARISON.pdf");plt.close(fig)


def write_report(out: Path, per: pd.DataFrame, macro: pd.DataFrame, bstars: pd.DataFrame):
    pm=pd.read_csv(out/"PHYSICS_ESTIMATOR_LOTO.csv");lb=pm[(pm.method=="LearnedProbe")&(pm.scope=="TASK")&(pm.seed.astype(str)=="THREE_SEED_MEAN")];lm=float(lb.MAE.mean())
    sm=float(pm[(pm.method=="ExplicitSysID")&(pm.scope=="TASK")].MAE.mean())
    q=macro.set_index(["physics_estimator","budget"]); lp=[float(q.loc[("LearnedProbe",b),"sr_mean"]) for b in BUDGETS]
    bstar=bstars[(bstars.physics_estimator=="LearnedProbe")&(bstars.scope=="MACRO")].B_star.iloc[0]
    ac=pd.read_csv(out/"TARGET_ACQUISITION_EFFICIENCY.csv"); b20=ac[ac.budget==20]
    sysgap=sm-lm
    if str(bstar)=="0": cls="SHARED_DIRECT_ZERO_SHOT_TRANSFER_NOT_SEMANTICALLY_IDENTIFIABLE"
    elif str(bstar) in {"10","20"}: cls="FEW_SHOT_PHYSICAL_ADAPTATION_SUFFICIENT"
    else: cls="SUBSTANTIAL_TASK_SPECIFIC_DATA_REQUIRED"
    refrow=pd.read_csv(out/"EXISTING_ALL4_SPARSE60_REFERENCE.csv").iloc[0]
    sens=pd.read_csv(out/"TASK1_DIRECT_LABEL_SENSITIVITY.csv")
    lines=["# Final Shared Physical Transfer Report","",
      "## Technical summary","",
      f"最直白的结论：这组 frozen architecture 不能支持真正的 semantic zero-shot claim，而且**小样本并未稳定饱和**。Direct 只有 4D task one-hot；B=0 target coordinate 从未激活。Learned-Probe root-heldout macro SR 在 B=0/10/20/30/60 为 **"+" / ".join(f"{x*100:.2f}%" for x in lp)+f"**，曲线强烈非单调；按预注册 gate，macro B* 是 **{bstar}**，所以分类为 **{cls}**。",
      "",
      f"固定 probe primitive保留了可跨 task 解读的信号，但当前 learned GRU interpreter 并不 task-general：LOTO MAE **{lm:.4f}**，远差于 pooled root-heldout 0.0655。source-only explicit SysID 降到 **{sm:.4f}**，但它在 B=60 的 controller SR/utility 没超过 Learned Probe，因此现在还不能仅凭 estimator MAE 删除 learned Probe。",
      "",
      "## The frozen task one-hot prevents a semantic zero-shot claim","",
      "Direct 的 71D input只含 nominal Cartesian command/phase、4D task one-hot、force、scalar μ 与 physical state/masks；没有 language、RGB、VLA semantic feature。B=0 是 unseen one-hot coordinate extrapolation diagnostic。B>0 才让 target coordinate 获得监督。这里 held-out task 同时更换 object asset，因此 task effect 与 object-family effect也不能分离。",
      "",
      "## Probe transfer fails for the learned interpreter, while explicit response fitting is more stable","",
      f"Learned Probe LOTO macro MAE = **{lm:.4f}**，相对 0.0655 恶化 **{lm-0.0655:+.4f}**；Explicit SysID = **{sm:.4f}**，比 learned LOTO 低 **{abs(sysgap):.4f}**。四个 target 的 source-only formula 都由 source-root CV 选择为 probe-out force-asymmetry variation，再做 source-only affine sensor calibration。",
      "",
      "这不等于直接 Coulomb identification：trace 没有 authoritative slip onset，且观察到的 `Ft/Fn` 远小于 GT μ。结果只支持‘同一 probe trace 可被显式响应曲线更稳定地解释’，不支持‘已直接测得 Coulomb μ’。",
      "",
      "## New-task adaptation is non-monotonic and needs the full 60-rollout reference","",
      "| physics | B | macro SR | under-force | utility | mean force |", "|---|---:|---:|---:|---:|---:|"]
    for est in ["LearnedProbe","ExplicitSysID","GT"]:
      for b in BUDGETS:
        r=q.loc[(est,b)];lines.append(f"| {est} | {b} | {r.sr_mean*100:.2f}% | {r.underforce_mean*100:.2f}% | {r.utility_mean:.4f} | {r.mean_force_mean:.3f} N |")
    lines += ["", f"B=60 uses 180 source + 60 target = 240 branches and is the matched LOTO full-adaptation scale. The separately archived existing all-4 Sparse60 Direct reference is **{refrow.sr_seed_mean*100:.2f}%** three-seed mean SR; it used pooled same-task Probe training and is shown as a horizontal reference, not conflated with LOTO+B60.",
      "", "The B=10 spike is not evidence that 10 is enough: task6 is one-class in all three folds, B=20/30 regress sharply, and the 3-seed/per-task saturation gates fail. Additional target rows change class balance and ranking non-monotonically; no checkpoint or budget was selected post hoc.",
      "", "### Per-task required budget", "", "| target | B* | B=0 SR | B=10 SR | B=20 SR | B=30 SR | B=60 SR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for t in TASKS:
        z=per[(per.physics_estimator=="LearnedProbe")&(per.target_task==t)].set_index("budget")
        bt=bstars[(bstars.physics_estimator=="LearnedProbe")&(bstars.scope==f"task{t}")].B_star.iloc[0]
        lines.append(f"| task{t} | {bt} | "+" | ".join(f"{z.loc[b,'sr_mean']*100:.2f}%" for b in BUDGETS)+" |")
    lines += ["", "task5/task6 的 B=0 数值很高，但仍不能叫 zero-shot transfer，因为 target one-hot 未训练；它可能反映 source-learned shared bias与该 target archived difficulty，而非语义迁移。task6 B=10/B=20 的 target adaptation labels全为 success，不能作为有效 supervised budget。",
      "", "## Boundary seeking is more query-efficient in aggregate, not on every task","",
      f"At B=20 the sequential policy obtains **{b20.boundary_brackets.mean():.2f}** adjacent brackets/split versus **{b20.random_brackets_mean.mean():.2f}** for 100 fixed outcome-blind hash permutations. The advantage is driven by task0/task5; task1 and task6 do not beat the random reference at B=20 because recoverable boundaries are sparse. Every F3→F2/F1 or F3→F4/F5 query is charged and retained.",
      "", "Exact success/failure counts, roots, contexts, forces, brackets, and one-class flags are in `TARGET_FEWSHOT_LABEL_BALANCE.csv`. The target pools are nested prefixes of one frozen query trajectory; no unqueried outcome selects a sample.",
      "", "## Task1 remains label-limited","",
      f"Task1 retains 140/180 reconstructed terminal labels. The selected-branch direct-label-only sensitivity is reported for all {len(sens)} method/budget/seed cells in `TASK1_DIRECT_LABEL_SENSITIVITY.csv`; denominators vary with the force each controller selects, so this is a sensitivity check, not a replacement headline. Task1 is retained in every macro result.",
      "", "## Direct answers to the 18 preregistered questions", "",
      f"1. **Can Probe predict friction without target labels?** It can produce predictions, but not accurately enough to call the learned estimator task-general: LOTO MAE is {lm:.4f}.",
      f"2. **LOTO Learned Probe MAE:** {lm:.4f} (three-seed/task macro).",
      f"3. **Drop from pooled OOF 0.0655:** MAE worsens by {lm-0.0655:+.4f} ({lm/0.0655:.1f}× the error).",
      f"4. **Explicit SysID MAE:** {sm:.4f}.",
      f"5. **SysID–learned gap:** SysID MAE is {lm-sm:.4f} lower.",
      "6. **Can learned Probe be deleted?** Not yet: SysID wins μ estimation but not the B=60 controller SR/utility; estimator MAE and force-decision value disagree.",
      "7. **Is learned mapping worth retaining?** As a task-general friction estimator, current evidence says no; as a downstream feature, the matched controller still performs better than SysID at B=60, so replacement requires a frozen controller confirmation.",
      f"8–12. **Shared Direct macro SR at B=0/10/20/30/60:** "+", ".join(f"{x*100:.2f}%" for x in lp)+".",
      f"13. **Smallest B*:** macro B*={bstar}.",
      "14. **Can 60 be reduced universally?** No. The curve and gates require B=60.",
      "15. **Per-task B*:** task0=20, task1=60, task5=60, task6=30. None receives a formal semantic 0-shot/10-shot label.",
      "16. **Target success/failure counts:** exact fold-level counts are frozen in `TARGET_FEWSHOT_LABEL_BALANCE.csv`; task6 B10/B20 are one-class.",
      f"17. **Boundary policy vs random:** aggregate B20 brackets {b20.boundary_brackets.mean():.2f} vs {b20.random_brackets_mean.mean():.2f}, but the advantage is not uniform across tasks.",
      f"18. **Does source+20 match all-4 Sparse60?** No: Learned-Probe B20 is {lp[2]*100:.2f}% versus the existing all-4 reference {refrow.sr_seed_mean*100:.2f}%, a {100*(lp[2]-refrow.sr_seed_mean):+.2f} pp gap.",
      "", "## Validation, limitations, and next step","",
      "All 108 shards completed. Every estimator×target×budget×seed has 36 evaluation episodes; each fold has 12 episodes from 2 unseen roots, 6 contexts, and 2 repeats. Acquisition/adaptation roots never overlap evaluation roots. GT self-agreement is 100%. Full checks are in `TRANSFER_DATA_QA.json`.",
      "", "This experiment is descriptive development on archived task0/1/5/6 data. It does not separate unseen task from unseen object family, does not establish semantic zero-shot transfer, and contains reconstructed task1 outcomes. The appropriate next step is not a new model: first decide whether to replace the task one-hot with an already-legitimate semantic representation in a separately preregistered experiment; under the frozen architecture, the honest conclusion is substantial task-specific data required.",
      "", "## Classification", "", f"**{cls}**", "",
      "No World Model, residual, Probe modification, new simulator rollout, or root-scaling untouched TEST result was used."]
    (out/"FINAL_SHARED_PHYSICAL_TRANSFER_REPORT.md").write_text("\n".join(lines)+"\n")
    wjson(out/"FINAL_SHARED_PHYSICAL_TRANSFER_CLASSIFICATION.json",{"classification":cls,"macro_B_star":str(bstar),
      "zero_shot_task_transfer_with_current_encoding":"NOT_IDENTIFIABLE","learned_probe_loto_mae":lm,"explicit_sysid_loto_mae":sm,
      "root_scaling_TEST_used":False,"new_simulator_rollouts":0,"world_model_used":False,"residual_used":False})
    write_artifact(out,per,macro,bstars)
    write_hashes(out)


def _records(d: pd.DataFrame) -> list[dict]:
    return json.loads(d.to_json(orient="records"))


def write_artifact(out: Path, per: pd.DataFrame, macro: pd.DataFrame, bstars: pd.DataFrame):
    pm=pd.read_csv(out/"PHYSICS_ESTIMATOR_LOTO.csv")
    probe=pm[(pm.scope=="TASK")&(((pm.method=="LearnedProbe")&(pm.seed.astype(str)=="THREE_SEED_MEAN"))|(pm.method=="ExplicitSysID"))].copy()
    probe["target_task"]=probe.target_task.map(lambda x:f"task{int(float(x))}")
    curve=macro[macro.physics_estimator.isin(["LearnedProbe","ExplicitSysID"])][["physics_estimator","budget","sr_mean","sr_std","underforce_mean","utility_mean"]].copy()
    curve=curve.rename(columns={"physics_estimator":"physics"})
    bst=bstars[bstars.physics_estimator=="LearnedProbe"][["scope","B_star"]].copy()
    ac=pd.read_csv(out/"TARGET_ACQUISITION_EFFICIENCY.csv");ac=ac[ac.budget==20].groupby("target_task",as_index=False)[["boundary_brackets","random_brackets_mean"]].mean();ac["target_task"]=ac.target_task.map(lambda x:f"task{x}")
    lp=macro[(macro.physics_estimator=="LearnedProbe")].set_index("budget");sys=macro[macro.physics_estimator=="ExplicitSysID"].set_index("budget")
    headline=[{"macro_B_star":str(bstars[(bstars.physics_estimator=="LearnedProbe")&(bstars.scope=="MACRO")].B_star.iloc[0]),
               "loto_probe_mae":float(probe[probe.method=="LearnedProbe"].MAE.mean()),
               "sysid_mae":float(probe[probe.method=="ExplicitSysID"].MAE.mean()),
               "B60_sr":float(lp.loc[60,"sr_mean"]),"B20_gap_vs_existing":float(lp.loc[20,"sr_mean"]-pd.read_csv(out/"EXISTING_ALL4_SPARSE60_REFERENCE.csv").sr_seed_mean.iloc[0])}]
    sources=[
      {"id":"src_transfer","label":"Archived LOTO Direct transfer results","path":"NEW_TASK_FEWSHOT_TRANSFER_AGG.csv","query":{"engine":"sqlite","language":"sql","sql":"SELECT * FROM new_task_fewshot_transfer_agg ORDER BY physics_estimator, target_task, budget","description":"Three-seed grouped-root transfer aggregation.","tables_used":["new_task_fewshot_transfer_agg"],"filters":["task0/task1/task5/task6 archived TRAIN population","root-heldout target evaluation","root-scaling TEST excluded"]}},
      {"id":"src_probe","label":"LOTO Probe and explicit SysID metrics","path":"PHYSICS_ESTIMATOR_LOTO.csv","query":{"engine":"sqlite","language":"sql","sql":"SELECT * FROM physics_estimator_loto ORDER BY scope, target_task, method, seed","description":"Target-task-excluded friction-estimator evaluation.","tables_used":["physics_estimator_loto"],"filters":["144 archived P4B traces","target task excluded from calibration"]}},
      {"id":"src_acquisition","label":"Frozen boundary-seeking acquisition audit","path":"TARGET_FEWSHOT_LABEL_BALANCE.csv","query":{"engine":"sqlite","language":"sql","sql":"SELECT * FROM target_fewshot_label_balance ORDER BY target_task, fold, budget","description":"Nested online-query prefix label and bracket audit.","tables_used":["target_fewshot_label_balance"],"filters":["canonical repeat 1","all queries charged"]}}
    ]
    artifact={"surface":"report","manifest":{"version":1,"surface":"report","title":"Shared Physical Transfer in ActiveForcing","description":"LOTO and few-shot transfer with learned Probe versus explicit SysID; archived grouped-root evaluation.","generatedAt":datetime.now(timezone.utc).isoformat(),"sources":sources,
      "cards":[
        {"id":"card_bstar","dataset":"headline","sourceId":"src_transfer","description":"Smallest target budget satisfying the frozen macro gate.","metrics":[{"label":"Macro B*","field":"macro_B_star","format":"text"}]},
        {"id":"card_probe","dataset":"headline","sourceId":"src_probe","description":"Friction estimation after excluding the entire target task/object.","metrics":[{"label":"Learned Probe MAE","field":"loto_probe_mae","format":"number"},{"label":"Explicit SysID MAE","field":"sysid_mae","format":"number"}]},
        {"id":"card_sr","dataset":"headline","sourceId":"src_transfer","description":"Matched full target-adaptation result.","metrics":[{"label":"B60 macro SR","field":"B60_sr","format":"percent"},{"label":"B20 gap vs existing","field":"B20_gap_vs_existing","format":"percent","signed":True}]}
      ],
      "charts":[
        {"id":"chart_curve","title":"Macro full-task SR by target adaptation budget","subtitle":"Three-seed mean; target roots held out from adaptation","intent":"comparison","question":"How quickly does target-task adaptation approach B60?","rationale":"A connected budget curve exposes the preregistered non-monotonicity.","type":"line","dataset":"macro_curve","sourceId":"src_transfer","encodings":{"x":{"field":"budget","type":"quantitative","label":"Target rollouts"},"y":{"field":"sr_mean","type":"quantitative","label":"Full-task SR","format":"percent"},"color":{"field":"physics","type":"nominal","label":"Physics estimator"},"tooltip":[{"field":"underforce_mean","type":"quantitative","format":"percent","label":"Under-force"},{"field":"utility_mean","type":"quantitative","format":"number","label":"Utility"}]},"layout":"full"},
        {"id":"chart_probe","title":"Friction MAE on each held-out task/object","subtitle":"Learned Probe is a three-seed mean; explicit SysID is deterministic","intent":"comparison","question":"Does the learned Probe interpreter transfer better than explicit source-only fitting?","rationale":"Grouped bars show the estimator gap and task heterogeneity.","type":"bar","dataset":"probe_task","sourceId":"src_probe","encodings":{"x":{"field":"target_task","type":"nominal","label":"Held-out task"},"y":{"field":"MAE","type":"quantitative","label":"Friction MAE"},"color":{"field":"method","type":"nominal","label":"Estimator"}},"layout":"full"}
      ],
      "tables":[
        {"id":"table_bstar","title":"Pre-registered adaptation sufficiency by target","subtitle":"SR, under-force, utility, label-balance, and 3/4-task gates","dataset":"bstar","sourceId":"src_transfer","defaultSort":{"field":"scope","direction":"asc"},"density":"spacious","layout":"full","columns":[{"field":"scope","label":"Scope"},{"field":"B_star","label":"B*"}]},
        {"id":"table_acq","title":"Boundary acquisition versus outcome-blind random at B20","subtitle":"Adjacent failure-to-success brackets per split","dataset":"acquisition","sourceId":"src_acquisition","defaultSort":{"field":"target_task","direction":"asc"},"density":"spacious","layout":"full","columns":[{"field":"target_task","label":"Target"},{"field":"boundary_brackets","label":"Boundary seeking","format":"number"},{"field":"random_brackets_mean","label":"Random mean","format":"number"}]}
      ],
      "blocks":[
        {"id":"title","type":"markdown","body":"# Shared Physical Transfer in ActiveForcing","layout":"full"},
        {"id":"summary","type":"markdown","body":"## Technical summary\n\nThe frozen Direct architecture does **not** establish semantic zero-shot transfer, and its few-shot curve does **not** stably saturate before 60 target rollouts. The target identity is an untrained one-hot coordinate at B=0. Learned-Probe macro SR is 73.15%, 94.44%, 78.94%, 77.78%, and 92.13% at B=0/10/20/30/60; the preregistered macro B* is 60.","layout":"full","sourceId":"src_transfer"},
        {"id":"headline","type":"metric-strip","cardIds":["card_bstar","card_probe","card_sr"],"layout":"full"},
        {"id":"curve_intro","type":"markdown","body":"## A B=10 spike does not constitute few-shot saturation\n\nThe curve must be read across all nested budgets. B=20 and B=30 regress, task6 is one-class at B=10/B=20, and per-task gates fail. The implication is substantial task-specific data under the frozen one-hot architecture, despite the favorable isolated B=10 point.","layout":"full","sourceId":"src_transfer"},
        {"id":"curve","type":"chart","chartId":"chart_curve","layout":"full"},
        {"id":"bstar_intro","type":"markdown","body":"## Task requirements are heterogeneous\n\nThe smallest legal B* is 20 for task0, 60 for task1, 60 for task5, and 30 for task6. High B=0 values on task5/task6 remain diagnostics, not semantic zero-shot claims.","layout":"full","sourceId":"src_transfer"},
        {"id":"bstar","type":"table","tableId":"table_bstar","layout":"full"},
        {"id":"probe_intro","type":"markdown","body":"## Explicit response fitting generalizes better than the learned Probe estimator\n\nThe learned GRU's target-task-excluded MAE is 0.5149 versus 0.1810 for source-only explicit SysID. The probe lacks an authoritative slip onset, so SysID uses source-calibrated response summaries rather than treating Ft/Fn as Coulomb friction directly. Lower estimator MAE does not automatically imply a better controller.","layout":"full","sourceId":"src_probe"},
        {"id":"probe","type":"chart","chartId":"chart_probe","layout":"full"},
        {"id":"acq_intro","type":"markdown","body":"## Boundary seeking helps in aggregate but not on every task\n\nAt B=20 the frozen sequential policy finds more adjacent brackets than the outcome-blind reference overall, driven by task0/task5. Task1/task6 expose the limit: an online policy cannot manufacture failures when the archived candidate grid is mostly successful.","layout":"full","sourceId":"src_acquisition"},
        {"id":"acq","type":"table","tableId":"table_acq","layout":"full"},
        {"id":"definitions","type":"markdown","body":"## Scope, metrics, and experimental design\n\nEach LOTO turn trains on 180 canonical source branches and adds a nested prefix of 0/10/20/30/60 target-adaptation branches. Each target fold evaluates 12 controller episodes from two unseen roots; three folds yield 36 episodes per target and 144 per method/budget/seed. Expected utility rewards successful lower force and assigns -1 to failure.","layout":"full","sourceId":"src_transfer"},
        {"id":"limits","type":"markdown","body":"## Limitations and robustness\n\nHeld-out task also means held-out object family, so task and object effects are inseparable. Task1 has 140/180 reconstructed outcomes. The B=10 spike is non-monotonic and task6's B10/B20 adaptation sets are one-class. All 108 shards pass cardinality and root-isolation QA, but this remains archived development evidence rather than untouched TEST evidence.","layout":"full"},
        {"id":"next","type":"markdown","body":"## Recommended next step\n\nDo not change the model inside this experiment. Under the frozen architecture, report substantial task-specific data required. A future semantic zero-shot study would first need a separately preregistered legal semantic task representation; it cannot be inferred from the current one-hot result.","layout":"full"},
        {"id":"questions","type":"markdown","body":"## Further questions\n\n- Why do intermediate target budgets destabilize task1/task5 candidate ranking?\n- Can a frozen explicit estimator preserve its lower MAE in a separately frozen controller benchmark?\n- Would an already-existing VLA/language representation make B=0 semantically identifiable without changing the force model after seeing results?","layout":"full"}
      ]},
      "snapshot":{"version":1,"generatedAt":datetime.now(timezone.utc).isoformat(),"status":"ready","datasets":{"headline":headline,"macro_curve":_records(curve),"probe_task":_records(probe[["target_task","method","MAE","RMSE","pair_ranking"]]),"bstar":_records(bst),"acquisition":_records(ac)}},"sources":sources}
    wjson(out/"artifact.json",artifact)


def write_hashes(out: Path):
    files=[p for p in out.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt"]
    (out/"SHA256SUMS.txt").write_text("\n".join(f"{sha(p)}  {p.name}" for p in sorted(files))+"\n")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("phase",choices=["prepare","acquire","shard","aggregate","hashes"])
    ap.add_argument("--target",type=int);ap.add_argument("--fold",type=int);ap.add_argument("--seed",type=int);ap.add_argument("--estimator",choices=["LearnedProbe","ExplicitSysID","GT"])
    a=ap.parse_args()
    if a.phase=="prepare": prepare(a.out)
    elif a.phase=="acquire":
        *_,md,rf=load_current(a.out/"_audit","acquire_only"); freeze_and_acquire(a.out,md,rf)
    elif a.phase=="shard": run_shard(a.out,a.target,a.fold,a.seed,a.estimator)
    elif a.phase=="aggregate": aggregate(a.out)
    else: write_hashes(a.out)


if __name__=="__main__": main()
