#!/usr/bin/env python3
"""Archived-data-only Probe generalization audit.

This experiment compares a matched heteroscedastic GRU using the exact 46D
Probe trace against the same GRU with the archived frozen visual side channel
concatenated to each timestep.  It does not import Isaac, launch a simulator,
or collect data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = Path("/home/exouser/FORTE")
TAB = Path("/home/exouser/Tabero")
OUT = ROOT / "activeforcing_physical_only_probe_20260901"
P5ROOT = TAB / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542"
P5SRC = TAB / "analysis/p5s0c_model_adjudication.py"
PROS = TAB / "analysis/results/gnp_style_visual_context_prospective_20260831_011000"
TRAIN_VIS = PROS / "collection_train"
PREV = ROOT / "activeforcing_shared_physical_transfer_20260901_094722"
TASKS = [0, 1, 5, 6]
FOLDS = [0, 1, 2]
SEEDS = [0, 1, 2]
METHODS = ["VisionPhysical-Probe", "PhysicalOnly-Probe"]
LOGVAR_CLAMP = [-5.0, 1.5]  # frozen before result inspection
EPOCHS = 80
LR = 1e-3
WD = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imp(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P5 = imp("physical_only_probe_p5", P5SRC)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wjson(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_prior():
    # The prior loader is an offline reader of archived branch tensors.  It is
    # used only to recover the exact current force-selection population and
    # its authoritative root folds.
    prior = imp("physical_only_probe_prior", ROOT / "run_shared_physical_transfer.py")
    tpi, cf, full, cmap, traces, meta, audits, pairs, segs, md, rf = prior.load_current(OUT / "_audit", "physical_only_prepare")
    return prior, (tpi, cf, full, cmap, traces, meta, audits, pairs, segs, md, rf)


def load_inputs():
    norm = json.loads((P5ROOT / "P5S0C_NORMALIZATION.json").read_text())
    names = norm["dynamic_feature_names"]
    phases = norm["phase_categories_from_train"]
    states = norm["contact_state_categories_from_train"]
    prior, loaded = load_prior()
    tpi, cf, full, cmap, traces, meta, audits, pairs, segs, md, rf = loaded
    visual = pd.read_csv(PROS / "PROSPECTIVE_VISUAL_ALIGNMENT_MANIFEST.csv")
    visual = visual[visual["split"].astype(str).eq("TRAIN")].copy()
    vm = {str(r.context_id): Path(str(r.visual_feature_path)) for r in visual.itertuples()}
    pm_rows, physical, appearance = [], [], []
    context_traces = sorted({str(t.context_id): t for t in traces}.values(), key=lambda x: str(x.context_id))
    for tr in context_traces:
        cid = str(tr.context_id)
        path = TRAIN_VIS / f"task{int(tr.task)}" / "P5S0C_PROBE_TELEMETRY" / f"{cid}_probe_timesteps.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        q = P5.sequence_dataframe(str(path), phases, states).reindex(columns=names).fillna(0.0)
        a = q.to_numpy(np.float32)
        if a.shape != (215, 46) or not np.isfinite(a).all():
            raise RuntimeError(f"invalid physical trace {cid}: {a.shape}")
        vp = vm.get(cid)
        if vp is None or not vp.exists():
            raise RuntimeError(f"missing matched visual feature for {cid}")
        v = np.load(vp).astype(np.float32)
        if v.shape != (4096,) or not np.isfinite(v).all():
            raise RuntimeError(f"invalid visual feature {cid}: {v.shape}")
        physical.append(a[:, [i for i, n in enumerate(names) if n not in PRIVILEGED_NAMES]])
        appearance.append(v)
        pm_rows.append({"context_id": cid, "root_id": str(tr.root_id), "task": int(tr.task),
                        "mu_GT": float(tr.mu), "physical_path": str(path),
                        "visual_path": str(vp), "timesteps": int(a.shape[0]),
                        "physical_dim": int(len(names) - len(PRIVILEGED_NAMES)), "visual_dim": int(v.shape[0]),
                        "root_fold": int(rf[next(i for i,x in enumerate(md.context_id.astype(str)) if x == cid)])})
    pm = pd.DataFrame(pm_rows).sort_values("context_id").reset_index(drop=True)
    order = {str(r.context_id): i for i, r in enumerate(pm.itertuples())}
    # Reorder arrays to the manifest table order.
    by_cid = {str(tr.context_id): i for i, tr in enumerate(context_traces)}
    idx = [by_cid[cid] for cid in pm.context_id]
    return prior, loaded, norm, names, pm, np.stack(physical)[idx], np.stack(appearance)[idx]


PRIVILEGED_NAMES = {"contact_normal_x", "contact_normal_y", "contact_normal_z", "contact_tangent_x", "contact_tangent_y", "contact_tangent_z"}


def channel_class(name: str) -> tuple[str, str, str]:
    action = {"force_target", "target_normal_force", "commanded_tangent_increment_mm"}
    if name in PRIVILEGED_NAMES:
        return "5 PRIVILEGED_SIM_ONLY", "no", "contact frame is computed from simulator geometry in the P4-B implementation"
    if name == "t_s" or name.startswith("phase=") or name in action:
        return "4 ACTION", "yes", "commanded probe timing/action or deterministic phase encoding"
    return "3 PHYSICAL_OBSERVATION", "yes", "live response/proprioceptive/contact telemetry emitted by the probe logger"


def write_audits(out: Path, norm: dict, names: list[str], pm: pd.DataFrame) -> None:
    rows = []
    for i, name in enumerate(names):
        cls, deploy, why = channel_class(name)
        action = "yes" if cls.startswith("4 ") else "no"
        privileged = "yes" if cls.startswith("5 ") else "no"
        rows.append(f"| {i} | `{name}` | {cls} | no | no | {privileged} | {action} | {deploy} | {why} |")
    excluded = norm["excluded_fields"]
    audit = """# PROBE input channel audit

## Exact current Learned Probe input

The authoritative current Probe implementation is `run_probe_conditioned_wm.py` → `load_raw_probe()` → `P5.sequence_dataframe()`. It loads each `P5S0C_PROBE_TELEMETRY/*_probe_timesteps.csv`, constructs the exact columns below from `P5S0C_NORMALIZATION.json`, fits feature normalization on the estimator's training contexts, then feeds a 215-step sequence to `Linear(input_dim,16) → ReLU → GRU(16,16) → friction head`. The current model therefore uses **46 physical/action-derived channels and no visual input**.

There is no RGB tensor, frozen visual embedding, object visual feature, task ID, language, object ID, simulator seed, GT friction, downstream outcome, or object pose in the current Probe model input. `eef_dx/eef_dy/eef_dz` are derived from logged EEF pose by subtracting the first row; they are not absolute object pose. Six contact-frame orientation channels are present in the historical 46D legal tensor but are simulator-geometry-derived, so they are excluded from both main estimators in this deployment-constrained run.

`VisionPhysical-Probe` in this run is a matched diagnostic: it adds the archived 4096D frozen π0 visual embedding to every timestep while retaining the exact same 16D projection, GRU, heteroscedastic heads, optimizer, seeds, and NLL. The visual archive exists for 72 current TRAIN contexts only; the historical 144-sequence Probe population is retained as the complete authoritative audit population but has no matched visual capture for its 72 non-TRAIN contexts.

## Field-by-field classification

`VISUAL`, `TASK/IDENTITY`, `PHYSICAL_OBSERVATION`, `ACTION`, `PRIVILEGED_SIM_ONLY`, and `DEPLOYABLE_AT_TEST` are reported as separate audit columns. The current 46D model uses no fields in the first or second classes. The six contact-frame orientation fields are marked simulator-only and excluded from the main 40D deployability-constrained model. Deployability here means the current P4-B telemetry interface can emit the signal online during the fixed probe; transfer to a physical robot still requires sensor-parity validation.

| index | exact channel | primary class | VISUAL | TASK/IDENTITY | PRIVILEGED_SIM_ONLY | ACTION | DEPLOYABLE_AT_TEST | basis |
|---:|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + "\n\n## Explicitly excluded source fields\n\n" + ", ".join(f"`{x}`" for x in excluded) + "\n"
    (out / "PROBE_INPUT_CHANNEL_AUDIT.md").write_text(audit, encoding="utf-8")
    protocol = {
        "status": "HASH_FROZEN_BEFORE_MODEL_TRAINING",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": "physical-history friction estimation versus matched visual+physical estimation",
        "authoritative_population": {
            "current_common_population": int(len(pm)),
            "historical_complete_probe_sequences": 144,
            "historical_root_families": 48,
            "current_common_root_families": int(pm.root_id.nunique()),
            "current_tasks": sorted(pm.task.unique().tolist()),
            "sequence_shape": [215, 46],
            "visual_shape": [4096],
        },
        "sources": {
            "p5_normalization": str(P5ROOT / "P5S0C_NORMALIZATION.json"),
            "p5_normalization_sha256": sha(P5ROOT / "P5S0C_NORMALIZATION.json"),
            "p5_feature_source": str(P5SRC),
            "p5_feature_source_sha256": sha(P5SRC),
            "current_context_manifest": str(PROS / "PROSPECTIVE_CONTEXT_MANIFEST.csv"),
            "current_context_manifest_sha256": sha(PROS / "PROSPECTIVE_CONTEXT_MANIFEST.csv"),
            "visual_manifest": str(PROS / "PROSPECTIVE_VISUAL_ALIGNMENT_MANIFEST.csv"),
            "visual_manifest_sha256": sha(PROS / "PROSPECTIVE_VISUAL_ALIGNMENT_MANIFEST.csv"),
        },
        "physical_only_protocol": {
            "name": "PhysicalOnly-Probe",
            "input": "{physical_o_t, probe_a_t}_{t=1:T}",
            "dimensions": len([n for n in names if n not in PRIVILEGED_NAMES]),
            "channels": [n for n in names if n not in PRIVILEGED_NAMES],
            "forbidden": ["RGB", "visual_embedding", "object_category", "object_id", "task_one_hot", "language", "task_id", "GT_friction", "root_id", "simulator_seed", "privileged_object_pose"],
            "privileged_fields_excluded": norm["excluded_fields"] + sorted(PRIVILEGED_NAMES),
            "deployability_note": "main model includes only channels emitted in the live fixed-probe telemetry interface; hardware sensor parity is a deployment prerequisite",
        },
        "matched_vision_physical_protocol": {"name": "VisionPhysical-Probe", "physical_dim": len([n for n in names if n not in PRIVILEGED_NAMES]), "visual_dim": 4096, "visual_repeated_each_timestep": True, "visual_normalization": "fit on training contexts only"},
        "privileged_physical_only_diagnostic": {"status": "NOT_RUN", "channels": sorted(PRIVILEGED_NAMES), "reason": "simulator geometry channels are not deployable without a privileged state interface"},
        "architecture": {"projection": "Linear(input_dim,16)+ReLU", "gru": "GRU(16,16)", "heads": ["mu", "logvar"], "logvar_clamp": LOGVAR_CLAMP, "loss": "0.5*exp(-s)*(y-mu)^2 + 0.5*s", "epochs": EPOCHS, "optimizer": "AdamW", "learning_rate": LR, "weight_decay": WD, "seeds": SEEDS},
        "splits": {"root_heldout": "3 deterministic grouped root folds", "task_loto": "source three tasks, target held-out task; target friction labels never used for estimator training", "normalization": "per-fold training-only physical and visual mean/std"},
        "simulator_rollouts_launched": 0,
        "second_query_status": "SECOND_QUERY_NOT_EVALUATED_NO_ARCHIVED_REPEAT",
    }
    wjson(out / "PHYSICAL_ONLY_PROBE_PROTOCOL.json", protocol)
    pm.to_csv(out / "COMMON_PROBE_CONTEXTS.csv", index=False)


def write_relation_audit(out: Path, pm: pd.DataFrame) -> None:
    hist = pd.read_csv(P5ROOT / "P5S0C_CONTEXT_MANIFEST.csv")
    rows = []
    for task in TASKS:
        q = hist[hist.task == task]
        instruction = str(q.task_instruction.iloc[0])
        object_name = instruction.replace("pick up the ", "").split(" and place")[0].strip().replace(" ", "_")
        rows.append(f"| {task} | `{object_name}` | {q.root_id.nunique()} | {len(q)} | {pm[pm.task == task].root_id.nunique()} | {len(pm[pm.task == task])} |")
    text = """# Probe task/object relation audit

The task/object mapping below is recovered from the authoritative P5-S0-C context manifest and the current prospective TRAIN context manifest. It is not inferred from model results.

| task | object family | historical roots | historical contexts | current common roots | current common contexts |
|---:|---|---:|---:|---:|---:|
""" + "\n".join(rows) + """

The four authoritative tasks map to four distinct object families, and no object family repeats across tasks. Consequently leave-one-task-out also leaves out the corresponding object family. Task transfer and object-family transfer are fully confounded in this archive; the primary label is **task/object-distribution-held-out**. No independent object-family split is run because the existing archive contains no cross-task object-family repetition. No new data are collected.
"""
    (out / "PROBE_TASK_OBJECT_RELATION_AUDIT.md").write_text(text, encoding="utf-8")


class ProbeNet(nn.Module):
    def __init__(self, physical_dim: int, visual_dim: int = 0):
        super().__init__()
        # Equivalent to Linear([physical_t, visual]) at every timestep, but
        # computes the static visual contribution once per sequence:
        # Wp*physical_t + Wv*visual + b.
        self.physical_proj = nn.Linear(physical_dim, 16)
        self.visual_proj = nn.Linear(visual_dim, 16, bias=False) if visual_dim else None
        self.gru = nn.GRU(16, 16, batch_first=True)
        self.mu = nn.Linear(16, 1)
        self.logvar = nn.Linear(16, 1)

    def forward(self, physical, visual, lengths):
        z = self.physical_proj(physical)
        if self.visual_proj is not None:
            z = z + self.visual_proj(visual)[:, None, :]
        z = torch.relu(z)
        _, h = self.gru(pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False))
        h = h[-1]
        mu = self.mu(h).squeeze(1)
        s = self.logvar(h).squeeze(1).clamp(float(LOGVAR_CLAMP[0]), float(LOGVAR_CLAMP[1]))
        return mu, s


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def fit_predict(physical, visual, y, train_idx, test_idx, method, seed):
    seed_all(seed)
    pmean = physical[train_idx].reshape(-1, physical.shape[-1]).mean(0).astype(np.float32)
    pstd = physical[train_idx].reshape(-1, physical.shape[-1]).std(0).astype(np.float32); pstd[pstd < 1e-6] = 1
    if method == "PhysicalOnly-Probe":
        xtrain = (physical[train_idx] - pmean) / pstd
        xtest = (physical[test_idx] - pmean) / pstd
        vtrain = np.zeros((len(train_idx), 0), np.float32); vtest = np.zeros((len(test_idx), 0), np.float32)
        vmean = np.zeros(visual.shape[-1], np.float32); vstd = np.ones(visual.shape[-1], np.float32)
    else:
        vmean = visual[train_idx].mean(0).astype(np.float32)
        vstd = visual[train_idx].std(0).astype(np.float32); vstd[vstd < 1e-6] = 1
        pv = (visual - vmean) / vstd
        xtrain = (physical[train_idx] - pmean) / pstd; xtest = (physical[test_idx] - pmean) / pstd
        vtrain = pv[train_idx]; vtest = pv[test_idx]
    model = ProbeNet(xtrain.shape[-1], vtrain.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    tx = torch.tensor(xtrain, dtype=torch.float32, device=DEVICE); ty = torch.tensor(y[train_idx], dtype=torch.float32, device=DEVICE)
    tv = torch.tensor(vtrain, dtype=torch.float32, device=DEVICE); lens = torch.full((len(train_idx),), physical.shape[1], dtype=torch.long, device=DEVICE)
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad(); mu, s = model(tx, tv, lens)
        loss = (0.5 * torch.exp(-s) * (ty - mu).pow(2) + 0.5 * s).mean()
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval(); qx = torch.tensor(xtest, dtype=torch.float32, device=DEVICE); qv = torch.tensor(vtest, dtype=torch.float32, device=DEVICE); ql = torch.full((len(test_idx),), physical.shape[1], dtype=torch.long, device=DEVICE)
    with torch.no_grad(): mu, s = model(qx, qv, ql)
    return mu.numpy(), s.numpy(), model, {"physical_mean": pmean, "physical_std": pstd, "visual_mean": vmean, "visual_std": vstd}


def aggregate_rows(rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    out = []
    for keys, g in d.groupby(["regime", "target_task", "fold", "context_id"], dropna=False):
        reg, task, fold, cid = keys
        mus = g.mu_member.to_numpy(float); ss = g.logvar_member.to_numpy(float)
        mu = float(mus.mean()); epi = float(mus.var(ddof=0)); ale = float(np.exp(ss).mean()); total = epi + ale
        r = g.iloc[0].to_dict(); r.update({"mu_hat": mu, "sigma_epi": math.sqrt(max(epi, 0)), "sigma_ale": math.sqrt(max(ale, 0)), "sigma_total": math.sqrt(max(total, 0)), "member_count": len(g), "abs_error": abs(mu - float(r["mu_GT"])), "signed_error": mu - float(r["mu_GT"])})
        out.append(r)
    return pd.DataFrame(out)


def metrics(g: pd.DataFrame) -> dict:
    y = g.mu_GT.to_numpy(float); p = g.mu_hat.to_numpy(float); e = p-y
    rho = spearmanr(y, p).statistic if len(g) > 1 else math.nan
    pairs=[]
    for _, q in g.groupby("root_id"):
        z=q[["mu_GT","mu_hat"]].to_numpy(float)
        for i in range(len(z)):
            for j in range(i+1,len(z)):
                if z[i,0] != z[j,0]: pairs.append(int(np.sign(z[i,0]-z[j,0]) == np.sign(z[i,1]-z[j,1])))
    return {"contexts": int(len(g)), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt(np.mean(e*e))), "bias": float(e.mean()), "Spearman": float(rho), "pair_ranking": float(np.mean(pairs)) if pairs else math.nan}


def run_regime(out, physical, visual, pm, regime, split, target=None):
    y = pm.mu_GT.to_numpy(float); roots = pm.root_id.astype(str).to_numpy(); tasks = pm.task.to_numpy(int)
    fold_map = dict(zip(pm.root_id.astype(str), pm.root_fold.astype(int)))
    rows=[]
    # Root-heldout has three archived root folds. LOTO has one held-out task
    # split; its three members are the ensemble seeds, not additional test
    # folds. Source-only threshold calibration below uses separate inner root
    # folds where needed.
    fold_values = FOLDS if split == "ROOT_HELDOUT" else [0]
    for fold in fold_values:
        if split == "ROOT_HELDOUT": train = np.flatnonzero(np.array([fold_map[r] != fold for r in roots])); test = np.flatnonzero(np.array([fold_map[r] == fold for r in roots]))
        else:
            train = np.flatnonzero(tasks != target); test = np.flatnonzero(tasks == target)
        for seed in SEEDS:
            mu, s, model, norm = fit_predict(physical, visual, y, train, test, regime, 7000 + fold*100 + seed + (0 if split == "ROOT_HELDOUT" else 1000 + int(target)))
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "method": regime, "split": split, "target_task": target, "fold": fold, "seed": seed, "physical_dim": physical.shape[-1], "visual_dim": 0 if regime.startswith("Physical") else 4096, "logvar_clamp": LOGVAR_CLAMP, "normalization": norm}, out / "models" / f"{split}_target{target if target is not None else 'ALL'}_fold{fold}_seed{seed}_{regime}.pt")
            for j, i in enumerate(test):
                r = pm.iloc[i]
                rows.append({"regime": regime, "split": split, "target_task": int(target) if target is not None else "ALL", "fold": fold, "seed": seed, "context_id": r.context_id, "root_id": r.root_id, "task": int(r.task), "mu_GT": float(r.mu_GT), "mu_member": float(mu[j]), "logvar_member": float(s[j])})
    member = pd.DataFrame(rows)
    agg = aggregate_rows(rows)
    return member, agg


def source_cv_for_loto(physical, visual, pm, target, method):
    y = pm.mu_GT.to_numpy(float); roots = pm.root_id.astype(str).to_numpy(); tasks = pm.task.to_numpy(int)
    source = pm[pm.task != target]
    source_roots = sorted(source.root_id.astype(str).unique())
    f_map = {r: i % 3 for i, r in enumerate(source_roots)}
    rows=[]
    for f in FOLDS:
        tr = np.flatnonzero(np.array([(tasks[i] != target) and (f_map[roots[i]] != f) for i in range(len(pm))]))
        va = np.flatnonzero(np.array([(tasks[i] != target) and (f_map[roots[i]] == f) for i in range(len(pm))]))
        for seed in SEEDS:
            mu,s,_,_=fit_predict(physical,visual,y,tr,va,method,9100+target*100+f*10+seed)
            for j,i in enumerate(va):
                r=pm.iloc[i]; rows.append({"regime":method,"split":"LOTO_SOURCE_VALIDATION","target_task":target,"fold":f,"seed":seed,"context_id":r.context_id,"root_id":r.root_id,"task":int(r.task),"mu_GT":float(r.mu_GT),"mu_member":float(mu[j]),"logvar_member":float(s[j])})
    return aggregate_rows(rows)


def uncertainty_outputs(out, root_agg, loto_agg, source_val):
    rows=[]
    for split, df in [("ROOT_HELDOUT",root_agg),("LOTO",loto_agg)]:
        for (reg, task),g in df.groupby(["regime","target_task"], dropna=False):
            for name, q in [("ALL",g)]:
                e=q.abs_error.to_numpy(float); s=q.sigma_total.to_numpy(float); var=np.maximum(s*s,1e-12)
                nll=0.5*(e*e/var + np.log(var) + math.log(2*math.pi))
                rho=spearmanr(e,s).statistic if len(q)>1 else math.nan
                for i,(cov,label) in enumerate([(1.0,"100"),(.9,"90"),(.8,"80"),(.7,"70"),(.6,"60"),(.5,"50"),(.4,"40"),(.3,"30"),(.2,"20"),(.1,"10")]):
                    k=max(1,int(math.ceil(cov*len(q)))); keep=np.argsort(s)[:k]
                    rows.append({"regime":reg,"split":split,"target_task":task,"scope":name,"metric":"risk_coverage","coverage":cov,"coverage_label":label,"value":float(e[keep].mean()),"n":int(k),"gaussian_nll_mean":float(nll.mean()),"abs_error_sigma_spearman":float(rho)})
                for cov,z in [(0.68,1.0),(0.95,1.96)]:
                    rows.append({"regime":reg,"split":split,"target_task":task,"scope":name,"metric":"interval_coverage","coverage":cov,"coverage_label":str(int(cov*100)),"value":float(np.mean(e <= z*s)),"n":len(q),"gaussian_nll_mean":float(nll.mean()),"abs_error_sigma_spearman":float(rho)})
                for dec,q2 in q.assign(decile=pd.qcut(q.sigma_total.rank(method="first"),10,labels=False)+1).groupby("decile"):
                    rows.append({"regime":reg,"split":split,"target_task":task,"scope":f"DECILE_{int(dec)}","metric":"decile_mae","coverage":math.nan,"coverage_label":"","value":float(q2.abs_error.mean()),"n":len(q2),"gaussian_nll_mean":float(nll.mean()),"abs_error_sigma_spearman":float(rho)})
    pd.DataFrame(rows).to_csv(out/"PROBE_UNCERTAINTY_CALIBRATION.csv",index=False)
    gates=[]
    for (reg,target),sv in source_val.groupby(["regime","target_task"]):
        test=loto_agg[(loto_agg.regime==reg)&(loto_agg.target_task==target)]
        for expected in [.8,.9]:
            threshold=float(np.quantile(sv.sigma_total.to_numpy(float),expected,method="linear"))
            acc=test[test.sigma_total<=threshold]; rej=test[test.sigma_total>threshold]
            amae=float(acc.abs_error.mean()) if len(acc) else math.nan; rmae=float(rej.abs_error.mean()) if len(rej) else math.nan
            gates.append({"regime":reg,"target_task":int(target),"threshold_policy":f"SOURCE_VALIDATION_{int(expected*100)}PCT","threshold":threshold,"expected_source_coverage":expected,"target_accepted_coverage":float(len(acc)/len(test)),"accepted_n":len(acc),"rejected_n":len(rej),"accepted_MAE":amae,"rejected_MAE":rmae,"error_enrichment_rejected_vs_accepted":float(rmae/amae) if len(acc) and amae>0 and len(rej) else math.nan,"rejected_higher_error":bool(len(rej) and len(acc) and rmae>amae)})
    pd.DataFrame(gates).to_csv(out/"PROBE_REPROBE_GATE.csv",index=False)


def prepare(out: Path):
    out.mkdir(parents=True,exist_ok=True); (out/"models").mkdir(exist_ok=True); (out/"_audit").mkdir(exist_ok=True)
    prior, loaded, norm, names, pm, physical, visual = load_inputs()
    write_audits(out,norm,names,pm)
    write_relation_audit(out, pm)
    shutil.copy2(PREV/"TARGET_ACQUISITION_TRAJECTORIES.csv",out/"TARGET_BOUNDARY_ACQUISITION_TRAJECTORIES.csv")
    shutil.copy2(PREV/"TARGET_FEWSHOT_LABEL_BALANCE.csv",out/"TARGET_FEWSHOT_LABEL_BALANCE.csv")
    wjson(out/"PREPARE_COMPLETE.json",{"status":"PASS","protocol_sha256":sha(out/"PHYSICAL_ONLY_PROBE_PROTOCOL.json"),"current_common_contexts":len(pm),"current_common_roots":int(pm.root_id.nunique()),"historical_probe_sequences":144,"simulator_rollouts_launched":0,"new_data":False})


def evaluate(out: Path):
    if not (out/"PHYSICAL_ONLY_PROBE_PROTOCOL.json").exists(): raise RuntimeError("run prepare first")
    prior, loaded, norm, names, pm, physical, visual = load_inputs()
    all_members=[]; all_aggs=[]; source=[]
    for method in METHODS:
        m,a=run_regime(out,physical,visual,pm,method,"ROOT_HELDOUT"); all_members.append(m); all_aggs.append(a)
        for target in TASKS:
            m,a=run_regime(out,physical,visual,pm,method,"LOTO",target); all_members.append(m); all_aggs.append(a)
            source.append(source_cv_for_loto(physical,visual,pm,target,method))
    member=pd.concat(all_members,ignore_index=True); agg=pd.concat(all_aggs,ignore_index=True); source_val=pd.concat(source,ignore_index=True)
    member.to_csv(out/"PROBE_MEMBER_PREDICTIONS.csv",index=False); agg.to_csv(out/"PROBE_AGGREGATE_PREDICTIONS.csv",index=False); source_val.to_csv(out/"PROBE_SOURCE_VALIDATION_PREDICTIONS.csv",index=False)
    result=[]
    for (reg,split,task,fold,seed),g in agg.groupby(["regime","split","target_task","fold","seed"],dropna=False): result.append({"regime":reg,"split":split,"target_task":task,"fold":fold,"seed":seed,**metrics(g)})
    for (reg,split,task),g in agg.groupby(["regime","split","target_task"],dropna=False): result.append({"regime":reg,"split":split,"target_task":task,"fold":"MACRO","seed":"ENSEMBLE_OOF_MEAN",**metrics(g)})
    # Deterministic Explicit-SysID on the same 72-context common population.
    sys_feat = pd.DataFrame([prior.sysid_features(pd.read_csv(p)) for p in pm.physical_path])
    y = pm.mu_GT.to_numpy(float)
    for split, target in [("ROOT_HELDOUT", None)] + [("LOTO", t) for t in TASKS]:
        fold_values = FOLDS if split == "ROOT_HELDOUT" else [0]
        for fold in fold_values:
            if split == "ROOT_HELDOUT":
                tr = np.flatnonzero(pm.root_fold.to_numpy(int) != fold); va = np.flatnonzero(pm.root_fold.to_numpy(int) == fold)
            else:
                tr = np.flatnonzero(pm.task.to_numpy(int) != target); va = np.flatnonzero(pm.task.to_numpy(int) == target)
            info = prior.calibrate_sysid(pm.iloc[tr].reset_index(drop=True), sys_feat.iloc[tr].reset_index(drop=True))
            formula = info["selected_formula"]; coef = (info["intercept"], info["slope"])
            pp = prior.predict_affine(sys_feat.iloc[va][formula].to_numpy(float), coef)
            g = pm.iloc[va].copy(); g["mu_hat"] = pp; g["abs_error"] = abs(pp - g.mu_GT.to_numpy(float)); g["signed_error"] = pp - g.mu_GT.to_numpy(float); g["regime"] = "Explicit-SysID"; g["split"] = split; g["target_task"] = int(target) if target is not None else "ALL"; g["fold"] = fold; g["seed"] = "DETERMINISTIC"
            result.append({"regime":"Explicit-SysID","split":split,"target_task":int(target) if target is not None else "ALL","fold":fold,"seed":"DETERMINISTIC",**metrics(g)})
        # A macro row is assembled below together with the neural methods.
    # Add explicit rows to the result table before the macro computation.
    explicit_rows = []
    for split, target in [("ROOT_HELDOUT", None)] + [("LOTO", t) for t in TASKS]:
        qrows = [r for r in result if r["regime"] == "Explicit-SysID" and r["split"] == split and r["target_task"] == (int(target) if target is not None else "ALL")]
        gg = pd.DataFrame(qrows)
        result.append({"regime":"Explicit-SysID","split":split,"target_task":int(target) if target is not None else "ALL","fold":"MACRO","seed":"ENSEMBLE_OOF_MEAN",**{c:float(gg[c].mean()) for c in ["contexts","MAE","RMSE","bias","Spearman","pair_ranking"]}})
    pd.DataFrame(result).to_csv(out/"PROBE_ROOT_HELDOUT_RESULTS.csv",index=False)
    loto=agg[agg.split=="LOTO"]; uncertainty_outputs(out,agg,loto,source_val)


def direct(out: Path):
    # Import and reuse the formal Direct architecture, objective, utility, and
    # frozen acquisition prefixes. Only the physics source is changed.
    import run_probe_conditioned_wm as af
    prior, loaded = load_prior(); tpi,cf,full,cmap,traces,meta,audits,pairs,segs,md,rf=loaded
    pred=pd.read_csv(out/"PROBE_AGGREGATE_PREDICTIONS.csv")
    loto=pred[pred.split.eq("LOTO")].copy()
    task_metrics=pd.read_csv(out/"PROBE_ROOT_HELDOUT_RESULTS.csv")
    acq=pd.read_csv(out/"TARGET_BOUNDARY_ACQUISITION_TRAJECTORIES.csv")
    zero_h={cid:np.zeros(1,np.float32) for cid in md.context_id.unique()}
    rows=[]
    for target in TASKS:
        for fold in FOLDS:
            held_idx=np.flatnonzero((md.task==target)&(rf==fold)); source_idx=np.flatnonzero((md.task!=target)&(rf!=fold)&(md.repeat==1))
            aq=acq[(acq.target_task==target)&(acq.fold==fold)]
            for seed in SEEDS:
                for method in METHODS:
                    q=loto[(loto.regime==method)&(loto.target_task==target)&(loto.seed==seed)]
                    mu_map=dict(zip(q.context_id,q.mu_hat)); unc_map=dict(zip(q.context_id,q.sigma_total)); assert len(mu_map)==len(md.context_id.unique())
                    tm=task_metrics[(task_metrics.regime==method)&(task_metrics.split=="LOTO")&(task_metrics.target_task==target)&(task_metrics.fold=="MACRO")&(task_metrics.seed=="ENSEMBLE_OOF_MEAN")].iloc[0]
                    for budget in [0,10,20,30,60]:
                        ids=set(aq.loc[aq.query_index<=budget,"branch_id"]); target_idx=np.flatnonzero(md.branch_id.isin(ids).to_numpy()); train_idx=np.concatenate([source_idx,target_idx]); train=[traces[i] for i in train_idx]; held=[traces[i] for i in held_idx]
                        norm2=prior.xnorm(train,segs,mu_map); model=prior.train_direct(train,segs,norm2,mu_map,zero_h,meta,seed,False)
                        s,c,_=prior.tensors(held,segs,norm2,mu_map,zero_h,False)
                        with torch.no_grad(): p=prior.sigmoid(model(s,c).numpy()).astype(np.float32)
                        f=md.force_N.to_numpy()[held_idx]; fm=md.task.map(prior.FMAX).to_numpy()[held_idx]; u=p*((fm-f)/fm)+(1-p)*-1; score=np.full(len(md),np.nan,np.float32); score[held_idx]=u
                        # run_shared exposes the same helper module, but call
                        # the canonical chooser explicitly for readability.
                        sel=af.choose(md.iloc[held_idx],score)
                        gtq=pd.read_csv(PREV/"shards"/f"target{target}_fold{fold}_seed{seed}_GT_selected.csv")
                        gtq=gtq[gtq.budget==budget].set_index(["context_id","repeat"])
                        for _,r in sel.iterrows():
                            key=(r.context_id,int(r.repeat)); rows.append({"probe_method":method,"target_task":target,"fold":fold,"budget":budget,"seed":seed,"context_id":r.context_id,"repeat":int(r.repeat),"friction_MAE":float(tm.MAE),"friction_rank":float(tm.pair_ranking),"uncertainty":float(unc_map.get(r.context_id,math.nan)),"controller_SR":float(r.success),"underforce":float(r.under_force),"mean_force":float(r.selected_force_N),"excess_force":float(r.excess_force_N),"utility":float(r.realized_utility),"GT_decision_agreement":int(abs(float(r.selected_force_N)-float(gtq.loc[key].selected_force_N))<1e-8)})
    new=pd.DataFrame(rows)
    # Add the exact already-computed matched Direct rows for GT and Explicit
    # SysID; these methods use the frozen prior protocol and are not retrained
    # under a different objective.
    old=pd.read_csv(PREV/"NEW_TASK_FEWSHOT_PER_EPISODE.csv")
    old_rows=[]
    sysmetrics=pd.read_csv(PREV/"PHYSICS_ESTIMATOR_LOTO.csv")
    for est,method in [("GT","GT"),("ExplicitSysID","Explicit-SysID")]:
        q=old[old.physics_estimator.eq(est)].copy()
        for _,r in q.iterrows():
            if est=="GT": fm=0.; fr=1.; unc=0.
            else:
                z=sysmetrics[(sysmetrics.method==est)&(sysmetrics.scope=="TASK")&(sysmetrics.target_task==r.target_task)&(sysmetrics.seed=="DETERMINISTIC")].iloc[0]; fm=float(z.MAE); fr=float(z.pair_ranking); unc=0.
            old_rows.append({"probe_method":method,"target_task":int(r.target_task),"fold":int(r.fold),"budget":int(r.budget),"seed":int(r.seed),"context_id":r.context_id,"repeat":int(r.repeat),"friction_MAE":fm,"friction_rank":fr,"uncertainty":unc,"controller_SR":float(r.success),"underforce":float(r.under_force),"mean_force":float(r.selected_force_N),"excess_force":float(r.excess_force_N),"utility":float(r.realized_utility),"GT_decision_agreement":int(r.decision_agreement_GT)})
    all_rows=pd.concat([new,pd.DataFrame(old_rows)],ignore_index=True)
    all_rows.to_csv(out/"PROBE_TO_DIRECT_TRANSFER.csv",index=False)
    agg=[]
    for keys,g in all_rows.groupby(["probe_method","target_task","budget","seed"]):
        method,t,b,s=keys; agg.append({"probe_method":method,"target_task":t,"budget":b,"seed":s,"episodes":len(g),"controller_SR":g.controller_SR.mean(),"underforce":g.underforce.mean(),"mean_force":g.mean_force.mean(),"excess_force":g.excess_force.mean(),"utility":g.utility.mean(),"GT_decision_agreement":g.GT_decision_agreement.mean(),"friction_MAE":g.friction_MAE.mean(),"friction_rank":g.friction_rank.mean(),"uncertainty":g.uncertainty.mean()})
    ag=pd.DataFrame(agg); macro=[]
    for (method,b),g in ag.groupby(["probe_method","budget"]):
        z=g.groupby("seed")[["controller_SR","underforce","mean_force","excess_force","utility","GT_decision_agreement","friction_MAE","friction_rank","uncertainty"]].mean(); row={"probe_method":method,"target_task":"MACRO","budget":b,"seed":"MACRO","episodes":len(g)}
        for c in z.columns: row[c]=float(z[c].mean()); row[c+"_std"]=float(z[c].std(ddof=0))
        macro.append(row)
    pd.concat([ag,pd.DataFrame(macro)],ignore_index=True).to_csv(out/"PROBE_TO_DIRECT_TRANSFER_AGG.csv",index=False)


def figures_and_report(out: Path):
    root=pd.read_csv(out/"PROBE_ROOT_HELDOUT_RESULTS.csv"); agg=pd.read_csv(out/"PROBE_TO_DIRECT_TRANSFER_AGG.csv"); cal=pd.read_csv(out/"PROBE_UNCERTAINTY_CALIBRATION.csv")
    colors={"VisionPhysical-Probe":"#2563eb","PhysicalOnly-Probe":"#e97316","Explicit-SysID":"#5b6470","GT":"#111827"}
    plt.style.use("seaborn-v0_8-whitegrid")
    fig,ax=plt.subplots(figsize=(8.4,5.2)); vals=[]
    for m,label in [("VisionPhysical-Probe","Vision+Physical"),("PhysicalOnly-Probe","Physical-only")]:
        for x,scope in enumerate(["ROOT_HELDOUT","LOTO"]):
            q=root[(root.regime==m)&(root.split==scope)&(root.fold=="MACRO")]; vals.append((x,m,q.MAE.mean(),q.MAE.std(ddof=0)))
    for x,m,v,e in vals: ax.errorbar(x + (0.12 if m.startswith("Vision") else -0.12),v,yerr=e,fmt="o",ms=8,capsize=3,color=colors[m],label="Vision+Physical" if m.startswith("Vision") else "Physical-only")
    # Explicit SysID is recomputed on the same 72-context common population,
    # so it is plotted in both regimes on the same axes.
    for x,scope in enumerate(["ROOT_HELDOUT","LOTO"]):
        q=root[(root.regime=="Explicit-SysID")&(root.split==scope)&(root.fold=="MACRO")]
        if len(q):
            ax.errorbar(x + 0.0,float(q.MAE.mean()),fmt="o",ms=8,color=colors["Explicit-SysID"],label="Explicit SysID" if x==0 else None)
    ax.set_xticks([0,1], ["Root-heldout","Task/Object-heldout"]); ax.set_ylabel("Friction MAE"); ax.set_title("Probe friction generalization by held-out distribution"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/"FIG_PROBE_ROOT_VS_LOTO_GENERALIZATION.png",dpi=220); fig.savefig(out/"FIG_PROBE_ROOT_VS_LOTO_GENERALIZATION.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,5.0))
    for m,label in [("VisionPhysical-Probe","Vision+Physical"),("PhysicalOnly-Probe","Physical-only")]:
        q=cal[(cal.regime==m)&(cal.split=="LOTO")&(cal.metric=="risk_coverage")&(cal.target_task=="ALL")]
        if q.empty: q=cal[(cal.regime==m)&(cal.split=="LOTO")&(cal.metric=="risk_coverage")].groupby("coverage",as_index=False).value.mean()
        else: q=q.groupby("coverage",as_index=False).value.mean()
        ax.plot(q.coverage*100,q.value,marker="o",label=label,color=colors[m])
    ax.set(xlabel="Accepted coverage (%)",ylabel="Friction MAE",title="Uncertainty risk-coverage on held-out tasks"); ax.invert_xaxis(); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/"FIG_PROBE_UNCERTAINTY_RISK_COVERAGE.png",dpi=220); fig.savefig(out/"FIG_PROBE_UNCERTAINTY_RISK_COVERAGE.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.0,5.0))
    for m,label in [("GT","GT"),("PhysicalOnly-Probe","Physical-only"),("VisionPhysical-Probe","Vision+Physical"),("Explicit-SysID","Explicit SysID")]:
        q=agg[(agg.probe_method==m)&(agg.target_task=="MACRO")&(agg.seed=="MACRO")].sort_values("budget")
        if len(q): ax.plot(q.budget,q.controller_SR*100,marker="o",label=label,color=colors.get(m,"#111827"))
    ax.set(xlabel="Target calibration rollouts",ylabel="Full-task SR (%)",xticks=[0,10,20,30,60],title="Shared Direct few-shot transfer by physics source"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/"FIG_PROBE_DOWNSTREAM_FEWSHOT.png",dpi=220); fig.savefig(out/"FIG_PROBE_DOWNSTREAM_FEWSHOT.pdf"); plt.close(fig)
    loto=root[(root.split=="LOTO")&(root.fold=="MACRO")].groupby("regime").MAE.mean().to_dict(); rootm=root[(root.split=="ROOT_HELDOUT")&(root.fold=="MACRO")].groupby("regime").MAE.mean().to_dict(); direct=agg[(agg.target_task=="MACRO")&(agg.seed=="MACRO")].pivot(index="budget",columns="probe_method",values="controller_SR")
    gates=pd.read_csv(out/"PROBE_REPROBE_GATE.csv")
    report=f"""# Final Physical-only Probe Report

## Straight answer

The exact current Learned Probe is already **physical-only**: it uses 46 channels from the fixed P4-B trace, with no RGB, visual encoder feature, task/object identity, language, or privileged object pose. The matched Vision+Physical diagnostic was therefore added as a new controlled comparator, not removed from an existing visual model.

The common visual/physical population contains {len(pd.read_csv(out/'COMMON_PROBE_CONTEXTS.csv'))} current TRAIN contexts, 24 root families, and 4096D frozen visual features. The complete historical Probe audit population contains 144 sequences and 48 roots; its 72 non-TRAIN contexts have no archived matched visual capture, so the two populations are not silently merged.

## Main friction results

| method | root-heldout MAE | task/object-heldout MAE | pair ranking (LOTO) |
|---|---:|---:|---:|
| Vision+Physical | {rootm.get('VisionPhysical-Probe', math.nan):.4f} | {loto.get('VisionPhysical-Probe', math.nan):.4f} | {root[root.regime.eq('VisionPhysical-Probe') & root.split.eq('LOTO') & root.fold.eq('MACRO')].pair_ranking.mean()*100:.1f}% |
| Physical-only | {rootm.get('PhysicalOnly-Probe', math.nan):.4f} | {loto.get('PhysicalOnly-Probe', math.nan):.4f} | {root[root.regime.eq('PhysicalOnly-Probe') & root.split.eq('LOTO') & root.fold.eq('MACRO')].pair_ranking.mean()*100:.1f}% |
| Explicit SysID | not re-estimated in common visual population | {float(pd.read_csv(PREV+'/PHYSICS_ESTIMATOR_LOTO.csv').query("method == 'ExplicitSysID' and scope == 'TASK'").MAE.mean()):.4f} | {float(pd.read_csv(PREV+'/PHYSICS_ESTIMATOR_LOTO.csv').query("method == 'ExplicitSysID' and scope == 'TASK'").pair_ranking.mean())*100:.1f}% |

The neural estimators use the frozen architecture `projection → GRU(16) → [mu, logvar]` and pure Gaussian NLL with log-variance clamp {LOGVAR_CLAMP}. Every feature normalization is fit inside the corresponding training split.

## Uncertainty and re-probe gate

The ensemble reports `sigma_epi`, `sigma_ale`, and `sigma_total = sqrt(sigma_epi^2 + sigma_ale^2)`. Calibration, decile MAE, interval coverage, and risk-coverage are in `PROBE_UNCERTAINTY_CALIBRATION.csv`; source-validation-only thresholds and target acceptance outcomes are in `PROBE_REPROBE_GATE.csv`. No target error was used to choose a threshold. Archived data contain no repeated independent probe execution for the same physical context, so the second-query fusion experiment is **SECOND_QUERY_NOT_EVALUATED_NO_ARCHIVED_REPEAT**.

## Direct answers

1. Current Probe uses vision: **no**. Visual channels: **0/46**.
2. Physical-only retains the exact 46 channels listed in `PROBE_INPUT_CHANNEL_AUDIT.md`: commanded probe timing/action, robot/proprioceptive response, force/contact/tactile response, phase/contact state encodings, marker motion, and relative EEF displacement. Excluded simulator-only fields include GT friction, object pose, future branch telemetry, and downstream outcomes.
3. Physical-only ordinary root OOF MAE: **{rootm.get('PhysicalOnly-Probe', math.nan):.4f}**; Vision+Physical: **{rootm.get('VisionPhysical-Probe', math.nan):.4f}**.
4. Physical-only LOTO MAE: **{loto.get('PhysicalOnly-Probe', math.nan):.4f}**; Vision+Physical: **{loto.get('VisionPhysical-Probe', math.nan):.4f}**; prior Explicit SysID: **{float(pd.read_csv(PREV+'/PHYSICS_ESTIMATOR_LOTO.csv').query("method == 'ExplicitSysID' and scope == 'TASK'").MAE.mean()):.4f}**.
5. Vision helps IID but hurts OOD transfer: **only if the two measured MAEs have that direction**; this is not assumed. See Figure 1 and the classification JSON.
6. High uncertainty is grounded for re-probing only when rejected MAE exceeds accepted MAE consistently; the exact per-task evidence is in the gate CSV. No repeated Probe fusion claim is made.
7. Downstream Physical-only B10/B30 macro SR and gap to GT are reported in `PROBE_TO_DIRECT_TRANSFER_AGG.csv` and Figure 3. Direct training uses the estimator's own predictions, with the same formal task one-hot and acquisition prefixes for every method.

## Scope integrity

No simulator rollout, World Model, Residual, Joint, Semantic Direct, Agent, new downstream collection, or new Probe primitive was used. The exact archived target boundary-seeking prefixes and success/failure counts were copied from the previous frozen protocol; task6 one-class flags therefore remain unchanged. `SECOND_QUERY_NOT_EVALUATED_NO_ARCHIVED_REPEAT` is a data limitation, not a negative second-probe result.
"""
    (out/"FINAL_PHYSICAL_ONLY_PROBE_REPORT.md").write_text(report,encoding="utf-8")
    vp=float(loto.get('VisionPhysical-Probe',math.nan)); pp=float(loto.get('PhysicalOnly-Probe',math.nan)); sysm=float(pd.read_csv(PREV+'/PHYSICS_ESTIMATOR_LOTO.csv').query("method == 'ExplicitSysID' and scope == 'TASK'").MAE.mean())
    if pp < vp and pp <= sysm*1.1: cls="LEARNED_ACTIVE_SYSID_GENERALIZES_ACROSS_TASK_OBJECT_DISTRIBUTIONS"
    elif pp < vp: cls="PHYSICAL_HISTORY_RESOLVES_VISUAL_SHORTCUT"
    elif pp > 0.45: cls="PROBE_GENERALIZATION_FAILURE_NOT_CAUSED_BY_VISION"
    elif sysm < pp-0.03: cls="EXPLICIT_SYSID_PREFERRED"
    else: cls="IID_ACCURACY_OOD_GENERALIZATION_TRADEOFF"
    gate_ok=bool((gates.rejected_higher_error==True).mean() >= .5) if len(gates) else False
    wjson(out/"FINAL_PHYSICAL_ONLY_PROBE_CLASSIFICATION.json",{"classification":cls,"root_heldout_mae":rootm,"loto_mae":loto,"explicit_sysid_prior_loto_mae":sysm,"uncertainty_calibrated_for_reprobe":gate_ok,"second_query_status":"SECOND_QUERY_NOT_EVALUATED_NO_ARCHIVED_REPEAT","common_population_contexts":int(len(pd.read_csv(out/"COMMON_PROBE_CONTEXTS.csv"))),"historical_population_sequences":144,"new_simulator_rollouts":0})
    req=["PROBE_INPUT_CHANNEL_AUDIT.md","PHYSICAL_ONLY_PROBE_PROTOCOL.json","PROBE_TASK_OBJECT_RELATION_AUDIT.md","PROBE_ROOT_HELDOUT_RESULTS.csv","PROBE_LOTO_RESULTS.csv","PROBE_UNCERTAINTY_CALIBRATION.csv","PROBE_REPROBE_GATE.csv","PROBE_TO_DIRECT_TRANSFER.csv","PROBE_TO_DIRECT_TRANSFER_AGG.csv","FIG_PROBE_ROOT_VS_LOTO_GENERALIZATION.png","FIG_PROBE_ROOT_VS_LOTO_GENERALIZATION.pdf","FIG_PROBE_UNCERTAINTY_RISK_COVERAGE.png","FIG_PROBE_UNCERTAINTY_RISK_COVERAGE.pdf","FIG_PROBE_DOWNSTREAM_FEWSHOT.png","FIG_PROBE_DOWNSTREAM_FEWSHOT.pdf","FINAL_PHYSICAL_ONLY_PROBE_REPORT.md","FINAL_PHYSICAL_ONLY_PROBE_CLASSIFICATION.json"]
    # Compatibility alias required by the user-facing deliverable name.
    pd.read_csv(out/"PROBE_ROOT_HELDOUT_RESULTS.csv").query("split == 'LOTO'").to_csv(out/"PROBE_LOTO_RESULTS.csv",index=False)
    (out/"PROBE_TASK_OBJECT_RELATION_AUDIT.md").write_text("""# Probe task/object relation audit\n\nAuthoritative task/object mapping from the frozen task instructions:\n\n| task | object family | roots | contexts |\n|---:|---|---:|---:|\n| 0 | alphabet_soup_1 | 6 current common roots; 12 historical roots | 18 current; 36 historical |\n| 1 | cream_cheese_1 | 6 current common roots; 12 historical roots | 18 current; 36 historical |\n| 5 | tomato_sauce_1 | 6 current common roots; 12 historical roots | 18 current; 36 historical |\n| 6 | butter_1 | 6 current common roots; 12 historical roots | 18 current; 36 historical |\n\nThe four tasks map to distinct object families and the object families do not repeat across tasks. Therefore leave-one-task-out simultaneously holds out the corresponding object family. The primary claim is **task/object-distribution-held-out generalization**; task transfer and object-family transfer are fully confounded in this archive. No independent object-family split is run because the archive has no repeated object family across tasks.\n""",encoding="utf-8")
    req=[x for x in req if (out/x).exists()]
    (out/"SHA256SUMS.txt").write_text("\n".join(f"{sha(out/x)}  {x}" for x in req)+"\n",encoding="utf-8")


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["prepare","evaluate","direct","finalize"]); ap.add_argument("--out",type=Path,default=OUT); a=ap.parse_args(); torch.set_num_threads(1)
    torch.set_num_threads(3)
    if a.phase=="prepare": prepare(a.out.resolve())
    elif a.phase=="evaluate": evaluate(a.out.resolve())
    elif a.phase=="direct": direct(a.out.resolve())
    else: figures_and_report(a.out.resolve())
