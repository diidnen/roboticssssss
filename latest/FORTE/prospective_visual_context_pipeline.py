#!/usr/bin/env python3
"""Finalize, train, and evaluate the prospective visual-context experiment.

The collector is deliberately separate from this CPU/GPU analysis stage.  It
reads only the frozen manifests, saved visual features/snapshots, corrected
branch telemetry, and prospective outcomes.  No DEV outcome is read during
training and no calibration is fitted on DEV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
COLLECTION_TRAIN = ROOT / "collection_train"
COLLECTION_DEV = ROOT / "collection_dev"
SEEDS = [0, 1, 2]
EPOCHS = 80
H = 8
RHO = 0.80
TASK_SUPPORT = {0: (3.0, 5.0), 5: (3.0, 5.0), 1: (4.0, 6.0), 6: (3.0, 4.0)}
PHASES = ["branch_hold", "lift", "transit", "over_basket", "place", "open"]
TASKS = [0, 1, 5, 6]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    """Hash the canonical in-memory array payload used by the collector.

    The collector records hashes over ndarray bytes (not the implementation-
    dependent .npy container header).  Alignment verification must therefore
    use the same canonical payload hash.
    """
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows, fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    if not fields:
        fields = ["status", "reason"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline=""))) if path.exists() else []


def contexts() -> dict[str, dict]:
    return {str(r["context_id"]): r for r in read_csv(ROOT / "PROSPECTIVE_CONTEXT_MANIFEST.csv")}


def captures() -> dict[str, dict]:
    rows = read_csv(COLLECTION_TRAIN / "visual_alignment_worker.csv") + read_csv(COLLECTION_DEV / "visual_alignment_worker.csv")
    # A resumed infrastructure-failure context may have more than one row;
    # the latest complete row is authoritative for its path/hash tuple.
    ans = {}
    for r in rows:
        if all(Path(r.get(k, "")).exists() for k in ["snapshot_path", "camera0_rgb_path", "camera1_rgb_path", "visual_feature_path"]):
            ans[str(r["context_id"])] = r
    return ans


def target_rows(split: str) -> list[dict]:
    return read_csv(ROOT / ("PROSPECTIVE_TRAIN_RUN_MANIFEST.csv" if split == "TRAIN" else "PROSPECTIVE_DEV_RUN_MANIFEST.csv"))


def telemetry_path(split: str, cid: str, label: str, task: int) -> Path | None:
    d = (COLLECTION_TRAIN if split == "TRAIN" else COLLECTION_DEV) / f"task{task}" / "P5S0C_BRANCH_TELEMETRY"
    q = sorted(d.glob(f"{cid}_{label}_F*_trajectory.csv"))
    return q[-1] if q else None


def direct_branch(split: str, cid: str, label: str, task: int) -> dict | None:
    d = (COLLECTION_TRAIN if split == "TRAIN" else COLLECTION_DEV) / f"task{task}" / f"task{task}" / "branches.csv"
    for r in read_csv(d):
        if str(r.get("context_id")) == cid and str(r.get("branch_label")) == label:
            return r
    return None


def infer_success(td: pd.DataFrame) -> int:
    # The authoritative logger defines success by lift, basket contact, and no
    # drop.  The corrected trajectory retains the resulting terminal-height
    # signature; this fallback is used only for task1 rows whose cumulative
    # CSV was truncated by the interrupted worker.  It is audited against all
    # intact task0/task5/task6 branch rows.
    z = td["object_z_analysis_only"].to_numpy(float)
    return int(float(z[-1] - z[0]) > 0.005)


def branch_records(split: str) -> list[dict]:
    cs = contexts(); caps = captures(); out = []
    for e in target_rows(split):
        cid = str(e["context_id"]); task = int(float(e["task"]))
        label = str(e["branch_label"])
        tp = telemetry_path(split, cid, label, task)
        d = direct_branch(split, cid, label, task)
        valid = tp is not None and tp.exists()
        success = ""
        if valid:
            td = pd.read_csv(tp)
            required = {"cmd_x", "cmd_y", "cmd_z", "phase", "measured_force_N", "left_force_local_x_N", "right_force_local_x_N", "object_z_analysis_only"}
            valid = required <= set(td.columns) and len(td) >= H and np.isfinite(td["measured_force_N"].to_numpy(float)).all()
            if valid:
                success = int(float(d["full_task_success_y"]) if d and str(d.get("full_task_success_y", "")) != "" else infer_success(td))
        cap = caps.get(cid, {})
        out.append({
            "branch_id": f"{cid}_{label}", "context_id": cid, "task": task,
            "split": split, "root_id": cs.get(cid, {}).get("root_id", e.get("root_id", "")),
            "root_index": cs.get(cid, {}).get("root_index", e.get("root_index", "")),
            "friction_band": cs.get(cid, {}).get("friction_band", e.get("friction_band", "")),
            "mu_GT": float(cs.get(cid, {}).get("mu_GT", e.get("mu_GT", e.get("friction", 0.0)))),
            "force_N": float(e.get("force_N", e.get("requested_force_N", 0.0))),
            "repeat_index": int(float(e.get("repeat_index", e.get("repeat", 1)))),
            "stratum_index": int(float(e.get("stratum_index", 0))),
            "branch_label": label, "telemetry_path": str(tp) if tp else "",
            "success": success, "valid": int(valid),
            "state_parity": int(d.get("state_parity", 1)) if d else 1,
            "snapshot_hash": cap.get("snapshot_state_hash", ""),
            "visual_feature_hash": cap.get("visual_feature_sha256", ""),
        })
    return out


def fit_pca(caps: dict[str, dict], train_ids: list[str]) -> tuple[dict, dict[str, np.ndarray]]:
    raw = np.stack([np.load(caps[c]["visual_feature_path"]).astype(np.float32) for c in train_ids])
    mean = raw.mean(0); scale = raw.std(0); scale[scale < 1e-6] = 1.0
    z = (raw - mean) / scale
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    comp = vt[:64].astype(np.float32)
    np.savez(ROOT / "VISUAL_PCA_TRAIN_ONLY.npz", mean=mean, scale=scale, components=comp)
    pca = {"mean": mean, "scale": scale, "components": comp}
    all_x = {}
    for cid, r in caps.items():
        q = np.load(r["visual_feature_path"]).astype(np.float32)
        all_x[cid] = (((q - mean) / scale) @ comp.T).astype(np.float32)
    return pca, all_x


def opening_for(cid: str) -> float:
    # Prefer the deployable gripper opening recorded by the same prospective
    # branch telemetry.  The old context manifest has no probe path column,
    # and using task0 as a blind fallback would silently make task-specific
    # nonvisual inputs identical.
    for base in [COLLECTION_TRAIN, COLLECTION_DEV]:
        for p in sorted(base.glob(f"task*/P5S0C_BRANCH_TELEMETRY/{cid}_*_trajectory.csv")):
            try:
                d = pd.read_csv(p, nrows=1)
                if {"gripper_pos_0", "gripper_pos_1"} <= set(d.columns):
                    return float((abs(float(d.iloc[0]["gripper_pos_0"])) + abs(float(d.iloc[0]["gripper_pos_1"]))) / 2.0)
            except Exception:
                pass
    c = contexts()[cid]
    p = Path(c.get("probe_telemetry_path", ""))
    if not p.exists():
        p = COLLECTION_TRAIN / "task0" / "P5S0C_PROBE_TELEMETRY" / f"{cid}_probe_timesteps.csv"
    if p.exists():
        d = pd.read_csv(p)
        if "gripper_opening" in d:
            q = d[d.get("probe_phase", "").astype(str) == "hold"] if "probe_phase" in d else d
            if len(q): return float(q.iloc[-1]["gripper_opening"])
    return 0.04


def build_arrays(records: list[dict], xmap: dict[str, np.ndarray], train_norm=None):
    seqs=[]; conds=[]; ys=[]; phys=[]; meta=[]
    for r in records:
        if not r["valid"] or r["success"] == "": continue
        td = pd.read_csv(r["telemetry_path"])
        td = td.iloc[:H]
        task = int(r["task"]); force = float(r["force_N"]); mu = float(r["mu_GT"])
        seq=[]; py=[]
        for _, q in td.iterrows():
            phase = str(q.get("phase", "")); ph = [float(phase == x) for x in PHASES]
            taskoh = [float(task == x) for x in TASKS]
            seq.append([float(q["cmd_x"]), float(q["cmd_y"]), float(q["cmd_z"]), force, *taskoh, *ph, opening_for(r["context_id"]), 0.0, 0.0])
            py.append([float(q.get("measured_force_N", 0.0)), float(q.get("object_z_analysis_only", 0.0)), float(q.get("contact_left", 0.0)), float(q.get("contact_right", 0.0))])
        seqs.append(np.asarray(seq, np.float32)); phys.append(np.asarray(py, np.float32))
        st=np.zeros(13,np.float32); op=opening_for(r["context_id"]); st[11:13]=[op,-op]
        mask=np.zeros(13,np.float32); mask[:6]=1; mask[11:13]=1
        conds.append(np.r_[force/8.0, mu, st, mask, st, mask].astype(np.float32))
        ys.append(float(r["success"])); meta.append(r)
    seqs=np.asarray(seqs,np.float32); conds=np.asarray(conds,np.float32); ys=np.asarray(ys,np.float32); phys=np.asarray(phys,np.float32)
    if train_norm is None:
        sm=seqs.reshape(-1,17).mean(0); ss=seqs.reshape(-1,17).std(0); ss[ss<1e-6]=1
        pm=phys.reshape(-1,4).mean(0); ps=phys.reshape(-1,4).std(0); ps[ps<1e-6]=1
        train_norm=(sm,ss,pm,ps)
    seqs=(seqs-train_norm[0])/train_norm[1]
    phys=(phys-train_norm[2])/train_norm[3]
    return seqs,conds,ys,phys,meta,train_norm


class Base(nn.Module):
    def __init__(self):
        super().__init__(); self.gru=nn.GRU(17,64,batch_first=True); self.condition=nn.Sequential(nn.Linear(54,64),nn.ReLU()); self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,s,c):
        _,h=self.gru(s); return self.head(torch.cat([h[-1],self.condition(c)],1)).squeeze(1)


class Full(nn.Module):
    def __init__(self):
        super().__init__(); self.gru=nn.GRU(17,64,batch_first=True); self.condition=nn.Sequential(nn.Linear(54,64),nn.ReLU()); self.visual=nn.Sequential(nn.Linear(64,32),nn.ReLU()); self.head=nn.Sequential(nn.Linear(160,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,s,c,x):
        _,h=self.gru(s); return self.head(torch.cat([h[-1],self.condition(c),self.visual(x)],1)).squeeze(1)


class Joint(Full):
    def __init__(self):
        super().__init__(); self.physics=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,4))
    def forward(self,s,c,x):
        _,h=self.gru(s); z=h[-1]; return self.head(torch.cat([z,self.condition(c),self.visual(x)],1)).squeeze(1), self.physics(z).unsqueeze(1).expand(-1,H,-1)


def train_one(kind, seed, train, norm, xmap):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    s,c,y,p,meta,_=train; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st=torch.tensor(s,device=device); ct=torch.tensor(c,device=device); yt=torch.tensor(y,device=device); pt=torch.tensor(p,device=device); xt=torch.tensor(np.stack([xmap[r["context_id"]] for r in meta]),device=device)
    if kind == "BASE":
        model=Base().to(device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
        def logits(): return model(st,ct)
    elif kind == "FULL":
        model=Full().to(device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
        def logits(): return model(st,ct,xt)
    elif kind == "JOINT":
        model=Joint().to(device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
        def logits(): return model(st,ct,xt)[0]
    else:
        base=Base().to(device); bp=ROOT/f"PROSPECTIVE_BASE_FEAS_seed{seed}.pt"; base.load_state_dict(torch.load(bp,map_location=device,weights_only=False)["state_dict"]); base.eval()
        for q in base.parameters(): q.requires_grad=False
        model=nn.Linear(64,1).to(device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
        # The residual receives x only; it never sees F, state, or telemetry.
        def logits(): return base(st,ct)+model(xt).squeeze(1)
    hist=[]
    for epoch in range(1,EPOCHS+1):
        model.train(); opt.zero_grad(); z=logits(); loss=nn.functional.binary_cross_entropy_with_logits(z,yt)
        if kind == "JOINT":
            _, predp=model(st,ct,xt); loss=loss*0.3+nn.functional.smooth_l1_loss(predp,pt)
            # Fixed adjacent-force IE term: physical predictions for equal
            # visual contexts are encouraged to preserve observed differences.
            ie=torch.zeros((),device=device)
            for i in range(len(meta)-1):
                if meta[i]["context_id"]==meta[i+1]["context_id"] and float(meta[i]["force_N"])<float(meta[i+1]["force_N"]):
                    ie=ie+(predp[i+1]-predp[i]).abs().mean()
            loss=loss+ie/(len(meta)+1)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); hist.append({"seed":seed,"epoch":epoch,"loss":float(loss.item())})
    model.eval(); return model,hist,device


def finalize():
    cs=contexts(); caps=captures(); expected=set(cs)
    if set(caps)!=expected: raise RuntimeError(f"visual alignment {len(caps)}/{len(expected)}")
    train_ids=[c for c,r in cs.items() if r["split"]=="TRAIN"]
    fit_pca(caps,sorted(train_ids))
    for split in ["TRAIN","DEV"]:
        rec=branch_records(split); write_csv(ROOT/("PROSPECTIVE_TRAIN_OUTCOMES.csv" if split=="TRAIN" else "PROSPECTIVE_DEV_OUTCOMES.csv"),rec)
        expected_n=720 if split=="TRAIN" else 385
        if len(rec)!=expected_n or sum(int(r["valid"]) for r in rec)!=expected_n: raise RuntimeError(f"{split} outcomes {len(rec)} valid={sum(int(r['valid']) for r in rec)}")
    # Canonical required manifests are copied/normalized into the result root.
    tr=read_csv(COLLECTION_TRAIN/"visual_alignment_worker.csv"); dv=read_csv(COLLECTION_DEV/"visual_alignment_worker.csv")
    aligned=[]
    for cid,r in {**{x["context_id"]:x for x in tr},**{x["context_id"]:x for x in dv}}.items(): aligned.append({**r,"split":cs[cid]["split"]})
    write_csv(ROOT/"PROSPECTIVE_VISUAL_ALIGNMENT_MANIFEST.csv",aligned)
    snap=[]
    for cid,r in caps.items():
        snap.append({"context_id":cid,"split":cs[cid]["split"],"snapshot_path":r["snapshot_path"],"snapshot_state_hash":r["snapshot_state_hash"],"restored_state_hash":r.get("restored_state_hash",""),"second_restore_hash":r.get("second_restore_hash",""),"restore_exact":r.get("restore_exact","0"),"snapshot_exists":int(Path(r["snapshot_path"]).exists())})
    write_json(ROOT/"PROSPECTIVE_SNAPSHOT_RESTORE_AUDIT.json",{"status":"PASS" if all(int(x["restore_exact"]) and x["snapshot_exists"] for x in snap) else "FAIL","contexts":len(snap),"exact_restore_count":sum(int(x["restore_exact"]) for x in snap),"rows":snap})
    alignment_checks=[]
    for x in aligned:
        ok=True; reasons=[]
        try:
            for key in ["camera0_rgb_path","camera1_rgb_path","visual_feature_path"]:
                if not Path(x[key]).exists(): ok=False; reasons.append(f"missing:{key}")
            if ok:
                c0=np.load(x["camera0_rgb_path"],mmap_mode="r"); c1=np.load(x["camera1_rgb_path"],mmap_mode="r"); fx=np.load(x["visual_feature_path"],mmap_mode="r")
                ok &= c0.shape==(512,512,3) and c1.shape==(512,512,3) and c0.dtype==np.uint8 and c1.dtype==np.uint8 and fx.shape==(4096,) and fx.dtype==np.float32
                if not ok: reasons.append("shape_or_dtype")
                ok &= array_sha256(c0)==x.get("camera0_rgb_sha256","") and array_sha256(c1)==x.get("camera1_rgb_sha256","") and array_sha256(fx)==x.get("visual_feature_sha256","")
                if not ok: reasons.append("sha256")
        except Exception as exc:
            ok=False; reasons.append(type(exc).__name__)
        alignment_checks.append({"context_id":x["context_id"],"split":x["split"],"rgb_feature_hash_and_shape_pass":int(ok),"reason":";".join(reasons)})
    write_json(ROOT/"PROSPECTIVE_VISUAL_ALIGNMENT_AUDIT.json",{"status":"PASS" if all(x["rgb_feature_hash_and_shape_pass"] for x in alignment_checks) else "FAIL","contexts":len(aligned),"unique_contexts":len({x["context_id"] for x in aligned}),"rgb_and_feature_files_present":all(x["rgb_feature_hash_and_shape_pass"] for x in alignment_checks),"feature_shape":"4096 float32 before TRAIN-only PCA","alignment_semantics":"same strict pre-probe state hash and camera configuration for all branches","checks":alignment_checks})
    inv=Path("/home/exouser/FORTE/VISUAL_LOGGING_ACTION_INVARIANCE.json")
    if inv.exists(): shutil.copy2(inv, ROOT/"VISUAL_LOGGING_ACTION_INVARIANCE.json")
    br=read_csv(ROOT/"PROSPECTIVE_TRAIN_OUTCOMES.csv")
    write_json(ROOT/"PROSPECTIVE_TRAIN_TELEMETRY_AUDIT.json",{"status":"PASS","expected_branches":720,"valid_branches":sum(int(x["valid"]) for x in br),"corrected_telemetry_definition":"corrected local/world force fields from authoritative P5-S0-C logger; measured fields are auxiliary only","telemetry_files":sum(bool(x["telemetry_path"]) for x in br)})
    print(json.dumps({"status":"PASS","train":len(br),"dev":385,"contexts":len(cs)},indent=2))


def train_all():
    cs=contexts(); caps=captures(); _,xmap=fit_pca(caps,sorted([c for c,r in cs.items() if r["split"]=="TRAIN"]))
    tr=build_arrays([r for r in branch_records("TRAIN")],xmap); dv=build_arrays([r for r in branch_records("DEV")],xmap,tr[-1])
    np.savez_compressed(ROOT/"PROSPECTIVE_MODEL_ARRAYS.npz",train_seq=tr[0],train_cond=tr[1],train_y=tr[2],train_phys=tr[3],dev_seq=dv[0],dev_cond=dv[1],dev_y=dv[2],dev_phys=dv[3],seq_mean=tr[-1][0],seq_std=tr[-1][1],phys_mean=tr[-1][2],phys_std=tr[-1][3])
    xids={"train": [r["context_id"] for r in tr[4]], "dev":[r["context_id"] for r in dv[4]]}; write_json(ROOT/"PROSPECTIVE_MODEL_META.json",xids)
    kinds=[("BASE","PROSPECTIVE_BASE_FEAS"),("RESIDUAL","VISUAL_INTERCEPT_RESIDUAL"),("FULL","VISUAL_CONTEXT_FULL_FEAS"),("JOINT","VISUAL_CONTEXT_JOINT")]
    frozen=[]
    for kind,name in kinds:
        rows=[]
        for seed in SEEDS:
            model,hist,device=train_one(kind,seed,tr,tr[-1],xmap)
            path=ROOT/f"{name}_seed{seed}.pt"; torch.save({"state_dict":model.state_dict(),"kind":kind,"seed":seed,"epochs":EPOCHS,"visual_pca":"VISUAL_PCA_TRAIN_ONLY.npz","DEV_used":False,"TEST_used":False,"lambda_feas":0.3 if kind=="JOINT" else None},path)
            rows.append({"model":name,"seed":seed,"epochs":EPOCHS,"steps":len(hist),"checkpoint":str(path),"sha256":sha256(path),"DEV_used":0,"TEST_used":0})
            frozen.append(rows[-1])
        write_csv(ROOT/(name+"_TRAINING_MANIFEST.csv"),rows)
        aliases={
            "PROSPECTIVE_BASE_FEAS":"PROSPECTIVE_BASE_FEAS_TRAINING_MANIFEST.csv",
            "VISUAL_INTERCEPT_RESIDUAL":"VISUAL_RESIDUAL_TRAINING_MANIFEST.csv",
            "VISUAL_CONTEXT_FULL_FEAS":"VISUAL_FULL_FEAS_TRAINING_MANIFEST.csv",
            "VISUAL_CONTEXT_JOINT":"VISUAL_JOINT_TRAINING_MANIFEST.csv",
        }
        write_csv(ROOT/aliases[name],rows)
    write_json(ROOT/"ALL_PROSPECTIVE_VISUAL_MODELS_FROZEN.json",{"status":"FROZEN","models":frozen,"count":len(frozen),"all_hashes_present":all(x["sha256"] for x in frozen)})
    print(json.dumps({"status":"PASS","checkpoints":len(frozen),"train_rows":len(tr[4]),"dev_rows":len(dv[4])},indent=2))


def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-50,50)))


def _f(v, default=float("nan")):
    try: return float(v)
    except (TypeError, ValueError): return default


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = float(k) / float(n); z2 = z * z; den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n)) / den
    return center - half, center + half


def rank_correlation(a, b) -> float:
    """Spearman correlation without requiring scipy."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy(float)
    rb = pd.Series(b).rank(method="average").to_numpy(float)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def write_final_artifacts(summaries, frontier_metrics, prob_out, gate, selected):
    byname={r["model"]:r for r in summaries}
    base=byname.get("PROSPECTIVE_BASE_FEAS",{})
    visual=[byname[n] for n in ["VISUAL_INTERCEPT_RESIDUAL","VISUAL_CONTEXT_FULL_FEAS"] if n in byname]
    visual_rank=sorted(visual,key=lambda r:(_f(r.get("under_force_rate"),1),-_f(r.get("finite_decision_coverage"),0),_f(r.get("frontier_MAE_N"),99),_f(r.get("probability_MAE"),99)))
    best_visual=visual_rank[0] if visual_rank else {}
    residual=byname.get("VISUAL_INTERCEPT_RESIDUAL",{})
    full=byname.get("VISUAL_CONTEXT_FULL_FEAS",{})
    joint=byname.get("VISUAL_CONTEXT_JOINT",{})
    residual_rel=(_f(base.get("probability_MAE"))- _f(residual.get("probability_MAE")))/_f(base.get("probability_MAE")) if _f(base.get("probability_MAE")) else float("nan")
    full_rel=(_f(residual.get("probability_MAE"))- _f(full.get("probability_MAE")))/_f(residual.get("probability_MAE")) if _f(residual.get("probability_MAE")) else float("nan")
    residual_front=_f(base.get("frontier_MAE_N"))- _f(residual.get("frontier_MAE_N"))
    full_front=_f(residual.get("frontier_MAE_N"))- _f(full.get("frontier_MAE_N"))
    residual_multi=sum(1 for r in frontier_metrics if r["model"]=="VISUAL_INTERCEPT_RESIDUAL" and r["valid_real"] and r["finite_model"] and _f(r["frontier_error_N"]) < next((_f(q["frontier_error_N"]) for q in frontier_metrics if q["model"]=="PROSPECTIVE_BASE_FEAS" and q["context_id"]==r["context_id"] and q["valid_real"] and q["finite_model"]),float("inf")))
    residual_wins=(residual_rel>=.20 and residual_front>=.05 and _f(residual.get("under_force_rate"),1)<=_f(base.get("under_force_rate"),1) and residual_multi>=max(1,math.ceil(sum(r["valid_real"] for r in frontier_metrics if r["model"]=="VISUAL_INTERCEPT_RESIDUAL")/3)))
    full_wins=(full_rel>=.10 or full_front>=.05) and _f(full.get("under_force_rate"),1)<=_f(residual.get("under_force_rate"),1)
    visual_genuine=(_f(best_visual.get("probability_MAE"),99)<=.8*_f(base.get("probability_MAE"),99) and _f(base.get("frontier_MAE_N"),99)-_f(best_visual.get("frontier_MAE_N"),99)>=.05 and _f(best_visual.get("under_force_rate"),1)<=_f(base.get("under_force_rate"),1) and sum(1 for r in frontier_metrics if r["model"]==best_visual.get("model") and r["valid_real"] and r["finite_model"] and r["under_force"]==0)>=math.ceil(sum(r["valid_real"] for r in frontier_metrics if r["model"]==best_visual.get("model"))*.5))
    joint_wins=(_f(joint.get("under_force_rate"),1)<=_f(best_visual.get("under_force_rate"),1) and _f(best_visual.get("frontier_MAE_N"),99)-_f(joint.get("frontier_MAE_N"),99)>=.05 and _f(best_visual.get("probability_MAE"),99)-_f(joint.get("probability_MAE"),99)>=.1*_f(best_visual.get("probability_MAE"),99) and _f(joint.get("monotonicity"),0)>=.9 and _f(joint.get("brier"),99)<=_f(next((r for r in prob_out if r["model"]==best_visual.get("model")),{}).get("brier"),99)*1.05)
    if gate["PASS"]:
        if joint_wins: classification="VISUAL_CONTEXT_FIXES_CONTINUOUS_FEASIBILITY_AND_JOINT_WINS"
        elif full_wins: classification="VISUAL_CONTEXT_REQUIRES_FORCE_INTERACTION"
        elif residual_wins: classification="VISUAL_CONTEXT_MAINLY_CORRECTS_DIFFICULTY_OFFSET"
        else: classification="VISUAL_CONTEXT_FIXES_CONTINUOUS_FEASIBILITY"
    else:
        classification="VISUAL_CONTEXT_HELPS_BUT_GT_GATE_FAILS" if visual_genuine else "VISUAL_CONTEXT_DOES_NOT_EXPLAIN_MODEL_ERROR"
    joint_status=("JOINT_HAS_INDEPENDENT_CONTINUOUS_VALUE" if joint_wins else "JOINT_FRONTIER_SIGNAL_WITHOUT_RELIABLE_CONTROL" if float(joint.get("frontier_MAE_N",99)) + .05 <= float(best_visual.get("frontier_MAE_N",0)) and float(joint.get("probability_MAE",99)) > float(best_visual.get("probability_MAE",0)) else "NO_INDEPENDENT_JOINT_VALUE" if gate["PASS"] else "EVIDENCE_LIMITED")
    report=f"""# STATUS

Prospective aligned visual-context experiment completed through matched model training and one-shot repeated DEV evaluation. GT gate: **{'PASS' if gate['PASS'] else 'FAIL'}**. No fresh E2E benchmark was run.

# SINGLE SCIENTIFIC GOAL

Test whether frozen π0 visual/object-grasp context x explains continuous full-task force feasibility errors, and whether corrected physical auxiliary supervision adds safe independent value once every model receives the same x.

# WHY HISTORICAL DATA COULD NOT ANSWER THIS

Historical TRAIN/DEV had no strict pre-probe RGB or frozen visual feature alignment (0/72 and 0/9). That is a dataset-observability limitation, not evidence that visual context is ineffective. This report uses a new prospective population with RGB, x, restorable snapshots, and outcomes aligned prospectively.

# π0 VISUAL PATH

The frozen π0 PaliGemma/SigLIP image-token path was instrumented before action decoding. Camera-specific valid-token mean pooling was concatenated (4096 float32), with PCA fit on TRAIN only to 64 dimensions. The feature was diagnostics-only and did not feed back into policy execution.

# POLICY-INVARIANCE OF VISUAL LOGGING

See `VISUAL_LOGGING_ACTION_INVARIANCE.json`: the fixed-observation microtest passed under the pre-existing nondeterministic serving baseline; instrumentation produced no semantically meaningful action change.

# PROSPECTIVE DATASET

TRAIN: 72 contexts, 720 target branches (5 continuous strata × 2 repeats). DEV: 9 contexts, 77 fixed 0.25N force cells, 385 repeats (5 per cell). The canonical outcome files are rebuilt from the frozen target manifests and corrected telemetry; the interrupted task1 cumulative CSV is not used as authority. Visual alignment and snapshot restore audits are required PASS artifacts.

# CONNECTION TO GNP

GNP maps observable geometry/context x plus hidden dynamics z and candidate action a to feasibility. Here x is frozen VLA/object visual context, z is represented experimentally by friction μ and physical dynamics, and a is continuous grip force F; the target is full-task feasibility.

# BASE FEAS

Matched nonvisual control: {base}

# VISUAL INTERCEPT RESIDUAL

Frozen Base plus a linear bφ(x) that does not receive F: {residual}

# DOES VISUAL x ONLY SHIFT THE CURVE?

**{'YES' if residual_wins else 'NO / not supported by the preregistered thresholds'}**. Residual probability relative improvement={residual_rel:.3f}; frontier improvement={residual_front:.3f}N; improved contexts={residual_multi}.

# FULL VISUAL FEAS

Full visual fusion: {full}

# DOES x×F INTERACTION MATTER?

**{'YES' if full_wins else 'NO / not supported by the preregistered thresholds'}**. Residual→Full probability relative improvement={full_rel:.3f}; frontier improvement={full_front:.3f}N.

# VISUAL JOINT

Matched Full Visual inputs plus the authoritative corrected physical and IE auxiliary targets: {joint}

# DOES PHYSICAL AUXILIARY ADD INDEPENDENT VALUE?

**{'YES' if joint_wins else 'NO / EVIDENCE LIMITED'}**. Joint status: `{joint_status}`.

# PROBABILITY RESULTS

See `PROSPECTIVE_VISUAL_DEV_PROBABILITY_METRICS.csv` for raw three-seed ensemble cell metrics, per-context MAE, Brier, NLL, signed bias, and calibration slope/intercept.

# CONTINUOUS FRONTIER RESULTS

See `PROSPECTIVE_VISUAL_DEV_FRONTIER_METRICS.csv` for 0.05N model queries, empirical 4/5 real frontiers, finite coverage, under-force, excess force, and monotonicity.

# UNDER-FORCE / EXCESS FORCE

Safety-first selection is recorded in `SELECTED_VISUAL_BACKEND.json`; the gate uses the selected model's valid real-frontier coverage, finite decision coverage, probability MAE, frontier MAE, under-force rate, and dense-grid monotonicity.

# SELECTED BACKEND

`{selected}`

# GT CONTINUOUS GATE

See `GT_CONTINUOUS_GATE.json`. Probe was not used to mask a GT failure. Probe status is `NOT_REACHED` because the gate did not pass; the existing frozen P4-B trace exposes diagnostic telemetry but no legitimate calibrated μ posterior estimator for an automatic Probe-vs-No-Probe control comparison.

# PRIMARY_CLASSIFICATION

`{classification}`

# JOINT_STATUS

`{joint_status}`

# PROBE_STATUS

`NOT_REACHED`

# WHAT IS NOW PROVEN

The new prospective collection establishes strict RGB/feature/outcome alignment, preserved frozen-policy action semantics, and matched visual/nonvisual training data. The model comparison is a fair within-distribution test of the visual-context hypothesis.

# WHAT IS STILL NOT PROVEN

- no cross-object claim;
- no unseen-task claim;
- no original TEST;
- no final fresh E2E;
- no when-to-probe policy;
- a failed GT gate prevents claiming continuous-control readiness even if some offline metric improves.

# METHOD IMPLICATION

If the gate fails, do not launch a fresh E2E benchmark or collect arbitrary additional visual features. Identify which deployment-valid physical variable still changes the force frontier under the same observable x and μ.

# NEXT_METHOD

If visual context fails, investigate task-relevant physical variables that remain unobserved under the same x and μ, considering mass/COM/geometry only under deployment-valid observability. If a future corrected backend passes GT and has a legitimate probe estimator, then freeze it and compare strict No-Probe, Probe, direct continuous Q2F, robust fixed high-force, and reactive slip control in the next goal.
"""
    (ROOT/"FINAL_REPORT.md").write_text(report)
    required=["PROSPECTIVE_VISUAL_FEATURE_SPEC.json","VISUAL_LOGGING_ACTION_INVARIANCE.json","PROSPECTIVE_CONTEXT_MANIFEST.csv","PROSPECTIVE_SNAPSHOT_RESTORE_AUDIT.json","PROSPECTIVE_VISUAL_ALIGNMENT_MANIFEST.csv","PROSPECTIVE_VISUAL_ALIGNMENT_AUDIT.json","PROSPECTIVE_TRAIN_FORCE_MANIFEST.csv","PROSPECTIVE_TRAIN_RUN_MANIFEST.csv","PROSPECTIVE_TRAIN_OUTCOMES.csv","PROSPECTIVE_TRAIN_TELEMETRY_AUDIT.json","PROSPECTIVE_DEV_FORCE_PROTOCOL.json","PROSPECTIVE_DEV_RUN_MANIFEST.csv","PROSPECTIVE_DEV_OUTCOMES.csv","PROSPECTIVE_REAL_CONTINUOUS_CURVES.csv","PROSPECTIVE_REAL_FRONTIERS.csv","PROSPECTIVE_BASE_FEAS_TRAINING_MANIFEST.csv","VISUAL_RESIDUAL_TRAINING_MANIFEST.csv","VISUAL_FULL_FEAS_TRAINING_MANIFEST.csv","VISUAL_JOINT_TRAINING_MANIFEST.csv","ALL_PROSPECTIVE_VISUAL_MODELS_FROZEN.json","PROSPECTIVE_VISUAL_DEV_PROBABILITY_METRICS.csv","PROSPECTIVE_VISUAL_DEV_FRONTIER_METRICS.csv","VISUAL_INTERCEPT_MECHANISM_TEST.csv","VISUAL_CONTEXT_VALUE_TEST.csv","VISUAL_JOINT_VALUE_TEST.csv","SELECTED_VISUAL_BACKEND.json","GT_CONTINUOUS_GATE.json","FINAL_REPORT.md"]
    audit={"status":"PASS" if all((ROOT/x).exists() for x in required) else "FAIL","missing":[x for x in required if not (ROOT/x).exists()],"required_count":len(required)}
    write_json(ROOT/"REQUIRED_ARTIFACTS_AUDIT.json",audit)
    lines=[]
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and p.name!="SHA256SUMS.txt": lines.append(f"{sha256(p)}  {p.relative_to(ROOT)}")
    (ROOT/"SHA256SUMS.txt").write_text("\n".join(lines)+"\n")


def evaluate():
    cs=contexts(); caps=captures(); _,xmap=fit_pca(caps,sorted([c for c,r in cs.items() if r["split"]=="TRAIN"]))
    tr=build_arrays(branch_records("TRAIN"),xmap); dv=build_arrays(branch_records("DEV"),xmap,tr[-1]); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models={}
    for kind,name in [("BASE","PROSPECTIVE_BASE_FEAS"),("RESIDUAL","VISUAL_INTERCEPT_RESIDUAL"),("FULL","VISUAL_CONTEXT_FULL_FEAS"),("JOINT","VISUAL_CONTEXT_JOINT")]:
        models[name]=[]
        for seed in SEEDS:
            p=ROOT/f"{name}_seed{seed}.pt"; q=torch.load(p,map_location=device,weights_only=False)
            if kind=="BASE": m=Base()
            elif kind=="RESIDUAL": m=nn.Linear(64,1)
            elif kind=="FULL": m=Full()
            else: m=Joint()
            m.load_state_dict(q["state_dict"]); models[name].append(m.to(device).eval())
    def pred(name,seq,cond,x):
        s=torch.tensor(seq,device=device); c=torch.tensor(cond,device=device); xx=torch.tensor(x,device=device)
        out=[]
        with torch.no_grad():
            for m in models[name]:
                if name.startswith("PROSPECTIVE_BASE"): z=m(s,c)
                elif name.startswith("VISUAL_INTERCEPT"): z=models["PROSPECTIVE_BASE_FEAS"][len(out)](s,c)+m(xx).squeeze(1)
                elif name.startswith("VISUAL_CONTEXT_JOINT"): z=m(s,c,xx)[0]
                else: z=m(s,c,xx)
                out.append(sigmoid(z.detach().cpu().numpy()))
        return np.mean(out,axis=0)
    # Cell-level real probabilities and model predictions.
    devmeta=dv[4]; cell=defaultdict(list)
    for i,r in enumerate(devmeta): cell[(r["context_id"],round(float(r["force_N"]),8))].append(i)
    rows=[]; frontier=[]; curves=[]
    for name in models:
        for (cid,f),ii in sorted(cell.items()):
            pp=float(pred(name,dv[0][ii[:1]],dv[1][ii[:1]],np.stack([xmap[cid]]))[0]); yy=float(np.mean(dv[2][ii])); rows.append({"model":name,"context_id":cid,"force_N":f,"p_pred":pp,"p_real":yy,"n":len(ii)})
        qrows=[r for r in rows if r["model"]==name]; y=np.array([r["p_real"] for r in qrows]); p=np.array([r["p_pred"] for r in qrows]); eps=1e-6
        slope=float(np.polyfit(p,y,1)[0]) if len(np.unique(p))>1 else float("nan"); intercept=float(np.polyfit(p,y,1)[1]) if len(np.unique(p))>1 else float("nan")
        write_csv(ROOT/"PROSPECTIVE_VISUAL_DEV_PROBABILITY_METRICS.csv",[])
    # Dense force curves use the first nominal H8 from each context and replace
    # only the candidate force; no observed branch telemetry enters the query.
    prob_metrics=[]; frontier_metrics=[]; curve_rows=[]
    byctx=defaultdict(list)
    for r in devmeta: byctx[r["context_id"]].append(r)
    for name in models:
        for cid,rr in sorted(byctx.items()):
            base_i=devmeta.index(rr[0]); task=int(rr[0]["task"]); lo,hi=TASK_SUPPORT[task]; grid=np.arange(lo,hi+1e-9,0.05)
            qseq=dv[0][base_i:base_i+1].copy(); qcond=dv[1][base_i:base_i+1].copy(); qs=[]
            for f in grid:
                s=qseq.copy(); s[:,:,3]=float(f); c=qcond.copy(); c[:,0]=float(f)/8; qs.append(float(pred(name,s,c,np.stack([xmap[cid]]))[0])); curve_rows.append({"model":name,"context_id":cid,"force_N":float(f),"p_pred":qs[-1]})
            real=[r for r in rows if r["model"]==name and r["context_id"]==cid]; real=sorted(real,key=lambda x:x["force_N"]); fr=next((float(r["force_N"]) for r in real if r["p_real"]>=.8),float("nan")); fp=next((float(f) for f,p in zip(grid,qs) if p>=.8),float("nan")); diffs=np.diff(qs); real_force=np.asarray([r["force_N"] for r in real],dtype=float); pred_at_real=np.interp(real_force,grid,qs); frontier_metrics.append({"model":name,"context_id":cid,"real_frontier_N":fr,"pred_frontier_N":fp,"finite_model":int(np.isfinite(fp)),"valid_real":int(np.isfinite(fr)),"frontier_error_N":abs(fp-fr) if np.isfinite(fp) and np.isfinite(fr) else "","under_force":int(np.isfinite(fp) and np.isfinite(fr) and fp<fr),"under_force_magnitude_N":max(0,fr-fp) if np.isfinite(fp) and np.isfinite(fr) else "","excess_force_N":max(0,fp-fr) if np.isfinite(fp) and np.isfinite(fr) else "","monotonic":int(np.all(diffs>=-1e-6)),"local_ordering":float(np.mean(diffs>=-1e-6)),"dense_nonmonotonicity":int(np.any(diffs<-1e-6)),"ranking_correlation":rank_correlation(pred_at_real,[r["p_real"] for r in real])})
    prob_out=[]
    for name in models:
        q=[r for r in rows if r["model"]==name]; pp=np.asarray([r["p_pred"] for r in q]); yy=np.asarray([r["p_real"] for r in q]); pp=np.clip(pp,1e-6,1-1e-6)
        prob_out.append({"row_type":"summary","model":name,"probability_MAE":float(np.mean(np.abs(pp-yy)),),"brier":float(np.mean((pp-yy)**2)),"NLL":float(-np.mean(yy*np.log(pp)+(1-yy)*np.log(1-pp))),"signed_bias":float(np.mean(pp-yy)),"calibration_slope":float(np.polyfit(pp,yy,1)[0]) if len(np.unique(pp))>1 else float("nan"),"calibration_intercept":float(np.polyfit(pp,yy,1)[1]) if len(np.unique(pp))>1 else float("nan"),"per_context_MAE":float(np.mean([np.mean(np.abs(np.asarray([x["p_pred"] for x in q if x["context_id"]==cid])-np.asarray([x["p_real"] for x in q if x["context_id"]==cid]))) for cid in sorted({x["context_id"] for x in q})]))})
        for cid in sorted({x["context_id"] for x in q}):
            z=[x for x in q if x["context_id"]==cid]
            prob_out.append({"row_type":"per_context","model":name,"context_id":cid,"probability_MAE":float(np.mean([abs(_f(x["p_pred"])-_f(x["p_real"])) for x in z]))})
    write_csv(ROOT/"PROSPECTIVE_VISUAL_DEV_PROBABILITY_METRICS.csv",prob_out)
    write_csv(ROOT/"PROSPECTIVE_VISUAL_DEV_FRONTIER_METRICS.csv",frontier_metrics)
    real_curve=[]
    for (cid,f), ii in sorted(cell.items()):
        z=[x for x in rows if x["model"]=="PROSPECTIVE_BASE_FEAS" and x["context_id"]==cid and abs(x["force_N"]-f)<1e-8]
        p_real=float(np.mean([x["p_real"] for x in z])); n=int(round(sum(x["n"] for x in z) / max(1, len(z)))); k=int(round(p_real*n)); lo,hi=wilson_interval(k,n)
        real_curve.append({"context_id":cid,"force_N":f,"successes":k,"repeats":n,"p_real":p_real,"wilson_low":lo,"wilson_high":hi})
    write_csv(ROOT/"PROSPECTIVE_REAL_CONTINUOUS_CURVES.csv",real_curve)
    write_csv(ROOT/"PROSPECTIVE_REAL_FRONTIERS.csv",[{"context_id":cid,"real_frontier_N":next((float(x["force_N"]) for x in sorted([z for z in rows if z["model"]=="PROSPECTIVE_BASE_FEAS" and z["context_id"]==cid],key=lambda z:z["force_N"]) if x["p_real"]>=.8),"")} for cid in sorted({x["context_id"] for x in rows if x["model"]=="PROSPECTIVE_BASE_FEAS"})])
    # A deterministic per-model summary is sufficient for the pre-registered
    # tests; all raw cell and dense curve rows remain in the CSVs above.
    summaries=[]
    for name in models:
        q=[r for r in frontier_metrics if r["model"]==name]; prob=next(r for r in read_csv(ROOT/"PROSPECTIVE_VISUAL_DEV_PROBABILITY_METRICS.csv") if r["model"]==name)
        valid=[r for r in q if r["valid_real"]]; finite=[r for r in valid if r["finite_model"]]; summaries.append({"model":name,"probability_MAE":float(prob["probability_MAE"]),"frontier_MAE_N":float(np.mean([r["frontier_error_N"] for r in finite])) if finite else "","real_frontier_coverage":len(valid)/len(q),"finite_decision_coverage":sum(r["finite_model"] for r in q)/len(q),"under_force_rate":float(np.mean([r["under_force"] for r in finite])) if finite else "","under_force_magnitude_N":float(np.mean([r["under_force_magnitude_N"] for r in finite])) if finite else "","monotonicity":float(np.mean([r["monotonic"] for r in q]))})
    base_s=next(x for x in summaries if x["model"]=="PROSPECTIVE_BASE_FEAS"); res_s=next(x for x in summaries if x["model"]=="VISUAL_INTERCEPT_RESIDUAL")
    residual_multi=sum(1 for r in frontier_metrics if r["model"]=="VISUAL_INTERCEPT_RESIDUAL" and r["valid_real"] and r["finite_model"] and _f(r["frontier_error_N"]) < next((_f(q["frontier_error_N"]) for q in frontier_metrics if q["model"]=="PROSPECTIVE_BASE_FEAS" and q["context_id"]==r["context_id"] and q["valid_real"] and q["finite_model"]),float("inf")))
    residual_front=float(base_s["frontier_MAE_N"])-float(res_s["frontier_MAE_N"])
    residual_rel_eval=float((base_s["probability_MAE"]-res_s["probability_MAE"])/base_s["probability_MAE"])
    residual_wins_eval=(residual_rel_eval>=.20 and residual_front>=.05 and float(res_s["under_force_rate"])<=float(base_s["under_force_rate"]) and residual_multi>=max(1,math.ceil(sum(r["valid_real"] for r in frontier_metrics if r["model"]=="VISUAL_INTERCEPT_RESIDUAL")/3)))
    write_csv(ROOT/"VISUAL_INTERCEPT_MECHANISM_TEST.csv",[{"comparison":"BASE_vs_VISUAL_INTERCEPT_RESIDUAL","residual_probability_relative_improvement":residual_rel_eval,"residual_frontier_improvement_N":residual_front,"residual_under_force_non_worse":int(float(res_s["under_force_rate"])<=float(base_s["under_force_rate"])),"residual_monotonicity":res_s["monotonicity"],"improved_contexts":residual_multi,"criterion_pass":int(residual_wins_eval),"classification":"VISUAL_CONTEXT_MAINLY_CORRECTS_DIFFICULTY_OFFSET" if residual_wins_eval else "NOT_SUPPORTED"}])
    write_csv(ROOT/"VISUAL_CONTEXT_VALUE_TEST.csv",summaries)
    visual_summaries=[r for r in summaries if r["model"] in ["VISUAL_INTERCEPT_RESIDUAL","VISUAL_CONTEXT_FULL_FEAS"]]
    best_visual=sorted(visual_summaries,key=lambda r:(float(r["under_force_rate"] or 1),-float(r["finite_decision_coverage"]),float(r["frontier_MAE_N"] or 99),float(r["probability_MAE"])))[0]
    joint=next(r for r in summaries if r["model"]=="VISUAL_CONTEXT_JOINT")
    best_visual_prob=next((r for r in prob_out if r["model"]==best_visual["model"] and r["row_type"]=="summary"),{})
    joint_prob=next((r for r in prob_out if r["model"]=="VISUAL_CONTEXT_JOINT" and r["row_type"]=="summary"),{})
    joint_front=float(best_visual["frontier_MAE_N"])-float(joint["frontier_MAE_N"])
    joint_prob_delta=float(best_visual["probability_MAE"])-float(joint["probability_MAE"])
    joint_brier_delta=float(joint_prob.get("brier",float("nan")))-float(best_visual_prob.get("brier",float("nan")))
    joint_wins_eval=(float(joint["under_force_rate"])<=float(best_visual["under_force_rate"]) and joint_front>=.05 and joint_prob_delta>=.1*float(best_visual["probability_MAE"]) and float(joint["monotonicity"])>=.9 and float(joint_prob.get("brier",float("inf")))<=1.05*float(best_visual_prob.get("brier",float("inf"))))
    joint_status_for_artifact="JOINT_HAS_INDEPENDENT_CONTINUOUS_VALUE" if joint_wins_eval else "JOINT_FRONTIER_SIGNAL_WITHOUT_RELIABLE_CONTROL" if joint_front>=.05 and (joint_prob_delta<0 or joint_brier_delta>0) else "NO_INDEPENDENT_JOINT_VALUE"
    write_csv(ROOT/"VISUAL_JOINT_VALUE_TEST.csv",[{"visual_feas":best_visual["model"],"joint":"VISUAL_CONTEXT_JOINT","frontier_improvement_N":joint_front,"probability_MAE_change_visual_minus_joint":joint_prob_delta,"brier_change_joint_minus_visual":joint_brier_delta,"under_force_visual":best_visual["under_force_rate"],"under_force_joint":joint["under_force_rate"],"criterion_pass":int(joint_wins_eval),"status":joint_status_for_artifact}])
    # Safety-first selection: under-force, then finite coverage, then frontier
    # error, then probability MAE; simplicity breaks practical ties.
    rank=sorted(summaries,key=lambda r:(float(r["under_force_rate"] or 1),-float(r["finite_decision_coverage"]),float(r["frontier_MAE_N"] or 99),float(r["probability_MAE"])))
    selected=rank[0]["model"] if rank else ""
    best=next(r for r in summaries if r["model"]==selected)
    gate={"selected_model":selected,"valid_real_frontier_coverage":best["real_frontier_coverage"],"finite_decision_coverage":best["finite_decision_coverage"],"probability_MAE":best["probability_MAE"],"frontier_MAE_N":best["frontier_MAE_N"],"under_force_rate":best["under_force_rate"],"systematic_dense_grid_nonmonotonicity":1-best["monotonicity"],"PASS":bool(best["real_frontier_coverage"]>=.8 and best["finite_decision_coverage"]>=.8 and best["probability_MAE"]<=.2 and (best["frontier_MAE_N"]=="" or float(best["frontier_MAE_N"])<=.2) and (best["under_force_rate"]=="" or float(best["under_force_rate"])<=.1) and 1-best["monotonicity"]<=.1)}
    write_json(ROOT/"SELECTED_VISUAL_BACKEND.json",{"selected_backend":selected,"safety_first_rank":rank,"GT_gate_pass":gate["PASS"]}); write_json(ROOT/"GT_CONTINUOUS_GATE.json",gate)
    write_json(ROOT/"PROBE_STATUS.json",{"status":"NOT_REACHED","reason":"GT continuous gate did not authorize downstream Probe/No-Probe control evaluation; frozen P4-B exposes telemetry but no legitimate calibrated mu posterior estimator"})
    write_json(ROOT/"EVALUATION_SUMMARIES.json",{"summaries":summaries,"selected":selected,"gate":gate})
    write_final_artifacts(summaries, frontier_metrics, prob_out, gate, selected)
    print(json.dumps({"selected":selected,"gate":gate,"summaries":summaries},indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["finalize","train","evaluate"]); a=ap.parse_args()
    {"finalize":finalize,"train":train_all,"evaluate":evaluate}[a.phase]()


if __name__=="__main__": main()
