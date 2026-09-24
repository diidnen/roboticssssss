#!/usr/bin/env python3
"""Minimal learned physical imagination successor.

This script trains a compact task-conditioned Physics-GRU on the existing
P5-S0-C deterministic branch telemetry.  It predicts physical trajectories,
not success or force.  The evaluator is deterministic and calibrated only on
DEV.  The script is deliberately separate from the old Query2Force code and
does not retrain the frozen friction estimator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO = Path("/home/exouser/Tabero")
RESULTS_ROOT = REPO / "analysis/results"
C_ARTIFACT = RESULTS_ROOT / "p5s0c_paired_boundary_probe_value_20260824_000542"
FROZEN_ESTIMATOR = RESULTS_ROOT / "active_friction_imagination_20260828_211106/FRICTION_GRU.pt"
OUT = Path()
SEED = 20260828
TASKS = [0, 1, 5, 6]
TASK_TO_IDX = {v: i for i, v in enumerate(TASKS)}
PHASES = ["branch_hold", "lift", "transit", "over_basket", "place", "release", "settle"]
MODES = ["track", "freeze", "open"]
FORCE_GRID = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
DT = 0.05
H_EVAL = 160


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader(); wr.writerows(rows)


@dataclass
class Branch:
    branch_id: str
    context_id: str
    root_id: str
    task: int
    split: str
    force: float
    friction: float
    success: int
    x_raw: np.ndarray
    y_raw: np.ndarray
    phase: list[str]
    length: int
    telemetry_path: str


def make_physical_state(d: pd.DataFrame) -> np.ndarray:
    """Available object–gripper physical state, without invented channels.

    Object and command positions use different simulator frames in the old
    telemetry.  Subtracting each trajectory's initial offset makes the target
    the relative displacement change, then we add finite-difference relative
    velocity and the two logged finger joint positions.
    """
    obj = d[["object_x_analysis_only", "object_y_analysis_only", "object_z_analysis_only"]].to_numpy(float)
    cmd = d[["cmd_x", "cmd_y", "cmd_z"]].to_numpy(float)
    rel_disp = (obj - cmd) - (obj[0] - cmd[0])
    rel_vel = np.vstack([np.zeros((1, 3)), np.diff(rel_disp, axis=0) / DT])
    aperture = d[["gripper_pos_0", "gripper_pos_1"]].to_numpy(float)
    return np.concatenate([rel_disp, rel_vel, aperture], axis=1).astype(np.float32)


def make_inputs(d: pd.DataFrame, state: np.ndarray, force: float, friction: float, task: int, initial: np.ndarray) -> np.ndarray:
    cmd = d[["cmd_x", "cmd_y", "cmd_z"]].to_numpy(float)
    cmd_rel = cmd - cmd[0]
    cmd_delta = np.vstack([np.zeros((1, 3)), np.diff(cmd, axis=0)])
    phase = np.stack([(d["phase"].to_numpy() == p).astype(float) for p in PHASES], axis=1)
    mode = np.stack([(d["mode"].to_numpy() == m).astype(float) for m in MODES], axis=1)
    task_onehot = np.zeros((len(d), len(TASKS)), dtype=float); task_onehot[:, TASK_TO_IDX[task]] = 1.0
    scalar = np.repeat([[force / 8.0, friction]], len(d), axis=0)
    init = np.repeat(initial[None, :], len(d), axis=0)
    return np.concatenate([cmd_rel, cmd_delta, phase, mode, scalar, task_onehot, state, init], axis=1).astype(np.float32)


def load_branches() -> list[Branch]:
    manifest = pd.read_csv(C_ARTIFACT / "P5S0C_BRANCH_MANIFEST.csv")
    branches: list[Branch] = []
    for r in manifest.itertuples(index=False):
        d = pd.read_csv(r.telemetry_path)
        state = make_physical_state(d)
        initial = state[0]
        inp = make_inputs(d, state, float(r.requested_force_N), float(r.hidden_friction_analysis_only), int(r.task), initial)
        branches.append(Branch(str(r.branch_id), str(r.context_id), str(r.root_id), int(r.task), str(r.split), float(r.requested_force_N), float(r.hidden_friction_analysis_only), int(r.full_task_success_y), inp[:-1], state[1:], [str(x) for x in d.phase[:-1]], len(d) - 1, str(r.telemetry_path)))
    if len(branches) != 576 or len({b.root_id for b in branches}) != 48:
        raise RuntimeError("P5-S0-C trajectory population mismatch")
    return branches


def stats(branches: list[Branch]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = [b for b in branches if b.split == "TRAIN"]
    xcat = np.concatenate([b.x_raw for b in train], axis=0)
    ycat = np.concatenate([b.y_raw for b in train], axis=0)
    xmu, xsd = xcat.mean(0), xcat.std(0); xsd[xsd < 1e-6] = 1.0
    ymu, ysd = ycat.mean(0), ycat.std(0); ysd[ysd < 1e-6] = 1.0
    return xmu.astype(np.float32), xsd.astype(np.float32), ymu.astype(np.float32), ysd.astype(np.float32)


class PhysicsGRU(nn.Module):
    def __init__(self, input_dim: int, state_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, state_dim))

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        z, hidden = self.gru(self.proj(x), hidden)
        return self.head(z), hidden


def padded_batch(items: list[Branch], xmu, xsd, ymu, ysd, device):
    maxlen = max(b.length for b in items); dim = items[0].x_raw.shape[1]
    x = np.zeros((len(items), maxlen, dim), np.float32); y = np.zeros((len(items), maxlen, 8), np.float32); mask = np.zeros((len(items), maxlen), np.float32)
    for i, b in enumerate(items):
        n = b.length; x[i, :n] = (b.x_raw - xmu) / xsd; y[i, :n] = (b.y_raw - ymu) / ysd; mask[i, :n] = 1.0
    return torch.tensor(x, device=device), torch.tensor(y, device=device), torch.tensor(mask, device=device)


def fit(branches: list[Branch], xmu, xsd, ymu, ysd, device) -> tuple[PhysicsGRU, dict[str, Any]]:
    train = [b for b in branches if b.split == "TRAIN"]; dev = [b for b in branches if b.split == "DEV"]
    model = PhysicsGRU(train[0].x_raw.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, best_epoch, stale, hist = float("inf"), None, 0, 0, []
    for epoch in range(1, 251):
        random.shuffle(train); model.train(); total = 0.0
        for start in range(0, len(train), 32):
            xb, yb, mb = padded_batch(train[start:start+32], xmu, xsd, ymu, ysd, device)
            opt.zero_grad(set_to_none=True); pred, _ = model(xb)
            loss = ((pred - yb).abs() * mb[..., None]).sum() / (mb.sum() * 8.0)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total += float(loss.item())
        model.eval(); dl = 0.0; dn = 0.0
        with torch.no_grad():
            for start in range(0, len(dev), 32):
                xb, yb, mb = padded_batch(dev[start:start+32], xmu, xsd, ymu, ysd, device); pred, _ = model(xb)
                dl += float(((pred-yb).abs()*mb[...,None]).sum().item()); dn += float(mb.sum().item()*8)
        val = dl / max(dn, 1.0); hist.append({"epoch": epoch, "train_loss": total / max((len(train)+31)//32,1), "dev_loss": val})
        if val < best - 1e-6:
            best, best_epoch, stale = val, epoch, 0; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 35: break
    if best_state is None: raise RuntimeError("world model did not produce checkpoint")
    model.load_state_dict(best_state); return model, {"best_epoch": best_epoch, "best_dev_loss": best, "history": hist}


def free_rollout(model, b: Branch, xmu, xsd, ymu, ysd, device, horizon: int | None = None) -> np.ndarray:
    n = min(b.length, horizon or b.length); state = b.y_raw[0].copy()  # state at telemetry step 1
    # Reconstruct the first physical state from the current input's state tail.
    state = b.x_raw[0, -8:].copy()
    # x_raw stores the initial state in its final 8 slots, not current state;
    # current state occupies the 8 slots immediately before that block.
    state = b.x_raw[0, -(8+8):-8].copy()
    preds = [state.copy()]
    hidden = None
    model.eval()
    with torch.no_grad():
        for t in range(n):
            raw = b.x_raw[t].copy(); raw[-16:-8] = state
            xt = torch.tensor(((raw-xmu)/xsd)[None,None,:], dtype=torch.float32, device=device)
            out, hidden = model(xt, hidden)
            state = (out[0,0].cpu().numpy() * ysd + ymu).astype(np.float32)
            preds.append(state.copy())
    return np.asarray(preds)


def teacher_prediction(model, b: Branch, xmu, xsd, ymu, ysd, device) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(((b.x_raw-xmu)/xsd)[None], dtype=torch.float32, device=device)
        out, _ = model(x)
    return out[0].cpu().numpy() * ysd + ymu


def physical_metrics(actual: np.ndarray, predicted: np.ndarray, phases: list[str]) -> dict[str, float]:
    n = min(len(actual), len(predicted)); a, p = actual[:n], predicted[:n]
    return {"rel_disp_mae_m": float(np.mean(np.abs(a[:,:3]-p[:,:3]))), "rel_vel_mae_mps": float(np.mean(np.abs(a[:,3:6]-p[:,3:6]))), "aperture_mae": float(np.mean(np.abs(a[:,6:8]-p[:,6:8]))), "state_rmse": float(np.sqrt(np.mean((a-p)**2)))}


def slip_score(state: np.ndarray, phases: list[str]) -> float:
    # During intended release, contact loss is allowed.  Score only the
    # transport segment where relative displacement is a physical slip proxy.
    ix = [i for i, ph in enumerate(phases[:len(state)]) if ph == "transit"]
    if not ix: return float("inf")
    return float(np.max(np.linalg.norm(state[ix, :3], axis=1)))


def actual_state_for_branch(b: Branch) -> np.ndarray:
    # y_raw starts at telemetry row 1; prepend the zero relative state is
    # unnecessary for the transport evaluator but improves alignment.
    return b.y_raw


def free_metric_rows(model, branches, xmu,xsd,ymu,ysd,device):
    rows=[]
    for b in branches:
        pred = free_rollout(model,b,xmu,xsd,ymu,ysd,device,H_EVAL)[1:]
        actual = b.y_raw[:len(pred)]
        init = b.x_raw[0,-16:-8]
        constant = np.repeat(init[None,:], len(actual), axis=0)
        pm = physical_metrics(actual,pred,b.phase)
        cm = physical_metrics(actual,constant,b.phase)
        rows.append({"branch_id":b.branch_id,"split":b.split,"task":b.task,"force":b.force,"success":b.success,"free_rel_disp_mae_m":pm["rel_disp_mae_m"],"free_rel_vel_mae_mps":pm["rel_vel_mae_mps"],"free_aperture_mae":pm["aperture_mae"],"free_state_rmse":pm["state_rmse"],"constant_rel_disp_mae_m":cm["rel_disp_mae_m"],"constant_rel_vel_mae_mps":cm["rel_vel_mae_mps"],"constant_aperture_mae":cm["aperture_mae"],"constant_state_rmse":cm["state_rmse"]})
    return rows


def choose_threshold(dev_rows: list[dict[str, Any]]) -> dict[int, float]:
    thresholds = {}
    for task in TASKS:
        g = [r for r in dev_rows if r["task"] == task]
        vals = sorted({float(r["pred_slip"]) for r in g})
        candidates = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45] + vals
        best = (float("inf"), 0.35)
        for tau in candidates:
            err = sum(int((r["pred_slip"] <= tau) != bool(r["actual_success"])) for r in g)
            if err < best[0]: best = (err, tau)
        thresholds[task] = float(best[1])
    return thresholds


def evaluator_success(state: np.ndarray, phases: list[str], task: int, thresholds: dict[int,float]) -> int:
    transport_idx = [i for i, ph in enumerate(phases[:len(state)]) if ph == "transit"]
    if not transport_idx: return 0
    slip = float(np.max(np.linalg.norm(state[transport_idx, :3], axis=1)))
    return int(slip <= thresholds[task])


def frontier_eval(model, branches, xmu,xsd,ymu,ysd,device,thresholds, use_mu: str) -> list[dict[str,Any]]:
    # use_mu: gt, prior, or estimated.  Estimated uses the frozen estimator
    # checkpoint only in the separate online-compatible analysis; this offline
    # gate uses the exact per-context saved probe through the estimator output
    # produced by the previous study when available, otherwise GT is excluded.
    out=[]
    for b in branches:
        if b.split not in {"DEV","TEST"}: continue
        if use_mu == "gt": mu = b.friction
        elif use_mu == "prior": mu = 0.58
        else: continue
        # Inputs were recorded with the branch's true mu. Rebuild the input to
        # keep the world-model query honest for the chosen imagined mu.
        d = pd.read_csv(b.telemetry_path); state = make_physical_state(d); inp = make_inputs(d,state,b.force,mu,b.task,state[0]); bb = Branch(b.branch_id,b.context_id,b.root_id,b.task,b.split,b.force,mu,b.success,inp[:-1],state[1:],b.phase,b.length,b.telemetry_path)
        pred = free_rollout(model,bb,xmu,xsd,ymu,ysd,device,H_EVAL)
        actual = actual_state_for_branch(b)
        pred_slip = slip_score(pred, b.phase); actual_slip = slip_score(actual,b.phase)
        out.append({"branch_id":b.branch_id,"context_id":b.context_id,"root_id":b.root_id,"task":b.task,"split":b.split,"force":b.force,"friction_gt":b.friction,"mu_used":mu,"actual_success":b.success,"pred_slip":pred_slip,"actual_slip":actual_slip,"pred_success":evaluator_success(pred,b.phase,b.task,thresholds)})
    return out


def main() -> int:
    global OUT
    OUT = RESULTS_ROOT / f"learned_physical_imagination_{now_tag()}"; OUT.mkdir(parents=True, exist_ok=False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    branches = load_branches(); xmu,xsd,ymu,ysd = stats(branches)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info = fit(branches,xmu,xsd,ymu,ysd,device)
    ckpt=OUT/"PHYSICS_GRU.pt"; torch.save({"state_dict":model.state_dict(),"input_dim":int(branches[0].x_raw.shape[1]),"state_dim":8,"hidden_dim":64,"seed":SEED},ckpt)
    write_json(OUT/"DATASET_AUDIT.json",{"branches":len(branches),"roots":len({b.root_id for b in branches}),"contexts":len({b.context_id for b in branches}),"success":sum(b.success for b in branches),"failure":sum(not b.success for b in branches),"splits":{s:len({b.root_id for b in branches if b.split==s}) for s in ["TRAIN","DEV","TEST"]},"tasks":TASKS,"friction_only":True,"telemetry_targets":["relative_object_command_displacement_m","relative_velocity_mps","gripper_pos_0","gripper_pos_1"],"excluded_unavailable":["relative_orientation","explicit_contact_indicator","normal_force"]})
    write_json(OUT/"NORMALIZATION.json",{"x_mean":xmu.tolist(),"x_std":xsd.tolist(),"y_mean":ymu.tolist(),"y_std":ysd.tolist(),"source":"TRAIN roots only"})
    write_json(OUT/"PROTOCOL.json",{"method":"Physics-GRU learned physical trajectory imagination","probe":"reused P4-B fixed probe","friction_estimator":str(FROZEN_ESTIMATOR),"world_model":{"hidden_dim":64,"target":"physical trajectory, not success/force","input":"nominal command/phase + force + friction + task + current state + initial state"},"candidate_forces_N":FORCE_GRID,"evaluator":"deterministic transport slip threshold calibrated on DEV","h_eval":H_EVAL,"root_split":"TRAIN/DEV/TEST by root_id"})
    write_csv(OUT/"TRAINING_LOG.csv",info.pop("history")); write_json(OUT/"TRAINING_RESULT.json",info)
    metric_rows=[]
    for split in ["TRAIN","DEV","TEST"]:
        for b in [x for x in branches if x.split==split]:
            pred=teacher_prediction(model,b,xmu,xsd,ymu,ysd,device); metric_rows.append({"branch_id":b.branch_id,"split":split,"task":b.task,"force":b.force,"success":b.success,**physical_metrics(b.y_raw,pred,b.phase)})
    mdf=pd.DataFrame(metric_rows); metrics={}
    for split in ["TRAIN","DEV","TEST"]:
        q=mdf[mdf.split==split]; metrics[split.lower()]={"n":len(q),"roots":len({b.root_id for b in branches if b.split==split}),"rel_disp_mae_m":float(q.rel_disp_mae_m.mean()),"rel_vel_mae_mps":float(q.rel_vel_mae_mps.mean()),"aperture_mae":float(q.aperture_mae.mean()),"state_rmse":float(q.state_rmse.mean())}
    write_csv(OUT/"TEACHER_FORCED_METRICS.csv",metric_rows); write_json(OUT/"TRAJECTORY_METRICS.json",metrics)
    free_rows = free_metric_rows(model, branches, xmu,xsd,ymu,ysd,device)
    write_csv(OUT/"FREE_ROLLOUT_METRICS.csv", free_rows)
    fmdf=pd.DataFrame(free_rows); free_metrics={}
    for split in ["TRAIN","DEV","TEST"]:
        q=fmdf[fmdf.split==split]
        free_metrics[split.lower()]={"n":len(q),"free_rel_disp_mae_m":float(q.free_rel_disp_mae_m.mean()),"constant_rel_disp_mae_m":float(q.constant_rel_disp_mae_m.mean()),"free_rel_vel_mae_mps":float(q.free_rel_vel_mae_mps.mean()),"constant_rel_vel_mae_mps":float(q.constant_rel_vel_mae_mps.mean()),"free_state_rmse":float(q.free_state_rmse.mean()),"constant_state_rmse":float(q.constant_state_rmse.mean())}
    write_json(OUT/"FREE_ROLLOUT_METRICS.json",free_metrics)
    # DEV-only deterministic evaluator calibration, then locked frontier tests.
    dev_raw=[]
    for b in [x for x in branches if x.split=="DEV"]:
        pred=free_rollout(model,b,xmu,xsd,ymu,ysd,device,H_EVAL); dev_raw.append({"task":b.task,"pred_slip":slip_score(pred,b.phase),"actual_success":b.success})
    thresholds=choose_threshold(dev_raw); write_json(OUT/"EVALUATOR_DEV_FREEZE.json",{"transport_slip_threshold_m_by_task":thresholds,"selection":"minimum DEV classification error; frozen before TEST"})
    gt=frontier_eval(model,branches,xmu,xsd,ymu,ysd,device,thresholds,"gt")
    write_csv(OUT/"GT_FRICTION_FRONTIER.csv",gt)
    g=pd.DataFrame(gt); frontier_rows=[]
    for cid,grp in g.groupby("context_id"):
        true=grp[grp.actual_success==1].force.min() if (grp.actual_success==1).any() else np.nan
        imagined=grp[grp.pred_success==1].force.min() if (grp.pred_success==1).any() else np.nan
        frontier_rows.append({"context_id":cid,"root_id":grp.root_id.iloc[0],"task":int(grp.task.iloc[0]),"split":grp.split.iloc[0],"true_min_force":true,"imagined_min_force":imagined,"exact_match":int(true==imagined) if np.isfinite(true) and np.isfinite(imagined) else 0,"under_force":int(np.isfinite(imagined) and np.isfinite(true) and imagined<true),"over_force":int(np.isfinite(imagined) and np.isfinite(true) and imagined>true)})
    write_csv(OUT/"GT_FRICTION_FRONTIER_SUMMARY.csv",frontier_rows)
    f=pd.DataFrame(frontier_rows); summary={"contexts":len(f),"frontier_exact_match":float(f.exact_match.mean()),"under_force_rate":float(f.under_force.mean()),"over_force_rate":float(f.over_force.mean()),"test_contexts":int((f.split=="TEST").sum()),"test_exact_match":float(f[f.split=="TEST"].exact_match.mean())}
    event_rows=[]
    for split in ["DEV","TEST"]:
        q=g[g.split==split]
        pred_event=[]; true_event=[]
        for r in q.itertuples():
            tau=thresholds[int(r.task)]; pred_event.append(float(r.pred_slip)>tau); true_event.append(float(r.actual_slip)>tau)
        tp=sum(a and p for a,p in zip(true_event,pred_event)); fp=sum((not a) and p for a,p in zip(true_event,pred_event)); fn=sum(a and (not p) for a,p in zip(true_event,pred_event)); precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); event_rows.append({"split":split,"n":len(q),"slip_event_precision":precision,"slip_event_recall":recall,"slip_event_f1":2*precision*recall/max(precision+recall,1e-9),"true_events":sum(true_event),"pred_events":sum(pred_event)})
    write_csv(OUT/"SLIP_EVENT_METRICS.csv",event_rows)
    ev={r["split"].lower():r for r in event_rows}; summary.update({"dev_slip_event_f1":ev["dev"]["slip_event_f1"],"test_slip_event_f1":ev["test"]["slip_event_f1"],"dev_gate_pass":int(float(f[f.split=="DEV"].exact_match.mean())>=0.80 and float(f[f.split=="DEV"].under_force.mean())<=0.10 and ev["dev"]["slip_event_f1"]>=0.70),"preregistered_gate":"DEV frontier exact >= 0.80, under-force <= 0.10, slip F1 >= 0.70"})
    write_json(OUT/"GT_FRICTION_ACTION_GATE.json",summary)
    report=f"""# Learned Physical Imagination\n\nSTATUS: GT_FRICTION_ACTION_GATE_RECORDED\n\n- branches: {len(branches)}; success/failure: {sum(b.success for b in branches)}/{sum(not b.success for b in branches)}\n- roots: TRAIN 24, DEV 8, TEST 16\n- target: relative object-command displacement, relative velocity, gripper aperture\n- model: projection -> GRU(64) -> next physical state head\n- GT frontier exact match: {summary['frontier_exact_match']:.3f}; TEST: {summary['test_exact_match']:.3f}\n- evaluator thresholds were frozen from DEV only\n\nThis report is the Gate-3 diagnostic. The model predicts trajectories rather than success/force.\n"""; (OUT/"REPORT.md").write_text(report)
    print(json.dumps({"out":str(OUT),"metrics":metrics,"gt_action_gate":summary},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
