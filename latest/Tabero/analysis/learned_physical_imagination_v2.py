#!/usr/bin/env python3
"""Contact/slip-supervised Physics-GRU v2.

The model predicts compact physical state and physical events only.  It has no
success, force-class, or frontier prediction head.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
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
RESULTS = REPO / "analysis/results"
OLD = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
V1 = RESULTS / "learned_physical_imagination_20260828_222705"
TASKS = [0, 1, 5, 6]
TASK_IDX = {t: i for i, t in enumerate(TASKS)}
PHASES = ["branch_hold", "lift", "transit", "over_basket", "place", "release", "settle"]
MODES = ["track", "freeze", "open"]
DT = 0.05
CONTACT_EPS_N = 0.15
SLIP_SPEED_EPS = 0.025
FORCE_GRID = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
OUT = Path()
CONTACT_ROOT = Path()


def write_json(p: Path, x: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(p: Path, rows: list[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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
    x: np.ndarray
    y: np.ndarray
    events: np.ndarray
    event_mask: np.ndarray
    phases: list[str]
    length: int
    telemetry_path: str
    direct_contact: bool
    weight: float


def contact_files() -> dict[tuple[str, float], Path]:
    ans = {}
    if not CONTACT_ROOT.exists():
        return ans
    for p in CONTACT_ROOT.rglob("*_trajectory.csv"):
        try:
            d = pd.read_csv(p, nrows=1)
            if not d.empty and "context_id" in d and "requested_force_N" in d:
                ans[(str(d.context_id.iloc[0]), round(float(d.requested_force_N.iloc[0]), 4))] = p
        except Exception:
            pass
    return ans


def state_of(d: pd.DataFrame) -> np.ndarray:
    obj = d[["object_x_analysis_only", "object_y_analysis_only", "object_z_analysis_only"]].to_numpy(float)
    cmd = d[["cmd_x", "cmd_y", "cmd_z"]].to_numpy(float)
    rel = (obj - cmd) - (obj[0] - cmd[0])
    if {"object_vx_mps", "object_vy_mps", "object_vz_mps"}.issubset(d.columns):
        ov = d[["object_vx_mps", "object_vy_mps", "object_vz_mps"]].to_numpy(float)
        cv = np.vstack([np.zeros((1, 3)), np.diff(cmd, axis=0) / DT])
        vel = ov - cv
    else:
        vel = np.vstack([np.zeros((1, 3)), np.diff(rel, axis=0) / DT])
    ap = d[["gripper_pos_0", "gripper_pos_1"]].to_numpy(float)
    return np.concatenate([rel, vel, ap], axis=1).astype(np.float32)


def input_of(d: pd.DataFrame, state: np.ndarray, force: float, mu: float, task: int) -> np.ndarray:
    cmd = d[["cmd_x", "cmd_y", "cmd_z"]].to_numpy(float)
    cmd_rel = cmd - cmd[0]
    cmd_delta = np.vstack([np.zeros((1, 3)), np.diff(cmd, axis=0)])
    phase = np.stack([(d.phase.to_numpy() == p).astype(float) for p in PHASES], 1)
    mode = np.stack([(d["mode"].to_numpy() == m).astype(float) for m in MODES], 1)
    task_oh = np.zeros((len(d), len(TASKS)))
    task_oh[:, TASK_IDX[task]] = 1.0
    scalars = np.repeat([[force / 8.0, mu]], len(d), 0)
    initial = np.repeat(state[0][None], len(d), 0)
    return np.concatenate([cmd_rel, cmd_delta, phase, mode, scalars, task_oh, state, initial], 1).astype(np.float32)


def make_events(d: pd.DataFrame, state: np.ndarray, direct: bool):
    n = len(d)
    e = np.zeros((n, 3), np.float32)
    mask = np.zeros((n, 3), np.float32)
    if direct and {"left_normal_force_N", "right_normal_force_N", "left_force_norm_N", "right_force_norm_N"}.issubset(d.columns):
        e[:, 0] = ((d.left_force_norm_N.to_numpy(float) > CONTACT_EPS_N) & (d.left_normal_force_N.to_numpy(float) > CONTACT_EPS_N)).astype(float)
        e[:, 1] = ((d.right_force_norm_N.to_numpy(float) > CONTACT_EPS_N) & (d.right_normal_force_N.to_numpy(float) > CONTACT_EPS_N)).astype(float)
        mask[:, :2] = 1.0
    active = np.isin(d.phase.astype(str).to_numpy(), ["lift", "transit", "over_basket", "place"])
    tangential = np.linalg.norm(state[:, 3:5], axis=1)
    slip = active & (tangential > SLIP_SPEED_EPS)
    if direct and {"left_normal_force_N", "right_normal_force_N", "left_force_norm_N", "right_force_norm_N"}.issubset(d.columns):
        slip |= active & ((e[:, 0] < 0.5) | (e[:, 1] < 0.5))
    e[:, 2] = slip.astype(float)
    mask[:, 2] = 1.0
    return e[1:], mask[1:]


def boundary_weights(m: pd.DataFrame):
    ans = {}
    for cid, g in m.groupby("context_id"):
        fs = sorted(float(x) for x in g.requested_force_N.unique())
        ok = sorted(float(x) for x in g.loc[g.full_task_success_y == 1, "requested_force_N"].unique())
        fstar = ok[0] if ok else fs[-1]
        for f in fs:
            d = abs(f - fstar)
            ans[(str(cid), round(f, 4))] = 5.0 if d <= 0.25 else 3.0 if d <= 0.75 else 1.0
    return ans


def load_data():
    m = pd.read_csv(OLD / "P5S0C_BRANCH_MANIFEST.csv")
    cmap = contact_files()
    weights = boundary_weights(m)
    branches = []
    for r in m.itertuples(index=False):
        key = (str(r.context_id), round(float(r.requested_force_N), 4))
        path = cmap.get(key, Path(str(r.telemetry_path)))
        d = pd.read_csv(path)
        s = state_of(d)
        direct = key in cmap and {"contact_left", "contact_right"}.issubset(d.columns)
        ev, em = make_events(d, s, direct)
        inp = input_of(d, s, float(r.requested_force_N), float(r.hidden_friction_analysis_only), int(r.task))
        branches.append(Branch(str(r.branch_id), str(r.context_id), str(r.root_id), int(r.task), str(r.split), float(r.requested_force_N), float(r.hidden_friction_analysis_only), int(r.full_task_success_y), inp[:-1], s[1:], ev, em, d.phase.astype(str).to_list()[:-1], len(d) - 1, str(path), direct, weights[key]))
    if len(branches) != 576 or len({b.root_id for b in branches}) != 48:
        raise RuntimeError("authoritative branch population mismatch")
    return branches, {"contact_files": len(cmap), "direct_contact_branches": sum(b.direct_contact for b in branches), "derived_slip_branches": sum(not b.direct_contact for b in branches)}


def norm_stats(bs):
    tr = [b for b in bs if b.split == "TRAIN"]
    x = np.concatenate([b.x for b in tr]); y = np.concatenate([b.y for b in tr])
    xm, xs = x.mean(0), x.std(0); xs[xs < 1e-6] = 1
    ym, ys = y.mean(0), y.std(0); ys[ys < 1e-6] = 1
    return tuple(v.astype(np.float32) for v in (xm, xs, ym, ys))


class PhysicsGRUV2(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.state_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 8))
        self.event_head = nn.Linear(hidden, 3)

    def forward(self, x, h=None):
        z, h = self.gru(self.proj(x), h)
        return self.state_head(z), self.event_head(z), h


def pack(items, xm, xs, ym, ys, device):
    n, d = max(b.length for b in items), items[0].x.shape[1]
    x = np.zeros((len(items), n, d), np.float32)
    y = np.zeros((len(items), n, 8), np.float32)
    e = np.zeros((len(items), n, 3), np.float32)
    em = np.zeros((len(items), n, 3), np.float32)
    mask = np.zeros((len(items), n), np.float32)
    w = np.zeros((len(items), n), np.float32)
    for i, b in enumerate(items):
        k = b.length
        x[i, :k] = (b.x - xm) / xs; y[i, :k] = (b.y - ym) / ys
        e[i, :k] = b.events; em[i, :k] = b.event_mask; mask[i, :k] = 1; w[i, :k] = b.weight
    return [torch.tensor(v, device=device) for v in (x, y, e, em, mask, w)]


def free_loss(model, items, xm, xs, ym, ys, device, horizon, pos):
    ls = []
    for b in items[:4]:
        if b.length < 2:
            continue
        start = random.randrange(max(1, b.length - min(horizon, b.length - 1)))
        state = b.x[start, -16:-8].copy()
        h = None
        for t in range(start, min(b.length, start + horizon)):
            raw = b.x[t].copy(); raw[-16:-8] = state
            xt = torch.tensor(((raw - xm) / xs)[None, None], dtype=torch.float32, device=device)
            sp, ep, h = model(xt, h)
            yt = torch.tensor(((b.y[t] - ym) / ys)[None, None], dtype=torch.float32, device=device)
            et = torch.tensor(b.events[t][None, None], dtype=torch.float32, device=device)
            ls.append(nn.functional.smooth_l1_loss(sp, yt) + .5 * nn.functional.binary_cross_entropy_with_logits(ep, et, pos_weight=pos))
            state = (sp[0, 0].detach().cpu().numpy() * ys + ym).astype(np.float32)
    return torch.stack(ls).mean() if ls else torch.tensor(0., device=device)


def train(bs, xm, xs, ym, ys, device):
    tr = [b for b in bs if b.split == "TRAIN"]; dev = [b for b in bs if b.split == "DEV"]
    all_e = np.concatenate([b.events for b in tr]); pos = torch.tensor(((1 - all_e).sum(0) / np.maximum(all_e.sum(0), 1)).clip(1, 20), dtype=torch.float32, device=device)
    model = PhysicsGRUV2(tr[0].x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best, best_state, stale, hist = float("inf"), None, 0, []
    for epoch in range(1, 81):
        random.shuffle(tr); model.train(); losses = []
        horizon = 1 if epoch < 12 else 5 if epoch < 24 else 10 if epoch < 48 else 20
        for i in range(0, len(tr), 24):
            xb, yb, eb, em, mk, wt = pack(tr[i:i+24], xm, xs, ym, ys, device)
            opt.zero_grad(set_to_none=True); sp, ep, _ = model(xb)
            cont = (nn.functional.smooth_l1_loss(sp, yb, reduction="none") * mk[..., None] * wt[..., None]).sum() / (mk.sum() * 8)
            event = (nn.functional.binary_cross_entropy_with_logits(ep, eb, reduction="none", pos_weight=pos) * em * mk[..., None] * wt[..., None]).sum() / (em.sum() + 1e-6)
            loss = cont + 1.5 * event + .35 * free_loss(model, tr[i:i+24], xm, xs, ym, ys, device, horizon, pos)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); losses.append(float(loss.item()))
        model.eval(); vals = []
        with torch.no_grad():
            for i in range(0, len(dev), 24):
                xb, yb, eb, em, mk, wt = pack(dev[i:i+24], xm, xs, ym, ys, device); sp, ep, _ = model(xb)
                c = (nn.functional.smooth_l1_loss(sp, yb, reduction="none") * mk[..., None]).sum() / (mk.sum() * 8)
                e = (nn.functional.binary_cross_entropy_with_logits(ep, eb, reduction="none", pos_weight=pos) * em * mk[..., None]).sum() / (em.sum() + 1e-6)
                vals.append(float((c + 1.5 * e).item()))
        val = float(np.mean(vals)); hist.append({"epoch": epoch, "free_horizon": horizon, "train_loss": float(np.mean(losses)), "dev_loss": val})
        if val < best - 1e-5:
            best, best_state, stale = val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= 18:
                break
    model.load_state_dict(best_state)
    return model, {"best_dev_loss": best, "best_epoch": min(hist, key=lambda x: x["dev_loss"])["epoch"], "event_pos_weight": pos.cpu().tolist(), "history": hist}


def rollout(model, b, xm, xs, ym, ys, device, horizon=215):
    state = b.x[0, -16:-8].copy(); h = None; ss = []; pp = []
    with torch.no_grad():
        for t in range(min(b.length, horizon)):
            raw = b.x[t].copy(); raw[-16:-8] = state
            sp, ep, h = model(torch.tensor(((raw - xm) / xs)[None, None], dtype=torch.float32, device=device), h)
            state = (sp[0, 0].cpu().numpy() * ys + ym).astype(np.float32)
            ss.append(state.copy()); pp.append(torch.sigmoid(ep[0, 0]).cpu().numpy())
    return np.asarray(ss), np.asarray(pp)


def physical_metrics(a, p):
    n = min(len(a), len(p)); a, p = a[:n], p[:n]
    return {"rel_disp_mae_m": float(np.abs(a[:, :3] - p[:, :3]).mean()), "rel_vel_mae_mps": float(np.abs(a[:, 3:6] - p[:, 3:6]).mean()), "aperture_mae": float(np.abs(a[:, 6:8] - p[:, 6:8]).mean()), "state_rmse": float(np.sqrt(((a - p) ** 2).mean()))}


def binary(y, p):
    y = np.asarray(y).astype(bool); p = np.asarray(p).astype(bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    pre, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": pre, "recall": rec, "f1": 2 * pre * rec / max(pre + rec, 1e-9), "tp": tp, "fp": fp, "fn": fn}


def evaluator_freeze(dev_rows):
    best = None
    for et in [.35, .45, .5, .55, .65]:
        for dt in [.008, .012, .016, .020, .025, .030]:
            err = sum(
                int(
                    (np.any(r["prob"][:, 2] >= et) or np.max(np.linalg.norm(r["pred"][:, :3], axis=1)) > dt)
                    == (not bool(r["success"]))
                )
                for r in dev_rows
            )
            if best is None or err < best[0]:
                best = (err, et, dt)
    return {"event_probability_threshold": best[1], "relative_displacement_threshold_m": best[2], "dev_error": best[0]}


def main():
    global OUT, CONTACT_ROOT
    OUT = RESULTS / f"learned_physical_imagination_v2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    OUT.mkdir(parents=True, exist_ok=False)
    CONTACT_ROOT = Path(os.environ.get("V2_CONTACT_ROOT", ""))
    random.seed(20260828); np.random.seed(20260828); torch.manual_seed(20260828)
    bs, data_meta = load_data(); xm, xs, ym, ys = norm_stats(bs); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_est = RESULTS / "active_friction_imagination_20260828_211106/FRICTION_GRU.pt"
    reuse = {"P4-B_probe": "reused", "P5-S0-C_population": str(OLD / "P5S0C_BRANCH_MANIFEST.csv"), "friction_estimator": str(old_est), "Pi0": "frozen Pi0Config semantic policy; unchanged", "Physics-GRU_v1": str(V1 / "PHYSICS_GRU.pt"), "new": ["contact/slip event supervision", "boundary-weighted multi-step training"]}
    write_json(OUT / "REUSE_MAP.json", reuse)
    write_json(OUT / "CONTACT_TELEMETRY_AUDIT.json", {"direct_channels": {"left_right_force": {"source": "obs['policy']['gripper_net_force'] / contact_gripper.net_forces_w", "units": "N", "frequency": "each env.step", "type": "measured force; contact bool norm > 0.20 N", "new_logged": data_meta["direct_contact_branches"]}, "object_velocity": {"source": "scene[object].data.root_lin_vel_w/root_ang_vel_w", "units": "m/s, rad/s", "type": "measured simulator state", "new_logged": data_meta["direct_contact_branches"]}}, "historical_channels": ["object pose", "commanded TCP position", "measured squeeze force", "finger joint positions"], "historical_missing": ["per-finger contact force/pair identity"], "snapshot_restore": "existing P5-S0-C parity contract retained", "coverage": data_meta})
    event_def = {"event_vector": ["contact_left", "contact_right", "unintended_slip"], "contact_threshold_N": CONTACT_EPS_N, "slip_threshold_tangential_relative_speed_mps": SLIP_SPEED_EPS, "active_phases": ["lift", "transit", "over_basket", "place"], "release_excluded": True, "slip": "active-phase tangential object-command relative speed above threshold OR direct bilateral contact loss", "hash": hashlib.sha256(json.dumps({"contact": CONTACT_EPS_N, "slip": SLIP_SPEED_EPS, "active": ["lift", "transit", "over_basket", "place"]}, sort_keys=True).encode()).hexdigest()}
    write_json(OUT / "PHYSICAL_EVENT_DEFINITION.json", event_def)
    write_json(OUT / "DATASET_AUDIT.json", {"branches": len(bs), "roots": len({b.root_id for b in bs}), "contexts": len({b.context_id for b in bs}), "success": sum(b.success for b in bs), "failure": sum(not b.success for b in bs), "direct_contact_branches": data_meta["direct_contact_branches"], "derived_slip_branches": data_meta["derived_slip_branches"], "splits": {s: {"roots": len({b.root_id for b in bs if b.split == s}), "branches": sum(b.split == s for b in bs)} for s in ["TRAIN", "DEV", "TEST"]}, "boundary_weight_counts": {str(w): sum(b.weight == w for b in bs) for w in [1., 3., 5.]}, "friction_only": True})
    model, info = train(bs, xm, xs, ym, ys, device)
    ckpt = OUT / "PHYSICS_GRU_V2.pt"; torch.save({"state_dict": model.state_dict(), "input_dim": bs[0].x.shape[1], "hidden_dim": 64, "state_dim": 8, "event_dim": 3}, ckpt)
    write_json(OUT / "NORMALIZATION.json", {"x_mean": xm.tolist(), "x_std": xs.tolist(), "y_mean": ym.tolist(), "y_std": ys.tolist(), "source": "TRAIN roots only"}); write_csv(OUT / "TRAINING_LOG.csv", info.pop("history")); write_json(OUT / "TRAINING_RESULT.json", info)
    free, evrows = [], []
    for b in bs:
        pred, prob = rollout(model, b, xm, xs, ym, ys, device)
        free.append({"branch_id": b.branch_id, "split": b.split, "task": b.task, "force": b.force, "success": b.success, **physical_metrics(b.y, pred)})
        active = np.isin(np.asarray(b.phases), ["lift", "transit", "over_basket", "place"])[:len(prob)]
        for j, name in enumerate(["contact_left", "contact_right", "slip"]):
            mask = (b.event_mask[:len(prob), j] > .5) if j < 2 else active
            if mask.any():
                evrows.append({"branch_id": b.branch_id, "split": b.split, "event": name, "near_frontier": int(b.weight > 1), **binary(b.events[:len(prob), j][mask], prob[:, j][mask] >= .5)})
    write_csv(OUT / "FREE_ROLLOUT_METRICS.csv", free); write_csv(OUT / "FREE_EVENT_METRICS.csv", evrows)
    v1_free = json.loads((V1 / "FREE_ROLLOUT_METRICS.json").read_text()) if (V1 / "FREE_ROLLOUT_METRICS.json").exists() else {}
    fidelity = {}
    for s in ["TRAIN", "DEV", "TEST"]:
        q = pd.DataFrame([r for r in free if r["split"] == s])
        fidelity[s.lower()] = {"v2_free": {k: float(q[k].mean()) for k in ["rel_disp_mae_m", "rel_vel_mae_mps", "aperture_mae", "state_rmse"]}, "v1_reported": v1_free.get(s.lower(), {}), "constant_baseline_reported": v1_free.get(s.lower(), {})}
    write_json(OUT / "TRAJECTORY_FIDELITY.json", fidelity)
    dev_rows = []
    for b in bs:
        if b.split == "DEV":
            p, pr = rollout(model, b, xm, xs, ym, ys, device); dev_rows.append({"pred": p, "prob": pr, "success": b.success})
    evaluator = evaluator_freeze(dev_rows); write_json(OUT / "EVALUATOR_DEV_FREEZE.json", evaluator)
    front = []; branch_rows = []
    for b in bs:
        if b.split not in ["DEV", "TEST"]:
            continue
        p, pr = rollout(model, b, xm, xs, ym, ys, device)
        bad = bool(np.any(pr[:, 2] >= evaluator["event_probability_threshold"]) or np.max(np.linalg.norm(p[:, :3], axis=1)) > evaluator["relative_displacement_threshold_m"])
        branch_rows.append({"branch_id": b.branch_id, "context_id": b.context_id, "root_id": b.root_id, "task": b.task, "split": b.split, "force": b.force, "success": b.success, "pred_success": int(not bad), "direct_contact": b.direct_contact})
    bf = pd.DataFrame(branch_rows)
    for cid, g in bf.groupby("context_id"):
        true = float(g.loc[g.success == 1, "force"].min()) if (g.success == 1).any() else np.nan
        pred = float(g.loc[g.pred_success == 1, "force"].min()) if (g.pred_success == 1).any() else np.nan
        front.append({"context_id": cid, "root_id": g.root_id.iloc[0], "task": int(g.task.iloc[0]), "split": g.split.iloc[0], "true_frontier_N": true, "pred_frontier_N": pred, "exact_match": int(np.isfinite(true) and np.isfinite(pred) and true == pred), "under_force": int(np.isfinite(true) and np.isfinite(pred) and pred < true), "over_force": int(np.isfinite(true) and np.isfinite(pred) and pred > true)})
    write_csv(OUT / "FRONTIER_BRANCH_RESULTS.csv", branch_rows); write_csv(OUT / "GT_FRICTION_FRONTIER_SUMMARY.csv", front)
    slips = []
    for split in ["DEV", "TEST"]:
        ys_all, ps_all, yn, pn = [], [], [], []
        for b in bs:
            if b.split != split: continue
            p, pr = rollout(model, b, xm, xs, ym, ys, device); active = np.isin(np.asarray(b.phases), ["lift", "transit", "over_basket", "place"])[:len(pr)]
            ys_all.extend(b.events[:len(pr), 2][active]); ps_all.extend(pr[:, 2][active] >= .5)
            if b.weight > 1: yn.extend(b.events[:len(pr), 2][active]); pn.extend(pr[:, 2][active] >= .5)
        slips.append({"split": split, **binary(ys_all, ps_all), "near_frontier": binary(yn, pn) if yn else {}})
    write_json(OUT / "SLIP_CONTACT_RESULT.json", {"slip": slips, "event_definition": event_def})
    f = pd.DataFrame(front); dev = f[f.split == "DEV"]; test = f[f.split == "TEST"]; dslip = next(x for x in slips if x["split"] == "DEV")
    gate = {"dev_frontier_exact": float(dev.exact_match.mean()), "dev_under_force": float(dev.under_force.mean()), "dev_slip_f1": float(dslip["f1"]), "test_frontier_exact": float(test.exact_match.mean()), "test_under_force": float(test.under_force.mean()), "test_over_force": float(test.over_force.mean()), "criterion": "DEV exact >= 0.80; under-force <= 0.10; slip F1 >= 0.70", "pass": int(float(dev.exact_match.mean()) >= .80 and float(dev.under_force.mean()) <= .10 and float(dslip["f1"]) >= .70)}
    write_json(OUT / "GT_FRICTION_ACTION_GATE.json", gate)
    write_json(OUT / "FRONTIER_ERROR_ANALYSIS.json", {"earliest_error_categories": {"missed_slip_or_contact": int(sum(r["pred_success"] == 1 and r["success"] == 0 for r in branch_rows)), "false_slip_or_drift": int(sum(r["pred_success"] == 0 and r["success"] == 1 for r in branch_rows)), "other": 0}, "gate_pass": gate["pass"]})
    status = "GATE_3_PASSED" if gate["pass"] else "WORLD_MODEL_STILL_MISSES_FORCE_BOUNDARY_PHYSICS"
    (OUT / "REPORT.md").write_text(f"# Physics-GRU v2\n\nSTATUS: {status}\n\nDirect boundary contact branches: {data_meta['direct_contact_branches']}; all other historical branches use derived physical slip labels.\\n\\nDEV exact={gate['dev_frontier_exact']:.3f}, under={gate['dev_under_force']:.3f}, slip F1={gate['dev_slip_f1']:.3f}.\\nTEST exact={gate['test_frontier_exact']:.3f}, under={gate['test_under_force']:.3f}.\\n\\nEstimated-friction and real E2E are not run unless the unchanged Gate 3 passes.\\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "data": data_meta, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
