#!/usr/bin/env python3
"""Isolated, CPU-only early training run on completed prospective task0 TRAIN data.

This is deliberately diagnostic.  It never reads DEV outcomes, never launches
Isaac/π0, and never writes into the live Tabero experiment directory.
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
import re
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

sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
CONTEXT_CSV = SOURCE / "collection_train/task0/task0/context.csv"
BRANCH_CSV = SOURCE / "collection_train/task0/task0/branches.csv"
VISUAL_CSV = SOURCE / "collection_train/visual_alignment_worker.csv"
TPI_CODE = Path("/home/exouser/Tabero/analysis/trajectory_physical_imagination.py")
CF_CODE = Path("/home/exouser/Tabero/analysis/counterfactual_force_world_model.py")
FULL_CODE = Path("/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py")

H = 8
SEEDS = [0, 1, 2]
EPOCHS = 80
BATCH = 64
LR = 8e-4
WEIGHT_DECAY = 1e-4
LAMBDA_FEAS = 0.3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Meta:
    branch_id: str
    context_id: str
    task: int
    root_id: str
    friction_band: str
    outcome: int
    force: float
    repeat: int
    stratum: int


class VisualFullFeas(nn.Module):
    """Authoritative Feas backbone plus the smallest visual projection/fusion."""

    def __init__(self, visual_dim: int):
        super().__init__()
        self.command_gru = nn.GRU(17, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.visual = nn.Sequential(nn.Linear(visual_dim, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(144, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step, cond, visual):
        _, h = self.command_gru(step)
        z = torch.cat([h[-1], self.condition(cond), self.visual(visual)], dim=-1)
        return self.head(z).squeeze(-1)


class VisualResidual(nn.Module):
    """A visual-only intercept; force is intentionally absent."""

    def __init__(self, visual_dim: int):
        super().__init__()
        self.intercept = nn.Linear(visual_dim, 1)

    def forward(self, visual):
        return self.intercept(visual).squeeze(-1)


class VisualJoint(nn.Module):
    """Authoritative physical/IE model with minimal visual feasibility fusion."""

    def __init__(self, tpi, physics_state: dict[str, Any], visual_dim: int):
        super().__init__()
        self.physics = tpi.ShortHorizonPhysicsGRU(17, 54, H)
        self.physics.load_state_dict(physics_state)
        self.visual = nn.Sequential(nn.Linear(visual_dim, 16), nn.ReLU())
        self.feas_head = nn.Sequential(nn.Linear(80, 32), nn.ReLU(), nn.Linear(32, 1))

    def hidden(self, step, cond):
        c = cond[:, None, :].expand(-1, step.shape[1], -1)
        z, _ = self.physics.gru(torch.cat([step, c], dim=-1))
        return z[:, -1]

    def forward(self, step, cond, visual):
        z = self.hidden(step, cond)
        traj = self.physics.head(z).view(-1, H, 13)
        logit = self.feas_head(torch.cat([z, self.visual(visual)], dim=-1)).squeeze(-1)
        return traj, logit


def parse_cell(label: str) -> tuple[int, int]:
    sm = re.search(r"_S(\d+)_", label)
    rm = re.search(r"_R([12])(?:_|$)", label)
    if not sm or not rm:
        raise RuntimeError(f"cannot parse stratum/repeat: {label}")
    return int(sm.group(1)), int(rm.group(1))


def strict_preprobe_state(probe_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    d = pd.read_csv(probe_path)
    hold = d[d.probe_phase.astype(str) == "hold"]
    if hold.empty or int(hold.step.max()) != 190:
        raise RuntimeError(f"strict last-hold step mismatch: {probe_path}")
    r = hold.sort_values("step").iloc[-1]
    opening = float(r.gripper_opening)
    state = np.zeros(13, np.float32)
    mask = np.zeros(13, np.float32)
    state[11:13] = [opening, -opening]
    mask[:6] = 1.0
    mask[11:13] = 1.0
    return state, mask, {"last_hold_step": int(r.step), "gripper_opening": opening}


def audit_and_load(out: Path, tpi):
    contexts = pd.read_csv(CONTEXT_CSV)
    branches = pd.read_csv(BRANCH_CSV)
    visual_all = pd.read_csv(VISUAL_CSV)
    visual = visual_all[visual_all.context_id.isin(set(contexts.context_id.astype(str)))].copy()
    failures: list[str] = []
    if len(contexts) != 18 or contexts.context_id.nunique() != 18:
        failures.append(f"contexts={len(contexts)} unique={contexts.context_id.nunique()} expected=18")
    if len(branches) != 180 or branches.branch_id.nunique() != 180:
        failures.append(f"branches={len(branches)} unique={branches.branch_id.nunique()} expected=180")
    if len(visual) != 18 or visual.context_id.nunique() != 18:
        failures.append(f"visual={len(visual)} unique={visual.context_id.nunique()} expected=18")
    if set(contexts.task.astype(int)) != {0} or set(branches.task.astype(int)) != {0}:
        failures.append("non-task0 row present")
    if set(contexts.split.astype(str)) != {"TRAIN"} or set(branches.split.astype(str)) != {"TRAIN"}:
        failures.append("non-TRAIN row present")
    if int(branches.state_parity.sum()) != 180 or int(contexts.strict_matched.sum()) != 18:
        failures.append("state parity/strict match incomplete")
    if len(visual) and int(visual.restore_exact.sum()) != 18:
        failures.append("visual snapshot restore parity incomplete")

    vmap = visual.set_index("context_id").to_dict("index")
    cmap: dict[str, dict[str, Any]] = {}
    raw_features: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []
    for r in contexts.sort_values("context_id").itertuples(index=False):
        cid = str(r.context_id)
        q = branches[branches.context_id.astype(str) == cid]
        if len(q) != 10 or q.requested_force_N.nunique() != 5:
            failures.append(f"{cid}: expected 10 branches and 5 forces")
        parsed = [parse_cell(str(x)) for x in q.branch_label]
        if Counter(parsed) != Counter((s, rep) for s in range(5) for rep in (1, 2)):
            failures.append(f"{cid}: incomplete 5x2 force/repeat cells")
        vr = vmap.get(cid)
        if vr is None:
            failures.append(f"{cid}: missing visual row")
            continue
        if str(vr["snapshot_state_hash"]) != str(r.post_probe_state_hash):
            failures.append(f"{cid}: visual/context state hash mismatch")
        if str(vr["restored_state_hash"]) != str(vr["snapshot_state_hash"]):
            failures.append(f"{cid}: first restore hash mismatch")
        if str(vr["second_restore_hash"]) != str(vr["snapshot_state_hash"]):
            failures.append(f"{cid}: second restore hash mismatch")
        fp = Path(str(vr["visual_feature_path"]))
        x = np.load(fp, allow_pickle=False)
        raw_hash = hashlib.sha256(np.asarray(x, np.float32).tobytes()).hexdigest()
        if x.shape != (4096,) or x.dtype != np.float32:
            failures.append(f"{cid}: visual shape/dtype={x.shape}/{x.dtype}")
        if raw_hash != str(vr["visual_feature_sha256"]):
            failures.append(f"{cid}: visual raw-byte hash mismatch")
        state, mask, premeta = strict_preprobe_state(Path(str(r.probe_telemetry_path)))
        cmap[cid] = {
            "context_id": cid,
            "root_id": str(r.root_id),
            "task": int(r.task),
            "friction": float(r.hidden_friction_analysis_only),
            "friction_band": str(r.friction_band),
            "preprobe_state": state,
            "preprobe_mask": mask,
            "visual_raw": np.asarray(x, np.float32),
        }
        raw_features.append(np.asarray(x, np.float32))
        feature_rows.append({
            "context_id": cid,
            "root_id": str(r.root_id),
            "task": int(r.task),
            "visual_feature_path": str(fp),
            "visual_feature_raw_sha256": raw_hash,
            "snapshot_state_hash": str(vr["snapshot_state_hash"]),
            "restore_exact": int(vr["restore_exact"]),
            **premeta,
        })

    corrected = {
        "left_normal_force_N", "right_normal_force_N", "left_tangential_force_N",
        "right_tangential_force_N", "object_vx_mps", "object_vy_mps", "object_vz_mps",
    }
    traces: list[Any] = []
    meta: dict[str, Meta] = {}
    for r in branches.sort_values("branch_id").itertuples(index=False):
        cid = str(r.context_id)
        if cid not in cmap:
            continue
        path = Path(str(r.telemetry_path))
        d = pd.read_csv(path)
        if len(d) < H + 1 or not corrected <= set(d.columns):
            failures.append(f"{r.branch_id}: short or missing corrected telemetry")
            continue
        if not np.isfinite(d[list(corrected)].to_numpy(float)).all():
            failures.append(f"{r.branch_id}: nonfinite corrected telemetry")
            continue
        state, mask = tpi.state_from(d)
        state = state.copy()
        mask = mask.copy()
        state[0] = cmap[cid]["preprobe_state"]
        mask[0] = cmap[cid]["preprobe_mask"]
        force = float(r.requested_force_N)
        mu = float(r.hidden_friction_analysis_only)
        nominal = tpi.nominal_from(d, 0, force, mu, state, mask)
        tr = tpi.Trace(
            str(r.branch_id), cid, str(r.root_id), 0, "TRAIN", force, mu,
            int(r.full_task_success_y), "continuous", path, state, mask, nominal,
            d.phase.astype(str).tolist(), 1.0, "PROSPECTIVE_TASK0",
        )
        s, rep = parse_cell(str(r.branch_label))
        traces.append(tr)
        meta[tr.branch_id] = Meta(tr.branch_id, cid, 0, str(r.root_id), str(r.friction_band),
                                  int(r.full_task_success_y), force, rep, s)

    if len(traces) != 180:
        failures.append(f"valid traces={len(traces)} expected=180")
    if failures:
        write_json(out / "TASK0_DATA_AUDIT.json", {"status": "FAIL", "failures": failures})
        raise RuntimeError("task0 audit failed: " + "; ".join(failures[:5]))

    xraw = np.stack(raw_features).astype(np.float32)
    xmean = xraw.mean(0)
    _, singular, vt = np.linalg.svd(xraw - xmean, full_matrices=False)
    rank = min(64, len(cmap) - 1, int(np.sum(singular > singular[0] * 1e-7)))
    components = vt[:rank].astype(np.float32)
    xpca = ((xraw - xmean) @ components.T).astype(np.float32)
    pmean = xpca.mean(0).astype(np.float32)
    pstd = xpca.std(0).astype(np.float32)
    pstd[pstd < 1e-6] = 1.0
    for cid, z in zip(sorted(cmap), (xpca - pmean) / pstd):
        cmap[cid]["visual"] = z.astype(np.float32)

    write_csv(out / "TASK0_FROZEN_VISUAL_ALIGNMENT.csv", feature_rows)
    np.savez(out / "TASK0_PCA17_PROVISIONAL.npz", raw_mean=xmean, components=components,
             projected_mean=pmean, projected_std=pstd, singular_values=singular)
    write_json(out / "TASK0_DATA_AUDIT.json", {
        "status": "PASS", "scope": "task0 TRAIN only", "contexts": 18, "roots": int(contexts.root_id.nunique()),
        "branches": 180, "successes": int(branches.full_task_success_y.sum()),
        "failures": int(180 - branches.full_task_success_y.sum()), "five_forces_two_repeats_each": True,
        "state_parity": "180/180", "visual_alignment": "18/18", "snapshot_restore_exact": "18/18",
        "corrected_physical_telemetry": "180/180", "DEV_read": False, "TEST_read": False,
        "pca_requested_components": 64, "pca_effective_components": rank,
        "pca_deviation_reason": "18 task0 contexts imply centered PCA rank at most 17; full 64-D PCA awaits all 72 TRAIN contexts",
        "source_manifest_hashes": {"context.csv": sha256(CONTEXT_CSV), "branches.csv": sha256(BRANCH_CSV)},
    })
    return contexts, branches, cmap, traces, meta, rank


def build_pairs(cf, traces, meta):
    by: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for tr in traces:
        by[(tr.context_id, meta[tr.branch_id].repeat)].append(tr)
    pairs = []
    for (cid, repeat), q in sorted(by.items()):
        q = sorted(q, key=lambda tr: meta[tr.branch_id].stratum)
        if len(q) != 5:
            raise RuntimeError(f"pair construction expected 5 strata: {cid}/R{repeat}")
        for a, b in zip(q[:-1], q[1:]):
            ma, mb = meta[a.branch_id], meta[b.branch_id]
            pairs.append(cf.Pair(
                f"task0:{cid}:R{repeat}:S{ma.stratum}_vs_S{mb.stratum}",
                f"task0:{cid}:R{repeat}", "TRAIN", "PROSPECTIVE_TASK0", cid, a.root_id,
                0, ma.friction_band, a.mu, a.force, b.force, "adjacent", False, a, b,
            ))
    if len(pairs) != 144:
        raise RuntimeError(f"expected 144 adjacent pairs, got {len(pairs)}")
    return pairs


def build_segments_and_norm(cf, tpi, traces):
    segs = {tr.branch_id: cf.build_seg(tpi, tr, tr.force, H) for tr in traces}
    x = np.concatenate([segs[tr.branch_id].x for tr in traces], axis=0)
    xm = x.mean(0).astype(np.float32)
    xs = x.std(0).astype(np.float32)
    xs[xs < 1e-6] = 1.0
    physical_segments = []
    for tr in traces:
        physical_segments.extend(tpi.make_segments([tr], H))
    delta = np.concatenate([s.y - s.trace.state[s.start] for s in physical_segments], axis=0)
    ym = delta.mean(0).astype(np.float32)
    ys = delta.std(0).astype(np.float32)
    ys[ys < 1e-6] = 1.0
    return segs, (xm, xs, ym, ys)


def sample_weights(traces, meta):
    keys = [(meta[t.branch_id].friction_band, meta[t.branch_id].outcome, meta[t.branch_id].stratum)
            for t in traces]
    counts = Counter(keys)
    w = np.asarray([1.0 / counts[k] for k in keys], float)
    return w / w.sum()


def sampled_ids(traces, meta, seed, epoch, n=None):
    rng = np.random.default_rng(seed * 1000003 + epoch * 1009 + 97)
    return rng.choice(len(traces), size=n or len(traces), replace=True, p=sample_weights(traces, meta))


def branch_tensors(traces, segs, norm, cmap, device):
    xm, xs = norm[:2]
    x = np.stack([(segs[t.branch_id].x - xm) / xs for t in traces]).astype(np.float32)
    visual = np.stack([cmap[t.context_id]["visual"] for t in traces]).astype(np.float32)
    y = np.asarray([t.outcome for t in traces], np.float32)
    return (torch.tensor(x[:, :, :17], device=device), torch.tensor(x[:, 0, 17:], device=device),
            torch.tensor(visual, device=device), torch.tensor(y, device=device))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_base(full, traces, segs, norm, cmap, meta, device, seed):
    seed_everything(seed)
    model = full.FeasibilityOnly().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    hist, steps = [], 0
    for epoch in range(1, EPOCHS + 1):
        ids = sampled_ids(traces, meta, seed, epoch)
        losses = []
        model.train()
        for st in range(0, len(ids), BATCH):
            q = [traces[int(i)] for i in ids[st:st+BATCH]]
            step, cond, _, y = branch_tensors(q, segs, norm, cmap, device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(step, cond), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            steps += 1
        hist.append({"variant": "PROSPECTIVE_BASE_FEAS", "seed": seed, "epoch": epoch,
                     "train_bce": float(np.mean(losses)), "optimizer_steps": steps})
        if epoch % 20 == 0:
            print(f"[BASE] seed={seed} epoch={epoch}/{EPOCHS} bce={np.mean(losses):.6f}", flush=True)
    model.eval()
    return model, hist, steps


def train_residual(base, traces, segs, norm, cmap, meta, device, seed, visual_dim):
    seed_everything(seed + 1000)
    for p in base.parameters():
        p.requires_grad = False
    base.eval()
    model = VisualResidual(visual_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    hist, steps = [], 0
    for epoch in range(1, EPOCHS + 1):
        ids = sampled_ids(traces, meta, seed + 10, epoch)
        losses = []
        model.train()
        for st in range(0, len(ids), BATCH):
            q = [traces[int(i)] for i in ids[st:st+BATCH]]
            step, cond, visual, y = branch_tensors(q, segs, norm, cmap, device)
            with torch.no_grad():
                base_logit = base(step, cond)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(base_logit + model(visual), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            steps += 1
        hist.append({"variant": "VISUAL_INTERCEPT_RESIDUAL", "seed": seed, "epoch": epoch,
                     "train_bce": float(np.mean(losses)), "optimizer_steps": steps,
                     "force_input_to_residual": False})
        if epoch % 20 == 0:
            print(f"[RESIDUAL] seed={seed} epoch={epoch}/{EPOCHS} bce={np.mean(losses):.6f}", flush=True)
    model.eval()
    return model, hist, steps


def train_full(traces, segs, norm, cmap, meta, device, seed, visual_dim):
    seed_everything(seed + 2000)
    model = VisualFullFeas(visual_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    hist, steps = [], 0
    for epoch in range(1, EPOCHS + 1):
        ids = sampled_ids(traces, meta, seed + 20, epoch)
        losses = []
        model.train()
        for st in range(0, len(ids), BATCH):
            q = [traces[int(i)] for i in ids[st:st+BATCH]]
            step, cond, visual, y = branch_tensors(q, segs, norm, cmap, device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(step, cond, visual), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            steps += 1
        hist.append({"variant": "VISUAL_CONTEXT_FULL_FEAS", "seed": seed, "epoch": epoch,
                     "train_bce": float(np.mean(losses)), "optimizer_steps": steps})
        if epoch % 20 == 0:
            print(f"[FULL] seed={seed} epoch={epoch}/{EPOCHS} bce={np.mean(losses):.6f}", flush=True)
    model.eval()
    return model, hist, steps


def physical_ie_loss(pred, batch, norm, device):
    losses = []
    offset = 0
    for unit in batch:
        if unit[1] is None:
            offset += 1
            continue
        sa, sb, pair = unit
        pa, pb = pred[offset], pred[offset + 1]
        phases = np.asarray([str(x) in {"branch_hold", "lift", "transit", "over_basket", "place"}
                             for x in pair.a.phase[1:H+1]], bool)
        mask = torch.tensor((sa.mask * sb.mask) * phases[:, None], dtype=torch.float32, device=device)
        ya = (sa.y - sa.trace.state[0] - norm[2]) / norm[3]
        yb = (sb.y - sb.trace.state[0] - norm[2]) / norm[3]
        target = torch.tensor(yb - ya, dtype=torch.float32, device=device)
        losses.append((nn.functional.smooth_l1_loss(pb - pa, target, reduction="none") * mask * 2.0).sum()
                      / (mask.sum() + 1e-6))
        offset += 2
    return torch.stack(losses).mean() if losses else torch.zeros((), device=device)


def train_joint(full, cf, tpi, traces, pairs, segs, norm, cmap, meta, device, seed, visual_dim):
    seed_everything(seed + 3000)
    base_ck, _, base_path = full.load_base(tpi, seed, device)
    model = VisualJoint(tpi, base_ck["state_dict"], visual_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    units = cf.make_units(tpi, traces, pairs)
    hist, steps = [], 0
    for epoch in range(1, EPOCHS + 1):
        batches = cf.batches_for_units(units, seed, epoch)
        fids = sampled_ids(traces, meta, seed + 100, epoch, n=len(batches) * BATCH)
        bases, ies, feass, totals = [], [], [], []
        model.train()
        for bi, batch in enumerate(batches):
            batch_segs, step, cond, ytraj, mask, weight = cf.batch_tensors(batch, norm, device)
            bvisual = torch.tensor(np.stack([cmap[s.trace.context_id]["visual"] for s in batch_segs]),
                                   dtype=torch.float32, device=device)
            fq = [traces[int(i)] for i in fids[bi*BATCH:(bi+1)*BATCH]]
            fstep, fcond, fvisual, fy = branch_tensors(fq, segs, norm, cmap, device)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(step, cond, bvisual)
            _, flogit = model(fstep, fcond, fvisual)
            base = (nn.functional.smooth_l1_loss(pred, ytraj, reduction="none") * mask * weight[:, None, None]).sum() / (mask.sum() + 1e-6)
            ie = physical_ie_loss(pred, batch, norm, device)
            feas = nn.functional.binary_cross_entropy_with_logits(flogit, fy)
            total = base + ie + LAMBDA_FEAS * feas
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            bases.append(float(base.item()))
            ies.append(float(ie.item()))
            feass.append(float(feas.item()))
            totals.append(float(total.item()))
        hist.append({"variant": "VISUAL_CONTEXT_JOINT", "seed": seed, "epoch": epoch,
                     "trajectory_loss": float(np.mean(bases)), "ie_loss": float(np.mean(ies)),
                     "feasibility_loss": float(np.mean(feass)), "native_total_loss": float(np.mean(totals)),
                     "optimizer_steps": steps, "physical_units": len(units)})
        if epoch % 10 == 0:
            print(f"[JOINT] seed={seed} epoch={epoch}/{EPOCHS} traj={np.mean(bases):.6f} ie={np.mean(ies):.6f} feas={np.mean(feass):.6f}", flush=True)
    model.eval()
    return model, hist, steps, len(units), base_path


def logits_for(model, kind, traces, segs, norm, cmap, device, base=None):
    ans = {}
    model.eval()
    with torch.no_grad():
        for st in range(0, len(traces), 256):
            q = traces[st:st+256]
            step, cond, visual, _ = branch_tensors(q, segs, norm, cmap, device)
            if kind == "BASE":
                z = model(step, cond)
            elif kind == "RESIDUAL":
                z = base(step, cond) + model(visual)
            elif kind == "FULL":
                z = model(step, cond, visual)
            else:
                z = model(step, cond, visual)[1]
            ans.update({tr.branch_id: float(v) for tr, v in zip(q, z.cpu().numpy())})
    return ans


def train_metrics(kind, seed, traces, logits):
    y = np.asarray([tr.outcome for tr in traces], float)
    z = np.asarray([logits[tr.branch_id] for tr in traces], float)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    nll = -np.mean(y * np.log(np.clip(p, 1e-8, 1)) + (1-y) * np.log(np.clip(1-p, 1e-8, 1)))
    return {"variant": kind, "seed": seed, "scope": "TRAIN_IN_SAMPLE_ONLY",
            "n": len(y), "positive_rate": float(y.mean()), "bce_nll": float(nll),
            "brier": float(np.mean((p-y)**2)), "accuracy_0.5": float(np.mean((p >= .5) == y)),
            "signed_bias": float(np.mean(p-y)), "DEV_used": False}


def checkpoint(path, model, variant, seed, norm, pca_path, extra=None):
    obj = {"state_dict": model.state_dict(), "variant": variant, "seed": seed, "task": 0,
           "scope": "TASK0_TRAIN_ONLY_PROVISIONAL", "epochs": EPOCHS, "device": "cpu",
           "normalization": {k: v.tolist() for k, v in zip(["x_mean", "x_std", "y_mean", "y_std"], norm)},
           "pca_path": str(pca_path), "pca_sha256": sha256(pca_path), "DEV_used": False, "TEST_used": False}
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def write_notebook(out: Path, summary: dict[str, Any]) -> None:
    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}
    def code(source, output):
        return {"cell_type": "code", "execution_count": 1, "metadata": {},
                "source": [line + "\n" for line in source.splitlines()],
                "outputs": [{"output_type": "stream", "name": "stdout", "text": [output + "\n"]}]}
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": sys.version.split()[0]}},
          "cells": [
              md("# task0 visual-context early run\n\n## tl;dr\nThis executed artifact records the isolated task0 TRAIN-only run. It is not a DEV result."),
              md("## Context & Methods\nRead-only source: the completed prospective task0 collection. Training used CPU only, 3 seeds, 80 epochs, and four preregistered model families."),
              code("import json\nsummary = json.load(open('TASK0_RUN_SUMMARY.json'))\nprint(json.dumps(summary, indent=2))", json.dumps(summary, indent=2)),
              md("## Data\nThe data audit verifies 18 contexts, 180 branches, 5 forces × 2 repeats, strict state parity, visual alignment, and corrected telemetry."),
              code("audit = json.load(open('TASK0_DATA_AUDIT.json'))\nprint(json.dumps(audit, indent=2))", (out / "TASK0_DATA_AUDIT.json").read_text()),
              md("## Results\nMetrics are in-sample TRAIN diagnostics only; they cannot support model selection or the final visual-context claim."),
              code("import pandas as pd\nmetrics = pd.read_csv('TASK0_TRAIN_IN_SAMPLE_METRICS.csv')\nprint(metrics.to_string(index=False))", pd.read_csv(out / "TASK0_TRAIN_IN_SAMPLE_METRICS.csv").to_string(index=False)),
              md("## Takeaways\nAll checkpoints are provisional. Full 64-D TRAIN-only PCA, held-out DEV curves, safety/frontier metrics, and scientific classification remain pending."),
          ]}
    write_json(out / "TASK0_EARLY_RUN.ipynb", nb)


def run(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    tpi = load_module("tpi_task0_early", TPI_CODE)
    cf = load_module("cf_task0_early", CF_CODE)
    full = load_module("full_task0_early", FULL_CODE)
    contexts, branches, cmap, traces, meta, visual_dim = audit_and_load(out, tpi)
    pairs = build_pairs(cf, traces, meta)
    segs, norm = build_segments_and_norm(cf, tpi, traces)
    norm_path = out / "TASK0_TRAIN_NORMALIZATION.npz"
    np.savez(norm_path, x_mean=norm[0], x_std=norm[1], y_mean=norm[2], y_std=norm[3])
    protocol = {
        "status": "TASK0_ONLY_PROVISIONAL_PROTOCOL", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE), "task": 0, "split": "TRAIN", "contexts": 18, "branches": 180,
        "models": ["PROSPECTIVE_BASE_FEAS", "VISUAL_INTERCEPT_RESIDUAL", "VISUAL_CONTEXT_FULL_FEAS", "VISUAL_CONTEXT_JOINT"],
        "seeds": SEEDS, "epochs": EPOCHS, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
        "loss": "BCEWithLogits; Joint additionally authoritative trajectory + IE with lambda_feas=0.3",
        "device": "CPU_ONLY", "torch_threads": 1, "launches_Isaac_or_pi0": False,
        "DEV_used": False, "TEST_used": False, "visual_pca_components": visual_dim,
        "deviation_from_full_protocol": "task0-only rank-limited PCA; not eligible for final comparison",
        "source_code_hashes": {str(p): sha256(p) for p in [TPI_CODE, CF_CODE, FULL_CODE, Path(__file__).resolve()]},
    }
    write_json(out / "TASK0_EARLY_PROTOCOL.json", protocol)
    histories = defaultdict(list)
    metrics, manifest = [], []
    for seed in SEEDS:
        print(f"[seed] {seed} starting base/residual/full", flush=True)
        base, bh, bsteps = train_base(full, traces, segs, norm, cmap, meta, device, seed)
        bp = out / f"PROSPECTIVE_BASE_FEAS_task0_seed{seed}.pt"
        checkpoint(bp, base, "PROSPECTIVE_BASE_FEAS", seed, norm, out / "TASK0_PCA17_PROVISIONAL.npz")
        residual, rh, rsteps = train_residual(base, traces, segs, norm, cmap, meta, device, seed, visual_dim)
        rp = out / f"VISUAL_INTERCEPT_RESIDUAL_task0_seed{seed}.pt"
        checkpoint(rp, residual, "VISUAL_INTERCEPT_RESIDUAL", seed, norm, out / "TASK0_PCA17_PROVISIONAL.npz",
                   {"base_checkpoint": str(bp), "base_checkpoint_sha256": sha256(bp), "force_input_to_residual": False})
        full_model, fh, fsteps = train_full(traces, segs, norm, cmap, meta, device, seed, visual_dim)
        fp = out / f"VISUAL_CONTEXT_FULL_FEAS_task0_seed{seed}.pt"
        checkpoint(fp, full_model, "VISUAL_CONTEXT_FULL_FEAS", seed, norm, out / "TASK0_PCA17_PROVISIONAL.npz")
        histories["PROSPECTIVE_BASE_FEAS"].extend(bh)
        histories["VISUAL_INTERCEPT_RESIDUAL"].extend(rh)
        histories["VISUAL_CONTEXT_FULL_FEAS"].extend(fh)
        for kind, model, path, steps, bmodel in [
            ("PROSPECTIVE_BASE_FEAS", base, bp, bsteps, None),
            ("VISUAL_INTERCEPT_RESIDUAL", residual, rp, rsteps, base),
            ("VISUAL_CONTEXT_FULL_FEAS", full_model, fp, fsteps, None),
        ]:
            lk = "BASE" if kind.startswith("PROSPECTIVE") else "RESIDUAL" if "RESIDUAL" in kind else "FULL"
            logits = logits_for(model, lk, traces, segs, norm, cmap, device, base=bmodel)
            metrics.append(train_metrics(kind, seed, traces, logits))
            manifest.append({"variant": kind, "seed": seed, "checkpoint": str(path),
                             "sha256": sha256(path), "optimizer_steps": steps})

        print(f"[seed] {seed} starting joint", flush=True)
        joint, jh, jsteps, units, initial = train_joint(full, cf, tpi, traces, pairs, segs, norm, cmap, meta,
                                                       device, seed, visual_dim)
        jp = out / f"VISUAL_CONTEXT_JOINT_task0_seed{seed}.pt"
        checkpoint(jp, joint, "VISUAL_CONTEXT_JOINT", seed, norm, out / "TASK0_PCA17_PROVISIONAL.npz",
                   {"initial_physics_checkpoint": str(initial), "initial_physics_checkpoint_sha256": sha256(initial),
                    "physical_units": units, "adjacent_ie_pairs": len(pairs), "lambda_feas": LAMBDA_FEAS})
        histories["VISUAL_CONTEXT_JOINT"].extend(jh)
        metrics.append(train_metrics("VISUAL_CONTEXT_JOINT", seed, traces,
                                     logits_for(joint, "JOINT", traces, segs, norm, cmap, device)))
        manifest.append({"variant": "VISUAL_CONTEXT_JOINT", "seed": seed, "checkpoint": str(jp),
                         "sha256": sha256(jp), "optimizer_steps": jsteps, "physical_units": units})
        print(f"[seed] {seed} complete", flush=True)

    for variant, rows in histories.items():
        write_csv(out / f"{variant}_TASK0_TRAINING_MANIFEST.csv", rows)
    write_csv(out / "TASK0_TRAIN_IN_SAMPLE_METRICS.csv", metrics)
    write_json(out / "ALL_PROSPECTIVE_VISUAL_MODELS_FROZEN_TASK0_ONLY.json", {
        "status": "TASK0_ONLY_PROVISIONAL_12_FROZEN", "checkpoint_count": len(manifest),
        "checkpoints": manifest, "DEV_used": False, "TEST_used": False,
        "eligible_for_final_model_selection": False,
    })
    summary = {
        "status": "COMPLETE_TASK0_ONLY_PROVISIONAL", "contexts": 18, "branches": 180,
        "successes": int(branches.full_task_success_y.sum()), "failures": int(180-branches.full_task_success_y.sum()),
        "checkpoints": len(manifest), "device": "cpu", "visual_pca_components": visual_dim,
        "DEV_used": False, "final_scientific_classification": "NOT_PERMITTED_WITH_TRAIN_ONLY_TASK0",
    }
    write_json(out / "TASK0_RUN_SUMMARY.json", summary)
    write_notebook(out, summary)
    hash_rows = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            hash_rows.append(f"{sha256(p)}  {p.name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(hash_rows) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(args.out.resolve())


if __name__ == "__main__":
    main()
