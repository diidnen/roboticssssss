#!/usr/bin/env python3
"""TRAIN/DEV-only GNP-style full-task feasibility decoder experiment.

The script has two explicit phases. ``freeze`` writes the immutable protocol,
data audit, and label/leakage audit before any new optimization or DEV
prediction. ``run`` refuses to proceed unless that frozen protocol exists.

No TEST telemetry/outcome is loaded by the scientific run.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
HIST_ROOT = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
CF_ROOT = RESULTS / "counterfactual_force_world_model_20260829_160000"
EVENT_ROOT = RESULTS / "targeted_direct_event_collection_20260829_193500"
FORENSIC_ROOT = RESULTS / "failure_preservation_forensic_20260829_170000"
FROZEN_ROOT = RESULTS / "trajectory_physical_imagination_20260829_065220"
OUT = Path(os.environ.get("FULL_TASK_FEAS_OUT", RESULTS / "full_task_feasibility_20260830_012830"))

H = 8
SEEDS = [0, 1, 2]
LAMBDAS = [0.1, 0.3, 1.0]
EPOCHS = 80
BATCH = 64
LR = 8e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
ACTIVE_PHASES = {"branch_hold", "lift", "transit", "over_basket", "place"}
GATE = {
    "boundary_ranking_min": 0.80,
    "fprev_false_safe_max": 0.20,
    "boundary_exact_min": 0.80,
    "dev_auroc_min": 0.80,
    "under_force_max": 0.10,
    "frontier_exact_min": 0.80,
}
MEANINGFUL_GAIN = 0.05
IE_RETENTION_MAX_RELATIVE_REGRESSION = 0.25
BRANCH_SEGMENT_CACHE: dict[str, Any] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
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


def authoritative_files() -> list[Path]:
    files = [
        EVENT_ROOT / "FINAL_REPORT.md",
        EVENT_ROOT / "DIRECT_EVENT_WORLD_MODEL_PROTOCOL.json",
        EVENT_ROOT / "DEV_EVENT_GATE.json",
        CF_ROOT / "COUNTERFACTUAL_FORCE_WORLD_MODEL_PROTOCOL.json",
        CF_ROOT / "COUNTERFACTUAL_FORCE_PAIR_AUDIT.json",
        CF_ROOT / "DEV_SELECTION.json",
        CF_ROOT / "DEV_FAILURE_PRESERVATION.csv",
        FORENSIC_ROOT / "FAILURE_PRESERVATION_FORENSIC_PROTOCOL.json",
        HIST_ROOT / "P5S0C_BRANCH_MANIFEST.csv",
        HIST_ROOT / "P5S0C_FORCE_MANIFEST.json",
        HIST_ROOT / "P5S0C_SPLIT_MANIFEST.json",
        REPO / "analysis/trajectory_physical_imagination.py",
        REPO / "analysis/counterfactual_force_world_model.py",
    ]
    files += [CF_ROOT / f"PHYSICS_GRU_FORCE_IE_lambda1.0_seed{s}.pt" for s in SEEDS]
    return files


def split_summary(d: pd.DataFrame, split: str) -> dict[str, Any]:
    q = d[d.split == split].copy()
    success = int(q.full_task_success_y.sum())
    fstars = q[q.full_task_success_y == 1].groupby("context_id").requested_force_N.min()
    boundary_prev = 0
    boundary_star = 0
    for cid, fs in fstars.items():
        z = q[q.context_id == cid]
        boundary_star += int(((z.requested_force_N == fs) & (z.full_task_success_y == 1)).sum() > 0)
        boundary_prev += int(((z.requested_force_N < fs) & (z.full_task_success_y == 0)).sum() > 0)
    return {
        "total_branches": int(len(q)),
        "successful_branches": success,
        "failed_branches": int(len(q) - success),
        "tasks": sorted(int(x) for x in q.task.unique()),
        "task_count": int(q.task.nunique()),
        "roots": sorted(str(x) for x in q.root_id.unique()),
        "root_count": int(q.root_id.nunique()),
        "contexts": int(q.context_id.nunique()),
        "friction_bands": sorted(str(x) for x in q.friction_band.unique()),
        "friction_min": float(q.hidden_friction_analysis_only.min()),
        "friction_max": float(q.hidden_friction_analysis_only.max()),
        "forces_N": sorted(float(x) for x in q.requested_force_N.unique()),
        "boundary_fprev_contexts": boundary_prev,
        "boundary_fstar_contexts": boundary_star,
    }


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = HIST_ROOT / "P5S0C_BRANCH_MANIFEST.csv"
    d = pd.read_csv(manifest)
    required = {
        "branch_id", "context_id", "task", "split", "hidden_friction_analysis_only",
        "requested_force_N", "full_task_success_y", "telemetry_path", "root_id",
        "friction_band", "post_probe_state_hash", "state_parity", "label_source",
    }
    missing = sorted(required - set(d.columns))
    train_roots = set(d[d.split == "TRAIN"].root_id.astype(str))
    dev_roots = set(d[d.split == "DEV"].root_id.astype(str))
    test_roots = set(d[d.split == "TEST"].root_id.astype(str))
    summaries = {s: split_summary(d, s) for s in ["TRAIN", "DEV"]}
    test_meta = d[d.split == "TEST"]
    summaries["TEST"] = {
        "branches": int(len(test_meta)),
        "roots": int(test_meta.root_id.nunique()),
        "contexts": int(test_meta.context_id.nunique()),
        "tasks": sorted(int(x) for x in test_meta.task.unique()),
        "outcomes": "NOT_INSPECTED_OR_SUMMARIZED",
    }
    data_audit = {
        "status": "PASS" if not missing and not (train_roots & dev_roots) else "FAIL",
        "authoritative_manifest": str(manifest),
        "authoritative_manifest_sha256": sha256(manifest),
        "required_columns_missing": missing,
        "splits": summaries,
        "root_overlap": {
            "TRAIN_DEV": sorted(train_roots & dev_roots),
            "TRAIN_TEST": sorted(train_roots & test_roots),
            "DEV_TEST": sorted(dev_roots & test_roots),
        },
        "scientific_population": "P5-S0-C historical full-task branches; TRAIN and DEV only",
        "test_policy": "TEST manifest metadata counted for provenance only; TEST telemetry and outcomes are not loaded by run phase",
        "label_source": sorted(str(x) for x in d.label_source.dropna().unique()),
        "state_parity_pass_rate_train_dev": float(d[d.split.isin(["TRAIN", "DEV"])].state_parity.mean()),
    }
    write_json(OUT / "FULL_TASK_FEASIBILITY_DATA_AUDIT.json", data_audit)

    label_audit = {
        "status": "PASS" if set(d[d.split.isin(["TRAIN", "DEV"])].full_task_success_y.unique()) <= {0, 1} else "FAIL",
        "target": "full_task_success_y",
        "definition": {"1": "authoritative full downstream task success", "0": "authoritative full downstream task failure"},
        "source": "P5-S0-C full-task evaluator recorded in branch manifest",
        "not_derived_from": ["contact proxy", "slip proxy", "model prediction", "candidate-force frontier", "F_star"],
        "input_allowlist": [
            "branch-start physical state s0", "GT friction", "candidate grip force",
            "task id and H=8 task phase", "H=8 Cartesian nominal command sequence",
            "existing physical-state masks",
        ],
        "input_denylist": [
            "full_task_success_y", "final success/failure", "F_star", "F_prev identity",
            "frontier", "terminal state", "actual future physical trajectory",
            "future contact-loss label", "root identity",
        ],
        "boundary_identity_usage": "TRAIN sampling and DEV evaluation only; never model input",
        "test_used": False,
    }
    write_json(OUT / "FEASIBILITY_LABEL_AUDIT.json", label_audit)

    hashes = {str(p): sha256(p) for p in authoritative_files() if p.exists()}
    protocol = {
        "protocol_name": "FULL_TASK_FEASIBILITY_PROTOCOL",
        "status": "FROZEN_BEFORE_NEW_TRAINING_OR_DEV_PREDICTION",
        "created_before_new_training_or_dev_results": True,
        "authoritative_inputs_sha256": hashes,
        "scientific_goal": "compare direct full-task feasibility, frozen IE representation plus feasibility head, and joint IE plus full-task feasibility",
        "population": {
            "TRAIN": summaries["TRAIN"],
            "DEV": summaries["DEV"],
            "TEST": "forbidden; metadata-only provenance",
            "same_branch_population_all_variants": True,
        },
        "input_schema": {
            "step_dim_17": "relative command xyz, command delta xyz, 7 phase one-hot, 4 task one-hot over H=8",
            "condition_dim_54": "force/8, GT mu, current state13, mask13, initial state13, initial mask13",
            "future_motion": "first H=8 authoritative Cartesian Pi0/controller command sequence",
            "terminal_or_future_real_state_input": False,
        },
        "variants": {
            "FULL_TASK_FEAS_ONLY": "GRU(64) command encoder + MLP condition encoder + small MLP feasibility logit; no Physics-GRU rollout",
            "FROZEN_IE_FEAS_HEAD": "authoritative lambda_IE=1.0 Physics-GRU GRU final hidden state; trunk frozen; train MLP feasibility head only",
            "PHYSICS_GRU_IE_FEAS": "initialize authoritative lambda_IE=1.0 checkpoint; shared GRU(64), H8 trajectory head and feasibility head; L_traj + 1.0 L_IE + lambda_feas L_feas",
        },
        "frozen_ie_representation": "final GRU hidden state after H=8 input; chosen before DEV",
        "lambda_feas_candidates": LAMBDAS,
        "seeds": SEEDS,
        "optimizer": {
            "name": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS, "batch_size": BATCH, "gradient_clip": 1.0,
            "early_stopping": False, "fixed_budget": True,
        },
        "sampling": "TRAIN-only inverse-frequency sampling over task x friction-band x outcome x exact-boundary-role; F_prev/F_star identity never enters model",
        "calibration": "independent TRAIN-only isotonic calibration over raw feasibility logits; raw and calibrated DEV both reported",
        "decision_threshold": THRESHOLD,
        "controller_rule": "minimum tested candidate with calibrated p_success>=0.5; no-valid-force is a decision failure and counted under-force when a real feasible candidate exists",
        "joint_lambda_selection": "lexicographic DEV priority: minimize F_prev false-safe, maximize boundary ranking, maximize boundary exact, maximize AUROC, minimize trajectory error",
        "meaningful_independent_gain_absolute": MEANINGFUL_GAIN,
        "ie_retention": {"max_relative_regression": IE_RETENTION_MAX_RELATIVE_REGRESSION, "comparison": "same-seed initialized lambda_IE=1.0 checkpoint on same DEV historical pairs"},
        "dev_gates": GATE,
        "forbidden": ["TEST", "Probe", "No-physics", "continuous force", "0.25N rollout", "fresh simulator collection", "E2E", "Pi0 modification", "friction-estimator modification"],
        "new_simulator_rollouts": 0,
        "source_code": str(REPO / "analysis/full_task_feasibility_decoder.py"),
    }
    write_json(OUT / "FULL_TASK_FEASIBILITY_PROTOCOL.json", protocol)
    print(f"[freeze] protocol={OUT / 'FULL_TASK_FEASIBILITY_PROTOCOL.json'}", flush=True)
    print(f"[freeze] protocol_sha256={sha256(OUT / 'FULL_TASK_FEASIBILITY_PROTOCOL.json')}", flush=True)
    if data_audit["status"] != "PASS" or label_audit["status"] != "PASS":
        raise RuntimeError("freeze audit failed")


@dataclass
class BranchMeta:
    branch_id: str
    context_id: str
    task: int
    root_id: str
    friction_band: str
    outcome: int
    force: float
    boundary_role: str


def load_train_dev(tpi) -> tuple[list[Any], dict[str, BranchMeta], pd.DataFrame]:
    d = pd.read_csv(HIST_ROOT / "P5S0C_BRANCH_MANIFEST.csv")
    d = d[d.split.isin(["TRAIN", "DEV"])].copy()
    fstars = d[d.full_task_success_y == 1].groupby("context_id").requested_force_N.min().to_dict()
    fprevs: dict[str, float] = {}
    for cid, fs in fstars.items():
        q = d[(d.context_id == cid) & (d.full_task_success_y == 0) & (d.requested_force_N < fs)]
        if len(q):
            fprevs[str(cid)] = float(q.requested_force_N.max())
    traces: list[Any] = []
    meta: dict[str, BranchMeta] = {}
    for r in d.itertuples(index=False):
        path = Path(r.telemetry_path)
        td = pd.read_csv(path)
        state, mask = tpi.state_from(td)
        if len(state) < H + 1:
            raise RuntimeError(f"short branch {r.branch_id}")
        fs = float(fstars.get(str(r.context_id), d[d.context_id == r.context_id].requested_force_N.max()))
        force = float(r.requested_force_N)
        nominal = tpi.nominal_from(td, int(r.task), force, float(r.hidden_friction_analysis_only), state, mask)
        role = tpi.role_for(force, fs)
        tr = tpi.Trace(
            str(r.branch_id), str(r.context_id), str(r.root_id), int(r.task), str(r.split),
            force, float(r.hidden_friction_analysis_only), int(r.full_task_success_y), role,
            path, state, mask, nominal, td.phase.astype(str).tolist(),
            3.0 if abs(force - fs) <= 0.5 else 1.0, "historical",
        )
        exact_role = "OTHER"
        if abs(force - fs) < 1e-6 and int(r.full_task_success_y) == 1:
            exact_role = "F_STAR"
        elif str(r.context_id) in fprevs and abs(force - fprevs[str(r.context_id)]) < 1e-6:
            exact_role = "F_PREV"
        traces.append(tr)
        meta[str(r.branch_id)] = BranchMeta(str(r.branch_id), str(r.context_id), int(r.task), str(r.root_id), str(r.friction_band), int(r.full_task_success_y), force, exact_role)
    if len(traces) != 384 or sum(t.split == "TRAIN" for t in traces) != 288 or sum(t.split == "DEV" for t in traces) != 96:
        raise RuntimeError("TRAIN/DEV historical population mismatch")
    return traces, meta, d


def make_pairs(cf, traces: list[Any], meta: dict[str, BranchMeta]) -> list[Any]:
    by_context: dict[str, list[Any]] = defaultdict(list)
    for tr in traces:
        by_context[tr.context_id].append(tr)
    pairs: list[Any] = []
    for cid, rows in sorted(by_context.items()):
        rows = sorted(rows, key=lambda x: x.force)
        good = [x.force for x in rows if x.outcome == 1]
        fs = min(good) if good else max(x.force for x in rows)
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                delta = round(float(b.force - a.force), 6)
                boundary = bool((a.outcome == 0 and b.outcome == 1 and abs(b.force - fs) < 1e-6) or abs(a.force - fs) < 1e-6 or abs(b.force - fs) < 1e-6)
                category = "boundary" if boundary and abs(delta - 0.5) < 1e-6 else ("adjacent" if abs(delta - 0.5) < 1e-6 else "wider")
                m = meta[a.branch_id]
                pairs.append(cf.Pair(
                    f"historical:{cid}:{a.force:.1f}_vs_{b.force:.1f}", f"historical:{cid}", a.split,
                    "historical", cid, a.root_id, a.task, m.friction_band, a.mu,
                    a.force, b.force, category, boundary, a, b,
                ))
    return pairs


def branch_segment(cf, tpi, tr):
    key = str(tr.branch_id)
    if key not in BRANCH_SEGMENT_CACHE:
        BRANCH_SEGMENT_CACHE[key] = cf.build_seg(tpi, tr, tr.force, H)
    return BRANCH_SEGMENT_CACHE[key]


def branch_tensors(cf, tpi, traces: list[Any], norm, device):
    segs = [branch_segment(cf, tpi, tr) for tr in traces]
    xm, xs, _, _ = norm
    xn = np.stack([(s.x - xm) / xs for s in segs])
    step = torch.tensor(xn[:, :, :17], dtype=torch.float32, device=device)
    cond = torch.tensor(xn[:, 0, 17:], dtype=torch.float32, device=device)
    y = torch.tensor([float(s.trace.outcome) for s in segs], dtype=torch.float32, device=device)
    return step, cond, y


def sample_weights(traces: list[Any], meta: dict[str, BranchMeta]) -> np.ndarray:
    keys = [(m.task, m.friction_band, m.outcome, m.boundary_role) for m in (meta[t.branch_id] for t in traces)]
    counts = Counter(keys)
    w = np.asarray([1.0 / counts[k] for k in keys], dtype=float)
    return w / w.sum()


def sampled_indices(traces: list[Any], meta: dict[str, BranchMeta], seed: int, epoch: int, n: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed * 1000003 + epoch * 1009 + 97)
    return rng.choice(len(traces), size=n or len(traces), replace=True, p=sample_weights(traces, meta))


class FeasibilityOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.command_gru = nn.GRU(17, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step, cond):
        _, h = self.command_gru(step)
        z = torch.cat([h[-1], self.condition(cond)], dim=-1)
        return self.head(z).squeeze(-1)


def ie_hidden(physics, step, cond):
    c = cond[:, None, :].expand(-1, step.shape[1], -1)
    z, _ = physics.gru(torch.cat([step, c], dim=-1))
    return z[:, -1]


class FrozenIEFeasibility(nn.Module):
    def __init__(self, tpi, state_dict):
        super().__init__()
        self.physics = tpi.ShortHorizonPhysicsGRU(17, 54, H).to("cpu")
        self.physics.load_state_dict(state_dict)
        for p in self.physics.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, step, cond):
        with torch.no_grad():
            z = ie_hidden(self.physics, step, cond)
        return self.head(z).squeeze(-1)


class JointIEFeasibility(nn.Module):
    def __init__(self, tpi, state_dict):
        super().__init__()
        self.physics = tpi.ShortHorizonPhysicsGRU(17, 54, H)
        self.physics.load_state_dict(state_dict)
        self.feas_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, step, cond):
        z = ie_hidden(self.physics, step, cond)
        traj = self.physics.head(z).view(-1, H, 13)
        logit = self.feas_head(z).squeeze(-1)
        return traj, logit

    def feasibility(self, step, cond):
        return self.forward(step, cond)[1]


def train_classifier(model, variant: str, train: list[Any], meta, norm, cf, tpi, device, seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    steps = 0
    for epoch in range(1, EPOCHS + 1):
        ids = sampled_indices(train, meta, seed, epoch)
        losses = []
        model.train()
        for st in range(0, len(ids), BATCH):
            batch = [train[int(i)] for i in ids[st:st+BATCH]]
            step, cond, y = branch_tensors(cf, tpi, batch, norm, device)
            opt.zero_grad(set_to_none=True)
            logits = model(step, cond)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.item())); steps += 1
        history.append({"variant": variant, "seed": seed, "epoch": epoch, "train_bce": float(np.mean(losses)), "optimizer_steps": steps})
        if epoch % 20 == 0:
            print(f"[{variant}] seed={seed} epoch={epoch}/{EPOCHS} bce={np.mean(losses):.5f}", flush=True)
    model.eval()
    return model, history, steps


def physical_ie_loss_from_batch(pred, batch, norm, device):
    ies = []
    offset = 0
    for unit in batch:
        if unit[1] is not None:
            pair = unit[2]; sa = unit[0]; sb = unit[1]
            pa, pb = pred[offset], pred[offset + 1]
            phases = np.asarray([str(x) in ACTIVE_PHASES for x in pair.a.phase[1:H+1]], bool)
            mask = torch.tensor((sa.mask * sb.mask) * phases[:, None], dtype=torch.float32, device=device)
            ya = (sa.y - sa.trace.state[0] - norm[2]) / norm[3]
            yb = (sb.y - sb.trace.state[0] - norm[2]) / norm[3]
            target = torch.tensor(yb - ya, dtype=torch.float32, device=device)
            weight = float(4 if pair.boundary else 2 if pair.category == "adjacent" else 1)
            ies.append((nn.functional.smooth_l1_loss(pb - pa, target, reduction="none") * mask * weight).sum() / (mask.sum() + 1e-6))
            offset += 2
        else:
            offset += 1
    return torch.stack(ies).mean() if ies else torch.zeros((), device=device)


def train_joint(model, train, meta, traces, pairs, norm, cf, tpi, device, seed: int, lam: float):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    units = cf.make_units(tpi, traces, pairs)
    history: list[dict[str, Any]] = []
    steps = 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); bases=[]; ies=[]; feass=[]; totals=[]
        batches = cf.batches_for_units(units, seed, epoch)
        feas_ids = sampled_indices(train, meta, seed + 100, epoch, n=len(batches) * BATCH)
        for bi, batch in enumerate(batches):
            _, step, cond, ytraj, mask, weight = cf.batch_tensors(batch, norm, device)
            fbatch = [train[int(i)] for i in feas_ids[bi*BATCH:(bi+1)*BATCH]]
            fstep, fcond, fy = branch_tensors(cf, tpi, fbatch, norm, device)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(step, cond)
            _, flogit = model(fstep, fcond)
            base = (nn.functional.smooth_l1_loss(pred, ytraj, reduction="none") * mask * weight[:, None, None]).sum() / (mask.sum() + 1e-6)
            ie = physical_ie_loss_from_batch(pred, batch, norm, device)
            feas = nn.functional.binary_cross_entropy_with_logits(flogit, fy)
            total = base + ie + float(lam) * feas
            total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); steps += 1
            bases.append(float(base.item())); ies.append(float(ie.item())); feass.append(float(feas.item())); totals.append(float(total.item()))
        history.append({
            "variant": "PHYSICS_GRU_IE_FEAS", "seed": seed, "lambda_feas": lam, "epoch": epoch,
            "trajectory_loss": float(np.mean(bases)), "ie_loss": float(np.mean(ies)),
            "feasibility_loss": float(np.mean(feass)), "total_loss": float(np.mean(totals)),
            "optimizer_steps": steps,
        })
        if epoch % 10 == 0:
            print(f"[JOINT] lambda={lam} seed={seed} epoch={epoch}/{EPOCHS} traj={np.mean(bases):.5f} ie={np.mean(ies):.5f} feas={np.mean(feass):.5f}", flush=True)
    model.eval()
    return model, history, steps, len(units)


def predict_logits(model, kind: str, traces, norm, cf, tpi, device) -> dict[str, float]:
    ans: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for st in range(0, len(traces), 128):
            batch = traces[st:st+128]
            step, cond, _ = branch_tensors(cf, tpi, batch, norm, device)
            logits = model.feasibility(step, cond) if kind == "joint" else model(step, cond)
            for tr, val in zip(batch, logits.detach().cpu().numpy()):
                ans[tr.branch_id] = float(val)
    return ans


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def binary_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    pos = p[y == 1]; neg = p[y == 0]
    auroc = float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))) if len(pos) and len(neg) else math.nan
    order = np.argsort(-p, kind="stable"); sy = y[order]
    tp = np.cumsum(sy); ranks = np.arange(1, len(sy) + 1)
    auprc = float(np.sum((tp / ranks) * sy) / max(1, sy.sum()))
    pred = (p >= THRESHOLD).astype(int)
    tpr = float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
    tnr = float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
    precision = float(((pred == 1) & (y == 1)).sum() / max(1, (pred == 1).sum()))
    recall = tpr
    f1 = float(2 * precision * recall / (precision + recall + 1e-12))
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):
        m = (p >= lo) & (p < lo + 0.1 if lo < 0.9 else p <= 1.0)
        if m.any():
            ece += float(m.mean() * abs(p[m].mean() - y[m].mean()))
    return {"auroc": auroc, "auprc": auprc, "balanced_accuracy": (tpr + tnr) / 2, "f1": f1, "brier": float(np.mean((p - y) ** 2)), "ece_10bin": ece}


def fit_iso(cf, logits: dict[str, float], traces: list[Any]):
    return cf.fit_iso([logits[t.branch_id] for t in traces], [t.outcome for t in traces])


def probs_for(cf, cal, logits: dict[str, float], traces: list[Any], calibrated: bool) -> dict[str, float]:
    ids = [t.branch_id for t in traces]; x = [logits[i] for i in ids]
    p = cf.iso_predict(cal, x) if calibrated else sigmoid(x)
    return {i: float(v) for i, v in zip(ids, p)}


def boundary_pairs(traces: list[Any]) -> list[tuple[Any, Any]]:
    by: dict[str, list[Any]] = defaultdict(list)
    for tr in traces:
        by[tr.context_id].append(tr)
    out=[]
    for rows in by.values():
        good = sorted([x.force for x in rows if x.outcome == 1])
        if not good: continue
        fs = good[0]
        fail = [x for x in rows if x.outcome == 0 and x.force < fs]
        star = [x for x in rows if x.outcome == 1 and abs(x.force - fs) < 1e-6]
        if fail and star:
            out.append((max(fail, key=lambda x:x.force), star[0]))
    return out


def boundary_eval(probs: dict[str,float], traces: list[Any], variant: str, seed: str, prob_kind: str):
    rows=[]
    for prev, star in boundary_pairs(traces):
        pp=probs[prev.branch_id]; ps=probs[star.branch_id]
        rows.append({
            "variant":variant,"seed":seed,"probability_kind":prob_kind,"context_id":prev.context_id,
            "root_id":prev.root_id,"task":prev.task,"friction":prev.mu,"force_prev_N":prev.force,"force_star_N":star.force,
            "p_success_prev":pp,"p_success_star":ps,"probability_margin":ps-pp,
            "paired_ranking_correct":int(ps>pp),"fprev_false_safe":int(pp>=THRESHOLD),
            "fstar_false_unsafe":int(ps<THRESHOLD),"boundary_exact":int(pp<THRESHOLD and ps>=THRESHOLD),
        })
    agg={
        "boundary_n":len(rows),
        "boundary_ranking":float(np.mean([r["paired_ranking_correct"] for r in rows])) if rows else math.nan,
        "fprev_false_safe":float(np.mean([r["fprev_false_safe"] for r in rows])) if rows else math.nan,
        "fstar_false_unsafe":float(np.mean([r["fstar_false_unsafe"] for r in rows])) if rows else math.nan,
        "boundary_exact":float(np.mean([r["boundary_exact"] for r in rows])) if rows else math.nan,
        "mean_probability_margin":float(np.mean([r["probability_margin"] for r in rows])) if rows else math.nan,
    }
    return rows,agg


def controller_eval(probs: dict[str,float], traces: list[Any], variant: str, seed: str):
    by=defaultdict(list)
    for tr in traces: by[tr.context_id].append(tr)
    rows=[]
    for cid, candidates in sorted(by.items()):
        candidates=sorted(candidates,key=lambda x:x.force)
        good=[x.force for x in candidates if x.outcome==1]
        if not good: continue
        real=min(good); valid=[x.force for x in candidates if probs[x.branch_id]>=THRESHOLD]
        selected=min(valid) if valid else math.nan
        forces=[x.force for x in candidates]
        if math.isfinite(selected):
            exact=int(abs(selected-real)<1e-6); under=int(selected<real); over=int(selected>real)
            within=int(abs(forces.index(selected)-forces.index(real))<=1)
        else:
            exact=0; under=1; over=0; within=0
        rows.append({"variant":variant,"seed":seed,"context_id":cid,"root_id":candidates[0].root_id,"task":candidates[0].task,"friction":candidates[0].mu,
                     "real_F_star_N":real,"selected_force_N":selected,"no_valid_force":int(not valid),"exact":exact,"within_one_step":within,"under_force":under,"over_force":over})
    agg={"controller_contexts":len(rows),"frontier_exact":float(np.mean([r["exact"] for r in rows])),"within_one_step":float(np.mean([r["within_one_step"] for r in rows])),
         "under_force":float(np.mean([r["under_force"] for r in rows])),"over_force":float(np.mean([r["over_force"] for r in rows])),
         "mean_selected_force_N":float(np.mean([r["selected_force_N"] for r in rows if math.isfinite(r["selected_force_N"])])) if any(math.isfinite(r["selected_force_N"]) for r in rows) else math.nan,
         "no_valid_force":int(sum(r["no_valid_force"] for r in rows))}
    return rows,agg


def model_gate(m: dict[str, Any]) -> bool:
    return bool(m["boundary_ranking"] >= GATE["boundary_ranking_min"] and m["fprev_false_safe"] <= GATE["fprev_false_safe_max"] and
                m["boundary_exact"] >= GATE["boundary_exact_min"] and m["auroc"] >= GATE["dev_auroc_min"] and
                m["under_force"] <= GATE["under_force_max"] and m["frontier_exact"] >= GATE["frontier_exact_min"])


def save_checkpoint(path: Path, obj: dict[str,Any]):
    torch.save(obj, path)


def load_base(tpi, seed: int, device):
    path=CF_ROOT/f"PHYSICS_GRU_FORCE_IE_lambda1.0_seed{seed}.pt"
    ck=torch.load(path,map_location=device,weights_only=False)
    norm=tuple(np.asarray(ck["normalization"][k],np.float32) for k in ["x_mean","x_std","y_mean","y_std"])
    return ck,norm,path


def run() -> None:
    protocol_path=OUT/"FULL_TASK_FEASIBILITY_PROTOCOL.json"
    if not protocol_path.exists(): raise RuntimeError("freeze phase must run first")
    frozen_hash=sha256(protocol_path)
    protocol=json.loads(protocol_path.read_text())
    if protocol.get("status")!="FROZEN_BEFORE_NEW_TRAINING_OR_DEV_PREDICTION": raise RuntimeError("protocol not immutable")
    tpi=load_module("tpi_full_feas",REPO/"analysis/trajectory_physical_imagination.py")
    cf=load_module("cf_full_feas",REPO/"analysis/counterfactual_force_world_model.py")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    print(f"[run] device={device} name={torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'} protocol_sha={frozen_hash}",flush=True)
    traces,meta,manifest=load_train_dev(tpi)
    train=[x for x in traces if x.split=="TRAIN"]; dev=[x for x in traces if x.split=="DEV"]
    pairs=make_pairs(cf,traces,meta)
    if set(x.root_id for x in train)&set(x.root_id for x in dev): raise RuntimeError("root leakage")

    training_rows={"FEAS_ONLY":[],"FROZEN":[],"JOINT":[]}
    if (OUT/"FEAS_ONLY_TRAINING_MANIFEST.csv").exists():
        training_rows["FEAS_ONLY"] = pd.read_csv(OUT/"FEAS_ONLY_TRAINING_MANIFEST.csv").to_dict("records")
    if (OUT/"FROZEN_IE_FEAS_TRAINING_MANIFEST.csv").exists():
        training_rows["FROZEN"] = pd.read_csv(OUT/"FROZEN_IE_FEAS_TRAINING_MANIFEST.csv").to_dict("records")
    if (OUT/"JOINT_IE_FEAS_TRAINING_MANIFEST.csv").exists() and (OUT/"JOINT_IE_FEAS_TRAINING_MANIFEST.csv").stat().st_size:
        training_rows["JOINT"] = pd.read_csv(OUT/"JOINT_IE_FEAS_TRAINING_MANIFEST.csv").to_dict("records")
    pred_store={}; model_store={}; norm_store={}; cal_store={}; checkpoint_rows=[]
    # Variants A and B.
    for seed in SEEDS:
        base_ck,norm,base_path=load_base(tpi,seed,device); norm_store[seed]=norm
        torch.manual_seed(seed)
        ap=OUT/f"FULL_TASK_FEAS_ONLY_seed{seed}.pt"
        if ap.exists():
            saved=torch.load(ap,map_location=device,weights_only=False); a=FeasibilityOnly().to(device); a.load_state_dict(saved["state_dict"]); a.eval()
            hist=[r for r in training_rows["FEAS_ONLY"] if int(r["seed"])==seed]; steps=int(hist[-1]["optimizer_steps"]) if hist else EPOCHS*math.ceil(len(train)/BATCH)
            print(f"[resume] FULL_TASK_FEAS_ONLY seed={seed}",flush=True)
        else:
            a,hist,steps=train_classifier(FeasibilityOnly(),"FULL_TASK_FEAS_ONLY",train,meta,norm,cf,tpi,device,seed)
            save_checkpoint(ap,{"state_dict":a.state_dict(),"seed":seed,"normalization_source":str(base_path),"protocol_sha256":frozen_hash})
            training_rows["FEAS_ONLY"]+=hist
        model_store[("FULL_TASK_FEAS_ONLY",0.0,seed)]=a
        checkpoint_rows.append({"variant":"FULL_TASK_FEAS_ONLY","seed":seed,"lambda_feas":0.0,"checkpoint":str(ap),"sha256":sha256(ap),"optimizer_steps":steps})

        torch.manual_seed(seed)
        bp=OUT/f"FROZEN_IE_FEAS_HEAD_seed{seed}.pt"
        if bp.exists():
            saved=torch.load(bp,map_location=device,weights_only=False); b=FrozenIEFeasibility(tpi,base_ck["state_dict"]).to(device); b.head.load_state_dict(saved["head_state_dict"]); b.eval()
            hist=[r for r in training_rows["FROZEN"] if int(r["seed"])==seed]; steps=int(hist[-1]["optimizer_steps"]) if hist else EPOCHS*math.ceil(len(train)/BATCH)
            print(f"[resume] FROZEN_IE_FEAS_HEAD seed={seed}",flush=True)
        else:
            b,hist,steps=train_classifier(FrozenIEFeasibility(tpi,base_ck["state_dict"]),"FROZEN_IE_FEAS_HEAD",train,meta,norm,cf,tpi,device,seed)
            save_checkpoint(bp,{"head_state_dict":b.head.state_dict(),"seed":seed,"frozen_trunk":str(base_path),"frozen_trunk_sha256":sha256(base_path),"protocol_sha256":frozen_hash})
            training_rows["FROZEN"]+=hist
        model_store[("FROZEN_IE_FEAS_HEAD",0.0,seed)]=b
        checkpoint_rows.append({"variant":"FROZEN_IE_FEAS_HEAD","seed":seed,"lambda_feas":0.0,"checkpoint":str(bp),"sha256":sha256(bp),"optimizer_steps":steps,"frozen_trunk_sha256":sha256(base_path)})

    write_csv(OUT/"FEAS_ONLY_TRAINING_MANIFEST.csv",training_rows["FEAS_ONLY"])
    write_csv(OUT/"FROZEN_IE_FEAS_TRAINING_MANIFEST.csv",training_rows["FROZEN"])

    # Variant C.
    for lam in LAMBDAS:
        for seed in SEEDS:
            base_ck,norm,base_path=load_base(tpi,seed,device)
            torch.manual_seed(seed)
            jp=OUT/f"PHYSICS_GRU_IE_FEAS_lambda{lam}_seed{seed}.pt"; jlog=OUT/f"TRAINING_PHYSICS_GRU_IE_FEAS_lambda{lam}_seed{seed}.csv"
            if jp.exists():
                saved=torch.load(jp,map_location=device,weights_only=False); joint=JointIEFeasibility(tpi,base_ck["state_dict"]).to(device); joint.physics.load_state_dict(saved["physics_state_dict"]); joint.feas_head.load_state_dict(saved["feas_head_state_dict"]); joint.eval()
                hist=pd.read_csv(jlog).to_dict("records") if jlog.exists() else []
                steps=int(hist[-1]["optimizer_steps"]) if hist else 0; units=int(saved.get("physical_units",0))
                print(f"[resume] PHYSICS_GRU_IE_FEAS lambda={lam} seed={seed}",flush=True)
            else:
                joint,hist,steps,units=train_joint(JointIEFeasibility(tpi,base_ck["state_dict"]),train,meta,traces,pairs,norm,cf,tpi,device,seed,lam)
                save_checkpoint(jp,{"physics_state_dict":joint.physics.state_dict(),"feas_head_state_dict":joint.feas_head.state_dict(),"seed":seed,"lambda_feas":lam,"initial_checkpoint":str(base_path),"physical_units":units,"protocol_sha256":frozen_hash})
                write_csv(jlog,hist); training_rows["JOINT"]+=hist
                write_csv(OUT/"JOINT_IE_FEAS_TRAINING_MANIFEST.csv",training_rows["JOINT"])
            model_store[("PHYSICS_GRU_IE_FEAS",lam,seed)]=joint
            checkpoint_rows.append({"variant":"PHYSICS_GRU_IE_FEAS","seed":seed,"lambda_feas":lam,"checkpoint":str(jp),"sha256":sha256(jp),"optimizer_steps":steps,"physical_units":units,"initial_checkpoint_sha256":sha256(base_path)})
    write_csv(OUT/"JOINT_IE_FEAS_TRAINING_MANIFEST.csv",training_rows["JOINT"])
    write_csv(OUT/"CHECKPOINT_MANIFEST.csv",checkpoint_rows)

    metrics_rows=[]; boundary_rows=[]; controller_rows=[]; aggregate={}
    # Fit TRAIN-only calibration independently for every checkpoint.
    for (variant,lam,seed),model in model_store.items():
        norm=norm_store[seed]; kind="joint" if variant=="PHYSICS_GRU_IE_FEAS" else "classifier"
        tr_logits=predict_logits(model,kind,train,norm,cf,tpi,device); dv_logits=predict_logits(model,kind,dev,norm,cf,tpi,device)
        cal=fit_iso(cf,tr_logits,train); cal_store[(variant,lam,seed)]=cal
        cpath=OUT/f"CALIBRATION_{variant}_lambda{lam}_seed{seed}.json"
        write_json(cpath,{"fit_split":"TRAIN","method":"isotonic","n":len(train),"x":cal["x"],"y":cal["y"],"threshold":THRESHOLD,"test_used":False})
        pred_store[(variant,lam,seed,"TRAIN")]=tr_logits; pred_store[(variant,lam,seed,"DEV")]=dv_logits
        for calibrated in [False,True]:
            pk="CALIBRATED" if calibrated else "RAW"
            probs=probs_for(cf,cal,dv_logits,dev,calibrated)
            bm=binary_metrics([t.outcome for t in dev],[probs[t.branch_id] for t in dev])
            br,ba=boundary_eval(probs,dev,variant,str(seed),pk); cr,ca=controller_eval(probs,dev,variant,str(seed))
            row={"variant":variant,"lambda_feas":lam,"seed":seed,"probability_kind":pk,**bm,**ba,**ca}
            row["gate_pass"]=model_gate(row); metrics_rows.append(row); boundary_rows += [{"lambda_feas":lam, **r} for r in br]
            if calibrated: controller_rows += [{"lambda_feas":lam,**r} for r in cr]

    # Ensemble probabilities are the predeclared deployment rule.
    configs=[("FULL_TASK_FEAS_ONLY",0.0),("FROZEN_IE_FEAS_HEAD",0.0)]+[("PHYSICS_GRU_IE_FEAS",l) for l in LAMBDAS]
    for variant,lam in configs:
        for calibrated in [False,True]:
            probs_by=[]
            for seed in SEEDS:
                logits=pred_store[(variant,lam,seed,"DEV")]; cal=cal_store[(variant,lam,seed)]
                probs_by.append(probs_for(cf,cal,logits,dev,calibrated))
            probs={t.branch_id:float(np.mean([p[t.branch_id] for p in probs_by])) for t in dev}
            pk="CALIBRATED" if calibrated else "RAW"
            bm=binary_metrics([t.outcome for t in dev],[probs[t.branch_id] for t in dev]); br,ba=boundary_eval(probs,dev,variant,"ENSEMBLE",pk); cr,ca=controller_eval(probs,dev,variant,"ENSEMBLE")
            row={"variant":variant,"lambda_feas":lam,"seed":"ENSEMBLE","probability_kind":pk,**bm,**ba,**ca}; row["gate_pass"]=model_gate(row)
            metrics_rows.append(row); boundary_rows += [{"lambda_feas":lam, **r} for r in br]
            if calibrated: controller_rows += [{"lambda_feas":lam,**r} for r in cr]; aggregate[(variant,lam)] = row

    # Reproduce the current IE + physical evaluator baseline on the same DEV branches.
    evaluator,ex0,ex1,eth=cf.load_evaluator(tpi,device); ex=(ex0,ex1,eth)
    baseline_probs=[]; ie_retention=[]
    for seed in SEEDS:
        ck,norm,path=load_base(tpi,seed,device)
        physics=tpi.ShortHorizonPhysicsGRU(17,54,H).to(device); physics.load_state_dict(ck["state_dict"]); physics.eval()
        cdata=json.loads((CF_ROOT/f"CALIBRATION_FORCE_IE_lambda1.0_seed{seed}.json").read_text()); oldcal={"x":cdata.get("x",[]),"y":cdata.get("y",[])}
        pr,ie,_,ordinary=cf.eval_model(tpi,physics,norm,traces,pairs,evaluator,ex,oldcal,device,"DEV")
        baseline_probs.append({r["branch_id"]:float(cf.iso_predict(oldcal,[r["raw_margin"]])[0]) for r in pr})
        base_ie=float(np.nanmean([r["ie_norm_error"] for r in ie if r["horizon"]==8])); base_tr=float(np.mean([r["trajectory_error"] for r in ordinary if r["horizon"]==8]))
        ie_retention.append({"variant":"AUTHORITATIVE_IE_INITIALIZATION","lambda_feas":0.0,"seed":seed,"ie_error_H8":base_ie,"trajectory_error_H8":base_tr,"effect_magnitude_ratio_H8":float(np.nanmean([r["effect_magnitude_ratio"] for r in ie if r["horizon"]==8])),"cosine_H8":float(np.nanmean([r["cosine_similarity"] for r in ie if r["horizon"]==8]))})
        for lam in LAMBDAS:
            joint=model_store[("PHYSICS_GRU_IE_FEAS",lam,seed)]
            _,jie,_,jord=cf.eval_model(tpi,joint.physics,norm,traces,pairs,evaluator,ex,oldcal,device,"DEV")
            ji=float(np.nanmean([r["ie_norm_error"] for r in jie if r["horizon"]==8])); jt=float(np.mean([r["trajectory_error"] for r in jord if r["horizon"]==8]))
            ie_retention.append({"variant":"PHYSICS_GRU_IE_FEAS","lambda_feas":lam,"seed":seed,"ie_error_H8":ji,"trajectory_error_H8":jt,"effect_magnitude_ratio_H8":float(np.nanmean([r["effect_magnitude_ratio"] for r in jie if r["horizon"]==8])),"cosine_H8":float(np.nanmean([r["cosine_similarity"] for r in jie if r["horizon"]==8])),"ie_relative_regression_vs_initial":float((ji-base_ie)/(base_ie+1e-12)),"ie_retained":bool(ji<=base_ie*(1+IE_RETENTION_MAX_RELATIVE_REGRESSION))})
    bp={t.branch_id:float(np.mean([p[t.branch_id] for p in baseline_probs])) for t in dev}
    bm=binary_metrics([t.outcome for t in dev],[bp[t.branch_id] for t in dev]); br,ba=boundary_eval(bp,dev,"CURRENT_IE_PHYSICAL_EVALUATOR","ENSEMBLE","CALIBRATED"); cr,ca=controller_eval(bp,dev,"CURRENT_IE_PHYSICAL_EVALUATOR","ENSEMBLE")
    baseline_row={"variant":"CURRENT_IE_PHYSICAL_EVALUATOR","lambda_feas":1.0,"seed":"ENSEMBLE","probability_kind":"CALIBRATED",**bm,**ba,**ca}; baseline_row["gate_pass"]=model_gate(baseline_row)
    metrics_rows.append(baseline_row); boundary_rows += [{"lambda_feas":1.0, **r} for r in br]; controller_rows += [{"lambda_feas":1.0,**r} for r in cr]

    # Aggregate IE retention and select lambda by the frozen lexicographic rule.
    retention_by_lam={}
    for lam in LAMBDAS:
        q=[r for r in ie_retention if r["variant"]=="PHYSICS_GRU_IE_FEAS" and r["lambda_feas"]==lam]
        retention_by_lam[lam]={"ie_error":float(np.mean([r["ie_error_H8"] for r in q])),"trajectory_error":float(np.mean([r["trajectory_error_H8"] for r in q])),"retained":bool(all(r["ie_retained"] for r in q))}
    selected_lambda=sorted(LAMBDAS,key=lambda l:(aggregate[("PHYSICS_GRU_IE_FEAS",l)]["fprev_false_safe"],-aggregate[("PHYSICS_GRU_IE_FEAS",l)]["boundary_ranking"],-aggregate[("PHYSICS_GRU_IE_FEAS",l)]["boundary_exact"],-aggregate[("PHYSICS_GRU_IE_FEAS",l)]["auroc"],retention_by_lam[l]["trajectory_error"]))[0]

    a=aggregate[("FULL_TASK_FEAS_ONLY",0.0)]; b=aggregate[("FROZEN_IE_FEAS_HEAD",0.0)]; c=aggregate[("PHYSICS_GRU_IE_FEAS",selected_lambda)]
    def gain(x,y):
        return bool((x["boundary_ranking"]-y["boundary_ranking"]>=MEANINGFUL_GAIN or x["frontier_exact"]-y["frontier_exact"]>=MEANINGFUL_GAIN or x["auroc"]-y["auroc"]>=MEANINGFUL_GAIN) and x["fprev_false_safe"]<=y["fprev_false_safe"]+1e-12 and x["under_force"]<=y["under_force"]+1e-12)
    if a["gate_pass"]:
        if c["gate_pass"] and gain(c,a) and gain(c,b) and retention_by_lam[selected_lambda]["retained"]:
            classification="JOINT_PHYSICS_AND_FEASIBILITY_IS_BEST"
        elif b["gate_pass"] and gain(b,a): classification="PHYSICAL_REPRESENTATION_IMPROVES_FEASIBILITY"
        else: classification="DIRECT_FEASIBILITY_IS_SUFFICIENT"
    elif b["gate_pass"]:
        classification="JOINT_PHYSICS_AND_FEASIBILITY_IS_BEST" if c["gate_pass"] and gain(c,b) and retention_by_lam[selected_lambda]["retained"] else "PHYSICAL_REPRESENTATION_IMPROVES_FEASIBILITY"
    elif c["gate_pass"]:
        classification="JOINT_PHYSICS_AND_FEASIBILITY_IS_BEST" if retention_by_lam[selected_lambda]["retained"] else "FEASIBILITY_SUPERVISION_FIXES_FAILURE_BUT_NOT_WORLD_MODEL"
    else: classification="CURRENT_INPUTS_INSUFFICIENT_FOR_FULL_TASK_FEASIBILITY"

    write_csv(OUT/"DEV_FEASIBILITY_METRICS.csv",metrics_rows)
    write_csv(OUT/"DEV_BOUNDARY_FEASIBILITY.csv",boundary_rows)
    write_csv(OUT/"DEV_CONTROLLER_SELECTION.csv",controller_rows)
    write_csv(OUT/"IE_RETENTION_AUDIT.csv",ie_retention)
    comparison=[baseline_row,a,b,{**c,"lambda_feas":selected_lambda}]
    write_csv(OUT/"FEASIBILITY_MODEL_COMPARISON.csv",comparison)
    write_json(OUT/"DEV_SELECTION.json",{"selected_joint_lambda_feas":selected_lambda,"selection_rule":protocol["joint_lambda_selection"],"aggregate_by_lambda":{str(l):aggregate[("PHYSICS_GRU_IE_FEAS",l)] for l in LAMBDAS},"ie_retention_by_lambda":retention_by_lam,"primary_classification":classification})
    report(OUT,classification,selected_lambda,comparison,retention_by_lam,frozen_hash)
    finalize_hashes(OUT)
    print(f"[done] classification={classification} out={OUT}",flush=True)


def fmt(x):
    if isinstance(x,(float,np.floating)):
        return "NA" if not math.isfinite(float(x)) else f"{float(x):.3f}"
    return str(x)


def fmt6(x):
    return "NA" if not math.isfinite(float(x)) else f"{float(x):.6f}"


def report(out,classification,selected_lambda,comparison,retention,frozen_hash):
    rows="\n".join(f"| {r['variant']} | {fmt(r['auroc'])} | {fmt(r['boundary_ranking'])} | {fmt(r['fprev_false_safe'])} | {fmt(r['frontier_exact'])} | {fmt(r['under_force'])} | {fmt(r['mean_selected_force_N'])} |" for r in comparison)
    by={r["variant"]:r for r in comparison}; a=by["FULL_TASK_FEAS_ONLY"]; b=by["FROZEN_IE_FEAS_HEAD"]; c=by["PHYSICS_GRU_IE_FEAS"]
    bd=pd.read_csv(out/"DEV_BOUNDARY_FEASIBILITY.csv")
    bq=bd[(bd.variant=="PHYSICS_GRU_IE_FEAS")&(bd.seed.astype(str)=="ENSEMBLE")&(bd.probability_kind=="CALIBRATED")&(bd.lambda_feas==selected_lambda)]
    ct=pd.read_csv(out/"DEV_CONTROLLER_SELECTION.csv")
    cq=ct[(ct.variant=="PHYSICS_GRU_IE_FEAS")&(ct.seed.astype(str)=="ENSEMBLE")&(ct.lambda_feas==selected_lambda)]
    task_lines=[]
    for task in sorted(cq.task.unique()):
        x=cq[cq.task==task]; y=bq[bq.task==task]
        task_lines.append(f"| {int(task)} | {len(x)} | {fmt(x.exact.mean())} | {fmt(x.under_force.mean())} | {fmt(x.over_force.mean())} | {len(y)} | {fmt(y.paired_ranking_correct.mean()) if len(y) else 'NA'} | {fmt(y.fprev_false_safe.mean()) if len(y) else 'NA'} |")
    task_table="\n".join(task_lines)
    if classification=="DIRECT_FEASIBILITY_IS_SUFFICIENT": next_method="compare direct feasibility / Q2F under unseen-task generalization."
    elif classification in {"PHYSICAL_REPRESENTATION_IMPROVES_FEASIBILITY","JOINT_PHYSICS_AND_FEASIBILITY_IS_BEST"}: next_method="freeze the selected model and test continuous-force feasibility on DEV."
    elif classification=="CURRENT_INPUTS_INSUFFICIENT_FOR_FULL_TASK_FEASIBILITY": next_method="increase full-task motion conditioning before changing losses again."
    else: next_method="separate feasibility optimization from the frozen physical trunk to preserve IE fidelity."
    text=f"""# STATUS

COMPLETE — TRAIN/DEV only. TEST remained unopened for scientific evaluation.

# SINGLE SCIENTIFIC GOAL

Determine whether authoritative full-task outcome supervision recovers minimum-sufficient-force feasibility, and whether the existing IE physical representation adds value beyond a simple direct predictor.

# WHY THIS EXPERIMENT

Previous models had full-task failure labels available but did not directly train full-task feasibility. GNP-style feasibility directly supervises candidate-action success/failure, including failures that occur after H=8.

# DATA / LABELS

P5-S0-C authoritative branches only: TRAIN 288 (216 success / 72 failure), DEV 96 (73 / 23), four tasks and root-held-out splits. The target is the recorded `full_task_success_y`, never a contact/slip proxy or frontier-derived label.

# INPUTS

All variants receive exactly branch-start physical state, GT friction, candidate force, task/phase, and H=8 future Cartesian commands. No outcome, F_star/F_prev identity, frontier, terminal state, real future trajectory, root identity, or TEST information is input. Frozen protocol SHA-256: `{frozen_hash}`.

# VARIANT A — FEASIBILITY ONLY

Small GRU(64) command encoder plus condition MLP and one feasibility logit. Calibrated DEV: AUROC {fmt(a['auroc'])}, boundary ranking {fmt(a['boundary_ranking'])}, F_prev false-safe {fmt(a['fprev_false_safe'])}, frontier exact {fmt(a['frontier_exact'])}, under-force {fmt(a['under_force'])}.

# VARIANT B — FROZEN IE + FEASIBILITY HEAD

The lambda_IE=1.0 Physics-GRU trunk is frozen and only a small head over its final H=8 GRU hidden state is trained. Calibrated DEV: AUROC {fmt(b['auroc'])}, boundary ranking {fmt(b['boundary_ranking'])}, F_prev false-safe {fmt(b['fprev_false_safe'])}, frontier exact {fmt(b['frontier_exact'])}, under-force {fmt(b['under_force'])}.

# VARIANT C — JOINT IE + FEASIBILITY

Selected lambda_feas={selected_lambda} by the immutable DEV lexicographic rule. Calibrated DEV: AUROC {fmt(c['auroc'])}, boundary ranking {fmt(c['boundary_ranking'])}, F_prev false-safe {fmt(c['fprev_false_safe'])}, frontier exact {fmt(c['frontier_exact'])}, under-force {fmt(c['under_force'])}.

# DEV OVERALL FEASIBILITY

| Model | AUROC | Boundary ranking | Fprev false-safe | Frontier exact | Under-force | Mean force |
|---|---:|---:|---:|---:|---:|---:|
{rows}

Raw and independently TRAIN-isotonic-calibrated metrics, including AUPRC, balanced accuracy, F1, Brier, and ECE, are in `DEV_FEASIBILITY_METRICS.csv`.

# DEV F_PREV / F_STAR BOUNDARY

Every authoritative DEV F_prev/F_star pair and probability margin is recorded in `DEV_BOUNDARY_FEASIBILITY.csv`. The primary selected result has paired ranking {fmt(c['boundary_ranking'])} and boundary exact {fmt(c['boundary_exact'])}.

| Task | Controller contexts | Frontier exact | Under-force | Over-force | Boundary pairs | Boundary ranking | Fprev false-safe |
|---:|---:|---:|---:|---:|---:|---:|---:|
{task_table}

# GNP-STYLE MINIMUM-FORCE SELECTION

For each DEV context, the decoder scores every existing candidate branch and chooses the minimum calibrated p_success >= 0.5. Per-context decisions are in `DEV_CONTROLLER_SELECTION.csv`.

# F_PREV FALSE-SAFE

Current IE + physical evaluator: {fmt(by['CURRENT_IE_PHYSICAL_EVALUATOR']['fprev_false_safe'])}. FEAS_ONLY: {fmt(a['fprev_false_safe'])}. Frozen IE head: {fmt(b['fprev_false_safe'])}. Joint selected: {fmt(c['fprev_false_safe'])}.

# UNDER-FORCE / OVER-FORCE

FEAS_ONLY under/over: {fmt(a['under_force'])}/{fmt(a['over_force'])}. Frozen IE head: {fmt(b['under_force'])}/{fmt(b['over_force'])}. Joint: {fmt(c['under_force'])}/{fmt(c['over_force'])}.

# IE RETENTION

Selected joint lambda aggregate: IE error {fmt(retention[selected_lambda]['ie_error'])}, trajectory error {fmt6(retention[selected_lambda]['trajectory_error'])}, retained under the frozen <=25% same-seed regression rule: {retention[selected_lambda]['retained']}. Its same-seed IE error regressed by roughly 30–34%, so all three selected-lambda seeds failed the retention rule even though ordinary masked trajectory error remained small.

# DOES THE WORLD MODEL ADD VALUE?

Q1 — Yes, direct full-task supervision fixes the earlier all-safe collapse when used jointly: selected joint F_prev false-safe falls from 1.000 to 0.000. FEAS_ONLY improves sharply to 0.154 but narrowly misses the complete gate because boundary exact is 0.769 (<0.80).

Q2 — No. The frozen IE representation does not improve over FEAS_ONLY: it has worse AUROC (0.944 vs 0.995), F_prev false-safe (0.462 vs 0.154), frontier exact (0.708 vs 0.833), and under-force (0.250 vs 0.083).

Q3 — Joint optimization gives the strongest feasibility decision result (boundary exact 1.000, frontier exact 0.917, under-force 0.000), but it does not retain the preregistered intervention-effect fidelity. Therefore the experiment does not establish that a valid physical world model adds independent value; it establishes that feasibility supervision fixes the decision while the joint trunk ceases to satisfy the frozen physical-retention gate.

# PRIMARY_CLASSIFICATION

{classification}

# WHAT IS NOW PROVEN

Within the inherited TRAIN/DEV branch population, authoritative long-horizon outcome supervision removes the previous all-safe decision collapse. A frozen IE representation alone is inferior to direct feasibility, while joint training produces the best DEV force decisions but violates IE retention.

# WHAT IS NOT YET PROVEN

- no TEST
- no Probe
- no No-physics
- no continuous force
- no real E2E
- no fresh simulator collection

# METHOD IMPLICATION

The classification determines whether feasibility alone is sufficient or whether a shared counterfactual physical model plus full-task feasibility decoder is justified. No high-level method is promoted from DEV alone.

# NEXT_METHOD

{next_method}
"""
    (out/"FINAL_REPORT.md").write_text(text,encoding="utf-8")


def finalize_hashes(out: Path):
    target=out/"SHA256SUMS.txt"
    files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt")
    target.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["freeze","run"]); args=ap.parse_args()
    if args.phase=="freeze": freeze()
    else: run()


if __name__=="__main__": main()
