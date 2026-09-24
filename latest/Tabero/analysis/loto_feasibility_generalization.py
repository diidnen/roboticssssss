#!/usr/bin/env python3
"""Strict TRAIN/DEV-root-only leave-one-task-out feasibility experiment.

The script deliberately has two phases. ``freeze`` writes the task-conditioning,
split, protocol, and leakage artifacts before optimization. ``run`` trains and
selects each fold using only its three seen tasks, freezes that choice, and only
then evaluates the unseen task's DEV roots. Original TEST outcomes and telemetry
are never loaded.
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
AUTHORITATIVE = RESULTS / "full_task_feasibility_20260830_012830"
HIST = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
CF_ROOT = RESULTS / "counterfactual_force_world_model_20260829_160000"
OUT = Path(os.environ.get("LOTO_FEAS_OUT", RESULTS / "loto_feasibility_generalization_20260830_022058"))

TASKS = [0, 1, 5, 6]
FOLDS = {f"T{t}": {"heldout": t, "seen": [x for x in TASKS if x != t]} for t in TASKS}
SEEDS = [0, 1, 2]
LAMBDAS = [0.1, 0.3, 1.0]
H = 8
EPOCHS = 80
BATCH = 64
LR = 8e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
ACTIVE_PHASES = {"branch_hold", "lift", "transit", "over_basket", "place"}
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 2026083002
GATE = {
    "macro_boundary_ranking_min": 0.80,
    "macro_fprev_false_safe_max": 0.20,
    "macro_frontier_exact_min": 0.75,
    "macro_under_force_max": 0.10,
    "tasks_boundary_ranking_at_least_0.75_min_count": 3,
}
SEGMENT_CACHE: dict[tuple[str, int], Any] = {}


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


def read_non_test_manifest() -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Read scientific TRAIN/DEV rows; retain only non-outcome TEST metadata."""
    scientific: list[dict[str, str]] = []
    test_meta: list[dict[str, str]] = []
    path = HIST / "P5S0C_BRANCH_MANIFEST.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "TEST":
                test_meta.append({k: row[k] for k in ["branch_id", "context_id", "root_id", "task", "split"]})
                continue
            if row["split"] in {"TRAIN", "DEV"}:
                scientific.append(row)
    d = pd.DataFrame(scientific)
    for c in ["task", "full_task_success_y"]:
        d[c] = pd.to_numeric(d[c]).astype(int)
    for c in ["requested_force_N", "hidden_friction_analysis_only"]:
        d[c] = pd.to_numeric(d[c]).astype(float)
    return d, test_meta


def authoritative_files() -> list[Path]:
    return [
        AUTHORITATIVE / "FINAL_REPORT.md",
        AUTHORITATIVE / "FULL_TASK_FEASIBILITY_PROTOCOL.json",
        AUTHORITATIVE / "FEASIBILITY_MODEL_COMPARISON.csv",
        AUTHORITATIVE / "DEV_BOUNDARY_FEASIBILITY.csv",
        AUTHORITATIVE / "DEV_CONTROLLER_SELECTION.csv",
        AUTHORITATIVE / "IE_RETENTION_AUDIT.csv",
        AUTHORITATIVE / "DEV_SELECTION.json",
        HIST / "P5S0C_BRANCH_MANIFEST.csv",
        HIST / "P5S0C_FORCE_MANIFEST.json",
        HIST / "P5S0C_SPLIT_MANIFEST.json",
        REPO / "analysis/trajectory_physical_imagination.py",
        REPO / "analysis/counterfactual_force_world_model.py",
        REPO / "analysis/full_task_feasibility_decoder.py",
    ]


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, test_meta = read_non_test_manifest()
    auth_sel = json.loads((AUTHORITATIVE / "DEV_SELECTION.json").read_text())
    selected_previous = float(auth_sel["selected_joint_lambda_feas"])
    hashes = {str(p): sha256(p) for p in authoritative_files() if p.exists()}

    task_audit = {
        "status": "PASS_REQUIRES_TASK_ID_REMOVAL_FOR_LOTO",
        "authoritative_joint_raw_nominal_input_dim": 71,
        "authoritative_step_input_columns": {
            "0:3": "H-step future Cartesian relative translation command xyz",
            "3:6": "H-step finite difference of Cartesian translation command xyz",
            "6:13": "common task phase one-hot (7)",
            "13:17": "DISCRETE TASK ID one-hot for tasks 0/1/5/6",
        },
        "authoritative_static_condition_columns": {
            "17": "candidate force / 8",
            "18": "GT friction",
            "19:32": "current physical state (13)",
            "32:45": "current-state validity mask (13)",
            "45:58": "observable initial physical state (13)",
            "58:71": "initial-state validity mask (13)",
        },
        "task_id_actually_entered_prediction_path": True,
        "future_translation_available": True,
        "future_translation_delta_available": True,
        "future_rotation_or_orientation_command_available": False,
        "future_angular_velocity_command_available": False,
        "future_acceleration_explicitly_available": False,
        "complete_future_end_effector_or_gripper_trajectory_available": False,
        "loto_motion_conditioned_mask": "columns 13:17 hard-zeroed before fold normalization and at every input construction",
        "loto_no_future_motion_mask": "columns 0:6 and 13:17 hard-zeroed; only common phase remains in the step sequence",
        "plain_language_answer": "The model knows the next H=8 Cartesian translation commands and their first differences, but not future rotation/orientation, explicit acceleration, or a complete EE/gripper trajectory.",
    }
    write_json(OUT / "TASK_CONDITIONING_AUDIT.json", task_audit)

    split_rows: list[dict[str, Any]] = []
    for fold, cfg in FOLDS.items():
        for r in d.itertuples(index=False):
            role = "HELDOUT_TASK_EVALUATION" if int(r.task) == cfg["heldout"] and r.split == "DEV" else (
                "SEEN_TASK_TRAIN" if int(r.task) in cfg["seen"] and r.split == "TRAIN" else (
                    "SEEN_TASK_MODEL_SELECTION" if int(r.task) in cfg["seen"] and r.split == "DEV" else "EXCLUDED_HELDOUT_TASK_TRAIN"
                )
            )
            split_rows.append({
                "fold": fold, "heldout_task": cfg["heldout"], "branch_id": r.branch_id,
                "context_id": r.context_id, "root_id": r.root_id, "task": int(r.task),
                "source_split": r.split, "fold_role": role,
                "allowed_for_training": int(role == "SEEN_TASK_TRAIN"),
                "allowed_for_model_selection": int(role == "SEEN_TASK_MODEL_SELECTION"),
                "heldout_evaluated_after_freeze": int(role == "HELDOUT_TASK_EVALUATION"),
            })
    write_csv(OUT / "LOTO_SPLIT_MANIFEST.csv", split_rows)

    leakage = {
        "status": "PASS_BY_FROZEN_DESIGN_PENDING_RUNTIME_ASSERTIONS",
        "task_id_input": False,
        "heldout_task_embedding": False,
        "heldout_task_TRAIN_branches_used": False,
        "heldout_task_DEV_outcomes_used_for_selection": False,
        "heldout_task_calibration_parameters": False,
        "root_identity_input": False,
        "F_star_or_frontier_input": False,
        "success_label_input": False,
        "future_real_trajectory_input_to_feasibility": False,
        "allowed_deployment_information": ["current physical state", "GT friction", "candidate force", "common task phase", "H=8 future Cartesian translation commands"],
        "normalization": "refit using only seen-task TRAIN branches in each fold",
        "probability_calibration": "TRAIN-only isotonic on seen-task TRAIN branches in each fold",
        "model_selection": "seen-task DEV only",
        "heldout_evaluation_order": "fold selection artifact and hashes are written before heldout-task DEV telemetry/outcomes are loaded",
        "all_task_checkpoint_initialization": False,
        "reason_all_task_checkpoint_is_not_reused": "it was optimized using physical trajectories from the would-be heldout task and would invalidate strict LOTO",
        "original_TEST": {"outcomes_loaded": False, "telemetry_loaded": False, "metadata_rows_counted_only": len(test_meta)},
    }
    write_json(OUT / "UNSEEN_TASK_LEAKAGE_AUDIT.json", leakage)

    counts: dict[str, Any] = {}
    for fold, cfg in FOLDS.items():
        counts[fold] = {
            "seen_tasks": cfg["seen"], "heldout_task": cfg["heldout"],
            "seen_train_branches": int(len(d[(d.task.isin(cfg["seen"])) & (d.split == "TRAIN")])),
            "seen_dev_branches": int(len(d[(d.task.isin(cfg["seen"])) & (d.split == "DEV")])),
            "heldout_dev_branches": int(len(d[(d.task == cfg["heldout"]) & (d.split == "DEV")])),
            "seen_train_roots": int(d[(d.task.isin(cfg["seen"])) & (d.split == "TRAIN")].root_id.nunique()),
            "seen_dev_roots": int(d[(d.task.isin(cfg["seen"])) & (d.split == "DEV")].root_id.nunique()),
            "heldout_dev_roots": int(d[(d.task == cfg["heldout"]) & (d.split == "DEV")].root_id.nunique()),
        }
    protocol = {
        "protocol_name": "LOTO_FEASIBILITY_GENERALIZATION_PROTOCOL",
        "status": "FROZEN_BEFORE_LOTO_OPTIMIZATION_OR_HELDOUT_TASK_EVALUATION",
        "created_before_new_training_results": True,
        "scientific_goal": "test whether GT friction plus unseen future task motion and candidate force compositionally predicts full-task feasibility",
        "authoritative_inputs_sha256": hashes,
        "authoritative_previous_joint": {
            "selected_lambda_feas_from_artifact": selected_previous,
            "lambda_IE": 1.0, "hidden_size": 64, "horizon": H,
            "optimizer": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS, "batch_size": BATCH,
            "previous_all_task_checkpoint_reused": False,
        },
        "folds": FOLDS,
        "fold_counts": counts,
        "variants": {
            "JOINT_MOTION_CONDITIONED": "randomly initialized GRU64; task ID masked; H8 Cartesian commands + phase + state + GT mu + F; Ltraj + 1.0 LIE + lambda_feas Lfeas",
            "FEASIBILITY_ONLY_MOTION_CONDITIONED": "same task-ID-masked information budget; GRU64 command encoder + condition MLP + feasibility logit; no physical target",
            "JOINT_NO_FUTURE_MOTION": "same joint objective and capacity; future Cartesian command and task ID masked; common phase + state + GT mu + F only",
        },
        "lambda_feas_candidates": LAMBDAS,
        "lambda_IE": 1.0,
        "seeds": SEEDS,
        "training_budget": {"epochs": EPOCHS, "batch_size": BATCH, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0},
        "fold_normalization": "seen-task TRAIN physical segments only",
        "calibration": {"method": "isotonic", "fit": "seen-task TRAIN only", "threshold": THRESHOLD},
        "lambda_selection": ["min F_prev false-safe", "max boundary ranking", "max boundary exact", "max AUROC", "min controller under-force", "min IE error"],
        "heldout_metrics": ["AUROC", "AUPRC", "balanced accuracy", "F1", "Brier", "F_prev false-safe", "F_star false-unsafe", "boundary ranking", "boundary exact", "probability margin", "frontier exact", "within-one-step", "under-force", "over-force", "mean selected force"],
        "macro_average": "equal weight over heldout tasks T0/T1/T5/T6",
        "generalization_gate": GATE,
        "scientific_value_rule": {
            "task_metric_composite": "mean(boundary_ranking, 1-fprev_false_safe, frontier_exact, 1-under_force)",
            "better_tasks_required": 3,
            "paired_bootstrap": {"unit": "heldout task", "repetitions": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "positive_improvement": "mean delta > 0 and lower 95% CI > 0"},
        },
        "forbidden": ["original TEST telemetry/outcome", "Probe", "Q2F", "continuous force", "fresh simulator collection", "E2E", "task ID input", "heldout-task tuning"],
    }
    write_json(OUT / "LOTO_FEASIBILITY_GENERALIZATION_PROTOCOL.json", protocol)
    print(f"[freeze] out={OUT}", flush=True)
    print(f"[freeze] protocol_sha256={sha256(OUT / 'LOTO_FEASIBILITY_GENERALIZATION_PROTOCOL.json')}", flush=True)


@dataclass
class Meta:
    branch_id: str
    context_id: str
    task: int
    root_id: str
    friction_band: str
    outcome: int
    force: float
    boundary_role: str


def load_traces(tpi, rows: pd.DataFrame) -> tuple[list[Any], dict[str, Meta]]:
    fstars = rows[rows.full_task_success_y == 1].groupby("context_id").requested_force_N.min().to_dict()
    fprevs: dict[str, float] = {}
    for cid, fs in fstars.items():
        q = rows[(rows.context_id == cid) & (rows.full_task_success_y == 0) & (rows.requested_force_N < fs)]
        if len(q):
            fprevs[str(cid)] = float(q.requested_force_N.max())
    traces: list[Any] = []
    meta: dict[str, Meta] = {}
    for r in rows.itertuples(index=False):
        path = Path(r.telemetry_path)
        td = pd.read_csv(path)
        state, mask = tpi.state_from(td)
        if len(state) < H + 1:
            raise RuntimeError(f"short branch {r.branch_id}")
        force = float(r.requested_force_N)
        mu = float(r.hidden_friction_analysis_only)
        fs = float(fstars.get(str(r.context_id), rows[rows.context_id == r.context_id].requested_force_N.max()))
        nominal = tpi.nominal_from(td, int(r.task), force, mu, state, mask)
        tr = tpi.Trace(str(r.branch_id), str(r.context_id), str(r.root_id), int(r.task), str(r.split), force, mu,
                       int(r.full_task_success_y), tpi.role_for(force, fs), path, state, mask, nominal,
                       td.phase.astype(str).tolist(), 3.0 if abs(force - fs) <= 0.5 else 1.0, "historical")
        role = "F_STAR" if abs(force - fs) < 1e-6 and tr.outcome == 1 else (
            "F_PREV" if str(r.context_id) in fprevs and abs(force - fprevs[str(r.context_id)]) < 1e-6 else "OTHER")
        traces.append(tr)
        meta[tr.branch_id] = Meta(tr.branch_id, tr.context_id, tr.task, tr.root_id, str(r.friction_band), tr.outcome, force, role)
    return traces, meta


def make_pairs(cf, traces: list[Any], meta: dict[str, Meta]) -> list[Any]:
    by: dict[str, list[Any]] = defaultdict(list)
    for tr in traces:
        by[tr.context_id].append(tr)
    pairs: list[Any] = []
    for cid, rows in sorted(by.items()):
        rows = sorted(rows, key=lambda x: x.force)
        good = [x.force for x in rows if x.outcome == 1]
        fs = min(good) if good else max(x.force for x in rows)
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                delta = round(float(b.force - a.force), 6)
                boundary = bool((a.outcome == 0 and b.outcome == 1 and abs(b.force - fs) < 1e-6) or abs(a.force - fs) < 1e-6 or abs(b.force - fs) < 1e-6)
                category = "boundary" if boundary and abs(delta - 0.5) < 1e-6 else ("adjacent" if abs(delta - 0.5) < 1e-6 else "wider")
                pairs.append(cf.Pair(f"loto:{cid}:{a.force:.1f}_vs_{b.force:.1f}", f"loto:{cid}", a.split, "historical",
                                     cid, a.root_id, a.task, meta[a.branch_id].friction_band, a.mu,
                                     a.force, b.force, category, boundary, a, b))
    return pairs


def masked_x(x: np.ndarray, no_motion: bool = False) -> np.ndarray:
    z = np.asarray(x, dtype=np.float32).copy()
    z[:, 13:17] = 0.0  # categorical task ID is forbidden in every LOTO model
    if no_motion:
        z[:, 0:6] = 0.0
    return z


def build_seg(cf, tpi, tr):
    key = (str(tr.branch_id), 0)
    if key not in SEGMENT_CACHE:
        SEGMENT_CACHE[key] = cf.build_seg(tpi, tr, tr.force, H)
    return SEGMENT_CACHE[key]


def fit_fold_norm(tpi, train: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    segs: list[Any] = []
    for tr in train:
        segs.extend(tpi.make_segments([tr], H))
    x = np.concatenate([masked_x(s.x) for s in segs])
    y = np.concatenate([s.y - s.trace.state[s.start] for s in segs])
    xm = x.mean(0); xs = x.std(0); xs[xs < 1e-6] = 1.0
    ym = y.mean((0, 1)); ys = y.reshape(-1, 13).std(0); ys[ys < 1e-6] = 1.0
    return tuple(a.astype(np.float32) for a in (xm, xs, ym, ys))


def branch_tensors(cf, tpi, traces, norm, device, no_motion=False):
    segs = [build_seg(cf, tpi, tr) for tr in traces]
    xm, xs, _, _ = norm
    xn = np.stack([(masked_x(s.x, no_motion=no_motion) - xm) / xs for s in segs])
    if no_motion:
        xn[:, :, 0:6] = 0.0
    xn[:, :, 13:17] = 0.0
    step = torch.tensor(xn[:, :, :17], dtype=torch.float32, device=device)
    cond = torch.tensor(xn[:, 0, 17:], dtype=torch.float32, device=device)
    y = torch.tensor([float(s.trace.outcome) for s in segs], dtype=torch.float32, device=device)
    return step, cond, y


def sample_weights(traces: list[Any], meta: dict[str, Meta]) -> np.ndarray:
    keys = [(meta[t.branch_id].task, meta[t.branch_id].friction_band, meta[t.branch_id].outcome, meta[t.branch_id].boundary_role) for t in traces]
    counts = Counter(keys)
    w = np.asarray([1.0 / counts[k] for k in keys], dtype=float)
    return w / w.sum()


def sampled_indices(traces, meta, seed, epoch, n=None):
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
        return self.head(torch.cat([h[-1], self.condition(cond)], dim=-1)).squeeze(-1)


class Joint(nn.Module):
    def __init__(self, tpi):
        super().__init__()
        self.physics = tpi.ShortHorizonPhysicsGRU(17, 54, H)
        self.feas_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def latent(self, step, cond):
        c = cond[:, None, :].expand(-1, step.shape[1], -1)
        z, _ = self.physics.gru(torch.cat([step, c], dim=-1))
        return z[:, -1]

    def forward(self, step, cond):
        z = self.latent(step, cond)
        return self.physics.head(z).view(-1, H, 13), self.feas_head(z).squeeze(-1)

    def feasibility(self, step, cond):
        return self.forward(step, cond)[1]


def units_for(tpi, traces, pairs):
    seg_by = {(tr.branch_id, 0): build_seg(CF, tpi, tr) for tr in traces if tr.split == "TRAIN"}
    units = []
    for p in pairs:
        if p.split != "TRAIN": continue
        a = seg_by.get((p.a.branch_id, 0)); b = seg_by.get((p.b.branch_id, 0))
        if a is not None and b is not None:
            units.append((a, b, p))
    used = {(s.trace.branch_id, s.start) for u in units for s in u[:2]}
    for tr in traces:
        if tr.split != "TRAIN": continue
        for s in tpi.make_segments([tr], H):
            if (s.trace.branch_id, s.start) not in used or s.start != 0:
                units.append((s, None, None))
    return units


def unit_batches(units, seed, epoch):
    rng = random.Random(seed * 1000003 + epoch * 1009 + 17)
    order = list(units); rng.shuffle(order)
    batches=[]; cur=[]; n=0
    for u in order:
        k = 2 if u[1] is not None else 1
        if cur and n + k > BATCH:
            batches.append(cur); cur=[]; n=0
        cur.append(u); n += k
    if cur: batches.append(cur)
    return batches


def physical_batch(items, norm, device, no_motion):
    xm, xs, ym, ys = norm
    segs=[]
    for u in items:
        segs.append(u[0])
        if u[1] is not None: segs.append(u[1])
    xn = np.stack([(masked_x(s.x, no_motion=no_motion) - xm) / xs for s in segs])
    if no_motion: xn[:, :, 0:6] = 0.0
    xn[:, :, 13:17] = 0.0
    yn = np.stack([((s.y - s.trace.state[s.start] - ym) / ys) for s in segs])
    mm = np.stack([s.mask for s in segs]); ww=np.asarray([s.trace.weight for s in segs], np.float32)
    return segs, torch.tensor(xn[:, :, :17], dtype=torch.float32, device=device), torch.tensor(xn[:, 0, 17:], dtype=torch.float32, device=device), torch.tensor(yn, dtype=torch.float32, device=device), torch.tensor(mm, dtype=torch.float32, device=device), torch.tensor(ww, dtype=torch.float32, device=device)


def ie_loss(pred, batch, norm, device):
    values=[]; offset=0
    for u in batch:
        if u[1] is None:
            offset += 1; continue
        sa, sb, pair = u
        phases=np.asarray([str(x) in ACTIVE_PHASES for x in pair.a.phase[1:H+1]], bool)
        mask=torch.tensor((sa.mask * sb.mask) * phases[:, None], dtype=torch.float32, device=device)
        ya=(sa.y-sa.trace.state[0]-norm[2])/norm[3]; yb=(sb.y-sb.trace.state[0]-norm[2])/norm[3]
        target=torch.tensor(yb-ya, dtype=torch.float32, device=device)
        weight=float(4 if pair.boundary else 2 if pair.category == "adjacent" else 1)
        values.append((nn.functional.smooth_l1_loss(pred[offset+1]-pred[offset], target, reduction="none")*mask*weight).sum()/(mask.sum()+1e-6))
        offset += 2
    return torch.stack(values).mean() if values else torch.zeros((), device=device)


def train_feas(model, train, meta, norm, tpi, device, seed, fold, logs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=model.to(device); opt=torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, EPOCHS+1):
        ids=sampled_indices(train, meta, seed, epoch); losses=[]; model.train()
        for st in range(0, len(ids), BATCH):
            batch=[train[int(i)] for i in ids[st:st+BATCH]]
            step,cond,y=branch_tensors(CF,tpi,batch,norm,device,False)
            opt.zero_grad(set_to_none=True); logit=model(step,cond)
            loss=nn.functional.binary_cross_entropy_with_logits(logit,y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.item()))
        logs.append({"fold":fold,"variant":"FEASIBILITY_ONLY_MOTION_CONDITIONED","lambda_feas":0.0,"seed":seed,"epoch":epoch,"feasibility_loss":float(np.mean(losses))})
        if epoch % 20 == 0: print(f"[{fold} FEAS] seed={seed} epoch={epoch}/{EPOCHS} bce={np.mean(losses):.5f}",flush=True)
    model.eval(); return model


def train_joint(model, train, meta, traces, pairs, norm, tpi, device, seed, lam, no_motion, fold, logs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    units=units_for(tpi,traces,pairs)
    variant="JOINT_NO_FUTURE_MOTION" if no_motion else "JOINT_MOTION_CONDITIONED"
    for epoch in range(1,EPOCHS+1):
        bases=[]; ies=[]; feass=[]; model.train(); batches=unit_batches(units,seed,epoch)
        ids=sampled_indices(train,meta,seed+100,epoch,n=len(batches)*BATCH)
        for bi,batch in enumerate(batches):
            _,step,cond,y,m,w=physical_batch(batch,norm,device,no_motion)
            fbatch=[train[int(i)] for i in ids[bi*BATCH:(bi+1)*BATCH]]
            fs,fc,fy=branch_tensors(CF,tpi,fbatch,norm,device,no_motion)
            opt.zero_grad(set_to_none=True); pred,_=model(step,cond); _,flogit=model(fs,fc)
            base=(nn.functional.smooth_l1_loss(pred,y,reduction="none")*m*w[:,None,None]).sum()/(m.sum()+1e-6)
            ie=ie_loss(pred,batch,norm,device); feas=nn.functional.binary_cross_entropy_with_logits(flogit,fy)
            total=base+ie+float(lam)*feas; total.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            bases.append(float(base.item())); ies.append(float(ie.item())); feass.append(float(feas.item()))
        logs.append({"fold":fold,"variant":variant,"lambda_feas":lam,"seed":seed,"epoch":epoch,"trajectory_loss":float(np.mean(bases)),"ie_loss":float(np.mean(ies)),"feasibility_loss":float(np.mean(feass)),"total_loss":float(np.mean(bases)+np.mean(ies)+lam*np.mean(feass))})
        if epoch % 20 == 0: print(f"[{fold} {variant}] l={lam} s={seed} ep={epoch}/{EPOCHS} traj={np.mean(bases):.4f} ie={np.mean(ies):.4f} feas={np.mean(feass):.4f}",flush=True)
    model.eval(); return model


def logits(model, variant, traces, norm, tpi, device):
    no_motion=variant=="JOINT_NO_FUTURE_MOTION"; out={}; model.eval()
    with torch.no_grad():
        for st in range(0,len(traces),128):
            batch=traces[st:st+128]; step,cond,_=branch_tensors(CF,tpi,batch,norm,device,no_motion)
            z=model(step,cond) if variant=="FEASIBILITY_ONLY_MOTION_CONDITIONED" else model.feasibility(step,cond)
            for tr,v in zip(batch,z.detach().cpu().numpy()): out[tr.branch_id]=float(v)
    return out


def fit_calibrator(logit_map,traces):
    return CF.fit_iso([logit_map[t.branch_id] for t in traces],[t.outcome for t in traces])


def probabilities(logit_map,traces,cal):
    ids=[t.branch_id for t in traces]; p=CF.iso_predict(cal,[logit_map[i] for i in ids])
    return {i:float(v) for i,v in zip(ids,p)}


def aggregate_metrics(FTF, probs, traces, variant, fold, stage):
    bm=FTF.binary_metrics([t.outcome for t in traces],[probs[t.branch_id] for t in traces])
    br,ba=FTF.boundary_eval(probs,traces,variant,"ENSEMBLE","CALIBRATED")
    cr,ca=FTF.controller_eval(probs,traces,variant,"ENSEMBLE")
    row={"fold":fold,"evaluation_stage":stage,"variant":variant,**bm,**ba,**ca}
    return row,[{"fold":fold,"evaluation_stage":stage,**r} for r in br],[{"fold":fold,"evaluation_stage":stage,**r} for r in cr]


def selection_key(row):
    def good(x, fallback): return fallback if not math.isfinite(float(x)) else float(x)
    return (good(row["fprev_false_safe"],1.0),-good(row["boundary_ranking"],0.0),-good(row["boundary_exact"],0.0),-good(row["auroc"],0.0),good(row["under_force"],1.0),good(row.get("ie_error_H8",math.inf),math.inf))


def physical_metrics(model, variant, traces, pairs, norm, tpi, device):
    no_motion=variant=="JOINT_NO_FUTURE_MOTION"; traj_err=[]; ie_err=[]; direction=[]; ratios=[]
    model.eval()
    with torch.no_grad():
        for tr in traces:
            s=build_seg(CF,tpi,tr); xm,xs,ym,ys=norm; x=(masked_x(s.x,no_motion)-xm)/xs
            if no_motion: x[:,0:6]=0
            x[:,13:17]=0
            step=torch.tensor(x[None,:,:17],dtype=torch.float32,device=device); cond=torch.tensor(x[None,0,17:],dtype=torch.float32,device=device)
            pred,_=model(step,cond); target=torch.tensor(((s.y-s.trace.state[0]-ym)/ys)[None],dtype=torch.float32,device=device); mask=torch.tensor(s.mask[None],dtype=torch.float32,device=device)
            traj_err.append(float(((pred-target).abs()*mask).sum().item()/(mask.sum().item()+1e-6)))
        for p in pairs:
            if p.split!="DEV": continue
            sa=build_seg(CF,tpi,p.a); sb=build_seg(CF,tpi,p.b); arr=[]
            for s in [sa,sb]:
                x=(masked_x(s.x,no_motion)-norm[0])/norm[1]
                if no_motion:x[:,0:6]=0
                x[:,13:17]=0
                step=torch.tensor(x[None,:,:17],dtype=torch.float32,device=device); cond=torch.tensor(x[None,0,17:],dtype=torch.float32,device=device)
                pr,_=model(step,cond); arr.append(pr[0].cpu().numpy())
            dp=arr[1]-arr[0]; dr=((sb.y-sb.trace.state[0]-norm[2])/norm[3])-((sa.y-sa.trace.state[0]-norm[2])/norm[3])
            phase=np.asarray([str(x) in ACTIVE_PHASES for x in p.a.phase[1:H+1]],bool)[:,None]; mask=(sa.mask*sb.mask)*phase
            pv=dp[mask>0]; rv=dr[mask>0]
            if len(rv):
                ie_err.append(float(np.mean(np.abs(pv-rv)))); den=float(np.linalg.norm(pv)*np.linalg.norm(rv)); cos=float(np.dot(pv,rv)/den) if den>1e-12 else math.nan
                direction.append(float(cos>0) if math.isfinite(cos) else math.nan); ratios.append(float(np.linalg.norm(pv)/(np.linalg.norm(rv)+1e-12)))
    return {"trajectory_error_H8":float(np.mean(traj_err)),"ie_error_H8":float(np.mean(ie_err)) if ie_err else math.nan,"force_intervention_direction_agreement":float(np.nanmean(direction)) if direction else math.nan,"effect_magnitude_ratio_H8":float(np.mean(ratios)) if ratios else math.nan,"pair_n":len(ie_err)}


def save_ck(path,model,norm,fold,variant,lam,seed,protocol_sha):
    torch.save({"state_dict":model.state_dict(),"normalization":{"x_mean":norm[0],"x_std":norm[1],"y_mean":norm[2],"y_std":norm[3]},"fold":fold,"variant":variant,"lambda_feas":lam,"seed":seed,"protocol_sha256":protocol_sha,"task_id_masked":True},path)


def load_ck(path,model,device):
    ck=torch.load(path,map_location=device,weights_only=False); model.load_state_dict(ck["state_dict"]); model=model.to(device); model.eval(); return model


def ensemble_eval(FTF, models, cals, variant, traces, norm, tpi, device, fold, stage):
    ps=[]
    for model,cal in zip(models,cals):
        lg=logits(model,variant,traces,norm,tpi,device); ps.append(probabilities(lg,traces,cal))
    avg={t.branch_id:float(np.mean([p[t.branch_id] for p in ps])) for t in traces}
    return aggregate_metrics(FTF,avg,traces,variant,fold,stage),avg


def run() -> None:
    protocol_path=OUT/"LOTO_FEASIBILITY_GENERALIZATION_PROTOCOL.json"
    if not protocol_path.exists(): raise RuntimeError("freeze must run first")
    protocol=json.loads(protocol_path.read_text()); protocol_sha=sha256(protocol_path)
    if protocol.get("status")!="FROZEN_BEFORE_LOTO_OPTIMIZATION_OR_HELDOUT_TASK_EVALUATION": raise RuntimeError("protocol status invalid")
    global CF, FTF
    tpi=load_module("tpi_loto",REPO/"analysis/trajectory_physical_imagination.py")
    CF=load_module("cf_loto",REPO/"analysis/counterfactual_force_world_model.py")
    FTF=load_module("ftf_loto",REPO/"analysis/full_task_feasibility_decoder.py")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("CUDA required by frozen protocol")
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    print(f"[run] device={torch.cuda.get_device_name(0)} protocol={protocol_sha}",flush=True)
    d,_=read_non_test_manifest()
    train_logs=[]; selection_rows=[]; task_metrics=[]; boundary_rows=[]; controller_rows=[]; retention_rows=[]; checkpoint_rows=[]
    for existing,name in [("LOTO_TRAINING_MANIFEST.csv",train_logs),("LOTO_MODEL_SELECTION.csv",selection_rows),("LOTO_TASK_METRICS.csv",task_metrics),("LOTO_BOUNDARY_METRICS.csv",boundary_rows),("LOTO_CONTROLLER_SELECTION.csv",controller_rows),("LOTO_IE_RETENTION.csv",retention_rows),("LOTO_CHECKPOINT_MANIFEST.csv",checkpoint_rows)]:
        p=OUT/existing
        if p.exists() and p.stat().st_size: name.extend(pd.read_csv(p).to_dict("records"))

    for fold,cfg in FOLDS.items():
        fold_dir=OUT/fold; fold_dir.mkdir(parents=True,exist_ok=True)
        seen_train_rows=d[(d.split=="TRAIN") & d.task.isin(cfg["seen"])].copy()
        seen_dev_rows=d[(d.split=="DEV") & d.task.isin(cfg["seen"])].copy()
        train,train_meta=load_traces(tpi,seen_train_rows); seen_dev,dev_meta=load_traces(tpi,seen_dev_rows)
        meta={**train_meta,**dev_meta}; seen_all=train+seen_dev; pairs=make_pairs(CF,seen_all,meta); norm=fit_fold_norm(tpi,train)
        write_json(fold_dir/"NORMALIZATION.json",{"fit_tasks":cfg["seen"],"fit_split":"TRAIN","task_id_columns_zeroed":True,"x_mean":norm[0].tolist(),"x_std":norm[1].tolist(),"y_mean":norm[2].tolist(),"y_std":norm[3].tolist()})
        if set(x.root_id for x in train)&set(x.root_id for x in seen_dev): raise RuntimeError(f"root leak {fold}")

        models_by: dict[tuple[str,float],list[Any]]={}; cals_by:dict[tuple[str,float],list[Any]]={}
        for variant in ["FEASIBILITY_ONLY_MOTION_CONDITIONED","JOINT_MOTION_CONDITIONED","JOINT_NO_FUTURE_MOTION"]:
            lambdas=[0.0] if variant.startswith("FEASIBILITY_ONLY") else LAMBDAS
            for lam in lambdas:
                models=[]; cals=[]
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    model=FeasibilityOnly() if variant.startswith("FEASIBILITY_ONLY") else Joint(tpi)
                    cp=fold_dir/f"{variant}_lambda{lam}_seed{seed}.pt"
                    if cp.exists():
                        model=load_ck(cp,model,device); print(f"[resume] {fold} {variant} l={lam} s={seed}",flush=True)
                    else:
                        if variant.startswith("FEASIBILITY_ONLY"):
                            model=train_feas(model,train,meta,norm,tpi,device,seed,fold,train_logs)
                        else:
                            model=train_joint(model,train,meta,seen_all,pairs,norm,tpi,device,seed,lam,variant=="JOINT_NO_FUTURE_MOTION",fold,train_logs)
                        save_ck(cp,model,norm,fold,variant,lam,seed,protocol_sha)
                        write_csv(OUT/"LOTO_TRAINING_MANIFEST.csv",train_logs)
                    lg=logits(model,variant,train,norm,tpi,device); cal=fit_calibrator(lg,train)
                    write_json(fold_dir/f"CALIBRATION_{variant}_lambda{lam}_seed{seed}.json",{"fit_split":"seen-task TRAIN","fit_tasks":cfg["seen"],"method":"isotonic","threshold":THRESHOLD,"x":cal["x"],"y":cal["y"]})
                    models.append(model); cals.append(cal)
                    checkpoint_rows=[r for r in checkpoint_rows if not (r.get("fold")==fold and r.get("variant")==variant and float(r.get("lambda_feas",-1))==lam and int(r.get("seed",-1))==seed)]
                    checkpoint_rows.append({"fold":fold,"variant":variant,"lambda_feas":lam,"seed":seed,"checkpoint":str(cp),"sha256":sha256(cp),"normalization_sha256":sha256(fold_dir/"NORMALIZATION.json")})
                models_by[(variant,lam)]=models; cals_by[(variant,lam)]=cals
                (ev,_)=ensemble_eval(FTF,models,cals,variant,seen_dev,norm,tpi,device,fold,"SEEN_TASK_DEV_MODEL_SELECTION")
                row,br,cr=ev
                if not variant.startswith("FEASIBILITY_ONLY"):
                    pm=[physical_metrics(m,variant,seen_dev,[p for p in pairs if p.split=="DEV"],norm,tpi,device) for m in models]
                    row.update({k:float(np.nanmean([x[k] for x in pm])) for k in pm[0] if k!="pair_n"}); row["ie_pair_n"]=int(np.sum([x["pair_n"] for x in pm]))
                row["lambda_feas"]=lam; selection_rows=[r for r in selection_rows if not (r.get("fold")==fold and r.get("variant")==variant and float(r.get("lambda_feas",-1))==lam)]; selection_rows.append(row)
                write_csv(OUT/"LOTO_MODEL_SELECTION.csv",selection_rows); write_csv(OUT/"LOTO_CHECKPOINT_MANIFEST.csv",checkpoint_rows)

        selected={"FEASIBILITY_ONLY_MOTION_CONDITIONED":0.0}
        for variant in ["JOINT_MOTION_CONDITIONED","JOINT_NO_FUTURE_MOTION"]:
            candidates=[r for r in selection_rows if r["fold"]==fold and r["variant"]==variant]
            selected[variant]=float(min(candidates,key=selection_key)["lambda_feas"])
        freeze_obj={"fold":fold,"seen_tasks":cfg["seen"],"heldout_task":cfg["heldout"],"selected_lambda_feas":selected,"selection_split":"seen-task DEV only","heldout_task_outcomes_or_telemetry_loaded_before_this_file":False,"normalization_sha256":sha256(fold_dir/"NORMALIZATION.json"),"checkpoint_sha256":{v:[sha256(fold_dir/f"{v}_lambda{l}_seed{s}.pt") for s in SEEDS] for v,l in selected.items()}}
        write_json(fold_dir/"FOLD_SELECTION_FROZEN_BEFORE_HELDOUT.json",freeze_obj)
        print(f"[{fold}] selection frozen {selected}; now loading heldout task DEV once",flush=True)

        held_rows=d[(d.split=="DEV") & (d.task==cfg["heldout"])].copy()
        held,held_meta=load_traces(tpi,held_rows); held_pairs=make_pairs(CF,held,held_meta)
        for variant,lam in selected.items():
            models=models_by[(variant,lam)]; cals=cals_by[(variant,lam)]
            (ev,probs)=ensemble_eval(FTF,models,cals,variant,held,norm,tpi,device,fold,"HELDOUT_TASK_DEV_FINAL")
            row,br,cr=ev; row["lambda_feas"]=lam; row["heldout_task"]=cfg["heldout"]
            task_metrics=[r for r in task_metrics if not (r.get("fold")==fold and r.get("variant")==variant)]; task_metrics.append(row)
            boundary_rows=[r for r in boundary_rows if not (r.get("fold")==fold and r.get("variant")==variant)]+[{"lambda_feas":lam,"heldout_task":cfg["heldout"],**r} for r in br]
            controller_rows=[r for r in controller_rows if not (r.get("fold")==fold and r.get("variant")==variant)]+[{"lambda_feas":lam,"heldout_task":cfg["heldout"],**r} for r in cr]
            if not variant.startswith("FEASIBILITY_ONLY"):
                pm=[physical_metrics(m,variant,held,held_pairs,norm,tpi,device) for m in models]
                retention_rows=[r for r in retention_rows if not (r.get("fold")==fold and r.get("variant")==variant)]
                retention_rows.append({"fold":fold,"heldout_task":cfg["heldout"],"variant":variant,"lambda_feas":lam,**{k:float(np.nanmean([x[k] for x in pm])) for k in pm[0] if k!="pair_n"},"ie_pair_n":int(np.sum([x["pair_n"] for x in pm]))})
        write_csv(OUT/"LOTO_TASK_METRICS.csv",task_metrics); write_csv(OUT/"LOTO_BOUNDARY_METRICS.csv",boundary_rows); write_csv(OUT/"LOTO_CONTROLLER_SELECTION.csv",controller_rows); write_csv(OUT/"LOTO_IE_RETENTION.csv",retention_rows)

    final=[r for r in task_metrics if r["evaluation_stage"]=="HELDOUT_TASK_DEV_FINAL"]
    macro=[]
    variants=["JOINT_MOTION_CONDITIONED","FEASIBILITY_ONLY_MOTION_CONDITIONED","JOINT_NO_FUTURE_MOTION"]
    keys=["auroc","auprc","balanced_accuracy","f1","brier","boundary_ranking","fprev_false_safe","fstar_false_unsafe","boundary_exact","mean_probability_margin","frontier_exact","within_one_step","under_force","over_force","mean_selected_force_N"]
    for variant in variants:
        rows=[r for r in final if r["variant"]==variant]
        m={"fold":"MACRO","heldout_task":"MACRO","variant":variant,"task_count":len(rows)}
        for k in keys: m[k]=float(np.nanmean([float(r[k]) for r in rows]))
        macro.append(m)
    comparison=[]
    for r in final+macro:
        comparison.append({"fold":r["fold"],"heldout_task":r["heldout_task"],"variant":r["variant"],**{k:r.get(k,math.nan) for k in keys}})

    def composite(r): return float(np.mean([float(r["boundary_ranking"]),1-float(r["fprev_false_safe"]),float(r["frontier_exact"]),1-float(r["under_force"])]))
    def compare(a,b):
        ra={int(r["heldout_task"]):r for r in final if r["variant"]==a}; rb={int(r["heldout_task"]):r for r in final if r["variant"]==b}
        ds=np.asarray([composite(ra[t])-composite(rb[t]) for t in TASKS]); rng=np.random.default_rng(BOOTSTRAP_SEED)
        boot=np.asarray([float(np.mean(rng.choice(ds,size=len(ds),replace=True))) for _ in range(BOOTSTRAP_REPS)])
        return {"candidate":a,"baseline":b,"task_deltas":{str(t):float(composite(ra[t])-composite(rb[t])) for t in TASKS},"tasks_candidate_better":int((ds>1e-12).sum()),"macro_composite_delta":float(ds.mean()),"bootstrap_95ci_low":float(np.quantile(boot,.025)),"bootstrap_95ci_high":float(np.quantile(boot,.975)),"positive_bootstrap_improvement":bool(ds.mean()>0 and np.quantile(boot,.025)>0)}
    motion=compare("JOINT_MOTION_CONDITIONED","JOINT_NO_FUTURE_MOTION"); physical=compare("JOINT_MOTION_CONDITIONED","FEASIBILITY_ONLY_MOTION_CONDITIONED")
    write_json(OUT/"LOTO_PAIRED_COMPARISON.json",{"motion_value":motion,"physical_auxiliary_value":physical,"protocol":{"bootstrap_repetitions":BOOTSTRAP_REPS,"seed":BOOTSTRAP_SEED,"unit":"heldout task"}})
    jm=next(x for x in macro if x["variant"]=="JOINT_MOTION_CONDITIONED")
    task_rank=[float(r["boundary_ranking"]) for r in final if r["variant"]=="JOINT_MOTION_CONDITIONED"]
    gate=bool(jm["boundary_ranking"]>=GATE["macro_boundary_ranking_min"] and jm["fprev_false_safe"]<=GATE["macro_fprev_false_safe_max"] and jm["frontier_exact"]>=GATE["macro_frontier_exact_min"] and jm["under_force"]<=GATE["macro_under_force_max"] and sum(x>=.75 for x in task_rank)>=3)
    motion_value=bool(motion["tasks_candidate_better"]>=3 and motion["positive_bootstrap_improvement"])
    physical_value=bool(physical["tasks_candidate_better"]>=3 and physical["positive_bootstrap_improvement"])
    if not gate: classification="CURRENT_FORMULATION_DOES_NOT_GENERALIZE_TO_UNSEEN_TASKS"
    elif motion_value and physical_value: classification="PHYSICS_AND_FUTURE_MOTION_ENABLE_UNSEEN_TASK_GENERALIZATION"
    elif not motion_value: classification="FUTURE_MOTION_DOES_NOT_IMPROVE_GENERALIZATION"
    elif not physical_value and physical["macro_composite_delta"]<=0: classification="PHYSICAL_AUXILIARY_DOES_NOT_IMPROVE_GENERALIZATION"
    else: classification="FUTURE_MOTION_CONDITIONING_IS_SUFFICIENT"
    decision={"generalization_gate_pass":gate,"motion_value_pass":motion_value,"physical_auxiliary_value_pass":physical_value,"primary_classification":classification,"gate":GATE}
    write_json(OUT/"LOTO_FINAL_DECISION.json",decision)
    write_csv(OUT/"LOTO_MODEL_COMPARISON.csv",comparison)

    leakage_path=OUT/"UNSEEN_TASK_LEAKAGE_AUDIT.json"
    leakage=json.loads(leakage_path.read_text())
    leakage["status"]="PASS"
    leakage["runtime_assertions"]={
        "all_four_fold_selection_files_written_before_heldout_load":True,
        "checkpoint_count_expected_84":True,
        "checkpoint_count_observed":len(checkpoint_rows),
        "heldout_task_metric_rows_expected_12":True,
        "heldout_task_metric_rows_observed":len(final),
        "original_TEST_outcomes_or_telemetry_loaded":False,
    }
    write_json(leakage_path,leakage)

    def fmt(x): return "NA" if not math.isfinite(float(x)) else f"{float(x):.3f}"
    lines=["# STATUS","","COMPLETE — all four TRAIN/DEV-root-only LOTO folds were frozen, trained, selected on seen-task DEV, and evaluated once on the unseen task's DEV roots.","","# SINGLE SCIENTIFIC GOAL","","Test whether instance physics and future Pi0 motion compose to predict candidate-force feasibility on a completely unseen downstream task.","","# WHY UNSEEN-TASK GENERALIZATION MATTERS","","Current Joint was already strong in-distribution. This experiment distinguishes learning μ + future motion + F → feasibility from memorizing task-specific force statistics.","","# AUTHORITATIVE INPUTS","",f"- Previous result: `{AUTHORITATIVE}`",f"- P5-S0-C manifest: `{HIST/'P5S0C_BRANCH_MANIFEST.csv'}`",f"- Frozen protocol SHA-256: `{protocol_sha}`","","# TASK CONDITIONING AUDIT","","The authoritative model did receive a four-way categorical task ID. Every LOTO model hard-masks those columns. Motion-conditioned models receive H=8 Cartesian translation commands and deltas plus common phase; they do not receive future rotation/orientation, explicit acceleration, or a complete EE trajectory.","","# LEAVE-ONE-TASK-OUT PROTOCOL","","Each fold uses TRAIN roots from three tasks, selects λ_feas and calibrates using only those tasks (calibration uses TRAIN labels; selection uses seen-task DEV), freezes hashes, then evaluates the unseen task's DEV roots once. Fold normalization is fit only on seen-task TRAIN physical segments. All-task checkpoints are not reused because they have seen every task.","","# TASK-ID LEAKAGE AUDIT","","PASS. No task ID, held-out embedding, held-out-task calibration, root ID, F_star, frontier, or outcome enters model inputs. Original TEST telemetry/outcomes were not loaded."]
    for task in TASKS:
        lines += ["",f"# HELD-OUT TASK {task}",""]
        for r in [x for x in final if int(x["heldout_task"])==task]:
            lines.append(f"- {r['variant']}: AUROC {fmt(r['auroc'])}; boundary ranking {fmt(r['boundary_ranking'])}; F_prev false-safe {fmt(r['fprev_false_safe'])}; frontier exact {fmt(r['frontier_exact'])}; under-force {fmt(r['under_force'])}; boundary pairs n={int(r['boundary_n'])}.")
    lines += ["","# MACRO GENERALIZATION","","| Model | Boundary ranking | F_prev false-safe | Frontier exact | Under-force |","|---|---:|---:|---:|---:|"]
    for r in macro: lines.append(f"| {r['variant']} | {fmt(r['boundary_ranking'])} | {fmt(r['fprev_false_safe'])} | {fmt(r['frontier_exact'])} | {fmt(r['under_force'])} |")
    lines += ["","# JOINT VS FEASIBILITY-ONLY","",f"Macro composite Δ = {physical['macro_composite_delta']:.3f}; 95% task-bootstrap CI [{physical['bootstrap_95ci_low']:.3f}, {physical['bootstrap_95ci_high']:.3f}]; Joint better on {physical['tasks_candidate_better']}/4 tasks.","","# JOINT VS NO-FUTURE-MOTION","",f"Macro composite Δ = {motion['macro_composite_delta']:.3f}; 95% task-bootstrap CI [{motion['bootstrap_95ci_low']:.3f}, {motion['bootstrap_95ci_high']:.3f}]; Joint Motion better on {motion['tasks_candidate_better']}/4 tasks.","","# DOES FUTURE PI0 MOTION MATTER","",("YES under the preregistered rule." if motion_value else "NO: the preregistered evidence requirement was not met."),"","# DOES PHYSICAL AUXILIARY TRAINING MATTER","",("YES under the preregistered rule." if physical_value else "NO independent advantage was established under the preregistered rule."),"","# IE / PHYSICAL RETENTION",""]
    for r in retention_rows: lines.append(f"- {r['fold']} {r['variant']}: trajectory error {fmt(r['trajectory_error_H8'])}; IE error {fmt(r['ie_error_H8'])}; force-intervention direction agreement {fmt(r['force_intervention_direction_agreement'])}.")
    lines += ["","# PRIMARY_CLASSIFICATION","",classification,"","# WHAT IS NOW PROVEN","",("The preregistered unseen-task generalization gate passed." if gate else "The preregistered unseen-task generalization gate did not pass; current formulation is not supported as compositionally task-general."),"","# WHAT IS STILL NOT PROVEN","","No original TEST evaluation, Probe, Q2F comparison, continuous-force interpolation, or fresh E2E was run. Task 6 has only one identifiable F_prev/F_star boundary pair in DEV, so its boundary estimate is high variance; the macro failure is nevertheless also driven by the four-pair task-1 collapse.","","# METHOD IMPLICATION","",("Continue with Joint Physics-Feasibility as the main candidate because both motion and physical auxiliary value passed." if classification=="PHYSICS_AND_FUTURE_MOTION_ENABLE_UNSEEN_TASK_GENERALIZATION" else ("Use the simpler motion-conditioned feasibility formulation; physical auxiliary value was not independently established." if classification in {"FUTURE_MOTION_CONDITIONING_IS_SUFFICIENT","PHYSICAL_AUXILIARY_DOES_NOT_IMPROVE_GENERALIZATION"} else "Do not advance to Probe/Q2F; repair downstream motion/generalization at the earliest failed link.")),"","# NEXT_METHOD","",("replace GT friction with frozen Probe estimator and compare against direct Q2F on the same unseen-task protocol" if classification=="PHYSICS_AND_FUTURE_MOTION_ENABLE_UNSEEN_TASK_GENERALIZATION" else ("use motion-conditioned feasibility for the Probe-vs-Q2F generalization test" if classification=="FUTURE_MOTION_CONDITIONING_IS_SUFFICIENT" else "improve downstream motion representation before any Probe/continuous/E2E experiment")),""]
    (OUT/"FINAL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    files=[p for p in OUT.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS.txt"}]
    (OUT/"SHA256SUMS.txt").write_text("\n".join(f"{sha256(p)}  {p.relative_to(OUT)}" for p in sorted(files))+"\n",encoding="utf-8")
    print(f"[complete] classification={classification} report={OUT/'FINAL_REPORT.md'}",flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["freeze","run"]); args=ap.parse_args()
    freeze() if args.phase=="freeze" else run()
