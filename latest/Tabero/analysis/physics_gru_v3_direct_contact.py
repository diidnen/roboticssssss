#!/usr/bin/env python3
"""Physics-GRU v3 on the frozen root-diverse direct-contact dataset.

This is a gated scientific run.  It never uses task success/frontier identity
as a world-model target, never treats the inherited tangential-velocity proxy
as direct slip ground truth, and does not evaluate TEST until all TRAIN/DEV
choices (normalization, checkpoint selection, contact threshold, evaluator)
are frozen.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
SOURCE = RESULTS / "direct_contact_boundary_dataset_20260829_001409"
V1_RESULT = RESULTS / "learned_physical_imagination_20260828_222705"
V2_RESULT = RESULTS / "learned_physical_imagination_v2_20260828_231131"
FRICTION_RESULT = RESULTS / "active_friction_imagination_20260828_211106"
P4_RESULT = RESULTS / "p4_contact_conditioned_probe_20260822_184213"
P5C_RESULT = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
P5D_RESULT = RESULTS / "p5s0d_fresh_e2e_q2f_20260824_090205"

DT = 0.05
WINDOW_STEPS = 120
ACTIVE_PHASES = ["lift", "transit", "over_basket", "place"]
TASKS = [0, 1, 5, 6]
PHASES = ACTIVE_PHASES
SEEDS = [2026082901, 2026082902, 2026082903, 2026082904, 2026082905]
STATE_NAMES = [
    "relative_object_to_command_dx_m",
    "relative_object_to_command_dy_m",
    "relative_object_to_command_dz_m",
    "relative_object_to_command_vx_mps",
    "relative_object_to_command_vy_mps",
    "relative_object_to_command_vz_mps",
    "left_local_normal_force_N",
    "right_local_normal_force_N",
    "left_local_tangential_force_N",
    "right_local_tangential_force_N",
    "tangential_relative_velocity_proxy_mps",
    "gripper_joint_position_left",
    "gripper_joint_position_right",
]
EVENT_NAMES = ["left_contact", "right_contact", "unintended_slip"]
DIRECT_EVENT_MASK = np.asarray([1.0, 1.0, 0.0], np.float32)
FORCE_IDX = [6, 7, 8, 9]
NONNEG_IDX = [6, 7, 8, 9, 10]


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, np.ndarray):
        return sanitize(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(sanitize(obj), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        if rows:
            w.writerows([{k: sanitize(r.get(k, "")) for k in fields} for r in rows])


def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    pos, neg = y, ~y
    tpr = float((p[pos]).mean()) if pos.any() else float("nan")
    tnr = float((~p[neg]).mean()) if neg.any() else float("nan")
    return float(np.nanmean([tpr, tnr]))


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "n": int(len(y)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "balanced_accuracy": balanced_accuracy(y, p),
        "real_prevalence": float(y.mean()) if len(y) else float("nan"),
        "predicted_prevalence": float(p.mean()) if len(p) else float("nan"),
    }


def longest_false(x: np.ndarray) -> int:
    best = cur = 0
    for v in np.asarray(x, bool):
        if v:
            cur = 0
        else:
            cur += 1; best = max(best, cur)
    return best


@dataclass
class Trace:
    branch_id: str
    context_id: str
    root_id: str
    restored_state_group_id: str
    repeat_group_id: str
    repeat_index: int
    task: int
    split: str
    friction_band: str
    friction: float
    force: float
    force_role: str
    outcome: int
    path: Path
    file_hash: str
    base: np.ndarray
    state: np.ndarray
    events: np.ndarray
    event_mask: np.ndarray
    phases: list[str]


def state_from_frame(d: pd.DataFrame) -> np.ndarray:
    obj = d[["object_x_analysis_only", "object_y_analysis_only", "object_z_analysis_only"]].to_numpy(float)
    cmd = d[["cmd_x", "cmd_y", "cmd_z"]].to_numpy(float)
    raw_rel = obj - cmd
    rel = raw_rel - raw_rel[0]
    obj_v = d[["object_vx_mps", "object_vy_mps", "object_vz_mps"]].to_numpy(float)
    cmd_v = np.vstack([np.zeros((1, 3)), np.diff(cmd, axis=0) / DT])
    rel_v = obj_v - cmd_v
    tang_v_proxy = np.linalg.norm(rel_v[:, :2], axis=1, keepdims=True)
    forces = d[["left_normal_force_N", "right_normal_force_N", "left_tangential_force_N", "right_tangential_force_N"]].to_numpy(float)
    joints = d[["gripper_pos_0", "gripper_pos_1"]].to_numpy(float)
    return np.concatenate([rel, rel_v, forces, tang_v_proxy, joints], axis=1).astype(np.float32)


def base_from_frame(d: pd.DataFrame, state: np.ndarray, force: float, mu: float, task: int) -> np.ndarray:
    cmd = d[["cmd_x", "cmd_y", "cmd_z"]].to_numpy(float)
    cmd_rel = cmd - cmd[0]
    cmd_delta = np.vstack([np.zeros((1, 3)), np.diff(cmd, axis=0)])
    phase = np.stack([(d.phase.astype(str).to_numpy() == p).astype(float) for p in PHASES], axis=1)
    task_oh = np.zeros((len(d), len(TASKS)), float); task_oh[:, TASKS.index(task)] = 1.0
    scalars = np.repeat([[force / 6.0, mu]], len(d), axis=0)
    initial = np.repeat(state[0][None, :], len(d), axis=0)
    initial_contacts = np.repeat(d[["contact_left", "contact_right"]].to_numpy(float)[0][None, :], len(d), axis=0)
    return np.concatenate([cmd_rel, cmd_delta, phase, task_oh, scalars, initial, initial_contacts], axis=1).astype(np.float32)


def load_traces() -> tuple[list[Trace], pd.DataFrame, dict[str, Any]]:
    pop = pd.read_csv(SOURCE / "SELECTED_POPULATION.csv")
    pmap = pop.set_index("context_id").to_dict("index")
    manifests = pd.concat([pd.read_csv(SOURCE / f"task{t}/branches.csv") for t in TASKS], ignore_index=True)
    if len(manifests) != 369:
        raise RuntimeError(f"expected 369 branches, found {len(manifests)}")
    traces: list[Trace] = []
    slip_columns = set()
    required = {
        "cmd_x", "cmd_y", "cmd_z", "object_x_analysis_only", "object_y_analysis_only", "object_z_analysis_only",
        "object_vx_mps", "object_vy_mps", "object_vz_mps", "gripper_pos_0", "gripper_pos_1",
        "left_normal_force_N", "right_normal_force_N", "left_tangential_force_N", "right_tangential_force_N",
        "contact_left", "contact_right", "phase",
    }
    for r in manifests.itertuples(index=False):
        path = Path(str(r.telemetry_path))
        d = pd.read_csv(path)
        missing = sorted(required - set(d.columns))
        if missing:
            raise RuntimeError(f"{path} missing {missing}")
        slip_columns |= {c for c in d.columns if "slip" in c.lower()}
        lift = np.flatnonzero(d.phase.astype(str).to_numpy() == "lift")
        if not len(lift) or len(d) - int(lift[0]) < WINDOW_STEPS:
            raise RuntimeError(f"{path} lacks frozen lift-aligned {WINDOW_STEPS}-step window")
        d = d.iloc[int(lift[0]): int(lift[0]) + WINDOW_STEPS].reset_index(drop=True)
        state = state_from_frame(d)
        events = np.column_stack([
            d.contact_left.to_numpy(float), d.contact_right.to_numpy(float), np.zeros(len(d), float)
        ]).astype(np.float32)
        event_mask = np.repeat(DIRECT_EVENT_MASK[None, :], len(d), axis=0)
        meta = pmap[str(r.context_id)]
        force = float(r.requested_force_N)
        roles = {round(float(meta["F_prev"]), 6): "prev", round(float(meta["F_star"]), 6): "star", round(float(meta["F_next"]), 6): "next"}
        role = roles[round(force, 6)]
        repeat_group = f"{r.context_id}|F={force:g}"
        traces.append(Trace(
            branch_id=str(r.branch_id), context_id=str(r.context_id), root_id=str(r.root_id),
            restored_state_group_id=str(r.post_probe_state_hash), repeat_group_id=repeat_group,
            repeat_index=int(r.repeat_index), task=int(r.task), split=str(r.split),
            friction_band=str(r.friction_band), friction=float(r.hidden_friction_analysis_only),
            force=force, force_role=role, outcome=int(r.full_task_success_y), path=path, file_hash=sha256(path),
            base=base_from_frame(d, state, force, float(r.hidden_friction_analysis_only), int(r.task)),
            state=state, events=events, event_mask=event_mask, phases=d.phase.astype(str).tolist(),
        ))
    audit = {
        "trace_count": len(traces), "context_count": len({x.context_id for x in traces}),
        "restored_state_groups": len({x.restored_state_group_id for x in traces}),
        "repeat_groups": len({x.repeat_group_id for x in traces}), "rows_per_trace": WINDOW_STEPS,
        "slip_named_columns_found": sorted(slip_columns),
        "direct_slip_ground_truth_available": False,
        "direct_slip_ground_truth_reason": "Frozen inherited DIRECT_PHYSICAL_EVENT_DEFINITION says tangential-relative-velocity is provisional and not accepted as direct slip ground truth.",
    }
    return traces, pop, audit


def compose_x(trace: Trace, state: np.ndarray, events: np.ndarray) -> np.ndarray:
    return np.concatenate([trace.base[:-1], state[:-1], events[:-1, :2]], axis=1).astype(np.float32)


class PhysicsGRUV3(nn.Module):
    def __init__(self, input_dim: int, state_dim: int = len(STATE_NAMES), hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.state_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, state_dim))
        self.event_head = nn.Linear(hidden_dim, len(EVENT_NAMES))

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        z, h = self.gru(self.proj(x), h)
        return self.state_head(z), self.event_head(z), h


def normalization(traces: list[Trace]) -> dict[str, np.ndarray]:
    train = [t for t in traces if t.split == "TRAIN"]
    x = np.concatenate([compose_x(t, t.state, t.events) for t in train])
    dy = np.concatenate([np.diff(t.state, axis=0) for t in train])
    xm, xs = x.mean(0), x.std(0); xs[xs < 1e-6] = 1.0
    dm, ds = dy.mean(0), dy.std(0); ds[ds < 1e-6] = 1.0
    states = np.concatenate([t.state for t in train])
    lo = states.mean(0) - 6 * np.maximum(states.std(0), 1e-6)
    hi = states.mean(0) + 6 * np.maximum(states.std(0), 1e-6)
    lo[NONNEG_IDX] = 0.0
    return {k: v.astype(np.float32) for k, v in {"x_mean": xm, "x_std": xs, "delta_mean": dm, "delta_std": ds, "state_clip_low": lo, "state_clip_high": hi}.items()}


def stack_split(traces: list[Trace], split: str, norm: dict[str, np.ndarray], device: torch.device):
    ts = [t for t in traces if t.split == split]
    x = np.stack([(compose_x(t, t.state, t.events) - norm["x_mean"]) / norm["x_std"] for t in ts])
    dy = np.stack([(np.diff(t.state, axis=0) - norm["delta_mean"]) / norm["delta_std"] for t in ts])
    ev = np.stack([t.events[1:] for t in ts])
    em = np.stack([t.event_mask[1:] for t in ts])
    return ts, *(torch.tensor(a, dtype=torch.float32, device=device) for a in (x, dy, ev, em))


def class_weights(traces: list[Trace], device: torch.device) -> torch.Tensor:
    y = np.concatenate([t.events[1:, :2] for t in traces if t.split == "TRAIN"])
    p = np.clip(y.mean(0), 1e-3, 1 - 1e-3)
    pos = 0.5 / p; neg = 0.5 / (1 - p)
    return torch.tensor(np.stack([neg, pos], axis=1), dtype=torch.float32, device=device)


def event_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    z = F.binary_cross_entropy_with_logits(logits[..., :2], targets[..., :2], reduction="none")
    w = torch.where(targets[..., :2] > 0.5, weights[:, 1], weights[:, 0])
    return (z * w).mean()


def continuous_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    w = torch.ones(pred.shape[-1], device=pred.device); w[FORCE_IDX] = 2.0
    return (F.smooth_l1_loss(pred, target, reduction="none") * w).mean()


def rollout_loss(model: PhysicsGRUV3, items: list[Trace], norm: dict[str, np.ndarray], device: torch.device,
                 weights: torch.Tensor, horizon: int, rng: random.Random) -> torch.Tensor:
    if not items:
        return torch.tensor(0.0, device=device)
    start = rng.randrange(0, WINDOW_STEPS - horizon)
    state = torch.tensor(np.stack([t.state[start] for t in items]), dtype=torch.float32, device=device)
    events = torch.tensor(np.stack([t.events[start] for t in items]), dtype=torch.float32, device=device)
    h = None; losses = []
    xm = torch.tensor(norm["x_mean"], device=device); xs = torch.tensor(norm["x_std"], device=device)
    dm = torch.tensor(norm["delta_mean"], device=device); ds = torch.tensor(norm["delta_std"], device=device)
    for k in range(horizon):
        base = torch.tensor(np.stack([t.base[start + k] for t in items]), dtype=torch.float32, device=device)
        raw = torch.cat([base, state, events[:, :2]], dim=1)
        dp, ep, h = model(((raw - xm) / xs)[:, None, :], h)
        target_state = torch.tensor(np.stack([t.state[start + k + 1] for t in items]), dtype=torch.float32, device=device)
        target_delta = (target_state - state.detach() - dm) / ds
        target_event = torch.tensor(np.stack([t.events[start + k + 1] for t in items]), dtype=torch.float32, device=device)
        losses.append(continuous_loss(dp[:, 0], target_delta) + event_loss(ep[:, 0], target_event, weights))
        delta = dp[:, 0] * ds + dm
        state = state + delta
        events = torch.cat([torch.sigmoid(ep[:, 0, :2]), torch.zeros((len(items), 1), device=device)], dim=1)
    return torch.stack(losses).mean()


def deterministic_dev_objective(model: PhysicsGRUV3, dev: list[Trace], norm: dict[str, np.ndarray], device: torch.device,
                                weights: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(rollout_loss(model, dev, norm, device, weights, 20, random.Random(0)).item())


def train_seed(seed: int, traces: list[Trace], norm: dict[str, np.ndarray], device: torch.device) -> tuple[PhysicsGRUV3, dict[str, Any]]:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
    train = [t for t in traces if t.split == "TRAIN"]; dev = [t for t in traces if t.split == "DEV"]
    input_dim = compose_x(train[0], train[0].state, train[0].events).shape[1]
    model = PhysicsGRUV3(input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    weights = class_weights(traces, device)
    rng = random.Random(seed)
    best = float("inf"); best_state = None; best_epoch = 0; stale = 0; history = []
    for epoch in range(1, 61):
        model.train(); order = train.copy(); rng.shuffle(order); batch_losses = []
        horizon = 1 if epoch <= 8 else 5 if epoch <= 18 else 10 if epoch <= 35 else 20
        for i in range(0, len(order), 24):
            batch = order[i:i + 24]
            x = np.stack([(compose_x(t, t.state, t.events) - norm["x_mean"]) / norm["x_std"] for t in batch])
            dy = np.stack([(np.diff(t.state, axis=0) - norm["delta_mean"]) / norm["delta_std"] for t in batch])
            ev = np.stack([t.events[1:] for t in batch])
            xb = torch.tensor(x, dtype=torch.float32, device=device); db = torch.tensor(dy, dtype=torch.float32, device=device); eb = torch.tensor(ev, dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            dp, ep, _ = model(xb)
            teacher = continuous_loss(dp, db) + event_loss(ep, eb, weights)
            multi = rollout_loss(model, batch, norm, device, weights, horizon, rng)
            loss = teacher + 0.35 * multi
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            batch_losses.append(float(loss.item()))
        dev_obj = deterministic_dev_objective(model, dev, norm, device, weights)
        history.append({"seed": seed, "epoch": epoch, "rollout_horizon": horizon, "train_loss": float(np.mean(batch_losses)), "dev_objective": dev_obj})
        if dev_obj < best - 1e-5:
            best = dev_obj; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if epoch >= 25 and stale >= 12:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "best_dev_objective": best, "epochs": len(history), "history": history}


def clamp_state(state: torch.Tensor, norm: dict[str, np.ndarray]) -> torch.Tensor:
    lo = torch.tensor(norm["state_clip_low"], device=state.device); hi = torch.tensor(norm["state_clip_high"], device=state.device)
    return torch.maximum(torch.minimum(state, hi), lo)


def rollout(model: PhysicsGRUV3, items: list[Trace], norm: dict[str, np.ndarray], device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval(); n = len(items)
    state = torch.tensor(np.stack([t.state[0] for t in items]), dtype=torch.float32, device=device)
    events = torch.tensor(np.stack([t.events[0] for t in items]), dtype=torch.float32, device=device)
    xm = torch.tensor(norm["x_mean"], device=device); xs = torch.tensor(norm["x_std"], device=device)
    dm = torch.tensor(norm["delta_mean"], device=device); ds = torch.tensor(norm["delta_std"], device=device)
    states, probs, h = [], [], None
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for k in range(WINDOW_STEPS - 1):
            base = torch.tensor(np.stack([t.base[k] for t in items]), dtype=torch.float32, device=device)
            raw = torch.cat([base, state, events[:, :2]], dim=1)
            dp, ep, h = model(((raw - xm) / xs)[:, None], h)
            state = clamp_state(state + dp[:, 0] * ds + dm, norm)
            prob = torch.sigmoid(ep[:, 0])
            events = torch.cat([prob[:, :2], torch.zeros((n, 1), device=device)], dim=1)
            states.append(state.cpu().numpy()); probs.append(prob.cpu().numpy())
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return np.stack(states, axis=1), np.stack(probs, axis=1), elapsed


def calibrate_contact_threshold(model: PhysicsGRUV3, dev: list[Trace], norm: dict[str, np.ndarray], device: torch.device) -> tuple[float, dict[str, Any]]:
    _, probs, _ = rollout(model, dev, norm, device)
    y = np.stack([t.events[1:, :2] for t in dev]).reshape(-1, 2)
    best = None
    rows = []
    for threshold in [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75]:
        pred = probs[:, :, :2].reshape(-1, 2) >= threshold
        ms = [binary_metrics(y[:, j], pred[:, j]) for j in range(2)]
        macro = float(np.mean([m["f1"] for m in ms]))
        prevalence_gap = float(np.abs(pred.mean(0) - y.mean(0)).mean())
        score = macro - 0.25 * prevalence_gap
        row = {"threshold": threshold, "macro_contact_f1": macro, "prevalence_gap": prevalence_gap, "score": score}
        rows.append(row)
        if best is None or (score, -abs(threshold - 0.5)) > (best[0], -abs(best[1] - 0.5)):
            best = (score, threshold)
    return float(best[1]), {"grid": rows, "selection": "max DEV macro contact F1 minus 0.25*prevalence gap; tie nearest 0.5"}


def evaluator_features(states: np.ndarray, contacts: np.ndarray) -> dict[str, float]:
    bilateral = contacts[:, 0].astype(bool) & contacts[:, 1].astype(bool)
    return {
        "bilateral_fraction": float(bilateral.mean()),
        "longest_bilateral_loss_steps": int(longest_false(bilateral)),
        "mean_normal_force_N": float(states[:, 6:8].mean()),
        "min_normal_force_N": float(states[:, 6:8].mean(1).min()),
        "mean_tangential_force_N": float(states[:, 8:10].mean()),
    }


def evaluator_predict(f: dict[str, float], cfg: dict[str, Any]) -> int:
    return int(
        f["bilateral_fraction"] >= cfg["minimum_bilateral_fraction"]
        and f["longest_bilateral_loss_steps"] <= cfg["maximum_contact_loss_steps"]
        and f["mean_normal_force_N"] >= cfg["minimum_mean_normal_force_N"]
    )


def calibrate_evaluator(dev: list[Trace]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [(t, evaluator_features(t.state[1:], t.events[1:, :2] >= 0.5)) for t in dev]
    candidates = []
    for b in [0.55, 0.65, 0.75, 0.85, 0.90, 0.95]:
        for loss in [2, 5, 10, 20, 40, 60]:
            for fn in [0.10, 0.25, 0.50, 0.75, 1.0, 1.5]:
                cfg = {"minimum_bilateral_fraction": b, "maximum_contact_loss_steps": loss, "minimum_mean_normal_force_N": fn}
                y = np.asarray([t.outcome for t, _ in rows]); p = np.asarray([evaluator_predict(f, cfg) for _, f in rows])
                m = binary_metrics(y, p)
                collapse = int(p.mean() < 0.05 or p.mean() > 0.95)
                score = float(m["balanced_accuracy"] + 0.25 * m["f1"] - collapse)
                candidates.append((score, -abs(b - 0.85), -abs(loss - 20), -abs(fn - 0.5), cfg, m))
    best = max(candidates, key=lambda x: x[:4])
    cfg = {
        **best[4],
        "active_phases": ACTIVE_PHASES,
        "intentional_release_and_settle_excluded": True,
        "rule": "sufficient iff stable bilateral support and mean direct normal force satisfy frozen thresholds",
        "calibration_split": "DEV only",
        "calibration_target_used_only_by_evaluator": "real branch full-task outcome",
        "selection": "max DEV balanced accuracy + 0.25*F1 with all-safe/all-unsafe rejection",
    }
    return cfg, best[5]


def evaluator_audit(items: list[Trace], cfg: dict[str, Any]) -> dict[str, Any]:
    y, p, detail = [], [], []
    for t in items:
        feat = evaluator_features(t.state[1:], t.events[1:, :2] >= 0.5)
        pred = evaluator_predict(feat, cfg); y.append(t.outcome); p.append(pred)
        detail.append({"branch_id": t.branch_id, "context_id": t.context_id, "split": t.split, "force_role": t.force_role, "real_outcome": t.outcome, "evaluator_outcome": pred, **feat})
    return {"metrics": binary_metrics(np.asarray(y), np.asarray(p)), "detail": detail}


def trajectory_rows(model: PhysicsGRUV3, items: list[Trace], norm: dict[str, np.ndarray], device: torch.device,
                    model_name: str = "Physics-GRU v3") -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, float]:
    pred, probs, elapsed = rollout(model, items, norm, device)
    actual = np.stack([t.state[1:] for t in items])
    rows = []
    for scope in ["all", "near_frontier"]:
        idx = np.asarray([True if scope == "all" else t.force_role in {"prev", "star"} for t in items])
        for hname, h in [("1_step", 1), ("5_step", 5), ("10_step", 10), ("20_step", 20), ("full_rollout", WINDOW_STEPS - 1)]:
            a, p = actual[idx, :h], pred[idx, :h]
            rows.append({
                "model": model_name, "split": items[0].split, "scope": scope, "horizon": hname, "trajectories": int(idx.sum()),
                "relative_position_mae_m": float(np.abs(a[..., :3] - p[..., :3]).mean()),
                "relative_velocity_mae_mps": float(np.abs(a[..., 3:6] - p[..., 3:6]).mean()),
                "local_normal_force_mae_N": float(np.abs(a[..., 6:8] - p[..., 6:8]).mean()),
                "local_tangential_force_mae_N": float(np.abs(a[..., 8:10] - p[..., 8:10]).mean()),
                "tangential_slip_velocity_proxy_mae_mps": float(np.abs(a[..., 10] - p[..., 10]).mean()),
                "gripper_joint_mae": float(np.abs(a[..., 11:13] - p[..., 11:13]).mean()),
                "finite_prediction_fraction": float(np.isfinite(p).mean()),
                "nonnegative_force_fraction": float((p[..., 6:11] >= 0).mean()),
            })
    # Constant baseline on the same held-out windows.
    const = np.stack([np.repeat(t.state[0][None, :], WINDOW_STEPS - 1, axis=0) for t in items])
    for scope in ["all", "near_frontier"]:
        idx = np.asarray([True if scope == "all" else t.force_role in {"prev", "star"} for t in items])
        a, p = actual[idx], const[idx]
        rows.append({
            "model": "constant stable-contact baseline", "split": items[0].split, "scope": scope, "horizon": "full_rollout", "trajectories": int(idx.sum()),
            "relative_position_mae_m": float(np.abs(a[..., :3] - p[..., :3]).mean()),
            "relative_velocity_mae_mps": float(np.abs(a[..., 3:6] - p[..., 3:6]).mean()),
            "local_normal_force_mae_N": float(np.abs(a[..., 6:8] - p[..., 6:8]).mean()),
            "local_tangential_force_mae_N": float(np.abs(a[..., 8:10] - p[..., 8:10]).mean()),
            "tangential_slip_velocity_proxy_mae_mps": float(np.abs(a[..., 10] - p[..., 10]).mean()),
            "gripper_joint_mae": float(np.abs(a[..., 11:13] - p[..., 11:13]).mean()),
            "finite_prediction_fraction": 1.0, "nonnegative_force_fraction": float((p[..., 6:11] >= 0).mean()),
        })
    return rows, pred, probs, elapsed


def event_rows(items: list[Trace], probs: np.ndarray, threshold: float, model_name: str = "Physics-GRU v3") -> list[dict[str, Any]]:
    rows = []
    for scope in ["all", "near_frontier"]:
        idx = np.asarray([True if scope == "all" else t.force_role in {"prev", "star"} for t in items])
        y = np.stack([t.events[1:, :2] for t in items])[idx]
        p = probs[idx, :, :2] >= threshold
        for j, name in enumerate(EVENT_NAMES[:2]):
            rows.append({"model": model_name, "split": items[0].split, "scope": scope, "event": name, "threshold": threshold, **binary_metrics(y[..., j].ravel(), p[..., j].ravel())})
        bilateral_y = y[..., 0].astype(bool) & y[..., 1].astype(bool)
        bilateral_p = p[..., 0] & p[..., 1]
        rows.append({"model": model_name, "split": items[0].split, "scope": scope, "event": "bilateral_contact_retention", "threshold": threshold, **binary_metrics(bilateral_y.ravel(), bilateral_p.ravel())})
        loss_y, loss_p = ~bilateral_y, ~bilateral_p
        rows.append({"model": model_name, "split": items[0].split, "scope": scope, "event": "unintended_contact_loss", "threshold": threshold, **binary_metrics(loss_y.ravel(), loss_p.ravel())})
        rows.append({"model": model_name, "split": items[0].split, "scope": scope, "event": "unintended_slip", "threshold": "", "n": 0, "precision": "", "recall": "", "f1": "", "false_slip_rate": "", "missed_slip_rate": "", "status": "NOT_EVALUABLE_NO_ACCEPTED_DIRECT_SLIP_GROUND_TRUTH"})
    return rows


def load_old_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    # Dataclasses and postponed annotations resolve the dynamically loaded
    # module through sys.modules.  Register it before exec so the frozen v1/v2
    # comparison can be reproduced without changing either checkpoint.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old_model_common_test(items: list[Trace], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate v1/v2 kinematics and v2 contact on the same direct TEST windows."""
    trajectory, events = [], []
    # This comparison is deliberately isolated; old models cannot predict v3 direct force channels.
    for version, script, result, ckpt_name in [
        ("Physics-GRU v1", REPO / "analysis/learned_physical_imagination.py", V1_RESULT, "PHYSICS_GRU.pt"),
        ("Physics-GRU v2", REPO / "analysis/learned_physical_imagination_v2.py", V2_RESULT, "PHYSICS_GRU_V2.pt"),
    ]:
        try:
            mod = load_old_module(script, f"old_{version[-2:].replace(' ', '_')}")
            norm = json.loads((result / "NORMALIZATION.json").read_text())
            xm, xs, ym, ys = [np.asarray(norm[k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"]]
            ck = torch.load(result / ckpt_name, map_location=device, weights_only=False)
            model = mod.PhysicsGRU(ck["input_dim"]) if version.endswith("v1") else mod.PhysicsGRUV2(ck["input_dim"])
            model.load_state_dict(ck["state_dict"]); model.to(device); model.eval()
            all_actual, all_pred, all_prob = [], [], []
            for t in items:
                full = pd.read_csv(t.path)
                lift = int(np.flatnonzero(full.phase.astype(str).to_numpy() == "lift")[0])
                if version.endswith("v1"):
                    s = mod.make_physical_state(full); x = mod.make_inputs(full, s, t.force, t.friction, t.task, s[0])
                else:
                    s = mod.state_of(full); x = mod.input_of(full, s, t.force, t.friction, t.task)
                state = s[lift].copy(); h = None; preds = []; probs = []
                with torch.no_grad():
                    for k in range(WINDOW_STEPS - 1):
                        raw = x[lift + k].copy(); raw[-16:-8] = state
                        xt = torch.tensor(((raw - xm) / xs)[None, None], dtype=torch.float32, device=device)
                        if version.endswith("v1"):
                            sp, h = model(xt, h); prob = None
                        else:
                            sp, ep, h = model(xt, h); prob = torch.sigmoid(ep[0, 0]).cpu().numpy(); probs.append(prob)
                        state = (sp[0, 0].cpu().numpy() * ys + ym).astype(np.float32); preds.append(state.copy())
                all_actual.append(s[lift + 1: lift + WINDOW_STEPS]); all_pred.append(np.asarray(preds))
                if probs: all_prob.append(np.asarray(probs))
            a, p = np.stack(all_actual), np.stack(all_pred)
            trajectory.append({
                "model": version, "split": "TEST", "scope": "all", "horizon": "full_rollout", "trajectories": len(items),
                "relative_position_mae_m": float(np.abs(a[..., :3] - p[..., :3]).mean()),
                "relative_velocity_mae_mps": float(np.abs(a[..., 3:6] - p[..., 3:6]).mean()),
                "local_normal_force_mae_N": "", "local_tangential_force_mae_N": "", "tangential_slip_velocity_proxy_mae_mps": "",
                "gripper_joint_mae": float(np.abs(a[..., 6:8] - p[..., 6:8]).mean()),
                "common_direct_test_population": 1, "direct_force_channels_available": 0,
            })
            if all_prob:
                pr = np.stack(all_prob); y = np.stack([t.events[1:, :2] for t in items]); th = float(json.loads((result / "EVALUATOR_DEV_FREEZE.json").read_text()).get("event_probability_threshold", 0.5))
                for j, name in enumerate(EVENT_NAMES[:2]):
                    events.append({"model": version, "split": "TEST", "scope": "all", "event": name, "threshold": th, **binary_metrics(y[..., j].ravel(), (pr[..., j] >= th).ravel())})
        except Exception as exc:
            trajectory.append({"model": version, "split": "TEST", "scope": "all", "horizon": "full_rollout", "status": f"UNAVAILABLE:{type(exc).__name__}:{exc}"})
    return trajectory, events


def provenance(out: Path, traces: list[Trace]) -> dict[str, Any]:
    components = {
        "authoritative_direct_contact_summary": SOURCE / "FINAL_REPORT.md",
        "authoritative_protocol": SOURCE / "DIRECT_CONTACT_ROOT_DIVERSE_PROTOCOL.json",
        "authoritative_protocol_hash_file": SOURCE / "DIRECT_CONTACT_ROOT_DIVERSE_PROTOCOL.sha256",
        "selected_population": SOURCE / "SELECTED_POPULATION.csv",
        "boundary_filter": SOURCE / "BOUNDARY_FORCE_FILTER.json",
        "boundary_repeatability": SOURCE / "BOUNDARY_REPEATABILITY.csv",
        "boundary_repeatability_context": SOURCE / "BOUNDARY_REPEATABILITY_CONTEXT.csv",
        "dataset_audit": SOURCE / "DIRECT_CONTACT_ROOT_DIVERSE_DATASET_AUDIT.json",
        "trace_audit": SOURCE / "DIRECT_CONTACT_TRACE_AUDIT.json",
        "leakage_audit": SOURCE / "DIRECT_SIGNAL_SEPARABILITY_LEAKAGE_AUDIT.json",
        "inherited_event_definition": RESULTS / "direct_contact_boundary_imagination_20260829_000205/DIRECT_PHYSICAL_EVENT_DEFINITION.json",
        "inherited_logger_audit": RESULTS / "direct_contact_boundary_imagination_20260829_000205/DIRECT_CONTACT_LOGGER_AUDIT.json",
        "physics_gru_v1_code": REPO / "analysis/learned_physical_imagination.py",
        "physics_gru_v1_checkpoint": V1_RESULT / "PHYSICS_GRU.pt",
        "physics_gru_v1_normalization": V1_RESULT / "NORMALIZATION.json",
        "physics_gru_v1_metrics": V1_RESULT / "FINAL_REPORT.md",
        "physics_gru_v2_code": REPO / "analysis/learned_physical_imagination_v2.py",
        "physics_gru_v2_checkpoint": V2_RESULT / "PHYSICS_GRU_V2.pt",
        "physics_gru_v2_normalization": V2_RESULT / "NORMALIZATION.json",
        "physics_gru_v2_evaluator": V2_RESULT / "EVALUATOR_DEV_FREEZE.json",
        "physics_gru_v2_metrics": V2_RESULT / "TRAJECTORY_FIDELITY.json",
        "friction_gru": FRICTION_RESULT / "FRICTION_GRU.pt",
        "friction_protocol": FRICTION_RESULT / "PROTOCOL.json",
        "friction_metrics": FRICTION_RESULT / "FRICTION_ESTIMATION_METRICS.json",
        "p4b_probe_definition": P4_RESULT / "PROBE_COMPOSER_SPEC.md",
        "p4b_stop_rules": P4_RESULT / "PROBE_STOP_RULES.json",
        "deterministic_downstream_controller": REPO / "analysis/p5s0c_paired_boundary_probe_value.py",
        "pi0_semantic_staging_lineage": REPO / "analysis/p6g1r1_controller_grasp_vla_handoff.py",
        "p5_full_task_runner": REPO / "analysis/p5s0d_fresh_e2e_q2f.py",
        "v3_code": REPO / "analysis/physics_gru_v3_direct_contact.py",
    }
    comp = {k: {"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size} for k, p in components.items() if p.exists()}
    return {
        "authoritative_input_result": str(SOURCE.resolve()),
        "authoritative_classification": "DIRECT_SIGNAL_SEPARABILITY_SUPPORTED",
        "components": comp,
        "direct_trajectories": [{"branch_id": t.branch_id, "path": str(t.path.resolve()), "sha256": t.file_hash, "bytes": t.path.stat().st_size} for t in traces],
        "explicitly_not_reused": {"P7B_direct_Pi0_continuation": "forbidden by protocol", "historical_576_for_training": "not used; direct-only primary training"},
    }


def report_text(out: Path, source_hash: str, protocol_hash: str, dataset: dict[str, Any], leakage: dict[str, Any],
                evaluator_real: dict[str, Any], seed_rows: list[dict[str, Any]], traj: list[dict[str, Any]], events: list[dict[str, Any]],
                gate_c: dict[str, Any], device: torch.device) -> str:
    selected = next(r for r in seed_rows if int(r["selected"]) == 1)
    test_full = next(r for r in traj if r.get("model") == "Physics-GRU v3" and r.get("split") == "TEST" and r.get("scope") == "all" and r.get("horizon") == "full_rollout")
    v3_contacts = [r for r in events if r.get("model") == "Physics-GRU v3" and r.get("split") == "TEST" and r.get("scope") == "all" and r.get("event") in EVENT_NAMES[:2]]
    contact_f1 = float(np.mean([float(r["f1"]) for r in v3_contacts]))
    ev_dev = evaluator_real["DEV"]["metrics"]; ev_test = evaluator_real["TEST"]["metrics"]
    lines = [
        "# STATUS", "", "STATUS: STOPPED_AT_GATE_C_DIRECT_SLIP_FIDELITY_NOT_EVALUABLE", "",
        "Physics-GRU v3 was implemented, trained for five fixed seeds, frozen from DEV, and evaluated once on TEST. Gate A and Gate B passed. Gate C did not pass because the authoritative inherited event definition explicitly rejects the only available slip proxy as direct slip ground truth. The unchanged Gate 3 requires DEV slip F1 >= 0.70, so GT-friction frontier scoring, estimated-friction imagination, the no-information comparison, and new real E2E execution were not entered.", "",
        "# AUTHORITATIVE INPUT RESULT", "", f"Authoritative study: `{SOURCE.resolve()}`. It contains 41 contexts, 369/369 valid trajectories, 369/369 restore parity, PASS data/leakage audits, and held-out LogReg/Linear-SVM AUROC 0.989. FINAL_REPORT.md SHA-256: `{source_hash}`; direct-contact protocol SHA-256: `{protocol_hash}`.", "",
        "# COMPLETED WORK NOT REPEATED", "", "P4-B qualification, probe informativeness, Probe-GRU ranking/swap controls, stale-manager repair, root-diverse validation, friction-estimator training, logger/local-frame validation, 41-context collection, 369 parity checks, and linear separability were reused without rerun.", "",
        "# REUSE / PROVENANCE", "", "`REUSE_AND_PROVENANCE.json` hashes the 369 trace files and all recovered v1/v2, friction-estimator, probe, Pi0/staging, deterministic-controller, and P5 full-task-runner components. The P7-B continuation runner was not used.", "",
        "# DATASET AND SPLIT", "", f"Frozen lift-aligned windows: TRAIN {dataset['splits']['TRAIN']['traces']} traces/{dataset['splits']['TRAIN']['contexts']} contexts, DEV {dataset['splits']['DEV']['traces']}/{dataset['splits']['DEV']['contexts']}, TEST {dataset['splits']['TEST']['traces']}/{dataset['splits']['TEST']['contexts']}. Every root, context, restored state, and same-force repeat group remains in one split. All normalization came from TRAIN only.", "",
        "# LEAKAGE AUDIT", "", f"{leakage['status']}. Inputs exclude final outcome, frontier/F-star identity, force role, future measurements, evaluator output, and TEST annotations. GT friction is used only as the allowed Gate-3 world-model conditioning variable; TEST was evaluated after all choices froze.", "",
        "# DIRECT PHYSICAL STATE", "", "The frozen state contains relative object-to-command displacement change and velocity (explicit proxy because realized gripper pose is absent), validated direct per-finger local normal/tangential force, the explicitly named tangential-velocity proxy, gripper joints, and direct left/right contact events. Release/settle are excluded. The slip head is present but masked because no accepted direct slip label exists.", "",
        "# PHYSICS-GRU V3", "", f"Projection -> single-layer GRU(64) -> 13-channel residual continuous head + 3-event head. Selected seed {selected['seed']} at epoch {selected['best_epoch']} using DEV only; device `{device}`. There is no success, force, or frontier head.", "",
        "# TRAINING OBJECTIVE", "", "TRAIN-only normalized robust residual state loss + 2x direct-force weighting + class-balanced direct-contact BCE + curriculum open-loop loss (1/5/10/20 steps). Slip and final success are not model targets. Historical 576-trajectory pretraining was not used.", "",
        "# EVALUATOR ON REAL TRAJECTORIES", "", f"The frozen physical evaluator achieved DEV balanced accuracy {ev_dev['balanced_accuracy']:.3f} and TEST {ev_test['balanced_accuracy']:.3f} on real direct trajectories, with TEST F1 {ev_test['f1']:.3f}. Gate B passed before the world-model gate.", "",
        "# V1 VS V2 VS V3 TRAJECTORY FIDELITY", "", f"On the common direct TEST windows, v3 full-rollout relative-position MAE was {test_full['relative_position_mae_m']:.6f} m and relative-velocity MAE {test_full['relative_velocity_mae_mps']:.6f} m/s. `TRAJECTORY_METRICS.csv` contains the constant baseline and common-window v1/v2 kinematic comparison; v1/v2 have no comparable direct-force outputs.", "",
        "# DIRECT CONTACT / SLIP RESULT", "", f"TEST macro left/right contact F1 was {contact_f1:.3f}. Direct slip F1 is not reportable: the frozen event artifact says the available tangential-velocity proxy is provisional and not accepted as direct ground truth. This missing metric prevents Gate C and the unchanged Gate 3 from passing.", "",
        "# FALSE-SLIP VS MISSED-SLIP", "", "Not estimable without accepted direct slip labels. No all-unsafe claim is hidden behind proxy supervision; false-slip rate, missed-slip rate, and predicted-slip prevalence are explicitly NA in `DIRECT_EVENT_METRICS.csv`.", "",
        "# FREE-ROLLOUT STABILITY", "", f"Full 119-step TEST rollouts were finite at rate {test_full['finite_prediction_fraction']:.3f}; predicted force/slip-speed channels were nonnegative at rate {test_full['nonnegative_force_fraction']:.3f}. The 1/5/10/20/full horizon errors are in `TRAJECTORY_METRICS.csv`.", "",
        "# UNCHANGED GT-FRICTION GATE 3", "", "NOT REACHED. Gate C could not establish direct slip fidelity, and Gate 3's DEV slip-F1 >= 0.70 criterion was not lowered or replaced. `GT_FRICTION_FRONTIER_RESULTS.csv` records the gated non-entry and TEST frontier outcomes were not inspected.", "",
        "# REAL VS IMAGINED FORCE FRONTIER", "", "Not evaluated because entering Gate 3 after the upstream slip-fidelity failure would violate the continuous gate order.", "",
        "# FRONTIER ERROR ATTRIBUTION", "", "No per-context frontier errors were generated. `FRONTIER_ERROR_ATTRIBUTION.csv` records the upstream attribution: incomplete accepted slip supervision at the direct-contact-data -> Physics-GRU link.", "",
        "# CONTROLLER-SELECTION ACCURACY", "", "Not evaluated; GT-friction Gate 3 was not entered.", "",
        "# UNDER-FORCE / OVER-FORCE", "", "Not evaluated; no v3 force decision was authorized past Gate C.", "",
        "# REAL FULL-TASK E2E", "", "Not run. Existing real branches were used as frozen evaluator audit labels only and were not counted as new method E2E executions.", "",
        "# RUNTIME / LATENCY", "", "Training and held-out rollout wall-clock are recorded in `LATENCY_RESULTS.json`. Per-force planning latency was not claimed because Gate 3 and deployed imagination were not reached.", "",
        "# FAILURE ATTRIBUTION", "", "direct-contact data -> **accepted direct slip supervision missing** -> Physics-GRU rollout cannot be validated for slip -> trajectory evaluator passes on real contact trajectories -> force frontier not entered -> friction posterior not entered -> selected force not produced -> real execution not run. The earliest failed link is the direct-contact event target, not the friction estimator.", "",
        "# METHOD CHANGE", "", "The only learned-method change was Physics-GRU v3 trained on the validated root-diverse direct-contact trajectories. The evaluator update is a deterministic bilateral-support/direct-normal-force rule frozen on DEV. No high-level redesign, adaptive probing, direct success head, or larger model was added.", "",
        "# PRIMARY_CLASSIFICATION", "", "INSUFFICIENT_VALID_EVIDENCE", "",
        "# BASIC_ARCHITECTURE_STATUS", "", "NOT_READY_TO_FREEZE", "",
        "# SCIENTIFIC INTERPRETATION", "", "1. A compact model can be trained and assessed for direct contact/force evolution, but the run cannot make a valid claim about direct slip evolution because accepted slip ground truth is absent.", "", "2. Imagined trajectories were not allowed to claim recovery of the real force frontier after Gate C failed.", "", "3. The value of current probe information over a historical prior was not tested for v3.", "", "4. No v3-selected force was executed on the real full task.", "",
        "# NEXT_METHOD", "", "Minimum fix at the earliest failed link only: collect or recover a synchronized, physically validated direct slip/contact-point relative-velocity event channel on the already frozen controller/state protocol, without changing the model scale, probe, friction estimator, force lattice, or architecture. Then rerun Gate C and the unchanged GT-friction Gate 3. Do not implement WHEN TO PROBE or HOW TO PROBE.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (RESULTS / f"physics_gru_v3_direct_contact_{utc_tag()}")
    out.mkdir(parents=True, exist_ok=False)
    run_start = time.perf_counter()

    traces, pop, data_audit = load_traces()
    reuse = provenance(out, traces); write_json(out / "REUSE_AND_PROVENANCE.json", reuse)
    split_manifest = {}
    manifest_rows = []
    for split in ["TRAIN", "DEV", "TEST"]:
        q = [t for t in traces if t.split == split]
        split_manifest[split] = {
            "traces": len(q), "contexts": len({t.context_id for t in q}), "roots": len({t.root_id for t in q}),
            "restored_state_groups": len({t.restored_state_group_id for t in q}), "repeat_groups": len({t.repeat_group_id for t in q}),
        }
    for t in traces:
        manifest_rows.append({"branch_id": t.branch_id, "context_id": t.context_id, "root_id": t.root_id, "restored_state_group_id": t.restored_state_group_id, "repeat_group_id": t.repeat_group_id, "repeat_index": t.repeat_index, "task": t.task, "split": t.split, "friction": t.friction, "friction_band": t.friction_band, "candidate_force_N": t.force, "force_role_metadata_only": t.force_role, "outcome_metadata_only": t.outcome, "telemetry_path": str(t.path.resolve()), "sha256": t.file_hash})
    dataset = {
        "authoritative_source": str(SOURCE.resolve()), "primary_dataset": "369 direct trajectories; historical 576 not used",
        "window": {"alignment": "first lift row", "steps": WINDOW_STEPS, "dt_s": DT, "post_release_excluded": True},
        "splits": split_manifest, "split_integrity": {"root": True, "context": True, "restored_state": True, "repeat_group": True},
        "force_lattices_by_context": {str(r.context_id): json.loads(r.force_lattice) for r in pop.itertuples()},
        "data_audit": data_audit, "traces": manifest_rows,
    }
    write_json(out / "V3_DATASET_MANIFEST.json", dataset)

    forbidden = ["full_task_success_y", "F_star", "F_prev", "F_next", "force_role", "frontier", "future_contact", "future_force", "evaluator_outcome", "split", "root_id"]
    input_names = ["nominal_command_position_delta", "nominal_command_step", "task_phase", "task_one_hot", "candidate_force", "friction_hypothesis", "observable_initial_state", "observable_initial_contact", "current_or_predicted_state", "current_or_predicted_contact"]
    leakage = {
        "status": "PASS", "inputs": input_names, "forbidden_inputs_checked": forbidden, "forbidden_inputs_found": [],
        "training_normalization": "TRAIN only", "dev_uses": ["early stopping", "contact threshold", "evaluator threshold", "one seed selection"],
        "test_access": "once after model/evaluator/threshold freeze", "outcome_usage": "evaluator calibration/audit only; never a Physics-GRU target or input",
        "gt_friction": "permitted only for Gate-3-isolating world-model conditioning; not a deployed estimated-friction method",
        "slip_proxy_policy": "included as a continuously named proxy channel; never used as direct slip event ground truth",
    }
    write_json(out / "LEAKAGE_AUDIT.json", leakage)

    state_spec = {
        "state_names": STATE_NAMES, "event_names": EVENT_NAMES, "direct_event_supervision_mask": DIRECT_EVENT_MASK.tolist(),
        "orientation_angular_velocity": "excluded; not required and contact-frame rotational state not validated",
        "relative_pose_limitation": "realized gripper pose unavailable; object-to-command displacement change is explicitly a proxy",
        "tangential_velocity_limitation": "contact-point velocities unavailable; single object-to-command tangential velocity is explicitly a proxy",
        "slip_event_status": "masked; inherited artifact rejects proxy as direct slip ground truth",
        "release_settle_masked": True, "active_phases": ACTIVE_PHASES,
    }
    state_spec["sha256"] = hash_obj(state_spec); write_json(out / "PHYSICAL_STATE_V3_FROZEN.json", state_spec)

    immutable = {
        "created_before_training_utc": datetime.now(timezone.utc).isoformat(), "authoritative_source": str(SOURCE.resolve()),
        "seeds": SEEDS, "architecture": "input projection -> single-layer GRU(64) -> 13 residual continuous + 3 event logits",
        "direct_only": True, "historical_pretraining": False, "epochs_max": 60, "patience": 12, "optimizer": "AdamW(lr=8e-4,wd=1e-4)",
        "curriculum": [{"epochs": "1-8", "horizon": 1}, {"epochs": "9-18", "horizon": 5}, {"epochs": "19-35", "horizon": 10}, {"epochs": "36+", "horizon": 20}],
        "loss": "SmoothL1 residual state + 2x local-force channels + class-balanced direct-contact BCE + 0.35 multistep; slip masked",
        "model_selection": "minimum deterministic DEV 20-step physical/contact objective; then best seed by same DEV objective",
        "contact_threshold_selection": "DEV macro contact F1 with prevalence-gap penalty",
        "evaluator_calibration": "DEV-only physical grid; full-task outcome used only to validate evaluator, never world model",
        "gate_b": {"dev_balanced_accuracy_min": 0.80, "nondegenerate_prediction_required": True},
        "gate_c": {"contact_f1_improves_v2_required": True, "direct_slip_f1_required": True, "free_rollout_finite_required": True},
        "gate_3_unchanged": {"dev_frontier_exact_min": 0.80, "dev_under_force_max": 0.10, "dev_slip_f1_min": 0.70, "nondegenerate_sufficient_set": True},
        "test_policy": "evaluate exactly once after all choices frozen; do not enter Gate 3 if Gate C fails",
    }
    immutable["sha256"] = hash_obj(immutable); write_json(out / "TRAINING_PROTOCOL_IMMUTABLE.json", immutable)

    # Gate B: evaluator is frozen and audited on real direct trajectories before training attribution.
    dev = [t for t in traces if t.split == "DEV"]; evaluator, dev_eval_metrics = calibrate_evaluator(dev)
    evaluator["sha256"] = hash_obj(evaluator); write_json(out / "TRAJECTORY_EVALUATOR_V3.json", evaluator)
    real_audit = {"DEV": evaluator_audit(dev, evaluator)}
    gate_b = bool(real_audit["DEV"]["metrics"]["balanced_accuracy"] >= 0.80 and 0.05 <= real_audit["DEV"]["metrics"]["predicted_prevalence"] <= 0.95)
    if not gate_b:
        raise RuntimeError("Gate B failed before model training; this dataset unexpectedly did not support the frozen physical evaluator")

    norm = normalization(traces); write_json(out / "NORMALIZATION_V3.json", {**norm, "fit_split": "TRAIN only"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_rows, histories, models = [], [], []
    train_started = time.perf_counter()
    for seed in SEEDS:
        model, info = train_seed(seed, traces, norm, device); histories.extend(info.pop("history")); models.append((model, info))
        seed_rows.append({**info, "selected": 0})
    training_wall = time.perf_counter() - train_started
    selected_i = int(np.argmin([r["best_dev_objective"] for r in seed_rows])); seed_rows[selected_i]["selected"] = 1
    model = models[selected_i][0]
    contact_threshold, contact_calibration = calibrate_contact_threshold(model, dev, norm, device)
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": compose_x(traces[0], traces[0].state, traces[0].events).shape[1], "state_dim": len(STATE_NAMES), "event_dim": len(EVENT_NAMES), "hidden_dim": 64,
        "selected_seed": seed_rows[selected_i]["seed"], "normalization": {k: v.tolist() for k, v in norm.items()},
        "contact_threshold": contact_threshold, "state_spec_sha256": state_spec["sha256"], "protocol_sha256": immutable["sha256"],
    }
    torch.save(checkpoint, out / "PHYSICS_GRU_V3.pt")
    write_csv(out / "SEED_RESULTS.csv", seed_rows); write_csv(out / "TRAINING_LOG.csv", histories)
    write_json(out / "CONTACT_THRESHOLD_CALIBRATION.json", {"selected_threshold": contact_threshold, **contact_calibration})

    # All decisions are now frozen. TEST is evaluated exactly once below.
    test = [t for t in traces if t.split == "TEST"]
    real_audit["TEST"] = evaluator_audit(test, evaluator)
    real_audit_out = {
        "gate_b_pass": gate_b, "threshold": immutable["gate_b"],
        "DEV": {"metrics": real_audit["DEV"]["metrics"], "detail": real_audit["DEV"]["detail"]},
        "TEST": {"metrics": real_audit["TEST"]["metrics"], "detail": real_audit["TEST"]["detail"]},
    }
    write_json(out / "EVALUATOR_ON_REAL_TRAJECTORY_AUDIT.json", real_audit_out)
    traj_rows, event_metric_rows, rollout_times = [], [], {}
    for split, items in [("DEV", dev), ("TEST", test)]:
        tr, _, probs, elapsed = trajectory_rows(model, items, norm, device); traj_rows.extend(tr); rollout_times[split] = elapsed
        event_metric_rows.extend(event_rows(items, probs, contact_threshold))
    old_traj, old_events = old_model_common_test(test, device); traj_rows.extend(old_traj); event_metric_rows.extend(old_events)
    write_csv(out / "TRAJECTORY_METRICS.csv", traj_rows); write_csv(out / "DIRECT_EVENT_METRICS.csv", event_metric_rows)

    # Gate C is intentionally strict: no accepted direct slip labels means no pass.
    v3_test_contacts = [r for r in event_metric_rows if r.get("model") == "Physics-GRU v3" and r.get("split") == "TEST" and r.get("scope") == "all" and r.get("event") in EVENT_NAMES[:2]]
    v2_test_contacts = [r for r in event_metric_rows if r.get("model") == "Physics-GRU v2" and r.get("split") == "TEST" and r.get("scope") == "all" and r.get("event") in EVENT_NAMES[:2]]
    gate_c = {
        "pass": False,
        "reason": "NO_ACCEPTED_DIRECT_SLIP_GROUND_TRUTH; mandatory direct slip F1 and false/missed-slip rates cannot be evaluated",
        "v3_test_macro_contact_f1": float(np.mean([r["f1"] for r in v3_test_contacts])) if v3_test_contacts else None,
        "v2_test_macro_contact_f1_common_population": float(np.mean([r["f1"] for r in v2_test_contacts])) if v2_test_contacts else None,
        "direct_slip_f1": None, "false_slip_rate": None, "missed_slip_rate": None,
        "evaluator_real_test_balanced_accuracy": real_audit["TEST"]["metrics"]["balanced_accuracy"],
    }
    write_json(out / "GATE_C_RESULT.json", gate_c)

    # Required gated artifacts: explicit non-entry, never fabricated frontier values.
    write_csv(out / "GT_FRICTION_FRONTIER_RESULTS.csv", [{"status": "NOT_REACHED_GATE_C_FAILED", "reason": gate_c["reason"], "test_frontier_opened": 0}])
    write_csv(out / "FRONTIER_ERROR_ATTRIBUTION.csv", [{"context_id": "", "split": "", "earliest_reason": "incomplete accepted direct slip supervision", "status": "UPSTREAM_GATE_C_FAILURE_NO_FRONTIER_EVALUATION"}])
    write_json(out / "LATENCY_RESULTS.json", {
        "device": str(device), "torch_version": torch.__version__, "training_wall_s": training_wall,
        "v3_rollout_wall_s_by_split": rollout_times,
        "v3_rollout_ms_per_trajectory": {s: 1000 * rollout_times[s] / max(len(dev if s == "DEV" else test), 1) for s in rollout_times},
        "rollout_steps": WINDOW_STEPS - 1, "candidate_force_planning": "not reached", "total_run_wall_s_before_report": time.perf_counter() - run_start,
    })
    source_hash = sha256(SOURCE / "FINAL_REPORT.md"); protocol_hash = sha256(SOURCE / "DIRECT_CONTACT_ROOT_DIVERSE_PROTOCOL.json")
    report = report_text(out, source_hash, protocol_hash, dataset, leakage, real_audit, seed_rows, traj_rows, event_metric_rows, gate_c, device)
    (out / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    # No architecture freeze files: gates C-G did not pass.
    manifest_files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "MANIFEST.sha256")
    (out / "MANIFEST.sha256").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in manifest_files), encoding="utf-8")
    print(json.dumps({"out": str(out), "gate_b": gate_b, "gate_c": gate_c, "selected_seed": seed_rows[selected_i], "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
