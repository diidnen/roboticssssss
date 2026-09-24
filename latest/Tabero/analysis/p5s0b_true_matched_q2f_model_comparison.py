#!/usr/bin/env python3
"""P5-S0-B true-matched Query2Force model comparison.

This is an offline model-comparison runner.  It consumes the frozen P5-S0-A
true-matched dataset, trains small force-sufficiency models, and never launches
Isaac or writes into the source dataset directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


REPO = Path("/home/exouser/Tabero")
DATASET = REPO / "analysis/results/p5s0a_true_matched_dataset_20260823_172635"
RESULTS_ROOT = REPO / "analysis/results"
TASKS = [0, 1, 5, 6]
TASK_TO_IDX = {t: i for i, t in enumerate(TASKS)}
SEEN_FORCES = [3.0, 4.0, 5.0, 6.0, 8.0]
HELDOUT_FORCES = [3.5, 4.5, 5.5, 6.5, 7.0, 7.5]
DEV_TEST_FORCES = SEEN_FORCES + HELDOUT_FORCES
MODEL_SEEDS = [0, 1, 2, 3, 4]
TRAINING = {
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 300,
    "early_stopping_patience": 30,
    "gradient_clip": 1.0,
    "latent_beta": 0.01,
    "latent_beta_ablation": 0.0,
    "primary_loss": "unweighted BCEWithLogitsLoss",
    "checkpoint_selection": "DEV seen-force NLL",
}

NUMERIC_PROBE_COLS = [
    "t_s",
    "force_target",
    "measured_squeeze",
    "target_normal_force",
    "measured_fn",
    "measured_ft",
    "ft_over_fn",
    "left_fx",
    "left_fy",
    "left_fz",
    "right_fx",
    "right_fy",
    "right_fz",
    "force_imbalance",
    "force_imbalance_ratio",
    "gripper_opening",
    "contact_normal_x",
    "contact_normal_y",
    "contact_normal_z",
    "contact_tangent_x",
    "contact_tangent_y",
    "contact_tangent_z",
    "commanded_tangent_increment_mm",
    "accumulated_displacement_mm",
    "marker_motion",
    "marker_tangential",
    "marker_velocity",
    "marker_loading_unloading",
    "contact_left",
    "contact_right",
    "tactile_ok",
]
DERIVED_PROBE_COLS = ["eef_dx", "eef_dy", "eef_dz"]
EXCLUDED_PROBE_COLS = [
    "friction",
    "object_x_priv",
    "object_y_priv",
    "object_z_priv",
    "object_qw_priv",
    "object_qx_priv",
    "object_qy_priv",
    "object_qz_priv",
    "trial_id",
    "object_id",
    "seed_idx",
    "task_id",
]


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


OUT = RESULTS_ROOT / f"p5s0b_true_matched_q2f_model_comparison_{now_tag()}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def force_norm(force_n: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return 2.0 * ((force_n - 3.0) / 5.0) - 1.0


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def audit_dataset(contexts: pd.DataFrame, branches: pd.DataFrame, parity: pd.DataFrame) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if sorted(contexts["task"].unique().tolist()) != TASKS:
        errors.append(f"tasks mismatch: {sorted(contexts['task'].unique().tolist())}")
    if (contexts["task"] == 2).any() or (branches["task"] == 2).any():
        errors.append("Task2 rows present")
    if len(contexts) != 40:
        errors.append(f"context count {len(contexts)} != 40")
    if len(branches) != 296:
        errors.append(f"branch count {len(branches)} != 296")
    if len(parity) != 296 or int(parity["parity_pass"].sum()) != 296:
        errors.append("state parity is not 296/296")
    if branches["label_source"].nunique() != 1 or branches["label_source"].iloc[0] != "TRUE_POST_PROBE_RESET_MATCHED_BRANCH":
        errors.append("unexpected label source")
    if not branches["state_parity"].astype(int).eq(1).all():
        errors.append("branch state parity failure present")
    split_by_context = contexts.set_index("context_id")["split"].to_dict()
    if any(split_by_context.get(r.context_id) != r.split for r in branches.itertuples()):
        errors.append("branch split differs from context split")
    split_counts = contexts["split"].value_counts().to_dict()
    if split_counts.get("TRAIN") != 24 or split_counts.get("DEV") != 8 or split_counts.get("TEST") != 8:
        errors.append(f"bad context split counts: {split_counts}")
    train_forces = sorted(branches[branches["split"] == "TRAIN"]["requested_force_N"].unique().tolist())
    if train_forces != SEEN_FORCES:
        errors.append(f"train force values {train_forces} != {SEEN_FORCES}")
    train_force_set = set(train_forces)
    if any(f in train_force_set for f in HELDOUT_FORCES):
        errors.append("held-out force occurs in train")
    for cid, g in branches.groupby("context_id"):
        if g["split"].nunique() != 1:
            errors.append(f"context split leakage for {cid}")
        if g["post_probe_state_hash"].nunique() != 1:
            errors.append(f"multiple post-probe hashes for {cid}")
    if contexts["probe_telemetry_path"].nunique() != len(contexts):
        errors.append("probe telemetry paths not one per context")
    return len(errors) == 0, errors


def equal_mass_ece(y: np.ndarray, p: np.ndarray, bins: int = 5) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    chunks = np.array_split(order, min(bins, len(order)))
    ece = 0.0
    for idx in chunks:
        if len(idx) == 0:
            continue
        ece += len(idx) / len(y) * abs(float(y[idx].mean()) - float(p[idx].mean()))
    return float(ece)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum()))


def auprc(y: np.ndarray, score: np.ndarray, positive_label: int = 1) -> float:
    y = (np.asarray(y).astype(int) == positive_label).astype(int)
    score = np.asarray(score).astype(float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    y_ord = y[order]
    tp = np.cumsum(y_ord)
    fp = np.cumsum(1 - y_ord)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.trapezoid(precision, recall))


def metric_dict(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = p >= threshold
    failures = y == 0
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "threshold": threshold,
        "nll": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(y) else float("nan"),
        "brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
        "auroc": auroc(y, p),
        "success_auprc": auprc(y, p, 1),
        "failure_auprc": auprc(y, 1.0 - p, 0),
        "accuracy": float(np.mean(pred == y)) if len(y) else float("nan"),
        "failure_recall": float(np.mean(~pred[failures])) if failures.any() else float("nan"),
        "false_sufficient_rate": float(np.mean(pred[failures])) if failures.any() else float("nan"),
        "ece": equal_mass_ece(y, p, 5),
    }


def context_bootstrap_ci(df: pd.DataFrame, metric: str, threshold: float, n_boot: int = 1000, seed: int = 123) -> tuple[float, float, float]:
    if df.empty:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    grouped = {
        cid: (
            g["y"].to_numpy(dtype=float),
            g["p_cal"].to_numpy(dtype=float),
        )
        for cid, g in df.groupby("context_id", sort=True)
    }
    contexts = np.array(sorted(grouped))
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(contexts, size=len(contexts), replace=True)
        ys = np.concatenate([grouped[cid][0] for cid in sample])
        ps = np.concatenate([grouped[cid][1] for cid in sample])
        vals.append(metric_dict(ys, ps, threshold)[metric])
    vals = np.asarray(vals, dtype=float)
    if not np.isfinite(vals).any():
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmean(vals)), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


@dataclass
class Tensors:
    xseq: torch.Tensor
    lengths: torch.Tensor
    task: torch.Tensor
    force: torch.Tensor
    y: torch.Tensor
    branch_ids: np.ndarray
    context_ids: np.ndarray
    split: np.ndarray
    force_n: np.ndarray
    task_raw: np.ndarray


def load_feature_sequences(contexts: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, Any]]:
    raw: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    train_phases: set[str] = set()
    train_contact_states: set[str] = set()
    for row in contexts.itertuples():
        df = pd.read_csv(row.probe_telemetry_path)
        if row.split == "TRAIN":
            train_phases.update(str(x) for x in df["probe_phase"].fillna("NA").unique())
            train_contact_states.update(str(x) for x in df["contact_state"].fillna("NA").unique())
    phases = sorted(train_phases)
    contact_states = sorted(train_contact_states)
    feature_names = NUMERIC_PROBE_COLS + DERIVED_PROBE_COLS
    feature_names += [f"phase={p}" for p in phases]
    feature_names += [f"contact_state={s}" for s in contact_states]
    feature_names += ["probe_phase_unknown", "contact_state_unknown"]
    train_arrays = []
    lengths: dict[str, int] = {}
    for row in contexts.itertuples():
        df = pd.read_csv(row.probe_telemetry_path)
        arr_df = pd.DataFrame()
        for col in NUMERIC_PROBE_COLS:
            arr_df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else 0.0
        ex0 = float(pd.to_numeric(df.get("eef_x", pd.Series([0.0])), errors="coerce").iloc[0])
        ey0 = float(pd.to_numeric(df.get("eef_y", pd.Series([0.0])), errors="coerce").iloc[0])
        ez0 = float(pd.to_numeric(df.get("eef_z", pd.Series([0.0])), errors="coerce").iloc[0])
        arr_df["eef_dx"] = pd.to_numeric(df.get("eef_x", 0.0), errors="coerce") - ex0
        arr_df["eef_dy"] = pd.to_numeric(df.get("eef_y", 0.0), errors="coerce") - ey0
        arr_df["eef_dz"] = pd.to_numeric(df.get("eef_z", 0.0), errors="coerce") - ez0
        phase_vals = df["probe_phase"].fillna("NA").astype(str) if "probe_phase" in df else pd.Series(["NA"] * len(df))
        state_vals = df["contact_state"].fillna("NA").astype(str) if "contact_state" in df else pd.Series(["NA"] * len(df))
        for p in phases:
            arr_df[f"phase={p}"] = (phase_vals == p).astype(float)
        for s in contact_states:
            arr_df[f"contact_state={s}"] = (state_vals == s).astype(float)
        arr_df["probe_phase_unknown"] = (~phase_vals.isin(phases)).astype(float)
        arr_df["contact_state_unknown"] = (~state_vals.isin(contact_states)).astype(float)
        arr = arr_df.reindex(columns=feature_names).fillna(0.0).to_numpy(dtype=np.float32)
        if arr.shape[0] == 0:
            arr = np.zeros((1, len(feature_names)), dtype=np.float32)
        raw[row.context_id] = arr
        lengths[row.context_id] = int(arr.shape[0])
        if row.split == "TRAIN":
            train_arrays.append(arr)
    train_cat = np.concatenate(train_arrays, axis=0)
    mu = train_cat.mean(axis=0)
    sd = train_cat.std(axis=0)
    sd[sd < 1e-6] = 1.0
    normed = {cid: (arr - mu) / sd for cid, arr in raw.items()}
    meta = {
        "feature_names": feature_names,
        "numeric_probe_columns": NUMERIC_PROBE_COLS,
        "derived_probe_columns": DERIVED_PROBE_COLS,
        "phase_categories_from_train": phases,
        "contact_state_categories_from_train": contact_states,
        "excluded_probe_columns": EXCLUDED_PROBE_COLS,
        "normalization_fit": "TRAIN contexts only, timestep pooled",
        "mean": mu.tolist(),
        "std": sd.tolist(),
    }
    return normed, lengths, meta


def make_tensors(contexts: pd.DataFrame, branches: pd.DataFrame, seqs: dict[str, np.ndarray], lengths: dict[str, int]) -> Tensors:
    max_len = max(lengths.values())
    nfeat = next(iter(seqs.values())).shape[1]
    xs, lens, task, force, y = [], [], [], [], []
    branch_ids, context_ids, splits, force_n, task_raw = [], [], [], [], []
    for br in branches.itertuples():
        cid = br.context_id
        arr = seqs[cid]
        padded = np.zeros((max_len, nfeat), dtype=np.float32)
        padded[: arr.shape[0]] = arr
        xs.append(padded)
        lens.append(lengths[cid])
        task.append(TASK_TO_IDX[int(br.task)])
        force.append(float(force_norm(float(br.requested_force_N))))
        y.append(float(br.full_task_success_y))
        branch_ids.append(br.branch_id)
        context_ids.append(cid)
        splits.append(br.split)
        force_n.append(float(br.requested_force_N))
        task_raw.append(int(br.task))
    return Tensors(
        xseq=torch.tensor(np.stack(xs), dtype=torch.float32),
        lengths=torch.tensor(lens, dtype=torch.long),
        task=torch.tensor(task, dtype=torch.long),
        force=torch.tensor(force, dtype=torch.float32).unsqueeze(1),
        y=torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        branch_ids=np.array(branch_ids),
        context_ids=np.array(context_ids),
        split=np.array(splits),
        force_n=np.array(force_n, dtype=float),
        task_raw=np.array(task_raw, dtype=int),
    )


class TaskForceMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4, 8)
        self.net = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, xseq, lengths, task, force):
        return self.net(torch.cat([self.emb(task), force], dim=1))


class TaskForceThreshold(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4, 8)
        self.head = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, xseq, lengths, task, force):
        out = self.head(self.emb(task))
        mu_req = 2.0 + 7.0 * torch.sigmoid(out[:, :1])
        scale = 0.1 + torch.nn.functional.softplus(out[:, 1:2])
        force_n = 3.0 + 2.5 * (force + 1.0)
        return (force_n - mu_req) / scale


class ProbeGRU(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.emb = nn.Embedding(4, 8)
        self.dec = nn.Sequential(nn.Linear(32 + 8 + 1, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode(self, xseq, lengths):
        z = self.proj(xseq)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]

    def forward(self, xseq, lengths, task, force):
        h = self.encode(xseq, lengths)
        return self.dec(torch.cat([h, self.emb(task), force], dim=1))


class Q2F(nn.Module):
    def __init__(self, input_dim: int, threshold: bool, zdim: int = 4):
        super().__init__()
        self.threshold = threshold
        self.proj = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.mu = nn.Linear(32, zdim)
        self.lv = nn.Linear(32, zdim)
        self.emb = nn.Embedding(4, 8)
        if threshold:
            self.head = nn.Sequential(nn.Linear(zdim + 8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        else:
            self.head = nn.Sequential(nn.Linear(zdim + 8 + 1, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode(self, xseq, lengths):
        z = self.proj(xseq)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        last = h[-1]
        return self.mu(last), torch.clamp(self.lv(last), -7.0, 4.0)

    def logits_from_z(self, z, task, force):
        te = self.emb(task)
        if not self.threshold:
            return self.head(torch.cat([z, te, force], dim=1))
        out = self.head(torch.cat([z, te], dim=1))
        mu_req = 2.0 + 7.0 * torch.sigmoid(out[:, :1])
        scale = 0.1 + torch.nn.functional.softplus(out[:, 1:2])
        force_n = 3.0 + 2.5 * (force + 1.0)
        return (force_n - mu_req) / scale

    def forward(self, xseq, lengths, task, force, sample: bool = True):
        mu, lv = self.encode(xseq, lengths)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if sample else mu
        return self.logits_from_z(z, task, force), mu, lv


def param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def subset_mask(t: Tensors, split: str | None = None, force_role: str = "all", context_subset: set[str] | None = None) -> np.ndarray:
    m = np.ones(len(t.y), dtype=bool)
    if split is not None:
        m &= t.split == split
    if force_role == "seen":
        m &= np.isin(t.force_n, SEEN_FORCES)
    elif force_role == "heldout":
        m &= np.isin(t.force_n, HELDOUT_FORCES)
    if context_subset is not None:
        m &= np.isin(t.context_ids, list(context_subset))
    return m


def masked_tensors(t: Tensors, mask: np.ndarray, device: torch.device):
    idx = torch.tensor(mask, dtype=torch.bool)
    return (
        t.xseq[idx].to(device),
        t.lengths[idx].to(device),
        t.task[idx].to(device),
        t.force[idx].to(device),
        t.y[idx].to(device),
    )


def nll_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 1.0
    log_t = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    lg = torch.tensor(logits.reshape(-1, 1), dtype=torch.float32)
    yy = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=80)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg / torch.exp(log_t).clamp(0.05, 20.0), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).detach().clamp(0.05, 20.0).item())


def make_prefix_lengths(lengths: torch.Tensor) -> torch.Tensor:
    frac = 0.4 + 0.6 * torch.rand_like(lengths.float())
    return torch.clamp((lengths.float() * frac).long(), min=1)


def kl_diag_gauss(mu_p, lv_p, mu_q, lv_q):
    var_p = torch.exp(lv_p)
    var_q = torch.exp(lv_q)
    return 0.5 * torch.mean(torch.sum(lv_q - lv_p + (var_p + (mu_p - mu_q).pow(2)) / var_q - 1.0, dim=1))


def train_one(model_name: str, model: nn.Module, t: Tensors, seed: int, beta: float, out_dir: Path, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
    train_mask = subset_mask(t, "TRAIN", "seen")
    dev_seen_mask = subset_mask(t, "DEV", "seen")
    best_state = None
    best_dev = float("inf")
    best_epoch = -1
    patience = 0
    logs = []
    for epoch in range(TRAINING["max_epochs"]):
        model.train()
        opt.zero_grad()
        x, lengths, task, force, y = masked_tensors(t, train_mask, device)
        if isinstance(model, Q2F):
            prefix_lengths = make_prefix_lengths(lengths).to(device)
            logits, mu_t, lv_t = model(x, prefix_lengths, task, force, sample=True)
            with torch.no_grad():
                _, mu_T, lv_T = model(x, lengths, task, force, sample=False)
            pred = nll_from_logits(logits, y)
            cons = kl_diag_gauss(mu_T.detach(), lv_T.detach(), mu_t, lv_t)
            loss = pred + beta * cons
        else:
            logits = model(x, lengths, task, force)
            pred = nll_from_logits(logits, y)
            cons = torch.tensor(0.0, device=device)
            loss = pred
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["gradient_clip"])
        opt.step()
        model.eval()
        with torch.no_grad():
            dx, dl, dtask, df, dy = masked_tensors(t, dev_seen_mask, device)
            if isinstance(model, Q2F):
                dlogits, _, _ = model(dx, dl, dtask, df, sample=False)
            else:
                dlogits = model(dx, dl, dtask, df)
            dev = float(nll_from_logits(dlogits, dy).item())
        logs.append({"epoch": epoch, "train_loss": float(loss.item()), "train_pred_loss": float(pred.item()), "train_consistency": float(cons.item()), "dev_seen_nll": dev})
        if dev < best_dev - 1e-6:
            best_dev = dev
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= TRAINING["early_stopping_patience"]:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / f"{model_name}_seed{seed}.pt")
    write_csv(out_dir / f"{model_name}_seed{seed}_training_log.csv", logs)
    with torch.no_grad():
        logits = predict_logits(model, t, np.ones(len(t.y), dtype=bool), device, mc=1, return_mean_logit=True)
    dev_seen = dev_seen_mask
    temp = fit_temperature(logits[dev_seen], t.y.squeeze(1).numpy()[dev_seen])
    return {"model": model, "seed": seed, "best_epoch": best_epoch, "best_dev_seen_nll": best_dev, "temperature": temp, "param_count": param_count(model), "beta": beta}


def predict_logits(model: nn.Module, t: Tensors, mask: np.ndarray, device: torch.device, mc: int = 1, return_mean_logit: bool = True) -> np.ndarray:
    x, lengths, task, force, _ = masked_tensors(t, mask, device)
    outs = []
    with torch.no_grad():
        if isinstance(model, Q2F):
            for _ in range(mc):
                logits, _, _ = model(x, lengths, task, force, sample=(mc > 1))
                outs.append(logits.detach().cpu().numpy().reshape(-1))
        else:
            logits = model(x, lengths, task, force)
            outs.append(logits.detach().cpu().numpy().reshape(-1))
    arr = np.stack(outs, axis=0)
    return arr.mean(axis=0) if return_mean_logit else arr


def predict_probs(model: nn.Module, t: Tensors, mask: np.ndarray, device: torch.device, temp: float, mc: int = 1) -> tuple[np.ndarray, np.ndarray | None]:
    logits = predict_logits(model, t, mask, device, mc=mc, return_mean_logit=False)
    probs = sigmoid_np(logits / temp)
    if probs.shape[0] == 1:
        return probs[0], None
    return probs.mean(axis=0), np.percentile(probs, 10.0, axis=0)


def baseline_predictions(branches: pd.DataFrame) -> list[dict[str, Any]]:
    train = branches[branches["split"] == "TRAIN"]
    global_prev = float(train["full_task_success_y"].mean())
    task_prev = train.groupby("task")["full_task_success_y"].mean().to_dict()
    rows = []
    for br in branches.itertuples():
        for name, p in [
            ("ALWAYS_SUCCESS", 1.0),
            ("TRAIN_PREVALENCE", global_prev),
            ("TASK_PREVALENCE", float(task_prev[int(br.task)])),
        ]:
            rows.append({"model": name, "seed": -1, "branch_id": br.branch_id, "context_id": br.context_id, "task": int(br.task), "split": br.split, "force": float(br.requested_force_N), "y": float(br.full_task_success_y), "p_raw": p, "p_cal": p, "temperature": 1.0})
    return rows


def model_factory(name: str, input_dim: int) -> nn.Module:
    if name == "TASK_FORCE_MLP":
        return TaskForceMLP()
    if name == "TASK_FORCE_THRESHOLD":
        return TaskForceThreshold()
    if name == "DETERMINISTIC_GRU":
        return ProbeGRU(input_dim)
    if name == "Q2F_GNP":
        return Q2F(input_dim, threshold=False)
    if name == "Q2F_THRESHOLD":
        return Q2F(input_dim, threshold=True)
    raise KeyError(name)


def evaluate_predictions(pred: pd.DataFrame, contexts: pd.DataFrame, branches: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    mixed_contexts = set()
    for cid, g in branches[branches["split"] == "TEST"].groupby("context_id"):
        ys = set(g["full_task_success_y"].astype(int).tolist())
        if ys == {0, 1}:
            mixed_contexts.add(cid)
    subsets = {
        "TEST_SEEN": set(pred[(pred["split"] == "TEST") & (pred["force"].isin(SEEN_FORCES))]["branch_id"]),
        "TEST_HELDOUT": set(pred[(pred["split"] == "TEST") & (pred["force"].isin(HELDOUT_FORCES))]["branch_id"]),
        "TEST_ALL": set(pred[pred["split"] == "TEST"]["branch_id"]),
        "MIXED_CONTEXT": set(pred[(pred["split"] == "TEST") & (pred["context_id"].isin(mixed_contexts))]["branch_id"]),
        "NEGATIVE_BRANCH": set(pred[(pred["split"] == "TEST") & (pred["y"] == 0)]["branch_id"]),
        "DEV_SEEN": set(pred[(pred["split"] == "DEV") & (pred["force"].isin(SEEN_FORCES))]["branch_id"]),
    }
    rows = []
    for (model, seed), g0 in pred.groupby(["model", "seed"]):
        for subset, bids in subsets.items():
            g = g0[g0["branch_id"].isin(bids)]
            for threshold in [0.5, 0.9]:
                md = metric_dict(g["y"].to_numpy(), g["p_cal"].to_numpy(), threshold)
                rows.append({"model": model, "seed": int(seed), "subset": subset, **md})
    return rows, {"mixed_contexts": mixed_contexts}


def aggregate_metrics(eval_rows: list[dict[str, Any]], pred: pd.DataFrame) -> list[dict[str, Any]]:
    df = pd.DataFrame(eval_rows)
    rows = []
    for (model, subset, threshold), g in df.groupby(["model", "subset", "threshold"]):
        for metric in ["nll", "brier", "auroc", "success_auprc", "failure_auprc", "accuracy", "failure_recall", "false_sufficient_rate", "ece"]:
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            std = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else 0.0
            psub = pred[(pred["model"] == model) & (pred["branch_id"].isin(pred_subset_branch_ids(pred, subset)))]
            if int(g["seed"].iloc[0]) == -1:
                ci_mean, ci_lo, ci_hi = context_bootstrap_ci(psub, metric, float(threshold), seed=123)
            else:
                # Bootstrap over seed-averaged probabilities for a cleaner context CI.
                avg = psub.groupby(["branch_id", "context_id", "y"], as_index=False)["p_cal"].mean()
                ci_mean, ci_lo, ci_hi = context_bootstrap_ci(avg, metric, float(threshold), seed=123)
            rows.append({"model": model, "subset": subset, "threshold": threshold, "metric": metric, "mean": mean, "std_across_seeds": std, "context_boot_mean": ci_mean, "context_boot_ci95_lo": ci_lo, "context_boot_ci95_hi": ci_hi})
    return rows


def pred_subset_branch_ids(pred: pd.DataFrame, subset: str) -> set[str]:
    if subset == "TEST_SEEN":
        return set(pred[(pred["split"] == "TEST") & (pred["force"].isin(SEEN_FORCES))]["branch_id"])
    if subset == "TEST_HELDOUT":
        return set(pred[(pred["split"] == "TEST") & (pred["force"].isin(HELDOUT_FORCES))]["branch_id"])
    if subset == "TEST_ALL":
        return set(pred[pred["split"] == "TEST"]["branch_id"])
    if subset == "NEGATIVE_BRANCH":
        return set(pred[(pred["split"] == "TEST") & (pred["y"] == 0)]["branch_id"])
    if subset == "MIXED_CONTEXT":
        mixed = set()
        for cid, g in pred[pred["split"] == "TEST"].groupby("context_id"):
            if set(g["y"].astype(int)) == {0, 1}:
                mixed.add(cid)
        return set(pred[(pred["split"] == "TEST") & (pred["context_id"].isin(mixed))]["branch_id"])
    if subset == "DEV_SEEN":
        return set(pred[(pred["split"] == "DEV") & (pred["force"].isin(SEEN_FORCES))]["branch_id"])
    return set()


def force_selection(pred: pd.DataFrame, branches: pd.DataFrame) -> list[dict[str, Any]]:
    test_br = branches[branches["split"] == "TEST"].copy()
    rows = []
    for (model, seed), gpred in pred[pred["split"] == "TEST"].groupby(["model", "seed"]):
        for mode, prob_col in [("posterior_mean", "p_cal"), ("posterior_p10", "p10_cal")]:
            if prob_col not in gpred.columns:
                if mode == "posterior_p10":
                    continue
                prob_col = "p_cal"
            gpred = gpred.copy()
            gpred[prob_col] = pd.to_numeric(gpred[prob_col], errors="coerce")
            if gpred[prob_col].isna().all():
                if mode == "posterior_p10":
                    continue
                prob_col = "p_cal"
                gpred[prob_col] = pd.to_numeric(gpred[prob_col], errors="coerce")
            for cid, gb in test_br.groupby("context_id"):
                actual = gb[["requested_force_N", "full_task_success_y"]].sort_values("requested_force_N")
                if actual["full_task_success_y"].sum() == len(actual):
                    censored = "left"
                    emp = 3.0
                elif actual["full_task_success_y"].sum() == 0:
                    censored = "right"
                    emp = 8.0
                else:
                    censored = "none"
                    emp = float(actual[actual["full_task_success_y"] == 1]["requested_force_N"].min())
                gp = gpred[gpred["context_id"] == cid].sort_values("force")
                choose = gp[gp[prob_col] >= 0.9]
                fhat = float(choose["force"].min()) if len(choose) else 8.0
                actual_at = actual[np.isclose(actual["requested_force_N"], fhat)]
                actual_success = float(actual_at["full_task_success_y"].iloc[0]) if len(actual_at) else float("nan")
                rows.append({
                    "model": model,
                    "seed": int(seed),
                    "selection_mode": mode,
                    "context_id": cid,
                    "task": int(gb["task"].iloc[0]),
                    "selected_force_N": fhat,
                    "selected_branch_actual_success": actual_success,
                    "under_force": int(actual_success == 0),
                    "mean_requested_selected_force_N": fhat,
                    "empirical_frontier_N": emp,
                    "frontier_censoring": censored,
                    "excess_force_regret_N": max(0.0, fhat - emp) if censored == "none" else float("nan"),
                    "absolute_frontier_error_N": abs(fhat - emp) if censored == "none" else float("nan"),
                })
    return rows


def monotonicity_rows(models: dict[tuple[str, int], tuple[nn.Module, float]], contexts: pd.DataFrame, seqs: dict[str, np.ndarray], lengths: dict[str, int], feature_dim: int, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, curve_rows = [], []
    test_contexts = contexts[contexts["split"] == "TEST"].copy()
    forces = np.round(np.arange(3.0, 8.0 + 1e-9, 0.1), 1)
    max_len = max(lengths.values())
    for (model_name, seed), (model, temp) in models.items():
        model.eval()
        for ctx in test_contexts.itertuples():
            arr = seqs[ctx.context_id]
            padded = np.zeros((len(forces), max_len, feature_dim), dtype=np.float32)
            for i in range(len(forces)):
                padded[i, : arr.shape[0]] = arr
            tmp = Tensors(
                xseq=torch.tensor(padded, dtype=torch.float32),
                lengths=torch.tensor([lengths[ctx.context_id]] * len(forces), dtype=torch.long),
                task=torch.tensor([TASK_TO_IDX[int(ctx.task)]] * len(forces), dtype=torch.long),
                force=torch.tensor(force_norm(forces), dtype=torch.float32).unsqueeze(1),
                y=torch.zeros((len(forces), 1)),
                branch_ids=np.array([f"{ctx.context_id}_dense_{f:.1f}" for f in forces]),
                context_ids=np.array([ctx.context_id] * len(forces)),
                split=np.array(["TEST"] * len(forces)),
                force_n=forces,
                task_raw=np.array([int(ctx.task)] * len(forces)),
            )
            # Monotonicity is a property of the predicted force curve for a
            # fixed inferred context.  Use the posterior mean latent here;
            # independent MC samples at adjacent force queries would add
            # sampling noise and can create artificial violations.
            p, _ = predict_probs(model, tmp, np.ones(len(forces), dtype=bool), device, temp, mc=1)
            diffs = np.diff(p)
            bad = diffs < -1e-6
            rows.append({
                "model": model_name,
                "seed": seed,
                "context_id": ctx.context_id,
                "task": int(ctx.task),
                "has_violation": int(bad.any()),
                "violation_count": int(bad.sum()),
                "max_probability_decrease": float(max(0.0, -diffs.min())) if len(diffs) else 0.0,
            })
            for f, prob in zip(forces, p):
                curve_rows.append({"model": model_name, "seed": seed, "context_id": ctx.context_id, "task": int(ctx.task), "requested_force_N": float(f), "predicted_success_probability": float(prob)})
    return rows, curve_rows


def probe_shuffle_eval(models: dict[tuple[str, int], tuple[nn.Module, float]], t: Tensors, branches: pd.DataFrame, device: torch.device, n_perm: int = 100) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(20260823)
    test_mask = subset_mask(t, "TEST", "all")
    test_contexts = sorted(set(t.context_ids[test_mask]))
    for (model_name, seed), (model, temp) in models.items():
        if model_name not in {"DETERMINISTIC_GRU", "Q2F_GNP", "Q2F_THRESHOLD"}:
            continue
        base_p, _ = predict_probs(model, t, test_mask, device, temp, mc=50 if isinstance(model, Q2F) else 1)
        base_y = t.y.squeeze(1).numpy()[test_mask]
        base_df = pd.DataFrame({"context_id": t.context_ids[test_mask], "y": base_y, "p_cal": base_p})
        base_metrics = metric_dict(base_y, base_p, 0.9)
        context_to_indices = {cid: np.where(t.context_ids == cid)[0] for cid in set(t.context_ids)}
        same_task = {}
        for task in TASKS:
            same_task[task] = sorted(set(t.context_ids[test_mask & (t.task_raw == task)]))
        vals = []
        for _ in range(n_perm):
            xperm = t.xseq.clone()
            lperm = t.lengths.clone()
            for task in TASKS:
                cids = same_task[task]
                perm = rng.permutation(cids)
                for src, dst in zip(cids, perm):
                    src_idx = context_to_indices[src][0]
                    for j in context_to_indices[dst]:
                        xperm[j] = t.xseq[src_idx]
                        lperm[j] = t.lengths[src_idx]
            tt = Tensors(xperm, lperm, t.task, t.force, t.y, t.branch_ids, t.context_ids, t.split, t.force_n, t.task_raw)
            p, _ = predict_probs(model, tt, test_mask, device, temp, mc=20 if isinstance(model, Q2F) else 1)
            vals.append(metric_dict(base_y, p, 0.9))
        for metric in ["nll", "brier", "failure_auprc", "false_sufficient_rate"]:
            arr = np.array([v[metric] for v in vals], dtype=float)
            rows.append({
                "model": model_name,
                "seed": seed,
                "diagnostic": "probe_shuffle_within_task_test_split",
                "metric": metric,
                "base": base_metrics[metric],
                "permuted_mean": float(np.nanmean(arr)),
                "permuted_std": float(np.nanstd(arr, ddof=1)),
                "degradation": float(np.nanmean(arr) - base_metrics[metric]) if metric in {"nll", "brier", "false_sufficient_rate"} else float(base_metrics[metric] - np.nanmean(arr)),
            })
        # Mean-evidence replacement.
        train_mask = subset_mask(t, "TRAIN", "seen")
        mean_seq = t.xseq[torch.tensor(train_mask)].mean(dim=0, keepdim=True)
        mean_len = int(torch.median(t.lengths[torch.tensor(train_mask)].float()).item())
        xmean = t.xseq.clone()
        lmean = t.lengths.clone()
        for j in np.where(test_mask)[0]:
            xmean[j] = mean_seq[0]
            lmean[j] = mean_len
        tt = Tensors(xmean, lmean, t.task, t.force, t.y, t.branch_ids, t.context_ids, t.split, t.force_n, t.task_raw)
        pmean, _ = predict_probs(model, tt, test_mask, device, temp, mc=20 if isinstance(model, Q2F) else 1)
        mm = metric_dict(base_y, pmean, 0.9)
        for metric in ["nll", "brier", "failure_auprc", "false_sufficient_rate"]:
            rows.append({
                "model": model_name,
                "seed": seed,
                "diagnostic": "train_mean_evidence_replacement",
                "metric": metric,
                "base": base_metrics[metric],
                "replacement": mm[metric],
                "degradation": float(mm[metric] - base_metrics[metric]) if metric in {"nll", "brier", "false_sufficient_rate"} else float(base_metrics[metric] - mm[metric]),
            })
    return rows


def posterior_diagnostic(models: dict[tuple[str, int], tuple[nn.Module, float]], t: Tensors, device: torch.device) -> list[dict[str, Any]]:
    rows = []
    all_mask = np.ones(len(t.y), dtype=bool)
    x, lengths, task, force, _ = masked_tensors(t, all_mask, device)
    for (model_name, seed), (model, _temp) in models.items():
        if not isinstance(model, Q2F):
            continue
        model.eval()
        with torch.no_grad():
            mu, lv = model.encode(x, lengths)
            var = torch.exp(lv).detach().cpu().numpy()
            mu_np = mu.detach().cpu().numpy()
        df = pd.DataFrame(
            {
                "context_id": t.context_ids,
                "split": t.split,
                "task": t.task_raw,
                "posterior_var_mean": var.mean(axis=1),
                "posterior_var_max": var.max(axis=1),
                "posterior_mu_norm": np.linalg.norm(mu_np, axis=1),
            }
        ).drop_duplicates(["context_id"])
        for r in df.itertuples():
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "context_id": r.context_id,
                    "split": r.split,
                    "task": int(r.task),
                    "posterior_var_mean": float(r.posterior_var_mean),
                    "posterior_var_max": float(r.posterior_var_max),
                    "posterior_mu_norm": float(r.posterior_mu_norm),
                }
            )
    return rows


def force_telemetry_diagnostic(branches: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for br in branches.itertuples():
        p = Path(br.telemetry_path)
        peak_phase = ""
        transport_mean = float("nan")
        if p.exists():
            df = pd.read_csv(p)
            if len(df):
                idx = pd.to_numeric(df["measured_force_N"], errors="coerce").idxmax()
                peak_phase = str(df.loc[idx, "phase"])
                tm = df[df["phase"].astype(str).str.contains("transport|move|place", case=False, regex=True, na=False)]
                if len(tm):
                    transport_mean = float(pd.to_numeric(tm["measured_force_N"], errors="coerce").mean())
        rows.append({
            "branch_id": br.branch_id,
            "context_id": br.context_id,
            "task": int(br.task),
            "split": br.split,
            "requested_force_N": float(br.requested_force_N),
            "steady_state_mean_N": float(br.steady_state_mean_N),
            "transport_phase_mean_N": transport_mean,
            "raw_instantaneous_peak_N": float(br.measured_force_peak_N),
            "raw_peak_phase": peak_phase,
            "tracking_mae_N": float(br.force_tracking_mae_N),
            "model_input": False,
        })
    return rows


def best_metric(summary: pd.DataFrame, subset: str, metric: str, threshold: float = 0.9, higher_is_better: bool = False) -> dict[str, Any]:
    g = summary[(summary["subset"] == subset) & (summary["metric"] == metric) & (summary["threshold"] == threshold)].copy()
    if g.empty:
        return {"model": "NA", "value": float("nan")}
    g = g.dropna(subset=["mean"])
    if g.empty:
        return {"model": "NA", "value": float("nan")}
    row = g.loc[g["mean"].idxmax() if higher_is_better else g["mean"].idxmin()]
    return {"model": row["model"], "value": float(row["mean"])}


def classify(summary: pd.DataFrame, shuffle: pd.DataFrame, selection: pd.DataFrame) -> tuple[str, list[str]]:
    reasons = []
    def val(model, subset, metric, threshold=0.9):
        g = summary[(summary["model"] == model) & (summary["subset"] == subset) & (summary["metric"] == metric) & (summary["threshold"] == threshold)]
        return float(g["mean"].iloc[0]) if len(g) else float("nan")
    probe_models = ["DETERMINISTIC_GRU", "Q2F_GNP", "Q2F_THRESHOLD"]
    task_refs = ["TASK_FORCE_MLP", "TASK_FORCE_THRESHOLD"]
    qualified_probe_value_model = None
    for pm in probe_models:
        nll = val(pm, "TEST_ALL", "nll")
        brier = val(pm, "TEST_ALL", "brier")
        beats_task = all(nll < val(tm, "TEST_ALL", "nll") and brier < val(tm, "TEST_ALL", "brier") for tm in task_refs)
        failure_signal = (
            val(pm, "TEST_ALL", "failure_auprc") > max(val(tm, "TEST_ALL", "failure_auprc") for tm in task_refs)
            or val(pm, "TEST_ALL", "false_sufficient_rate") < min(val(tm, "TEST_ALL", "false_sufficient_rate") for tm in task_refs)
        )
        shuffle_degrades = False
        if not shuffle.empty:
            sg = shuffle[(shuffle["model"] == pm) & (shuffle["diagnostic"] == "probe_shuffle_within_task_test_split") & (shuffle["metric"] == "nll")]
            mg = shuffle[(shuffle["model"] == pm) & (shuffle["diagnostic"] == "train_mean_evidence_replacement") & (shuffle["metric"] == "nll")]
            shuffle_degrades = (len(sg) and float(sg["degradation"].mean()) > 0) and (len(mg) and float(mg["degradation"].mean()) > 0)
        sel = selection.groupby("model")["under_force"].mean().to_dict() if not selection.empty else {}
        selection_signal = pm in sel and sel[pm] <= min(sel.get(tm, 1.0) for tm in task_refs)
        if beats_task:
            reasons.append(f"{pm} beats Task+F baselines on TEST_ALL NLL/Brier")
        if beats_task and failure_signal and selection_signal and shuffle_degrades:
            qualified_probe_value_model = pm
    if qualified_probe_value_model:
        reasons.append(f"{qualified_probe_value_model} also degrades under probe shuffle/mean-evidence replacement")
        return "P5S0B_TRUE_MATCHED_PROBE_VALUE_DEVELOPMENT_SIGNAL", reasons
    offgrid_best = best_metric(summary, "TEST_HELDOUT", "nll")
    seen_best = best_metric(summary, "TEST_SEEN", "nll")
    if math.isfinite(offgrid_best["value"]) and math.isfinite(seen_best["value"]) and offgrid_best["value"] > seen_best["value"] + 0.3:
        return "P5S0B_SEEN_FORCE_WORKS_BUT_OFFGRID_GENERALIZATION_FAILS", reasons + ["held-out force NLL is substantially worse than seen-force NLL"]
    if min([val(m, "TEST_ALL", "failure_recall") for m in probe_models + task_refs]) < 0.25:
        return "P5S0B_CLASS_IMBALANCE_COLLAPSE", reasons + ["failure recall is poor under the 0.9 sufficiency threshold"]
    return "P5S0B_PIPELINE_WORKS_BUT_NO_PROBE_VALUE_DETECTED", reasons + ["no single probe-conditioned model both improved over Task+F and degraded under evidence perturbation"]


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    for d in [
        "P5S0B_ALWAYS_SUCCESS",
        "P5S0B_TASK_FORCE_MLP",
        "P5S0B_TASK_FORCE_THRESHOLD",
        "P5S0B_DETERMINISTIC_GRU",
        "P5S0B_Q2F_GNP",
        "P5S0B_Q2F_THRESHOLD",
        "P5S0B_CHECKPOINTS",
        "P5S0B_TRAINING_LOGS",
        "P5S0B_PREDICTED_FORCE_CURVES",
    ]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "P5-S0-B True-Matched Query2Force Model Comparison",
        "method_change": "NONE",
        "dataset": str(DATASET),
        "tasks": TASKS,
        "task2_used": False,
        "no_new_simulation_data": True,
        "no_new_e2e_rollouts": True,
        "models": ["ALWAYS_SUCCESS", "TRAIN_PREVALENCE", "TASK_PREVALENCE", "TASK_FORCE_MLP", "TASK_FORCE_THRESHOLD", "DETERMINISTIC_GRU", "Q2F_GNP", "Q2F_THRESHOLD"],
        "model_seeds": MODEL_SEEDS,
        "training": TRAINING,
    }
    write_json(OUT / "P5S0B_PROTOCOL.json", protocol)
    (OUT / "P5S0B_PROTOCOL_HASH.txt").write_text(sha256_obj(protocol) + "\n", encoding="utf-8")
    write_json(OUT / "P5S0B_CODE_HASH.txt", {"git_commit": git_commit(), "script_sha256": sha256_file(Path(__file__))})

    contexts = pd.read_csv(DATASET / "P5S0A_CONTEXT_MANIFEST.csv")
    branches = pd.read_csv(DATASET / "P5S0A_BRANCH_MANIFEST.csv")
    parity = pd.read_csv(DATASET / "P5S0A_STATE_PARITY.csv")
    ok, audit_errors = audit_dataset(contexts, branches, parity)
    audit_md = [
        "# P5-S0-B Data Audit",
        f"Dataset: `{DATASET}`",
        f"Audit status: {'PASS' if ok else 'FAIL'}",
        "",
        f"- Contexts: {len(contexts)}",
        f"- Branches: {len(branches)}",
        f"- Parity passes: {int(parity['parity_pass'].sum())}/{len(parity)}",
        f"- Train forces: {sorted(branches[branches['split']=='TRAIN']['requested_force_N'].unique().tolist())}",
        f"- Held-out force in train: {bool(set(HELDOUT_FORCES) & set(branches[branches['split']=='TRAIN']['requested_force_N'].unique().tolist()))}",
        f"- Task2 rows: {int((contexts['task']==2).sum() + (branches['task']==2).sum())}",
        f"- Friction used as model input: False",
        "",
        "Errors:",
    ] + [f"- {e}" for e in audit_errors]
    (OUT / "P5S0B_DATA_AUDIT.md").write_text("\n".join(audit_md) + "\n", encoding="utf-8")
    if not ok:
        verdict = {"STATUS": "FAIL", "PRIMARY_CLASSIFICATION": "P5S0B_INVALID_DATA_OR_SPLIT", "audit_errors": audit_errors}
        write_json(OUT / "P5S0B_FINAL_VERDICT.json", verdict)
        print("STATUS:\nFAIL\nPRIMARY_CLASSIFICATION:\nP5S0B_INVALID_DATA_OR_SPLIT")
        return

    seqs, lengths, feature_meta = load_feature_sequences(contexts)
    t = make_tensors(contexts, branches, seqs, lengths)
    input_dim = t.xseq.shape[-1]
    write_json(OUT / "P5S0B_FEATURE_MANIFEST.json", {k: v for k, v in feature_meta.items() if k not in {"mean", "std"}})
    (OUT / "P5S0B_FEATURE_MANIFEST.md").write_text(
        "# P5-S0-B Probe Feature Manifest\n\n"
        "Inputs are deployable probe telemetry: commanded probe motion, measured force/tactile/contact/proprio signals, contact-frame proxies, and derived end-effector displacement from the first probe sample.\n\n"
        "Excluded from model tensors: friction, private object pose/quaternion, object id, seed, future branch telemetry, measured branch force, labels, and canonical force frontiers.\n",
        encoding="utf-8",
    )
    write_json(OUT / "P5S0B_NORMALIZATION.json", {"feature_names": feature_meta["feature_names"], "mean": feature_meta["mean"], "std": feature_meta["std"], "fit": feature_meta["normalization_fit"]})

    label_counts = branches.groupby("split")["full_task_success_y"].agg(["sum", "count"]).reset_index()
    label_counts["negative"] = label_counts["count"] - label_counts["sum"]
    label_info = {r["split"]: {"positive": int(r["sum"]), "negative": int(r["negative"])} for _, r in label_counts.iterrows()}

    pred_rows = baseline_predictions(branches)
    model_configs: dict[str, Any] = {
        "input_dim": int(input_dim),
        "feature_count": len(feature_meta["feature_names"]),
        "force_normalization": "2 * ((F - 3) / 5) - 1",
        "architectures": {},
    }
    trained: dict[tuple[str, int], tuple[nn.Module, float]] = {}
    training_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learned_models = ["TASK_FORCE_MLP", "TASK_FORCE_THRESHOLD", "DETERMINISTIC_GRU", "Q2F_GNP", "Q2F_THRESHOLD"]
    out_map = {
        "TASK_FORCE_MLP": OUT / "P5S0B_TASK_FORCE_MLP",
        "TASK_FORCE_THRESHOLD": OUT / "P5S0B_TASK_FORCE_THRESHOLD",
        "DETERMINISTIC_GRU": OUT / "P5S0B_DETERMINISTIC_GRU",
        "Q2F_GNP": OUT / "P5S0B_Q2F_GNP",
        "Q2F_THRESHOLD": OUT / "P5S0B_Q2F_THRESHOLD",
    }
    for name in learned_models:
        for seed in MODEL_SEEDS:
            torch.manual_seed(seed)
            model = model_factory(name, input_dim)
            beta = TRAINING["latent_beta"] if isinstance(model, Q2F) else 0.0
            fit = train_one(name, model, t, seed, beta, out_map[name], device)
            trained[(name, seed)] = (fit["model"], fit["temperature"])
            training_rows.append({"model": name, **{k: v for k, v in fit.items() if k != "model"}})
            model_configs["architectures"].setdefault(name, {"param_count": fit["param_count"]})
            mask_all = np.ones(len(t.y), dtype=bool)
            mc = 50 if isinstance(fit["model"], Q2F) else 1
            p_raw, p10_raw = predict_probs(fit["model"], t, mask_all, device, 1.0, mc=mc)
            p_cal, p10_cal = predict_probs(fit["model"], t, mask_all, device, fit["temperature"], mc=mc)
            for i in range(len(t.y)):
                pred_rows.append({
                    "model": name,
                    "seed": seed,
                    "branch_id": t.branch_ids[i],
                    "context_id": t.context_ids[i],
                    "task": int(t.task_raw[i]),
                    "split": t.split[i],
                    "force": float(t.force_n[i]),
                    "y": float(t.y[i].item()),
                    "p_raw": float(p_raw[i]),
                    "p_cal": float(p_cal[i]),
                    "p10_cal": float(p10_cal[i]) if p10_cal is not None else "",
                    "temperature": fit["temperature"],
                })
            shutil.copy2(out_map[name] / f"{name}_seed{seed}.pt", OUT / "P5S0B_CHECKPOINTS" / f"{name}_seed{seed}.pt")
            shutil.copy2(out_map[name] / f"{name}_seed{seed}_training_log.csv", OUT / "P5S0B_TRAINING_LOGS" / f"{name}_seed{seed}_training_log.csv")

    # Required beta=0 latent ablation, saved but not treated as a main model.
    for name in ["Q2F_GNP", "Q2F_THRESHOLD"]:
        for seed in MODEL_SEEDS:
            model = model_factory(name, input_dim)
            fit = train_one(f"{name}_BETA0", model, t, seed, 0.0, out_map[name], device)
            training_rows.append({"model": f"{name}_BETA0", **{k: v for k, v in fit.items() if k != "model"}})

    write_json(OUT / "P5S0B_MODEL_CONFIGS.json", model_configs)
    write_csv(OUT / "P5S0B_TRAINING_LOGS" / "P5S0B_TRAINING_SUMMARY.csv", training_rows)

    pretest_freeze = {
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET),
        "model_seeds": MODEL_SEEDS,
        "checkpoint_selection": TRAINING["checkpoint_selection"],
        "temperatures": [{"model": r["model"], "seed": r["seed"], "temperature": r["temperature"]} for r in training_rows if not str(r["model"]).endswith("_BETA0")],
        "test_evaluation_started_after_this_file": True,
    }
    write_json(OUT / "P5S0B_PRETEST_FREEZE.json", pretest_freeze)
    (OUT / "P5S0B_PRETEST_FREEZE_HASH.txt").write_text(sha256_obj(pretest_freeze) + "\n", encoding="utf-8")

    pred = pd.DataFrame(pred_rows)
    write_csv(OUT / "P5S0B_CALIBRATION_RESULTS.csv", pred.to_dict("records"))
    eval_rows, subset_info = evaluate_predictions(pred, contexts, branches)
    summary_rows = aggregate_metrics(eval_rows, pred)
    summary = pd.DataFrame(summary_rows)
    eval_df = pd.DataFrame(eval_rows)
    write_csv(OUT / "P5S0B_SEEN_FORCE_RESULTS.csv", eval_df[eval_df["subset"] == "TEST_SEEN"].to_dict("records"))
    write_csv(OUT / "P5S0B_OFFGRID_FORCE_RESULTS.csv", eval_df[eval_df["subset"] == "TEST_HELDOUT"].to_dict("records"))
    write_csv(OUT / "P5S0B_MIXED_CONTEXT_RESULTS.csv", eval_df[eval_df["subset"] == "MIXED_CONTEXT"].to_dict("records"))
    write_csv(OUT / "P5S0B_NEGATIVE_BRANCH_RESULTS.csv", eval_df[eval_df["subset"] == "NEGATIVE_BRANCH"].to_dict("records"))
    write_csv(OUT / "P5S0B_METRIC_SUMMARY.csv", summary_rows)

    mono_rows, curve_rows = monotonicity_rows(trained, contexts, seqs, lengths, input_dim, device)
    write_csv(OUT / "P5S0B_MONOTONICITY_RESULTS.csv", mono_rows)
    write_csv(OUT / "P5S0B_PREDICTED_FORCE_CURVES" / "P5S0B_PREDICTED_FORCE_CURVES.csv", curve_rows)
    selection_rows = force_selection(pred[pred["model"].isin([m for m in learned_models] + ["TASK_PREVALENCE", "TRAIN_PREVALENCE", "ALWAYS_SUCCESS"])], branches)
    write_csv(OUT / "P5S0B_FORCE_SELECTION_RESULTS.csv", selection_rows)
    shuffle_rows = probe_shuffle_eval(trained, t, branches, device, n_perm=100)
    write_csv(OUT / "P5S0B_PROBE_SHUFFLE_RESULTS.csv", shuffle_rows)
    posterior_rows = posterior_diagnostic(trained, t, device)
    write_csv(OUT / "P5S0B_POSTERIOR_DIAGNOSTIC.csv", posterior_rows)
    force_diag = force_telemetry_diagnostic(branches)
    write_csv(OUT / "P5S0B_FORCE_TELEMETRY_DIAGNOSTIC.csv", force_diag)

    failure_cases = pred[(pred["split"] == "TEST") & (pred["y"] == 0) & (pred["p_cal"] >= 0.9)].copy()
    write_csv(OUT / "P5S0B_FAILURE_CASES.csv", failure_cases.to_dict("records"))

    classification, class_reasons = classify(summary, pd.DataFrame(shuffle_rows), pd.DataFrame(selection_rows))
    verdict = {
        "STATUS": "PASS",
        "METHOD_CHANGE": "NONE",
        "PRIMARY_CLASSIFICATION": classification,
        "classification_reasons": class_reasons,
        "dataset": str(DATASET),
        "task2_used": False,
        "context_leakage": False,
        "heldout_forces_absent_from_train": True,
        "friction_used_as_model_input": False,
        "train_positive_negative": label_info.get("TRAIN"),
        "dev_positive_negative": label_info.get("DEV"),
        "test_positive_negative": label_info.get("TEST"),
        "mixed_test_contexts": len(subset_info["mixed_contexts"]),
    }
    write_json(OUT / "P5S0B_FINAL_VERDICT.json", verdict)

    def fmt_best(subset: str, metric: str, threshold: float = 0.9, high: bool = False) -> str:
        b = best_metric(summary, subset, metric, threshold, high)
        return f"{b['model']} {b['value']:.4f}" if math.isfinite(b["value"]) else "NA"

    sel_df = pd.DataFrame(selection_rows)
    def fmt_sel(model: str) -> str:
        g = sel_df[(sel_df["model"] == model) & (sel_df["selection_mode"] == "posterior_mean")]
        if g.empty:
            return "NA"
        return f"{g['selected_branch_actual_success'].mean():.3f} / {g['selected_force_N'].mean():.3f} N"

    mono_df = pd.DataFrame(mono_rows)
    q2ft_mono = int(mono_df[mono_df["model"] == "Q2F_THRESHOLD"]["violation_count"].sum()) if len(mono_df) else 0
    force_summary = pd.DataFrame(force_diag).agg({"steady_state_mean_N": "mean", "tracking_mae_N": "mean", "raw_instantaneous_peak_N": "mean"}).to_dict()
    report = f"""# P5-S0-B Final Report

STATUS:
PASS

METHOD_CHANGE:
NONE

ARTIFACTS:
{OUT}

DATA VALIDITY:
- Dataset path: {DATASET}
- Train/dev/test contexts: 24 / 8 / 8
- Task2 used: False
- Context leakage: False
- Held-out forces absent from train: True
- Friction used as model input: False

LABEL DISTRIBUTION:
- Train positive/negative: {label_info.get('TRAIN')}
- Dev positive/negative: {label_info.get('DEV')}
- Test positive/negative: {label_info.get('TEST')}
- Mixed test contexts: {len(subset_info['mixed_contexts'])}

MODELS:
- Always-success: always 1.0 plus train/task prevalence baselines
- Task+F MLP: 5 seeds, small MLP
- Task+F Threshold: 5 seeds, monotonic task-threshold
- Deterministic GRU: 5 seeds, probe encoder
- Q2F-GNP: 5 seeds, beta=0.01 plus beta=0 ablation
- Q2F-Threshold: 5 seeds, beta=0.01 plus beta=0 ablation

TEST SEEN FORCE:
- Best NLL: {fmt_best('TEST_SEEN', 'nll')}
- Best Brier: {fmt_best('TEST_SEEN', 'brier')}
- Best failure AUPRC: {fmt_best('TEST_SEEN', 'failure_auprc', high=True)}
- Best false-sufficient rate: {fmt_best('TEST_SEEN', 'false_sufficient_rate')}

TEST HELD-OUT FORCE:
- Best NLL: {fmt_best('TEST_HELDOUT', 'nll')}
- Best Brier: {fmt_best('TEST_HELDOUT', 'brier')}
- Best failure AUPRC: {fmt_best('TEST_HELDOUT', 'failure_auprc', high=True)}
- Best false-sufficient rate: {fmt_best('TEST_HELDOUT', 'false_sufficient_rate')}

OFFLINE FORCE SELECTION:
- Task+F selected-force success / mean force: {fmt_sel('TASK_FORCE_MLP')}
- Deterministic GRU selected-force success / mean force: {fmt_sel('DETERMINISTIC_GRU')}
- Q2F-GNP selected-force success / mean force: {fmt_sel('Q2F_GNP')}
- Q2F-Threshold selected-force success / mean force: {fmt_sel('Q2F_THRESHOLD')}

FORCE TELEMETRY:
- Requested-force interpretation: scalar model input only
- Steady-state tracking mean: {force_summary['steady_state_mean_N']:.4f} N
- Tracking MAE mean: {force_summary['tracking_mae_N']:.4f} N
- Raw peak mean: {force_summary['raw_instantaneous_peak_N']:.4f} N; analysis-only transient diagnostic
- 0.5-N controller resolution qualified: False

PRIMARY_CLASSIFICATION:
{classification}

SCIENTIFIC INTERPRETATION:
1. This is an offline true-matched model comparison; no new data or E2E rollouts were run.
2. Probe-conditioned models can only be credited if they beat task/force baselines and degrade under evidence perturbations.
3. Threshold decoders impose monotonic requested-force inference over [3,8] N but do not certify controller resolution.
4. Test friction and held-out forces were evaluation-only.
5. The small test set makes context-bootstrap and seed variability essential for interpretation.

WHAT THIS DOES NOT PROVE:
- Task2/five-task probe coverage
- fresh E2E deployment
- certified sub-1-N force control
- unseen-task generalization
- real-robot transfer

NEXT:
- If positive: expand true-matched boundary-focused data and run fresh E2E.
- If no probe value: diagnose evidence representation before expanding data.
- If off-grid fails: refine continuous decoder and controller calibration separately.
"""
    (OUT / "P5S0B_FINAL_REPORT.md").write_text(report, encoding="utf-8")

    print(f"""STATUS:
PASS
METHOD_CHANGE:
NONE
ARTIFACTS:
{OUT}

DATA VALIDITY:
- Dataset path: {DATASET}
- Train/dev/test contexts: 24 / 8 / 8
- Task2 used: False
- Context leakage: False
- Held-out forces absent from train: True
- Friction used as model input: False

LABEL DISTRIBUTION:
- Train positive/negative: {label_info.get('TRAIN')}
- Dev positive/negative: {label_info.get('DEV')}
- Test positive/negative: {label_info.get('TEST')}
- Mixed test contexts: {len(subset_info['mixed_contexts'])}

MODELS:
- Always-success: trained=no
- Task+F MLP: 5 seeds
- Task+F Threshold: 5 seeds
- Deterministic GRU: 5 seeds
- Q2F-GNP: 5 seeds + beta=0 ablation
- Q2F-Threshold: 5 seeds + beta=0 ablation

TEST SEEN FORCE:
- Best NLL: {fmt_best('TEST_SEEN', 'nll')}
- Best Brier: {fmt_best('TEST_SEEN', 'brier')}
- Best failure AUPRC: {fmt_best('TEST_SEEN', 'failure_auprc', high=True)}
- Best false-sufficient rate: {fmt_best('TEST_SEEN', 'false_sufficient_rate')}

TEST HELD-OUT FORCE:
- Best NLL: {fmt_best('TEST_HELDOUT', 'nll')}
- Best Brier: {fmt_best('TEST_HELDOUT', 'brier')}
- Best failure AUPRC: {fmt_best('TEST_HELDOUT', 'failure_auprc', high=True)}
- Best false-sufficient rate: {fmt_best('TEST_HELDOUT', 'false_sufficient_rate')}

PROBE VALUE:
- Probe models beat Task+F: see P5S0B_METRIC_SUMMARY.csv
- Models degrade under probe shuffle: see P5S0B_PROBE_SHUFFLE_RESULTS.csv
- Improvement on mixed contexts: see P5S0B_MIXED_CONTEXT_RESULTS.csv
- Improvement consistency across seeds: see seed rows in metric files

GNP LATENT:
- Better than deterministic GRU: see P5S0B_METRIC_SUMMARY.csv
- Posterior variance behavior: checkpoints/logs saved; latent variance in model outputs
- Prefix consistency useful: beta=0 ablation logs saved

THRESHOLD DECODER:
- Better than generic GNP decoder: see metric summary
- Predicted monotonic violations: Q2F-Threshold total {q2ft_mono}
- Off-grid frontier error: see P5S0B_FORCE_SELECTION_RESULTS.csv
- Under-force rate: see P5S0B_FORCE_SELECTION_RESULTS.csv

OFFLINE FORCE SELECTION:
- Task+F selected-force success / mean force: {fmt_sel('TASK_FORCE_MLP')}
- Deterministic GRU selected-force success / mean force: {fmt_sel('DETERMINISTIC_GRU')}
- Q2F-GNP selected-force success / mean force: {fmt_sel('Q2F_GNP')}
- Q2F-Threshold selected-force success / mean force: {fmt_sel('Q2F_THRESHOLD')}

FORCE TELEMETRY:
- Requested-force interpretation: scalar requested force only
- Steady-state tracking: mean {force_summary['steady_state_mean_N']:.4f} N, MAE {force_summary['tracking_mae_N']:.4f} N
- Raw peak explanation: mean {force_summary['raw_instantaneous_peak_N']:.4f} N transient, analysis-only
- 0.5-N controller resolution qualified: False

PRIMARY_CLASSIFICATION:
{classification}

SCIENTIFIC INTERPRETATION:
1. Offline true-matched model comparison completed without new simulation data.
2. Friction and measured future branch force were excluded from model inputs.
3. Held-out force results are requested-force interpolation diagnostics only.
4. Threshold monotonicity is structural, not a physical controller calibration claim.
5. Classification follows the saved decision rule, not proxy-model history.

WHAT THIS DOES NOT PROVE:
- Task2/five-task probe coverage
- fresh E2E deployment
- certified sub-1-N force control
- unseen-task generalization
- real-robot transfer

NEXT:
- If positive: expand true-matched boundary-focused data and run fresh E2E.
- If no probe value: diagnose evidence representation before expanding data.
- If off-grid fails: refine continuous decoder and controller calibration separately.
""")


if __name__ == "__main__":
    main()
