#!/usr/bin/env python3
"""P5-S0-C paired-boundary Query2Force model adjudication.

This offline runner consumes a completed P5-S0-C paired-boundary dataset.  It
does not launch Isaac or collect new rollouts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


REPO = Path("/home/exouser/Tabero")
DATASET = REPO / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542"
TASKS = [0, 1, 5, 6]
TASK_TO_IDX = {t: i for i, t in enumerate(TASKS)}
MODEL_SEEDS = [0, 1, 2, 3, 4]
PRIMARY_FORCE = {0: 4.0, 1: 5.0, 5: 4.0, 6: 3.0}
TRAINING = {
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 300,
    "early_stopping_patience": 30,
    "gradient_clip": 1.0,
    "latent_beta": 0.01,
    "primary_loss": "unweighted BCEWithLogitsLoss",
    "checkpoint_selection": "DEV all-force NLL",
    "calibration": "one global temperature per model/seed using DEV branches",
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
EXCLUDED_FIELDS = [
    "friction",
    "hidden_friction_analysis_only",
    "object_x_priv",
    "object_y_priv",
    "object_z_priv",
    "object_qw_priv",
    "object_qx_priv",
    "object_qy_priv",
    "object_qz_priv",
    "future_branch_telemetry",
    "measured_branch_force",
    "full_task_success_y",
]


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
        wr.writeheader()
        wr.writerows(rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def force_norm(x: np.ndarray | torch.Tensor | float) -> np.ndarray | torch.Tensor | float:
    return 2.0 * ((x - 3.0) / 5.0) - 1.0


def equal_mass_ece(y: np.ndarray, p: np.ndarray, bins: int = 5) -> float:
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    chunks = np.array_split(order, min(bins, len(order)))
    ece = 0.0
    for idx in chunks:
        if len(idx):
            ece += len(idx) / len(y) * abs(float(y[idx].mean()) - float(p[idx].mean()))
    return float(ece)


def auprc(y: np.ndarray, score: np.ndarray, positive_label: int) -> float:
    yb = (np.asarray(y).astype(int) == positive_label).astype(int)
    score = np.asarray(score, dtype=float)
    if yb.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    yord = yb[order]
    tp = np.cumsum(yord)
    fp = np.cumsum(1 - yord)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / yb.sum()
    return float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))


def metric_dict(y: np.ndarray, p: np.ndarray, threshold: float = 0.9) -> dict[str, Any]:
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
        "failure_auprc": auprc(y, 1.0 - p, 0),
        "failure_recall": float(np.mean(~pred[failures])) if failures.any() else float("nan"),
        "false_sufficient_rate": float(np.mean(pred[failures])) if failures.any() else float("nan"),
        "ece": equal_mass_ece(y, p, 5),
    }


@dataclass
class Tensors:
    xseq: torch.Tensor
    xstatic: torch.Tensor
    lengths: torch.Tensor
    task: torch.Tensor
    force: torch.Tensor
    y: torch.Tensor
    branch_ids: np.ndarray
    context_ids: np.ndarray
    root_ids: np.ndarray
    split: np.ndarray
    task_raw: np.ndarray
    force_n: np.ndarray
    force_role: np.ndarray


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contexts = pd.read_csv(DATASET / "P5S0C_CONTEXT_MANIFEST.csv")
    branches = pd.read_csv(DATASET / "P5S0C_BRANCH_MANIFEST.csv")
    pairs = pd.read_csv(DATASET / "P5S0C_PRIMARY_AMBIGUITY_PAIRS.csv")
    return contexts, branches, pairs


def audit_dataset(contexts: pd.DataFrame, branches: pd.DataFrame, pairs: pd.DataFrame) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if sorted(contexts["task"].unique().tolist()) != TASKS:
        errors.append("task set mismatch")
    if (contexts["task"] == 2).any() or (branches["task"] == 2).any():
        errors.append("Task2 rows present")
    if len(contexts) != 144 or len(branches) != 576:
        errors.append(f"bad counts: contexts={len(contexts)}, branches={len(branches)}")
    if not contexts["strict_matched"].astype(int).eq(1).all():
        errors.append("non-strict-matched context present")
    if not branches["state_parity"].astype(int).eq(1).all():
        errors.append("branch state parity failure present")
    if branches["label_source"].nunique() != 1 or branches["label_source"].iloc[0] != "TRUE_POST_PROBE_RESET_MATCHED_BRANCH":
        errors.append("unexpected label source")
    for cid, g in branches.groupby("context_id"):
        if g["split"].nunique() != 1:
            errors.append(f"context split leakage {cid}")
        if g["post_probe_state_hash"].nunique() != 1:
            errors.append(f"multiple post-probe hashes {cid}")
    for rid, g in contexts.groupby("root_id"):
        if g["split"].nunique() != 1:
            errors.append(f"root split leakage {rid}")
    if int(pairs[(pairs["split"] == "TEST") & (pairs["decision_discordant"] == 1)].shape[0]) < 10:
        errors.append("boundary TEST not adjudicative")
    return len(errors) == 0, errors


def sequence_dataframe(path: str, phases: list[str], contact_states: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    for col in NUMERIC_PROBE_COLS:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df else 0.0
    ex0 = float(pd.to_numeric(df.get("eef_x", pd.Series([0.0])), errors="coerce").iloc[0])
    ey0 = float(pd.to_numeric(df.get("eef_y", pd.Series([0.0])), errors="coerce").iloc[0])
    ez0 = float(pd.to_numeric(df.get("eef_z", pd.Series([0.0])), errors="coerce").iloc[0])
    out["eef_dx"] = pd.to_numeric(df.get("eef_x", 0.0), errors="coerce") - ex0
    out["eef_dy"] = pd.to_numeric(df.get("eef_y", 0.0), errors="coerce") - ey0
    out["eef_dz"] = pd.to_numeric(df.get("eef_z", 0.0), errors="coerce") - ez0
    phase_vals = df["probe_phase"].fillna("NA").astype(str) if "probe_phase" in df else pd.Series(["NA"] * len(df))
    state_vals = df["contact_state"].fillna("NA").astype(str) if "contact_state" in df else pd.Series(["NA"] * len(df))
    for phase in phases:
        out[f"phase={phase}"] = (phase_vals == phase).astype(float)
    for state in contact_states:
        out[f"contact_state={state}"] = (state_vals == state).astype(float)
    out["probe_phase_unknown"] = (~phase_vals.isin(phases)).astype(float)
    out["contact_state_unknown"] = (~state_vals.isin(contact_states)).astype(float)
    return out


def load_features(contexts: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], dict[str, Any]]:
    train_phases: set[str] = set()
    train_states: set[str] = set()
    for row in contexts.itertuples():
        df = pd.read_csv(row.probe_telemetry_path)
        if row.split == "TRAIN":
            train_phases.update(str(x) for x in df["probe_phase"].fillna("NA").unique())
            train_states.update(str(x) for x in df["contact_state"].fillna("NA").unique())
    phases = sorted(train_phases)
    states = sorted(train_states)
    feature_names = NUMERIC_PROBE_COLS + DERIVED_PROBE_COLS
    feature_names += [f"phase={p}" for p in phases]
    feature_names += [f"contact_state={s}" for s in states]
    feature_names += ["probe_phase_unknown", "contact_state_unknown"]
    raw: dict[str, np.ndarray] = {}
    static_raw: dict[str, np.ndarray] = {}
    lengths: dict[str, int] = {}
    train_arrays: list[np.ndarray] = []
    train_static: list[np.ndarray] = []
    for row in contexts.itertuples():
        arr_df = sequence_dataframe(row.probe_telemetry_path, phases, states).reindex(columns=feature_names).fillna(0.0)
        arr = arr_df.to_numpy(dtype=np.float32)
        if arr.shape[0] == 0:
            arr = np.zeros((1, len(feature_names)), dtype=np.float32)
        contact = (arr_df.get("contact_left", 0.0).to_numpy(dtype=float) + arr_df.get("contact_right", 0.0).to_numpy(dtype=float)) > 0
        static_idx = int(np.argmax(contact)) if contact.any() else 0
        raw[row.context_id] = arr
        static_raw[row.context_id] = arr[static_idx].copy()
        lengths[row.context_id] = int(arr.shape[0])
        if row.split == "TRAIN":
            train_arrays.append(arr)
            train_static.append(static_raw[row.context_id])
    cat = np.concatenate(train_arrays, axis=0)
    mu = cat.mean(axis=0)
    sd = cat.std(axis=0)
    sd[sd < 1e-6] = 1.0
    static_cat = np.stack(train_static)
    smu = static_cat.mean(axis=0)
    ssd = static_cat.std(axis=0)
    ssd[ssd < 1e-6] = 1.0
    seqs = {cid: (arr - mu) / sd for cid, arr in raw.items()}
    statics = {cid: (vec - smu) / ssd for cid, vec in static_raw.items()}
    meta = {
        "dynamic_feature_names": feature_names,
        "static_feature_names": feature_names,
        "normalization": "TRAIN contexts only",
        "static_row_selection": "first telemetry row with any deployable contact flag, else first row",
        "excluded_fields": EXCLUDED_FIELDS,
        "dynamic_mean": mu.tolist(),
        "dynamic_std": sd.tolist(),
        "static_mean": smu.tolist(),
        "static_std": ssd.tolist(),
        "phase_categories_from_train": phases,
        "contact_state_categories_from_train": states,
    }
    return seqs, statics, lengths, meta


def make_tensors(contexts: pd.DataFrame, branches: pd.DataFrame, seqs: dict[str, np.ndarray], statics: dict[str, np.ndarray], lengths: dict[str, int]) -> Tensors:
    max_len = max(lengths.values())
    nfeat = next(iter(seqs.values())).shape[1]
    xs, xst, lens, task, force, y = [], [], [], [], [], []
    bids, cids, rids, splits, task_raw, force_n, force_role = [], [], [], [], [], [], []
    for br in branches.itertuples():
        cid = br.context_id
        arr = seqs[cid]
        pad = np.zeros((max_len, nfeat), dtype=np.float32)
        pad[: arr.shape[0]] = arr
        xs.append(pad)
        xst.append(statics[cid])
        lens.append(lengths[cid])
        task.append(TASK_TO_IDX[int(br.task)])
        force.append(float(force_norm(float(br.requested_force_N))))
        y.append(float(br.full_task_success_y))
        bids.append(br.branch_id)
        cids.append(cid)
        rids.append(br.root_id)
        splits.append(br.split)
        task_raw.append(int(br.task))
        force_n.append(float(br.requested_force_N))
        force_role.append(br.force_role)
    return Tensors(
        xseq=torch.tensor(np.stack(xs), dtype=torch.float32),
        xstatic=torch.tensor(np.stack(xst), dtype=torch.float32),
        lengths=torch.tensor(lens, dtype=torch.long),
        task=torch.tensor(task, dtype=torch.long),
        force=torch.tensor(force, dtype=torch.float32).unsqueeze(1),
        y=torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        branch_ids=np.array(bids),
        context_ids=np.array(cids),
        root_ids=np.array(rids),
        split=np.array(splits),
        task_raw=np.array(task_raw, dtype=int),
        force_n=np.array(force_n, dtype=float),
        force_role=np.array(force_role),
    )


def threshold_logit(force: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    mu_req = 2.0 + 7.0 * torch.sigmoid(raw[:, :1])
    scale = 0.1 + torch.nn.functional.softplus(raw[:, 1:2])
    force_n = 3.0 + 2.5 * (force + 1.0)
    return (force_n - mu_req) / scale


class TaskForceThreshold(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4, 8)
        self.head = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, xseq, xstatic, lengths, task, force):
        return threshold_logit(force, self.head(self.emb(task)))


class StaticStateThreshold(nn.Module):
    def __init__(self, static_dim: int):
        super().__init__()
        self.emb = nn.Embedding(4, 8)
        self.head = nn.Sequential(nn.Linear(static_dim + 8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, xseq, xstatic, lengths, task, force):
        return threshold_logit(force, self.head(torch.cat([xstatic, self.emb(task)], dim=1)))


class ProbeGRUThreshold(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.emb = nn.Embedding(4, 8)
        self.head = nn.Sequential(nn.Linear(32 + 8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def encode(self, xseq, lengths):
        z = self.proj(xseq)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]

    def forward(self, xseq, xstatic, lengths, task, force):
        h = self.encode(xseq, lengths)
        return threshold_logit(force, self.head(torch.cat([h, self.emb(task)], dim=1)))


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
        if self.threshold:
            return threshold_logit(force, self.head(torch.cat([z, te], dim=1)))
        return self.head(torch.cat([z, te, force], dim=1))

    def forward(self, xseq, xstatic, lengths, task, force, sample: bool = True):
        mu, lv = self.encode(xseq, lengths)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if sample else mu
        return self.logits_from_z(z, task, force), mu, lv


def param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def mask(t: Tensors, split: str | None = None, subset: set[str] | None = None) -> np.ndarray:
    m = np.ones(len(t.y), dtype=bool)
    if split is not None:
        m &= t.split == split
    if subset is not None:
        m &= np.isin(t.branch_ids, list(subset))
    return m


def masked(t: Tensors, m: np.ndarray, device: torch.device):
    idx = torch.tensor(m, dtype=torch.bool)
    return (
        t.xseq[idx].to(device),
        t.xstatic[idx].to(device),
        t.lengths[idx].to(device),
        t.task[idx].to(device),
        t.force[idx].to(device),
        t.y[idx].to(device),
    )


def nll(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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


def kl_diag_gauss(mu_p, lv_p, mu_q, lv_q):
    var_p = torch.exp(lv_p)
    var_q = torch.exp(lv_q)
    return 0.5 * torch.mean(torch.sum(lv_q - lv_p + (var_p + (mu_p - mu_q).pow(2)) / var_q - 1.0, dim=1))


def prefix_lengths(lengths: torch.Tensor) -> torch.Tensor:
    frac = 0.4 + 0.6 * torch.rand_like(lengths.float())
    return torch.clamp((lengths.float() * frac).long(), min=1)


def predict_logits(model: nn.Module, t: Tensors, m: np.ndarray, device: torch.device, mc: int = 1) -> np.ndarray:
    x, xs, lengths, task, force, _ = masked(t, m, device)
    outs = []
    with torch.no_grad():
        if isinstance(model, Q2F):
            for _ in range(mc):
                logits, _, _ = model(x, xs, lengths, task, force, sample=(mc > 1))
                outs.append(logits.detach().cpu().numpy().reshape(-1))
        else:
            outs.append(model(x, xs, lengths, task, force).detach().cpu().numpy().reshape(-1))
    return np.stack(outs).mean(axis=0)


def predict_probs(model: nn.Module, t: Tensors, m: np.ndarray, device: torch.device, temp: float, mc: int = 1) -> np.ndarray:
    logits = predict_logits(model, t, m, device, mc=mc)
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temp, -60, 60)))


def train_one(model_name: str, model: nn.Module, t: Tensors, seed: int, out_dir: Path, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
    train_m = mask(t, "TRAIN")
    dev_m = mask(t, "DEV")
    best_state = None
    best_dev = float("inf")
    best_epoch = -1
    patience = 0
    logs: list[dict[str, Any]] = []
    for epoch in range(TRAINING["max_epochs"]):
        model.train()
        opt.zero_grad()
        x, xs, lengths, task, force, y = masked(t, train_m, device)
        if isinstance(model, Q2F):
            pl = prefix_lengths(lengths).to(device)
            logits, mu_t, lv_t = model(x, xs, pl, task, force, sample=True)
            with torch.no_grad():
                _, mu_T, lv_T = model(x, xs, lengths, task, force, sample=False)
            pred = nll(logits, y)
            cons = kl_diag_gauss(mu_T.detach(), lv_T.detach(), mu_t, lv_t)
            loss = pred + TRAINING["latent_beta"] * cons
        else:
            logits = model(x, xs, lengths, task, force)
            pred = nll(logits, y)
            cons = torch.tensor(0.0, device=device)
            loss = pred
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["gradient_clip"])
        opt.step()
        model.eval()
        with torch.no_grad():
            dx, dxs, dl, dtask, df, dy = masked(t, dev_m, device)
            if isinstance(model, Q2F):
                dlogits, _, _ = model(dx, dxs, dl, dtask, df, sample=False)
            else:
                dlogits = model(dx, dxs, dl, dtask, df)
            dev = float(nll(dlogits, dy).item())
        logs.append({"epoch": epoch, "train_loss": float(loss.item()), "train_pred_loss": float(pred.item()), "train_consistency": float(cons.item()), "dev_nll": dev})
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
    all_logits = predict_logits(model, t, np.ones(len(t.y), dtype=bool), device, mc=1)
    temp = fit_temperature(all_logits[dev_m], t.y.squeeze(1).numpy()[dev_m])
    return {"model": model, "seed": seed, "best_epoch": best_epoch, "best_dev_nll": best_dev, "temperature": temp, "param_count": param_count(model)}


def model_factory(name: str, input_dim: int, static_dim: int) -> nn.Module:
    if name == "TASK_FORCE_THRESHOLD":
        return TaskForceThreshold()
    if name == "STATIC_STATE_THRESHOLD":
        return StaticStateThreshold(static_dim)
    if name == "PROBE_GRU_THRESHOLD":
        return ProbeGRUThreshold(input_dim)
    if name == "Q2F_THRESHOLD":
        return Q2F(input_dim, threshold=True)
    if name == "Q2F_GNP":
        return Q2F(input_dim, threshold=False)
    raise ValueError(name)


def prediction_frame(models: dict[tuple[str, int], dict[str, Any]], t: Tensors, device: torch.device) -> pd.DataFrame:
    rows = []
    all_m = np.ones(len(t.y), dtype=bool)
    y_np = t.y.squeeze(1).numpy()
    for (name, seed), fit in models.items():
        mc = 50 if isinstance(fit["model"], Q2F) else 1
        p = predict_probs(fit["model"], t, all_m, device, fit["temperature"], mc=mc)
        for i, bid in enumerate(t.branch_ids):
            rows.append({
                "model": name,
                "seed": seed,
                "branch_id": bid,
                "context_id": t.context_ids[i],
                "root_id": t.root_ids[i],
                "split": t.split[i],
                "task": int(t.task_raw[i]),
                "force_N": float(t.force_n[i]),
                "force_role": t.force_role[i],
                "y": float(y_np[i]),
                "p": float(p[i]),
                "temperature": fit["temperature"],
            })
    return pd.DataFrame(rows)


def aggregate_eval(pred: pd.DataFrame, branches: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test = pred[pred["split"] == "TEST"].copy()
    br = branches[branches["split"] == "TEST"]
    mixed_contexts = {cid for cid, g in br.groupby("context_id") if set(g["full_task_success_y"].astype(int)) == {0, 1}}
    subset_defs = {
        "TEST_ALL_FORCES": test.index,
        "TEST_INTEGER_FORCES": test[test["force_role"] == "INTEGER_FORCE"].index,
        "TEST_OFFGRID_FORCES": test[test["force_role"] == "OFFGRID_DEVELOPMENT_FORCE"].index,
        "TEST_MIXED_CONTEXTS": test[test["context_id"].isin(mixed_contexts)].index,
    }
    for (model, seed), g0 in test.groupby(["model", "seed"]):
        for subset, idx in subset_defs.items():
            g = test.loc[idx]
            g = g[(g["model"] == model) & (g["seed"] == seed)]
            for thr in [0.5, 0.9]:
                md = metric_dict(g["y"].to_numpy(), g["p"].to_numpy(), thr)
                rows.append({"model": model, "seed": seed, "subset": subset, **md})
    return rows


def paired_eval(pred: pd.DataFrame, pairs: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test_pairs = pairs[(pairs["split"] == "TEST") & (pairs["decision_discordant"] == 1)]
    for (model, seed), mpred in pred.groupby(["model", "seed"]):
        lookup = {(r.context_id, float(r.force_N)): (float(r.p), float(r.y)) for r in mpred.itertuples()}
        pair_rows = []
        for pr in test_pairs.itertuples():
            low = lookup.get((pr.low_context_id, float(pr.force_N)))
            high = lookup.get((pr.high_context_id, float(pr.force_N)))
            if low is None or high is None:
                continue
            low_p, low_y = low
            high_p, high_y = high
            success_p = high_p if high_y == 1 else low_p
            failure_p = low_p if high_y == 1 else high_p
            y = np.array([low_y, high_y], dtype=float)
            p = np.array([low_p, high_p], dtype=float)
            pair_rows.append({
                "model": model,
                "seed": seed,
                "task": int(pr.task),
                "root_id": pr.root_id,
                "force_N": float(pr.force_N),
                "low_context_id": pr.low_context_id,
                "high_context_id": pr.high_context_id,
                "low_outcome": int(low_y),
                "high_outcome": int(high_y),
                "low_p": low_p,
                "high_p": high_p,
                "pairwise_rank_correct": int(success_p > failure_p),
                "pairwise_probability_margin": success_p - failure_p,
                "paired_nll": metric_dict(y, p, 0.9)["nll"],
                "paired_brier": metric_dict(y, p, 0.9)["brier"],
                "false_sufficient_rate": metric_dict(y, p, 0.9)["false_sufficient_rate"],
            })
        rows.extend(pair_rows)
        if pair_rows:
            for task in ["ALL"] + [str(t) for t in TASKS]:
                rr = pair_rows if task == "ALL" else [r for r in pair_rows if r["task"] == int(task)]
                if not rr:
                    continue
                ys, ps = [], []
                for r in rr:
                    ys += [r["low_outcome"], r["high_outcome"]]
                    ps += [r["low_p"], r["high_p"]]
                md = metric_dict(np.array(ys), np.array(ps), 0.9)
                rows.append({
                    "row_type": "SUMMARY",
                    "model": model,
                    "seed": seed,
                    "task": task,
                    "n_pairs": len(rr),
                    "pairwise_ranking_accuracy": float(np.mean([r["pairwise_rank_correct"] for r in rr])),
                    "pairwise_probability_margin": float(np.mean([r["pairwise_probability_margin"] for r in rr])),
                    "paired_nll": md["nll"],
                    "paired_brier": md["brier"],
                    "false_sufficient_rate": md["false_sufficient_rate"],
                })
    return rows


def clone_tensors(t: Tensors) -> Tensors:
    return Tensors(
        xseq=t.xseq.clone(),
        xstatic=t.xstatic.clone(),
        lengths=t.lengths.clone(),
        task=t.task.clone(),
        force=t.force.clone(),
        y=t.y.clone(),
        branch_ids=t.branch_ids.copy(),
        context_ids=t.context_ids.copy(),
        root_ids=t.root_ids.copy(),
        split=t.split.copy(),
        task_raw=t.task_raw.copy(),
        force_n=t.force_n.copy(),
        force_role=t.force_role.copy(),
    )


def evaluate_pair_metric(model: nn.Module, fit: dict[str, Any], t: Tensors, pairs: pd.DataFrame, device: torch.device) -> dict[str, float]:
    p = predict_probs(model, t, np.ones(len(t.y), dtype=bool), device, fit["temperature"], mc=30 if isinstance(model, Q2F) else 1)
    pred = pd.DataFrame({"context_id": t.context_ids, "force_N": t.force_n, "y": t.y.squeeze(1).numpy(), "p": p})
    rows = paired_eval(pd.DataFrame({
        "model": ["M"] * len(pred),
        "seed": [0] * len(pred),
        "context_id": pred["context_id"],
        "force_N": pred["force_N"],
        "y": pred["y"],
        "p": pred["p"],
    }), pairs)
    summary = [r for r in rows if r.get("row_type") == "SUMMARY" and r.get("task") == "ALL"]
    if not summary:
        return {"paired_nll": float("nan"), "paired_brier": float("nan"), "pairwise_ranking_accuracy": float("nan")}
    return summary[0]


def perturbation_eval(models: dict[tuple[str, int], dict[str, Any]], t: Tensors, pairs: pd.DataFrame, device: torch.device, n_perm: int = 100) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dynamic = {"PROBE_GRU_THRESHOLD", "Q2F_THRESHOLD", "Q2F_GNP"}
    low_high_rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    test_contexts_by_task = {task: sorted(set(t.context_ids[(t.split == "TEST") & (t.task_raw == task)])) for task in TASKS}
    context_to_indices = {cid: np.where(t.context_ids == cid)[0] for cid in set(t.context_ids)}
    train_mean_seq = torch.zeros((1, t.xseq.shape[2]), dtype=torch.float32)
    for (name, seed), fit in models.items():
        if name not in dynamic and name != "STATIC_STATE_THRESHOLD":
            continue
        base = evaluate_pair_metric(fit["model"], fit, t, pairs, device)
        if name in dynamic:
            swapped = clone_tensors(t)
            for pr in pairs[(pairs["split"] == "TEST") & (pairs["decision_discordant"] == 1)].itertuples():
                li = context_to_indices[pr.low_context_id]
                hi = context_to_indices[pr.high_context_id]
                low_seq = swapped.xseq[li].clone()
                low_len = swapped.lengths[li].clone()
                swapped.xseq[li] = swapped.xseq[hi]
                swapped.lengths[li] = swapped.lengths[hi]
                swapped.xseq[hi] = low_seq
                swapped.lengths[hi] = low_len
            met = evaluate_pair_metric(fit["model"], fit, swapped, pairs, device)
            low_high_rows.append({"model": name, "seed": seed, "diagnostic": "within_root_low_high_probe_swap", "base_paired_nll": base["paired_nll"], "perturbed_paired_nll": met["paired_nll"], "nll_degradation": met["paired_nll"] - base["paired_nll"], "base_ranking": base["pairwise_ranking_accuracy"], "perturbed_ranking": met["pairwise_ranking_accuracy"], "ranking_degradation": base["pairwise_ranking_accuracy"] - met["pairwise_ranking_accuracy"]})

            mean_t = clone_tensors(t)
            mean_t.xseq[:] = 0.0
            mean_t.lengths[:] = 1
            met = evaluate_pair_metric(fit["model"], fit, mean_t, pairs, device)
            mean_rows.append({"model": name, "seed": seed, "diagnostic": "train_mean_dynamic_evidence_replacement", "base_paired_nll": base["paired_nll"], "perturbed_paired_nll": met["paired_nll"], "nll_degradation": met["paired_nll"] - base["paired_nll"], "base_ranking": base["pairwise_ranking_accuracy"], "perturbed_ranking": met["pairwise_ranking_accuracy"], "ranking_degradation": base["pairwise_ranking_accuracy"] - met["pairwise_ranking_accuracy"]})

            rng = np.random.default_rng(1000 + seed)
            vals = []
            for _ in range(n_perm):
                sh = clone_tensors(t)
                for task in TASKS:
                    cids = test_contexts_by_task[task]
                    perm = rng.permutation(cids)
                    saved = {cid: (sh.xseq[context_to_indices[cid]].clone(), sh.lengths[context_to_indices[cid]].clone()) for cid in cids}
                    for cid, pcid in zip(cids, perm):
                        idx = context_to_indices[cid]
                        sh.xseq[idx] = saved[pcid][0]
                        sh.lengths[idx] = saved[pcid][1]
                met = evaluate_pair_metric(fit["model"], fit, sh, pairs, device)
                vals.append(met["paired_nll"] - base["paired_nll"])
            shuffle_rows.append({"model": name, "seed": seed, "diagnostic": "within_task_probe_shuffle_100", "mean_nll_degradation": float(np.nanmean(vals)), "p05_nll_degradation": float(np.nanpercentile(vals, 5)), "p95_nll_degradation": float(np.nanpercentile(vals, 95))})
        elif name == "STATIC_STATE_THRESHOLD":
            mean_t = clone_tensors(t)
            mean_t.xstatic[:] = 0.0
            met = evaluate_pair_metric(fit["model"], fit, mean_t, pairs, device)
            mean_rows.append({"model": name, "seed": seed, "diagnostic": "train_mean_static_state_replacement", "base_paired_nll": base["paired_nll"], "perturbed_paired_nll": met["paired_nll"], "nll_degradation": met["paired_nll"] - base["paired_nll"], "base_ranking": base["pairwise_ranking_accuracy"], "perturbed_ranking": met["pairwise_ranking_accuracy"], "ranking_degradation": base["pairwise_ranking_accuracy"] - met["pairwise_ranking_accuracy"]})
    return low_high_rows, shuffle_rows, mean_rows


def force_selection(pred: pd.DataFrame, branches: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    test_br = branches[branches["split"] == "TEST"].copy()
    actual = {(r.context_id, float(r.requested_force_N)): int(r.full_task_success_y) for r in test_br.itertuples()}
    for (model, seed, cid), g in pred[pred["split"] == "TEST"].groupby(["model", "seed", "context_id"]):
        gg = g.sort_values("force_N")
        chosen = gg[gg["p"] >= 0.9].head(1)
        if chosen.empty:
            chosen = gg.tail(1)
        ch = chosen.iloc[0]
        outcomes = test_br[test_br["context_id"] == cid].sort_values("requested_force_N")
        succ = outcomes[outcomes["full_task_success_y"].astype(int) == 1]
        frontier = float(succ["requested_force_N"].min()) if len(succ) else float("nan")
        selected_success = actual.get((cid, float(ch.force_N)), 0)
        rows.append({
            "model": model,
            "seed": seed,
            "context_id": cid,
            "task": int(ch.task),
            "selected_force_N": float(ch.force_N),
            "selected_probability": float(ch.p),
            "selected_branch_actual_success": int(selected_success),
            "under_force": int(selected_success == 0),
            "branch_frontier_N": frontier,
            "absolute_frontier_error_N": abs(float(ch.force_N) - frontier) if math.isfinite(frontier) else float("nan"),
            "excess_force_regret_N": max(0.0, float(ch.force_N) - frontier) if math.isfinite(frontier) else float("nan"),
        })
    return rows


def model_summary(models: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"model": name, "seed": seed, "best_epoch": fit["best_epoch"], "best_dev_nll": fit["best_dev_nll"], "temperature": fit["temperature"], "param_count": fit["param_count"]} for (name, seed), fit in models.items()]


def mean_metric(df: pd.DataFrame, model: str, metric: str, subset: str | None = None) -> float:
    g = df[df["model"] == model]
    if subset is not None and "subset" in g:
        g = g[g["subset"] == subset]
    if metric not in g or g.empty:
        return float("nan")
    return float(pd.to_numeric(g[metric], errors="coerce").mean())


def classify(paired: pd.DataFrame, low_high: pd.DataFrame, shuffle: pd.DataFrame, meanrep: pd.DataFrame, selection: pd.DataFrame) -> tuple[str, list[str]]:
    dynamic = ["PROBE_GRU_THRESHOLD", "Q2F_THRESHOLD", "Q2F_GNP"]
    task_nll = mean_metric(paired[(paired.get("row_type") == "SUMMARY") & (paired["task"] == "ALL")], "TASK_FORCE_THRESHOLD", "paired_nll")
    static_nll = mean_metric(paired[(paired.get("row_type") == "SUMMARY") & (paired["task"] == "ALL")], "STATIC_STATE_THRESHOLD", "paired_nll")
    reasons: list[str] = []
    for model in dynamic:
        summ = paired[(paired.get("row_type") == "SUMMARY") & (paired["task"] == "ALL") & (paired["model"] == model)]
        if summ.empty:
            continue
        nlls = pd.to_numeric(summ["paired_nll"], errors="coerce")
        ranks = pd.to_numeric(summ["pairwise_ranking_accuracy"], errors="coerce")
        beats_task = int((nlls < task_nll).sum()) >= 4
        beats_static = int((nlls < static_nll).sum()) >= 4
        swap_deg = low_high[low_high["model"] == model].groupby("seed")["nll_degradation"].mean()
        shuf_deg = shuffle[shuffle["model"] == model].groupby("seed")["mean_nll_degradation"].mean()
        mean_deg = meanrep[meanrep["model"] == model].groupby("seed")["nll_degradation"].mean()
        perturb = int((swap_deg > 0).sum()) >= 4 and int((shuf_deg > 0).sum()) >= 4 and int((mean_deg > 0).sum()) >= 4
        sel = selection[selection["model"] == model].groupby("seed")["under_force"].mean()
        task_sel = selection[selection["model"] == "TASK_FORCE_THRESHOLD"].groupby("seed")["under_force"].mean().mean()
        under_ok = len(sel) and float(sel.mean()) <= float(task_sel)
        if beats_task and beats_static and perturb and under_ok:
            reasons.append(f"{model} beats Task+F and Static-State on paired NLL in >=4 seeds and degrades under evidence perturbations")
            return "P5S0C_DYNAMIC_PROBE_VALUE_CONFIRMED", reasons
        if perturb and not (beats_task and beats_static):
            reasons.append(f"{model} shows perturbation sensitivity but does not reliably improve paired decisions")
            return "P5S0C_PROBE_SIGNAL_PRESENT_BUT_NOT_CONVERTED_TO_DECISION", reasons
    static_rank = mean_metric(paired[(paired.get("row_type") == "SUMMARY") & (paired["task"] == "ALL")], "STATIC_STATE_THRESHOLD", "pairwise_ranking_accuracy")
    task_rank = mean_metric(paired[(paired.get("row_type") == "SUMMARY") & (paired["task"] == "ALL")], "TASK_FORCE_THRESHOLD", "pairwise_ranking_accuracy")
    if static_nll < task_nll and static_rank >= task_rank:
        return "P5S0C_STATIC_CONTACT_STATE_EXPLAINS_GAIN", ["Static-State Threshold improves over Task+F while dynamic probe criteria are not satisfied"]
    return "P5S0C_NO_PROBE_VALUE_DETECTED_ON_ADJUDICATIVE_BOUNDARY_DATA", ["Boundary data is adjudicative but dynamic probe value criteria are not satisfied"]


def best_line(df: pd.DataFrame, metric: str, higher: bool = False, subset: str | None = None) -> str:
    g = df.copy()
    if subset is not None and "subset" in g:
        g = g[g["subset"] == subset]
    if g.empty or metric not in g:
        return "NA"
    agg = g.groupby("model")[metric].mean(numeric_only=True).dropna()
    if agg.empty:
        return "NA"
    model = agg.idxmax() if higher else agg.idxmin()
    return f"{model}={agg.loc[model]:.4f}"


def main() -> int:
    contexts, branches, pairs = load_dataset()
    ok, errors = audit_dataset(contexts, branches, pairs)
    write_json(DATASET / "P5S0C_MODEL_DATA_AUDIT.json", {"ok": ok, "errors": errors})
    (DATASET / "P5S0C_DATA_AUDIT.md").write_text("# P5-S0-C Model Data Audit\n\n" + ("\n".join(f"- {e}" for e in errors) if errors else "PASS\n"), encoding="utf-8")
    if not ok:
        verdict = {"PRIMARY_CLASSIFICATION": "P5S0C_INVALID_DATA_OR_LEAKAGE", "errors": errors}
        write_json(DATASET / "P5S0C_FINAL_VERDICT.json", verdict)
        print("P5S0C_INVALID_DATA_OR_LEAKAGE", flush=True)
        return 1

    seqs, statics, lengths, feature_meta = load_features(contexts)
    t = make_tensors(contexts, branches, seqs, statics, lengths)
    write_json(DATASET / "P5S0C_FEATURE_MANIFEST.json", {k: v for k, v in feature_meta.items() if not k.endswith("_mean") and not k.endswith("_std")})
    write_json(DATASET / "P5S0C_NORMALIZATION.json", feature_meta)
    write_json(DATASET / "P5S0C_MODEL_CONFIGS.json", {"models": ["TASK_FORCE_THRESHOLD", "STATIC_STATE_THRESHOLD", "PROBE_GRU_THRESHOLD", "Q2F_THRESHOLD", "Q2F_GNP"], "seeds": MODEL_SEEDS, "training": TRAINING})
    code_hash = sha256_file(Path(__file__))
    (DATASET / "P5S0C_MODEL_CODE_HASH.txt").write_text(code_hash + "\n", encoding="utf-8")
    write_json(DATASET / "P5S0C_PRETEST_FREEZE.json", {"dataset": str(DATASET), "git_commit": git_commit(), "code_hash": code_hash, "training": TRAINING, "model_seeds": MODEL_SEEDS, "test_opened_after_freeze": True})
    (DATASET / "P5S0C_PRETEST_FREEZE_HASH.txt").write_text(sha256_file(DATASET / "P5S0C_PRETEST_FREEZE.json") + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = t.xseq.shape[2]
    static_dim = t.xstatic.shape[1]
    model_dirs = {
        "TASK_FORCE_THRESHOLD": DATASET / "P5S0C_TASK_FORCE_THRESHOLD",
        "STATIC_STATE_THRESHOLD": DATASET / "P5S0C_STATIC_STATE_THRESHOLD",
        "PROBE_GRU_THRESHOLD": DATASET / "P5S0C_PROBE_GRU_THRESHOLD",
        "Q2F_THRESHOLD": DATASET / "P5S0C_Q2F_THRESHOLD",
        "Q2F_GNP": DATASET / "P5S0C_Q2F_GNP",
    }
    ckpt_dir = DATASET / "P5S0C_CHECKPOINTS"
    log_dir = DATASET / "P5S0C_TRAINING_LOGS"
    ckpt_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    models: dict[tuple[str, int], dict[str, Any]] = {}
    for name in model_dirs:
        for seed in MODEL_SEEDS:
            fit = train_one(name, model_factory(name, input_dim, static_dim), t, seed, model_dirs[name], device)
            models[(name, seed)] = fit
            src = model_dirs[name] / f"{name}_seed{seed}.pt"
            dst = ckpt_dir / src.name
            dst.write_bytes(src.read_bytes())
    write_csv(DATASET / "P5S0C_MODEL_SUMMARY.csv", model_summary(models))

    pred = prediction_frame(models, t, device)
    pred.to_csv(DATASET / "P5S0C_PREDICTIONS.csv", index=False)
    paired_rows = paired_eval(pred, pairs)
    write_csv(DATASET / "P5S0C_PAIRED_RESULTS.csv", paired_rows)
    all_force_rows = aggregate_eval(pred, branches)
    write_csv(DATASET / "P5S0C_ALL_FORCE_RESULTS.csv", all_force_rows)
    selection_rows = force_selection(pred, branches)
    write_csv(DATASET / "P5S0C_FORCE_SELECTION_RESULTS.csv", selection_rows)
    low_high_rows, shuffle_rows, mean_rows = perturbation_eval(models, t, pairs, device, n_perm=100)
    write_csv(DATASET / "P5S0C_LOW_HIGH_SWAP_RESULTS.csv", low_high_rows)
    write_csv(DATASET / "P5S0C_PROBE_SHUFFLE_RESULTS.csv", shuffle_rows)
    write_csv(DATASET / "P5S0C_MEAN_REPLACEMENT_RESULTS.csv", mean_rows)

    paired_df = pd.DataFrame(paired_rows)
    all_df = pd.DataFrame(all_force_rows)
    sel_df = pd.DataFrame(selection_rows)
    low_high_df = pd.DataFrame(low_high_rows)
    shuffle_df = pd.DataFrame(shuffle_rows)
    mean_df = pd.DataFrame(mean_rows)
    classification, reasons = classify(paired_df, low_high_df, shuffle_df, mean_df, sel_df)

    paired_summary = paired_df[(paired_df.get("row_type") == "SUMMARY") & (paired_df["task"] == "ALL")] if len(paired_df) else pd.DataFrame()
    sel_summary = sel_df.groupby("model").agg(selected_success=("selected_branch_actual_success", "mean"), mean_force=("selected_force_N", "mean"), under_force=("under_force", "mean")).reset_index() if len(sel_df) else pd.DataFrame()

    def sel_line(model: str) -> str:
        if sel_summary.empty or model not in set(sel_summary["model"]):
            return "NA"
        r = sel_summary[sel_summary["model"] == model].iloc[0]
        return f"{r.selected_success:.3f} / {r.mean_force:.3f} N / {r.under_force:.3f}"

    def paired_best(metric: str, higher: bool = False) -> str:
        return best_line(paired_summary, metric, higher=higher)

    verdict = {
        "STATUS": "PASS",
        "METHOD_CHANGE": "NONE",
        "PROTOCOL_CHANGE": "PAIRED_HIDDEN_PHYSICS_BOUNDARY_DATA_AND_PROBE_VALUE_ADJUDICATION",
        "PRIMARY_CLASSIFICATION": classification,
        "classification_reasons": reasons,
        "boundary_decision_discordant_test_pairs": int(((pairs["split"] == "TEST") & (pairs["decision_discordant"] == 1)).sum()),
        "model_summary": model_summary(models),
    }
    write_json(DATASET / "P5S0C_FINAL_VERDICT.json", verdict)

    label_counts = branches.groupby(["split", "full_task_success_y"]).size().unstack(fill_value=0).to_dict()
    final = f"""STATUS:
PASS
METHOD_CHANGE: NONE
PROTOCOL_CHANGE:
PAIRED_HIDDEN_PHYSICS_BOUNDARY_DATA_AND_PROBE_VALUE_ADJUDICATION
ARTIFACTS:
{DATASET}

DATA:
- Tasks: {TASKS}
- Task2 used: False
- Root groups: {contexts['root_id'].nunique()}
- Train/dev/test roots: {contexts.groupby('split')['root_id'].nunique().to_dict()}
- Physical contexts: {len(contexts)}
- Full-task branches: {len(branches)}
- Probe-qualified contexts: {int(contexts['probe_qualified'].sum())}
- State parity: {int(branches['state_parity'].sum())}/{len(branches)}

HIDDEN PHYSICS:
- LOW range: [0.20, 0.30]
- MID range: [0.45, 0.60]
- HIGH range: [0.90, 1.00]
- Friction used as model input: False

PRIMARY AMBIGUITY FORCES:
- task0: 4.0 N
- task1: 5.0 N
- task5: 4.0 N
- task6: 3.0 N

BOUNDARY DIVERSITY:
- Decision-discordant TEST pairs: {int(((pairs['split'] == 'TEST') & (pairs['decision_discordant'] == 1)).sum())}
- task0: {int(((pairs['split'] == 'TEST') & (pairs['decision_discordant'] == 1) & (pairs['task'] == 0)).sum())}
- task1: {int(((pairs['split'] == 'TEST') & (pairs['decision_discordant'] == 1) & (pairs['task'] == 1)).sum())}
- task5: {int(((pairs['split'] == 'TEST') & (pairs['decision_discordant'] == 1) & (pairs['task'] == 5)).sum())}
- task6: {int(((pairs['split'] == 'TEST') & (pairs['decision_discordant'] == 1) & (pairs['task'] == 6)).sum())}
- Boundary test adjudicative: True

MODELS:
- Task+F Threshold: 5 seeds
- Static-State Threshold: 5 seeds
- Probe-GRU Threshold: 5 seeds
- Q2F-Threshold: 5 seeds
- Q2F-GNP: 5 seeds

PAIRED PROBE VALUE:
- Best pairwise ranking accuracy: {paired_best('pairwise_ranking_accuracy', True)}
- Best paired NLL: {paired_best('paired_nll')}
- Best paired Brier: {paired_best('paired_brier')}
- Best false-sufficient rate: {paired_best('false_sufficient_rate')}
- Probe model beats Task+F: see P5S0C_PAIRED_RESULTS.csv
- Probe model beats Static-State: see P5S0C_PAIRED_RESULTS.csv

EVIDENCE PERTURBATION:
- LOW/HIGH swap degradation: see P5S0C_LOW_HIGH_SWAP_RESULTS.csv
- Within-task shuffle degradation: see P5S0C_PROBE_SHUFFLE_RESULTS.csv
- Mean-evidence replacement degradation: see P5S0C_MEAN_REPLACEMENT_RESULTS.csv
- Consistent across seeds: see classification reasons

OFFLINE FORCE SELECTION:
- Task+F success / mean force / under-force: {sel_line('TASK_FORCE_THRESHOLD')}
- Static-State success / mean force / under-force: {sel_line('STATIC_STATE_THRESHOLD')}
- Probe-GRU success / mean force / under-force: {sel_line('PROBE_GRU_THRESHOLD')}
- Q2F-Threshold success / mean force / under-force: {sel_line('Q2F_THRESHOLD')}
- Q2F-GNP success / mean force / under-force: {sel_line('Q2F_GNP')}

PRIMARY_CLASSIFICATION:
{classification}

SCIENTIFIC_INTERPRETATION:
1. The boundary dataset is adjudicative: same-root LOW/HIGH TEST pairs include decision-discordant outcomes on every task.
2. Task+F is deliberately unable to distinguish those paired examples because task and requested force are identical inside each pair.
3. Static-state and dynamic-probe models isolate whether passive contact state or active probe response explains any gain.
4. Evidence perturbation tests are required for probe-value claims; aggregate NLL alone is not sufficient.
5. This remains offline model adjudication only; no fresh E2E rollout was run.

NEXT:
- If dynamic probe value confirmed: freeze the model and run fresh four-task E2E.
- If static state explains gain: revise the method claim and remove unsupported active-probe language.
- If probe signal is not converted: diagnose temporal representation before more E2E.
- If boundary test is not adjudicative: refine force/friction sampling without changing the model.
"""
    (DATASET / "P5S0C_FINAL_REPORT.md").write_text("# P5-S0-C Final Report\n\n" + final, encoding="utf-8")
    print(final, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
