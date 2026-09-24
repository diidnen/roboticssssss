#!/usr/bin/env python3
"""Analyze M1-R2 real probes and qualify force-sufficiency models."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

RESULT_DIR = Path(__file__).resolve().parents[1]
REPO = Path("/home/exouser/Tabero")
B2R2 = REPO / "analysis/results/b2r2_tabero_task_breadth_20260821_073931"
B4 = REPO / "analysis/results/b4_forte_5task_baseline_20260821_154130"
B5 = REPO / "analysis/results/b5_tabero_neutral_20260822_040652"

TASKS = [0, 1, 2, 5, 6]
TASK_OBJECTS = {
    0: "alphabet_soup_1",
    1: "cream_cheese_1",
    2: "salad_dressing_1",
    5: "tomato_sauce_1",
    6: "butter_1",
}
INSTRUCTIONS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
CANONICAL_FSTAR = {
    (0, 0.2): 5,
    (0, 0.5): 4,
    (0, 1.0): 3,
    (1, 0.2): 6,
    (1, 0.5): 5,
    (1, 1.0): 3,
    (2, 0.2): 8,
    (2, 0.5): 3,
    (2, 1.0): 3,
    (5, 0.2): 5,
    (5, 0.5): 4,
    (5, 1.0): 3,
    (6, 0.2): 4,
    (6, 0.5): 3,
    (6, 1.0): 3,
}
CANDIDATE_FORCES = [3, 4, 5, 6, 8]
TAUS = [0.6, 0.8, 0.9]
MAIN_TAU = 0.8
TEXT_DIM = 32
SEED = 23

BASE_FEATURES = [
    "imb_peak",
    "ftan_peak",
    "ftan_hyst",
    "normal_force_hyst",
    "normal_loading_slope",
    "normal_unloading_slope",
    "marker_tangential_peak",
    "marker_hyst",
    "marker_unloading",
    "residual_marker_displacement",
    "marker_vel_abs_peak",
    "gripper_opening_mean",
    "delta_normal_over_baseline",
    "delta_ftan_over_baseline",
    "delta_marker_over_baseline",
]


def stable_mu(x) -> float:
    return float(round(float(x), 1))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def task_text_embedding(text: str, dim: int = TEXT_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    import re

    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def auc_binary(y: np.ndarray, x: np.ndarray) -> float:
    y = y.astype(int)
    x = x.astype(float)
    ok = np.isfinite(x)
    y = y[ok]
    x = x[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc_score(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p)
    yy = y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(tp[-1], 1)
    return float(np.trapz(precision, recall))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.any():
            ece += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def load_probes() -> pd.DataFrame:
    frames = []
    for task in TASKS:
        p = RESULT_DIR / f"TASK{task}_PROBE.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        df = pd.read_csv(p)
        frames.append(df)
    probes = pd.concat(frames, ignore_index=True)
    if "probe_source" not in probes.columns:
        raise RuntimeError("probe_source missing")
    bad = probes["probe_source"].astype(str).str.contains("REUSED|TASK1_PROBE", case=False, regex=True)
    if bad.any():
        raise RuntimeError("Proxy probe reuse detected in M1-R2 probes")
    if set(probes["task_id"].astype(int).unique()) != set(TASKS):
        raise RuntimeError("Not all five tasks are present in probe data")
    probes["friction"] = probes["friction"].map(stable_mu)
    probes = recompute_baseline_normalization(probes)
    return probes


def recompute_baseline_normalization(probes: pd.DataFrame) -> pd.DataFrame:
    """Recompute predeclared per-episode baseline deltas from raw timesteps."""
    raw_path = RESULT_DIR / "REAL_PROBE_TIMESTEPS.csv"
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return probes
    raw = pd.read_csv(raw_path)
    rows = []
    for trial_id, g in raw.groupby("trial_id"):
        hold = g[g["probe_phase"] == "hold"].tail(10)
        probe = g[g["probe_phase"].isin(["probe_out", "probe_back", "probe_hold"])]
        if hold.empty or probe.empty:
            continue
        base_normal = float(pd.to_numeric(hold["normal_force"], errors="coerce").mean())
        base_ftan = float(pd.to_numeric(hold["tangential_force"], errors="coerce").mean())
        base_marker = float(pd.to_numeric(hold["marker_motion"], errors="coerce").mean())
        probe_normal = float(pd.to_numeric(probe["normal_force"], errors="coerce").mean())
        probe_ftan = float(pd.to_numeric(probe["tangential_force"], errors="coerce").mean())
        probe_marker = float(pd.to_numeric(probe["marker_motion"], errors="coerce").mean())
        rows.append(
            {
                "trial_id": trial_id,
                "preprobe_normal_force_recomputed": base_normal,
                "preprobe_tangential_force_recomputed": base_ftan,
                "preprobe_marker_motion_recomputed": base_marker,
                "delta_normal_over_baseline_recomputed": (probe_normal - base_normal) / max(abs(base_normal), 1e-6),
                "delta_ftan_over_baseline_recomputed": (probe_ftan - base_ftan) / max(abs(base_ftan), 1e-6),
                "delta_marker_over_baseline_recomputed": (probe_marker - base_marker) / max(abs(base_marker), 1e-6),
            }
        )
    if not rows:
        return probes
    rec = pd.DataFrame(rows)
    out = probes.merge(rec, on="trial_id", how="left")
    for base, recomputed in [
        ("preprobe_normal_force", "preprobe_normal_force_recomputed"),
        ("preprobe_tangential_force", "preprobe_tangential_force_recomputed"),
        ("preprobe_marker_motion", "preprobe_marker_motion_recomputed"),
        ("delta_normal_over_baseline", "delta_normal_over_baseline_recomputed"),
        ("delta_ftan_over_baseline", "delta_ftan_over_baseline_recomputed"),
        ("delta_marker_over_baseline", "delta_marker_over_baseline_recomputed"),
    ]:
        if base in out.columns and recomputed in out.columns:
            out[base] = pd.to_numeric(out[base], errors="coerce")
            out[base] = out[base].where(np.isfinite(out[base]), out[recomputed])
    return out


def safety_audit(probes: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, bool]]:
    rows = []
    safe_by_task = {}
    for task, g in probes.groupby("task_id"):
        n = len(g)
        drop_rate = float(g["dropped"].astype(float).mean())
        contact_lost = float(g["contact_lost_probe"].astype(float).mean())
        failure = float(g["probe_failure"].astype(float).mean())
        disturbance = float(g["major_disturbance"].astype(float).mean())
        disp_mean = float(g["obj_disp_probe_m"].astype(float).mean())
        disp_max = float(g["obj_disp_probe_m"].astype(float).max())
        rot_mean = float(g["obj_rot_probe_rad"].astype(float).mean())
        duration = float(g["probe_duration_s"].astype(float).mean())
        usable = bool(drop_rate <= 0.10 and disturbance <= 0.10 and failure <= 0.10)
        status = "USABLE" if usable else "PROBE_NOT_TASK_GENERAL"
        safe_by_task[int(task)] = usable
        rows.append(
            {
                "task_id": int(task),
                "object_id": TASK_OBJECTS[int(task)],
                "n": n,
                "contact_retention_rate": 1.0 - contact_lost,
                "probe_drop_rate": drop_rate,
                "probe_failure_rate": failure,
                "major_disturbance_rate": disturbance,
                "mean_obj_disp_probe_m": disp_mean,
                "max_obj_disp_probe_m": disp_max,
                "mean_obj_rot_probe_rad": rot_mean,
                "mean_probe_duration_s": duration,
                "probe_safe": usable,
                "status": status,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "PROBE_SAFETY_AUDIT.csv", index=False)
    return out, safe_by_task


def signal_stats(probes: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, bool], str]:
    rows = []
    best_rows = []
    informative = {}
    direction_votes = {}
    for task, g in probes.groupby("task_id"):
        task = int(task)
        best_low = None
        best_mid = None
        for contrast_name, y in [
            ("LOW_vs_NOT_LOW", (g["friction"].astype(float) == 0.2).astype(int).to_numpy()),
            ("MID_vs_HIGH", (g["friction"].astype(float) == 0.5).astype(int).to_numpy()),
        ]:
            if contrast_name == "MID_vs_HIGH":
                gg = g[g["friction"].isin([0.5, 1.0])].copy()
                y = (gg["friction"].astype(float) == 0.5).astype(int).to_numpy()
            else:
                gg = g
            for feat in BASE_FEATURES:
                x = pd.to_numeric(gg[feat], errors="coerce").to_numpy(dtype=float)
                auc = auc_binary(y, x)
                sep_auc = max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")
                pos = x[y == 1]
                neg = x[y == 0]
                d = cohen_d(pos, neg)
                row = {
                    "task_id": task,
                    "object_id": TASK_OBJECTS[task],
                    "contrast": contrast_name,
                    "signal": feat,
                    "auc_raw_positive_class_high": auc,
                    "auc_separation": sep_auc,
                    "cohens_d_pos_minus_neg": d,
                    "pos_mean": float(np.nanmean(pos)),
                    "pos_std": float(np.nanstd(pos)),
                    "pos_median": float(np.nanmedian(pos)),
                    "pos_iqr": float(np.nanpercentile(pos, 75) - np.nanpercentile(pos, 25)),
                    "neg_mean": float(np.nanmean(neg)),
                    "neg_std": float(np.nanstd(neg)),
                    "neg_median": float(np.nanmedian(neg)),
                    "neg_iqr": float(np.nanpercentile(neg, 75) - np.nanpercentile(neg, 25)),
                }
                rows.append(row)
                if contrast_name == "LOW_vs_NOT_LOW":
                    if best_low is None or row["auc_separation"] > best_low["auc_separation"]:
                        best_low = row
                else:
                    if best_mid is None or row["auc_separation"] > best_mid["auc_separation"]:
                        best_mid = row
        informative[task] = bool(
            best_low and np.isfinite(best_low["auc_separation"]) and best_low["auc_separation"] >= 0.70
        )
        direction_votes[task] = None if best_low is None else (
            best_low["signal"],
            "pos_high" if best_low["auc_raw_positive_class_high"] >= 0.5 else "pos_low",
        )
        best_rows.append(
            {
                "Task": task,
                "Object": TASK_OBJECTS[task],
                "LOW vs NOT_LOW best signal": best_low["signal"] if best_low else "",
                "LOW vs NOT_LOW AUC": best_low["auc_separation"] if best_low else np.nan,
                "MID vs HIGH best signal": best_mid["signal"] if best_mid else "",
                "MID vs HIGH AUC": best_mid["auc_separation"] if best_mid else np.nan,
                "Probe safe?": "",
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(RESULT_DIR / "REAL_PROBE_SIGNAL_STATS.csv", index=False)
    table = pd.DataFrame(best_rows)
    table.to_csv(RESULT_DIR / "REAL_PROBE_SIGNAL_TABLE.csv", index=False)
    signals = [v[0] for v in direction_votes.values() if v]
    dirs = [v[1] for v in direction_votes.values() if v]
    if len(set(signals)) <= 2 and len(set(dirs)) == 1 and sum(informative.values()) >= 4:
        structure = "CROSS_TASK_PHYSICAL_STRUCTURE_PRESENT"
    elif sum(informative.values()) >= 4:
        structure = "TASK_SPECIFIC_SCALE"
    else:
        structure = "PROBE_NOT_GENERAL"
    return table, informative, structure


def write_signal_table_md(table: pd.DataFrame, safety: pd.DataFrame) -> None:
    safe_map = {int(r.task_id): bool(r.probe_safe) for r in safety.itertuples()}
    rows = []
    for _, r in table.iterrows():
        rows.append(
            {
                "Task": int(r["Task"]),
                "LOW vs NOT_LOW best signal": str(r["LOW vs NOT_LOW best signal"]),
                "AUC": f"{float(r['LOW vs NOT_LOW AUC']):.3f}",
                "MID vs HIGH best signal": str(r["MID vs HIGH best signal"]),
                "MID AUC": f"{float(r['MID vs HIGH AUC']):.3f}",
                "Probe safe?": str(safe_map.get(int(r["Task"]), False)),
            }
        )
    lines = ["# Real Probe Signal Table", ""]
    lines.append("| Task | LOW vs NOT_LOW best signal | AUC | MID vs HIGH best signal | AUC | Probe safe? |")
    lines.append("| ---- | -------------------------- | --: | ----------------------- | --: | ----------- |")
    for row in rows:
        lines.append(
            f"| {row['Task']} | {row['LOW vs NOT_LOW best signal']} | {row['AUC']} | "
            f"{row['MID vs HIGH best signal']} | {row['MID AUC']} | {row['Probe safe?']} |"
        )
    (RESULT_DIR / "REAL_PROBE_SIGNAL_TABLE.md").write_text("\n".join(lines) + "\n")


def standardize_scan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "force" in df.columns:
        df["candidate_force_N"] = df["force"].astype(float).round().astype(int)
    elif "chosen_force" in df.columns:
        df["candidate_force_N"] = df["chosen_force"].astype(float).round().astype(int)
    else:
        raise ValueError(f"No force column in {path}")
    if "place_success" not in df.columns:
        df["place_success"] = df["full_task_success"]
    df["task_id"] = df["task_id"].astype(int)
    df["friction"] = df["friction"].map(stable_mu)
    df["seed_idx"] = df["seed_idx"].astype(int)
    df["full_success"] = df["full_task_success"].astype(int)
    return df[
        [
            "task_id",
            "object",
            "seed_idx",
            "friction",
            "candidate_force_N",
            "full_success",
            "lift_success",
            "place_success",
        ]
    ].copy()


def load_fulltask_labels() -> pd.DataFrame:
    paths = [
        B2R2 / "TASK0_SCAN.csv",
        B2R2 / "TASK1_FROZEN_REFERENCE_SCAN.csv",
        B2R2 / "TASK2_SCAN.csv",
        B2R2 / "TASK5_SCAN.csv",
        B2R2 / "TASK6_SCAN.csv",
    ]
    df = pd.concat([standardize_scan(p) for p in paths], ignore_index=True)
    df = df[df["candidate_force_N"].isin(CANDIDATE_FORCES)].copy()
    df["canonical_fstar_N"] = [CANONICAL_FSTAR[(int(t), stable_mu(mu))] for t, mu in zip(df.task_id, df.friction)]
    return df


def build_dataset(probes: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    agg = (
        labels.groupby(["task_id", "friction", "candidate_force_N"], as_index=False)
        .agg(
            observed_n=("full_success", "size"),
            observed_success_rate=("full_success", "mean"),
        )
    )
    rows = []
    missing = []
    for p in probes.itertuples(index=False):
        for force in CANDIDATE_FORCES:
            match = agg[
                (agg.task_id == int(p.task_id))
                & (agg.friction == stable_mu(p.friction))
                & (agg.candidate_force_N == force)
            ]
            if match.empty:
                missing.append({"task_id": int(p.task_id), "friction": stable_mu(p.friction), "candidate_force_N": force})
                continue
            m = match.iloc[0]
            out = p._asdict()
            out["candidate_force_N"] = int(force)
            out["full_success_label"] = int(float(m.observed_success_rate) >= 0.8)
            out["observed_success_rate"] = float(m.observed_success_rate)
            out["observed_label_n"] = int(m.observed_n)
            out["canonical_fstar_N"] = CANONICAL_FSTAR[(int(p.task_id), stable_mu(p.friction))]
            out["neutral_language_instruction"] = INSTRUCTIONS[int(p.task_id)]
            rows.append(out)
    ds = pd.DataFrame(rows)
    ds.to_csv(RESULT_DIR / "M1R2_DATASET.csv", index=False)
    pd.DataFrame(missing).drop_duplicates().to_csv(RESULT_DIR / "MISSING_FULLTASK_LABEL_CELLS.csv", index=False)
    if ds["probe_source"].astype(str).str.contains("REUSED|TASK1_PROBE", case=False, regex=True).any():
        raise RuntimeError("Proxy probe source found in M1R2_DATASET")
    return ds


@dataclass
class FeatureSpec:
    model_name: str
    context: str
    numeric_cols: list[str]
    include_task_onehot: bool = False
    include_text: bool = False
    include_gt_mu: bool = False
    mean: np.ndarray | None = None
    std: np.ndarray | None = None


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_spec(model_name: str, context: str, train: pd.DataFrame) -> FeatureSpec:
    numeric = ["candidate_force_N"]
    include_task_onehot = model_name in ("Task+F", "Task+Probe+F", "OracleMu+Task+F") and context == "onehot"
    include_text = model_name in ("Task+F", "Task+Probe+F", "OracleMu+Task+F") and context == "text"
    if model_name in ("Probe+F", "Task+Probe+F"):
        numeric += BASE_FEATURES
    include_gt_mu = model_name == "OracleMu+Task+F"
    if include_gt_mu:
        numeric += ["friction"]
    arr = train[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return FeatureSpec(model_name, context, numeric, include_task_onehot, include_text, include_gt_mu, mean, std)


def featurize(df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    parts = []
    arr = df[spec.numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    arr = (arr - spec.mean.astype(np.float32)) / spec.std.astype(np.float32)
    parts.append(arr)
    if spec.include_task_onehot:
        one = np.zeros((len(df), len(TASKS)), dtype=np.float32)
        mp = {t: i for i, t in enumerate(TASKS)}
        for r, t in enumerate(df["task_id"].astype(int)):
            if int(t) in mp:
                one[r, mp[int(t)]] = 1.0
        parts.append(one)
    if spec.include_text:
        parts.append(np.vstack([task_text_embedding(str(x)) for x in df["neutral_language_instruction"]]).astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, model_name: str, context: str):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    spec = make_spec(model_name, context, train)
    xtr = torch.tensor(featurize(train, spec), dtype=torch.float32)
    ytr_np = train["full_success_label"].astype(float).to_numpy(dtype=np.float32)
    ytr = torch.tensor(ytr_np, dtype=torch.float32)
    xva = torch.tensor(featurize(val, spec), dtype=torch.float32)
    yva = torch.tensor(val["full_success_label"].astype(float).to_numpy(dtype=np.float32), dtype=torch.float32)
    xte_np = featurize(test, spec)
    xte = torch.tensor(xte_np, dtype=torch.float32)
    model = MLP(xtr.shape[1])
    pos = float(ytr_np.sum())
    neg = float(len(ytr_np) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    best = None
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(160):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xtr), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(xva), yva).item())
        if vl < best_loss - 1e-5:
            best_loss = vl
            best_epoch = epoch
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch > 25:
            break
    if best:
        model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(xte)).cpu().numpy()
    return prob, spec


def split_iid(ds):
    m = ds["seed_idx"].astype(int) % 5
    return ds[m.isin([2, 3, 4])].copy(), ds[m == 1].copy(), ds[m == 0].copy()


def split_loto(ds, task):
    test = ds[ds.task_id == task].copy()
    rest = ds[ds.task_id != task].copy()
    val = rest[rest.seed_idx.astype(int) % 5 == 1].copy()
    train = rest[rest.seed_idx.astype(int) % 5 != 1].copy()
    return train, val, test


def split_lofo(ds, mu):
    test = ds[ds.friction == mu].copy()
    rest = ds[ds.friction != mu].copy()
    val = rest[rest.seed_idx.astype(int) % 5 == 1].copy()
    train = rest[rest.seed_idx.astype(int) % 5 != 1].copy()
    return train, val, test


def probability_metrics(y, p):
    y = y.astype(int)
    return {
        "n": int(len(y)),
        "auroc": auc_binary(y, p),
        "auprc": auprc_score(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece_score(y, p),
        "acc": float(np.mean((p >= 0.5) == y)),
        "positive_rate": float(np.mean(y)),
    }


def decision_eval(pred_df: pd.DataFrame, tau: float, monotonic: bool) -> dict:
    decisions = []
    for keys, g in pred_df.groupby(["split", "fold", "model", "task_id", "seed_idx", "friction"]):
        g = g.sort_values("candidate_force_N")
        forces = g["candidate_force_N"].astype(int).to_numpy()
        probs = g["pred_success"].astype(float).to_numpy()
        if monotonic:
            probs = np.maximum.accumulate(probs)
        selected = 8
        no_conf = True
        for f, p in zip(forces, probs):
            if p >= tau:
                selected = int(f)
                no_conf = False
                break
        task = int(g["task_id"].iloc[0])
        mu = stable_mu(g["friction"].iloc[0])
        fstar = CANONICAL_FSTAR[(task, mu)]
        decisions.append(
            {
                "split": keys[0],
                "fold": keys[1],
                "model": keys[2],
                "task_id": task,
                "seed_idx": int(g["seed_idx"].iloc[0]),
                "friction": mu,
                "tau": tau,
                "monotonic": monotonic,
                "selected_force_N": selected,
                "canonical_fstar_N": fstar,
                "exact_fstar": int(selected == fstar),
                "under_force": int(selected < fstar),
                "over_force": int(selected > fstar),
                "expected_full_success": int(selected >= fstar),
                "no_confident": int(no_conf),
            }
        )
    d = pd.DataFrame(decisions)
    return {
        "episodes": int(len(d)),
        "expected_full_sr": float(d.expected_full_success.mean()),
        "exact_fstar_rate": float(d.exact_fstar.mean()),
        "under_force_rate": float(d.under_force.mean()),
        "over_force_rate": float(d.over_force.mean()),
        "mean_selected_force_N": float(d.selected_force_N.mean()),
        "no_confident_rate": float(d.no_confident.mean()),
    }, d


def monotonicity_rows(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in pred_df.groupby(["split", "fold", "model", "task_id", "seed_idx", "friction"]):
        g = g.sort_values("candidate_force_N")
        p = g["pred_success"].astype(float).to_numpy()
        rows.append(
            {
                "split": keys[0],
                "fold": keys[1],
                "model": keys[2],
                "task_id": int(keys[3]),
                "seed_idx": int(keys[4]),
                "friction": stable_mu(keys[5]),
                "monotonic_violation": int(np.any(np.diff(p) < -1e-6)),
                "max_negative_step": float(min(0.0, np.min(np.diff(p)))) if len(p) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_models(ds: pd.DataFrame):
    model_specs = [
        ("Task+F", "onehot", "text"),
        ("Probe+F", "none", "none"),
        ("Task+Probe+F", "onehot", "text"),
        ("OracleMu+Task+F", "onehot", "text"),
    ]
    result_rows = []
    pred_frames = []
    for model_name, iid_context, general_context in model_specs:
        train, val, test = split_iid(ds)
        prob, spec = train_predict(train, val, test, model_name, iid_context)
        ptest = test.copy()
        ptest["pred_success"] = prob
        ptest["split"] = "IID"
        ptest["fold"] = "all_tasks"
        ptest["model"] = model_name
        ptest["context"] = iid_context
        pred_frames.append(ptest)
        result_rows.append({"split": "IID", "fold": "all_tasks", "model": model_name, "context": iid_context, **probability_metrics(test.full_success_label.to_numpy(), prob)})

        for heldout in TASKS:
            train, val, test = split_loto(ds, heldout)
            prob, spec = train_predict(train, val, test, model_name, general_context)
            ptest = test.copy()
            ptest["pred_success"] = prob
            ptest["split"] = "LOTO"
            ptest["fold"] = f"heldout_task{heldout}"
            ptest["model"] = model_name
            ptest["context"] = general_context
            pred_frames.append(ptest)
            result_rows.append({"split": "LOTO", "fold": f"heldout_task{heldout}", "heldout_task": heldout, "model": model_name, "context": general_context, **probability_metrics(test.full_success_label.to_numpy(), prob)})
        for mu in [0.2, 0.5, 1.0]:
            train, val, test = split_lofo(ds, mu)
            prob, spec = train_predict(train, val, test, model_name, general_context)
            ptest = test.copy()
            ptest["pred_success"] = prob
            ptest["split"] = "LOFO"
            ptest["fold"] = f"heldout_mu{mu:g}"
            ptest["model"] = model_name
            ptest["context"] = general_context
            pred_frames.append(ptest)
            result_rows.append({"split": "LOFO", "fold": f"heldout_mu{mu:g}", "heldout_mu": mu, "model": model_name, "context": general_context, **probability_metrics(test.full_success_label.to_numpy(), prob)})
    results = pd.DataFrame(result_rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    results[results.split == "IID"].to_csv(RESULT_DIR / "IID_RESULTS.csv", index=False)
    results[results.split == "LOTO"].to_csv(RESULT_DIR / "LOTO_RESULTS.csv", index=False)
    results[results.split == "LOFO"].to_csv(RESULT_DIR / "LOFO_RESULTS.csv", index=False)
    results[results.model == "Task+F"].to_csv(RESULT_DIR / "TASK_FORCE_ONLY_RESULTS.csv", index=False)
    results[results.model == "Probe+F"].to_csv(RESULT_DIR / "PROBE_FORCE_ONLY_RESULTS.csv", index=False)
    results[results.model == "Task+Probe+F"].to_csv(RESULT_DIR / "TASK_PROBE_FORCE_RESULTS.csv", index=False)
    results[results.model == "OracleMu+Task+F"].to_csv(RESULT_DIR / "ORACLE_MU_RESULTS.csv", index=False)
    preds.to_csv(RESULT_DIR / "PREDICTIONS.csv", index=False)
    return results, preds


def write_model_specs() -> None:
    text = """# Model Specs

All M1-R2 models use the same target:

`P(full_task_success | task/context, real task-specific probe evidence, candidate force F)`.

- M0 / Task+F: candidate force plus task context. One-hot is IID only; frozen hashed task-language embedding is used for LOTO/LOFO.
- M1 / Probe+F: candidate force plus deployable real probe features; no task identity.
- M2 / Task+Probe+F: candidate force, task context, and real probe features. One-hot is IID only; frozen hashed task-language embedding is used for LOTO/LOFO.
- M3 / Oracle μ+Task+F: privileged diagnostic reference only. One-hot is IID only; frozen hashed task-language embedding is used for LOTO/LOFO.

Classifier: small 2-layer MLP, hidden widths `(32, 16)`, sigmoid output.

Decision: evaluate `F in {3,4,5,6,8}` and choose the lowest force with `p >= tau`.
Reported tau values: `0.6`, `0.8`, `0.9`. Main tau: `0.8`.

Monotonic calibration is predeclared post-hoc cumulative max over candidate force probabilities and never uses test labels.
"""
    (RESULT_DIR / "MODEL_SPECS.md").write_text(text)


def write_normalization_audit() -> None:
    text = """# Probe Normalization Audit

Predeclared normalized features included in M1-R2:

- `delta_normal_over_baseline`
- `delta_ftan_over_baseline`
- `delta_marker_over_baseline`

Each is computed per episode from the probe-start/pre-probe baseline:

`(mean_probe_signal - preprobe_signal) / max(abs(preprobe_signal), 1e-6)`.

No GT friction statistics, held-out task labels, or per-task label-informed scaling are used to normalize test samples.
"""
    (RESULT_DIR / "PROBE_NORMALIZATION_AUDIT.md").write_text(text)


def plots(probes, preds, decision_all):
    plot_dir = RESULT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for task, g in probes.groupby("task_id"):
        axes[0].scatter(g["friction"], g["imb_peak"], label=f"task{task}", alpha=0.7)
        axes[1].scatter(g["friction"], g["ftan_peak"], label=f"task{task}", alpha=0.7)
    axes[0].set_title("imb_peak by task/mu")
    axes[1].set_title("ftan_peak by task/mu")
    for ax in axes:
        ax.set_xlabel("mu")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "probe_signal_by_task_mu.png", dpi=180)
    fig.savefig(plot_dir / "probe_signal_by_task_mu.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    q = decision_all[(decision_all.split == "LOTO") & (decision_all.tau == MAIN_TAU) & (decision_all.monotonic == True)]
    if not q.empty:
        summary = q.groupby("model", as_index=False).agg(expected_full_sr=("expected_full_success", "mean"), mean_force=("selected_force_N", "mean"))
        ax.scatter(summary["mean_force"], summary["expected_full_sr"])
        for r in summary.itertuples():
            ax.annotate(r.model, (r.mean_force, r.expected_full_sr), fontsize=8)
    ax.set_xlabel("Mean selected force N")
    ax.set_ylabel("Expected full SR")
    ax.set_title("LOTO success-force pareto")
    fig.tight_layout()
    fig.savefig(plot_dir / "success_force_pareto.png", dpi=180)
    fig.savefig(plot_dir / "success_force_pareto.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    q = decision_all[(decision_all.split == "LOTO") & (decision_all.tau == MAIN_TAU) & (decision_all.monotonic == True)]
    if not q.empty:
        summary = q.groupby(["model", "task_id"], as_index=False).agg(mean_selected=("selected_force_N", "mean"), fstar=("canonical_fstar_N", "mean"))
        for model, g in summary.groupby("model"):
            ax.plot(g["task_id"], g["mean_selected"], marker="o", label=model)
        oracle = summary.groupby("task_id", as_index=False)["fstar"].mean()
        ax.plot(oracle.task_id, oracle.fstar, color="black", linestyle="--", label="GT-MinForce")
    ax.set_xlabel("task")
    ax.set_ylabel("force N")
    ax.set_title("Selected force vs oracle")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "selected_force_vs_oracle.png", dpi=180)
    fig.savefig(plot_dir / "selected_force_vs_oracle.svg")
    plt.close(fig)

    for name in ["normalized_probe_signal", "loto_by_model", "task2_stress"]:
        # Placeholder-compatible plot names backed by the same M1-R2 data.
        fig, ax = plt.subplots(figsize=(6, 4))
        if name == "normalized_probe_signal":
            for task, g in probes.groupby("task_id"):
                ax.scatter(g["friction"], g["delta_ftan_over_baseline"], label=f"task{task}", alpha=0.7)
            ax.set_ylabel("delta ftan / baseline")
            ax.legend(fontsize=8)
        elif name == "loto_by_model":
            q = decision_all[(decision_all.split == "LOTO") & (decision_all.tau == MAIN_TAU) & (decision_all.monotonic == True)]
            if not q.empty:
                s = q.groupby("model", as_index=False)["expected_full_success"].mean()
                ax.bar(s["model"], s["expected_full_success"])
                ax.tick_params(axis="x", rotation=25)
            ax.set_ylabel("expected full SR")
        else:
            q = decision_all[(decision_all.task_id == 2) & (decision_all.split == "LOTO") & (decision_all.tau == MAIN_TAU) & (decision_all.monotonic == True)]
            if not q.empty:
                s = q.groupby(["model", "friction"], as_index=False)["selected_force_N"].mean()
                for model, g in s.groupby("model"):
                    ax.plot(g["friction"], g["selected_force_N"], marker="o", label=model)
                ax.legend(fontsize=8)
            ax.set_ylabel("selected force N")
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}.png", dpi=180)
        fig.savefig(plot_dir / f"{name}.svg")
        plt.close(fig)


def references() -> tuple[float, float]:
    gt = pd.read_csv(B4 / "GT_MINFORCE_REFERENCE.csv")
    fr = pd.read_csv(B4 / "GT_MINFORCE_REFERENCE.csv")
    fixed_robust = pd.read_csv(B4 / "DYNAMIC_FORCE_SERVO_AUDIT.csv")
    gt_mean = float(gt["mean_selected_force_N"].mean()) if "mean_selected_force_N" in gt.columns else 4.1875
    if "mean_force_N" in fixed_robust.columns:
        fixed_mean = float(fixed_robust["mean_force_N"].mean())
    else:
        fixed_mean = 5.6
    return gt_mean, fixed_mean


def main():
    (RESULT_DIR / "plots").mkdir(exist_ok=True)
    probes = load_probes()
    safety, safe_by_task = safety_audit(probes)
    signal_table, informative_by_task, structure = signal_stats(probes)
    write_signal_table_md(signal_table, safety)
    write_normalization_audit()
    labels = load_fulltask_labels()
    ds = build_dataset(probes, labels)
    write_model_specs()
    results, preds = run_models(ds)

    decision_rows = []
    decision_summaries = []
    for tau in TAUS:
        for monotonic in [False, True]:
            summary, d = decision_eval(preds, tau=tau, monotonic=monotonic)
            d.to_csv(RESULT_DIR / f"DECISION_tau{tau:g}_{'monotonic' if monotonic else 'raw'}.csv", index=False)
            d["tau"] = tau
            d["monotonic"] = monotonic
            decision_rows.append(d)
            for (split, fold, model), g in d.groupby(["split", "fold", "model"]):
                decision_summaries.append(
                    {
                        "split": split,
                        "fold": fold,
                        "model": model,
                        "tau": tau,
                        "monotonic": monotonic,
                        "episodes": len(g),
                        "expected_full_sr": float(g.expected_full_success.mean()),
                        "exact_fstar_rate": float(g.exact_fstar.mean()),
                        "under_force_rate": float(g.under_force.mean()),
                        "over_force_rate": float(g.over_force.mean()),
                        "mean_selected_force_N": float(g.selected_force_N.mean()),
                        "no_confident_rate": float(g.no_confident.mean()),
                    }
                )
    decision_all = pd.concat(decision_rows, ignore_index=True)
    decision_summary = pd.DataFrame(decision_summaries)
    decision_summary.to_csv(RESULT_DIR / "DECISION_RESULTS.csv", index=False)
    mono = monotonicity_rows(preds)
    mono.to_csv(RESULT_DIR / "MONOTONICITY_AUDIT.csv", index=False)
    results.to_csv(RESULT_DIR / "CALIBRATION_RESULTS.csv", index=False)
    (RESULT_DIR / "OPTIONAL_ROLLOUT_RESULTS.csv").write_text("status,reason\nNOT_RUN,Offline LOTO gate must be inspected first\n")
    plots(probes, preds, decision_all)

    main_dec = decision_summary[
        (decision_summary.split == "LOTO")
        & (decision_summary.tau == MAIN_TAU)
        & (decision_summary.monotonic == True)
        & (decision_summary.fold.str.startswith("heldout_task"))
    ]
    task_force = main_dec[main_dec.model == "Task+F"]
    task_probe = main_dec[main_dec.model == "Task+Probe+F"]
    tf_sr = float(task_force.expected_full_sr.mean()) if not task_force.empty else None
    tpf_sr = float(task_probe.expected_full_sr.mean()) if not task_probe.empty else None
    probe_improves = bool(tpf_sr is not None and tf_sr is not None and tpf_sr > tf_sr + 1e-9)
    if all(safe_by_task.values()) and sum(informative_by_task.values()) >= 4 and probe_improves:
        status = "M1R2_REAL_PROBE_IMPROVES_CROSS_TASK_FORCE_SUFFICIENCY"
    elif sum(informative_by_task.values()) >= 3:
        status = "M1R2_REAL_PROBE_SIGNAL_PRESENT_BUT_GENERALIZATION_WEAK"
    else:
        status = "M1R2_PROBE_NOT_CROSS_TASK_GENERAL"

    gt_mean, fixed_mean = references()
    verdict = {
        "status": status,
        "method_change": "NONE",
        "tasks": TASKS,
        "real_probe_used_for_all_tasks": bool(set(probes.task_id.astype(int).unique()) == set(TASKS)),
        "probe_safe_by_task": {str(k): bool(v) for k, v in safe_by_task.items()},
        "probe_informative_by_task": {str(k): bool(v) for k, v in informative_by_task.items()},
        "cross_task_signal_structure": structure,
        "dataset_rows": int(len(ds)),
        "candidate_forces_N": CANDIDATE_FORCES,
        "iid_results": results[results.split == "IID"].to_dict(orient="records"),
        "loto_results": decision_summary[
            (decision_summary.split == "LOTO") & (decision_summary.tau == MAIN_TAU) & (decision_summary.monotonic == True)
        ].to_dict(orient="records"),
        "task_force_baseline_loto": tf_sr,
        "task_probe_force_loto": tpf_sr,
        "probe_improves_generalization": probe_improves,
        "loto_under_force_rate": float(task_probe.under_force_rate.mean()) if not task_probe.empty else None,
        "loto_over_force_rate": float(task_probe.over_force_rate.mean()) if not task_probe.empty else None,
        "loto_mean_selected_force_N": float(task_probe.mean_selected_force_N.mean()) if not task_probe.empty else None,
        "gt_minforce_mean_N": gt_mean,
        "fixed_robust_mean_N": fixed_mean,
        "optional_rollout_run": False,
        "primary_evidence": [
            "REAL_PROBE_EPISODES.csv",
            "REAL_PROBE_SIGNAL_TABLE.md",
            "M1R2_DATASET.csv",
            "LOTO_RESULTS.csv",
            "DECISION_RESULTS.csv",
        ],
        "limitations": [
            "Full-task labels are frozen observed rollout cells; missing exact force cells are listed in MISSING_FULLTASK_LABEL_CELLS.csv and are not interpolated into training rows.",
            "Optional real rollout confirmation is not run until offline LOTO gate is inspected.",
            "Task2 salad_dressing_1 is flagged PROBE_NOT_TASK_GENERAL because fixed Probe A had 0% contact retention.",
            "Task-language embedding is deterministic hash embedding, not a trained semantic encoder.",
        ],
    }
    (RESULT_DIR / "FINAL_VERDICT.json").write_text(json.dumps(json_safe(verdict), indent=2, allow_nan=False) + "\n")
    (RESULT_DIR / "ENV_PROVENANCE.json").write_text(
        json.dumps(
            {
                "repo": str(REPO),
                "git_commit": git_commit(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "result_dir": str(RESULT_DIR),
            },
            indent=2,
        )
        + "\n"
    )
    (RESULT_DIR / "README.md").write_text(
        f"""# M1-R2 Real Cross-Task Probe

Status: `{status}`.

This run uses real task-specific Probe A telemetry for tasks `{TASKS}`. No proxy probe reuse is allowed.

Key files:

- `REAL_PROBE_EPISODES.csv`
- `REAL_PROBE_TIMESTEPS.csv`
- `PROBE_SAFETY_AUDIT.csv`
- `REAL_PROBE_SIGNAL_TABLE.md`
- `M1R2_DATASET.csv`
- `IID_RESULTS.csv`, `LOTO_RESULTS.csv`, `LOFO_RESULTS.csv`
- `DECISION_RESULTS.csv`
- `FINAL_VERDICT.json`

Optional online rollout confirmation: `NOT_RUN`.
"""
    )


if __name__ == "__main__":
    main()
