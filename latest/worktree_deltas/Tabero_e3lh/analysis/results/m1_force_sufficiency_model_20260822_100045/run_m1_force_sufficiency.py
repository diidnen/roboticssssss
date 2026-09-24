#!/usr/bin/env python3
"""M1 force-sufficiency offline experiment.

This script is intentionally self-contained and writes only to its containing
result directory. It reads frozen Tabero analysis artifacts as immutable inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

RESULT_DIR = Path(__file__).resolve().parent
REPO_ROOT = RESULT_DIR.parents[2]
VENDOR = RESULT_DIR / "_vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

CANDIDATE_FORCES = [3, 4, 5, 6, 8]
POSITIVE_TASKS = [0, 1, 2, 5, 6]
NEGATIVE_CONTROL_TASKS = [3, 7, 9]
TAU = 0.8
TEXT_DIM = 32
HIDDEN = (32, 16)
SEED = 17

FROZEN_B2R2 = REPO_ROOT / "analysis/results/b2r2_tabero_task_breadth_20260821_073931"
FROZEN_P3 = REPO_ROOT / "analysis/results/p3_probe_belief_decision_20260819_104513"
FROZEN_B4 = REPO_ROOT / "analysis/results/b4_forte_5task_baseline_20260821_154130"
FROZEN_B5 = REPO_ROOT / "analysis/results/b5_tabero_neutral_20260822_040652"

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

NEGATIVE_FSTAR = {
    (3, 0.2): 3,
    (3, 0.5): 3,
    (3, 1.0): 3,
    (7, 0.2): 3,
    (7, 0.5): 3,
    (7, 1.0): 3,
    (9, 0.2): 3,
    (9, 0.5): 3,
    (9, 1.0): 3,
}

INSTRUCTIONS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    3: "pick up the bbq sauce and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
    7: "pick up the milk and place it in the basket",
    9: "pick up the orange juice and place it in the basket",
}

PROBE_FEATURES = [
    "z_marker_tang_peak",
    "z_marker_tang_mean",
    "z_marker_mean_peak",
    "z_marker_unloading",
    "z_marker_asym_peak",
    "z_marker_vel_abs_peak",
    "z_hm_mean",
    "z_f_meas_mean",
    "z_imbalance_peak",
    "z_ftan_peak",
    "z_rel_xy_peak",
    "z_rel_z_min",
    "tactile_ok",
    "probe_duration_s",
    "probe_amp_mm",
    "probe_path_mm",
    "obj_disp_probe_m",
]


def stable_float_mu(x: float) -> float:
    return float(round(float(x), 1))


def task_text_embedding(text: str, dim: int = TEXT_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def ensure_dirs() -> None:
    for name in ["checkpoints", "plots", "logs"]:
        (RESULT_DIR / name).mkdir(exist_ok=True)


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        return "UNKNOWN"


def standardize_task_scan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "force" in df.columns:
        df["candidate_force_N"] = df["force"].astype(float).round().astype(int)
    elif "chosen_force" in df.columns:
        df["candidate_force_N"] = df["chosen_force"].astype(float).round().astype(int)
    else:
        raise ValueError(f"No force column in {path}")

    if "instruction" not in df.columns:
        df["instruction"] = df["task_id"].map(INSTRUCTIONS)
    if "place_success" not in df.columns:
        df["place_success"] = df["full_task_success"]
    if "mean_grip_force" not in df.columns and "mean_force" in df.columns:
        df["mean_grip_force"] = df["mean_force"]
    if "peak_force" not in df.columns:
        df["peak_force"] = np.nan
    if "integrated_force" not in df.columns:
        df["integrated_force"] = np.nan
    if "task_suite" not in df.columns:
        df["task_suite"] = "libero_object"
    if "phase" not in df.columns:
        df["phase"] = "FROZEN_REFERENCE"
    if "trial_id" not in df.columns:
        df["trial_id"] = [f"{path.stem}_{i}" for i in range(len(df))]

    df["task_id"] = df["task_id"].astype(int)
    df["friction_label"] = df["friction"].map(stable_float_mu)
    df["seed"] = df["seed_idx"].astype(int)
    df["object_id"] = df["object"].astype(str)
    df["neutral_language_instruction"] = df["instruction"].astype(str)
    df["full_success"] = df["full_task_success"].astype(int)
    df["lift_success"] = df["lift_success"].astype(int)
    df["place_success"] = df["place_success"].astype(int)
    df["source_file"] = str(path.relative_to(REPO_ROOT))
    keep = [
        "trial_id",
        "task_id",
        "object_id",
        "seed",
        "friction_label",
        "candidate_force_N",
        "neutral_language_instruction",
        "full_success",
        "lift_success",
        "place_success",
        "peak_force",
        "mean_grip_force",
        "integrated_force",
        "source_file",
        "phase",
    ]
    return df[keep]


def load_fulltask_positive() -> pd.DataFrame:
    paths = [
        FROZEN_B2R2 / "TASK0_SCAN.csv",
        FROZEN_B2R2 / "TASK1_FROZEN_REFERENCE_SCAN.csv",
        FROZEN_B2R2 / "TASK2_SCAN.csv",
        FROZEN_B2R2 / "TASK5_SCAN.csv",
        FROZEN_B2R2 / "TASK6_SCAN.csv",
    ]
    frames = [standardize_task_scan(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["task_id"].isin(POSITIVE_TASKS)]
    df = df[df["candidate_force_N"].isin(CANDIDATE_FORCES)]
    df["canonical_fstar_N"] = [
        CANONICAL_FSTAR[(int(t), stable_float_mu(mu))]
        for t, mu in zip(df["task_id"], df["friction_label"])
    ]
    df["label_source"] = "OBSERVED_FROZEN_FULLTASK_ROLLOUT"
    return df.reset_index(drop=True)


def copy_probe_timeseries() -> str:
    src = FROZEN_P3 / "PROBE_STEP_TRAJECTORIES.csv"
    dst = RESULT_DIR / "probe_timeseries_p3_A.csv"
    if not dst.exists():
        shutil.copyfile(src, dst)
    return str(dst.relative_to(RESULT_DIR))


def load_probe_features() -> pd.DataFrame:
    df = pd.read_csv(FROZEN_P3 / "PROBE_BELIEF_RESULTS.csv")
    df = df[df["probe"] == "A"].copy()
    df["friction_label"] = df["friction"].map(stable_float_mu)
    df["probe_seed"] = df["seed_idx"].astype(int)
    for col in PROBE_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df[["trial_id", "probe", "probe_seed", "friction_label"] + PROBE_FEATURES].reset_index(drop=True)


def attach_probe_features(full_df: pd.DataFrame, probe_df: pd.DataFrame, probe_path: str) -> pd.DataFrame:
    groups: Dict[float, pd.DataFrame] = {
        mu: g.sort_values("probe_seed").reset_index(drop=True) for mu, g in probe_df.groupby("friction_label")
    }
    rows = []
    for _, row in full_df.iterrows():
        mu = stable_float_mu(row["friction_label"])
        probes = groups[mu]
        probe_row = probes.iloc[int(row["seed"]) % len(probes)]
        out = row.to_dict()
        out["probe_id"] = "P3_A_4N_baseY_plus2mm_return"
        out["probe_trial_id"] = probe_row["trial_id"]
        out["probe_sequence_path"] = probe_path
        out["probe_evidence_origin"] = "P3_TASK1_PROBE_A_REUSED_BY_FRICTION_AND_SEED_MODULO"
        for col in PROBE_FEATURES:
            out[col] = float(probe_row[col])
        rows.append(out)
    df = pd.DataFrame(rows)
    df["target_force"] = 4.0
    df["measured_force"] = df["z_f_meas_mean"]
    df["squeeze_force"] = df["z_f_meas_mean"]
    df["tangential_force"] = df["z_ftan_peak"]
    df["force_imbalance"] = df["z_imbalance_peak"]
    df["force_derivative_proxy"] = df["z_marker_vel_abs_peak"]
    df["tactile_marker_motion"] = df["z_marker_tang_peak"]
    df["tangential_displacement"] = df["z_rel_xy_peak"]
    df["unloading"] = df["z_marker_unloading"]
    df["gripper_opening_proxy"] = df["z_hm_mean"]
    return df.reset_index(drop=True)


def write_parquet(df: pd.DataFrame, path: Path) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path)
        return True
    except Exception as exc:
        (RESULT_DIR / "logs" / "parquet_write_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return False


def make_dataset() -> pd.DataFrame:
    probe_path = copy_probe_timeseries()
    full = load_fulltask_positive()
    probe = load_probe_features()
    dataset = attach_probe_features(full, probe, probe_path)
    dataset["full_success"] = dataset["full_success"].astype(int)
    dataset["lift_success"] = dataset["lift_success"].astype(int)
    dataset["place_success"] = dataset["place_success"].astype(int)
    dataset["candidate_force_N"] = dataset["candidate_force_N"].astype(int)
    dataset["task_id"] = dataset["task_id"].astype(int)
    dataset["friction_label"] = dataset["friction_label"].astype(float)
    dataset.to_csv(RESULT_DIR / "M1_DATASET.csv", index=False)
    ok = write_parquet(dataset, RESULT_DIR / "M1_DATASET.parquet")
    if not ok:
        raise RuntimeError("Unable to write M1_DATASET.parquet")
    return dataset


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class FeatureSpec:
    variant: str
    task_context_type: str
    include_task_id: bool
    include_task_text: bool
    include_probe: bool
    include_gt_friction: bool
    numeric_cols: List[str]
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    task_ids: Optional[List[int]] = None


def build_feature_spec(variant: str, task_context_type: str, train_df: pd.DataFrame) -> FeatureSpec:
    include_task_id = variant in {"task_only", "task_probe", "gt_friction_task"} and task_context_type == "context_id"
    include_task_text = variant in {"task_only", "task_probe", "gt_friction_task"} and task_context_type == "context_text"
    include_probe = variant in {"probe_only", "task_probe"}
    include_gt_friction = variant == "gt_friction_task"
    numeric_cols = ["candidate_force_N"]
    if include_probe:
        numeric_cols += PROBE_FEATURES
    if include_gt_friction:
        numeric_cols += ["friction_label"]
    num = train_df[numeric_cols].astype(float).to_numpy()
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    std[std < 1e-6] = 1.0
    return FeatureSpec(
        variant=variant,
        task_context_type=task_context_type,
        include_task_id=include_task_id,
        include_task_text=include_task_text,
        include_probe=include_probe,
        include_gt_friction=include_gt_friction,
        numeric_cols=numeric_cols,
        mean=mean,
        std=std,
        task_ids=POSITIVE_TASKS,
    )


def featurize(df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    parts = []
    num = df[spec.numeric_cols].astype(float).to_numpy(dtype=np.float32)
    num = (num - spec.mean.astype(np.float32)) / spec.std.astype(np.float32)
    parts.append(num)
    if spec.include_task_id:
        one = np.zeros((len(df), len(spec.task_ids)), dtype=np.float32)
        task_to_i = {t: i for i, t in enumerate(spec.task_ids)}
        for r, t in enumerate(df["task_id"].astype(int)):
            if int(t) in task_to_i:
                one[r, task_to_i[int(t)]] = 1.0
        parts.append(one)
    if spec.include_task_text:
        text = np.vstack(
            [task_text_embedding(str(x), TEXT_DIM) for x in df["neutral_language_instruction"]]
        ).astype(np.float32)
        parts.append(text)
    return np.concatenate(parts, axis=1).astype(np.float32)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    total = len(y)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(len(y)),
        "auroc": auc_score(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece_score(y, p),
        "binary_accuracy": float(np.mean((p >= 0.5) == y)),
        "positive_rate": float(np.mean(y)),
    }


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: str,
    task_context_type: str,
    split_name: str,
    fold_name: str,
) -> Tuple[MLP, FeatureSpec, Dict[str, float], np.ndarray, float, float]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    spec = build_feature_spec(variant, task_context_type, train_df)
    x_train = torch.tensor(featurize(train_df, spec), dtype=torch.float32)
    y_train_np = train_df["full_success"].astype(float).to_numpy(dtype=np.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    x_val = torch.tensor(featurize(val_df, spec), dtype=torch.float32)
    y_val = torch.tensor(val_df["full_success"].astype(float).to_numpy(dtype=np.float32), dtype=torch.float32)
    x_test_np = featurize(test_df, spec)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)

    model = MLP(x_train.shape[1])
    pos = float(y_train_np.sum())
    neg = float(len(y_train_np) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    start = time.time()
    for epoch in range(180):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch > 25:
            break
    train_time = time.time() - start
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        p_test = torch.sigmoid(model(x_test)).cpu().numpy()
        latency = (time.perf_counter() - t0) / max(len(x_test_np), 1)
    metrics = binary_metrics(test_df["full_success"].astype(int).to_numpy(), p_test)
    metrics.update(
        {
            "best_epoch": int(best_epoch),
            "train_seconds": float(train_time),
            "inference_latency_ms_per_candidate": float(latency * 1000.0),
            "parameter_count": parameter_count(model),
            "input_dim": int(x_train.shape[1]),
        }
    )
    ckpt = {
        "model_state_dict": model.state_dict(),
        "feature_spec": {
            "variant": spec.variant,
            "task_context_type": spec.task_context_type,
            "numeric_cols": spec.numeric_cols,
            "mean": spec.mean.tolist(),
            "std": spec.std.tolist(),
            "task_ids": spec.task_ids,
            "text_dim": TEXT_DIM,
            "hidden": list(HIDDEN),
        },
        "metrics": metrics,
    }
    torch.save(ckpt, RESULT_DIR / "checkpoints" / f"{split_name}_{fold_name}_{variant}_{task_context_type}.pt")
    return model, spec, metrics, p_test, train_time, latency


def split_iid(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[int]]]:
    seed_mod = df["seed"] % 5
    test = df[seed_mod == 0].copy()
    val = df[seed_mod == 1].copy()
    train = df[(seed_mod != 0) & (seed_mod != 1)].copy()
    manifest = {
        "train_seed_mod": [2, 3, 4],
        "val_seed_mod": [1],
        "test_seed_mod": [0],
    }
    return train, val, test, manifest


def split_loto(df: pd.DataFrame, heldout_task: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = df[df["task_id"] == heldout_task].copy()
    rest = df[df["task_id"] != heldout_task].copy()
    val = rest[rest["seed"] % 5 == 1].copy()
    train = rest[rest["seed"] % 5 != 1].copy()
    return train, val, test


def split_ood_friction(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pool = df[df["friction_label"].isin([0.2, 1.0])].copy()
    val = train_pool[train_pool["seed"] % 5 == 1].copy()
    train = train_pool[train_pool["seed"] % 5 != 1].copy()
    test = df[df["friction_label"] == 0.5].copy()
    return train, val, test


def make_candidate_rows(episodes: pd.DataFrame, force: int) -> pd.DataFrame:
    rows = episodes.copy()
    rows["candidate_force_N"] = int(force)
    return rows


def episode_frame(test_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "task_id",
        "object_id",
        "seed",
        "friction_label",
        "neutral_language_instruction",
        "canonical_fstar_N",
        "probe_id",
        "probe_trial_id",
        "probe_sequence_path",
        "probe_evidence_origin",
        "target_force",
        "measured_force",
        "squeeze_force",
        "tangential_force",
        "force_imbalance",
        "force_derivative_proxy",
        "tactile_marker_motion",
        "tangential_displacement",
        "unloading",
        "gripper_opening_proxy",
    ] + PROBE_FEATURES
    return test_df[cols].drop_duplicates(["task_id", "seed", "friction_label"]).reset_index(drop=True)


def decision_eval(
    model: MLP,
    spec: FeatureSpec,
    test_df: pd.DataFrame,
    split_name: str,
    fold_name: str,
    variant: str,
    task_context_type: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    episodes = episode_frame(test_df)
    lookup = {
        (int(r.task_id), int(r.seed), stable_float_mu(r.friction_label), int(r.candidate_force_N)): int(r.full_success)
        for r in test_df.itertuples()
    }
    pred_rows = []
    decisions = []
    model.eval()
    for _, ep in episodes.iterrows():
        probs = []
        for force in CANDIDATE_FORCES:
            cand = make_candidate_rows(pd.DataFrame([ep.to_dict()]), force)
            with torch.no_grad():
                x = torch.tensor(featurize(cand, spec), dtype=torch.float32)
                prob = float(torch.sigmoid(model(x))[0].item())
            probs.append(prob)
            pred_rows.append(
                {
                    "split": split_name,
                    "fold": fold_name,
                    "variant": variant,
                    "task_context_type": task_context_type,
                    "task_id": int(ep["task_id"]),
                    "seed": int(ep["seed"]),
                    "friction_label": float(ep["friction_label"]),
                    "candidate_force_N": force,
                    "pred_success": prob,
                    "canonical_fstar_N": int(ep["canonical_fstar_N"]),
                }
            )
        selected = 8
        no_conf = True
        for force, prob in zip(CANDIDATE_FORCES, probs):
            if prob >= TAU:
                selected = force
                no_conf = False
                break
        key = (int(ep["task_id"]), int(ep["seed"]), stable_float_mu(ep["friction_label"]), int(selected))
        if key in lookup:
            success = lookup[key]
            eval_source = "OBSERVED_SELECTED_FORCE"
        else:
            success = int(selected >= int(ep["canonical_fstar_N"]))
            eval_source = "CANONICAL_FSTAR_COUNTERFACTUAL"
        fstar = int(ep["canonical_fstar_N"])
        decisions.append(
            {
                "split": split_name,
                "fold": fold_name,
                "variant": variant,
                "task_context_type": task_context_type,
                "task_id": int(ep["task_id"]),
                "object_id": ep["object_id"],
                "seed": int(ep["seed"]),
                "friction_label": float(ep["friction_label"]),
                "selected_force_N": int(selected),
                "canonical_fstar_N": fstar,
                "selected_success": int(success),
                "under_force": int(selected < fstar),
                "over_force": int(selected > fstar),
                "no_confident_sufficient_force": int(no_conf),
                "eval_source": eval_source,
                "p3": probs[0],
                "p4": probs[1],
                "p5": probs[2],
                "p6": probs[3],
                "p8": probs[4],
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    dec_df = pd.DataFrame(decisions)
    metrics = {
        "decision_episodes": int(len(dec_df)),
        "full_task_sr": float(dec_df["selected_success"].mean()) if len(dec_df) else float("nan"),
        "mean_selected_force_N": float(dec_df["selected_force_N"].mean()) if len(dec_df) else float("nan"),
        "under_force_rate": float(dec_df["under_force"].mean()) if len(dec_df) else float("nan"),
        "over_force_rate": float(dec_df["over_force"].mean()) if len(dec_df) else float("nan"),
        "no_confident_rate": float(dec_df["no_confident_sufficient_force"].mean()) if len(dec_df) else float("nan"),
        "gt_minforce_mean_force_N": float(dec_df["canonical_fstar_N"].mean()) if len(dec_df) else float("nan"),
        "observed_selected_fraction": float((dec_df["eval_source"] == "OBSERVED_SELECTED_FORCE").mean())
        if len(dec_df)
        else float("nan"),
    }
    return dec_df, metrics


def monotonicity_audit(pred_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["split", "fold", "variant", "task_context_type", "task_id", "seed", "friction_label"]
    for key, g in pred_rows.groupby(keys):
        g = g.sort_values("candidate_force_N")
        probs = g["pred_success"].to_numpy()
        forces = g["candidate_force_N"].to_numpy()
        violations = []
        for i in range(len(probs) - 1):
            if probs[i + 1] + 1e-6 < probs[i]:
                violations.append(f"{int(forces[i])}>{int(forces[i+1])}:{probs[i]:.3f}->{probs[i+1]:.3f}")
        rows.append(
            dict(
                zip(keys, key),
                n_force_pairs=len(probs) - 1,
                n_violations=len(violations),
                has_violation=int(bool(violations)),
                violation_detail="; ".join(violations),
            )
        )
    return pd.DataFrame(rows)


def run_suite(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    all_result_rows = []
    all_decisions = []
    all_preds = []
    split_manifest = {"candidate_forces_N": CANDIDATE_FORCES, "tau": TAU}

    variants_iid = [
        ("task_only", "context_id"),
        ("task_only", "context_text"),
        ("probe_only", "none"),
        ("task_probe", "context_id"),
        ("task_probe", "context_text"),
        ("gt_friction_task", "context_id"),
        ("gt_friction_task", "context_text"),
    ]
    train, val, test, manifest_iid = split_iid(dataset)
    split_manifest["split_A_iid_seed"] = manifest_iid
    for variant, ctx in variants_iid:
        model, spec, metrics, p_test, _, _ = train_model(train, val, test, variant, ctx, "IID", "all_tasks")
        dec, dmetrics = decision_eval(model, spec, test, "IID", "all_tasks", variant, ctx)
        row = {"split": "IID", "fold": "all_tasks", "variant": variant, "task_context_type": ctx}
        row.update(metrics)
        row.update(dmetrics)
        all_result_rows.append(row)
        all_decisions.append(dec)
        pred_rows = []
        _, dmetrics_again = dec, dmetrics
        pred_ep = episode_frame(test)
        for _, ep in pred_ep.iterrows():
            for force in CANDIDATE_FORCES:
                cand = make_candidate_rows(pd.DataFrame([ep.to_dict()]), force)
                with torch.no_grad():
                    prob = float(torch.sigmoid(model(torch.tensor(featurize(cand, spec), dtype=torch.float32)))[0])
                pred_rows.append(
                    {
                        "split": "IID",
                        "fold": "all_tasks",
                        "variant": variant,
                        "task_context_type": ctx,
                        "task_id": int(ep["task_id"]),
                        "seed": int(ep["seed"]),
                        "friction_label": float(ep["friction_label"]),
                        "candidate_force_N": force,
                        "pred_success": prob,
                    }
                )
        all_preds.append(pd.DataFrame(pred_rows))

    variants_loto = [
        ("task_only", "context_text"),
        ("probe_only", "none"),
        ("task_probe", "context_text"),
        ("gt_friction_task", "context_text"),
    ]
    split_manifest["split_B_loto"] = {}
    for heldout in POSITIVE_TASKS:
        train, val, test = split_loto(dataset, heldout)
        split_manifest["split_B_loto"][str(heldout)] = {
            "train_tasks": sorted(train["task_id"].unique().astype(int).tolist()),
            "val_seed_mod": [1],
            "test_tasks": [heldout],
        }
        for variant, ctx in variants_loto:
            model, spec, metrics, p_test, _, _ = train_model(
                train, val, test, variant, ctx, "LOTO", f"heldout_task{heldout}"
            )
            dec, dmetrics = decision_eval(model, spec, test, "LOTO", f"heldout_task{heldout}", variant, ctx)
            row = {
                "split": "LOTO",
                "fold": f"heldout_task{heldout}",
                "heldout_task": heldout,
                "variant": variant,
                "task_context_type": ctx,
            }
            row.update(metrics)
            row.update(dmetrics)
            all_result_rows.append(row)
            all_decisions.append(dec)
            pred_rows = []
            pred_ep = episode_frame(test)
            for _, ep in pred_ep.iterrows():
                for force in CANDIDATE_FORCES:
                    cand = make_candidate_rows(pd.DataFrame([ep.to_dict()]), force)
                    with torch.no_grad():
                        prob = float(torch.sigmoid(model(torch.tensor(featurize(cand, spec), dtype=torch.float32)))[0])
                    pred_rows.append(
                        {
                            "split": "LOTO",
                            "fold": f"heldout_task{heldout}",
                            "variant": variant,
                            "task_context_type": ctx,
                            "task_id": int(ep["task_id"]),
                            "seed": int(ep["seed"]),
                            "friction_label": float(ep["friction_label"]),
                            "candidate_force_N": force,
                            "pred_success": prob,
                        }
                    )
            all_preds.append(pd.DataFrame(pred_rows))

    train, val, test = split_ood_friction(dataset)
    split_manifest["split_C_ood_friction"] = {
        "train_friction": [0.2, 1.0],
        "test_friction": [0.5],
        "val_seed_mod": [1],
    }
    for variant, ctx in variants_loto:
        model, spec, metrics, p_test, _, _ = train_model(train, val, test, variant, ctx, "OOD_FRICTION", "mu05")
        dec, dmetrics = decision_eval(model, spec, test, "OOD_FRICTION", "mu05", variant, ctx)
        row = {"split": "OOD_FRICTION", "fold": "mu05", "variant": variant, "task_context_type": ctx}
        row.update(metrics)
        row.update(dmetrics)
        all_result_rows.append(row)
        all_decisions.append(dec)
        pred_rows = []
        pred_ep = episode_frame(test)
        for _, ep in pred_ep.iterrows():
            for force in CANDIDATE_FORCES:
                cand = make_candidate_rows(pd.DataFrame([ep.to_dict()]), force)
                with torch.no_grad():
                    prob = float(torch.sigmoid(model(torch.tensor(featurize(cand, spec), dtype=torch.float32)))[0])
                pred_rows.append(
                    {
                        "split": "OOD_FRICTION",
                        "fold": "mu05",
                        "variant": variant,
                        "task_context_type": ctx,
                        "task_id": int(ep["task_id"]),
                        "seed": int(ep["seed"]),
                        "friction_label": float(ep["friction_label"]),
                        "candidate_force_N": force,
                        "pred_success": prob,
                    }
                )
        all_preds.append(pd.DataFrame(pred_rows))

    (RESULT_DIR / "SPLIT_MANIFEST.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    results = pd.DataFrame(all_result_rows)
    decisions = pd.concat(all_decisions, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    mono = monotonicity_audit(preds)
    return results, decisions, preds, mono


def aggregate_baselines() -> Tuple[pd.DataFrame, pd.DataFrame]:
    b4 = pd.read_csv(FROZEN_B4 / "FORTE_MAIN_RESULTS.csv")
    b4 = b4[b4["task_id"].isin(POSITIVE_TASKS)].copy()
    baseline_rows = []
    for method, g in b4.groupby("method"):
        n = g["n"].astype(float)
        if method == "fixed_low":
            force_col = "mean_final_target_N"
        elif method == "fixed_robust":
            force_col = "mean_fixed_robust_N"
        elif method == "gt_minforce":
            force_col = "mean_fstar_N"
        elif method == "forte_gt_reactive":
            force_col = "mean_final_target_N"
        else:
            force_col = "mean_force_N"
        baseline_rows.append(
            {
                "method": method,
                "label": str(g["method_label"].iloc[0]),
                "weighted_full_sr": float(np.average(g["full_sr"], weights=n)),
                "weighted_mean_force_N": float(np.average(g[force_col], weights=n)),
                "mean_force_definition": force_col,
                "cells": int(len(g)),
                "episodes": int(g["n"].sum()),
                "source": "B4_FORTE_MAIN_RESULTS",
            }
        )
    b5 = pd.read_csv(FROZEN_B5 / "TABERO_NEUTRAL_MAIN_RESULTS.csv")
    b5 = b5[b5["task"].isin(POSITIVE_TASKS)].copy()
    n = b5["n"].astype(float)
    baseline_rows.append(
        {
            "method": "tabero_neutral",
            "label": "Tabero Neutral",
            "weighted_full_sr": float(np.average(b5["full_sr"], weights=n)),
            "weighted_mean_force_N": float(np.average(b5["mean_measured_force_N"], weights=n)),
            "mean_force_definition": "mean_measured_force_N",
            "cells": int(len(b5)),
            "episodes": int(b5["n"].sum()),
            "source": "B5_TABERO_NEUTRAL_MAIN_RESULTS",
        }
    )
    return pd.DataFrame(baseline_rows), b5


def write_result_tables(results: pd.DataFrame, decisions: pd.DataFrame, mono: pd.DataFrame, baselines: pd.DataFrame) -> None:
    results.to_csv(RESULT_DIR / "CALIBRATION_METRICS.csv", index=False)
    mono.to_csv(RESULT_DIR / "MONOTONICITY_AUDIT.csv", index=False)

    def subset(variant: str) -> pd.DataFrame:
        return results[results["variant"] == variant].copy()

    subset("task_only").to_csv(RESULT_DIR / "TASK_ONLY_RESULTS.csv", index=False)
    subset("probe_only").to_csv(RESULT_DIR / "PROBE_ONLY_RESULTS.csv", index=False)
    subset("task_probe").to_csv(RESULT_DIR / "TASK_PROBE_RESULTS.csv", index=False)
    subset("gt_friction_task").to_csv(RESULT_DIR / "GT_FRICTION_DIAGNOSTIC.csv", index=False)
    results[results["split"] == "IID"].to_csv(RESULT_DIR / "IID_OFFLINE_RESULTS.csv", index=False)
    results[results["split"] == "LOTO"].to_csv(RESULT_DIR / "LOTO_OFFLINE_RESULTS.csv", index=False)
    results[results["split"] == "OOD_FRICTION"].to_csv(RESULT_DIR / "OOD_FRICTION_RESULTS.csv", index=False)

    main_dec = decisions[
        (decisions["split"] == "LOTO")
        & (decisions["variant"].isin(["task_only", "task_probe"]))
        & (decisions["task_context_type"].isin(["context_text"]))
    ].copy()
    main_dec["online_status"] = "ONLINE_NOT_RUN"
    main_dec["online_blocker"] = "offline LOTO calibration/generalization insufficient; per-task probe evidence incomplete"
    main_dec["row_type"] = "OFFLINE_COUNTERFACTUAL_DECISION_PROXY"
    main_dec.to_csv(RESULT_DIR / "ONLINE_MAIN_RESULTS.csv", index=False)
    task_summary = (
        main_dec.groupby(["variant", "task_id", "friction_label"])
        .agg(
            n=("selected_success", "size"),
            full_sr=("selected_success", "mean"),
            mean_force_N=("selected_force_N", "mean"),
            under_force_rate=("under_force", "mean"),
            over_force_rate=("over_force", "mean"),
        )
        .reset_index()
    )
    task_summary["status"] = "OFFLINE_COUNTERFACTUAL_PROXY_ONLINE_NOT_RUN"
    task_summary.to_csv(RESULT_DIR / "ONLINE_TASK_SUMMARY.csv", index=False)
    baselines.to_csv(RESULT_DIR / "BASELINES_REFERENCE.csv", index=False)


def fmt_float(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{float(x):.3f}"


def select_row(results: pd.DataFrame, split: str, variant: str, ctx: str, fold: Optional[str] = None) -> pd.Series:
    q = (results["split"] == split) & (results["variant"] == variant) & (results["task_context_type"] == ctx)
    if fold is not None:
        q &= results["fold"] == fold
    m = results[q]
    if len(m) == 0:
        raise KeyError((split, variant, ctx, fold))
    return m.iloc[0]


def write_docs(
    dataset: pd.DataFrame,
    results: pd.DataFrame,
    decisions: pd.DataFrame,
    mono: pd.DataFrame,
    baselines: pd.DataFrame,
) -> Dict[str, object]:
    iid_task = select_row(results, "IID", "task_only", "context_id")
    iid_probe = select_row(results, "IID", "probe_only", "none")
    iid_main = select_row(results, "IID", "task_probe", "context_id")
    iid_gt = select_row(results, "IID", "gt_friction_task", "context_id")
    loto_task = results[(results["split"] == "LOTO") & (results["variant"] == "task_only") & (results["task_context_type"] == "context_text")]
    loto_main = results[(results["split"] == "LOTO") & (results["variant"] == "task_probe") & (results["task_context_type"] == "context_text")]
    loto_probe = results[(results["split"] == "LOTO") & (results["variant"] == "probe_only")]
    loto_gt = results[(results["split"] == "LOTO") & (results["variant"] == "gt_friction_task")]
    main_param_count = int(results[
        (results["split"] == "IID")
        & (results["variant"] == "task_probe")
        & (results["task_context_type"] == "context_text")
    ]["parameter_count"].iloc[0])

    fixed_robust = baselines[baselines["method"] == "fixed_robust"].iloc[0]
    gt_min = baselines[baselines["method"] == "gt_minforce"].iloc[0]
    fixed_low = baselines[baselines["method"] == "fixed_low"].iloc[0]
    forte = baselines[baselines["method"] == "forte_gt_reactive"].iloc[0]
    tabero = baselines[baselines["method"] == "tabero_neutral"].iloc[0]

    loto_table = []
    for task in POSITIVE_TASKS:
        trow = loto_task[loto_task["heldout_task"] == task].iloc[0]
        mrow = loto_main[loto_main["heldout_task"] == task].iloc[0]
        oracle_force = float(
            dataset[dataset["task_id"] == task][["seed", "friction_label", "canonical_fstar_N"]]
            .drop_duplicates()["canonical_fstar_N"]
            .mean()
        )
        loto_table.append(
            {
                "heldout_task": task,
                "task_only_sr": float(trow["full_task_sr"]),
                "task_only_force_N": float(trow["mean_selected_force_N"]),
                "task_probe_sr": float(mrow["full_task_sr"]),
                "task_probe_force_N": float(mrow["mean_selected_force_N"]),
                "oracle_mean_force_N": oracle_force,
            }
        )
    loto_table_df = pd.DataFrame(loto_table)

    task2_dec = decisions[
        (decisions["split"] == "LOTO")
        & (decisions["fold"] == "heldout_task2")
        & (decisions["variant"] == "task_probe")
    ].copy()
    task2_summary = (
        task2_dec.groupby("friction_label")
        .agg(sr=("selected_success", "mean"), selected_force_N=("selected_force_N", "mean"), under=("under_force", "mean"), n=("selected_success", "size"))
        .reset_index()
    )

    probe_improves = float(loto_main["full_task_sr"].mean()) > float(loto_task["full_task_sr"].mean()) + 0.02
    task_probe_loto_sr = float(loto_main["full_task_sr"].mean())
    task_only_loto_sr = float(loto_task["full_task_sr"].mean())
    task_probe_loto_force = float(loto_main["mean_selected_force_N"].mean())

    if float(iid_main["full_task_sr"]) >= 0.95 and task_probe_loto_sr < 0.85:
        status = "M1_IID_WORKS_BUT_CROSS_TASK_FAILS"
    elif probe_improves:
        status = "M1_PROBE_EVIDENCE_IMPROVES_CROSS_TASK_GENERALIZATION"
    elif float(loto_main["ece"].mean()) > 0.15:
        status = "M1_MODEL_CALIBRATION_WEAK"
    else:
        status = "M1_TASK_PRIOR_DOMINATES_PROBE"

    main_failure = (
        "leave-one-task-out cannot infer task2 low-friction 8N requirement from other tasks; "
        "frozen probe evidence is reused from P3/task1 rather than collected per object/task"
    )

    verdict = {
        "status": status,
        "method_change": "M1_FORCE_SUFFICIENCY_MODEL",
        "benchmark_tasks": POSITIVE_TASKS,
        "candidate_forces_N": CANDIDATE_FORCES,
        "tau": TAU,
        "probe_frozen": True,
        "probe_type": "base-Y +2mm return",
        "model_type": "2-hidden-layer MLP sigmoid force-sufficiency classifier",
        "parameter_count": main_param_count,
        "task_context_type": "Context-ID for IID; Context-Text frozen hash embedding for LOTO",
        "probe_evidence_type": "P3-A raw probe summary features; no GT friction input",
        "iid_full_sr": float(iid_main["full_task_sr"]),
        "iid_mean_force_N": float(iid_main["mean_selected_force_N"]),
        "fixed_robust_full_sr": float(fixed_robust["weighted_full_sr"]),
        "fixed_robust_mean_force_N": float(fixed_robust["weighted_mean_force_N"]),
        "gt_minforce_mean_force_N": float(gt_min["weighted_mean_force_N"]),
        "loto_results": loto_table,
        "loto_mean_sr": task_probe_loto_sr,
        "loto_mean_force_N": task_probe_loto_force,
        "task_only_loto_sr": task_only_loto_sr,
        "task_probe_loto_sr": task_probe_loto_sr,
        "probe_improves_generalization": probe_improves,
        "under_force_rate": float(loto_main["under_force_rate"].mean()),
        "over_force_rate": float(loto_main["over_force_rate"].mean()),
        "task2_low_mu_sr": float(task2_summary[task2_summary["friction_label"] == 0.2]["sr"].iloc[0]),
        "main_failure_mode": main_failure,
        "primary_evidence": "offline observed force-sufficiency labels from frozen B2-R2/E2E2 scans; LOTO decision metrics use canonical-F* counterfactuals when selected force cell is unobserved",
        "limitations": [
            "No new online full downstream evaluation was run.",
            "Probe evidence is not available for every positive task/object; P3-A task1 probe features are reused by friction/seed as deployable evidence proxy.",
            "Candidate 8N labels are observed for task2 but mostly missing for tasks 0/1/5/6, so some decision outcomes are canonical counterfactuals.",
            "Context-Text uses a fixed hash embedding, not a pretrained semantic encoder.",
        ],
    }
    (RESULT_DIR / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    env = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(REPO_ROOT),
        "result_dir": str(RESULT_DIR),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "frozen_inputs": {
            "B2R2": str(FROZEN_B2R2.relative_to(REPO_ROOT)),
            "P3": str(FROZEN_P3.relative_to(REPO_ROOT)),
            "B4": str(FROZEN_B4.relative_to(REPO_ROOT)),
            "B5": str(FROZEN_B5.relative_to(REPO_ROOT)),
        },
    }
    (RESULT_DIR / "ENV_PROVENANCE.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    dataset_spec = f"""# M1 Dataset Spec

Each row is `(task/context, probe evidence, candidate_force_N) -> full_success`.

Rows: {len(dataset)}

Tasks: {POSITIVE_TASKS}

Candidate forces: {CANDIDATE_FORCES} N.

Labels: observed frozen full-task rollouts from B2-R2 and E2E2 task1 reference scans. The model is not trained on canonical F* directly.

Probe evidence: frozen P3 Probe A, 4N contact, base-Y +2mm, return, stabilize. Raw deployable summary features are attached by matching friction label and seed modulo because per-task probe telemetry is not present for all positive tasks. `friction_label` and `canonical_fstar_N` are retained for evaluation and analysis only and are excluded from deployable M1 variants.

Unavailable in frozen probe artifact: left/right fingertip force xyz and true gripper opening. Preserved proxies: measured force, squeeze force, tangential force, force imbalance, force derivative proxy, tactile marker motion, tangential displacement, unloading, and heightmap/gripper-opening proxy.
"""
    (RESULT_DIR / "DATASET_SPEC.md").write_text(dataset_spec, encoding="utf-8")

    model_spec = f"""# M1 Model Spec

Main deployable model: task-conditioned force-sufficiency classifier.

Output: `p_theta(y_full_success = 1 | task/context, probe evidence, F)`.

Decision rule: evaluate forces `{CANDIDATE_FORCES}` and choose the lowest force with probability >= {TAU}. If none qualify, choose 8N and mark `NO_CONFIDENT_SUFFICIENT_FORCE`.

Architecture: MLP with hidden widths {HIDDEN}, ReLU activations, sigmoid output.

Task representations:

- Context-ID: fixed one-hot over positive task IDs; used for IID only.
- Context-Text: deterministic frozen hashed bag-of-words embedding of the neutral instruction; used for LOTO.

Probe representation: normalized raw P3-A probe summary features. No GT friction, simulator material, GT F*, future success, future slip, or privileged object state is included in deployable variants.

Diagnostic only: A3 uses GT friction + task and is not deployable.

Monotonicity: audited after prediction; not architecturally enforced in round 1.
"""
    (RESULT_DIR / "MODEL_SPEC.md").write_text(model_spec, encoding="utf-8")

    training_config = {
        "seed": SEED,
        "candidate_forces_N": CANDIDATE_FORCES,
        "tau": TAU,
        "optimizer": "AdamW",
        "lr": 0.01,
        "weight_decay": 1e-4,
        "loss": "BCEWithLogitsLoss with train-split pos_weight",
        "early_stopping": "validation BCE, patience 25 epochs",
        "normalization": "numeric means/std fit on train split only",
        "hidden": list(HIDDEN),
        "text_dim": TEXT_DIM,
    }
    (RESULT_DIR / "TRAINING_CONFIG.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")

    ablation_lines = [
        "# Ablation Table",
        "",
        "| Split | Variant | Context | SR | Mean F | Brier | ECE | Under | Over |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in results.iterrows():
        if r["split"] in {"IID", "OOD_FRICTION"} or (r["split"] == "LOTO" and r.get("heldout_task", None) in POSITIVE_TASKS):
            ablation_lines.append(
                f"| {r['split']} {r['fold']} | {r['variant']} | {r['task_context_type']} | "
                f"{fmt_float(r['full_task_sr'])} | {fmt_float(r['mean_selected_force_N'])} | "
                f"{fmt_float(r['brier'])} | {fmt_float(r['ece'])} | "
                f"{fmt_float(r['under_force_rate'])} | {fmt_float(r['over_force_rate'])} |"
            )
    (RESULT_DIR / "ABLATIon_TABLE.tmp").write_text("\n".join(ablation_lines) + "\n", encoding="utf-8")
    os.replace(RESULT_DIR / "ABLATIon_TABLE.tmp", RESULT_DIR / "ABLATION_TABLE.md")

    main_lines = [
        "# Main Method Table",
        "",
        "| Method | SR | Mean force N | Episodes | Source |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| Fixed Low | {fmt_float(fixed_low['weighted_full_sr'])} | {fmt_float(fixed_low['weighted_mean_force_N'])} | {int(fixed_low['episodes'])} | B4 |",
        f"| Fixed Robust | {fmt_float(fixed_robust['weighted_full_sr'])} | {fmt_float(fixed_robust['weighted_mean_force_N'])} | {int(fixed_robust['episodes'])} | B4 |",
        f"| FORTE-GT | {fmt_float(forte['weighted_full_sr'])} | {fmt_float(forte['weighted_mean_force_N'])} | {int(forte['episodes'])} | B4 |",
        f"| Tabero Neutral | {fmt_float(tabero['weighted_full_sr'])} | {fmt_float(tabero['weighted_mean_force_N'])} | {int(tabero['episodes'])} | B5 |",
        f"| M1 Task-Only IID | {fmt_float(iid_task['full_task_sr'])} | {fmt_float(iid_task['mean_selected_force_N'])} | {int(iid_task['decision_episodes'])} | offline |",
        f"| M1 Task+Probe IID | {fmt_float(iid_main['full_task_sr'])} | {fmt_float(iid_main['mean_selected_force_N'])} | {int(iid_main['decision_episodes'])} | offline |",
        f"| M1 Task-Only LOTO | {fmt_float(task_only_loto_sr)} | {fmt_float(float(loto_task['mean_selected_force_N'].mean()))} | {int(loto_task['decision_episodes'].sum())} | offline |",
        f"| M1 Task+Probe LOTO | {fmt_float(task_probe_loto_sr)} | {fmt_float(task_probe_loto_force)} | {int(loto_main['decision_episodes'].sum())} | offline |",
        f"| GT-MinForce | {fmt_float(gt_min['weighted_full_sr'])} | {fmt_float(gt_min['weighted_mean_force_N'])} | {int(gt_min['episodes'])} | B4 |",
    ]
    (RESULT_DIR / "MAIN_METHOD_TABLE.md").write_text("\n".join(main_lines) + "\n", encoding="utf-8")

    gen_lines = [
        "# Generalization Table",
        "",
        "| Held-out task | Task Only SR+F | Task+Probe SR+F | Oracle mean F |",
        "| ---: | --- | --- | ---: |",
    ]
    for r in loto_table:
        gen_lines.append(
            f"| {r['heldout_task']} | {fmt_float(r['task_only_sr'])} / {fmt_float(r['task_only_force_N'])} | "
            f"{fmt_float(r['task_probe_sr'])} / {fmt_float(r['task_probe_force_N'])} | {fmt_float(r['oracle_mean_force_N'])} |"
        )
    (RESULT_DIR / "GENERALIZATION_TABLE.md").write_text("\n".join(gen_lines) + "\n", encoding="utf-8")

    failure = f"""# Failure Analysis

Primary failure mode: {main_failure}.

M1 learns an IID success curve over candidate forces when the task ID is present. In leave-one-task-out, probe evidence transfers the low/mid/high friction ordering, but it does not transfer the held-out task-specific force mapping reliably. Task2 is the hardest case: the model has no training task whose low-friction condition requires 8N while mid/high require 3N.

The probe evidence is deployable in form but incomplete in coverage: frozen P3-A probe telemetry exists as a task1-style probe artifact, not as per-task/per-object probe rollouts across tasks 0/2/5/6. Therefore the result is an offline M1 sufficiency-model test, not a completed online physical-force method.

No monotonic architecture was used. Violations are audited in `MONOTONICITY_AUDIT.csv`.
"""
    (RESULT_DIR / "FAILURE_ANALYSIS.md").write_text(failure, encoding="utf-8")

    readme = f"""# M1 Force-Sufficiency Model

Status: `{status}`.

This directory contains the first OURS-style M1 experiment: a small task-conditioned force-sufficiency model trained on frozen full-task force labels.

The model answers: for this task/context, this probe evidence, and candidate force F, is full-task success likely?

No benchmark, D2, Tabero core, Tabero-VTLA checkpoint, FORTE optimization, probe policy, or canonical F* was modified.

Key artifacts:

- `M1_DATASET.parquet` and `M1_DATASET.csv`
- `TASK_ONLY_RESULTS.csv`, `PROBE_ONLY_RESULTS.csv`, `TASK_PROBE_RESULTS.csv`
- `IID_OFFLINE_RESULTS.csv`, `LOTO_OFFLINE_RESULTS.csv`, `OOD_FRICTION_RESULTS.csv`
- `MAIN_METHOD_TABLE.md`, `GENERALIZATION_TABLE.md`, `FAILURE_ANALYSIS.md`
- `FINAL_VERDICT.json`

Online full downstream was not run. `ONLINE_MAIN_RESULTS.csv` and `ONLINE_TASK_SUMMARY.csv` contain the offline counterfactual decision proxy and are explicitly marked as such.
"""
    (RESULT_DIR / "README.md").write_text(readme, encoding="utf-8")

    return verdict


def plot_reliability(results: pd.DataFrame, decisions: pd.DataFrame) -> None:
    # Decision-level selected success vs force is the more useful quick visual here.
    main = decisions[(decisions["split"] == "LOTO") & (decisions["variant"] == "task_probe")].copy()
    plt.figure(figsize=(7, 4))
    for task, g in main.groupby("task_id"):
        x = sorted(g["friction_label"].unique())
        y = [g[g["friction_label"] == mu]["selected_force_N"].mean() for mu in x]
        plt.plot(x, y, marker="o", label=f"task {task}")
    plt.xlabel("friction label (eval only)")
    plt.ylabel("selected force N")
    plt.title("M1 Task+Probe LOTO selected force")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "plots" / "loto_selected_force_by_task.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    loto = results[(results["split"] == "LOTO") & (results["variant"].isin(["task_only", "task_probe"]))]
    for variant, g in loto.groupby("variant"):
        plt.bar(
            [f"{int(x)}-{variant.replace('_', '+')}" for x in g["heldout_task"]],
            g["full_task_sr"],
            label=variant,
        )
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("offline decision SR")
    plt.title("LOTO decision success by held-out task")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "plots" / "loto_sr_by_task.png", dpi=160)
    plt.close()


def main() -> None:
    ensure_dirs()
    dataset = make_dataset()
    results, decisions, preds, mono = run_suite(dataset)
    baselines, b5 = aggregate_baselines()
    write_result_tables(results, decisions, mono, baselines)
    verdict = write_docs(dataset, results, decisions, mono, baselines)
    plot_reliability(results, decisions)
    print(json.dumps({"result_dir": str(RESULT_DIR), "status": verdict["status"]}, indent=2))


if __name__ == "__main__":
    main()
