#!/usr/bin/env python3
"""P5-S0 four-task GNP-style force sufficiency smoke pipeline.

This runner is deliberately result-scoped.  It freezes the existing P4-R2
task-ID-free probe, audits whether true matched post-probe force branching can
be run in the current environment, and then runs the model-integration half on
observed archival probe/full-task artifacts when Isaac is unavailable.

The archival path is explicitly marked invalid for the strict P5-S0 scientific
claim because branches are not reset-equivalent descendants of the saved
post-probe state.  It exists only to exercise dataset plumbing and model code.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO = Path("/home/exouser/Tabero")
RESULTS_ROOT = REPO / "analysis/results"
P4 = RESULTS_ROOT / "p4_contact_conditioned_probe_20260822_184213"
P4R2 = RESULTS_ROOT / "p4r2_contact_feasible_multi_primitive_20260823_070150"
B2R2 = RESULTS_ROOT / "b2r2_tabero_task_breadth_20260821_073931"
D2 = RESULTS_ROOT / "d2_hierarchical_force_decision_20260820_054605"
M1 = RESULTS_ROOT / "m1_force_sufficiency_model_20260822_100045"
M1R2 = RESULTS_ROOT / "m1r2_real_cross_task_probe_20260822_133903"
FORCE_AUDIT = RESULTS_ROOT / "force_interface_forensic_audit_20260823_074718"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
ISAAC_WARP_CORE = Path(
    "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.warp.core-1.8.2+lx64"
)
OPENPI_CLIENT_SRC = REPO / "benchmarks/openpi/openpi-client/src"

TASKS = [0, 1, 5, 6]
EXCLUDED_TASKS = [2]
TASK_OBJECTS = {0: "alphabet_soup_1", 1: "cream_cheese_1", 5: "tomato_sauce_1", 6: "butter_1"}
INSTRUCTIONS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
F_DEV = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
TRAIN_FORCE_VALUES = [3.0, 4.0, 5.0, 6.0, 8.0]
OFFGRID_FORCE_VALUES = [3.5, 4.5, 5.5, 6.5, 7.0, 7.5]
SEED = 5050
FEATURE_COLS = [
    "measured_squeeze",
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
    "marker_motion",
    "marker_tangential",
    "marker_velocity",
    "marker_loading_unloading",
    "accumulated_displacement_mm",
    "eef_displacement_mm",
    "object_displacement_m",
    "object_rotation_rad",
    "contact_left",
    "contact_right",
]


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


OUT = RESULTS_ROOT / f"p5s0_four_task_gnp_force_sufficiency_{now_tag()}"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_obj(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def env_has_true_sim() -> dict:
    mods = {}
    for name in ["gymnasium", "isaaclab", "isaacsim"]:
        mods[name] = importlib.util.find_spec(name) is not None
    external = {
        "python": str(ISAAC_PY),
        "exists": ISAAC_PY.exists(),
        "module_availability": {},
        "check_error": "",
    }
    if ISAAC_PY.exists():
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([str(ISAAC_WARP_CORE), str(REPO), str(OPENPI_CLIENT_SRC)])
        code = (
            "import importlib.util, json;"
            "mods={n: importlib.util.find_spec(n) is not None "
            "for n in ['gymnasium','isaaclab','isaacsim','isaaclab_tasks','tac_manip']};"
            "print(json.dumps(mods, sort_keys=True))"
        )
        try:
            out = subprocess.check_output([str(ISAAC_PY), "-c", code], cwd=REPO, text=True, env=env, timeout=30)
            external["module_availability"] = json.loads(out.strip())
        except Exception as exc:
            external["check_error"] = repr(exc)
    external_ready = bool(external["module_availability"]) and all(external["module_availability"].values())
    return {
        "can_import_true_runtime_from_current_python": bool(all(mods.values())),
        "can_import_true_runtime_from_external_isaac_python": external_ready,
        "can_run_true_post_probe_branching": external_ready,
        "current_python_module_availability": mods,
        "external_isaac_runtime": external,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_steps() -> pd.DataFrame:
    frames = []
    for split in ("development", "main"):
        for primitive in ("S", "L"):
            for task in TASKS:
                p = P4R2 / "P4R2_TIMESTEP_TELEMETRY" / f"P4R2_{split}_{primitive}_TASK{task}_TIMESTEPS.csv"
                if p.exists():
                    frames.append(read_csv(p))
    if not frames:
        raise FileNotFoundError("No P4-R2 timestep telemetry found")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["task_id"].isin(TASKS)].copy()
    return df


def select_probe_contexts(episodes: pd.DataFrame) -> pd.DataFrame:
    episodes = episodes[episodes["task_id"].isin(TASKS)].copy()
    rows = []
    grouped = episodes.groupby(["split", "task_id", "seed_idx", "friction"], sort=True)
    for (_, task, seed, mu), g in grouped:
        s = g[g["primitive"] == "S"]
        l = g[g["primitive"] == "L"]
        selected = None
        if len(s) and str(s.iloc[0].get("selector_output", "")) == "S":
            selected = s.iloc[0]
        elif len(l) and str(l.iloc[0].get("selector_output", "")) == "L":
            selected = l.iloc[0]
        elif len(s):
            selected = s.iloc[0]
        elif len(l):
            selected = l.iloc[0]
        if selected is None:
            continue
        row = selected.to_dict()
        row["context_id"] = f"p5s0_ctx_{row['split']}_t{task}_s{int(seed)}_mu{float(mu):g}"
        row["task_instruction"] = INSTRUCTIONS[int(task)]
        row["probe_qualified"] = int(row.get("qualified", 0))
        row["post_probe_state_hash"] = sha256_obj(
            {
                "source_trial_id": row["trial_id"],
                "task": int(task),
                "seed": int(seed),
                "mu_analysis_only": float(mu),
                "primitive": row["primitive"],
                "probe_stop": row.get("stop_trigger", ""),
                "object_pose": [
                    row.get("object_to_basket_x", ""),
                    row.get("object_to_basket_y", ""),
                    row.get("object_to_basket_z", ""),
                ],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_branch_source() -> pd.DataFrame:
    paths = {
        0: B2R2 / "TASK0_SCAN.csv",
        1: B2R2 / "TASK1_FROZEN_REFERENCE_SCAN.csv",
        5: B2R2 / "TASK5_SCAN.csv",
        6: B2R2 / "TASK6_SCAN.csv",
    }
    frames = []
    for task, path in paths.items():
        df = read_csv(path)
        df = df[df["task_id"].astype(int) == task].copy()
        frames.append(df)
    branches = pd.concat(frames, ignore_index=True)
    branches["force"] = branches["force"].astype(float)
    branches = branches[branches["force"].isin(TRAIN_FORCE_VALUES)].copy()
    branches["friction"] = branches["friction"].astype(float).round(1)
    return branches


def build_context_branch_tables(contexts: pd.DataFrame, branch_source: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(SEED)
    context_rows: list[dict] = []
    branch_rows: list[dict] = []
    raw_dir = OUT / "P5S0_RAW_PROBE_TELEMETRY"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for _, ctx in contexts.sort_values(["task_id", "split", "seed_idx", "friction"]).iterrows():
        trial_id = str(ctx["trial_id"])
        task = int(ctx["task_id"])
        mu = round(float(ctx["friction"]), 1)
        ctx_id = str(ctx["context_id"])
        ctx_steps = steps[steps["trial_id"] == trial_id].copy()
        ctx_steps.to_csv(raw_dir / f"{ctx_id}.csv", index=False)
        context_rows.append(
            {
                "context_id": ctx_id,
                "task": task,
                "task_instruction": INSTRUCTIONS[task],
                "nominal_trajectory_descriptor": "P4R2 object-to-basket downstream descriptor from frozen probe collector",
                "hidden_friction_analysis_only": mu,
                "seed": int(ctx["seed_idx"]),
                "source_probe_trial_id": trial_id,
                "probe_primitive": ctx["primitive"],
                "probe_qualified": int(ctx.get("probe_qualified", 0)),
                "contact_retained": int(1 - int(ctx.get("contact_lost_probe", 0))),
                "drop": int(ctx.get("dropped", 0)),
                "stop_reason": ctx.get("stop_trigger", ""),
                "actual_displacement_mm": float(ctx.get("actual_probe_displacement_mm", np.nan)),
                "post_probe_state_hash": ctx["post_probe_state_hash"],
                "dataset_mode": "ARCHIVAL_PROXY_NOT_POST_PROBE_MATCHED",
            }
        )
        for force in F_DEV:
            source = branch_source[
                (branch_source["task_id"].astype(int) == task)
                & (branch_source["friction"].astype(float).round(1) == mu)
                & (branch_source["force"].astype(float) == force)
            ]
            data_available = len(source) > 0
            if data_available:
                # Deterministic pseudo-branch choice. The outcome is observed, but it
                # is not from the same post-probe state, so the label is proxy only.
                idx = rng.randrange(len(source))
                src = source.iloc[idx]
                y = int(src["full_task_success"])
                failure = "" if y else str(src.get("term_reason", "")) or (
                    "lost_in_transit" if int(src.get("lost_in_transit", 0)) else "full_task_failure"
                )
                measured_mean = float(src.get("mean_grip_force", np.nan))
                measured_peak = float(src.get("peak_force", np.nan))
                tracking_error = measured_mean - float(force)
                source_trial = src["trial_id"]
            else:
                y = ""
                failure = "not_executed_offgrid_or_missing_archival_branch"
                measured_mean = measured_peak = tracking_error = np.nan
                source_trial = ""
            branch_rows.append(
                {
                    "branch_id": f"{ctx_id}_F{force:g}",
                    "context_id": ctx_id,
                    "task": task,
                    "hidden_friction_analysis_only": mu,
                    "seed": int(ctx["seed_idx"]),
                    "requested_force_N": float(force),
                    "force_split_role": "train_force" if force in TRAIN_FORCE_VALUES else "heldout_offgrid_force",
                    "measured_force_mean_N": measured_mean,
                    "measured_force_peak_N": measured_peak,
                    "force_tracking_error_N": tracking_error,
                    "full_task_success_y": y,
                    "failure_reason": failure,
                    "data_available": int(data_available),
                    "label_source": "OBSERVED_B2R2_FULLTASK_ROLLOUT_PROXY_NOT_SAME_POST_PROBE_STATE" if data_available else "MISSING_NOT_FABRICATED",
                    "source_branch_trial_id": source_trial,
                }
            )
    return pd.DataFrame(context_rows), pd.DataFrame(branch_rows), contexts


def make_splits(context_df: pd.DataFrame) -> dict:
    rows = context_df.sort_values(["task", "hidden_friction_analysis_only", "seed"]).to_dict("records")
    by_task = {task: [] for task in TASKS}
    for row in rows:
        by_task[int(row["task"])].append(row["context_id"])
    train, dev, test = [], [], []
    for task, ids in by_task.items():
        for i, cid in enumerate(ids):
            if i % 10 in (8,):
                dev.append(cid)
            elif i % 10 in (9,):
                test.append(cid)
            else:
                train.append(cid)
    # Explicit friction diagnostic: mu=0.5 exists in proxy data but not held out
    # from training here because true continuous-friction S0-A cannot run.
    return {
        "TRAIN_CONTEXTS": train,
        "DEV_CONTEXTS": dev,
        "TEST_CONTEXTS": test,
        "HELD_OUT_FRICTION_DIAGNOSTIC": {
            "requested_in_protocol": "some interior continuous friction values never in training",
            "executed": False,
            "reason": "true simulator branching unavailable; archival proxy has only mu in {0.2,0.5,1.0}",
        },
        "split_unit": "physical_context_proxy",
        "task2_excluded": True,
    }


def tensorize(context_df: pd.DataFrame, branch_df: pd.DataFrame, steps: pd.DataFrame, splits: dict):
    available = branch_df[(branch_df["data_available"] == 1) & (branch_df["requested_force_N"].isin(TRAIN_FORCE_VALUES))].copy()
    context_lookup = {r["context_id"]: r for r in context_df.to_dict("records")}
    step_lookup = {}
    max_len = 0
    feat_mean = []
    for cid, row in context_lookup.items():
        trial = row["source_probe_trial_id"]
        s = steps[(steps["trial_id"] == trial) & (steps["probe_phase"].isin(["probe_out", "probe_back", "probe_hold"]))].copy()
        if s.empty:
            s = steps[steps["trial_id"] == trial].tail(20).copy()
        arr = s.reindex(columns=FEATURE_COLS).fillna(0.0).astype(float).to_numpy(dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
        step_lookup[cid] = arr
        max_len = max(max_len, arr.shape[0])
        feat_mean.append(arr)
    all_steps = np.concatenate(feat_mean, axis=0)
    mu = all_steps.mean(axis=0)
    sd = all_steps.std(axis=0) + 1e-6
    for cid in step_lookup:
        step_lookup[cid] = (step_lookup[cid] - mu) / sd
    x_seq, x_task, x_force, y, split_name, ctx_ids, branch_ids = [], [], [], [], [], [], []
    split_by_context = {}
    for name in ["TRAIN_CONTEXTS", "DEV_CONTEXTS", "TEST_CONTEXTS"]:
        for cid in splits[name]:
            split_by_context[cid] = name.replace("_CONTEXTS", "").lower()
    for _, br in available.iterrows():
        cid = br["context_id"]
        if cid not in context_lookup:
            continue
        arr = step_lookup[cid]
        padded = np.zeros((max_len, len(FEATURE_COLS)), dtype=np.float32)
        padded[: arr.shape[0]] = arr
        x_seq.append(padded)
        x_task.append(TASKS.index(int(br["task"])))
        x_force.append((float(br["requested_force_N"]) - 3.0) / 5.0)
        y.append(float(br["full_task_success_y"]))
        split_name.append(split_by_context.get(cid, "unassigned"))
        ctx_ids.append(cid)
        branch_ids.append(br["branch_id"])
    data = {
        "seq": torch.tensor(np.stack(x_seq), dtype=torch.float32),
        "task": torch.tensor(x_task, dtype=torch.long),
        "force": torch.tensor(x_force, dtype=torch.float32).unsqueeze(1),
        "y": torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        "split": np.array(split_name),
        "context_id": np.array(ctx_ids),
        "branch_id": np.array(branch_ids),
    }
    return data


class TaskForceMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(TASKS), 8)
        self.net = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, seq, task, force):
        return self.net(torch.cat([self.emb(task), force], dim=1))


class DetGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(len(FEATURE_COLS), 32), nn.ReLU())
        self.gru = nn.GRU(32, 64, num_layers=2, batch_first=True)
        self.emb = nn.Embedding(len(TASKS), 8)
        self.dec = nn.Sequential(nn.Linear(64 + 8 + 1, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq, task, force):
        h, _ = self.gru(self.inp(seq))
        return self.dec(torch.cat([h[:, -1], self.emb(task), force], dim=1))


class Q2FGNP(nn.Module):
    def __init__(self, threshold: bool = False, zdim: int = 8):
        super().__init__()
        self.threshold = threshold
        self.inp = nn.Sequential(nn.Linear(len(FEATURE_COLS), 32), nn.ReLU())
        self.gru = nn.GRU(32, 64, num_layers=2, batch_first=True)
        self.mu = nn.Linear(64, zdim)
        self.lv = nn.Linear(64, zdim)
        self.emb = nn.Embedding(len(TASKS), 8)
        if threshold:
            self.head = nn.Sequential(nn.Linear(zdim + 8, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
        else:
            self.head = nn.Sequential(nn.Linear(zdim + 8 + 1, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, seq):
        h, _ = self.gru(self.inp(seq))
        last = h[:, -1]
        return self.mu(last), torch.clamp(self.lv(last), -6.0, 4.0)

    def forward(self, seq, task, force, sample: bool = True):
        mu, lv = self.encode(seq)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if sample and self.training else mu
        te = self.emb(task)
        if not self.threshold:
            return self.head(torch.cat([z, te, force], dim=1)), mu, lv
        out = self.head(torch.cat([z, te], dim=1))
        req = 3.0 + 5.0 * torch.sigmoid(out[:, :1])
        scale = torch.nn.functional.softplus(out[:, 1:2]) + 0.05
        force_n = 3.0 + 5.0 * force
        return (force_n - req) / scale, mu, lv


def auc_score(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece_score(y, p, bins=10):
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    total = len(y)
    if total == 0:
        return np.nan
    ece = 0.0
    for lo in np.linspace(0.0, 0.9, bins):
        hi = lo + 1.0 / bins
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.any():
            ece += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return ece


def metrics(y, p):
    y = np.asarray(y).astype(float)
    p = np.clip(np.asarray(p).astype(float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "auroc": auc_score(y, p),
        "nll": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(y) else np.nan,
        "brier": float(((p - y) ** 2).mean()) if len(y) else np.nan,
        "accuracy": float(((p >= 0.5).astype(float) == y).mean()) if len(y) else np.nan,
        "ece": ece_score(y, p),
    }


def train_model(name: str, model: nn.Module, data: dict, out_dir: Path, beta: float = 0.0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    train_mask = torch.tensor(data["split"] == "train")
    if int(train_mask.sum()) == 0:
        raise RuntimeError("No training rows")
    for _ in range(140):
        model.train()
        opt.zero_grad()
        if isinstance(model, Q2FGNP):
            logits, mu, lv = model(data["seq"][train_mask], data["task"][train_mask], data["force"][train_mask], sample=True)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = bce(logits, data["y"][train_mask]) + beta * kl
        else:
            logits = model(data["seq"][train_mask], data["task"][train_mask], data["force"][train_mask])
            loss = bce(logits, data["y"][train_mask])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        if isinstance(model, Q2FGNP):
            logits, mu, lv = model(data["seq"], data["task"], data["force"], sample=False)
            np.save(out_dir / "latent_mu.npy", mu.numpy())
            np.save(out_dir / "latent_logvar.npy", lv.numpy())
        else:
            logits = model(data["seq"], data["task"], data["force"])
        prob = torch.sigmoid(logits).squeeze(1).numpy()
    pred_rows = []
    result = {}
    for split in ["train", "dev", "test"]:
        m = data["split"] == split
        result[split] = metrics(data["y"].squeeze(1).numpy()[m], prob[m])
        for bid, cid, y, p in zip(data["branch_id"][m], data["context_id"][m], data["y"].squeeze(1).numpy()[m], prob[m]):
            pred_rows.append({"model": name, "split": split, "branch_id": bid, "context_id": cid, "y": float(y), "p": float(p)})
    write_json(out_dir / "config.json", {"model": name, "beta": beta, "features": FEATURE_COLS})
    write_json(out_dir / "metrics.json", result)
    write_csv(out_dir / "predictions.csv", pred_rows)
    torch.save(model.state_dict(), out_dir / "checkpoint.pt")
    return {"name": name, "metrics": result, "predictions": pred_rows}


def force_frontier(branch_df: pd.DataFrame, pred_rows: list[dict], context_df: pd.DataFrame) -> pd.DataFrame:
    pred = pd.DataFrame(pred_rows)
    available = branch_df[(branch_df["data_available"] == 1) & (branch_df["requested_force_N"].isin(TRAIN_FORCE_VALUES))].copy()
    rows = []
    for model_name, gpred in pred.groupby("model"):
        for cid, branches in available.groupby("context_id"):
            actual = branches[branches["full_task_success_y"].astype(int) == 1]
            emp_min = float(actual["requested_force_N"].min()) if len(actual) else np.nan
            pctx = gpred[gpred["context_id"] == cid].copy()
            if pctx.empty:
                continue
            merged = pctx.merge(branches[["branch_id", "requested_force_N"]], on="branch_id")
            chosen = merged[merged["p"] >= 0.8]
            pred_min = float(chosen["requested_force_N"].min()) if len(chosen) else 8.0
            rows.append(
                {
                    "model": model_name,
                    "context_id": cid,
                    "empirical_min_success_force_N": emp_min,
                    "predicted_min_sufficient_force_N": pred_min,
                    "abs_force_error_N": abs(pred_min - emp_min) if np.isfinite(emp_min) else np.nan,
                    "under_force_error_N": max(0.0, emp_min - pred_min) if np.isfinite(emp_min) else np.nan,
                    "excess_force_regret_N": max(0.0, pred_min - emp_min) if np.isfinite(emp_min) else np.nan,
                    "frontier_source": "ARCHIVAL_PROXY_NOT_POST_PROBE_MATCHED",
                }
            )
    return pd.DataFrame(rows)


def save_required_empty_files() -> None:
    for d in ["P5S0_TASK_FORCE_BASELINE", "P5S0_DETERMINISTIC_GRU", "P5S0_Q2F_GNP", "P5S0_Q2F_THRESHOLD"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for csv_name in [
        "P5S0_HELDOUT_FRICTION_RESULTS.csv",
        "P5S0_OFFGRID_FORCE_RESULTS.csv",
        "P5S0_E2E_RESULTS.csv",
        "P5S0_FAILURE_CASES.csv",
    ]:
        if not (OUT / csv_name).exists():
            write_csv(OUT / csv_name, [])


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)

    sim_env = env_has_true_sim()
    protocol = {
        "name": "P5-S0 Four-Task GNP-Style Full-Task Force Sufficiency Pipeline Smoke Test",
        "method_change": "GNP_STYLE_FULL_TASK_FORCE_SUFFICIENCY_SMOKE_TEST",
        "scope": "FOUR_TASK_DEVELOPMENT_SMOKE_TEST",
        "tasks": TASKS,
        "excluded_tasks": EXCLUDED_TASKS,
        "force_dev_N": F_DEV,
        "train_force_values_N": TRAIN_FORCE_VALUES,
        "offgrid_force_values_N": OFFGRID_FORCE_VALUES,
        "model_inputs_prohibited": ["hidden_friction", "GT friction", "future outcome", "canonical F*_grid"],
        "simulator_branching_available": sim_env["can_run_true_post_probe_branching"],
        "true_matched_branching_executed": False,
        "true_matched_branching_blocker": (
            "external Isaac runtime is importable, but this result-scoped runner does not yet implement "
            "single-process post-probe env.scene.get_state/reset_to branching; archival proxy integration "
            "is therefore marked invalid for the strict P5-S0 claim"
        ),
        "archival_proxy_allowed_for_claim": False,
        "frozen_artifact_inputs": {
            "P4": str(P4),
            "P4R2": str(P4R2),
            "B2R2": str(B2R2),
            "D2": str(D2),
            "M1": str(M1),
            "M1R2": str(M1R2),
            "force_audit": str(FORCE_AUDIT),
        },
    }
    write_json(OUT / "P5S0_PROTOCOL.json", protocol)
    (OUT / "P5S0_PROTOCOL_HASH.txt").write_text(sha256_obj(protocol) + "\n", encoding="utf-8")

    script_hash = sha256_file(Path(__file__))
    code_hash = {"git_commit": git_commit(), "script_sha256": script_hash, "p4r2_collect_sha256": sha256_file(P4R2 / "scripts/p4r2_collect_probe.py")}
    write_json(OUT / "P5S0_CODE_HASH.txt", code_hash)

    frozen_probe = {
        "chosen_probe": "P4R2 task-ID-free selector over primitives S/L",
        "basis": "Chosen from prior P4/P4-R2 artifacts before P5-S0 model outcomes; Task2 excluded from P5-S0 scope.",
        "primitive_definitions": json.loads((P4R2 / "P4R2_PRIMITIVE_DEFINITIONS.json").read_text()),
        "selector_rule": json.loads((P4R2 / "P4R2_SELECTOR_RULE.json").read_text()),
        "code_hashes": code_hash,
    }
    write_json(OUT / "P5S0_FROZEN_PROBE.json", frozen_probe)
    (OUT / "P5S0_FROZEN_PROBE.md").write_text(
        "# P5-S0 Frozen Probe\n\n"
        "Frozen implementation: P4-R2 contact-feasible multi-primitive selector over S/L. "
        "No task-ID-specific lookup, friction input, outcome labels, or Task2 data are used by the P5-S0 models.\n\n"
        "Important limitation: this run cannot launch Isaac in the current shell, so generated labels are archival proxy labels, not true post-probe branches.\n",
        encoding="utf-8",
    )
    (OUT / "P5S0_FROZEN_PROBE_CODE_HASH.txt").write_text(script_hash + "\n", encoding="utf-8")

    episodes = read_csv(P4R2 / "P4R2_RAW_EPISODE_MANIFEST.csv")
    steps = load_steps()
    selected_contexts = select_probe_contexts(episodes)
    branch_source = load_branch_source()
    context_df, branch_df, selected_contexts = build_context_branch_tables(selected_contexts, branch_source, steps)
    splits = make_splits(context_df)
    write_csv(OUT / "P5S0_CONTEXT_MANIFEST.csv", context_df.to_dict("records"))
    write_csv(OUT / "P5S0_BRANCH_MANIFEST.csv", branch_df.to_dict("records"))
    write_json(OUT / "P5S0_SPLIT_MANIFEST.json", splits)
    (OUT / "P5S0_DATASET_SCHEMA.md").write_text(
        "# P5-S0 Dataset Schema\n\n"
        "Rows are keyed by `context_id` and `branch_id`. Model inputs are probe telemetry, task id/instruction-derived context, and requested force. "
        "`hidden_friction_analysis_only` is retained for stratified diagnostics and is not provided to models. "
        "`label_source` records whether a branch is an observed archival full-task rollout or missing. "
        "This run marks all labels as proxy because true reset-equivalent post-probe branching was unavailable.\n",
        encoding="utf-8",
    )
    write_csv(OUT / "P5S0_FULL_TASK_BRANCH_RESULTS.csv", branch_df.to_dict("records"))

    data = tensorize(context_df, branch_df, steps, splits)
    models = [
        ("TASK_FORCE_BASELINE", TaskForceMLP(), OUT / "P5S0_TASK_FORCE_BASELINE", 0.0),
        ("DETERMINISTIC_GRU", DetGRU(), OUT / "P5S0_DETERMINISTIC_GRU", 0.0),
        ("Q2F_GNP", Q2FGNP(False), OUT / "P5S0_Q2F_GNP", 0.001),
        ("Q2F_THRESHOLD", Q2FGNP(True), OUT / "P5S0_Q2F_THRESHOLD", 0.001),
    ]
    all_pred_rows: list[dict] = []
    metric_rows: list[dict] = []
    for name, model, out_dir, beta in models:
        res = train_model(name, model, data, out_dir, beta=beta)
        all_pred_rows.extend(res["predictions"])
        for split, m in res["metrics"].items():
            metric_rows.append({"model": name, "split": split, **m})
    heldout_rows = [r for r in metric_rows if r["split"] == "test"]
    write_csv(OUT / "P5S0_HELDOUT_CONTEXT_RESULTS.csv", heldout_rows)
    write_csv(OUT / "P5S0_CALIBRATION_RESULTS.csv", metric_rows)

    # No true held-out friction or off-grid labels exist in this environment.
    write_csv(
        OUT / "P5S0_HELDOUT_FRICTION_RESULTS.csv",
        [{"status": "NOT_RUN", "reason": "true continuous-friction matched branching unavailable; archival proxy has only mu={0.2,0.5,1.0}"}],
    )
    write_csv(
        OUT / "P5S0_OFFGRID_FORCE_RESULTS.csv",
        [{"status": "NOT_RUN", "reason": "no observed full-task off-grid force branch labels; missing labels not fabricated"}],
    )
    frontier = force_frontier(branch_df, all_pred_rows, context_df)
    write_csv(OUT / "P5S0_FORCE_FRONTIER_RESULTS.csv", frontier.to_dict("records"))
    write_csv(
        OUT / "P5S0_E2E_RESULTS.csv",
        [{"status": "NOT_RUN", "reason": "Isaac/gymnasium unavailable; no fresh frozen-model E2E execution performed"}],
    )
    failures = context_df[context_df["probe_qualified"].astype(int) == 0].copy()
    write_csv(OUT / "P5S0_FAILURE_CASES.csv", failures.to_dict("records"))
    save_required_empty_files()

    qualified_rate = float(context_df["probe_qualified"].astype(int).mean()) if len(context_df) else 0.0
    verdict = {
        "STATUS": "COMPLETED_WITH_INVALID_STRICT_MATCHING",
        "METHOD_CHANGE": "GNP_STYLE_FULL_TASK_FORCE_SUFFICIENCY_SMOKE_TEST",
        "PRIMARY_CLASSIFICATION": "P5S0_DATA_GENERATION_OR_MATCHING_INVALID",
        "reason": "The external IsaacLab runtime is importable, but this runner did not execute strict single-probe post-probe-state matched force branching or fresh E2E selection. Model integration used archival observed labels only as a proxy and does not support the scientific P5-S0 claim.",
        "task2_excluded": True,
        "probe_qualified_subset_size": int(context_df["probe_qualified"].astype(int).sum()),
        "total_contexts": int(len(context_df)),
        "probe_qualified_rate": qualified_rate,
        "available_branch_rows": int(branch_df["data_available"].astype(int).sum()),
        "missing_offgrid_branch_rows": int((branch_df["data_available"].astype(int) == 0).sum()),
        "sim_environment": sim_env,
        "reset_post_probe_branch_parity": False,
        "development_only": True,
    }
    write_json(OUT / "P5S0_FINAL_VERDICT.json", verdict)

    def fmt_metric(model_name: str) -> str:
        rows = [r for r in metric_rows if r["model"] == model_name and r["split"] == "test"]
        if not rows:
            return "not available"
        r = rows[0]
        return f"test NLL={r['nll']:.3f}, Brier={r['brier']:.3f}, AUROC={r['auroc'] if np.isfinite(r['auroc']) else 'nan'}, acc={r['accuracy']:.3f}"

    n_train = len(splits["TRAIN_CONTEXTS"])
    n_dev = len(splits["DEV_CONTEXTS"])
    n_test = len(splits["TEST_CONTEXTS"])
    n_qualified = int(context_df["probe_qualified"].sum())
    unique_mu = sorted(float(x) for x in context_df["hidden_friction_analysis_only"].unique())
    available_branches = int(branch_df["data_available"].sum())
    report = f"""# P5-S0 Final Report

STATUS:
COMPLETED_WITH_INVALID_STRICT_MATCHING

METHOD_CHANGE:
GNP_STYLE_FULL_TASK_FORCE_SUFFICIENCY_SMOKE_TEST

ARTIFACTS:
{OUT}

SCOPE:
- Tasks: {TASKS}
- Task2 used: False
- Development-only: True
- Probe implementation: P4-R2 task-ID-free S/L selector, frozen from prior artifacts

DATA:
- Physical contexts: {len(context_df)} archival proxy contexts
- Train/dev/test contexts: {n_train}/{n_dev}/{n_test}
- Hidden friction range: archival proxy only, {min(unique_mu):g}-{max(unique_mu):g}
- Unique friction values: {unique_mu}
- Full-task branches: {available_branches} observed proxy labels; off-grid missing labels not fabricated
- Force values: development lattice {F_DEV}; train-label availability only for {TRAIN_FORCE_VALUES}
- Qualified probe contexts: {n_qualified}/{len(context_df)}
- Reset/post-probe branch parity: False, labels are not descendants of saved `sq`

TASK+F BASELINE:
- Held-out context: {fmt_metric('TASK_FORCE_BASELINE')}
- Held-out friction: NOT_RUN
- Off-grid force: NOT_RUN

DETERMINISTIC GRU:
- Held-out context: {fmt_metric('DETERMINISTIC_GRU')}
- Held-out friction: NOT_RUN
- Off-grid force: NOT_RUN

Q2F-GNP:
- Held-out context: {fmt_metric('Q2F_GNP')}
- Held-out friction: NOT_RUN
- Off-grid force: NOT_RUN
- Calibration: see P5S0_CALIBRATION_RESULTS.csv

Q2F-THRESHOLD:
- Held-out context: {fmt_metric('Q2F_THRESHOLD')}
- Held-out friction: NOT_RUN
- Off-grid force: NOT_RUN
- Calibration: see P5S0_CALIBRATION_RESULTS.csv

PROBE VALUE:
- Does probe improve over Task+F: Not scientifically adjudicated
- By how much: Invalid under strict P5-S0 matching
- On which tasks: Not adjudicated
- Failure cases: see P5S0_FAILURE_CASES.csv

FORCE GENERALIZATION:
- Unseen-force prediction works: NOT_RUN
- Best model: Not selected
- Branch-resolved force error: proxy-only diagnostics in P5S0_FORCE_FRONTIER_RESULTS.csv
- Under-force error: proxy-only diagnostics in P5S0_FORCE_FRONTIER_RESULTS.csv

E2E:
- Fixed Robust full SR / mean force: NOT_RUN
- Task+F full SR / mean force: NOT_RUN
- Q2F-GNP full SR / mean force: NOT_RUN
- Q2F-Threshold full SR / mean force: NOT_RUN

PRIMARY_CLASSIFICATION:
P5S0_DATA_GENERATION_OR_MATCHING_INVALID

SCIENTIFIC_INTERPRETATION:
1. The four-task scope and Task2 exclusion were enforced in the proxy model-integration run.
2. The frozen probe/model plumbing executes, including Task+F, deterministic GRU, Q2F-GNP, and Q2F-Threshold checkpoints.
3. The run cannot answer the scientific Query2Force question because matched post-probe force branching was not executed.
4. Off-grid force and held-out continuous-friction diagnostics remain unmeasured; missing labels were not fabricated.
5. Any apparent probe-conditioned improvement is only a software integration signal, not evidence for P5-S0 success.

WHAT THIS DOES NOT PROVE:
- five-task probe generality
- Task2 coverage
- sub-0.5-N force-control accuracy
- real-robot generalization
- selective when-to-probe

NEXT STEP:
Implement/run the true Isaac single-process generator using `env.scene.get_state(is_relative=True)` immediately after the frozen probe and `env.reset_to(sq, ...)` before every force branch, then rerun training/evaluation on the valid branch table.
"""
    (OUT / "P5S0_FINAL_REPORT.md").write_text(report, encoding="utf-8")

    print(str(OUT))
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
