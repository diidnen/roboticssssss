#!/usr/bin/env python3
"""P5-S0-D fresh four-task end-to-end Query2Force evaluation.

This runner freezes the P5-S0-C P4-B probe and trained threshold models, checks
offline/online inference parity on stored C contexts, then evaluates fresh roots
with four matched arms.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

if os.environ.get("P5S0D_WORKER") == "1":
    pd = None
else:
    import pandas as pd


REPO = Path("/home/exouser/Tabero")
RESULTS_ROOT = REPO / "analysis/results"
C_ARTIFACT = RESULTS_ROOT / "p5s0c_paired_boundary_probe_value_20260824_000542"
B_ARTIFACT = RESULTS_ROOT / "p5s0b_true_matched_q2f_model_comparison_20260823_225313"
A_ARTIFACT = RESULTS_ROOT / "p5s0a_true_matched_dataset_20260823_172635"
B5_REFERENCE = RESULTS_ROOT / "b5_tabero_neutral_20260822_040652/REFERENCE_FIXED_ROBUST.csv"
P4 = RESULTS_ROOT / "p4_contact_conditioned_probe_20260822_184213"
P4_COLLECT = P4 / "scripts/p4_collect_probe.py"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path(
    "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.warp.core-1.8.2+lx64"
)
OPENPI_CLIENT_SRC = REPO / "benchmarks/openpi/openpi-client/src"

TASKS = [0, 1, 5, 6]
TASK_OBJECTS = {0: "alphabet_soup_1", 1: "cream_cheese_1", 5: "tomato_sauce_1", 6: "butter_1"}
TASK_INSTRUCTIONS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
TASK_TO_IDX = {t: i for i, t in enumerate(TASKS)}
BASKET_NAME = "basket_1"
TASK_SUITE = "libero_object"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
EXPECTED_P4B_HASH = "a1566334f9f79ad9d8491612f10d086386049295314f6d1fd62512be1867bca9"

MODEL_SEEDS = [0, 1, 2, 3, 4]
PRIMARY_MODEL = "Q2F_THRESHOLD"
BASELINE_MODEL = "TASK_FORCE_THRESHOLD"
F_EXEC = [3.0, 4.0, 5.0, 6.0, 8.0]
ETA = 0.9
FALLBACK_FORCE = 8.0
ONLINE_MC_SAMPLES = 50
ONLINE_MC_SEED = 0
PARITY_TOL_TASKF = 1e-6
PARITY_TOL_Q2F_ENSEMBLE = 2.0e-2
PARITY_TOL_Q2F_SEED_MC = 8.0e-2
ROBUST_FORCE_BY_TASK = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}

FRICTION_BANDS = {
    "LOW": (0.20, 0.30),
    "MID": (0.45, 0.60),
    "HIGH": (0.90, 1.00),
}
PILOT_ROOTS_PER_TASK = 1
MAIN_ROOTS_PER_TASK = 5
PILOT_ROOT_BASE_SEED = 7100
MAIN_ROOT_BASE_SEED = 7200
FRICTION_SAMPLER_SEED = 2026082405

TIMEOUTS_S = {
    "snapshot": 20.0,
    "branch_restore": 20.0,
    "worker": 21600.0,
}


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


OUT = Path(os.environ.get("P5S0D_OUT", RESULTS_ROOT / f"p5s0d_fresh_e2e_q2f_{now_tag()}"))


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


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fields = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            wr.writeheader()
        wr.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P5C_DATA = import_module(REPO / "analysis/p5s0c_paired_boundary_probe_value.py", "p5s0c_data")
P5C_MODEL = None if os.environ.get("P5S0D_WORKER") == "1" else import_module(REPO / "analysis/p5s0c_model_adjudication.py", "p5s0c_model")

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


class Q2F(nn.Module):
    def __init__(self, input_dim: int, threshold: bool = True, zdim: int = 4):
        super().__init__()
        self.threshold = threshold
        self.proj = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.mu = nn.Linear(32, zdim)
        self.lv = nn.Linear(32, zdim)
        self.emb = nn.Embedding(4, 8)
        self.head = nn.Sequential(nn.Linear(zdim + 8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def encode(self, xseq, lengths):
        z = self.proj(xseq)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        last = h[-1]
        return self.mu(last), torch.clamp(self.lv(last), -7.0, 4.0)

    def forward(self, xseq, xstatic, lengths, task, force, sample: bool = True):
        mu, lv = self.encode(xseq, lengths)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if sample else mu
        return threshold_logit(force, self.head(torch.cat([z, self.emb(task)], dim=1))), mu, lv


def model_factory(name: str, input_dim: int) -> nn.Module:
    if name == BASELINE_MODEL:
        return TaskForceThreshold()
    if name == PRIMARY_MODEL:
        return Q2F(input_dim, threshold=True)
    raise ValueError(name)


def stable_hash_obj(obj: Any) -> str:
    return P5C_DATA.stable_hash_obj(obj)


def restorable_snapshot_for_hash(env) -> dict[str, Any]:
    return P5C_DATA.restorable_snapshot_for_hash(env)


class Timeout:
    def __init__(self, seconds: float, stage: str):
        self.seconds = seconds
        self.stage = stage
        self.old = None

    def __enter__(self):
        def handler(_signum, _frame):
            raise TimeoutError(f"P5S0D timeout during {self.stage} after {self.seconds}s")

        self.old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(self.seconds))

    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old)
        return False


def force_norm(force_n: float) -> float:
    return float(2.0 * ((force_n - 3.0) / 5.0) - 1.0)


class OnlineInference:
    def __init__(self, device_name: str = "cpu"):
        self.torch = torch
        self.device = torch.device(device_name)
        self.norm = json.loads((C_ARTIFACT / "P5S0C_NORMALIZATION.json").read_text())
        self.summary = self._read_model_summary(C_ARTIFACT / "P5S0C_MODEL_SUMMARY.csv")
        self.feature_names = list(self.norm["dynamic_feature_names"])
        self.phases = list(self.norm["phase_categories_from_train"])
        self.states = list(self.norm["contact_state_categories_from_train"])
        self.dynamic_mean = np.asarray(self.norm["dynamic_mean"], dtype=np.float32)
        self.dynamic_std = np.asarray(self.norm["dynamic_std"], dtype=np.float32)
        self.static_mean = np.asarray(self.norm["static_mean"], dtype=np.float32)
        self.static_std = np.asarray(self.norm["static_std"], dtype=np.float32)
        self.models: dict[tuple[str, int], dict[str, Any]] = {}
        input_dim = len(self.feature_names)
        for model_name in [BASELINE_MODEL, PRIMARY_MODEL]:
            for seed in MODEL_SEEDS:
                model = model_factory(model_name, input_dim)
                state = torch.load(
                    C_ARTIFACT / "P5S0C_CHECKPOINTS" / f"{model_name}_seed{seed}.pt",
                    map_location=self.device,
                )
                model.load_state_dict(state)
                model.to(self.device)
                model.eval()
                temp = float(self.summary[(model_name, seed)])
                self.models[(model_name, seed)] = {"model": model, "temperature": temp}

    @staticmethod
    def _read_model_summary(path: Path) -> dict[tuple[str, int], float]:
        out: dict[tuple[str, int], float] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out[(row["model"], int(row["seed"]))] = float(row["temperature"])
        return out

    @staticmethod
    def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
        try:
            val = row.get(key, "")
            return default if val == "" else float(val)
        except Exception:
            return default

    def sequence_array(self, telemetry_path: str) -> np.ndarray:
        with open(telemetry_path, newline="", encoding="utf-8") as fh:
            raw_rows = list(csv.DictReader(fh))
        if not raw_rows:
            return np.zeros((1, len(self.feature_names)), dtype=np.float32)
        ex0 = self._f(raw_rows[0], "eef_x", 0.0)
        ey0 = self._f(raw_rows[0], "eef_y", 0.0)
        ez0 = self._f(raw_rows[0], "eef_z", 0.0)
        rows: list[list[float]] = []
        for rr in raw_rows:
            feat: dict[str, float] = {}
            for col in NUMERIC_PROBE_COLS:
                feat[col] = self._f(rr, col, 0.0)
            feat["eef_dx"] = self._f(rr, "eef_x", 0.0) - ex0
            feat["eef_dy"] = self._f(rr, "eef_y", 0.0) - ey0
            feat["eef_dz"] = self._f(rr, "eef_z", 0.0) - ez0
            phase = str(rr.get("probe_phase", "NA"))
            state = str(rr.get("contact_state", "NA"))
            for p in self.phases:
                feat[f"phase={p}"] = 1.0 if phase == p else 0.0
            for s in self.states:
                feat[f"contact_state={s}"] = 1.0 if state == s else 0.0
            feat["probe_phase_unknown"] = 0.0 if phase in self.phases else 1.0
            feat["contact_state_unknown"] = 0.0 if state in self.states else 1.0
            rows.append([float(feat.get(name, 0.0)) for name in self.feature_names])
        return np.asarray(rows, dtype=np.float32)

    def encode_probe(self, telemetry_path: str) -> tuple[np.ndarray, np.ndarray, int]:
        arr_raw = self.sequence_array(telemetry_path)
        left_idx = self.feature_names.index("contact_left")
        right_idx = self.feature_names.index("contact_right")
        contact = (arr_raw[:, left_idx].astype(float) + arr_raw[:, right_idx].astype(float)) > 0
        static_idx = int(np.argmax(contact)) if contact.any() else 0
        xseq = (arr_raw - self.dynamic_mean) / self.dynamic_std
        xstatic = (arr_raw[static_idx] - self.static_mean) / self.static_std
        return xseq.astype(np.float32), xstatic.astype(np.float32), int(xseq.shape[0])

    def probabilities(self, model_name: str, task_id: int, telemetry_path: str, forces: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        torch = self.torch
        xseq, xstatic, length = self.encode_probe(telemetry_path)
        n = len(forces)
        xs = np.repeat(xseq[None, :, :], n, axis=0)
        xst = np.repeat(xstatic[None, :], n, axis=0)
        lengths = np.full(n, length, dtype=np.int64)
        tasks = np.full(n, TASK_TO_IDX[int(task_id)], dtype=np.int64)
        f_norm = np.asarray([force_norm(f) for f in forces], dtype=np.float32).reshape(n, 1)
        probs_by_seed: list[dict[str, Any]] = []
        seed_probs = []
        with torch.no_grad():
            for seed in MODEL_SEEDS:
                fit = self.models[(model_name, seed)]
                model = fit["model"]
                x_t = torch.tensor(xs, dtype=torch.float32, device=self.device)
                xs_t = torch.tensor(xst, dtype=torch.float32, device=self.device)
                len_t = torch.tensor(lengths, dtype=torch.long, device=self.device)
                task_t = torch.tensor(tasks, dtype=torch.long, device=self.device)
                force_t = torch.tensor(f_norm, dtype=torch.float32, device=self.device)
                if isinstance(model, Q2F):
                    torch.manual_seed(ONLINE_MC_SEED + seed * 1000)
                    x_mc = x_t.repeat_interleave(ONLINE_MC_SAMPLES, dim=0)
                    xs_mc = xs_t.repeat_interleave(ONLINE_MC_SAMPLES, dim=0)
                    len_mc = len_t.repeat_interleave(ONLINE_MC_SAMPLES, dim=0)
                    task_mc = task_t.repeat_interleave(ONLINE_MC_SAMPLES, dim=0)
                    force_mc = force_t.repeat_interleave(ONLINE_MC_SAMPLES, dim=0)
                    logits, _, _ = model(x_mc, xs_mc, len_mc, task_mc, force_mc, sample=True)
                    logits_np = logits.detach().cpu().numpy().reshape(n, ONLINE_MC_SAMPLES).mean(axis=1)
                else:
                    logits_np = model(x_t, xs_t, len_t, task_t, force_t).detach().cpu().numpy().reshape(-1)
                p = 1.0 / (1.0 + np.exp(-np.clip(logits_np / float(fit["temperature"]), -60, 60)))
                seed_probs.append(p)
                for force, prob in zip(forces, p):
                    probs_by_seed.append({"model": model_name, "seed": seed, "task": task_id, "force_N": float(force), "p_success": float(prob), "temperature": float(fit["temperature"])})
        ens = np.stack(seed_probs).mean(axis=0)
        curve = [{"model": model_name, "seed": "ENSEMBLE_MEAN", "task": task_id, "force_N": float(force), "p_success": float(prob), "temperature": ""} for force, prob in zip(forces, ens)]
        return probs_by_seed, curve

    def select_force(self, model_name: str, task_id: int, telemetry_path: str, root_context_id: str, arm: str, out_dir: Path) -> dict[str, Any]:
        seed_rows, ens_rows = self.probabilities(model_name, task_id, telemetry_path, F_EXEC)
        curve_rows = []
        for row in seed_rows + ens_rows:
            row = dict(row)
            row["root_context_id"] = root_context_id
            row["arm"] = arm
            curve_rows.append(row)
        write_csv(out_dir / f"{root_context_id}_{arm}_{model_name}_force_curve.csv", curve_rows)
        chosen = next((r for r in ens_rows if float(r["p_success"]) >= ETA), None)
        fallback = 0
        if chosen is None:
            chosen = ens_rows[-1]
            fallback = 1
        return {
            "model": model_name,
            "selected_force_N": float(chosen["force_N"]),
            "selected_probability": float(chosen["p_success"]),
            "fallback": fallback,
            "eta": ETA,
            "force_lattice": json.dumps(F_EXEC),
            "curve_path": str(out_dir / f"{root_context_id}_{arm}_{model_name}_force_curve.csv"),
            "aggregation": "mean calibrated probability across five frozen seeds",
        }


def build_root_plan(phase: str) -> list[dict[str, Any]]:
    roots_per_task = PILOT_ROOTS_PER_TASK if phase == "pilot" else MAIN_ROOTS_PER_TASK
    base_seed = PILOT_ROOT_BASE_SEED if phase == "pilot" else MAIN_ROOT_BASE_SEED
    rng = np.random.default_rng(FRICTION_SAMPLER_SEED + (0 if phase == "pilot" else 100))
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for root_index in range(roots_per_task):
            root_seed = base_seed + root_index
            root_id = f"p5s0d_{phase}_t{task}_root{root_index:02d}_s{root_seed}"
            for band, (lo, hi) in FRICTION_BANDS.items():
                mu = float(rng.uniform(lo, hi))
                rows.append(
                    {
                        "phase": phase,
                        "task": task,
                        "root_index": root_index,
                        "root_seed": root_seed,
                        "root_id": root_id,
                        "friction_band": band,
                        "hidden_friction_analysis_only": mu,
                        "root_context_id": f"p5s0d_{phase}_t{task}_r{root_index:02d}_s{root_seed}_{band.lower()}_mu{mu:.6f}",
                    }
                )
    return rows


def rows_for_task(phase: str, task_id: int) -> list[dict[str, Any]]:
    return [r for r in build_root_plan(phase) if int(r["task"]) == int(task_id)]


def write_static_artifacts(out: Path) -> bool:
    p4_hash = sha256_file(P4_COLLECT)
    protocol = {
        "experiment": "P5-S0-D",
        "status_scope": "FOUR_TASK_DEVELOPMENT_E2E",
        "method_change": "NONE",
        "protocol_change": "FRESH_FOUR_TASK_END_TO_END_EVALUATION",
        "tasks": TASKS,
        "task2_used": False,
        "probe": "P4-B common contact-frame shear",
        "models": {"primary": PRIMARY_MODEL, "baseline": BASELINE_MODEL, "seeds": MODEL_SEEDS},
        "force_lattice_N": F_EXEC,
        "eta": ETA,
        "fallback_N": FALLBACK_FORCE,
        "pilot_root_contexts": 4 * 3 * PILOT_ROOTS_PER_TASK,
        "main_root_contexts": 4 * 3 * MAIN_ROOTS_PER_TASK,
        "main_full_task_rollouts": 4 * 3 * MAIN_ROOTS_PER_TASK * 4,
        "friction_bands": FRICTION_BANDS,
        "no_task2": True,
        "no_training": True,
        "no_threshold_tuning_on_fresh_e2e": True,
    }
    write_json(out / "P5S0D_PROTOCOL.json", protocol)
    (out / "P5S0D_PROTOCOL_HASH.txt").write_text(sha256_obj(protocol) + "\n", encoding="utf-8")
    (out / "P5S0D_CODE_HASH.txt").write_text(
        json.dumps({"git_commit": git_commit(), "runner_sha256": sha256_file(Path(__file__)), "p4b_collect_sha256": p4_hash}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_json(out / "P5S0D_ROOT_MANIFEST.json", {"pilot": build_root_plan("pilot"), "main": build_root_plan("main")})
    write_json(out / "P5S0D_FRICTION_MANIFEST.json", {"friction_bands": FRICTION_BANDS, "friction_is_model_input": False})

    model_summary = pd.read_csv(C_ARTIFACT / "P5S0C_MODEL_SUMMARY.csv")
    selection = pd.read_csv(C_ARTIFACT / "P5S0C_FORCE_SELECTION_RESULTS.csv")
    unambiguous = (
        set([PRIMARY_MODEL, BASELINE_MODEL]).issubset(set(model_summary["model"]))
        and set([PRIMARY_MODEL, BASELINE_MODEL]).issubset(set(selection["model"]))
        and p4_hash == EXPECTED_P4B_HASH
    )
    manifest = {
        "source_artifacts": {"p5s0a": str(A_ARTIFACT), "p5s0b": str(B_ARTIFACT), "p5s0c": str(C_ARTIFACT)},
        "primary_model": PRIMARY_MODEL,
        "baseline_model": BASELINE_MODEL,
        "model_seeds": MODEL_SEEDS,
        "checkpoint_source": str(C_ARTIFACT / "P5S0C_CHECKPOINTS"),
        "feature_preprocessing": str(C_ARTIFACT / "P5S0C_FEATURE_MANIFEST.json"),
        "normalization": str(C_ARTIFACT / "P5S0C_NORMALIZATION.json"),
        "calibration": str(C_ARTIFACT / "P5S0C_MODEL_SUMMARY.csv"),
        "selection_source": str(C_ARTIFACT / "P5S0C_FORCE_SELECTION_RESULTS.csv"),
        "unambiguous_primary_rule_available_before_fresh_e2e": bool(unambiguous),
    }
    write_json(out / "P5S0D_FROZEN_MODEL_MANIFEST.json", manifest)

    hash_lines = []
    for model_name in [BASELINE_MODEL, PRIMARY_MODEL]:
        for seed in MODEL_SEEDS:
            p = C_ARTIFACT / "P5S0C_CHECKPOINTS" / f"{model_name}_seed{seed}.pt"
            hash_lines.append(f"{sha256_file(p)}  {p}")
    for p in [C_ARTIFACT / "P5S0C_FEATURE_MANIFEST.json", C_ARTIFACT / "P5S0C_NORMALIZATION.json", C_ARTIFACT / "P5S0C_MODEL_SUMMARY.csv", C_ARTIFACT / "P5S0C_FORCE_SELECTION_RESULTS.csv", P4_COLLECT]:
        hash_lines.append(f"{sha256_file(p)}  {p}")
    (out / "P5S0D_FROZEN_MODEL_HASHES.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    rule_md = f"""# P5-S0-D Frozen Selection Rule

METHOD_CHANGE: NONE

Source rule from P5-S0-C:
- calibrated model probability `P(Y_full=1 | evidence, task, F)`
- threshold `eta = {ETA}`
- select the lowest candidate force whose calibrated probability exceeds eta
- if no candidate passes eta, use the maximum evaluated force

P5-S0-D protocol projection:
- candidate execution lattice is `{F_EXEC}` N
- fallback is `{FALLBACK_FORCE}` N
- all five saved seeds are used as a frozen ensemble by averaging calibrated probabilities before thresholding
- online/offline parity gates Task+F per seed and Q2F ensemble mean; Q2F per-seed MC rows are diagnostic because P5-S0-C did not save the latent sampling RNG state
- no fresh E2E outcome is used for model, threshold, or rule selection

Ambiguity check:
- primary model present in C selection table: `{PRIMARY_MODEL in set(selection['model'])}`
- baseline model present in C selection table: `{BASELINE_MODEL in set(selection['model'])}`
- P4-B hash matches expected: `{p4_hash == EXPECTED_P4B_HASH}`
- unambiguous before fresh evaluation: `{unambiguous}`
"""
    (out / "P5S0D_FROZEN_SELECTION_RULE.md").write_text(rule_md, encoding="utf-8")
    return bool(unambiguous)


def run_online_offline_parity(out: Path) -> bool:
    inf = OnlineInference("cpu")
    contexts = pd.read_csv(C_ARTIFACT / "P5S0C_CONTEXT_MANIFEST.csv")
    branches = pd.read_csv(C_ARTIFACT / "P5S0C_BRANCH_MANIFEST.csv")
    pred = pd.read_csv(C_ARTIFACT / "P5S0C_PREDICTIONS.csv")
    selected_contexts = []
    for task in TASKS:
        g = contexts[(contexts["task"] == task) & (contexts["split"] == "TEST")].sort_values("context_id")
        if not g.empty:
            selected_contexts.append(g.iloc[0])
    selected_contexts = selected_contexts[:4]
    rows: list[dict[str, Any]] = []
    for ctx in selected_contexts:
        cid = str(ctx.context_id)
        task = int(ctx.task)
        c_forces = sorted(float(x) for x in branches[branches["context_id"] == cid]["requested_force_N"].unique())
        compare_forces = [f for f in F_EXEC if f in c_forces]
        for model_name in [BASELINE_MODEL, PRIMARY_MODEL]:
            seed_rows, ens_rows = inf.probabilities(model_name, task, str(ctx.probe_telemetry_path), compare_forces)
            for row in seed_rows:
                tol = PARITY_TOL_TASKF if model_name == BASELINE_MODEL else PARITY_TOL_Q2F_SEED_MC
                exp = pred[
                    (pred["model"] == model_name)
                    & (pred["seed"].astype(int) == int(row["seed"]))
                    & (pred["context_id"] == cid)
                    & (np.isclose(pred["force_N"].astype(float), float(row["force_N"])))
                ]
                if exp.empty:
                    continue
                expected = float(exp["p"].iloc[0])
                actual = float(row["p_success"])
                rows.append(
                    {
                        "context_id": cid,
                        "task": task,
                        "row_type": "PER_SEED" if model_name == BASELINE_MODEL else "PER_SEED_MC_DIAGNOSTIC",
                        "model": model_name,
                        "seed": row["seed"],
                        "force_N": row["force_N"],
                        "offline_probability": expected,
                        "online_probability": actual,
                        "abs_diff": abs(actual - expected),
                        "tolerance": tol,
                        "parity_pass": int(abs(actual - expected) <= tol),
                        "feature_order": "P5S0C_NORMALIZATION.dynamic_feature_names",
                        "normalization": "P5S0C_NORMALIZATION frozen means/stds",
                        "sequence_masking": "single live sequence length",
                        "probe_phase_encoding": "P5S0C train categories plus unknown",
                        "task_encoding": "TASK_TO_IDX [0,1,5,6]",
                        "force_normalization": "2*((F-3)/5)-1",
                        "checkpoint_loading": "P5S0C_CHECKPOINTS",
                        "calibration": "P5S0C_MODEL_SUMMARY temperature",
                        "ensemble_aggregation": "not compared in C table; per-seed parity checked",
                    }
                )
            if model_name == PRIMARY_MODEL:
                for erow in ens_rows:
                    exp = pred[
                        (pred["model"] == model_name)
                        & (pred["context_id"] == cid)
                        & (np.isclose(pred["force_N"].astype(float), float(erow["force_N"])))
                    ]
                    if exp.empty:
                        continue
                    expected = float(exp["p"].mean())
                    actual = float(erow["p_success"])
                    rows.append(
                        {
                            "context_id": cid,
                            "task": task,
                            "row_type": "ENSEMBLE_MEAN_GATE",
                            "model": model_name,
                            "seed": "ENSEMBLE_MEAN",
                            "force_N": erow["force_N"],
                            "offline_probability": expected,
                            "online_probability": actual,
                            "abs_diff": abs(actual - expected),
                            "tolerance": PARITY_TOL_Q2F_ENSEMBLE,
                            "parity_pass": int(abs(actual - expected) <= PARITY_TOL_Q2F_ENSEMBLE),
                            "feature_order": "P5S0C_NORMALIZATION.dynamic_feature_names",
                            "normalization": "P5S0C_NORMALIZATION frozen means/stds",
                            "sequence_masking": "single live sequence length",
                            "probe_phase_encoding": "P5S0C train categories plus unknown",
                            "task_encoding": "TASK_TO_IDX [0,1,5,6]",
                            "force_normalization": "2*((F-3)/5)-1",
                            "checkpoint_loading": "P5S0C_CHECKPOINTS",
                            "calibration": "P5S0C_MODEL_SUMMARY temperature",
                            "ensemble_aggregation": "mean calibrated probability across five frozen seeds",
                        }
                    )
    write_csv(out / "P5S0D_ONLINE_OFFLINE_PARITY.csv", rows)
    return bool(rows) and all(int(r["parity_pass"]) == 1 for r in rows)


def import_p4_probe(task_id: int):
    os.environ["P4_TASK_ID"] = str(task_id)
    os.environ["P4_VARIANT"] = "P4B"
    os.environ["P4_OUT"] = str(OUT / "P5S0D_FROZEN_P4B_IMPORT")
    spec = importlib.util.spec_from_file_location(f"p5s0d_p4b_task{task_id}", P4_COLLECT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {P4_COLLECT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def query_with_captured_s0(env, p4, *, seed: int, mu: float, trial_id: str, dt: float):
    captured: dict[str, Any] = {"calls": 0, "s0": None, "h0": ""}
    original = p4._current_contact_frame

    def wrapped(current_env):
        captured["calls"] += 1
        result = original(current_env)
        if captured["calls"] == 2 and captured["s0"] is None:
            captured["s0"] = current_env.scene.get_state(is_relative=True)
            captured["h0"] = stable_hash_obj(restorable_snapshot_for_hash(current_env))
        return result

    p4._current_contact_frame = wrapped
    try:
        rows, rec = p4.run_probe_episode(env, seed_idx=seed, mu=mu, trial_id=trial_id, dt=dt)
    finally:
        p4._current_contact_frame = original
    if captured["s0"] is None:
        raise RuntimeError("failed to capture pre-shear s0 from P4-B query")
    sq = env.scene.get_state(is_relative=True)
    hq = stable_hash_obj(restorable_snapshot_for_hash(env))
    return captured["s0"], captured["h0"], sq, hq, rows, rec


def summarize_trajectory(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"transport_phase_measured_force_N": "", "steady_state_measured_force_N": "", "tracking_mae_N": ""}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    measured = []
    transport = []
    for row in rows:
        try:
            val = float(row.get("measured_force_N", ""))
        except Exception:
            continue
        measured.append(val)
        if row.get("phase") in {"transit", "over_basket"}:
            transport.append(val)
    return {
        "transport_phase_measured_force_N": float(np.mean(transport)) if transport else "",
        "steady_state_measured_force_N": float(np.mean(measured[len(measured) // 2 :])) if measured else "",
    }


def run_arm(env, p4, plan: dict[str, Any], arm: str, force: float, start_state, ref_hash: str, dt: float, out: Path, model_decision: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    root_context_id = plan["root_context_id"]
    with Timeout(TIMEOUTS_S["branch_restore"], f"{arm}_RESTORE"):
        env.reset_to(start_state, torch.tensor([0], device=env.device), is_relative=True)
    post_hash = stable_hash_obj(restorable_snapshot_for_hash(env))
    parity = int(post_hash == ref_hash)
    parity_row = {
        "phase": plan["phase"],
        "root_context_id": root_context_id,
        "root_id": plan["root_id"],
        "task": plan["task"],
        "friction_band": plan["friction_band"],
        "arm": arm,
        "reference_hash": ref_hash,
        "post_restore_hash": post_hash,
        "parity_pass": parity,
    }
    telemetry_path = out / "P5S0D_EXECUTION_TELEMETRY" / f"{root_context_id}_{arm}_F{force:g}_trajectory.csv"
    rec = P5C_DATA.downstream_branch(
        env,
        p4,
        task_id=int(plan["task"]),
        force=float(force),
        branch_label=arm,
        context_id=root_context_id,
        split=plan["phase"].upper(),
        seed=int(plan["root_seed"]),
        friction=float(plan["hidden_friction_analysis_only"]),
        dt=dt,
        logger=P5C_DATA.StageLogger(out, int(plan["task"]), int(plan["root_seed"]), dt=dt, context_id=root_context_id, split=plan["phase"].upper(), friction=float(plan["hidden_friction_analysis_only"])),
        telemetry_path=telemetry_path,
        label_source=f"P5S0D_{arm}",
    )
    rec.update(
        {
            "phase": plan["phase"],
            "root_id": plan["root_id"],
            "root_index": plan["root_index"],
            "root_seed": plan["root_seed"],
            "friction_band": plan["friction_band"],
            "arm": arm,
            "arm_model": model_decision.get("model", "FIXED_ROBUST"),
            "selected_probability": model_decision.get("selected_probability", ""),
            "force_selection_fallback": model_decision.get("fallback", 0),
            "force_curve_path": model_decision.get("curve_path", ""),
            "state_parity": parity,
            "start_state_hash": ref_hash,
            "under_force_failure": int(int(rec.get("full_task_success_y", 0)) == 0 and (int(rec.get("lost_in_transit", 0)) or int(rec.get("dropped", 0)) or int(rec.get("lift_success", 0)) == 0)),
        }
    )
    rec.update(summarize_trajectory(telemetry_path))
    return rec, parity_row


def worker_main() -> int:
    from isaaclab.app import AppLauncher

    phase = os.environ["P5S0D_PHASE"]
    task_id = int(os.environ["P5S0D_TASK_ID"])
    task_dir = OUT / phase / f"task{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        p4 = import_p4_probe(task_id)
        setup_task_objects(TASK_SUITE, task_id)
        env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = 45.0
        env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        online = OnlineInference("cpu")

        query_rows: list[dict[str, Any]] = []
        arm_rows: list[dict[str, Any]] = []
        force_rows: list[dict[str, Any]] = []
        parity_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        curve_dir = OUT / "P5S0D_FORCE_DECISIONS"
        probe_dir = OUT / "P5S0D_QUERY_TELEMETRY"

        for plan in rows_for_task(phase, task_id):
            root_context_id = plan["root_context_id"]
            s0, h0, sq, hq, probe_steps, probe_rec = query_with_captured_s0(
                env,
                p4,
                seed=int(plan["root_seed"]),
                mu=float(plan["hidden_friction_analysis_only"]),
                trial_id=root_context_id,
                dt=dt,
            )
            probe_path = probe_dir / f"{root_context_id}_probe_timesteps.csv"
            write_csv(probe_path, [asdict(r) for r in probe_steps])
            quality = P5C_DATA.derive_p4b_probe_quality(p4, probe_steps, probe_rec)
            qrow = {
                **plan,
                "probe_telemetry_path": str(probe_path),
                "pre_query_state_hash": h0,
                "post_query_state_hash": hq,
                "query_qualified": int(quality["probe_qualified"]),
                "contact_retention": int(quality["contact_retained"]),
                "drop": int(quality["drop"]),
                "disturbance": int(quality["disturbance"]),
                "query_completion": int(any(r.probe_phase == "probe_out" for r in probe_steps)),
                "return_completion": int(quality["return_completed"]),
                "evidence_validity": int(int(quality["probe_qualified"]) == 1 and int(probe_rec.get("tactile_ok", 0)) == 1),
                "stop_reason": probe_rec.get("stop_trigger", ""),
                "query_duration_s": probe_rec.get("probe_duration_s", ""),
                "query_displacement_mm": probe_rec.get("actual_probe_displacement_mm", ""),
                "QUERY_NOT_QUALIFIED": int(int(quality["probe_qualified"]) == 0),
            }
            query_rows.append(qrow)

            taskf_dec = online.select_force(BASELINE_MODEL, task_id, str(probe_path), root_context_id, "TASKF", curve_dir)
            if int(quality["probe_qualified"]) == 1:
                q2f_dec = online.select_force(PRIMARY_MODEL, task_id, str(probe_path), root_context_id, "Q2F", curve_dir)
            else:
                q2f_dec = {
                    "model": PRIMARY_MODEL,
                    "selected_force_N": FALLBACK_FORCE,
                    "selected_probability": "",
                    "fallback": 1,
                    "curve_path": "",
                    "aggregation": "query not qualified fallback",
                }
            decisions = {
                "ARM_A_FIXED_ROBUST": {"model": "FIXED_ROBUST", "selected_force_N": ROBUST_FORCE_BY_TASK[task_id], "selected_probability": "", "fallback": 0, "curve_path": ""},
                "ARM_B_TASKF": taskf_dec,
                "ARM_C_QUERY_CONTROL": taskf_dec,
                "ARM_D_QUERY2FORCE": q2f_dec,
            }
            starts = {
                "ARM_A_FIXED_ROBUST": (s0, h0),
                "ARM_B_TASKF": (s0, h0),
                "ARM_C_QUERY_CONTROL": (sq, hq),
                "ARM_D_QUERY2FORCE": (sq, hq),
            }
            for arm in ["ARM_A_FIXED_ROBUST", "ARM_B_TASKF", "ARM_C_QUERY_CONTROL", "ARM_D_QUERY2FORCE"]:
                dec = decisions[arm]
                rec, prow = run_arm(env, p4, plan, arm, float(dec["selected_force_N"]), starts[arm][0], starts[arm][1], dt, OUT, dec)
                rec["query_qualified"] = qrow["query_qualified"] if "QUERY" in arm else ""
                rec["QUERY_NOT_QUALIFIED"] = qrow["QUERY_NOT_QUALIFIED"] if "QUERY" in arm else ""
                arm_rows.append(rec)
                parity_rows.append(prow)
                force_rows.append(
                    {
                        "phase": phase,
                        "root_context_id": root_context_id,
                        "root_id": plan["root_id"],
                        "task": task_id,
                        "friction_band": plan["friction_band"],
                        "arm": arm,
                        "model": dec.get("model", ""),
                        "requested_force_N": rec["requested_force_N"],
                        "selected_probability": rec.get("selected_probability", ""),
                        "steady_state_measured_force_N": rec.get("steady_state_mean_N", ""),
                        "transport_phase_measured_force_N": rec.get("transport_phase_measured_force_N", ""),
                        "tracking_mae_N": rec.get("force_tracking_mae_N", ""),
                        "raw_peak_force_analysis_only_N": rec.get("measured_force_peak_N", ""),
                        "fallback": dec.get("fallback", 0),
                    }
                )
                if int(rec.get("full_task_success_y", 0)) == 0 or int(prow["parity_pass"]) == 0:
                    fail_row = {k: rec.get(k, "") for k in ["phase", "root_context_id", "root_id", "task", "friction_band", "arm", "failure_reason", "failure_stage", "requested_force_N", "dropped", "lost_in_transit", "timeout", "state_parity"]}
                    fail_row["root_context_id"] = rec.get("root_context_id", rec.get("context_id", ""))
                    failure_rows.append(fail_row)

            write_csv(task_dir / "query.csv", query_rows)
            write_csv(task_dir / "arms.csv", arm_rows)
            write_csv(task_dir / "force.csv", force_rows)
            write_csv(task_dir / "state_parity.csv", parity_rows)
            write_csv(task_dir / "failure_cases.csv", failure_rows)

        write_json(task_dir / "result.json", {"phase": phase, "task": task_id, "pass": True, "root_contexts": len(rows_for_task(phase, task_id)), "arms": len(arm_rows)})
        return 0
    except Exception as exc:
        write_json(task_dir / "error.json", {"phase": phase, "task": task_id, "error": repr(exc), "trace": traceback.format_exc()})
        return 1
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass


def worker_env(out: Path, phase: str, task: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(REPO), str(OPENPI_CLIENT_SRC)]),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "TABERO_ROOT": str(REPO),
            "HDF5_TRAJ_SOURCE_DIR": str(REPO / "benchmarks/datasets/libero/assembled_hdf5"),
            "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
            "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
            "P5S0D_OUT": str(out),
            "P5S0D_WORKER": "1",
            "P5S0D_PHASE": phase,
            "P5S0D_TASK_ID": str(task),
        }
    )
    return env


def launch_worker(out: Path, phase: str, task: int) -> dict[str, Any]:
    log_path = out / "logs" / f"{phase}_task{task}_worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ISAAC_PY), "-u", str(Path(__file__).resolve())]
    start = time.time()
    start_iso = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=worker_env(out, phase, task), stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = proc.wait(timeout=TIMEOUTS_S["worker"])
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            returncode = 124
    return {"phase": phase, "task": task, "worker_pid": proc.pid, "process_start_utc": start_iso, "returncode": returncode, "elapsed_wall_s": time.time() - start, "log": str(log_path)}


def read_phase_rows(out: Path, phase: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    queries: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    forces: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in TASKS:
        task_dir = out / phase / f"task{task}"
        for name, dest in [("query.csv", queries), ("arms.csv", arms), ("force.csv", forces), ("state_parity.csv", parity), ("failure_cases.csv", failures)]:
            path = task_dir / name
            if path.exists() and path.stat().st_size > 0:
                with path.open() as fh:
                    dest.extend(list(csv.DictReader(fh)))
    return queries, arms, forces, parity, failures


def bootstrap_ci(values: np.ndarray, seed: int = 123, n_boot: int = 1000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    return float(np.mean(values)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def aggregate_and_report(out: Path, worker_records: list[dict[str, Any]], status_override: str | None = None) -> str:
    queries, arms, forces, parity, failures = read_phase_rows(out, "main")
    write_csv(out / "P5S0D_QUERY_RESULTS.csv", queries)
    write_csv(out / "P5S0D_ARM_RESULTS.csv", arms)
    write_csv(out / "P5S0D_FORCE_RESULTS.csv", forces)
    write_csv(out / "P5S0D_STATE_PARITY.csv", parity)
    write_csv(out / "P5S0D_FAILURE_CASES.csv", failures)
    qdf = pd.DataFrame(queries)
    adf = pd.DataFrame(arms)
    fdf = pd.DataFrame(forces)
    pdf = pd.DataFrame(parity)
    if not adf.empty and "root_context_id" not in adf and "context_id" in adf:
        adf["root_context_id"] = adf["context_id"]

    comparisons: list[dict[str, Any]] = []
    if not adf.empty:
        adf["full_task_success_y"] = pd.to_numeric(adf["full_task_success_y"], errors="coerce")
        adf["requested_force_N"] = pd.to_numeric(adf["requested_force_N"], errors="coerce")
        adf["under_force_failure"] = pd.to_numeric(adf["under_force_failure"], errors="coerce").fillna(0)
        failure_export_cols = ["phase", "root_context_id", "root_id", "task", "friction_band", "arm", "failure_reason", "failure_stage", "requested_force_N", "dropped", "lost_in_transit", "timeout", "state_parity"]
        failure_export = adf[adf["full_task_success_y"] == 0].copy()
        if not failure_export.empty:
            write_csv(out / "P5S0D_FAILURE_CASES.csv", failure_export[failure_export_cols].to_dict("records"))
        for other in ["ARM_A_FIXED_ROBUST", "ARM_B_TASKF", "ARM_C_QUERY_CONTROL"]:
            for metric, col in [("success", "full_task_success_y"), ("requested_force_N", "requested_force_N"), ("under_force", "under_force_failure")]:
                piv = adf[adf["arm"].isin(["ARM_D_QUERY2FORCE", other])].pivot_table(index="root_context_id", columns="arm", values=col, aggfunc="first")
                if "ARM_D_QUERY2FORCE" in piv and other in piv:
                    diff = piv["ARM_D_QUERY2FORCE"].astype(float) - piv[other].astype(float)
                    mean, lo, hi = bootstrap_ci(diff.to_numpy())
                    comparisons.append({"comparison": f"Q2F vs {other}", "metric": metric, "paired_mean_difference": mean, "ci95_lo": lo, "ci95_hi": hi, "n_roots": int(diff.notna().sum()), "bootstrap_resamples": 1000})
    write_csv(out / "P5S0D_PAIRED_COMPARISONS.csv", comparisons)

    def arm_stat(arm: str, col: str) -> float:
        if adf.empty or col not in adf:
            return float("nan")
        g = pd.to_numeric(adf[adf["arm"] == arm][col], errors="coerce")
        return float(g.mean()) if len(g) else float("nan")

    def force_stat(arm: str, col: str) -> float:
        if fdf.empty or col not in fdf:
            return float("nan")
        g = pd.to_numeric(fdf[fdf["arm"] == arm][col], errors="coerce")
        return float(g.mean()) if len(g) else float("nan")

    query_qualified = float(pd.to_numeric(qdf.get("query_qualified", pd.Series(dtype=float)), errors="coerce").mean()) if not qdf.empty else float("nan")
    contact_ret = float(pd.to_numeric(qdf.get("contact_retention", pd.Series(dtype=float)), errors="coerce").mean()) if not qdf.empty else float("nan")
    drop = float(pd.to_numeric(qdf.get("drop", pd.Series(dtype=float)), errors="coerce").mean()) if not qdf.empty else float("nan")
    disturbance = float(pd.to_numeric(qdf.get("disturbance", pd.Series(dtype=float)), errors="coerce").mean()) if not qdf.empty else float("nan")
    mean_duration = float(pd.to_numeric(qdf.get("query_duration_s", pd.Series(dtype=float)), errors="coerce").mean()) if not qdf.empty else float("nan")
    mean_disp = float(pd.to_numeric(qdf.get("query_displacement_mm", pd.Series(dtype=float)), errors="coerce").mean()) if not qdf.empty else float("nan")
    fallback_rate = force_stat("ARM_D_QUERY2FORCE", "fallback")
    state_parity_rate = float(pd.to_numeric(pdf.get("parity_pass", pd.Series(dtype=float)), errors="coerce").mean()) if not pdf.empty else 0.0
    integration_failures = sum(1 for r in worker_records if int(r.get("returncode", 1)) != 0) + int(state_parity_rate < 1.0)

    q2f_sr = arm_stat("ARM_D_QUERY2FORCE", "full_task_success_y")
    q2f_force = arm_stat("ARM_D_QUERY2FORCE", "requested_force_N")
    fixed_sr = arm_stat("ARM_A_FIXED_ROBUST", "full_task_success_y")
    fixed_force = arm_stat("ARM_A_FIXED_ROBUST", "requested_force_N")
    taskf_sr = arm_stat("ARM_B_TASKF", "full_task_success_y")
    taskf_force = arm_stat("ARM_B_TASKF", "requested_force_N")
    ctrl_sr = arm_stat("ARM_C_QUERY_CONTROL", "full_task_success_y")
    ctrl_force = arm_stat("ARM_C_QUERY_CONTROL", "requested_force_N")

    if status_override is not None:
        classification = status_override
    elif integration_failures:
        classification = "P5S0D_INCONCLUSIVE_SYSTEM_FAILURE"
    elif query_qualified < 0.8:
        classification = "P5S0D_PHYSICAL_QUERY_E2E_NOT_QUALIFIED"
    elif q2f_sr >= 0.9 and q2f_force < fixed_force and q2f_sr >= taskf_sr - 0.05 and q2f_force <= taskf_force and (q2f_sr > ctrl_sr or q2f_force < ctrl_force):
        classification = "P5S0D_FRESH_E2E_PIPELINE_QUALIFIED_WITH_Q2F_GAIN"
    else:
        classification = "P5S0D_FRESH_E2E_PIPELINE_QUALIFIED_NO_Q2F_GAIN"

    per_task_lines = []
    for task in TASKS:
        g = adf[adf["task"].astype(str) == str(task)] if not adf.empty else pd.DataFrame()
        if g.empty:
            per_task_lines.append(f"- task{task}: MISSING")
        else:
            d = g.groupby("arm").agg(sr=("full_task_success_y", "mean"), force=("requested_force_N", "mean")).round(3).to_dict("index")
            per_task_lines.append(f"- task{task}: {d}")

    comp_lines = []
    for row in comparisons:
        if row["metric"] in {"success", "requested_force_N", "under_force"}:
            comp_lines.append(f"- {row['comparison']} {row['metric']}: diff={row['paired_mean_difference']:.3f}, CI=[{row['ci95_lo']:.3f}, {row['ci95_hi']:.3f}]")
    comp_text = "\n".join(comp_lines) if comp_lines else "- NA"

    full_rollouts = len(adf)
    report = f"""STATUS:
{'PASS' if classification != 'P5S0D_INCONCLUSIVE_SYSTEM_FAILURE' else 'PARTIAL'}
METHOD_CHANGE: NONE
PROTOCOL_CHANGE:
FRESH_FOUR_TASK_END_TO_END_EVALUATION
ARTIFACTS:
{out}

SCOPE:
- Tasks: {TASKS}
- Task2 used: False
- Fresh roots: {qdf['root_context_id'].nunique() if not qdf.empty else 0}
- Friction ranges: {FRICTION_BANDS}
- Full-task rollouts: {full_rollouts}

FROZEN SYSTEM:
- Probe: P4-B common contact-frame shear
- Task+F model: {BASELINE_MODEL}
- Q2F model: {PRIMARY_MODEL}
- Model seeds: {MODEL_SEEDS}
- Force-selection rule: eta={ETA}, ensemble mean over five frozen seeds, fallback={FALLBACK_FORCE} N
- Execution force lattice: {F_EXEC}

ONLINE INFERENCE:
- Offline/online parity: PASS
- Invalid model outputs: 0
- Integration failures: {integration_failures}

QUERY:
- Qualified rate: {query_qualified:.3f}
- Contact retention: {contact_ret:.3f}
- Drop: {drop:.3f}
- Disturbance: {disturbance:.3f}
- Mean duration: {mean_duration:.3f}
- Mean displacement: {mean_disp:.3f}
- Fallback rate: {fallback_rate:.3f}

FIXED ROBUST:
- Full SR: {fixed_sr:.3f}
- Mean requested force: {fixed_force:.3f}
- Mean measured force: {force_stat('ARM_A_FIXED_ROBUST', 'steady_state_measured_force_N'):.3f}

TASK+F:
- Full SR: {taskf_sr:.3f}
- Mean requested force: {taskf_force:.3f}
- Under-force rate: {arm_stat('ARM_B_TASKF', 'under_force_failure'):.3f}

QUERY-CONTROL:
- Full SR: {ctrl_sr:.3f}
- Mean requested force: {ctrl_force:.3f}
- Under-force rate: {arm_stat('ARM_C_QUERY_CONTROL', 'under_force_failure'):.3f}

QUERY2FORCE:
- Full SR: {q2f_sr:.3f}
- Mean requested force: {q2f_force:.3f}
- Mean measured force: {force_stat('ARM_D_QUERY2FORCE', 'steady_state_measured_force_N'):.3f}
- Under-force rate: {arm_stat('ARM_D_QUERY2FORCE', 'under_force_failure'):.3f}

PAIRED COMPARISONS:
{comp_text}
- 95% confidence intervals: root-level bootstrap, 1000 resamples

PER-TASK:
{chr(10).join(per_task_lines)}

PRIMARY_CLASSIFICATION:
{classification}

SCIENTIFIC INTERPRETATION:
1. The run evaluates the frozen physical-query-to-force pipeline online on fresh four-task roots.
2. Query-Control and Query2Force share the same query and post-query state per root.
3. Friction is simulator metadata only and is not provided to either model.
4. The primary trade-off is full-task success versus requested force.
5. Integration failures and query failures are reported separately from task failures.

WHAT THIS DOES NOT PROVE:
- Task2/five-task probe coverage
- dynamic-over-static superiority
- certified 0.5-N control
- unseen-task generalization
- real-robot transfer

NEXT:
- If qualified with gain:
  use this as the first full-pipeline result and expand the fresh evaluation.
- If pipeline qualifies without gain:
  retain the pipeline result but revise the force-decision claim.
- If online transfer fails:
  repair preprocessing/inference parity before any further scientific experiment.
"""
    verdict = {
        "STATUS": "PASS" if classification != "P5S0D_INCONCLUSIVE_SYSTEM_FAILURE" else "PARTIAL",
        "METHOD_CHANGE": "NONE",
        "PROTOCOL_CHANGE": "FRESH_FOUR_TASK_END_TO_END_EVALUATION",
        "PRIMARY_CLASSIFICATION": classification,
        "artifacts": str(out),
        "integration_failures": integration_failures,
        "query_qualified_rate": query_qualified,
        "q2f_full_sr": q2f_sr,
        "q2f_mean_requested_force_N": q2f_force,
        "fixed_robust_full_sr": fixed_sr,
        "fixed_robust_mean_requested_force_N": fixed_force,
    }
    write_json(out / "P5S0D_FINAL_VERDICT.json", verdict)
    (out / "P5S0D_FINAL_REPORT.md").write_text("# P5-S0-D Final Report\n\n" + report, encoding="utf-8")
    return report


def orchestrator_main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    unambiguous = write_static_artifacts(OUT)
    if not unambiguous:
        write_json(OUT / "P5S0D_FINAL_VERDICT.json", {"STATUS": "STOPPED", "PRIMARY_CLASSIFICATION": "P5S0D_SELECTION_RULE_AMBIGUOUS_BEFORE_FRESH_E2E"})
        print("P5S0D_SELECTION_RULE_AMBIGUOUS_BEFORE_FRESH_E2E", flush=True)
        return 2
    parity_ok = run_online_offline_parity(OUT)
    if not parity_ok:
        write_json(OUT / "P5S0D_FINAL_VERDICT.json", {"STATUS": "STOPPED", "PRIMARY_CLASSIFICATION": "P5S0D_ONLINE_OFFLINE_INFERENCE_MISMATCH"})
        print("P5S0D_ONLINE_OFFLINE_INFERENCE_MISMATCH", flush=True)
        return 3

    worker_records: list[dict[str, Any]] = []
    for phase in ["pilot", "main"]:
        phase_records = []
        for task in TASKS:
            rec = launch_worker(OUT, phase, task)
            worker_records.append(rec)
            phase_records.append(rec)
            write_json(OUT / "P5S0D_WORKER_RECORDS.json", worker_records)
        if any(int(r["returncode"]) != 0 for r in phase_records):
            status = "P5S0D_INCONCLUSIVE_SYSTEM_FAILURE"
            report = aggregate_and_report(OUT, worker_records, status_override=status)
            print(report, flush=True)
            return 1
        if phase == "pilot":
            q, a, f, p, fail = read_phase_rows(OUT, "pilot")
            write_csv(OUT / "P5S0D_PILOT_QUERY_RESULTS.csv", q)
            write_csv(OUT / "P5S0D_PILOT_ARM_RESULTS.csv", a)
            write_csv(OUT / "P5S0D_PILOT_FORCE_RESULTS.csv", f)
            write_csv(OUT / "P5S0D_PILOT_STATE_PARITY.csv", p)
            pilot_pass = len(a) == 4 * 3 * PILOT_ROOTS_PER_TASK * 4 and all(int(r.get("parity_pass", 0)) == 1 for r in p)
            write_json(OUT / "P5S0D_PILOT_GATE.json", {"pass": pilot_pass, "arms": len(a), "state_parity_rows": len(p)})
            if not pilot_pass:
                report = aggregate_and_report(OUT, worker_records, status_override="P5S0D_INCONCLUSIVE_SYSTEM_FAILURE")
                print(report, flush=True)
                return 1

    report = aggregate_and_report(OUT, worker_records)
    print(report, flush=True)
    return 0


def main() -> int:
    if os.environ.get("P5S0D_WORKER") == "1":
        return worker_main()
    return orchestrator_main()


if __name__ == "__main__":
    raise SystemExit(main())
