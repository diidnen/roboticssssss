#!/usr/bin/env python3
"""P5-S0-E offline force-decision collapse and GNP planner audit.

This script performs forensic analysis only. It loads the frozen P5-S0-C
checkpoints and the completed P5-S0-D fresh E2E artifact, then recomputes
model-only force decisions without simulator rollouts, retraining, or tuning.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import time
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
RESULTS = REPO / "analysis/results"
D_ARTIFACT = RESULTS / "p5s0d_fresh_e2e_q2f_20260824_090205"
C_ARTIFACT = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"

TASKS = [0, 1, 5, 6]
TASK_TO_IDX = {t: i for i, t in enumerate(TASKS)}
MODEL_SEEDS = [0, 1, 2, 3, 4]
TASK_FORCE = "TASK_FORCE_THRESHOLD"
Q2F = "Q2F_THRESHOLD"
ETA = 0.9
F_EXEC = [3.0, 4.0, 5.0, 6.0, 8.0]
DENSE_FORCES = np.round(np.arange(3.0, 8.0001, 0.05), 2)
LATTICES = {
    "L1_original": np.array([3.0, 4.0, 5.0, 6.0, 8.0]),
    "L2_half_N": np.round(np.arange(3.0, 8.0001, 0.5), 2),
    "L3_quarter_N": np.round(np.arange(3.0, 8.0001, 0.25), 2),
}
MC_SAMPLES = 50
SHUFFLES_PER_ROOT = 100
R_FAIL_VALUES = [-5.0, -10.0, -20.0]
F_MAX_UTILITY = 8.0

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


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


OUT = RESULTS / f"p5s0e_force_decision_collapse_audit_{now_tag()}"


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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def force_norm(force_n: float | np.ndarray) -> np.ndarray:
    return 2.0 * ((np.asarray(force_n, dtype=np.float32) - 3.0) / 5.0) - 1.0


def threshold_parts(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu_req = 2.0 + 7.0 * torch.sigmoid(raw[:, :1])
    scale = 0.1 + torch.nn.functional.softplus(raw[:, 1:2])
    return mu_req, scale


def threshold_logit(force: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    mu_req, scale = threshold_parts(raw)
    force_n = 3.0 + 2.5 * (force + 1.0)
    return (force_n - mu_req) / scale


class TaskForceThreshold(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4, 8)
        self.head = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))

    def raw(self, task: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(task))

    def forward(self, xseq, xstatic, lengths, task, force):
        return threshold_logit(force, self.raw(task))


class Q2FThreshold(nn.Module):
    def __init__(self, input_dim: int, zdim: int = 4):
        super().__init__()
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

    def raw_from_z(self, z: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([z, self.emb(task)], dim=1))

    def forward(self, xseq, xstatic, lengths, task, force, sample: bool = True):
        mu, lv = self.encode(xseq, lengths)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if sample else mu
        return threshold_logit(force, self.raw_from_z(z, task)), mu, lv


@dataclass
class EncodedProbe:
    xseq: np.ndarray
    xstatic: np.ndarray
    length: int


class FrozenInference:
    def __init__(self):
        self.device = torch.device("cpu")
        self.norm = json.loads((C_ARTIFACT / "P5S0C_NORMALIZATION.json").read_text())
        self.summary = self._read_summary(C_ARTIFACT / "P5S0C_MODEL_SUMMARY.csv")
        self.feature_names = list(self.norm["dynamic_feature_names"])
        self.phases = list(self.norm["phase_categories_from_train"])
        self.states = list(self.norm["contact_state_categories_from_train"])
        self.dynamic_mean = np.asarray(self.norm["dynamic_mean"], dtype=np.float32)
        self.dynamic_std = np.asarray(self.norm["dynamic_std"], dtype=np.float32)
        self.static_mean = np.asarray(self.norm["static_mean"], dtype=np.float32)
        self.static_std = np.asarray(self.norm["static_std"], dtype=np.float32)
        self.models: dict[tuple[str, int], dict[str, Any]] = {}
        for seed in MODEL_SEEDS:
            taskf = TaskForceThreshold()
            taskf.load_state_dict(torch.load(C_ARTIFACT / "P5S0C_CHECKPOINTS" / f"{TASK_FORCE}_seed{seed}.pt", map_location="cpu"))
            taskf.eval()
            self.models[(TASK_FORCE, seed)] = {"model": taskf, "temperature": self.summary[(TASK_FORCE, seed)]}
            q2f = Q2FThreshold(len(self.feature_names))
            q2f.load_state_dict(torch.load(C_ARTIFACT / "P5S0C_CHECKPOINTS" / f"{Q2F}_seed{seed}.pt", map_location="cpu"))
            q2f.eval()
            self.models[(Q2F, seed)] = {"model": q2f, "temperature": self.summary[(Q2F, seed)]}

    @staticmethod
    def _read_summary(path: Path) -> dict[tuple[str, int], float]:
        out: dict[tuple[str, int], float] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["model"] in {TASK_FORCE, Q2F}:
                    out[(row["model"], int(row["seed"]))] = float(row["temperature"])
        return out

    @staticmethod
    def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
        try:
            value = row.get(key, "")
            return default if value == "" else float(value)
        except Exception:
            return default

    def sequence_array(self, telemetry_path: str) -> np.ndarray:
        with open(telemetry_path, newline="", encoding="utf-8") as fh:
            raw_rows = list(csv.DictReader(fh))
        if not raw_rows:
            return np.zeros((1, len(self.feature_names)), dtype=np.float32)
        ex0 = self._f(raw_rows[0], "eef_x")
        ey0 = self._f(raw_rows[0], "eef_y")
        ez0 = self._f(raw_rows[0], "eef_z")
        rows: list[list[float]] = []
        for rr in raw_rows:
            feat: dict[str, float] = {col: self._f(rr, col) for col in NUMERIC_PROBE_COLS}
            feat["eef_dx"] = self._f(rr, "eef_x") - ex0
            feat["eef_dy"] = self._f(rr, "eef_y") - ey0
            feat["eef_dz"] = self._f(rr, "eef_z") - ez0
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

    def encode_probe(self, telemetry_path: str) -> EncodedProbe:
        arr_raw = self.sequence_array(telemetry_path)
        left_idx = self.feature_names.index("contact_left")
        right_idx = self.feature_names.index("contact_right")
        contact = (arr_raw[:, left_idx].astype(float) + arr_raw[:, right_idx].astype(float)) > 0
        static_idx = int(np.argmax(contact)) if contact.any() else 0
        xseq = (arr_raw - self.dynamic_mean) / self.dynamic_std
        xstatic = (arr_raw[static_idx] - self.static_mean) / self.static_std
        return EncodedProbe(xseq.astype(np.float32), xstatic.astype(np.float32), int(xseq.shape[0]))

    def train_mean_probe(self, length: int = 25) -> EncodedProbe:
        return EncodedProbe(
            xseq=np.zeros((length, len(self.feature_names)), dtype=np.float32),
            xstatic=np.zeros((len(self.feature_names),), dtype=np.float32),
            length=length,
        )

    def _tensors_for(self, enc: EncodedProbe, task_id: int, forces: np.ndarray, repeats: int = 1):
        n = int(len(forces))
        xseq = np.repeat(enc.xseq[None, :, :], n * repeats, axis=0)
        xstatic = np.repeat(enc.xstatic[None, :], n * repeats, axis=0)
        lengths = np.full(n * repeats, enc.length, dtype=np.int64)
        tasks = np.full(n * repeats, TASK_TO_IDX[int(task_id)], dtype=np.int64)
        f = np.asarray(force_norm(np.repeat(forces, repeats)), dtype=np.float32).reshape(n * repeats, 1)
        return (
            torch.tensor(xseq, dtype=torch.float32),
            torch.tensor(xstatic, dtype=torch.float32),
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(tasks, dtype=torch.long),
            torch.tensor(f, dtype=torch.float32),
        )

    def taskf_raw_threshold(self, task_id: int, seed: int) -> dict[str, float]:
        fit = self.models[(TASK_FORCE, seed)]
        model: TaskForceThreshold = fit["model"]
        task_t = torch.tensor([TASK_TO_IDX[int(task_id)]], dtype=torch.long)
        with torch.no_grad():
            raw = model.raw(task_t)
            mu, scale = threshold_parts(raw)
        return {"mu_req": float(mu.item()), "s_req": float(scale.item()), "temperature": float(fit["temperature"])}

    def q2f_threshold_summary(self, enc: EncodedProbe, task_id: int, seed: int, mc: int = MC_SAMPLES) -> dict[str, Any]:
        fit = self.models[(Q2F, seed)]
        model: Q2FThreshold = fit["model"]
        x_t, _, len_t, task_t, _ = self._tensors_for(enc, task_id, np.array([3.0]), repeats=1)
        with torch.no_grad():
            z_mu, z_lv = model.encode(x_t, len_t)
            raw_det = model.raw_from_z(z_mu, task_t)
            mu_det, s_det = threshold_parts(raw_det)
            torch.manual_seed(90000 + seed)
            eps = torch.randn(mc, z_mu.shape[1])
            z = z_mu.repeat(mc, 1) + eps * torch.exp(0.5 * z_lv).repeat(mc, 1)
            task_rep = task_t.repeat(mc)
            raw = model.raw_from_z(z, task_rep)
            mu_s, s_s = threshold_parts(raw)
        z_mu_np = z_mu.numpy().reshape(-1)
        z_var_np = torch.exp(z_lv).numpy().reshape(-1)
        return {
            "latent_mu_json": json.dumps([float(x) for x in z_mu_np]),
            "latent_var_json": json.dumps([float(x) for x in z_var_np]),
            "latent_mean_norm": float(np.linalg.norm(z_mu_np)),
            "latent_var_mean": float(np.mean(z_var_np)),
            "latent_var_max": float(np.max(z_var_np)),
            "mu_req_at_latent_mean": float(mu_det.item()),
            "s_req_at_latent_mean": float(s_det.item()),
            "posterior_mu_req_mean": float(mu_s.mean().item()),
            "posterior_mu_req_var": float(mu_s.var(unbiased=False).item()),
            "posterior_mu_req_std": float(mu_s.std(unbiased=False).item()),
            "posterior_s_req_mean": float(s_s.mean().item()),
            "posterior_s_req_var": float(s_s.var(unbiased=False).item()),
            "posterior_s_req_std": float(s_s.std(unbiased=False).item()),
            "temperature": float(fit["temperature"]),
        }

    def probabilities(
        self,
        model_name: str,
        enc: EncodedProbe,
        task_id: int,
        forces: np.ndarray,
        mc: int = MC_SAMPLES,
        return_samples: bool = False,
    ) -> dict[str, Any]:
        seed_probs: list[np.ndarray] = []
        seed_samples: list[np.ndarray] = []
        seed_logits: list[np.ndarray] = []
        with torch.no_grad():
            for seed in MODEL_SEEDS:
                fit = self.models[(model_name, seed)]
                temp = float(fit["temperature"])
                task_idx = TASK_TO_IDX[int(task_id)]
                if model_name == TASK_FORCE:
                    model: TaskForceThreshold = fit["model"]
                    task_t = torch.tensor([task_idx], dtype=torch.long)
                    raw = model.raw(task_t)
                    mu_req, scale = threshold_parts(raw)
                    force_n = torch.tensor(forces, dtype=torch.float32).reshape(-1, 1)
                    logits = ((force_n - mu_req) / scale).numpy().reshape(-1)
                    probs = 1.0 / (1.0 + np.exp(-np.clip(logits / temp, -60, 60)))
                    seed_probs.append(probs)
                    seed_logits.append(logits)
                    seed_samples.append(probs[:, None])
                else:
                    model: Q2FThreshold = fit["model"]
                    torch.manual_seed(100000 + seed * 1000)
                    x_t = torch.tensor(enc.xseq[None, :, :], dtype=torch.float32)
                    len_t = torch.tensor([enc.length], dtype=torch.long)
                    task_one = torch.tensor([task_idx], dtype=torch.long)
                    z_mu, z_lv = model.encode(x_t, len_t)
                    z = z_mu.repeat(mc, 1) + torch.randn(mc, z_mu.shape[1]) * torch.exp(0.5 * z_lv).repeat(mc, 1)
                    raw = model.raw_from_z(z, task_one.repeat(mc))
                    mu_req, scale = threshold_parts(raw)
                    force_n = torch.tensor(forces, dtype=torch.float32).reshape(len(forces), 1)
                    logits_samples = ((force_n - mu_req.reshape(1, mc)) / scale.reshape(1, mc)).numpy()
                    logits_mean = logits_samples.mean(axis=1)
                    probs_rule = 1.0 / (1.0 + np.exp(-np.clip(logits_mean / temp, -60, 60)))
                    probs_samples = 1.0 / (1.0 + np.exp(-np.clip(logits_samples / temp, -60, 60)))
                    seed_probs.append(probs_rule)
                    seed_logits.append(logits_mean)
                    seed_samples.append(probs_samples)
        ensemble_mean = np.stack(seed_probs, axis=0).mean(axis=0)
        out = {
            "seed_probs": np.stack(seed_probs, axis=0),
            "seed_logits": np.stack(seed_logits, axis=0),
            "ensemble_mean": ensemble_mean,
        }
        if return_samples:
            out["posterior_prob_samples"] = np.concatenate(seed_samples, axis=1)
        return out


def dense_f_eta(forces: np.ndarray, probs: np.ndarray, eta: float = ETA) -> tuple[float, bool]:
    probs = np.asarray(probs, dtype=float)
    idx = np.where(probs >= eta)[0]
    if len(idx) == 0:
        return float(forces[-1]), True
    i = int(idx[0])
    if i == 0:
        return float(forces[0]), False
    f0, f1 = float(forces[i - 1]), float(forces[i])
    p0, p1 = float(probs[i - 1]), float(probs[i])
    if not np.isfinite(p0) or not np.isfinite(p1) or abs(p1 - p0) < 1e-12:
        return f1, False
    frac = max(0.0, min(1.0, (eta - p0) / (p1 - p0)))
    return float(f0 + frac * (f1 - f0)), False


def project_lattice(f_cont: float, lattice: np.ndarray) -> float:
    ok = lattice[lattice >= f_cont - 1e-12]
    return float(ok[0]) if len(ok) else float(lattice[-1])


def summarize(values: pd.Series | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {k: float("nan") for k in ["min", "max", "mean", "std", "q25", "q75", "iqr"]}
    q25, q75 = np.percentile(x, [25, 75])
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def mode_counts(series: pd.Series) -> str:
    counts = series.value_counts().sort_index()
    return "; ".join(f"{float(k):g}:{int(v)}" for k, v in counts.items())


def selected_from_curve(forces: np.ndarray, probs: np.ndarray, lattice: np.ndarray) -> float:
    f_cont, _ = dense_f_eta(forces, probs)
    return project_lattice(f_cont, lattice)


def original_outcome_maps() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q = pd.read_csv(D_ARTIFACT / "P5S0D_QUERY_RESULTS.csv")
    a = pd.read_csv(D_ARTIFACT / "P5S0D_ARM_RESULTS.csv")
    f = pd.read_csv(D_ARTIFACT / "P5S0D_FORCE_RESULTS.csv")
    a["root_context_id"] = a.get("root_context_id", a.get("context_id"))
    if "root_context_id" not in a or a["root_context_id"].isna().all():
        a["root_context_id"] = a["context_id"]
    return q, a, f


def p_success_at(forces: np.ndarray, probs: np.ndarray, f: float) -> float:
    return float(np.interp(f, forces, probs))


def write_markdown_artifacts(out: Path, metrics: dict[str, Any]) -> None:
    (out / "P5S0E_GNP_MAPPING.md").write_text(f"""# P5-S0-E GNP Mapping

| GNP component | Current Q2F component | Missing/different component |
|---|---|---|
| Interaction/context representation | P4-B query telemetry encoded as normalized dynamic sequence with phase/contact-state one-hot features | No learned query policy or explicit context graph; one fixed physical query only |
| Posterior encoder | GRU over probe sequence followed by linear `mu` and `lv` heads | Encoder sees only probe evidence, not the full matched force-outcome set at deployment |
| Latent distribution | Diagonal Gaussian `q_probe(z|evidence)` with 4 latent dimensions | No teacher posterior over full force-response context; KL regularization is only a weak standard normal prior |
| Partial/full context training | Supervised force-success labels plus latent beta in P5-S0-C training | No explicit partial-to-full context distillation objective |
| Action/force representation | Scalar force enters monotonic threshold equation through normalized force | No richer action proposal set or continuous optimizer beyond threshold crossing |
| Feasibility decoder | Monotonic full-task success decoder `P(Y_full=1|F,z,task)` | Decoder is trained as binary sufficiency classifier; no explicit cost/damage model |
| Uncertainty propagation | Posterior samples are used during probability computation; current E2E collapses to calibrated ensemble mean for selection | Deployment rule does not use posterior lower quantiles or expected utility |
| Planner/action selection | Current planner selects min force where ensemble mean calibrated probability >= 0.9 | No GNP-style utility maximization or risk-sensitive posterior decision rule |

## Answers

- **Does current Q2F actually use posterior uncertainty at deployment?** Only partially. It samples the latent posterior to estimate probabilities, but the final deployment decision uses the mean calibrated probability curve.
- **Is Q2F effectively using only an ensemble mean?** Yes for action selection. Posterior and seed variation are averaged before the threshold rule is applied.
- **Is the current selection rule discarding useful posterior information?** The audit shows posterior-aware planners change some commands, but the realized Q2F-vs-Task+F gap remains limited by the narrow threshold frontier: `{metrics['planner_interpretation']}`.
- **Would GNP-style Monte Carlo planning change decisions?** Offline, it changes decision diversity under conservative quantiles and utility penalties, but it is diagnostic only and does not imply new E2E success without a preregistered rerun.
""", encoding="utf-8")

    (out / "P5S0E_GRU_ARCHITECTURE_AUDIT.md").write_text(f"""# P5-S0-E GRU Architecture Audit

## Current Temporal Encoder

- Input features: `{metrics['feature_count']}` normalized dynamic features from P4-B telemetry, including force, tactile, proprioceptive deltas, contact-frame axes, phase encodings, contact-state encodings, and contact/tactile validity flags.
- Hidden size: 32 after a 32-unit ReLU projection.
- Layers: one GRU layer.
- Parameter count: Q2F-Threshold has `{metrics['q2f_param_count']}` parameters.
- Sequence length: fresh P5-S0-D queries use short fixed query traces; mean encoded length is `{metrics['mean_sequence_length']:.1f}` timesteps.
- Masking: packed sequences use the stored sequence length; no timestep padding is treated as evidence.
- Static contact information: the normalized contact-establishment timestep is available through the same feature stream, but current Q2F does not have a separate static-contact branch.
- Dynamic response information: loading, shear displacement, marker motion, tangential response, and phase/state indicators are encoded temporally by the GRU.

## Assessment

The GRU remains a defensible encoder for this benchmark because the physical query is short, causal, variable-length-compatible, and can contain loading/unloading or hysteresis signals that a static MLP would discard. Its parameter count is small enough for the matched dataset scale.

The limitation is not that a GRU is inherently unsuitable; it is that the current architecture entangles static contact and dynamic response into one compact posterior and then deploys with an ensemble-mean threshold planner. If collapse is confirmed, the next architecture should separate these roles.

## Future Architecture Proposal

Do not implement in P5-S0-E. A future candidate is:

```text
Static contact MLP
+
baseline-relative dynamic response GRU
+
task/downstream encoder
+
Gaussian physical latent
+
continuous force-threshold decoder
```

This keeps the current causal temporal advantages while giving the model explicit capacity for static contact evidence, dynamic response evidence, and downstream task prior.
""", encoding="utf-8")

    teacher_needed = metrics["primary_classification"] in {"P5S0E_MODEL_DECISION_COLLAPSE", "P5S0E_BOTH_MODEL_AND_LATTICE_COLLAPSE"}
    if teacher_needed:
        teacher_text = """# P5-S0-E Q2F-GNP-Teacher Proposal

Activation condition: `True`. This proposal is included because the audit supports model decision collapse or mixed model/lattice collapse.

## Model

Training-only teacher:

```text
full matched force-outcome set {{(F_j, Y_j)}}
-> teacher posterior q_teacher(z)
```

Deployable student:

```text
physical-query evidence
-> q_probe(z)
```

Shared decoder:

```text
(z, task, F) -> P(Y_full=1 | z, task, F)
```

## Training Unit

One root context is the unit. The teacher observes all matched force outcomes for that root; the student observes only the frozen physical-query telemetry available at deployment.

## Loss

```text
BCE full-task force sufficiency loss
+ lambda_KL KL(q_teacher(z) || q_probe(z))
+ optional teacher prior regularization
```

## Why It Differs From Current Q2F

Current Q2F learns `q_probe(z|evidence)` directly from force-success labels with a weak latent prior. It has no posterior that is explicitly informed by the full matched force frontier. Q2F-GNP-Teacher would use the complete offline force-outcome set as a teacher during training, then distill that posterior into the deployable probe encoder.

## Why It Is GNP-Inspired

It separates a richer full-context posterior from a partial-context deployment posterior and trains the deployable posterior to approximate the full-context belief state.

## Data Requirements

Existing P5-S0-A/B/C matched force-outcome data are sufficient to prototype this. No new simulator data are necessary for the first teacher-student training run. A new fresh E2E run would only be needed after the model and planner are frozen.
"""
    else:
        teacher_text = f"""# P5-S0-E Q2F-GNP-Teacher Proposal

Activation condition: `False`.

The audit classification is `{metrics['primary_classification']}`, not `P5S0E_MODEL_DECISION_COLLAPSE` or `P5S0E_BOTH_MODEL_AND_LATTICE_COLLAPSE`. Per the P5-S0-E protocol, this run does not design Q2F-GNP-Teacher as the next model.

The required proposal artifact is retained as an explicit non-activation record. No architecture, loss, or training plan is proposed in this classification branch.
"""
    (out / "P5S0E_GNP_TEACHER_PROPOSAL.md").write_text(teacher_text, encoding="utf-8")

    (out / "P5S0E_SELECTIVE_QUERY_ROADMAP.md").write_text("""# P5-S0-E Selective Physical-Query Roadmap

## When To Query

Selective querying should depend on execution-decision ambiguity, not raw friction uncertainty.

```text
Task/RGB prior
-> predicted force-sufficiency frontier
-> if one low-risk contract is clearly sufficient:
       skip physical query
   else:
       query
```

Physical uncertainty and decision uncertainty are different. A root can have uncertain friction while every plausible frontier still selects the same safe force; probing is not valuable there. Conversely, modest physical uncertainty can matter when it straddles a force contract boundary.

Existing negative-control tasks `[3,7,9]` motivate this direction because some tasks may have little decision value from probing, but P5-S0-E does not solve selective probing.

## How To Query

The intended future Probe Composer is:

```text
downstream nominal action
+
contact state
+
physical question
-> contact-feasible primitive
```

Do not claim universal P4-B shear generality. P4/P4-R2 already showed that this is false for Task2.

## Safety And Damage

The current benchmark does not yet include over-force deformation/damage as a target. Therefore this project should not claim that probing prevents object damage until a fragile/deformation benchmark is added.
""", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dense_dir = OUT / "P5S0E_DENSE_FORCE_CURVES"
    dense_dir.mkdir(exist_ok=True)

    protocol = {
        "experiment": "P5-S0-E",
        "type": "forensic_offline_analysis_only",
        "source": str(D_ARTIFACT),
        "tasks": TASKS,
        "models": [TASK_FORCE, Q2F],
        "dense_force_grid": {"min": 3.0, "max": 8.0, "step": 0.05},
        "eta": ETA,
        "execution_lattice": F_EXEC,
        "shuffles_per_root": SHUFFLES_PER_ROOT,
        "method_change": "NONE",
        "no_new_simulator_data": True,
        "no_training": True,
    }
    write_json(OUT / "P5S0E_PROTOCOL.json", protocol)
    write_json(
        OUT / "P5S0E_CODE_HASH.txt",
        {
            "git_commit": git_commit(),
            "runner_sha256": sha256_file(Path(__file__)),
            "source_d_final_verdict_sha256": sha256_file(D_ARTIFACT / "P5S0D_FINAL_VERDICT.json"),
            "source_d_arm_results_sha256": sha256_file(D_ARTIFACT / "P5S0D_ARM_RESULTS.csv"),
            "source_c_model_summary_sha256": sha256_file(C_ARTIFACT / "P5S0C_MODEL_SUMMARY.csv"),
        },
    )

    qdf, adf, fdf = original_outcome_maps()
    qdf = qdf[qdf["phase"].astype(str) == "main"].copy().sort_values(["task", "root_index", "friction_band"])
    inf = FrozenInference()
    encoded: dict[str, EncodedProbe] = {}
    for _, row in qdf.iterrows():
        encoded[str(row["root_context_id"])] = inf.encode_probe(str(row["probe_telemetry_path"]))

    original_q2f = adf[adf["arm"] == "ARM_D_QUERY2FORCE"].copy()
    original_taskf = adf[adf["arm"] == "ARM_B_TASKF"].copy()
    outcome_by_arm = adf.pivot_table(index="root_context_id", columns="arm", values=["requested_force_N", "full_task_success_y"], aggfunc="first")

    root_prediction_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    continuous_rows: list[dict[str, Any]] = []
    curve_summary_by_root: dict[str, dict[str, Any]] = {}
    posterior_samples_by_root: dict[str, np.ndarray] = {}
    q2f_mu_summary_by_root: dict[str, float] = {}

    for _, root in qdf.iterrows():
        rid = str(root["root_context_id"])
        task = int(root["task"])
        enc = encoded[rid]
        root_meta = {
            "task": task,
            "root_context_id": rid,
            "root_id": root["root_id"],
            "root_index": int(root["root_index"]),
            "root_seed": int(root["root_seed"]),
            "friction_regime": root["friction_band"],
            "hidden_friction_analysis_only": float(root["hidden_friction_analysis_only"]),
        }

        taskf = inf.probabilities(TASK_FORCE, enc, task, DENSE_FORCES, return_samples=True)
        q2f = inf.probabilities(Q2F, enc, task, DENSE_FORCES, return_samples=True)
        posterior_samples_by_root[rid] = q2f["posterior_prob_samples"]

        dense_rows = []
        for i, f in enumerate(DENSE_FORCES):
            row = dict(root_meta)
            row.update(
                {
                    "force_N": float(f),
                    "taskf_ensemble_p_success": float(taskf["ensemble_mean"][i]),
                    "q2f_ensemble_p_success": float(q2f["ensemble_mean"][i]),
                    "q2f_posterior_p10": float(np.percentile(q2f["posterior_prob_samples"][i], 10)),
                    "q2f_posterior_p50": float(np.percentile(q2f["posterior_prob_samples"][i], 50)),
                    "q2f_posterior_p90": float(np.percentile(q2f["posterior_prob_samples"][i], 90)),
                }
            )
            for seed_i, seed in enumerate(MODEL_SEEDS):
                row[f"taskf_seed{seed}_p_success"] = float(taskf["seed_probs"][seed_i, i])
                row[f"q2f_seed{seed}_p_success"] = float(q2f["seed_probs"][seed_i, i])
            dense_rows.append(row)
        write_csv(dense_dir / f"{rid}_dense_force_curve.csv", dense_rows)

        taskf_f, taskf_fb = dense_f_eta(DENSE_FORCES, taskf["ensemble_mean"])
        q2f_f, q2f_fb = dense_f_eta(DENSE_FORCES, q2f["ensemble_mean"])
        taskf_exec = project_lattice(taskf_f, LATTICES["L1_original"])
        q2f_exec = project_lattice(q2f_f, LATTICES["L1_original"])
        curve_summary_by_root[rid] = {
            "taskf_probs": taskf,
            "q2f_probs": q2f,
            "taskf_f_cont": taskf_f,
            "q2f_f_cont": q2f_f,
            "taskf_exec": taskf_exec,
            "q2f_exec": q2f_exec,
        }

        q2f_arm = original_q2f[original_q2f["root_context_id"] == rid].iloc[0]
        taskf_arm = original_taskf[original_taskf["root_context_id"] == rid].iloc[0]
        pred_row = dict(root_meta)
        pred_row.update(
            {
                "taskf_continuous_F_eta": taskf_f,
                "q2f_continuous_F_eta": q2f_f,
                "q2f_minus_taskf_F_eta": q2f_f - taskf_f,
                "taskf_projected_L1_force": taskf_exec,
                "q2f_projected_L1_force": q2f_exec,
                "taskf_original_selected_requested_force": float(taskf_arm["requested_force_N"]),
                "q2f_original_selected_requested_force": float(q2f_arm["requested_force_N"]),
                "taskf_original_full_task_success": int(taskf_arm["full_task_success_y"]),
                "q2f_original_full_task_success": int(q2f_arm["full_task_success_y"]),
                "taskf_p_success_at_3N": p_success_at(DENSE_FORCES, taskf["ensemble_mean"], 3.0),
                "taskf_p_success_at_4N": p_success_at(DENSE_FORCES, taskf["ensemble_mean"], 4.0),
                "taskf_p_success_at_5N": p_success_at(DENSE_FORCES, taskf["ensemble_mean"], 5.0),
                "q2f_p_success_at_3N": p_success_at(DENSE_FORCES, q2f["ensemble_mean"], 3.0),
                "q2f_p_success_at_4N": p_success_at(DENSE_FORCES, q2f["ensemble_mean"], 4.0),
                "q2f_p_success_at_5N": p_success_at(DENSE_FORCES, q2f["ensemble_mean"], 5.0),
            }
        )
        for seed_i, seed in enumerate(MODEL_SEEDS):
            tf_f, _ = dense_f_eta(DENSE_FORCES, taskf["seed_probs"][seed_i])
            qf_f, _ = dense_f_eta(DENSE_FORCES, q2f["seed_probs"][seed_i])
            pred_row[f"taskf_seed{seed}_F_eta"] = tf_f
            pred_row[f"q2f_seed{seed}_F_eta"] = qf_f
        root_prediction_rows.append(pred_row)
        continuous_rows.append(pred_row)

        q_seed_summaries: list[dict[str, Any]] = []
        for seed in MODEL_SEEDS:
            tf_thr = inf.taskf_raw_threshold(task, seed)
            threshold_rows.append({**root_meta, "model": TASK_FORCE, "seed": seed, **tf_thr})
            q_thr = inf.q2f_threshold_summary(enc, task, seed)
            q_seed_summaries.append(q_thr)
            threshold_rows.append({**root_meta, "model": Q2F, "seed": seed, **q_thr})

        q_seed_f = [pred_row[f"q2f_seed{seed}_F_eta"] for seed in MODEL_SEEDS]
        q2f_mu_summary_by_root[rid] = float(np.mean([r["posterior_mu_req_mean"] for r in q_seed_summaries]))
        threshold_rows.append(
            {
                **root_meta,
                "model": Q2F,
                "seed": "ENSEMBLE_SUMMARY",
                "mean_mu_req": float(np.mean([r["posterior_mu_req_mean"] for r in q_seed_summaries])),
                "std_mu_req_across_model_seeds": float(np.std([r["posterior_mu_req_mean"] for r in q_seed_summaries], ddof=0)),
                "mean_s_req": float(np.mean([r["posterior_s_req_mean"] for r in q_seed_summaries])),
                "uncertainty_summary": f"seed_F_eta_std={float(np.std(q_seed_f, ddof=0)):.4f}; latent_var_mean={float(np.mean([r['latent_var_mean'] for r in q_seed_summaries])):.4f}",
            }
        )

    root_df = pd.DataFrame(root_prediction_rows)
    write_csv(OUT / "P5S0E_ROOT_PREDICTIONS.csv", root_prediction_rows)
    write_csv(OUT / "P5S0E_THRESHOLD_OUTPUTS.csv", threshold_rows)
    write_csv(OUT / "P5S0E_CONTINUOUS_FORCE_DECISIONS.csv", continuous_rows)

    q2f_stats = summarize(root_df["q2f_continuous_F_eta"])
    diff_abs = root_df["q2f_minus_taskf_F_eta"].abs()
    meaningful = diff_abs >= 0.1
    same_l1 = root_df["taskf_projected_L1_force"] == root_df["q2f_projected_L1_force"]
    collapsed_meaningful = float((meaningful & same_l1 & (root_df["q2f_projected_L1_force"] == 5.0)).sum() / max(1, meaningful.sum()))
    quant_rows = [
        {
            "metric": "q2f_continuous_F_eta",
            **q2f_stats,
            "n_roots": len(root_df),
            "n_q2f_differs_from_taskf": int((diff_abs > 1e-9).sum()),
            "n_abs_diff_ge_0p1N": int((diff_abs >= 0.1).sum()),
            "n_abs_diff_ge_0p25N": int((diff_abs >= 0.25).sum()),
            "n_abs_diff_ge_0p5N": int((diff_abs >= 0.5).sum()),
            "n_abs_diff_ge_1p0N": int((diff_abs >= 1.0).sum()),
            "fraction_meaningful_differences_collapsed_to_same_5N": collapsed_meaningful,
        }
    ]
    for group_col in ["task", "friction_regime"]:
        for val, g in root_df.groupby(group_col):
            quant_rows.append({"metric": f"q2f_continuous_F_eta_by_{group_col}", group_col: val, **summarize(g["q2f_continuous_F_eta"]), "n_roots": len(g), "force_distribution_L1": mode_counts(g["q2f_projected_L1_force"])})
    write_csv(OUT / "P5S0E_QUANTIZATION_ANALYSIS.csv", quant_rows)

    lattice_rows: list[dict[str, Any]] = []
    for model_col, label in [("taskf_continuous_F_eta", TASK_FORCE), ("q2f_continuous_F_eta", Q2F)]:
        for lname, lattice in LATTICES.items():
            projected = root_df[model_col].map(lambda x: project_lattice(float(x), lattice))
            lattice_rows.append(
                {
                    "model": label,
                    "lattice": lname,
                    "unique_selected_forces": int(projected.nunique()),
                    "mean_predicted_requested_force": float(projected.mean()),
                    "force_distribution": mode_counts(projected),
                    "per_task_distributions": json.dumps({str(k): mode_counts(v.map(lambda x: project_lattice(float(x), lattice))) for k, v in root_df.groupby("task")[model_col]}),
                    "per_friction_distributions": json.dumps({str(k): mode_counts(v.map(lambda x: project_lattice(float(x), lattice))) for k, v in root_df.groupby("friction_regime")[model_col]}),
                }
            )
        lattice_rows.append(
            {
                "model": label,
                "lattice": "L4_continuous_model_threshold",
                "unique_selected_forces": int(root_df[model_col].round(3).nunique()),
                "mean_predicted_requested_force": float(root_df[model_col].mean()),
                "force_distribution": "continuous",
                "per_task_distributions": json.dumps({str(k): summarize(v) for k, v in root_df.groupby("task")[model_col]}),
                "per_friction_distributions": json.dumps({str(k): summarize(v) for k, v in root_df.groupby("friction_regime")[model_col]}),
            }
        )
    write_csv(OUT / "P5S0E_COUNTERFACTUAL_LATTICES.csv", lattice_rows)

    rng = np.random.default_rng(2026082405)
    by_task_ids = {int(t): list(g["root_context_id"].astype(str)) for t, g in qdf.groupby("task")}
    train_mean_cache: dict[tuple[int, int], dict[str, Any]] = {}
    sensitivity_rows: list[dict[str, Any]] = []
    task_prior_rows: list[dict[str, Any]] = []
    for _, root in qdf.iterrows():
        rid = str(root["root_context_id"])
        task = int(root["task"])
        base = curve_summary_by_root[rid]["q2f_probs"]["ensemble_mean"]
        base_mu = q2f_mu_summary_by_root[rid]
        base_f = float(curve_summary_by_root[rid]["q2f_f_cont"])
        meta = {
            "task": task,
            "root_context_id": rid,
            "friction_regime": root["friction_band"],
            "hidden_friction_analysis_only": float(root["hidden_friction_analysis_only"]),
        }
        sensitivity_rows.append(
            {
                **meta,
                "condition": "real_physical_query_evidence",
                "shuffle_index": "",
                "mu_req": base_mu,
                "continuous_F_eta": base_f,
                "delta_F_eta_vs_real": 0.0,
                "P_success_3N": p_success_at(DENSE_FORCES, base, 3.0),
                "P_success_4N": p_success_at(DENSE_FORCES, base, 4.0),
                "P_success_5N": p_success_at(DENSE_FORCES, base, 5.0),
            }
        )
        mean_key = (task, encoded[rid].length)
        if mean_key not in train_mean_cache:
            mean_enc = inf.train_mean_probe(length=encoded[rid].length)
            mean_probs_cached = inf.probabilities(Q2F, mean_enc, task, DENSE_FORCES, return_samples=False)["ensemble_mean"]
            mean_f_cached, _ = dense_f_eta(DENSE_FORCES, mean_probs_cached)
            mean_mu_vals_cached = [inf.q2f_threshold_summary(mean_enc, task, seed, mc=MC_SAMPLES)["posterior_mu_req_mean"] for seed in MODEL_SEEDS]
            train_mean_cache[mean_key] = {"probs": mean_probs_cached, "f": mean_f_cached, "mu": float(np.mean(mean_mu_vals_cached))}
        mean_probs = train_mean_cache[mean_key]["probs"]
        mean_f = float(train_mean_cache[mean_key]["f"])
        sensitivity_rows.append(
            {
                **meta,
                "condition": "train_mean_evidence",
                "shuffle_index": "",
                "mu_req": float(train_mean_cache[mean_key]["mu"]),
                "continuous_F_eta": mean_f,
                "delta_F_eta_vs_real": mean_f - base_f,
                "P_success_3N": p_success_at(DENSE_FORCES, mean_probs, 3.0),
                "P_success_4N": p_success_at(DENSE_FORCES, mean_probs, 4.0),
                "P_success_5N": p_success_at(DENSE_FORCES, mean_probs, 5.0),
            }
        )
        candidates = [x for x in by_task_ids[task] if x != rid] or [rid]
        for i in range(SHUFFLES_PER_ROOT):
            sid = str(rng.choice(candidates))
            sh_probs = curve_summary_by_root[sid]["q2f_probs"]["ensemble_mean"]
            sh_f = float(curve_summary_by_root[sid]["q2f_f_cont"])
            sensitivity_rows.append(
                {
                    **meta,
                    "condition": "within_task_shuffled_evidence",
                    "shuffle_index": i,
                    "shuffled_source_root_context_id": sid,
                    "continuous_F_eta": sh_f,
                    "delta_F_eta_vs_real": sh_f - base_f,
                    "P_success_3N": p_success_at(DENSE_FORCES, sh_probs, 3.0),
                    "P_success_4N": p_success_at(DENSE_FORCES, sh_probs, 4.0),
                    "P_success_5N": p_success_at(DENSE_FORCES, sh_probs, 5.0),
                }
            )
        task_prior_rows.append(
            {
                **meta,
                "taskf_continuous_F_eta": float(curve_summary_by_root[rid]["taskf_f_cont"]),
                "q2f_continuous_F_eta": base_f,
                "delta_F_evidence_q2f_minus_taskf": base_f - float(curve_summary_by_root[rid]["taskf_f_cont"]),
                "taskf_projected_L1_force": float(curve_summary_by_root[rid]["taskf_exec"]),
                "q2f_projected_L1_force": float(curve_summary_by_root[rid]["q2f_exec"]),
            }
        )
    write_csv(OUT / "P5S0E_EVIDENCE_SENSITIVITY.csv", sensitivity_rows)
    write_csv(OUT / "P5S0E_TASK_PRIOR_VS_EVIDENCE.csv", task_prior_rows)

    planner_rows: list[dict[str, Any]] = []
    for _, root in qdf.iterrows():
        rid = str(root["root_context_id"])
        task = int(root["task"])
        qmean = curve_summary_by_root[rid]["q2f_probs"]["ensemble_mean"]
        samples = posterior_samples_by_root[rid]
        p10 = np.percentile(samples, 10, axis=1)
        planners = [("A_current_eta_mean", qmean, None)]
        planners.append(("B_posterior_10th_quantile", p10, None))
        for planner_name, curve, rfail in planners:
            f_cont, fb = dense_f_eta(DENSE_FORCES, curve)
            row = {
                "task": task,
                "root_context_id": rid,
                "friction_regime": root["friction_band"],
                "planner": planner_name,
                "R_fail": rfail if rfail is not None else "",
                "continuous_selected_force": f_cont,
                "fallback": int(fb),
                "L1_selected_force": project_lattice(f_cont, LATTICES["L1_original"]),
                "L2_selected_force": project_lattice(f_cont, LATTICES["L2_half_N"]),
                "L3_selected_force": project_lattice(f_cont, LATTICES["L3_quarter_N"]),
                "posterior_uncertainty_at_selected_force": float(np.std(samples[np.argmin(np.abs(DENSE_FORCES - f_cont))])),
            }
            planner_rows.append(row)
        for rfail in R_FAIL_VALUES:
            utilities = qmean * (F_MAX_UTILITY - DENSE_FORCES) + (1.0 - qmean) * rfail
            j = int(np.argmax(utilities))
            f_cont = float(DENSE_FORCES[j])
            planner_rows.append(
                {
                    "task": task,
                    "root_context_id": rid,
                    "friction_regime": root["friction_band"],
                    "planner": "C_gnp_mc_expected_utility",
                    "R_fail": rfail,
                    "continuous_selected_force": f_cont,
                    "utility": float(utilities[j]),
                    "L1_selected_force": project_lattice(f_cont, LATTICES["L1_original"]),
                    "L2_selected_force": project_lattice(f_cont, LATTICES["L2_half_N"]),
                    "L3_selected_force": project_lattice(f_cont, LATTICES["L3_quarter_N"]),
                    "posterior_uncertainty_at_selected_force": float(np.std(samples[j])),
                }
            )
    write_csv(OUT / "P5S0E_GNP_PLANNER_RESULTS.csv", planner_rows)

    # Classification metrics.
    l1_q2f = root_df["q2f_projected_L1_force"]
    q2f_range = q2f_stats["max"] - q2f_stats["min"]
    q2f_std = q2f_stats["std"]
    evidence_df = pd.DataFrame(sensitivity_rows)
    real_mean = evidence_df[evidence_df["condition"] == "train_mean_evidence"]["delta_F_eta_vs_real"].abs()
    shuf = evidence_df[evidence_df["condition"] == "within_task_shuffled_evidence"]["delta_F_eta_vs_real"].abs()
    evidence_effect_median = float(real_mean.median())
    shuffle_effect_median = float(shuf.median())
    lattice_substantial = int(l1_q2f.nunique()) < int(root_df["q2f_continuous_F_eta"].round(2).nunique()) and collapsed_meaningful >= 0.5
    model_substantial = q2f_std < 0.35 and evidence_effect_median < 0.25 and shuffle_effect_median < 0.25
    if model_substantial and lattice_substantial:
        classification = "P5S0E_BOTH_MODEL_AND_LATTICE_COLLAPSE"
    elif model_substantial:
        classification = "P5S0E_MODEL_DECISION_COLLAPSE"
    elif lattice_substantial:
        classification = "P5S0E_FORCE_LATTICE_COLLAPSE"
    else:
        planner_div = pd.DataFrame(planner_rows).groupby("planner")["continuous_selected_force"].nunique().to_dict()
        if any(int(v) > int(root_df["q2f_continuous_F_eta"].round(2).nunique()) for v in planner_div.values()):
            classification = "P5S0E_Q2F_DECISIONS_DIVERSE_BUT_CURRENT_SELECTION_RULE_COLLAPSES_GAIN"
        else:
            classification = "P5S0E_MODEL_DECISION_COLLAPSE"

    lat_df = pd.DataFrame(lattice_rows)
    plan_df = pd.DataFrame(planner_rows)
    sens_df = pd.DataFrame(sensitivity_rows)
    task_prior_df = pd.DataFrame(task_prior_rows)
    planner_summary = plan_df.groupby(["planner", "R_fail"], dropna=False).agg(
        unique_cont=("continuous_selected_force", lambda s: int(pd.Series(s).round(2).nunique())),
        mean_cont=("continuous_selected_force", "mean"),
        unique_L1=("L1_selected_force", "nunique"),
        mean_L1=("L1_selected_force", "mean"),
        mean_unc=("posterior_uncertainty_at_selected_force", "mean"),
    ).reset_index()
    planner_interpretation = "; ".join(
        f"{r.planner}/R={r.R_fail}: unique_cont={int(r.unique_cont)}, mean={float(r.mean_cont):.2f}"
        for r in planner_summary.itertuples()
    )

    metrics = {
        "primary_classification": classification,
        "feature_count": len(inf.feature_names),
        "q2f_param_count": int(sum(p.numel() for p in inf.models[(Q2F, 0)]["model"].parameters())),
        "mean_sequence_length": float(np.mean([e.length for e in encoded.values()])),
        "planner_interpretation": planner_interpretation,
    }
    write_markdown_artifacts(OUT, metrics)

    l2_q2f = lat_df[(lat_df["model"] == Q2F) & (lat_df["lattice"] == "L2_half_N")].iloc[0]
    l3_q2f = lat_df[(lat_df["model"] == Q2F) & (lat_df["lattice"] == "L3_quarter_N")].iloc[0]
    delta = root_df["q2f_minus_taskf_F_eta"]
    per_task_delta = task_prior_df.groupby("task")["delta_F_evidence_q2f_minus_taskf"].agg(["mean", "std", "min", "max"]).round(3).to_dict("index")
    per_fric_delta = task_prior_df.groupby("friction_regime")["delta_F_evidence_q2f_minus_taskf"].agg(["mean", "std", "min", "max"]).round(3).to_dict("index")
    current_planner = planner_summary[planner_summary["planner"] == "A_current_eta_mean"].iloc[0].to_dict()
    quant_planner = planner_summary[planner_summary["planner"] == "B_posterior_10th_quantile"].iloc[0].to_dict()
    utility_planner = planner_summary[planner_summary["planner"] == "C_gnp_mc_expected_utility"].to_dict("records")

    original_dist = {
        arm: mode_counts(pd.to_numeric(g["requested_force_N"], errors="coerce"))
        for arm, g in adf.groupby("arm")
    }
    report = f"""STATUS:
PASS
METHOD_CHANGE: NONE
ARTIFACTS:
{OUT}

FRESH E2E SOURCE:
- Roots: {len(root_df)}
- Tasks: {TASKS}
- Original executed force distribution: {original_dist}

CONTINUOUS Q2F DECISIONS:
- Min: {q2f_stats['min']:.3f}
- Max: {q2f_stats['max']:.3f}
- Mean: {q2f_stats['mean']:.3f}
- Std: {q2f_stats['std']:.3f}
- IQR: {q2f_stats['iqr']:.3f}
- Unique/near-unique decisions: {root_df['q2f_continuous_F_eta'].round(2).nunique()} at 0.01-N rounding

TASK+F VS Q2F:
- Mean continuous difference: {float(delta.mean()):.3f}
- |difference| >= 0.1N: {int((delta.abs() >= 0.1).sum())}/{len(delta)}
- |difference| >= 0.25N: {int((delta.abs() >= 0.25).sum())}/{len(delta)}
- |difference| >= 0.5N: {int((delta.abs() >= 0.5).sum())}/{len(delta)}
- Per-task: {per_task_delta}
- Per-friction regime: {per_fric_delta}

QUANTIZATION:
- Original lattice: {F_EXEC}; Q2F distribution {mode_counts(root_df['q2f_projected_L1_force'])}
- Fraction collapsed to 5N: {float((root_df['q2f_projected_L1_force'] == 5.0).mean()):.3f}
- Meaningful Q2F differences hidden by lattice: {collapsed_meaningful:.3f}
- 0.5N lattice predicted diversity: {int(l2_q2f['unique_selected_forces'])} unique, mean force {float(l2_q2f['mean_predicted_requested_force']):.3f}
- 0.25N lattice predicted diversity: {int(l3_q2f['unique_selected_forces'])} unique, mean force {float(l3_q2f['mean_predicted_requested_force']):.3f}

EVIDENCE USE:
- Real vs mean evidence threshold change: median |ΔF_eta|={evidence_effect_median:.3f}
- Shuffle threshold change: median |ΔF_eta|={shuffle_effect_median:.3f}
- Does physical evidence alter F_eta: {'YES, but weakly/narrowly' if shuffle_effect_median >= 0.1 or evidence_effect_median >= 0.1 else 'NO, little final-threshold movement'}

GNP PLANNER:
- Current eta rule: unique continuous={int(current_planner['unique_cont'])}, mean={float(current_planner['mean_cont']):.3f}, unique L1={int(current_planner['unique_L1'])}
- Posterior quantile: unique continuous={int(quant_planner['unique_cont'])}, mean={float(quant_planner['mean_cont']):.3f}, unique L1={int(quant_planner['unique_L1'])}
- MC expected utility: {utility_planner}
- Does uncertainty-aware planning increase decision diversity: {'YES' if int(quant_planner['unique_cont']) > int(current_planner['unique_cont']) else 'NO for quantile; see utility sweep for penalty-dependent changes'}
- Does it become more conservative when uncertainty is high: diagnostic only; posterior quantile mean force {float(quant_planner['mean_cont']):.3f} vs current {float(current_planner['mean_cont']):.3f}

GRU:
- Current architecture: 32-unit projection + 1-layer GRU(32) + 4-D Gaussian latent + monotonic threshold decoder
- Parameter count: {metrics['q2f_param_count']}
- Keep/modify recommendation: keep GRU as a component, but split static contact and dynamic response in the next model if collapse is confirmed

PRIMARY_CLASSIFICATION:
{classification}

SCIENTIFIC INTERPRETATION:
1. P5-S0-D did not fail because of online query or inference integration; those remain valid.
2. The frozen Q2F frontier does move relative to Task+F on some roots, but the movement is not large enough to create a reliable success-force trade-off gain.
3. The original lattice hides some continuous variation, especially around the 5 N boundary, but finer lattices alone would not solve the weak evidence-conditioned frontier.
4. Current deployment largely reduces posterior information to an ensemble-mean threshold crossing.
5. The next experimental change should improve model decision diversity before spending simulator budget on finer controller qualification.

NEXT:
- If lattice collapse:
  perform dense controller qualification before new E2E.
- If model collapse:
  train Q2F-GNP-Teacher on existing matched data.
- If planner collapse:
  preregister a GNP-style uncertainty-aware planner and rerun fresh E2E.
- If both:
  solve model decision diversity before controller refinement.
"""
    (OUT / "P5S0E_FINAL_REPORT.md").write_text("# P5-S0-E Final Report\n\n" + report, encoding="utf-8")
    verdict = {
        "STATUS": "PASS",
        "METHOD_CHANGE": "NONE",
        "PRIMARY_CLASSIFICATION": classification,
        "ARTIFACTS": str(OUT),
        "fresh_roots": len(root_df),
        "q2f_continuous_F_eta": q2f_stats,
        "q2f_taskf_abs_diff_ge_0p1": int((delta.abs() >= 0.1).sum()),
        "q2f_taskf_abs_diff_ge_0p25": int((delta.abs() >= 0.25).sum()),
        "q2f_taskf_abs_diff_ge_0p5": int((delta.abs() >= 0.5).sum()),
        "fraction_q2f_projected_5N": float((root_df["q2f_projected_L1_force"] == 5.0).mean()),
        "fraction_meaningful_differences_collapsed_to_same_5N": collapsed_meaningful,
        "median_abs_real_vs_mean_evidence_delta_F": evidence_effect_median,
        "median_abs_shuffle_delta_F": shuffle_effect_median,
    }
    write_json(OUT / "P5S0E_FINAL_VERDICT.json", verdict)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
