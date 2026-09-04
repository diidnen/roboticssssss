#!/usr/bin/env python3
"""Minimal active-friction successor analysis.

This module deliberately separates the new explicit friction estimator from
the frozen P5-S0-C Query2Force models.  It reuses the P5-S0-C feature adapter
and the matched probe/branch manifests, trains a tiny GRU on root-disjoint
friction labels, and emits an auditable artifact namespace.  Isaac execution
is intentionally a separate worker entry point so that data/model audit does
not silently mutate the old experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


REPO = Path("/home/exouser/Tabero")
RESULTS_ROOT = REPO / "analysis/results"
C_ARTIFACT = RESULTS_ROOT / "p5s0c_paired_boundary_probe_value_20260824_000542"
MODEL_SOURCE = REPO / "analysis/p5s0c_model_adjudication.py"
OUT = Path()
SEED = 20260828
TASKS = [0, 1, 5, 6]
FRICTION_BANDS = {"LOW": (0.20, 0.30), "MID": (0.45, 0.60), "HIGH": (0.90, 1.00)}
FORCES = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P5C_MODEL = import_module(MODEL_SOURCE, "active_friction_p5s0c_model")


@dataclass
class Example:
    context_id: str
    root_id: str
    task: int
    split: str
    band: str
    friction: float
    sequence: np.ndarray


class FrictionGRU(nn.Module):
    """Small explicit p(mu | probe) estimator: projection -> GRU -> heads."""

    def __init__(self, input_dim: int, projection_dim: int = 16, hidden_dim: int = 16):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, projection_dim), nn.ReLU())
        self.gru = nn.GRU(projection_dim, hidden_dim, batch_first=True)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.projection(x)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = h[-1]
        mu = self.mu_head(h).squeeze(1)
        log_sigma = self.log_sigma_head(h).squeeze(1).clamp(-5.0, 1.5)
        return mu, log_sigma


def load_examples() -> tuple[list[Example], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    contexts = pd.read_csv(C_ARTIFACT / "P5S0C_CONTEXT_MANIFEST.csv")
    branches = pd.read_csv(C_ARTIFACT / "P5S0C_BRANCH_MANIFEST.csv")
    seqs, _, lengths, meta = P5C_MODEL.load_features(contexts)
    examples: list[Example] = []
    for row in contexts.itertuples(index=False):
        examples.append(
            Example(
                context_id=str(row.context_id),
                root_id=str(row.root_id),
                task=int(row.task),
                split=str(row.split),
                band=str(row.friction_band),
                friction=float(row.hidden_friction_analysis_only),
                sequence=np.asarray(seqs[str(row.context_id)], dtype=np.float32),
            )
        )
    if len(examples) != 144 or len({e.root_id for e in examples}) != 48:
        raise RuntimeError("unexpected P5-S0-C probe dataset size")
    if any(e.sequence.shape[0] != 215 for e in examples):
        raise RuntimeError("unexpected probe sequence length")
    return examples, meta, contexts, branches


def split_examples(examples: list[Example], split: str) -> list[Example]:
    return [e for e in examples if e.split == split]


def batch(examples: list[Example], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(e.sequence.shape[0] for e in examples)
    x = np.zeros((len(examples), max_len, examples[0].sequence.shape[1]), dtype=np.float32)
    lengths = []
    y = []
    for i, e in enumerate(examples):
        x[i, : e.sequence.shape[0]] = e.sequence
        lengths.append(e.sequence.shape[0])
        y.append(e.friction)
    return (
        torch.tensor(x, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
    )


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigma = torch.exp(log_sigma).clamp_min(1e-3)
    return 0.5 * (((y - mu) / sigma) ** 2 + 2.0 * log_sigma)


def predict(model: FrictionGRU, examples: list[Example], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x, lengths, _ = batch(examples, device)
        mu, log_sigma = model(x, lengths)
    return mu.cpu().numpy(), np.exp(log_sigma.cpu().numpy())


def fit_model(examples: list[Example], device: torch.device) -> tuple[FrictionGRU, dict[str, Any]]:
    train = split_examples(examples, "TRAIN")
    dev = split_examples(examples, "DEV")
    model = FrictionGRU(train[0].sequence.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_dev = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, 401):
        model.train()
        x, lengths, y = batch(train, device)
        opt.zero_grad(set_to_none=True)
        mu, log_sigma = model(x, lengths)
        # A small Gaussian NLL plus a mild mean-error term prevents inflated
        # uncertainty from hiding a biased estimate.
        loss = gaussian_nll(mu, log_sigma, y).mean() + 0.05 * torch.abs(mu - y).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            dx, dl, dy = batch(dev, device)
            dmu, dlog = model(dx, dl)
            dev_loss = float((gaussian_nll(dmu, dlog, dy) + 0.05 * torch.abs(dmu - dy)).mean().item())
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "dev_loss": dev_loss})
        if dev_loss < best_dev - 1e-7:
            best_dev = dev_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 50:
                break
    if best_state is None:
        raise RuntimeError("no estimator checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_dev_loss": best_dev, "history": history}


def calibration_rows(mu: np.ndarray, sigma: np.ndarray, examples: list[Example]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for pred, sd, e in zip(mu, sigma, examples):
        rows.append({
            "context_id": e.context_id, "root_id": e.root_id, "task": e.task,
            "split": e.split, "friction_band": e.band, "friction_gt": e.friction,
            "mu_hat": float(pred), "sigma_mu": float(sd),
            "abs_error": float(abs(pred - e.friction)),
            "interval_90_lo": float(pred - 1.645 * sd),
            "interval_90_hi": float(pred + 1.645 * sd),
            "covered_90": int(pred - 1.645 * sd <= e.friction <= pred + 1.645 * sd),
        })
    return rows, {}


def summarize_predictions(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    g = [r for r in rows if r["split"] == split]
    if not g:
        return {"n": 0}
    err = np.asarray([r["abs_error"] for r in g], dtype=float)
    gt = np.asarray([r["friction_gt"] for r in g], dtype=float)
    pred = np.asarray([r["mu_hat"] for r in g], dtype=float)
    sd = np.asarray([r["sigma_mu"] for r in g], dtype=float)
    return {
        "n": len(g), "roots": len({r["root_id"] for r in g}),
        "mae": float(err.mean()), "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "relative_mae": float(np.mean(err / np.maximum(np.abs(gt), 1e-6))),
        "mean_sigma": float(sd.mean()), "coverage_90": float(np.mean([r["covered_90"] for r in g])),
        "mean_mu_hat": float(pred.mean()), "mean_mu_gt": float(gt.mean()),
    }


def frontier(branches: pd.DataFrame, context_id: str) -> dict[float, int]:
    g = branches[branches.context_id == context_id]
    return {float(r.requested_force_N): int(r.full_task_success_y) for r in g.itertuples()}


def minimum_success_force(outcomes: dict[float, int]) -> float | None:
    ok = [f for f, y in outcomes.items() if y == 1]
    return min(ok) if ok else None


def empirical_imagination_audit(pred_rows: list[dict[str, Any]], branches: pd.DataFrame) -> list[dict[str, Any]]:
    """Audit the decision bridge against frozen deterministic branch frontiers.

    This is explicitly an offline action audit, not a substitute for the new
    Isaac snapshot-branch imagination pilot.  It uses only TRAIN contexts to
    construct a conservative nearest-friction frontier prior, then evaluates
    held-out contexts against their stored real branch frontier.
    """
    train_rows = [r for r in pred_rows if r["split"] == "TRAIN"]
    result = []
    for r in pred_rows:
        if r["split"] not in {"DEV", "TEST"}:
            continue
        same_task = [x for x in train_rows if int(x["task"]) == int(r["task"])]
        if not same_task:
            continue
        # Map each force to the empirical TRAIN success probability at the
        # closest friction. This is a decision audit only; no GT is input.
        imagined = {}
        for f in FORCES:
            vals = []
            for x in same_task:
                bg = frontier(branches, x["context_id"])
                if f in bg:
                    vals.append((abs(float(x["friction_gt"]) - float(r["mu_hat"])), bg[f]))
            vals.sort(key=lambda z: z[0])
            imagined[f] = float(np.mean([v for _, v in vals[: min(12, len(vals))]])) if vals else 0.0
        chosen = next((f for f in FORCES if imagined[f] >= 0.9), max(FORCES))
        real = frontier(branches, r["context_id"])
        oracle = minimum_success_force(real)
        result.append({
            "context_id": r["context_id"], "root_id": r["root_id"], "task": r["task"],
            "split": r["split"], "friction_gt": r["friction_gt"], "mu_hat": r["mu_hat"],
            "sigma_mu": r["sigma_mu"], "imagined_force_N": chosen,
            "oracle_min_force_N": oracle if oracle is not None else "",
            "real_success_at_imagined": real.get(chosen, 0),
            "absolute_force_gap": abs(chosen - oracle) if oracle is not None else "",
            "imagined_curve_json": json.dumps(imagined, sort_keys=True),
        })
    return result


def write_archaeology(examples: list[Example], meta: dict[str, Any], branches: pd.DataFrame) -> None:
    reuse = [
        {"component": "fixed active probe", "implementation": "P4-B common contact-frame shear", "file": str(C_ARTIFACT / "P5S0C_FROZEN_PROBE.json"), "status": "qualified/reused", "action": "reuse"},
        {"component": "probe feature adapter", "implementation": "sequence_dataframe/load_features", "file": str(MODEL_SOURCE), "status": "exact frozen feature order", "action": "reuse"},
        {"component": "probe normalization", "implementation": "P5S0C TRAIN-only means/stds", "file": str(C_ARTIFACT / "P5S0C_NORMALIZATION.json"), "status": "verified", "action": "reuse"},
        {"component": "matched force branches", "implementation": "deterministic downstream_branch", "file": str(REPO / "analysis/p5s0c_paired_boundary_probe_value.py"), "status": "576 real branch outcomes", "action": "reuse as validation/reference"},
        {"component": "semantic policy", "implementation": "Pi0Config frozen VLA", "file": str(REPO / "analysis/p6g1r1_controller_grasp_vla_handoff.py"), "status": "Pi0, not Pi0.5; semantic context only", "action": "reuse contract"},
        {"component": "explicit friction estimator", "implementation": "projection(16) -> GRU(16) -> mu/sigma heads", "file": str(Path(__file__)), "status": "new", "action": "new method"},
        {"component": "runtime imagination", "implementation": "snapshot branch around deterministic downstream controller", "file": str(REPO / "analysis/p5s0c_paired_boundary_probe_value.py"), "status": "new pilot still required", "action": "minimally adapt"},
    ]
    write_json(OUT / "REUSE_MAP.json", reuse)
    write_json(OUT / "DATA_AUDIT.json", {
        "contexts": len(examples), "roots": len({e.root_id for e in examples}),
        "branches": len(branches), "sequence_lengths": sorted({int(e.sequence.shape[0]) for e in examples}),
        "feature_count": int(examples[0].sequence.shape[1]), "feature_names": meta["dynamic_feature_names"],
        "excluded_fields": meta["excluded_fields"], "root_split": {s: len({e.root_id for e in examples if e.split == s}) for s in ["TRAIN", "DEV", "TEST"]},
        "friction_only_randomized_intent": True, "gt_friction_model_input": False,
    })


def run() -> int:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()
    OUT = args.out or RESULTS_ROOT / f"active_friction_imagination_{now_tag()}"
    OUT.mkdir(parents=True, exist_ok=False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    examples, meta, contexts, branches = load_examples()
    write_archaeology(examples, meta, branches)
    protocol = {
        "name": "Active friction imagination successor v1",
        "method_change": "explicit friction estimate plus deterministic runtime imagination",
        "hidden_randomized_property": "friction only",
        "friction_bands": FRICTION_BANDS,
        "probe": "reused P4-B common contact-frame shear",
        "candidate_forces_N": FORCES,
        "root_split": "TRAIN/DEV/TEST by root_id, never random frames",
        "estimator": {"projection_dim": 16, "gru_hidden_dim": 16, "outputs": ["mu_hat", "sigma_mu"]},
        "runtime_imagination": "new snapshot-branch pilot; existing deterministic downstream_branch reused",
        "pi0": "Pi0Config (pi0, not pi0.5), semantic task context only",
        "source_hashes": {str(p): sha256_file(p) for p in [MODEL_SOURCE, C_ARTIFACT / "P5S0C_NORMALIZATION.json", C_ARTIFACT / "P5S0C_FEATURE_MANIFEST.json"]},
    }
    write_json(OUT / "PROTOCOL.json", protocol)
    (OUT / "PROTOCOL_SHA256.txt").write_text(sha256_file(OUT / "PROTOCOL.json") + "\n", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_info = fit_model(examples, device)
    ckpt = OUT / "FRICTION_GRU.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": int(examples[0].sequence.shape[1]), "projection_dim": 16, "hidden_dim": 16, "seed": SEED}, ckpt)
    rows: list[dict[str, Any]] = []
    for split in ["TRAIN", "DEV", "TEST"]:
        mu, sigma = predict(model, split_examples(examples, split), device)
        split_rows, _ = calibration_rows(mu, sigma, split_examples(examples, split))
        rows.extend(split_rows)
    write_csv(OUT / "FRICTION_PREDICTIONS.csv", rows)
    metrics = {s.lower(): summarize_predictions(rows, s) for s in ["TRAIN", "DEV", "TEST"]}
    write_json(OUT / "FRICTION_ESTIMATION_METRICS.json", {"device": str(device), "train": {k: v for k, v in train_info.items() if k != "history"}, "splits": metrics})
    write_csv(OUT / "TRAINING_LOG.csv", train_info["history"])
    audit_rows = empirical_imagination_audit(rows, branches)
    write_csv(OUT / "OFFLINE_IMAGINATION_DECISION_AUDIT.csv", audit_rows)
    if audit_rows:
        d = pd.DataFrame(audit_rows)
        summary = {
            "n": len(d), "split_counts": d.groupby("split").size().to_dict(),
            "selection_accuracy": float((pd.to_numeric(d["real_success_at_imagined"]) == 1).mean()),
            "mean_force_gap": float(pd.to_numeric(d["absolute_force_gap"], errors="coerce").mean()),
            "under_force_rate": float((pd.to_numeric(d["real_success_at_imagined"]) == 0).mean()),
        }
    else:
        summary = {"n": 0}
    write_json(OUT / "OFFLINE_IMAGINATION_DECISION_AUDIT_SUMMARY.json", summary)
    report = f"""# Active friction imagination successor v1\n\nSTATUS: ESTIMATOR_TRAINED_RUNTIME_PILOT_PENDING\n\nThis namespace is successor-method development. P5-S0-C/P5-S0-D artifacts are preserved.\n\n- contexts: {len(examples)}; roots: {len({e.root_id for e in examples})}; branches: {len(branches)}\n- train/dev/test roots: {metrics['train']['roots']}/{metrics['dev']['roots']}/{metrics['test']['roots']}\n- estimator: projection 16 -> GRU 16 -> mu_hat, sigma_mu\n- estimator checkpoint: {ckpt}\n- offline decision audit: {summary}\n- runtime Isaac snapshot-branch imagination and real full-task pilot: NOT yet executed by this module\n\nThe offline decision audit is not presented as runtime-imagination evidence; it is only a gate diagnostic against the frozen deterministic branch frontiers.\n"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"out": str(OUT), "metrics": metrics, "decision_audit": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
