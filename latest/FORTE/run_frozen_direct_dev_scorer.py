#!/usr/bin/env python3
"""Score the frozen ActiveForcing-Direct backend on one DEV context.

This is an inference-only adapter around the frozen implementation in
``gnp_style_continuous.py``.  It deliberately loads only DEV contexts and
the three frozen FEASIBILITY_ONLY checkpoints; it never reads TEST rows.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


FORTE = Path("/home/exouser/FORTE")
GNP = FORTE / "gnp_style_continuous.py"
GNP_OUT = FORTE / "gnp_style_continuous_20260830_125107"
TASK = 0
DEV_ROOT_TOKEN = "root06_s5106"
THRESHOLD = 0.5
CANDIDATES = [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def load_cpu_stack(gnp, tpi, full, pre, active):
    norm_q = json.loads((GNP_OUT / "GNP_STYLE_TRAIN_NORMALIZATION.json").read_text())
    norm = tuple(np.asarray(norm_q[k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
    models = []
    cals = []
    for seed in gnp.SEEDS:
        ck = torch.load(GNP_OUT / f"GNP_STYLE_CONTINUOUS_FEAS_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = full.FeasibilityOnly().to(torch.device("cpu"))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        models.append(model)
        cals.append(gnp.load_calibration(GNP_OUT / f"GNP_STYLE_FEAS_TRAIN_CALIBRATION_seed{seed}.json"))
    contexts = pre.all_contexts_for_split("DEV", active)
    values = list(contexts.values()) if isinstance(contexts, dict) else list(contexts)
    matching = [c for c in values if int(c.task) == TASK and DEV_ROOT_TOKEN in str(c.root_id)]
    if not matching:
        raise RuntimeError("no frozen DEV task0/root06 context")
    ctx = sorted(matching, key=lambda c: abs(float(c.mu_gt) - 0.2))[0]
    return norm, (models, cals), ctx


def main() -> int:
    gnp = load(GNP, "frozen_direct_scorer_gnp")
    tpi, cf, full, pre, active = gnp.modules()
    norm, stack, ctx = load_cpu_stack(gnp, tpi, full, pre, active)
    rows = []
    for force in CANDIDATES:
        pred = gnp.predict_probability(
            tpi, cf, ctx, force, [float(ctx.mu_gt)], "FEASIBILITY_ONLY", stack, norm,
            torch.device("cpu"),
        )
        rows.append({
            "candidate_force_N": force,
            **pred,
            "threshold": THRESHOLD,
            "passes_threshold": int(pred["raw_probability"] >= THRESHOLD),
        })
    passing = [r for r in rows if r["passes_threshold"]]
    selected = float(min(passing, key=lambda r: r["candidate_force_N"])["candidate_force_N"]) if passing else 5.0
    payload = {
        "method": "ActiveForcing-Direct",
        "backend": "gnp_style_continuous.py:FEASIBILITY_ONLY",
        "context": {
            "context_id": str(ctx.context_id),
            "root_id": str(ctx.root_id),
            "task": int(ctx.task),
            "friction_band": str(ctx.friction_band),
            "backend_context_friction": float(ctx.mu_gt),
            "source": "frozen DEV context manifest; no new context representation",
        },
        "direct_context": {
            "context_id": str(ctx.context_id),
            "root_id": str(ctx.root_id),
            "task": int(ctx.task),
            "friction_band": str(ctx.friction_band),
            "backend_context_friction": float(ctx.mu_gt),
            "backend_context_source": "frozen DEV context manifest; no new context representation",
        },
        "candidate_scores": rows,
        "threshold": THRESHOLD,
        "selected_force_N": selected,
        "fallback_used": not bool(passing),
        "fallback": "max candidate 5.00 N when no candidate passes",
        "device": "cpu inference adapter; checkpoint and feature code unchanged",
        "test_roots_accessed": [],
    }
    print(json.dumps(jsonable(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
