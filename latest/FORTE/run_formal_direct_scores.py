#!/usr/bin/env python3
"""Freeze TEST Direct candidate scores from the existing backend, inference-only."""
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
THRESHOLD = 0.5
CANDIDATES = [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
ROOTS = [5174, 5175, 5176, 5177, 5178, 5179]
MUS = [0.2, 0.5, 1.0]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)): return value.item()
    return value


def main() -> int:
    gnp = load(GNP, "formal_direct_gnp")
    tpi, cf, full, pre, active = gnp.modules()
    norm_q = json.loads((GNP_OUT / "GNP_STYLE_TRAIN_NORMALIZATION.json").read_text())
    norm = tuple(np.asarray(norm_q[k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
    models, cals = [], []
    for seed in gnp.SEEDS:
        ck = torch.load(GNP_OUT / f"GNP_STYLE_CONTINUOUS_FEAS_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = full.FeasibilityOnly().to(torch.device("cpu")); model.load_state_dict(ck["state_dict"]); model.eval()
        models.append(model); cals.append(gnp.load_calibration(GNP_OUT / f"GNP_STYLE_FEAS_TRAIN_CALIBRATION_seed{seed}.json"))
    contexts = pre.all_contexts_for_split("TEST", active)
    values = list(contexts.values()) if isinstance(contexts, dict) else list(contexts)
    rows, decisions = [], []
    for root in ROOTS:
        token = f"root{root - 5100:02d}_s{root}" if root < 5200 else f"s{root}"
        matching = [c for c in values if int(c.task) == 0 and token in str(c.root_id)]
        if not matching:
            # Fall back to the explicit seed token used by some manifests.
            matching = [c for c in values if int(c.task) == 0 and f"s{root}" in str(c.root_id)]
        if not matching:
            raise RuntimeError(f"no frozen TEST task0 context for root {root}")
        for mu in MUS:
            ctx = sorted(matching, key=lambda c: abs(float(c.mu_gt) - mu))[0]
            scores = []
            for force in CANDIDATES:
                pred = gnp.predict_probability(tpi, cf, ctx, force, [float(ctx.mu_gt)], "FEASIBILITY_ONLY", (models, cals), norm, torch.device("cpu"))
                row = {"root_seed": root, "mu": mu, "backend_context_id": str(ctx.context_id), "backend_context_mu_gt": float(ctx.mu_gt), "candidate_force_N": force, **pred, "threshold": THRESHOLD, "passes_threshold": int(pred["raw_probability"] >= THRESHOLD)}
                rows.append(row); scores.append(row)
            passing = [r for r in scores if r["passes_threshold"]]
            selected = float(min(passing, key=lambda r: r["candidate_force_N"])["candidate_force_N"]) if passing else 5.0
            decisions.append({"root_seed": root, "mu": mu, "context_id": str(ctx.context_id), "context_root_id": str(ctx.root_id), "context_friction_band": str(ctx.friction_band), "context_mu_gt": float(ctx.mu_gt), "threshold": THRESHOLD, "selected_force_N": selected, "fallback_used": not bool(passing), "candidate_scores": scores})
    payload = {"method": "ActiveForcing-Direct", "backend": "gnp_style_continuous.py:FEASIBILITY_ONLY", "threshold": THRESHOLD, "candidate_grid_N": CANDIDATES, "decisions": decisions, "candidate_scores": rows, "test_roots": ROOTS, "test_mus": MUS, "test_roots_accessed": ROOTS}
    print(json.dumps(jsonable(payload), sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
