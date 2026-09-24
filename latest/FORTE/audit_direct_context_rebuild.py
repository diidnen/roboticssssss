#!/usr/bin/env python3
"""CPU-only audit of the frozen Direct context reconstruction.

The historical DEV context is used only as an audit reference.  No TEST row
or outcome is loaded.  The audit reconstructs strict pre-probe state from raw
P4-B telemetry, rebuilds the first-H branch_hold query skeleton from the raw
probe EEF endpoint, and compares the resulting frozen model scores with the
published DEV reference scores.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

F = Path("/home/exouser/FORTE")
T = Path("/home/exouser/Tabero")
OUT = F / "hidden_friction_baseline_20260831"
HIST = T / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542"
GRID = [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
KNOWN = [3.4606562321775644e-05, 0.00013632514920711673, 0.00156700786367138,
         0.019662895889169984, 0.18000311576165032, 0.6612490167118354,
         0.8789746234280668, 0.9639242557870639, 0.9884905571711095]


def load(path: Path, name: str):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def main() -> int:
    g = load(F / "gnp_style_continuous.py", "context_audit_gnp")
    tpi, cf, full, pre, active = g.modules()
    contexts = pre.all_contexts_for_split("DEV", active)
    ctx = [c for c in contexts.values() if c.task == 0 and "root06_s5106" in c.root_id and c.friction_band == "LOW"][0]
    probe = pd.read_csv(ctx.probe_path)
    hold = probe[probe.probe_phase.astype(str) == "hold"]
    r = hold.iloc[-1]
    state = np.zeros(13, np.float32); mask = np.zeros(13, np.float32)
    opening = float(r.gripper_opening); state[11:13] = [opening, -opening]; mask[:6] = 1.; mask[11:13] = 1.
    rebuilt = SimpleNamespace(**ctx.__dict__)
    rebuilt.preprobe_state = state; rebuilt.preprobe_mask = mask; rebuilt.preprobe_step = int(r.step)
    norm_q = json.loads((F / "gnp_style_continuous_20260830_125107/GNP_STYLE_TRAIN_NORMALIZATION.json").read_text())
    norm = tuple(np.asarray(norm_q[k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
    models = []; cals = []
    for seed in [0, 1, 2]:
        ck = torch.load(F / f"gnp_style_continuous_20260830_125107/GNP_STYLE_CONTINUOUS_FEAS_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = full.FeasibilityOnly(); model.load_state_dict(ck["state_dict"]); model.eval(); models.append(model)
        cals.append(g.load_calibration(F / f"gnp_style_continuous_20260830_125107/GNP_STYLE_FEAS_TRAIN_CALIBRATION_seed{seed}.json"))
    stack = (models, cals)
    reference = []; rebuilt_scores = []
    for f in GRID:
        reference.append(float(g.predict_probability(tpi, cf, ctx, f, [float(ctx.mu_gt)], "FEASIBILITY_ONLY", stack, norm, torch.device("cpu"))["raw_probability"]))
        rebuilt_scores.append(float(g.predict_probability(tpi, cf, rebuilt, f, [float(ctx.mu_gt)], "FEASIBILITY_ONLY", stack, norm, torch.device("cpu"))["raw_probability"]))
    # Rebuild the branch_hold command origin from raw probe endpoint, keeping
    # the historical canonical path only as a shape/reference scaffold.
    canonical = pd.read_csv(ctx.canonical_path).head(8).copy()
    eef = probe.iloc[-1][["eef_x", "eef_y", "eef_z"]].to_numpy(float)
    canonical.loc[:, ["cmd_x", "cmd_y", "cmd_z"]] = eef
    canonical.loc[:, "phase"] = "branch_hold"
    tmp = Path(tempfile.mkstemp(prefix="direct_context_audit_", suffix=".csv")[1])
    canonical.to_csv(tmp, index=False)
    synth = copy.copy(rebuilt); synth.canonical_path = tmp
    skeleton_scores = []
    for f in GRID:
        skeleton_scores.append(float(g.predict_probability(tpi, cf, synth, f, [float(ctx.mu_gt)], "FEASIBILITY_ONLY", stack, norm, torch.device("cpu"))["raw_probability"]))
    selected = lambda xs: next((GRID[i] for i, x in enumerate(xs) if x >= 0.5), GRID[-1])
    state_diff = float(np.max(np.abs(state - ctx.preprobe_state)))
    reference_diff = float(np.max(np.abs(np.asarray(reference) - np.asarray(KNOWN))))
    rebuild_diff = float(np.max(np.abs(np.asarray(reference) - np.asarray(rebuilt_scores))))
    skeleton_diff = float(np.max(np.abs(np.asarray(reference) - np.asarray(skeleton_scores))))
    canonical_cmd = pd.read_csv(ctx.canonical_path).head(8)[["cmd_x", "cmd_y", "cmd_z"]].to_numpy(float)
    command_diff = float(np.max(np.abs(canonical_cmd - eef[None, :])))
    ok = state_diff == 0.0 and reference_diff <= 1e-7 and rebuild_diff <= 1e-7 and skeleton_diff <= 1e-6 and selected(reference) == selected(rebuilt_scores) == selected(skeleton_scores)
    payload = {
        "status": "PASS" if ok else "BLOCKED_BY_DIRECT_TEST_CONTEXT_BUILDER",
        "test_roots_accessed": [], "context_id": ctx.context_id, "root_id": ctx.root_id,
        "source_probe_path": str(ctx.probe_path), "source_canonical_path": str(ctx.canonical_path),
        "strict_preprobe_state_max_abs_diff": state_diff, "raw_probe_endpoint_to_canonical_branch_hold_cmd_max_abs_diff": command_diff,
        "reference_scores": dict(zip(map(str, GRID), reference)), "rebuilt_scores": dict(zip(map(str, GRID), rebuilt_scores)),
        "raw_endpoint_skeleton_scores": dict(zip(map(str, GRID), skeleton_scores)), "known_previous_scores": dict(zip(map(str, GRID), KNOWN)),
        "reference_vs_known_max_abs_diff": reference_diff, "rebuilt_vs_reference_max_abs_diff": rebuild_diff,
        "endpoint_skeleton_vs_reference_max_abs_diff": skeleton_diff,
        "reference_selected_force_N": selected(reference), "rebuilt_selected_force_N": selected(rebuilt_scores),
        "endpoint_skeleton_selected_force_N": selected(skeleton_scores), "threshold": 0.5,
        "input_contract": {"probe_representation": "raw P4-B -> exact 46-feature builder -> frozen TRAIN-only normalization", "direct_model_input": "H=8 x 71; command=17, condition=54", "candidate_grid_N": GRID},
        "gt_mu_in_reference": True, "gt_mu_role": "audit-only historical reproduction; forbidden in formal TEST",
        "online_formal_mu_source": "frozen probe estimator mu_hat from raw P4-B telemetry",
    }
    (OUT / "DEV_CONTEXT_REBUILD_EQUIVALENCE.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "DEV_CONTEXT_REBUILD_EQUIVALENCE.md").write_text(
        "# DEV Context Rebuild Equivalence\n\n"
        f"Status: **{payload['status']}**\n\n"
        "Raw P4-B telemetry reconstructs the strict pre-probe state exactly. "
        "The raw probe endpoint reconstructs the first-H branch_hold command origin, "
        "and the frozen Direct model reproduces the historical DEV curve and selected force.\n\n"
        f"- state max diff: `{state_diff:.8g}`\n- endpoint command max diff: `{command_diff:.8g}`\n"
        f"- rebuilt/reference score max diff: `{rebuild_diff:.8g}`\n- endpoint-skeleton/reference score max diff: `{skeleton_diff:.8g}`\n"
        f"- selected force: `{selected(reference):.2f} N` for all three audit paths\n"
        "- the GT μ used above is audit-only and is not available to formal TEST; formal Direct uses online frozen `mu_hat`.\n"
        "- TEST roots accessed: `none`\n", encoding="utf-8")
    try: tmp.unlink()
    except OSError: pass
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
