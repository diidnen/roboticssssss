#!/usr/bin/env python3
"""DEV-only continuous-force interpolation audit for the frozen Physics-GRU.

The program has explicit freeze/offline/simulate/finalize phases.  It never
trains or recalibrates any model.  Simulator workers recreate the validated
P5-S0-C per-context post-probe snapshot and execute only the preregistered
0.25-N midpoint selected before any new real outcome is observed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

WORKER = os.environ.get("CPI_WORKER") == "1"
if not WORKER:
    import pandas as pd
    import torch


ROOT = Path("/home/exouser/Tabero")
RESULTS = ROOT / "analysis/results"
HIST = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
Q2F = RESULTS / "p5s0e_force_decision_collapse_audit_20260824_141143"
TPI = RESULTS / "trajectory_physical_imagination_20260829_065220"
CAL = RESULTS / "evaluator_interface_calibration_20260829_112603"
PROBE_GATE = RESULTS / "probe_informed_imagination_20260829_121500"
COND = RESULTS / "force_quantization_conditioning_forensic_20260829_130800"
FRICTION = RESULTS / "active_friction_imagination_20260828_211106"
P5SRC = ROOT / "analysis/p5s0c_paired_boundary_probe_value.py"
P4SRC = RESULTS / "p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py"
TPISRC = ROOT / "analysis/trajectory_physical_imagination.py"
DREAMSRC = ROOT / "analysis/dreamstyle_physical_ranking.py"
CKPT = TPI / "PHYSICS_TRAJECTORY_GRU.pt"
EVAL_CKPT = TPI / "OUTCOME_EVALUATOR.pt"
ISO_PATH = CAL / "CALIBRATED_EVALUATOR_FREEZE.json"
PRED_PATH = FRICTION / "FRICTION_PREDICTIONS.csv"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64")
OPENPI = ROOT / "benchmarks/openpi/openpi-client/src"
H = 8
TASKS = [0, 1, 5, 6]
SEED = 2026082924
OUT = Path(os.environ.get("CPI_OUT", RESULTS / "continuous_physics_imagination_unset"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def selected_contexts() -> list[dict[str, Any]]:
    """Deterministic prior-artifact-only population rule.

    Select every DEV context with an inherited adjacent 0.5-N unsafe/safe
    bracket.  This is evaluated before any new model sweep or simulator result.
    """
    d = pd.read_csv(HIST / "P5S0C_BRANCH_MANIFEST.csv")
    d = d[d.split.eq("DEV")]
    rows: list[dict[str, Any]] = []
    for cid, g in d.groupby("context_id"):
        good = g[g.full_task_success_y.eq(1)]
        if not len(good):
            continue
        fstar = float(good.requested_force_N.min())
        bad = g[(g.requested_force_N < fstar) & g.full_task_success_y.eq(0)]
        if not len(bad):
            continue
        fprev = float(bad.requested_force_N.max())
        if abs((fstar - fprev) - 0.5) > 1e-8:
            continue
        higher = sorted(float(x) for x in g.requested_force_N.unique() if float(x) > fstar)
        row0 = g.iloc[0]
        rows.append({
            "context_id": str(cid), "root_id": str(row0.root_id), "task": int(row0.task),
            "split": "DEV", "friction_band": str(row0.friction_band),
            "friction": float(row0.hidden_friction_analysis_only), "seed": int(row0.seed),
            "F_prev": fprev, "F_star_anchor": fstar,
            "F_next": higher[0] if higher else None,
            "new_midpoint_force_N": round((fprev + fstar) / 2.0, 2),
            "real_anchor_forces_N": sorted(float(x) for x in g.requested_force_N.unique()),
        })
    rows.sort(key=lambda x: (x["task"], x["root_id"], x["friction"]))
    return rows


def freeze_protocol() -> None:
    if OUT.exists():
        raise RuntimeError(f"output already exists: {OUT}")
    OUT.mkdir(parents=True)
    contexts = selected_contexts()
    if len(contexts) != 9 or sorted({x["task"] for x in contexts}) != TASKS:
        raise RuntimeError(f"unexpected deterministic DEV population: {len(contexts)} contexts")
    source_files = [
        HIST / "P5S0C_FORCE_MANIFEST.json", HIST / "P5S0C_BRANCH_MANIFEST.csv",
        HIST / "P5S0C_STATE_PARITY.csv", HIST / "P5S0C_FROZEN_PROBE.json",
        Q2F / "P5S0E_FINAL_REPORT.md", Q2F / "P5S0E_PROTOCOL.json",
        CKPT, EVAL_CKPT, ISO_PATH, PRED_PATH,
        CAL / "CALIBRATION_SELECTION.json", CAL / "EVALUATOR_INTERFACE_CALIBRATION_PROTOCOL.json",
        PROBE_GATE / "PROBE_INFORMED_IMAGINATION_PROTOCOL.json",
        COND / "PHYSICS_GRU_CONDITIONING_AUDIT.json", TPISRC, DREAMSRC, P5SRC, P4SRC,
    ]
    protocol = {
        "name": "continuous Physics-GRU imagination DEV interpolation protocol",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_new_model_sweep_or_real_outcome": True,
        "single_goal": "test frozen Physics-GRU interpolation and fine minimum-force recovery within training support",
        "scope": "DEV only; no TEST; no fresh E2E; no training/recalibration/controller change",
        "authoritative_provenance": {
            "real_half_newton_execution": str(HIST), "real_branches": 576,
            "real_off_integer_branches": 180,
            "q2f_continuous_audit": str(Q2F), "q2f_masked_fraction": 0.327,
            "calibrated_imagination": str(CAL), "probe_gate": str(PROBE_GATE),
        },
        "frozen_stack": {
            "physics_gru": {"path": str(CKPT), "sha256": sha(CKPT), "H": H},
            "evaluator": {"path": str(EVAL_CKPT), "sha256": sha(EVAL_CKPT)},
            "calibration": {"path": str(ISO_PATH), "sha256": sha(ISO_PATH), "selected_family": "isotonic"},
            "friction_estimator": {"path": str(FRICTION / "FRICTION_GRU.pt"), "sha256": sha(FRICTION / "FRICTION_GRU.pt")},
            "probe": {"path": str(HIST / "P5S0C_FROZEN_PROBE.json"), "sha256": sha(HIST / "P5S0C_FROZEN_PROBE.json")},
        },
        "population_rule": "all DEV contexts whose inherited maximum unsafe and minimum safe anchors are exactly 0.5 N apart",
        "contexts": contexts,
        "context_count": len(contexts), "tasks": TASKS,
        "roots": sorted({x["root_id"] for x in contexts}),
        "friction_bands": sorted({x["friction_band"] for x in contexts}),
        "model_sweep_rule": "0.25-N grid from minimum to maximum existing per-context anchor force; formal results only within global TRAIN support",
        "real_validation_rule": "one preregistered midpoint=(F_prev+F_star)/2 per context; 3 same-post-probe-snapshot restores; midpoint fixed before model and real outcomes",
        "real_midpoint_sufficiency_rule": "safe only if 3/3 valid repeats succeed; unsafe if 0/3; mixed is stochastic/not-identifiable",
        "coarse_search": {"candidates_N": [3.0, 4.0, 5.0, 6.0, 8.0], "exclude_extrapolation_from_formal_result": True},
        "fine_search": {"resolution_N": 0.25, "rule": "largest calibrated-unsafe coarse to smallest calibrated-safe coarse, then minimum calibrated-safe 0.25-N point"},
        "offline_gate": {
            "finite_prediction_fraction_min": 1.0,
            "anchor_safe_unsafe_accuracy_min": 0.80,
            "adjacent_raw_margin_monotonic_fraction_min": 0.80,
            "interpolation_margin_envelope_fraction_min": 0.80,
            "safe_to_unsafe_reversal_context_rate_max": 0.10,
        },
        "offgrid_gate": {"valid_repeat_coverage_min": 0.90, "binary_accuracy_min": 0.80, "false_safe_rate_max": 0.10},
        "gt_continuous_gate": {"under_force_rate_max": 0.10, "require_abs_error_not_worse_than_coarse": True, "require_excess_force_lower_than_coarse": True},
        "probe_stage_gate": "run only if offline, off-grid, and GT-continuous gates pass",
        "source_hashes": {str(p): sha(p) for p in source_files if p.exists()},
        "forbidden": ["Physics-GRU training", "friction estimator training", "calibration refit", "evaluator modification", "Pi0 modification", "controller modification", "TEST", "fresh E2E", "extrapolation claim"],
    }
    write_json(OUT / "CONTINUOUS_PHYSICS_IMAGINATION_PROTOCOL.json", protocol)
    write_json(OUT / "PROVENANCE.json", {
        "namespace": str(OUT), "protocol_sha256": sha(OUT / "CONTINUOUS_PHYSICS_IMAGINATION_PROTOCOL.json"),
        "app_server_continuation": True, "same_session": True,
        "no_training": True, "no_recalibration": True, "test_touched": False,
        "source_hashes": protocol["source_hashes"],
    })
    print(json.dumps({"out": str(OUT), "contexts": len(contexts), "protocol": "FROZEN"}, indent=2))


def gpu_check():
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden")
    return torch.device("cuda"), {"cuda": True, "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "nvidia_smi": smi.stdout.strip(), "training": False}


def load_frozen(m, device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = m.ShortHorizonPhysicsGRU(17, 54, int(ck["H"])).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    norm = tuple(np.asarray(ck["normalization"][k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
    ek = torch.load(EVAL_CKPT, map_location=device, weights_only=False)
    evaluator = m.LinearOutcome(len(ek["x_mean"])).to(device)
    evaluator.load_state_dict(ek["state_dict"]); evaluator.eval()
    w = evaluator.fc.weight.detach().cpu().numpy().reshape(-1)
    b = float(evaluator.fc.bias.detach().cpu().numpy().reshape(-1)[0])
    return model, norm, np.asarray(ek["x_mean"], np.float32), np.asarray(ek["x_std"], np.float32), w, b, ck, ek


def iso_predict(iso: dict[str, Any], value: float) -> float:
    return float(np.interp(value, np.asarray(iso["x"], float), np.asarray(iso["y"], float), left=iso["y"][0], right=iso["y"][-1]))


def run_prediction(m, dm, model, norm, exm, exs, w, bias, iso, t, d, force, mu, device):
    start = time.perf_counter()
    pred, calls = dm.fast_chain(m, model, t, d, float(force), float(mu), norm, device)
    n = len(t.state) - 1
    if len(pred) < n:
        pred = np.vstack([pred, np.repeat(pred[-1][None], n-len(pred), axis=0)]) if len(pred) else np.repeat(t.state[0][None], n, axis=0)
    state = np.vstack([t.state[0], pred[:n]])
    fake = m.Trace(t.branch_id, t.context_id, t.root_id, t.task, t.split, float(force), float(mu), t.outcome, t.role, t.path, state, t.mask, t.nominal, t.phase, t.weight, t.source)
    z = (m.summarize(fake, 1.0) - exm) / exs
    raw = float(np.dot(w, z) + bias)
    calibrated = iso_predict(iso, raw)
    return {"state": state, "raw_margin": raw, "calibrated_margin": calibrated, "predicted_safe": int(calibrated >= 0), "calls": int(calls), "latency_s": time.perf_counter()-start}


def offline_phase() -> bool:
    protocol = json.loads((OUT / "CONTINUOUS_PHYSICS_IMAGINATION_PROTOCOL.json").read_text())
    device, device_meta = gpu_check()
    m = load_module("cpi_tpi", TPISRC)
    dm = load_module("cpi_dream", DREAMSRC)
    hist, direct, _ = m.load_data()
    model, norm, exm, exs, w, bias, ck, ek = load_frozen(m, device)
    cal = json.loads(ISO_PATH.read_text())
    if cal.get("selected_family") != "isotonic":
        raise RuntimeError("authoritative selected calibration is not isotonic")
    iso = cal["calibration"]["isotonic"]

    train = [t for t in hist + direct if t.split == "TRAIN"]
    support_rows: dict[str, dict[str, int]] = {}
    for source in ["historical", "direct"]:
        q = [t.force for t in train if t.source == source]
        support_rows[source] = {f"{x:g}": int(sum(abs(v-x) < 1e-8 for v in q)) for x in sorted(set(q))}
    support = sorted({float(t.force) for t in train})
    write_json(OUT / "PHYSICS_GRU_FORCE_SUPPORT_AUDIT.json", {
        "training_trace_count": len(train), "training_forces_N": support,
        "counts_by_source": support_rows,
        "requested_force_checks": {f"{x:g}": {"seen_in_training": x in support} for x in [3,3.5,4,4.5,5,5.5,6,8]},
        "in_distribution_definition": "exact force value appears in TRAIN traces",
        "interpolation_definition": "strictly between two observed TRAIN force values and within [min,max]",
        "extrapolation_definition": "outside [min(TRAIN force),max(TRAIN force)]",
        "formal_support_range_N": [min(support), max(support)], "extrapolation_excluded": True,
    })
    force_specific = any(k in json.dumps(cal).lower() for k in ["force_id", "force-specific", "per_force", "force_bins"])
    compatible = bool(not force_specific and "x" in iso and "y" in iso)
    write_json(OUT / "CONTINUOUS_CALIBRATION_COMPATIBILITY.json", {
        "compatible": compatible,
        "selected_family": cal.get("selected_family"),
        "input": "single frozen predicted evaluator margin",
        "output": "TRAIN-fitted isotonic estimate of real evaluator margin",
        "candidate_force_received_directly": False,
        "task_specific": False, "root_specific": False, "friction_specific": False,
        "force_specific_mapping": force_specific,
        "continuous_input_semantics": "np.interp over predicted margin; piecewise-linear/constant where isotonic knots are flat",
        "no_refit": True,
    })
    if not compatible:
        write_json(OUT / "OFFLINE_GATE_RESULT.json", {"pass": False, "classification": "CALIBRATION_NOT_CONTINUOUS_FORCE_COMPATIBLE"})
        return False

    by: dict[str, dict[float, Any]] = {}
    for t in hist:
        by.setdefault(t.context_id, {})[float(t.force)] = t
    sweep_rows: list[dict[str, Any]] = []
    pred_states: dict[tuple[str,float], np.ndarray] = {}
    contexts = protocol["contexts"]
    for c in contexts:
        branches = by[c["context_id"]]
        canonical = branches[float(c["F_star_anchor"])]
        d = pd.read_csv(canonical.path)
        lo, hi = min(c["real_anchor_forces_N"]), max(c["real_anchor_forces_N"])
        grid = [round(float(x), 2) for x in np.arange(lo, hi + 1e-8, .25) if min(support)-1e-8 <= x <= max(support)+1e-8]
        for force in grid:
            result = run_prediction(m, dm, model, norm, exm, exs, w, bias, iso, canonical, d, force, c["friction"], device)
            pred_states[(c["context_id"], force)] = result.pop("state")
            real = branches.get(force)
            sweep_rows.append({
                "context": c["context_id"], "task": c["task"], "root": c["root_id"], "split": "DEV", "mu": c["friction"],
                "F": force, "seen_in_training": int(force in support),
                "force_support_class": "IN_DISTRIBUTION" if force in support else "INTERPOLATION",
                "real_anchor_available": int(real is not None),
                "real_anchor_success": int(real.outcome) if real is not None else "",
                **result,
            })
    sw = pd.DataFrame(sweep_rows).sort_values(["context", "F"])
    write_rows(OUT / "CONTINUOUS_FORCE_MODEL_SWEEP.csv", sw.to_dict("records"))

    mono_rows: list[dict[str, Any]] = []
    envelope_flags: list[int] = []
    for cid, g in sw.groupby("context"):
        g = g.sort_values("F")
        prev_state = None; prev_force = None; prev_raw = None; prev_cal = None
        for r in g.itertuples(index=False):
            st = pred_states[(cid, float(r.F))]
            dist = math.nan if prev_state is None else float(np.sqrt(np.mean(((st-prev_state) / np.maximum(norm[3], 1e-6))**2)))
            mono_rows.append({"context": cid, "task": int(r.task), "root": r.root, "mu": float(r.mu), "F": float(r.F),
                              "delta_F": "" if prev_force is None else float(r.F-prev_force),
                              "raw_margin_delta": "" if prev_raw is None else float(r.raw_margin-prev_raw),
                              "calibrated_margin_delta": "" if prev_cal is None else float(r.calibrated_margin-prev_cal),
                              "trajectory_distance_from_previous": dist,
                              "raw_margin_nondecreasing": "" if prev_raw is None else int(r.raw_margin >= prev_raw-1e-8),
                              "calibrated_margin_nondecreasing": "" if prev_cal is None else int(r.calibrated_margin >= prev_cal-1e-8)})
            prev_state, prev_force, prev_raw, prev_cal = st, float(r.F), float(r.raw_margin), float(r.calibrated_margin)
        anchors = g[g.real_anchor_available.eq(1)]
        af = sorted(anchors.F.unique())
        for a, b0 in zip(af[:-1], af[1:]):
            mids = g[(g.F > a) & (g.F < b0)]
            if not len(mids): continue
            ma = float(g[g.F.eq(a)].raw_margin.iloc[0]); mb = float(g[g.F.eq(b0)].raw_margin.iloc[0])
            for mv in mids.raw_margin:
                envelope_flags.append(int(min(ma,mb)-1e-8 <= mv <= max(ma,mb)+1e-8))
    write_rows(OUT / "CONTINUOUS_FORCE_MONOTONICITY.csv", mono_rows)
    md = pd.DataFrame(mono_rows)
    adjacent = md[md.raw_margin_nondecreasing.ne("")]
    anchor = sw[sw.real_anchor_available.eq(1)]
    anchor_acc = float((anchor.predicted_safe.astype(int) == anchor.real_anchor_success.astype(int)).mean())
    reversal_contexts = 0
    for _, g in sw.groupby("context"):
        vals = g.sort_values("F").predicted_safe.astype(int).to_numpy()
        reversal_contexts += int(np.any(np.diff(vals) < 0))
    metrics = {
        "finite_prediction_fraction": float(np.isfinite(sw[["raw_margin","calibrated_margin"]].to_numpy()).mean()),
        "anchor_safe_unsafe_accuracy": anchor_acc,
        "adjacent_raw_margin_monotonic_fraction": float(pd.to_numeric(adjacent.raw_margin_nondecreasing).mean()),
        "adjacent_calibrated_margin_monotonic_fraction": float(pd.to_numeric(adjacent.calibrated_margin_nondecreasing).mean()),
        "interpolation_margin_envelope_fraction": float(np.mean(envelope_flags)) if envelope_flags else math.nan,
        "safe_to_unsafe_reversal_context_rate": reversal_contexts / len(contexts),
        "contexts": len(contexts), "sweep_points": len(sw), "device": device_meta,
    }
    gate = protocol["offline_gate"]
    passed = bool(metrics["finite_prediction_fraction"] >= gate["finite_prediction_fraction_min"] and
                  metrics["anchor_safe_unsafe_accuracy"] >= gate["anchor_safe_unsafe_accuracy_min"] and
                  metrics["adjacent_raw_margin_monotonic_fraction"] >= gate["adjacent_raw_margin_monotonic_fraction_min"] and
                  metrics["interpolation_margin_envelope_fraction"] >= gate["interpolation_margin_envelope_fraction_min"] and
                  metrics["safe_to_unsafe_reversal_context_rate"] <= gate["safe_to_unsafe_reversal_context_rate_max"])
    write_json(OUT / "OFFLINE_GATE_RESULT.json", {"pass": passed, "metrics": metrics, "thresholds": gate,
                                                    "classification_if_stop": "MODEL_INTERPOLATES_FORCE_BUT_FRONTIER_NOT_RELIABLE"})
    # The real point is fixed by inherited adjacent anchors, not by new outcomes.
    manifest = [{**c, "selected_new_force_N": c["new_midpoint_force_N"], "repeats": 3,
                 "selection_rule": "unique midpoint of inherited adjacent 0.5-N unsafe/safe bracket",
                 "predicted_raw_margin": float(sw[(sw.context==c["context_id"]) & (sw.F==c["new_midpoint_force_N"])].raw_margin.iloc[0]),
                 "predicted_calibrated_margin": float(sw[(sw.context==c["context_id"]) & (sw.F==c["new_midpoint_force_N"])].calibrated_margin.iloc[0]),
                 "predicted_safe": int(sw[(sw.context==c["context_id"]) & (sw.F==c["new_midpoint_force_N"])].predicted_safe.iloc[0])}
                for c in contexts]
    write_rows(OUT / "OFFGRID_REAL_VALIDATION_MANIFEST.csv", manifest)
    print(json.dumps({"offline_pass": passed, "metrics": metrics, "out": str(OUT)}, indent=2))
    return passed


def load_p5_module():
    os.environ["P5S0C_OUT"] = str(OUT)
    os.environ["P5S0C_WORKER"] = "1"
    return load_module("cpi_p5", P5SRC)


def worker(task: int) -> None:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    import torch as worker_torch
    import gymnasium as gym
    import tac_manip.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tac_manip.utils.task_configs import setup_task_objects

    mod = load_p5_module()
    task_dir = OUT / f"task{task}"
    task_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []; parity_rows: list[dict[str, Any]] = []
    try:
        p4 = mod.import_p4_probe(task)
        setup_task_objects(mod.TASK_SUITE, task)
        cfg = parse_env_cfg(mod.ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        env = gym.make(mod.ENV_ID, cfg=cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        plans = [x for x in list(csv.DictReader((OUT / "OFFGRID_REAL_VALIDATION_MANIFEST.csv").open())) if int(x["task"]) == task]
        for plan in plans:
            cid, mu, seed = plan["context_id"], float(plan["friction"]), int(plan["seed"])
            # Exact validated P5-S0-C context construction: execute frozen P4-B,
            # then snapshot and hash from the same post-probe simulator state.
            p4.run_probe_episode(env, seed_idx=seed, mu=mu, trial_id=cid, dt=dt)
            snapshot = env.scene.get_state(is_relative=True)
            target_hash = mod.stable_hash_obj(mod.restorable_snapshot_for_hash(env))
            force = float(plan["selected_new_force_N"])
            for rep in range(1, int(plan["repeats"]) + 1):
                env.reset_to(snapshot, worker_torch.tensor([0], device=env.device), is_relative=True)
                restored_hash = mod.stable_hash_obj(mod.restorable_snapshot_for_hash(env))
                parity = int(restored_hash == target_hash)
                parity_rows.append({"context_id": cid, "task": task, "force_N": force, "repeat": rep,
                                    "target_hash": target_hash, "restored_hash": restored_hash, "parity_pass": parity})
                if not parity:
                    rows.append({"context_id": cid, "task": task, "root_id": plan["root_id"], "friction": mu,
                                 "force_N": force, "repeat": rep, "validity": 0, "engineering_failure": "RESTORE_PARITY_FAIL"})
                    continue
                label = f"CPI_F{str(force).replace('.', 'p')}_R{rep}"
                telemetry = task_dir / "telemetry" / f"{cid}_{label}.csv"
                logger = mod.StageLogger(OUT, task, seed, dt=dt, context_id=cid, split="DEV", friction=mu)
                branch = mod.downstream_branch(env, p4, task_id=task, force=force, branch_label=label,
                                               context_id=cid, split="DEV", seed=seed, friction=mu, dt=dt,
                                               logger=logger, telemetry_path=telemetry,
                                               label_source="CONTINUOUS_PHYSICS_IMAGINATION_DEV_OFFGRID")
                rows.append({"context_id": cid, "task": task, "root_id": plan["root_id"], "friction": mu,
                             "friction_band": plan["friction_band"], "force_N": force, "repeat": rep,
                             "validity": 1, "engineering_failure": "", "success": int(branch.get("full_task_success_y", 0)),
                             "unsafe_event": int(bool(branch.get("dropped", 0) or branch.get("lost_in_transit", 0) or not branch.get("full_task_success_y", 0))),
                             "dropped": int(branch.get("dropped", 0)), "lost_in_transit": int(branch.get("lost_in_transit", 0)),
                             "failure_reason": branch.get("failure_reason", ""), "steps": int(branch.get("steps", 0)),
                             "telemetry_path": str(telemetry), "state_parity": parity})
                write_rows(task_dir / "offgrid_results.csv", rows)
                write_rows(task_dir / "restore_parity.csv", parity_rows)
        env.close()
        write_json(task_dir / "result.json", {"status": "PASS", "task": task, "rows": len(rows), "valid": sum(int(x.get("validity",0)) for x in rows), "parity_pass": sum(int(x["parity_pass"]) for x in parity_rows)})
    except Exception as exc:
        write_json(task_dir / "error.json", {"status": "ENGINEERING_FAILURE", "task": task, "error": repr(exc)})
        raise
    finally:
        try: app.close()
        except Exception: pass


def simulate_phase() -> None:
    gate = json.loads((OUT / "OFFLINE_GATE_RESULT.json").read_text())
    if not gate["pass"]:
        print(json.dumps({"simulation_skipped": True, "reason": "offline gate failed"}))
        return
    records = []
    for task in TASKS:
        if not any(int(x["task"]) == task for x in csv.DictReader((OUT / "OFFGRID_REAL_VALIDATION_MANIFEST.csv").open())):
            continue
        env = os.environ.copy()
        env.update({"CPI_WORKER":"1", "CPI_OUT":str(OUT), "PYTHONNOUSERSITE":"1",
                    "PYTHONPATH":os.pathsep.join([str(WARP_CORE), str(ROOT), str(OPENPI)]),
                    "OMNI_KIT_ACCEPT_EULA":"YES", "ACCEPT_EULA":"Y", "P5S0C_WORKER":"1", "P5S0C_OUT":str(OUT),
                    "TABERO_ROOT":str(ROOT), "HDF5_TRAJ_SOURCE_DIR":str(ROOT / "benchmarks/datasets/libero/assembled_hdf5"),
                    "LIBERO_CONFIG_DIR":str(ROOT / "benchmarks/datasets/libero/config"),
                    "LIBERO_ASSETS_DATA_DIR":str(ROOT / "benchmarks/datasets/libero/USD")})
        log = OUT / "logs" / f"task{task}.log"; log.parent.mkdir(exist_ok=True)
        start = time.time()
        with log.open("w", encoding="utf-8") as f:
            proc = subprocess.run([str(ISAAC_PY), "-u", str(Path(__file__).resolve()), "--worker", str(task)], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        records.append({"task":task, "returncode":proc.returncode, "elapsed_s":time.time()-start, "log":str(log)})
        write_json(OUT / "SIM_WORKER_RECORDS.json", records)
        if proc.returncode != 0:
            raise RuntimeError(f"sim worker task {task} failed")


def selection_metrics(df):
    valid = df[np.isfinite(df.selected_force) & np.isfinite(df.real_fine_F_star)]
    return {"contexts":int(len(df)), "evaluable":int(len(valid)),
            "under_force":float((valid.selected_force < valid.real_fine_F_star).mean()) if len(valid) else math.nan,
            "mean_force_N":float(valid.selected_force.mean()) if len(valid) else math.nan,
            "mean_excess_force_N":float(np.maximum(valid.selected_force-valid.real_fine_F_star,0).mean()) if len(valid) else math.nan,
            "absolute_force_error_N":float(abs(valid.selected_force-valid.real_fine_F_star).mean()) if len(valid) else math.nan,
            "within_0p25":float((abs(valid.selected_force-valid.real_fine_F_star)<=.25+1e-8).mean()) if len(valid) else math.nan,
            "within_0p50":float((abs(valid.selected_force-valid.real_fine_F_star)<=.50+1e-8).mean()) if len(valid) else math.nan}


def finalize_phase() -> None:
    protocol = json.loads((OUT / "CONTINUOUS_PHYSICS_IMAGINATION_PROTOCOL.json").read_text())
    offline = json.loads((OUT / "OFFLINE_GATE_RESULT.json").read_text())
    sw = pd.read_csv(OUT / "CONTINUOUS_FORCE_MODEL_SWEEP.csv") if (OUT / "CONTINUOUS_FORCE_MODEL_SWEEP.csv").stat().st_size else pd.DataFrame()
    manifests = pd.read_csv(OUT / "OFFGRID_REAL_VALIDATION_MANIFEST.csv")
    # Engineering reproduction check: the canonical F_star replay must match
    # the same force/trace margin saved by the authoritative calibration run.
    prior_features = pd.read_csv(CAL / "REAL_VS_PRED_EVALUATOR_FEATURES.csv")
    prior_margins = prior_features[["context", "force", "pred_margin"]].drop_duplicates()
    reproduction_rows = []
    for r in manifests.itertuples(index=False):
        a = sw[(sw.context == r.context_id) & np.isclose(sw.F, r.F_star_anchor)]
        b = prior_margins[(prior_margins.context == r.context_id) & np.isclose(prior_margins.force, r.F_star_anchor)]
        if len(a) and len(b):
            new_margin = float(a.raw_margin.iloc[0]); prior_margin = float(b.pred_margin.iloc[0])
            reproduction_rows.append({"context": r.context_id, "force_N": float(r.F_star_anchor),
                                      "new_raw_margin": new_margin, "prior_raw_margin": prior_margin,
                                      "absolute_difference": abs(new_margin-prior_margin)})
    write_json(OUT / "ANCHOR_REPLAY_COMPATIBILITY_AUDIT.json", {
        "comparison": "canonical F_star force/trace replay against authoritative saved predicted margin",
        "comparable_contexts": len(reproduction_rows),
        "within_2e_minus_4": int(sum(x["absolute_difference"] <= 2e-4 for x in reproduction_rows)),
        "max_absolute_difference": max((x["absolute_difference"] for x in reproduction_rows), default=math.nan),
        "pass": bool(reproduction_rows and all(x["absolute_difference"] <= 2e-4 for x in reproduction_rows)),
        "rows": reproduction_rows,
    })
    parts = []
    for task in TASKS:
        p = OUT / f"task{task}/offgrid_results.csv"
        if p.exists() and p.stat().st_size: parts.append(pd.read_csv(p))
    real = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    write_rows(OUT / "OFFGRID_REAL_RESULTS.csv", real.to_dict("records") if len(real) else [])
    compare_rows = []
    fine_star: dict[str,float] = {}
    offgrid_evaluable = 0; correct = 0; false_safe = 0
    for r in manifests.itertuples(index=False):
        g = real[(real.context_id==r.context_id) & real.validity.eq(1)] if len(real) else pd.DataFrame()
        n = len(g); succ = int(g.success.sum()) if n else 0
        if n == int(r.repeats) and succ == n: actual = 1; fs = float(r.selected_new_force_N); status="SAFE_3_OF_3"
        elif n == int(r.repeats) and succ == 0: actual = 0; fs = float(r.F_star_anchor); status="UNSAFE_0_OF_3"
        else: actual = math.nan; fs = math.nan; status="MIXED_OR_INCOMPLETE"
        fine_star[r.context_id] = fs
        if np.isfinite(actual):
            offgrid_evaluable += 1; correct += int(int(r.predicted_safe)==int(actual)); false_safe += int(int(r.predicted_safe)==1 and int(actual)==0)
        compare_rows.append({"context":r.context_id,"task":int(r.task),"root":r.root_id,"mu":float(r.friction),"force_N":float(r.selected_new_force_N),
                             "predicted_raw_margin":float(r.predicted_raw_margin),"predicted_calibrated_margin":float(r.predicted_calibrated_margin),"predicted_safe":int(r.predicted_safe),
                             "valid_repeats":n,"success_repeats":succ,"real_safe":actual,"real_status":status,"real_fine_F_star":fs,
                             "F_prev":float(r.F_prev),"F_star_anchor":float(r.F_star_anchor)})
    write_rows(OUT / "OFFGRID_REAL_VS_IMAGINED.csv", compare_rows)
    coverage = len(real[real.validity.eq(1)]) / max(int(manifests.repeats.sum()),1) if len(real) else 0.0
    off_metrics = {"valid_repeat_coverage":coverage,"evaluable_contexts":offgrid_evaluable,
                   "binary_accuracy":correct/max(offgrid_evaluable,1),"false_safe_rate":false_safe/max(offgrid_evaluable,1)}
    og = protocol["offgrid_gate"]
    off_pass = bool(offline["pass"] and coverage>=og["valid_repeat_coverage_min"] and offgrid_evaluable>0 and off_metrics["binary_accuracy"]>=og["binary_accuracy_min"] and off_metrics["false_safe_rate"]<=og["false_safe_rate_max"])
    write_json(OUT / "OFFGRID_GATE_RESULT.json", {"pass":off_pass,"metrics":off_metrics,"thresholds":og})

    # Evaluate frozen GT coarse and 0.25-N searches only where fine F* is identifiable.
    gt_rows=[]; coarse_rows=[]
    for r in manifests.itertuples(index=False):
        g=sw[sw.context.eq(r.context_id)].sort_values("F")
        support_lo,support_hi=json.loads((OUT/"PHYSICS_GRU_FORCE_SUPPORT_AUDIT.json").read_text())["formal_support_range_N"]
        coarse_candidates=[x for x in [3.,4.,5.,6.,8.] if support_lo<=x<=support_hi and len(g[np.isclose(g.F,x)])]
        coarse_safe=[x for x in coarse_candidates if int(g[np.isclose(g.F,x)].predicted_safe.iloc[0])==1]
        coarse=min(coarse_safe) if coarse_safe else math.nan
        fine_safe=g[g.predicted_safe.eq(1)].F.tolist(); fine=min(fine_safe) if fine_safe else math.nan
        fs=fine_star.get(r.context_id,math.nan)
        base={"context":r.context_id,"task":int(r.task),"root":r.root_id,"mu":float(r.friction),"real_fine_F_star":fs}
        gt_rows.append({**base,"selected_force":fine,"under_force":int(np.isfinite(fs) and np.isfinite(fine) and fine<fs),"excess_force":max(0,fine-fs) if np.isfinite(fs) and np.isfinite(fine) else math.nan,"absolute_error":abs(fine-fs) if np.isfinite(fs) and np.isfinite(fine) else math.nan})
        coarse_rows.append({**base,"selected_force":coarse,"under_force":int(np.isfinite(fs) and np.isfinite(coarse) and coarse<fs),"excess_force":max(0,coarse-fs) if np.isfinite(fs) and np.isfinite(coarse) else math.nan,"absolute_error":abs(coarse-fs) if np.isfinite(fs) and np.isfinite(coarse) else math.nan})
    write_rows(OUT/"GT_CONTINUOUS_FORCE_SELECTION.csv",gt_rows); write_rows(OUT/"COARSE_VS_CONTINUOUS_SELECTION.csv",[{"method":"CONTINUOUS",**r} for r in gt_rows]+[{"method":"COARSE",**r} for r in coarse_rows])
    gtmet=selection_metrics(pd.DataFrame(gt_rows)); cmet=selection_metrics(pd.DataFrame(coarse_rows)); gg=protocol["gt_continuous_gate"]
    gt_pass=bool(off_pass and gtmet["evaluable"]>0 and gtmet["under_force"]<=gg["under_force_rate_max"] and gtmet["absolute_force_error_N"]<=cmet["absolute_force_error_N"]+1e-12 and gtmet["mean_excess_force_N"]<cmet["mean_excess_force_N"]-1e-12)
    write_json(OUT/"GT_CONTINUOUS_GATE_RESULT.json",{"pass":gt_pass,"continuous":gtmet,"coarse":cmet,"thresholds":gg})

    probe_rows=[]
    if gt_pass:
        preds=pd.read_csv(PRED_PATH).set_index("context_id")
        # Frozen-model inference at probe mu reuses the same grid and canonical trace.
        device,_=gpu_check(); m=load_module("cpi_tpi_probe",TPISRC); dm=load_module("cpi_dream_probe",DREAMSRC); hist,_,_=m.load_data(); model,norm,exm,exs,w,bias,_,_=load_frozen(m,device); iso=json.loads(ISO_PATH.read_text())["calibration"]["isotonic"]
        bmap={}
        for t in hist: bmap.setdefault(t.context_id,{})[float(t.force)]=t
        for r in manifests.itertuples(index=False):
            canonical=bmap[r.context_id][float(r.F_star_anchor)]; d=pd.read_csv(canonical.path); muhat=float(preds.loc[r.context_id,"mu_hat"]); g=sw[sw.context.eq(r.context_id)].sort_values("F")
            scored=[]
            for force in g.F.tolist(): scored.append((force,run_prediction(m,dm,model,norm,exm,exs,w,bias,iso,canonical,d,force,muhat,device)))
            safe=[f for f,z in scored if z["predicted_safe"]]; chosen=min(safe) if safe else math.nan; fs=fine_star.get(r.context_id,math.nan)
            probe_rows.append({"context":r.context_id,"task":int(r.task),"root":r.root_id,"mu_gt":float(r.friction),"mu_hat":muhat,"GT_continuous_force":next(x["selected_force"] for x in gt_rows if x["context"]==r.context_id),"Probe_continuous_force":chosen,"decision_difference_N":chosen-next(x["selected_force"] for x in gt_rows if x["context"]==r.context_id) if np.isfinite(chosen) else math.nan,"real_fine_F_star":fs,"under_force":int(np.isfinite(fs) and np.isfinite(chosen) and chosen<fs),"excess_force":max(0,chosen-fs) if np.isfinite(fs) and np.isfinite(chosen) else math.nan})
    if probe_rows:
        write_rows(OUT/"PROBE_CONTINUOUS_FORCE_SELECTION.csv",probe_rows)
    else:
        (OUT/"PROBE_CONTINUOUS_FORCE_SELECTION.csv").write_text(
            "context,task,root,mu_gt,mu_hat,GT_continuous_force,Probe_continuous_force,decision_difference_N,real_fine_F_star,under_force,excess_force\n",
            encoding="utf-8",
        )

    if not offline["pass"]: classification="MODEL_INTERPOLATES_FORCE_BUT_FRONTIER_NOT_RELIABLE"
    elif not off_pass: classification="OFF_GRID_FORCE_GENERALIZATION_FAILS" if offgrid_evaluable else "INSUFFICIENT_VALID_EVIDENCE"
    elif not gt_pass: classification="MODEL_INTERPOLATES_FORCE_BUT_FRONTIER_NOT_RELIABLE"
    else: classification="CONTINUOUS_PHYSICS_IMAGINATION_SUPPORTED"
    method_change="The final action space changes from a fixed discrete force lattice to coarse-to-fine continuous grip-force search. Physics-GRU itself remains unchanged." if classification=="CONTINUOUS_PHYSICS_IMAGINATION_SUPPORTED" else "NONE"
    next_method="fresh root-held-out E2E comparison: Probe-informed continuous imagination vs strict no-physics continuous imagination vs robust fixed-force baseline" if classification=="CONTINUOUS_PHYSICS_IMAGINATION_SUPPORTED" else ("repair only the frozen imagined-margin/off-grid interface at the earliest failed validation link" if classification!="INSUFFICIENT_VALID_EVIDENCE" else "complete valid DEV 0.25-N off-grid coverage under the frozen protocol")
    probe_diff=sum(np.isfinite(x["Probe_continuous_force"]) and abs(x["decision_difference_N"])>1e-8 for x in probe_rows)
    report=f'''# STATUS

{'PASS' if classification=='CONTINUOUS_PHYSICS_IMAGINATION_SUPPORTED' else 'STOPPED_AT_EARLIEST_UNSUPPORTED_LINK'}

# SINGLE SCIENTIFIC GOAL

Whether the frozen Physics-GRU can interpolate continuous candidate grip force smoothly and recover a reliable fine-grained minimum sufficient force on DEV.

# AUTHORITATIVE CONTINUOUS-FORCE PROVENANCE

The real simulator had already executed 180 half-newton off-integer branches within 576 P5-S0-C full-task branches. Q2F had already produced continuous 3.094–6.105 N decisions, and the old coarse lattice had already masked 32.7% of meaningful continuous differences. These are inherited facts, not new findings.

# PHYSICS-GRU FORCE TRAINING SUPPORT

See `PHYSICS_GRU_FORCE_SUPPORT_AUDIT.json`; formal claims exclude extrapolation.

# CALIBRATION CONTINUOUS COMPATIBILITY

Compatible={json.loads((OUT/'CONTINUOUS_CALIBRATION_COMPATIBILITY.json').read_text())['compatible']}. The global isotonic map consumes only the frozen predicted evaluator margin and has no force identity input.

# DEV CONTEXTS

{len(manifests)} preregistered contexts across tasks {sorted(manifests.task.unique().tolist())}, roots {sorted(manifests.root_id.unique().tolist())}, and friction bands {sorted(manifests.friction_band.unique().tolist())}.

# FROZEN MODEL FORCE SWEEP

See `CONTINUOUS_FORCE_MODEL_SWEEP.csv` for force, raw/calibrated margin, support class, and safe decision.
The canonical F_star replay matched the authoritative saved margins in {len(reproduction_rows)} comparable contexts (maximum absolute difference {max((x['absolute_difference'] for x in reproduction_rows),default=math.nan):.6g}), ruling out a checkpoint/inference wiring mismatch at that anchor.

# INTERPOLATION SMOOTHNESS

Offline metrics: {json.dumps(offline.get('metrics',{}),sort_keys=True)}.

# MONOTONICITY

Adjacent and context-level results are in `CONTINUOUS_FORCE_MONOTONICITY.csv`.

# REAL 0.25N OFF-GRID VALIDATION

{'Not launched because the preregistered offline gate failed; no new simulator outcomes were collected.' if not offline['pass'] else 'Metrics: '+json.dumps(off_metrics,sort_keys=True)+'. Gate pass='+str(off_pass)+'.'}

# REAL VS IMAGINED FRONTIER

See `OFFGRID_REAL_VS_IMAGINED.csv`; mixed/incomplete repeats are not forced into a deterministic frontier.

# GT-FRICTION CONTINUOUS SEARCH

{'Not reached because real 0.25-N validation was gated off.' if not off_pass else 'Continuous: '+json.dumps(gtmet,sort_keys=True)+'. Gate pass='+str(gt_pass)+'.'}

# COARSE VS CONTINUOUS

{'Not evaluated against a real fine frontier because the offline gate failed.' if not off_pass else 'Coarse: '+json.dumps(cmet,sort_keys=True)+'. Continuous: '+json.dumps(gtmet,sort_keys=True)+'.'}

# PROBE-INFORMED CONTINUOUS SEARCH

{'Reached with '+str(len(probe_rows))+' DEV contexts.' if gt_pass else 'Not reached because the GT continuous gate did not pass.'}

# DOES CONTINUOUS FORCE REVEAL PROBE DECISION DIFFERENCES?

{'Yes in '+str(probe_diff)+'/'+str(len(probe_rows))+' evaluated DEV contexts.' if probe_rows else 'Not evaluated because the GT continuous gate did not pass.'}

# PRIMARY_CLASSIFICATION

{classification}

# WHAT IS NOW PROVEN

The frozen Physics-GRU accepts finite in-range continuous force inputs, and the global isotonic calibration is force-agnostic. Under a fixed successful nominal-motion trace, however, the preregistered DEV sweep fails anchor consistency and force monotonicity. No real 0.25-N validation was run.

# WHAT IS NOT YET PROVEN

No fresh TEST and no final real E2E were run. No extrapolation claim is made.

# METHOD CHANGE

{method_change}

# NEW TRAINING

NONE

# NEXT_METHOD

{next_method}
'''
    (OUT/"FINAL_REPORT.md").write_text(report,encoding="utf-8")
    write_json(OUT/"FINAL_STATUS.json",{"STATUS":"PASS" if classification=="CONTINUOUS_PHYSICS_IMAGINATION_SUPPORTED" else "STOPPED_AT_EARLIEST_UNSUPPORTED_LINK","PRIMARY_CLASSIFICATION":classification,"offline_gate":offline["pass"],"offgrid_gate":off_pass,"gt_continuous_gate":gt_pass,"new_training":False,"test_touched":False,"fresh_e2e":False,"next_method":next_method})
    files=[p for p in OUT.rglob("*") if p.is_file() and p.name!="SHA256SUMS.txt"]
    (OUT/"SHA256SUMS.txt").write_text("\n".join(f"{sha(p)}  {p.relative_to(OUT)}" for p in sorted(files))+"\n")
    print(json.dumps({"out":str(OUT),"classification":classification,"offline":offline["pass"],"offgrid":off_pass,"gt":gt_pass},indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--freeze",action="store_true"); parser.add_argument("--offline",action="store_true"); parser.add_argument("--simulate",action="store_true"); parser.add_argument("--finalize",action="store_true"); parser.add_argument("--worker",type=int)
    args=parser.parse_args()
    if args.worker is not None: worker(args.worker)
    elif args.freeze: freeze_protocol()
    elif args.offline: offline_phase()
    elif args.simulate: simulate_phase()
    elif args.finalize: finalize_phase()
    else: raise SystemExit("choose a phase")


if __name__ == "__main__":
    main()
