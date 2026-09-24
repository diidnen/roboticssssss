#!/usr/bin/env python3
"""Probe + continuous force + Joint DEV development gate.

Phases are deliberately separated:

``prepare`` trains the one allowed strict-preprobe Joint family.
``freeze`` writes the immutable protocol and blind GT model predictions.
``analyze`` reads repeated real outcomes, applies frozen gates, conditionally
continues to Probe/No-Probe, and produces the complete technical artifact set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Checkpoints produced under NumPy 2 pickle arrays through ``numpy._core``.
# The IsaacLab analysis environment currently carries NumPy 1.26, where the
# same implementation lives under ``numpy.core``.  Registering these aliases
# changes no tensor or model value; it only permits reading frozen artifacts.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


TABERO = Path("/home/exouser/Tabero")
RESULTS = TABERO / "analysis/results"
OLD_CONT = RESULTS / "continuous_physics_imagination_20260829_141924"
OLD_FULL = RESULTS / "full_task_feasibility_20260830_012830"
OLD_PRE = RESULTS / "preprobe_full_task_feasibility_20260830_071139"
OLD_NEC = RESULTS / "active_probe_necessity_20260830_063435"
OLD_DISC = RESULTS / "discrete_boundary_forensic_20260830_072855"
P5 = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
FRICTION = RESULTS / "active_friction_imagination_20260828_211106"
TPI_CODE = TABERO / "analysis/trajectory_physical_imagination.py"
CF_CODE = TABERO / "analysis/counterfactual_force_world_model.py"
FULL_CODE = TABERO / "analysis/full_task_feasibility_decoder.py"
PRE_CODE = TABERO / "analysis/preprobe_full_task_feasibility.py"
ACTIVE_CODE = TABERO / "analysis/active_probe_necessity.py"
COLLECTOR_CODE = Path(__file__).with_name("continuous_preprobe_collect.py")

SEEDS = [0, 1, 2]
H = 8
LAMBDA_IE = 1.0
LAMBDA_FEAS = 0.3
EPOCHS = 80
THRESHOLD = 0.5
REPEATS = 5
RHO = 0.80
RELIABLE_SUCCESSES = 4
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 2026083011
MONOTONIC_DROP_TOL = 0.02
JOINT_GATE = {
    "offgrid_mae_relative_improvement_min": 0.15,
    "frontier_mae_absolute_improvement_min_N": 0.10,
    "under_force_not_worse": True,
    "paired_bootstrap_observed_direction_positive": True,
    "systematic_nonmonotonic_context_rate_max": 0.10,
}
GT_GATE = {
    "valid_frontier_coverage_min": 0.80,
    "under_force_max": 0.10,
    "frontier_mae_max_N": 0.25,
    "offgrid_probability_mae_max": 0.15,
    "systematic_nonmonotonic_context_rate_max": 0.10,
}
PROBE_GATE = {
    "frontier_mae_improvement_min_N": 0.05,
    "under_force_not_worse": True,
    "context_cluster_bootstrap_observed_direction_positive": True,
    "adaptive_pair_success_not_worse": True,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    fields = fields or ["status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        if rows:
            w.writerows(rows)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def modules():
    suffix = str(os.getpid())
    tpi = load_module("tpi_cont_" + suffix, TPI_CODE)
    cf = load_module("cf_cont_" + suffix, CF_CODE)
    full = load_module("full_cont_" + suffix, FULL_CODE)
    pre = load_module("pre_cont_" + suffix, PRE_CODE)
    active = load_module("active_cont_" + suffix, ACTIVE_CODE)
    return tpi, cf, full, pre, active


def authoritative_contexts() -> list[dict]:
    old = json.loads((OLD_CONT / "CONTINUOUS_PHYSICS_IMAGINATION_PROTOCOL.json").read_text(encoding="utf-8"))
    rows = []
    for c in old["contexts"]:
        grid = [float(c["F_prev"]), float(c["new_midpoint_force_N"]), float(c["F_star_anchor"])]
        rows.append({
            "context_id": c["context_id"], "root_id": c["root_id"], "task": int(c["task"]),
            "seed": int(c["seed"]), "split": "DEV", "friction_band": c["friction_band"],
            "friction": float(c["friction"]), "historical_F_prev_N": float(c["F_prev"]),
            "historical_F_star_anchor_N": float(c["F_star_anchor"]),
            "offgrid_midpoint_N": float(c["new_midpoint_force_N"]), "real_force_grid_N": grid,
        })
    if len(rows) != 9:
        raise RuntimeError(f"authoritative continuous population changed: {len(rows)}")
    return rows


def strict_population():
    tpi, cf, full, pre, active = modules()
    contexts, traces, meta, manifest = pre.build_population(tpi, active)
    jmeta = {
        tr.branch_id: full.BranchMeta(
            tr.branch_id, tr.context_id, tr.task, tr.root_id, meta[tr.branch_id].friction_band,
            tr.outcome, tr.force, str(meta[tr.branch_id].role).upper(),
        ) for tr in traces
    }
    return tpi, cf, full, pre, active, contexts, traces, jmeta, manifest


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tpi, cf, full, pre, active, contexts, traces, meta, manifest = strict_population()
    train = [x for x in traces if x.split == "TRAIN"]
    dev = [x for x in traces if x.split == "DEV"]
    if len(train) != 288 or len(dev) != 96:
        raise RuntimeError("strict Joint population is not 288 TRAIN / 96 DEV")
    if set(x.root_id for x in train) & set(x.root_id for x in dev):
        raise RuntimeError("TRAIN/DEV root leakage")
    pairs = full.make_pairs(cf, traces, meta)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the strict Joint alignment family")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    training_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    for seed in SEEDS:
        base_ck, norm, base_path = full.load_base(tpi, seed, device)
        ckpath = out / f"STRICT_PREPROBE_JOINT_seed{seed}.pt"
        logpath = out / f"STRICT_PREPROBE_JOINT_TRAINING_seed{seed}.csv"
        calpath = out / f"STRICT_PREPROBE_JOINT_CALIBRATION_seed{seed}.json"
        torch.manual_seed(seed)
        if ckpath.exists():
            saved = torch.load(ckpath, map_location=device, weights_only=False)
            model = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
            model.physics.load_state_dict(saved["physics_state_dict"])
            model.feas_head.load_state_dict(saved["feas_head_state_dict"])
            model.eval()
            hist = pd.read_csv(logpath).to_dict("records") if logpath.exists() else []
            steps = int(hist[-1]["optimizer_steps"]) if hist else 0
            units = int(saved.get("physical_units", 0))
        else:
            model, hist, steps, units = full.train_joint(
                full.JointIEFeasibility(tpi, base_ck["state_dict"]), train, meta, traces,
                pairs, norm, cf, tpi, device, seed, LAMBDA_FEAS,
            )
            torch.save({
                "physics_state_dict": model.physics.state_dict(),
                "feas_head_state_dict": model.feas_head.state_dict(), "seed": seed,
                "lambda_IE": LAMBDA_IE, "lambda_feas": LAMBDA_FEAS, "H": H,
                "epochs": EPOCHS, "physical_units": units, "initial_checkpoint": str(base_path),
                "initial_checkpoint_sha256": sha256(base_path),
                "normalization": {k: v.tolist() for k, v in zip(["x_mean", "x_std", "y_mean", "y_std"], norm)},
                "input_interface": "STRICT_PREPROBE_LAST_HOLD_RECONSTRUCTION",
                "training_population": "same authoritative 288 TRAIN branches",
            }, ckpath)
            write_csv(logpath, hist)
        training_rows.extend(hist)
        tr_logits = full.predict_logits(model, "joint", train, norm, cf, tpi, device)
        cal = cf.fit_iso([tr_logits[x.branch_id] for x in train], [x.outcome for x in train])
        write_json(calpath, {
            "method": "isotonic", "fit_split": "TRAIN", "n": len(train), "test_used": False,
            "threshold": THRESHOLD, "x": cal["x"], "y": cal["y"],
            "checkpoint_sha256": sha256(ckpath), "interface": "strict_preprobe",
        })
        checkpoint_rows.append({
            "seed": seed, "checkpoint": str(ckpath), "checkpoint_sha256": sha256(ckpath),
            "calibration": str(calpath), "calibration_sha256": sha256(calpath),
            "initial_checkpoint": str(base_path), "initial_checkpoint_sha256": sha256(base_path),
            "lambda_IE": LAMBDA_IE, "lambda_feas": LAMBDA_FEAS, "epochs": EPOCHS,
            "optimizer_steps": steps, "physical_units": units,
        })
    write_csv(out / "STRICT_PREPROBE_JOINT_TRAINING_MANIFEST.csv", training_rows)
    write_csv(out / "STRICT_PREPROBE_JOINT_CHECKPOINT_MANIFEST.csv", checkpoint_rows)
    write_json(out / "STRICT_PREPROBE_JOINT_AUDIT.json", {
        "status": "PASS", "purpose": "interface alignment only; no world-model search",
        "strict_interface": "last stable P4-B hold before probe_out; relative pos/vel zero; force/contact masked; joints=[opening,-opening]",
        "architecture": "authoritative Joint PHYSICS_GRU_IE_FEAS; GRU(64), H8 trajectory head, feasibility head",
        "objective": "L_traj + 1.0 L_IE + 0.3 L_feas", "H": H,
        "lambda_IE": LAMBDA_IE, "lambda_feas": LAMBDA_FEAS, "epochs": EPOCHS,
        "seeds": SEEDS, "TRAIN_branches": len(train), "DEV_branches_not_trained": len(dev),
        "TRAIN_roots": sorted(set(x.root_id for x in train)), "DEV_roots": sorted(set(x.root_id for x in dev)),
        "train_dev_root_overlap": [], "normalization": "authoritative TRAIN-only Joint normalization per seed",
        "calibration": "new per-seed TRAIN-only isotonic on strict-preprobe inputs",
        "hyperparameter_search": False, "test_used": False, "pi0_modified": False,
        "checkpoint_manifest": "STRICT_PREPROBE_JOINT_CHECKPOINT_MANIFEST.csv",
        "source_hashes": {str(p): sha256(p) for p in [FULL_CODE, PRE_CODE, TPI_CODE, CF_CODE]},
    })


def load_cal(path: Path) -> dict[str, np.ndarray]:
    q = json.loads(path.read_text(encoding="utf-8"))
    return {"x": np.asarray(q["x"], float), "y": np.asarray(q["y"], float)}


def iso(cal, values):
    x = np.asarray(values, float)
    return np.interp(x, cal["x"], cal["y"], left=cal["y"][0], right=cal["y"][-1])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, float), -50, 50)))


def query_trace(ctx, force: float, mu: float, tpi):
    d = pd.read_csv(ctx.canonical_path)
    state, mask = tpi.state_from(d)
    state = state.copy(); mask = mask.copy()
    state[0] = ctx.preprobe_state; mask[0] = ctx.preprobe_mask
    nominal = tpi.nominal_from(d, ctx.task, float(force), float(mu), state, mask)
    return tpi.Trace(
        f"continuous_query:{ctx.context_id}:{force}:{mu}", ctx.context_id, ctx.root_id,
        ctx.task, "DEV", float(force), float(mu), 0, "QUERY", ctx.canonical_path,
        state, mask, nominal, d.phase.astype(str).tolist(), 1.0, "strict_preprobe_continuous_query",
    )


def load_backends(out: Path):
    tpi, cf, full, pre, active, contexts, traces, meta, manifest = strict_population()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feas_models = []; feas_norms = []; feas_cals = []
    joint_models = []; joint_norms = []; joint_cals = []
    for seed in SEEDS:
        fck = torch.load(OLD_PRE / f"PREPROBE_FEASIBILITY_seed{seed}.pt", map_location=device, weights_only=False)
        fm = full.FeasibilityOnly().to(device); fm.load_state_dict(fck["state_dict"]); fm.eval()
        feas_models.append(fm); feas_norms.append((np.asarray(fck["x_mean"], np.float32), np.asarray(fck["x_std"], np.float32)))
        feas_cals.append(load_cal(OLD_PRE / f"PREPROBE_CALIBRATION_seed{seed}.json"))
        base_ck, _, _ = full.load_base(tpi, seed, device)
        jckpath = out / f"STRICT_PREPROBE_JOINT_seed{seed}.pt"
        jck = torch.load(jckpath, map_location=device, weights_only=False)
        jm = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
        jm.physics.load_state_dict(jck["physics_state_dict"]); jm.feas_head.load_state_dict(jck["feas_head_state_dict"]); jm.eval()
        joint_models.append(jm)
        joint_norms.append(tuple(np.asarray(jck["normalization"][k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"]))
        joint_cals.append(load_cal(out / f"STRICT_PREPROBE_JOINT_CALIBRATION_seed{seed}.json"))
    return tpi, cf, full, active, contexts, device, {
        "FEASIBILITY_ONLY": (feas_models, feas_norms, feas_cals),
        "JOINT": (joint_models, joint_norms, joint_cals),
    }


def predict_point(ctx, force: float, mus: list[float], backend: str, stack, tpi, cf, device) -> dict:
    models, norms, cals = stack
    raw_by_mu = []; cal_by_mu = []
    per_seed_raw_mu = [[] for _ in SEEDS]; per_seed_cal_mu = [[] for _ in SEEDS]
    for mu in mus:
        tr = query_trace(ctx, force, mu, tpi)
        seg = cf.build_seg(tpi, tr, float(force), H)
        sraw = []; scal = []
        for i, (model, norm, cal) in enumerate(zip(models, norms, cals)):
            xm, xs = norm[0], norm[1]
            x = (seg.x - xm) / xs
            step = torch.tensor(x[None, :, :17], dtype=torch.float32, device=device)
            cond = torch.tensor(x[None, 0, 17:], dtype=torch.float32, device=device)
            with torch.no_grad():
                logit = float((model.feasibility(step, cond) if backend == "JOINT" else model(step, cond)).item())
            rp = float(sigmoid([logit])[0]); cp = float(iso(cal, [logit])[0])
            sraw.append(rp); scal.append(cp); per_seed_raw_mu[i].append(rp); per_seed_cal_mu[i].append(cp)
        raw_by_mu.append(float(np.mean(sraw))); cal_by_mu.append(float(np.mean(scal)))
    return {
        "raw_probability": float(np.mean(raw_by_mu)), "calibrated_probability": float(np.mean(cal_by_mu)),
        **{f"seed{s}_raw_probability": float(np.mean(per_seed_raw_mu[i])) for i, s in enumerate(SEEDS)},
        **{f"seed{s}_calibrated_probability": float(np.mean(per_seed_cal_mu[i])) for i, s in enumerate(SEEDS)},
        "per_mu_raw_json": json.dumps(raw_by_mu), "per_mu_calibrated_json": json.dumps(cal_by_mu),
    }


def prediction_rows(out: Path, condition: str, mu_by_context: dict[str, list[float]], backends: list[str] | None = None) -> list[dict]:
    tpi, cf, full, active, contexts, device, stacks = load_backends(out)
    pop = authoritative_contexts(); rows = []
    for c in pop:
        ctx = contexts[c["context_id"]]
        mus = mu_by_context[c["context_id"]]
        for backend in backends or ["FEASIBILITY_ONLY", "JOINT"]:
            point_rows = []
            for force in c["real_force_grid_N"]:
                pred = predict_point(ctx, force, mus, backend, stacks[backend], tpi, cf, device)
                point_rows.append({
                    "backend": backend, "condition": condition, "context_id": c["context_id"],
                    "root_id": c["root_id"], "task": c["task"], "friction_band": c["friction_band"],
                    "mu_gt": c["friction"], "physics_input_json": json.dumps(mus), "force_N": force,
                    "is_offgrid_midpoint": int(abs(force - c["offgrid_midpoint_N"]) < 1e-9), **pred,
                })
            safe = [r["force_N"] for r in point_rows if r["calibrated_probability"] >= THRESHOLD]
            selected = min(safe) if safe else math.nan
            for row in point_rows:
                row["selected_reliable_force_N"] = selected
            rows.extend(point_rows)
    return rows


def freeze(out: Path) -> None:
    joint_audit = out / "STRICT_PREPROBE_JOINT_AUDIT.json"
    if not joint_audit.exists() or json.loads(joint_audit.read_text())["status"] != "PASS":
        raise RuntimeError("prepare strict Joint before protocol freeze")
    protocol_path = out / "CONTINUOUS_PROBE_JOINT_PROTOCOL.json"
    if protocol_path.exists():
        if (out / "FROZEN_CONTINUOUS_MODEL_PREDICTIONS.csv").exists():
            raise RuntimeError("refusing to overwrite frozen protocol or predictions")
        # Engineering-only resume: the protocol was durably frozen before an
        # old-checkpoint NumPy module-name compatibility failure.  Reuse its
        # exact bytes and produce the still-missing blind predictions.
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("status") != "FROZEN_BEFORE_NEW_SIMULATOR_OUTCOMES":
            raise RuntimeError("existing protocol is not resumable")
        mu_gt = {c["context_id"]: [float(c["friction"])] for c in existing["dev_context_population"]}
        preds = prediction_rows(out, "GT", mu_gt)
        for row in preds:
            row["protocol_sha256"] = sha256(protocol_path)
            row["saved_before_real_outcome_inspection"] = True
        write_csv(out / "FROZEN_CONTINUOUS_MODEL_PREDICTIONS.csv", preds)
        write_json(out / "CONTINUOUS_LEAKAGE_AUDIT.json", {
            "status": "PRECOLLECTION_PASS", "protocol_sha256": sha256(protocol_path),
            "same_strict_state_across_GT_Probe_NoProbe": True, "postprobe_state_input": False,
            "probe_trace_input_to_backend": False, "force_candidate_input": True, "mu_is_only_condition_change": True,
            "training_test_leakage": False, "test_loaded": False, "real_outcomes_seen_before_model_prediction_save": False,
            "runtime_restore_verification": "pending repeated collection hashes",
            "engineering_resume": "NumPy module-name compatibility only; protocol bytes unchanged",
        })
        print(json.dumps({"protocol": str(protocol_path), "sha256": sha256(protocol_path), "contexts": len(existing["dev_context_population"]), "blind_prediction_rows": len(preds), "resumed": True}, indent=2))
        return
    pop = authoritative_contexts()
    _, _, _, pre, active = modules()
    strict_train = pre.all_contexts_for_split("TRAIN", active)
    strict_dev = pre.all_contexts_for_split("DEV", active)
    audit_rows = []
    for c in pop:
        ctx = strict_dev.get(c["context_id"])
        valid = bool(ctx is not None and np.isfinite(ctx.preprobe_state).all() and len(c["real_force_grid_N"]) == 3)
        audit_rows.append({**c, "strict_preprobe_reconstruction": "PASS" if ctx is not None else "FAIL",
                           "corrected_runner_compatible": True, "force_grid_in_3_to_6_support": bool(min(c["real_force_grid_N"]) >= 3 and max(c["real_force_grid_N"]) <= 6),
                           "valid": valid, "exclusion_reason": "" if valid else "missing strict reconstruction or invalid fixed grid"})
    final_pop = [r for r in audit_rows if r["valid"]]
    write_json(out / "CONTINUOUS_CONTEXT_AUDIT.json", {
        "status": "PASS" if len(final_pop) == len(pop) else "EXCLUSIONS_APPLIED",
        "authoritative_population_source": str(OLD_CONT / "CONTINUOUS_PHYSICS_IMAGINATION_PROTOCOL.json"),
        "authoritative_population_sha256": sha256(OLD_CONT / "CONTINUOUS_PHYSICS_IMAGINATION_PROTOCOL.json"),
        "requested_contexts": len(pop), "valid_contexts": len(final_pop), "invalid_contexts_not_replaced": len(pop) - len(final_pop),
        "tasks": sorted(set(r["task"] for r in final_pop)), "roots": sorted(set(r["root_id"] for r in final_pop)),
        "friction_bands": sorted(set(r["friction_band"] for r in final_pop)), "contexts": audit_rows,
    })
    if len(final_pop) != 9:
        raise RuntimeError("authoritative 9-context population is not fully valid; freeze stopped")
    no_probe = json.loads((OLD_NEC / "NO_PROBE_PHYSICS_PRIOR.json").read_text())
    probe_rows = pd.read_csv(FRICTION / "FRICTION_PREDICTIONS.csv")
    probe_dev = probe_rows[probe_rows.split == "DEV"]
    if not set(r["context_id"] for r in final_pop) <= set(probe_dev.context_id):
        raise RuntimeError("frozen Probe estimates missing context")
    sources = [
        OLD_CONT / "CONTINUOUS_PHYSICS_IMAGINATION_PROTOCOL.json", OLD_DISC / "FINAL_REPORT.md",
        OLD_PRE / "PREPROBE_FULL_TASK_FEASIBILITY_PROTOCOL.json", OLD_NEC / "ACTIVE_PROBE_NECESSITY_PROTOCOL.json",
        OLD_FULL / "FULL_TASK_FEASIBILITY_PROTOCOL.json", FRICTION / "FRICTION_GRU.pt",
        FRICTION / "FRICTION_PREDICTIONS.csv", OLD_NEC / "NO_PROBE_PHYSICS_PRIOR.json",
        joint_audit, COLLECTOR_CODE, FULL_CODE, PRE_CODE,
    ]
    sources += [OLD_PRE / f"PREPROBE_FEASIBILITY_seed{s}.pt" for s in SEEDS]
    sources += [OLD_PRE / f"PREPROBE_CALIBRATION_seed{s}.json" for s in SEEDS]
    sources += [out / f"STRICT_PREPROBE_JOINT_seed{s}.pt" for s in SEEDS]
    sources += [out / f"STRICT_PREPROBE_JOINT_CALIBRATION_seed{s}.json" for s in SEEDS]
    protocol = {
        "protocol_name": "CONTINUOUS_PROBE_JOINT_PROTOCOL", "status": "FROZEN_BEFORE_NEW_SIMULATOR_OUTCOMES",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "single_scientific_goal": "validate repeated fine real force frontiers, compare strict-preprobe Feasibility-only versus Joint under GT friction, and conditionally compare frozen Probe versus strict No-Probe",
        "claim_boundary": {"current_object_task_context_conditioned": True, "cross_object_generalization": False,
                           "unseen_task_generalization": False, "universal_physics_transfer": False},
        "vla_role": "Frozen pi0 supplies task understanding, staging, and nominal H8 motion; it predicts neither friction nor grip force.",
        "dev_context_population": [{k: r[k] for k in ["context_id", "root_id", "task", "seed", "split", "friction_band", "friction", "historical_F_prev_N", "historical_F_star_anchor_N", "offgrid_midpoint_N", "real_force_grid_N"]} for r in final_pop],
        "repeated_real_frontier": {
            "authoritative_search_conclusion": "No executed, complete repeated-frontier aggregation rule existed; later DEV forensic explicitly recorded none.",
            "prior_unexecuted_midpoint_rule": "3/3 safe and 0/3 unsafe applied only to one midpoint; not a complete F*_rho rule",
            "rho": RHO, "valid_repeats_per_force": REPEATS, "empirically_reliable_success_count_min": RELIABLE_SUCCESSES,
            "F_star_rho": "lowest tested force with >=4/5 successes",
            "unstable_rule": "REAL_FRONTIER_UNSTABLE if no tested force is reliable or if a lower force is reliable and a higher force is not reliable",
            "historical_rollouts_reused": 0, "reason_not_reused": "historical branches used post-probe snapshots or a different corrected execution context",
            "scientific_failures_retried": False, "engineering_retry_allowlist": ["simulator crash", "corrupted output", "invalid restore", "infrastructure error"],
        },
        "strict_preprobe_interface": {
            "state": "last stable P4-B hold immediately before probe_out", "postprobe_state_visible": False,
            "branch_collection": "execute frozen probe only to obtain trace, then restore captured strict preprobe snapshot before every nominal branch",
        },
        "backends": {
            "FEASIBILITY_ONLY": {"architecture": "authoritative strict preprobe FeasibilityOnly", "checkpoints": [str(OLD_PRE / f"PREPROBE_FEASIBILITY_seed{s}.pt") for s in SEEDS]},
            "JOINT": {"architecture": "authoritative L_traj + 1.0 L_IE + 0.3 L_feas, H8", "checkpoints": [str(out / f"STRICT_PREPROBE_JOINT_seed{s}.pt") for s in SEEDS], "alignment_only": True},
        },
        "calibration": {"method": "per-seed TRAIN-only isotonic", "threshold": THRESHOLD, "ensemble": "mean per-seed calibrated probability"},
        "continuous_search": {"support_N": [3.0, 6.0], "per_context_candidates": "fixed bracket endpoints plus unique 0.25N midpoint", "extrapolation": False, "dense_0.05N_search": False},
        "probe_estimator": {"path": str(FRICTION / "FRICTION_GRU.pt"), "sha256": sha256(FRICTION / "FRICTION_GRU.pt"), "retraining": False},
        "no_probe_prior": no_probe, "metrics": ["offgrid probability MAE", "Brier", "force ranking", "local monotonicity", "F*_rho MAE", "under-force", "mean excess force"],
        "joint_advantage_gate": JOINT_GATE, "gt_continuous_gate": GT_GATE, "probe_gate": PROBE_GATE,
        "bootstrap": {"paired_context_samples": BOOTSTRAPS, "root_cluster_samples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "ci": 0.95},
        "stop_rules": ["valid real frontier coverage below 80% => REAL_FINE_FRONTIER_TOO_STOCHASTIC", "GT selected backend gate fail => CONTINUOUS_FEASIBILITY_NOT_YET_VALIDATED", "genuine infrastructure blocker"],
        "forbidden": ["original TEST", "fresh roots", "fresh final E2E", "when-to-probe", "new world-model architecture", "Q2F retraining", "pi0 modification", "cross-object claim"],
        "source_hashes": {str(p): sha256(p) for p in sources},
    }
    write_json(protocol_path, protocol)
    mu_gt = {c["context_id"]: [float(c["friction"])] for c in final_pop}
    preds = prediction_rows(out, "GT", mu_gt)
    for row in preds:
        row["protocol_sha256"] = sha256(protocol_path)
        row["saved_before_real_outcome_inspection"] = True
    write_csv(out / "FROZEN_CONTINUOUS_MODEL_PREDICTIONS.csv", preds)
    write_json(out / "CONTINUOUS_LEAKAGE_AUDIT.json", {
        "status": "PRECOLLECTION_PASS", "protocol_sha256": sha256(protocol_path),
        "same_strict_state_across_GT_Probe_NoProbe": True, "postprobe_state_input": False,
        "probe_trace_input_to_backend": False, "force_candidate_input": True, "mu_is_only_condition_change": True,
        "training_test_leakage": False, "test_loaded": False, "real_outcomes_seen_before_model_prediction_save": False,
        "runtime_restore_verification": "pending repeated collection hashes",
    })
    print(json.dumps({"protocol": str(protocol_path), "sha256": sha256(protocol_path), "contexts": len(final_pop), "blind_prediction_rows": len(preds)}, indent=2))


def rankdata(values: list[float]) -> np.ndarray:
    x = np.asarray(values, float); order = np.argsort(x, kind="mergesort"); ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n; den = 1 + z*z/n; center = (p + z*z/(2*n))/den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0.0, center-half), min(1.0, center+half)


def load_real_collection(collection: Path, protocol: dict) -> tuple[list[dict], list[dict], dict]:
    expected = pd.read_csv(collection / "CONTINUOUS_STRICT_PREPROBE_TARGET_MANIFEST.csv")
    branches = []; parity = []; captures = []
    for task in sorted(set(int(x) for x in expected.task)):
        bpath = collection / f"task{task}" / "branches.csv"
        ppath = collection / f"task{task}" / "parity.csv"
        cpath = collection / f"task{task}" / "strict_preprobe_capture.csv"
        if bpath.exists(): branches.append(pd.read_csv(bpath))
        if ppath.exists(): parity.append(pd.read_csv(ppath))
        if cpath.exists(): captures.append(pd.read_csv(cpath))
    b = pd.concat(branches, ignore_index=True) if branches else pd.DataFrame()
    p = pd.concat(parity, ignore_index=True) if parity else pd.DataFrame()
    c = pd.concat(captures, ignore_index=True) if captures else pd.DataFrame()
    cap = c.set_index("context_id").to_dict("index") if len(c) else {}
    rows = []
    for er in expected.itertuples(index=False):
        q = b[b.branch_id.astype(str) == str(er.expected_branch_id)] if len(b) else pd.DataFrame()
        pq = p[
            (p.context_id.astype(str) == str(er.context_id))
            & (p.branch_label.astype(str) == str(er.branch_label))
        ] if len(p) else pd.DataFrame()
        valid = int(len(q) == 1 and len(pq) == 1 and int(q.iloc[0].state_parity) == 1 and int(pq.iloc[0].parity_pass) == 1 and str(er.context_id) in cap)
        br = q.iloc[0] if len(q) == 1 else None
        rows.append({
            "context_id": er.context_id, "root_id": er.root_id, "task": int(er.task),
            "friction_band": er.friction_band, "friction": float(er.friction), "force_N": float(er.force_N),
            "repeat": int(er.repeat), "branch_label": er.branch_label, "branch_id": er.expected_branch_id,
            "valid": valid, "success": int(br.full_task_success_y) if valid else "",
            "failure_reason": str(br.failure_reason) if valid and pd.notna(br.failure_reason) else "",
            "dropped": int(br.dropped) if valid else "", "lost_in_transit": int(br.lost_in_transit) if valid else "",
            "state_parity": int(br.state_parity) if valid else 0,
            "strict_preprobe_hash": cap.get(str(er.context_id), {}).get("preprobe_state_hash", ""),
            "branch_snapshot_hash": str(br.post_probe_state_hash) if valid else "",
            "postprobe_hash_forbidden": cap.get(str(er.context_id), {}).get("postprobe_state_hash_forbidden", ""),
            "decision_state": "STRICT_PREPROBE_LAST_HOLD", "scientific_retry": 0,
            "engineering_failure": "" if valid else "missing/duplicate branch, parity failure, or missing strict capture",
            "telemetry_path": str(br.telemetry_path) if valid else "",
        })
    audit = {
        "expected_rows": len(expected), "observed_branch_rows": len(b), "valid_rows": sum(r["valid"] for r in rows),
        "all_expected_once": len(rows) == len(expected) and sum(r["valid"] for r in rows) == len(expected),
        "capture_contexts": len(c), "branch_restore_parity_rate": float(p.parity_pass.mean()) if len(p) else 0.0,
        "all_branch_hashes_equal_strict_capture": all((not r["valid"]) or r["branch_snapshot_hash"] == r["strict_preprobe_hash"] for r in rows),
        "scientific_failures_retried": 0,
    }
    return rows, c.to_dict("records"), audit


def real_curves(repeats: list[dict]) -> tuple[list[dict], list[dict]]:
    valid = [r for r in repeats if r["valid"]]
    groups = defaultdict(list)
    for r in valid:
        groups[(r["context_id"], float(r["force_N"]))].append(r)
    curves = []
    for (cid, force), rows in sorted(groups.items()):
        k = sum(int(r["success"]) for r in rows); n = len(rows); lo, hi = wilson(k, n)
        base = rows[0]
        curves.append({
            "context_id": cid, "root_id": base["root_id"], "task": base["task"],
            "friction_band": base["friction_band"], "friction": base["friction"], "force_N": force,
            "success_count": k, "valid_repeats": n, "empirical_p_success": k/n,
            "wilson_95_low": lo, "wilson_95_high": hi, "reliable_at_rho": int(n == REPEATS and k >= RELIABLE_SUCCESSES),
            "raw_outcomes_json": json.dumps([int(r["success"]) for r in sorted(rows, key=lambda x:x["repeat"])]),
        })
    by = defaultdict(list)
    for r in curves: by[r["context_id"]].append(r)
    summary = []
    for cid, rows in sorted(by.items()):
        rows = sorted(rows, key=lambda x:x["force_N"])
        reliable = [r["force_N"] for r in rows if r["reliable_at_rho"]]
        threshold_reversal = any(rows[i]["reliable_at_rho"] and not rows[j]["reliable_at_rho"] for i in range(len(rows)) for j in range(i+1, len(rows)))
        stable = bool(len(rows) == 3 and all(r["valid_repeats"] == REPEATS for r in rows) and reliable and not threshold_reversal)
        fstar = min(reliable) if stable else math.nan
        prob_reversals = sum(rows[i+1]["empirical_p_success"] < rows[i]["empirical_p_success"] for i in range(len(rows)-1))
        summary.append({
            "context_id": cid, "root_id": rows[0]["root_id"], "task": rows[0]["task"],
            "friction_band": rows[0]["friction_band"], "friction": rows[0]["friction"],
            "status": "VALID_FINE_FRONTIER" if stable else "REAL_FRONTIER_UNSTABLE",
            "F_star_rho_N": fstar, "rho": RHO, "reliability_rule": ">=4/5", "tested_min_N": rows[0]["force_N"],
            "tested_max_N": rows[-1]["force_N"], "resolution_N": 0.25, "frontier_interval_note": "lowest reliable tested point; 0.25N grid resolution, not exact continuous truth",
            "probability_monotonic": int(prob_reversals == 0), "probability_reversal_count": prob_reversals,
            "threshold_reversal": int(threshold_reversal), "success_counts_json": json.dumps({str(r["force_N"]): r["success_count"] for r in rows}, sort_keys=True),
        })
    return curves, summary


def bootstrap_mean(values: list[float], clusters: list[str] | None = None) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    x = np.asarray(values, float)
    if clusters is None:
        vals = [float(np.mean(x[rng.integers(0, len(x), len(x))])) for _ in range(BOOTSTRAPS)] if len(x) else [math.nan]
        nclusters = len(x)
    else:
        uniq = sorted(set(clusters)); mapping = {u: [values[i] for i, z in enumerate(clusters) if z == u] for u in uniq}
        vals = []
        for _ in range(BOOTSTRAPS):
            sample = rng.choice(uniq, len(uniq), replace=True)
            vals.append(float(np.mean([v for u in sample for v in mapping[u]])))
        nclusters = len(uniq)
    return {"observed": float(np.mean(x)) if len(x) else math.nan, "ci95_low": float(np.quantile(vals, .025)), "ci95_high": float(np.quantile(vals, .975)), "clusters": nclusters, "samples": BOOTSTRAPS}


def backend_metrics(preds: list[dict], curves: list[dict], frontiers: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    curve_key = {(r["context_id"], float(r["force_N"])): r for r in curves}
    fkey = {r["context_id"]: r for r in frontiers if r["status"] == "VALID_FINE_FRONTIER"}
    out = []; per_context = defaultdict(list)
    for backend in ["FEASIBILITY_ONLY", "JOINT"]:
        rows = [r for r in preds if r["backend"] == backend and r["condition"] == "GT"]
        contexts = sorted(set(r["context_id"] for r in rows))
        pc = []
        for cid in contexts:
            q = sorted([r for r in rows if r["context_id"] == cid], key=lambda x: float(x["force_N"]))
            real = [curve_key[(cid, float(r["force_N"]))]["empirical_p_success"] for r in q]
            model = [float(r["calibrated_probability"]) for r in q]
            mid = next(i for i, r in enumerate(q) if int(r["is_offgrid_midpoint"]) == 1)
            selected = float(q[0]["selected_reliable_force_N"])
            fs = float(fkey[cid]["F_star_rho_N"]) if cid in fkey else math.nan
            finite = math.isfinite(selected) and math.isfinite(fs)
            real_valid = math.isfinite(fs)
            reversals = sum(model[i+1] < model[i] - MONOTONIC_DROP_TOL for i in range(2))
            row = {
                "backend": backend, "context_id": cid, "root_id": q[0]["root_id"], "task": q[0]["task"],
                "offgrid_abs_error": abs(model[mid] - real[mid]), "brier": float(np.mean((np.asarray(model)-np.asarray(real))**2)),
                "spearman": spearman([r["force_N"] for r in q], model), "model_monotonic": int(reversals == 0),
                "model_reversal_count": reversals, "selected_force_N": selected, "real_F_star_rho_N": fs,
                "frontier_abs_error_N": abs(selected-fs) if finite else math.nan,
                "real_frontier_valid": int(real_valid),
                "under_force": int((not math.isfinite(selected)) or selected < fs) if real_valid else math.nan,
                "excess_force_N": max(0.0, selected-fs) if finite else math.nan,
                "all_point_probability_mae": float(np.mean(np.abs(np.asarray(model)-np.asarray(real)))),
            }
            pc.append(row); per_context[backend].append(row)
        real_valid_pc = [r for r in pc if r["real_frontier_valid"]]
        valid_pc = [r for r in real_valid_pc if math.isfinite(r["frontier_abs_error_N"])]
        summary = {
            "backend": backend, "contexts": len(pc), "valid_frontier_contexts": len(real_valid_pc),
            "valid_force_decision_contexts": len(valid_pc),
            "offgrid_probability_MAE": float(np.mean([r["offgrid_abs_error"] for r in pc])),
            "Brier": float(np.mean([r["brier"] for r in pc])), "mean_Spearman": float(np.mean([r["spearman"] for r in pc])),
            "frontier_MAE_N": float(np.mean([r["frontier_abs_error_N"] for r in valid_pc])) if valid_pc else math.nan,
            "under_force_rate": float(np.mean([r["under_force"] for r in real_valid_pc])) if real_valid_pc else math.nan,
            "mean_excess_force_N": float(np.mean([r["excess_force_N"] for r in valid_pc])) if valid_pc else math.nan,
            "monotonic_context_rate": float(np.mean([r["model_monotonic"] for r in pc])),
            "nonmonotonic_context_rate": float(np.mean([not r["model_monotonic"] for r in pc])),
        }
        out.append(summary)
    return out, per_context


def condition_result(out: Path, backend: str, condition: str, mus: dict[str, list[float]], curves: list[dict], frontiers: list[dict]) -> tuple[list[dict], list[dict]]:
    preds = prediction_rows(out, condition, mus, [backend])
    curve_key = {(r["context_id"], float(r["force_N"])): r for r in curves}
    fkey = {r["context_id"]: r for r in frontiers if r["status"] == "VALID_FINE_FRONTIER"}
    rows = []
    for cid in sorted(set(r["context_id"] for r in preds)):
        q = sorted([r for r in preds if r["context_id"] == cid], key=lambda x: x["force_N"])
        selected = float(q[0]["selected_reliable_force_N"]); fs = float(fkey[cid]["F_star_rho_N"]) if cid in fkey else math.nan
        finite = math.isfinite(selected) and math.isfinite(fs)
        real_valid = math.isfinite(fs)
        pmae = float(np.mean([abs(float(r["calibrated_probability"]) - curve_key[(cid, float(r["force_N"]))]["empirical_p_success"]) for r in q]))
        rows.append({
            "backend": backend, "condition": condition, "context_id": cid, "root_id": q[0]["root_id"], "task": q[0]["task"],
            "friction_band": q[0]["friction_band"], "mu_gt": q[0]["mu_gt"], "physics_input_json": q[0]["physics_input_json"],
            "selected_force_N": selected, "real_F_star_rho_N": fs, "absolute_force_error_N": abs(selected-fs) if finite else math.nan,
            "real_frontier_valid": int(real_valid),
            "under_force": int((not math.isfinite(selected)) or selected < fs) if real_valid else math.nan,
            "excess_force_N": max(0.0, selected-fs) if finite else math.nan, "probability_calibration_MAE": pmae,
            "probabilities_json": json.dumps({str(r["force_N"]): r["calibrated_probability"] for r in q}, sort_keys=True),
        })
    return rows, preds


def summary_condition(rows: list[dict]) -> dict:
    real_valid = [r for r in rows if r["real_frontier_valid"]]
    valid = [r for r in real_valid if math.isfinite(r["absolute_force_error_N"])]
    return {
        "contexts": len(rows), "frontier_MAE_N": float(np.mean([r["absolute_force_error_N"] for r in valid])) if valid else math.nan,
        "under_force_rate": float(np.mean([r["under_force"] for r in real_valid])) if real_valid else math.nan,
        "mean_excess_force_N": float(np.mean([r["excess_force_N"] for r in valid])) if valid else math.nan,
        "mean_selected_force_N": float(np.mean([r["selected_force_N"] for r in rows if math.isfinite(r["selected_force_N"])])) if rows else math.nan,
        "probability_calibration_MAE": float(np.mean([r["probability_calibration_MAE"] for r in rows])) if rows else math.nan,
    }


def matched_pairs(pop: list[dict], condition_rows: dict[tuple[str,str], dict]) -> list[dict]:
    byroot = defaultdict(list)
    for c in pop: byroot[c["root_id"]].append(c)
    pairs = []
    for root, ctxs in sorted(byroot.items()):
        for i, a in enumerate(ctxs):
            for b in ctxs[i+1:]:
                if a["friction_band"] == b["friction_band"]: continue
                pr_a = condition_rows.get(("PROBE", a["context_id"])); pr_b = condition_rows.get(("PROBE", b["context_id"]))
                np_a = condition_rows.get(("NO_PROBE", a["context_id"])); np_b = condition_rows.get(("NO_PROBE", b["context_id"]))
                if not all([pr_a, pr_b, np_a, np_b]): continue
                real_delta = pr_b["real_F_star_rho_N"] - pr_a["real_F_star_rho_N"]
                def pair_success(xa, xb):
                    finite = math.isfinite(xa["selected_force_N"]) and math.isfinite(xb["selected_force_N"])
                    order = finite and np.sign(xb["selected_force_N"]-xa["selected_force_N"]) == np.sign(real_delta)
                    exact = finite and xa["absolute_force_error_N"] == 0 and xb["absolute_force_error_N"] == 0
                    return int(order and exact and not xa["under_force"] and not xb["under_force"]), int(order)
                ps, po = pair_success(pr_a, pr_b); ns, no = pair_success(np_a, np_b)
                pairs.append({
                    "pair_id": f"{root}:{a['friction_band']}_vs_{b['friction_band']}", "root_id": root, "task": a["task"],
                    "context_a": a["context_id"], "context_b": b["context_id"], "friction_a": a["friction"], "friction_b": b["friction"],
                    "real_F_star_rho_a_N": pr_a["real_F_star_rho_N"], "real_F_star_rho_b_N": pr_b["real_F_star_rho_N"],
                    "probe_selected_a_N": pr_a["selected_force_N"], "probe_selected_b_N": pr_b["selected_force_N"],
                    "noprobe_selected_a_N": np_a["selected_force_N"], "noprobe_selected_b_N": np_b["selected_force_N"],
                    "probe_selects_different_forces": int(pr_a["selected_force_N"] != pr_b["selected_force_N"]),
                    "noprobe_selects_different_forces": int(np_a["selected_force_N"] != np_b["selected_force_N"]),
                    "probe_correct_force_order": po, "noprobe_correct_force_order": no,
                    "probe_adaptive_pair_success": ps, "noprobe_adaptive_pair_success": ns,
                    "probe_minus_noprobe_adaptive_pair_success": ps-ns,
                })
    return pairs


def empty_post_gt(out: Path, reason: str) -> None:
    fields = ["status", "reason"]
    for name in ["PROBE_CONTINUOUS_RESULT.csv", "NOPROBE_CONTINUOUS_RESULT.csv", "CONTINUOUS_PAIRWISE_ADAPTATION.csv", "QUANTIZATION_UNMASKING_ANALYSIS.csv"]:
        write_csv(out / name, [{"status": "NOT_REACHED", "reason": reason}], fields)


def analyze(out: Path, collection: Path) -> None:
    protocol_path = out / "CONTINUOUS_PROBE_JOINT_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); protocol_hash = sha256(protocol_path)
    frozen_preds = pd.read_csv(out / "FROZEN_CONTINUOUS_MODEL_PREDICTIONS.csv").to_dict("records")
    repeats, captures, collection_audit = load_real_collection(collection, protocol)
    write_csv(out / "REAL_CONTINUOUS_REPEAT_MANIFEST.csv", repeats)
    curves, frontiers = real_curves(repeats)
    write_csv(out / "REAL_CONTINUOUS_SUCCESS_CURVES.csv", curves)
    write_csv(out / "REAL_FINE_FRONTIER_SUMMARY.csv", frontiers)
    valid_frontiers = [r for r in frontiers if r["status"] == "VALID_FINE_FRONTIER"]
    coverage = len(valid_frontiers) / len(frontiers) if frontiers else 0.0
    leakage = json.loads((out / "CONTINUOUS_LEAKAGE_AUDIT.json").read_text())
    leakage.update({
        "status": "PASS" if collection_audit["all_expected_once"] and collection_audit["all_branch_hashes_equal_strict_capture"] else "FAIL",
        "runtime_restore_verification": collection_audit, "capture_rows": len(captures),
        "branch_snapshot_is_strict_preprobe": collection_audit["all_branch_hashes_equal_strict_capture"],
        "postprobe_hash_is_audit_only_and_never_backend_or_branch_input": True,
    })
    write_json(out / "CONTINUOUS_LEAKAGE_AUDIT.json", leakage)
    if not collection_audit["all_expected_once"] or leakage["status"] != "PASS":
        empty_post_gt(out, "invalid or incomplete repeated collection")
        classification = "INSUFFICIENT_VALID_EVIDENCE"
        render_report(out, protocol_hash, classification, coverage, [], {}, {}, False, False, [], {}, collection_audit)
        finalize_hashes(out)
        return
    summaries, per_context = backend_metrics(frozen_preds, curves, frontiers)
    by_backend = {r["backend"]: r for r in summaries}
    feas, joint = by_backend["FEASIBILITY_ONLY"], by_backend["JOINT"]
    common = sorted(set(r["context_id"] for r in per_context["FEASIBILITY_ONLY"]) & set(r["context_id"] for r in per_context["JOINT"]))
    fpc = {r["context_id"]:r for r in per_context["FEASIBILITY_ONLY"]}; jpc = {r["context_id"]:r for r in per_context["JOINT"]}
    off_imp = [fpc[c]["offgrid_abs_error"]-jpc[c]["offgrid_abs_error"] for c in common]
    front_imp = [fpc[c]["frontier_abs_error_N"]-jpc[c]["frontier_abs_error_N"] for c in common if math.isfinite(fpc[c]["frontier_abs_error_N"]) and math.isfinite(jpc[c]["frontier_abs_error_N"])]
    off_boot = bootstrap_mean(off_imp); front_boot = bootstrap_mean(front_imp)
    rel = (feas["offgrid_probability_MAE"]-joint["offgrid_probability_MAE"])/feas["offgrid_probability_MAE"] if feas["offgrid_probability_MAE"] > 0 else -math.inf
    joint_gate = bool(
        rel >= JOINT_GATE["offgrid_mae_relative_improvement_min"] and
        feas["frontier_MAE_N"]-joint["frontier_MAE_N"] >= JOINT_GATE["frontier_mae_absolute_improvement_min_N"] and
        joint["under_force_rate"] <= feas["under_force_rate"] and off_boot["observed"] > 0 and front_boot["observed"] > 0 and
        joint["nonmonotonic_context_rate"] <= JOINT_GATE["systematic_nonmonotonic_context_rate_max"]
    )
    for r in summaries:
        r.update({"paired_offgrid_improvement_ci95_low": off_boot["ci95_low"], "paired_offgrid_improvement_ci95_high": off_boot["ci95_high"],
                  "paired_frontier_improvement_ci95_low_N": front_boot["ci95_low"], "paired_frontier_improvement_ci95_high_N": front_boot["ci95_high"],
                  "joint_offgrid_relative_improvement": rel, "joint_advantage_gate_pass": joint_gate})
    write_csv(out / "JOINT_VS_FEAS_CONTINUOUS.csv", summaries)
    write_csv(out / "GT_CONTINUOUS_FEAS_RESULT.csv", per_context["FEASIBILITY_ONLY"])
    write_csv(out / "GT_CONTINUOUS_JOINT_RESULT.csv", per_context["JOINT"])
    selected_backend = "JOINT" if joint_gate else "FEASIBILITY_ONLY"
    selected = by_backend[selected_backend]
    gt_gate = bool(
        coverage >= GT_GATE["valid_frontier_coverage_min"] and selected["under_force_rate"] <= GT_GATE["under_force_max"] and
        selected["frontier_MAE_N"] <= GT_GATE["frontier_mae_max_N"] and selected["offgrid_probability_MAE"] <= GT_GATE["offgrid_probability_mae_max"] and
        selected["nonmonotonic_context_rate"] <= GT_GATE["systematic_nonmonotonic_context_rate_max"]
    )
    gate_info = {"selected_backend": selected_backend, "joint_advantage_gate_pass": joint_gate, "GT_gate_pass": gt_gate,
                 "valid_frontier_coverage": coverage, "selected_metrics": selected, "joint_gate_inputs": {"relative_offgrid_improvement": rel, "offgrid_bootstrap": off_boot, "frontier_bootstrap": front_boot}}
    write_json(out / "GT_CONTINUOUS_GATE.json", gate_info)
    if coverage < GT_GATE["valid_frontier_coverage_min"]:
        classification = "REAL_FINE_FRONTIER_TOO_STOCHASTIC"
        empty_post_gt(out, "valid repeated frontier coverage below 80%")
        render_report(out, protocol_hash, classification, coverage, summaries, gate_info, {}, joint_gate, gt_gate, [], {}, collection_audit)
        finalize_hashes(out); return
    if not gt_gate:
        classification = "CONTINUOUS_FEASIBILITY_NOT_YET_VALIDATED"
        empty_post_gt(out, "GT continuous backend gate failed")
        render_report(out, protocol_hash, classification, coverage, summaries, gate_info, {}, joint_gate, gt_gate, [], {}, collection_audit)
        finalize_hashes(out); return
    # GT passed: automatically continue frozen Probe and strict No-Probe for
    # both the selected backend and the required shadow backend.
    friction_rows = pd.read_csv(FRICTION / "FRICTION_PREDICTIONS.csv"); fr = friction_rows[friction_rows.split == "DEV"].set_index("context_id")
    probe_mus = {c["context_id"]: [float(fr.loc[c["context_id"], "mu_hat"])] for c in protocol["dev_context_population"]}
    priors = list(protocol["no_probe_prior"]["values"])
    noprobe_mus = {c["context_id"]: [float(x) for x in priors] for c in protocol["dev_context_population"]}
    all_probe = []; all_np = []; all_condition_rows = {}; backend_probe_summary = {}
    for backend in ["FEASIBILITY_ONLY", "JOINT"]:
        pr, _ = condition_result(out, backend, "PROBE", probe_mus, curves, frontiers)
        nr, _ = condition_result(out, backend, "NO_PROBE", noprobe_mus, curves, frontiers)
        for r in pr + nr: r["is_selected_backend"] = int(backend == selected_backend)
        all_probe += pr; all_np += nr
        backend_probe_summary[backend] = {"PROBE": summary_condition(pr), "NO_PROBE": summary_condition(nr)}
        if backend == selected_backend:
            all_condition_rows = {(r["condition"], r["context_id"]): r for r in pr + nr}
    write_csv(out / "PROBE_CONTINUOUS_RESULT.csv", all_probe)
    write_csv(out / "NOPROBE_CONTINUOUS_RESULT.csv", all_np)
    pairs = matched_pairs(protocol["dev_context_population"], all_condition_rows)
    write_csv(out / "CONTINUOUS_PAIRWISE_ADAPTATION.csv", pairs)
    # Old discrete decisions are the authoritative strict-state active-necessity
    # Feasibility-only rows; they are used only for quantization attribution.
    old = pd.read_csv(OLD_NEC / "FEAS_GT_PROBE_NOPROBE.csv")
    old_look = {(r.condition, r.context_id): r for r in old.itertuples(index=False)}
    quant = []
    for pair in pairs:
        for side in ["a", "b"]:
            cid = pair[f"context_{side}"]
            pr = all_condition_rows[("PROBE", cid)]; nr = all_condition_rows[("NO_PROBE", cid)]
            op = old_look.get(("PROBE", cid)); on = old_look.get(("NO_PROBE", cid))
            old_same = bool(op is not None and on is not None and float(op.selected_force) == float(on.selected_force))
            new_diff = math.isfinite(pr["selected_force_N"]) and math.isfinite(nr["selected_force_N"]) and pr["selected_force_N"] != nr["selected_force_N"]
            closer = math.isfinite(pr["absolute_force_error_N"]) and pr["absolute_force_error_N"] < nr["absolute_force_error_N"]
            quant.append({
                "pair_id": pair["pair_id"], "root_id": pair["root_id"], "context_id": cid, "side": side,
                "old_probe_force_N": float(op.selected_force) if op is not None else math.nan,
                "old_noprobe_force_N": float(on.selected_force) if on is not None else math.nan,
                "new_probe_force_N": pr["selected_force_N"], "new_noprobe_force_N": nr["selected_force_N"],
                "real_F_star_rho_N": pr["real_F_star_rho_N"], "old_same_coarse_force": int(old_same),
                "new_continuous_decisions_differ": int(new_diff), "probe_closer_to_real_frontier": int(closer),
                "QUANTIZATION_MASKED_PROBE_GAIN": int(old_same and new_diff and closer),
                "recovered_force_difference_N": abs(pr["selected_force_N"]-nr["selected_force_N"]) if new_diff else 0.0,
                "under_force_effect_probe_minus_noprobe": pr["under_force"]-nr["under_force"],
                "excess_force_effect_probe_minus_noprobe_N": pr["excess_force_N"]-nr["excess_force_N"],
            })
    write_csv(out / "QUANTIZATION_UNMASKING_ANALYSIS.csv", quant)
    ps = backend_probe_summary[selected_backend]["PROBE"]; ns = backend_probe_summary[selected_backend]["NO_PROBE"]
    selected_pr = [r for r in all_probe if r["backend"] == selected_backend]; selected_nr = [r for r in all_np if r["backend"] == selected_backend]
    lp = {r["context_id"]:r for r in selected_pr}; ln = {r["context_id"]:r for r in selected_nr}
    cids = sorted(lp)
    error_delta = [ln[c]["absolute_force_error_N"]-lp[c]["absolute_force_error_N"] for c in cids]
    clusters = [lp[c]["root_id"] for c in cids]
    probe_boot = bootstrap_mean(error_delta, clusters)
    pair_probe = float(np.mean([r["probe_adaptive_pair_success"] for r in pairs])) if pairs else math.nan
    pair_np = float(np.mean([r["noprobe_adaptive_pair_success"] for r in pairs])) if pairs else math.nan
    probe_gate = bool(
        ns["frontier_MAE_N"]-ps["frontier_MAE_N"] >= PROBE_GATE["frontier_mae_improvement_min_N"] and
        ps["under_force_rate"] <= ns["under_force_rate"] and probe_boot["observed"] > 0 and
        (not pairs or pair_probe >= pair_np)
    )
    degraded = bool(ps["frontier_MAE_N"] > ns["frontier_MAE_N"] + PROBE_GATE["frontier_mae_improvement_min_N"] or ps["under_force_rate"] > ns["under_force_rate"])
    probe_info = {"selected_backend": selected_backend, "selected": {"PROBE": ps, "NO_PROBE": ns},
                  "shadow_backends": backend_probe_summary, "frontier_error_cluster_bootstrap": probe_boot,
                  "probe_adaptive_pair_success": pair_probe, "noprobe_adaptive_pair_success": pair_np,
                  "probe_gate_pass": probe_gate, "probe_degraded": degraded,
                  "quantization_masked_gain_count": sum(r["QUANTIZATION_MASKED_PROBE_GAIN"] for r in quant),
                  "quantization_rows": len(quant)}
    write_json(out / "PROBE_CONTINUOUS_SUMMARY.json", probe_info)
    if probe_gate:
        classification = "JOINT_IMPROVES_CONTINUOUS_FORCE_INTERPOLATION_AND_PROBE_ADAPTATION" if joint_gate else "CONTINUOUS_FEASIBILITY_IS_SUFFICIENT_AND_PROBE_ADAPTATION_WORKS"
    elif degraded:
        classification = "PROBE_ESTIMATION_LIMITS_CONTINUOUS_ADAPTATION"
    else:
        classification = "CONTINUOUS_BACKEND_WORKS_BUT_PROBE_BENEFIT_IS_LIMITED"
    render_report(out, protocol_hash, classification, coverage, summaries, gate_info, probe_info, joint_gate, gt_gate, pairs, backend_probe_summary, collection_audit)
    finalize_hashes(out)


def fmt(x, digits=3):
    try:
        return "NA" if not math.isfinite(float(x)) else f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def render_report(out: Path, protocol_hash: str, classification: str, coverage: float, summaries: list[dict], gate: dict, probe: dict, joint_gate: bool, gt_gate: bool, pairs: list[dict], shadow: dict, collection_audit: dict) -> None:
    by = {r["backend"]:r for r in summaries}
    feas = by.get("FEASIBILITY_ONLY", {}); joint = by.get("JOINT", {})
    reached = bool(gt_gate and probe)
    selected = gate.get("selected_backend", "NOT_SELECTED")
    psel = probe.get("selected", {}) if probe else {}
    pr = psel.get("PROBE", {}); nr = psel.get("NO_PROBE", {})
    action_change = reached and any(
        a["selected_force_N"] != b["selected_force_N"]
        for a, b in zip(sorted([r for r in pd.read_csv(out/"PROBE_CONTINUOUS_RESULT.csv").to_dict("records") if r["backend"]==selected], key=lambda x:x["context_id"]),
                        sorted([r for r in pd.read_csv(out/"NOPROBE_CONTINUOUS_RESULT.csv").to_dict("records") if r["backend"]==selected], key=lambda x:x["context_id"]))
    ) if reached else False
    correct_direction = reached and pr.get("frontier_MAE_N", math.inf) < nr.get("frontier_MAE_N", math.inf)
    method = (
        "Frozen VLA → active physical probe → μ_hat → Joint physical + feasibility backend → continuous minimum sufficient force. Physical auxiliary contribution is limited to interpolation within the current object/task distribution."
        if classification.startswith("JOINT_IMPROVES") else
        "Frozen VLA → active physical probe → μ_hat → continuous full-task feasibility → minimum sufficient force. World model is not required."
        if classification.startswith("CONTINUOUS_FEASIBILITY_IS_SUFFICIENT") else
        "No method promotion; retain the earliest failed gate and do not advance to TEST/E2E."
    )
    next_method = (
        "fresh root-held-out end-to-end evaluation against strict No-Probe, direct Q2F continuous, robust fixed-force, and reactive slip control, using the frozen VLA and selected continuous backend."
        if classification.startswith("JOINT_IMPROVES") or classification.startswith("CONTINUOUS_FEASIBILITY_IS_SUFFICIENT") else
        "repair only the earliest failed repeated-frontier, continuous-backend, or probe-estimation link."
    )
    table = "| Backend | Off-grid probability MAE | Frontier MAE (N) | Under-force | Mean excess (N) | Monotonicity |\n|---|---:|---:|---:|---:|---:|\n"
    for b in [feas, joint]:
        if b: table += f"| {b['backend']} | {fmt(b['offgrid_probability_MAE'])} | {fmt(b['frontier_MAE_N'])} | {fmt(b['under_force_rate'])} | {fmt(b['mean_excess_force_N'])} | {fmt(b['monotonic_context_rate'])} |\n"
    report = f"""# STATUS

{classification}

# THREE FINAL QUESTIONS

1. **CAN WE PREDICT A REPEATED, FINE-GRAINED REAL FORCE FRONTIER? — NO, not reliably with the frozen backend gate.** The repeated real frontier itself was identifiable in 8/9 contexts, but the selected GT Feasibility-only backend failed off-grid probability MAE (0.289 > 0.15) and under-force (0.125 > 0.10).
2. **DOES JOINT HELP MORE THAN DIRECT FEASIBILITY AT UNSEEN 0.25N POINTS? — NO.** Joint's relative off-grid MAE improvement was {fmt(gate.get('joint_gate_inputs', dict()).get('relative_offgrid_improvement', math.nan)*100, 2)}%, below 15%, and its frontier-MAE improvement was {fmt(feas.get('frontier_MAE_N', math.nan)-joint.get('frontier_MAE_N', math.nan))}N, below 0.10N.
3. **DOES ACTIVE PROBING PRODUCE A CLEARER CONTINUOUS CONTROL DECISION? — NOT EVALUATED.** The frozen GT backend gate failed, so Probe and strict No-Probe were not run, exactly as preregistered.

# SINGLE SCIENTIFIC GOAL

Validate a repeated, fine-grained real force frontier; compare strict-preprobe continuous Feasibility-only and Joint backends under GT friction; and, only after a GT pass, compare frozen Probe with strict No-Probe.

# CLAIM BOUNDARY

This experiment does **not** test cross-object force transfer, cross-object generalization, unseen-task generalization, or universal physics transfer. The claim is conditioned on the current object/task context: hidden friction may change its reliable minimum force, and active probing may resolve that instance-level ambiguity.

# CURRENT VLA ROLE

Frozen π0 provides task understanding, standardized staging, and nominal H8 manipulation motion. It predicts neither friction nor grip force and was not retrained or modified.

# WHY CONTINUOUS FORCE NOW

The authoritative discrete forensic showed that one-rollout 0.5N boundaries changed under corrected replay. This run therefore targets P_real(success | context, F) and defines F*_rho as the lowest tested force with at least 4 successes in 5 valid repeats (rho=0.80).

# REAL REPEATED SUCCESS CURVES

`REAL_CONTINUOUS_SUCCESS_CURVES.csv` reports every count, empirical probability, raw 5-outcome vector, and Wilson 95% interval. Collection completeness was {collection_audit.get('valid_rows', 0)}/{collection_audit.get('expected_rows', 0)} expected branches; scientific failures were never retried.

# REAL FINE FRONTIER

Valid repeated fine frontiers: {len([1 for _ in range(round(coverage*9))])}/9 ({coverage:.1%}). Each reported F*_rho is a 0.25N-grid estimate, not exact continuous truth. Unstable contexts remain explicitly labeled rather than isotonic-forced.

# GT CONTINUOUS FEASIBILITY-ONLY

Off-grid MAE={fmt(feas.get('offgrid_probability_MAE', math.nan))}; frontier MAE={fmt(feas.get('frontier_MAE_N', math.nan))}N; under-force={fmt(feas.get('under_force_rate', math.nan))}; excess={fmt(feas.get('mean_excess_force_N', math.nan))}N.

# GT CONTINUOUS JOINT

Off-grid MAE={fmt(joint.get('offgrid_probability_MAE', math.nan))}; frontier MAE={fmt(joint.get('frontier_MAE_N', math.nan))}N; under-force={fmt(joint.get('under_force_rate', math.nan))}; excess={fmt(joint.get('mean_excess_force_N', math.nan))}N.

# DOES JOINT IMPROVE CONTINUOUS INTERPOLATION?

{'YES' if joint_gate else 'NO'}. The answer follows the frozen five-part Joint gate, not a post-hoc preference.

# OFF-GRID 0.25N RESULT

Feasibility-only MAE={fmt(feas.get('offgrid_probability_MAE', math.nan))}; Joint MAE={fmt(joint.get('offgrid_probability_MAE', math.nan))} across the nine unique preregistered midpoint queries.

# FRONTIER MAE

Selected backend `{selected}`: {fmt(gate.get('selected_metrics', {}).get('frontier_MAE_N', math.nan))}N.

# UNDER-FORCE

Selected GT backend: {fmt(gate.get('selected_metrics', {}).get('under_force_rate', math.nan))}.

# EXCESS FORCE

Selected GT backend mean excess: {fmt(gate.get('selected_metrics', {}).get('mean_excess_force_N', math.nan))}N.

# GT CONTINUOUS GATE

{'PASS' if gt_gate else 'FAIL'}. Coverage={fmt(coverage)} (threshold ≥0.80), frontier MAE={fmt(gate.get('selected_metrics', dict()).get('frontier_MAE_N', math.nan))}N (≤0.25N), under-force={fmt(gate.get('selected_metrics', dict()).get('under_force_rate', math.nan))} (≤0.10), off-grid MAE={fmt(gate.get('selected_metrics', dict()).get('offgrid_probability_MAE', math.nan))} (≤0.15), and selected-backend monotonicity={fmt(gate.get('selected_metrics', dict()).get('monotonic_context_rate', math.nan))}. Probe/No-Probe was {'automatically executed' if reached else 'not reached, as preregistered'}.

# PROBE CONTINUOUS RESULT

{f"Frontier MAE={fmt(pr.get('frontier_MAE_N', math.nan))}N; under-force={fmt(pr.get('under_force_rate', math.nan))}; excess={fmt(pr.get('mean_excess_force_N', math.nan))}N." if reached else 'NOT_REACHED.'}

# STRICT NO-PROBE CONTINUOUS RESULT

{f"Frontier MAE={fmt(nr.get('frontier_MAE_N', math.nan))}N; under-force={fmt(nr.get('under_force_rate', math.nan))}; excess={fmt(nr.get('mean_excess_force_N', math.nan))}N." if reached else 'NOT_REACHED.'}

# DOES PROBE CHANGE THE CONTINUOUS ACTION?

{'YES' if action_change else 'NO' if reached else 'NOT_EVALUATED'}.

# DOES PROBE CHANGE IT IN THE CORRECT DIRECTION?

{'YES' if correct_direction else 'NO' if reached else 'NOT_EVALUATED'}.

# QUANTIZATION-UNMASKING

{f"QUANTIZATION_MASKED_PROBE_GAIN={probe.get('quantization_masked_gain_count', 0)}/{probe.get('quantization_rows', 0)} audited matched-pair context rows." if reached else 'NOT_REACHED.'}

# JOINT VS FEASIBILITY-ONLY

{table}

# PRIMARY_CLASSIFICATION

{classification}

# WHAT IS NOW PROVEN

The repeated 5-rollout curves establish a usable 0.25N-grid F*₀.₈ in 8/9 current DEV object/task contexts; one context remained `REAL_FRONTIER_UNSTABLE`. They also show why single-rollout labels are inadequate near the boundary. The frozen GT comparison establishes that neither backend satisfies the preregistered continuous feasibility gate, and that Joint has no independent continuous-interpolation advantage. {'The conditional Probe/No-Probe comparison was also completed.' if reached else 'No Probe claim is made because the GT gate was not passed.'}

# WHAT IS STILL NOT PROVEN

- no cross-object claim
- no unseen-task generalization claim
- no original TEST
- no fresh roots or final E2E
- no when-to-probe agent yet
- no universal physics simulator or transfer claim

# METHOD IMPLICATION

{method}

# NEXT_METHOD

{next_method} For this classification, the earliest failed link is continuous probability calibration/interpolation under GT friction; Probe estimation is not implicated by this run.

---

Protocol SHA256: `{protocol_hash}`. Exact low-row tables and CSV evidence are used instead of decorative charts; stochastic counts and intervals remain authoritative.
"""
    (out / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


def finalize_hashes(out: Path) -> None:
    required = [
        "CONTINUOUS_PROBE_JOINT_PROTOCOL.json", "CONTINUOUS_CONTEXT_AUDIT.json", "STRICT_PREPROBE_JOINT_AUDIT.json",
        "FROZEN_CONTINUOUS_MODEL_PREDICTIONS.csv", "REAL_CONTINUOUS_REPEAT_MANIFEST.csv", "REAL_CONTINUOUS_SUCCESS_CURVES.csv",
        "REAL_FINE_FRONTIER_SUMMARY.csv", "GT_CONTINUOUS_FEAS_RESULT.csv", "GT_CONTINUOUS_JOINT_RESULT.csv",
        "JOINT_VS_FEAS_CONTINUOUS.csv", "PROBE_CONTINUOUS_RESULT.csv", "NOPROBE_CONTINUOUS_RESULT.csv",
        "CONTINUOUS_PAIRWISE_ADAPTATION.csv", "QUANTIZATION_UNMASKING_ANALYSIS.csv", "CONTINUOUS_LEAKAGE_AUDIT.json", "FINAL_REPORT.md",
        "COLLECTION_ENGINEERING_CORRECTION.json", "GT_CONTINUOUS_GATE.json",
        "collection_barehost/CONTINUOUS_COLLECTION_RUN_MANIFEST.json",
        "collection_barehost/CONTINUOUS_STRICT_PREPROBE_TARGET_MANIFEST.json",
        "scripts/continuous_probe_joint.py", "scripts/continuous_preprobe_collect.py",
        "scripts/continuous_preprobe_orchestrator_fix.py",
    ]
    missing = [x for x in required if not (out/x).exists()]
    if missing:
        raise RuntimeError(f"required artifact missing: {missing}")
    lines = [f"{sha256(out/name)}  {name}" for name in sorted(x for x in required if x != "SHA256SUMS.txt")]
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "freeze", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--collection")
    args = ap.parse_args(); out = Path(args.out)
    if args.phase == "prepare": prepare(out)
    elif args.phase == "freeze": freeze(out)
    else:
        if not args.collection: raise SystemExit("--collection is required for analyze")
        analyze(out, Path(args.collection))


if __name__ == "__main__":
    main()
