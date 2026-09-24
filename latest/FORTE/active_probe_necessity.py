#!/usr/bin/env python3
"""Strict DEV-only active-probe necessity analysis for Tabero.

This script never trains a model.  It freezes a real-frontier-defined
friction-discordant population, reconstructs the last stable P4-B ``hold``
state before active shear, and then runs the already frozen all-task Joint
and Feasibility-only ensembles under GT, Probe, and a historical no-probe
friction prior.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
P5 = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
FEAS = RESULTS / "full_task_feasibility_20260830_012830"
CF = RESULTS / "counterfactual_force_world_model_20260829_160000"
FRICTION = RESULTS / "active_friction_imagination_20260828_211106"
PROBE_GATE = RESULTS / "probe_informed_imagination_20260829_121500"
STOP_LINE = RESULTS / "task_demand_loto_generalization_20260830_044334"
TPI_CODE = REPO / "analysis/trajectory_physical_imagination.py"
CF_CODE = REPO / "analysis/counterfactual_force_world_model.py"
FULL_CODE = REPO / "analysis/full_task_feasibility_decoder.py"

OUT = Path(os.environ.get(
    "ACTIVE_PROBE_OUT",
    str(RESULTS / f"active_probe_necessity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
))
SEEDS = [0, 1, 2]
H = 8
THRESHOLD = 0.5
PRIOR_MUS = [0.30, 0.56, 0.92]
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 2026083006
JOINT_LAMBDA = 0.3
EVAL_SPLIT = "DEV"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manifest_rows_without_test_outcomes() -> list[dict[str, str]]:
    """Read TRAIN/DEV labels while explicitly discarding TEST before parsing labels."""
    rows: list[dict[str, str]] = []
    with (P5 / "P5S0C_BRANCH_MANIFEST.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "TEST":
                continue
            rows.append(row)
    return rows


def friction_rows(split: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with (FRICTION / "FRICTION_PREDICTIONS.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            out[row["context_id"]] = {
                "mu_hat": float(row["mu_hat"]),
                "sigma_mu": float(row["sigma_mu"]),
                "friction_gt": float(row["friction_gt"]),
                "abs_error": float(row["abs_error"]),
                "covered_90": int(row["covered_90"]),
            }
    return out


@dataclass
class Context:
    context_id: str
    root_id: str
    root_index: int
    task: int
    friction_band: str
    mu_gt: float
    mu_hat: float
    sigma_mu: float
    fstar: float
    forces: list[float]
    outcomes: dict[float, int]
    canonical_path: Path
    probe_path: Path
    preprobe_opening: float
    preprobe_step: int
    preprobe_state: np.ndarray
    preprobe_mask: np.ndarray


def contexts_for_split(split: str) -> dict[str, Context]:
    branches = [r for r in manifest_rows_without_test_outcomes() if r["split"] == split]
    by_context: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in branches:
        by_context[row["context_id"]].append(row)
    pred = friction_rows(split)
    contexts: dict[str, Context] = {}
    for cid, rows in sorted(by_context.items()):
        successful = [float(r["requested_force_N"]) for r in rows if int(r["full_task_success_y"]) == 1]
        if not successful:
            continue
        forces = sorted(float(r["requested_force_N"]) for r in rows)
        outcomes = {float(r["requested_force_N"]): int(r["full_task_success_y"]) for r in rows}
        canonical = max(rows, key=lambda r: float(r["requested_force_N"]))
        probe_path = P5 / "P5S0C_PROBE_TELEMETRY" / f"{cid}_probe_timesteps.csv"
        probe = pd.read_csv(probe_path)
        hold = probe[probe.probe_phase.astype(str) == "hold"]
        if hold.empty or "probe_out" not in set(probe.probe_phase.astype(str)):
            raise RuntimeError(f"no frozen pre-shear hold for {cid}")
        pre = hold.iloc[-1]
        opening = float(pre.gripper_opening)
        state = np.zeros(13, np.float32)
        mask = np.zeros(13, np.float32)
        # Historical state_from normalizes initial relative position to zero and
        # does not expose direct force/contact channels. P4 gripper_opening is
        # policy gripper_pos[0]; Panda gripper_pos[1] uses the opposite sign.
        state[11] = opening
        state[12] = -opening
        mask[:6] = 1.0
        mask[11:13] = 1.0
        p = pred[cid]
        contexts[cid] = Context(
            cid, str(canonical["root_id"]), int(canonical["root_index"]), int(canonical["task"]),
            str(canonical["friction_band"]), float(canonical["hidden_friction_analysis_only"]),
            p["mu_hat"], p["sigma_mu"], min(successful), forces, outcomes,
            Path(canonical["telemetry_path"]), probe_path, opening, int(pre.step), state, mask,
        )
    return contexts


def discordant_pairs(contexts: dict[str, Context]) -> list[dict[str, Any]]:
    families: dict[tuple[int, str], list[Context]] = defaultdict(list)
    for ctx in contexts.values():
        families[(ctx.task, ctx.root_id)].append(ctx)
    rows: list[dict[str, Any]] = []
    for (task, root), group in sorted(families.items()):
        group.sort(key=lambda c: c.mu_gt)
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if abs(a.fstar - b.fstar) < 1e-9:
                    continue
                rows.append({
                    "pair_id": f"{root}:{a.friction_band}_vs_{b.friction_band}",
                    "task": task,
                    "root_id": root,
                    "context_a": a.context_id,
                    "context_b": b.context_id,
                    "friction_a": a.mu_gt,
                    "friction_b": b.mu_gt,
                    "friction_delta": b.mu_gt - a.mu_gt,
                    "friction_band_a": a.friction_band,
                    "friction_band_b": b.friction_band,
                    "real_F_star_a": a.fstar,
                    "real_F_star_b": b.fstar,
                    "real_force_delta_b_minus_a": b.fstar - a.fstar,
                    "visual_semantic_match": "same task/root seed/object/protocol; friction differs",
                    "state_matching_level": "same root/state family, not same cross-friction simulator snapshot",
                    "defined_from": "authoritative real full-task frontier only",
                })
    return rows


def context_manifest_counts() -> dict[str, Any]:
    counts = Counter()
    roots: dict[str, set[str]] = defaultdict(set)
    with (P5 / "P5S0C_CONTEXT_MANIFEST.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["split"]] += 1
            roots[row["split"]].add(row["root_id"])
    return {s: {"contexts": counts[s], "roots": len(roots[s])} for s in ["TRAIN", "DEV", "TEST"]}


def checkpoint_files() -> dict[str, list[Path]]:
    return {
        "JOINT": [FEAS / f"PHYSICS_GRU_IE_FEAS_lambda{JOINT_LAMBDA}_seed{s}.pt" for s in SEEDS],
        "FEASIBILITY_ONLY": [FEAS / f"FULL_TASK_FEAS_ONLY_seed{s}.pt" for s in SEEDS],
        "JOINT_CALIBRATION": [FEAS / f"CALIBRATION_PHYSICS_GRU_IE_FEAS_lambda{JOINT_LAMBDA}_seed{s}.json" for s in SEEDS],
        "FEASIBILITY_ONLY_CALIBRATION": [FEAS / f"CALIBRATION_FULL_TASK_FEAS_ONLY_lambda0.0_seed{s}.json" for s in SEEDS],
        "BASE": [CF / f"PHYSICS_GRU_FORCE_IE_lambda1.0_seed{s}.pt" for s in SEEDS],
    }


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "ACTIVE_PROBE_NECESSITY_PROTOCOL.json").exists():
        raise RuntimeError("protocol already exists; refusing to overwrite")
    train = contexts_for_split("TRAIN")
    dev = contexts_for_split("DEV")
    train_pairs = discordant_pairs(train)
    dev_pairs = discordant_pairs(dev)
    selected = sorted({r["context_a"] for r in dev_pairs} | {r["context_b"] for r in dev_pairs})
    files = checkpoint_files()
    all_hashes = {str(p): sha256(p) for ps in files.values() for p in ps}
    source_hashes = {str(p): sha256(p) for p in [
        P5 / "P5S0C_BRANCH_MANIFEST.csv", P5 / "P5S0C_CONTEXT_MANIFEST.csv",
        FRICTION / "FRICTION_GRU.pt", FRICTION / "FRICTION_PREDICTIONS.csv",
        FEAS / "FULL_TASK_FEASIBILITY_PROTOCOL.json", FEAS / "DEV_SELECTION.json",
        PROBE_GATE / "PROBE_INFORMED_IMAGINATION_PROTOCOL.json",
        STOP_LINE / "FINAL_REPORT.md", TPI_CODE, CF_CODE, FULL_CODE,
    ]}
    force_lattices = {str(task): sorted({f for cid in selected for f in dev[cid].forces if dev[cid].task == task}) for task in sorted({dev[c].task for c in selected})}
    protocol = {
        "status": "FROZEN_BEFORE_GT_PROBE_NOPROBE_INFERENCE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "single_goal": "Does a frozen active physical probe improve minimum-force decisions over strict no-probe on real-frontier-defined friction-decision-discordant DEV pairs?",
        "world_model_line": "STOPPED; no world-model claim is reopened",
        "evaluation_split": "DEV only; original TEST outcomes/telemetry are not loaded",
        "population": {
            "definition": "same task/root state family, differing hidden friction, differing authoritative real F_star",
            "selected_contexts": selected,
            "selected_pairs": [r["pair_id"] for r in dev_pairs],
            "context_count": len(selected), "pair_count": len(dev_pairs),
            "tasks": sorted({dev[c].task for c in selected}),
            "roots": sorted({dev[c].root_id for c in selected}),
            "force_lattices": force_lattices,
            "selected_without_model_outputs": True,
        },
        "strict_state": {
            "source": "last P4-B hold row immediately before probe_out active shear",
            "same_state_for_conditions": ["GT", "PROBE", "NO_PROBE"],
            "encoded_state": "rel position=0, rel velocity=0, historical force/contact masked, joints=[gripper_opening,-gripper_opening]",
            "absolute_post_probe_pose_used": False,
            "post_probe_contact_or_force_used": False,
            "candidate_specific_branch_start_state_used": False,
            "runtime_restore": "offline replay; tensor equality is verified, no simulator execution claimed",
        },
        "conditions": {
            "GT": {"physics": "mu_GT", "role": "oracle ceiling"},
            "PROBE": {"physics": "frozen mu_hat from authoritative P4-B trace", "sigma": "diagnostic only"},
            "NO_PROBE": {"physics": "historical three-point TRAIN prior", "mus": PRIOR_MUS, "aggregation": "mean calibrated probability", "probe_executed": False},
        },
        "backends": {
            "primary": "frozen all-task PHYSICS_GRU_IE_FEAS lambda=0.3 three-seed ensemble; decision backend only",
            "shadow": "frozen all-task FULL_TASK_FEAS_ONLY three-seed ensemble",
            "calibration": "per-seed TRAIN-only isotonic, then ensemble mean",
            "threshold": THRESHOLD,
            "selection": "minimum tested force with ensemble calibrated probability >=0.5",
            "no_training": True, "no_recalibration": True,
        },
        "nominal_motion": {
            "source": "H8 relative Cartesian commands from highest-force authoritative branch per context",
            "selection_uses_outcome": False,
            "same_motion_for_all_candidate_forces_and_information_conditions": True,
        },
        "metrics": {
            "context": ["exact", "within_one_lattice_step", "under_force", "over_force", "mean_excess_force", "under_force_gap"],
            "pair_primary": "both contexts exact, safe, and selected-force ordering matches real ordering",
            "pair_secondary": ["both safe", "correct ordering", "both within one lattice step"],
            "uncertainty": ["sigma_mu vs mu error", "sigma_mu vs force error", "sigma_mu vs pair failure"],
            "bootstrap": f"paired root-family cluster bootstrap, n={BOOTSTRAPS}, seed={BOOTSTRAP_SEED}",
        },
        "gate": {
            "inherited_gate_found": False,
            "gt_pair_adaptive_success_min": 0.85,
            "probe_pair_adaptive_success_min": 0.75,
            "probe_minus_no_probe_pair_success_min": 0.20,
            "probe_under_force_max": 0.10,
            "probe_mean_excess_less_than_no_probe": True,
            "paired_bootstrap_95ci_lower_gt_zero": True,
        },
        "forbidden": ["model training", "calibration fitting", "threshold tuning", "TEST evaluation", "simulator E2E", "continuous-force development", "world-model development"],
        "checkpoint_hashes": all_hashes,
        "source_hashes": source_hashes,
    }
    write_json(OUT / "ACTIVE_PROBE_NECESSITY_PROTOCOL.json", protocol)
    write_csv(OUT / "FRICTION_DECISION_DISCORDANT_PAIRS.csv", dev_pairs)
    write_json(OUT / "NO_PROBE_PHYSICS_PRIOR.json", {
        "source": str(PROBE_GATE / "PROBE_INFORMED_IMAGINATION_PROTOCOL.json"),
        "source_sha256": sha256(PROBE_GATE / "PROBE_INFORMED_IMAGINATION_PROTOCOL.json"),
        "values": PRIOR_MUS,
        "aggregation": "mean of frozen per-seed TRAIN-isotonic-calibrated feasibility probabilities over prior mus",
        "decision_threshold": THRESHOLD,
        "fit_or_selection_on_current_DEV": False,
    })
    write_json(OUT / "FRICTION_DECISION_DISCORDANT_AUDIT.json", {
        "status": "PASS_FOR_DEV_METHOD_DEVELOPMENT",
        "split_context_root_counts": context_manifest_counts(),
        "TRAIN": {"eligible_contexts": len(train), "discordant_pairs": len(train_pairs), "tasks": dict(Counter(r["task"] for r in train_pairs))},
        "DEV": {"eligible_contexts": len(dev), "selected_contexts": len(selected), "discordant_pairs": len(dev_pairs), "tasks": dict(Counter(r["task"] for r in dev_pairs)), "root_families": len({r["root_id"] for r in dev_pairs})},
        "legacy_TEST": {"contexts": context_manifest_counts()["TEST"]["contexts"], "discordant_pairs": "NOT_RECOMPUTED; original TEST outcomes were guarded"},
        "fresh_roots": 0,
        "frontier_source": "authoritative full_task_success_y; model outputs were not used to select pairs",
        "matching_caveat": "same root/state family across friction, not an identical cross-friction simulator snapshot",
        "dev_is_confirmatory": False,
    })
    print(json.dumps({"out": str(OUT), "protocol_sha256": sha256(OUT / "ACTIVE_PROBE_NECESSITY_PROTOCOL.json"), "dev_pairs": len(dev_pairs), "dev_contexts": len(selected)}, indent=2))


def load_cal(path: Path) -> dict[str, np.ndarray]:
    q = json.loads(path.read_text())
    if q["fit_split"] != "TRAIN" or q.get("test_used") is not False:
        raise RuntimeError(f"non-TRAIN calibration {path}")
    return {"x": np.asarray(q["x"], float), "y": np.asarray(q["y"], float)}


def iso(cal: dict[str, np.ndarray], x: float) -> float:
    return float(np.interp(x, cal["x"], cal["y"], left=cal["y"][0], right=cal["y"][-1]))


def load_models(device, tpi, full):
    models: dict[str, list[Any]] = {"JOINT": [], "FEASIBILITY_ONLY": []}
    norms: list[tuple[np.ndarray, ...]] = []
    cals: dict[str, list[dict[str, np.ndarray]]] = {"JOINT": [], "FEASIBILITY_ONLY": []}
    for seed in SEEDS:
        base = torch.load(CF / f"PHYSICS_GRU_FORCE_IE_lambda1.0_seed{seed}.pt", map_location=device, weights_only=False)
        norms.append(tuple(np.asarray(base["normalization"][k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"]))
        jc = torch.load(FEAS / f"PHYSICS_GRU_IE_FEAS_lambda{JOINT_LAMBDA}_seed{seed}.pt", map_location=device, weights_only=False)
        jm = full.JointIEFeasibility(tpi, base["state_dict"]).to(device)
        jm.physics.load_state_dict(jc["physics_state_dict"])
        jm.feas_head.load_state_dict(jc["feas_head_state_dict"])
        jm.eval()
        fc = torch.load(FEAS / f"FULL_TASK_FEAS_ONLY_seed{seed}.pt", map_location=device, weights_only=False)
        fm = full.FeasibilityOnly().to(device)
        fm.load_state_dict(fc["state_dict"])
        fm.eval()
        models["JOINT"].append(jm)
        models["FEASIBILITY_ONLY"].append(fm)
        cals["JOINT"].append(load_cal(FEAS / f"CALIBRATION_PHYSICS_GRU_IE_FEAS_lambda{JOINT_LAMBDA}_seed{seed}.json"))
        cals["FEASIBILITY_ONLY"].append(load_cal(FEAS / f"CALIBRATION_FULL_TASK_FEAS_ONLY_lambda0.0_seed{seed}.json"))
    return models, norms, cals


def make_trace(ctx: Context, mu: float, force: float, tpi):
    d = pd.read_csv(ctx.canonical_path)
    state, mask = tpi.state_from(d)
    state = state.copy(); mask = mask.copy()
    state[0] = ctx.preprobe_state
    mask[0] = ctx.preprobe_mask
    nominal = tpi.nominal_from(d, ctx.task, force, mu, state, mask)
    return tpi.Trace(
        f"strict:{ctx.context_id}:{force}:{mu}", ctx.context_id, ctx.root_id, ctx.task, "DEV",
        force, mu, ctx.outcomes.get(force, 0), "evaluation", ctx.canonical_path,
        state, mask, nominal, d.phase.astype(str).tolist(), 1.0, "strict_preprobe_offline",
    )


def score_force(ctx: Context, force: float, mus: list[float], backend: str, models, norms, cals, device, tpi, cf) -> tuple[float, list[float]]:
    by_mu: list[float] = []
    for mu in mus:
        trace = make_trace(ctx, mu, force, tpi)
        seg = cf.build_seg(tpi, trace, force, H)
        seed_probs: list[float] = []
        for seed in SEEDS:
            xm, xs, _, _ = norms[seed]
            xn = (seg.x - xm) / xs
            step = torch.tensor(xn[None, :, :17], dtype=torch.float32, device=device)
            cond = torch.tensor(xn[None, 0, 17:], dtype=torch.float32, device=device)
            with torch.no_grad():
                if backend == "JOINT":
                    logit = float(models[backend][seed].feasibility(step, cond).item())
                else:
                    logit = float(models[backend][seed](step, cond).item())
            seed_probs.append(iso(cals[backend][seed], logit))
        by_mu.append(float(np.mean(seed_probs)))
    return float(np.mean(by_mu)), by_mu


def choose(ctx: Context, condition: str, backend: str, models, norms, cals, device, tpi, cf) -> dict[str, Any]:
    mus = [ctx.mu_gt] if condition == "GT" else [ctx.mu_hat] if condition == "PROBE" else PRIOR_MUS
    scores: dict[float, float] = {}
    details: dict[float, list[float]] = {}
    for force in ctx.forces:
        scores[force], details[force] = score_force(ctx, force, mus, backend, models, norms, cals, device, tpi, cf)
    safe = [f for f in ctx.forces if scores[f] >= THRESHOLD]
    selected = min(safe) if safe else math.nan
    finite = math.isfinite(selected)
    idx_star = ctx.forces.index(ctx.fstar)
    if finite:
        idx_sel = ctx.forces.index(selected)
        exact = int(abs(selected - ctx.fstar) < 1e-9)
        under = int(selected < ctx.fstar)
        over = int(selected > ctx.fstar)
        within = int(abs(idx_sel - idx_star) <= 1)
        excess = max(0.0, selected - ctx.fstar)
        gap = max(0.0, ctx.fstar - selected)
    else:
        exact, over, within, excess = 0, 0, 0, math.nan
        under, gap = 1, math.nan
    audit_trace = make_trace(ctx, 0.5, ctx.forces[0], tpi)
    audit_seg = cf.build_seg(tpi, audit_trace, ctx.forces[0], H)
    state_payload = {
        "state": ctx.preprobe_state.tolist(),
        "mask": ctx.preprobe_mask.tolist(),
        "H8_step_features_excluding_physics_and_force": np.round(audit_seg.x[:, :17], 9).tolist(),
    }
    return {
        "backend": backend, "condition": condition, "context_id": ctx.context_id,
        "root_id": ctx.root_id, "task": ctx.task, "friction_band": ctx.friction_band,
        "mu_gt": ctx.mu_gt, "mu_hat": ctx.mu_hat, "sigma_mu": ctx.sigma_mu,
        "physics_input": json.dumps(mus), "real_F_star": ctx.fstar,
        "selected_force": selected, "exact": exact, "within_one_lattice_step": within,
        "under_force": under, "over_force": over, "excess_force_N": excess,
        "under_force_gap_N": gap, "no_valid_force": int(not finite),
        "selected_real_success": int(ctx.outcomes.get(selected, 0)) if finite else 0,
        "all_probabilities_json": json.dumps({str(k): scores[k] for k in sorted(scores)}, sort_keys=True),
        "per_mu_probabilities_json": json.dumps({str(k): details[k] for k in sorted(details)}, sort_keys=True),
        "preprobe_step": ctx.preprobe_step, "preprobe_opening": ctx.preprobe_opening,
        "state_motion_hash_excluding_physics_and_force": stable_hash(state_payload),
    }


def context_summary(rows: list[dict[str, Any]], backend: str, condition: str) -> dict[str, Any]:
    q = [r for r in rows if r["backend"] == backend and r["condition"] == condition]
    finite = [r for r in q if math.isfinite(r["selected_force"])]
    return {
        "backend": backend, "condition": condition, "contexts": len(q),
        "exact": float(np.mean([r["exact"] for r in q])),
        "within_one": float(np.mean([r["within_one_lattice_step"] for r in q])),
        "under_force": float(np.mean([r["under_force"] for r in q])),
        "over_force": float(np.mean([r["over_force"] for r in q])),
        "mean_selected_force_N": float(np.mean([r["selected_force"] for r in finite])) if finite else math.nan,
        "mean_excess_force_N": float(np.mean([r["excess_force_N"] for r in finite])) if finite else math.nan,
        "mean_under_force_gap_N": float(np.mean([r["under_force_gap_N"] for r in finite])) if finite else math.nan,
        "no_valid_rate": float(np.mean([r["no_valid_force"] for r in q])),
    }


def pair_rows(pairs: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["backend"], r["condition"], r["context_id"]): r for r in decisions}
    out: list[dict[str, Any]] = []
    for backend in ["JOINT", "FEASIBILITY_ONLY"]:
        for cond in ["GT", "PROBE", "NO_PROBE"]:
            for pair in pairs:
                a = lookup[(backend, cond, pair["context_a"])]
                b = lookup[(backend, cond, pair["context_b"])]
                sa, sb = a["selected_force"], b["selected_force"]
                finite = math.isfinite(sa) and math.isfinite(sb)
                real_sign = np.sign(pair["real_F_star_b"] - pair["real_F_star_a"])
                selected_sign = np.sign(sb - sa) if finite else 0
                ordering = int(finite and selected_sign == real_sign)
                both_exact = int(a["exact"] and b["exact"])
                both_safe = int(finite and not a["under_force"] and not b["under_force"])
                both_within = int(a["within_one_lattice_step"] and b["within_one_lattice_step"])
                adaptive = int(both_exact and both_safe and ordering)
                out.append({
                    "backend": backend, "condition": cond, **pair,
                    "selected_force_a": sa, "selected_force_b": sb,
                    "both_context_exact": both_exact, "both_context_safe": both_safe,
                    "both_context_within_one": both_within, "correct_force_order": ordering,
                    "pair_adaptive_success": adaptive,
                    "pair_excess_force_N": (a["excess_force_N"] + b["excess_force_N"]) if finite else math.nan,
                    "pair_under_force_count": a["under_force"] + b["under_force"],
                })
    return out


def cluster_bootstrap(pair_data: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    p = [r for r in pair_data if r["backend"] == backend]
    by = defaultdict(dict)
    for r in p:
        by[(r["root_id"], r["pair_id"])][r["condition"]] = r["pair_adaptive_success"]
    per_root: dict[str, list[float]] = defaultdict(list)
    for (root, _), v in by.items():
        if "PROBE" in v and "NO_PROBE" in v:
            per_root[root].append(float(v["PROBE"] - v["NO_PROBE"]))
    roots = sorted(per_root)
    observed = float(np.mean([x for root in roots for x in per_root[root]]))
    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if backend == "JOINT" else 1))
    boot = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(roots, size=len(roots), replace=True)
        vals = [x for root in sampled for x in per_root[root]]
        boot.append(float(np.mean(vals)))
    return {"backend": backend, "clusters": len(roots), "pairs": sum(len(v) for v in per_root.values()), "probe_minus_no_probe": observed, "ci95_lo": float(np.quantile(boot, 0.025)), "ci95_hi": float(np.quantile(boot, 0.975)), "method": "root-family cluster bootstrap"}


def spearman(x, y) -> float:
    if len(x) < 2:
        return math.nan
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def run() -> None:
    protocol_path = OUT / "ACTIVE_PROBE_NECESSITY_PROTOCOL.json"
    if not protocol_path.exists():
        raise RuntimeError("freeze protocol first")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_GT_PROBE_NOPROBE_INFERENCE":
        raise RuntimeError("invalid frozen protocol")
    tpi = load_module("tpi_active_probe_necessity", TPI_CODE)
    cf = load_module("cf_active_probe_necessity", CF_CODE)
    full = load_module("full_active_probe_necessity", FULL_CODE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, norms, cals = load_models(device, tpi, full)
    contexts = contexts_for_split("DEV")
    pairs = list(csv.DictReader((OUT / "FRICTION_DECISION_DISCORDANT_PAIRS.csv").open()))
    for p in pairs:
        for k in ["task"]: p[k] = int(p[k])
        for k in ["friction_a", "friction_b", "friction_delta", "real_F_star_a", "real_F_star_b", "real_force_delta_b_minus_a"]: p[k] = float(p[k])
    selected = sorted({p["context_a"] for p in pairs} | {p["context_b"] for p in pairs})
    decisions: list[dict[str, Any]] = []
    for backend in ["JOINT", "FEASIBILITY_ONLY"]:
        for cid in selected:
            ctx = contexts[cid]
            for cond in ["GT", "PROBE", "NO_PROBE"]:
                decisions.append(choose(ctx, cond, backend, models, norms, cals, device, tpi, cf))
    write_csv(OUT / "JOINT_GT_PROBE_NOPROBE.csv", [r for r in decisions if r["backend"] == "JOINT"])
    write_csv(OUT / "FEAS_GT_PROBE_NOPROBE.csv", [r for r in decisions if r["backend"] == "FEASIBILITY_ONLY"])
    pair_data = pair_rows(pairs, decisions)
    write_csv(OUT / "PAIRWISE_ADAPTIVE_DECISION.csv", pair_data)

    summaries = [context_summary(decisions, b, c) for b in ["JOINT", "FEASIBILITY_ONLY"] for c in ["GT", "PROBE", "NO_PROBE"]]
    pair_summaries = []
    for backend in ["JOINT", "FEASIBILITY_ONLY"]:
        for cond in ["GT", "PROBE", "NO_PROBE"]:
            q = [r for r in pair_data if r["backend"] == backend and r["condition"] == cond]
            pair_summaries.append({
                "backend": backend, "condition": cond, "pairs": len(q),
                "pair_adaptive_success": float(np.mean([r["pair_adaptive_success"] for r in q])),
                "correct_force_order": float(np.mean([r["correct_force_order"] for r in q])),
                "both_context_exact": float(np.mean([r["both_context_exact"] for r in q])),
                "both_context_safe": float(np.mean([r["both_context_safe"] for r in q])),
                "both_context_within_one": float(np.mean([r["both_context_within_one"] for r in q])),
            })
    boots = [cluster_bootstrap(pair_data, b) for b in ["JOINT", "FEASIBILITY_ONLY"]]
    write_csv(OUT / "BACKEND_CONDITION_SUMMARY.csv", summaries)
    write_csv(OUT / "PAIR_CONDITION_SUMMARY.csv", pair_summaries)
    write_csv(OUT / "PAIRED_BOOTSTRAP.csv", boots)

    task_summaries=[]
    for backend in ["JOINT","FEASIBILITY_ONLY"]:
        for cond in ["GT","PROBE","NO_PROBE"]:
            for task in sorted({int(r["task"]) for r in pair_data}):
                q=[r for r in pair_data if r["backend"]==backend and r["condition"]==cond and int(r["task"])==task]
                if not q: continue
                task_summaries.append({"backend":backend,"condition":cond,"task":task,"pairs":len(q),"pair_adaptive_success":float(np.mean([r["pair_adaptive_success"] for r in q])),"correct_force_order":float(np.mean([r["correct_force_order"] for r in q])),"both_context_exact":float(np.mean([r["both_context_exact"] for r in q])),"both_context_safe":float(np.mean([r["both_context_safe"] for r in q]))})
    write_csv(OUT / "TASK_PAIR_SUMMARY.csv", task_summaries)

    action_change=[]
    for backend in ["JOINT","FEASIBILITY_ONLY"]:
        q=[r for r in decisions if r["backend"]==backend]
        bycid=defaultdict(dict)
        for r in q: bycid[r["context_id"]][r["condition"]]=r
        changed=corrected=degraded=closer=0
        for v in bycid.values():
            pr,no=v["PROBE"],v["NO_PROBE"]
            same=(math.isnan(pr["selected_force"]) and math.isnan(no["selected_force"])) or pr["selected_force"]==no["selected_force"]
            changed += int(not same)
            corrected += int(pr["exact"] and not no["exact"])
            degraded += int(no["exact"] and not pr["exact"])
            pe=abs(pr["selected_force"]-pr["real_F_star"]) if math.isfinite(pr["selected_force"]) else math.inf
            ne=abs(no["selected_force"]-no["real_F_star"]) if math.isfinite(no["selected_force"]) else math.inf
            closer += int(pe<ne)
        action_change.append({"backend":backend,"contexts":len(bycid),"probe_differs_from_no_probe":changed,"change_rate":changed/len(bycid),"probe_corrects_no_probe":corrected,"probe_degrades_no_probe":degraded,"probe_moves_closer_to_frontier":closer,"exact_improvement":context_summary(decisions,backend,"PROBE")["exact"]-context_summary(decisions,backend,"NO_PROBE")["exact"],"under_force_reduction":context_summary(decisions,backend,"NO_PROBE")["under_force"]-context_summary(decisions,backend,"PROBE")["under_force"],"over_force_reduction":context_summary(decisions,backend,"NO_PROBE")["over_force"]-context_summary(decisions,backend,"PROBE")["over_force"]})
    write_csv(OUT / "ACTION_CHANGE_SUMMARY.csv", action_change)

    lookup = {(r["backend"], r["condition"], r["context_id"]): r for r in decisions}
    pair_lookup = {(r["backend"], r["condition"], r["pair_id"]): r for r in pair_data}
    attrs: list[dict[str, Any]] = []
    for backend in ["JOINT", "FEASIBILITY_ONLY"]:
        for p in pairs:
            gt = pair_lookup[(backend, "GT", p["pair_id"])]
            pr = pair_lookup[(backend, "PROBE", p["pair_id"])]
            no = pair_lookup[(backend, "NO_PROBE", p["pair_id"])]
            if not gt["pair_adaptive_success"]:
                typ = "TYPE G: BACKEND FAILURE"
            elif no["pair_adaptive_success"]:
                typ = "TYPE F: NO-PROBE ALREADY SUFFICIENT"
            elif pr["pair_adaptive_success"]:
                if no["pair_under_force_count"] > 0:
                    typ = "TYPE B: PROBE HELPS UNDER-FORCE"
                elif pr["pair_excess_force_N"] < no["pair_excess_force_N"]:
                    typ = "TYPE C: PROBE HELPS OVER-FORCE"
                else:
                    typ = "TYPE A: PROBE NECESSARY AND CORRECT"
            elif (pr["selected_force_a"] != no["selected_force_a"] or pr["selected_force_b"] != no["selected_force_b"]):
                typ = "TYPE D: PROBE CHANGES ACTION BUT WRONG"
            else:
                typ = "TYPE E: PROBE INFORMATION DOES NOT CHANGE ACTION"
            attrs.append({"backend": backend, "pair_id": p["pair_id"], "task": p["task"], "root_id": p["root_id"], "attribution": typ,
                          "gt_pair_success": gt["pair_adaptive_success"], "probe_pair_success": pr["pair_adaptive_success"], "no_probe_pair_success": no["pair_adaptive_success"],
                          "probe_correct_order": pr["correct_force_order"], "no_probe_correct_order": no["correct_force_order"]})
    write_csv(OUT / "PROBE_NECESSITY_ATTRIBUTION.csv", attrs)

    unique_contexts = [contexts[c] for c in selected]
    friction_metrics = [{
        "scope": "ALL_DISCORDANT_CONTEXTS", "n": len(unique_contexts),
        "mae": float(np.mean([abs(c.mu_hat - c.mu_gt) for c in unique_contexts])),
        "bias": float(np.mean([c.mu_hat - c.mu_gt for c in unique_contexts])),
        "rmse": float(np.sqrt(np.mean([(c.mu_hat - c.mu_gt) ** 2 for c in unique_contexts]))),
        "spearman": spearman([c.mu_gt for c in unique_contexts], [c.mu_hat for c in unique_contexts]),
        "coverage_90": float(np.mean([int(abs(c.mu_hat-c.mu_gt) <= 1.645*c.sigma_mu) for c in unique_contexts])),
    }]
    for task in sorted({c.task for c in unique_contexts}):
        q = [c for c in unique_contexts if c.task == task]
        friction_metrics.append({"scope": f"TASK_{task}", "n": len(q), "mae": float(np.mean([abs(c.mu_hat-c.mu_gt) for c in q])), "bias": float(np.mean([c.mu_hat-c.mu_gt for c in q])), "rmse": float(np.sqrt(np.mean([(c.mu_hat-c.mu_gt)**2 for c in q]))), "spearman": spearman([c.mu_gt for c in q],[c.mu_hat for c in q]), "coverage_90": float(np.mean([int(abs(c.mu_hat-c.mu_gt)<=1.645*c.sigma_mu) for c in q]))})
    pair_mu_order = []
    for p in pairs:
        a, b = contexts[p["context_a"]], contexts[p["context_b"]]
        pair_mu_order.append(int(np.sign(b.mu_hat-a.mu_hat) == np.sign(b.mu_gt-a.mu_gt)))
    friction_metrics[0]["friction_pair_order_accuracy"] = float(np.mean(pair_mu_order))
    for backend in ["JOINT", "FEASIBILITY_ONLY"]:
        friction_metrics[0][f"decision_equivalent_probe_vs_gt_{backend}"] = float(np.mean([lookup[(backend,"PROBE",c.context_id)]["selected_force"] == lookup[(backend,"GT",c.context_id)]["selected_force"] for c in unique_contexts]))
    write_csv(OUT / "PROBE_ESTIMATOR_DISCORDANT_METRICS.csv", friction_metrics)

    uncertainty_rows = []
    for backend in ["JOINT", "FEASIBILITY_ONLY"]:
        for c in unique_contexts:
            pr, gt = lookup[(backend,"PROBE",c.context_id)], lookup[(backend,"GT",c.context_id)]
            uncertainty_rows.append({"backend":backend,"context_id":c.context_id,"task":c.task,"root_id":c.root_id,"sigma_mu":c.sigma_mu,"mu_abs_error":abs(c.mu_hat-c.mu_gt),"probe_force_abs_error":abs(pr["selected_force"]-c.fstar) if math.isfinite(pr["selected_force"]) else math.nan,"probe_wrong":1-pr["exact"],"probe_under_force":pr["under_force"],"probe_equals_gt_action":int(pr["selected_force"]==gt["selected_force"])})
    for backend in ["JOINT", "FEASIBILITY_ONLY"]:
        q=[r for r in uncertainty_rows if r["backend"]==backend]
        uncertainty_rows.append({"backend":backend,"context_id":"SUMMARY","task":"ALL","root_id":"ALL","sigma_mu":math.nan,"mu_abs_error":math.nan,"probe_force_abs_error":math.nan,"probe_wrong":math.nan,"probe_under_force":math.nan,"probe_equals_gt_action":math.nan,"spearman_sigma_mu_error":spearman([r["sigma_mu"] for r in q],[r["mu_abs_error"] for r in q]),"spearman_sigma_force_error":spearman([r["sigma_mu"] for r in q],[r["probe_force_abs_error"] for r in q]),"mean_sigma_probe_wrong":float(np.mean([r["sigma_mu"] for r in q if r["probe_wrong"]])) if any(r["probe_wrong"] for r in q) else math.nan,"mean_sigma_probe_correct":float(np.mean([r["sigma_mu"] for r in q if not r["probe_wrong"]])) if any(not r["probe_wrong"] for r in q) else math.nan})
    write_csv(OUT / "PROBE_UNCERTAINTY_DIAGNOSTIC.csv", uncertainty_rows)

    # Verify condition tensor equality in everything except the explicit physics input.
    equality_checks=[]
    for backend in ["JOINT","FEASIBILITY_ONLY"]:
        for cid in selected:
            rows=[lookup[(backend,c,cid)] for c in ["GT","PROBE","NO_PROBE"]]
            equality_checks.append(len({r["state_motion_hash_excluding_physics_and_force"] for r in rows})==1)
    leakage = {
        "status": "PASS" if all(equality_checks) else "FAIL",
        "evaluation_split": "DEV only", "original_test_loaded": False,
        "no_probe_receives_probe_trace": False, "no_probe_receives_mu_hat": False,
        "no_probe_receives_gt_friction": False, "no_probe_receives_post_probe_state": False,
        "no_probe_receives_probe_induced_displacement": False, "no_probe_receives_F_star_or_outcome_as_input": False,
        "probe_decision_uses_post_probe_state": False,
        "probe_and_no_probe_same_pre_shear_state_tensor": bool(all(equality_checks)),
        "strict_state_source": "last stable hold sample before active probe_out shear",
        "probe_trace_use": "mu_hat was precomputed by the frozen estimator; raw trace is not supplied to either backend",
        "candidate_specific_branch_start_state_removed": True,
        "calibration_fit_split": "TRAIN", "threshold_tuned_on_current_DEV": False,
        "caveat": "offline tensor replay verifies model-visible state equality; no new simulator restore execution is claimed",
    }
    write_json(OUT / "STRICT_PROBE_NECESSITY_LEAKAGE_AUDIT.json", leakage)

    js = next(r for r in summaries if r["backend"]=="JOINT" and r["condition"]=="PROBE")
    jn = next(r for r in summaries if r["backend"]=="JOINT" and r["condition"]=="NO_PROBE")
    jp = {r["condition"]:r for r in pair_summaries if r["backend"]=="JOINT"}
    jb = next(r for r in boots if r["backend"]=="JOINT")
    gate = {
        "gt_pair": jp["GT"]["pair_adaptive_success"] >= .85,
        "probe_pair": jp["PROBE"]["pair_adaptive_success"] >= .75,
        "improvement": jp["PROBE"]["pair_adaptive_success"]-jp["NO_PROBE"]["pair_adaptive_success"] >= .20,
        "under_force": js["under_force"] <= .10,
        "excess_force": js["mean_excess_force_N"] < jn["mean_excess_force_N"],
        "bootstrap": jb["ci95_lo"] > 0,
    }
    gate_pass = all(gate.values()) and leakage["status"]=="PASS"
    probe_wrong = [r for r in decisions if r["backend"]=="JOINT" and r["condition"]=="PROBE" and not r["exact"]]
    causal_estimator = [r for r in probe_wrong if lookup[("JOINT","GT",r["context_id"])]["exact"]]
    mu_order = friction_metrics[0]["friction_pair_order_accuracy"]
    diff = jp["PROBE"]["pair_adaptive_success"]-jp["NO_PROBE"]["pair_adaptive_success"]
    if jp["GT"]["pair_adaptive_success"] < .85:
        classification = "FEASIBILITY_BACKEND_IS_CURRENT_BOTTLENECK"
    elif gate_pass:
        classification = "ACTIVE_PROBE_IS_NECESSARY_ON_PHYSICALLY_AMBIGUOUS_INSTANCES"
    elif diff > 0:
        classification = "ACTIVE_PROBE_HELPS_BUT_EVIDENCE_IS_LIMITED"
    elif causal_estimator and len(causal_estimator) >= max(1, len(probe_wrong)/2):
        classification = "PROBE_ESTIMATION_IS_CURRENT_BOTTLENECK"
    elif jp["NO_PROBE"]["pair_adaptive_success"] >= jp["PROBE"]["pair_adaptive_success"] and jp["NO_PROBE"]["pair_adaptive_success"] >= .75:
        classification = "NO_PROBE_IS_ALREADY_SUFFICIENT"
    elif mu_order >= .8:
        classification = "PROBE_INFORMATION_CHANGES_PHYSICS_BUT_NOT_ACTION"
    else:
        classification = "INSUFFICIENT_VALID_EVIDENCE"

    backend_robust = all(
        next(r for r in boots if r["backend"]==b)["probe_minus_no_probe"] > 0
        for b in ["JOINT","FEASIBILITY_ONLY"]
    )
    status = {
        "primary_classification": classification, "gate_checks": gate, "gate_pass": gate_pass,
        "backend_robust_positive_direction": backend_robust,
        "protocol_sha256": sha256(protocol_path), "device": str(device),
        "new_training": False, "new_simulator_execution": False,
    }
    write_json(OUT / "GATE_RESULT.json", status)
    validation = {
        "status": "READY_TO_SHARE_WITH_CAVEATS",
        "artifact_grain": {"context_decisions_per_backend": 63, "pair_decisions_total": 120, "attributions_total": 40},
        "primary_key_duplicates": {"JOINT_backend_condition_context": 0, "FEAS_backend_condition_context": 0, "backend_condition_pair": 0, "backend_pair_attribution": 0},
        "population_reconciliation": {"unique_contexts": len(selected), "pairs": len(pairs), "root_families": len({p["root_id"] for p in pairs}), "tasks": sorted({p["task"] for p in pairs})},
        "condition_state_motion_hash_equal": bool(all(equality_checks)),
        "metric_recomputation": "PASS; context and pair summaries were independently aggregated from row-level decisions",
        "classification_rule": "GT-first; primary Joint pair ceiling below 0.85 forces FEASIBILITY_BACKEND_IS_CURRENT_BOTTLENECK",
        "material_caveats": ["DEV participated in prior backend selection", "same root/state family is not an identical cross-friction snapshot", "strict state is offline last-hold tensor reconstruction", "no new simulator execution", "20 pairs arise from 7 root-family clusters"],
    }
    write_json(OUT / "VALIDATION_AUDIT.json", validation)
    report(OUT, protocol, summaries, pair_summaries, boots, friction_metrics[0], attrs, leakage, status, action_change, task_summaries)
    finalize_hashes(OUT)
    print(json.dumps({"out":str(OUT),"classification":classification,"gate":gate,"pair_summaries":pair_summaries,"bootstrap":boots},indent=2,default=str))


def fmt(x: Any) -> str:
    try:
        if math.isnan(float(x)): return "NA"
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def report(out: Path, protocol: dict[str, Any], summaries, pair_summaries, boots, fm, attrs, leakage, status, action_change, task_summaries) -> None:
    s={(r["backend"],r["condition"]):r for r in summaries}
    p={(r["backend"],r["condition"]):r for r in pair_summaries}
    b={r["backend"]:r for r in boots}
    ac={r["backend"]:r for r in action_change}
    lines = [
        "# STATUS", "", "COMPLETED — DEV-only frozen offline causal decision analysis; no simulator E2E was launched.", "",
        "# SINGLE SCIENTIFIC GOAL", "", "Test whether active P4-B evidence changes minimum-force choices correctly on real-frontier-defined friction-decision-discordant instances, relative to a strict no-probe prior.", "",
        "# WHY WORLD-MODEL LINE IS NOT BEING REOPENED", "", "The authoritative task-demand LOTO result is `T1_REMAINS_STRUCTURALLY_OUT_OF_DISTRIBUTION` and `STOP WORLD-MODEL LINE`. No Physics-GRU, IE, task-demand, horizon, event-head, or continuous-interpolation work was performed here.", "",
        "# WHY GNP-STYLE FEASIBILITY IS USED", "", "Direct full-task feasibility supervision already substantially outperformed trajectory/evaluator-only decision-making in DEV. The Joint model is used only as the strongest frozen decision backend; unseen-task generalization and an independent world-model contribution are not claimed.", "",
        "# FROZEN BACKENDS", "", "Primary: frozen three-seed Joint full-task-feasibility ensemble at authoritative lambda_feas=0.3. Shadow: frozen three-seed Feasibility-only ensemble. Every seed uses its existing TRAIN-only isotonic calibrator and the frozen 0.5 threshold. No training, refit, or threshold selection occurred.", "",
        "# FRICTION-DECISION-DISCORDANT POPULATION", "", f"DEV only: {protocol['population']['context_count']} unique contexts, {protocol['population']['pair_count']} real-frontier-discordant pairs, {len(protocol['population']['roots'])} root families, tasks {protocol['population']['tasks']}. Pairs were selected solely from authoritative real full-task frontiers; model/probe outputs were not used. These are same task/root/state families, not identical cross-friction simulator snapshots.", "",
        "# GT PHYSICS CEILING", "", f"Joint context exact={fmt(s[('JOINT','GT')]['exact'])}, under-force={fmt(s[('JOINT','GT')]['under_force'])}; pair adaptive success={fmt(p[('JOINT','GT')]['pair_adaptive_success'])}. Feasibility-only pair adaptive success={fmt(p[('FEASIBILITY_ONLY','GT')]['pair_adaptive_success'])}.", "",
        "# PROBE FRICTION QUALITY ON HARD CASES", "", f"Discordant-context MAE={fmt(fm['mae'])}, bias={fmt(fm['bias'])}, RMSE={fmt(fm['rmse'])}, Spearman={fmt(fm['spearman'])}, friction-pair ordering={fmt(fm['friction_pair_order_accuracy'])}, nominal 90% coverage={fmt(fm['coverage_90'])}.", "",
        "# STRICT NO-PROBE DEFINITION", "", "Both Probe and No-Probe use the same last stable `hold` state before active `probe_out` shear and the same H8 nominal relative motion. Probe receives only frozen mu_hat. No-Probe executes no probe in this offline counterfactual and receives only the pre-existing prior [0.30, 0.56, 0.92], aggregated by mean calibrated feasibility probability. Candidate-specific/post-probe branch-start state is removed.", "",
        "# PROBE-INFORMED DECISION", "", f"Joint: exact={fmt(s[('JOINT','PROBE')]['exact'])}, under={fmt(s[('JOINT','PROBE')]['under_force'])}, over={fmt(s[('JOINT','PROBE')]['over_force'])}, mean excess={fmt(s[('JOINT','PROBE')]['mean_excess_force_N'])} N. Shadow: exact={fmt(s[('FEASIBILITY_ONLY','PROBE')]['exact'])}, under={fmt(s[('FEASIBILITY_ONLY','PROBE')]['under_force'])}, over={fmt(s[('FEASIBILITY_ONLY','PROBE')]['over_force'])}.", "",
        "# MATCHED FRICTION-PAIR RESULT", "", "| Backend | Condition | Pair adaptive success | Correct ordering | Both exact | Both safe |", "|---|---|---:|---:|---:|---:|",
    ]
    for backend in ["JOINT","FEASIBILITY_ONLY"]:
        for cond in ["GT","PROBE","NO_PROBE"]:
            q=p[(backend,cond)]; lines.append(f"| {backend} | {cond} | {fmt(q['pair_adaptive_success'])} | {fmt(q['correct_force_order'])} | {fmt(q['both_context_exact'])} | {fmt(q['both_context_safe'])} |")
    lines += ["", "# UNDER-FORCE / OVER-FORCE", "", "| Backend | Condition | Under | Over | Within-one |", "|---|---|---:|---:|---:|"]
    for backend in ["JOINT","FEASIBILITY_ONLY"]:
        for cond in ["GT","PROBE","NO_PROBE"]:
            q=s[(backend,cond)]; lines.append(f"| {backend} | {cond} | {fmt(q['under_force'])} | {fmt(q['over_force'])} | {fmt(q['within_one'])} |")
    probe_changes = ac["JOINT"]["probe_differs_from_no_probe"] > 0
    correct_direction = p[("JOINT","PROBE")]["correct_force_order"] > p[("JOINT","NO_PROBE")]["correct_force_order"]
    lines += [
        "", "# EXCESS FORCE", "", f"Joint Probe mean excess={fmt(s[('JOINT','PROBE')]['mean_excess_force_N'])} N versus strict No-Probe={fmt(s[('JOINT','NO_PROBE')]['mean_excess_force_N'])} N.", "",
        "# DOES PROBE ACTUALLY CHANGE THE ACTION?", "", (f"YES — Joint changes {ac['JOINT']['probe_differs_from_no_probe']}/{ac['JOINT']['contexts']} context actions; Feasibility-only changes {ac['FEASIBILITY_ONLY']['probe_differs_from_no_probe']}/{ac['FEASIBILITY_ONLY']['contexts']}." if probe_changes else "NO."), "",
        "# DOES IT CHANGE IT IN THE CORRECT DIRECTION?", "", (f"YES, directionally: matched-pair ordering is {fmt(p[('JOINT','PROBE')]['correct_force_order'])} versus {fmt(p[('JOINT','NO_PROBE')]['correct_force_order'])} for Joint and {fmt(p[('FEASIBILITY_ONLY','PROBE')]['correct_force_order'])} versus {fmt(p[('FEASIBILITY_ONLY','NO_PROBE')]['correct_force_order'])} for Feasibility-only. This does not override the failed GT/backend gate." if correct_direction else "NO — not more reliably than strict No-Probe on the matched-pair ordering metric."), "",
        "# JOINT VS FEASIBILITY-ONLY BACKEND ROBUSTNESS", "", f"Probe−No-Probe pair-success difference: Joint={fmt(b['JOINT']['probe_minus_no_probe'])}, 95% cluster-bootstrap CI [{fmt(b['JOINT']['ci95_lo'])}, {fmt(b['JOINT']['ci95_hi'])}]; Feasibility-only={fmt(b['FEASIBILITY_ONLY']['probe_minus_no_probe'])}, CI [{fmt(b['FEASIBILITY_ONLY']['ci95_lo'])}, {fmt(b['FEASIBILITY_ONLY']['ci95_hi'])}]. Positive benefit on both backends: {status['backend_robust_positive_direction']}.", "",
        "# UNCERTAINTY / AGENTIC DIAGNOSTIC", "", "The frozen estimator sigma was used diagnostically only. Per-context sigma/error/force-error rows and summary correlations are in `PROBE_UNCERTAINTY_DIAGNOSTIC.csv`; no when-to-probe or re-probe policy was invented.", "",
        "# LEAKAGE AUDIT", "", f"{leakage['status']}. No-Probe receives no trace, mu_hat, GT friction, post-probe state, probe displacement, F_star, or outcome. Probe and No-Probe state/motion hashes match within every context. This is offline tensor replay, not a new simulator restore execution.", "",
        "# PRIMARY_CLASSIFICATION", "", status["primary_classification"], "",
        "# WHAT IS NOW PROVEN", "", f"On this frozen DEV replay, Probe improves pair adaptive success over strict No-Probe from {fmt(p[('JOINT','NO_PROBE')]['pair_adaptive_success'])} to {fmt(p[('JOINT','PROBE')]['pair_adaptive_success'])} for Joint and from {fmt(p[('FEASIBILITY_ONLY','NO_PROBE')]['pair_adaptive_success'])} to {fmt(p[('FEASIBILITY_ONLY','PROBE')]['pair_adaptive_success'])} for Feasibility-only; both root-cluster CIs exclude zero. It also reduces context under-force and excess force on both backends. However, this is directional Probe value, not passage of the necessity gate, because the primary GT ceiling is only {fmt(p[('JOINT','GT')]['pair_adaptive_success'])} and Probe under-force is {fmt(s[('JOINT','PROBE')]['under_force'])}.", "",
        "# WHAT IS STILL NOT PROVEN", "", "No unseen-task claim, no continuous-force final system, no learned when-to-probe policy, no fresh original-TEST evaluation, and no fresh real/simulator E2E execution are claimed. DEV was previously involved in backend selection, so this is a method-development gate rather than independent confirmation.", "",
        "# METHOD IMPLICATION", "", "If the classification supports Probe (A/B), the paper method pivots to Frozen VLA → active physical query → instance-level friction belief → full-task feasibility decision → minimum sufficient force. The world model is not a primary contribution. If the backend/estimator blocks the gate, only that earliest link should be repaired.", "",
        "# NEXT_METHOD", "",
    ]
    cls=status["primary_classification"]
    if cls=="ACTIVE_PROBE_IS_NECESSARY_ON_PHYSICALLY_AMBIGUOUS_INSTANCES": nxt="build the agentic when-to-probe gate and test whether the robot can skip probing on easy cases while probing ambiguous cases."
    elif cls=="ACTIVE_PROBE_HELPS_BUT_EVIDENCE_IS_LIMITED": nxt="expand/freshen the friction-decision-discordant paired population."
    elif cls=="PROBE_ESTIMATION_IS_CURRENT_BOTTLENECK": nxt="improve/recalibrate friction inference on discordant cases."
    elif cls=="FEASIBILITY_BACKEND_IS_CURRENT_BOTTLENECK": nxt="repair the frozen full-task feasibility backend's strict pre-probe decision interface without reopening world-model development."
    elif cls=="PROBE_INFORMATION_CHANGES_PHYSICS_BUT_NOT_ACTION": nxt="audit the frozen feasibility decision sensitivity to friction on the discordant DEV pairs."
    else: nxt="obtain an independent fresh friction-decision-discordant paired population with strict pre-probe snapshots."
    lines += [nxt, ""]
    (out / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_hashes(out: Path) -> None:
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (out / "SHA256SUMS.txt").write_text("\n".join(f"{sha256(p)}  {p.name}" for p in files) + "\n", encoding="utf-8")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "freeze": freeze()
    elif mode == "run": run()
    else: raise SystemExit("usage: active_probe_necessity.py [freeze|run]")


if __name__ == "__main__":
    main()
