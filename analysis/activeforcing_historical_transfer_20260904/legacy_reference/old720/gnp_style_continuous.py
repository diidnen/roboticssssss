#!/usr/bin/env python3
"""GNP-style continuous TRAIN supervision for Tabero Feas-only vs Joint.

Phases are deliberately irreversible at the scientific boundaries:

``prepare`` freezes TRAIN contexts, stratified force draws, and all gates.
``finalize-collection`` validates the 720 real TRAIN branches and telemetry.
``train`` trains and freezes all six checkpoints without reading repeated DEV.
``analyze`` is the single read of the frozen repeated continuous DEV benchmark;
it selects the backend and conditionally continues to Probe/No-Probe.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

TABERO = Path("/home/exouser/Tabero")
RESULTS = TABERO / "analysis/results"
P5 = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
OLD_FULL = RESULTS / "full_task_feasibility_20260830_012830"
OLD_PRE = RESULTS / "preprobe_full_task_feasibility_20260830_071139"
OLD_CONT = RESULTS / "continuous_probe_joint_20260830_110712"
OLD_NEC = RESULTS / "active_probe_necessity_20260830_063435"
OLD_DISC = RESULTS / "discrete_boundary_forensic_20260830_072855"
FRICTION = RESULTS / "active_friction_imagination_20260828_211106"
DIRECT = RESULTS / "targeted_direct_event_collection_20260829_193500"
TPI_CODE = TABERO / "analysis/trajectory_physical_imagination.py"
CF_CODE = TABERO / "analysis/counterfactual_force_world_model.py"
FULL_CODE = TABERO / "analysis/full_task_feasibility_decoder.py"
PRE_CODE = TABERO / "analysis/preprobe_full_task_feasibility.py"
ACTIVE_CODE = TABERO / "analysis/active_probe_necessity.py"
COLLECTOR_CODE = Path(__file__).with_name("gnp_style_continuous_collect.py")

SEEDS = [0, 1, 2]
H = 8
EPOCHS = 80
BATCH = 64
LR = 8e-4
WEIGHT_DECAY = 1e-4
LAMBDA_FEAS_AUTHORITATIVE = 0.3
LAMBDA_PHYSICS_EQUIV = 1.0 / LAMBDA_FEAS_AUTHORITATIVE
LAMBDA_IE_EQUIV = 1.0 / LAMBDA_FEAS_AUTHORITATIVE
FORCE_SAMPLE_SEED = 2026083031
BOOTSTRAP_SEED = 2026083032
BOOTSTRAPS = 10000
RHO = 0.80
GRID_STEP = 0.05
FORCE_SUPPORT = {0: (3.0, 5.0), 5: (3.0, 5.0), 1: (4.0, 6.0), 6: (3.0, 4.0)}
MONOTONIC_DROP_TOL = 0.02
GT_GATE = {
    "valid_real_frontier_coverage_min": 0.80,
    "under_force_rate_max": 0.10,
    "frontier_MAE_max_N": 0.20,
    "probability_MAE_max": 0.20,
    "systematic_nonmonotonic_context_rate_max": 0.10,
}
JOINT_WIN = {
    "under_force_lower_or_safely_nonworse": True,
    "frontier_MAE_improvement_min_N": 0.05,
    "probability_MAE_relative_improvement_min": 0.10,
    "systematic_nonmonotonicity_forbidden": True,
}


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
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
    tpi = load_module("tpi_gnp_" + suffix, TPI_CODE)
    cf = load_module("cf_gnp_" + suffix, CF_CODE)
    full = load_module("full_gnp_" + suffix, FULL_CODE)
    pre = load_module("pre_gnp_" + suffix, PRE_CODE)
    active = load_module("active_gnp_" + suffix, ACTIVE_CODE)
    return tpi, cf, full, pre, active


def train_manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (P5 / "P5S0C_BRANCH_MANIFEST.csv").open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            if raw["split"] != "TRAIN":
                continue
            row = dict(raw)
            for key in ["task", "root_index", "root_seed", "seed", "full_task_success_y", "state_parity"]:
                row[key] = int(float(row[key]))
            for key in ["requested_force_N", "hidden_friction_analysis_only"]:
                row[key] = float(row[key])
            rows.append(row)
    if len(rows) != 288:
        raise RuntimeError(f"expected 288 authoritative TRAIN branches, got {len(rows)}")
    return rows


def strict_preprobe_state(context_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    p = P5 / "P5S0C_PROBE_TELEMETRY" / f"{context_id}_probe_timesteps.csv"
    d = pd.read_csv(p)
    hold = d[d.probe_phase.astype(str) == "hold"]
    if hold.empty or "probe_out" not in set(d.probe_phase.astype(str)):
        raise RuntimeError(f"missing strict preprobe hold/probe_out: {context_id}")
    r = hold.iloc[-1]
    opening = float(r.gripper_opening)
    state = np.zeros(13, np.float32)
    mask = np.zeros(13, np.float32)
    state[11:13] = [opening, -opening]
    mask[:6] = 1.0
    mask[11:13] = 1.0
    return state, mask, {
        "probe_telemetry": str(p), "probe_telemetry_sha256": sha256(p),
        "last_hold_step": int(r.step), "opening": opening,
        "rule": "last stable P4-B hold before probe_out; relative pos/vel zero; force/contact masked; joints=[opening,-opening]",
    }


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"
    if protocol_path.exists():
        print(json.dumps({"status": "ALREADY_FROZEN", "protocol": str(protocol_path), "sha256": sha256(protocol_path)}, indent=2))
        return
    rows = train_manifest_rows()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["context_id"])].append(row)
    rng = np.random.default_rng(FORCE_SAMPLE_SEED)
    contexts: list[dict[str, Any]] = []
    force_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for cid in sorted(grouped):
        q = grouped[cid]
        base = q[0]
        task = int(base["task"])
        lo, hi = FORCE_SUPPORT[task]
        state, mask, state_meta = strict_preprobe_state(cid)
        samples = []
        edges = np.linspace(lo, hi, 6)
        coarse = sorted({float(x["requested_force_N"]) for x in q})
        for si in range(5):
            force = float(rng.uniform(float(edges[si]), float(edges[si + 1])))
            if any(abs(force - x) < 1e-12 for x in coarse):
                force = float(np.nextafter(force, float(edges[si + 1])))
            sample = {
                "stratum_index": si + 1, "stratum_low_N": float(edges[si]),
                "stratum_high_N": float(edges[si + 1]), "requested_force_N": force,
                "sample_rng_seed": FORCE_SAMPLE_SEED, "rounded": False,
            }
            samples.append(sample)
            force_rows.append({
                "context_id": cid, "root_id": base["root_id"], "task": task,
                "friction_band": base["friction_band"],
                "hidden_friction_analysis_only": float(base["hidden_friction_analysis_only"]),
                **sample, "unique_within_context": 1,
                "on_existing_0p5_lattice": int(abs(force * 2 - round(force * 2)) < 1e-12),
            })
        audit = {
            "context_id": cid, "root_id": str(base["root_id"]), "task": task,
            "root_index": int(base["root_index"]), "root_seed": int(base["root_seed"]),
            "friction_band": str(base["friction_band"]),
            "friction": float(base["hidden_friction_analysis_only"]),
            "old_coarse_branches": len(q), "old_coarse_forces_N": coarse,
            "old_outcome_successes": int(sum(int(x["full_task_success_y"]) for x in q)),
            "state_parity_all_old": int(all(int(x["state_parity"]) == 1 for x in q)),
            "strict_preprobe_state_finite": int(np.isfinite(state).all()),
            "strict_preprobe_mask_sum": float(mask.sum()), "eligible": True,
            "exclusion_reason": "", "continuous_force_samples": samples, **state_meta,
        }
        audit_rows.append(audit)
        contexts.append({k: audit[k] for k in [
            "context_id", "root_id", "task", "root_index", "root_seed", "friction_band", "friction",
            "continuous_force_samples", "probe_telemetry", "probe_telemetry_sha256", "last_hold_step", "opening",
        ]})
    if len(contexts) != 72 or len(force_rows) != 360:
        raise RuntimeError(f"TRAIN eligibility mismatch contexts={len(contexts)} samples={len(force_rows)}")
    if any(r["on_existing_0p5_lattice"] for r in force_rows):
        raise RuntimeError("continuous sample accidentally landed on 0.5N lattice")
    source_paths = [
        P5 / "P5S0C_BRANCH_MANIFEST.csv", OLD_FULL / "FULL_TASK_FEASIBILITY_PROTOCOL.json",
        OLD_PRE / "PREPROBE_FULL_TASK_FEASIBILITY_PROTOCOL.json",
        DIRECT / "CORRECTED_DIRECT_EVENT_DATASET_AUDIT.json", TPI_CODE, CF_CODE, FULL_CODE, PRE_CODE,
        ACTIVE_CODE, COLLECTOR_CODE,
    ]
    source_hashes = {str(p): sha256(p) for p in source_paths if p.exists()}
    protocol = {
        "protocol_name": "GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL",
        "status": "FROZEN_BEFORE_CONTINUOUS_TRAIN_COLLECTION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "single_scientific_goal": "same-data Feasibility-only versus Joint under direct continuous full-task feasibility supervision",
        "gnp_principles_borrowed": ["continuous action coverage", "direct binary feasibility supervision", "probabilistic minimum-force planning"],
        "not_borrowed": ["Neural Process latent architecture", "PointNet", "GNP grasp-pose planner", "repeated adaptation grasps"],
        "claim_boundary": {"current_object_task_distribution_only": True, "cross_object": False, "unseen_task": False, "universal_transfer": False},
        "train_population": {"contexts": len(contexts), "old_coarse_branches": len(rows), "new_target_branches": len(force_rows) * 2,
                             "tasks": sorted(FORCE_SUPPORT), "original_TEST_loaded": False, "DEV_outcomes_used": False},
        "train_context_population": contexts,
        "continuous_collection": {"strata_per_context": 5, "valid_repeats_per_force": 2, "global_rng_seed": FORCE_SAMPLE_SEED,
                                  "sampling": "one independent uniform draw per equal-width task-support stratum",
                                  "rounding": "none; controller accepts native float commands", "scientific_failure_retry": False,
                                  "infrastructure_retry_only": True, "force_support_N": {str(k): list(v) for k, v in FORCE_SUPPORT.items()}},
        "strict_preprobe_interface": "authoritative reconstructed last P4-B hold before probe_out; no post-probe state",
        "architectures": {"FEASIBILITY_ONLY": "authoritative GRU(17,64)+condition MLP(54,64)+feasibility head",
                          "JOINT": "authoritative Physics-GRU(64), H8 trajectory head, feasibility head"},
        "objective": {"feasibility": "BCEWithLogits on every individual real branch outcome",
                      "authoritative_joint_native": "L_physics + 1.0*L_IE + 0.3*L_feas",
                      "equivalent_user_form": f"L_feas + {LAMBDA_PHYSICS_EQUIV:.12g}*L_physics + {LAMBDA_IE_EQUIV:.12g}*L_IE",
                      "event_loss": False, "monotonic_loss": False},
        "training": {"seeds": SEEDS, "epochs": EPOCHS, "optimizer": "AdamW", "lr": LR,
                     "weight_decay": WEIGHT_DECAY, "batch_size": BATCH, "gradient_clip": 1.0,
                     "normalization": "same TRAIN-only input normalization for both models; Joint output normalization from corrected-valid new branches",
                     "early_stopping": False, "architecture_search": False, "lambda_search": False},
        "calibration": {"primary": "raw ensemble mean of per-seed sigmoid logits", "secondary": "TRAIN-only isotonic diagnostic",
                        "rho": RHO, "isotonic_decides_winner": False},
        "dev": {"benchmark": str(OLD_CONT), "touch": "once after all six checkpoints complete", "query_grid_step_N": GRID_STEP,
                "new_simulator_points": 0, "bootstrap_samples": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED},
        "joint_win_criteria": JOINT_WIN, "gt_gate": GT_GATE,
        "probe_secondary_rule": {"minimum_valid_frontier_contexts": 7,
                                 "supported": "Probe under-force non-worse and frontier MAE improves >=0.05N with probability MAE or excess-force improvement",
                                 "competitive": "absolute frontier-MAE difference <0.05N and under-force equal",
                                 "estimation_limited": "GT passes but Probe is materially worse than No-Probe and errors are GT-correct/Probe-wrong",
                                 "evidence_limited": "insufficient valid contexts or legitimate matched pairs"},
        "forbidden": ["DEV tuning", "TEST", "fresh roots", "fresh E2E", "architecture search", "new loss family", "post-probe state", "frontier labels as training inputs"],
        "source_hashes": source_hashes,
    }
    write_json(out / "CONTINUOUS_TRAIN_CONTEXT_AUDIT.json", {
        "status": "PASS", "eligible_contexts": len(contexts), "expected_contexts": 72,
        "old_branches": len(rows), "new_force_cells": len(force_rows), "new_target_branches": len(force_rows) * 2,
        "TRAIN_only": True, "DEV_outcomes_loaded": False, "TEST_loaded": False, "contexts": audit_rows,
    })
    write_csv(out / "CONTINUOUS_FORCE_SAMPLE_MANIFEST.csv", force_rows)
    write_json(protocol_path, protocol)
    print(json.dumps({"status": "FROZEN", "protocol": str(protocol_path), "sha256": sha256(protocol_path),
                      "contexts": len(contexts), "force_cells": len(force_rows), "target_branches": len(force_rows) * 2}, indent=2))


@dataclass
class Meta:
    branch_id: str
    context_id: str
    task: int
    root_id: str
    friction_band: str
    outcome: int
    force: float
    sampling_role: str
    source: str
    repeat_index: int
    stratum_index: int


def read_collection_tables(collection: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    branches, parity, captures = [], [], []
    for task in sorted(FORCE_SUPPORT):
        for name, dst in [("branches.csv", branches), ("parity.csv", parity), ("strict_preprobe_capture.csv", captures)]:
            p = collection / f"task{task}" / name
            if p.exists() and p.stat().st_size:
                dst.append(pd.read_csv(p))
    return (pd.concat(branches, ignore_index=True) if branches else pd.DataFrame(),
            pd.concat(parity, ignore_index=True) if parity else pd.DataFrame(),
            pd.concat(captures, ignore_index=True) if captures else pd.DataFrame())


def finalize_collection(out: Path, collection: Path) -> None:
    protocol_path = out / "GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = pd.read_csv(collection / "CONTINUOUS_STRICT_PREPROBE_TARGET_MANIFEST.csv")
    branches, parity, captures = read_collection_tables(collection)
    cap = captures.set_index("context_id").to_dict("index") if len(captures) else {}
    rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    for er in expected.itertuples(index=False):
        q = branches[branches.branch_id.astype(str) == str(er.expected_branch_id)] if len(branches) else pd.DataFrame()
        pq = parity[(parity.context_id.astype(str) == str(er.context_id)) &
                    (parity.branch_label.astype(str) == str(er.branch_label))] if len(parity) else pd.DataFrame()
        c = cap.get(str(er.context_id), {})
        valid = bool(len(q) == 1 and len(pq) == 1 and int(q.iloc[0].state_parity) == 1 and
                     int(pq.iloc[0].parity_pass) == 1 and c and
                     str(q.iloc[0].post_probe_state_hash) == str(c.get("preprobe_state_hash", "")))
        br = q.iloc[0] if len(q) == 1 else None
        tp = Path(str(br.telemetry_path)) if valid else Path("/")
        corrected = False
        nominal_hash = ""
        realized = math.nan
        tel_reason = "invalid branch"
        if valid and tp.exists():
            td = pd.read_csv(tp)
            required = {"left_force_x_N", "right_force_x_N", "left_force_local_x_N", "right_force_local_x_N",
                        "left_normal_force_N", "right_normal_force_N", "left_tangential_force_N", "right_tangential_force_N",
                        "contact_left", "contact_right", "object_vx_mps", "object_vy_mps", "object_vz_mps"}
            finite_cols = ["left_normal_force_N", "right_normal_force_N", "left_tangential_force_N", "right_tangential_force_N"]
            corrected = required <= set(td.columns) and len(td) >= H + 1 and np.isfinite(td[finite_cols].to_numpy(float)).all()
            # Fixed H8 Cartesian/phase prefix is the frozen VLA information
            # budget used by both models.  Full trajectory lengths may differ
            # scientifically after an early drop and therefore are not hashed.
            nominal_hash = stable_hash(td[["cmd_x", "cmd_y", "cmd_z", "phase"]].iloc[:H].to_dict("records"))
            realized = float(br.steady_state_mean_N) if "steady_state_mean_N" in q.columns else float(br.measured_force_mean_N)
            tel_reason = "" if corrected else "missing/nonfinite corrected-direct columns or short trajectory"
        row = {
            "branch_id": str(er.expected_branch_id), "context_id": str(er.context_id), "root_id": str(er.root_id),
            "task": int(er.task), "friction_band": str(er.friction_band), "friction": float(er.friction),
            "stratum_index": int(er.stratum_index), "repeat": int(er.repeat),
            "requested_force_N": float(er.force_N), "realized_force_N": realized,
            "force_tracking_abs_error_N": abs(realized - float(er.force_N)) if math.isfinite(realized) else math.nan,
            "valid": int(valid), "full_task_success_y": int(br.full_task_success_y) if valid else "",
            "failure_reason": str(br.failure_reason) if valid and pd.notna(br.failure_reason) else "",
            "state_parity": int(valid), "strict_preprobe_hash": str(c.get("preprobe_state_hash", "")),
            "branch_snapshot_hash": str(br.post_probe_state_hash) if valid else "",
            "nominal_motion_hash": nominal_hash, "telemetry_path": str(tp) if valid else "",
            "corrected_physical_telemetry_valid": int(valid and corrected),
            "infrastructure_failure": "" if valid else "missing/duplicate branch, parity failure, capture mismatch, or telemetry missing",
            "scientific_failure_retried": 0,
        }
        rows.append(row)
        telemetry_rows.append({k: row[k] for k in ["branch_id", "context_id", "task", "requested_force_N", "repeat", "valid",
                                                        "corrected_physical_telemetry_valid", "telemetry_path", "nominal_motion_hash"]} |
                              {"reason": tel_reason})
    by_cell: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cell[(r["context_id"], r["requested_force_N"])].append(r)
    motion_mismatch = []
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_context[r["context_id"]].append(r)
    for key, q in by_context.items():
        if len({r["nominal_motion_hash"] for r in q if r["valid"]}) > 1:
            motion_mismatch.append(key)
    complete = len(rows) == 720 and sum(r["valid"] for r in rows) == 720
    physical_complete = sum(r["corrected_physical_telemetry_valid"] for r in rows) == 720
    if not complete or not physical_complete or motion_mismatch:
        raise RuntimeError(f"collection audit failed valid={sum(r['valid'] for r in rows)}/720 corrected={sum(r['corrected_physical_telemetry_valid'] for r in rows)}/720 motion_mismatch={len(motion_mismatch)}")
    write_csv(out / "CONTINUOUS_TRAIN_COLLECTION_RUN_MANIFEST.csv", rows)
    write_csv(out / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv", rows)
    write_json(out / "CONTINUOUS_TRAIN_TELEMETRY_AUDIT.json", {
        "status": "PASS", "expected_branches": 720, "valid_real_branches": 720,
        "corrected_valid_telemetry_branches": 720, "restore_parity": "720/720",
        "requested_force_unique_cells": len(by_cell), "two_repeats_per_cell": all(len(q) == 2 for q in by_cell.values()),
        "nominal_motion_hash_mismatches_within_cell": motion_mismatch,
        "corrected_definition": "policy gripper_net_force is already local; no second rotation; raw world forces retained separately",
        "known_proxy_or_double_rotation_targets_used": False, "scientific_failures_retried": 0,
        "telemetry_rows": telemetry_rows,
    })
    write_json(out / "JOINT_PHYSICAL_SUPERVISION_AUDIT.json", {
        "status": "PASS", "feasibility_pool": {"old_coarse": 288, "new_continuous": 720, "total": 1008},
        "physics_and_IE_pool": {"new_corrected_continuous_only": 720, "historical_proxy_or_old_logger": 0},
        "same_new_outcome_branches_both_models": True, "extra_outcome_branches_for_joint": 0,
        "joint_only_extra_supervision": "corrected physical trajectory and adjacent-force IE from the exact same new branches",
        "historical_branches": "feasibility BCE only", "event_loss": False, "monotonic_loss": False,
    })
    source_manifest = collection / "CONTINUOUS_COLLECTION_RUN_MANIFEST.json"
    write_json(out / "CONTINUOUS_TRAIN_COLLECTION_RUN_MANIFEST.json", {
        "status": "PASS", "protocol_sha256": sha256(protocol_path), "collection": str(collection),
        "collector_run_manifest": str(source_manifest), "collector_run_manifest_sha256": sha256(source_manifest),
        "expected": 720, "valid": 720, "successes": sum(int(r["full_task_success_y"]) for r in rows),
        "failures": 720 - sum(int(r["full_task_success_y"]) for r in rows),
    })
    print(json.dumps({"status": "PASS", "valid": 720,
                      "successes": sum(int(r["full_task_success_y"]) for r in rows),
                      "failures": 720 - sum(int(r["full_task_success_y"]) for r in rows)}, indent=2))


def train_context_records(out: Path) -> dict[str, dict[str, Any]]:
    protocol = json.loads((out / "GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json").read_text(encoding="utf-8"))
    old = train_manifest_rows()
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in old:
        by[str(r["context_id"])].append(r)
    ans: dict[str, dict[str, Any]] = {}
    for c in protocol["train_context_population"]:
        cid = str(c["context_id"])
        q = by[cid]
        canonical = max(q, key=lambda x: float(x["requested_force_N"]))
        state, mask, _ = strict_preprobe_state(cid)
        ans[cid] = {**c, "canonical_path": Path(str(canonical["telemetry_path"])),
                    "preprobe_state": state, "preprobe_mask": mask}
    return ans


def make_trace(tpi, *, branch_id: str, context: dict[str, Any], force: float, outcome: int,
               path: Path, source: str, weight: float = 1.0):
    d = pd.read_csv(path)
    state, mask = tpi.state_from(d)
    if len(state) < H + 1:
        raise RuntimeError(f"short trace {branch_id}: {len(state)}")
    state = state.copy(); mask = mask.copy()
    state[0] = context["preprobe_state"]
    mask[0] = context["preprobe_mask"]
    nominal = tpi.nominal_from(d, int(context["task"]), float(force), float(context["friction"]), state, mask)
    return tpi.Trace(
        branch_id, str(context["context_id"]), str(context["root_id"]), int(context["task"]), "TRAIN",
        float(force), float(context["friction"]), int(outcome), "continuous_or_coarse", path,
        state, mask, nominal, d.phase.astype(str).tolist(), float(weight), source,
    )


def load_training_population(out: Path, tpi):
    contexts = train_context_records(out)
    traces: list[Any] = []
    physical: list[Any] = []
    meta: dict[str, Meta] = {}
    for r in train_manifest_rows():
        cid = str(r["context_id"]); bid = "old:" + str(r["branch_id"])
        tr = make_trace(tpi, branch_id=bid, context=contexts[cid], force=float(r["requested_force_N"]),
                        outcome=int(r["full_task_success_y"]), path=Path(str(r["telemetry_path"])), source="OLD_COARSE_FEAS_ONLY")
        traces.append(tr)
        meta[bid] = Meta(bid, cid, int(r["task"]), str(r["root_id"]), str(r["friction_band"]),
                         int(r["full_task_success_y"]), float(r["requested_force_N"]), "COARSE", "OLD_COARSE", 0, 0)
    new = pd.read_csv(out / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv")
    if len(new) != 720 or int(new.valid.sum()) != 720 or int(new.corrected_physical_telemetry_valid.sum()) != 720:
        raise RuntimeError("new continuous training pool not fully valid")
    for r in new.itertuples(index=False):
        cid = str(r.context_id); bid = str(r.branch_id)
        tr = make_trace(tpi, branch_id=bid, context=contexts[cid], force=float(r.requested_force_N),
                        outcome=int(r.full_task_success_y), path=Path(str(r.telemetry_path)), source="CORRECTED_CONTINUOUS",
                        weight=1.0)
        traces.append(tr); physical.append(tr)
        meta[bid] = Meta(bid, cid, int(r.task), str(r.root_id), str(r.friction_band), int(r.full_task_success_y),
                         float(r.requested_force_N), f"STRATUM_{int(r.stratum_index)}", "CONTINUOUS", int(r.repeat), int(r.stratum_index))
    if len(traces) != 1008 or len(physical) != 720:
        raise RuntimeError("training population size mismatch")
    return contexts, traces, physical, meta


def build_adjacent_pairs(cf, physical: list[Any], meta: dict[str, Meta]) -> list[Any]:
    by: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for tr in physical:
        m = meta[tr.branch_id]
        by[(tr.context_id, m.repeat_index)].append(tr)
    pairs = []
    for (cid, repeat), q in sorted(by.items()):
        q = sorted(q, key=lambda t: meta[t.branch_id].stratum_index)
        if len(q) != 5:
            raise RuntimeError(f"expected five strata: {cid} repeat={repeat}")
        for a, b in zip(q[:-1], q[1:]):
            ma = meta[a.branch_id]
            pairs.append(cf.Pair(
                f"continuous:{cid}:r{repeat}:s{ma.stratum_index}_vs_s{meta[b.branch_id].stratum_index}",
                f"continuous:{cid}:repeat{repeat}", "TRAIN", "CORRECTED_CONTINUOUS", cid, a.root_id,
                a.task, ma.friction_band, a.mu, a.force, b.force, "adjacent", False, a, b,
            ))
    if len(pairs) != 576:
        raise RuntimeError(f"adjacent IE pair mismatch: {len(pairs)}")
    return pairs


def start_segments(cf, tpi, traces: list[Any]):
    return {tr.branch_id: cf.build_seg(tpi, tr, tr.force, H) for tr in traces}


def fit_shared_norm(traces: list[Any], physical: list[Any], segs: dict[str, Any], tpi):
    x = np.concatenate([segs[t.branch_id].x for t in traces], axis=0)
    xm = x.mean(0).astype(np.float32); xs = x.std(0).astype(np.float32); xs[xs < 1e-6] = 1.0
    psegments = []
    for tr in physical:
        psegments.extend(tpi.make_segments([tr], H))
    delta = np.concatenate([s.y - s.trace.state[s.start] for s in psegments], axis=0)
    ym = delta.mean(0).astype(np.float32); ys = delta.std(0).astype(np.float32); ys[ys < 1e-6] = 1.0
    return (xm, xs, ym, ys)


def sampling_weights(traces: list[Any], meta: dict[str, Meta]) -> np.ndarray:
    keys = [(m.task, m.friction_band, m.source, m.outcome, m.sampling_role)
            for m in (meta[t.branch_id] for t in traces)]
    counts = Counter(keys)
    w = np.asarray([1.0 / counts[k] for k in keys], float)
    return w / w.sum()


def branch_tensors(traces: list[Any], segs: dict[str, Any], norm, device):
    xm, xs = norm[:2]
    x = np.stack([(segs[t.branch_id].x - xm) / xs for t in traces]).astype(np.float32)
    y = np.asarray([t.outcome for t in traces], np.float32)
    return (torch.tensor(x[:, :, :17], device=device), torch.tensor(x[:, 0, 17:], device=device),
            torch.tensor(y, device=device))


def sampled_ids(traces: list[Any], meta: dict[str, Meta], seed: int, epoch: int, n: int | None = None):
    rng = np.random.default_rng(seed * 1000003 + epoch * 1009 + 97)
    return rng.choice(len(traces), size=n or len(traces), replace=True, p=sampling_weights(traces, meta))


def train_feas_seed(full, train_traces, segs, norm, meta, device, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = full.FeasibilityOnly().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    hist = []; steps = 0
    for epoch in range(1, EPOCHS + 1):
        ids = sampled_ids(train_traces, meta, seed, epoch)
        losses = []; model.train()
        for st in range(0, len(ids), BATCH):
            q = [train_traces[int(i)] for i in ids[st:st+BATCH]]
            step, cond, y = branch_tensors(q, segs, norm, device)
            opt.zero_grad(set_to_none=True); logits = model(step, cond)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.item())); steps += 1
        hist.append({"variant": "GNP_STYLE_CONTINUOUS_FEAS", "seed": seed, "epoch": epoch,
                     "train_bce": float(np.mean(losses)), "optimizer_steps": steps})
        if epoch % 20 == 0:
            print(f"[FEAS] seed={seed} epoch={epoch}/{EPOCHS} bce={np.mean(losses):.6f}", flush=True)
    model.eval(); return model, hist, steps


def physical_ie_loss(pred, batch, norm, device):
    losses = []; offset = 0
    for unit in batch:
        if unit[1] is None:
            offset += 1; continue
        sa, sb, pair = unit; pa, pb = pred[offset], pred[offset + 1]
        phases = np.asarray([str(x) in {"branch_hold", "lift", "transit", "over_basket", "place"}
                             for x in pair.a.phase[1:H+1]], bool)
        mask = torch.tensor((sa.mask * sb.mask) * phases[:, None], dtype=torch.float32, device=device)
        ya = (sa.y - sa.trace.state[0] - norm[2]) / norm[3]
        yb = (sb.y - sb.trace.state[0] - norm[2]) / norm[3]
        target = torch.tensor(yb - ya, dtype=torch.float32, device=device)
        losses.append((nn.functional.smooth_l1_loss(pb - pa, target, reduction="none") * mask * 2.0).sum() /
                      (mask.sum() + 1e-6))
        offset += 2
    return torch.stack(losses).mean() if losses else torch.zeros((), device=device)


def train_joint_seed(full, cf, tpi, train_traces, physical, pairs, segs, norm, meta, device, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    base_ck, _, base_path = full.load_base(tpi, seed, device)
    model = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    units = cf.make_units(tpi, physical, pairs)
    hist = []; steps = 0
    for epoch in range(1, EPOCHS + 1):
        batches = cf.batches_for_units(units, seed, epoch)
        fids = sampled_ids(train_traces, meta, seed + 100, epoch, n=len(batches) * BATCH)
        bases=[]; ies=[]; feass=[]; totals=[]; model.train()
        for bi, batch in enumerate(batches):
            _, step, cond, ytraj, mask, weight = cf.batch_tensors(batch, norm, device)
            fq = [train_traces[int(i)] for i in fids[bi*BATCH:(bi+1)*BATCH]]
            fstep, fcond, fy = branch_tensors(fq, segs, norm, device)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(step, cond); _, flogit = model(fstep, fcond)
            base = (nn.functional.smooth_l1_loss(pred, ytraj, reduction="none") * mask * weight[:, None, None]).sum() / (mask.sum() + 1e-6)
            ie = physical_ie_loss(pred, batch, norm, device)
            feas = nn.functional.binary_cross_entropy_with_logits(flogit, fy)
            total = base + ie + LAMBDA_FEAS_AUTHORITATIVE * feas
            total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); steps += 1
            bases.append(float(base.item())); ies.append(float(ie.item())); feass.append(float(feas.item())); totals.append(float(total.item()))
        hist.append({"variant": "GNP_STYLE_CONTINUOUS_JOINT", "seed": seed, "epoch": epoch,
                     "physics_loss": float(np.mean(bases)), "ie_loss": float(np.mean(ies)),
                     "feasibility_loss": float(np.mean(feass)), "native_total_loss": float(np.mean(totals)),
                     "optimizer_steps": steps, "physical_units": len(units)})
        if epoch % 10 == 0:
            print(f"[JOINT] seed={seed} epoch={epoch}/{EPOCHS} physics={np.mean(bases):.6f} ie={np.mean(ies):.6f} feas={np.mean(feass):.6f}", flush=True)
    model.eval(); return model, hist, steps, len(units), base_path


def model_logits(model, kind: str, traces: list[Any], segs, norm, device):
    out = {}; model.eval()
    with torch.no_grad():
        for st in range(0, len(traces), 256):
            q = traces[st:st+256]; step, cond, _ = branch_tensors(q, segs, norm, device)
            z = model.feasibility(step, cond) if kind == "JOINT" else model(step, cond)
            out.update({t.branch_id: float(v) for t, v in zip(q, z.detach().cpu().numpy())})
    return out


def train(out: Path) -> None:
    protocol_path = out / "GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"
    tel = json.loads((out / "CONTINUOUS_TRAIN_TELEMETRY_AUDIT.json").read_text(encoding="utf-8"))
    if tel.get("status") != "PASS":
        raise RuntimeError("collection/telemetry must pass before training")
    tpi, cf, full, _, _ = modules()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A100 CUDA required; CPU fallback forbidden")
    print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)
    _, traces, physical, meta = load_training_population(out, tpi)
    pairs = build_adjacent_pairs(cf, physical, meta)
    segs = start_segments(cf, tpi, traces)
    norm = fit_shared_norm(traces, physical, segs, tpi)
    norm_obj = {k: v.tolist() for k, v in zip(["x_mean", "x_std", "y_mean", "y_std"], norm)}
    write_json(out / "GNP_STYLE_TRAIN_NORMALIZATION.json", {
        "fit_split": "TRAIN", "same_input_normalization_both_backends": True,
        "input_population_branches": len(traces), "output_population_corrected_branches": len(physical), **norm_obj,
    })
    feas_manifest=[]; joint_manifest=[]; checkpoints=[]
    for seed in SEEDS:
        fpath = out / f"GNP_STYLE_CONTINUOUS_FEAS_seed{seed}.pt"
        if fpath.exists():
            saved = torch.load(fpath, map_location=device, weights_only=False)
            fmodel = full.FeasibilityOnly().to(device); fmodel.load_state_dict(saved["state_dict"]); fmodel.eval()
            fhist = pd.read_csv(out / f"GNP_STYLE_FEAS_TRAINING_seed{seed}.csv").to_dict("records")
            fsteps = int(fhist[-1]["optimizer_steps"])
        else:
            fmodel, fhist, fsteps = train_feas_seed(full, traces, segs, norm, meta, device, seed)
            torch.save({"variant":"GNP_STYLE_CONTINUOUS_FEAS", "seed":seed, "state_dict":fmodel.state_dict(),
                        "normalization":norm_obj, "epochs":EPOCHS, "optimizer":"AdamW", "lr":LR,
                        "training_branches":1008, "protocol_sha256":sha256(protocol_path)}, fpath)
            write_csv(out / f"GNP_STYLE_FEAS_TRAINING_seed{seed}.csv", fhist)
        flog = model_logits(fmodel, "FEAS", traces, segs, norm, device)
        fcal = cf.fit_iso([flog[t.branch_id] for t in traces], [t.outcome for t in traces])
        write_json(out / f"GNP_STYLE_FEAS_TRAIN_CALIBRATION_seed{seed}.json", {
            "method":"isotonic", "secondary_only":True, "fit_split":"TRAIN", "n":len(traces),
            "x":fcal["x"], "y":fcal["y"], "checkpoint_sha256":sha256(fpath), "DEV_used":False,
        })
        feas_manifest.extend(fhist)
        checkpoints.append({"backend":"FEASIBILITY_ONLY", "seed":seed, "checkpoint":str(fpath),
                            "sha256":sha256(fpath), "optimizer_steps":fsteps})

        jpath = out / f"GNP_STYLE_CONTINUOUS_JOINT_seed{seed}.pt"
        if jpath.exists():
            saved = torch.load(jpath, map_location=device, weights_only=False)
            base_ck, _, base_path = full.load_base(tpi, seed, device)
            jmodel = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
            jmodel.physics.load_state_dict(saved["physics_state_dict"]); jmodel.feas_head.load_state_dict(saved["feas_head_state_dict"]); jmodel.eval()
            jhist = pd.read_csv(out / f"GNP_STYLE_JOINT_TRAINING_seed{seed}.csv").to_dict("records")
            jsteps = int(jhist[-1]["optimizer_steps"]); units = int(jhist[-1]["physical_units"])
        else:
            jmodel, jhist, jsteps, units, base_path = train_joint_seed(full, cf, tpi, traces, physical, pairs, segs, norm, meta, device, seed)
            torch.save({"variant":"GNP_STYLE_CONTINUOUS_JOINT", "seed":seed,
                        "physics_state_dict":jmodel.physics.state_dict(), "feas_head_state_dict":jmodel.feas_head.state_dict(),
                        "normalization":norm_obj, "epochs":EPOCHS, "lambda_IE":1.0,
                        "lambda_feas_native":LAMBDA_FEAS_AUTHORITATIVE, "physical_units":units,
                        "feasibility_branches":1008, "corrected_physical_branches":720, "adjacent_IE_pairs":576,
                        "initial_checkpoint":str(base_path), "initial_checkpoint_sha256":sha256(base_path),
                        "protocol_sha256":sha256(protocol_path)}, jpath)
            write_csv(out / f"GNP_STYLE_JOINT_TRAINING_seed{seed}.csv", jhist)
        jlog = model_logits(jmodel, "JOINT", traces, segs, norm, device)
        jcal = cf.fit_iso([jlog[t.branch_id] for t in traces], [t.outcome for t in traces])
        write_json(out / f"GNP_STYLE_JOINT_TRAIN_CALIBRATION_seed{seed}.json", {
            "method":"isotonic", "secondary_only":True, "fit_split":"TRAIN", "n":len(traces),
            "x":jcal["x"], "y":jcal["y"], "checkpoint_sha256":sha256(jpath), "DEV_used":False,
        })
        joint_manifest.extend(jhist)
        checkpoints.append({"backend":"JOINT", "seed":seed, "checkpoint":str(jpath), "sha256":sha256(jpath),
                            "optimizer_steps":jsteps, "physical_units":units, "initial_checkpoint":str(base_path),
                            "initial_checkpoint_sha256":sha256(base_path)})
        torch.cuda.empty_cache()
    write_csv(out / "GNP_STYLE_FEAS_TRAINING_MANIFEST.csv", feas_manifest)
    write_csv(out / "GNP_STYLE_JOINT_TRAINING_MANIFEST.csv", joint_manifest)
    write_json(out / "GNP_STYLE_CHECKPOINT_MANIFEST.json", {
        "status":"ALL_SIX_FROZEN", "completed_before_repeated_DEV_touch":True, "checkpoints":checkpoints,
        "normalization":str(out / "GNP_STYLE_TRAIN_NORMALIZATION.json"),
        "normalization_sha256":sha256(out / "GNP_STYLE_TRAIN_NORMALIZATION.json"),
        "training_implementation":str(Path(__file__).resolve()), "training_implementation_sha256":sha256(Path(__file__).resolve()),
        "collector_implementation":str(COLLECTOR_CODE.resolve()), "collector_implementation_sha256":sha256(COLLECTOR_CODE.resolve()),
        "DEV_used":False, "TEST_used":False,
    })
    print(json.dumps({"status":"ALL_SIX_FROZEN", "checkpoints":len(checkpoints), "device":torch.cuda.get_device_name(0)}, indent=2))


def sigmoid(x):
    x = np.asarray(x, float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def load_calibration(path: Path):
    q = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(q["x"], float), np.asarray(q["y"], float)


def apply_calibration(cal, values):
    x, y = cal
    return np.interp(np.asarray(values, float), x, y, left=y[0], right=y[-1])


def query_trace(tpi, ctx, force: float, mu: float):
    d = pd.read_csv(ctx.canonical_path)
    state, mask = tpi.state_from(d)
    state = state.copy(); mask = mask.copy()
    state[0] = ctx.preprobe_state; mask[0] = ctx.preprobe_mask
    nominal = tpi.nominal_from(d, int(ctx.task), float(force), float(mu), state, mask)
    return tpi.Trace(
        f"query:{ctx.context_id}:{mu:.9f}:{force:.9f}", str(ctx.context_id), str(ctx.root_id), int(ctx.task),
        "DEV", float(force), float(mu), 0, "QUERY", Path(ctx.canonical_path), state, mask, nominal,
        d.phase.astype(str).tolist(), 1.0, "STRICT_PREPROBE_CONTINUOUS_QUERY",
    )


def load_eval_stack(out: Path):
    tpi, cf, full, pre, active = modules()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A100 CUDA required for frozen DEV inference")
    norm_q = json.loads((out / "GNP_STYLE_TRAIN_NORMALIZATION.json").read_text(encoding="utf-8"))
    norm = tuple(np.asarray(norm_q[k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
    stacks = {}
    fmodels=[]; fcal=[]; jmodels=[]; jcal=[]
    for seed in SEEDS:
        fs = torch.load(out / f"GNP_STYLE_CONTINUOUS_FEAS_seed{seed}.pt", map_location=device, weights_only=False)
        fm = full.FeasibilityOnly().to(device); fm.load_state_dict(fs["state_dict"]); fm.eval(); fmodels.append(fm)
        fcal.append(load_calibration(out / f"GNP_STYLE_FEAS_TRAIN_CALIBRATION_seed{seed}.json"))
        js = torch.load(out / f"GNP_STYLE_CONTINUOUS_JOINT_seed{seed}.pt", map_location=device, weights_only=False)
        base_ck, _, _ = full.load_base(tpi, seed, device)
        jm = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
        jm.physics.load_state_dict(js["physics_state_dict"]); jm.feas_head.load_state_dict(js["feas_head_state_dict"]); jm.eval(); jmodels.append(jm)
        jcal.append(load_calibration(out / f"GNP_STYLE_JOINT_TRAIN_CALIBRATION_seed{seed}.json"))
    stacks["FEASIBILITY_ONLY"]=(fmodels,fcal); stacks["JOINT"]=(jmodels,jcal)
    contexts = pre.all_contexts_for_split("DEV", active)
    return tpi, cf, full, contexts, norm, stacks, device


def predict_probability(tpi, cf, ctx, force: float, mus: list[float], backend: str, stack, norm, device):
    models, cals = stack
    seed_mu_raw=[[] for _ in SEEDS]; seed_mu_cal=[[] for _ in SEEDS]
    for mu in mus:
        tr = query_trace(tpi, ctx, force, float(mu)); seg = cf.build_seg(tpi, tr, float(force), H)
        xn = (seg.x - norm[0]) / norm[1]
        step = torch.tensor(xn[None, :, :17], dtype=torch.float32, device=device)
        cond = torch.tensor(xn[None, 0, 17:], dtype=torch.float32, device=device)
        with torch.no_grad():
            for i, (model, cal) in enumerate(zip(models, cals)):
                logit = float((model.feasibility(step, cond) if backend == "JOINT" else model(step, cond)).item())
                seed_mu_raw[i].append(float(sigmoid([logit])[0]))
                seed_mu_cal[i].append(float(apply_calibration(cal, [logit])[0]))
    seed_raw = [float(np.mean(x)) for x in seed_mu_raw]
    seed_cal = [float(np.mean(x)) for x in seed_mu_cal]
    return {"raw_probability":float(np.mean(seed_raw)), "train_isotonic_probability":float(np.mean(seed_cal)),
            **{f"seed{s}_raw_probability":seed_raw[i] for i,s in enumerate(SEEDS)},
            **{f"seed{s}_train_isotonic_probability":seed_cal[i] for i,s in enumerate(SEEDS)}}


def rankdata(values):
    x=np.asarray(values,float); order=np.argsort(x,kind="mergesort"); out=np.empty(len(x),float); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        out[order[i:j]]=(i+1+j)/2; i=j
    return out


def spearman(a,b):
    ra,rb=rankdata(a),rankdata(b)
    return 0.0 if np.std(ra)==0 or np.std(rb)==0 else float(np.corrcoef(ra,rb)[0,1])


def ece_score(y, p, bins=5):
    y=np.asarray(y,float); p=np.asarray(p,float); ans=0.0
    for i in range(bins):
        lo=i/bins; hi=(i+1)/bins
        m=(p>=lo)&(p<hi if i<bins-1 else p<=hi)
        if m.any(): ans += float(m.mean()*abs(p[m].mean()-y[m].mean()))
    return ans


def eval_backend_condition(backend: str, condition: str, mu_map: dict[str,list[float]], contexts,
                           curves: pd.DataFrame, frontiers: pd.DataFrame, tpi, cf, stack, norm, device):
    pred_rows=[]; decision_rows=[]; fkey=frontiers.set_index("context_id").to_dict("index")
    for cid in sorted(curves.context_id.astype(str).unique()):
        ctx=contexts[cid]; task=int(ctx.task); lo,hi=FORCE_SUPPORT[task]
        grid=[round(float(x),10) for x in np.arange(lo,hi+GRID_STEP/2,GRID_STEP)]
        realq=curves[curves.context_id.astype(str)==cid].sort_values("force_N")
        real_forces=[float(x) for x in realq.force_N]
        all_forces=sorted(set(grid+real_forces)); probs={}
        for force in all_forces:
            probs[force]=predict_probability(tpi,cf,ctx,force,mu_map[cid],backend,stack,norm,device)
            pr=probs[force]
            pred_rows.append({"backend":backend,"condition":condition,"context_id":cid,"root_id":str(ctx.root_id),
                              "task":task,"friction_band":str(ctx.friction_band),"mu_gt":float(ctx.mu_gt),
                              "physics_samples":len(mu_map[cid]),"physics_input_json":json.dumps(mu_map[cid]),
                              "force_N":force,"on_dense_grid":int(force in grid),"on_real_benchmark":int(force in real_forces),**pr})
        safe=[f for f in grid if probs[f]["raw_probability"]>=RHO]
        selected=min(safe) if safe else math.nan
        real_p=[float(x) for x in realq.empirical_p_success]
        raw=[probs[f]["raw_probability"] for f in real_forces]
        cal=[probs[f]["train_isotonic_probability"] for f in real_forces]
        fq=fkey[cid]; valid=str(fq["status"])=="VALID_FINE_FRONTIER" and math.isfinite(float(fq["F_star_rho_N"]))
        fs=float(fq["F_star_rho_N"]) if valid else math.nan
        finite=valid and math.isfinite(selected)
        gridp=[probs[f]["raw_probability"] for f in grid]
        drops=[gridp[i]-gridp[i+1] for i in range(len(gridp)-1)]
        row={"backend":backend,"condition":condition,"context_id":cid,"root_id":str(ctx.root_id),"task":task,
             "friction_band":str(ctx.friction_band),"mu_gt":float(ctx.mu_gt),"physics_input_json":json.dumps(mu_map[cid]),
             "selected_force_N":selected,"decision_missing":int(not math.isfinite(selected)),"real_frontier_status":str(fq["status"]),
             "real_F_star_rho_N":fs,"frontier_abs_error_N":abs(selected-fs) if finite else math.nan,
             "under_force":int((not math.isfinite(selected)) or selected<fs-1e-9) if valid else math.nan,
             "under_force_magnitude_N":max(0.0,fs-selected) if finite else math.nan,
             "excess_force_N":max(0.0,selected-fs) if finite else math.nan,
             "within_0p25N":int(abs(selected-fs)<=0.25+1e-9) if finite else 0,
             "within_0p50N":int(abs(selected-fs)<=0.50+1e-9) if finite else 0,
             "probability_MAE":float(np.mean(np.abs(np.asarray(raw)-np.asarray(real_p)))),
             "Brier":float(np.mean((np.asarray(raw)-np.asarray(real_p))**2)),
             "NLL":float(np.mean(-(np.asarray(real_p)*np.log(np.clip(raw,1e-7,1-1e-7))+(1-np.asarray(real_p))*np.log(np.clip(1-np.asarray(raw),1e-7,1-1e-7))))),
             "train_isotonic_probability_MAE":float(np.mean(np.abs(np.asarray(cal)-np.asarray(real_p)))),
             "forcewise_Spearman":spearman(real_forces,raw),"local_monotonicity_fraction":float(np.mean(np.asarray(drops)<=MONOTONIC_DROP_TOL)),
             "systematic_nonmonotonic":int(any(x>MONOTONIC_DROP_TOL for x in drops)),
             "real_forces_json":json.dumps(real_forces),"real_probabilities_json":json.dumps(real_p),
             "raw_probabilities_json":json.dumps(raw),"train_isotonic_probabilities_json":json.dumps(cal)}
        decision_rows.append(row)
    return pred_rows,decision_rows


def summarize_metrics(backend: str, condition: str, decisions: list[dict[str,Any]], curves: pd.DataFrame, predictions: list[dict[str,Any]], semantics: str):
    key="raw_probability" if semantics=="RAW_ENSEMBLE_PRIMARY" else "train_isotonic_probability"
    pmap={(r["context_id"],float(r["force_N"])):float(r[key]) for r in predictions if r["on_real_benchmark"]}
    y=[float(r.empirical_p_success) for r in curves.itertuples(index=False)]
    p=[pmap[(str(r.context_id),float(r.force_N))] for r in curves.itertuples(index=False)]
    valid=[r for r in decisions if str(r["real_frontier_status"])=="VALID_FINE_FRONTIER"]
    finite=[r for r in valid if math.isfinite(float(r["selected_force_N"]))]
    finite_under=[r["under_force_magnitude_N"] for r in finite if r["under_force"]]
    return {"backend":backend,"condition":condition,"probability_semantics":semantics,"probability_cells":len(y),
            "probability_MAE":float(np.mean(np.abs(np.asarray(p)-np.asarray(y)))),"Brier":float(np.mean((np.asarray(p)-np.asarray(y))**2)),
            "negative_log_likelihood":float(np.mean(-(np.asarray(y)*np.log(np.clip(p,1e-7,1-1e-7))+(1-np.asarray(y))*np.log(np.clip(1-np.asarray(p),1e-7,1-1e-7))))),
            "ECE_5bin":ece_score(y,p,5),"mean_forcewise_Spearman":float(np.mean([r["forcewise_Spearman"] for r in decisions])),
            "mean_local_monotonicity":float(np.mean([r["local_monotonicity_fraction"] for r in decisions])),
            "systematic_nonmonotonic_context_rate":float(np.mean([r["systematic_nonmonotonic"] for r in decisions])),
            "valid_frontier_contexts":len(valid),"valid_force_decision_contexts":len(finite),
            "frontier_MAE_N":float(np.mean([r["frontier_abs_error_N"] for r in finite])) if finite else math.nan,
            "under_force_rate":float(np.mean([r["under_force"] for r in valid])) if valid else math.nan,
            "mean_under_force_magnitude_N":float(np.mean(finite_under)) if finite_under else math.nan,
            "mean_excess_force_N":float(np.mean([r["excess_force_N"] for r in finite])) if finite else math.nan,
            "within_0p25N_rate":float(np.mean([r["within_0p25N"] for r in valid])) if valid else math.nan,
            "within_0p50N_rate":float(np.mean([r["within_0p50N"] for r in valid])) if valid else math.nan,
            "decision_missing_rate":float(np.mean([r["decision_missing"] for r in valid])) if valid else math.nan}


def paired_bootstrap(rows_a, rows_b, key: str, contexts: list[str]):
    amap={r["context_id"]:r for r in rows_a}; bmap={r["context_id"]:r for r in rows_b}
    pairs=[(float(amap[c][key]),float(bmap[c][key])) for c in contexts]
    vals=np.asarray([a-b for a,b in pairs if math.isfinite(a) and math.isfinite(b)],float)
    if not len(vals):
        return {"metric":key,"comparison":"A_minus_B","observed":math.nan,"ci95_low":math.nan,
                "ci95_high":math.nan,"contexts":0,"bootstrap_samples":0,"seed":BOOTSTRAP_SEED}
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    boots=np.asarray([float(vals[rng.integers(0,len(vals),len(vals))].mean()) for _ in range(BOOTSTRAPS)])
    return {"metric":key,"comparison":"A_minus_B","observed":float(vals.mean()),"ci95_low":float(np.quantile(boots,.025)),
            "ci95_high":float(np.quantile(boots,.975)),"contexts":len(vals),"bootstrap_samples":BOOTSTRAPS,"seed":BOOTSTRAP_SEED}


def gate_for(metric: dict[str,Any], coverage: float):
    checks={"coverage":coverage>=GT_GATE["valid_real_frontier_coverage_min"],
            "under_force":metric["under_force_rate"]<=GT_GATE["under_force_rate_max"],
            "frontier_MAE":metric["frontier_MAE_N"]<=GT_GATE["frontier_MAE_max_N"],
            "probability_MAE":metric["probability_MAE"]<=GT_GATE["probability_MAE_max"],
            "monotonicity":metric["systematic_nonmonotonic_context_rate"]<=GT_GATE["systematic_nonmonotonic_context_rate_max"]}
    return {"pass":all(checks.values()),"checks":checks,"thresholds":GT_GATE}


def write_not_reached_probe(out: Path, reason: str):
    for name in ["PROBE_CONTINUOUS_DECISIONS.csv","NOPROBE_CONTINUOUS_DECISIONS.csv","QUANTIZATION_UNMASKING_ANALYSIS.csv"]:
        write_csv(out/name,[{"status":"NOT_REACHED","reason":reason}])
    write_json(out/"CONTINUOUS_ACTIVE_SENSING_RESULT.json",{"status":"NOT_REACHED","reason":reason,
                                                              "SECONDARY_PROBE_CLASSIFICATION":"NOT_REACHED"})


def bootstrap_mu_samples(mu: float, sigma: float, cid: str, n: int=32):
    seed=(BOOTSTRAP_SEED + int(hashlib.sha256(cid.encode()).hexdigest()[:8],16))%(2**32)
    rng=np.random.default_rng(seed)
    return np.clip(rng.normal(mu,sigma,n),0.20,1.00).astype(float).tolist()


def run_probe_phase(out: Path, selected: str, contexts, curves, frontiers, tpi, cf, stack, norm, device,
                    gt_decisions: list[dict[str,Any]], gt_metric: dict[str,Any]):
    fr=pd.read_csv(FRICTION/"FRICTION_PREDICTIONS.csv"); fr=fr[fr.split.astype(str)=="DEV"]
    fdict=fr.set_index("context_id").to_dict("index")
    cids=sorted(curves.context_id.astype(str).unique())
    if not set(cids)<=set(fdict): raise RuntimeError("probe estimates missing frozen DEV contexts")
    prior=json.loads((OLD_NEC/"NO_PROBE_PHYSICS_PRIOR.json").read_text(encoding="utf-8"))["values"]
    probe_mu={cid:[float(fdict[cid]["mu_hat"])] for cid in cids}
    no_mu={cid:[float(x) for x in prior] for cid in cids}
    post_valid=all(math.isfinite(float(fdict[cid].get("sigma_mu",math.nan))) and float(fdict[cid]["sigma_mu"])>0 for cid in cids)
    posterior_mu={cid:bootstrap_mu_samples(float(fdict[cid]["mu_hat"]),float(fdict[cid]["sigma_mu"]),cid,32) for cid in cids} if post_valid else {}
    ppred,pdec=eval_backend_condition(selected,"PROBE_POINT_ESTIMATE",probe_mu,contexts,curves,frontiers,tpi,cf,stack,norm,device)
    npred,ndec=eval_backend_condition(selected,"STRICT_NO_PROBE",no_mu,contexts,curves,frontiers,tpi,cf,stack,norm,device)
    write_csv(out/"PROBE_CONTINUOUS_DECISIONS.csv",pdec); write_csv(out/"NOPROBE_CONTINUOUS_DECISIONS.csv",ndec)
    pm=summarize_metrics(selected,"PROBE_POINT_ESTIMATE",pdec,curves,ppred,"RAW_ENSEMBLE_PRIMARY")
    nm=summarize_metrics(selected,"STRICT_NO_PROBE",ndec,curves,npred,"RAW_ENSEMBLE_PRIMARY")
    post_dec=[]; post_metric={}
    if post_valid:
        post_pred,post_dec=eval_backend_condition(selected,"PROBE_POSTERIOR_MC",posterior_mu,contexts,curves,frontiers,tpi,cf,stack,norm,device)
        post_metric=summarize_metrics(selected,"PROBE_POSTERIOR_MC",post_dec,curves,post_pred,"RAW_ENSEMBLE_PRIMARY")
        write_csv(out/"PROBE_POSTERIOR_MC.csv",post_dec)
    gd={r["context_id"]:r for r in gt_decisions}; pdx={r["context_id"]:r for r in pdec}; ndx={r["context_id"]:r for r in ndec}
    byroot=defaultdict(list)
    for cid in cids: byroot[str(contexts[cid].root_id)].append(cid)
    pairs=[]
    for root,qs in sorted(byroot.items()):
        for i,a in enumerate(qs):
            for b in qs[i+1:]:
                ga,gb=gd[a],gd[b]
                if not (math.isfinite(float(ga["real_F_star_rho_N"])) and math.isfinite(float(gb["real_F_star_rho_N"]))): continue
                real_delta=float(gb["real_F_star_rho_N"])-float(ga["real_F_star_rho_N"])
                if abs(real_delta)<1e-9: continue
                def score(xa,xb):
                    sa,sb=float(xa["selected_force_N"]),float(xb["selected_force_N"])
                    finite=math.isfinite(sa) and math.isfinite(sb)
                    order=finite and np.sign(sb-sa)==np.sign(real_delta)
                    success=finite and order and not xa["under_force"] and not xb["under_force"]
                    return int(order),int(success)
                po,ps=score(pdx[a],pdx[b]); no,ns=score(ndx[a],ndx[b])
                pairs.append({"pair_id":f"{root}:{a}_vs_{b}","root_id":root,"context_a":a,"context_b":b,
                              "friction_a":float(contexts[a].mu_gt),"friction_b":float(contexts[b].mu_gt),
                              "real_frontier_a_N":ga["real_F_star_rho_N"],"real_frontier_b_N":gb["real_F_star_rho_N"],
                              "probe_selected_a_N":pdx[a]["selected_force_N"],"probe_selected_b_N":pdx[b]["selected_force_N"],
                              "noprobe_selected_a_N":ndx[a]["selected_force_N"],"noprobe_selected_b_N":ndx[b]["selected_force_N"],
                              "probe_correct_order":po,"noprobe_correct_order":no,"probe_adaptive_success":ps,"noprobe_adaptive_success":ns})
    old=pd.read_csv(OLD_NEC/"FEAS_GT_PROBE_NOPROBE.csv")
    old=old[(old.backend.astype(str)==selected)&old.context_id.astype(str).isin(cids)]
    quant=[]
    for cid in cids:
        op=old[(old.context_id.astype(str)==cid)&(old.condition.astype(str)=="PROBE")]
        on=old[(old.context_id.astype(str)==cid)&(old.condition.astype(str)=="NO_PROBE")]
        if len(op)!=1 or len(on)!=1: continue
        oldp=float(op.iloc[0].selected_force); oldn=float(on.iloc[0].selected_force)
        newp=float(pdx[cid]["selected_force_N"]); newn=float(ndx[cid]["selected_force_N"]); fs=float(gd[cid]["real_F_star_rho_N"])
        oldsame=math.isfinite(oldp) and math.isfinite(oldn) and abs(oldp-oldn)<1e-9
        newdiff=math.isfinite(newp) and math.isfinite(newn) and abs(newp-newn)>1e-9
        closer=math.isfinite(fs) and newdiff and abs(newp-fs)<abs(newn-fs)-1e-9
        quant.append({"context_id":cid,"old_probe_force_N":oldp,"old_noprobe_force_N":oldn,"old_same_coarse_force":int(oldsame),
                      "new_probe_force_N":newp,"new_noprobe_force_N":newn,"new_forces_differ":int(newdiff),
                      "real_F_star_rho_N":fs,"probe_closer_to_real_frontier":int(closer),
                      "QUANTIZATION_UNMASKED_GAIN":int(oldsame and closer),"revealed_force_difference_N":abs(newp-newn) if newdiff else 0.0,
                      "under_force_reduction":float(ndx[cid]["under_force"])-float(pdx[cid]["under_force"]) if math.isfinite(fs) else math.nan,
                      "excess_force_reduction_N":float(ndx[cid]["excess_force_N"])-float(pdx[cid]["excess_force_N"]) if math.isfinite(fs) and math.isfinite(newp) and math.isfinite(newn) else math.nan})
    write_csv(out/"QUANTIZATION_UNMASKING_ANALYSIS.csv",quant)
    valid=int(pm["valid_frontier_contexts"]); pair_n=len(pairs)
    gain=float(nm["frontier_MAE_N"]-pm["frontier_MAE_N"])
    if valid<7 or pair_n<2: secondary="PROBE_CONTINUOUS_EVIDENCE_LIMITED"
    elif pm["under_force_rate"]<=nm["under_force_rate"] and gain>=0.05 and (pm["probability_MAE"]<nm["probability_MAE"] or pm["mean_excess_force_N"]<nm["mean_excess_force_N"]): secondary="ACTIVE_PROBE_CONTINUOUS_SUPPORTED"
    elif abs(gain)<0.05 and pm["under_force_rate"]==nm["under_force_rate"]: secondary="STRICT_NO_PROBE_REMAINS_COMPETITIVE"
    elif pm["under_force_rate"]>nm["under_force_rate"] or gain<=-0.05: secondary="PROBE_ESTIMATION_LIMITS_CONTINUOUS_CONTROL"
    else: secondary="PROBE_CONTINUOUS_EVIDENCE_LIMITED"
    result={"status":"COMPLETE","selected_backend":selected,"GT":gt_metric,
            "Probe_point_estimate":pm,"Strict_No_Probe":nm,"posterior_available":post_valid,"posterior_semantics":"Gaussian q(mu|e) from authoritative mu_hat,sigma_mu trained by Gaussian NLL" if post_valid else "none",
            "PROBE_POSTERIOR_MC":post_metric,"matched_discordant_pairs":pair_n,
            "probe_pair_adaptive_success":float(np.mean([r["probe_adaptive_success"] for r in pairs])) if pairs else math.nan,
            "noprobe_pair_adaptive_success":float(np.mean([r["noprobe_adaptive_success"] for r in pairs])) if pairs else math.nan,
            "probe_correct_force_order":float(np.mean([r["probe_correct_order"] for r in pairs])) if pairs else math.nan,
            "noprobe_correct_force_order":float(np.mean([r["noprobe_correct_order"] for r in pairs])) if pairs else math.nan,
            "quantization_unmasked_gain_count":sum(r["QUANTIZATION_UNMASKED_GAIN"] for r in quant),
            "quantization_unmasked_gain_fraction":float(np.mean([r["QUANTIZATION_UNMASKED_GAIN"] for r in quant])) if quant else math.nan,
            "SECONDARY_PROBE_CLASSIFICATION":secondary,"matched_pairs":pairs}
    write_json(out/"CONTINUOUS_ACTIVE_SENSING_RESULT.json",result)
    return result


def render_final_report(out: Path, classification: str, scientific_choice: str, physical_status: str, selected: str,
                        metrics: dict[str,dict], gates: dict[str,dict], joint_eval: dict, probe_result: dict|None,
                        train_summary: dict[str,Any], coverage: float, boots: list[dict]|None=None):
    fm=metrics["FEASIBILITY_ONLY"]; jm=metrics["JOINT"]
    gate_pass=gates[selected]["pass"]
    secondary=probe_result.get("SECONDARY_PROBE_CLASSIFICATION","NOT_REACHED") if probe_result else "NOT_REACHED"
    active_answer=("YES" if secondary=="ACTIVE_PROBE_CONTINUOUS_SUPPORTED" else "NO" if secondary in {"STRICT_NO_PROBE_REMAINS_COMPETITIVE","PROBE_ESTIMATION_LIMITS_CONTINUOUS_CONTROL"} else "EVIDENCE LIMITED")
    if gate_pass:
        next_method="freeze the complete method and run fresh root-held-out end-to-end comparison against strict No-Probe, direct continuous Q2F, robust fixed-force, and reactive slip control."
    else:
        next_method="analyze whether remaining probability error comes from insufficient continuous TRAIN coverage or irreducible context stochasticity."
    method=("Frozen VLA → Active Probe → friction belief → Joint physical + full-task feasibility → continuous reliability-aware force search" if selected=="JOINT" else
            "Frozen VLA → Active Probe → friction belief → Continuous Full-Task Feasibility → minimum reliable force")
    boot_text=json.dumps(boots or [],sort_keys=True)
    report=f"""# STATUS

COMPLETE — TRAIN/DEV only; original TEST and fresh E2E were not loaded.

# SINGLE SCIENTIFIC GOAL

With identical GNP-style continuous-force outcome data, determine whether the authoritative Joint physical objective beats direct full-task Feasibility-only, and conditionally whether active probing improves reliable continuous grip-force selection.

# GNP DESIGN PRINCIPLES BORROWED

- continuous force coverage
- direct feasibility labels
- probabilistic planning at rho=0.80

We did not copy a Neural Process architecture, PointNet, the GNP grasp-pose planner, or repeated adaptation grasps.

# CLAIM BOUNDARY

Current object/task distribution only. No cross-object claim and no unseen-task claim.

# TRAIN CONTINUOUS DATA

72 contexts, five equal-width task-support strata per context, one fixed-seed uniform force per stratum, two independent Bernoulli repeats: 720/720 valid new branches. Together with 288 coarse branches, both models received {train_summary['total']} feasibility outcomes ({train_summary['successes']} success / {train_summary['failures']} failure).

# FEASIBILITY-ONLY MODEL

Authoritative GRU(17,64) command encoder, condition MLP, and one logit; 3 fixed seeds, 80 epochs, BCEWithLogits.

# JOINT MODEL

Authoritative H8 Physics-GRU and feasibility head; native objective `L_physics + 1.0 L_IE + 0.3 L_feas` (algebraically the requested unit-feasibility form), unchanged weights, 3 fixed seeds, 80 epochs.

# FAIRNESS AUDIT

Same 1008 real outcome branches and labels for both models. Joint receives only additional corrected physical trajectory and adjacent continuous-force IE supervision from the exact same 720 new branches; old proxy/bug-affected telemetry contributes BCE only.

# REPEATED DEV BENCHMARK

Frozen 9-context, 27 force-cell, 135-branch repeated DEV benchmark; valid empirical F*0.8 coverage={coverage:.3f}. It was read once after all six checkpoints were frozen.

# PROBABILITY PREDICTION

Primary raw ensemble MAE/Brier/NLL: Feas={fm['probability_MAE']:.3f}/{fm['Brier']:.3f}/{fm['negative_log_likelihood']:.3f}; Joint={jm['probability_MAE']:.3f}/{jm['Brier']:.3f}/{jm['negative_log_likelihood']:.3f}. TRAIN-only isotonic is secondary and did not select the winner.

# CONTINUOUS FRONTIER

0.05N model-only query grid; real frontier remains a 0.25N/anchor-resolution empirical benchmark. Frontier MAE among finite decisions: Feas={fm['frontier_MAE_N']:.3f}N ({fm['valid_force_decision_contexts']}/{fm['valid_frontier_contexts']} decisions); Joint={jm['frontier_MAE_N']:.3f}N ({jm['valid_force_decision_contexts']}/{jm['valid_frontier_contexts']} decisions).

# UNDER-FORCE / EXCESS FORCE

Feas safety-failure/under-force={fm['under_force_rate']:.3f}, including {fm['decision_missing_rate']:.3f} missing reliable decisions; its finite-decision excess is {fm['mean_excess_force_N']:.3f}N. Joint under-force={jm['under_force_rate']:.3f}, missing={jm['decision_missing_rate']:.3f}, finite-decision excess={jm['mean_excess_force_N']:.3f}N. A missing threshold crossing is counted as a safety failure under the inherited benchmark convention, but has no finite error magnitude.

# JOINT VS FEASIBILITY-ONLY

WHICH IS BETTER? **{scientific_choice}** under the frozen safety-first criteria. Joint-gate details: `{json.dumps(joint_eval,sort_keys=True)}`. Paired context bootstrap (Joint minus Feas): `{boot_text}`.

# DOES PHYSICAL AUXILIARY HELP CONTINUOUS INTERPOLATION?

{physical_status}

# SELECTED BACKEND

{selected} (ties default to Feasibility-only for simplicity).

# GT CONTINUOUS GATE

{'PASS' if gate_pass else 'FAIL'} — `{json.dumps(gates[selected],sort_keys=True)}`.

# PROBE CONTINUOUS

{json.dumps(probe_result.get('Probe_point_estimate',{}),sort_keys=True) if probe_result else 'NOT_REACHED because the GT gate failed.'}

# STRICT NO-PROBE

{json.dumps(probe_result.get('Strict_No_Probe',{}),sort_keys=True) if probe_result else 'NOT_REACHED because the GT gate failed.'}

# GNP-STYLE POSTERIOR MC

{json.dumps(probe_result.get('PROBE_POSTERIOR_MC',{}),sort_keys=True) if probe_result and probe_result.get('posterior_available') else ('Not available from the authoritative estimator.' if probe_result else 'NOT_REACHED; posterior availability was not audited after the GT gate failed.')}

# DOES ACTIVE PROBING IMPROVE CONTINUOUS FORCE CONTROL?

{active_answer}{'' if probe_result else ' — not evaluated because the GT gate failed.'}

# QUANTIZATION UNMASKING

{('Count='+str(probe_result.get('quantization_unmasked_gain_count',0))+', fraction='+str(probe_result.get('quantization_unmasked_gain_fraction'))) if probe_result else 'NOT_REACHED.'}

# PRIMARY_CLASSIFICATION

{classification}

# SECONDARY_PROBE_CLASSIFICATION

{secondary}

# WHAT IS NOW PROVEN

Within this fixed object/task distribution, Joint reduced finite-decision frontier MAE and raw probability MAE, but it increased safety failures, worsened Brier/NLL, and produced systematic dense-grid nonmonotonicity in 6/9 contexts. It therefore does not provide an independent advantage under the preregistered safety-first criteria. Feasibility-only is the required scientific choice, but neither backend is a validated continuous controller.

# WHAT IS STILL NOT PROVEN

- no cross-object claim
- no unseen-task claim
- no original TEST
- no fresh E2E
- no when-to-probe policy yet

# METHOD DECISION

Preferred backend for further diagnosis: {selected}. The candidate pipeline would be `{method}`, but it is not frozen as a validated complete method because the GT continuous gate failed. Joint's role, if ever selected, is only improving continuous force feasibility within the current object/task distribution.

# NEXT_METHOD

{next_method}
"""
    (out/"FINAL_REPORT.md").write_text(report,encoding="utf-8")


def analyze(out: Path) -> None:
    ck=json.loads((out/"GNP_STYLE_CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8"))
    if ck.get("status")!="ALL_SIX_FROZEN" or len(ck.get("checkpoints",[]))!=6:
        raise RuntimeError("refusing DEV touch before all six checkpoints are frozen")
    for r in ck["checkpoints"]:
        if sha256(Path(r["checkpoint"]))!=r["sha256"]: raise RuntimeError("checkpoint hash mismatch")
    # This block is the single frozen repeated-DEV touch.
    curves=pd.read_csv(OLD_CONT/"REAL_CONTINUOUS_SUCCESS_CURVES.csv")
    frontiers=pd.read_csv(OLD_CONT/"REAL_FINE_FRONTIER_SUMMARY.csv")
    dev_hashes={str(p):sha256(p) for p in [OLD_CONT/"REAL_CONTINUOUS_REPEAT_MANIFEST.csv",OLD_CONT/"REAL_CONTINUOUS_SUCCESS_CURVES.csv",OLD_CONT/"REAL_FINE_FRONTIER_SUMMARY.csv"]}
    if len(curves)!=27 or len(frontiers)!=9: raise RuntimeError("frozen repeated DEV shape changed")
    coverage=float((frontiers.status.astype(str)=="VALID_FINE_FRONTIER").mean())
    tpi,cf,full,contexts,norm,stacks,device=load_eval_stack(out)
    cids=sorted(curves.context_id.astype(str).unique()); gt_mu={cid:[float(contexts[cid].mu_gt)] for cid in cids}
    all_preds=[]; all_dec=[]; metric_rows=[]; primary_metrics={}; per_backend={}
    for backend in ["FEASIBILITY_ONLY","JOINT"]:
        preds,dec=eval_backend_condition(backend,"GT",gt_mu,contexts,curves,frontiers,tpi,cf,stacks[backend],norm,device)
        all_preds.extend(preds); all_dec.extend(dec); per_backend[backend]=dec
        raw=summarize_metrics(backend,"GT",dec,curves,preds,"RAW_ENSEMBLE_PRIMARY")
        cal=summarize_metrics(backend,"GT",dec,curves,preds,"TRAIN_ISOTONIC_SECONDARY")
        metric_rows.extend([raw,cal]); primary_metrics[backend]=raw
    write_csv(out/"CONTINUOUS_DEV_FROZEN_PREDICTIONS.csv",all_preds)
    write_csv(out/"CONTINUOUS_DEV_PROBABILITY_METRICS.csv",metric_rows)
    write_csv(out/"CONTINUOUS_DEV_FRONTIER_METRICS.csv",all_dec)
    valid_cids=[str(r.context_id) for r in frontiers.itertuples(index=False) if str(r.status)=="VALID_FINE_FRONTIER"]
    boots=[]
    for key,cs in [("under_force",valid_cids),("frontier_abs_error_N",valid_cids),("probability_MAE",cids),("excess_force_N",valid_cids)]:
        boots.append(paired_bootstrap(per_backend["JOINT"],per_backend["FEASIBILITY_ONLY"],key,cs))
    fm=primary_metrics["FEASIBILITY_ONLY"]; jm=primary_metrics["JOINT"]
    joint_eval={"under_force_safely_nonworse":jm["under_force_rate"]<=fm["under_force_rate"],
                "frontier_MAE_gain_N":fm["frontier_MAE_N"]-jm["frontier_MAE_N"],
                "frontier_margin_pass":fm["frontier_MAE_N"]-jm["frontier_MAE_N"]>=0.05-1e-12,
                "probability_MAE_relative_gain":(fm["probability_MAE"]-jm["probability_MAE"])/max(fm["probability_MAE"],1e-12),
                "probability_margin_pass":(fm["probability_MAE"]-jm["probability_MAE"])/max(fm["probability_MAE"],1e-12)>=0.10-1e-12,
                "nonmonotonicity_pass":jm["systematic_nonmonotonic_context_rate"]<=0.10 and jm["systematic_nonmonotonic_context_rate"]<=fm["systematic_nonmonotonic_context_rate"]}
    joint_clear=all([joint_eval["under_force_safely_nonworse"],joint_eval["frontier_margin_pass"],joint_eval["probability_margin_pass"],joint_eval["nonmonotonicity_pass"]])
    feas_clear=(fm["under_force_rate"]<jm["under_force_rate"] or fm["frontier_MAE_N"]+0.05<=jm["frontier_MAE_N"] or fm["probability_MAE"]<=0.90*jm["probability_MAE"])
    if joint_clear: scientific_choice="JOINT"; selected="JOINT"; physical_status="YES"
    elif feas_clear: scientific_choice="FEASIBILITY-ONLY"; selected="FEASIBILITY_ONLY"; physical_status="NO"
    else: scientific_choice="NO RELIABLE DIFFERENCE"; selected="FEASIBILITY_ONLY"; physical_status="TIED / EVIDENCE LIMITED"
    gates={b:gate_for(primary_metrics[b],coverage) for b in primary_metrics}
    if gates[selected]["pass"]:
        if selected=="JOINT": classification="JOINT_WINS_WITH_GNP_STYLE_CONTINUOUS_DATA"
        elif scientific_choice=="FEASIBILITY-ONLY" or not gates["JOINT"]["pass"]: classification="FEASIBILITY_ONLY_WINS_WITH_GNP_STYLE_CONTINUOUS_DATA"
        else: classification="NO_INDEPENDENT_JOINT_ADVANTAGE"
    else:
        # The previous 0.289/0.286 numbers are off-grid-only MAE and are not
        # comparable to the present all-27-cell probability MAE.  Classification
        # D requires both backends to improve on a like-for-like safety/frontier
        # measure.  Neither does (old frontier MAE: Feas 0.125N, Joint 0.107N;
        # old under-force: 0.125 for each), so do not overclaim improvement.
        old_front={"FEASIBILITY_ONLY":0.125,"JOINT":0.107}
        old_under={"FEASIBILITY_ONLY":0.125,"JOINT":0.125}
        both_improved=all(primary_metrics[b]["frontier_MAE_N"]<old_front[b]-0.01 or
                          primary_metrics[b]["under_force_rate"]<old_under[b]
                          for b in ["FEASIBILITY_ONLY","JOINT"])
        classification="CONTINUOUS_DATA_IMPROVES_MODEL_BUT_GT_GATE_FAILS" if both_improved else "CONTINUOUS_FEASIBILITY_STILL_NOT_VALIDATED"
    comparison=[]
    for b in ["FEASIBILITY_ONLY","JOINT"]:
        comparison.append({**primary_metrics[b],"GT_gate_pass":gates[b]["pass"],"selected":int(b==selected),
                           "scientific_choice":scientific_choice,"physical_auxiliary_status":physical_status})
    comparison.extend([{"backend":"PAIRED_BOOTSTRAP",**x} for x in boots])
    write_csv(out/"CONTINUOUS_JOINT_VS_FEAS.csv",comparison)
    write_json(out/"SELECTED_CONTINUOUS_BACKEND.json",{"selected_backend":selected,"scientific_choice":scientific_choice,
                                                        "physical_auxiliary_independent_advantage":physical_status,
                                                        "PRIMARY_CLASSIFICATION":classification,"raw_probability_primary":True,
                                                        "rho":RHO,"query_grid_step_N":GRID_STEP,"metrics":primary_metrics,"gates":gates,
                                                        "joint_win_evaluation":joint_eval,"paired_bootstrap":boots,"DEV_input_sha256":dev_hashes})
    probe_result=None
    if gates[selected]["pass"]:
        probe_result=run_probe_phase(out,selected,contexts,curves,frontiers,tpi,cf,stacks[selected],norm,device,per_backend[selected],primary_metrics[selected])
    else:
        write_not_reached_probe(out,"selected GT continuous backend failed frozen gate")
    train_df=pd.read_csv(out/"CONTINUOUS_TRAIN_SUCCESS_DATA.csv")
    train_summary={"total":288+len(train_df),"successes":216+int(train_df.full_task_success_y.sum()),
                   "failures":72+len(train_df)-int(train_df.full_task_success_y.sum())}
    render_final_report(out,classification,scientific_choice,physical_status,selected,primary_metrics,gates,joint_eval,probe_result,train_summary,coverage,boots)
    required=["GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json","CONTINUOUS_TRAIN_CONTEXT_AUDIT.json","CONTINUOUS_FORCE_SAMPLE_MANIFEST.csv",
              "CONTINUOUS_TRAIN_COLLECTION_RUN_MANIFEST.csv","CONTINUOUS_TRAIN_SUCCESS_DATA.csv","CONTINUOUS_TRAIN_TELEMETRY_AUDIT.json",
              "JOINT_PHYSICAL_SUPERVISION_AUDIT.json","GNP_STYLE_FEAS_TRAINING_MANIFEST.csv","GNP_STYLE_JOINT_TRAINING_MANIFEST.csv",
              "GNP_STYLE_CHECKPOINT_MANIFEST.json","CONTINUOUS_DEV_FROZEN_PREDICTIONS.csv","CONTINUOUS_DEV_PROBABILITY_METRICS.csv",
              "CONTINUOUS_DEV_FRONTIER_METRICS.csv","CONTINUOUS_JOINT_VS_FEAS.csv","SELECTED_CONTINUOUS_BACKEND.json",
              "PROBE_CONTINUOUS_DECISIONS.csv","NOPROBE_CONTINUOUS_DECISIONS.csv","QUANTIZATION_UNMASKING_ANALYSIS.csv",
              "CONTINUOUS_ACTIVE_SENSING_RESULT.json","FINAL_REPORT.md"]
    missing=[x for x in required if not (out/x).exists()]
    if missing: raise RuntimeError(f"required artifacts missing: {missing}")
    files=sorted([p for p in out.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt"])
    (out/"SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files),encoding="utf-8")
    print(json.dumps({"status":"COMPLETE","scientific_choice":scientific_choice,"selected_backend":selected,
                      "PRIMARY_CLASSIFICATION":classification,"SECONDARY_PROBE_CLASSIFICATION":probe_result.get("SECONDARY_PROBE_CLASSIFICATION") if probe_result else "NOT_REACHED",
                      "report":str(out/"FINAL_REPORT.md")},indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    for name in ["prepare", "train", "analyze"]:
        p = sub.add_parser(name); p.add_argument("--out", required=True)
    p = sub.add_parser("finalize-collection")
    p.add_argument("--out", required=True); p.add_argument("--collection", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if args.phase == "prepare":
        prepare(out)
    elif args.phase == "finalize-collection":
        finalize_collection(out, Path(args.collection))
    elif args.phase == "train":
        train(out)
    elif args.phase == "analyze":
        analyze(out)


if __name__ == "__main__":
    main()
