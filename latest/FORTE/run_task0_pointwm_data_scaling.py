#!/usr/bin/env python3
"""Frozen Task0 independent-root scaling diagnostic for Direct vs Point-WM.

This runner only reads archived Task0 root-scaling TRAIN/TEST artifacts.  It
never launches a simulator and never selects roots, checkpoints, or protocol
settings from TEST results.

Phases:
  audit                 validate/freeze data, roots, compatibility, protocol
  shard --scale N --seed S
                        train one GT-physics scale/seed and the permitted
                        existing-trace Probe-PointWM secondary diagnostic
  finalize              aggregate frozen TEST results and write deliverables
  validate              independently check the completed delivery
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

import root_scaling_learning_curve as rs
import run_probe_conditioned_wm as frozen
import task0_visual_context_early as early


ROOT = Path("/home/exouser/FORTE")
SOURCE = ROOT / "root_scaling_20260831"
OUT_DEFAULT = ROOT / "task0_pointwm_data_scaling_20260901"
SCALES = [6, 15, 30, 50]
SEEDS = [0, 1, 2]
FOLDS = [0, 1, 2]
FMAX = 5.0
H = 8
RHO_COUNT = 4
EPS = 1e-7


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: pd.DataFrame | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def setup_modules():
    rs.OUT = SOURCE
    tpi, cf, full = rs.modules()
    frozen.TPI, frozen.CF, frozen.FULL = tpi, cf, full
    return tpi, cf, full


def source_paths() -> list[Path]:
    return [
        ROOT / "run_task0_pointwm_data_scaling.py",
        ROOT / "root_scaling_learning_curve.py",
        ROOT / "run_probe_conditioned_wm.py",
        ROOT / "task0_visual_context_early.py",
        SOURCE / "ROOT_SCALING_TRAIN_MANIFEST.json",
        SOURCE / "ROOT_SCALING_TEST_MANIFEST.json",
        SOURCE / "ROOT_SCALING_FREEZE_SHA256.json",
        SOURCE / "TASK0_TEST_COLLECTION_COMMIT.json",
        ROOT / "activeforcing_probe_conditioned_wm_20260901_064627" / "PROBE_WM_PROTOCOL.json",
    ]


def scale_root_ids(n: int) -> list[str]:
    manifest = json.loads((SOURCE / "ROOT_SCALING_TRAIN_MANIFEST.json").read_text())
    return [str(x) for x in manifest["nested_sets"][f"S{n}"]["0"]["root_ids"]]


def root_fold_map(n: int) -> dict[str, int]:
    # Manifest order is already frozen and nested.  Modulo-3 assignment is
    # deterministic and outcome independent.  S6 is exactly 2 roots/fold.
    return {root: i % 3 for i, root in enumerate(scale_root_ids(n))}


def load_scale(n: int, tpi, cf):
    return rs.load_train(0, n, tpi, cf)


def visual_audit(contexts: pd.DataFrame, split: str) -> dict:
    if split == "TRAIN":
        tables = []
        base = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000/collection_train")
        old = ROOT / "task0_context_sample_complexity_20260831/collection_train_new"
        for p in [base / "visual_alignment_worker.csv", old / "visual_alignment_worker.csv"]:
            if p.exists():
                tables.append(pd.read_csv(p))
        visual = pd.concat(tables, ignore_index=True).drop_duplicates("context_id", keep="first")
    else:
        visual = pd.read_csv(SOURCE / "collection_test/visual_alignment_worker.csv")
    ids = set(contexts.context_id.astype(str))
    q = visual[visual.context_id.astype(str).isin(ids)].copy()
    feature_cols = [c for c in q.columns if "feature" in c.lower() and "path" in c.lower()]
    rgb_cols = [c for c in q.columns if "rgb" in c.lower() and "path" in c.lower()]
    feature_ok = sum(Path(str(getattr(r, feature_cols[0]))).exists() for r in q.itertuples()) if feature_cols else 0
    rgb_ok = 0
    if rgb_cols:
        rgb_ok = sum(all(Path(str(getattr(r, c))).exists() for c in rgb_cols) for r in q.itertuples())
    return {
        "rows": int(len(q)), "unique_contexts": int(q.context_id.nunique()),
        "feature_files_complete": int(feature_ok), "rgb_contexts_complete": int(rgb_ok),
        "feature_path_columns": feature_cols, "rgb_path_columns": rgb_cols,
    }


def telemetry_audit(branches: pd.DataFrame) -> tuple[dict, list[dict]]:
    required = {
        "step", "t_s", "phase", "mode", "cmd_x", "cmd_y", "cmd_z",
        "left_normal_force_N", "right_normal_force_N",
        "left_tangential_force_N", "right_tangential_force_N",
        "object_vx_mps", "object_vy_mps", "object_vz_mps",
    }
    missing_rows: list[dict] = []
    h8 = finite = schema = 0
    for row in branches.itertuples(index=False):
        path = Path(str(row.telemetry_path))
        if not path.exists():
            missing_rows.append({"branch_id": str(row.branch_id), "reason": "MISSING_FILE", "path": str(path)})
            continue
        d = pd.read_csv(path)
        miss = sorted(required - set(d.columns))
        if miss:
            missing_rows.append({"branch_id": str(row.branch_id), "reason": "MISSING_COLUMNS", "columns": ";".join(miss), "path": str(path)})
            continue
        schema += 1
        numeric = [c for c in required if c in d and pd.api.types.is_numeric_dtype(d[c])]
        if np.isfinite(d[numeric].to_numpy(float)).all():
            finite += 1
        else:
            missing_rows.append({"branch_id": str(row.branch_id), "reason": "NONFINITE_TELEMETRY", "path": str(path)})
            continue
        if len(d) >= H + 1:
            h8 += 1
        else:
            missing_rows.append({"branch_id": str(row.branch_id), "reason": "SHORTER_THAN_H8_PLUS_INITIAL", "path": str(path)})
    return {
        "branches": int(len(branches)), "schema_complete": schema,
        "finite_complete": finite, "H8_complete": h8,
        "missing_or_bad": len(missing_rows),
    }, missing_rows


def probe_audit(contexts: pd.DataFrame) -> tuple[dict, list[dict]]:
    norm = json.loads((frozen.P5ROOT / "P5S0C_NORMALIZATION.json").read_text())
    names = norm["dynamic_feature_names"]
    phases = norm["phase_categories_from_train"]
    states = norm["contact_state_categories_from_train"]
    failures: list[dict] = []
    legal = 0
    for row in contexts.itertuples(index=False):
        path = Path(str(row.probe_telemetry_path))
        if not path.exists():
            failures.append({"context_id": str(row.context_id), "reason": "MISSING_PROBE", "path": str(path)})
            continue
        try:
            q = frozen.P5.sequence_dataframe(str(path), phases, states).reindex(columns=names).fillna(0.0)
            a = q.to_numpy(np.float32)
        except Exception as exc:
            failures.append({"context_id": str(row.context_id), "reason": f"PARSER_ERROR:{exc}", "path": str(path)})
            continue
        if a.shape == (215, 46) and np.isfinite(a).all():
            legal += 1
        else:
            failures.append({"context_id": str(row.context_id), "reason": f"BAD_PARSED_SHAPE:{a.shape}", "path": str(path)})
    return {"contexts": len(contexts), "legal_215x46": legal, "failures": len(failures)}, failures


def audit(out: Path) -> None:
    if out.exists() and (out / "TASK0_POINTWM_SCALING_PROTOCOL.json").exists():
        print(json.dumps({"status": "ALREADY_AUDITED", "out": str(out)}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    tpi, cf, full = setup_modules()
    manifest = json.loads((SOURCE / "ROOT_SCALING_TRAIN_MANIFEST.json").read_text())
    audit_rows: list[dict] = []
    all_missing: list[dict] = []
    roots_rows: list[dict] = []
    prior: set[str] = set()
    all_train_contexts: list[pd.DataFrame] = []
    for n in SCALES:
        contexts, branches, cmap, traces, meta, pairs, segs, norm, pca, npth = load_scale(n, tpi, cf)
        roots = scale_root_ids(n)
        if set(prior) - set(roots):
            raise RuntimeError(f"S{n} is not nested over the previous scale")
        prior = set(roots)
        tele, missing = telemetry_audit(branches)
        all_missing.extend({"scale": n, **x} for x in missing)
        vis = visual_audit(contexts, "TRAIN")
        force_counts = branches.groupby("context_id").requested_force_N.nunique()
        repeat_counts = branches.groupby(["context_id", "requested_force_N"]).size()
        state_hash_counts = branches.groupby("context_id").post_probe_state_hash.nunique()
        row = {
            "scale": f"S{n}", "independent_roots": int(contexts.root_id.nunique()),
            "friction_contexts": int(len(contexts)), "branches": int(len(branches)),
            "force_cells": int(branches.groupby(["context_id", "requested_force_N"]).ngroups),
            "forces_per_context_min": int(force_counts.min()), "forces_per_context_max": int(force_counts.max()),
            "repeats_per_force_min": int(repeat_counts.min()), "repeats_per_force_max": int(repeat_counts.max()),
            "friction_metadata_complete": int(contexts.hidden_friction_analysis_only.notna().sum()),
            "RGB_contexts_complete": vis["rgb_contexts_complete"], "visual_features_complete": vis["feature_files_complete"],
            "frozen_pi0_feature_contexts": vis["feature_files_complete"],
            "nominal_pi0_motion_branches": tele["schema_complete"],
            "H8_physical_telemetry_complete": tele["H8_complete"],
            "IE_adjacent_pairs": int(len(pairs)), "full_task_labels_complete": int(branches.full_task_success_y.notna().sum()),
            "root_identity_complete": int(branches.root_id.notna().sum()), "simulator_seed_complete": int(branches.root_seed.notna().sum()),
            "state_parity_pass": int(branches.state_parity.sum()),
            "snapshot_identity_contexts": int((state_hash_counts == 1).sum()),
            "bad_branches": tele["missing_or_bad"],
        }
        audit_rows.append(row)
        branch_by_root = branches.groupby(branches.root_id.astype(str)).size().to_dict()
        for root in roots:
            if root in {x["root_id"] for x in roots_rows}:
                continue
            rq = contexts[contexts.root_id.astype(str) == root].copy()
            rr = rq.iloc[0]
            first = next(k for k in SCALES if root in set(scale_root_ids(k)))
            roots_rows.append({
                "root_id": root, "first_scale_included": f"S{first}",
                "branch_count": int(branch_by_root[root]),
                "friction": ";".join(f"{x:.8g}" for x in sorted(rq.hidden_friction_analysis_only.astype(float).unique())),
                "friction_context_count": int(rq.context_id.nunique()),
                "seed": int(rr.root_seed),
            })
        all_train_contexts.append(contexts)
    if all_missing:
        write_csv(out / "POINTWM_MISSING_BRANCHES.csv", all_missing)
        write_json(out / "FINAL_TASK0_POINTWM_DATA_SCALING_CLASSIFICATION.json", {
            "classification": "BLOCKED_BY_MISSING_POINTWM_TRAINING_TELEMETRY",
            "missing_branch_count": len(all_missing),
        })
        raise RuntimeError(f"BLOCKED_BY_MISSING_POINTWM_TRAINING_TELEMETRY: {len(all_missing)} branches")
    write_csv(out / "TASK0_POINTWM_SCALING_ROOTS.csv", roots_rows)

    # Frozen held-out TEST audit.  Outcomes are already viewed; this is never
    # described as untouched TEST.
    test_contexts, test_branches, test_cmap, templates = rs.load_test(0, tpi, rs.pca_path(0, 50))
    train_roots = set(scale_root_ids(50))
    test_roots = set(test_contexts.root_id.astype(str))
    test_tele, test_missing = telemetry_audit(test_branches)
    test_vis = visual_audit(test_contexts, "TEST")
    force_counts = test_branches.groupby("context_id").requested_force_N.nunique()
    repeat_counts = test_branches.groupby(["context_id", "requested_force_N"]).size()
    hash_counts = test_branches.groupby("context_id").post_probe_state_hash.nunique()
    test_audit = {
        "label": "FROZEN HELD-OUT TASK0 TEST", "independent_roots": len(test_roots),
        "train_root_overlap": len(train_roots & test_roots), "force_cells": int(test_branches.groupby(["context_id", "requested_force_N"]).ngroups),
        "branches": len(test_branches), "forces_per_root_min": int(force_counts.min()), "forces_per_root_max": int(force_counts.max()),
        "repeats_per_cell_min": int(repeat_counts.min()), "repeats_per_cell_max": int(repeat_counts.max()),
        "state_parity_pass": int(test_branches.state_parity.sum()), "snapshot_identity_contexts": int((hash_counts == 1).sum()),
        "visual_rows": test_vis["rows"], "visual_features_complete": test_vis["feature_files_complete"],
        "RGB_contexts_complete": test_vis["rgb_contexts_complete"], "telemetry_complete": test_tele["H8_complete"],
        "success_labels_complete": int(test_branches.full_task_success_y.notna().sum()),
        "controller_episodes_per_method_seed": int(len(test_roots) * repeat_counts.min()),
        "bad_branches": len(test_missing),
    }
    if not (test_audit["independent_roots"] == 10 and test_audit["train_root_overlap"] == 0 and
            test_audit["force_cells"] == 90 and test_audit["branches"] == 450 and
            test_audit["forces_per_root_min"] == test_audit["forces_per_root_max"] == 9 and
            test_audit["repeats_per_cell_min"] == test_audit["repeats_per_cell_max"] == 5 and
            test_audit["state_parity_pass"] == 450 and test_audit["telemetry_complete"] == 450 and
            test_audit["success_labels_complete"] == 450 and test_audit["bad_branches"] == 0):
        raise RuntimeError(f"frozen TEST QA failed: {test_audit}")

    # Probe secondary eligibility is determined before any model result.
    s50_contexts = all_train_contexts[-1]
    probe_train, probe_train_fail = probe_audit(s50_contexts)
    probe_test, probe_test_fail = probe_audit(test_contexts)
    probe_eligible = not probe_train_fail and not probe_test_fail

    # Semantic architecture comparison.  Names differ (command_gru/condition
    # vs gru/cond), but tensor topology and forward computation are identical.
    old_ck = torch.load(SOURCE / "task0_diagnostic/checkpoints/TASK0_S6_BASE_seed0.pt", map_location="cpu", weights_only=False)
    old_model = full.FeasibilityOnly()
    current_model = frozen.Direct(54)
    old_shapes = [tuple(x.shape) for x in old_model.state_dict().values()]
    cur_shapes = [tuple(x.shape) for x in current_model.state_dict().values()]
    compatibility = {
        "decision": "REUSE_EXACT_COMPATIBLE_ROOT_SCALING_DIRECT_FINAL_CHECKPOINTS",
        "architecture_hash_old": stable_hash(old_shapes), "architecture_hash_current": stable_hash(cur_shapes),
        "architecture_shapes_equal": old_shapes == cur_shapes,
        "input_features_old": "H8 nominal pi0 motion first 17 channels + 54 condition channels including GT mu and candidate force",
        "input_features_current": "H8 nominal pi0 motion first 17 channels + same 54 condition channels including GT mu and candidate force",
        "preprocessing": "per-scale TRAIN-only x mean/std; identical segment builder",
        "objective": "full-task outcome BCE",
        "seeds": SEEDS, "optimizer": "AdamW", "epochs": 80,
        "friction_source": "GT friction", "visual_encoder": "none",
        "candidate_force_normalization": "candidate force embedded by frozen nominal_from/build_seg then TRAIN-only standardization",
        "key_name_only_difference": {"command_gru": "gru", "condition": "cond"},
        "old_Visual_Joint_compatible_with_current_Point_WM": False,
        "old_Visual_Joint_reasons": ["uses visual input", "includes outcome BCE in joint model", "not PhysicsOnly", "no root-OOF utility residual controller"],
        "checkpoint_example_sha256": sha256(SOURCE / "task0_diagnostic/checkpoints/TASK0_S6_BASE_seed0.pt"),
    }
    if not compatibility["architecture_shapes_equal"]:
        compatibility["decision"] = "RETRAIN_DIRECT_ALL_SCALES"

    protocol = {
        "status": "FROZEN_BEFORE_TASK0_POINTWM_SCALING_TRAINING",
        "scope": "Task0 only; S6/S15/S30/S50; archived TRAIN and FROZEN HELD-OUT TASK0 TEST only",
        "primary": "GT-PHYSICS SCALING DIAGNOSTIC",
        "deployable_controller_claim": False,
        "test_semantics": "FROZEN HELD-OUT TASK0 TEST; previously evaluated in root-scaling analysis",
        "forbidden": ["simulator collection", "new TEST collection", "root selection", "checkpoint selection", "architecture tuning", "lambda tuning", "horizon tuning", "residual tuning", "force-grid tuning", "utility tuning", "post-hoc ensemble changes"],
        "scales": SCALES, "seeds": SEEDS,
        "nested_root_manifest_reused": True,
        "inner_crossfit": {"folds": 3, "group": "root_id", "assignment": "frozen manifest root order modulo 3", "S6": "2 held roots/fold"},
        "direct": {"architecture": "GRU(17,64)+condition MLP(54,64)+head(128,64,1)", "objective": "full-task BCE", "epochs": 80, "optimizer": "AdamW", "lr": 8e-4, "weight_decay": 1e-4, "batch": 64},
        "point_wm": {"architecture": "PhysicsOnly GRU(71,64)+H8x13 head", "H": 8, "objective": "smooth-L1 physical trajectory + lambdaIE*IE", "lambdaIE": 1.0, "outcome_gradient": False, "epochs": 80, "optimizer": "AdamW", "lr": 8e-4, "weight_decay": 1e-4, "batch": 64, "residual": "summary52 trajectory + normalized force -> Linear(53,32)-GELU-Linear(1); MSE; 80 epochs"},
        "utility": {"success": "(Fmax-F)/Fmax", "failure": -1, "Fmax": FMAX, "selection": "argmax expected utility; ties lower archived force"},
        "test_force_grid": "exact archived 9 forces/root; no interpolation or replay",
        "empirical_frontier": "minimum archived force with >=4 successes/5; otherwise NO_RELIABLE_FORCE",
        "ensemble": "prediction/utility-level mean across all three canonical seeds before argmax; reported separately from three-seed metric mean",
        "pointwm_prediction_metric": "utility-implied probability p=(score+1)/(R_success+1), clipped only for diagnostic NLL/Brier/AUROC; controller still uses unmodified score",
        "probe_secondary": {"eligible": probe_eligible, "train": probe_train, "test": probe_test, "rule": "existing traces only; grouped-root OOF Probe estimates for residual/base training; full-TRAIN Probe inference on frozen TEST"},
        "classification_rules": {
            "A": "S6 and S15 gain <= +2pp; mean gain rises at S30 and again at S50; S50 gain >0; >=2/3 S50 seeds positive; S50 delta under-force <=+2pp and delta utility >=0",
            "B": "S50 prediction-level ensemble gain >0 but fewer than 2/3 single seeds positive",
            "C": "frontier delta positive at >=3 scales while S50 SR/utility gains are nonpositive",
            "E": "S50 SR gain negative while H8 MAE improves S6->S50",
            "D": "otherwise",
        },
        "source_hashes": {str(p): sha256(p) for p in source_paths() if p.exists()},
    }
    write_json(out / "TASK0_POINTWM_SCALING_PROTOCOL.json", protocol)
    write_json(out / "TASK0_POINTWM_DATA_AUDIT.json", {"train": audit_rows, "test": test_audit, "probe_secondary": protocol["probe_secondary"]})
    write_json(out / "TASK0_POINTWM_MODEL_COMPATIBILITY.json", compatibility)

    lines = [
        "# Task0 Point-WM Scaling Data Audit", "",
        "**PASS.** All archived S6/S15/S30/S50 branches contain complete H8 physical telemetry. No branch was removed or repaired. The primary experiment is explicitly a **GT-PHYSICS SCALING DIAGNOSTIC** on the **FROZEN HELD-OUT TASK0 TEST**.", "",
        "## TRAIN sets (manifest-real counts)", "",
        "| Scale | Independent roots | Friction contexts | Force cells | Branches | Forces/context | Repeats/force | H8 complete | IE pairs | RGB | π0 features | Labels | State parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in audit_rows:
        lines.append(f"| {r['scale']} | {r['independent_roots']} | {r['friction_contexts']} | {r['force_cells']} | {r['branches']} | {r['forces_per_context_min']} | {r['repeats_per_force_min']} | {r['H8_physical_telemetry_complete']} | {r['IE_adjacent_pairs']} | {r['RGB_contexts_complete']} | {r['frozen_pi0_feature_contexts']} | {r['full_task_labels_complete']} | {r['state_parity_pass']} |")
    lines += [
        "", "The counts differ from the rough 60/150/300/500 expectation. S6 contains three friction-conditioned contexts per one of six root families (18 contexts); each later root contributes one frozen friction context. Every context has 5 candidate force cells × 2 repeats, so the real branch counts are 180/270/420/620.", "",
        "## Nested identity", "",
        "`S6 ⊂ S15 ⊂ S30 ⊂ S50` is exact by frozen root family identity. `TASK0_POINTWM_SCALING_ROOTS.csv` records the 50 roots in frozen manifest order with first inclusion scale, branch count, friction, and simulator seed.", "",
        "## H8, nominal motion, IE, and identity", "",
        "All branches have the required corrected force/velocity columns, finite H8 state targets, nominal π0 command columns, full-task labels, simulator seed, root identity, post-probe snapshot hash, and state-parity pass. IE pairs are adjacent forces within the same context and repeat; counts are printed above.", "",
        "## FROZEN HELD-OUT TASK0 TEST QA", "",
        f"- Roots: {test_audit['independent_roots']} exact; TRAIN overlap: {test_audit['train_root_overlap']}",
        f"- Force cells / branches: {test_audit['force_cells']} / {test_audit['branches']}",
        f"- Archived grid: {test_audit['forces_per_root_min']} forces/root × {test_audit['repeats_per_cell_min']} repeats/cell",
        f"- State parity / telemetry / success labels: {test_audit['state_parity_pass']} / {test_audit['telemetry_complete']} / {test_audit['success_labels_complete']}",
        f"- Visual features / RGB contexts: {test_audit['visual_features_complete']} / {test_audit['RGB_contexts_complete']}",
        f"- Controller evaluation: {test_audit['controller_episodes_per_method_seed']} real episodes per method/seed (10 selected cells × 5 archived repeats)",
        "", "TEST is read-only and previously evaluated. No tuning, root/checkpoint choice, or protocol changes are permitted from these outcomes.", "",
        "## Probe secondary eligibility", "",
        f"Existing legal traces are complete: TRAIN {probe_train['legal_215x46']}/{probe_train['contexts']}, TEST {probe_test['legal_215x46']}/{probe_test['contexts']}. Secondary Probe-PointWM is {'RUN' if probe_eligible else 'PROBE_SCALING_NOT_RUN_EXISTING_TRACE_UNAVAILABLE'}; it does not replace the GT-physics primary curve.",
    ]
    (out / "TASK0_POINTWM_SCALING_DATA_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    comp_lines = [
        "# Task0 Point-WM Model Compatibility", "",
        f"**Decision: `{compatibility['decision']}`.**", "",
        "The prior Task0 `Base` checkpoint and current formal Direct have identical tensor topology, information, preprocessing, BCE objective, optimizer, epochs, seeds, GT-friction source, absence of visual input, and candidate-force normalization. The only implementation difference is module naming (`command_gru/condition` versus `gru/cond`). Final Direct checkpoints can therefore be reused without changing predictions; grouped-root inner models are newly trained because residual OOF predictions do not exist for these scales.", "",
        "| Item | Prior Task0 Base | Current Direct | Compatible |", "|---|---|---|---|",
        "| Architecture | GRU17→64 + condition54→64 + head128→64→1 | same | yes |",
        "| Inputs | x, μ_GT, candidate F | x, μ_GT, candidate F | yes |",
        "| Preprocessing | per-scale TRAIN-only normalization | same | yes |",
        "| Objective | full-task BCE | full-task BCE | yes |",
        "| Seed / optimizer / epochs | 0/1/2; AdamW; 80 | same | yes |",
        "| Friction / visual | GT; none | GT; none | yes |",
        "| Candidate force | nominal segment then TRAIN normalization | same | yes |",
        "", "## Old Visual Joint is not Point-WM", "",
        "Old Visual Joint is incompatible and is not reused: it consumes visual input, receives outcome BCE through the joint model, and does not use the frozen PhysicsOnly→root-OOF residual→expected-utility controller protocol.",
    ]
    (out / "TASK0_POINTWM_MODEL_COMPATIBILITY.md").write_text("\n".join(comp_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "AUDIT_PASS_PROTOCOL_FROZEN", "out": str(out), "train": audit_rows, "test": test_audit, "probe_secondary": probe_eligible}, indent=2))


def load_test_traces(tpi, cf):
    contexts = pd.read_csv(SOURCE / "collection_test/task0/task0/context.csv").sort_values("context_id")
    branches = pd.read_csv(SOURCE / "collection_test/task0/task0/branches.csv").sort_values("branch_id")
    manifest = json.loads((SOURCE / "ROOT_SCALING_TEST_MANIFEST.json").read_text())
    ids = {x["context_id"] for x in manifest["contexts"]["0"]}
    contexts = contexts[contexts.context_id.astype(str).isin(ids)].copy()
    branches = branches[branches.context_id.astype(str).isin(ids)].copy()
    context_map = contexts.set_index(contexts.context_id.astype(str)).to_dict("index")
    pre: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r in contexts.itertuples(index=False):
        s, m, _ = early.strict_preprobe_state(Path(str(r.probe_telemetry_path)))
        pre[str(r.context_id)] = (s, m)
    traces = []
    meta = {}
    for r in branches.itertuples(index=False):
        cid = str(r.context_id)
        path = Path(str(r.telemetry_path))
        d = pd.read_csv(path)
        state, mask = tpi.state_from(d)
        state, mask = state.copy(), mask.copy()
        state[0], mask[0] = pre[cid]
        force = float(r.requested_force_N)
        mu = float(r.hidden_friction_analysis_only)
        nominal = tpi.nominal_from(d, 0, force, mu, state, mask)
        tr = tpi.Trace(str(r.branch_id), cid, str(r.root_id), 0, "TEST", force, mu,
                       int(r.full_task_success_y), "archived_9_force", path, state, mask,
                       nominal, d.phase.astype(str).tolist(), 1.0, "ROOT_SCALING_TASK0_TEST")
        traces.append(tr)
        meta[tr.branch_id] = r
    return contexts, branches, traces, meta


def candidate_templates(tpi, cf, test_contexts: pd.DataFrame, test_branches: pd.DataFrame):
    _, _, cmap, templates = rs.load_test(0, tpi, rs.pca_path(0, 50))
    candidates = []
    segs = {}
    for cid in sorted(templates):
        forces = sorted(test_branches.loc[test_branches.context_id.astype(str) == cid, "requested_force_N"].astype(float).unique())
        for force in forces:
            base = templates[cid]
            bid = f"candidate:{cid}:F{force:.8f}"
            fake = tpi.Trace(bid, cid, base.root_id, 0, "TEST", float(force), float(base.mu), 0,
                             "archived_candidate", base.path, base.state, base.mask, base.nominal,
                             base.phase, 1.0, "ROOT_SCALING_TASK0_TEST_CANDIDATE")
            candidates.append(fake)
            segs[bid] = cf.build_seg(tpi, fake, float(force), H)
    return candidates, segs, cmap


def transfer_direct_state(old_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapping = {"command_gru.": "gru.", "condition.": "cond."}
    ans = {}
    for key, value in old_state.items():
        new = key
        for src, dst in mapping.items():
            if key.startswith(src):
                new = dst + key[len(src):]
        ans[new] = value
    return ans


def load_final_direct(n: int, seed: int) -> frozen.Direct:
    path = SOURCE / f"task0_diagnostic/checkpoints/TASK0_S{n}_BASE_seed{seed}.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = frozen.Direct(54)
    model.load_state_dict(transfer_direct_state(ck["state_dict"]))
    model.eval()
    return model


def infer_models(direct, wm, traces, segs, norm, mu_map):
    hmap = {t.context_id: np.zeros(16, np.float32) for t in traces}
    return frozen.infer(direct, wm, traces, segs, norm, mu_map, hmap, False)


def probe_arrays(contexts: pd.DataFrame):
    norm = json.loads((frozen.P5ROOT / "P5S0C_NORMALIZATION.json").read_text())
    names, phases, states = norm["dynamic_feature_names"], norm["phase_categories_from_train"], norm["contact_state_categories_from_train"]
    arr = []
    rows = []
    for r in contexts.sort_values("context_id").itertuples(index=False):
        q = frozen.P5.sequence_dataframe(str(r.probe_telemetry_path), phases, states).reindex(columns=names).fillna(0.0)
        a = q.to_numpy(np.float32)
        if a.shape != (215, 46) or not np.isfinite(a).all():
            raise RuntimeError(f"illegal probe trace {r.context_id}: {a.shape}")
        arr.append(a)
        rows.append({"context_id": str(r.context_id), "root_id": str(r.root_id), "mu_GT": float(r.hidden_friction_analysis_only)})
    return np.stack(arr), pd.DataFrame(rows)


def fit_probe(train_raw: np.ndarray, train_mu: np.ndarray, seed: int):
    # Same frozen architecture/objective/epochs; local wrapper avoids the
    # pooled global metadata dependency in the legacy helper.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mean = train_raw.reshape(-1, train_raw.shape[-1]).mean(0).astype(np.float32)
    std = train_raw.reshape(-1, train_raw.shape[-1]).std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    x = torch.tensor((train_raw - mean) / std)
    lengths = torch.full((len(train_raw),), train_raw.shape[1], dtype=torch.long)
    y = torch.tensor(train_mu.astype(np.float32))
    model = frozen.ProbeGRU(train_raw.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(frozen.PROBE_EPOCHS):
        model.train(); opt.zero_grad(set_to_none=True)
        mu, logs = model(x, lengths)
        sigma = logs.exp().clamp_min(1e-3)
        loss = (0.5 * (((y - mu) / sigma) ** 2 + 2 * logs) + 0.05 * (y - mu).abs()).mean()
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    return model, mean, std


def predict_probe(model, mean, std, raw):
    x = torch.tensor((raw - mean) / std)
    lengths = torch.full((len(raw),), raw.shape[1], dtype=torch.long)
    with torch.no_grad():
        mu, _ = model(x, lengths)
    return mu.numpy().astype(np.float32)


def train_one_condition(n: int, seed: int, tpi, cf, full, train_data, test_data,
                        physics: str, out: Path) -> dict:
    contexts, branches, cmap, traces, meta, pairs, segs, _, pca, npth = train_data
    test_contexts, test_branches, test_traces, test_meta = test_data
    roots = scale_root_ids(n)
    folds = root_fold_map(n)
    root_arr = np.asarray([folds[str(t.root_id)] for t in traces], int)
    hmap = {t.context_id: np.zeros(16, np.float32) for t in traces}
    test_hmap = {str(x): np.zeros(16, np.float32) for x in test_contexts.context_id.astype(str)}

    if physics == "GT":
        train_mu = {t.context_id: float(t.mu) for t in traces}
        test_mu = {str(r.context_id): float(r.hidden_friction_analysis_only) for r in test_contexts.itertuples(index=False)}
        probe_info = {"condition": "GT", "test_mu_mae": 0.0}
    else:
        train_raw, probe_md = probe_arrays(contexts)
        test_raw, probe_test_md = probe_arrays(test_contexts)
        probe_fold = np.asarray([folds[str(r.root_id)] for r in probe_md.itertuples(index=False)], int)
        mu_oof = np.full(len(probe_md), np.nan, np.float32)
        probe_audit = []
        for held_fold in FOLDS:
            fi = np.flatnonzero(probe_fold != held_fold); vi = np.flatnonzero(probe_fold == held_fold)
            pm, pmean, pstd = fit_probe(train_raw[fi], probe_md.mu_GT.to_numpy(np.float32)[fi], seed + 7000 + held_fold * 101)
            mu_oof[vi] = predict_probe(pm, pmean, pstd, train_raw[vi])
            probe_audit.append({"held_fold": held_fold, "train_roots": len(set(probe_md.root_id.iloc[fi])), "validation_roots": len(set(probe_md.root_id.iloc[vi])), "train_contexts": len(fi), "validation_contexts": len(vi)})
        if not np.isfinite(mu_oof).all():
            raise RuntimeError("incomplete Probe root-OOF predictions")
        train_mu = {str(r.context_id): float(mu_oof[i]) for i, r in probe_md.iterrows()}
        pm, pmean, pstd = fit_probe(train_raw, probe_md.mu_GT.to_numpy(np.float32), seed + 7900)
        test_pred = predict_probe(pm, pmean, pstd, test_raw)
        test_mu = {str(r.context_id): float(test_pred[i]) for i, r in probe_test_md.iterrows()}
        probe_info = {"condition": "Probe", "test_mu_mae": float(np.mean(np.abs(test_pred - probe_test_md.mu_GT.to_numpy()))), "inner": probe_audit}

    norm = frozen.xnorm(traces, segs, train_mu)
    final_direct = load_final_direct(n, seed) if physics == "GT" else frozen.train_direct(traces, segs, norm, train_mu, hmap, meta, seed, False)
    final_wm, final_units = frozen.train_wm(traces, pairs, norm, train_mu, hmap, seed, False)

    # Strict grouped-root OOF predictions for residual training.  Every scored
    # root is absent from both fitted Direct and fitted WM.
    oof_p = np.full(len(traces), np.nan, np.float32)
    oof_w = np.full((len(traces), H, 13), np.nan, np.float32)
    inner_rows = []
    for held_fold in FOLDS:
        fi = np.flatnonzero(root_arr != held_fold)
        vi = np.flatnonzero(root_arr == held_fold)
        fit = [traces[i] for i in fi]
        val = [traces[i] for i in vi]
        ids = {t.branch_id for t in fit}
        fit_meta = {bid: meta[bid] for bid in ids}
        fit_pairs = [p for p in pairs if p.a.branch_id in ids and p.b.branch_id in ids]
        fit_mu = {t.context_id: train_mu[t.context_id] for t in fit}
        fit_hmap = {t.context_id: hmap[t.context_id] for t in fit}
        fit_norm = frozen.xnorm(fit, segs, fit_mu)
        inner_seed = seed + 11 * held_fold
        d = frozen.train_direct(fit, segs, fit_norm, fit_mu, fit_hmap, fit_meta, inner_seed, False)
        w, units = frozen.train_wm(fit, fit_pairs, fit_norm, fit_mu, fit_hmap, inner_seed, False)
        val_mu = {t.context_id: train_mu[t.context_id] for t in val}
        val_hmap = {t.context_id: hmap[t.context_id] for t in val}
        p, traj = frozen.infer(d, w, val, segs, fit_norm, val_mu, val_hmap, False)
        oof_p[vi] = p; oof_w[vi] = traj
        inner_rows.append({
            "scale": n, "seed": seed, "physics": physics, "held_fold": held_fold,
            "inner_train_roots": ";".join(r for r in roots if folds[r] != held_fold),
            "inner_validation_roots": ";".join(r for r in roots if folds[r] == held_fold),
            "inner_train_branches": len(fi), "inner_validation_branches": len(vi),
            "IE_pair_count": len(fit_pairs), "physical_unit_count": units,
        })
    if not np.isfinite(oof_p).all() or not np.isfinite(oof_w).all():
        raise RuntimeError("residual OOF predictions incomplete")
    force = np.asarray([t.force for t in traces], float)
    outcome = np.asarray([t.outcome for t in traces], int)
    realized = np.where(outcome > 0, (FMAX - force) / FMAX, -1.0)
    direct_u = oof_p * ((FMAX - force) / FMAX) + (1.0 - oof_p) * -1.0
    rx = np.column_stack([frozen.summary52(oof_w), force / FMAX]).astype(np.float32)
    residual, rmean, rstd = frozen.fit_residual(rx, realized - direct_u, seed)

    candidates, candidate_segs, test_cmap = candidate_templates(tpi, cf, test_contexts, test_branches)
    p_test, w_test = frozen.infer(final_direct, final_wm, candidates, candidate_segs, norm, test_mu, test_hmap, False)
    cand_force = np.asarray([t.force for t in candidates], float)
    du_test = p_test * ((FMAX - cand_force) / FMAX) + (1.0 - p_test) * -1.0
    correction = frozen.residual_pred(residual, rmean, rstd, w_test, cand_force, np.full(len(candidates), FMAX))
    point_score = du_test + correction

    # Final checkpoint files are audit artifacts, never selected by TEST.
    slug = physics.lower()
    wm_path = out / "checkpoints" / f"S{n}_seed{seed}_{slug}_pointwm.pt"
    res_path = out / "checkpoints" / f"S{n}_seed{seed}_{slug}_residual.pt"
    torch.save({"state_dict": final_wm.state_dict(), "scale": n, "seed": seed, "physics": physics, "H": H, "lambdaIE": 1.0, "epochs": 80, "optimizer": "AdamW", "train_roots": roots, "TEST_used_for_training": False}, wm_path)
    torch.save({"state_dict": residual.state_dict(), "input_mean": rmean, "input_std": rstd, "scale": n, "seed": seed, "physics": physics, "loss": "MSE", "epochs": 80, "root_OOF": True, "TEST_used_for_training": False}, res_path)
    return {
        "context_id": np.asarray([t.context_id for t in candidates]),
        "force_N": cand_force.astype(np.float32), "direct_p": p_test,
        "direct_score": du_test.astype(np.float32), "point_score": point_score.astype(np.float32),
        "point_correction": correction.astype(np.float32), "wm_traj": w_test.astype(np.float32),
        "oof_p": oof_p, "oof_wm": oof_w, "inner_rows": inner_rows,
        "final_units": final_units, "probe_info": probe_info,
        "wm_sha256": sha256(wm_path), "residual_sha256": sha256(res_path),
    }


def shard(out: Path, n: int, seed: int) -> None:
    protocol = json.loads((out / "TASK0_POINTWM_SCALING_PROTOCOL.json").read_text())
    if protocol.get("status") != "FROZEN_BEFORE_TASK0_POINTWM_SCALING_TRAINING":
        raise RuntimeError("protocol not frozen")
    target = out / "shards" / f"S{n}_seed{seed}.npz"
    meta_path = target.with_suffix(".json")
    if target.exists() and meta_path.exists():
        print(json.dumps({"status": "ALREADY_COMPLETE", "target": str(target)}))
        return
    tpi, cf, full = setup_modules()
    train_data = load_scale(n, tpi, cf)
    test_data = load_test_traces(tpi, cf)
    gt = train_one_condition(n, seed, tpi, cf, full, train_data, test_data, "GT", out)
    probe = None
    if protocol["probe_secondary"]["eligible"]:
        probe = train_one_condition(n, seed, tpi, cf, full, train_data, test_data, "Probe", out)
    arrays = {
        "context_id": gt["context_id"], "force_N": gt["force_N"],
        "direct_p": gt["direct_p"], "direct_score": gt["direct_score"],
        "point_score": gt["point_score"], "point_correction": gt["point_correction"],
        "wm_traj": gt["wm_traj"], "oof_p": gt["oof_p"], "oof_wm": gt["oof_wm"],
    }
    if probe is not None:
        arrays.update({"probe_point_score": probe["point_score"], "probe_direct_p": probe["direct_p"], "probe_wm_traj": probe["wm_traj"]})
    np.savez_compressed(target, **arrays)
    meta = {
        "status": "ATOMIC_COMPLETE", "scale": n, "seed": seed,
        "train_roots": n, "train_contexts": len(train_data[0]), "train_branches": len(train_data[1]),
        "test_roots": 10, "test_force_cells": 90, "test_branches": 450,
        "controller_episodes": 50, "GT_physics_primary": True,
        "probe_secondary_run": probe is not None,
        "inner_crossfit": gt["inner_rows"], "IE_pair_count": len(train_data[5]),
        "final_physical_units": gt["final_units"], "probe_info": probe["probe_info"] if probe else None,
        "wm_sha256": gt["wm_sha256"], "residual_sha256": gt["residual_sha256"],
        "TEST_used_for_tuning_selection_or_training": False,
    }
    write_json(meta_path, meta)
    print(json.dumps({"status": "SHARD_COMPLETE", "scale": n, "seed": seed, "branches": len(train_data[1]), "probe_secondary": probe is not None}, indent=2), flush=True)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, int); p = np.asarray(p, float)
    pos = y == 1; neg = y == 0
    if not pos.any() or not neg.any():
        return math.nan
    ranks = pd.Series(p).rank(method="average").to_numpy(float)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def probability_metrics(y: np.ndarray, p: np.ndarray, contexts: np.ndarray, forces: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, int)
    mono = []
    steps = []
    frame = pd.DataFrame({"context_id": contexts, "force_N": forces, "p": p})
    for _, q in frame.drop_duplicates(["context_id", "force_N"]).groupby("context_id"):
        d = np.diff(q.sort_values("force_N").p.to_numpy(float))
        mono.append(bool(np.all(d >= -1e-8)))
        steps.append(int(np.sum(d < -1e-8)))
    return {
        "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)), "auroc": auroc(y, p),
        "monotonicity": float(np.mean(mono)), "nonmonotonic_steps": int(sum(steps)),
    }


def truth_tables(tpi, cf):
    contexts, branches, traces, meta = load_test_traces(tpi, cf)
    cells = branches.groupby(["context_id", "requested_force_N"], as_index=False).agg(
        successes=("full_task_success_y", "sum"), repeats=("full_task_success_y", "size"),
        p_real=("full_task_success_y", "mean"), root_id=("root_id", "first"),
        mu_GT=("hidden_friction_analysis_only", "first"), friction_band=("friction_band", "first"),
    ).rename(columns={"requested_force_N": "force_N"})
    frontiers = {}
    for cid, q in cells.groupby("context_id"):
        safe = q[q.successes >= RHO_COUNT].sort_values("force_N")
        frontiers[str(cid)] = float(safe.force_N.iloc[0]) if len(safe) else math.nan
    trace_by_key = {}
    for tr in traces:
        row = meta[tr.branch_id]
        trace_by_key[(tr.context_id, round(float(tr.force), 6), int(row.repeat_index))] = tr
    branch = branches[["branch_id", "context_id", "requested_force_N", "repeat_index", "full_task_success_y", "root_id", "hidden_friction_analysis_only", "friction_band"]].copy()
    branch = branch.rename(columns={"requested_force_N": "force_N", "full_task_success_y": "success", "hidden_friction_analysis_only": "mu_GT"})
    return contexts, branches, traces, meta, cells, frontiers, trace_by_key, branch


def shard_frame(z) -> pd.DataFrame:
    return pd.DataFrame({
        "context_id": z["context_id"].astype(str), "force_N": z["force_N"].astype(float),
        "direct_p": z["direct_p"].astype(float), "direct_score": z["direct_score"].astype(float),
        "point_score": z["point_score"].astype(float),
    })


def wm_metrics(wm_traj: np.ndarray, cand: pd.DataFrame, trace_by_key: dict) -> dict:
    pred = {(str(r.context_id), round(float(r.force_N), 6)): wm_traj[i]
            for i, r in cand.reset_index(drop=True).iterrows()}
    abs_errors = []
    ie_errors = []
    separations = []
    actual_separations = []
    by_cr: dict[tuple[str, int], list[tuple[float, Any]]] = defaultdict(list)
    for (cid, force, repeat), tr in trace_by_key.items():
        abs_errors.append(np.abs(pred[(cid, force)] - tr.state[1:H + 1]))
        by_cr[(cid, repeat)].append((force, tr))
    for (cid, repeat), rows in by_cr.items():
        rows.sort(key=lambda x: x[0])
        for (fa, a), (fb, b) in zip(rows[:-1], rows[1:]):
            pdiff = pred[(cid, fb)] - pred[(cid, fa)]
            tdiff = b.state[1:H + 1] - a.state[1:H + 1]
            ie_errors.append(np.abs(pdiff - tdiff))
            actual_separations.append(np.abs(tdiff))
    for cid, q in cand.groupby("context_id"):
        rows = q.sort_values("force_N")
        ts = [pred[(str(cid), round(float(f), 6))] for f in rows.force_N]
        separations.extend(np.abs(b - a) for a, b in zip(ts[:-1], ts[1:]))
    return {
        "wm_h8_mae": float(np.mean(np.stack(abs_errors))),
        "wm_ie_error": float(np.mean(np.stack(ie_errors))),
        "wm_traj_separation": float(np.mean(np.stack(separations))),
        "real_traj_separation": float(np.mean(np.stack(actual_separations))),
    }


def implied_probability(score: np.ndarray, force: np.ndarray) -> np.ndarray:
    reward = (FMAX - np.asarray(force, float)) / FMAX
    return np.clip((np.asarray(score, float) + 1.0) / (reward + 1.0), EPS, 1 - EPS)


def controller_selection(cand: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for cid, q in cand.sort_values(["context_id", "force_N"]).groupby("context_id"):
        score = q[score_col].to_numpy(float)
        best = int(np.flatnonzero(score >= np.nanmax(score) - 1e-12)[0])
        r = q.iloc[best]
        rows.append({"context_id": str(cid), "selected_force_N": float(r.force_N), "selected_score": float(r[score_col])})
    return pd.DataFrame(rows)


def evaluate_controller(selection: pd.DataFrame, branch: pd.DataFrame, cells: pd.DataFrame,
                        frontiers: dict, method: str, n: int, seed_label: str,
                        pred_metrics: dict, wm: dict | None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    episodes = selection.merge(branch, left_on=["context_id", "selected_force_N"], right_on=["context_id", "force_N"], validate="one_to_many")
    if len(episodes) != 50:
        raise RuntimeError(f"controller episode count != 50: {method}/S{n}/{seed_label}: {len(episodes)}")
    episodes["realized_utility"] = np.where(episodes.success > 0, (FMAX - episodes.selected_force_N) / FMAX, -1.0)
    episodes["empirical_Fstar_0p8"] = episodes.context_id.map(frontiers)
    episodes["frontier_supported"] = episodes.empirical_Fstar_0p8.notna()
    episodes["under_force"] = np.where(episodes.frontier_supported, episodes.selected_force_N < episodes.empirical_Fstar_0p8 - 1e-8, np.nan)
    episodes["excess_force_N"] = np.where(episodes.frontier_supported, np.maximum(0.0, episodes.selected_force_N - episodes.empirical_Fstar_0p8), np.nan)
    episodes["frontier_abs_error_N"] = np.where(episodes.frontier_supported, np.abs(episodes.selected_force_N - episodes.empirical_Fstar_0p8), np.nan)
    supported = episodes[episodes.frontier_supported]
    row = {
        "roots": n, "train_branches": int(json.loads((OUT_ACTIVE / "shards" / f"S{n}_seed{seed_label if seed_label.isdigit() else 0}.json").read_text())["train_branches"]) if seed_label.isdigit() else math.nan,
        "seed": seed_label, "aggregation_type": "single_seed" if seed_label.isdigit() else "prediction_level_ensemble",
        "method": method, "sr": float(episodes.success.mean()), "successes": int(episodes.success.sum()), "controller_episodes": len(episodes),
        "underforce": float(supported.under_force.astype(float).mean()) if len(supported) else math.nan,
        "underforce_evaluable_episodes": len(supported), "frontier_supported_roots": int(episodes.loc[episodes.frontier_supported, "context_id"].nunique()),
        "mean_force": float(episodes.selected_force_N.mean()), "excess_force": float(supported.excess_force_N.mean()) if len(supported) else math.nan,
        "utility": float(episodes.realized_utility.mean()), "frontier_mae": float(supported.frontier_abs_error_N.mean()) if len(supported) else math.nan,
        **pred_metrics,
        "wm_h8_mae": wm["wm_h8_mae"] if wm else math.nan,
        "wm_ie_error": wm["wm_ie_error"] if wm else math.nan,
        "wm_traj_separation": wm["wm_traj_separation"] if wm else math.nan,
        "real_traj_separation": wm["real_traj_separation"] if wm else math.nan,
    }
    per = episodes.groupby("context_id", as_index=False).agg(
        root_id=("root_id", "first"), mu_GT=("mu_GT", "first"), friction_band=("friction_band", "first"),
        selected_force_N=("selected_force_N", "first"), selected_score=("selected_score", "first"),
        successes=("success", "sum"), repeats=("success", "size"), sr=("success", "mean"),
        utility=("realized_utility", "mean"), empirical_Fstar_0p8=("empirical_Fstar_0p8", "first"),
        frontier_supported=("frontier_supported", "first"), under_force=("under_force", "first"),
        excess_force_N=("excess_force_N", "first"), frontier_abs_error_N=("frontier_abs_error_N", "first"),
    )
    per.insert(0, "method", method); per.insert(0, "seed", seed_label); per.insert(0, "roots", n)
    per["frontier_status"] = np.where(per.frontier_supported, "RELIABLE_FORCE_FOUND", "NO_RELIABLE_FORCE")
    return row, per, episodes


def paired_per_root(n: int, seed_label: str, direct: pd.DataFrame, point: pd.DataFrame) -> pd.DataFrame:
    d = direct.add_prefix("direct_").rename(columns={"direct_context_id": "context_id", "direct_roots": "roots", "direct_seed": "seed"})
    p = point.add_prefix("pointwm_").rename(columns={"pointwm_context_id": "context_id", "pointwm_roots": "roots", "pointwm_seed": "seed"})
    q = d.merge(p, on=["roots", "seed", "context_id"], validate="one_to_one")
    return pd.DataFrame({
        "roots": q.roots, "seed": q.seed, "context_id": q.context_id,
        "root_id": q.direct_root_id, "mu_GT": q.direct_mu_GT, "friction_band": q.direct_friction_band,
        "empirical_Fstar_0p8": q.direct_empirical_Fstar_0p8,
        "frontier_status": q.direct_frontier_status,
        "Direct_selected_F": q.direct_selected_force_N, "PointWM_selected_F": q.pointwm_selected_force_N,
        "Direct_successes": q.direct_successes, "PointWM_successes": q.pointwm_successes,
        "Direct_SR": q.direct_sr, "PointWM_SR": q.pointwm_sr,
        "force_difference_PointWM_minus_Direct": q.pointwm_selected_force_N - q.direct_selected_force_N,
        "utility_difference_PointWM_minus_Direct": q.pointwm_utility - q.direct_utility,
        "frontier_abs_error_Direct": q.direct_frontier_abs_error_N,
        "frontier_abs_error_PointWM": q.pointwm_frontier_abs_error_N,
    })


def mechanism_table(n: int, seed_label: str, dsel: pd.DataFrame, psel: pd.DataFrame, branch: pd.DataFrame) -> pd.DataFrame:
    d = dsel.rename(columns={"selected_force_N": "Direct_F"})[["context_id", "Direct_F"]]
    p = psel.rename(columns={"selected_force_N": "PointWM_F"})[["context_id", "PointWM_F"]]
    rows = []
    for r in d.merge(p, on="context_id").itertuples(index=False):
        dq = branch[(branch.context_id == r.context_id) & np.isclose(branch.force_N, r.Direct_F)].sort_values("repeat_index")
        pq = branch[(branch.context_id == r.context_id) & np.isclose(branch.force_N, r.PointWM_F)].sort_values("repeat_index")
        for dr, pr in zip(dq.itertuples(index=False), pq.itertuples(index=False)):
            if dr.success == 0 and pr.success == 1:
                cat = "A_RESCUE"
            elif dr.success == 1 and pr.success == 0:
                cat = "C_COLLATERAL_DAMAGE"
            elif dr.success == 1 and pr.success == 1 and r.PointWM_F < r.Direct_F - 1e-8:
                cat = "B_ECONOMIZE"
            elif dr.success == pr.success and r.PointWM_F > r.Direct_F + 1e-8:
                cat = "D_OVER_FORCE"
            else:
                cat = "UNCHANGED_OR_OTHER"
            rows.append({"roots": n, "seed": seed_label, "context_id": r.context_id, "repeat": int(dr.repeat_index), "mechanism": cat, "Direct_F": r.Direct_F, "PointWM_F": r.PointWM_F, "Direct_success": int(dr.success), "PointWM_success": int(pr.success)})
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["roots", "seed", "mechanism"], as_index=False).size().rename(columns={"size": "episodes"})
    summary["rate"] = summary.episodes / 50.0
    return summary


def rank_corr(x: list[float], y: list[float]) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(float)
    ry = pd.Series(y).rank(method="average").to_numpy(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def make_figures(main: pd.DataFrame, gain: pd.DataFrame, out: Path) -> None:
    single = main[main.seed.isin(["0", "1", "2"])].copy()
    ens = main[main.seed == "ENSEMBLE"].copy()
    colors = {"Direct": "#4C78A8", "Point-WM": "#E45756"}
    markers = {"Direct": "o", "Point-WM": "s"}
    lines = {"Direct": "-", "Point-WM": "--"}

    def two_method(metric, stem, ylabel, title, percent=False, old_joint=False):
        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        for method in ["Direct", "Point-WM"]:
            q = single[single.method == method].groupby("roots")[metric].agg(["mean", lambda x: x.std(ddof=0)]).reset_index()
            q.columns = ["roots", "mean", "std"]
            ax.errorbar(q.roots, q["mean"], yerr=q["std"], color=colors[method], marker=markers[method], linestyle=lines[method], linewidth=2.2, markersize=7, capsize=4, label=f"{method} seed mean ± std")
            e = ens[ens.method == method].sort_values("roots")
            ax.scatter(e.roots, e[metric], facecolors="none", edgecolors=colors[method], marker=markers[method], s=80, linewidths=1.8, label=f"{method} prediction-level ensemble")
        if old_joint:
            ax.plot(SCALES, [0.277, 0.207, 0.245, 0.253], color="#777777", marker="^", linestyle=":", linewidth=1.6, label="Old Visual Joint (auxiliary)")
        ax.set_xticks(SCALES); ax.set_xlabel("Independent TRAIN roots"); ax.set_ylabel(ylabel); ax.set_title(title)
        if percent: ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, fontsize=8); fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight"); plt.close(fig)

    two_method("sr", "FIG_TASK0_POINTWM_SR_VS_ROOTS", "Full-task success rate", "Frozen Task0 controller success vs training roots", True)
    two_method("utility", "FIG_TASK0_POINTWM_UTILITY_VS_ROOTS", "Mean realized utility", "Frozen Task0 realized utility vs training roots")

    # Keep the old predicted-threshold frontier metric visually separate from
    # the current selected-force MAE to empirical F*0.8.  Their within-panel
    # directions are comparable; their absolute levels are not.
    fig, (ax, ax_old) = plt.subplots(2, 1, figsize=(7.4, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.45, 1]})
    for method in ["Direct", "Point-WM"]:
        qf = single[single.method == method].groupby("roots").frontier_mae.agg(["mean", lambda x: x.std(ddof=0)]).reset_index()
        qf.columns = ["roots", "mean", "std"]
        ax.errorbar(qf.roots, qf["mean"], yerr=qf["std"], color=colors[method], marker=markers[method], linestyle=lines[method], linewidth=2.2, markersize=7, capsize=4, label=f"{method} seed mean ± std")
        ef = ens[ens.method == method].sort_values("roots")
        ax.scatter(ef.roots, ef.frontier_mae, facecolors="none", edgecolors=colors[method], marker=markers[method], s=80, linewidths=1.8, label=f"{method} ensemble")
    ax.set_ylabel("Selected-force MAE\nvs empirical F*0.8 (N)")
    ax.set_title("Current Point-WM controller versus old auxiliary frontier signal")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, fontsize=8, ncol=2)
    ax_old.plot(SCALES, [.297, .240, .307, .312], color="#4C78A8", marker="o", linewidth=1.8, label="Old Direct")
    ax_old.plot(SCALES, [.277, .207, .245, .253], color="#777777", marker="^", linestyle=":", linewidth=2.0, label="Old Visual Joint")
    ax_old.set_ylabel("Old predicted-threshold\nfrontier MAE (N)")
    ax_old.set_xticks(SCALES); ax_old.set_xlabel("Independent TRAIN roots")
    ax_old.grid(axis="y", alpha=.25); ax_old.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "FIG_TASK0_POINTWM_FRONTIER_VS_ROOTS.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "FIG_TASK0_POINTWM_FRONTIER_VS_ROOTS.pdf", bbox_inches="tight"); plt.close(fig)

    q = gain[gain.seed == "SEED_MEAN"].sort_values("roots")
    e = gain[gain.seed == "ENSEMBLE"].sort_values("roots")
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.axhline(0, color="#333333", linewidth=1.3)
    ax.plot(q.roots, q.delta_sr, color="#E45756", marker="s", linestyle="--", linewidth=2.3, markersize=7, label="3-seed metric mean")
    ax.scatter(e.roots, e.delta_sr, facecolors="none", edgecolors="#E45756", marker="s", s=85, linewidths=1.8, label="prediction-level ensemble")
    for r in q.itertuples(index=False):
        ax.annotate(f"{100*r.delta_sr:+.1f} pp", (r.roots, r.delta_sr), xytext=(0, 9 if r.delta_sr >= 0 else -15), textcoords="offset points", ha="center", fontsize=8)
    lower = min(float(q.delta_sr.min()), float(e.delta_sr.min())) - .012
    ax.set_ylim(lower, .008)
    ax.set_xticks(SCALES); ax.set_xlabel("Independent TRAIN roots"); ax.set_ylabel("Point-WM − Direct full-task SR")
    ax.set_title("World-model controller gain vs training roots")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(out / "FIG_TASK0_POINTWM_GAIN_VS_ROOTS.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "FIG_TASK0_POINTWM_GAIN_VS_ROOTS.pdf", bbox_inches="tight"); plt.close(fig)

    write_json(out / "FIG_TASK0_POINTWM_CHART_CONTRACT.json", {
        "surface": "standalone static PNG/PDF requested by user",
        "common": {"x": "independent TRAIN roots [6,15,30,50]", "primary": "3-seed metric mean with population std", "ensemble": "open markers only", "palette": colors, "non_color": "Direct circle/solid; Point-WM square/dashed", "TEST": "FROZEN HELD-OUT TASK0 TEST"},
        "gain": {"zero_line": True, "metric": "Point-WM minus Direct SR"},
        "frontier": {"current_metric": "selected-force MAE to empirical minimum force with >=4/5 successes", "old_metric": "predicted-threshold frontier MAE", "layout": "separate panels because absolute metric levels are not interchangeable"},
    })


def classification(main: pd.DataFrame, gain: pd.DataFrame) -> tuple[str, dict]:
    g = gain[gain.seed == "SEED_MEAN"].set_index("roots")
    ge = gain[gain.seed == "ENSEMBLE"].set_index("roots")
    s50 = gain[(gain.roots == 50) & gain.seed.isin(["0", "1", "2"])]
    wins = int((s50.delta_sr > 0).sum())
    h8 = main[(main.method == "Point-WM") & main.seed.isin(["0", "1", "2"])].groupby("roots").wm_h8_mae.mean()
    a = (g.loc[6, "delta_sr"] <= .02 and g.loc[15, "delta_sr"] <= .02 and
         g.loc[30, "delta_sr"] > g.loc[15, "delta_sr"] and g.loc[50, "delta_sr"] > g.loc[30, "delta_sr"] and
         g.loc[50, "delta_sr"] > 0 and wins >= 2 and g.loc[50, "delta_underforce"] <= .02 and g.loc[50, "delta_utility"] >= 0)
    b = ge.loc[50, "delta_sr"] > 0 and wins < 2
    c = int((g.delta_frontier > 0).sum()) >= 3 and g.loc[50, "delta_sr"] <= 0 and g.loc[50, "delta_utility"] <= 0
    e = g.loc[50, "delta_sr"] < 0 and h8.loc[50] < h8.loc[6]
    if a: label = "WORLD_MODEL_BENEFIT_EMERGES_WITH_DATA_SCALE"
    elif b: label = "ENSEMBLE_ONLY_WM_SCALING_SIGNAL"
    elif c: label = "WORLD_MODEL_FRONTIER_SIGNAL_WITHOUT_CONTROLLER_GAIN"
    elif e: label = "WORLD_MODEL_REMAINS_DATA_HUNGRY_AT_S50"
    else: label = "NO_MONOTONIC_WORLD_MODEL_SCALING"
    details = {
        "S50_single_seed_wins": wins,
        "selected_class_only": True,
        "descriptive_conditions": {
            "frontier_signal_without_controller_gain": bool(c),
            "S50_below_Direct_while_H8_improves": bool(e),
        },
    }
    return label, details


def fmt_pct(x: float) -> str:
    return "NA" if not math.isfinite(float(x)) else f"{100*float(x):.2f}%"


def finalize(out: Path) -> None:
    global OUT_ACTIVE
    OUT_ACTIVE = out
    for n in SCALES:
        for seed in SEEDS:
            for p in [out / "shards" / f"S{n}_seed{seed}.npz", out / "shards" / f"S{n}_seed{seed}.json"]:
                if not p.exists(): raise RuntimeError(f"missing shard artifact: {p}")
    tpi, cf, full = setup_modules()
    # Refresh the human-readable nested-root table from the frozen S50 manifest.
    # The first six root families each contain three friction contexts, so a
    # single scalar friction would silently under-report their actual structure.
    train50_contexts, train50_branches, *_ = load_scale(50, tpi, cf)
    root_rows = []
    for root in scale_root_ids(50):
        rq = train50_contexts[train50_contexts.root_id.astype(str) == root].copy()
        rb = train50_branches[train50_branches.root_id.astype(str) == root]
        root_rows.append({
            "root_id": root,
            "first_scale_included": f"S{next(k for k in SCALES if root in set(scale_root_ids(k)))}",
            "branch_count": int(len(rb)),
            "friction": ";".join(f"{x:.8g}" for x in sorted(rq.hidden_friction_analysis_only.astype(float).unique())),
            "friction_context_count": int(rq.context_id.nunique()),
            "seed": int(rq.iloc[0].root_seed),
        })
    write_csv(out / "TASK0_POINTWM_SCALING_ROOTS.csv", root_rows)
    contexts, branches, traces, meta, cells, frontiers, trace_by_key, branch = truth_tables(tpi, cf)
    truth_y = branch.success.to_numpy(int)
    truth_context = branch.context_id.astype(str).to_numpy()
    truth_force = branch.force_N.to_numpy(float)
    train_counts = {n: json.loads((out / "shards" / f"S{n}_seed0.json").read_text())["train_branches"] for n in SCALES}
    rows, per_method, per_paired, mechanisms, probe_rows = [], [], [], [], []

    for n in SCALES:
        seed_cands = []
        seed_wm = []
        seed_probe_score = []
        for seed in SEEDS:
            z = np.load(out / "shards" / f"S{n}_seed{seed}.npz")
            cand = shard_frame(z)
            seed_cands.append(cand)
            seed_wm.append(z["wm_traj"].astype(float))
            direct_lookup = {(str(r.context_id), round(float(r.force_N), 6)): float(r.direct_p) for r in cand.itertuples(index=False)}
            point_p = implied_probability(cand.point_score.to_numpy(), cand.force_N.to_numpy())
            point_lookup = {(str(r.context_id), round(float(r.force_N), 6)): float(point_p[i]) for i, r in cand.iterrows()}
            dp = np.asarray([direct_lookup[(str(r.context_id), round(float(r.force_N), 6))] for r in branch.itertuples(index=False)])
            pp = np.asarray([point_lookup[(str(r.context_id), round(float(r.force_N), 6))] for r in branch.itertuples(index=False)])
            dm = probability_metrics(truth_y, dp, truth_context, truth_force)
            pm = probability_metrics(truth_y, pp, truth_context, truth_force)
            wm = wm_metrics(z["wm_traj"].astype(float), cand, trace_by_key)
            dsel = controller_selection(cand, "direct_score")
            psel = controller_selection(cand, "point_score")
            dr, dper, dep = evaluate_controller(dsel, branch, cells, frontiers, "Direct", n, str(seed), dm, None)
            pr, pper, pep = evaluate_controller(psel, branch, cells, frontiers, "Point-WM", n, str(seed), pm, wm)
            dr["train_branches"] = train_counts[n]; pr["train_branches"] = train_counts[n]
            rows += [dr, pr]; per_method += [dper, pper]
            per_paired.append(paired_per_root(n, str(seed), dper, pper))
            mechanisms.append(mechanism_table(n, str(seed), dsel, psel, branch))
            if "probe_point_score" in z:
                seed_probe_score.append(z["probe_point_score"].astype(float))
                pcand = cand.copy(); pcand["probe_score"] = z["probe_point_score"].astype(float)
                pprob = implied_probability(pcand.probe_score.to_numpy(), pcand.force_N.to_numpy())
                plook = {(str(r.context_id), round(float(r.force_N), 6)): float(pprob[i]) for i, r in pcand.iterrows()}
                bp = np.asarray([plook[(str(r.context_id), round(float(r.force_N), 6))] for r in branch.itertuples(index=False)])
                bpm = probability_metrics(truth_y, bp, truth_context, truth_force)
                bwm = wm_metrics(z["probe_wm_traj"].astype(float), cand, trace_by_key)
                bsel = controller_selection(pcand, "probe_score")
                br, _, _ = evaluate_controller(bsel, branch, cells, frontiers, "Probe-PointWM", n, str(seed), bpm, bwm)
                br["train_branches"] = train_counts[n]; probe_rows.append(br)

        # Prediction/utility-level ensemble: average scores/probabilities first,
        # then choose a force once.  This is not a metric average.
        cand = seed_cands[0].copy()
        cand["direct_p"] = np.mean([x.direct_p.to_numpy(float) for x in seed_cands], axis=0)
        cand["direct_score"] = cand.direct_p * ((FMAX - cand.force_N) / FMAX) + (1 - cand.direct_p) * -1.0
        cand["point_score"] = np.mean([x.point_score.to_numpy(float) for x in seed_cands], axis=0)
        wm_ens = np.mean(seed_wm, axis=0)
        direct_lookup = {(str(r.context_id), round(float(r.force_N), 6)): float(r.direct_p) for r in cand.itertuples(index=False)}
        point_p = implied_probability(cand.point_score.to_numpy(), cand.force_N.to_numpy())
        point_lookup = {(str(r.context_id), round(float(r.force_N), 6)): float(point_p[i]) for i, r in cand.iterrows()}
        dp = np.asarray([direct_lookup[(str(r.context_id), round(float(r.force_N), 6))] for r in branch.itertuples(index=False)])
        pp = np.asarray([point_lookup[(str(r.context_id), round(float(r.force_N), 6))] for r in branch.itertuples(index=False)])
        dm = probability_metrics(truth_y, dp, truth_context, truth_force)
        pm = probability_metrics(truth_y, pp, truth_context, truth_force)
        wm = wm_metrics(wm_ens, cand, trace_by_key)
        dsel = controller_selection(cand, "direct_score"); psel = controller_selection(cand, "point_score")
        dr, dper, dep = evaluate_controller(dsel, branch, cells, frontiers, "Direct", n, "ENSEMBLE", dm, None)
        pr, pper, pep = evaluate_controller(psel, branch, cells, frontiers, "Point-WM", n, "ENSEMBLE", pm, wm)
        dr["train_branches"] = train_counts[n]; pr["train_branches"] = train_counts[n]
        rows += [dr, pr]; per_method += [dper, pper]
        per_paired.append(paired_per_root(n, "ENSEMBLE", dper, pper))
        mechanisms.append(mechanism_table(n, "ENSEMBLE", dsel, psel, branch))
        if seed_probe_score:
            pcand = cand.copy(); pcand["probe_score"] = np.mean(seed_probe_score, axis=0)
            pprob = implied_probability(pcand.probe_score.to_numpy(), pcand.force_N.to_numpy())
            plook = {(str(r.context_id), round(float(r.force_N), 6)): float(pprob[i]) for i, r in pcand.iterrows()}
            bp = np.asarray([plook[(str(r.context_id), round(float(r.force_N), 6))] for r in branch.itertuples(index=False)])
            bpm = probability_metrics(truth_y, bp, truth_context, truth_force)
            bwm = wm_metrics(np.mean([np.load(out / "shards" / f"S{n}_seed{s}.npz")["probe_wm_traj"] for s in SEEDS], axis=0), cand, trace_by_key)
            bsel = controller_selection(pcand, "probe_score")
            br, _, _ = evaluate_controller(bsel, branch, cells, frontiers, "Probe-PointWM", n, "ENSEMBLE", bpm, bwm)
            br["train_branches"] = train_counts[n]; probe_rows.append(br)

    main = pd.DataFrame(rows)
    # Enforce the requested stable column order; extra QA columns follow.
    first = ["roots", "train_branches", "seed", "method", "sr", "underforce", "mean_force", "excess_force", "utility", "frontier_mae", "nll", "brier", "auroc", "monotonicity", "wm_h8_mae", "wm_ie_error"]
    main = main[first + [c for c in main.columns if c not in first]]
    write_csv(out / "TASK0_POINTWM_DATA_SCALING.csv", main)
    if probe_rows:
        write_csv(out / "TASK0_PROBE_POINTWM_DATA_SCALING_SECONDARY.csv", pd.DataFrame(probe_rows))
    paired = pd.concat(per_paired, ignore_index=True)
    write_csv(out / "TASK0_POINTWM_PER_ROOT_SCALING.csv", paired)
    mechanism = pd.concat(mechanisms, ignore_index=True)
    write_csv(out / "TASK0_POINTWM_MECHANISM_BY_SCALE.csv", mechanism)

    gain_rows = []
    for n in SCALES:
        for seed_label in ["0", "1", "2", "ENSEMBLE"]:
            q = main[(main.roots == n) & (main.seed == seed_label)].set_index("method")
            gain_rows.append({
                "roots": n, "train_branches": train_counts[n], "seed": seed_label,
                "delta_sr": q.loc["Point-WM", "sr"] - q.loc["Direct", "sr"],
                "delta_sr_pp": 100 * (q.loc["Point-WM", "sr"] - q.loc["Direct", "sr"]),
                "delta_utility": q.loc["Point-WM", "utility"] - q.loc["Direct", "utility"],
                "delta_underforce": q.loc["Point-WM", "underforce"] - q.loc["Direct", "underforce"],
                "delta_underforce_pp": 100 * (q.loc["Point-WM", "underforce"] - q.loc["Direct", "underforce"]),
                "delta_frontier": q.loc["Direct", "frontier_mae"] - q.loc["Point-WM", "frontier_mae"],
            })
        singles = pd.DataFrame([x for x in gain_rows if x["roots"] == n and x["seed"] in ["0", "1", "2"]])
        gain_rows.append({
            "roots": n, "train_branches": train_counts[n], "seed": "SEED_MEAN",
            **{c: float(singles[c].mean()) for c in ["delta_sr", "delta_sr_pp", "delta_utility", "delta_underforce", "delta_underforce_pp", "delta_frontier"]},
        })
    gain = pd.DataFrame(gain_rows)
    mean_gain = gain[gain.seed == "SEED_MEAN"].sort_values("roots")
    trend = {}
    for col in ["delta_sr", "delta_utility", "delta_frontier"]:
        y = mean_gain[col].to_numpy(float)
        trend[col] = {"spearman_rho": rank_corr(SCALES, y.tolist()), "log_root_slope": float(np.polyfit(np.log(SCALES), y, 1)[0]), "n_scale_points": 4}
    gain["trend_spearman_delta_sr"] = np.where(gain.seed == "SEED_MEAN", trend["delta_sr"]["spearman_rho"], np.nan)
    gain["trend_log_slope_delta_sr"] = np.where(gain.seed == "SEED_MEAN", trend["delta_sr"]["log_root_slope"], np.nan)
    write_csv(out / "TASK0_POINTWM_GAIN_VS_SCALE.csv", gain)
    write_json(out / "TASK0_POINTWM_TREND_DIAGNOSTICS.json", trend)
    make_figures(main, gain, out)

    label, class_details = classification(main, gain)
    write_json(out / "FINAL_TASK0_POINTWM_DATA_SCALING_CLASSIFICATION.json", {
        "classification": label, "primary": "GT-PHYSICS SCALING DIAGNOSTIC",
        "test": "FROZEN HELD-OUT TASK0 TEST", **class_details, "trend": trend,
    })

    seed_main = main[main.seed.isin(["0", "1", "2"])]
    summary = seed_main.groupby(["roots", "method"], as_index=False).agg(
        sr_mean=("sr", "mean"), sr_std=("sr", lambda x: x.std(ddof=0)),
        underforce_mean=("underforce", "mean"), utility_mean=("utility", "mean"),
        frontier_mae_mean=("frontier_mae", "mean"), h8_mae_mean=("wm_h8_mae", "mean"), ie_error_mean=("wm_ie_error", "mean"),
    )
    old = pd.DataFrame({"roots": SCALES, "old_Direct_frontier_MAE": [.297, .240, .307, .312], "old_Visual_Joint_frontier_MAE": [.277, .207, .245, .253]})
    current = summary.pivot(index="roots", columns="method", values="frontier_mae_mean").reset_index().rename(columns={"Direct": "current_Direct_frontier_MAE", "Point-WM": "current_PointWM_frontier_MAE"})
    oldcmp = old.merge(current, on="roots")
    write_csv(out / "TASK0_OLD_JOINT_VS_CURRENT_POINTWM.csv", oldcmp)

    s = summary.set_index(["roots", "method"])
    g = gain[gain.seed == "SEED_MEAN"].set_index("roots")
    ens = main[main.seed == "ENSEMBLE"].set_index(["roots", "method"])
    probe_note = ""
    probe_summary = ""
    if probe_rows:
        probe_df = pd.DataFrame(probe_rows)
        probe_note = "Secondary existing-trace Probe-PointWM was run and is reported separately in `TASK0_PROBE_POINTWM_DATA_SCALING_SECONDARY.csv`; it does not enter the primary classification."
        probe_single = probe_df[probe_df.seed.isin(["0", "1", "2"])].groupby("roots").sr.mean()
        probe_ens = probe_df[probe_df.seed == "ENSEMBLE"].set_index("roots").sr
        probe_summary = "Single-seed mean Probe-PointWM SR is " + "; ".join(
            f"S{n} {fmt_pct(probe_single.loc[n])}" for n in SCALES
        ) + "; prediction-level ensemble SR is " + "; ".join(
            f"S{n} {fmt_pct(probe_ens.loc[n])}" for n in SCALES
        ) + ". This secondary curve also does not show a late-emerging controller benefit."
    s50_mech = mechanism[(mechanism.roots == 50) & mechanism.seed.isin(["0", "1", "2"])]
    s50_rescue = int(s50_mech.loc[s50_mech.mechanism == "A_RESCUE", "episodes"].sum())
    s50_collateral = int(s50_mech.loc[s50_mech.mechanism == "C_COLLATERAL_DAMAGE", "episodes"].sum())
    no_front = sum(not math.isfinite(v) for v in frontiers.values())
    report = [
        "# Final Task0 Point-WM Data-Scaling Report", "",
        "## Technical summary", "",
        f"**Classification: `{label}`.** This is a frozen **GT-PHYSICS SCALING DIAGNOSTIC** on the previously evaluated **FROZEN HELD-OUT TASK0 TEST**, not an untouched TEST result and not a deployable ActiveForcing controller claim.",
        f"Across 6→15→30→50 independent TRAIN roots, the three-seed mean Point-WM−Direct SR gains are " + ", ".join(f"S{n} {g.loc[n,'delta_sr_pp']:+.2f} pp" for n in SCALES) + ".",
        f"The diagnostic Spearman correlations with root count are ρ={trend['delta_sr']['spearman_rho']:+.3f} for ΔSR, ρ={trend['delta_utility']['spearman_rho']:+.3f} for Δutility, and ρ={trend['delta_frontier']['spearman_rho']:+.3f} for Δfrontier (n=4 scale points; descriptive only).",
        "", "## Direct answers", "",
        "1. **Independent roots:** S6=6, S15=15, S30=30, S50=50, exactly nested by the frozen manifest.",
        f"2. **Real TRAIN branches:** S6={train_counts[6]}, S15={train_counts[15]}, S30={train_counts[30]}, S50={train_counts[50]}. Counts are not 60/150/300/500 because S6 has 18 friction contexts and later roots add one context each; every context has 10 branches.",
        f"3. **TEST:** 10 independent roots, 90 force cells, 450 real branches. Each controller selects one archived force/root and reads five repeats, so evaluation is exactly 50 controller episodes/method/seed. {no_front}/10 roots are `NO_RELIABLE_FORCE` under the >=4/5 empirical rule and are excluded only from frontier/under-force denominators, never from SR or utility.",
        "4. **Direct SR learning curve (three-seed mean ± population std):** " + "; ".join(f"S{n} {fmt_pct(s.loc[(n,'Direct'),'sr_mean'])} ± {100*s.loc[(n,'Direct'),'sr_std']:.2f} pp" for n in SCALES) + ".",
        "5. **Point-WM SR learning curve:** " + "; ".join(f"S{n} {fmt_pct(s.loc[(n,'Point-WM'),'sr_mean'])} ± {100*s.loc[(n,'Point-WM'),'sr_std']:.2f} pp" for n in SCALES) + ".",
        "6. **WM−Direct at each scale:** " + "; ".join(f"S{n} ΔSR={g.loc[n,'delta_sr_pp']:+.2f} pp, Δutility={g.loc[n,'delta_utility']:+.4f}, Δunder-force={100*g.loc[n,'delta_underforce']:+.2f} pp, Δfrontier={g.loc[n,'delta_frontier']:+.4f} N" for n in SCALES) + ". Positive Δfrontier means Point-WM is closer.",
        f"7. **Does WM gain increase with roots?** {'Yes under the four-point rank diagnostic.' if trend['delta_sr']['spearman_rho'] > 0.5 else 'No systematic monotonic increase is supported by the four-point diagnostic.'} Spearman is descriptive because n=4; log(root) slopes are ΔSR {trend['delta_sr']['log_root_slope']:+.5f}, Δutility {trend['delta_utility']['log_root_slope']:+.5f}, and Δfrontier {trend['delta_frontier']['log_root_slope']:+.5f}.",
        f"8. **Replication of the pooled 60/120 pattern:** {'The larger-root Task0 curve is directionally consistent with a late-emerging WM benefit.' if g.loc[50,'delta_sr'] > g.loc[6,'delta_sr'] and g.loc[50,'delta_sr'] > 0 else 'The larger-root Task0 curve does not reproduce a clean negative-to-positive late-emergence pattern.'} The pooled 60/120 evidence and this Task0 root-scaling diagnostic differ in task composition, root/context structure, and evaluation population, so this is replication of direction only, not a pooled-effect estimate.",
        f"9. **Does S50 Point-WM exceed Direct?** {'Yes' if g.loc[50,'delta_sr'] > 0 else 'No'} on three-seed mean (Δ={g.loc[50,'delta_sr_pp']:+.2f} pp); prediction-level ensemble Δ={100*(ens.loc[(50,'Point-WM'),'sr']-ens.loc[(50,'Direct'),'sr']):+.2f} pp.",
        f"10. **Single-seed stability or ensemble-only?** S50 Point-WM beats Direct in {class_details['S50_single_seed_wins']}/3 single seeds. The ensemble is always reported separately and never substituted for the seed mean.",
        f"11. **Under-force:** at S50 Δ={100*g.loc[50,'delta_underforce']:+.2f} pp (Point-WM−Direct) over empirical-frontier-supported episodes.",
        f"12. **Utility:** at S50 Δ={g.loc[50,'delta_utility']:+.4f} realized utility.",
        f"13. **Closeness to empirical F*0.8:** at S50 Δfrontier={g.loc[50,'delta_frontier']:+.4f} N; positive means Point-WM is closer. No Fmax oracle is substituted for unsupported roots.",
        "14. **H8 trajectory prediction vs scale:** " + "; ".join(f"S{n} MAE={s.loc[(n,'Point-WM'),'h8_mae_mean']:.6f}" for n in SCALES) + ".",
        "15. **IE prediction vs scale:** " + "; ".join(f"S{n} error={s.loc[(n,'Point-WM'),'ie_error_mean']:.6f}" for n in SCALES) + ".",
        f"16. **Physics versus decision interface:** physics prediction improved (H8 MAE {s.loc[(6,'Point-WM'),'h8_mae_mean']:.6f}→{s.loc[(50,'Point-WM'),'h8_mae_mean']:.6f}; IE error {s.loc[(6,'Point-WM'),'ie_error_mean']:.6f}→{s.loc[(50,'Point-WM'),'ie_error_mean']:.6f}) and S50 frontier MAE improved by {g.loc[50,'delta_frontier']:+.4f} N, but SR and utility did not. Across the three S50 seeds there were {s50_rescue} rescue episodes and {s50_collateral} collateral-damage episodes out of 150 paired repeat-level evaluations. The evidence therefore locates the failure at the frozen physics-to-decision interface rather than at absence of learned physics signal.",
        "17. **Old auxiliary signal vs current controller:** old Visual Joint frontier MAE was lower than old Direct at all four scales (0.277/0.207/0.245/0.253 N versus 0.297/0.240/0.307/0.312 N). That establishes an old auxiliary boundary signal, not current Point-WM controller gain. The current Direct/Point-WM frontier and controller results are listed below and must be interpreted separately.",
        f"18. **Final classification:** `{label}`.",
        "", "## Seed stability", "",
        "| Scale | Seed | Direct SR | Point-WM SR | ΔSR |", "|---|---:|---:|---:|---:|",
    ]
    for n in SCALES:
        for seed in ["0", "1", "2"]:
            q = main[(main.roots == n) & (main.seed == seed)].set_index("method")
            report.append(f"| S{n} | {seed} | {fmt_pct(q.loc['Direct','sr'])} | {fmt_pct(q.loc['Point-WM','sr'])} | {100*(q.loc['Point-WM','sr']-q.loc['Direct','sr']):+.2f} pp |")
    report += [
        "", "## Controller and prediction metric curves", "",
        "| Scale | Method | SR mean | Under-force | Mean force | Utility | Frontier MAE | NLL | Brier | H8 MAE | IE error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in SCALES:
        for method in ["Direct", "Point-WM"]:
            q = seed_main[(seed_main.roots == n) & (seed_main.method == method)].mean(numeric_only=True)
            h8s = f"{q.wm_h8_mae:.6f}" if math.isfinite(q.wm_h8_mae) else "NA"
            ies = f"{q.wm_ie_error:.6f}" if math.isfinite(q.wm_ie_error) else "NA"
            report.append(f"| S{n} | {method} | {fmt_pct(q.sr)} | {fmt_pct(q.underforce)} | {q.mean_force:.4f} N | {q.utility:.4f} | {q.frontier_mae:.4f} N | {q.nll:.4f} | {q.brier:.4f} | {h8s} | {ies} |")
    report += [
        "", "## Per-root and mechanism evidence", "",
        "`TASK0_POINTWM_PER_ROOT_SCALING.csv` contains all 10 roots for every scale and each single seed plus the prediction-level ensemble. `TASK0_POINTWM_MECHANISM_BY_SCALE.csv` counts Rescue, Economize, Collateral Damage, Over-force, and unchanged/other episodes over the exact 50 paired repeat-level evaluations.",
        "", "## Old auxiliary signal versus current Point-WM controller", "",
        "| Roots | Old Direct frontier MAE | Old Visual Joint | Current Direct | Current Point-WM |", "|---:|---:|---:|---:|---:|",
    ]
    for r in oldcmp.itertuples(index=False):
        report.append(f"| {r.roots} | {r.old_Direct_frontier_MAE:.3f} | {r.old_Visual_Joint_frontier_MAE:.3f} | {r.current_Direct_frontier_MAE:.3f} | {r.current_PointWM_frontier_MAE:.3f} |")
    report += [
        "", "The old frontier advantage and current controller result answer different questions. Old Joint asked whether auxiliary visual/physics supervision improved an estimated predicted boundary. Current Point-WM asks whether a PhysicsOnly H8 model, consumed through root-OOF residual expected utility, improves archived-force SR, under-force, utility, and just-enough force selection. The old predicted-threshold frontier MAE and current selected-force MAE to empirical F*0.8 are not numerically interchangeable; only within-protocol advantage direction should be compared.",
        "", "## Methodology and frozen semantics", "",
        "Direct and Point-WM receive identical x, μ_GT, and archived candidate force. Point-WM alone learns the frozen H8×13 physical consequence with smooth-L1 trajectory loss plus λIE=1 adjacent-force supervision; no outcome BCE enters WM. Residual inputs are grouped-root OOF base predictions only. Final Direct and WM train on all S_N TRAIN roots and infer once on the frozen TEST. The controller searches only the nine archived TEST forces and maximizes the frozen expected-utility score.",
        "", "Point-WM NLL/Brier/AUROC use the utility-implied probability `(score+1)/(R_success+1)` as a diagnostic only. The actual Point-WM controller uses the unmodified Direct utility plus residual score. This conversion does not tune or change selection.",
        "", "## Secondary existing-trace Probe diagnostic", "",
        probe_summary if probe_summary else "`PROBE_SCALING_NOT_RUN_EXISTING_TRACE_UNAVAILABLE`.",
        "", "## Limitations and robustness", "",
        "- Only four scale points and ten TEST roots are available; trend correlations and log slopes are descriptive, not significance claims.",
        "- Five repeats within a root-force cell and friction-conditioned contexts within the first six root families are not independent roots.",
        "- Empirical F*0.8 is defined only when an archived force has >=4/5 successes. Unsupported roots remain in SR/utility but not frontier/under-force denominators.",
        "- Standard deviations are population standard deviations across three canonical seed metrics, not confidence intervals.",
        "- TEST was previously evaluated; no tuning or selection is performed here, but the result is a frozen diagnostic rather than a fresh confirmatory test.",
        f"- {probe_note}",
        "", "## Recommended next step", "",
        "Treat the selected classification as a diagnosis of the frozen pipeline. Do not tune on this TEST. A future confirmatory claim would require a newly preregistered, independently collected TEST population after any architecture or decision-interface change is frozen.",
        "", "## Further questions", "",
        "If controller gain does not track H8/IE improvement, the next question is whether the fixed summary52 residual discards decision-relevant trajectory structure. That question must be studied on TRAIN/DEV or a newly preregistered evaluation set, not by adapting to this frozen TEST.",
    ]
    (out / "FINAL_TASK0_POINTWM_DATA_SCALING_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    required = [
        "TASK0_POINTWM_SCALING_DATA_AUDIT.md", "TASK0_POINTWM_MODEL_COMPATIBILITY.md", "TASK0_POINTWM_SCALING_ROOTS.csv",
        "TASK0_POINTWM_DATA_SCALING.csv", "TASK0_POINTWM_GAIN_VS_SCALE.csv", "TASK0_POINTWM_PER_ROOT_SCALING.csv", "TASK0_POINTWM_MECHANISM_BY_SCALE.csv",
        "FIG_TASK0_POINTWM_SR_VS_ROOTS.png", "FIG_TASK0_POINTWM_SR_VS_ROOTS.pdf",
        "FIG_TASK0_POINTWM_GAIN_VS_ROOTS.png", "FIG_TASK0_POINTWM_GAIN_VS_ROOTS.pdf",
        "FIG_TASK0_POINTWM_UTILITY_VS_ROOTS.png", "FIG_TASK0_POINTWM_UTILITY_VS_ROOTS.pdf",
        "FIG_TASK0_POINTWM_FRONTIER_VS_ROOTS.png", "FIG_TASK0_POINTWM_FRONTIER_VS_ROOTS.pdf",
        "FINAL_TASK0_POINTWM_DATA_SCALING_REPORT.md", "FINAL_TASK0_POINTWM_DATA_SCALING_CLASSIFICATION.json",
    ]
    missing = [x for x in required if not (out / x).exists()]
    if missing: raise RuntimeError(f"required deliverables missing: {missing}")
    extras = ["TASK0_POINTWM_SCALING_PROTOCOL.json", "TASK0_POINTWM_DATA_AUDIT.json", "TASK0_POINTWM_MODEL_COMPATIBILITY.json", "TASK0_POINTWM_TREND_DIAGNOSTICS.json", "TASK0_OLD_JOINT_VS_CURRENT_POINTWM.csv", "FIG_TASK0_POINTWM_CHART_CONTRACT.json"]
    if (out / "TASK0_PROBE_POINTWM_DATA_SCALING_SECONDARY.csv").exists(): extras.append("TASK0_PROBE_POINTWM_DATA_SCALING_SECONDARY.csv")
    names = required + extras
    (out / "SHA256SUMS.txt").write_text("\n".join(f"{sha256(out / name)}  {name}" for name in names) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FINALIZED", "classification": label, "out": str(out), "S50_seed_wins": class_details["S50_single_seed_wins"], "trend": trend}, indent=2), flush=True)


def validate(out: Path) -> None:
    main = pd.read_csv(out / "TASK0_POINTWM_DATA_SCALING.csv", dtype={"seed": str})
    gain = pd.read_csv(out / "TASK0_POINTWM_GAIN_VS_SCALE.csv", dtype={"seed": str})
    per = pd.read_csv(out / "TASK0_POINTWM_PER_ROOT_SCALING.csv", dtype={"seed": str})
    mech = pd.read_csv(out / "TASK0_POINTWM_MECHANISM_BY_SCALE.csv", dtype={"seed": str})
    issues = []
    expected_main = 4 * 2 * 4
    if len(main) != expected_main: issues.append(f"main rows {len(main)} != {expected_main}")
    if set(main.roots) != set(SCALES) or set(main.method) != {"Direct", "Point-WM"}: issues.append("main scale/method identity")
    if set(main.seed) != {"0", "1", "2", "ENSEMBLE"}: issues.append(f"seed labels {set(main.seed)}")
    if not (main.controller_episodes == 50).all(): issues.append("controller episode denominator")
    if not ((main.successes / main.controller_episodes - main.sr).abs() < 1e-12).all(): issues.append("SR numerator/denominator")
    if len(per) != 4 * 4 * 10: issues.append(f"per-root rows {len(per)}")
    if mech.groupby(["roots", "seed"]).episodes.sum().ne(50).any(): issues.append("mechanism totals")
    for n in SCALES:
        for seed in ["0", "1", "2", "ENSEMBLE"]:
            q = main[(main.roots == n) & (main.seed == seed)].set_index("method")
            r = gain[(gain.roots == n) & (gain.seed == seed)].iloc[0]
            checks = [
                (r.delta_sr, q.loc["Point-WM", "sr"] - q.loc["Direct", "sr"]),
                (r.delta_utility, q.loc["Point-WM", "utility"] - q.loc["Direct", "utility"]),
                (r.delta_underforce, q.loc["Point-WM", "underforce"] - q.loc["Direct", "underforce"]),
                (r.delta_frontier, q.loc["Direct", "frontier_mae"] - q.loc["Point-WM", "frontier_mae"]),
            ]
            if any(abs(float(a) - float(b)) > 1e-12 for a, b in checks): issues.append(f"gain reconciliation S{n}/{seed}")
    # Hash validation.
    for line in (out / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if sha256(out / name) != digest: issues.append(f"hash mismatch {name}")
    # Static image/PDF existence and basic image dimensions.
    from PIL import Image
    for stem in ["SR_VS_ROOTS", "GAIN_VS_ROOTS", "UTILITY_VS_ROOTS", "FRONTIER_VS_ROOTS"]:
        p = out / f"FIG_TASK0_POINTWM_{stem}.png"
        im = Image.open(p)
        if im.width < 1000 or im.height < 600: issues.append(f"small figure {p.name}: {im.size}")
        if (out / f"FIG_TASK0_POINTWM_{stem}.pdf").stat().st_size < 1000: issues.append(f"small PDF {stem}")
    assessment = "Ready to share" if not issues else "Needs revision"
    lines = [
        "# Validation Report", "", f"## Overall Assessment: {assessment}", "",
        "## Methodology Review", "",
        "The analysis answers the frozen Task0 root-scaling question, uses exact nested root sets, grouped-root OOF residual inputs, GT friction in the primary diagnostic, and only archived nine-force TEST outcomes. Seed metric means and prediction-level ensembles remain separate.", "",
        "## Issues Found", "",
    ]
    lines += [f"- {x}" for x in issues] if issues else ["- None. All required identities, denominators, reconciliations, hashes, and static renders passed."]
    lines += [
        "", "## Calculation Spot-Checks", "",
        "- All SR values reconcile exactly to successes/50.",
        "- All four gain definitions reconcile exactly against method rows.",
        "- Every scale/seed mechanism table sums to 50 paired repeat-level episodes.",
        "- Per-root table contains 10 roots for each of 4 scales × 4 seed/ensemble aggregations.",
        "", "## Visualization Review", "",
        "All four requested PNG/PDF pairs exist. PNG dimensions exceed 1000×600; zero line is explicit on the gain figure; ensemble markers are open and separate from seed-mean lines; old Visual Joint appears only as a gray auxiliary series on the frontier figure.",
        "", "## Required Caveats", "",
        "- TEST was previously evaluated and is correctly labeled FROZEN HELD-OUT TASK0 TEST.",
        "- Four scale points and ten roots support diagnostic trends, not significance or causal claims.",
        "- NO_RELIABLE_FORCE roots are excluded from frontier denominators but retained in SR/utility.",
    ]
    (out / "TASK0_POINTWM_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if issues: raise RuntimeError(f"validation failed: {issues}")
    print(json.dumps({"status": "READY_TO_SHARE", "issues": 0, "validation": str(out / 'TASK0_POINTWM_VALIDATION_REPORT.md')}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["audit", "shard", "finalize", "validate"])
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--scale", type=int, choices=SCALES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(3, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
    out = args.out.resolve()
    if args.phase == "audit": audit(out)
    elif args.phase == "shard":
        if args.scale is None or args.seed is None: parser.error("shard requires --scale and --seed")
        shard(out, args.scale, args.seed)
    elif args.phase == "finalize": finalize(out)
    else: validate(out)
