#!/usr/bin/env python3
"""Freeze and collect the root-diverse direct-contact boundary dataset.

This is a data-only successor run.  It reuses the authoritative P5-S0-C
runner and historical force frontiers, and deliberately stops before any
Physics-GRU v3 training.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
OLD = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
RUNNER = REPO / "analysis/p5s0c_paired_boundary_probe_value.py"
PRIOR = RESULTS / "direct_contact_boundary_imagination_20260829_000205"
TASKS = [0, 1, 5, 6]
BANDS = ["LOW", "MID", "HIGH"]
SPLITS = ["TRAIN", "DEV", "TEST"]
MAX_ROOTS_PER_TASK_SPLIT_BAND = 2
REPEATS = 3

FEATURES = {
    "input_policy": "corrected direct physical telemetry only",
    "active_phases": ["lift", "transit", "over_basket", "place"],
    "fixed_window": {
        "purpose": "prevent early-termination trace length from becoming a label shortcut",
        "rows": 120,
        "alignment": "first 120 physics steps beginning at first lift row; phase-aligned when phase is present",
        "release_settling_excluded": True,
    },
    "summary_features": [
        "bilateral_contact_fraction",
        "longest_bilateral_contact_loss_duration_s",
        "left_normal_force_mean_min_std_N",
        "right_normal_force_mean_min_std_N",
        "left_right_normal_force_imbalance_mean_abs_N",
        "left_right_normal_force_imbalance_max_abs_N",
        "left_tangential_force_mean_max_N",
        "right_tangential_force_mean_max_N",
        "valid_tangential_normal_ratio_mean_max",
        "relative_position_drift_mean_max_m",
        "relative_velocity_mean_max_mps",
        "tangential_relative_velocity_proxy_mean_max_mps",
        "gripper_joint_deviation_mean_max",
        "phase_wise_versions_where_available",
    ],
    "forbidden_inputs": [
        "commanded_force",
        "force_class",
        "ground_truth_friction",
        "friction_band",
        "task_success",
        "frontier",
        "F_star",
        "root_id",
        "split_id",
        "post_release_data",
        "trace_length",
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_worker():
    spec = importlib.util.spec_from_file_location("p5s0c_boundary_worker", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_triplets(manifest: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    population: list[dict] = []
    ineligible: list[dict] = []
    for (task, split, band), group in manifest.groupby(["task", "split", "friction_band"], sort=True):
        rows = []
        for context_id, context in group.groupby("context_id", sort=True):
            forces = sorted(float(x) for x in context.requested_force_N.unique())
            successes = sorted(float(x) for x in context.loc[context.full_task_success_y == 1, "requested_force_N"].unique())
            reason = None
            if not successes:
                reason = "NO_HISTORICAL_SUCCESS_FORCE"
            else:
                star = successes[0]
                if star not in forces:
                    reason = "FRONTIER_NOT_IN_CONTEXT_FORCE_LATTICE"
                else:
                    index = forces.index(star)
                    if index == 0:
                        reason = "NO_F_PREV_IN_AUTHORITATIVE_LATTICE"
                    elif index == len(forces) - 1:
                        reason = "NO_F_NEXT_IN_AUTHORITATIVE_LATTICE"
            if reason:
                ineligible.append({
                    "context_id": str(context_id), "task": int(task), "split": str(split),
                    "friction_band": str(band), "reason": reason, "forces": forces,
                    "historical_successes": successes,
                })
                continue
            idx = forces.index(star)
            rows.append({
                "context_id": str(context_id), "root_id": str(context.root_id.iloc[0]),
                "root_index": int(context.root_index.iloc[0]), "root_seed": int(context.root_seed.iloc[0]),
                "task": int(task), "split": str(split), "friction_band": str(band),
                "hidden_friction_analysis_only": float(context.hidden_friction_analysis_only.iloc[0]),
                "force_lattice": forces, "F_prev": forces[idx - 1], "F_star": star,
                "F_next": forces[idx + 1], "historical_frontier_definition": "minimum successful force",
            })
        rows.sort(key=lambda x: (x["root_index"], x["context_id"]))
        population.extend(rows[:MAX_ROOTS_PER_TASK_SPLIT_BAND])
        for extra in rows[MAX_ROOTS_PER_TASK_SPLIT_BAND:]:
            extra = dict(extra)
            extra["reason"] = "ELIGIBLE_BUT_NOT_SAMPLED_SMALL_POPULATION"
            ineligible.append(extra)
    population.sort(key=lambda x: (SPLITS.index(x["split"]), x["task"], BANDS.index(x["friction_band"]), x["root_index"]))
    return population, ineligible


def main() -> int:
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RESULTS / f"direct_contact_boundary_dataset_{tag}"
    out.mkdir(parents=True, exist_ok=False)

    manifest_path = OLD / "P5S0C_BRANCH_MANIFEST.csv"
    manifest = pd.read_csv(manifest_path)
    population, ineligible = historical_triplets(manifest)
    selected_ids = [x["context_id"] for x in population]
    filter_map = {x["context_id"]: [x["F_prev"], x["F_star"], x["F_next"]] for x in population}

    prior_files = [
        PRIOR / "DIRECT_CONTACT_LOGGER_AUDIT.json",
        PRIOR / "DIRECT_PHYSICAL_EVENT_DEFINITION.json",
        PRIOR / "DIRECT_CONTACT_DATASET_AUDIT.json",
        RUNNER,
        manifest_path,
    ]
    protocol = {
        "protocol_name": "DIRECT_CONTACT_ROOT_DIVERSE_DATASET",
        "protocol_version": "1.0-data-only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_goal": "test whether corrected direct physical telemetry separates F_prev from F_star/F_next on unseen roots without leakage",
        "status_before_outcomes": "FROZEN_BEFORE_NEW_SCIENTIFIC_ROLLOUTS",
        "reused_artifacts": {str(p.relative_to(REPO)): sha256(p) for p in prior_files if p.exists()},
        "tasks": TASKS,
        "friction_bands": BANDS,
        "root_split_membership": "inherited from authoritative P5-S0-C root split",
        "candidate_population": population,
        "ineligible_contexts": ineligible,
        "selected_context_count": len(population),
        "repeats_per_force": REPEATS,
        "force_definitions": "F_prev, F_star, F_next are adjacent entries in the authoritative context-specific lattice; no frontier search or retuning",
        "telemetry_fields": [
            "relative_object_gripper_position", "relative_object_gripper_velocity",
            "left_normal_force_N", "right_normal_force_N", "left_tangential_force_N",
            "right_tangential_force_N", "contact_left", "contact_right",
            "tangential_relative_velocity_proxy", "gripper_joint_state", "task_phase",
            "object_pose", "object_linear_velocity", "object_angular_velocity",
        ],
        "event_definition": "inherit corrected DIRECT_PHYSICAL_EVENT_DEFINITION.json; active phases only; intentional release and settling excluded",
        "feature_set": FEATURES,
        "classifier_family": {
            "models": ["LogisticRegression(C=1, class_weight=balanced)", "LinearSVM(C=1, class_weight=balanced)"],
            "fit_scope": "TRAIN roots only",
            "normalization": "TRAIN-fitted median imputation and standardization",
            "no_feature_redesign_after_outcomes": True,
        },
        "metrics": ["AUROC", "AUPRC", "balanced_accuracy", "F1", "F_prev_sensitivity", "F_star_next_specificity", "paired_boundary_ranking", "strict_triplet_ordering", "bootstrap_95_CI_by_context"],
        "formal_threshold_audit": "No formal Gates B-D threshold was found in the inherited repository artifacts; descriptive evidence will be reported without post-hoc pass/fail threshold invention.",
        "collection_contract": {"same_snapshot_triplet": True, "scientific_failure_retry": False, "engineering_retry_only": True, "fresh_worker_per_task": True},
    }
    protocol_path = out / "DIRECT_CONTACT_ROOT_DIVERSE_PROTOCOL.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "DIRECT_CONTACT_ROOT_DIVERSE_PROTOCOL.sha256").write_text(sha256(protocol_path) + "\n", encoding="utf-8")
    (out / "BOUNDARY_FORCE_FILTER.json").write_text(json.dumps(filter_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(population).to_csv(out / "SELECTED_POPULATION.csv", index=False)
    pd.DataFrame(ineligible).to_json(out / "INELIGIBLE_CONTEXTS.json", orient="records", indent=2)
    (out / "REUSE_MANIFEST.json").write_text(json.dumps({str(p.relative_to(REPO)): sha256(p) for p in prior_files if p.exists()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not population:
        raise RuntimeError("no eligible contexts with historical F_prev/F_star/F_next triplets")

    module = load_worker()
    os.environ.update({
        "P5S0C_OUT": str(out),
        "P5S0C_FORCE_FILTER_JSON": str(out / "BOUNDARY_FORCE_FILTER.json"),
        "P5S0C_CONTEXT_IDS": ",".join(selected_ids),
        "P5S0C_CONTEXT_ID": "",
        "P5S0C_REPEAT_ALL_FORCES": str(REPEATS),
        "P5S0C_SKIP_REPLAY": "1",
    })
    records = []
    for task in module.TASKS:
        records.append(module.launch_worker(out, task))
    (out / "WORKER_LAUNCH.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    status = 0 if all(int(x.get("returncode", 1)) == 0 for x in records) else 1
    (out / "COLLECTION_STATUS.json").write_text(json.dumps({"returncode": status, "selected_contexts": len(population), "expected_traces": len(population) * 3 * REPEATS, "workers": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "selected_contexts": len(population), "ineligible_contexts": len(ineligible), "expected_traces": len(population) * 3 * REPEATS, "returncode": status}, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
