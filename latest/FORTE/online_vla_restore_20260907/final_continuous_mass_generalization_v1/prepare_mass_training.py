#!/usr/bin/env python3
"""Freeze the root-disjoint MASS training acquisition before physics.

This file creates plans and hashes only.  It neither starts Isaac Sim nor
opens outcome files.  Existing output is never overwritten.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/exouser/FORTE")
SEARCH_ROOTS = (
    Path("/home/exouser/FORTE"),
    Path("/home/exouser/Tabero"),
    Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907"),
)
TASKS = {
    0: ("alphabet_soup_1", "basket_1"),
    1: ("cream_cheese_1", "basket_1"),
    5: ("tomato_sauce_1", "basket_1"),
    6: ("butter_1", "basket_1"),
}
MASS_ANCHORS_KG = (0.05, 0.10, 0.20)
MASS_LABELS = ("LOW", "MID", "HIGH")
ROOTS_BY_SPLIT = {
    "TRAIN": (181000, 181001, 181002, 181003),
    "VAL": (181010,),
    "HELDOUT": (181011,),
}
DEVELOPMENT_ONLINE_ROOT = 181020
RESERVED_FINAL_ROOTS = (181100, 181101)
CANDIDATE_FORCES_N = (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def text_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(value)


def collision_audit(roots):
    # Exact JSON field matches avoid false hits on floating point substrings.
    pattern = r'"(?:root|root_seed|seed)"\s*:\s*(?:' + "|".join(map(str, roots)) + r')(?:\s*[,}])'
    cmd = ["rg", "-l", "--glob", "*.json", pattern, *map(str, SEARCH_ROOTS)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError("root collision scan failed: " + result.stderr[-2000:])
    paths = []
    for raw in result.stdout.splitlines():
        path = Path(raw).resolve()
        if HERE.resolve() not in path.parents:
            paths.append(str(path))
    return sorted(set(paths))


def main():
    frozen = [
        HERE / "MASS_TRAINING_CONTEXT_PLAN.json",
        HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json",
        HERE / "MASS_TRAINING_PROTOCOL.json",
        HERE / "MASS_TRAINING_HASHES.txt",
    ]
    if any(path.exists() for path in frozen):
        raise FileExistsError("training freeze already exists; refusing overwrite")
    all_roots = tuple(r for values in ROOTS_BY_SPLIT.values() for r in values) + (DEVELOPMENT_ONLINE_ROOT,) + RESERVED_FINAL_ROOTS
    collisions = collision_audit(all_roots)
    if collisions:
        raise RuntimeError("proposed roots already appear in experiment JSON: " + json.dumps(collisions[:20], indent=2))

    contexts = []
    for split, roots in ROOTS_BY_SPLIT.items():
        for root in roots:
            for task, (obj, target) in TASKS.items():
                for label, mass in zip(MASS_LABELS, MASS_ANCHORS_KG):
                    cid = f"t{task}_r{root}_m{int(round(mass * 1000)):03d}g"
                    contexts.append({
                        "id": cid,
                        "task": task,
                        "root": root,
                        "split": split,
                        "object": obj,
                        "target": target,
                        "mass_band_analysis_only": label,
                        "mass_kg": mass,
                        "mu": 0.5,
                    })
    plan = {
        "version": "CURRENT_MASS_ROOT_DISJOINT_TRAINING_PLAN_V1",
        "created_utc": now(),
        "support_type": "DISCRETE_ANCHORS",
        "mass_unit": "kg",
        "mass_anchors_kg": list(MASS_ANCHORS_KG),
        "fixed_friction": 0.5,
        "intervention": "total object mass plus proportional inertia scaling",
        "geometry_appearance_changed": False,
        "roots_by_split": {k: list(v) for k, v in ROOTS_BY_SPLIT.items()},
        "root_disjoint": True,
        "sibling_force_branches_same_split": True,
        "development_online_root_reserved": DEVELOPMENT_ONLINE_ROOT,
        "final_roots_reserved_not_training": list(RESERVED_FINAL_ROOTS),
        "contexts": contexts,
    }
    plan_path = HERE / "MASS_TRAINING_CONTEXT_PLAN.json"
    dump_new(plan_path, plan)

    sources = [
        HERE / "mass_runtime_core.py",
        HERE / "mass_training_worker.py",
        HERE / "launch_mass_training.py",
        HERE / "train_mass_belief.py",
        HERE / "mass_belief.py",
        HERE / "train_mass_feasibility.py",
        HERE / "mass_feasibility.py",
        ROOT / "activeforcing_current_probe.py",
        ROOT / "activeforcing_probe_friction_contract.py",
        ROOT / "current_contract_belief_features.py",
        ROOT / "current_contract_physical_belief.py",
        ROOT / "activeforcing_execution_snapshot.py",
        ROOT / "analysis/results/current_runtime_recovery_v2_20260905/geometry_grasp_initializer.py",
        ROOT / "analysis/results/current_runtime_sensor_repair_v3_candidate_20260905/measurement_hooks.py",
        ROOT / "analysis/results/current_detached_no_chase_candidate_v1_20260906/branch_execution.py",
        Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py"),
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("freeze source missing: " + json.dumps(missing))
    manifest = {
        "version": "CURRENT_MASS_MATCHED_TRAINING_RUNTIME_V1",
        "created_utc": now(),
        "context_plan": str(plan_path),
        "context_plan_sha256": sha(plan_path),
        "query": "restored P4-B contact-conditioned tangential shear under current runtime",
        "query_feature_schema": "CURRENT_CONTRACT_58D_REAL_PATCH_V1",
        "mass_intervention": "TOTAL_OBJECT_MASS_WITH_PROPORTIONAL_INERTIA_V1",
        "fixed_friction": 0.5,
        "candidate_forces_N": list(CANDIDATE_FORCES_N),
        "label_version": "PROSPECTIVE_EARLY_LIFT_TERMINAL_AND_FROZEN_FINAL_GEOMETRY_V3",
        "preaction_shape": [8, 71],
        "phasefree_sequence_columns": [0, 1, 2, 3, 4, 5, 6, 7, 15, 16],
        "phase_columns_removed": [8, 9, 10, 11, 12, 13, 14],
        "candidate_force_column": 17,
        "latent_mass_column": 18,
        "historical_labels_used": False,
        "postaction_belief_features_allowed": False,
        "true_mass_is_training_target_only": True,
        "true_mass_deployment_observable": False,
        "source_hashes": {str(path): sha(path) for path in sources},
    }
    manifest_path = HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"
    dump_new(manifest_path, manifest)

    protocol = {
        "version": "MASS_TRAINING_FREEZE_V1",
        "created_utc": now(),
        "runtime_manifest_sha256": sha(manifest_path),
        "context_plan_sha256": sha(plan_path),
        "ordering": [
            "collect all 72 query references",
            "fit/qualify belief without feasibility or final outcomes",
            "only after PASS collect 648 matched force branches",
            "fit/qualify phase-free full-task feasibility",
            "development-only online qualification",
            "freeze final protocol",
        ],
        "belief_recipe": {
            "architecture": "three independent GRU heteroscedastic members; 58D deployable sequence",
            "seeds": [0, 1, 2],
            "epochs": 80,
            "checkpoint_rule": "minimum VAL Gaussian NLL per seed; no heldout access before lock",
            "positive_support": "deployment posterior is Gaussian mixture truncated at mass>0",
            "qualification": {
                "VAL_and_HELDOUT_MAE_better_than_train_prior": True,
                "HELDOUT_MAE_kg_max": 0.04,
                "HELDOUT_SPEARMAN_min": 0.75,
                "HELDOUT_pairwise_ranking_accuracy_min": 0.75,
                "HELDOUT_90_coverage_min": 0.75,
                "probe_admission_fraction_min": 0.90,
            },
        },
        "feasibility_recipe": {
            "architecture": "current phase-free GRU(10,64)+MLP(54,64)+head",
            "seeds": [0, 1, 2],
            "epochs": 80,
            "checkpoint_rule": "minimum VAL raw NLL per seed; no heldout access before lock",
            "label": "complete current terminal evaluator y_full",
            "posterior_integration": "deterministic positive-support mass mixture quadrature",
            "utility": "p*(5-F)/5 + (1-p)*(-1), lower-force tie break",
            "force_grid_N": [3.0, 5.0, 0.05],
            "qualification": {
                "VAL_and_HELDOUT_NLL_better_than_split_prevalence": True,
                "HELDOUT_AUROC_min_if_defined": 0.60,
                "mean_predicted_p5_minus_p3_min": 0.0,
                "known_label_fraction": 1.0,
            },
        },
        "outcome_driven_tuning": False,
        "automatic_expansion": False,
        "old_mass_data_role": "belief provenance/control only; not pooled into primary current model",
        "root_collision_audit": {
            "exact_json_field_pattern": True,
            "searched_roots": [str(p) for p in SEARCH_ROOTS],
            "proposed_roots": list(all_roots),
            "collisions": collisions,
        },
    }
    protocol_path = HERE / "MASS_TRAINING_PROTOCOL.json"
    dump_new(protocol_path, protocol)
    hashes = {
        path.name: sha(path)
        for path in (plan_path, manifest_path, protocol_path)
    }
    text_new(HERE / "MASS_TRAINING_HASHES.txt", "".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    print(json.dumps({
        "frozen": True,
        "contexts": len(contexts),
        "branches_after_belief_pass": len(contexts) * len(CANDIDATE_FORCES_N),
        "roots_by_split": plan["roots_by_split"],
        "root_collisions": collisions,
        "protocol_sha256": sha(protocol_path),
    }, indent=2))


if __name__ == "__main__":
    main()
