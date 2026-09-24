"""Freeze the 32-context frozen-model motion expansion before any outcome."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
V4 = HERE.parent / "af_dump_maxf8_20260913"
MODELS = V4 / "models_v4"
RUNTIME = V4 / "additional_data_v1/RUNTIME_MANIFEST.json"
PRECOMMIT = HERE / "PRECOMMITTED_MOTION_SEEDS.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    if (HERE / "STAGE_I_FREEZE_LOCK.json").exists():
        raise FileExistsError("Stage I is already frozen")
    precommit = json.loads(PRECOMMIT.read_text())
    expected = [
        160200002,
        170200002,
        180200002,
        190200002,
        200200002,
        210200002,
        220200002,
        230200002,
    ]
    if precommit["fresh_policy_seeds"] != expected:
        raise ValueError("Precommitted seeds changed")
    if precommit["outcomes_seen_before_precommit"] is not False:
        raise ValueError("Invalid precommit")
    if precommit["methods"] != [
        "ActiveForcing",
        "Fixed-1N",
        "Fixed-3N",
        "Fixed-5N",
        "Fixed-6N",
        "Fixed-8N",
    ]:
        raise ValueError("Fixed-1 decision or method list changed")

    training_complete = MODELS / "TRAINING_COMPLETE.json"
    complete = json.loads(training_complete.read_text())
    if not complete["completed"] or complete["test_groups_executed"] != 0:
        raise ValueError("Original model lock is invalid")

    frozen_files = [
        MODELS / "feasibility/FEASIBILITY_MANIFEST.json",
        MODELS / "feasibility/CURRENT_FULLTASK_FEAS_seed0.pt",
        MODELS / "feasibility/CURRENT_FULLTASK_FEAS_seed1.pt",
        MODELS / "feasibility/CURRENT_FULLTASK_FEAS_seed2.pt",
        MODELS / "belief/BELIEF_MANIFEST.json",
        MODELS / "belief/CURRENT_MULTITASK58_seed0.pt",
        MODELS / "belief/CURRENT_MULTITASK58_seed1.pt",
        MODELS / "belief/CURRENT_MULTITASK58_seed2.pt",
        training_complete,
        RUNTIME,
        V4 / "additional_data_v1/COLLECTION_COMPLETE.json",
        HERE.parent
        / "af_dump_original_restore_20260912/original_rootlocal_dataset_v3_rim20/COLLECTION_COMPLETE.json",
        V4 / "train_v4.py",
        V4 / "infer_v4.py",
        V4 / "maxf8_runtime.py",
        V4 / "max_force_utility.py",
    ]
    source_files = [
        HERE / "freeze_stage_i.py",
        HERE / "run_stage_i.py",
        HERE / "analyze_stage_i.py",
    ]
    for path in frozen_files + source_files:
        if not path.exists():
            raise FileNotFoundError(path)

    original_freeze = {
        "model_name": "AF_original",
        "created_utc": now(),
        "frozen_files": {str(path): sha(path) for path in frozen_files},
        "architecture_changed": False,
        "hyperparameters_changed": False,
        "utility_changed": False,
        "force_controller_changed": False,
        "success_criterion_changed": False,
        "probe_policy_changed": False,
        "friction_estimator_changed": False,
    }
    write(HERE / "AF_ORIGINAL_FREEZE_MANIFEST.json", original_freeze)

    protocol = {
        "version": "MOTION_DIVERSITY_STAGE_I_FROZEN_MODEL_V1",
        "created_utc": now(),
        "precommit_path": str(PRECOMMIT),
        "precommit_sha256": sha(PRECOMMIT),
        "root": 200002,
        "task": "dump_bin_bigbin",
        "frictions": [0.375, 0.525, 0.675, 0.825],
        "policy_seeds": expected,
        "methods": precommit["methods"],
        "fixed_force_grid_N": [1.0, 3.0, 5.0, 6.0, 8.0],
        "decision_contexts": 32,
        "paired_rollouts": 192,
        "model": "AF_original",
        "models_path": str(MODELS),
        "training_complete_sha256": sha(training_complete),
        "runtime_manifest_path": str(RUNTIME),
        "runtime_manifest_sha256": sha(RUNTIME),
        "matching": {
            "same_root": True,
            "same_reset_state": True,
            "same_object_pose": True,
            "same_friction": True,
            "same_policy_seed": True,
            "same_frozen_pi0": True,
            "same_success_criterion": True,
            "same_controller": True,
            "same_first_action_chunk_discipline": True,
            "only_active_change": "grip-force policy",
        },
        "statistics_precommitted": {
            "feasible_set_primary": (
                "successful forces on common fixed grid {1,3,5,6,8}; "
                "AF-selected force is reported separately because it is not common across contexts"
            ),
            "nonmonotonic_definition": "exists F1<F2 with y(F1)=1 and y(F2)=0",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260913,
            "confidence_interval": "95% percentile",
            "bootstrap_unit": (
                "context for nonmonotonic rate; friction-stratified motion resampling "
                "for within-friction pairwise landscape statistics"
            ),
            "set_metrics": [
                "exact_set_mismatch_rate",
                "Jaccard_similarity",
                "Jaccard_distance",
                "minimum_successful_force_difference",
                "maximum_successful_force_difference",
            ],
            "paired_AF_Fixed8": [
                "both_success",
                "AF_only",
                "Fixed8_only",
                "both_failure",
            ],
            "force_saving_condition": "AF=1 and Fixed8=1",
        },
        "population_rules": {
            "all_32_contexts_in_denominator": True,
            "no_seed_replacement_after_failure": True,
            "no_outcome_based_rerun": True,
            "case_studies_separate_from_population_statistics": True,
            "stage_i_seeds_forbidden_from_stage_ii_train_val": True,
        },
        "storage": {
            "local_capacity_limited": True,
            "per_context_remote_archive_required": True,
            "remote_archive_host": "x-csong7@anvil.rcac.purdue.edu",
            "remote_archive_dir": (
                "/anvil/projects/x-cis250966/tabero-transfer/"
                "jetstream-activeforcing-20260912/motion_diversity_20260913/stage_i"
            ),
            "raw_working_copy_may_be_removed_only_after_remote_sha256_and_gzip_test": True,
            "compact_local_audit_retained": True,
        },
        "source_hashes": {str(path): sha(path) for path in source_files},
    }
    write(HERE / "STAGE_I_PROTOCOL.json", protocol)

    contexts = []
    for seed in expected:
        for mu in protocol["frictions"]:
            contexts.append(
                {
                    "id": f"final_mu{mu:.3f}_root200002_ps{seed}",
                    "split": "TEST",
                    "friction": mu,
                    "root": 200002,
                    "policy_seed": seed,
                    "task": "dump_bin_bigbin",
                    "force_support_N": [0.5, 8.0],
                    "forces_N": None,
                    "runtime_manifest_path": str(RUNTIME),
                    "runtime_manifest_sha256": sha(RUNTIME),
                    "collection_protocol_path": str(HERE / "STAGE_I_PROTOCOL.json"),
                    "collection_protocol_sha256": sha(HERE / "STAGE_I_PROTOCOL.json"),
                }
            )
    write(HERE / "CONTEXTS.json", contexts)
    lock = {
        "created_utc": now(),
        "precommit_sha256": sha(PRECOMMIT),
        "protocol_sha256": sha(HERE / "STAGE_I_PROTOCOL.json"),
        "contexts_sha256": sha(HERE / "CONTEXTS.json"),
        "original_freeze_sha256": sha(HERE / "AF_ORIGINAL_FREEZE_MANIFEST.json"),
        "source_hashes": protocol["source_hashes"],
        "outcomes_present_at_freeze": False,
    }
    write(HERE / "STAGE_I_FREEZE_LOCK.json", lock)
    # infer_v4 derives this canonical lock name from collection_protocol_path.
    write(HERE / "FREEZE_LOCK.json", lock)
    print(json.dumps({"frozen": True, "contexts": 32, "rollouts": 192}))


if __name__ == "__main__":
    main()
