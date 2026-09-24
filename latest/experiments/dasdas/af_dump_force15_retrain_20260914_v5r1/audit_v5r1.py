"""Independent, fail-closed audit of the force-15 v5r1 training result."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset_v5"
MODELS = HERE / "models_v5"
FORCES = [5.0, 8.0, 10.0, 12.0, 15.0]
EXPECTED_SPLIT_CONTEXTS = {"TRAIN": 12, "VAL": 6, "POSTHOC_TEST": 6}
EXPECTED_SPLIT_LABELS = {"TRAIN": 60, "VAL": 30, "POSTHOC_TEST": 30}


def read(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_sequence_hash(feature: dict) -> str:
    payload = json.dumps(feature["sequence"], separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_sequence_hash(feature: dict) -> str:
    return hashlib.sha256(np.asarray(feature["sequence"], dtype=np.float32).tobytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, atol: float = 1e-9) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=atol))


def auc_score(y: list[int], p: list[float]) -> float | None:
    positives = [score for target, score in zip(y, p) if target == 1]
    negatives = [score for target, score in zip(y, p) if target == 0]
    if not positives or not negatives:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in positives for b in negatives)
    return float(wins / (len(positives) * len(negatives)))


def paired_counts(candidate: list[int], baseline: list[int]) -> dict:
    result = {"both_success": 0, "candidate_only": 0, "baseline_only": 0, "both_fail": 0}
    for left, right in zip(candidate, baseline):
        key = "both_success" if left and right else "candidate_only" if left else "baseline_only" if right else "both_fail"
        result[key] += 1
    discordant = result["candidate_only"] + result["baseline_only"]
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(result["candidate_only"], result["baseline_only"]) + 1)) / (2**discordant)
        result["exact_mcnemar_two_sided_p"] = min(1.0, 2 * tail)
    else:
        result["exact_mcnemar_two_sided_p"] = 1.0
    result["net_paired_wins"] = result["candidate_only"] - result["baseline_only"]
    return result


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    checks: list[str] = []
    profile = read(DATASET / "DATASET_PROFILE.json")
    index = read(DATASET / "DATASET_INDEX.json")
    freeze_lock = read(DATASET / "FREEZE_LOCK.json")
    protocol = read(MODELS / "TRAINING_PROTOCOL.json")
    protocol_lock = read(MODELS / "TRAINING_PROTOCOL_LOCK.json")
    calibration = read(MODELS / "REALIZED_FORCE_CALIBRATION.json")
    manifest = read(MODELS / "feasibility/FEASIBILITY_MANIFEST.json")
    checkpoint_lock = read(MODELS / "feasibility/CHECKPOINT_SELECTION_LOCK.json")
    reload_qualification = read(MODELS / "feasibility/RUNTIME_RELOAD_QUALIFICATION.json")
    model_lock = read(MODELS / "MODEL_UTILITY_LOCK.json")
    test_receipt = read(MODELS / "TEST_ACCESS_RECEIPT.json")
    results = read(MODELS / "POSTHOC_TEST_RESULTS.json")
    complete = read(MODELS / "TRAINING_COMPLETE.json")

    require(profile["completed"] is True, "Dataset profile is not complete")
    require(profile["contexts"] == 24 and profile["labels"] == 120, "New dataset denominator mismatch")
    require(profile["split_contexts"] == EXPECTED_SPLIT_CONTEXTS, "New context split mismatch")
    require(profile["split_labels"] == EXPECTED_SPLIT_LABELS, "New label split mismatch")
    require(profile["feature_shape"] == [8, 64], "Feature shape mismatch")
    require(profile["all_branch_audits_passed"] is True, "A source branch audit failed")
    require(profile["measured_squeeze_is_feasibility_input"] is False, "Post-action squeeze leaked into feasibility")
    require(freeze_lock["dataset_index_sha256"] == sha(DATASET / "DATASET_INDEX.json"), "Dataset index hash mismatch")
    require(freeze_lock["dataset_profile_sha256"] == sha(DATASET / "DATASET_PROFILE.json"), "Dataset profile hash mismatch")
    require(freeze_lock["freeze_script_sha256"] == sha(HERE / "freeze_force15_dataset.py"), "Freezer source hash mismatch")
    checks.append("new_dataset_freeze_and_profile")

    contexts = index["contexts"]
    require(len(contexts) == 24 and len({entry["id"] for entry in contexts}) == 24, "Context IDs are not unique")
    require(Counter(entry["split"] for entry in contexts) == Counter(EXPECTED_SPLIT_CONTEXTS), "Index split counts mismatch")
    require({entry["root"] for entry in contexts} == {200002}, "Unexpected physical root")
    feature_splits: dict[str, set[str]] = defaultdict(set)
    labels_by_context: dict[str, dict[float, dict]] = {}
    all_label_hashes = {}
    for entry in contexts:
        feature = read(Path(entry["feature_path"]))
        sequence = np.asarray(feature["sequence"], dtype=np.float32)
        require(sequence.shape == (8, 64), f"Bad feature shape: {entry['id']}")
        require(feature.get("candidate_actions_executed") == 0, f"Action leakage: {entry['id']}")
        require(frozen_sequence_hash(feature) == entry["feature_sequence_sha256"], f"Frozen feature sequence hash mismatch: {entry['id']}")
        feature_splits[tensor_sequence_hash(feature)].add(entry["split"])
        label_path = Path(entry["label_file"])
        require(sha(label_path) == entry["label_file_sha256"], f"Label hash mismatch: {entry['id']}")
        rows = read(label_path)
        require(len(rows) == 5, f"Not five force branches: {entry['id']}")
        require(sorted(float(row["force"]) for row in rows) == FORCES, f"Force coverage mismatch: {entry['id']}")
        require(all(row["context_id"] == entry["id"] and row["split"] == entry["split"] for row in rows), f"Branch escaped context split: {entry['id']}")
        require(all(row["full_task_success_y"] in (0, 1) for row in rows), f"Invalid label: {entry['id']}")
        labels_by_context[entry["id"]] = {float(row["force"]): row for row in rows}
        all_label_hashes[entry["id"]] = sha(label_path)
    require(len(feature_splits) == 24, "New context features are not unique")
    require(all(len(splits) == 1 for splits in feature_splits.values()), "Exact feature duplicate crosses new split")
    checks.append("context_grain_split_and_five_branch_pairing")

    # Import the loaders to independently enumerate the frozen pre-existing pool.
    sys.path.insert(0, str(HERE))
    import train_v5

    old_records, _, _ = train_v5.load_old()
    require(sum(len(record["branches"]) for record in old_records) == 192, "Existing label denominator is not 192")
    require({record["root"] for record in old_records} == {200002}, "Existing pool includes another root")
    old_feasibility_records = [record for record in old_records if record["branches"]]
    require(len(old_feasibility_records) == 24, "Expected 24 existing feasibility contexts")
    require(Counter(record["split"] for record in old_feasibility_records) == Counter({"TRAIN": 18, "VAL": 6}), "Existing feasibility split mismatch")
    combined_feature_splits: dict[str, set[str]] = defaultdict(set)
    for record in old_feasibility_records:
        sequence = np.asarray(record["feature"]["sequence"], dtype=np.float32)
        combined_feature_splits[hashlib.sha256(sequence.tobytes()).hexdigest()].add(record["split"])
    for entry in contexts:
        combined_feature_splits[tensor_sequence_hash(read(Path(entry["feature_path"])))].add(entry["split"])
    require(all(len(splits) == 1 for splits in combined_feature_splits.values()), "Exact feasibility feature crosses combined split")
    old_hashes = {
        hashlib.sha256(np.asarray(record["feature"]["sequence"], dtype=np.float32).tobytes()).hexdigest()
        for record in old_feasibility_records
    }
    require(old_hashes.isdisjoint(set(feature_splits)), "Old/new exact feature overlap")
    require(protocol["feasibility_labels_train_val"] == 282, "Protocol training/validation label denominator mismatch")
    require(192 + EXPECTED_SPLIT_LABELS["TRAIN"] + EXPECTED_SPLIT_LABELS["VAL"] == 282, "Integrated denominator arithmetic mismatch")
    checks.append("old_192_plus_new_90_integrated_without_exact_feature_leakage")

    require(calibration["fit_split"] == "TRAIN", "Calibration was not fit on TRAIN only")
    require(calibration["feasibility_input"] is False and calibration["utility_only"] is True, "Calibration role mismatch")
    train_rows = [row for entry in contexts if entry["split"] == "TRAIN" for row in labels_by_context[entry["id"]].values()]
    medians = [float(np.median([row["target_contact_mean_squeeze_N"] for row in train_rows if float(row["force"]) == force])) for force in FORCES]
    raw_expected = [0.5] + medians
    monotone_expected = np.maximum.accumulate(np.asarray(raw_expected, dtype=float)).tolist()
    require(np.allclose(calibration["raw_train_medians_N"], raw_expected, rtol=0.0, atol=1e-12), "Calibration medians do not reproduce from TRAIN")
    require(np.allclose(calibration["realized_squeeze_knots_N"], monotone_expected, rtol=0.0, atol=1e-12), "Monotone calibration knots do not reproduce")
    require(calibration["observations_per_new_force"] == [0, 12, 12, 12, 12, 12], "Calibration observation counts mismatch")
    checks.append("train_only_realized_squeeze_calibration")

    protocol_sha = sha(MODELS / "TRAINING_PROTOCOL.json")
    calibration_sha = sha(MODELS / "REALIZED_FORCE_CALIBRATION.json")
    manifest_sha = sha(MODELS / "feasibility/FEASIBILITY_MANIFEST.json")
    model_lock_sha = sha(MODELS / "MODEL_UTILITY_LOCK.json")
    results_sha = sha(MODELS / "POSTHOC_TEST_RESULTS.json")
    require(protocol_lock["sha256"] == protocol_sha, "Protocol lock mismatch")
    require(protocol["force_support"] == [0.5, 15.0] and protocol["force_feature_normalization_N"] == 15.0, "Protocol force contract mismatch")
    require(protocol["measured_squeeze_used_by_feasibility"] is False, "Protocol admits squeeze leakage")
    require(protocol["test_labels_accessed_before_model_utility_lock"] is False, "Protocol admits pre-lock test access")
    require(protocol["belief_intervention"] == "NONE_REUSE_FROZEN_V4", "Belief was not frozen")
    require(manifest["force_support"] == [0.5, 15.0], "Manifest support mismatch")
    require(manifest["force_feature_normalization_N"] == 15.0 and manifest["utility_normalization_N"] == 15.0, "Manifest normalization mismatch")
    require(manifest["label_target"] == "full_task_success_y", "Wrong feasibility target")
    require(manifest["training_protocol_sha256"] == protocol_sha, "Manifest protocol hash mismatch")
    require(manifest["realized_force_calibration_sha256"] == calibration_sha, "Manifest calibration hash mismatch")
    require(len(manifest["checkpoints"]) == 3, "Not a three-member ensemble")
    for checkpoint in manifest["checkpoints"]:
        require(sha(Path(checkpoint["path"])) == checkpoint["sha256"], f"Checkpoint hash mismatch: seed {checkpoint['seed']}")
    require(checkpoint_lock["checkpoints"] == manifest["checkpoints"], "Checkpoint selection lock differs from manifest")
    require(checkpoint_lock["test_labels_accessed_before_lock"] is False, "Checkpoint lock admits test leakage")
    require(reload_qualification["passed"] is True and reload_qualification["tested_real_VAL_branches"] == 78, "Runtime reload qualification failed")
    require(reload_qualification["max_probability_error"] < 1e-7, "Runtime reload error too large")
    require(model_lock["training_protocol_sha256"] == protocol_sha, "Model lock protocol hash mismatch")
    require(model_lock["feasibility_manifest_sha256"] == manifest_sha, "Model lock manifest hash mismatch")
    require(model_lock["realized_force_calibration_sha256"] == calibration_sha, "Model lock calibration hash mismatch")
    require(model_lock["test_labels_accessed"] is False, "Model lock admits test access")
    require(sha(Path(model_lock["belief_manifest_path"])) == model_lock["belief_manifest_sha256"], "Frozen belief hash mismatch")
    require(test_receipt["model_utility_lock_sha256"] == model_lock_sha, "Test receipt does not reference model lock")
    require(parse_time(test_receipt["accessed_utc"]) >= parse_time(model_lock["created_utc"]), "Test labels were accessed before lock")
    require(set(test_receipt["post_lock_test_label_hashes"].values()) == set(freeze_lock["sealed_test_label_hashes"].values()), "Test receipt label hashes mismatch")
    require(complete["completed"] is True, "Training completion marker false")
    require(complete["training_protocol_sha256"] == protocol_sha, "Completion protocol hash mismatch")
    require(complete["model_utility_lock_sha256"] == model_lock_sha, "Completion model lock hash mismatch")
    require(complete["posthoc_test_results_sha256"] == results_sha, "Completion results hash mismatch")
    checks.append("model_utility_and_post_lock_test_hash_chain")

    for path_string, expected in protocol["input_hashes_pre_test"].items():
        require(sha(Path(path_string)) == expected, f"Pre-test input hash changed: {path_string}")
    checks.append("all_pre_test_input_hashes_reverified")

    test_entries = {entry["id"]: entry for entry in contexts if entry["split"] == "POSTHOC_TEST"}
    require(len(results["decisions"]) == 6, "Not six posthoc decisions")
    require({decision["context_id"] for decision in results["decisions"]} == set(test_entries), "Posthoc decision context mismatch")
    calibration_x = np.asarray(calibration["command_knots_N"], dtype=float)
    calibration_y = np.asarray(calibration["realized_squeeze_knots_N"], dtype=float)
    targets: list[int] = []
    probabilities: list[float] = []
    selected: list[int] = []
    fixed: dict[float, list[int]] = {force: [] for force in FORCES}
    calibration_errors: list[float] = []
    for decision in results["decisions"]:
        context_id = decision["context_id"]
        outcomes = {force: int(labels_by_context[context_id][force]["full_task_success_y"]) for force in FORCES}
        observed_p = np.asarray([float(decision["observed_force_probabilities"][str(force)]) for force in FORCES])
        realized = np.interp(np.asarray(FORCES), calibration_x, calibration_y)
        utilities = observed_p * (15.0 - realized) / 15.0 - (1.0 - observed_p)
        chosen_force = FORCES[int(np.argmax(utilities))]
        require(close(chosen_force, float(decision["selected_observed_force_N"])), f"Selection mismatch: {context_id}")
        require(np.allclose(utilities, [decision["observed_force_utilities"][str(force)] for force in FORCES], rtol=0.0, atol=1e-12), f"Utility mismatch: {context_id}")
        require(int(decision["selected_success"]) == outcomes[chosen_force], f"Selected outcome mismatch: {context_id}")
        selected.append(outcomes[chosen_force])
        for force, probability in zip(FORCES, observed_p):
            targets.append(outcomes[force])
            probabilities.append(float(probability))
            fixed[force].append(outcomes[force])
            measured = labels_by_context[context_id][force]["target_contact_mean_squeeze_N"]
            if measured is not None:
                calibration_errors.append(abs(float(np.interp(force, calibration_x, calibration_y)) - float(measured)))
    y = np.asarray(targets, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    nll = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
    brier = float(np.mean((p - y) ** 2))
    auc = auc_score(targets, probabilities)
    require(close(nll, results["branch_nll"]), "Posthoc NLL does not recompute")
    require(close(brier, results["branch_brier"]), "Posthoc Brier does not recompute")
    require(close(float(auc), results["branch_auc"]), "Posthoc AUC does not recompute")
    require(sum(selected) == results["selected_observed_force_successes"], "Selected success count mismatch")
    require({str(force): sum(values) for force, values in fixed.items()} == results["fixed_successes"], "Fixed success counts mismatch")
    require(paired_counts(selected, fixed[8.0]) == results["selected_vs_fixed8"], "Paired fixed-8 result mismatch")
    require(paired_counts(selected, fixed[12.0]) == results["selected_vs_fixed12"], "Paired fixed-12 result mismatch")
    require(close(float(np.mean(calibration_errors)), results["realized_calibration_mae_N"]), "Calibration MAE mismatch")
    checks.append("posthoc_metrics_and_decisions_recomputed_from_sealed_labels")

    # Reload the frozen runtime and reproduce all five probabilities per held-out context.
    new_records, _, _ = train_v5.load_new(include_test_labels=True)
    train_v5.attach_frozen_v4_belief(new_records, protocol["old_data_locks"][1]["runtime_sha256"])
    from force15_runtime import Force15Feasibility

    runtime = Force15Feasibility(MODELS / "feasibility/FEASIBILITY_MANIFEST.json", manifest_sha256=manifest_sha, device="cuda")
    decision_by_id = {decision["context_id"]: decision for decision in results["decisions"]}
    max_probability_error = 0.0
    for record in new_records:
        if record["split"] != "POSTHOC_TEST":
            continue
        curve = runtime.curve(record["feature"]["sequence"], record["posterior"])
        for force in FORCES:
            index_value = int(np.flatnonzero(np.isclose(runtime.force_grid, force, rtol=0.0, atol=1e-8))[0])
            expected = float(decision_by_id[record["id"]]["observed_force_probabilities"][str(force)])
            max_probability_error = max(max_probability_error, abs(float(curve[index_value]) - expected))
    require(max_probability_error < 1e-7, "Frozen runtime does not reproduce held-out probabilities")
    checks.append("frozen_runtime_heldout_probability_replay")

    prevalence = float(np.mean(y))
    constant_nll = float(-(prevalence * math.log(prevalence) + (1.0 - prevalence) * math.log(1.0 - prevalence)))
    constant_brier = float(prevalence * (1.0 - prevalence))
    selected_forces = [float(decision["selected_observed_force_N"]) for decision in results["decisions"]]
    repair_supported = bool(
        sum(selected) > sum(fixed[8.0])
        and sum(selected) > sum(fixed[12.0])
        and results["branch_auc"] > 0.5
        and results["branch_nll"] < constant_nll
    )
    audit = {
        "version": "AF_DUMP_FORCE15_V5R1_INDEPENDENT_AUDIT_V1",
        "passed": True,
        "checks_passed": checks,
        "dataset": {
            "new_contexts": 24,
            "new_labels": 120,
            "new_split_contexts": EXPECTED_SPLIT_CONTEXTS,
            "new_split_labels": EXPECTED_SPLIT_LABELS,
            "existing_labels": 192,
            "feasibility_train_val_labels": 282,
            "sealed_posthoc_labels": 30,
            "root_scope": [200002],
            "exact_feature_leakage_across_splits": False,
        },
        "training": {
            "force_support_N": [0.5, 15.0],
            "force_feature_normalization_N": 15.0,
            "ensemble_members": 3,
            "selected_epochs": [checkpoint["selected_epoch"] for checkpoint in manifest["checkpoints"]],
            "validation_member_nll": [checkpoint["validation_posterior_nll"] for checkpoint in manifest["checkpoints"]],
            "belief": "frozen_v4",
            "realized_squeeze_calibration": "TRAIN-only, utility-only",
            "model_utility_locked_before_test_access": True,
        },
        "heldout_posthoc": {
            "contexts": 6,
            "branch_labels": 30,
            "success_prevalence": prevalence,
            "branch_nll": nll,
            "branch_brier": brier,
            "branch_auc": auc,
            "constant_prevalence_nll": constant_nll,
            "constant_prevalence_brier": constant_brier,
            "selected_forces_N": selected_forces,
            "selected_successes": int(sum(selected)),
            "fixed8_successes": int(sum(fixed[8.0])),
            "fixed12_successes": int(sum(fixed[12.0])),
            "selected_vs_fixed8": paired_counts(selected, fixed[8.0]),
            "selected_vs_fixed12": paired_counts(selected, fixed[12.0]),
            "realized_calibration_mae_N": float(np.mean(calibration_errors)),
            "runtime_replay_max_probability_error": max_probability_error,
        },
        "scientific_conclusion": {
            "activeforcing_repair_supported": repair_supported,
            "conclusion": "NOT_SUPPORTED: v5r1 selects 15 N for every held-out context, achieves 2/6 (the same as fixed 8 N and fixed 12 N), and its held-out branch ranking/calibration is weak (AUC near chance; NLL and Brier worse than the constant-prevalence baseline).",
            "what_was_fixed": "The candidate range, force normalization, and force-cost realization mismatch were repaired without changing the frozen v4 belief.",
            "remaining_failure": "The feasibility model learned an almost uniformly increasing force-success curve and did not recover the observed non-monotonic, context-dependent landscape.",
            "claim_boundary": "Root-local posthoc diagnostic only; the 6 contexts are context-held-out from fitting but not a fresh confirmation set because aggregate outcomes were inspected before protocol freeze.",
        },
        "hashes": {
            "dataset_freeze_lock_sha256": sha(DATASET / "FREEZE_LOCK.json"),
            "training_protocol_sha256": protocol_sha,
            "model_utility_lock_sha256": model_lock_sha,
            "posthoc_results_sha256": results_sha,
            "training_complete_sha256": sha(MODELS / "TRAINING_COMPLETE.json"),
        },
    }
    output = HERE / "INDEPENDENT_AUDIT.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
