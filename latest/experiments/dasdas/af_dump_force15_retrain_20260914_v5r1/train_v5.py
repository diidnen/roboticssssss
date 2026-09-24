"""Train and audit the root-local force-15 ActiveForcing feasibility ensemble."""
from __future__ import annotations

import ast
from collections import defaultdict
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
OLD = BASE / "experiments/af_dump_original_restore_20260912"
V4 = BASE / "experiments/af_dump_maxf8_20260913"
DATASET = HERE / "dataset_v5"
sys.path.insert(0, str(V4))
sys.path.insert(0, str(OLD))

import train_original_rootlocal as original
import train_v4 as v4
from native_original_evidence_adapter import original_evidence
from current_contract_physical_belief import verify_decision_prefix
from rootlocal_collection_contract import now, read, sha, write
from force15_runtime import Force15Feasibility
from force15_utility import UTILITY_DEFINITION, expected_utility


FORCE_NORMALIZATION_N = 15.0
FORCE_SUPPORT = [0.5, 15.0]
OBSERVED_NEW_FORCES = [5.0, 8.0, 10.0, 12.0, 15.0]


def verify_dataset_lock() -> tuple[dict, dict]:
    lock = read(DATASET / "FREEZE_LOCK.json")
    if lock["dataset_index_sha256"] != sha(DATASET / "DATASET_INDEX.json"):
        raise ValueError("Dataset index changed")
    if lock["dataset_profile_sha256"] != sha(DATASET / "DATASET_PROFILE.json"):
        raise ValueError("Dataset profile changed")
    if lock["freeze_script_sha256"] != sha(HERE / "freeze_force15_dataset.py"):
        raise ValueError("Dataset freezer changed")
    index = read(DATASET / "DATASET_INDEX.json")
    if index.get("version") != "AF_DUMP_FORCE15_DATASET_V5" or len(index.get("contexts", [])) != 24:
        raise ValueError("Unexpected force-15 dataset")
    return index, lock


def load_old() -> tuple[list[dict], dict, list[dict]]:
    previous, inputs, old_lock = original.load_dataset(OLD / "original_rootlocal_dataset_v3_rim20")
    added, added_inputs, added_lock = v4.load_added()
    inputs.update(added_inputs)
    records = previous + added
    if sum(len(record["branches"]) for record in records if record["branches"]) != 192:
        raise ValueError("Existing v4 label denominator changed")
    return records, inputs, [old_lock, added_lock]


def load_new(*, include_test_labels: bool) -> tuple[list[dict], dict, dict]:
    index, lock = verify_dataset_lock()
    records = []
    inputs = {
        str(DATASET / "DATASET_INDEX.json"): sha(DATASET / "DATASET_INDEX.json"),
        str(DATASET / "DATASET_PROFILE.json"): sha(DATASET / "DATASET_PROFILE.json"),
        str(DATASET / "FREEZE_LOCK.json"): sha(DATASET / "FREEZE_LOCK.json"),
    }
    for entry in index["contexts"]:
        split = entry["split"]
        feature_path = Path(entry["feature_path"])
        raw_path = Path(entry["query_raw_path"])
        patch_path = Path(entry["patch_path"])
        npy_path = Path(entry["original58_path"])
        for path in [feature_path, raw_path, patch_path, npy_path]:
            expected = index["input_hashes"][str(path)]
            if sha(path) != expected:
                raise ValueError("Frozen new input changed: " + str(path))
            inputs[str(path)] = expected
        feature = read(feature_path)
        sequence = np.asarray(feature["sequence"], np.float32)
        if sequence.shape != (8, 64) or feature.get("candidate_actions_executed") != 0:
            raise ValueError("Invalid/leaky new pre-action feature")
        raw_rows = read(raw_path)
        patch_rows = read(patch_path)
        verify_decision_prefix(raw_rows)
        values = original_evidence(raw_rows, patch_rows)
        np.testing.assert_array_equal(values, np.load(npy_path, allow_pickle=False))
        branches = []
        if split != "POSTHOC_TEST" or include_test_labels:
            label_path = Path(entry["label_file"])
            if sha(label_path) != entry["label_file_sha256"]:
                raise ValueError("Frozen label file changed")
            labels = read(label_path)
            if len(labels) != 5 or sorted(float(row["force"]) for row in labels) != OBSERVED_NEW_FORCES:
                raise ValueError("New label coverage mismatch")
            for row in labels:
                if row["context_id"] != entry["id"] or row["split"] != split or row["root"] != 200002:
                    raise ValueError("New label context/split mismatch")
                if row["full_task_success_y"] not in (0, 1):
                    raise ValueError("Invalid success target")
            branches = labels
            inputs[str(label_path)] = entry["label_file_sha256"]
        records.append(
            {
                "id": entry["id"],
                "root": 200002,
                "split": split,
                "y": float(entry["friction"]),
                "raw": values,
                "query_raw": raw_rows,
                "patch": patch_rows,
                "branches": branches,
                "feature": feature,
                "reference": str(feature_path.parent),
                "context": entry,
                "source": "FORCE15_SWEEP_V3",
            }
        )
    return records, inputs, lock


def attach_frozen_v4_belief(records: list[dict], runtime_manifest_sha256: str) -> tuple[str, str]:
    manifest_path = V4 / "models_v4/belief/BELIEF_MANIFEST.json"
    complete = read(V4 / "models_v4/TRAINING_COMPLETE.json")
    manifest_sha256 = sha(manifest_path)
    if complete.get("belief_manifest_sha256") != manifest_sha256:
        raise ValueError("Frozen v4 belief manifest hash mismatch")
    belief = original.load_native_belief(manifest_path, manifest_sha256=manifest_sha256)
    for record in records:
        value = belief.rows(
            record["query_raw"],
            record["patch"],
            runtime_manifest_sha256=runtime_manifest_sha256,
        )
        value.update(candidate_actions_executed=0)
        record["posterior"] = value
    return str(manifest_path), manifest_sha256


def feasibility_rows(records, split):
    rows = []
    for record in records:
        if record["split"] != split:
            continue
        base = np.asarray(record["feature"]["sequence"], np.float32)
        for branch in record["branches"]:
            values = base.copy()
            values[:, 10] = float(branch["force"]) / FORCE_NORMALIZATION_N
            values[:, 11] = record["y"]
            rows.append(
                {
                    "record": record,
                    "branch": branch,
                    "x": values,
                    "y": float(branch["full_task_success_y"]),
                }
            )
    return rows


def posterior_predictions(models, rows, mean, std, device):
    import torch

    by_context = defaultdict(list)
    for row in rows:
        by_context[row["record"]["id"]].append(row)
    predictions = {}
    for context_id, group in by_context.items():
        record = group[0]["record"]
        posterior = record["posterior"]
        nodes = np.asarray(posterior["integration_nodes"], np.float32)
        weights = np.asarray(posterior["integration_weights"], float)
        forces = np.asarray([row["branch"]["force"] for row in group], np.float32)
        force = np.repeat(forces, len(nodes))
        mu = np.tile(nodes, len(forces))
        base = np.asarray(record["feature"]["sequence"], np.float32)
        values = np.broadcast_to(base, (len(force),) + base.shape).copy()
        values[:, :, 10] = force[:, None] / FORCE_NORMALIZATION_N
        values[:, :, 11] = mu[:, None]
        values = (values - mean[None, None]) / std[None, None]
        member_values = []
        for model in models:
            model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, len(values), 4096):
                    tensor = torch.as_tensor(values[start : start + 4096], dtype=torch.float32, device=device)
                    chunks.append(torch.sigmoid(model(tensor[:, :, :10], tensor[:, 0, 10:])).cpu().numpy())
            member_values.append(np.concatenate(chunks))
        marginal = np.mean(member_values, axis=0).reshape(len(forces), len(nodes)) @ weights
        for row, probability in zip(group, marginal):
            predictions[context_id, float(row["branch"]["force"])] = float(probability)
    return np.asarray(
        [predictions[row["record"]["id"], float(row["branch"]["force"])] for row in rows]
    )


def fit_realized_calibration(new_records: list[dict], out: Path) -> dict:
    raw_medians = []
    counts = []
    for force in OBSERVED_NEW_FORCES:
        values = [
            float(branch["target_contact_mean_squeeze_N"])
            for record in new_records
            if record["split"] == "TRAIN"
            for branch in record["branches"]
            if float(branch["force"]) == force and branch["target_contact_mean_squeeze_N"] is not None
        ]
        if len(values) < 6:
            raise ValueError(f"Insufficient TRAIN contact calibration at {force:g}N")
        raw_medians.append(float(np.median(values)))
        counts.append(len(values))
    monotone = np.maximum.accumulate(np.asarray(raw_medians, float))
    monotone = np.clip(monotone, 0, 15)
    calibration = {
        "version": "TRAIN_ONLY_MONOTONE_REALIZED_SQUEEZE_V1",
        "created_utc": now(),
        "fit_split": "TRAIN",
        "fit_contexts": sum(record["split"] == "TRAIN" for record in new_records),
        "source": "target_contact_mean_squeeze_N",
        "source_condition": "measured only on target-contact steps; all eligible TRAIN branches retained",
        "command_knots_N": [0.5] + OBSERVED_NEW_FORCES,
        "raw_train_medians_N": [0.5] + raw_medians,
        "realized_squeeze_knots_N": [0.5] + monotone.tolist(),
        "observations_per_new_force": [0] + counts,
        "low_force_anchor": "0.5 N identity anchor; linear interpolation to observed 5 N TRAIN median",
        "isotonic_rule": "cumulative maximum over ordered observed command knots",
        "feasibility_input": False,
        "utility_only": True,
        "engineering_scale_N": 15.0,
    }
    write(out, calibration)
    return calibration


def extended_feasibility(calibration_path: Path):
    tree = ast.parse(inspect.getsource(original.train_feasibility))
    baseline = ast.dump(tree)
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 81:
            changes += 1
            node.value = 401
        if isinstance(node, ast.Dict):
            keys = [key.value if isinstance(key, ast.Constant) else None for key in node.keys]
            for index, key in enumerate(keys):
                if key == "utility_normalization_N":
                    changes += 1
                    node.values[index] = ast.Constant(15.0)
                elif key == "force_support":
                    value = node.values[index]
                    if not isinstance(value, ast.List) or len(value.elts) != 2:
                        raise ValueError("Unexpected force-support AST")
                    changes += 1
                    value.elts[1] = ast.Constant(15.0)
    if changes != 4:
        raise ValueError(f"Unexpected feasibility AST change count: {changes}")
    executable = deepcopy(tree)
    if ast.dump(ast.parse(inspect.getsource(original.train_feasibility))) != baseline:
        raise ValueError("Unexpected original feasibility recipe change")
    namespace = dict(original.__dict__)
    namespace["feas_rows"] = feasibility_rows
    namespace["posterior_predictions"] = posterior_predictions
    namespace["NativeOriginalFeasibility"] = Force15Feasibility

    original_write = write

    def write_with_force15_contract(path, value):
        if Path(path).name == "FEASIBILITY_MANIFEST.json":
            value = deepcopy(value)
            value.update(
                force_support=FORCE_SUPPORT,
                force_feature_normalization_N=15.0,
                utility_normalization_N=15.0,
                utility_engineering_scale_N=15.0,
                realized_force_calibration_path=str(calibration_path),
                realized_force_calibration_sha256=sha(calibration_path),
                utility_definition=UTILITY_DEFINITION,
            )
            for source in [HERE / "force15_runtime.py", HERE / "force15_utility.py", OLD / "native_original_feasibility_binding.py"]:
                value["source_hashes"][str(source)] = sha(source)
        elif Path(path).name == "RUNTIME_RELOAD_QUALIFICATION.json":
            value = deepcopy(value)
            value.update(
                force_feature_normalization_N=15.0,
                realized_force_calibration_sha256=sha(calibration_path),
                utility_definition=UTILITY_DEFINITION,
            )
        original_write(path, value)

    namespace["write"] = write_with_force15_contract
    exec(compile(ast.fix_missing_locations(executable), str(Path(__file__)), "exec"), namespace)
    return namespace["train_feasibility"]


def auc_score(y, p) -> float | None:
    positives = [score for target, score in zip(y, p) if target == 1]
    negatives = [score for target, score in zip(y, p) if target == 0]
    if not positives or not negatives:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in positives for b in negatives)
    return float(wins / (len(positives) * len(negatives)))


def paired_counts(candidate, baseline) -> dict:
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


def evaluate_posthoc_test(records: list[dict], runtime: Force15Feasibility, out: Path) -> dict:
    targets, probabilities = [], []
    selected, fixed = [], {force: [] for force in OBSERVED_NEW_FORCES}
    decisions = []
    calibration_errors = []
    for record in records:
        if record["split"] != "POSTHOC_TEST":
            continue
        curve = runtime.curve(record["feature"]["sequence"], record["posterior"])
        observed_indices = [int(np.flatnonzero(np.isclose(runtime.force_grid, force, atol=1e-8, rtol=0))[0]) for force in OBSERVED_NEW_FORCES]
        observed_p = np.asarray([curve[index] for index in observed_indices])
        observed_grid = np.asarray(OBSERVED_NEW_FORCES)
        observed_utility = expected_utility(observed_p, observed_grid, runtime.realized_calibration)
        chosen_index = int(np.argmax(observed_utility))
        chosen_force = float(observed_grid[chosen_index])
        outcomes = {float(branch["force"]): int(branch["full_task_success_y"]) for branch in record["branches"]}
        selected.append(outcomes[chosen_force])
        for force in OBSERVED_NEW_FORCES:
            fixed[force].append(outcomes[force])
        for force, probability, index in zip(OBSERVED_NEW_FORCES, observed_p, observed_indices):
            targets.append(outcomes[force])
            probabilities.append(float(probability))
            branch = next(row for row in record["branches"] if float(row["force"]) == force)
            if branch["target_contact_mean_squeeze_N"] is not None:
                predicted_realized = float(np.interp(force, runtime.realized_calibration["command_knots_N"], runtime.realized_calibration["realized_squeeze_knots_N"]))
                calibration_errors.append(abs(predicted_realized - float(branch["target_contact_mean_squeeze_N"])))
        decisions.append(
            {
                "context_id": record["id"],
                "friction": record["y"],
                "selected_observed_force_N": chosen_force,
                "selected_success": outcomes[chosen_force],
                "observed_force_probabilities": dict(zip(map(str, OBSERVED_NEW_FORCES), map(float, observed_p))),
                "observed_force_utilities": dict(zip(map(str, OBSERVED_NEW_FORCES), map(float, observed_utility))),
                "continuous_runtime_selection": runtime.select(record["feature"], record["posterior"]),
            }
        )
    y = np.asarray(targets, float)
    p = np.clip(np.asarray(probabilities, float), 1e-7, 1 - 1e-7)
    result = {
        "completed": True,
        "evaluation_scope": "POSTHOC_INTERNAL_TEST_NOT_FRESH_CONFIRMATION",
        "contexts": len(selected),
        "branch_labels": len(targets),
        "branch_nll": float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))),
        "branch_brier": float(np.mean((p - y) ** 2)),
        "branch_auc": auc_score(y.tolist(), p.tolist()),
        "selected_observed_force_successes": int(sum(selected)),
        "selected_observed_force_success_rate": float(np.mean(selected)),
        "fixed_successes": {str(force): int(sum(values)) for force, values in fixed.items()},
        "selected_vs_fixed8": paired_counts(selected, fixed[8.0]),
        "selected_vs_fixed12": paired_counts(selected, fixed[12.0]),
        "realized_calibration_mae_N": float(np.mean(calibration_errors)),
        "decisions": decisions,
        "claim_boundary": "posthoc context-held-out diagnostic; fresh same-root confirmation still required",
        "finished_utc": now(),
    }
    write(out, result)
    return result


def main() -> None:
    out = HERE / "models_v5"
    if out.exists():
        raise FileExistsError("Refusing to overwrite models_v5")
    old_records, inputs, old_locks = load_old()
    new_records, new_inputs, new_lock = load_new(include_test_labels=False)
    inputs.update(new_inputs)
    records = old_records + new_records
    labeled_train_val = [record for record in records if record["branches"]]
    if sum(len(record["branches"]) for record in labeled_train_val) != 282:
        raise ValueError("Expected 192 old + 90 new TRAIN/VAL labels")
    if sum(record["split"] == "POSTHOC_TEST" for record in records) != 6:
        raise ValueError("Expected six sealed posthoc test contexts")

    query_hashes = {}
    feature_splits = {}
    for record in records:
        key = hashlib.sha256(record["raw"].tobytes()).hexdigest()
        query_hashes[record["id"]] = key
        if record["feature"] is not None:
            feature_key = hashlib.sha256(
                np.asarray(record["feature"]["sequence"], np.float32).tobytes()
            ).hexdigest()
            feature_splits.setdefault(feature_key, set()).add(record["split"])
    if any(len(splits) > 1 for splits in feature_splits.values()):
        raise ValueError("Exact feasibility feature duplicate crosses split")

    out.mkdir()
    calibration_path = out / "REALIZED_FORCE_CALIBRATION.json"
    fit_realized_calibration(new_records, calibration_path)
    protocol = {
        "version": "AF_DUMP_FORCE15_RETRAIN_V5R1",
        "created_utc": now(),
        "root_scope": [200002],
        "task": "dump_bin_bigbin",
        "old_data_locks": old_locks,
        "new_dataset_lock": new_lock,
        "input_hashes_pre_test": inputs,
        "query_deduplication": query_hashes,
        "unique_belief_queries": len(set(query_hashes.values())),
        "belief_intervention": "NONE_REUSE_FROZEN_V4",
        "repeated_belief_queries_are_not_retrained": True,
        "feasibility_labels_train_val": 282,
        "sealed_posthoc_test_labels": 30,
        "belief_architecture": [58, 16, 16],
        "feasibility_feature_shape": [8, 64],
        "belief_max_updates": 400,
        "feasibility_max_epochs": 400,
        "seeds": [0, 1, 2],
        "selection": "per seed first minimum VAL NLL",
        "force_support": FORCE_SUPPORT,
        "force_feature_normalization_N": 15.0,
        "planner_grid_step": 0.05,
        "utility_definition": UTILITY_DEFINITION,
        "utility_engineering_scale_N": 15.0,
        "realized_force_calibration_path": str(calibration_path),
        "realized_force_calibration_sha256": sha(calibration_path),
        "measured_squeeze_used_by_feasibility": False,
        "test_labels_accessed_before_model_utility_lock": False,
        "posthoc_test_not_fresh_confirmation": True,
        "source_hashes": {
            str(path): sha(path)
            for path in [
                Path(__file__),
                HERE / "force15_runtime.py",
                HERE / "force15_utility.py",
                HERE / "freeze_force15_dataset.py",
                OLD / "train_original_rootlocal.py",
                V4 / "train_v4.py",
            ]
        },
    }
    write(out / "TRAINING_PROTOCOL.json", protocol)
    write(out / "TRAINING_PROTOCOL_LOCK.json", {"sha256": sha(out / "TRAINING_PROTOCOL.json")})

    runtime_manifest_sha256 = old_locks[1]["runtime_sha256"]
    belief_manifest_path, belief_manifest_sha256 = attach_frozen_v4_belief(
        records, runtime_manifest_sha256
    )
    extended_feasibility(calibration_path)(labeled_train_val, out / "feasibility", out / "TRAINING_PROTOCOL.json")
    for path, digest in inputs.items():
        if sha(Path(path)) != digest:
            raise ValueError("Pre-test training input changed: " + path)

    manifest_path = out / "feasibility/FEASIBILITY_MANIFEST.json"
    model_lock = {
        "version": "AF_DUMP_FORCE15_MODEL_UTILITY_LOCK_V1R1",
        "created_utc": now(),
        "training_protocol_sha256": sha(out / "TRAINING_PROTOCOL.json"),
        "belief_manifest_path": belief_manifest_path,
        "belief_manifest_sha256": belief_manifest_sha256,
        "feasibility_manifest_sha256": sha(manifest_path),
        "realized_force_calibration_sha256": sha(calibration_path),
        "utility_definition": UTILITY_DEFINITION,
        "test_labels_accessed": False,
    }
    write(out / "MODEL_UTILITY_LOCK.json", model_lock)

    test_records, test_inputs, _ = load_new(include_test_labels=True)
    test_by_id = {record["id"]: record for record in test_records if record["split"] == "POSTHOC_TEST"}
    for record in records:
        if record["split"] == "POSTHOC_TEST":
            loaded = test_by_id[record["id"]]
            if not np.array_equal(record["raw"], loaded["raw"]) or record["feature"] != loaded["feature"]:
                raise ValueError("Post-lock test input identity mismatch")
            record["branches"] = loaded["branches"]
    runtime = Force15Feasibility(manifest_path, manifest_sha256=sha(manifest_path), device="cuda")
    result = evaluate_posthoc_test(records, runtime, out / "POSTHOC_TEST_RESULTS.json")
    test_label_inputs = {
        path: digest
        for path, digest in test_inputs.items()
        if "sealed_test_labels" in path
    }
    for path, digest in test_label_inputs.items():
        if sha(Path(path)) != digest:
            raise ValueError("Test label changed after access")
    write(
        out / "TEST_ACCESS_RECEIPT.json",
        {
            "model_utility_lock_sha256": sha(out / "MODEL_UTILITY_LOCK.json"),
            "post_lock_test_label_hashes": test_label_inputs,
            "test_contexts": 6,
            "test_labels": 30,
            "accessed_utc": now(),
        },
    )
    write(
        out / "TRAINING_COMPLETE.json",
        {
            "completed": True,
            "training_protocol_sha256": sha(out / "TRAINING_PROTOCOL.json"),
            "model_utility_lock_sha256": sha(out / "MODEL_UTILITY_LOCK.json"),
            "posthoc_test_results_sha256": sha(out / "POSTHOC_TEST_RESULTS.json"),
            "posthoc_selected_success_rate": result["selected_observed_force_success_rate"],
            "fresh_confirmation_required": True,
            "finished_utc": now(),
        },
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
