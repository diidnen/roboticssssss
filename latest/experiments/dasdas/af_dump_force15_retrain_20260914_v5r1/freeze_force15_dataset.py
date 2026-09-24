"""Freeze the audited 24-context force-range sweep for v5 retraining."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np


BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
SWEEP = BASE / "experiments/af_dump_fixed_force_5_15_20260914_v3"
OLD = BASE / "experiments/af_dump_original_restore_20260912"
HERE = Path(__file__).resolve().parent
OUT = HERE / "dataset_v5"
FORCES = [5.0, 8.0, 10.0, 12.0, 15.0]
TRAIN_SEEDS = set(range(80200002, 80200006))
VAL_SEEDS = set(range(80200006, 80200008))
TEST_SEEDS = set(range(80200008, 80200010))

sys.path.insert(0, str(OLD))
from native_original_evidence_adapter import original_evidence


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(Path(path).read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sequence_hash(feature: dict) -> str:
    payload = json.dumps(feature["sequence"], separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def split_for(seed: int) -> str:
    if seed in TRAIN_SEEDS:
        return "TRAIN"
    if seed in VAL_SEEDS:
        return "VAL"
    if seed in TEST_SEEDS:
        return "POSTHOC_TEST"
    raise ValueError(f"Unplanned policy seed {seed}")


def verify_hashed_sources(report: dict, inputs: dict[str, str]) -> None:
    for name, digest in report.get("source_hashes", {}).items():
        path = Path(name)
        if sha(path) != digest:
            raise ValueError("Audited source changed: " + name)
        inputs[str(path)] = digest


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen dataset: {OUT}")
    status = read(SWEEP / "STATUS.json")
    final = read(SWEEP / "FINAL_RESULTS.json")
    contexts = read(SWEEP / "plan/MAIN_CONTEXTS.json")
    if status.get("status") != "complete" or status.get("completed_contexts") != 24:
        raise ValueError("Sweep is not complete")
    if final.get("contexts") != 24 or final.get("paired_rollouts") != 120 or len(contexts) != 24:
        raise ValueError("Frozen sweep denominator changed")
    if [Path(p).stem for p in final["records"]] != [c["id"] for c in contexts]:
        raise ValueError("Final record schedule differs from frozen context plan")

    OUT.mkdir(parents=True)
    inputs: dict[str, str] = {}
    entries = []
    current_feature_hashes: dict[str, str] = {}
    split_counts = {"TRAIN": 0, "VAL": 0, "POSTHOC_TEST": 0}
    label_counts = {"TRAIN": 0, "VAL": 0, "POSTHOC_TEST": 0}

    for context in contexts:
        cid = context["id"]
        split = split_for(int(context["policy_seed"]))
        case = SWEEP / "main_raw" / cid
        job = case / "job"
        compact_path = SWEEP / "main_records" / f"{cid}.json"
        context_audit_path = case / "INDEPENDENT_CONTEXT_AUDIT.json"
        compact = read(compact_path)
        audit = read(context_audit_path)
        if not audit.get("passed") or not compact.get("strict_matched") or not compact.get("first_chunk_pairing_exact"):
            raise ValueError("Context audit/pairing failed: " + cid)
        if compact["context"] != context or audit["context"] != context:
            raise ValueError("Context metadata mismatch: " + cid)
        if sorted(float(row["force_N"]) for row in compact["outcomes"]) != FORCES:
            raise ValueError("Force coverage mismatch: " + cid)
        if not all(row.get("raw_audit_passed") for row in compact["outcomes"]):
            raise ValueError("A failed raw branch entered compact record: " + cid)

        feature_path = job / "PREACTION_FEATURE.json"
        feature = read(feature_path)
        sequence = np.asarray(feature.get("sequence"), dtype=np.float32)
        if sequence.shape != (8, 64) or not np.isfinite(sequence).all():
            raise ValueError("Invalid feasibility feature shape/value: " + cid)
        if feature.get("candidate_actions_executed") != 0:
            raise ValueError("Post-action feature leakage: " + cid)
        digest = sequence_hash(feature)
        if digest in current_feature_hashes:
            raise ValueError(f"Exact feature duplicate: {cid} and {current_feature_hashes[digest]}")
        current_feature_hashes[digest] = cid

        query = job / "query"
        query_audit_path = query / "INDEPENDENT_NATIVE_FORCE_REBUILD.json"
        query_audit = read(query_audit_path)
        if not query_audit.get("passed"):
            raise ValueError("Query audit failed: " + cid)
        verify_hashed_sources(query_audit, inputs)
        raw_path = query / "original_raw_rows.json"
        patch_path = query / "patch_readbacks.json"
        npy_path = query / "original58_engineering.npy"
        rebuilt = original_evidence(read(raw_path), read(patch_path))
        np.testing.assert_array_equal(rebuilt, np.load(npy_path, allow_pickle=False))

        labels = []
        audit_branches = {float(row["force_N"]): row for row in audit["branches"]}
        for outcome in sorted(compact["outcomes"], key=lambda row: float(row["force_N"])):
            force = float(outcome["force_N"])
            branch_audit = audit_branches[force]
            if not branch_audit["raw_audit"].get("passed"):
                raise ValueError(f"Branch audit failed: {cid} {force:g}N")
            branch_dir = next(iter(job.glob(f"branch_*_{force:g}N")), None)
            if branch_dir is None:
                raise ValueError(f"Missing branch directory: {cid} {force:g}N")
            independent_path = branch_dir / "INDEPENDENT_BRANCH_AUDIT.json"
            independent = read(independent_path)
            if not independent.get("passed"):
                raise ValueError(f"Independent branch audit failed: {cid} {force:g}N")
            verify_hashed_sources(independent, inputs)
            result_path = branch_dir / "result.json"
            result = read(result_path)
            if float(result["force_setpoint_bilateral_n"]) != force:
                raise ValueError("Commanded-force label mismatch")
            if int(result["success"]) != int(result["official_final_check"]) or int(result["success"]) != int(outcome["success"]):
                raise ValueError("Official success label mismatch")
            labels.append(
                {
                    "context_id": cid,
                    "split": split,
                    "root": 200002,
                    "force": force,
                    "full_task_success_y": int(outcome["success"]),
                    "target_contact_mean_squeeze_N": outcome["target_contact_mean_squeeze_N"],
                    "target_contact_fraction": outcome["target_contact_fraction"],
                    "result_path": str(result_path),
                    "result_sha256": sha(result_path),
                    "branch_audit_path": str(independent_path),
                    "branch_audit_sha256": sha(independent_path),
                    "job": str(branch_dir),
                }
            )

        label_dir = "sealed_test_labels" if split == "POSTHOC_TEST" else "train_val_labels"
        label_path = OUT / label_dir / f"{cid}.json"
        write(label_path, labels)
        input_paths = [feature_path, raw_path, patch_path, npy_path, query_audit_path, context_audit_path, compact_path]
        for path in input_paths:
            inputs[str(path)] = sha(path)
        entries.append(
            {
                "id": cid,
                "root": 200002,
                "task": "dump_bin_bigbin",
                "friction": float(context["friction"]),
                "policy_seed": int(context["policy_seed"]),
                "split": split,
                "feature_sequence_sha256": digest,
                "feature_path": str(feature_path),
                "query_raw_path": str(raw_path),
                "patch_path": str(patch_path),
                "original58_path": str(npy_path),
                "label_file": str(label_path),
                "label_file_sha256": sha(label_path),
                "labels_sealed_until_model_utility_lock": split == "POSTHOC_TEST",
            }
        )
        split_counts[split] += 1
        label_counts[split] += len(labels)

    old_features = list((OLD / "original_rootlocal_dataset_v3_rim20").glob(
        "groups/*/attempt_*/job/branch_*/original_motion_feature.json"
    ))
    old_hashes = {sequence_hash(read(path)) for path in old_features}
    overlap = old_hashes.intersection(current_feature_hashes)
    if overlap:
        raise ValueError("New features exactly overlap the old dataset")
    if split_counts != {"TRAIN": 12, "VAL": 6, "POSTHOC_TEST": 6}:
        raise ValueError("Context split count mismatch")
    if label_counts != {"TRAIN": 60, "VAL": 30, "POSTHOC_TEST": 30}:
        raise ValueError("Label split count mismatch")

    index = {
        "version": "AF_DUMP_FORCE15_DATASET_V5",
        "created_utc": now(),
        "root_scope": [200002],
        "task": "dump_bin_bigbin",
        "context_grain_split": True,
        "split_rule": {
            "TRAIN_policy_seeds": sorted(TRAIN_SEEDS),
            "VAL_policy_seeds": sorted(VAL_SEEDS),
            "POSTHOC_TEST_policy_seeds": sorted(TEST_SEEDS),
        },
        "force_set_N": FORCES,
        "contexts": entries,
        "source_sweep_final_sha256": sha(SWEEP / "FINAL_RESULTS.json"),
        "source_sweep_plan_sha256": sha(SWEEP / "plan/MAIN_CONTEXTS.json"),
        "source_sweep_lock_sha256": sha(SWEEP / "plan/FREEZE_LOCK.json"),
        "input_hashes": inputs,
        "test_status": "POSTHOC_NOT_FRESH_CONFIRMATION",
    }
    profile = {
        "completed": True,
        "contexts": len(entries),
        "labels": sum(label_counts.values()),
        "split_contexts": split_counts,
        "split_labels": label_counts,
        "feature_shape": [8, 64],
        "unique_new_feature_hashes": len(current_feature_hashes),
        "old_unique_feature_hashes": len(old_hashes),
        "cross_dataset_exact_feature_overlap": 0,
        "all_branch_audits_passed": True,
        "measured_squeeze_is_feasibility_input": False,
        "claim_boundary": "root-local posthoc task-form adaptation; not fresh confirmation",
    }
    write(OUT / "DATASET_INDEX.json", index)
    write(OUT / "DATASET_PROFILE.json", profile)
    lock = {
        "version": "AF_DUMP_FORCE15_DATASET_V5_LOCK",
        "dataset_index_sha256": sha(OUT / "DATASET_INDEX.json"),
        "dataset_profile_sha256": sha(OUT / "DATASET_PROFILE.json"),
        "freeze_script_sha256": sha(Path(__file__)),
        "train_val_label_hashes": {
            path.name: sha(path) for path in sorted((OUT / "train_val_labels").glob("*.json"))
        },
        "sealed_test_label_hashes": {
            path.name: sha(path) for path in sorted((OUT / "sealed_test_labels").glob("*.json"))
        },
    }
    write(OUT / "FREEZE_LOCK.json", lock)
    print(json.dumps(profile, indent=2), flush=True)


if __name__ == "__main__":
    main()
