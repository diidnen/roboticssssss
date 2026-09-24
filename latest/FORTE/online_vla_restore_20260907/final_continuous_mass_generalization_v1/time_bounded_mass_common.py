"""Shared immutable-contract checks for TIME_BOUNDED_MASS_SUBSET_V1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SUBSET_PATH = HERE / "TIME_BOUNDED_MASS_SUBSET_V1.json"
SUBSET_HASH_PATH = HERE / "TIME_BOUNDED_MASS_SUBSET_V1_SHA256.txt"


def read(path: Path):
    return json.loads(Path(path).read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_subset():
    expected = SUBSET_HASH_PATH.read_text().split()[0]
    actual = sha256(SUBSET_PATH)
    if actual != expected:
        raise RuntimeError(f"time-bounded subset hash mismatch: {actual} != {expected}")
    subset = read(SUBSET_PATH)
    if subset.get("schema") != "TIME_BOUNDED_MASS_SUBSET_V1" or len(subset.get("jobs", [])) != 216:
        raise RuntimeError("invalid time-bounded subset schema or cardinality")
    for relative, expected_hash in subset["authoritative_original_artifact_sha256"].items():
        path = HERE / relative
        if sha256(path) != expected_hash:
            raise RuntimeError(f"authoritative original artifact changed: {path}")
    return subset, actual


def inertia_matches(material) -> bool:
    nominal = np.asarray(material.get("nominal_inertia", []), dtype=float)
    actual = np.asarray(material.get("actual_inertia", []), dtype=float)
    ratio = material.get("mass_ratio")
    return bool(
        nominal.size and nominal.shape == actual.shape and ratio is not None
        and np.isfinite(nominal).all() and np.isfinite(actual).all()
        and np.allclose(actual, nominal * float(ratio), rtol=3e-6, atol=1e-12)
    )


def validate_branch(job: Path, spec: dict, runtime: dict, reference: Path) -> dict:
    required = {
        name: job / name for name in (
            "LAUNCH_CLAIM.json", "WORKER_COMPLETION.json", "BRANCH_RESULT.json",
            "PREACTION_STATE_COMPARISON.json", "PREACTION_SEQUENCE.npy",
            "SOURCE_HASHES_BEFORE.json", "SOURCE_HASHES_AFTER.json",
            "MASS_INTERVENTION_READBACK.json", "RUNTIME_CONFIG.json",
        )
    }
    reference_sequence = reference / "PREACTION_SEQUENCE.npy"
    missing = [name for name, path in required.items() if not path.is_file()]
    if not reference_sequence.is_file():
        missing.append("reference/PREACTION_SEQUENCE.npy")
    if missing:
        return {"valid": False, "missing": missing, "failed_checks": ["required_files"]}

    completion = read(required["WORKER_COMPLETION.json"])
    result = read(required["BRANCH_RESULT.json"])
    state = read(required["PREACTION_STATE_COMPARISON.json"])
    before = read(required["SOURCE_HASHES_BEFORE.json"])
    after = read(required["SOURCE_HASHES_AFTER.json"])
    material = read(required["MASS_INTERVENTION_READBACK.json"])
    claim = read(required["LAUNCH_CLAIM.json"])
    sequence = np.load(required["PREACTION_SEQUENCE.npy"], allow_pickle=False)
    reference_sha = sha256(reference_sequence)
    force = float(spec["force_N"])
    context_id = spec["context_id"]
    plan_lookup = {row["id"]: row for row in read(HERE / "MASS_TRAINING_CONTEXT_PLAN.json")["contexts"]}
    context = plan_lookup[context_id]
    outcome = result.get("outcome", {})
    label = result.get("full_task_success_y")
    steps = result.get("steps")
    early_proof = outcome.get("proof", {})
    settled_endpoint = (
        steps == 350
        and outcome.get("unchanged_success_predicate_version")
        == "PROSPECTIVE_WHOLE_MESH_CONTAINMENT_RELEASE_FULLTASK_V1"
    )
    proven_early_terminal = (
        isinstance(steps, int) and 0 < steps < 350 and label == 0
        and outcome.get("label_version") == runtime["label_version"]
        and early_proof.get("proven") is True
        and early_proof.get("safe_stop_after_this_observation") is True
        and early_proof.get("contract") == runtime["label_version"]
    )

    checks = {
        "logical_success": completion.get("logical_success") is True,
        "branch_passed": result.get("passed") is True,
        "claim_context_exact": claim.get("context_id") == context_id,
        "claim_force_exact": float(claim.get("force")) == force,
        "claim_runtime_manifest_exact": claim.get("runtime_manifest_sha256") == read(HERE / "MASS_TRAINING_PROTOCOL.json")["runtime_manifest_sha256"],
        "context_plan_exact": result.get("plan") == context,
        "candidate_force_exact": float(result.get("candidate_F")) == force,
        "candidate_state_exact": state.get("passed") is True and result.get("candidate_preact_state_equality") is True,
        "all_exposed_state_groups_exact": all(state.get("exposed_state_equality", {}).values()),
        "no_candidate_action_before_branch": state.get("candidate_actions_executed") == 0,
        "reference_feature_exact": result.get("reference_feature_sha256") == reference_sha == sha256(required["PREACTION_SEQUENCE.npy"]),
        "phasefree_sequence_shape": sequence.shape == (8, 71) and bool(np.isfinite(sequence).all()),
        "full_horizon_or_proven_early_terminal": settled_endpoint or proven_early_terminal,
        "label_valid_binary_consistent": outcome.get("label_valid") is True and label in (0, 1) and label == outcome.get("full_task_success_y"),
        "full_task_endpoint": settled_endpoint or proven_early_terminal,
        "source_hashes_before_exact": before.get("source_hashes") == runtime["source_hashes"],
        "source_hashes_after_exact": after.get("source_hashes") == runtime["source_hashes"],
        "branch_embedded_source_hashes_present": bool(result.get("source_hashes")),
        "actual_mass_exact": bool(np.isclose(material.get("actual_total_mass_kg"), spec["mass_kg"], rtol=0, atol=2e-8)),
        "friction_fixed": bool(material.get("object_static_dynamic_friction")) and all(
            np.allclose(pair, [0.5, 0.5], rtol=0, atol=1e-7)
            for pair in material.get("object_static_dynamic_friction", [])
        ),
        "inertia_scaled_geometry_fixed": material.get("inertia_scaled_by_mass_ratio") is True
            and inertia_matches(material) and material.get("geometry_or_appearance_modified") is False,
        "no_retry_or_rerun_artifact": not any("retry" in path.name.lower() or "rerun" in path.name.lower() for path in job.iterdir()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "valid": not failed,
        "missing": [],
        "failed_checks": failed,
        "checks": checks,
        "artifact_sha256": {name: sha256(path) for name, path in required.items()},
    }


def spec_lookup(subset: dict):
    return {(job["context_id"], float(job["force_N"])): job for job in subset["jobs"]}
