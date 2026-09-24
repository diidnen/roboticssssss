#!/usr/bin/env python3
"""Freeze the user-authorized, outcome-independent 216-branch MASS subset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TASKS = [0, 1, 5, 6]
MASSES_KG = [0.05, 0.10, 0.20]
FORCES_N = [3.0, 4.0, 5.0]
ROOTS_BY_SPLIT = {
    "TRAIN": [181000, 181001, 181002, 181003],
    "VAL": [181010],
    "HELDOUT": [181011],
}
# A fixed Latin-style ordering. Each consecutive block of four spans all tasks,
# and each consecutive block of three positions spans all mass anchors.
CONTEXT_ORDER = [
    (0, 0.05), (1, 0.10), (5, 0.20), (6, 0.05),
    (0, 0.10), (1, 0.20), (5, 0.05), (6, 0.10),
    (0, 0.20), (1, 0.05), (5, 0.10), (6, 0.20),
]
PAUSE_SNAPSHOT = {
    "captured_at_utc": "2026-09-10T06:21:20Z",
    "valid_superset_branches": 54,
    "active_claims": [
        "branches/t0_r181000_m100g/F4.5",
        "branches/t0_r181001_m100g/F4",
        "branches/t0_r181002_m100g/F4",
        "branches/t0_r181003_m100g/F4",
    ],
    "paused_launcher_pids": [86067, 86673, 87289, 87885],
}


def read(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: str) -> None:
    with path.open("x") as stream:
        stream.write(value)


def main() -> None:
    plan_path = HERE / "MASS_TRAINING_CONTEXT_PLAN.json"
    runtime_path = HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"
    protocol_path = HERE / "MASS_TRAINING_PROTOCOL.json"
    addendum_path = HERE / "MASS_MATCHED_BRANCH_ORCHESTRATION_ADDENDUM.json"
    plan = read(plan_path)
    runtime = read(runtime_path)

    if runtime["candidate_forces_N"] != [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]:
        raise RuntimeError("authoritative 648 force grid changed")
    lookup = {(row["split"], row["root"], row["task"], float(row["mass_kg"])): row for row in plan["contexts"]}
    expected_keys = {
        (split, root, task, mass)
        for split, roots in ROOTS_BY_SPLIT.items()
        for root in roots for task in TASKS for mass in MASSES_KG
    }
    if set(lookup) != expected_keys:
        raise RuntimeError("authoritative context plan is not the frozen 72-context factorial")

    jobs = []
    queue_index = 0
    for split, roots in ROOTS_BY_SPLIT.items():
        for force_cycle in range(len(FORCES_N)):
            for context_position, (task, mass) in enumerate(CONTEXT_ORDER):
                for root_position, root in enumerate(roots):
                    # Rotating force assignment prevents four-worker TRAIN waves
                    # from concentrating on one force while remaining factorial.
                    force = FORCES_N[(force_cycle + root_position) % len(FORCES_N)]
                    row = lookup[(split, root, task, mass)]
                    jobs.append({
                        "queue_index": queue_index,
                        "split": split,
                        "root": root,
                        "task": task,
                        "mass_kg": mass,
                        "force_N": force,
                        "context_id": row["id"],
                        "relative_branch_path": f"branches/{row['id']}/F{force:g}",
                        "context_position": context_position,
                        "force_cycle": force_cycle,
                    })
                    queue_index += 1

    keys = [(j["context_id"], j["force_N"]) for j in jobs]
    if len(jobs) != 216 or len(set(keys)) != 216:
        raise RuntimeError("reduced queue is not exactly 216 unique branches")
    split_counts = {split: sum(j["split"] == split for j in jobs) for split in ROOTS_BY_SPLIT}
    if split_counts != {"TRAIN": 144, "VAL": 36, "HELDOUT": 36}:
        raise RuntimeError(f"unexpected split counts: {split_counts}")

    original_hashes = {
        str(path.relative_to(HERE)): sha256(path)
        for path in (plan_path, runtime_path, protocol_path, addendum_path)
    }
    subset = {
        "schema": "TIME_BOUNDED_MASS_SUBSET_V1",
        "status": "FROZEN_BEFORE_REDUCED_COLLECTION",
        "frozen_at_utc": "2026-09-10T06:21:20Z",
        "authorization_basis": "User-fixed end-of-day compute budget; no scientific outcome criterion.",
        "original_superset_target": 648,
        "reduced_primary_target": 216,
        "tasks": TASKS,
        "masses_kg": MASSES_KG,
        "forces_N": FORCES_N,
        "roots_by_split": ROOTS_BY_SPLIT,
        "split_branch_targets": split_counts,
        "split_context_targets": {"TRAIN": 48, "VAL": 12, "HELDOUT": 12},
        "fixed_friction_mu": 0.5,
        "queue_order": {
            "split_order": list(ROOTS_BY_SPLIT),
            "context_order_task_mass": [{"task": t, "mass_kg": m} for t, m in CONTEXT_ORDER],
            "algorithm": "split, then force_cycle, then fixed context order, then root; force index=(force_cycle+root_position) mod 3",
            "outcome_dependent": False,
            "maximum_concurrent_gpu_workers": 4,
        },
        "heldout_policy": {
            "physical_collection_before_model_freeze_allowed": True,
            "labels_sealed_until_checkpoint_and_analysis_protocol_freeze": True,
            "checkpoint_selection_sources": ["TRAIN", "VAL"],
            "heldout_checkpoint_selection_forbidden": True,
        },
        "primary_denominator_rule": "Only exact 3.0, 4.0, and 5.0 N branches in jobs below.",
        "auxiliary_preservation_rule": "Valid 3.25, 3.5, 3.75, 4.25, 4.5, and 4.75 N branches remain intact and are excluded from the primary denominator.",
        "automatic_expansion_to_648": False,
        "pause_snapshot": PAUSE_SNAPSHOT,
        "authoritative_original_artifact_sha256": original_hashes,
        "jobs": jobs,
    }
    subset_path = HERE / "TIME_BOUNDED_MASS_SUBSET_V1.json"
    write_new(subset_path, json.dumps(subset, indent=2, sort_keys=True) + "\n")
    digest = sha256(subset_path)
    write_new(HERE / "TIME_BOUNDED_MASS_SUBSET_V1_SHA256.txt", f"{digest}  TIME_BOUNDED_MASS_SUBSET_V1.json\n")

    rationale = f"""# Time-bounded MASS subset rationale

## Decision

The user authorized a reduced primary feasibility dataset of 216 branches to meet a fixed end-of-day compute budget. The rule is purely factorial: retain every frozen task, every mass anchor, every frozen root, and only the predeclared Fixed-3, Fixed-4, and Fixed-5 force anchors.

The primary design is 4 tasks × 3 masses × 6 roots × 3 forces = 216 branches: 144 TRAIN, 36 VAL, and 36 HELDOUT. Its SHA-256 is `{digest}`.

## Timing disclosure

Some outcomes from the original deterministic 648-branch queue were already available before this reduction. At the recorded pause snapshot, 54 superset branches were valid and four workers were active. The reduced force set was specified by the user from compute budget alone—not from success rates, labels, force requirements, or ActiveForcing outcomes. No branch-level scientific outcome was used to choose the subset.

## Preservation and scope

The original 648 protocol, manifests, orchestration addendum, references, and branches remain unchanged and resumable. Valid 3.25–4.75 N intermediate-force branches are preserved as auxiliary evidence but excluded from the 216 primary denominator. The reduced experiment can assess coarse mass-conditioned force sufficiency at 3/4/5 N; it cannot establish 0.25-N force calibration or dense continuous force-response identification.

HELDOUT root 181011 remains unavailable to training, checkpoint selection, calibration, threshold selection, and early stopping. If its physics is collected before model freeze, its labels remain sealed until the selected checkpoint and analysis protocol are frozen.
"""
    write_new(HERE / "TIME_BOUNDED_MASS_SUBSET_RATIONALE.md", rationale)
    print(json.dumps({"subset": str(subset_path), "sha256": digest, "jobs": len(jobs), "split_counts": split_counts}, indent=2))


if __name__ == "__main__":
    main()
