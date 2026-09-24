#!/usr/bin/env python3
"""Data-quality gate for current MASS references and matched branches."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inertia_readback_matches(material):
    nominal = np.asarray(material.get("nominal_inertia", []), dtype=float)
    actual = np.asarray(material.get("actual_inertia", []), dtype=float)
    ratio = material.get("mass_ratio")
    return bool(nominal.size and nominal.shape == actual.shape and ratio is not None and
                np.isfinite(nominal).all() and np.isfinite(actual).all() and
                np.allclose(actual, nominal * float(ratio), rtol=3e-6, atol=1e-12))


def main():
    json_out = HERE / "MASS_CURRENT_TRAINING_COLLECTION_DATA_QUALITY.json"
    md_out = HERE / "MASS_CURRENT_TRAINING_COLLECTION_DATA_QUALITY.md"
    if json_out.exists() or md_out.exists(): raise FileExistsError("training data-quality decision already exists")
    plan = read(HERE / "MASS_TRAINING_CONTEXT_PLAN.json")
    runtime = read(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json")
    contexts = plan["contexts"]; forces = [float(value) for value in runtime["candidate_forces_N"]]
    if len(contexts) != 72 or len(forces) != 9: raise RuntimeError("unexpected frozen training design")
    reference_rows, branch_rows, violations = [], [], []
    reference_artifact_hashes, branch_artifact_hashes = {}, {}
    label_counts, group_forces = Counter(), defaultdict(set)
    for context in contexts:
        cid = context["id"]; reference = HERE / "references" / cid
        required = [reference / name for name in ("WORKER_COMPLETION.json", "RESULT.json", "PREACTION_SEQUENCE.npy",
                                                    "DECISION_STATE.pt", "MASS_INTERVENTION_READBACK.json",
                                                    "SOURCE_HASHES_BEFORE.json", "SOURCE_HASHES_AFTER.json")]
        if not all(path.is_file() for path in required):
            violations.append([cid, "REFERENCE_INCOMPLETE"]); continue
        completion, result = read(required[0]), read(required[1]); material = read(required[4])
        reference_before, reference_after = read(required[5]), read(required[6])
        sequence = np.load(required[2], allow_pickle=False)
        reference_checks = {
            "logical_success": completion.get("logical_success") is True,
            "probe_qualified": result.get("probe_qualified") is True,
            "context_plan_exact": result.get("plan") == context,
            "candidate_actions_zero": result.get("candidate_actions_executed") == 0,
            "feature_shape_8x71": sequence.shape == (8, 71) and bool(np.isfinite(sequence).all()),
            "requested_mass_exact": material.get("requested_total_mass_kg") == context["mass_kg"],
            "actual_mass_tolerance": bool(np.isclose(material.get("actual_total_mass_kg"), context["mass_kg"], rtol=0, atol=2e-8)),
            "inertia_scaled": material.get("inertia_scaled_by_mass_ratio") is True and inertia_readback_matches(material),
            "friction_fixed": bool(material.get("object_static_dynamic_friction")) and
                all(np.allclose(pair, [.5, .5], rtol=0, atol=1e-7) for pair in material.get("object_static_dynamic_friction", [])),
            "geometry_appearance_unchanged": material.get("geometry_or_appearance_modified") is False,
            "source_hashes_before_exact": reference_before.get("source_hashes") == runtime["source_hashes"],
            "source_hashes_after_exact": reference_after.get("source_hashes") == runtime["source_hashes"]}
        if not all(reference_checks.values()): violations.append([cid, "REFERENCE_CHECK", *[k for k, v in reference_checks.items() if not v]])
        reference_rows.append({"context_id": cid, "split": context["split"], "task": context["task"],
                               "root": context["root"], "mass_kg": context["mass_kg"], "checks": reference_checks,
                               "sequence_sha256": sha(required[2]), "snapshot_sha256": sha(required[3])})
        reference_artifact_hashes[cid] = {path.name: sha(path) for path in required}
        for force in forces:
            job = HERE / "branches" / cid / f"F{force:g}"
            files = {name: job / name for name in ("WORKER_COMPLETION.json", "BRANCH_RESULT.json",
                "PREACTION_STATE_COMPARISON.json", "PREACTION_SEQUENCE.npy", "SOURCE_HASHES_BEFORE.json",
                "SOURCE_HASHES_AFTER.json", "MASS_INTERVENTION_READBACK.json")}
            if not all(path.is_file() for path in files.values()):
                violations.append([cid, force, "BRANCH_INCOMPLETE"]); continue
            done, branch, state = read(files["WORKER_COMPLETION.json"]), read(files["BRANCH_RESULT.json"]), read(files["PREACTION_STATE_COMPARISON.json"])
            before, after, branch_material = read(files["SOURCE_HASHES_BEFORE.json"]), read(files["SOURCE_HASHES_AFTER.json"]), read(files["MASS_INTERVENTION_READBACK.json"])
            candidate_sequence = np.load(files["PREACTION_SEQUENCE.npy"], allow_pickle=False)
            outcome = branch.get("outcome", {}); label = branch.get("full_task_success_y")
            label_value = int(label) if label in (0, 1) else -1
            branch_checks = {
                "logical_success": done.get("logical_success") is True,
                "branch_passed": branch.get("passed") is True,
                "context_plan_exact": branch.get("plan") == context,
                "candidate_force_exact": branch.get("candidate_F") == force,
                "candidate_state_exact": state.get("passed") is True and branch.get("candidate_preact_state_equality") is True,
                "all_state_groups_exact": all(state.get("exposed_state_equality", {}).values()),
                "candidate_actions_zero_before_branch": state.get("candidate_actions_executed") == 0,
                "reference_feature_exact": branch.get("reference_feature_sha256") == reference_rows[-1]["sequence_sha256"] == sha(files["PREACTION_SEQUENCE.npy"]),
                "sequence_shape_8x71": candidate_sequence.shape == (8, 71) and bool(np.isfinite(candidate_sequence).all()),
                "full_horizon": branch.get("steps") == 350,
                "label_valid_binary_consistent": outcome.get("label_valid") is True and label in (0, 1) and label == outcome.get("full_task_success_y"),
                "full_task_endpoint": outcome.get("unchanged_success_predicate_version") == "PROSPECTIVE_WHOLE_MESH_CONTAINMENT_RELEASE_FULLTASK_V1",
                "source_hashes_before_exact": before.get("source_hashes") == runtime["source_hashes"],
                "source_hashes_after_exact": after.get("source_hashes") == runtime["source_hashes"],
                "mass_friction_exact": bool(np.isclose(branch_material.get("actual_total_mass_kg"), context["mass_kg"], rtol=0, atol=2e-8)) and
                    bool(branch_material.get("object_static_dynamic_friction")) and
                    all(np.allclose(pair, [.5, .5], rtol=0, atol=1e-7) for pair in branch_material.get("object_static_dynamic_friction", [])),
                "inertia_geometry_contract": branch_material.get("inertia_scaled_by_mass_ratio") is True and
                    inertia_readback_matches(branch_material) and branch_material.get("geometry_or_appearance_modified") is False}
            if not all(branch_checks.values()): violations.append([cid, force, "BRANCH_CHECK", *[k for k, v in branch_checks.items() if not v]])
            label_counts[(context["split"], context["task"], context["mass_kg"], label_value)] += 1
            group_forces[cid].add(force)
            branch_rows.append({"context_id": cid, "split": context["split"], "task": context["task"],
                "root": context["root"], "mass_kg": context["mass_kg"], "force_N": force,
                "full_task_success_y": label_value, "checks": branch_checks, "path": str(job)})
            branch_artifact_hashes[f"{cid}|F{force:g}"] = {name: sha(path) for name, path in files.items()}
    split_roots = defaultdict(set)
    for row in reference_rows: split_roots[row["split"]].add(row["root"])
    checks = {
        "72_complete_unique_references": len(reference_rows) == 72 and len({row["context_id"] for row in reference_rows}) == 72,
        "648_complete_unique_branches": len(branch_rows) == 648 and len({(row["context_id"], row["force_N"]) for row in branch_rows}) == 648,
        "all_sibling_force_sets_exact": len(group_forces) == 72 and all(values == set(forces) for values in group_forces.values()),
        "root_splits_disjoint": not (split_roots["TRAIN"] & split_roots["VAL"] or split_roots["TRAIN"] & split_roots["HELDOUT"] or split_roots["VAL"] & split_roots["HELDOUT"]),
        "no_reserved_dev_final_roots": not ({plan["development_online_root_reserved"], *plan["final_roots_reserved_not_training"]} & set().union(*split_roots.values())),
        "all_record_checks_pass": not violations,
        "label_valid_fraction_one": len(branch_rows) == 648 and all(row["full_task_success_y"] in (0, 1) for row in branch_rows)}
    passed = all(checks.values())
    result = {"as_of_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if passed else "FAIL",
        "intended_grain": "one full-task label per frozen (physical context, candidate force) branch",
        "reference_count": len(reference_rows), "branch_count": len(branch_rows), "checks": checks,
        "violations": violations, "roots_by_split": {key: sorted(value) for key, value in split_roots.items()},
        "reference_artifact_hashes": reference_artifact_hashes, "branch_artifact_hashes": branch_artifact_hashes,
        "label_profile": [{"split": key[0], "task": key[1], "mass_kg": key[2], "label": key[3], "count": value}
                          for key, value in sorted(label_counts.items())],
        "source_hashes": {str(path): sha(path) for path in (Path(__file__), HERE / "MASS_TRAINING_CONTEXT_PLAN.json",
                                                            HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json")},
        "severity_findings": [] if passed else [{"severity": "CRITICAL", "finding": "Frozen collection is incomplete or violates its context/branch contract."}],
        "automated_tests": ["unique (context_id, force_N)", "nine-force sibling completeness", "root split disjointness",
                            "exact pre-action state/hash parity", "fixed mass/friction/inertia readback", "binary valid full-task label", "frozen source hashes"]}
    with json_out.open("x") as stream: json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
    md_out.write_text(f"""# Current MASS training-collection data quality

## Dataset and grain

Assessment: **{result['status']}**. The intended grain is one complete full-task label per frozen physical context and candidate-force sibling. The audit found {len(reference_rows)}/72 query references and {len(branch_rows)}/648 unique matched branches.

## Checks performed

The audit verifies completeness, composite-key uniqueness, all nine force siblings, root-disjoint splits, exclusion of development/final roots, exact reference/candidate post-query states, current full-task label validity, mass/friction/inertia readback, finite 8×71 auxiliary sequences, and source hashes before and after every branch.

## Findings

{'No data-quality violations were found.' if passed else f"{len(violations)} violations were found; see the JSON audit before training."}

## Impacted use

{'The collection is admissible for frozen phase-free feasibility training and independent offline qualification.' if passed else 'The collection is not admissible for model training or citation.'}

## Automated gates

The stable checks are encoded in `audit_mass_training_collection.py`; the complete label profile by split, task, mass, and outcome is preserved in the JSON audit. Physical outcome balance is descriptive and is not a reason to alter or recollect valid branches.

## Assumptions and open questions

The auxiliary sequences come from controlled downstream motion by design. Transfer to the current phase-free online-VLA motion context remains a separate development-only gate.
""")
    print(json.dumps({"status": result["status"], "references": len(reference_rows), "branches": len(branch_rows),
                      "violations": len(violations)}, indent=2))
    if not passed: raise SystemExit(2)


if __name__ == "__main__": main()
