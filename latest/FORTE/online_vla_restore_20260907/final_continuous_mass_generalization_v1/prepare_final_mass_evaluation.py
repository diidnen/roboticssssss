#!/usr/bin/env python3
"""Outcome-blind freeze of the 48-context/144-branch MASS evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/exouser/FORTE")
BASE = ROOT / "analysis/results"
FRICTION_FINAL = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1")
FRICTION_SOURCE = FRICTION_FINAL / "SOURCE_SNAPSHOT"
TASKS = {0: ("alphabet_soup_1", "basket_1"), 1: ("cream_cheese_1", "basket_1"),
         5: ("tomato_sauce_1", "basket_1"), 6: ("butter_1", "basket_1")}
ROOTS = (181100, 181101)
ANCHORS = (0.05, 0.10, 0.20)
MASS_VALUES = (0.0625, 0.075, 0.0875, 0.125, 0.15, 0.175)
POSITIONS = tuple((value - ANCHORS[0]) / (ANCHORS[-1] - ANCHORS[0]) for value in MASS_VALUES)
SEGMENTS = ("LOW_MID", "LOW_MID", "LOW_MID", "MID_HIGH", "MID_HIGH", "MID_HIGH")
FRACTIONS = (0.25, 0.50, 0.75, 0.25, 0.50, 0.75)
METHODS = ("ACTIVEFORCING_MASS", "GT_MASS", "FIXED_4")
FRICTION_MANIFEST_SHA256 = "b5f38c56037ef51ab7843c99b8f05755ce8d28c845e4120961139028aa550ab0"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def root_audit():
    scopes = [Path("/home/exouser/FORTE"), Path("/home/exouser/Tabero"),
              Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907"), Path("/media/volume/data/exouser")]
    pattern = r'"(?:root|root_id|root_seed|seed_idx|seed)"\s*:\s*(?:' + "|".join(map(str, ROOTS)) + r')(?:\s*[,}])'
    proc = subprocess.run(["rg", "-l", "--glob", "*.json", pattern, *map(str, scopes)],
                          text=True, capture_output=True, check=False)
    if proc.returncode not in (0, 1): raise RuntimeError(proc.stderr)
    # Search the experiment namespace too.  The reservation field in the
    # training plan deliberately does not match the root-seed field pattern;
    # any actual context/result record carrying a final root must fail freeze.
    matches = sorted({str(Path(raw).resolve()) for raw in proc.stdout.splitlines()})
    if matches: raise RuntimeError("final root collision: " + json.dumps(matches, indent=2))
    return {"roots": list(ROOTS), "exact_root_seed_field_matches": [], "fresh_nonexposure_verified": True,
            "search_scopes": [str(path) for path in scopes], "outcome_used": False}


def friction_integrity_audit():
    manifest_path = FRICTION_FINAL / "CONTINUOUS_FRICTION_RUNTIME_MANIFEST.json"
    if sha(manifest_path) != FRICTION_MANIFEST_SHA256:
        raise RuntimeError("closed friction runtime manifest changed")
    manifest = read(manifest_path); checks = []
    for group in ("source_hashes", "frozen_artifact_hashes"):
        for path, expected in sorted(manifest[group].items()):
            actual = sha(path) if Path(path).is_file() else None
            checks.append({"group": group, "path": path, "expected_sha256": expected,
                           "actual_sha256": actual, "exact": actual == expected})
    if not all(row["exact"] for row in checks): raise RuntimeError("closed friction source/artifact changed")
    return {"status": "PASS", "friction_runtime_manifest": str(manifest_path),
            "friction_runtime_manifest_sha256": FRICTION_MANIFEST_SHA256,
            "verified_entries": len(checks), "checks": checks,
            "mass_experiment_write_namespace": str(HERE), "friction_experiment_modified": False}


def main():
    eval_path = HERE / "FINAL_MASS_EVALUATION_MANIFEST.json"
    runtime_path = HERE / "FINAL_MASS_RUNTIME_MANIFEST.json"
    hashes_path = HERE / "FINAL_MASS_EVALUATION_HASHES.txt"
    if any(path.exists() for path in (eval_path, runtime_path, hashes_path)): raise FileExistsError("final MASS freeze already exists")
    development = HERE / "MASS_ONLINE_DEVELOPMENT_QUALIFICATION.json"
    if not development.is_file() or not read(development).get("qualified"): raise RuntimeError("development online qualification did not authorize final freeze")
    training = read(HERE / "MASS_TRAINING_CONTEXT_PLAN.json")
    belief_support = sorted({c["mass_kg"] for c in training["contexts"] if c["split"] == "TRAIN"})
    feasibility_support = belief_support.copy()
    old_eval = [0.05, 0.10, 0.20]
    collisions = {"belief_training": sorted(set(MASS_VALUES) & set(belief_support)),
                  "feasibility_training": sorted(set(MASS_VALUES) & set(feasibility_support)),
                  "canonical_anchors": sorted(set(MASS_VALUES) & set(ANCHORS)),
                  "historical_old_evaluation": sorted(set(MASS_VALUES) & set(old_eval))}
    if any(collisions.values()): raise RuntimeError("unseen MASS collision: " + json.dumps(collisions))
    support = {"training_support_type": "DISCRETE_ANCHORS", "mass_unit": "kg",
        "belief_training_masses_per_task_kg": {str(task): belief_support for task in TASKS},
        "feasibility_training_masses_per_task_kg": {str(task): feasibility_support for task in TASKS},
        "canonical_anchors_kg": list(ANCHORS), "historical_old_evaluation_masses_kg": old_eval,
        "final_unseen_masses_per_task_kg": {str(task): list(MASS_VALUES) for task in TASKS},
        "exact_set_collisions": collisions, "claim_wording": "interpolation to unseen masses between training anchors"}
    write(HERE / "MASS_TRAINING_SUPPORT_AUDIT.json", support)
    friction_audit = friction_integrity_audit(); write(HERE / "FRICTION_UNMODIFIED_AUDIT.json", friction_audit)
    roots = root_audit(); write(HERE / "FINAL_MASS_ROOT_NONEXPOSURE_AUDIT.json", roots)
    contexts = []
    for root in ROOTS:
        for task, (obj, target) in TASKS.items():
            for index, (mass, position, segment, fraction) in enumerate(zip(MASS_VALUES, POSITIONS, SEGMENTS, FRACTIONS), 1):
                contexts.append({"id": f"t{task}_r{root}_m{index}", "task": task, "root": root,
                    "mass_kg": mass, "mass_index": index, "normalized_mass_position": position,
                    "interpolation_segment": segment, "interpolation_fraction": fraction,
                    "mu": 0.5, "object": obj, "target": target,
                    "split": "FINAL_EXACT_HELDOUT_IN_SUPPORT_MASS_INTERPOLATION"})
    branch_queue = [{"context_index": index, "context_id": context["id"], "method": method}
                    for index, context in enumerate(contexts) for method in METHODS]
    random.Random(2026091002).shuffle(branch_queue)
    execution = ([{"context_index": index, "context_id": context["id"], "method": "REFERENCE"}
                  for index, context in enumerate(contexts)] + branch_queue)
    evaluation = {"version": "FINAL_CONTINUOUS_MASS_GENERALIZATION_V1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_final_physics": True, "training_support_type": "DISCRETE_ANCHORS",
        "claim_wording": support["claim_wording"], "roots": list(ROOTS), "tasks": list(TASKS),
        "masses_per_task_kg": list(MASS_VALUES), "normalized_mass_positions": list(POSITIONS),
        "contexts": contexts, "methods": [{"name": "ACTIVEFORCING_MASS", "true_mass_available_to_planner": False},
            {"name": "GT_MASS", "true_mass_available_to_planner": True, "role": "decision oracle/reference"},
            {"name": "FIXED_4", "force_N": 4.0}],
        "num_contexts": 48, "num_method_branches": 144, "num_query_references": 48,
        "execution_queue": execution, "method_order_seed": 2026091002,
        "seeds": {"simulator_root_groups": list(ROOTS), "method_order_seed": 2026091002,
                  "policy_noise_seed_rule": "int(sha256('v1:{context_id}:{branch_step}')[:8], 16)"},
        "fixed_friction": 0.5, "geometry_appearance_modified": False,
        "mass_intervention": "total object mass with inertia tensor scaled by identical mass ratio",
        "query": "identical restored P4-B query before all three branches",
        "force_support_N": [3.0, 5.0], "force_grid_step_N": 0.05,
        "utility": "p*(5-F)/5 + (1-p)*(-1); lower-force tie break", "one_force_setpoint_per_rollout": True,
        "online_vla": {"checkpoint_sha256": "0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17",
            "predicted_chunk_steps": 50, "executed_steps_per_query": 10, "downstream_horizon": 350},
        "terminal_evaluator": "ONLINE_VLA_350STEP_WHOLE_MESH_RELEASE_SUPPORT_V1",
        "measured_squeeze_metric": "branch mean of 2*min(|N_L|,|N_R|) over all non-release frames, retaining zero-contact frames",
        "lower_upper_split": {"LOWER": [1, 2, 3], "UPPER": [4, 5, 6]},
        "failure_taxonomy": {
            "precedence": ["UNDER_FORCE", "VLA_EXECUTION_VARIANCE", "POST_LIFT_GEOMETRIC", "OTHER"],
            "rules": {
                "UNDER_FORCE": "AF has NO_LIFT or DROP, AF force is at least 0.05 N below a successful paired method, and that paired method used higher force.",
                "VLA_EXECUTION_VARIANCE": "After excluding UNDER_FORCE, AF failed a placement/release/support geometric terminal check and a paired online-VLA method within 0.05 N succeeded.",
                "POST_LIFT_GEOMETRIC": "After excluding UNDER_FORCE and VLA_EXECUTION_VARIANCE, AF lifted, did not drop, and failed one or more placement/release/support geometric terminal checks.",
                "OTHER": "All remaining valid AF failures. Categories are diagnostics, not causal proof."
            }
        },
        "claim_decision_rules": {
            "belief_informative_supported": "final Spearman >=0.75 and MAE below frozen TRAIN-mean prior MAE",
            "belief_continuous_supported": "every task ranking accuracy >=0.75 and every task has >=4 distinct aggregate posterior means",
            "force_adaptation_supported": "all four task-wise mass-force Spearman coefficients are positive",
            "force_adaptation_mixed": "overall coefficient is positive but the supported rule fails",
            "close_to_GT_supported": "AF-GT force MAE <=0.10 N and >=75% pairs are within 0.10 N",
            "close_to_GT_mixed": "AF-GT force MAE <=0.25 N or >=50% pairs are within 0.25 N",
            "fixed4_dominance_supported": "AF success is no lower, mean squeeze is lower, and there are zero Fixed4-only successes",
            "failures_primarily_under_force_supported": "strictly more than half of AF failures receive the frozen UNDER_FORCE diagnostic",
            "shared_interface_supported": "belief informative and continuous, all four task-wise mass-force Spearman coefficients positive, and AF close-to-GT supported",
            "shared_interface_mixed": "belief informative and either the overall mass-force coefficient is positive or AF close-to-GT is mixed"
        },
        "rerun_policy": "zero result-driven reruns; only independently verified infrastructure corruption",
        "automatic_expansion": False, "stop_after_valid_method_branches": 144,
        "runtime_manifest_path": str(runtime_path), "analysis_script": str(HERE / "analyze_continuous_mass.py")}
    write(eval_path, evaluation)
    sources = [HERE / name for name in ("mass_online_worker.py", "mass_online_variants.py", "mass_online_launch.py",
        "run_mass_online_queue.py", "start_mass_policy_server.py", "mass_runtime_core.py", "mass_belief.py", "mass_feasibility.py",
        "qualify_mass_online.py", "prepare_final_mass_evaluation.py", "analyze_continuous_mass.py",
        "analyze_mass_offline_qualification.py", "audit_mass_training_collection.py",
        "validate_final_mass_analysis.py")]
    sources += [FRICTION_SOURCE / name for name in ("runtime.py", "arbitration.py", "common.py")]
    sources += [Path("/home/exouser/FORTE/online_vla_restore_20260907/server_v3_frozen/SOURCE_SNAPSHOT/server.py"),
                Path("/home/exouser/FORTE/online_vla_restore_20260907/server_v3_frozen/SOURCE_SNAPSHOT/common.py")]
    sources += [BASE / "current_runtime_sensor_repair_v3_candidate_20260905/measurement_hooks.py",
                BASE / "current_runtime_branch_execution_v6_20260905/branch_execution.py",
                ROOT / "activeforcing_current_probe.py", ROOT / "current_contract_belief_features.py"]
    artifacts = [eval_path, HERE / "MASS_TRAINING_SUPPORT_AUDIT.json", HERE / "FINAL_MASS_ROOT_NONEXPOSURE_AUDIT.json",
        HERE / "FRICTION_UNMODIFIED_AUDIT.json",
        development, HERE / "MASS_TRAINING_PROTOCOL.json", HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json",
        HERE / "MASS_BELIEF_MANIFEST.json", HERE / "MASS_BELIEF_QUALIFICATION.json",
        HERE / "MASS_FEASIBILITY_MANIFEST.json", HERE / "MASS_FEASIBILITY_QUALIFICATION.json",
        HERE / "MASS_BELIEF_RUNTIME_PARITY_REPAIR.json", HERE / "MASS_QUERY_PROTOCOL.md",
        HERE / "MASS_OFFLINE_QUALIFICATION_AUDIT.json", HERE / "MASS_FEASIBILITY_MASS_FORCE_INSPECTION.csv",
        HERE / "MASS_OFFLINE_QUALIFICATION_REPORT.md", HERE / "MASS_BELIEF_IDENTITY_CONTROL.json",
        HERE / "MASS_CURRENT_TRAINING_COLLECTION_DATA_QUALITY.json",
        HERE / "MASS_CURRENT_TRAINING_COLLECTION_DATA_QUALITY.md", HERE / "MASS_ANALYSIS_PROTOCOL.md"]
    for model_manifest in ("MASS_BELIEF_MANIFEST.json", "MASS_FEASIBILITY_MANIFEST.json"):
        artifacts += [Path(row["path"]) for row in read(HERE / model_manifest)["checkpoints"]]
    runtime = {"version": "FINAL_CONTINUOUS_MASS_CURRENT_ONLINE_VLA_RUNTIME_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_hashes": {str(path): sha(path) for path in sources},
        "artifact_hashes": {str(path): sha(path) for path in artifacts}, "evaluation_manifest_sha256": sha(eval_path),
        "phase_free": True, "online_vla": True, "candidate_independent_state": True,
        "full_task_label": True, "true_mass_AF_deployment_leakage": False,
        "friction_experiment_modified": False}
    write(runtime_path, runtime)
    with hashes_path.open("x") as stream:
        for path in (eval_path, runtime_path, HERE / "MASS_TRAINING_SUPPORT_AUDIT.json", HERE / "FINAL_MASS_ROOT_NONEXPOSURE_AUDIT.json",
                     HERE / "FRICTION_UNMODIFIED_AUDIT.json"):
            stream.write(f"{sha(path)}  {path.name}\n")
        for path, digest in sorted(runtime["source_hashes"].items()):
            stream.write(f"{digest}  SOURCE  {path}\n")
        for path, digest in sorted(runtime["artifact_hashes"].items()):
            stream.write(f"{digest}  ARTIFACT  {path}\n")
    print(json.dumps({"frozen": True, "contexts": len(contexts), "method_branches": len(branch_queue),
        "masses": list(MASS_VALUES), "collisions": collisions, "evaluation_sha256": sha(eval_path),
        "runtime_sha256": sha(runtime_path)}, indent=2))


if __name__ == "__main__": main()
