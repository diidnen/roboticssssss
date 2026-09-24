"""Zero-physics freeze for unseen continuous-friction generalization.

This program audits the actual frozen model training rows, verifies that the
predeclared interpolation values are absent, collision-checks exactly two
prospective simulator roots, snapshots the already-qualified online runtime,
and freezes the 144-branch execution plan.  It never imports Isaac Lab.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
HERE = ROOT / "online_vla_restore_20260907/continuous_friction_generalization"
BASE = ROOT / "analysis/results"
BELIEF_TRAIN = BASE / "current_multitask58_trained_v2_20260905"
BELIEF_PLAN = BELIEF_TRAIN / "FROZEN_INPUT_PLAN.json"
BELIEF_PROVENANCE = BELIEF_TRAIN / "PROVENANCE.json"
FEAS = BASE / "current_fulltask_feasibility_baseline_v1_20260906"
FEAS_PROTOCOL = FEAS / "TRAINING_PROTOCOL.json"
FEAS_STAGE = BASE / "current_matched_stage1_648_v1_20260906/STAGE_MANIFEST.json"
V5 = BASE / "current_matched_runtime_collection_v5_20260906"
SECONDARY = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_secondary_baselines_v1")
PARENT_SNAPSHOT = SECONDARY / "SOURCE_SNAPSHOT"
TASK_META = {
    0: ("alphabet_soup_1", "basket_1"),
    1: ("cream_cheese_1", "basket_1"),
    5: ("tomato_sauce_1", "basket_1"),
    6: ("butter_1", "basket_1"),
}
CANONICAL = {
    0: (0.2937102019159983, 0.45058012388233365, 0.9401893373882813),
    1: (0.24001855706823938, 0.4670478243968788, 0.9215002256494116),
    5: (0.2708052527683458, 0.538330484944251, 0.9450036831015114),
    6: (0.2575768733744629, 0.5585950380614311, 0.9893482298937868),
}
FRACTIONS = (("LOW_MID", 0.25), ("LOW_MID", 0.50), ("LOW_MID", 0.75),
             ("MID_HIGH", 0.25), ("MID_HIGH", 0.50), ("MID_HIGH", 0.75))
METHODS = ("ACTIVEFORCING", "GT_PHYSICS", "FIXED_4")
METHOD_ORDER_SEED = 2026090901


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, value) -> None:
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def hash_receipt(path: Path, receipt: Path) -> None:
    receipt.write_text(f"{sha(path)}  {path.name}\n")


def support_audit(out: Path):
    belief_plan = read(BELIEF_PLAN)
    belief_values = {task: set() for task in TASK_META}
    belief_examples = {}
    for item in belief_plan["examples"]:
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise RuntimeError(f"Belief training example changed: {path}")
        result = read(path)
        plan = result["plan"]
        if plan["split"] == "TRAIN":
            belief_values[int(plan["task"])].add(float(plan["mu"]))
            belief_examples[str(path)] = item["sha256"]

    protocol = read(FEAS_PROTOCOL)
    if sha(FEAS_STAGE) != protocol["stage_manifest_sha256"]:
        raise RuntimeError("Feasibility stage manifest changed")
    stage = read(FEAS_STAGE)
    contexts = {x["id"]: x for x in stage["contexts"]}
    feasibility_values = {task: set() for task in TASK_META}
    training_row_hashes = {}
    training_rows = 0
    for raw_path, digest in protocol["train_val_row_hashes"].items():
        path = Path(raw_path)
        if sha(path) != digest:
            raise RuntimeError(f"Feasibility row changed: {path}")
        row = read(path)
        if row["split"] != "TRAIN":
            continue
        ctx = contexts[row["context_id"]]
        if stage["split_by_context"][row["context_id"]] != "TRAIN":
            raise RuntimeError("Feasibility row split mismatch")
        feasibility_values[int(ctx["task"])].add(float(ctx["mu"]))
        training_row_hashes[str(path)] = digest
        training_rows += 1

    belief_values = {str(k): sorted(v) for k, v in belief_values.items()}
    feasibility_values = {str(k): sorted(v) for k, v in feasibility_values.items()}
    if any(len(v) != 9 for v in belief_values.values()):
        raise RuntimeError(f"Unexpected belief friction support: {belief_values}")
    if any(len(v) != 12 for v in feasibility_values.values()):
        raise RuntimeError(f"Unexpected feasibility friction support: {feasibility_values}")
    sampler = Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py")
    sampler_text = sampler.read_text()
    continuously_sampled = "rng.uniform(lo, hi)" in sampler_text
    audit = {
        "role": "FINAL_FROZEN_MODEL_TRAINING_FRICTION_SUPPORT_AUDIT",
        "created_utc": now(),
        "belief_model": str(V5 / "BELIEF_MANIFEST.json"),
        "feasibility_model": str(BASE / "current_fulltask_feasibility_runtime_freeze_v2_20260906/FINAL_FEASIBILITY_RUNTIME_MANIFEST.json"),
        "BELIEF_TRAIN_FRICTION_VALUES_PER_TASK": belief_values,
        "FEASIBILITY_TRAIN_FRICTION_VALUES_PER_TASK": feasibility_values,
        "NUM_BELIEF_TRAIN_FRICTION_VALUES_PER_TASK": {k: len(v) for k, v in belief_values.items()},
        "NUM_FEASIBILITY_TRAIN_FRICTION_VALUES_PER_TASK": {k: len(v) for k, v in feasibility_values.items()},
        "BELIEF_CONTINUOUSLY_SAMPLED_DURING_TRAINING": bool(continuously_sampled),
        "FEASIBILITY_CONTINUOUSLY_SAMPLED_DURING_TRAINING": bool(continuously_sampled),
        "interpretation": "Finite root-specific draws from continuous LOW/MID/HIGH intervals; not three fixed anchors and not dense coverage.",
        "belief_training_roots": [5101, 5104, 5109],
        "feasibility_training_roots": [5100, 5101, 5102, 5106],
        "belief_training_examples": 36,
        "feasibility_training_rows": training_rows,
        "source_hashes": {
            str(BELIEF_PLAN): sha(BELIEF_PLAN),
            str(BELIEF_PROVENANCE): sha(BELIEF_PROVENANCE),
            str(FEAS_PROTOCOL): sha(FEAS_PROTOCOL),
            str(FEAS_STAGE): sha(FEAS_STAGE),
            str(sampler): sha(sampler),
        },
        "belief_training_example_hashes": belief_examples,
        "feasibility_training_row_hashes": training_row_hashes,
    }
    write_json(out / "TRAINING_FRICTION_SUPPORT_AUDIT.json", audit)
    return audit


def unseen_plan(out: Path, audit):
    all_canonical = {x for triple in CANONICAL.values() for x in triple}
    tasks = []
    for task, (low, mid, high) in CANONICAL.items():
        seen_belief = set(audit["BELIEF_TRAIN_FRICTION_VALUES_PER_TASK"][str(task)])
        seen_feas = set(audit["FEASIBILITY_TRAIN_FRICTION_VALUES_PER_TASK"][str(task)])
        values = []
        for index, (segment, fraction) in enumerate(FRACTIONS, 1):
            a, b = (low, mid) if segment == "LOW_MID" else (mid, high)
            value = a + fraction * (b - a)
            absent_belief = value not in seen_belief
            absent_feas = value not in seen_feas
            absent_canonical = value not in all_canonical
            if not (a < value < b and absent_belief and absent_feas and absent_canonical):
                raise RuntimeError(f"Predeclared interpolation collision task={task} mu={value}")
            values.append({
                "index": index,
                "mu": value,
                "interpolation_segment": segment,
                "interpolation_fraction": fraction,
                "proof_of_absence_from_belief_training": absent_belief,
                "proof_of_absence_from_feasibility_training": absent_feas,
                "proof_of_absence_from_existing_final_canonical_evaluation": absent_canonical,
            })
        tasks.append({"task": task, "low_anchor": low, "mid_anchor": mid, "high_anchor": high,
                      "anchor_role": "predeclared final canonical root-5100 physical values", "unseen_values": values})
    plan = {
        "version": "FINAL_UNSEEN_CONTINUOUS_FRICTION_PLAN_V1",
        "created_utc": now(),
        "frozen_before_new_physics": True,
        "selection_rule": "Exactly 0.25/0.50/0.75 interpolation within canonical L-M and M-H; no replacement fraction was needed.",
        "training_support_audit_sha256": sha(out / "TRAINING_FRICTION_SUPPORT_AUDIT.json"),
        "tasks": tasks,
        "UNSEEN_FRICTION_PLAN_FROZEN": True,
    }
    path = out / "FINAL_UNSEEN_FRICTION_PLAN.json"
    write_json(path, plan)
    hash_receipt(path, out / "FINAL_UNSEEN_FRICTION_PLAN_SHA256.txt")
    return plan


def root_audit(roots: list[int], out: Path):
    if len(roots) != 2 or len(set(roots)) != 2:
        raise ValueError("Exactly two distinct roots required")
    locations = [Path("/home/exouser/FORTE"), Path("/home/exouser/Tabero"),
                 Path("/home/exouser/E3_E6_E7_LANES"), Path("/media/volume/newdata/exouser")]
    locations = [p for p in locations if p.exists()]
    number = "(?:" + "|".join(map(str, roots)) + ")"
    root_field = rf'"(?:root|root_id|root_seed|seed_idx|seed)"\s*:\s*"?{number}(?:"|\s*[,}}])'
    common = ["--hidden", "--no-ignore", "-g", "!**/.git/**", "-g", "!**/.venv/**",
              "-g", "!**/env_isaaclab51/**", "-g", "!**/site-packages/**", "-g", "!**/node_modules/**"]
    content_globs = sum((["-g", "*." + ext] for ext in
                         ("json", "jsonl", "csv", "tsv", "log", "txt", "md", "yaml", "yml", "toml", "py", "sh",
                          "json.gz", "jsonl.gz", "csv.gz")), [])
    content = subprocess.run(["rg", *common, *content_globs, "--search-zip", "--files-with-matches",
                              root_field, *map(str, locations)],
                             text=True, capture_output=True)
    if content.returncode not in (0, 1):
        raise RuntimeError(content.stderr)
    files = subprocess.run(["rg", "--files", *common, *map(str, locations)], text=True, capture_output=True)
    if files.returncode not in (0, 1):
        raise RuntimeError(files.stderr)
    path_re = re.compile(rf"(?:_r|root|_s){number}(?=[_/\.\-]|$)")
    path_matches = [line for line in files.stdout.splitlines() if path_re.search(line)]
    broad = subprocess.run(["rg", *common, "--search-zip", "--files-with-matches",
                            "-g", "*.json", "-g", "*.jsonl", "-g", "*.csv", "-g", "*.py", "-g", "*.md",
                            rf"(^|[^0-9.]){number}([^0-9.]|$)", *map(str, locations)],
                           text=True, capture_output=True)
    if broad.returncode not in (0, 1):
        raise RuntimeError(broad.stderr)
    broad_matches = sorted(set(broad.stdout.splitlines()))
    known_irrelevant = {
        "/media/volume/newdata/exouser/stepwise_hf_cache_20260826/hub/models--google--paligemma-3b-pt-224/snapshots/0000000000000000000000000000000000000000/tokenizer.json": "Tokenizer vocabulary integer ids, not simulator roots or seeds.",
        "/media/volume/newdata/exouser/budgeted_successor_runtime/project/analysis/results/open_budgeted_successor_probing_20260824_050543/BUDGETED_SELECTION_TRACES.csv": "Digits occur inside a SHA256 field in an unrelated experiment, not a simulator root or seed.",
        str(HERE / "analyze_continuous_friction.py"): "Prospective report template names the predeclared candidate roots; it contains no physics, prediction, or outcome evidence.",
    }
    unresolved_broad = [p for p in broad_matches if p not in known_irrelevant]
    if content.stdout.splitlines() or path_matches or unresolved_broad:
        raise RuntimeError(f"Candidate-root collision: fields={content.stdout.splitlines()} paths={path_matches} broad={unresolved_broad}")
    result = {
        "version": "CONTINUOUS_FRICTION_GENERALIZATION_ROOT_AUDIT_V1",
        "created_utc": now(),
        "roots": roots,
        "selection_rule": "First contiguous two-root block immediately after frozen secondary roots 170050/170051; chosen before any candidate physics and rejected only on identifier collision.",
        "root_selection_used_outcomes": False,
        "physics_branches_started_before_freeze": 0,
        "root_or_seed_field_matches": [],
        "path_identifier_matches": [],
        "broad_numeric_matches_reviewed": {p: known_irrelevant[p] for p in broad_matches},
        "FRESH_ROOT_NONEXPOSURE_VERIFIED": True,
        "scope": "Known belief/feasibility splits, main/ablation/secondary/development artifacts, source literals, JSON root/seed fields, and artifact path identifiers under the local experiment stores.",
    }
    write_json(out / "ROOT_NONEXPOSURE_AUDIT.json", result)
    root_plan = {
        "version": "CONTINUOUS_FRICTION_ROOTS_V1",
        "created_utc": now(),
        "roots": roots,
        "count": 2,
        "outcome_blind_generation": True,
        "do_not_replace": True,
        "audit_sha256": sha(out / "ROOT_NONEXPOSURE_AUDIT.json"),
    }
    path = out / "CONTINUOUS_FRICTION_ROOTS.json"
    write_json(path, root_plan)
    hash_receipt(path, out / "CONTINUOUS_FRICTION_ROOTS_SHA256.txt")
    return root_plan


def contexts_and_execution(out: Path, unseen, root_plan):
    contexts = []
    for root in root_plan["roots"]:
        for task_entry in unseen["tasks"]:
            task = int(task_entry["task"])
            obj, target = TASK_META[task]
            for value in task_entry["unseen_values"]:
                index = int(value["index"])
                contexts.append({
                    "id": f"t{task}_r{root}_mu{index}",
                    "task": task,
                    "root": root,
                    "mu": float(value["mu"]),
                    "mu_index": index,
                    "interpolation_segment": value["interpolation_segment"],
                    "interpolation_fraction": float(value["interpolation_fraction"]),
                    "object": obj,
                    "target": target,
                    "split": "FINAL_UNSEEN_CONTINUOUS_FRICTION_TEST",
                })
    if len(contexts) != 48:
        raise RuntimeError("Expected exactly 48 contexts")
    write_json(out / "DEV_PLAN.json", {
        "role": "FINAL_UNSEEN_CONTINUOUS_FRICTION_TEST_ONLY",
        "roots": root_plan["roots"], "contexts": contexts, "methods": list(METHODS),
        "num_contexts": 48, "planned_valid_full_task_branches": 144,
        "model_training_or_tuning_authorized": False,
    })

    rng = random.Random(METHOD_ORDER_SEED)
    rows = []
    execution_index = 0
    for context in contexts:
        order = list(METHODS)
        rng.shuffle(order)
        for method_order, method in enumerate(order, 1):
            execution_index += 1
            rows.append({
                "execution_index": execution_index,
                "root": context["root"],
                "task": context["task"],
                "mu_test": format(context["mu"], ".17g"),
                "interpolation_segment": context["interpolation_segment"],
                "interpolation_fraction": format(context["interpolation_fraction"], ".2f"),
                "method": method,
                "method_order": method_order,
                "snapshot": str(out / "references" / context["id"] / "DECISION_STATE.pt"),
                "random_seed": context["root"],
                "context_id": context["id"],
            })
    plan_path = out / "CONTINUOUS_FRICTION_EXECUTION_PLAN.csv"
    with plan_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hash_receipt(plan_path, out / "CONTINUOUS_FRICTION_EXECUTION_PLAN_SHA256.txt")
    return contexts, rows


def snapshot_and_manifest(out: Path, audit, unseen, roots, contexts, rows):
    snapshot = out / "SOURCE_SNAPSHOT"
    snapshot.mkdir()
    inherited = ("common.py", "runtime.py", "arbitration.py", "phase_free_feasibility.py",
                 "drop_diagnostics.py", "audit_geometric_label.py")
    for name in inherited:
        shutil.copy2(PARENT_SNAPSHOT / name, snapshot / name)
    templates = {
        "worker.py": HERE / "worker.py",
        "continuous_variants.py": HERE / "continuous_variants.py",
        "launch.py": HERE / "launch.py",
        "audit_continuous_rollout.py": HERE / "audit_continuous_rollout.py",
        "run_continuous_friction.py": HERE / "run_continuous_friction.py",
        "analyze_continuous_friction.py": HERE / "analyze_continuous_friction.py",
        "prepare.py": HERE / "prepare.py",
    }
    for name, source in templates.items():
        shutil.copy2(source, snapshot / name)

    belief = read(V5 / "BELIEF_MANIFEST.json")
    feas_manifest_path = BASE / "current_fulltask_feasibility_runtime_freeze_v2_20260906/FINAL_FEASIBILITY_RUNTIME_MANIFEST.json"
    feas_manifest = read(feas_manifest_path)
    source_hashes = {str(p): sha(p) for p in snapshot.glob("*.py")}
    key_sources = [
        V5 / "BELIEF_MANIFEST.json", feas_manifest_path,
        BASE / "current_runtime_core_snapshot_v2_20260905/current_runtime_core.py",
        BASE / "current_runtime_sensor_repair_v3_candidate_20260905/measurement_hooks.py",
        ROOT / "activeforcing_current_probe.py", ROOT / "activeforcing_probe_friction_contract.py",
        ROOT / "activeforcing_execution_snapshot.py", ROOT / "activeforcing_command_handoff.py",
        Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py"),
        Path("/home/exouser/Tabero/analysis/p6g1_primitive_ik_vla_grasp_realization.py"),
        Path("/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"),
    ]
    key_sources += [Path(x["path"]) for x in belief["checkpoints"]]
    key_sources += [Path(x["path"]) for x in feas_manifest["checkpoints"]]
    source_hashes.update({str(p): sha(p) for p in key_sources})
    frozen = {
        str(out / "TRAINING_FRICTION_SUPPORT_AUDIT.json"): sha(out / "TRAINING_FRICTION_SUPPORT_AUDIT.json"),
        str(out / "FINAL_UNSEEN_FRICTION_PLAN.json"): sha(out / "FINAL_UNSEEN_FRICTION_PLAN.json"),
        str(out / "CONTINUOUS_FRICTION_ROOTS.json"): sha(out / "CONTINUOUS_FRICTION_ROOTS.json"),
        str(out / "ROOT_NONEXPOSURE_AUDIT.json"): sha(out / "ROOT_NONEXPOSURE_AUDIT.json"),
        str(out / "CONTINUOUS_FRICTION_EXECUTION_PLAN.csv"): sha(out / "CONTINUOUS_FRICTION_EXECUTION_PLAN.csv"),
        str(out / "DEV_PLAN.json"): sha(out / "DEV_PLAN.json"),
    }
    parent = SECONDARY / "SECONDARY_BASELINE_RUNTIME_PARENT_MANIFEST.json"
    manifest = {
        "version": "FINAL_UNSEEN_CONTINUOUS_FRICTION_RUNTIME_V1",
        "created_utc": now(),
        "parent_scientific_runtime": str(parent),
        "parent_scientific_runtime_sha256": sha(parent),
        "source_hashes": source_hashes,
        "frozen_artifact_hashes": frozen,
        "models_retrained": False,
        "new_training_rows": 0,
        "methods": list(METHODS),
        "tasks": sorted(TASK_META),
        "roots": roots["roots"],
        "contexts": len(contexts),
        "planned_valid_full_task_branches": len(rows),
        "physical_probe": "unchanged P4B established-grasp probe",
        "physical_belief": "unchanged frozen 58D positive-support continuous Gaussian mixture ensemble",
        "feasibility": "unchanged frozen phase-free full-task ensemble",
        "posterior_marginalization": "unchanged frozen quadrature",
        "force_support_N": [3.0, 5.0],
        "force_grid_step_N": 0.05,
        "utility": "p*(Fmax-F)/Fmax + (1-p)*(-1); lower-force tie break",
        "controller": "unchanged P4 force servo",
        "evaluator": "ONLINE_VLA_350STEP_WHOLE_MESH_RELEASE_SUPPORT_V1",
        "vla_checkpoint_sha256": "0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17",
        "DOWNSTREAM_ACTION_SOURCE": "ONLINE_VLA",
        "VLA_CHECKPOINT_LOADED": True,
        "ONLINE_POLICY_INFERENCE": True,
        "VLA_ACTION_PROVENANCE_VERIFIED": True,
        "friction_intervention": {
            "object_static_friction": "mu_test",
            "object_dynamic_friction": "mu_test",
            "finger_gelpad_case_material_modified": False,
            "material_combine_mode_modified": False,
            "interpretation": "object-side material friction intervention; not claimed as resolved pair coefficient",
            "per_branch_actual_material_readback_required": True,
        },
        "retry_policy": {
            "allowed": ["infrastructure crash", "ENOSPC before valid provenance", "policy-server failure", "corrupted snapshot", "incomplete provenance"],
            "forbidden": ["slip", "drop", "bad placement", "poor belief prediction", "strange force", "bad paper result"],
        },
        "stop_after_valid_branches": 144,
    }
    write_json(out / "CONTINUOUS_FRICTION_RUNTIME_MANIFEST.json", manifest)
    hash_receipt(out / "CONTINUOUS_FRICTION_RUNTIME_MANIFEST.json", out / "CONTINUOUS_FRICTION_RUNTIME_MANIFEST_SHA256.txt")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", type=int, nargs=2, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Freeze destination already exists")
    args.out.mkdir(parents=True)
    roots = root_audit(args.roots, args.out)
    audit = support_audit(args.out)
    unseen = unseen_plan(args.out, audit)
    contexts, rows = contexts_and_execution(args.out, unseen, roots)
    manifest = snapshot_and_manifest(args.out, audit, unseen, roots, contexts, rows)
    write_json(args.out / "PREPHYSICS_FREEZE_STATUS.json", {
        "created_utc": now(),
        "UNSEEN_FRICTION_PLAN_FROZEN": True,
        "roots_frozen": True,
        "execution_plan_frozen": True,
        "physics_branches_started": 0,
        "num_roots": 2,
        "num_unseen_mu_per_task": 6,
        "num_contexts": 48,
        "num_planned_valid_branches": 144,
        "manifest_sha256": sha(args.out / "CONTINUOUS_FRICTION_RUNTIME_MANIFEST.json"),
    })
    print(json.dumps({
        "out": str(args.out),
        "belief_training_counts": audit["NUM_BELIEF_TRAIN_FRICTION_VALUES_PER_TASK"],
        "feasibility_training_counts": audit["NUM_FEASIBILITY_TRAIN_FRICTION_VALUES_PER_TASK"],
        "roots": roots["roots"],
        "contexts": len(contexts),
        "branches": len(rows),
        "manifest_sha256": sha(args.out / "CONTINUOUS_FRICTION_RUNTIME_MANIFEST.json"),
        "physics_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
