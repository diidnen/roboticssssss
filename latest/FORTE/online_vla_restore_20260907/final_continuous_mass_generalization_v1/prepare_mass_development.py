#!/usr/bin/env python3
"""Freeze a development-only online-VLA transfer qualification."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/exouser/FORTE")
BASE = ROOT / "analysis/results"
FRICTION_SOURCE = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT")
OUT = HERE / "development_online"
TASKS = {0: ("alphabet_soup_1", "basket_1"), 1: ("cream_cheese_1", "basket_1"),
         5: ("tomato_sauce_1", "basket_1"), 6: ("butter_1", "basket_1")}
MASSES = (0.05, 0.20)
METHODS = ("ACTIVEFORCING_MASS", "GT_MASS", "FIXED_4")
ROOT_SEED = 181020


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def main():
    plan_path, manifest_path = HERE / "MASS_DEVELOPMENT_PLAN.json", HERE / "MASS_DEVELOPMENT_RUNTIME_MANIFEST.json"
    if plan_path.exists() or manifest_path.exists(): raise FileExistsError("MASS development already frozen")
    if not read(HERE / "MASS_BELIEF_QUALIFICATION.json").get("qualified"): raise RuntimeError("belief qualification failed")
    if not read(HERE / "MASS_FEASIBILITY_QUALIFICATION.json").get("qualified"): raise RuntimeError("feasibility qualification failed")
    if read(HERE / "MASS_CURRENT_TRAINING_COLLECTION_DATA_QUALITY.json").get("status") != "PASS":
        raise RuntimeError("current training collection data-quality audit failed")
    if read(HERE / "MASS_OFFLINE_QUALIFICATION_AUDIT.json").get("overall_assessment") != "READY_TO_SHARE":
        raise RuntimeError("independent offline qualification audit failed")
    training = read(HERE / "MASS_TRAINING_CONTEXT_PLAN.json")
    training_roots = {c["root"] for c in training["contexts"]}
    if ROOT_SEED in training_roots or ROOT_SEED in set(training["final_roots_reserved_not_training"]): raise RuntimeError("dev root collision")
    contexts = []
    for task, (obj, target) in TASKS.items():
        for mass in MASSES:
            contexts.append({"id": f"dev_t{task}_r{ROOT_SEED}_m{int(mass*1000):03d}g", "task": task,
                "root": ROOT_SEED, "mass_kg": mass, "normalized_mass_position": 0.0 if mass == MASSES[0] else 1.0,
                "mu": 0.5, "object": obj, "target": target, "split": "DEVELOPMENT_ONLY_ONLINE_VLA_TRANSFER"})
    queue = [{"context_index": index, "context_id": context["id"], "method": "REFERENCE"}
             for index, context in enumerate(contexts)]
    branches = [{"context_index": index, "context_id": context["id"], "method": method}
                for index, context in enumerate(contexts) for method in METHODS]
    random.Random(2026091001).shuffle(branches); queue.extend(branches)
    plan = {"version": "MASS_DEVELOPMENT_ONLINE_QUALIFICATION_PLAN_V1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "development_only": True, "outcome_blind": True, "root": ROOT_SEED, "contexts": contexts,
            "methods": list(METHODS), "execution_queue": queue, "references": len(contexts), "method_branches": len(branches),
            "qualification_rule": {"all_evidence_valid": True, "online_vla_verified_all": True,
                "candidate_independent_state_all": True, "true_mass_AF_leakage_count": 0,
                "phase_feature_count": 0, "selected_force_unique_count_min": 2,
                "force_tracking_auditable_all": True, "release_arbitration_violations": 0,
                "full_task_label_valid_all": True}}
    write(plan_path, plan)
    sources = [HERE / name for name in ("mass_online_worker.py", "mass_online_variants.py", "mass_online_launch.py",
        "run_mass_online_queue.py", "start_mass_policy_server.py", "mass_runtime_core.py", "mass_belief.py", "mass_feasibility.py",
        "analyze_mass_offline_qualification.py", "audit_mass_training_collection.py",
        "prepare_mass_development.py", "qualify_mass_online.py")]
    sources += [FRICTION_SOURCE / name for name in ("runtime.py", "arbitration.py", "common.py")]
    sources += [Path("/home/exouser/FORTE/online_vla_restore_20260907/server_v3_frozen/SOURCE_SNAPSHOT/server.py"),
                Path("/home/exouser/FORTE/online_vla_restore_20260907/server_v3_frozen/SOURCE_SNAPSHOT/common.py")]
    sources += [BASE / "current_runtime_sensor_repair_v3_candidate_20260905/measurement_hooks.py",
                BASE / "current_runtime_branch_execution_v6_20260905/branch_execution.py",
                ROOT / "activeforcing_current_probe.py", ROOT / "current_contract_belief_features.py"]
    artifacts = [HERE / name for name in ("MASS_TRAINING_PROTOCOL.json", "MASS_TRAINING_RUNTIME_MANIFEST.json",
        "MASS_BELIEF_MANIFEST.json", "MASS_BELIEF_QUALIFICATION.json", "MASS_FEASIBILITY_MANIFEST.json",
        "MASS_FEASIBILITY_QUALIFICATION.json", "MASS_BELIEF_RUNTIME_PARITY_REPAIR.json", "MASS_QUERY_PROTOCOL.md",
        "MASS_OFFLINE_QUALIFICATION_AUDIT.json", "MASS_FEASIBILITY_MASS_FORCE_INSPECTION.csv",
        "MASS_OFFLINE_QUALIFICATION_REPORT.md", "MASS_BELIEF_IDENTITY_CONTROL.json")]
    artifacts += [HERE / "MASS_CURRENT_TRAINING_COLLECTION_DATA_QUALITY.json",
                  HERE / "MASS_CURRENT_TRAINING_COLLECTION_DATA_QUALITY.md"]
    for model_manifest in ("MASS_BELIEF_MANIFEST.json", "MASS_FEASIBILITY_MANIFEST.json"):
        artifacts += [Path(row["path"]) for row in read(HERE / model_manifest)["checkpoints"]]
    manifest = {"version": "MASS_DEVELOPMENT_CURRENT_ONLINE_VLA_RUNTIME_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "plan_path": str(plan_path), "plan_sha256": sha(plan_path),
        "source_hashes": {str(path): sha(path) for path in sources},
        "artifact_hashes": {str(path): sha(path) for path in artifacts},
        "query_runtime_manifest_sha256": sha(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"),
        "phase_free": True, "online_vla": True, "replan_steps": 10, "predicted_chunk_steps": 50,
        "downstream_horizon": 350, "force_support_N": [3.0, 5.0], "force_grid_step_N": 0.05,
        "terminal_evaluator": "ONLINE_VLA_350STEP_WHOLE_MESH_RELEASE_SUPPORT_V1",
        "fixed_friction": 0.5, "final_roots_exposed": False}
    write(manifest_path, manifest)
    print(json.dumps({"frozen": True, "contexts": len(contexts), "branches": len(branches),
                      "plan_sha256": sha(plan_path), "runtime_manifest_sha256": sha(manifest_path)}, indent=2))


if __name__ == "__main__": main()
