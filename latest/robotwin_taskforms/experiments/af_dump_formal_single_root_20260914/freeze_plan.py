from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
V4 = HERE.parent / "af_dump_maxf8_20260913"
OLD = HERE.parent / "af_dump_original_restore_20260912"
MODELS = V4 / "models_v4"
RUNTIME = V4 / "additional_data_v1/RUNTIME_MANIFEST.json"
SEEDS = list(range(80200002, 80200010))
FRICTIONS = [0.425, 0.575, 0.85]
METHODS = ["Nominal Frozen VLA", "Fixed-Strong 8N", "ActiveForcing"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rotate(values: list[str], offset: int) -> list[str]:
    offset %= len(values)
    return values[offset:] + values[:offset]


def main() -> None:
    plan = HERE / "plan"
    plan.mkdir(exist_ok=False)
    training = read(MODELS / "TRAINING_COMPLETE.json")
    if not training.get("completed") or training.get("test_groups_executed") != 0:
        raise ValueError("Frozen training completion gate failed")
    stage = read(HERE.parent / "af_motion_diversity_study_20260913/STAGE_I_BUDGET_REVISION_01.json")
    exposed = {30200002, 40200002, 50200002, 60200002, *stage["retained_policy_seeds"], *stage["reserved_unrun_policy_seeds"]}
    if exposed.intersection(SEEDS):
        raise ValueError("Formal seed overlaps a known training/development/Stage-I seed")
    source_names = [
        "freeze_plan.py",
        "infer_formal_context.py",
        "run_formal_queue.py",
        "audit_formal_results.py",
        "EXPERIMENT_CARD.md",
    ]
    sources = {str(HERE / name): sha(HERE / name) for name in source_names}
    sources.update(
        {
            str(V4 / "infer_v4.py"): sha(V4 / "infer_v4.py"),
            str(V4 / "maxf8_runtime.py"): sha(V4 / "maxf8_runtime.py"),
            str(V4 / "max_force_utility.py"): sha(V4 / "max_force_utility.py"),
            str(OLD / "qualify_original_online_forks.py"): sha(OLD / "qualify_original_online_forks.py"),
            str(OLD / "native_original_force_controller.py"): sha(OLD / "native_original_force_controller.py"),
        }
    )
    protocol = {
        "version": "AF_DUMP_SINGLE_ROOT_FORMAL_V1",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "physical_root": 200002,
        "existing_feasibility_labels": 192,
        "frozen_models": str(MODELS),
        "training_complete_sha256": sha(MODELS / "TRAINING_COMPLETE.json"),
        "runtime_manifest": str(RUNTIME),
        "runtime_manifest_sha256": sha(RUNTIME),
        "formal_policy_seeds": SEEDS,
        "formal_frictions": FRICTIONS,
        "formal_contexts": 24,
        "formal_methods": METHODS,
        "formal_rollouts": 72,
        "smoke_rollouts": 1,
        "method_order_rule": "left rotation by seed-index plus friction-index modulo 3",
        "no_new_roots": True,
        "no_retraining_or_reselection": True,
        "claim_boundary": "same-root held-out-path performance only",
        "source_hashes": sources,
    }
    write(plan / "PROTOCOL.json", protocol)
    protocol_hash = sha(plan / "PROTOCOL.json")
    common = {
        "root": 200002,
        "task": "dump_bin_bigbin",
        "models": str(MODELS),
        "runtime_manifest_path": str(RUNTIME),
        "runtime_manifest_sha256": sha(RUNTIME),
        "formal_protocol_path": str(plan / "PROTOCOL.json"),
        "formal_protocol_sha256": protocol_hash,
    }
    smoke = {
        **common,
        "id": "smoke_mu0.575_root200002_ps40200002",
        "split": "SMOKE",
        "friction": 0.575,
        "policy_seed": 40200002,
        "methods": ["Nominal Frozen VLA"],
    }
    formal = []
    for seed_index, seed in enumerate(SEEDS):
        for friction_index, friction in enumerate(FRICTIONS):
            formal.append(
                {
                    **common,
                    "id": f"formal_mu{friction:.3f}_root200002_ps{seed}",
                    "split": "FORMAL_TEST",
                    "friction": friction,
                    "policy_seed": seed,
                    "methods": rotate(METHODS, seed_index + friction_index),
                }
            )
    write(plan / "SMOKE_CONTEXT.json", smoke)
    write(plan / "FORMAL_CONTEXTS.json", formal)
    write(
        plan / "NON_EXPOSURE_AUDIT.json",
        {
            "created_utc": now(),
            "passed": True,
            "formal_policy_seeds": SEEDS,
            "known_exposed_policy_seeds": sorted(exposed),
            "overlap": [],
            "targeted_structured_repository_search_matches": 0,
            "search_scope": "experiment JSON/JSONL policy_seed fields and runner policy-seed arguments, excluding this new formal directory",
        },
    )
    write(
        plan / "FREEZE_LOCK.json",
        {
            "protocol_sha256": protocol_hash,
            "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
            "formal_contexts_sha256": sha(plan / "FORMAL_CONTEXTS.json"),
            "non_exposure_audit_sha256": sha(plan / "NON_EXPOSURE_AUDIT.json"),
        },
    )
    print(json.dumps({"frozen": str(plan), "smoke": 1, "formal_contexts": len(formal), "formal_rollouts": len(formal) * 3}, indent=2))


if __name__ == "__main__":
    main()

