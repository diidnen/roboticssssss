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
METHODS = ["Nominal Frozen VLA", "Fixed-Strong 8N", "ActiveForcing"]
# Feasibility VAL population: held-out friction, in-distribution motion seeds.
VAL_POPULATION = [
    (0.325, 30200002),
    (0.475, 30200002),
    (0.625, 30200002),
    (0.775, 30200002),
    (0.475, 40200002),
    (0.775, 40200002),
]


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
    if any(plan.iterdir()):
        raise RuntimeError("plan/ already populated; refuse to overwrite freeze")
    training = read(MODELS / "TRAINING_COMPLETE.json")
    if not training.get("completed") or training.get("test_groups_executed") != 0:
        raise ValueError("Frozen training completion gate failed")
    source_names = [
        "freeze_val_plan.py",
        "infer_val_context.py",
        "run_val_queue.py",
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
        "version": "AF_DUMP_VAL_POLICY_SANITY_V1",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "physical_root": 200002,
        "existing_feasibility_labels": 192,
        "frozen_models": str(MODELS),
        "training_complete_sha256": sha(MODELS / "TRAINING_COMPLETE.json"),
        "runtime_manifest": str(RUNTIME),
        "runtime_manifest_sha256": sha(RUNTIME),
        "val_population": [{"friction": f, "policy_seed": s} for f, s in VAL_POPULATION],
        "val_contexts": len(VAL_POPULATION),
        "val_methods": METHODS,
        "val_rollouts": len(VAL_POPULATION) * 3,
        "smoke_rollouts": 1,
        "method_order_rule": "left rotation by context-index modulo 3",
        "no_new_roots": True,
        "no_retraining_or_reselection": True,
        "storage_root": "/media/volume/dasdas/exouser/af_dump_val_policy_sanity_20260914",
        "claim_boundary": "feasibility-VAL motion/friction policy sanity only",
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
    contexts = []
    for index, (friction, seed) in enumerate(VAL_POPULATION):
        contexts.append(
            {
                **common,
                "id": f"valpol_mu{friction:.3f}_root200002_ps{seed}",
                "split": "VAL_POLICY",
                "friction": friction,
                "policy_seed": seed,
                "methods": rotate(METHODS, index),
            }
        )
    write(plan / "SMOKE_CONTEXT.json", smoke)
    write(plan / "VAL_CONTEXTS.json", contexts)
    write(
        plan / "FREEZE_LOCK.json",
        {
            "protocol_sha256": sha(plan / "PROTOCOL.json"),
            "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
            "val_contexts_sha256": sha(plan / "VAL_CONTEXTS.json"),
        },
    )
    print(json.dumps({"contexts": len(contexts), "rollouts": len(contexts) * 3}, indent=2))


if __name__ == "__main__":
    main()
