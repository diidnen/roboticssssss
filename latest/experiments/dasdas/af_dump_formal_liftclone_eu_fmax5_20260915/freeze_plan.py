from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
RUNTIME = BASE / "experiments/af_dump_maxf8_20260913/additional_data_v1/RUNTIME_MANIFEST.json"
RUNTIME_SHA = "01d0289b14b812a4a12455e826a0b213ef4bf4fcdb27717892289b7e137b5034"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError("refusing to overwrite " + str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    plan = HERE / "plan"
    models = HERE / "models_deploy"
    sources = {
        str(HERE / name): sha(HERE / name)
        for name in [
            "EXPERIMENT_CARD.md",
            "audit_formal_results.py",
            "establish_grasp.py",
            "freeze_plan.py",
            "infer_af_context.py",
            "liftstyle_runtime.py",
            "max_force_utility.py",
            "run_af_queue.py",
        ]
    }
    protocol = {
        "version": "AF_DUMP_FORMAL_LIFTCLONE_EU_FMAX5",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "physical_root": 200002,
        "claim_boundary": (
            "same-root AF lift-clone EU Fmax5; original P4 v4 belief, 12N grasp, "
            "dump shear, last-command handoff, pi0 remainder; executed EU maxF=5 "
            "on [0.25,5]@0.05; official comparator is matched Nominal on this "
            "prefix when that queue exists; old pre-grasp Nominal 10/24 is not it"
        ),
        "formal_contexts": 24,
        "formal_frictions": [0.425, 0.575, 0.85],
        "formal_policy_seeds": [80200002, 80200003, 80200004, 80200005, 80200006, 80200007, 80200008, 80200009],
        "methods_executed": ["ActiveForcing"],
        "methods_official_comparator": "matched Nominal on this prefix; not yet run",
        "methods_unofficial_only": ["Nominal Frozen VLA reused formal 10/24"],
        "methods_skipped": ["Fixed-Strong 8N"],
        "selector": "LiftstyleFeasibility Fmax5 expected_utility on [0.25,5]@0.05 maxF=5",
        "executed_selector": "expected_utility",
        "utility_normalization_N": 5.0,
        "force_support": [0.25, 5.0],
        "planner_grid_step": 0.05,
        "established_grasp_N": 12.0,
        "established_grasp_rule": "keep P4 hold; 12N is force cap; grasp_actor only from open state",
        "pi0_remainder_only": True,
        "belief": "frozen dump v4 on original P4 rows",
        "probe_for_belief": "original dump P4",
        "probe_for_handoff": "dump run_activeforcing_query after established grasp",
        "handoff": "last gripper command after dump shear query",
        "feature_source": "ONLINE_VLA_ACTION_CHUNK",
        "models_deploy": str(models),
        "runtime_manifest": str(RUNTIME),
        "runtime_manifest_sha256": RUNTIME_SHA,
        "source_hashes": sources,
    }
    write(plan / "PROTOCOL.json", protocol)
    protocol_sha = sha(plan / "PROTOCOL.json")
    common = {
        "task": "dump_bin_bigbin",
        "root": 200002,
        "methods": ["ActiveForcing"],
        "models": str(models),
        "formal_protocol_path": str(plan / "PROTOCOL.json"),
        "formal_protocol_sha256": protocol_sha,
        "runtime_manifest_path": str(RUNTIME),
        "runtime_manifest_sha256": RUNTIME_SHA,
    }
    smoke = {
        **common,
        "id": "smoke_liftclone_eu_fmax5_mu0.575_root200002_ps40200002",
        "split": "SMOKE",
        "friction": 0.575,
        "policy_seed": 40200002,
        "excluded_from_main_analysis": True,
    }
    contexts = []
    for seed in protocol["formal_policy_seeds"]:
        for mu in protocol["formal_frictions"]:
            contexts.append(
                {
                    **common,
                    "id": f"liftclone_eu_fmax5_mu{mu:.3f}_root200002_ps{seed}",
                    "split": "FORMAL_LIFTCLONE_EU_FMAX5",
                    "friction": mu,
                    "policy_seed": seed,
                }
            )
    write(plan / "SMOKE_CONTEXT.json", smoke)
    write(plan / "FORMAL_CONTEXTS.json", contexts)
    write(
        plan / "FREEZE_LOCK.json",
        {
            "protocol_sha256": protocol_sha,
            "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
            "formal_contexts_sha256": sha(plan / "FORMAL_CONTEXTS.json"),
        },
    )
    print(json.dumps({"contexts": len(contexts), "protocol_sha256": protocol_sha}, indent=2))


if __name__ == "__main__":
    main()
