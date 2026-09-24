from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
AF = Path("/media/volume/dasdas/exouser/af_dump_formal_liftclone_v3_20260915")
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
            "audit_nominal_results.py",
            "deploy_prior.py",
            "establish_grasp.py",
            "freeze_plan.py",
            "hold_chunk.py",
            "infer_nominal_context.py",
            "liftstyle_runtime.py",
            "max_force_utility.py",
            "run_nominal_queue.py",
            "wait_then_run.py",
        ]
    }
    protocol = {
        "version": "AF_DUMP_FORMAL_LIFTCLONE_NOMINAL_V3",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "physical_root": 200002,
        "claim_boundary": (
            "same-root Nominal lift-clone vs AF lift-clone; "
            "12N established grasp then dump probe then native pi0 remainder; "
            "reused pre-grasp Nominal is not the matched comparator"
        ),
        "formal_contexts": 24,
        "formal_frictions": [0.425, 0.575, 0.85],
        "formal_policy_seeds": [80200002, 80200003, 80200004, 80200005, 80200006, 80200007, 80200008, 80200009],
        "methods_executed": ["Nominal Frozen VLA"],
        "methods_skipped": ["ActiveForcing", "Fixed-Strong 8N"],
        "selector": "none; native VLA remainder after lift-clone prefix",
        "established_grasp_N": 12.0,
        "pi0_remainder_only": True,
        "probe": "dump run_activeforcing_query after established grasp",
        "handoff": "last gripper command after dump shear query",
        "models_deploy": str(models.resolve()),
        "runtime_manifest": str(RUNTIME),
        "runtime_manifest_sha256": RUNTIME_SHA,
        "matched_af_experiment": str(AF),
        "source_hashes": sources,
    }
    write(plan / "PROTOCOL.json", protocol)
    protocol_sha = sha(plan / "PROTOCOL.json")
    common = {
        "task": "dump_bin_bigbin",
        "root": 200002,
        "methods": ["Nominal Frozen VLA"],
        "models": str(models.resolve()),
        "formal_protocol_path": str(plan / "PROTOCOL.json"),
        "formal_protocol_sha256": protocol_sha,
        "runtime_manifest_path": str(RUNTIME),
        "runtime_manifest_sha256": RUNTIME_SHA,
    }
    smoke = {
        **common,
        "id": "smoke_liftclone_nominal_v3_mu0.575_root200002_ps40200002",
        "split": "SMOKE",
        "friction": 0.575,
        "policy_seed": 40200002,
        "excluded_from_main_analysis": True,
        "matched_af_context_id": "smoke_liftclone_v3_mu0.575_root200002_ps40200002",
    }
    contexts = []
    for seed in protocol["formal_policy_seeds"]:
        for mu in protocol["formal_frictions"]:
            contexts.append(
                {
                    **common,
                    "id": f"liftclone_nominal_v3_mu{mu:.3f}_root200002_ps{seed}",
                    "split": "FORMAL_LIFTCLONE_NOMINAL",
                    "friction": mu,
                    "policy_seed": seed,
                    "matched_af_context_id": f"liftclone_v3_mu{mu:.3f}_root200002_ps{seed}",
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
