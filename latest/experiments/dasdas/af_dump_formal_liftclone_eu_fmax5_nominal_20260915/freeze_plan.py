from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
AF = Path("/media/volume/dasdas/exouser/af_dump_formal_liftclone_eu_fmax5_20260915")
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
RUNTIME = BASE / "experiments/af_dump_maxf8_20260913/additional_data_v1/RUNTIME_MANIFEST.json"
RUNTIME_SHA = "01d0289b14b812a4a12455e826a0b213ef4bf4fcdb27717892289b7e137b5034"
MODELS = AF / "models_deploy"


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
    sources = {
        str(HERE / name): sha(HERE / name)
        for name in [
            "EXPERIMENT_CARD.md",
            "audit_nominal_results.py",
            "establish_grasp.py",
            "freeze_plan.py",
            "infer_nominal_context.py",
            "run_nominal_queue.py",
        ]
    }
    protocol = {
        "version": "AF_DUMP_FORMAL_LIFTCLONE_EU_FMAX5_NOMINAL",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "physical_root": 200002,
        "claim_boundary": (
            "same-root matched Nominal on the EU Fmax5 prefix; original P4, 12N "
            "force cap, dump shear, last-command handoff, native pi0 remainder; "
            "no EU force; old pre-grasp Nominal 10/24 is not this comparator"
        ),
        "formal_contexts": 24,
        "formal_frictions": [0.425, 0.575, 0.85],
        "formal_policy_seeds": [80200002, 80200003, 80200004, 80200005, 80200006, 80200007, 80200008, 80200009],
        "methods_executed": ["Nominal Frozen VLA"],
        "methods_official_pair": str(AF),
        "methods_unofficial_only": ["Nominal Frozen VLA reused formal 10/24"],
        "methods_skipped": ["ActiveForcing", "Fixed-Strong 8N"],
        "selector": "none; native VLA remainder",
        "executed_selector": "none",
        "utility_normalization_N": None,
        "force_support": [0.25, 5.0],
        "established_grasp_N": 12.0,
        "established_grasp_rule": "keep P4 hold; 12N is force cap; grasp_actor only from open state",
        "pi0_remainder_only": True,
        "belief": "original P4 preserved; not used to select force",
        "probe_for_prefix": "original dump P4",
        "probe_for_handoff": "dump run_activeforcing_query after established grasp",
        "handoff": "last gripper command after dump shear query",
        "feature_source": "ONLINE_VLA_ACTION_CHUNK",
        "models_deploy": str(MODELS),
        "runtime_manifest": str(RUNTIME),
        "runtime_manifest_sha256": RUNTIME_SHA,
        "source_hashes": sources,
    }
    write(plan / "PROTOCOL.json", protocol)
    protocol_sha = sha(plan / "PROTOCOL.json")
    common = {
        "task": "dump_bin_bigbin",
        "root": 200002,
        "methods": ["Nominal Frozen VLA"],
        "models": str(MODELS),
        "formal_protocol_path": str(plan / "PROTOCOL.json"),
        "formal_protocol_sha256": protocol_sha,
        "runtime_manifest_path": str(RUNTIME),
        "runtime_manifest_sha256": RUNTIME_SHA,
    }
    smoke = {
        **common,
        "id": "smoke_liftclone_eu_fmax5_nominal_mu0.575_root200002_ps40200002",
        "split": "SMOKE",
        "friction": 0.575,
        "policy_seed": 40200002,
        "excluded_from_main_analysis": True,
        "matched_af_context_id": "smoke_liftclone_eu_fmax5_mu0.575_root200002_ps40200002",
    }
    contexts = []
    for seed in protocol["formal_policy_seeds"]:
        for mu in protocol["formal_frictions"]:
            contexts.append(
                {
                    **common,
                    "id": f"liftclone_eu_fmax5_nominal_mu{mu:.3f}_root200002_ps{seed}",
                    "split": "FORMAL_LIFTCLONE_EU_FMAX5_NOMINAL",
                    "friction": mu,
                    "policy_seed": seed,
                    "matched_af_context_id": f"liftclone_eu_fmax5_mu{mu:.3f}_root200002_ps{seed}",
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
