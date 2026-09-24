from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
OLD = HERE.parent / "af_dump_original_restore_20260912"
V4 = HERE.parent / "af_dump_maxf8_20260913"
RUNTIME = V4 / "additional_data_v1/RUNTIME_MANIFEST.json"
PRIOR = HERE.parent / "af_dump_fixed_force_5_15_20260914_v3"
FORCES = [16.0, 18.0, 20.0]
FRICTIONS = [0.425, 0.575, 0.85]
POLICY_SEEDS = list(range(80200002, 80200010))
SMOKE_FORCES = [16.0, 20.0]
STORAGE = Path("/media/volume/dasdas/exouser/af_dump_fixed_force_9_20_20260914")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rotate(values: list[float], offset: int) -> list[float]:
    offset %= len(values)
    return values[offset:] + values[:offset]


def main() -> None:
    plan = HERE / "plan"
    plan.mkdir(exist_ok=True)
    if any(plan.iterdir()):
        raise RuntimeError("plan/ already populated; refuse to overwrite freeze")
    if not (PRIOR / "FINAL_RESULTS.json").exists():
        raise RuntimeError("Prior 5-15 sweep FINAL_RESULTS missing; cannot claim reuse")
    source_names = [
        "original_arbitration_binding.py",
        "infer_fixed_sweep_context.py",
        "audit_fixed_sweep.py",
        "freeze_fixed_sweep.py",
        "run_fixed_sweep_queue.py",
        "EXPERIMENT_CARD.md",
    ]
    sources = {str(HERE / name): sha(HERE / name) for name in source_names}
    sources.update(
        {
            str(OLD / "qualify_original_online_forks.py"): sha(OLD / "qualify_original_online_forks.py"),
            str(OLD / "native_original_force_controller.py"): sha(OLD / "native_original_force_controller.py"),
            str(OLD / "original_squeeze_inner.py"): sha(OLD / "original_squeeze_inner.py"),
            str(OLD / "qualify_original_p4_native.py"): sha(OLD / "qualify_original_p4_native.py"),
            str(PRIOR / "FINAL_RESULTS.json"): sha(PRIOR / "FINAL_RESULTS.json"),
        }
    )
    population = [
        {"friction": friction, "policy_seed": seed}
        for seed in POLICY_SEEDS
        for friction in FRICTIONS
    ]
    protocol = {
        "version": "AF_DUMP_FIXED_FORCE_16_18_20_UPWARD_FILL_V1",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "physical_root": 200002,
        "population_source": "same 24 contexts as prior fixed 5/8/10/12/15 sweep",
        "population": population,
        "forces_N": FORCES,
        "main_contexts": len(population),
        "main_rollouts": len(population) * len(FORCES),
        "smoke_forces_N": SMOKE_FORCES,
        "smoke_rollouts": len(SMOKE_FORCES),
        "force_order_rule": "left rotation by context index modulo three",
        "primary_endpoint": "official full-task success at commanded 16/18/20 N",
        "secondary_endpoints": [
            "target-contact fraction",
            "target-contact realized squeeze / command",
            "joined monotonicity with prior 10/12/15 on same context_id",
        ],
        "paired_analysis": "join prior FINAL_RESULTS 10/12/15 with new 16/18/20; report interval nondecreasing fractions",
        "main_stop_rule": "retain and run all 24 contexts; stop only on engineering/audit failure",
        "engineering_admission_rule": "smoke 16/20 N; raw audits pass; 20 N >=100 target-contact steps; aperture in [0,0.04]; max squeeze <=400 N",
        "no_new_roots": True,
        "no_feasibility_or_utility_used": True,
        "reuses_prior_5_15_sweep": str(PRIOR / "FINAL_RESULTS.json"),
        "prior_forces_N_already_collected": [5.0, 8.0, 10.0, 12.0, 15.0],
        "storage_root": str(STORAGE),
        "runtime_manifest": str(RUNTIME),
        "runtime_manifest_sha256": sha(RUNTIME),
        "claim_boundary": "same-root upward commanded-force fill on existing contexts; commanded != realized load",
        "smoke_gate": {
            "20N_target_contact_steps_min": 100,
            "20N_max_squeeze_N": 400.0,
            "aperture_max_m": 0.04,
        },
        "source_hashes": sources,
        "eta_note": "approx 2.5-3.5 h based on prior ~2.3 min/force",
    }
    write(plan / "PROTOCOL.json", protocol)
    common = {
        "root": 200002,
        "task": "dump_bin_bigbin",
        "runtime_manifest_path": str(RUNTIME),
        "runtime_manifest_sha256": sha(RUNTIME),
        "protocol_path": str(plan / "PROTOCOL.json"),
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
    }
    smoke = {
        **common,
        "id": "smoke_mu0.850_root200002_ps80200002_16v20",
        "split": "ENGINEERING_SMOKE",
        "friction": 0.85,
        "policy_seed": 80200002,
        "forces_N": SMOKE_FORCES,
        "excluded_from_main_analysis": True,
        "selection_reason": "known-contact formal seed for 20 N engineering admission",
    }
    contexts = []
    for index, row in enumerate(population):
        friction = row["friction"]
        seed = row["policy_seed"]
        contexts.append(
            {
                **common,
                "id": f"range_mu{friction:.3f}_root200002_ps{seed}",
                "split": "MAIN_UPWARD_FILL",
                "friction": friction,
                "policy_seed": seed,
                "forces_N": rotate(FORCES, index),
            }
        )
    write(plan / "SMOKE_CONTEXT.json", smoke)
    write(plan / "MAIN_CONTEXTS.json", contexts)
    write(
        plan / "FREEZE_LOCK.json",
        {
            "protocol_sha256": sha(plan / "PROTOCOL.json"),
            "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
            "main_contexts_sha256": sha(plan / "MAIN_CONTEXTS.json"),
        },
    )
    print(
        json.dumps(
            {
                "contexts": len(contexts),
                "main_rollouts": len(contexts) * len(FORCES),
                "smoke_rollouts": len(SMOKE_FORCES),
                "forces_N": FORCES,
                "seeds": POLICY_SEEDS,
                "joins_prior": str(PRIOR / "FINAL_RESULTS.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
