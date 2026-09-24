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
V2 = HERE.parent / "af_dump_fixed_force_5_15_20260914_v2"
V2_SMOKE = V2 / "smoke_records/smoke_mu0.850_root200002_ps80200002_8v15.json"
FORCES = [5.0, 8.0, 10.0, 12.0, 15.0]
FRICTIONS = [0.425, 0.575, 0.85]
POLICY_SEEDS = list(range(80200002, 80200010))


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


def rotate(values: list[float], offset: int) -> list[float]:
    offset %= len(values)
    return values[offset:] + values[:offset]


def main() -> None:
    plan = HERE / "plan"
    plan.mkdir(exist_ok=True)
    if any(plan.iterdir()):
        raise RuntimeError("plan/ already populated; refuse to overwrite freeze")
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
        }
    )
    population = [
        {"friction": friction, "policy_seed": seed}
        for seed in POLICY_SEEDS
        for friction in FRICTIONS
    ]
    protocol = {
        "version": "AF_DUMP_FIXED_FORCE_5_15_POSTHOC_V3_COMMANDED_SETPOINT",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "physical_root": 200002,
        "population_source": "all 24 contexts from the frozen formal same-root held-out-path evaluation",
        "population": population,
        "forces_N": FORCES,
        "main_contexts": len(population),
        "main_rollouts": len(population) * len(FORCES),
        "smoke_forces_N": [8.0, 15.0],
        "smoke_rollouts": 2,
        "force_order_rule": "left rotation by context index modulo five",
        "primary_endpoint": "official full-task success per commanded force setpoint",
        "secondary_endpoints": [
            "target-contact fraction",
            "target-contact realized squeeze / command",
            "post-contact loss duration",
        ],
        "paired_analysis": "each 10/12/15 N arm versus 8 N with exact McNemar and paired outcome table",
        "main_stop_rule": "retain and run all 24 contexts regardless of interim outcomes; stop only on engineering/audit failure",
        "engineering_admission_rule": "reuse V2 raw-audited smoke; exact command dispatch, finite bounded aperture, at least 100 target-contact steps at 15 N command, and raw impulse peak below 150 N",
        "no_new_roots": True,
        "no_feasibility_or_utility_used": True,
        "posthoc_range_diagnostic": True,
        "storage_root": "/media/volume/dasdas/exouser/af_dump_fixed_force_5_15_20260914_v3",
        "supersedes_v2_tracking_gate": str(V2),
        "v2_interpretation_change": "15 N command is not a realized 15 N load; label all arms as commanded setpoints and report realized squeeze",
        "runtime_manifest": str(RUNTIME),
        "runtime_manifest_sha256": sha(RUNTIME),
        "claim_boundary": "same-root posthoc physical force-range diagnostic; >8 N is outside feasibility training support",
        "source_hashes": sources,
    }
    prior = read(V2_SMOKE)
    prior_rows = {float(row["force_N"]): row for row in prior["outcomes"]}
    prior_checks = {
        "both_commands_present": sorted(prior_rows) == [8.0, 15.0],
        "all_raw_audits_pass": all(row["raw_audit_passed"] for row in prior_rows.values()),
        "15N_target_contact_steps_at_least_100": prior_rows[15.0]["target_contact_steps"] >= 100,
        "15N_raw_peak_below_150N": prior_rows[15.0]["max_squeeze_N"] < 150,
        "apertures_within_physical_bounds": all(0 <= row["active_min_aperture_m"] <= 0.04 for row in prior_rows.values()),
    }
    if not all(prior_checks.values()):
        raise ValueError("Prior V2 engineering smoke cannot admit V3")
    protocol["prior_engineering_smoke"] = {
        "record_path": str(V2_SMOKE),
        "record_sha256": sha(V2_SMOKE),
        "checks": prior_checks,
        "characterization": {
            "8N_success": prior_rows[8.0]["success"],
            "15N_success": prior_rows[15.0]["success"],
            "8N_contact_fraction": prior_rows[8.0]["target_contact_fraction"],
            "15N_contact_fraction": prior_rows[15.0]["target_contact_fraction"],
            "8N_realized_contact_mean_N": prior_rows[8.0]["target_contact_mean_squeeze_N"],
            "15N_realized_contact_mean_N": prior_rows[15.0]["target_contact_mean_squeeze_N"],
            "15N_tracking_ratio": prior_rows[15.0]["target_contact_tracking_ratio"],
            "15N_aperture_saturation_fraction": prior_rows[15.0]["active_zero_aperture_fraction"],
        },
        "excluded_from_main_analysis": True,
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
        "id": "smoke_mu0.850_root200002_ps80200002_8v15",
        "split": "ENGINEERING_SMOKE",
        "friction": 0.85,
        "policy_seed": 80200002,
        "forces_N": [8.0, 15.0],
        "excluded_from_main_analysis": True,
        "selection_reason": "pre-existing Fixed-8 success provides stable contact for engineering tracking admission",
    }
    contexts = []
    for index, row in enumerate(population):
        friction = row["friction"]
        seed = row["policy_seed"]
        contexts.append(
            {
                **common,
                "id": f"range_mu{friction:.3f}_root200002_ps{seed}",
                "split": "MAIN_POSTHOC_RANGE",
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
    print(json.dumps({"contexts": len(contexts), "main_rollouts": len(contexts) * 5, "smoke_rollouts": 2}, indent=2))


if __name__ == "__main__":
    main()
