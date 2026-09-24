from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "af_dump_maxf8_20260913"
OLD = HERE.parent / "af_dump_original_restore_20260912"
V3 = Path("/media/volume/dasdas/exouser/af_dump_fixed_force_5_15_20260914_v3")
RUNTIME = V4 / "additional_data_v1/RUNTIME_MANIFEST.json"
FORCES = [10.0, 12.0, 15.0, 16.0, 18.0, 20.0]
RECORD_FORCE = 12.0
STORAGE = Path("/media/volume/dasdas/exouser/af_dump_openloop_highforce_20260914")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prior_12n_successes() -> list[dict]:
    rows = []
    for rec_path in sorted((V3 / "main_records").glob("*.json")):
        rec = json.loads(rec_path.read_text())
        outs = {o["force_N"]: o["success"] for o in rec["outcomes"]}
        if outs.get(12.0) != 1:
            continue
        cid = rec_path.stem
        branches = list((V3 / "main_raw" / cid / "job").glob("branch_*_12N"))
        if len(branches) != 1:
            raise RuntimeError("Expected one 12N branch for " + cid)
        branch = branches[0]
        result = json.loads((branch / "result.json").read_text())
        if not result.get("success"):
            raise RuntimeError("Record says 12N success but branch failed: " + cid)
        mu = float(cid.split("mu")[1].split("_")[0])
        seed = int(cid.split("ps")[1])
        rows.append(
            {
                "friction": mu,
                "policy_seed": seed,
                "source_context_id": cid,
                "prior_successful_branch": str(branch.resolve()),
                "prior_native_actions": int(result["native_actions"]),
                "prior_v3_successes": int(sum(outs.values())),
            }
        )
    if len(rows) < 1:
        raise RuntimeError("No prior 12N successes found in v3")
    return rows


def main() -> None:
    plan = HERE / "plan"
    plan.mkdir(exist_ok=True)
    if any(plan.iterdir()):
        raise RuntimeError("plan/ already populated")
    sources = {
        str(HERE / name): sha(HERE / name)
        for name in [
            "EXPERIMENT_CARD.md",
            "original_arbitration_binding.py",
            "openloop_execution.py",
            "infer_openloop_context.py",
            "run_openloop_queue.py",
            "freeze_openloop_plan.py",
            "audit_openloop.py",
        ]
    }
    sources.update(
        {
            str(OLD / "qualify_original_p4_native.py"): sha(OLD / "qualify_original_p4_native.py"),
            str(OLD / "native_original_force_controller.py"): sha(OLD / "native_original_force_controller.py"),
            str(OLD / "original_squeeze_inner.py"): sha(OLD / "original_squeeze_inner.py"),
        }
    )
    population = prior_12n_successes()
    # Prefer historically strong smoke: mu0.85 seed06 (5/5 in v3 including 12N)
    smoke_row = next(r for r in population if r["policy_seed"] == 80200006 and r["friction"] == 0.85)
    protocol = {
        "version": "AF_DUMP_OPENLOOP_PRIORSUCC_V2",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "physical_root": 200002,
        "motion_mode": "import_prior_successful_12N_then_openloop_replay",
        "record_force_N": RECORD_FORCE,
        "forces_N": FORCES,
        "force_support_N": [0.5, 20.0],
        "population": population,
        "main_contexts": len(population),
        "main_replay_rollouts": len(population) * len(FORCES),
        "main_record_rollouts": 0,
        "smoke_forces_N": [12.0, 16.0, 20.0],
        "claim_boundary": (
            "controlled open-loop feasibility / force-causal diagnostic labels; "
            "force feasibility conditioned on a prospectively fixed prior-successful "
            "12N trajectory imported from af_dump_fixed_force_5_15_v3; "
            "identical arm actions across forces; not online continuous AF efficacy"
        ),
        "not_claimed": [
            "online continuous ActiveForcing efficacy",
            "closed-loop VLA force-success under replanning",
            "force-neutral scripted motion population",
        ],
        "primary_endpoint": "official full-task success vs commanded force under fixed open-loop trajectory",
        "secondary_endpoints": ["contact fraction", "realized squeeze", "interval nondecreasing fractions"],
        "no_feasibility_or_utility_used": True,
        "record_parallelism": 1,
        "runtime_manifest": str(RUNTIME),
        "runtime_manifest_sha256": sha(RUNTIME),
        "storage_root": str(STORAGE),
        "prior_source": str(V3),
        "smoke_gate": {
            "12N_must_succeed": True,
            "20N_target_contact_steps_min": 50,
            "20N_max_squeeze_N": 400.0,
            "aperture_max_m": 0.04,
            "identical_first_chunk_required": True,
        },
        "source_hashes": sources,
    }
    write(plan / "PROTOCOL.json", protocol)
    common = {
        "root": 200002,
        "task": "dump_bin_bigbin",
        "runtime_manifest_path": str(RUNTIME),
        "runtime_manifest_sha256": sha(RUNTIME),
        "protocol_path": str(plan / "PROTOCOL.json"),
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "force_support_N": [0.5, 20.0],
        "record_force_N": RECORD_FORCE,
        "motion_source": "import_prior_successful_12N",
    }
    smoke = {
        **common,
        **{k: smoke_row[k] for k in ("friction", "policy_seed", "prior_successful_branch", "source_context_id")},
        "id": "smoke_priorsucc_mu0.850_ps80200006",
        "split": "ENGINEERING_SMOKE",
        "forces_N": [12.0, 16.0, 20.0],
        "excluded_from_main_analysis": True,
    }
    contexts = []
    for row in population:
        contexts.append(
            {
                **common,
                **{k: row[k] for k in ("friction", "policy_seed", "prior_successful_branch", "source_context_id")},
                "id": f"priorsucc_mu{row['friction']:.3f}_root200002_ps{row['policy_seed']}",
                "split": "MAIN_OPENLOOP",
                "forces_N": FORCES,
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
                "version": protocol["version"],
                "contexts": len(contexts),
                "replay_rollouts": len(contexts) * len(FORCES),
                "smoke": smoke["id"],
                "forces_N": FORCES,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
