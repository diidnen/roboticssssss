from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORCES = [round(0.25 + 0.25 * i, 2) for i in range(32)]  # 0.25 .. 8.00
FRICTIONS = [0.30, 0.35, 0.425, 0.50, 0.575, 0.70, 0.75, 0.85]
SEEDS = [200002, 200003, 200010]
STORAGE = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v2_dense_20260914")
SMOKE_FORCES = [0.25, 4.0, 8.0]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    plan = HERE / "plan"
    if any(plan.iterdir()):
        raise RuntimeError("plan already populated")
    if FORCES != [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0]:
        raise RuntimeError(f"unexpected force grid {FORCES}")
    sources = {
        str(HERE / name): sha(HERE / name)
        for name in ["run_liftstyle_context.py", "run_liftstyle_queue.py", "freeze_liftstyle_plan.py"]
    }
    population = [{"friction": mu, "seed": seed} for mu in FRICTIONS for seed in SEEDS]
    protocol = {
        "version": "AF_DUMP_LIFTSTYLE_FEAS_V2",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "recipe": "lift V5 mimic v2: scripted grasp, dump shear query, 8-step EEF hold chunk, force-forked scripted remainder",
        "forces_N": FORCES,
        "force_step_N": 0.25,
        "force_band_N": [0.25, 8.0],
        "grasp_force_N": 12.0,
        "query_force_N": 4.0,
        "population": population,
        "main_contexts": 24,
        "main_rollouts": 24 * len(FORCES),
        "smoke_forces_N": SMOKE_FORCES,
        "no_pi0": True,
        "belief": "not retrained; frozen dump v4 unused at train BCE (point mu)",
        "claim_boundary": "controlled-motion auxiliary; hold-chunk features; not online AF",
        "storage_root": str(STORAGE),
        "smoke_gate": {"all_completed": True, "force_limits_match_command": True},
        "source_hashes": sources,
        "seed_split": {"TRAIN": [200002], "VAL": [200003], "TEST": [200010]},
        "predecessor": "af_dump_liftstyle_feas_20260914",
        "expansion_note": "v1 was 9 ctx x 9 F in 1-5N@0.5; v2 is 24 ctx x 32 F in 0.25-8N@0.25",
    }
    write(plan / "PROTOCOL.json", protocol)
    common = {
        "task": "dump_bin_bigbin",
        "protocol_path": str(plan / "PROTOCOL.json"),
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "motion_mode": "liftstyle_grasp_query_holdchunk_remainder",
    }
    smoke = {
        **common,
        "id": "smoke_liftstyle_v2_mu0.500_seed200002",
        "split": "ENGINEERING_SMOKE",
        "friction": 0.50,
        "seed": 200002,
        "forces_N": SMOKE_FORCES,
        "excluded_from_main_analysis": True,
    }
    contexts = [
        {
            **common,
            "id": f"liftstyle_mu{row['friction']:.3f}_seed{row['seed']}",
            "split": "MAIN",
            "friction": row["friction"],
            "seed": row["seed"],
            "forces_N": FORCES,
        }
        for row in population
    ]
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
    print(json.dumps({"contexts": len(contexts), "forces": len(FORCES), "rollouts": 24 * len(FORCES)}, indent=2))


if __name__ == "__main__":
    main()
