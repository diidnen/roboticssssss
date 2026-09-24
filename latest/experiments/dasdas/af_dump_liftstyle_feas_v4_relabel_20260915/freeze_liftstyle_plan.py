from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORCES = [round(0.25 * i, 2) for i in range(1, 21)]  # 0.25 .. 5.00
FRICTIONS = [0.30, 0.35, 0.425, 0.50, 0.575, 0.70, 0.75, 0.85]
SEEDS = [200002, 200003, 200010, 200014]
STORAGE = HERE
SMOKE_FORCES = [0.25, 2.5, 5.0]


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
    if FORCES != [
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.25,
        4.5,
        4.75,
        5.0,
    ]:
        raise RuntimeError(f"unexpected force grid {FORCES}")
    sources = {
        str(HERE / name): sha(HERE / name)
        for name in ["run_liftstyle_context.py", "run_liftstyle_queue.py", "freeze_liftstyle_plan.py"]
    }
    population = [{"friction": mu, "seed": seed} for mu in FRICTIONS for seed in SEEDS]
    protocol = {
        "version": "AF_DUMP_LIFTSTYLE_FEAS_V4_RELABEL",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "recipe": "lift V5 snapshot fork: scripted grasp, dump shear query, one 8-step EEF hold chunk, restore physics and force-fork scripted remainder",
        "forces_N": FORCES,
        "force_step_N": 0.25,
        "force_band_N": [0.25, 5.0],
        "grasp_force_N": 12.0,
        "query_force_N": 4.0,
        "population": population,
        "main_contexts": 32,
        "main_rollouts": 32 * len(FORCES),
        "smoke_forces_N": SMOKE_FORCES,
        "no_pi0": True,
        "snapshot_fork": True,
        "belief": "not retrained here; frozen dump v4 unused at train BCE (point mu)",
        "claim_boundary": "controlled-motion auxiliary; hold-chunk features; not pi0; not online AF; not official 18/19",
        "storage_root": str(STORAGE),
        "smoke_gate": {"all_completed": True, "force_limits_match_command": True},
        "source_hashes": sources,
        "seed_split": {
            "TRAIN": [200002, 200003],
            "VAL": [200010],
            "TEST": [200014],
        },
        "predecessor": "af_dump_liftstyle_feas_v3_fork_20260914",
        "expansion_note": (
            "v3 VAL seed 200003 was 160/160 success. v4 keeps 200003 in TRAIN, uses 200010 as VAL "
            "(v3 TRAIN+TEST mixed labels), TEST seed 200014 (probed stable). Seeds 200004/200006/200012 "
            "UnStableError on garbage actors. Same lift-style method; grid [0.25,5]."
        ),
        "official_records_untouched": "/media/volume/dasdas/exouser/af_dump_formal_liftclone_eu_fmax5_20260915",
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
        "id": "smoke_liftstyle_v4_mu0.500_seed200002",
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
    print(json.dumps({"contexts": len(contexts), "forces": len(FORCES), "rollouts": 32 * len(FORCES)}, indent=2))


if __name__ == "__main__":
    main()
