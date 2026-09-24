from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORCES = [10.0, 12.0, 15.0, 16.0, 18.0, 20.0]
FRICTIONS = [0.425, 0.575, 0.85]
SEEDS = [200002, 200003, 200004]  # RoboTwin episode seeds (scripted layout)
STORAGE = Path("/media/volume/dasdas/exouser/af_dump_scripted_highforce_20260914")


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
        raise RuntimeError("plan/ already populated")
    sources = {
        str(HERE / name): sha(HERE / name)
        for name in [
            "EXPERIMENT_CARD.md",
            "run_scripted_context.py",
            "run_scripted_queue.py",
            "freeze_scripted_plan.py",
        ]
    }
    population = [{"friction": mu, "seed": seed} for seed in SEEDS for mu in FRICTIONS]
    protocol = {
        "version": "AF_DUMP_SCRIPTED_HIGHFORCE_V1",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "physical_root_note": "episode seeds include formal root 200002 plus 200003/200004 for layout diversity",
        "motion_mode": "scripted_play_once_full_task",
        "force_convention": "RoboTwin AF single-finger gripper drive limit N on both arms",
        "forces_N": FORCES,
        "population": population,
        "main_contexts": len(population),
        "main_rollouts": len(population) * len(FORCES),
        "smoke_forces_N": [12.0, 16.0, 20.0],
        "claim_boundary": (
            "scripted controlled-motion auxiliary feasibility / force-causal labels; "
            "identical play_once motion across forces within a seed×friction context; "
            "not online continuous AF efficacy"
        ),
        "not_claimed": [
            "online continuous ActiveForcing efficacy",
            "VLA closed-loop force-success",
            "NativeOriginal bilateral F/2 servo (this protocol uses AF single-finger caps)",
        ],
        "no_pi0": True,
        "storage_root": str(STORAGE),
        "smoke_gate": {
            "12N_must_succeed": True,
            "all_completed": True,
            "force_limits_match_command": True,
        },
        "source_hashes": sources,
    }
    write(plan / "PROTOCOL.json", protocol)
    common = {
        "task": "dump_bin_bigbin",
        "protocol_path": str(plan / "PROTOCOL.json"),
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "motion_mode": "scripted_play_once",
        "force_convention": protocol["force_convention"],
    }
    smoke = {
        **common,
        "id": "smoke_scripted_mu0.850_seed200002",
        "split": "ENGINEERING_SMOKE",
        "friction": 0.85,
        "seed": 200002,
        "forces_N": [12.0, 16.0, 20.0],
        "excluded_from_main_analysis": True,
    }
    contexts = []
    for row in population:
        contexts.append(
            {
                **common,
                "id": f"scripted_mu{row['friction']:.3f}_seed{row['seed']}",
                "split": "MAIN_SCRIPTED",
                "friction": row["friction"],
                "seed": row["seed"],
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
    print(json.dumps({"contexts": len(contexts), "rollouts": len(contexts) * len(FORCES), "smoke": smoke["id"]}, indent=2))


if __name__ == "__main__":
    main()
