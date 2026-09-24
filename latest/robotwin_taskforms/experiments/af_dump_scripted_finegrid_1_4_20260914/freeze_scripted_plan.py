from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORCES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
SEEDS_BY_FRICTION = {
    0.425: [200002, 200003, 200004],
    0.575: [200002, 200003, 200004],
    0.85: [200002, 200003, 200010],
}
STORAGE = Path("/media/volume/dasdas/exouser/af_dump_scripted_finegrid_1_4_20260914")

def now():
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
    sources = {str(HERE / n): sha(HERE / n) for n in [
        "run_scripted_context.py", "run_scripted_queue.py", "freeze_scripted_plan.py"]}
    population = [{"friction": mu, "seed": seed}
                  for mu, seeds in SEEDS_BY_FRICTION.items() for seed in seeds]
    protocol = {
        "version": "AF_DUMP_SCRIPTED_FINEGRID_1_4_V1",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "motion_mode": "scripted_play_once_full_task",
        "force_convention": "RoboTwin AF single-finger gripper drive limit N on both arms",
        "forces_N": FORCES,
        "grid_step_N": 0.25,
        "population": population,
        "main_contexts": len(population),
        "main_rollouts": len(population) * len(FORCES),
        "smoke_forces_N": [1.0, 2.5, 4.0],
        "claim_boundary": "scripted 1-4N 0.25-step auxiliary labels; not online AF; not pi0 chunks",
        "no_pi0": True,
        "storage_root": str(STORAGE),
        "smoke_gate": {"all_completed": True, "force_limits_match_command": True},
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
    smoke = {**common, "id": "smoke_finegrid_1_4_mu0.850_seed200002", "split": "ENGINEERING_SMOKE",
             "friction": 0.85, "seed": 200002, "forces_N": [1.0, 2.5, 4.0], "excluded_from_main_analysis": True}
    contexts = [{**common, "id": f"scripted_mu{row['friction']:.3f}_seed{row['seed']}",
                 "split": "MAIN_SCRIPTED", "friction": row["friction"], "seed": row["seed"], "forces_N": FORCES}
                for row in population]
    write(plan / "SMOKE_CONTEXT.json", smoke)
    write(plan / "MAIN_CONTEXTS.json", contexts)
    write(plan / "FREEZE_LOCK.json", {
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
        "main_contexts_sha256": sha(plan / "MAIN_CONTEXTS.json"),
    })
    print(json.dumps({"contexts": len(contexts), "rollouts": len(contexts)*len(FORCES)}, indent=2))

if __name__ == "__main__":
    main()
