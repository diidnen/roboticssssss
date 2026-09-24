from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORCES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
SEEDS_BY_FRICTION = {
    0.425: [200002, 200003, 200004],
    0.575: [200002, 200003, 200004],
    0.85: [200002, 200003, 200010],
}
STORAGE = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_20260914")

def now():
    return datetime.now(timezone.utc).isoformat()
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def main():
    plan = HERE / "plan"
    if any(plan.iterdir()):
        raise RuntimeError("plan already populated")
    sources = {str(HERE / n): sha(HERE / n) for n in [
        "run_liftstyle_context.py", "run_liftstyle_queue.py", "freeze_liftstyle_plan.py"]}
    population = [{"friction": mu, "seed": seed} for mu, seeds in SEEDS_BY_FRICTION.items() for seed in seeds]
    protocol = {
        "version": "AF_DUMP_LIFTSTYLE_FEAS_V1",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "recipe": "lift V5 mimic: scripted grasp, dump shear query, 8-step EEF hold chunk, force-forked scripted remainder",
        "forces_N": FORCES,
        "grasp_force_N": 12.0,
        "query_force_N": 4.0,
        "population": population,
        "main_contexts": 9,
        "main_rollouts": 81,
        "smoke_forces_N": [2.0, 3.5, 5.0],
        "no_pi0": True,
        "belief": "not retrained; frozen dump v4 unused at train BCE (point mu)",
        "claim_boundary": "controlled-motion auxiliary; hold-chunk features; not online AF",
        "storage_root": str(STORAGE),
        "smoke_gate": {"all_completed": True, "force_limits_match_command": True},
        "source_hashes": sources,
    }
    write(plan / "PROTOCOL.json", protocol)
    common = {
        "task": "dump_bin_bigbin",
        "protocol_path": str(plan / "PROTOCOL.json"),
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "motion_mode": "liftstyle_grasp_query_holdchunk_remainder",
    }
    smoke = {**common, "id": "smoke_liftstyle_mu0.850_seed200002", "split": "ENGINEERING_SMOKE",
             "friction": 0.85, "seed": 200002, "forces_N": [2.0, 3.5, 5.0], "excluded_from_main_analysis": True}
    contexts = [{**common, "id": f"liftstyle_mu{row['friction']:.3f}_seed{row['seed']}",
                 "split": "MAIN", "friction": row["friction"], "seed": row["seed"], "forces_N": FORCES}
                for row in population]
    write(plan / "SMOKE_CONTEXT.json", smoke)
    write(plan / "MAIN_CONTEXTS.json", contexts)
    write(plan / "FREEZE_LOCK.json", {
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
        "main_contexts_sha256": sha(plan / "MAIN_CONTEXTS.json"),
    })
    print(json.dumps({"contexts": len(contexts), "rollouts": 81}, indent=2))

if __name__ == "__main__":
    main()
