"""PRE-only scan: which (seed, episode, μ) are VALID wall pinches. No F sweep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grasp_geom_logger import GraspGeomLogger
from run_isolation_sweep import (
    OFFICIAL_MU,
    apply_grasp_surface_materials,
    apply_retention_collision_filters,
    classify_pre,
    pack_state,
    scripted_establish_grasp,
    write,
)


def scan(seed: int) -> list[dict]:
    from run_isolation_sweep import make_task
    rows = []
    for mu in OFFICIAL_MU:
        for ep in range(10):
            task = None
            try:
                task = make_task(seed, mu, ep)
                apply_grasp_surface_materials(task, mu)
                apply_retention_collision_filters(task)
                task.activate_activeforcing_candidate_force()
                scripted_establish_grasp(task)
                helper = GraspGeomLogger(task, HERE / "_tmp_geom", 0.0)
                window = []
                for _ in range(20):
                    task.scene.step()
                    window.append(pack_state(task, helper))
                gate = classify_pre(window)
                row = {
                    "seed": seed,
                    "mu": mu,
                    "episode_id": ep,
                    "gate": gate["gate"],
                    "cr": gate["window_cr"],
                    "ap_mm": 1000.0 * float(window[-1]["aperture"]),
                    "squeeze": window[-1]["squeeze"],
                    "nxnx": window[-1]["nx_nx"],
                }
            except Exception as exc:
                row = {"seed": seed, "mu": mu, "episode_id": ep, "gate": "ERROR", "error": repr(exc)}
            print(json.dumps(row), flush=True)
            rows.append(row)
            if task is not None:
                try:
                    task.close_env()
                except Exception:
                    pass
            if row.get("gate") == "ERROR" and "UnStableError" in row.get("error", ""):
                break
    return rows


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 200014
    rows = scan(seed)
    write(HERE / f"PRE_SCAN_seed{seed}.json", {"rows": rows})
    valid = {}
    for row in rows:
        if row.get("gate") == "VALID":
            valid.setdefault(row["episode_id"], []).append(row["mu"])
    common = [ep for ep, mus in valid.items() if set(mus) >= set([0.425, 0.575, 0.85])]
    print(json.dumps({"common_valid_episodes_all_mu": common, "valid_by_ep": valid}), flush=True)
