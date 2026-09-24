"""Adjudicate the frozen V6 minimum-force interface gate."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

OUT = Path("/media/volume/data/exouser/activeforcing_table_push_v6_20260910")
CONTRACT = json.loads((OUT / "R2_CONTRACT.json").read_text())


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def load(step, kind, force):
    return json.loads((OUT / "r2" / f"s{step}_{kind}_{force:+.0f}N.json").read_text())


def main():
    output = OUT / "R2_RESULT.json"
    if output.exists():
        raise RuntimeError(f"immutable result exists: {output}")
    records = {}
    parity = {}
    all_safe = True
    all_nondecreasing = True
    all_strict_enough = True
    all_never_reduce = True
    for step in CONTRACT["steps"]:
        untouched = load(step, "untouched", 0)
        disabled = load(step, "disabled", 0)
        parity[str(step)] = {
            "post_state_exact": untouched["post_state_sha256"] == disabled["post_state_sha256"],
            "post_qpos_exact": untouched["post_qpos"] == disabled["post_qpos"],
            "post_qvel_exact": untouched["post_qvel"] == disabled["post_qvel"],
            "final_ctrl_exact": untouched["final_ctrl"] == disabled["final_ctrl"],
        }
        branch = [load(step, "minimum", force) for force in CONTRACT["minimum_force_grid_n"]]
        forces = [row["post_aligned_force_n"] for row in branch]
        differences = np.diff(forces)
        records[str(step)] = {"commands_n": CONTRACT["minimum_force_grid_n"], "measured_n": forces, "adjacent_differences_n": differences.tolist()}
        all_nondecreasing &= bool(np.all(differences >= -1e-9))
        all_strict_enough &= int(np.sum(differences > 1e-6)) >= 2
        all_safe &= all(row["finite_torque"] and row["total_clip_count"] == 0 for row in branch)
        all_never_reduce &= all(row["never_reduced_stock_axis"] for row in branch)
    checks = {
        "exact_disabled_parity": all(all(value.values()) for value in parity.values()),
        "response_nondecreasing": all_nondecreasing,
        "at_least_two_strict_adjacent_responses": all_strict_enough,
        "finite_zero_clip": all_safe,
        "stock_axis_never_reduced": all_never_reduce,
    }
    result = {"gate": CONTRACT["gate"], "pass": all(checks.values()), "checks": checks, "parity": parity, "responses": records}
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
