import json
import os
import tempfile
from pathlib import Path

import numpy as np

OUT = Path("/media/volume/data/exouser/activeforcing_knob_damping_v2_20260911")
CONTRACT = json.loads((OUT / "R2_CONTRACT.json").read_text())


def load(kind, command):
    return json.loads((OUT / "r2" / f"{kind}_{command:+.0f}Nm.json").read_text())


def main():
    output = OUT / "R2_RESULT.json"
    if output.exists():
        raise RuntimeError(f"immutable result exists: {output}")
    untouched, disabled = load("untouched", 0), load("disabled", 0)
    branches = [load("minimum", value) for value in CONTRACT["minimum_yaw_torque_grid_nm"]]
    velocities = [row["knob_post_qvel"] for row in branches]
    differences = np.diff(velocities)
    checks = {
        "untouched_disabled_exact_parity": untouched["post_state_sha256"] == disabled["post_state_sha256"] and untouched["post_qpos"] == disabled["post_qpos"] and untouched["post_qvel"] == disabled["post_qvel"] and untouched["final_ctrl"] == disabled["final_ctrl"],
        "knob_qvel_nondecreasing": bool(np.all(differences >= -1e-10)),
        "two_strict_adjacent_responses": int(np.sum(differences > 1e-7)) >= 2,
        "finite_zero_clip": all(row["finite_torque"] and row["total_clip_count"] == 0 for row in branches),
        "stock_yaw_never_reduced": all(row["stock_yaw_never_reduced"] for row in branches),
    }
    result = {"gate": CONTRACT["gate"], "pass": all(checks.values()), "checks": checks, "commands_nm": CONTRACT["minimum_yaw_torque_grid_nm"], "knob_post_qvel": velocities, "adjacent_differences": differences.tolist()}
    fd, tmp = tempfile.mkstemp(dir=OUT, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(result, indent=2) + "\n")
    os.replace(tmp, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
