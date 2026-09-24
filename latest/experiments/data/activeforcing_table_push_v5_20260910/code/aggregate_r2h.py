"""Aggregate immutable fresh-process V5 R2H branch receipts."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

OUT = Path("/media/volume/data/exouser/activeforcing_table_push_v5_20260910")
STEPS = [55, 57]
GRID = [10.0, 15.0, 20.0, 25.0, 30.0]


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def max_delta(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def main():
    result_path = OUT / "R2H_RESULT.json"
    if result_path.exists():
        raise RuntimeError("immutable R2H result already exists")
    rows = [json.loads(path.read_text()) for path in sorted((OUT / "branches").glob("*.json"))]
    expected = len(STEPS) * (2 + len(GRID))
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} branch receipts, got {len(rows)}")

    snapshots = []
    for step in STEPS:
        subset = [row for row in rows if row["step"] == step]
        untouched = next(row for row in subset if row["kind"] == "untouched")
        disabled = next(row for row in subset if row["kind"] == "disabled")
        hybrids = [
            next(
                row
                for row in subset
                if row["kind"] == "hybrid" and row["axis_command_n"] == command
            )
            for command in GRID
        ]
        parity = {
            "post_state_hash": untouched["post_state"]["hash"] == disabled["post_state"]["hash"],
            "post_state_max_abs_delta": max_delta(untouched["post_state"]["array"], disabled["post_state"]["array"]),
            "qpos_max_abs_delta": max_delta(untouched["post_state"]["qpos"], disabled["post_state"]["qpos"]),
            "qvel_max_abs_delta": max_delta(untouched["post_state"]["qvel"], disabled["post_state"]["qvel"]),
            "force_max_abs_delta": max_delta(untouched["post_force_world_on_plate"], disabled["post_force_world_on_plate"]),
            "arm_ctrl_max_abs_delta": max_delta(untouched["final_arm_ctrl"], disabled["final_arm_ctrl"]),
            "contact_identities": untouched["post_contacts"] == disabled["post_contacts"],
            "nominal_action_hash": untouched["nominal_action_sha256"] == disabled["nominal_action_sha256"],
            "direction_hash": untouched["direction_hash"] == disabled["direction_hash"],
            "pre_observation_hash": untouched["observation_hash_pre"] == disabled["observation_hash_pre"],
            "post_observation_hash": untouched["observation_hash_post"] == disabled["observation_hash_post"],
            "disabled_raw_equals_base_exact": all(
                item.get("disabled_raw_equals_base_exact") is True
                for item in disabled["adapter_trace"]
            ),
        }
        parity_pass = (
            parity["post_state_hash"]
            and parity["post_state_max_abs_delta"] == 0.0
            and parity["qpos_max_abs_delta"] == 0.0
            and parity["qvel_max_abs_delta"] == 0.0
            and parity["force_max_abs_delta"] == 0.0
            and parity["arm_ctrl_max_abs_delta"] == 0.0
            and all(
                parity[key]
                for key in (
                    "contact_identities",
                    "nominal_action_hash",
                    "direction_hash",
                    "pre_observation_hash",
                    "post_observation_hash",
                    "disabled_raw_equals_base_exact",
                )
            )
        )
        measured = [row["post_aligned_force_n"] for row in hybrids]
        strict_order = bool(np.all(np.diff(measured) > 0))
        contacts_retained = all(bool(row["post_contacts"]) for row in hybrids)
        torque_valid = all(
            item["finite"] and item["shape"] == [7] and item["clip_count"] == 0
            for row in hybrids
            for item in row["adapter_trace"]
        )
        nominal_identical = len({row["nominal_action_sha256"] for row in hybrids + [untouched, disabled]}) == 1
        no_replan = all(row["no_replan"] for row in hybrids + [untouched, disabled])
        snapshot_pass = parity_pass and strict_order and contacts_retained and torque_valid and nominal_identical and no_replan
        snapshots.append(
            {
                "step": step,
                "untouched_disabled_parity": parity,
                "axis_command_grid_n": GRID,
                "measured_aligned_force_n": measured,
                "strict_order": strict_order,
                "contacts_retained": contacts_retained,
                "torque_valid_no_clipping": torque_valid,
                "nominal_action_byte_identical": nominal_identical,
                "no_replan": no_replan,
                "pass": snapshot_pass,
            }
        )

    passed = all(snapshot["pass"] for snapshot in snapshots)
    result = {
        "gate": "R2H",
        "fresh_process": True,
        "source_osc_sha256": "cfa62a0e719bc53ef0c701efa66f7b3e2272d4fca2150ec73c05bb145eba85cb",
        "snapshot_results": snapshots,
        "pass": passed,
        "decision": "R2H_PASS" if passed else "R2H_FAIL_STOP_BEFORE_A3",
        "new_task_outcomes": 0,
    }
    atomic_json(result_path, result)


if __name__ == "__main__":
    main()
