"""Immutable adjudication for the preregistered V5 paired online rollout."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

OUT = Path("/media/volume/data/exouser/activeforcing_table_push_v5_20260910")
CONTRACT = json.loads((OUT / "A3_ONLINE_CONTRACT.json").read_text())


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def rows(mode):
    return [json.loads(line) for line in (OUT / "online" / mode / "telemetry.jsonl").read_text().splitlines()]


def main():
    result_path = OUT / "A3_ONLINE_RESULT.json"
    if result_path.exists():
        raise RuntimeError(f"immutable result exists: {result_path}")
    baseline_receipt = json.loads((OUT / "online/untouched/receipt.json").read_text())
    hybrid_receipt = json.loads((OUT / "online/hybrid/receipt.json").read_text())
    baseline, hybrid = rows("untouched"), rows("hybrid")
    first = hybrid_receipt["first_intervention_step"]
    prefix_steps = list(range(first + 1)) if first is not None else []
    pairing = []
    for step in prefix_steps:
        b, h = baseline[step], hybrid[step]
        qpos_diff = float(np.max(np.abs(np.asarray(b["pre_qpos"]) - np.asarray(h["pre_qpos"]))))
        qvel_diff = float(np.max(np.abs(np.asarray(b["pre_qvel"]) - np.asarray(h["pre_qvel"]))))
        action_diff = float(np.max(np.abs(np.asarray(b["nominal_action"]) - np.asarray(h["nominal_action"]))))
        image_same = b["pre_observation_hashes"].get("agentview_image") == h["pre_observation_hashes"].get("agentview_image")
        wrist_same = b["pre_observation_hashes"].get("robot0_eye_in_hand_image") == h["pre_observation_hashes"].get("robot0_eye_in_hand_image")
        pairing.append(
            {
                "step": step,
                "qpos_max_abs_diff": qpos_diff,
                "qvel_max_abs_diff": qvel_diff,
                "nominal_action_max_abs_diff": action_diff,
                "agentview_hash_equal": image_same,
                "wrist_hash_equal": wrist_same,
                "pre_state_hash_equal": b["pre_state_sha256"] == h["pre_state_sha256"],
            }
        )
    pair_ok = bool(pairing) and all(
        row["qpos_max_abs_diff"] <= CONTRACT["pairing_gate"]["qpos_qvel_max_abs_tolerance"]
        and row["qvel_max_abs_diff"] <= CONTRACT["pairing_gate"]["qpos_qvel_max_abs_tolerance"]
        and row["nominal_action_max_abs_diff"] <= CONTRACT["pairing_gate"]["nominal_action_max_abs_tolerance"]
        for row in pairing
    )
    gate = CONTRACT["pass_gate"]
    checks = {
        "both_runs_error_free": baseline_receipt["error"] is None and hybrid_receipt["error"] is None,
        "paired_pre_intervention_prefix": pair_ok,
        "hybrid_native_success": bool(hybrid_receipt["native_success"]),
        "minimum_intervention_steps": hybrid_receipt["intervention_steps"] >= gate["minimum_hybrid_intervention_steps"],
        "finite_torque": bool(hybrid_receipt["all_torques_finite"]),
        "zero_actuator_clips": hybrid_receipt["total_actuator_clips"] == 0,
    }
    result = {
        "gate": CONTRACT["gate"],
        "contract_version": CONTRACT["version"],
        "pass": all(checks.values()),
        "checks": checks,
        "baseline_native_success": baseline_receipt["native_success"],
        "hybrid_native_success": hybrid_receipt["native_success"],
        "outcome_class": (
            "force_control_recovery" if hybrid_receipt["native_success"] and not baseline_receipt["native_success"]
            else "no_regression_both_success" if hybrid_receipt["native_success"] and baseline_receipt["native_success"]
            else "hybrid_failed"
        ),
        "first_intervention_step": first,
        "hybrid_intervention_steps": hybrid_receipt["intervention_steps"],
        "pairing_rows": pairing,
        "image_hash_mismatches": sum(not (row["agentview_hash_equal"] and row["wrist_hash_equal"]) for row in pairing),
        "note": "Image hashes are diagnostic only under the preregistered contract; physical state and nominal actions control pairing.",
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
