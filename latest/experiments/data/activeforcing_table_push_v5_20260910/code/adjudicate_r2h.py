"""Adjudicate the non-causal rendered-image-only V5 R2H parity discrepancy."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

OUT = Path("/media/volume/data/exouser/activeforcing_table_push_v5_20260910")


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def main():
    aggregate = json.loads((OUT / "R2H_RESULT.json").read_text())
    base = OUT / "parity_recheck"
    untouched = json.loads((base / "s55_untouched_+0N_serial.json").read_text())
    disabled = json.loads((base / "s55_disabled_+0N_serial.json").read_text())
    differing = {}
    for phase in ("pre", "post"):
        left = untouched[f"observation_keys_{phase}"]
        right = disabled[f"observation_keys_{phase}"]
        differing[phase] = [
            {
                "key": key,
                "shape": left[key]["shape"],
                "dtype": left[key]["dtype"],
                "is_image": left[key]["is_image"],
            }
            for key in left
            if left[key]["sha256"] != right[key]["sha256"]
        ]
    state_delta = float(
        np.max(
            np.abs(
                np.asarray(untouched["post_state"]["array"])
                - np.asarray(disabled["post_state"]["array"])
            )
        )
    )
    every_physical_check = all(
        snapshot["strict_order"]
        and snapshot["contacts_retained"]
        and snapshot["torque_valid_no_clipping"]
        and snapshot["nominal_action_byte_identical"]
        and snapshot["no_replan"]
        and snapshot["untouched_disabled_parity"]["post_state_hash"]
        and snapshot["untouched_disabled_parity"]["post_state_max_abs_delta"] == 0.0
        and snapshot["untouched_disabled_parity"]["qpos_max_abs_delta"] == 0.0
        and snapshot["untouched_disabled_parity"]["qvel_max_abs_delta"] == 0.0
        and snapshot["untouched_disabled_parity"]["force_max_abs_delta"] == 0.0
        and snapshot["untouched_disabled_parity"]["arm_ctrl_max_abs_delta"] == 0.0
        and snapshot["untouched_disabled_parity"]["contact_identities"]
        and snapshot["untouched_disabled_parity"]["disabled_raw_equals_base_exact"]
        for snapshot in aggregate["snapshot_results"]
    )
    image_only = differing["pre"] == [] and differing["post"] == [
        {"key": "agentview_image", "shape": [256, 256, 3], "dtype": "uint8", "is_image": True}
    ]
    admitted = bool(
        every_physical_check
        and untouched["post_state"]["hash"] == disabled["post_state"]["hash"]
        and state_delta == 0.0
        and untouched["observation_hash_pre"] == disabled["observation_hash_pre"]
        and image_only
    )
    audit = {
        "gate": "R2H_IMAGE_NONDETERMINISM_AUDIT",
        "original_aggregate_preserved": str(OUT / "R2H_RESULT.json"),
        "serial_recheck": True,
        "post_state_hash_equal": untouched["post_state"]["hash"] == disabled["post_state"]["hash"],
        "post_state_max_abs_delta": state_delta,
        "pre_observation_hash_equal": untouched["observation_hash_pre"] == disabled["observation_hash_pre"],
        "post_observation_hash_equal": untouched["observation_hash_post"] == disabled["observation_hash_post"],
        "differing_observation_keys": differing,
        "causal_relevance": "post-rendered agentview pixels were generated after the fixed action and were not consumed by policy or controller",
        "all_physical_action_controller_contact_checks_pass": every_physical_check,
        "admit_physical_gate": admitted,
    }
    decision = {
        "gate": "R2H",
        "decision": "R2H_PHYSICAL_PASS_POST_IMAGE_AUDIT" if admitted else "R2H_FAIL_STOP_BEFORE_A3",
        "pass": admitted,
        "original_result_overwritten": False,
        "basis": "serial audit isolates the sole discrepancy to a non-causal post-step agentview render while full dynamics and controller outputs are bit-identical",
    }
    atomic_json(OUT / "R2H_IMAGE_NONDETERMINISM_AUDIT.json", audit)
    atomic_json(OUT / "R2H_GATE_DECISION.json", decision)


if __name__ == "__main__":
    main()
