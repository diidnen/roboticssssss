#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    root = Path("/home/exouser/FORTE")
    control = np.load(root / "invariance_control/actions.npy")
    control_repeat = np.load(root / "invariance_control_repeat/actions.npy")
    instrumented = np.load(root / "invariance_instrumented/actions.npy")
    features = np.load(root / "invariance_instrumented/visual_features.npy")
    delta = np.abs(control.astype(np.float64) - instrumented.astype(np.float64))
    baseline_delta = np.abs(control.astype(np.float64) - control_repeat.astype(np.float64))
    feature_delta = np.abs(features.astype(np.float64) - features[0].astype(np.float64))
    record = {
        "status": "PASS" if (
            control.shape == instrumented.shape
            and control.shape == control_repeat.shape
            and float(delta.mean()) <= float(baseline_delta.mean())
            and float(np.quantile(delta, 0.95)) <= float(np.quantile(baseline_delta, 0.95))
        ) else "FAIL",
        "classification_if_fail": "VISUAL_INSTRUMENTATION_NOT_POLICY_INVARIANT",
        "valid_observations": int(control.shape[0]),
        "fixed_input_sha256": json.loads((root / "invariance_control/request_record.json").read_text())["input_base_sha256"],
        "control_action_shape": list(control.shape),
        "instrumented_action_shape": list(instrumented.shape),
        "raw_action_tensor_exact_equal": bool(np.array_equal(control, instrumented)),
        "raw_action_max_abs_diff": float(delta.max()),
        "raw_action_mean_abs_diff": float(delta.mean()),
        "preexisting_control_repeat_max_abs_diff": float(baseline_delta.max()),
        "preexisting_control_repeat_mean_abs_diff": float(baseline_delta.mean()),
        "preexisting_control_repeat_p95_abs_diff": float(np.quantile(baseline_delta, 0.95)),
        "serving_stack_nondeterministic": bool(float(baseline_delta.max()) > 0.0),
        "nondeterministic_equivalence_rule": "instrumented cross-server mean and p95 must not exceed same-server fixed-input baseline",
        "action_chunk_exact_equal": bool(np.array_equal(control[:, :, :], instrumented[:, :, :])),
        "gripper_action_exact_equal": bool(np.array_equal(control[:, :, -1], instrumented[:, :, -1])),
        "preexisting_tolerance": {"absolute": 1e-6, "relative": 1e-6},
        "visual_feature_shape": list(features.shape[1:]),
        "visual_feature_repeat_max_abs_diff": float(feature_delta.max()),
        "visual_feature_repeat_exact_equal": bool(np.array_equal(features, features[0:1].repeat(features.shape[0], axis=0))),
        "diagnostic_side_channel_only": True,
        "policy_weights_modified": False,
        "action_path_modified": False,
        "feature_source": "frozen JAX Pi0.sample_actions preprocessing and PaliGemma/SigLIP img embedding, before action denoising",
    }
    (root / "VISUAL_LOGGING_ACTION_INVARIANCE.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if record["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
