#!/usr/bin/env python3
"""Send the same fixed observations to a diagnostic π0 websocket server."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi_client.websocket_client_policy import WebsocketClientPolicy


def sha(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--base", type=Path, default=Path("/tmp/visual_camera_probe/agentview.npy"))
    ap.add_argument("--wrist", type=Path, default=Path("/tmp/visual_camera_probe/wrist.npy"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = np.load(args.base).astype(np.uint8, copy=False)
    wrist = np.load(args.wrist).astype(np.uint8, copy=False)
    # All requests are byte-identical. Repeated calls exercise the serving path
    # without introducing observation or RNG differences.
    obs = {
        "image": base,
        "wrist_image": wrist,
        "state": np.zeros(7, dtype=np.float32),
        "tactile_marker_motion": np.zeros((9, 198, 2), dtype=np.float32),
        "prompt": "pick up the alphabet soup and place it in the basket",
    }
    client = WebsocketClientPolicy("127.0.0.1", args.port)
    actions = []
    features = []
    response_keys = None
    for _ in range(args.count):
        response = client.infer(obs)
        response_keys = sorted(response.keys())
        action = np.asarray(response["actions"])
        actions.append(action.copy())
        if "diagnostics_visual_feature" in response:
            features.append(np.asarray(response["diagnostics_visual_feature"]).copy())
    np.save(args.out / "actions.npy", np.asarray(actions))
    if features:
        np.save(args.out / "visual_features.npy", np.asarray(features))
    record = {
        "count": args.count,
        "input_base_shape": list(base.shape),
        "input_wrist_shape": list(wrist.shape),
        "input_base_sha256": sha(base),
        "input_wrist_sha256": sha(wrist),
        "response_keys": response_keys,
        "action_shape": list(np.asarray(actions[0]).shape),
        "action_dtype": str(np.asarray(actions[0]).dtype),
        "feature_shape": list(np.asarray(features[0]).shape) if features else None,
        "feature_dtype": str(np.asarray(features[0]).dtype) if features else None,
    }
    (args.out / "request_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
