"""V5-owned deterministic wrapper around the frozen fine-tuned LIBERO pi0 policy."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

OUT = Path("/media/volume/data/exouser/activeforcing_table_push_v5_20260910")
CHECKPOINT = Path("/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero")
TREE_SHA256 = "92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103"
PORT = 8015


def main():
    import jax
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared import normalize
    from openpi.training import config

    policy = policy_config.create_trained_policy(
        config.get_config("pi0_libero"),
        CHECKPOINT,
        norm_stats=normalize.load(CHECKPOINT / "assets/physical-intelligence/libero"),
    )

    class SeededPolicy:
        def infer(self, observation):
            seed = observation.pop("_activeforcing_episode_start_seed", None)
            if seed is not None:
                policy._rng = jax.random.key(int(seed))
            answer = policy.infer(observation)
            actions = np.asarray(answer["actions"])
            record = {
                "seed": seed,
                "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                "shape": list(actions.shape),
            }
            with (OUT / "server" / "INFERENCE.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            return answer

    receipt = {
        "pid": os.getpid(),
        "port": PORT,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_tree_sha256": TREE_SHA256,
        "policy_config": "pi0_libero",
        "v5_owned": True,
        "jax_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    }
    (OUT / "server" / "SERVER_READY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    WebsocketPolicyServer(SeededPolicy(), host="127.0.0.1", port=PORT, metadata=receipt).serve_forever()


if __name__ == "__main__":
    main()
