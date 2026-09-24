#!/usr/bin/env python3
"""B5 serving wrapper for the official Tabero-VTLA policy.

This keeps the checkpoint directory read-only. The downloaded checkpoint has
norm stats under assets/NathanWu7/tabero, while the current config asks for
assets/NathanWu7/tabero_object_25. Instead of creating a symlink inside the
checkpoint, B5 passes the existing norm stats explicitly.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import socket

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18019)
    parser.add_argument("--policy-config", default="pi0_lora_tacfield_tabero")
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--norm-stats-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    policy_dir = Path(args.policy_dir)
    norm_stats_dir = Path(args.norm_stats_dir)
    norm_stats = _normalize.load(norm_stats_dir)
    train_config = _config.get_config(args.policy_config)
    policy = _policy_config.create_trained_policy(train_config, policy_dir, norm_stats=norm_stats)

    metadata = dict(policy.metadata)
    metadata.update(
        {
            "b5_explicit_norm_stats_dir": str(norm_stats_dir),
            "b5_checkpoint_read_only": True,
            "b5_policy_config": args.policy_config,
        }
    )
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating B5 server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

