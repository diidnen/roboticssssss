#!/usr/bin/env python3
"""Serve one prelocked onboarding candidate with an auditable metadata handshake."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import socket
from pathlib import Path

import torch

from openpi.policies import policy_config as policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config as training_config

from validate_and_lock_task5_onboarded_candidate import tree_manifest, validate_norm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LockedPolicy:
    def __init__(self, policy, metadata: dict[str, object]):
        self._policy = policy
        self._metadata = dict(policy.metadata)
        self._metadata.update(metadata)

    @property
    def metadata(self):
        return self._metadata

    def infer(self, observation):
        return self._policy.infer(observation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--candidate-lock", type=Path, required=True)
    args = ap.parse_args()
    lock = json.loads(args.candidate_lock.read_text())
    if lock.get("status") != "CANDIDATE_LOCKED_FOR_NOMINAL_DEV_NOT_FINAL_FREEZE":
        raise RuntimeError("candidate lock status mismatch")
    if lock.get("config") != "pi0_lora_tacfield_e3_task5_5demo_7dpf" or lock.get("selected_step") != 999:
        raise RuntimeError("candidate config/step mismatch")
    lock_sha = sha256(args.candidate_lock)
    checkpoint = Path(lock["checkpoint_dir"])
    norm_dir = Path(lock["norm_stats"]).parent
    if not checkpoint.is_dir() or not (norm_dir / "norm_stats.json").is_file():
        raise RuntimeError("locked checkpoint or normalization asset missing")
    checkpoint_digest, _ = tree_manifest(checkpoint)
    if checkpoint_digest != lock.get("checkpoint_tree_sha256"):
        raise RuntimeError("checkpoint changed after candidate lock")
    if validate_norm(norm_dir / "norm_stats.json") != lock.get("norm_stats_sha256"):
        raise RuntimeError("normalization asset changed after candidate lock")
    torch.manual_seed(20260902)
    torch.cuda.manual_seed_all(20260902)
    config = training_config.get_config(lock["config"])
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        norm_stats=normalize.load(norm_dir),
        pytorch_device="cuda:0",
    )
    metadata = {
        "e3_locked_candidate": True,
        "e3_task": "libero_10/task5",
        "e3_policy_config": lock["config"],
        "e3_checkpoint_step": 999,
        "e3_checkpoint_tree_sha256": lock["checkpoint_tree_sha256"],
        "e3_candidate_lock_sha256": lock_sha,
        "e3_openpi_commit": lock["openpi_commit"],
    }
    wrapped = LockedPolicy(policy, metadata)
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Starting locked task5 onboarding candidate on %s:%s", socket.gethostname(), args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host="0.0.0.0",
        port=args.port,
        metadata=dict(wrapped.metadata),
    ).serve_forever()


if __name__ == "__main__":
    main()
