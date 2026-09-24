#!/usr/bin/env python3
"""Promote the unchanged terminal candidate only after the preregistered DEV gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from validate_and_lock_task5_onboarded_candidate import tree_manifest, validate_norm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_lock", type=Path)
    ap.add_argument("nominal_dev_gate", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite final freeze: {args.output}")
    lock = json.loads(args.candidate_lock.read_text())
    gate = json.loads(args.nominal_dev_gate.read_text())
    lock_sha = sha256(args.candidate_lock)
    if lock.get("status") != "CANDIDATE_LOCKED_FOR_NOMINAL_DEV_NOT_FINAL_FREEZE":
        raise RuntimeError("candidate is not in pre-DEV locked state")
    if gate.get("gate_pass") is not True or gate.get("status") != "NOMINAL_DEV_PASS":
        raise RuntimeError("nominal DEV gate did not pass")
    if gate.get("candidate_lock_sha256") != lock_sha:
        raise RuntimeError("nominal DEV gate used another candidate lock")
    checkpoint = Path(lock["checkpoint_dir"])
    digest, files = tree_manifest(checkpoint)
    if digest != lock["checkpoint_tree_sha256"]:
        raise RuntimeError("checkpoint changed after candidate lock")
    norm = Path(lock["norm_stats"])
    if validate_norm(norm) != lock["norm_stats_sha256"]:
        raise RuntimeError("normalization asset changed after candidate lock")
    result = dict(lock)
    result.update({
        "status": "TASK5_ONBOARDED_PI0_FROZEN_AFTER_NOMINAL_DEV_PASS",
        "checkpoint_files": files,
        "candidate_lock_sha256": lock_sha,
        "nominal_dev_gate": str(args.nominal_dev_gate.resolve()),
        "nominal_dev_gate_sha256": sha256(args.nominal_dev_gate),
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fairness_rule": "this exact checkpoint/config/norm fingerprint must be shared by every later force method",
        "checkpoint_selected_using_activeforcing_utility_or_test": False,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checkpoint_files"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
