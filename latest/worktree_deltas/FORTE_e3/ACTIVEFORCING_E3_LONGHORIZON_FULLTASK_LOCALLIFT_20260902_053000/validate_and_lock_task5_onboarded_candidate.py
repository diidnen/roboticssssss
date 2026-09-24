#!/usr/bin/env python3
"""Fail-closed CPU lock for the preselected terminal 5-demo checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CONFIG = "pi0_lora_tacfield_e3_task5_5demo_7dpf"
EXP = "task5_5demo_7dpf_20260902_120000"
OPENPI_COMMIT = "31049447d685cb36ddaeddda4f1d62fec0bc6392"
CONFIG_SHA256 = "296b01a51c985583bab688402808296315bd449ddd2aa343f4f5458ba1d1295e"
POLICY_SHA256 = "f5eb0161b831c1f4a65b763f24183f5333a3efe54ae0537088a9450fdd54781a"
EXPECTED_STEP = 999
EXPECTED_NORM_DIMS = {"state": 7, "actions": 13, "tactile_prefix": 396}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> tuple[str, list[dict[str, object]]]:
    records = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256(path)
        records.append({"path": rel, "bytes": size, "sha256": digest})
        aggregate.update(rel.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
    if not records:
        raise RuntimeError(f"empty checkpoint tree: {root}")
    return aggregate.hexdigest(), records


def validate_norm(path: Path) -> str:
    outer = json.loads(path.read_text())
    if set(outer) != {"norm_stats"}:
        raise RuntimeError("normalization envelope mismatch")
    stats = outer["norm_stats"]
    if set(stats) != set(EXPECTED_NORM_DIMS):
        raise RuntimeError(f"normalization keys mismatch: {sorted(stats)}")
    for name, dim in EXPECTED_NORM_DIMS.items():
        if set(stats[name]) != {"mean", "std", "q01", "q99"}:
            raise RuntimeError(f"incomplete normalization statistics: {name}")
        if any(len(stats[name][field]) != dim for field in ("mean", "std", "q01", "q99")):
            raise RuntimeError(f"normalization dimension mismatch: {name}")
    return sha256(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpi", type=Path, required=True)
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--norm-stats", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite candidate lock: {args.output}")
    git_head = __import__("subprocess").check_output(
        ["git", "-C", str(args.openpi), "rev-parse", "HEAD"], text=True
    ).strip()
    if git_head != OPENPI_COMMIT:
        raise RuntimeError(f"OpenPI commit mismatch: {git_head}")
    if sha256(args.openpi / "src/openpi/training/config.py") != CONFIG_SHA256:
        raise RuntimeError("onboarding config hash mismatch")
    if sha256(args.openpi / "src/openpi/policies/libero_policy.py") != POLICY_SHA256:
        raise RuntimeError("policy transform hash mismatch")

    numeric_steps = sorted(int(path.name) for path in args.experiment_root.iterdir() if path.is_dir() and path.name.isdigit())
    if EXPECTED_STEP not in numeric_steps:
        raise RuntimeError(f"preselected terminal step {EXPECTED_STEP} missing; observed {numeric_steps}")
    selected = args.experiment_root / str(EXPECTED_STEP)
    params = selected / "params"
    required = [selected / "_CHECKPOINT_METADATA", params / "_METADATA", params / "manifest.ocdbt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"incomplete terminal checkpoint: {missing}")
    norm_sha = validate_norm(args.norm_stats)
    checkpoint_digest, files = tree_manifest(selected)

    result = {
        "status": "CANDIDATE_LOCKED_FOR_NOMINAL_DEV_NOT_FINAL_FREEZE",
        "task": "libero_10/task5",
        "split": "TRAINED_ON_TRAIN_ONLY",
        "config": CONFIG,
        "experiment": EXP,
        "selected_step": EXPECTED_STEP,
        "selection_rule": "terminal step 999 preselected before nominal DEV; no checkpoint comparison",
        "checkpoint_dir": str(selected.resolve()),
        "checkpoint_tree_sha256": checkpoint_digest,
        "checkpoint_files": files,
        "norm_stats": str(args.norm_stats.resolve()),
        "norm_stats_sha256": norm_sha,
        "openpi_commit": OPENPI_COMMIT,
        "config_sha256": CONFIG_SHA256,
        "policy_transform_sha256": POLICY_SHA256,
        "nominal_control_dimensions_supervised": 7,
        "force_output_supervision_used": False,
        "test_used": False,
        "activeforcing_or_utility_outcomes_used": False,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checkpoint_files"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
