#!/usr/bin/env python3
"""Wait for collection, then audit, train, and run held-out inference."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "experiments/af_taskforms_216_v1"
)
RECORDS = ROOT / "branches.jsonl"
MODELS = ROOT / "models"


def record_count() -> int:
    if not RECORDS.exists():
        return 0
    return sum(1 for line in RECORDS.open() if line.strip())


def run(command: list[str]) -> None:
    print("PIPELINE_RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    last = None
    while True:
        count = record_count()
        if count != last:
            print(f"PIPELINE_WAIT collection={count}/216", flush=True)
            last = count
        if count >= 216:
            break
        time.sleep(30)
    audit = ROOT / "collection_audit.json"
    run(
        [
            sys.executable,
            str(repo / "scripts/audit_af_taskforms_collection.py"),
            "--records",
            str(RECORDS),
            "--out",
            str(audit),
        ]
    )
    if not json.loads(audit.read_text())["passed"]:
        raise SystemExit("collection audit did not pass")
    training_report = MODELS / "training_report.json"
    run(
        [
            sys.executable,
            str(repo / "scripts/train_af_taskforms_models.py"),
            "--records",
            str(RECORDS),
            "--out",
            str(MODELS),
            "--device",
            "cpu",
        ]
    )
    run(
        [
            sys.executable,
            str(repo / "scripts/evaluate_af_taskforms_inference.py"),
            "--records",
            str(RECORDS),
            "--training-report",
            str(training_report),
            "--out",
            str(ROOT / "heldout_inference_report.json"),
        ]
    )
    print("PIPELINE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
