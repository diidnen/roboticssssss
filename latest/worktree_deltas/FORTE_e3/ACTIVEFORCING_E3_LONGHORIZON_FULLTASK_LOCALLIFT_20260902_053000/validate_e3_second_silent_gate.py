#!/usr/bin/env python3
"""Validate a short-lived dual-coordinator authorization for one E3 GPU action."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", type=Path)
    ap.add_argument("operation", choices=("E3_TASK5_CANDIDATE_SERVER_START", "E3_TASK5_NOMINAL_DEV_CELL"))
    ap.add_argument("--root-seed", type=int)
    args = ap.parse_args()
    gate = json.loads(args.gate.read_text())
    if gate.get("authorized") is not True:
        raise RuntimeError("second-silent gate is not authorized")
    if gate.get("root_go") is not True or gate.get("core_coordinator_go") is not True:
        raise RuntimeError("both root and core coordinator GO are required")
    if gate.get("operation") != args.operation:
        raise RuntimeError("authorization operation mismatch")
    if gate.get("e5_silence_seconds", 0) < 120:
        raise RuntimeError("second E5-silent observation window is shorter than 120 seconds")
    if gate.get("active_e5_pids") != []:
        raise RuntimeError("gate records active E5 processes")
    if gate.get("protected_processes_audited") is not True or gate.get("duplicate_scan_empty") is not True:
        raise RuntimeError("protected-process or duplicate audit missing")
    if args.operation == "E3_TASK5_NOMINAL_DEV_CELL" and gate.get("root_seed") != args.root_seed:
        raise RuntimeError("nominal DEV root does not match authorization")
    issued = datetime.fromisoformat(str(gate["issued_at_utc"]).replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - issued).total_seconds()
    if age < 0 or age > 120:
        raise RuntimeError(f"authorization is not fresh: age={age:.1f}s")
    print("E3_SECOND_SILENT_GATE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
