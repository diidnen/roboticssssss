#!/usr/bin/env python3
"""Validate the explicit coordinator V2 final gate after the E5 race."""

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
    checks = {
        "schema": gate.get("schema") == "E3_COORDINATOR_FINAL_GATE_V2",
        "status": gate.get("status") == "FINAL_GATE_PASS",
        "authorized": gate.get("authorized") is True,
        "operation": gate.get("operation") == args.operation,
        "external_e5": gate.get("external_e5_pid") == 2059639 and gate.get("external_e5_exit") == "NATURAL",
        "no_active_e5": gate.get("active_e5_pids") == [],
        "silence": gate.get("e5_silence_seconds", 0) >= 10,
        "final_recheck": gate.get("final_recheck_pass") is True,
        "protected_audit": gate.get("protected_processes_audited") is True,
        "duplicate_scan": gate.get("duplicate_scan_empty") is True,
        "coordinator": gate.get("coordinator_final_status") == "FINAL_GATE_PASS",
        "rejected_core_partial": (
            gate.get("core_e5_pid") == 2059513
            and gate.get("core_e5_exit_code") == -9
            and gate.get("core_e5_partial_rejected") is True
            and gate.get("locked_test_coverage") == "17/60"
        ),
    }
    if args.operation == "E3_TASK5_NOMINAL_DEV_CELL":
        checks["root_seed"] = gate.get("root_seed") == args.root_seed
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"V2 coordinator final gate rejected: {failed}")
    issued = datetime.fromisoformat(str(gate["issued_at_utc"]).replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - issued).total_seconds()
    if age < 0 or age > 60:
        raise RuntimeError(f"V2 final gate not fresh: age={age:.1f}s")
    print("E3_COORDINATOR_FINAL_GATE_V2_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
