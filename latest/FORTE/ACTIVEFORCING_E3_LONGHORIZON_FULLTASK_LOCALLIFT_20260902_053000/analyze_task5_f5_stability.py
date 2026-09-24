#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root", type=Path)
    args = ap.parse_args()
    files = sorted((args.out_root / "task5" / "logs").glob("*_episodes.csv"))
    errors = []
    rows = []
    if len(files) != 1:
        errors.append(f"expected one episode CSV, got {len(files)}")
    else:
        with files[0].open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    if len(rows) != 5:
        errors.append(f"expected 5 episodes, got {len(rows)}")
    hashes = sorted({row.get("root_state_hash", "") for row in rows})
    if len(hashes) != 1 or len(hashes[0] if hashes else "") != 64:
        errors.append(f"root hash mismatch or missing: {hashes}")
    methods = sorted({row.get("method", "") for row in rows})
    if methods != ["E3_TRAIN_QUALIFICATION_F5_STABILITY"]:
        errors.append(f"unexpected methods: {methods}")
    successes = sum(int(float(row.get("official_success", 0) or 0)) for row in rows)
    requested = {float(row.get("mean_predicted_force_slot_N", "nan")) for row in rows}
    if requested and requested != {5.0}:
        errors.append(f"force setpoint mismatch: {sorted(requested)}")
    audit = {
        "status": "PASS" if not errors and successes >= 2 else "FAIL",
        "gate": "E3_TASK5_INDOMAIN_F5_NOMINAL_STABILITY",
        "episodes": len(rows),
        "official_full_task_successes": successes,
        "required_successes": 2,
        "official_success_rate": successes / len(rows) if rows else 0.0,
        "root_state_hashes": hashes,
        "force_setpoints_N": sorted(requested),
        "errors": errors,
        "next_step": "freeze task and collect systematic TRAIN/DEV regimes" if not errors and successes >= 2 else "activate 5-demo benchmark onboarding without changing Utility/Fmax",
    }
    path = args.out_root / "E3_TASK5_F5_STABILITY_GATE.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
