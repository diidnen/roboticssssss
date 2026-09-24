#!/usr/bin/env python3
"""Merge independently completed fresh-root runs into the final E2E bundle."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).parent
TARGET = ROOT / "fresh_e2e"
SOURCES = [
    (ROOT / "fresh_e2e_root8400_LOW_v2", 5),
    (ROOT / "fresh_e2e_root8400_mid_v3", 5),
    (ROOT / "fresh_e2e_root8400_high_v2", 5),
    (ROOT / "fresh_e2e_root8401", 15),
]
SKIP = {"MASS_FRESH_E2E_PROTOCOL.json", "MASS_FRESH_E2E_ROWS.csv", "MASS_FRESH_E2E_DECISIONS.csv", "run.log"}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=fields)
        out.writeheader()
        out.writerows(rows)


def main():
    all_rows = []
    all_decisions = []
    protocols = []
    for source, expected_rows in SOURCES:
        protocol = json.loads((source / "MASS_FRESH_E2E_PROTOCOL.json").read_text(encoding="utf-8"))
        if protocol.get("status") != "COMPLETED":
            raise RuntimeError(f"source not completed: {source}: {protocol.get('status')}")
        rows = read_csv(source / "MASS_FRESH_E2E_ROWS.csv")
        decisions = read_csv(source / "MASS_FRESH_E2E_DECISIONS.csv")
        if len(rows) != expected_rows or len(decisions) != expected_rows:
            raise RuntimeError(f"source row count mismatch: {source}: {len(rows)}, {len(decisions)}")
        all_rows.extend(rows)
        all_decisions.extend(decisions)
        protocols.append(protocol)
        for path in source.rglob("*"):
            if not path.is_file() or path.name in SKIP:
                continue
            rel = path.relative_to(source)
            destination = TARGET / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    if len(all_rows) != 30 or len(all_decisions) != 30:
        raise RuntimeError(f"merged row count mismatch: {len(all_rows)}, {len(all_decisions)}")
    merged = dict(protocols[0])
    merged.update({
        "status": "COMPLETED",
        "roots": [8400, 8401],
        "fresh_roots": [8400, 8401],
        "rows": 30,
        "contexts": 6,
        "mass_bands_kg": {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20},
        "merged_source_protocols": [str(p / "MASS_FRESH_E2E_PROTOCOL.json") for p, _ in SOURCES],
    })
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "MASS_FRESH_E2E_PROTOCOL.json").write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(TARGET / "MASS_FRESH_E2E_ROWS.csv", all_rows)
    write_csv(TARGET / "MASS_FRESH_E2E_DECISIONS.csv", all_decisions)
    print(json.dumps({"status": merged["status"], "rows": len(all_rows), "contexts": merged["contexts"], "roots": merged["roots"]}, sort_keys=True))


if __name__ == "__main__":
    main()
