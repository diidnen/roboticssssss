#!/usr/bin/env python3
"""Merge the completed split Mass fresh-E2E runs into the canonical artifact."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).parent
CANONICAL = ROOT / "fresh_e2e"
RUNS = [
    ROOT / "fresh_e2e_root8400_LOW_v2",
    ROOT / "fresh_e2e_root8400_mid_v3",
    ROOT / "fresh_e2e_root8400_high_v2",
    ROOT / "fresh_e2e_root8401",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def merge_csv(name: str) -> int:
    parts = [read_rows(run / name) for run in RUNS]
    rows = [row for part in parts for row in part]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    CANONICAL.mkdir(parents=True, exist_ok=True)
    with (CANONICAL / name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    protocols = []
    for run in RUNS:
        protocol = json.loads((run / "MASS_FRESH_E2E_PROTOCOL.json").read_text(encoding="utf-8"))
        if protocol.get("status") != "COMPLETED":
            raise RuntimeError(f"incomplete run: {run} -> {protocol.get('status')}")
        protocols.append(protocol)
        expected = 5 if protocol.get("contexts") == 1 else 15
        if protocol.get("rows") != expected:
            raise RuntimeError(f"unexpected row count: {run} -> {protocol.get('rows')}")

    rows = merge_csv("MASS_FRESH_E2E_ROWS.csv")
    decisions = merge_csv("MASS_FRESH_E2E_DECISIONS.csv")
    contexts = sum(int(protocol["contexts"]) for protocol in protocols)
    roots = sorted({int(root) for protocol in protocols for root in protocol["roots"]})
    bands = {band: mass for protocol in protocols for band, mass in protocol["mass_bands_kg"].items()}
    merged = dict(protocols[-1])
    merged.update(
        {
            "status": "COMPLETED",
            "rows": rows,
            "decision_rows": decisions,
            "contexts": contexts,
            "fresh_roots": roots,
            "roots": roots,
            "mass_bands_kg": dict(sorted(bands.items())),
            "source_runs": [str(run) for run in RUNS],
            "merged_from_split_runs": True,
        }
    )
    (CANONICAL / "MASS_FRESH_E2E_PROTOCOL.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETED", "rows": rows, "decision_rows": decisions, "contexts": contexts, "roots": roots}, sort_keys=True))


if __name__ == "__main__":
    main()
