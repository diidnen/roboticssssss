#!/usr/bin/env python3
"""Combine policy and simulator CUDA diagnostics into the required artifact."""

import csv
from pathlib import Path


base = Path(__file__).resolve().parent
inputs = [
    base / "CUDA_RUNTIME_REPEAT_TEST_POLICY.csv",
    base / "CUDA_RUNTIME_REPEAT_TEST_SIMULATOR.csv",
]
rows = []
for path in inputs:
    with path.open(newline="") as handle:
        rows.extend(csv.DictReader(handle))

with (base / "CUDA_RUNTIME_REPEAT_TEST.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"combined_rows={len(rows)}")
