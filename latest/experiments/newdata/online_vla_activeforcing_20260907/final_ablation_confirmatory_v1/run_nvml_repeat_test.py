#!/usr/bin/env python3
"""Repeat the exact launcher NVML query and preserve every result."""

import argparse
import csv
import datetime as dt
import subprocess
import time
from pathlib import Path


QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    rows = []
    for index in range(args.attempts):
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        proc = subprocess.run(QUERY, text=True, capture_output=True)
        rows.append(
            {
                "attempt": index + 1,
                "timestamp_utc": started,
                "exit_code": proc.returncode,
                "stdout": proc.stdout.strip().replace("\n", " | "),
                "stderr": proc.stderr.strip().replace("\n", " | "),
            }
        )
        if index + 1 < args.attempts:
            time.sleep(args.interval)

    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    successes = sum(row["exit_code"] == 0 for row in rows)
    print(f"NVML_SUCCESS_COUNT={successes}")
    print(f"NVML_FAILURE_COUNT={len(rows) - successes}")


if __name__ == "__main__":
    main()
