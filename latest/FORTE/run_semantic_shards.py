#!/usr/bin/env python3
"""Bounded CPU launcher for the 36 frozen semantic-transfer shards."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys

PYTHON = sys.executable
RUNNER = "/home/exouser/FORTE/run_semantic_task_transfer.py"
OUT = "/home/exouser/FORTE/activeforcing_semantic_task_transfer_20260901_120000"


def run(spec):
    t, f, s = spec
    cmd = [PYTHON, RUNNER, "--out", OUT, "shard", "--target", str(t), "--fold", str(f), "--seed", str(s)]
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return spec, p.returncode, p.stdout


def main():
    specs = [(t, f, s) for t in [0, 1, 5, 6] for f in [0, 1, 2] for s in [0, 1, 2]]
    failures = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(run, x) for x in specs]
        for fut in as_completed(futures):
            spec, code, output = fut.result()
            print({"shard": spec, "exit": code}, flush=True)
            if code:
                failures.append((spec, output))
    if failures:
        for spec, output in failures:
            print(f"FAILED {spec}\n{output}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
