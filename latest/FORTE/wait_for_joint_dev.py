#!/usr/bin/env python3
"""Read-only progress watcher for the single already-running DEV collector."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000/collection_dev")
EXPECTED = {1: (4, 180), 5: (2, 90), 6: (1, 25)}


def counts(task: int) -> tuple[int, int]:
    context = ROOT / f"task{task}/task{task}/context.csv"
    branches = ROOT / f"task{task}/task{task}/branches.csv"
    return (
        len(pd.read_csv(context)) if context.exists() and context.stat().st_size else 0,
        len(pd.read_csv(branches)) if branches.exists() and branches.stat().st_size else 0,
    )


def main() -> None:
    previous = None
    while True:
        state = {task: counts(task) for task in EXPECTED}
        if state != previous:
            print(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "state": state}), flush=True)
            previous = state
        if all(state[task] == EXPECTED[task] for task in EXPECTED):
            print(json.dumps({"status": "ALL_TASKWISE_DEV_ROWS_PRESENT"}), flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
