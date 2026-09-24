#!/usr/bin/env python3
"""Collect only missing contact/slip telemetry for Physics-GRU v2.

This wrapper reuses the frozen P5-S0-C Isaac worker.  The worker's scientific
execution semantics are unchanged; a compatibility flag selects the minimum
force-boundary neighborhood per context and the worker logs direct finger
contact forces and object velocity in the new output namespace.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO = Path("/home/exouser/Tabero")
ARTIFACT = REPO / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542"
WORKER = REPO / "analysis/p5s0c_paired_boundary_probe_value.py"
RESULTS = REPO / "analysis/results"


def load_module():
    spec = importlib.util.spec_from_file_location("p5s0c_worker_for_v2", WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {WORKER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boundary_filter() -> dict[str, list[float]]:
    manifest = pd.read_csv(ARTIFACT / "P5S0C_BRANCH_MANIFEST.csv")
    out: dict[str, list[float]] = {}
    groups = list(manifest.groupby("context_id", sort=True))
    single = os.environ.get("V2_SINGLE_CONTEXT", "")
    if single:
        groups = [(cid, g) for cid, g in groups if str(cid) == single]
    if os.environ.get("V2_STRATIFIED_FRONTIER_ONLY"):
        chosen_groups = {}
        for context_id, g in groups:
            key = (int(g.task.iloc[0]), str(g.split.iloc[0]), str(g.friction_band.iloc[0]))
            chosen_groups.setdefault(key, (context_id, g))
        groups = list(chosen_groups.values())
    for context_id, g in groups:
        forces = sorted(float(x) for x in g.requested_force_N.unique())
        successes = sorted(float(x) for x in g.loc[g.full_task_success_y == 1, "requested_force_N"].unique())
        if not successes:
            chosen = forces
        else:
            frontier = successes[0]
            idx = forces.index(frontier)
            if os.environ.get("V2_STRATIFIED_FRONTIER_ONLY"):
                chosen = [frontier]
            else:
                chosen = forces[max(0, idx - 1) : min(len(forces), idx + 2)]
        out[str(context_id)] = chosen
    return out


def main() -> int:
    out = RESULTS / ("learned_physical_imagination_v2_contact_telemetry_20260828_" + datetime.now(timezone.utc).strftime("%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    filt = boundary_filter()
    filter_path = out / "BOUNDARY_FORCE_FILTER.json"
    filter_path.write_text(json.dumps(filt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    module = load_module()
    os.environ.update(
        {
            "P5S0C_OUT": str(out),
            "P5S0C_FORCE_FILTER_JSON": str(filter_path),
            "P5S0C_SKIP_REPLAY": "" if os.environ.get("V2_ENABLE_REPLAY") else "1",
            "V2_STRATIFIED_FRONTIER_ONLY": os.environ.get("V2_STRATIFIED_FRONTIER_ONLY", ""),
            "P5S0C_CONTEXT_ID": os.environ.get("V2_SINGLE_CONTEXT", ""),
        }
    )
    module.OUT = out
    if os.environ.get("P5S0C_WORKER") == "1":
        return int(module.worker_main())
    # The original launcher is intentionally not duplicated: it launches one
    # fresh Isaac process per task and inherits the filter/output contract.
    records = [module.launch_worker(out, task) for task in module.TASKS]
    (out / "WORKER_LAUNCH.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return 0 if all(int(r["returncode"]) == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
