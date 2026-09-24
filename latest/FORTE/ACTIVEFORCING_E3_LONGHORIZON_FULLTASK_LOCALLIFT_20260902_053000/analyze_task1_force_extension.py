#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

FORCES = {1.0, 2.0, 4.0, 5.0, 6.0}
FRICTIONS = {0.2, 0.6, 1.0}
CELL_RE = re.compile(r"^(TRAIN|DEV)_root(\d+)_mu([0-9.]+)_F([0-9.]+)N$")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0") or 0))
    except (TypeError, ValueError):
        return 0


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_root", type=Path)
    parser.add_argument("--split", choices=("TRAIN", "DEV"), required=True)
    args = parser.parse_args()
    records: list[dict[str, object]] = []
    errors: list[str] = []
    for cell in sorted(path for path in args.pilot_root.iterdir() if path.is_dir()):
        match = CELL_RE.match(cell.name)
        if not match or match.group(1) != args.split:
            continue
        split, root_text, friction_text, force_text = match.groups()
        root_seed, friction, force = int(root_text), float(friction_text), float(force_text)
        episode_files = sorted((cell / "logs").glob("*_episodes.csv"))
        step_files = sorted((cell / "logs").glob("*_steps.csv"))
        if len(episode_files) != 1 or len(step_files) != 1:
            errors.append(f"{cell}: expected one episode/step CSV pair")
            continue
        episodes, steps = read_rows(episode_files[0]), read_rows(step_files[0])
        if len(episodes) != 1 or not steps:
            errors.append(f"{cell}: incomplete episode or empty steps")
            continue
        episode = episodes[0]
        root_hash = str(episode.get("root_state_hash", ""))
        if len(root_hash) != 64:
            errors.append(f"{cell}: missing/invalid root_state_hash")
        query = max(as_int(row, "contact") for row in steps)
        lift, final = as_int(episode, "lift_success"), as_int(episode, "official_success")
        if not query:
            regime = "UNINFORMATIVE_NO_QUERY_STATE"
        elif not lift:
            regime = "LOCAL_FAILURE"
        elif not final:
            regime = "LOCAL_SUCCESS_DOWNSTREAM_FAILURE"
        else:
            regime = "FULL_TASK_SUCCESS"
        records.append({
            "split": split,
            "root_seed": root_seed,
            "friction": friction,
            "force_N": force,
            "root_state_hash": root_hash,
            "query_state_reached": query,
            "grasp": as_int(episode, "pick_success"),
            "lift": lift,
            "transport": int(max(as_float(row, "e3_obj_xy_displacement_m") for row in steps) >= 0.10),
            "placement": max(as_int(row, "e3_relationship_success") for row in steps),
            "official_final_success": final,
            "regime": regime,
            "measured_force_mean_N": episode.get("mean_measured_force_N", ""),
            "measured_force_peak_N": episode.get("peak_measured_force_N", ""),
            "steps": as_int(episode, "steps"),
            "cell_dir": str(cell),
        })
    expected_roots = {7600} if args.split == "TRAIN" else {7700, 7701}
    expected = {(root, friction, force) for root in expected_roots for friction in FRICTIONS for force in FORCES}
    observed = {(int(row["root_seed"]), float(row["friction"]), float(row["force_N"])) for row in records}
    context_regimes: dict[tuple[int, float], set[str]] = defaultdict(set)
    hashes_by_root: dict[int, set[str]] = defaultdict(set)
    for row in records:
        context_regimes[(int(row["root_seed"]), float(row["friction"]))].add(str(row["regime"]))
        hashes_by_root[int(row["root_seed"])].add(str(row["root_state_hash"]))
    needed = {"LOCAL_FAILURE", "LOCAL_SUCCESS_DOWNSTREAM_FAILURE", "FULL_TASK_SUCCESS"}
    three_regime = [
        {"root_seed": root, "friction": friction}
        for (root, friction), labels in sorted(context_regimes.items()) if needed <= labels
    ]
    mismatches = {str(root): sorted(values) for root, values in hashes_by_root.items() if len(values) != 1}
    full_in_domain = any(row["regime"] == "FULL_TASK_SUCCESS" and float(row["force_N"]) <= 6 for row in records)
    common = not errors and observed == expected and not mismatches and bool(three_regime) and full_in_domain
    gate = common if args.split == "TRAIN" else common and {int(x["root_seed"]) for x in three_regime} == expected_roots
    table_path = args.pilot_root / f"E3_TASK1_{args.split}_REGIME_TABLE.csv"
    fields = list(records[0]) if records else ["split", "root_seed", "friction", "force_N", "regime"]
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "split": args.split,
        "records": len(records),
        "expected_records": len(expected),
        "errors": errors,
        "missing_cells": sorted(expected - observed),
        "extra_cells": sorted(observed - expected),
        "root_state_hashes": {str(root): sorted(values) for root, values in sorted(hashes_by_root.items())},
        "root_state_hash_mismatches": mismatches,
        "regime_counts": {name: sum(row["regime"] == name for row in records) for name in sorted(needed)},
        "three_regime_contexts": three_regime,
        "full_success_within_authoritative_Fmax_6N": full_in_domain,
        "gate_pass": gate,
        "table": str(table_path),
    }
    gate_path = args.pilot_root / f"E3_TASK1_{args.split}_REGIME_GATE.json"
    gate_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
