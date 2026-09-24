#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

CELL_RE = re.compile(r"^TRAIN_root7400_mu0\.6_F([0-9.]+)N$")
EXPECTED_ROOT_HASH = "a26a8457835421558a23e34b54c04a9f3f6815b6f4cad87a84fcda6e9bb6c816"
ALLOWED_FORCES = set(range(1, 9))


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
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    errors: list[str] = []
    for cell in sorted(path for path in args.pilot_root.iterdir() if path.is_dir()):
        match = CELL_RE.match(cell.name)
        if not match:
            continue
        raw_force = float(match.group(1))
        if not raw_force.is_integer() or int(raw_force) not in ALLOWED_FORCES:
            errors.append(f"{cell}: force must be one of 1..8 N")
            continue
        force = int(raw_force)
        episode_files = sorted((cell / "logs").glob("*_episodes.csv"))
        step_files = sorted((cell / "logs").glob("*_steps.csv"))
        if len(episode_files) != 1 or len(step_files) != 1:
            errors.append(f"{cell}: expected exactly one episode/step CSV pair")
            continue
        episodes = read_rows(episode_files[0])
        steps = read_rows(step_files[0])
        if len(episodes) != 1 or not steps:
            errors.append(f"{cell}: incomplete episode or empty steps")
            continue
        episode = episodes[0]
        root_hash = str(episode.get("root_state_hash", ""))
        query = max(as_int(row, "contact") for row in steps)
        lift = as_int(episode, "lift_success")
        full = as_int(episode, "official_success")
        if not query:
            regime = "UNINFORMATIVE_NO_QUERY_STATE"
        elif not lift:
            regime = "LOCAL_FAILURE"
        elif not full:
            regime = "LOCAL_SUCCESS_DOWNSTREAM_FAILURE"
        else:
            regime = "FULL_TASK_SUCCESS"
        records.append(
            {
                "split": "TRAIN",
                "suite": "libero_10",
                "task_id": 5,
                "root_seed": 7400,
                "friction": 0.6,
                "force_N": force,
                "root_state_hash": root_hash,
                "query_state_reached": query,
                "grasp": as_int(episode, "pick_success"),
                "lift": lift,
                "transport": int(
                    max(as_float(row, "e3_obj_xy_displacement_m") for row in steps) >= 0.10
                ),
                "placement": max(as_int(row, "e3_relationship_success") for row in steps),
                "official_final_success": full,
                "regime": regime,
                "measured_force_mean_N": episode.get("mean_measured_force_N", ""),
                "measured_force_peak_N": episode.get("peak_measured_force_N", ""),
                "steps": as_int(episode, "steps"),
                "cell_dir": str(cell),
            }
        )

    records.sort(key=lambda row: int(row["force_N"]))
    observed_forces = {int(row["force_N"]) for row in records}
    full_forces = sorted(
        int(row["force_N"]) for row in records if row["regime"] == "FULL_TASK_SUCCESS"
    )
    terminal_force = min(full_forces) if full_forces else 8
    expected_forces = set(range(1, terminal_force + 1))
    missing_forces = sorted(expected_forces - observed_forces)
    unexpected_forces = sorted(observed_forces - ALLOWED_FORCES)
    regimes = {str(row["regime"]) for row in records}
    required_regimes = {
        "LOCAL_FAILURE",
        "LOCAL_SUCCESS_DOWNSTREAM_FAILURE",
        "FULL_TASK_SUCCESS",
    }
    hashes = {str(row["root_state_hash"]) for row in records}
    hashes_match = hashes == {EXPECTED_ROOT_HASH}
    query_valid = all(int(row["query_state_reached"]) == 1 for row in records)
    gate_pass = bool(
        not errors
        and not missing_forces
        and not unexpected_forces
        and hashes_match
        and query_valid
        and required_regimes <= regimes
    )

    table_path = args.pilot_root / "E3_TASK5_TRAIN_SUPPORT_EXTENSION_TABLE.csv"
    fields = list(records[0]) if records else ["force_N", "regime"]
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "protocol": "E3_TASK5_TRAIN_FORCE_SUPPORT_EXTENSION_V1",
        "records": len(records),
        "observed_forces_N": sorted(observed_forces),
        "stop_after_first_full_task_success": True,
        "first_full_task_success_force_N": min(full_forces) if full_forces else None,
        "expected_forces_through_stop_N": sorted(expected_forces),
        "missing_forces_through_stop_N": missing_forces,
        "unexpected_forces_N": unexpected_forces,
        "errors": errors,
        "expected_root_state_hash": EXPECTED_ROOT_HASH,
        "observed_root_state_hashes": sorted(hashes),
        "root_state_hash_match": hashes_match,
        "all_query_states_valid": query_valid,
        "regime_counts": {
            name: sum(row["regime"] == name for row in records)
            for name in sorted(required_regimes)
        },
        "three_regime_support": required_regimes <= regimes,
        "gate_pass": gate_pass,
        "utility_claim": False,
        "Fmax_defined_or_inferred": False,
        "table": str(table_path),
    }
    gate_path = args.pilot_root / "E3_TASK5_TRAIN_SUPPORT_EXTENSION_GATE.json"
    gate_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
