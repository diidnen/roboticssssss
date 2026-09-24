#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def as_int(row: dict, key: str) -> int:
    try:
        return int(float(row.get(key, 0) or 0))
    except Exception:
        return 0


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--suite", choices=("libero_10", "libero_goal"), required=True)
    args = ap.parse_args()
    logs = args.task_dir / "logs"
    episodes_files = sorted(logs.glob("*_episodes.csv"))
    steps_files = sorted(logs.glob("*_steps.csv"))
    if len(episodes_files) != 1 or len(steps_files) != 1:
        raise RuntimeError(f"expected one episodes and one steps CSV in {logs}")
    episodes = read_csv(episodes_files[0])
    steps = read_csv(steps_files[0])
    if len(episodes) != 1 or not steps:
        raise RuntimeError("minimal qualification must contain one complete episode and nonempty steps")
    ep = episodes[0]
    task_id = as_int(ep, "task_id")
    goal_statuses = []
    for row in steps:
        try:
            value = json.loads(row.get("e3_goal_status_json", "[]") or "[]")
        except json.JSONDecodeError:
            value = []
        goal_statuses.append([int(bool(item)) for item in value])
    goal_count = max((len(value) for value in goal_statuses), default=0)
    goal_status_ever_each = [
        max((value[index] if index < len(value) else 0) for value in goal_statuses)
        for index in range(goal_count)
    ]
    goal_status_final = goal_statuses[-1] if goal_statuses else []
    query_state_reached = max(as_int(r, "contact") for r in steps)
    max_xy_displacement_m = max(float(r.get("e3_obj_xy_displacement_m", 0) or 0) for r in steps)
    transport_threshold_m = 0.10
    task_suite = args.suite
    close_required = task_suite == "libero_10" and task_id in (3, 9)
    open_required = task_suite == "libero_goal" and task_id == 3
    turn_required = task_suite == "libero_10" and task_id in (2, 8)
    summary = {
        "task_suite": task_suite,
        "task_id": task_id,
        "goal_status_ever_each": goal_status_ever_each,
        "goal_status_final": goal_status_final,
        "query_state_reached": query_state_reached,
        "grasp": as_int(ep, "pick_success"),
        "lift": as_int(ep, "lift_success"),
        "transport": int(max_xy_displacement_m >= transport_threshold_m),
        "max_object_xy_displacement_m": max_xy_displacement_m,
        "transport_distance_threshold_m": transport_threshold_m,
        "placement": max(as_int(r, "e3_relationship_success") for r in steps),
        "close_required": int(close_required),
        "close": max(as_int(r, "e3_close_success") for r in steps) if close_required else None,
        "open_required": int(open_required),
        "open": max(as_int(r, "e3_open_success") for r in steps) if open_required else None,
        "turn_required": int(turn_required),
        "turn": max(as_int(r, "e3_turn_success") for r in steps) if turn_required else None,
        "official_final_success": as_int(ep, "official_success"),
        "max_steps": as_int(ep, "steps"),
        "mean_measured_force_N": ep.get("mean_measured_force_N", ""),
        "peak_measured_force_N": ep.get("peak_measured_force_N", ""),
        "evidence": {"episodes": str(episodes_files[0]), "steps": str(steps_files[0])},
    }
    summary["nominal_capability_qualified"] = int(summary["official_final_success"] == 1)
    (args.task_dir / "QUALIFICATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.task_dir / "QUALIFICATION_SUMMARY.csv").open("w", newline="", encoding="utf-8") as fh:
        row = {k: v for k, v in summary.items() if k != "evidence"}
        wr = csv.DictWriter(fh, fieldnames=list(row))
        wr.writeheader()
        wr.writerow(row)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
