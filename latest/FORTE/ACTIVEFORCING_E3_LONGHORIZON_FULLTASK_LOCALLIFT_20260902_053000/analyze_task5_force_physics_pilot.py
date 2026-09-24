#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

FORCES = {1.0, 2.0, 3.0, 4.0, 5.0}
FRICTIONS = {0.2, 0.6, 1.0}
CELL_RE = re.compile(r"^(TRAIN|DEV)_root(\d+)_mu([0-9.]+)_F([0-9.]+)N$")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def i(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0") or 0))
    except (TypeError, ValueError):
        return 0


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot_root", type=Path)
    ap.add_argument("--split", choices=("TRAIN", "DEV"), required=True)
    args = ap.parse_args()

    records: list[dict[str, object]] = []
    errors: list[str] = []
    for cell in sorted(p for p in args.pilot_root.iterdir() if p.is_dir()):
        match = CELL_RE.match(cell.name)
        if not match or match.group(1) != args.split:
            continue
        split, root_s, mu_s, force_s = match.groups()
        root_seed, mu, force = int(root_s), float(mu_s), float(force_s)
        ep_files = sorted((cell / "logs").glob("*_episodes.csv"))
        st_files = sorted((cell / "logs").glob("*_steps.csv"))
        if len(ep_files) != 1 or len(st_files) != 1:
            errors.append(f"{cell}: expected one episode/step CSV pair")
            continue
        episodes, steps = rows(ep_files[0]), rows(st_files[0])
        if len(episodes) != 1 or not steps:
            errors.append(f"{cell}: incomplete episode or empty steps")
            continue
        ep = episodes[0]
        root_hash = str(ep.get("root_state_hash", ""))
        if len(root_hash) != 64:
            errors.append(f"{cell}: missing/invalid root_state_hash")
        query = max(i(r, "contact") for r in steps)
        lift, final = i(ep, "lift_success"), i(ep, "official_success")
        if not query:
            label = "UNINFORMATIVE_NO_QUERY_STATE"
        elif not lift:
            label = "LOCAL_FAILURE"
        elif not final:
            label = "LOCAL_SUCCESS_DOWNSTREAM_FAILURE"
        else:
            label = "FULL_TASK_SUCCESS"
        records.append(
            {
                "split": split,
                "root_seed": root_seed,
                "friction": mu,
                "force_N": force,
                "root_state_hash": root_hash,
                "query_state_reached": query,
                "grasp": i(ep, "pick_success"),
                "lift": lift,
                "transport": int(max(f(r, "e3_obj_xy_displacement_m") for r in steps) >= 0.10),
                "placement": max(i(r, "e3_relationship_success") for r in steps),
                "official_final_success": final,
                "regime": label,
                "measured_force_mean_N": ep.get("mean_measured_force_N", ""),
                "measured_force_peak_N": ep.get("peak_measured_force_N", ""),
                "steps": i(ep, "steps"),
                "cell_dir": str(cell),
            }
        )

    expected_roots = {7400} if args.split == "TRAIN" else {7500, 7501}
    expected = {(r, mu, force) for r in expected_roots for mu in FRICTIONS for force in FORCES}
    observed = {(int(r["root_seed"]), float(r["friction"]), float(r["force_N"])) for r in records}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    by_context: dict[tuple[int, float], set[str]] = defaultdict(set)
    hashes_by_root: dict[int, set[str]] = defaultdict(set)
    for record in records:
        by_context[(int(record["root_seed"]), float(record["friction"]))].add(str(record["regime"]))
        hashes_by_root[int(record["root_seed"])].add(str(record["root_state_hash"]))
    root_hash_mismatches = {
        str(root): sorted(hashes) for root, hashes in hashes_by_root.items() if len(hashes) != 1
    }
    needed = {"LOCAL_FAILURE", "LOCAL_SUCCESS_DOWNSTREAM_FAILURE", "FULL_TASK_SUCCESS"}
    three_regime_contexts = [
        {"root_seed": root, "friction": mu}
        for (root, mu), labels in sorted(by_context.items())
        if needed <= labels
    ]
    full_in_domain = any(r["regime"] == "FULL_TASK_SUCCESS" and float(r["force_N"]) <= 5.0 for r in records)
    if args.split == "TRAIN":
        gate_pass = (
            not errors and not missing and not extra and not root_hash_mismatches
            and bool(three_regime_contexts) and full_in_domain
        )
    else:
        roots_with_three = {int(c["root_seed"]) for c in three_regime_contexts}
        gate_pass = (
            not errors and not missing and not extra and not root_hash_mismatches
            and roots_with_three == expected_roots and full_in_domain
        )

    csv_path = args.pilot_root / f"E3_TASK5_{args.split}_REGIME_TABLE.csv"
    fields = list(records[0]) if records else ["split", "root_seed", "friction", "force_N", "regime"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(records)
    summary = {
        "split": args.split,
        "records": len(records),
        "expected_records": len(expected),
        "errors": errors,
        "missing_cells": missing,
        "extra_cells": extra,
        "root_state_hashes": {str(root): sorted(hashes) for root, hashes in sorted(hashes_by_root.items())},
        "root_state_hash_mismatches": root_hash_mismatches,
        "regime_counts": {name: sum(r["regime"] == name for r in records) for name in sorted(needed)},
        "three_regime_contexts": three_regime_contexts,
        "full_success_within_authoritative_Fmax_5N": full_in_domain,
        "gate_pass": gate_pass,
        "table": str(csv_path),
    }
    out = args.pilot_root / f"E3_TASK5_{args.split}_REGIME_GATE.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
