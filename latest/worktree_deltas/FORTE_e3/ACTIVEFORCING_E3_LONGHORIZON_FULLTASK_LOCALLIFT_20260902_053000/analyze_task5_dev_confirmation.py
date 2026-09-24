#!/usr/bin/env python3
"""Audit the fixed six-cell root-heldout E3 DEV confirmation.

This analyzer is label-only.  It never computes Utility, selects force, infers
Fmax, or fits/replaces a production Direct model.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOTS = (7500, 7501)
FORCES = (4, 6, 7)
FRICTION = 0.6
CELL_RE = re.compile(r"^DEV_root(7500|7501)_mu0\.6_F(4|6|7)N$")
REGIMES = ("LOCAL_FAILURE", "LOCAL_SUCCESS_DOWNSTREAM_FAILURE", "FULL_TASK_SUCCESS")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot_root", type=Path)
    args = ap.parse_args()

    records: list[dict[str, object]] = []
    errors: list[str] = []
    seen_cells: set[tuple[int, int]] = set()
    for cell in sorted(p for p in args.pilot_root.iterdir() if p.is_dir()):
        match = CELL_RE.match(cell.name)
        if not match:
            continue
        root, force = map(int, match.groups())
        key = (root, force)
        if key in seen_cells:
            errors.append(f"duplicate DEV cell key {key}")
            continue
        seen_cells.add(key)
        ep_files = sorted((cell / "logs").glob("*_episodes.csv"))
        st_files = sorted((cell / "logs").glob("*_steps.csv"))
        if len(ep_files) != 1 or len(st_files) != 1:
            errors.append(f"{cell}: expected one episode/step CSV pair")
            continue
        episodes, steps = read_rows(ep_files[0]), read_rows(st_files[0])
        if len(episodes) != 1 or not steps:
            errors.append(f"{cell}: expected one episode and non-empty step rows")
            continue
        ep = episodes[0]
        ep_force = as_float(ep, "peak_predicted_force_slot_N")
        ep_mu = as_float(ep, "friction")
        root_hash = str(ep.get("root_state_hash", ""))
        query = max(as_int(row, "contact") for row in steps)
        local = as_int(ep, "lift_success")
        full = as_int(ep, "official_success")
        if abs(ep_force - force) > 1e-6:
            errors.append(f"{cell}: force telemetry {ep_force} != planned {force}")
        if abs(ep_mu - FRICTION) > 1e-6:
            errors.append(f"{cell}: friction telemetry {ep_mu} != planned {FRICTION}")
        if len(root_hash) != 64:
            errors.append(f"{cell}: invalid root_state_hash")
        if full and not local:
            errors.append(f"{cell}: FullTask success contradicts LocalLift failure")
        if not query:
            regime = "UNINFORMATIVE_NO_QUERY_STATE"
        elif not local:
            regime = "LOCAL_FAILURE"
        elif not full:
            regime = "LOCAL_SUCCESS_DOWNSTREAM_FAILURE"
        else:
            regime = "FULL_TASK_SUCCESS"
        records.append({
            "split": "DEV",
            "suite": "libero_10",
            "task_id": 5,
            "root_seed": root,
            "friction": ep_mu,
            "force_N": force,
            "root_state_hash": root_hash,
            "query_state_reached": query,
            "local_lift_label": local,
            "full_task_label": full,
            "local_positive_full_negative": int(local == 1 and full == 0),
            "regime": regime,
            "grasp": as_int(ep, "pick_success"),
            "lift": local,
            "transport": as_int(ep, "transport_success"),
            "placement": as_int(ep, "place_success"),
            "measured_force_mean_N": ep.get("mean_measured_force_N", ""),
            "measured_force_peak_N": ep.get("peak_measured_force_N", ""),
            "steps": as_int(ep, "steps"),
            "cell_dir": str(cell),
        })

    expected = {(root, force) for root in ROOTS for force in FORCES}
    observed = {(int(r["root_seed"]), int(r["force_N"])) for r in records}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    regimes_by_root: dict[int, set[str]] = defaultdict(set)
    hashes_by_root: dict[int, set[str]] = defaultdict(set)
    for record in records:
        regimes_by_root[int(record["root_seed"])].add(str(record["regime"]))
        hashes_by_root[int(record["root_seed"])].add(str(record["root_state_hash"]))
    stable_hash_each_root = all(len(hashes_by_root.get(root, set())) == 1 for root in ROOTS)
    root_hashes_distinct = (
        all(len(hashes_by_root.get(root, set())) == 1 for root in ROOTS)
        and len({next(iter(hashes_by_root[root])) for root in ROOTS}) == len(ROOTS)
    )
    three_regime_each_root = all(set(REGIMES) <= regimes_by_root.get(root, set()) for root in ROOTS)
    query_all = bool(records) and all(int(r["query_state_reached"]) == 1 for r in records)
    gate_pass = (
        not errors and not missing and not extra and query_all
        and stable_hash_each_root and root_hashes_distinct and three_regime_each_root
    )

    table_path = args.pilot_root / "E3_TASK5_DEV_FIXED_ANCHOR_TABLE.csv"
    fields = list(records[0]) if records else ["split", "root_seed", "force_N", "regime"]
    with table_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    counts = Counter(str(r["regime"]) for r in records)
    n = len(records)
    local_pos = sum(int(r["local_lift_label"]) for r in records)
    full_pos = sum(int(r["full_task_label"]) for r in records)
    downstream_only = sum(int(r["local_positive_full_negative"]) for r in records)
    comparison = {
        "estimand": "paired label-target divergence on fixed DEV branches",
        "n": n,
        "local_lift_positive": local_pos,
        "full_task_positive": full_pos,
        "local_positive_full_negative": downstream_only,
        "local_lift_positive_rate": local_pos / n if n else None,
        "full_task_positive_rate": full_pos / n if n else None,
        "paired_acceptance_overstatement_rate": downstream_only / n if n else None,
        "paired_positive_rate_gap": (local_pos - full_pos) / n if n else None,
        "interpretation_scope": "label-target comparison only; n=6 is not a powered classifier-generalization estimate",
        "utility_or_Fmax_claim": False,
    }
    summary = {
        "protocol": "E3_TASK5_DEV_CONFIRMATION_V2",
        "records": n,
        "expected_records": 6,
        "errors": errors,
        "missing_cells": missing,
        "extra_cells": extra,
        "query_state_reached_all": query_all,
        "root_state_hashes": {str(root): sorted(hashes_by_root.get(root, set())) for root in ROOTS},
        "stable_hash_each_root": stable_hash_each_root,
        "root_hashes_distinct": root_hashes_distinct,
        "regime_counts": {regime: counts[regime] for regime in REGIMES},
        "regimes_by_root": {str(root): sorted(regimes_by_root.get(root, set())) for root in ROOTS},
        "three_regime_each_root": three_regime_each_root,
        "gate_pass": gate_pass,
        "comparison": comparison,
        "table": str(table_path),
        "Fmax_defined_or_inferred": False,
        "utility_runtime_evaluation": False,
        "test_data_used": False,
    }
    gate_path = args.pilot_root / "E3_TASK5_DEV_CONFIRMATION_GATE.json"
    gate_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = args.pilot_root / "E3_FULLTASK_VS_LOCALLIFT_DEV_REPORT.md"
    report_path.write_text(
        "# E3 FullTask vs LocalLift — fixed DEV branches\n\n"
        f"Gate: `{'PASS' if gate_pass else 'INCOMPLETE_OR_FAILED'}`.\n\n"
        f"- Fixed DEV cells: {n}/6\n"
        f"- LocalLift positive: {local_pos}/{n if n else 0}\n"
        f"- FullTask positive: {full_pos}/{n if n else 0}\n"
        f"- LocalLift-positive / FullTask-negative: {downstream_only}/{n if n else 0}\n"
        f"- Regimes: {dict(counts)}\n"
        f"- Same-root hash stability: {stable_hash_each_root}; distinct held-out roots: {root_hashes_distinct}\n\n"
        "This is a paired label-target comparison on a six-cell fixed DEV confirmation. "
        "It is not a powered classifier-generalization claim, does not replace the authoritative "
        "Shared Direct checkpoint, and makes no Utility, Fmax, or safety-limit claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
