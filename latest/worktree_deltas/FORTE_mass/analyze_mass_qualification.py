#!/usr/bin/env python3
"""Summarize qualification branches without touching formal-test decisions."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

OUT = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500")
TASKS = {
    0: "alphabet_soup_1", 1: "cream_cheese_1", 2: "salad_dressing_1",
    3: "bbq_sauce_1", 5: "tomato_sauce_1", 6: "butter_1",
    7: "milk_1", 8: "chocolate_pudding_1", 9: "orange_juice_1",
}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_int(row, key):
    try:
        return int(float(row.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def as_float(row, key):
    try:
        return float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def main():
    all_branches = []
    all_contexts = []
    for task in TASKS:
        for contexts in OUT.glob(f"M2_TASK{task}_QUAL*/M2_TASK{task}_QUAL_CONTEXTS.csv"):
            all_contexts.extend(read_csv(contexts))
        for branches in OUT.glob(f"M2_TASK{task}_QUAL*/M2_TASK{task}_QUAL_BRANCHES.csv"):
            all_branches.extend(read_csv(branches))
    summary = []
    for task in TASKS:
        rows = [r for r in all_branches if int(float(r["task_id"])) == task]
        valid_contexts = [r for r in all_contexts if int(float(r["task_id"])) == task and as_int(r, "probe_valid")]
        by_force = defaultdict(list)
        by_mass = defaultdict(list)
        for r in rows:
            by_force[as_float(r, "requested_force_N")].append(as_int(r, "full_task_success_y"))
            by_mass[r["mass_band"]].append(as_int(r, "full_task_success_y"))
        force_rates = {str(k): sum(v) / len(v) for k, v in sorted(by_force.items())}
        mass_rates = {k: sum(v) / len(v) for k, v in sorted(by_mass.items())}
        delayed = sum(as_int(r, "lift_success") and not as_int(r, "full_task_success_y") for r in rows)
        successes = sum(as_int(r, "full_task_success_y") for r in rows)
        force_sensitive = len(set(force_rates.values())) >= 2
        mass_sensitive = len(set(mass_rates.values())) >= 2
        outcome_mixed = 0 < successes < len(rows)
        qualifies = bool(valid_contexts and outcome_mixed and (force_sensitive or mass_sensitive))
        summary.append({
            "task_id": task,
            "task_name": f"libero_object_{task}",
            "object": TASKS[task],
            "probe_valid_contexts": len(valid_contexts),
            "branches": len(rows),
            "full_task_successes": successes,
            "full_task_success_rate": successes / len(rows) if rows else 0.0,
            "force_rates_json": str(force_rates),
            "mass_band_rates_json": str(mass_rates),
            "force_sensitive": int(force_sensitive),
            "mass_sensitive": int(mass_sensitive),
            "delayed_failure_count": delayed,
            "delayed_failure_frequency": delayed / len(rows) if rows else 0.0,
            "qualification_status": "QUALIFIED" if qualifies else "NOT_QUALIFIED",
            "vla_reach_evidence": "EXISTING_LIBERO_TASK_ASSET; fresh VLA E2E deferred to M7",
        })
    out_csv = OUT / "MASS_TASK_QUALIFICATION.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ["status"])
        writer.writeheader(); writer.writerows(summary)
    lines = ["# Mass task qualification report", "", "Qualification is a development screen and is not part of formal totals.", "", "| task | valid query contexts | branches | SR | force-sensitive | mass-sensitive | delayed failures | status |", "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in summary:
        lines.append(f"| {r['task_id']} | {r['probe_valid_contexts']} | {r['branches']} | {float(r['full_task_success_rate']):.3f} | {r['force_sensitive']} | {r['mass_sensitive']} | {r['delayed_failure_count']} | {r['qualification_status']} |")
    lines += ["", "## Promotion rule", "", "A task is promoted only if the query is valid, outcomes are mixed, and force or mass changes the outcome. The existing frozen-LIBERO reach asset is recorded as a prerequisite; no new VLA is trained."]
    (OUT / "MASS_TASK_QUALIFICATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
