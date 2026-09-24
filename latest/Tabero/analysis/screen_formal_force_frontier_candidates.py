#!/usr/bin/env python3
"""Screen only the formal P5S0C 433/143 data for historical force candidates.

Legacy requested-force values are retained as command labels only.  This
script performs candidate discovery; it does not infer a command-to-newton
mapping and does not run any simulator branch.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


REPO = Path("/home/exouser/Tabero")
SOURCE = REPO / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542"
OUT = REPO / "analysis/results/true_force_frontier_candidate_discovery_20260904"


def yes(row: dict[str, str]) -> bool:
    return row.get("full_task_success_y", "0") in {"1", "1.0", "True", "true"}


def f(row: dict[str, str]) -> float:
    return float(row["requested_force_N"])


def main() -> None:
    rows: list[dict[str, str]] = []
    for path in sorted(SOURCE.glob("task*/branches.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["context_id"]].append(row)

    candidates: list[dict[str, object]] = []
    for context_id, branch_rows in sorted(groups.items()):
        branch_rows.sort(key=f)
        outcomes = {f(row): int(yes(row)) for row in branch_rows}
        failed = [force for force, outcome in outcomes.items() if not outcome]
        succeeded = [force for force, outcome in outcomes.items() if outcome]
        if not failed or not succeeded:
            continue
        low_failed = min(failed)
        first_success = min(succeeded)
        high_row = next(row for row in branch_rows if f(row) == first_success)
        lower_rows = [row for row in branch_rows if f(row) < first_success]
        repeated_lower_fail = len(lower_rows) >= 2 and all(not yes(row) for row in lower_rows)
        full_high = yes(high_row)
        task = int(branch_rows[0]["task"])
        source = str((SOURCE / f"task{task}" / "branches.csv").resolve())
        candidates.append(
            {
                "rank": 0,
                "priority": "A" if full_high and repeated_lower_fail else "B" if full_high else "C",
                "task": task,
                "split": branch_rows[0]["split"],
                "context_id": context_id,
                "root_id": branch_rows[0]["root_id"],
                "root_index": branch_rows[0]["root_index"],
                "root_seed": branch_rows[0]["root_seed"],
                "friction_band": branch_rows[0]["friction_band"],
                "hidden_friction_analysis_only": branch_rows[0]["hidden_friction_analysis_only"],
                "legacy_commands_tested": ";".join(f"{force:g}" for force in sorted(outcomes)),
                "legacy_outcomes": ";".join(f"{force:g}:{outcomes[force]}" for force in sorted(outcomes)),
                "lowest_failed_legacy_command": low_failed,
                "lowest_successful_legacy_command": first_success,
                "mixed_success_failure": "YES",
                "repeated_lower_failures": "YES" if repeated_lower_fail else "NO",
                "high_command_full_remaining_task_success": "YES" if full_high else "NO",
                "high_command_lift_success": high_row.get("lift_success", ""),
                "high_command_transport_retention": high_row.get("transport_retention", ""),
                "high_command_place_success": high_row.get("place_success", ""),
                "high_command_failure_reason": high_row.get("failure_reason", ""),
                "post_probe_state_hash": branch_rows[0].get("post_probe_state_hash", ""),
                "state_parity": branch_rows[0].get("state_parity", ""),
                "probe_telemetry_path": branch_rows[0].get("telemetry_path", ""),
                "source_artifact": source,
                "candidate_selection_basis": "historical formal boundary-focused data only; legacy command labels are not Newtons",
            }
        )

    # High full-task success, repeated lower failures, and narrow historical
    # transition are deterministic discovery-time priorities only.
    candidates.sort(
        key=lambda row: (
            0 if row["priority"] == "A" else 1 if row["priority"] == "B" else 2,
            -(1 if row["high_command_full_remaining_task_success"] == "YES" else 0),
            float(row["lowest_successful_legacy_command"]) - float(row["lowest_failed_legacy_command"]),
            int(row["task"]),
            str(row["context_id"]),
        )
    )
    for rank, row in enumerate(candidates, 1):
        row["rank"] = rank

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "HISTORICAL_FORCE_FRONTIER_CANDIDATES.csv"
    fields = list(candidates[0]) if candidates else ["rank"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)

    counts = defaultdict(int)
    for row in candidates:
        counts[row["priority"]] += 1
    top = candidates[:5]
    md = [
        "# Historical formal force-frontier candidate screening",
        "",
        f"Source: `{SOURCE}`",
        "",
        "This screening uses only the formal P5S0C boundary-focused 433-success/143-failure dataset (576 branches, 144 physical contexts). The old720 reconstructed diagnostic set is excluded. Legacy requested-force values are command labels for candidate discovery only; no command-to-Newton conversion is used.",
        "",
        f"- Formal branches scanned: {len(rows)}",
        f"- Contexts scanned: {len(groups)}",
        f"- Mixed fail/success contexts: {len(candidates)}",
        f"- Priority counts: {dict(counts)}",
        "",
        "## Top candidates",
        "",
        "| rank | priority | task | context | legacy outcome sequence | low failed | first successful | high full task |",
        "|---:|:---:|---:|:---|:---|---:|---:|:---:|",
    ]
    for row in top:
        md.append(
            f"| {row['rank']} | {row['priority']} | Task{row['task']} | `{row['context_id']}` | `{row['legacy_outcomes']}` | {float(row['lowest_failed_legacy_command']):g} | {float(row['lowest_successful_legacy_command']):g} | {row['high_command_full_remaining_task_success']} |"
        )
    md += [
        "",
        "## Selection rule",
        "",
        "Priority A requires a mixed context, at least two lower failed commands, and a first-success branch that completed the historical remaining task. Priority B requires a mixed context and a high-command full-task success. Current true-force reruns must still establish handoff, probe, same-state snapshot, force realization, and fresh-process branch isolation; historical outcomes are not formal physical-force evidence.",
        "",
        "The complete ranked table is in `HISTORICAL_FORCE_FRONTIER_CANDIDATES.csv`.",
        "",
    ]
    (OUT / "HISTORICAL_FORCE_FRONTIER_SCREENING.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "screening_summary.json").write_text(
        json.dumps({"formal_rows": len(rows), "contexts": len(groups), "mixed_contexts": len(candidates), "priority_counts": dict(counts), "top_candidates": top}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"formal_rows": len(rows), "contexts": len(groups), "mixed_contexts": len(candidates), "top": top}, indent=2, default=str))


if __name__ == "__main__":
    main()
