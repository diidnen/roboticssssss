#!/usr/bin/env python3
"""Write auditable candidate/status artifacts after historical screening.

This file deliberately records failed reproducibility as an infrastructure
eligibility result, not as a physical frontier label.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path("/home/exouser/Tabero")
DISCOVERY = REPO / "analysis/results/true_force_frontier_candidate_discovery_20260904"
OUT = REPO / "analysis/results/true_force_frontier_screening_20260904"
ATTEMPT = REPO / "analysis/results/true_force_frontier_candidate_t0_r00_low_20260904/anchor_6N_retry6"


def main() -> None:
    candidates = list(csv.DictReader((DISCOVERY / "HISTORICAL_FORCE_FRONTIER_CANDIDATES.csv").open(newline="", encoding="utf-8")))
    top = candidates[:5]
    rows = []
    for row in top:
        attempted = row["context_id"] == "p5s0c_train_t0_r00_s5100_low_mu0.293710"
        rows.append({
            "rank": row["rank"],
            "task": row["task"],
            "root_id": row["root_id"],
            "context_id": row["context_id"],
            "selection_reason": "formal historical mixed fail-to-success; high-command full-task success",
            "historical_low_failed_legacy_command": row["lowest_failed_legacy_command"],
            "historical_first_successful_legacy_command": row["lowest_successful_legacy_command"],
            "historical_high_full_task_success": row["high_command_full_remaining_task_success"],
            "current_attempted": "YES" if attempted else "NO",
            "current_handoff_valid": "NO" if attempted else "NOT_RUN",
            "current_probe_valid": "NO" if attempted else "NOT_RUN",
            "current_snapshot_valid": "NO" if attempted else "NOT_RUN",
            "current_high_force_anchor": "LOW_LEVEL_FORCE_EXECUTION_FAILURE" if attempted else "NOT_RUN",
            "current_force_addressable": "NO" if attempted else "NOT_SCREENED",
            "current_screening_note": "scene-only historical snapshot restored without bilateral contact; 6N static mean 0N; not eligible as frontier evidence" if attempted else "awaiting compatible current Frozen-VLA action trace and regenerated current snapshot",
        })

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "TRUE_FORCE_FRONTIER_CANDIDATES.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    attempted_summary = {}
    summary_path = ATTEMPT / "POST_GRASP_FORCE_FRONTIER_SUMMARY.csv"
    if summary_path.exists():
        attempted_summary = next(csv.DictReader(summary_path.open(newline="", encoding="utf-8")), {})
    result = {
        "FINAL_STATUS": "HISTORICAL_CANDIDATE_SCREENING_COMPLETE_CURRENT_FRONTIER_NOT_VALIDATED",
        "historical_formal_data": {"source": str(DISCOVERY / "HISTORICAL_FORCE_FRONTIER_CANDIDATES.csv"), "branches": 576, "contexts": 144, "mixed_contexts": 81, "old720_included": False},
        "root7400_retired": True,
        "top_candidates_screened": [row["context_id"] for row in top],
        "current_runtime_attempt": {
            "context_id": "p5s0c_train_t0_r00_s5100_low_mu0.293710",
            "snapshot_source": "/home/exouser/Tabero/E7_V2_POSTQUERY_20260903/task0_all/POSTQUERY_STATE_SNAPSHOTS/p5s0c_train_t0_r00_s5100_low_mu0.293710.pt",
            "fresh_isaac_process": True,
            "handoff_valid": False,
            "probe_valid": False,
            "snapshot_compatibility": False,
            "high_force_target_N": 6.0,
            "high_force_static_mean_N": attempted_summary.get("static_mean_force_N", 0.0),
            "high_force_realization_valid": False,
            "classification": "HISTORICAL_CANDIDATE_NOT_REPRODUCIBLE",
            "not_frontier_evidence": True,
        },
        "clean_frontier": {"validated": False, "task": None, "root": None, "physical_context": None, "interval": None},
        "continuous_force_realization_generalized": False,
        "one_branch_one_fresh_isaac_process": True,
        "ready_for_5_context_per_task_pilot": False,
        "ready_for_60_context_per_task_collection": False,
        "ready_for_continuous_posterior_training": False,
        "blocker": "The formal mixed contexts have archived scene-only/P5S0C artifacts but no compatible current Frozen-VLA remaining-action trace plus current full runtime snapshot. The attempted historical snapshot restores without bilateral contact in the current runtime, so no valid force branch can be labeled.",
    }
    (OUT / "CLEAN_TRUE_FORCE_FRONTIER_RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUT / "CROSS_ROOT_SANITY_RESULT.json").write_text(json.dumps({"status": "NOT_RUN", "reason": "no clean true-force frontier was validated", "continuous_force_realization_generalized": False}, indent=2) + "\n", encoding="utf-8")
    with (OUT / "CLEAN_TRUE_FORCE_FRONTIER_TIMESERIES.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["context_id", "force_target_N", "static_mean_force_N", "force_realization_valid", "remaining_task_success", "eligibility_class"])
        writer.writeheader()
        writer.writerow({"context_id": result["current_runtime_attempt"]["context_id"], "force_target_N": 6.0, "static_mean_force_N": attempted_summary.get("static_mean_force_N", 0.0), "force_realization_valid": "NO", "remaining_task_success": "NO", "eligibility_class": "HISTORICAL_CANDIDATE_NOT_REPRODUCIBLE"})

    report = """# Current true-force frontier screening report

## Conclusion

The formal historical source contains 576 branches over 144 contexts and 81 mixed success/failure contexts. The top five candidates were ranked without using current outcomes, and old720 reconstructed diagnostics were excluded.

The first current runtime attempt was `Task0 / train root00 / LOW μ=0.293710`, selected because the formal history showed `3/4 FAIL -> 4.5/5 SUCCESS`. Its archived scene-only snapshot restored under the current `libero_object` runtime without bilateral contact; the 6N anchor therefore measured 0N and is classified `HISTORICAL_CANDIDATE_NOT_REPRODUCIBLE`, not `POST_GRASP_FORCE_INSUFFICIENT` and not frontier evidence. A same-seed archived raw trace also failed to establish a current handoff and timed out.

No clean same-state true-physical-force frontier was validated in this run. A compatible current Frozen-VLA remaining-action trace and a current full runtime snapshot for one of the formal candidates are required before any 2/4/6 result can be interpreted scientifically.

The controller, force semantics, probe, estimator, VLA, and frontier taxonomy were not changed. `ONE_BRANCH = ONE_FRESH_ISAAC_PROCESS` was preserved.
"""
    (OUT / "CLEAN_TRUE_FORCE_FRONTIER_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
