#!/usr/bin/env python3
"""Independent QA for the CPU-only closure artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "BOUNDARY_AWARE_FULL_TASK_FEASIBILITY_DATA_CLOSURE_20260902"


def check(name: str, value: bool, detail: str = "") -> dict:
    return {"check": name, "pass": bool(value), "detail": detail}


def main() -> None:
    results = []
    labels = pd.read_csv(OUT / "AUTHORITATIVE_720_BOUNDARY_LABELS.csv")
    boundaries = pd.read_csv(OUT / "EXISTING_BOUNDARY_STATUS.csv")
    support = pd.read_csv(OUT / "CURRENT_FORCE_SUPPORT_AUDIT.csv")
    nxt = pd.read_csv(OUT / "BOUNDARY_NEXT_QUERY_STATE.csv")
    cal = pd.read_csv(OUT / "OLD_DIRECT_CALIBRATION_BASELINE.csv")
    aug = json.loads((OUT / "BOUNDARY_AUGMENTED_DATASET_MANIFEST.json").read_text())
    e3 = json.loads((OUT / "CURRENT_TASK_E3_MECHANISM_READY.json").read_text())
    resource = json.loads((OUT / "RESOURCE_GATE_STATUS.json").read_text())

    results += [
        check("authoritative_rows_720", len(labels) == 720, str(len(labels))),
        check("task_counts_180_each", labels.groupby("task").size().to_dict() == {0: 180, 1: 180, 5: 180, 6: 180}, str(labels.groupby("task").size().to_dict())),
        check("contexts_72", labels.context_id.nunique() == 72, str(labels.context_id.nunique())),
        check("telemetry_720", int(labels.telemetry_exists.sum()) == 720, str(int(labels.telemetry_exists.sum()))),
        check("boundary_rows_72", len(boundaries) == 72, str(len(boundaries))),
        check("qualification_contexts_24", len(nxt) == 24 and nxt.context_id.nunique() == 24, str(len(nxt))),
        check("two_roots_three_frictions_every_task", all(len(q) == 6 and q.root_index.nunique() == 2 and q.friction_band.nunique() == 3 for _, q in nxt.groupby("task"))),
        check("no_candidate_launchable", int(nxt.candidate_is_launchable.sum()) == 0, str(int(nxt.candidate_is_launchable.sum()))),
        check("utility_not_used", int(nxt.utility_used.sum()) == 0),
        check("augmented_manifest_truthful_pending", aug["status"] == "PENDING_BOUNDARY_COLLECTION" and aug["new_boundary_branches"] == 0 and not aug["training_allowed"]),
        check("no_simulator_launch_decision", resource["decision"] == "NO_SIMULATOR_LAUNCH" and resource["processes_killed_or_modified"] == 0),
        check("mechanism_ready_tasks_1_6", e3["ready_tasks"] == [1, 6], str(e3["ready_tasks"])),
    ]

    # Recompute Brier independently from the exported joined labels.
    for task, q in labels.groupby("task", sort=True):
        brier = float(np.mean((q.p_D_OOF_ensemble.to_numpy(float) - q.full_task_success.to_numpy(int)) ** 2))
        reported = float(cal[(cal.scope == "TASK") & (cal.task.astype(str) == str(task))].brier.iloc[0])
        results.append(check(f"task{task}_brier_recomputed", abs(brier - reported) < 1e-12, f"{brier} vs {reported}"))

    # Recompute label regimes, without importing the generator.
    balances = {}
    for task, q in labels.groupby("task", sort=True):
        balances[int(task)] = {
            "local_failure": int((q.local_lift_success == 0).sum()),
            "delayed": int(((q.local_lift_success == 1) & (q.full_task_success == 0)).sum()),
            "full_success": int((q.full_task_success == 1).sum()),
        }
    expected = {0: {"local_failure": 0, "delayed": 39, "full_success": 141}, 1: {"local_failure": 11, "delayed": 44, "full_success": 125}, 5: {"local_failure": 0, "delayed": 47, "full_success": 133}, 6: {"local_failure": 2, "delayed": 3, "full_success": 175}}
    results.append(check("regime_counts_recomputed", balances == expected, str(balances)))

    task_support = support[support.row_scope == "TASK_SUMMARY"].set_index("task")
    expected_ranges = {0: (3.034164, 4.978178), 1: (4.001775, 5.990140), 5: (3.004209, 4.964331), 6: (3.001526, 3.996558)}
    for task, (lo, hi) in expected_ranges.items():
        observed = (float(task_support.loc[task, "requested_force_min_N"]), float(task_support.loc[task, "requested_force_max_N"]))
        results.append(check(f"task{task}_force_support", abs(observed[0]-lo) < 1e-5 and abs(observed[1]-hi) < 1e-5, str(observed)))

    required = [
        "CURRENT_FORCE_SUPPORT_AUDIT.csv", "CURRENT_FORCE_SUPPORT_AUDIT.md", "EXISTING_BOUNDARY_STATUS.csv",
        "BOUNDARY_ACQUISITION_PROTOCOL.json", "FORCE_INTERFACE_SAFETY_AUDIT.csv", "FORCE_INTERFACE_SAFETY_AUDIT.md",
        "OLD_DIRECT_CALIBRATION_BASELINE.csv", "OLD_DIRECT_BOUNDARY_METRICS.csv", "DIRECT_BASELINE_BOUNDARY_REPORT.md",
        "CURRENT_TASK_E3_MECHANISM_READY.json", "BOUNDARY_AUGMENTED_DATASET_MANIFEST.json",
        "BOUNDARY_DIRECT_MATCHED_TRAINING_PROTOCOL.json", "BOUNDARY_DIRECT_FRESH_TEST_PLAN.md",
        "BOUNDARY_COLLECTION_HANDOFF_TO_E3.md", "BOUNDARY_COLLECTION_HANDOFF_TO_MASS.md", "BOUNDARY_COLLECTION_HANDOFF_TO_JOINT.md",
    ]
    results.append(check("required_phase1_files", all((OUT / x).exists() for x in required), str([x for x in required if not (OUT / x).exists()])))

    ok = all(x["pass"] for x in results)
    report = {"status": "PASS" if ok else "FAIL", "checks": results, "source_independent_of_generators": True}
    (OUT / "PHASE1_INDEPENDENT_VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
    md = ["# Phase-1 independent validation", "", f"Status: `{'PASS' if ok else 'FAIL'}`", "", "| Check | Pass | Detail |", "|---|---:|---|"]
    for r in results:
        md.append(f"| {r['check']} | {str(r['pass']).upper()} | {r['detail'].replace('|', '/')} |")
    md += ["", "No simulator was launched and no protected process or source archive was modified by this phase.", ""]
    (OUT / "PHASE1_INDEPENDENT_VALIDATION.md").write_text("\n".join(md))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
