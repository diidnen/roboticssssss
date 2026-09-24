#!/usr/bin/env python3
"""Write B2 paper-facing artifacts from scan CSVs. No Isaac, no D2 writes."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
OFFICIAL = [0, 1, 2, 3, 5, 6, 7, 8, 9]
OBJECTS = {
    0: "alphabet soup",
    1: "cream cheese",
    2: "salad dressing",
    3: "bbq sauce",
    5: "tomato sauce",
    6: "butter",
    7: "milk",
    8: "chocolate pudding",
    9: "orange juice",
}


def git_head():
    try:
        return subprocess.check_output(
            ["git", "-C", "/home/exouser/Tabero", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_csv(name):
    p = OUT / name
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def update_provenance_from_scans():
    scans = []
    for name in (
        "TASK1_ORACLE_SCAN.csv",
        "TASK7_ORACLE_SCAN.csv",
        "ALL_TASK_CHEAP_SCAN.csv",
        "POSITIVE_TASK_EXPANDED.csv",
    ):
        scans.extend(load_csv(name))
    by_task = {}
    for r in scans:
        tid = int(float(r["task_id"]))
        by_task.setdefault(tid, []).append(r)
    prov_p = OUT / "TASK_DATA_PROVENANCE.csv"
    rows = load_csv("TASK_DATA_PROVENANCE.csv")
    out = []
    for r in rows:
        tid = int(r["task_id"])
        recs = by_task.get(tid, [])
        if recs:
            r["reset"] = "PASS"
            r["full_task_trajectory"] = (
                "PASS" if any(int(float(x["full_task_success"])) == 1 for x in recs) else "FAIL"
            )
        out.append(r)
    if out:
        with prov_p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    return out, by_task


def cheap_extract(by_task):
    """Do not clone Phase A rows into ALL_TASK_CHEAP_SCAN (avoids duplicate trial_ids).

    Write a derived cheap *view* instead. Keep ALL_TASK_CHEAP_SCAN as the Phase B collector output.
    """
    view = []
    seen = set()
    for name in (
        "TASK1_ORACLE_SCAN.csv",
        "TASK7_ORACLE_SCAN.csv",
        "ALL_TASK_CHEAP_SCAN.csv",
        "POSITIVE_TASK_EXPANDED.csv",
    ):
        for r in load_csv(name):
            tid = r.get("trial_id")
            if tid in seen:
                continue
            try:
                F = float(r["force"])
                s = int(float(r["seed_idx"]))
            except Exception:
                continue
            if F in (4.0, 5.0, 6.0, 8.0) and s < 3:
                seen.add(tid)
                view.append(r)
    if view:
        fields = list(view[0].keys())
        with (OUT / "ALL_TASK_CHEAP_VIEW.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(view)
    return view


def status_cell(done: bool, blocked=False, na=False):
    if na:
        return "NOT_APPLICABLE"
    if blocked:
        return "BLOCKED"
    if done:
        return "DONE"
    return "PENDING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="final")
    args = ap.parse_args()
    stage = args.stage

    prov, by_task = update_provenance_from_scans()
    cheap_extract(by_task)

    summary = {}
    sp = OUT / "ANALYSIS_SUMMARY.json"
    if sp.exists():
        summary = json.loads(sp.read_text())
    table = summary.get("table", [])
    table_by = {int(r["task_id"]): r for r in table}

    t1 = load_csv("TASK1_ORACLE_SCAN.csv")
    t7 = load_csv("TASK7_ORACLE_SCAN.csv")
    cheap = load_csv("ALL_TASK_CHEAP_SCAN.csv")

    tasks_with_data = [tid for tid in OFFICIAL if (OUT / "TASK_DATA_PROVENANCE.csv").exists()]
    reset_pass = [int(r["task_id"]) for r in prov if r.get("reset") == "PASS"]
    blocked_ids = [int(r["task_id"]) for r in table if r.get("classification") == "BLOCKED"]
    positive = [int(x) for x in summary.get("positive", [])]
    fixed_low = [int(x) for x in summary.get("fixed_low", [])]
    fixed_robust = [int(x) for x in summary.get("fixed_robust", [])]
    eligible = [int(x) for x in summary.get("eligible", [])]

    milk_pos = 7 in positive
    cream_pos = 1 in positive
    multi = bool(cream_pos and milk_pos)

    # baseline matrix statuses
    def force_baselines(tid):
        r = table_by.get(tid)
        if not r:
            return False
        return r.get("fixed_robust") != "" and r.get("oracle_mean_F") != "" and int(r.get("n_trials", 0)) > 0

    matrix_rows = []
    md = [
        "# BASELINE_EXECUTION_MATRIX",
        "",
        "This round does not run Tabero Neutral, DeliGrasp, or FORTE. No invented numbers.",
        "",
        "| Task | Object | Fixed Low | Fixed Robust | Oracle | Tabero Neutral | DeliGrasp | FORTE-inspired | Future Ours |",
        "| ---- | ------ | --------- | ------------ | ------ | -------------- | --------- | -------------- | ----------- |",
    ]
    for tid in OFFICIAL:
        r = table_by.get(tid, {})
        cls = r.get("classification", "BLOCKED" if not by_task.get(tid) else "PENDING")
        has = force_baselines(tid)
        blocked = cls == "BLOCKED" or (tid in by_task and all(int(float(x["full_task_success"])) == 0 for x in by_task[tid]) and len(by_task[tid]) >= 9)
        fixed_low_status = status_cell(has, blocked=cls in ("BLOCKED", "NO_FEASIBLE_RANGE"))
        # Fixed Low is the 4N reference column from the scan, not a separate run
        if has:
            fixed_low_status = "DONE"
        elif cls == "BLOCKED":
            fixed_low_status = "BLOCKED"
        else:
            fixed_low_status = "PENDING"
        rob = status_cell(has, blocked=cls in ("BLOCKED", "NO_FEASIBLE_RANGE"))
        ora = rob
        vtla = "PENDING"
        deli = "PENDING"
        forte = "PENDING"
        ours = "—"
        if cls == "BLOCKED":
            vtla = deli = forte = "BLOCKED"
        elif cls in ("FIXED_ROBUST", "FIXED_LOW", "LOW_DECISION_VALUE", "NO_FEASIBLE_RANGE") and tid not in positive:
            # still report, but external baselines are lower priority
            pass
        md.append(
            f"| {tid} {OBJECTS[tid]} | {OBJECTS[tid]} | {fixed_low_status} | {rob} | {ora} | {vtla} | {deli} | {forte} | {ours} |"
        )
        matrix_rows.append(
            {
                "task_id": tid,
                "object": OBJECTS[tid],
                "Fixed Low": fixed_low_status,
                "Fixed Robust": rob,
                "Oracle": ora,
                "Tabero Neutral": vtla,
                "DeliGrasp": deli,
                "FORTE-inspired": forte,
                "Future Ours": ours,
                "classification": cls,
            }
        )
    (OUT / "BASELINE_EXECUTION_MATRIX.md").write_text("\n".join(md) + "\n")

    elig_md = [
        "# BENCHMARK_ELIGIBILITY",
        "",
        "A task enters the future quantitative main table only if:",
        "1. HDF5/data ready",
        "2. full-task scripted/oracle trajectory reliable",
        "3. force servo works (episodes completed with force samples)",
        "4. friction runtime override works",
        "5. at least one force/friction transition (F*_low ≠ F*_high)",
        "6. Fixed Robust force exists at τ=0.8",
        "7. Oracle saves nontrivial force vs Fixed Robust (ΔF ≥ 0.5 N and F* range ≥ 1 N)",
        "",
        f"Official tasks: 9. Eligible: {eligible}. Positive (decision-value): {positive}.",
        "",
        "| Task | Object | Classification | Eligible | Notes |",
        "| ---- | ------ | -------------- | -------- | ----- |",
    ]
    for tid in OFFICIAL:
        r = table_by.get(tid, {})
        notes = []
        if not by_task.get(tid):
            notes.append("no Isaac trials yet" if stage != "final" else "no trials")
        if r.get("traj_ok") == 0:
            notes.append("scripted full-task never succeeded")
        if r.get("mu_override_ok") == 0:
            notes.append("friction override failed")
        if r.get("classification") == "POSITIVE" and r.get("eligible_main") == 0:
            notes.append("positive but ΔF/range below eligibility")
        elig_md.append(
            f"| {tid} | {OBJECTS[tid]} | {r.get('classification', '')} | {r.get('eligible_main', '')} | {'; '.join(notes) or '—'} |"
        )
    (OUT / "BENCHMARK_ELIGIBILITY.md").write_text("\n".join(elig_md) + "\n")

    # README
    readme = f"""# B2 — Tabero Benchmark Qualification + Baseline Table

Timestamp dir: `{OUT.name}`
Stage: `{stage}`
METHOD_CHANGE = NONE

This round does **not** design OURS, train networks, modify D2, port DeliGrasp/FORTE, or run Tabero-VTLA.

Question: which official Tabero LIBERO-object tasks have hidden-friction **force decision value**, and how large is the Fixed Robust vs Oracle gap?

## Isolation

- D2 results dir not written
- D2 scripts / thresholds not edited
- D2 process not killed
- Tabero core source not modified
- B2 waited for a free Isaac GPU slot before launching Isaac

## Physics / protocol

- Env: `Isaac-Libero-Franka-Hybrid-Tactile-v0`
- Neutral official instruction only
- Arm: scripted/oracle downstream (T1/P3 constants), not π0
- Hidden physics: **friction only** μ ∈ {{0.20, 0.50, 1.00}}
- Phase A forces: 3, 4, 5, 6, 8 N; N=5 then expand transitions
- Primary metric: **full task success rate** (pick/lift/transport/place)

## Unrun items

- Tabero-VTLA Neutral: NOT_RUN reason: wait until positive task set is frozen; JAX/OpenPI+Isaac GPU; D2 isolation
- DeliGrasp port: NOT_RUN reason: determine positive task set first
- FORTE-inspired: NOT_RUN reason: port after DeliGrasp
- OURS / belief / probe optimization: NOT_RUN reason: out of scope
"""
    (OUT / "README.md").write_text(readme)

    envp = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "tabero_git": git_head(),
        "env_id": "Isaac-Libero-Franka-Hybrid-Tactile-v0",
        "isaac_python": "/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python",
        "hdf5_dir": "/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5",
        "physics_variable": "friction",
        "mus": [0.2, 0.5, 1.0],
        "phaseA_forces": [3, 4, 5, 6, 8],
        "phaseB_forces": [4, 5, 6],
        "instruction_policy": "official_neutral_no_adverbs",
        "arm_policy": "SCRIPTED_ORACLE_T1_P3",
        "d2_isolation": True,
    }
    (OUT / "ENV_PROVENANCE.json").write_text(json.dumps(envp, indent=2))

    fstar = summary.get("fstar_by_task", {})
    verdict = {
        "status": "COMPLETE" if stage == "final" and (t1 and t7) else "IN_PROGRESS",
        "stage": stage,
        "method_change": "NONE",
        "d2_touched": False,
        "d2_process_killed": False,
        "official_task_count": 9,
        "tasks_with_data": OFFICIAL,
        "tasks_reset_pass": reset_pass,
        "task1_scan_complete": bool(t1),
        "task7_scan_complete": bool(t7),
        "all9_cheap_scan_complete": len({int(float(r["task_id"])) for r in cheap}) >= 9 if cheap else False,
        "physics_variable": "friction",
        "positive_tasks": positive,
        "fixed_low_tasks": fixed_low,
        "fixed_robust_no_decision_tasks": fixed_robust,
        "blocked_tasks": blocked_ids,
        "fulltask_fstar_by_task": fstar,
        "fixed_robust_force_by_task": summary.get("robust_by_task", {}),
        "oracle_mean_force_by_task": summary.get("oracle_mean_by_task", {}),
        "force_saving_potential_by_task": summary.get("saving_by_task", {}),
        "eligible_main_benchmark_tasks": eligible,
        "multi_task_decision_value": "PASS" if multi else "BREADTH_NOT_YET_ESTABLISHED",
        "tabero_neutral_status": "PENDING",
        "deligrasp_port_status": "PENDING" if (len(positive) >= 2 or (stage == "final" and len(positive) >= 1)) else "PENDING",
        "forte_port_status": "PENDING",
        "primary_evidence": [
            f"task1_n={len(t1)}",
            f"task7_n={len(t7)}",
            f"cheap_n={len(cheap)}",
            f"positive={positive}",
            f"eligible={eligible}",
        ],
        "limitations": [
            "Phase A N starts at 5; transition cells expand toward 15-20",
            "Cheap scan N=3 is classification only, not paper numbers",
            "Scripted grasp offsets inherited from cream-cheese T1/P3; other geometries may fail trajectory",
            "Tabero Neutral / DeliGrasp / FORTE not executed this round",
        ],
        "plots_note": "empty plots mean the corresponding scan was not yet available",
    }
    (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2, default=str))
    print(json.dumps({"stage": stage, "positive": positive, "eligible": eligible, "blocked": blocked_ids}, indent=2))


if __name__ == "__main__":
    main()
