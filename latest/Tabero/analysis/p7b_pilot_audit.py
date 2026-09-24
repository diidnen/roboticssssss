#!/usr/bin/env python3
"""Audit-only P7-B query infrastructure gate; never changes labels."""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def audit(out: Path) -> dict:
    c = pd.read_csv(out / "P7B_CONTEXT_MANIFEST.csv")
    b = pd.read_csv(out / "P7B_BRANCH_MANIFEST.csv")
    u = c.drop_duplicates("context_id", keep="last")
    horizon = b.error.fillna("").eq("vla_chunk_budget_exhausted") | num(b.get("vla_chunk_budget_exhausted", 0)).eq(1)
    true_runtime = b.error.fillna("").ne("") & ~horizon
    query_failures = num(u.query_failure).eq(1)
    qualified = num(u.query_qualified).eq(1)
    parity = num(b.state_parity).eq(1) & num(b.query_state_parity).eq(1)
    reasons = Counter(str(x) for x in u.loc[query_failures, "query_failure_reason"].fillna("unknown")) if "query_failure_reason" in u else Counter({"legacy_reason_unavailable": int(query_failures.sum())})
    per_force = {}
    if not b.empty:
        for force, g in b.groupby("requested_force_N"):
            per_force[str(force)] = {"n": int(len(g)), "successes": int(num(g.full_task_success_y).sum()), "rate": float(num(g.full_task_success_y).mean())}
    force_rates = [v["rate"] for v in per_force.values()]
    legacy = Path("/home/exouser/Tabero/analysis/results/p7b_gnp_physical_belief_force_planning_20260827_pilot_v2/P7B_BRANCH_MANIFEST.csv")
    legacy_check = {}
    if legacy.exists():
        old = pd.read_csv(legacy)
        old_horizon = old.error.fillna("").eq("vla_chunk_budget_exhausted")
        legacy_check = {"path": str(legacy), "records": int(len(old)), "legacy_horizon_records": int(old_horizon.sum()), "legacy_true_runtime_errors": int((old.error.fillna("").ne("") & ~old_horizon).sum()), "legacy_error_field_nonempty": int(old.error.fillna("").ne("").sum()), "reclassified_horizon_as_chunk_budget_exhausted": int(old_horizon.sum()), "recheck_pass": bool(len(old) == 80 and old_horizon.sum() == 56 and (old.error.fillna("").ne("") & ~old_horizon).sum() == 0)}
    unique = int(u.context_id.nunique())
    branch_count = int(len(b))
    return_valid = num(u.get("return_state_valid", 0)).eq(1) if "return_state_valid" in u else pd.Series(False, index=u.index)
    gate_checks = {"unique_contexts_20": unique == 20, "query_qualified_20": int(qualified.sum()) >= 20, "query_rate_100pct": unique == 20 and float(qualified.mean() if unique else 0.0) >= 1.0, "contact_retention_20": int(num(u.contact_retained).sum()) >= 20, "return_state_valid_20": int(return_valid.sum()) >= 20, "state_parity_all": bool(len(b) == 80 and bool(parity.all())), "branch_count_80": branch_count == 80, "true_runtime_errors_zero": int(true_runtime.sum()) == 0}
    passed = bool(all(gate_checks.values()))
    summary = {"contexts_manifest_rows": int(len(c)), "unique_contexts": unique, "branches": branch_count, "query_qualified": int(qualified.sum()), "query_qualification_rate": float(qualified.mean() if unique else 0.0), "query_failures": int(query_failures.sum()), "failure_reason_breakdown": dict(reasons), "contact_retained": int(num(u.contact_retained).sum()), "return_state_valid": int(return_valid.sum()), "state_parity_all": bool(len(b) > 0 and num(b.state_parity).eq(1).all()), "query_state_parity_all": bool(len(b) > 0 and num(b.query_state_parity).eq(1).all()), "true_runtime_errors": int(true_runtime.sum()), "runtime_error_rows_raw": int(b.error.fillna("").ne("").sum()), "chunk_budget_exhaustion_count": int(horizon.sum()), "chunk_budget_exhaustion_is_runtime_error": False, "object_displacement_m": {"max": float(num(u.get("obj_disp_probe_m", 0)).max() if len(u) else 0.0), "mean": float(num(u.get("obj_disp_probe_m", 0)).mean() if len(u) else 0.0)}, "object_rotation_rad": {"max": float(num(u.get("obj_rot_probe_rad", 0)).max() if len(u) else 0.0), "mean": float(num(u.get("obj_rot_probe_rad", 0)).mean() if len(u) else 0.0)}, "major_disturbance_count": int(num(u.get("major_disturbance", 0)).sum()), "return_state_valid_count": int(return_valid.sum()), "force_outcomes": per_force, "force_dependent_full_task_outcomes_exist": bool(len(set(round(x, 12) for x in force_rates)) > 1), "gate_checks": gate_checks, "pilot_pass": passed, "pilot_gate_reason": "P7B_QUERY_INFRASTRUCTURE_QUALIFIED" if passed else "QUERY_QUALIFICATION_GATE_FAILED", "legacy_56_recheck": legacy_check}
    (out / "P7B_PILOT_AUDIT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = "\n".join(["# P7-B Clean Pilot Audit", "", "This is an infrastructure gate only; no belief/planner training or scientific classification is authorized here.", "", f"- Unique contexts: {summary['unique_contexts']}", f"- Query qualified: {summary['query_qualified']}/{summary['unique_contexts']} ({summary['query_qualification_rate']:.1%})", f"- Query failures: {summary['query_failures']} ({json.dumps(summary['failure_reason_breakdown'], sort_keys=True)})", f"- Branches: {summary['branches']}", f"- State parity: {summary['state_parity_all']} (query parity: {summary['query_state_parity_all']})", f"- Contact retained: {summary['contact_retained']}", f"- Return-state valid: {summary['return_state_valid']}", f"- True runtime errors: {summary['true_runtime_errors']}", f"- Chunk-budget exhaustion: {summary['chunk_budget_exhaustion_count']} (not runtime errors)", f"- Force-dependent full-task outcomes: {summary['force_dependent_full_task_outcomes_exist']}", "", "## Verdict", "", f"`{summary['pilot_gate_reason']}`", "", "No main collection, model training, planner adjudication, held-out evaluation, evidence ablation, fresh E2E, or A–G classification is launched by this audit."])
    (out / "P7B_PILOT_REPORT.md").write_text(report + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    audit(Path(sys.argv[1]))
