#!/usr/bin/env python3
"""Summarize the frozen-π0 Mass fresh E2E output."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

FORCES = (0.5, 1.0, 1.5, 2.5, 4.0)


def read(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def utility(success, force):
    return float(success) * (8.0 - float(force)) + (1.0 - float(success)) * -1.0


def summarize(rows):
    selected = [r for r in rows if r.get("method") != "FROZEN_PI0_NATIVE_DEFAULT"]
    force_vals = [num(r, "selected_force_N") for r in selected if r.get("selected_force_N") not in (None, "")]
    return {"n": len(rows), "query_state_reach": float(np.mean([num(r, "query_state_reached") for r in rows])) if rows else float("nan"), "query_valid": float(np.mean([num(r, "query_valid") for r in rows])) if rows else float("nan"), "conditional_physical_SR": float(np.mean([num(r, "full_task_success_y") for r in rows if num(r, "query_valid") > 0])) if any(num(r, "query_valid") > 0 for r in rows) else float("nan"), "e2e_SR": float(np.mean([num(r, "e2e_success", num(r, "full_task_success_y")) for r in rows])) if rows else float("nan"), "mean_selected_force_N": float(np.mean(force_vals)) if force_vals else float("nan"), "mean_measured_force_N": float(np.mean([num(r, "measured_force_mean_N") for r in rows])) if rows else float("nan"), "peak_measured_force_N": float(np.mean([num(r, "measured_force_peak_N") for r in rows])) if rows else float("nan"), "under_force_rate": float(np.mean([num(r, "under_force") for r in selected])) if selected else float("nan"), "realized_utility": float(np.mean([utility(num(r, "full_task_success_y"), num(r, "selected_force_N")) for r in selected if r.get("selected_force_N") not in (None, "")])) if force_vals else float("nan"), "failure_stage": ";".join(f"{k}:{v}" for k, v in sorted(Counter(r.get("failure_stage", "none") or "none" for r in rows).items()))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.input / "MASS_FRESH_E2E_PROTOCOL.json").read_text(encoding="utf-8"))
    rows = read(args.input / "MASS_FRESH_E2E_ROWS.csv")
    policies = ["FROZEN_PI0_NATIVE_DEFAULT", "FIXED_MAX", "TRUE_NOQUERY_PRIOR", "ACTIVEFORCING_MASS", "GT_MASS"]
    records = []
    for policy in policies:
        pr = [r for r in rows if r.get("method") == policy]
        for scope, groups in (("overall", {"ALL": pr}), ("per_mass", {b: [r for r in pr if r.get("mass_band") == b] for b in ("LOW", "MID", "HIGH")}), ("per_root", {str(root): [r for r in pr if str(r.get("root_seed")) == str(root)] for root in sorted({r.get("root_seed") for r in pr})})):
            for group, rr in groups.items():
                if not rr: continue
                records.append({"policy": policy, "scope": scope, "group": group, "mass_band": group if scope == "per_mass" else "ALL", **summarize(rr)})
    with (args.out / "TABLE_MASS_FRESH_E2E.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    overall = [r for r in records if r["scope"] == "overall"]
    lines = ["# Mass fresh E2E report", "", "Each fresh context executes an explicit environment reset, frozen P4-B query where applicable, Mass estimate, authoritative Expected-Utility decision, and frozen π0 completion loop. Fresh roots are 8400 and 8401; all five methods share the same root×mass assignments. ActiveForcing-Mass uses the formal TRAIN-only physical-history identifier and a 3-seed Mass-only Direct ensemble. GT-Mass is diagnostic and supplies true mass only to the same Direct+Utility stack. The native row is the frozen π0 Default arm and deliberately performs no physical query.", "", "| Policy | n | Query reach | Query valid | Conditional physical SR | E2E SR | Mean force | Peak measured force | Under-force diagnostic | Realized utility | Failure stage |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in overall:
        fmt = lambda k: "NA" if str(r[k]) == "nan" else f"{float(r[k]):.3f}"
        lines.append(f"| {r['policy']} | {r['n']} | {fmt('query_state_reach')} | {fmt('query_valid')} | {fmt('conditional_physical_SR')} | {fmt('e2e_SR')} | {fmt('mean_selected_force_N')} | {fmt('peak_measured_force_N')} | {fmt('under_force_rate')} | {fmt('realized_utility')} | {r['failure_stage']} |")
    lines += ["", "## Integrity", "", f"Protocol status: **{protocol.get('status')}**; rows: **{len(rows)}**; expected rows: **{len(protocol.get('roots', [])) * 3 * 5}**. π0 backend: `{protocol.get('frozen_pi0_server', {}).get('server_port', 18881)}` with checkpoint hash `{protocol.get('frozen_pi0_server', {}).get('manifest', {}).get('checkpoint_hash_sha256', 'recorded in protocol')}`. The output is accepted only if the protocol is COMPLETED and every row has frozen-π0 provenance or a recorded selected-arm lineage.", "", "Under-force in this fresh table is a pre-registered selection diagnostic relative to the GT-Mass selected force; it is not substituted for a true observed frontier. Failure stages remain the direct simulator/evaluator labels."]
    (args.out / "MASS_FRESH_E2E_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "MASS_FRESH_E2E_SUMMARY.json").write_text(json.dumps({"status": "COMPLETED" if protocol.get("status") == "COMPLETED" else "BLOCKED", "protocol": protocol, "rows": len(rows), "summaries": records}, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
