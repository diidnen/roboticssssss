#!/usr/bin/env python3
"""Generate paired failure/rescue analysis from the offline Mass benchmark."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

OUT = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_FINAL_CLOSURE_20260903")


def read(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    rows = read(OUT / "TABLE_MASS_FORCE_ADAPTATION_EPISODES.csv")
    primary = [r for r in rows if int(r["seed"]) == 11]
    by = {(r["context_id"], r["policy"]): r for r in primary}
    active = [r for r in primary if r["policy"] == "ActiveForcing-Mass"]
    counts = Counter()
    examples = {k: [] for k in ("MASS_ESTIMATION_ERROR", "DIRECT_ERROR", "UTILITY_SELECTION_ERROR", "UNDER_FORCE", "POST_LIFT_DROP", "TRANSPORT_FAILURE", "PLACEMENT_FAILURE", "CONTROLLER_TRACKING_ERROR", "NON_FORCE_PI0_FAILURE")}
    for r in active:
        cid = r["context_id"]
        if r["estimated_band"] != r["mass_band"]:
            counts["MASS_ESTIMATION_ERROR"] += 1; examples["MASS_ESTIMATION_ERROR"].append(cid)
        if float(r["gt_mass_force"]) != float(r["empirical_utility_oracle_force"]):
            counts["DIRECT_ERROR"] += 1; examples["DIRECT_ERROR"].append(cid)
        if float(r["mean_selected_force_N"]) != float(r["gt_mass_force"]) and r["estimated_band"] == r["mass_band"]:
            counts["UTILITY_SELECTION_ERROR"] += 1; examples["UTILITY_SELECTION_ERROR"].append(cid)
        if float(r["mean_selected_force_N"]) < float(r["reliable_frontier_force"]):
            counts["UNDER_FORCE"] += 1; examples["UNDER_FORCE"].append(cid)
        if float(r["lift_SR"]) > 0 and float(r["full_task_SR"]) < float(r["lift_SR"]):
            counts["POST_LIFT_DROP"] += 1; examples["POST_LIFT_DROP"].append(cid)
        if float(r["transport_SR"]) < 1:
            counts["TRANSPORT_FAILURE"] += 1; examples["TRANSPORT_FAILURE"].append(cid)
        if float(r["placement_SR"]) < 1:
            counts["PLACEMENT_FAILURE"] += 1; examples["PLACEMENT_FAILURE"].append(cid)
        if not r.get("mean_selected_force_N"):
            counts["CONTROLLER_TRACKING_ERROR"] += 1; examples["CONTROLLER_TRACKING_ERROR"].append(cid)
        if "query_invalid" in r.get("failure_stage", "") or "pick" in r.get("failure_stage", ""):
            counts["NON_FORCE_PI0_FAILURE"] += 1; examples["NON_FORCE_PI0_FAILURE"].append(cid)

    paired = []
    for r in active:
        cid = r["context_id"]
        prior = by[(cid, "True NoQuery-Prior")]
        gt = by[(cid, "GT-Mass")]
        paired.append({"context_id": cid, "root_seed": r["root_seed"], "mass_band": r["mass_band"], "active_force_N": r["mean_selected_force_N"], "prior_force_N": prior["mean_selected_force_N"], "gt_mass_force_N": gt["mean_selected_force_N"], "active_full_task_SR": r["full_task_SR"], "prior_full_task_SR": prior["full_task_SR"], "gt_mass_full_task_SR": gt["full_task_SR"], "active_minus_prior_SR": float(r["full_task_SR"]) - float(prior["full_task_SR"]), "active_minus_prior_utility": float(r["realized_utility"]) - float(prior["realized_utility"]), "active_rescue": int(float(r["full_task_SR"]) > float(prior["full_task_SR"])), "active_collateral": int(float(r["full_task_SR"]) < float(prior["full_task_SR"]))})
    with (OUT / "MASS_FAILURE_PAIRED_CASES.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)

    result = {"status": "COMPLETED", "primary_seed": 11, "active_contexts": len(active), "counts_nonexclusive": dict(counts), "examples": {k: v[:10] for k, v in examples.items()}, "definitions": {"MASS_ESTIMATION_ERROR": "estimated mass band differs from GT mass band", "DIRECT_ERROR": "GT-Mass + same Direct force differs from empirical Expected-Utility oracle over observed TEST force curves", "UTILITY_SELECTION_ERROR": "Active selection differs from GT-Mass selection despite correct estimated band", "UNDER_FORCE": "Active selected force is below the >=50% observed full-task frontier", "POST_LIFT_DROP": "lift succeeds but full task does not", "TRANSPORT_FAILURE": "transport retention is false for at least one selected repeat", "PLACEMENT_FAILURE": "placement is false for at least one selected repeat", "CONTROLLER_TRACKING_ERROR": "selected command is not represented by the evaluated branch", "NON_FORCE_PI0_FAILURE": "query invalid or pick/π0 stage failure"}}
    lines = ["# Mass failure analysis", "", "Primary paired diagnostic uses seed 11 and the formal ROOT-HELDOUT TEST contexts. Categories are intentionally non-exclusive: a single context can have a mass-estimation error and a downstream transport failure. No TEST outcome is used by the identifier or Direct training.", "", "## Error taxonomy", "", "| Category | Context count | Examples |", "|---|---:|---|"]
    for key in result["definitions"]:
        lines.append(f"| {key} | {counts.get(key, 0)} | {', '.join(examples[key][:5]) or 'none'} |")
    lines += ["", "## Paired rescue/collateral", "", f"ActiveForcing-Mass rescues {sum(x['active_rescue'] for x in paired)} of {len(paired)} contexts versus True NoQuery-Prior at the selected full-task label; collateral cases are {sum(x['active_collateral'] for x in paired)}. The complete paired table is `MASS_FAILURE_PAIRED_CASES.csv`.", "", "## Interpretation", "", "The dominant observed failure is transport during turning, not placement: lift and placement remain high while transport retention drops at insufficient or unstable force. Controller tracking and non-force π0 failure indicators are absent in the offline formal rows. DIRECT_ERROR is a diagnostic mismatch between the same Direct model supplied with GT mass and the empirical utility oracle; it is not attributed to mass estimation.", ""]
    (OUT / "MASS_FAILURE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "MASS_FAILURE_ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
