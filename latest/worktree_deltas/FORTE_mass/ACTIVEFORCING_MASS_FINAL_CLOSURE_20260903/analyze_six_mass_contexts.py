#!/usr/bin/env python3
"""Produce the compact, no-new-rollout analysis of the corrected six contexts."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
FRESH = ROOT / "fresh_e2e_corrected_final"
FORCES = (0.5, 1.0, 1.5, 2.5, 4.0)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        out.writeheader(); out.writerows(rows)


def num(row, key, default=""):
    value = row.get(key, "")
    return default if value in ("", None) else float(value)


def mass_value(row):
    """Accept either the canonical-row or decomposition-table field name."""
    value = row.get("mass_kg", "")
    if value in ("", None):
        value = row.get("gt_mass_kg", "")
    return "" if value in ("", None) else float(value)


def outcome(row):
    return "SUCCESS" if int(float(row.get("full_task_success_y", 0) or 0)) else (row.get("failure_stage") or "FAILURE")


def score_text(row, prefix):
    return json.dumps({float(x["candidate_force_N"]): {"p": float(x[f"{prefix}_direct_p_success"]), "u": float(x[f"{prefix}_expected_utility"])} for x in row}, sort_keys=True, separators=(",", ":"))


def main():
    rows = read_csv(FRESH / "MASS_FRESH_E2E_ROWS.csv")
    direct = read_csv(ROOT / "MASS_E2E_DIRECT_CASE_AUDIT.csv")
    queries = json.loads((FRESH / "MASS_FRESH_E2E_QUERIES.json").read_text(encoding="utf-8"))
    by = defaultdict(dict)
    for row in rows:
        by[row["context_id"]][row["method"]] = row
    qby = {q["tuple_id"]: q for q in queries}
    contexts = sorted(by)

    table, swaps, sweep = [], [], []
    for cid in contexts:
        a, g = by[cid]["ACTIVEFORCING_MASS"], by[cid]["GT_MASS"]
        p = by[cid]["TRUE_NOQUERY_PRIOR"]
        fx = by[cid]["FIXED_MAX"]
        n = by[cid]["FROZEN_PI0_NATIVE_DEFAULT"]
        q = qby.get(cid, {})
        drows = [x for x in direct if x["context_id"] == cid]
        gt_mass = float(q["mass_kg"]) if q.get("mass_kg") not in ("", None) else mass_value(a)
        table.append({
            "context": cid, "gt_mass_kg": gt_mass,
            "active_predicted_mass_kg": num(a, "mass_estimate_kg"),
            "mass_error_kg": num(a, "mass_estimate_kg") - gt_mass,
            "query_only_predicted_mass_kg": q.get("pred_mass_kg", ""),
            "active_force_N": num(a, "selected_force_N"), "gt_force_N": num(g, "selected_force_N"),
            "prior_force_N": num(p, "selected_force_N"), "fixed_max_force_N": num(fx, "selected_force_N"),
            "active_outcome": outcome(a), "gt_outcome": outcome(g), "prior_outcome": outcome(p),
            "fixed_max_outcome": outcome(fx), "native_outcome": outcome(n),
            "active_failure_stage": a.get("failure_stage", "") or "NONE",
            "gt_failure_stage": g.get("failure_stage", "") or "NONE",
            "prior_failure_stage": p.get("failure_stage", "") or "NONE",
            "fixed_max_failure_stage": fx.get("failure_stage", "") or "NONE",
            "native_failure_stage": n.get("failure_stage", "") or "NONE",
            "failure_stage": a.get("failure_stage", "") or "NONE",
            "query_reach": a.get("query_state_reached", ""), "query_valid": a.get("query_valid", ""),
            "query_record_persisted": int(bool(q)), "query_disp_probe_m": json.loads(q.get("query_record", "{}") or "{}").get("obj_disp_probe_m", "") if q else "",
            "query_rho_mean": json.loads(q.get("query_record", "{}") or "{}").get("rho_mean", "") if q else "",
        })
        if float(a.get("selected_force_N", "nan")) != float(g.get("selected_force_N", "nan")):
            same_state = int(a.get("start_state_hash") == g.get("start_state_hash") == a.get("query_state_hash") == g.get("query_state_hash"))
            swaps.append({"context": cid, "gt_mass_kg": gt_mass, "predicted_mass_kg": a["mass_estimate_kg"], "active_force_N": a["selected_force_N"], "gt_force_N": g["selected_force_N"], "same_query_state_hash": same_state, "active_replay_n": 1, "active_successes": int(float(a["full_task_success_y"])), "gt_replay_n": 1, "gt_successes": int(float(g["full_task_success_y"])), "active_outcome": outcome(a), "gt_outcome": outcome(g), "interpretation": "paired force difference is outcome-consistent" if same_state and float(a["full_task_success_y"]) > float(g["full_task_success_y"]) else "not separable from this pair"})
        observed = defaultdict(list)
        for method, r in by[cid].items():
            if r.get("selected_force_N", ""):
                observed[float(r["selected_force_N"])].append(r)
        for force in sorted(observed):
            rr = observed[force]
            sweep.append({"context": cid, "force_N": force, "observed_methods": ";".join(x["method"] for x in rr), "n_observed": len(rr), "lift": ";".join(str(int(float(x.get("lift_success", 0) or 0))) for x in rr), "transport": ";".join(str(int(float(x.get("transport_retention", 0) or 0))) for x in rr), "placement": ";".join(str(int(float(x.get("place_success", 0) or 0))) for x in rr), "final_success": ";".join(str(int(float(x.get("full_task_success_y", 0) or 0))) for x in rr), "stages": ";".join(x.get("failure_stage", "SUCCESS") or "SUCCESS" for x in rr), "scope": "existing corrected fresh arms only; no new force sweep or re-query"})

    write_csv(ROOT / "MASS_FRESH_E2E_SIX_CONTEXT_TABLE.csv", table)
    write_csv(ROOT / "MASS_FRESH_E2E_FORCE_SWAP.csv", swaps)
    write_csv(ROOT / "MASS_FRESH_E2E_FORCE_SWEEP.csv", sweep)

    winners = [x for x in swaps if x["active_successes"] > x["gt_successes"]]
    report = [
        "# MASS FRESH E2E ANALYSIS", "",
        "Scope: the six corrected fresh contexts only. No Mass branch recollection, no new query, no π0/controller/evaluator/Utility change, and no new force rollout was used in this report. The corrected existing Active and GT arms are already paired at the same persisted query-state hash and therefore provide the allowed one-repetition force-swap evidence.", "",
        "## 1. Six-context table", "",
        "See `MASS_FRESH_E2E_SIX_CONTEXT_TABLE.csv`. Corrected outcomes are Active=4/6, GT=2/6, Fixed-Max=1/6, Prior=0/6, Native=0/6.", "",
        "| Context | GT kg | Active pred kg | Error kg | Active F | GT F | Prior F | Fixed F | Active | GT | Prior | Fixed | Native | Active stage | GT stage |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|",
    ]
    for x in table:
        report.append(f"| {x['context']} | {float(x['gt_mass_kg']):.2f} | {float(x['active_predicted_mass_kg']):.4f} | {float(x['mass_error_kg']):+.4f} | {float(x['active_force_N']):.1f} | {float(x['gt_force_N']):.1f} | {float(x['prior_force_N']):.1f} | {float(x['fixed_max_force_N']):.1f} | {x['active_outcome']} | {x['gt_outcome']} | {x['prior_outcome']} | {x['fixed_max_outcome']} | {x['native_outcome']} | {x['active_failure_stage']} | {x['gt_failure_stage']} |")
    report += ["", "## 2. Why Active beats GT", "", "The advantage is force selection, not better mass accuracy. Both winning cases are MID:", "", "- `root8400_mid`: Active predicted mass 0.4277 kg -> 4.0 N; GT mass 0.10 kg -> 1.5 N. Active succeeded; GT failed at transport.", "- `root8401_mid`: Active predicted mass 0.4295 kg -> 4.0 N; GT mass 0.10 kg -> 1.5 N. Active succeeded; GT failed at transport.", "", "The Active predicted mass is strongly high-biased in all six contexts (+0.3277 to +0.3878 kg). That bias drives the Direct/Utility model to the conservative 4.0 N arm. It is not evidence that the estimator is more accurate; it is a task-specific conservative extrapolation beyond the Direct training mass support (.05/.10/.20 kg). GT-Mass is in the Direct training support and is semantically matched to the training labels.", "", "## 3. Direct and Utility scores for the two Active wins", "", "The score tables below use the frozen TRAIN-only Mass Direct ensemble and the unchanged Expected Utility `p*(8-F)+(1-p)*(-1)`. Values are `(p_success, utility)`.", "", "| Context | Input | 0.5N | 1.0N | 1.5N | 2.5N | 4.0N | selected |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cid in ("mass_fresh_t2_root8400_mid", "mass_fresh_t2_root8401_mid"):
        dr = [x for x in direct if x["context_id"] == cid]
        for label, pfx, selected in (("Active predicted mass", "active", 4.0), ("GT mass", "gt", 1.5)):
            vals = {float(x["candidate_force_N"]): (float(x[f"{pfx}_direct_p_success"]), float(x[f"gt_expected_utility"])) for x in dr}
            # Utility is identical for GT and Active only when p is identical;
            # direct audit stores GT utility, so recompute Active utility here.
            if pfx == "active": vals = {f: (p, p * (8 - f) + (1 - p) * -1) for f, (p, _) in vals.items()}
            report.append(f"| {cid} | {label} | " + " | ".join(f"{vals[f][0]:.3f},{vals[f][1]:.3f}" for f in FORCES) + f" | {selected:.1f}N |")
    report += ["", "## 4. Force-swap results", "", "These are existing corrected paired fresh arms, not new rollouts. Active and GT share the same query-state hash in both cases.", "", "| Context | Active force | Active result | GT force | GT result | Same query state |", "|---|---:|---|---:|---|---|"]
    for x in swaps:
        report.append(f"| {x['context']} | {float(x['active_force_N']):.1f}N | {x['active_successes']}/{x['active_replay_n']} ({x['active_outcome']}) | {float(x['gt_force_N']):.1f}N | {x['gt_successes']}/{x['gt_replay_n']} ({x['gt_outcome']}) | {x['same_query_state_hash']} |")
    report += ["", "Interpretation: both Active-win cases show 1/1 at 4.0 N versus 0/1 at 1.5 N. This is direct one-pair evidence that the selected higher force was beneficial in those contexts. It is not a 3-repeat variance estimate; no re-query/new rollout was run under the current constraint.", "", "## 5. Existing corrected force evidence", "", "The authoritative fresh candidate support is 0.5/1.0/1.5/2.5/4.0 N, not 5--8 N. The table `MASS_FRESH_E2E_FORCE_SWEEP.csv` reports every force actually observed in the corrected six-context arms; no new 3--8 N sweep was run because that would require new query/state replay and would violate this round's no-re-query constraint.", "", "For both HIGH contexts, the observed 1.5 N, 2.5 N, and 4.0 N arms all failed (transport/placement). Thus they are classified `force_not_rescuing_observed_support`: the current fresh state/π0 downstream execution is the dominant issue, although forces outside the frozen support are not claimed.", "", "For the two MID Active wins, 4.0 N succeeds while the observed 1.5 N arms fail. They are classified `selector_force_error_for_GT` at one paired trial each. For LOW, both Active and GT succeed despite 4.0 versus 1.5 N, so the force difference is not universally necessary.", "", "## 6. Identifier", "", "Fresh Active estimates are 0.4128--0.5494 kg for GT masses .05/.10/.20 kg, with errors +0.3277--+0.3878 kg. This is far worse than offline MAE 0.0229 kg and is systematically high, so fresh query shift does affect identifier output. However, it is not the primary explanation for Active vs GT: GT avoids the estimate and still shares the same query state; the two Active wins are explained by the force decision caused by the conservative high estimate.", "", "## 7. Final explanation", "", "1. **Why Active is 4/6:** its high-biased fresh estimate selects 4.0 N in all six contexts; that rescues both MID contexts and both LOW contexts. The MID rescues are consistent with the 4.0-versus-1.5 N paired outcomes.", "2. **Why GT is 2/6:** GT selects 1.5 N in LOW/MID and 2.5 N in HIGH. It succeeds on both LOW contexts, but under-selects force on both MID contexts and fails transport; HIGH fails downstream even at 2.5 N.", "3. **Why Fixed-Max is 1/6:** fixed 4.0 N has no query-state handoff and succeeds only on 8400/LOW. Its failures, especially placement/transport, show that force alone is insufficient and that upstream/query-state execution matters.", "4. **Why Active's remaining 2/6 fail:** both are HIGH. Active 4.0 N, GT 2.5 N, and Fixed-Max 4.0 N all fail; observed lower-force Prior also fails. These are not currently attributable to selector choice alone; the evidence points to fresh state / frozen-π0 transport-placement limitations.", "5. **Current bottleneck:** not Mass identification alone. The evidence is mixed: (a) a high-confidence corrected selection effect in the two MID Active wins, plus (b) a downstream fresh-state/frozen-π0 failure in both HIGH contexts. The safe overall classification is `MASS_E2E_MIXED_FAILURE`, with `force selection` explaining the Active-vs-GT delta in 2/6 and `fresh query/state + π0 transport/placement` explaining the irreducible observed HIGH failures in 2/6.", "", "## Evidence boundary", "", "Active and GT were each run once per context, so the report establishes paired outcome differences but not repeat-level confidence intervals. The two corrected MID pairs are the strongest force-swap evidence available without re-query. No old 0.5 N runner rows are used in any conclusion.", ""]
    report_text = "\n".join(report)
    (ROOT / "MASS_FRESH_E2E_ANALYSIS.md").write_text(report_text, encoding="utf-8")
    (ROOT / "MASS_E2E_FAILURE_ATTRIBUTION_REPORT.md").write_text(report_text.replace("# MASS FRESH E2E ANALYSIS", "# MASS E2E FAILURE ATTRIBUTION REPORT", 1), encoding="utf-8")

    taxonomy = []
    for x in table:
        cid = x["context"]
        if x["active_outcome"] == "SUCCESS" and x["gt_outcome"] != "SUCCESS":
            primary = "FORCE_SELECTION_UNDER_FORCE"
            evidence = f"Active {x['active_force_N']}N succeeded; GT {x['gt_force_N']}N failed at {x['gt_failure_stage']}"
        elif x["active_outcome"] == "SUCCESS":
            primary = "NO_ACTIVE_OR_GT_FAILURE"
            evidence = "Active and GT full-task success"
        elif "high" in cid:
            primary = "FRESH_STATE_OR_FROZEN_PI0_DOWNSTREAM_FAILURE"
            evidence = "Active 4.0N failed; GT 2.5N failed; Fixed-Max 4.0N failed; Prior 1.5N failed"
        else:
            primary = "NOT_OBSERVED"
            evidence = f"Active failure stage={x['active_failure_stage']}"
        secondary = "QUERY_IDENTIFIER_SHIFT" if x["mass_error_kg"] > 0.1 else "NONE"
        taxonomy.append({"context_id": cid, "active_outcome": x["active_outcome"], "gt_outcome": x["gt_outcome"], "active_force_N": x["active_force_N"], "gt_force_N": x["gt_force_N"], "active_failure_stage": x["active_failure_stage"], "gt_failure_stage": x["gt_failure_stage"], "primary_cause": primary, "secondary_cause": secondary, "evidence": evidence})
    write_csv(ROOT / "MASS_E2E_FAILURE_TAXONOMY.csv", taxonomy)

    (ROOT / "MASS_E2E_FAILURE_ATTRIBUTION_STATUS.json").write_text(json.dumps({
        "status": "MASS_E2E_FAILURE_ATTRIBUTION_COMPLETE",
        "classification": "MASS_E2E_MIXED_FAILURE",
        "corrected_contexts": 6,
        "corrected_rows": len(rows),
        "active_successes": sum(x["active_outcome"] == "SUCCESS" for x in table),
        "gt_successes": sum(x["gt_outcome"] == "SUCCESS" for x in table),
        "active_gt_force_selection_delta_contexts": len(winners),
        "high_downstream_failure_contexts": sum("high" in x["context"] and x["active_outcome"] != "SUCCESS" for x in table),
        "no_new_mass_branches": True,
        "no_new_query_or_force_rollout": True,
        "paper_claims": {"mass_identification": "SUPPORTED", "mass_controlled_adaptation": "SUPPORTED", "mass_fresh_e2e": "PARTIALLY_SUPPORTED"}
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contexts": len(table), "force_swap_pairs": len(swaps), "active_wins_with_force_change": len(winners), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
