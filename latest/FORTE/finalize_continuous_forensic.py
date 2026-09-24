#!/usr/bin/env python3
"""Robustness checks, final report, and portable report payload for the forensic."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("/home/exouser/FORTE/continuous_coverage_stochasticity_20260830_233650")
BOOTSTRAP_SEED = 2026083033
BOOTSTRAPS = 10000
MODEL_MAE = 0.24491496104839813


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def context_bootstrap(values: pd.DataFrame, value_col: str) -> dict:
    by = values.groupby("context_id")[value_col].mean()
    arr = by.to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = np.asarray([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(BOOTSTRAPS)])
    return {"observed": float(arr.mean()), "ci95_low": float(np.quantile(boots, 0.025)),
            "ci95_high": float(np.quantile(boots, 0.975)), "contexts": len(arr),
            "bootstrap_samples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED}


def main() -> None:
    forensic = json.loads((OUT / "FORENSIC_CLASSIFICATION.json").read_text())
    dev = pd.read_csv(OUT / "DEV_REPEAT_STOCHASTICITY.csv")
    noise = pd.read_csv(OUT / "REPEAT_COUNT_NOISE_FLOOR.csv")
    train = pd.read_csv(OUT / "TRAIN_REPEAT_CELL_AUDIT.csv")
    coverage = pd.read_csv(OUT / "TRAIN_CONTINUOUS_FORCE_COVERAGE.csv")
    attr = pd.read_csv(OUT / "DEV_ERROR_ATTRIBUTION.csv")
    tax = pd.read_csv(OUT / "CURRENT_CONTINUOUS_FAILURE_TAXONOMY.csv")

    n2 = noise[(noise.scope == "CELL") & (noise.subsample_m == 2)][
        ["context_id", "force_N", "expected_MAE"]
    ].rename(columns={"expected_MAE": "m2_noise_MAE"})
    paired = attr.merge(n2, on=["context_id", "force_N"], validate="one_to_one")
    paired["model_minus_m2_MAE"] = paired.absolute_probability_error - paired.m2_noise_MAE
    robustness = {
        "model_probability_MAE_context_bootstrap": context_bootstrap(attr, "absolute_probability_error"),
        "m2_noise_MAE_context_bootstrap": context_bootstrap(paired, "m2_noise_MAE"),
        "model_minus_m2_MAE_context_bootstrap": context_bootstrap(paired, "model_minus_m2_MAE"),
        "dev_boundary_stochastic_fraction_context_bootstrap": context_bootstrap(dev, "boundary_stochastic"),
        "dev_bernoulli_variance_context_bootstrap": context_bootstrap(dev, "bernoulli_variance"),
        "interpretation": "Context bootstrap is a robustness diagnostic over nine frozen contexts; it does not turn p5 into exact physical truth.",
    }
    write_json(OUT / "FORENSIC_ROBUSTNESS.json", robustness)

    group_rows = []
    for dim in ["task", "friction_band"]:
        for key, g in dev.groupby(dim):
            group_rows.append({"section": "DEV_STOCHASTICITY", "group_dimension": dim, "group_value": key,
                               "cells": len(g), "boundary_stochastic_fraction": float(g.boundary_stochastic.mean()),
                               "mean_bernoulli_variance": float(g.bernoulli_variance.mean()),
                               "mean_entropy_bits": float(g.entropy_bits.mean())})
    for dim in ["task", "friction_band", "stratum_index"]:
        for key, g in train.groupby(dim):
            group_rows.append({"section": "TRAIN_REPEAT_QUALITY", "group_dimension": dim, "group_value": key,
                               "cells": len(g), "zero_of_two_fraction": float((g.success_count == 0).mean()),
                               "one_of_two_fraction": float((g.success_count == 1).mean()),
                               "two_of_two_fraction": float((g.success_count == 2).mean())})
    for dim in ["task", "friction_band"]:
        for key, g in attr.groupby(dim):
            group_rows.append({"section": "DEV_MODEL_ERROR", "group_dimension": dim, "group_value": key,
                               "cells": len(g), "mean_probability_MAE": float(g.absolute_probability_error.mean()),
                               "mean_dev_variance": float(g.dev_bernoulli_variance.mean()),
                               "mean_nearest_train_force_distance_N": float(g.distance_to_nearest_task_train_force_N.mean()),
                               "mean_train_density_within_0p10N": float(g.train_density_within_0p10N.mean())})
    pd.DataFrame(group_rows).to_csv(OUT / "FORENSIC_GROUP_SUMMARY.csv", index=False)

    # Extend the controlling classification with robustness without changing the frozen classification.
    forensic["robustness"] = robustness
    forensic["collection_decision"] = {
        "status": "STOP_BEFORE_COLLECTION",
        "reason": "MODEL_ERROR_DOMINATES is not eligible under the frozen targeted-collection rule",
        "potential_manifest_cells_not_executed": int(forensic["target_cells_if_collection_eligible"]),
        "potential_branches_not_executed": int(forensic["expected_new_branches_if_collection_eligible"]),
    }
    write_json(OUT / "FORENSIC_CLASSIFICATION.json", forensic)

    counts = {f"{k}/5": int((dev.success_count == k).sum()) for k in range(6)}
    noise_agg = noise[noise.scope == "ALL_27_CELLS"].set_index("subsample_m")
    train_counts = {f"{k}/2": int((train.success_count == k).sum()) for k in range(3)}
    task_range = coverage.groupby("task").requested_force_N.agg(["min", "max"])
    task_gap = coverage.groupby("task").apply(
        lambda g: float(np.diff(np.sort(g.requested_force_N.unique())).max()), include_groups=False
    )
    taxonomy = Counter(tax.failure_type)
    delta_ci = robustness["model_minus_m2_MAE_context_bootstrap"]

    report = f"""# STATUS

COMPLETE — offline forensic ended at the preregistered STOP_BEFORE_COLLECTION gate. No new TRAIN branches, retraining, repeat-enriched DEV evaluation, Probe, TEST, or fresh E2E were run.

# SINGLE SCIENTIFIC GOAL

Determine whether the failed continuous reliability gate is primarily explained by two-repeat label noise, continuous-force coverage, finite-repeat physical stochasticity, or residual Feasibility-only model error.

# CURRENT BACKEND

Feasibility-only is the selected backend. Joint/world-model development is not reopened.

# REAL DEV STOCHASTICITY

Nine of 27 repeated DEV cells (33.3%) are boundary-stochastic (1/5 through 4/5); 18/27 (66.7%) are deterministic-like (0/5 or 5/5). Mean empirical Bernoulli variance is {forensic['dev_mean_bernoulli_variance']:.3f}, below the frozen 0.10 criterion. With only five repeats, these values quantify finite-repeat evidence and do not establish irreducible physical randomness.

Task 1 contains the most mixed cells (6/12, 50%); task 5 has 0/6 mixed cells. MID-friction cells are more mixed (4/6, 66.7%) than LOW-friction cells (5/21, 23.8%), but probability MAE is nearly identical across MID and LOW friction (0.248 vs 0.244), arguing against observed stochasticity as the main error driver.

# HOW MANY DEV CELLS ARE 0/5 ... 5/5?

| Outcome | Cells | Fraction |
|---|---:|---:|
""" + "\n".join(f"| {label} | {count} | {count/27:.3f} |" for label, count in counts.items()) + f"""

# TWO-REPEAT SUPERVISION NOISE FLOOR

Exact enumeration over every size-2 subsample gives expected MAE **{noise_agg.loc[2, 'expected_MAE']:.3f}** against observed p5, RMSE {noise_agg.loc[2, 'RMSE']:.3f}, and p>=0.8 classification mismatch probability {noise_agg.loc[2, 'reliability_misclassification_probability']:.3f}. Current model probability MAE is {MODEL_MAE:.3f}, or {MODEL_MAE/noise_agg.loc[2, 'expected_MAE']:.2f}x the two-repeat observation floor.

The context-bootstrap model-minus-noise MAE gap is {delta_ci['observed']:.3f} (95% CI {delta_ci['ci95_low']:.3f} to {delta_ci['ci95_high']:.3f}). Thus two-repeat label noise is materially too small to explain the current model error.

| Subsample repeats | Exact expected MAE vs p5 | RMSE | Reliability mismatch |
|---:|---:|---:|---:|
""" + "\n".join(f"| {m} | {noise_agg.loc[m, 'expected_MAE']:.3f} | {noise_agg.loc[m, 'RMSE']:.3f} | {noise_agg.loc[m, 'reliability_misclassification_probability']:.3f} |" for m in [1,2,3,4]) + f"""

# TRAIN TWO-REPEAT QUALITY

The 360 TRAIN cells contain {train_counts['0/2']} 0/2 cells ({train_counts['0/2']/360:.1%}), {train_counts['1/2']} 1/2 cells ({train_counts['1/2']/360:.1%}), and {train_counts['2/2']} 2/2 cells ({train_counts['2/2']/360:.1%}). Only 4.2% are ambiguous 1/2 cells, below the frozen 10% materiality threshold. Ambiguity is not broadly distributed enough for repeat noise to dominate.

# CONTINUOUS FORCE COVERAGE

All task supports are covered by 90 continuous TRAIN forces per task. Observed sampled ranges are task 0 [{task_range.loc[0,'min']:.3f}, {task_range.loc[0,'max']:.3f}]N, task 1 [{task_range.loc[1,'min']:.3f}, {task_range.loc[1,'max']:.3f}]N, task 5 [{task_range.loc[5,'min']:.3f}, {task_range.loc[5,'max']:.3f}]N, and task 6 [{task_range.loc[6,'min']:.3f}, {task_range.loc[6,'max']:.3f}]N. Maximum pooled within-task force gaps are {', '.join(f'task {int(t)} {task_gap.loc[t]:.3f}N' for t in task_gap.index)}.

DEV error does not rise with sparse force coverage: Spearman(error, nearest-force distance) is {forensic['coverage_error_spearman']:.3f}; the sparse-distance quartile has {abs(forensic['coverage_sparse_minus_dense_error']):.3f} lower MAE than the dense quartile. This fails both frozen coverage criteria.

# CURRENT MODEL ERROR TAXONOMY

Among eight valid real frontiers: {taxonomy.get('TYPE_D_OVER_FORCE',0)} are ordered but over-force; {taxonomy.get('TYPE_B_NO_FINITE_DECISION',0)} are ordered but never cross p=0.80; there are no finite under-force decisions and no ordering failures. The inherited under-force rate counts the two missing decisions as safety failures. This pattern is predominantly underconfidence/miscalibration of the learned probability scale, not a failure to order force correctly.

# WHAT DOMINATES THE ERROR?

**MODEL ERROR.** Repeat noise is only 32.7% of model MAE, TRAIN ambiguity is 4.2%, observed mixed DEV cells have mean variance below threshold, and coverage diagnostics do not worsen in sparse regions. The evidence does not justify calling the five-repeat DEV target exact or the process deterministic; it shows only that these data limitations are insufficient to explain the much larger current error.

# TARGETED TRAIN TOP-UP

NOT EXECUTED. The frozen TRAIN-only selection rule would have identified {forensic['target_cells_if_collection_eligible']} cells ({forensic['expected_new_branches_if_collection_eligible']} branches), but `MODEL_ERROR_DOMINATES` is explicitly ineligible for collection. Executing them would violate the preregistered causal test.

# REPEAT-ENRICHED MODEL

NOT_REACHED. No architecture, optimizer, calibration, checkpoint, or normalization was changed.

# OLD VS NEW GT CONTINUOUS RESULT

NOT_REACHED. The authoritative old Feasibility-only result remains probability MAE 0.245, frontier MAE 0.325N, safety-failure/under-force 2/8, finite decisions 6/8, and monotonicity 9/9.

# GT GATE

NOT_REACHED for a repeat-enriched model. The current authoritative model remains failed; no new DEV evaluation was performed.

# PROBE VS STRICT NO-PROBE

NOT_REACHED because no repeat-enriched backend reached the GT gate.

# QUANTIZATION UNMASKING

NOT_REACHED.

# PRIMARY_CLASSIFICATION

MODEL_ERROR_DOMINATES

# SECONDARY_PROBE_CLASSIFICATION

NOT_REACHED

# WHAT IS NOW PROVEN

Within the current object/task distribution, the current probability error is much larger than the exact empirical two-repeat subsampling noise floor, and neither pooled force coverage nor observed five-repeat stochasticity passes its preregistered materiality criterion. More repeats at the existing forces are therefore not scientifically justified as the next intervention.

# WHAT IS STILL NOT PROVEN

- no claim that five-repeat empirical probabilities are exact or physical stochasticity is irreducible
- no cross-object claim
- no unseen-task claim
- no original TEST
- no fresh E2E
- no when-to-probe policy
- no conclusion about Probe continuous control because GT reliability was not reached

# METHOD IMPLICATION

The candidate pipeline is not yet validated. The next bottleneck is the Feasibility-only context-to-probability mapping—especially probability scale/underconfidence—not evidence quality from two repeats.

# NEXT_METHOD

Audit which context variables or state features are missing, and test probability-scale/representation diagnostics without collecting more force outcomes. Keep reliability-aware force selection conservative; uncertainty lower bounds are a secondary safeguard, not a substitute for fixing the model relation.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    outcome_rows = [{"outcome_category": f"{k}/5", "success_count": k, "cell_count": counts[f"{k}/5"],
                     "cell_fraction": counts[f"{k}/5"] / 27,
                     "stochasticity_class": "deterministic-like" if k in {0,5} else "boundary-stochastic"}
                    for k in range(6)]
    evidence_rows = [
        {"diagnostic": "Current model probability MAE", "value": MODEL_MAE, "threshold": None, "result": "Observed error"},
        {"diagnostic": "Exact m=2 observation MAE vs p5", "value": float(noise_agg.loc[2, "expected_MAE"]), "threshold": 0.75 * MODEL_MAE, "result": "Below repeat-noise materiality"},
        {"diagnostic": "TRAIN ambiguous 1/2 fraction", "value": forensic["train_ambiguous_1of2_fraction"], "threshold": 0.10, "result": "Below repeat-noise materiality"},
        {"diagnostic": "DEV boundary-stochastic fraction", "value": forensic["dev_boundary_stochastic_fraction"], "threshold": 1/3, "result": "At fraction threshold"},
        {"diagnostic": "DEV mean Bernoulli variance", "value": forensic["dev_mean_bernoulli_variance"], "threshold": 0.10, "result": "Below stochasticity materiality"},
        {"diagnostic": "Coverage error Spearman", "value": forensic["coverage_error_spearman"], "threshold": 0.35, "result": "Wrong sign / below coverage threshold"},
    ]
    generated = datetime.now(timezone.utc).isoformat()
    sources = [
        {"id": "dev_stochasticity", "label": "Repeated DEV stochasticity", "path": "DEV_REPEAT_STOCHASTICITY.csv",
         "query": {"engine": "duckdb", "language": "sql", "description": "Counts observed success outcomes across the 27 repeated DEV cells.",
                   "tables_used": ["DEV_REPEAT_STOCHASTICITY.csv"],
                   "sql": "SELECT outcome_category, success_count, COUNT(*) AS cell_count, COUNT(*) / 27.0 AS cell_fraction, CASE WHEN success_count IN (0,5) THEN 'deterministic-like' ELSE 'boundary-stochastic' END AS stochasticity_class FROM read_csv_auto('DEV_REPEAT_STOCHASTICITY.csv') GROUP BY outcome_category, success_count ORDER BY success_count"}},
        {"id": "noise_floor", "label": "Exact repeat-count noise floor", "path": "REPEAT_COUNT_NOISE_FLOOR.csv",
         "query": {"engine": "duckdb", "language": "sql", "description": "Loads exact all-combination repeat-subsampling metrics.",
                   "tables_used": ["REPEAT_COUNT_NOISE_FLOOR.csv"],
                   "filters": ["scope = ALL_27_CELLS"],
                   "sql": "SELECT subsample_m, expected_MAE, RMSE, reliability_misclassification_probability FROM read_csv_auto('REPEAT_COUNT_NOISE_FLOOR.csv') WHERE scope = 'ALL_27_CELLS' ORDER BY subsample_m"}},
        {"id": "forensic_classification", "label": "Frozen forensic classification", "path": "FORENSIC_CLASSIFICATION.json",
         "query": {"engine": "duckdb", "language": "sql", "description": "Loads the preregistered forensic criteria and resulting classification.",
                   "tables_used": ["FORENSIC_CLASSIFICATION.json"],
                   "sql": "SELECT * FROM read_json_auto('FORENSIC_CLASSIFICATION.json')"}},
        {"id": "failure_taxonomy", "label": "Current continuous failure taxonomy", "path": "CURRENT_CONTINUOUS_FAILURE_TAXONOMY.csv",
         "query": {"engine": "duckdb", "language": "sql", "description": "Counts failure taxonomy categories across eight valid real-frontier contexts.",
                   "tables_used": ["CURRENT_CONTINUOUS_FAILURE_TAXONOMY.csv"],
                   "sql": "SELECT failure_type, COUNT(*) AS contexts FROM read_csv_auto('CURRENT_CONTINUOUS_FAILURE_TAXONOMY.csv') GROUP BY failure_type ORDER BY contexts DESC"}},
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": "Continuous Feasibility Reliability Forensic", "generatedAt": generated,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Continuous Feasibility Reliability Forensic"},
                {"id": "technical_summary", "type": "markdown", "body": "## Technical Summary\n\n**MODEL_ERROR_DOMINATES.** Exact two-repeat observation MAE is 0.080 versus 0.245 model MAE. Only 4.2% of TRAIN cells are ambiguous 1/2, and neither coverage nor observed stochasticity meets the frozen materiality criteria. Targeted top-up, retraining, and Probe were therefore not run."},
                {"id": "distribution_text", "type": "markdown", "body": "## Repeated DEV Outcomes Are Mostly Deterministic-Like\n\nEighteen of 27 cells are 0/5 or 5/5; nine are mixed. The chart shows the full observed success-count distribution. Five repeats still provide a finite-sample reference, not exact physical truth.", "sourceId": "dev_stochasticity"},
                {"id": "distribution_chart_block", "type": "chart", "chartId": "outcome_distribution"},
                {"id": "noise_text", "type": "markdown", "body": "## Two Repeats Cannot Explain the Model Gap\n\nExact enumeration gives expected m=2 MAE 0.080 against p5. The model error is 3.06 times larger, and the context-bootstrap model-minus-noise gap remains positive.", "sourceId": "noise_floor"},
                {"id": "diagnostic_table_block", "type": "table", "tableId": "diagnostic_evidence"},
                {"id": "taxonomy_text", "type": "markdown", "body": "## The Failure Pattern Is Underconfidence, Not Wrong Force Ordering\n\nSix valid-frontier contexts over-force and two never cross p=0.80. All eight preserve anchor-force ordering; there are no finite under-force or ordering failures.", "sourceId": "failure_taxonomy"},
                {"id": "limits", "type": "markdown", "body": "## Limits and Robustness\n\nThe conclusion is diagnostic, not causal proof of a missing feature. The DEV reference has nine contexts and five repeats per cell. It does not establish irreducible stochasticity or exact probabilities."},
                {"id": "next", "type": "markdown", "body": "## Recommended Next Step\n\nAudit missing context/state variables and the probability-scale representation before collecting more force outcomes. The frozen rule prohibits repeat top-up for MODEL_ERROR_DOMINATES; Probe remains NOT_REACHED."},
                {"id": "questions", "type": "markdown", "body": "## Further Questions\n\nWhich pre-probe state variables distinguish task 6 and the no-finite-decision contexts? Does the current BCE model collapse context effects into force-only confidence? Can reliability lower bounds protect planning while representation diagnostics proceed?"},
            ],
            "charts": [{"id": "outcome_distribution", "title": "Observed outcomes across repeated DEV cells",
                        "description": "27 context-force cells, five repeats per cell", "type": "bar", "dataset": "dev_outcome_distribution",
                        "sourceId": "dev_stochasticity", "encodings": {"x": {"field": "outcome_category", "type": "nominal"},
                        "y": {"field": "cell_count", "type": "quantitative"}}}],
            "tables": [{"id": "diagnostic_evidence", "title": "Frozen diagnostic evidence",
                        "description": "Observed values against preregistered materiality thresholds", "dataset": "diagnostic_evidence",
                        "sourceId": "forensic_classification", "columns": [
                            {"field": "diagnostic", "label": "Diagnostic"}, {"field": "value", "label": "Observed", "format": "number"},
                            {"field": "threshold", "label": "Threshold", "format": "number"}, {"field": "result", "label": "Interpretation"}],
                        "defaultSort": {"field": "diagnostic", "direction": "asc"}}],
            "sources": sources,
        },
        "snapshot": {"version": 1, "generatedAt": generated, "status": "ready",
                     "datasets": {"dev_outcome_distribution": outcome_rows, "diagnostic_evidence": evidence_rows}},
        "sources": sources,
    }
    write_json(OUT / "artifact.json", artifact)
    source_notes = {
        "audience": "technical",
        "delivery_mode": "html",
        "technical_report_required_structure_mapping": {
            "title": "artifact title block", "technical_summary": "technical_summary", "key_findings_with_visual_evidence": "distribution_text + distribution_chart_block + diagnostic table",
            "scope_data_metric_definitions": "FINAL_REPORT CURRENT BACKEND / REAL DEV STOCHASTICITY / TRAIN TWO-REPEAT QUALITY",
            "methodology": "protocol + exact enumeration CSV", "limitations_uncertainty_robustness": "limits + FORENSIC_ROBUSTNESS.json",
            "recommended_next_steps": "next", "further_questions": "questions"},
        "chart_map": [{"section": "Repeated DEV outcomes", "question": "How much of DEV is intrinsically mixed in the observed five repeats?",
                       "family": "comparison", "type": "bar", "fields": ["outcome_category", "cell_count"],
                       "takeaway": "18/27 deterministic-like; 9/27 mixed", "palette_policy": "single-root preferred",
                       "delivery": "report.html native artifact chart"}],
        "omitted_visuals": ["Repeat-noise values use a four-row exact table because exact lookup is the analytical point.",
                            "Coverage attribution uses CSV/table evidence because 27 observations and tied nearest distances make a scatter easy to overread."],
    }
    write_json(OUT / "REPORT_SOURCE_NOTES.json", source_notes)
    print(json.dumps({"status": "FINALIZED", "classification": forensic["FORENSIC_CLASSIFICATION"],
                      "collection": forensic["collection_decision"]["status"], "report": str(OUT / "FINAL_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
