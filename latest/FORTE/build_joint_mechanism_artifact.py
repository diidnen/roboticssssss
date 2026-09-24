#!/usr/bin/env python3
"""Build the canonical portable technical-report manifest from final evidence."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "joint_mechanism_20260831"


def rounded_records(frame: pd.DataFrame, digits: int = 4) -> list[dict]:
    rows = []
    for row in frame.to_dict("records"):
        clean = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[key] = None
            elif isinstance(value, float):
                clean[key] = round(value, digits)
            elif hasattr(value, "item"):
                clean[key] = value.item()
            else:
                clean[key] = value
        rows.append(clean)
    return rows


def main() -> None:
    delta_raw = pd.read_csv(OUT / "MECHANISM_DELTA_SUMMARY.csv")
    taskwise_raw = pd.read_csv(OUT / "TASKWISE_ROOT_HELDOUT_COMPARISON.csv")
    pooled_raw = pd.read_csv(OUT / "POOLED_JOINT_COMPARISON.csv")
    db = sqlite3.connect(":memory:")
    delta_raw.to_sql("mechanism_delta_summary", db, index=False)
    taskwise_raw.to_sql("taskwise_root_heldout", db, index=False)
    pooled_raw.to_sql("pooled_joint_comparison", db, index=False)
    headline_sql = """SELECT 'Case C' AS case_name,
      SUM(CASE WHEN single_JNV_minus_Base_NLL > 0 THEN 1 ELSE 0 END) AS single_nll_worse,
      SUM(CASE WHEN VisualJoint_minus_Full_NLL > 0 THEN 1 ELSE 0 END) AS visual_nll_worse,
      SUM(CASE WHEN pooled_JNV_minus_Base_NLL < 0 THEN 1 ELSE 0 END) AS pooled_nll_better,
      SUM(CASE WHEN pooled_JNV_minus_Base_under_force > 0 THEN 1 ELSE 0 END) AS pooled_under_force_worse
      FROM mechanism_delta_summary"""
    delta_sql = """SELECT 'task' || CAST(task AS INTEGER) AS task,
      'single-task CV' AS regime, single_JNV_minus_Base_NLL AS nll_delta
      FROM mechanism_delta_summary
      UNION ALL
      SELECT 'task' || CAST(task AS INTEGER), 'matched pooled CV', pooled_JNV_minus_Base_NLL
      FROM mechanism_delta_summary"""
    taskwise_sql = """SELECT task, model, NLL, probability_MAE, Brier, frontier_MAE_N,
      under_force_rate, context_monotonic_rate FROM taskwise_root_heldout
      WHERE model IN ('Base','Full','Visual Joint','Joint-NoVisual')"""
    pooled_sql = """SELECT task, model, NLL, probability_MAE, Brier, frontier_MAE_N,
      under_force_rate, context_monotonic_rate FROM pooled_joint_comparison
      WHERE evaluation_scope = 'MATCHED_POOLED_TRAIN_ROOT_HELDOUT_CV'"""
    headline = rounded_records(pd.read_sql_query(headline_sql, db), 0)
    chart_rows = rounded_records(pd.read_sql_query(delta_sql, db), 6)
    taskwise = pd.read_sql_query(taskwise_sql, db)
    pooled = pd.read_sql_query(pooled_sql, db)
    db.close()

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Joint mechanism audit: old frontier signal does not transfer reliably",
        "description": "Frozen taskwise and pooled evidence separating visual conditioning from the physics auxiliary.",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {"id": "headline_source", "label": "Frozen mechanism delta summary", "query": {"sql": headline_sql}},
            {"id": "delta_source", "label": "Frozen taskwise and pooled NLL deltas", "query": {"sql": delta_sql}},
            {"id": "taskwise_source", "label": "Frozen taskwise root-heldout comparison", "query": {"sql": taskwise_sql}},
            {"id": "pooled_source", "label": "Frozen pooled root-heldout comparison", "query": {"sql": pooled_sql}},
        ],
        "cards": [{
            "id": "headline_card", "dataset": "headline",
            "sourceId": "headline_source",
            "description": "Counts use four tasks: task0, task1, task5 and task6.",
            "metrics": [
                {"label": "Single-task JNV NLL worse", "field": "single_nll_worse", "format": "number"},
                {"label": "Visual Joint NLL worse vs Full", "field": "visual_nll_worse", "format": "number"},
                {"label": "Pooled JNV NLL better", "field": "pooled_nll_better", "format": "number"},
                {"label": "Pooled under-force worse", "field": "pooled_under_force_worse", "format": "number"},
            ],
        }],
        "charts": [{
            "id": "nll_delta_chart",
            "title": "Joint-NoVisual minus Base held-root NLL",
            "subtitle": "Positive values are worse; pooling produces only a task-dependent partial recovery.",
            "type": "bar", "dataset": "nll_delta",
            "sourceId": "delta_source",
            "encodings": {
                "x": {"field": "task", "type": "nominal", "label": "Task"},
                "y": {"field": "nll_delta", "type": "quantitative", "label": "NLL delta"},
                "color": {"field": "regime", "type": "nominal", "label": "Training regime"},
                "tooltip": [
                    {"field": "task", "type": "nominal"},
                    {"field": "regime", "type": "nominal"},
                    {"field": "nll_delta", "type": "quantitative"},
                ],
            },
            "xAxisTitle": "Task", "yAxisTitle": "JNV − Base NLL",
            "referenceLines": [{"value": 0, "label": "no difference"}],
        }],
        "tables": [
            {
                "id": "taskwise_table", "title": "Single-task retrospective held-root CV",
                "subtitle": "TRAIN roots only; diagnostic and ineligible for selection.",
                "dataset": "taskwise", "defaultSort": {"field": "task", "direction": "asc"},
                "sourceId": "taskwise_source",
                "columns": [
                    {"field": "task", "label": "Task", "type": "number"},
                    {"field": "model", "label": "Model", "type": "text"},
                    {"field": "NLL", "label": "NLL", "type": "number"},
                    {"field": "probability_MAE", "label": "Probability MAE", "type": "number"},
                    {"field": "Brier", "label": "Brier", "type": "number"},
                    {"field": "frontier_MAE_N", "label": "Frontier MAE", "type": "number", "unit": "N"},
                    {"field": "under_force_rate", "label": "Under-force", "type": "percent"},
                    {"field": "context_monotonic_rate", "label": "Monotonic contexts", "type": "percent"},
                ],
            },
            {
                "id": "pooled_table", "title": "Matched pooled retrospective held-root CV",
                "subtitle": "Four tasks pooled for training and reported separately within task.",
                "dataset": "pooled", "defaultSort": {"field": "task", "direction": "asc"},
                "sourceId": "pooled_source",
                "columns": [
                    {"field": "task", "label": "Task", "type": "number"},
                    {"field": "model", "label": "Model", "type": "text"},
                    {"field": "NLL", "label": "NLL", "type": "number"},
                    {"field": "probability_MAE", "label": "Probability MAE", "type": "number"},
                    {"field": "Brier", "label": "Brier", "type": "number"},
                    {"field": "frontier_MAE_N", "label": "Frontier MAE", "type": "number", "unit": "N"},
                    {"field": "under_force_rate", "label": "Under-force", "type": "percent"},
                    {"field": "context_monotonic_rate", "label": "Monotonic contexts", "type": "percent"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "layout": "full",
             "body": "# Joint mechanism audit: old frontier signal does not transfer reliably\n\nFrozen same-task, held-physical-root diagnostic; no DEV selection or hyperparameter tuning."},
            {"id": "summary", "type": "markdown", "layout": "full",
             "body": "## Technical summary\n\n**Case C — `OLD_JOINT_GAIN_DOES_NOT_TRANSFER_TO_CURRENT_DATA_REGIME`.** Joint-NoVisual fits TRAIN strongly but worsens held-root NLL on all four single tasks. Visual Joint is also worse than Full on all four tasks, so visual conditioning is an additional failure path, not the sole sufficient explanation. Pooling 72 contexts helps root-CV NLL on only task5/task6; on prospective pooled DEV, JNV remains worse on task0/task1/task5 and improves only the single-root task6 result. It does not rescue controller-quality Joint behavior."},
            {"id": "headline", "type": "metric-strip", "cardIds": ["headline_card"], "layout": "full"},
            {"id": "finding", "type": "markdown", "layout": "full",
             "body": "## The no-visual auxiliary reproduces the key generalization failure\n\nThe decisive comparison is Joint-NoVisual minus Base because their inputs are identical. Positive NLL deltas on all four single tasks implicate the physics auxiliary or its optimization interaction even without RGB/PCA. Probability MAE alone can improve while NLL, frontier placement, safety, or monotonicity worsen, so it is not used as the deciding metric."},
            {"id": "nll_chart", "type": "chart", "chartId": "nll_delta_chart", "layout": "full"},
            {"id": "taskwise_heading", "type": "markdown", "layout": "full",
             "body": "## Single-task held-root evidence\n\nEach fold holds two roots and six independent contexts. These previously inspected TRAIN splits are retrospective mechanism diagnostics only and cannot select models or tune losses."},
            {"id": "taskwise", "type": "table", "tableId": "taskwise_table", "layout": "full"},
            {"id": "pooled_heading", "type": "markdown", "layout": "full",
             "body": "## More pooled contexts provide partial, not reliable, recovery\n\nMatched pooled training uses 72 contexts and 24 roots; each CV fold trains on 48 contexts and 16 roots. Root-CV recovery is limited to task5/task6 NLL, while prospective pooled DEV is worse for task0/task1/task5 and better only for task6's one DEV root. Results remain taskwise within the same task/object distribution and are not unseen-task generalization."},
            {"id": "pooled", "type": "table", "tableId": "pooled_table", "layout": "full"},
            {"id": "old", "type": "markdown", "layout": "full",
             "body": "## Why the old Joint looked better\n\nThe old no-visual pooled Joint improved frontier MAE from 0.325 N to 0.125 N on eight valid frontier contexts. It simultaneously worsened Brier, NLL, under-force, and monotonicity. The old regime also had 72 contexts, 24 roots, and 288 extra coarse BCE branches. The shared authoritative auxiliary architecture is aligned with current taskwise Joint-NoVisual, but dataset, supervision, normalization, force/evaluator population, and the small old DEV remain confounds."},
            {"id": "limits", "type": "markdown", "layout": "full",
             "body": "## Limits and robustness\n\nTask1 retains a material reconstructed-label caveat, while task5/task6 direct labels show the pattern is not reducible to task1 recovery. Task6 has one prospective DEV root and cannot alone establish multi-context generalization. The original pooled Visual pipeline is implementation-confounded and cannot isolate diversity. Therefore `DATA_TOO_SMALL` is not established; small context count remains plausible but unproven."},
            {"id": "decision", "type": "markdown", "layout": "full",
             "body": "## Decision and next step\n\nDo not present Joint as a validated controller novelty. Remove or simplify it from the controller-facing claim, then audit old/current distribution, supervision, evaluator, and auxiliary overconfidence before collecting many more roots. A later preregistered context-count learning curve is appropriate only after those confounds are resolved."},
            {"id": "questions", "type": "markdown", "layout": "full",
             "body": "## Further questions\n\nWhich old coarse outcomes drive the frontier shift? Does matched old/current force sampling preserve the signal? Does calibration-aware evaluation show the auxiliary merely sharpens response curves? These are follow-up audits, not permission to tune on the current held-out roots."},
        ],
    }
    artifact = {
        "surface": "report", "manifest": manifest,
        "snapshot": {
            "version": 1, "generatedAt": datetime.now(timezone.utc).isoformat(), "status": "ready",
            "datasets": {
                "headline": headline,
                "nll_delta": chart_rows,
                "taskwise": rounded_records(taskwise),
                "pooled": rounded_records(pooled),
            },
        },
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ARTIFACT_WRITTEN", "datasets": {k: len(v) for k, v in artifact["snapshot"]["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
