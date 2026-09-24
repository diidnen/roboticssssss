#!/usr/bin/env python3
"""Build the canonical portable report input for the Task 0-only curve."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import root_scaling_finalize as finalizer


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831/task0_diagnostic"
LC_PATH = OUT / "ROOT_SCALING_LEARNING_CURVE.csv"
PER_ROOT_PATH = OUT / "ROOT_SCALING_PER_ROOT_METRICS.csv"
ARTIFACT_PATH = OUT / "TASK0_ROOT_SCALING_REPORT_ARTIFACT.json"
NOTES_PATH = OUT / "TASK0_ROOT_SCALING_REPORT_NOTES.json"
LEVELS = [6, 15, 30, 50]
DIRECT = "Full Visual"
WM = "Visual Joint"

EVIDENCE_SQL = """WITH learning_curve AS (
    SELECT
        'learning_curve' AS grain,
        CAST(N_root AS TEXT) AS evidence_key,
        model,
        aggregation,
        TEST_frontier_MAE_N AS frontier_metric,
        TEST_probability_MAE AS probability_metric,
        TEST_NLL AS nll_metric,
        TEST_under_force_rate AS under_force_metric,
        TEST_dense_monotonic_root_fraction AS monotonic_metric,
        TRAIN_to_TEST_NLL_gap AS train_test_gap_metric
    FROM root_scaling_learning_curve
    WHERE task = 0
      AND model IN ('Full Visual', 'Visual Joint')
      AND aggregation IN ('SEED_MEAN', 'ENSEMBLE', 'SEED_0', 'SEED_1', 'SEED_2')
), paired_s50 AS (
    SELECT
        'per_root_s50' AS grain,
        context_id AS evidence_key,
        model,
        aggregation,
        absolute_frontier_error_N AS frontier_metric,
        probability_MAE AS probability_metric,
        NLL AS nll_metric,
        CAST(under_force AS REAL) AS under_force_metric,
        CAST(dense_monotonic AS REAL) AS monotonic_metric,
        NULL AS train_test_gap_metric
    FROM root_scaling_per_root_metrics
    WHERE task = 0
      AND N_root = 50
      AND model IN ('Full Visual', 'Visual Joint')
      AND aggregation = 'ENSEMBLE'
)
SELECT * FROM learning_curve
UNION ALL
SELECT * FROM paired_s50
ORDER BY grain, evidence_key, model, aggregation"""


def number(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def metric_row(lc: pd.DataFrame, roots: int, model: str, aggregation: str) -> pd.Series:
    q = lc[(lc.task == 0) & (lc.N_root == roots) &
           (lc.model == model) & (lc.aggregation == aggregation)]
    if len(q) != 1:
        raise RuntimeError(f"metric row mismatch: roots={roots} model={model} aggregation={aggregation}")
    return q.iloc[0]


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    lc = pd.read_csv(LC_PATH)
    per_root = pd.read_csv(PER_ROOT_PATH)
    with sqlite3.connect(":memory:") as connection:
        lc.to_sql("root_scaling_learning_curve", connection, index=False)
        per_root.to_sql("root_scaling_per_root_metrics", connection, index=False)
        reviewed_evidence = pd.read_sql_query(EVIDENCE_SQL, connection)
    if len(reviewed_evidence) != 60:
        raise RuntimeError(f"reviewed evidence query returned {len(reviewed_evidence)} rows, expected 60")
    gates = finalizer.task_gates(lc, per_root, 0)

    primary_rows = []
    gap_rows = []
    for roots in LEVELS:
        for model, label in ((DIRECT, "Direct"), (WM, "WM / Joint")):
            row = metric_row(lc, roots, model, "SEED_MEAN")
            primary_rows.append({
                "roots": roots,
                "method": label,
                "frontier_mae_n": number(row.TEST_frontier_MAE_N),
                "probability_mae": number(row.TEST_probability_MAE),
                "nll": number(row.TEST_NLL),
                "under_force_rate": number(row.TEST_under_force_rate),
                "monotonic_root_fraction": number(row.TEST_dense_monotonic_root_fraction),
                "train_test_nll_gap": number(row.TRAIN_to_TEST_NLL_gap),
                "test_roots": int(row.TEST_roots),
                "aggregation": "Three-seed mean",
            })

        direct_mean = metric_row(lc, roots, DIRECT, "SEED_MEAN")
        wm_mean = metric_row(lc, roots, WM, "SEED_MEAN")
        direct_ensemble = metric_row(lc, roots, DIRECT, "ENSEMBLE")
        wm_ensemble = metric_row(lc, roots, WM, "ENSEMBLE")
        for aggregation, direct, wm in (
            ("Three-seed mean", direct_mean, wm_mean),
            ("Ensemble", direct_ensemble, wm_ensemble),
        ):
            gap_rows.append({
                "roots": roots,
                "aggregation": aggregation,
                "direct_minus_wm_frontier_mae_n": number(
                    direct.TEST_frontier_MAE_N - wm.TEST_frontier_MAE_N
                ),
                "direct_frontier_mae_n": number(direct.TEST_frontier_MAE_N),
                "wm_frontier_mae_n": number(wm.TEST_frontier_MAE_N),
                "direct_probability_mae": number(direct.TEST_probability_MAE),
                "wm_probability_mae": number(wm.TEST_probability_MAE),
                "direct_nll": number(direct.TEST_NLL),
                "wm_nll": number(wm.TEST_NLL),
                "direct_under_force_rate": number(direct.TEST_under_force_rate),
                "wm_under_force_rate": number(wm.TEST_under_force_rate),
                "test_roots": int(direct.TEST_roots),
            })

    s50 = per_root[(per_root.task == 0) & (per_root.N_root == 50) &
                   (per_root.aggregation == "ENSEMBLE")]
    direct_s50 = s50[s50.model == DIRECT].set_index("context_id")
    wm_s50 = s50[s50.model == WM].set_index("context_id")
    shared = sorted(set(direct_s50.index) & set(wm_s50.index))
    paired_rows = []
    for context_id in shared:
        d = direct_s50.loc[context_id]
        w = wm_s50.loc[context_id]
        paired_rows.append({
            "context_id": context_id,
            "friction_band": str(d.friction_band),
            "direct_frontier_mae_n": number(d.absolute_frontier_error_N),
            "wm_frontier_mae_n": number(w.absolute_frontier_error_N),
            "direct_minus_wm_frontier_gain_n": number(
                d.absolute_frontier_error_N - w.absolute_frontier_error_N
            ),
            "direct_nll": number(d.NLL),
            "wm_nll": number(w.NLL),
            "direct_minus_wm_nll_gain": number(d.NLL - w.NLL),
        })

    source = {
        "id": "task0_curve_files",
        "label": "Frozen Task 0 root-scaling outputs",
        "path": "ROOT_SCALING_LEARNING_CURVE.csv",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": EVIDENCE_SQL,
            "description": "Reviewed Task 0-only learning-curve and per-root outputs produced by the frozen evaluator.",
            "executed_at": generated_at,
            "tables_used": [
                "ROOT_SCALING_LEARNING_CURVE.csv",
                "ROOT_SCALING_PER_ROOT_METRICS.csv",
                "TASK0_TEST_COLLECTION_AUDIT.json",
                "ROOT_SCALING_MODEL_HASHES.csv",
            ],
            "filters": [
                "task = 0",
                "training roots in {6, 15, 30, 50}",
                "10 untouched TEST roots",
                "Direct = Full Visual",
                "WM / Joint = Visual Joint",
            ],
            "metric_definitions": {
                "frontier_mae_n": "Mean absolute error in the minimum force whose predicted success probability reaches rho=0.80; lower is better.",
                "probability_mae": "Mean absolute error between predicted and empirical full-task success probability over the nine stored TEST forces.",
                "nll": "Bernoulli cross-entropy against empirical success frequency; lower is better.",
                "under_force_rate": "Fraction of TEST roots where the predicted frontier is below the empirical rho=0.80 frontier.",
                "three_seed_mean": "Arithmetic mean of metrics computed separately for seeds 0, 1, and 2; this is the preregistered primary summary.",
                "ensemble": "Metrics computed after averaging the three seed probabilities before making frontier decisions; reported as a sensitivity view.",
            },
        },
    }

    title = "Task 0 Root-Scaling: WM 的 sample-efficiency 故事未被稳健支持"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Technical Task 0-only readout for Direct versus physics-auxiliary WM / Joint across 6, 15, 30, and 50 independent training roots.",
            "generatedAt": generated_at,
            "charts": [{
                "id": "frontier_gap_chart",
                "title": "Direct 与 WM / Joint 的 frontier MAE 差值",
                "subtitle": "正值表示 WM / Joint 的 frontier MAE 更低；每个档位使用相同的 10 个 untouched TEST roots。",
                "type": "bar",
                "dataset": "frontier_gap",
                "sourceId": "task0_curve_files",
                "valueFormat": "number",
                "encodings": {
                    "x": {"field": "roots", "type": "ordinal", "label": "Independent TRAIN roots"},
                    "y": {"field": "direct_minus_wm_frontier_mae_n", "type": "quantitative", "label": "Direct − WM frontier MAE (N)", "format": "number"},
                    "color": {"field": "aggregation", "type": "nominal", "label": "Aggregation"},
                    "tooltip": [
                        {"field": "direct_frontier_mae_n", "type": "quantitative", "label": "Direct frontier MAE", "format": "number"},
                        {"field": "wm_frontier_mae_n", "type": "quantitative", "label": "WM frontier MAE", "format": "number"},
                        {"field": "direct_nll", "type": "quantitative", "label": "Direct NLL", "format": "number"},
                        {"field": "wm_nll", "type": "quantitative", "label": "WM NLL", "format": "number"},
                    ],
                },
                "yAxisTitle": "Frontier MAE gap (N)",
                "layout": "full",
            }],
            "tables": [
                {
                    "id": "primary_metrics_table",
                    "title": "三-seed 主口径的完整指标",
                    "subtitle": "Task 0；每个档位固定 10 个 TEST roots；所有误差指标越低越好。",
                    "dataset": "primary_metrics",
                    "sourceId": "task0_curve_files",
                    "defaultSort": {"field": "roots", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "roots", "label": "TRAIN roots", "format": "number"},
                        {"field": "method", "label": "Method", "type": "text"},
                        {"field": "frontier_mae_n", "label": "Frontier MAE (N)", "format": "number"},
                        {"field": "probability_mae", "label": "Probability MAE", "format": "number"},
                        {"field": "nll", "label": "NLL", "format": "number"},
                        {"field": "under_force_rate", "label": "Under-force rate", "format": "percent"},
                        {"field": "monotonic_root_fraction", "label": "Monotonic roots", "format": "percent"},
                    ],
                },
                {
                    "id": "s50_paired_table",
                    "title": "S50 ensemble 的逐-root 配对检查",
                    "subtitle": "正的 gain 表示 WM / Joint 的误差更低；同一批 10 个 TEST roots。",
                    "dataset": "s50_paired",
                    "sourceId": "task0_curve_files",
                    "defaultSort": {"field": "direct_minus_wm_frontier_gain_n", "direction": "desc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "context_id", "label": "TEST context", "type": "text"},
                        {"field": "friction_band", "label": "Friction band", "type": "text"},
                        {"field": "direct_minus_wm_frontier_gain_n", "label": "Frontier gain (N)", "format": "number", "movement": True},
                        {"field": "direct_minus_wm_nll_gain", "label": "NLL gain", "format": "number", "movement": True},
                    ],
                },
            ],
            "sources": [{"id": source["id"], "label": source["label"], "path": source["path"]}],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "task0_curve_files",
                    "body": "## 技术结论\n\n**Task 0-only 的结果不支持把主结论写成‘WM 只在低数据 regime 明显领先，随后 Direct 单调追上’。** 按预注册的三-seed mean 主口径，WM / Joint 相对 Direct 的 frontier MAE 优势在 S6/S15/S30/S50 分别为 **0.020/0.033/0.062/0.058 N**：差距没有随 roots 单调缩小，S50 也没有稳健追平。与此同时，S6 的 WM / Joint under-force rate 更差（70.0% vs 53.3%），NLL 也更差（0.718 vs 0.286），因此‘6 roots 明显有帮助’不能作为全面结论。",
                },
                {
                    "id": "key_finding",
                    "type": "markdown",
                    "sourceId": "task0_curve_files",
                    "body": "## 只有 ensemble 切片呈现 S50 追平\n\n图中三-seed mean 是预注册主口径，ensemble 是敏感性分析。S50 ensemble 的 frontier MAE 为 Direct **0.330 N**、WM / Joint **0.320 N**，差距只剩 **0.010 N**；但三-seed mean 对应差距仍为 **0.058 N**。因此‘Direct 在 50 roots 基本追上’高度依赖 aggregation 口径，不是跨 seeds 稳健的结果。",
                },
                {"id": "frontier_gap", "type": "chart", "chartId": "frontier_gap_chart", "layout": "full"},
                {
                    "id": "primary_evidence",
                    "type": "markdown",
                    "sourceId": "task0_curve_files",
                    "body": "## 更多 roots 没有形成干净的 learning curve\n\nDirect 的三-seed frontier MAE 在 S15 最好（0.240 N），随后回升到 S30 0.307 N 和 S50 0.312 N；WM / Joint 同样在 S15 最好（0.207 N），随后回升到 0.245 N 和 0.253 N。WM / Joint 的 NLL 从 S6 0.718 改善到 S50 0.550，TRAIN→TEST NLL gap 也从 0.657 缩到 0.493，但其 NLL 在四个档位的三-seed mean 中始终差于 Direct。整体更像‘physics auxiliary 改变了 frontier 行为并在部分 roots 上有收益’，而不是稳定的低数据 sample-efficiency 曲线。",
                },
                {"id": "primary_metrics", "type": "table", "tableId": "primary_metrics_table", "layout": "full"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": "## 比较对象、数据与指标\n\n- **Direct**：Full Visual，直接学习 feasibility。\n- **WM / Joint**：Visual Joint；在相同视觉输入和 feasibility 目标之外，加入 H8 physical trajectory 与 intervention-effect auxiliary losses。它是 physics-auxiliary joint model，不应被表述成完全独立的 runtime world model。\n- 每个训练档位使用 6/15/30/50 个独立 root families；TEST 固定为同一批 10 个 untouched roots。\n- 每个 TEST root 包含 9 个 forces × 5 repeats；经验 frontier 是至少 4/5 full-task successes 的最小 force。\n- 主口径是三 seed 分别计算指标后取均值；ensemble 是先平均概率再作决策的敏感性口径。",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": "## 冻结实验设计\n\n只有 independent TRAIN root count 改变。视觉编码/PCA 维数、模型架构、优化器、学习率、epochs、batch size、physics/IE/feasibility loss 权重、force semantics 和归一化语义保持冻结。36 个 Task 0 checkpoints 在 TEST 合并前已经训练并哈希冻结；TEST 没有参与模型、seed、特征、loss、校准或阈值选择。本报告只评估 Task 0，不含 Task 5，也不作跨任务结论。",
                },
                {
                    "id": "validation",
                    "type": "markdown",
                    "sourceId": "task0_curve_files",
                    "body": "## 预注册 gates 与复算结果\n\nTask 0 的 `full_sample_complexity_gate`、`joint_scale_gate` 和 `joint_emerges_gate` 均为 **FAIL**。S50 ensemble 的逐-root 配对中，WM / Joint frontier 改善 5/10 roots、持平 4/10、变差 1/10；median frontier gain 为 0.025 N。NLL 只改善 3/10 roots，median gain 为 −0.030，因此没有通过要求至少 6/10 roots 且 frontier/NLL median 同向的 gate。所有 headline frontier means 均从 per-root rows 独立复算，最大绝对差小于 1e−16。",
                },
                {"id": "s50_paired", "type": "table", "tableId": "s50_paired_table", "layout": "full"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## 局限与稳健性\n\n10 个 TEST roots 使逐-root 比例以 10 percentage points 跳变；三-seed mean 的 rate 又以 3.33 points 跳变。Frontier、NLL、under-force 与 monotonicity 没有完全同向，不能用单一 frontier MAE 覆盖安全和校准代价。S50 ensemble 看起来接近追平，但 seed-level endpoint 只有 1/3 seeds 同时满足 frontier 更优、under-force 不更差且 NLL 不更差。",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## 下一步\n\n- 对‘漂亮曲线是否在 Task 0 出现’这个问题，可以在这里停止：答案是 **没有稳健出现**。\n- 如果论文仍要主张 physics inductive bias 提高 sample efficiency，应先预注册一个更直接的 physics-target/force-regression 指标或扩大 TEST roots，再复现实验；不要依据 S50 ensemble 单点改写主结论。\n- 如果只需要方法选择，Task 0 证据支持保留 WM / Joint 作为 frontier diagnostic，但不足以把它提升为已验证的 sample-efficiency 贡献。",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": "## 仍待回答的问题\n\nWM / Joint 的 frontier 收益为何集中在少数 roots，且与 NLL/monotonicity 不一致？需要按 friction band、真实 frontier 位置和 boundary-local calibration 做预注册分层，才能区分 physics bias、阈值效应与模型非单调性。Task 5 只有在目标升级为跨任务 generality 时才需要继续。",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "primary_metrics": primary_rows,
                "frontier_gap": gap_rows,
                "s50_paired": paired_rows,
            },
        },
        "sources": [source],
        "package_info": {
            "root": ".",
            "manifestPath": ARTIFACT_PATH.name,
            "snapshotPath": ARTIFACT_PATH.name,
        },
    }

    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    notes = {
        "audience": "technical",
        "delivery_mode": "html",
        "report_structure": [
            "title", "technical summary", "key findings with visual evidence",
            "scope/data/metric definitions", "frozen experimental design",
            "limitations and robustness", "recommended next steps", "further questions",
        ],
        "chart_map": [{
            "section": "Only the ensemble slice suggests S50 catch-up",
            "question": "Does the Direct-versus-WM frontier gap shrink as roots increase?",
            "family": "comparison",
            "type": "grouped bar",
            "fields": ["roots", "aggregation", "direct_minus_wm_frontier_mae_n"],
            "supported_claim": "The gap is not monotonically shrinking under the primary three-seed mean; S50 catch-up is aggregation-sensitive.",
            "palette_policy": "hard two-root cap plus neutrals",
            "delivery_artifact": "TASK0_ROOT_SCALING_REPORT.html",
        }],
        "validation": {
            "task0_test_contexts": 10,
            "task0_test_branches": 450,
            "force_cells": 90,
            "repeats_per_cell": 5,
            "per_root_rows": 480,
            "duplicate_per_root_keys": 0,
            "max_abs_frontier_recompute_difference": 8.326672684688674e-17,
            "max_abs_seed_mean_recompute_difference": 1.1102230246251565e-16,
            "reviewed_evidence_query_rows": len(reviewed_evidence),
            "all_real_frontiers_supported": True,
            "all_decisions_finite": True,
        },
        "preregistered_task0_gates": gates,
        "omissions": {
            "task5": "Explicitly skipped per user request; no across-task classification is made.",
            "second_chart": "Exact NLL, probability, safety, and monotonicity values are provided in the table; a second chart would duplicate the comparison without improving the core decision.",
        },
    }
    NOTES_PATH.write_text(json.dumps(notes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact": str(ARTIFACT_PATH),
        "notes": str(NOTES_PATH),
        "primary_rows": len(primary_rows),
        "gap_rows": len(gap_rows),
        "paired_rows": len(paired_rows),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
