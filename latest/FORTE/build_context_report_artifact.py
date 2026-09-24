#!/usr/bin/env python3
"""Build the final Chinese mechanism report and canonical portable artifact."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("/home/exouser/FORTE/structured_grasp_context_20260831")


def clean(records):
    return json.loads(json.dumps(records, allow_nan=False, default=str))


def records(frame: pd.DataFrame):
    return clean(frame.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict("records"))


def fmt(x, n=3):
    return "NA" if pd.isna(x) else f"{float(x):.{n}f}"


def md_table(frame: pd.DataFrame) -> str:
    def cell(x):
        if pd.isna(x):
            return "NA"
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.3f}"
        return str(x).replace("|", "\\|")
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    taskwise = pd.read_csv(OUT / "TASKWISE_STRUCTURED_VS_VISUAL.csv")
    joint = pd.read_csv(OUT / "TASKWISE_STRUCTURED_JOINT.csv")
    audit = pd.read_csv(OUT / "ROOT_PHYSICAL_CONTEXT_AUDIT.csv")
    corr = pd.read_csv(OUT / "ROOT_FRONTIER_PHYSICAL_CORRELATIONS.csv")
    classification = json.loads((OUT / "FINAL_CONTEXT_REPRESENTATION_CLASSIFICATION.json").read_text())
    comparison = json.loads((OUT / "ROOT06_ROOT07_PHYSICAL_COMPARISON.json").read_text())
    repro = json.loads((OUT / "COMPARATOR_REPRODUCIBILITY_AUDIT.json").read_text())

    compact = taskwise[[
        "task", "model", "NLL", "probability_MAE", "Brier", "frontier_MAE_N",
        "under_force_rate", "finite_decision_coverage", "TRAIN_to_rootheldout_NLL_gap",
        "fraction_root_seed_groups_improved_over_Base_NLL",
    ]].copy()
    compact.columns = [
        "task", "model", "NLL", "prob MAE", "Brier", "frontier MAE (N)",
        "under-force", "decision coverage", "TRAIN→held NLL gap", "root groups > Base",
    ]

    train_corr = corr[(corr["split"] == "TRAIN") & corr["feature"].isin([
        "friction", "horizontal_grasp_eccentricity_m", "vertical_grasp_offset_m",
        "gripper_opening_m", "finger_joint_asymmetry_m",
    ])][[
        "task", "feature", "n_contexts", "n_root_seed_groups", "spearman_rho",
        "linear_R2", "partial_pearson_controlling_friction",
    ]]

    report = f"""# Vision vs Structured Context：同 task 新 physical root 机制诊断

## 结论先行

**分类：`{classification['primary_classification']}`。** 当前冻结的 11-D structured grasp context 没有解决 root-heldout 退化：它在 0/3 个 task 上同时满足 NLL 更低、frontier MAE 更低且 under-force 不恶化；只有 task5 相对 Full Visual 更好，但仍不如 Base。Structured Joint 在 0/3 个 task 上取得安全约束下的独立增益。

因此，本轮不支持“这些已记录低维 grasp geometry 就是正确表示”。同时，本轮也没有证明 generic vision 只是 sample-limited；它目前表现为 **TRAIN signal + held-root 退化，即当前六个 root-seed 家族下的 root memorization**。下一步应先冻结新的 untouched TEST，再从 S30 开始做 independent-context scaling，而不是直接采用 structured，也不是加网络容量。

## 六个问题的直答

### 1. 不同 root 到底主要哪里不同？

每个 task 有 18 个 friction-conditioned physical contexts，但只有 6 个 root-seed 家族。跨 root 最明显的已记录变化是 EEF/scene 的毫米级平移、object/EEF relative pose、gripper opening/finger asymmetry 与 friction；第一段 H8 全是 `branch_hold`，命令位移数值上近零。task0/task5 的 relative geometry 变化多为亚毫米，task1 更大，但缺少 EEF quaternion、真实 contact microstate 与 COM offset，因此 audit 不是完整接触几何描述。

特别地，task0 root06 与 root07 的 F*0.8 分别是 4.25 N 与 3.75 N，但 horizontal eccentricity 仅差 0.016 mm、vertical offset 仅差 0.003 mm、opening 仅差 0.016 mm、friction 仅差 0.0065，H8 都近零。记录到的 grasp offset / motion 解释不了 0.5 N 差值；未记录的 contact microstate 与有限 repeats 的随机性仍是合理解释。

### 2. 不同 grasp geometry 是否真的对应不同 F*0.8？

描述性相关存在，但不是稳定机制证据。TRAIN 上 horizontal eccentricity 与 F* 的 Spearman ρ：task0=0.637、task1=0.654、task5=0.715；控制 friction 后的 partial Pearson 分别为 0.734、-0.024、0.413，方向在 task1 翻转。更关键的是，把这些变量一起交给 oracle Structured 模型后仍未在任何 task 击败 Base。结论：部分几何变量与 F* 共变，但当前样本无法证明它们在同 friction 下稳定解释 root frontier。

### 3. generic vision 的 TRAIN gain 到底是不是 root memorization？

**在当前数据规模下，是。** Full Visual 的 TRAIN NLL 均低于 Base，但 root-heldout NLL 在 task0/1/5 均更差：0.215/0.247/0.231 对 0.163/0.161/0.158；TRAIN→held gap 也从 Base 的 0.018/0.041/0.058 增至 0.120/0.191/0.161。这是当前 regime 的 memorization 诊断，不是“vision 永远无效”的证明。

### 4. 低维 grasp context 能不能更稳定地预测新 root？

**不能。** Structured 的 held-root NLL 为 0.317/2.353/0.182，Base 为 0.163/0.161/0.158；frontier MAE 为 0.367/0.183/0.242 N，Base 为 0.191/0.202/0.200 N。task1 虽 frontier MAE 略低，但 decision coverage 降至 0.578、under-force 升至 0.889，且 NLL 崩溃，不能算成功。该结果甚至来自含 simulator-GT object pose 的 oracle vector，因此更不能直接形成 deployable claim。

### 5. physics Joint 在 structured context 下还有没有独立价值？

**没有稳定独立价值。** Structured Joint 相对 Structured 的 NLL 在 task0/1/5 分别恶化 +0.380/+0.727/+0.067；under-force 分别恶化 +0.056/+0.111/+0.167。task5 frontier MAE 改善 0.023 N，但安全与 NLL 同时变差，不通过预定规则。分类为 `{classification['physics_auxiliary_classification']}`。

### 6. 下一步应该增加 independent visual roots，还是直接采用 structured？

**不要采用当前 structured 作为方法；先做冻结的 independent-context scaling。** 在训练选择冻结前创建新的 8–12 roots/task untouched TEST（优先 task0/task5），然后只扩到 S30，检查 Full Visual 是否在新 TEST 上持续改善；通过后再按 S18⊂S30⊂S50⊂S80 扩展。当前不能宣称 `VISUAL_CONTEXT_IS_SAMPLE_LIMITED`，也不能宣称 generic vision 已被证明为低效表示；学习曲线才是区分 A/B 的必要证据。

## 任务级结果

{md_table(compact)}

数值为 3-fold root-heldout 的 fold-macro 均值；每 fold 整体留出 2 个 root IDs（6 个 friction-conditioned contexts）。Lower is better，under-force 是安全侧错误。

## Structured Joint 机制比较

{md_table(joint)}

## 物理相关性（仅描述）

{md_table(train_corr)}

这些相关性把 18 个 friction-conditioned contexts 列出，但有效独立 root-seed 家族只有 6；p 值和 partial correlation 都不应作独立样本推断，也没有被用于 feature selection。

## Deployment legality

11-D vector 中 6 个维度依赖 simulator GT object pose，因此整体只能标为 `STRUCTURED_CONTEXT_ORACLE_DIAGNOSTIC`。EEF base pose、opening、finger asymmetry 是 legal/derived-legal；object-to-EEF translation 与 object orientation 是 privileged。EEF-relative rotation、EEF quaternion、H8 orientation change 不可用。friction 已在 Base 中，合法前提仍是未来 active probe；本轮没有运行 Probe。

## 复现与限制

- Base 与 Full Visual 的所有 frozen CV comparators 在绝对容差 5e-8 内复现。
- task1/task5 Visual Joint 复现；task0 Visual Joint 的旧 CV 在相同源码/数据哈希与精确调用顺序下仍出现 material mismatch，因此报告采用当前同环境配对重训值，并在 `COMPARATOR_REPRODUCIBILITY_AUDIT.json` 中完整记录。task0 Joint 方向应视为带此 caveat。
- task1 有 reconstructed-label caveat；task5 是 direct-label 锚点。
- 这些 held roots 已看过，只是 retrospective mechanism diagnostic，不是新 TEST。
- 未运行 Probe、未采新 simulator 数据、未修改 encoder/PCA/architecture/optimizer/epoch/λ/force semantics/split/labels。

## 最终建议

`{classification['next_step']}`

具体执行：先冻结新 TEST；只采更多独立 π0 physical roots，不增加同 root repeats 冒充 context；先完成 S30 gate。若 S30→S50→S80 的新 TEST 表现不持续改善，则接受 generic visual embedding 是低效 physical representation，并转向重新冻结、可合法观测的更完整 contact/grasp state，而不是维护 vision 假设。
"""
    (OUT / "VISION_VS_STRUCTURED_CONTEXT_REPORT.md").write_text(report, encoding="utf-8")

    chart_rows = taskwise[[
        "task", "task_name", "model", "NLL", "frontier_MAE_N", "under_force_rate",
        "Brier", "TRAIN_to_rootheldout_NLL_gap", "finite_decision_coverage",
    ]].copy()
    chart_rows["task"] = chart_rows["task"].astype(str)
    scatter = audit[(audit["split"] == "TRAIN") & audit["empirical_Fstar_0p8_N"].notna()][[
        "task", "context_id", "root_id", "friction_band", "friction",
        "horizontal_grasp_eccentricity_m", "gripper_opening_m", "empirical_Fstar_0p8_N",
    ]].copy()
    scatter["task"] = scatter["task"].astype(str)
    scatter["horizontal_eccentricity_mm"] = scatter["horizontal_grasp_eccentricity_m"] * 1000
    scatter["gripper_opening_mm"] = scatter["gripper_opening_m"] * 1000
    summary_rows = [{
        "structured_win_tasks": classification["counts"]["structured_beats_base_tasks"],
        "structured_joint_win_tasks": classification["counts"]["structured_joint_win_tasks"],
        "visual_train_signal_tasks": 3,
        "independent_root_seed_groups_per_task": 6,
    }]

    sources = [
        {
            "id": "headline",
            "label": "Headline mechanism summary",
            "path": "TASKWISE_STRUCTURED_VS_VISUAL.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": "WITH t AS (SELECT * FROM read_csv_auto('TASKWISE_STRUCTURED_VS_VISUAL.csv')), j AS (SELECT * FROM read_csv_auto('TASKWISE_STRUCTURED_JOINT.csv')) SELECT 0 AS structured_win_tasks, SUM(CASE WHEN StructuredJoint_better_NLL_and_frontier_underforce_nonworse THEN 1 ELSE 0 END) AS structured_joint_win_tasks, 3 AS visual_train_signal_tasks, 6 AS independent_root_seed_groups_per_task FROM j",
                "description": "Frozen rule counts and audited root-seed-family cardinality.",
                "tables_used": ["TASKWISE_STRUCTURED_VS_VISUAL.csv", "TASKWISE_STRUCTURED_JOINT.csv"],
                "metric_definitions": ["A win requires lower NLL and frontier MAE with non-worse under-force."],
            },
        },
        {
            "id": "taskwise",
            "label": "Taskwise structured vs visual metrics",
            "path": "TASKWISE_STRUCTURED_VS_VISUAL.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('TASKWISE_STRUCTURED_VS_VISUAL.csv') ORDER BY task, model",
                "description": "Three-fold grouped root-heldout summary joined to three-seed full-TRAIN metrics.",
                "tables_used": ["TASKWISE_STRUCTURED_VS_VISUAL.csv"],
                "filters": ["tasks in {0,1,5}", "retrospective TRAIN root-heldout folds", "seed0 CV"],
                "metric_definitions": [
                    "NLL = mean binary negative log likelihood over held branches",
                    "frontier_MAE_N = mean absolute error of finite p>=0.8 force frontier",
                    "under_force_rate = share of valid contexts whose predicted frontier is below empirical F*0.8 or absent",
                ],
            },
        },
        {
            "id": "audit",
            "label": "Root physical context audit",
            "path": "ROOT_PHYSICAL_CONTEXT_AUDIT.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('ROOT_PHYSICAL_CONTEXT_AUDIT.csv') WHERE task IN (0,1,5)",
                "description": "Authoritative pre-probe snapshot, telemetry, frontier, split, and legality audit.",
                "tables_used": ["ROOT_PHYSICAL_CONTEXT_AUDIT.csv"],
                "filters": ["task in {0,1,5}", "authoritative collected contexts only"],
            },
        },
        {
            "id": "joint",
            "label": "Structured Joint comparisons",
            "path": "TASKWISE_STRUCTURED_JOINT.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('TASKWISE_STRUCTURED_JOINT.csv') ORDER BY task",
                "description": "Paired taskwise Joint-minus-non-Joint differences under frozen losses and folds.",
                "tables_used": ["TASKWISE_STRUCTURED_JOINT.csv"],
            },
        },
        {
            "id": "correlations",
            "label": "Root-frontier physical correlations",
            "path": "ROOT_FRONTIER_PHYSICAL_CORRELATIONS.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('ROOT_FRONTIER_PHYSICAL_CORRELATIONS.csv') WHERE split='TRAIN'",
                "description": "Descriptive Spearman, linear fit, and partial Pearson relationships; not used for feature selection.",
                "tables_used": ["ROOT_FRONTIER_PHYSICAL_CORRELATIONS.csv"],
            },
        },
        {
            "id": "repro",
            "label": "Comparator reproducibility audit",
            "path": "COMPARATOR_REPRODUCIBILITY_AUDIT.json",
            "query": {
                "language": "python",
                "query": "json.load(open('COMPARATOR_REPRODUCIBILITY_AUDIT.json'))",
                "description": "Current exact-protocol comparator values versus frozen CV references.",
            },
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Vision vs Structured Context：同 task 新 physical root 机制诊断",
            "description": "Tabero / ActiveForce retrospective grouped root-heldout diagnostic.",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "cards": [
                {"id": "structured-wins", "dataset": "summary", "sourceId": "headline", "description": "同时改善 NLL、frontier MAE 且 under-force 不恶化的 task 数。", "metrics": [{"label": "Structured 胜 Base", "field": "structured_win_tasks", "format": "number"}]},
                {"id": "joint-wins", "dataset": "summary", "sourceId": "headline", "description": "满足同一安全规则的 Structured Joint 独立胜出 task 数。", "metrics": [{"label": "Structured Joint 胜出", "field": "structured_joint_win_tasks", "format": "number"}]},
                {"id": "root-groups", "dataset": "summary", "sourceId": "headline", "description": "每 task 真正用于 grouped CV 的 root-seed 家族数；不是 18 个完全独立 roots。", "metrics": [{"label": "独立 root 家族 / task", "field": "independent_root_seed_groups_per_task", "format": "number"}]},
            ],
            "charts": [
                {
                    "id": "nll-comparison", "title": "Root-heldout NLL", "subtitle": "三个 task 中 Structured 与 Visual 都未稳定击败 Base；task1 Structured 明显崩溃。", "intent": "comparison", "question": "Which context representation generalizes across held root groups?", "rationale": "Grouped bars preserve the task and model comparison at the frozen evaluation grain.", "type": "bar", "dataset": "taskwise", "sourceId": "taskwise", "layout": "full", "maxRows": 18,
                    "encodings": {"x": {"field": "task_name", "type": "nominal", "label": "Task"}, "y": {"field": "NLL", "type": "quantitative", "label": "NLL", "format": "number"}, "color": {"field": "model", "type": "nominal", "label": "Model"}, "tooltip": [{"field": "frontier_MAE_N", "type": "quantitative", "label": "Frontier MAE", "unit": "N"}, {"field": "under_force_rate", "type": "quantitative", "label": "Under-force", "format": "percent"}]},
                },
                {
                    "id": "frontier-comparison", "title": "Root-heldout frontier MAE", "subtitle": "较低的概率误差并不自动带来更安全的 force decision；需与 under-force 一起读。", "intent": "comparison", "question": "Which model predicts empirical F*0.8 most accurately?", "rationale": "The force-facing metric is shown separately from probability NLL.", "type": "bar", "dataset": "taskwise", "sourceId": "taskwise", "layout": "full", "maxRows": 18, "unit": "N",
                    "encodings": {"x": {"field": "task_name", "type": "nominal", "label": "Task"}, "y": {"field": "frontier_MAE_N", "type": "quantitative", "label": "Frontier MAE", "unit": "N", "format": "number"}, "color": {"field": "model", "type": "nominal", "label": "Model"}, "tooltip": [{"field": "NLL", "type": "quantitative", "label": "NLL"}, {"field": "finite_decision_coverage", "type": "quantitative", "label": "Coverage", "format": "percent"}]},
                },
                {
                    "id": "geometry-scatter", "title": "Horizontal grasp eccentricity vs empirical F*0.8", "subtitle": "点按 task 着色；相关性受 friction 与仅 6 个 root-seed 家族限制，只作描述。", "intent": "relationship", "question": "Does logged horizontal grasp offset track the empirical force frontier?", "rationale": "One point per valid TRAIN physical context preserves the observed context grain.", "type": "scatter", "dataset": "root_scatter", "sourceId": "audit", "layout": "full", "maxRows": 53,
                    "encodings": {"x": {"field": "horizontal_eccentricity_mm", "type": "quantitative", "label": "Horizontal eccentricity", "unit": "mm"}, "y": {"field": "empirical_Fstar_0p8_N", "type": "quantitative", "label": "Empirical F*0.8", "unit": "N"}, "color": {"field": "task", "type": "nominal", "label": "Task"}, "tooltip": [{"field": "context_id", "type": "text", "label": "Context"}, {"field": "root_id", "type": "text", "label": "Root"}, {"field": "friction", "type": "quantitative", "label": "Friction"}, {"field": "gripper_opening_mm", "type": "quantitative", "label": "Opening", "unit": "mm"}]},
                },
            ],
            "tables": [
                {
                    "id": "taskwise-table", "title": "Taskwise model evidence", "subtitle": "3-fold macro means; lower NLL/frontier MAE/under-force is better.", "dataset": "taskwise", "sourceId": "taskwise", "layout": "full", "density": "dense", "defaultSort": {"field": "task", "direction": "asc"},
                    "columns": [
                        {"field": "task", "label": "Task", "type": "text"}, {"field": "model", "label": "Model", "type": "text"},
                        {"field": "NLL", "label": "NLL", "format": "number"}, {"field": "Brier", "label": "Brier", "format": "number"},
                        {"field": "frontier_MAE_N", "label": "Frontier MAE (N)", "format": "number"}, {"field": "under_force_rate", "label": "Under-force", "format": "percent"},
                        {"field": "finite_decision_coverage", "label": "Coverage", "format": "percent"}, {"field": "TRAIN_to_rootheldout_NLL_gap", "label": "TRAIN→held gap", "format": "number"},
                    ],
                },
                {
                    "id": "joint-table", "title": "Structured Joint mechanism deltas", "subtitle": "正 NLL/under-force delta 表示 Joint 更差。", "dataset": "joint", "sourceId": "joint", "layout": "full", "density": "dense", "defaultSort": {"field": "task", "direction": "asc"},
                    "columns": [
                        {"field": "task", "label": "Task", "type": "text"},
                        {"field": "StructuredJoint_minus_Structured_NLL", "label": "Δ NLL", "format": "number"},
                        {"field": "StructuredJoint_minus_Structured_frontier_MAE_N", "label": "Δ frontier MAE (N)", "format": "number"},
                        {"field": "StructuredJoint_minus_Structured_under_force_rate", "label": "Δ under-force", "format": "number"},
                        {"field": "StructuredJoint_better_NLL_and_frontier_underforce_nonworse", "label": "Joint passes", "type": "text"},
                    ],
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Vision vs Structured Context：同 task 新 physical root 机制诊断"},
                {"id": "executive", "type": "markdown", "body": "## Executive Summary\n\n**CURRENT_STRUCTURED_CONTEXT_INSUFFICIENT。** 当前 11-D oracle structured context 在 0/3 tasks 击败 Base；Structured Joint 在 0/3 tasks 有安全约束下的独立价值。Generic vision 当前是 TRAIN signal + held-root 退化，属于六个 root-seed 家族 regime 下的 root memorization；但是否 sample-limited 仍需冻结的新 TEST 与 S18→S30→S50→S80 学习曲线。"},
                {"id": "headline", "type": "metric-strip", "cardIds": ["structured-wins", "joint-wins", "root-groups"]},
                {"id": "answers", "type": "markdown", "body": "## 六个问题的直答\n\n1. Roots 主要在毫米级 EEF/scene pose、relative pose、opening/finger state 与 friction 上不同；H8 是近零 branch_hold，contact microstate 未记录。\n2. 几何与 F* 有描述性相关，但控制 friction 后不跨 task 稳定，oracle Structured 也未击败 Base。\n3. Generic vision 当前表现为 root memorization，不是已建立的 physical generalization。\n4. 低维 grasp context 没有更稳定预测新 root。\n5. Structured Joint 没有独立价值。\n6. 不直接采用 structured；先冻结新 TEST，从 S30 开始增加 independent roots。"},
                {"id": "probability-heading", "type": "markdown", "body": "## Held-root probability generalization"},
                {"id": "nll-chart-block", "type": "chart", "chartId": "nll-comparison"},
                {"id": "force-heading", "type": "markdown", "body": "## Force-frontier behavior"},
                {"id": "frontier-chart-block", "type": "chart", "chartId": "frontier-comparison"},
                {"id": "taskwise-table-block", "type": "table", "tableId": "taskwise-table"},
                {"id": "root-heading", "type": "markdown", "body": "## Root meaning and interpretability\n\nTask0 root06/root07 的 frontier 相差 0.5 N，但记录到的 horizontal offset、vertical offset、opening 和 friction 几乎相同。下图仅展示描述性共变；每 task 只有 6 个 root-seed 家族。"},
                {"id": "scatter-block", "type": "chart", "chartId": "geometry-scatter"},
                {"id": "joint-heading", "type": "markdown", "body": "## Physics Joint value\n\nStructured Joint 在三个 task 的 NLL 与 under-force 都没有通过非劣规则；task5 的小幅 frontier 改善不足以抵消安全退化。"},
                {"id": "joint-table-block", "type": "table", "tableId": "joint-table"},
                {"id": "caveats", "type": "markdown", "body": "## Caveats and reproducibility\n\nStructured vector 含 6 个 simulator-GT object-pose 维度，只是 oracle diagnostic。Held roots 已看过，不是新 TEST。task1 有 reconstructed-label caveat。Base/Full Visual 全部复现；task0 legacy Visual Joint 未在完全相同哈希下复现，当前报告使用同环境 paired retraining 并保留审计。没有 Probe、采集、PCA/encoder/architecture/lambda 修改。"},
                {"id": "next", "type": "markdown", "body": "## Recommendation\n\n先创建 8–12 roots/task 的 untouched TEST（优先 task0/task5），再采独立 contexts 到 S30；只有新 TEST 持续改善才扩到 S50/S80。不要把 repeats 当 contexts，也不要因为 TRAIN gain 扩网络。当前 bottleneck 是否为 sample complexity 仍待学习曲线证明。"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "status": "ready",
            "datasets": {
                "summary": summary_rows,
                "taskwise": records(chart_rows),
                "joint": records(joint),
                "root_scatter": records(scatter),
                "correlations": records(train_corr),
                "reproducibility": clean(repro["rows"]),
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
