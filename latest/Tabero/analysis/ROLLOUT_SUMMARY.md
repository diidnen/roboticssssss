# Tabero Rollout / Experiment Summary

统计日期：2026-09-01
数据根目录：`analysis/results/`

## 总体规模

`analysis/results/` 不是单一实验，而是多个阶段、重复运行、失败恢复和 forensic audit 的归档。因此不把所有目录合并成一个总体成功率；下面先报告可核对的库存，再报告协议明确的代表性结果。

| 项目 | 数值 |
|---|---:|
| 实验结果目录 | 334 |
| 文件总数 | 20,341 |
| 原始文件体积 | 1,791,608,484 bytes（约 1.79 GB） |
| 结果日期范围 | 2026-08-18 至 2026-08-31 |
| CSV | 12,544 |
| NPZ | 2,642 |
| JSON | 1,693 |
| 日志 | 673 |
| Markdown 报告 | 303 |
| Verdict JSON | 61 |

主要 rollout/表格数据（CSV、NPZ、NPY、Parquet、ORC、Feather、PT、JSON/JSONL）约 1.61 GB。项目中另有约 1.1 GB 的外部研究仓库快照和 154 MB 的 `_vendor/pyarrow` 缓存；这些不是 Tabero 自己的 rollout 数据，上传时排除。

## 代表性结果

| 阶段/结果 | 统计结果 | 解释 |
|---|---|---|
| B2 Tabero benchmark | 9 个 official tasks 有数据并通过 reset；positive tasks 为 0、1、5；fixed-low tasks 为 3、6、7、9；blocked tasks 为 0 个 | 固定鲁棒力在部分任务有效，但不是所有任务都需要/支持同一决策形式 |
| B5 neutral baseline | mean full SR = 0.55；fixed-robust mean SR = 0.993；差值 = -0.443；5 个任务中只有 task 0 测得正确的力排序，预测 force-slot 排序为 0 个任务 | neutral policy 没有学到可靠的 hidden-physics force adaptation，且多任务本身失败 |
| P5S0C paired probe | 14 个 boundary decision-discordant pairs；probe 有扰动敏感性，但没有可靠转化为 paired decision gain | 有 probe signal，不等于有可部署的决策价值 |
| P6G0 grasp-force benchmark | 筛选 tasks = 0/1/2/5/6；进入主 sweep = 1/6；force-effect rows = 12；grasp-position-effect rows = 7；lift-success/full-failure rows = 27；hidden-physics ranking flips = 2 | benchmark 条件成立，但尚未证明 learned policy、physical query 或 DreamTrajectory 带来提升 |
| GNP-style continuous | 1008 个同源 feasibility outcomes，其中 791 success / 217 failure；Feas frontier MAE = 0.325N，Joint = 0.125N；但 Feas under-force = 0.250，Joint = 0.375，Joint 在 9 个 context 中有 6 个出现 dense-grid nonmonotonicity | 按预注册 safety-first gate 选择 FEASIBILITY_ONLY；但连续控制整体仍未通过验证，GT continuous gate = FAIL |

## Rollout 表质量检查

检查了几个有明确 trial-level schema 的代表性表：

- `P7A_ROLLOUT_RESULTS.csv`：240 行、240 个唯一 `trial_id`；`full_task_success` 非空率 100%，成功率 48/240 = 20.0%；96/240 行标记为 `vla_chunk_budget_exhausted`，该预算问题需要与任务失败分开解释。
- `P6G1_ROLLOUT_RESULTS.csv`：220 行、220 个唯一 `trial_id`；`local_grasp_success` = 211/220 = 95.9%，`short_lift_success` = 169/220 = 76.8%，`contact_loss` = 42/220 = 19.1%，`policy_timeout` = 51/220 = 23.2%。
- `P6G1R2_FINAL_SINGLE_SHOT_RESULTS.csv`：288 行、288 个唯一 `trial_id`；`stable_lift` = 159/288 = 55.2%，`policy_timeout` = 129/288 = 44.8%。
- `P6G1R2_FINAL_RETRY_RESULTS.csv`：144 行、144 个唯一 `trial_id`；`stable_lift` = 73/144 = 50.7%，`policy_timeout` = 71/144 = 49.3%。

这些表的主键和核心结果字段没有发现空值或重复；但不同阶段的 success 定义、任务集合、force protocol 和 failure stage 不同，不能直接横向合并成一个总成功率。

## 当前总体判断

截至 2026-08-31，Tabero 已形成大规模的 rollout、telemetry、训练/审计和复现实验归档。证据支持：固定鲁棒力基线在若干任务上很强；probe/physics signal 可以被观测到；但从 signal 到可靠的 hidden-physics action decision、跨任务泛化和连续最小力控制仍有明显缺口。最稳妥的当前方法选择是 `FEASIBILITY_ONLY`，并继续处理连续 frontier 的安全性、单调性和跨任务验证问题。

## GitHub 上传范围

上传 Tabero 项目源码、实验脚本、报告、rollout/telemetry 原始文件和本统计文件。明确排除：

- `analysis/**/repositories/`：外部项目的完整克隆和其 `.git` 历史；
- `analysis/results/**/_vendor/`：本地运行时依赖缓存；
- `analysis/**/.git/`：嵌套 Git 元数据。

按该范围，rollout 归档仍约 1.8 GB，GitHub 普通仓库并不适合作为大规模数据仓库；若远程服务拒绝超大仓库或推送超时，需要改用 Git LFS/Release/对象存储，而不是删掉原始数据。

## 主要来源

- `analysis/results/b2_tabero_benchmark_table_20260820_065520/FINAL_VERDICT.json`
- `analysis/results/b5_tabero_neutral_20260822_040652/FINAL_VERDICT.json`
- `analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_FINAL_VERDICT.json`
- `analysis/results/p6g0_grasp_force_physics_benchmark_20260824_182251/P6G0_FINAL_REPORT.md`
- `analysis/results/gnp_style_continuous_20260830_125107/FINAL_REPORT.md`
- `analysis/results/p7a_fixed_invocation_force_frontier_20260826_153426/P7A_ROLLOUT_RESULTS.csv`
- `analysis/results/p6g1_primitive_ik_vla_grasp_realization_20260825_103146/P6G1_ROLLOUT_RESULTS.csv`
