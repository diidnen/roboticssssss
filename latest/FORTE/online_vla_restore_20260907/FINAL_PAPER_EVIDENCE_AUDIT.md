# FINAL PAPER EVIDENCE AUDIT — ActiveForcing

审计时间：2026-09-08T12:59:01.615952+00:00。本轮只读，NEW_PHYSICS=0、NEW_TRAINING=0、模型/utility/controller/evaluator/root 均未修改。

## 1. Executive Summary

当前真实系统是 established-grasp 起点上的 **online frozen π0/Tabero VLA + ActiveForcing 抓力适配**。最终 online-VLA 主实验有 8 个 root groups（前 4 个锁定结果 + 新 4 个 untouched confirmatory），96 contexts、384 branches；所有最终主分支都有在线推理 provenance。核心论文 claim 可以成立，但 Fixed-5 可靠性只能写成部分匹配，不能写 non-inferiority 或 superiority。

## 2. Final Method

common established grasp → 固定 P4-B physical probe → 58D 三成员 physical-belief ensemble → sigma-aware、正支撑的三高斯混合 posterior $p(\mu\mid D_q)$ → 在线 frozen VLA 预测 50-step action chunk → 从 chunk 和当前可观测状态构造 8×64 feasibility sequence → phase-free full-task feasibility → 外层 posterior quadrature 与三 seed ensemble 平均 → 3.00–5.00 N、0.05 N 网格上的 expected-utility 搜索 → 选定 $F^*$ → VLA 的六维 arm/Cartesian action 原样执行，AF 或 fixed 策略独占 squeeze force → 在线 lift/transport/place/release → whole-mesh geometric full-task label。

- **ONLINE_VLA:** YES；checkpoint `/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999`，SHA256 `0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17`。
- **调度:** predict 50、execute 10、每 10 个控制步重新 query，20 Hz simulated control。
- **action arbitration:** arm = online frozen VLA；gripper squeeze = AF/fixed controller；raw VLA opening intent 保留为 release gate；VLA 不覆盖 AF force setpoint。
- **phase:** scripted phase = NO；7 个 phase channel 已删除，最终没有 phase proxy。
- **feasibility 输入:** total 64 = sequence 10 + condition 54；源为 `ONLINE_VLA_ACTION_CHUNK`；无 final scripted-only 输入。
- **belief:** 58D，3 members；使用 sigma、正支撑、数值 quadrature；不是 posterior mean only。
- **force:** support [3,5] N，41 个候选、0.05 N；primary measured squeeze = `2 × min(abs(left object-filtered finger-local grasp-normal), abs(right ...))`。
- **utility:** `p*(5-F)/5 + (1-p)*(-1)`，低力 tie-break。

因此，准确表述是：**ActiveForcing performs online physical adaptation on top of a frozen pretrained VLA without retraining the VLA.**

## 3. Dataset and Training

### Feasibility

当前冻结模型实际使用 **647 条有效 rows、72 个独立 contexts、6 个 root families**（TRAIN 431/48 contexts/4 roots；VAL 108/12/1 root；TEST 108/12/1 root）。每个 context 计划 9 个 candidate forces；其中 1 个 context 只有 8 条有效记录，原因是一个 unknown/quarantined row 被排除。

`CONTINUOUS_TRAIN_SUCCESS_DATA.csv` 的 **720 条**是另一份 24-root 原始 collection（72 contexts × 10 branches，575 positive/145 negative），但当前冻结 feasibility 的训练 manifest 明确记录 `old720_rows_used=0`；它不能被写成当前模型的训练行数。647 是最终模型有效训练 corpus，应在论文中使用；720 只能作为未使用的原始/辅助 collection 记录。

### Belief

最终 paper-ready held-out belief metrics（12 TEST sequences、TEST root 5105）为：MAE 0.06942、bias −0.01243、Spearman 0.89510、68% coverage 83.3%、90% 100%、95% 100%，positive-mixture NLL −0.85575。VAL 为 MAE 0.07539、Spearman 0.96503；不要把旧 development 指标混入 final 指标。

### Feasibility quality

当前 phase-free frozen model 的 ensemble metrics：

| split | NLL | Brier | AUROC | AUPRC | ECE |
|---|---:|---:|---:|---:|---:|
| VAL | 0.27367 | 0.08553 | 0.95128 | 0.98216 | 0.07700 |
| TEST | 0.26308 | 0.07304 | 0.93932 | 0.96931 | 0.07200 |

这些是对 candidate force 与单个 $\mu$ 条件预测后，经 outer planner 做 posterior marginalization 的 frozen feasibility 质量；训练目标本身不是直接把 posterior 当输入的黑箱。

## 4. Online-VLA Main Results

### New 4 untouched roots (170044–170047)

| Method | Full-task SR | Lift SR | Drop | Selected F | Measured squeeze |
|---|---:|---:|---:|---:|---:|
| Fixed-3 | 21/48 (43.8%) | 91.7% | 39.6% | 3.000 | 1.746 N |
| Fixed-4 | 34/48 (70.8%) | 100.0% | 16.7% | 4.000 | 3.483 N |
| Fixed-5 | 35/48 (72.9%) | 100.0% | 8.3% | 5.000 | 4.687 N |
| ActiveForcing | 37/48 (77.1%) | 100.0% | 10.4% | 3.943 | 3.496 N |

AF 相对 Fixed-5 的 measured-force saving = **1.191 N（25.39%）**。四个新 root 的 saving 分别为 1.343、0.806、1.222、1.393 N，方向一致；success 方向不一致。

### Pooled 8-root descriptive evaluation

| Method | Full-task SR | Lift SR | Drop | Selected F | Measured squeeze |
|---|---:|---:|---:|---:|---:|
| Fixed-3 | 37/96 (38.5%) | 90.6% | 40.6% | 3.000 | 1.762 N |
| Fixed-4 | 65/96 (67.7%) | 99.0% | 16.7% | 4.000 | 3.443 N |
| Fixed-5 | 71/96 (74.0%) | 99.0% | 7.3% | 5.000 | 4.689 N |
| ActiveForcing | 69/96 (71.9%) | 100.0% | 10.4% | 3.987 | 3.519 N |

Pooled AF 比 Fixed-3 高 32/96 个成功、比 Fixed-4 高 4/96、比 Fixed-5 少 2/96；相对 Fixed-5 measured squeeze 少 **1.170 N（24.94%）**。AF 与 Fixed-4 的成功率接近且 measured force 仅高约 0.076 N，所以 reviewer 的“为什么不直接 Fixed-4？”应回答：AF 不是只复现均值，而是在未知 context 中产生 graded force；但 pooled full-task advantage 相对 Fixed-4 很小，不能夸大。

### Paired pooled 8-root counts

| comparison | both success | AF only | baseline only | both fail |
|---|---:|---:|---:|---:|
| AF vs Fixed-3 | 35 | 34 | 2 | 25 |
| AF vs Fixed-4 | 54 | 15 | 11 | 16 |
| AF vs Fixed-5 | 62 | 7 | 9 | 18 |

AF–Fixed-5 both-success contexts 的平均 measured saving 为 1.250 N。只有 8 个 root groups，统计应以 root-cluster/descriptive 为主，不能宣称 population-level superiority。

## 5. Context-sensitive Adaptation

Pooled anchor analysis：AF selected-force 与 empirical minimum successful fixed anchor 的 Spearman = **0.673**；CLASS3/4/5 平均 selected force = **3.424/4.072/4.573 N**。类别计数为 14/13/9，unresolved 8，non-monotonic 4。共有 16 easy contexts、14 easy-save、22 hard contexts、15 hard-rescue；AF 有 25 个 distinct selected setpoints，范围 3–5 N。

这支持 **graded context-sensitive force selection**，不支持“精确识别 true minimum force”。Friction sensitivity 的 mean $S_\mu$=0.339、mean $S_F$=0.606；task-level $S_\mu$ 为 task0 0.403、task1 0.565、task5 0.318、task6 0.069，说明 friction conditioning 存在但 task6 较弱。

## 6. Failure Analysis

Pooled AF-fail / Fixed-5-success 共 9 条：under-force 3、post-lift geometric 6、VLA execution 0、other 0。诊断中 model error 0、utility-aggressive 3、execution variance 6。因此不能把 AF 的全部差距称为 slip；大部分是 placement/containment/release 几何或执行方差。AF 仍有 100% pooled lift SR，但 full-task label 要求完整 placement/release，主 SR 不可事后放宽。

## 7. Ablation Inventory

| Ablation | Status | Runtime | Roots/contexts/branches | Result | Paper-usable? |
|---|---|---|---|---|---|
| Posterior Mean | ONLINE_VLA_PHYSICS_COMPLETE | online VLA | 5100,6200 / 24 / 24 | 17/24 (70.8%), selected 3.894 N, measured 3.343 N | 可作 burned-root development ablation；不能证明 uncertainty improves |
| Prior / No-Posterior | ONLINE_VLA_PHYSICS_COMPLETE | online VLA | 5100,6200 / 24 / 24 | 17/24, selected 4.129 N, measured 3.666 N；probe retained | 可用，但名称必须是 PRIOR/NO-POSTERIOR，不是 NoProbe |
| Coarse Grid | ONLINE_VLA_PHYSICS_COMPLETE | online VLA | 5100,6200 / 24 / 24 | 15/24 (62.5%), selected 4.083 N, measured 3.574 N | 可作 development evidence；支持 dense grid 的弱/描述性证据 |
| Local-Lift | PARTIAL | online VLA rollout；model labels scripted auxiliary | 5100,6200 / 24 / 24 | 10/24 (41.7%), selected 3.096 N, measured 1.737 N | 不能作为干净 full-task-vs-lift-only 因果消融 |

这四项来自 `/media/volume/newdata/exouser/online_vla_activeforcing_20260907/online_ablation_results_v2/ONLINE_VLA_ABLATION_RESULTS.json`；总计 96 variant branches + 24 AF controls，非 final fresh-root evidence。

## 8. Other Baselines

Tabero-Neutral 有旧 legacy artifact，但不是 current-V5 matched contract；FORTE-style 使用 privileged slip signal，亦非 apples-to-apples online-VLA。二者都不应放进 final online-VLA main table。当前没有合格 GT-Physics/oracle matched baseline。旧 baseline 只能用于实现 provenance 或 appendix 说明。

## 9. What Current Evidence Proves

- frozen pretrained VLA 在 rollout 中真实在线 inference，且没有 retraining；
- AF 只改变 gripper squeeze-force control，arm/manipulation semantics 由 VLA 生成；
- physical probe + 58D belief + continuous positive-support posterior + posterior-marginalized feasibility + dense force search 已接入真实 online-VLA runtime；
- AF 在 unseen matched contexts 上产生 graded force selection，并相对 Fixed-5 显著降低 measured bilateral squeeze；
- 失败不是单一 slip 机制，placement/VLA/full-task geometry 是重要限制。

## 10. What Current Evidence Does NOT Prove

- posterior uncertainty 比 posterior mean 带来更高最终性能（online ablation 的 SR 相同）；
- AF 在统计意义上 non-inferior 或 superior 于 Fixed-5；
- full-task feasibility 一定优于 lift-only feasibility（Local-Lift 训练标签来源不同）；
- AF 精确恢复每个 context 的真实最小力；
- Tabero-Neutral/FORTE-style 在相同 frozen-VLA、probe、label contract 下的比较优势。

## 11. Missing Experiments

### MUST RUN

**NONE for the core online frozen-VLA + force-adaptation paper claim.** 主实验已完成，provenance 完整，当前模型未因 final roots 改动。

### NICE TO HAVE

如果审稿人明确要求 component-level causal claims，可另行预注册并运行更干净的 VLA-native Local-Lift matched ablation，或扩大 ablation roots；这不是当前投稿核心 claim 的必需条件。

### DO NOT RUN

不再自动跑 physics、补 roots、重训 feasibility、重调 utility、加入 Tabero-Neutral/FORTE-style/GT oracle，除非用户另行授权并重新定义 protocol。

## 12. Minimal Additional Physics Budget

当前核心论文建议 **0 branches**。任何 nice-to-have 消融都不应在本轮自动执行。

## 13. Recommended Final ICRA Tables/Figures

- 正文 Table 1：new4 confirmatory，或明确标注 pooled8 为 descriptive；四方法 Fixed3/4/5/AF。
- 正文 Table 2：pooled8 measured-force / success trade-off、paired counts、per-root results。
- 正文 Table 3：仅放已有 online-VLA burned-root ablations，并标注 development；Local-Lift 注明 scripted auxiliary training。
- 正文或 appendix Table 4：belief 与 feasibility quality，写清 647 effective rows 与 720 unused raw collection 的区别。
- 图：force-success Pareto、task×friction selected/measured force heatmap、anchor-class force plot；failure composition 放 appendix。
- Appendix：完整 provenance、checkpoint/hash、runtime/action arbitration、旧 scripted V5（ENGINEERING CONTROLLED-MOTION DEVELOPMENT EVIDENCE）。

## 14. Recommended Final Claims

最强且诚实的写法：

> ActiveForcing performs online physical adaptation on top of a frozen pretrained VLA without retraining the VLA. Starting from a common established-grasp state, it uses a physical probe and a 58D uncertainty-aware friction belief to select a continuous grip-force setpoint while the frozen VLA generates downstream arm actions online. Across 96 matched contexts, it reduces measured bilateral squeeze by 24.9% relative to Fixed-5, while achieving 71.9% full-task success versus 74.0% for Fixed-5; this demonstrates a reliability–force trade-off rather than Fixed-5 non-inferiority.

## 15. Complete Provenance Index

关键数字对应文件：

- runtime/checkpoint/hash/chunk/arbitration：`/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1/FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json`；manifest SHA256 `f1441d83aa4eba7a5177815b5c25fde3302b95de548e0878a9e6b88f2a5d5f58`。
- new4 主表：`/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1/TABLE_CONFIRMATORY_NEW4.csv`。
- pooled8 主表：`/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1/TABLE_POOLED_8ROOT_DESCRIPTIVE.csv`。
- per-root：`/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1/TABLE_PER_ROOT_RESULTS.csv`。
- paired contexts：`/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1/POOLED8_CONTEXT_RESULTS.csv`。
- failure taxonomy：`/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1/FAILURE_TAXONOMY.csv`。
- feasibility metrics：`/home/exouser/FORTE/analysis/results/current_fulltask_feasibility_baseline_v1_20260906/TRAINING_RESULTS.json`。
- belief metrics：`/home/exouser/FORTE/analysis/results/current_v5_existing_data_paper_package_v1_20260907/BELIEF_DATA_AND_PREDICTIONS.csv`。
- effective 647 rows：`/home/exouser/FORTE/analysis/results/current_v5_existing_data_paper_package_v1_20260907/FEASIBILITY_VALID_ROWS.csv`；raw 720：`/home/exouser/FORTE/gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv`。
- ablations：`/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1/../online_ablation_results_v2/ONLINE_VLA_ABLATION_RESULTS.json`。
- old scripted evidence：`/home/exouser/FORTE/analysis/results/current_v5_existing_data_paper_package_v1_20260907/CURRENT_METHOD_SPEC.md`。

