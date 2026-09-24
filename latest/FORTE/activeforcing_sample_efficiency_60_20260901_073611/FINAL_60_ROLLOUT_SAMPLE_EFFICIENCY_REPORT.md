# ActiveForcing 60-Rollout Sample-Efficiency Report

## Technical summary

最终分类：**`60_ROLLOUTS_INSUFFICIENT_AND_WM_DOES_NOT_HELP`**。核心比较采用三 seed metric mean/std；prediction-level ensemble 单独报告，不与 seed mean 混用。
Sparse-60 下 Direct SR=90.05%，Point-WM SR=87.50%，ΔWM_60=-2.55 pp。Full-120 下 Direct SR=93.06%，Point-WM SR=93.29%，ΔWM_FULL=+0.23 pp。

## Direct answers

1. **当前原始数据：**4 tasks、24 independent root families、72 friction-conditioned contexts、720 archived branches；每 task 180 branches。没有加入 historical coarse outcomes、root-scaling roots 或 untouched TEST。
2. **严格 OOF 的 Full training budget：**每 outer fold 每 task 4 roots × 3 friction × 5 forces × 2 repeats = 120 rollouts；全四 task 为 480 training branches/fold。
3. **Sparse budget：**每 cell 保留 deterministic first archived repeat（实际 repeat IDs 为 1/2，故保留 repeat 1），每 task 60、全四 task 240 training branches/fold，恰为 Full 的 50%。
4. **60 条时 Direct：**三-seed SR mean/std=90.05% ± 1.31 pp；prediction-level ensemble=91.67%。
5. **60 条时 Point-WM：**三-seed SR mean/std=87.50% ± 2.60 pp；prediction-level ensemble=92.36%。
6. **60 条时 WM 相对 Direct：**三-seed mean Δ=-2.55 pp；ensemble Δ=+0.69 pp。
7. **120 条 Full：**Direct=93.06% ± 0.98 pp，Point-WM=93.29% ± 3.12 pp；ensembles 分别为 93.75% / 95.14%。
8. **WM advantage 的数据依赖：**ΔWM_60=-2.55 pp，ΔWM_FULL=+0.23 pp；前者不更大。
9. **Direct-60 接近 Full 吗：**GapToFull_Direct=+3.01 pp，SR recovery ratio=0.9677。按预冻结 1.0 pp 近似阈值，答案是 否。
10. **PointWM-60 接近或超过 Direct-Full 吗：**差值=-5.56 pp；按相同 1.0 pp 阈值，答案是 否。
11. **under-force / utility：**Sparse-60 Point-WM 相对 Direct 的 under-force Δ=+2.55 pp，utility Δ=-0.0337。三 seed 中 Point-WM 的 SR 改善次数为 0/3；虽然 prediction-level ensemble 是 +0.69 pp，但它不能覆盖 0/3 单 seed 改善与 aggregate mean 的负方向。
12. **各 task（Sparse-60；三-seed mean 为主，prediction-level ensemble 单列）：**

| task | Direct SR mean | Point-WM SR mean | ΔSR mean | ensemble ΔSR | Δunder-force mean | Δutility mean |
|---|---:|---:|---:|---:|---:|---:|
| task0 | 89.81% | 87.96% | -1.85 pp | +0.00 pp | +1.85 pp | -0.0173 |
| task1 | 89.81% | 74.07% | -15.74 pp | -5.56 pp | +15.74 pp | -0.1618 |
| task5 | 94.44% | 92.59% | -1.85 pp | +0.00 pp | +1.85 pp | -0.0262 |
| task6 | 86.11% | 95.37% | +9.26 pp | +8.33 pp | -9.26 pp | +0.0705 |

13. **三个 Sparse-60 seeds（POOLED）：**

| seed | Direct SR | Point-WM SR | ΔSR | Direct UF | Point-WM UF | utility Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 88.19% | 84.72% | -3.47 pp | 10.42% | 13.89% | -0.0593 |
| 1 | 90.97% | 86.81% | -4.17 pp | 7.64% | 11.81% | -0.0420 |
| 2 | 90.97% | 90.97% | +0.00 pp | 7.64% | 7.64% | +0.0003 |

14. **上一轮唯一 negative seed：**seed0，Direct 91.67% → Point-WM 88.89%，Δ=-2.78 pp；少 4/144 次成功，但仍为 88.89%，是有限下降而不是 collapse。
15. **Best-seed 图：**Sparse-60 后验最大 Point-WM−Direct seed 是 seed2，但 pooled ΔSR 仅为 +0.00 pp；它是最不差的 seed，不是正收益 seed。`FIG_SAMPLE_EFFICIENCY_BEST_SEED.png` 标题明确写有 “Representative Best Seed” 和 “POST-HOC — VISUALIZATION ONLY”，不用于 aggregate classification。
16. **最诚实分类：**`60_ROLLOUTS_INSUFFICIENT_AND_WM_DOES_NOT_HELP`。

## Full metric table (three-seed mean ± population std; ensemble separate)

| controller | SR mean±std | ensemble SR | under-force mean | mean force | excess force | utility | GT agreement | selected-force error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct-60 | 90.05% ± 1.31 pp | 91.67% | 8.56% | 4.1344 N | 0.3127 N | 0.0532 | 59.72% | 0.2345 N |
| PointWM-60 | 87.50% ± 2.60 pp | 92.36% | 11.11% | 4.1140 N | 0.2920 N | 0.0196 | 37.96% | 0.3930 N |
| Direct-Full | 93.06% ± 0.98 pp | 93.75% | 5.56% | 4.1936 N | 0.3728 N | 0.0817 | 73.61% | 0.1372 N |
| PointWM-Full | 93.29% ± 3.12 pp | 95.14% | 5.32% | 4.1935 N | 0.3726 N | 0.0828 | 51.85% | 0.2752 N |

## Scientific interpretation

Direct 从稀疏 terminal full-task success/failure 学习 force selection；Point-WM 在完全相同的 simulator executions 上，额外利用已有 H8 continuous physical trajectory 与 IE supervision，再通过冻结的 residual-utility recipe 决策。因此本实验测试的是 trajectory-level predictive physical supervision 是否从每次 force-conditioned rollout 提取更多学习信号，不是声称 WM 直接估计 force 更准。

## Protocol integrity and caveats

- Sparse 与 Full 共用相同 outer-heldout roots、candidate force grid、OOF scalar mu_hat semantics、Direct architecture、Point-WM H8/PhysicsOnly/IE/residual architecture、optimizer、epochs、utility 与 search。
- residual training 使用 strict inner grouped-root OOF predictions；outer validation roots 不进入 Direct、WM 或 residual training。
- Full-120 直接复用上一轮 9 个 authoritative atomic-complete shards；Sparse-60 新训练 9 个 shards，但不产生任何 simulator data。
- 95.14% 一类 ensemble 数字来自 branch prediction/utility 先跨 seed 平均、再重新 argmax 决策；绝不是三-seed metric mean。
- `excess_force` 延续既有定义，是 selected force minus empirical success frontier 的有符号均值；under-force episode 因而可贡献负值。
- task1 仍有既知 `MATERIAL_RECONSTRUCTED_LABEL_CAVEAT` caveat（140/180 reconstructed labels）。
- std 是三个 seed metric 的 population std（ddof=0），不是置信区间；144 held-out episodes 共享 24 root families，因此不要把 episodes 当作 144 个独立实验单位。
