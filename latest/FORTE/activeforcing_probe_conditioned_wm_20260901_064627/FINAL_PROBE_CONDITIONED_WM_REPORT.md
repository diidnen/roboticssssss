# Final Probe-Conditioned WM Report

**直白结论：真实 root-heldout Probe error 下，rich Probe history 和 Probe-conditioned PhysicsOnly WM 都没有可靠地改善 force selection。分类为 `CONTROLLED_ROBUSTNESS_SIGNAL_WITHOUT_REAL_PROBE_GAIN`。当前最终候选应保留 `scalar Probe + Direct Utility`，不要把 Probe-conditioned WM 放进默认 controller。**

OOF Probe 的 MAE 是 **0.0655**，signed error 是 **-0.0165**，friction pair-ranking 是 **98.6%**。Normal pooled 上，ProbeScalar-Direct 的 SR / under-force / mean force / realized utility 是 **93.75% / 4.86% / 4.202 N / 0.0879**；ProbeRich-Direct 是 **92.36% / 6.25% / 4.093 N / 0.0932**；Probe-conditioned WM 是 **93.06% / 5.56% / 4.153 N / 0.0888**。

## Direct answers

1. **Probe 的真实 held-root error 有多大？** 72 个 friction-conditioned contexts 上，ensemble OOF MAE=0.0655，bias=-0.0165；LOW/MID/HIGH boundaries 在训练结果查看前冻结为 0.0304/0.0741。这是 24 个 root families 的 grouped OOF 结果，不是 overlap estimator 的 in-sample 数字。

2. **Probe 出错时 Direct 是否跟着变差？** 有，但信号不强且集中在 HIGH_ERROR。HIGH_ERROR 中 ProbeScalar SR=87.50%、under-force=12.50%，GT-Direct 为 91.67%/8.33%；ProbeScalar 与 GT 的全体 force-decision agreement 只有 81.94%。LOW_ERROR 中两者 SR 分别为 95.83% 和 95.83%。

3. **仅把 rich Probe 信息给 Direct 能恢复多少？** 没有恢复：SR 从 93.75% 降到 92.36%，under-force 从 4.86% 升到 6.25%。因此现有固定 16D trace summary 并未成为有效的额外 physics source。

4. **Point-WM 看到相同错误 mu_hat 时能否纠错？** Ensemble 指标表面上改善到 SR=95.14%、under-force=3.47%，但 3 seeds 方向不稳定（逐 seed gate=[False, True, True]），不能作为稳定 WM evidence；它也没有检验 rich Probe 信息的独立价值。

5. **Probe-conditioned WM 是否明显更强？** 否。它相对 ProbeScalar 的 SR 低 0.69 pp、under-force 高 +0.69 pp；相对 ProbeRich 虽多成功 1/144 episode，但 utility 更低（0.0888 vs 0.0932）。

6. **它是否超过 ProbeRich Direct？** 没有通过 matched gate：`ProbeConditionedWM_gt_ProbeRich=False`。这排除了“只因输入更多就把收益算给 WM”的解释。

7. **WM gain 是否随 Probe error 増大？** 没有。Probe error 与 Probe-conditioned-WM 相对 Scalar 的 context-level utility gain Spearman rho=-0.126；HIGH_ERROR 的 Probe-conditioned WM SR/under-force 是 85.42%/14.58%，没有超过 Scalar 的 87.50%/12.50%。

8. **friction over-estimation 的 under-force 能救吗？** 没有稳定救回。OVER_ESTIMATE 全体下 Scalar SR/under-force=85.71%/10.71%，Probe-conditioned WM=85.71%/10.71%。

9. **friction under-estimation 的 excess force 能减少吗？** Probe-conditioned WM 的 mean force 相对 Scalar 可在明细中检查，但 realized utility 没有形成跨 task/seed 的稳定优势；当 Scalar 与 GT force 不同时，WM 向 GT 移动 10 次、反向移动 6 次。

10. **controlled scalar corruption 是否更 robust？** 有一个受控但不足以晋升的方法信号：在最大正/负 corruption 下，Probe-conditioned WM 的 utility 分别为 0.0916/0.0990，Scalar 为 0.0768/0.0788。但 Probe-conditioned WM 没有超过 RichDirect 的平均大扰动 utility，因此这是 `controlled robustness`，不是真实 Probe-error gain，也不是 WM 独立价值。

11. **normal pooled 是否不伤害 Direct？** 没通过：SR 下降、under-force 上升，并出现 2 个 Scalar-success -> WM-failure collateral cases。2 个 ProbeScalar-fail/GT-success 可恢复案例中，WM 救回 0 个。

12. **最终选择哪个版本？** 在三个候选中选择 **Version A: scalar Probe + Direct Utility**。Version B 的 rich summary 没有收益；Version C 没通过 Scalar、Rich、HIGH_ERROR、3-seed 和 multi-task gates。NoProbe SR=94.44%、GT SR=95.14%；所以这一轮也不能单独把 Active Probe 的整体论文贡献宣称为已最终验证，只能回答 Probe-conditioned WM 不应加入。

## Why this classification is conservative

`CONTROLLED_ROBUSTNESS_SIGNAL_WITHOUT_REAL_PROBE_GAIN` 只表示：人为破坏 scalar 时，保留 interaction input 的 pipeline 比纯 Scalar 的 utility 退化更慢。它不表示 WM 已经利用真实 Probe 误差，因为真实 HIGH_ERROR、normal pooled、matched RichDirect、多个 task 和三个 seeds 的晋升条件均未同时成立。

## Protocol integrity and caveats

- Probe、Direct、Point-WM、Probe-conditioned WM 和 residual 均采用 nested grouped root-family OOF；同一 root 的 friction contexts、forces 和 repeats 不跨 fold。
- PhysicsOnly WM 固定 H=8、13 channels、`Lphysics + lambda_IE LIE`，没有 outcome BCE；residual architecture、MSE、epochs、optimizer、seeds 均未修改。
- Rich input 是在看 WM 结果前冻结的、fold-invariant 16D raw-trace physical summary（四个 signal 的 final/mean/std/max）。OOF GRU hidden 也导出供审计，但未直接拼接，因为各 fold 独立训练的 hidden 坐标不可识别。
- `sigma_mu` 仅作 diagnostic，没有 posterior 或 Bayesian claim。
- corruption 的 delta=0 严格复用未 clipping 的正式 baseline；只有非零扰动按预注册要求 clip 到 [0.2,1.0]。
- task1 outcome 保留 140/180 reconstructed-label caveat。
- 本轮没有读取 root-scaling untouched TEST，没有启动 simulator，也没有开发 Hard Verifier。
