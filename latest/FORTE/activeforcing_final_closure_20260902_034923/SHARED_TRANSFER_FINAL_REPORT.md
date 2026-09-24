# Shared transfer final report

# Final Shared Physical Transfer Report

## Technical summary

最直白的结论：这组 frozen architecture 不能支持真正的 semantic zero-shot claim，而且**小样本并未稳定饱和**。Direct 只有 4D task one-hot；B=0 target coordinate 从未激活。Learned-Probe root-heldout macro SR 在 B=0/10/20/30/60 为 **73.15% / 94.44% / 78.94% / 77.78% / 92.13%**，曲线强烈非单调；按预注册 gate，macro B* 是 **60**，所以分类为 **SUBSTANTIAL_TASK_SPECIFIC_DATA_REQUIRED**。

固定 probe primitive保留了可跨 task 解读的信号，但当前 learned GRU interpreter 并不 task-general：LOTO MAE **0.5149**，远差于 pooled root-heldout 0.0655。source-only explicit SysID 降到 **0.1810**，但它在 B=60 的 controller SR/utility 没超过 Learned Probe，因此现在还不能仅凭 estimator MAE 删除 learned Probe。

## The frozen task one-hot prevents a semantic zero-shot claim

Direct 的 71D input只含 nominal Cartesian command/phase、4D task one-hot、force、scalar μ 与 physical state/masks；没有 language、RGB、VLA semantic feature。B=0 是 unseen one-hot coordinate extrapolation diagnostic。B>0 才让 target coordinate 获得监督。这里 held-out task 同时更换 object asset，因此 task effect 与 object-family effect也不能分离。

## Probe transfer fails for the learned interpreter, while explicit response fitting is more stable

Learned Probe LOTO macro MAE = **0.5149**，相对 0.0655 恶化 **+0.4494**；Explicit SysID = **0.1810**，比 learned LOTO 低 **0.3340**。四个 target 的 source-only formula 都由 source-root CV 选择为 probe-out force-asymmetry variation，再做 source-only affine sensor calibration。

这不等于直接 Coulomb identification：trace 没有 authoritative slip onset，且观察到的 `Ft/Fn` 远小于 GT μ。结果只支持‘同一 probe trace 可被显式响应曲线更稳定地解释’，不支持‘已直接测得 Coulomb μ’。

## New-task adaptation is non-monotonic and needs the full 60-rollout reference

| physics | B | macro SR | under-force | utility | mean force |
|---|---:|---:|---:|---:|---:|
| LearnedProbe | 0 | 73.15% | 25.46% | -0.1738 | 3.997 N |
| LearnedProbe | 10 | 94.44% | 4.17% | 0.0472 | 4.473 N |
| LearnedProbe | 20 | 78.94% | 19.68% | -0.0728 | 3.978 N |
| LearnedProbe | 30 | 77.78% | 20.83% | -0.0824 | 3.934 N |
| LearnedProbe | 60 | 92.13% | 6.48% | 0.0399 | 4.342 N |
| ExplicitSysID | 0 | 74.07% | 24.54% | -0.1327 | 3.912 N |
| ExplicitSysID | 10 | 90.97% | 7.64% | 0.0248 | 4.365 N |
| ExplicitSysID | 20 | 82.64% | 15.97% | -0.0262 | 4.016 N |
| ExplicitSysID | 30 | 81.48% | 17.13% | -0.0458 | 4.022 N |
| ExplicitSysID | 60 | 89.58% | 9.03% | 0.0273 | 4.252 N |
| GT | 0 | 81.94% | 16.67% | -0.0218 | 3.869 N |
| GT | 10 | 91.90% | 6.71% | 0.0852 | 4.097 N |
| GT | 20 | 90.97% | 7.64% | 0.0944 | 3.979 N |
| GT | 30 | 92.36% | 6.25% | 0.0977 | 4.047 N |
| GT | 60 | 92.36% | 6.25% | 0.1058 | 4.004 N |

B=60 uses 180 source + 60 target = 240 branches and is the matched LOTO full-adaptation scale. The separately archived existing all-4 Sparse60 Direct reference is **90.05%** three-seed mean SR; it used pooled same-task Probe training and is shown as a horizontal reference, not conflated with LOTO+B60.

The B=10 spike is not evidence that 10 is enough: task6 is one-class in all three folds, B=20/30 regress sharply, and the 3-seed/per-task saturation gates fail. Additional target rows change class balance and ranking non-monotonically; no checkpoint or budget was selected post hoc.

### Per-task required budget

| target | B* | B=0 SR | B=10 SR | B=20 SR | B=30 SR | B=60 SR |
|---|---:|---:|---:|---:|---:|---:|
| task0 | 20 | 41.67% | 97.22% | 93.52% | 97.22% | 91.67% |
| task1 | 60 | 52.78% | 94.44% | 56.48% | 58.33% | 87.96% |
| task5 | 60 | 98.15% | 100.00% | 79.63% | 64.81% | 96.30% |
| task6 | 30 | 100.00% | 86.11% | 86.11% | 90.74% | 92.59% |

task5/task6 的 B=0 数值很高，但仍不能叫 zero-shot transfer，因为 target one-hot 未训练；它可能反映 source-learned shared bias与该 target archived difficulty，而非语义迁移。task6 B=10/B=20 的 target adaptation labels全为 success，不能作为有效 supervised budget。

## Boundary seeking is more query-efficient in aggregate, not on every task

At B=20 the sequential policy obtains **1.42** adjacent brackets/split versus **0.51** for 100 fixed outcome-blind hash permutations. The advantage is driven by task0/task5; task1 and task6 do not beat the random reference at B=20 because recoverable boundaries are sparse. Every F3→F2/F1 or F3→F4/F5 query is charged and retained.

Exact success/failure counts, roots, contexts, forces, brackets, and one-class flags are in `TARGET_FEWSHOT_LABEL_BALANCE.csv`. The target pools are nested prefixes of one frozen query trajectory; no unqueried outcome selects a sample.

## Task1 remains label-limited

Task1 retains 140/180 reconstructed terminal labels. The selected-branch direct-label-only sensitivity is reported for all 45 method/budget/seed cells in `TASK1_DIRECT_LABEL_SENSITIVITY.csv`; denominators vary with the force each controller selects, so this is a sensitivity check, not a replacement headline. Task1 is retained in every macro result.

## Direct answers to the 18 preregistered questions

1. **Can Probe predict friction without target labels?** It can produce predictions, but not accurately enough to call the learned estimator task-general: LOTO MAE is 0.5149.
2. **LOTO Learned Probe MAE:** 0.5149 (three-seed/task macro).
3. **Drop from pooled OOF 0.0655:** MAE worsens by +0.4494 (7.9× the error).
4. **Explicit SysID MAE:** 0.1810.
5. **SysID–learned gap:** SysID MAE is 0.3340 lower.
6. **Can learned Probe be deleted?** Not yet: SysID wins μ estimation but not the B=60 controller SR/utility; estimator MAE and force-decision value disagree.
7. **Is learned mapping worth retaining?** As a task-general friction estimator, current evidence says no; as a downstream feature, the matched controller still performs better than SysID at B=60, so replacement requires a frozen controller confirmation.
8–12. **Shared Direct macro SR at B=0/10/20/30/60:** 73.15%, 94.44%, 78.94%, 77.78%, 92.13%.
13. **Smallest B*:** macro B*=60.
14. **Can 60 be reduced universally?** No. The curve and gates require B=60.
15. **Per-task B*:** task0=20, task1=60, task5=60, task6=30. None receives a formal semantic 0-shot/10-shot label.
16. **Target success/failure counts:** exact fold-level counts are frozen in `TARGET_FEWSHOT_LABEL_BALANCE.csv`; task6 B10/B20 are one-class.
17. **Boundary policy vs random:** aggregate B20 brackets 1.42 vs 0.51, but the advantage is not uniform across tasks.
18. **Does source+20 match all-4 Sparse60?** No: Learned-Probe B20 is 78.94% versus the existing all-4 reference 90.05%, a -11.11 pp gap.

## Validation, limitations, and next step

All 108 shards completed. Every estimator×target×budget×seed has 36 evaluation episodes; each fold has 12 episodes from 2 unseen roots, 6 contexts, and 2 repeats. Acquisition/adaptation roots never overlap evaluation roots. GT self-agreement is 100%. Full checks are in `TRANSFER_DATA_QA.json`.

This experiment is descriptive development on archived task0/1/5/6 data. It does not separate unseen task from unseen object family, does not establish semantic zero-shot transfer, and contains reconstructed task1 outcomes. The appropriate next step is not a new model: first decide whether to replace the task one-hot with an already-legitimate semantic representation in a separately preregistered experiment; under the frozen architecture, the honest conclusion is substantial task-specific data required.

## Classification

**SUBSTANTIAL_TASK_SPECIFIC_DATA_REQUIRED**

No World Model, residual, Probe modification, new simulator rollout, or root-scaling untouched TEST result was used.


Source reused without rerun: `/home/exouser/FORTE/activeforcing_shared_physical_transfer_20260901_094722/NEW_TASK_FEWSHOT_TRANSFER_AGG.csv`.
