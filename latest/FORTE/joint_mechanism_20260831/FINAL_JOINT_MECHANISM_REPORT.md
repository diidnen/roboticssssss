# FINAL JOINT MECHANISM REPORT

## Answer first

**Classification: Case C — `OLD_JOINT_GAIN_DOES_NOT_TRANSFER_TO_CURRENT_DATA_REGIME`.**

Joint-NoVisual reproduces the central generalization failure without any RGB/PCA input: TRAIN fitting improves sharply, but root-heldout NLL worsens on all four single tasks (4/4). Visual Joint also worsens relative to Full on all four tasks (4/4), so visual conditioning is an additional instability path, especially on task0/task1, but it is not the sole or primary sufficient explanation.

Increasing pooled training diversity from 12 training contexts/fold in single-task CV to 48 training contexts/fold partially helps task5/task6 NLL, but it helps only 2/4 tasks and worsens under-force on 4/4. Therefore pooled diversity does not rescue a reliable Joint, and `DATA_TOO_SMALL` is not established.

Prospective evidence reaches the same decision despite mixed individual metrics: single-task JNV NLL is worse than Base on 3/4 tasks, and matched pooled JNV NLL is worse on 3/4. Its only matched-pooled NLL recovery is task6, which has one DEV root and cannot establish multi-context recovery.

## Five direct answers

1. **Why did old Joint look better?** It localized the frontier on a small old benchmark (0.325→0.125 N) in a no-visual, pooled 72-context regime with 288 extra coarse BCE branches. That gain was narrow: safety, Brier/NLL and monotonicity were worse. Current matched pooled evidence shows context diversity alone does not reproduce a reliable advantage.
2. **Is current Joint failure mainly visual shortcut or physics auxiliary?** The physics auxiliary itself is implicated because JNV fails held-root NLL on 4/4 tasks and worsens safety/shape metrics. Visual conditioning adds further failure—Visual Joint is worse than Full on 4/4 tasks—but “visual shortcut alone” is rejected.
3. **Can more pooled contexts rescue Joint?** Root-CV partially improves NLL on task5/task6, but prospective pooled DEV remains worse on task0/task1/task5; only single-root task6 improves. Safety is not consistently recovered. The answer for a controller-quality Joint is no.
4. **Is Joint still an effective paper novelty?** Not as a validated controller or generalizing feasibility model. It can be retained only as a frontier-localization/world-model diagnostic signal or as an explicitly negative mechanistic result; it should not carry the main effectiveness claim.
5. **Next: add contexts or delete/simplify Joint?** Delete/simplify Joint from the controller-facing method first. Audit old supervision/distribution/evaluator and the current auxiliary's overconfidence before collecting substantially more roots. A later preregistered context learning curve may be useful, but the present pooled result does not justify “more contexts” as the next primary fix.

## Scope

This study tests **same task, same object/task distribution, held-out physical root/context**. It does not test unseen-task, cross-object or universal physical reasoning. Pooled task0/1/5/6 training is evaluated separately within each task and is not unseen-task generalization.

## Frozen design

- Single task: 18 contexts, six roots, 180 branches; CV trains on 12 contexts/four roots and holds six contexts/two roots.
- Pooled CV: 72 total contexts/24 roots; each fold trains on 48 contexts/16 roots and holds 24 contexts/eight roots, reported taskwise.
- Joint-NoVisual input equals Base input; no RGB, frozen visual feature or PCA visual representation.
- Physics auxiliary: hidden64 H8×13 authoritative Physics-GRU, true matched trajectory-difference IE, λphysics=1, λIE=1, λfeas=0.3.
- AdamW 8e-4/1e-4, gradient clip1, 80 epochs, seeds0/1/2; CV seed0 matches the existing retrospective protocol.
- No DEV selection, tuning, best-seed choice, Probe, force-grid change or architecture/loss change.

## TRAIN → held-root evidence

| task | model | TRAIN_BCE | heldout_NLL | gap | ratio |
| --- | --- | --- | --- | --- | --- |
| 0 | Base | 0.1444 | 0.1626 | 0.0182 | 1.1263 |
| 0 | Full | 0.0949 | 0.2148 | 0.1199 | 2.2633 |
| 0 | Visual Joint | 0.0611 | 0.4806 | 0.4196 | 7.8704 |
| 0 | Joint-NoVisual | 0.061 | 0.1901 | 0.1292 | 3.1195 |
| 1 | Base | 0.1197 | 0.1607 | 0.041 | 1.3425 |
| 1 | Full | 0.0557 | 0.2471 | 0.1914 | 4.4386 |
| 1 | Visual Joint | 0.036 | 1.0731 | 1.0371 | 29.8151 |
| 1 | Joint-NoVisual | 0.0597 | 0.267 | 0.2073 | 4.4706 |
| 5 | Base | 0.0997 | 0.158 | 0.0583 | 1.5844 |
| 5 | Full | 0.0702 | 0.2311 | 0.1608 | 3.2904 |
| 5 | Visual Joint | 0.029 | 0.2857 | 0.2567 | 9.8433 |
| 5 | Joint-NoVisual | 0.0355 | 0.3015 | 0.2661 | 8.5027 |
| 6 | Base | 0.071 | 0.1716 | 0.1006 | 2.4177 |
| 6 | Full | 0.0641 | 0.1843 | 0.1202 | 2.8767 |
| 6 | Visual Joint | 0.0487 | 0.2605 | 0.2118 | 5.3486 |
| 6 | Joint-NoVisual | 0.0601 | 0.2715 | 0.2114 | 4.5207 |

Joint-NoVisual has a larger positive TRAIN→heldout NLL gap than Base on every task. Visual Joint has the largest or near-largest gap. This is direct evidence of context-dependent overfitting, not merely weak TRAIN optimization.

## Mechanism contrasts

All deltas are candidate minus comparator; positive NLL/Brier/frontier/under-force is worse.

| task | single_JNV_minus_Base_NLL | single_JNV_minus_Base_probability_MAE | single_JNV_minus_Base_Brier | single_JNV_minus_Base_frontier_MAE_N | single_JNV_minus_Base_under_force | VisualJoint_minus_Full_NLL | pooled_JNV_minus_Base_NLL | pooled_JNV_minus_Base_probability_MAE | pooled_JNV_minus_Base_Brier | pooled_JNV_minus_Base_frontier_MAE_N | pooled_JNV_minus_Base_under_force |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0276 | -0.0545 | -0.0042 | 0.0454 | 0.3333 | 0.2658 | 0.1662 | -0.0232 | 0.0136 | 0.0748 | 0.2222 |
| 1 | 0.1063 | -0.0545 | -0.0047 | -0.0006 | 0.0667 | 0.8259 | 0.1146 | -0.0238 | -0.0005 | 0.0332 | 0.1889 |
| 5 | 0.1435 | -0.0433 | 0.0046 | 0.0096 | 0.2222 | 0.0547 | -0.0056 | -0.0222 | -0.0007 | 0.0437 | 0.0556 |
| 6 | 0.0999 | -0.0289 | 0.0089 | 0.0302 | 0.0 | 0.0763 | -0.0542 | -0.0282 | -0.0055 | -0.0068 | 0.2222 |

JNV's probability MAE improves on all tasks even while NLL and safety worsen. This combination is consistent with an overly sharp force-response curve: average distance to empirical probabilities can fall while confident errors, threshold placement and under-force risk increase. Therefore no conclusion is based on probability MAE alone.

## Prospective taskwise DEV

Prospective DEV taskwise rows are complete and shown below:

| task | model | probability_MAE | Brier | NLL | frontier_MAE_N | under_force_rate | contexts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Base | 0.1824 | 0.1113 | 0.4814 | 0.4 | 0.5 | 2 |
| 0 | Full | 0.159 | 0.0904 | 0.3865 | 0.3 | 1.0 | 2 |
| 0 | Visual Joint | 0.211 | 0.1446 | 1.17 | 0.45 | 1.0 | 2 |
| 0 | Joint-NoVisual | 0.1536 | 0.082 | 0.3131 | 0.375 | 0.5 | 2 |
| 1 | Base | 0.1133 | 0.034 | 0.1875 | 0.25 | 0.0 | 4 |
| 1 | Full | 0.0648 | 0.0249 | 0.1404 | 0.1333 | 0.3333 | 4 |
| 1 | Visual Joint | 0.0493 | 0.019 | 0.1428 | 0.1 | 0.3333 | 4 |
| 1 | Joint-NoVisual | 0.0877 | 0.0495 | 0.3208 | 0.1125 | 0.25 | 4 |
| 5 | Base | 0.1826 | 0.1091 | 0.3748 | 0.4 | 1.0 | 2 |
| 5 | Full | 0.2829 | 0.2214 | 0.9678 | 0.625 | 1.0 | 2 |
| 5 | Visual Joint | 0.3291 | 0.2964 | 2.4897 | 0.8 | 1.0 | 2 |
| 5 | Joint-NoVisual | 0.236 | 0.1884 | 1.5119 | 0.6 | 1.0 | 2 |
| 6 | Base | 0.0292 | 0.0021 | 0.116 | 0.1 | 0.0 | 1 |
| 6 | Full | 0.0113 | 0.0003 | 0.1076 | 0.05 | 0.0 | 1 |
| 6 | Visual Joint | 0.0155 | 0.0012 | 0.1033 | 0.1 | 0.0 | 1 |
| 6 | Joint-NoVisual | 0.0863 | 0.0372 | 0.1781 | 0.15 | 0.0 | 1 |

task6 has only one DEV root; it is reported but cannot independently support multi-context generalization. Root-heldout TRAIN CV is the primary task6 pattern test.

## Matched pooled prospective DEV

| task | model | probability_MAE | Brier | NLL | frontier_MAE_N | under_force_rate | context_monotonic | DEV_contexts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | POOLED_BASE | 0.0756 | 0.0374 | 0.1848 | 0.225 | 0.5 | 1.0 | 2 |
| 0 | POOLED_JOINT_NOVISUAL | 0.0742 | 0.042 | 0.4395 | 0.225 | 0.5 | 1.0 | 2 |
| 1 | POOLED_BASE | 0.0941 | 0.0455 | 0.2151 | 0.1833 | 0.0 | 1.0 | 4 |
| 1 | POOLED_JOINT_NOVISUAL | 0.0638 | 0.0384 | 0.2286 | 0.1 | 0.75 | 0.0 | 4 |
| 5 | POOLED_BASE | 0.0277 | 0.0049 | 0.0641 | 0.1 | 0.5 | 1.0 | 2 |
| 5 | POOLED_JOINT_NOVISUAL | 0.0301 | 0.0149 | 0.0866 | 0.05 | 1.0 | 1.0 | 2 |
| 6 | POOLED_BASE | 0.0383 | 0.0059 | 0.1516 | 0.0 | 0.0 | 1.0 | 1 |
| 6 | POOLED_JOINT_NOVISUAL | 0.0163 | 0.0013 | 0.1055 | 0.0 | 0.0 | 1.0 | 1 |

Matched pooled JNV has worse NLL on 3/4 tasks (task0/task1/task5). It improves task6 only, where the single DEV root forbids a multi-context recovery claim. Under-force worsens on 2/4 tasks and is unchanged on the others.

## Original pooled Visual pipeline

| task | model | probability_MAE | Brier | NLL | frontier_MAE_N | under_force_or_missing_rate | monotonic_context_rate | implementation_confounded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | PROSPECTIVE_BASE_FEAS | 0.2481 | 0.1165 | 0.3991 | 1.0 | 1.0 | 1.0 | True |
| 0 | VISUAL_INTERCEPT_RESIDUAL | 0.1992 | 0.0689 | 0.3002 | 1.0 | 1.0 | 1.0 | True |
| 0 | VISUAL_CONTEXT_FULL_FEAS | 0.2184 | 0.0944 | 0.3441 | 1.0 | 1.0 | 1.0 | True |
| 0 | VISUAL_CONTEXT_JOINT | 0.3872 | 0.2266 | 0.6806 | 1.0 | 1.0 | 1.0 | True |
| 1 | PROSPECTIVE_BASE_FEAS | 0.3898 | 0.2014 | 0.6253 | 1.125 | 1.0 | 1.0 | True |
| 1 | VISUAL_INTERCEPT_RESIDUAL | 0.3469 | 0.1624 | 0.5324 | 1.125 | 1.0 | 1.0 | True |
| 1 | VISUAL_CONTEXT_FULL_FEAS | 0.2838 | 0.1132 | 0.4203 | 1.125 | 1.0 | 1.0 | True |
| 1 | VISUAL_CONTEXT_JOINT | 0.3524 | 0.1581 | 0.5271 | 0.7375 | 0.75 | 1.0 | True |
| 5 | PROSPECTIVE_BASE_FEAS | 0.3916 | 0.2242 | 0.6497 | 1.5 | 1.0 | 1.0 | True |
| 5 | VISUAL_INTERCEPT_RESIDUAL | 0.4958 | 0.3686 | 1.003 | 1.5 | 1.0 | 1.0 | True |
| 5 | VISUAL_CONTEXT_FULL_FEAS | 0.4141 | 0.2599 | 0.725 | 1.5 | 1.0 | 1.0 | True |
| 5 | VISUAL_CONTEXT_JOINT | 0.5053 | 0.3307 | 0.894 | 1.5 | 1.0 | 1.0 | True |
| 6 | PROSPECTIVE_BASE_FEAS | 0.043 | 0.0038 | 0.1349 | 0.0 | 0.0 | 1.0 | True |
| 6 | VISUAL_INTERCEPT_RESIDUAL | 0.0472 | 0.0052 | 0.1478 | 0.0 | 0.0 | 1.0 | True |
| 6 | VISUAL_CONTEXT_FULL_FEAS | 0.0426 | 0.0046 | 0.1394 | 0.0 | 0.0 | 1.0 | True |
| 6 | VISUAL_CONTEXT_JOINT | 0.0514 | 0.0064 | 0.1676 | 0.0 | 0.0 | 1.0 | True |

These rows are implementation-confounded and cannot isolate context diversity: the original pooled Joint uses a repeated H8×4 target, no authoritative Physics-GRU trajectory head, and an adjacent-prediction-difference penalty rather than target-matched IE.

Even under that different formulation, Visual Joint has worse taskwise NLL than Full on 4/4 tasks. Its aggregate frontier improvement therefore remains a narrow localization signal rather than a calibrated feasibility-model win.

## Hypothesis adjudication

- **Hypothesis A, visual shortcut:** supported as an additional path/interaction, because Visual Joint is worse than Full on every task and task0/task1 show especially large gaps. Rejected as the sole explanation because no-visual Joint also fails.
- **Hypothesis B, sample-limited physics auxiliary:** plausible at 18 contexts, and pooled task5/task6 probability results show partial regularization. Not proven as the main explanation because pooled JNV does not recover task0/task1 and worsens under-force across tasks.
- **Most defensible case:** Case C. The old frontier signal does not transfer into a consistently calibrated and safe current formulation, including the higher-context pooled regime.

## Data quality and limitations

- task0/task5/task6 labels are direct; task1 has 140/180 reconstructed TRAIN labels and remains caveated.
- task5/task6 direct-label replications show the pattern is not reducible to task1 label recovery.
- CV frontier estimates use five random forces × two repeats and are noisier than prospective DEV.
- Original pooled visual results are not a matched formulation.
- The partial old nested-context run is excluded because its task0-only setting was evaluated on all-task DEV and the nested schedule did not complete.

## Decision

Do not use current Joint as the paper's controller-quality novelty and do not justify another large task0-root collection by saying “18 contexts is too small.” Retain Base/Full as the empirical controller baselines; keep the physics auxiliary only as a diagnostic research object until it passes same-task new-root calibration, frontier, under-force and monotonicity together.
