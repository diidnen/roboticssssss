# Global force scale and Utility calibration study

## Executive verdict

`UTILITY_FORCE_SCALE_CAUSAL_DIAGNOSIS_COMPLETE`

`PROBLEM_CLASS = MIXED_DIRECT_AND_UTILITY_PROBLEM`

`METHOD_PROMOTION = NO`

Task-specific candidate-grid normalization is physically weakly grounded and does over-penalize a Newton on task6 relative to the other tasks. However, the frozen paired evidence does not show that it is the main cause of task6’s 86.11% result. The primary 8 N global scale rescues only one of five failures and reaches 88.89%, not 100%. Absolute force cost also rescues only one task6 failure and introduces two task1 collateral failures. The evidence is insufficient to promote `ACTIVEFORCING_GLOBAL_SCALE_UTILITY` as a new method.

## Fixed study design

The study reuses 720 archived candidate branches: 4 tasks × 18 contexts × 2 repeats × 5 candidates. It holds fixed π0, P4-B query, friction estimator, Direct checkpoints and predicted probabilities, candidates, episode tuples, root split, labels, and controller semantics. Only the force-cost definition inside the selector changes. No model was trained and no simulator was launched.

Primary definitions:

- A: `p*(1-F/Fmax_task)+(1-p)*(-1)`
- B: `p*(1-F/8)+(1-p)*(-1)`
- C: `p*(1-clip((F-3)/(8-3),0,1))+(1-p)*(-1)`; no archived candidate required clipping
- D: `p-F/16`; lambda follows `1/(2*Fref)` with primary `Fref=8`
- E: `p*(8-F)+(1-p)*(-8)`, an analytical GNP-style ablation equivalent to `8*B`

Eight Newtons is selected from an existing project-wide 3–8 N action support and an executed 8 N controller setpoint, not from task6 outcomes. It is not claimed to be a certified safety limit. Global sensitivity uses only `{5,6,8}` N from historical supports. Absolute-cost sensitivity uses `lambda={1/10,1/12,1/16}` by the predeclared `1/(2*Fref)` mapping.

## Main results

| Method | Macro/micro SR | Mean force | Under-force | Decision change vs A | Rescue | Collateral |
|---|---:|---:|---:|---:|---:|---:|
| A task-specific | 93.06% | 4.0628 N | 5.56% | 0% | 0 | 0 |
| B global 8 N | 94.44% | 4.1163 N | 4.17% | 19.44% | 2 | 0 |
| C global range 3–8 N | 94.44% | 4.0907 N | 4.17% | 9.72% | 2 | 0 |
| D absolute `p-F/16` | 93.06% | 4.1234 N | 5.56% | 27.78% | 2 | 2 |
| Success-Only diagnostic | 98.61% | 4.8213 N | 0% | 83.33% | 8 | 0 |
| Fixed-Max diagnostic | 98.61% | 4.8329 N | 0% | 83.33% | 8 | 0 |

B’s two rescues are one task0 episode and one task6 episode. Its task1 and task5 SRs are unchanged. This is a favorable directional result, but it is small and does not isolate scale as task6’s dominant mechanism.

## Per-task A versus B

| Task | A SR | B SR | A force | B force | B decision changes | B rescue/collateral |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 97.22% | 100.00% | 4.0714 | 4.1469 | 22.22% | 1 / 0 |
| 1 | 94.44% | 94.44% | 5.0779 | 5.0779 | 0% | 0 / 0 |
| 5 | 94.44% | 94.44% | 3.8781 | 3.9167 | 11.11% | 0 / 0 |
| 6 | 86.11% | 88.89% | 3.2237 | 3.3238 | 44.44% | 1 / 0 |

Global-reference sensitivity is not needle-like: 5 N reproduces A’s 93.06% macro SR, while 6 N and 8 N both yield 94.44%. Task6 is 86.11% at 5 N and 88.89% at 6/8 N. Thus the small rescue does not require exactly 8 N, but no tested global reference repairs the other four failures.

## Cost geometry

The marginal normalized force cost is `1/Fmax_task` per Newton: task6 pays 0.25/N, task0 and task5 pay 0.20/N, and task1 pays 0.1667/N. The same 1 N physical increment therefore carries a 50% larger normalized cost on task6 than on task1. B makes the slope 0.125/N for all tasks; C also uses a common 0.20/N slope after a 3 N zero point.

This geometry explains why task6 shifts more often under B. It does not by itself establish that the shift reaches the right branch: 16/36 task6 decisions change, yet only one failed episode is rescued.

## Direct versus Utility

All five current task6 failures select the lowest candidate, approximately 3.01–3.06 N, with Direct p(success) 0.9394–1.0000. All have higher-force paired successes; Fixed-Max and Success-Only both succeed 36/36. GT friction selects the same failed branch, excluding the identifier as the primary explanation.

The operational attribution is 1/5 failures scale-rescuable and 4/5 residual after global scaling. The most defensible conclusion is that Direct’s localized low-boundary overconfidence is primary, while task-dependent normalization is a secondary amplifier. Aggregate task6 calibration looks mildly conservative, so the error is conditional and would be missed by task-average reliability alone.

## Absolute cost

Absolute cost is not more stable in this archive. Every tested lambda rescues one task6 failure, but also creates two task1 collateral failures; macro SR remains 93.06%. Its task6 benefit therefore does not survive the required cross-task guardrail. D2 with `lambda=1/8` and failure penalty `-1` would be algebraically identical to B, so it is not independent evidence.

## Direct answers

1. **Why did task6 fall from 100% to about 86%?** Full Utility chose the lowest-force branch in five episodes because Direct assigned those branches 0.9394–1.0000 success probability and force cost favored them. Success-Only/Fixed-Max moved to successful higher-force branches.
2. **What is Fmax=4?** A historical task6 robust-force and candidate/collection support endpoint, later reused as a Utility denominator—not a recovered physical safety limit.
3. **Does task-specific Fmax over-penalize high force on task6?** Geometrically yes; causally it is a secondary contributor, not the dominant explanation.
4. **Does a global physical scale fix task6?** Partially: 86.11%→88.89%, one of five failures rescued. It does not restore 100%.
5. **Does it harm tasks 0/1/5?** B does not reduce archived SR: task0 gains one success; task1/task5 are unchanged. Mean force rises modestly on tasks 0 and 5. This is development evidence only.
6. **Is absolute force cost more stable?** No. It produces two task1 collateral failures and no net macro-SR gain.
7. **How much comes from Direct versus scale?** On the five task6 failures, the global-scale intervention accounts for one rescue (20%); four remain (80%). This is case accounting, not a statistical variance decomposition.
8. **Is this general enough for a method contribution?** Not yet. The geometry critique is general, but the measured benefit is too small and Direct calibration remains the main bottleneck.
9. **Is it only a task6 special case?** The normalization mismatch is structurally cross-task, but the observed failure concentration is task6-local and insufficient for promotion. A task6-specific Fmax change is explicitly rejected.

## Promotion and fresh validation

`METHOD_PROMOTION = NO`; therefore no `ACTIVEFORCING_GLOBAL_SCALE_UTILITY.json` is created and no simulator validation is authorized by this study. If future TRAIN/DEV work first improves low-boundary Direct calibration and then pre-registers a global-scale candidate, validation must use genuinely fresh physical roots across tasks 0/1/5/6 with frozen π0, query, identifier, Direct, candidate support, and baselines. Previously viewed E1/720 roots cannot be relabeled as sealed TEST.

## Evidence limitations

- These are already-observed grouped-root OOF TRAIN/DEV branches, not fresh TEST.
- There are 36 episodes and six root families per task; uncertainty is root-clustered and coarse.
- Paired archived branches identify within-support selector counterfactuals, not behavior outside the five candidates.
- Task1 retains the prior 140/180 reconstructed-label caveat; conclusions that depend on task1 collateral require fresh direct labels.
- No controller-certified safe global range was found; 8 N is an authoritative project action-support endpoint, not safety certification.

