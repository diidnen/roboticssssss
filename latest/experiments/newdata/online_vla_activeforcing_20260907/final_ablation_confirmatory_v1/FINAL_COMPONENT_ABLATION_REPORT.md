# Final Confirmatory Component Ablation Report

## Evidence contract

The new table contains exactly 24 untouched ablation-only contexts (roots 170048 and 170049) and 120 online frozen-VLA physical branches. The five variants use the pre-frozen execution order. All admitted branches passed online-action provenance, checkpoint, action-arbitration, geometric-label, source-stability, selector-reproduction, initial-state/chunk identity, and common-random-number checks.

The sole launcher change was the documented multi-signal GPU health gate. It did not alter the VLA, probe, posterior, feasibility models, utility, controller, evaluator, roots, or execution plan.

One branch attempt was quarantined after ENOSPC prevented complete provenance writes; its directory was preserved and the frozen retry policy permitted one exact retry. Thus there are 120 admitted physical branches and 121 physical attempts in total.

## New 24-context confirmatory result

| Variant | Success | Full SR | Lift SR | Drop | Selected F (N) | Measured squeeze (N) | Observed utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full ActiveForcing | 19/24 | 0.792 | 1.000 | 0.083 | 3.981 | 3.559 | -0.043 |
| Posterior Mean | 18/24 | 0.750 | 0.958 | 0.125 | 3.840 | 3.268 | -0.075 |
| Prior / No-Posterior | 19/24 | 0.792 | 1.000 | 0.125 | 4.150 | 3.759 | -0.072 |
| Coarse Grid {3,4,5} | 16/24 | 0.667 | 1.000 | 0.167 | 4.042 | 3.508 | -0.192 |
| Local-Lift | 9/24 | 0.375 | 0.917 | 0.500 | 3.092 | 1.597 | -0.475 |

## Paired results

- FULL_AF vs Posterior Mean: both success 17, Full-AF only 2, ablation only 1, both fail 4; mean measured-force difference (Full minus ablation) 0.291 N.
- FULL_AF vs Prior / No-Posterior: both success 18, Full-AF only 1, ablation only 1, both fail 4; mean measured-force difference (Full minus ablation) -0.200 N.
- FULL_AF vs Coarse Grid {3,4,5}: both success 15, Full-AF only 4, ablation only 1, both fail 4; mean measured-force difference (Full minus ablation) 0.050 N.
- FULL_AF vs Local-Lift: both success 8, Full-AF only 11, ablation only 1, both fail 4; mean measured-force difference (Full minus ablation) 1.961 N.

## Per-root result

| Root | Variant | Success | SR | Selected F (N) | Measured squeeze (N) |
|---:|---|---:|---:|---:|---:|
| 170048 | Full ActiveForcing | 10/12 | 0.833 | 3.925 | 3.430 |
| 170048 | Posterior Mean | 9/12 | 0.750 | 3.712 | 2.908 |
| 170048 | Prior / No-Posterior | 10/12 | 0.833 | 4.092 | 3.626 |
| 170048 | Coarse Grid {3,4,5} | 8/12 | 0.667 | 3.917 | 3.143 |
| 170048 | Local-Lift | 5/12 | 0.417 | 3.058 | 1.613 |
| 170049 | Full ActiveForcing | 9/12 | 0.750 | 4.037 | 3.687 |
| 170049 | Posterior Mean | 9/12 | 0.750 | 3.967 | 3.628 |
| 170049 | Prior / No-Posterior | 9/12 | 0.750 | 4.208 | 3.891 |
| 170049 | Coarse Grid {3,4,5} | 8/12 | 0.667 | 4.167 | 3.874 |
| 170049 | Local-Lift | 4/12 | 0.333 | 3.125 | 1.582 |

## Adaptation and failure interpretation

The frozen 120-branch plan contains no Fixed-3/4/5 anchor branches, so minimum-successful-fixed-anchor, easy-save, and hard-rescue metrics are not identifiable on these new roots. They are reported as unavailable rather than inferred from ablation outcomes. The table includes the pre-action friction-condition Spearman statistic as an explicitly labeled adaptation diagnostic.

- Full ActiveForcing: friction-difficulty Spearman rho=0.484; under-force-classified failures=2; post-lift geometric failures=3; other failures=0.
- Posterior Mean: friction-difficulty Spearman rho=0.569; under-force-classified failures=4; post-lift geometric failures=2; other failures=0.
- Prior / No-Posterior: friction-difficulty Spearman rho=0.129; under-force-classified failures=3; post-lift geometric failures=2; other failures=0.
- Coarse Grid {3,4,5}: friction-difficulty Spearman rho=0.349; under-force-classified failures=4; post-lift geometric failures=4; other failures=0.
- Local-Lift: friction-difficulty Spearman rho=0.321; under-force-classified failures=14; post-lift geometric failures=1; other failures=0.

## Claim audit

- `posterior_uncertainty`: **MIXED**. Full posterior was 19/24 versus 18/24 for posterior mean, but used 0.291 N more measured squeeze. The paired success discordance was 2 versus 1. This subset does not show a clear unqualified uncertainty benefit.
- `instance_posterior`: **SUPPORTED_FOR_FORCE_EFFICIENCY_AND_ADAPTATION_NOT_SUCCESS_RATE**. Full AF and Prior were both 19/24, while Full AF used 0.200 N less measured squeeze and had stronger friction-difficulty force correlation (rho 0.484 versus 0.129). This supports efficiency/adaptation, not a success-rate gain.
- `dense_force_search`: **SUPPORTED_WITH_MIXED_FORCE_EVIDENCE**. Full AF was 19/24 versus 16/24, with four Full-only and one coarse-only successes. Full AF selected 0.060 N less on average; all-branch measured squeeze was 0.050 N higher, while both-success squeeze was 0.065 N lower. The reliability/utility evidence supports dense search; force evidence is mixed.
- `full_task_supervision`: **SUPPORTED_FOR_FULL_TASK_RELIABILITY**. Full AF was 19/24 versus 9/24, drop rate was 0.083 versus 0.500, and under-force-classified failures were 2 versus 14. Local-Lift's lower force accompanied major reliability loss and is not an efficiency win.

These component conclusions are descriptive because there are two independent ablation root groups. Equality or a small count difference is not presented as population-level significance.

## Existing-24 pooling

`ABLATION_POOLING_VALID = YES`. Scientific runtime signatures were compared per variant, including checkpoint, controller configuration, feasibility manifest and derivation, phase representation, posterior interface, force grid, Local-Lift sources, arbitration, and frozen label.

The pooled 48-context table is descriptive: the first 24 contexts were prior burned-root evaluation and the new 24 are untouched confirmatory contexts.

| Variant | Success | Full SR | Lift SR | Drop | Selected F (N) | Measured squeeze (N) |
|---|---:|---:|---:|---:|---:|---:|
| Full ActiveForcing | 36/48 | 0.750 | 1.000 | 0.104 | 3.970 | 3.562 |
| Posterior Mean | 35/48 | 0.729 | 0.979 | 0.125 | 3.867 | 3.305 |
| Prior / No-Posterior | 36/48 | 0.750 | 1.000 | 0.125 | 4.140 | 3.712 |
| Coarse Grid {3,4,5} | 31/48 | 0.646 | 1.000 | 0.146 | 4.062 | 3.541 |
| Local-Lift | 19/48 | 0.396 | 0.958 | 0.500 | 3.094 | 1.667 |

## Final boundary

Physics stops here. No third ablation root, outcome-based repeat, new variant, or method adjustment is included or recommended as a remaining must-run experiment.
