# 1. Scientific Contract

This is a prospectively frozen secondary-only online-VLA evaluation on roots 170050 and 170051. It does not alter or pool into the locked 8-root main table. All 72 admitted branches use checkpoint `0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17`, predict-50/execute-10/re-query-10, the frozen P4-B probe, controller, and full-task evaluator. The primary force statistic is recomputed per branch over `branch_step >= 1 AND vla_release_intent = false` as `2*min(|N_left|, |N_right|)`, including zero-contact frames, before equal branch averaging.

# 2. True NoProbe Validity

`TRUE_NOPROBE_PATHWAY_EXISTS = NO`. The frozen decision path consumes the real probe trace/contact readback and post-probe decision state. No frozen no-observation belief/state adapter exists. Zero-filled probe features, a fabricated trace, or a pre-probe state passed into the post-probe model would violate the stated contract. Prior/No-Posterior is not relabeled as NoProbe; no NoProbe physics was run.

# 3. GT-Physics Definition

`GT_PHYSICS_PATHWAY_VALID = YES`. GT-Physics executes the same probe and downstream runtime. Its only planner change is replacing the physical posterior quadrature by `delta(mu - mu_GT)` using privileged simulator friction. The frozen feasibility model, utility, dense force support, controller, VLA, and evaluator are unchanged.

# 4. Tabero-Neutral Validity

`TABERO_NEUTRAL_MATCHED_VALID = YES` as a separately captioned external baseline. It shares the established-grasp/post-probe state, instruction, checkpoint, online observations and re-query schedule, first-six arm-action semantics, release gate, evaluator, and force measurement. Its gripper/controller pathway intentionally differs: during grasp it preserves the VLA-native aperture and six fingertip-force outputs. Therefore this comparison must not be described as differing only in scalar force choice.

# 5. New Secondary Root Results

| Variant | Full SR | Lift SR | Drop rate | Mean selected F (N) | Mean measured squeeze (N) |
|---|---|---|---|---|---|
| Full ActiveForcing | 20/24 (83.3%) | 100.0% | 0.0% | 3.952 | 3.417 |
| GT Physics (privileged friction) | 20/24 (83.3%) | 100.0% | 0.0% | 3.917 | 3.625 |
| Tabero Neutral | 19/24 (79.2%) | 100.0% | 0.0% | N/A | 42.274 |

Per-root descriptive results:

| root | Variant | Full SR | Mean measured squeeze (N) |
|---|---|---|---|
| 170050 | Full ActiveForcing | 10/12 (83.3%) | 3.393 |
| 170050 | GT Physics (privileged friction) | 10/12 (83.3%) | 3.575 |
| 170050 | Tabero Neutral | 9/12 (75.0%) | 43.456 |
| 170051 | Full ActiveForcing | 10/12 (83.3%) | 3.441 |
| 170051 | GT Physics (privileged friction) | 10/12 (83.3%) | 3.674 |
| 170051 | Tabero Neutral | 10/12 (83.3%) | 41.093 |

# 6. Physics Information Ladder

The legally available ladder is ActiveForcing versus privileged GT-Physics; True NoProbe is absent because the frozen runtime has no valid no-observation pathway. GT-Physics and ActiveForcing are close on this 24-context subset; perfect friction information exposes little observed planner headroom here.

# 7. GT Gap Analysis

GT minus AF full-task SR is +0.0 percentage points. GT minus AF mean selected force is -0.035 N; GT minus AF measured force is +0.208 N. Paired counts are both success=18, AF only=2, GT only=2, both fail=2.

All four AF failures and all four GT failures occurred after a successful lift, with zero recorded drops. The aggregate tie plus the two-for-two paired swaps therefore does not identify friction estimation as the dominant remaining error source; post-lift geometry and online execution variation remain visible.

# 8. Probe Value Analysis

The value of probing is not causally measured in this round because True NoProbe was invalid and skipped. AF-versus-GT isolates posterior estimation quality conditional on the same probe; it does not estimate probe-versus-no-probe reliability or overhead benefit.

# 9. Tabero Comparison

Tabero minus AF full-task SR is -4.2 percentage points and Tabero minus AF mean measured squeeze is +38.857 N. Paired counts are both success=15, AF only=5, Tabero only=4, both fail=0. This is an online matched-scope external comparison with a different native gripper/controller interface.

# 10. Per-Task Results

| task | Variant | Full SR | Mean measured squeeze (N) |
|---|---|---|---|
| 0 | Full ActiveForcing | 6/6 (100.0%) | 4.237 |
| 0 | GT Physics (privileged friction) | 4/6 (66.7%) | 4.261 |
| 0 | Tabero Neutral | 4/6 (66.7%) | 44.452 |
| 1 | Full ActiveForcing | 4/6 (66.7%) | 3.329 |
| 1 | GT Physics (privileged friction) | 4/6 (66.7%) | 3.291 |
| 1 | Tabero Neutral | 6/6 (100.0%) | 45.600 |
| 5 | Full ActiveForcing | 6/6 (100.0%) | 4.106 |
| 5 | GT Physics (privileged friction) | 6/6 (100.0%) | 4.040 |
| 5 | Tabero Neutral | 3/6 (50.0%) | 43.847 |
| 6 | Full ActiveForcing | 4/6 (66.7%) | 1.996 |
| 6 | GT Physics (privileged friction) | 6/6 (100.0%) | 2.907 |
| 6 | Tabero Neutral | 6/6 (100.0%) | 35.197 |

# 11. Per-Friction Results

| friction | Variant | Full SR | Mean measured squeeze (N) |
|---|---|---|---|
| LOW | Full ActiveForcing | 4/8 (50.0%) | 2.795 |
| LOW | GT Physics (privileged friction) | 6/8 (75.0%) | 3.617 |
| LOW | Tabero Neutral | 6/8 (75.0%) | 43.974 |
| MID | Full ActiveForcing | 8/8 (100.0%) | 4.044 |
| MID | GT Physics (privileged friction) | 8/8 (100.0%) | 4.104 |
| MID | Tabero Neutral | 6/8 (75.0%) | 39.035 |
| HIGH | Full ActiveForcing | 8/8 (100.0%) | 3.412 |
| HIGH | GT Physics (privileged friction) | 6/8 (75.0%) | 3.154 |
| HIGH | Tabero Neutral | 7/8 (87.5%) | 43.813 |

# 12. What These Experiments Add to the Paper

- A privileged-information test separating friction-estimation headroom from the frozen feasibility/utility/VLA stack.
- A real online Tabero-native gripper baseline on the same secondary contexts and checkpoint, with common full-task labels and measured-force semantics.
- Branch-level provenance for every result and paired context analysis.

# 13. What They Do NOT Prove

- They do not establish the causal value or cost-benefit of probing because no legal True NoProbe comparator exists.
- GT friction is an information upper bound, not an oracle outcome or oracle force label.
- Two secondary roots do not support strong population-level significance claims.
- Tabero is not an only-force-selector ablation because its gripper/controller interface differs.

# 14. Recommended Main-Paper / Appendix Placement

Place the AF-versus-GT information analysis in the main paper if space permits because it diagnoses physical-information headroom. Put the matched Tabero comparison and per-task/per-friction breakdown in the appendix or supplementary material, with the controller-interface difference stated in the caption. Report True NoProbe as not implemented under the frozen scientific contract, rather than substituting Prior.
