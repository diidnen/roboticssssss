# ActiveForcing Utility and Causal Ablation Closure

## Status

**UTILITY_CAUSAL_ABLATIONS_COMPLETE**

The four requested ablations are complete on the existing 720-branch TRAIN archive using grouped-root out-of-fold predictions. No IsaacLab process, new pi0 rollout, sealed TEST read, large-model training, or GPU job was used.

## Frozen analysis contract

- Population: 720 post-query branches, 72 friction contexts, 24 task-specific roots, 144 paired context/repeat episodes, and exactly five archived continuous force candidates per episode.
- Query: the same 215-step P4-B action and post-query restored branch state. Query-Ignored executes the same query semantically but masks its physical response downstream.
- Direct: the same frozen grouped-root OOF three-seed Direct ensemble for every comparison.
- Posterior: three equally weighted grouped-root OOF friction-estimator members. Point evaluates utility at their mean; Posterior averages utility over the three member values.
- Authoritative utility: `U(F|x)=p(success)*(1-F/Fmax)+(1-p(success))*(-1)`, lower-force tie break, Fmax={0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}.
- Frozen utility config SHA-256: `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10`.
- Candidate-pool SHA-256: `fb2d96be8cf6576f116842684f34fd091b42a6af0e24f1f5b3248514c98d9916`.

The five archived candidates are context-specific continuous draws, not the later nine-point 0.25 N runtime grid. No nearest-force substitution is used, so all conclusions are archive-compatible offline results.

## A. Point estimate vs posterior expected utility

| method | n_episodes | full_task_SR | mean_selected_force_N | under_force_rate | mean_excess_force_N | mean_realized_utility | decision_changed_rate | point_fail_to_posterior_success_count | point_success_to_posterior_fail_count |
|---|---|---|---|---|---|---|---|---|---|
| Point | 144.0000 | 93.06% | 4.0628 | 5.56% | 0.2553 | 0.1056 | 8.33% | 1.0000 | 1.0000 |
| Posterior | 144.0000 | 93.06% | 4.0720 | 5.56% | 0.2642 | 0.1040 | 8.33% | 1.0000 | 1.0000 |

The comparison changes only how the same Direct and utility integrate friction uncertainty. It is a discrete ensemble posterior approximation, not a claim of calibrated Bayesian posterior density.

On this archive Posterior is neutral in net SR (+0.0000), with one rescue and one collateral; it increases mean force by +0.0092 N and changes realized utility by -0.0016. This is a negative result for a pooled posterior-utility advantage.

## B. Query-Ignored vs Active Query

| method | friction_MAE | friction_RMSE | mean_selected_force_N | full_task_SR | under_force_rate | mean_excess_force_N | paired_rescue_count | paired_collateral_count | decision_changed_rate |
|---|---|---|---|---|---|---|---|---|---|
| Query-Ignored | 0.2504 | 0.2868 | 4.4667 | 93.75% | 4.86% | 0.6704 | 4.0000 | 5.0000 | 69.44% |
| Active | 0.0655 | 0.0893 | 4.0628 | 93.06% | 5.56% | 0.2553 | 4.0000 | 5.0000 | 69.44% |

Because every archived branch follows P4-B, Query-Ignored is correctly interpreted as **no physical information downstream**, not as a zero-query action baseline. Same root, friction, initial condition, query action, and candidate semantics are verified in the CSV audit columns. Therefore the paired difference isolates information use rather than state change caused by the query action.

The physical response sharply improves friction MAE and reduces mean force by -0.4039 N and mean excess by -0.4151 N. It does not improve pooled SR: four paired rescues are offset by five collaterals (net -0.0069), while realized utility changes by +0.0638. Thus the information benefit is force efficiency/utility, not an established SR gain.

## C. Success-only Direct vs full utility

| method | full_task_SR | mean_selected_force_N | max_selected_force_N | under_force_rate | mean_excess_force_N | mean_realized_utility | confirmed_delayed_failure_lower_bound_rate | delayed_failure_metadata_coverage |
|---|---|---|---|---|---|---|---|---|
| Success-Only | 98.61% | 4.8213 | 5.9901 | 0.00% | 1.0092 | 0.0216 | 0.00% | 40.28% |
| Full Utility | 93.06% | 4.0628 | 5.9461 | 5.56% | 0.2553 | 0.1056 | 2.08% | 40.28% |
| Fixed-Max | 98.61% | 4.8329 | 5.9901 | 0.00% | 1.0210 | 0.0189 | 0.00% | 40.28% |

Delayed-failure rate is not exactly identifiable: 430/720 raw branch rows lack the required stage telemetry. The table therefore leaves the exact rate empty and reports only a confirmed lower bound plus observed metadata coverage. This negative estimability result is retained rather than filling missing outcomes.

Success-Only gains +0.0556 SR but uses +0.7585 N more mean force and changes authoritative realized utility by -0.0840. Full Utility therefore expresses the intended success-force tradeoff; it is not an SR-maximizing rule.

## D. DEV-only utility sensitivity

The small local grid, fixed in the runner before computation, varies force cost and failure penalty by +/-25% around the authoritative `(1.0, 1.0)` configuration. Across the nine pooled settings, SR ranges from 0.9167 to 0.9444, and mean force ranges from 4.0087 N to 4.1007 N. The frozen final point gives SR=0.9306 and mean force=4.0628 N, inside rather than at an extreme of both ranges. These are success-force tradeoff diagnostics only; no coefficient is selected from outcomes, and sealed TEST is untouched.

## Data and causal QA

- All 144 episode keys pair exactly across every compared policy; each policy selects from the identical five branch IDs for that key.
- Re-inference at GT friction reproduces the archived frozen Direct probabilities with maximum absolute error `2.97e-08`.
- Four archive audits record `TEST_read=false`, five forces x two repeats, and exact snapshot/state parity. The P4-B estimator predictions are grouped-root OOF.
- Task 1 retains the known material caveat: 140/180 terminal labels are reconstructed. No outcome-based protocol change was made.
- Archive telemetry is incomplete for exact delayed-failure classification, and this limitation is visible in the output rather than silently proxied.
- The analysis reuses frozen predictions/checkpoints; no estimator, Direct model, controller, utility coefficient, candidate set, or outcome label was retrained or changed.

## Deliverables

- `TABLE_POINT_VS_POSTERIOR.csv`
- `TABLE_QUERY_INFORMATION_ABLATION.csv`
- `TABLE_UTILITY_ABLATION.csv`
- `UTILITY_DEV_SENSITIVITY.csv`
- `UTILITY_FINAL_CONFIG.json`
- `UTILITY_CAUSAL_ABLATION_PER_EPISODE.csv`

UTILITY_CAUSAL_ABLATIONS_COMPLETE
