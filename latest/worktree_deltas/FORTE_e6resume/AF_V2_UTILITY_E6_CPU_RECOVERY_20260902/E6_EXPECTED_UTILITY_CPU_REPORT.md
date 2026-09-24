# E6 Expected-Utility member disagreement and second-query readiness

## Assessment

**CPU recompute complete; paired re-query evaluation blocked.** The final V2 Expected-Utility member decisions are now reproducible on the frozen 720-branch, 144-episode root-held-out OOF archive. The five-arm comparison is not ready because no genuine qualified second-query continuation or post-second-query posterior exists.

Legacy hard-rho outputs are explicitly excluded. No `FINAL_RHO.json`, rho-selection table, minimum-passing decision, or rho-induced disagreement row was loaded as evidence or used in a calculation.

## Expected-Utility disagreement

Every member selected `argmax_F U(F|z_m,x)` from the same five archived candidates for the same `(context_id, repeat)` pair. Utility is `p_success*(1-F/Fmax) + (1-p_success)*(-1)` and ties choose the lower force.

| scope | episodes | disagreement | rate | mean spread N | max spread N | posterior below frontier |
|---|---:|---:|---:|---:|---:|---:|
| POOLED | 144 | 36 | 25.00% | 0.0985 | 0.7315 | 5.56% |
| task0 | 36 | 10 | 27.78% | 0.1097 | 0.7171 | 0.00% |
| task1 | 36 | 8 | 22.22% | 0.1214 | 0.7187 | 2.78% |
| task5 | 36 | 10 | 27.78% | 0.0789 | 0.5220 | 5.56% |
| task6 | 36 | 8 | 22.22% | 0.0840 | 0.7315 | 13.89% |

Pooled disagreement is **36/144 (25.00%)**. Mean member-force spread is **0.0985 N**, with maximum **0.7315 N**. These are CPU/DEV diagnostics from grouped-root OOF folds, not a final paired controller outcome claim.

The empirical-frontier columns are retrospective diagnostics only. They never enter candidate selection or the re-query trigger.

## Source and calculation checks

- Archive grain: 720 unique branches = 144 paired episodes × five exact context-specific candidates; no duplicate branch or `(context, repeat, force)` key.
- Physical belief: three finite OOF member estimates for all 72 contexts.
- Direct: nine frozen fold/seed checkpoints were run on CPU; GT-physics reproduction maximum absolute error was `2.97e-08`.
- Utility file SHA-256: `19c93ad7a57ff5823c53e0171cf297424256d1bbac62df79b7736f0c680415e1`; candidate-pool SHA-256: `fb2d96be8cf6576f116842684f34fd091b42a6af0e24f1f5b3248514c98d9916`.
- Split isolation: source audits report no sealed TEST read; output split is `TRAIN_ROOT_HELDOUT_OOF_DEV_FOLDS`.
- Per-candidate `p_success`, rewards, Expected Utility, and member selections are retained in `E6_EXPECTED_UTILITY_MEMBER_SCORES.csv`.

## Second-query readiness

The prerequisite remains unavailable:

- The P7-B scientific run attempted 100 contexts but qualified 44, all TRAIN; qualified DEV = 0 and qualified TEST = 0. Its frozen gate classified the comparison as `SCIENTIFIC_COMPARISON_INVALID_BY_FROZEN_GATE`.
- The later physical-only protocol explicitly records `SECOND_QUERY_NOT_EVALUATED_NO_ARCHIVED_REPEAT`.
- Duplicate first-query telemetry is forbidden as a substitute for a second query.
- Therefore uncertainty reduction, posterior updating after query two, and the matched One/Always-Two/Raw-Uncertainty/Utility-Consensus/Oracle comparison are not estimable offline.

`E6_PAIRED_EVALUATION_MANIFEST.json` freezes the five arms, exact pair keys, candidate-pool identity, and fail-closed gate. `E6_PAIRED_EVALUATION_ROWS.csv` contains 720 planned pair-arm rows. It authorizes no GPU work and no rollout launch.

## Validation verdict

- Member disagreement recompute: **READY TO SHARE AS CPU/DEV DIAGNOSTIC**.
- Final five-arm E6 result: **NEEDS PREREQUISITE; NOT EVALUATED**.
- Second-query launch readiness: **BLOCKED** until a genuine continuation passes the frozen action/state/query qualification contract and produces a post-query posterior.
- Recovery action: **remain idle**; do not launch GPU or rollout work from this bundle.

## Required caveat

This bundle closes the CPU rescoring task only. It does not convert E6 into a complete-negative scientific result: the final paired hypothesis is untested because the required second-query evidence is absent.
