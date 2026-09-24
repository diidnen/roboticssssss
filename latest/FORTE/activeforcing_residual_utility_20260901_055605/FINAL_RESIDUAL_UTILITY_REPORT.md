# ActiveForcing residual-utility development report

## Direct answer

The current historical Direct controller was **already using GNP-style expected utility**, not a probability crossing. The new normalized reward divides every candidate utility within a task by the same positive `Fmax`, so P0 and P1 are decision-identical: **True**. Therefore removing 0.5/0.8 thresholds produces no new gain in this matched experiment.

The World-Model residual gate is **FAIL**. On normal pooled grouped-root OOF, DirectUtility has macro SR 0.9514, mean force 4.0327 N, under-force 0.0347, and realized utility 0.1374. Current-WM Residual has SR 0.9306, mean force 4.0211 N, under-force 0.0556, and utility 0.1104. Direct-Only Residual has SR 0.9583 and utility 0.1548. Thus predicted physics does not satisfy the preregistered independent-value conditions.

The final development choice is **P1 Direct Utility**. No untouched TEST was opened; this is not a final paper claim.

## Reliability–force tradeoff on normal pooled OOF

| Planner | Macro SR | Under-force | Mean force (N) | Success-only normalized regret | Realized utility |
|---|---:|---:|---:|---:|---:|
| DirectUtility | 0.9514 | 0.0347 | 4.0327 | 0.0448 | 0.1374 |
| Direct-Only Residual | 0.9583 | 0.0278 | 3.9868 | 0.0354 | 0.1548 |
| Current-WM Residual | 0.9306 | 0.0556 | 4.0211 | 0.0532 | 0.1104 |
| PhysicsOnly-WM Residual | 0.9514 | 0.0347 | 4.0634 | 0.0543 | 0.1297 |
| One-Step | 0.9861 | 0.0000 | 4.3228 | 0.1004 | 0.1223 |
| Fixed-Max | 0.9861 | 0.0000 | 4.8329 | 0.2052 | 0.0189 |

Per-task, pooled, and all three residual-seed rows are in `NORMAL_POOLED_UTILITY_TABLE.csv`; the table above is macro aggregation, not a hidden pooled-only average.

## Frozen force-critical stress subset

Membership was not changed. It remains the already-frozen model-independent 14-context, 11-root retrospective subset, selected using real adjacent-force outcomes before case-level model queries. DirectUtility already has SR 1.0000 here and 0 recoverable Direct failures, so rescue rate is **not identifiable (zero denominator)**. Current-WM Residual reaches SR 1.0000; One-Step reaches 1.0000. This subset cannot demonstrate rescue when Direct has no recoverable failure under the frozen utility proposal.

The prospective Force-Critical Challenge remains incomplete and untouched. These retrospective rows cannot replace a prospective untouched challenge confirmation.

## Current WM versus PhysicsOnly WM

Both use the same residual architecture and cross-fitting. Full branch-level residual errors and trajectory fidelity are in `WORLD_MODEL_RESIDUAL_DECOUPLING.csv`. Current-WM control utility is 0.1104; PhysicsOnly-WM is 0.1297. The two representations are not equivalent in this development result; any Current-WM advantage must be described as outcome-shaped rather than pure physics.

## Probe / NoProbe / GT

The frozen estimator has no legal posterior decision rule; `sigma_mu` is diagnostic-only, so Probe uses only the point estimate. NoProbe averages utility over the frozen support `[0.30, 0.56, 0.92]`. On the normal pooled diagnostic, NoProbe/Probe/GT macro SR values are 0.9375, 0.9306, and 0.9514.

This is **not root-heldout Probe evidence**: the current roots reuse estimator TRAIN seeds 5100–5105. It verifies that the frozen Probe can be wired into the utility pipeline, but cannot establish the final NoProbe→Probe→GT paper gradient.

## Mechanism interpretation

- Expected-utility planning follows the GNP principle: combine success benefit, force cost, and a fixed failure penalty, then optimize the expectation.
- Residual utility is nominal-plus-correction: Direct remains the strong nominal decision and predicted physics can only shift candidate utilities; it never casts a binary veto.
- The historical Hard Verifier remains an ablation. Its risk–coverage failure came from false-negative rejection and `NO_VALID_FORCE`; it is not eligible for the selected controller.

## Answers to the preregistered questions

1. **Threshold removal:** no measured change, because the existing Direct rule was already the same utility argmax.
2. **WM residual improves DirectUtility:** no reliable independent-value result.
3. **Beyond ordinary calibration:** not supported.
4. **PhysicsOnly:** see matched rows; it does not match Current-WM utility.
5. **Normal-safe and challenge-beneficial:** not jointly supported.
6. **More selective than One-Step:** supported on the frozen retrospective subset.
7. **Versus Fixed-Max:** Fixed-Max uses 4.8329 N on normal pooled; the selected method's force/utility tradeoff is reported separately rather than hidden in one scalar.
8. **NoProbe→Probe→GT:** diagnostic rows exist, but the root-overlap caveat prevents a final gradient claim.
9. **Probe changes full-task decisions:** exact per-context changes and GT agreement are in `PROBE_UTILITY_PER_CONTEXT.csv`; they are diagnostic only.
10. **Final method needs WM:** **no** under the frozen gate.
11. **Selected planner:** **P1 Direct Utility**, not Direct threshold and not Hard Verifier.
12. **Ready for final paper benchmark:** development components are hash-frozen, but final method evidence is **not complete** until a valid untouched evaluation and root-heldout Probe comparison are available.

## Data-quality limitations

- task1 contains reconstructed outcomes in the authoritative pooled population; see the prior pooled audit.
- The force-critical result is retrospective and has zero recoverable Direct failures under the current utility proposal.
- Probe outputs on these 72 contexts are not root-heldout from the frozen friction estimator.
- OOF results are development evidence. `ACTIVEFORCING_RESIDUAL_UTILITY_FREEZE.json` explicitly forbids treating them as untouched TEST.
