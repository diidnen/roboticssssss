# Final E0–E8 Utility matrix

Protocol: `ACTIVEFORCING_FULL_CLAIM_V2_UTILITY`. Every final controller row uses expected Utility; empirical frontiers are evaluation-only.

| Stage | Required question and method | Required comparisons | Required evidence / metrics | Current reconciliation |
|---|---|---|---|---|
| E0 | Frozen π0, nominal motion, reset, controller, and force-interface invariance | hash/replay/interface audits | state/action/controller hashes; commanded→measured force; off-grid resolution | Existing audits reusable; certify continuous setpoint resolution before E7 TEST |
| E1 | Friction adaptation using Utility | Fixed-Max; Success-Only; Full Utility; GT diagnostic | Full SR; mean/max force; under/excess; delayed failure; realized Utility | 720 raw branches reusable; threshold-selected controller rows must be recomputed |
| E2 | Physical identification and calibrated belief | point; posterior; Vision/Physical/Vision+Physical/SysID/GT where available | MAE; Brier/NLL/ECE; coverage; force-decision change | Friction partial; E6 belief checkpoints reusable after provenance QA; mass development reusable |
| E3 | FullTask versus LocalLift | matched targets, inputs, architecture, folds | full/stage SR; downstream failures; calibration; Utility | Current 720 LocalLift label degenerate; requires non-degenerate stage-labeled task/data |
| E4 | Shared transfer | shared versus task-specific Direct under same Utility | heldout SR; calibration; mean force; frontier error; Utility | Core evidence development-only; complementary mass/long task still required |
| E5 | Fresh reset-to-end π0 E2E with final Utility | Fixed-Max; Success-Only; Full Utility; Point; Posterior; Query-Ignored; Active; GT diagnostic | Full SR; force; under/excess; delayed failure; realized Utility; directional point/posterior flips | Current smoke is hard-threshold diagnostic; corrected Utility smoke/full rerun required |
| E6 | Decision-aware re-query from induced Utility decisions | One; Always Two; Raw Uncertainty; Utility Consensus; Oracle diagnostic | Full SR; query count/rate; precision; harmful-decision enrichment; force; Utility; latency | Belief reusable; all rho-induced decisions require rerun/recompute |
| E7 | Posterior-aware continuous planning with expected Utility | Frozen Grid; Dense Reference; Uniform; Stratified; Proposal/Posterior-Aware at matched K | Full SR; force; under/excess; frontier regret; Utility; latency; K/sample sensitivity; off-grid tracking | Existing continuous checkpoints/raw predictions diagnostic/reusable; Utility planner and exact off-grid execution required |
| E8a | Mass identification and Utility force adaptation | point/posterior mass; nominal/friction-only; Fixed-Max; Success-Only; Full Utility | mass MAE/calibration; Full SR; mean/max force; under/excess; Utility | P4-B mass identifiability and task0 qualification reusable; final adaptation missing |
| E8b | Joint friction×mass belief and Utility adaptation | point joint; posterior joint; friction-only; mass-only; Fixed-Max | both MAEs; cross-confusion; decision accuracy; SR; force; under/excess; interaction; Utility | Required fresh 3×3 factorial; old Joint neural architecture rejected |

## Common method contract

- `U(F|z,x)=p_D(success|x,z,F)*(Fmax-F)/Fmax + (1-p_D(success|x,z,F))*(-1)`.
- Point uses `z_hat=E[z]`; posterior integrates Utility, not probability-threshold crossings.
- Tie: lower force. Fallback is frozen before TEST and reported; fallback does not change the objective.
- Adaptation changes only the force setpoint.
- Query-Ignored and Active Query share initial state, P4-B action, duration, and physical transition.
- Pair by task, root, physics cell, initial state, and repeat; use root-clustered uncertainty.

## Required metric definitions

- `under-force`, `excess-force`, and `frontier regret` are computed against a pre-registered empirical evaluation frontier. They are not runtime inputs.
- `realized Utility = (Fmax-F)/Fmax` on full-task success and `-1` on full-task failure.
- `delayed failure = lift success followed by full-task failure in transport/turning/placement/close`, with cause logged.
- Point/posterior reports force-decision-change rate, Point fail→Posterior success, and Point success→Posterior fail.
