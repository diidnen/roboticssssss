# ActiveForcing authoritative protocol

Status: `AUTHORITATIVE`  
Protocol version: `ACTIVEFORCING_FULL_CLAIM_V2_UTILITY`  
Frozen at: `2026-09-02T04:50:06Z`  
Final force selector: `EXPECTED_UTILITY`

This file supersedes every plan, config, handoff, and result description that makes `minimum force subject to predicted success >= rho` the final runtime selector. Hard-rho selection remains permitted only as a clearly tagged diagnostic ablation.

## Final method

Frozen authoritative π0 → short active physical query → probabilistic belief over hidden physics → Shared Direct full-task feasibility → optional decision-aware re-query → posterior-aware candidate-force evaluation → expected-utility force selection → selected force passed only as a low-level controller setpoint.

π0, nominal motion, grasp pose, language/task execution, low-level feedback law, object geometry, and reset semantics are invariant across force-selection methods.

## Authoritative utility

For hidden physics `z`, task/context `x`, and candidate force `F`:

`U(F|z,x) = p_D(y_task=1|x,z,F) R_succ(F) + [1-p_D(y_task=1|x,z,F)] R_fail`.

The recovered project definition is:

- `R_succ(F) = (Fmax_task - F) / Fmax_task`
- `R_fail = -1`
- `Fmax_task = {task0: 5 N, task1: 6 N, task5: 5 N, task6: 4 N}` for the current four-task archive
- ties choose the lower force

The historical unnormalized form, `R_succ(F)=Fmax_task-F` and `R_fail=-Fmax_task`, is decision-equivalent within each task. The normalized form above is canonical for reporting realized utility.

Authoritative source: `activeforcing_residual_utility_20260901_055605/RESIDUAL_UTILITY_DEVELOPMENT_PROTOCOL.json` (SHA-256 `f697eaa902559fffabaf77a7aa8cb2e8c319007162896a9b5b6a67b22c3857d7`) and `run_residual_utility.py` (SHA-256 `6da4acaa935de7a9a938ac92b56d88b07e34bf93e9614d8458467383e9f28929`).

Point decision:

`F*_point = argmax_F U(F|z_hat,x)`.

Posterior decision:

`U_bar(F|Dq,x) = E_{z~b_phi(z|Dq)}[U(F|z,x)]`,  
`F*_posterior = argmax_F U_bar(F|Dq,x)`.

There is no required runtime `rho_sel`.

## Query and belief semantics

- The short physical query is the frozen P4-B primitive unless a future TRAIN/DEV-only amendment is separately versioned before TEST.
- `QUERY-IGNORED` and `ACTIVE QUERY` share the same initial state, P4-B action, query duration, and physical state transition. They differ only in whether query evidence is provided to the estimator/controller.
- Belief may cover friction, mass, or joint `z=(friction,mass)`.
- Mass and joint friction×mass closure are required. The old rejected “Joint” neural architecture is not reinstated.

## Decision-aware re-query

For each ensemble member `m`, compute `F*_m = argmax_F U(F|z_m,x)`. Re-query disagreement means disagreement among induced utility-optimal force decisions, not disagreement about crossing a reliability threshold.

Required comparison: One Query, Always Two Queries, Raw Uncertainty Threshold, Decision-Aware Utility Consensus, and Oracle Requery diagnostic.

## Continuous planning

At matched candidate budget `K`, compare Frozen Grid, Dense Deterministic Reference, Uniform Continuous, Stratified Continuous, and Proposal-Guided/Posterior-Aware Continuous. Every method selects `argmax_F E_z[U(F|z)]`. Simulator validation must execute the selected float force as an actual off-grid setpoint; nearest-grid replay is forbidden.

## Runtime utility versus evaluation frontier

Runtime decisions maximize expected utility. Empirical reliability thresholds may define under-force, excess-force, empirical frontier, and frontier regret for evaluation only. Test outcomes or empirical TEST frontiers are never selector inputs.

## Required ablations

1. Fixed-Max.
2. Success-Only Direct: `argmax_F p_D(success|x,z,F)`.
3. Full Expected-Utility Direct: `argmax_F U(F|x,z)`.
4. Point: `argmax_F U(F|E[z])` versus Posterior: `argmax_F E_z[U(F|z)]` using identical Direct checkpoint, query, candidates, utility, and episode tuples.
5. Query-Ignored versus Active Query with identical physical transition.

Required controller metrics: Full-task SR, mean force, max force, under-force, excess force, delayed failure, and realized utility. Point/posterior additionally reports force-decision-change rate and both directional success flips.

## Split and locked-TEST discipline

- TRAIN fits models and normalizers. DEV selects/fixes allowed hyperparameters and checks gates. Locked TEST is single-pass evaluation only.
- Before locked TEST, freeze: protocol version, Utility config, Direct checkpoint hashes, belief checkpoint hashes, query/re-query rules, candidate generator, `K`, posterior sample count, fallback, force bounds, task set, root manifest, metrics, and statistics.
- Every output row and directory must contain `protocol_version=ACTIVEFORCING_FULL_CLAIM_V2_UTILITY` and `final_force_selector=EXPECTED_UTILITY`.
- Obsolete outputs must be tagged `protocol_version=MIN_RELIABLE_RHO` or another explicit legacy value and `evidence_role=DIAGNOSTIC_ONLY`.
- No final table may combine Utility-selector and hard-rho rows under one method label.
- Task0 root-scaling TEST roots 5174–5183 (450 branches) have already been collected and evaluated; they are observed and are not a fresh sealed final set for V2 Utility. Reserve new roots for corrected final TEST.

## Claims

Allowed: task-aware force adaptation, low-cost force selection, success-force tradeoff, utility-optimal force selection, posterior-aware force selection.

Preferred wording: “selects a low-cost force by maximizing expected full-task utility”; “marginalizes expected task utility over the physical belief”; “triggers additional physical interaction when epistemic uncertainty changes the induced utility-optimal force decision.”

Forbidden: a guarantee of the absolute minimum feasible/reliable force. “Just-enough force” is descriptive only if supported empirically.

## Frozen flags

- `FINAL_FORCE_SELECTOR = EXPECTED_UTILITY`
- `HARD_RHO_RUNTIME_SELECTOR = DEPRECATED`
- `MASS = REQUIRED`
- `JOINT_FRICTION_MASS = REQUIRED`
- `DECISION_AWARE_REQUERY = REQUIRED`
- `POSTERIOR_CONTINUOUS_PLANNING = REQUIRED`
- `OLD_JOINT_NEURAL_ARCHITECTURE = REJECTED`

Git baselines: FORTE `7f88d0184c1617ed95e67502da96e60be07b3689`; Tabero `80ab3be09ce884f86cfc2037d3af30bc28061426`.
