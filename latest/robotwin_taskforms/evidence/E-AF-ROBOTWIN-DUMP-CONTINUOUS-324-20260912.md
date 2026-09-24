# E-AF-ROBOTWIN-DUMP-CONTINUOUS-324-20260912

## Decision and scope

On 2026-09-12 the user replaced the handover-first milestone with a `dump_bin_bigbin`-only priority result because the dump task gives the clearer ActiveForcing test: the frozen pi0 trajectory must retain a grasp under downstream rotation, inertia, and moment load. All 249 completed `handover_mic` rows are retained as negative/partial evidence but are excluded from this task-specific training and conclusion.

This is a declared single-task confirmation milestone. It must not be reported as cross-task-form generalization or as completion of the original two-task experiment.

## Hypothesis

A continuous-force-conditioned ActiveForcing selector trained on root-disjoint `dump_bin_bigbin` branches can use pre-execution shear-query evidence to choose grip force for the subsequent dump motion and can match or improve the strongest frozen-force control on fresh roots without using greater mean force than necessary.

## Competing explanations

1. Dump success is dominated by the frozen pi0 trajectory, container pose, or contents dynamics, so grip-force selection cannot improve task success.
2. Any apparent benefit comes from development-root reuse or downstream-outcome leakage.
3. Requested grip force is not physically realized monotonically during the rotation, making continuous-force selection unidentifiable.

## Frozen protocol

- Task: `dump_bin_bigbin` only.
- Development roots: slots 0--7 train, 8--9 validation, 10--11 test.
- Friction: 0.25, 0.55, 0.85.
- Force grid: 3.00--5.00 N in 0.25 N increments.
- Development rows: 324 = 12 roots x 3 frictions x 9 forces; 108 sealed parent rows are reused and 216 forces are collected.
- Split rows: 216 train, 54 validation, 54 test.
- Model: three belief members and three continuous-force feasibility members; monotonic penalty 0.2.
- Selection support: 3.00--5.00 N in 0.05 N increments with isotonic nondecreasing projection.
- Fresh confirmation: two outcome-blind fresh roots x three frictions = six contexts.
- Controls: fixed 3.00 N, 4.25 N, and 5.00 N; the nine-force empirical oracle is opened only for failure decomposition after all six selections are locked.
- Checkpoint: RoboTwin-tuned pi0 checkpoint `pi0_robotwin_30000/30000`.

## Primary evidence and success threshold

Primary metric: fresh-root full-task success count of ActiveForcing versus the best fixed-force control across six contexts. The result is supportive only if ActiveForcing is not worse in success count and does not use a higher mean selected force than the best successful fixed control. Exact counts and the small sample size must be reported.

Secondary evidence:

- requested-versus-realized force Spearman correlation at least 0.80 and at most one gross adjacent reversal;
- root-disjoint training and validation-only threshold selection;
- all six selections cryptographically locked before any fresh downstream outcome;
- all-grid-fail trajectory/semantics failures separated from force-selection failures.

## Falsifiers and stop conditions

- Stop and reject continuous-force authority if the force-response audit fails.
- Do not claim a causal benefit if ActiveForcing loses to a fixed control on fresh success count.
- Treat all-grid-fail contexts as pi0/trajectory or environment failures, not force failures.
- Failed engineering runs are invalid until repaired; never record pseudo-outcomes.
- Preserve the partial handover collection but exclude it from dump-only fitting and metrics.

## Required artifacts

- 324 dump-only development branches and filtered sealed-parent snapshot with hashes;
- task-specific root map and manifest;
- collection audit, training report, six model hashes, and dense inference report;
- two-root outcome-blind fresh map and six-context selection lock;
- fresh online raw branches, summary, and independent validation report.

## Resource estimate

At the observed dump mean of approximately 106 seconds per development branch, 216 additional branches require about 6.3 hours. The six-context fresh comparison adds roughly 1.8--2.5 hours. Audit, CPU training, dense inference, qualification, and validation add less than 30 minutes. Estimated end-to-end time is about 8--9 hours if no engineering retries are needed.
