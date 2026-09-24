# E-AF-ROBOTWIN-HANDOVER-CONTINUOUS-324-20260912

## Decision and scope

On 2026-09-12 the user asked to prioritize one task because the two-task protocol would not finish in time. The selected task is `handover_mic`, because its development collection is already in progress and it directly tests maintaining grip during a bilateral handover and releasing at the correct phase. The existing `dump_bin_bigbin` records are preserved, but no new dump branches are required for this interim conclusion.

This is a declared single-task confirmation milestone. It must not be reported as evidence across task forms or as completion of the original two-task experiment.

## Hypothesis

A continuous-force-conditioned ActiveForcing selector trained on root-disjoint `handover_mic` branches can choose force from a dense 3.00--5.00 N support without reading downstream outcomes and can match or improve the strongest frozen-force control on fresh roots while using no more mean force than necessary.

## Competing explanations

1. Handover success is dominated by the frozen pi0 trajectory or release timing, so force selection cannot improve task success.
2. Any apparent benefit comes from repeatedly opening development test outcomes rather than from transferable query evidence.
3. Requested force is not physically realized monotonically, so a continuous-force selector is not identifiable.

## Frozen protocol

- Task: `handover_mic` only.
- Development roots: slots 0--7 train, 8--9 validation, 10--11 test.
- Friction: 0.25, 0.55, 0.85.
- Force grid: 3.00--5.00 N in 0.25 N increments.
- Development rows: 324 = 12 roots x 3 frictions x 9 forces.
- Split rows: 216 train, 54 validation, 54 test.
- Model: three belief members and three continuous-force feasibility members; monotonic penalty 0.2.
- Selection support: 3.00--5.00 N in 0.05 N increments with isotonic nondecreasing projection.
- Fresh confirmation: two outcome-blind fresh roots x three frictions = six contexts.
- Controls: fixed 3.00 N, 4.25 N, and 5.00 N; full nine-force empirical oracle used only for failure decomposition after all six selections are locked.
- Checkpoint: RoboTwin-tuned pi0 checkpoint `pi0_robotwin_30000/30000`.

## Primary evidence and success threshold

Primary metric: fresh-root full-task success count of ActiveForcing versus the best fixed-force control across six contexts. The result is supportive only if ActiveForcing is not worse in success count and does not use a higher mean selected force than the best successful fixed control. Report exact counts because six contexts are too few for a strong population claim.

Secondary evidence:

- requested-versus-realized force Spearman correlation at least 0.80 and at most one gross adjacent reversal;
- root-disjoint training and validation-only threshold selection;
- all six selections cryptographically locked before any fresh downstream outcome;
- explicit separation of all-grid-fail semantic/trajectory failures from force-selection failures.

## Falsifiers and stop conditions

- Stop and reject continuous-force authority if the force-response audit fails.
- Do not report a causal benefit if ActiveForcing loses to a fixed control on fresh success count.
- Treat all-grid-fail contexts as pi0 semantic/trajectory failures, not force failures.
- Treat failed engineering runs as invalid until repaired; do not record pseudo-outcomes.
- Pause the original two-task collector only after all 324 handover development rows are durably present.

## Required artifacts

- `branches.jsonl` and filtered sealed-parent snapshot with hashes;
- root map and manifest documenting the user-approved scope change;
- collection audit;
- training report and six checkpoint hashes;
- dense inference report;
- two-root outcome-blind fresh map;
- six-context selection lock and lock manifest;
- fresh online raw branches and summary;
- independent validation report.

## Resource estimate

At the observed handover mean of approximately 146 seconds per development branch, the remaining handover collection is about 3.3 hours. The six-context fresh comparison is expected to take another 2--2.5 hours. CPU audit, training, dense inference, and validation should take less than 30 minutes. Estimated time to a complete single-task conclusion: 5--6 hours from the scope-change decision, assuming no retries.
