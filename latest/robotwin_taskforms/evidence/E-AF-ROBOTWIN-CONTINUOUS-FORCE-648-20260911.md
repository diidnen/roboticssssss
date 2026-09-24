# E-AF-ROBOTWIN-CONTINUOUS-FORCE-648-20260911

Status: frozen before new force-grid outcomes.

## Decision question

Does adding measured 0.25 N force support let a task-conditioned ActiveForcing
feasibility model recover a smooth minimum-safe-force boundary and improve the
success--force tradeoff over strong fixed-force controls on frozen RoboTwin pi0
execution?

This is a development/confirmation bridge. The original slots 10--11 TEST outcomes
have already been inspected, so results on the reused 12-root population remain
development evidence. A separate outcome-blind fresh-root online comparison is
required for any confirmation wording.

## Frozen provenance and reuse

- Parent experiment: `af_taskforms_216_v1`.
- Parent records SHA-256:
  `234082d3843a65ea32a82768ff0b9c80080d175beaffb93747991f2cb1681dc1`.
- Exact root map SHA-256:
  `2d54f08fb1f9243c110b830017780fb1eff4aaada6281bf337c2cbc9d183c844`.
- Frozen policy: RoboTwin-tuned pi0 checkpoint 30000 from pinned source revision
  `f163ea4a7d4b4806a811644117e2e4e6ab29e1a9`.
- Tasks remain `handover_mic` and `dump_bin_bigbin`; hidden physical variable remains
  object--gripper friction only at `0.25`, `0.55`, and `0.85`.
- Existing `3.00`, `4.00`, and `5.00 N` branches are copied byte-for-byte after hash
  verification. No parent file is edited.

## Force grid and split

- Full requested-force grid: `3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00 N`.
- Existing branches: 2 tasks x 12 roots x 3 frictions x 3 forces = 216.
- New branches: 2 x 12 x 3 x 6 = 432.
- Final development grid: 648 valid branch records.
- Root split is unchanged: slots 0--7 TRAIN, 8--9 VALIDATION, 10--11 reused TEST.
- Every force sibling resets simulator/policy RNG to the same exact root seed. Runs are
  sequential against one policy server to prevent shared-RNG interleaving.

## Primary hypotheses and competing explanations

H1: denser force coverage improves minimum-safe-force selection because the prior
3/4/5 N grid aliased thresholds such as 4.25 N.

H2: apparent improvement is only interpolation on repeated force siblings; the number
of independent physical contexts remains 72 and generalization does not improve.

H3: failures are dominated by frozen-pi0 semantic/trajectory failures. In that case
all forces fail together and no grip-force selector can recover the task.

H4: requested force is not faithfully realized between 0.25 N levels. In that case the
measured squeeze calibration is too noisy for a continuous-force claim.

## Measurements and gates

1. Audit all 648 keys, root/split disjointness, observable-query purity, exact force
   readback, finite action/state tensors, and requested-versus-measured force response.
2. Report context masks over all nine forces. `all-fail` contexts are classified as
   unsalvageable in the tested force range, not as force-selection failures.
3. Report force-sensitive contexts separately: at least one success and one failure
   within 3--5 N. Preserve non-monotone contexts rather than deleting them.
4. Train only on slots 0--7; select threshold/curve treatment on slots 8--9; slots
   10--11 are diagnostic only because their old outcomes were previously opened.
5. The feasibility model receives force as a continuous scalar. Inference evaluates a
   dense 0.05 N grid, applies the frozen nondecreasing latent-feasibility projection,
   and chooses the lowest force above the validation-selected probability threshold.
6. Log both requested setpoint and realized mean/P95 squeeze. Continuous authority is
   supported only if requested force positively orders realized force in both tasks and
   gross reversals are absent over the supported range.

## Fresh online comparison

- Freeze fresh actual root seeds without opening downstream outcomes; query validity
  may be checked because it is upstream of force selection.
- At least two fresh roots per task and all three friction values (12 contexts).
- Lock model, probability threshold, and selected force before reading any outcome.
- Run the selected continuous force and fixed controls at `3.00`, `4.25`, and `5.00 N`
  from matched seeds. Run missing force-grid points only after selection when needed to
  classify a failure as force-sensitive versus all-force semantic failure.
- Primary endpoint: lexicographic full-task success then mean requested force among
  successes. Secondary: regret to the empirical minimum successful force, per-task
  outcome, measured squeeze, and all-force-fail rate.

## Stop rules

- Stop for any parent hash mismatch, query leakage, force readback mismatch, duplicate
  key, split overlap, or persistent process failure.
- If denser force commands do not produce an ordered realized-force response, reject
  the continuous-force extension rather than relabeling position or contact spikes as
  force.
- If fresh online outcomes show no force-sensitive contexts, conclude that this task
  population cannot test continuous selection; do not claim that more sibling rows
  solved the independent-context problem.

## Required artifacts

- immutable parent-copy manifest and hashes;
- resume-safe 648 branch JSONL and collection audit;
- requested/realized force calibration report;
- model checkpoints, training/validation report, and dense force curves;
- locked fresh-root selection decisions before outcomes;
- matched online comparison report with semantic-failure decomposition;
- final experiment-card result and negative evidence.

## Execution log

- The first 3.25 N smoke exited before the first policy action because the non-login
  SSH environment did not expose an `ffmpeg` executable. No scientific record was
  written. A scoped `runtime_bin/ffmpeg` link to the already installed
  `imageio_ffmpeg` binary repaired only the launch environment.
- Repeating the identical smoke completed successfully: `handover_mic`, slot 0,
  friction 0.25, requested force 3.25 N, valid observable query, 132 downstream
  steps, full-task success, and dynamic contact ratio 1.0.
- Formal resume-safe sequential collection started from the sealed 216-row snapshot.
  A separate supervisor verifies the live collector, permits at most three restarts
  without row progress, and will run audit, CPU training, dense inference, fresh-root
  qualification, and online comparison in order.
