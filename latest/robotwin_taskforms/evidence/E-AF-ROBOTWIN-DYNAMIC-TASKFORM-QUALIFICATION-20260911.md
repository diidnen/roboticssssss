# E-AF-ROBOTWIN-DYNAMIC-TASKFORM-QUALIFICATION-20260911

## Decision question

Can a frozen RoboTwin-tuned pi0 execute maintained-grasp downstream behavior for
`shake_bottle`, `shake_bottle_horizontally`, and `dump_bin_bigbin` with a real,
audited terminal evaluator and a controllable/measurable grip-force channel?

This is a development qualification. It does not test ActiveForcing efficacy and it
does not authorize final-task claims.

## Frozen candidate order

1. `shake_bottle` -- dynamic reorientation plus vertical displacement.
2. `shake_bottle_horizontally` -- dynamic reorientation with a different inertial load profile.
3. `dump_bin_bigbin` -- maintained grasp, large reorientation, and content transfer.

`click_bell`, `click_alarmclock`, and `press_stapler` are excluded because the audited
RoboTwin implementations use static objects and contact-only success predicates.

## Baseline and provenance

- Simulator: RoboTwin 2.0 at `96c1feab536306b50c26af200044fcdf126e8904`;
  XPolicyLab submodule at `c37109c500be67d0dea6b36bf7337bbd26e763cd`.
- Policy: a RoboTwin/Aloha pi0 checkpoint; source revision, complete file hashes,
  normalization asset key, and load log must be recorded before any outcome is used.
- The existing LIBERO/Panda pi0 checkpoint is forbidden because the embodiment and
  action interface differ.
- VLA parameters remain frozen for every qualification and later force branch.

## Measurement repairs frozen before policy outcomes

The upstream shake predicate (`bottle_z > threshold`) is diagnostic only and cannot
serve as `y_full`. Before qualification outcomes are inspected, implement an independent
trajectory evaluator requiring all of:

1. established grasp is reached;
2. grasp/contact is maintained through the evaluated downstream window;
3. no drop or table impact;
4. the required number of alternating motion reversals is completed;
5. per-task displacement/orientation amplitude exceeds thresholds derived from the
   repository expert trajectory, not from ActiveForcing outcomes.

For `dump_bin_bigbin`, preserve the native content-transfer predicate and add explicit
grasp-retention and validity logging.

## Supplied-grasp protocol

- Prefer a state captured from a successful nominal frozen-pi0 rollout.
- If native grasp acquisition fails, use a deterministic repository expert trajectory
  only to produce the established-grasp state.
- After the handoff state, all motion must come from the online frozen pi0.
- Exact sibling force branches must restore the same simulator, robot, object, contact,
  RNG, and policy state.

## Development roots and gate

- Six development roots per candidate.
- Fixed conservative grip setting only during semantic qualification.
- Candidate passes downstream qualification at at least 5/6 valid full-task successes.
- Invalid setup/evaluator/checkpoint episodes do not become scientific failures; they
  trigger an engineering repair and rerun before freeze.
- Valid semantic failures remain in the denominator and cannot be tuned away.

## Grip-force authority gate

Before data collection, verify requested setpoint -> gripper drive/effort -> bilateral
contact force -> object response with logged units and no hidden clipping. Repeated
identical states must reproduce measured squeeze within a frozen tolerance. If the
simulator cannot provide a physically meaningful force channel, stop rather than use
gripper position as a mislabeled force.

## Physics and sensitivity pilot

- Hidden variable remains object--gripper friction only.
- Object mass/inertia may be changed once to a fixed plausible controlled value during
  DEV calibration, but may not vary across friction cells or be inferred/claimed.
- Use three friction values within the validated ActiveForcing support and the existing
  3--5 N candidate-force support where the authority audit confirms those units.
- Run same-state oracle branches before collecting a training set.

Candidate force sensitivity passes only if there is at least one valid context with a
monotone success boundary across force and at least one task-form contrast at matched
friction. A candidate with all-success, all-failure, or persistently non-monotone outcomes
is excluded before training.

## Competing explanations

1. The tasks are genuinely grip-force sensitive under different downstream motion.
2. Reported success is caused by weak upstream evaluators rather than task completion.
3. Aloha gripper position commands do not expose an authoritative force intervention.
4. pi0 semantic competence does not survive an established-grasp handoff.

## Stop rule

Do not collect the 216-branch training grid until checkpoint, evaluator, supplied-grasp,
semantic 5/6, force-authority, exact-restore, and force-sensitivity gates pass. If fewer
than two genuinely distinct task forms qualify, report the negative result and return to
task search; do not lower thresholds or add roots after outcomes.

## Required artifacts

- `TASK_FORM_EXPANSION_INPUT_AUDIT.md`
- `ALL_TASK_FORM_CANDIDATES.csv`
- `TASK_FORM_TAXONOMY.md`
- `TASK_FORM_DIVERSITY_MATRIX.csv`
- `TASK_FORM_SCOPE.md`
- `TASK_FORM_SUPPLIED_GRASP_AUDIT.md`
- `TASK_SPECIFIC_FULL_SUCCESS_EVALUATORS.md`
- checkpoint and repository hash manifests
- per-step qualification traces and videos
- machine-readable gate verdict

## Development log (not final evaluation)

- Community checkpoint source: `Heisen0928/pi0_robotwin`, pinned Hugging Face
  revision `f163ea4a7d4b4806a811644117e2e4e6ab29e1a9`.
- `30000`, vertical shake, root `100000`: strict failure after a valid maintained
  grasp; path and rotation gates passed, but the policy produced zero required
  vertical direction reversals. This is a semantic policy failure, not a renderer
  or evaluator failure.
- `30000`, horizontal shake, root `100000`: strict failure with zero established-
  grasp trace samples.
- `30000`, dump, requested root `100000`: repository expert check rejected the
  unstable setup; next valid root `100002` completed all-content transfer in 115
  policy steps and passed the native terminal predicate.
- A LoRA pilot initialized from community `45000` used 60 demonstrations (20 per
  task; 15,933 frames). Both steps 499 and 1999 failed vertical-shake root
  `100000` with zero established-grasp trace samples. Video confirms that the
  bottle was rotated upright but not lifted, then dropped. Action logging confirms
  the right-gripper channel correctly transitions from open (`~1`) to closed
  (`~0`), ruling out an open/close convention or packed-dimension error.
- Because `30000` is behaviorally stronger than `45000` on the strict vertical
  task, the next LoRA qualification run is initialized from `30000`. These are
  development outcomes and do not alter the frozen success thresholds.
- Native-reset `handover_mic` reached only 4/6 because two trials dropped the
  microphone during initial acquisition. Under the frozen supplied-grasp protocol,
  repository `grasp_actor` establishes contact and pi0 controls all downstream
  handover motion; this qualified at 6/6 valid roots.
- Under the same supplied-grasp protocol, `dump_bin_bigbin` qualified at 5/6 valid
  roots. Unstable repository setups were skipped before entering the denominator.
- The task set is therefore frozen to `handover_mic` and `dump_bin_bigbin`.
- Grip-force authority was verified on `handover_mic`, root `100000`, friction
  `0.25`: requested 3 N produced mean/P95 measured single-finger forces of
  3.174/3.478 N; requested 5 N produced 5.116/5.260 N. Both completed the task.
- A matched-root force boundary was found for `dump_bin_bigbin`, seed `300000`,
  friction `0.25`: 3 N failed with contact ratio 0.146, while 4 N succeeded with
  contact ratio 0.571. At the same friction, `handover_mic` succeeds at 3 N,
  establishing the required task-form contrast.
- An 8 N handover pilot made the PhysX contact solve stop advancing while consuming
  a full CPU core. The branch was terminated as an engineering-invalid run. The
  supported candidate set is frozen to 3, 4, and 5 N.
- The 2 mm observable shear query passed the frozen contact/return gate on 24 exact
  roots (12 per task) at all three friction values. The immutable root map has SHA-256
  `2d54f08fb1f9243c110b830017780fb1eff4aaada6281bf337c2cbc9d183c844`.
- An audit after the first eight formal branches found that the JAX pi0 policy advanced
  one persistent sampling RNG across episodes: the RoboTwin adapter's `reset()` cleared
  observations but did not reset policy noise. Those eight records were invalidated and
  recoverably archived under `archive/pre_crn_rng_fix_20260911_0140`; they are excluded
  from all training and evaluation.
- `Pi_0.Model.prepare_case` now resets the policy RNG from the exact episode seed, giving
  physical siblings common random numbers. A duplicate branch test retained small
  PhysX/contact nondeterminism (pre-action state max absolute difference `2.01e-4`,
  first-chunk action max difference `7.91e-3`) but reproduced the semantic outcome in
  both trials. Formal collection restarts from zero with adapter and policy hashes in
  the manifest.
- The restarted formal collection completed `216/216` branches with zero failed
  records; the collection audit passed. CPU training completed with root-disjoint
  train/validation/test splits (`48/12/12` belief contexts and `144/36/36`
  feasibility rows). Offline held-out selection was `7/12` (`0.5833`) with mean
  selected force `3.5 N`; this did not justify an online improvement claim.
- The final fresh online held-out evaluation completed all `12/12` test contexts
  without using test labels for selection. ActiveForcing selected-force success was
  `7/12` (`0.5833`), selecting `3 N` nine times and `5 N` three times. By task,
  `handover_mic` was `6/6`, while `dump_bin_bigbin` was `1/6`. Matched post-selection
  fixed-force outcomes were `3 N: 7/12`, `4 N: 7/12`, and `5 N: 8/12`.
  Therefore this run demonstrates a reproducible task-form/force-sensitive pipeline,
  but not a statistically stronger AF policy than the fixed-force baselines.
