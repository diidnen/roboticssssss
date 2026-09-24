# Validation report

## Overall assessment: Share with caveats; not ready for a TEST or paper decision

The development artifacts are internally consistent and reproducible, but the Predictive Verifier fails its promotion gate. ActiveForcing-Direct remains the supported controller before untouched TEST.

## Methodology review

- PASS: 480 TRAIN branches resolve to 48 contexts and 22 independent roots; all branches, five forces, and two repeats from a root remain in one of four folds.
- PASS: architecture selection uses only grouped TRAIN-rootheldout CV. The viewed DEV rows are appended only as retrospective rows after selection.
- PASS: only the four pre-registered architectures, three seeds, unweighted BCE, final epoch, and `logit > 0` were used.
- PASS: search is upward-only and strict exhaustion is `NO_VALID_FORCE`; Max fallback is labeled as execution fallback rather than verifier-approved safety.
- PASS: PhysicsOnly uses the same H8 GRU/inputs/normalization/trajectory and IE targets with `Lphysics + LIE`; no outcome gradient enters that WM.
- PASS: no untouched TEST manifest, labels, prediction, result, or outcome was opened by the run.

## Material caveats

1. Controller-level CV is verifier-heldout, not fully system-OOF: Direct proposals come from frozen Direct checkpoints trained on all TRAIN roots. Therefore Direct-versus-verifier SR in this diagnostic is a development mechanism comparison, not an unbiased held-out system estimate.
2. The old 24-context baseline averages repeats and seed scores within context-force cells. New CV retains repeat-level episodes and per-seed Boolean decisions. Those metrics have different estimands.
3. Receding-H8 can be audited for later rejection signals, but post-switch trajectory and full-task outcome are absent. Its SR, under-force, and realized force are not identifiable and remain `NA`.
4. OOF outcome errors identify verifier false-positive/false-negative and no-valid-force events, but do not uniquely split every error into WM shift versus verifier representation. Zero counts for non-identifiable categories are not evidence of absence.

## Calculation spot-checks

- Baseline: exact match to the frozen 24-context artifact: Direct SR 0.875, under-force 1/24, mean force 3.932455 N; strict finite 19/24 with conditional SR 0.973684; Max fallback SR 0.9375.
- Linear reproduction: 720 old-DEV branch-seed decisions have zero hard-decision disagreements versus the frozen V0; mean absolute probability difference is `1.91e-08`.
- Search ablation: Direct, one-step, Fixed Max, and Max fallback each have 93/96 successes; strict verifier has macro coverage/SR 0.805556 and conditional SR 1.0.
- Failure reconciliation: 47 verifier false-negative cases plus 9 true no-success-in-range cases equals 56 strict residual rows across three seed runs.
- Required deliverables: all 13 requested files exist and are non-empty.
- Integrity: every entry in `SHA256SUMS.txt` verifies.

## Decision

The evidence supports retaining ActiveForcing-Direct and keeping the Predictive Verifier as a frozen, non-promoted diagnostic candidate. It does not support launching untouched TEST, selecting Receding-H8, treating Max fallback as verifier safety, or making a predictive-physics paper claim.
