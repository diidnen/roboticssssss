# AF dump force-15 retraining v5

## Status

Implementation authorized on 2026-09-14. The fixed-force range sweep is complete and independently re-aggregated at 24 contexts / 120 paired rollouts.

## Question

Does repairing the feasibility force support and controller-aware force cost improve root-local force selection for `dump_bin_bigbin` under physical root `200002`?

## Evidence entering this intervention

- Existing v4 feasibility data: 192 audited labels with commanded forces in 0.5--8 N.
- New range diagnostic: 24 contexts x 5 commanded setpoints (5, 8, 10, 12, 15 N) = 120 audited labels.
- Final success counts: 5 N 9/24, 8 N 4/24, 10 N 8/24, 12 N 10/24, 15 N 9/24.
- Commands 10--15 N realize similar target-contact squeeze near 8.7--8.8 N, so commanded force and realized load must remain distinct.

## Hypothesis and competing explanation

- H1: v4 fails partly because the feasibility model and utility cannot represent or select useful commands above 8 N. Adding audited labels, using a 15 N force-feature normalization, and charging utility by calibrated realized squeeze will improve context-conditioned selection.
- H0: the sweep advantage is a post-hoc context effect or controller saturation artifact; retraining will not improve held-out predictive or selected-policy results.

## Frozen data grain and split

- Unit of splitting: complete context. All five force branches from one context stay together.
- New TRAIN: policy seeds 80200002--80200005 at each of the three frictions (12 contexts, 60 labels).
- New VAL: policy seeds 80200006--80200007 at each friction (6 contexts, 30 labels).
- New POSTHOC_TEST: policy seeds 80200008--80200009 at each friction (6 contexts, 30 labels).
- The test subset is an internal post-hoc test because aggregate sweep outcomes were inspected before this intervention. It is not fresh confirmation evidence.
- Existing v4 TRAIN/VAL assignments remain unchanged.

## Intervention

1. Freeze references and hashes for all 24 new contexts without rewriting raw data.
2. Convert each eligible branch to `{feature, force, full_task_success_y}`.
3. Retrain the unchanged phase-free architecture with commanded force normalized by 15 N and support `[0.5, 15]` N.
4. Fit a monotone command-to-realized-squeeze calibration using only new TRAIN telemetry. Realized squeeze is never a feasibility feature.
5. Use utility `p*(15-realized_squeeze(command))/15 - (1-p)`.
6. Lock model, normalization, calibration, and utility before loading POSTHOC_TEST outcome labels.

## Primary observations

- POSTHOC_TEST branch NLL and Brier score.
- Success of the utility-selected command when selection is restricted to the five actually tested commands.
- Paired selected-policy success versus Fixed-8 and Fixed-12 on the same six contexts.

## Falsifier and rollback

- H1 is weakened if the selected policy does not exceed Fixed-8 on POSTHOC_TEST or if predictive ranking does not improve enough to support a fresh confirmation run.
- Any raw-audit, split-integrity, feature-shape, source-hash, reload-parity, or pre-lock test-label-access failure stops the run.
- No previous dataset, checkpoint, protocol, runtime, or result is overwritten.

## Claim boundary

This intervention supports only root-local task-form adaptation on root `200002`. It does not establish cross-root generalization, monotonic benefit from force, or realized 12/15 N physical contact loads.
