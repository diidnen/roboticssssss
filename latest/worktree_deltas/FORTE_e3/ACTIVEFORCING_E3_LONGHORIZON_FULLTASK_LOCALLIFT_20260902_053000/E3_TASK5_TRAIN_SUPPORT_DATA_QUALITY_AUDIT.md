# E3 task5 TRAIN support data-quality audit

## Dataset and grain

- Semantic task: `libero_10/task5`, black book -> caddy back compartment.
- Intended grain: one physical branch per `(split, root_seed, friction, force_N)`.
- Sources: the original 15-cell grid plus the stopped support extension at 6/7 N for root7400, mu=0.6.
- Net unique branches: 17. The 1--5 N rows at mu=0.6 appear in both source tables but point to the same immutable cell directories and are counted once in the 17-branch total.

## Checks

| Check | Evidence | Verdict |
|---|---:|---|
| Original grid completeness | 15/15 for 3 frictions x 5 forces | pass |
| Support-extension completeness through stop | 7/7 for mu=0.6, forces 1--7 | pass |
| Missing/unexpected extension forces | 0 / 0 | pass |
| Query-state validity | 7/7 extension anchors; 15/15 original grid | pass |
| Reset-state integrity | one 64-char hash, `a26a845...bb6c816` | pass |
| Label implication | no FullTask positive with LocalLift negative | pass |
| Three-regime support | 4 Local failures, 2 downstream-only, 1 FullTask success in the mu=0.6 support curve | pass |
| Fmax / Utility contamination | analyzer flags both false | pass |

Across all 17 unique TRAIN branches, the labels are 10 Local failures, 6 Local-success/downstream-failures, and 1 FullTask success. LocalLift therefore has 7 positives / 10 negatives, while FullTask has 1 positive / 16 negatives. Six branches (35.3%) are accepted by LocalLift but rejected by FullTask.

## Risk and interpretation

The original archive's critical one-class LocalLift defect is repaired for TRAIN. FullTask is also no longer one-class, but its single positive is highly imbalanced and all 17 branches share one root. This is sufficient to open a root-heldout DEV confirmation; it is not sufficient for a powered classifier-generalization claim or a production Direct checkpoint replacement.

The 7 N FullTask success is physical label-support evidence only. It does not define a safety limit, candidate maximum, or Utility normalization. The E1 `libero_object/task5` Fmax remains inapplicable.

## Required remediation before E3 closure

Run the pre-outcome fixed DEV anchors `(4,6,7 N)` on each of roots 7500 and 7501, enforce stable within-root and distinct between-root reset hashes, and require all three regimes independently on both roots. Report the exact paired LocalLift-positive/FullTask-negative cases and keep uncertainty explicit at n=6.
