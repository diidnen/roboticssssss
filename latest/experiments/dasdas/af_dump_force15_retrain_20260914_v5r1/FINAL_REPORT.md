# ActiveForcing force-15 root-local adaptation — final report

Date: 2026-09-14  
Physical root: `200002`  
Task: `dump_bin_bigbin`  
Revision: `AF_DUMP_FORCE15_RETRAIN_V5R1`

## Verdict

The engineering changes are complete and independently reproducible, but the held-out evidence does **not** support that this revision repairs ActiveForcing. The new policy selects 15 N in all six context-held-out cases and succeeds in 2/6, exactly tying both fixed 8 N and fixed 12 N. Its held-out branch AUC is 0.529, and its NLL/Brier are worse than a constant-prevalence predictor.

The expanded range and realized-force utility correction therefore fix genuine protocol defects, but the remaining feasibility model still fails to learn the non-monotonic, context-dependent force landscape observed in the fixed-force sweep.

## Frozen data and split

- Existing root-200002 pool: 192 labels, comprising 144 TRAIN labels and 48 VAL labels.
- New fixed-force sweep: 24 contexts × 5 matched force branches = 120 labels.
- New split, at whole-context grain:
  - TRAIN: 12 contexts / 60 labels
  - VAL: 6 contexts / 30 labels
  - POSTHOC_TEST: 6 contexts / 30 labels
- Actual feasibility fitting/selection pool: 282 labels total:
  - TRAIN: 204 labels (144 old + 60 new)
  - VAL: 78 labels (48 old + 30 new)
- All five branches of a context remain in the same split.
- No exact feasibility-feature hash crosses a split, and no exact feature overlaps the old and new pools.
- All 120 new branch audits passed. Every feature is pre-action `8×64` with `candidate_actions_executed=0`.
- Realized squeeze is absent from feasibility inputs.

The six POSTHOC_TEST contexts are held out from fitting and checkpoint selection. They are not a fresh confirmation set because aggregate sweep outcomes had already been inspected before this protocol was frozen.

## Locked model and utility contract

- Frozen v4 belief; no belief retraining or changed belief intervention.
- Three-member feasibility ensemble, selected epochs 105, 120, and 88.
- Force support: `0.5–15 N`.
- Force feature normalization: `force/15`.
- Utility: `p*(15-realized_squeeze(command))/15-(1-p)`.
- Command→realized-squeeze calibration fit only from the 12 new TRAIN contexts.
- Calibration is utility-only; it is never a feasibility input.
- Model, checkpoints, frozen belief, calibration, utility, and source hashes were locked before sealed test labels were opened.
- Frozen runtime replay reproduced all held-out probabilities with maximum error `0.0`.

The TRAIN-only calibration captures controller saturation: its realized-squeeze knots are approximately 5.34 N at 5 N command, 7.88 N at 8 N, 8.68 N at 10–12 N, and 8.69 N at 15 N. Held-out calibration MAE is 0.326 N.

## Held-out result

| Metric | Result |
|---|---:|
| Contexts / branch labels | 6 / 30 |
| Observed success prevalence | 9/30 = 30.0% |
| Branch AUC | 0.529 |
| Branch NLL | 0.911 |
| Constant-prevalence NLL | 0.611 |
| Branch Brier | 0.288 |
| Constant-prevalence Brier | 0.210 |
| Selected-force success | 2/6 = 33.3% |
| Fixed 8 N success | 2/6 = 33.3% |
| Fixed 12 N success | 2/6 = 33.3% |

Selected force sequence: `[15, 15, 15, 15, 15, 15] N`.

Against both fixed 8 N and fixed 12 N, the paired table is: one selected-only success, one baseline-only success, one shared success, and three shared failures. Net paired wins are zero; exact two-sided McNemar `p=1.0`.

## Interpretation

The earlier fixed-force sweep established that 8 N was unusually weak and that 12 N had the best aggregate result. Revision v5r1 successfully removes the old 8 N ceiling and stops the utility from charging 15 N command as though the robot realized 15 N contact load. This changes the decision regime: the selector no longer stays near 5 N and instead moves to 15 N.

However, the feasibility network predicts a largely increasing success curve and chooses the upper boundary for every held-out context. That conflicts with the observed sweep ordering (`12 N > 5 N = 15 N > 10 N > 8 N`) and produces no held-out gain over either fixed baseline. The current failure is therefore no longer primarily the force-range cap; it is inadequate feasibility generalization/structure for a sparse, non-monotonic force-by-context response.

## Decision

Do not promote v5r1 as a repaired online ActiveForcing policy. Preserve it as a negative, fully audited diagnostic result. The next model should encode or regularize non-monotonic force response and be selected without exposing a new same-root confirmation set; only after it passes that fresh confirmation should it be compared online with fixed 8 N and fixed 12 N.

## Audit artifacts

- `INDEPENDENT_AUDIT.json`: fail-closed recomputation of data, split, calibration, hash chain, metrics, paired outcomes, and frozen-runtime probabilities.
- `models_v5/TRAINING_COMPLETE.json`: terminal training marker.
- `models_v5/MODEL_UTILITY_LOCK.json`: pre-test model and utility lock.
- `models_v5/TEST_ACCESS_RECEIPT.json`: post-lock sealed-label access receipt.
- `models_v5/POSTHOC_TEST_RESULTS.json`: context-level decisions and prediction curves.

Key hashes:

- Training protocol: `67d26a6514e49657d2121458925d6923b02a2353144bdf1fade143beddf86851`
- Model/utility lock: `cfaa2bee4c2de70aaae804ced199dec70ae3cbd8daeb7f070612093e4197aa98`
- Posthoc results: `a85181cbc19c99990155f9de85b48dedce59e3642e78545ac20136a8a88580c9`
