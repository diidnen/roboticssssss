## Validation Report

### Overall Assessment: Share with caveats

The forensic classification and stop decision are methodologically supported and independently reproducible. The mandatory caveats are the nine-context DEV sample, five-repeat empirical reference, and structural-only HTML verification.

### Methodology Review

- The protocol was frozen before outcome analysis and hashes the authoritative TRAIN, DEV, prediction, decision, checkpoint, and prior-protocol inputs.
- The exact subsampling calculation enumerates all index combinations without replacement inside each observed five-repeat cell.
- TRAIN top-up eligibility follows the frozen classification; `MODEL_ERROR_DOMINATES` correctly triggers `STOP_BEFORE_COLLECTION`.
- The report distinguishes finite-repeat empirical noise from irreducible physical stochasticity and does not make causal, cross-object, unseen-task, TEST, or Probe claims.

### Issues Found

1. **Medium — finite reference sample:** `p5` is based on five repeats and is not exact physical probability. This limitation is visible in the report and classification artifact.
2. **Medium — nine-context inference:** context-bootstrap intervals quantify robustness only over the frozen nine contexts; they do not establish generalization beyond this population.
3. **Low — browser QA unavailable:** canonical artifact validation, packaging, payload equality, and semantic structure passed, but no installed Chromium was available for viewport and source-dialog verification.

### Calculation Spot-Checks

- DEV outcome distribution: verified independently from all 135 branch rows as 5, 3, 2, 1, 3, and 13 cells for success counts 0 through 5.
- Exact two-repeat noise: verified independently as MAE 0.080, RMSE 0.152753, and reliability-class mismatch probability 0.062963.
- TRAIN cell outcomes: verified independently from all 720 branches as 65 cells at 0/2, 15 at 1/2, and 280 at 2/2.
- Required report structure: all 22 requested headings are present.
- Classification: verified against frozen thresholds as `MODEL_ERROR_DOMINATES`; top-up is scientifically ineligible.

### Visualization Review

The single bar chart is appropriate for six ordered outcome categories, uses a zero-based magnitude comparison, and is backed by a richer dataset containing cell fraction, success count, and stochasticity class. Portable HTML verification status is `structural_only` because Chromium was unavailable.

### Suggested Improvements

1. Before external publication, open `report.html` in a Chromium-capable environment and verify desktop/narrow layout and source dialogs.
2. Any follow-up model experiment should preregister a feature/representation hypothesis rather than add repeats at the same force cells.

### Required Caveats for Stakeholders

- Five-repeat empirical probabilities are finite-sample references.
- The conclusion is limited to the current object/task distribution and nine frozen DEV contexts.
- `MODEL_ERROR_DOMINATES` means the measured data limitations do not explain the error; it does not prove one specific missing variable or architecture defect.
