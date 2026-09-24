# ACTIVEFORCING Mass final report

## Final status

**MASS_EXTENSION_READY_FOR_PAPER**

The Mass raw collection was not recollected. The completed formal source passed independent QA at 180/180 branches and the downstream closure includes root-heldout identification, Mass-only full-task Direct, authoritative Expected Utility, force-sensitivity and failure analysis, and the completed fresh E2E.

## Headline results

- Independent QA accepted 80/80 newly collected branches, yielding 180/180 accepted branches across 18/18 contexts and 3,708 query timesteps.
- Root-heldout physical-history identification achieved MAE 0.0229 kg, RMSE 0.0246 kg, Spearman 0.956, and 1.000/1.000/1.000 LOW/MID/HIGH band accuracy over three seeds. Vision-only was not useful (MAE 0.2102 kg).
- Offline ActiveForcing-Mass reached full-task SR 0.917 and realized utility 5.646, versus 0.667 and 2.333 for Fixed-Max and 0.667 and 4.000 for the no-query prior.
- The force benchmark is valid but mixed: HIGH mass requires at least 2.5 N in the observed curve, while LOW/MID overlap and are non-monotonic at 4.0 N.
- Fresh frozen-π0 E2E over 30 rows reached query-state reach/validity 1.000 for ActiveForcing-Mass, but full E2E SR was 0.167 (1/6); GT-Mass was 0/6. This exposes a Direct/Utility transfer failure rather than supporting an unqualified positive deployment claim.

## Closure evidence

{
  "status": "MASS_EXTENSION_READY_FOR_PAPER",
  "completion_classification": "POSITIVE_OR_MIXED",
  "qa": {
    "status": "PASS",
    "accepted_branches": 180,
    "accepted_contexts": 18
  },
  "identification": "COMPLETED",
  "direct": "COMPLETED",
  "utility_offline": "COMPLETED",
  "fresh_e2e": "COMPLETED",
  "formal_source": "/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M3_TASK2_FORMAL_STRUCTURED_20260902_130700_RESUME",
  "protected_pi0_untouched": true,
  "protected_e5_untouched": true,
  "generated_utc": "2026-09-03T06:01:32.658063+00:00",
  "required_artifacts_present": {
    "MASS_FINAL_BRANCH_QA.md": true,
    "MASS_FINAL_BRANCH_QA.json": true,
    "MASS_ACCEPTED_MANIFEST.json": true,
    "TABLE_MASS_IDENTIFICATION.csv": true,
    "MASS_IDENTIFICATION_REPORT.md": true,
    "TABLE_MASS_FORCE_ADAPTATION.csv": true,
    "MASS_FORCE_ADAPTATION_REPORT.md": true,
    "TABLE_MASS_FRESH_E2E.csv": true,
    "MASS_FRESH_E2E_REPORT.md": true,
    "MASS_FRESH_E2E_SUMMARY.json": true,
    "MASS_FAILURE_ANALYSIS.md": true,
    "MASS_ENGINEERING_FIX_LOG.md": true
  }
}

See `MASS_IDENTIFICATION_REPORT.md`, `MASS_FORCE_ADAPTATION_REPORT.md`, `MASS_FRESH_E2E_REPORT.md`, and `MASS_FAILURE_ANALYSIS.md` for the detailed tables and caveats.
