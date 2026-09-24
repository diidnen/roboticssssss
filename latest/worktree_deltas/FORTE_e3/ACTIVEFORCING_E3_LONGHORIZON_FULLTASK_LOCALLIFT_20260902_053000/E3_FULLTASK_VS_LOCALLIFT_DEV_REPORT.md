# E3 FullTask vs LocalLift — task5 DEV evidence

## Confirmatory verdict

`TASK5_DEV_FAIL_NO_ROBUST_REGIME_TRANSFER`

The frozen confirmatory gate failed: root7500 produced `LOCAL_FAILURE` at all three fixed anchors (4, 6, and 7 N), while root7501 produced two LocalLift failures and one LocalLift-positive/FullTask-negative branch but no FullTask success. It therefore did not satisfy the preregistered requirement for all three regimes independently in each DEV root.

## Complete observed evidence

All six cells in the preregistered DEV plan completed and are included in the confirmatory table. Process-ownership ambiguity does not change their exact match to the frozen root/force cells, their valid telemetry, or the pre-outcome plan membership.

- Observed cells: 6/6; valid query states: 6/6.
- Same-root hash stability: pass; held-out roots distinct: pass.
- LocalLift positive: 1/6.
- FullTask positive: 0/6.
- LocalLift-positive / FullTask-negative: 1/6.
- Regimes: 5 `LOCAL_FAILURE`, 1 `LOCAL_SUCCESS_DOWNSTREAM_FAILURE`, 0 `FULL_TASK_SUCCESS`.

The one divergent branch is useful descriptive evidence that LocalLift can over-accept a downstream-failing trajectory. It is not sufficient for the E3 classifier objective: the FullTask target still has zero positives on DEV, and task5 failed robust regime transfer across roots.

## Claim boundary

This is a paired label-target audit, not a powered classifier-generalization result. It does not replace the authoritative Shared Direct checkpoint and makes no Utility, Fmax, safety-limit, TEST, or selector claim. The authoritative Utility hash remains `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10`; no hard-rho selector was used.

Machine-generated six-row data and gate summaries remain in the pilot root. The six-cell fixed plan and no-exclusion interpretation are authoritative in `E3_TASK5_DEV_CONFIRMATION_PLAN.json`, `E3_TASK5_DEV_CONFIRMATION_RESULT.md`, and the machine gate.
