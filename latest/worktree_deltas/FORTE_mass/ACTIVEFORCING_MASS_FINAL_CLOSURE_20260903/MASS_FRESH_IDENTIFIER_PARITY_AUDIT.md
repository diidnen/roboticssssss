# Fresh Mass identifier preprocessing/parity audit

Scope: existing six corrected fresh query records only; no query, Mass branch, model, π0, controller, or evaluator rerun.

## Checks

- Formal TRAIN query contexts: 12; fresh query records: 6; required physical features: 12.
- Fresh query valid: 6/6; missing required fields: 0.
- Recomputed runner parity: max absolute difference 0.00000000 kg.
- Frozen preprocessing source: `mass_modeling_final.py`, SHA-256 `e579e72f4ce3275b21b842641f7213dfffb0cf3d367f834db626c84f87d98892`.
- `obj_disp_probe_m` is consumed as metres; `actual_probe_displacement_mm` remains a separate millimetre feature, matching the formal feature definition.
- The runner's seven-field `query_features` helper is not used for the authoritative identifier; the runner calls `mass_modeling_final.physical_vector` with the full 12-feature vector.

## Finding

Fresh identifier predictions are 0.4128--0.5493 kg while GT is 0.05--0.20 kg. All six are high-biased; the recomputed values agree with the persisted runner values to numerical precision.
Therefore this is not a simple wrong-unit, missing-field, stale-checkpoint, or runner-vs-model preprocessing mismatch. The observed 0.41--0.55 kg output is produced by the frozen identifier when fresh query features are fed through the same preprocessing path; it is an out-of-support feature/calibration shift relative to formal TRAIN.

## Risk and next action

Severity: HIGH for fresh downstream force selection; confidence: HIGH for parity correctness and high bias, MEDIUM for the physical cause of the feature shift. The smallest next diagnostic is feature-level ablation/clip sensitivity on these saved records; it must remain diagnostic and must not be used to alter final E2E claims without fresh validation.

See `MASS_FRESH_IDENTIFIER_PARITY_AUDIT.csv` and `MASS_FRESH_IDENTIFIER_CONTEXT_PARITY.csv` for feature ranges, train-range violations, z-scores, and per-context recomputation.
