# Corrected handoff — E6/E7

The active E6/E7 closure is allowed to finish, but its selector-dependent outputs are legacy diagnostic data.

## Reusable outputs

- Three `PHYSICAL_BELIEF_member_*.pt` checkpoints, if manifests prove TRAIN-only fitting.
- TRAIN-fitted uncertainty scale and DEV calibration tables.
- Root-heldout context split.
- Frozen `GNP_STYLE_CONTINUOUS_FEAS` probability checkpoints and raw candidate predictions.

## E6 correction

For every ensemble member `m` and the identical candidate set, compute `F*_m=argmax_F U(F|z_m,x)`. Decision-aware consensus is agreement among these `F*_m`. Compare One Query, Always Two, Raw Uncertainty Threshold, Decision-Aware Utility Consensus, and Oracle Requery diagnostic. The raw threshold applies only to deciding whether uncertainty is large; it must not change the force selector.

Existing `RHO_SELECTION_TABLE.csv`, `FINAL_RHO.json`, threshold-induced decision disagreement, and threshold-selected E6 results are `DIAGNOSTIC_ONLY`.

## E7 correction

Replace `plan_one(...rho...)` with candidate-level Utility and choose the maximum posterior mean Utility. Compare Frozen Grid, Dense Deterministic Reference, Uniform Continuous, Stratified Continuous, and Proposal/Posterior-Aware at matched K.

Proposal guidance must be derived from Utility/posterior information or a selector-independent proposal distribution; locating a `p>=0.8` boundary and concentrating around it is legacy hard-rho semantics.

Stored predictions can support CPU rescoring. Simulator execution is required only for chosen off-grid forces not already executed exactly. Never use nearest-grid replay.

## Output tags

- Current outputs: `protocol_version=MIN_RELIABLE_RHO`, `evidence_role=DIAGNOSTIC_ONLY`.
- Corrected outputs: `protocol_version=ACTIVEFORCING_FULL_CLAIM_V2_UTILITY`, `selector=EXPECTED_UTILITY`.
