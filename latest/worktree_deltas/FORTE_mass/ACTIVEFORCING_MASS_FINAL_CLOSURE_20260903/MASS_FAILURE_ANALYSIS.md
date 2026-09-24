# Mass failure analysis

Primary paired diagnostic uses seed 11 and the formal ROOT-HELDOUT TEST contexts. Categories are intentionally non-exclusive: a single context can have a mass-estimation error and a downstream transport failure. No TEST outcome is used by the identifier or Direct training.

## Error taxonomy

| Category | Context count | Examples |
|---|---:|---|
| MASS_ESTIMATION_ERROR | 0 | none |
| DIRECT_ERROR | 4 | mass_structured_test_t2_root8200_low, mass_structured_test_t2_root8200_mid, mass_structured_test_t2_root8201_low, mass_structured_test_t2_root8201_mid |
| UTILITY_SELECTION_ERROR | 1 | mass_structured_test_t2_root8200_low |
| UNDER_FORCE | 0 | none |
| POST_LIFT_DROP | 1 | mass_structured_test_t2_root8201_mid |
| TRANSPORT_FAILURE | 1 | mass_structured_test_t2_root8201_mid |
| PLACEMENT_FAILURE | 0 | none |
| CONTROLLER_TRACKING_ERROR | 0 | none |
| NON_FORCE_PI0_FAILURE | 0 | none |

## Paired rescue/collateral

ActiveForcing-Mass rescues 3 of 6 contexts versus True NoQuery-Prior at the selected full-task label; collateral cases are 0. The complete paired table is `MASS_FAILURE_PAIRED_CASES.csv`.

## Interpretation

The dominant observed failure is transport during turning, not placement: lift and placement remain high while transport retention drops at insufficient or unstable force. Controller tracking and non-force π0 failure indicators are absent in the offline formal rows. DIRECT_ERROR is a diagnostic mismatch between the same Direct model supplied with GT mass and the empirical utility oracle; it is not attributed to mass estimation.
