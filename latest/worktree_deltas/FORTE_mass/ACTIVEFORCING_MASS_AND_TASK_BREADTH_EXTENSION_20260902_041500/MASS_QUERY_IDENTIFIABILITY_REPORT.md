# Mass query identifiability report

Status: **P4B_USABLE_FOR_MASS**

## Protocol

- Query: frozen P4-B contact-conditioned shear.
- Fixed nuisance variables: task0 alphabet soup, friction 0.5, appearance, geometry, initial-root generation and controller unchanged.
- Mass bands: LOW 0.05 kg, MID 0.10 kg, HIGH 0.20 kg.
- Development roots: 6100–6105; heldout roots: 6200–6201.
- No downstream force-branch or full-task outcome was read.
- Estimator: standardized Ridge on development roots only; no outcome labels.

## Heldout result

| Feature group | MAE kg | Median AE kg | Spearman | Pairwise ranking | Band accuracy |
|---|---:|---:|---:|---:|---:|
| force_torque | 0.01694 | 0.01193 | 0.886 | 1.000 | 1.000 |
| position_motion | 0.02230 | 0.01907 | 0.886 | 1.000 | 1.000 |
| gripper_response | 0.00867 | 0.00303 | 0.886 | 1.000 | 1.000 |
| marker_tactile | 0.05102 | 0.00655 | 0.943 | 1.000 | 0.833 |
| all_history | 0.00780 | 0.00539 | 0.943 | 1.000 | 1.000 |

The pre-registered usability gate is MAE ≤ 0.04 kg, pairwise ranking ≥ 0.80,
and mass-band accuracy ≥ 0.80 on heldout roots. Best group: **all_history**.
The machine-readable values and heldout predictions are in
`MASS_QUERY_P4B_IDENTIFIABILITY.csv` and `MASS_QUERY_P4B_PREDICTIONS.json`.

## Interpretation

This test answers observability, not task usefulness. A usable result means
P4-B can remain the mass query; it does not establish that mass changes the
downstream force-optimal policy. If the gate is not met, a separate short
vertical-load query must be designed on development roots and then validated
on fresh heldout roots before any formal benchmark collection.
