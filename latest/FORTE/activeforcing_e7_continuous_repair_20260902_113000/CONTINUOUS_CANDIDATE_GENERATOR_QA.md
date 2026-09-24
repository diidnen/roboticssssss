# Continuous Candidate Generator QA

| generator | rows | all_pass | min_unique_K | offgrid_any |
|---|---|---|---|---|
| DENSE_REFERENCE | 8 | True | 201 | True |
| FIXED_GRID | 8 | True | 5 | True |
| PROPOSAL_GUIDED | 8 | True | 5 | True |
| STRATIFIED_CONTINUOUS | 8 | True | 5 | True |
| UNIFORM_CONTINUOUS | 8 | True | 5 | True |

- FIXED_GRID is the current frozen taskwise ActiveForcing grid from E5 (0.25 N for tasks 0/1/5 and 0.125 N for task 6), clipped to the certified range. When K exceeds available grid points, the last point is repeated and `effective_unique_K` exposes the limitation; no fictitious grid point is created.
- DENSE_REFERENCE is deterministic and diagnostic only.
- UNIFORM_CONTINUOUS uses independent seeded random-uniform draws; it is not `linspace`.
- STRATIFIED_CONTINUOUS draws independently within K equal-width strata.
- PROPOSAL_GUIDED combines broad stratified coverage with samples near the TRAIN-frozen Expected-Utility maximum.

All random generators are bounded, reproducible, pairwise distinct in QA, and preserve native floats without snapping.
