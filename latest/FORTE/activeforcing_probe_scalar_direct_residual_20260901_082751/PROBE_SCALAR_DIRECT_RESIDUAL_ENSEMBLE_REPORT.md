# OOF ProbeScalar Direct-Only Residual Report

**Matched result: PROBE_SCALAR_DIRECT_RESIDUAL_IMPROVES_UTILITY.** With fully root-heldout Probe estimates, Direct-Only residual changes pooled ensemble SR from 93.75% to 93.75%, under-force from 4.86% to 4.86%, mean force from 4.202 N to 4.191 N, and realized utility from 0.0879 to 0.0891.

## Interpretation

This exact matched protocol first ensembles three nested OOF ProbeScalar Direct predictions, then trains three residual seeds. It is directly comparable to the prior Direct-Only residual experiment. Utility seed gates are [True, True, True]. The seed-paired sensitivity is retained separately and is not substituted for this primary comparison.

## Scope and limits

All 24 root families are grouped; Probe, inner Direct and residual do not see an evaluated root. No World Model or untouched TEST is used. This remains pooled TRAIN-root OOF development evidence.
