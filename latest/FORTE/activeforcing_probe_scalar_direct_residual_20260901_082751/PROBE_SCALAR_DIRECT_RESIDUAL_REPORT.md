# ProbeScalar Direct-Only Residual Diagnostic

**PROBE_SCALAR_DIRECT_RESIDUAL_NOT_STABLE**

This matched diagnostic uses fully root-heldout Probe estimates and strict nested Direct/residual cross-fitting. It does not use a World Model or untouched TEST.

| Method | SR | Under-force | Mean force (N) | Excess force (N) | Realized utility |
|---|---:|---:|---:|---:|---:|
| ProbeScalar Direct | 0.9375 | 0.0486 | 4.2016 | 0.3809 | 0.0879 |
| + Direct-Only residual | 0.9375 | 0.0486 | 4.2219 | 0.4015 | 0.0829 |

Utility improves in all three seeds: [True, True, False]. Results are OOF development evidence, not untouched TEST evidence.
