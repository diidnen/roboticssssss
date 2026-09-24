# Verifier baseline reproduction

PASS: the already-inspected fixed-scene DEV baseline was independently recomputed from frozen rows and matches the saved hard-search artifact exactly. No TEST was read.

| Policy | Coverage | SR | Under-force | Mean force |
|---|---:|---:|---:|---:|
| Direct | 1.000 | 0.8750 | 0.0417 | 3.9325 |
| Strict hard verifier | 0.7917 | 0.9737 conditional | 0.0000 | 3.8094 |
| Verifier + Max fallback | 1.000 | 0.9375 | 0.0000 | 4.0690 |

The Max row is a deployment fallback: Fmax was executed after NO_VALID_FORCE; it was not judged safe by the verifier.

This reproduction preserves the old 24-context metric grain: repeats and three seed scores are averaged within each context-force cell. The new TRAIN-CV architecture/ensemble diagnostics instead retain each repeat and use each seed's Boolean `logit>0` decision. The two tables therefore have different estimands by design.
