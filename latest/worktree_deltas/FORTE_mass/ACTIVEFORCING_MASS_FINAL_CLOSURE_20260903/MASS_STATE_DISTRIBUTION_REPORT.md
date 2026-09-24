# Mass state distribution report

This report uses the corrected six-context fresh E2E query records. All six query arms reached and validated the query state; the corrected query-only pass persisted one raw query record per context.

## Evidence

- Query reach/valid: 6/6.
- Root/query hashes are distinct with restore parity preserved; the query is an actual state transition, not a root-state reuse.
- Fresh predicted masses are 0.4128--0.5494 kg while GT masses are 0.05/0.10/0.20 kg. Errors are +0.3277--+0.3878 kg in all six cases.
- Fresh probe displacement is about 0.40--0.52 mm in the persisted record, while the formal P4-B query statistics use a different telemetry schema and cannot be treated as exact hash-equivalent features.
- The fresh identifier output is therefore clear evidence of query-feature/state distribution mismatch or calibration shift, but not proof that a particular individual feature caused the shift.

## Interpretation

Query reach itself is not the failure: it is 6/6. The identifier has a strong high bias on these fresh contexts, and that bias changes the downstream force decision. It explains why Active selects 4.0 N, but it is not the primary explanation for the two HIGH downstream failures because GT-Mass shares the same query-state lineage and also fails there.

State/query shift is a secondary cause with medium confidence. The primary corrected Active-vs-GT difference is force selection; the remaining Active failures are downstream lift/transport/placement failures under the frozen π0 path.
