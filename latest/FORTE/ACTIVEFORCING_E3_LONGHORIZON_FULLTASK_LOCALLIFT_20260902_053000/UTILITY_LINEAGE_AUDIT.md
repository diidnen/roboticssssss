# Expected-Utility lineage audit

The current authoritative file `/home/exouser/FORTE/UTILITY_FINAL_CONFIG.json` matches the requested frozen config hash exactly: `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10`.

- selector: `argmax_candidate_expected_utility`
- equation: `p(success) * (1 - F/Fmax) + (1 - p(success)) * (-1)`
- task Fmax: `{0:5, 1:6, 5:5, 6:4}` N
- tie-break: lower force

Lineage discrepancy found: the older closure artifact `activeforcing_final_closure_20260902_034923/ACTIVEFORCING_FINAL_PROTOCOL.json` still describes `minimum p>=0.5`. `ACTIVEFORCING_AUTHORITATIVE_PROTOCOL.md` explicitly supersedes that runtime rule. The old closure is retained only as `MIN_RELIABLE_RHO / DIAGNOSTIC_ONLY`; E3 will not use it as the final selector.

For new long-horizon tasks, no task-specific Fmax exists yet. Qualification uses a robust safe force solely to test nominal semantic capability. It does not invent a Utility Fmax or run an ActiveForcing comparison. A new task Fmax must be frozen on TRAIN/DEV before later Utility experiments.
