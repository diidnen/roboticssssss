# Engineering fix log

No scientific method change was made by the closure assembler. The only engineering action is aggregation/validation of existing artifacts into a timestamped bundle. Existing historical fixes remain in the source experiment directories.

| symptom | root cause | change | scientific-method impact | validation |
|---|---|---|---|---|
| required closure files absent | prior experiments wrote scoped artifacts only | recompute/assemble auditable tables and manifests | none | source hashes and row-count checks recorded |
| archive/grid mismatch | 720 collection sampled continuous force strata; frozen Direct uses 0.25N grid | record data gap; do not nearest-force substitute | preserves frozen method | audit reports exact observed supports |
