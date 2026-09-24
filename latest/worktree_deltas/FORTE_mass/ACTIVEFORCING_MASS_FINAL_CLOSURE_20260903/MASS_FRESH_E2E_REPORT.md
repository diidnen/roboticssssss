# Mass fresh E2E report — corrected six-context result

The authoritative corrected result is based on 30 rows over six fresh contexts using the repaired five-candidate scoring path. The old singleton-candidate rollouts are not included.

| Policy | n | Query reach | Query valid | Full-task SR | Selected force | Failure stages |
|---|---:|---:|---:|---:|---:|---|
| Native π0 | 6 | 0/6 | 0/6 | 0/6 | N/A | placement / transport / VLA timeout |
| Prior | 6 | N/A | N/A | 0/6 | 1.5 N observed arm | placement |
| Fixed-Max | 6 | N/A | N/A | 1/6 | 4.0 N | placement / transport |
| ActiveForcing-Mass | 6 | 6/6 | 6/6 | 4/6 | 4.0 N in all six | 2 HIGH downstream failures |
| GT-Mass | 6 | 6/6 | 6/6 | 2/6 | 1.5 N LOW/MID; 2.5 N HIGH | MID transport and HIGH transport/placement |

## Context attribution

- `root8400_low` and `root8401_low`: Active and GT both succeed; force difference is not necessary in these two cases.
- `root8400_mid` and `root8401_mid`: Active 4.0 N succeeds; GT 1.5 N fails at transport. These are paired at the same query-state hash and are the direct evidence for a beneficial higher-force selection.
- `root8400_high`: Active 4.0 N, GT 2.5 N, Fixed-Max 4.0 N, and Prior 1.5 N all fail at placement.
- `root8401_high`: Active 4.0 N and GT 2.5 N fail at transport; Fixed-Max 4.0 N and Prior 1.5 N fail at placement.

## Interpretation

Active's 4/6 is partially supported fresh E2E performance, not a universal success claim. Its two-context advantage over GT is a force-selection effect produced by a strongly high-biased fresh identifier output. The two HIGH failures remain downstream frozen-π0/state failures within the observed force support.
