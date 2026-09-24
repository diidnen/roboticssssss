# E5 Expected-Utility recovery audit

Generated: `2026-09-02T14:27:00Z` (CPU-only, source data read-only).

## Verdict

CPU QA: **PASS**. Authoritative coverage is **21/60 tuples** (**105/300 rollouts**); collection is **INCOMPLETE**.

The exact canonical resume cursor is **task 0, tuple offset 7**, `activeforcing_full_t0_root02_s7202_mid_mu0.544174`. It must start in a new timestamped directory with `--max-tuples 1`.

The existing task0 offsets07_08 attempt has no completion marker and contributes zero accepted rows. It remains in place but is logically quarantined; no in-place append or reuse is permitted.

## Coverage by task

| Task | Accepted | Missing | Planned |
|---:|---:|---:|---:|
| 0 | 8 | 7 | 15 |
| 1 | 5 | 10 | 15 |
| 5 | 4 | 11 | 15 |
| 6 | 4 | 11 | 15 |

## Evidence disposition

- Accepted physical shards: 17 (each independently revalidated).
- Quarantined partial/attempt directories: 15.
- Non-admissible rollout rows retained inside quarantined attempts: 12 (all excluded).
- Rejected completed directories: 0.
- Duplicate admissible tuple keys: 0.
- Quarantined and rejected rows are never counted, even when CSV rows exist.

## Frozen controls

- Qualified runner SHA-256: `13b33cbc784e2d9bc712a8211ad11b846eca1f8be773dd2888d7b1d33cd9b0e8`.
- Frozen canonical Utility hash: `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10`.
- Selector: `EXPECTED_UTILITY`; exact ties choose the lower force.
- Population: four independent single-task plans (tasks 0, 1, 5, 6), 15 tuples per task.

## Crash-safe handoff

The resume manifest is a CPU-prepared handoff, not launch authorization. A future launcher must recompute coverage, verify the frozen hashes, require no existing active E5 worker, pass its live GPU and pi0 prerequisites, create a brand-new output directory atomically, run one tuple, and only then promote that shard after `FULL_STATUS=PASS` plus independent QA. Any interrupted directory stays quarantined and the same cursor is retried in another fresh directory.

This lane did not launch GPU, Isaac, pi0, or training and did not modify any existing E5 data.
