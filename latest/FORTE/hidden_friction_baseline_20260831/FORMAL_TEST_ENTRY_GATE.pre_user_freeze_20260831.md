# Formal TEST Entry Gate

Evaluation time: 2026-08-31T12:45:17Z  
Overall: **TEST NOT OPENED**

| Gate | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | `ACTIVEFORCING_DECISION_SEMANTICS_RESOLVED` | **BLOCKED** | AFI η=0.9 is the only executed closed selector; later Direct records conflict between 0.5 and 0.8 and lack P4-B→Direct E2E provenance |
| 2 | `ORIGINAL_ROOT_RESTORE_EQUIVALENCE_PASS` | **BLOCKED / NOT RUN** | Existing scene serialization and parity infrastructure found; task0 probe→R0→RGB/first-action QA not executed |
| 3 | `FORCE_DECISION_TO_PI0_INTERFACE_PASS` | **BLOCKED / NOT RUN** | Continuous Newton override exists and has historical P7A/P7B evidence; current task0 closed-loop validation not executed |
| 4 | `DEV_PROBE_RESET_PI0_CLOSED_LOOP_PASS` | **BLOCKED / NOT RUN** | Depends on Gates 1–3 |

## Formal population remains sealed

- TEST roots: 5174–5179 — none accessed.
- Friction values: {0.2, 0.5, 1.0} — unchanged.
- Empirical frontier: 810 planned, 0 completed.
- Method rollouts: 720 planned, 0 completed.
- No scientific failure was retried or encoded as zero.

The appropriate current blocker is `ACTIVEFORCING_DECISION_SEMANTICS_AMBIGUOUS`, not a restore-contamination verdict and not a method performance result.
