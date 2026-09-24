# Mass final branch QA

Status: **PASS**

The source was inspected read-only. No Mass branch was recollected. The QA explicitly compares the completed source against the 100-row accepted snapshot and treats prior partial/interrupted spillover as excluded quarantine.

- Total: 18/18 contexts; 180/180 branches
- Newly QA'd: 8/8 contexts; 80/80 branches
- Promoted accepted count: 180/180 branches, 18/18 contexts

## Checks

| Check | Status | Evidence |
|---|---|---|
| protocol_completed | PASS | value=COMPLETED |
| total_context_coverage | PASS | contexts=18, expected=18, missing=[], unexpected=[] |
| total_branch_coverage | PASS | branches=180, unique_keys=180, expected=180 |
| new_branch_scope | PASS | new_branches=80, new_contexts=8, old_branches=100, old_contexts=10 |
| query_observation_coverage | PASS | observations=18 |
| query_step_coverage | PASS | steps=3708, expected=3708 |
| context_semantics | PASS | bad_contexts=[] |
| state_hash_parity | PASS | bad_rows=0 |
| evaluator_semantics | PASS | bad_rows=0 |
| friction_force_metadata | PASS | bad_force_rows=0, bad_metadata_rows=0 |
| telemetry_atomic_completeness | PASS | bad_files=0, phase_counts={'hold': 3600, 'lift': 10800, 'turn_0': 9035, 'turn_1': 10800, 'turn_2': 10800, 'turn_3': 10800, 'place': 3465, 'release': 9000, 'settle': 9000}, expected_files=180, actual_files=180 |
| accepted_prefix_immutability | PASS | drift_rows=0 |
| interruption_spillover_quarantine | PASS | prior_quarantine_excluded=True, source_has_unexpected_contexts=False |

## Acceptance rule

Any partial, interrupted, duplicate, post-gate spillover, state mismatch, evaluator mismatch, invalid query, or incomplete telemetry would remain unaccepted. All required checks passed for this source.
