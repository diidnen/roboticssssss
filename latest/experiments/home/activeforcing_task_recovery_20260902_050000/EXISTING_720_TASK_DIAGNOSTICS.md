# Existing 720-task diagnostics

The raw archive has 720 rows: 180 per task, 18 contexts per task, and 6 root families per task. The archive is valid/state-parity clean, but each context uses continuous force draws rather than the exact frozen 0.25 N grid. The diagnostics CSV includes task and context grain.

`full_success_rate` is the observed archive label rate, not the later GT-Direct OOF rate. `contexts_with_both_success_and_failure` is a force-sensitivity diagnostic under two repeats per context, not a causal threshold estimate. `post_lift_failure_candidates` means full failure with telemetry reaching a post-lift phase; it does not prove grip loss, transport failure, or placement failure. The closure taxonomy reports 123 `UNDER_FORCE` rows overall, but task attribution was not recovered and is intentionally not fabricated.
